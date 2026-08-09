"""The integration march can be captured by its own BDF history (issue #235).

CVODE can reach a configuration where the state, the step size, the order and the
accumulated history are mutually self-consistent at a point that is NOT a steady
state, and then reproduce it indefinitely. Measured on the fixture below, the
march held ``h = 8.091``, ``q = 2``, every failure counter frozen and the residual
constant at ``1.7400e-07`` for over 100,000 steps until the budget ran out. It was
not chattering and not struggling; it was stuck.

The trap is in the integrator's history, not in the problem. The captured state is
2e-9 (relative) from the true root, and a fresh march started from that exact
state converges in FOUR steps. So the escape is to keep the state and discard the
history — ``CVodeReInit`` — once the residual has stopped improving.

**Why the obvious diagnoses are wrong**, each discarded against measurement, and
each pinned below because each is a plausible thing to re-propose:

* Not the convergence criterion. The absolute ``||f||_2/n_species`` rule looks
  suspect on a model whose concentrations are ~958, but the same march reaches
  5e-15 once it is not captured.
* Not a residual floor. The stuck march sat at 1.2e-7; the escaped one reaches
  1e-13 on the same model at the same tolerances.
* Not chatter at the discontinuity, which is what #235 was originally filed as.

The symptom presented as an isolated-island lottery in two unrelated parameters —
#176's parking gap and ``max_time`` — because neither creates the trap. They only
decide whether a given trajectory falls into it (``max_time`` because CVODE also
derives the initial step from the ``tout`` it is handed). Tuning either just
re-rolls the dice, so the tests here pin the mechanism instead.
"""

from __future__ import annotations

import re

import bngsim
import pytest

FIXTURE = "ltype_calcium_discontinuous_jacobian.net"
TOL = 1e-8

# k_v_stim values placing the trajectory a defined gap below the v_rec step, i.e.
# #176's parking-gap sweep. Three of these (2e-12, 9e-11, 1e-10) were captured at
# the default budget before the escape existed.
PARKING_GAPS = ["49.999999999998", "49.99999999991", "49.9999999999", "49.99999999999"]
BUDGETS = [1e4, 1e5, 1e6, 2e6, 1e7]


@pytest.fixture
def fixture_net(data_dir):
    return str(data_dir / FIXTURE)


def _regapped(tmp_path, data_dir, k_v_stim):
    """The fixture with its parking gap moved, written beside the test."""
    source = (data_dir / FIXTURE).read_text()
    path = tmp_path / f"gap_{k_v_stim}.net"
    path.write_text(re.sub(r"(16 k_v_stim\s+)\S+", rf"\g<1>{k_v_stim}", source))
    return str(path)


def _solve(net, max_time=1e6, tol=None, jacobian="fd"):
    kwargs = {} if tol is None else {"tol": tol}
    return bngsim.Simulator(
        bngsim.Model.from_net(net), method="ode", jacobian=jacobian
    ).steady_state(method="integration", rtol=TOL, atol=TOL, max_time=max_time, **kwargs)


def test_the_captured_march_now_converges(fixture_net, tmp_path, data_dir):
    """The exact configuration that sat still for 124,189 steps."""
    net = _regapped(tmp_path, data_dir, "49.999999999998")
    result = _solve(net)
    assert result.converged
    # Escaping costs a bounded amount of work. The pre-fix run burned the entire
    # budget; anything of that order means the escape did not fire.
    assert result.n_steps < 10_000, (
        f"converged but in {result.n_steps} steps — the stagnation escape is not firing"
    )


def test_it_lands_on_the_true_root_not_a_passing_dip(fixture_net, tmp_path, data_dir):
    """The load-bearing test, and the one that separates a fix from a fluke.

    The march stops the moment the residual dips below ``tol``, so merely
    converging at ``tol=1e-8`` could mean it clipped a low point in transit. Ask
    for a residual far below anything a transient would supply: reaching it means
    the march is sitting on the steady state itself.
    """
    net = _regapped(tmp_path, data_dir, "49.999999999998")
    result = _solve(net, tol=1e-13)
    assert result.converged
    assert result.residual < 1e-13


@pytest.mark.parametrize("k_v_stim", PARKING_GAPS)
@pytest.mark.parametrize("max_time", BUDGETS)
def test_no_parking_gap_and_budget_combination_is_captured(tmp_path, data_dir, k_v_stim, max_time):
    """The lottery is closed: neither knob can strand the march any more.

    Before the escape, three of these gaps failed at the default budget and the
    same gap could pass at one budget and fail at the next — which is what made
    the failure look like a property of the fixture rather than of the solver.
    """
    result = _solve(_regapped(tmp_path, data_dir, k_v_stim), max_time=max_time)
    assert result.converged, (
        f"k_v_stim={k_v_stim} at max_time={max_time:g} did not converge "
        f"(residual {result.residual:.3e} after {result.n_steps} steps)"
    )


def test_the_auto_path_inherits_the_escape(fixture_net):
    """The #127 retry re-marches on difference quotients, and that is the path the
    default config actually takes on this model, so it has to escape too."""
    result = _solve(fixture_net, jacobian="auto")
    assert result.converged
    assert result.solver_jacobian_retried
    assert result.solver_jacobian_source == "finite-difference"


@pytest.mark.parametrize(
    "net_name", ["simple_decay.net", "two_species_reversible.net", "ssa_abc.net"]
)
def test_an_ordinary_model_is_untouched(data_dir, net_name):
    """The escape must be inert where nothing is stuck.

    It fires on a residual that has stopped improving for 400 consecutive steps,
    and an ordinary march converges in far fewer than that, so it should never be
    reached. Asserted rather than assumed because a stagnation rule that trips on
    healthy models would silently discard good BDF history everywhere.
    """
    result = _solve(str(data_dir / net_name), jacobian="auto")
    assert result.converged
    assert result.n_steps < 400
