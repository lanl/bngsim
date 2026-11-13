"""steady_state_batch says where its entries start (issue #1521).

After ``Model.set_concentration`` (or a ``run()``), ``steady_state_batch`` solved
every entry from the initial conditions: its clone's ``reset()`` dropped the
write. In ``A <-> B`` it returned the steady state of the seed total, 10, where
the caller had set 20; in a bistable model it returned the other branch. Each
entry said ``converged=True``, and nothing warned, though ``run_batch`` warns on
the same state (issue #1362).

The reset stays: it is what moves an initial-condition parameter's species to
the value an entry writes (issue #129), so a dose scan after a ``run()`` still
moves its totals. The batch now warns, as ``run_batch`` does, and its docs say
how to start from the current state. ``bngsim.jax.differentiable_solve``, which
resets its clone the same way, warns too.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path

import bngsim
import numpy as np
import pytest

WARNING = r"steady_state_batch starts every entry from the model's initial conditions"
NET = (
    "begin parameters\n 1 kf 1.0\n 2 kr 0.5\n 3 A0 10\nend parameters\n"
    "begin species\n 1 A() A0\n 2 B() 0\nend species\n"
    "begin reactions\n 1 1 2 kf\n 2 2 1 kr\nend reactions\n"
)
# A <-> B conserves T = A + B; its steady state is T * [kr, kf] / (kf + kr).
SEED = np.array([10 * 0.5, 10 * 1.0]) / 1.5
WRITTEN = np.array([20 * 0.5, 20 * 1.0]) / 1.5


@pytest.fixture
def model(tmp_path: Path) -> bngsim.Model:
    path = tmp_path / "ab.net"
    path.write_text(NET)
    return bngsim.Model.from_net(str(path))


@contextmanager
def _no_start_warning():
    """A context in which the start warning would fail the test."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=WARNING)
        yield


def test_a_written_state_is_said_to_be_dropped(model):
    """The issue's A <-> B: the entry answers for the seed total, and says so."""
    model.set_concentration("A()", 20.0)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.warns(UserWarning, match=WARNING) as caught:
        (entry,) = sim.steady_state_batch(params=[{}])
    np.testing.assert_allclose(entry.concentrations, SEED, rtol=1e-6)
    message = str(caught[0].message)
    assert "states=[model.get_state()] * len(params)" in message
    assert "Model.save_concentrations()" in message


def test_the_end_of_a_run_is_said_to_be_dropped(model):
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 1.0), n_points=2)
    with pytest.warns(UserWarning, match=WARNING):
        sim.steady_state_batch(params=[{}])


def test_the_bistable_case_warns():
    """No conservation law: the seed X = 0.1 falls to the low branch, the
    written X = 3 to the high one (the issue's 0.1367 for 1.4633)."""
    pytest.importorskip("antimony")
    m = bngsim.Model.from_antimony_string(
        "J0: -> X; k0 + k1*X^2/(1 + X^2)\nJ1: X -> ; kd*X\nX = 0.1; k0 = 0.05; k1 = 1; kd = 0.5\n"
    )
    m.set_concentration("X", 3.0)
    sim = bngsim.Simulator(m, method="ode")
    high = float(np.asarray(sim.steady_state().concentrations)[0])
    with pytest.warns(UserWarning, match=WARNING):
        (entry,) = sim.steady_state_batch(params=[{}])
    assert float(np.asarray(entry.concentrations)[0]) < 0.5 < high


# ── The ways the warning names ───────────────────────────────────────────────


def test_states_starts_from_the_current_state_without_a_word(model):
    """Control: passes on main. The remedy the warning names."""
    model.set_concentration("A()", 20.0)
    sim = bngsim.Simulator(model, method="ode")
    with _no_start_warning():
        (entry,) = sim.steady_state_batch(params=[{}], states=[model.get_state()])
    np.testing.assert_allclose(entry.concentrations, WRITTEN, rtol=1e-6)


def test_save_concentrations_makes_the_written_state_the_start(model):
    """Control: passes on main. The other remedy the warning names."""
    model.set_concentration("A()", 20.0)
    model.save_concentrations()
    sim = bngsim.Simulator(model, method="ode")
    with _no_start_warning():
        (entry,) = sim.steady_state_batch(params=[{}])
    np.testing.assert_allclose(entry.concentrations, WRITTEN, rtol=1e-6)


@pytest.mark.parametrize("prepare", ["fresh", "reset-after-a-write"])
def test_a_model_at_its_initial_conditions_does_not_warn(model, prepare):
    """Control: passes on main."""
    if prepare == "reset-after-a-write":
        model.set_concentration("A()", 20.0)
        model.reset()
    sim = bngsim.Simulator(model, method="ode")
    with _no_start_warning():
        (entry,) = sim.steady_state_batch(params=[{}])
    np.testing.assert_allclose(entry.concentrations, SEED, rtol=1e-6)


# ── A dose over an initial-condition parameter ───────────────────────────────

DOSES = [{"A0": 5.0}, {"A0": 40.0}]
# Each entry's total is its dose where the reset applies it, and the run's 10,
# which A <-> B conserves, where the start is the state the run left. A total
# is compared at rel=1e-6, as the steady states above are: the solve's own
# tolerance, far inside the gap between a dose and 10.
DOSED, UNDOSED = [5.0, 40.0], [10.0, 10.0]
CAVEAT = "Such a start does not follow an initial-condition parameter the call writes"


def _after_a_run(model):
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 1.0), n_points=2)
    return sim


def _totals(entries):
    return [float(np.sum(e.concentrations)) for e in entries]


def test_a_dose_scan_after_a_run_still_moves_its_totals(model):
    """Control: passes on main. The reset is what puts each entry's A0 into
    A(): after a run, the totals are the doses, 5 and 40, not the run's 10."""
    sim = _after_a_run(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        entries = sim.steady_state_batch(params=DOSES)
    assert _totals(entries) == pytest.approx(DOSED, rel=1e-6)


def test_the_warning_on_a_dose_scan_names_the_reset_not_a_remedy_that_drops_it(model):
    """The scan above is right, and warns. Both remedies the warning names
    start from the state the run left, which drops every dose; nothing said
    so. The warning now does, on the caller's line, and Model.reset() first
    keeps the doses with no warning."""
    sim = _after_a_run(model)
    with pytest.warns(UserWarning, match=WARNING) as caught:
        sim.steady_state_batch(params=DOSES)
    (warning,) = [w for w in caught if "steady_state_batch starts" in str(w.message)]
    assert CAVEAT in str(warning.message)
    assert "Model.reset() first" in str(warning.message)
    assert warning.filename == __file__

    # What the caveat says: either remedy gives the run's total to every entry.
    with _no_start_warning():
        by_states = sim.steady_state_batch(params=DOSES, states=[model.get_state()] * 2)
    assert _totals(by_states) == pytest.approx(UNDOSED, rel=1e-6)
    model.save_concentrations()
    with _no_start_warning():
        by_save = sim.steady_state_batch(params=DOSES)
    assert _totals(by_save) == pytest.approx(UNDOSED, rel=1e-6)


def test_a_reset_first_keeps_a_dose_without_a_word(model):
    """Control: passes on main. The start the caveat names for a dose."""
    sim = _after_a_run(model)
    model.reset()
    with _no_start_warning():
        entries = sim.steady_state_batch(params=DOSES)
    assert _totals(entries) == pytest.approx(DOSED, rel=1e-6)


def test_run_batch_says_so_through_the_shared_helper(model):
    """``run_batch``'s message, moved into the shared helper, with the dose
    caveat its remedies need as much: after a run, ``parameter_scan`` and
    ``save_concentrations()`` each start a dose over A0 at the run's 10."""
    model.set_concentration("A()", 20.0)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.warns(UserWarning) as caught:
        sim.run_batch(t_span=(0.0, 1.0), n_points=2, params=[{}])
    texts = [str(w.message) for w in caught]
    assert (
        "run_batch starts every point from the model's initial conditions (Model.reset), "
        "not from its current state, which differs from them (a set_concentration, or the "
        "end of a run). To start each point from the current state, use parameter_scan, or "
        "call Model.save_concentrations() first, which makes it the state reset returns "
        "to. Such a start does not follow an initial-condition parameter the call writes, "
        "which only the reset applies: a dose over it is dropped, and its derivative is 0. "
        "Call Model.reset() first where the initial conditions are the start you want, as "
        "they are for such a dose."
    ) in texts
    assert [w.filename for w in caught if "run_batch starts" in str(w.message)] == [__file__]


# ── The JAX sibling ──────────────────────────────────────────────────────────


def _jax():
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


def test_differentiable_solve_says_so_too(model):
    """It resets its clone the same way: row 0 after a write of A() = 20 is
    the initial A() = 10."""
    _jax()
    import jax.numpy as jnp
    from bngsim.jax import differentiable_solve

    model.set_concentration("A()", 20.0)
    p = jnp.asarray([model.get_param(n) for n in model.primary_param_names])
    with pytest.warns(UserWarning, match="differentiable_solve starts every solve") as caught:
        y = differentiable_solve(model, p, (0.0, 1.0), 2)
    assert float(np.asarray(y)[0, 0]) == pytest.approx(10.0)
    assert [w.filename for w in caught if "differentiable_solve" in str(w.message)] == [__file__]


def _d_start_d_a0(model):
    """dA(0)/dA0 of a differentiable_solve at the model's own parameters."""
    jax = _jax()
    import jax.numpy as jnp
    from bngsim.jax import differentiable_solve

    names = list(model.primary_param_names)
    p = jnp.asarray([model.get_param(n) for n in names])
    jac = jax.jacfwd(lambda q: differentiable_solve(model, q, (0.0, 1.0), 2))(p)
    return float(np.asarray(jac)[0, 0, names.index("A0")])


def test_differentiable_solve_s_remedy_zeroes_an_initial_condition_derivative(model):
    """theta writes every primary parameter, A0 among them, and the reset makes
    the start follow it: dA(0)/dA0 = 1, exactly, since A() starts at A0. After
    a run the call warns; the remedy it names, save_concentrations(), makes the
    start a constant and the derivative 0, which the warning now says."""
    _after_a_run(model)
    with pytest.warns(UserWarning, match="its derivative is 0"):
        assert _d_start_d_a0(model) == 1.0
    model.save_concentrations()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="differentiable_solve starts")
        assert _d_start_d_a0(model) == 0.0


def test_differentiable_solve_after_a_reset_keeps_the_derivative_without_a_word(model):
    """Control: passes on main. The start the caveat names after a run."""
    _after_a_run(model)
    model.reset()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="differentiable_solve starts")
        assert _d_start_d_a0(model) == 1.0
