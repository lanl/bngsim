"""``steady_state()`` on a model whose rates read the time, or that has an
event (issue #710).

Both steady-state solvers look for a root of ``f(y)`` and read the right-hand
side at ``t = 0``: the residual, the Newton step, the Jacobian and ``∂f/∂p``.
The march integrates at the true time and tests convergence at ``t = 0``, and
fires no event. So ``A -> B`` at ``k·(1 − e^(−t))``, whose rate is 0 at
``t = 0``, came back ``[1, 0]`` with ``converged=True`` after one step, where A
goes to 0; and ``A' = kp − kd·A`` with an event that sets ``kp = 1`` at
``t = 2`` came back ``A = 0`` for ``1/kd``, with ``dA/dkd = 0`` for
``−1/kd²``. Nothing was logged.

A rate that reads the time has no ``f(y) = 0`` without a time being chosen, and
an event needs a trajectory. Both are refused now, by name, and
``run(steady_state=True)`` is what the message points at: it integrates at the
true time with the model's events.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

RAMPED = """begin parameters
    1 k 1
    2 A0 1
end parameters
begin functions
    1 kt() {law}
    2 seen() 2*time()
end functions
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 {rate}
end reactions
"""

EVENT = (
    "species A; A = 1; kp = 0; kd = 1\nJ0: -> A; kp\nJ1: A -> ; kd*A\nE1: at (time >= 2): kp = 1\n"
)


def _ramped(tmp_path, law="k*(1-exp(-time()))", rate="kt"):
    path = tmp_path / "ramped.net"
    path.write_text(RAMPED.format(law=law, rate=rate))
    return bngsim.Model.from_net(str(path))


def _refused(call, what):
    with pytest.raises(bngsim.SimulationError, match=rf"{what}.*\(issue #710\)"):
        call()


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_rate_that_reads_the_time_is_refused(tmp_path, method):
    """A(t) = exp(−k·(t − 1 + e^(−t))) goes to 0. It came back [1, 0],
    converged, with a residual of 0."""
    sim = bngsim.Simulator(_ramped(tmp_path), method="ode")
    _refused(lambda: sim.steady_state(method=method), "reads the time")


def test_a_rate_that_reads_the_time_through_a_derived_rate_is_refused(tmp_path):
    """The function is read through another."""
    path = tmp_path / "nested.net"
    path.write_text(
        RAMPED.format(law="k*(1-exp(-time()))", rate="outer").replace(
            "    2 seen() 2*time()\n", "    2 seen() 2*time()\n    3 outer() 0.5*kt()\n"
        )
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    _refused(sim.steady_state, "reads the time")


def test_a_table_indexed_by_time_is_refused(tmp_path):
    law = "k*tfun([0,1,2,50],[0,0.5,1,1],time)"
    sim = bngsim.Simulator(_ramped(tmp_path, law=law), method="ode")
    _refused(sim.steady_state, "reads the time")


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_model_with_an_event_is_refused(method):
    """After the event A' = 1 − kd·A, so A goes to 1/kd. It came back 4.3e-10
    by integration and 0 by Newton, converged."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    _refused(lambda: sim.steady_state(method=method), "1 event")


def test_the_sensitivity_of_a_model_with_an_event_is_refused():
    """dA/dkd came back 0 for −1/kd²."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    _refused(lambda: sim.steady_state(sensitivity_params=["kd"]), "1 event")


def test_a_batch_is_refused_for_both(tmp_path):
    """steady_state_batch came back converged at [1, 6e-9] and at A = 0."""
    ramped = bngsim.Simulator(_ramped(tmp_path), method="ode")
    _refused(lambda: ramped.steady_state_batch([{"k": 1.0}, {"k": 2.0}]), "reads the time")
    event = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    _refused(lambda: event.steady_state_batch([{"kd": 1.0}, {"kd": 2.0}]), "1 event")


def test_the_run_the_message_points_at_reaches_the_steady_state(tmp_path):
    """Control. ``run(steady_state=True)`` integrates at the true time, with
    the event."""
    ramped = bngsim.Simulator(_ramped(tmp_path), method="ode")
    out = ramped.run(t_span=(0.0, 200.0), n_points=201, steady_state=True)
    np.testing.assert_allclose(np.asarray(out.species)[-1], [0.0, 1.0], atol=1e-6)
    event = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    out = event.run(t_span=(0.0, 200.0), n_points=201, steady_state=True)
    names = list(out.species_names)
    assert np.asarray(out.species)[-1, names.index("A")] == pytest.approx(1.0, abs=1e-6)


def test_a_function_that_reads_the_time_and_is_in_no_rate_is_not_refused(tmp_path):
    """Control. ``seen()`` reads the time and no reaction reads it: the
    right-hand side does not move with the time."""
    sim = bngsim.Simulator(_ramped(tmp_path, law="k"), method="ode")
    out = sim.steady_state()
    assert out.converged
    np.testing.assert_allclose(out.concentrations, [0.0, 1.0], atol=1e-6)


def test_a_rate_that_reads_its_own_derivative_is_not_refused():
    """Control. ``rateOf`` reads the state's derivatives, which are 0 at a
    steady state whatever the time: B settles at 1 + rateOf(A) = 1."""
    text = (
        "species A, B; A = 3; B = 0; kp = 1; kd = 0.5\n"
        "J0: -> A; kp\nJ1: A -> ; kd*A\nJ2: -> B; 1 + rateOf(A)\nJ3: B -> ; B\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    out = sim.steady_state()
    assert out.converged
    names = list(out.species_names)
    got = np.asarray(out.concentrations)
    assert got[names.index("A")] == pytest.approx(2.0, rel=1e-6)
    assert got[names.index("B")] == pytest.approx(1.0, rel=1e-6)


def test_a_model_with_neither_is_solved_as_it_was(tmp_path):
    """Control."""
    path = tmp_path / "ab.net"
    path.write_text(
        "begin parameters\n 1 kf 1\n 2 kr 0.5\nend parameters\n"
        "begin species\n 1 A() 3\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 kf\n 2 2 1 kr\nend reactions\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    out = sim.steady_state(sensitivity_params=["kf"])
    np.testing.assert_allclose(out.concentrations, [1.0, 2.0], rtol=1e-7)
    rows = sim.steady_state_batch([{"kf": 1.0}, {"kf": 2.0}])
    assert len(rows) == 2
