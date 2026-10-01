"""SSA/PSA hold a rate that is a step in time constant between breakpoints
(issue #719 follow-up).

``if((time() > 10) && (time() < 150), 2.6, 0)`` reads the clock only inside
conditions whose crossings are the run's breakpoints, so between two of them it
is a constant. Such a rate is no longer integrated panel by panel: it stays in
the selection tree, the loop stops at each breakpoint and re-reads it inside the
next interval. A function that reads the clock anywhere else stays a function
of time. A guard at output times refuses a run in which such a rate moved
between breakpoints, so a misclassification cannot freeze a rate silently.

Oracles are Poisson means of the integrated rate, and the ODE for linear
systems.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._simulator import Simulator

pytest.importorskip("antimony")


def _ant(text):
    return bngsim.Model.from_antimony_string(text)


def _col(r, name):
    return np.asarray(r.species)[..., list(r.species_names).index(name)]


def _within(got, want, reps, z=4.5):
    se = np.sqrt(max(want, 1e-12) / reps)
    assert abs(got - want) <= z * se, (got, want, se)


def _pc(model, t_end):
    return Simulator._ssa_piecewise_constant(
        model, 0.0, t_end, model.time_discontinuity_conditions()
    )


@pytest.mark.parametrize(("method", "kw"), [("ssa", {}), ("psa", {"poplevel": 100})])
def test_a_step(method, kw):
    """``piecewise(10, time > 5, 0)``: E[N(20)] = 150. At t = 5 the condition is
    still false; the rate holds from just after."""
    m = _ant("species N = 0; J: => N; piecewise(10, time > 5, 0);")
    assert _pc(m, 20.0)
    reps = 200
    r = bngsim.Simulator(m, method=method, **kw).run_replicates(
        reps, t_span=(0, 20), n_points=3, seed=9, squeeze=True
    )
    _within(_col(r, "N")[:, -1].mean(), 150.0, reps)


def test_a_window_placed_by_a_parameter_per_batch_row():
    m = _ant(
        "species N = 0; t_on = 5;"
        " J: => N; piecewise(100, (time > t_on) && (time < t_on + 0.5), 0);"
    )
    assert _pc(m, 100.0)
    rows = [{"t_on": v} for v in (12.5, 77.0) for _ in range(150)]
    res = bngsim.Simulator(m, method="ssa").run_batch(
        t_span=(0, 100), n_points=2, params=rows, seed=4
    )
    n = np.array([_col(r, "N")[-1] for r in res])
    _within(n[:150].mean(), 50.0, 150)
    _within(n[150:].mean(), 50.0, 150)


def test_a_step_rate_on_a_species_matches_the_ode():
    """``A -> 0`` at ``k(t)·A`` with k a step: linear, so the mean is the ODE's."""
    txt = "species A = 200; J: A => ; piecewise(0.5, time > 2, 0.05)*A;"
    m = _ant(txt)
    assert _pc(m, 6.0)
    ode = bngsim.Simulator(_ant(txt), method="ode").run(t_span=(0, 6), n_points=4)
    reps = 300
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 6), n_points=4, seed=2, squeeze=True
    )
    a = _col(r, "A")
    for i in range(1, 4):
        se = a[:, i].std(ddof=1) / np.sqrt(reps)
        assert abs(a[:, i].mean() - _col(ode, "A")[i]) <= 4.5 * se + 1e-9, i


def test_a_step_beside_a_rate_rule():
    """The continuous loop (a rate rule) holds the step's rate too, re-reading
    it at the breakpoint: E[N(10)] = 50 and E[M(10)] = ∫₀¹⁰ t dt = 50."""
    txt = (
        "x = 0; x' = 1; species N = 0; species M = 0;"
        " J1: => N; piecewise(10, time > 5, 0); J2: => M; x;"
    )
    m = _ant(txt)
    assert _pc(m, 10.0)
    reps = 300
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 10), n_points=2, seed=3, squeeze=True
    )
    _within(_col(r, "N")[:, -1].mean(), 50.0, reps)
    _within(_col(r, "M")[:, -1].mean(), 50.0, reps)


def test_a_clock_outside_a_condition_keeps_a_function_of_time():
    """``piecewise(5, time > 5, 0) + time``: the second term is a curve."""
    m = _ant("species N = 0; J: => N; piecewise(5, time > 5, 0) + time;")
    assert _pc(m, 10.0) == []
    reps = 200
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 10), n_points=2, seed=5, squeeze=True
    )
    _within(_col(r, "N")[:, -1].mean(), 25.0 + 50.0, reps)


NET_SCHEDULE = """begin parameters
    1 k 20
end parameters
begin functions
    1 dose() if(time() - 10*floor(time()/10) < 1, k, 0)
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 dose
end reactions
begin groups
    1 Atot 1
end groups
"""


def test_a_repeating_dosing_schedule(tmp_path):
    """One unit of 20 every 10 time units over (0, 50): E[A] = 100."""
    p = tmp_path / "sched.net"
    p.write_text(NET_SCHEDULE)
    m = bngsim.Model.from_net(str(p))
    assert _pc(m, 50.0) == ["dose"]
    reps = 200
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 50), n_points=2, seed=6, squeeze=True
    )
    _within(np.asarray(r.observables)[:, -1, 0].mean(), 100.0, reps)


NET_SINE = """begin parameters
    1 k 5
end parameters
begin functions
    1 f() k*(1 + sin(time()))
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 f
end reactions
begin groups
    1 Atot 1
end groups
"""


def test_the_guard_refuses_a_rate_that_moves_between_breakpoints(tmp_path):
    """Told that a sine is piecewise constant, the engine would freeze it; the
    guard at the output rows refuses the run instead."""
    from bngsim._bngsim_core import SsaSimulator, TimeSpec

    p = tmp_path / "sine.net"
    p.write_text(NET_SINE)
    m = bngsim.Model.from_net(str(p))
    sim = SsaSimulator(m._core)
    sim.set_breakpoints([5.0])
    sim.set_piecewise_constant_functions(["f"])
    ts = TimeSpec()
    ts.t_start, ts.t_end, ts.n_points = 0.0, 10.0, 11
    with pytest.raises(RuntimeError, match="moved between breakpoints"):
        sim.run(ts, 1, 0.0)
