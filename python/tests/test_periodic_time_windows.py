"""A narrow window of a condition that is a sinusoid or a polynomial in time
(issue #714, and its SSA/PSA counterpart).

``sin(10*time) > 0.99`` is true on windows a few hundredths wide. No resolver
placed them, so a step of the ODE integrator, a panel of the SSA's continuous
loop, or a look of its event probe (a grid of horizon/1000) could lie across a
window unseen: events never fired and pulses never ran, depending on the output
grid or the run's horizon. Their crossing times are solved in closed form now
and every engine stops on them.

Oracles: the analytic count of up-crossings, quadrature of the pulse, and
Poisson means of the integrated rate.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import (
    _function_slot_bodies,
    _periodic_stop_times,
    switch_condition_scope,
)

pytest.importorskip("antimony")


def _ant(text):
    return bngsim.Model.from_antimony_string(text)


def _col(r, name):
    return np.asarray(r.species)[..., list(r.species_names).index(name)]


# sin(10 t) > 0.99 rises 159 times on (0, 100]; the pulse's measure there.
UP = 159
PULSE = 4.50096


@pytest.mark.parametrize("n_points", [2, 101])
@pytest.mark.parametrize(
    "extra",
    ["", " species B = 0; J0: => B; 2;", " species B = 0; J0: => B; 2*(1 + 0.5*sin(time));"],
)
def test_ssa_fires_every_narrow_window(extra, n_points):
    """The idle path, the discrete loop and the continuous loop alike."""
    m = _ant("species n = 0; E: at sin(10*time) > 0.99: n = n + 1;" + extra)
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 100), n_points=n_points, seed=1)
    assert _col(r, "n")[-1] == UP


@pytest.mark.parametrize(("method", "kw"), [("ssa", {}), ("psa", {"poplevel": 100})])
def test_stochastic_runs_every_narrow_pulse(method, kw):
    m = _ant("species X = 0; J0: => X; piecewise(1, sin(10*time) > 0.99, 0);")
    reps = 400
    r = bngsim.Simulator(m, method=method, **kw).run_replicates(
        reps, t_span=(0, 100), n_points=2, seed=2, squeeze=True
    )
    x = _col(r, "X")[:, -1]
    assert abs(x.mean() - PULSE) <= 4.5 * np.sqrt(PULSE / reps)


def test_ssa_daily_dosing_window_over_a_long_horizon():
    """A 1.2-hour daily window at rate 10 over 30 days: E[X] = 10 * 30 * width."""
    m = _ant("species X = 0; J0: => X; piecewise(10, sin(2*pi*time/24) > 0.99, 0);")
    width = 24 * (np.pi - 2 * np.arcsin(0.99)) / (2 * np.pi)
    exact = 10 * 30 * width
    reps = 200
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 720), n_points=2, seed=3, squeeze=True
    )
    x = _col(r, "X")[:, -1]
    assert abs(x.mean() - exact) <= 4.5 * np.sqrt(exact / reps)


@pytest.mark.parametrize("n_points", [2, 11, 1001])
def test_ode_fires_every_narrow_window(n_points):
    m = _ant("species n = 0; E: at sin(10*time) > 0.99: n = n + 1;")
    r = bngsim.Simulator(m).run(t_span=(0, 100), n_points=n_points)
    assert _col(r, "n")[-1] == UP


@pytest.mark.parametrize(
    ("law", "t_end", "exact"),
    [
        ("piecewise(1, sin(10*time) > 0.99, 0)", 100, PULSE),
        ("piecewise(1, cos(time) > 0.999, 0)", 100, 1.38648),
        ("piecewise(1, (time - 3)^2 < 4e-4, 0)", 100, 0.04),
    ],
)
@pytest.mark.parametrize("n_points", [2, 101])
def test_ode_runs_every_narrow_pulse(law, t_end, exact, n_points):
    m = _ant(f"species X = 0; J0: => X; {law};")
    r = bngsim.Simulator(m).run(t_span=(0, t_end), n_points=n_points)
    assert _col(r, "X")[-1] == pytest.approx(exact, rel=1e-4)


def test_ode_daily_pulse_with_decay():
    """dX/dt = [sin(2πt/24) > 0.9] - 0.01·X, X(480) from a fine-step reference."""
    m = _ant("species X = 0; J: => X; piecewise(1, sin(2*pi*time/24) > 0.9, 0); K: X => ; 0.01*X;")
    r = bngsim.Simulator(m).run(t_span=(0, 480), n_points=2)
    assert _col(r, "X")[-1] == pytest.approx(13.37783, rel=1e-5)


def _crossings(text, cond, t0, t1):
    m = _ant(text)
    ctx = m._core.functional_jacobian_context()
    return _periodic_stop_times(
        cond, switch_condition_scope(m._core, ctx), t0, t1, _function_slot_bodies(ctx)
    )


def test_the_crossings_are_solved_exactly():
    t = _crossings("species X = 0; w = 10;", "sin(w*time())>0.99", 0.0, 1.0)
    want = sorted(
        x
        for th in (np.arcsin(0.99), np.pi - np.arcsin(0.99))
        for k in range(3)
        if 0 < (x := (th + 2 * np.pi * k) / 10) <= 1
    )
    np.testing.assert_allclose(t, want, rtol=1e-12)
    np.testing.assert_allclose(
        _crossings("species X = 0;", "(time()-3)^2<4e-4", 0.0, 10.0), [2.98, 3.02], rtol=1e-12
    )
    # Never reaching the threshold: no crossing.
    assert _crossings("species X = 0;", "sin(time())>2", 0.0, 100.0) == []
    # A threshold that reads state is not a constant of the run.
    assert _crossings("species X = 0;", "sin(time())>X", 0.0, 100.0) == []


def test_a_high_degree_polynomial_is_solved_exactly():
    """Expanded in doubles, (t - 50)^8 - 1e-3 lost its 1e-3 to the 3.9e13
    beside it: the stops came back at 49.145 and 50.864, bracketing the window,
    and the ODE stepped over it (X = 0)."""
    half = 1e-3 ** (1 / 8)
    np.testing.assert_allclose(
        _crossings("species X = 0;", "(time()-50)^8<1e-3", 0.0, 100.0),
        [50 - half, 50 + half],
        rtol=1e-13,
    )
    m = _ant("species X = 0; J: => X; piecewise(1, (time - 50)^8 < 1e-3, 0);")
    r = bngsim.Simulator(m).run(t_span=(0, 100), n_points=2)
    assert _col(r, "X")[-1] == pytest.approx(2 * half, rel=1e-6)


def test_a_stop_lands_past_its_crossing():
    """On the crossing the evaluator's rounding can read the branch that ends
    there, and the restart's first step then held a 1e4 jump within ulps of its
    start: at rtol=1e-10 CVODE made no progress (main, which had no stop, ran)."""
    text = (
        "species A = 1; species B = 0;"
        " J1: A => B; piecewise(1e4, sin(0.3*time) > 0.5, 1)*A; J2: B => A; 50*B;"
    )
    for tol in ({}, {"rtol": 1e-8, "atol": 1e-10}, {"rtol": 1e-10, "atol": 1e-12}):
        r = bngsim.Simulator(_ant(text)).run(t_span=(0, 100), n_points=11, **tol)
        assert _col(r, "A")[-1] == pytest.approx(50 / 51, rel=1e-6)
    # Each stop is past its crossing, and within 32 ulps of it.
    stops = _crossings("species X = 0;", "sin(0.3*time())>0.5", 0.0, 100.0)
    exact = sorted(
        x
        for th in (np.arcsin(0.5), np.pi - np.arcsin(0.5))
        for k in range(8)
        if 0 < (x := (th + 2 * np.pi * k) / 0.3) <= 100
    )
    assert len(stops) == len(exact)
    for stop, t in zip(stops, exact, strict=False):
        assert 0 < stop - t <= 32 * np.finfo(float).eps * max(t, 1.0)


@pytest.mark.parametrize("n_points", [5, 2001])
def test_a_sensitivity_run_fires_every_narrow_window(n_points):
    """Sensitivity runs had no stops: n = 3 of 32 at n_points=5, and S_k at 20
    was 3.40 for 7.71."""
    text = (
        "species X = 1; species n = 0; k = 3; kd = 0.2; K: X =>; kd*X;"
        " E: at sin(10*time) > 0.99: X = X + k, n = n + 1;"
    )
    r = bngsim.Simulator(_ant(text), sensitivity_params=["k"]).run(
        t_span=(0, 20), n_points=n_points
    )
    assert _col(r, "n")[-1] == 32
    h = 1e-6
    fd = []
    for k in (3 * (1 + h), 3 * (1 - h)):
        m = _ant(text)
        m.set_param("k", k)
        out = bngsim.Simulator(m).run(t_span=(0, 20), n_points=2, rtol=1e-12, atol=1e-14)
        fd.append(_col(out, "X")[-1])
    s = np.asarray(r.sensitivities)[-1, list(r.species_names).index("X"), 0]
    assert s == pytest.approx((fd[0] - fd[1]) / (6 * h), rel=1e-5)


COUNTER = """begin parameters
    1 w    10
    2 thr  0.99
    3 one  1
end parameters
begin functions
    1 f1() {law}
end functions
begin species
    1 $Src() 1
    2 X() 0
    3 Tm() 0
end species
begin reactions
    1 1 1,2 f1 #R1
    2 0 3 one #Clock
end reactions
begin groups
    1 Xt 2
    2 Tobs 3
end groups
"""


@pytest.mark.parametrize(
    ("law", "exact"),
    [("if(sin(w*Tobs)>thr,1,0)", PULSE), ("if((Tobs-3)^2<4e-4,1,0)", 0.04)],
)
def test_a_window_on_a_counter_clock(tmp_path, law, exact):
    """BNGL's clock is a counter species; its conditions were recovered, then
    skipped: X = 0 for 4.50."""
    path = tmp_path / "counter.net"
    path.write_text(COUNTER.format(law=law))
    for n_points in (2, 101):
        r = bngsim.Simulator(bngsim.Model.from_net(str(path))).run(
            t_span=(0, 100), n_points=n_points
        )
        assert _col(r, "X()")[-1] == pytest.approx(exact, rel=1e-5)


@pytest.mark.parametrize(("cond", "fires"), [("-(time-3)^2 >= 0", 1), ("sin(time) >= 1", 16)])
@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_a_tangent_condition_fires_the_same_on_any_grid(cond, fires, method):
    """A condition that touches its threshold without crossing it has a stop on
    the touch. The double root came back complex and was dropped, so the count
    depended on the output grid."""
    for n_points in (2, 11, 1001):
        r = bngsim.Simulator(_ant(f"species n = 0; E: at {cond}: n = n + 1;"), method=method).run(
            t_span=(0, 100), n_points=n_points, **({"seed": 1} if method == "ssa" else {})
        )
        assert _col(r, "n")[-1] == fires


def test_too_many_crossings_are_declined_loudly_and_cheaply(caplog):
    """sin(1e6*time) over 1e4 has 3e9 crossings: they are counted, not built
    (that took 2 GB at 3e4)."""
    import time

    m = _ant("species X = 0; J0: => X; piecewise(1, sin(1e6*time) > 0.5, 0);")
    t0 = time.perf_counter()
    with caplog.at_level("WARNING"):
        assert _crossings("species X = 0;", "sin(1e6*time())>0.5", 0.0, 1e4) == []
    assert time.perf_counter() - t0 < 2
    assert "more than the" in caplog.text
    assert m is not None


def test_a_linear_condition_is_not_parsed_again(monkeypatch):
    """A dosing time already has its crossing; sending it through sympy too made
    every run of a PEtab dosing model 3 to 10 times slower."""
    import bngsim._switch_sensitivity as ss

    def fail(*a, **k):
        raise AssertionError("parsed a linear condition")

    monkeypatch.setattr(ss, "_periodic_shape", fail)
    m = _ant("species X = 0; td = 5; J0: => X; piecewise(1, (time > td) && (time < td + 0.5), 0);")
    r = bngsim.Simulator(m).run(t_span=(0, 10), n_points=2)
    assert _col(r, "X")[-1] == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("n_points", [2, 3, 6, 11, 21])
def test_sbml_suite_00936_on_a_coarse_grid(n_points):
    """SBML semantic suite 00936: S1 := piecewise(sin(10*time), time < 2, 1),
    and an event at S1 < 0 with a delay of 2 adds 1 to S2 three times. The
    windows lie inside one step on a coarse grid: S2(5) was 0, 2 or 1 for 3."""
    m = _ant(
        "species S1; species S2 = 0; S1 := piecewise(sin(time*10), time < 2, 1);"
        " E0: at 2 after S1 < 0, fromTrigger=false: S2 = S2 + 1;"
    )
    r = bngsim.Simulator(m).run(t_span=(0, 5), n_points=n_points)
    assert _col(r, "S2")[-1] == 3
    if n_points == 11:
        np.testing.assert_array_equal(_col(r, "S2"), [0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3])


def test_a_window_inside_an_if_branch():
    """A branch's crossings count only where its guard selects it."""
    t = _crossings("species X = 0;", "if((time()>3)&&(time()<5),(time()-4)^2,1)<0.01", 0.0, 10.0)
    np.testing.assert_allclose(t, [3.9, 4.1], rtol=1e-12)
    t = _crossings("species X = 0;", "if(time()<=2,sin(10*time()),1)<0", 0.0, 5.0)
    assert len(t) == 6 and t[-1] < 2
