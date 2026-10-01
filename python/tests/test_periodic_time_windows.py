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
    _sinusoid_or_polynomial_crossings,
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
    return _sinusoid_or_polynomial_crossings(
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
    assert _crossings("species X = 0;", "sin(time())>X", 0.0, 100.0) is None
