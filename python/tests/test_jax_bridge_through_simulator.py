"""Issues #902 and #913 — differentiable_solve runs through Simulator.

The JAX bridge built its own SolverOptions and called the bare CvodeSimulator,
so both its primal and its unchunked sensitivity run skipped everything
Simulator.run arranges: the crossing stops of a rate law switched on time or a
counter clock, the roots of one switched on model state, the switch-time and
saltation jumps on them, and the initial-condition seed of a primary that
reaches a species through a derived parameter. A window inside one step came
back as if it never opened, and dB/dR0 through ``B() Rt``, ``Rt = 3*R0`` came
back 0, with no warning.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from bngsim._jax_bridge import differentiable_solve  # noqa: E402

DERIVED_IC = """\
begin parameters
    1 R0  10
    2 kd  0.5
    3 Rt  3*R0
end parameters
begin species
    1 B() Rt
end species
begin reactions
    1 1 0 kd
end reactions
"""

TIME_WINDOW = """\
begin parameters
    1 k  0.1
end parameters
begin functions
    1 dose() if(time()>=100,if(time()<=140,k,0),0)
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 dose
end reactions
"""

STATE_WINDOW = """\
begin parameters
    1 k   10.0
    2 w   0.01
    3 kd  0.5
end parameters
begin functions
    1 onset() 4*Sobs
    2 r() if((time()>=onset())&&(time()<(onset()+w)),k,0)
end functions
begin species
    1 S() 1
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""


def _model(tmp_path, text, name):
    p = tmp_path / name
    p.write_text(text)
    return bngsim.Model.from_net(p)


def _p0(model):
    return jnp.array([model.get_param(n) for n in model.primary_param_names])


def test_a_primary_reaching_an_ic_through_a_derived_parameter_is_seeded(tmp_path):
    """#913: jacfwd on the default (single-call) path, dB/dR0 = 3·e^(−kd·t)."""
    model = _model(tmp_path, DERIVED_IC, "ic.net")
    assert list(model.primary_param_names) == ["R0", "kd"]

    def solve(p):
        return differentiable_solve(model, p, (0.0, 2.0), 3, rtol=1e-10, atol=1e-12)

    jac = jax.jacfwd(solve)(_p0(model))
    t = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(jac)[:, 0, 0], 3.0 * np.exp(-0.5 * t), rtol=1e-6)


def test_a_time_window_inside_one_step_is_integrated(tmp_path):
    """#902: the accumulator gets k over [100, 140], A = 40·k = 4 and dA/dk = 40,
    where the bridge gave 0 and 0."""
    model = _model(tmp_path, TIME_WINDOW, "tw.net")

    def final_a(p):
        return differentiable_solve(model, p, (0.0, 240.0), 3, rtol=1e-8, atol=1e-10)[-1, 0]

    p0 = _p0(model)
    assert float(final_a(p0)) == pytest.approx(4.0, rel=1e-6)
    assert float(jax.grad(final_a)(p0)[0]) == pytest.approx(40.0, rel=1e-6)

    # jacfwd and the chunked path agree with grad on the same quantity.
    def solve(p, chunk):
        return differentiable_solve(
            model, p, (0.0, 240.0), 3, rtol=1e-8, atol=1e-10, chunk_size=chunk
        )

    assert float(jax.jacfwd(lambda p: solve(p, 0))(p0)[-1, 0, 0]) == pytest.approx(40.0, rel=1e-6)
    assert float(jax.jacfwd(lambda p: solve(p, 2))(p0)[-1, 0, 0]) == pytest.approx(40.0, rel=1e-6)


def test_a_state_window_inside_one_step_is_integrated(tmp_path):
    """#902 on #897's model: Y = k·(t2 − t1) with crossing times independent of k,
    so dY/dk = Y/k. The bridge gave 0 and 0."""
    model = _model(tmp_path, STATE_WINDOW, "sw.net")
    ref = bngsim.Simulator(_model(tmp_path, STATE_WINDOW, "ref.net"), method="ode").run(
        t_span=(0.0, 10.0), n_points=3, rtol=1e-10, atol=1e-12
    )
    y_ref = float(np.asarray(ref.species)[-1, 1])
    assert y_ref == pytest.approx(0.0540116, rel=1e-5)

    def final_y(p):
        return differentiable_solve(model, p, (0.0, 10.0), 3, rtol=1e-10, atol=1e-12)[-1, 1]

    p0 = _p0(model)
    assert float(final_y(p0)) == pytest.approx(y_ref, rel=1e-8)

    # Every column in closed form. The window opens at t1 = 4·e^(−kd·t1) and
    # closes at t2 = 4·e^(−kd·t2) + w, and Y = k·(t2 − t1). Differentiating each
    # crossing implicitly: dt/dkd = −4·t·e^(−kd·t)/(1 + 4·kd·e^(−kd·t)), and
    # dt2/dw = 1/(1 + 4·kd·e^(−kd·t2)).
    k, w, kd = 10.0, 0.01, 0.5

    def crossing(offset):
        t = 1.0
        for _ in range(60):
            t -= (t - 4 * math.exp(-kd * t) - offset) / (1 + 4 * kd * math.exp(-kd * t))
        return t

    t1, t2 = crossing(0.0), crossing(w)
    denom = [1 + 4 * kd * math.exp(-kd * t) for t in (t1, t2)]
    dkd = [-4 * t * math.exp(-kd * t) / d for t, d in zip((t1, t2), denom, strict=True)]
    exact = [t2 - t1, k / denom[1], k * (dkd[1] - dkd[0])]
    np.testing.assert_allclose(np.asarray(jax.grad(final_y)(p0)), exact, rtol=1e-5)


COUNTER_CLOCK = """\
begin parameters
    1 c0     0.5
    2 rc     1
    3 sigma  3
    4 k      2
end parameters
begin functions
    1 rate_X() if(t>=sigma,k,0)
end functions
begin species
    1 C() c0
    2 X() 0
end species
begin reactions
    1 0 1 rc
    2 0 2 rate_X
end reactions
begin groups
    1 t                    1
end groups
"""


def test_a_counter_clock_switch_matches_its_closed_form(tmp_path):
    """#902's counter-clock case. C = c0 + rc·t crosses sigma at t* = (sigma − c0)/rc
    = 2.5, and X(5) = k·(5 − t*), so dX/d[c0, rc, sigma, k] = [2, 5, −2, 2.5]. The
    clock's own columns need issue #725's term."""
    model = _model(tmp_path, COUNTER_CLOCK, "cc.net")
    assert list(model.primary_param_names) == ["c0", "rc", "sigma", "k"]

    def final_x(p):
        return differentiable_solve(model, p, (0.0, 5.0), 3, rtol=1e-10, atol=1e-12)[-1, 1]

    np.testing.assert_allclose(
        np.asarray(jax.grad(final_x)(_p0(model))), [2.0, 5.0, -2.0, 2.5], rtol=1e-6
    )
