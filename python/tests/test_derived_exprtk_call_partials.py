"""A derived rate constant over an ExprTk call sympy has under another name or
not at all keeps the analytic sensitivity right-hand side (issue #1214).

Each call reached sympy as an undefined function, whose derivative it leaves
unevaluated. ``k = log10(p)`` declined the analytic right-hand side for the
whole model, as ``floor()`` does, and the run fell back to CVODES' difference
quotient; ``logn``, ``ncdf``, ``deg2rad`` and ``rad2deg`` were refused
outright, as calls the jump detector cannot read, and so was a clock threshold
``time() > log10(p)``. Each is now written out as ExprTk defines it. The calls
that step (``floor``, ``sign`` and the rest) still have no derivative to give.

Oracle: one decay at rate k, A(0) = 1, so A(1) = e^(-k) and dA/dp =
-k'(p)·e^(-k), with k' by hand from ExprTk's definitions
(third_party/exprtk/exprtk.hpp).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
import sympy as sp
from bngsim._jacobian import exprtk_closed_form_bindings

# text of k over p, p, k(p), k'(p)
CLOSED_FORMS = {
    "log10": ("log10(p)", 100.0, math.log10, lambda p: 1 / (p * math.log(10))),
    "log2": ("log2(p)", 100.0, math.log2, lambda p: 1 / (p * math.log(2))),
    "logn in its value": (
        "logn(p, 3)",
        5.0,
        lambda p: math.log(p, 3),
        lambda p: 1 / (p * math.log(3)),
    ),
    "logn in its base": (
        "logn(3, p)",
        5.0,
        lambda p: math.log(3) / math.log(p),
        lambda p: -math.log(3) / (p * math.log(p) ** 2),
    ),
    "log1p": ("log1p(p)", 0.5, math.log1p, lambda p: 1 / (1 + p)),
    "expm1": ("expm1(p)", 0.5, math.expm1, math.exp),
    "hypot": ("hypot(p, 2)", 0.5, lambda p: math.hypot(p, 2), lambda p: p / math.hypot(p, 2)),
    "ncdf": (
        "ncdf(p)",
        0.5,
        lambda p: 0.5 * math.erfc(-p / math.sqrt(2)),
        lambda p: math.exp(-p * p / 2) / math.sqrt(2 * math.pi),
    ),
    "deg2rad": ("deg2rad(p)", 30.0, lambda p: p * math.pi / 180, lambda p: math.pi / 180),
    "rad2deg": ("rad2deg(p)", 0.01, lambda p: p * 180 / math.pi, lambda p: 180 / math.pi),
    "deg2grad": ("deg2grad(p)", 0.5, lambda p: p * 10 / 9, lambda p: 10 / 9),
    "grad2deg": ("grad2deg(p)", 0.5, lambda p: p * 9 / 10, lambda p: 9 / 10),
    "avg": ("avg(p, 2, 3)", 0.5, lambda p: (p + 5) / 3, lambda p: 1 / 3),
    "sum": ("sum(p, 2)", 0.5, lambda p: p + 2, lambda p: 1.0),
    "mul": ("mul(p, p, 2)", 0.5, lambda p: 2 * p * p, lambda p: 4 * p),
    "clamp, inside": ("clamp(0, p, 2)", 0.5, lambda p: p, lambda p: 1.0),
    "clamp, above": ("clamp(0, p, 0.2)", 0.5, lambda p: 0.2, lambda p: 0.0),
    # (x < lo) ? lo : ..., which Min(Max(x, lo), hi) would read as hi
    "clamp, lo above hi": ("clamp(3, p, 1)", 2.0, lambda p: 3.0, lambda p: 0.0),
    # the derivative keeps the call, so the generated source prints log10 itself
    "a derivative that keeps the call": (
        "0.1*log10(p)^2",
        100.0,
        lambda p: 0.1 * math.log10(p) ** 2,
        lambda p: 0.2 * math.log10(p) / (p * math.log(10)),
    ),
}


def _decay(tmp_path, k_text, p, extra=""):
    path = tmp_path / "m.net"
    path.write_text(
        f"begin parameters\n 1 p {p!r}\n{extra} 9 k {k_text}\nend parameters\n"
        "begin species\n 1 A() 1.0\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 k\nend reactions\n"
        "begin groups\n 1 Aobs 1\nend groups\n"
    )
    return bngsim.Model.from_net(str(path))


def _sens_run(model, **kw):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["p"])
    result = sim.run(rtol=1e-10, atol=1e-14, **kw)
    return sim, np.asarray(result.species)[:, 0], np.asarray(result.sensitivities)[:, 0, 0]


@pytest.mark.parametrize("case", list(CLOSED_FORMS))
def test_a_derived_rate_constant_over_the_call_is_differentiated(tmp_path, case):
    text, p, k, dk = CLOSED_FORMS[case]
    sim, a, s = _sens_run(_decay(tmp_path, text, p), t_span=(0.0, 1.0), n_points=2)
    assert sim.sens_rhs_decline_reason is None
    np.testing.assert_allclose(a[-1], math.exp(-k(p)), rtol=1e-8)
    np.testing.assert_allclose(s[-1], -dk(p) * math.exp(-k(p)), rtol=1e-7, atol=1e-14)


@pytest.mark.parametrize("call", ["floor", "ceil", "round", "trunc", "frac", "sgn", "sign"])
def test_a_call_that_steps_still_has_no_derivative(tmp_path, call):
    model = _decay(tmp_path, f"{call}(p)", 0.5)
    with pytest.raises(Exception, match="could not be differentiated"):
        _sens_run(model, t_span=(0.0, 1.0), n_points=2)


def test_a_parameter_named_like_a_call_stays_a_parameter(tmp_path):
    model = _decay(tmp_path, "avg*log10(p)", 100.0, extra=" 2 avg 1.0\n")
    sim, _, s = _sens_run(model, t_span=(0.0, 1.0), n_points=2)
    assert sim.sens_rhs_decline_reason is None
    np.testing.assert_allclose(s[-1], -math.exp(-2.0) / (100 * math.log(10)), rtol=1e-7)


def test_a_chain_of_derived_parameters_through_the_call(tmp_path):
    """k = log2(k1), k1 = 10p: dk/dp = 10/(k1 ln 2) at k1 = 4."""
    model = _decay(tmp_path, "log2(k1)", 0.4, extra=" 2 k1 10*p\n")
    sim, _, s = _sens_run(model, t_span=(0.0, 1.0), n_points=2)
    assert sim.sens_rhs_decline_reason is None
    np.testing.assert_allclose(s[-1], -(10 / (4 * math.log(2))) * math.exp(-2.0), rtol=1e-7)


@pytest.mark.parametrize(
    ("threshold", "dtdp"),
    [("log10(p)", 1 / (100 * math.log(10))), ("log2(p/25)", 1 / (100 * math.log(2)))],
)
def test_a_clock_threshold_over_the_call_crosses_where_it_says(tmp_path, threshold, dtdp):
    """A decays at 1 once time > t* = 2: A(3) = e^-(3 - t*), dA/dp = A·dt*/dp.
    The threshold was refused as one no machinery can locate."""
    path = tmp_path / "c.net"
    path.write_text(
        "begin parameters\n 1 p 100.0\nend parameters\n"
        "begin species\n 1 A() 1.0\n 2 B() 0\nend species\n"
        f"begin functions\n 1 f() if(time() > {threshold}, 1, 0)\nend functions\n"
        "begin reactions\n 1 1 2 f #_FUNC\nend reactions\n"
        "begin groups\n 1 Aobs 1\nend groups\n"
    )
    _, a, s = _sens_run(bngsim.Model.from_net(str(path)), sample_times=[0.0, 1.0, 3.0])
    np.testing.assert_allclose(a[-1], math.exp(-1.0), rtol=1e-8)
    np.testing.assert_allclose(s[-1], math.exp(-1.0) * dtdp, rtol=1e-7)


@pytest.mark.parametrize("v", [1e-300, 1e-20, 1e-16, 1e-8, 0.5, 1e10, -0.5, -1e-20])
def test_log1p_is_valued_as_cs_own(v):
    """A crossing time reads the value through sympy. Written log(1 + x), or
    left to sympy's own log1p, it read 0 below x = 1e-16."""
    log1p = exprtk_closed_form_bindings(sp)["log1p"]
    x = sp.Symbol("x")
    assert float(log1p(x).subs(x, v).evalf()) == math.log1p(v)
    assert float(log1p(sp.Rational(v)).evalf()) == math.log1p(v)
