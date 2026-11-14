"""Issue #1531: a function's output sensitivity after ``run()`` reads smooth calls.

``sum``, ``avg``, ``asinh``, ``acosh`` and ``atanh`` are smooth ExprTk built-ins
that BNG2.pl accepts in a ``begin functions`` block. ``run()`` refused the
output sensitivity of a function that called one: ``sum`` resolved to Python's
builtin, which raised ("could not parse expression for differentiation"),
``avg`` was an undefined sympy function whose derivative stays unevaluated, and a
partial that keeps ``asinh`` / ``acosh`` / ``atanh`` (d/dsc of ``sc*asinh(u)``)
had no C spelling. ``steady_state()`` answered every one through its dual-number
fallback (issue #1330), so the two paths disagreed.

``0 -> A`` at s, ``A -> 0`` at k, A(0) = 1: A(t) = s/k + (1 - s/k)e^(-kt). Each
function is F = sc*g(A, K), so dF/dsc = g(A) and dF/dk = sc*g'(A)*dA/dk.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest


def _has_cc() -> bool:
    from bngsim import _codegen as cg

    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

S, K_DECAY, K, SC, T = 1.5, 0.3, 2.0, 1.37, 2.0

#: name: (body, g(A), dg/dA), the issue's five and the ``+`` control.
CLOSED = {
    "o_sum": ("sc*sum(Atot,K)", lambda a: a + K, lambda a: 1.0),
    "o_plus": ("sc*(Atot+K)", lambda a: a + K, lambda a: 1.0),
    "o_avg": ("sc*avg(Atot,K)", lambda a: (a + K) / 2, lambda a: 0.5),
    "o_asinh": (
        "sc*asinh(Atot/K)",
        lambda a: math.asinh(a / K),
        lambda a: 1 / (K * math.sqrt(1 + (a / K) ** 2)),
    ),
    "o_acosh": (
        "sc*acosh(1+Atot/K)",
        lambda a: math.acosh(1 + a / K),
        lambda a: 1 / (K * math.sqrt((1 + a / K) ** 2 - 1)),
    ),
    "o_atanh": (
        "sc*atanh(Atot/(Atot+K))",
        lambda a: math.atanh(a / (a + K)),
        lambda a: K / (a + K) ** 2 / (1 - (a / (a + K)) ** 2),
    ),
}

#: Other smooth calls a .net may hold, each checked against central differences
#: of plain runs: the closed forms the derived-parameter parse already read
#: (#1214), and the <math.h> calls a kept partial prints.
OTHERS = {
    "o_erf": "sc*erf(Atot/K)",
    "o_erfc": "sc*erfc(Atot/K)",
    "o_log1p": "sc*log1p(Atot/K)",
    "o_expm1": "sc*expm1(Atot/(10*K))",
    "o_mul": "sc*mul(Atot,K)",
    "o_logn": "sc*logn(Atot,K)",
    "o_ncdf": "sc*ncdf(Atot/K)",
    "o_deg2rad": "sc*deg2rad(Atot)",
    "o_rad2deg": "sc*rad2deg(Atot)",
}


@pytest.fixture(scope="module")
def net(tmp_path_factory) -> str:
    bodies = {**{n: b for n, (b, _, _) in CLOSED.items()}, **OTHERS}
    text = (
        f"begin parameters\n 1 s {S}\n 2 k {K_DECAY}\n 3 K {K}\n 4 sc {SC}\nend parameters\n"
        "begin species\n 1 A() 1.0\nend species\n"
        "begin reactions\n 1 0 1 s\n 2 1 0 k\nend reactions\n"
        "begin groups\n 1 Atot 1\nend groups\nbegin functions\n"
        + "".join(f" {i} {n}() {b}\n" for i, (n, b) in enumerate(bodies.items(), 1))
        + "end functions\n"
    )
    path = tmp_path_factory.mktemp("smooth_calls") / "m.net"
    path.write_text(text)
    return str(path)


@pytest.fixture(scope="module")
def result(net):
    sim = bngsim.Simulator(
        bngsim.Model.from_net(net), method="ode", sensitivity_params=["k", "sc"]
    )
    return sim.run(sample_times=[0.0, 1.0, T], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", list(CLOSED))
def test_run_gives_the_closed_form(result, name):
    """On main, all but ``o_plus`` raised ValueError."""
    _, g, dg = CLOSED[name]
    e = math.exp(-K_DECAY * T)
    a = S / K_DECAY + (1 - S / K_DECAY) * e
    da_dk = -(S / K_DECAY**2) * (1 - e) - T * (1 - S / K_DECAY) * e
    got = result.output_sensitivities(f"expression:{name}")[-1, 0, :]
    np.testing.assert_allclose(got, [SC * dg(a) * da_dk, g(a)], rtol=1e-7)


@pytest.mark.parametrize("name", list(CLOSED))
def test_run_and_steady_state_agree_at_the_steady_state(net, name):
    """Long enough that A is at A* = s/k, the two paths give one answer."""
    sim = bngsim.Simulator(
        bngsim.Model.from_net(net), method="ode", sensitivity_params=["k", "sc"]
    )
    run = sim.run(sample_times=[0.0, 200.0], rtol=1e-10, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = bngsim.Simulator(bngsim.Model.from_net(net), method="ode").steady_state(
            sensitivity_params=["k", "sc"]
        )
    np.testing.assert_allclose(
        run.output_sensitivities(f"expression:{name}")[-1, 0, :],
        ss.output_sensitivities([f"expression:{name}"])[0],
        rtol=1e-6,
    )


@pytest.mark.parametrize("name", list(OTHERS))
def test_other_smooth_calls_match_central_differences(net, result, name):
    def value(k: float, sc: float) -> float:
        m = bngsim.Model.from_net(net)
        m.set_param("k", k)
        m.set_param("sc", sc)
        run = bngsim.Simulator(m, method="ode").run(
            sample_times=[0.0, 1.0, T], rtol=1e-12, atol=1e-14
        )
        return float(np.asarray(run.expressions[name])[-1])

    h = 1e-5
    fd = [
        (value(K_DECAY + h, SC) - value(K_DECAY - h, SC)) / (2 * h),
        (value(K_DECAY, SC + h) - value(K_DECAY, SC - h)) / (2 * h),
    ]
    got = result.output_sensitivities(f"expression:{name}")[-1, 0, :]
    np.testing.assert_allclose(got, fd, rtol=1e-6)
