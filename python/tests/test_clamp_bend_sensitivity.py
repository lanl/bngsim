"""clamp() is a bend with an analytic sensitivity right-hand side (issue #1227).

Since #1131 a pulse of min / max / abs over the state is differentiated
through the kink twins. clamp() had no sympy counterpart, so it was declined
("calls unsupported function(s): clamp") and ran on CVODES' difference
quotient, where the column of a parameter that moves a narrow pulse was off by
up to five-fold and did not converge.

It is now the ternary twin ``clampval`` with ExprTk's ordered semantics, ``lo``
where x < lo, else ``hi`` where x > hi, else x; its derivative in each argument
is the region where that argument is the value. Crossed bounds (lo > hi) make
it ``lo`` below x = lo and ``hi`` from there on: a step, whose jump a bend does
not carry. So only a clamp whose bounds are numbers with lo <= hi is
differentiated on the sensitivity path; any other declines, as before.

The model is ``test_bend_sensitivity_rhs.py``'s: a counter T at k_clock = 2, and
X -> Y at a triangle of area 0.5 in T centred on 50. Exact dY/dk_clock =
-0.973501.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
import sympy as sp
from bngsim._jacobian import _finish_kinks, _kink_bindings

EXACT = -0.973501
CLAMP = "kx*(1/w)*(clamp(0,(Tobs-c+w/2)/(w/2),1)-clamp(0,(Tobs-c)/(w/2),1))"
MAX_ABS = "kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))"


def _model(tmp_path, law, extra=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "m.net"
    path.write_text(
        "begin parameters\n 1 kx 1\n 2 k_clock 2\n 3 c 50\n 4 w 0.01\n 5 lo 0\n 6 hi 1\n"
        "end parameters\nbegin species\n 1 T() 0\n 2 X() 10\n 3 Y() 0\nend species\n"
        f"begin functions\n 1 f() {law}\n{extra}end functions\n"
        "begin reactions\n 1 0 1 k_clock\n 2 2 3 f #_FUNC\nend reactions\n"
        "begin groups\n 1 Tobs 1\nend groups\n"
    )
    return bngsim.Model.from_net(str(path))


def _clock_column(model, **kw):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kx", "k_clock"])
    run = sim.run(t_span=(0, 30), n_points=3, **kw)
    return sim, float(np.asarray(run.sensitivities)[-1, 2, 1])


# At w = 0.01 and rtol 1e-8 the analytic column is only as good as the narrow
# pulse lets it be at that tolerance: 1.2e-4 on arm64 macOS, 7.7e-4 on x86-64
# Linux, and 3.1e-4 for the same pulse written with max/abs. 2e-3 holds on both
# and still separates it from the difference quotient's 7.4e-3 there (#1227).
@pytest.mark.parametrize(
    ("w", "rtol", "rel"),
    [(0.01, 1e-8, 2e-3), (0.01, 1e-9, 1e-4), (0.1, 1e-10, 1e-5), (1.0, 1e-10, 1e-6)],
)
def test_a_clamp_pulse_has_the_analytic_column(tmp_path, w, rtol, rel):
    model = _model(tmp_path, CLAMP)
    model.set_param("w", w)
    sim, dy = _clock_column(model, rtol=rtol, atol=rtol / 100)
    assert sim.sens_rhs_decline_reason is None
    assert dy == pytest.approx(EXACT, rel=rel)


def test_a_clamp_pulse_agrees_with_the_same_pulse_in_max_and_abs(tmp_path):
    """The two spellings of one triangle, both on the analytic path."""
    kw = {"rtol": 1e-10, "atol": 1e-12}
    clamp, max_abs = (_model(tmp_path / s, law) for s, law in (("a", CLAMP), ("b", MAX_ABS)))
    for m in (clamp, max_abs):
        m.set_param("w", 0.1)
    assert _clock_column(clamp, **kw)[1] == pytest.approx(
        _clock_column(max_abs, **kw)[1], rel=1e-5
    )


@pytest.mark.parametrize(
    "law",
    [
        "kx*clamp(1,(Tobs-c)/10,0)",  # crossed: a step at Tobs = c + 10
        "kx*(1/w)*(clamp(lo,(Tobs-c+w/2)/(w/2),hi)-clamp(lo,(Tobs-c)/(w/2),hi))",
    ],
    ids=["crossed bounds", "parameter bounds"],
)
def test_a_clamp_not_known_to_bend_declines_as_before(tmp_path, law):
    sim = bngsim.Simulator(
        _model(tmp_path, law), method="ode", sensitivity_params=["kx", "k_clock"]
    )
    sim.run(t_span=(0, 30), n_points=3)
    assert "clamp" in (sim.sens_rhs_decline_reason or "")


def test_a_clamp_beside_a_condition_declines_as_before(tmp_path):
    """Bends are differentiated only in a model with no condition (#1126)."""
    model = _model(tmp_path, CLAMP, extra=" 2 g() if(Tobs>10,1,0)\n")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kx", "k_clock"])
    sim.run(t_span=(0, 30), n_points=3)
    assert "clamp" in (sim.sens_rhs_decline_reason or "")


@pytest.mark.parametrize(
    "law",
    [CLAMP, "kx*clamp(1,(Tobs-c)/10,0)", "kx*clamp(lo,(Tobs-c)/10,hi)"],
    ids=["ordered", "crossed", "parameter bounds"],
)
def test_a_plain_run_takes_the_analytic_jacobian_of_a_clamp(tmp_path, law):
    """The state Jacobian differentiates a clamp as it does min / max / abs; the
    run agrees with the one on the difference-quotient Jacobian."""
    runs = []
    for jac in ("auto", "fd"):
        model = _model(tmp_path / jac, law)
        run = bngsim.Simulator(model, method="ode", jacobian=jac).run(
            t_span=(0, 30), n_points=31, rtol=1e-9, atol=1e-11
        )
        runs.append(np.asarray(run.species))
        if jac == "auto":
            assert model.analytical_jacobian_status == "complete"
    assert np.abs(runs[0] - runs[1]).max() <= 1e-7 * max(1.0, np.abs(runs[1]).max())


@pytest.mark.parametrize(("lo", "hi"), [(0.0, 1.0), (1.0, 0.0)], ids=["ordered", "crossed"])
@pytest.mark.parametrize("x", [-0.5, 0.25, 0.75, 1.5])
def test_the_twin_differentiates_as_the_engine_reads_it(lo, hi, x):
    """Each partial of the twin against a central difference of ExprTk's
    clamp, x < lo ? lo : (x > hi ? hi : x), away from its edges."""

    def engine(a, b, c):
        return a if b < a else (c if b > c else b)

    clampval = _kink_bindings(sp)["clampval"]
    a, b, c = sp.symbols("a b c")
    expr = clampval(a, b, c)
    point = {a: lo, b: x, c: hi}
    h = 1e-7
    for k, sym in enumerate((a, b, c)):
        args = [lo, x, hi]
        up, down = list(args), list(args)
        up[k] += h
        down[k] -= h
        fd = (engine(*up) - engine(*down)) / (2 * h)
        got = float(_finish_kinks(sp.diff(expr, sym)).subs(point))
        assert got == pytest.approx(fd, abs=1e-6), (sym, lo, x, hi)
    # and its value, written back, as the engine's
    assert float(_finish_kinks(expr).subs(point)) == engine(lo, x, hi)
