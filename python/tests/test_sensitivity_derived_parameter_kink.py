"""Issue #1530: a bend in a derived parameter is judged as a bend in a rate law.

BNG2.pl writes an inline BNGL rate law such as ``0 -> X() 10*min(q,1)+s`` as a
derived parameter, ``_rateLaw1 = (10*min(q,1))+s``, and the reaction is
elementary. #1332's refusal of a sensitivity parameter on a kink read only
functional rate laws, so on this spelling ``min`` at q = 1 returned dX*/dq =
23.81 (neither side's slope, 47.62 or 0), ``clamp`` at an edge one side's, and
``if()`` on its switch at a steady state one side's, with no error. The derived
chain rule also skipped #1126's tie weighting, so a three-way tie of one slope,
which has a derivative, came back 1.5 times too large: 71.43 for 47.62.

``0 -> X`` at r and ``X -> 0`` at k1 = 0.21, so X* = r/k1 and dX*/dq = (dr/dq)/k1.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

K1 = 0.21
REFUSED = r"sits on a kink of a rate law"


def _net(params: list[str], functions: list[str] = (), reactions: list[str] = ()) -> str:
    lines = ["s 1.3", f"k1 {K1}", *params]
    text = "begin parameters\n" + "".join(f"    {i} {t}\n" for i, t in enumerate(lines, 1))
    text += "end parameters\nbegin species\n    1 X() 0\n    2 E() 1\n    3 S() 0\nend species\n"
    if functions:
        text += "begin functions\n"
        text += "".join(f"    {i} {t}\n" for i, t in enumerate(functions, 1))
        text += "end functions\n"
    rxns = list(reactions) or ["0 1 r", "1 0 k1"]
    text += "begin reactions\n" + "".join(f"    {i} {t}\n" for i, t in enumerate(rxns, 1))
    return text + "end reactions\n"


@pytest.fixture
def model(tmp_path):
    def build(params, functions=(), reactions=()) -> bngsim.Model:
        path = tmp_path / "derived_kink.net"
        path.write_text(_net(params, functions, reactions))
        return bngsim.Model.from_net(str(path))

    return build


def _steady(m, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=params)


def _run(m, params, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=params)
        return sim, sim.run(t_span=(0, 100), n_points=2, **kw)


#: (law, q): the repro's bends, each written as BNG2.pl writes an inline law.
#: ``abs`` is not here: its derivative over a derived parameter is left
#: unevaluated, and a sensitivity through it was refused before #1530.
KINKS = [
    pytest.param("10*min(q, 1) + s", 1.0, id="min"),
    pytest.param("10*max(q, 1) + s", 1.0, id="max"),
    pytest.param("10*clamp(0, q, 2) + s", 0.0, id="clamp"),
]


@pytest.mark.parametrize(("law", "q"), KINKS)
def test_a_run_on_a_derived_rate_constants_kink_is_refused(model, law, q):
    """On main, min gave dX(100)/dq = 23.81 and clamp 47.62, with no error."""
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED) as raised:
        _run(model([f"q {q!r}", f"r {law}"]), ["s", "q"])
    assert "'q' in reaction 1 (rate constant r)" in str(raised.value)


@pytest.mark.parametrize(
    ("law", "q"), [*KINKS, pytest.param("if(q > 0, 10*q, 0) + s", 0.0, id="if")]
)
def test_a_steady_state_on_a_derived_rate_constants_kink_is_refused(model, law, q):
    """On main, if() on its switch gave dX*/dq = 0, one side's slope (issue #1330)."""
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED):
        _steady(model([f"q {q!r}", f"r {law}"]), ["s", "q"])


def test_a_batch_row_on_the_kink_is_refused(model):
    m = model(["q 0.5", "r 10*min(q, 1) + s"])
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["q"])
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="run_batch row 1: .*" + REFUSED):
        sim.run_batch(t_span=(0, 100), n_points=2, params=[{"q": 0.5}, {"q": 1.0}])


def test_compute_all_sensitivities_on_the_kink_is_refused(model):
    sim = bngsim.Simulator(model(["q 1.0", "r 10*min(q, 1) + s"]), method="ode")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED):
            sim.compute_all_sensitivities(t_span=(0, 100), n_points=2, params=["s", "q"])


def test_a_function_reading_a_derived_bend_is_refused(model):
    """``r() = kd + s`` with ``kd = 10*min(q, 1)``: the law's own text has no
    bend, and the prefilter dropped it before the derived parameter was inlined."""
    m = model(["q 1.0", "kd 10*min(q, 1)"], functions=["r() kd + s"])
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED) as raised:
        _run(m, ["q"])
    assert "'q' in reaction 1 (r)" in str(raised.value)


def test_an_mm_reactions_derived_kcat_on_its_kink_is_refused(model):
    m = model(
        ["q 1.0", "Km 3", "kc 2*min(q, 1)"],
        reactions=["0 3 s", "2,3 2,1 MM kc Km", "1 0 k1"],
    )
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED) as raised:
        _run(m, ["q"])
    assert "'q' in reaction 2 (rate constant kc)" in str(raised.value)


def test_off_the_kink_the_slope_is_the_derivative(model):
    """Control: at q = 0.5, min is smooth and dX*/dq = 10/k1 = 47.62."""
    m = model(["q 0.5", "r 10*min(q, 1) + s"])
    sim, res = _run(m, ["s", "q"], rtol=1e-10, atol=1e-12)
    assert sim.has_analytic_sens_rhs
    np.testing.assert_allclose(
        np.asarray(res.sensitivities)[-1, 0, :], [1 / K1, 10 / K1], rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(_steady(m, ["s", "q"]).sensitivity)[0], [1 / K1, 10 / K1], rtol=1e-8
    )


TIE = ["q 1.0", "a 1.0", "b 1.0"]


def test_a_tie_of_one_slope_weighs_one_in_a_run_and_at_a_steady_state(model):
    """``max(q, q*a, q*b)`` at a = b = 1 has the one slope 10 in q: dX*/dq = 47.62.
    sympy's d/dq Max weighs each tied argument a half, 71.43 on main."""
    m = model([*TIE, "r 10*max(q, q*a, q*b) + s"])
    sim, res = _run(m, ["q"], rtol=1e-10, atol=1e-12)
    assert sim.has_analytic_sens_rhs
    np.testing.assert_allclose(np.asarray(res.sensitivities)[-1, 0, 0], 10 / K1, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(_steady(m, ["q"]).sensitivity)[0, 0], 10 / K1, rtol=1e-8)


def test_a_tie_in_a_derived_mm_kcat_weighs_one(model):
    """The derived kcat's chain rule: ``kc = max(q, q*a, q*b)`` at a = b = 1 is
    ``kc = q``, value and slope, so the two models' q columns are the same."""
    rxns = ["0 3 s", "2,3 2,1 MM kc Km", "1 0 k1"]
    tied = model([*TIE, "Km 3", "kc max(q, q*a, q*b)"], reactions=rxns)
    plain = model([*TIE, "Km 3", "kc q"], reactions=rxns)
    got, want = (
        np.asarray(_run(m, ["q"], rtol=1e-10, atol=1e-12)[1].sensitivities) for m in (tied, plain)
    )
    assert np.any(want != 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12)


def test_a_tie_in_a_derived_initial_amount_weighs_one(tmp_path):
    """The derived species-IC seed's chain rule: ``X0 = 10*max(q, q*a, q*b)`` at
    a = b = 1 gives dX(0)/dq = 10, not 15."""
    path = tmp_path / "ic_tie.net"
    path.write_text(
        _net([*TIE, "X0 10*max(q, q*a, q*b)"], reactions=["1 0 k1"]).replace("1 X() 0", "1 X() X0")
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["q"]
    )
    res = sim.run(sample_times=[0.0, 5.0], rtol=1e-10, atol=1e-12)
    sens = np.asarray(res.sensitivities)[:, 0, 0]
    np.testing.assert_allclose(sens, [10.0, 10.0 * np.exp(-K1 * 5.0)], rtol=1e-6)
