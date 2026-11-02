"""Issue #1332: a sensitivity to a parameter sitting on a kink of a rate law is refused.

``0 -> X`` at ``r()`` and ``X -> 0`` at k1 = 0.21, so X* = r/k1. With
``r() = 10*abs(q) + s`` at q = 0, X* has slopes -47.62 and +47.62 in q, and no
derivative; with ``10*min(q, 1) + s`` at q = 1, 47.62 and 0. The compiled ∂f/∂p
took sign(0) = 0 and the average at a tie, so dX*/dq came back 0 and 23.81, at
the steady state and in a run, with no error. Both are refused now, naming the
parameter and the bend; a parameter off the kink, or one the bend does not
read, is unchanged.
"""

from __future__ import annotations

import re
import warnings

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 s   1.3
    2 k1  0.21
    3 q   {q!r}
end parameters
begin species
    1 X() 0
end species
begin groups
    1 Xo 1
end groups
begin functions
    1 r() {law}
end functions
begin reactions
    1 0 1 r
    2 1 0 k1
end reactions
"""
REFUSED = r"sits on a kink of a rate law"


@pytest.fixture
def model(tmp_path):
    def build(law: str, q: float) -> bngsim.Model:
        path = tmp_path / "kink.net"
        path.write_text(NET.format(q=q, law=law))
        return bngsim.Model.from_net(str(path))

    return build


def _steady(model, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=params)


def _run(model, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
        return sim.run(t_span=(0, 100), n_points=2)


KINKS = [
    pytest.param("10*abs(q) + s", 0.0, r"abs\(q\), on its switch at q = 0", id="abs"),
    pytest.param("10*min(q, 1) + s", 1.0, r"min\(1, q\), whose arguments \[1, q\] tie", id="min"),
    pytest.param("10*max(q, 1) + s", 1.0, r"max\(1, q\)", id="max"),
    pytest.param(
        "10*clamp(0, q, 2) + s", 0.0, r"clamp\(0, q, 2\), on its switch at q = 0", id="clamp"
    ),
    # A bend over a parameter in a law that also reads the state: the two sides
    # differ by 20*Xo, which is not 0 for every state.
    pytest.param("10*abs(q)*Xo + s", 0.0, r"abs\(q\), on its switch at q = 0", id="abs times Xo"),
    # Bends inside bends. Differentiated over the parser's plain symbols, |u|
    # gave re()/im() and a Derivative the printer raised on, and lambdify knew no
    # clampval: each raised, rather than being refused.
    pytest.param("10*abs(abs(q) - 1) + s", 1.0, r"abs\(abs\(q\) - 1\)", id="abs of abs"),
    pytest.param(
        "10*min(clamp(0, q, 2), 1) + s", 1.0, r"min\(1, clamp\(0, q, 2\)\)", id="min of clamp"
    ),
    # A bend over a math call, which the exact-double printer must evaluate: a
    # printer handed to lambdify as it stands writes `math.exp`, for which its
    # namespace has no `math`, and the bend would go unjudged.
    pytest.param("10*max(exp(q) - 1, 0) + s", 0.0, r"max\(0, exp\(q\) - 1\)", id="max of exp"),
    # Exactly on the kink in double precision, 3.0/3.0 - 1 == 0, where sympy's
    # 15-digit printing of 1/3.0 read -1e-15 and let it through.
    pytest.param("10*abs(q/3.0 - 1) + s", 3.0, r"on its switch at q = 3", id="q over 3.0"),
]
# A steady state on an if() switch is refused (issue #1330); a run returns the
# derivative of the branch taken (issue #382), below.
STEADY_ONLY = [
    pytest.param("if(q > 0, 10*q, 0) + s", 0.0, r"if\(q > 0, \.\.\.\)", id="if"),
]


@pytest.mark.parametrize(("law", "q", "names"), KINKS + STEADY_ONLY)
def test_a_steady_state_sensitivity_on_a_kink_is_refused(model, law, q, names):
    """On main, abs gave dX*/dq = 0 and min 23.81."""
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED) as raised:
        _steady(model(law, q), ["s", "q"])
    assert re.search(names, str(raised.value)), str(raised.value)
    assert "'q' in reaction 1 (r)" in str(raised.value)


@pytest.mark.parametrize(("law", "q", "names"), KINKS)
def test_a_run_sensitivity_on_a_kink_is_refused(model, law, q, names):
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED):
        _run(model(law, q), ["s", "q"])


def test_a_run_on_an_if_switch_returns_the_branch_taken(model):
    """Issue #382: at ``if(q > 0, ...)`` with q = 0 the compiled column is the
    derivative of the branch the engine takes, one side's, and a time course
    returns it, as AMICI does. Here that is the else branch, 0, so dX/dq = 0."""
    from bngsim._parameter_kinks import parameters_on_a_kink

    m = model("if(q > 0, 10*q, 0) + s", 0.0)
    assert parameters_on_a_kink(m._core, ["q"], conditions=False) == []
    s = np.asarray(_run(m, ["s", "q"]).sensitivities)[-1, 0]
    np.testing.assert_allclose(s, [(1 - np.exp(-21)) / 0.21, 0.0], rtol=1e-6, atol=1e-9)


def test_a_condition_over_a_bend_is_judged_and_refused(model):
    """``if(abs(q) > 1, ...)`` at q = 1. Differentiated over the parser's plain
    symbols, d|q| raised from sympy's printer instead. The compiled ∂f/∂p
    declines this law, so the steady state's dual numbers take it from each
    side and refuse it themselves (issue #1330)."""
    from bngsim._parameter_kinks import parameters_on_a_kink

    m = model("if(abs(q) > 1, 10*q, 0) + s", 1.0)
    assert parameters_on_a_kink(m._core, ["q"]) == [
        "'q' in reaction 1 (r): if(abs(q) > 1, ...), on its switch at q = 1"
    ]
    with pytest.raises(bngsim.SimulationError, match=r"issue #1330"):
        _steady(m, ["q"])


def test_a_write_onto_the_kink_after_the_simulator_is_built_is_refused(model):
    m = model("10*abs(q) + s", 0.5)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["q"])
    m.set_param("q", 0.0)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED):
        sim.run(t_span=(0, 100), n_points=2)


def test_a_batch_row_on_the_kink_is_refused(model):
    """The refusal names the row, which a scan needs to know which point to move."""
    sim = bngsim.Simulator(model("10*abs(q) + s", 0.5), method="ode", sensitivity_params=["q"])
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="run_batch row 1: .*" + REFUSED):
        sim.run_batch(t_span=(0, 100), n_points=2, params=[{"q": 0.5}, {"q": 0.0}])


@pytest.mark.parametrize("params", [["s", "q"], None], ids=["named", "default"])
def test_compute_all_sensitivities_on_the_kink_is_refused(model, params):
    """The default (params=None) set was asked nothing: its chunks run CVODES
    directly, not through run(), and the q column came back 0."""
    sim = bngsim.Simulator(model("10*abs(q) + s", 0.0), method="ode")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED):
            sim.compute_all_sensitivities(t_span=(0, 100), n_points=2, params=params)


def test_a_parameter_named_after_a_keyword_is_refused(tmp_path):
    """The parser spells `lambda` by an alias, under which the law's symbol was
    not looked for: the bend went unseen and dX*/dlambda came back 0."""
    path = tmp_path / "kink.net"
    path.write_text(NET.replace("3 q   ", "3 lambda ").format(q=0.0, law="10*abs(lambda) + s"))
    m = bngsim.Model.from_net(str(path))
    with pytest.raises(
        bngsim.SensitivityUnsupportedError, match=r"'lambda' in reaction 1 \(r\): abs\(lambda\)"
    ):
        _steady(m, ["lambda"])


def test_off_the_kink_the_slope_is_the_derivative(model):
    """Control: at q = 0.5, abs is smooth and dX*/dq = 10/k1 = 47.62."""
    ss = _steady(model("10*abs(q) + s", 0.5), ["s", "q"])
    np.testing.assert_allclose(np.asarray(ss.sensitivity)[0], [1 / 0.21, 10 / 0.21], rtol=1e-8)


@pytest.mark.parametrize(
    ("law", "q", "slope"),
    [
        # Off any kink, a clamp inside a min: lambdify had no clampval, and this
        # raised NameError where main returned 10/k1.
        pytest.param("10*min(clamp(0, q, 2), 1) + s", 0.5, 10.0, id="min of clamp, off the kink"),
        # On a bend the law does not feel, it has a derivative, and main returned
        # it: the bend's own branches differ in slope, but the law's sides do not.
        pytest.param("10*abs(q)*q + s", 0.0, 0.0, id="abs(q)*q at 0"),
        pytest.param("if(s > 5, 10*abs(q), 0) + s", 0.0, 0.0, id="abs in a branch not taken"),
        pytest.param(
            "if(q >= 0 && s > 5, 10*q, 0) + s", 0.0, 0.0, id="a condition q cannot switch"
        ),
        pytest.param("if(q > 0, 10*q, 10*sin(q)) + s", 0.0, 10.0, id="an if() whose sides meet"),
    ],
)
def test_a_law_with_a_derivative_there_is_not_refused(model, law, q, slope):
    """Control: dX*/dq = (dr/dq)/k1, the derivative the law has at q."""
    ss = _steady(model(law, q), ["q"])
    np.testing.assert_allclose(
        np.asarray(ss.sensitivity)[0], [slope / 0.21], rtol=1e-8, atol=1e-12
    )


def test_a_parameter_the_bend_does_not_move_is_not_refused(model):
    """Control: s is not under abs(q); its column is the derivative at q = 0."""
    ss = _steady(model("10*abs(q) + s", 0.0), ["s"])
    np.testing.assert_allclose(np.asarray(ss.sensitivity)[0], [1 / 0.21], rtol=1e-8)


@pytest.mark.parametrize(
    ("law", "q"),
    [
        pytest.param("10*min(q, q*(s - 0.3)) + s", 1.0, id="a tie of one slope"),
        pytest.param("abs(Xo - q) + s", 0.5, id="a bend over the state"),
    ],
)
def test_bends_this_does_not_judge_are_left_alone(model, law, q):
    """Control: a tie whose arguments share a slope has a derivative; a bend
    over the state is a crossing in time, which the envelope already judges."""
    from bngsim._parameter_kinks import parameters_on_a_kink

    assert parameters_on_a_kink(model(law, q)._core, ["q"]) == []
