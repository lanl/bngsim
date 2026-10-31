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
    pytest.param("if(q > 0, 10*q, 0) + s", 0.0, r"if\(q > 0, \.\.\.\)", id="if"),
]


@pytest.mark.parametrize(("law", "q", "names"), KINKS)
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


def test_a_write_onto_the_kink_after_the_simulator_is_built_is_refused(model):
    m = model("10*abs(q) + s", 0.5)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["q"])
    m.set_param("q", 0.0)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED):
        sim.run(t_span=(0, 100), n_points=2)


def test_a_batch_row_on_the_kink_is_refused(model):
    sim = bngsim.Simulator(model("10*abs(q) + s", 0.5), method="ode", sensitivity_params=["q"])
    with pytest.raises(
        (bngsim.SensitivityUnsupportedError, bngsim.SimulationError), match=REFUSED
    ):
        sim.run_batch(t_span=(0, 100), n_points=2, params=[{"q": 0.5}, {"q": 0.0}])


def test_compute_all_sensitivities_on_the_kink_is_refused(model):
    sim = bngsim.Simulator(model("10*abs(q) + s", 0.0), method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=REFUSED):
        sim.compute_all_sensitivities(t_span=(0, 100), n_points=2, params=["s", "q"])


def test_off_the_kink_the_slope_is_the_derivative(model):
    """Control: at q = 0.5, abs is smooth and dX*/dq = 10/k1 = 47.62."""
    ss = _steady(model("10*abs(q) + s", 0.5), ["s", "q"])
    np.testing.assert_allclose(np.asarray(ss.sensitivity)[0], [1 / 0.21, 10 / 0.21], rtol=1e-8)


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
