"""Exact derivatives where there is no closed form, by dual numbers (issue #1330).

A difference quotient of the right-hand side steps each species by 1.5e-8 of
the larger of itself and the state's largest concentration, and reads a secant.
#1100, #1294 and #1337 solve ``steady_state(sensitivity_params=...)`` again
with two more steps and refuse where the answers differ, and no finite set of
steps catches every law: one can be written whose secants agree at exactly
those steps. With ``jacobian="fd"``, or one rate law the closed form declines
anywhere in the model, a Hill law at X* = 0 with K = h/√2 returned dG*/ds =
2.130 for 0, a kink just past X* 4.762 for 0, and dX*/dp 2.02e-4 for 3.62e-4
beside an unrelated species at 1e8, with no error.

Where the closed form is declined or not asked for, the Jacobian, ∂f/∂p and a
function's row are taken by dual numbers through the interpreter now: each law
as the interpreter evaluates it, its derivative along the branch it takes, with
no step. At a state that sits exactly where a law changes branch the answer is
taken from each side, and refused where the two differ.

Every expected value is a closed form, or the closed-form Jacobian itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
SQRT_EPS = 1.4901161193847656e-8
# The floored step of a species at or near 0 beside G at 1e8.
H = SQRT_EPS * 1e8
K1 = 0.21

# 0 -> X at s, X -> 0 at k1, so X* = s/k1. G is made at sg = 1e8 and at a rate
# r that reads X, and lost at 1, so dG*/ds = (dr/dX)(X*)/k1. With `declined`,
# a species Z made at 1 and lost at 1 + 1e-12·rint(Z), a law the closed form
# declines: the model then has none, at the default settings.
_NET = """begin parameters
    1 s   {s!r}
    2 k1  0.21
    3 sg  1e8
    4 dg  1.0
    5 one 1.0
end parameters
begin species
    1 X() 0
    2 G() 1e8
    3 Z() 1
end species
begin groups
    1 Xo 1
    2 Zo 3
end groups
begin functions
    1 r() {rate}
    2 zl() 1 + 1e-12*rint(Zo)
end functions
begin reactions
    1 0 1 s
    2 1 0 k1
    3 0 2 r
    4 0 2 sg
    5 2 0 dg
{declined}end reactions
"""
_DECLINED = "    6 0 3 one\n    7 3 0 zl\n"


def _issue_model(tmp_path, s, rate, declined=False):
    path = tmp_path / "m.net"
    path.write_text(_NET.format(s=s, rate=rate, declined=_DECLINED if declined else ""))
    return bngsim.Model.from_net(str(path))


def _hill_at_zero():
    # Hill n = 2 at X* = 0 with K = h/√2: its secants over h and h/2 agree.
    return 0.0, f"Xo^2/({H / math.sqrt(2)!r}^2 + Xo^2)", 0.0


def _kink_beyond():
    # A kink 1e-5·h beyond X*: the slope at X* is 0.
    c = 1.3 / K1 + 1e-5 * H
    return 1.3, f"if(Xo > {c!r}, Xo - {c!r}, 0)", 0.0


def _small_entry():
    # X + X -> X + X + G at k = 1e-6: dG*/ds = 2·k·X*/k1, an entry small
    # within its column (12% off on a difference quotient).
    return 1.3, "1e-6*Xo*Xo", 2e-6 * (1.3 / K1) / K1


_CASES = {"hill at 0": _hill_at_zero, "kink beyond": _kink_beyond, "small entry": _small_entry}


class TestTheIssuesCases:
    @pytest.mark.parametrize("case", ["hill at 0", "kink beyond"])
    def test_jacobian_fd(self, tmp_path, case):
        """``jacobian="fd"`` asks for no closed form: 2.130 and 4.762 on main."""
        s, rate, want = _CASES[case]()
        sim = bngsim.Simulator(_issue_model(tmp_path, s, rate), method="ode", jacobian="fd")
        out = sim.steady_state(sensitivity_params=["s"])
        assert out.sens_jacobian_source == "automatic-differentiation"
        np.testing.assert_allclose(
            np.asarray(out.sensitivity)[:, 0], [1 / K1, want, 0.0], rtol=1e-12
        )
        assert np.asarray(out.sensitivity)[1, 0] == want

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_a_law_the_closed_form_declines(self, tmp_path, case):
        """One declined law takes the closed form away from every law; at the
        default settings both factors are dual numbers' now, and exact."""
        s, rate, want = _CASES[case]()
        sim = bngsim.Simulator(_issue_model(tmp_path, s, rate, declined=True), method="ode")
        out = sim.steady_state(sensitivity_params=["s"])
        assert out.sens_jacobian_source == "automatic-differentiation"
        assert out.sens_dfdp_source == "automatic-differentiation"
        np.testing.assert_allclose(np.asarray(out.sensitivity)[1, 0], want, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("q", [1e-3, 0.1, 1.0])
    def test_an_entry_far_below_its_columns_largest(self, tmp_path, q):
        """#1337's floor: an entry of a small species far below its column's
        largest passed at any relative error under 1e-10 of that largest, and
        at q = 1e-3 dX*/dp came back 2.024e-4 for 3.617e-4. Here the law the
        closed form declines is in another species, Z, so that X's own loss is
        the quadratic 2·k2·X² (#1330's comment wrote it as
        2·k2·rint(X·1e12)/1e12, a staircase whose exact derivative is not the
        quadratic's)."""
        text = (
            "begin parameters\n 1 s 1.3\n 2 k1 0.21\n 3 k2 0.73\n 4 p 1.0\n"
            f" 5 q {q!r}\n 6 c 1e8\n 7 dg 1.0\n 8 xp q*p\n 9 gp c*p\n 10 one 1.0\n"
            "end parameters\nbegin functions\n 1 zl() 1 + 1e-12*rint(Zo)\nend functions\n"
            "begin species\n 1 X() 0\n 2 G() 1e8\n 3 Z() 1\nend species\n"
            "begin groups\n 1 Xo 1\n 2 Zo 3\nend groups\n"
            "begin reactions\n 1 0 1 s\n 2 0 1 xp\n 3 1 0 k1\n 4 1,1 0 k2\n"
            " 5 0 2 gp\n 6 2 0 dg\n 7 0 3 one\n 8 3 0 zl\nend reactions\n"
        )
        path = tmp_path / "floor.net"
        path.write_text(text)
        out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
            sensitivity_params=["p"]
        )
        assert out.sens_jacobian_source == "automatic-differentiation"
        x = float(out.concentrations[0])
        np.testing.assert_allclose(
            np.asarray(out.sensitivity)[0, 0], q / (0.21 + 4 * 0.73 * x), rtol=1e-12
        )

    def test_with_no_dual_evaluation_the_difference_quotient_returns(self, tmp_path, monkeypatch):
        """The control: with dual numbers switched off the same solve takes the
        difference quotient, and its answer is the secant main returned."""
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        s, rate, _ = _kink_beyond()
        sim = bngsim.Simulator(_issue_model(tmp_path, s, rate), method="ode", jacobian="fd")
        out = sim.steady_state(sensitivity_params=["s"])
        assert out.sens_jacobian_source == "finite-difference"
        np.testing.assert_allclose(np.asarray(out.sensitivity)[1, 0], 1 / K1, rtol=1e-3)


# Models whose closed-form Jacobian is complete, across the rate-law kinds the
# dual evaluation mirrors: mass action, functional laws through observables and
# functions, Michaelis-Menten, derived rate constants, tables, abs/min/max,
# compartments of different size.
_CLOSED_FORM_MODELS = [
    "mm_tqssa.net",
    "mm_tqssa_stiff.net",
    "jac_kinks_max_min_abs.net",
    "tfun_param_indexed.net",
    "derived_rate_const.net",
    "nested_derived_rate_const.net",
    "func_composition.net",
    "hill_rewrite.net",
    "sat_rewrite.net",
    "saturation.net",
    "per_observable_jac.net",
    "egfr_simple_seed.net",
    "ige_receptor_cycle.net",
    "sink_compart.net",
    "BIOMD0000000002.xml",
    "BIOMD0000000038.xml",
    "MODEL1712050008.xml",
    "MODEL1812040006.xml",  # amount-valued (hasOnlySubstanceUnits) species
    "BIOMD0000000407.xml",
    "fixed_species.net",
]

# Compartments: of different size (a law across them divides each species'
# term by its own), and one whose size moves, so that a species' term is
# divided by the live size, which is a species too.
_CLOSED_FORM_TEXTS = {
    "two sizes": (
        "compartment C1 = 1, C2 = 5; species A in C1 = 2; species B in C2 = 0.5;\n"
        "k = 0.3; kb = 0.1; J1: A -> B; k*A*C1; J2: B -> A; kb*B*C2;\n"
    ),
    "a growing size": (
        "compartment C = 1; C' = 0.05*C; species A in C = 2; species B in C = 0.1;\n"
        "k = 0.3; J1: A -> B; k*A*B*C;\n"
    ),
    "a growing size across compartments": (
        "compartment C1 = 1, C2 = 5; C1' = 0.05*C1; species A in C1 = 2; species B in C2 = 0.5;\n"
        "k = 0.3; kb = 0.1; J1: A -> B; k*A*C1; J2: B -> A; kb*B*C2;\n"
    ),
}


def _load(name):
    path = str(_DATA / name)
    return bngsim.Model.from_net(path) if name.endswith(".net") else bngsim.Model.from_sbml(path)


@pytest.mark.parametrize(
    "name", _CLOSED_FORM_MODELS + [f"antimony:{key}" for key in _CLOSED_FORM_TEXTS]
)
def test_the_matrix_is_the_closed_forms(name):
    """Where the closed form exists the dual matrix is the same matrix, at the
    model's state and at one moved off it: the two are derived independently
    (sympy, and the interpreter on dual numbers), and agree to rounding."""
    if name.startswith("antimony:"):
        model = bngsim.Model.from_antimony_string(_CLOSED_FORM_TEXTS[name.split(":", 1)[1]])
    else:
        model = _load(name)
    assert model.prepare_analytical_jacobian(), f"{name} was chosen for a complete closed form"
    core = model._core
    y0 = np.asarray(core.get_state(), dtype=float)
    rng = np.random.default_rng(1330)
    scale = max(float(np.max(np.abs(y0))), 1e-12)
    y1 = np.abs(y0) * (1 + 0.1 * rng.random(y0.size)) + 1e-3 * scale * rng.random(y0.size)
    for y in (y0, y1):
        closed = np.asarray(core.fill_dense_analytical_jacobian(0.0, y))
        got = core.fill_dense_dual_jacobian(0.0, y, 0)
        assert got is not None, core.dual_unsupported_reason()
        dual, _, _ = got
        # To the rounding of each row's terms: an entry that is a cancellation
        # in one derivation may be written without it in the other (sympy
        # cancels; the interpreter evaluates the law as written).
        rows = np.max(np.abs(closed), axis=1, keepdims=True)
        apart = np.abs(dual - closed)
        held = 1e-10 * np.maximum(np.abs(closed), rows)
        i, j = np.unravel_index(int(np.argmax(apart - held)), apart.shape)
        assert np.all(apart <= held), (int(i), int(j), dual[i, j], closed[i, j])


# X is never made and sits at 0; G is made at g·max(X, 0) + s. At X = 0 the
# kink of max is met, but X stays at 0 whatever the parameters, so the side
# does not decide the answer: dG*/ds = 1/dg.
_KINK_NOT_READ = """begin parameters
    1 s  2.0
    2 g  3.0
    3 dg 0.5
    4 kx 0.4
    5 one 1.0
end parameters
begin species
    1 X() 0
    2 G() 4
    3 Z() 1
end species
begin groups
    1 Xo 1
    2 Zo 3
end groups
begin functions
    1 r() g*max(Xo, 0) + s
    2 zl() 1 + 1e-12*rint(Zo)
end functions
begin reactions
    1 0 2 r
    2 2 0 dg
    3 1 0 kx
    4 0 3 one
    5 3 0 zl
end reactions
"""

# The same with a parameter q on a kink: X made at k·|q| + s (written with
# clamp, which leaves the model no compiled ∂f/∂p), at q = 0, where d|q|/dq is
# 1 from above and -1 from below.
_PARAMETER_ON_A_KINK = """begin parameters
    1 s   2.0
    2 k   3.0
    3 k1  0.5
    4 q   0.0
    5 big 1e300
end parameters
begin species
    1 X() 4
end species
begin functions
    1 r() k*(clamp(0,q,big) + clamp(0,-q,big)) + s
end functions
begin reactions
    1 0 1 r
    2 1 0 k1
end reactions
"""


class TestATie:
    def test_a_kink_the_answer_does_not_read_is_returned(self, tmp_path):
        path = tmp_path / "k.net"
        path.write_text(_KINK_NOT_READ)
        out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
            sensitivity_params=["s", "g"]
        )
        assert out.sens_jacobian_source == "automatic-differentiation"
        assert out.sens_tie_species is None
        np.testing.assert_allclose(np.asarray(out.sensitivity)[1], [1 / 0.5, 0.0], rtol=1e-12)

    def test_a_parameter_on_a_kink_is_refused(self, tmp_path):
        """dX*/dq is 3/0.5 = 6 from above and -6 from below: there is no
        derivative, and the solve says so with both."""
        path = tmp_path / "q.net"
        path.write_text(_PARAMETER_ON_A_KINK)
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
        with pytest.raises(
            bngsim.SimulationError, match=r"#1330.*dX\(\)/dq is (-?6|-6|6).*from one side"
        ) as err:
            sim.steady_state(sensitivity_params=["s", "q"])
        assert "6" in str(err.value) and "-6" in str(err.value)

    def test_off_the_kink_it_is_returned(self, tmp_path):
        path = tmp_path / "q.net"
        path.write_text(_PARAMETER_ON_A_KINK.replace("4 q   0.0", "4 q   -0.5"))
        out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
            sensitivity_params=["s", "q"]
        )
        assert out.sens_dfdp_source == "automatic-differentiation"
        np.testing.assert_allclose(np.asarray(out.sensitivity)[0], [2.0, -6.0], rtol=1e-12)

    def test_model_jacobian_on_a_kink_is_refused(self, tmp_path):
        """Model.jacobian() of the model above at X = 0 has no value in the
        column of X: max(X, 0) is 1 from above and 0 from below. Beside it, it
        is exact."""
        path = tmp_path / "k.net"
        path.write_text(_KINK_NOT_READ)
        model = bngsim.Model.from_net(str(path))
        with pytest.raises(bngsim.SimulationError, match=r"#1330.*d\(rate of G\(\)\)/dX\(\)"):
            model.jacobian([0.0, 4.0, 1.0])
        J = model.jacobian([0.5, 4.0, 1.0])
        assert J.source == "automatic-differentiation"
        np.testing.assert_array_equal(J[:2, :2], [[-0.4, 0.0], [3.0, -0.5]])


# kf = kon/KD is a derived rate constant, so ∂f/∂KD reaches the law through
# it; the clamp() law leaves the model no compiled ∂f/∂p. A <-> B with A + B = 1:
# A* = kr/(kf + kr), so dA*/dKD = kr·(kon/KD²)/(kf + kr)².
_DERIVED = """begin parameters
    1 kon 2.0
    2 KD  0.5
    3 kf  kon/KD
    4 kr  1.5
    5 big 1e300
end parameters
begin species
    1 A() 1
    2 B() 0
end species
begin groups
    1 Ao 1
end groups
begin functions
    1 fwd() kf*clamp(0, Ao, big)/Ao
end functions
begin reactions
    1 1 2 fwd
    2 2 1 kr
end reactions
"""


def test_dfdp_reaches_a_law_through_a_derived_parameter(tmp_path):
    path = tmp_path / "d.net"
    path.write_text(_DERIVED)
    out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        sensitivity_params=["KD", "kr"]
    )
    assert out.sens_dfdp_source == "automatic-differentiation"
    kf, kr = 4.0, 1.5
    np.testing.assert_allclose(
        np.asarray(out.sensitivity)[0],
        [kr * (2.0 / 0.25) / (kf + kr) ** 2, kf / (kf + kr) ** 2],
        rtol=1e-12,
    )


def test_a_state_parked_on_a_jump_has_columns_from_either_side(tmp_path):
    """The upstream l-type calcium model (k_v_stim = 50 with k_v_leak = 1):
    Voltage relaxes to exactly 50, where v_rec = if((-70 + V) < -20, ...)
    jumps. From either side the law's value differs, and the column of V is
    that side's branch; the columns solved with each agree, as the steady
    state does not cross the jump as the parameters move. They are the closed
    form's, which re-solves at p ± h confirm (-0.534842 and 0.445702 for the
    largest entries)."""
    text = (_DATA / "ltype_calcium_discontinuous_jacobian.net").read_text()
    path = tmp_path / "ltype.net"
    path.write_text(text.replace("49.99999999999", "50.0"))
    params = ["k_open_max", "V_half"]
    dual = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", jacobian="fd")
    got = dual.steady_state(sensitivity_params=params)
    assert got.sens_jacobian_source == "automatic-differentiation"
    closed = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    want = closed.steady_state(sensitivity_params=params)
    assert want.sens_jacobian_source != "automatic-differentiation"
    np.testing.assert_allclose(got.sensitivity, want.sensitivity, rtol=1e-6, atol=1e-12)
    # The state is on the jump: from a side, a law's value is not its value.
    core = bngsim.Model.from_net(str(path))._core
    y = np.asarray(got.concentrations)
    assert core.fill_dense_dual_jacobian(0.0, y, 0)[1]  # tied
    assert any(core.fill_dense_dual_jacobian(0.0, y, side)[2] for side in (1, -1))  # jumped


_TABLE = """begin parameters
    1 drug_conc {dose!r}
    2 one       1.0
end parameters
begin functions
    1 response() tfun('dose_response.tfun', drug_conc)
end functions
begin species
    1 B() 0
end species
begin reactions
    1 0 1 response
    2 1 0 one
end reactions
"""


@pytest.mark.parametrize(
    ("dose", "slope"),
    [(0.3, 50.0), (0.5, 50.0), (0.0, None), (1.0, None)],
    ids=["inside", "a knot with no corner", "the first knot", "a corner"],
)
def test_a_table_over_a_parameter(tmp_path, dose, slope):
    """B* = response(dose), linear between knots: its slope inside a piece and
    at a knot where the slopes either side are the same; refused at a knot
    where they are not, the first one included (0 below it, 50 above)."""
    (tmp_path / "dose_response.tfun").write_text((_DATA / "dose_response.tfun").read_text())
    path = tmp_path / "t.net"
    path.write_text(_TABLE.format(dose=dose))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    if slope is None:
        with pytest.raises(bngsim.SimulationError, match=r"#1330.*changes branch"):
            sim.steady_state(sensitivity_params=["drug_conc"])
    else:
        out = sim.steady_state(sensitivity_params=["drug_conc"])
        np.testing.assert_allclose(np.asarray(out.sensitivity)[0, 0], slope, rtol=1e-12)


def test_a_hill_coefficient_over_a_species_at_zero(tmp_path):
    """r = X^n/(K^n + X^n) at X* = 0: G* does not move with n or K. The dual
    of X^n in n read 0·ln(0) = NaN there, and the solve was refused as singular
    (issue #1330's review)."""
    path = tmp_path / "h.net"
    path.write_text(
        _NET.format(s=0.0, rate="Xo^n/(K^n + Xo^n)", declined=_DECLINED).replace(
            "    5 one 1.0\n", "    5 one 1.0\n    6 n 2.5\n    7 K 0.3\n"
        )
    )
    out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        sensitivity_params=["n", "K"]
    )
    assert out.sens_jacobian_source == "automatic-differentiation"
    np.testing.assert_array_equal(np.asarray(out.sensitivity)[1], [0.0, 0.0])


def test_a_state_moved_across_a_jump_is_refused(tmp_path):
    """G is made at floor(X) with X* = s/k1 = 2: the steady state sits on the
    jump of floor, and s moves X across it (G* steps from 1 to 2 as s goes
    through 1). From either side floor is flat, so the two sides' columns
    agree; the steady state still has no derivative in s."""
    path = tmp_path / "f.net"
    path.write_text(
        _NET.format(s=1.0, rate="floor(Xo)", declined=_DECLINED).replace(
            "    2 k1  0.21\n", "    2 k1  0.5\n"
        )
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#1330.*value jumps where X\(\) sits"):
        sim.steady_state(sensitivity_params=["s"])


def test_a_state_moved_across_a_jump_beside_a_large_column_is_refused(tmp_path):
    """The same, with W made at 1e10·s in s's column. X moves by dX*/ds = 2,
    as much as X* is per unit of s, which against 1e-8 of dW*/ds = 1e10 read
    as not moving, and dG*/ds = 0 was returned."""
    path = tmp_path / "f.net"
    path.write_text(
        _NET.format(s=1.0, rate="floor(Xo)", declined=_DECLINED + "    8 0 4 big\n    9 4 0 one\n")
        .replace("    2 k1  0.21\n", "    2 k1  0.5\n")
        .replace("    5 one 1.0\n", "    5 one 1.0\n    6 big 1e10*s\n")
        .replace("    3 Z() 1\n", "    3 Z() 1\n    4 W() 0\n")
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#1330.*value jumps where X\(\) sits"):
        sim.steady_state(sensitivity_params=["s"])


# #1330's comment: X's loss written as a staircase, 2·k2·rint(X·1e12)/1e12·X,
# with steps of 1e-12 in X, under the solver's tolerance.
_STAIRCASE = """begin parameters
    1 s   1.3
    2 k1  0.21
    3 k2  0.73
    4 p   1.0
    5 q   {q!r}
    6 c   1e8
    7 dg  1.0
    8 xp  q*p
    9 gp  c*p
end parameters
begin functions
    1 loss() 2*k2*rint(Xo*1e12)/1e12
end functions
begin species
    1 X() 0
    2 G() 1e8
end species
begin groups
    1 Xo 1
end groups
begin reactions
    1 0 1 s
    2 0 1 xp
    3 1 0 k1
    4 1 0 loss
    5 0 2 gp
    6 2 0 dg
end reactions
"""


@pytest.mark.parametrize("steps", ["1e12", "1e13"])
def test_a_state_held_on_a_jump_is_refused(tmp_path, steps):
    """At q = 1e-3 the step the state is on has no root: f jumps across 0 at
    its edge, and the state is held there, within the tolerance. The branch's
    dX*/dp, q/(k1 + 2·k2·rint(X·1e12)/1e12) = 6.7e-4, is not the held state's,
    and was returned. With steps of 1e-13 the jump is 1e-13 of X's terms."""
    path = tmp_path / "stair.net"
    path.write_text(_STAIRCASE.format(q=1e-3).replace("1e12", steps))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#1330.*held on a jump.*of X\(\)"):
        sim.steady_state(sensitivity_params=["p"])


def test_a_state_held_on_a_jump_is_refused_with_the_closed_form(tmp_path):
    """The same hold where the closed form differentiates every law: X's loss
    k1·(1 + 1e-12·if(X > c, 1, 0)), c 3e-12 under s/k1, so that neither branch
    has a root on its side of c. The state is held at X* = c, and dX*/ds =
    1/k1 = 4.76 was returned, where the held state does not move (issue
    #1330's review)."""
    model = _if_jump(tmp_path)
    assert model.prepare_analytical_jacobian()
    with pytest.raises(bngsim.SimulationError, match=r"#1330.*held on a jump.*of X\(\)"):
        bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=["s"])


def test_a_held_species_no_parameter_moves_is_returned(tmp_path):
    """Held where it is, X does not move with g, which makes only G: the
    columns are the held state's, and are returned."""
    out = bngsim.Simulator(_if_jump(tmp_path), method="ode").steady_state(sensitivity_params=["g"])
    assert out.sens_held_species is None
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [0.0, 1.0], atol=1e-12)


def _if_jump(tmp_path):
    # X's loss k1·(1 + 1e-12·if(X > c, 1, 0)), c 3e-12 under s/k1; G made at g.
    path = tmp_path / "ifjump.net"
    path.write_text(
        f"begin parameters\n 1 s 1.3\n 2 k1 0.21\n 3 c {1.3 / K1 - 3e-12!r}\n 4 g 2.0\n"
        " 5 one 1.0\nend parameters\n"
        "begin functions\n 1 loss() k1*(1 + 1e-12*if(Xo > c, 1, 0))\nend functions\n"
        "begin species\n 1 X() 0\n 2 G() 0\nend species\nbegin groups\n 1 Xo 1\nend groups\n"
        "begin reactions\n 1 0 1 s\n 2 1 0 loss\n 3 0 2 g\n 4 2 0 one\nend reactions\n"
    )
    return bngsim.Model.from_net(str(path))


def test_a_root_on_a_step_is_returned(tmp_path):
    """At q = 0.1 the root is inside a step, and dX*/dp is the step's: the
    staircase's own derivative, not the quadratic's it approximates."""
    path = tmp_path / "stair.net"
    path.write_text(_STAIRCASE.format(q=0.1))
    out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        sensitivity_params=["p"]
    )
    assert out.sens_jacobian_source == "automatic-differentiation"
    assert out.sens_held_species is None
    on_step = round(float(out.concentrations[0]) * 1e12) / 1e12
    np.testing.assert_allclose(
        np.asarray(out.sensitivity)[0, 0], 0.1 / (0.21 + 2 * 0.73 * on_step), rtol=1e-10
    )


def test_a_linear_tables_knot_is_not_a_jump(tmp_path):
    """At the first knot of response(A), with B at 0 so that B's terms are all
    nothing, a side's value was the one a rounding past the knot reads (50·ulp)
    and read as a jump. The table is continuous there; its slope is 0 from
    below and 50 from above (issue #1330's review)."""
    (tmp_path / "dose_response.tfun").write_text((_DATA / "dose_response.tfun").read_text())
    path = tmp_path / "t.net"
    path.write_text(
        "begin parameters\n 1 one 1.0\nend parameters\n"
        "begin functions\n 1 response() tfun('dose_response.tfun', drug_conc)\nend functions\n"
        "begin species\n 1 A() 0\n 2 B() 0\nend species\n"
        "begin groups\n 1 drug_conc 1\nend groups\n"
        "begin reactions\n 1 0 2 response\n 2 2 0 one\nend reactions\n"
    )
    model = bngsim.Model.from_net(str(path))
    assert not any(model._core.fill_dense_dual_jacobian(0.0, [0.0, 0.0], s)[2] for s in (1, -1))
    with pytest.raises(bngsim.SimulationError, match=r"is 50 from one side and 0 from the other"):
        model.jacobian([0.0, 0.0])


class TestWhereThereIsNoDualEvaluation:
    def test_rateof(self):
        model = bngsim.Model.from_antimony_string(
            "compartment C0 = 1; k1 = 0.3; g = 0.7;\nspecies P in C0 = 1; species S in C0 = 0;\n"
            "J1: 2P -> ; k1*P*C0;\nS' = g*rateOf(P);\n"
        )
        assert "rateOf" in model._core.dual_unsupported_reason()
        assert model._core.fill_dense_dual_jacobian(0.0, [1.0, 0.0], 0) is None

    def test_a_ring_of_functions(self, tmp_path):
        path = tmp_path / "cyc.net"
        path.write_text(
            "begin parameters\n 1 A0 1.0\nend parameters\nbegin functions\n"
            " 1 F1() Atot-F2()\n 2 F2() 0.5*F1()\n 3 kr() F2()/Atot\nend functions\n"
            "begin species\n 1 A() A0\n 2 B() 0\nend species\n"
            "begin reactions\n 1 1 2 kr\nend reactions\nbegin groups\n 1 Atot 1\nend groups\n"
        )
        model = bngsim.Model.from_net(str(path))
        assert "read each other" in model._core.dual_unsupported_reason()


# One law per built-in the dual evaluation gives a derivative of its own, in
# X = 2 beside Y = 3: each against its closed form.
_BUILTINS = {
    "mratio(-3, 2, -Xo/Yo)": None,  # against a central difference
    "Xo^2.5 + exp(-Xo)*log(Yo)": lambda x, y: (2.5 * x**1.5 - math.exp(-x) * math.log(y)),
    "sqrt(Xo*Yo) + atan2(Xo, Yo)": lambda x, y: (0.5 * math.sqrt(y / x) + y / (x * x + y * y)),
    "tanh(Xo - Yo) + erf(Xo/3)": lambda x, y: (
        1 - math.tanh(x - y) ** 2 + 2 / math.sqrt(math.pi) * math.exp(-((x / 3) ** 2)) / 3
    ),
    "if(Xo > 1, Xo*Xo, 0) + floor(Yo*1.3)": lambda x, y: 2 * x,
    "min(Xo, Yo) + max(Xo/2, 0.1) + abs(Xo - Yo)": lambda x, y: 1 + 0.5 - 1,
    "ln(Xo) + log10(Xo) + Xo % 1.5": lambda x, y: 1 / x + 1 / (x * math.log(10)) + 1,
    "tgamma(Xo + 1)": lambda x, y: math.gamma(x + 1) * (1.5 - 0.5772156649015329),
    # exp(1000·X) overflows: the logistic is 0 there, and so is its slope.
    "1/(1 + exp(1000*Xo)) + Xo": lambda x, y: 1.0,
}


@pytest.mark.parametrize("law", ["floor(Xo)", "ceil(Xo)", "rint(Xo - 0.5)", "round(Xo + 0.5)"])
def test_a_step_at_its_jump_is_a_tie_that_jumps(tmp_path, law):
    """A step function of X at X = 2, where it jumps: the dual meets the tie,
    and from a side the law's value is not its value."""
    path = tmp_path / "s.net"
    path.write_text(
        "begin parameters\n 1 one 1.0\nend parameters\nbegin species\n"
        " 1 X() 2\n 2 G() 0\nend species\nbegin groups\n 1 Xo 1\nend groups\n"
        f"begin functions\n 1 law() {law}\nend functions\n"
        "begin reactions\n 1 0 2 law\n 2 2 0 one\nend reactions\n"
    )
    core = bngsim.Model.from_net(str(path))._core
    y = np.array([2.0, 0.0])
    _, tied, _ = core.fill_dense_dual_jacobian(0.0, y, 0)
    assert tied
    assert any(core.fill_dense_dual_jacobian(0.0, y, side)[2] for side in (1, -1))
    # Beside the jump it is a step, flat, and no tie.
    J, tied, _ = core.fill_dense_dual_jacobian(0.0, np.array([2.25, 0.0]), 0)
    assert not tied and J[1, 0] == 0.0


@pytest.mark.parametrize("law", sorted(_BUILTINS))
def test_a_builtin(tmp_path, law):
    """d(law)/dX at X = 2, Y = 3, as the entry of G's row in X's column (G is
    made at the law and lost at 1). tgamma'(3) = Γ(3)·ψ(3) = 2·(3/2 - γ)."""
    path = tmp_path / "b.net"
    path.write_text(
        "begin parameters\n 1 one 1.0\nend parameters\nbegin species\n"
        " 1 X() 2\n 2 Y() 3\n 3 G() 0\nend species\n"
        "begin groups\n 1 Xo 1\n 2 Yo 2\nend groups\n"
        f"begin functions\n 1 law() {law}\nend functions\n"
        "begin reactions\n 1 0 3 law\n 2 3 0 one\nend reactions\n"
    )
    core = bngsim.Model.from_net(str(path))._core
    got = core.fill_dense_dual_jacobian(0.0, np.array([2.0, 3.0, 0.0]), 0)
    assert got is not None, core.dual_unsupported_reason()
    entry = got[0][2, 0]
    want_of = _BUILTINS[law]
    if want_of is None:
        h = 1e-6
        f = [core.compute_derivs(0.0, np.array([2.0 + d, 3.0, 0.0]))[2] for d in (h, -h)]
        np.testing.assert_allclose(entry, (f[0] - f[1]) / (2 * h), rtol=1e-6)
    else:
        np.testing.assert_allclose(entry, want_of(2.0, 3.0), rtol=1e-12)
