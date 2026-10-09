"""dY_ss/dp is returned only where the steady state is an isolated root and the
returned state is on it (issue #995).

``steady_state(sensitivity_params=...)`` solves ``J·dY/dp = -∂f/∂p``. That is
the derivative of the steady state only where the steady state is an isolated
root. ``S + I -> 2 I``, ``I -> R`` ends wherever the epidemic left it, every
state with I = 0 being a steady state, and the solve returned dS*/dg = 26.25
where the final-size relation gives 66.05, with ``converged=True`` and one
logged warning; with the sink masked out it returned 0 and no warning, at a
conditioning ratio of exactly 1.

The solve now takes one Newton step from the state it returned and factors its
system again there, and takes an integration on from that state for
``max_time``. The columns are refused on any of what comes of that:

- a determinant that keeps less than 0.6 of itself: the Jacobian is singular
  at the steady state the solve was approaching;
- a pivot that is under 1e-13 of the terms it was computed from, or a
  componentwise condition number above 1e13: the Jacobian is singular
  whatever the state, but for rounding, or its columns are not known to 1%;
- a column that moves by more than 1%, each species over its own scale: the
  returned state is short of the steady state;
- a column that moves by more than 1% where the run ends: the returned state
  is one a run leaves;
- an eigenvalue far enough right of zero for ``max_time``: the system does
  not rest at the state;
- a column of which a run of ``max_time`` would leave more than 1%
  unestablished: the steady state is one no run of that length reaches;

and on a species ``mask=`` left out that an equation of the kept ones reads.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest

SIR = """begin parameters
    1 I0   1
    2 b    0.018
    3 g    1
    4 S0   99
end parameters
begin species
    1 S() S0
    2 I() I0
    3 R() 0
end species
begin reactions
    1 1,2 2,2 b
    2 2 3 g
end reactions
"""


def _net(tmp_path, text: str, name: str = "model.net") -> bngsim.Model:
    path = tmp_path / name
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _among_many(text: str, n: int = 520) -> str:
    """``text`` with ``n`` species beside it, each made and removed on its own
    and at its steady state: more unknowns than the eigenvalues are taken for
    (512), so that what is left to say whether the system rests at a state is
    the run that is taken on."""

    def lines(block: str) -> int:
        body = re.search(rf"begin {block}\n(.*?)end {block}", text, re.S).group(1)
        return len([line for line in body.splitlines() if line.strip()])

    p, s, r = lines("parameters"), lines("species"), lines("reactions")
    text = text.replace("end parameters", f"    {p + 1} kz 1.0\nend parameters")
    species = "".join(f"    {s + 1 + i} Z{i}() 1.0\n" for i in range(n))
    reactions = "".join(
        f"    {r + 1 + 2 * i} 0 {s + 1 + i} kz\n    {r + 2 + 2 * i} {s + 1 + i} 0 kz\n"
        for i in range(n)
    )
    text = text.replace("end species", species + "end species")
    return text.replace("end reactions", reactions + "end reactions")


def _final_size(g: float = 1.0, i0: float = 1.0, s0: float = 99.0, b: float = 0.018) -> float:
    """S at the end of the epidemic: S0 + I0 - S = (g/b)·ln(S0/S), the root
    below the threshold g/b."""
    lo, hi = 1e-9, g / b
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if s0 + i0 - mid - (g / b) * math.log(s0 / mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _final_size_derivatives() -> tuple[float, float]:
    """dS*/dg and dS*/dI0, from the final-size relation differentiated."""
    s, g, b, s0 = _final_size(), 1.0, 0.018, 99.0
    slope = -1.0 + g / (b * s)
    return math.log(s0 / s) / b / slope, -1.0 / slope


def test_the_final_size_relation():
    """Control. What the tests below are held to: S* = 26.2477, dS*/dg =
    66.0515 and dS*/dI0 = -0.8956."""
    assert _final_size() == pytest.approx(26.247705, rel=1e-7)
    dg, di0 = _final_size_derivatives()
    assert dg == pytest.approx(66.0515, rel=1e-5)
    assert di0 == pytest.approx(-0.89559, rel=1e-4)
    h = 1e-5
    assert dg == pytest.approx((_final_size(g=1 + h) - _final_size(g=1 - h)) / (2 * h), rel=1e-6)


def test_an_epidemic_that_burns_out_is_refused(tmp_path):
    """Every state with I = 0 is a steady state. By integration the solve
    stops at I = 1.3e-9, where the reduced Jacobian has a pivot of b·I: a
    number, 2.4e-11 of the largest, and the columns came back, 26.25 for
    66.05, beside a warning."""
    sim = bngsim.Simulator(_net(tmp_path, SIR), method="ode")
    with pytest.raises(bngsim.SimulationError) as caught:
        sim.steady_state(sensitivity_params=["g", "I0"])
    message = str(caught.value)
    assert "#995" in message and "not an isolated root" in message
    assert "a determinant that is 0 of the one" in message and "pivot for R()" in message
    assert "time course with forward sensitivities" in message
    assert "pure_sink_species" in message


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_same_with_the_sink_masked_out_is_refused(tmp_path, method):
    """With R masked out the solve had one unknown, I, with S following from
    S + I = T - R and R held: a 1x1 system with a pivot of order one, which
    returned dS*/dg = 0 at a conditioning ratio of exactly 1 and with no
    warning. The equations of the species the mask kept, f_S = 0 and f_I = 0
    in S and I, are the singular pair of the unmasked model, and the columns
    are solved on those."""
    model = _net(tmp_path, SIR)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SimulationError) as caught:
        sim.steady_state(
            sensitivity_params=["g", "I0"],
            mask=~np.asarray(model.is_pure_sink()),
            method=method,
        )
    message = str(caught.value)
    assert "#995" in message and "pivot for S()" in message
    assert "With mask=" in message


def test_by_newton_and_unmasked_it_is_refused_as_it_was(tmp_path):
    """Control. The Newton root has I = 0 exactly, the pivot is an exact zero,
    and the refusal is the one there has been for that."""
    sim = bngsim.Simulator(_net(tmp_path, SIR), method="ode")
    with pytest.raises(bngsim.SimulationError, match="dY_ss/dp does not exist"):
        sim.steady_state(sensitivity_params=["g", "I0"], method="newton")


def test_the_steady_state_itself_is_where_the_run_ends(tmp_path):
    """Control. ``steady_state()`` without sensitivities is by integration, and
    is right: S* is the final size."""
    out = bngsim.Simulator(_net(tmp_path, SIR), method="ode").steady_state()
    assert out.converged
    assert np.asarray(out.concentrations)[0] == pytest.approx(_final_size(), rel=1e-7)


def test_a_time_course_gives_the_columns_the_refusal_points_to(tmp_path):
    """Control. Forward sensitivities of a run to the steady state are the
    derivative of where it ends: 66.0515 and -0.8956."""
    sim = bngsim.Simulator(_net(tmp_path, SIR), method="ode", sensitivity_params=["g", "I0"])
    result = sim.run(t_span=(0.0, 1e4), n_points=2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[-1][0], _final_size_derivatives(), rtol=1e-6
    )


def test_with_a_differenced_jacobian_it_is_refused_too(tmp_path):
    """``jacobian="fd"`` factors a difference quotient, at both states."""
    sim = bngsim.Simulator(_net(tmp_path, SIR), method="ode", jacobian="fd")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*not an isolated root.*a determinant that is 0 of the one.*pivot for R\(\)",
    ):
        sim.steady_state(sensitivity_params=["g"])


# A in c1 and B in c2, where c2 grows to 3. The amount A*c1 + B*c2 is kept, which
# is no linear law of the state, so none is found, and the Jacobian is singular
# at the steady state.
NOT_A_LINEAR_LAW = (
    "compartment c1, c2; c1 = 1; c2 = 2; c2' = kg*(3 - c2); species A in c1, B in c2;\n"
    "A = 1; B = 0.2; k1 = 1; k2 = 0.5; kg = 0.4;\nR1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
)


def test_a_total_that_no_law_of_the_model_holds_is_refused():
    """A* = 1.4·k2·3/(k1 + k2·3) by the amount that is kept, so dA*/dk1 =
    -0.336; the solve returned -0.4667 at min|U|/max|U| = 7e-11, beside the
    warning. The determinant there is k1·(dc2/dt)/c2, which is what is left of
    the residual."""
    model = bngsim.Model.from_antimony_string(NOT_A_LINEAR_LAW)
    assert model.conservation_laws["n_laws"] == 0
    with pytest.raises(bngsim.SimulationError, match=r"#995.*not an isolated root"):
        bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=["k1"])


TWO_RATES = """begin parameters
    1 kf     1.0
    2 kr     1.0
    3 kff    {fast}
    4 kfr    {fast}
end parameters
begin species
    1 A() 1.0
    2 B() 0.0
    3 C() 1.0
    4 D() 0.0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
    3 3 4 kff
    4 4 3 kfr
end reactions
"""


@pytest.mark.parametrize("fast", [1e10, 1e14])
def test_an_isolated_root_with_rates_far_apart_is_returned(tmp_path, caplog, fast):
    """Two reversible pairs with rates a factor ``fast`` apart: an isolated
    root, with min|U|/max|U| = 1/fast. At 1e10 the columns came back right
    beside a warning that they may not be; at 1e14 likewise. The matrix is
    diagonal, with a condition number of 1 whatever its entries, and neither
    pivot moves. Fails on main for the warning."""
    sim = bngsim.Simulator(_net(tmp_path, TWO_RATES.format(fast=fast)), method="ode")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        out = sim.steady_state(sensitivity_params=["kf", "kff"], tol=1e-12)
    assert not [r for r in caplog.records if "conditioned" in r.message or "reliable" in r.message]
    assert out.sens_jacobian_rcond == pytest.approx(1.0 / fast, rel=1e-9)
    assert out.sens_root_condition < 10
    assert out.sens_root_determinant_ratio == 1.0 and out.sens_root_column_shift < 1e-6
    exact = np.array([[-0.25, 0.0], [0.25, 0.0], [0.0, -0.25 / fast], [0.0, 0.25 / fast]])
    assert np.asarray(out.sensitivity) == pytest.approx(exact, rel=1e-9, abs=1e-30)


# A and X in the cytoplasm (V = 1), B and C in a nucleus of size V2: whichever
# species the law is solved for, there are unknowns in compartments of both
# sizes.
BOTH_SIZES = (
    "compartment cyt, nuc; cyt = 1; nuc = {v2};\n"
    "species A in cyt, X in cyt, B in nuc, C in nuc; A = 1; X = 0; B = 0; C = 0;\n"
    "k1 = 1; k2 = 0.5; k3 = 2; k4 = 1; k5 = 1; k6 = 2;\n"
    "R1: A -> B; k1*A*cyt\nR2: B -> A; k2*B*nuc\nR3: B -> C; nuc*(k3*B - k4*C)\n"
    "R4: A -> X; cyt*(k5*A - k6*X)\n"
)


@pytest.mark.parametrize("v2", [1e5, 1e7, 1e-6])
def test_an_isolated_root_with_sizes_far_apart_is_returned(v2):
    """A* = 2/15, X* = A*/2, B* = 2·A*/V2 and C* = 2·B*: an isolated root,
    whose min|U|/max|U| falls with the size ratio. At V2 = 1e-6 the ratio is
    below 1e-8 and it was refused for a law across sizes (issue #758), though
    the columns are right: dA*/dk1 = -8/75, dX*/dk1 = -4/75,
    dB*/dk1 = 4/(75·V2). (At 1e5 and 1e7 main returned them; what is new there
    is that the condition number says the same at every size.)"""
    model = bngsim.Model.from_antimony_string(BOTH_SIZES.format(v2=v2))
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["k1"], tol=1e-13, method="newton"
    )
    assert out.converged
    order = [list(model.species_names).index(name) for name in ("A", "X", "B", "C")]
    np.testing.assert_allclose(
        np.asarray(out.concentrations)[order],
        [2 / 15, 1 / 15, 4 / (15 * v2), 8 / (15 * v2)],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out.sensitivity)[order, 0],
        [-8 / 75, -4 / 75, 4 / (75 * v2), 8 / (75 * v2)],
        rtol=1e-7,
    )
    # 6.16 at every size ratio: the root of |A⁻¹|·|A| is the same in any units.
    assert out.sens_root_condition == pytest.approx(6.16, rel=2e-3)


# A <-> B and A <-> C at concentrations of 1e-6, both forward rates k1. The law
# A + B + C is solved for A, and dA*/dk1 = -(dB*/dk1 + dC*/dk1) with the two of
# one sign: of the three entries of the column it is A's that is furthest off
# at a state short of the steady state.
STAR = """begin parameters
    1 k1 5e-3
    2 k2 2e-3
    3 k4 4e-3
    4 A0 1e-6
end parameters
begin species
    1 A() A0
    2 B() 0
    3 C() 0
end species
begin reactions
    1 1 2 k1
    2 2 1 k2
    3 1 3 k1
    4 3 1 k4
end reactions
"""


# A <-> B at concentrations of 1e-6 and rates of 5e-3: the residual is below
# tol = 1e-9 while B is still 20% short.
SMALL = """begin parameters
    1 kf     5e-3
    2 kr     5e-3
    3 A0     1e-6
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
"""


def test_a_state_short_of_the_steady_state_is_stepped_to_it(tmp_path):
    """``tol`` bounds ||f||/n. With concentrations of 1e-6 and rates of 5e-3
    the residual passes 1e-9 while A is 20% from A* = 5e-7, and dA*/dkf
    came back 20% from -5e-5 with ``converged=True`` and nothing logged. The
    columns are solved again one Newton step on, and where they move the state
    is stepped on until they do not: the state and the columns returned are
    those of the root."""
    sim = bngsim.Simulator(_net(tmp_path, SMALL), method="ode")
    out = sim.steady_state(sensitivity_params=["kf", "kr"])
    np.testing.assert_allclose(np.asarray(out.concentrations), [5e-7, 5e-7], rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[-5e-5, 5e-5], [5e-5, -5e-5]], rtol=1e-9
    )
    assert out.sens_root_newton_steps == 2 and out.residual < 1e-20
    assert out.sens_root_column_shift < 1e-9
    # Without the columns, the state is the solver's own, as it was.
    plain = np.asarray(sim.steady_state().concentrations)
    assert abs(plain[0] - 5e-7) > 0.2 * 5e-7


def test_the_same_solved_to_its_steady_state_is_as_the_solver_left_it(tmp_path):
    """With ``tol=1e-14`` the solver's own state is on the root, no column
    moves, and no step is taken: the state is the one ``steady_state()``
    returns, to the last bit. A* = B* = 5e-7 and dA*/dkf = -A0·kr/(kf + kr)²
    = -5e-5, dA*/dkr = +5e-5."""
    sim = bngsim.Simulator(_net(tmp_path, SMALL), method="ode")
    out = sim.steady_state(sensitivity_params=["kf", "kr"], tol=1e-14)
    np.testing.assert_allclose(np.asarray(out.concentrations), [5e-7, 5e-7], rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[-5e-5, 5e-5], [5e-5, -5e-5]], rtol=1e-3
    )
    assert out.sens_root_newton_steps == 0
    plain = sim.steady_state(tol=1e-14)
    assert np.array_equal(np.asarray(plain.concentrations), np.asarray(out.concentrations))


DECAY = """begin parameters
    1 k1  2.0
    2 k2  0.8
    3 k3  0.5
end parameters
begin species
    1 G() 100
    2 X() 0
    3 Y() 0
end species
begin reactions
    1 1 2 k1
    2 1 3 k2
    3 2 0 k3
    4 3 0 k3
end reactions
"""


def test_a_state_short_of_the_steady_state_by_less_is_stepped_to_it_too(tmp_path):
    """The star below at ``tol=1e-10``: the column of k1 was 8.5% from the one
    at the steady state. A* = A0/(1 + k1/k2 + k1/k4) and dA*/dk1 =
    -A0·(1/k2 + 1/k4)/(1 + k1/k2 + k1/k4)²."""
    sim = bngsim.Simulator(_net(tmp_path, STAR), method="ode")
    out = sim.steady_state(sensitivity_params=["k1"], tol=1e-10)
    assert np.asarray(out.concentrations)[0] == pytest.approx(1e-6 / 4.75, rel=1e-9)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(-750e-6 / 4.75**2, rel=1e-9)
    assert out.sens_root_newton_steps >= 1


SECOND_ORDER = """begin parameters
    1 k  1.0
end parameters
begin species
    1 A() 1.0
end species
begin reactions
    1 1,1 0 k
end reactions
"""


def test_a_root_of_higher_order_is_refused(tmp_path):
    """A + A -> 0 nears zero as 1/t and has a Jacobian of -2·k·A, which is zero
    there. A Newton step halves A, and the determinant with it. dA*/dk is 0
    here and the solve returned -A/(2·k), 1e-5; with a source at rate 0 beside
    it, A* is its square root and the derivative with respect to it has no
    value, where the solve returned 1/(4·k·A)."""
    sim = bngsim.Simulator(_net(tmp_path, SECOND_ORDER), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*that is 0\.5 of.*higher order"):
        sim.steady_state(sensitivity_params=["k"])


ORDER_M = """begin parameters
    1 k   1.0
    2 p   0
    3 m   {m}
end parameters
begin functions
    1 loss() k*Xobs^(m-1)
end functions
begin species
    1 X() 1.0
end species
begin reactions
    1 1 0 loss
    2 0 1 p
end reactions
begin groups
    1 Xobs 1
end groups
"""


def test_a_root_of_an_order_under_two_has_no_column_for_what_moves_it(tmp_path):
    """X' = p - k·X^m at p = 0 has a root at nothing of order m. Under a Newton
    step the determinant keeps ((m-1)/m)^(m-1) of itself: 0.58 at m = 1.5,
    which is refused for that, and 0.70 at 1.2, which the limit lets by. The
    columns answer there. X* = (p/k)^(1/m) has no derivative with respect to p
    at p = 0, and the column of p is 6^0.2 times what it was at every step, so
    that it moves by 30% of itself and never settles: 29.4 came back, and 721
    at 1.5. With respect to k the derivative is 0, which the steps settle at."""
    low = bngsim.Simulator(_net(tmp_path, ORDER_M.format(m="1.2")), method="ode")
    for asked in (["p"], ["k", "p"]):
        with pytest.raises(
            bngsim.SimulationError,
            match=r"#995.*no state near.*the column of p still moves by 30\.1%",
        ):
            low.steady_state(sensitivity_params=asked)
    out = low.steady_state(sensitivity_params=["k"])
    assert out.sens_root_determinant_ratio == pytest.approx((1 / 6) ** 0.2, rel=1e-6)
    assert out.sens_root_newton_steps >= 2
    assert abs(np.asarray(out.concentrations)[0]) < 1e-9
    assert abs(np.asarray(out.sensitivity)[0, 0]) < 1e-9
    half = bngsim.Simulator(_net(tmp_path, ORDER_M.format(m="1.5"), "half.net"), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*that is 0\.58 of"):
        half.steady_state(sensitivity_params=["k", "p"])


FAR = """begin parameters
    1 k   1e-12
    2 d   1.0
end parameters
begin species
    1 A() 3e-7
end species
begin reactions
    1 0 1 k
    2 1,1 0 d
end reactions
"""


def test_a_determinant_that_grows_is_refused(tmp_path):
    """dA/dt = k - 2·d·A² has an isolated root, sqrt(k/(2·d)) = 7.07e-7. With
    k = 1e-12 the residual at the start, A = 3e-7, is below ``tol`` already, and
    the solve returns the start. A Newton step from there lands at 9.8e-7,
    where the Jacobian, -4·d·A, is 3.3 times what it was: the state is not
    near the root, and dA*/dk came back 8.3e5 for 3.5e5 with nothing logged."""
    sim = bngsim.Simulator(_net(tmp_path, FAR), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*that is 3\.3 of.*grows.*smaller tol"):
        sim.steady_state(sensitivity_params=["k"])


def test_the_same_solved_to_its_root_is_returned(tmp_path):
    """Control. What the refusal advises, done: at ``tol=1e-17`` the solve runs
    to the root, and the columns are 1/(4·d·A*) and -A*/(2·d)."""
    sim = bngsim.Simulator(_net(tmp_path, FAR), method="ode")
    out = sim.steady_state(sensitivity_params=["k", "d"], tol=1e-17, max_time=1e9)
    root = math.sqrt(1e-12 / 2.0)
    assert np.asarray(out.concentrations)[0] == pytest.approx(root, rel=1e-5)
    assert np.asarray(out.sensitivity)[0] == pytest.approx(
        [1.0 / (4.0 * root), -root / 2.0], rel=1e-4
    )


TWO_SIDES = """begin parameters
    1 c   1e-12
    2 k0  6*c
    3 k1  11*c
    4 k2  6*c
    5 k3  c
    6 kb  2.0
    7 db  1.0
end parameters
begin species
    1 A() 2.475
    2 B() 2.0
end species
begin reactions
    1 0 1 k0
    2 1 0 k1
    3 1,1 1,1,1 k2
    4 1,1,1 1,1 k3
    5 0 2 kb
    6 2 0 db
end reactions
"""


@pytest.mark.parametrize("param", ["k0", "kb"])
def test_a_determinant_that_changes_sign_is_refused(tmp_path, param):
    """dA/dt = -c·(A-1)·(A-2)·(A-3), which a run takes from 2.475 to 3, passes
    ``tol`` where it starts with c = 1e-12; B beside it is at its root. The
    Jacobian of A is zero at 1.42 and at 2.58, and a Newton step from 2.475
    lands at 1.34, past the first: the determinant is -0.99 of what it was.
    dA*/dk0 came back -3.1e12 for 5e11. B's column, the one ``kb`` asks for, is
    the same at both states and was right: it is refused with a state that has
    a singular Jacobian between it and its correction, which only the sign of
    the ratio shows."""
    sim = bngsim.Simulator(_net(tmp_path, TWO_SIDES), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*that is -0\.99 of.*changes sign"):
        sim.steady_state(sensitivity_params=[param])


def test_a_column_that_is_nothing_at_the_root_is_returned_as_nothing(tmp_path):
    """Everything decays to zero, and so does every column. What came back was
    what ``tol`` left of them, 1e-9, and one Newton step takes all of that
    away: a column that moves by all of itself. The state is stepped, the
    columns are nothing at the next state and at the one after, and that is
    what is returned. (X and Y are at a zero, taken over the 100 that G, which
    makes them, started at. The last step leaves one of them at -6e-59, and
    it is returned at zero: what rounding left below a zero is no
    concentration.)"""
    out = bngsim.Simulator(_net(tmp_path, DECAY), method="ode").steady_state(
        sensitivity_params=["k1", "k2", "k3"]
    )
    assert out.sens_root_newton_steps == 2
    assert np.max(np.abs(np.asarray(out.sensitivity))) < 1e-30
    assert np.max(np.abs(np.asarray(out.concentrations))) < 1e-30
    assert np.min(np.asarray(out.concentrations)) >= 0.0
    np.testing.assert_allclose(out.sens_species_scale, [100.0, 100.0, 100.0])


# A <-> B -> P beside A -> C <-> D: with P masked out, A and B drain into it and
# what C and D hold between them is set by the race between the two branches.
SHARE_OUT_OF_THE_SINK = """begin parameters
    1 kf  1.0
    2 kr  0.5
    3 kb  0.7
    4 k2  0.3
    5 kc  0.4
    6 kd  0.9
end parameters
begin species
    1 A() 1.0
    2 B() 0
    3 P() 0
    4 C() 0
    5 D() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
    3 2 3 kb
    4 1 4 k2
    5 4 5 kc
    6 5 4 kd
end reactions
"""


def test_a_share_of_a_total_that_stays_out_of_the_masked_sink_is_refused(tmp_path):
    """C + D ends at the share of A that took the second branch, which no
    steady-state equation holds. With P masked out the columns of C and D came
    back 0 for every parameter, k2 among them, where differences of runs give
    dC*/dk2 = 0.59: the law A + B + C + D + P was held with P a constant."""
    model = _net(tmp_path, SHARE_OUT_OF_THE_SINK)
    assert model.pure_sink_species() == ["P()"]
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"dY_ss/dp does not exist.*singular"):
        sim.steady_state(sensitivity_params=["k2"], mask=~np.asarray(model.is_pure_sink()))


DRAINED = """begin parameters
    1 kf    1.0
    2 kr    0.5
    3 kcat  0.7
    4 S0    2.0
    5 E0    0.3
end parameters
begin species
    1 S() S0
    2 E() E0
    3 ES() 0
    4 P() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 3 2,4 kcat
end reactions
"""


def test_a_law_the_masked_sink_drains_is_as_it_was(tmp_path):
    """Control. S + E <-> ES -> E + P with P masked out: S and ES end at 0
    whatever the parameters, E at E0, and the equations of the kept species
    say so."""
    model = _net(tmp_path, DRAINED)
    sim = bngsim.Simulator(model, method="ode")
    out = sim.steady_state(
        sensitivity_params=["kcat", "S0", "E0"], mask=~np.asarray(model.is_pure_sink()), tol=1e-12
    )
    column = np.asarray(out.sensitivity)
    np.testing.assert_allclose(column[0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(column[1], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(column[2], [0, 0, 0], atol=1e-6)
    assert np.all(np.isnan(column[3]))


LEFT_OVER = """begin parameters
    1 k   1.0
    2 A0  2.0
    3 B0  1.3
end parameters
begin species
    1 A() A0
    2 B() B0
    3 P() 0
end species
begin reactions
    1 1,2 3 k
end reactions
"""


def test_two_laws_that_share_a_masked_sink_leave_the_one_the_kept_species_keep(tmp_path):
    """A + B -> P ends with B gone and A* = A0 - B0. The laws are A + P and
    B + P, and with P masked out neither is an equation of A and B; their
    difference is. With both laws held there was no unknown left and the
    request was refused as having no gradient. The masked species is taken
    out of one law by the other: dA*/d(k, A0, B0) = (0, 1, -1) and dB* = 0."""
    model = _net(tmp_path, LEFT_OVER)
    assert model.conservation_laws["n_laws"] == 2 and model.pure_sink_species() == ["P()"]
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["k", "A0", "B0"], mask=~np.asarray(model.is_pure_sink()), tol=1e-12
    )
    column = np.asarray(out.sensitivity)
    np.testing.assert_allclose(column[0], [0, 1, -1], atol=1e-8)
    np.testing.assert_allclose(column[1], [0, 0, 0], atol=1e-8)
    assert np.all(np.isnan(column[2]))


SLOW = """begin parameters
    1 s   0
    2 kd  1e-12
end parameters
begin species
    1 W() 0
end species
begin reactions
    1 0 1 s
    2 1 0 kd
end reactions
"""


def test_a_column_that_would_take_longer_than_max_time_is_refused(tmp_path):
    """W is made at s, which is 0, and removed at kd = 1e-12: W* = s/kd, a
    steady state that takes 1e12 to reach. The state is a root to the last
    bit, the system is 1x1, and dW*/ds came back 1e12 at a conditioning ratio
    of 1.0 with nothing logged, where a run of ``max_time`` = 1e6 moves W by
    1e6 per unit of s."""
    sim = bngsim.Simulator(_net(tmp_path, SLOW), method="ode")
    with pytest.raises(bngsim.SimulationError) as caught:
        sim.steady_state(sensitivity_params=["s"])
    message = str(caught.value)
    assert "#995" in message and "column of s" in message
    assert "max_time (1e+06) would leave 100% of the column" in message
    assert "Raise max_time" in message
    # The time it names is the one the solve was given.
    again = bngsim.Simulator(_net(tmp_path, SLOW), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"max_time \(1e\+09\) would leave 100%"):
        again.steady_state(sensitivity_params=["s"], max_time=1e9)


@pytest.mark.parametrize("max_time", [1e14, 3e13])
def test_the_same_given_the_time_is_returned(tmp_path, max_time):
    """Control. With ``max_time`` at 100 or at 30 relaxation times the column
    is that of a steady state a run of that length reaches: 1e12. (At 30, one
    implicit step of the whole time would leave 3% of it and eight leave
    4e-6, where a run leaves 1e-13.)"""
    sim = bngsim.Simulator(_net(tmp_path, SLOW), method="ode")
    out = sim.steady_state(sensitivity_params=["s"], max_time=max_time)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(1e12, rel=1e-9)


SMALL_SHARE = """begin parameters
    1 s    0
    2 kd   1e-12
    3 eps  1e-15
    4 kx   1.0
end parameters
begin functions
    1 made() eps*s
end functions
begin species
    1 W() 0
    2 X() 0
end species
begin reactions
    1 0 1 made
    2 1 0 kd
    3 0 2 s
    4 2 0 kx
end reactions
"""


def test_a_small_share_of_a_column_on_a_slow_mode_is_returned(tmp_path):
    """Control. X is made at s and removed at 1: dX*/ds = 1. W is made at
    1e-15·s and removed at 1e-12: dW*/ds = 1e-3, of a W* no run of 1e6
    reaches. The column is right to that 0.1% of its largest entry.
    A⁻¹·column/max_time is 1e3 for it, as it would be for a column that was
    all W; taken in steps, the run leaves the share that is W's and no
    more."""
    sim = bngsim.Simulator(_net(tmp_path, SMALL_SHARE), method="ode")
    out = sim.steady_state(sensitivity_params=["s"])
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [1e-3, 1.0], rtol=1e-9)


def test_what_a_run_of_max_time_gives_for_it(tmp_path):
    """Control. dW(T)/ds = (1 - exp(-kd·T))/kd, which is T to one part in
    1e6 at T = 1e6."""
    sim = bngsim.Simulator(_net(tmp_path, SLOW), method="ode", sensitivity_params=["s"])
    result = sim.run(t_span=(0.0, 1e6), n_points=2, rtol=1e-10, atol=1e-12)
    assert np.asarray(result.sensitivities)[-1][0, 0] == pytest.approx(1e6, rel=1e-5)


def test_a_healthy_solve_reports_what_was_measured(tmp_path):
    """A <-> B at ordinary concentrations: no pivot moves, there is one, and
    the columns stay where they are. Without sensitivities nothing is
    measured."""
    text = SMALL.replace("1e-6", "1.0").replace("5e-3", "0.5")
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    out = sim.steady_state(sensitivity_params=["kf"])
    assert out.sens_root_determinant_ratio == pytest.approx(1.0, abs=1e-6)
    assert out.sens_root_condition == pytest.approx(1.0)
    assert out.sens_root_column_shift < 1e-6
    # A <-> B at kf = kr = 0.5 relaxes at 1 per unit time: the bound is 1e-6
    # of the column left after max_time = 1e6.
    assert out.sens_root_relaxation == pytest.approx(1e-6, rel=1e-6)
    assert out.sens_root_condition_species in ("A()", "B()")
    assert out.sens_root_relaxation_param == "kf"
    # One pivot, and nothing cancelled into it; the run taken on from a
    # millionth beside the state gets to max_time and is back at it; the one
    # eigenvalue is -(kf + kr).
    assert out.sens_root_pivot_share == pytest.approx(1.0) and out.sens_root_pivot_species
    assert out.sens_root_hold_shift < 1e-6 and out.sens_root_hold_drift < 1e-6
    assert out.sens_root_hold_time == pytest.approx(1e6) and out.sens_root_hold_steps > 0
    assert out.sens_root_growth_rate == pytest.approx(-1.0, rel=1e-9)
    assert out.sens_root_spectral_radius == pytest.approx(1.0, rel=1e-9)
    assert out.sens_mask_held_species is None and out.sens_mask_reader_species is None
    plain = sim.steady_state()
    assert plain.sens_root_determinant_ratio == 1.0 and plain.sens_root_condition == 1.0
    assert plain.sens_root_column_shift == 0.0 and plain.sens_root_relaxation == 0.0
    assert plain.sens_root_determinant_species is None and plain.sens_root_column_param is None
    assert plain.sens_root_relaxation_param is None
    assert plain.sens_root_pivot_share == 1.0 and plain.sens_root_hold_shift == 0.0
    assert math.isnan(plain.sens_root_growth_rate) and plain.sens_root_spectral_radius == 0.0
    assert plain.sens_root_hold_steps == 0 and plain.sens_root_hold_time == 0.0


@pytest.mark.parametrize("v1, v2, k", [(0.7, 2.3, 0.37), (0.3, 1.9, 0.41), (1.1, 0.7, 0.53)])
def test_a_pivot_that_the_law_reduction_cancelled_is_refused(v1, v2, k):
    """A in c1 goes to B in c2 at k·(A·c1 + B·c2) and comes back at k·T0, with
    T0 the amount the model starts with: nothing moves, and every state with
    that amount is a steady state. The reduced system is 1x1, the law's column
    and A's own cancelling to what rounding leaves, 1e-17, where with one size
    they cancel to an exact zero. A 1x1 matrix has min|U|/max|U| = 1, and
    dA*/dk came back 258.7, 1556 and -1.19 for the three pairs of sizes, where
    it is 0, with no warning. What the entry was made of is carried into the
    pivot's share of them, which is 1e-16 to 1e-18."""
    model = bngsim.Model.from_antimony_string(
        f"compartment c1, c2; c1 = {v1}; c2 = {v2}; species A in c1, B in c2; A = 1.3; B = 0.2;\n"
        f"k = {k}; T0 = {v1 * 1.3 + v2 * 0.2!r};\nR1: A -> B; k*(A*c1 + B*c2)\nR2: B -> A; k*T0\n"
    )
    assert model.conservation_laws["n_laws"] == 1
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*pivot for B is \d\.\de-1[5-9] of the terms"
    ):
        sim.steady_state(sensitivity_params=["k"])
    out = sim.steady_state()  # the state itself is where the model started
    np.testing.assert_allclose(np.asarray(out.concentrations), [1.3, 0.2], rtol=1e-9)


def test_the_column_shift_is_the_distance_to_the_columns_of_the_root(tmp_path):
    """For a linear model one Newton step lands on the steady state, so the
    column shift is how far the returned column is from the one a solve to
    1e-15 returns, over its largest entry: 0.1% here at ``tol=3e-12``, where
    neither it nor the state moves by the 1% that would have the state stepped
    on. The
    largest of the three differences is that of A, which the law is solved
    for and which follows from the other two."""
    model = _net(tmp_path, STAR)
    assert [model.species_names[i] for i in model.conservation_laws["dependent"]] == ["A()"]
    loose = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["k1"], tol=3e-12
    )
    assert loose.sens_root_newton_steps == 0 and 1e-4 < loose.sens_root_state_shift < 1e-2
    tight = bngsim.Simulator(_net(tmp_path, STAR), method="ode").steady_state(
        sensitivity_params=["k1"], tol=1e-15
    )
    off = np.abs(np.asarray(tight.sensitivity) - np.asarray(loose.sensitivity))[:, 0]
    assert int(np.argmax(off)) == 0  # A's
    expected = off.max() / np.abs(np.asarray(loose.sensitivity)).max()
    assert 3e-4 < expected < 1e-2
    assert loose.sens_root_column_shift == pytest.approx(expected, rel=2e-3)
    assert loose.sens_root_column_param == "k1"


# X decays, and Y is made at a Hill rate of X with an exponent of 2.5, which
# has no value for a negative X.
BELOW_ZERO = """begin parameters
    1 kd  1.0
    2 k2  0.5
    3 v   2.0
    4 K   0.3
    5 ky  0.7
end parameters
begin functions
    1 hill() v*(Xtot^2.5/(K^2.5+Xtot^2.5))
end functions
begin species
    1 X() 1.0
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 1,1 1,1,1 k2
    3 0 2 hill
    4 2 0 ky
end reactions
begin groups
    1 Xtot 1
end groups
"""


def test_a_concentration_the_newton_step_takes_below_zero_is_set_to_zero(tmp_path):
    """X' = -kd·X + k2·X² ends at 0, and from X = 6e-14 a Newton step
    overshoots it, to -k2·X²/kd. The Hill rate there is not a number, and
    neither would the Jacobian at the stepped state be: the columns, which are
    nothing, would be refused for a pivot that is not one. The stepped state
    has 0 for such a species, and that is the state returned."""
    sim = bngsim.Simulator(_net(tmp_path, BELOW_ZERO), method="ode")
    out = sim.steady_state(sensitivity_params=["kd", "v", "K"], atol=1e-14, rtol=1e-10)
    assert out.converged and np.all(np.asarray(out.concentrations) == 0.0)
    assert np.max(np.abs(np.asarray(out.sensitivity))) < 1e-10


# ── What the first review found ──────────────────────────────────────────────

READ_BY_ITS_NEIGHBOUR = """begin parameters
    1 s  1.0
    2 k1 1.0
    3 k2 0.5
    4 k3 0.25
end parameters
begin species
    1 A() 0
    2 M() 0
end species
begin reactions
    1 0 1 s
    2 1 2 k1
    3 2 1 k2
    4 2 0 k3
end reactions
"""


def test_a_mask_that_leaves_out_a_species_an_equation_reads_is_refused(tmp_path):
    """A is made at s and exchanges with M, which is removed at k3: A* =
    s·(k2 + k3)/(k1·k3), 3 here, and dA*/ds = 3. With M masked out, A is
    solved for with M held where it is, 4, and that has dA/ds = 1/k1 = 1,
    which came back with nothing said. A species the mask leaves out is held,
    and that is its part in the derivative only where no equation the columns
    are solved on reads it."""
    sim = bngsim.Simulator(_net(tmp_path, READ_BY_ITS_NEIGHBOUR), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*mask= leaves out M\(\), which the rate of A\(\) reads.*pure_sink_species",
    ):
        sim.steady_state(sensitivity_params=["s", "k1"], mask=["A()"])


def test_the_same_with_nothing_left_out_is_returned(tmp_path):
    """Control. dA*/ds = 3, dA*/dk1 = -3, dM*/ds = 4 and dM*/dk1 = 0."""
    sim = bngsim.Simulator(_net(tmp_path, READ_BY_ITS_NEIGHBOUR), method="ode")
    out = sim.steady_state(sensitivity_params=["s", "k1"], tol=1e-12)
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[3.0, -3.0], [4.0, 0.0]], rtol=1e-6, atol=1e-6
    )


EXCHANGE = """begin parameters
    1 kf 1.0
    2 kr 0.5
    3 A0 2.0
end parameters
begin species
    1 A() A0
    2 M() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
"""


def test_a_masked_species_a_law_holds_with_one_that_reads_it_is_refused(tmp_path):
    """A <-> M, with A + M conserved, A the species the law is solved for, and
    M masked out. The law is not an equation of A alone, so A is an unknown
    with its own equation, and that equation reads M: dA*/dkf is -4/9, and
    with M held it is -2/3, which a first version of this fix returned where
    main refused the request as leaving nothing to solve for."""
    model = _net(tmp_path, EXCHANGE)
    assert [model.species_names[i] for i in model.conservation_laws["dependent"]] == ["A()"]
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*mask= leaves out M\(\)"):
        sim.steady_state(sensitivity_params=["kf", "kr", "A0"], mask=["A()"])


def test_a_masked_species_its_law_gives_is_not_held(tmp_path):
    """Control. The same with A masked out and M kept: A is what the law is
    solved for, it follows M through the law, and M's columns are those of
    M* = A0·kf/(kf + kr): 4/9, -8/9 and 2/3."""
    sim = bngsim.Simulator(_net(tmp_path, EXCHANGE), method="ode")
    out = sim.steady_state(sensitivity_params=["kf", "kr", "A0"], mask=["M()"], tol=1e-12)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[1], [4 / 9, -8 / 9, 2 / 3], rtol=1e-6)
    assert np.all(np.isnan(np.asarray(out.sensitivity)[0]))


BESIDE = {
    "a species nothing touches": ("    3 Z() 1", "", "    1 1 2 kf"),
    "a pair of its own": (
        "    3 C() 0.5\n    4 D() 0.5",
        "    3 3 4 kc\n    4 4 3 kc",
        "    1 1 2 kf",
    ),
    # A + G -> B + G: the rate of A reads G, at 1.
    "a species its rate reads": ("    3 G() 1", "", "    1 1,3 2,3 kf"),
}


def _small_beside(beside: str) -> str:
    species, reactions, forward = BESIDE[beside]
    text = SMALL.replace("    3 A0     1e-6\n", "    3 A0     1e-6\n    4 kc     5.0\n")
    text = text.replace("    1 1 2 kf\n", forward + "\n")
    text = text.replace("    2 B() 0\n", f"    2 B() 0\n{species}\n")
    return text.replace(
        "    2 2 1 kr\n", f"    2 2 1 kr\n{reactions}\n" if reactions else "    2 2 1 kr\n"
    )


@pytest.mark.parametrize("beside", sorted(BESIDE))
def test_a_state_short_of_the_steady_state_is_stepped_to_it_whatever_stands_beside_it(
    tmp_path, beside
):
    """The pair at 1e-6 above, 20% and more short of its steady state, in a
    model that also holds a species at 1 that nothing touches, or a pair at
    0.5 each that is at its own steady state, or a species at 1 that the
    pair's own rate reads. A column was small or not against the largest
    concentration in the model, and then against the largest among the
    species a rate couples it to, and the 1 made every entry of this one
    small: dA*/dkf came back 37% and 55% off, with nothing said. Each species
    is taken over its own concentration now, the columns are seen to move,
    and the state is stepped to the root."""
    sim = bngsim.Simulator(_net(tmp_path, _small_beside(beside)), method="ode")
    out = sim.steady_state(sensitivity_params=["kf", "kr"])
    np.testing.assert_allclose(
        np.asarray(out.sensitivity)[:2], [[-5e-5, 5e-5], [5e-5, -5e-5]], rtol=1e-8
    )
    assert np.all(np.asarray(out.sensitivity)[2:] == 0.0)
    assert out.sens_root_newton_steps >= 1
    np.testing.assert_allclose(out.sens_species_scale[:2], [5e-7, 5e-7], rtol=1e-6)


@pytest.mark.parametrize("beside", sorted(BESIDE))
def test_the_same_solved_to_its_steady_state_beside_them_is_returned(tmp_path, beside):
    """Control. dA*/dkf = -A0·kr/(kf + kr)² = -5e-5, and nothing for the
    species beside."""
    sim = bngsim.Simulator(_net(tmp_path, _small_beside(beside)), method="ode")
    out = sim.steady_state(sensitivity_params=["kf"], tol=1e-15)
    column = np.asarray(out.sensitivity)[:, 0]
    np.testing.assert_allclose(column[:2], [-5e-5, 5e-5], rtol=1e-4)
    assert np.all(column[2:] == 0.0)


BRANCH = """begin parameters
    1 k1 1.0
    2 k2 0.7
    3 k3 0.9
    4 k4 0.8025
    5 k5 1.5115
end parameters
begin species
    1 X() 3.432
    2 Y() 0
    3 Z() 0
    4 P1() 0
    5 P2() 0
end species
begin reactions
    1 1 2 k1
    2 2 1 k2
    3 2 3 k3
    4 3 4 k4
    5 3 5 k5
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_branch_to_two_products_is_refused(tmp_path, method):
    """X <-> Y -> Z, and Z goes to P1 at k4 and to P2 at k5: P1* is the share
    k4/(k4 + k5) of the total, which no equation of the steady state holds
    once Z is gone, and dP1*/dk4 is 0.969. The two products have one column
    between them in the reduced system. Its second pivot is 4e-18 of the
    terms it was computed from, and the condition number, 15, does not see
    it: the null vectors of the two sides have no entry in common."""
    sim = bngsim.Simulator(_net(tmp_path, BRANCH), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*pivot for P[12]\(\) is \d\.\de-1[5-9] of the terms.*singular whatever",
    ):
        sim.steady_state(sensitivity_params=["k4", "k5"], method=method)


SWITCH = """begin parameters
    1 d   0.5
    2 Xu  1e-5
    3 K   1.0
    4 k1  d*Xu*K
    5 k2  d*(Xu+K)
    6 k3  d
    7 c   1e-5
    8 g   1e-5
end parameters
begin species
    1 X() 0
    2 Y() 5e-5
end species
begin reactions
    1 1 0 k1
    2 1,1 1,1,1 k2
    3 1,1,1 1,1 k3
    4 2 1,2 c
    5 2 0 g
end reactions
"""


def test_a_state_that_a_run_leaves_is_refused(tmp_path):
    """X' = -d·X·(X - Xu)·(X - K) + c·Y and Y' = -g·Y: X rests at 0 and at K,
    with a threshold at Xu = 1e-5 between, and Y, which decays, pushes X over
    it. The residual where the model starts is 3.5e-10, under ``tol``, so the
    solve returns the start, X = 0 beside the root at 0, which is stable: no
    pivot moves, the columns stay where they are and dX*/dK = 0. A run ends at
    X = K, where dX*/dK = 1. The run is taken on from the returned state for
    ``max_time``, and the columns solved again where it ends."""
    sim = bngsim.Simulator(_net(tmp_path, SWITCH), method="ode")
    with pytest.raises(bngsim.SimulationError) as caught:
        sim.steady_state(sensitivity_params=["K"])
    message = str(caught.value)
    assert "#995" in message and "not one a run stays at" in message
    assert "max_time (1e+06)" in message and "column of K" in message
    assert "Solve again with a smaller tol" in message


def test_the_same_solved_past_its_switch_is_returned(tmp_path):
    """Control. What the refusal advises: at ``tol=1e-14`` the solve runs on
    to X = K, and dX*/dK = 1."""
    sim = bngsim.Simulator(_net(tmp_path, SWITCH), method="ode")
    out = sim.steady_state(sensitivity_params=["K"], tol=1e-14)
    assert np.asarray(out.concentrations)[0] == pytest.approx(1.0, rel=1e-6)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(1.0, rel=1e-6)


def test_where_a_run_of_it_ends(tmp_path):
    """Control. X = K at 1e6."""
    sim = bngsim.Simulator(_net(tmp_path, SWITCH), method="ode")
    end = np.asarray(sim.run(t_span=(0, 1e6), n_points=3).species)[-1]
    assert end[0] == pytest.approx(1.0, rel=1e-6) and abs(end[1]) < 1e-8


SIGNED = """begin parameters
    1 c  1e-9
    2 a  1.0
    3 b  1.6487212707001282
    4 s  1e-5
end parameters
begin functions
    1 rate() c*(a-b*exp(Xobs/s))
end functions
begin species
    1 X() 1e-8
end species
begin reactions
    1 0 1 rate
end reactions
begin groups
    1 Xobs 1
end groups
"""


def test_a_variable_with_a_sign_is_not_stopped_at_zero(tmp_path):
    """X' = c·(a - b·exp(X/s)) rests at s·ln(a/b), -5e-6, where dX*/da = s/a
    = 1e-5. It starts at +1e-8 with a residual under ``tol`` and is returned
    there, 6.1e-6 for the column. The Newton step takes it below zero, and
    with the stepped state held at zero, as it is for a concentration a rate
    has no value below, the column moved by 0.1% and came back. It is stepped
    to where it rests, until a step moves the column by under 1%."""
    sim = bngsim.Simulator(_net(tmp_path, SIGNED), method="ode")
    out = sim.steady_state(sensitivity_params=["a"])
    assert np.asarray(out.concentrations)[0] == pytest.approx(-5e-6, rel=1e-4)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(1e-5, rel=1e-4)
    assert out.sens_root_newton_steps >= 2


def test_the_same_solved_to_its_root_below_zero_is_returned(tmp_path):
    """Control."""
    sim = bngsim.Simulator(_net(tmp_path, SIGNED), method="ode")
    out = sim.steady_state(sensitivity_params=["a"], tol=1e-16, max_time=1e9)
    assert np.asarray(out.concentrations)[0] == pytest.approx(-5e-6, rel=1e-5)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(1e-5, rel=1e-5)


THREE_ROOTS = """begin parameters
    1 k  1.0
    2 a  0.25
    3 k1 k*a
    4 k2 k*(1+a)
    5 k3 k
end parameters
begin species
    1 X() {x0}
end species
begin reactions
    1 1 0 k1
    2 1,1 1,1,1 k2
    3 1,1,1 1,1 k3
end reactions
"""


def test_a_root_the_system_does_not_rest_at_is_refused(tmp_path):
    """X' = -k·X·(X - a)·(X - 1) started on its middle root, a: the residual
    is zero, the solve returns the start, and nothing moves it. The Jacobian
    there is +k·a·(1 - a) = 0.1875, and a state beside a goes to 0 or to 1.
    dX*/da = 1 came back, which is how the root moves and not where a run
    ends."""
    sim = bngsim.Simulator(_net(tmp_path, THREE_ROOTS.format(x0="a")), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 0\.188"
    ):
        sim.steady_state(sensitivity_params=["a"])


def test_the_root_a_run_of_it_ends_at_is_returned(tmp_path):
    """Control. From 0.9 the run ends at 1, which a does not move."""
    sim = bngsim.Simulator(_net(tmp_path, THREE_ROOTS.format(x0="0.9")), method="ode")
    out = sim.steady_state(sensitivity_params=["a"])
    assert np.asarray(out.concentrations)[0] == pytest.approx(1.0, rel=1e-8)
    assert abs(np.asarray(out.sensitivity)[0, 0]) < 1e-8


BRUSSELATOR = """begin parameters
    1 A   1.0
    2 B   {B}
    3 one 1.0
end parameters
begin species
    1 X() 1.0
    2 Y() {B}
end species
begin reactions
    1 0 1 A
    2 1 2 B
    3 1,1,2 1,1,1 one
    4 1 0 one
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_focus_the_system_spirals_out_of_is_refused(tmp_path, method):
    """The Brusselator started on its fixed point, (A, B/A), with B = 3: the
    eigenvalues are 0.5 ± 0.87i, and a state beside it spirals out to a limit
    cycle. The determinant is positive, as a stable point's is in two
    unknowns, so it is the spectrum that says so. Either solver returns the
    start."""
    sim = bngsim.Simulator(_net(tmp_path, BRUSSELATOR.format(B="3.0")), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 0\.5 "
    ):
        sim.steady_state(sensitivity_params=["A", "B"], method=method)


def test_a_focus_the_system_spirals_into_is_returned(tmp_path):
    """Control. With B = 1.5 the eigenvalues are -0.25 ± 0.97i, and the
    columns are those of (A, B/A): 1, 0, -B/A² and 1/A."""
    sim = bngsim.Simulator(_net(tmp_path, BRUSSELATOR.format(B="1.5")), method="ode")
    out = sim.steady_state(sensitivity_params=["A", "B"])
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[1.0, 0.0], [-1.5, 1.0]], rtol=1e-9, atol=1e-12
    )


def test_what_is_reported_of_a_state_the_system_rests_at(tmp_path):
    """The two controls above, as the result describes them: the cubic at its
    root at 1, where the Jacobian is -k·(1 - a) = -0.75, and the focus with
    eigenvalues -0.25 ± 0.97i, of size 1. The run taken on gets to
    ``max_time`` from both."""
    cubic = bngsim.Simulator(_net(tmp_path, THREE_ROOTS.format(x0="0.9")), method="ode")
    out = cubic.steady_state(sensitivity_params=["a"])
    assert out.sens_root_growth_rate == pytest.approx(-0.75, rel=1e-6)
    assert out.sens_root_spectral_radius == pytest.approx(0.75, rel=1e-6)
    assert out.sens_root_hold_time == pytest.approx(1e6)
    focus = bngsim.Simulator(_net(tmp_path, BRUSSELATOR.format(B="1.5")), method="ode")
    out = focus.steady_state(sensitivity_params=["A", "B"])
    assert out.sens_root_growth_rate == pytest.approx(-0.25, rel=1e-9)
    assert out.sens_root_spectral_radius == pytest.approx(1.0, rel=1e-9)
    assert out.sens_root_hold_time == pytest.approx(1e6)


# The Brusselator with a species beside it that is made and removed at a rate
# of kfast, and has nothing to do with it.
BESIDE_A_FAST_SPECIES = (
    BRUSSELATOR.replace("    3 one 1.0\n", "    3 one 1.0\n    4 kfast {kfast}\n")
    .replace("    2 Y() {B}\n", "    2 Y() {B}\n    3 F() 1.0\n")
    .replace("    4 1 0 one\n", "    4 1 0 one\n    5 0 3 kfast\n    6 3 0 kfast\n")
)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_focus_beside_a_fast_species_is_refused_all_the_same(tmp_path, method):
    """The same fixed point beside a species turned over at 1e7. The
    eigenvalue of 0.5 is 5e-8 of the largest, and it was asked against that:
    under a millionth of it the point was taken for one the system rests at,
    and the columns came back. It is asked against the time the solve was
    given, whatever stands beside it."""
    text = BESIDE_A_FAST_SPECIES.format(B="3.0", kfast="1e7")
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 0\.5 "
    ):
        sim.steady_state(sensitivity_params=["A", "B"], method=method)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_focus_among_too_many_species_for_the_spectrum_is_refused_by_the_run(tmp_path, method):
    """The fixed point with B = 3 in a model of 522 unknowns, where the
    eigenvalues are not taken. The run taken on from a millionth beside the
    state is what says the system leaves it: it is on the limit cycle when
    its steps are used up."""
    text = _among_many(BRUSSELATOR.format(B="3.0"))
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*not one a run stays at.*used its steps.*the run moves [XY]\(\) by \d+%",
    ):
        sim.steady_state(sensitivity_params=["A", "B"], method=method)


def test_a_focus_that_grows_slowly_is_refused_by_its_eigenvalue(tmp_path):
    """With B = 2.01 the eigenvalues are 0.005 ± i, and in a ``max_time`` of
    300 a millionth beside the fixed point grows to 4.5 millionths: the run
    taken on gets to its time, 1.6e-5 from where it started, and the columns
    it is solved at have not moved. A run of that length from the state a
    changed parameter leaves the model in is 4.5 times as far from the new
    fixed point as it started, and its end is not the root's derivative."""
    sim = bngsim.Simulator(_net(tmp_path, BRUSSELATOR.format(B="2.01")), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 0\.005 "
    ):
        sim.steady_state(sensitivity_params=["A", "B"], max_time=300.0)


def test_a_run_that_does_not_get_to_max_time_is_not_known_to_stay(tmp_path):
    """With B = 2.001 the growth is 0.0005, and among 520 other species there
    are no eigenvalues to say so. The run has 10,000 steps, gets to t = 6,000
    in them with ``max_time`` at 1e6, and is 1e-4 from where it started. That
    is not a run that stayed."""
    text = _among_many(BRUSSELATOR.format(B="2.001"))
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*did not get to max_time \(1e\+06\).*10000 steps.*not known to stay",
    ) as caught:
        sim.steady_state(sensitivity_params=["A", "B"])
    assert "more steps (max_steps)" in str(caught.value)
    assert "max_time such a run reaches" not in str(caught.value)


def test_a_shorter_max_time_does_not_get_a_growing_focus_returned(tmp_path):
    """The refusal above said to give a ``max_time`` the run reaches. With
    2,000, beside a species turned over at 1e7, the run does reach it, a
    millionth has grown to 2.7 of them, and the columns came back: (1, -2.001;
    0, 1) where differences of runs of that length give (6.16, -3.66; -2.48,
    4.52). The eigenvalue of 0.0005 is asked against the 2,000: a state beside
    this one is 172% further off by then."""
    text = BESIDE_A_FAST_SPECIES.format(B="2.001", kfast="1e7")
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*does not rest at.*real part of 0\.0005 .*172% further",
    ):
        sim.steady_state(sensitivity_params=["A", "B"], max_time=2000.0)


def test_a_focus_that_dies_away_slowly_is_returned(tmp_path):
    """With B = 1.9 the eigenvalues are -0.05 ± i: the run taken on winds in
    for some 1,100 steps and then gets to ``max_time``, and the columns are
    returned, as they were."""
    sim = bngsim.Simulator(_net(tmp_path, BRUSSELATOR.format(B="1.9")), method="ode")
    out = sim.steady_state(sensitivity_params=["A", "B"])
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[1.0, 0.0], [-1.9, 1.0]], rtol=1e-9, atol=1e-12
    )
    assert out.sens_root_hold_time == pytest.approx(1e6)
    assert 200 < out.sens_root_hold_steps < 10000


# X' = -k·X·(X - a)·(X - 1), slow, started a hundredth above its middle root,
# beside W made at s·g(X) with g = 1 + c·(X - a)·(X - 1): g is 1 at the middle
# root and at 1, and 0.963 where X starts.
BETWEEN_TWO_ROOTS = """begin parameters
    1 k   4e-7
    2 a   0.25
    3 k1  k*a
    4 k2  k*(1+a)
    5 k3  k
    6 s   2.0
    7 c   5.0
    8 kd  1.0
    9 W0  s*(1+c*(0.26-a)*(0.26-1))/kd
end parameters
begin functions
    1 made() s*(1+c*(Xobs-a)*(Xobs-1))
end functions
begin species
    1 X() 0.26
    2 W() W0
end species
begin reactions
    1 1 0 k1
    2 1,1 1,1,1 k2
    3 1,1,1 1,1 k3
    4 0 2 made
    5 2 0 kd
end reactions
begin groups
    1 Xobs 1
end groups
"""


def test_a_root_the_solve_steps_to_and_the_system_leaves_is_refused(tmp_path):
    """The residual is under ``tol`` where the model starts, and the column of
    s, g(X)/kd for W, is 3.7% from its value at a root: the state is stepped,
    and Newton takes X to the middle root, 0.25, which the system leaves at
    7.5e-8. In the 1e9 it is given that is a growth of exp(75)."""
    sim = bngsim.Simulator(_net(tmp_path, BETWEEN_TWO_ROOTS), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 7\.5e-08 "
    ):
        sim.steady_state(sensitivity_params=["s"], max_time=1e9)


def test_a_root_that_was_stepped_to_is_where_a_run_ends(tmp_path):
    """The same among 520 other species, with no eigenvalues. A run from
    where the model starts takes X to 1, and not to the 0.25 Newton stepped
    to. The column of s is 1/kd at both, so that it does not move, and what
    says the root is the wrong one is where the run ends."""
    sim = bngsim.Simulator(_net(tmp_path, _among_many(BETWEEN_TWO_ROOTS)), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*stepped to is not where a run ends.*ends \d\d% from that root in X\(\)",
    ):
        sim.steady_state(sensitivity_params=["s"], max_time=1e9)


def test_the_same_started_on_the_side_it_stays_on_is_returned(tmp_path):
    """Started at 0.99 the column of s is 0.4% from its value at the root, no
    step is taken, and the run ends at 1: the column is returned, as it
    was."""
    text = BETWEEN_TWO_ROOTS.replace("0.26", "0.99")
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    out = sim.steady_state(sensitivity_params=["s"], max_time=1e9)
    assert np.asarray(out.sensitivity)[1, 0] == pytest.approx(1.0, rel=1e-2)
    assert out.sens_root_newton_steps == 0


SLOW_SADDLE = """begin parameters
    1 e   1e-9
    2 kd  1e3
end parameters
begin species
    1 X() 0
    2 Y() 0
end species
begin reactions
    1 1 1,1 e
    2 2 0 kd
end reactions
"""


def test_a_species_that_is_not_there_is_not_moved(tmp_path):
    """X' = e·X with X = 0, beside Y' = -kd·Y. Any X grows, and there is
    none: a run from the state the model starts in stays, whatever e is, and
    the columns are zeros, as they were. The run that is taken on moves each
    concentration by a millionth of itself, which leaves a species that is
    absent absent: a state is not asked whether it would last an invasion
    nothing in the request brings."""
    sim = bngsim.Simulator(_net(tmp_path, SLOW_SADDLE), method="ode")
    out = sim.steady_state(sensitivity_params=["e", "kd"])
    assert np.all(np.asarray(out.sensitivity) == 0.0)
    assert out.sens_root_hold_drift == 0.0


def test_the_refusal_of_a_model_with_a_rule_species_does_not_speak_of_a_mask():
    """A species an assignment rule sets is left out of the solve by the
    solver itself, and the refusal took that for a mask of the caller's: it
    gave the paragraph on masks in place of the advice on pure sinks."""
    model = bngsim.Model.from_antimony_string(
        "species S, I, R, Q; S = 99; I = 1; R = 0; b = 0.018; g = 1;\n"
        "Q := S + I;\nJ1: S + I -> 2 I; b*S*I;\nJ2: I -> R; g*I;\n"
    )
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SimulationError) as caught:
        sim.steady_state(sensitivity_params=["g"])
    message = str(caught.value)
    assert "#995" in message and "the limits are 0.6 and 1.67" in message
    assert "pure_sink_species" in message and "With mask=" not in message


def test_the_share_a_slow_mode_has_of_a_column_is_what_is_reported(tmp_path):
    """The column of the control above: a thousandth of it is W's."""
    sim = bngsim.Simulator(_net(tmp_path, SMALL_SHARE), method="ode")
    out = sim.steady_state(sensitivity_params=["s"])
    assert out.sens_root_relaxation == pytest.approx(1e-3, rel=1e-3)
    assert out.sens_root_relaxation_param == "s"


SCALES = """begin parameters
    1 k   1.0
    2 kf  1.0
    3 kr  1.0
    4 kc  1.0
    5 s   1e-3
    6 kd  1.0
    7 km  1.0
end parameters
begin species
    1 X() 5
    2 S() 2
    3 E() 0.5
    4 C() 0
    5 P() 0
    6 Z() 7
    7 W() 0
    8 A() 1e-12
    9 B() 0
   10 M() 1000
end species
begin reactions
    1 1 0 k
    2 2,3 4 kf
    3 4 2,3 kr
    4 4 5,3 kc
    5 0 7 s
    6 7 0 kd
    7 8,10 9,10 km
end reactions
"""


def test_what_an_entry_for_each_species_is_small_against(tmp_path):
    """The scale of a species is its own concentration, the larger of the
    returned and the corrected one: P, which ends at 2, E, Z, which nothing
    touches, M, W at its steady 1e-3, and B, which ends with the 1e-12 that
    A started with. A species at a zero is taken over where it has been and
    what stands beside it: X, which decays from 5; S, used up from 2; C, which
    starts at nothing and ends there, and is given the 0.5 that it shares
    with E and not the 2 beside it; and A, used up at a rate that reads M, for
    which 1e-9 of M's 1000 is the least."""
    sim = bngsim.Simulator(_net(tmp_path, SCALES), method="ode")
    out = sim.steady_state(sensitivity_params=["kf"], tol=1e-13)
    np.testing.assert_allclose(
        out.sens_species_scale,
        [5.0, 2.0, 0.5, 0.5, 2.0, 7.0, 1e-3, 1e-6, 1e-12, 1000.0],
        rtol=1e-6,
    )
    assert sim.steady_state().sens_species_scale.size == 0


# S1 is made by M, at 1e-15·M, and S2 by S1: each at 1e-12 beside M's 1,000.
MADE_IN_A_CHAIN = """begin parameters
    1 k0  1e-15
    2 kd  1.0
    3 k2  1.0
    4 kx  1.0
end parameters
begin species
    1 M()  1000
    2 S1() 0
    3 S2() 0
    4 X()  5
end species
begin reactions
    1 1 1,2 k0
    2 2 0 kd
    3 2 2,3 k2
    4 3 0 kd
    5 4 0 kx
end reactions
"""


def test_a_species_made_by_one_that_is_made_is_not_at_a_zero_either(tmp_path):
    """S1 and S2 are 1e-15 of M, which is what rounding leaves of a zero
    beside it, and with both set to nothing the rate of S2 is nothing: only S1
    makes it. S1 is made by M, though, and is not at a zero, and with S1 where
    it is S2 is made too. The species are gone over until none is taken out.
    X, which decays from 5, is at a zero, and is taken over the 5. dS1*/dk0 =
    M/kd and dS2*/dk0 = k2·M/kd²."""
    sim = bngsim.Simulator(_net(tmp_path, MADE_IN_A_CHAIN), method="ode")
    out = sim.steady_state(sensitivity_params=["k0"])
    np.testing.assert_allclose(out.sens_species_scale, [1000.0, 1e-12, 1e-12, 5.0], rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(out.sensitivity)[:, 0], [0.0, 1000.0, 1000.0, 0.0], rtol=1e-6, atol=1e-12
    )


# B + B -> 0 at k/B: a flux of k·B, through a rate that has no value at B = 0.
NO_VALUE_AT_NOTHING = """begin parameters
    1 k   1.0
end parameters
begin functions
    1 rate() k/Bobs
end functions
begin species
    1 B() 1.0
end species
begin reactions
    1 1,1 0 rate
end reactions
begin groups
    1 Bobs 1
end groups
"""


def test_a_species_that_runs_out_through_a_rate_with_no_value_at_nothing_is_refused(tmp_path):
    """B runs out, and its column is what ``tol`` left of a zero, -7.7e-11.
    Whether B is at a zero is asked of its rate with B at nothing, where k/B
    has no value: B keeps its own scale, and is stepped to where the rate has
    none either. The columns are refused, here with entries that are right:
    nothing says so."""
    sim = bngsim.Simulator(_net(tmp_path, NO_VALUE_AT_NOTHING), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*no state near.*a Newton step leaves where it is.*no value at nothing",
    ) as caught:
        sim.steady_state(sensitivity_params=["k"])
    assert "the columns cannot be solved there: a rate has no value" in str(caught.value)


# X' = k0 - (k + eps·q)·X^1.4, from 1: X falls towards (k0/k)^(1/1.4) = 3.7e-22,
# and a Newton step from above takes it 2/7 of the way down, whatever is left.
STILL_FALLING = """begin parameters
    1 k0  1e-30
    2 k   1.0
    3 eps 1e-20
    4 q   1.0
end parameters
begin functions
    1 loss() (k+eps*q)*Xobs^0.4
end functions
begin species
    1 X() 1.0
end species
begin reactions
    1 0 1 k0
    2 1 0 loss
end reactions
begin groups
    1 Xobs 1
end groups
"""


@pytest.mark.parametrize(
    "asked, moving",
    [
        ("k", r"the column of k still moves by 71\.4% of its largest entry"),
        ("q", r"a step still moves X\(\) by 71\.4% of what it is taken over"),
    ],
)
def test_a_state_that_ten_steps_do_not_settle_is_refused(tmp_path, asked, moving):
    """The solve stops at 3.5e-7, fifteen orders above the root, and each
    Newton step takes X to 2/7 of where it was: ten of them are not enough.
    The determinant keeps 0.61 of itself, which is inside its limits. Asked
    for k, the column still moves. Asked for q, which is 1e-20 of the rate,
    the column is rounding and moves by none of that, and what has not
    settled is the state."""
    sim = bngsim.Simulator(_net(tmp_path, STILL_FALLING), method="ode")
    with pytest.raises(bngsim.SimulationError, match=rf"#995.*no state near.*{moving}"):
        sim.steady_state(sensitivity_params=[asked])


# A' = s + k·Ec·A - k·E·A with E held: A* = s/(k·(E - Ec)), and E is 2e-5 above
# Ec, so that A* moves 50,000 times as far as E does.
HELD_AT_THE_EDGE = """begin parameters
    1 s   2e-2
    2 k   1e3
    3 Ec  1.0
    4 kEc k*Ec
end parameters
begin species
    1 A() 1.0
    2 $E() 1.00002
end species
begin reactions
    1 0 1 s
    2 1 1,1 kEc
    3 1,2 2 k
end reactions
"""


def test_a_species_that_is_held_is_not_moved_for_the_run(tmp_path):
    """The run is taken on from a millionth beside the state, and a species
    the model holds fixed, or the mask leaves out, is no part of the state: E
    moved by its share, half a millionth, would put A* 2.6% from where it is,
    and the columns with it. dA*/ds = 1/(k·(E - Ec)) = 50."""
    sim = bngsim.Simulator(_net(tmp_path, HELD_AT_THE_EDGE), method="ode")
    out = sim.steady_state(sensitivity_params=["s"], mask=["A()"], tol=1e-12)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(50.0, rel=1e-6)
    assert out.sens_root_hold_shift < 1e-4


# X -> 0 at k1 beside X + X -> 0 at k2: X runs out, and near the end a Newton
# step leaves 2·k2·X²/k1 of it, which is not nothing: 5e-4 of where it was. W
# is made and removed on its own.
RUNS_OUT_IN_TWO_WAYS = """begin parameters
    1 k1  1.0
    2 k2  1e6
    3 s   2.0
    4 kw  1.0
end parameters
begin species
    1 X() 1e-3
    2 W() 2.0
end species
begin reactions
    1 1 0 k1
    2 1,1 0 k2
    3 0 2 s
    4 2 0 kw
end reactions
"""


def test_a_species_a_newton_step_takes_most_of_is_running_out(tmp_path):
    """X is returned at 1e-9, and one Newton step takes it to a five-hundredth
    of that: more than rounding leaves of the 1e-3 it started at, and nothing
    like where it was. It is at a zero for that, and taken over the 1e-3: the
    state has not moved, the column of s, which is W's, has not either, and
    the state is the solver's own. Asked for a value under rounding only, X
    kept its own scale, moved by all of it, and had the state stepped for
    it."""
    sim = bngsim.Simulator(_net(tmp_path, RUNS_OUT_IN_TWO_WAYS), method="ode")
    out = sim.steady_state(sensitivity_params=["s"])
    assert out.sens_root_newton_steps == 0
    assert out.sens_species_scale[0] == pytest.approx(1e-3)
    assert 1e-10 < abs(np.asarray(out.concentrations)[0]) < 1e-8
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [0.0, 1.0], atol=1e-12)


def test_the_same_asked_for_its_own_columns_is_stepped_to_nothing(tmp_path):
    """The columns of k1 and k2 are X's, what ``tol`` left of nothing, and
    they move by all of themselves: the state is stepped, and X and its
    columns come back as nothing."""
    sim = bngsim.Simulator(_net(tmp_path, RUNS_OUT_IN_TWO_WAYS), method="ode")
    out = sim.steady_state(sensitivity_params=["k1", "k2"])
    assert out.sens_root_newton_steps >= 2
    assert abs(np.asarray(out.concentrations)[0]) < 1e-15
    assert np.max(np.abs(np.asarray(out.sensitivity))) < 1e-15


# A corpus network, verbatim but for its comments and its observables
# (benchmarks/suites/ode_fullnet: my_models/ode/egfr_path.bngl). EGF starts at
# nothing, so that all that happens is Grb2 + Sos <-> Grb2_Sos, and the thirteen
# species EGF would make are returned at 1e-24 to 1e-52, of either sign: what
# the integrator's linear solves left of nothing beside species at 1e5.
NO_LIGAND = """begin parameters
    1 EGF_tot       1.2e6
    2 Rec_tot       1.8e5
    3 Grb2_tot      1.0e5
    4 Shc_tot       2.7e5
    5 SOS_tot       1.3e4
    6 Grb2_SOS_tot  4.9e4
    7 kp1           1.667e-06
    8 km1           0.06
    9 kp2           5.556e-06
   10 km2           0.1
   11 kp3           1
   12 km3           9
   13 kp14          6
   14 km14          0.06
   15 km16          0.005
   16 kp9           1.666e-6
   17 km9           0.05
   18 kp10          5.556e-06
   19 km10          0.06
   20 kp11          2.5e-06
   21 km11          0.03
   22 kp13          5e-05
   23 km13          0.6
   24 kp15          5e-07
   25 km15          0.3
   26 kp17          1.667e-06
   27 km17          0.1
   28 kp18          5e-07
   29 km18          0.3
   30 kp19          5.556e-06
   31 km19          0.0214
   32 kp20          1.333e-07
   33 km20          0.12
   34 kp24          5e-06
   35 km24          0.0429
   36 kp21          1.667e-06
   37 km21          0.01
   38 kp23          1.167e-05
   39 km23          0.1
   40 kp12          5.556e-08
   41 km12          0.0015
   42 kp22          1.667e-05
   43 km22          0.064
   44 loop1         ((kp9/km9)*(kp10/km10))/((kp11/km11)*(kp12/km12))
   45 loop2         ((kp15/km15)*(kp17/km17))/((kp21/km21)*(kp18/km18))
   46 loop3         ((kp18/km18)*(kp19/km19))/((kp22/km22)*(kp20/km20))
   47 loop4         ((kp12/km12)*(kp23/km23))/((kp22/km22)*(kp21/km21))
   48 loop5         ((kp15/km15)*(kp24/km24))/((kp20/km20)*(kp23/km23))
end parameters
begin species
    1 EGF() 0
    2 Grb2() Grb2_tot
    3 Grb2_Sos() Grb2_SOS_tot
    4 Shc() Shc_tot
    5 ShcP() 0
    6 ShcP_Grb2() 0
    7 ShcP_Grb2_Sos() 0
    8 Sos() SOS_tot
    9 R() Rec_tot
   10 RA() 0
   11 R2() 0
   12 RP() 0
   13 R_Sh() 0
   14 R_ShP() 0
   15 R_Sh_G() 0
   16 R_Sh_G_S() 0
   17 R_G() 0
   18 R_G_S() 0
end species
begin reactions
    1 1,9 10 kp1
    2 10 1,9 km1
    3 10,10 11 0.5*kp2
    4 11 10,10 km2
    5 11 12 kp3
    6 12 11 km3
    7 2,12 17 kp9
    8 17 2,12 km9
    9 8,17 18 kp10
   10 18 8,17 km10
   11 3,12 18 kp11
   12 18 3,12 km11
   13 4,12 13 kp13
   14 13 4,12 km13
   15 13 14 kp14
   16 14 13 km14
   17 5,12 14 kp15
   18 14 5,12 km15
   19 2,14 15 kp17
   20 15 2,14 km17
   21 6,12 15 kp18
   22 15 6,12 km18
   23 8,15 16 kp19
   24 16 8,15 km19
   25 7,12 16 kp20
   26 16 7,12 km20
   27 3,14 16 kp24
   28 16 3,14 km24
   29 2,5 6 kp21
   30 6 2,5 km21
   31 3,5 7 kp23
   32 7 3,5 km23
   33 5 4 km16
   34 2,8 3 kp12
   35 3 2,8 km12
   36 6,8 7 kp22
   37 7 6,8 km22
end reactions
"""


def test_species_at_rounding_beside_large_ones_are_at_a_zero(tmp_path):
    """A Newton step leaves each of the thirteen where it is, at 1e-24, so
    that none is halved; and each is made by others of them, so that what
    makes it is as small as it is. What they are rounding of is the species
    the Jacobian couples them to, at 1e5. Taken over themselves, their entries
    moved by seven times the column at every step, and the model was refused.
    Nothing but Grb2, Sos and their complex moves with kp12, and nothing at
    all with kp1 or km16."""
    sim = bngsim.Simulator(_net(tmp_path, NO_LIGAND), method="ode")
    out = sim.steady_state(sensitivity_params=["kp12", "km12", "kp1", "km16"])
    assert out.sens_root_newton_steps == 0
    names = list(out.species_names)
    scale = dict(zip(names, out.sens_species_scale, strict=True))
    assert min(scale.values()) > 1e4 and scale["EGF()"] > 1e4
    columns = np.asarray(out.sensitivity)
    # kp1 multiplies nothing, and km16 what is left of ShcP, 1e-23.
    assert np.all(columns[:, 2] == 0.0) and np.max(np.abs(columns[:, 3])) < 1e-12
    moving = [names.index(n) for n in ("Grb2()", "Grb2_Sos()", "Sos()")]
    # Grb2 + Sos <-> Grb2_Sos: with x the complex, kp12·G·S = km12·x and
    # dG = dS = -dx, so that dx/dkp12 = G·S/(km12 + kp12·(G + S)).
    g, x, s = (float(np.asarray(out.concentrations)[i]) for i in moving)
    exact = g * s / (0.0015 + 5.556e-08 * (g + s))
    np.testing.assert_allclose(columns[moving, 0], [-exact, exact, -exact], rtol=1e-6)
    still = [i for i in range(len(names)) if i not in moving]
    assert np.max(np.abs(columns[still, 0])) < 1e-6 * exact


# A is made at k0·G and removed at kd: A* = k0·G/kd = 1e-7 beside G at 1.
OPEN_POOL = """begin parameters
    1 k0  {k0}
    2 kd  5e-3
    3 sg  1.0
    4 dg  1.0
end parameters
begin species
    1 A() 0
    2 G() 1.0
end species
begin reactions
    1 2 1,2 k0
    2 1 0 kd
    3 0 2 sg
    4 2 0 dg
end reactions
"""


@pytest.mark.parametrize(
    "k0, options",
    [
        ("5e-10", {}),
        ("5e-10", {"atol": 1e-5}),
        ("5e-13", {}),
        ("5e-16", {}),
        ("5e-10", {"tol": 1e-15}),
    ],
)
def test_a_small_species_beside_a_large_one_is_solved_to_its_own_steady_state(
    tmp_path, k0, options
):
    """A, at 1e-7 to 1e-13 of the G that makes it, is returned where it
    starts, at 3e-11: the residual is under ``tol`` from the first step. Its
    entries were small against G's 1, and against the solve's absolute
    tolerance where A is under that, and dA*/dkd came back -5.8e-9 for -2e-5.
    A has a steady value, k0·G/kd, and its entries are taken over that: the
    column moves, and the state is stepped to the root. dA*/dkd = -k0·G/kd²,
    and nothing for G."""
    sim = bngsim.Simulator(_net(tmp_path, OPEN_POOL.format(k0=k0)), method="ode")
    out = sim.steady_state(sensitivity_params=["kd"], **options)
    steady = float(k0) / 5e-3
    np.testing.assert_allclose(np.asarray(out.concentrations), [steady, 1.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [-steady / 5e-3, 0.0], rtol=1e-6)
    np.testing.assert_allclose(out.sens_species_scale, [steady, 1.0], rtol=1e-6)
    assert (out.sens_root_newton_steps == 0) == ("tol" in options)


# 0 -> S -> 0 started at 1: S* = k0/kd.
FROM_ABOVE = """begin parameters
    1 k0  {k0}
    2 kd  5e-3
end parameters
begin species
    1 S() 1.0
end species
begin reactions
    1 0 1 k0
    2 1 0 kd
end reactions
"""


# The same with S removed in pairs as well, at k2·S².
FROM_ABOVE_IN_PAIRS = FROM_ABOVE.replace(
    "    2 kd  5e-3\n", "    2 kd  5e-3\n    3 k2  3.5\n"
).replace("    2 1 0 kd\n", "    2 1 0 kd\n    3 1,1 0 k2\n")


@pytest.mark.parametrize("k0", ["5e-9", "5e-14"])
def test_a_species_stopped_above_a_steady_value_it_has_is_stepped_down_to_it(tmp_path, k0):
    """S decays from 1 towards k0/kd, 1e-6 or 1e-11, and the solve stops when
    the residual is under ``tol``: at 1.1e-6, and at 1.2e-7. dS*/dkd = -S/kd
    came back 12% off, and four orders off, against the 1 that S started at.
    S is not running out: with S at nothing its rate is k0. Its entries are
    taken over itself, and the state is stepped to k0/kd."""
    sim = bngsim.Simulator(_net(tmp_path, FROM_ABOVE.format(k0=k0)), method="ode")
    out = sim.steady_state(sensitivity_params=["kd"])
    steady = float(k0) / 5e-3
    assert np.asarray(out.concentrations)[0] == pytest.approx(steady, rel=1e-8)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(-steady / 5e-3, rel=1e-8)
    assert out.sens_species_scale[0] == pytest.approx(steady, rel=1e-8)
    assert out.sens_root_newton_steps == 2


FROM_ABOVE_WITH_AN_OUTPUT = (
    FROM_ABOVE.replace(
        "begin species", "begin functions\n    1 sq() Sobs^2\nend functions\nbegin species"
    )
    + "begin groups\n    1 Sobs 1\nend groups\n"
)


def test_the_outputs_are_those_of_the_state_that_was_stepped_to(tmp_path):
    """sq = S² beside the model above: d(sq)/dkd = 2·S·dS*/dkd, which is
    -4e-10 at S* = 1e-6, and 10% more at the 1.1e-6 where the solver
    stopped."""
    sim = bngsim.Simulator(
        _net(tmp_path, FROM_ABOVE_WITH_AN_OUTPUT.format(k0="5e-9")), method="ode"
    )
    out = sim.steady_state(sensitivity_params=["kd"])
    assert out.sens_root_newton_steps == 2
    assert list(out.expression_names) == ["sq"]
    assert np.asarray(out.sensitivities_expressions)[0, 0] == pytest.approx(-4e-10, rel=1e-6)
    assert np.asarray(out.output_sensitivities(["expression:sq"]))[0, 0] == pytest.approx(
        -4e-10, rel=1e-6
    )


def test_the_same_removed_in_pairs_too_takes_a_step_more(tmp_path):
    """With S + S -> 0 at 3.5 beside it, one Newton step from 1.2e-7 lands at
    twice the steady value and the next on it: a species that one step takes
    to a thousandth of where it was, and the next halves, is still not one
    that is running out. S* = k0/kd to one part in 1e8, the pairs taking
    k2·S*² of 5e-14."""
    sim = bngsim.Simulator(_net(tmp_path, FROM_ABOVE_IN_PAIRS.format(k0="5e-14")), method="ode")
    out = sim.steady_state(sensitivity_params=["kd"])
    assert np.asarray(out.concentrations)[0] == pytest.approx(1e-11, rel=1e-6)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(-2e-9, rel=1e-6)
    assert out.sens_root_newton_steps >= 3
    # The eigenvalue, -kd - 4·k2·S, is that of the state that is returned: at
    # the 1.2e-7 where the solver stopped it is 3.4e-4 further from -kd.
    stepped_to = float(np.asarray(out.concentrations)[0])
    assert out.sens_root_growth_rate == pytest.approx(-5e-3 - 4 * 3.5 * stepped_to, rel=1e-7)


# E + S <-> ES -> E + P at a millionth, with nothing left out.
USED_UP = """begin parameters
    1 kf  1e6
    2 kr  1.0
    3 kc  1.0
    4 E0  1e-6
    5 S0  2e-6
end parameters
begin species
    1 E() E0
    2 S() S0
    3 ES() 0
    4 P() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 3 1,4 kc
end reactions
"""


def test_species_that_ran_out_are_at_a_zero_where_the_run_ends_too(tmp_path):
    """S is used up, and the solver stops with S and ES at -3e-9 and -2e-9, a
    thousandth of what they were. The steps take them to nothing. The run that
    is taken on for 200 ends with them at what it has left of that, 1e-17,
    which is not rounding beside the 2e-6 they share with P: taken over that
    they would have moved by all of themselves. The two states are asked
    either way round what a species is taken over, and one that is at a zero
    in either is at a zero. dP*/dS0 = 1 and dE*/dE0 = 1, and nothing else
    moves anything."""
    sim = bngsim.Simulator(_net(tmp_path, USED_UP), method="ode")
    for max_time in (200.0, 1e6):
        out = sim.steady_state(sensitivity_params=["kf", "kc", "S0", "E0"], max_time=max_time)
        assert out.sens_root_newton_steps >= 2
        np.testing.assert_allclose(
            out.concentrations, [1e-6, 0.0, 0.0, 2e-6], rtol=1e-12, atol=1e-20
        )
        expected = np.zeros((4, 4))
        expected[0, 3] = expected[3, 2] = 1.0
        np.testing.assert_allclose(out.sensitivity, expected, rtol=1e-12, atol=1e-12)
        assert out.sens_root_hold_drift < 1e-3


# B is made at s·sqrt(1 - A), beside an A that nothing moves, a billionth under 1.
NO_VALUE_A_MILLIONTH_ON = """begin parameters
    1 s   1.0
    2 d   1.0
    3 A0  0.999999999
end parameters
begin functions
    1 made() s*sqrt(1-Aobs)
end functions
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 0 2 made
    2 2 0 d
end reactions
begin groups
    1 Aobs 1
end groups
"""


def test_a_run_that_cannot_be_taken_on_is_refused(tmp_path):
    """The state is a root, B* = s·sqrt(1 - A)/d, and its column is right. The
    run that is to show the system stays there starts a millionth beside the
    state, where A is above 1 and the rate is not a number, and fails. The
    columns are refused for a state that is not known to be one a run stays
    at: a failed run is not a run that stayed."""
    sim = bngsim.Simulator(_net(tmp_path, NO_VALUE_A_MILLIONTH_ON), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*not one a run stays at.*failed, so that"
    ):
        sim.steady_state(sensitivity_params=["s"])


# N' = r·N·S·(1 - N/K): N makes itself on S, from the 1e-8 it starts with.
SEEDED = """begin parameters
    1 r   1e-3
    2 K   1e-6
end parameters
begin functions
    1 crowd() r*Sobs*Nobs/K
end functions
begin species
    1 N() 1e-8
    2 S() 1.0
end species
begin reactions
    1 1,2 1,1,2 r
    2 1 0 crowd
end reactions
begin groups
    1 Nobs 1
    2 Sobs 2
end groups
"""


def test_a_species_on_its_way_up_from_a_seed_is_not_stepped_down_to_nothing(tmp_path):
    """N starts at a hundredth of what S carries and grows towards it at 1e-3,
    with a residual under ``tol``: the solve returns the start. Newton steps
    from there to the root beside it, N = 0, which the system leaves, and
    there N is at rounding beside S: it was taken for a species at a zero, its
    entries over S's 1, and the state came back stepped to -1e-14 with dN*/dK
    = -1e-16, where a run ends at K and dN*/dK = 1. The eigenvalue of 1e-3
    says the system does not rest where the solve stopped."""
    sim = bngsim.Simulator(_net(tmp_path, SEEDED), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 0\.001 "
    ):
        sim.steady_state(sensitivity_params=["K", "r"])


def test_the_same_among_too_many_species_for_the_spectrum_is_refused_too(tmp_path):
    """Among 520 other species the eigenvalues of the whole system are not
    taken, and the state is stepped to N = 0. The run that is taken on from
    the seed does not show that the system leaves it: its error test, over
    522 species, is met by a step that is long for r·S, and an implicit step
    that long lands on N = 0 and stays, so that dN*/dK = 0 came back for 1.
    The species the returned state has at a zero are asked by the eigenvalues
    of their own block, which is N's own r·S."""
    sim = bngsim.Simulator(_net(tmp_path, _among_many(SEEDED)), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 0\.001 "
    ):
        sim.steady_state(sensitivity_params=["K", "r"])


# X' = v·X²/(K² + X²) - d·X: a species that activates its own making. It rests
# at nothing, and at b = (v/d + sqrt((v/d)² - 4·K²))/2 = 9.9e-10, with a root
# it leaves between the two.
ACTIVATES_ITSELF = """begin parameters
    1 v   5e-12
    2 K   1e-10
    3 d   5e-3
end parameters
begin functions
    1 made() v*Xobs/(K^2+Xobs^2)
end functions
begin species
    1 X() 1.0
end species
begin reactions
    1 1 1,1 made
    2 1 0 d
end reactions
begin groups
    1 Xobs 1
end groups
"""


def _self_activation() -> tuple[float, float, float]:
    """(b, dX*/dd, dX*/dK) at the upper root, by the implicit function
    theorem on f = v·X²/(K² + X²) - d·X."""
    v, k, d = 5e-12, 1e-10, 5e-3
    b = (v / d + math.sqrt((v / d) ** 2 - 4 * k * k)) / 2
    f_x = 2 * v * b * k * k / (k * k + b * b) ** 2 - d
    f_k = -2 * v * b * b * k / (k * k + b * b) ** 2
    return b, b / f_x, -f_k / f_x


@pytest.mark.parametrize("beside", [False, True])
def test_a_species_that_could_rest_at_nothing_and_does_not_is_stepped_to_its_root(
    tmp_path, beside
):
    """From 1, X falls to its upper root, 9.9e-10, and the solve stops on the
    way at 1.3e-7. Everything said X was running out: a Newton step takes most
    of it, nothing makes it with X at nothing, and from next to nothing it
    does not grow, nothing being a state it rests at too. Its entries were
    taken over the 1 it started at, and dX*/dd came back 124 times -2.02e-7,
    318 times beside a species turned over at 200. Its column moves by nearly
    all of itself all the same, which is asked of every column now, and the
    state is stepped to the root: the upper one, where the run that is taken
    on ends as well."""
    text = ACTIVATES_ITSELF
    if beside:
        text = (
            text.replace("    1 X() 1.0\n", "    1 X() 1e-4\n    2 Z() 1.0\n")
            .replace("    3 d   5e-3\n", "    3 d   5e-3\n    4 kz  200.0\n")
            .replace("    2 1 0 d\n", "    2 1 0 d\n    3 0 2 kz\n    4 2 0 kz\n")
        )
    root, by_d, by_k = _self_activation()
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    out = sim.steady_state(sensitivity_params=["d", "K"])
    assert out.sens_root_newton_steps >= 2
    assert np.asarray(out.concentrations)[0] == pytest.approx(root, rel=1e-6)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[0], [by_d, by_k], rtol=1e-6)
    assert out.sens_species_scale[0] == pytest.approx(root, rel=1e-3)
    assert out.sens_root_hold_drift < 1e-2


def test_the_same_solved_to_the_root_it_rests_at_is_returned(tmp_path):
    """Control. With ``tol=1e-15`` the solve stops on the upper root."""
    root, by_d, by_k = _self_activation()
    sim = bngsim.Simulator(_net(tmp_path, ACTIVATES_ITSELF), method="ode")
    out = sim.steady_state(sensitivity_params=["d", "K"], tol=1e-15)
    assert np.asarray(out.concentrations)[0] == pytest.approx(root, rel=1e-4)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[0], [by_d, by_k], rtol=1e-4)


def test_the_same_beside_a_species_with_the_larger_entry_is_refused_by_the_run(tmp_path):
    """W, made at s and removed at the same d, is at 200, and the column of d
    is asked alone. Its largest entry is W's, which no step moves. X's own
    entry, over the 1 that X started at, is a millionth of that, so that no
    step is taken and the state is the solver's, with X at 1.3e-7: dX*/dd
    came back 124 times what it is. The run that is taken on ends with X at
    9.9e-10 and at rest. X is then not at a zero: it has moved by all of
    itself, and its entry with it."""
    text = (
        ACTIVATES_ITSELF.replace("    3 d   5e-3\n", "    3 d   5e-3\n    4 s   1.0\n")
        .replace("    1 X() 1.0\n", "    1 X() 1.0\n    2 W() 200.0\n")
        .replace("    2 1 0 d\n", "    2 1 0 d\n    3 0 2 s\n    4 2 0 d\n")
    )
    assert text.count("\n") == ACTIVATES_ITSELF.count("\n") + 4
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*not one a run stays at.*moves X\(\) by 1\d\d% of what it is taken over",
    ):
        sim.steady_state(sensitivity_params=["d"])


def test_a_parameter_that_hardly_matters_does_not_hide_a_state_a_run_leaves(tmp_path):
    """d = d0 + eps·q with eps = 1e-12, and the column of q asked alone: it is
    eps times that of d. At the state the solver stopped at, with X at 1.3e-7
    and taken over the 1 it started at, the column is under what rounding
    makes of one, and no step is taken. The run that is taken on ends with X
    at 9.9e-10. Where a run moved a species, a column is asked how far it
    moved of itself, however small it is, and this one moved by 99%. (At
    eps = 1e-8 the column is something, the steps are taken, and
    dX*/dq = eps·dX*/dd is returned.)"""
    weak = ACTIVATES_ITSELF.replace(
        "    3 d   5e-3\n", "    3 d0  5e-3\n    4 eps {eps}\n    5 q   1.0\n    6 d   d0+eps*q\n"
    )
    assert weak != ACTIVATES_ITSELF
    root, by_d, _ = _self_activation()
    sim = bngsim.Simulator(_net(tmp_path, weak.format(eps="1e-8")), method="ode")
    out = sim.steady_state(sensitivity_params=["q"])
    assert np.asarray(out.concentrations)[0] == pytest.approx(root, rel=1e-5)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(1e-8 * by_d, rel=1e-5)
    sim = bngsim.Simulator(_net(tmp_path, weak.format(eps="1e-12"), "weaker.net"), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*not one a run stays at.*moves X\(\) by 99% .*the column of q",
    ):
        sim.steady_state(sensitivity_params=["q"])


# X' = s - k·(X - c)²: X rests at c + sqrt(s/k), 1e-4 above a point where the
# rate has a double root.
BESIDE_A_DOUBLE_ROOT = """begin parameters
    1 s   1e-8
    2 k   1.0
    3 c   1.0
end parameters
begin functions
    1 net() s-k*(Xobs-c)^2
end functions
begin species
    1 X() 1.001
end species
begin reactions
    1 0 1 net
end reactions
begin groups
    1 Xobs 1
end groups
"""


def test_a_column_that_moves_where_the_state_hardly_does_has_the_state_stepped(tmp_path):
    """dX*/ds = 1/(2·k·(X - c)) = 5,000. The solve stops 4e-6 above the root,
    which is 4% of its distance from c, and the column came back 4% off, at
    4,799. For a parameter of 1e-8 that is a column of 5e-5 against 1/|s|,
    which was taken for nothing and not asked how far it moved; and the state
    itself moves by 4e-6. Every column is asked, against the largest it has
    been."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_A_DOUBLE_ROOT), method="ode")
    out = sim.steady_state(sensitivity_params=["s"])
    assert out.sens_root_newton_steps == 2
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(5000.0, rel=1e-5)
    assert np.asarray(out.concentrations)[0] == pytest.approx(1.0001, rel=1e-10)


# 0 -> S -> 0 from 1, removed at kd + eps·q: q is a thousandth of a millionth
# of the rate.
A_PARAMETER_THAT_HARDLY_MATTERS = """begin parameters
    1 k0  5e-14
    2 kd  5e-3
    3 eps 1e-8
    4 q   1.0
    5 kq  kd+eps*q
end parameters
begin species
    1 S() 1.0
end species
begin reactions
    1 0 1 k0
    2 1 0 kq
end reactions
"""


@pytest.mark.parametrize("eps", [1e-8, 1e-20])
def test_a_state_short_of_its_root_is_stepped_whatever_is_asked(tmp_path, eps):
    """The solve stops with S at 1.2e-7, for a steady 1e-11. Asked with kd,
    the column of kd moves and the state is stepped. Asked alone, the column
    of q is eps·S/kd: 2e-6 of S over q at eps = 1e-8, where dS*/dq came back
    -2.5e-13 for -2e-17 at the state as the solver left it, and 2e-18 of it at
    1e-20, which is rounding beside 1/|q| and is not asked how far it moved.
    The state is asked as well as the columns, and S moves by all of itself.
    dS*/dq = -k0·eps/kd²."""
    text = A_PARAMETER_THAT_HARDLY_MATTERS.replace("    3 eps 1e-8\n", f"    3 eps {eps:g}\n")
    assert text != A_PARAMETER_THAT_HARDLY_MATTERS or eps == 1e-8
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    out = sim.steady_state(sensitivity_params=["q"])
    assert out.sens_root_newton_steps == 2
    assert np.asarray(out.concentrations)[0] == pytest.approx(1e-11, rel=1e-5)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(-2e-9 * eps, rel=1e-5)
    assert out.sens_root_state_shift < 1e-6 and out.sens_root_state_species == "S()"


# E + S <-> ES -> E + P, at a millionth: with P masked out, the solve stops
# with most of S still there.
A_SINK_THAT_IS_MASKED = """begin parameters
    1 kf  1e6
    2 kr  1.0
    3 kc  1.0
    4 E0  1e-6
    5 S0  2e-6
end parameters
begin species
    1 S()  S0
    2 E()  E0
    3 ES() 0
    4 P()  0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 3 2,4 kc
end reactions
"""


def test_a_masked_sink_is_given_what_the_stepped_species_lost(tmp_path):
    """The state is stepped to S = ES = 0, on the equations of the species
    the mask kept. P, which the mask left out, was not stepped, and came back
    as the solver left it, at 1e-14: a state with 1e-14 of a substrate of
    2e-6 in it. A law of the model holds P with S and ES, and P is what it
    leaves. dE*/dE0 = 1, and nothing else moves."""
    model = _net(tmp_path, A_SINK_THAT_IS_MASKED)
    sim = bngsim.Simulator(model, method="ode")
    out = sim.steady_state(sensitivity_params=["kc", "E0"], mask=~np.asarray(model.is_pure_sink()))
    assert out.sens_root_newton_steps >= 1
    state = np.asarray(out.concentrations)
    assert state[[0, 2, 3]].sum() == pytest.approx(2e-6, rel=1e-12)
    np.testing.assert_allclose(state, [0.0, 1e-6, 0.0, 2e-6], rtol=1e-9, atol=1e-20)
    columns = np.asarray(out.sensitivity)
    np.testing.assert_allclose(columns[:3], [[0, 0], [0, 1], [0, 0]], atol=1e-9)
    assert np.all(np.isnan(columns[3]))


# X <-> Y, and beside it A + A -> D, which has a root of second order.
BESIDE_A_PART_WITH_NO_ROOT = """begin parameters
    1 kf  1.0
    2 kr  2.0
    3 ka  1.0
end parameters
begin species
    1 X() 2.0
    2 Y() 0
    3 A() 1.0
    4 D() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
    3 3,3 4 ka
end reactions
"""


def test_the_refusal_of_one_part_says_how_to_ask_for_another(tmp_path):
    """Every column is refused where any part of the model has no isolated
    root, the columns of X <-> Y among them, which are -T·kr/(kf + kr)² and
    T·kf/(kf + kr)² and came back right. The refusal said to mask the pure
    sinks, which refuses again, A being no sink. It says now that a mask on
    the species of the part that is wanted does it."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_A_PART_WITH_NO_ROOT), method="ode")
    with pytest.raises(bngsim.SimulationError) as caught:
        sim.steady_state(sensitivity_params=["kf", "kr"])
    message = str(caught.value)
    assert "#995" in message and "a determinant that is 0.5 of the one" in message
    assert "mask= with the species of that other part" in message
    assert "difference steady states" not in message
    out = sim.steady_state(sensitivity_params=["kf", "kr"], mask=["X()", "Y()"])
    np.testing.assert_allclose(
        np.asarray(out.sensitivity)[:2], [[-4 / 9, 2 / 9], [4 / 9, -2 / 9]], rtol=1e-8
    )


# N' = r·N·S·(1 - (N/K)^0.2): N makes itself, on S, and is crowded out above K.
ABOVE_ITS_CAPACITY = """begin parameters
    1 r   1e-7
    2 K   1e-6
end parameters
begin functions
    1 crowding() r*Sobs*(Nobs/K)^0.2
end functions
begin species
    1 N() 1e-4
    2 S() 1.0
end species
begin reactions
    1 1,2 1,1,2 r
    2 1 0 crowding
end reactions
begin groups
    1 Nobs 1
    2 Sobs 2
end groups
"""


def test_a_species_that_grows_from_nothing_is_not_running_out(tmp_path):
    """N starts at a hundred times what S carries, K, and falls towards K. The
    solve stops on the way, under ``tol``, and a Newton step from there takes
    N a sixth of the way down: more than halved, with nothing at N = 0 to make
    it, as a species that is running out is. But N makes itself, and from next
    to nothing it grows: it does not rest at zero. Taken for a species at a
    zero its entries were small against S's 1, and dN*/dK came back 2.78 for
    1, at 7.4e-6 for K. Its entries are taken over itself, and the state is
    stepped to K."""
    sim = bngsim.Simulator(_net(tmp_path, ABOVE_ITS_CAPACITY), method="ode")
    out = sim.steady_state(sensitivity_params=["K"], max_time=1e9)
    assert np.asarray(out.concentrations)[0] == pytest.approx(1e-6, rel=1e-2)
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(1.0, rel=1e-3)
    assert out.sens_species_scale[0] == pytest.approx(1e-6, rel=0.1)
    assert out.sens_root_newton_steps >= 2


def test_the_same_in_the_time_it_does_not_get_there_in_is_refused(tmp_path):
    """N falls at r·S = 1e-7, and the default ``max_time`` of 1e6 is a tenth
    of the time that takes: the run taken on from where the solver stopped is
    still on its way."""
    sim = bngsim.Simulator(_net(tmp_path, ABOVE_ITS_CAPACITY), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*not one a run stays at.*the run moves N\(\) by"
    ):
        sim.steady_state(sensitivity_params=["K"])


# X is held at 1 by 0 -> X -> 0, and Y is made at c·sqrt(1.0000001 - X): a
# rate that has no value a tenth of a millionth above where X rests.
NO_VALUE_BESIDE = """begin parameters
    1 k0  1.0
    2 kd  1.0
    3 c   1.0
    4 ky  1.0
end parameters
begin functions
    1 made() c*sqrt(1.0000001-Xobs)
end functions
begin species
    1 X() 1.0
    2 Y() 0
end species
begin reactions
    1 0 1 k0
    2 1 0 kd
    3 0 2 made
    4 2 0 ky
end reactions
begin groups
    1 Xobs 1
end groups
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_run_that_cannot_be_taken_on_is_not_known_to_stay(tmp_path, method):
    """The run is taken on from the returned state with X moved up by a
    quarter of a millionth, where the rate that makes Y is the square root of
    a negative number. The integrator stops at once, on a differenced
    Jacobian as on the closed form. A state a run was not seen to stay at is
    refused, here with columns that are right."""
    sim = bngsim.Simulator(_net(tmp_path, NO_VALUE_BESIDE), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*not one a run stays at.*max_time \(1e\+06\) failed.*not known to stay",
    ):
        sim.steady_state(sensitivity_params=["c", "ky"], method=method)


BOUNDARY = """begin parameters
    1 kf  1.0
    2 kr  0.5
end parameters
begin species
    1 A() 2.0
    2 B() 0
    3 $E() 3.0
end species
begin reactions
    1 1,3 2,3 kf
    2 2 1 kr
end reactions
"""


def test_a_boundary_species_the_mask_leaves_out_is_the_constant_it_is_held_as(tmp_path):
    """Control. A + E -> B + E with E a boundary species, and E masked out.
    A's rate reads E, and E is what a masked species is held as: a constant.
    With T = A + B, dA*/dkf = -T·kr·E/(kf·E + kr)² and dA*/dkr =
    T·kf·E/(kf·E + kr)²."""
    sim = bngsim.Simulator(_net(tmp_path, BOUNDARY), method="ode")
    out = sim.steady_state(sensitivity_params=["kf", "kr"], mask=["A()", "B()"], tol=1e-12)
    np.testing.assert_allclose(
        np.asarray(out.sensitivity)[:2],
        [[-3.0 / 12.25, 6.0 / 12.25], [3.0 / 12.25, -6.0 / 12.25]],
        rtol=1e-8,
    )
    assert np.all(np.isnan(np.asarray(out.sensitivity)[2]))


# ── The corpus witnesses ────────────────────────────────────────────────────
#
# BioModels files that no small model stood in for. The corpus is fetched,
# not vendored, and these skip where it is absent.


def _biomodel(name: str) -> str:
    fetched = (
        Path(__file__).resolve().parents[2] / "benchmarks/suites/biomodels/data/sbml_downloads"
    )
    path = Path(os.environ.get("BIOMODELS_SBML_DIR", fetched)) / f"{name}.xml"
    if not path.exists():
        pytest.skip(f"{name} not available")
    return str(path)


def test_biomd599_is_refused_on_its_condition_number():
    """A group of species that exchange among themselves, with what fed them
    gone: their total is where the run left it, and the Jacobian has an
    eigenvalue of 2e-18. No pivot of its factorization is under 4% of the
    terms it was computed from, and one Newton step leaves the determinant and
    the columns where they were. The condition number is 3e16. Every small
    model built to be refused on it was refused on a pivot first."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000000599")), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*magnified \d\.\de\+1[5-9] times"):
        sim.steady_state(sensitivity_params=["parameter_1", "parameter_2", "parameter_3"])


def test_biomd1000_is_taken_on_with_a_differenced_jacobian():
    """The run taken on from the returned state stops the integrator on the
    closed-form Jacobian, started again or not, and reaches ``max_time`` on a
    differenced one, as the solve itself does where its integrator gives up
    (issue #127). The state is stepped on until a cascade of species that is
    running out (pS2_n, from 9e-6) is at rounding, and the run ends with them
    at nothing."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000001000")), method="ode")
    out = sim.steady_state(
        sensitivity_params=["R1_total_C3", "index_k_out_1_relative_speed_C3", "k_in_R1_C3"]
    )
    assert np.all(np.isfinite(np.asarray(out.sensitivity)))
    assert out.sens_root_hold_time >= 1e6 and out.sens_root_newton_steps >= 2
    assert out.sens_root_hold_drift < 1e-3


def test_biomd1001_species_that_ran_out_are_taken_over_what_stands_beside_them():
    """Ten of its species have run out, at 1e-8 to 1e-21 beside others at
    1,000, and nothing that is left makes them: their entries are taken over
    what stands beside them, and the columns are returned, as they were."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000001001")), method="ode")
    out = sim.steady_state(sensitivity_params=["R1_total_C4", "k_in_R1_C4", "kdeg_R1"])
    assert np.all(np.isfinite(np.asarray(out.sensitivity)))
    assert out.sens_root_hold_time >= 1e6
    scale = dict(zip(out.species_names, out.sens_species_scale, strict=True))
    assert scale["TGFb_In"] > 1.0 and scale["pS2_c"] > 1.0


def test_biomd92_species_that_ran_out_are_at_a_zero_where_the_run_ends():
    """e + z <-> ez -> e + w has used z up. The steps take z and ez to
    nothing, and the run that is taken on ends with them at -5e-14 and 4e-13:
    what its tolerance left, of either sign and in no proportion, and 2e-8 of
    the total they share with w. Asked whether one grows from next to nothing
    in the proportions they have there, ez does, towards what z would keep of
    it, and it was taken for a species with a value of its own that had moved
    by all of itself. That is asked of a species that came down to something,
    and these came down to nothing."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000000092")), method="ode")
    out = sim.steady_state(
        sensitivity_params=["_lp_v1_k1", "_lp_v2_k21", "_lp_v2_k22", "_lp_v3_k3"]
    )
    assert out.sens_root_newton_steps >= 2
    at = dict(zip(out.species_names, np.asarray(out.concentrations), strict=True))
    assert abs(at["z"]) < 1e-20 and abs(at["ez"]) < 1e-20
    assert at["e"] == pytest.approx(2.4e-5, rel=1e-9)
    assert out.sens_root_hold_drift < 1e-3


def test_biomd416_a_column_of_nothing_that_never_settles_is_returned_as_that():
    """The steady state does not move with ``etaAuxTIR1``, and its column is
    what rounding makes of that at each state: 1e-12, of either sign, and all
    of itself different at the next. No step settles it, the ten are taken,
    and the last state is returned with the column it has there: it is under
    1e-5 of 1/|p|, which is nothing, and is not refused for moving."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000000416")), method="ode")
    out = sim.steady_state(sensitivity_params=["etaAuxTIR1"])
    assert out.sens_root_newton_steps == 10
    assert np.max(np.abs(np.asarray(out.sensitivity))) < 1e-8
    assert out.sens_root_column_shift < 1e-6 and out.sens_root_state_shift < 1e-9


def test_biomd5_by_newton_is_a_root_inside_its_limit_cycle():
    """Tyson's cell cycle oscillates, and Newton finds the fixed point the
    cycle goes round, with eigenvalues 0.16 ± 0.18i beside one of -1e6: 1.6e-7
    of the largest, which was under what they were asked against, so that the
    point was taken for one the system rests at."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000000005")), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 0\.163 "
    ):
        sim.steady_state(
            sensitivity_params=["_lp_Reaction1_k6", "_lp_Reaction4_k3", "_lp_Reaction9_k4"],
            method="newton",
        )


def test_model2502210001_ends_a_species_below_zero_where_a_rate_has_no_value():
    """Control. With its four sinks masked out, the run taken on from the
    returned state ends with a species its last step took below zero, where a
    rate has no value and the Jacobian is not a number. The columns are solved
    there with that species at zero, and returned, as they were."""
    model = bngsim.Model.from_sbml(_biomodel("MODEL2502210001"))
    sim = bngsim.Simulator(model, method="ode")
    kept = ~np.asarray(model.is_pure_sink())
    out = sim.steady_state(
        sensitivity_params=["_lp_r1_0_k1", "_lp_r2_0_Km", "_lp_r2_0_V"], mask=kept
    )
    assert np.all(np.isfinite(np.asarray(out.sensitivity)[kept]))


# 0 -> A at s, A <-> B at F each way, B -> 0 at k.
BESIDE_A_FAST_EXCHANGE = """begin parameters
    1 s   1.0
    2 F   {F}
    3 k   1.0
end parameters
begin species
    1 A() 0
    2 B() 0
end species
begin reactions
    1 0 1 s
    2 1 2 F
    3 2 1 F
    4 2 0 k
end reactions
"""


@pytest.mark.parametrize("fast", [1e10, 1e12])
def test_a_step_beside_a_fast_exchange_is_returned(tmp_path, fast):
    """An isolated root, B* = s/k and A* = s/k + s/F, with a pivot for B of
    -k that is computed from terms of F: k/(2·F) of them, 5e-11 and 5e-13.
    Under 1e-10 it was refused, and called singular whatever the state, where
    main had the columns to twelve digits. Rounding leaves 1e-16 of the terms,
    which at 5e-13 is still a pivot known to a five-thousandth."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_A_FAST_EXCHANGE.format(F=fast)), method="ode")
    out = sim.steady_state(sensitivity_params=["s", "k"])
    assert out.sens_root_pivot_share == pytest.approx(0.5 / fast, rel=1e-3)
    assert out.sens_root_condition == pytest.approx(4 * fast, rel=0.5)
    np.testing.assert_allclose(out.concentrations, [1 + 1 / fast, 1.0], rtol=1e-9)
    np.testing.assert_allclose(out.sensitivity, [[1 + 1 / fast, -1.0], [1.0, -1.0]], rtol=1e-3)


@pytest.mark.parametrize(
    ("fast", "said"),
    [
        (3e12, r"magnified 1\.2e\+13 times.*or two rates of the model are more than 1e12 apart"),
        (1e14, r"The pivot for B\(\) is 5\.0e-15 of the terms.*or two rates of the model are"),
    ],
)
def test_rates_too_far_apart_for_one_percent_are_refused_as_that(tmp_path, fast, said):
    """The same with the exchange 3e12 and 1e14 times the step, a condition
    number of 1.2e13 and a pivot share of 5e-15: the columns are not known to
    1%, and the refusal says that it is this or a Jacobian that is singular
    whatever the state, and not that it is the second."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_A_FAST_EXCHANGE.format(F=fast)), method="ode")
    with pytest.raises(bngsim.SimulationError, match=rf"#995.*cannot be computed.*{said}"):
        sim.steady_state(sensitivity_params=["s", "k"])


def test_a_condition_number_above_the_limit_is_refused_and_one_under_it_is_not(tmp_path):
    """The limit on the condition number, on a result that has everything
    else in order: the corpus models that only it refuses are not in the
    repository (BIOMD0000000599 below)."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_A_FAST_EXCHANGE.format(F=10.0)), method="ode")
    out = sim.steady_state(sensitivity_params=["s", "k"])
    out.sens_root_condition = 0.9e13
    sim._raise_if_not_an_isolated_root(out, 1e6, False)
    out.sens_root_condition = 1.1e13
    with pytest.raises(bngsim.SimulationError, match=r"#995.*magnified 1\.1e\+13 times"):
        sim._raise_if_not_an_isolated_root(out, 1e6, False)


# A resident at its capacity b/c, and an invader that makes itself at g and is
# removed by the resident and by itself at d: N' = N·(g - d·R - d·N).
BESIDE_AN_INVADER = """begin parameters
    1 b   1.0
    2 c   1.0
    3 g   2.0
    4 d   1.0
end parameters
begin species
    1 R() 0.5
    2 N() {n0}
end species
begin reactions
    1 1 1,1 b
    2 1,1 1 c
    3 2 2,2 g
    4 1,2 1 d
    5 2,2 2 d
end reactions
"""


def test_a_species_the_model_does_not_have_is_not_asked_what_its_arrival_would_do(tmp_path):
    """Control. With N at nothing the resident rests at b/c, and that is where
    every run of this model ends: dR*/db = 1/c, dR*/dc = -b/c², and nothing
    for g. The Jacobian has an eigenvalue of g - d·R = 1 there, which is what
    N would do if it arrived. A species that is absent and that nothing
    present makes is left out of the eigenvalues, as the run that is taken on
    leaves it where it is; with it in, this was refused as a state the system
    does not rest at (and 10 of 354 random networks with it)."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_AN_INVADER.format(n0="0")), method="ode")
    out = sim.steady_state(sensitivity_params=["b", "c", "g"])
    np.testing.assert_allclose(out.concentrations, [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out.sensitivity, [[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]], atol=1e-7)


def test_the_eigenvalues_are_those_of_the_species_the_model_has(tmp_path):
    """The same result's growth rate is the resident's own, -b, and not the
    invader's 1."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_AN_INVADER.format(n0="0")), method="ode")
    out = sim.steady_state(sensitivity_params=["b", "c", "g"])
    assert out.sens_root_growth_rate == pytest.approx(-1.0, rel=1e-6)


def test_an_invader_that_is_there_takes_the_system_where_it_goes(tmp_path):
    """Control. Seeded at 1e-9, N grows, and the solve ends where the two
    coexist: R* = b/c and N* = g/d - R*."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_AN_INVADER.format(n0="1e-9")), method="ode")
    out = sim.steady_state(sensitivity_params=["b", "g"])
    np.testing.assert_allclose(out.concentrations, [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(out.sensitivity, [[1.0, 0.0], [-1.0, 1.0]], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_that_is_not_there_yet_and_is_being_made_is_asked(tmp_path, method):
    """The resident, started at its capacity, makes the invader at 1e-12: N
    starts at nothing, the residual is under ``tol`` there, and the solve
    stops beside the start (on it, by Newton, with N at exactly nothing),
    which N leaves at g - d·R = 1 for where the two coexist. N is at nothing,
    and it is not absent: its row of the Jacobian has an entry in R's column.
    It is in the eigenvalues, and they say that the system does not rest
    here."""
    made = (
        BESIDE_AN_INVADER.format(n0="0")
        .replace("    4 d   1.0\n", "    4 d   1.0\n    5 eps 1e-12\n")
        .replace("    5 2,2 2 d\n", "    5 2,2 2 d\n    6 1 1,2 eps\n")
        .replace("    1 R() 0.5\n", "    1 R() 1.0\n")
    )
    assert made.count("eps") == 2 and "R() 1.0" in made
    sim = bngsim.Simulator(_net(tmp_path, made), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 1 "):
        sim.steady_state(sensitivity_params=["b", "c", "g"], method=method)


def test_biomd908_rests_where_it_is_without_the_species_it_does_not_start_with():
    """S is at nothing from the start and nothing makes it; the Jacobian has
    an eigenvalue of 0.277 that is S's own. The run that is taken on stays,
    the columns agree with differences of runs, and they were refused for a
    state the system does not rest at."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000000908")), method="ode")
    out = sim.steady_state(sensitivity_params=["d", "l", "s"])
    at = dict(zip(out.species_names, np.asarray(out.concentrations), strict=True))
    assert at["S"] == 0.0
    assert out.sens_root_growth_rate < 0.0
    assert np.all(np.isfinite(np.asarray(out.sensitivity)))


def test_model1607210000_says_that_the_run_stayed_and_the_columns_did_not():
    """A species whose turnover has stopped: the run that is taken on ends
    3e-6 from the state, and the column of v15_h solved again there is 40
    times what it was. The refusal said the state was not one a run stays at,
    and then that the run moved a species by 0%."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("MODEL1607210000")), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*though the run ends beside it.*column of v15_h.*moves by 40\.\d times its",
    ):
        sim.steady_state(sensitivity_params=["v15_h"])


# 0 -> X at s, X -> A at k1, X -> 0 at kx, A <-> B at 1.37·F and 0.73·F, B -> 0.
A_STEP_INTO_A_FAST_EXCHANGE = """begin parameters
    1 s  1.3
    2 k1 0.28
    3 kx 3.2
    4 F1 {forward!r}
    5 F2 {back!r}
    6 k  0.917
end parameters
begin species
    1 X() 0
    2 A() 0
    3 B() 0
end species
begin reactions
    1 0 1 s
    2 1 2 k1
    3 1 0 kx
    4 2 3 F1
    5 3 2 F2
    6 3 0 k
end reactions
"""


def _into_a_fast_exchange(tmp_path, fast, **simulator):
    """The model, its dY*/dkx in closed form over Y*, and the Simulator."""
    s, k1, kx, k = 1.3, 0.28, 3.2, 0.917
    forward, back = 1.37 * fast, 0.73 * fast
    x = s / (k1 + kx)
    b = k1 * x / k
    a = (k1 * x + back * b) / forward
    dx = -s / (k1 + kx) ** 2
    relative = np.array([dx / x, (k1 + back * k1 / k) / forward * dx / a, k1 / k * dx / b])
    text = A_STEP_INTO_A_FAST_EXCHANGE.format(forward=forward, back=back)
    model = _net(tmp_path, text, f"exchange_{fast:g}_{len(simulator)}.net")
    return bngsim.Simulator(model, method="ode", **simulator), relative, np.array([x, a, b])


@pytest.mark.parametrize("fast", [1e8, 1e9])
def test_a_differenced_jacobian_is_held_to_what_a_difference_knows(tmp_path, fast):
    """With ``jacobian="fd"`` the entry dA'/dX = 0.28 is under what a
    difference quotient resolves beside the fluxes of A's row, F·A, and reads
    as nothing: the columns are those of a model in which X does not make A,
    dB*/dkx = 0 for -0.033 at F = 1e9 and 20% off at 1e8, with every measure
    clean, the measures being taken on the same matrix. The pivot share, 6e-9
    and 6e-10, is far above what rounding leaves and under what a difference
    does, and that is the limit such a Jacobian is held to."""
    sim, _, _ = _into_a_fast_exchange(tmp_path, fast, jacobian="fd")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*cannot be computed.*the difference quotient this Jacobian is",
    ):
        sim.steady_state(sensitivity_params=["kx"], tol=1e-6)


@pytest.mark.parametrize("fast", [1e8, 1e9, 1e11])
def test_the_same_with_the_closed_form_jacobian_is_returned(tmp_path, fast):
    """Control. The closed-form Jacobian has the entry to rounding, and the
    columns are right to a ten-thousandth at every one of these."""
    sim, relative, _ = _into_a_fast_exchange(tmp_path, fast)
    out = sim.steady_state(sensitivity_params=["kx"], tol=1e-6)
    assert out.sens_jacobian_source != "finite-difference"
    got = np.asarray(out.sensitivity)[:, 0] / np.asarray(out.concentrations)
    np.testing.assert_allclose(got, relative, rtol=1e-4)


def test_a_differenced_jacobian_of_rates_that_are_not_far_apart_is_returned(tmp_path):
    """Control. At F = 1e3 the difference quotient has every entry, and the
    columns are right."""
    sim, relative, _ = _into_a_fast_exchange(tmp_path, 1e3, jacobian="fd")
    out = sim.steady_state(sensitivity_params=["kx"], tol=1e-9)
    got = np.asarray(out.sensitivity)[:, 0] / np.asarray(out.concentrations)
    np.testing.assert_allclose(got, relative, rtol=1e-4)


# N' = eps + g·N - d·N², from nothing.
MADE_FROM_NOTHING = """begin parameters
    1 g   1.0
    2 d   1.0
    3 eps 1e-12
end parameters
begin species
    1 N() 0
end species
begin reactions
    1 1 1,1 g
    2 1,1 1 d
    3 0 1 eps
end reactions
"""

# D + N -> 2 N at g, N -> D at kr, D -> N at eps: D + N is conserved, and a run
# from D = 1 ends at N = 1 - kr/g.
MADE_BY_WHAT_A_LAW_GIVES = """begin parameters
    1 g   2.0
    2 kr  1.0
    3 eps 1e-12
end parameters
begin species
    1 D() 1.0
    2 N() 0
end species
begin reactions
    1 1,2 2,2 g
    2 2 1 kr
    3 1 2 eps
end reactions
"""


@pytest.mark.parametrize("text", [MADE_FROM_NOTHING, MADE_BY_WHAT_A_LAW_GIVES], ids=["0", "law"])
def test_a_species_at_nothing_that_is_being_made_is_not_absent(tmp_path, text):
    """N is at exactly nothing where a Newton solve stops, the residual being
    1e-12, and it is made: from nothing, or by D, which the conservation law
    gives and which has no column of its own among the unknowns. Its row has
    no entry in a present species' column either way, and it was taken for a
    species the model does not have: N came back at -1e-12 with dN*/dg =
    1e-12, where a run ends at 1 and at 1/2. A species is absent where its
    rate is nothing too."""
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*does not rest at"):
        sim.steady_state(sensitivity_params=["g"], method="newton")


def test_a_run_that_stayed_beside_columns_that_did_not_is_said_as_that(tmp_path):
    """The refusal for a run that ends beside the state with other columns,
    on a result that has everything else in order: the corpus model that it
    was written for is not in the repository (MODEL1607210000 below)."""
    sim = bngsim.Simulator(_net(tmp_path, BESIDE_A_FAST_EXCHANGE.format(F=10.0)), method="ode")
    out = sim.steady_state(sensitivity_params=["s", "k"])
    out.sens_root_hold_shift, out.sens_root_hold_drift, out.sens_root_hold_param = 40.8, 3e-6, "k"
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*though the run ends beside it.*no species by more than 3e-06.*"
        r"column of k.*moves by 40\.8 times its largest entry",
    ):
        sim._raise_if_not_an_isolated_root(out, 1e6, False)


# The resident and the invader again, with what could make the invader at
# nothing: a source of it at eps = 0, and its own initial amount N0 = 0.
AN_INVADER_A_PARAMETER_WOULD_MAKE = """begin parameters
    1 b   1.0
    2 c   1.0
    3 g   2.0
    4 d   1.0
    5 eps 0
    6 N0  0
end parameters
begin species
    1 R() 1.0
    2 N() N0
end species
begin reactions
    1 1 1,1 b
    2 1,1 1 c
    3 2 2,2 g
    4 1,2 1 d
    5 2,2 2 d
    6 0 2 eps
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("asked", [["eps"], ["N0"], ["b", "eps"]])
def test_an_absent_species_is_asked_where_the_parameter_asked_for_would_make_it(
    tmp_path, asked, method
):
    """N is at nothing and its rate is nothing: with eps = 0 and N0 = 0 it is
    a species the model does not have. But any eps above nothing, or any N0,
    makes some, and N then leaves nothing at g - d·R = 1 for where the two
    coexist: a run ends at N = 1 for eps = 1e-9. The steady state jumps
    there, and -J⁻¹·∂f/∂eps = -1 is the slope of the branch the system
    leaves. Left out of the eigenvalues, N came back with that."""
    sim = bngsim.Simulator(_net(tmp_path, AN_INVADER_A_PARAMETER_WOULD_MAKE), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 1 "):
        sim.steady_state(sensitivity_params=asked, method=method)


def test_the_same_asked_for_the_resident_alone_is_returned(tmp_path):
    """Control. Asked for b alone, nothing that is asked makes N, and the
    resident's column is that of where a run ends."""
    sim = bngsim.Simulator(_net(tmp_path, AN_INVADER_A_PARAMETER_WOULD_MAKE), method="ode")
    out = sim.steady_state(sensitivity_params=["b"])
    np.testing.assert_allclose(out.concentrations, [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out.sensitivity, [[1.0], [0.0]], atol=1e-7)


# The resident and the invader, and E, which makes the invader at k·E and is no
# unknown of the solve: nothing changes it, so it is the dependent of a
# conservation law of its own, whose total E0 sets.
AN_INVADER_MADE_BY_A_CATALYST = """begin parameters
    1 b   1.0
    2 c   1.0
    3 g   {g}
    4 d   1.0
    5 k   1.0
    6 E0  0
end parameters
begin species
    1 R() 1.0
    2 N() 0
    3 E() E0
end species
begin reactions
    1 1 1,1 b
    2 1,1 1 c
    3 2 2,2 g
    4 1,2 1 d
    5 2,2 2 d
    6 3 2,3 k
end reactions
"""

# The same with E fixed, `$E`, and set by E0. (Y is there for the solve to be
# the one on the unknowns: with no law and no rule a fixed species is refused
# as a singular Jacobian, as on main.)
AN_INVADER_MADE_BY_A_FIXED_SPECIES = (
    "species $E, R, N, $Y; E0 = 0; E = E0; R = 1; N = 0; "
    "b = 1; c = 1; g = {g}; d = 1; k = 1; Y := 2*R\n"
    "J1: R -> 2 R; b*R\nJ2: 2 R -> R; c*R*R\nJ3: N -> 2 N; g*N\n"
    "J4: N -> ; d*R*N\nJ5: N -> ; d*N*N\nJ6: -> N; k*E\n"
)


def _made_by_what_is_no_unknown(tmp_path, which, g):
    if which == "catalyst":
        return _net(tmp_path, AN_INVADER_MADE_BY_A_CATALYST.format(g=g))
    return bngsim.Model.from_antimony_string(AN_INVADER_MADE_BY_A_FIXED_SPECIES.format(g=g))


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("asked", [["E0"], ["b", "E0"]])
@pytest.mark.parametrize("which", ["catalyst", "fixed"])
def test_an_absent_species_is_asked_where_what_is_no_unknown_would_make_it(
    tmp_path, which, asked, method
):
    """N is made at k·E, and E is at E0 = 0 and is no unknown: the dependent
    of its own conservation law, or fixed. Neither ∂f/∂E0 nor the start of N
    reads E0, and N's row has no entry in an unknown's column; what makes N is
    the total, or the fixed species, that E0 moves. Any E0 above nothing makes
    some N, which leaves nothing at g - d·R = 1: a run ends at N = 1 for E0 =
    1e-9, and -1 came back for dN*/dE0, the slope of the branch the system
    leaves."""
    sim = bngsim.Simulator(_made_by_what_is_no_unknown(tmp_path, which, 2.0), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 1 "):
        sim.steady_state(sensitivity_params=asked, method=method)


@pytest.mark.parametrize("which", ["catalyst", "fixed"])
def test_the_same_where_the_invader_dies_out_is_returned(tmp_path, which):
    """Control. At g = 1/2 what E0 makes of N is removed at d·R - g = 1/2,
    N* = k·E0/(d·R - g) beside nothing, and dN*/dE0 = 2 is the derivative."""
    sim = bngsim.Simulator(_made_by_what_is_no_unknown(tmp_path, which, 0.5), method="ode")
    out = sim.steady_state(sensitivity_params=["E0"])
    names = [n.rstrip("()") for n in out.species_names]
    got = np.asarray(out.sensitivity)[[names.index(n) for n in ("R", "N", "E")], 0]
    np.testing.assert_allclose(got, [0.0, 2.0, 1.0], atol=1e-7)


# The invader is made at k·(R - 1): at nothing where the resident is at its
# capacity of 1, and nowhere else.
AN_INVADER_MADE_ANYWHERE_BUT_HERE = """begin parameters
    1 b   1.0
    2 c   1.0
    3 g   2.0
    4 d   1.0
    5 k   1e-3
end parameters
begin functions
    1 leak() k*(Robs-1)
end functions
begin species
    1 R() 1.0
    2 N() 0
end species
begin reactions
    1 1 1,1 b
    2 1,1 1 c
    3 2 2,2 g
    4 1,2 1 d
    5 2,2 2 d
    6 0 2 leak
end reactions
begin groups
    1 Robs 1
end groups
"""


def test_a_species_made_by_a_rate_that_is_nothing_only_here_is_not_left_out(tmp_path):
    """N is at nothing with a rate of exactly nothing, the resident being at
    exactly 1 where a Newton solve stops; but its row of the Jacobian has an
    entry in R's column, and beside this state something makes it. Its block
    is not one of its own, and it is in the eigenvalues."""
    sim = bngsim.Simulator(_net(tmp_path, AN_INVADER_MADE_ANYWHERE_BUT_HERE), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#995.*does not rest at.*real part of 1 "):
        sim.steady_state(sensitivity_params=["b"], method="newton")


@pytest.mark.parametrize(
    ("limit", "measure"),
    [
        ("_SS_ROOT_PIVOT_SHARE_MIN_DIFFERENCED", "pivot"),
        ("_SS_ROOT_CONDITION_MAX_DIFFERENCED", "x"),
    ],
)
def test_each_limit_of_a_differenced_jacobian_refuses_alone(tmp_path, monkeypatch, limit, measure):
    """The pivot share and the condition number go together in a stiff model
    (one is about twice the reciprocal of the other), and either limit refuses
    the fast exchange at F = 1e8 with the other out of the way."""
    other = {
        "_SS_ROOT_PIVOT_SHARE_MIN_DIFFERENCED": ("_SS_ROOT_CONDITION_MAX_DIFFERENCED", math.inf),
        "_SS_ROOT_CONDITION_MAX_DIFFERENCED": ("_SS_ROOT_PIVOT_SHARE_MIN_DIFFERENCED", 0.0),
    }[limit]
    monkeypatch.setattr(bngsim.Simulator, other[0], other[1])
    sim, _, _ = _into_a_fast_exchange(tmp_path, 1e8, jacobian="fd")
    said = "The pivot for" if measure == "pivot" else "magnified"
    with pytest.raises(bngsim.SimulationError, match=rf"#995.*{said}.*difference quotient"):
        sim.steady_state(sensitivity_params=["kx"], tol=1e-6)


def test_a_differenced_jacobian_just_inside_its_limits_is_right(tmp_path):
    """At F = 2e5 the pivot share is 3e-6 and the condition number 5e5, just
    inside 1e-6 and 1e6, and the columns of the difference quotient are 0.13%
    off: within the 1% that is asked, which is where the limits are put."""
    sim, relative, _ = _into_a_fast_exchange(tmp_path, 2e5, jacobian="fd")
    out = sim.steady_state(sensitivity_params=["kx"], tol=1e-9)
    assert out.sens_jacobian_source == "finite-difference"
    assert 1e-6 < out.sens_root_pivot_share < 1e-5 and out.sens_root_condition < 1e6
    got = np.asarray(out.sensitivity)[:, 0] / np.asarray(out.concentrations)
    np.testing.assert_allclose(got, relative, rtol=5e-3)
