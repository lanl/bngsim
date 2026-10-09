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
- a pivot that is under 1e-10 of the terms it was computed from, or a
  componentwise condition number above 1e12: the Jacobian is singular
  whatever the state, but for rounding;
- a column that moves by more than 1%, each species over its own scale: the
  returned state is short of the steady state;
- a column that moves by more than 1% where the run ends: the returned state
  is one a run leaves;
- an eigenvalue right of zero, or a determinant of the sign an odd number of
  them gives it: the system does not rest at the state;
- a column of which a run of ``max_time`` would leave more than 1%
  unestablished: the steady state is one no run of that length reaches;

and on a species ``mask=`` left out that an equation of the kept ones reads.
"""

from __future__ import annotations

import logging
import math
import os
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
    whose min|U|/max|U| falls with the size ratio. It was refused for a law
    across sizes at a ratio below 1e-8 (issue #758), though the columns are
    right: dA*/dk1 = -8/75, dX*/dk1 = -4/75, dB*/dk1 = 4/(75·V2)."""
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
# tol = 1e-9 while B is still 28% short.
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
    the residual passes 1e-9 while A is 28% from A* = 5e-7, and dA*/dkf
    came back 28% from -5e-5 with ``converged=True`` and nothing logged. The
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
    """Control. With ``tol=1e-14`` the solver's own state is on the root, no
    column moves, and no step is taken: A* = B* = 5e-7 and dA*/dkf =
    -A0·kr/(kf + kr)² = -5e-5, dA*/dkr = +5e-5."""
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


def test_a_column_of_zeros_is_not_a_column_that_moved(tmp_path):
    """Control. Everything decays to zero, and so does every column. What is
    returned is what ``tol`` left of them, 1e-9, and one Newton step takes all
    of that away. X and Y are at a zero: each Newton step takes them to
    nothing, and their entries are taken over the 100 that G, which makes
    them, started at. Every entry is then under 1e-3 of 1/|p|, which is a
    column of zeros, and a move of all of it is not a move of a derivative."""
    out = bngsim.Simulator(_net(tmp_path, DECAY), method="ode").steady_state(
        sensitivity_params=["k1", "k2", "k3"]
    )
    assert np.max(np.abs(np.asarray(out.sensitivity))) < 1e-6


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
    1e-15 returns, over its largest entry: 0.1% here at ``tol=1e-11``. The
    largest of the three differences is that of A, which the law is solved
    for and which follows from the other two."""
    model = _net(tmp_path, STAR)
    assert [model.species_names[i] for i in model.conservation_laws["dependent"]] == ["A()"]
    loose = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["k1"], tol=1e-11
    )
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
    """Control. X' = -kd·X + k2·X² ends at 0, and from X = 6e-14 a Newton step
    overshoots it, to -k2·X²/kd. The Hill rate there is not a number, and
    neither would the Jacobian at the corrected state be: the columns, which
    are right, would be refused for a pivot that is not one. The corrected
    state has 0 for such a species."""
    sim = bngsim.Simulator(_net(tmp_path, BELOW_ZERO), method="ode")
    out = sim.steady_state(sensitivity_params=["kd", "v", "K"], atol=1e-14, rtol=1e-10)
    assert out.converged and 0 < np.asarray(out.concentrations)[0] < 1e-12
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
def test_a_focus_beside_a_fast_species_is_refused_by_the_run(tmp_path, method):
    """The same fixed point beside a species turned over at 1e7. The
    eigenvalue of 0.5 is 5e-8 of the largest, under what the spectrum is
    trusted to (1e-6 of it), and the columns came back. The run taken on from
    a millionth beside the state is what says so: it is on the limit cycle
    when its steps are used up."""
    text = BESIDE_A_FAST_SPECIES.format(B="3.0", kfast="1e7")
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
    """With B = 2.001 beside the fast species, the growth of 0.0005 is too
    slow for the spectrum (5e-11 of the largest eigenvalue) and for the run:
    in the 10,000 steps it has it gets to t = 2,500, 1.2e-4 from where it
    started, with ``max_time`` at 1e6. That is not a run that stayed."""
    text = BESIDE_A_FAST_SPECIES.format(B="2.001", kfast="1e7")
    sim = bngsim.Simulator(_net(tmp_path, text), method="ode")
    with pytest.raises(
        bngsim.SimulationError,
        match=r"#995.*did not get to max_time \(1e\+06\).*10000 steps.*not known to stay",
    ):
        sim.steady_state(sensitivity_params=["A", "B"])


def test_a_focus_that_dies_away_slowly_is_returned(tmp_path):
    """Control. With B = 1.9 the eigenvalues are -0.05 ± i: the run taken on
    winds in for some 1,100 steps and then gets to ``max_time``."""
    sim = bngsim.Simulator(_net(tmp_path, BRUSSELATOR.format(B="1.9")), method="ode")
    out = sim.steady_state(sensitivity_params=["A", "B"])
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[1.0, 0.0], [-1.9, 1.0]], rtol=1e-9, atol=1e-12
    )
    assert out.sens_root_hold_time == pytest.approx(1e6)
    assert 200 < out.sens_root_hold_steps < 10000


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
    """Control. X' = e·X with X = 0, beside Y' = -kd·Y. Any X grows, and there
    is none: a run from the state the model starts in stays, whatever e is,
    and the columns are zeros. The run that is taken on moves each
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


def test_biomd1001_is_taken_on_with_a_differenced_jacobian():
    """Control. The run taken on from the returned state stops the integrator
    on the closed-form Jacobian, started again or not, and reaches
    ``max_time`` on a differenced one, as the solve itself does where its
    integrator gives up (issue #127). Ten of its species have run out, at
    1e-8 to 1e-21 beside others at 1,000, and nothing that is left makes
    them: their entries are taken over what stands beside them. The columns
    are returned, as they were."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000001001")), method="ode")
    out = sim.steady_state(sensitivity_params=["R1_total_C4", "k_in_R1_C4", "kdeg_R1"])
    assert np.all(np.isfinite(np.asarray(out.sensitivity)))
    assert out.sens_root_hold_time >= 1e6
    scale = dict(zip(out.species_names, out.sens_species_scale, strict=True))
    assert scale["TGFb_In"] > 1.0 and scale["pS2_c"] > 1.0


def test_biomd5_by_newton_is_a_root_inside_its_limit_cycle():
    """Tyson's cell cycle oscillates, and Newton finds the fixed point the
    cycle goes round, with eigenvalues 0.16 ± 0.18i beside one of -1e6: 1.6e-7
    of the largest, which the spectrum is not trusted to. The run taken on
    leaves for the cycle."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml(_biomodel("BIOMD0000000005")), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*not one a run stays at.*the run moves \w+ by \d+%"
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
