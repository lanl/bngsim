"""A transfer between compartments when one of them changes size.

A species stored as a concentration in a compartment whose size is integrated
(a rate rule) or reset (an event) obeys ``d[S]/dt = stoich·law/V(t) − [S]·V'/V``.
The dilution term was there. The transfer term divided by the size the
compartment had when the model was loaded, unless the reaction was an
irreversible mass-action monomial, the one shape the loader certifies for the
SSA's live-volume correction. Every other transfer was integrated with the
load-time size, with no warning:

* a monomial on a reaction flagged reversible, which SBML Level 2 does by
  default and Antimony's ``->`` does;
* a reversible difference ``k*A - k2*B``;
* a saturating law.

With ``A -> B; k*A`` into a compartment growing from 2 at rate 1, [B](5) came
back 0.284 for 0.196.

The reference integrates the two amounts with scipy, which shares nothing with
bngsim, and divides by the sizes at the output times.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from scipy.integrate import solve_ivp

K, K2, KM = 0.7, 0.3, 0.5
A0, B0 = 1.0, 0.2
T = np.linspace(0.0, 5.0, 11)

# name -> (Antimony reaction, flux in amount per time from concentrations and sizes)
LAWS = {
    "monomial-flagged-reversible": ("J1: A -> B; k*A;", lambda a, b, v1, v2: K * a),
    "reversible-difference": ("J1: A -> B; k*A - k2*B;", lambda a, b, v1, v2: K * a - K2 * b),
    "saturating": ("J1: A => B; k*A/(Km + A);", lambda a, b, v1, v2: K * a / (KM + a)),
    "saturating-flagged-reversible": (
        "J1: A -> B; k*A/(Km + A);",
        lambda a, b, v1, v2: K * a / (KM + a),
    ),
    "reversible-with-explicit-sizes": (
        "J1: A -> B; C1*k*A - C2*k2*B;",
        lambda a, b, v1, v2: v1 * K * a - v2 * K2 * b,
    ),
    # The shape the loader already certified: the control.
    "monomial-irreversible": ("J1: A => B; k*A;", lambda a, b, v1, v2: K * a),
}

# name -> (C1 at load, C2 at load, Antimony rules, V1(t), V2(t), resize time or None)
COMPARTMENTS = {
    "product-side-rate-rule": (1.0, 2.0, "C2' = 1;", lambda t: 1.0, lambda t: 2.0 + t, None),
    "equal-sizes-at-load": (2.0, 2.0, "C2' = 1;", lambda t: 2.0, lambda t: 2.0 + t, None),
    "reactant-side-rate-rule": (
        1.0,
        2.0,
        "C1' = 0.5;",
        lambda t: 1.0 + 0.5 * t,
        lambda t: 2.0,
        None,
    ),
    "both-rate-rules": (
        1.0,
        2.0,
        "C1' = 0.5; C2' = 1;",
        lambda t: 1.0 + 0.5 * t,
        lambda t: 2.0 + t,
        None,
    ),
    # Neither size is a parameter a caller can write, and they are equal when the
    # model is loaded, which is what the single representative divide keys on.
    "both-rate-rules-equal-at-load": (
        2.0,
        2.0,
        "C1' = 0.5; C2' = 1;",
        lambda t: 2.0 + 0.5 * t,
        lambda t: 2.0 + t,
        None,
    ),
    "resized-by-an-event": (
        1.0,
        2.0,
        "E1: at (time > 1.25): C2 = 5;",
        lambda t: 1.0,
        lambda t: 2.0 if t <= 1.25 else 5.0,
        1.25,  # between two output times
    ),
}


def _text(compartment: str, law: str) -> str:
    c1, c2, rules, _v1, _v2, _resize = COMPARTMENTS[compartment]
    return (
        f"compartment C1 = {c1}, C2 = {c2};\n{rules}\n"
        f"species A in C1, B in C2; A = {A0}; B = {B0};\n"
        f"k = {K}; k2 = {K2}; Km = {KM};\n{LAWS[law][0]}\n"
    )


def _reference(compartment: str, law: str) -> np.ndarray:
    """[A] and [B] at T from the amounts, which a resize leaves as they are."""
    c1, c2, _rules, v1, v2, resize = COMPARTMENTS[compartment]
    flux = LAWS[law][1]

    def rhs(side):
        def f(t, n):
            # `side` picks the branch of a size that is reset at `resize`.
            s1, s2 = v1(t + side), v2(t + side)
            j = flux(n[0] / s1, n[1] / s2, s1, s2)
            return [-j, j]

        return f

    amounts = np.empty((len(T), 2))
    stops = [T[0], T[-1]] if resize is None else [T[0], resize, T[-1]]
    n = np.array([A0 * c1, B0 * c2])
    for lo, hi in zip(stops[:-1], stops[1:], strict=True):
        side = 0.0 if resize is None else (-1e-9 if hi == resize else 1e-9)
        mask = (lo <= T) & ((hi > T) if hi != T[-1] else (hi >= T))
        sol = solve_ivp(
            rhs(side), (lo, hi), n, t_eval=T[mask], rtol=1e-12, atol=1e-14, dense_output=True
        )
        amounts[mask] = sol.y.T
        n = sol.sol(hi)
    sizes = np.array([[v1(t), v2(t)] for t in T])
    return amounts / sizes


def _run(text: str, **kw) -> np.ndarray:
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", **kw).run(
        sample_times=list(T), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)
    return np.stack([x[:, names.index("A")], x[:, names.index("B")]], axis=1)


@pytest.mark.parametrize("law", sorted(LAWS))
@pytest.mark.parametrize("compartment", sorted(COMPARTMENTS))
def test_a_transfer_divides_by_the_size_the_compartment_has(compartment, law):
    """Every law over every changing compartment. The irreversible monomial was
    already right in each."""
    got = _run(_text(compartment, law))
    np.testing.assert_allclose(got, _reference(compartment, law), rtol=1e-7, atol=1e-9)


SUBSTANCE_ONLY = {
    # B is an amount, so the law names its concentration as B/C2.
    "product": (
        "product-side-rate-rule",
        "species A in C1; substanceOnly species B in C2; A = 1; B = 0.4;",
        "J1: A -> B; k*A - k2*B/C2;",
        "reversible-difference",
    ),
    "reactant": (
        "reactant-side-rate-rule",
        "substanceOnly species A in C1; species B in C2; A = 1; B = 0.2;",
        "J1: A -> B; k*A/C1;",
        "monomial-flagged-reversible",
    ),
}


@pytest.mark.parametrize("which", sorted(SUBSTANCE_ONLY))
def test_a_substance_only_species_is_left_as_it_was(which):
    """Control. A species declared in substance units is stored as its amount
    over the load-time size, and its row is divided by that size. It was right,
    and the live divide is not for it."""
    compartment, species, reaction, law = SUBSTANCE_ONLY[which]
    c1, c2, rules, _v1, _v2, _resize = COMPARTMENTS[compartment]
    text = (
        f"compartment C1 = {c1}, C2 = {c2};\n{rules}\n{species}\n"
        f"k = {K}; k2 = {K2}; Km = {KM};\n{reaction}\n"
    )
    np.testing.assert_allclose(_run(text), _reference(compartment, law), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("jacobian", ["analytical", "fd"])
def test_the_analytical_jacobian_and_the_finite_difference_one_agree(jacobian):
    """The live size is a state, so the divide has a column of its own in the
    Jacobian."""
    got = _run(_text("both-rate-rules", "reversible-difference"), jacobian=jacobian)
    want = _reference("both-rate-rules", "reversible-difference")
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


def test_the_sensitivity_follows_the_live_size():
    """d[B]/dk against a central difference of the reference."""
    text = _text("product-side-rate-rule", "reversible-difference")
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
        sample_times=list(T), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    got = np.asarray(run.sensitivities)[:, names.index("B"), 0]

    def b_at(value):
        m = bngsim.Model.from_antimony_string(text)
        m.set_param("k", value)
        r = bngsim.Simulator(m, method="ode").run(sample_times=list(T), rtol=1e-12, atol=1e-14)
        return np.asarray(r.species)[:, list(r.species_names).index("B")]

    def central(h):
        return (b_at(K + h) - b_at(K - h)) / (2 * h)

    want = (4.0 * central(5e-4) - central(1e-3)) / 3.0
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)
    # And the plain run those differences are taken from is the right one.
    np.testing.assert_allclose(
        b_at(K),
        _reference("product-side-rate-rule", "reversible-difference")[:, 1],
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.parametrize("law", ["reversible-difference", "saturating"])
def test_the_ssa_still_refuses_what_it_cannot_correct(law):
    """Control. Only the ODE divide is general: the SSA's live-volume correction
    is exact for a monomial and nothing else, so these stay refused there."""
    model = bngsim.Model.from_antimony_string(_text("product-side-rate-rule", law))
    with pytest.raises(Exception, match="variable-volume"):
        bngsim.Simulator(model, method="ssa").run(t_span=(0.0, 1.0), n_points=3)
