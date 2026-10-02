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

A reaction that also changes a species in an assignment-rule compartment, or
that mixes conversion factors, is written out one species at a time, each over
its own compartment's live size. A species the reaction does not change, a
catalyst, does not count as one it moves between compartments.

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


# ─── Three species, an assignment-rule compartment, conversion factors ──────


def _final(flux, v_a, v_b, n_a, n_b, cf_a=1.0, cf_b=1.0):
    """[A](5) and [B](5) from the amounts: A loses cf_a·flux, B gains cf_b·flux."""

    def rhs(t, n):
        j = flux(n[0] / v_a(t), n[1] / v_b(t), t)
        return [-cf_a * j, cf_b * j]

    n = solve_ivp(rhs, (0.0, 5.0), [n_a, n_b], rtol=1e-12, atol=1e-14).y[:, -1]
    return np.array([n[0] / v_a(5.0), n[1] / v_b(5.0)])


def _final_of(model) -> np.ndarray:
    run = bngsim.Simulator(model, method="ode").run(
        sample_times=[0.0, 2.5, 5.0], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)[-1]
    return np.array([x[names.index("A")], x[names.index("B")]])


def _half(t):
    return 2.0 + 0.5 * t


def _one(t):
    return 2.0 + t


@pytest.mark.parametrize("arrow", ["->", "=>"])
def test_a_catalyst_in_a_changing_compartment_is_not_a_species_it_moves(arrow):
    """Control. A and B share an assignment-rule compartment and the catalyst E
    sits in one that follows a rate rule. Nothing crosses a compartment: E is
    left as it was found. The single divide by A and B's own compartment was
    right, and a first cut of this fix took E for a changed species and sent the
    reaction to the per-species divide, which for an assignment-rule compartment
    is the load-time size (#745): [A](5) = 0.287 for 0.326."""
    text = (
        "compartment C1 = 2, C3 = 2; C1 := 2 + 0.5*time; C3' = 1;\n"
        "species A in C1, B in C1, E in C3; A = 1; B = 0.2; E = 0.5; k = 0.7;\n"
        f"J1: A + E {arrow} B + E; k*A*E;\n"
    )
    want = _final(lambda a, b, t: K * a * (1.0 / _one(t)), _half, _half, 2.0, 0.4)
    model = bngsim.Model.from_antimony_string(text)
    np.testing.assert_allclose(_final_of(model), want, rtol=1e-7)
    # And it is still the one reaction it was, not one per species.
    reactions = model._core.codegen_data()["reactions"]
    assert len([r for r in reactions if "J1" in r.get("function_name", "")]) == 1


def test_a_static_catalyst_beside_two_species_in_a_changing_compartment():
    """The mirror image: the catalyst is in a static compartment and A and B in
    one that grows. The reaction spans two compartments, so its rows took the
    per-species divide, by the load-time size: [A](5) = 0.119 for 0.184."""
    text = (
        "compartment C1 = 1, C3 = 2; C3' = 1;\n"
        "species E in C1, A in C3, B in C3; A = 1; B = 0.2; E = 0.5; k = 0.7;\n"
        "J1: E + A -> E + B; k*A*E;\n"
    )
    want = _final(lambda a, b, t: K * a * 0.5, _one, _one, 2.0, 0.4)
    got = _final_of(bngsim.Model.from_antimony_string(text))
    np.testing.assert_allclose(got, want, rtol=1e-7)


@pytest.mark.parametrize(
    ("species", "law", "flux", "v_a", "v_b"),
    [
        ("A in C1, B in C3", "k*A", lambda a, b, t: K * a, _half, _one),
        ("A in C3, B in C1", "k*A", lambda a, b, t: K * a, _one, _half),
        ("A in C1, B in C3", "k*A - k2*B", lambda a, b, t: K * a - K2 * b, _half, _one),
    ],
    ids=["out-of-the-rule", "into-the-rule", "reversible-difference"],
)
@pytest.mark.parametrize("arrow", ["->", "=>"])
def test_a_transfer_between_an_assignment_rule_and_a_rate_rule_compartment(
    species, law, flux, v_a, v_b, arrow
):
    """C1 follows an assignment rule and C3 a rate rule, with the same size at
    load. The single divide was by C1's live size for both species, so the one
    in C3 was wrong: [B](5) = 0.302 for 0.251. Each row is now over its own
    compartment's live size.

    Flagged irreversible, a monomial was certified for the SSA's correction and
    kept the per-species divide, by the load-time size of the assignment-rule
    compartment (#745): [A](5) = 0.077 for 0.143. The SSA refuses an
    assignment-rule compartment anyway, so it is written out as the reversible
    one is."""
    text = (
        "compartment C1 = 2, C3 = 2; C1 := 2 + 0.5*time; C3' = 1;\n"
        f"species {species}; A = 1; B = 0.2; k = {K}; k2 = {K2};\n"
        f"J1: A {arrow} B; {law};\n"
    )
    got = _final_of(bngsim.Model.from_antimony_string(text))
    np.testing.assert_allclose(got, _final(flux, v_a, v_b, 2.0, 0.4), rtol=1e-7)


@pytest.mark.parametrize(
    ("rules", "v_a"),
    [("C2' = 1;", lambda t: 2.0), ("C1' = 1; C2' = 1;", _one)],
    ids=["one-changing", "both-in-step"],
)
@pytest.mark.parametrize("arrow", ["->", "=>"])
def test_a_transfer_that_mixes_conversion_factors(tmp_path, rules, v_a, arrow):
    """A loses 1.5 per unit of flux and B gains 0.5. With only C2 changing, [B]
    was divided by the size at load: 0.212 for 0.145. With the two compartments
    growing in step the single divide happened to be right, and that one is the
    control: a first cut sent it to an emission that refuses mixed factors."""
    import antimony
    import libsbml

    antimony.clearPreviousLoads()
    antimony.loadAntimonyString(
        f"compartment C1 = 2, C2 = 2; {rules}\n"
        "species A in C1, B in C2; A = 1; B = 0.2; k = 0.7; cfA = 1.5; cfB = 0.5;\n"
        f"J1: A {arrow} B; k*A;\n"
    )
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    model = doc.getModel()
    for pid in ("cfA", "cfB"):
        model.getParameter(pid).setConstant(True)
    model.getSpecies("A").setConversionFactor("cfA")
    model.getSpecies("B").setConversionFactor("cfB")
    path = tmp_path / "m.xml"
    path.write_text(libsbml.writeSBMLToString(doc))

    want = _final(lambda a, b, t: K * a, v_a, _one, 2.0, 0.4, cf_a=1.5, cf_b=0.5)
    np.testing.assert_allclose(_final_of(bngsim.Model.from_sbml(str(path))), want, rtol=1e-7)


@pytest.mark.parametrize(
    ("stoich_b", "factors"),
    [(1.0, {"A": 2.0, "B": 3.0, "H": 1.5}), (1.5, {})],
    ids=["mixed-conversion-factors", "non-integer-stoichiometry"],
)
def test_a_concentration_and_an_amount_changed_in_one_changing_compartment(
    tmp_path, stoich_b, factors
):
    """B is a concentration and H an amount, both in C2, which grows, and one
    reaction makes both. Written out one species at a time, B's row is divided
    by C2's live size and H's by its size at load. The two functions were
    registered under one name and the second was dropped, so both rows divided
    by the same one: [B](5) = 0.6154 for 0.5624 with a stoichiometry of 1.5.
    With mixed conversion factors the model did not load before this fix, and
    a first cut of it gave 0.6196 for 0.5802.

    H is reported over the size C2 has, like every species."""
    import antimony
    import libsbml

    antimony.clearPreviousLoads()
    antimony.loadAntimonyString(
        "compartment C1 = 1, C2 = 2; C2' = 0.4;\n"
        "species A in C1; species B in C2; substanceOnly species H in C2;\n"
        "A = 1; B = 0.6; H = 0.4; k = 0.7; k2 = 0.3;\n"
        f"J1: A -> {stoich_b!r} B + H; k*A - k2*B*H;\n"
    )
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    model = doc.getModel()
    for sid, value in factors.items():
        factor = model.createParameter()
        factor.setId(f"cf{sid}")
        factor.setValue(value)
        factor.setConstant(True)
        model.getSpecies(sid).setConversionFactor(f"cf{sid}")
    path = tmp_path / "m.xml"
    path.write_text(libsbml.writeSBMLToString(doc))

    def rhs(t, n):
        flux = 0.7 * n[0] - 0.3 * (n[1] / (2.0 + 0.4 * t)) * n[2]
        return [
            -factors.get("A", 1.0) * flux,
            stoich_b * factors.get("B", 1.0) * flux,
            factors.get("H", 1.0) * flux,
        ]

    n = solve_ivp(rhs, (0.0, 5.0), [1.0, 1.2, 0.4], rtol=1e-12, atol=1e-14).y[:, -1]
    run = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode").run(
        sample_times=[0.0, 2.5, 5.0], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    got = [np.asarray(run.species)[-1][names.index(sid)] for sid in "ABH"]
    np.testing.assert_allclose(got, [n[0], n[1] / 4.0, n[2] / 4.0], rtol=1e-7)


def test_the_certified_monomial_keeps_the_emission_its_ssa_correction_is_for():
    """Control. An irreversible monomial from a static compartment into one that
    follows a rate rule is the one shape the SSA can correct, and its correction
    is written for the one-reaction per-species emission. (Between an
    assignment-rule compartment and a rate-rule one it is not certified: the SSA
    refuses an assignment-rule compartment, and the per-species divide there
    was by its load-time size.)"""
    text = (
        "compartment C1 = 2, C3 = 2; C3' = 1;\n"
        "species A in C1, B in C3; A = 1; B = 0.2; k = 0.7;\n"
        "J1: A => B; k*A;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    reactions = model._core.codegen_data()["reactions"]
    transfer = [r for r in reactions if r.get("function_name") == "J1"]
    assert len(transfer) == 1
    assert transfer[0]["per_species_volume_scaling"]
    assert sorted(transfer[0]["reactants"]) != []
    want = _final(lambda a, b, t: K * a, lambda t: 2.0, _one, 2.0, 0.4)
    np.testing.assert_allclose(_final_of(model), want, rtol=1e-7)
    assert not [i for i in model.validate_for_ssa() if i.severity == "error"]
