"""An assignment rule that reads a later rule's target gets its load-time value (issue #994).

The loader evaluates assignment rules for their load-time values in one pass, in
document order, and repeated the pass to a fixed point only for a model with an
``initialAssignment``. Antimony writes ``v := 2*p; c2 := v`` as ``c2`` then
``v``, so ``c2 := v`` read ``v`` unset and the compartment loaded with volume
factor 1 instead of 4. Every reaction across compartments then divided by 1:
``run()`` and ``steady_state()`` ended at ``[0.8, 0.4]`` for ``[0.6, 0.3]``, with
nothing raised. The settle loop now runs for every model, and long enough for a
chain of any length.
"""

from __future__ import annotations

import bngsim
import libsbml
import numpy as np
import pytest

# A (in c1 = 1) <-> B (in c2, size 4 every way it is written). The exact steady
# state is A* = k2*T/(k1 + k2) = 0.6 and B* = (T - A*)/4 = 0.3, T = 1 + 0.2*4.
TEXT = (
    "compartment c1, c2; c1 = 1; {c2};\n"
    "species A in c1, B in c2; A = 1; B = 0.2; k1 = 1; k2 = 0.5;\n"
    "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
)
STEADY = [0.6, 0.3]

SPELLINGS = {
    "constant": "c2 = 4",
    "one rule": "p = 2; c2 := 2*p",
    "rule over a rule": "p = 2; v := 2*p; c2 := v",
    "chain of three": "p = 2; v := 2*p; w := v; c2 := w",
}


def _volume_factors(model: bngsim.Model) -> list[float]:
    return [s["volume_factor"] for s in model._core.codegen_data()["species"]]


def _end(model: bngsim.Model) -> np.ndarray:
    out = bngsim.Simulator(model, "ode").run(t_span=(0, 200.0), n_points=2, rtol=1e-11, atol=1e-13)
    return np.asarray(out.species)[-1]


def _sbml(c2: str, *, reverse_rules: bool) -> str:
    """The Antimony model as SBML, with its assignment rules in document order
    or reversed, so each test sees a rule both before and after what it reads."""
    import antimony

    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(TEXT.format(c2=c2)) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    model = doc.getModel()
    rules = [model.getRule(j).clone() for j in range(model.getNumRules())]
    while model.getNumRules():
        model.removeRule(0)
    for rule in reversed(rules) if reverse_rules else rules:
        assert model.addRule(rule) == libsbml.LIBSBML_OPERATION_SUCCESS
    return libsbml.writeSBMLToString(doc)


@pytest.mark.parametrize("c2", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_the_compartment_loads_at_its_size(c2: str) -> None:
    model = bngsim.Model.from_antimony_string(TEXT.format(c2=c2))
    assert model.get_param("c2") == pytest.approx(4.0)
    assert _volume_factors(model) == pytest.approx([1.0, 4.0])
    np.testing.assert_allclose(_end(model), STEADY, rtol=1e-6)


@pytest.mark.parametrize("reverse_rules", [False, True], ids=["as written", "reversed"])
@pytest.mark.parametrize(
    "c2", [SPELLINGS["rule over a rule"], SPELLINGS["chain of three"]], ids=["two", "three"]
)
def test_either_rule_order_in_the_document(c2: str, reverse_rules: bool) -> None:
    model = bngsim.Model.from_sbml_string(_sbml(c2, reverse_rules=reverse_rules))
    assert _volume_factors(model) == pytest.approx([1.0, 4.0])
    np.testing.assert_allclose(_end(model), STEADY, rtol=1e-6)


def test_a_chain_longer_than_ten_links_settles() -> None:
    # Written against document order, each pass settles one more link; the loop
    # used to stop after 10 passes whatever the chain's length.
    links = ["c2 := w13"] + [f"w{i} := w{i - 1}" for i in range(13, 1, -1)] + ["w1 := 2*p"]
    model = bngsim.Model.from_antimony_string(TEXT.format(c2="p = 2; " + "; ".join(links)))
    assert _volume_factors(model) == pytest.approx([1.0, 4.0])
    np.testing.assert_allclose(_end(model), STEADY, rtol=1e-6)


def test_steady_state_is_the_systems_own() -> None:
    model = bngsim.Model.from_antimony_string(TEXT.format(c2=SPELLINGS["rule over a rule"]))
    result = bngsim.Simulator(model, "ode").steady_state()
    assert result.converged
    np.testing.assert_allclose(np.asarray(result.concentrations).ravel(), STEADY, rtol=1e-6)


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_steady_state_sensitivities_are_right(spelling) -> None:
    # With both volume factors 1 the model looked like one with no law across
    # sizes, and steady_state returned columns [-0.2667, 0.2667] for
    # [-0.4, 0.1]. At the right sizes it is a rule-sized compartment under a law
    # across sizes, which #745/#992 refused; a rule that reads nothing that
    # moves gives the size the right-hand side reads, and the columns are
    # computed now, and right (issue #961).
    model = bngsim.Model.from_antimony_string(TEXT.format(c2=SPELLINGS[spelling]))
    out = bngsim.Simulator(model, "ode").steady_state(sensitivity_params=["k1"])
    np.testing.assert_allclose(np.asarray(out.sensitivity).ravel(), [-0.4, 0.1], rtol=1e-6)


def test_a_chain_through_reaction_ids_settles() -> None:
    # `q := J1` and `c2 := q`, where J1's law reads J2, ..., and J10's reads
    # `r := 2*p`. A reaction id refolds at the end of a pass, and J1 reads J2
    # before J2 refolds, so each reaction is a link of its own. A cap of one
    # pass per rule and initialAssignment ran out with J1 still moving, and c2
    # loaded at q's declared 7, with nothing raised.
    n = 10
    lines = [
        "compartment c1 = 1, c2; p = 2; r = 1; r := 2*p; q := J1; c2 := q;",
        "species X in c1, Z in c2; X = 1; Z = 1;",
        *(f"J{i}: X -> X; J{i + 1}" for i in range(1, n)),
        f"J{n}: X -> X; r",
        "R0: Z -> X; Z",
    ]
    import antimony

    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString("\n".join(lines)) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    doc.getModel().getParameter("q").setValue(7.0)
    model = bngsim.Model.from_sbml_string(libsbml.writeSBMLToString(doc))
    assert model.get_param("c2") == pytest.approx(4.0)
    assert _volume_factors(model) == pytest.approx([1.0, 4.0])


# A species declared by the quantity its symbol does not mean is converted by its
# compartment's size: B declared as amount 0.8 in c2 is concentration 0.8/c2 to
# every expression that reads it. With c2 declared 1 and given 4 by
# `p = 2; c2 = 2*p` or `c2 := 2*p`, an initialAssignment that read B used to
# read 0.8 / 1.
SIZED = {
    "initialAssignment": "c2 = 2*p",
    "rule": "c2 := 2*p",
    "rule over a rule": "v := 2*p; c2 := v",
}
SPECIES_TEXT = (
    "compartment c1, c2; c1 = 1; c2 = 1; p = 2; {sized}; {reads};\n"
    "species A in c1, B in c2, C in c1; B = 0.8; C = 0;\n"
    "R1: -> C; q\n"
)


def _species_sbml(sized: str, reads: str, *, amount_only: bool) -> str:
    import antimony

    antimony.clearPreviousLoads()
    text = SPECIES_TEXT.format(sized=sized, reads=reads)
    assert antimony.loadAntimonyString(text) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    b = doc.getModel().getSpecies("B")
    if amount_only:
        b.unsetInitialConcentration()
        b.setInitialAmount(0.8)  # concentration 0.8/4 = 0.2
    else:
        b.setHasOnlySubstanceUnits(True)
        b.unsetInitialAmount()
        b.setInitialConcentration(0.2)  # amount 0.2*4 = 0.8
    return libsbml.writeSBMLToString(doc)


@pytest.mark.parametrize("sized", SIZED.values(), ids=SIZED.keys())
def test_a_species_initial_assignment_reads_the_size_at_the_start(sized: str) -> None:
    model = bngsim.Model.from_sbml_string(_species_sbml(sized, "q = 1; A = B", amount_only=True))
    out = bngsim.Simulator(model, "ode").run(t_span=(0, 1.0), n_points=2)
    assert np.asarray(out.species)[0, 0] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "amount_only", [True, False], ids=["amount for concentration", "concentration for amount"]
)
@pytest.mark.parametrize("sized", SIZED.values(), ids=SIZED.keys())
def test_a_parameter_initial_assignment_reads_the_size_at_the_start(
    sized: str, amount_only: bool
) -> None:
    # B's symbol means its concentration (0.2) when it is declared by amount, and
    # its amount (0.8) when it has only substance units and is declared by
    # concentration.
    model = bngsim.Model.from_sbml_string(
        _species_sbml(sized, "A = 1; q = B", amount_only=amount_only)
    )
    q = 0.2 if amount_only else 0.8
    assert model.get_param("q") == pytest.approx(q)
    out = bngsim.Simulator(model, "ode").run(t_span=(0, 1.0), n_points=2)
    assert np.asarray(out.species)[-1, 2] == pytest.approx(q)  # C(1) = q*1


def test_a_size_rule_reads_a_species_in_a_rule_sized_compartment() -> None:
    # The volume-factor path of #994 through a converted species: c2 := 1 + B,
    # with B declared as amount 0.8 in c1 := 2*p (declared 1), is 1 + 0.8/4.
    import antimony

    antimony.clearPreviousLoads()
    text = (
        "compartment c1 = 1, c2 = 1, c0 = 1; p = 2; c1 := 2*p; c2 := 1 + B;\n"
        "species B in c1, Z in c2, Y in c0; B = 0.8; Z = 1; Y = 0;\n"
        "R1: Z -> Y; Z*c2\n"
    )
    assert antimony.loadAntimonyString(text) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    b = doc.getModel().getSpecies("B")
    b.unsetInitialConcentration()
    b.setInitialAmount(0.8)
    model = bngsim.Model.from_sbml_string(libsbml.writeSBMLToString(doc))
    assert model.get_param("c2") == pytest.approx(1.2)
    names = [s["name"] for s in model._core.codegen_data()["species"]]
    assert dict(zip(names, _volume_factors(model), strict=True))["Z"] == pytest.approx(1.2)


def test_a_stoichiometry_reads_a_species_in_a_rule_sized_compartment() -> None:
    # The speciesReference id of Y in X -> Y takes 10*B at the start, with B
    # declared as amount 0.8 in c2 := 2*p: 10*0.2 = 2, so Y(1) = 2*(1 - e^-0.5).
    import antimony

    antimony.clearPreviousLoads()
    text = (
        "compartment c1 = 1, c2 = 1; p = 2; c2 := 2*p;\n"
        "species B in c2, X in c1, Y in c1; B = 0.8; X = 1; Y = 0;\n"
        "R1: X -> Y; 0.5*X\n"
    )
    assert antimony.loadAntimonyString(text) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    sbml_model = doc.getModel()
    b = sbml_model.getSpecies("B")
    b.unsetInitialConcentration()
    b.setInitialAmount(0.8)
    product = sbml_model.getReaction("R1").getProduct(0)
    product.setId("sY")
    product.setConstant(True)
    ia = sbml_model.createInitialAssignment()
    ia.setSymbol("sY")
    ia.setMath(libsbml.parseL3Formula("10*B"))
    model = bngsim.Model.from_sbml_string(libsbml.writeSBMLToString(doc))
    out = bngsim.Simulator(model, "ode").run(t_span=(0, 1.0), n_points=2, rtol=1e-10, atol=1e-12)
    names = [s["name"] for s in model._core.codegen_data()["species"]]
    y = dict(zip(names, out.species[-1], strict=True))
    assert y["Y"] == pytest.approx(2.0 * (1.0 - np.exp(-0.5)), rel=1e-6)


def test_a_rate_of_reads_a_chain_of_reaction_ids() -> None:
    # `q = rateOf(Y)`, where Y's only reaction runs at J1, J1 reads J2, ..., and
    # J20 reads `r := 2*p` (declared 1). No initialAssignment or rule names a
    # reaction id, but the rateOf reads them through Y's law, so the loop must
    # still allow a pass per reaction; q loaded at its declared 7 for 4.
    n = 20
    lines = [
        "compartment c1 = 1; p = 2; r = 1; r := 2*p; q = 0; q = rateOf(Y);",
        "species X in c1, Y in c1; X = 1; Y = 0;",
        *(f"J{i}: X -> X; J{i + 1}" for i in range(1, n)),
        f"J{n}: X -> X; r",
        "R0: -> Y; J1",
    ]
    import antimony

    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString("\n".join(lines)) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    doc.getModel().getParameter("q").setValue(7.0)
    model = bngsim.Model.from_sbml_string(libsbml.writeSBMLToString(doc))
    assert model.get_param("q") == pytest.approx(4.0)


def test_a_stoichiometry_math_reads_a_chain_of_reaction_ids() -> None:
    # SBML Level 2: the stoichiometryMath of Y in R0 reads J1, J1's law reads
    # J2, ..., and J11's reads `r := 2*p` (declared 1), so Y' = 4. Nothing in the
    # loop reads a reaction id, but the stoichiometry does after it, so each law
    # that reads another reaction's id still needs its pass; with a pass per
    # reader only, J1 came out of the loop stale and Y(1) was 1, with nothing
    # raised.
    n = 11
    doc = libsbml.SBMLDocument(2, 4)
    sbml_model = doc.createModel()
    sbml_model.setId("m")
    c = sbml_model.createCompartment()
    c.setId("c")
    c.setSize(1.0)
    for sid, amount, boundary in (("X", 1.0, True), ("Y", 0.0, False)):
        s = sbml_model.createSpecies()
        s.setId(sid)
        s.setCompartment("c")
        s.setInitialAmount(amount)
        s.setBoundaryCondition(boundary)
    for pid, value, constant in (("p", 2.0, True), ("r", 1.0, False)):
        p = sbml_model.createParameter()
        p.setId(pid)
        p.setValue(value)
        p.setConstant(constant)
    rule = sbml_model.createAssignmentRule()
    rule.setVariable("r")
    rule.setMath(libsbml.parseL3Formula("2*p"))
    for i in range(1, n + 1):
        rxn = sbml_model.createReaction()
        rxn.setId(f"J{i}")
        rxn.setReversible(False)
        rxn.createReactant().setSpecies("X")
        rxn.createProduct().setSpecies("X")
        rxn.createKineticLaw().setMath(libsbml.parseL3Formula(f"J{i + 1}" if i < n else "r"))
    rxn = sbml_model.createReaction()
    rxn.setId("R0")
    rxn.setReversible(False)
    product = rxn.createProduct()
    product.setSpecies("Y")
    product.createStoichiometryMath().setMath(libsbml.parseL3Formula("J1"))
    rxn.createKineticLaw().setMath(libsbml.parseL3Formula("1"))
    model = bngsim.Model.from_sbml_string(libsbml.writeSBMLToString(doc))
    out = bngsim.Simulator(model, "ode").run(t_span=(0, 1.0), n_points=2, rtol=1e-10, atol=1e-12)
    names = [s["name"] for s in model._core.codegen_data()["species"]]
    assert dict(zip(names, out.species[-1], strict=True))["Y"] == pytest.approx(4.0)
