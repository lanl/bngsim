"""SBML models SSA/PSA refused although they run exactly.

- A reaction flagged reversible whose law is no difference (``Vm*A/(Km+A)*c``,
  COPASI's default flag) was refused as a forward-minus-reverse net flux it is
  not. It runs as the one channel the same law runs as when flagged
  irreversible: the same trajectory for the same seed.
- In a variable-volume compartment a monomial is corrected exactly, and was
  admitted as ``k*A*c`` but refused as ``0.3*A*c``, ``3*c`` or ``k2*A*c`` with
  ``k2`` an assignment rule over time: only a constant parameter made the
  coefficient. The correction reads the species factors alone.
- A zeroth-order synthesis of an amount-valued species, ``=> H; k``, is
  volume-independent like ``k*H`` and was refused.

Oracle: the ODE's amounts, which the SSA mean must match for these first- and
zeroth-order laws, and libRoadRunner for the ODE (it agrees to 1e-4).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")


def _ant(text):
    return bngsim.Model.from_antimony_string(text)


def _errors(text):
    return {i.code for i in _ant(text).validate_for_ssa() if i.severity == "error"}


FORWARD_ONLY = [
    "compartment c = 2; species A in c = 30; species B in c = 0; Vm = 5; Km = 2;"
    " J1: A {arr} B; Vm*A/(Km+A)*c;",
    "compartment c = 2; species A in c = 30; species B in c = 0;"
    " function mm(s, v, k) v*s/(k+s) end; J1: A {arr} B; mm(A, 5, 2)*c;",
    "compartment c = 1; c' = 0.1; species A in c = 30; species B in c = 0;"
    " J1: A {arr} B; 0.3*A*c;",
]


@pytest.mark.parametrize("text", FORWARD_ONLY)
@pytest.mark.parametrize(("method", "kw"), [("ssa", {}), ("psa", {"poplevel": 10})])
def test_a_reversible_flag_on_a_law_with_no_difference_is_a_label(text, method, kw):
    runs = []
    for arrow in ("=>", "->"):
        m = _ant(text.format(arr=arrow))
        r = bngsim.Simulator(m, method=method, **kw).run(t_span=(0, 10), n_points=11, seed=7)
        runs.append(np.asarray(r.as_roadrunner([s for s in m.species_names if s != "c"])))
    np.testing.assert_array_equal(runs[0], runs[1])


NET_FLUX = (
    "compartment c = 2; species A in c = 30; species B in c = 1; kf = 1; kr = 0.5;"
    " krn = -0.5; kx = 1; E: at time > 1: kx = -1; v := kf*A - kr*B;"
    " function f(a, b) a - b end; function g(a) a end;"
)


@pytest.mark.parametrize(
    "law",
    [
        "5*(A - B/2)/(2+A)",  # a difference
        "5*A/(2+A) + -1*B",  # a negative term
        "f(3*A, B)",  # a difference inside a called function
        "c*v",  # a difference in an assignment rule: the usual BioModels spelling
        "c*(kf*A + krn*B)",  # a negative parameter
        "c*ln(A/B)",
        "c*kf*A*sin(time)",
        "piecewise(kf*A, A > B, krn*B)*c",
        "c*kx*A",  # a parameter an event makes negative
        "g(krn)*A*c",
    ],
)
def test_a_reversible_law_not_proven_non_negative_is_still_refused(law):
    """A net flux written without a minus was admitted at first: `c*v` with
    `v := kf*A - kr*B` ran as one channel, the variance of A(1) 114 for 256."""
    assert "reversible_non_mass_action" in _errors(NET_FLUX + f" J1: A -> B; {law};")


@pytest.mark.parametrize(
    "extra",
    [
        "species S in c = 2; S' = -1;",  # a rate rule drives it negative
        "species S in c = 2; E1: at time >= 1: S = -5;",  # an event writes it
        "species $S in c = -3;",  # it starts negative
        "compartment c2 = 2; c2' = -1; species S in c2 = 1;",
    ],
)
def test_a_species_that_can_go_negative_is_not_taken_for_non_negative(extra):
    text = NET_FLUX + f" {extra} J1: A -> B; c*(kf*A + S);"
    assert _errors(text) & {"reversible_non_mass_action", "variable_compartment_read"}


def test_an_initial_assignment_the_load_cannot_fold_is_read_by_its_math():
    """S is declared 5 with S = delay(-3, 0): the load cannot fold the delay and
    kept the 5, while the run starts S at -3."""
    import antimony
    import libsbml

    antimony.clearPreviousLoads()
    antimony.loadAntimonyString(NET_FLUX + " species $S in c = 5; J1: A -> B; c*(kf*A + S);")
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    assignment = doc.getModel().createInitialAssignment()
    assignment.setSymbol("S")
    assignment.setMath(libsbml.parseL3Formula("delay(-3, 0)"))
    model = bngsim.Model.from_sbml_string(libsbml.writeSBMLToString(doc))
    assert "reversible_non_mass_action" in {i.code for i in model.validate_for_ssa()}


@pytest.mark.parametrize(
    "extra",
    [
        "species S in c = 2; S' = 0.2*S;",  # grows from a non-negative start
        "species S in c = 2; E1: at time >= 1: S = S/2;",
        "species S in c = 2; S' = 0.5;",  # a rate rule that only adds
        "species S in c = 2; E1: at time >= 1: S = 5;",  # an event that writes a positive
    ],
)
def test_a_species_kept_non_negative_by_what_writes_it_is(extra):
    text = NET_FLUX + f" {extra} J1: A -> B; c*(kf*A + S);"
    assert "reversible_non_mass_action" not in _errors(text)


def test_a_kinetic_laws_local_parameter_does_not_reach_a_global_rule():
    """`v := kf*A + krn*B` with the global krn = -0.5 is a difference; a local
    krn = 0.5 in a reaction reading v used to stand in for it there, and the
    result was cached for every later reaction."""
    import antimony
    import libsbml

    antimony.clearPreviousLoads()
    antimony.loadAntimonyString(NET_FLUX + " w := kf*A + krn*B; J0: A -> B; c*w; J1: A -> B; c*w;")
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    law = doc.getModel().getReaction("J0").getKineticLaw()
    local = law.createLocalParameter() if doc.getLevel() >= 3 else law.createParameter()
    local.setId("krn")
    local.setValue(0.5)
    model = bngsim.Model.from_sbml_string(libsbml.writeSBMLToString(doc))
    refused = {
        i.location for i in model.validate_for_ssa() if i.code == "reversible_non_mass_action"
    }
    assert refused == {"reaction:J0", "reaction:J1"}


@pytest.mark.parametrize(
    "law",
    [
        "kf*A*exp(-kr*time)*c",
        "piecewise(kf*A, A > B, kr*B)*c",
        "kf*A*pow(B + 1, krn)*c",  # a negative power of a non-negative base
        "mm(A, 5, 2)*c",
    ],
)
def test_a_reversible_law_proven_non_negative_is_admitted(law):
    text = NET_FLUX + f" function mm(s, v, k) v*s/(k+s) end; J1: A -> B; {law};"
    assert "reversible_non_mass_action" not in _errors(text)


GROW = "compartment c = 1; c' = 0.1;"
RESIZE = "compartment c = 1; E1: at time >= 2: c = 2.5;"
MONOMIALS = [
    "species A in c = 40; species B in c = 0; J1: A => B; 0.3*A*c;",
    "species A in c = 40; species B in c = 0; kr := 0.3*(1 + 0.5*sin(time)); J1: A => B; kr*A*c;",
    "species B in c = 0; J1: => B; 3*c;",
    "substanceOnly species H in c = 0; k = 3; J1: => H; k;",
    "substanceOnly species G in c = 40; substanceOnly species H in c = 0; J1: G => H; 0.3*G;",
]


@pytest.mark.parametrize("vol", [GROW, RESIZE], ids=["rate-rule", "event-resize"])
@pytest.mark.parametrize("body", MONOMIALS)
def test_a_variable_volume_monomial_runs_with_the_odes_mean(vol, body):
    text = vol + body
    assert not _errors(text)
    m = _ant(text)
    names = [s for s in m.species_names if s != "c"]
    ode = np.asarray(
        bngsim.Simulator(m)
        .run(t_span=(0, 6), n_points=4, rtol=1e-10, atol=1e-12)
        .as_roadrunner(names)
    )
    sim = bngsim.Simulator(_ant(text), method="ssa")
    reps = 800
    runs = []
    for i in range(reps):
        sim.model.reset()
        runs.append(
            np.asarray(sim.run(t_span=(0, 6), n_points=4, seed=100 + i).as_roadrunner(names))
        )
    x = np.array(runs)
    se = x.std(0, ddof=1) / np.sqrt(reps)
    assert np.all(np.abs(x.mean(0) - ode) <= 4.5 * se + 1e-9)


@pytest.mark.parametrize(
    "law",
    [
        "A => B; kB*A*c",  # the coefficient reads a species: not a monomial
        "A => B; 5*A/(2+A)*c",  # no monomial at all
    ],
)
def test_a_variable_volume_law_that_is_no_monomial_is_still_refused(law):
    text = GROW + f"species A in c = 40; species B in c = 1; kB := 0.3*B; J1: {law};"
    assert "varvol_non_mass_action" in _errors(text)
