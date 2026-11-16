"""AMICI's amount states are compared as concentrations.

AMICI keeps a species with ``hasOnlySubstanceUnits="true"`` as an amount in
``rdata.x``; bngsim and RoadRunner (``[id]``) report every species as a
concentration. ``amici_ode`` compared the two as they came, so such a species in
a compartment whose size is not 1 read as a DIFF off by exactly that size: 12 of
the 26 AMICI DIFFs in the BioModels coverage sweep (BIOMD0000000197, 305, 407,
MODEL1102210001, ...) were this alone. ``amount_states_as_concentrations`` divides
each amount state by its compartment's size: constant, integrated by AMICI (a
rate rule), or set by an assignment rule. A size it cannot read for certain
(a compartment a rule or event changes that AMICI reports no column for, a
size that is not finite and positive) drops the species rather than guess.
The sensitivity job converts ``x`` and ``sx`` the same way, from the SBML.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libsbml")

import _amici_common as ac  # noqa: E402

MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'
TIME = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>'

# A: amount species in a constant compartment of size 2, decaying at rate k.
#    concentration = 2 exp(-k t)
# B: concentration species in the same compartment: left as it is, exp(-k t).
# C: amount species (constant amount 3) in a compartment with dV/dt = 1, V(0) = 1.
#    concentration = 3 / (1 + t)
# D: amount species (constant amount 5) in a compartment W = 2 + time.
#    concentration = 5 / (2 + t)
# E: amount species (constant amount 6) in a constant compartment declared with
#    size 1 and sized by an initial assignment, cE = 4 k = 2. concentration = 3
SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="amount_species">
    <listOfCompartments>
      <compartment id="c" size="2" constant="true" spatialDimensions="3"/>
      <compartment id="V" size="1" constant="false" spatialDimensions="3"/>
      <compartment id="W" size="2" constant="false" spatialDimensions="3"/>
      <compartment id="cE" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="4" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="C" compartment="V" initialAmount="3" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="D" compartment="W" initialAmount="5" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="E" compartment="cE" initialAmount="6" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="cE"><math {MATH}><apply><times/><cn> 4 </cn><ci> k </ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="V"><math {MATH}><cn> 1 </cn></math></rateRule>
      <assignmentRule variable="W"><math {MATH}><apply><plus/><cn> 2 </cn>{TIME}</apply></math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="decayA" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math {MATH}><apply><times/><ci> k </ci><ci> A </ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="decayB" reversible="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math {MATH}><apply><times/><ci> c </ci><ci> k </ci><ci> B </ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

GRID = (0.0, 4.0, 9)


def test_only_amount_species_are_listed():
    assert ac.amount_species_compartments(SBML) == {"A": "c", "C": "V", "D": "W", "E": "cE"}


def _sbml(text=SBML):
    import libsbml

    return libsbml.readSBMLFromString(text).getModel()


def _constant(sbml):
    return ac._rc._effectively_constant(sbml)


def test_a_compartment_sized_by_an_initial_assignment_is_evaluated():
    sbml = _sbml()
    k = _constant(sbml)
    assert ac._initial_assignment_value("cE", sbml, {"k": 0.5}.get, k) == 2.0
    # Not a guess: an id it cannot value gives None.
    assert ac._initial_assignment_value("cE", sbml, {}.get, k) is None
    assert ac._initial_assignment_value("c", sbml, {"k": 0.5}.get, k) is None


@pytest.mark.parametrize(
    ("math", "want"),
    [
        ("<apply><power/><cn> 2 </cn><cn> 3 </cn></apply>", 8.0),
        ("<apply><times/><pi/><cn> 2 </cn></apply>", 2.0 * np.pi),
        ("<apply><minus/><ci> k </ci></apply>", -0.5),
        # The clock, a species, a compartment a rule changes: no constant size.
        (f"<apply><plus/><cn> 1 </cn>{TIME}</apply>", None),
        ("<apply><times/><cn> 2 </cn><ci> A </ci></apply>", None),
        ("<apply><times/><cn> 2 </cn><ci> V </ci></apply>", None),
        ("<apply><power/><cn> -8 </cn><cn> 0.5 </cn></apply>", None),
    ],
)
def test_initial_assignment_forms(math, want):
    text = SBML.replace("<apply><times/><cn> 4 </cn><ci> k </ci></apply>", math)
    sbml = _sbml(text)
    got = ac._initial_assignment_value("cE", sbml, ac.sbml_value_of(sbml), _constant(sbml))
    assert got == pytest.approx(want) if want is not None else got is None


def test_the_ids_a_size_reads_are_collected():
    sbml = _sbml()
    deps: set[str] = set()
    assert ac.constant_size("cE", sbml, ac.sbml_value_of(sbml), _constant(sbml), deps) == 2.0
    assert deps == {"cE", "k"}


EVENT_SBML = SBML.replace(
    "</listOfReactions>",
    """</listOfReactions>
    <listOfEvents>
      <event id="grow" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true"><math MATHNS><apply><geq/>TIMESYM<cn> 1 </cn></apply></math></trigger>
        <listOfEventAssignments>
          <eventAssignment variable="c"><math MATHNS><cn> 4 </cn></math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>""".replace("MATHNS", MATH).replace("TIMESYM", TIME),
).replace(
    '<compartment id="c" size="2" constant="true"', '<compartment id="c" size="2" constant="false"'
)


class _FakeModel:
    def __init__(self, free=(), fixed=(), exprs=()):
        self._free, self._fixed, self._exprs = dict(free), dict(fixed), list(exprs)

    def get_free_parameter_ids(self):
        return list(self._free)

    def get_unscaled_parameters(self):
        return list(self._free.values())

    def get_fixed_parameter_ids(self):
        return list(self._fixed)

    def get_fixed_parameters(self):
        return list(self._fixed.values())

    def get_expression_ids(self):
        return list(self._exprs)


class _FakeRdata:
    def __init__(self, state_ids=(), w=None):
        self.state_ids = list(state_ids)
        self.w = np.empty((0, 0)) if w is None else w


def test_a_changed_compartment_with_no_amici_column_is_not_given_its_declared_size():
    """W (an assignment rule) with no usable rdata.w, V (a rate rule) not among
    AMICI's states, and c (an event target) held as a parameter: each would be
    its declared size at every time, a confident wrong divisor."""
    x = np.ones((3, 1))
    model = _FakeModel(fixed={"k": 0.5, "c": 2.0})
    volume_of = ac.amici_volume_of(model, _FakeRdata(), EVENT_SBML, x)
    assert volume_of("W") is None
    assert volume_of("V") is None
    assert volume_of("c") is None
    # A compartment nothing changes is still sized, and an IA-sized one too.
    assert ac.amici_volume_of(model, _FakeRdata(), SBML, x)("c") == 2.0
    assert volume_of("cE") == 2.0


def test_a_changed_compartment_amici_reports_is_read_from_its_column():
    t = np.linspace(*GRID)
    x = np.column_stack([np.ones_like(t), 1.0 + t])
    w = (2.0 + t)[:, None]
    volume_of = ac.amici_volume_of(
        _FakeModel(fixed={"k": 0.5}, exprs=["W"]), _FakeRdata(["A", "V"], w), SBML, x
    )
    np.testing.assert_array_equal(volume_of("V"), 1.0 + t)
    np.testing.assert_array_equal(volume_of("W"), 2.0 + t)


def test_amount_states_are_divided_by_their_compartments_size():
    t = np.linspace(*GRID)
    x = np.column_stack(
        [4.0 * np.ones_like(t), np.ones_like(t), 3.0 * np.ones_like(t), 7.0 * np.ones_like(t)]
    )
    names = ["A", "B", "C", "V"]
    sizes = {"c": 2.0, "V": 1.0 + t}
    amounts = ac.amount_species_compartments(SBML)
    out, kept, dropped = ac.amount_states_as_concentrations(x, names, amounts, sizes.get)
    assert kept == names and dropped == []
    np.testing.assert_array_equal(out[:, 0], 2.0)  # A: amount 4 in size 2
    np.testing.assert_array_equal(out[:, 1], 1.0)  # B: a concentration, untouched
    np.testing.assert_array_equal(out[:, 2], 3.0 / (1.0 + t))  # C: per-time size
    np.testing.assert_array_equal(out[:, 3], 7.0)  # V: a compartment, not a species
    np.testing.assert_array_equal(x[:, 0], 4.0)  # the input is not modified


def test_an_amount_species_with_no_known_size_is_dropped_not_compared():
    x = np.ones((3, 2))
    amounts = ac.amount_species_compartments(SBML)
    out, kept, dropped = ac.amount_states_as_concentrations(
        x, ["A", "B"], amounts, lambda cid: None
    )
    assert kept == ["B"] and dropped == ["A"] and out.shape == (3, 1)


def _raises(cid):
    raise RuntimeError("no size")


@pytest.mark.parametrize(
    "volume_of",
    [
        {"c": 0.0}.get,
        {"c": -1.0}.get,
        {"c": float("nan")}.get,
        lambda cid: np.array([1.0, 0.0, 1.0]),
        _raises,
    ],
    ids=["zero", "negative", "nan", "zero-at-one-time", "raises"],
)
def test_a_size_that_is_not_finite_and_positive_drops_the_species(volume_of):
    x = np.ones((3, 2))
    amounts = ac.amount_species_compartments(SBML)
    out, kept, dropped = ac.amount_states_as_concentrations(x, ["A", "B"], amounts, volume_of)
    assert kept == ["B"] and dropped == ["A"] and np.isfinite(out).all()


def test_amicis_renamed_ids_are_matched_to_their_sbml_species():
    sbml = (
        SBML.replace('id="A"', 'id="x"')
        .replace("<ci> A </ci>", "<ci> x </ci>")
        .replace('species="A"', 'species="x"')
    )
    out, kept, _ = ac.amount_states_as_concentrations(
        np.full((2, 1), 4.0), ["amici_x"], ac.amount_species_compartments(sbml), {"c": 2.0}.get
    )
    assert kept == ["amici_x"]
    np.testing.assert_array_equal(out[:, 0], 2.0)


def test_amici_ode_reports_every_species_as_a_concentration():
    """End to end through AMICI: fails on the adapter that compared amounts."""
    pytest.importorskip("amici")
    t, x, names, timing = ac.amici_ode(SBML, *GRID, 1e-10, 1e-12)
    col = {n: x[:, i] for i, n in enumerate(ac.normalize_amici_names(list(names)))}
    k = 0.5
    np.testing.assert_allclose(col["A"], 2.0 * np.exp(-k * t), rtol=1e-6)
    np.testing.assert_allclose(col["B"], np.exp(-k * t), rtol=1e-6)
    np.testing.assert_allclose(col["C"], 3.0 / (1.0 + t), rtol=1e-6)
    np.testing.assert_allclose(col["D"], 5.0 / (2.0 + t), rtol=1e-6)
    np.testing.assert_allclose(col["E"], 3.0, rtol=1e-12)
    assert timing["amount_states_dropped"] == []


def test_the_sensitivity_job_converts_x_and_sx_from_the_sbml():
    """The same basis for the sensitivity job, x and sx alike, from the SBML
    alone so a replayed reference converts as a live run does. V (a rate rule)
    and W (an assignment rule) change, so C and D are dropped; E's size reads k,
    so with k sensitized it is dropped too."""
    import _amici_sens as asens

    names = ["A", "B", "C", "D", "E"]
    x = np.tile([4.0, 1.0, 3.0, 5.0, 6.0], (2, 1))
    sx = np.tile(np.array([[8.0], [1.0], [1.0], [1.0], [12.0]]), (2, 1, 1))
    ox, osx, kept, dropped = asens.amount_states_as_concentrations(x, sx, names, SBML, ["p"])
    assert kept == ["A", "B", "E"] and dropped == ["C", "D"]
    np.testing.assert_array_equal(ox[0], [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(osx[0, :, 0], [4.0, 1.0, 6.0])
    np.testing.assert_array_equal(x[0], [4.0, 1.0, 3.0, 5.0, 6.0])  # not modified

    _, _, kept, dropped = asens.amount_states_as_concentrations(x, sx, names, SBML, ["k"])
    assert kept == ["A", "B"] and dropped == ["C", "D", "E"]
    _, _, kept, dropped = asens.amount_states_as_concentrations(x, sx, names, SBML, ["amici_c"])
    assert "A" in dropped
