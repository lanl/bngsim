"""GH #706: the non-finite-RHS retry clamps only concentrations.

The GH #135 retry re-evaluates a non-finite RHS on a copy of the state with
negative components set to 0, which rescues a sqrt/fractional-power law at a
concentration the predictor pushed a hair below zero. It set EVERY negative
component to 0, so a quantity that is negative by design followed the wrong
ODE for the rest of the run, silently: an SBML rate-rule parameter V = -61
relaxing toward -70 ran away past -287, and a rate-rule X fed by a boundary
species B = -5 froze at -4.32. Now only slots that are concentrations by
construction are clamped.

In every model S is consumed at C*k*sqrt(S), reaches 0 at t = 2 and parks a
hair below it, which is what fires the retry: S(t) = (1 - t/2)^2 up to t = 2,
then 0. V and X do not read S: V(t) = -70 + 9 exp(-0.1 t) and
X(t) = N (1 - exp(-t)) for the value N it relaxes to.

The runs use the finite-difference Jacobian. With S pinned at 0 the analytical
d/dS sqrt(S) is infinite even after the clamp, and that Jacobian freezes S's
Newton correction (issue #867), so S drifts off wherever the step sequence takes
it and the compiled-Jacobian solve can fail outright. The difference quotients
go through the RHS and its retry, which is what this file is about.

What decides "negative by design" for a species is the DECLARED initial value,
not the IC baseline: save_concentrations() can make the baseline a depleted
concentration a hair below 0, and that one must keep the rescue.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

M = "http://www.w3.org/1998/Math/MathML"
_SQRT_SINK = (
    '<reaction id="R1" reversible="false"><listOfReactants>'
    '<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>'
    f'<kineticLaw><math xmlns="{M}"><apply><times/><ci>C</ci><ci>k</ci>'
    "<apply><root/><ci>S</ci></apply></apply></math></kineticLaw></reaction>"
)
_S = (
    '<species id="S" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
)


def _doc(species: str, params: str, rule: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>{_S}{species}</listOfSpecies>
<listOfParameters><parameter id="k" value="1" constant="true"/>{params}</listOfParameters>
<listOfRules>{rule}</listOfRules>
<listOfReactions>{_SQRT_SINK}</listOfReactions>
</model></sbml>"""


v1 = _doc(
    "",
    '<parameter id="V" value="-61" constant="false"/>'
    '<parameter id="g" value="0.1" constant="true"/>'
    '<parameter id="E" value="-70" constant="true"/>',
    f'<rateRule variable="V"><math xmlns="{M}"><apply><times/><ci>g</ci>'
    "<apply><minus/><ci>E</ci><ci>V</ci></apply></apply></math></rateRule>",
)


def _relax_to(species: str, name: str) -> str:
    # A rate-rule X relaxing to the value of `name`, which `species` declares.
    return _doc(
        species,
        '<parameter id="X" value="0" constant="false"/>',
        f'<rateRule variable="X"><math xmlns="{M}">'
        f"<apply><minus/><ci>{name}</ci><ci>X</ci></apply></math></rateRule>",
    )


v2 = _relax_to(
    '<species id="B" compartment="C" initialConcentration="-5" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="true" constant="true"/>',
    "B",
)
# A floating species declared negative, which no reaction touches.
v3 = _relax_to(
    '<species id="N" compartment="C" initialConcentration="-3" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>',
    "N",
)


def _col(r, name):
    if name in r.species_names:
        return np.asarray(r.species[:, list(r.species_names).index(name)])
    return np.asarray(r.observables[name])


def _run(doc, codegen, t_span=(0, 40)):
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(doc), method="ode", codegen=codegen, jacobian="fd"
    )
    return sim, sim.run(t_span=t_span, n_points=int(t_span[1] - t_span[0]) + 1)


def _check_s(r):
    t = np.asarray(r.time)
    s = _col(r, "S")
    assert s.min() < 0.0  # the retry did fire
    np.testing.assert_allclose(s, np.where(t < 2.0, (1.0 - t / 2.0) ** 2, 0.0), atol=1e-6)


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
def test_a_negative_rate_rule_parameter_follows_its_own_ode(codegen):
    _, r = _run(v1, codegen)
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "V"), -70.0 + 9.0 * np.exp(-0.1 * t), rtol=1e-4)
    _check_s(r)


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
@pytest.mark.parametrize("doc, n", [(v2, -5.0), (v3, -3.0)], ids=["boundary", "floating"])
def test_a_species_declared_negative_keeps_its_value(doc, n, codegen):
    _, r = _run(doc, codegen)
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "X"), n * (1.0 - np.exp(-t)), rtol=1e-4, atol=1e-6)
    _check_s(r)


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
def test_a_saved_depleted_concentration_keeps_the_rescue(codegen):
    # Equilibrate, save and restore: S's baseline is now a hair below 0. It is
    # still a concentration, so a restart from it must still be rescued, not
    # fail at the first RHS call; N keeps the value it was declared with.
    sim, r = _run(v3, codegen, t_span=(0, 10))
    assert _col(r, "S")[-1] < 0.0
    sim.save_concentrations()
    sim.restore_concentrations()
    r = sim.run(t_span=(0, 10), n_points=11)
    np.testing.assert_allclose(_col(r, "S"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_col(r, "N"), -3.0)
