"""Parameters that an SBML document's model was built with as numbers
(issues #695, #696, and the residue of #313).

An SBML document may write a compartment's size, a stoichiometry or a
conversion factor over its parameters, or set a parameter's initial value by an
initialAssignment that cannot be kept symbolic. bngsim evaluates each once, at
load. A parameter that reaches the model only that way stayed an ordinary
writable parameter that nothing read: ``set_param`` took the value and moved
nothing, and its forward-sensitivity column was an exact 0 at every species and
time, with nothing raised or logged (the #313 residue was named in one warning,
at load).

- ``c = 2*p`` with ``S -> `` at ``k*S`` in ``c``: ``set_param("p", 2)`` left
  [S](1) at 0.6065 where a rebuild gives 0.7788, and d[S]/dp was 0 for 0.3033.
- a stoichiometry ``2*f``: P(1) stayed 12.64 where a rebuild at f = 3 gives
  37.93, and dP/df was 0 for 12.64.

The write and the column are refused now, by name, with what the parameter was
folded into. Expected values are closed forms.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest

SIZED = "compartment c; p = 1; c = 2*p; species S in c; S = 1; k = 1;\nJ: S -> ; k*S\n"

L3 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m"{model_attrs}>
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="P" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"{species_attrs}/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
      <parameter id="z" value="1" constant="true"/>
      <parameter id="cf" value="1" constant="true"/>
    </listOfParameters>
    {assignments}
    <listOfReactions>
      <reaction id="J" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference {reference_id}species="P" stoichiometry="2" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""
ONTO_THE_REFERENCE = (
    '<listOfInitialAssignments><initialAssignment symbol="st">'
    '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>z</ci><cn>1</cn></apply>'
    "</math></initialAssignment></listOfInitialAssignments>"
)
L2 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10"/>
      <species id="P" compartment="c" initialConcentration="0"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="1"/><parameter id="f" value="1"/></listOfParameters>
    <listOfReactions>
      <reaction id="J" reversible="false">
        <listOfReactants><speciesReference species="A"/></listOfReactants>
        <listOfProducts><speciesReference species="P"><stoichiometryMath>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><cn>2</cn><ci>f</ci></apply></math>
        </stoichiometryMath></speciesReference></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _l3(model_attrs="", species_attrs="", assignments="", reference_id=""):
    return bngsim.Model.from_sbml_string(
        L3.format(
            model_attrs=model_attrs,
            species_attrs=species_attrs,
            assignments=assignments,
            reference_id=reference_id,
        )
    )


SHAPES = {
    "a-size-from-a-parameter": (lambda: bngsim.Model.from_antimony_string(SIZED), ["p"], "'c'"),
    "a-size-through-a-derived-parameter": (
        lambda: bngsim.Model.from_antimony_string(SIZED.replace("c = 2*p", "q = 3*p; c = 2*q")),
        ["p", "q"],
        "'c'",
    ),
    "a-size-from-a-size": (
        lambda: bngsim.Model.from_antimony_string(
            "compartment cell, lumen; cell = 2; vr = 4; lumen = cell/vr; species S in lumen;"
            " S = 1; k = 1;\nJ: S -> ; k*S\n"
        ),
        ["cell", "vr"],
        "'lumen'",
    ),
    "a-stoichiometry-that-is-an-id": (
        lambda: bngsim.Model.from_antimony_string(
            "species A, P; A = 10; P = 0; f = 2; k = 1;\nJ: A -> f P; k*A\n"
        ),
        ["f"],
        "stoichiometry of 'P' in reaction 'J'",
    ),
    "a-stoichiometry-assigned-at-the-start": (
        lambda: _l3(assignments=ONTO_THE_REFERENCE, reference_id='id="st" '),
        ["z", "st"],
        "stoichiometry of 'P' in reaction 'J'",
    ),
    "stoichiometry-math": (
        lambda: bngsim.Model.from_sbml_string(L2),
        ["f"],
        "stoichiometry of 'P' in reaction 'J'",
    ),
    "the-model-s-conversion-factor": (
        lambda: _l3(model_attrs=' conversionFactor="cf"'),
        ["cf"],
        "conversionFactor",
    ),
    "a-species-conversion-factor": (
        lambda: _l3(species_attrs=' conversionFactor="cf"'),
        ["cf"],
        "conversionFactor of species 'P'",
    ),
    "an-initial-value-that-reads-a-rate": (
        lambda: bngsim.Model.from_antimony_string(
            "species A; A = 10; k1 = 2; k2 = k1*J0; J0: A -> ; k1*A; J1: -> A; k2\n"
        ),
        ["k1"],
        "initial value of 'k2'",
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_parameter_folded_at_load_is_named(shape):
    load, frozen, _what = SHAPES[shape]
    assert load().frozen_params == frozen


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_write_that_changes_one_is_refused(shape):
    """It took the value and moved nothing it was folded into."""
    load, frozen, what = SHAPES[shape]
    for name in frozen:
        model = load()
        before = model.get_param(name)
        with pytest.raises(bngsim.ParameterError, match=r"#695, #696") as caught:
            model.set_param(name, before + 1.0)
        assert repr(name) in str(caught.value)
        assert model.get_param(name) == before
        # A write of the value it holds is no change.
        model.set_param(name, before)
    model = load()
    with pytest.raises(bngsim.ParameterError) as caught:
        model.set_param(frozen[0], model.get_param(frozen[0]) * 2.0 + 1.0)
    assert what in str(caught.value)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_sensitivity_column_for_one_is_refused(shape):
    """It was an exact 0 at every species and time."""
    load, frozen, what = SHAPES[shape]
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"#695, #696") as caught:
        bngsim.Simulator(load(), method="ode", sensitivity_params=[frozen[-1], "k"])
    assert what in str(caught.value)
    sim = bngsim.Simulator(load(), method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"#695, #696"):
        sim.compute_all_sensitivities(t_span=(0.0, 1.0), n_points=3, params=[frozen[-1]])


def _end(model, species, **kw):
    run = bngsim.Simulator(model, method="ode", **kw).run(
        t_span=(0.0, 1.0), n_points=3, rtol=1e-10, atol=1e-12
    )
    column = list(run.species_names).index(species)
    return run, column


def test_the_size_itself_is_written_and_differentiated_as_it_was():
    """Control. ``c`` is a parameter the model reads: a write to it is a
    rebuild at that size, and its column and the rate constant's are right.
    [S](t) = exp(−k·t/c)."""
    model = bngsim.Model.from_antimony_string(SIZED)
    run, s = _end(model, "S", sensitivity_params=["c", "k"])
    assert np.asarray(run.species)[-1, s] == pytest.approx(math.exp(-0.5), rel=1e-8)
    np.testing.assert_allclose(
        np.asarray(run.sensitivities)[-1, s], [math.exp(-0.5) / 4, -math.exp(-0.5) / 2], rtol=1e-6
    )
    model = bngsim.Model.from_antimony_string(SIZED)
    model.set_param("c", 4.0)
    run, s = _end(model, "S")
    assert np.asarray(run.species)[-1, s] == pytest.approx(math.exp(-0.25), rel=1e-8)


def test_every_column_at_once_leaves_one_out_and_says_so():
    """``compute_all_sensitivities()`` with no list is every column that can
    be computed: p was one of them, with a column of 0."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(SIZED), method="ode")
    with pytest.warns(UserWarning, match=r"skipping 1 parameter.*'p'.*#695, #696"):
        result = sim.compute_all_sensitivities(
            t_span=(0.0, 1.0), n_points=3, rtol=1e-10, atol=1e-12
        )
    assert list(result.sensitivity_params) == ["c", "k"]
    s = list(result.species_names).index("S")
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[-1, s],
        [math.exp(-0.5) / 4, -math.exp(-0.5) / 2],
        rtol=1e-6,
    )


def test_several_writes_at_once_are_refused_whole():
    """``set_params`` writes nothing where one entry is refused."""
    model = bngsim.Model.from_antimony_string(SIZED)
    with pytest.raises(bngsim.ParameterError, match="'p'"):
        model.set_params({"k": 3.0, "p": 2.0})
    assert model.get_param("k") == 1.0
    # The whole vector, as it stands, goes back in.
    model.set_params({name: model.get_param(name) for name in model.param_names})


def test_a_clone_and_a_batch_row_are_refused_too():
    model = bngsim.Model.from_antimony_string(SIZED)
    clone = model.clone()
    assert clone.frozen_params == ["p"]
    with pytest.raises(bngsim.ParameterError, match="'p'"):
        clone.set_param("p", 2.0)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.ParameterError, match="'p'"):
        sim.run_batch(params=[{"p": 2.0}], t_span=(0.0, 1.0), n_points=3)
    rows = sim.run_batch(params=[{"k": 2.0}], t_span=(0.0, 1.0), n_points=3)
    s = list(rows[0].species_names).index("S")
    assert np.asarray(rows[0].species)[-1, s] == pytest.approx(math.exp(-1.0), rel=1e-5)


def test_a_steady_state_column_for_one_is_refused():
    text = "compartment c; p = 1; c = 2*p; species S in c; S = 1; k = 1; kp = 3;\n"
    text += "J0: -> S; c*kp\nJ: S -> ; c*k*S\n"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"'p'.*#695, #696"):
        sim.steady_state(sensitivity_params=["p"])
    out = sim.steady_state(sensitivity_params=["k"], tol=1e-12)
    s = list(out.species_names).index("S")
    assert np.asarray(out.sensitivity)[s, 0] == pytest.approx(-3.0, rel=1e-6)


@pytest.mark.parametrize(
    "text",
    [
        "species A; A = 10; k1 = 2; k2 = 2*k1; J0: A -> ; k2*A\n",
        "compartment c; c = 2; species S in c; S = 1; k = 1;\nJ: S -> ; k*S\n",
        "species A, P; A = 10; P = 0; k = 1;\nJ: A -> 2 P; k*A\n",
    ],
    ids=["an-assignment-that-is-kept-symbolic", "a-size-that-is-a-number", "a-number"],
)
def test_a_model_with_nothing_folded_freezes_nothing(text):
    """Control. An initial assignment over parameters alone is kept as an
    expression of them, and a write to what it reads moves it."""
    model = bngsim.Model.from_antimony_string(text)
    assert model.frozen_params == []
    for name in model.primary_param_names:
        model.set_param(name, model.get_param(name) * 1.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bngsim.Simulator(model, method="ode").compute_all_sensitivities(
            t_span=(0.0, 1.0), n_points=3
        )
    assert not [w for w in caught if "#695" in str(w.message)]


def test_a_stoichiometry_that_a_rule_sets_is_not_frozen():
    """Control. A stoichiometry that moves in time is kept symbolic, and what
    it reads is a parameter the model reads."""
    text = "species A, P; A = 10; P = 0; k = 1; g = 2; f := g*(1 + time);\nJ: A -> f P; k*A\n"
    model = bngsim.Model.from_antimony_string(text)
    assert model.frozen_params == []
    run, p = _end(model, "P")
    before = float(np.asarray(run.species)[-1, p])
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("g", 4.0)
    run, p = _end(model, "P")
    assert float(np.asarray(run.species)[-1, p]) == pytest.approx(2.0 * before, rel=1e-6)


def test_a_net_model_freezes_nothing(tmp_path):
    """Control."""
    path = tmp_path / "ab.net"
    path.write_text(
        "begin parameters\n 1 kf 1\nend parameters\n"
        "begin species\n 1 A() 3\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 kf\nend reactions\n"
    )
    assert bngsim.Model.from_net(str(path)).frozen_params == []
