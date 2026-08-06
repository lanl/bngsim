"""Issue #170 stage 2 — the compartment volume the *generated C* used to bake.

Stage 1 (:mod:`test_compartment_size_live`) put every interpreted use of a
compartment size back on the parameter. Two shapes were left refused, because the
emitted C still carried the volume as a load-time literal:

* an **amount-valued** (``hasOnlySubstanceUnits``) species — ``× V_c`` appears in
  the rate's amount factor, in every observable weight, in the ``∂/∂x`` chain
  factor and in the ``rateOf`` accessor; and
* a **cross-compartment** reaction — the ``static const double inv_vf[]``
  reciprocal table and the per-row Jacobian / ``∂f/∂p`` divisors.

Honoring the write there would have honored it with codegen off and half-applied
it with codegen on, which is exactly the invisible path-dependence #164 refused
over. This module is the other half of the fix, and it tests the thing that makes
it safe rather than just the answer: **a write no longer changes the emitted
source**, so the ``.so`` a model was loaded with stays valid after one and the two
backends cannot disagree. The write-then-run assertions come first, that
invariant second, and the arithmetic-shape invariants last — those are what keep
"bit-identical at the nominal point" true rather than approximately true.

The failure mode the ordering matters for: writing the volume *before*
constructing the Simulator regenerates the source, so a baked literal is still
correct there. The write has to land after generation — which is what
``parameter_scan`` and any post-construction ``set_param`` do.
"""

from __future__ import annotations

import hashlib

import bngsim
import numpy as np
import pytest
from bngsim import _codegen

# ── Fixtures ────────────────────────────────────────────────────────────────

ONE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="one">
    <listOfCompartments>
      <compartment id="C" size="{v}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="100" hasOnlySubstanceUnits="{hosu}" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="{hosu}" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">{law}</math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# Two compartments of DIFFERENT size spanned by one reaction: the classifier
# cannot fold that into a single scalar, so it takes the per-species volume
# scaling path — the `inv_vf` half of stage 2.
XCOMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp">
    <listOfCompartments>
      <compartment id="C1" size="{v}" constant="true"/>
      <compartment id="C2" size="5" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C2" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="kb" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="transport" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C1</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="degB" reversible="false" fast="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C2</ci><apply><times/><ci>kb</ci><ci>B</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# Mass-action: the amount factor rides the Elementary rate and the Elementary
# Jacobian's `amount_factor`.
L_CkA = "<apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>"
# A + A -> ... would give the amount factor two occurrences; this gives it one per
# distinct reactant instead (see the two-factor test, which builds its own source).
# Not mass-action: takes the Functional path, so the amount factor enters through
# the per-species ∂func/∂x chain rule rather than the scalar rate.
L_SAT = (
    "<apply><divide/><apply><times/><ci>k</ci><ci>A</ci></apply>"
    "<apply><plus/><ci>C</ci><ci>A</ci></apply></apply>"
)

T_SPAN = (0.0, 20.0)
N_POINTS = 5


def _one(v, law=L_CkA, hosu="true"):
    return ONE.format(v=repr(float(v)), law=law, hosu=hosu)


def _hosu_elementary(v):
    return _one(v)


def _hosu_functional(v):
    return _one(v, law=L_SAT)


def _xcomp(v):
    return XCOMP.format(v=repr(float(v)))


# (id, source builder, the compartment written)
SHAPES = [
    ("hosu_elementary", _hosu_elementary, "C"),
    ("hosu_functional", _hosu_functional, "C"),
    ("cross_compartment", _xcomp, "C1"),
]

V_LOAD, V_NEW = 1.0, 3.0


def _sim(model, codegen):
    return bngsim.Simulator(model, method="ode", codegen=codegen)


def _traj(sim):
    return np.asarray(sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14).species)


def _sens(sim):
    return np.asarray(
        sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14).sensitivities
    )


CODEGEN = [pytest.param(None, id="interpreted"), pytest.param(True, id="codegen")]
SHAPE_PARAMS = [pytest.param(b, c, id=i) for i, b, c in SHAPES]


# ── The two rows stage 1 refused ────────────────────────────────────────────


@pytest.mark.parametrize("codegen", CODEGEN)
@pytest.mark.parametrize(("build", "comp"), SHAPE_PARAMS)
def test_the_write_is_honored_and_reproduces_a_rebuild_exactly(build, comp, codegen):
    """The stage-2 acceptance row, on both backends. Exact, not ``allclose``: the
    volume nearly cancels on some of these shapes, so an approximate assertion
    would pass on the refusal-era behaviour too."""
    rebuilt = _traj(_sim(bngsim.Model.from_sbml_string(build(V_NEW)), codegen))
    m = bngsim.Model.from_sbml_string(build(V_LOAD))
    m.set_param(comp, V_NEW)
    assert np.array_equal(rebuilt, _traj(_sim(m, codegen)))


@pytest.mark.parametrize(("build", "comp"), SHAPE_PARAMS)
def test_these_shapes_really_do_move_with_the_volume(build, comp):
    """Guard on the guard: if a shape's trajectory were V-invariant, the test
    above would pass without the volume reaching anything."""
    lo = _traj(_sim(bngsim.Model.from_sbml_string(build(V_LOAD)), None))
    hi = _traj(_sim(bngsim.Model.from_sbml_string(build(V_NEW)), None))
    assert not np.allclose(lo, hi)


@pytest.mark.parametrize(("build", "comp"), SHAPE_PARAMS)
def test_nothing_about_these_shapes_is_refused_any_more(build, comp):
    m = bngsim.Model.from_sbml_string(build(V_LOAD))
    assert comp in m.compartment_size_params
    assert m.unwritable_compartment_size_params == []


# ── Why it is safe: the source stopped depending on the volume ──────────────

EMITTERS = [
    ("rhs", _codegen.generate_rhs_from_model),
    ("jac", _codegen.generate_jacobian_from_model),
    ("outputs", _codegen.generate_outputs_from_model),
    ("sens", _codegen.generate_sens_from_model),
    ("output_sens", _codegen.generate_output_sens_from_model),
]


@pytest.mark.parametrize(("emitter", "emit"), [pytest.param(n, f, id=n) for n, f in EMITTERS])
@pytest.mark.parametrize(("build", "comp"), SHAPE_PARAMS)
def test_a_write_does_not_change_the_emitted_source(build, comp, emitter, emit):
    """The invariant that makes an already-compiled ``.so`` valid after a write, and
    the one a surviving volume literal breaks: emitting from the same loaded model
    before and after ``set_param`` on its compartment must give identical text.
    ``codegen_data()`` reports the *post-write* volume, so a baked ``V_c`` moves the
    source here — and since the ``.so`` cache key is a hash of that text, the write
    would silently recompile rather than being honored by the loaded binary.

    Note what is deliberately NOT asserted: that two *loads* at different sizes
    emit the same text. They do not, by stage 1's design — the Elementary
    ``stat_factor`` still carries ``1/V_load`` and a write is routed through the
    rate parameter's ``k·(C/_V0_C)^n`` ratio instead, so ``load(V=3)`` and
    ``load(V=1); set_param(C, 3)`` have different sources and equal runtime values.
    Asserting source identity across loads would be asserting stage 1 away."""
    m = bngsim.Model.from_sbml_string(build(V_LOAD))
    before = emit(m)
    m.set_param(comp, V_NEW)
    after = emit(m)
    if before is None or after is None:
        # An emitter that declines a model must decline it either side of a write:
        # a write-dependent decline is the same defect one level up.
        assert before is None and after is None, f"{emitter} declines on one side only"
        return
    assert (
        hashlib.sha256(before.encode()).hexdigest() == hashlib.sha256(after.encode()).hexdigest()
    )


@pytest.mark.parametrize("codegen", CODEGEN)
@pytest.mark.parametrize(("build", "comp"), SHAPE_PARAMS)
def test_a_write_after_the_source_is_generated_still_lands(build, comp, codegen):
    """The order the refusal actually existed for. ``vtable``-style tests write
    before constructing the Simulator, which regenerates the source and hides a
    baked literal; ``parameter_scan`` and a post-construction ``set_param`` do
    not."""
    rebuilt = _traj(_sim(bngsim.Model.from_sbml_string(build(V_NEW)), codegen))
    m = bngsim.Model.from_sbml_string(build(V_LOAD))
    sim = _sim(m, codegen)  # generates + compiles at V_LOAD
    m.set_param(comp, V_NEW)
    assert np.array_equal(rebuilt, _traj(sim))


@pytest.mark.parametrize("codegen", CODEGEN)
@pytest.mark.parametrize(("build", "comp"), SHAPE_PARAMS)
def test_a_write_after_generation_still_lands_in_the_sensitivity(build, comp, codegen):
    """``J·yS`` is part of the forward-sensitivity RHS, not a preconditioner, so a
    volume frozen in the per-species ``∂func/∂x`` coefficient is a wrong *answer*.
    On ``hosu_functional`` this was off by 2.3e-05 at a 3x write and by 100% at a
    400x one, identically on both backends — the fold lives in the shared symbolic
    core, so the codegen half alone did not fix it."""
    rebuilt = _sens(
        bngsim.Simulator(
            bngsim.Model.from_sbml_string(build(V_NEW)),
            method="ode",
            codegen=codegen,
            sensitivity_params=["k"],
        )
    )
    m = bngsim.Model.from_sbml_string(build(V_LOAD))
    sim = bngsim.Simulator(m, method="ode", codegen=codegen, sensitivity_params=["k"])
    m.set_param(comp, V_NEW)
    got = _sens(sim)
    assert np.abs(rebuilt).max() > 0.0
    assert np.array_equal(rebuilt, got)


def test_a_harsh_write_lands_too():
    """A 400x shrink, where a stale Jacobian showed up as a 3x step-count
    inflation as well as a wrong number."""
    rebuilt = _traj(_sim(bngsim.Model.from_sbml_string(_hosu_functional(0.01)), None))
    m = bngsim.Model.from_sbml_string(_hosu_functional(4.0))
    sim = _sim(m, None)
    m.set_param("C", 0.01)
    assert np.array_equal(rebuilt, _traj(sim))


# ── The arithmetic shapes that keep it bit-identical ────────────────────────


def _rhs_src(model):
    return _codegen.generate_rhs_from_model(model)


def test_the_reciprocal_table_stays_a_reciprocal():
    """``inv_vf`` held 1/V_c, so a cross-compartment row was ONE MULTIPLY by a
    reciprocal. ``rate / p[k]`` would be a different operation: ``x*(1/V)`` and
    ``x/V`` differ in the last digit, and only the first reproduces the value the
    table produced. This is the single substitution most likely to be 'simplified'
    later, so it is pinned in the text."""
    src = _rhs_src(bngsim.Model.from_sbml_string(_xcomp(3.0)))
    assert "(1.0 / p[" in src, src
    assert "rate / p[" not in src


def test_a_multi_factor_amount_product_is_one_parenthesised_group():
    """Two amount-valued reactants in one reaction: the pre-#170 emission folded
    ``V_a·V_b`` into a single literal, so the factors have to go out as one
    parenthesised product. Emitted loose, C's left-associative ``*`` would
    re-associate them against the rate constant and move the last digit."""
    src = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="two">
    <listOfCompartments><compartment id="C" size="3" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="10" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="10" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts><speciesReference species="P" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci><ci>B</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
    m = bngsim.Model.from_sbml_string(src)
    rate_lines = [ln for ln in _rhs_src(m).splitlines() if "rate =" in ln]
    assert rate_lines, "no rate line emitted"
    grouped = [ln for ln in rate_lines if "(p[" in ln and "* p[" in ln]
    assert grouped, rate_lines
    # …and the model still runs, and still matches its own rebuild after a write.
    rebuilt = _traj(_sim(bngsim.Model.from_sbml_string(src.replace('size="3"', 'size="7"')), None))
    m.set_param("C", 7.0)
    assert np.array_equal(rebuilt, _traj(_sim(m, None)))


def test_a_model_with_no_writable_compartment_size_reads_no_parameter():
    """The byte-identity rule, as a rule rather than as a corpus number: only a
    species whose compartment size the loader bound to a parameter takes the live
    path. Everything else — ``.net``, V=1, hOSU=false — keeps an empty map, and an
    empty map is what makes the emitted text unchanged."""
    m = bngsim.Model.from_sbml_string(_one(1.0, hosu="false"))
    species = m._core.codegen_data()["species"]
    av_factor, av_param = _codegen._amount_volume_factors(species)
    assert av_factor == {} and av_param == {}
    obs = m._core.codegen_data()["observables"]
    if obs:
        assert not any("p[" in ln for ln in _codegen._emit_observable_lines(obs, {}, {}))


def test_the_analytical_jacobian_is_still_complete_with_a_symbolic_volume():
    """The per-species derivative now carries the compartment as a symbol; if that
    symbol did not resolve, the whole model would silently drop to the
    finite-difference Jacobian (the #170 stage-1 keyword-alias defect, one level
    down). Assert it did not."""
    for build, _comp in [(b, c) for _i, b, c in SHAPES]:
        m = bngsim.Model.from_sbml_string(build(V_NEW))
        bngsim.Simulator(m, method="ode")
        assert m._core.analytical_jacobian_complete, build


# ── The V=1 normalisations lifting the gate made reachable ──────────────────
#
# Stage 1 removed the V=1 normalisation for a *single-compartment Functional*
# storage divide, because that is the shape issue #170's rows 6 and 7 tabulated.
# Making 38 more models writable reached three more instances of the same thing:
# a load at V=1 emitted something a load at V≠1 did not, so `set_param` had
# nothing to move and did not reproduce a rebuild. Each is asserted the only way
# that distinguishes a fix from a coincidence — the write must equal a rebuild at
# the same size EXACTLY, and the shape must actually move with the volume.

_V1_MULTI_COMPARTMENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="mc">
    <listOfCompartments>
      <compartment id="C1" size="{v}" constant="true"/>
      <compartment id="C2" size="{v}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C2" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="tr" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/><apply><times/><ci>k</ci><ci>A</ci></apply>
            <apply><plus/><cn>1</cn><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

_V1_EVENT_DOSE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="dose">
    <listOfCompartments><compartment id="C" size="{v}" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="dose1" useValuesFromTriggerTime="true">
        <trigger><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><geq/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>2</cn></apply>
        </math></trigger>
        <listOfEventAssignments>
          <eventAssignment variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>500</cn></math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

_V1_AR_TARGET = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar">
    <listOfCompartments><compartment id="C" size="{v}" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="T" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="T"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>3</cn><ci>A</ci></apply>
      </math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

V1_SHAPES = [
    # (id, template, compartment written, why the V=1 load used to differ)
    ("multi_compartment_functional", _V1_MULTI_COMPARTMENT, "C1"),
    ("event_assigned_amount", _V1_EVENT_DOSE, "C"),
    ("assignment_rule_target", _V1_AR_TARGET, "C"),
]


@pytest.mark.parametrize(("tmpl", "comp"), [pytest.param(t, c, id=i) for i, t, c in V1_SHAPES])
def test_a_write_from_a_v1_load_reproduces_a_rebuild(tmpl, comp):
    """Loaded at V=1 — the value at which each of these three used to drop the
    volume out of the model entirely — then written. Exact, because a rebuild is
    what the write is defined to reproduce."""
    rebuilt = _traj(_sim(bngsim.Model.from_sbml_string(tmpl.format(v=repr(V_NEW))), None))
    m = bngsim.Model.from_sbml_string(tmpl.format(v=repr(V_LOAD)))
    assert m.unwritable_compartment_size_params == []
    m.set_param(comp, V_NEW)
    assert np.array_equal(rebuilt, _traj(_sim(m, None)))


@pytest.mark.parametrize(("tmpl", "comp"), [pytest.param(t, c, id=i) for i, t, c in V1_SHAPES])
def test_these_v1_shapes_really_do_move_with_the_volume(tmpl, comp):
    """Without this the test above would pass on a model the volume never reaches
    — which is exactly how the V=1 normalisation stayed invisible."""
    lo = _traj(_sim(bngsim.Model.from_sbml_string(tmpl.format(v=repr(V_LOAD))), None))
    hi = _traj(_sim(bngsim.Model.from_sbml_string(tmpl.format(v=repr(V_NEW))), None))
    assert not np.allclose(lo, hi)


def test_the_assignment_rule_report_divide_is_read_live():
    """The AR target's amount→concentration divide is the one conversion that lives
    entirely on the Python side — a report-time rescale in ``_ar_report_map``, not
    emitted math — so no engine refresh reaches it and it has to name its
    compartment. The 4th element is appended only when there IS a live size, so a
    model without one keeps the 3-tuple its other consumers pin."""
    live = bngsim.Model.from_sbml_string(_V1_AR_TARGET.format(v=repr(V_LOAD)))
    entry = live._ar_report_map["T"]
    assert len(entry) == 4 and entry[3] == "C", entry
    # …and a target whose compartment is not a writable size keeps the short form.
    plain = bngsim.Model.from_sbml_string(
        _V1_AR_TARGET.format(v=repr(V_LOAD)).replace(
            'hasOnlySubstanceUnits="true"', 'hasOnlySubstanceUnits="false"'
        )
    )
    assert len(plain._ar_report_map["T"]) == 3, plain._ar_report_map["T"]
