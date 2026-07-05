"""Regression tests for SBML loader fixes (PUNCHLIST §A1, §A2, §A3).

Each test reproduces a class of SBML load failure exposed by surveying
the 963-BioModels Antimony corpus on PR #17 (see
``dev/SBML_LOADER_PUNCHLIST.md``). The corpus itself does not live on
``feature/bngsim``, so these synthetic fixtures serve as the tracked
regression cases.
"""

from __future__ import annotations

import bngsim
import pytest

pytest.importorskip("antimony")


# ─── A1: event-assigned compartment must not double-register ───────────


def test_a1_event_assigned_variable_compartment_loads():
    """Reproduce BIOMD0000000338/339 pattern: a non-constant compartment
    is also the target of an event assignment. Pre-fix, the loader added
    the compartment as a parameter in step 1 AND as a species during
    event-promotion in step 10, colliding on the same ExprTk symbol.
    """
    text = """
    model var_compartment_event()
        compartment compartment_1 = 1.0;
        compartment_1 has volume;
        var compartment_1;
        S in compartment_1;
        S = 5.0;
        k = 0.1;
        J0: S -> ; k*S;
        E1: at time > 0.5: compartment_1 = 2.0;
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    assert model.n_species >= 1
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 1.0), n_points=11)
    assert result.species.shape[0] == 11


# ─── A2: rate rule referencing a reaction id must resolve ──────────────


def test_a2_rate_rule_references_reaction_id():
    """Reproduce BIOMD0000000542 pattern: rate rule whose RHS is a bare
    reaction id (SBML L3 <ci>r77</ci> evaluates to that reaction's
    kineticLaw value). Pre-fix, the rate-rule expression for the promoted
    species was a bare reaction-id symbol that no ExprTk variable matched.
    """
    sbml = """<?xml version="1.0"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="ratrul_refs_rxn">
        <listOfCompartments>
          <compartment id="cell" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="cell" initialAmount="10"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
          <species id="B" compartment="cell" initialAmount="0"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
          <species id="Ap" compartment="cell" initialAmount="0"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k1" value="0.5" constant="true"/>
        </listOfParameters>
        <listOfRules>
          <rateRule variable="Ap">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <ci>r1</ci>
            </math>
          </rateRule>
        </listOfRules>
        <listOfReactions>
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B" stoichiometry="1" constant="true"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k1</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 4.0), n_points=41)
    sp_names = list(result.species_names)
    a_idx = sp_names.index("A")
    ap_idx = sp_names.index("Ap")
    # A decays exponentially; ∫r1 dt drives Ap upward in lockstep.
    # Ap(∞) ≈ A(0)·k1/k1 == A(0) == 10. Pre-fix the model would not load.
    assert result.species[-1, a_idx] < result.species[0, a_idx]
    assert result.species[-1, ap_idx] > 0


# ─── A3: reaction without kineticLaw → clear error, not silent rate=0 ─


def test_a3_reaction_without_kinetic_law_raises():
    """Reproduce MODEL0568648427 / MODEL1204270001 pattern: a reaction
    declared without a <kineticLaw> child. Pre-fix the loader fell
    through with a confusing
    ``"reaction 0 (Functional) references unknown function 'X'"`` from
    C++ validate(). Post-fix the loader rejects with an actionable
    message so the user does not silently get rate=0 dynamics.
    """
    sbml = """<?xml version="1.0"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="no_kinetic_law">
        <listOfCompartments>
          <compartment id="cell" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="cell" initialAmount="1"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
          <species id="B" compartment="cell" initialAmount="0"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
        </listOfSpecies>
        <listOfReactions>
          <reaction id="r_no_law" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B" stoichiometry="1" constant="true"/>
            </listOfProducts>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    with pytest.raises(Exception) as excinfo:
        bngsim.Model.from_sbml_string(sbml)
    msg = str(excinfo.value)
    assert "no kineticLaw" in msg
    assert "r_no_law" in msg


# ─── C2 (#94): referenced missing-value parameter — reject by default ──

# Parameter declared without `value=`, no IA, but consumed by a kineticLaw —
# the model is under-specified. Defaulting it to 0.0 silently zeros out the
# reaction rate and returns a wrong-but-plausible trajectory; RoadRunner refuses
# the same model. Shared fixture for the reject/escape-hatch pair below.
_C2_UNSET_CONSUMED_SBML = """<?xml version="1.0"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="unset_consumed">
        <listOfCompartments>
          <compartment id="cell" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="cell" initialAmount="10"
                   hasOnlySubstanceUnits="true"
                   boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="kdeg" constant="true"/>
        </listOfParameters>
        <listOfReactions>
          <reaction id="death" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>kdeg</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""


def test_c2_unset_param_referenced_in_kineticlaw_rejects(monkeypatch):
    """Default: a referenced valueless parameter is a hard ModelError,
    matching RoadRunner — bngsim refuses rather than silently defaulting it
    to 0.0. The error must name the offending parameter."""
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(_C2_UNSET_CONSUMED_SBML)
    assert "kdeg" in str(exc.value)


def test_c2_unset_param_referenced_escape_hatch_loads(monkeypatch, caplog):
    """BNGSIM_ALLOW_UNSET_PARAMS=1 restores the legacy lenient behavior:
    load with the parameter defaulted to 0.0 and a load-time warning that
    names it (the bngsim↔rr triage escape hatch)."""
    import logging

    monkeypatch.setenv("BNGSIM_ALLOW_UNSET_PARAMS", "1")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(_C2_UNSET_CONSUMED_SBML)
    msgs = [r.getMessage() for r in caplog.records]
    matched = [m for m in msgs if "kdeg" in m and "no value attribute" in m]
    assert matched, f"expected load-time warning naming 'kdeg'; got: {msgs}"


def test_c2_unset_param_unused_does_not_warn(caplog):
    """Parameter declared without `value=` but never referenced in any
    expression — bngsim defaults to 0.0, but nobody reads it. Must NOT
    warn (otherwise the corpus's 80+ documentation/extension-package
    placeholders would all become noise)."""
    import logging

    sbml = """<?xml version="1.0"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="unset_unused">
        <listOfCompartments>
          <compartment id="cell" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="cell" initialAmount="10"
                   hasOnlySubstanceUnits="true"
                   boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="kdeg" value="0.5" constant="true"/>
          <parameter id="placeholder_for_extension" constant="true"/>
        </listOfParameters>
        <listOfReactions>
          <reaction id="death" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>kdeg</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(sbml)
    msgs = [r.getMessage() for r in caplog.records]
    placeholder_msgs = [m for m in msgs if "placeholder_for_extension" in m]
    assert not placeholder_msgs, (
        f"unset-but-unreferenced param must not warn; got: {placeholder_msgs}"
    )


def test_c2_unset_param_with_initial_assignment_does_not_warn(caplog):
    """Parameter declared without `value=` but covered by an
    initialAssignment — the IA fills in the value at t=0, so the silent
    default-to-0 never matters. Must NOT warn."""
    import logging

    sbml = """<?xml version="1.0"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="unset_ia">
        <listOfCompartments>
          <compartment id="cell" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="cell" initialAmount="10"
                   hasOnlySubstanceUnits="true"
                   boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="base" value="0.5" constant="true"/>
          <parameter id="kdeg" constant="true"/>
        </listOfParameters>
        <listOfInitialAssignments>
          <initialAssignment symbol="kdeg">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><cn>2.0</cn><ci>base</ci></apply>
            </math>
          </initialAssignment>
        </listOfInitialAssignments>
        <listOfReactions>
          <reaction id="death" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>kdeg</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(sbml)
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "kdeg" in m and "no value attribute" in m], msgs


# ─── GH #91: one reaction's kineticLaw references another reaction's id ──


def test_gh91_kinetic_law_references_reaction_id():
    """Reproduce MODEL2306170002 r0 pattern: a reaction's kineticLaw names
    OTHER reaction ids (SBML L3 <ci>r1</ci> evaluates to that reaction's
    rate of progress). Pre-fix only *rules* referencing a reaction id were
    pre-registered as functions, so a kineticLaw referencing one failed to
    compile with ``ERR239 - Undefined symbol``.

    Fixture: r1 (A -> B, rate k1*A) is mass-action; r2 ( -> C, rate 2*r1)
    references r1's rate. Then d[C]/dt = 2*d[B]/dt with C(0)=B(0)=0, so the
    exact invariant C(t) == 2*B(t) holds at every sample — a direct check
    that the reaction-id symbol resolved to r1's actual rate.
    """
    sbml = """<?xml version="1.0"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="kl_refs_rxn">
        <listOfCompartments>
          <compartment id="cell" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="cell" initialAmount="10"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
          <species id="B" compartment="cell" initialAmount="0"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
          <species id="C" compartment="cell" initialAmount="0"
                   hasOnlySubstanceUnits="true" boundaryCondition="false"
                   constant="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k1" value="0.5" constant="true"/>
        </listOfParameters>
        <listOfReactions>
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B" stoichiometry="1" constant="true"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k1</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
          <reaction id="r2" reversible="false">
            <listOfProducts>
              <speciesReference species="C" stoichiometry="1" constant="true"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><cn>2</cn><ci>r1</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 6.0), n_points=31, atol=1e-10, rtol=1e-10)
    names = list(result.species_names)
    b = result.species[:, names.index("B")]
    c = result.species[:, names.index("C")]
    a = result.species[:, names.index("A")]
    # r2 fires at twice r1's rate into a fresh species ⇒ C == 2·B exactly.
    assert c == pytest.approx(2.0 * b, abs=1e-6)
    # And B tracks the consumed A (mass balance on the r1 channel).
    assert b == pytest.approx(10.0 - a, abs=1e-6)
    assert c[-1] > 0  # dynamics actually advanced


# ─── GH #92: NaN literal in an event trigger (COPASI rateOf stub) ────────


def test_gh92_real_literal_renders_nonfinite_as_arithmetic():
    """``_real_literal`` maps a non-finite IEEE double to ExprTk-parseable pure
    arithmetic and round-trips a finite value via ``repr`` (GH #92).

    ExprTk's lexer has no bare ``nan``/``inf`` token (``init_builtins`` registers
    neither), and its ``<n>#nan`` form does not survive embedding inside a
    parenthesised subexpression, so a NaN/Inf ``AST_REAL`` must render as
    arithmetic that constant-folds to the same IEEE value.
    """
    from bngsim._sbml_loader import _real_literal

    assert _real_literal(float("nan")) == "(0.0/0.0)"
    assert _real_literal(float("inf")) == "(1.0/0.0)"
    assert _real_literal(float("-inf")) == "(-1.0/0.0)"
    assert _real_literal(2.5) == repr(2.5)


def test_gh92_nan_literal_in_event_trigger():
    """Reproduce MODEL1910030001: COPASI exports the SBML ``rateOf`` csymbol as
    a ``functionDefinition`` whose body is ``<notanumber/>`` (a NaN literal).
    Inlining that function into an event trigger produced ``nan*c > 0``. Pre-#92
    the loader rendered the NaN ``AST_REAL`` as the bare token ``nan``, which
    ExprTk cannot compile, so the whole model failed to load (#92 fixed the load
    crash; the bare-token rendering is still guarded by the ``_real_literal``
    unit test above).

    Post-#106 the ``rateOf`` idiom is intercepted at load and the funcDef's NaN
    body is never inlined: ``rateOf(X)`` becomes the live derivative dX/dt. Here
    ``X`` has no rate rule and no reaction (its only writer is this very event),
    so dX/dt = 0; the trigger ``rateOf(X)*c > 0`` is ``0 > 0`` = false and the
    event never fires — matching libRoadRunner. The observable outcome is
    identical to the NaN-propagation era (``X`` holds 1.0) but now for the
    semantically-correct reason.

    Fixture: ``rateOf(X)`` gates an event that would set ``X := 999``. The event
    must never fire, so ``X`` holds its initial value 1.0.
    """
    sbml = """<?xml version="1.0"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="nan_trigger">
        <listOfFunctionDefinitions>
          <functionDefinition id="rateOf">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <lambda><bvar><ci>a</ci></bvar><notanumber/></lambda>
            </math>
          </functionDefinition>
        </listOfFunctionDefinitions>
        <listOfParameters>
          <parameter id="X" value="1" constant="false"/>
          <parameter id="c" value="2" constant="true"/>
        </listOfParameters>
        <listOfEvents>
          <event id="control_of_X">
            <trigger initialValue="false" persistent="true">
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><gt/>
                  <apply><times/>
                    <apply><ci>rateOf</ci><ci>X</ci></apply>
                    <ci>c</ci>
                  </apply>
                  <cn>0</cn>
                </apply>
              </math>
            </trigger>
            <listOfEventAssignments>
              <eventAssignment variable="X">
                <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>999</cn></math>
              </eventAssignment>
            </listOfEventAssignments>
          </event>
        </listOfEvents>
      </model>
    </sbml>"""
    # Pre-fix this raised ModelError("... Undefined symbol: 'nan'").
    model = bngsim.Model.from_sbml_string(sbml)
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 5.0), n_points=6)
    names = list(result.species_names)
    # X is event-mutated, so it is promoted to a reported species (GH #71).
    x = result.species[:, names.index("X")]
    assert all(v == pytest.approx(1.0) for v in x)  # NaN-gated event never fired
