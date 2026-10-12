"""Regression tests for SBML loader fixes (PUNCHLIST §A1, §A2, §A3).

Each test reproduces a class of SBML load failure exposed by surveying
the 963-BioModels Antimony corpus on PR #17 (see
``dev/SBML_LOADER_PUNCHLIST.md``). The corpus itself does not live on
``feature/bngsim``, so these synthetic fixtures serve as the tracked
regression cases.
"""

from __future__ import annotations

import math

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


def test_c2_unset_param_refusal_is_typed(monkeypatch):
    """It is `UnderSpecifiedModelError`, not a bare `ModelError` (issue #323).

    `ModelError` also covers .net parse failures and invalid model state, so a
    caller could not tell "this model is under-specified" — a documented refusal
    with an escape hatch — from an actionable bug without matching message text.
    The amici_parity suites bucket this one `UNSUPPORTED` rather than
    `EXCEPTION`, by type; 12 corpus models were mislabelled before it existed.

    The test above deliberately still catches `bngsim.ModelError`: that is the
    back-compat contract, and it fails if the subclassing is ever dropped.
    """
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    with pytest.raises(bngsim.UnderSpecifiedModelError) as exc:
        bngsim.Model.from_sbml_string(_C2_UNSET_CONSUMED_SBML)
    assert issubclass(bngsim.UnderSpecifiedModelError, bngsim.ModelError)
    assert "kdeg" in str(exc.value)
    # The hatch must stay advertised — it is the only way past this refusal.
    assert "BNGSIM_ALLOW_UNSET_PARAMS" in str(exc.value)


def test_c2_other_model_errors_are_not_typed_as_under_specified(monkeypatch):
    """Guard against over-typing: an unrelated load failure must NOT become
    `UnderSpecifiedModelError`, or the parity suites would route real bugs into
    a non-scoring bucket."""
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    with pytest.raises(Exception) as exc:
        bngsim.Model.from_sbml_string("<not-sbml/>")
    assert not isinstance(exc.value, bngsim.UnderSpecifiedModelError)


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


# ─── C2b (#580): a kineticLaw's valueless LOCAL parameter ───────────────
#
# C2 above covers only the model's global <parameter>s. A local one with no
# `value=` loaded as 0.0 on the L2 path (the reaction silently switched off) and
# as libsbml's NaN on the L3 path (a CVODE failure blaming negative states).


def _local_param_sbml(
    level: int, local_decl: str, law_reads: str = "k", model_k: bool = False
) -> str:
    """`A -> 0` at rate `<law_reads> * A`, with `local_decl` as the law's local list.

    `model_k` adds a global `k = 5` for the local one to shadow, and `p`, whose
    initialAssignment reads the reaction id `R1` (its rate, `k * A` at t=0).
    """
    if level == 2:
        head = '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">'
        comp = '<compartment id="c" size="1"/>'
        spec = '<species id="A" compartment="c" initialConcentration="10"/>'
        rxn = '<reaction id="R1" reversible="false">'
        ref = '<speciesReference species="A"/>'
        lps = f"<listOfParameters>{local_decl}</listOfParameters>"
        params = '<parameter id="k" value="5"/><parameter id="p" value="7"/>'
    else:
        head = '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">'
        comp = '<compartment id="c" size="1" constant="true"/>'
        spec = (
            '<species id="A" compartment="c" initialConcentration="10" '
            'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        )
        rxn = '<reaction id="R1" reversible="false" fast="false">'
        ref = '<speciesReference species="A" stoichiometry="1" constant="true"/>'
        lps = f"<listOfLocalParameters>{local_decl}</listOfLocalParameters>"
        params = (
            '<parameter id="k" value="5" constant="true"/>'
            '<parameter id="p" value="7" constant="true"/>'
        )
    globals_ = (
        f"""<listOfParameters>{params}</listOfParameters>
        <listOfInitialAssignments>
          <initialAssignment symbol="p">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>R1</ci></math>
          </initialAssignment>
        </listOfInitialAssignments>"""
        if model_k
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    {head}
      <model id="m">
        <listOfCompartments>{comp}</listOfCompartments>
        <listOfSpecies>{spec}</listOfSpecies>
        {globals_}
        <listOfReactions>
          {rxn}
            <listOfReactants>{ref}</listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>{law_reads}</ci><ci>A</ci></apply>
              </math>
              {lps}
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""


def _unset_local(level: int) -> str:
    return '<parameter id="k"/>' if level == 2 else '<localParameter id="k"/>'


@pytest.mark.parametrize("level", [2, 3])
def test_c2b_unset_local_param_read_by_its_law_refuses(monkeypatch, level):
    """Refused like a global one, naming the parameter, its reaction and the hatch."""
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    with pytest.raises(bngsim.UnderSpecifiedModelError) as exc:
        bngsim.Model.from_sbml_string(_local_param_sbml(level, _unset_local(level)))
    msg = str(exc.value)
    assert "'k'" in msg and "'R1'" in msg, msg
    assert "BNGSIM_ALLOW_UNSET_PARAMS" in msg


@pytest.mark.parametrize("level", [2, 3])
def test_c2b_unset_local_param_hatch_loads_at_zero(monkeypatch, caplog, level):
    """Under the hatch both levels load at 0.0 and say so. L3 used to read
    libsbml's NaN here and fail in CVODE instead, so the two paths now agree."""
    import logging

    import numpy as np

    monkeypatch.setenv("BNGSIM_ALLOW_UNSET_PARAMS", "1")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        model = bngsim.Model.from_sbml_string(_local_param_sbml(level, _unset_local(level)))
    msgs = [r.getMessage() for r in caplog.records]
    assert [m for m in msgs if "'k'" in m and "'R1'" in m and "no value" in m], msgs
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)
    np.testing.assert_array_equal(result.species[:, 0], [10.0, 10.0, 10.0])


@pytest.mark.parametrize("level", [2, 3])
def test_c2b_unset_local_param_the_law_does_not_read_loads(monkeypatch, caplog, level):
    """A valueless local parameter its law never reads changes nothing: no
    refusal and no warning, as for an unread global one."""
    import logging

    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    local = (
        '<parameter id="k" value="0.5"/>' if level == 2 else '<localParameter id="k" value="0.5"/>'
    )
    spare = local.replace('id="k" value="0.5"', 'id="spare"')
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        model = bngsim.Model.from_sbml_string(_local_param_sbml(level, local + spare))
    assert not [r for r in caplog.records if "spare" in r.getMessage()]
    # Nothing can read it, so it is not registered: not as a plausible 0.0, and
    # not as a NaN that a writer would then refuse to serialise.
    assert not [n for n in model.param_names if "spare" in n]
    assert all(math.isfinite(model.get_param(n)) for n in model.param_names)


@pytest.mark.parametrize("level", [2, 3])
def test_c2b_unset_local_param_shadowing_a_global_refuses(monkeypatch, level):
    """SBML scoping: the law's `k` is the local one, so a global `k` does not
    give it a value."""
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    with pytest.raises(bngsim.UnderSpecifiedModelError, match="'k' \\(reaction 'R1'\\)"):
        bngsim.Model.from_sbml_string(_local_param_sbml(level, _unset_local(level), model_k=True))


@pytest.mark.parametrize("level", [2, 3])
def test_local_param_shadows_global_where_initial_values_fold(level):
    """`p = R1` folds R1's law at load. On L2 that fold read only
    `getLocalParameter`, which libsbml leaves empty there, so the global `k = 5`
    stood in for the local `k = 0.5`: p came out 50 where RoadRunner gives 5."""
    local = (
        '<parameter id="k" value="0.5"/>'
        if level == 2
        else ('<localParameter id="k" value="0.5"/>')
    )
    model = bngsim.Model.from_sbml_string(_local_param_sbml(level, local, model_k=True))
    assert model.get_param("p") == pytest.approx(5.0)


@pytest.mark.parametrize("level", [2, 3])
def test_c2b_hatch_zero_reaches_initial_value_folds(monkeypatch, level):
    """Under the hatch the unset local `k` is 0.0 in the fold too. The fold used
    to skip it and read the global `k = 5`, giving p = 50 beside a warning that
    the rate law was zeroed."""
    monkeypatch.setenv("BNGSIM_ALLOW_UNSET_PARAMS", "1")
    model = bngsim.Model.from_sbml_string(
        _local_param_sbml(level, _unset_local(level), model_k=True)
    )
    assert model.get_param("p") == 0.0


@pytest.mark.parametrize("level", [2, 3])
def test_duplicate_local_param_id_refuses(level):
    """Two locals with one id is invalid SBML with no right value to pick. The
    builder's duplicate-name check used to refuse it; an id -> value map must not
    quietly keep one of the two."""
    tag = "parameter" if level == 2 else "localParameter"
    dup = f'<{tag} id="k" value="0.5"/><{tag} id="k" value="2"/>'
    with pytest.raises(
        bngsim.ModelError, match="'R1' declares local parameter 'k' more than once"
    ):
        bngsim.Model.from_sbml_string(_local_param_sbml(level, dup))


def test_unfoldable_global_does_not_refuse_a_fold_its_local_shadows():
    """The failure trace behind "initialAssignment ... cannot be evaluated" refolds
    each reaction's law. On L2 it saw no locals, so `p = R1` was blamed on the
    global `k`, which an assignment rule over a valueless species leaves unfoldable,
    though the law reads the local `k = 0.5`. It loads, at RoadRunner's p = 5."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
      <model id="m">
        <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="c" initialConcentration="10"/>
          <species id="B" compartment="c"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="5" constant="false"/>
          <parameter id="p" value="7"/>
        </listOfParameters>
        <listOfInitialAssignments>
          <initialAssignment symbol="p">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>R1</ci></math>
          </initialAssignment>
        </listOfInitialAssignments>
        <listOfRules>
          <assignmentRule variable="k">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>B</ci></math>
          </assignmentRule>
        </listOfRules>
        <listOfReactions>
          <reaction id="R1" reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
              <listOfParameters><parameter id="k" value="0.5"/></listOfParameters>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.get_param("p") == pytest.approx(5.0)


@pytest.mark.parametrize("level", [2, 3])
def test_floor_ceiling_roots_read_the_laws_local_parameter(level):
    """The state-threshold scan (GH #244) evaluated a law with no local scope: a
    local `k` read the global `k` (wrong thresholds) or nothing (no roots). The
    roots for `ceiling(k*A)` with a local `k = 0.5` match a global `k = 0.5`."""

    def triggers(local: str, global_k: str, k_moves: bool = False, via_g: bool = False) -> int:
        # via_g: the same law as `g(k, A)` with `g(a, x) = ceiling(a*x)`, so the
        # scan meets the local as a call argument rather than in the law itself.
        law = (
            "<apply><ci>g</ci><ci>k</ci><ci>A</ci></apply>"
            if via_g
            else "<apply><ceiling/><apply><times/><ci>k</ci><ci>A</ci></apply></apply>"
        )
        sbml = _local_param_sbml(level, local).replace(
            "<apply><times/><ci>k</ci><ci>A</ci></apply>", law
        )
        if via_g:
            sbml = sbml.replace(
                "<listOfCompartments>",
                '<listOfFunctionDefinitions><functionDefinition id="g">'
                '<math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>'
                "<bvar><ci>a</ci></bvar><bvar><ci>x</ci></bvar><apply><ceiling/>"
                "<apply><times/><ci>a</ci><ci>x</ci></apply></apply></lambda></math>"
                "</functionDefinition></listOfFunctionDefinitions><listOfCompartments>",
                1,
            )
        if global_k:
            const = ' constant="false"' if k_moves else ("" if level == 2 else ' constant="true"')
            rule = (
                '<listOfRules><assignmentRule variable="k">'
                '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><cn>5</cn>'
                '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">'
                "t</csymbol></apply></math></assignmentRule></listOfRules>"
                if k_moves
                else ""
            )
            sbml = sbml.replace(
                "<listOfReactions>",
                f'<listOfParameters><parameter id="k" value="{global_k}"{const}/>'
                f"</listOfParameters>{rule}<listOfReactions>",
            )
        return bngsim.Model.from_sbml_string(sbml)._core.n_discontinuity_triggers

    tag = "parameter" if level == 2 else "localParameter"
    reference = triggers("", "0.5")
    assert reference > 0
    assert triggers(f'<{tag} id="k" value="0.5"/>', "") == reference
    assert triggers(f'<{tag} id="k" value="0.5"/>', "5") == reference
    # The global `k` moving with time says nothing about the local one, in the
    # law or passed into a function.
    assert triggers(f'<{tag} id="k" value="0.5"/>', "5", k_moves=True) == reference
    assert triggers(f'<{tag} id="k" value="0.5"/>', "5", k_moves=True, via_g=True) == reference


@pytest.mark.parametrize("level", [2, 3])
def test_local_param_with_an_explicit_nan_value_is_kept(level):
    """`value="NaN"` is a value the model wrote: the local is registered and the
    law reads it, not the global `k = 5` it shadows."""
    tag = "parameter" if level == 2 else "localParameter"
    const = "" if level == 2 else ' constant="true"'
    sbml = _local_param_sbml(level, f'<{tag} id="k" value="NaN"/>').replace(
        "<listOfReactions>",
        f'<listOfParameters><parameter id="k" value="5"{const}/></listOfParameters>'
        "<listOfReactions>",
    )
    model = bngsim.Model.from_sbml_string(sbml)
    assert math.isnan(model.get_param("_lp_R1_k"))


@pytest.mark.parametrize("level", [2, 3])
def test_c2b_unset_local_param_read_through_a_function_body_refuses(monkeypatch, level):
    """`f(x) = k*x` called as `f(A)`: the body's free `k` resolves to the law's
    local (invalid SBML, but the translator allows it), so it counts as read."""
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    sbml = _local_param_sbml(level, _unset_local(level), model_k=True).replace(
        "<apply><times/><ci>k</ci><ci>A</ci></apply>", "<apply><ci>f</ci><ci>A</ci></apply>"
    )
    fdef = """<listOfFunctionDefinitions><functionDefinition id="f">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><lambda><bvar><ci>x</ci></bvar>
          <apply><times/><ci>k</ci><ci>x</ci></apply></lambda></math>
        </functionDefinition></listOfFunctionDefinitions>
        <listOfCompartments>"""
    sbml = sbml.replace("<listOfCompartments>", fdef, 1)
    with pytest.raises(bngsim.UnderSpecifiedModelError, match="'k' \\(reaction 'R1'\\)"):
        bngsim.Model.from_sbml_string(sbml)


def _with_functions(sbml: str, defs: dict[str, tuple[list[str], str]]) -> str:
    """Insert <functionDefinition>s: name -> (formals, body MathML)."""
    fds = "".join(
        f'<functionDefinition id="{name}"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"<lambda>{''.join(f'<bvar><ci>{a}</ci></bvar>' for a in formals)}{body}</lambda>"
        "</math></functionDefinition>"
        for name, (formals, body) in defs.items()
    )
    return sbml.replace(
        "<listOfCompartments>",
        f"<listOfFunctionDefinitions>{fds}</listOfFunctionDefinitions><listOfCompartments>",
        1,
    )


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize(
    "defs",
    [
        {"f": (["k"], "<apply><times/><cn>2</cn><ci>k</ci></apply>")},
        {
            "f": (["k"], "<apply><ci>g</ci><ci>A</ci></apply>"),
            "g": (["y"], "<apply><times/><ci>k</ci><ci>y</ci></apply>"),
        },
    ],
    ids=["formal-k", "outer-formal-k"],
)
def test_c2b_function_formal_named_like_an_unset_local_is_not_a_read(monkeypatch, level, defs):
    """A formal `k`, or an outer function's formal `k` that a nested body reads,
    binds `k` on that path; the unset local `k` is not read and must not refuse."""
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    sbml = _with_functions(
        _local_param_sbml(level, _unset_local(level)).replace(
            "<apply><times/><ci>k</ci><ci>A</ci></apply>", "<apply><ci>f</ci><ci>A</ci></apply>"
        ),
        defs,
    )
    model = bngsim.Model.from_sbml_string(sbml)
    assert not [n for n in model.param_names if n == "_lp_R1_k"]


def _root_count(law: str, defs: dict, extra_params: str = "") -> int:
    sbml = _with_functions(
        _local_param_sbml(3, "").replace("<apply><times/><ci>k</ci><ci>A</ci></apply>", law), defs
    )
    sbml = sbml.replace(
        "<listOfReactions>",
        f"<listOfParameters>{extra_params}</listOfParameters><listOfReactions>"
        if extra_params
        else "<listOfReactions>",
    )
    return bngsim.Model.from_sbml_string(sbml)._core.n_discontinuity_triggers


_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)


def test_floor_ceiling_formal_rebound_to_a_constant_is_not_time_dependent():
    """`f(a, x) = g(0.5, x)` with `g(a, x) = ceiling(a*x)`: g's `a` is 0.5 whatever
    f's `a` was, so `f(time, A)` gets the roots `f(0.5, A)` does. The scan kept
    f's time dependence on the rebound name and registered none."""
    defs = {
        "f": (["a", "x"], "<apply><ci>g</ci><cn>0.5</cn><ci>x</ci></apply>"),
        "g": (["a", "x"], "<apply><ceiling/><apply><times/><ci>a</ci><ci>x</ci></apply></apply>"),
    }
    reference = _root_count("<apply><ci>f</ci><cn>0.5</cn><ci>A</ci></apply>", defs)
    assert reference > 0
    assert _root_count(f"<apply><ci>f</ci>{_TIME}<ci>A</ci></apply>", defs) == reference


def test_floor_ceiling_formal_with_no_load_time_value_does_not_read_a_global():
    """`g(q, x) = ceiling(q*x)` called as `g(B, A)`, B a species with no value at
    load: the scan dropped `q` and started from the global `q = 2` instead,
    registering 20 roots on a value the model never has."""
    ceil_qx = "<apply><ceiling/><apply><times/><ci>q</ci><ci>x</ci></apply></apply>"
    defs = {"g": (["q", "x"], ceil_qx)}
    law = "<apply><ci>g</ci><ci>B</ci><ci>A</ci></apply>"
    sbml_b = (
        '<species id="B" compartment="c" hasOnlySubstanceUnits="false" '
        'boundaryCondition="false" constant="false"/>'
    )
    sbml = _with_functions(
        _local_param_sbml(3, "")
        .replace("<apply><times/><ci>k</ci><ci>A</ci></apply>", law)
        .replace("</listOfSpecies>", sbml_b + "</listOfSpecies>")
        .replace(
            "<listOfReactions>",
            '<listOfParameters><parameter id="q" value="2" constant="true"/>'
            "</listOfParameters><listOfReactions>",
        ),
        defs,
    )
    assert bngsim.Model.from_sbml_string(sbml)._core.n_discontinuity_triggers == 0


@pytest.mark.parametrize("level", [2, 3])
def test_c2_global_shadowed_by_the_laws_local_is_not_read(monkeypatch, level):
    """A valueless global `k` that the law's local `k = 0.5` shadows is not read
    by the law, so C2 must not refuse it (it named the global as read)."""
    monkeypatch.delenv("BNGSIM_ALLOW_UNSET_PARAMS", raising=False)
    tag = "parameter" if level == 2 else "localParameter"
    const = "" if level == 2 else ' constant="true"'
    sbml = _local_param_sbml(level, f'<{tag} id="k" value="0.5"/>').replace(
        "<listOfReactions>",
        f'<listOfParameters><parameter id="k"{const}/></listOfParameters><listOfReactions>',
    )
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.get_param("_lp_R1_k") == 0.5


def test_floor_ceiling_formal_with_no_load_time_value_reads_as_unknown():
    """Bound to an unknown argument, a formal must stay unknown through a
    comparison: `ceiling(piecewise(1, q >= 0, 2) * x)` gets no thresholds, so the
    roots are the piecewise edges alone. A NaN binding chose the `otherwise`
    branch (NaN >= 0 is false) and registered 20; the global `q` gave 10."""
    sp_b = (
        '<species id="B" compartment="c" hasOnlySubstanceUnits="false" '
        'boundaryCondition="false" constant="false"/>'
    )
    pw = (
        "<apply><times/><piecewise><piece><cn>1</cn><apply><geq/><ci>q</ci><cn>0</cn>"
        "</apply></piece><otherwise><cn>2</cn></otherwise></piecewise><ci>x</ci></apply>"
    )

    def triggers(body: str) -> int:
        sbml = _with_functions(
            _local_param_sbml(3, "")
            .replace(
                "<apply><times/><ci>k</ci><ci>A</ci></apply>",
                "<apply><ci>g</ci><ci>B</ci><ci>A</ci></apply>",
            )
            .replace("</listOfSpecies>", sp_b + "</listOfSpecies>")
            .replace(
                "<listOfReactions>",
                '<listOfParameters><parameter id="q" value="2" constant="true"/>'
                "</listOfParameters><listOfReactions>",
            ),
            {"g": (["q", "x"], body)},
        )
        return bngsim.Model.from_sbml_string(sbml)._core.n_discontinuity_triggers

    assert triggers(f"<apply><ceiling/>{pw}</apply>") == triggers(pw)


@pytest.mark.parametrize(
    ("period", "global_period"),
    [("0.5", ""), ("0.5", "5"), ("100", ""), ("100", "0.5")],
    ids=["local", "local-over-global", "long-local", "long-local-over-short-global"],
)
def test_periodic_step_bound_reads_the_laws_local_period(period, global_period):
    """GH #88's step bound for `A * rem(floor(time/period), 2)` with a local
    `period` matches the bound for a global one. Read in global scope it
    vanished (no `period`) or came out 10x too wide (a global `period = 5`);
    and a long local period read through a short global one sized the scan
    window too short to see a cycle, losing the bound."""

    def bound(local: str, global_period: str) -> float | None:
        law = (
            "<apply><times/><ci>A</ci><apply><rem/><apply><floor/><apply><divide/>"
            f"{_TIME}<ci>period</ci></apply></apply><cn>2</cn></apply></apply>"
        )
        params = '<parameter id="z" constant="false"/>' + (
            f'<parameter id="period" value="{global_period}" constant="true"/>'
            if global_period
            else ""
        )
        sbml = (
            _local_param_sbml(3, local)
            .replace("<apply><times/><ci>k</ci><ci>A</ci></apply>", law)
            .replace(
                "<listOfReactions>",
                f"<listOfParameters>{params}</listOfParameters>"
                '<listOfRules><assignmentRule variable="z">'
                '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>A</ci></math>'
                "</assignmentRule></listOfRules><listOfReactions>",
            )
        )
        return bngsim.Model.from_sbml_string(sbml)._periodic_disc_max_step

    reference = bound("", period)
    assert reference is not None
    got = bound(f'<localParameter id="period" value="{period}"/>', global_period)
    assert got == pytest.approx(reference)


def test_c2b_hatch_zero_reaches_the_rateof_fold(monkeypatch):
    """Same for `rateOf(A)` in an initialAssignment, which folds R1's law too:
    it read the global `k = 5` and gave p = -50."""
    monkeypatch.setenv("BNGSIM_ALLOW_UNSET_PARAMS", "1")
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="m">
        <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="c" initialConcentration="10"
                   hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="5" constant="true"/>
          <parameter id="p" value="7" constant="true"/>
        </listOfParameters>
        <listOfInitialAssignments>
          <initialAssignment symbol="p">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <csymbol encoding="text"
                         definitionURL="http://www.sbml.org/sbml/symbols/rateOf">rateOf</csymbol>
                <ci>A</ci>
              </apply>
            </math>
          </initialAssignment>
        </listOfInitialAssignments>
        <listOfReactions>
          <reaction id="R1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
              <listOfLocalParameters><localParameter id="k"/></listOfLocalParameters>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.get_param("p") == 0.0


# ─── #580: a species the model never gives an initial value ─────────────
#
# SBML leaves it undefined. bngsim starts it at 0.0, as RoadRunner does (7 of the
# 8 BioModels with one pass rr_parity that way), but says so instead of passing
# the zero off as a value the model wrote.


def _species_sbml(species: str, extra: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
      <model id="m">
        <listOfCompartments>
          <compartment id="c" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>{species}</listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="2" constant="true"/>
        </listOfParameters>
        {extra}
      </model>
    </sbml>"""


def _sp(sid: str, ic: str = "") -> str:
    return (
        f'<species id="{sid}" compartment="c" {ic} hasOnlySubstanceUnits="false" '
        'boundaryCondition="false" constant="false"/>'
    )


def test_unset_species_initial_value_warns_and_names_it(caplog):
    import logging

    sbml = _species_sbml(_sp("A") + _sp("B", 'initialConcentration="1"'))
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        model = bngsim.Model.from_sbml_string(sbml)
    hits = [r.getMessage() for r in caplog.records if "initial value undefined" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    assert "'A'" in hits[0] and "'B'" not in hits[0], hits[0]
    # The value itself is unchanged: still RoadRunner's 0.0.
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    assert result.species[0, result.species_names.index("A")] == 0.0


def test_unset_species_initial_value_warning_is_one_capped_message(caplog):
    """A genome-scale model has thousands of these: one message, not one each."""
    import logging

    sbml = _species_sbml("".join(_sp(f"S{i}") for i in range(15)))
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(sbml)
    hits = [r.getMessage() for r in caplog.records if "initial value undefined" in r.getMessage()]
    assert len(hits) == 1, hits
    assert hits[0].startswith("15 species have") and "(5 more)" in hits[0], hits[0]
    assert "'S9'" in hits[0] and "'S10'" not in hits[0], hits[0]


# `k * C` reads a species with no value, so the rule cannot fold to a number at
# load and `A` reaches the no-initial-value branch; a foldable `A := k` would be
# resolved before it and test nothing. The model still defines `A` (through `C`),
# so only `C` is named.
@pytest.mark.parametrize(
    "extra",
    [
        """<listOfInitialAssignments>
             <initialAssignment symbol="A">
               <math xmlns="http://www.w3.org/1998/Math/MathML">
                 <apply><times/><ci>k</ci><ci>C</ci></apply>
               </math>
             </initialAssignment>
           </listOfInitialAssignments>""",
        """<listOfRules>
             <assignmentRule variable="A">
               <math xmlns="http://www.w3.org/1998/Math/MathML">
                 <apply><times/><ci>k</ci><ci>C</ci></apply>
               </math>
             </assignmentRule>
           </listOfRules>""",
    ],
    ids=["initialAssignment", "assignmentRule"],
)
def test_species_valued_by_a_rule_or_assignment_is_not_named(caplog, extra):
    import logging

    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(_species_sbml(_sp("A") + _sp("C"), extra))
    hits = [r.getMessage() for r in caplog.records if "initial value undefined" in r.getMessage()]
    assert len(hits) == 1, hits
    assert "'C'" in hits[0] and "'A'" not in hits[0], hits[0]


# ─── #580: a compartment the model never gives a size ───────────────────
#
# SBML Level 2 and 3 leave it undefined (Level 1 defaults it to 1.0). bngsim
# uses 1.0, as RoadRunner does, and says so where the 1.0 can reach a number.


def _compartment_sbml(level: int, comp: str, species: str, extra: str = "") -> str:
    head = (
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">'
        if level == 2
        else '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">'
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    {head}
      <model id="m">
        <listOfCompartments>{comp}</listOfCompartments>
        <listOfSpecies>{species}</listOfSpecies>
        {extra}
      </model>
    </sbml>"""


def _comp(level: int, attrs: str = "") -> str:
    const = "" if level == 2 else ' constant="true"'
    return f'<compartment id="c"{const} {attrs}/>'


def _species_in_c(level: int, ic: str, hosu: bool) -> str:
    extra = "" if level == 2 else ' boundaryCondition="false" constant="false"'
    return (
        f'<species id="A" compartment="c" {ic} '
        f'hasOnlySubstanceUnits="{str(hosu).lower()}"{extra}/>'
    )


_RULE_READS_C = """<listOfParameters><parameter id="p" constant="false"/></listOfParameters>
        <listOfRules><assignmentRule variable="p">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>c</ci></math>
        </assignmentRule></listOfRules>"""


def _size_warnings(caplog, sbml: str):
    import logging

    with caplog.at_level(logging.WARNING, logger="bngsim"):
        model = bngsim.Model.from_sbml_string(sbml)
    hits = [r.getMessage() for r in caplog.records if "size undefined" in r.getMessage()]
    return model, hits


@pytest.mark.parametrize("level", [2, 3])
def test_unsized_compartment_warns_and_names_it(caplog, level):
    """A concentration-valued species makes the size reach a number. The value
    is unchanged, RoadRunner's 1.0, but the load says it was chosen."""
    sbml = _compartment_sbml(
        level, _comp(level), _species_in_c(level, 'initialConcentration="2"', False)
    )
    model, hits = _size_warnings(caplog, sbml)
    assert len(hits) == 1 and "'c'" in hits[0], hits
    assert model.get_param("c") == 1.0


_MATH = '<math xmlns="http://www.w3.org/1998/Math/MathML">'
_SP_D = (
    '<species id="D" compartment="d" initialAmount="1" hasOnlySubstanceUnits="true" '
    'boundaryCondition="false" constant="false"/>'
)
_TIME_GT_1 = (
    f'<trigger initialValue="false" persistent="true">{_MATH}<apply><gt/>'
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t'
    "</csymbol><cn>1</cn></apply></math></trigger>"
)
_READERS = {
    # case -> (species in c as "<ic attr> hosu=<bool>", extra model elements, warns?)
    "empty": ("", "", False),
    "conc-species": ('initialConcentration="2" hosu=false', "", True),
    "amount-species": ('initialAmount="2" hosu=false', "", True),
    "amount-only-species": ('initialAmount="2" hosu=true', "", True),
    "amount-only-species-given-conc": ('initialConcentration="2" hosu=true', "", True),
    "rule-names-c": ("", _RULE_READS_C, True),
    "initialAssignment-names-c": (
        "",
        '<listOfParameters><parameter id="p" constant="true"/></listOfParameters>'
        f'<listOfInitialAssignments><initialAssignment symbol="p">{_MATH}<ci>c</ci></math>'
        "</initialAssignment></listOfInitialAssignments>",
        True,
    ),
    "law-names-c": (
        "",
        '<listOfReactions><reaction id="R1" reversible="false" fast="false">'
        '<listOfReactants><speciesReference species="D" stoichiometry="1" constant="true"/>'
        f"</listOfReactants><kineticLaw>{_MATH}<apply><times/><ci>c</ci><ci>D</ci></apply>"
        "</math></kineticLaw></reaction></listOfReactions>",
        True,
    ),
    "event-trigger-names-c": (
        "",
        '<listOfParameters><parameter id="p" value="0" constant="false"/></listOfParameters>'
        '<listOfEvents><event id="E" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true">{_MATH}<apply><gt/><ci>c</ci>'
        "<cn>5</cn></apply></math></trigger><listOfEventAssignments>"
        f'<eventAssignment variable="p">{_MATH}<cn>1</cn></math></eventAssignment>'
        "</listOfEventAssignments></event></listOfEvents>",
        True,
    ),
    "rate-rule-sets-c": (
        "",
        f'<listOfRules><rateRule variable="c">{_MATH}<cn>0.5</cn></math></rateRule></listOfRules>',
        True,
    ),
    "event-sets-c": (
        "",
        f'<listOfEvents><event id="E" useValuesFromTriggerTime="true">{_TIME_GT_1}'
        f'<listOfEventAssignments><eventAssignment variable="c">{_MATH}<cn>5</cn></math>'
        "</eventAssignment></listOfEventAssignments></event></listOfEvents>",
        True,
    ),
}


@pytest.mark.parametrize("case", list(_READERS))
def test_unsized_compartment_warns_only_where_the_size_reaches_a_number(caplog, case):
    """Each way the 1.0 reaches a result warns on its own: a species in `c`
    (bngsim reports every species as amount over its size), math naming `c`, or a
    rate rule or event setting `c` (a state that starts at the size). An empty
    compartment nothing names stays silent."""
    sp, extra, warns = _READERS[case]
    species = _SP_D
    if sp:
        ic, hosu = sp.rsplit(" hosu=", 1)
        species += _species_in_c(3, ic, hosu == "true")
    const = ' constant="false"' if "sets-c" in case else ' constant="true"'
    comps = f'<compartment id="c"{const}/><compartment id="d" size="1" constant="true"/>'
    _, hits = _size_warnings(caplog, _compartment_sbml(3, comps, species, extra))
    assert bool(hits) is warns, hits
    if warns:
        assert "1 compartment has" in hits[0] and "'c'" in hits[0] and "'d'" not in hits[0]


def test_unsized_compartment_warning_names_each_in_one_message(caplog):
    comps = '<compartment id="c1" constant="true"/><compartment id="c2" constant="true"/>'
    species = "".join(
        f'<species id="{s}" compartment="{c}" initialConcentration="1" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        for s, c in (("A", "c1"), ("B", "c2"))
    )
    _, hits = _size_warnings(caplog, _compartment_sbml(3, comps, species))
    assert len(hits) == 1, hits
    assert hits[0].startswith("2 compartments have") and "'c1', 'c2'" in hits[0], hits[0]


@pytest.mark.parametrize(
    "extra",
    [
        """<listOfInitialAssignments><initialAssignment symbol="c">
             <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math>
           </initialAssignment></listOfInitialAssignments>""",
        """<listOfRules><assignmentRule variable="c">
             <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math>
           </assignmentRule></listOfRules>""",
    ],
    ids=["initialAssignment", "assignmentRule"],
)
def test_compartment_sized_by_an_assignment_or_rule_is_silent(caplog, extra):
    comp = '<compartment id="c" constant="false"/>' if "Rule" in extra else _comp(3)
    sbml = _compartment_sbml(3, comp, _species_in_c(3, 'initialConcentration="2"', False), extra)
    model, hits = _size_warnings(caplog, sbml)
    assert not hits, hits
    assert model.get_param("c") == 2.0


def test_zero_dimensional_compartment_is_silent(caplog):
    """A 0-D compartment has no size to leave undefined."""
    sbml = _compartment_sbml(
        3,
        _comp(3, 'spatialDimensions="0"'),
        _species_in_c(3, 'initialAmount="2"', True),
        _RULE_READS_C,
    )
    _, hits = _size_warnings(caplog, sbml)
    assert not hits, hits


def test_kinetic_law_local_named_like_the_compartment_does_not_read_it(caplog):
    """A law's local `c` shadows the compartment `c`, so the law reads its local;
    the size still reaches no number."""
    rxn = """<listOfReactions><reaction id="R1" reversible="false" fast="false">
          <listOfReactants>
            <speciesReference species="A" stoichiometry="1" constant="true"/>
          </listOfReactants>
          <kineticLaw>
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><ci>c</ci><ci>A</ci></apply>
            </math>
            <listOfLocalParameters><localParameter id="c" value="0.5"/></listOfLocalParameters>
          </kineticLaw>
        </reaction></listOfReactions>"""
    comps = '<compartment id="c" constant="true"/><compartment id="d" size="1" constant="true"/>'
    rxn = rxn.replace('species="A"', 'species="D"').replace("<ci>A</ci>", "<ci>D</ci>")
    sbml = _compartment_sbml(3, comps, _SP_D, rxn)
    _, hits = _size_warnings(caplog, sbml)
    assert not hits, hits


def _comp_doc(main_comp: str, sub_size: str = "") -> str:
    """A comp document: `main_comp` in the main model, which holds species A, and
    a submodel `S` whose definition declares compartment `C`."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"
          xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1"
          comp:required="true">
      <model id="m">
        <listOfCompartments>{main_comp}</listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="c" initialConcentration="2"
                   hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <comp:listOfSubmodels>
          <comp:submodel comp:id="S" comp:modelRef="sub"/>
        </comp:listOfSubmodels>
      </model>
      <comp:listOfModelDefinitions>
        <comp:modelDefinition id="sub">
          <listOfCompartments>
            <compartment id="C" spatialDimensions="3" {sub_size} constant="true"/>
          </listOfCompartments>
        </comp:modelDefinition>
      </comp:listOfModelDefinitions>
    </sbml>"""


@pytest.mark.parametrize("source", ["string", "file"])
@pytest.mark.parametrize(
    ("main_comp", "sub_size", "sizes", "want"),
    [
        ('<compartment id="c" constant="true"/>', "", None, None),
        ('<compartment id="c" constant="true"/>', "", {"c": 3.0}, 3.0),
        (
            '<compartment id="c" constant="true">'
            '<comp:replacedBy comp:idRef="C" comp:submodelRef="S"/></compartment>',
            'size="4"',
            None,
            4.0,
        ),
    ],
    ids=["unsized", "compartment_sizes", "replacedBy-sized"],
)
def test_comp_model_compartment_size_is_checked_after_flattening(
    caplog, tmp_path, source, main_comp, sub_size, sizes, want
):
    """A comp model is checked once flattened. An unsized main-model compartment
    still warns. A size the caller passes is a chosen one, and a placeholder
    `replacedBy` a sized submodel compartment takes that size (SBML test suite
    01374-01379): both stay silent."""
    import logging

    sbml = _comp_doc(main_comp, sub_size)
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        if source == "string":
            model = bngsim.Model.from_sbml_string(sbml, compartment_sizes=sizes)
        else:
            path = tmp_path / "comp.xml"
            path.write_text(sbml)
            model = bngsim.Model.from_sbml(str(path), compartment_sizes=sizes)
    hits = [r.getMessage() for r in caplog.records if "size undefined" in r.getMessage()]
    if want is None:
        assert len(hits) == 1 and "'c'" in hits[0], hits
        assert model.get_param("c") == 1.0
    else:
        assert not hits, hits
        assert model.get_param("c") == want


def test_level1_compartment_takes_the_spec_default_silently(caplog):
    """Level 1 defines a missing volume as 1.0, so nothing was invented."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level1" level="1" version="2">
      <model name="m">
        <listOfCompartments><compartment name="c"/></listOfCompartments>
        <listOfSpecies>
          <species name="A" compartment="c" initialAmount="2"/>
          <species name="B" compartment="c" initialAmount="0"/>
        </listOfSpecies>
        <listOfReactions>
          <reaction name="R1" reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <listOfProducts><speciesReference species="B"/></listOfProducts>
            <kineticLaw formula="0.1 * A"/>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    model, hits = _size_warnings(caplog, sbml)
    assert not hits, hits
    assert model.get_param("c") == 1.0


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
