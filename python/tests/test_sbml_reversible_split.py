"""Phase 7: SBML reversible-reaction split for SSA.

A COPASI/Antimony-idiomatic reversible kineticLaw is written as a single
net-rate expression (``kf*A - kr*B`` or ``compartment * (kf*A - kr*B)``).
Under SSA a single net-rate channel locks at the deterministic
equilibrium; correct emission needs two independent channels. The
loader's ``_split_reversible_kinetic_law`` recognizes this shape and
emits two Elementary reactions; non-splittable reversible kineticLaws
(Hill, MM, function-defined, signed-difference of non-mass-action) keep
falling through to the Phase 6 ``reversible_non_mass_action`` gate.
"""

from pathlib import Path

import bngsim
import libsbml
import numpy as np
import pytest
from bngsim._sbml_loader import _factor_minus_subtree

# ── Unit tests for the AST walker ─────────────────────────────────────
#
# libsbml AST children are owned by the parent — if the parent is
# GC'd before the children are read, you segfault. Each test below holds
# the parsed AST in a local variable for the duration of the assertions.


def test_factor_minus_bare():
    """Bare ``kf*A - kr*B`` splits with no wrapper leaves."""
    ast = libsbml.parseL3Formula("kf*A - kr*B")
    r = _factor_minus_subtree(ast)
    assert r is not None
    wrapper, left, right = r
    assert wrapper == []
    assert left.getType() == libsbml.AST_TIMES
    assert right.getType() == libsbml.AST_TIMES


def test_factor_minus_compartment_wrapped():
    """COPASI shape ``compartment * (kf*A - kr*B)``: top-level TIMES."""
    ast = libsbml.parseL3Formula("compartment * (kf*A - kr*B)")
    r = _factor_minus_subtree(ast)
    assert r is not None
    wrapper, _, _ = r
    assert [w.getName() for w in wrapper] == ["compartment"]


def test_factor_minus_double_wrapped():
    """Multiple wrapper factors collected."""
    ast = libsbml.parseL3Formula("c1 * c2 * (kf*A - kr*B)")
    r = _factor_minus_subtree(ast)
    assert r is not None
    wrapper, _, _ = r
    assert sorted(w.getName() for w in wrapper) == ["c1", "c2"]


def test_factor_minus_power_wrapper():
    """Integer-power wrapper expands into N copies of the base."""
    ast = libsbml.parseL3Formula("c^2 * (kf*A - kr*B)")
    r = _factor_minus_subtree(ast)
    assert r is not None
    wrapper, _, _ = r
    assert [w.getName() for w in wrapper] == ["c", "c"]


def test_factor_minus_no_minus_returns_none():
    """Pure mass-action without a MINUS is not a reversible-split shape."""
    ast = libsbml.parseL3Formula("kf*A")
    assert _factor_minus_subtree(ast) is None


def test_factor_minus_two_minuses_at_siblings_rejected():
    """Two MINUS subtrees as siblings under TIMES is ambiguous."""
    ast = libsbml.parseL3Formula("(a-b)*(c-d)")
    assert _factor_minus_subtree(ast) is None


def test_factor_minus_unary_minus_rejected():
    """Unary minus has nc==1; the splitter requires binary MINUS."""
    ast = libsbml.parseL3Formula("-(kf*A)")
    assert _factor_minus_subtree(ast) is None


def test_factor_minus_division_rejected():
    """Hill-style divisions kill the wrapper-product walk before MINUS."""
    ast = libsbml.parseL3Formula("Vmax*S/(Km+S)")
    assert _factor_minus_subtree(ast) is None


def test_factor_minus_chained_returns_outer_only():
    """``a-b-c`` parses left-associatively; the walker records the OUTER
    binary MINUS and lets the per-side classifier reject the nested
    operand."""
    ast = libsbml.parseL3Formula("kf*A - kr*B - kx*C")
    r = _factor_minus_subtree(ast)
    assert r is not None
    _, left, _ = r
    assert left.getType() == libsbml.AST_MINUS  # nested MINUS rejected by classifier


# ── End-to-end loader tests ───────────────────────────────────────────


def _build_reversible_sbml(
    kinetic_math: str,
    reversible: bool = True,
    comp_size: float = 1.0,
    init_a: float = 20.0,
    init_b: float = 0.0,
    use_concentration: bool = False,
) -> str:
    """Two-species reversible reaction with parameterized kineticLaw math
    and compartment size. Used to exercise both bare and wrapped shapes."""
    init_attr = "initialConcentration" if use_concentration else "initialAmount"
    hosu = "false" if use_concentration else "true"
    rev_attr = "true" if reversible else "false"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="rev_split">
    <listOfCompartments>
      <compartment id="c" size="{comp_size}" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" {init_attr}="{init_a}"
               hasOnlySubstanceUnits="{hosu}"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" {init_attr}="{init_b}"
               hasOnlySubstanceUnits="{hosu}"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="0.1" constant="true"/>
      <parameter id="kr" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="rev" reversible="{rev_attr}">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            {kinetic_math}
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


_BARE_MINUS_MATH = """
<apply><minus/>
  <apply><times/><ci>kf</ci><ci>A</ci></apply>
  <apply><times/><ci>kr</ci><ci>B</ci></apply>
</apply>"""

_WRAPPED_MATH = """
<apply><times/>
  <ci>c</ci>
  <apply><minus/>
    <apply><times/><ci>kf</ci><ci>A</ci></apply>
    <apply><times/><ci>kr</ci><ci>B</ci></apply>
  </apply>
</apply>"""


def test_split_emits_two_reactions_bare_form():
    sbml = _build_reversible_sbml(_BARE_MINUS_MATH)
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.n_reactions == 2  # forward + reverse, not 1 net channel


def test_split_emits_two_reactions_copasi_wrapped_form():
    """``compartment * (kf*A - kr*B)`` splits even though top-level is TIMES."""
    sbml = _build_reversible_sbml(_WRAPPED_MATH)
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.n_reactions == 2


def test_split_passes_ssa_validation():
    """Splittable reversible kineticLaw must NOT raise at Simulator
    construction (the Phase 6 gate would have rejected this)."""
    sbml = _build_reversible_sbml(_BARE_MINUS_MATH)
    model = bngsim.Model.from_sbml_string(sbml)
    # Should not raise
    sim = bngsim.Simulator(model, method="ssa")
    assert sim is not None
    assert bngsim.validate_for_ssa(model) == []


def test_irreversible_minus_form_does_not_split():
    """Phase 7 only attempts splitting when reversible='true'. A
    non-reversible reaction with a difference-shape kineticLaw stays as
    a single Functional channel (and gets the non-mass-action fall-
    through; current behavior is unchanged from pre-Phase-7)."""
    sbml = _build_reversible_sbml(_BARE_MINUS_MATH, reversible=False)
    model = bngsim.Model.from_sbml_string(sbml)
    # Single Functional channel (1 emitted reaction, not 2).
    assert model.n_reactions == 1


def test_split_with_v_neq_1_compartment():
    """``c * (kf*A - kr*B)`` with V_c != 1 — the wrapper distributes onto
    each side; the per-side classifier folds c into stat_factor / volume
    product correctly."""
    sbml = _build_reversible_sbml(
        _WRAPPED_MATH,
        comp_size=2.5,
        init_a=50.0,
        init_b=0.0,
        use_concentration=True,
    )
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.n_reactions == 2
    sim = bngsim.Simulator(model, method="ssa")
    # Smoke: trajectory advances and conserves total amount
    res = sim.run(t_span=(0.0, 10.0), n_points=11, seed=42)
    # Concentration storage * V_c = amount; total amount conserved at 125
    # within ±1 due to integer-amount SSA discretization.
    total_amount = (res.species[-1, 0] + res.species[-1, 1]) * 2.5
    assert abs(total_amount - 125.0) < 1.0


def test_split_trajectory_fluctuates_not_locked():
    """Pre-Phase-7 the single net-rate channel locked at the deterministic
    equilibrium (propensity = 0 there). Post-split, A↔B with kf=kr should
    fluctuate with binomial spread N·p·q at equilibrium."""
    sbml = _build_reversible_sbml(_BARE_MINUS_MATH, init_a=200.0, init_b=0.0)
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ssa")
    n_reps = 60
    end_a = []
    for s in range(n_reps):
        res = sim.run(t_span=(0.0, 200.0), n_points=11, seed=s)
        end_a.append(res.species[-1, 0])
    end_a = np.asarray(end_a)
    # Equilibrium mean: A=100, B=100. SD ≈ √(N·p·q) = √(200·0.5·0.5) = √50 ≈ 7.07.
    # Pre-fix (locked): all values exactly equal → std == 0.
    assert end_a.std(ddof=1) > 3.0, (
        f"std(A_end) = {end_a.std(ddof=1):.3f}; expected ~7 for fluctuating "
        "trajectory. std≈0 indicates trajectory is locked at deterministic "
        "equilibrium → splitter not active or per-channel rate wrong."
    )
    assert abs(end_a.mean() - 100.0) < 5.0


def test_copasi_abc_xml_loads():
    """The original PyBNF abc.xml — COPASI-style 3-species reversible
    chain with `compartment * (kf*A - kr*B)` shape — must load cleanly
    under SSA after Phase 7. Fixture lives in the PyBNF tree; skip if
    not present."""
    abc_path = Path(__file__).resolve().parents[4] / "PyBNF" / "tests" / "bngl_files" / "abc.xml"
    if not abc_path.exists():
        pytest.skip(f"abc.xml not at {abc_path}")
    model = bngsim.Model.from_sbml(str(abc_path))
    # 2 SBML reactions × 2 channels each = 4 emitted reactions
    assert model.n_reactions == 4
    sim = bngsim.Simulator(model, method="ssa")
    res = sim.run(t_span=(0.0, 100.0), n_points=11, seed=1)
    # Conservation: A+B+C == initial total (20).
    total = res.species[-1].sum()
    assert abs(total - 20.0) < 0.5


# ── Phase 6 gate still fires on un-splittable reversible kineticLaws ──


_HILL_REV_MATH = """
<apply><minus/>
  <apply><divide/>
    <apply><times/><ci>kf</ci><ci>A</ci></apply>
    <apply><plus/><ci>Km</ci><ci>A</ci></apply>
  </apply>
  <apply><times/><ci>kr</ci><ci>B</ci></apply>
</apply>"""


def test_hill_reversible_rejected_by_phase6_gate():
    """A reversible kineticLaw whose forward operand is Hill-shaped does
    not classify as mass-action — the Phase 6 gate must still raise."""
    sbml = _build_reversible_sbml(_HILL_REV_MATH).replace(
        '<parameter id="kr" value="0.1" constant="true"/>',
        '<parameter id="kr" value="0.1" constant="true"/>'
        '<parameter id="Km" value="5.0" constant="true"/>',
    )
    model = bngsim.Model.from_sbml_string(sbml)
    issues = bngsim.validate_for_ssa(model)
    assert any(i.code == "reversible_non_mass_action" for i in issues), issues


# ── Matched-seed parity: reversible-SBML (Phase 7 split) vs hand-split SBML ──


_HAND_SPLIT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="rev_handsplit">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="200"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="0.1" constant="true"/>
      <parameter id="kr" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="fwd" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>kf</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="rev" reversible="false">
        <listOfReactants>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>kr</ci><ci>B</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_split_distribution_matches_hand_split():
    """Per-seed trajectories from a Phase-7-split reversible SBML should
    track the equivalent hand-split SBML in distribution. Both models
    have the same emitted reactions (forward A→B and reverse B→A) — we
    assert distribution-level agreement on end-of-run A amounts.
    """
    auto_sbml = _build_reversible_sbml(_BARE_MINUS_MATH, init_a=200.0, init_b=0.0)
    auto_model = bngsim.Model.from_sbml_string(auto_sbml)
    hand_model = bngsim.Model.from_sbml_string(_HAND_SPLIT_SBML)

    auto_sim = bngsim.Simulator(auto_model, method="ssa")
    hand_sim = bngsim.Simulator(hand_model, method="ssa")

    n_reps = 100
    auto_end = []
    hand_end = []
    for s in range(n_reps):
        r1 = auto_sim.run(t_span=(0.0, 100.0), n_points=11, seed=s)
        auto_end.append(r1.species[-1, 0])
        r2 = hand_sim.run(t_span=(0.0, 100.0), n_points=11, seed=s)
        hand_end.append(r2.species[-1, 0])
    auto_end = np.asarray(auto_end)
    hand_end = np.asarray(hand_end)

    # Equilibrium mean ≈ 100 for both (kf=kr); diff-of-means SE ≈ √(2·50/100) = 1.
    delta_mean = abs(auto_end.mean() - hand_end.mean())
    assert delta_mean < 4.0, (
        f"|mean_auto - mean_hand| = {delta_mean:.2f} "
        f"(auto={auto_end.mean():.2f}, hand={hand_end.mean():.2f})"
    )
    delta_var = abs(auto_end.var(ddof=1) - hand_end.var(ddof=1))
    assert delta_var < 30.0, f"|var_auto - var_hand| = {delta_var:.2f}"


# ── Cache shape: mixed reaction kinds in one model ────────────────────


_MIXED_KINDS_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="mixed_kinds">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="D" compartment="c" initialAmount="50"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="E" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_irrev" value="0.05" constant="true"/>
      <parameter id="kf" value="0.1" constant="true"/>
      <parameter id="kr" value="0.1" constant="true"/>
      <parameter id="Vmax" value="1.0" constant="true"/>
      <parameter id="Km" value="5.0" constant="true"/>
      <parameter id="kCB" value="0.05" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="irrev_ma" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k_irrev</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="rev_ma" reversible="true">
        <listOfReactants>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/>
            <apply><times/><ci>kf</ci><ci>B</ci></apply>
            <apply><times/><ci>kr</ci><ci>C</ci></apply>
          </apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="rev_hill" reversible="true">
        <listOfReactants>
          <speciesReference species="D" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="E" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/>
            <apply><divide/>
              <apply><times/><ci>Vmax</ci><ci>D</ci></apply>
              <apply><plus/><ci>Km</ci><ci>D</ci></apply>
            </apply>
            <apply><times/><ci>kCB</ci><ci>E</ci></apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_cache_handles_mixed_reaction_kinds():
    """Aggressive cache-shape test: a single SBML with one of each
    reaction kind exercises both 1-element and 2-element entries in the
    mass_action_rxns cache simultaneously, plus a cache-MISS reaction
    that triggers the Phase 6 gate.

    Layout:
      - irrev_ma   (irreversible mass-action) → cache[0] = [(rate, ...)]
      - rev_ma     (reversible mass-action)   → cache[1] = [fwd, rev]
      - rev_hill   (reversible Hill)          → not cached; gate fires

    Expected: 1 + 2 + 1 = 4 emitted reactions, 1 SsaIssue
    (reversible_non_mass_action on rev_hill), Simulator(method='ssa')
    raises.
    """
    model = bngsim.Model.from_sbml_string(_MIXED_KINDS_SBML)
    assert model.n_reactions == 4, f"expected 4 emitted reactions, got {model.n_reactions}"
    issues = bngsim.validate_for_ssa(model)
    rev_issues = [i for i in issues if i.code == "reversible_non_mass_action"]
    assert len(rev_issues) == 1, f"expected 1 gate fire on rev_hill; got {issues}"
    assert rev_issues[0].location == "reaction:rev_hill"
    with pytest.raises(bngsim.SsaValidationError):
        bngsim.Simulator(model, method="ssa")


_MULTI_COMPONENT_RATE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="multi_component">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c"
               initialAmount="100" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c"
               initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.05" constant="true"/>
      <parameter id="k2" value="2.0" constant="true"/>
      <parameter id="k3" value="0.05" constant="true"/>
      <parameter id="k4" value="2.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="rev" reversible="true">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/>
            <apply><times/><ci>k1</ci><ci>k2</ci><ci>A</ci></apply>
            <apply><times/><ci>k3</ci><ci>k4</ci><ci>B</ci></apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_split_synthesizes_distinct_derived_param_names():
    """When each split side has multiple rate-parameter components,
    `_classification_to_cache_tuple` synthesizes derived params for both.
    The fwd/rev suffix must keep them distinct so `add_parameter` doesn't
    raise on duplicate-name registration."""
    model = bngsim.Model.from_sbml_string(_MULTI_COMPONENT_RATE_SBML)
    assert model.n_reactions == 2  # split succeeded
    names = set(model.param_names)
    # The synthesized derived params should appear with _fwd / _rev suffixes
    assert "_rateLaw_rev_fwd" in names, f"missing fwd derived param; have {sorted(names)}"
    assert "_rateLaw_rev_rev" in names, f"missing rev derived param; have {sorted(names)}"
    # Initial values should equal the products of constituent params
    # fwd: k1*k2 = 0.05 * 2.0 = 0.1
    # rev: k3*k4 = 0.05 * 2.0 = 0.1
    assert abs(model.get_param("_rateLaw_rev_fwd") - 0.1) < 1e-12
    assert abs(model.get_param("_rateLaw_rev_rev") - 0.1) < 1e-12

    # Smoke: simulator constructs and runs
    sim = bngsim.Simulator(model, method="ssa")
    res = sim.run(t_span=(0.0, 100.0), n_points=11, seed=7)
    assert abs(res.species[-1].sum() - 100.0) < 1e-9


def test_many_splits_in_sequence():
    """A 4-reaction reversible chain A↔B↔C↔D↔E (kf=kr=0.1) — every
    reaction splits, exercising 4 consecutive 2-element cache entries.
    Final emitted-reaction count must be 8. Trajectory must conserve
    total mass A+B+C+D+E and equilibrate roughly uniformly across the
    five species."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="chain5">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c"
               initialAmount="500" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c"
               initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c"
               initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="D" compartment="c"
               initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="E" compartment="c"
               initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>"""
    for src, dst in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")):
        sbml += f"""
      <reaction id="r_{src}_{dst}" reversible="true">
        <listOfReactants>
          <speciesReference species="{src}" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="{dst}" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/>
            <apply><times/><ci>k</ci><ci>{src}</ci></apply>
            <apply><times/><ci>k</ci><ci>{dst}</ci></apply>
          </apply>
        </math></kineticLaw>
      </reaction>"""
    sbml += "\n    </listOfReactions>\n  </model>\n</sbml>"

    model = bngsim.Model.from_sbml_string(sbml)
    assert model.n_reactions == 8, f"expected 8 (4 reversibles × 2), got {model.n_reactions}"
    assert bngsim.validate_for_ssa(model) == []

    sim = bngsim.Simulator(model, method="ssa")
    res = sim.run(t_span=(0.0, 500.0), n_points=11, seed=2026)
    # Conservation: total mass = 500
    total = res.species[-1].sum()
    assert abs(total - 500.0) < 1e-9, f"mass not conserved: total={total}"
    # Rough equilibration check: at long t with uniform k_f=k_r, expect
    # ~100 each species (5-way uniform). Allow generous range.
    final = res.species[-1]
    assert all(40.0 < x < 200.0 for x in final), f"non-equilibrium final state {final}"


def test_cache_handles_mixed_reaction_kinds_under_ode():
    """Same mixed model under ODE — Phase 6 gate doesn't fire (it's
    SSA-only); all 4 emitted reactions are integrated. Smoke check:
    conservation A+B+C and D+E hold."""
    model = bngsim.Model.from_sbml_string(_MIXED_KINDS_SBML)
    sim = bngsim.Simulator(model, method="ode")
    res = sim.run(t_span=(0.0, 10.0), n_points=11)
    # A+B+C conservation (initial: A=100, B=0, C=0 → total 100)
    abc_total = res.species[-1, 0] + res.species[-1, 1] + res.species[-1, 2]
    assert abs(abc_total - 100.0) < 1e-6
    # D+E conservation (initial: D=50, E=0 → total 50)
    de_total = res.species[-1, 3] + res.species[-1, 4]
    assert abs(de_total - 50.0) < 1e-6
