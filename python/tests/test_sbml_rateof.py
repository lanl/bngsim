"""GH #106 — SBML ``rateOf`` csymbol (live species derivative dx/dt).

``rateOf(x)`` is the instantaneous time-derivative of ``x`` — a quantity only
the integrator knows. bngsim normalizes BOTH SBML encodings to a single
per-species accessor (``rate_of__<species>``) and publishes the live dx/dt into
a buffer via a derivative *probe* before every RHS / trigger evaluation:

  * the official csymbol — libsbml ``AST_FUNCTION_RATE_OF`` (type 323);
  * the COPASI ``functionDefinition`` idiom — a funcDef whose body is
    ``<notanumber/>`` carrying a ``…/Derivative`` annotation.

One probe is exact because every rateOf argument is a species whose derivative
is independent of the rateOf consumers (no algebraic loop). See
``dev/notes/gh106_rateof_kickoff.md`` for the full design.

The primary regression cases below use **closed-form oracles** (dependency-free)
because the rr_parity BioModels corpus does not live on this branch. An optional
roadrunner-gated section cross-checks the four real corpus models when both
roadrunner and the corpus are present locally.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

# rateOf csymbol MathML (official encoding). libsbml parses this to type 323.
_RATEOF_CSYMBOL = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/rateOf"> rateOf </csymbol>'
)


def _sbml(model_body: str) -> str:
    """Wrap a ``<model>`` body in a minimal SBML L3V2 document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
        'level="3" version="2">\n'
        f"{model_body}\n"
        "</sbml>\n"
    )


def _decay_reaction(species: str = "A", rate_param: str = "k") -> str:
    """A first-order decay reaction ``species -> ∅`` with rate ``k*species*c``
    (compartment ``c`` size 1, so the kinetic law is k·[species])."""
    return f"""
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="{species}" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>{rate_param}</ci><ci>{species}</ci><ci>c</ci></apply>
        </math></kineticLaw>
      </reaction>"""


def _series(result, name: str) -> np.ndarray | None:
    """Pull a named series from a result as species or function, else None."""
    sp = list(result.species_names)
    if name in sp:
        return np.asarray(result.species[:, sp.index(name)])
    fn = list(getattr(result, "function_names", []) or [])
    if name in fn:
        return np.asarray(result.functions[:, fn.index(name)])
    return None


def _run(model, *, codegen: bool = False, t_end: float = 5.0, n_points: int = 11):
    return bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0.0, t_end), n_points=n_points
    )


# ─── Closed-form oracles (always run) ────────────────────────────────────────


@pytest.mark.parametrize("codegen", [False, True])
def test_rate_rule_rateof_tracks_its_argument(codegen):
    """A rate rule ``B' = rateOf(A)`` makes B integrate dA/dt, so d(B-A)/dt = 0
    and ``B - A`` is conserved exactly — a derivative-free oracle for
    ``rateOf(A) == dA/dt``. The decay also lets us check the closed form
    ``A(t) = A0·exp(-k·t)``. Exercises the official csymbol on the rate-rule /
    functional-reaction RHS path (BIOMD0000000696 pattern), interpreted and
    codegen.
    """
    A0, k = 10.0, 0.7
    body = f"""
  <model id="rate_rule_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True

    res = _run(model, codegen=codegen)
    t = np.asarray(res.time)
    A, B = _series(res, "A"), _series(res, "B")

    # rateOf(A) == dA/dt: B integrates it, so B - A holds its initial value.
    assert np.allclose(B - A, -A0, atol=1e-9, rtol=0)
    # And the species itself follows the analytic decay.
    assert np.allclose(A, A0 * np.exp(-k * t), atol=0, rtol=1e-5)


@pytest.mark.parametrize("codegen", [False, True])
def test_assignment_rule_rateof_feeds_dynamics(codegen):
    """An assignment rule ``r := rateOf(A)`` whose value drives a second state
    (``C' = -r = k·A``) — the BIOMD0000000775 pattern where rateOf is consumed
    *inside* the RHS. Since ``C' = -A' ``, ``C + A`` is conserved at ``A0``. This
    is the case that needs rateOf live during ``compute_derivs`` (not just in a
    trigger).
    """
    A0, k = 10.0, 0.7
    body = f"""
  <model id="assign_rule_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="{k}" constant="true"/>
      <parameter id="r" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="r"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></assignmentRule>
      <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><ci>r</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True

    res = _run(model, codegen=codegen)
    A, C = _series(res, "A"), _series(res, "C")
    assert np.allclose(C + A, A0, atol=1e-9, rtol=0)


def test_event_trigger_on_rateof_fires_at_predicted_time():
    """A trigger gated on ``rateOf(A) > -1`` with ``A' = -A`` (k=1). rateOf(A)
    = -A starts at -10 (< -1, false) and rises toward 0, crossing -1 when
    ``A = 1`` ⇒ ``t = ln(10)``. The event sets a reported species ``marker := 1``.

    Also pins the t=0 initialization fix: with a zero-initialized derivative
    buffer the trigger would read ``0 > -1`` = true and fire spuriously at t=0.
    """
    body = f"""
  <model id="event_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="marker" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="1" constant="true"/></listOfParameters>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
    <listOfEvents>
      <event id="ev1" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply><cn>-1</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="marker">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True

    res = _run(model, t_end=5.0, n_points=501)
    t = np.asarray(res.time)
    marker = _series(res, "marker")

    assert marker[0] == pytest.approx(0.0)  # no spurious t=0 fire
    fire_t = t[int(np.argmax(marker > 0.5))]
    assert fire_t == pytest.approx(math.log(10.0), abs=2 * (t[1] - t[0]))


def test_copasi_funcdef_idiom_normalized_to_derivative():
    """The COPASI rateOf idiom — a ``functionDefinition`` whose body is
    ``<notanumber/>`` with a ``…/Derivative`` annotation — must be intercepted
    and emit the live derivative rather than inlining NaN. Same B-A invariant as
    the csymbol case, driven through the funcDef call ``rateOf(A)``.
    """
    A0, k = 8.0, 0.5
    body = f"""
  <model id="copasi_idiom">
    <listOfFunctionDefinitions>
      <functionDefinition id="rateOf">
        <annotation>
          <symbols xmlns="http://sbml.org/annotations/symbols"
                   definition="http://en.wikipedia.org/wiki/Derivative"/>
        </annotation>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda><bvar><ci>a</ci></bvar><notanumber/></lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><ci>rateOf</ci><ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True
    res = _run(model)
    A, B = _series(res, "A"), _series(res, "B")
    assert np.all(np.isfinite(B))  # NaN body must NOT have been inlined
    assert np.allclose(B - A, -A0, atol=1e-9, rtol=0)


def test_unused_rateof_funcdef_does_not_enable_feature():
    """A model that *defines* a rateOf idiom funcDef but never *calls* it
    (MODEL2403070001 pattern) must stay on the byte-identical non-rateOf path —
    ``uses_rateof`` is False and the dead funcDef is simply never consumed.
    """
    body = (
        """
  <model id="unused_rateof">
    <listOfFunctionDefinitions>
      <functionDefinition id="rateOf">
        <annotation>
          <symbols xmlns="http://sbml.org/annotations/symbols"
                   definition="http://en.wikipedia.org/wiki/Derivative"/>
        </annotation>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda><bvar><ci>a</ci></bvar><notanumber/></lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>"""
        + _decay_reaction()
        + """</listOfReactions>
  </model>"""
    )
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is False
    res = _run(model)
    A = _series(res, "A")
    assert np.allclose(A, 10.0 * np.exp(-0.3 * np.asarray(res.time)), atol=0, rtol=1e-5)


def test_ssa_rejects_rateof_model():
    """rateOf has no defined value in a stochastic trajectory; SSA must reject
    a rateOf model loudly rather than feed a stale/zero derivative."""
    A0, k = 10.0, 0.7
    body = f"""
  <model id="ssa_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    with pytest.raises(Exception, match="rateOf"):
        bngsim.Simulator(model, method="ssa").run(t_span=(0.0, 1.0), n_points=2)


def test_clone_preserves_rateof():
    """A clone (used by every parallel worker) must rebind the rateOf accessors
    to its own derivative buffer and reproduce the original trajectory."""
    from bngsim._model import Model

    A0, k = 10.0, 0.7
    body = f"""
  <model id="clone_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    clone = Model(_core=model._core.clone())
    assert clone._core.uses_rateof is True

    r0 = _run(model)
    r1 = _run(clone)
    assert np.allclose(_series(r0, "B"), _series(r1, "B"), atol=0, rtol=0)


# ─── Optional corpus cross-check vs libRoadRunner ────────────────────────────
#
# The four rr_parity models that use rateOf, with the libRoadRunner oracle. The
# corpus is not committed on this branch, so each case skips unless its file and
# roadrunner are both present locally.

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"

# (model dir, xml filename, [(series, max_rel_tol)], t_end, n_points)
_CORPUS = [
    ("MODEL1910030001", "MODEL1910030001.xml", [("A", 1e-3), ("N_2", 1e-3)], 100.0, 21),
    ("BIOMD0000000775", "Iarosz2015.xml", [("G", 1e-6), ("N", 1e-6)], 50.0, 11),
    (
        "BIOMD0000000696",
        "BIOMD0000000696_url.xml",
        [("x8", 1e-4), ("sum_abs_dx8", 1e-4)],
        50.0,
        11,
    ),
]


@pytest.mark.parametrize("model_dir,xml,checks,t_end,n_points", _CORPUS)
def test_corpus_rateof_matches_roadrunner(model_dir, xml, checks, t_end, n_points):
    roadrunner = pytest.importorskip("roadrunner")
    path = _MODELS_DIR / model_dir / xml
    if not path.exists():
        pytest.skip(f"rr_parity corpus model not present: {path}")
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)

    sel = [name for name, _tol in checks]
    rr = roadrunner.RoadRunner(str(path))
    rr.timeCourseSelections = ["time"] + sel
    ref = np.asarray(rr.simulate(0.0, t_end, n_points))

    model = bngsim.Model.from_sbml(str(path))
    assert model._core.uses_rateof is True
    res = _run(model, t_end=t_end, n_points=n_points)

    for i, (name, tol) in enumerate(checks):
        got = _series(res, name)
        assert got is not None, f"{name} not found in bngsim result"
        want = ref[:, 1 + i]
        max_rel = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9))
        assert max_rel < tol, f"{model_dir}:{name} max_rel={max_rel:.3e} >= {tol}"


# ─── GH #231: per-row refresh + local-param / no-math-rate-rule rateOf ────────
#
# The recorded value of a rate_of__<species> accessor was the last *internal*
# integration step's derivative — and at t=0, before any step, a stale (zero)
# buffer. fill_row / the event loop now probe dx/dt at the exact recorded
# (t, y) so every row is exact. Plus two loader fixes: a kinetic-law local
# parameter is constant so rateOf(local) ≡ 0 (it also has no species index, so
# the accessor would mis-bind or fail to compile); and a rate rule with no
# <math> means dvar/dt = 0, which still promotes var to a species so a
# rateOf(var) accessor binds.


def _read_value(result, name):
    """A scalar trajectory by SBML id from any column kind."""
    for names, data in (
        ("species_names", "species"),
        ("observable_names", "observables"),
    ):
        col = list(getattr(result, names))
        if name in col:
            return np.asarray(getattr(result, data))[:, col.index(name)]
    exprs = list(result.expression_names)
    if name in exprs:
        return np.asarray(result.expressions[name])
    return None


def test_rateof_recorded_value_exact_at_t0():
    # p1 rate-ruled dp1/dt = 0.5*p1, p1(0)=2 ⇒ rateOf(p1)(0) = 1 exactly. Before
    # the per-row refresh the t=0 row read a stale derivative buffer.
    body = f"""
      <model id="m">
        <listOfParameters>
          <parameter id="p1" value="2" constant="false"/>
          <parameter id="p2" constant="false"/>
        </listOfParameters>
        <listOfRules>
          <rateRule variable="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><cn>0.5</cn><ci>p1</ci></apply></math></rateRule>
          <assignmentRule variable="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>{_RATEOF_CSYMBOL}<ci>p1</ci></apply></math></assignmentRule>
        </listOfRules>
      </model>"""
    res = _run(bngsim.Model.from_sbml_string(_sbml(body)), t_end=4.0, n_points=5)
    p1 = _read_value(res, "p1")
    p2 = _read_value(res, "p2")
    assert p2 is not None
    # rateOf(p1) == 0.5*p1 at every recorded row, including t=0.
    np.testing.assert_allclose(p2, 0.5 * p1, rtol=1e-7, atol=1e-9)
    assert abs(p2[0] - 1.0) < 1e-9


def test_rateof_of_local_parameter_is_zero():
    # The kinetic law rateOf(k) reads the kinetic-law-local k (constant) ⇒ 0, so
    # S never changes. (Previously load-failed on an unbound rate_of__k.)
    body = """
      <model id="m">
        <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
        <listOfSpecies>
          <species id="S" compartment="c" initialAmount="4" hasOnlySubstanceUnits="true"
                   boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <listOfReactions>
          <reaction id="J0" reversible="false">
            <listOfProducts><speciesReference species="S" stoichiometry="1" constant="true"/>
            </listOfProducts>
            <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>%s<ci>k</ci></apply></math>
              <listOfLocalParameters><localParameter id="k" value="3"/></listOfLocalParameters>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>""" % _RATEOF_CSYMBOL
    res = _run(bngsim.Model.from_sbml_string(_sbml(body)), t_end=5.0, n_points=6)
    S = _read_value(res, "S")
    np.testing.assert_allclose(S, 4.0, rtol=0, atol=1e-9)


def test_rateof_of_no_math_rate_rule_is_zero():
    # p has a rate rule with no <math> ⇒ dp/dt = 0 (p constant); x := rateOf(p)
    # must bind and evaluate to 0. (Previously load-failed: p stayed a parameter
    # with no rate_of__p.)
    body = f"""
      <model id="m">
        <listOfParameters>
          <parameter id="p" value="2" constant="false"/>
          <parameter id="x" constant="false"/>
        </listOfParameters>
        <listOfRules>
          <rateRule variable="p"/>
          <assignmentRule variable="x"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>{_RATEOF_CSYMBOL}<ci>p</ci></apply></math></assignmentRule>
        </listOfRules>
      </model>"""
    res = _run(bngsim.Model.from_sbml_string(_sbml(body)), t_end=3.0, n_points=4)
    p = _read_value(res, "p")
    x = _read_value(res, "x")
    np.testing.assert_allclose(p, 2.0, rtol=0, atol=1e-12)
    assert x is not None
    np.testing.assert_allclose(x, 0.0, rtol=0, atol=1e-12)


# ─── GH #231 sub-cluster 3 / 01463: rateOf of a hasOnlySubstanceUnits=true species
#
# A hOSU=true species's symbol denotes its AMOUNT, so its rateOf csymbol is the
# amount rate d(amount)/dt. The engine stores amount/V_static (rescaling to live
# volume separately), so the rateOf buffer holds d(amount)/dt / V_static and
# ×volume_factor recovers the amount rate — for a CONSTANT-volume compartment
# (01455 / 01457) AND a VARIABLE-volume compartment promoted to a state (rate-rule
# / event-resized, 01463), with no conc·V̇ correction. Off by exactly the factor V
# before the fix.


def _hosu_production_model(rule_xml: str, *, vol: float = 2.0) -> str:
    """A hOSU=true species ``A`` in a constant V=`vol` compartment, produced by a
    constant-rate reaction ``-> A`` at amount rate 2 (so rateOf(A) == 2, while the
    stored d(conc)/dt == 2/vol). ``rule_xml`` adds the rateOf consumer under test."""
    return f"""
  <model id="hosu_rateof">
    <listOfCompartments><compartment id="c" size="{vol}" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="x" value="0" constant="false"/></listOfParameters>
    <listOfRules>{rule_xml}</listOfRules>
    <listOfReactions>
      <reaction id="prod" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn>2</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""


@pytest.mark.parametrize("codegen", [False, True])
def test_rateof_hosu_amount_rate_in_rhs(codegen):
    """rateOf(A) consumed *inside* the RHS — a rate rule ``B' = rateOf(A)`` (B
    hOSU=false). B integrates the amount rate 2, so ``B(t) = 2t`` exactly (it
    would be ``t`` if rateOf reported the unscaled d(conc)/dt = 1). Exercises the
    interpreted buffer scaling and the codegen rateOf map in lock-step."""
    rule = (
        '<rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"<apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply></math></rateRule>"
    )
    model = bngsim.Model.from_sbml_string(_sbml(_hosu_production_model(rule)))
    assert model._core.uses_rateof is True
    res = _run(model, codegen=codegen, t_end=5.0, n_points=11)
    t = np.asarray(res.time)
    B = _series(res, "B")
    np.testing.assert_allclose(B, 2.0 * t, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("codegen", [False, True])
def test_rateof_hosu_amount_rate_as_output(codegen):
    """The 01455 / 01457 pattern: ``x := rateOf(A)`` reported as an output. x must
    read the amount rate 2 (d(amount)/dt) at every row, not 2/V = 1. Pins the
    output recorder — the path the SBML test suite grades."""
    rule = (
        '<assignmentRule variable="x"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"<apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply></math></assignmentRule>"
    )
    model = bngsim.Model.from_sbml_string(_sbml(_hosu_production_model(rule)))
    res = _run(model, codegen=codegen, t_end=10.0, n_points=11)
    x = _read_value(res, "x")
    assert x is not None
    np.testing.assert_allclose(x, 2.0, rtol=1e-6, atol=1e-9)


def test_rateof_hosu_variable_volume_amount_scaled():
    """A hOSU=true species in a VARIABLE-volume rate-ruled compartment IS flagged
    for the amount scaling (01463): the engine stores amount/V_static, so
    ×volume_factor recovers d(amount)/dt with no conc·V̇ term. Pins the gate — both
    the constant-volume and the rate-ruled-compartment hOSU species carry
    report_rateof_amount; an AR-compartment hOSU species (whose volume is a rule
    function, not an integrator state) does not."""
    body = """
  <model id="m">
    <listOfCompartments>
      <compartment id="cc" size="2" constant="true"/>
      <compartment id="cv" size="2" constant="false"/>
      <compartment id="ca" size="2" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="Ac" compartment="cc" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Av" compartment="cv" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Aa" compartment="ca" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="q" value="3" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="cv"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>0.1</cn></math></rateRule>
      <assignmentRule variable="ca"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>q</ci></math></assignmentRule>
    </listOfRules>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    flags = {s["name"]: s["report_rateof_amount"] for s in model._core.codegen_data()["species"]}
    assert flags["Ac"] is True  # constant-volume hOSU ⇒ amount-scaled
    assert flags["Av"] is True  # rate-ruled variable-volume hOSU ⇒ amount-scaled (01463)
    assert flags.get("Aa") is False  # AR-compartment hOSU ⇒ left unscaled (not a state)


def test_rateof_hosu_variable_volume_amount_rate_value():
    """01463 end to end: a hOSU species S under a rate rule (d(amount)/dt = 0.4) in
    a rate-ruled compartment C (dV/dt = 0.2, V(0) = 0.5), with ``x := rateOf(S)``.
    The reported amount-rate is the constant 0.4 for all t — NOT 0.4/V(0) = 0.8
    (unscaled) nor a drifting chain-rule value. Pins the ×volume_factor recovery on
    the variable-volume path."""
    body = """
  <model id="varvol_rateof">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="0.5" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="true"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="x" constant="false"/></listOfParameters>
    <listOfRules>
      <rateRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>0.4</cn></math></rateRule>
      <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>0.2</cn></math></rateRule>
      <assignmentRule variable="x"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{csym}<ci>S</ci></apply></math></assignmentRule>
    </listOfRules>
  </model>""".format(csym=_RATEOF_CSYMBOL)
    model = bngsim.Model.from_sbml_string(_sbml(body))
    res = _run(model, t_end=1.0, n_points=11)
    x = _read_value(res, "x")
    assert x is not None
    np.testing.assert_allclose(x, 0.4, rtol=1e-6, atol=1e-9)


# ─── GH #231 sub-cluster 1: rateOf in an initialAssignment ────────────────────
#
# ``p = rateOf(X)`` in an initialAssignment folds to the INITIAL dX/dt: for a
# rate-ruled X the rule's RHS (01250), for a reaction-driven species the signed
# Σ net_stoich·kineticLaw in the species' reporting units (01251/01252/01254). The
# fold reuses the same derivative the runtime rateOf buffer holds; without it the
# IA target silently keeps its declared value.


def test_rateof_in_initial_assignment_rate_rule():
    """01250: ``p2 = rateOf(p1)`` where p1 has rate rule 1 ⇒ p2 folds to 1."""
    body = """
  <model id="m">
    <listOfParameters>
      <parameter id="p1" value="2" constant="false"/>
      <parameter id="p2" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{csym}<ci>p1</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>1</cn></math></rateRule>
    </listOfRules>
  </model>""".format(csym=_RATEOF_CSYMBOL)
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p2") == pytest.approx(1.0)


def test_rateof_in_initial_assignment_reaction_species():
    """01251 / 01252: ``p2 = rateOf(S)`` folds to the signed reaction rate — negative
    when S is a reactant, positive when a product (kinetic law 1, V=1)."""

    def _model(as_product: bool) -> str:
        ref = "listOfProducts" if as_product else "listOfReactants"
        return """
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="100" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="p2" constant="false"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{csym}<ci>S</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <{ref}><speciesReference species="S" stoichiometry="1" constant="true"/></{ref}>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>""".format(csym=_RATEOF_CSYMBOL, ref=ref)

    prod = bngsim.Model.from_sbml_string(_sbml(_model(True)))
    reac = bngsim.Model.from_sbml_string(_sbml(_model(False)))
    assert prod.get_param("p2") == pytest.approx(1.0)  # product ⇒ +rate
    assert reac.get_param("p2") == pytest.approx(-1.0)  # reactant ⇒ -rate


# ─── GH #231 sub-cluster 2: rateOf in an event trigger (atol-robust firing) ────
#
# A trigger gated on rateOf fires when the CVODE root finder brackets the crossing.
# The rising-edge CONFIRMATION in the main loop must refresh the live derivative
# buffer first, else it reads a stale value and the event fires only erratically —
# seen at some solver tolerances, missed at others (01261 / 01293).


def test_rateof_event_trigger_fires_robustly_across_atol():
    """01261 pattern: p1 rate rule 0.01*p1, event ``rateOf(p1) > 0.0105`` sets
    ``p2 := 5``. rateOf(p1) = 0.01*p1 crosses at p1 = 1.05 ⇒ t = ln(1.05)/0.01. The
    fire is deterministic across solver tolerances — before the confirmation refresh
    it was seen only at some atols."""
    body = """
  <model id="m">
    <listOfParameters>
      <parameter id="p1" value="1" constant="false"/>
      <parameter id="p2" value="10" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>0.01</cn><ci>p1</ci></apply></math></rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="ev" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><apply>{csym}<ci>p1</ci></apply><cn>0.0105</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn>5</cn></math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>""".format(csym=_RATEOF_CSYMBOL)
    t_cross = math.log(1.05) / 0.01
    for atol in (1e-8, 1e-10, 1e-12, 1e-14):
        model = bngsim.Model.from_sbml_string(_sbml(body))
        res = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, 10.0), n_points=1001, rtol=1e-8, atol=atol
        )
        p2 = _read_value(res, "p2")
        assert p2 is not None, f"p2 column missing at atol={atol}"
        t = np.asarray(res.time)
        assert p2[0] == pytest.approx(10.0)
        assert p2[-1] == pytest.approx(5.0), f"event never fired at atol={atol}"
        fire_t = t[int(np.argmax(p2 < 7.5))]
        assert fire_t == pytest.approx(t_cross, abs=3 * (t[1] - t[0])), f"atol={atol}"
