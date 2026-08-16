"""Issue #313 — a parameter that reaches the model only through an
``<initialAssignment>`` was frozen, in both directions and both silently.

bngsim evaluates every ``<initialAssignment>`` once at load and hands the
builder a number. A parameter whose only route into the model is that
expression therefore became a dangling constant: ``set_param`` on it succeeded
and moved nothing, and — the half a trajectory cannot see — its forward
sensitivity was identically zero at every species and every time, because the
symbol was absent from the RHS the chain rule differentiates.

``BIOMD0000000569`` is the case that found it. The document defines a chain of
initialAssignments (``BSk0 = BSk1*BSc^p``, ``BSk1 = BSk2*BSc^p``, …), puts the
``BSk*`` constants in the rate laws, and mentions ``BSc`` nowhere else, so a 50%
write to ``BSc`` left all four derived constants where they were. AMICI keeps the
dependency symbolic and both its trajectory and its sensitivities respond.
8.5% of the vendored BioModels corpus (113 of 1324 models, 678 parameters) has
at least one such parameter; it is the ordinary COPASI spelling of a derived rate
constant.

The fix is issue #170's lift, applied to every parameter initialAssignment
rather than only the volume-dependent ones: the target becomes a *derived*
parameter, and #43's chain rule re-derives it on a write and differentiates
through it for the sensitivity.

The fixture is deliberately written in the order 569 uses — each target declared
*before* the parameter it reads — because that is what the lift has to survive.
Derived parameters are re-evaluated in one pass over the parameter list, so the
lifted targets have to be emitted in dependency order, not document order; lift
them in place and ``k0`` reads a stale ``k1`` for one write.

The oracle is exact. ``A' = -k0·A`` with ``k0 = k1·c`` and ``k1 = k2·c`` is
``A(t) = A0·exp(-k2·c²·t)``, so both the written trajectory and ``dA/dc`` have a
closed form to check against, and the rebuild-at-the-new-value comparison is the
same document loaded with the new number in it.
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest

A0 = 1.0
K2 = 3.0
C_LOAD = 0.5
T_SPAN = (0.0, 2.0)
N_POINTS = 5

# ── Fixture ─────────────────────────────────────────────────────────────────
#
# `k0 = k1*c`, `k1 = k2*c`, and only `k0` appears in a rate law. `c` and `k2`
# are the primaries; `k0` and `k1` are declared BEFORE what they read.

MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_param_chain">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="{a0}" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k0" value="-1" constant="true"/>
      <parameter id="k1" value="-1" constant="true"/>
      <parameter id="k2" value="{k2}" constant="true"/>
      <parameter id="c" value="{c}" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="k0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k1</ci><ci>c</ci></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="k1">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k2</ci><ci>c</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k0</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _src(c=C_LOAD, k2=K2, a0=A0):
    return MODEL.format(c=repr(float(c)), k2=repr(float(k2)), a0=repr(float(a0)))


def _traj(model, **kw):
    return np.asarray(
        bngsim.Simulator(model, method="ode", **kw)
        .run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14)
        .species
    )


# ── The write propagates ────────────────────────────────────────────────────


def test_the_load_time_values_are_the_folded_chain():
    """Nothing may move at the nominal point: the lift only changes what a
    *write* reaches, and a derived parameter is not re-evaluated until one
    arrives."""
    m = bngsim.Model.from_sbml_string(_src())
    assert m.get_param("k1") == K2 * C_LOAD
    assert m.get_param("k0") == K2 * C_LOAD * C_LOAD


@pytest.mark.parametrize("c_new", [1.0, 0.25])
def test_set_param_moves_the_whole_chain_in_one_pass(c_new):
    """The regression test for the declaration order.

    ``k0`` is declared before the ``k1`` it reads. One re-derivation pass in
    document order would leave ``k0 = k1_old·c_new`` — right only after a second
    write.
    """
    m = bngsim.Model.from_sbml_string(_src())
    m.set_param("c", c_new)
    assert m.get_param("k1") == K2 * c_new
    assert m.get_param("k0") == K2 * c_new * c_new


def test_the_write_reproduces_the_rebuild():
    """The contract a write is held to: the same document loaded with the new
    number in it."""
    c_new = 1.0
    written = bngsim.Model.from_sbml_string(_src())
    written.set_param("c", c_new)
    rebuilt = bngsim.Model.from_sbml_string(_src(c=c_new))
    assert np.array_equal(_traj(written), _traj(rebuilt))


def test_the_written_trajectory_matches_the_closed_form():
    c_new = 1.0
    m = bngsim.Model.from_sbml_string(_src())
    m.set_param("c", c_new)
    x = _traj(m)
    t = np.linspace(*T_SPAN, N_POINTS)
    assert np.allclose(x[:, 0], A0 * np.exp(-K2 * c_new * c_new * t), rtol=1e-8)


def test_the_chain_only_parameter_is_derived_not_primary():
    """The cost, pinned. ``k0``/``k1`` are defined by the document, so they drop
    out of ``primary_param_names`` exactly as ``_rateLaw_<rid>`` does, and the
    two parameters that actually carry the model stay."""
    m = bngsim.Model.from_sbml_string(_src())
    assert "k0" not in m.primary_param_names
    assert "k1" not in m.primary_param_names
    assert "c" in m.primary_param_names
    assert "k2" in m.primary_param_names


# ── The sensitivity is no longer an identical zero ──────────────────────────


def test_the_sensitivity_of_a_chain_only_parameter_matches_the_closed_form():
    """``A(t) = A0·exp(-k2·c²·t)`` ⇒ ``dA/dc = -2·k2·c·t·A(t)``.

    The pre-#313 answer was `0` at every species and every time — not merely
    inaccurate, since `c` was absent from the RHS being differentiated.
    """
    m = bngsim.Model.from_sbml_string(_src())
    r = bngsim.Simulator(m, method="ode", sensitivity_params=["c"]).run(
        t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14
    )
    s = np.asarray(r.sensitivities)[:, 0, 0]  # dA/dc
    t = np.linspace(*T_SPAN, N_POINTS)
    expected = -2 * K2 * C_LOAD * t * A0 * np.exp(-K2 * C_LOAD * C_LOAD * t)
    assert np.abs(s).max() > 0
    assert np.allclose(s, expected, rtol=1e-6, atol=1e-9)


def test_the_sensitivity_agrees_with_a_finite_difference_through_set_param():
    """The two halves of the bug were one bug: the FD oracle only exists because
    the write now propagates, and it agrees with the analytic column."""
    h = 1e-6
    r = bngsim.Simulator(
        bngsim.Model.from_sbml_string(_src()), method="ode", sensitivity_params=["c"]
    ).run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14)
    s = np.asarray(r.sensitivities)[:, 0, 0]

    def at(c):
        m = bngsim.Model.from_sbml_string(_src())
        m.set_param("c", c)
        return _traj(m)[:, 0]

    fd = (at(C_LOAD + h) - at(C_LOAD - h)) / (2 * h)
    assert np.allclose(s, fd, rtol=1e-5, atol=1e-8)


# ── What cannot be lifted is no longer silent ───────────────────────────────

UNLIFTABLE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_unliftable">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="2" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k0" value="-1" constant="true"/>
      <parameter id="q" value="0.25" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="k0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>q</ci><ci>A</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k0</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_an_initial_assignment_reading_a_plain_species_now_lifts():
    """``k0 = q*A`` reads a species, meaning A's *initial value*.

    This used to be refused, which froze ``q`` behind the fold. An
    initialAssignment is a t=0 evaluation, so a species whose own IC is a
    declared constant contributes a constant: it is substituted and ``q`` stays
    symbolic (issue #379). The folded value is unchanged either way.
    """
    m = bngsim.Model.from_sbml_string(UNLIFTABLE)
    assert m.get_param("k0") == 0.5
    # `q` is no longer frozen: a write moves the derived constant.
    m.set_param("q", 0.5)
    assert m.get_param("k0") == 1.0


def test_an_initial_assignment_that_cannot_be_lifted_names_what_it_freezes(caplog):
    """The residue, on the shape the substitution above must NOT take.

    ``A`` is ``hasOnlySubstanceUnits``, so section 0 binds its symbol to an
    *amount* — ``conc*V`` — and substituting the number would bake a writable
    compartment size into the lifted expression as a literal. That is the fold
    #164 refuses a size over, one layer down, so the lift declines and ``q``
    stays frozen. Say so, rather than let ``set_param`` take and a sensitivity
    column read a confident zero.
    """
    unliftable = UNLIFTABLE.replace(
        '<species id="A" compartment="C" initialConcentration="2" hasOnlySubstanceUnits="false"',
        '<species id="A" compartment="C" initialConcentration="2" hasOnlySubstanceUnits="true"',
    )
    assert unliftable != UNLIFTABLE, "fixture text drifted"
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        m = bngsim.Model.from_sbml_string(unliftable)
    assert m.get_param("k0") == 0.5
    frozen = [r.getMessage() for r in caplog.records if "frozen" in r.getMessage()]
    assert len(frozen) == 1
    assert "q" in frozen[0]


def test_a_liftable_model_says_nothing(caplog):
    """The warning is the residue, not the common case: the chain fixture is
    fully lifted and must load quietly."""
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(_src())
    assert [r.getMessage() for r in caplog.records if "frozen" in r.getMessage()] == []


# ── An assignmentRule target is not a symbol the lift can rest on ───────────
#
# `q` is defined by an assignmentRule that reads a species, so its parameter
# slot is function-backed: the engine rewrites it from the rule before every
# derivative evaluation. `k0 = q*2` is therefore NOT liftable — lift it and the
# one-pass re-derivation reads whatever that slot happens to hold, which after a
# run is `q` at the last integrated point rather than the t=0 value the
# initialAssignment means. BIOMD0000000570 (`ModelValue_60 = O2c_bar`) is the
# corpus case; 21 more models carry the same shape.

AR_DEPENDENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_over_assignment_rule">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="2" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k0" value="-1" constant="true"/>
      <parameter id="q" value="-1" constant="false"/>
      <parameter id="base" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="k0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>q</ci><cn>2</cn></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <assignmentRule variable="q">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>base</ci><ci>A</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k0</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

K0_AT_LOAD = 2.0  # q = base*A = 0.5*2 at t=0, so k0 = q*2


def test_an_initial_assignment_over_an_assignment_rule_target_is_not_lifted():
    m = bngsim.Model.from_sbml_string(AR_DEPENDENT)
    assert m.get_param("k0") == K0_AT_LOAD
    assert not m.param_is_expression[list(m.param_names).index("k0")]
    assert "k0" in m.primary_param_names


def test_a_write_after_a_run_does_not_move_it_to_the_rules_current_value():
    """The failure the refusal exists for, at its sharpest: an *identity* write,
    which :meth:`set_param` documents as not an override at all, so
    ``set_params(dict(zip(param_names, vec)))`` round-trips unchanged."""
    m = bngsim.Model.from_sbml_string(AR_DEPENDENT)
    bngsim.Simulator(m, method="ode").run(t_span=T_SPAN, n_points=N_POINTS)
    assert m.get_param("q") != pytest.approx(K0_AT_LOAD / 2), "precondition: the rule moved"
    m.set_param("base", m.get_param("base"))
    assert m.get_param("k0") == K0_AT_LOAD
