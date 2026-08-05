"""Issue #164 — an SBML compartment size is not writable, and says so.

A compartment volume has two representations in a loaded model, and only one of
them is a parameter. The kinetic law reads ``p[]``; the *storage convention* is
folded at load into constants nothing re-derives — ``Species::volume_factor``, an
amount-declared ``initial_conc`` (= amount/V), the Elementary scalar rate's
``Π V^n / V_storage``, ``Reaction::ssa_volume_factor``, and the ``inv_vf`` table
in the emitted C. ``set_param`` moves the first and cannot reach the second.

The result is not a stale value but an internally inconsistent model. On the
two-compartment model below ``set_param("C1", 3.0)`` used to move ``A(5)`` from
22.3 to 1.11 — a factor of 20 — on a trajectory that is *exactly* ``C1``-
invariant, because ``C1`` survives in ``transport``'s law but not in the divide.
The other direction is a silent no-op: ``set_param("C2", 7.0)`` changed nothing,
because ``C2`` cancelled out of ``degB`` at load and only the stale storage
divide still read it.

**Which half of a write lands is not uniform inside one model**, which is why
this refuses rather than tries to patch a subset. Measured against RoadRunner:
a mass-action law folds the volume away entirely (write dropped); a Functional
law loaded at V ≠ 1 divides by the live compartment symbol (write honored); the
same law loaded at V = 1 had that divide normalized out (write dropped). Nothing
visible to the caller distinguishes them.

So the write is refused, and :meth:`Model.from_sbml` grows ``compartment_sizes=``
as the path that *is* correct: it moves the size before any fold happens, so a
volume scan or fit is a loop over loads. The oracle used throughout here is
loading an SBML source that carries the size outright — a rebuild is the only
thing that ever disagreed with the buggy answer, since a finite difference
through ``set_param`` inherits the same staleness — plus RoadRunner where it is
installed.

Issue #170 tracks making the volume live everywhere, which retires the refusal.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import ModelError

# ── Fixtures ────────────────────────────────────────────────────────────────
#
# Issue #164's own model: two compartments, one transport reaction across them.
# `A` is exactly C1-invariant (C1 appears once in the law and once in A's storage
# divide); `B` genuinely depends on both volumes.
XCOMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp">
    <listOfCompartments>
      <compartment id="C1" size="%(v1)s" constant="true"/>
      <compartment id="C2" size="%(v2)s" constant="true"/>
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

# Single compartment, bare `k*A` law (no compartment factor) — the shape #164
# believed was safe. It is not: d[A]/dt = k·A/V, so the trajectory moves with V
# and the write is dropped, because the volume is folded into the mass-action
# scalar. This is the common BioModels convention, so it is the wide half of the
# exposure, not an exotic case.
ONECOMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="one">
    <listOfCompartments>
      <compartment id="C" size="%(v)s" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# An initialAssignment on a compartment takes precedence over its size attribute
# at load, so an override that left it in place would be silently discarded.
IA_COMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/><parameter id="v0" value="5" constant="true"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>v0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# A compartment whose size an ASSIGNMENT RULE computes: the rule, not the size
# attribute, is its volume, so an override there would be silently recomputed
# away. Refused for the same reason set_param is.
AR_COMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar">
    <listOfCompartments><compartment id="C" size="1" constant="false"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/><parameter id="g" value="2" constant="true"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

T_SPAN = (0.0, 20.0)
N_POINTS = 5


def _traj(model, t_span=T_SPAN, n_points=N_POINTS):
    return np.asarray(
        bngsim.Simulator(model, method="ode")
        .run(t_span=t_span, n_points=n_points, rtol=1e-12, atol=1e-14)
        .species
    )


def _xcomp(v1="1", v2="5", **kw):
    return bngsim.Model.from_sbml_string(XCOMP % {"v1": v1, "v2": v2}, **kw)


# ── The marking ─────────────────────────────────────────────────────────────


def test_compartment_sizes_are_marked_and_nothing_else_is():
    m = _xcomp()
    assert m.compartment_size_params == ["C1", "C2"]
    # Rate constants are ordinary parameters and stay writable.
    assert "k" not in m.compartment_size_params
    m.set_param("k", 0.4)
    assert m.get_param("k") == 0.4


def test_net_models_mark_nothing(simple_decay_net):
    """A ``.net`` network is post-BNG2.pl: the volumes are already folded into
    its rate constants, so there is no compartment parameter to protect and the
    flag must not fire on one of its ordinary parameters."""
    m = bngsim.Model.from_net(simple_decay_net)
    assert m.compartment_size_params == []
    assert not any(m._core.param_is_compartment_size)


def test_a_promoted_compartment_is_not_flagged():
    """A rate-rule compartment is promoted to a *species* — genuine live state
    whose value the integrator owns, not a folded constant. It is not a
    parameter at all, so it must not be flagged; only the static ``dish`` is."""
    pytest.importorskip("antimony")
    m = bngsim.Model.from_antimony_string(
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; cell'=g; J1: A+B=>P; k*A*B; end"
    )
    assert "cell" in list(m.species_names)
    assert m.compartment_size_params == ["dish"]


# ── The refusal ─────────────────────────────────────────────────────────────


def test_set_param_refuses_a_compartment_write():
    m = _xcomp()
    with pytest.raises(ValueError) as exc:
        m.set_param("C2", 7.0)
    msg = str(exc.value)
    assert "C2" in msg and "compartment size" in msg
    assert "compartment_sizes=" in msg or "compartment_sizes" in msg
    assert "#164" in msg
    # And the refusal is total: nothing moved.
    assert m.get_param("C2") == 5.0
    assert m._core.codegen_data()["species"][1]["volume_factor"] == 5.0


def test_the_invented_dependence_is_refused_not_merely_dropped():
    """#164's worst symptom: a write to a compartment the trajectory is exactly
    invariant to used to *move* it by 20x. The refusal must cover this half too
    — it is the one a caller has no way to notice."""
    m = _xcomp()
    with pytest.raises(ValueError, match="C1"):
        m.set_param("C1", 3.0)
    assert np.allclose(_traj(m)[:, 0], _traj(_xcomp(v1="3"))[:, 0], rtol=1e-9)


def test_the_single_compartment_bare_law_is_refused_too():
    """#164 scoped the defect to cross-compartment models. It is wider: a bare
    ``k*A`` law (no compartment factor — the common BioModels convention) folds
    V into the mass-action scalar, so the trajectory moves with V and the write
    is silently dropped. One compartment, one reaction, still wrong."""
    m = bngsim.Model.from_sbml_string(ONECOMP % {"v": "1"})
    with pytest.raises(ValueError, match="compartment size"):
        m.set_param("C", 4.0)
    # The rebuild really does move — this is not a V-invariant model.
    assert not np.allclose(
        _traj(bngsim.Model.from_sbml_string(ONECOMP % {"v": "1"}))[:, 0],
        _traj(bngsim.Model.from_sbml_string(ONECOMP % {"v": "4"}))[:, 0],
    )


def test_writing_the_value_it_already_holds_is_allowed():
    """A fitting harness that writes back a full parameter vector round-trips
    unchanged entries through here. A write that changes nothing has nothing to
    desync, so refusing it would be a gratuitous break."""
    m = _xcomp()
    m.set_param("C2", 5.0)  # exactly the load-time size
    assert m.get_param("C2") == 5.0
    m.set_params({n: m.get_param(n) for n in m.param_names})
    assert m.get_param("C1") == 1.0 and m.get_param("C2") == 5.0


def test_set_params_stays_atomic_across_the_refusal():
    """``set_params`` documents all-or-nothing. The compartment check therefore
    runs in the validation phase, not from the apply loop — otherwise ``k``
    would be written and then the dict would raise."""
    m = _xcomp()
    with pytest.raises(ValueError, match="C2"):
        m.set_params({"k": 0.9, "C2": 7.0})
    assert m.get_param("k") == 0.3
    assert m.get_param("C2") == 5.0


def test_parameter_scan_over_a_compartment_refuses():
    """It used to return one trajectory N times."""
    sim = bngsim.Simulator(_xcomp(), method="ode")
    with pytest.raises(ValueError, match="compartment size"):
        sim.parameter_scan("C2", [5.0, 7.0, 9.0], t_span=T_SPAN, n_points=3)


# ── The sensitivity column ──────────────────────────────────────────────────


def test_sensitivity_params_refuses_a_compartment():
    with pytest.raises(ValueError) as exc:
        bngsim.Simulator(_xcomp(), method="ode", sensitivity_params=["C1", "k"])
    msg = str(exc.value)
    assert "C1" in msg and "'k'" not in msg  # only the offending column is named
    assert "#164" in msg


def test_ordinary_sensitivity_still_works():
    """The refusal is scoped to compartment columns; a rate constant on the same
    model is untouched."""
    res = bngsim.Simulator(_xcomp(), method="ode", sensitivity_params=["k"]).run(
        t_span=T_SPAN, n_points=N_POINTS, rtol=1e-10, atol=1e-12
    )
    sens = np.asarray(res.sensitivities)
    assert sens.shape == (N_POINTS, 2, 1)
    assert np.abs(sens).max() > 0.0


def test_compute_all_sensitivities_skips_compartments_with_a_warning():
    """``params=None`` means "everything computable". On an SBML model that list
    leads with the compartments, so raising would make the method unusable for
    the sake of columns nobody named — drop them, loudly."""
    sim = bngsim.Simulator(_xcomp(), method="ode")
    with pytest.warns(UserWarning, match="compartment size"):
        res = sim.compute_all_sensitivities(t_span=T_SPAN, n_points=N_POINTS)
    assert "C1" not in res.sensitivity_params and "C2" not in res.sensitivity_params
    assert "k" in res.sensitivity_params and "kb" in res.sensitivity_params


def test_compute_all_sensitivities_raises_when_asked_by_name():
    """An explicit ask gets a hard answer, not a silently smaller tensor."""
    sim = bngsim.Simulator(_xcomp(), method="ode")
    with pytest.raises(ValueError, match="compartment size"):
        sim.compute_all_sensitivities(t_span=T_SPAN, n_points=N_POINTS, params=["C1", "k"])


def test_steady_state_sensitivity_refuses_a_compartment():
    sim = bngsim.Simulator(_xcomp(), method="ode")
    with pytest.raises(ValueError, match="compartment size"):
        sim.steady_state(sensitivity_params=["C2"])


# ── The rebuild path ────────────────────────────────────────────────────────


@pytest.mark.parametrize("v2", ["7", "0.37", "12.5"])
def test_compartment_sizes_equals_editing_the_source(v2):
    """The contract: an override is exactly a load of the same document with a
    different ``size=``. Identical trajectories, bit for bit — not merely close
    — because the override runs before any interpretation, so the two loads
    build the same model."""
    edited = _traj(_xcomp(v2=v2))
    override = _traj(_xcomp(v2="5", compartment_sizes={"C2": float(v2)}))
    assert np.array_equal(edited, override)


def test_the_override_reaches_the_state_a_write_could_not():
    """#164 symptom 1, from the other side: ``volume_factor`` is the constant
    ``set_param`` cannot move, so the rebuild path is only a fix if it does.

    ``ssa_volume_factor`` is checked with it, which settles the issue's own open
    "I have not checked SSA": the SSA propensity volume is folded by the same
    load-time pass, so it is exposed to a write exactly as the ODE side is — and
    the override moves both, to the same values an edited source produces."""
    over = _xcomp(v2="5", compartment_sizes={"C2": 7.0})._core.codegen_data()
    edit = _xcomp(v2="7")._core.codegen_data()
    assert [s["volume_factor"] for s in over["species"]] == [1.0, 7.0]
    assert [r["ssa_volume_factor"] for r in over["reactions"]] == [1.0, 7.0]
    assert [s["volume_factor"] for s in over["species"]] == [
        s["volume_factor"] for s in edit["species"]
    ]
    assert [r["ssa_volume_factor"] for r in over["reactions"]] == [
        r["ssa_volume_factor"] for r in edit["reactions"]
    ]


def test_a_volume_scan_is_a_loop_over_loads():
    """#164 symptom 4 was a scan returning one trajectory three times. The
    supported replacement must produce three *different* ones, each equal to a
    rebuild at that volume."""
    seen = []
    for v in (5.0, 7.0, 9.0):
        seen.append(_traj(_xcomp(v2="5", compartment_sizes={"C2": v}))[:, 1])
    for a, b in zip(seen, seen[1:], strict=False):
        assert not np.allclose(a, b)
    for v, got in zip(("5", "7", "9"), seen, strict=False):
        assert np.array_equal(got, _traj(_xcomp(v2=v))[:, 1])


def test_the_gradient_is_available_by_rebuilding():
    """What the refused column would have answered. #164 measured the reported
    ``dB/dC2`` as 0 against a true 2.30; a central difference over two loads
    recovers it, and this is the recipe the error message points at."""
    h = 1e-5
    lo = _traj(_xcomp(v2="5", compartment_sizes={"C2": 5.0 - h}), n_points=21)
    hi = _traj(_xcomp(v2="5", compartment_sizes={"C2": 5.0 + h}), n_points=21)
    dB_dC2 = (hi[:, 1] - lo[:, 1]) / (2 * h)
    assert np.abs(dB_dC2).max() == pytest.approx(2.3011, rel=1e-3)
    # ...and dA/dC1 really is zero, which is what made the reported 36.6 wrong.
    lo = _traj(_xcomp(compartment_sizes={"C1": 1.0 - h}), n_points=21)
    hi = _traj(_xcomp(compartment_sizes={"C1": 1.0 + h}), n_points=21)
    assert np.abs((hi[:, 0] - lo[:, 0]) / (2 * h)).max() < 1e-4


def test_override_drops_a_competing_initial_assignment():
    """An ``initialAssignment`` on a compartment wins over ``size=`` at load, so
    an override that left it standing would be silently discarded — the exact
    failure this issue is about."""
    base = bngsim.Model.from_sbml_string(IA_COMP)
    assert base.get_param("C") == 5.0  # the IA, not size="1"
    over = bngsim.Model.from_sbml_string(IA_COMP, compartment_sizes={"C": 3.0})
    assert over.get_param("C") == 3.0
    assert over._core.codegen_data()["species"][0]["volume_factor"] == 3.0


def test_override_refuses_an_assignment_rule_compartment():
    """Its volume is the rule's output, recomputed every step; the size
    attribute is not the model's volume and overriding it would do nothing."""
    with pytest.raises(ModelError, match="assignment rule"):
        bngsim.Model.from_sbml_string(AR_COMP, compartment_sizes={"C": 3.0})


@pytest.mark.parametrize(
    "sizes, match",
    [
        ({"nope": 2.0}, "unknown compartment"),
        ({"C2": 0.0}, "finite and positive"),
        ({"C2": -1.0}, "finite and positive"),
        ({"C2": float("inf")}, "finite and positive"),
        ({"C2": "big"}, "expected a number"),
    ],
)
def test_override_validation(sizes, match):
    with pytest.raises(ModelError, match=match):
        _xcomp(compartment_sizes=sizes)


def test_override_rejected_for_net_models(simple_decay_net):
    """There is no compartment left in a generated network to override, so a
    dict here would do nothing. Say so."""
    with pytest.raises(ModelError, match="not supported for .net"):
        bngsim.Model.load(simple_decay_net, compartment_sizes={"cell": 2.0})


# ── Independent oracle ──────────────────────────────────────────────────────


@pytest.mark.parametrize("v", [2.0, 4.0])
def test_rebuild_matches_roadrunner(v):
    """The rebuild is not merely self-consistent: it is what SBML means. This is
    the measurement that widened #164's scope — RoadRunner moves with V on the
    bare-law single-compartment model, which #164 expected to be V-invariant, so
    a dropped write there is a wrong answer and not a harmless no-op."""
    roadrunner = pytest.importorskip("roadrunner")
    src = ONECOMP % {"v": "1"}
    rr = roadrunner.RoadRunner(ONECOMP % {"v": repr(v)})
    rr.integrator.relative_tolerance = 1e-12
    rr.integrator.absolute_tolerance = 1e-14
    truth = np.asarray(rr.simulate(T_SPAN[0], T_SPAN[1], N_POINTS, ["time", "[A]"]))[:, 1]
    got = _traj(bngsim.Model.from_sbml_string(src, compartment_sizes={"C": v}))[:, 0]
    assert np.allclose(got, truth, rtol=1e-6, atol=1e-9)
