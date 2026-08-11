"""A registered time-discontinuity root has to be *reachable* (issue #305).

GH #72 registers every ``time`` inequality in a piecewise as a CVODE root, so
the integrator stops at each pulse edge instead of stepping over it. That is
only half of reaching the crossing. **CVODE tests for a root solely on a step it
accepts**, and where the branch jump is large enough that the local error test
rejects every step containing the crossing, the accepted steps land short: ``h``
shrinks, ``t`` creeps up to the last representable double below ``t*``, and from
there every step that would carry it across is smaller than one ulp, so
``t + h == t``. The run dies with the issue #54 stall error having never once
evaluated ``g`` past the crossing — the registered root never fires.

Weber_BMC2015 (a `piecewise` on ``(time - PdBu_time) < 0`` at a fixed
``PdBu_time = 24``) loses 6% of a fitting box to this, with **zero** root
returns in the whole run and half of 20,000 steps rejected. It is not a
sensitivity defect: the plain state solve dies identically, and it is not honest
stiffness either — the same parameter points integrate at the same tolerances
the moment the step is made to land on the crossing.

The fix resolves each registered condition to a crossing *time* and hands it to
``CVodeSetStopTime`` (``SolverOptions.set_crossing_stop_times``), which is the
mechanism issue #48 already uses for a crossing a *fitted* parameter moves —
here applied to the far more common crossing that nothing moves and that
therefore has no ``dt*/dp`` to jump by.

What this locks:

  1. the wedging model integrates, over the (jump x atol) grid that used to
     select the failure, and hits a closed-form oracle;
  2. the crossing is resolved from the spelling PEtab exports,
     ``(time - T) < 0``, and not only from the bare ``time < T`` that
     ``_clock_threshold_split`` recognizes;
  3. crossings are resolved against LIVE parameter values, so a phase whose
     condition parameter puts the crossing outside the window gets no stop;
  4. a condition over model state resolves to nothing (its crossing moves with
     the trajectory — issue #150's business, not a fixed stop);
  5. a model with no time-dependent piecewise gets no stops at all, so its
     stepping is untouched.
"""

import math

import bngsim
import pytest

# ── Model ───────────────────────────────────────────────────────────────────
# dX/dt = kin*u - d*X with u = piecewise(0, (time - T) < 0, dose): X sits at
# rest until T and the RHS then jumps to kin*dose. The jump is what the error
# test cannot swallow, so `kin` is the knob that selects the failure.
_TIME = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>'
)

_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="fixed_crossing">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kin" value="{kin!r}" constant="true"/>
      <parameter id="d" value="1" constant="true"/>
      <parameter id="T" value="{tstar!r}" constant="false"/>
      <parameter id="dose" value="1" constant="true"/>
      <parameter id="u" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="u">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece>
              <cn type="integer"> 0 </cn>
              <apply><lt/>{cond}</apply>
            </piece>
            <otherwise><ci> dose </ci></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="prod" reversible="false">
        <listOfProducts>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> kin </ci><ci> u </ci><ci> C </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> d </ci><ci> X </ci><ci> C </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# The PEtab spelling — `time` is not bare on either side, which is exactly what
# `_clock_threshold_split` declines and what #259 taught root registration to
# admit.
SHIFTED = f'<apply><minus/>{_TIME}<ci> T </ci></apply><cn type="integer"> 0 </cn>'
# The spelling `_clock_threshold_split` does recognize, for contrast.
BARE = f"{_TIME}<ci> T </ci>"

T_STAR = 24.0


def _write(tmp_path, *, kin, cond=SHIFTED, tstar=T_STAR, name="m.xml"):
    path = tmp_path / name
    path.write_text(_SBML.format(kin=float(kin), tstar=float(tstar), cond=cond))
    return str(path)


def _oracle(t, *, kin, dose=1.0, d=1.0, tstar=T_STAR):
    """X(t) for t >= tstar, from X(tstar) = 0: the step response of dX/dt =
    kin*dose - d*X."""
    return (kin * dose / d) * (1.0 - math.exp(-d * (t - tstar)))


@pytest.mark.parametrize("kin", [1e6, 1e10, 1e14])
@pytest.mark.parametrize("atol", [1e-4, 1e-8, 1e-12])
def test_fixed_crossing_is_reachable_across_the_jump_tolerance_grid(tmp_path, kin, atol):
    """Every cell of the grid integrates, and lands on the closed form.

    Pre-fix this table is the failure selector: `kin=1e14` wedges at `atol=1e-4`,
    `kin=1e10` at `1e-8`, and even `kin=1e6` at `1e-12` — the governing quantity
    is the jump-to-atol ratio, which is why a *tighter* tolerance makes it
    strictly worse and why "does it move with the tolerance?" is the wrong
    question to separate this from a stiffness problem.
    """
    model = bngsim.Model.from_sbml(_write(tmp_path, kin=kin))
    sim = bngsim.Simulator(model, "ode")
    result = sim.run((0.0, 30.0), n_points=31, rtol=1e-6, atol=atol)

    x = result.species[:, result.species_names.index("X")]
    assert x[24] == pytest.approx(0.0, abs=1e-9 * kin), "the dose starts at t = 24"
    assert x[-1] == pytest.approx(_oracle(30.0, kin=kin), rel=1e-5)


def test_the_stall_is_at_the_crossing_and_not_the_post_jump_dynamics(tmp_path):
    """The wedging cell integrates in about as many steps as the mild one.

    The pre-fix failure spends 20,000 steps, half of them rejected, creeping
    towards `t*`. If what was hard were the post-jump problem the step count
    would still be large here; it is not, because nothing about the post-jump
    problem was ever the difficulty.
    """
    sims = []
    for kin in (1e6, 1e14):
        model = bngsim.Model.from_sbml(_write(tmp_path, kin=kin, name=f"k{kin:g}.xml"))
        sim = bngsim.Simulator(model, "ode")
        result = sim.run((0.0, 30.0), n_points=31, rtol=1e-6, atol=1e-4)
        sims.append(result.solver_stats)

    mild, violent = sims
    assert violent["n_steps"] < 3 * mild["n_steps"]
    # The creep IS the error-test failures, so it is what has to be gone.
    assert violent["n_err_test_fails"] <= 5


@pytest.mark.parametrize("cond,label", [(SHIFTED, "(time - T) < 0"), (BARE, "time < T")])
def test_both_spellings_of_the_same_crossing_resolve(tmp_path, cond, label):
    """`(time - T) < 0` and `time < T` are one crossing and must both stop.

    The shifted form is what a PEtab export writes and what Weber_BMC2015
    carries. It is registered as a root either way (#259); what it was missing
    is a resolved crossing *time*, since `_clock_threshold_split` — which
    governs the #48 sensitivity-compensation path — requires a bare clock symbol
    on one side and declines this.
    """
    from bngsim._switch_sensitivity import fixed_time_crossings

    model = bngsim.Model.from_sbml(_write(tmp_path, kin=1e14, cond=cond, name="spell.xml"))
    assert model._time_disc_conditions, f"{label}: no discontinuity trigger registered"
    assert fixed_time_crossings(model._core, 0.0, 30.0, model._time_disc_conditions) == [T_STAR], (
        label
    )

    sim = bngsim.Simulator(model, "ode")
    result = sim.run((0.0, 30.0), n_points=31, rtol=1e-6, atol=1e-4)
    x = result.species[:, result.species_names.index("X")]
    assert x[-1] == pytest.approx(_oracle(30.0, kin=1e14), rel=1e-5)


def test_crossings_are_resolved_against_live_parameter_values(tmp_path):
    """A stop is only ever armed for a phase that actually has the crossing.

    This is what makes a pre-equilibration protocol safe. The same condition
    parameter puts Weber's crossing at `t = 24` in the measured phase and at
    `t = 0` in the equilibration that precedes it; arming 24 in the second one
    is a pure perturbation of its steady-state march, and moved two of eight
    box points by 100% when the prototype did exactly that.
    """
    from bngsim._switch_sensitivity import fixed_time_crossings

    model = bngsim.Model.from_sbml(_write(tmp_path, kin=1e6, name="live.xml"))
    conds = model._time_disc_conditions

    assert fixed_time_crossings(model._core, 0.0, 30.0, conds) == [24.0]
    model.set_param("T", 12.5)
    assert fixed_time_crossings(model._core, 0.0, 30.0, conds) == [12.5]
    model.set_param("T", 0.0)
    # On the window's own start: nothing to stop at, exactly as the #48 filter
    # treats a crossing at t_start.
    assert fixed_time_crossings(model._core, 0.0, 30.0, conds) == []
    model.set_param("T", 100.0)
    assert fixed_time_crossings(model._core, 0.0, 30.0, conds) == []


def test_a_state_threshold_resolves_to_no_stop(tmp_path):
    """`X < 1` has no fixed crossing time and must not produce one.

    Its crossing moves with the trajectory, which is issue #150's saltation
    root, not a stop time. Two probes in `time` cannot answer it, and answering
    it wrongly would clamp a step onto a time nothing happens at.
    """
    from bngsim._switch_sensitivity import fixed_time_crossings

    model = bngsim.Model.from_sbml(_write(tmp_path, kin=1e6, name="state.xml"))
    assert fixed_time_crossings(model._core, 0.0, 30.0, ("(X<1)",)) == []
    # …and a condition in which `time` cancels has no crossing either.
    assert fixed_time_crossings(model._core, 0.0, 30.0, ("((time()-time())<1)",)) == []


def test_a_model_with_no_time_piecewise_gets_no_stops(tmp_path):
    """No discontinuity trigger ⇒ no crossing stops ⇒ stepping untouched."""
    from bngsim._switch_sensitivity import fixed_time_crossings

    plain = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="plain">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="d" value="1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> d </ci><ci> X </ci><ci> C </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""
    path = tmp_path / "plain.xml"
    path.write_text(plain)
    model = bngsim.Model.from_sbml(str(path))
    assert model._core.n_discontinuity_triggers == 0
    assert model._time_disc_conditions == ()
    assert fixed_time_crossings(model._core, 0.0, 30.0, model._time_disc_conditions) == []


def test_the_crossing_stop_survives_a_model_clone(tmp_path):
    """A cloned model carries the registered conditions.

    The parallel sensitivity and parameter-scan paths run on clones, and a clone
    that lost them would silently revert to the pre-fix stepping on exactly the
    workloads that run the most integrations.
    """
    model = bngsim.Model.from_sbml(_write(tmp_path, kin=1e14, name="clone.xml"))
    clone = model.clone()
    assert clone._time_disc_conditions == model._time_disc_conditions

    result = bngsim.Simulator(clone, "ode").run((0.0, 30.0), n_points=31, rtol=1e-6, atol=1e-4)
    x = result.species[:, result.species_names.index("X")]
    assert x[-1] == pytest.approx(_oracle(30.0, kin=1e14), rel=1e-5)
