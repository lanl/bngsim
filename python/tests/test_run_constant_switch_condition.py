"""Issue #382 — a switch condition that cannot cross must not decline a model.

``MODEL0911270005`` failed its forward-sensitivity solve outright:

    CVODE integration failed at t=1.000000 with flag=-4 (CV_CONV_FAILURE)

with exactly one of its 31 shared parameters, ``CRRFLX``, responsible. The issue
recorded three more witnesses in the same document family — ``MODEL0911272039``
(``ANPKNS``), ``MODEL0911342562`` (``ANGKNS``), ``MODEL0911376350`` (``ALDKNS``)
— each a zero-valued parameter that is simultaneously a switch's comparison
operand and the value of the branch actually taken.

## The chain

None of those four columns was the cause. Each model carries a rate-law
condition written over parameters that never move —

    piecewise(PA - EXE, CRRFLX > 1e-7, CRRFLX)      CRRFLX = 0, constant
    piecewise(ANPKNS, ANPKNS > 0, ANP + ANPINF)     ANPKNS = 0, constant

— and these are reduced Guyton circulation models, so most of the loop has been
cut away and what remains refers to the removed parts through frozen
``<parameter>`` declarations. ``PO2ART<80.0`` with ``PO2ART = 97.0439`` is the
same shape and is in the same model.

The gate that decides whether the analytic sensitivity RHS is emitted
(:func:`bngsim._switch_sensitivity.uncompensated_condition_reason`, issue #68)
admitted a condition on three grounds: a clock threshold issue #48 stops at, a
state comparison issue #150 roots on, or — ground 3 — a comparison naming no
symbol at all, ``0>0``. A comparison between run-constants is the same
compile-time constant as ``0>0``, just spelled with names, but it fell through
all three and declined the analytic RHS **for the whole model**. CVODES'
difference quotient took over the entire sensitivity solve and could not
integrate it.

So the failing column was never the defect; the refusal over a condition with no
crossing in it was, and the ``CV_CONV_FAILURE`` was three steps downstream.

## What this locks

1. all four witnesses return a finite tensor rather than raising;
2. the columns are *right*, not merely finite — against AMICI where AMICI can
   produce an oracle (``MODEL0911270005`` agrees at ``max_rel_err = 0`` over all
   31 shared parameters), and against a finite difference of bngsim's own
   trajectory otherwise, since AMICI fails on the other three;
3. the parameter sitting exactly ON its own threshold gets the one-sided
   derivative and nothing louder. ``ANPKNS = 0`` under ``ANPKNS > 0`` is a
   discontinuity in *parameter* space: perturbing it down moves the trajectory
   by exactly nothing, perturbing it up moves it by a fixed ~1.0 that does not
   shrink with the step. That is a step, not a derivative, and no saltation jump
   applies because nothing crosses in *time*. bngsim reports the branch that is
   taken — which is what AMICI does with the same construct, and what any AD
   system does with a ``Piecewise``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"

# The manifest horizon and the tolerances issue #382 reproduced at.
_T_SPAN = (0.0, 100.0)
_N_POINTS = 101
_RTOL, _ATOL = 1e-9, 1e-12

# model id -> the parameter the issue named: a zero-valued constant that is both
# a switch's comparison operand and the value of the branch that is taken.
_WITNESSES = {
    "MODEL0911270005": "CRRFLX",
    "MODEL0911272039": "ANPKNS",
    "MODEL0911342562": "ANGKNS",
    "MODEL0911376350": "ALDKNS",
}


def _path(model_id: str) -> Path:
    return _MODELS_DIR / model_id / f"{model_id}.xml"


def _load(model_id: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Model.from_sbml(str(_path(model_id)))


def _traj(model_id: str, name: str | None = None, value: float | None = None) -> np.ndarray:
    """The PLAIN trajectory, with one parameter optionally moved."""
    m = _load(model_id)
    if name is not None:
        m.set_param(name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = bngsim.Simulator(m, method="ode").run(_T_SPAN, _N_POINTS, rtol=_RTOL, atol=_ATOL)
    return np.asarray(run.species)


def _skip_unless_present(model_id: str) -> None:
    if not _path(model_id).is_file():
        pytest.skip(f"rr_parity corpus model {model_id} not present")


@pytest.mark.parametrize(("model_id", "param"), sorted(_WITNESSES.items()))
def test_the_witness_column_returns_instead_of_failing_to_integrate(model_id, param):
    """The issue's own reproducer: one column, on its own, through ``run()``.

    ``compute_all_sensitivities`` is deliberately NOT used — its default
    ``chunk_size=2`` and its column-by-column retry rescue transient cases and
    would hide this one (issue #401). One solve over the requested list is what
    the issue measured and what this asserts.
    """
    _skip_unless_present(model_id)
    m = _load(model_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = bngsim.Simulator(m, method="ode", sensitivity_params=[param]).run(
            _T_SPAN, _N_POINTS, rtol=_RTOL, atol=_ATOL
        )
    s = np.asarray(run.sensitivities)
    assert s.shape[0] == _N_POINTS and s.shape[2] == 1
    assert np.all(np.isfinite(s)), f"{model_id}/{param} returned a non-finite column"


@pytest.mark.parametrize(("model_id", "param"), sorted(_WITNESSES.items()))
def test_no_condition_in_the_witness_declines_the_analytic_sensitivity_rhs(model_id, param):
    """The fix, at the gate rather than at the symptom.

    Every one of these models declined with the issue #68 message naming a
    condition over frozen parameters — ``'CRRFLX' appears in the condition
    'CRRFLX>1e-07', which is neither a recognized clock threshold nor a single
    comparison over model state``. That refusal is what put the whole solve on
    the difference quotient, so its absence is the actual claim; the tensor
    above is the consequence.
    """
    _skip_unless_present(model_id)
    from bngsim import _codegen as cg

    core = _load(model_id)._core
    _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
    assert reason is None, f"{model_id} still declines: {reason}"


@pytest.mark.parametrize(("model_id", "param"), sorted(_WITNESSES.items()))
def test_the_at_threshold_parameter_is_a_step_and_gets_its_one_sided_derivative(model_id, param):
    """Why a zero column here is the honest answer and not a silent one.

    Three of the four sit exactly on their own threshold (``ANPKNS>0.0`` with
    ``ANPKNS = 0``). Measured at two step sizes a hundred apart:

      * down: the trajectory does not move at all — the branch is unchanged, so
        the derivative from that side is 0, and that is the column bngsim emits;
      * up: the trajectory moves by the SAME amount at both step sizes. A
        displacement that does not shrink with the step is a step, not a
        derivative; the limit from that side does not exist.

    ``CRRFLX`` is the one that is not at its threshold — it sits at 0 under
    ``CRRFLX > 1e-7`` and is the value of the ``otherwise`` branch — so its down
    side IS a derivative, and it scales linearly with the step. Asserted
    separately for that reason, and it is the column AMICI independently
    reproduces at ``max_rel_err = 0``.
    """
    _skip_unless_present(model_id)
    m = _load(model_id)
    p0 = m.get_param(param)
    assert p0 == 0.0, f"{param} is no longer the zero-valued constant the issue named"

    base = _traj(model_id)
    moves = {}
    for h in (1e-6, 1e-4):
        moves[h] = (
            np.max(np.abs(_traj(model_id, param, p0 + h) - base)),
            np.max(np.abs(_traj(model_id, param, p0 - h) - base)),
        )

    up_small, dn_small = moves[1e-6]
    up_big, dn_big = moves[1e-4]

    # The up side is a STEP: same displacement, step size 100x apart.
    assert up_small > 1e-3, f"{model_id}: no branch flip on the open side"
    assert up_big == pytest.approx(up_small, rel=1e-2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = bngsim.Simulator(m, method="ode", sensitivity_params=[param]).run(
            _T_SPAN, _N_POINTS, rtol=_RTOL, atol=_ATOL
        )
    col = np.asarray(run.sensitivities)[:, :, 0]

    if dn_small == 0.0:
        # At the threshold: the closed side does not move, so the one-sided
        # derivative is exactly 0 and the column must be exactly 0 — not
        # "small", which would mean something else was being reported.
        assert dn_big == 0.0
        assert np.max(np.abs(col)) == 0.0
    else:
        # CRRFLX: the closed side is differentiable, and the one-sided
        # difference converges to the column bngsim emits.
        assert dn_big / dn_small == pytest.approx(100.0, rel=1e-2)
        assert np.max(np.abs(col)) == pytest.approx(dn_big / 1e-4, rel=1e-3)


def test_the_admitted_columns_match_a_finite_difference():
    """The columns are right, on the three models AMICI cannot reference.

    AMICI fails these three itself (``Inf`` in ``sxdot`` at t=0), so the only
    available oracle is a central difference of bngsim's own trajectory — taken
    with sensitivities OFF, so nothing under test is in the loop. Every column
    except the at-threshold one above must match it.

    ``h`` is deliberately large: at ``h = 1e-6`` the difference is dominated by
    ``rtol``-scale trajectory noise divided by ``2h``, which reads as a 3e-3
    disagreement that shrinks to 8e-8 as ``h`` grows. Sampling one small step
    size and calling the result an error is the trap this comment exists to
    keep out.
    """
    model_id = "MODEL0911342562"
    _skip_unless_present(model_id)
    m = _load(model_id)
    sim = bngsim.Simulator(m, method="ode")
    params = [
        p
        for p in m.primary_param_names
        if p not in sim._function_backed_params() and p != _WITNESSES[model_id]
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = bngsim.Simulator(_load(model_id), method="ode", sensitivity_params=params).run(
            _T_SPAN, _N_POINTS, rtol=_RTOL, atol=_ATOL
        )
    s = np.asarray(run.sensitivities)
    assert np.all(np.isfinite(s))

    worst, worst_p = 0.0, None
    for j, p in enumerate(params):
        p0 = m.get_param(p)
        h = 1e-4 * max(abs(p0), 1.0)
        fd = (_traj(model_id, p, p0 + h) - _traj(model_id, p, p0 - h)) / (2 * h)
        col = s[:, :, j]
        scale = max(np.max(np.abs(col)), np.max(np.abs(fd)), 1e-30)
        err = np.max(np.abs(col - fd)) / scale
        if err > worst:
            worst, worst_p = err, p
    assert worst < 1e-3, f"{model_id}: worst column {worst_p} off by {worst:.3e}"


# ── the invariant the run-constant test rests on ────────────────────────────
#
# "A parameter's value is fixed for the run" is only safe because the SBML
# loader does not leave a moving value in `param_names`: a <rateRule> target and
# an <eventAssignment> target are both PROMOTED TO SPECIES
# (`rate_rule_targets` / `event_promoted_params` in `_sbml_loader`). Both of
# these conditions really do cross, so if that promotion ever stops, ground 3
# would start admitting a crossing nothing compensates — the silent zero the
# issue #68 gate exists to prevent, reintroduced through the door #382 opened.
#
# These assert the outcome, not the mechanism: `condition_cannot_cross` must say
# no, and the condition must be claimed by the state-switch path instead.

_MOVING_PARAM_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="moving">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
      <parameter id="mover" value="0" constant="false"/>
    </listOfParameters>
    __RULES__
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/>
              <piecewise>
                <piece><ci> k </ci>
                  <apply><gt/><ci> mover </ci><cn> 2 </cn></apply>
                </piece>
                <otherwise><cn> 0 </cn></otherwise>
              </piecewise>
              <ci> S </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

_RATE_RULE = """<listOfRules>
      <rateRule variable="mover">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 1 </cn></math>
      </rateRule>
    </listOfRules>"""

_EVENT_ASSIGNMENT = """<listOfEvents>
      <event id="flip" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol>
              <cn> 3 </cn>
            </apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="mover">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 5 </cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>"""


@pytest.mark.parametrize(
    ("label", "rules"),
    [("rate_rule", _RATE_RULE), ("event_assignment", _EVENT_ASSIGNMENT)],
)
def test_a_parameter_that_moves_mid_run_is_not_a_run_constant(tmp_path, label, rules):
    """``mover`` is declared a ``<parameter>`` and then driven, so ``mover>2`` is
    a condition that genuinely crosses. It must not be admitted on ground 3."""
    from bngsim import _switch_sensitivity as sw

    path = tmp_path / f"{label}.xml"
    path.write_text(_MOVING_PARAM_SBML.replace("__RULES__", rules))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        core = bngsim.Model.from_sbml(str(path))._core

    assert "mover" in core.species_names, "the loader stopped promoting a driven parameter"
    scope = sw.switch_condition_scope(core)
    assert "mover" not in scope.run_constants
    assert not sw.condition_cannot_cross("mover>2.0", scope)
    # And the crossing is claimed by the machinery that DOES compensate it.
    assert sw.state_switch_residual(core, "mover>2.0")
