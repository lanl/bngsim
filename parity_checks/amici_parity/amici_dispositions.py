"""AMICI-calibrated per-row dispositions for the amici_parity suites (issue #380).

`amici_parity` deliberately honors only the engine-agnostic ``tol`` overrides that
ride the shared manifest, and refuses rr_parity's ``known_artifact`` /
``invalid_reference`` / ``no_oracle_adjudicated`` entries: those are calibrated
against *RoadRunner*, so applying them here could mask a genuine bngsim-vs-AMICI
difference. That reasoning is sound and unchanged. The gap it left is that there
was no **amici-calibrated** path either — when AMICI is the wrong engine, or when
neither engine is wrong and the metric simply cannot resolve a cell, the row had
nowhere to go and scored as a bngsim ``DIFF``.

This module is that path. It is deliberately NOT baked into the job manifest the
way rr_parity's ``overrides.py`` is: the SBML manifest is shared with rr_parity
and is built by rr_parity's builder, so an AMICI-calibrated entry written into it
would leak into the RoadRunner suite. The runners read this module directly
instead.

Two dispositions, and the difference between them is which engine (if either) is
at fault:

  INVALID_REFERENCE    — AMICI *ran* and returned finite numbers, but a defect in
                         AMICI's own handling of the model makes them unusable as
                         an oracle. bngsim ran fine, so there is no parity verdict
                         to be had: the row is reclassified ``DIFF`` ->
                         ``REFERENCE_FAILED`` (non-scoring) with
                         ``reference_refusal = "invalid_result"``.

                         REFERENCE_FAILED and not PASS, because PASS asserts the
                         engines agreed and they did not — this is the mirror of
                         what #319/#323 did for bngsim's own declared refusals
                         (UNSUPPORTED rather than a silent EXCEPTION): name the
                         gap, do not claim the win. REFERENCE_FAILED and not
                         BAD_TEST, because BAD_TEST means neither engine could run
                         the model; here bngsim did, and that is a capability win,
                         not a bad test. (rr_parity's ``invalid_reference`` lands
                         on BAD_TEST for exactly the reason it does not apply
                         here: there, bngsim had *also* failed.)

  COMPARISON_ARTIFACT  — neither engine is wrong. The two agree everywhere the
                         quantity is resolvable, and the residual is a property of
                         comparing at a point where no oracle can resolve the
                         answer (a discontinuity edge). Reclassified ``DIFF`` ->
                         ``PASS`` with the reason recorded — never a silent pass.

                         This is NARROWER than rr_parity's ``KNOWN_ARTIFACT``,
                         which also covers "the reference engine has the bug".
                         That case is INVALID_REFERENCE here, precisely so a
                         reference defect is never dressed up as agreement.

Both are keyed ``"<model_id>:<regime>"`` with regime ``ode`` (``amici_run.py``) or
``sens`` (``amici_sens_run.py``) — regime-scoped like rr_parity's keys, but NOT
scoped to the CVODES corrector method: #325 measured every one of these models
producing a near-identical ``max_rel`` under both ``staggered`` and
``simultaneous``, so a per-method key would be two copies of one fact.

The bar is the same as rr_parity's, and it is high: an entry that turned a real
bngsim defect non-scoring would hide a regression. Every entry MUST cite evidence
that names which engine is wrong and how we know, from an oracle that is NOT
bngsim — otherwise bngsim would be adjudicating its own case. In practice that
means RoadRunner as an independent third engine, with the perturbation applied to
the SBML *text* so no engine can discard the write.

Self-maintaining, three ways, because a disposition that could go quietly wrong is
worse than none:

  * it applies ONLY to a ``DIFF``. A row that PASSes on its own is flagged STALE
    (prune it); a row that is EXCEPTION / REFERENCE_FAILED / TIMEOUT / BAD_TEST /
    UNSUPPORTED produced no comparison at all, so the entry is reported
    *inapplicable* rather than stale — a timeout falsifies nothing.
  * the reason is written into ``JobResult.comment`` on every applied row, so the
    report says why in the row itself and not only here.
  * an entry may record the ``max_rel`` it was authored against. If a later sweep
    disagrees by more than 2x, the row is still disposed of but the comment says
    DRIFTED: the disagreement changed character and the attribution is no longer
    known to cover it. That is the guard against an entry silently absorbing a
    *different*, real defect on the same model.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:  # so `from _core import ...` resolves standalone
    sys.path.insert(0, str(HERE.parent))
from _core import Outcome  # noqa: E402

# The ``reference_refusal`` class an INVALID_REFERENCE row carries. The other
# classes (``feature_gap`` / ``compile`` / ``integrator`` / ``other``) are
# auto-derived from an AMICI *exception*; this one has no exception behind it —
# AMICI answered, and the answer is what is wrong — so it can only be authored.
# It is settled by construction: an entry exists only with third-oracle evidence,
# so a future sweep should not re-triage it.
INVALID_RESULT_REFUSAL = "invalid_result"

REGIMES = ("ode", "sens")


# --------------------------------------------------------------------------- #
# AMICI ran, and its answer is not a usable oracle -> DIFF becomes
# REFERENCE_FAILED / invalid_result. Keyed "model_id:regime".
# --------------------------------------------------------------------------- #
INVALID_REFERENCE: dict[str, dict] = {
    # NOTE: MODEL1607210000:sens — #380's first row, NOT authored, because #383
    # landed first and the row is no longer a DIFF. #380 measured it at HEAD
    # d5c4323 as max_rel=1 with 2849 hard-failing cells, and attributed that to
    # AMICI being inert in 30 of its own 31 free parameters. What #383 fixed was
    # the OTHER side of the same construct: bngsim could not keep those
    # <initialAssignment>s symbolic (the #313 freeze warning fired on this model,
    # naming Saci1181KO, v12_k1, v1a_v, v3_k1, v9_k1) and so carried no
    # initial-condition term for them. With that seeded, the row is PASS at
    # max_rel=0 on BOTH corrector methods — 0 of 31310 cells failing, states ok —
    # so there is nothing left to dispose of, and an entry here would be dead
    # weight the runner would only flag STALE.
    #
    # The exact zero is the MODEL's answer, not a shared engine gap (issue #387,
    # which this NOTE used to leave open pending a third engine). Run on a venv
    # that has RoadRunner, the 1% SBML-TEXT perturbation moves RoadRunner by
    # exactly 0 as well — so all three engines agree, and #380's RoadRunner
    # column (0.069 / 0.0098 / 0.089 / 0.089 / 0.089) does not reproduce. It is
    # not just those five parameters: all 32 free parameters (27 global, 5
    # kineticLaw-local) move both engines by 0, bar 1.11e-16 — one ulp — on
    # Saci1181KO / v7b_k1 / v17_k1.
    #
    # The model cannot move. Every one of its nine <initialAssignment>s sets a
    # species' first-order decay constant to that species' TOTAL production flux
    # at t=0 over its own initial concentration (k4 = v3_k1*saci1181/Saci1181;
    # k11b = v12_k1*Saci1181KO*Saci1181*AbfR_P/AbfR; and so on), so t=0 is an
    # exact fixed point BY CONSTRUCTION for any values of the production
    # parameters — and a fixed point stays one for all t. Scaling a production
    # flux scales the matched decay constant identically. For k11b the cancelling
    # pair is visible in the reactions: v11b (AbfR->AbfR_P at k11b*AbfR) against
    # v12 (AbfR_P->AbfR at v12_k1*Saci1181KO*Saci1181*AbfR_P), and k11b is
    # DEFINED as the ratio that makes the two identical.
    #
    # The control, because a sweep of zeros on its own proves only that nothing
    # was measured: Starvation ships at 0 — which is also why a multiplicative
    # perturbation OF it reads zero, and why the five kineticLaw-local parameters
    # do, all of them being in the Starvation-gated v1c / v5b / v11. At
    # Starvation=1 the model comes alive (trajectory drift 213.9, the two engines
    # agreeing to 1.4e-6 relative) and those same five writes move BOTH engines
    # by identical amounts: 0.205 / 0.276 / 0.987 / 5.99 / 4.93.
    #
    # So #383 moved bngsim off a wrong non-zero column onto a right zero one.
    # #380's RoadRunner column is most likely a RUNTIME write (rr[pid] = ...)
    # rather than a text edit: RoadRunner does not re-evaluate <initialAssignment>
    # on a runtime write, so k11b keeps its base value while v12's flux takes the
    # perturbed one, the cancellation breaks, and the trajectory moves. A 2%
    # runtime write reproduces all five reported numbers to within 5% and in the
    # right proportions. That is the mechanism, not a confirmed reconstruction —
    # the exact recipe was not recovered, and nothing here depends on it.
    #
    # MODEL0910846879 is authored on BOTH regimes because the defect is in the
    # state trajectory: the sens row cannot agree while the x(t) it is taken about
    # already differs by 92%, so disposing of only `:sens` would leave the `:ode`
    # row scoring against bngsim for the same one cause.
    #
    # Surfaced by issue #382. That fix admits the model's run-constant conditions
    # and so moved it off CVODES' difference quotient, which is what put a
    # comparison in front of this row — the DIFF itself is older, and identical
    # before and after (max_rel 0.918 ode / 1.0 sens either way).
    "MODEL0910846879:ode": {
        "issue": "GH #382",
        "max_rel": 0.9184,
        "reason": (
            "AMICI evaluates one <assignmentRule> piecewise as 0 for the whole run "
            "while its own operand is positive. The oracle here is not another "
            "engine but the model's CLOSED FORM, which is stronger: the model has "
            "no reactions and one state, so it reduces exactly. Every assignment "
            "rule is over constants, giving TVZ = 9.341479411e-4, and the lone "
            "rateRule dTVD/dt = (TVZ + DR - TVD)/TVDDL then integrates to "
            "TVD(t) = TVZ + (TVD0 - TVZ)*exp(-t/30) with TVD0 = 9.80838e-4. bngsim "
            "matches that to 1.9e-9 (integration tolerance) at every output point. "
            "AMICI decays to 3.499040825e-5 at t=100, which is TVD0*exp(-100/30) to "
            "10 significant figures — the trajectory for TVZ = 0 exactly. "
            "AMICI's own expression vector says the same thing internally and is "
            "the second half of the evidence: it computes TVZ1 = 9.3414794e-4 "
            "correctly at every time point, then reports TVZ = "
            "piecewise(0, TVZ1 < 0, TVZ1) as 0 at every time point — the otherwise "
            "branch of a condition that is false throughout. The control is in the "
            "same model: AHTH = piecewise(0, AHTH1 < 0, AHTH1) is structurally "
            "identical and AMICI gets it right (9.5283044e-4), so this is not the "
            "piecewise shape as such. What separates the two is that TVZ's "
            "condition reads TVZ1, which is itself a sum over another piecewise "
            "(AHTH), while AHTH's condition reads a plain expression. Reported "
            "upstream as AMICI-dev/AMICI#3233, with a minimized reproducer: when a "
            "piecewise appears inside another piecewise's CONDITION, AMICI takes "
            "the wrong branch of the outer one. Changing the inner piecewise's "
            "first-piece value -- which cannot change the inner value, since its "
            "own condition is false either way -- flips the outer result, which is "
            "what pins the mechanism. "
            "No RoadRunner run is needed or would add anything: a closed form is "
            "not an engine's opinion, and bngsim is not adjudicating its own case "
            "because the form is derived from the SBML text by hand."
        ),
    },
    "MODEL0910846879:sens": {
        "issue": "GH #382",
        "max_rel": 1.0,
        "reason": (
            "Downstream of the ':ode' entry on this model, and disposed of for the "
            "same reason: AMICI holds one <assignmentRule> piecewise (TVZ) at 0 "
            "though its operand TVZ1 is positive throughout, so its trajectory "
            "relaxes toward 0 instead of toward TVZ. The row's own comment records "
            "'state DIFF' with state_max_rel = 0.918 — a sensitivity tensor cannot "
            "be compared when the trajectory it is differentiated about is already "
            "92% apart, so the 1.0 here carries no information about d(x)/dp. See "
            "the ':ode' entry for the closed form and for AMICI's internal "
            "inconsistency, which is the whole of the evidence."
        ),
    },
    "MODEL2105110001:sens": {
        "issue": "GH #380",
        "max_rel": 1.0,
        "reason": (
            "AMICI computes no switch-time (saltation) term, so the derivative "
            "with respect to a parameter that SETS a clock switch time is missing "
            "the jump contribution entirely. 'recruit_neu_t_switch' is a "
            "kineticLaw-local parameter inside piecewise(n1*CYT, t < t_switch, 0) "
            "— a switch at t=10 — and AMICI returns exactly 0 for it. RoadRunner's "
            "finite difference, taken away from the crossing to avoid the half-step "
            "artifact, converges to bngsim as h shrinks: at h=0.5 RR -5.2584 vs "
            "bngsim -6.5085 vs AMICI 0; at h=0.1 RR -6.71147 vs bngsim -6.78896 vs "
            "AMICI 0. On 'depletion_neu_t_switch' (same construct) RR -0.941533 and "
            "bngsim -0.947201 agree while AMICI gives -15.4723. This is SYSTEMATIC, "
            "not a one-model artifact: it recurs on every model where a parameter "
            "sets a piecewise or clock switch time — exactly the class bngsim built "
            "GH #48 / #56 / #358 / #375 for — so without this disposition it "
            "re-fills the triage queue with rows where bngsim is right. Worth "
            "reporting upstream to AMICI-dev. As always REFERENCE_FAILED records "
            "the absence of an oracle, not a bngsim confirmation: bngsim DECLINES "
            "the analytic sensitivity RHS on this model (a Functional rate law's "
            "species derivative could not be emitted as C, GH #232) and falls back "
            "to CVODES' difference quotient, so its columns come with their own "
            "loud caveat. The finite-difference agreement above is what stands."
        ),
    },
}


# --------------------------------------------------------------------------- #
# Neither engine is wrong; the metric cannot resolve the cell -> DIFF becomes
# PASS with the reason recorded. Keyed "model_id:regime".
# --------------------------------------------------------------------------- #
COMPARISON_ARTIFACT: dict[str, dict] = {
    "BIOMD0000000117:sens": {
        "issue": "GH #380",
        "max_rel": 0.1746,
        "reason": (
            "Unresolvable cell at a stimulus edge (no engine at fault). bngsim and "
            "AMICI agree to 7 significant figures on the columns in question "
            "(tstim: -40.13189 vs -40.13190; v0: 181.9811 vs 181.9811) and "
            "RoadRunner's finite difference converges to both as h shrinks (-11.04 "
            "at h=0.04 -> -37.93 at h=0.004). What survives is 1 hard cell out of "
            "22022, sitting on the stimulus discontinuity where no finite-difference "
            "oracle can resolve the answer and the two engines' step placement "
            "decides the value. Not an invalid reference — AMICI is right here — and "
            "not something a tighter tolerance fixes, since the cell is a "
            "discontinuity rather than an ill-conditioned but smooth one."
        ),
    },
}


_KIND_INVALID_REFERENCE = "invalid_reference"
_KIND_COMPARISON_ARTIFACT = "comparison_artifact"
_LABEL = {
    _KIND_INVALID_REFERENCE: "invalid reference",
    _KIND_COMPARISON_ARTIFACT: "comparison artifact",
}
_SOURCE = {
    _KIND_INVALID_REFERENCE: INVALID_REFERENCE,
    _KIND_COMPARISON_ARTIFACT: COMPARISON_ARTIFACT,
}

ALL_KEYS = frozenset(INVALID_REFERENCE) | frozenset(COMPARISON_ARTIFACT)


def disposition_for(model_id: str, regime: str) -> dict | None:
    """The authored disposition for ``(model_id, regime)``, or None.

    Returns ``{"kind", "reason", "issue", "max_rel"}``. A key present in BOTH
    dicts is an authoring contradiction — the same row cannot be both "the
    reference is wrong" and "neither engine is wrong" — so it raises rather than
    letting a dict-ordering accident pick a winner.
    """
    key = f"{model_id}:{regime}"
    found = [(kind, src[key]) for kind, src in _SOURCE.items() if key in src]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"{key} is authored in {' and '.join(k for k, _ in found)}; a row is "
            "either a defective reference or an artifact of the comparison, never both"
        )
    kind, d = found[0]
    return {
        "kind": kind,
        "reason": d["reason"],
        "issue": d.get("issue"),
        "max_rel": d.get("max_rel"),
    }


def stale_keys(model_ids: set[str], regime: str) -> list[str]:
    """Authored keys for ``regime`` whose model is NOT in ``model_ids``.

    A non-empty result means an entry names a model the corpus does not build —
    the same build-time check rr_parity's ``overrides.stale_keys`` performs. It is
    only meaningful against the FULL model set: a filtered run (``--models``)
    legitimately omits most of them, which is why the runners do not call this and
    the test suite does.
    """
    return sorted(
        k for k in ALL_KEYS if k.endswith(f":{regime}") and k.rsplit(":", 1)[0] not in model_ids
    )


def _drifted(recorded: float | None, fresh: float | None) -> bool:
    """True when the fresh metric no longer matches what the entry was written
    against, within a factor of 2 either way.

    A factor and not a percentage: these values span exactly-1.0 (a saturated
    relative metric, which does not wobble at all) to a fraction that moves with
    tolerance, and a tight band would cry wolf on the second kind. Any
    finite/non-finite crossing counts, and so does a zero appearing on one side
    only — those are changes of kind, not of degree.
    """
    if recorded is None or fresh is None:
        return False
    finite = [x == x and abs(x) != float("inf") for x in (recorded, fresh)]
    if not all(finite):
        return finite[0] != finite[1]
    if recorded == 0 or fresh == 0:
        return recorded != fresh
    ratio = abs(fresh) / abs(recorded)
    return not (0.5 <= ratio <= 2.0)


def _join(*parts: str) -> str:
    return " | ".join(p for p in parts if p)


def apply_disposition(
    disp: dict,
    outcome: Outcome,
    value: float | None,
    comment: str,
) -> tuple[Outcome, str, str | None, str]:
    """Resolve one authored disposition against a job's natural result.

    Returns ``(outcome, comment, reference_refusal, state)`` where ``state`` is
    one of ``"applied"`` / ``"stale"`` / ``"inapplicable"`` and
    ``reference_refusal`` is non-None only when the disposition itself set one
    (so a caller can leave an auto-derived refusal alone).

    ``value`` is the row's metric (``max_rel_err``); it is only read to compare
    against the entry's recorded ``max_rel``, and ``None`` simply skips that check.
    """
    kind = disp["kind"]
    label = _LABEL[kind]
    tag = f" ({disp['issue']})" if disp.get("issue") else ""

    if outcome == Outcome.DIFF:
        note = f"{label}{tag}: {disp['reason']}"
        if value is not None:
            note += f" | was DIFF at max_rel={value:.4g}"
        if _drifted(disp.get("max_rel"), value):
            note += (
                f" | DRIFTED: entry was written against max_rel={disp['max_rel']:.4g} — the "
                "disagreement changed character, so the recorded attribution is no longer "
                "known to cover it; re-triage"
            )
        if kind == _KIND_INVALID_REFERENCE:
            return (
                Outcome.REFERENCE_FAILED,
                _join(note, comment),
                INVALID_RESULT_REFUSAL,
                "applied",
            )
        return Outcome.PASS, _join(note, comment), None, "applied"

    if outcome == Outcome.PASS:
        return (
            outcome,
            _join(
                comment,
                f"STALE {label} entry{tag}: this row agrees on its own now — re-triage or prune",
            ),
            None,
            "stale",
        )

    return (
        outcome,
        _join(
            comment,
            f"{label} entry{tag} did not apply: this row is {outcome}, so no comparison "
            "happened for it to dispose of",
        ),
        None,
        "inapplicable",
    )


def dispositions_for_regime(regime: str) -> dict[str, dict]:
    """Every authored disposition for ``regime``, keyed by bare ``model_id``.

    The shape the runners want: they hold a model id per result row, not the
    composite key. Built once per run rather than per row.
    """
    out: dict[str, dict] = {}
    for key in ALL_KEYS:
        model_id, _, key_regime = key.rpartition(":")
        if key_regime == regime:
            out[model_id] = disposition_for(model_id, regime)  # type: ignore[assignment]
    return out
