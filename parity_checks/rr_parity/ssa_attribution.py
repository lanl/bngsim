"""Map the SSA screen's native attribution onto the shared ``_core`` taxonomy.

``ssa_screen.py`` carries a richer status vocabulary than the cross-suite
``_core.Outcome`` set, because it attributes every surviving cross-engine
divergence against a third oracle (each engine's own ODE — GH #107 follow-up):

    pass            both engines agree within the z-gate
    fail            bngsim SSA deviates from its own ODE, or unexplained residual
                    — the only status that means "look at bngsim"
    diff_ode_level  the two ODEs themselves disagree (loader/ODE-level)
    diff_not_bngsim oracle-attributed RR-gillespie-side divergence
    rr_known        a specifically-filed RoadRunner issue (KNOWN_RR_ISSUES)
    too_slow        killed at the per-model wall cap (exact-SSA-intractable)
    error           an engine raised / failed to load
    skip            the SBML file was missing

This module is the single bridge that turns those into a ``_core`` report so the
SSA screen becomes a tracked regression sharing the suite's schema (GH #108).
The mapping keeps the **honest** outcome — an oracle-attributed RR-side
divergence is still a real ``DIFF`` — and records WHY in ``JobResult.subclass``,
rather than burying the attributed cases in ``PASS`` (the screen's ethos is
"never silently green; coverage explicit"). A suite *scoring policy*
(:func:`is_expected`) then decides which ``(outcome, subclass)`` pairs are
acceptable, so the regression diff (GH #108 step 4) can flag only the genuine
PASS↔FAIL moves while treating ``rr_known`` / ``too_slow`` / ``diff_not_bngsim``
as expected classes.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:  # so `from _core import ...` resolves standalone
    sys.path.insert(0, str(HERE.parent))
from _core import JobResult, Outcome, tally  # noqa: E402

METHOD = "ssa"
REFERENCE_ENGINE = "roadrunner"

# Screen status -> honest _core Outcome. ``diff_not_bngsim`` / ``rr_known`` stay
# DIFF (a real cross-engine divergence exists) and are greened by the scoring
# policy via their subclass, NOT collapsed into PASS.
STATUS_TO_OUTCOME: dict[str, Outcome] = {
    "pass": Outcome.PASS,
    "fail": Outcome.DIFF,
    "diff_ode_level": Outcome.DIFF,
    "diff_not_bngsim": Outcome.DIFF,
    "rr_known": Outcome.DIFF,
    # Provisional (yellow) adjudications of a model that cannot be fully judged:
    # ``partial`` — only a [0,H<t_end] window was simulable; ``attribution_incomplete``
    # — the full SSA verdict was a fail but the deterministic-ODE oracle that would
    # assign fault did not finish. Both are real cross-engine comparisons we could
    # not turn into a full bngsim/RR verdict, so they ride the DIFF outcome and are
    # greened (triaged) by their subclass — never a red bngsim defect.
    # ``rr_time_event`` — the model has time-triggered events, which RoadRunner's
    # gillespie silently does not fire (it warns and freezes at the IC), so RR
    # cannot anchor it. bngsim was validated against its OWN ODE instead; this is a
    # documented RR capability gap, greened (yellow) via its subclass, never scored
    # against bngsim. (A bngsim SSA that DEVIATES from its own ODE returns "fail"
    # instead → a real bngsim defect, red.)
    "rr_time_event": Outcome.DIFF,
    # ``rr_time_event_unverified`` — same RR capability gap, but bngsim could not be
    # positively validated against its own ODE here: either the SSA mean diverges
    # from the ODE (the ODE is not the SSA mean for nonlinear / heavy-tailed models,
    # so this is NOT a bngsim fault — flagging it would false-positive on rare
    # explosive seeds) or bngsim was quiet at every tried horizon. Greened (yellow).
    "rr_time_event_unverified": Outcome.DIFF,
    "partial": Outcome.DIFF,
    # ``extended`` — the requested horizon was too quiet to compare (every cell
    # below the low-count floor), so the comparison was run over a LARGER horizon
    # where a slow/delayed-onset model accumulates above-floor signal (symmetric
    # to ``partial``); greened (provisional) via its subclass.
    "extended": Outcome.DIFF,
    "attribution_incomplete": Outcome.DIFF,
    "too_slow": Outcome.TIMEOUT,
    "error": Outcome.EXCEPTION,
    "skip": Outcome.SKIP,
}

# Screen status -> machine-readable DIFF subclass. ``pass`` / ``too_slow`` /
# ``error`` / ``skip`` carry none (their Outcome already says it all).
STATUS_TO_SUBCLASS: dict[str, str] = {
    "fail": "bngsim_suspect",
    "diff_ode_level": "ode_level",
    "diff_not_bngsim": "diff_not_bngsim",
    "rr_known": "rr_known",
    "rr_time_event": "rr_time_event",
    "rr_time_event_unverified": "rr_time_event_unverified",
    "partial": "partial_horizon",
    "extended": "extended_horizon",
    "attribution_incomplete": "attribution_incomplete",
}

# Subclasses that make a DIFF EXPECTED (attributed away from bngsim by the oracle /
# a filed RR issue, or a provisional partial-window / unattributed-fault verdict
# we deliberately do not score against bngsim). Their counterpart — a real, scoring
# DIFF — is tagged ``bngsim_suspect`` / ``ode_level``.
GREEN_DIFF_SUBCLASSES = frozenset(
    {
        "diff_not_bngsim",
        "rr_known",
        "rr_time_event",
        "rr_time_event_unverified",
        "partial_horizon",
        "extended_horizon",
        "attribution_incomplete",
    }
)
RED_DIFF_SUBCLASSES = frozenset({"bngsim_suspect", "ode_level"})

# Outcomes that are expected regardless of subclass for THIS suite: the generic
# _core CLEAN set (PASS / UNSUPPORTED / SKIP / REFERENCE_FAILED / BAD_TEST — none
# of which reflect on bngsim) plus ``TIMEOUT``. TIMEOUT is expected here (a
# ``too_slow`` model is a known exact-SSA coverage gap, not a regression) even
# though ``_core`` lists it among the generic FAILING set — the suite policy
# deliberately overrides the cross-suite default for coverage gaps.
_EXPECTED_OUTCOMES = frozenset(
    {
        Outcome.PASS,
        Outcome.UNSUPPORTED,
        Outcome.SKIP,
        Outcome.REFERENCE_FAILED,
        Outcome.BAD_TEST,
        Outcome.TIMEOUT,
    }
)


def is_expected(outcome, subclass: str | None) -> bool:
    """True if ``(outcome, subclass)`` is a non-regression state for the SSA screen.

    Expected: a clean PASS, an UNSUPPORTED refusal, a SKIP (missing model), a
    REFERENCE_FAILED / BAD_TEST (the *reference*, not bngsim, couldn't anchor a
    comparison), a TIMEOUT (``too_slow`` coverage gap), or a DIFF the oracle
    attributed away from bngsim (``diff_not_bngsim`` / ``rr_known``). The only
    NOT-expected states are a ``bngsim_suspect`` / ``ode_level`` DIFF and a
    bngsim-side EXCEPTION — i.e. exactly the screen's own red exit-code set.
    """
    outcome = Outcome(outcome) if not isinstance(outcome, Outcome) else outcome
    if outcome in _EXPECTED_OUTCOMES:
        return True
    if outcome == Outcome.DIFF:
        return subclass in GREEN_DIFF_SUBCLASSES
    return False


def is_regression(baseline_outcome, baseline_subclass, fresh_outcome, fresh_subclass) -> bool:
    """A model regresses iff it was expected in the baseline and is not now —
    i.e. it crossed from a green/expected state into a scoring DIFF/EXCEPTION."""
    return is_expected(baseline_outcome, baseline_subclass) and not is_expected(
        fresh_outcome, fresh_subclass
    )


# --------------------------------------------------------------------------- #
# Case -> JobResult
# --------------------------------------------------------------------------- #
def _wall_sec(case: dict):
    """Best-effort wall time: compared = bn+rr engine time; too_slow = wall cap."""
    if case.get("elapsed_sec_wall") is not None:
        return float(case["elapsed_sec_wall"])
    bn = case.get("elapsed_sec_bn")
    rr = case.get("elapsed_sec_rr")
    if bn is None and rr is None:
        return None
    return round(float(bn or 0.0) + float(rr or 0.0), 3)


def _comment(case: dict, status: str) -> str:
    """Human-readable attribution / coverage note for a case (may be empty)."""
    parts: list[str] = []
    if status == "rr_known":
        info = case.get("known_rr_issue") or {}
        issue = info.get("issue", "")
        note = info.get("note", "")
        parts.append(f"known RR issue {issue}: {note}".strip().rstrip(":"))
    elif status in ("diff_not_bngsim", "diff_ode_level", "fail"):
        reason = (case.get("diff_attribution") or {}).get("reason", "")
        if reason:
            parts.append(reason)
    elif status == "partial":
        h = case.get("partial_horizon")
        agreed = "agreed" if case.get("partial_pass") else "diverged"
        parts.append(
            f"partial-horizon adjudication over t=0→{h:g}: ensembles {agreed} "
            "(exact SSA intractable beyond this window — no full-horizon verdict)"
            if isinstance(h, (int, float))
            else "partial-horizon adjudication (no full-horizon verdict)"
        )
    elif status == "extended":
        h = case.get("extended_horizon")
        agreed = "agreed" if case.get("extended_pass") else "diverged"
        parts.append(
            f"extended-horizon adjudication over t=0→{h:g}: ensembles {agreed} "
            "(requested horizon too quiet — every cell below the low-count floor; "
            "grew the window until species cleared it)"
            if isinstance(h, (int, float))
            else "extended-horizon adjudication (requested horizon too quiet)"
        )
    elif status in ("rr_time_event", "rr_time_event_unverified"):
        h = case.get("extended_horizon")
        n_cmp = int(case.get("n_compared", 0) or 0)
        if status == "rr_time_event":
            hs = f" over t=0→{h:g}" if isinstance(h, (int, float)) else ""
            body = (
                f"bngsim SSA tracks its own ODE on all {n_cmp} above-floor cells{hs} (validated)"
            )
        elif n_cmp == 0:
            body = "bngsim was quiet at every tried horizon, so it could not be validated either"
        else:
            body = (
                "bngsim SSA mean diverges from its own ODE, but the ODE is not the SSA mean for "
                "nonlinear / heavy-tailed models (e.g. rare explosive seeds) — so bngsim is "
                "neither confirmed nor faulted here"
            )
        parts.append(
            "RoadRunner's gillespie does not fire time-triggered events (RR's own warning: "
            "'time is not treated continuously in a gillespie simulation') — RR cannot anchor "
            f"this model and is not used as the reference; {body}"
        )
    elif status == "attribution_incomplete":
        parts.append(
            "SSA ensembles diverged, but the deterministic-ODE oracle that assigns fault "
            "did not finish — provisional; fault undetermined (not a confirmed bngsim defect)"
        )
    if (
        status == "pass"
        and int(case.get("n_compared", 0) or 0) == 0
        and int(case.get("n_skipped_lowcount", 0) or 0) > 0
    ):
        parts.append("vacuous pass: every (t>0, species) cell fell below the low-count floor")
    sign = case.get("sign_indefinite")
    if sign:
        parts.append(
            f"sign-indefinite (GH #109): bngsim reverse-fired {sign.get('reaction', '?')} "
            f"×{sign.get('reverse_fires', '?')}"
        )
    return " | ".join(p for p in parts if p)


def _classify_error(msg: str) -> tuple[Outcome, str | None]:
    """Split a screen ``error`` by which engine raised — the screen runs bngsim
    first, so a ``roadrunner:``-prefixed message means bngsim ran fine and only
    RoadRunner refused. That is REFERENCE_FAILED, not a bngsim defect (bngsim
    succeeding where the reference can't is a capability win); it is tagged
    ``feature_gap`` because every such refusal here is RR's gillespie/llvm
    rejecting a feature the bngsim-validated SSA corpus accepts (rate rules).

    A bngsim *deliberate* refusal of a construct it cannot faithfully simulate
    under ODE (GH #113: delay DDEs, AlgebraicRules, fast reactions — now refused
    at load rather than silently approximated) is BAD_TEST, not a scoring
    EXCEPTION: RR's gillespie cannot run these either, so it is a feature BOTH
    engines lack (no parity signal, not a bngsim defect). This mirrors the ODE
    runner's "both raised -> BAD_TEST" auto-derivation; before #113 landed these
    loaded silently in bngsim and so read as a plain REFERENCE_FAILED here.

    A bngsim loader fail-close on an SBML that **references an undefined symbol**
    (GH #119/#147: a symbol binding to no declared component in an initialAssignment
    or rule) is likewise BAD_TEST: the SBML is malformed, and RoadRunner rejects the
    same models — verified on MODEL2205030001, where RR raises ``Could not find
    requested symbol 'Kd_pAKT_basal'``. Both engines correctly refuse, so it is a
    bad test (no parity signal), never a bngsim defect; this matches the ODE runner,
    which already classifies MODEL2205030001 as BAD_TEST.

    Any other ``bngsim:``-prefixed or structural error (disjoint species /
    time-grid mismatch) is a real, scoring EXCEPTION.
    """
    low = (msg or "").lower()
    if low.startswith("roadrunner:"):
        return Outcome.REFERENCE_FAILED, "feature_gap"
    # GH #113 refusal signature (see _sbml_loader._check_unsupported_constructs).
    if "constructs bngsim cannot faithfully simulate" in low:
        return Outcome.BAD_TEST, None
    # GH #119/#147 undefined-symbol fail-close (see _sbml_loader._check_undefined_symbols)
    # — malformed SBML that RoadRunner rejects too.
    if "references symbols that bind to no declared component" in low:
        return Outcome.BAD_TEST, None
    return Outcome.EXCEPTION, None


def case_to_job_result(
    case: dict,
    *,
    mean_z_tol: float = 5.0,
    versions: dict | None = None,
    timestamp: str = "",
) -> JobResult:
    """Translate one screen ``cases[...]`` entry into a ``_core`` JobResult.

    The z-gate metric (``mean_zscore`` = worst per-cell mean z-score) is recorded
    only for compared cases; ``too_slow`` / ``error`` / ``skip`` carry no metric.
    An ``error`` is split by engine: a RoadRunner-side refusal becomes
    REFERENCE_FAILED (see :func:`_classify_error`), a bngsim-side one EXCEPTION.
    """
    status = case["status"]
    outcome = STATUS_TO_OUTCOME.get(status, Outcome.EXCEPTION)
    subclass = STATUS_TO_SUBCLASS.get(status)
    # GH #190 — a "pass" that compared NOTHING (every t>0 cell fell below the
    # low-count floor: the model fires ~no reactions over this horizon) is a
    # COVERAGE GAP, not a real agreement. Tag it so the matrix renders it gray —
    # the mirror of a too_slow TIMEOUT (too active to compare) is a model too quiet
    # to compare. Outcome stays PASS (it did not fail, and PASS is always an
    # expected/non-regression state), so this never flags a false regression.
    if (
        status == "pass"
        and int(case.get("n_compared", 0) or 0) == 0
        and int(case.get("n_skipped_lowcount", 0) or 0) > 0
    ):
        subclass = "vacuous_lowcount"
    reference_refusal = None
    if status == "error":
        outcome, reference_refusal = _classify_error(str(case.get("error") or ""))
    compared = "max_mean_z" in case
    return JobResult(
        model_id=case["name"],
        method=METHOD,
        reference_engine=REFERENCE_ENGINE,
        outcome=outcome.value,
        metric="mean_zscore" if compared else None,
        value=float(case["max_mean_z"]) if compared else None,
        tol=float(mean_z_tol) if compared else None,
        exception=str(case.get("error") or ""),
        wall_sec=_wall_sec(case),
        timestamp=timestamp,
        versions=dict(versions or {}),
        comment=_comment(case, status),
        reference_refusal=reference_refusal,
        subclass=subclass,
        # Per-engine SSA timing (#135), measurement-only — passed straight through
        # from the screen case ({"warmup", "bngsim", "roadrunner"}); None on a
        # legacy/timeout/skip case that recorded none. The SSA timing taxonomy
        # differs from ODE (no Jacobian/codegen; per-replicate ensemble integration
        # instead of a cold/warm-reuse split) — the renderer keys off the shape.
        timing=case.get("timing"),
    )


def _versions_of(machine: dict) -> dict:
    """Pull a versions dict from a screen payload's machine_info, tolerating both
    the new ``versions`` block (``_core.versions.stamp``) and the older flat
    ``bngsim_version`` / ``run_network_version`` layout of the legacy baseline."""
    v = machine.get("versions")
    if isinstance(v, dict) and v:
        return dict(v)
    out = {}
    if machine.get("bngsim_version"):
        out["bngsim"] = machine["bngsim_version"]
    if machine.get("run_network_version"):
        out["run_network"] = machine["run_network_version"]
    return out


def payload_to_report(payload: dict) -> tuple[dict, list[JobResult]]:
    """Convert a full screen JSON payload into ``(meta, [JobResult, ...])``.

    Pure (no engines), so it round-trips either a fresh ``runs/ssa_screen.json`` or
    the legacy committed baseline. ``meta`` echoes the screen config + version
    stamp and adds an outcome tally and a subclass breakdown for quick scanning.
    """
    config = payload.get("config", {}) or {}
    # The corpus dir is environment-specific (ssa_spec.json pins the run knobs but NOT
    # biomodels_sbml_dir / jobs); redact the absolute path so the committed baseline does
    # not bake in a developer home dir. $BIOMODELS_SBML_DIR is how a fresh run resolves it.
    # We also strip it from any bngsim load-fail exception (which embeds the absolute SBML
    # path), keeping just the filename for diagnostics — see the results loop below.
    corpus_dir = config.get("biomodels_sbml_dir")
    if corpus_dir:
        config = {**config, "biomodels_sbml_dir": "$BIOMODELS_SBML_DIR"}
    machine = payload.get("machine_info", {}) or {}
    mean_z_tol = float(config.get("mean_z_tol", 5.0))
    versions = _versions_of(machine)
    timestamp = machine.get("date", "")
    cases = payload.get("cases", {}) or {}

    results = sorted(
        (
            case_to_job_result(c, mean_z_tol=mean_z_tol, versions=versions, timestamp=timestamp)
            for c in cases.values()
        ),
        key=lambda r: r.model_id,
    )

    # Redact the corpus dir from bngsim load-fail exceptions too (they embed the absolute
    # SBML path, e.g. "Failed to load SBML file <corpus_dir>/MODEL....xml"); the filename
    # stays, so the diagnostic survives without the developer home dir.
    if corpus_dir:
        for r in results:
            if r.exception and corpus_dir in r.exception:
                r.exception = r.exception.replace(corpus_dir, "$BIOMODELS_SBML_DIR")

    subclass_breakdown: dict[str, int] = {}
    for r in results:
        if r.subclass:
            subclass_breakdown[r.subclass] = subclass_breakdown.get(r.subclass, 0) + 1
    n_regression_eligible = sum(1 for r in results if not is_expected(r.outcome, r.subclass))

    meta = {
        "description": "bngsim-vs-RoadRunner SSA cross-engine screen — _core report (GH #108).",
        "reference_engine": REFERENCE_ENGINE,
        "method": METHOD,
        "config": config,
        "versions": versions,
        # CPU / OS / concurrency context so the #135 timing numbers are
        # interpretable (timings were collected under up-to-``config.jobs``-way CPU
        # contention — the renderer adds that caveat). Carried through from the
        # screen's machine_info; absent on a legacy payload that recorded none.
        "machine": {
            k: machine.get(k) for k in ("platform", "processor", "cpu_count") if machine.get(k)
        },
        "outcome_tally": tally(r.outcome for r in results),
        "subclass_breakdown": subclass_breakdown,
        "n_not_expected": n_regression_eligible,
    }
    return meta, results
