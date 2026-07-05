"""Lock the per-engine failure taxonomy that `rr_parity` derives automatically.

The reference engine (RoadRunner) is the existence proof, so a job's verdict is
attributed to the engine that failed — never short-circuited on the first raise.
Two of these buckets are *non-scoring* (REFERENCE_FAILED, BAD_TEST): they carry
no signal about bngsim, so they must stay out of the pass/fail tally, and being
auto-derived, a model *leaving* one is a visible win rather than a manual edit.

These tests pin:
  * `_classify_failure` maps each reachable (bn, rr) raise combination to the
    right worker-status string (the SSA-refusal and both-ran cases are handled
    elsewhere and never reach it);
  * `_OUTCOME` translates every worker status — including the two new ones and
    the unattributable `dead` — to the intended `_core.Outcome`;
  * CLEAN / FAILING partition the whole taxonomy, so a future Outcome added to
    ALL must be deliberately classified as scoring or not (this test breaks
    until it is), and the new buckets land on the non-scoring side.
"""

from __future__ import annotations

import rr_run
from _core import ALL, CLEAN, FAILING, Outcome


# --------------------------------------------------------------------------- #
# _classify_failure — the auto-derived per-engine attribution
# --------------------------------------------------------------------------- #
def test_bngsim_raised_reference_ran_is_an_actionable_exception():
    # The reference is the existence proof: it ran, bngsim didn't -> bngsim bug.
    status, exc = rr_run._classify_failure("bngsim: ValueError: bad", "")
    assert status == "exception"
    assert exc == "bngsim: ValueError: bad"


def test_reference_raised_bngsim_ran_is_reference_failed():
    # bngsim produced a trajectory but there is no reference to compare against.
    status, exc = rr_run._classify_failure("", "roadrunner: RuntimeError: nope")
    assert status == "reference_failed"
    assert exc == "roadrunner: RuntimeError: nope"


def test_both_raised_is_bad_test_and_keeps_both_reasons():
    status, exc = rr_run._classify_failure("bngsim: A: x", "roadrunner: B: y")
    assert status == "bad_test"
    # The reference reason leads (it's why the job is non-actionable), but the
    # bngsim reason is preserved too — neither is silently dropped.
    assert "roadrunner: B: y" in exc and "bngsim: A: x" in exc


def test_classify_outputs_are_known_worker_statuses():
    # Every status _classify_failure can emit must have an _OUTCOME mapping,
    # or a real failure would fall through to the EXCEPTION default unnoticed.
    for bn, rr in (("e", ""), ("", "e"), ("e", "e")):
        status, _ = rr_run._classify_failure(bn, rr)
        assert status in rr_run._OUTCOME


# --------------------------------------------------------------------------- #
# classify_reference_refusal — the REFERENCE_FAILED sub-classification (#94)
# --------------------------------------------------------------------------- #
def test_missing_value_refusal_is_the_settled_overstrict_class():
    # Post-#94 the loader hard-rejects a *referenced* genuinely-missing param, so
    # a model bngsim still ran (REFERENCE_FAILED) where RR refused "missing a
    # value" had an unreferenced / rule-defined one — RR is over-strict, bngsim
    # correct by construction. This is the one class that needs no re-triage.
    cls = rr_run.classify_reference_refusal(
        "roadrunner: RuntimeError: Global parameter 'time_environment' is missing a value."
    )
    assert cls == "overstrict_missing_value"
    assert cls in rr_run.SETTLED_REFUSALS


def test_feature_gap_and_integrator_refusals_are_not_settled():
    # The unverified buckets: bngsim ran but there is no oracle, so they stay
    # triage-worthy (must NOT be marked settled, or a real divergence hides).
    cases = {
        "roadrunner: fast reaction not supported": "feature_gap",
        "roadrunner: delay differential equations unsupported": "feature_gap",
        "roadrunner: symbol 'max' is not physically stored": "feature_gap",
        "roadrunner: CVODE failed with too much work": "integrator",
        "roadrunner: recursive assignment rule detected": "recursive",
        "roadrunner: something we have never seen": "other",
    }
    for exc, expected in cases.items():
        cls = rr_run.classify_reference_refusal(exc)
        assert cls == expected, f"{exc!r} -> {cls!r}, expected {expected!r}"
        assert cls not in rr_run.SETTLED_REFUSALS


def test_refusal_classifier_is_total_and_lowercase_insensitive():
    # Always returns a known class (never None / crash), case-insensitively, so a
    # report row can rely on the field being a stable enum-like token.
    vocab = {"overstrict_missing_value", "feature_gap", "integrator", "recursive", "other"}
    for exc in ("", "MISSING A VALUE", "CVODE", "weird"):
        assert rr_run.classify_reference_refusal(exc) in vocab


# --------------------------------------------------------------------------- #
# _OUTCOME — worker status string -> _core.Outcome
# --------------------------------------------------------------------------- #
def test_outcome_map_attributes_each_status_to_the_right_bucket():
    assert rr_run._OUTCOME["reference_failed"] is Outcome.REFERENCE_FAILED
    assert rr_run._OUTCOME["bad_test"] is Outcome.BAD_TEST
    assert rr_run._OUTCOME["exception"] is Outcome.EXCEPTION
    # A segfaulted child leaves no per-engine status, so it can't be attributed
    # to an engine — it stays in the conservative "investigate" bucket.
    assert rr_run._OUTCOME["dead"] is Outcome.EXCEPTION


# --------------------------------------------------------------------------- #
# Taxonomy invariants (guard future Outcome additions)
# --------------------------------------------------------------------------- #
def test_reference_failed_and_bad_test_are_non_scoring():
    # The whole point: a reference-side failure must not count against bngsim.
    for o in (Outcome.REFERENCE_FAILED, Outcome.BAD_TEST):
        assert o in CLEAN
        assert o not in FAILING


def test_clean_and_failing_partition_the_taxonomy():
    # Disjoint and exhaustive over ALL: a new Outcome added to ALL but not
    # classified scoring/non-scoring trips this, forcing a deliberate choice.
    assert CLEAN.isdisjoint(FAILING)
    assert set(ALL) == CLEAN | FAILING


# --------------------------------------------------------------------------- #
# _apply_invalid_reference — the narrow "RR ran but non-finite -> BAD_TEST"
# override, and the staleness contract that keeps it from masking a real bug.
# --------------------------------------------------------------------------- #
def test_invalid_reference_reclassifies_nonfinite_exception_to_bad_test():
    # bngsim raised, RR ran but emitted NaN (rr_finite False): both broken.
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.EXCEPTION, False, "RR all-NaN here", None, ""
    )
    assert outcome is Outcome.BAD_TEST and stale is False
    assert "invalid reference" in comment and "RR all-NaN here" in comment
    assert "was EXCEPTION" in comment  # the natural verdict is preserved in-line


def test_invalid_reference_confirms_a_natural_bad_test():
    # Both engines raised (RR raised -> rr_finite None): already BAD_TEST, the
    # override just documents why, with no misleading "was ..." suffix.
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.BAD_TEST, None, "both fail to integrate", None, "rr || bn"
    )
    assert outcome is Outcome.BAD_TEST and stale is False
    assert "was BAD_TEST" not in comment


def test_invalid_reference_goes_stale_when_reference_becomes_finite():
    # THE bug-masking guard: RR now returns a finite trajectory but bngsim still
    # raises -> that is a real EXCEPTION (actionable bngsim bug). The override
    # must NOT force BAD_TEST; it keeps EXCEPTION and flags itself stale.
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.EXCEPTION, True, "RR all-NaN here", None, ""
    )
    assert outcome is Outcome.EXCEPTION and stale is True
    assert "STALE invalid-reference" in comment


def test_invalid_reference_goes_stale_when_bngsim_runs_again():
    # bngsim now produces a trajectory (natural REFERENCE_FAILED / PASS): the
    # "both broken" premise is gone, so keep the natural outcome and flag stale.
    for natural in (Outcome.REFERENCE_FAILED, Outcome.PASS, Outcome.DIFF):
        outcome, comment, stale = rr_run._apply_invalid_reference(natural, True, "r", None, "")
        assert outcome is natural and stale is True
        assert "STALE invalid-reference" in comment


def test_invalid_reference_does_not_fabricate_bad_test_from_a_dead_worker():
    # A segfaulted child is EXCEPTION with no rr_finite recorded (None). We can't
    # confirm the premise, so the override must not force BAD_TEST — flag stale.
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.EXCEPTION, None, "r", None, ""
    )
    assert outcome is Outcome.EXCEPTION and stale is True
