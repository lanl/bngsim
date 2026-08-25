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
    until it is), and the new buckets land on the non-scoring side;
  * `_apply_invalid_reference` — the one authored way into BAD_TEST — holds only
    while the reference is genuinely unusable, and `_reference_nonfinite_covers`,
    the per-cell guard the relaxed #485 premise rests on: a DIFF is reclassified
    only when every surviving failure lies in a column the reference left
    non-finite, so a real bngsim divergence beside the NaN cannot be buried.
"""

from __future__ import annotations

import numpy as np
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


def test_invalid_reference_goes_stale_when_reference_is_finite_whatever_bngsim_did():
    # A usable reference breaks the premise outright — the entry is stale on
    # every natural outcome, not only the ones where bngsim failed.
    for natural in (Outcome.REFERENCE_FAILED, Outcome.PASS, Outcome.DIFF):
        outcome, comment, stale = rr_run._apply_invalid_reference(natural, True, "r", None, "")
        assert outcome is natural and stale is True
        assert "STALE invalid-reference" in comment


def test_invalid_reference_reclassifies_a_diff_the_reference_nan_accounts_for():
    # #485: bngsim integrates the model now, RR still answers NaN, so the natural
    # verdict is a DIFF scoring RR's NaN against bngsim. There is no comparison
    # basis here whatever bngsim did -> BAD_TEST, with the DIFF kept in-line.
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.DIFF, False, "RR all-NaN here", None, "fail 606/707", rr_covers_diff=True
    )
    assert outcome is Outcome.BAD_TEST and stale is False
    assert "invalid reference" in comment and "RR all-NaN here" in comment
    assert "was DIFF: fail 606/707" in comment


def test_invalid_reference_keeps_a_diff_the_reference_nan_does_not_account_for():
    # THE bug-masking guard for the relaxed premise: RR is non-finite somewhere,
    # but a failing cell survives in a column it kept finite — that is a real
    # bngsim divergence beside the NaN, and it must stay a DIFF.
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.DIFF, False, "r", None, "", rr_covers_diff=False
    )
    assert outcome is Outcome.DIFF and stale is True
    assert "STALE invalid-reference" in comment and "rr_covers_diff=False" in comment


def test_invalid_reference_does_not_swallow_a_pass():
    # RR emitted a non-finite cell somewhere, but the compared columns agreed --
    # the reference WAS usable for the comparison, so the pass stands.
    outcome, comment, stale = rr_run._apply_invalid_reference(Outcome.PASS, False, "r", None, "")
    assert outcome is Outcome.PASS and stale is True
    assert "STALE invalid-reference" in comment


def test_invalid_reference_does_not_fabricate_bad_test_from_a_dead_worker():
    # A segfaulted child is EXCEPTION with no rr_finite recorded (None). We can't
    # confirm the premise, so the override must not force BAD_TEST — flag stale.
    # (The relaxed #485 premise keys on rr_finite is *False*, never on None, for
    # exactly this reason: "RR raised" and "we never learned" look alike here.)
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.EXCEPTION, None, "r", None, ""
    )
    assert outcome is Outcome.EXCEPTION and stale is True
    assert "STALE invalid-reference" in comment


def test_invalid_reference_does_not_reclassify_a_reference_failed_row():
    # RR raised while bngsim ran: that is REFERENCE_FAILED (no oracle), already
    # non-scoring and already the right bucket. The override adds nothing and
    # must not move it to BAD_TEST, which would read as "bngsim failed too".
    outcome, comment, stale = rr_run._apply_invalid_reference(
        Outcome.REFERENCE_FAILED, None, "r", None, ""
    )
    assert outcome is Outcome.REFERENCE_FAILED and stale is True


# --------------------------------------------------------------------------- #
# _reference_nonfinite_covers — the per-cell guard the relaxed premise rests on
# --------------------------------------------------------------------------- #
def test_reference_nonfinite_covers_when_every_bad_column_is_the_references():
    # Two columns; the reference is NaN in the first and agrees in the second ->
    # the whole divergence is the reference's, nothing of bngsim's hides in it.
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[np.nan, 2.0], [np.nan, 4.0]])
    assert rr_run._reference_nonfinite_covers(a, b) is True


def test_reference_nonfinite_does_not_cover_a_divergence_beside_it():
    # Same NaN column, but the second column now diverges by 10x — a real
    # divergence the reference's NaN cannot account for.
    a = np.array([[1.0, 2.0], [3.0, 40.0]])
    b = np.array([[np.nan, 2.0], [np.nan, 4.0]])
    assert rr_run._reference_nonfinite_covers(a, b) is False


def test_reference_nonfinite_covers_an_all_nonfinite_reference():
    # No finite reference column remains to disagree in.
    a = np.array([[1.0], [3.0]])
    b = np.array([[np.nan], [np.nan]])
    assert rr_run._reference_nonfinite_covers(a, b) is True


def test_reference_nonfinite_covers_nothing_when_the_reference_is_finite():
    # A finite reference has no non-finite output to attribute anything to, so
    # it can never cover a divergence — however large that divergence is.
    a = np.array([[1.0], [30.0]])
    b = np.array([[1.0], [3.0]])
    assert rr_run._reference_nonfinite_covers(a, b) is False


def test_reference_nonfinite_covers_ignores_a_column_nan_on_both_engines():
    # Both engines NaN in a column is a zero-diff pass for the differ, and the
    # column is dropped from the finite-only re-run either way; a divergence in
    # the remaining column still keeps the entry honest.
    a = np.array([[np.nan, 1.0], [np.nan, 30.0]])
    b = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    assert rr_run._reference_nonfinite_covers(a, b) is False


# --------------------------------------------------------------------------- #
# _compare_ode — the flag reaches the override through the worker's result dict
# --------------------------------------------------------------------------- #
def _runs(bn_v, rr_v, names=("x1", "x7")):
    """Two aligned runs on a shared 4-point grid, as _compare_ode takes them."""
    t = np.linspace(0.0, 1.0, 4)
    return (t, np.asarray(bn_v), list(names)), (t, np.asarray(rr_v), list(names))


def test_compare_ode_reports_a_diff_the_reference_nan_covers():
    # MODEL2002070001's shape in miniature: the reference is NaN in the species
    # column and agrees on the one variable it kept finite (#485).
    bn, rr = _runs(
        [[1.0, 2.0], [2.0, 2.0], [4.0, 2.0], [8.0, 2.0]],
        [[np.nan, 2.0], [np.nan, 2.0], [np.nan, 2.0], [np.nan, 2.0]],
    )
    status, value, _comment, _metric, _tol, covers = rr_run._compare_ode(bn, rr)
    assert status == "diff" and value == float("inf")
    assert covers is True


def test_compare_ode_does_not_flag_a_diff_the_reference_nan_leaves_unexplained():
    # Same NaN column, but the finite column diverges too — the reference's NaN
    # does not account for that, so the override must not get its licence.
    bn, rr = _runs(
        [[1.0, 2.0], [2.0, 2.0], [4.0, 2.0], [8.0, 90.0]],
        [[np.nan, 2.0], [np.nan, 2.0], [np.nan, 2.0], [np.nan, 2.0]],
    )
    status, _value, _comment, _metric, _tol, covers = rr_run._compare_ode(bn, rr)
    assert status == "diff"
    assert covers is False


def test_compare_ode_flags_nothing_on_a_pass():
    bn, rr = _runs(
        [[1.0, 2.0], [2.0, 2.0], [4.0, 2.0], [8.0, 2.0]],
        [[1.0, 2.0], [2.0, 2.0], [4.0, 2.0], [8.0, 2.0]],
    )
    status, _value, _comment, _metric, _tol, covers = rr_run._compare_ode(bn, rr)
    assert status == "pass"
    assert covers is False


def test_compare_ode_flags_nothing_on_a_disjoint_species_set():
    # A pre-comparison diff: no reference NaN explains a species set that never
    # lined up, so the override cannot claim it.
    bn, rr = _runs(
        [[1.0, 2.0], [2.0, 2.0], [4.0, 2.0], [8.0, 2.0]],
        [[1.0, 2.0], [2.0, 2.0], [4.0, 2.0], [8.0, 2.0]],
    )
    rr = (rr[0], rr[1], ["y1", "y7"])
    status, _value, comment, _metric, _tol, covers = rr_run._compare_ode(bn, rr)
    assert status == "diff" and "disjoint" in comment
    assert covers is False
