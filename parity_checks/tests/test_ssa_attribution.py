"""Hermetic tests for ssa_attribution — the SSA-screen → _core taxonomy bridge.

The screen carries a richer status vocabulary than _core.Outcome; this module
maps each status onto an honest Outcome + a machine-readable subclass, and a
suite scoring policy (is_expected) decides which (outcome, subclass) pairs are
non-regression states. These tests pin the two contracts that the GH #108
regression diff depends on:

  * the mapping is total and honest (attributed RR-side divergences stay DIFF,
    never silently collapse into PASS); and
  * the green/red policy greens exactly {clean pass, unsupported, skip, too_slow
    coverage gap, oracle-attributed-not-bngsim DIFF} and reds exactly
    {bngsim_suspect / ode_level DIFF, engine EXCEPTION} — i.e. the screen's own
    red exit-code set.

No SBML, no engines — pure dict→dataclass translation, so it runs anywhere.
"""

from __future__ import annotations

import pytest
import ssa_attribution as sa
from _core import Outcome


# --- status -> (outcome, subclass) map ------------------------------------
@pytest.mark.parametrize(
    "status, outcome, subclass",
    [
        ("pass", Outcome.PASS, None),
        ("fail", Outcome.DIFF, "bngsim_suspect"),
        ("diff_ode_level", Outcome.DIFF, "ode_level"),
        ("diff_not_bngsim", Outcome.DIFF, "diff_not_bngsim"),
        ("rr_known", Outcome.DIFF, "rr_known"),
        ("partial", Outcome.DIFF, "partial_horizon"),
        ("extended", Outcome.DIFF, "extended_horizon"),
        ("rr_time_event", Outcome.DIFF, "rr_time_event"),
        ("rr_time_event_unverified", Outcome.DIFF, "rr_time_event_unverified"),
        ("attribution_incomplete", Outcome.DIFF, "attribution_incomplete"),
        ("too_slow", Outcome.TIMEOUT, None),
        ("error", Outcome.EXCEPTION, None),
        ("skip", Outcome.SKIP, None),
    ],
)
def test_status_mapping_is_honest_and_total(status, outcome, subclass):
    assert sa.STATUS_TO_OUTCOME[status] == outcome
    assert sa.STATUS_TO_SUBCLASS.get(status) == subclass


@pytest.mark.parametrize(
    "subclass",
    [
        "partial_horizon",
        "extended_horizon",
        "rr_time_event",
        "rr_time_event_unverified",
        "attribution_incomplete",
    ],
)
def test_provisional_adjudications_are_yellow_not_red(subclass):
    # A partial-horizon verdict or an unattributed (oracle-timed-out) fail is a real
    # cross-engine comparison we could not turn into a full bngsim/RR verdict — it is
    # triaged (yellow / expected), never a red bngsim defect, so it never regresses.
    assert sa.is_expected(Outcome.DIFF, subclass) is True
    assert subclass in sa.GREEN_DIFF_SUBCLASSES


def test_partial_comment_names_the_window():
    case = {
        "name": "M",
        "status": "partial",
        "source": "biomodels",
        "max_mean_z": 1.8,
        "partial_horizon": 10.0,
        "partial_pass": True,
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "DIFF" and r.subclass == "partial_horizon"
    assert "t=0→10" in r.comment and "agreed" in r.comment
    assert r.value == pytest.approx(1.8)  # the z-metric over the partial window


def test_attributed_diffs_are_not_collapsed_into_pass():
    # The honesty guarantee: an oracle-attributed RR-side divergence and a filed
    # RR issue are still DIFF (a real divergence exists), greened only by policy.
    assert sa.STATUS_TO_OUTCOME["diff_not_bngsim"] == Outcome.DIFF
    assert sa.STATUS_TO_OUTCOME["rr_known"] == Outcome.DIFF


# --- scoring policy: is_expected ------------------------------------------
@pytest.mark.parametrize(
    "outcome, subclass, expected",
    [
        (Outcome.PASS, None, True),
        (Outcome.UNSUPPORTED, None, True),
        (Outcome.SKIP, None, True),
        (Outcome.REFERENCE_FAILED, "feature_gap", True),  # RR refused; bngsim ran
        (Outcome.BAD_TEST, None, True),
        (Outcome.TIMEOUT, None, True),  # too_slow coverage gap — expected here
        (Outcome.DIFF, "diff_not_bngsim", True),
        (Outcome.DIFF, "rr_known", True),
        (Outcome.DIFF, "bngsim_suspect", False),
        (Outcome.DIFF, "ode_level", False),
        (Outcome.DIFF, None, False),  # a bare DIFF with no attribution stays red
        (Outcome.EXCEPTION, None, False),
    ],
)
def test_is_expected_policy(outcome, subclass, expected):
    assert sa.is_expected(outcome, subclass) is expected


def test_is_expected_accepts_string_outcomes():
    # The diff reads outcomes back from JSON as plain strings.
    assert sa.is_expected("PASS", None) is True
    assert sa.is_expected("DIFF", "rr_known") is True
    assert sa.is_expected("DIFF", "bngsim_suspect") is False


def test_timeout_policy_overrides_core_failing_default():
    # TIMEOUT is in _core.FAILING generically, but the SSA suite treats a
    # too_slow coverage gap as expected — the override this suite documents.
    from _core import FAILING

    assert Outcome.TIMEOUT in FAILING
    assert sa.is_expected(Outcome.TIMEOUT, None) is True


# --- regression detector --------------------------------------------------
def test_regression_only_when_crossing_from_expected_to_red():
    # newly broken: was a clean pass, now bngsim-suspect -> regression.
    assert sa.is_regression("PASS", None, "DIFF", "bngsim_suspect") is True
    # green-to-green attribution churn (pass -> RR-side DIFF) is NOT a regression.
    assert sa.is_regression("PASS", None, "DIFF", "diff_not_bngsim") is False
    # already broken in the baseline -> not a NEW regression.
    assert sa.is_regression("DIFF", "bngsim_suspect", "EXCEPTION", None) is False
    # improvement (red -> green) is not a regression.
    assert sa.is_regression("DIFF", "bngsim_suspect", "PASS", None) is False


# --- case_to_job_result ---------------------------------------------------
def _compared(name, status, max_z, **extra):
    base = {
        "name": name,
        "status": status,
        "source": "biomodels",
        "max_mean_z": max_z,
        "n_compared": 5,
        "n_skipped_lowcount": 0,
        "elapsed_sec_bn": 0.4,
        "elapsed_sec_rr": 0.6,
    }
    base.update(extra)
    return base


def test_compared_pass_carries_zscore_metric():
    r = sa.case_to_job_result(_compared("BIOMD1", "pass", 0.9), mean_z_tol=5.0)
    assert r.outcome == "PASS"
    assert r.subclass is None
    assert r.metric == "mean_zscore"
    assert r.value == pytest.approx(0.9)
    assert r.tol == pytest.approx(5.0)
    assert r.wall_sec == pytest.approx(1.0)  # bn + rr
    assert r.method == "ssa" and r.reference_engine == "roadrunner"


def test_fail_is_red_diff_with_reason_in_comment():
    case = _compared(
        "BIOMD2", "fail", 41.0, diff_attribution={"reason": "bngsim SSA deviates from bngsim ODE"}
    )
    r = sa.case_to_job_result(case)
    assert r.outcome == "DIFF" and r.subclass == "bngsim_suspect"
    assert "deviates from bngsim ODE" in r.comment
    assert sa.is_expected(r.outcome, r.subclass) is False


def test_diff_not_bngsim_is_green_diff():
    case = _compared(
        "BIOMD3",
        "diff_not_bngsim",
        12.0,
        diff_attribution={"reason": "bngsim SSA tracks bngsim ODE; RR does not; ODEs agree"},
    )
    r = sa.case_to_job_result(case)
    assert r.outcome == "DIFF" and r.subclass == "diff_not_bngsim"
    assert sa.is_expected(r.outcome, r.subclass) is True
    assert "RR does not" in r.comment


def test_rr_known_comment_includes_issue():
    case = _compared(
        "BIOMD0000001026",
        "rr_known",
        30.0,
        known_rr_issue={
            "issue": "sys-bio/roadrunner#1318",
            "note": "time-dependent assignmentRule lag",
        },
    )
    r = sa.case_to_job_result(case)
    assert r.outcome == "DIFF" and r.subclass == "rr_known"
    assert "roadrunner#1318" in r.comment
    assert sa.is_expected(r.outcome, r.subclass) is True


def test_vacuous_lowcount_pass_is_flagged_in_comment():
    case = _compared("BIOMD4", "pass", 0.0, n_compared=0, n_skipped_lowcount=10)
    r = sa.case_to_job_result(case)
    assert r.outcome == "PASS"
    assert "vacuous pass" in r.comment


def test_sign_indefinite_annotation_appended():
    case = _compared(
        "BIOMD5",
        "diff_not_bngsim",
        9.0,
        diff_attribution={"reason": "RR-side"},
        sign_indefinite={"reaction": "R8 (0 -> nTeff)", "reverse_fires": 1213},
    )
    r = sa.case_to_job_result(case)
    assert "sign-indefinite" in r.comment and "R8" in r.comment


def test_too_slow_maps_to_timeout_with_wall_and_no_metric():
    case = {
        "name": "BIOMD6",
        "status": "too_slow",
        "source": "biomodels",
        "error": "exceeded 40.0s wall cap",
        "elapsed_sec_wall": 40.1,
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "TIMEOUT"
    assert r.metric is None and r.value is None
    assert r.wall_sec == pytest.approx(40.1)
    assert "wall cap" in r.exception
    assert sa.is_expected(r.outcome, r.subclass) is True


def test_roadrunner_side_error_is_reference_failed_green():
    # bngsim ran first; only RR refused (its gillespie can't do rate rules).
    # Not a bngsim defect -> REFERENCE_FAILED, tagged feature_gap, expected.
    case = {
        "name": "BIOMD7",
        "status": "error",
        "source": "biomodels",
        "error": "roadrunner: RuntimeError: gillespie integrator is unable to simulate rate rules",
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "REFERENCE_FAILED"
    assert r.reference_refusal == "feature_gap"
    assert "rate rules" in r.exception
    assert sa.is_expected(r.outcome, r.subclass) is True


def test_bngsim_side_error_is_exception_red():
    case = {
        "name": "BIOMD7b",
        "status": "error",
        "source": "biomodels",
        "error": "bngsim: ValueError: from_sbml failed",
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "EXCEPTION"
    assert r.reference_refusal is None
    assert sa.is_expected(r.outcome, r.subclass) is False


def test_structural_error_is_exception_red():
    # A disjoint-species / time-grid error carries no engine prefix -> EXCEPTION.
    case = {
        "name": "BIOMD7c",
        "status": "error",
        "source": "biomodels",
        "error": "no common species",
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "EXCEPTION"
    assert sa.is_expected(r.outcome, r.subclass) is False


def test_bngsim_undefined_symbol_loadfail_is_bad_test_green():
    # GH #119/#147: bngsim fail-closes on an SBML referencing an undefined symbol.
    # The SBML is malformed and RoadRunner rejects it too (verified on
    # MODEL2205030001: RR raises "Could not find requested symbol"), so it is a
    # BAD_TEST (bad SBML), not a red bngsim EXCEPTION — matching the ODE runner.
    case = {
        "name": "MODEL2205030001",
        "status": "error",
        "source": "biomodels",
        "error": (
            "bngsim: ModelError: Failed to load SBML file /x/MODEL2205030001.xml: "
            "Model references symbols that bind to no declared component (no global "
            "SId, kinetic-law parameter, or time csymbol) in an initialAssignment or "
            "rule:\n  - 'Kd_pAKT_basal' [initialAssignment]"
        ),
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "BAD_TEST"
    assert r.reference_refusal is None
    assert sa.is_expected(r.outcome, r.subclass) is True


def test_bngsim_unsupported_construct_refusal_is_bad_test_green():
    # GH #113: bngsim deliberately refuses a construct it cannot faithfully
    # simulate under ODE (delay DDE / AlgebraicRule / fast) rather than silently
    # approximating. RR's gillespie cannot run these either -> a feature BOTH
    # engines lack -> BAD_TEST (non-scoring), NOT a red EXCEPTION. Distinguishes
    # a deliberate refusal from a genuine bngsim load failure (the test above).
    case = {
        "name": "MODEL8102792069",
        "status": "error",
        "source": "biomodels",
        "error": (
            "bngsim: ModelError: Failed to load SBML file /x/MODEL8102792069.xml: "
            "Model contains constructs bngsim cannot faithfully simulate under ODE "
            "(no DDE/DAE solver):\n  - delay [reaction:infection]\n"
            "Previously these were dropped silently, integrating a different model."
        ),
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "BAD_TEST"
    assert r.reference_refusal is None  # reserved for REFERENCE_FAILED rows
    assert sa.is_expected(r.outcome, r.subclass) is True


def test_skip_maps_to_skip_green():
    case = {
        "name": "BIOMD8",
        "status": "skip",
        "source": "biomodels",
        "error": "sbml file not found",
    }
    r = sa.case_to_job_result(case)
    assert r.outcome == "SKIP"
    assert sa.is_expected(r.outcome, r.subclass) is True


# --- payload_to_report ----------------------------------------------------
def _payload(cases, machine=None):
    return {
        "config": {"mean_z_tol": 5.0, "corpus": "biomodels"},
        "machine_info": machine
        or {"date": "2026-06-06 00:00 UTC", "versions": {"bngsim": "0.9.36"}},
        "cases": {c["name"]: c for c in cases},
    }


def test_payload_to_report_tally_and_sorting():
    cases = [
        _compared("BIOMD_C", "pass", 0.1),
        _compared("BIOMD_A", "fail", 30.0, diff_attribution={"reason": "x"}),
        _compared("BIOMD_B", "rr_known", 20.0, known_rr_issue={"issue": "rr#1", "note": "n"}),
        {"name": "BIOMD_D", "status": "too_slow", "error": "cap", "elapsed_sec_wall": 40.0},
    ]
    meta, results = sa.payload_to_report(_payload(cases))
    assert [r.model_id for r in results] == ["BIOMD_A", "BIOMD_B", "BIOMD_C", "BIOMD_D"]
    assert meta["outcome_tally"]["PASS"] == 1
    assert meta["outcome_tally"]["DIFF"] == 2
    assert meta["outcome_tally"]["TIMEOUT"] == 1
    assert meta["subclass_breakdown"] == {"bngsim_suspect": 1, "rr_known": 1}
    # bngsim_suspect (fail) is the only NOT-expected case here.
    assert meta["n_not_expected"] == 1
    assert meta["versions"] == {"bngsim": "0.9.36"}


def test_payload_to_report_reads_legacy_flat_versions():
    # The committed legacy baseline stamps bngsim_version flat, no versions block.
    machine = {"date": "d", "bngsim_version": "0.9.30", "run_network_version": "rn 2.9"}
    meta, results = sa.payload_to_report(_payload([_compared("M", "pass", 0.0)], machine=machine))
    assert meta["versions"]["bngsim"] == "0.9.30"
    assert results[0].versions["bngsim"] == "0.9.30"


# --- #135 timing passthrough ----------------------------------------------
def test_timing_passes_through_unchanged():
    # The screen records a per-engine SSA timing dict on the case; attribution is
    # a pure passthrough into JobResult.timing — it must not reshape or drop it.
    timing = {
        "warmup": {"bngsim_sec": 0.07, "roadrunner_sec": 0.04, "roadrunner_source": "engine"},
        "bngsim": {"load_sec": 0.005, "rep_median_sec": 0.0001, "n_rep": 30},
        "roadrunner": {"load_sec": 0.07, "jit_sec": 0.06, "rep_median_sec": 0.00005},
    }
    r = sa.case_to_job_result(_compared("BIOMD_T", "pass", 0.5, timing=timing))
    assert r.timing == timing  # identical object content, nothing reshaped


def test_timing_absent_is_none_not_error():
    # A legacy / timeout / skip case that recorded no timing leaves it None.
    r = sa.case_to_job_result(_compared("BIOMD_NT", "pass", 0.5))
    assert r.timing is None


def test_payload_to_report_carries_machine_context():
    # CPU/OS context is threaded into meta so the matrix can show the concurrency
    # caveat; only the present keys are kept (no empty placeholders).
    machine = {
        "date": "d",
        "versions": {"bngsim": "0.9.36"},
        "platform": "macOS-14.0-arm64",
        "processor": "arm",
        "cpu_count": 6,
        "irrelevant": "dropped",
    }
    meta, _ = sa.payload_to_report(_payload([_compared("M", "pass", 0.0)], machine=machine))
    assert meta["machine"] == {
        "platform": "macOS-14.0-arm64",
        "processor": "arm",
        "cpu_count": 6,
    }


def test_payload_to_report_machine_empty_when_absent():
    # A legacy payload with no platform/cpu info yields an empty machine block,
    # never a KeyError.
    machine = {"date": "d", "versions": {"bngsim": "0.9.36"}}
    meta, _ = sa.payload_to_report(_payload([_compared("M", "pass", 0.0)], machine=machine))
    assert meta["machine"] == {}
