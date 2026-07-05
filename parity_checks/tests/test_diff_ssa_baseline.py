"""Hermetic tests for the SSA regression diff (diff_ssa_baseline).

The diff is the GH #108 gate: it must flag exactly the models that cross from an
EXPECTED state into a scoring DIFF/EXCEPTION, and must NOT flag green↔green
attribution churn, improvements, or a partial fresh run. These tests drive the
pure ``diff_reports`` on hand-built JobResult lists (no engines, no files) plus a
round-trip of the file loader over both report shapes.
"""

from __future__ import annotations

import json

import diff_ssa_baseline as dsb
from _core import JobResult, write_report


def jr(model_id, outcome, subclass=None, *, comment="", exception=""):
    return JobResult(
        model_id=model_id,
        method="ssa",
        reference_engine="roadrunner",
        outcome=outcome,
        subclass=subclass,
        comment=comment,
        exception=exception,
    )


# A small all-green baseline spanning every expected class.
BASELINE = [
    jr("M_pass", "PASS"),
    jr("M_rrknown", "DIFF", "rr_known"),
    jr("M_notbngsim", "DIFF", "diff_not_bngsim"),
    jr("M_slow", "TIMEOUT"),
    jr("M_reffail", "REFERENCE_FAILED", "feature_gap"),
]


def test_identical_run_is_clean():
    d = dsb.diff_reports(BASELINE, list(BASELINE))
    assert d["regressions"] == []
    assert d["improvements"] == []
    assert d["changed_attribution"] == []
    assert d["new_models"] == [] and d["not_in_fresh"] == []


def test_pass_to_bngsim_suspect_is_a_regression():
    fresh = [jr("M_pass", "DIFF", "bngsim_suspect", comment="bngsim deviates")] + BASELINE[1:]
    d = dsb.diff_reports(BASELINE, fresh)
    assert [r["model_id"] for r in d["regressions"]] == ["M_pass"]
    r = d["regressions"][0]
    assert r["from"] == "PASS" and r["to"] == "DIFF/bngsim_suspect"
    assert "bngsim deviates" in r["why"]


def test_pass_to_bngsim_exception_is_a_regression():
    fresh = [jr("M_pass", "EXCEPTION", exception="bngsim: boom")] + BASELINE[1:]
    d = dsb.diff_reports(BASELINE, fresh)
    assert [r["model_id"] for r in d["regressions"]] == ["M_pass"]
    assert d["regressions"][0]["to"] == "EXCEPTION"


def test_ode_level_diff_is_a_regression():
    fresh = [jr("M_pass", "DIFF", "ode_level")] + BASELINE[1:]
    d = dsb.diff_reports(BASELINE, fresh)
    assert [r["model_id"] for r in d["regressions"]] == ["M_pass"]


def test_green_attribution_churn_is_not_a_regression():
    # pass -> oracle-attributed RR-side DIFF: both expected, no regression.
    fresh = [jr("M_pass", "DIFF", "diff_not_bngsim")] + BASELINE[1:]
    d = dsb.diff_reports(BASELINE, fresh)
    assert d["regressions"] == []
    assert [c["model_id"] for c in d["changed_attribution"]] == ["M_pass"]


def test_pass_to_too_slow_is_green_churn_not_regression():
    fresh = [jr("M_pass", "TIMEOUT")] + BASELINE[1:]
    d = dsb.diff_reports(BASELINE, fresh)
    assert d["regressions"] == []
    assert [c["model_id"] for c in d["changed_attribution"]] == ["M_pass"]


def test_red_to_green_is_an_improvement_not_a_regression():
    baseline = [jr("M_x", "DIFF", "bngsim_suspect")] + BASELINE
    fresh = [jr("M_x", "PASS")] + BASELINE
    d = dsb.diff_reports(baseline, fresh)
    assert d["regressions"] == []
    assert [i["model_id"] for i in d["improvements"]] == ["M_x"]


def test_partial_fresh_run_lists_not_in_fresh_without_regressions():
    fresh = BASELINE[:2]  # only the first two models were run
    d = dsb.diff_reports(BASELINE, fresh)
    assert d["regressions"] == []
    assert d["not_in_fresh"] == ["M_notbngsim", "M_reffail", "M_slow"]


def test_new_model_in_fresh_is_listed():
    fresh = BASELINE + [jr("M_new", "PASS")]
    d = dsb.diff_reports(BASELINE, fresh)
    assert d["new_models"] == ["M_new"]
    assert d["regressions"] == []


def test_multiple_regressions_sorted_by_model_id():
    fresh = [
        jr("M_pass", "DIFF", "bngsim_suspect"),
        jr("M_rrknown", "EXCEPTION", exception="bngsim: x"),
    ] + BASELINE[2:]
    d = dsb.diff_reports(BASELINE, fresh)
    assert [r["model_id"] for r in d["regressions"]] == ["M_pass", "M_rrknown"]


# --- file loader round-trips both report shapes ---------------------------
def test_load_results_reads_core_report(tmp_path):
    p = tmp_path / "report.json"
    write_report(p, BASELINE, meta={"description": "x"})
    loaded = dsb.load_results(p)
    assert [r.model_id for r in loaded] == [r.model_id for r in BASELINE]
    assert dsb.diff_reports(BASELINE, loaded)["regressions"] == []


def test_load_results_reads_native_screen_json(tmp_path):
    native = {
        "config": {"mean_z_tol": 5.0},
        "machine_info": {"date": "d", "versions": {"bngsim": "0.9.36"}},
        "cases": {
            "M_pass": {"name": "M_pass", "status": "pass", "max_mean_z": 0.5},
            "M_fail": {
                "name": "M_fail",
                "status": "fail",
                "max_mean_z": 40.0,
                "diff_attribution": {"reason": "bngsim deviates"},
            },
        },
    }
    p = tmp_path / "ssa_screen.json"
    p.write_text(json.dumps(native))
    loaded = dsb.load_results(p)
    by = {r.model_id: r for r in loaded}
    assert by["M_pass"].outcome == "PASS"
    assert by["M_fail"].outcome == "DIFF" and by["M_fail"].subclass == "bngsim_suspect"
