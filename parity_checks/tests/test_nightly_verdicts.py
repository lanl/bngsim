"""Hermetic tests for the nightly regression check (parity_checks/nightly/verdicts.py, GH #702).

The nightly job treats a report as its verdict, so this tool decides what alerts.
It must flag a good case that stops being good (including one that silently
drops out of the run), a failing case that starts crashing, a jump in bngsim's
solver work and a run that compared too little. It must NOT flag improvements,
churn between two failing states, or new cases. Everything here is synthetic
JSON: no engines, no corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import verdicts as V
from _core import JobResult, write_report

BASELINES = Path(V.__file__).resolve().parent / "baselines"


def core(cases: dict, suite="t") -> dict:
    return {
        "schema": V.SCHEMA,
        "suite": suite,
        "kind": "core",
        "source": {},
        "n_cases": len(cases),
        "tally": V.tally(cases),
        "cases": cases,
    }


def rec(outcome, subclass=None, work=None, sec=None, backend=None):
    r = {"outcome": outcome, "subclass": subclass}
    if work is not None:
        r["work"] = work
    if sec is not None:
        r["bngsim_sec"] = sec
    if backend is not None:
        r["backend"] = backend
    return r


W = {"n_steps": 100, "n_rhs_evals": 200, "n_jac_evals": 20}

BASE = core(
    {
        "A|ode": rec("PASS", work=W, sec=1.0),
        "B|ode": rec("PASS", work=W, sec=1.0),
        "C|ode": rec("DIFF", "known_artifact"),
        "D|ode": rec("REFERENCE_FAILED"),
        "E|ode": rec("TIMEOUT"),
    }
)


def fresh_with(**changes) -> dict:
    cases = {k: dict(v) for k, v in BASE["cases"].items()}
    for key, value in changes.items():
        cid = key.replace("_", "|")
        if value is None:
            cases.pop(cid)
        else:
            cases[cid] = value
    return core(cases)


def test_identical_run_is_clean():
    r = V.diff(BASE, BASE)
    assert r["n_alerts"] == 0
    assert not (r["regressed"] or r["new_crash"] or r["improved"] or r["changed"] or r["new"])


@pytest.mark.parametrize(
    "after", ["DIFF", "EXCEPTION", "TIMEOUT", "SKIP", "REFERENCE_FAILED", "BAD_TEST"]
)
def test_good_to_anything_else_regresses(after):
    r = V.diff(BASE, fresh_with(A_ode=rec(after)))
    assert [x["case"] for x in r["regressed"]] == ["A|ode"]
    assert r["alerts"]["regressed"] == 1 and r["n_alerts"] >= 1


def test_good_case_missing_from_run_regresses():
    r = V.diff(BASE, fresh_with(A_ode=None))
    assert r["regressed"] == [{"case": "A|ode", "before": "PASS", "after": "MISSING", "note": ""}]


def test_failing_case_missing_from_run_is_reported_not_alerted():
    # 4 of 5 cases trips the default 90 % coverage floor; this test is about the per-case rule.
    r = V.diff(BASE, fresh_with(C_ode=None), min_fraction=0.5)
    assert r["n_alerts"] == 0
    assert [x["case"] for x in r["dropped"]] == ["C|ode"]


def test_failing_case_that_starts_crashing_alerts():
    r = V.diff(BASE, fresh_with(C_ode=rec("EXCEPTION")))
    assert [x["case"] for x in r["new_crash"]] == ["C|ode"]
    assert r["alerts"]["new_crash"] == 1


def test_crash_that_stays_a_crash_does_not_alert_again():
    base = fresh_with(C_ode=rec("EXCEPTION"))
    r = V.diff(base, base)
    assert r["n_alerts"] == 0


def test_improvement_and_failing_churn_do_not_alert():
    r = V.diff(BASE, fresh_with(C_ode=rec("PASS"), E_ode=rec("DIFF")))
    assert r["n_alerts"] == 0
    assert [x["case"] for x in r["improved"]] == ["C|ode"]
    assert [x["case"] for x in r["changed"]] == ["E|ode"]


def test_subclass_change_is_a_change_not_an_alert():
    r = V.diff(BASE, fresh_with(C_ode=rec("DIFF", "rr_known")))
    assert r["n_alerts"] == 0
    assert r["changed"][0]["after"] == "DIFF (rr_known)"


def test_timeout_flips_are_reported_not_alerted_between_two_nights():
    last_night = fresh_with(E_ode=rec("PASS"))
    tonight = fresh_with(E_ode=rec("TIMEOUT"), A_ode=rec("TIMEOUT"))
    r = V.diff(last_night, tonight, timeout_flips_info=True)
    assert r["n_alerts"] == 0
    assert sorted(x["case"] for x in r["changed"]) == ["A|ode", "E|ode"]
    assert all("timeout flip" in x["note"] for x in r["changed"])
    # ... but against the committed baseline PASS -> TIMEOUT is still a regression.
    assert V.diff(BASE, fresh_with(A_ode=rec("TIMEOUT")))["alerts"]["regressed"] == 1


def test_timeout_flips_info_does_not_hide_other_regressions():
    r = V.diff(BASE, fresh_with(A_ode=rec("DIFF")), timeout_flips_info=True)
    assert r["alerts"]["regressed"] == 1


def test_flaky_case_changes_never_alert():
    flaky = {"A|ode": "flips with runner speed"}
    r = V.diff(BASE, fresh_with(A_ode=rec("DIFF"), C_ode=rec("EXCEPTION")), flaky=flaky)
    assert [x["case"] for x in r["regressed"]] == []
    assert [x["case"] for x in r["new_crash"]] == ["C|ode"]  # not listed: still alerts
    assert r["changed"][0]["case"] == "A|ode" and "flaky" in r["changed"][0]["note"]


def test_a_row_re_baselined_since_that_night_is_reported_not_alerted():
    # #1427: a stricter check moves A from PASS to DIFF and re-records it. Against
    # the night before, that row is the reviewed change, on the pull request's own
    # run and on the first night after it merges.
    then = BASE
    now = fresh_with(A_ode=rec("DIFF"))
    r = V.diff(BASE, now, rebaselined=V.rebaselined(then, now))
    assert r["n_alerts"] == 0
    assert r["changed"][0]["case"] == "A|ode"
    assert "re-baselined since that night" in r["changed"][0]["note"]
    # Only into the verdict recorded: a run where A crashes instead still alerts.
    r = V.diff(BASE, fresh_with(A_ode=rec("EXCEPTION")), rebaselined=V.rebaselined(then, now))
    assert r["alerts"]["regressed"] == 1


def test_a_fix_that_regresses_into_a_stale_baseline_still_alerts():
    # The baseline still says DIFF from before C was fixed; last night C passed and
    # tonight it is back to DIFF. Nobody recorded that, so it is the regression the
    # night-to-night comparison exists for.
    stale = BASE
    last_night = fresh_with(C_ode=rec("PASS"))
    r = V.diff(last_night, BASE, rebaselined=V.rebaselined(stale, stale))
    assert [x["case"] for x in r["regressed"]] == ["C|ode"]


def test_rebaselined_lists_new_and_moved_rows_only():
    now = fresh_with(A_ode=rec("DIFF"), C_ode=rec("DIFF", "rr_known"), F_ode=rec("PASS"))
    assert sorted(V.rebaselined(BASE, now)) == ["A|ode", "C|ode", "F|ode"]
    assert V.rebaselined(BASE, BASE) == {}


def test_cli_reads_the_baseline_then_and_now(tmp_path):
    last_night = _write(tmp_path, "prev.json", BASE)
    tonight = _write(tmp_path, "tonight.json", fresh_with(A_ode=rec("DIFF")))
    then = _write(tmp_path, "then.json", BASE)
    assert V.main(["diff", "--baseline", last_night, "--fresh", tonight]) == 1
    args = ["diff", "--baseline", last_night, "--fresh", tonight, "--baseline-then", then]
    assert V.main([*args, "--baseline-now", tonight]) == 0
    # Without a baseline of that night's commit, nothing is excused.
    missing = str(tmp_path / "absent.json")
    assert V.main([*args[:5], "--baseline-then", missing, "--baseline-now", tonight]) == 1


def test_cli_reads_the_flaky_file_for_its_suite(tmp_path):
    base = _write(tmp_path, "base.json", BASE)
    worse = _write(tmp_path, "worse.json", fresh_with(A_ode=rec("TIMEOUT")))
    flaky = tmp_path / "flaky.json"
    flaky.write_text(json.dumps({"t": {"A|ode": "measured"}, "other": {"B|ode": "x"}}))
    assert V.main(["diff", "--baseline", base, "--fresh", worse]) == 1
    assert V.main(["diff", "--baseline", base, "--fresh", worse, "--flaky", str(flaky)]) == 0


def test_the_committed_flaky_file_is_well_formed():
    from pathlib import Path

    path = Path(V.__file__).with_name("flaky.json")
    data = json.loads(path.read_text())
    for suite, cases in data.items():
        if suite.startswith("_"):
            continue
        assert isinstance(cases, dict) and cases, suite
        for cid, reason in cases.items():
            assert "|" in cid or suite in ("sbml_semantic", "dsmts"), cid
            assert len(reason) > 20, f"{cid}: give the measurement, not just a label"


def test_new_case_is_reported_not_alerted():
    fresh = fresh_with()
    fresh["cases"]["Z|ode"] = rec("EXCEPTION")
    r = V.diff(BASE, fresh)
    assert r["n_alerts"] == 0
    assert [x["case"] for x in r["new"]] == ["Z|ode"]


def test_work_jump_on_one_case_alerts():
    more = {**W, "n_rhs_evals": 400}
    r = V.diff(BASE, fresh_with(A_ode=rec("PASS", work=more, sec=1.0)))
    assert [(x["case"], x["counter"]) for x in r["efficiency"]] == [("A|ode", "n_rhs_evals")]
    assert r["efficiency"][0]["ratio"] == 2.0
    assert r["alerts"]["efficiency_cases"] == 1


def test_work_jump_on_a_flaky_case_is_reported_not_alerted():
    # GH #1153: --flaky promises a listed case's changes never alert, but a
    # listed case's per-case work increase still did.
    more = {**W, "n_steps": 250}
    fresh = fresh_with(A_ode=rec("PASS", work=more, sec=1.0))
    r = V.diff(BASE, fresh, flaky={"A|ode": "steps follow the platform's rounding"})
    assert [(x["case"], x["counter"]) for x in r["efficiency"]] == [("A|ode", "n_steps")]
    assert r["efficiency"][0]["note"].startswith("flaky: ")
    assert r["efficiency"][0]["flaky"] is True
    assert r["alerts"]["efficiency_cases"] == 0
    # It still counts in the totals: 350 steps against 200 is past x1.05, and
    # that is the one alert.
    assert r["work_totals"]["n_steps"]["after"] == 350
    assert r["work_totals"]["n_steps"]["alert"] is True
    assert r["n_alerts"] == r["alerts"]["efficiency_totals"] == 1
    md = V.render_md(r)
    assert "flaky: steps follow" in md
    # The header counts it apart, not in bold among the alerting increases.
    assert "Efficiency: 0 counter increase(s) above x1.25, and 1 on flaky cases" in md
    assert "**Efficiency" not in md
    # The same jump on a case not listed alerts as before.
    assert V.diff(BASE, fresh, flaky={"B|ode": "x"})["alerts"]["efficiency_cases"] == 1


def test_small_absolute_work_change_does_not_alert():
    # x2.5 on the Jacobian count clears the ratio bar, but +9 is under the n_jac_evals floor of 10.
    base = core({"A|ode": rec("PASS", work={"n_steps": 10, "n_rhs_evals": 20, "n_jac_evals": 6})})
    fresh = core(
        {"A|ode": rec("PASS", work={"n_steps": 10, "n_rhs_evals": 20, "n_jac_evals": 15})}
    )
    r = V.diff(base, fresh)
    assert r["efficiency"] == []


def test_work_on_a_case_that_is_not_good_in_both_is_ignored():
    r = V.diff(BASE, fresh_with(A_ode=rec("DIFF", work={**W, "n_steps": 10_000})))
    assert r["efficiency"] == []
    assert r["alerts"]["regressed"] == 1


def test_corpus_total_work_growth_alerts_even_below_the_per_case_bar():
    # +10 % on every case: under the x1.25 per-case bar, over the x1.05 total bar.
    more = {k: int(v * 1.10) for k, v in W.items()}
    r = V.diff(
        BASE,
        fresh_with(A_ode=rec("PASS", work=more, sec=1.0), B_ode=rec("PASS", work=more, sec=1.0)),
    )
    assert r["efficiency"] == []
    assert r["work_totals"]["n_rhs_evals"]["alert"] is True
    assert r["alerts"]["efficiency_totals"] == 3


def _on(obj, cpu):
    return dict(obj, source={"cpu": cpu})


def test_wall_on_the_same_cpu_alerts_past_the_coarse_ratio():
    base = _on(BASE, "EPYC 7763")
    slower = fresh_with(A_ode=rec("PASS", work=W, sec=1.9), B_ode=rec("PASS", work=W, sec=1.9))
    assert V.diff(base, _on(slower, "EPYC 7763"), wall_min_delta=0)["alerts"]["wall"] == 0
    much = fresh_with(A_ode=rec("PASS", work=W, sec=2.5), B_ode=rec("PASS", work=W, sec=2.5))
    r = V.diff(base, _on(much, "EPYC 7763"), wall_min_delta=0)
    assert r["alerts"]["wall"] == 1 and r["wall"]["ratio"] == 2.5 and r["wall"]["same_cpu"]


def test_wall_across_cpu_models_alerts_only_on_a_blowup():
    # Measured: the compiled arm ran 1.8x slower on an EPYC 7763 than on a 9V45.
    base = _on(BASE, "EPYC 9V45")
    much = fresh_with(A_ode=rec("PASS", work=W, sec=2.5), B_ode=rec("PASS", work=W, sec=2.5))
    r = V.diff(base, _on(much, "EPYC 7763"), wall_min_delta=0)
    assert r["alerts"]["wall"] == 0 and not r["wall"]["same_cpu"] and r["wall"]["limit"] == 4.0
    huge = fresh_with(A_ode=rec("PASS", work=W, sec=5.0), B_ode=rec("PASS", work=W, sec=5.0))
    assert V.diff(base, _on(huge, "EPYC 7763"), wall_min_delta=0)["alerts"]["wall"] == 1
    # an unknown CPU on either side counts as "not the same"
    assert V.diff(BASE, much, wall_min_delta=0)["alerts"]["wall"] == 0


def test_wall_growth_under_the_absolute_floor_does_not_alert():
    # bng_nf, 2026-09-29: its bngsim_sec was only the model load (before #901), and #885 added a
    # millisecond XML scan to it. 0.011 s -> 0.67 s over 231 cases is x61 and
    # 0.66 s of wall, on a leg whose runs take minutes.
    base = _on(core({f"m{i}|stochastic": rec("PASS", sec=0.00005) for i in range(200)}), "Intel")
    fresh = _on(core({f"m{i}|stochastic": rec("PASS", sec=0.003) for i in range(200)}), "AMD")
    r = V.diff(base, fresh)
    assert r["wall"]["ratio"] > 50 and r["wall"]["min_delta"] == V.WALL_MIN_DELTA_SEC
    assert r["alerts"]["wall"] == 0 and r["n_alerts"] == 0
    assert "+30.0 s" in V.render_md(r, "")


def test_wall_blowup_past_the_floor_still_alerts():
    # A real blowup on an ODE leg's scale: 80 s -> 400 s across CPU models.
    base = _on(core({f"m{i}|ode": rec("PASS", sec=0.8) for i in range(100)}), "Intel")
    fresh = _on(core({f"m{i}|ode": rec("PASS", sec=4.0) for i in range(100)}), "AMD")
    r = V.diff(base, fresh)
    assert r["alerts"]["wall"] == 1
    # the same ratio on a total too small to clear the floor does not
    tiny_b = _on(core({"m|ode": rec("PASS", sec=1.0)}), "Intel")
    tiny_f = _on(core({"m|ode": rec("PASS", sec=5.0)}), "AMD")
    assert V.diff(tiny_b, tiny_f)["alerts"]["wall"] == 0
    assert V.diff(tiny_b, tiny_f, wall_min_delta=2.0)["alerts"]["wall"] == 1


def test_incomplete_run_alerts():
    fresh = core({"C|ode": rec("DIFF", "known_artifact")})
    r = V.diff(BASE, fresh)
    assert r["alerts"]["incomplete"] == 1


def test_empty_run_alerts_even_against_an_empty_baseline():
    r = V.diff(core({}), core({}))
    assert r["alerts"]["incomplete"] == 1


def test_expected_backend_flags_a_good_case_that_ran_another_backend():
    base = core({"A|ode": rec("PASS", backend="cc"), "B|ode": rec("PASS", backend="cc")})
    fresh = core({"A|ode": rec("PASS", backend="cc"), "B|ode": rec("PASS", backend="exprtk")})
    r = V.diff(base, fresh, expect_backend="cc")
    assert [x["case"] for x in r["backend_miss"]] == ["B|ode"]
    assert r["alerts"]["backend"] == 1
    assert V.diff(base, fresh)["alerts"]["backend"] == 0


def test_interpreter_arm_flags_a_case_that_compiled_by_itself():
    # The #872 shape: a >=256-species model auto-compiled in the interpreter arm and
    # still passed, so only the backend check can name it. A row that recorded no
    # backend (bngsim never ran) and a row that is not good are judged by their
    # outcomes instead, not as backend misses.
    base = core(
        {
            "A|ode": rec("PASS", backend="exprtk"),
            "B|ode": rec("PASS", backend="exprtk"),
            "C|ode": rec("BAD_TEST"),
            "D|ode": rec("DIFF", backend="exprtk"),
        }
    )
    fresh = core(
        {
            "A|ode": rec("PASS", backend="exprtk"),
            "B|ode": rec("PASS", backend="cc"),
            "C|ode": rec("BAD_TEST"),
            "D|ode": rec("DIFF", backend="cc"),
        }
    )
    r = V.diff(base, fresh, expect_backend="exprtk")
    assert r["backend_miss"] == [{"case": "B|ode", "after": "cc", "note": "expected exprtk"}]
    assert r["alerts"]["backend"] == 1
    assert V.diff(base, base, expect_backend="exprtk")["alerts"]["backend"] == 0


def test_kind_mismatch_is_refused():
    other = dict(BASE, kind="dsmts")
    with pytest.raises(ValueError, match="kind mismatch"):
        V.diff(BASE, other)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def _jr(model_id, outcome, method="ode", **kw):
    return JobResult(
        model_id=model_id, method=method, reference_engine="roadrunner", outcome=outcome, **kw
    )


def test_extract_core_reads_outcome_work_wall_and_backend(tmp_path):
    timing = {
        "bngsim": {
            "parse_sec": 0.5,
            "integrate_cold_sec": 1.5,
            "integrate_sec": 99.0,  # warm-min headline, not a phase: must not be summed
            "config": {"codegen": "cc"},
            "work": {"n_steps": 7, "n_rhs_evals": 11, "n_jac_evals": 2, "unrelated": 5},
        }
    }
    path = tmp_path / "report.json"
    write_report(
        path,
        [
            _jr("M1", "PASS", timing=timing),
            _jr("M2", "EXCEPTION", exception="bngsim: RuntimeError: boom"),
            _jr("M2", "PASS", method="sens/staggered"),
        ],
        meta={"suite": "rr_parity", "git_rev": "abc1234", "hardware": {"cpu": "EPYC 7763"}},
    )
    obj = V.extract("core", path, "rr_ode")
    assert obj["n_cases"] == 3
    m1 = obj["cases"]["M1|ode"]
    assert m1 == {
        "outcome": "PASS",
        "subclass": None,
        "work": {"n_steps": 7, "n_rhs_evals": 11, "n_jac_evals": 2},
        "bngsim_sec": 2.0,
        "backend": "cc",
    }
    assert obj["cases"]["M2|ode"]["note"] == "bngsim: RuntimeError: boom"
    assert obj["cases"]["M2|sens/staggered"]["outcome"] == "PASS"
    assert obj["source"]["git_rev"] == "abc1234"
    assert obj["source"]["cpu"] == "EPYC 7763"


def test_extract_counts_a_stochastic_ensemble(tmp_path):
    """Issue #901: a stochastic row's simulation is its ``ensemble_sec``, which
    the wall check did not count. bng_nf summed the session's construction
    alone, 0.012 s beside 4004.5 s of NFsim. Each replicate's
    ``NfsimSession.initialize`` is timed apart from its run, and recorded only
    as a mean, so it counts ``n_rep`` times: 452 s of bng_nf's 4126 s. The other
    per-replicate statistics are views of ``ensemble_sec`` and are not summed."""
    timing = {
        "bngsim": {
            "io_sec": 0.0,
            "load_sec": 0.9,
            "n_rep": 10,
            "rep_mean_sec": 0.03,
            "rep_median_sec": 0.03,
            "rep_cold_sec": 0.03,
            "ensemble_sec": 0.3,
            "setup_per_rep_sec": 0.36,
        }
    }
    path = tmp_path / "report.json"
    write_report(path, [_jr("M1", "PASS", method="stochastic", timing=timing)])
    obj = V.extract("core", path, "bng_nf")
    assert obj["cases"]["M1|stochastic"]["bngsim_sec"] == pytest.approx(0.9 + 0.3 + 3.6)
    assert "setup_sec" in obj["bngsim_phases_seen"]


def test_an_ssa_row_has_no_setup_phase(tmp_path):
    """A replicate with no initialize of its own (SSA's reset and run) adds
    nothing, and neither does a mean the runner could not take."""
    ssa = {"bngsim": {"load_sec": 0.2, "n_rep": 10, "ensemble_sec": 5.0}}
    nf = {"bngsim": {"load_sec": 0.2, "n_rep": 0, "setup_per_rep_sec": None}}
    path = tmp_path / "report.json"
    write_report(
        path,
        [
            _jr("S", "PASS", method="stochastic", timing=ssa),
            _jr("N", "PASS", method="stochastic", timing=nf),
        ],
    )
    obj = V.extract("core", path, "bng_ssa")
    assert obj["cases"]["S|stochastic"]["bngsim_sec"] == pytest.approx(5.2)
    assert obj["cases"]["N|stochastic"]["bngsim_sec"] == pytest.approx(0.2)
    assert "setup_sec" not in obj["bngsim_phases_seen"]


STOCH_SEEN = {"io_sec", "load_sec", "ensemble_sec", "setup_sec"}


def _phased(obj, seen, counted=None):
    out = dict(obj, bngsim_phases_seen=sorted(seen))
    out["bngsim_phases"] = list(counted if counted is not None else V.BNGSIM_PHASES)
    return out


def test_a_slower_ensemble_alerts():
    """With the ensemble counted, a 2x slower NFsim on the same CPU clears both
    the ratio and the absolute floor, at bng_nf's scale (about 4100 s over
    235 cases), between two files that both record the phases they sum."""

    def night(sec):
        cases = {f"m{i}|stochastic": rec("PASS", sec=sec) for i in range(235)}
        return _phased(_on(core(cases), "EPYC 9V45"), STOCH_SEEN)

    base = night(17.0)
    r = V.diff(base, night(34.5))
    assert r["wall"]["same_phases"] and r["wall"]["same_cpu"]
    assert r["alerts"]["wall"] == 1 and r["n_alerts"] == 1
    assert V.diff(base, night(18.0))["n_alerts"] == 0


def test_a_load_phase_is_not_compared_with_an_ensemble():
    """The first night that counts the ensemble, against the night before it
    (issue #901): bng_ssa's 0.212 s of parse and load against 533 s of SSA
    read as x2515. The file from before says nothing of its phases, so it
    summed the old ones, and tonight's timed one it did not count."""
    base = _on(core({f"m{i}|stochastic": rec("PASS", sec=0.004) for i in range(50)}), "EPYC")
    fresh = _phased(
        _on(core({f"m{i}|stochastic": rec("PASS", sec=10.0) for i in range(50)}), "EPYC"),
        {"load_sec", "ensemble_sec"},
    )
    r = V.diff(base, fresh, phases_may_differ=True)
    assert r["alerts"]["wall"] == 0 and r["n_alerts"] == 0
    assert r["wall"]["same_phases"] is False
    md = V.render_md(r, "")
    assert "not compared" in md and "ALERT" not in md


def test_a_baseline_that_sums_other_phases_alerts():
    """Against the committed baseline, the same mismatch alerts: the wall check
    would otherwise stay off, silently, until someone re-extracted it, which is
    how #893's floor blinded these legs. Here the baseline was extracted before
    ``setup_sec`` was counted, and tonight's NF rows time it."""
    before_setup = [p for p in V.BNGSIM_PHASES if p != "setup_sec"]
    base = _phased(
        _on(core({f"m{i}|stochastic": rec("PASS", sec=17.0) for i in range(50)}), "EPYC"),
        {"load_sec", "ensemble_sec"},
        before_setup,
    )
    fresh = _phased(
        _on(core({f"m{i}|stochastic": rec("PASS", sec=19.0) for i in range(50)}), "EPYC"),
        STOCH_SEEN,
    )
    r = V.diff(base, fresh)
    assert r["wall"]["same_phases"] is False
    assert r["alerts"]["wall_phases"] == 1 and r["n_alerts"] == 1
    assert "re-extract the baseline" in V.render_md(r, "")
    assert V.diff(base, fresh, phases_may_differ=True)["n_alerts"] == 0


def test_a_phase_added_later_is_not_read_as_legacy():
    """A file that does not say what it summed summed the phases from before
    #901, and only those: ``setup_sec``, counted later, is not one of them."""
    assert set(V.LEGACY_BNGSIM_PHASES) < set(V.BNGSIM_PHASES)
    assert not {"ensemble_sec", "setup_sec"} & set(V.LEGACY_BNGSIM_PHASES)
    base = _on(core({f"m{i}|stochastic": rec("PASS", sec=17.0) for i in range(50)}), "EPYC")
    fresh = _phased(
        _on(core({f"m{i}|stochastic": rec("PASS", sec=19.0) for i in range(50)}), "EPYC"),
        {"load_sec", "setup_sec"},
    )
    assert V.diff(base, fresh, phases_may_differ=True)["wall"]["same_phases"] is False


def test_an_ode_leg_compares_across_the_change():
    """Control. An ODE row records no ensemble, so its bngsim_sec is the same
    sum before and after #901, and a blowup against a baseline from before
    still alerts."""
    base = _on(core({f"m{i}|ode": rec("PASS", sec=0.8) for i in range(100)}), "EPYC")
    fresh = _phased(
        _on(core({f"m{i}|ode": rec("PASS", sec=4.0) for i in range(100)}), "EPYC"),
        {"parse_sec", "integrate_cold_sec"},
    )
    r = V.diff(base, fresh)
    assert r["wall"]["same_phases"] is True
    assert r["alerts"]["wall"] == 1 and r["alerts"]["wall_phases"] == 0


def test_extract_records_the_phases_it_summed(tmp_path):
    timing = {"bngsim": {"load_sec": 0.9, "ensemble_sec": 0.3, "rep_mean_sec": 0.03}}
    path = tmp_path / "report.json"
    write_report(path, [_jr("M1", "PASS", method="stochastic", timing=timing)])
    obj = V.extract("core", path, "bng_nf")
    assert obj["bngsim_phases"] == list(V.BNGSIM_PHASES)
    assert obj["bngsim_phases_seen"] == ["ensemble_sec", "load_sec"]


def test_the_stochastic_baselines_count_their_ensembles():
    """The committed bng_nf and bng_ssa baselines were extracted with the
    ensemble, and on bng_nf each replicate's setup, counted (issue #901), so
    tonight's extract is compared like for like and does not read about 4100 s
    against 0.012 s. A baseline that sums fewer phases than tonight's extract
    alerts every night, so they record the current BNGSIM_PHASES."""
    for suite, at_least, seen in (
        ("bng_nf", 1000.0, {"ensemble_sec", "setup_sec"}),
        ("bng_ssa", 100.0, {"ensemble_sec"}),
    ):
        obj = json.loads((BASELINES / f"{suite}.json").read_text())
        total = sum(c.get("bngsim_sec") or 0.0 for c in obj["cases"].values())
        assert total > at_least, suite
        assert seen <= set(obj["bngsim_phases_seen"]), suite
        assert obj["bngsim_phases"] == list(V.BNGSIM_PHASES), suite


def test_multi_segment_backend_wins_over_the_replay_label(tmp_path):
    timing = {"bngsim": {"backend": "cc", "config": {"codegen": "full-protocol replay"}}}
    path = tmp_path / "report.json"
    write_report(path, [_jr("M1", "PASS", timing=timing)], meta={})
    assert V.extract("core", path, "x")["cases"]["M1|ode"]["backend"] == "cc"


def test_work_keys_match_the_harness_recorder():
    # verdicts.py is stdlib-only and standalone, so it keeps its own copy of the list
    # _core.work records; they must not drift apart.
    from _core.work import WORK_KEYS, work_counters

    assert V.WORK_KEYS == WORK_KEYS
    assert set(V.WORK_ALERT_KEYS) <= set(WORK_KEYS)
    stats = {"n_steps": 3.0, "n_rhs_evals": 5, "linear_solver": 1, "n_jac_evals": None}
    assert work_counters(stats) == {"n_steps": 3, "n_rhs_evals": 5}
    assert work_counters(None) == {}


def test_extract_merges_passes_and_refuses_overlap(tmp_path):
    light, heavy = tmp_path / "light.json", tmp_path / "heavy.json"
    write_report(light, [_jr("M1", "PASS")], meta={"suite": "rr_parity", "elapsed_sec": 10})
    write_report(heavy, [_jr("M9", "TIMEOUT")], meta={"suite": "rr_parity", "elapsed_sec": 90})
    obj = V.extract("core", [light, heavy], "rr_ode")
    assert sorted(obj["cases"]) == ["M1|ode", "M9|ode"]
    assert [p["elapsed_sec"] for p in obj["source"]["parts"]] == [10, 90]
    with pytest.raises(ValueError, match="more than one report"):
        V.extract("core", [light, light], "rr_ode")


def test_extract_core_refuses_a_duplicate_case(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, [_jr("M1", "PASS"), _jr("M1", "DIFF")], meta={})
    with pytest.raises(ValueError, match="duplicate"):
        V.extract("core", path, "x")


def test_extract_sbml_suite_drops_out_of_scope_cases(tmp_path):
    payload = {
        "cases": [
            {"case": "00001", "status": "pass"},
            {"case": "00002", "status": "value_mismatch", "error": "S1 off by 3%"},
            {"case": "00003", "status": "skipped"},
        ]
    }
    path = tmp_path / "sbml.json"
    path.write_text(json.dumps(payload))
    obj = V.extract("sbml-suite", path, "sbml_semantic")
    assert sorted(obj["cases"]) == ["00001", "00002"]
    assert obj["cases"]["00002"]["note"] == "S1 off by 3%"


def test_extract_dsmts(tmp_path):
    payload = {
        "cases": {"00002": {"status": "fail"}, "00001": {"status": "pass"}},
        "summary": {"total": 2},
    }
    path = tmp_path / "dsmts.json"
    path.write_text(json.dumps(payload))
    obj = V.extract("dsmts", path, "dsmts")
    assert list(obj["cases"]) == ["00001", "00002"]
    assert obj["tally"] == {"fail": 1, "pass": 1}


def test_verdict_file_round_trips_with_one_case_per_line(tmp_path):
    path = tmp_path / "v.json"
    V.dump_verdicts(BASE, path)
    assert V.load_verdicts(path) == BASE
    lines = path.read_text().splitlines()
    assert sum(1 for ln in lines if ln.lstrip().startswith('"A|ode"')) == 1


def test_load_refuses_an_unknown_schema(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps(dict(BASE, schema=99)))
    with pytest.raises(ValueError, match="schema"):
        V.load_verdicts(path)


# --------------------------------------------------------------------------- #
# CLI exit codes (what the workflow keys on)
# --------------------------------------------------------------------------- #
def _write(tmp_path, name, obj):
    p = tmp_path / name
    V.dump_verdicts(obj, p)
    return str(p)


def test_cli_diff_exit_codes(tmp_path):
    base = _write(tmp_path, "base.json", BASE)
    same = _write(tmp_path, "same.json", BASE)
    worse = _write(tmp_path, "worse.json", fresh_with(A_ode=rec("DIFF")))
    md = tmp_path / "out.md"
    js = tmp_path / "out.json"
    assert V.main(["diff", "--baseline", base, "--fresh", same]) == 0
    assert (
        V.main(
            [
                "diff",
                "--baseline",
                base,
                "--fresh",
                worse,
                "--md-out",
                str(md),
                "--json-out",
                str(js),
            ]
        )
        == 1
    )
    assert "Regressed" in md.read_text()
    assert json.loads(js.read_text())["alerts"]["regressed"] == 1


def test_cli_missing_baseline(tmp_path):
    fresh = _write(tmp_path, "fresh.json", BASE)
    missing = str(tmp_path / "nope.json")
    assert V.main(["diff", "--baseline", missing, "--fresh", fresh]) == 2
    js = tmp_path / "d.json"
    assert (
        V.main(
            [
                "diff",
                "--baseline",
                missing,
                "--fresh",
                fresh,
                "--allow-missing-baseline",
                "--json-out",
                str(js),
            ]
        )
        == 0
    )
    assert json.loads(js.read_text())["no_baseline"] is True


def test_cli_summarize(tmp_path):
    base = _write(tmp_path, "base.json", BASE)
    worse = _write(tmp_path, "worse.json", fresh_with(A_ode=rec("EXCEPTION")))
    clean_js, bad_js, none_js = (tmp_path / n for n in ("clean.json", "bad.json", "none.json"))
    V.main(["diff", "--baseline", base, "--fresh", base, "--json-out", str(clean_js)])
    V.main(
        [
            "diff",
            "--baseline",
            base,
            "--fresh",
            worse,
            "--json-out",
            str(bad_js),
            "--label",
            "vs baseline",
        ]
    )
    V.main(
        [
            "diff",
            "--baseline",
            str(tmp_path / "x"),
            "--fresh",
            base,
            "--allow-missing-baseline",
            "--json-out",
            str(none_js),
        ]
    )
    out = tmp_path / "body.md"
    assert V.main(["summarize", str(clean_js), str(none_js)]) == 0
    assert V.main(["summarize", str(clean_js), str(bad_js), str(none_js), "--out", str(out)]) == 1
    body = out.read_text()
    assert "vs baseline" in body and "A\\|ode" in body  # table cells escape the pipe
    assert V.main(["summarize", str(tmp_path / "unreadable.json")]) == 1


# --------------------------------------------------------------------------- #
# present_models.py (which corpus models the nightly can run)
# --------------------------------------------------------------------------- #
def test_present_models_lists_only_models_on_disk(tmp_path):
    import present_models as P

    (tmp_path / "models" / "M1").mkdir(parents=True)
    (tmp_path / "models" / "M1" / "M1.xml").write_text("<sbml/>")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {"model_id": "M2", "model": "models/M2/M2.xml"},
                    {"model_id": "M1", "model": "models/M1/M1.xml"},
                ]
            }
        )
    )
    assert P.present(manifest, root=tmp_path) == (["M1"], 2)
    (tmp_path / "models" / "M2").mkdir()
    (tmp_path / "models" / "M2" / "M2.xml").write_text("<sbml>" + "x" * 100 + "</sbml>")
    assert P.present(manifest, root=tmp_path, max_bytes=50) == (["M1"], 2)
    assert P.present(manifest, root=tmp_path, min_bytes=51) == (["M2"], 2)
    assert P.present(manifest, root=tmp_path, skip=frozenset({"M1"})) == (["M2"], 2)


# --------------------------------------------------------------------------- #
# sbml-suite specifics (GH #700)
# --------------------------------------------------------------------------- #
def sbml(cases: dict, suite="sbml_semantic") -> dict:
    return {
        "schema": V.SCHEMA,
        "suite": suite,
        "kind": "sbml-suite",
        "source": {},
        "n_cases": len(cases),
        "tally": V.tally(cases),
        "cases": cases,
    }


def srec(outcome, note=None):
    r = {"outcome": outcome, "subclass": None}
    if note is not None:
        r["note"] = note
    return r


SBML_BASE = sbml(
    {
        "00001": srec("pass"),
        "00002": srec("pass"),
        "00003": srec("load_fail", "delay"),
        "00004": srec("sim_fail", "fast"),
        "00005": srec("value_mismatch", "max_err=0.02"),
    }
)


def sbml_with(**changes) -> dict:
    cases = {k: dict(v) for k, v in SBML_BASE["cases"].items()}
    for key, value in changes.items():
        cid = key.lstrip("c")
        if value is None:
            cases.pop(cid)
        else:
            cases[cid] = value
    return sbml(cases)


def test_extract_sbml_suite_drops_the_machine_specific_suite_path(tmp_path):
    """A load failure names its file by absolute path; a baseline recorded on
    another machine would otherwise rewrite every such note."""
    err = (
        "Failed to load SBML file /home/runner/work/_temp/sbml-test-suite/cases/semantic/"
        "00039/00039-sbml-l3v2.xml: Model contains constructs bngsim cannot simulate"
    )
    path = tmp_path / "sbml.json"
    path.write_text(
        json.dumps({"cases": [{"case": "00039", "status": "load_fail", "error": err}]})
    )
    note = V.extract("sbml-suite", path, "sbml_semantic")["cases"]["00039"]["note"]
    assert note == (
        "Failed to load SBML file 00039/00039-sbml-l3v2.xml: "
        "Model contains constructs bngsim cannot simulate"
    )


def test_a_failing_case_the_harness_can_no_longer_score_is_a_new_crash():
    r = V.diff(SBML_BASE, sbml_with(c00003=srec("harness_error")))
    assert [x["case"] for x in r["new_crash"]] == ["00003"]


# --------------------------------------------------------------------------- #
# gate: the per-PR check is exact (GH #700)
# --------------------------------------------------------------------------- #
def _gate_classes(r) -> dict:
    names = [name for name, _ in V.GATE_FIX + V.GATE_RECORD]
    return {name: [x["case"] for x in r[name]] for name in names if r[name]}


def test_gate_passes_an_identical_run():
    r = V.gate(SBML_BASE, SBML_BASE)
    assert r["ok"] and r["n_fix"] == r["n_record"] == 0 and _gate_classes(r) == {}


def test_gate_ignores_a_reworded_failure():
    r = V.gate(SBML_BASE, sbml_with(c00003=srec("load_fail", "delay() is refused")))
    assert r["ok"]


@pytest.mark.parametrize(
    "after, cls",
    [
        (srec("value_mismatch"), "regressed"),
        (srec("load_fail"), "regressed"),
        (None, "regressed"),
        (srec("harness_error"), "harness_error"),
    ],
)
def test_gate_fails_a_good_case_that_is_not_good_now(after, cls):
    r = V.gate(SBML_BASE, sbml_with(c00001=after))
    assert not r["ok"] and r["n_fix"] == 1
    assert _gate_classes(r) == {cls: ["00001"]}


def test_gate_fails_a_refusal_that_became_a_wrong_answer():
    """The defect class a nightly ``diff`` lets through as churn between two
    failing states: a loud refusal traded for a silent wrong answer."""
    r = V.gate(SBML_BASE, sbml_with(c00003=srec("value_mismatch"), c00004=srec("var_missing")))
    assert _gate_classes(r) == {"now_wrong": ["00003", "00004"]}
    assert r["n_fix"] == 2
    assert V.diff(SBML_BASE, sbml_with(c00003=srec("value_mismatch")))["n_alerts"] == 0


def test_gate_fails_a_failing_case_the_harness_can_no_longer_score():
    r = V.gate(SBML_BASE, sbml_with(c00005=srec("harness_error")))
    assert _gate_classes(r) == {"harness_error": ["00005"]} and r["n_fix"] == 1


@pytest.mark.parametrize(
    "changes, cls",
    [
        ({"c00003": srec("pass")}, "improved"),
        ({"c00005": srec("load_fail")}, "changed"),
        ({"c00004": srec("load_fail")}, "changed"),
        ({"c00006": srec("pass")}, "new"),
        ({"c00003": None}, "dropped"),
    ],
)
def test_gate_fails_a_change_the_baseline_has_not_recorded(changes, cls):
    """Not a regression, but unrecorded: a later PR that undid it would pass."""
    r = V.gate(SBML_BASE, sbml_with(**changes))
    assert not r["ok"] and r["n_fix"] == 0 and r["n_record"] == 1
    assert list(_gate_classes(r)) == [cls]


def test_gate_fails_a_run_with_no_cases():
    r = V.gate(sbml({}), sbml({}))
    assert r["empty"] and not r["ok"]


def test_gate_refuses_a_kind_mismatch():
    with pytest.raises(ValueError, match="kind mismatch"):
        V.gate(SBML_BASE, BASE)


def test_gate_markdown_lists_every_change_by_class():
    r = V.gate(SBML_BASE, sbml_with(c00001=srec("sim_fail"), c00003=srec("pass")))
    md = V.render_gate_md(r)
    assert "1 to fix, 1 to record" in md
    assert "Regressed: good in the baseline, not now: 1" in md
    assert "Improved: failing in the baseline, good now: 1" in md
    assert "| 00001 | pass | sim_fail |" in md and "| 00003 | load_fail | pass |" in md


def test_cli_gate_exit_codes(tmp_path):
    b = _write(tmp_path, "b.json", SBML_BASE)
    same = _write(tmp_path, "same.json", SBML_BASE)
    better = _write(tmp_path, "better.json", sbml_with(c00003=srec("pass")))
    assert V.main(["gate", "--baseline", b, "--fresh", same]) == 0
    out = tmp_path / "g.json"
    assert V.main(["gate", "--baseline", b, "--fresh", better, "--json-out", str(out)]) == 1
    assert [x["case"] for x in json.loads(out.read_text())["improved"]] == ["00003"]


def test_gate_fails_a_new_case_the_harness_cannot_score():
    """A new case is normally only to record, but a harness_error recorded in a
    baseline would match itself on every later run."""
    r = V.gate(SBML_BASE, sbml_with(c00006=srec("harness_error")))
    assert _gate_classes(r) == {"harness_error": ["00006"]} and r["n_fix"] == 1
