"""The discontinuity suite's judge, sweep driver and nightly runner
(issue #1912). Every rule is shown to fail on the defect it is there to see."""

from __future__ import annotations

import dataclasses
import json

import _closed_forms as CF
import _judge as J
import _sweep as S
import bngsim
import discontinuity_run as R
import numpy as np
import pytest
import verdicts
from _case import Aligned, Uniform, point
from _cases import BY_NAME
from _reference import Band, Invariant, NoReference

SPAN = (0.0, 30.0)


def _window(times=None, sens=("kx", "t_on", "t_off")):
    case = BY_NAME["time_window"]
    values = case.values()
    t = np.asarray(
        times if times is not None else Aligned(31).request(SPAN, (7.3, 12.1))["sample_times"]
    )
    return (
        case,
        values,
        CF.window_decay("X()", "Y()", 10.0).solve(values, SPAN, t, case.observe, sens),
    )


def _judge(sol, x, S_, values, rtol=1e-10, sens=("kx", "t_on", "t_off")):
    return J.judge(
        sol,
        x,
        S_,
        rtol=rtol,
        atol=rtol * 1e-2,
        observe=("X()", "Y()"),
        sens=sens,
        params=values,
        k_state=1e3,
        k_sens=1e3,
    )


def test_the_reference_itself_passes_with_nothing_to_spare():
    _, values, sol = _window()
    v = _judge(sol, sol.species.lo, sol.sens.lo, values)
    assert v.outcome == "PASS" and v.ratio == 0.0 and v.sides == {"right": 2}


@pytest.mark.parametrize(("times", "outcome"), [(2.0, "DIFF"), (0.5, "PASS")])
def test_a_column_off_by_its_allowance_fails_and_one_inside_passes(times, outcome):
    _, values, sol = _window()
    S_ = sol.sens.lo.copy()
    i = int(np.flatnonzero(sol.times == 20.0)[0])
    peak = np.abs(sol.sens.lo[:, 0, 1]).max()
    tol = 1e-12 / values["t_on"] + 1e-10 * peak
    S_[i, 0, 1] += times * 1e3 * tol
    v = _judge(sol, sol.species.lo, S_, values)
    assert v.outcome == outcome
    if outcome == "DIFF":
        assert "dX()/dt_on at t=20" in v.note


def test_a_crossing_row_takes_either_limit_but_not_one_between():
    _, values, sol = _window()
    v = _judge(sol, sol.species_left.lo, sol.sens_left.lo, values)
    assert v.outcome == "PASS" and v.sides == {"left": 2}
    S_ = sol.sens.lo.copy()
    i = int(np.flatnonzero(sol.at_crossing)[0])
    S_[i] = 0.5 * (sol.sens.lo[i] + sol.sens_left.lo[i])
    v = _judge(sol, sol.species.lo, S_, values)
    assert v.outcome == "DIFF" and "(at a crossing)" in v.note


def test_a_column_the_run_cannot_resolve_is_named_not_judged():
    case = BY_NAME["clock_step"]
    values = case.values()
    t = np.linspace(*SPAN, 31)
    sol = CF.clock_exposure("step").solve(values, SPAN, t, case.observe, ("kx",))
    S_ = sol.sens.lo.copy()
    S_[:, 0, 0] = 1e-20  # dT()/dkx is 0; the run's rounding is not a finding
    v = J.judge(
        sol,
        sol.species.lo,
        S_,
        rtol=1e-10,
        atol=1e-12,
        observe=case.observe,
        sens=("kx",),
        params=values,
        k_state=1e3,
        k_sens=1e3,
    )
    assert v.outcome == "PASS" and v.unresolved == ["dT()/dkx"]


def test_an_error_at_the_reference_floor_is_none_and_one_past_it_counts():
    _, values, sol = _window(sens=())
    x = sol.species.lo * (1 + 5 * sol.floor)
    assert _judge(sol, x, None, values, rtol=1e-12, sens=()).state_ratio == 0.0
    x = sol.species.lo * (1 + 1e-12)
    assert _judge(sol, x, None, values, rtol=1e-12, sens=()).state_ratio > 0.0


def test_a_band_measures_only_what_lies_outside_it():
    b = Band(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))
    np.testing.assert_array_equal(b.distance([1.5, 0.5, 2.25]), [0.0, 0.5, 0.25])
    assert np.isinf(b.distance([np.nan, 1.0, 1.0])[0])


@pytest.mark.parametrize(
    ("ratios", "outcome"),
    [
        ([1.0, 1.2, 1.5, 2.0], "PASS"),  # the error tracks the tolerance
        ([0.01, 1.0, 100.0, 1e4], "DIFF"),  # stuck: #1227's signature
        ([0.0, 0.0, 3.0, 9.0], "PASS"),  # grew, but within CONVERGE_FLOOR
        ([0.0, 0.0, 3.0, 12.0], "DIFF"),
        ([1.0, None, None, None], "SKIP"),
    ],
)
def test_the_convergence_rule(ratios, outcome):
    assert (
        J.converge(list(zip((1e-6, 1e-8, 1e-10, 1e-12), ratios, strict=False))).outcome == outcome
    )


# ── Through bngsim ──────────────────────────────────────────────────────────


def test_a_point_is_judged_against_its_reference_and_a_wrong_reference_fails_it():
    case = BY_NAME["time_window"]
    p = point(1e-10, Aligned(31), ("kx", "t_on", "t_off"))
    row = S.run_point(case, p)
    assert row.outcome == "PASS", row.note
    assert row.work["n_steps"] > 0 and row.backend and row.extra["sides"]
    off = dataclasses.replace(case, reference=CF.window_decay("X()", "Y()", 10.0 * (1 + 1e-6)))
    row = S.run_point(off, p)
    assert row.outcome == "DIFF" and "species" in row.note


def test_a_known_point_keeps_its_true_outcome_and_names_its_issue():
    case = BY_NAME["clock_step"]
    key = sorted(case.known)[0]
    p = next(q for q in case.points() if case.key(q) == key)
    row = S.run_point(case, p)
    assert row.extra["known"] == 1799 and row.note.startswith("known #1799")


def test_a_run_must_report_the_sample_times_it_was_given_to_the_bit():
    ask = {"sample_times": [0.0, 25.0, 30.0]}
    assert S.grid_mismatch(ask, [0.0, 25.0, 30.0]) is None
    assert S.grid_mismatch({"n_points": 3}, [0.0, 1.0, 2.0]) is None
    note = S.grid_mismatch(ask, [0.0, np.nextafter(25.0, 0), 30.0])
    assert note == "the row asked for at t=25.0 came back at t=24.999999999999996 (1 rows moved)"
    assert "2 rows reported where 3" in S.grid_mismatch(ask, [0.0, 30.0])


def test_a_failure_is_classified_by_what_bngsim_raised():
    assert S._failure(bngsim.SimulationTimeout("slow", timeout=1.0, elapsed=2.0)) == (
        "TIMEOUT",
        None,
    )
    assert S._failure(bngsim.SensitivityUnsupportedError("no"))[0] == "UNSUPPORTED"
    assert S._failure(ValueError("x")) == ("EXCEPTION", "harness:ValueError")


def test_no_reference_checks_its_invariants_and_its_expected_refusal():
    case = BY_NAME["time_window"]
    total = Invariant("X + Y", lambda t, sp, p: (sp["X()"] + sp["Y()"] - 10.0, 10.0))
    ok = dataclasses.replace(case, reference=NoReference("none yet", (total,)))
    assert S.run_point(ok, point(1e-10, Uniform(31))).outcome == "PASS"
    leak = Invariant("X + Y", lambda t, sp, p: (sp["X()"] + sp["Y()"] - 10.001, 10.0))
    bad = dataclasses.replace(case, reference=NoReference("none yet", (leak,)))
    assert S.run_point(bad, point(1e-10, Uniform(31))).outcome == "DIFF"
    refuse = dataclasses.replace(
        case, reference=NoReference("none yet", expect_refusal=bngsim.SimulationError)
    )
    row = S.run_point(refuse, point(1e-10, Uniform(31)))
    assert row.outcome == "DIFF" and "was expected" in row.note


# ── The nightly runner ──────────────────────────────────────────────────────


def test_a_job_that_fails_reports_every_row_it_owed():
    case = BY_NAME["decay_bend"]
    row = case.rows()[1]
    spec = {"keys": R.job_keys(case, row), "versions": {}, "timeout": 5.0}
    n = len(case.points(row)) + len(case.groups(case.points(row)))
    for res, outcome in (({"status": "timeout"}, "TIMEOUT"), ({"status": "dead"}, "EXCEPTION")):
        rows = R.to_results(spec, res)
        assert len(rows) == n and {r["outcome"] for r in rows} == {outcome}
    partial = R.to_results(spec, {"status": "ok", "results": []})
    assert len(partial) == n and {r["subclass"] for r in partial} == {"harness"}


def test_the_runners_report_reads_as_a_verdict_file_holding_every_key(tmp_path):
    out = tmp_path / "report.json"
    assert R.main(["--cases", "time_window", "--workers", "2", "--out", str(out)]) == 0
    v = verdicts.extract("core", [out], "discontinuity")
    case = BY_NAME["time_window"]
    assert set(v["cases"]) == case.all_keys()
    assert v["tally"]["PASS"] == len(case.all_keys()), json.dumps(v["tally"])
    some = next(iter(v["cases"].values()))
    assert some.get("work", {}).get("n_steps", 0) > 0 or some["outcome"] == "PASS"


def test_a_spec_that_says_another_model_than_its_text_is_caught():
    case = BY_NAME["time_window"]
    spec = case.spec
    fast = dataclasses.replace(
        spec, rhs=lambda x, s, t, p: {k: 2 * v for k, v in spec.rhs(x, s, t, p).items()}
    )
    problems = S.check_spec(dataclasses.replace(case, spec=fast))
    assert problems and all("flows" in p for p in problems)
    moved = dataclasses.replace(spec, x0=lambda p: {"X()": 11, "Y()": 0})
    problems = S.check_spec(dataclasses.replace(case, spec=moved))
    assert any("starts at" in p for p in problems)


@pytest.mark.parametrize(("times", "outcome"), [(2.0, "DIFF"), (0.5, "PASS")])
def test_a_species_off_by_its_allowance_fails_a_plain_point(times, outcome):
    _, values, sol = _window(sens=())
    x = sol.species.lo.copy()
    i = int(np.flatnonzero(sol.times == 20.0)[0])
    peak = np.abs(sol.species.lo[:, 1]).max()
    x[i, 1] += times * 1e3 * (1e-12 + 1e-10 * peak)
    v = _judge(sol, x, None, values, sens=())
    assert v.outcome == outcome and v.sens_ratio is None
