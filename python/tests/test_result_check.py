"""``Result.check()`` (issue #1707): a run made again with one setting changed.

The README tells a user to confirm a result at a tighter tolerance or with a
second engine. ``check()`` makes that one call. These tests hold each verdict
to the rule it claims (the developer-side differential harness's, imported
from the same place), with real reruns where bngsim's own paths give the case
(a loose tolerance moves the answer) and with an injected disagreement where
they do not (bngsim's paths agree on every bundled example, so a ``diff`` from
the Jacobian or the grid has to be made to be seen).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import Result, Simulator, UsageError, _check, examples

KEYS = [
    "check_version",
    "ok",
    "method",
    "stochastic",
    "rtol",
    "atol",
    "drift",
    "bound",
    "checks",
    "n_pass",
    "n_diff",
    "next",
]
CHECK_KEYS = ["name", "status", "value", "worst_column", "changed", "note"]


def _run(name: str = "chain", fmt: str = "sbml", **kw) -> Result:
    opts = {"t_span": (0, 20), "n_points": 21, **kw}
    return Simulator(examples.load(name, format=fmt), method="ode").run(**opts)


def _by_name(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["checks"]}


def _perturbing(monkeypatch, when, factor):
    """Make Result.rerun scale the species of a rerun whose changes satisfy
    *when* by *factor*: a path that disagrees, made to order."""
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if when(changes):
            out = copy.copy(out)
            out._species = np.asarray(out._species, dtype=float) * factor
        return out

    monkeypatch.setattr(Result, "rerun", rerun)


# ─── The report's shape ──────────────────────────────────────────────────────


def test_the_default_report_has_its_keys_and_is_strict_json():
    r = _run()
    rep = r.check()
    assert list(rep)[: len(KEYS) - 3] == KEYS[:-3] and list(rep)[-3:] == KEYS[-3:]
    assert [c["name"] for c in rep["checks"]] == ["tolerance", "grid"]
    for c in rep["checks"]:
        assert list(c)[: len(CHECK_KEYS)] == CHECK_KEYS
    assert json.loads(json.dumps(rep, allow_nan=False)) == rep
    assert rep["ok"] is True and rep["n_pass"] == 2 and rep["n_diff"] == 0
    assert (rep["rtol"], rep["atol"], rep["method"]) == (1e-8, 1e-8, "ode")
    text = str(rep)
    assert text == rep.text() and text.startswith("OK: 2 passed, 0 moved the answer")
    assert len(text.splitlines()) == 1 + 2 + 1


def test_all_runs_every_deterministic_check_in_order():
    rep = _run().check("all")
    assert [c["name"] for c in rep["checks"]] == list(_check.DETERMINISTIC)
    statuses = _by_name(rep)
    for name in ("tolerance", "grid", "jacobian"):
        assert statuses[name]["status"] == "pass", statuses[name]
    assert statuses["jacobian"]["changed"] == {"jacobian": "fd"}
    assert statuses["grid"]["changed"] == {"n_points": 161}


def test_a_check_named_runs_alone_and_unknown_names_are_refused():
    r = _run()
    assert [c["name"] for c in r.check("jacobian")["checks"]] == ["jacobian"]
    assert [c["name"] for c in r.check(["grid", "tolerance"])["checks"]] == ["tolerance", "grid"]
    with pytest.raises(UsageError, match="does not know 'tighter'"):
        r.check(["tighter"])
    for bad in ({"budget": 0}, {"budget": -1.0}, {"replicates": 1}, {"replicates": True}):
        with pytest.raises(UsageError):
            r.check(**bad)


def test_a_stack_is_refused():
    sim = Simulator(examples.load("decay"), method="ode")
    stack = sim.run_batch(t_span=(0, 1), n_points=3, params=[{"k": 1.0}, {"k": 2.0}], squeeze=True)
    with pytest.raises(UsageError, match="stack"):
        stack.check()


def test_a_result_with_no_record_has_nothing_to_check():
    t = np.linspace(0, 1, 3)
    r = Result(_time=t, _species=np.ones((3, 1)), _observables=np.ones((3, 1)))
    rep = r.check()
    assert rep["ok"] is False
    assert {c["status"] for c in rep["checks"]} == {"unsupported"}
    assert "No check could run" in rep["next"]


# ─── The rule, against the arrays ────────────────────────────────────────────


def test_gap_floor_and_bound():
    ref = np.array([[0.0, 100.0], [50.0, 10.0]])
    assert _check.gap(ref, ref, 0.0) == 0.0
    x = ref + np.array([[0.0, 2.0], [0.0, 0.0]])
    assert _check.gap(x, ref, 0.0) == pytest.approx(2.0 / 100.0)
    assert _check.gap(x, ref, 1.0) == pytest.approx(1.0 / 100.0)
    assert _check.gap(x, ref, 5.0) == 0.0
    assert _check.gap(x[:1], ref, 0.0) == math.inf
    y = ref.copy()
    y[0, 0] = np.nan
    assert _check.gap(y, ref, 0.0) == math.inf
    # The floor is 1e3·atol, but at most 1% of the peak.
    assert _check.floor(1e-8, ref) == pytest.approx(1e-5)
    assert _check.floor(1e-2, ref) == pytest.approx(1.0)
    assert _check.floor(1e-2, np.zeros((2, 2))) == pytest.approx(10.0)
    # The bound is 1e3·rtol (at most 1%) or ten times the drift.
    assert _check.bound(1e-8, 0.0) == pytest.approx(1e-5)
    assert _check.bound(1e-3, 0.0) == pytest.approx(1e-2)
    assert _check.bound(1e-8, 1e-4) == pytest.approx(1e-3)


def test_the_differential_harness_judges_by_the_same_rule():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "parity_checks" / "differential"))
    try:
        import _differential as D
    finally:
        sys.path.pop(0)
    assert D.gap is _check.gap
    for name in ("PATH_RTOL_FACTOR", "DRIFT_FACTOR", "UNJUDGEABLE_DRIFT", "GRID_FACTOR"):
        assert getattr(D, name) == getattr(_check, name), name


def test_a_path_is_judged_against_the_bound(monkeypatch):
    # The run's own drift at rtol 1e-8 is under the floor, so the bound is
    # 1e3·rtol = 1e-5 of the peak (100): a path 0.5e-5 off passes, 2e-5 fails.
    r = _run()
    for factor, status in ((1 + 0.5e-5, "pass"), (1 + 2e-5, "diff")):
        with monkeypatch.context() as m:
            _perturbing(m, lambda c: c.get("jacobian") == "fd", factor)
            rep = r.check("jacobian")
        (c,) = rep["checks"]
        assert c["status"] == status, (factor, c)
        assert c["bound"] == pytest.approx(1e-5)
        assert rep["ok"] is (status == "pass")
    assert c["worst_column"] in ("A", "B", "C")
    assert c["value"] == pytest.approx(2e-5 - _check.floor(1e-8, r.species) / 100, rel=1e-3)
    assert "Two bngsim paths disagree (jacobian)" in rep["next"]
    assert "NOT OK" in str(rep)


def test_a_path_two_percent_off_fails_even_at_a_loose_tolerance(monkeypatch):
    # At rtol 1e-3 the 1e3·rtol term is the whole peak; capped at 1%, a path
    # 2% off still fails.
    r = _run(rtol=1e-3, atol=1e-6)
    _perturbing(monkeypatch, lambda c: c.get("jacobian") == "fd", 1.02)
    (c,) = r.check("jacobian")["checks"]
    assert c["status"] == "diff" and c["bound"] == pytest.approx(1e-2, rel=0.2)


def test_a_run_not_converged_at_its_tolerance_judges_no_path(monkeypatch):
    # The tenfold-tighter run 1% off: the run moves past UNJUDGEABLE_DRIFT.
    r = _run()
    _perturbing(monkeypatch, lambda c: c.get("rtol") == pytest.approx(1e-9), 1.01)
    rep = r.check(["tolerance", "grid"])
    tol, grid = rep["checks"]
    assert tol["status"] == "diff" and grid["status"] == "no_base"
    assert rep["ok"] is False
    assert "moves when the tolerance is tightened" in rep["next"]


def test_a_rerun_that_raises_is_an_exception_not_a_crash(monkeypatch):
    real = Result.rerun

    def rerun(self, **changes):
        if changes.get("jacobian") == "fd":
            raise bngsim.SimulationError("CVODE failed: too much work")
        return real(self, **changes)

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = _run().check(["tolerance", "jacobian"])
    assert _by_name(rep)["jacobian"]["status"] == "exception"
    assert "SimulationError: CVODE failed" in _by_name(rep)["jacobian"]["note"]
    assert rep["ok"] is False and "raised" in rep["next"]


# ─── The tolerance check, on real reruns ─────────────────────────────────────


def test_a_loose_tolerance_moves_the_answer():
    # chain at rtol = atol = 1e-2: B moves by 0.4% of the peak at rtol/10.
    r = _run(rtol=1e-2, atol=1e-2)
    rep = r.check("tolerance")
    (c,) = rep["checks"]
    tight = r.rerun(rtol=1e-3, atol=1e-3)
    expected = _check.gap(np.asarray(tight.species), np.asarray(r.species), 1e-2)
    assert c["value"] == pytest.approx(expected, rel=1e-12)
    assert c["status"] == "diff" and c["bound"] == _check.UNJUDGEABLE_DRIFT
    assert c["worst_column"] == "B"
    assert c["changed"] == {"rtol": 1e-3, "atol": 1e-3}
    assert c["value_100"] is not None
    # The default tolerances do not move it.
    assert _run().check("tolerance")["checks"][0]["status"] == "pass"


def test_a_vector_atol_is_tightened_entry_by_entry(monkeypatch):
    seen = []
    real = Result.rerun

    def rerun(self, **changes):
        seen.append(changes)
        return real(self, **changes)

    monkeypatch.setattr(Result, "rerun", rerun)
    _run(atol=[1e-6, 1e-7, 1e-8]).check("tolerance")
    assert seen[0]["atol"] == pytest.approx([1e-7, 1e-8, 1e-9])
    assert seen[0]["rtol"] == pytest.approx(1e-9)


def test_the_grid_check_reads_a_sample_times_run_at_its_own_times():
    r = Simulator(examples.load("chain", format="sbml"), method="ode").run(
        sample_times=[0, 1, 3, 7]
    )
    (c,) = r.check("grid")["checks"]
    assert c["status"] == "pass" and c["changed"] == {"sample_times": "8x the points"}


def test_the_backend_check_runs_the_other_way():
    r = Simulator(examples.load("chain", format="sbml"), method="ode", codegen=False).run(
        t_span=(0, 5), n_points=6
    )
    (c,) = r.check("backend")["checks"]
    assert c["status"] in ("pass", "skip"), c
    if c["status"] == "pass":
        assert c["changed"] == {"codegen": True}


# ─── The second engine ───────────────────────────────────────────────────────


def test_the_second_engine_agrees_where_libroadrunner_is_installed():
    pytest.importorskip("roadrunner")
    for name in ("chain", "bolus", "schlogl"):
        (c,) = _run(name).check("second_engine")["checks"]
        assert c["status"] == "pass", (name, c)
        assert c["changed"]["engine"].startswith("libRoadRunner")


def test_the_second_engine_says_why_it_did_not_run(monkeypatch):
    (c,) = _run(fmt="net").check("second_engine")["checks"]
    assert c["status"] == "unsupported" and "'net'" in c["note"]
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "roadrunner", None)
        (c,) = _run().check("second_engine")["checks"]
    assert c["status"] == "skip" and "pip install libroadrunner" in c["note"]
    pytest.importorskip("roadrunner")
    sim = Simulator(examples.load("chain", format="sbml"), method="ode")
    sim.run(t_span=(0, 1), n_points=3)
    with pytest.warns(bngsim.LiveStartWarning):
        live = sim.run(t_span=(0, 1), n_points=3)
    (c,) = live.check("second_engine")["checks"]
    assert c["status"] == "skip" and "initial state" in c["note"]
    m2 = examples.load("chain", format="sbml")
    m2.set_param("k1", 2.0)
    (c,) = (
        Simulator(m2, method="ode").run(t_span=(0, 1), n_points=3).check("second_engine")["checks"]
    )
    assert c["status"] == "skip" and "k1" in c["note"]


# ─── Stochastic runs ─────────────────────────────────────────────────────────


def test_replicates_give_the_mean_and_its_standard_error():
    sim = Simulator(examples.load("birth_death"), method="ssa")
    r = sim.run(t_span=(0, 10), n_points=11, seed=4)
    rep = r.check()
    (c,) = rep["checks"]
    assert c["name"] == "replicates" and c["status"] == "pass" and c["n_replicates"] == 10
    (col,) = c["columns"]
    lasts = [float(r.rerun(seed=5 + i).observables["A"][-1]) for i in range(10)]
    assert col["mean_last"] == pytest.approx(np.mean(lasts))
    assert col["se_last"] == pytest.approx(np.std(lasts, ddof=1) / math.sqrt(10))
    assert col["run_last"] == r.observables["A"][-1]
    assert col["z"] == pytest.approx((col["run_last"] - col["mean_last"]) / col["se_last"])
    exact = examples.exact("birth_death", np.array([10.0]))["A"][0]
    assert abs(col["mean_last"] - exact) < 5 * col["se_last"] * math.sqrt(10)
    # The deterministic checks do not apply.
    rep = r.check("all")
    assert [x["name"] for x in rep["checks"]] == ["replicates"]
    (c,) = r.check("tolerance")["checks"]
    assert c["status"] == "unsupported" and "stochastic" in c["note"]


# ─── What it costs, and what it leaves ───────────────────────────────────────


def test_the_session_model_is_left_alone():
    m = examples.load("chain", format="sbml")
    sim = Simulator(m, method="ode")
    r = sim.run(t_span=(0, 5), n_points=6)
    state, params = m.get_state().copy(), {p: m.get_param(p) for p in m.param_names}
    r.check("all")
    assert np.array_equal(m.get_state(), state)
    assert {p: m.get_param(p) for p in m.param_names} == params


def test_a_budget_reports_what_did_not_fit():
    r = _run()
    rep = r.check("all", budget=1e-9)
    assert [c["name"] for c in rep["checks"]] == list(_check.DETERMINISTIC)
    timed = [c for c in rep["checks"] if c["status"] == "timeout"]
    assert timed and all("budget" in c["note"] for c in timed)
    assert rep["ok"] is False
