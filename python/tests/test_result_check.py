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
    "converged",
    "n_not_resolved",
    "not_resolved",
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


def test_differences_moves_and_allowance():
    ref = np.array([[0.0, 100.0], [50.0, 10.0]])
    d, peak = _check.differences(ref, ref)
    assert d.tolist() == [0.0, 0.0] and peak.tolist() == [50.0, 100.0]
    x = ref + np.array([[0.0, 2.0], [1.0, 0.0]])
    d, peak = _check.differences(x, ref)
    assert d.tolist() == [1.0, 2.0]
    # Each column over its own peak, past its own atol.
    assert _check.moves(d, peak, np.array([0.0, 0.0])).tolist() == [1.0 / 50, 2.0 / 100]
    assert _check.moves(d, peak, np.array([0.5, 2.0])).tolist() == [0.5 / 50, 0.0]
    # A column within its atol has nothing that counts.
    assert _check.moves(np.array([1.0]), np.array([1e-9]), np.array([1e-8])).tolist() == [0.0]
    # Another shape, or a NaN on one side only, is inf.
    assert np.isinf(_check.differences(x[:1], ref)[0]).all()
    y = ref.copy()
    y[0, 0] = np.nan
    assert _check.differences(y, ref)[0].tolist()[0] == math.inf
    assert _check.differences(y, ref)[0].tolist()[1] == 0.0
    # The allowance: 1e3·(atol + rtol·peak), at most 1% of the peak (or atol),
    # or ten times the column's own move.
    peak = np.array([100.0, 100.0, 0.0])
    atol = np.array([1e-8, 1e-2, 1e-8])
    a = _check.allowance(peak, atol, 1e-8, np.zeros(3))
    # ... and never under 100 atol: the third column, at 0, is allowed 1e-6.
    assert a.tolist() == pytest.approx([1e3 * (1e-8 + 1e-6), 1.0, 1e-6])
    a = _check.allowance(peak, atol, 1e-8, np.array([1e-3, 0, 0]))
    assert a[0] == pytest.approx(1e-2)


def test_the_constants_are_the_differential_harnesss():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "parity_checks" / "differential"))
    try:
        import _differential as D
    finally:
        sys.path.pop(0)
    assert D.PATH_RTOL_FACTOR == _check.PATH_RTOL_FACTOR
    assert D.DRIFT_FACTOR == _check.DRIFT_FACTOR
    assert D.UNJUDGEABLE_DRIFT == _check.TOLERANCE_MOVE
    assert D.GRID_FACTOR == _check.GRID_FACTOR


def test_a_path_is_judged_against_the_allowance(monkeypatch):
    # At rtol = atol = 1e-8, A (peak 100) is allowed 1e3·(1e-8 + 1e-6) =
    # 1.01e-3: a path 0.5e-5 of its peak off passes, one 2e-5 off fails.
    r = _run()
    for factor, status in ((1 + 0.5e-5, "pass"), (1 + 2e-5, "diff")):
        with monkeypatch.context() as m:
            _perturbing(m, lambda c: c.get("jacobian") == "fd", factor)
            rep = r.check("jacobian")
        (c,) = rep["checks"]
        assert c["status"] == status, (factor, c)
        assert (c["margin"] <= 1.0) is (status == "pass")
        assert rep["ok"] is (status == "pass")
    assert c["worst_column"] in ("A", "B", "C")
    assert c["value"] == pytest.approx(2e-5, rel=1e-6)
    assert "Two bngsim paths disagree (jacobian)" in rep["next"]
    assert "NOT OK" in str(rep)


def test_a_path_two_percent_off_fails_even_at_a_loose_tolerance(monkeypatch):
    # At rtol = atol = 1e-3 the 1e3·(atol + rtol·peak) term is the whole peak;
    # capped at 1%, a path 1.9% off still fails, and the atol is not forgiven
    # on top of it.
    r = _run(rtol=1e-3, atol=1e-3)
    _perturbing(monkeypatch, lambda c: c.get("jacobian") == "fd", 1.019)
    (c,) = r.check("jacobian")["checks"]
    assert c["status"] == "diff", c


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
    # chain at rtol = atol = 1e-2: B moves by about 0.9% of its peak at rtol/10.
    r = _run(rtol=1e-2, atol=1e-2)
    rep = r.check("tolerance")
    (c,) = rep["checks"]
    tight = r.rerun(rtol=1e-3, atol=1e-3)
    d, peak = _check.differences(np.asarray(tight.species), np.asarray(r.species))
    expected = _check.moves(d, peak, np.full(3, 1e-2))
    assert c["value"] == pytest.approx(expected.max(), rel=1e-12)
    assert c["status"] == "diff" and c["bound"] == _check.TOLERANCE_MOVE
    assert c["worst_column"] == r.species_names[int(np.argmax(expected))]
    assert c["changed"] == {"rtol": 1e-3, "atol": 1e-3}
    assert c["value_100"] is not None
    assert rep["drift"] == pytest.approx(expected.max())
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


# ─── The review's cases ──────────────────────────────────────────────────────


def test_a_small_species_is_judged_on_its_own_scale(monkeypatch):
    # A at 1000 beside X at 1e-2: a path that has X 100x off must fail, though
    # X is 1e-5 of the largest value.
    m = bngsim.Model.from_antimony_string("A -> B; k*A; X -> Y; k*X; A = 1000; X = 0.01; k = 0.5")
    r = Simulator(m, method="ode").run(t_span=(0, 5), n_points=11, rtol=1e-6, atol=1e-12)
    names = r.species_names
    jx = names.index("X")
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("jacobian") == "fd":
            out = copy.copy(out)
            sp = np.asarray(out._species, dtype=float).copy()
            sp[:, jx] *= 100
            out._species = sp
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    (c,) = r.check("jacobian")["checks"]
    assert c["status"] == "diff" and c["worst_column"] == "X", c


def test_a_run_unconverged_in_small_units_fails_the_tolerance_check():
    # Robertson's problem at nanomolar scale, at rtol 0.1: the run is far off,
    # and its species sit under the scalar atol of 1e-8. Judged at the atol the
    # run applied to each species, the tolerance check fails.
    m = bngsim.Model.from_antimony_string(
        "y1 -> y2; 0.04*y1; y2 + y2 -> y3 + y2; 3e16*y2*y2; y2 + y3 -> y1 + y3; 1e13*y2*y3;"
        " y1 = 1e-9; y2 = 0; y3 = 0"
    )
    with pytest.warns(bngsim.AtolScaleWarning):
        r = Simulator(m, method="ode").run(t_span=(0, 40), n_points=41, rtol=0.1)
    rep = r.check()
    assert _by_name(rep)["tolerance"]["status"] == "diff", rep
    assert rep["ok"] is False


def test_a_steady_state_run_compares_where_it_stopped():
    r = Simulator(examples.load("chain", format="sbml"), method="ode").run(
        t_span=(0, 2000), n_points=21, steady_state=True
    )
    rep = r.check("all")
    by = _by_name(rep)
    assert by["grid"]["status"] == "unsupported"
    for name in ("tolerance", "jacobian"):
        assert by[name]["status"] == "pass", by[name]
    assert rep["ok"] is True


def test_a_run_on_a_difference_quotient_jacobian_skips_that_check():
    r = Simulator(examples.load("chain", format="sbml"), method="ode", jacobian="fd").run(
        t_span=(0, 20), n_points=21
    )
    assert r.record["backend"]["jacobian"] == "difference_quotient"
    (c,) = r.check("jacobian")["checks"]
    assert c["status"] == "skip"
    assert r.check(["tolerance", "jacobian"])["ok"] is True


def test_the_second_engine_skips_a_run_from_a_saved_state():
    pytest.importorskip("roadrunner")
    m = examples.load("chain", format="sbml")
    m.set_concentration("A", 3.0)
    m.save_concentrations()
    r = Simulator(m, method="ode").run(t_span=(0, 5), n_points=6)
    (c,) = r.check("second_engine")["checks"]
    assert c["status"] == "skip" and "file's initial state" in c["note"], c


def test_a_partial_budget_is_not_ok(monkeypatch):
    # The tolerance check finishes; the budget runs out before the grid.
    real = _check._Context.check_grid

    def slow(self):
        self.deadline = 0.0  # spent
        return real(self)

    monkeypatch.setattr(_check._Context, "check_grid", slow)
    rep = _run().check(["tolerance", "grid"], budget=60)
    assert [c["status"] for c in rep["checks"]] == ["pass", "timeout"]
    assert rep["ok"] is False and "budget ran out before grid" in rep["next"]
    with pytest.raises(UsageError):
        _run().check(budget=True)


def test_replicate_seeds_wrap_and_are_listed():
    sim = Simulator(examples.load("birth_death"), method="ssa")
    r = sim.run(t_span=(0, 1), n_points=3, seed=2**31 - 3)
    (c,) = r.check(replicates=3)["checks"]
    assert c["seeds"] == [2**31 - 2, 0, 1]
    assert "2147483646, ..., 1" in c["note"]


def test_a_steady_state_run_is_compared_by_its_last_state_alone(monkeypatch):
    # A rerun that reached the steady state in fewer output rows, at the same
    # state, agrees.
    r = Simulator(examples.load("chain", format="sbml"), method="ode").run(
        t_span=(0, 2000), n_points=21, steady_state=True
    )
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("jacobian") == "fd":
            out = copy.copy(out)
            out._species = np.asarray(out._species, dtype=float)[-1:]
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    (c,) = r.check("jacobian")["checks"]
    assert c["status"] == "pass", c


def test_a_runs_own_timeout_still_bounds_its_reruns(monkeypatch):
    seen = []
    real = Result.rerun

    def rerun(self, **changes):
        seen.append(changes.get("timeout"))
        return real(self, **changes)

    monkeypatch.setattr(Result, "rerun", rerun)
    _run(timeout=5.0).check("tolerance", budget=600)
    assert seen and all(t is not None and t <= 5.0 for t in seen)


def test_a_species_not_converged_is_not_judged(monkeypatch):
    # B moves 1% at the tighter tolerance (made so): it is not converged at
    # the run's own, so a path 20% off in B (past even 10x its own move) is not
    # held against the run;
    # the report says one column was not judged.
    r = _run()
    jb = r.species_names.index("B")
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("rtol") == pytest.approx(1e-9) or changes.get("jacobian") == "fd":
            out = copy.copy(out)
            sp = np.asarray(out._species, dtype=float).copy()
            sp[:, jb] *= 1.01 if "rtol" in changes else 1.2
            out._species = sp
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = r.check(["tolerance", "jacobian"])
    tol, jac = rep["checks"]
    assert tol["status"] == "diff" and tol["worst_column"] == "B"
    assert jac["status"] == "pass" and jac["n_not_judged"] == 1, jac


def _trace_model(eps: float) -> bngsim.Model:
    # A decays into B; a slow leak at eps*A makes I, a trace species degraded
    # fast, which feeds a trace Z -> W chain.
    return bngsim.Model.from_antimony_string(
        "J1: A -> B; 0.1*A; J2: A -> A + I; eps*A; J3: I -> ; 50*I; J4: I -> Z; 1*I;"
        f" J5: Z -> W; 1*Z; J6: W -> ; 1*W; A = 100; eps = {eps}"
    )


@pytest.mark.parametrize("eps, rtol", [(1e-6, 1e-8), (1e-6, 1e-4), (1e-9, 1e-8), (1e-15, 1e-8)])
def test_trace_species_raise_no_spurious_finding(eps, rtol):
    # The review's case: trace species at tens to hundreds of atol, whose
    # solver noise read as a path disagreement or an unconverged run.
    r = Simulator(_trace_model(eps), method="ode").run(t_span=(0, 30), n_points=31, rtol=rtol)
    rep = r.check(["tolerance", "grid", "jacobian", "backend"])
    assert rep["ok"] is True, str(rep)
    assert rep["converged"] is True


def test_a_trace_species_is_counted_not_judged(monkeypatch):
    # I peaks near 2e-7, within 100 times its atol of 1e-8: not resolved by
    # the run. A path 10x off in I alone passes, and the report names I.
    r = Simulator(_trace_model(1e-7), method="ode").run(t_span=(0, 30), n_points=31)
    ji = r.species_names.index("I")
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("jacobian") == "fd":
            out = copy.copy(out)
            sp = np.asarray(out._species, dtype=float).copy()
            sp[:, ji] *= 10
            out._species = sp
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = r.check("jacobian")
    (c,) = rep["checks"]
    assert c["status"] == "pass" and c["n_trace"] >= 1 and "atol of zero" in c["note"]
    assert "I" in rep["not_resolved"] and "I" in rep["next"]
    n = rep["n_not_resolved"]
    assert str(rep).startswith(f"OK: 1 passed, 0 moved the answer, {n} species not resolved")


def test_an_unconverged_run_is_not_ok_whichever_checks_ran(monkeypatch):
    # Only the jacobian check asked for, on a run whose B moves 1% when the
    # tolerance is tightened: the path passes on the other columns, but the
    # report is not ok, and says why.
    r = _run()
    jb = r.species_names.index("B")
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("rtol") == pytest.approx(1e-9):
            out = copy.copy(out)
            sp = np.asarray(out._species, dtype=float).copy()
            sp[:, jb] *= 1.01
            out._species = sp
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = r.check("jacobian")
    assert rep["checks"][0]["status"] == "pass"
    assert rep["converged"] is False and rep["ok"] is False
    assert "not converged" in rep["next"]


def test_a_trace_species_moving_at_a_tighter_tolerance_is_not_a_finding(monkeypatch):
    # I is within 100 atol of zero: doubled in the tighter run, it is still
    # not resolved, and the tolerance check passes, naming it.
    r = Simulator(_trace_model(1e-7), method="ode").run(t_span=(0, 30), n_points=31)
    ji = r.species_names.index("I")
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if "rtol" in changes:
            out = copy.copy(out)
            sp = np.asarray(out._species, dtype=float).copy()
            sp[:, ji] *= 2
            out._species = sp
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = r.check("tolerance")
    (c,) = rep["checks"]
    assert c["status"] == "pass" and c["n_trace"] >= 1, c
    assert rep["converged"] is True and rep["ok"] is True


def test_a_small_species_the_run_resolves_is_judged_beside_a_large_pool(monkeypatch):
    # The review's case: nanomolar Robertson beside an unused pool at 1e-3. The
    # run holds y2 and y3 to their own tiny atols, so they are judged, however
    # small beside the pool: a path 100x off in y3 fails.
    m = bngsim.Model.from_antimony_string(
        "y1 -> y2; 0.04*y1; y2 + y2 -> y3 + y2; 3e16*y2*y2; y2 + y3 -> y1 + y3; 1e13*y2*y3;"
        " y1 = 1e-9; y2 = 0; y3 = 0; species ATP; ATP = 1e-3"
    )
    r = Simulator(m, method="ode").run(t_span=(0, 40), n_points=41)
    j3 = r.species_names.index("y3")
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("jacobian") == "fd":
            out = copy.copy(out)
            sp = np.asarray(out._species, dtype=float).copy()
            sp[:, j3] *= 100
            out._species = sp
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = r.check("jacobian")
    assert "y3" not in rep["not_resolved"]
    assert rep["checks"][0]["status"] == "diff" and rep["checks"][0]["worst_column"] == "y3"


def test_a_species_two_hundred_atol_from_zero_is_judged(monkeypatch):
    # I peaks near 2e-6, 200 times its atol of 1e-8: resolved, so a path 10x
    # off in I alone fails.
    r = Simulator(_trace_model(1e-6), method="ode").run(t_span=(0, 30), n_points=31)
    ji = r.species_names.index("I")
    assert 100 * 1e-8 < np.asarray(r.species)[:, ji].max() < 1e3 * 1e-8
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("jacobian") == "fd":
            out = copy.copy(out)
            sp = np.asarray(out._species, dtype=float).copy()
            sp[:, ji] *= 10
            out._species = sp
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = r.check("jacobian")
    assert "I" not in rep["not_resolved"]
    assert rep["checks"][0]["status"] == "diff" and rep["checks"][0]["worst_column"] == "I"
