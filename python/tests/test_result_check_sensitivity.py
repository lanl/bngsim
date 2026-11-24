"""``Result.check()`` of a run's forward sensitivities (issue #1893).

``"sensitivity"`` judges the sensitivities of the tolerance check's run, a
tenth of the tolerance, series by series; ``"sensitivity_fd"`` makes a
difference quotient of runs with the sensitivities off. The oracle is a closed
form: for ``A' = -kA``, ``dA/dk = -t·A0·e^{-kt}``, ``dA/dA0 = e^{-kt}``. A
wrong sensitivity is made to order in two ways: a run whose sensitivities are
off and its reruns' are not (a solve that is not converged, which the
tolerance tier sees), and every sensitivity solve off the same way (a wrong
right-hand side or seed, which only the quotient sees, as its runs compute
no sensitivities).
"""

from __future__ import annotations

import copy
import json
import math

import bngsim
import numpy as np
import pytest
from bngsim import Model, Result, Simulator, _check, _check_sensitivity, _rerun, examples

DECAY = {"k": 0.5, "A0": 100.0}


def _decay(**kw) -> Result:
    sim = Simulator(
        examples.load("decay", format="net"),
        method="ode",
        sensitivity_params=kw.pop("params", ["k", "A0"]),
        sensitivity_ic=kw.pop("ic", ["A()"]),
    )
    return sim.run(**{"t_span": (0, 10), "n_points": 11, **kw})


def _by_name(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["checks"]}


def _scaled(r: Result, column: int, factor: float) -> Result:
    """*r* with one parameter column of its sensitivities scaled."""
    out = copy.copy(r)
    s = np.asarray(r._sensitivities, dtype=float).copy()
    s[:, :, column] *= factor
    out._sensitivities = s
    return out


def _every_solve_off(monkeypatch, column: int, factor: float) -> None:
    """Every sensitivity solve's column *column* scaled by *factor*, as a wrong
    right-hand side or seed would make it: the run's and its reruns' alike."""
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        if len(out._sensitivity_params) > column:
            return _scaled(out, column, factor)
        return out

    monkeypatch.setattr(Simulator, "run", run)


# ─── Which checks run ────────────────────────────────────────────────────────


def test_the_closed_form_holds_and_both_tiers_pass_on_a_correct_run():
    r = _decay()
    t = np.asarray(r.time)
    s = np.asarray(r._sensitivities)
    exact_k = -t * DECAY["A0"] * np.exp(-DECAY["k"] * t)
    assert np.allclose(s[:, 0, 0], exact_k, rtol=1e-6, atol=1e-8)
    assert np.allclose(s[:, 0, 1], np.exp(-DECAY["k"] * t), rtol=1e-6, atol=1e-10)
    assert np.allclose(np.asarray(r._sensitivities_ic)[:, 0, 0], np.exp(-DECAY["k"] * t))
    rep = r.check("all")
    by = _by_name(rep)
    assert by["sensitivity"]["status"] == "pass" and by["sensitivity_fd"]["status"] == "pass"
    assert by["sensitivity"]["value"] < 1e-6
    assert by["sensitivity_fd"]["n_judged"] == 3
    assert {c["name"] for c in by["sensitivity_fd"]["columns"]} == {"k", "A0", "A()"}
    assert rep["ok"] is True
    assert json.loads(json.dumps(rep, allow_nan=False)) == rep


def test_the_default_set_adds_the_tolerance_tier_for_a_sensitivity_run_only():
    assert [c["name"] for c in _decay().check()["checks"]] == ["tolerance", "grid", "sensitivity"]
    plain = Simulator(examples.load("decay", format="net"), method="ode").run(
        t_span=(0, 10), n_points=11
    )
    assert [c["name"] for c in plain.check()["checks"]] == ["tolerance", "grid"]
    assert [c["name"] for c in plain.check("all")["checks"]] == list(_check.DETERMINISTIC)
    rep = plain.check(["sensitivity", "sensitivity_fd"])
    assert {c["status"] for c in rep["checks"]} == {"unsupported"}
    assert "computed no sensitivities" in rep["checks"][0]["note"]


def test_all_on_a_sensitivity_run_skips_the_interpreted_backend_rather_than_raising():
    # A rerun repeats the sensitivities, which only compiled code computes: a
    # codegen=False rerun is refused, which is no finding about the run.
    by = _by_name(_decay().check("all"))
    assert list(by) == [*_check.DETERMINISTIC, *_check.SENSITIVITY]
    assert by["backend"]["status"] == "skip", by["backend"]
    assert "compiled" in by["backend"]["note"]


def test_a_run_stopped_at_a_steady_state_is_not_judged():
    r = _decay(steady_state=True, t_span=(0, 200))
    rep = r.check(["sensitivity", "sensitivity_fd"])
    assert {c["status"] for c in rep["checks"]} == {"unsupported"}
    assert "steady state" in rep["checks"][0]["note"]


# ─── The tolerance tier ──────────────────────────────────────────────────────


def test_a_sensitivity_that_moves_at_the_tighter_tolerance_fails_the_tolerance_tier():
    bad = _scaled(_decay(), 0, 1.01)
    rep = bad.check(["sensitivity"])
    v = rep["checks"][0]
    assert v["status"] == "diff" and rep["ok"] is False
    assert v["worst_column"] == "dA()/dk"
    assert v["columns"][0]["name"] == "k" and v["columns"][0]["worst_species"] == "A()"
    assert v["value"] == pytest.approx(0.01 / 1.01, rel=1e-3)
    assert "sensitivities move" in rep["next"]
    # The trajectory itself is fine: the species' tolerance check passes.
    assert _by_name(bad.check(["tolerance"]))["tolerance"]["status"] == "pass"


def test_a_small_series_beside_a_large_one_is_judged_on_its_own_peak():
    # B is a thousandth of A, and both move with k: B's dB/dk off by 1% is far
    # under a thousandth of A's dA/dk, and still a finding.
    ant = """
    model two
      J1: A -> ; k*A
      J2: B -> ; k*B
      k = 0.5
      A = 1000; B = 1
    end
    """
    r = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 10), n_points=11
    )
    out = copy.copy(r)
    s = np.asarray(r._sensitivities, dtype=float).copy()
    s[:, 1, 0] *= 1.01
    out._sensitivities = s
    v = _by_name(out.check(["sensitivity"]))["sensitivity"]
    assert v["status"] == "diff" and v["worst_column"] == "dB/dk"
    assert np.abs(s[:, 1, 0] - r._sensitivities[:, 1, 0]).max() < 1e-4 * np.abs(s[:, 0, 0]).max()


def test_a_series_at_zero_is_named_not_resolved_and_still_judged_on_its_column():
    # C never moves with k: dC/dk is 0 throughout, which the run does not
    # resolve. Its noise is no finding; a value as large as the column's is.
    ant = """
    model nr
      J1: A -> ; k*A
      J2: -> C; kc
      k = 0.5; kc = 0
      A = 100; C = 0
    end
    """
    r = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 10), n_points=11
    )
    v = _by_name(r.check(["sensitivity"]))["sensitivity"]
    assert v["status"] == "pass" and v["not_resolved"] == ["dC/dk"] and v["n_not_resolved"] == 1
    out = copy.copy(r)
    s = np.asarray(r._sensitivities, dtype=float).copy()
    s[5, 1, 0] = 0.5 * np.abs(s[:, 0, 0]).max()
    out._sensitivities = s
    v = _by_name(out.check(["sensitivity"]))["sensitivity"]
    assert v["status"] == "diff" and v["worst_column"] == "dC/dk"
    nan = copy.copy(r)
    s = np.asarray(r._sensitivities, dtype=float).copy()
    s[5, 1, 0] = np.nan
    nan._sensitivities = s
    assert _by_name(nan.check(["sensitivity"]))["sensitivity"]["status"] == "diff"


def test_the_floor_is_the_atol_the_solver_held_each_series_to():
    # atolS_ij = atol_i·max(|x_i(0)|, 1)/|p_j|: the k column's floor is
    # 1e-8·100/0.5, the A0 column's 1e-8·100/100, the initial amount's 1e-8·100.
    r = _decay()
    ctx = _check._Context(r, r.record, None)
    cols = _check_sensitivity._Columns(ctx)
    assert cols.names == ["k", "A0", "A()"]
    assert cols.atol[0] == pytest.approx([1e-8 * 100 / 0.5, 1e-8, 1e-6])


# ─── The difference quotient ─────────────────────────────────────────────────


@pytest.mark.parametrize(("column", "name"), [(0, "dA()/dk"), (1, "dA()/dA0")])
def test_a_wrong_solve_every_tolerance_agrees_on_is_found_by_the_quotient(
    monkeypatch, column, name
):
    # A0 sets A's initial amount: its column is all seed, e^{-kt}·dA(0)/dA0.
    r = _decay()
    _every_solve_off(monkeypatch, column, 1.01)
    bad = _scaled(r, column, 1.01)
    by = _by_name(bad.check(["sensitivity", "sensitivity_fd"]))
    assert by["sensitivity"]["status"] == "pass"
    v = by["sensitivity_fd"]
    assert v["status"] == "diff", v
    assert v["worst_column"] == name and v["margin"] > 1.0
    assert v["columns"][0]["name"] == ("k", "A0")[column]


def test_an_initial_amount_column_off_is_found_by_the_quotient(monkeypatch):
    r = _decay()
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        if len(out._sensitivity_ic_species):
            out = copy.copy(out)
            out._sensitivities_ic = np.asarray(out._sensitivities_ic) * 0.98
        return out

    monkeypatch.setattr(Simulator, "run", run)
    bad = copy.copy(r)
    bad._sensitivities_ic = np.asarray(r._sensitivities_ic) * 0.98
    v = _by_name(bad.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "diff" and v["worst_column"] == "dA()/dA()(0)"


def test_the_quotient_runs_compute_no_sensitivities_and_move_one_value_each(monkeypatch):
    seen = []
    real = _rerun.perturbed

    def perturbed(result, **kw):
        out = real(result, **kw)
        seen.append((kw.get("parameters"), kw.get("state"), out._sensitivity_params))
        return out

    monkeypatch.setattr(_rerun, "perturbed", perturbed)
    _decay().check(["sensitivity_fd"])
    # One base run, then two per column: k and A0 by 1e-3 of themselves, and
    # the initial amount A(0) = 100 through the start state.
    assert seen[0][:2] == (None, None)
    moved = [(p, s) for p, s, _ in seen[1:]]
    assert moved == [
        ({"k": 0.5 * 1.001}, None),
        ({"k": 0.5 * 0.999}, None),
        ({"A0": 100.0 * 1.001}, None),
        ({"A0": 100.0 * 0.999}, None),
        (None, {0: 100.0 * 1.001}),
        (None, {0: 100.0 * 0.999}),
    ]
    assert all(len(names) == 0 for _, _, names in seen)


def test_a_value_at_zero_is_moved_up_only(monkeypatch):
    ant = """
    model z
      J1: A -> B; k*A
      J2: -> B; kb
      k = 0.5; kb = 0
      A = 10; B = 0
    end
    """
    r = Simulator(
        Model.from_antimony_string(ant), method="ode", sensitivity_params=["k", "kb"],
        sensitivity_ic=["B"],
    ).run(t_span=(0, 5), n_points=11)  # fmt: skip
    seen = []
    real = _rerun.perturbed

    def perturbed(result, **kw):
        seen.append((kw.get("parameters"), kw.get("state")))
        return real(result, **kw)

    monkeypatch.setattr(_rerun, "perturbed", perturbed)
    v = _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "pass", v
    assert ({"kb": 1e-3}, None) in seen and ({"kb": 2e-3}, None) in seen
    assert all(p is None or all(x >= 0 for x in p.values()) for p, _ in seen)
    # B starts at 0: moved up on the scale of the state (A = 10).
    assert (None, {1: 1e-2}) in seen and (None, {1: 2e-2}) in seen
    # dB/dkb = t, which the one-way quotient resolves.
    assert np.allclose(np.asarray(r._sensitivities)[:, 1, 1], r.time, rtol=1e-6)


def test_a_switch_time_the_parameter_moves_is_widened_not_failed():
    # At t = ton exactly the run reports the right limit of dA/dton, and a
    # central quotient is half of it (issue #368): no finding.
    ant = """
    model sw
      J1: -> A; k1*(1 - step)
      J2: A -> ; k2*A
      step := piecewise(1, time >= ton, 0)
      k1 = 2; k2 = 0.3; ton = 4
      A = 0
    end
    """
    r = Simulator(
        Model.from_antimony_string(ant), method="ode", sensitivity_params=["k1", "k2", "ton"]
    ).run(t_span=(0, 10), n_points=11)
    s = np.asarray(r._sensitivities)[:, 0, 2]
    assert s[4] == pytest.approx(2.0, rel=1e-6)  # the right limit, at t = ton
    by = _by_name(r.check(["sensitivity", "sensitivity_fd"]))
    assert by["sensitivity"]["status"] == "pass" and by["sensitivity_fd"]["status"] == "pass"


def test_a_wrong_jump_at_a_switch_time_is_still_found_past_it(monkeypatch):
    ant = """
    model sw
      J1: -> A; k1*(1 - step)
      J2: A -> ; k2*A
      step := piecewise(1, time >= ton, 0)
      k1 = 2; k2 = 0.3; ton = 4
      A = 0
    end
    """
    r = Simulator(
        Model.from_antimony_string(ant), method="ode", sensitivity_params=["k1", "k2", "ton"]
    ).run(t_span=(0, 10), n_points=11)
    _every_solve_off(monkeypatch, 2, 1.05)
    v = _by_name(_scaled(r, 2, 1.05).check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "diff" and v["worst_column"] == "dA/dton"


@pytest.mark.parametrize(
    "event", ["at (time >= t_dose): A = A + dose", "at (A < thr): B = B + dose"]
)
def test_events_whose_times_the_parameters_move_pass(event):
    ant = f"""
    model ev
      J1: A -> B; k*A
      E1: {event}
      k = 0.5; dose = 100; t_dose = 5; thr = 30
      A = 100; B = 0
    end
    """
    params = ["k", "dose", "t_dose"] if "t_dose" in event else ["k", "dose", "thr"]
    r = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=params).run(
        t_span=(0, 10), n_points=11
    )
    by = _by_name(r.check(["sensitivity", "sensitivity_fd"]))
    assert by["sensitivity"]["status"] == "pass", by["sensitivity"]
    assert by["sensitivity_fd"]["status"] == "pass", by["sensitivity_fd"]


def test_an_initial_assignment_seed_is_checked_through_set_param(monkeypatch):
    # A = A0*s: the seed dA(0)/ds = A0 is in the run's dA/ds, and the
    # quotient's set_param of s moves A(0) by itself.
    ant = """
    model ia
      J1: A -> ; k*A
      k = 0.5; A0 = 10; s = 2
      A = A0*s
    end
    """
    r = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["s"]).run(
        t_span=(0, 5), n_points=6
    )
    assert r.ic_sensitivity_seed == {"A": {"s": 10.0}}
    assert _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]["status"] == "pass"
    _every_solve_off(monkeypatch, 0, 0.5)  # half the seed's effect, at every tolerance
    v = _by_name(_scaled(r, 0, 0.5).check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "diff" and v["worst_column"] == "dA/ds"


def test_a_loose_run_is_still_judged_by_the_quotient():
    # At rtol 1e-4 the quotient's noise floor reads the run's own drift, so it
    # still resolves the columns: a quotient that resolves nothing is vacuous.
    r = Simulator(examples.load("chain"), method="ode", sensitivity_params=["k1", "k2"]).run(
        t_span=(0, 20), n_points=21, rtol=1e-4, atol=1e-6
    )
    v = _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "pass"
    assert v["n_not_resolved"] <= 1  # of 6 series (dA/dk2 is exactly 0)


def test_a_refused_row_is_counted_not_judged():
    r = _decay()
    out = copy.copy(r)
    s = np.asarray(r._sensitivities, dtype=float).copy()
    s[:, 0, :] = np.nan  # the k and A0 series; the initial amount's stays
    out._sensitivities = s
    out._ar_sens_refused = frozenset({"A()"})
    # The reruns' rows are not refused, so each tier sees a row that is finite
    # where the run's is not: refused rows are left out of both.
    by = _by_name(out.check(["sensitivity", "sensitivity_fd"]))
    for name in ("sensitivity", "sensitivity_fd"):
        assert by[name]["status"] == "pass" and by[name]["n_refused"] == 2, by[name]
    out._ar_sens_refused = frozenset()
    by = _by_name(out.check(["sensitivity", "sensitivity_fd"]))
    assert by["sensitivity"]["status"] == "diff" and by["sensitivity_fd"]["status"] == "diff"


def test_a_budget_that_runs_out_names_the_columns_it_did_not_reach(monkeypatch):
    r = _decay()
    real = _rerun.perturbed
    calls = []

    def perturbed(result, **kw):
        calls.append(kw)
        if len(calls) > 3:  # the base run and k's two
            from bngsim._exceptions import SimulationError

            raise SimulationError("timed out", code="sim.timeout")
        return real(result, **kw)

    monkeypatch.setattr(_rerun, "perturbed", perturbed)
    rep = r.check(["sensitivity_fd"], budget=600)
    v = rep["checks"][0]
    assert v["status"] == "timeout" and rep["ok"] is False
    assert v["timed_out"] == ["A0", "A()"] and v["n_judged"] == 1


# ─── The rerun the quotient makes ────────────────────────────────────────────


def test_perturbed_refuses_a_derived_parameter_and_records_what_it_moved():
    ant = """
    model ia
      J1: A -> ; k*A
      k = 0.5; A0 = 10; s = 2
      A = A0*s
    end
    """
    r = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["s"]).run(
        t_span=(0, 5), n_points=6
    )
    derived = [n for n in r.record["run"]["parameters"] if n == "_ic_A"]
    assert derived
    with pytest.raises(bngsim.UsageError) as e:
        _rerun.perturbed(r, parameters={"_ic_A": 21.0})
    assert e.value.details["reason"] == "derived_parameter"
    again = _rerun.perturbed(r, parameters={"s": 3.0})
    assert again.record["run"]["start_state"][0] == pytest.approx(30.0)
    assert len(again._sensitivity_params) == 0 and again.notices == []
    changed = again.record["run"]["rerun_of"]["changed"]
    assert changed["parameters"] == {"s": 3.0} and changed["sensitivities"] is None


def test_bngsim_check_exits_6_on_a_sensitivity_diff_with_no_change_to_the_command(tmp_path):
    import subprocess
    import sys

    from bngsim import _cli

    script = (
        "import copy, numpy as np, bngsim\n"
        "S = bngsim.Simulator\n"
        "init, run = S.__init__, S.run\n"
        "def __init__(self, model, **kw):\n"
        "    kw.setdefault('sensitivity_params', ['k'])\n"
        "    init(self, model, **kw)\n"
        "def wrong(self, *a, **kw):\n"
        "    out = run(self, *a, **kw)\n"
        "    if len(out._sensitivity_params):\n"
        "        out = copy.copy(out)\n"
        "        out._sensitivities = np.asarray(out._sensitivities) * 1.02\n"
        "    return out\n"
        "S.__init__, S.run = __init__, wrong\n"
    )
    decay = str(examples.path("decay", "net"))
    args = ["check", decay, "--t-end", "10", "--n-points", "11", "--json"]

    def main(*extra: str) -> subprocess.CompletedProcess:
        code = (
            script + f"from bngsim._cli import main\nraise SystemExit(main({[*args, *extra]!r}))\n"
        )
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )

    ok = main()
    assert ok.returncode == 0, ok.stderr
    assert [c["name"] for c in json.loads(ok.stdout)["check"]["checks"]][-1] == "sensitivity"
    bad = main("--checks", "sensitivity_fd")
    assert bad.returncode == _cli.EXIT_CHECK_FAILED, bad.stderr
    assert json.loads(bad.stdout)["check"]["checks"][0]["status"] == "diff"


def test_the_constants_say_what_the_docs_say():
    assert _check_sensitivity.FD_STEP == 1e-3 and _check_sensitivity.FD_TIGHTER == 100.0
    assert math.isclose(_check_sensitivity.FD_NOISE, _check.DRIFT_FACTOR)
    assert _check.SENSITIVITY == ("sensitivity", "sensitivity_fd")
