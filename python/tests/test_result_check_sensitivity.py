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
from pathlib import Path

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
    assert 0.0 < by["sensitivity_fd"]["allowance"] < 0.01
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
    # The refusal the skip stands in for: if it goes, so should the skip.
    with pytest.raises(bngsim.UsageError, match="codegen=False"):
        _decay().rerun(codegen=False)


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


TWO = """
model two
  J1: A -> ; k*A
  J2: B -> ; k*B
  k = 0.5
  A = {a0}; B = 1
end
"""


@pytest.mark.parametrize(
    ("a0", "row", "offsets", "status", "held"),
    [
        # At A = 1e12 every entry of the k column carries the column's
        # roundoff, eps times dA/dk's peak (1.6e-4, #177): dB/dk errs by as
        # much at every tolerance, 2e-3 of its peak or 18 of that floor. A
        # floor holds it, and on its column's scale it is noise.
        (1e12, 1, {-8: 2e-3, -9: -2e-3, -10: 2e-3}, "pass", ["dB/dk"]),
        # At A = 1e11 the floor is ten times smaller and the same error about
        # 180 floors: past the 100 a floor's move is within.
        (1e11, 1, {-8: 2e-3, -9: -2e-3, -10: 2e-3}, "diff", []),
        # The same error under the column's largest series is 1e5 of its
        # floor: no floor's, whatever it does with the tolerance.
        (1e12, 0, {-8: 2e-3, -9: -2e-3, -10: 2e-3}, "diff", []),
        # And at A = 1000, where dB/dk's floor is its atol, 1e5 of it too.
        (1000.0, 1, {-8: 2e-3, -9: -2e-3, -10: 2e-3}, "diff", []),
        # An error that shrinks tenfold with the tolerance is the tolerance's.
        (1e12, 1, {-8: 2e-3, -9: 2e-4, -10: 2e-5}, "diff", []),
        # One that shrinks slowly, the same way, its second move more than
        # half its first, is the tolerance's too (#1962).
        (1000.0, 1, {-8: 1e-2, -9: 6e-3, -10: 3.6e-3}, "diff", []),
        # And one that barely moves at rtol/10 and then drops, as #1962's
        # series do: its second move is larger than its first.
        (1000.0, 1, {-8: 5e-3, -9: 3e-3, -10: 2e-4}, "diff", []),
        # One the same at rtol and rtol/10, gone at rtol/100: its move at
        # rtol/10 is under the bound, its move to rtol/100 ten times it (#1967).
        (1000.0, 1, {-8: 5e-3, -9: 4.5e-3, -10: 0.0}, "diff", []),
        # The floor's size is read on the larger move: at A = 1e12 an error 9
        # floors between rtol and rtol/10 and 138 to rtol/100 is no floor's.
        (1e12, 1, {-8: 1e-2, -9: 8e-3, -10: -2e-2}, "diff", []),
        # Nor is one past the bound only at rtol/100, though within 100 floors:
        # a floor's error shows at rtol/10 as well (#1967).
        (1e12, 1, {-8: 5e-3, -9: 4.95e-3, -10: 0.0}, "diff", []),
    ],
)
def test_a_series_a_floor_holds_is_named_and_judged_on_its_column(
    monkeypatch, a0, row, offsets, status, held
):
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        if len(out._sensitivity_params):
            out = copy.copy(out)
            s = np.asarray(out._sensitivities, dtype=float).copy()
            off = offsets[round(math.log10(kw.get("rtol", 1e-8)))]
            s[:, row, 0] += off * np.abs(s[:, row, 0]).max() * np.asarray(out.time) / 10.0
            out._sensitivities = s
        return out

    monkeypatch.setattr(Simulator, "run", run)
    sim = Simulator(
        Model.from_antimony_string(TWO.format(a0=a0)), method="ode", sensitivity_params=["k"]
    )
    r = sim.run(t_span=(0, 10), n_points=11, rtol=1e-8)
    by = _by_name(r.check(["sensitivity", "sensitivity_fd"]))
    v = by["sensitivity"]
    assert v["status"] == status, v
    assert v["n_floor_held"] == len(held)
    assert all(h in v["not_resolved"] for h in held)
    if held:
        assert v["value_10"] <= _check.TOLERANCE_MOVE and v["value_100"] <= _check.TOLERANCE_MOVE
    if status == "pass":
        # The quotient reads a floor-held series' move as its drift too.
        assert by["sensitivity_fd"]["status"] == "pass", by["sensitivity_fd"]


@pytest.mark.parametrize("rtol", [2e-8, 8e-8, 9.9e-8, 1e-7, 2e-7, 2.759e-7])
def test_a_series_converging_slowly_is_not_read_as_held_by_a_floor(rtol):
    # Issue #1962: at these rtols BIOMD0000000810's gamma_C series err by 0.4%
    # to 3.5% of their peaks against a run at rtol 1e-12, and their error
    # often barely moves at rtol/10 and drops at rtol/100. The tier read their
    # second move, as large as the first or larger, as a floor's and passed
    # every one of these runs. T_H1's moves are 1e8 to 1e9 of its floor, and
    # IL2's, whose peak is 1.9e5 floors, at least 190: never a floor's.
    path = Path(__file__).resolve().parents[2] / "tests" / "data" / "BIOMD0000000810.xml"
    params = ["gamma_C", "delta_Treg", "myu_TcS", "k6", "tck"]
    r = Simulator(Model.load(path), method="ode", sensitivity_params=params).run(
        t_span=(0, 500), n_points=21, rtol=rtol
    )
    ctx = _check._Context(r, r.record, None)
    v = _check_sensitivity.check_sensitivity(ctx)
    held = set(_check_sensitivity._named(_check_sensitivity._context(ctx), ctx.sens_drift[3]))
    assert not held & {"dType_I_T_helper_Cell_T_H1/dgamma_C", "dCytokine_IL2/dgamma_C"}, held
    # The verdict too, where the move is well past the bound (at 2e-8 to
    # 9.9e-8 it is 1.5 to 3 times it, and a different step sequence moves it).
    if rtol >= 1e-7:
        assert v["status"] == "diff", v


@pytest.mark.parametrize("rtol", [5.151e-9, 1.314e-8])
def test_a_series_whose_error_is_flat_over_the_first_decade_is_found_at_the_second(rtol):
    # Issue #1967: at these rtols BIOMD0000000810's IL2 and T_H1 gamma_C
    # series are 0.8% to 0.9% off against a run at rtol 1e-12, but move only
    # 5e-4 to 8e-4 of their peaks at rtol/10: their error is about the same
    # there. To rtol/100 they move 7e-3 to 9e-3, seven to nine times the bound.
    path = Path(__file__).resolve().parents[2] / "tests" / "data" / "BIOMD0000000810.xml"
    params = ["gamma_C", "delta_Treg", "myu_TcS", "k6", "tck"]
    r = Simulator(Model.load(path), method="ode", sensitivity_params=params).run(
        t_span=(0, 500), n_points=21, rtol=rtol
    )
    # The two moves themselves swing with the step sequence (a 1e-6 change of
    # rtol moves value_10 fivefold); the verdict holds, and the synthetic case
    # in the floor test pins the mechanism.
    v = _by_name(r.check(["sensitivity"]))["sensitivity"]
    assert v["status"] == "diff", v


def test_a_budget_that_runs_out_before_rtol_100_is_a_timeout(monkeypatch):
    # Without the run at rtol/100 an error flat over the first decade is not
    # seen: the check does not pass what it could not judge.
    real = _check._Context.rerun

    def rerun(self, **changes):
        if changes.get("rtol", 1.0) < self.rtol / 50:
            raise _check._Out
        return real(self, **changes)

    monkeypatch.setattr(_check._Context, "rerun", rerun)
    rep = _decay().check(["sensitivity"])
    v = rep["checks"][0]
    assert v["status"] == "timeout" and "rtol/100" in v["note"] and rep["ok"] is False, v


def test_a_run_at_rtol_100_that_raises_leaves_the_tenfold_verdict(monkeypatch):
    real = Result.rerun

    def rerun(self, **changes):
        if changes.get("rtol", 1.0) < 5e-10:
            from bngsim._exceptions import SimulationError

            raise SimulationError("did not converge", code="sim.failed")
        return real(self, **changes)

    monkeypatch.setattr(Result, "rerun", rerun)
    v = _by_name(_decay().check(["sensitivity"]))["sensitivity"]
    assert v["status"] == "pass" and v["value_100"] is None, v
    assert "rtol/100" in v["note"] and "SimulationError" in v["note"], v


def test_a_run_at_rtol_100_that_raises_is_tried_once_in_a_check(monkeypatch):
    # The tolerance check tries it first; the sensitivity check reads why it
    # raised and does not try again.
    tried = []
    real = Result.rerun

    def rerun(self, **changes):
        if changes.get("rtol", 1.0) < 5e-10:
            tried.append(changes["rtol"])
            from bngsim._exceptions import SimulationError

            raise SimulationError("did not converge", code="sim.failed")
        return real(self, **changes)

    monkeypatch.setattr(Result, "rerun", rerun)
    v = _by_name(_decay().check())["sensitivity"]
    assert len(tried) == 1, tried
    assert v["status"] == "pass" and v["value_100"] is None and "did not converge" in v["note"]


def test_a_rerun_past_the_run_s_own_timeout_is_no_budget_running_out(monkeypatch):
    # The run's own timeout is shorter than the budget: a rerun that runs past
    # it raises as the run would have, and the move at rtol/10 is judged. Only
    # where the budget is the shorter is it a timeout.
    real = Result.rerun

    def rerun(self, **changes):
        if changes.get("rtol", 1.0) < 5e-10:
            from bngsim._exceptions import SimulationError

            raise SimulationError("timed out", code="sim.timeout")
        return real(self, **changes)

    monkeypatch.setattr(Result, "rerun", rerun)
    r = _decay(timeout=60.0)
    v = r.check(["sensitivity"], budget=600)["checks"][0]
    assert v["status"] == "pass" and v["value_100"] is None and "timed out" in v["note"], v
    v = r.check(["sensitivity"], budget=30)["checks"][0]
    assert v["status"] == "timeout", v


def test_the_quotient_alone_makes_no_run_at_rtol_100(monkeypatch):
    # Only the sensitivity check judges by it; the quotient reads it where the
    # tolerance check made it, so a budget it would use up costs the quotient
    # nothing.
    real = _check._Context.rerun
    tried = []

    def rerun(self, **changes):
        if changes.get("rtol", 1.0) < self.rtol / 50 and changes.get("rtol") > 1e-13:
            tried.append(changes["rtol"])
            raise _check._Out
        return real(self, **changes)

    monkeypatch.setattr(_check._Context, "rerun", rerun)
    v = _by_name(_decay().check(["sensitivity_fd"], budget=600))["sensitivity_fd"]
    assert v["status"] == "pass" and tried == [], (v, tried)


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
    rep = bad.check(["sensitivity_fd"])
    assert "difference quotient" in rep["next"] and rep["ok"] is False


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
    # One base run, then four per column: k and A0 by 1e-3 and 2e-3 of
    # themselves each way, and the initial amount A(0) = 100 through the
    # start state.
    assert seen[0][:2] == (None, None)
    moved = [(p, s) for p, s, _ in seen[1:]]
    steps = (1.001, 0.999, 1.002, 0.998)
    assert moved == [
        *(({"k": 0.5 * f}, None) for f in steps),
        *(({"A0": 100.0 * f}, None) for f in steps),
        *((None, {0: 100.0 * f}) for f in steps),
    ]
    assert all(len(names) == 0 for _, _, names in seen)


def test_a_value_at_zero_is_not_judged_by_the_quotient():
    # No step size follows from 0: a step of 1e-3 in absolute terms is far past
    # a saturating response at K = 1e-6, and the quotient is then far off.
    ant = """
    model z
      J1: -> X; kb
      J2: X -> ; kd*X
      J3: -> Y; v*X/(K + X)
      kb = 0; kd = 1; v = 1; K = 1e-6
      X = 0; Y = 0
    end
    """
    r = Simulator(
        Model.from_antimony_string(ant), method="ode", sensitivity_params=["kb", "kd"],
        sensitivity_ic=["X"],
    ).run(t_span=(0, 2), n_points=11)  # fmt: skip
    v = _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "pass", v
    assert set(v["not_judged"]) == {"kb", "X"}
    assert "value is 0" in v["not_judged"]["kb"]
    assert v["n_judged"] == 1


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


def test_a_switch_sampled_densely_is_widened_there_and_judged_past_it(monkeypatch):
    # 201 output points within 0.01 of ton: the quotient's Richardson term is
    # large there as well as its kink term, and the cells past the switch are
    # still judged, so a wrong dA/dton is still found.
    ant = """
    model sw
      J1: -> A; k1*(1 - step)
      J2: A -> ; k2*A
      step := piecewise(1, time >= ton, 0)
      k1 = 2; k2 = 0.3; ton = 4
      A = 0
    end
    """
    times = sorted({*np.linspace(0, 10, 11).tolist(), *np.linspace(3.99, 4.01, 201).tolist()})
    sim = Simulator(
        Model.from_antimony_string(ant), method="ode", sensitivity_params=["k1", "k2", "ton"]
    )
    r = sim.run(sample_times=times)
    assert _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]["status"] == "pass"
    _every_solve_off(monkeypatch, 2, 1.05)
    v = _by_name(_scaled(r, 2, 1.05).check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "diff" and v["worst_column"] == "dA/dton", v


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
    assert v["changed"]["rtol"] == _check_sensitivity.FD_RTOL_MOST


@pytest.mark.parametrize(("factor", "status"), [(1.05, "diff"), (1.002, "pass")])
def test_a_loose_run_s_wrong_column_is_found_to_its_tolerance(monkeypatch, factor, status):
    # At rtol 1e-3 two correct paths may differ by 1% of a peak: a column 5%
    # off is a finding, one 0.2% off is within what the run's tolerance allows.
    r = _decay(rtol=1e-3, atol=1e-6)
    _every_solve_off(monkeypatch, 0, factor)
    v = _by_name(_scaled(r, 0, factor).check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == status, v
    assert v["allowance"] is not None and v["allowance"] <= 0.011


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
        if len(calls) > 5:  # the base run and k's four
            from bngsim._exceptions import SimulationError

            raise SimulationError("timed out", code="sim.timeout")
        return real(result, **kw)

    monkeypatch.setattr(_rerun, "perturbed", perturbed)
    rep = r.check(["sensitivity_fd"], budget=600)
    v = rep["checks"][0]
    assert v["status"] == "timeout" and rep["ok"] is False
    assert v["timed_out"] == ["A0", "A()"] and v["n_judged"] == 1


@pytest.mark.parametrize(
    ("w", "t_end", "n"), [(2.0, 50.0, 501), (2.0, 200.0, 401), (math.pi, 10.0, 11)]
)
def test_an_oscillator_over_many_periods_passes_the_quotient(w, t_end, n):
    # x = cos(wt), y = -sin(wt): dx/dw = -t·sin(wt), dy/dw = -t·cos(wt). Where
    # d²Y/dw² crosses 0 the quotient still has its truncation error, which the
    # Richardson difference of its two steps measures.
    ant = f"""
    model osc
      J1: -> x; w*y
      J2: -> y; -w*x
      w = {w!r}
      x = 1; y = 0
    end
    """
    r = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["w"]).run(
        t_span=(0, t_end), n_points=n
    )
    t = np.asarray(r.time)
    s = np.asarray(r._sensitivities)[:, :, 0]
    assert np.allclose(s[:, 1], -t * np.cos(w * t), atol=1e-5 * t_end)
    v = _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "pass", v


OSC = """
model osc
  J1: -> x; w*y
  J2: -> y; -w*x
  w = 2
  x = 1; y = 0
end
"""


@pytest.mark.parametrize(
    ("t_end", "n", "f", "status", "named"),
    [
        # 3% off at the end of 16 periods: within what the quotient resolves.
        (50.0, 501, 0.03, "diff", []),
        # 10% off at the end of 64 periods: the step moves the phase too far
        # for the quotient to tell, from some time on, and it says so.
        (200.0, 401, 0.1, "pass", ["dx/dw", "dy/dw"]),
        # A correct run over 500 periods, past the step's phase: no finding.
        (1600.0, 3201, 0.0, "pass", ["dx/dw", "dy/dw"]),
    ],
)
def test_a_long_oscillation_is_judged_as_far_as_the_quotient_resolves_it(
    monkeypatch, t_end, n, f, status, named
):
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        if len(out._sensitivity_params):
            out = copy.copy(out)
            s = np.asarray(out._sensitivities, dtype=float).copy()
            s[:, :, 0] *= (1.0 + f * (np.asarray(out.time) / t_end) ** 2)[:, None]
            out._sensitivities = s
        return out

    monkeypatch.setattr(Simulator, "run", run)
    r = Simulator(Model.from_antimony_string(OSC), method="ode", sensitivity_params=["w"]).run(
        t_span=(0, t_end), n_points=n, rtol=1e-10, atol=1e-12
    )
    v = _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == status, v
    assert v["not_resolved"] == named
    if named == ["dx/dw", "dy/dw"]:
        assert v["allowance"] is None  # no series is resolved to claim an accuracy for
    else:
        assert 0.0 < v["allowance"] < 0.02


@pytest.mark.parametrize("factor", [0.0, 1e-6])
def test_a_column_every_solve_gives_as_zero_is_found(monkeypatch, factor):
    # A dropped seed or a lost right-hand side term reads as 0, or next to it:
    # the quotient's masks follow the quotient's scale, not the run's own.
    r = _decay()
    _every_solve_off(monkeypatch, 0, factor)
    v = _by_name(_scaled(r, 0, factor).check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "diff" and v["worst_column"] == "dA()/dk", v


def test_a_series_whose_only_values_sit_on_a_switch_is_not_judged_and_says_so(monkeypatch):
    # dA/dt_dose is [0, 0, 50] at [0, 2.5, 5]: its one value is on the event
    # time, where the quotient is masked. Three times too large is then no
    # pass: nothing was compared.
    ant = """
    model bol
      J1: A -> ; k*A
      E1: at (time >= t_dose): A = A + dose
      k = 0.5; dose = 100; t_dose = 5
      A = 100
    end
    """
    sim = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["t_dose"])
    r = sim.run(sample_times=[0.0, 2.5, 5.0])
    _every_solve_off(monkeypatch, 0, 3.0)
    v = _by_name(_scaled(r, 0, 3.0).check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "skip" and v["n_judged"] == 0, v
    assert "t_dose" in v["not_judged"]


@pytest.mark.parametrize(("factor", "status"), [(1.0, "pass"), (1.05, "diff")])
def test_an_event_sampled_on_its_time_is_judged_on_the_cells_past_it(monkeypatch, factor, status):
    # On t_dose the quotient of the dose's jump is about dose/(2h), no
    # derivative: it must not set the series' scale, or the five clean cells
    # past the event go unjudged.
    ant = """
    model bol
      J1: A -> ; k*A
      E1: at (time >= t_dose): A = A + dose
      k = 0.5; dose = 100; t_dose = 5
      A = 100
    end
    """
    sim = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["t_dose"])
    r = sim.run(t_span=(0, 10), n_points=11)
    _every_solve_off(monkeypatch, 0, factor)
    v = _by_name(_scaled(r, 0, factor).check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == status and v["n_judged"] == 1, v
    if status == "diff":
        assert v["worst_column"] == "dA/dt_dose"
    else:
        assert v["not_resolved"] == []


def test_a_value_every_solve_gives_as_nan_fails_the_tolerance_tier(monkeypatch):
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        if len(out._sensitivity_params):
            out = copy.copy(out)
            s = np.asarray(out._sensitivities, dtype=float).copy()
            s[5:, 0, 0] = np.nan
            out._sensitivities = s
        return out

    monkeypatch.setattr(Simulator, "run", run)
    r = _decay()
    rep = r.check()
    assert _by_name(rep)["sensitivity"]["status"] == "diff" and rep["ok"] is False
    assert _by_name(rep)["sensitivity"]["worst_column"] == "dA()/dk"


def test_a_quotient_diff_on_unconverged_series_says_to_tighten():
    r = Simulator(examples.load("chain"), method="ode", sensitivity_params=["k1", "k2"]).run(
        t_span=(0, 20), n_points=21, rtol=3e-2, atol=1e-4
    )
    rep = r.check(["sensitivity_fd"])
    v = rep["checks"][0]
    assert v["status"] == "diff" and v["diff_not_converged"] is True and v["n_not_converged"] > 0
    assert "not converged" in rep["next"] and "bug" not in rep["next"]


def test_a_late_time_error_of_a_fraction_of_a_percent_is_found(monkeypatch):
    # An error that grows as t²: 0.5% of dA/dk at t = 10, where a first-order
    # estimate of the quotient's error would be about 1%.
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        if len(out._sensitivity_params):
            out = copy.copy(out)
            s = np.asarray(out._sensitivities, dtype=float).copy()
            s[:, 0, 0] *= 1.0 + 0.005 * (np.asarray(out.time) / 10.0) ** 2
            out._sensitivities = s
        return out

    monkeypatch.setattr(Simulator, "run", run)
    r = _decay()
    v = _by_name(r.check(["sensitivity", "sensitivity_fd"]))
    assert v["sensitivity"]["status"] == "pass"
    assert v["sensitivity_fd"]["status"] == "diff" and v["sensitivity_fd"]["worst_column"] == (
        "dA()/dk"
    )


def test_an_unconverged_series_does_not_widen_its_own_quotient_check():
    # The run's dA/dk alone 1% off: the tolerance check sees it move, and the
    # quotient is not to forgive it by that move.
    by = _by_name(_scaled(_decay(), 0, 1.01).check(["sensitivity", "sensitivity_fd"]))
    assert by["sensitivity"]["status"] == "diff"
    v = by["sensitivity_fd"]
    assert v["status"] == "diff" and v["worst_column"] == "dA()/dk", v
    assert v["n_not_converged"] == 1


def test_a_derived_parameter_column_is_not_judged_by_the_quotient():
    ant = """
    model ia
      J1: A -> ; k*A
      k = 0.5; A0 = 10; s = 2
      A = A0*s
    end
    """
    r = Simulator(
        Model.from_antimony_string(ant), method="ode", sensitivity_params=["_ic_A", "k"]
    ).run(t_span=(0, 5), n_points=6)
    v = _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "pass" and v["n_judged"] == 1
    assert "derived parameter" in v["not_judged"]["_ic_A"]
    assert [c["name"] for c in v["columns"]] == ["k"]


NR = """
model nr
  J1: A -> ; k*A
  J2: -> C; kc
  k = 0.5; kc = 0
  A = 100; C = 0
end
"""


@pytest.mark.parametrize(("share", "status"), [(0.5, "pass"), (1.05, "diff")])
def test_a_series_at_zero_is_judged_on_its_column_s_peak(monkeypatch, share, status):
    # dC/dk is 0 throughout and its atolS 1e-8/0.5: the tighter run reading it
    # at a share of a thousandth of its column's peak (dA/dk's) past 100 atolS
    # is within the bound on that scale, and past it at a little more.
    r = Simulator(Model.from_antimony_string(NR), method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 10), n_points=11
    )
    col_peak = np.abs(np.asarray(r._sensitivities)[:, 0, 0]).max()
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        out = copy.copy(out)
        s = np.asarray(out._sensitivities, dtype=float).copy()
        s[5, 1, 0] = share * 1e-3 * col_peak + 100 * 2e-8
        out._sensitivities = s
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    v = _by_name(r.check(["sensitivity"]))["sensitivity"]
    assert v["status"] == status, v
    assert "dC/dk" in v["not_resolved"]


def test_the_quotient_names_a_series_it_does_not_resolve():
    r = Simulator(Model.from_antimony_string(NR), method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 10), n_points=11
    )
    v = _by_name(r.check(["sensitivity_fd"]))["sensitivity_fd"]
    assert v["status"] == "pass" and v["not_resolved"] == ["dC/dk"]


def test_a_column_s_roundoff_is_a_floor_under_its_small_entries():
    # B starts at 0 (atolS 1e-8·1/0.5) and is made from A = 1e12: dB/dk is of
    # order 1e12, and every entry of that column carries eps of it.
    ant = """
    model big
      J1: A -> B; k*A
      k = 0.5
      A = 1e12; B = 0
    end
    """
    r = Simulator(Model.from_antimony_string(ant), method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 10), n_points=11
    )
    ctx = _check._Context(r, r.record, None)
    cols = _check_sensitivity._Columns(ctx)
    colmax = np.abs(np.asarray(r._sensitivities)[:, :, 0]).max()
    assert cols.atol[1, 0] == pytest.approx(np.finfo(float).eps * colmax)
    assert cols.atol[1, 0] > 1e-8 / 0.5
    assert cols.atol[0, 0] == pytest.approx(1e-8 * 1e12 / 0.5)


def test_a_budget_that_runs_out_telling_a_floor_from_the_tolerance_is_a_timeout(monkeypatch):
    # A move far past its floor is the tolerance's: no floor is told apart,
    # and the run at rtol/100 is made once, for the move to it (#1967).
    rtols: list[float] = []
    counted = Result.rerun

    def count(self, **changes):
        rtols.append(changes.get("rtol"))
        return counted(self, **changes)

    with monkeypatch.context() as m:
        m.setattr(Result, "rerun", count)
        v = _scaled(_decay(), 0, 1.01).check(["sensitivity"], budget=600)["checks"][0]
    assert v["status"] == "diff", v
    assert rtols == [pytest.approx(1e-9), pytest.approx(1e-10)], rtols
    # dB/dk past the bound at rtol/10 by a few of its floor (the case the floor
    # test holds): the rtol/100 run that would tell a floor from the tolerance
    # runs out of budget, so the move is not reported as a diff it may not be.
    real_run = Simulator.run

    def run(self, *a, **kw):
        out = real_run(self, *a, **kw)
        if len(out._sensitivity_params):
            out = copy.copy(out)
            s = np.asarray(out._sensitivities, dtype=float).copy()
            off = 2e-3 * (-1) ** round(math.log10(kw.get("rtol", 1e-8)))
            s[:, 1, 0] += off * np.abs(s[:, 1, 0]).max() * np.asarray(out.time) / 10.0
            out._sensitivities = s
        return out

    real = Result.rerun

    def rerun(self, **changes):
        if changes.get("rtol", 1.0) < 5e-10:
            from bngsim._exceptions import SimulationError

            raise SimulationError("timed out", code="sim.timeout")
        return real(self, **changes)

    monkeypatch.setattr(Simulator, "run", run)
    monkeypatch.setattr(Result, "rerun", rerun)
    sim = Simulator(
        Model.from_antimony_string(TWO.format(a0=1e12)), method="ode", sensitivity_params=["k"]
    )
    rep = sim.run(t_span=(0, 10), n_points=11, rtol=1e-8).check(["sensitivity"], budget=600)
    v = rep["checks"][0]
    assert v["status"] == "timeout" and "floor" in v["note"] and rep["ok"] is False


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


def test_the_constants_say_what_the_docs_say():
    assert _check_sensitivity.FD_STEP == 1e-3 and _check_sensitivity.FD_TIGHTER == 100.0
    assert (_check_sensitivity.FD_RTOL_LEAST, _check_sensitivity.FD_RTOL_MOST) == (1e-14, 1e-8)
    assert math.isclose(_check_sensitivity.FD_NOISE, _check.DRIFT_FACTOR)
    assert _check.SENSITIVITY == ("sensitivity", "sensitivity_fd", "sensitivity_second_engine")
