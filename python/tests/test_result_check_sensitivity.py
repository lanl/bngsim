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
  A = 1000; B = 1
end
"""


@pytest.mark.parametrize(
    ("row", "offsets", "status", "held"),
    [
        # dB/dk errs by as much at every tolerance: a floor holds it, and on
        # its column's scale (dA/dk's, a thousand times larger) it is noise.
        (1, {-8: 2e-3, -9: -2e-3, -10: 2e-3}, "pass", ["dB/dk"]),
        # The same floor under the column's largest series: on its own scale.
        (0, {-8: 2e-3, -9: -2e-3, -10: 2e-3}, "diff", ["dA/dk"]),
        # An error that shrinks tenfold with the tolerance is the tolerance's.
        (1, {-8: 2e-3, -9: 2e-4, -10: 2e-5}, "diff", []),
    ],
)
def test_a_series_a_floor_holds_is_named_and_judged_on_its_column(
    monkeypatch, row, offsets, status, held
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
    sim = Simulator(Model.from_antimony_string(TWO), method="ode", sensitivity_params=["k"])
    r = sim.run(t_span=(0, 10), n_points=11, rtol=1e-8)
    by = _by_name(r.check(["sensitivity", "sensitivity_fd"]))
    v = by["sensitivity"]
    assert v["status"] == status, v
    assert v["n_floor_held"] == len(held)
    assert all(h in v["not_resolved"] for h in held)
    if status == "pass":
        # The quotient reads a floor-held series' move as its drift too.
        assert by["sensitivity_fd"]["status"] == "pass", by["sensitivity_fd"]


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
    # The run's dA/dk is past the bound at rtol/10; the rtol/100 run that would
    # tell a floor from the tolerance runs out of budget, so the move is not
    # reported as a diff it may not be.
    real = Result.rerun

    def rerun(self, **changes):
        if changes.get("rtol", 1.0) < 5e-10:
            from bngsim._exceptions import SimulationError

            raise SimulationError("timed out", code="sim.timeout")
        return real(self, **changes)

    monkeypatch.setattr(Result, "rerun", rerun)
    rep = _scaled(_decay(), 0, 1.01).check(["sensitivity"], budget=600)
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
    assert _check.SENSITIVITY == ("sensitivity", "sensitivity_fd")
