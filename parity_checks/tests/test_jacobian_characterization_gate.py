"""Issues #508–#510 and #512: the Jacobian harnesses run on the parity gate's own horizon
and tolerances, take their trajectory states from the full integrator state, survive a
degenerate sample, and characterize from the Jacobian the model actually carries.

Unit (always run): the gate-report contract of ``load_horizons`` / ``gate_run_settings``
and ``main()``'s refusal without a report; the rr harness's tolerance rule, cross-checked
against ``rr_run``'s on the whole manifest; the sweep's skip / clamp / count behaviour on
a fake evaluator; and the analysis step's handling of incomplete-Jacobian and sweep-less
rows. Corpus (a vendored SBML present): the issue #508 model end to end. Engine (BNG2.pl +
perl): one vendored BNGL characterized with and without a gate horizon.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import _core
import numpy as np
import pytest

_PC = Path(__file__).resolve().parent.parent
_BNG_PARITY = _PC / "bng_parity"
_RR_PARITY = _PC / "rr_parity"
for _p in (_BNG_PARITY, _RR_PARITY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # the rr harness imports the bng one by this name
    spec.loader.exec_module(mod)
    return mod


J = _load("jacobian_characterization", _BNG_PARITY / "jacobian_characterization.py")
R = _load("jacobian_characterization_sbml", _RR_PARITY / "jacobian_characterization_sbml.py")


def _gate_row(model_id="m.bngl", outcome="PASS", **spec):
    s = {"t_start": 0.0, "t_end": 10.0, "n_steps": 20, "rtol": 1e-8, "atol": 1e-10, "n_species": 3}
    s.update(spec)
    bs = {"integrate_warm_median_sec": 0.01, "config": {"linear_solver": "klu"}}
    return {"model_id": model_id, "outcome": outcome, "timing": {"spec": s, "bngsim": bs}}


def _report(tmp_path, rows):
    rp = tmp_path / "report_ode.json"
    rp.write_text(json.dumps({"results": rows}))
    return rp


# --------------------------------------------------------------------------- #
# Issue #509 — bng_parity: the gate report is the horizon, and it is required.
# --------------------------------------------------------------------------- #
def test_load_horizons_refuses_a_missing_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="report_ode.json"):
        J.load_horizons(tmp_path / "report_ode.json")


def test_load_horizons_reads_the_gates_resolved_spec(tmp_path):
    rp = _report(tmp_path, [_gate_row(rtol=1e-10, atol=1e-16)])
    h = J.load_horizons(rp)["m.bngl"]
    assert (h["t_start"], h["t_end"], h["n_steps"]) == (0.0, 10.0, 20)
    assert (h["rtol"], h["atol"]) == (1e-10, 1e-16)
    assert (h["outcome"], h["cost_sec"], h["linear_solver"]) == ("PASS", 0.01, "klu")
    assert h["source"] == str(rp)


def test_gate_run_settings_are_the_reports_values():
    h = {"t_start": 1.0, "t_end": 5.0, "n_steps": 4, "rtol": 1e-10, "atol": 1e-16, "source": "x"}
    assert J.gate_run_settings(h) == {
        "t_start": 1.0,
        "t_end": 5.0,
        "n_steps": 4,
        "rtol": 1e-10,
        "atol": 1e-16,
        "horizon_source": "x",
        "tolerance_source": "gate_report",
    }


def test_gate_run_settings_fall_back_to_the_gates_default_tolerances_and_say_so():
    run = J.gate_run_settings({"t_end": 5.0, "rtol": None, "atol": None})
    assert (run["rtol"], run["atol"]) == (J.bc.DEFAULT_RTOL, J.bc.DEFAULT_ATOL)
    assert run["tolerance_source"] == "gate_default"
    assert run["n_steps"] == J.bc.DEFAULT_N_STEPS


def test_no_gate_row_means_no_run():
    """The old code integrated on t_end=100 here, and nothing in the output said so."""
    assert J.gate_run_settings({}) is None
    assert J.gate_run_settings(None) is None
    assert J.gate_run_settings({"outcome": "PASS", "t_end": None}) is None


def test_main_refuses_to_run_without_the_gate_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(J, "HERE", tmp_path)
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["jacobian_characterization.py", "--out", str(out)])
    assert J.main() == 2
    assert "report_ode.json" in capsys.readouterr().err
    assert not out.exists()


def test_main_honours_an_explicit_horizons_path(tmp_path, monkeypatch, capsys):
    """An empty jobs manifest and a stubbed BNG2.pl resolver: main() runs to completion
    with no model, and the output's metadata names the report it was pointed at."""
    monkeypatch.setattr(J, "HERE", tmp_path)
    monkeypatch.setattr(J.bc, "resolve_bng2_pl", lambda _env: "BNG2.pl")  # no job invokes it
    (tmp_path / "jobs.json").write_text(json.dumps({"jobs": []}))
    (tmp_path / "elsewhere").mkdir()
    rp = _report(tmp_path / "elsewhere", [_gate_row()])
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["jc", "--horizons", str(rp), "--out", str(out)])
    assert J.main() == 0
    meta = json.loads(out.read_text())["_meta"]
    assert meta["horizons"] == str(rp)
    assert meta["n_models"] == 0
    assert "refusing to run" not in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Issue #509 — rr_parity: the gate's tolerances, not the SED-ML's.
# --------------------------------------------------------------------------- #
def _job(**params):
    p = {"t_start": 0.0, "t_end": 10.0, "n_points": 101, "rtol": 1e-6, "atol": 1e-9}
    p.update(params)
    return {"model_id": "B", "params": p, "overrides": []}


def test_gate_tolerances_default_to_the_gates_shared_pair_not_the_sedml_values():
    job = _job()
    assert R.gate_tolerances(job) == (R.rc.DEFAULT_RTOL, R.rc.DEFAULT_ATOL, "gate_default")
    run = R.gate_run_settings(job)
    assert (run["rtol"], run["atol"]) == (R.rc.DEFAULT_RTOL, R.rc.DEFAULT_ATOL)
    assert (run["sedml_rtol"], run["sedml_atol"]) == (1e-6, 1e-9)
    assert (run["t_start"], run["t_end"], run["n_points"]) == (0.0, 10.0, 101)


def test_gate_tolerances_take_the_per_model_tol_override():
    job = _job()
    job["overrides"] = [
        {"field": "known_artifact", "value": {"issue": None}, "reason": "x"},
        {"field": "tol", "value": {"rtol": 1e-10, "atol": 1e-16}, "reason": "ill-conditioned"},
    ]
    assert R.gate_tolerances(job) == (1e-10, 1e-16, "tol_override")
    assert R.gate_run_settings(job)["tolerance_source"] == "tol_override"


def test_gate_run_settings_integrate_from_initial_time():
    assert R.gate_run_settings(_job(initial_time=0.0, t_start=5.0))["t_start"] == 0.0
    assert R.gate_run_settings(_job(t_start=5.0))["t_start"] == 5.0


def test_rr_tolerance_rule_agrees_with_rr_run_on_the_whole_manifest():
    """Same rule, same baked overrides: what rr_run forces on both engines per job."""
    import rr_run

    _meta, rjobs = _core.read_manifest(R.HERE / "ode_jobs.json")
    by_id = {j["model_id"]: j for j in R.load_ode_jobs()}
    n_over = 0
    for rj in rjobs:
        if rj.method != "ode":
            continue
        tol = rr_run._job_overrides(rj)[0]
        expect = (tol["rtol"], tol["atol"]) if tol else (rr_run.DEFAULT_RTOL, rr_run.DEFAULT_ATOL)
        rtol, atol, src = R.gate_tolerances(by_id[rj.model_id])
        assert (rtol, atol) == expect, rj.model_id
        n_over += src == "tol_override"
    assert n_over >= 1  # the manifest carries per-model overrides, so the branch ran


def test_load_gate_outcomes_is_optional(tmp_path):
    assert R.load_gate_outcomes(tmp_path / "nope.json") == {}
    rp = tmp_path / "r.json"
    rows = [
        {"model_id": "B1", "method": "ode", "outcome": "PASS"},
        {"model_id": "B2", "method": "ssa", "outcome": "DIFF"},
    ]
    rp.write_text(json.dumps({"results": rows}))
    assert R.load_gate_outcomes(rp) == {"B1": "PASS"}


# --------------------------------------------------------------------------- #
# Issue #510 — a degenerate sample is skipped and counted, never fatal.
# --------------------------------------------------------------------------- #
_X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
_T = np.array([0.0, 1.0, 2.0])
_IND, _L = [0, 1], np.eye(2)


def _jac(y, t=0.0):
    return np.diag([-1.0, -100.0])


def _rhs(y, t=0.0):
    return -y


def test_sweep_skips_a_sample_whose_rhs_is_nonfinite_and_counts_it():
    def rhs(y, t=0.0):
        return np.array([np.nan, 0.0]) if t == 1.0 else -y

    sw = J.sweep_trajectory(_jac, rhs, _X, _T, [0, 1, 2], _IND, _L, atol=1e-12)
    assert (sw["n_used"], sw["n_skipped"], sw["n_clamped"]) == (2, 1, 0)
    assert [p["t"] for p in sw["per_time"]] == [0.0, 2.0]
    assert sw["pattern"].tolist() == [[True, False], [False, True]]
    assert all(p["ratio"] == pytest.approx(100.0) for p in sw["per_time"])


def test_sweep_skips_a_sample_whose_jacobian_is_nonfinite_though_its_rhs_is_finite():
    """The BIOMD0000000436 class: a species exactly 0 where an analytical entry is singular."""

    def jac(y, t=0.0):
        J_ = np.diag([-1.0, -100.0])
        if t == 1.0:
            J_[0, 1] = np.inf
        return J_

    sw = J.sweep_trajectory(jac, _rhs, _X, _T, [0, 1, 2], _IND, _L, atol=1e-12)
    assert (sw["n_used"], sw["n_skipped"]) == (2, 1)
    assert sw["pattern"].tolist() == [[True, False], [False, True]]  # the inf never joined it


def test_sweep_raises_when_no_sample_survives():
    def rhs(y, t=0.0):
        return np.array([np.nan, np.nan])

    with pytest.raises(J.NoFiniteSample, match="all 3 sampled"):
        J.sweep_trajectory(_jac, rhs, _X, _T, [0, 1, 2], _IND, _L, atol=1e-12)


def test_sweep_clamps_a_negative_below_atol_to_zero_before_evaluating():
    seen = []

    def rhs(y, t=0.0):
        seen.append(y.copy())
        return -y

    X = np.array([[1.0, -1e-14], [1.0, -1e-3]])
    sw = J.sweep_trajectory(_jac, rhs, X, np.array([0.0, 1.0]), [0, 1], _IND, _L, atol=1e-12)
    assert (sw["n_used"], sw["n_skipped"], sw["n_clamped"]) == (2, 0, 1)
    assert seen[0][1] == 0.0  # -1e-14 is zero to the solver
    assert seen[1][1] == -1e-3  # -1e-3 is a real value: evaluated as is
    assert X[0, 1] == -1e-14  # the trajectory itself is untouched


def test_sweep_is_the_plain_sweep_when_every_sample_is_finite():
    sw = J.sweep_trajectory(_jac, _rhs, _X, _T, [0, 1, 2], _IND, _L, atol=1e-12)
    assert (sw["n_used"], sw["n_skipped"], sw["n_clamped"]) == (3, 0, 0)
    expect = J._classify_eigs(np.linalg.eigvals(_jac(None)))
    for p in sw["per_time"]:
        assert {k: v for k, v in p.items() if k != "t"} == expect


# --------------------------------------------------------------------------- #
# Issue #512 — the analysis step reads the flag.
# --------------------------------------------------------------------------- #
def _row(model_id, status="ok", method="native_analytical", complete=True, **kw):
    r = {
        "model_id": model_id,
        "status": status,
        "N": 3,
        "density": 0.5,
        "jacobian_method": method,
        "analytical_jacobian_complete": complete,
    }
    r.update(kw)
    return r


def _swept(max_re, ratio, osc=False):
    return {
        "stiffness_ratio_max": ratio,
        "per_time": [
            {"t": 0.0, "max_re": max_re, "min_re": 1.0, "ratio": ratio, "oscillatory": osc}
        ],
    }


def test_analyze_keeps_incomplete_and_sweepless_rows_out_of_degenerate(tmp_path, capsys):
    rows = [
        _row("live_a", **_swept(1.0, 10.0)),
        _row("live_b", N=30, density=0.7, **_swept(1.0, 20.0)),
        # a report from before #512: the partial assembly, filed as incomplete — not as the
        # zero-Jacobian degenerate it looks like
        _row("partial", complete=False, density=0.0, **_swept(0.0, float("inf"))),
        # characterized from the difference quotient instead: an ordinary live row
        _row(
            "fd",
            N=300,
            method="finite_difference",
            complete=False,
            density=0.889,
            **_swept(2.0, 50.0),
        ),
        # density only (N > EIG_MAX_N, or no gate horizon): no spectrum was measured
        _row("dens_only", status="ok_density_only", N=6000, density=0.01),
        # a genuinely zero Jacobian
        _row(
            "zero",
            method="finite_difference",
            complete=False,
            density=0.0,
            **_swept(0.0, float("inf")),
        ),
    ]
    cp = tmp_path / "char.json"
    cp.write_text(json.dumps({"results": rows}))
    out = J.analyze(cp, None, None, horizons_path=tmp_path / "missing.json")
    assert out["incomplete_jacobian"] == ["partial"]
    assert out["degenerate"] == ["zero"]
    assert out["no_stiffness"] == ["dens_only"]
    grouped = sum((out["groups"][c] for c in ("sparse_stiff", "dense_stiff", "nonstiff")), [])
    assert sorted(grouped) == ["fd", "live_a", "live_b"]
    assert out["summary"]["counts"] == {
        "ok": 5,
        "degenerate": 1,
        "oscillatory": 0,
        "no_stiffness": 1,
        "live": 3,
        "incomplete_jacobian": 1,
    }
    assert out["summary"]["jacobian_method"] == {"native_analytical": 3, "finite_difference": 2}
    assert "will have no costs" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Issue #508 — the corpus model, end to end (the vendored SBML is gitignored; skip
# where it has not been materialized).
# --------------------------------------------------------------------------- #
_B338 = "BIOMD0000000338"
_B338_XML = _RR_PARITY / "models" / _B338 / f"{_B338}_url.xml"


@pytest.mark.skipif(not _B338_XML.exists(), reason=f"{_B338_XML} not materialized")
def test_issue_508_model_is_characterized_from_its_full_state():
    """57 core state entries, 55 reported species: the Jacobian hook read a 55-vector past
    its end. Now the rows are ``Result.state``, every sample is finite, and the row
    records the gate's run."""
    job = next(j for j in R.load_ode_jobs() if j["model_id"] == _B338)
    row = R.characterize_sbml(_B338, job["model"], job, gate_outcome="PASS")
    assert row["status"] == "ok", row.get("detail")
    assert (row["N"], row["n_reported_species"]) == (57, 55)
    assert row["jacobian_method"] == "native_analytical" and row["analytical_jacobian_complete"]
    assert row["n_time_points_skipped"] == 0 and row["n_time_points"] >= 1
    assert np.isfinite(row["stiffness_ratio_max"])
    assert (row["run"]["rtol"], row["run"]["atol"]) == (R.rc.DEFAULT_RTOL, R.rc.DEFAULT_ATOL)
    assert (row["run"]["t_end"], row["run"]["n_points"]) == (2.0, 101)
    assert row["gate_outcome"] == "PASS"


# --------------------------------------------------------------------------- #
# Engine: one BNGL model with and without a gate horizon.
# --------------------------------------------------------------------------- #
_BNG = _core.resolve_bng()
engine = pytest.mark.skipif(not _BNG.ok, reason=_BNG.why_not())
_FAST = "fast/rulehub/Published/Lang2024/Lang_2024.bngl"  # 73 species, seconds of netgen


@engine
def test_characterize_model_without_a_gate_horizon_is_density_only():
    row = J.characterize_model(_FAST, {}, str(_BNG.bng2_pl), timeout=600)
    assert row["status"] == "ok_density_only" and "no horizon" in row["detail"]
    assert row["run"] is None and row["gate_outcome"] is None
    assert row["jacobian_method"] in ("native_analytical", "finite_difference")
    assert 0 < row["density"] <= 1 and "stiffness_ratio_max" not in row


@engine
def test_characterize_model_runs_on_the_gate_horizon_and_records_it():
    h = {
        "t_start": 0.0,
        "t_end": 1.0,
        "n_steps": 10,
        "rtol": 1e-7,
        "atol": 1e-9,
        "outcome": "PASS",
        "source": "t",
    }
    row = J.characterize_model(_FAST, h, str(_BNG.bng2_pl), timeout=600)
    assert row["status"] == "ok", row.get("detail")
    assert row["run"] == {
        "t_start": 0.0,
        "t_end": 1.0,
        "n_steps": 10,
        "rtol": 1e-7,
        "atol": 1e-9,
        "horizon_source": "t",
        "tolerance_source": "gate_report",
    }
    assert row["gate_outcome"] == "PASS"
    assert row["n_time_points"] + row["n_time_points_skipped"] == 11  # N <= FULL_GRID_MAX_N
    assert row["per_time"][-1]["t"] == pytest.approx(1.0)
    assert row["n_reported_species"] == row["N"] == 73
