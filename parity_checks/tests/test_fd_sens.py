"""The finite-difference sensitivity oracle (issue #804, parity_checks/fd_sens).

The step, the Richardson extrapolation, the convergence gate and the columns a
difference can judge, on synthetic data; then one real model end to end, with
the reference corrupted to show the verdict catches a wrong column, with the
analytic side corrupted toward zero to show the silent zero is judged rather
than left out, and with an unconverged, non-finite or failed reference to show
it is refused rather than blamed on the analytic side.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_FD = Path(__file__).resolve().parent.parent / "fd_sens"
# The variables that pick the repository git answers for (python/tests/_net_corpus.py).
_REPO_LOCATING = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)


def _load():
    spec = importlib.util.spec_from_file_location("fd_sens_run", _FD / "fd_sens_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load()


def test_the_step_is_relative_and_absolute_at_zero():
    assert F.fd_step(2.0) == pytest.approx(2.0 * F.FD_REL_STEP)
    assert F.fd_step(-4.0) == pytest.approx(4.0 * F.FD_REL_STEP)
    assert F.fd_step(0.0) == F.FD_REL_STEP


def test_richardson_removes_the_h_squared_term():
    true = np.array([1.5, -2.0])
    a = np.array([0.3, 0.7])
    h = 1e-2
    d_h, d_h2 = true + a * h**2, true + a * (h / 2) ** 2
    np.testing.assert_allclose(F.richardson(d_h, d_h2), true, rtol=1e-14)


def _column(*species):
    """One parameter's column, ``(n_t, n_sp)``, from per-species time series."""
    return np.stack(species, axis=-1)


def test_step_disagreement_is_on_each_columns_own_scale():
    t = np.array([0.0, 1e-9, 10.0])
    ref = _column(t)
    assert F.step_disagreement(ref, ref, np.zeros_like(ref)) == 0.0
    # A cell near zero moving by 1e-9 is nothing on a column whose peak is 10.
    gap = _column(np.array([0.0, 1e-9, 0.0]))
    assert F.step_disagreement(ref, ref, gap) == pytest.approx(1e-10)
    # Noise: the two quotients of an unresolved column.
    assert F.step_disagreement(ref, ref, _column(np.array([0.0, 6.0, 6.0]))) > 0.5
    z = np.zeros((3, 1))
    assert F.step_disagreement(z, z, z) == 0.0


def test_step_disagreement_judges_each_species_as_the_verdict_does():
    """The verdict judges each species of a parameter's column on its own peak,
    so the gate does too. Species 1 peaks at 0.1, past differ's significance
    floor of 1e-3 of the column's 10, and its gap of 1e-3 is 1% of that; beside
    species 0, the column's peak would read it as 1e-4."""
    big, small = np.array([1.0, 5.0, 10.0]), np.array([0.02, 0.05, 0.1])
    ref = _column(big, small)
    gap = _column(np.zeros(3), np.array([0.0, 0.0, 1e-3]))
    assert F.step_disagreement(ref, ref, gap) == pytest.approx(1e-2)
    # A species under the significance floor is not one the verdict counts.
    tiny = np.array([0.0, 1e-6, 2e-6])
    ref = _column(big, tiny)
    gap = _column(np.zeros(3), np.array([0.0, 0.0, 2e-6]))
    assert F.step_disagreement(ref, ref, gap) == 0.0


def test_step_disagreement_covers_the_columns_the_movement_gate_judges():
    """mm_tqssa_stiff's shape: dE/dE_0 = 1 holds still and sets the peak, so
    differ's movement gate takes its scale from a species of 1e-12 noise, which
    the peak test alone would skip, and fails it. The gate must see it first,
    unless those cells are inside a noise floor the verdict forgives anyway."""
    still = np.ones(4)
    noise = np.array([0.0, -2.8e-12, 4.4e-12, 8.7e-18])
    ref = _column(still, noise)
    gap = _column(np.zeros(4), np.array([0.0, 2.1e-12, 3.3e-12, 6.6e-18]))
    assert F.step_disagreement(ref, ref, gap) > 0.5
    settled = _column(np.zeros(4, bool), np.ones(4, bool))
    assert F.step_disagreement(ref, ref, gap, settled) == 0.0


def test_a_reference_that_is_not_finite_never_passes_the_gate():
    ref = _column(np.array([1.0, np.nan, 2.0]))
    assert F.step_disagreement(ref, ref, np.zeros_like(ref)) == float("inf")
    ok = _column(np.array([1.0, 2.0, 3.0]))
    assert F.step_disagreement(ok, ok, ref) == float("inf")
    # A non-finite analytic cell is the verdict's to fail, and sets no scale here.
    bad = _column(np.array([1.0, np.inf, 3.0]))
    assert F.step_disagreement(bad, ok, np.zeros_like(ok)) == 0.0


def test_only_columns_clearing_both_floors_are_judged():
    # (n_t, n_sp, n_p): column 0 large; column 1 under the verdict's atol/|p|;
    # column 2 over that (1e-6 at p = 1e-4) but under the difference's own noise,
    # 10·(1e-12·1e4)/(5e-8) = 2e-3, since a small parameter value makes a small step.
    analytic = np.zeros((3, 2, 3))
    analytic[..., 0] = 1.0
    analytic[..., 1] = 1e-12
    analytic[..., 2] = 1e-3
    values = [2.0, 2.0, 1e-4]
    floors = F.asens.sens_resolution_floors(values, 1e-12)
    assert floors[2] < 1e-3 < F.fd_noise_floor(1e4, 1e-4)
    judged = F.judgeable_columns(analytic, analytic, x_max=1e4, values=values, atol=1e-12)
    assert judged == [0]


def test_a_column_only_just_over_a_floor_is_not_judged():
    """A floor forgives every cell inside it, so a column at twice a floor would
    have all its disagreement forgiven (wofsy_goldstein's K11 passed a 30% error
    at 2.2x). A column is judged only at JUDGE_FACTOR (20x) over each floor."""
    values, atol, x_max = [2.0, 2.0], 1e-30, 1.0
    floor = F.fd_noise_floor(x_max, 2.0)
    analytic = np.zeros((3, 1, 2))
    analytic[..., 0] = 2.0 * floor
    analytic[..., 1] = 2.0 * F.JUDGE_FACTOR * floor
    assert F.judgeable_columns(analytic, analytic, x_max, values, atol) == [1]
    # The same for the verdict's own atol/|p| floor (nanomolar_dimer's 2.5x).
    atol = 1e-12
    vfloor = F.asens.sens_resolution_floors([2.0], atol)[0]
    analytic = np.full((3, 1, 2), 2.0 * vfloor)
    analytic[..., 1] = 2.0 * F.JUDGE_FACTOR * vfloor
    assert F.judgeable_columns(analytic, analytic, 1e-30, values, atol) == [1]


def test_a_job_can_set_its_own_atol():
    import argparse

    args = argparse.Namespace(rtol=1e-9, atol=1e-12, fd_tol=1e-3, param_cap=6, timeout=300.0)
    jobs = [
        {"model_id": "a", "path": "a.net", "t_end": 1},
        {"model_id": "b", "path": "b.net", "t_end": 1, "atol": 1e-21},
    ]
    atols = {s["model_id"]: s["atol"] for s in F.make_specs(jobs, args)}
    assert atols == {"a": 1e-12, "b": 1e-21}


def _ramp(peak, n_t=21, n_sp=2):
    """A column that rises to ``peak``, a little differently per species."""
    t = np.linspace(0.0, 1.0, n_t)
    return np.stack([peak * t ** (1 + k) for k in range(n_sp)], axis=-1)


def test_each_column_is_judged_on_its_own_scale():
    """A parameter whose sensitivities are 1e-9 of another's is still judged:
    zeroing it is a DIFF. Through one tensor-wide scale it would PASS, as the
    combined sens_verdict call shows."""
    ref = np.stack([_ramp(1.0), _ramp(1e-9)], axis=-1)
    wrong = ref.copy()
    wrong[..., 1] = 0.0
    gap = np.zeros_like(ref)
    # Floors far under both columns: a tiny state scale and atol.
    values, atol, x_max = [1.0, 1.0], 1e-30, 1e-12
    assert F.fd_noise_floor(x_max, 1.0) < 1e-10
    assert F.asens.sens_verdict(wrong, ref, param_values=values, atol=atol)["passed"]
    verdicts = F.column_verdicts(wrong, ref, gap, x_max, values, atol, [0, 1])
    assert verdicts[0]["passed"]
    assert not verdicts[1]["passed"]


def test_a_tensor_under_the_underflow_floor_is_still_judged():
    """nanomolar_dimer's shape: sensitivities near 1e-20, every cell under
    differ's absolute 1e-12 underflow floor. Scaled to its own peak, a negated
    column fails."""
    ref = _ramp(6e-20)[..., np.newaxis]
    # kdis = 4e9 on a 1.5e-9 state scale: its difference resolves 5e-22.
    assert F.fd_noise_floor(1.5e-9, 4e9) < 1e-21
    verdicts = F.column_verdicts(-ref, ref, np.zeros_like(ref), 1.5e-9, [4e9], 1e-40, [0])
    assert not verdicts[0]["passed"]


def test_a_difference_inside_the_references_noise_is_not_blamed():
    """mm_tqssa_stiff's E_0 column: dE/dE_0 = 1 holds still, and the other
    species' true sensitivity is ~1e-16, which the difference reads as ~4e-12 of
    noise, inside its 2e-11 floor at E_0 = 9000 on a state scale of 9000.
    Without that floor, differ's movement gate takes those species as the ones
    that move and fails them."""
    still = np.ones(4)
    analytic = _column(still, np.array([0.0, -6.6e-16, -1.8e-16, -1e-22]))[..., np.newaxis]
    reference = _column(still, np.array([0.0, -2.8e-12, 4.4e-12, 8.7e-18]))[..., np.newaxis]
    gap = np.zeros_like(reference)
    assert F.fd_noise_floor(9000.0, 9000.0) > 4.4e-12
    assert F.column_verdicts(analytic, reference, gap, 9000.0, [9000.0], 1e-30, [0])[0]["passed"]
    bare = F.asens.sens_verdict(analytic, reference, param_values=[9000.0], atol=1e-30)
    assert not bare["passed"]


def test_a_difference_inside_the_references_own_error_is_not_blamed():
    """The analytic column 2% off the difference, everywhere: a DIFF when the
    reference is sure of itself, forgiven when its two steps are 1% apart, since
    3x that gap covers the 2%. The gate, not the verdict, decides whether a
    reference that unsure can stand."""
    ref = _ramp(1.0)[..., np.newaxis]
    analytic = 1.02 * ref
    sure = np.zeros_like(ref)
    unsure = 0.01 * np.abs(ref)
    assert not F.column_verdicts(analytic, ref, sure, 1.0, [1.0], 1e-30, [0])[0]["passed"]
    assert F.column_verdicts(analytic, ref, unsure, 1.0, [1.0], 1e-30, [0])[0]["passed"]
    # And no further: steps 0.2% apart do not cover a 2% error.
    nearly = 0.002 * np.abs(ref)
    assert not F.column_verdicts(analytic, ref, nearly, 1.0, [1.0], 1e-30, [0])[0]["passed"]


def test_a_column_either_side_resolves_is_judged():
    """A column wrong in either direction is judged: one the difference resolves
    but the analytic run reads as zero (the silent zero), and one holding a
    non-finite analytic cell, which the verdict then fails."""
    values, atol = [2.0, 2.0, 2.0], 1e-12
    analytic = np.ones((3, 2, 3))
    reference = np.ones((3, 2, 3))
    analytic[..., 1] = 0.0
    analytic[1, 0, 2] = np.nan
    judged = F.judgeable_columns(analytic, reference, x_max=10.0, values=values, atol=atol)
    assert judged == [0, 1, 2]
    # Under both floors on both sides, a column is still left out.
    reference[..., 1] = 1e-14
    judged = F.judgeable_columns(analytic, reference, x_max=10.0, values=values, atol=atol)
    assert judged == [0, 2]


def test_a_scheduler_result_still_says_what_happened():
    spec = {"model_id": "m", "sens_method": "staggered"}
    timeout = F.to_result(spec, {"status": "timeout", "cap": 300.0, "wall": 301.2}, {})
    assert timeout.outcome == "TIMEOUT"
    assert "300" in timeout.exception
    assert timeout.wall_sec == 301.2
    dead = F.to_result(spec, {"status": "dead", "exitcode": -11}, {})
    assert dead.outcome == "EXCEPTION"
    assert "-11" in dead.exception


def _worker(spec):
    out = []

    class Q:
        def put(self, item):
            out.append(item)

    F._worker(spec, Q())
    (res,) = out
    return res


@pytest.fixture
def spec():
    pytest.importorskip("bngsim")
    jobs = json.loads((_FD / "fd_sens_jobs.json").read_text())["jobs"]
    job = next(j for j in jobs if j["model_id"] == "derived_rate_const")
    return {
        "key": "derived_rate_const|sens/staggered",
        "model_id": job["model_id"],
        "path": job["path"],
        "t_start": 0.0,
        "t_end": float(job["t_end"]),
        "n_points": int(job.get("n_points", 51)),
        "rtol": F.DEFAULT_RTOL,
        "atol": F.DEFAULT_ATOL,
        "fd_tol": 1e-3,
        "param_cap": 6,
        "sens_method": "staggered",
        "timeout": 120.0,
    }


def test_a_real_model_passes_against_its_own_differences(spec):
    res = _worker(spec)
    assert res["status"] == "pass", res
    assert res["n_columns_judged"] == len(res["params"]) == 3
    assert res["fd_step_disagreement"] <= res["fd_step_disagreement_raw"] < 1e-4
    assert res["timing"]["bngsim"]["work"]["n_steps"] > 0


def test_a_wrong_column_is_a_diff(spec, monkeypatch):
    """The reference 5% off in one column: the verdict says DIFF."""
    real = F.fd_tensor

    def off(path, params, values, job, deadline=None):
        ref, dis = real(path, params, values, job, deadline)
        ref = ref.copy()
        ref[..., 1] *= 1.05
        return ref, dis

    monkeypatch.setattr(F, "fd_tensor", off)
    res = _worker(spec)
    assert res["status"] == "diff", res
    assert res["value"] > 1e-3


def test_an_unconverged_reference_is_refused_not_blamed(spec, monkeypatch):
    """A reference lost in its step's noise: off by a third of its peak, its two
    steps half a peak apart. BAD_TEST, not a DIFF."""
    real = F.fd_tensor

    def noisy(path, params, values, job, deadline=None):
        ref, gap = real(path, params, values, job, deadline)
        ref, gap = ref.copy(), gap.copy()
        peak = np.abs(ref[..., 0]).max()
        sign = np.where(np.arange(ref.shape[0]) % 2, 1.0, -1.0)[:, np.newaxis]
        ref[..., 0] += sign * peak / 3.0
        gap[..., 0] = 0.5 * peak
        return ref, gap

    monkeypatch.setattr(F, "fd_tensor", noisy)
    res = _worker(spec)
    assert res["status"] == "bad_test", res
    assert "did not converge" in res["why"]


def test_a_reference_that_is_not_finite_is_refused_not_blamed(spec, monkeypatch):
    real = F.fd_tensor

    def broken(path, params, values, job, deadline=None):
        ref, gap = real(path, params, values, job, deadline)
        ref = ref.copy()
        ref[3, 0, 1] = np.nan
        return ref, gap

    monkeypatch.setattr(F, "fd_tensor", broken)
    res = _worker(spec)
    assert res["status"] == "bad_test", res
    assert "not finite" in res["why"]


def test_a_reference_too_slow_for_the_cap_is_the_references_failure(spec):
    """The analytic run finishes, then the plain runs would not fit in the cap:
    REFERENCE_FAILED, not a TIMEOUT scored against bngsim."""
    res = _worker({**spec, "timeout": 1e-6})
    assert res["status"] == "reference_failed", res
    assert "past the job's cap" in res["exception"]


def test_a_plain_run_that_raises_is_the_references_failure(spec, monkeypatch):
    def fails(path, params, values, job, deadline=None):
        raise RuntimeError("CVODE: too much work")

    monkeypatch.setattr(F, "fd_tensor", fails)
    res = _worker(spec)
    assert res["status"] == "reference_failed", res
    assert "too much work" in res["exception"]


def _analytic_altered(monkeypatch, alter):
    """Make every analytic run hand back ``alter(sensitivities)``."""
    import bngsim

    real = bngsim.Simulator

    class Result:
        def __init__(self, run):
            self._run = run

        def __getattr__(self, name):
            return getattr(self._run, name)

        @property
        def sensitivities(self):
            return alter(np.array(self._run.sensitivities, dtype=float))

    class Simulator(real):
        def run(self, *args, **kwargs):
            return Result(super().run(*args, **kwargs))

    monkeypatch.setattr(bngsim, "Simulator", Simulator)


@pytest.mark.parametrize(
    "wrong",
    [
        pytest.param(lambda col: np.zeros_like(col), id="zero"),
        pytest.param(lambda col: col * 1e-12, id="vanishing"),
        pytest.param(lambda col: np.full_like(col, np.nan), id="nan"),
    ],
)
def test_an_analytic_column_wrong_toward_zero_is_a_diff(spec, monkeypatch, wrong):
    """The silent zero: an analytic column that comes back zero, vanishingly
    small or NaN where the difference resolves a sensitivity is judged, and
    fails, rather than being left out of the comparison."""

    def alter(sx):
        sx[..., 1] = wrong(sx[..., 1])
        return sx

    _analytic_altered(monkeypatch, alter)
    res = _worker(spec)
    assert res["status"] == "diff", res
    assert res["n_columns_judged"] == 3, res


def test_a_declined_analytic_rhs_is_named_and_alerts(spec, monkeypatch):
    """A model on CVODES' difference quotient compares a difference with a
    difference. The row says so, and its backend fails --expect-backend cc."""
    import bngsim

    real = bngsim.Simulator

    class Simulator(real):
        has_analytic_sens_rhs = False
        sens_rhs_decline_reason = "uses unsupported construct: test"

    monkeypatch.setattr(bngsim, "Simulator", Simulator)
    res = _worker(spec)
    bn = res["timing"]["bngsim"]
    assert bn["backend"] == "cc+dq"
    assert bn["config"]["sens_rhs"] == "difference_quotient"
    assert "unsupported construct" in res["sens_rhs_decline_reason"]
    verdicts = _load_verdicts()
    row = F.to_result({"model_id": "m", "sens_method": "staggered"}, res, {}).to_dict()
    assert verdicts._core_backend(row) == "cc+dq"


def _load_verdicts():
    path = _FD.parent / "nightly" / "verdicts.py"
    spec = importlib.util.spec_from_file_location("nightly_verdicts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_manifest_names_tracked_models_and_why_some_are_left_out():
    payload = json.loads((_FD / "fd_sens_jobs.json").read_text())
    repo = _FD.parents[1]
    ids = [j["model_id"] for j in payload["jobs"]]
    assert len(ids) == len(set(ids)) >= 30
    for job in payload["jobs"]:
        assert job["path"].endswith(".net"), job
        assert job["t_end"] > 0
    for model_id, reason in payload["excluded"].items():
        assert model_id not in ids
        assert len(reason) > 40, f"{model_id}: give the measurement, not a label"
    if (repo / ".git").exists():
        import os
        import subprocess

        # An exported GIT_DIR or GIT_WORK_TREE would answer for another repository.
        env = {k: v for k, v in os.environ.items() if k not in _REPO_LOCATING}
        tracked = set(
            subprocess.run(
                ["git", "ls-files", *[j["path"] for j in payload["jobs"]]],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.split()
        )
        assert tracked == {j["path"] for j in payload["jobs"]}, "a manifest model is not tracked"
