"""CI gate for the SSA cross-engine screen's committed baseline (GH #108).

Two layers:

  * HERMETIC (always runs) — the committed ``ssa_baseline.json`` is well-formed
    and all-green under the suite scoring policy, is self-consistent under the
    diff, the diff actually catches a synthetic regression against the REAL
    baseline (so the gate can't silently pass), and the baseline was produced at
    the pinned ``ssa_spec.json`` config. No engines, no corpus — runs anywhere the
    package tests do.
  * ENGINE SMOKE (opt-in) — a 2-model live slice through ``ssa_screen.py`` diffed
    against the baseline, skipped unless RoadRunner and the biomodels SBML corpus
    are present. The full 318-model breadth run stays a manual/heavy job.

The screen + diff live outside the package under ``parity_checks/rr_parity``; they
self-bootstrap their own ``sys.path``, so putting both that dir and
``parity_checks/`` on the path is enough to import them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BNGSIM = Path(__file__).resolve().parents[2]
_RR = _BNGSIM / "parity_checks" / "rr_parity"
_PC = _RR.parent
for _p in (_PC, _RR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if not (_RR / "ssa_baseline.json").exists():  # pragma: no cover - layout guard
    pytest.skip("rr_parity SSA baseline not found", allow_module_level=True)

import diff_ssa_baseline as dsb  # noqa: E402
import ssa_attribution as sa  # noqa: E402
from _core import Outcome, read_report  # noqa: E402

BASELINE_PATH = _RR / "ssa_baseline.json"
SPEC_PATH = _RR / "ssa_spec.json"


@pytest.fixture(scope="module")
def baseline():
    return dsb.load_results(BASELINE_PATH)


# --- hermetic: baseline is well-formed ------------------------------------
def test_baseline_loads_and_is_sane(baseline):
    assert len(baseline) >= 300, "baseline unexpectedly small — truncated?"
    ids = [r.model_id for r in baseline]
    assert len(ids) == len(set(ids)), "duplicate model_id in the baseline"
    valid = {o.value for o in Outcome}
    assert all(r.outcome in valid for r in baseline)


def test_baseline_is_all_green(baseline):
    # The committed baseline must carry no unacknowledged red (a bngsim_suspect /
    # ode_level DIFF or a bngsim-side EXCEPTION). A genuine new red is either a
    # regression that should not be committed, or a deliberate acknowledgement
    # that updates this expectation.
    red = [
        (r.model_id, r.outcome, r.subclass)
        for r in baseline
        if not sa.is_expected(r.outcome, r.subclass)
    ]
    assert red == [], f"baseline has not-expected (red) rows: {red}"


def test_baseline_diffed_against_itself_is_clean(baseline):
    d = dsb.diff_reports(baseline, baseline)
    assert d["regressions"] == []
    assert d["improvements"] == []
    assert d["changed_attribution"] == []
    assert d["new_models"] == [] and d["not_in_fresh"] == []


def test_gate_catches_a_synthetic_regression(baseline):
    # Flip the first PASS model to a bngsim-suspect DIFF: the diff MUST flag it,
    # proving the gate can fail (not a silent always-green).
    fresh = list(baseline)
    victim = next(i for i, r in enumerate(fresh) if r.outcome == Outcome.PASS.value)
    mid = fresh[victim].model_id
    from _core import JobResult

    fresh[victim] = JobResult(
        model_id=mid,
        method="ssa",
        reference_engine="roadrunner",
        outcome=Outcome.DIFF.value,
        subclass="bngsim_suspect",
        comment="synthetic injected regression",
    )
    d = dsb.diff_reports(baseline, fresh)
    assert [r["model_id"] for r in d["regressions"]] == [mid]


def test_gate_ignores_a_synthetic_green_churn(baseline):
    # The same model flipped to an oracle-attributed RR-side DIFF must NOT fail.
    fresh = list(baseline)
    victim = next(i for i, r in enumerate(fresh) if r.outcome == Outcome.PASS.value)
    from _core import JobResult

    fresh[victim] = JobResult(
        model_id=fresh[victim].model_id,
        method="ssa",
        reference_engine="roadrunner",
        outcome=Outcome.DIFF.value,
        subclass="diff_not_bngsim",
    )
    assert dsb.diff_reports(baseline, fresh)["regressions"] == []


# --- hermetic: baseline matches the pinned spec ---------------------------
def test_baseline_config_matches_spec():
    spec = json.loads(SPEC_PATH.read_text())
    spec_cfg = {k: v for k, v in spec.items() if k != "_meta"}
    meta, _ = read_report(BASELINE_PATH)
    base_cfg = meta.get("config", {})
    # Every spec key the baseline records must agree (the baseline may predate a
    # later-added knob like self_z_tol, so only assert on shared keys).
    drift = {
        k: (spec_cfg[k], base_cfg[k])
        for k in spec_cfg
        if k in base_cfg and base_cfg[k] != spec_cfg[k]
    }
    assert drift == {}, f"baseline config drifted from ssa_spec.json: {drift}"


# --- opt-in engine smoke --------------------------------------------------
SMOKE_MODELS = ["BIOMD0000000617", "BIOMD0000000414"]


def _corpus_dir() -> Path:
    return Path(
        os.environ.get(
            "BIOMODELS_SBML_DIR",
            os.path.expanduser("~/Code/ssys/biomodels_batch/data/sbml_downloads"),
        )
    )


def _smoke_available() -> bool:
    try:
        import roadrunner  # noqa: F401
    except Exception:
        return False
    d = _corpus_dir()
    return all((d / f"{m}.xml").exists() for m in SMOKE_MODELS)


@pytest.mark.skipif(
    not _smoke_available(),
    reason="engine smoke needs roadrunner + the biomodels SBML corpus ($BIOMODELS_SBML_DIR)",
)
def test_engine_slice_smoke_has_no_regressions(tmp_path, baseline):
    # A live slice through the screen must reproduce the baseline outcomes for the
    # smoke models (no PASS->red). Subprocess so the spawn/engine stays out of the
    # pytest process.
    core_out = tmp_path / "ssa_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "ssa_screen.py"),
            "--corpus",
            "biomodels",
            "--models",
            ",".join(SMOKE_MODELS),
            "--n",
            "8",
            "--jobs",
            "2",
            "--timeout",
            "40",
            "--out",
            str(tmp_path / "ssa_screen.json"),
            "--core-out",
            str(core_out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert core_out.exists(), f"screen did not emit a report:\n{proc.stdout}\n{proc.stderr}"
    fresh = dsb.load_results(core_out)
    assert {r.model_id for r in fresh} == set(SMOKE_MODELS)
    d = dsb.diff_reports(baseline, fresh)
    assert d["regressions"] == [], f"engine smoke regressed: {d['regressions']}"
