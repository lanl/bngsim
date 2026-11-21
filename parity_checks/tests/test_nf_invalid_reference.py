"""#1818: an NF model whose legacy NFsim reference cannot be compared against.

Legacy NFsim v1.14.3 stops firing partway through every run of
AMPK_MTORC1_ULK1_v2, and bngsim-NF did too until #1821. The nightly's PASS on
that row compared two frozen ensembles. ``NF_INVALID_REF`` records such a model:
its row is REFERENCE_FAILED with ``reference_refusal="known_invalid"`` and the
reason, and neither engine runs. These tests pin the catalogue, the row and the
worker's early return without running any engine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bng_stoch_run as bsr  # noqa: E402
from _core import Outcome  # noqa: E402

AMPK = "AMPK_MTORC1_ULK1_v2"


def _nf_job_stems() -> dict[str, list[str]]:
    jobs = json.loads((HERE.parent / "bng_parity" / "jobs.json").read_text())
    jobs = jobs["jobs"] if isinstance(jobs, dict) else jobs
    stems: dict[str, list[str]] = {}
    for job in jobs:
        if job.get("method") == "stochastic" and "nf" in (job["params"].get("methods") or []):
            stems.setdefault(Path(job["model_id"]).stem, []).append(job["model_id"])
    return stems


def test_ampk_is_catalogued_with_the_freeze_and_its_evidence():
    art = bsr.NF_INVALID_REF.get(AMPK)
    assert art is not None
    assert art["issue"] == "#1818"
    for fact in ("stops firing", "t = 504-540", "#1821", "two frozen ensembles", "RuleMonkey"):
        assert fact in art["reason"], fact


def test_every_entry_names_one_nf_job_and_is_not_also_a_known_ref():
    """A stale entry, or one that matches two models, would hide a row silently."""
    stems = _nf_job_stems()
    for stem in bsr.NF_INVALID_REF:
        assert len(stems.get(stem, [])) == 1, (stem, stems.get(stem))
        assert stem not in bsr.NF_KNOWN_REF, stem


def test_the_row_is_reference_failed_and_its_note_says_why():
    row = bsr.known_invalid_reference("nf", AMPK)
    assert row is not None
    assert bsr._OUTCOME[row["status"]] is Outcome.REFERENCE_FAILED
    assert row["reference_refusal"] == "known_invalid"
    assert row["value"] is None
    # A verdict file keeps the first 200 characters of the exception as the note.
    assert "known_invalid" in row["exception"][:200] and "#1818" in row["exception"][:200]
    assert "Neither engine ran" in row["exception"][:200]
    assert bsr.NF_INVALID_REF[AMPK]["reason"] in row["comment"]


def test_only_a_catalogued_model_on_the_nf_track_is_refused():
    assert bsr.known_invalid_reference("ssa", AMPK) is None
    assert bsr.known_invalid_reference("nf", "some_other_model") is None


@pytest.mark.parametrize("replay", [False, True])
def test_the_worker_runs_neither_engine_nor_reads_the_archive(tmp_path, monkeypatch, replay):
    def must_not_run(*args, **kwargs):
        raise AssertionError("an engine, a build or the archive was reached")

    monkeypatch.setattr(bsr, "Legacy", must_not_run)
    for name in ("generate_xml", "generate_network", "bn_nf_xml", "nfsim_run"):
        monkeypatch.setattr(bsr.bc, name, must_not_run)

    bngl = tmp_path / f"{AMPK}.bngl"
    bngl.write_text("begin model\nend model\n")
    spec = {
        "key": f"{AMPK}|stochastic",
        "model_id": f"original/bngl_models/my_models/ode/{AMPK}.bngl",
        "method": "stochastic",
        "metric": "mean_zscore",
        "tol": 5.0,
        "track": "nf",
        "bngl": str(bngl),
        "legacy_bin": None,
        "legacy_label": "NFsim",
        "reference_archive": str(tmp_path / "archive") if replay else None,
    }
    out: list[dict] = []

    class Q:
        def put(self, item):
            out.append(item)

    bsr._worker(spec, Q())
    assert len(out) == 1
    assert out[0]["status"] == "reference_failed"
    assert out[0]["reference_refusal"] == "known_invalid"
    assert out[0]["model_id"] == spec["model_id"]
