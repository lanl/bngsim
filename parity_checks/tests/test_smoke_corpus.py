"""Hermetic smoke test for the two fetched ODE corpora (rr_parity + biomodels).

Both suites' full corpora are gitignored and reconstructed by a pinned ``fetch.py``
(~300 MB + ~270 MB). To prove the *pipeline* still runs end-to-end without that
download, a tiny provenance-pinned subset (6 small BioModels) is committed beside
each suite:

  * ``rr_parity/smoke/``  -- models/<id>/{*.xml,*.sedml} + ode_jobs_smoke.json
  * ``benchmarks/suites/biomodels/smoke/`` -- sbml/<id>.xml + manifest_smoke.csv

Two layers of assertion:

  * **Provenance gate** (always runs; pure hashing, no engine): every committed
    smoke byte's sha256 matches its committed manifest row -- so the subset *is*
    the pinned corpus, and a silent edit / drift fails loudly.
  * **Pipeline gate** (needs libroadrunner + bngsim): each suite's real runner
    drives the bngsim-vs-RoadRunner comparison over the committed subset and every
    model comes back green -- rr_parity PASS, biomodels load+simulate+accuracy.

The subset was curated from the committed manifests: small (fast), pinned (in the
sha256 manifest), verified-green through the actual pipeline, and spanning a
feature or two (BIOMD1 carries events -> the biomodels events split / event path;
BIOMD5 is rr_parity's ``figure_sedml`` horizon tier, the rest ``template_sedml``).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_PC = Path(__file__).resolve().parent.parent  # parity_checks/
_RR = _PC / "rr_parity"
_BM = _PC.parent / "benchmarks" / "suites" / "biomodels"

# The curated subset — same ids for both suites (their bytes differ: temp-biomodels
# SBML for rr_parity vs BioModels-REST SBML for biomodels, a nice thing to show).
SMOKE_IDS = [
    "BIOMD0000000001",  # 12sp/17rx, has events
    "BIOMD0000000003",  # 3sp/7rx
    "BIOMD0000000004",  # 5sp/7rx
    "BIOMD0000000005",  # 9sp/9rx, rr_parity figure_sedml tier
    "BIOMD0000000006",  # 4sp/3rx
    "BIOMD0000000010",  # 8sp/10rx
]

_SUBPROCESS_TIMEOUT = 600  # generous; the whole subset runs in seconds warm


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


needs_engines = pytest.mark.skipif(
    not (_have("roadrunner") and _have("bngsim")),
    reason="needs libroadrunner + bngsim (install bngsim[roadrunner])",
)


# --------------------------------------------------------------------------- #
# Provenance gate — the committed subset IS the pinned corpus (no engine).
# --------------------------------------------------------------------------- #
def _rr_smoke_jobs() -> list[dict]:
    return json.loads((_RR / "smoke" / "ode_jobs_smoke.json").read_text())["jobs"]


def _rr_pin_index() -> dict[str, str]:
    """id ('<model_id>/<basename>') -> sha256, from the committed temp-biomodels manifest."""
    recs = json.loads((_RR / "temp_biomodels_manifest.json").read_text())["records"]
    return {r["id"]: r["sha256"] for r in recs}


def test_rr_smoke_subset_covers_the_curated_ids():
    jobs = {j["model_id"] for j in _rr_smoke_jobs()}
    assert jobs == set(SMOKE_IDS)


@pytest.mark.parametrize("job", _rr_smoke_jobs(), ids=lambda j: j["model_id"])
def test_rr_smoke_bytes_match_temp_biomodels_pin(job):
    """Each committed rr_parity smoke SBML + SED-ML sha256 == its manifest pin."""
    pin = _rr_pin_index()
    mid = job["model_id"]
    model_rel = Path(job["model"])
    assert model_rel.parts[0] == "smoke", "smoke jobs must point at the committed smoke tree"
    sbml = _RR / model_rel
    sedml = _RR / "smoke" / "models" / mid / job["params"]["sedml"]
    for f, key in ((sbml, f"{mid}/{sbml.name}"), (sedml, f"{mid}/{sedml.name}")):
        assert f.is_file(), f"missing committed smoke file {f}"
        assert key in pin, f"{key} not in temp_biomodels_manifest.json"
        assert _sha(f) == pin[key], f"{key} drifted from the pinned sha256"


def _bm_smoke_rows() -> list[dict]:
    path = _BM / "smoke" / "manifest_smoke.csv"
    with open(path) as fh:
        first = fh.readline()
        if not first.startswith("#"):
            fh.seek(0)
        return list(csv.DictReader(fh))


def _bm_provenance_index() -> dict[str, str]:
    recs = json.loads((_BM / "biomodels_provenance.json").read_text())["records"]
    return {r["id"]: r["sha256"] for r in recs}


def test_bm_smoke_subset_covers_the_curated_ids():
    rows = {r["model_id"] for r in _bm_smoke_rows()}
    assert rows == set(SMOKE_IDS)


@pytest.mark.parametrize("row", _bm_smoke_rows(), ids=lambda r: r["model_id"])
def test_bm_smoke_bytes_match_biomodels_pin(row):
    """Each committed biomodels smoke SBML sha256 == manifest_smoke.csv == provenance pin."""
    prov = _bm_provenance_index()
    mid = row["model_id"]
    assert row["verdict"] == "keep"
    sbml = _BM / "smoke" / "sbml" / f"{mid}.xml"
    assert sbml.is_file(), f"missing committed smoke SBML {sbml}"
    got = _sha(sbml)
    assert got == row["sha256"], f"{mid} drifted from manifest_smoke.csv sha256"
    assert mid in prov, f"{mid} not in biomodels_provenance.json"
    assert got == prov[mid], f"{mid} manifest_smoke.csv and provenance sha256 disagree"


# --------------------------------------------------------------------------- #
# Pipeline gate — the real runners go green on the committed subset.
# --------------------------------------------------------------------------- #
@needs_engines
def test_rr_parity_pipeline_all_pass(tmp_path):
    """rr_run.py drives bngsim-vs-RoadRunner on the smoke jobs; all PASS, no fetch."""
    out = tmp_path / "report_smoke.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "rr_run.py"),
            "--jobs",
            "smoke/ode_jobs_smoke.json",
            "--workers",
            "2",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert proc.returncode == 0, f"rr_run.py failed:\n{proc.stderr}\n{proc.stdout}"
    results = json.loads(out.read_text())["results"]
    outcomes = {r["model_id"]: r["outcome"] for r in results}
    assert set(outcomes) == set(SMOKE_IDS)
    bad = {m: o for m, o in outcomes.items() if o != "PASS"}
    assert not bad, f"non-PASS outcomes: {bad}\n{proc.stdout}"


@needs_engines
def test_biomodels_pipeline_all_green(tmp_path):
    """run.py drives bngsim + RoadRunner on the smoke subset; all load, simulate, agree."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_BM / "run.py"),
            "--manifest",
            str(_BM / "smoke" / "manifest_smoke.csv"),
            "--sbml-dir",
            str(_BM / "smoke" / "sbml"),
            "--engines",
            "roadrunner,bngsim",
            "--results-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert proc.returncode == 0, f"run.py failed:\n{proc.stderr}\n{proc.stdout}"
    with open(tmp_path / "coverage.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["model_id"] for r in rows} == set(SMOKE_IDS)
    for row in rows:
        for col in (
            "rr_load_ok",
            "rr_sim_ok",
            "bngsim_load_ok",
            "bngsim_sim_ok",
            "bngsim_acc_ok",
        ):
            assert row[col] == "1", (
                f"{row['model_id']} {col}={row[col]!r} (error: {row.get('bngsim_error')})"
            )
