"""The RoadRunner reference archive (issue #805).

``rr_run.py --write-reference-archive`` records each job's RoadRunner side;
``--reference-archive`` compares bngsim against that record without running
RoadRunner. An entry that no longer matches its job is stale and never
compared. The end-to-end test records the committed smoke subset, then replays
it with RoadRunner blocked from importing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rr_run as R

_RR = Path(R.__file__).resolve().parent
SMOKE_JOBS = _RR / "smoke" / "ode_jobs_smoke.json"


def _spec(tmp_path: Path, **params) -> dict:
    xml = tmp_path / "m.xml"
    if not xml.exists():
        xml.write_text("<sbml>the model</sbml>")
    p = {"t_start": 0.0, "t_end": 10.0, "n_points": 11, "rtol": 1e-6, "atol": 1e-9}
    p.update(params)
    return {
        "key": "BIOMD1:ode",
        "model_id": "BIOMD1",
        "method": "ode",
        "xml": str(xml),
        "params": p,
    }


def _rr():
    t = np.linspace(0.0, 10.0, 11)
    return t, np.column_stack([np.exp(-t), 1 - np.exp(-t)]), ["A", "B"]


def test_a_recorded_reference_comes_back_as_it_was(tmp_path):
    spec = _spec(tmp_path)
    R.archive_save(tmp_path / "arch", spec, _rr(), "", {"integrate_sec": 0.01})
    rr, exc, timing, meta = R.archive_load(tmp_path / "arch", spec)
    assert exc == "" and timing == {"integrate_sec": 0.01}
    for got, want in zip(rr[:2], _rr()[:2], strict=False):
        np.testing.assert_array_equal(got, want)
    assert rr[2] == ["A", "B"]
    assert meta["model_sha256"] == R._file_sha256(spec["xml"])


def test_a_reference_that_raised_replays_its_exception(tmp_path):
    spec = _spec(tmp_path)
    R.archive_save(
        tmp_path / "arch", spec, None, "roadrunner: RuntimeError: missing a value", None
    )
    rr, exc, _timing, _meta = R.archive_load(tmp_path / "arch", spec)
    assert rr is None and exc == "roadrunner: RuntimeError: missing a value"
    # ...and classifies as the live refusal did.
    assert R.classify_reference_refusal(exc) == "overstrict_missing_value"


def test_no_entry_is_stale(tmp_path):
    rr, exc, _t, _m = R.archive_load(tmp_path / "arch", _spec(tmp_path))
    assert rr is None and exc.startswith("archive_stale") and "no entry" in exc


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"rtol": 1e-12}, "rtol"),
        ({"atol": 1e-16}, "atol"),
        ({"n_points": 101}, "n_points"),
        ({"t_end": 20.0}, "t_end"),
    ],
)
def test_a_changed_grid_or_tolerance_is_stale(tmp_path, change, field):
    """#1099's change of BIOMD0000000374's rtol is the case this exists for:
    a reference at the old tolerance is no reference for the new job."""
    R.archive_save(tmp_path / "arch", _spec(tmp_path), _rr(), "", None)
    rr, exc, _t, _m = R.archive_load(tmp_path / "arch", _spec(tmp_path, **change))
    assert rr is None and exc == f"archive_stale: the job's {field} changed"


def test_a_changed_model_file_is_stale(tmp_path):
    spec = _spec(tmp_path)
    R.archive_save(tmp_path / "arch", spec, _rr(), "", None)
    Path(spec["xml"]).write_text("<sbml>another model</sbml>")
    _rr_, exc, _t, _m = R.archive_load(tmp_path / "arch", spec)
    assert exc == "archive_stale: the job's model_sha256 changed"


def test_a_stale_row_has_its_own_refusal_class():
    exc = "roadrunner (archive): archive_stale: the job's rtol changed"
    assert R.classify_reference_refusal(exc) == "archive_stale"
    assert "archive_stale" not in R.SETTLED_REFUSALS  # it wants re-recording


def test_the_two_flags_are_exclusive(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "rr_run.py"),
            "--reference-archive",
            "a",
            "--write-reference-archive",
            "b",
        ],
        capture_output=True,
        text=True,
        cwd=_RR,
        timeout=120,
    )
    assert proc.returncode != 0 and "--write-reference-archive records one" in proc.stderr


def _run(tmp_path, *args, env=None):
    out = tmp_path / f"report{len(list(tmp_path.glob('report*')))}.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "rr_run.py"),
            "--jobs",
            str(SMOKE_JOBS),
            "--workers",
            "2",
            "--out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_RR,
        timeout=600,
        env=env,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return {r["model_id"]: r for r in json.loads(out.read_text())["results"]}


def test_a_replay_without_roadrunner_reproduces_the_live_verdicts(tmp_path):
    pytest.importorskip("roadrunner")
    pytest.importorskip("bngsim")
    if not SMOKE_JOBS.is_file():
        pytest.skip("smoke subset not in this checkout")
    arch = tmp_path / "arch"
    live = _run(tmp_path, "--write-reference-archive", str(arch))
    assert len(list(arch.glob("*.npz"))) == len(live)
    # RoadRunner cannot be imported by the replay or any worker it spawns.
    blocked = tmp_path / "no_roadrunner" / "roadrunner"
    blocked.mkdir(parents=True)
    (blocked / "__init__.py").write_text("raise ImportError('blocked for this test')\n")
    env = {
        **os.environ,
        "PYTHONPATH": f"{blocked.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    replay = _run(tmp_path, "--reference-archive", str(arch), env=env)
    assert set(replay) == set(live)
    for model_id, row in live.items():
        got = replay[model_id]
        assert (got["outcome"], got["value"], got["comment"]) == (
            row["outcome"],
            row["value"],
            row["comment"],
        ), model_id
        assert got["timing"]["roadrunner"]["replayed_from_archive"] is True
