"""The AMICI sensitivity reference archive (issue #805).

``amici_sens_run.py --write-reference-archive`` records each job's AMICI side;
``--reference-archive`` compares bngsim against that record without importing
AMICI. An entry that no longer matches its job is stale and never compared.
The end-to-end test records the committed smoke subset, then replays it with
AMICI blocked from importing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import amici_sens_run as A
import numpy as np
import pytest

_AM = Path(A.__file__).resolve().parent
SMOKE_JOBS = _AM.parent / "rr_parity" / "smoke" / "ode_jobs_smoke.json"


def _spec(tmp_path: Path, **change) -> dict:
    xml = tmp_path / "m.xml"
    if not xml.exists():
        xml.write_text("<sbml>the model</sbml>")
    p = {"t_start": 0.0, "t_end": 10.0, "n_points": 11, "rtol": 1e-6, "atol": 1e-9}
    p.update({k: v for k, v in change.items() if k in p})
    spec = {
        "key": "BIOMD1:fsa",
        "model_id": "BIOMD1",
        "method": "fsa",
        "xml": str(xml),
        "params": p,
        "max_steps": 10000,
        "param_cap": 0,
        "param_budget": 50,
    }
    spec.update({k: v for k, v in change.items() if k not in p})
    return spec


def _am():
    t = np.linspace(0.0, 10.0, 11)
    x = np.column_stack([np.exp(-t), 1 - np.exp(-t)])
    sx = np.stack([x * 0.5, x * 2.0], axis=1)
    return t, x, sx, ["A", "B"]


def test_a_recorded_reference_comes_back_as_it_was(tmp_path):
    spec = _spec(tmp_path)
    record = {**A.archive_identity(spec), "amici_free_ids": ["k1", "k2"], "shared_ids": ["k1"]}
    A.archive_save(tmp_path / "arch", spec, record, _am())
    got, am = A.archive_load(tmp_path / "arch", spec)
    assert got == record
    for g, w in zip(am[:3], _am()[:3], strict=True):
        np.testing.assert_array_equal(g, w)
    assert am[3] == ["A", "B"]


def test_a_reference_that_failed_has_no_tensors(tmp_path):
    spec = _spec(tmp_path)
    record = {
        **A.archive_identity(spec),
        "build_error": {"text": "amici: build failed", "cls": "amici:build", "full": ""},
    }
    A.archive_save(tmp_path / "arch", spec, record)
    got, am = A.archive_load(tmp_path / "arch", spec)
    assert got["build_error"]["text"] == "amici: build failed" and am is None


def test_no_entry_loads_as_none(tmp_path):
    assert A.archive_load(tmp_path / "arch", _spec(tmp_path)) == (None, None)


@pytest.mark.parametrize(
    "change",
    [
        {"rtol": 1e-12},
        {"atol": 1e-16},
        {"n_points": 101},
        {"t_end": 20.0},
        {"t_start": 1.0},
        {"max_steps": 500},
        {"param_cap": 3},
        {"param_budget": 10},
    ],
)
def test_every_input_of_the_job_is_in_its_identity(tmp_path, change):
    """A changed field makes the recorded identity differ, which the worker
    turns into an ``archive_stale`` row rather than a comparison."""
    was = A.archive_identity(_spec(tmp_path))
    now = A.archive_identity(_spec(tmp_path, **change))
    assert [k for k in was if was[k] != now[k]] == list(change)


def test_a_changed_model_file_changes_the_identity(tmp_path):
    spec = _spec(tmp_path)
    was = A.archive_identity(spec)
    Path(spec["xml"]).write_text("<sbml>another model</sbml>")
    assert A.archive_identity(spec)["model_sha256"] != was["model_sha256"]


def test_a_stale_row_is_a_reference_failure_with_its_own_refusal():
    res = A._stale({"status": "ok"}, "the job's rtol changed")
    assert res["status"] == "reference_failed"
    assert res["reference_refusal"] == "archive_stale"
    assert res["exception"] == "amici (archive): archive_stale: the job's rtol changed"


def test_the_two_flags_are_exclusive():
    proc = subprocess.run(
        [
            sys.executable,
            str(_AM / "amici_sens_run.py"),
            "--reference-archive",
            "a",
            "--write-reference-archive",
            "b",
        ],
        capture_output=True,
        text=True,
        cwd=_AM,
        timeout=120,
    )
    assert proc.returncode != 0 and "--write-reference-archive records one" in proc.stderr


def _run(tmp_path, *args, env=None):
    out = tmp_path / f"report{len(list(tmp_path.glob('report*')))}.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_AM / "amici_sens_run.py"),
            "--manifest",
            str(SMOKE_JOBS),
            "--workers",
            "2",
            "--out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_AM,
        timeout=1800,
        env=env,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    rows = json.loads(out.read_text())["results"]
    return {(r["model_id"], r["method"]): r for r in rows}


def test_a_replay_without_amici_reproduces_the_live_verdicts(tmp_path):
    pytest.importorskip("amici")
    pytest.importorskip("bngsim")
    if not SMOKE_JOBS.is_file():
        pytest.skip("smoke subset not in this checkout")
    arch = tmp_path / "arch"
    live = _run(tmp_path, "--write-reference-archive", str(arch))
    assert len(list(arch.glob("*.npz"))) == len(live)
    # AMICI cannot be imported by the replay or any worker it spawns.
    blocked = tmp_path / "no_amici" / "amici"
    blocked.mkdir(parents=True)
    (blocked / "__init__.py").write_text("raise ImportError('blocked for this test')\n")
    env = {
        **os.environ,
        "PYTHONPATH": f"{blocked.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    replay = _run(tmp_path, "--reference-archive", str(arch), env=env)
    assert set(replay) == set(live)
    for key, row in live.items():
        got = replay[key]
        assert (got["outcome"], got["value"], got["comment"]) == (
            row["outcome"],
            row["value"],
            row["comment"],
        ), key
        if got["timing"].get("amici"):
            assert got["timing"]["amici"]["replayed_from_archive"] is True
    stale = _run(tmp_path, "--reference-archive", str(arch), "--rtol", "1e-7", env=env)
    assert {r["reference_refusal"] for r in stale.values()} == {"archive_stale"}
