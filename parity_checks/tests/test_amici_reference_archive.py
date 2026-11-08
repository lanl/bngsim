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
from test_sens_step_budget import _MINIMAL_SBML

_AM = Path(A.__file__).resolve().parent
SMOKE_JOBS = _AM.parent / "rr_parity" / "smoke" / "ode_jobs_smoke.json"


def _spec(tmp_path: Path, **change) -> dict:
    xml = tmp_path / "m.xml"
    if not xml.exists():
        xml.write_text("<sbml>the model</sbml>")
    p = {"t_start": 0.0, "t_end": 10.0, "n_points": 11, "rtol": 1e-6, "atol": 1e-9}
    params = {*p, "initial_time"}
    p.update({k: v for k, v in change.items() if k in params})
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
    spec.update({k: v for k, v in change.items() if k not in params})
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
    "change,keys",
    [
        ({"rtol": 1e-12}, ["rtol"]),
        ({"atol": 1e-16}, ["atol"]),
        ({"n_points": 101}, ["n_points"]),
        ({"t_end": 20.0}, ["t_end"]),
        # Where both runs start (issue #1564): the job's SED-ML initial_time,
        # or, in a job file predating the field, its t_start.
        ({"initial_time": 1.0}, ["t_int"]),
        ({"t_start": 1.0}, ["t_int"]),
        ({"max_steps": 500}, ["max_steps"]),
        ({"param_cap": 3}, ["param_cap"]),
        ({"param_budget": 10}, ["param_budget"]),
    ],
)
def test_every_input_of_the_job_is_in_its_identity(tmp_path, change, keys):
    """A changed field makes the recorded identity differ, which the worker
    turns into an ``archive_stale`` row rather than a comparison."""
    was = A.archive_identity(_spec(tmp_path))
    now = A.archive_identity(_spec(tmp_path, **change))
    assert [k for k in was if was[k] != now[k]] == keys


def test_an_entry_recorded_from_the_output_start_is_stale(tmp_path):
    """An archive recorded before #1564 started AMICI at the outputStartTime and
    holds ``t_start``; the identity now holds where the run starts, ``t_int``,
    so such an entry is refused, not replayed against a run from the initial
    time. On a job whose window starts after the model does, the two differ."""
    spec = _spec(tmp_path, initial_time=0.0, t_start=120.0)
    now = A.archive_identity(spec)
    assert now["t_int"] == 0.0 and "t_start" not in now
    assert now["format"] == 3
    old = {**{k: v for k, v in now.items() if k != "t_int"}, "format": 2, "t_start": 120.0}
    assert sorted(k for k in now if old.get(k) != now[k]) == ["format", "t_int"]


def test_the_output_start_alone_is_not_in_the_identity(tmp_path):
    """Neither run reads the outputStartTime any more: both start at the initial
    time and the compared grid begins there."""
    was = A.archive_identity(_spec(tmp_path, initial_time=0.0, t_start=0.0))
    now = A.archive_identity(_spec(tmp_path, initial_time=0.0, t_start=5.0))
    assert was == now


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


def test_keys_that_sanitize_alike_get_their_own_files(tmp_path):
    assert A._archive_file(tmp_path, "a/b:sens:fsa") != A._archive_file(tmp_path, "a_b:sens:fsa")


class _Queue(list):
    put = list.append


def _minimal_spec(tmp_path: Path) -> dict:
    xml = tmp_path / "m.xml"
    xml.write_text(_MINIMAL_SBML)
    return {
        "key": "M:sens:staggered",
        "model_id": "M",
        "method": "sens/staggered",
        "sens_method": "staggered",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": str(xml),
        "params": {"t_start": 0.0, "t_end": 10.0, "n_points": 11, "rtol": 1e-9, "atol": 1e-12},
        "param_cap": 20,
        "param_budget": 20_000,
        "max_steps": 10_000,
        "config_env": {},
        "reference_archive": str(tmp_path / "arch"),
    }


def _am_one_param():
    t, x, _sx, names = _am()
    return t, x, x[:, :, None] * 0.5, names


def _replay(monkeypatch, spec, *, bn_sens):
    """Drive the worker in-process on a replay of the minimal model: building or
    running AMICI fails the test, and bngsim's sensitivity call is ``bn_sens``.
    The model load and the shared-list negotiation are bngsim's own."""
    monkeypatch.setattr(A.asens, "measure_warmup", lambda include_amici=True: {})
    monkeypatch.setattr(
        A.asens, "amici_free_parameter_ids", lambda xml: pytest.fail("AMICI built on a replay")
    )
    monkeypatch.setattr(
        A.asens, "amici_sens", lambda *a, **k: pytest.fail("AMICI ran on a replay")
    )
    monkeypatch.setattr(
        A.asens, "shared_sensitivity_params", lambda *a, **k: (["k1"], {"k1": "k1"}, 1)
    )
    monkeypatch.setattr(A.asens, "bn_sens", bn_sens)
    q = _Queue()
    A._worker(spec, q)
    (res,) = q
    return res


def _bngsim_must_not_run(*a, **k):
    pytest.fail("bngsim's sensitivity run ran for a row that is never compared")


def test_an_unreadable_entry_is_refused_rather_than_raised(tmp_path, monkeypatch):
    """It used to raise out of the worker, which died, and the row was EXCEPTION
    blamed on bngsim."""
    spec = _minimal_spec(tmp_path)
    path = A._archive_file(spec["reference_archive"], spec["key"])
    path.parent.mkdir()
    path.write_bytes(b"not an npz")
    res = _replay(monkeypatch, spec, bn_sens=_bngsim_must_not_run)
    assert (res["status"], res["reference_refusal"]) == ("reference_failed", "archive_unreadable")
    assert res["exception"].startswith("amici (archive): archive_unreadable: ")


def test_a_run_recorded_short_of_amicis_sensitivities_is_refused_before_bngsim(
    tmp_path, monkeypatch
):
    """The recorded run stopped once it had AMICI's ids (a bngsim-params refusal,
    no shared parameter); the replay now negotiates a list, so it has nothing to
    compare against, and finds that out before bngsim's sensitivity run."""
    spec = _minimal_spec(tmp_path)
    A.archive_save(
        spec["reference_archive"], spec, {**A.archive_identity(spec), "amici_free_ids": ["k1"]}
    )
    res = _replay(monkeypatch, spec, bn_sens=_bngsim_must_not_run)
    assert res["reference_refusal"] == "archive_stale"
    assert res["exception"].endswith("the recorded run stopped before AMICI's sensitivities")


def test_a_replay_scores_the_record_and_names_the_recorded_version(tmp_path, monkeypatch):
    spec = _minimal_spec(tmp_path)
    record = {
        **A.archive_identity(spec),
        "reference_version": "1.0.1",
        "amici_free_ids": ["k1"],
        "shared_ids": ["k1"],
        "timing": {"integrate_sec": 0.01},
    }
    A.archive_save(spec["reference_archive"], spec, record, _am_one_param())
    res = _replay(monkeypatch, spec, bn_sens=lambda *a, **k: (*_am_one_param(), {}))
    assert res["status"] == "pass" and res["reference_version"] == "1.0.1"
    assert res["timing"]["amici"]["replayed_from_archive"] is True
    assert res["timing"]["amici"]["reference_version"] == "1.0.1"


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
