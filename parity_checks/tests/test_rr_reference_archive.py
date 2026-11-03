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
import time
import types
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


def test_an_unreadable_entry_is_refused_rather_than_raised(tmp_path):
    spec = _spec(tmp_path)
    path = R._archive_file(tmp_path / "arch", spec["key"])
    path.parent.mkdir()
    path.write_bytes(b"not an npz")
    rr, exc, _t, _m = R.archive_load(tmp_path / "arch", spec)
    assert rr is None and exc.startswith("archive_unreadable: ")
    assert R.classify_reference_refusal(f"roadrunner (archive): {exc}") == "archive_unreadable"
    assert "archive_unreadable" not in R.SETTLED_REFUSALS


def test_keys_that_sanitize_alike_get_their_own_files(tmp_path):
    assert R._archive_file(tmp_path, "a/b:ode") != R._archive_file(tmp_path, "a_b:ode")


class _Queue(list):
    put = list.append


def _replay(tmp_path, monkeypatch, spec, *, bn):
    """Drive the worker in-process on a replay, with RoadRunner's own call made
    to fail the test and bngsim's answered by ``bn``."""
    monkeypatch.setattr(R.rc, "measure_warmup", lambda include_roadrunner=True: {})
    monkeypatch.setattr(R.rc, "rr_ode", lambda *a, **k: pytest.fail("RoadRunner ran on a replay"))
    monkeypatch.setattr(R.rc, "bn_ode", bn)
    q = _Queue()
    R._worker(
        {**spec, "metric": "max_rel_err", "tol": 1e-4, "reference_archive": tmp_path / "arch"}, q
    )
    (res,) = q
    return res


@pytest.mark.parametrize("kind", ["archive_stale", "archive_unreadable"])
def test_an_unusable_entry_is_refused_before_bngsim_runs(tmp_path, monkeypatch, kind):
    """A bngsim raise beside an unusable entry used to make the row BAD_TEST,
    "neither engine ran", with the refusal lost; the row is never compared, so
    bngsim is not run at all."""
    R.archive_save(tmp_path / "arch", _spec(tmp_path), _rr(), "", None)
    spec = _spec(tmp_path, rtol=1e-12)
    if kind == "archive_unreadable":
        spec = _spec(tmp_path)
        R._archive_file(tmp_path / "arch", spec["key"]).write_bytes(b"truncated")
    res = _replay(
        tmp_path,
        monkeypatch,
        spec,
        bn=lambda *a, **k: pytest.fail("bngsim ran for a row that is never compared"),
    )
    assert (res["status"], res["archive_refusal"]) == ("reference_failed", kind)
    assert R.classify_reference_refusal(res["exception"]) == kind


def test_a_replay_scores_the_record_and_names_the_recorded_version(tmp_path, monkeypatch):
    monkeypatch.setattr(R.versions, "roadrunner_version", lambda: "2.7.0")
    R.archive_save(tmp_path / "arch", _spec(tmp_path), _rr(), "", {"integrate_sec": 0.01})
    monkeypatch.setattr(R.versions, "roadrunner_version", lambda: "9.9.9")  # the local one
    res = _replay(tmp_path, monkeypatch, _spec(tmp_path), bn=lambda *a, **k: (*_rr(), {}))
    assert res["status"] == "pass" and res["reference_version"] == "2.7.0"
    assert res["timing"]["roadrunner"]["replayed_from_archive"] is True
    assert res["timing"]["roadrunner"]["reference_version"] == "2.7.0"


def test_an_archive_refusal_is_judged_by_no_override(tmp_path, monkeypatch):
    """An adjudication would turn archive_stale into a settled class, and an
    invalid-reference entry would call itself stale and ask to be pruned; a row
    that had no reference at all can do neither. The report also names the
    RoadRunner version the archive recorded, not a local one."""
    adjudicated, invalid = "BIOMD0000000003", "BIOMD0000000004"

    def overrides(job):
        if job.model_id == adjudicated:
            return None, None, None, ("confirm", "an independent oracle agreed", "#117")
        if job.model_id == invalid:
            return None, None, ("RoadRunner answers NaN", "#485"), None
        return None, None, None, None

    def schedule(specs, worker, **kw):
        rows = []
        for s in specs:
            row = {k: s[k] for k in ("key", "model_id", "method", "metric", "tol")}
            if s["model_id"] == "BIOMD0000000001":
                row.update(status="pass", value=0.0, reference_version="2.7.0")
            else:
                row.update(
                    status="reference_failed",
                    exception="roadrunner (archive): archive_stale: the job's rtol changed",
                    archive_refusal="archive_stale",
                )
            rows.append(row)
        return rows

    monkeypatch.setattr(R, "_job_overrides", overrides)
    monkeypatch.setattr(R.rc, "schedule", schedule)
    out = tmp_path / "report.json"
    argv = ["rr_run.py", "--jobs", str(SMOKE_JOBS), "--reference-archive", str(tmp_path / "a")]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])
    assert R.main() == 0
    report = json.loads(out.read_text())
    rows = {r["model_id"]: r for r in report["results"]}
    for model_id in (adjudicated, invalid):
        assert rows[model_id]["outcome"] == "REFERENCE_FAILED"
        assert rows[model_id]["reference_refusal"] == "archive_stale"
        assert "adjudicated" not in rows[model_id]["comment"]
        assert "STALE" not in rows[model_id]["comment"]
    assert rows["BIOMD0000000001"]["versions"]["roadrunner"] == "2.7.0"
    assert rows[adjudicated]["versions"]["roadrunner"] is None
    assert report["_meta"]["versions"]["roadrunner"] == "2.7.0"


def _engines(monkeypatch, *, rr_sec=0.0, bn=None) -> list[str]:
    """Answer both engines in-process: RoadRunner with ``_rr()`` after ``rr_sec``,
    bngsim with ``bn`` (default: the same trajectory after 0.05 s). Returns the
    list each call appends its engine to."""
    calls: list[str] = []

    def rr_ode(*a, **k):
        calls.append("roadrunner")
        time.sleep(rr_sec)
        return (*_rr(), {"integrate_sec": 0.01})

    def bn_ode(*a, **k):
        calls.append("bngsim")
        time.sleep(0.05)
        return (*_rr(), {"integrate_sec": 0.02})

    monkeypatch.setattr(R.rc, "measure_warmup", lambda include_roadrunner=True: {})
    monkeypatch.setattr(R.rc, "set_rr_quiet", lambda: None)
    monkeypatch.setattr(R.rc, "rr_ode", rr_ode)
    monkeypatch.setattr(R.rc, "bn_ode", bn or bn_ode)
    return calls


def _work(spec: dict, **extra) -> dict:
    q = _Queue()
    R._worker({**spec, "metric": "max_rel_err", "tol": 1e-4, "cap": 300.0, **extra}, q)
    (res,) = q
    return res


def test_a_recording_saves_the_reference_before_bngsim_starts(tmp_path, monkeypatch):
    """Issue #1290: saved after bngsim, a job the two engines together carried past
    the cap left no entry, and every replay refused a job it could have compared.
    RoadRunner now runs first and its entry is on disk before bngsim starts, so a
    kill anywhere in bngsim's side keeps it."""
    arch = tmp_path / "arch"
    spec = _spec(tmp_path)

    def bn_ode(*a, **k):
        assert R._archive_file(arch, spec["key"]).is_file(), "bngsim ran before the save"
        raise KeyboardInterrupt  # the parent's kill, as far as this worker is concerned

    _engines(monkeypatch, bn=bn_ode)
    with pytest.raises(KeyboardInterrupt):
        _work(spec, write_reference_archive=arch)
    rr, exc, timing, _meta = R.archive_load(arch, spec)
    assert exc == "" and timing == {"integrate_sec": 0.01}
    np.testing.assert_array_equal(rr[1], _rr()[1])


def test_a_recording_scores_the_job_as_a_live_run_does(tmp_path, monkeypatch):
    """Only the order changes: the row is the one a live run gives, with both
    engines' timing, and a wall that counts both. Each engine runs once."""
    calls = _engines(monkeypatch, rr_sec=0.05)
    live = _work(_spec(tmp_path))
    assert calls == ["bngsim", "roadrunner"]
    calls.clear()
    recorded = _work(_spec(tmp_path), write_reference_archive=tmp_path / "arch")
    assert calls == ["roadrunner", "bngsim"]
    for row in (live, recorded):
        assert row.pop("wall_sec") >= 0.1  # 0.05 s each
    assert recorded == live and live["status"] == "pass"
    assert live["timing"]["roadrunner"] == {"integrate_sec": 0.01}
    assert live["timing"]["bngsim"] == {"integrate_sec": 0.02}


def test_a_recording_past_the_cap_is_a_timeout_with_its_reference_saved(tmp_path, monkeypatch):
    """A recording is killed only at twice the cap, so it can save the entry; the
    row is still TIMEOUT past the cap itself, as a live night would make it, not a
    PASS no other night would give. On a clock the engines advance, 0.65 s against
    a 0.5 s cap: past the cap, short of twice it, and RoadRunner alone is past it,
    so judging by twice the cap or by bngsim's time alone would pass the row."""
    now = [0.0]

    def run(sec):
        def engine(*a, **k):
            now[0] += sec
            return (*_rr(), {})

        return engine

    _engines(monkeypatch, bn=run(0.05))
    monkeypatch.setattr(R.rc, "rr_ode", run(0.6))
    monkeypatch.setattr(R, "time", types.SimpleNamespace(perf_counter=lambda: now[0]))
    spec = _spec(tmp_path)
    res = _work(spec, write_reference_archive=tmp_path / "arch", cap=0.5)
    assert (res["status"], res["cap"]) == ("timeout", 0.5)
    assert "value" not in res and "timing" not in res
    assert res["comment"] == (
        "ran past the 0.5s wall cap (a recording runs to 1.0s to save its reference)"
    )
    assert R.archive_load(tmp_path / "arch", spec)[1] == ""
    # A live run is killed by the parent at the cap; the worker leaves it be.
    assert _work(spec, cap=0.5)["status"] == "pass"


@pytest.mark.parametrize(
    ("flag", "timeout", "kill_at"),
    [
        (None, 300.0, 300.0),
        ("--reference-archive", 300.0, 300.0),
        ("--write-reference-archive", 300.0, 600.0),  # the light pass
        ("--write-reference-archive", 100.0, 200.0),
        ("--write-reference-archive", 600.0, 600.0),  # the giant pass: no later kill
        ("--write-reference-archive", 450.0, 600.0),
        ("--write-reference-archive", 1800.0, 1800.0),  # never before the cap
    ],
)
def test_a_recording_is_killed_at_twice_the_cap_up_to_a_ceiling(
    tmp_path, monkeypatch, flag, timeout, kill_at
):
    """Issue #1290: three jobs that sit at the 300 s cap timed out in the
    2026-10-24 recording, where RoadRunner compiled about 1.7x slower, and left no
    entry. The giants that time out at 600 s need 20 minutes and
    14 GB of RoadRunner alone, so their kill is not put off. The cap the worker
    judges by stays the cap."""
    seen = {}

    def schedule(specs, worker, *, timeout_of, **kw):
        seen.update({s["model_id"]: (s["cap"], timeout_of(s)) for s in specs})
        return [{**s, "status": "timeout", "cap": timeout_of(s)} for s in specs]

    monkeypatch.setattr(R.rc, "schedule", schedule)
    out = tmp_path / "report.json"
    argv = ["rr_run.py", "--jobs", str(SMOKE_JOBS), "--timeout", str(timeout)]
    argv += [flag, str(tmp_path / "arch")] if flag else []
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])
    R.main()
    assert seen and set(seen.values()) == {(timeout, kill_at)}
    comments = {r["comment"] for r in json.loads(out.read_text())["results"]}
    assert comments == {f"killed at {kill_at}s wall cap"}


@pytest.mark.parametrize(
    ("flag", "slow_kill_at"),
    [(None, 1200.0), ("--write-reference-archive", 1200.0)],
)
def test_a_model_roadrunner_is_slow_to_load_keeps_its_own_cap(
    tmp_path, monkeypatch, flag, slow_kill_at
):
    """libRoadRunner 2.11.0 takes about 550 s to load MODEL1006230049, 053 and 077
    on a CI runner, so the light pass's 300 s timed them out every live night
    (#1288). Their own cap wins over a shorter --timeout, live or recording; every
    other job keeps the --timeout's."""
    manifest = json.loads(SMOKE_JOBS.read_text())
    manifest["jobs"][0]["model_id"] = "MODEL1006230049"
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(manifest))
    seen = {}

    def schedule(specs, worker, *, timeout_of, **kw):
        seen.update({s["model_id"]: (s["cap"], timeout_of(s)) for s in specs})
        return [{**s, "status": "timeout", "cap": timeout_of(s)} for s in specs]

    monkeypatch.setattr(R.rc, "schedule", schedule)
    argv = ["rr_run.py", "--jobs", str(jobs), "--timeout", "300"]
    argv += [flag, str(tmp_path / "arch")] if flag else []
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(tmp_path / "report.json")])
    R.main()
    assert seen.pop("MODEL1006230049") == (1200.0, slow_kill_at)
    other_kill_at = 600.0 if flag else 300.0
    assert seen and set(seen.values()) == {(300.0, other_kill_at)}


def test_a_timeout_the_worker_judged_keeps_its_own_note(tmp_path, monkeypatch):
    """A recording's row past the cap was not killed there; the report says what
    happened rather than "killed at"."""
    note = "ran past the 300.0s wall cap (a recording runs to 600.0s to save its reference)"

    def schedule(specs, worker, **kw):
        keep = ("key", "model_id", "method", "metric", "tol")
        return [
            {**{k: s[k] for k in keep}, "status": "timeout", "cap": 300.0, "comment": note}
            for s in specs
        ]

    monkeypatch.setattr(R.rc, "schedule", schedule)
    out = tmp_path / "report.json"
    argv = ["rr_run.py", "--jobs", str(SMOKE_JOBS), "--timeout", "300"]
    argv += ["--write-reference-archive", str(tmp_path / "arch"), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    rows = json.loads(out.read_text())["results"]
    assert rows and {(r["outcome"], r["comment"]) for r in rows} == {("TIMEOUT", note)}


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
