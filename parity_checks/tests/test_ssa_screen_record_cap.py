"""A recording of the SSA screen keeps what RoadRunner computes past the cap (#1290).

``ssa_screen.py`` runs RoadRunner after bngsim's ensemble, and ``Reference``
saves the model's entry after each RoadRunner call. A worker killed at the
per-model cap before its first RoadRunner call returned left no entry, so every
replay refused the model as ``archive_stale``, an alert on the night after a live
night that passed it. MODEL1102210001 sits at the 120 s cap (RoadRunner about
86 s, bngsim about 20 s): it passed in reference-data-v1's recording and timed out
in v2's, which has no entry for it.

A recording now takes each row at the cap exactly as a live run does, and kills
the worker only at ``rc.recording_kill_after(cap)``. These tests drive the real
spawn scheduler with stand-in workers (``smoke_workers.screen_worker``).
"""

from __future__ import annotations

import argparse
import time

import _rr_common as rc
import pytest
import smoke_workers as sw
import ssa_screen as SS

NAMES = ("fast", "slow_rr", "prelim", "late", "hang")


def _run(tmp_path, monkeypatch, *, record: bool):
    monkeypatch.setattr(SS, "PROBE_DIR", tmp_path / "probe")
    monkeypatch.setattr(SS, "_print_progress", lambda *a: None)
    arch, rr_done = tmp_path / "arch", tmp_path / "rr_done"
    arch.mkdir(parents=True, exist_ok=True)
    rr_done.mkdir()
    args = argparse.Namespace(
        jobs=len(NAMES),
        timeout=2.0,
        n=4,
        seed_base=1,
        mean_z_tol=9.0,
        min_mean_count=2.0,
        self_z_tol=4.5,
        write_reference_archive=str(arch) if record else "",
    )
    specs = [
        {
            "name": name,
            "t_end": 1.0,
            "sbml_path": "unused",
            "write_reference_archive": args.write_reference_archive or None,
            "rr_done": str(rr_done),
        }
        for name in NAMES
    ]
    t0 = time.monotonic()
    rows = SS._schedule(specs, args, worker=sw.screen_worker)
    return {r["name"]: r for r in rows}, time.monotonic() - t0, arch, rr_done


def test_a_recording_lets_roadrunner_finish_past_the_cap(tmp_path, monkeypatch):
    rows, took, arch, _rr_done = _run(tmp_path, monkeypatch, record=True)
    # RoadRunner finished at 2.5 s, past the 2 s cap: its result was saved...
    assert (arch / "slow_rr.saved").is_file()
    # ...and the row is still the one the cap gives, not the pass that followed.
    assert rows["slow_rr"]["status"] == "too_slow"
    assert rows["slow_rr"]["error"] == "exceeded 2.0s wall cap"
    # A worker that never finishes is killed at twice the cap, and a row it puts
    # past the cap is not the row.
    assert rows["hang"]["status"] == "too_slow"
    assert rows["late"]["status"] == "too_slow"
    assert 4.0 <= took < 5.0
    # The row names no path of the run's own.
    assert not {"sbml_path", "reference_archive", "write_reference_archive"} & set(rows["hang"])


def test_a_recording_gives_every_row_a_live_run_gives(tmp_path, monkeypatch):
    """The record night's verdicts match a live night's: the extra time only
    serves the archive."""
    live, live_took, _arch, live_rr_done = _run(tmp_path / "live", monkeypatch, record=False)
    recorded, _took, _arch, rec_rr_done = _run(tmp_path / "rec", monkeypatch, record=True)
    for name in NAMES:
        assert (recorded[name]["status"], recorded[name].get("error")) == (
            live[name]["status"],
            live[name].get("error"),
        ), name
    assert live["fast"]["status"] == "pass"
    assert (
        live["prelim"]
        == recorded["prelim"]
        == {
            "name": "prelim",
            "status": "error",
            "error": "the red row first",
        }
    )
    assert live["late"]["status"] == "too_slow"
    # A live run kills at the cap, before RoadRunner's result; a recording does not.
    assert not (live_rr_done / "slow_rr").exists()
    assert (rec_rr_done / "slow_rr").exists()
    assert 2.0 <= live_took < 3.0


@pytest.mark.parametrize(
    ("cap", "kill_at"),
    [(120.0, 240.0), (300.0, 600.0), (450.0, 600.0), (600.0, 600.0), (1800.0, 1800.0)],
)
def test_the_later_kill_is_twice_the_cap_up_to_a_ceiling(cap, kill_at):
    assert rc.recording_kill_after(cap) == kill_at
