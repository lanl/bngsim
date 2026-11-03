"""The SSA screen's RoadRunner reference archive (issue #805).

``ssa_screen.py --write-reference-archive`` records every RoadRunner result the
screen used for a model; ``--reference-archive`` replays them without
RoadRunner. An entry that no longer matches its model is stale, and a replay
that needs a result the recorded run never produced is unreplayable; neither
is compared. The end-to-end test records a few BioModels and replays them with
RoadRunner blocked from importing; it skips where the corpus is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import ssa_attribution as SA
import ssa_screen as SS

_RR = Path(SS.__file__).resolve().parent


def _spec(tmp_path: Path, **change) -> dict:
    xml = tmp_path / "m.xml"
    if not xml.exists():
        xml.write_text("<sbml>the model</sbml>")
    return {"name": "BIOMD1", "sbml_path": str(xml), "t_end": 100.0, "n_steps": 10, **change}


def _fake_rr_ssa(calls):
    def rr_ssa(xml, t_start, t_end, n_points, n, seed_base):
        calls.append(t_end)
        if t_end > 1000.0:
            raise ValueError("gillespie cannot do rate rules")
        t = np.linspace(t_start, t_end, n_points)
        return t, np.full((n, n_points, 2), t_end), ["A", "B"], {"ensemble_sec": 0.5}

    return rr_ssa


def test_every_result_the_live_run_used_comes_back(tmp_path, monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(SS.rc, "rr_ssa", _fake_rr_ssa(calls))
    monkeypatch.setattr(SS.versions, "roadrunner_version", lambda: "9.9.9-test")
    monkeypatch.setattr(SS, "_rr_ode", lambda xml, t_end, n_steps: (np.ones((11, 2)), ["A", "B"]))
    spec, arch = _spec(tmp_path), tmp_path / "arch"
    live = SS.Reference(spec, 30, 7, record_to=arch)
    for h in (100.0, 1000.0):  # the full horizon, then one extended-horizon rung
        live.ssa("m.xml", 0.0, h, 11, 30, 7)
    with pytest.raises(ValueError):
        live.ssa("m.xml", 0.0, 10000.0, 11, 30, 7)
    live.ode("m.xml", 100.0, 10)

    replay = SS.Reference(spec, 30, 7, replay=arch)
    assert replay.stale == ""
    t, arr, names, timing = replay.ssa("m.xml", 0.0, 1000.0, 11, 30, 7)
    np.testing.assert_array_equal(arr, np.full((30, 11, 2), 1000.0))
    assert names == ["A", "B"] and timing == {
        "ensemble_sec": 0.5,
        "replayed_from_archive": True,
        "reference_version": "9.9.9-test",
    }
    assert replay.elapsed_sec(100.0) == live.elapsed_sec(100.0)
    # A recorded failure is raised again under its class name, as the live one was.
    with pytest.raises(RuntimeError) as exc:
        replay.ssa("m.xml", 0.0, 10000.0, 11, 30, 7)
    assert type(exc.value).__name__ == "ValueError" and "rate rules" in str(exc.value)
    np.testing.assert_array_equal(replay.ode("m.xml", 100.0, 10)[0], np.ones((11, 2)))
    assert calls == [100.0, 1000.0, 10000.0]  # the replay itself ran nothing


def test_a_result_the_recorded_run_never_produced_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(SS.rc, "rr_ssa", _fake_rr_ssa([]))
    spec, arch = _spec(tmp_path), tmp_path / "arch"
    SS.Reference(spec, 30, 7, record_to=arch).ssa("m.xml", 0.0, 100.0, 11, 30, 7)
    replay = SS.Reference(spec, 30, 7, replay=arch)
    with pytest.raises(SS.ArchiveMiss, match="t=0→10,"):
        replay.ssa("m.xml", 0.0, 10.0, 11, 30, 7)  # a partial-horizon rung
    with pytest.raises(SS.ArchiveMiss, match="holds none for this model"):
        replay.ode("m.xml", 100.0, 10)
    entry = SS._not_from_archive({"max_mean_z": 0.0}, replay, SS.ArchiveMiss("x"))
    assert entry["status"] == "archive_unreplayable" and "max_mean_z" not in entry


def test_an_entry_that_no_longer_matches_its_model_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(SS.rc, "rr_ssa", _fake_rr_ssa([]))
    arch = tmp_path / "arch"
    SS.Reference(_spec(tmp_path), 30, 7, record_to=arch).ssa("m.xml", 0.0, 100.0, 11, 30, 7)
    for spec, n, seed, field in (
        (_spec(tmp_path, t_end=50.0), 30, 7, "t_end"),
        (_spec(tmp_path, n_steps=20), 30, 7, "n_steps"),
        (_spec(tmp_path), 100, 7, "n_rep"),
        (_spec(tmp_path), 30, 8, "seed_base"),
    ):
        assert SS.Reference(spec, n, seed, replay=arch).stale == f"the model's {field} changed"
    Path(_spec(tmp_path)["sbml_path"]).write_text("<sbml>another model</sbml>")
    ref = SS.Reference(_spec(tmp_path), 30, 7, replay=arch)
    assert ref.stale == "the model's model_sha256 changed"
    with pytest.raises(SS.ArchiveMiss):
        ref.ssa("m.xml", 0.0, 100.0, 11, 30, 7)
    assert SS.Reference(_spec(tmp_path, name="X"), 30, 7, replay=arch).stale == (
        "no entry for this model"
    )


def test_an_entry_that_cannot_be_read_is_stale(tmp_path):
    """A file in the archive that is not an entry is refused like a stale one.
    It used to raise in the worker, whose row then read as a bngsim crash."""
    spec, arch = _spec(tmp_path), tmp_path / "arch"
    arch.mkdir()
    SS._archive_file(arch, spec["name"]).write_bytes(b"not an npz")
    ref = SS.Reference(spec, 30, 7, replay=arch)
    assert ref.stale.startswith("the entry could not be read (")
    with pytest.raises(SS.ArchiveMiss):
        ref.ssa("m.xml", 0.0, 100.0, 11, 30, 7)


@pytest.mark.parametrize("status", ["archive_stale", "archive_unreplayable"])
def test_an_archive_refusal_is_an_uncompared_reference_failure(status):
    case = {
        "name": "BIOMD1",
        "status": status,
        "max_mean_z": 0.0,
        "error": f"roadrunner (archive): {status}: the model's rtol changed",
    }
    r = SA.case_to_job_result(case)
    assert (r.outcome, r.reference_refusal, r.value) == ("REFERENCE_FAILED", status, None)
    assert SA.is_expected(r.outcome, r.subclass)
    assert "the model's rtol changed" in r.comment


def test_two_names_that_sanitize_alike_get_their_own_files(tmp_path):
    """``a/b`` and ``a_b`` shared one entry file; the second model's recording
    overwrote the first's, which then replayed as stale for good."""
    assert SS._archive_file(tmp_path, "a/b") != SS._archive_file(tmp_path, "a_b")


# ── The worker itself, in-process, with bngsim's own SSA (other seeds) standing in
# for RoadRunner's gillespie: no RoadRunner and no corpus needed.


def _birth_death_sbml(path: Path) -> None:
    import libsbml

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId("bd")
    c = m.createCompartment()
    c.setId("C")
    c.setSize(1)
    c.setConstant(True)
    for sid, ic in (("A", 20.0), ("B", 0.0)):
        sp = m.createSpecies()
        sp.setId(sid)
        sp.setCompartment("C")
        sp.setInitialAmount(ic)
        sp.setHasOnlySubstanceUnits(True)
        sp.setBoundaryCondition(False)
        sp.setConstant(False)
    for pid, v in (("k1", 5.0), ("k2", 0.3)):
        par = m.createParameter()
        par.setId(pid)
        par.setValue(v)
        par.setConstant(True)
    for rid, reactant, product, law in (
        ("J0", None, "A", "k1"),
        ("J1", "A", "B", "k2*A"),
        ("J2", "B", None, "0.5*B"),
    ):
        r = m.createReaction()
        r.setId(rid)
        r.setReversible(False)
        for sid, make in ((reactant, r.createReactant), (product, r.createProduct)):
            if sid:
                ref = make()
                ref.setSpecies(sid)
                ref.setStoichiometry(1)
                ref.setConstant(True)
        r.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    path.write_text(libsbml.writeSBMLToString(doc))


class _Queue:
    def __init__(self):
        self.items: list = []

    def put(self, item):
        self.items.append(item)


def _run_worker(spec: dict, n: int = 30) -> dict:
    q = _Queue()
    SS._worker(
        spec,
        n,
        2000,
        SS.DEFAULT_MEAN_Z_TOL,
        5,
        SS.DEFAULT_MIN_MEAN_COUNT,
        SS.DEFAULT_SELF_Z_TOL,
        q,
    )
    return q.items[0]


@pytest.fixture
def recorded_model(tmp_path, monkeypatch):
    """A model recorded at the full horizon, then RoadRunner made to fail if called."""
    xml = tmp_path / "bd.xml"
    _birth_death_sbml(xml)
    spec = {
        "name": "bd",
        "source": "synthetic",
        "sbml_path": str(xml),
        "exists": True,
        "t_end": 20.0,
        "n_steps": 40,
        "effort": "low",
    }

    def stand_in(xml_, t_start, t_end, n_points, n, seed_base):
        times, arr, names, _diag, timing = SS.rc.bn_ssa_replicates(
            xml_, t_start, t_end, n_points, n, seed_base + 777, rep_timeout=None
        )
        return times, arr, names, timing

    monkeypatch.setattr(SS.rc, "rr_ssa", stand_in)
    monkeypatch.setattr(SS.versions, "roadrunner_version", lambda: "9.9.9-test")
    arch = tmp_path / "arch"
    SS.Reference(spec, 30, 2000, record_to=arch).ssa(str(xml), 0.0, 20.0, 41, 30, 2000)

    def live(*args, **kwargs):
        raise AssertionError("RoadRunner was run during a replay")

    monkeypatch.setattr(SS.rc, "rr_ssa", live)
    return {**spec, "reference_archive": str(arch)}, arch


def test_a_replay_compares_bngsim_and_names_the_recorded_roadrunner(recorded_model):
    spec, _arch = recorded_model
    row = _run_worker(spec)
    assert row["status"] == "pass"
    rr = row["timing"]["roadrunner"]
    assert rr["replayed_from_archive"] is True and rr["reference_version"] == "9.9.9-test"
    result = SA.case_to_job_result(row, versions={"roadrunner": "2.11.0"})
    assert result.outcome == "PASS" and result.versions["roadrunner"] == "9.9.9-test"


def test_a_replay_against_a_wrong_reference_still_fails(recorded_model):
    """RoadRunner's ensembles scaled by 1.5 in the record: the replay is a red
    fail, left unattributed because the record has no ODE."""
    spec, arch = recorded_model
    path = SS._archive_file(arch, "bd")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] * 1.5 if k.startswith("arr_") else z[k] for k in z.files}
    np.savez_compressed(path, **arrays)
    row = _run_worker(spec)
    assert row["status"] == "fail"
    assert row["diff_attribution"]["reason"].startswith("unattributed:")
    assert SA.case_to_job_result(row).outcome == "DIFF"


def test_a_stale_or_unreadable_entry_runs_no_bngsim_ensemble(recorded_model, monkeypatch):
    spec, arch = recorded_model
    calls = []
    monkeypatch.setattr(SS.rc, "bn_ssa_replicates", lambda *a, **k: calls.append(a))
    row = _run_worker(spec, n=100)  # recorded at n = 30
    assert row["status"] == "archive_stale" and "the model's n_rep changed" in row["error"]
    SS._archive_file(arch, "bd").write_bytes(b"not an npz")
    row = _run_worker(spec)
    assert row["status"] == "archive_stale" and "the entry could not be read" in row["error"]
    assert SA.case_to_job_result(row).outcome == "REFERENCE_FAILED"
    assert calls == []


def _screen(tmp_path, *args, env=None):
    out = tmp_path / f"core{len(list(tmp_path.glob('core*')))}.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "ssa_screen.py"),
            "--corpus",
            "biomodels",
            "--jobs",
            "4",
            "--out",
            str(tmp_path / "native.json"),
            "--core-out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_RR,
        timeout=1200,
        env=env,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc, out


# A pass, a vacuous pass that walks the extended ladder, an extended-horizon
# verdict, a partial-horizon verdict, and BIOMD0000001026: an attributed fail
# (rr_known, sys-bio/roadrunner#1318) on libRoadRunner 2.9.2, a pass on 2.11.0
# (GH #1288).
_E2E = "BIOMD0000000417,BIOMD0000000414,BIOMD0000000573,BIOMD0000000760,BIOMD0000001026"


def test_a_replay_without_roadrunner_reproduces_the_live_verdicts(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "ssa_screen.py"),
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
    pytest.importorskip("roadrunner")
    pytest.importorskip("bngsim")
    sbml = Path(os.environ.get("BIOMODELS_SBML_DIR", "/nonexistent"))
    if not all((sbml / f"{m}.xml").is_file() for m in _E2E.split(",")):
        pytest.skip("BioModels SSA corpus not present (set BIOMODELS_SBML_DIR)")
    arch = tmp_path / "arch"
    _p, live = _screen(tmp_path, "--models", _E2E, "--write-reference-archive", str(arch))
    blocked = tmp_path / "no_roadrunner" / "roadrunner"
    blocked.mkdir(parents=True)
    (blocked / "__init__.py").write_text("raise ImportError('blocked for this test')\n")
    env = {
        **os.environ,
        "PYTHONPATH": f"{blocked.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    _p, replay = _screen(tmp_path, "--models", _E2E, "--reference-archive", str(arch), env=env)

    def rows(path):
        return {r["model_id"]: r for r in json.loads(path.read_text())["results"]}

    live_rows, replay_rows = rows(live), rows(replay)
    assert set(live_rows) == set(replay_rows) == set(_E2E.split(","))
    for model_id, row in live_rows.items():
        got = replay_rows[model_id]
        assert (got["outcome"], got["subclass"], got["value"], got["comment"]) == (
            row["outcome"],
            row["subclass"],
            row["value"],
            row["comment"],
        ), model_id
        if (row.get("timing") or {}).get("roadrunner"):
            assert got["timing"]["roadrunner"]["replayed_from_archive"] is True
