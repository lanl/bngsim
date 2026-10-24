"""The BNG stochastic reference archive (issue #805).

``bng_stoch_run.py --write-reference-archive`` records each job's legacy side:
the artifact BNG2.pl built and every reference ensemble the job ran, keyed by
what it was asked for. ``--reference-archive`` replays them with neither
BNG2.pl nor the legacy binary. An entry that no longer matches its job is
stale; a replay that needs an ensemble the recorded run never ran is
unreplayable; neither is compared. The end-to-end test records a few SSA jobs
(one per legacy call kind) and replays them with BNG unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bng_stoch_run as bsr  # noqa: E402

_BNG = HERE.parent / "bng_parity"


def _spec(tmp_path, **change):
    return {
        "key": "m.bngl:ssa",
        "model_id": "m.bngl",
        "track": "ssa",
        "n_rep": 10,
        "seed_base": 1,
        "nf_bscb": True,
        "inject": None,
        **change,
    }


def _ens(n_rep, value):
    t = np.linspace(0.0, 1.0, 5)
    return t, np.full((n_rep, 5, 2), float(value)), ["A", "B"]


def _record(tmp_path, spec, bngl="the model"):
    arch, work = tmp_path / "arch", tmp_path / "work"
    work.mkdir(exist_ok=True)
    live = bsr.Legacy(spec, bngl, record_to=arch)

    def build():
        (work / "built.net").write_text("begin parameters\nend parameters\n")
        return work / "built.net", 1.5, ""

    live.build(build, work, ".net")
    live.check_run({"t_start": 0.0, "t_end": 1.0, "n_steps": 4}, False)
    live.call("legacy:10:1", lambda: (*_ens(10, 3.0), {"ensemble_sec": 0.2}))
    live.call("roadrunner:5:10:1", lambda: None)
    with pytest.raises(OSError):
        live.call("legacy:20:11", lambda: (_ for _ in ()).throw(OSError("edgepop")))
    return arch


def test_everything_the_live_run_recorded_comes_back(tmp_path):
    spec = _spec(tmp_path)
    arch = _record(tmp_path, spec)
    replay = bsr.Legacy(spec, "the model", replay=arch)
    assert replay.stale == ""
    out = tmp_path / "replay"
    out.mkdir()
    artifact, sec, err = replay.build(lambda: pytest.fail("built again"), out, ".net")
    assert artifact.read_text().startswith("begin parameters") and (sec, err) == (1.5, "")
    replay.check_run({"t_start": 0.0, "t_end": 1.0, "n_steps": 4}, False)
    t, v, n, timing = replay.call("legacy:10:1", pytest.fail)
    np.testing.assert_array_equal(v, _ens(10, 3.0)[1])
    assert n == ["A", "B"] and timing == {"ensemble_sec": 0.2, "replayed_from_archive": True}
    assert replay.call("roadrunner:5:10:1", pytest.fail) is None
    # A recorded failure is raised again under its class name, as the live one was.
    with pytest.raises(RuntimeError) as exc:
        replay.call("legacy:20:11", pytest.fail)
    assert type(exc.value).__name__ == "OSError" and str(exc.value) == "edgepop"


def test_an_ensemble_the_recorded_run_never_ran_is_unreplayable(tmp_path):
    spec = _spec(tmp_path)
    replay = bsr.Legacy(spec, "the model", replay=_record(tmp_path, spec))
    with pytest.raises(bsr.ArchiveMiss, match="legacy:70:31") as miss:
        replay.call("legacy:70:31", pytest.fail)  # an escalation rung it never reached
    res = bsr._not_from_archive(
        {"value": 0.5, "escalation": []}, replay, miss.value, "run_network"
    )
    assert (res["status"], res["reference_refusal"], res["value"]) == (
        "reference_failed",
        "archive_unreplayable",
        None,
    )
    assert "escalation" not in res
    assert res["exception"].startswith("run_network (archive): archive_unreplayable: ")
    # ...and an archive miss is never swallowed as an escalation that "aborted".
    with pytest.raises(bsr.ArchiveMiss):
        bsr._run_escalation(
            _ens(10, 1.0),
            _ens(10, 2.0),
            "diff",
            0.5,
            "",
            "ensemble_frac_pass",
            0.99,
            n_rep_base=10,
            seed_base=1,
            compare=lambda b, leg: ("diff", 0.5, "", "ensemble_frac_pass", 0.99),
            escalate=lambda d, s, r: replay.call("legacy:70:31", pytest.fail),
        )


def test_an_entry_that_no_longer_matches_its_job_is_stale(tmp_path):
    arch = _record(tmp_path, _spec(tmp_path))
    for spec, bngl, field in (
        (_spec(tmp_path), "another model", "bngl_sha256"),
        (_spec(tmp_path, inject="simulate({method=>'ssa'})"), "the model", "inject_sha256"),
        (_spec(tmp_path, n_rep=30), "the model", "n_rep"),
        (_spec(tmp_path, seed_base=2), "the model", "seed_base"),
    ):
        assert bsr.Legacy(spec, bngl, replay=arch).stale == f"the job's {field} changed"
    assert bsr.Legacy(_spec(tmp_path, key="x:ssa"), "the model", replay=arch).stale == (
        "no entry for this job"
    )
    replay = bsr.Legacy(_spec(tmp_path), "the model", replay=arch)
    with pytest.raises(bsr.ArchiveMiss, match="the run's t_end, mode changed"):
        replay.check_run({"t_start": 0.0, "t_end": 2.0, "n_steps": 4}, True)
    assert replay.stale  # so the row is archive_stale, not unreplayable


def test_the_two_flags_are_exclusive():
    proc = subprocess.run(
        [
            sys.executable,
            str(_BNG / "bng_stoch_run.py"),
            "--track",
            "ssa",
            "--reference-archive",
            "a",
            "--write-reference-archive",
            "b",
        ],
        capture_output=True,
        text=True,
        cwd=_BNG,
        timeout=120,
    )
    assert proc.returncode != 0 and "--write-reference-archive records one" in proc.stderr


# A single-segment pass, a carry-over (native protocol) pass, a run_network
# failure scored against the fallback oracles, and an escalated borderline.
_MY = "original/bngl_models/my_models"
_E2E = ",".join(
    f"{_MY}/{m}.bngl"
    for m in ("ssa/PushPull", "ode/gardner2000_toggle_swtich", "ode/v18", "ssa/ExampleModel5_v2")
)


def _bng_tools() -> bool:
    root = Path(os.environ.get("BNGPATH", "/nonexistent"))
    return (root / "BNG2.pl").is_file() and (root / "bin" / "run_network").is_file()


def _run(tmp_path, *args, env=None):
    out = tmp_path / f"report{len(list(tmp_path.glob('report*')))}.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_BNG / "bng_stoch_run.py"),
            "--track",
            "ssa",
            "--models",
            _E2E,
            "--workers",
            "4",
            "--timeout",
            "300",
            "--out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_BNG,
        timeout=1800,
        env=env,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return {r["model_id"]: r for r in json.loads(out.read_text())["results"]}


def test_a_replay_without_bng_reproduces_the_live_verdicts(tmp_path):
    pytest.importorskip("bngsim")
    if not _bng_tools() or shutil.which("perl") is None:
        pytest.skip("BioNetGen not available (set BNGPATH)")
    if not all((_BNG / "models" / m).is_file() for m in _E2E.split(",")):
        pytest.skip("BNG corpus not in this checkout")
    arch = tmp_path / "arch"
    live = _run(tmp_path, "--write-reference-archive", str(arch))
    assert len(list(arch.glob("*.npz"))) == len(live) == 4
    # Neither BNG2.pl nor run_network can be found by the replay or its workers.
    replay = _run(
        tmp_path, "--reference-archive", str(arch), env={**os.environ, "BNGPATH": "/nonexistent"}
    )
    assert set(replay) == set(live)
    for model_id, row in live.items():
        got = replay[model_id]
        assert (got["outcome"], got["value"], got["comment"], got["exception"]) == (
            row["outcome"],
            row["value"],
            row["comment"],
            row["exception"],
        ), model_id
        if (row.get("timing") or {}).get("legacy"):  # absent where run_network failed
            assert got["timing"]["legacy"]["replayed_from_archive"] is True
