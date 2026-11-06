"""How bng_stoch_run calls the legacy NFsim (issue #1549).

bngsim runs network-free with same-complex binding blocked, as BNG2.pl's network
does, and reads a fractional seed count rounded half-up (GH #51). Legacy NFsim
v1.14.3 blocks only under ``-cb`` (its ``-cb`` and ``-bscb`` set one flag), which
BNG2.pl's ``simulate_nf`` passes by default, and truncates a fractional count. The
harness gave it ``-cb`` only for a Species observable and the raw XML, so 14 of the
15 unexplained network-free DIFFs in 0.17.0's sweep were the harness's, not
bngsim's. These tests pin the legacy call: ``-cb`` whenever bngsim blocks, and the
XML with bngsim's seed counts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_BNG = HERE.parent / "bng_parity"
sys.path.insert(0, str(_BNG))

import _bng_common as bc  # noqa: E402
import bng_stoch_run as bsr  # noqa: E402

_FIXTURE = HERE.parent.parent / "tests" / "data" / "nfsim" / "fractional_init.xml"


def _xml(tmp_path: Path, value: float, conc: str = "n_init") -> Path:
    """The one-species fixture with its seed parameter set to ``value`` and its
    concentration attribute set to ``conc``."""
    text = _FIXTURE.read_text()
    text = text.replace('value="1.9"', f'value="{value!r}"')
    text = text.replace('concentration="n_init"', f'concentration="{conc}"')
    path = tmp_path / f"seed_{value!r}_{abs(hash(conc))}.xml"
    path.write_text(text)
    return path


def _bngsim_seed_count(xml: Path) -> float:
    """X_total at t = 0 as bngsim's own NFsim reads ``xml``."""
    bngsim = pytest.importorskip("bngsim")
    if not getattr(bngsim, "HAS_NFSIM", False):
        pytest.skip("bngsim built without NFsim")
    sess = bngsim.NfsimSession(str(xml))
    try:
        sess.initialize(seed=1)
        r = sess.simulate(0.0, 1.0, 2)
        return float(np.asarray(r.observables)[0, list(r.observable_names).index("X_total")])
    finally:
        sess.destroy()


@pytest.mark.parametrize(
    "value,expected",
    [(1.9, 2), (0.5, 1), (2.5, 3), (421.5498, 422), (0.4999, 0), (7.0, 7)],
)
def test_the_legacy_xml_holds_the_seed_count_bngsim_reads(tmp_path, value, expected):
    """A parameter-valued seed is rounded half-up, to the count bngsim's engine
    seeds; an integer one is left alone and no copy is written."""
    xml = _xml(tmp_path, value)
    out = tmp_path / "rounded.xml"
    ids = bc.seed_rounded_xml(xml, out)
    if value != int(value):
        assert ids == ["S1"]
        assert f'concentration="{expected}"' in out.read_text()
        assert bc.seed_rounded_xml(out, tmp_path / "again.xml") == []  # idempotent
    else:
        assert ids == [] and not out.exists()
    assert _bngsim_seed_count(xml) == expected


def test_a_literal_seed_is_rounded_and_an_expression_left_as_bngsim_leaves_it(tmp_path):
    out = tmp_path / "rounded.xml"
    assert bc.seed_rounded_xml(_xml(tmp_path, 1.9, conc="5.7"), out) == ["S1"]
    assert 'concentration="6"' in out.read_text()
    assert _bngsim_seed_count(_xml(tmp_path, 1.9, conc="5.7")) == 6
    # A compound expression is not resolved, here or in bngsim's C++ rewrite.
    assert bc.seed_rounded_xml(_xml(tmp_path, 1.9, conc="2*n_init"), out.with_name("x")) == []


_SPECIES_OBS = """begin model
begin observables
  Species  Free  R(l,l)
end observables
end model
"""
_MOLECULES_OBS = _SPECIES_OBS.replace("Species  Free", "Molecules  Free")


def test_legacy_gets_cb_whenever_bngsim_blocks_same_complex_binding():
    assert bsr.legacy_cb(_MOLECULES_OBS, True) is True
    assert bsr.legacy_cb(_SPECIES_OBS, True) is True
    # Blocking off (a timing comparison): -cb only where a Species observable needs it.
    assert bsr.legacy_cb(_MOLECULES_OBS, False) is False
    assert bsr.legacy_cb(_SPECIES_OBS, False) is True


def _fake_nfsim(calls):
    """A stand-in for subprocess.run that records the NFsim command and writes a
    .gdat holding the seed concentration of the XML it was given."""

    def run(cmd, **_kw):
        calls.append(cmd)
        xml = Path(cmd[cmd.index("-xml") + 1]).read_text()
        conc = xml.split('concentration="', 1)[1].split('"', 1)[0]
        if conc == "n_init":  # the parameter's value, as NFsim reads it before truncating
            conc = xml.split('value="', 1)[1].split('"', 1)[0]
        out = cmd[cmd.index("-o") + 1]
        Path(out).write_text(f"# time X_total\n0 {conc}\n1 {conc}\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return run


def test_nfsim_run_gives_the_rounded_xml_and_cb(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(bc.subprocess, "run", _fake_nfsim(calls))
    xml = _xml(tmp_path, 421.5498)
    kw = dict(t_end=1.0, n_steps=1, n_rep=2, seed_base=1, timeout=10.0)
    _t, obs, names, _ = bc.nfsim_run(
        xml, "NFsim", out_prefix=str(tmp_path / "leg"), complex_bookkeeping=True, **kw
    )
    assert names == ["X_total"] and obs[:, 0, 0].tolist() == [422.0, 422.0]
    assert all("-cb" in c and c[c.index("-xml") + 1] != str(xml) for c in calls)
    # round_seeds=False hands NFsim the XML as BNG2.pl wrote it.
    calls.clear()
    _t, obs, _n, _ = bc.nfsim_run(
        xml, "NFsim", out_prefix=str(tmp_path / "raw"), round_seeds=False, **kw
    )
    assert obs[:, 0, 0].tolist() == [421.5498, 421.5498]
    assert all("-cb" not in c and c[c.index("-xml") + 1] == str(xml) for c in calls)


def test_an_nf_entry_recorded_with_the_old_legacy_call_is_stale(tmp_path, monkeypatch):
    """An archived legacy NFsim run made before #1549 (no -cb, raw XML) is not a
    reference for the call the harness now makes: a replay refuses it."""
    spec = {
        "key": "m.bngl:nf",
        "model_id": "m.bngl",
        "track": "nf",
        "n_rep": 10,
        "seed_base": 1,
        "nf_bscb": True,
        "inject": None,
    }
    arch = tmp_path / "arch"
    monkeypatch.setattr(bsr, "NF_REFERENCE_CALL", None)  # what an older entry holds
    bsr.Legacy(spec, "the model", record_to=arch).check_run(
        {"t_start": 0.0, "t_end": 1.0, "n_steps": 4}, False
    )
    monkeypatch.undo()
    assert bsr.Legacy(spec, "the model", replay=arch).stale == (
        "the job's nf_reference_call changed"
    )
    # An SSA entry's identity does not carry it, so SSA archives stay valid.
    ssa = {**spec, "key": "m.bngl:ssa", "track": "ssa"}
    assert bsr.archive_identity(ssa, "the model")["nf_reference_call"] is None


# --------------------------------------------------------------------------- #
# End to end, against BioNetGen 2.9.3's NFsim
# --------------------------------------------------------------------------- #
# A ring-former: A(r) + A(l) closes chains into rings unless same-complex binding
# is blocked, and legacy NFsim without -cb then breaks ring bonds through the "+"
# product, so its conserved A_total drifts above 30. Measured before #1549: 21% of
# points agreed, after: 100%.
_RING = """begin model
begin parameters
  kf  0.5
  kr  0.05
end parameters
begin molecule types
  A(l,r)
end molecule types
begin seed species
  A(l,r)  30
end seed species
begin observables
  Molecules  A_total  A()
  Molecules  A_bound  A(r!+)
end observables
begin reaction rules
  A(r) + A(l) <-> A(r!1).A(l!1)  kf, kr
end reaction rules
end model
begin actions
simulate({method=>"nf",t_end=>40,n_steps=>40})
end actions
"""
# A fractional seed: bngsim seeds 31 A, legacy NFsim on the raw XML 30, so A_total
# differs at every point. Measured before #1549: 50% of points agreed, after: 100%.
_FRACTIONAL = """begin model
begin parameters
  A0  30.6
  kf  0.01
  kr  0.1
end parameters
begin molecule types
  A(b)
  B(a)
end molecule types
begin seed species
  A(b)  A0
  B(a)  20
end seed species
begin observables
  Molecules  A_total  A()
  Molecules  AB       A(b!1).B(a!1)
end observables
begin reaction rules
  A(b) + B(a) <-> A(b!1).B(a!1)  kf, kr
end reaction rules
end model
begin actions
simulate({method=>"nf",t_end=>20,n_steps=>20})
end actions
"""


def _nfsim_available() -> bool:
    root = Path(os.environ.get("BNGPATH", "/nonexistent"))
    return (root / "BNG2.pl").is_file() and (root / "bin" / "NFsim").is_file()


def test_ring_former_and_fractional_seed_pass_against_legacy_nfsim(tmp_path):
    pytest.importorskip("bngsim")
    if not _nfsim_available() or shutil.which("perl") is None:
        pytest.skip("BioNetGen not available (set BNGPATH)")
    models = tmp_path / "models" / "t"
    models.mkdir(parents=True)
    jobs = []
    for name, text in (("ring_chain", _RING), ("fractional_seed", _FRACTIONAL)):
        (models / f"{name}.bngl").write_text(text)
        mid = f"t/{name}.bngl"
        jobs.append(
            {
                "model_id": mid,
                "input_format": "bngl",
                "method": "stochastic",
                "reference_engine": "bng",
                "model": mid,
                "oracle": {"metric": "mean_zscore", "tol": 5.0},
                "params": {"tier": "t", "source": "t", "methods": ["nf"], "seed": 1, "n_rep": 10},
                "overrides": [],
                "notes": "",
            }
        )
    (tmp_path / "jobs.json").write_text(json.dumps({"_meta": {}, "jobs": jobs}))
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_BNG / "bng_stoch_run.py"),
            "--track",
            "nf",
            "--jobs",
            str(tmp_path / "jobs.json"),
            "--workers",
            "2",
            "--timeout",
            "300",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=_BNG,
        timeout=900,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    rows = {r["model_id"]: r for r in json.loads(out.read_text())["results"]}
    for mid in ("t/ring_chain.bngl", "t/fractional_seed.bngl"):
        assert rows[mid]["outcome"] == "PASS", (mid, rows[mid].get("comment"))


# --------------------------------------------------------------------------- #
# mlnr.bngl's cap
# --------------------------------------------------------------------------- #
def _job(*overrides):
    return SimpleNamespace(
        overrides=[SimpleNamespace(field=f, value=v, reason=f"{v} s") for f, v in overrides]
    )


def test_a_timeout_override_extends_the_default_cap_and_never_cuts_it():
    """mlnr's 300 s override, written when the default was 180 s, cut its cap from
    the 1800 s default and timed out a job whose ten replicates take about 1,170 s."""
    assert bsr._job_timeout_override(_job(("timeout", 300))) is None
    assert bsr._job_timeout_override(_job(("timeout", bsr.DEFAULT_STOCH_TIMEOUT))) is None
    assert bsr._job_timeout_override(_job(("timeout", 3600))) == {
        "timeout": 3600.0,
        "reason": "3600 s",
    }
    assert bsr._job_timeout_override(_job(("tol", 1e-3))) is None


def test_no_committed_job_carries_a_timeout_below_its_runners_default():
    import bng_ode_run as bor
    from _core import read_manifest

    floor = {"stochastic": bsr.DEFAULT_STOCH_TIMEOUT, "ode": bor.DEFAULT_ODE_TIMEOUT}
    _meta, jobs = read_manifest(_BNG / "jobs.json")
    low = {
        j.model_id: o.value
        for j in jobs
        for o in j.overrides
        if o.field == "timeout" and float(o.value) <= floor[j.method]
    }
    assert low == {}


def test_the_ode_runner_also_only_extends_its_default_cap():
    import bng_ode_run as bor

    assert bor._job_timeout_override(_job(("timeout", 200))) is None
    assert bor._job_timeout_override(_job(("timeout", 600))) == {
        "timeout": 600.0,
        "reason": "600 s",
    }


# --------------------------------------------------------------------------- #
# The worker's wiring, without BNG2.pl
# --------------------------------------------------------------------------- #
def test_an_nf_run_spec_gives_legacy_cb_whenever_bngsim_blocks():
    sspec = bsr.nf_run_settings({"t_start": 5.0}, True, _MOLECULES_OBS)
    assert sspec == {"t_start": 0.0, "nf_bscb": True, "nf_complex_bookkeeping": True}
    off = bsr.nf_run_settings({"t_start": 0.0}, False, _MOLECULES_OBS)
    assert (off["nf_bscb"], off["nf_complex_bookkeeping"]) == (False, False)


def test_the_legacy_ensemble_passes_cb_and_rounded_seeds_to_nfsim(monkeypatch):
    seen = {}

    def fake(xml, nfsim_bin, **kw):
        seen.update(kw)
        return np.zeros(2), np.zeros((1, 2, 1)), ["X"], {}

    monkeypatch.setattr(bsr.bc, "nfsim_run", fake)
    sspec = bsr.nf_run_settings({"t_end": 1.0, "n_steps": 1}, True, _MOLECULES_OBS)
    bsr._legacy_stoch_ensemble("nf", "m.xml", sspec, "NFsim", 10, 1, "leg", 60.0)
    assert seen["complex_bookkeeping"] is True and seen["round_seeds"] is True


def test_the_worker_gives_the_legacy_run_cb_and_rounded_seeds(tmp_path, monkeypatch):
    """The worker itself, in process: BNG2.pl's XML build and the legacy NFsim run
    are stubbed, bngsim runs for real. A model with no Species observable, and a
    fractional seed, gets -cb and rounded seeds on the legacy side."""
    _bngsim_seed_count(_FIXTURE)  # skips without bngsim's NFsim
    bngl = tmp_path / "m.bngl"
    bngl.write_text(
        _MOLECULES_OBS.replace("Free  R(l,l)", "X_total  X()")
        + 'begin actions\nsimulate({method=>"nf",t_end=>1,n_steps=>2})\nend actions\n'
    )

    def fake_xml(text, bng2_pl, workdir, **_kw):
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(_FIXTURE, workdir / "model.xml")
        return workdir / "model.xml", 0.0, ""

    seen = {}

    def fake_nfsim(xml, nfsim_bin, *, n_rep, n_steps, **kw):
        seen.update(kw)
        t = np.linspace(0.0, 1.0, n_steps + 1)
        return t, np.full((n_rep, n_steps + 1, 1), 2.0), ["X_total"], {}

    monkeypatch.setattr(bsr.bc, "measure_stoch_warmup", lambda *a, **k: {})
    monkeypatch.setattr(bsr.bc, "generate_xml", fake_xml)
    monkeypatch.setattr(bsr.bc, "nfsim_run", fake_nfsim)
    spec = {
        "key": "m.bngl:nf",
        "model_id": "m.bngl",
        "method": "stochastic",
        "metric": "mean_zscore",
        "tol": 5.0,
        "bngl": str(bngl),
        "track": "nf",
        "n_rep": 3,
        "seed_base": 1,
        "rep_timeout": 60.0,
        "nf_bscb": True,
        "cap": 300.0,
        "sub_timeout": 270.0,
        "settings": {},
        "bng2_pl": "BNG2.pl",
        "legacy_bin": "NFsim",
        "legacy_label": "NFsim",
        "inject": None,
        "reference_archive": None,
        "write_reference_archive": None,
    }
    rows = []
    bsr._worker(spec, SimpleNamespace(put=rows.append))
    assert seen["complex_bookkeeping"] is True and seen["round_seeds"] is True
    assert rows[0]["status"] == "pass", rows[0]
