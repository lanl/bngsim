"""Guard the UPSTREAM_FIXES re-source mechanism (vendor_corpus.UPSTREAM_FIXES / resolve()).

A ``bngl_models`` model whose model-side fix landed in wshlavacek/BNGL-Models after the
``bngl_models`` pin is re-sourced, at its own path, from the dedicated ``bngl_fixes`` pin,
so the ~430 other models at the ``bngl_models`` pin do not move with it (the CURATED_SIX
precedent; issue #543 for SIR_v4). This locks that contract so a re-pin or re-vendor
cannot silently drop a fix:

  * every UPSTREAM_FIXES key is a live manifest membership entry;
  * resolve() sends each to the ``bngl_fixes`` pin, at the path the model has upstream;
  * the committed manifest and jobs.json record that pin and the fixed bytes;
  * the vendored file is the upstream file, byte for byte — nothing is repaired at import.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BNG = HERE.parent / "bng_parity"


def _load(unique_name: str, path: Path):
    # A unique module name, for the reason test_curated_resource.py gives.
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


vc = _load("bng_parity_vendor_corpus_upstream_fixes", BNG / "vendor_corpus.py")

_MANIFEST = json.loads((BNG / "manifest.json").read_text())["records"]
_BY_KEY = {(m["source"], m["relpath"]): m for m in _MANIFEST}
_JOBS = {j["model_id"]: j for j in json.loads((BNG / "jobs.json").read_text())["jobs"]}
_FIXES_PIN = vc.PINS["bngl_fixes"]


def test_fix_keys_are_live():
    stale = [k for k in vc.UPSTREAM_FIXES if k not in _BY_KEY]
    assert not stale, f"UPSTREAM_FIXES has stale keys (no matching manifest model): {stale}"


def test_resolve_sends_each_fix_to_the_fixes_pin_at_its_own_path():
    for source, relpath in vc.UPSTREAM_FIXES:
        repo_key, path = vc.resolve(source, relpath)
        assert repo_key == "bngl_fixes", f"{relpath} did not redirect to the fixes pin"
        _plain_key, prefix = vc.SOURCE_TO_REPO[source]
        assert path == f"{prefix}{relpath}"
    assert _FIXES_PIN.slug == vc.PINS["bngl_models"].slug == "wshlavacek/BNGL-Models"
    assert _FIXES_PIN.sha != vc.PINS["bngl_models"].sha


def test_manifest_and_jobs_record_the_fix():
    for (source, relpath), spec in vc.UPSTREAM_FIXES.items():
        entry = _BY_KEY[(source, relpath)]
        origin = entry["origin"]
        assert origin["repo"] == _FIXES_PIN.slug
        assert origin["commit"] == _FIXES_PIN.sha  # pinned, never @main
        assert origin["path"] == vc.resolve(source, relpath)[1]
        assert entry["upstream_fix"]["issue"] == spec["issue"]
        assert entry["patched"] is False and entry["curated"] is False
        job = _JOBS[entry["id"]]
        assert job["params"]["sha256"] == entry["sha256"], "jobs.json was not rebuilt"


def test_vendored_bytes_are_the_pinned_upstream_bytes():
    for key in vc.UPSTREAM_FIXES:
        entry = _BY_KEY[key]
        on_disk = hashlib.sha256((BNG / "models" / entry["vendored"]).read_bytes()).hexdigest()
        assert on_disk == entry["sha256"], f"{entry['id']} drifted from its pinned bytes"
        assert key not in vc.CORPUS_REPAIRS


def test_sir_v4_carries_its_fix():
    """What #543 re-sourced it for, stated on the text: no year-selection chain falls back
    to 0 after day 1461, and each season opens on ``t>t_start()`` (lanl/bngsim#545)."""
    entry = _BY_KEY[("bngl_models", "my_models/ode/SIR_v4.bngl")]
    text = (BNG / "models" / entry["vendored"]).read_text()
    last_year = [ln.strip() for ln in text.splitlines() if "if(t<=1461," in ln]
    assert len(last_year) == 4, last_year
    assert not [ln for ln in last_year if ln.endswith(",0))))")], last_year
    assert "scaled_time()=if(t>t_start() && t<=t_end()," in text
