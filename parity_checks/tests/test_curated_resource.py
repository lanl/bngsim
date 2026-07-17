"""Guard the CURATED_SIX re-source mechanism (vendor_corpus.CURATED_SIX / resolve()).

The six "representative published" models behind the paper's Supplementary Table S3 and the
KLU-scaling figure keep their rulehub membership IDs but resolve their corrected, bug-fixed
bytes from wshlavacek/BNGL-Models at the dedicated ``bngl_curated`` pin (the bench_rulehub
precedent: a rulehub-labelled entry sourced from BNGL-Models). This guard locks that contract
so a re-pin / re-vendor can't silently drop, mis-key, or mis-provenance the re-source:

  * every CURATED_SIX key is a live manifest membership entry (no stale keys);
  * resolve() redirects each to the ``bngl_curated`` pin at its house-curated path;
  * the committed manifest records BNGL-Models provenance + ``curated: true`` for each;
  * Barua2013 keeps its historical ``__PATCHED`` corpus ID (stable downstream), and none of
    the six are import-patched (they are used verbatim, not repaired).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BNG = HERE.parent / "bng_parity"


def _load(unique_name: str, path: Path):
    # Load under a UNIQUE module name (not the bare basename) so this test never
    # caches bng_parity's `overrides`/`vendor_corpus` into sys.modules under a name
    # that rr_parity's tests import (both packages ship an `overrides.py`; a bare
    # `import overrides` would collide across suites — test-isolation). Register in
    # sys.modules under that unique name before exec so @dataclass can resolve its
    # module (dataclasses looks the defining module up in sys.modules).
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


vc = _load("bng_parity_vendor_corpus", BNG / "vendor_corpus.py")
ov = _load("bng_parity_overrides", BNG / "overrides.py")

_MANIFEST = json.loads((BNG / "manifest.json").read_text())["records"]
_BY_KEY = {(m["source"], m["relpath"]): m for m in _MANIFEST}
_CURATED_PIN = vc.PINS["bngl_curated"].sha


def test_there_are_exactly_six():
    assert len(vc.CURATED_SIX) == 6


def test_curated_keys_are_live():
    # Every re-source must target a model actually in the manifest membership (mirrors
    # test_corpus_repairs_keys_are_live) — no stale keys left behind after a corpus refresh.
    live = set(_BY_KEY)
    stale = [k for k in vc.CURATED_SIX if k not in live]
    assert not stale, f"CURATED_SIX has stale keys (no matching manifest model): {stale}"


def test_resolve_redirects_each_to_the_curated_pin():
    for (source, relpath), spec in vc.CURATED_SIX.items():
        repo_key, path = vc.resolve(source, relpath)
        assert repo_key == "bngl_curated", f"{relpath} did not redirect to the curated pin"
        assert path == spec["src"]
    # The curated pin is BNGL-Models (a second pin of the same repo, newer commit).
    assert vc.PINS["bngl_curated"].slug == "wshlavacek/BNGL-Models"
    assert vc.PINS["bngl_curated"].sha != vc.PINS["bngl_models"].sha


def test_manifest_records_curated_provenance():
    for (source, relpath), spec in vc.CURATED_SIX.items():
        entry = _BY_KEY[(source, relpath)]
        assert entry.get("curated") is True, f"{relpath} manifest record missing curated flag"
        o = entry["origin"]
        assert o["repo"] == "wshlavacek/BNGL-Models"
        assert o["commit"] == _CURATED_PIN  # pinned, never @main
        assert o["path"] == spec["src"]  # points at the house-curated body
        # sha256 is the on-disk hash here (the file is used verbatim, not import-repaired).
        assert entry["patched"] is False


def test_barua2013_keeps_patched_id_and_is_no_longer_import_repaired():
    key = ("rulehub", "Published/Barua2013/Barua_2013.bngl")
    entry = _BY_KEY[key]
    assert entry["vendored"].endswith("Barua_2013__PATCHED.bngl")  # stable historical ID
    assert vc.CURATED_SIX[key]["out"] == "Published/Barua2013/Barua_2013__PATCHED.bngl"
    # The former atoll->atol import repair is retired now that the corrected body is sourced.
    assert key not in vc.CORPUS_REPAIRS


def test_none_of_the_six_are_import_repaired():
    for key in vc.CURATED_SIX:
        assert key not in vc.CORPUS_REPAIRS


def test_horizon_added_models_are_no_longer_rehabbed():
    # Kocieniewski_2012 and fceri_fyn shipped protocol-less upstream (hence REHAB); the
    # curated bodies carry their own ODE horizon, so the rehab fixture must be gone or the
    # runner would inject t_end=1 over the model's real horizon.
    for mid in (
        "slow/rulehub/Published/Kocieniewski2012/Kocieniewski_2012.bngl",
        "slow/rulehub/Published/fcerifyn/fceri_fyn.bngl",
    ):
        assert mid not in ov.REHAB
