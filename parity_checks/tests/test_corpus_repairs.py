"""GH #182: repair-at-import machinery for known-malformed UPSTREAM models.

A handful of corpus models carry a plain modeler typo upstream. Patching the local
checkout would only fix one machine, so ``vendor_corpus.apply_corpus_repairs`` repairs
the model at vendor time and records the change in the manifest. The repair is
count-guarded so an upstream change fails loudly instead of silently mis-applying.
This is for genuine malformations only — a well-formed model with an awkward identifier
(e.g. a parameter named ``lambda``) is NOT repaired (see test_resolve_token_keyword).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BNG = HERE.parent / "bng_parity"
sys.path.insert(0, str(BNG))

import pytest  # noqa: E402
import vendor_corpus as vc  # noqa: E402

_KEY = ("bngl_models", "my_models/ode/HarmonicOscillator_redo_v1.bngl")


def test_harmonic_oscillator_typo_is_registered():
    assert _KEY in vc.CORPUS_REPAIRS
    spec = vc.CORPUS_REPAIRS[_KEY]
    assert spec["issue"] == "internal#182"


def test_repair_fixes_the_six_unbalanced_parens():
    src = "".join(
        f'setConcentration("IGF1(ds,hs,label~cold",{v})\n'
        for v in (12646, 126465, 1264649, 12646494, 126464940, 1264649400)
    )
    out, record = vc.apply_corpus_repairs(*_KEY, src)
    assert record is not None and record["edits"] == 6
    assert 'IGF1(ds,hs,label~cold",' not in out  # malformed gone
    assert out.count('IGF1(ds,hs,label~cold)",') == 6  # all closed


def test_count_guard_raises_on_upstream_drift():
    # Only 2 occurrences where the registry expects 6 -> loud failure, not a silent patch.
    src = 'setConcentration("IGF1(ds,hs,label~cold",1)\nsetConcentration("IGF1(ds,hs,label~cold",2)\n'
    with pytest.raises(SystemExit):
        vc.apply_corpus_repairs(*_KEY, src)


def test_unregistered_model_is_a_clean_noop():
    out, record = vc.apply_corpus_repairs("bngl_models", "some/other/model.bngl", "text")
    assert out == "text"
    assert record is None


def test_vendored_model_on_disk_is_repaired_and_manifest_records_it():
    manifest = json.loads((BNG / "manifest.json").read_text())["records"]
    entry = next(e for e in manifest if e["source"] == _KEY[0] and e["relpath"] == _KEY[1])
    assert entry.get("repairs"), "manifest must record the repair provenance"
    assert entry["repairs"][0]["edits"] == 6
    # A repaired model carries the __PATCHED sentinel in its vendored model_id (so the
    # patch is visible in jobs/golden/matrix); the upstream relpath stays the real name.
    assert entry["vendored"].endswith("__PATCHED.bngl")
    assert "__PATCHED" not in entry["relpath"]

    # The vendored file on disk holds the REPAIRED text (== what runs).
    text = (BNG / "models" / entry["vendored"]).read_text()
    assert 'IGF1(ds,hs,label~cold",' not in text  # no malformed pattern remains
    assert text.count('IGF1(ds,hs,label~cold)",') == 6


# GH #173: a second class of upstream defect — a stray `par_scan_vals=>[...]` arg left on an
# explicit `simulate` (belongs to `parameter_scan`). BNG2.pl ignores it, but PyBioNetGen's
# bngmodel parser rejects it, silently routing the job off bngsim to legacy BNG2.pl. Keyed by
# the MANIFEST source (`rulehub`), though the bytes resolve from BNGL-Models/bench_rulehub/.
_EX3_KEY = ("rulehub", "bench_rulehub/example3_fit.bngl")


def test_example3_fit_par_scan_vals_repair_is_registered():
    assert _EX3_KEY in vc.CORPUS_REPAIRS
    spec = vc.CORPUS_REPAIRS[_EX3_KEY]
    assert spec["issue"] == "internal#173"
    # A single count-guarded edit that deletes the stray arg (replace -> "").
    assert len(spec["edits"]) == 1
    find, replace, expected = spec["edits"][0]
    assert find.startswith("par_scan_vals=>[") and replace == "" and expected == 1


def test_example3_fit_repair_drops_the_stray_arg():
    # Build a minimal actions block from the registered find string so the test can never
    # drift from the registry, then confirm the repair removes it and leaves the simulate.
    find = vc.CORPUS_REPAIRS[_EX3_KEY]["edits"][0][0]
    src = (
        "begin actions\n\nsimulate({\\\n                "
        + find
        + 'method=>"nf"})\n\nend actions\n'
    )
    out, record = vc.apply_corpus_repairs(*_EX3_KEY, src)
    assert record is not None and record["edits"] == 1
    assert "par_scan_vals" not in out  # stray arg gone
    assert 'simulate({\\\n                method=>"nf"})' in out  # single simulate intact


def test_example3_fit_count_guard_raises_when_arg_absent():
    # An upstream fix (the arg already dropped) must fail loudly, not silently no-op.
    with pytest.raises(SystemExit):
        vc.apply_corpus_repairs(*_EX3_KEY, 'simulate({method=>"nf"})\n')


def test_example3_fit_vendored_on_disk_is_repaired_and_manifest_records_it():
    manifest = json.loads((BNG / "manifest.json").read_text())["records"]
    entry = next(e for e in manifest if e["source"] == _EX3_KEY[0] and e["relpath"] == _EX3_KEY[1])
    assert entry.get("repairs"), "manifest must record the repair provenance"
    assert entry["repairs"][0]["edits"] == 1
    assert entry["vendored"].endswith("__PATCHED.bngl")
    assert "__PATCHED" not in entry["relpath"]
    # sha256 is the UPSTREAM (pristine) hash, not the repaired bytes (provenance + drift).
    assert entry["sha256"] == "9c49d0bc5a1445ea068fcad6ffdfecc0a053c5eb49974272cddb5c85985cec34"

    # The vendored file on disk holds the REPAIRED text: the stray ARGUMENT is gone. (The
    # provenance comment still mentions the word "par_scan_vals", so anchor on the `=>[`
    # syntax that made it an argument — exactly what the count-guarded find keys on.)
    text = (BNG / "models" / entry["vendored"]).read_text()
    assert "par_scan_vals=>[" not in text
