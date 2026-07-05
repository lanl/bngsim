"""Corpus-cleanliness CI guard (the durable defense against re-vendor regressions).

The whole point of `vendor_corpus.CORPUS_REPAIRS` is that a known-broken upstream model
is repaired AT IMPORT, so the on-disk corpus is always well-formed. This guard asserts
that invariant over the committed corpus: if a re-pin/re-vendor ever ships a model with
malformed action syntax that no repair covered, this test fails loudly — the typo is
caught here, not painfully rediscovered in a sweep months later.

It also keeps the repair registry honest: every CORPUS_REPAIRS key must map to a live
manifest model (no stale entries), mirroring build_jobs' stale-override warning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BNG = HERE.parent / "bng_parity"
sys.path.insert(0, str(BNG))

import vendor_corpus as vc  # noqa: E402

MODELS = BNG / "models"


def test_committed_corpus_has_no_malformed_action_args():
    offenders = []
    for f in sorted(MODELS.rglob("*.bngl")):
        bad = vc.find_malformed_action_args(f.read_text(errors="replace"))
        if bad:
            rel = f.relative_to(MODELS)
            offenders.append(
                f"{rel}: " + "; ".join(f'L{ln} {act}("{arg}")' for ln, act, arg in bad[:3])
            )
    assert not offenders, (
        "Malformed action arguments in the vendored corpus — add a CORPUS_REPAIRS entry "
        "in vendor_corpus.py (and re-vendor) so the model is repaired at import:\n  "
        + "\n  ".join(offenders)
    )


def test_corpus_repairs_keys_are_live():
    # Every repair must target a model that is actually in the manifest membership — no
    # stale entries left behind when a model leaves the corpus (cf. SYMBOL_RENAME's dead
    # frac entries that build_jobs warns about).
    manifest = json.loads((BNG / "manifest.json").read_text())["records"]
    live = {(m["source"], m["relpath"]) for m in manifest}
    stale = [key for key in vc.CORPUS_REPAIRS if key not in live]
    assert not stale, f"CORPUS_REPAIRS has stale keys (no matching manifest model): {stale}"


def test_known_harmonic_oscillator_repair_is_registered():
    # Regression for the GH #182 typo: the repair must stay registered so a re-vendor
    # keeps fixing the 6 unbalanced-paren setConcentration patterns.
    key = ("bngl_models", "my_models/ode/HarmonicOscillator_redo_v1.bngl")
    assert key in vc.CORPUS_REPAIRS
