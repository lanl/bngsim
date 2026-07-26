"""Issue #51 — the codegen cache key must react to a codegen change.

The ``.net`` path keys its compiled ``.so`` on the model content plus a
hand-maintained ``_CODEGEN_VERSION`` constant, not on the generated C. That made
the constant load-bearing: a change that altered the emitted forward-sensitivity
RHS **without** bumping it was invisible to any machine with a warm
``~/.cache/bngsim/codegen``, which kept loading the stale library and returning
the pre-change numbers. #41 and #43 both shipped that way — on a warm cache the
reported ``dy/dp`` for every fitted parameter came back ``0`` while a cold cache
gave values agreeing with AMICI to 1.5e-5. Same wheel, same model; only the
cache differed.

``_CODEGEN_CACHE_KEY`` now folds a digest of the emitters' own source in beside
the constant, so editing an emitter invalidates stale artifacts whether or not
anyone remembered to bump it. These tests pin both halves: the digest reacts to
a source change, and a model hash computed under an older key is never served to
a newer one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import bngsim._codegen as cg
import numpy as np
import pytest

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _fake_src_tree(root: Path, marker: str = "") -> Path:
    """A directory holding a stand-in for every module the digest covers."""
    root.mkdir(parents=True, exist_ok=True)
    for name in cg._CODEGEN_SOURCE_MODULES:
        (root / f"{name}.py").write_text(f"# {name}\n{marker}\n")
    return root


class TestSourceDigest:
    """The digest is what makes forgetting the constant harmless."""

    def test_reacts_to_an_emitter_source_change(self, tmp_path):
        """The whole point: edit an emitter, get a different key — no constant
        bump required. Every module the digest covers must count, since a change
        in any of them can change the generated C."""
        tree = _fake_src_tree(tmp_path / "pkg")
        base = cg._compute_codegen_source_digest(tree)
        assert base != ""

        for name in cg._CODEGEN_SOURCE_MODULES:
            original = (tree / f"{name}.py").read_text()
            (tree / f"{name}.py").write_text(original + "\n# an emitter change\n")
            assert cg._compute_codegen_source_digest(tree) != base, (
                f"editing {name}.py left the codegen cache key unchanged — a change "
                f"to it would be silently inert on a warm cache (issue #51)"
            )
            (tree / f"{name}.py").write_text(original)

        # Restoring every file restores the digest: the key is content-addressed,
        # not order- or time-dependent, so a cache stays warm across processes.
        assert cg._compute_codegen_source_digest(tree) == base

    def test_degrades_to_empty_when_sources_are_unreadable(self, tmp_path):
        """A ``.pyc``-only or zipped install cannot read the sources. That must
        fall back to ``_CODEGEN_VERSION`` alone — the pre-#51 behavior — never to
        something weaker or to a crash at import."""
        empty = tmp_path / "no_sources"
        empty.mkdir()
        assert cg._compute_codegen_source_digest(empty) == ""

    def test_live_key_carries_both_halves(self):
        """The shipped key must actually contain the constant and a real digest;
        an empty digest here would mean the package cannot read its own source."""
        assert cg._CODEGEN_CACHE_KEY.startswith(cg._CODEGEN_VERSION + "+")
        assert cg._CODEGEN_SOURCE_DIGEST, "codegen source digest is empty in a source install"
        assert f"{cg._CODEGEN_VERSION}+{cg._CODEGEN_SOURCE_DIGEST}" == cg._CODEGEN_CACHE_KEY


class TestModelHashHonorsTheKey:
    def test_model_hash_changes_with_the_cache_key(self, tmp_path, monkeypatch):
        net = tmp_path / "m.net"
        shutil.copy(DATA_DIR / "nested_derived_rate_const.net", net)

        current = cg.compute_model_hash(str(net))
        monkeypatch.setattr(cg, "_CODEGEN_CACHE_KEY", cg._CODEGEN_CACHE_KEY + "+moved")
        assert cg.compute_model_hash(str(net)) != current

    def test_model_hash_is_stable_for_an_unchanged_key(self, tmp_path):
        """Caching must still work — the key reacts to codegen changes, not to
        the phase of the moon."""
        net = tmp_path / "m.net"
        shutil.copy(DATA_DIR / "nested_derived_rate_const.net", net)
        assert cg.compute_model_hash(str(net)) == cg.compute_model_hash(str(net))


@needs_cc
class TestStaleArtifactIsNotServed:
    """The end-to-end shape of the bug, on the nested-derived-parameter model
    (#41) whose sensitivities were among those the stale cache zeroed."""

    _SAMPLE_TIMES = list(np.linspace(0.0, 2.0, 21))

    @staticmethod
    def _analytic(net: str) -> np.ndarray:
        import bngsim

        m = bngsim.Model.from_net(net)
        r = bngsim.Simulator(
            m, method="ode", sensitivity_params=["kcr"], codegen=True, net_path=net
        ).run(sample_times=TestStaleArtifactIsNotServed._SAMPLE_TIMES, rtol=1e-11, atol=1e-13)
        return np.asarray(r.sensitivities)[:, :, 0]

    @classmethod
    def _rebuild_fd(cls, net: str, tmp_path: Path) -> np.ndarray:
        """Central FD w.r.t. kcr, rebuilding the model from a perturbed .net."""
        import re

        import bngsim

        src = Path(net).read_text()

        def traj(kcr: float) -> np.ndarray:
            txt = re.sub(r"(\bkcr\s+)[0-9.]+", rf"\g<1>{kcr}", src, count=1)
            p = tmp_path / f"fd_{kcr}.net"
            p.write_text(txt)
            m = bngsim.Model.from_net(str(p))
            r = bngsim.Simulator(m, method="ode").run(
                sample_times=cls._SAMPLE_TIMES, rtol=1e-12, atol=1e-14
            )
            return np.asarray(r.species)

        return (traj(0.3303) - traj(0.3297)) / 0.0006

    def test_artifact_keyed_under_an_older_key_is_ignored(self, tmp_path, monkeypatch):
        """Pre-populate the cache with a deliberately corrupt artifact at the hash
        the *previous* cache key produced, then run for real.

        Under the pre-#51 design an emitter change left the key untouched, so this
        artifact would have been found and loaded. The assertion that the two
        hashes differ is the fix; the FD comparison confirms the run really did
        recompile and produce the analytic chain rule rather than the zeros the
        stale library returned.
        """
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(cg, "CACHE_DIR", cache)
        cg._PREPARE_CODEGEN_MEMO.clear()

        net = tmp_path / "nested.net"
        shutil.copy(DATA_DIR / "nested_derived_rate_const.net", net)

        # The hash an older codegen (different emitters, same _CODEGEN_VERSION)
        # would have produced for this same .net.
        monkeypatch.setattr(cg, "_CODEGEN_CACHE_KEY", "23+staleemitterdigest")
        stale_hash = cg.compute_model_hash(str(net))
        stale_so = cache / f"rhs_{stale_hash}{cg._shared_lib_suffix()}"
        stale_so.write_bytes(b"not a shared library - loading this must never happen\n")

        monkeypatch.undo()
        monkeypatch.setattr(cg, "CACHE_DIR", cache)
        cg._PREPARE_CODEGEN_MEMO.clear()

        current_hash = cg.compute_model_hash(str(net))
        assert current_hash != stale_hash, (
            "an emitter change left the cache key unchanged, so the stale .so would "
            "be reused and the fix would be silently inert (issue #51)"
        )

        sx = self._analytic(str(net))
        fd = self._rebuild_fd(str(net), tmp_path)

        # kcr reaches the rate laws only through a1prime and the nested a2prime,
        # so a stale pre-#41 library reports these as identically zero.
        assert np.abs(sx).max() > 1e-3, "dy/dkcr is ~0 — a stale artifact was served"
        np.testing.assert_allclose(sx, fd, rtol=1e-3, atol=1e-6)

        # The corrupt file is still sitting there unread, which is the point.
        assert stale_so.exists()
