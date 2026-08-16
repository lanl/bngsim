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

import os
import shutil
from pathlib import Path

import bngsim._codegen as cg
import numpy as np
import pytest

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _source_root() -> Path | None:
    """The repository root, or ``None`` when this is not a source checkout.

    Restated rather than imported from ``test_version_consistency`` for the
    reason that file gives: ``run_tests.sh`` copies the tests to a temp
    directory, where a sibling import does not resolve and a fixed ``__file__``
    walk-up does not reach the tree. Same three candidates, same order.
    """
    candidates: list[Path] = []
    if env_data := os.environ.get("BNGSIM_TEST_DATA"):
        candidates.extend(Path(env_data).resolve().parents)
    if env_root := os.environ.get("BNGSIM_SOURCE_ROOT"):
        candidates.append(Path(env_root).resolve())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        py = candidate / "pyproject.toml"
        if py.is_file() and 'name = "bngsim"' in py.read_text(encoding="utf-8"):
            return candidate
    return None


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


class TestTheKeyIsFilenameSafe:
    """Issue #363 — the key is carried in the artifact filename, so its *format* is
    now load-bearing rather than an internal detail.

    ``rhs_<key>_<hash><suffix>`` parses on ``_`` and on the ``.<pid>_<n>`` token of
    an in-flight compile, so a key that grew either character would move the field
    boundary and make ``bngsim-cache`` misread names it wrote itself. Pinned here
    rather than assumed, which is what the issue asked for.
    """

    def test_the_shipped_key_needs_no_escaping(self):
        """``<version>+<digest>`` is alphanumerics and a ``+``, which is legal on
        POSIX and NTFS alike — so the field in the name IS the key, and what
        ``info`` prints is what ``prune --keep-key`` takes."""
        assert cg._artifact_key_field() == cg._CODEGEN_CACHE_KEY
        assert cg._artifact_stem("0123456789abcdef").startswith(f"rhs_{cg._CODEGEN_CACHE_KEY}_")

    @pytest.mark.parametrize("key", ["28+abcdef0123456789", "28+", "28", "test", ""])
    def test_the_field_always_carries_the_marker(self, key, monkeypatch):
        """The marker is what a reader splits on, so it is a shape contract rather
        than a property of today's key format.

        Without it the split has to be "the first underscore", and a ``model_hash``
        may contain those: the real ``rhs_test_sens_0cec059f`` parsed as key ``test``.
        A key that lost its ``+`` would be worse than mislabeled — every artifact
        this install writes would read as pre-#363, i.e. as orphaned, and
        ``prune --orphaned`` would delete a live cache. So the field is given one
        when the key has none.
        """
        monkeypatch.setattr(cg, "_CODEGEN_CACHE_KEY", key)
        assert cg._KEY_FIELD_MARKER in cg._artifact_key_field()

    @pytest.mark.parametrize(
        "key",
        ["28+", "28", "28+abcdef/../.././etc", "28+a b", "28+a_b", "28+a.4711_0", "test"],
        ids=[
            "pyc-only-install",
            "no-digest",
            "path-traversal",
            "space",
            "underscore",
            "temp-token",
            "no-marker",
        ],
    )
    def test_an_exotic_key_still_yields_one_parseable_field(self, key, monkeypatch):
        """The empty digest of a ``.pyc``-only install is the case that actually
        happens; the rest are the ways a future key format could break the name.
        Each must produce a single field a reader can lift back out, and none may
        introduce a path separator or forge the ``<pid>_<n>`` token of a partial.
        """
        from bngsim.cache import KIND_RHS, artifact_key, classify

        monkeypatch.setattr(cg, "_CODEGEN_CACHE_KEY", key)
        field = cg._artifact_key_field()
        name = f"{cg._artifact_stem('0123456789abcdef')}{cg._shared_lib_suffix()}"

        assert "/" not in name and os.sep not in name and "_" not in field
        assert classify(name, is_dir=False) == KIND_RHS
        assert artifact_key(name, is_dir=False) == field

    def test_two_keys_never_collapse_onto_one_field(self, monkeypatch):
        """Offending characters are mapped, not dropped: two installs that differ
        only in an escaped character must still be two rows in ``info`` — and, more
        to the point, must not have one's ``--keep-key`` spare the other's."""
        monkeypatch.setattr(cg, "_CODEGEN_CACHE_KEY", "28+a b")
        one = cg._artifact_key_field()
        monkeypatch.setattr(cg, "_CODEGEN_CACHE_KEY", "28+a  b")
        assert cg._artifact_key_field() != one


class TestTheDocumentedModuleList:
    """Issue #267 — the prose a contributor reads must not carry its own copy.

    ``CONTRIBUTING.md``'s "Changing generated code" section is how someone
    decides whether their edit needs a hand-written ``_CODEGEN_VERSION`` bump:
    the digest covers the modules on the list, and the constant is the escape
    hatch for everything else. It used to restate the list, and #68 added
    ``_switch_sensitivity`` to the tuple without updating it — so from then on
    the file told anyone editing that module to bump a constant the digest had
    already made unnecessary. Over-invalidation, so nothing broke; but a bump
    discards every user's cache and puts an entry in a comment block that is a
    curated record of real reasons.

    The fix is that there is no second copy, and these two tests are what keep
    it that way — one for the pointer, one for the enumeration that must not
    come back partial."""

    @pytest.fixture(scope="class")
    def contributing(self) -> str:
        root = _source_root()
        if root is None or not (root / "CONTRIBUTING.md").is_file():
            pytest.skip("not a source checkout — CONTRIBUTING.md is not present")
        return (root / "CONTRIBUTING.md").read_text(encoding="utf-8")

    def test_it_points_at_the_tuple(self, contributing):
        """Naming the tuple is what makes the list findable without a copy. If a
        rewrite drops the pointer, the section stops answering the question it
        exists to answer."""
        assert "_CODEGEN_SOURCE_MODULES" in contributing

    def test_it_does_not_enumerate_a_subset(self, contributing):
        """A restated list is allowed only if it is complete. Partial is the
        exact shape that drifted: three of four names, indistinguishable from a
        deliberate statement that the fourth is not covered."""
        named = [n for n in cg._CODEGEN_SOURCE_MODULES if f"`{n}.py`" in contributing]
        assert not named or len(named) == len(cg._CODEGEN_SOURCE_MODULES), (
            "CONTRIBUTING.md names "
            + ", ".join(f"{n}.py" for n in named)
            + " but not "
            + ", ".join(f"{n}.py" for n in cg._CODEGEN_SOURCE_MODULES if n not in named)
            + " — a contributor reading it would bump _CODEGEN_VERSION for an edit the "
            "source digest already covers (issue #267). Point at "
            "_CODEGEN_SOURCE_MODULES rather than restating it."
        )


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

        # The hash — and, since issue #363, the name — an older codegen (different
        # emitters, same _CODEGEN_VERSION) would have produced for this same .net.
        # Built through _artifact_stem under the patched key, so it is that install's
        # real filename rather than this file's guess at it.
        monkeypatch.setattr(cg, "_CODEGEN_CACHE_KEY", "23+staleemitterdigest")
        stale_hash = cg.compute_model_hash(str(net))
        stale_so = cache / f"{cg._artifact_stem(stale_hash)}{cg._shared_lib_suffix()}"
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
