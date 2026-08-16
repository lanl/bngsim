"""The suite's artifact caches belong to the suite (issue #372).

bngsim resolves two content-addressed caches at import — compiled ``.so``
artifacts and BNG2.pl-generated networks — and until ``conftest`` redirected them
the test suite wrote into the ones a *user* runs ``bngsim-cache`` against. The
damage was cumulative (146 MB of orphans on the box the issue was filed from),
and part of it was not even reachable: a test that monkeypatches the codegen key
to stand in for another install left an artifact under a key no install has ever
had, which since #363 shows up as a row in somebody's cache report.

These tests pin the redirect itself. The two files that used to leave the
recognizable debris — ``test_prepare_codegen_memo.py`` (fabricated key) and
``test_codegen_sensitivity.py`` (invented model hash) — assert their own
containment where the artifact is written, since that is where the leak was.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import bngsim._bngl_loader as bl
import bngsim._codegen as cg
import pytest
from bngsim._bngpath import resolve_bng

_BNG = resolve_bng()
needs_bng2 = pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")

_USER_CACHE = Path.home() / ".cache" / "bngsim"


def test_neither_live_cache_is_the_users_own() -> None:
    """The whole point: a suite run must not deposit anything in ``~/.cache/bngsim``.

    Asserted on the resolved directories rather than on what a run happens to
    leave behind, because the failure this guards against is silent — artifacts
    land in a real cache, every test still passes, and the only symptom is a
    number in ``bngsim-cache info`` weeks later.
    """
    for cache in (cg.CACHE_DIR, bl.CACHE_DIR):
        assert not cache.is_relative_to(_USER_CACHE), f"{cache} is inside the user's cache"


def test_the_redirect_reaches_subprocesses() -> None:
    """A child process that imports bngsim must land in the same place.

    ``CACHE_DIR`` is resolved once at import from an env var, so patching the
    module attribute alone would isolate this process and leave every subprocess
    — the ``python -m bngsim.cache`` entry points, anything a fitting harness
    spawns — writing to the user's cache. Setting the env var is what covers
    them, and this is the test that would notice it being dropped.
    """
    probe = (
        "import bngsim._codegen as c, bngsim._bngl_loader as b;"
        "print(c.CACHE_DIR);print(b.CACHE_DIR)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.splitlines()  # not .split(): a checkout path may contain spaces
    assert [Path(p) for p in out] == [cg.CACHE_DIR, bl.CACHE_DIR]


def test_an_explicit_root_wins_over_every_default(
    artifact_caches: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``BNGSIM_TEST_CACHE_DIR`` is the knob: node-local scratch, a CI-restored
    warm cache, or a throwaway for a guaranteed cold run. It beats the
    ``.pytest_cache`` default even when the cache provider is available."""
    monkeypatch.setenv(artifact_caches.root_env, str(tmp_path / "elsewhere"))
    root, ephemeral = artifact_caches.resolve_root(SimpleNamespace(cache=None))
    assert (root, ephemeral) == (tmp_path / "elsewhere", False)
    assert artifact_caches.resolve_root(SimpleNamespace(cache=object()))[0] == root


def test_the_default_root_goes_through_the_pytest_cache(
    artifact_caches: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Routing through ``Cache.mkdir`` rather than composing the path by hand is
    what makes ``pytest --cache-clear`` wipe these artifacts: it clears exactly
    the ``d/`` subtree that call writes into. Verified against a stub so the
    assertion holds on the CI legs that disable the cache provider outright."""
    monkeypatch.delenv(artifact_caches.root_env, raising=False)
    asked: list[str] = []

    class _StubCache:
        def mkdir(self, name: str) -> Path:
            asked.append(name)
            d = tmp_path / "d" / name
            d.mkdir(parents=True)
            return d

    root, ephemeral = artifact_caches.resolve_root(SimpleNamespace(cache=_StubCache()))
    assert (root, ephemeral) == (tmp_path / "d" / "bngsim", False)
    assert asked == ["bngsim"]  # a single component; Cache.mkdir rejects separators


def test_without_a_cache_provider_the_root_is_a_throwaway(
    artifact_caches: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-p no:cacheprovider`` (what every CI leg passes) says "do not write under
    the rootdir". Honor it with a temp dir, and report it as this session's to
    remove — a cold run, which a fresh runner gets regardless."""
    monkeypatch.delenv(artifact_caches.root_env, raising=False)
    root, ephemeral = artifact_caches.resolve_root(SimpleNamespace(cache=None))
    try:
        assert ephemeral is True
        assert root.is_dir()
        assert not root.is_relative_to(Path.cwd())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_this_session_took_one_of_those_two_branches(request: pytest.FixtureRequest) -> None:
    """The stubs above pin the branches; this pins that the running session is on
    one of them, whichever leg it is.

    ``getattr`` because ``-p no:cacheprovider`` removes the attribute rather than
    nulling it — reading ``config.cache`` directly raised ``AttributeError`` and
    took down every CI leg with an INTERNALERROR before this was written that way.
    """
    root = cg.CACHE_DIR.parent
    cache = getattr(request.config, "cache", None)
    if cache is None:  # -p no:cacheprovider
        assert not root.is_relative_to(request.config.rootpath)
    else:
        assert root == cache.mkdir("bngsim")


@needs_bng2
def test_a_generated_network_lands_in_the_test_cache(tmp_path: Path) -> None:
    """End to end for the second cache. Network generation is the half the issue
    did not name, and it leaks the same way: two BNGL test files alone left seven
    ``.net`` files in the reporter's ``~/.cache/bngsim/networks``."""
    src = tmp_path / "isolated.bngl"
    src.write_text(
        "begin model\n"
        "begin parameters\n  k 1.0\nend parameters\n"
        "begin species\n  A() 100\nend species\n"
        "begin reaction rules\n  A() -> 0 k\nend reaction rules\n"
        "end model\n"
    )
    net = bl.bngl_to_net(str(src))
    assert net.parent == bl.CACHE_DIR
    assert not net.is_relative_to(_USER_CACHE)
