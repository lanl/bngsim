"""Regression guards for the shared BNG2.pl resolver, now :mod:`bngsim._bngpath`.

The bug this module replaced: six near-duplicate helpers, and the test-side ones
asked the installed PyBioNetGen for its bundled BNG2.pl *first*, consulting
``$BNGPATH`` / ``$BNG2_PL`` only from an ``except`` branch. With bionetgen
importable an explicit ``export BNGPATH=...`` was therefore silently ignored —
you could point at a different BioNetGen and the suite would keep using the
bundled one, with nothing said. These pin the precedence so that inversion
cannot come back.

The resolver moved into the shipped package for GH #162 (``Model.from_bngl``
needs it, and ``parity_checks/`` is not packaged), so these test it where it now
lives; ``_core.bngpath`` re-exports, and :func:`test_core_reexports_the_shipped_resolver`
is what keeps that a re-export rather than a seventh copy.
"""

from __future__ import annotations

import shutil

import pytest
from bngsim import _bngpath as bngpath


@pytest.fixture(autouse=True)
def _no_bng2_on_path(monkeypatch):
    """Neutralize the ``$PATH`` mechanism, so each test measures the one it names.

    Every precedence test below fixes the env vars and the bundled copy but says
    nothing about ``$PATH`` — which became a mechanism in GH #162. On a machine
    that happens to have ``BNG2.pl`` on ``PATH`` those tests would resolve to it
    and fail for a reason having nothing to do with what they assert. ``perl``
    still resolves normally, since ``BngResolution.ok`` depends on it.
    """
    real = shutil.which
    monkeypatch.setattr(
        bngpath.shutil,
        "which",
        lambda name, *a, **k: None if name == "BNG2.pl" else real(name, *a, **k),
    )


@pytest.fixture
def fake_bng(tmp_path):
    """A directory that looks like a BioNetGen install."""

    def _make(name: str) -> str:
        root = tmp_path / name
        (root / "bin").mkdir(parents=True)
        (root / "BNG2.pl").write_text("#!/usr/bin/env perl\n")
        return str(root)

    return _make


def test_env_beats_bundled(monkeypatch, fake_bng):
    """An explicit env var must OVERRIDE PyBioNetGen's bundled copy (the old bug)."""
    explicit = fake_bng("explicit")
    monkeypatch.setattr(bngpath, "_bundled_bngpath", lambda: (fake_bng("bundled"), "bundled"))
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.setenv("BNGPATH", explicit)

    r = bngpath.resolve_bng()
    assert r.ok
    assert r.source == bngpath.ENV_BNGPATH
    assert str(r.root) == explicit


def test_bng2_pl_beats_bngpath(monkeypatch, fake_bng):
    """$BNG2_PL is the more specific override, so it wins over $BNGPATH."""
    monkeypatch.setenv("BNG2_PL", fake_bng("specific"))
    monkeypatch.setenv("BNGPATH", fake_bng("general"))

    r = bngpath.resolve_bng()
    assert r.source == bngpath.ENV_BNG2_PL
    assert "specific" in str(r.root)


def test_explicit_argument_beats_everything(monkeypatch, fake_bng):
    monkeypatch.setenv("BNG2_PL", fake_bng("env"))
    r = bngpath.resolve_bng(fake_bng("arg"))
    assert r.source == bngpath.EXPLICIT
    assert "arg" in str(r.root)


def test_bundled_used_when_no_env(monkeypatch, fake_bng):
    bundled = fake_bng("bundled")
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.delenv("BNGPATH", raising=False)
    monkeypatch.setattr(bngpath, "_bundled_bngpath", lambda: (bundled, bundled))

    r = bngpath.resolve_bng()
    assert r.ok
    assert r.source == bngpath.BUNDLED


def test_stale_env_falls_through_instead_of_poisoning(monkeypatch, fake_bng):
    """A env var pointing nowhere must not veto a working install behind it.

    The old helpers took the first non-empty candidate and then failed on it, so
    one stale export made an otherwise-working machine look BNG-less.
    """
    bundled = fake_bng("bundled")
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.setenv("BNGPATH", "/nonexistent/definitely/not/here")
    monkeypatch.setattr(bngpath, "_bundled_bngpath", lambda: (bundled, bundled))

    r = bngpath.resolve_bng()
    assert r.ok
    assert r.source == bngpath.BUNDLED


def test_direct_bng2_pl_file_path_is_accepted(monkeypatch, fake_bng):
    """$BNGPATH may be the BNG2.pl script itself, not just its folder."""
    root = fake_bng("asfile")
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.setenv("BNGPATH", f"{root}/BNG2.pl")

    r = bngpath.resolve_bng()
    assert r.ok
    assert str(r.bng2_pl) == f"{root}/BNG2.pl"
    assert str(r.root) == root


def test_failure_names_every_mechanism_tried(monkeypatch):
    """The whole point of the rewrite: a failure has to be actionable."""
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.delenv("BNGPATH", raising=False)
    monkeypatch.setattr(bngpath, "_bundled_bngpath", lambda: (None, "not installed"))

    r = bngpath.resolve_bng()
    assert not r.ok
    why = r.why_not()
    for mechanism in (bngpath.ENV_BNG2_PL, bngpath.ENV_BNGPATH, bngpath.BUNDLED):
        assert mechanism in why, f"{mechanism} missing from the failure message"
    # and it must say what to DO about it
    assert "--group parity" in why or "$BNGPATH" in why


def test_skip_reason_is_none_when_usable(monkeypatch, fake_bng):
    monkeypatch.setenv("BNG2_PL", fake_bng("ok"))
    assert bngpath.skip_reason() is None


def test_path_lookup_loses_to_env_but_beats_bundled(monkeypatch, fake_bng):
    """``BNG2.pl`` on ``$PATH`` sits between the env vars and PyBioNetGen.

    That mechanism arrived with GH #162: ``bngsim.convert._bng2.find_bng2`` had
    it and this resolver did not, so folding the two together had to keep it or
    quietly stop finding a BNG2.pl somebody had put on their PATH. It ranks
    below the env vars (an override is an override) and above the bundled copy
    (putting it on PATH is a deliberate act; having a package installed is not).
    """
    on_path = fake_bng("onpath")
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.delenv("BNGPATH", raising=False)
    monkeypatch.setattr(
        bngpath.shutil,
        "which",
        lambda n, *a, **k: f"{on_path}/BNG2.pl" if n == "BNG2.pl" else "/usr/bin/perl",
    )
    monkeypatch.setattr(bngpath, "_bundled_bngpath", lambda: (fake_bng("bundled"), "bundled"))

    r = bngpath.resolve_bng()
    assert r.source == bngpath.ON_PATH
    assert str(r.root) == on_path

    monkeypatch.setenv("BNGPATH", fake_bng("env"))
    assert bngpath.resolve_bng().source == bngpath.ENV_BNGPATH


def test_no_perl_is_reported_as_its_own_failure(monkeypatch, fake_bng):
    """A found BNG2.pl with no perl must not read as "no BioNetGen".

    They need opposite fixes, and this is the stock-Windows case: `pip install
    'bngsim[bngl]'` succeeds there and buys nothing, so the message has to say
    the missing piece is the interpreter.
    """
    monkeypatch.setenv("BNG2_PL", fake_bng("present"))
    monkeypatch.setattr(bngpath.shutil, "which", lambda n, *a, **k: None)

    r = bngpath.resolve_bng()
    assert not r.ok
    assert r.bng2_pl is not None
    assert "perl" in r.why_not()


def test_core_reexports_the_shipped_resolver():
    """``_core.bngpath`` must stay a re-export, not a copy (GH #162).

    The whole module exists because duplicated locators drifted. A parity script
    reaching it through ``_core`` and a ``Model.from_bngl`` reaching it through
    ``bngsim`` have to be resolving with the same code, so assert object
    identity rather than equal behavior.
    """
    from _core import bngpath as core_bngpath

    assert core_bngpath.resolve_bng is bngpath.resolve_bng
    assert core_bngpath.skip_reason is bngpath.skip_reason
    assert core_bngpath.BngResolution is bngpath.BngResolution
