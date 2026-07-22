"""Regression guards for :mod:`_core.bngpath` — the shared BNG2.pl resolver.

The bug this module replaced: six near-duplicate helpers, and the test-side ones
asked the installed PyBioNetGen for its bundled BNG2.pl *first*, consulting
``$BNGPATH`` / ``$BNG2_PL`` only from an ``except`` branch. With bionetgen
importable an explicit ``export BNGPATH=...`` was therefore silently ignored —
you could point at a different BioNetGen and the suite would keep using the
bundled one, with nothing said. These pin the precedence so that inversion
cannot come back.
"""

from __future__ import annotations

import pytest
from _core import bngpath


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
