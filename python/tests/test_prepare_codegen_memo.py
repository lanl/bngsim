"""T2 — prepare_codegen process-local memo.

Repeated ``Simulator(codegen=True, net_path=...)`` construction on an unchanged
.net otherwise re-reads, re-parses, and SHA-256-hashes the file on every call
just to resolve an already-cached .so. The memo keyed by ``(net_path, mtime,
_CODEGEN_VERSION)`` returns the cached .so after only a few ``stat()`` calls.

These tests prove the file is parsed+hashed once across N calls, that touching
the .net (mtime change) forces a recompute, and that the resolved .so is
identical across all calls.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import bngsim._codegen as cg
import pytest

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")


@needs_cc
def test_memo_parses_and_hashes_once(simple_decay_net: Path, tmp_path, monkeypatch) -> None:
    """N prepare_codegen calls on an unchanged .net parse+hash it exactly once."""
    net = tmp_path / "memo_model.net"
    shutil.copy(simple_decay_net, net)

    # Warm the compiled .so cache first, then clear only the memo, so the counted
    # loop measures the MEMO (cache-hit) path the test is about. The cold-compile
    # path legitimately re-parses the .net more than once (RHS + sensitivity-RHS
    # generation each parse it), and whether this content's .so is already cached is
    # global-test-order dependent: an fd-codegen of this model caches under a
    # ":codegen_outputs" key (GH #163), not the plain one, so the plain .so may be
    # cold by the time this test runs. The warm-up makes the assertion order-robust.
    cg.prepare_codegen(str(net))
    cg._PREPARE_CODEGEN_MEMO.clear()

    parse_calls = {"n": 0}
    hash_calls = {"n": 0}
    real_parse = cg._parse_net_file
    real_hash = cg.compute_model_hash

    def counting_parse(p):
        parse_calls["n"] += 1
        return real_parse(p)

    def counting_hash(p):
        hash_calls["n"] += 1
        return real_hash(p)

    monkeypatch.setattr(cg, "_parse_net_file", counting_parse)
    monkeypatch.setattr(cg, "compute_model_hash", counting_hash)

    results = [cg.prepare_codegen(str(net)) for _ in range(5)]

    # Parsed + hashed exactly once; calls 2..5 were pure memo hits.
    assert parse_calls["n"] == 1
    assert hash_calls["n"] == 1
    # Every call returned the same resolved .so.
    assert all(r == results[0] for r in results)
    assert results[0].exists()


@needs_cc
def test_memo_invalidates_on_mtime_change(simple_decay_net: Path, tmp_path, monkeypatch) -> None:
    """Touching the .net (mtime change) forces a re-parse+re-hash."""
    net = tmp_path / "memo_touch.net"
    shutil.copy(simple_decay_net, net)

    cg._PREPARE_CODEGEN_MEMO.clear()

    hash_calls = {"n": 0}
    real_hash = cg.compute_model_hash

    def counting_hash(p):
        hash_calls["n"] += 1
        return real_hash(p)

    monkeypatch.setattr(cg, "compute_model_hash", counting_hash)

    first = cg.prepare_codegen(str(net))
    cg.prepare_codegen(str(net))  # memo hit
    assert hash_calls["n"] == 1

    # Bump the mtime forward (content unchanged) → memo must invalidate.
    st = net.stat()
    os.utime(net, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    second = cg.prepare_codegen(str(net))
    assert hash_calls["n"] == 2  # recompute path was taken
    # Content is identical ⇒ same model hash ⇒ same cached .so (results identical).
    assert second == first


@needs_cc
def test_memo_invalidates_on_codegen_version_bump(
    simple_decay_net: Path, tmp_path, monkeypatch
) -> None:
    """A _CODEGEN_VERSION change must invalidate stale memo entries."""
    net = tmp_path / "memo_version.net"
    shutil.copy(simple_decay_net, net)

    cg._PREPARE_CODEGEN_MEMO.clear()
    cg.prepare_codegen(str(net))

    hash_calls = {"n": 0}
    real_hash = cg.compute_model_hash

    def counting_hash(p):
        hash_calls["n"] += 1
        return real_hash(p)

    monkeypatch.setattr(cg, "compute_model_hash", counting_hash)
    monkeypatch.setattr(cg, "_CODEGEN_VERSION", cg._CODEGEN_VERSION + "_bump")

    cg.prepare_codegen(str(net))
    assert hash_calls["n"] == 1  # stale-version memo entry was ignored
