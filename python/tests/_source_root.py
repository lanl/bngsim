"""Locate the bngsim source tree without relying on ``__file__`` (issue #578).

``run_tests.sh`` copies the suite into a temp directory so the source tree's
``python/bngsim/`` cannot shadow the *installed* package it exists to exercise.
That copy breaks every ``Path(__file__).resolve().parents[N]`` walk-up: the
parents of ``$TMPDIR/tests/test_x.py`` are unrelated to the repo. Modules that
walked up to reach `benchmarks/` or `parity_checks/` therefore raised at
collection time (aborting the whole run) or skipped themselves silently, which
is the worse of the two — 168 tests reported as "not collected" rather than as
a failure.

The rig sets ``BNGSIM_TEST_DATA`` (``<root>/tests/data``, so the root is one of
its parents) and ``BNGSIM_SOURCE_ROOT`` (the root itself). Both are consulted
before the walk-up, and every candidate is confirmed by finding bngsim's own
``pyproject.toml`` there, so a wrong guess cannot be returned as a right one.

Four modules predating this helper — ``test_version_consistency.py``,
``test_codegen_cache_key.py``, ``test_curated_benchmark_corpus.py``,
``test_sbml_unsupported_manifest.py`` — still carry their own inline copies of
this same search, and 69 more still do the bare walk-up and so still misreport
under the copy. Folding them all in here is issue #590; new code should import
this rather than grow a fifth copy or a seventieth walk-up.

Import it the way the other sibling helpers in this directory are imported,
since pytest puts ``python/`` on ``sys.path`` rather than ``python/tests/``::

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _source_root import bngsim_source_root
"""

from __future__ import annotations

import os
from pathlib import Path


def bngsim_source_root() -> Path | None:
    """The bngsim source tree, or ``None`` when this checkout is not one.

    Candidates in order: the parents of ``$BNGSIM_TEST_DATA``, then
    ``$BNGSIM_SOURCE_ROOT``, then the parents of this file. The first that holds
    a ``pyproject.toml`` naming bngsim wins; a wheel-only install has none and
    gets ``None``, which callers turn into a skip.
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
