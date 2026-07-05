#!/usr/bin/env python3
"""Normalized .gdat/.cdat/.scan loaders for the bng_parity suite.

What this module USED to be: the legacy "correctness suite" verdict — a
subprocess sweep (``parity_sweep.py``) diffed against a bngsim sweep by this
file's inline cross-engine comparison (``deterministic_compare`` /
``stochastic_compare``), driven by ``parity_core_run.py`` / ``parity_run.py``,
with its own copy of the tolerance constants and a hand-maintained
KNOWN_ARTIFACT catalog.

That whole pipeline was **retired in GH #69**. The single source of truth for the
parity verdict is now ``_core.differ`` (``deterministic_verdict`` /
``ensemble_verdict``), consumed by the live matrix runners (``bng_ode_run.py`` /
``bng_stoch_run.py`` → ``generate_bng_matrix.py``); the matrix is also the home
for the adjudicated dispositions (``KNOWN_DETERMINISTIC_ARTIFACTS`` in
``bng_ode_run``; ``NF_KNOWN_REF`` / ``SSA_KNOWN_DISPOSITION`` in ``bng_stoch_run``),
which were folded out of this file's catalogs and restricted to the models that
still DIFF in the current full sweep.

All that remains here is the small, suite-specific **file-I/O** layer that
``parity_golden.py`` still imports to read and normalize a BNG output file
(``load_normalized``): strip the synthetic ``_rateLaw<digits>`` columns
run_network injects and canonicalize the ``()``-suffixed function-column headers,
so two BNG outputs compare positionally. This is BNGL/``.gdat``-specific and does
not belong in ``_core``.
"""

import re

import numpy as np


def load_array(path):
    """Load a .gdat/.cdat/.scan as float ndarray. Skip BNG comment header."""
    return np.loadtxt(str(path), comments="#", ndmin=2)


def safe_load(path):
    try:
        return load_array(path), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# bngsim settles .gdat/.scan on a single, method-independent schema that
# diverges from BNG2.pl in two cosmetic ways (per the bngsim devs, 2026-05):
#   (1) function-column headers are bare — bngsim writes ``kf_BSA`` where
#       BNG2.pl writes ``kf_BSA()`` on its NFsim output;
#   (2) the synthetic ``_rateLaw<digits>`` columns run_network injects into
#       the ODE .gdat are omitted by default (internal rate-law intermediates).
# Values, observables, species and .cdat are identical — the only diffs are
# header text and the presence of those columns. We normalize both sides
# before the positional compare so these don't read as failures. The
# normalization is deliberately narrow: a trailing ``()`` is stripped from
# headers and ``_rateLaw<digits>`` columns are dropped; ANY other column
# difference still falls through to the shape/value checks. See
# internal#58.
_RATELAW_RE = re.compile(r"^_rateLaw\d+$")


def _read_columns(path):
    """Column names from the BNG header (the last ``#`` line before data).

    BNG .gdat/.cdat/.scan carry a single comment header
    ``#   time   A   B   kf()``. Returns ``list[str]`` or ``None`` if no
    header is present / the file can't be read (caller falls back to a
    positional compare).
    """
    names = None
    try:
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    names = s.lstrip("#").split()
                else:
                    break
    except OSError:
        return None
    return names


def _canon(name):
    """Canonical function-column name: drop one trailing ``()``."""
    return name[:-2] if name.endswith("()") else name


def _normalize_columns(data, names):
    """Drop ``_rateLaw<digits>`` columns and canonicalize headers.

    Returns ``(data, names)``. When the header is unavailable or its token
    count doesn't match the data width, returns ``(data, None)`` — a
    positional fallback that changes nothing.
    """
    if names is None or len(names) != data.shape[1]:
        return data, None
    # Canonicalize (strip a trailing ``()``) BEFORE testing the rate-law
    # pattern: BNG2.pl's NFsim path writes every function column with the
    # ``()`` suffix, including synthetic intermediates (``_rateLaw1()``), so
    # matching the raw token against ``^_rateLaw\d+$`` would miss them.
    canon = [_canon(n) for n in names]
    keep = [i for i, n in enumerate(canon) if not _RATELAW_RE.match(n)]
    return data[:, keep], [canon[i] for i in keep]


def load_normalized(path):
    """``safe_load`` + drop ``_rateLaw`` columns + canonicalize headers.

    Returns ``(data, names, err)``; ``names`` is ``None`` when the header is
    unavailable (positional fallback).
    """
    data, err = safe_load(path)
    if err:
        return None, None, err
    data, names = _normalize_columns(data, _read_columns(path))
    return data, names, None
