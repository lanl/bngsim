"""Seed resolution helper for stochastic simulation.

When a caller passes ``seed=None`` to a stochastic method, draw a fresh
seed from system entropy so that consecutive calls produce independent
trajectories. When the caller passes an explicit integer, use it verbatim
for reproducibility.

The contract is described in `bngsim/README.md` under "Seed semantics for
stochastic methods" and is exercised by `python/tests/test_seed_semantics.py`.
"""

from __future__ import annotations

import secrets

# Default seed for random tie-breaking among simultaneous equal-priority events
# on the ODE path (GH #242). The ODE integrator is deterministic, so — unlike the
# stochastic methods, which draw a fresh seed from entropy when the caller passes
# ``None`` — the ODE path uses a FIXED default so a run is reproducible out of the
# box (the PyBNF-fitting requirement). Only consumed at a genuine equal-priority
# event tie (SBML L3v2 §4.11.6), so event-free / tie-free models are unaffected by
# it. MUST match ``SolverOptions::event_seed``'s C++ default (include/bngsim/
# types.hpp) so a no-arg ODE run behaves identically whether the seed is set from
# Python or left at the C++ default.
_DEFAULT_EVENT_SEED = 0x9E3779B97F4A7C15


def _resolve_seed(seed: int | None) -> int:
    """Return a usable integer seed.

    ``None`` → a fresh non-negative seed drawn from system entropy.
    An integer → returned as-is via ``int(...)``.

    The fresh draw is 31 bits so the result always fits in a signed
    32-bit C int (the underlying NFsim/RuleMonkey/SSA backends accept
    ``int32`` seeds).
    """
    if seed is None:
        return secrets.randbits(31)
    return int(seed)


# The width _resolve_seed draws to, and every derived seed keeps: the
# NFsim/RuleMonkey/SSA backends take a signed 32-bit seed.
_SEED_MASK = 0x7FFFFFFF


def derived_seeds(base_seed: int, n: int) -> list[int]:
    """The seeds of rows 0..n-1 of a batch, an ensemble or a scan run from
    ``base_seed`` (issue #1365).

    Row *i* used ``base_seed + i``, so an ensemble from seed 1 was the one from
    seed 0 shifted by a row: 49 of 50 trajectories bit-identical, and two
    ensembles that read as independent shared almost everything. These are drawn
    from numpy's ``SeedSequence(base_seed)``, whose stream is a hash of the base
    and the position, so nearby bases give unrelated seeds. Each row's seed is
    still on its ``Result.seed``, and a run from it repeats that row alone.

    Distinct within one call: a 31-bit draw repeats inside an ensemble of 10000
    about once in 40, and a repeated seed is a duplicated trajectory, which
    ``base_seed + i`` never gave. The first *n* distinct values of the stream
    are taken in order, so row *i*'s seed does not depend on *n*.
    """
    import numpy as np

    if n <= 0:
        return []
    ss = np.random.SeedSequence(int(base_seed) % (1 << 128))
    width = n
    while True:
        seen: set[int] = set()
        out: list[int] = []
        for word in ss.generate_state(width, dtype=np.uint32):
            seed = int(word) & _SEED_MASK
            if seed not in seen:
                seen.add(seed)
                out.append(seed)
                if len(out) == n:
                    return out
        width *= 2
