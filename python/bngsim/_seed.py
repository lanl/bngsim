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
