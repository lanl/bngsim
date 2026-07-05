"""bngsim's single stochastic initial-count rounding policy (GH #51).

When a continuous molecule amount has to become a discrete integer count for a
stochastic engine, bngsim rounds **half-up** — to the nearest integer, ties away
from zero: ``floor(x + 0.5)`` for non-negative counts. This matches BNG2.pl
``run_network`` (Network3) SSA *and* COPASI's stochastic solvers, which both use
the same rule; see GH #51 for the cross-engine survey.

This module is the single source of that policy for every continuous→discrete
boundary bngsim controls from Python:

* the :class:`bngsim.NfsimSession` / :class:`bngsim.RuleMonkeySession` count
  mutations (``add_molecules``, ``add_species``, ``remove_species``,
  ``set_species_count``) — warm continue, dosing, and parameter scans, and
* downstream hosts (e.g. the PyBNF NF bridge) that integerize a fractional
  ``setConcentration`` before handing a count to a network-free session.

Two more boundaries enforce the same rule in their own layers and must stay in
sync with this one:

* bngsim SSA/PSA setup, in C++ (``round_initial_population_to_storage`` in
  ``src/ssa_simulator.cpp``), and
* the cold-start seed concentrations bngsim rounds into the XML the network-free
  engines parse (``round_half_up_count`` in ``src/seed_count_rounding.hpp``).

The array form of the same rule is :func:`bngsim.coupling.round_to_counts`
(``policy="nearest"`` rounds half away from zero, identical to this for the
non-negative populations that occur in practice).
"""

from __future__ import annotations

import math

__all__ = ["round_half_up"]


def round_half_up(value: float) -> int:
    """Round a fractional molecule count to the nearest integer, ties away from zero.

    bngsim's repo-wide stochastic initial-count policy (GH #51):
    ``floor(x + 0.5)`` for ``x >= 0`` and ``ceil(x - 0.5)`` for ``x < 0``. The
    two branches differ only at negative half-integers, which never arise for
    molecule populations. The rule is idempotent on integers
    (``round_half_up(n) == n``), so warm-restart paths that re-round
    already-integer state are no-ops.

    Examples
    --------
    >>> [round_half_up(x) for x in (0.389, 0.10001, 5.7, 155.6747, 466.98)]
    [0, 0, 6, 156, 467]
    >>> [round_half_up(x) for x in (0.5, 1.5, 2.5)]  # ties away from zero, not bankers'
    [1, 2, 3]

    Parameters
    ----------
    value : float
        Continuous molecule count, already in molecule-number space. Any
        volume / concentration factors must be applied before calling.

    Returns
    -------
    int
        Nearest integer count.

    Raises
    ------
    ValueError
        If *value* is not finite.
    """
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"cannot round non-finite molecule count {value!r}")
    return int(math.floor(x + 0.5)) if x >= 0.0 else int(math.ceil(x - 0.5))
