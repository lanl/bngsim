"""
_effort.py
----------
Shared effort-tier helper for the bngsim benchmark suites.

A benchmark *effort tier* -- ``low`` / ``medium`` / ``high`` -- buckets
each model (or fitting problem) by cost, so a suite can be run as a cheap
subset instead of the full sweep.  Runners expose this through a
``--effort`` flag with *cumulative* semantics: ``low`` runs only the low
tier, ``medium`` runs low + medium, ``high`` runs everything.  The
default is ``high``, so an unflagged run is unchanged.

Every benchmark runner that filters by effort should go through this
module, so the tier names, the cumulative direction and the ``high``
default stay identical across the whole ``bngsim/benchmarks/`` tree.

Each suite tags its models with a tier in whatever registry it already
uses -- a ``"effort"`` key in the ``suite_*.json`` model dicts, an
``effort`` field on the fitting ``Problem`` dataclass, etc.  The tiers
themselves are assigned from measured wall clock (the supplementary
benchmark tables, or a short calibration); see each suite's docs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

# Effort tiers in ascending cost order.  ``--effort X`` admits every tier
# at or below X, so this tuple's order *is* the cumulative semantics.
EFFORT_LEVELS = ("low", "medium", "high")

_T = TypeVar("_T")


def _check(name: str) -> None:
    if name not in EFFORT_LEVELS:
        raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {name!r}")


def add_effort_arg(parser, *, default: str = "high") -> None:
    """Register the standard ``--effort {low,medium,high}`` option.

    ``parser`` is an ``argparse.ArgumentParser`` (or subparser).  The
    default is ``high`` -- the full sweep -- so adding the flag never
    changes what an unflagged run does.
    """
    _check(default)
    parser.add_argument(
        "--effort",
        choices=list(EFFORT_LEVELS),
        default=default,
        help="Cumulative effort threshold: 'low' runs only low-effort "
        "models, 'medium' runs low + medium, 'high' runs all "
        "(default: %(default)s -- the full sweep).",
    )


def effort_allows(threshold: str, tier: str) -> bool:
    """True if an item at ``tier`` should run under ``--effort threshold``.

    Both arguments must be members of :data:`EFFORT_LEVELS`.  The relation
    is cumulative: a tier runs when it sits at or below the threshold.
    """
    _check(threshold)
    _check(tier)
    return EFFORT_LEVELS.index(tier) <= EFFORT_LEVELS.index(threshold)


def filter_by_effort(items: Iterable[_T], threshold: str, key: Callable[[_T], str]) -> list[_T]:
    """Return the ``items`` admitted at the cumulative ``threshold``.

    ``key`` maps an item to its effort-tier string, so this works for any
    registry shape -- ``suite_*.json`` dicts (``key=lambda m: m["effort"]``),
    the fitting ``Problem`` dataclass (``key=lambda p: p.effort``), etc.
    """
    _check(threshold)
    return [it for it in items if effort_allows(threshold, key(it))]
