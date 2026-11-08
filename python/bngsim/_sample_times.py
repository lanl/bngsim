"""Shared validation for the output times a run is asked for.

Two forms of the same question reach every backend: an explicit ``sample_times``
list and a uniform ``t_span`` / ``n_points`` grid. Each is checked here, once,
so every entry point that takes one refuses the same requests in the same
words.

:func:`normalize_sample_times` began as the session validator (GH #184): the
stateful network-free sessions (:class:`bngsim.NfsimSession`,
:class:`bngsim.RuleMonkeySession`) accept an explicit ``sample_times`` list so a
host can record observables at a dataset's exact time points instead of a
uniform grid, and routing both through one validator keeps NFsim and RuleMonkey
interchangeable. :meth:`bngsim.Simulator.run` uses it too (issue #1504): it
used to sort and count the list itself and never asked whether the times were
finite, so ``sample_times=[0, 5, inf]`` sent CVODE toward a target it could
never reach, and the run spun inside one CVode call that ``timeout=`` cannot
interrupt.

:func:`validate_time_span` is the same check for a uniform grid, used by
``run``, ``run_batch``, ``run_replicates``, ``parameter_scan`` and
``compute_all_sensitivities`` (issue #1523: the last checked neither the span
nor the count, and ``n_points=0`` there killed the interpreter).
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def normalize_sample_times(sample_times: Iterable[float]) -> list[float]:
    """Validate and sort an explicit output-time list.

    Parameters
    ----------
    sample_times : iterable of float
        The requested output instants (typically a dataset's independent
        variable). Treated as absolute times; the first element is the
        segment start.

    Returns
    -------
    list[float]
        The same instants as ``float``, sorted ascending.

    Raises
    ------
    ValueError
        If fewer than two points are supplied, or any point is not finite.
    """
    times = [float(t) for t in sample_times]
    if len(times) < 2:
        raise ValueError(f"sample_times must contain at least 2 points, got {len(times)}")
    # Checked on the list as given, ahead of the sort: sorted() does not order
    # a list that holds a NaN, so the sorted list's ends say nothing about it.
    bad = [t for t in times if not math.isfinite(t)]
    if bad:
        raise ValueError(f"sample_times must all be finite, got {bad[0]}")
    return sorted(times)


def validate_time_span(t_span: tuple[float, float], n_points: int) -> tuple[float, float, int]:
    """Validate a uniform output grid: ``n_points`` times from ``t_start`` to ``t_end``.

    Parameters
    ----------
    t_span : tuple[float, float]
        ``(t_start, t_end)``.
    n_points : int
        Number of output points, both ends included.

    Returns
    -------
    tuple[float, float, int]
        ``(t_start, t_end, n_points)``, the times as ``float``.

    Raises
    ------
    ValueError
        If either end is not finite, if ``t_end <= t_start``, or if
        ``n_points < 2``.
    """
    t_start, t_end = (float(t) for t in t_span)
    # Ahead of the order test: a NaN fails every comparison, so
    # `t_end <= t_start` is false for it and for inf alike, and both used to
    # pass through to the solver (issue #1504).
    if not (math.isfinite(t_start) and math.isfinite(t_end)):
        raise ValueError(f"t_span must be finite, got ({t_start}, {t_end})")
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
    if n_points < 2:
        raise ValueError(f"n_points ({n_points}) must be >= 2")
    return t_start, t_end, n_points


def require_finite_time(name: str, t: float) -> float:
    """``t`` as a float, or ``ValueError`` naming ``name`` unless it is finite."""
    value = float(t)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value
