"""Shared validation for explicit session output times (GH #184).

The stateful network-free sessions (:class:`bngsim.NfsimSession`,
:class:`bngsim.RuleMonkeySession`) accept an explicit ``sample_times`` list so a
host can record observables at a dataset's exact time points instead of a
uniform grid. Routing both backends through one validator keeps the
"emit at explicit times" contract byte-for-byte identical between them — NFsim
and RuleMonkey stay interchangeable, which is the whole point of the feature.

The contract mirrors :meth:`bngsim.Simulator.run`'s ``sample_times`` handling:
at least two points, all finite, returned sorted ascending.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def normalize_sample_times(sample_times: Iterable[float]) -> list[float]:
    """Validate and sort an explicit output-time list for the session API.

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
    sorted_times = sorted(float(t) for t in sample_times)
    if len(sorted_times) < 2:
        raise ValueError(f"sample_times must contain at least 2 points, got {len(sorted_times)}")
    if not all(math.isfinite(t) for t in sorted_times):
        raise ValueError("sample_times must all be finite")
    return sorted_times
