"""Every case of the discontinuity suite (issue #1912), validated on import,
so a case written wrongly fails the collection of every test that reads it.

A family's cases live in a module of their own; add it to ``_FAMILIES``."""

from __future__ import annotations

from _cases import _flips, bend, clock, coincident, event, parked, pulse, sawtooth, step

_FAMILIES = (pulse, step, sawtooth, bend, event, clock, parked, coincident)

CASES = tuple(c for m in _FAMILIES for c in m.CASES)
BY_NAME = {c.name: c for c in CASES}

# #1914's and #1915's coin flips, measured over their cases' own runs (``_flips``).
for _c in CASES:
    _c.exclude = {**_c.exclude, **_flips.for_case(_c.name)}
_stray = set(_flips.ALL) - {k for c in CASES for k in c.exclude}
if _stray:
    raise ValueError(f"_flips names points no case sweeps: {sorted(_stray)[:3]}")

if len(BY_NAME) != len(CASES):
    raise ValueError("two cases share a name")
for _c in CASES:
    _c.validate()
