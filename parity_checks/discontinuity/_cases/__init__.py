"""Every case of the discontinuity suite (issue #1912), validated on import,
so a case written wrongly fails the collection of every test that reads it.

A family's cases live in a module of their own; add it to ``_FAMILIES``."""

from __future__ import annotations

from _cases import bend, event, pulse, step

_FAMILIES = (pulse, step, bend, event)

CASES = tuple(c for m in _FAMILIES for c in m.CASES)
BY_NAME = {c.name: c for c in CASES}

if len(BY_NAME) != len(CASES):
    raise ValueError("two cases share a name")
for _c in CASES:
    _c.validate()
