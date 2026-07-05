"""Path setup for the parity_checks unit tests.

The suites are self-contained scripts that prepend their own directory to
``sys.path`` at runtime (so ``from _core import ...`` and ``import _rr_common``
resolve). The tests import those modules directly, so we make the same three
locations importable here — and, because ``_rr_common.schedule`` uses the
``spawn`` start method (which propagates the parent's ``sys.path`` to each
child), this also lets a spawned worker import ``smoke_workers``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PC = Path(__file__).resolve().parent.parent  # parity_checks/
for p in (_PC, _PC / "rr_parity", _PC / "tests"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
