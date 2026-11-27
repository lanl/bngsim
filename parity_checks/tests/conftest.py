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
for p in (
    _PC,
    _PC / "rr_parity",
    _PC / "amici_parity",
    _PC / "nightly",
    _PC / "codegen_paths",
    _PC / "discontinuity",
    _PC / "tests",
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def pytest_terminal_summary(terminalreporter: object) -> None:
    """Say how a checkout gets BNG2.pl when tests skipped for want of it (GH #1692).

    The skip reason is the resolver's ``why_not()``, which names the fix for a
    wheel install; this suite is the one ``--group parity`` exists for.
    """
    try:
        from bngsim._bngpath import _checkout_remedy
    except Exception:  # a broken build: its collection errors are the report
        return

    reasons = [
        str(getattr(report, "longrepr", ""))
        for report in getattr(terminalreporter, "stats", {}).get("skipped", [])
    ]
    lines = _checkout_remedy(reasons)
    if lines:
        terminalreporter.write_sep("─", "BNG2.pl")  # type: ignore[attr-defined]
        for line in lines:
            terminalreporter.write_line(line)  # type: ignore[attr-defined]
