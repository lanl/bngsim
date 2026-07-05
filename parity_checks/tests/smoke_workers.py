"""Module-level workers for the ``_rr_common.schedule`` smoke test.

The scheduler uses the ``spawn`` start method, so the worker target must be a
top-level (picklable-by-qualified-name) function in an importable module — a
closure or a test-local function would fail to pickle. These two trivial
workers stand in for the real bngsim/RoadRunner job runner.
"""

from __future__ import annotations

import time


def echo_worker(spec: dict, q) -> None:
    """Return a result that echoes the spec's key and a doubled value."""
    q.put({**spec, "status": "ok", "doubled": spec["value"] * 2})


def hang_worker(spec: dict, q) -> None:
    """Sleep past any sane cap so the parent must kill it (TIMEOUT path)."""
    time.sleep(30)
    q.put({**spec, "status": "ok"})  # never reached under a short cap


def branch_worker(spec: dict, q) -> None:
    """Dispatcher: hang for the 'slow' key, otherwise echo. Module-level so it
    pickles by qualified name under the spawn start method."""
    if spec["key"] == "slow":
        hang_worker(spec, q)
    else:
        echo_worker(spec, q)
