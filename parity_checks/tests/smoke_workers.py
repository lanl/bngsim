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


def screen_worker(
    spec: dict, n, seed_base, mean_z_tol, max_failures, min_mean_count, self_z_tol, q
):
    """Stands in for ``ssa_screen._worker`` (same signature), by ``spec["name"]``:

    - ``fast``: a pass at once;
    - ``slow_rr``: bngsim for 0.2 s, then RoadRunner until 2.5 s, which leaves a
      marker in ``spec["rr_done"]`` (and, on a recording, its archive) before the
      final pass;
    - ``prelim``: a red row at once (as ``_worker`` puts before judging a refused
      propensity), then, at 3 s, a different final row;
    - ``late``: a red row at 2.5 s, past the cap, then never finishes;
    - ``hang``: never finishes.
    """
    from pathlib import Path

    t0 = time.monotonic()
    name = spec["name"]
    if name == "fast":
        q.put({"name": name, "status": "pass"})
    elif name == "slow_rr":
        time.sleep(2.5 - (time.monotonic() - t0))
        Path(spec["rr_done"], name).write_text("rr")
        if spec.get("write_reference_archive"):
            Path(spec["write_reference_archive"], f"{name}.saved").write_text("rr")
        q.put({"name": name, "status": "pass"})
    elif name == "prelim":
        q.put({"name": name, "status": "error", "error": "the red row first"})
        time.sleep(3.0 - (time.monotonic() - t0))
        q.put({"name": name, "status": "nonfinite_rate"})
    elif name == "late":
        time.sleep(2.5 - (time.monotonic() - t0))
        q.put({"name": name, "status": "error", "error": "a red row past the cap"})
        time.sleep(30)
    else:
        time.sleep(30)
