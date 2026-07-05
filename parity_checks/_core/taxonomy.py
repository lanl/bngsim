"""The single outcome taxonomy shared by every parity suite and consumer.

A parity job (one model × one method × one reference engine) resolves to
exactly one of these outcomes. Suites and downstream consumers (PyBNF,
PyBioNetGen, regenerating our golden references) all bucket into the same set
so reports are directly comparable across suites and across the engine/bridge
layers.
"""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    # bngsim and the reference engine agree within the job's oracle tolerance.
    PASS = "PASS"
    # Both ran and produced comparable output, but it exceeds the tolerance.
    DIFF = "DIFF"
    # bngsim raised / crashed / failed to load a model the reference engine ran
    # fine — the reference is the existence proof, so this is an actionable
    # bngsim bug. (When the *reference* fails see REFERENCE_FAILED / BAD_TEST.)
    EXCEPTION = "EXCEPTION"
    # The run exceeded its wall-clock budget and was cut off.
    TIMEOUT = "TIMEOUT"
    # The model exercises a feature the engine declares it does not support
    # (a clean, expected refusal — not a bug). e.g. validate_for_ssa rejects.
    UNSUPPORTED = "UNSUPPORTED"
    # The reference engine could not run the model but bngsim did, so there is
    # no trusted trajectory to compare against — no parity verdict is possible.
    # bngsim succeeding where the reference can't is a quiet capability win, not
    # a defect; derived from per-engine status, never a manual list.
    REFERENCE_FAILED = "REFERENCE_FAILED"
    # Neither engine could run the model — no parity signal and not a bngsim
    # bug. The test/model itself is the problem (malformed, or a feature both
    # engines lack). Also derived from per-engine status.
    BAD_TEST = "BAD_TEST"
    # Deliberately not run/compared (excluded by the manifest, no oracle, etc.).
    SKIP = "SKIP"

    def __str__(self) -> str:  # so f"{outcome}" prints "PASS" not "Outcome.PASS"
        return self.value


# Non-scoring outcomes: they carry no parity signal about bngsim, so they never
# count toward a suite's pass/fail tally. PASS is success; UNSUPPORTED/SKIP are
# expected non-runs; REFERENCE_FAILED/BAD_TEST are runs the *reference* (not
# bngsim) couldn't anchor, so they don't reflect on bngsim either. The scoring
# signal a report must surface is DIFF/EXCEPTION/TIMEOUT.
CLEAN = frozenset(
    {Outcome.PASS, Outcome.UNSUPPORTED, Outcome.SKIP, Outcome.REFERENCE_FAILED, Outcome.BAD_TEST}
)
FAILING = frozenset({Outcome.DIFF, Outcome.EXCEPTION, Outcome.TIMEOUT})

# Stable column order for report summaries.
ALL = (
    Outcome.PASS,
    Outcome.DIFF,
    Outcome.EXCEPTION,
    Outcome.TIMEOUT,
    Outcome.UNSUPPORTED,
    Outcome.REFERENCE_FAILED,
    Outcome.BAD_TEST,
    Outcome.SKIP,
)


def tally(outcomes) -> dict[str, int]:
    """Count an iterable of Outcome (or their string values) by name, in ALL order."""
    counts = {o.value: 0 for o in ALL}
    for o in outcomes:
        key = o.value if isinstance(o, Outcome) else str(o)
        counts[key] = counts.get(key, 0) + 1
    return counts
