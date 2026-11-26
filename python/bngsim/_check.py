"""A one-call check of a run (issue #1707).

:meth:`bngsim.Result.check` makes the run a result records again with one
setting changed (:meth:`Result.rerun`, issue #1703) and says whether the answer
moved. Each rerun starts from the state the run started from, on a fresh model,
so a check costs the session nothing and compares like with like.

Every comparison is made column by column (species), each on its own scale, so
a species at 1e-3 beside one at 1e3 is judged as carefully as the large one,
and each column's absolute floor is the atol the run applied to that species:

- the *tolerance* check makes the run at a tenth of its ``rtol`` and atol. A
  column's *move* is its largest difference from the run, past its atol, over
  its peak; the check fails where a column moves by more than
  :data:`TOLERANCE_MOVE` (0.1% of its peak), which says the run is not
  converged at its own tolerance. That move is also the run's *drift*;
- every other check (a *path*) is judged column by column against the
  allowance ``max(min(PATH_RTOL_FACTOR·(atol_j + rtol·peak_j),
  max(PATH_CEILING·peak_j, atol_j)), PATH_ATOL_FACTOR·atol_j,
  DRIFT_FACTOR·d_j)``, where ``d_j`` is how far the column itself moves at the
  tighter tolerance, or half how far at twice the tolerance, whichever is
  larger (:func:`spread`): two correct paths each err by about the tolerance, scaled
  by the problem's conditioning, which the drift measures, and never by more
  than 1% of the column's peak for the tolerance alone, unless the column is
  within ``PATH_ATOL_FACTOR`` atol of zero, which resolves it no better;
- a column whose peak is within :data:`TRACE_ATOLS` times its atol, where the
  atol and not rtol holds it (``atol_j > rtol·peak_j``), is not resolved by
  the run, and the report names it. It is judged on the run's whole scale
  alone: against the allowance of a column at the largest peak and atol, with
  the largest drift, so its solver noise is no finding, and a path on which
  it departs from zero as far as the run's own species are large is one
  (issue #1887);
- a column that is not converged at the run's own tolerance (its move past
  :data:`TOLERANCE_MOVE`) is judged on the run's whole scale, as one not
  resolved is: a path on which it blows up still fails. Where no column is
  converged, the verdict is ``no_base``, and a report on a run that is not
  converged is never ok.

The constants are the developer-side differential harness's
(``parity_checks/differential``, issue #1257). Its comparison is over the whole
trajectory at once, against the largest value of any species, which a column
far smaller than the largest cannot fail; this one is per column.

A run that computed forward sensitivities gets checks of them too
(:data:`SENSITIVITY`, in :mod:`bngsim._check_sensitivity`, issue #1893), each
judged series by series on the same rules: the tolerance check's run's
sensitivities, and a difference quotient.

Verdicts: ``pass``, ``diff``, ``skip`` (with the reason), ``unsupported`` (the
check does not apply to this run), ``exception`` (the changed run raised),
``no_base`` (no column is converged enough to judge a path against) and
``timeout`` (the ``budget`` ran out).
"""

from __future__ import annotations

import math
import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim import _record
from bngsim._exceptions import RoundoffStepWarning, UsageError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._result import Result

#: Incremented by a change that removes or renames a key, or changes what one
#: holds. A key added is no change of version.
CHECK_VERSION = 1

#: How many times the run's tolerance two correct paths may differ by: their
#: errors are each about the tolerance, scaled by the problem's conditioning,
#: which a tolerance does not bound.
PATH_RTOL_FACTOR = 1e3
#: And how many times past a column's own move at a tenfold tighter tolerance.
DRIFT_FACTOR = 10.0
#: A path's allowance reads a column's drift as the larger of its move at the
#: tenfold tighter tolerance and its move at this many times the tolerance,
#: over this factor: how far a correct run strays at the run's tolerance, which
#: one tighter run measures only from the run's own error, small by luck as
#: often as not. MODEL1912090001's x15_PI34P2 at rtol 1e-6 erred by 9.5e-5 of
#: its peak, and by 1.0e-3 at half the tolerance and 9.6e-4 at twice it; a
#: compiled run at 1.6e-3 was another correct run, not a finding (issue #1887).
SPREAD_FACTOR = 2.0
#: A column that moves by more than this share of its peak at a tenfold tighter
#: tolerance is not converged at the run's own (the harness's
#: ``UNJUDGEABLE_DRIFT``): the tolerance check fails, and no path is judged on
#: that column.
TOLERANCE_MOVE = 1e-3
#: The most the tolerance alone may forgive a path, as a share of a column's
#: peak. At an rtol of 1e-8 the first term is 1e-5 and this never bites; at a
#: caller's rtol of 1e-3 the first term alone is the whole peak.
PATH_CEILING = 1e-2
#: But a path may always differ by this many times a column's atol: a column
#: whose peak is within a few hundred atol of zero is resolved to no better.
PATH_ATOL_FACTOR = 100.0
#: A column whose peak is within this many times the atol the run applied to
#: it, where that atol (not rtol) holds it, is not resolved by the run: its
#: values are within the solver's absolute error of zero. No check judges it,
#: and the report names it.
TRACE_ATOLS = 100.0
#: How many such columns a report names.
MAX_NAMED = 20
#: The grid check's output points, as a multiple of the run's intervals.
GRID_FACTOR = 8
#: How many reruns the replicates check makes of a stochastic run.
DEFAULT_REPLICATES = 10

#: The checks, in the order they run; ``tolerance`` first, as the others are
#: judged against its drift.
DETERMINISTIC = ("tolerance", "grid", "jacobian", "linear_solver", "backend", "second_engine")
STOCHASTIC = ("replicates",)
#: The checks of a run's forward sensitivities (issue #1893): what ``"all"``
#: adds for a run that computed them. ``sensitivity`` judges the tolerance
#: check's run; ``sensitivity_fd`` makes a difference quotient;
#: ``sensitivity_second_engine`` makes the run in AMICI, where it is installed
#: (issue #1923, :mod:`bngsim._check_amici`).
SENSITIVITY = ("sensitivity", "sensitivity_fd", "sensitivity_second_engine")
#: What ``checks=None`` runs: the cheap checks that catch the common case, and
#: of a run that computed sensitivities, :data:`DEFAULT_SENSITIVITY` too.
DEFAULT_DETERMINISTIC = ("tolerance", "grid")
DEFAULT_STOCHASTIC = ("replicates",)
DEFAULT_SENSITIVITY = ("sensitivity",)

_STATUSES = ("pass", "diff", "skip", "unsupported", "exception", "no_base", "timeout")
_NOT_OK = ("diff", "exception", "timeout")


def column_atols(
    result: Result, names: list[str], per_species: Any, scalar: float | None
) -> NDArray[np.float64]:
    """Each of *names*' atol as the run applied it. The per-species vector runs
    over the run's state (:attr:`Result.state_names`), which holds more than
    the species where an event or a rule promotes a parameter into it
    (BIOMD0000000001: 12 species, 20 entries), so it is matched by name. A
    Result with no state names (loaded, stacked) has its species first and the
    promoted entries after them, so the vector's first ``len(names)``. A
    column the vector does not cover takes the scalar, the run's ceiling."""
    n = len(names)
    fill = float(scalar) if scalar is not None else 0.0
    if per_species is None:
        return np.full(n, fill)
    per = [float(a) for a in per_species]
    try:
        state = list(result.state_names)
    except Exception:  # noqa: BLE001 - a loaded or stacked Result holds none
        state = []
    if len(state) == len(per):
        index = {name: i for i, name in enumerate(state)}
        return np.asarray([per[index[c]] if c in index else fill for c in names], dtype=float)
    if len(per) >= n:
        return np.asarray(per[:n], dtype=float)
    return np.full(n, fill)


def differences(
    x: NDArray[np.float64], ref: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per column: the largest ``|x - ref|`` over time (``inf`` where the two
    are not finite at the same cells), and the reference's peak ``|ref|``.
    Arrays of another shape are ``inf`` everywhere."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    n = ref.shape[-1] if ref.ndim == 2 else 0
    fin = np.isfinite(ref)
    peak = np.where(fin, np.abs(ref), 0.0).max(axis=0) if ref.size else np.zeros(n)
    if x.shape != ref.shape:
        return np.full(n, np.inf), peak
    same = (np.isfinite(x) == fin).all(axis=0)
    d = (
        np.where(fin & np.isfinite(x), np.abs(x - ref), 0.0).max(axis=0)
        if ref.size
        else np.zeros(n)
    )
    return np.where(same, d, np.inf), peak


def moves(
    d: NDArray[np.float64], peak: NDArray[np.float64], atol: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Each column's difference past its atol, over its peak; 0 for a column
    whose peak is within its atol (nothing in it can count)."""
    over = np.maximum(d - atol, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(peak > atol, over / np.where(peak > 0, peak, 1.0), 0.0)
    return np.where(np.isinf(d), np.inf, rel)


def allowance(
    peak: NDArray[np.float64], atol: NDArray[np.float64], rtol: float, drift: NDArray[np.float64]
) -> NDArray[np.float64]:
    """How far two correct paths may be apart in each column: see the module
    docstring."""
    by_tol = np.minimum(
        PATH_RTOL_FACTOR * (atol + rtol * peak), np.maximum(PATH_CEILING * peak, atol)
    )
    return np.maximum(np.maximum(by_tol, PATH_ATOL_FACTOR * atol), DRIFT_FACTOR * drift)


def resolved(
    peak: NDArray[np.float64], atol: NDArray[np.float64], rtol: float
) -> NDArray[np.bool_]:
    """The columns the run resolves: not within :data:`TRACE_ATOLS` of their
    atol of zero where that atol, not rtol, holds them (a column at 0 with an
    atol of 0 resolves nothing either)."""
    return ~((peak <= TRACE_ATOLS * atol) & ((atol > rtol * peak) | (peak == 0)))


def spread(
    tight: NDArray[np.float64],
    loose: NDArray[np.float64] | None,
    peak: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Each column's drift for a path's allowance: its move at the tighter
    tolerance, or its move at :data:`SPREAD_FACTOR` times the tolerance over
    that factor, whichever is larger (*loose* None where that run failed).
    The looser run widens a column only where its move is finite, and by at
    most :data:`TOLERANCE_MOVE` of the column's *peak*: a looser run that
    blows up says nothing of a correct run's scatter, and one not converged
    itself would forgive any path."""
    if loose is None:
        return tight
    half = np.asarray(loose, dtype=float) / SPREAD_FACTOR
    ok = np.isfinite(half)
    half = np.where(ok, half, 0.0)
    if peak is not None:
        half = np.minimum(half, TOLERANCE_MOVE * np.asarray(peak, dtype=float))
    return np.maximum(tight, half)


def path_margins(
    d: NDArray[np.float64],
    peak: NDArray[np.float64],
    atol: NDArray[np.float64],
    rtol: float,
    drift: NDArray[np.float64],
    judged: NDArray[np.bool_],
    own: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per column: its difference *d* over its allowance, and over its scale,
    each 0 where *judged* is False. A column in *own* (default: those the run
    resolves, :func:`resolved`) is held to its own allowance and peak. Any
    other, one the run does not resolve or one not converged at its
    tolerance, is held to the run's whole scale as well: the larger of its own
    allowance and that of a column at the largest peak and atol with the
    largest drift of an *own* column, read on the largest peak. Within its
    atols, or its own scatter, it is noise; past the run's scale it is a
    finding, and so is a value non-finite where the reference's is not.
    Shared by :meth:`bngsim.Result.check` and the developer-side differential
    harness, so a verdict means one thing in both (issue #1887)."""
    mine = resolved(peak, atol, rtol) if own is None else np.asarray(own, dtype=bool)
    allow = allowance(peak, atol, rtol, drift)
    scale = peak
    other = ~mine
    if other.any() and peak.size:
        keep = np.isfinite(drift) & mine
        dmax = float(drift[keep].max()) if keep.any() else 0.0
        largest = float(peak.max())
        whole = allowance(
            np.array([largest]), np.array([float(atol.max())]), rtol, np.array([dmax])
        )[0]
        allow = np.where(other, np.maximum(allow, whole), allow)
        scale = np.where(other, largest, peak)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            allow > 0, d / np.where(allow > 0, allow, 1.0), np.where(d > 0, np.inf, 0.0)
        )
        rel = np.where(scale > atol, d / np.where(scale > 0, scale, 1.0), 0.0)
    ratio = np.where(judged, np.nan_to_num(ratio, nan=np.inf), 0.0)
    return ratio, np.where(judged, rel, 0.0)


def _num(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


class Report(dict):  # type: ignore[type-arg]
    """:meth:`Result.check`'s dict, whose ``str()`` is its plain-text form.

    It is a ``dict`` in every other way: ``json.dumps(report, allow_nan=False)``
    writes it.
    """

    __slots__ = ()

    def __str__(self) -> str:
        return text(self)

    def text(self) -> str:
        """The plain-text form, as ``str()`` gives it."""
        return text(self)


class _Out(Exception):
    """The budget ran out before a check could start or finish."""


def check(
    result: Result,
    *,
    checks: Any = None,
    budget: float | None = None,
    replicates: int = DEFAULT_REPLICATES,
) -> Report:
    """:meth:`Result.check`."""
    if result._species.ndim == 3:
        raise UsageError(
            "Result.check() checks one run; this is a stack of runs. Check each row "
            "(the list run_batch or run_replicates returns without squeeze=True).",
            code="usage.invalid_argument",
        )
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int | float) or not budget > 0
    ):
        raise UsageError(f"check(budget=...) takes seconds > 0 or None, got {budget!r}")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise UsageError(
            f"check(replicates=...) takes a whole number of 2 or more, got {replicates!r}"
        )
    rec = result.record
    run = rec.get("run")
    method = None if run is None else run.get("method")
    stochastic = None if method is None else method in _record.STOCHASTIC
    known = DETERMINISTIC + SENSITIVITY + STOCHASTIC
    from bngsim import _check_sensitivity

    sens = not stochastic and _check_sensitivity.present(result)
    if checks is None:
        names = list(DEFAULT_STOCHASTIC if stochastic else DEFAULT_DETERMINISTIC)
        names += list(DEFAULT_SENSITIVITY) if sens else []
    elif checks == "all":
        names = list(STOCHASTIC if stochastic else DETERMINISTIC)
        names += list(SENSITIVITY) if sens else []
    else:
        names = [checks] if isinstance(checks, str) else list(checks)
        bad = [n for n in names if n not in known]
        if bad:
            raise UsageError(
                f"check(checks=...) does not know {bad[0]!r}: it takes None (the default "
                f"set), 'all', or names from {', '.join(known)}.",
                code="usage.invalid_argument",
                details={"checks": bad, "accepted": list(known)},
            )
        names = list(dict.fromkeys(names))
    deadline = None if budget is None else time.monotonic() + float(budget)

    out = Report()
    out["check_version"] = CHECK_VERSION
    out["ok"] = False
    out["method"] = method
    out["stochastic"] = stochastic
    out["rtol"] = None
    out["atol"] = None
    out["drift"] = None
    out["bound"] = None
    out["converged"] = None
    out["n_not_resolved"] = None
    out["not_resolved"] = []
    out["checks"] = []

    roundoff = any(n.code == "solver.roundoff_steps" for n in result._notices)
    if run is None:
        for name in names:
            out["checks"].append(
                _verdict(name, "unsupported", note="the result records no run to make again")
            )
        return _finish(out, roundoff)

    ctx = _Context(result, rec, deadline)
    out["rtol"] = ctx.rtol
    out["atol"] = ctx.atol
    order = [n for n in known if n in names]
    for name in order:
        v = ctx.run_check(name, replicates)
        out["checks"].append(v)
    if ctx.moved is not None and ctx.moved.size:
        out["drift"] = _num(ctx.moved.max())
        out["converged"] = bool(ctx.moved.max() <= TOLERANCE_MOVE)
        trace = [n for n, keep in zip(ctx.names, ctx.relevant, strict=True) if not keep]
        out["n_not_resolved"] = len(trace)
        out["not_resolved"] = trace[:MAX_NAMED]
    out["bound"] = TOLERANCE_MOVE
    return _finish(out, roundoff)


def _finish(out: Report, roundoff: bool = False) -> Report:
    statuses = [c["status"] for c in out["checks"]]
    out["ok"] = (
        bool(statuses)
        and "pass" in statuses
        and not any(s in _NOT_OK for s in statuses)
        and out.get("converged") is not False
    )
    out["n_pass"] = statuses.count("pass")
    out["n_diff"] = statuses.count("diff")
    out["next"] = _next(out, roundoff)
    return out


#: The next step for a run that took steps too small to move its clock
#: (issue #1837) and moves with the tolerance: a tighter one makes it worse.
ROUNDOFF_NEXT = (
    "The answer moves with the tolerance, and the run took steps too small to move "
    "its clock (its notice solver.roundoff_steps), which a tighter tolerance makes "
    "worse: read the rows from the time the notice names on as suspect, and run "
    "nearer t = 0 where the model allows it."
)


def _next(out: Report, roundoff: bool = False) -> str:
    """What to do next, in one sentence. *roundoff*: the run took steps
    where t + h == t."""
    diffs = [c for c in out["checks"] if c["status"] == "diff"]
    if diffs and diffs[0]["name"] == "tolerance":
        if roundoff:
            return ROUNDOFF_NEXT
        return (
            "The answer moves when the tolerance is tightened: rerun at a tighter rtol and "
            "atol (result.rerun(rtol=..., atol=...)) until it does not, and report that run."
        )
    if diffs and diffs[0]["name"] == "sensitivity":
        return (
            "The sensitivities move when the tolerance is tightened: rerun at a tighter rtol "
            "and atol (result.rerun(rtol=..., atol=...)) until they do not, and report that "
            "run's sensitivities."
        )
    if diffs and diffs[0]["name"] == "sensitivity_fd" and diffs[0].get("diff_not_converged"):
        return (
            "The sensitivities that disagree with a difference quotient are not converged at "
            "the run's tolerance: rerun at a tighter rtol and atol (result.rerun(rtol=..., "
            "atol=...)) and check that run's sensitivities."
        )
    if diffs and diffs[0]["name"] == "sensitivity_fd":
        return (
            "The forward sensitivities disagree with a difference quotient of the trajectories "
            "(sensitivity_fd): on a model bngsim accepts, that is a bug worth reporting at "
            "https://github.com/lanl/bngsim/issues, with json.dumps(report), result.record and "
            "the model file. Do not report those sensitivities."
        )
    if diffs and diffs[0]["name"] == "sensitivity_second_engine":
        if diffs[0].get("diff_not_converged"):
            return (
                "The sensitivities that disagree with AMICI's are not converged at the run's "
                "tolerance: rerun at a tighter rtol and atol (result.rerun(rtol=..., atol=...)) "
                "and check that run's sensitivities."
            )
        what = "trajectory" if diffs[0].get("state_diff") else "forward sensitivities"
        return (
            f"bngsim and AMICI disagree on this model's {what}: one of them reads the model "
            "differently, or computes its sensitivities wrong. Report it at "
            "https://github.com/lanl/bngsim/issues with json.dumps(report) and the model file, "
            "and do not report those sensitivities."
        )
    if diffs and diffs[0]["name"] == "second_engine":
        return (
            "bngsim and libRoadRunner disagree on this model: one of them reads it "
            "differently. Report it at https://github.com/lanl/bngsim/issues with "
            "json.dumps(report) and the model file."
        )
    if diffs:
        return (
            f"Two bngsim paths disagree ({diffs[0]['name']}): on a model bngsim accepts, that "
            "is a bug worth reporting at https://github.com/lanl/bngsim/issues, with "
            "json.dumps(report), result.record and the model file. Do not report the "
            "number as resolved."
        )
    timed = [c["name"] for c in out["checks"] if c["status"] == "timeout"]
    if timed:
        return (
            f"The budget ran out before {', '.join(timed)}: the result is not checked until "
            "they run; pass a larger budget=."
        )
    if any(c["status"] == "exception" for c in out["checks"]):
        return "A changed run raised: read its note before relying on the result."
    if any(c["status"] == "no_base" for c in out["checks"]) or out.get("converged") is False:
        if roundoff:
            return ROUNDOFF_NEXT
        return (
            "The run is not converged at its own tolerance, so no path could be judged: "
            "rerun at a tighter rtol and atol and check that."
        )
    if any(c["name"] == "replicates" and c["status"] == "pass" for c in out["checks"]):
        return (
            "A stochastic run is one draw: report the mean over replicates with its "
            "standard error, not this trajectory."
        )
    if not out["ok"]:
        return "No check could run on this result: see each check's note."
    trace = out.get("not_resolved") or []
    if trace:
        more = out["n_not_resolved"] - len(trace)
        names = ", ".join(trace) + (f" and {more} more" if more > 0 else "")
        return (
            "No check moved the answer past its bound, but "
            f"{names} {'stays' if out['n_not_resolved'] == 1 else 'stay'} within "
            f"{TRACE_ATOLS:g} times {'its' if out['n_not_resolved'] == 1 else 'their'} atol "
            "of zero, which the run does not resolve, and "
            f"{'was' if out['n_not_resolved'] == 1 else 'were'} judged only against the "
            "run's largest values: to report "
            f"{'it' if out['n_not_resolved'] == 1 else 'them'}, rerun with an atol well "
            f"below {'it' if out['n_not_resolved'] == 1 else 'them'} "
            "(result.rerun(atol=...)) and check that."
        )
    return "No check moved the answer past its bound."


def _verdict(
    name: str,
    status: str,
    value: float | None = None,
    note: str = "",
    *,
    worst: str | None = None,
    changed: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert status in _STATUSES, status
    v: dict[str, Any] = {
        "name": name,
        "status": status,
        "value": _num(value),
        "worst_column": worst,
        "changed": changed or {},
        "note": note,
    }
    if extra:
        v.update(extra)
    return v


class _Context:
    """The run a result records, and what its checks share: the tolerances, the
    columns' atols and the drift."""

    def __init__(self, result: Result, rec: dict[str, Any], deadline: float | None) -> None:
        self.result = result
        self.rec = rec
        self.run = rec["run"]
        self.deadline = deadline
        self.method = self.run.get("method")
        self.stochastic = self.method in _record.STOCHASTIC
        self.steady = bool(self.run.get("steady_state"))
        self.names = list(result._species_names)
        self.x = self._view(result)
        atol = self.run.get("atol") or {}
        self.rtol: float | None = self.run.get("rtol")
        per = atol.get("per_species")
        self.per_species = None if per is None else [float(a) for a in per]
        self.atol: float | None = atol.get("scalar")
        n = len(self.names)
        self.atols = column_atols(result, self.names, self.per_species, self.atol)
        self.drift_abs: NDArray[np.float64] | None = None
        self.spread_abs: NDArray[np.float64] | None = None
        self.moved: NDArray[np.float64] | None = None
        self.relevant: NDArray[np.bool_] = np.ones(n, dtype=bool)
        self.tol_error = ""
        # The tighter and looser runs, kept for the sensitivity checks, which
        # judge their sensitivities (issue #1893).
        self.tight: Result | None = None
        self.tight100: Result | None = None
        self.loose: Result | None = None
        # The sensitivity checks' columns and their drift, made once.
        self.sens_columns: Any = None
        self.sens_drift: Any = None

    def _view(self, result: Result) -> NDArray[np.float64]:
        """What is compared of *result*: its species, or of a run that stopped
        at a steady state, the state it stopped at."""
        x = np.asarray(result._species, dtype=float)
        return x[-1:] if self.steady else x

    # --- running ---------------------------------------------------------

    def _left(self) -> float | None:
        if self.deadline is None:
            return None
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise _Out
        return left

    def rerun(self, **changes: Any) -> Result:
        left = self._left()
        if left is not None:
            own = self.run.get("timeout")
            changes["timeout"] = left if own is None else min(left, float(own))
        try:
            with warnings.catch_warnings():
                # A rerun's steps where t + h == t are the check's to judge, not
                # the caller's to be warned of: on a run that took them, every
                # rerun takes more (issue #1837). They stay on its notices.
                warnings.simplefilter("ignore", RoundoffStepWarning)
                return self.result.rerun(**changes)
        except Exception as e:
            if self.deadline is not None and _timed_out(e):
                raise _Out from e
            raise

    def perturbed(self, **changes: Any) -> Result:
        """:func:`bngsim._rerun.perturbed`, within the budget as :meth:`rerun`."""
        from bngsim import _rerun

        left = self._left()
        if left is not None:
            own = self.run.get("timeout")
            changes["timeout"] = left if own is None else min(left, float(own))
        try:
            return _rerun.perturbed(self.result, **changes)
        except Exception as e:
            if self.deadline is not None and _timed_out(e):
                raise _Out from e
            raise

    def run_check(self, name: str, replicates: int) -> dict[str, Any]:
        fn = getattr(self, f"check_{name}")
        try:
            self._left()
            return fn() if name != "replicates" else fn(replicates)
        except _Out:
            return _verdict(name, "timeout", note="the budget ran out before this check finished")
        except Exception as e:  # noqa: BLE001 - a check that raises is a finding
            if getattr(e, "code", None) in ("rerun.unsupported", "rerun.not_recorded"):
                # The run cannot be made again from its record (a network-free
                # run, a stop condition, ...): the check does not apply.
                return _verdict(name, "unsupported", note=f"no rerun: {_first_line(e)}")
            return _verdict(name, "exception", note=f"{type(e).__name__}: {_first_line(e)}")

    # --- the reference ---------------------------------------------------

    def _deterministic(self, name: str) -> dict[str, Any] | None:
        if self.stochastic:
            return _verdict(
                name, "unsupported", note=f"a {self.method} run is stochastic: see replicates"
            )
        if self.method != "ode":
            return _verdict(name, "unsupported", note=f"the run's method is {self.method!r}")
        if self.rtol is None or self.atol is None:
            return _verdict(name, "unsupported", note="the record gives no rtol or atol")
        return None

    def _tighter(self, by: float) -> dict[str, Any]:
        assert self.rtol is not None and self.atol is not None
        if self.per_species is not None:
            atol: Any = [a / by for a in self.per_species]
        else:
            atol = float(self.atol) / by
        return {"rtol": float(self.rtol) / by, "atol": atol}

    def _ensure_drift(self) -> str | None:
        """Each column's move at a tenfold tighter tolerance, made once; the
        reason there is none, or None."""
        if self.drift_abs is None and not self.tol_error:
            try:
                tight = self.rerun(**self._tighter(10.0))
            except _Out:
                raise
            except Exception as e:  # noqa: BLE001 - recorded on every check's note
                self.tol_error = f"{type(e).__name__}: {_first_line(e)}"
                return self.tol_error
            self.tight = tight
            d, peak = differences(self._view(tight), self.x)
            self.drift_abs = d
            # Not resolved: within TRACE_ATOLS atol of zero, where the atol,
            # not rtol, holds the column (a loose rtol makes every atol a
            # large share of its species, and those columns are resolved to
            # that share, which the tolerance check judges).
            self.relevant = resolved(peak, self.atols, float(self.rtol or 0.0))
            self.moved = np.where(self.relevant, moves(d, peak, self.atols), 0.0)
        return self.tol_error or None

    def _ensure_spread(self) -> NDArray[np.float64]:
        """The drift a path's allowance reads (:func:`spread`), made once; the
        tighter run's alone where the looser one raises."""
        assert self.drift_abs is not None
        if self.spread_abs is None:
            try:
                loose = self.rerun(**self._tighter(1.0 / SPREAD_FACTOR))
            except _Out:
                raise
            except Exception:  # noqa: BLE001 - the tighter run's drift stands
                self.spread_abs = self.drift_abs
            else:
                self.loose = loose
                d, peak = differences(self._view(loose), self.x)
                self.spread_abs = spread(self.drift_abs, d, peak)
        return self.spread_abs

    def _worst(self, score: NDArray[np.float64]) -> str | None:
        if not score.size or not (score > 0).any():
            return None
        j = int(np.argmax(np.where(np.isnan(score), np.inf, score)))
        return self.names[j] if j < len(self.names) else None

    def compare(self, name: str, again: Result, changed: dict[str, Any]) -> dict[str, Any]:
        return self.compare_x(name, self._view(again), changed)

    def compare_x(
        self, name: str, x: NDArray[np.float64], changed: dict[str, Any]
    ) -> dict[str, Any]:
        err = self._ensure_drift()
        if err:
            return _verdict(
                name,
                "no_base",
                note=f"the run at a tenfold tighter tolerance raised ({err})",
                changed=changed,
            )
        assert self.drift_abs is not None and self.moved is not None and self.rtol is not None
        converged = self.relevant & (self.moved <= TOLERANCE_MOVE)
        if not converged.any():
            return _verdict(
                name,
                "no_base",
                note=f"no column is converged at the run's tolerance (each moves past "
                f"{TOLERANCE_MOVE:g} of its peak at a tenfold tighter one): no path is judged",
                changed=changed,
            )
        # Every column: a converged one on its own allowance, any other (not
        # resolved, or not converged at the run's tolerance) on the run's
        # whole scale too, so that a path on which it blows up still fails.
        judged = np.ones(converged.shape, dtype=bool)
        d, peak = differences(x, self.x)
        ratio, rel = path_margins(
            d, peak, self.atols, float(self.rtol), self._ensure_spread(), judged, own=converged
        )
        worst = self._worst(ratio)
        margin = float(ratio.max()) if ratio.size else 0.0
        value = float(rel.max()) if rel.size else 0.0
        status = "pass" if margin <= 1.0 else "diff"
        n_out = int((self.relevant & ~converged).sum())
        note = (
            f"{value:.2e} of a column's peak at most; the largest difference is "
            f"{margin:.2g} of its allowance"
        )
        if n_out:
            note += (
                f"; {n_out} column(s) not converged at the run's tolerance, judged on the "
                f"run's whole scale"
            )
        n_trace = int((~self.relevant).sum())
        if n_trace:
            note += (
                f"; {n_trace} column(s) within {TRACE_ATOLS:g} atol of zero, judged on the "
                f"run's whole scale"
            )
        return _verdict(
            name,
            status,
            value,
            note,
            worst=worst,
            changed=changed,
            extra={
                "margin": _num(margin),
                "n_not_judged": n_out,
                "n_trace": int((~self.relevant).sum()),
            },
        )

    # --- the checks ------------------------------------------------------

    def check_tolerance(self) -> dict[str, Any]:
        skip = self._deterministic("tolerance")
        if skip:
            return skip
        changed = self._tighter(10.0)
        err = self._ensure_drift()
        if err:
            return _verdict(
                "tolerance",
                "exception",
                note=f"the run at a tenfold tighter tolerance raised: {err}",
                changed=changed,
            )
        assert self.moved is not None
        moved = float(self.moved.max()) if self.moved.size else 0.0
        extra: dict[str, Any] = {
            "bound": TOLERANCE_MOVE,
            "value_100": None,
            "n_trace": int((~self.relevant).sum()),
        }
        # A hundredfold tighter too, where the budget allows: whether the
        # answer keeps moving, or has stopped.
        try:
            tighter = self.rerun(**self._tighter(100.0))
            self.tight100 = tighter
            d, peak = differences(self._view(tighter), self.x)
            m100 = np.where(self.relevant, moves(d, peak, self.atols), 0.0)
            extra["value_100"] = _num(m100.max()) if m100.size else 0.0
        except _Out:
            pass
        except Exception:  # noqa: BLE001 - the tenfold run decides; this one adds
            pass
        status = "pass" if moved <= TOLERANCE_MOVE else "diff"
        note = (
            f"the answer moves by {moved:.2e} of a column's peak at rtol/10 and atol/10 "
            f"(bound {TOLERANCE_MOVE:g})"
        )
        if extra["value_100"] is not None:
            note += f", by {extra['value_100']:.2e} at rtol/100"
        if extra["n_trace"]:
            note += (
                f"; {extra['n_trace']} column(s) within {TRACE_ATOLS:g} atol of zero not judged"
            )
        worst = self._worst(self.moved)
        return _verdict(
            "tolerance", status, moved, note, worst=worst, changed=changed, extra=extra
        )

    def check_grid(self) -> dict[str, Any]:
        skip = self._deterministic("grid")
        if skip:
            return skip
        if self.steady:
            return _verdict("grid", "unsupported", note="the run stopped at a steady state")
        t = np.asarray(self.result._time, dtype=float)
        if t.size < 2:
            return _verdict("grid", "unsupported", note="the run has fewer than two time points")
        if self.run.get("sample_times") is not None:
            pieces = [
                np.linspace(a, b, GRID_FACTOR + 1)[:-1] for a, b in zip(t[:-1], t[1:], strict=True)
            ]
            fine = np.concatenate([*pieces, t[-1:]])
            changed: dict[str, Any] = {"sample_times": f"{GRID_FACTOR}x the points"}
            again = self.rerun(sample_times=[float(v) for v in fine])
        else:
            n = (int(self.run["n_points"]) - 1) * GRID_FACTOR + 1
            changed = {"n_points": n}
            again = self.rerun(n_points=n)
        x = np.asarray(again._species, dtype=float)[::GRID_FACTOR]
        return self.compare_x("grid", x, changed)

    def check_jacobian(self) -> dict[str, Any]:
        skip = self._deterministic("jacobian")
        if skip:
            return skip
        backend = self.rec.get("backend") or {}
        if backend.get("jacobian") in ("fd", "difference_quotient") or backend.get(
            "jacobian_retried"
        ):
            return _verdict(
                "jacobian", "skip", note="the run already used a difference-quotient Jacobian"
            )
        changed = {"jacobian": "fd"}
        return self.compare("jacobian", self.rerun(jacobian="fd"), changed)

    def check_linear_solver(self) -> dict[str, Any]:
        skip = self._deterministic("linear_solver")
        if skip:
            return skip
        used = (self.rec.get("backend") or {}).get("linear_solver")
        if used is None:
            return _verdict(
                "linear_solver", "unsupported", note="the record does not say which solver ran"
            )
        other = "dense" if used == "klu" else "klu"
        changed = {"linear_solver": other}
        from bngsim._exceptions import SparseJacobianUnavailableError

        try:
            with warnings.catch_warnings():
                # A solver that did not run is this check's skip, said once.
                warnings.filterwarnings(
                    "ignore",
                    message=r"This rerun ran other code than the run it repeats: "
                    r"linear_solver '\w+' for 'klu' asked;",
                )
                again = self.rerun(linear_solver=other)
        except SparseJacobianUnavailableError as e:
            return _verdict(
                "linear_solver",
                "skip",
                note="KLU cannot run this model: it gives no sparse Jacobian to factorize "
                f"({type(e).__name__})",
                changed=changed,
            )
        ran = (again.record.get("backend") or {}).get("linear_solver")
        if (ran == "klu") != (other == "klu"):
            return _verdict(
                "linear_solver",
                "skip",
                note=f"linear_solver={other!r} ran {ran!r}: {_no_klu(self.result, self.rec)}",
                changed=changed,
            )
        return self.compare("linear_solver", again, changed)

    def check_backend(self) -> dict[str, Any]:
        skip = self._deterministic("backend")
        if skip:
            return skip
        rhs = (self.rec.get("backend") or {}).get("rhs")
        from bngsim import _check_sensitivity

        if rhs != "interpreted" and _check_sensitivity.present(self.result):
            # A rerun repeats the run's sensitivities, which only compiled
            # code computes: codegen=False is refused, not a finding.
            return _verdict(
                "backend",
                "skip",
                note="a sensitivity run has only the compiled path: forward sensitivities "
                "need compiled code",
            )
        if rhs == "interpreted":
            from bngsim._exceptions import CodegenToolchainError

            changed: dict[str, Any] = {"codegen": True}
            try:
                again = self.rerun(codegen=True)
            except CodegenToolchainError as e:
                return _verdict("backend", "skip", note=f"no compiled path here: {_first_line(e)}")
            used = (again.record.get("backend") or {}).get("rhs")
            if used == "interpreted":
                return _verdict(
                    "backend",
                    "skip",
                    note="codegen=True ran interpreted: no compiled path to compare",
                )
        else:
            changed = {"codegen": False}
            again = self.rerun(codegen=False)
        return self.compare("backend", again, changed)

    def check_second_engine(self) -> dict[str, Any]:
        skip = self._deterministic("second_engine")
        if skip:
            return skip
        return _second_engine(self)

    def check_sensitivity(self) -> dict[str, Any]:
        from bngsim import _check_sensitivity

        return _check_sensitivity.check_sensitivity(self)

    def check_sensitivity_fd(self) -> dict[str, Any]:
        from bngsim import _check_sensitivity

        return _check_sensitivity.check_sensitivity_fd(self)

    def check_sensitivity_second_engine(self) -> dict[str, Any]:
        from bngsim import _check_amici

        return _check_amici.check(self)

    def check_replicates(self, n: int) -> dict[str, Any]:
        if not self.stochastic:
            return _verdict("replicates", "unsupported", note="the run is deterministic")
        seed = self.run.get("seed")
        if seed is None:
            return _verdict("replicates", "unsupported", note="the record gives no seed")
        obs = np.asarray(self.result._observables, dtype=float)
        names = list(self.result._observable_names)
        if not names:
            obs, names = self.x, self.names
        last = np.empty((n, len(names)))
        seeds = [(int(seed) + 1 + i) % (2**31 - 1) for i in range(n)]
        for i, s in enumerate(seeds):
            again = self.rerun(seed=s)
            block = again._observables if self.result._observable_names else again._species
            last[i] = np.asarray(block, dtype=float)[-1]
        mean = last.mean(axis=0)
        se = last.std(axis=0, ddof=1) / math.sqrt(n)
        mine = obs[-1]
        z = np.where(se > 0, (mine - mean) / np.where(se > 0, se, 1.0), np.nan)
        cols = [
            {
                "name": names[j],
                "run_last": _num(mine[j]),
                "mean_last": _num(mean[j]),
                "se_last": _num(se[j]),
                "z": _num(z[j]),
            }
            for j in range(len(names))
        ]
        absz = np.where(np.isfinite(z), np.abs(z), -1.0)
        worst = int(np.argmax(absz)) if len(names) and (absz >= 0).any() else None
        note = (
            f"{n} reruns at seeds {seeds[0]}, ..., {seeds[-1]}: at the last time each "
            "observable's mean over them, its standard error, and how many standard errors "
            "this run is from it (None where they all agree); it reports, and judges nothing"
        )
        return _verdict(
            "replicates",
            "pass",
            None if worst is None else float(absz[worst]),
            note,
            worst=None if worst is None else names[worst],
            changed={"seed": f"{n} others"},
            extra={"n_replicates": n, "seeds": seeds, "columns": cols},
        )


def same_as_file(ctx: _Context, name: str, engine: str) -> dict[str, Any] | None:
    """Why another *engine* cannot make the run again from the model file
    alone, as the check *name*'s skip verdict, or None where it can: the run
    started from the model's initial state, of a model unchanged since its
    load, over a plain ``t_span``, at the file's own parameters (none pinned)
    and from the file's initial state. Shared by :func:`_second_engine` and the
    AMICI check (:mod:`bngsim._check_amici`), whose reruns read the file and
    nothing the session did to the model."""
    model = ctx.rec.get("model") or {}
    if ctx.run.get("started_from") != "initial":
        return _verdict(name, "skip", note="the run did not start from the model's initial state")
    if model.get("edits"):
        return _verdict(name, "skip", note="the model was changed after its load")
    if ctx.run.get("sample_times") is not None or ctx.run.get("steady_state"):
        return _verdict(name, "skip", note=f"only a plain t_span run is made again in {engine}")
    if ctx.run.get("parameters_pinned"):
        return _verdict(name, "skip", note="a derived parameter was pinned by set_param")
    path = model.get("path")
    import bngsim

    try:
        fresh = bngsim.Model.load(path) if path else None
    except Exception as e:  # noqa: BLE001
        return _verdict(
            name, "skip", note=f"the model file could not be read again: {_first_line(e)}"
        )
    if fresh is None:
        return _verdict(
            name, "skip", note="the record names no model file (a model loaded from text)"
        )
    held = ctx.run.get("parameters") or {}
    for p in fresh.param_names:
        if p in held and held[p] is not None:
            a, b = float(held[p]), float(fresh.get_param(p))
            if not (a == b or (math.isnan(a) and math.isnan(b))):
                return _verdict(
                    name, "skip", note=f"the run's parameter {p} = {a:g} is not the file's {b:g}"
                )
    start = ctx.run.get("start_state")
    if start is None or not np.allclose(
        np.asarray(start, dtype=float),
        np.asarray(fresh.get_state(), dtype=float),
        rtol=1e-12,
        atol=0.0,
        equal_nan=True,
    ):
        return _verdict(name, "skip", note="the run did not start from the file's initial state")
    return None


def _second_engine(ctx: _Context) -> dict[str, Any]:
    """The run made again in libRoadRunner, from the model file, where it can be
    made the same: an SBML or Antimony model, unchanged since its load, run
    from its initial state at the file's own parameters."""
    name = "second_engine"
    model = ctx.rec.get("model") or {}
    fmt = model.get("format")
    if fmt not in ("sbml", "antimony"):
        return _verdict(
            name,
            "unsupported",
            note=f"libRoadRunner reads SBML and Antimony; this model is {fmt!r}",
        )
    try:
        import roadrunner
    except ImportError:
        return _verdict(
            name,
            "skip",
            note="libRoadRunner is not installed (pip install 'bngsim[roadrunner]'): the model as "
            "bngsim read it was not checked against an independent reader",
        )
    skip = same_as_file(ctx, name, "libRoadRunner")
    if skip:
        return skip
    path = model.get("path")
    try:
        if fmt == "antimony":
            import antimony

            if antimony.loadAntimonyFile(path) < 0:
                raise RuntimeError(antimony.getLastError())
            sbml = antimony.getSBMLString(antimony.getMainModuleName())
            rr = roadrunner.RoadRunner(sbml)
        else:
            rr = roadrunner.RoadRunner(path)
        ids = list(rr.model.getFloatingSpeciesIds()) + list(rr.model.getBoundarySpeciesIds())
        missing = [s for s in ctx.names if s not in ids]
        if missing:
            return _verdict(name, "skip", note=f"libRoadRunner has no species {missing[0]!r}")
        assert ctx.rtol is not None and ctx.atol is not None
        rr.integrator.relative_tolerance = float(ctx.rtol)
        rr.integrator.absolute_tolerance = (
            float(ctx.atols.min()) if ctx.atols.size else float(ctx.atol)
        )
        sel = ["time"] + [f"[{s}]" for s in ctx.names]
        out = rr.simulate(
            float(ctx.run["t_start"]), float(ctx.run["t_end"]), int(ctx.run["n_points"]), sel
        )
        x = np.asarray(out, dtype=float)[:, 1:]
    except Exception as e:  # noqa: BLE001 - the other engine's failure is not bngsim's
        return _verdict(
            name,
            "skip",
            note=f"libRoadRunner could not run it: {type(e).__name__}: {_first_line(e)}",
        )
    v = ctx.compare_x(name, x, {"engine": f"libRoadRunner {roadrunner.__version__}"})
    return v


def _no_klu(result: Result, rec: dict[str, Any]) -> str:
    """Why a run asked onto KLU ran dense."""
    from bngsim._simulator import _HAS_KLU

    if not _HAS_KLU:
        return "this bngsim was built without KLU"
    if (rec.get("backend") or {}).get("jacobian") == "jax":
        return "a JAX Jacobian runs dense only"
    return "KLU needs the model's sparse Jacobian pattern, which this model does not give it"


def _timed_out(e: BaseException) -> bool:
    from bngsim._exceptions import BngsimError

    code = getattr(e, "code", "") if isinstance(e, BngsimError) else ""
    return "timeout" in str(code) or "timed out" in str(e).lower()


def _first_line(e: BaseException) -> str:
    return str(e).splitlines()[0][:300] if str(e) else ""


def _g(x: Any) -> str:
    return "?" if x is None else f"{x:.3g}"


def text(r: dict[str, Any]) -> str:
    """The plain-text form of a report: a verdict line, one line per check,
    and what to do next."""
    lines = []
    verdict = "OK" if r.get("ok") else "NOT OK"
    head = f"{verdict}: {r.get('n_pass', 0)} passed, {r.get('n_diff', 0)} moved the answer"
    if r.get("n_not_resolved"):
        head += f", {r['n_not_resolved']} species not resolved"
    if r.get("method"):
        head += f" ({r['method']} run"
        if r.get("rtol") is not None:
            head += f", rtol {_g(r['rtol'])}"
        head += ")"
    lines.append(head)
    for c in r.get("checks", []):
        line = f"  {c['name']}: {c['status']}"
        if c.get("note"):
            line += f": {c['note']}"
        if c.get("worst_column"):
            line += f"; most in {c['worst_column']}"
        lines.append(line)
    if r.get("next"):
        lines.append(f"next: {r['next']}")
    return "\n".join(lines)
