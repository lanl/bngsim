"""A digest of a Result's values, bounded in size (issue #1704).

:meth:`bngsim.Result.digest` is built here. It is for a caller that cannot look
at a plot (an AI agent, a ``bngsim`` command's ``--json``, a tool server): the
dozen numbers per column such a caller looks for, and no reading of them beyond
what the run's own tolerances allow.

Where a reading needs a threshold, it takes the one the steady-state code's
``criterion="scaled"`` takes (issue #1001; ``ss_scaled_tolerance`` in
``src/steady_state.cpp``): a move of a value ``y`` is within tolerance when it is
at most ``rtol·|y| + min(atol, rtol·largest)``, with the run's ``rtol`` (at least
1e-10) and ``atol`` (a species' own as applied; for an observable, whose species
the record does not name, what the run's rule gives a species of its size: see
:func:`_column_atol`), and ``largest`` the largest magnitude of any column of the
same kind over the run. That one tolerance decides

- whether a column has *settled*: no value from the last sample at or before
  the start of the last tenth of the window on is further from the last value
  than the tolerance (the run test of ``criterion="scaled"``, taken over the
  run's own tail in place of a run taken on from its end);
- what counts as an *interior extremum*: an extreme the column reaches by more
  than the tolerance and leaves by more than the tolerance at it, so an
  integrator's wobble around a value it has reached is not counted, whatever
  phase a periodic column starts in (and *monotone* is no such turn);
- whether a column *never moves* (its range is within the tolerance) or is
  *negative* (its minimum is below ``-min(atol, rtol·largest)``).

A stochastic run's values are counts that move by whole molecules, so of a
single stochastic trajectory the digest gives the first, last, minimum and
maximum values and the flags, and reads nothing from its shape. A stack of runs
(``Result.squeeze``, ``run_batch(squeeze=True)``) gets statistics across its
runs at the last time. Where the record gives no ``rtol`` or ``atol`` (an older
file, a stack whose rows ran at different ones) nothing is judged against a
tolerance: there are no readings of shape, and the flags take any value below 0
as negative and any move as a move.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim import _record
from bngsim._exceptions import UsageError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._result import Result

#: Incremented by a change that removes or renames a key, or changes what one
#: holds. A key added is no change of version.
DIGEST_VERSION = 1

#: How many columns of each kind a digest lists by default.
DEFAULT_MAX_COLUMNS = 20

#: The share of the window, at its end, that ``settled`` and ``tail_change``
#: read.
TAIL_FRACTION = 0.1

#: The finest rtol the tolerance takes: ``kCriterionRtolLeast`` in
#: ``src/steady_state.cpp``, under which a run does not tell a move from its
#: own error.
RTOL_LEAST = 1e-10

#: The share of the largest value under which the run's atol rule
#: (``Simulator._ATOL_SCALE_RELEVANT``, issue #1361) gives a species no scale
#: of its own.
ATOL_SCALE_RELEVANT = 1e-6

#: How many of a result's notices a digest lists, in their order, and the
#: length a notice's message is cut to; ``Result.notices`` holds them whole.
MAX_NOTICES = 20
MAX_NOTICE_CHARS = 500

#: The quantiles a stack's statistics give at the last time, by key.
QUANTILES = (("q05", 0.05), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q95", 0.95))

#: The kinds of column a digest can list, with the Result's attributes for
#: their values and names.
_KINDS = {
    "observables": ("_observables", "_observable_names"),
    "species": ("_species", "_species_names"),
}


class Digest(dict):  # type: ignore[type-arg]
    """:meth:`Result.digest`'s dict, whose ``str()`` is its plain-text form.

    It is a ``dict`` in every other way: ``json.dumps(digest, allow_nan=False)``
    writes it, and every number in it is finite or ``None``.
    """

    __slots__ = ()

    def __str__(self) -> str:
        return text(self)

    def text(self) -> str:
        """The plain-text form, as ``str()`` gives it."""
        return text(self)


def _num(x: Any) -> float | None:
    """*x* as a float, or ``None`` where it is not finite (JSON has no NaN)."""
    v = float(x)
    return v if math.isfinite(v) else None


def _run_fields(result: Result) -> tuple[str | None, float | None, float | None, Any]:
    """``(method, rtol, atol ceiling, per-species atol)`` as *result*'s record
    holds them, each ``None`` where it holds none (or a stack's rows differ),
    without building the record."""
    frozen = result._record_frozen
    info = result._run_info
    # A stack whose rows ran at different absolute tolerances merges them to
    # None, and its "given" None is no "the caller gave nothing": no ceiling.
    differs = result._record_differs or ()
    atol_differs = any(path.startswith("run.atol") for path in differs)
    method, rtol, ceiling, per = _recorded(result, frozen, info)
    if atol_differs:
        return method, rtol, None, None
    return method, rtol, ceiling, per


def _recorded(
    result: Result, frozen: Any, info: Any
) -> tuple[str | None, float | None, float | None, Any]:
    if frozen is not None:
        run = frozen.get("run") or {}
        atol = run.get("atol") or {}
        per = atol.get("per_species")
        ceiling = _ceiling(atol.get("given"), atol.get("scalar"), per)
        return run.get("method"), run.get("rtol"), ceiling, per
    if info is None:
        return None, None, None, None
    if isinstance(info, dict):  # a stack's merged record (_record.squeeze_into)
        run = info.get("run") or {}
        atol = run.get("atol") or {}
        per = result._run_atol
        ceiling = _ceiling(atol.get("given"), atol.get("scalar"), per)
        return run.get("method"), run.get("rtol"), ceiling, per
    per = result._run_atol
    return info.method, info.rtol, _ceiling(info.atol_given, info.atol_scalar, per), per


def _ceiling(given: Any, scalar: float | None, per_species: Any) -> float | None:
    """The run's ceiling on any species' atol: the scalar where the caller gave
    a number or nothing (each species took the smaller of it and its own
    scale's), and the largest per-species value where the caller gave its own
    answer (a vector, ``"auto"``, a tracking atol), which the record's scalar,
    left at the default, is no ceiling of."""
    if given is None or isinstance(given, int | float):
        return scalar
    if per_species is not None and len(per_species):
        return float(max(per_species))
    return scalar


def _n_turns(col: NDArray[np.float64], rtol: float, floor: float) -> int:
    """The interior extrema of *col* (all finite): turns whose swing back from
    the extreme they turn at exceeds the tolerance there."""
    d = np.diff(col)
    moving = np.flatnonzero(d)
    if moving.size < 2:
        return 0
    sign = np.sign(d[moving])
    flips = np.flatnonzero(sign[1:] != sign[:-1])
    if flips.size == 0:
        return 0
    # The first point, each local extreme (a plateau's first point), the last:
    # between two of them the column moves one way, so a turn can only be at one.
    at = np.concatenate(([0], moving[flips] + 1, [col.size - 1]))
    values = col[at].tolist()
    # Until the column has moved by more than the tolerance it has no way: the
    # extremes it has reached so far, either of which it may leave from. The
    # one it leaves from is no turn (it was reached by less than the
    # tolerance), whichever comes first, so the count does not depend on the
    # phase a periodic column starts in.
    hi = lo = values[0]
    way = 0
    ext = hi
    n = 0
    for x in values[1:]:
        if way == 0:
            if x - lo > rtol * abs(lo) + floor:
                way, ext = 1, x
            elif hi - x > rtol * abs(hi) + floor:
                way, ext = -1, x
            else:
                hi, lo = max(hi, x), min(lo, x)
        elif way > 0:
            if x > ext:
                ext = x
            elif ext - x > rtol * abs(ext) + floor:
                n += 1
                way, ext = -1, x
        else:
            if x < ext:
                ext = x
            elif x - ext > rtol * abs(ext) + floor:
                n += 1
                way, ext = 1, x
    return n


def _column_atol(
    atol_vec: Any, atol: float, j: int, species: bool, scale: float, largest: float, rtol: float
) -> float:
    """The absolute tolerance column *j* is judged by: a species' own as the
    run applied it; any other column (an observable sums species the record
    does not name) the one the run's rule gives a species of its size (issue
    #1420, ``Simulator._atol_from_scale``): ``rtol`` times its largest
    magnitude, but not under :data:`ATOL_SCALE_RELEVANT` of *largest*, and
    never over the scalar ``atol``, the run's ceiling."""
    if species and atol_vec is not None and j < len(atol_vec):
        return float(atol_vec[j])
    return min(atol, rtol * max(scale, ATOL_SCALE_RELEVANT * largest))


def _notice(n: dict[str, str]) -> dict[str, str]:
    """A notice with its message cut at :data:`MAX_NOTICE_CHARS`."""
    msg = n.get("message", "")
    if len(msg) > MAX_NOTICE_CHARS:
        n = {**n, "message": msg[: MAX_NOTICE_CHARS - 1] + "…"}
    return n


def _rank(nonfinite: NDArray[np.bool_], span: NDArray[np.float64], n: int) -> list[int]:
    """The indices of the *n* columns to list, in column order: those holding a
    non-finite value first, then by range, largest first, ties by index."""
    if n >= span.size:
        return list(range(span.size))
    order = np.lexsort((np.arange(span.size), -np.nan_to_num(span, nan=0.0), ~nonfinite))
    return sorted(int(i) for i in order[:n])


def digest(
    result: Result, *, species: bool = False, max_columns: int = DEFAULT_MAX_COLUMNS
) -> Digest:
    """:meth:`Result.digest`."""
    if not isinstance(species, bool):
        raise UsageError(f"digest(species=...) takes True or False, got {species!r}")
    if isinstance(max_columns, bool) or not isinstance(max_columns, int | np.integer):
        raise UsageError(f"digest(max_columns=...) takes a whole number, got {max_columns!r}")
    if max_columns < 0:
        raise UsageError(f"digest(max_columns=...) takes 0 or more, got {max_columns}")
    max_columns = int(max_columns)

    method, rtol, atol, atol_vec = _run_fields(result)
    stochastic = None if method is None else method in _record.STOCHASTIC
    time = np.asarray(result._time, dtype=np.float64)
    stack = result._species.ndim == 3
    n_points = int(time.shape[-1]) if time.size else 0
    t_start = _num(time[0]) if n_points else None
    t_end = _num(time[-1]) if n_points else None

    out = Digest()
    out["digest_version"] = DIGEST_VERSION
    out["method"] = method
    out["stochastic"] = stochastic
    out["n_runs"] = int(result._species.shape[0]) if stack else None
    out["t_start"] = t_start
    out["t_end"] = t_end
    out["n_points"] = n_points
    out["n_sensitivity_params"] = len(result._sensitivity_params)
    out["n_sensitivity_ic_species"] = len(result._sensitivity_ic_species)
    out["n_notices"] = len(result._notices)
    out["notices"] = [_notice(n.as_dict()) for n in result._notices[:MAX_NOTICES]]

    kinds = ["observables"] if len(result._observable_names) else ["species"]
    if species and "species" not in kinds:
        kinds.append("species")
    out["kinds"] = list(kinds)

    # Readings against the tolerance need a deterministic method whose rtol and
    # atol the record gives (a stack whose rows ran at different ones gives
    # none); readings of shape, one run besides.
    tol_known = False
    r = a = 0.0
    if stochastic is False and rtol is not None and atol is not None:
        tol_known = True
        r, a = max(float(rtol), RTOL_LEAST), float(atol)
    reads = tol_known and not stack
    out["rtol"] = r if tol_known else None
    out["atol"] = a if tol_known else None
    for kind in kinds:
        values_attr, names_attr = _KINDS[kind]
        data = np.asarray(getattr(result, values_attr), dtype=np.float64)
        names = list(getattr(result, names_attr))
        out[kind] = _kind(
            data,
            names,
            time,
            stack=stack,
            reads=reads,
            tol_known=tol_known,
            rtol=r,
            atol=a,
            atol_vec=atol_vec if tol_known else None,
            species=kind == "species",
            max_columns=max_columns,
        )
    return out


def _kind(
    data: NDArray[np.float64],
    names: list[str],
    time: NDArray[np.float64],
    *,
    stack: bool,
    reads: bool,
    tol_known: bool,
    rtol: float,
    atol: float,
    atol_vec: Any,
    species: bool,
    max_columns: int,
) -> dict[str, Any]:
    n_cols = len(names)
    n_points = int(time.shape[-1]) if time.size else 0
    if n_cols == 0 or n_points == 0 or data.size == 0:
        return {"n_columns": n_cols, "n_left_out": 0, "columns": []}
    axes = (0, 1) if stack else (0,)
    finite = np.isfinite(data)
    nonfinite = ~finite.all(axis=axes)
    lo = np.where(finite, data, np.inf).min(axis=axes)
    hi = np.where(finite, data, -np.inf).max(axis=axes)
    span = hi - lo  # inf for a column with no finite value
    magnitude = np.abs(np.where(finite, data, 0.0)).max(axis=axes)
    largest = float(magnitude.max())
    chosen = _rank(nonfinite, np.where(np.isfinite(span), span, np.inf), max_columns)

    tail = None
    if reads and n_points >= 2 and time[-1] > time[0]:
        # From the last sample at or before the start of the last tenth, so
        # the samples read span the whole tenth however the grid falls.
        t0, t1 = float(time[0]), float(time[-1])
        cut = t1 - TAIL_FRACTION * (t1 - t0)
        cut += 1e-12 * max(abs(cut), t1 - t0)  # a sample on the cut, give or take
        first = int(np.searchsorted(time, cut, side="right")) - 1
        tail = slice(max(first, 0), None)

    columns = []
    for j in chosen:
        own = _column_atol(atol_vec, atol, j, species, float(magnitude[j]), largest, rtol)
        floor = min(own, rtol * largest)
        mn, mx = float(lo[j]), float(hi[j])
        any_finite = math.isfinite(mn)
        row: dict[str, Any] = {"name": names[j]}
        if stack:
            row.update(_stack_stats(data[:, -1, j]))
            row["min"] = _num(mn)
            row["max"] = _num(mx)
        else:
            col = data[:, j]
            row["first"] = _num(col[0])
            row["last"] = _num(col[-1])
            row["min"] = _num(mn)
            if reads:
                row["t_min"] = (
                    _num(time[int(np.argmin(np.where(finite[:, j], col, np.inf)))])
                    if any_finite
                    else None
                )
            row["max"] = _num(mx)
            if reads:
                row["t_max"] = (
                    _num(time[int(np.argmax(np.where(finite[:, j], col, -np.inf)))])
                    if any_finite
                    else None
                )
        if reads:
            col = data[:, j]
            if nonfinite[j]:
                row.update(
                    monotone=None,
                    n_interior_extrema=None,
                    tail_from=None,
                    tail_change=None,
                    settled=None,
                )
            else:
                n = _n_turns(col, rtol, floor)
                row["monotone"] = n == 0
                row["n_interior_extrema"] = n
                if tail is None:
                    row["tail_from"] = None
                    row["tail_change"] = None
                    row["settled"] = None
                else:
                    last = float(col[-1])
                    move = float(np.abs(col[tail] - last).max())
                    rng = mx - mn
                    row["tail_from"] = _num(time[tail.start])
                    row["tail_change"] = _num(move / rng) if rng > 0 else 0.0
                    row["settled"] = bool(move <= rtol * abs(last) + floor)
        row["non_finite"] = bool(nonfinite[j])
        if nonfinite[j]:
            bad = ~finite[:, :, j].all(axis=0) if stack else ~finite[:, j]
            row["t_non_finite"] = _num(time[int(np.argmax(bad))])
        else:
            row["t_non_finite"] = None
        if tol_known:
            row["negative"] = bool(any_finite and mn < -floor)
            row["constant"] = bool(
                not nonfinite[j] and mx - mn <= rtol * max(abs(mn), abs(mx)) + floor
            )
        else:
            row["negative"] = bool(any_finite and mn < 0.0)
            row["constant"] = bool(not nonfinite[j] and mx == mn)
        columns.append(row)
    return {"n_columns": n_cols, "n_left_out": n_cols - len(chosen), "columns": columns}


def _stack_stats(last: NDArray[np.float64]) -> dict[str, Any]:
    """Statistics across a stack's runs of one column at the last time."""
    ok = last[np.isfinite(last)]
    stats: dict[str, Any] = {"last_n_finite": int(ok.size)}
    if ok.size == 0:
        stats["last_mean"] = None
        stats["last_std"] = None
        for key, _ in QUANTILES:
            stats[f"last_{key}"] = None
        stats["last_min"] = None
        stats["last_max"] = None
        return stats
    stats["last_mean"] = _num(ok.mean())
    stats["last_std"] = _num(ok.std(ddof=1)) if ok.size > 1 else None
    qs = np.quantile(ok, [q for _, q in QUANTILES])
    stats["last_min"] = _num(ok.min())
    for (key, _), v in zip(QUANTILES, qs, strict=True):
        stats[f"last_{key}"] = _num(v)
    stats["last_max"] = _num(ok.max())
    return stats


def _g(x: Any) -> str:
    return "?" if x is None else f"{x:.6g}"


def text(d: dict[str, Any]) -> str:
    """The plain-text form of a digest: one line for the run, one per notice,
    and one per column."""
    lines = []
    method = d.get("method") or "unknown method"
    runs = f", {d['n_runs']} runs" if d.get("n_runs") is not None else ""
    if d.get("n_points"):
        window = f"t {_g(d['t_start'])} to {_g(d['t_end'])}, {d['n_points']} points"
    else:
        window = "no time points"
    head = f"{method} run{runs}, {window}"
    if d.get("rtol") is not None:
        head += f", rtol {_g(d['rtol'])}, atol {_g(d['atol'])}"
    if d.get("n_sensitivity_params") or d.get("n_sensitivity_ic_species"):
        head += (
            f"; sensitivities to {d['n_sensitivity_params']} parameters and "
            f"{d['n_sensitivity_ic_species']} initial amounts"
        )
    lines.append(head)
    notices = d.get("notices") or []
    if not notices:
        lines.append("notices: none")
    for n in notices:
        lines.append(f"notice {n['code']}: {n['message']}")
    more = d.get("n_notices", len(notices)) - len(notices)
    if more > 0:
        lines.append(f"{more} more notices in Result.notices")
    for kind in d.get("kinds", []):
        block = d[kind]
        shown = len(block["columns"])
        line = f"{kind}: {shown} of {block['n_columns']}"
        if block["n_left_out"]:
            line += (
                f" ({block['n_left_out']} left out; listed are any with a non-finite "
                f"value, then the widest ranges)"
            )
        lines.append(line)
        for row in block["columns"]:
            lines.append("  " + _row_text(row))
    return "\n".join(lines)


def _row_text(row: dict[str, Any]) -> str:
    parts = []
    if "last_mean" in row:
        parts.append(
            f"at the last time over {row['last_n_finite']} finite runs: mean "
            f"{_g(row['last_mean'])}, sd {_g(row['last_std'])}, min {_g(row['last_min'])}, "
            f"q05 {_g(row['last_q05'])}, q25 {_g(row['last_q25'])}, median "
            f"{_g(row['last_median'])}, q75 {_g(row['last_q75'])}, q95 {_g(row['last_q95'])}, "
            f"max {_g(row['last_max'])}"
        )
        parts.append(f"over all runs min {_g(row['min'])}, max {_g(row['max'])}")
    else:
        parts.append(f"first {_g(row['first'])}, last {_g(row['last'])}")
        if "t_min" in row:
            parts.append(
                f"min {_g(row['min'])} at t={_g(row['t_min'])}, "
                f"max {_g(row['max'])} at t={_g(row['t_max'])}"
            )
        else:
            parts.append(f"min {_g(row['min'])}, max {_g(row['max'])}")
    if row.get("monotone") is not None:
        n = row["n_interior_extrema"]
        parts.append(
            "monotone"
            if row["monotone"]
            else f"{n} interior {'extremum' if n == 1 else 'extrema'}"
        )
    if "settled" in row and row["settled"] is not None:
        state = "settled" if row["settled"] else "not settled"
        parts.append(
            f"change since t={_g(row['tail_from'])} {_g(row['tail_change'])} of range, {state}"
        )
    flags = []
    if row["non_finite"]:
        flags.append(f"non-finite from t={_g(row['t_non_finite'])}")
    if row["negative"]:
        flags.append("negative")
    if row["constant"]:
        flags.append("never moves")
    if flags:
        parts.append("FLAGS: " + ", ".join(flags))
    return f"{row['name']}: " + "; ".join(parts)
