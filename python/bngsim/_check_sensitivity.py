""":meth:`bngsim.Result.check`'s checks of a run's forward sensitivities
(issue #1893).

A run made with ``sensitivity_params=`` or ``sensitivity_ic=`` holds
``dY_i/dθ_j`` for each species ``i`` and each parameter or initial amount
``θ_j``. These checks judge each such *series* (one species, one ``θ``, over
time) on its own scale, as :mod:`bngsim._check` judges each species: a raw
``dY/dθ`` is in the units of ``Y`` over those of ``θ``, so two parameters'
columns share no scale, and a small series beside a large one is judged as
carefully as the large one.

A series' absolute floor is the one the solver gave it. CVODES holds a
sensitivity to ``rtol·|s| + atolS``, and bngsim sets
``atolS_ij = atol_i·max(|x_i(0)|, 1)/pbar_j`` (``src/cvode_simulator.cpp``,
issues #196 and #214), with ``atol_i`` the species' own atol as applied and
``pbar_j`` the parameter's magnitude (1 for an initial amount, or for a
parameter at 0). A step solves a whole column at once, so every entry of it
also carries about ``eps·max|s_j|`` (issue #177): the floor is the larger of
the two.

- ``"sensitivity"``: the sensitivities of the tolerance check's run, at a
  tenth of the ``rtol`` and atol, against the run's. A series' *move* is its
  largest difference past its floor, over its peak; the check fails where a
  series moves by more than :data:`bngsim._check.TOLERANCE_MOVE`. A series
  whose peak is within :data:`bngsim._check.TRACE_ATOLS` of its floor, where
  the floor and not ``rtol`` holds it, is not resolved by the run, and is
  named: its move past that many floors is read on the largest peak of its
  column, so its noise near zero is no finding and a departure as large as the
  column's own values is one. So is a series past the bound that moves as far
  again at a hundredth of the tolerance (:data:`FLOOR_SHARE`): a floor the
  solver's arithmetic sets (issue #177), not the tolerance, holds it, and its
  move is read on its column's largest peak.
- ``"sensitivity_fd"``: a difference quotient. Each ``θ`` is moved by
  :data:`FD_STEP` of its value both ways (an initial amount through the start
  state, a parameter through :meth:`Model.set_param`, so a derived parameter
  and an initial amount that read it follow), and the run made again at a
  hundredth of its tolerance with no sensitivities, so the quotient reads
  nothing of the sensitivity code: not its right-hand side, not its seeds.
  The quotient's own error is read cell by cell from how far its forward and
  backward halves differ: about ``h·d²Y/dθ²`` where the trajectory is smooth in
  ``θ``, and as large as the jump itself near a switch or event time that
  ``θ`` moves, where a central quotient is half the jump (issue #368). So the
  check widens there by itself, and needs no list of switch times. On top of
  that, the quotient may be off by :data:`FD_NOISE` times its runs' error over
  its step, each run erring by about the run's own drift scaled to the
  quotient's tolerance (and by no less than that tolerance), and the run's own
  sensitivities by what a path may differ by (:func:`bngsim._check.allowance`,
  with the series' move at the tighter tolerance as its drift). A series whose
  peak is within the quotient's noise of zero is not resolved by it, and is
  named. A ``θ`` at 0 (a parameter, or an initial amount at 0) is moved one
  way, up, and its quotient's error read from a second step of twice the size.

Neither check judges a row that is not finite in the run by design (an
assignment-rule species whose sensitivity is refused,
``Result.ar_sensitivity_refused``); the verdict counts them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim import _check
from bngsim._check import (
    MAX_NAMED,
    TOLERANCE_MOVE,
    TRACE_ATOLS,
    _num,
    _Out,
    _verdict,
    allowance,
    differences,
    moves,
    resolved,
    spread,
)
from bngsim._exceptions import UsageError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._check import _Context
    from bngsim._result import Result

#: A difference quotient's step, as a share of the value it moves: the step at
#: which a quotient of whole trajectories is clear of their integration noise
#: (one of 1e-4 or less reads the solver's own error, systematic in the step,
#: as a derivative; issue #49) and its truncation error, of order the square of
#: the step for a central quotient, is still small.
FD_STEP = 1e-3
#: How much tighter than the run the quotient's runs are made.
FD_TIGHTER = 100.0
#: How many times the quotient's error estimate (how far its two halves
#: differ, cell by cell) a series may differ from it by, on top of the rest.
FD_SPREAD = 2.0
#: A series past the bound whose move between the hundredfold and the tenfold
#: tighter runs is at least this share of its move at the tenfold one is held
#: by a floor, not the tolerance (:func:`_floor_held`).
FLOOR_SHARE = 0.5
#: How many times its error over its step a quotient may be off by: each of
#: its runs errs by about the run's own drift (or spread) scaled to the
#: quotient's tolerance, and by no less than that tolerance's rtol·peak plus
#: atol; the factor is the one a path's drift takes
#: (:data:`bngsim._check.DRIFT_FACTOR`).
FD_NOISE = _check.DRIFT_FACTOR


def present(result: Result) -> bool:
    """Whether *result*'s run computed forward sensitivities."""
    return bool(len(result._sensitivity_params) or len(result._sensitivity_ic_species))


def _block(result: Result) -> NDArray[np.float64]:
    """The run's sensitivities as one ``(n_times, n_species, n_columns)``
    array: the parameter columns, then the initial-amount columns."""
    n_t = len(result._time)
    n_sp = len(result._species_names)
    parts = []
    for block, names in (
        (result._sensitivities, result._sensitivity_params),
        (result._sensitivities_ic, result._sensitivity_ic_species),
    ):
        b = np.asarray(block, dtype=float)
        if len(names) == 0:
            continue
        if b.ndim != 3 or b.shape != (n_t, n_sp, len(names)):
            raise ValueError(
                f"the sensitivity block is {b.shape}, not ({n_t}, {n_sp}, {len(names)})"
            )
        parts.append(b)
    if not parts:
        return np.empty((n_t, n_sp, 0))
    return np.concatenate(parts, axis=2)


class _Columns:
    """The run's sensitivity columns, and the floors the solver held each
    series to."""

    def __init__(self, ctx: _Context) -> None:
        result = ctx.result
        self.rtol: float | None = ctx.rtol
        self.params = list(result._sensitivity_params)
        self.ics = list(result._sensitivity_ic_species)
        self.names = self.params + self.ics
        self.species = list(ctx.names)
        self.S = _block(result)
        values = ctx.run.get("parameters") or {}
        self.theta: list[float | None] = []
        pbar = []
        for p in self.params:
            v = values.get(p)
            v = None if v is None else float(v)
            self.theta.append(v)
            pbar.append(abs(v) if v is not None and math.isfinite(v) and v != 0.0 else 1.0)
        # The start state, by name: the state the solver set its tolerances
        # from, which holds promoted entries after the species where an event
        # or a rule promotes a parameter into it.
        start = ctx.run.get("start_state")
        state = [float(x) for x in start] if start is not None else []
        try:
            state_names = list(result.state_names)
        except Exception:  # noqa: BLE001 - a loaded or stacked Result holds none
            state_names = []
        if len(state_names) != len(state):
            state_names = self.species[: len(state)]
        self.state_index = {n: i for i, n in enumerate(state_names)}
        self.state = state
        x0 = np.asarray(
            [state[self.state_index[s]] if s in self.state_index else 0.0 for s in self.species],
            dtype=float,
        )
        for ic in self.ics:
            v = state[self.state_index[ic]] if ic in self.state_index else None
            self.theta.append(v)
            pbar.append(1.0)
        self.pbar = np.asarray(pbar, dtype=float)
        scale = np.maximum(np.where(np.isfinite(x0), np.abs(x0), 0.0), 1.0)
        # atolS_ij = atol_i·max(|x_i(0)|, 1)/pbar_j, as the solver set it.
        self.atol = (ctx.atols * scale)[:, None] / self.pbar[None, :]
        # A row the run reports as not finite throughout is refused by design.
        refused = set(getattr(result, "_ar_sens_refused", ()) or ())
        finite_any = np.isfinite(self.S).any(axis=0) if self.S.size else np.zeros(self.atol.shape)
        self.refused = np.zeros(self.atol.shape, dtype=bool)
        for i, s in enumerate(self.species):
            if s in refused:
                self.refused[i, :] = ~finite_any[i, :]
        # Every entry of a column also carries the roundoff of the column's
        # largest (issue #177): the floor is the larger of the two.
        with np.errstate(invalid="ignore"):
            colmax = np.where(np.isfinite(self.S), np.abs(self.S), 0.0).max(axis=(0, 1))
        if not self.S.size:
            colmax = np.zeros(len(self.names))
        eps = float(np.finfo(float).eps)
        self.atol = np.maximum(self.atol, eps * colmax[None, :])

    def label(self, i: int, j: int) -> str:
        theta = self.names[j] if j < len(self.params) else f"{self.names[j]}(0)"
        return f"d{self.species[i]}/d{theta}"

    def kind(self, j: int) -> str:
        return "parameter" if j < len(self.params) else "initial_amount"

    def flat(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_times, n_species, n_columns)`` as ``(n_times, series)``."""
        return np.asarray(x, dtype=float).reshape(x.shape[0], -1)


def _skip(ctx: _Context, name: str) -> dict[str, Any] | None:
    """The verdict of a sensitivity check that does not apply, or None."""
    skip = ctx._deterministic(name)
    if skip:
        return skip
    if not present(ctx.result):
        return _verdict(name, "unsupported", note="the run computed no sensitivities")
    if ctx.steady:
        return _verdict(
            name,
            "unsupported",
            note="the run stopped at a steady state, at a time its tolerance sets: its "
            "sensitivities are not checked",
        )
    return None


def _moves(
    cols: _Columns, x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Each series' move from the run's in *x* (as :func:`bngsim._check.moves`
    on its own peak and floor, or for a series the run does not resolve, its
    move past :data:`TRACE_ATOLS` floors on its column's largest peak), which
    series the run resolves, and each series' largest difference."""
    n_sp, n_cols = cols.atol.shape
    d, peak = differences(cols.flat(x), cols.flat(cols.S))
    atol = cols.atol.reshape(-1)
    keep = resolved(peak, atol, 0.0 if cols.rtol is None else cols.rtol)
    col_peak = np.where(cols.refused, 0.0, peak.reshape(n_sp, n_cols)).max(axis=0)
    scale = np.maximum(np.repeat(col_peak[None, :], n_sp, axis=0).reshape(-1), peak)
    over = np.maximum(d - TRACE_ATOLS * atol, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        trace = np.where(
            scale > 0, over / np.where(scale > 0, scale, 1.0), np.where(over > 0, np.inf, 0.0)
        )
    move = np.where(keep, moves(d, peak, atol), np.where(np.isinf(d), np.inf, trace))
    move = np.where(cols.refused.reshape(-1), 0.0, move)
    return move, keep, d


def _columns_entry(cols: _Columns, score: NDArray[np.float64]) -> dict[str, Any]:
    """Each column's worst series, the worst columns first."""
    n_sp, n_cols = cols.atol.shape
    s = np.where(np.isnan(score), np.inf, score).reshape(n_sp, n_cols)
    out: list[tuple[float, int, str | None, float | None]] = []
    for j in range(n_cols):
        live = ~cols.refused[:, j]
        if not live.any():
            out.append((-1.0, j, None, None))
            continue
        col = np.where(live, s[:, j], -np.inf)
        i = int(np.argmax(col))
        out.append((float(col[i]), j, cols.species[i], float(col[i])))
    out.sort(key=lambda e: (-e[0], e[1]))
    return {
        "n_columns": n_cols,
        "columns": [
            {"name": cols.names[j], "kind": cols.kind(j), "worst_species": sp, "value": _num(v)}
            for _, j, sp, v in out[:MAX_NAMED]
        ],
    }


def _named(cols: _Columns, mask: NDArray[np.bool_]) -> list[str]:
    n_sp, n_cols = cols.atol.shape
    m = mask.reshape(n_sp, n_cols)
    return [cols.label(i, j) for j in range(n_cols) for i in range(n_sp) if m[i, j]]


def _worst(cols: _Columns, score: NDArray[np.float64]) -> str | None:
    if not score.size or not (score > 0).any():
        return None
    k = int(np.argmax(np.where(np.isnan(score), np.inf, score)))
    n_cols = cols.atol.shape[1]
    return cols.label(k // n_cols, k % n_cols)


def _context(ctx: _Context) -> _Columns:
    """The run's columns, made once per check() call."""
    if ctx.sens_columns is None:
        ctx.sens_columns = _Columns(ctx)
    cols: _Columns = ctx.sens_columns
    return cols


def _drift(
    ctx: _Context, cols: _Columns
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Each series' move at the tenfold tighter tolerance, which series the run
    resolves, and the drift a quotient's allowance reads (:func:`spread`, on
    the 2x looser run's sensitivities where it ran)."""
    if ctx.sens_drift is None:
        assert ctx.tight is not None
        ctx.sens_drift = _moves(cols, _block(ctx.tight))
    drift: tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]] = ctx.sens_drift
    return drift


def check_sensitivity(ctx: _Context) -> dict[str, Any]:
    """The ``"sensitivity"`` check: see the module docstring."""
    name = "sensitivity"
    skip = _skip(ctx, name)
    if skip:
        return skip
    changed = ctx._tighter(10.0)
    err = ctx._ensure_drift()
    if err:
        return _verdict(
            name,
            "exception",
            note=f"the run at a tenfold tighter tolerance raised: {err}",
            changed=changed,
        )
    cols = _context(ctx)
    move, keep, d10 = _drift(ctx, cols)
    live = ~cols.refused.reshape(-1)
    trace = live & ~keep
    held = _floor_held(ctx, cols, move, d10, live & keep)
    if held.any():
        # Read on its column's largest peak, as a series the run does not
        # resolve is: the arithmetic, not the tolerance, holds it.
        n_sp, n_cols = cols.atol.shape
        peak = np.where(np.isfinite(cols.S), np.abs(cols.S), 0.0).max(axis=0)
        col_peak = np.where(cols.refused, 0.0, peak).max(axis=0)
        scale = np.maximum(np.repeat(col_peak[None, :], n_sp, axis=0), peak).reshape(-1)
        over = np.maximum(d10 - cols.atol.reshape(-1), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            on_column = np.where(
                scale > 0, over / np.where(scale > 0, scale, 1.0), np.where(over > 0, np.inf, 0.0)
            )
        move = np.where(held, on_column, move)
    value = float(move.max()) if move.size else 0.0
    status = "pass" if value <= TOLERANCE_MOVE else "diff"
    n_series = int(live.sum())
    note = (
        f"the sensitivities move by {value:.2e} of a series' peak at rtol/10 and atol/10 "
        f"(bound {TOLERANCE_MOVE:g}), over {n_series} series of {len(cols.names)} column(s)"
    )
    n_trace = int(trace.sum())
    if n_trace:
        note += (
            f"; {n_trace} series within {TRACE_ATOLS:g} times their sensitivity atol of zero, "
            "judged on their column's largest peak"
        )
    n_held = int(held.sum())
    if n_held:
        note += (
            f"; {n_held} series moving as far at rtol/100 as at rtol/10, held by the "
            "arithmetic's floor and not the tolerance, judged on their column's largest peak"
        )
    n_refused = int((~live).sum())
    if n_refused:
        note += f"; {n_refused} series not finite in the run by design, not judged"
    named = _named(cols, trace | held)
    extra: dict[str, Any] = {
        "bound": TOLERANCE_MOVE,
        **_columns_entry(cols, move),
        "n_series": n_series,
        "n_not_resolved": n_trace + n_held,
        "not_resolved": named[:MAX_NAMED],
        "n_floor_held": n_held,
        "n_refused": n_refused,
    }
    return _verdict(
        name, status, value, note, worst=_worst(cols, move), changed=changed, extra=extra
    )


def _floor_held(
    ctx: _Context,
    cols: _Columns,
    move: NDArray[np.float64],
    d10: NDArray[np.float64],
    candidates: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """The series past the bound that a floor holds, not the tolerance: those
    that move between the hundredfold and the tenfold tighter runs by at least
    :data:`FLOOR_SHARE` of their move at the tenfold one. A move the tolerance
    sets shrinks about tenfold from one to the next; the floor the solver's
    arithmetic sets under a sensitivity's error (issue #177) does not shrink
    with the tolerance. Made only where a series is past the bound, from the
    tolerance check's hundredfold run where it made one."""
    out = np.zeros(move.shape, dtype=bool)
    over = candidates & np.isfinite(move) & (move > TOLERANCE_MOVE)
    if not over.any():
        return out
    if ctx.tight100 is None:
        try:
            ctx.tight100 = ctx.rerun(**ctx._tighter(100.0))
        except _Out:
            raise
        except Exception:  # noqa: BLE001 - no floor is told apart: the move stands
            return out
    assert ctx.tight is not None
    try:
        d, _ = differences(cols.flat(_block(ctx.tight100)), cols.flat(_block(ctx.tight)))
    except ValueError:
        return out
    return over & np.isfinite(d) & (d >= FLOOR_SHARE * d10)


def _step(cols: _Columns, j: int) -> tuple[float, float, bool]:
    """Column *j*'s value, its quotient's step, and whether it is moved one
    way only (from 0, up)."""
    v = cols.theta[j]
    assert v is not None
    if v != 0.0:
        return v, FD_STEP * abs(v), False
    if j < len(cols.params):
        # A parameter has no scale but its own; at 0, the historical 1.
        return v, FD_STEP, True
    # An initial amount at 0, on the scale of the state it is part of.
    s = max((abs(x) for x in cols.state if math.isfinite(x)), default=0.0)
    return v, FD_STEP * (s if s > 0 else 1.0), True


def check_sensitivity_fd(ctx: _Context) -> dict[str, Any]:
    """The ``"sensitivity_fd"`` check: see the module docstring."""
    name = "sensitivity_fd"
    skip = _skip(ctx, name)
    if skip:
        return skip
    err = ctx._ensure_drift()
    if err:
        return _verdict(
            name, "no_base", note=f"the run at a tenfold tighter tolerance raised ({err})"
        )
    cols = _context(ctx)
    tighter = ctx._tighter(FD_TIGHTER)
    rtol_fd = float(tighter["rtol"])
    atol_fd = np.asarray(ctx.atols, dtype=float) / FD_TIGHTER
    changed: dict[str, Any] = {
        **tighter,
        "step": FD_STEP,
        "sensitivities": None,
    }
    n_sp, n_cols = cols.atol.shape
    # The forward sensitivities' own error: a path's allowance on each series,
    # with its move at the tighter tolerance (and the looser one) as its drift.
    _, keep, d_tight = _drift(ctx, cols)
    peak = (
        np.where(np.isfinite(cols.S), np.abs(cols.S), 0.0).max(axis=0)
        if cols.S.size
        else (np.zeros((n_sp, n_cols)))
    )
    drift = d_tight.reshape(n_sp, n_cols)
    loose = getattr(ctx, "loose", None)
    try:
        ctx._ensure_spread()
        loose = getattr(ctx, "loose", None)
    except _Out:
        raise
    if loose is not None:
        try:
            d_loose, _ = differences(cols.flat(_block(loose)), cols.flat(cols.S))
            drift = spread(drift.reshape(-1), d_loose, peak.reshape(-1)).reshape(n_sp, n_cols)
        except ValueError:
            pass
    own = allowance(
        peak.reshape(-1), cols.atol.reshape(-1), float(ctx.rtol or 0.0), drift.reshape(-1)
    ).reshape(n_sp, n_cols)
    try:
        base = ctx.perturbed(**tighter)
    except _Out:
        return _verdict(name, "timeout", note="the budget ran out before the quotient's base run")
    y0 = np.asarray(base._species, dtype=float)
    if y0.shape != np.asarray(ctx.result._species).shape:
        return _verdict(
            name,
            "exception",
            note=f"the quotient's base run has {y0.shape} values, the run "
            f"{np.asarray(ctx.result._species).shape}",
            changed=changed,
        )
    y_peak = np.where(np.isfinite(y0), np.abs(y0), 0.0).max(axis=0)
    # Each run of the quotient errs by about the run's own drift, scaled to
    # the quotient's tighter tolerance, and by no less than that tolerance.
    held = ctx.spread_abs if ctx.spread_abs is not None else ctx.drift_abs
    assert held is not None
    sp_drift = np.where(np.isfinite(held), held, 0.0)
    err_run = np.maximum(sp_drift / FD_TIGHTER, rtol_fd * y_peak) + atol_fd

    ratio = np.zeros((n_sp, n_cols))
    rel = np.zeros((n_sp, n_cols))
    judged = np.zeros(n_cols, dtype=bool)
    fd_trace = np.zeros((n_sp, n_cols), dtype=bool)
    not_judged: dict[str, str] = {}
    timed_out: list[str] = []
    for j in range(n_cols):
        label = cols.names[j]
        if cols.theta[j] is None or not math.isfinite(float(cols.theta[j])):  # type: ignore[arg-type]
            not_judged[label] = "the record gives it no finite value"
            continue
        is_param = j < len(cols.params)
        if not is_param and label not in cols.state_index:
            not_judged[label] = "not in the run's start state"
            continue
        v, h, one_way = _step(cols, j)
        hi = v + h
        lo = v - h if not one_way else v + 2.0 * h

        def run_at(x: float, j: int = j, is_param: bool = is_param) -> NDArray[np.float64]:
            moved: dict[str, Any] = (
                {"parameters": {cols.names[j]: x}}
                if is_param
                else {"state": {cols.state_index[cols.names[j]]: x}}
            )
            again = ctx.perturbed(**moved, **tighter)
            return np.asarray(again._species, dtype=float)

        try:
            y_hi = run_at(hi)
            y_lo = run_at(lo)
        except _Out:
            timed_out = cols.names[j:]
            break
        except UsageError as e:
            if (getattr(e, "details", None) or {}).get("reason") != "derived_parameter":
                raise
            not_judged[label] = (
                "a derived parameter, whose column is a partial derivative no set_param moves"
            )
            continue
        h_hi, h_lo = hi - v, v - lo  # the steps as realized; h_lo < 0 one way
        if one_way:
            fwd = (y_hi - y0) / h_hi
            fwd2 = (y_lo - y0) / (-h_lo)
            quotient, est = fwd, np.abs(fwd2 - fwd)
            noise = FD_NOISE * 2.0 * err_run / h_hi
        else:
            quotient = (y_hi - y_lo) / (hi - lo)
            est = np.abs((y_hi - y0) / h_hi - (y0 - y_lo) / h_lo)
            noise = FD_NOISE * 2.0 * err_run / (hi - lo)
        s = cols.S[:, :, j]
        fin_s, fin_q = np.isfinite(s), np.isfinite(quotient) & np.isfinite(est)
        differ = np.where(fin_s & fin_q, np.abs(s - quotient) - FD_SPREAD * est, 0.0)
        differ = np.maximum(differ, 0.0).max(axis=0) if differ.size else np.zeros(n_sp)
        bad = (fin_s != fin_q).any(axis=0) if s.size else np.zeros(n_sp, dtype=bool)
        allow = np.maximum(own[:, j], noise)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(
                allow > 0,
                differ / np.where(allow > 0, allow, 1.0),
                np.where(differ > 0, np.inf, 0.0),
            )
            scale_j = np.maximum(peak[:, j], noise)
            rel_j = np.where(scale_j > 0, differ / np.where(scale_j > 0, scale_j, 1.0), 0.0)
        r = np.where(bad, np.inf, r)
        r = np.where(cols.refused[:, j], 0.0, r)
        ratio[:, j] = r
        rel[:, j] = np.where(cols.refused[:, j], 0.0, np.where(bad, np.inf, rel_j))
        fd_trace[:, j] = (noise >= peak[:, j]) & ~cols.refused[:, j]
        judged[j] = True

    flat = ratio.reshape(-1)
    margin = float(flat.max()) if flat.size else 0.0
    n_judged = int(judged.sum())
    live = ~cols.refused & judged[None, :]
    value = float(np.where(live, rel, 0.0).max()) if rel.size else 0.0
    if margin > 1.0:
        status = "diff"
    elif timed_out:
        status = "timeout"
    elif n_judged == 0:
        status = "skip"
    else:
        status = "pass"
    note = (
        f"forward sensitivities against a difference quotient of runs at rtol/{FD_TIGHTER:g} "
        f"with each value moved by {FD_STEP:g} of itself: {value:.2e} of a series' peak at "
        f"most, past the quotient's own error, and {margin:.2g} of the allowance; "
        f"{n_judged} of {n_cols} column(s) judged"
    )
    n_fd_trace = int((fd_trace & live).sum())
    if n_fd_trace:
        note += (
            f"; {n_fd_trace} series within the quotient's noise of zero, which it does not resolve"
        )
    if not_judged:
        first = next(iter(not_judged.items()))
        note += f"; {len(not_judged)} column(s) not judged ({first[0]}: {first[1]})"
    if timed_out:
        note += f"; the budget ran out before {len(timed_out)} column(s), from {timed_out[0]} on"
    n_refused = int((cols.refused & judged[None, :]).sum())
    if n_refused:
        note += f"; {n_refused} series not finite in the run by design, not judged"
    score = np.where(live, ratio, 0.0).reshape(-1)
    entry = _columns_entry(cols, np.where(live, ratio, -1.0).reshape(-1))
    entry["columns"] = [c for c in entry["columns"] if c["name"] not in not_judged]
    entry["columns"] = [c for c in entry["columns"] if c["name"] not in set(timed_out)]
    extra: dict[str, Any] = {
        "margin": _num(margin),
        **entry,
        "n_judged": n_judged,
        "not_judged": dict(list(not_judged.items())[:MAX_NAMED]),
        "timed_out": timed_out[:MAX_NAMED],
        "n_not_resolved": n_fd_trace,
        "not_resolved": _named(cols, fd_trace & live)[:MAX_NAMED],
        "n_refused": n_refused,
    }
    return _verdict(
        name,
        status,
        value if n_judged else None,
        note,
        worst=_worst(cols, score),
        changed=changed,
        extra=extra,
    )
