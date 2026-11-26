""":meth:`bngsim.Result.check`'s checks of a run's forward sensitivities
(issue #1893).

A run made with ``sensitivity_params=`` or ``sensitivity_ic=`` holds
``dY_i/dθ_j`` for each species ``i`` and each parameter or initial amount
``θ_j``. These checks judge each such *series* (one species, one ``θ``, over
time) on its own scale, as :mod:`bngsim._check` judges each species: a raw
``dY/dθ`` is in the units of ``Y`` over those of ``θ``, so two parameters'
columns share no scale, and a small series beside a large one is judged as
carefully as the large one.

A series' absolute floor is the base one the solver gave it (the issue #177
floor can raise it during the run). CVODES holds a
sensitivity to ``rtol·|s| + atolS``, and bngsim sets
``atolS_ij = atol_i·max(|x_i(0)|, 1)/pbar_j`` (``src/cvode_simulator.cpp``,
issues #196 and #214), with ``atol_i`` the species' own atol as applied and
``pbar_j`` the parameter's magnitude (1 for an initial amount, or for a
parameter at 0). A step solves a whole column at once, so every entry of it
also carries about ``eps·max|s_j|`` (issue #177): the floor is the larger of
the two.

- ``"sensitivity"``: the sensitivities of the tolerance check's runs, at a
  tenth and a hundredth of the ``rtol`` and atol, against the run's. A
  series' *move* is the larger of its largest differences from those two runs,
  past its floor, over its peak; the check fails where a series moves by more
  than :data:`bngsim._check.TOLERANCE_MOVE`. The hundredth sees an error about
  the same at rtol and rtol/10, which moves little between them (issue
  #1967). Where the budget runs out before it the verdict is a timeout; where
  it raises, the move at a tenth is judged alone, and the note says so. A series
  whose peak is within :data:`bngsim._check.TRACE_ATOLS` of its floor, where
  the floor and not ``rtol`` holds it, is not resolved by the run, and is
  named: its move past that many floors is read on the largest peak of its
  column, so its noise near zero is no finding and a departure as large as the
  column's own values is one. A series past the bound whose move from rtol/10
  to rtol/100 is at least :data:`FLOOR_SHARE` of its move at rtol/10, and
  whose move is within :data:`bngsim._check.TRACE_ATOLS` of its floor, is
  taken to be held by a floor (issue #177) and not the tolerance; a larger
  move is the tolerance's, however it shrinks (issue #1962). Such a series is
  named (``n_floor_held``), not failed: its move is read on its column's
  largest peak, and the quotient check allows it ten times that move. An error
  in such a series that is a few of its floors therefore passes both checks,
  whatever its cause (a series between 100 and 1e5 floors high can be held
  with an error the tolerance would shrink); the name is the only sign of it.
  No model is known whose series a floor holds: the tests make one. The floor
  the solver raises a series to during the run, where its right-hand side's
  terms cancel (issue #177), is not known here, so a series held by one more
  than 100 of its base floors up is judged on its own peak.
- ``"sensitivity_fd"``: a central difference quotient. Each ``θ`` is moved by
  :data:`FD_STEP` of its value, and by twice that, both ways (an initial
  amount through the start state, a parameter through
  :meth:`Model.set_param`, so a derived parameter and an initial amount that
  read it follow), and the run made again at a hundredth of its tolerance
  (its rtol clamped to :data:`FD_RTOL_LEAST` to :data:`FD_RTOL_MOST`, never
  looser than the run's, and its atol scaled by the same factor) with no
  sensitivities, so the quotient reads nothing of the sensitivity code: not
  its right-hand side, not its seeds. Its own error is read cell by cell from
  the two steps: their Richardson difference, three times the truncation
  error of the smaller where the trajectory is smooth in ``θ``; and what
  doubling the step leaves of the forward half less the backward, which is
  next to nothing there and about the jump itself within reach of a switch or
  event time ``θ`` moves, where a central quotient is half the jump (issue
  #368). So the check widens there by itself, and needs no list of switch
  times: a cell where that term alone is :data:`FD_RESOLVED` of a series'
  peak is not judged. Where the Richardson term reaches that share away from
  such a cell, the step is too wide for the series (an oscillation whose
  phase the step moves, by more as time goes on), and the series is not
  judged from there until both terms are back under :data:`FD_CLEAR` of the
  share, and is named. On top of that, the quotient may be off by :data:`FD_NOISE`
  times its runs' error over its step, each run erring by about the run's own drift
  scaled to the quotient's tolerance (and by no less than that tolerance), and
  the run's own sensitivities by what a path may differ by
  (:func:`bngsim._check.allowance`, with the series' move at the tighter
  tolerance as its drift; a series not converged at the run's tolerance gets
  none). A series whose allowance is :data:`FD_RESOLVED` of its peak or more
  is not resolved by the quotient, and is named. A ``θ`` at 0 has no step
  size that follows from its value, and is not judged; nor is a derived
  parameter, whose column is a partial derivative.

Neither check judges a row that is not finite in the run by design (an
assignment-rule species whose sensitivity is refused,
``Result.ar_sensitivity_refused``); the verdict counts them. Any other value
the run gives as not finite is a finding. The quotient judges a series on the
larger of its peak in the run and in the quotient, so a column the run gives
as 0 is judged; a column none of whose cells that carry it is clear of a
switch or of a step too wide for it is not judged, and says why.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from bngsim import _check
from bngsim._check import (
    MAX_NAMED,
    TOLERANCE_MOVE,
    TRACE_ATOLS,
    _first_line,
    _num,
    _Out,
    _verdict,
    allowance,
    differences,
    local_moves,
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
#: How much tighter than the run the quotient's runs are made, within
#: :data:`FD_RTOL_LEAST` and :data:`FD_RTOL_MOST`.
FD_TIGHTER = 100.0
#: The loosest rtol a quotient's runs are made at: a hundredth of a loose
#: run's own tolerance still leaves the quotient's noise over its step a large
#: share of the derivative.
FD_RTOL_MOST = 1e-8
#: The tightest: CVODES refuses an rtol near the double's own precision.
FD_RTOL_LEAST = 1e-14
#: How many times the quotient's error estimate (cell by cell, the Richardson
#: difference of its two steps and what doubling the step leaves of the
#: forward half less the backward) a series may differ from it by, on top of
#: the rest.
FD_SPREAD = 2.0
#: A series whose allowance in the quotient check is this share of its peak
#: or more is named as one the quotient does not resolve.
FD_RESOLVED = 0.05
#: A stretch the quotient's step is too wide for ends where its Richardson
#: and kink terms are both back under this share of :data:`FD_RESOLVED`.
FD_CLEAR = 0.01
#: A series past the bound whose move between the hundredfold and the tenfold
#: tighter runs is at least this share of its move at the tenfold one, and
#: whose move is within :data:`bngsim._check.TRACE_ATOLS` of its floor, is held
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
    cols: _Columns, x: NDArray[np.float64], local: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Each series' move from the run's in *x* (as :func:`bngsim._check.moves`
    on its own peak and floor, or for a series the run does not resolve, its
    move past :data:`TRACE_ATOLS` floors on its column's largest peak), which
    series the run resolves, and each series' largest difference. With
    *local*, each series' :func:`bngsim._check.local_peaks`, a resolved
    series' move is read at each point on the larger of its peak and the
    values beside that point (issue #1926)."""
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
    own = (
        moves(d, peak, atol)
        if local is None
        else local_moves(cols.flat(x), cols.flat(cols.S), atol, peak, local)
    )
    move = np.where(keep, own, np.where(np.isinf(d), np.inf, trace))
    # A value the run gives as not finite is a finding, wherever the rerun
    # agrees with it, unless the run refuses the row by design.
    nonfinite = (
        ~np.isfinite(cols.flat(cols.S)).all(axis=0)
        if cols.S.size
        else np.zeros(d.shape, dtype=bool)
    )
    move = np.where(nonfinite, np.inf, move)
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


def _hundredfold(ctx: _Context) -> Result | None:
    """The run at a hundredth of the tolerance, made once, or None where it
    raised (the reason on ``ctx.tight100_error``). Raises
    :class:`bngsim._check._Out` where the budget runs out first."""
    if ctx.tight100 is None and not ctx.tight100_error:
        try:
            ctx.tight100 = ctx.rerun(**ctx._tighter(100.0))
        except _Out:
            raise
        except Exception as e:  # noqa: BLE001 - recorded on the verdict's note
            ctx.tight100_error = f"{type(e).__name__}: {_first_line(e)}"
    return ctx.tight100


def _on_column(cols: _Columns, d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each series' difference *d* past its floor, over its column's largest
    peak (or its own where that is larger)."""
    n_sp, _ = cols.atol.shape
    peak = np.where(np.isfinite(cols.S), np.abs(cols.S), 0.0).max(axis=0)
    col_peak = np.where(cols.refused, 0.0, peak).max(axis=0)
    scale = np.maximum(np.repeat(col_peak[None, :], n_sp, axis=0), peak).reshape(-1)
    over = np.maximum(d - cols.atol.reshape(-1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: NDArray[np.float64] = np.where(
            scale > 0, over / np.where(scale > 0, scale, 1.0), np.where(over > 0, np.inf, 0.0)
        )
    return out


def _sens_local(ctx: _Context, cols: _Columns, hundredfold: bool) -> NDArray[np.float64] | None:
    """Each series' :func:`bngsim._check.local_peaks`, made once, or None (the
    budget out before the grid check's run, which with *hundredfold* makes the
    ``"sensitivity"`` check a timeout; or that run raised)."""
    if ctx.sens_local is None:
        n_t = cols.flat(cols.S).shape
        made = ctx.local_between(lambda r: cols.flat(_block(r)), n_t)
        ctx.sens_local = False if made is None else made
        if made is None and ctx.fine_out and hundredfold:
            ctx.sens_fine_out = True
    if ctx.sens_local is False:
        return None
    return cast("NDArray[np.float64]", ctx.sens_local)


def _tier1(
    ctx: _Context, cols: _Columns, *, hundredfold: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], NDArray[np.bool_]]:
    """Each series' move, the larger of its moves to the tenfold and the
    hundredfold tighter runs (a series a floor holds read on its column's
    largest peak); which series the run resolves; each series' largest
    difference at the tenfold one; and which a floor holds (:func:`_floor_held`).
    Made once (and again for *hundredfold* where it was made without that run).

    With *hundredfold* (the ``"sensitivity"`` check, which judges by it) the
    run at a hundredth is made where it was not; the other tiers read it only
    where a series a floor may hold needs it, made or not. Where it raises,
    the tenfold move is the move (and ``ctx.tight100_error`` says why). Where
    the budget runs out before it, ``ctx.sens_100_out`` marks it for the
    ``"sensitivity"`` check, which is then a timeout; for another tier, where
    a series a floor may hold needs it, :class:`bngsim._check._Out` is raised,
    as no move of that series can be judged."""
    if ctx.sens_drift is not None and not (
        hundredfold
        and ctx.sens_moves[1] is None
        and not ctx.tight100_error
        and not ctx.sens_100_out
    ):
        cached: tuple[
            NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], NDArray[np.bool_]
        ] = ctx.sens_drift
        return cached
    assert ctx.tight is not None
    move10, keep, d10 = _moves(cols, _block(ctx.tight))
    live = ~cols.refused.reshape(-1)
    # A series whose points all fall on or near its zeros has a peak there
    # about its integration error, and moves past the bound on it however right
    # the run is (issue #1926): each of its points is judged on the values the
    # series reaches beside it too, on the grid check's run at GRID_FACTOR times
    # the points. Made before the floor screen, which reads the move.
    plain10 = move10
    past = live & keep & (move10 > TOLERANCE_MOVE)
    local = _sens_local(ctx, cols, hundredfold) if past.any() else None
    if local is not None:
        move10 = np.where(past, _moves(cols, _block(ctx.tight), local)[0], move10)
    # Past the bound at rtol/10, by about a floor: a series a floor may hold.
    maybe_floor = (
        live
        & keep
        & np.isfinite(move10)
        & (move10 > TOLERANCE_MOVE)
        & (d10 <= TRACE_ATOLS * cols.atol.reshape(-1))
    )
    # The other tiers read it only for such a series, whether or not the
    # tolerance check made it, so their verdicts do not depend on which other
    # checks a call asked for.
    tight100: Result | None = None
    if hundredfold or maybe_floor.any():
        try:
            tight100 = _hundredfold(ctx)
        except _Out:
            if not hundredfold:
                raise
            ctx.sens_100_out = True
            tight100 = None
    move100: NDArray[np.float64] | None = None
    d100: NDArray[np.float64] | None = None
    if tight100 is not None:
        try:
            move100, _, d100 = _moves(cols, _block(tight100))
        except ValueError:
            ctx.tight100_error = "its sensitivities do not line up with the run's"
            move100 = d100 = None
    if move100 is not None and tight100 is not None:
        past100 = live & keep & (move100 > TOLERANCE_MOVE)
        if (past100 & ~past).any() and local is None:
            local = _sens_local(ctx, cols, hundredfold)
        past = past | past100
        if local is not None:
            move10 = np.where(past, _moves(cols, _block(ctx.tight), local)[0], plain10)
            move100 = np.where(past, _moves(cols, _block(tight100), local)[0], move100)
    n_between = 0
    ctx.sens_scale = None
    if local is not None:
        later = move10 if move100 is None else np.maximum(move10, move100)
        passed = past & (later <= TOLERANCE_MOVE)
        n_between = int(passed.sum())
        peak = np.where(np.isfinite(cols.flat(cols.S)), np.abs(cols.flat(cols.S)), 0.0).max(axis=0)
        # A series passed this way is held by the quotient to the allowance of
        # the scale it moves on at every point.
        ctx.sens_scale = np.where(passed, np.maximum(peak, local.min(axis=0)), peak)
    ctx.sens_between = n_between
    # An error about the same at rtol and at rtol/10 moves little between
    # them, and is seen at rtol/100 (issue #1967).
    move = move10 if move100 is None else np.maximum(move10, move100)
    d_far = d10 if d100 is None else np.maximum(d10, d100)
    held = _floor_held(ctx, cols, move10, d10, d_far, live & keep, tight100)
    if held.any():
        # Read on its column's largest peak, as a series the run does not
        # resolve is: the arithmetic, not the tolerance, holds it.
        move = np.where(held, _on_column(cols, d_far), move)
        move10 = np.where(held, _on_column(cols, d10), move10)
        if move100 is not None and d100 is not None:
            move100 = np.where(held, _on_column(cols, d100), move100)
    ctx.sens_drift = (move, keep, d10, held)
    ctx.sens_moves = (move10, move100)
    out: tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], NDArray[np.bool_]] = (
        ctx.sens_drift
    )
    return out


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
    try:
        move, keep, _, held = _tier1(ctx, cols, hundredfold=True)
        if ctx.sens_100_out or ctx.sens_fine_out:
            raise _Out
    except _Out:
        missing = (
            "the run at 8x the points, which judges a series past the bound on its peak at "
            "the run's points on the values beside each of them"
            if ctx.sens_fine_out and not ctx.sens_100_out
            else "the run at rtol/100, which sees an error about the same at rtol and rtol/10 "
            "and tells a floor the arithmetic sets from the tolerance"
        )
        return _verdict(
            name,
            "timeout",
            note=f"the budget ran out before {missing}",
            changed=changed,
        )
    live = ~cols.refused.reshape(-1)
    trace = live & ~keep
    value = float(move.max()) if move.size else 0.0
    status = "pass" if value <= TOLERANCE_MOVE else "diff"
    n_series = int(live.sum())
    move10, move100 = ctx.sens_moves
    value_10 = float(move10.max()) if move10.size else 0.0
    value_100 = None if move100 is None else (float(move100.max()) if move100.size else 0.0)
    if value_100 is None:  # the run at rtol/100 raised, or did not line up
        note = (
            f"the sensitivities move by {value:.2e} of a series' scale at rtol/10 and "
            f"atol/10 (bound {TOLERANCE_MOVE:g}), over {n_series} series of "
            f"{len(cols.names)} column(s); the run at rtol/100 raised "
            f"({ctx.tight100_error}), so an error about the same at rtol and rtol/10 is not "
            "seen"
        )
    else:
        note = (
            f"the sensitivities move by {value:.2e} of a series' scale, the larger of their "
            f"moves at rtol/10 ({value_10:.2e}) and at rtol/100 ({value_100:.2e}), atol "
            f"alike (bound {TOLERANCE_MOVE:g}), over {n_series} series of "
            f"{len(cols.names)} column(s)"
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
            f"; {n_held} series moving at least half as far at rtol/100 as at rtol/10, by "
            f"at most {TRACE_ATOLS:g} times their floor, taken to be held by a floor and not "
            "the tolerance, judged on their column's largest peak"
        )
    n_refused = int((~live).sum())
    if n_refused:
        note += f"; {n_refused} series not finite in the run by design, not judged"
    if ctx.sens_between:
        note += (
            f"; {ctx.sens_between} series past the bound on their peak at the run's points, "
            "within it on the values beside each point (the grid check's run)"
        )
    if ctx.sens_local is False and ctx.fine_exc is not None and value > TOLERANCE_MOVE:
        note += (
            f"; the run at 8x the points raised ({type(ctx.fine_exc).__name__}: "
            f"{_first_line(ctx.fine_exc)}), so a series past the bound is judged on its peak "
            "at the run's points alone"
        )
    named = _named(cols, trace | held)
    extra: dict[str, Any] = {
        "bound": TOLERANCE_MOVE,
        "value_10": _num(value_10),
        "value_100": None if value_100 is None else _num(value_100),
        **_columns_entry(cols, move),
        "n_series": n_series,
        "n_not_resolved": n_trace + n_held,
        "not_resolved": named[:MAX_NAMED],
        "n_floor_held": n_held,
        "n_between": ctx.sens_between,
        "n_refused": n_refused,
    }
    return _verdict(
        name, status, value, note, worst=_worst(cols, move), changed=changed, extra=extra
    )


def _floor_held(
    ctx: _Context,
    cols: _Columns,
    move10: NDArray[np.float64],
    d10: NDArray[np.float64],
    d_far: NDArray[np.float64],
    candidates: NDArray[np.bool_],
    tight100: Result | None,
) -> NDArray[np.bool_]:
    """The series past the bound at rtol/10 that a floor holds, not the
    tolerance. A series past the bound only at rtol/100 is judged on its own
    peak: its error is about the same at rtol and at rtol/10 (issue #1967).
    Three runs cannot tell that from a floor's noise whose sign happens to agree
    at the two, and the check takes it for the error; a floor's noise that
    differs at the two is still held. Both of two signs are needed:

    - The move is about the size of a floor: its larger difference, to the
      tenfold or the hundredfold tighter run, is within :data:`TRACE_ATOLS` of
      the floor the solver held the series to (its sensitivity atol, or its
      column's roundoff, ``eps`` times the column's largest entry, where that
      is larger), as a series within that many floors of zero is not resolved.
    - The move does not shrink with the tolerance: between the hundredfold
      and the tenfold tighter runs the series moves by at least
      :data:`FLOOR_SHARE` of its move at the tenfold one. A move the
      tolerance sets shrinks about tenfold from one to the next; the floor the
      solver's arithmetic sets under a sensitivity's error (issue #177) does
      not shrink with the tolerance at all.

    The second sign alone cannot tell a floor from an error that shrinks
    slowly, or not at first and then fast (issue #1962): BIOMD0000000810's
    ``dType_I_T_helper_Cell_T_H1/dgamma_C`` at rtol 1e-7 moves 1.7 times as
    far at rtol/100 as at rtol/10, and is 2.6% off; its move is 4e9 times its
    floor. A floor the solver raises during the run past that many of the
    floor it set (issue #177's) is therefore not told from the tolerance, and
    its move is judged on the series' own peak. Where the hundredfold run
    raised, no floor is told apart, and the move stands."""
    out = np.zeros(move10.shape, dtype=bool)
    over = candidates & np.isfinite(move10) & (move10 > TOLERANCE_MOVE)
    if over.any():
        over &= d_far <= TRACE_ATOLS * cols.atol.reshape(-1)
    if not over.any() or tight100 is None:
        return out
    assert ctx.tight is not None
    try:
        d, _ = differences(cols.flat(_block(tight100)), cols.flat(_block(ctx.tight)))
    except ValueError:
        return out
    return over & np.isfinite(d) & (d >= FLOOR_SHARE * d10)


def _too_wide(start: NDArray[np.bool_], clear: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Per cell, ``(n_times, n_series)``: on from a cell in *start* until the
    next cell in *clear*."""
    out = np.zeros(start.shape, dtype=bool)
    on = np.zeros(start.shape[1:], dtype=bool)
    for k in range(start.shape[0]):
        on = (on & ~clear[k]) | start[k]
        out[k] = on
    return out


def _fd_tolerance(ctx: _Context) -> float:
    """How many times tighter than the run's its quotient's runs are made: a
    hundredfold, to an rtol of no more than :data:`FD_RTOL_MOST` and no less
    than :data:`FD_RTOL_LEAST`."""
    rtol = float(ctx.rtol or 0.0)
    if not rtol > 0:
        return FD_TIGHTER
    return max(1.0, rtol / min(max(rtol / FD_TIGHTER, FD_RTOL_LEAST), FD_RTOL_MOST))


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
    by = _fd_tolerance(ctx)
    tighter = ctx._tighter(by)
    rtol_fd = float(tighter["rtol"])
    atol_fd = np.asarray(ctx.atols, dtype=float) / by
    changed: dict[str, Any] = {**tighter, "step": FD_STEP, "sensitivities": None}
    n_sp, n_cols = cols.atol.shape
    # The forward sensitivities' own error: a path's allowance on each series,
    # with its move at the tighter tolerance (and the looser one) as its drift.
    # A series not converged at the run's tolerance gets no drift, which would
    # forgive its own error (the tolerance check reports it).
    # (A series a floor holds has its move read on its column's peak already.)
    # Convergence is the tenfold move's, as in the AMICI tier, so the verdict
    # does not depend on whether the sensitivity check made the run at rtol/100.
    _, _, d_tight, _ = _tier1(ctx, cols)
    move10, _ = ctx.sens_moves
    converged = (move10 <= TOLERANCE_MOVE).reshape(n_sp, n_cols)
    peak = (
        np.where(np.isfinite(cols.S), np.abs(cols.S), 0.0).max(axis=0)
        if cols.S.size
        else np.zeros((n_sp, n_cols))
    )
    if ctx.sens_scale is not None and ctx.sens_scale.size == peak.size:
        peak = np.maximum(peak, ctx.sens_scale.reshape(n_sp, n_cols))
    drift = d_tight.reshape(n_sp, n_cols)
    ctx._ensure_spread()
    if ctx.loose is not None:
        try:
            d_loose, _ = differences(cols.flat(_block(ctx.loose)), cols.flat(cols.S))
        except ValueError:
            pass
        else:
            drift = spread(drift.reshape(-1), d_loose, peak.reshape(-1)).reshape(n_sp, n_cols)
    drift = np.where(converged, drift, 0.0)
    own = allowance(
        peak.reshape(-1), cols.atol.reshape(-1), float(ctx.rtol or 0.0), drift.reshape(-1)
    ).reshape(n_sp, n_cols)
    base = ctx.perturbed(**tighter)
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
    sp = ctx.spread_abs if ctx.spread_abs is not None else ctx.drift_abs
    assert sp is not None
    err_run = np.maximum(np.where(np.isfinite(sp), sp, 0.0) / by, rtol_fd * y_peak) + atol_fd

    ratio = np.zeros((n_sp, n_cols))
    rel = np.zeros((n_sp, n_cols))
    share = np.zeros((n_sp, n_cols))
    judged = np.zeros(n_cols, dtype=bool)
    fd_trace = np.zeros((n_sp, n_cols), dtype=bool)
    not_judged: dict[str, str] = {}
    timed_out: list[str] = []
    for j in range(n_cols):
        label = cols.names[j]
        v = cols.theta[j]
        if v is None or not math.isfinite(v):
            not_judged[label] = "the record gives it no finite value"
            continue
        if v == 0.0:
            not_judged[label] = "its value is 0, so no step size follows from it"
            continue
        is_param = j < len(cols.params)
        if not is_param and label not in cols.state_index:
            not_judged[label] = "not in the run's start state"
            continue
        h = FD_STEP * abs(v)
        at = (v + h, v - h, v + 2.0 * h, v - 2.0 * h)

        def run_at(x: float, j: int = j, is_param: bool = is_param) -> NDArray[np.float64]:
            moved: dict[str, Any] = (
                {"parameters": {cols.names[j]: x}}
                if is_param
                else {"state": {cols.state_index[cols.names[j]]: x}}
            )
            again = ctx.perturbed(**moved, **tighter)
            return np.asarray(again._species, dtype=float)

        try:
            p1 = run_at(at[0])
            m1 = run_at(at[1])
            p2 = run_at(at[2])
            m2 = run_at(at[3])
        except _Out:
            timed_out = [n for n in cols.names[j:] if n not in not_judged]
            break
        except UsageError as e:
            if (getattr(e, "details", None) or {}).get("reason") != "derived_parameter":
                raise
            not_judged[label] = (
                "a derived parameter, whose column is a partial derivative no set_param moves"
            )
            continue
        except Exception as e:
            # A quotient run past the run's own timeout (with no budget, or one
            # longer than it): this column is not judged, and the others still
            # are. Known by its code, not its message.
            if "timeout" not in str(getattr(e, "code", "") or ""):
                raise
            not_judged[label] = f"its quotient's runs passed the run's timeout ({_first_line(e)})"
            continue
        # The steps as realized, on each side.
        a1, b1, a2, b2 = at[0] - v, v - at[1], at[2] - v, v - at[3]
        q1 = (p1 - m1) / (a1 + b1)
        q2 = (p2 - m2) / (a2 + b2)
        # Richardson: where the trajectory is smooth in the value, q1 errs by
        # about a third of how far it is from q2.
        rich = np.abs(q2 - q1)
        # A step's forward half less its backward half is about h·d²Y/dθ²
        # where the trajectory is smooth, so twice the step doubles it. Within
        # reach of a switch or event time the value moves, it is about the
        # jump at either step, and what doubling leaves of it marks the kink.
        d1 = (p1 - y0) / a1 - (y0 - m1) / b1
        d2 = (p2 - y0) / a2 - (y0 - m2) / b2
        kink = np.abs(d2 - d1 * ((a2 + b2) / (a1 + b1)))
        noise = FD_NOISE * 2.0 * err_run / (a1 + b1)
        s = cols.S[:, :, j]
        fin_s = np.isfinite(s)
        fin_q = np.isfinite(q1) & np.isfinite(rich) & np.isfinite(kink)
        bad = (fin_s != fin_q).any(axis=0) if s.size else np.zeros(n_sp, dtype=bool)
        # A series' scale is the larger of the run's and the quotient's peak,
        # so a column the run gives as 0 where the quotient does not is judged
        # on the quotient's scale, not masked by its own.
        # (Not at a cell its kink term dominates: on an event time the value
        # moves, the quotient of the jump is the jump over 2h, no derivative.)
        smooth_q = fin_q & (FD_SPREAD * kink < FD_RESOLVED * np.abs(q1))
        q_peak = np.where(smooth_q, np.abs(q1), 0.0).max(axis=0) if s.size else np.zeros(n_sp)
        pk = np.maximum(peak[:, j], q_peak)
        cut = FD_RESOLVED * pk[None, :]
        # Within reach of a switch or event time the value moves: that cell.
        # (A series 0 in the run and the quotient alike has nothing to mask.)
        near = (FD_SPREAD * kink >= cut) & (cut > 0)
        # Once the Richardson difference reaches FD_RESOLVED of a series'
        # peak away from a switch, the step is too wide for it, and stays so
        # until both terms are back under FD_CLEAR of that share: as they are
        # just past a switch, where the kink term crosses zero inside the
        # window, and never are in an oscillation whose phase the step moves,
        # where one term or the other is near its largest at every phase.
        wide = _too_wide(
            (FD_SPREAD * rich >= cut) & ~near & (cut > 0),
            FD_SPREAD * (rich + kink) < FD_CLEAR * cut,
        )
        use = fin_s & fin_q & ~near & ~wide
        base_allow = np.maximum(own[:, j], noise)
        allow_t = base_allow[None, :] + FD_SPREAD * (rich + kink)
        gap = np.where(use, np.abs(s - q1), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_t = np.where(
                allow_t > 0,
                gap / np.where(allow_t > 0, allow_t, 1.0),
                np.where(gap > 0, np.inf, 0.0),
            )
            past = np.maximum(gap - FD_SPREAD * (rich + kink), 0.0)
            scale_j = np.maximum(pk, base_allow)[None, :]
            rel_t = np.where(scale_j > 0, past / np.where(scale_j > 0, scale_j, 1.0), 0.0)
            pk_t = np.where(pk > 0, pk, 1.0)[None, :]
            share_t = np.where(pk[None, :] > 0, allow_t / pk_t, np.inf)
        r = np.where(use, r_t, 0.0).max(axis=0) if s.size else np.zeros(n_sp)
        r = np.where(bad, np.inf, r)
        r = np.where(cols.refused[:, j], 0.0, r)
        ratio[:, j] = r
        rel_j = np.where(use, rel_t, 0.0).max(axis=0) if s.size else np.zeros(n_sp)
        rel[:, j] = np.where(cols.refused[:, j], 0.0, np.where(bad, np.inf, rel_j))
        share[:, j] = np.where(use, share_t, 0.0).max(axis=0) if s.size else np.zeros(n_sp)
        wide_any = wide.any(axis=0) if s.size else np.zeros(n_sp, dtype=bool)
        # A series none of whose cells that carry it (a value of FD_RESOLVED of
        # its scale, in the run or the quotient) is judged was not compared.
        carries = np.maximum(np.where(fin_s, np.abs(s), 0.0), np.where(fin_q, np.abs(q1), 0.0))
        compared = (
            (use & ((carries >= cut) | (cut == 0))).any(axis=0)
            if s.size
            else np.zeros(n_sp, dtype=bool)
        )
        unresolved = (base_allow >= FD_RESOLVED * pk) | wide_any | ~compared
        fd_trace[:, j] = unresolved & ~cols.refused[:, j]
        if not (compared & ~cols.refused[:, j]).any() and not bad.any():
            not_judged[label] = (
                "no cell that carries it is clear of a switch or event time it moves, or of "
                "a step too wide for it"
            )
            ratio[:, j] = 0.0
            rel[:, j] = 0.0
            continue
        judged[j] = True

    flat = ratio.reshape(-1)
    margin = float(flat.max()) if flat.size else 0.0
    n_judged = int(judged.sum())
    live = ~cols.refused & judged[None, :]
    value = float(np.where(live, rel, 0.0).max()) if rel.size else 0.0
    resolved_fd = live & ~fd_trace
    accuracy = float(np.where(resolved_fd, share, 0.0).max()) if resolved_fd.any() else None
    failing = live & (ratio > 1.0)
    # A diff whose every failing series the tolerance check finds unconverged
    # is the tolerance's, not the sensitivity code's.
    unconverged_only = bool(failing.any() and not (failing & converged).any())
    if margin > 1.0:
        status = "diff"
    elif timed_out:
        status = "timeout"
    elif n_judged == 0:
        status = "skip"
    else:
        status = "pass"
    if n_judged:
        note = (
            f"forward sensitivities against central difference quotients, each value moved by "
            f"{FD_STEP:g} and {2 * FD_STEP:g} of itself, of runs at rtol {rtol_fd:.3g}: "
            f"{value:.2e} of a series' peak at most, past the quotient's own error, and "
            f"{margin:.2g} of the allowance; {n_judged} of {n_cols} column(s) judged"
        )
    else:
        note = f"no column judged by a difference quotient, of {n_cols}"
    if accuracy is not None:
        note += f", to within {accuracy:.1g} of a series' peak"
    n_out = int((live & ~converged).sum())
    if n_out:
        note += (
            f"; {n_out} series not converged at the run's tolerance, judged on the tolerance alone"
        )
    n_fd_trace = int((fd_trace & live).sum())
    if n_fd_trace:
        note += (
            f"; {n_fd_trace} series the quotient resolves to no better than "
            f"{FD_RESOLVED:g} of the series' peak, everywhere or over a stretch of time it "
            "does not judge"
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
    gone = set(not_judged) | set(timed_out)
    entry["columns"] = [c for c in entry["columns"] if c["name"] not in gone]
    extra: dict[str, Any] = {
        "margin": _num(margin),
        **entry,
        "n_judged": n_judged,
        "not_judged": dict(list(not_judged.items())[:MAX_NAMED]),
        "timed_out": timed_out[:MAX_NAMED],
        "allowance": None if accuracy is None else _num(accuracy),
        "n_not_converged": n_out,
        "diff_not_converged": unconverged_only,
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
