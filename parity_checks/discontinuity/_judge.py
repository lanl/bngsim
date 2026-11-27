"""How a run is judged against its reference (issue #1912).

**A point.** Each judged column, a species or a sensitivity column, gets one
number, its error over its tolerance:

    ratio = max_t dist(x(t), band(t)) / (atol_col + rtol·peak)

where ``dist`` is how far the run lies outside the reference's band (a point
reference's band is a point), ``peak`` the column's largest magnitude over the
window, ``atol_col`` the run's atol for a species and ``atol/|p|`` for the
column of parameter p (CVODES's default scaling of a column's tolerance;
bngsim scales each state's column tolerance further, which this rule does not
mirror: it is stated in the run's own atol, not rebuilt from the engine). The
point passes when every species' ratio is at most the case's ``k_state`` and
every column's at most its ``k_sens``.

* A column neither the run nor the reference resolves is not judged, and is
  named: its peak, over the reference and the run both, is within
  ``bngsim._check.TRACE_ATOLS`` atols of zero, where that atol rather than
  rtol holds it. The rule is ``_check.resolved`` itself, so this judge and
  ``Result.check()`` agree on what a run can resolve. A column whose reference
  is zero but whose run is not (a stray jump, a NaN) is resolved by the run's
  own values, and judged.
* An error inside ``FLOOR_FACTOR`` of the reference's own floor is no error.
* On a row taken exactly at a crossing time, the side the run matches better
  is used, and the counts are reported. Which side is owed is #1916's to say.

**Convergence.** For one row, grid and set of columns, take the point ratios
across the rtol axis. A correct run's error tracks its tolerance, so its ratio
stays about level. An error stuck at a crossing does not fall, and its ratio
grows as the tolerance tightens; that is how #1227's columns showed. The row
fails when, for any two of its tolerances, the tighter run's ratio is past
``CONVERGE_FLOOR`` and past the looser one's times
``(rtol_loose/rtol_tight)**CONVERGE_ORDER``: an error that falls slower than
the square root of the tolerance anywhere along the axis, a plateau between
the two tightest included.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from _reference import Band, Solution
from bngsim import _check

FLOOR_FACTOR = 10.0
CONVERGE_FLOOR = 10.0
CONVERGE_ORDER = 0.5


@dataclass
class Verdict:
    outcome: str  # "PASS" or "DIFF"
    note: str
    state_ratio: float = 0.0
    sens_ratio: float | None = None
    worst: str = ""
    sides: dict[str, int] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return max(self.state_ratio, self.sens_ratio or 0.0)


def _col_atol(atol: float, sens: Sequence[str], params: Mapping[str, float]) -> np.ndarray:
    out = []
    for q in sens:
        p = abs(float(params[q]))
        out.append(atol / p if p > 0 else atol)
    return np.asarray(out, dtype=float)


def judge(
    sol: Solution,
    x: np.ndarray,
    S: np.ndarray | None,
    *,
    rtol: float,
    atol: float,
    observe: Sequence[str],
    sens: Sequence[str],
    params: Mapping[str, float],
    k_state: float,
    k_sens: float,
) -> Verdict:
    """One point: the run's species ``x`` (n_t, n_obs) and columns ``S``
    (n_t, n_obs, n_cols) against ``sol``."""
    x = np.asarray(x, dtype=float)
    n_t, n_obs = x.shape
    have_s = bool(sens)
    if have_s and (S is None or sol.sens is None):
        return Verdict("DIFF", "the run or the reference has no sensitivity columns")
    # Per-column scales: atol and peak over both sides.
    sp_atol = np.full(n_obs, float(atol))
    sp_peak = np.maximum(sol.species.magnitude(), sol.species_left.magnitude()).max(axis=0)
    sp_tol = sp_atol + rtol * sp_peak
    if have_s:
        S = np.asarray(S, dtype=float)
        se_atol = np.broadcast_to(_col_atol(atol, sens, params), (n_obs, len(sens)))
        se_peak = np.maximum(sol.sens.magnitude(), sol.sens_left.magnitude()).max(axis=0)
        se_tol = se_atol + rtol * se_peak

    def scaled(side: str) -> tuple[np.ndarray, np.ndarray | None]:
        sp_band = sol.species if side == "right" else sol.species_left
        d = sp_band.distance(x)
        d = np.where(d <= FLOOR_FACTOR * sol.floor * sp_peak, 0.0, d) / sp_tol
        if not have_s:
            return d, None
        se_band = sol.sens if side == "right" else sol.sens_left
        e = se_band.distance(S)
        e = np.where(e <= FLOOR_FACTOR * sol.floor * se_peak, 0.0, e) / se_tol
        return d, e

    dr, er = scaled("right")
    dl, el = scaled("left")
    # Choose a side per row (only rows at a crossing have two).
    sides = {"left": 0, "right": 0, "both": 0}
    d, e = dr.copy(), None if er is None else er.copy()
    for i in np.flatnonzero(sol.at_crossing):
        row_r = max(dr[i].max(), er[i].max() if er is not None else 0.0)
        row_l = max(dl[i].max(), el[i].max() if el is not None else 0.0)
        if row_l == row_r:
            sides["both"] += 1
        elif row_l < row_r:
            sides["left"] += 1
            d[i] = dl[i]
            if e is not None:
                e[i] = el[i]
        else:
            sides["right"] += 1
    sides = {k: v for k, v in sides.items() if v}

    unresolved: list[str] = []
    # Resolved by the reference or by the run: a NaN or an inf in the run
    # makes its peak non-finite, which counts as resolved and fails below.
    sp_ok = _check.resolved(np.maximum(sp_peak, np.abs(x).max(axis=0)), sp_atol, rtol)
    unresolved += [f"{observe[j]}" for j in range(n_obs) if not sp_ok[j]]
    sp_ratio = np.where(sp_ok, d.max(axis=0), 0.0)
    worst_state = float(sp_ratio.max()) if n_obs else 0.0
    worst = ""
    if n_obs and worst_state > 0:
        j = int(np.argmax(sp_ratio))
        i = int(np.argmax(d[:, j]))
        worst = _describe(observe[j], sol.times[i], x[i, j], sol, i, (j,), worst_state)
    sens_ratio = None
    if have_s:
        se_ok = _check.resolved(np.maximum(se_peak, np.abs(S).max(axis=0)), se_atol, rtol)
        unresolved += [
            f"d{observe[j]}/d{q}"
            for j in range(n_obs)
            for k, q in enumerate(sens)
            if not se_ok[j, k]
        ]
        col_ratio = np.where(se_ok, e.max(axis=0), 0.0)
        sens_ratio = float(col_ratio.max())
        if sens_ratio > worst_state:
            j, k = np.unravel_index(int(np.argmax(col_ratio)), col_ratio.shape)
            i = int(np.argmax(e[:, j, k]))
            worst = _describe(
                f"d{observe[j]}/d{sens[k]}", sol.times[i], S[i, j, k], sol, i, (j, k), sens_ratio
            )
    bad = worst_state > k_state or (sens_ratio is not None and sens_ratio > k_sens)
    if not np.isfinite(worst_state) or (sens_ratio is not None and not np.isfinite(sens_ratio)):
        bad = True
    head = f"species {worst_state:.3g}x tol (k {k_state:g})" + (
        "" if sens_ratio is None else f", columns {sens_ratio:.3g}x tol (k {k_sens:g})"
    )
    note = head + (f"; worst {worst}" if worst else "")
    if unresolved:
        note += f"; not resolved: {', '.join(unresolved[:6])}" + (
            " ..." if len(unresolved) > 6 else ""
        )
    return Verdict(
        "DIFF" if bad else "PASS",
        note,
        state_ratio=worst_state,
        sens_ratio=sens_ratio,
        worst=worst,
        sides=sides,
        unresolved=unresolved,
    )


def _describe(
    label: str, t: float, got: float, sol: Solution, i: int, idx: tuple, ratio: float
) -> str:
    band: Band = sol.species if len(idx) == 1 else sol.sens
    lo, hi = float(band.lo[(i, *idx)]), float(band.hi[(i, *idx)])
    want = f"{lo:.10g}" if lo == hi else f"[{lo:.10g}, {hi:.10g}]"
    at = " (at a crossing)" if sol.at_crossing[i] else ""
    return f"{label} at t={float(t):.10g}{at}: {got:.10g} vs {want}, {ratio:.3g}x tol"


@dataclass
class Convergence:
    outcome: str  # "PASS", "DIFF" or "SKIP"
    note: str
    ratios: list[tuple[float, float]]


def converge(points: Sequence[tuple[float, float | None]]) -> Convergence:
    """``(rtol, ratio)`` per point of one group, ratio None where the point
    gave no values. Every pair of tolerances that gave values is judged."""
    have = sorted(((r, q) for r, q in points if q is not None), key=lambda rq: -rq[0])
    if len(have) < 2:
        return Convergence("SKIP", "fewer than two tolerances gave values", have)
    trend = ", ".join(f"{q:.3g}x at {r:g}" for r, q in have)
    for j in range(1, len(have)):
        rj, qj = have[j]
        for ri, qi in have[:j]:
            allowed = max(CONVERGE_FLOOR, qi * (ri / rj) ** CONVERGE_ORDER)
            if not np.isfinite(qj) or qj > allowed:
                return Convergence(
                    "DIFF",
                    f"the error does not fall with rtol ({trend}): {qj:.3g}x at {rj:g}, "
                    f"where {qi:.3g}x at {ri:g} allows {allowed:.3g}x",
                    have,
                )
    return Convergence("PASS", f"{trend}", have)


def judge_invariants(
    invariants, times, species: Mapping[str, np.ndarray], params, *, rtol, atol, k_state
):
    """``NoReference``'s invariants: each residual within ``k_state`` of
    ``atol + rtol·scale``."""
    notes, bad = [], False
    for inv in invariants:
        res, scale = inv.fn(np.asarray(times, float), species, dict(params))
        ratio = float(np.max(np.abs(res))) / (atol + rtol * abs(float(scale)))
        notes.append(f"{inv.name} {ratio:.3g}x tol")
        bad |= not np.isfinite(ratio) or ratio > k_state
    return ("DIFF" if bad else "PASS"), "; ".join(notes)
