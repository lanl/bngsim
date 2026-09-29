"""Central-difference oracle for forward-sensitivity columns (issue #788).

A forward-sensitivity column is checked here against a central difference of
*plain* runs: runs with no sensitivities, which never enter the event jump, the
switch jump, the clock branch or the comoving frame. That is what makes it an
oracle for those terms rather than a second copy of them.

A central difference is only a reference when its own error is known, and issue
#368 found both of the ways a careless one misleads
(``test_time_switch_sens_fd_reference.py`` pins them on a closed-form model):

* **A step below the solver's floor.** At a fixed ``h = 1e-6·|p|`` the signal
  ``x(p+h) − x(p−h)`` can be smaller than the error of either run, and the
  quotient is then that error divided by ``2h``. So the step is not fixed: every
  step of a geometric ladder is taken, the per-run noise is *measured* (the same
  run at the working tolerance and at a 10x looser one), and each cell keeps the
  step whose error bound — truncation plus ``noise/h`` — is smallest.
* **A sample on a crossing.** Where ``x`` has a kink in ``p`` (a sample time
  equal to a switch or event time that ``p`` moves), a central difference is the
  average of the two one-sided derivatives at every step, so it converges,
  cleanly, to the wrong number (half the jump). Convergence cannot catch that;
  comparing the Richardson-extrapolated forward and backward quotients can, and
  such a cell is reported as a kink and refused rather than compared.

The analytic side of the comparison gets the resolution floor the AMICI parity
suite already uses for the same question — ``atol/|p|``, CVODES' own absolute
tolerance on ``s = ∂x/∂p`` — imported from
``parity_checks/amici_parity/_amici_sens.py`` rather than restated, so the two
cannot come to disagree about what a solver can resolve. Nothing else there fits:
``bn_sens`` loads SBML and times warm repeats, and ``sens_verdict`` is a
corpus-scale relative verdict tuned against AMICI, where a per-term test wants a
per-cell bound with a known derivation.

Every run must start from the model's initial state. A bngsim run continues from
wherever the last one left the Model, ODE as well as SSA, so an ``observe``
callback builds a fresh Model per call; reusing one makes every difference
accumulate and read as a huge sensitivity error.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
_AMICI_PARITY = _ROOT / "parity_checks" / "amici_parity"
if str(_AMICI_PARITY) not in sys.path:
    sys.path.insert(0, str(_AMICI_PARITY))

from _amici_sens import sens_resolution_floors  # noqa: E402

#: Relative steps, largest first. Halving keeps neighbouring quotients close
#: enough for their difference to estimate the truncation error of the smaller
#: one, and the range spans the whole trade-off on the models here: at 0.1 the
#: ``h²`` term dominates, and by ~1e-5 the ``noise/h`` term does at rtol 1e-12.
DEFAULT_REL_STEPS: tuple[float, ...] = tuple(0.1 * 0.5**k for k in range(14))

#: How much looser the noise probe runs than the working tolerance. The error of
#: the tight run is bounded by the difference between the two — the loose run's
#: own error — which overstates it by about this factor. That errs toward a
#: larger step, never toward reading noise as signal.
#:
#: The estimate assumes the error shrinks as the tolerance does. A run whose
#: error does not (both probes wrong by the same amount) would understate the
#: bound. The failure that produces is a reported mismatch, not a silent pass, but
#: under a strict xfail it could keep the case failing after its defect is fixed.
#: So the neighbouring-step gaps are part of every bound, and a mismatch is worth
#: re-checking at another tolerance before it is filed.
NOISE_PROBE_FACTOR = 10.0

#: The most an analytic column is ever allowed to be off, relative to itself,
#: whatever tolerance it ran at. ``100·rtol`` alone would pass a column twice the
#: true value at rtol 1e-2.
MAX_ANALYTIC_RELATIVE = 1e-2


class ReferenceRefused(Exception):
    """The difference cannot judge the column, so no verdict is returned.

    Deliberately not an ``AssertionError``: a strict ``xfail`` for a known defect
    names ``raises=AssertionError``, and a reference that could not judge must
    not satisfy it.
    """


@dataclass(frozen=True)
class CentralDifference:
    """A per-cell central difference and what it can be trusted to.

    Every array has the shape ``observe`` returned.

    ``value``  the central quotient at the step chosen for that cell.
    ``err``    a bound on ``|value − ∂x/∂p|``: the gap to both neighbouring steps'
               quotients (a truncation estimate, conservative by ~3x for an
               ``h²`` error) plus the measured noise divided by the step.
    ``step``   the absolute step each cell's value was read at.
    ``kink``   ``x`` is not differentiable in ``p`` there: the extrapolated
               forward and backward quotients disagree by far more than ``err``.
               A central difference there is the average of the two sides.
    ``noise``  the measured per-run error of one plain run at ``rtol``.
    ``typical`` the sensitivity an output would have if it moved in proportion
               to the parameter, ``max|x| / scale``: the yardstick for a column
               whose true value is 0, which has no peak of its own to be judged
               against.
    """

    value: np.ndarray
    err: np.ndarray
    step: np.ndarray
    kink: np.ndarray
    noise: np.ndarray
    typical: float


def central_difference(
    observe: Callable[[float, float], np.ndarray],
    p0: float,
    *,
    rtol: float = 1e-12,
    scale: float | None = None,
    rel_steps: Sequence[float] = DEFAULT_REL_STEPS,
) -> CentralDifference:
    """Central difference of ``observe`` in its parameter, around ``p0``.

    ``observe(value, rtol)`` runs a *plain* simulation (no sensitivities) from a
    fresh model with the parameter set to ``value``, at relative tolerance
    ``rtol``, and returns the outputs as an array — any shape, the same every
    call. Its absolute tolerance is the caller's to scale with ``rtol``.

    ``scale`` is what the relative steps multiply: ``|p0|`` by default, and it
    must be given when ``p0`` is 0 or when ``|p0|`` is no guide to how far the
    parameter can move (a switch time 1e-8 from a sample, say).
    """
    if scale is None:
        if p0 == 0.0:
            raise ValueError("p0 is 0, so a relative step has no scale; pass scale=")
        scale = abs(p0)
    if len(rel_steps) < 3:
        raise ValueError("at least three steps are needed to bound each quotient")
    if any(b >= a or b <= 0.0 for a, b in zip(rel_steps, rel_steps[1:], strict=False)):
        raise ValueError("rel_steps must be positive and strictly decreasing")

    x0 = np.asarray(observe(p0, rtol), dtype=float)
    x_loose = np.asarray(observe(p0, rtol * NOISE_PROBE_FACTOR), dtype=float)
    # Never below one ulp-scale unit of the solution itself, so a cell where the
    # two runs happen to agree to the last digit still carries a floor.
    noise = np.maximum(np.abs(x_loose - x0), 4.0 * np.finfo(float).eps * np.abs(x0))

    hs = np.asarray(rel_steps, dtype=float) * scale
    central, forward_r, backward_r = [], [], []
    prev = None
    for h in hs:
        xp = np.asarray(observe(p0 + h, rtol), dtype=float)
        xm = np.asarray(observe(p0 - h, rtol), dtype=float)
        central.append((xp - xm) / (2.0 * h))
        fwd, bwd = (xp - x0) / h, (x0 - xm) / h
        # One-sided quotients carry an O(h) error; one Richardson step against
        # the previous, larger step H removes it, leaving each side's own
        # derivative to O(h·H): (H·q(h) − h·q(H)) / (H − h), which is 2q(h) − q(2h)
        # on the default halving ladder. Where x is smooth the two sides then
        # agree to the central quotient's accuracy; at a kink they converge to
        # different limits.
        if prev is None:
            forward_r.append(fwd)
            backward_r.append(bwd)
        else:
            H, prev_fwd, prev_bwd = prev
            forward_r.append((H * fwd - h * prev_fwd) / (H - h))
            backward_r.append((H * bwd - h * prev_bwd) / (H - h))
        prev = (h, fwd, bwd)

    D = np.stack(central)  # (n_steps, *shape)
    n = len(hs)
    shape = x0.shape
    best_val = np.zeros(shape)
    best_err = np.full(shape, np.inf)
    best_step = np.zeros(shape)
    best_k = np.zeros(shape, dtype=int)
    # Interior steps only: each needs a neighbour on both sides for its bound.
    for k in range(1, n - 1):
        trunc = np.maximum(np.abs(D[k - 1] - D[k]), np.abs(D[k] - D[k + 1]))
        err = trunc + noise / hs[k]
        better = err < best_err
        best_val = np.where(better, D[k], best_val)
        best_err = np.where(better, err, best_err)
        best_step = np.where(better, hs[k], best_step)
        best_k = np.where(better, k, best_k)

    F = np.stack(forward_r)
    B = np.stack(backward_r)
    idx = best_k[np.newaxis, ...]
    f_at = np.take_along_axis(F, idx, axis=0)[0]
    b_at = np.take_along_axis(B, idx, axis=0)[0]
    # The one-sided quotients each involve x0 once more than the central one,
    # so their noise is a few times noise/h; 12x covers the extrapolation's
    # 2x - 1x combination of two such quotients on each side.
    kink = np.abs(f_at - b_at) > 10.0 * best_err + 12.0 * noise / best_step
    typical = float(np.max(np.abs(x0))) / scale
    return CentralDifference(best_val, best_err, best_step, kink, noise, typical)


def analytic_tolerance(S: np.ndarray, p0: float, *, rtol: float, atol: float) -> np.ndarray:
    """What an analytic column run at ``(rtol, atol)`` can be held to, per cell.

    ``100·rtol·|S|`` for the relative part — local error control does not bound
    global error, and 100x is the headroom the parity suite's noise mask uses for
    the same reason — capped at :data:`MAX_ANALYTIC_RELATIVE`, plus
    ``sens_resolution_floors`` for the absolute part, CVODES' ``atol/|p|`` on
    ``s = ∂x/∂p`` at that same factor.
    """
    floor = sens_resolution_floors([p0], atol)[0]
    return min(100.0 * rtol, MAX_ANALYTIC_RELATIVE) * np.abs(S) + floor


def mismatches(
    S: np.ndarray,
    fd: CentralDifference,
    p0: float,
    *,
    rtol: float,
    atol: float,
    resolution: float = 1e-5,
    labels: Callable[[tuple], str] | None = None,
) -> list[str]:
    """Cells where the analytic column ``S`` disagrees with the difference.

    Refuses — raises :class:`ReferenceRefused`, rather than returning a verdict —
    when the reference cannot judge: a cell sits on a kink (the #368 half-value
    trap), or the difference's own bound is wider than ``resolution`` times the
    yardstick below. Either way a
    comparison would pass or fail for reasons unrelated to the analytic column,
    and a vacuous pass is the outcome this oracle exists to rule out.

    The yardstick is the column's own peak, or the difference's ``typical`` scale
    when that is larger: a column that is truly 0 is judged against what a
    proportional response would be, so an analytic column that should be 0 and
    is not still fails.
    """
    S = np.asarray(S, dtype=float)
    if S.shape != fd.value.shape:
        raise ValueError(f"shape {S.shape} does not match the difference's {fd.value.shape}")
    label = labels or (lambda cell: str(tuple(int(i) for i in cell)))

    kinks = [label(c) for c in zip(*np.nonzero(fd.kink), strict=True)]
    if kinks:
        raise ReferenceRefused(
            "the finite-difference reference sits on a kink in the parameter at "
            f"{kinks}: a central difference there is the mean of the one-sided "
            "derivatives, not the derivative (issue #368). Move the sample off the "
            "crossing."
        )
    yardstick = max(float(np.max(np.abs(fd.value))), fd.typical)
    coarse = fd.err > resolution * yardstick
    if np.any(coarse):
        worst = np.unravel_index(np.argmax(np.where(coarse, fd.err, 0.0)), fd.err.shape)
        raise ReferenceRefused(
            f"the finite-difference reference is too coarse to judge: at {label(worst)} "
            f"its bound is {fd.err[worst]:.3e} against a yardstick of {yardstick:.3e}"
        )

    bound = fd.err + analytic_tolerance(S, p0, rtol=rtol, atol=atol)
    bad = np.abs(S - fd.value) > bound
    return [
        f"{label(c)}: analytic {S[c]:.10g}, finite difference {fd.value[c]:.10g} "
        f"(± {fd.err[c]:.2e}, step {fd.step[c]:.2e})"
        for c in zip(*np.nonzero(bad), strict=True)
    ]
