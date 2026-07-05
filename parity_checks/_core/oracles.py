"""Cross-engine comparison oracles.

Different engines never go byte-identical, so every comparison here is a
numeric tolerance. Each oracle takes already-aligned arrays (the runner is
responsible for matching variables and time points between bngsim and the
reference) and returns a single scalar; the job PASSes iff that scalar is at
or below the manifest oracle's ``tol``.

Two metrics cover the suites:

  max_rel_err   — deterministic (ODE) parity. A combined absolute+relative
                  cell error so a trailing-digit difference on a value decaying
                  toward zero doesn't blow the relative term up to ~2.0.
  mean_zscore   — stochastic (SSA/NF) parity. Both engines are sampled; the
                  statistic is the worst per-(t>0, variable) z-score of the
                  difference of ensemble means, normalized by the combined
                  standard error.
"""

from __future__ import annotations

import numpy as np

# A floor on the denominator of the relative term: below this magnitude a
# value is treated as numerical underflow, so the cell is judged on absolute
# difference alone (prevents sign-flip noise near zero scoring as ~2.0).
DEFAULT_ABS_FLOOR = 1e-9


def max_rel_err(a: np.ndarray, b: np.ndarray, abs_floor: float = DEFAULT_ABS_FLOOR) -> float:
    """Worst combined abs/rel cell error between bngsim `a` and reference `b`.

    Per cell the error is ``|a-b| / max(|b|, abs_floor)`` — i.e. relative to the
    reference where the reference is resolvable, absolute (scaled by abs_floor)
    where it underflows. NaNs in either array make the result NaN (caller treats
    a non-finite metric as DIFF/EXCEPTION).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = np.maximum(np.abs(b), abs_floor)
    return float(np.max(np.abs(a - b) / denom))


def mean_zscore(
    mean_a: np.ndarray,
    sem_a: np.ndarray,
    mean_b: np.ndarray,
    sem_b: np.ndarray,
    sem_floor: float = DEFAULT_ABS_FLOOR,
) -> float:
    """Worst per-cell z-score between two sampled ensembles.

    `mean_*`/`sem_*` are the ensemble mean and standard error (std/sqrt(n)) of
    each engine, aligned cell-for-cell. The z-score is
    ``|mean_a - mean_b| / sqrt(sem_a^2 + sem_b^2)`` with the combined SE floored
    so a cell where both ensembles are exactly constant (SE 0) doesn't divide by
    zero — there it reduces to a floored absolute test.
    """
    mean_a, sem_a = np.asarray(mean_a, float), np.asarray(sem_a, float)
    mean_b, sem_b = np.asarray(mean_b, float), np.asarray(sem_b, float)
    if not (mean_a.shape == sem_a.shape == mean_b.shape == sem_b.shape):
        raise ValueError("mean/sem shape mismatch")
    combined_se = np.sqrt(sem_a**2 + sem_b**2)
    combined_se = np.maximum(combined_se, sem_floor)
    return float(np.max(np.abs(mean_a - mean_b) / combined_se))


def ensemble_stats(replicates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a stack of replicate trajectories to (mean, sem) over replicates.

    `replicates` is shape (n_rep, ...). Returns the per-cell mean and standard
    error of the mean (sample std, ddof=1, / sqrt(n_rep)).
    """
    rep = np.asarray(replicates, float)
    n = rep.shape[0]
    mean = rep.mean(axis=0)
    sem = rep.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


# Registry so the runner can dispatch on a manifest oracle's metric name.
METRICS = {
    "max_rel_err": max_rel_err,
    "mean_zscore": mean_zscore,
}
