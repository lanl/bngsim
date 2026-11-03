"""bngsim solver work counters recorded beside each parity verdict (GH #702).

Wall-clock time on a shared CI runner is too noisy to catch a modest efficiency
regression, but the solver's work is not: for a fixed model, build and platform,
CVODE's step, RHS-evaluation and Jacobian-evaluation counts are deterministic. A
change that makes bngsim do more work for the same answer (a Jacobian falling
back to finite differences, a step-size control change, an event forcing
restarts) shows up here with no timing noise at all. The nightly check
(``parity_checks/nightly/verdicts.py``) compares these per model against a
baseline.
"""

from __future__ import annotations

# Kept in lockstep with parity_checks/nightly/verdicts.py WORK_KEYS (a test pins it).
WORK_KEYS = (
    "n_steps",
    "n_rhs_evals",
    "n_jac_evals",
    "n_jac_rhs_evals",
    "n_nonlin_iters",
    "n_err_test_fails",
    "n_nonlin_conv_fails",
)


def work_counters(stats) -> dict[str, int]:
    """The work counters present in a ``Result.solver_stats`` mapping, as ints."""
    stats = stats or {}
    return {k: int(stats[k]) for k in WORK_KEYS if isinstance(stats.get(k), (int, float))}
