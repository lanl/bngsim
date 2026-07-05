"""
problems.py
-----------
Shared registry of fitting-benchmark problems for the bngsim paper.

The benchmark is multi-model: each problem is one fitting job run two
ways -- ``conf/subprocess.conf`` (``bngl_backend=bionetgen``, the legacy
BNG2.pl / run_network / NFsim stack) versus ``conf/bngsim.conf``
(``bngl_backend=bngsim``, in-process). One table row per problem.

``run_jobs.py``, ``consistency_check.py`` and ``../emit.py`` all
import this list so the benchmark is defined in exactly one place.

Problems are drawn from the PyBNF iScience-2019 Data S1 set (mmc3/);
``family`` records which simulator the problem natively uses:

  * ``ode`` -- BioNetGen ODE.  Deterministic; subprocess and in-process
    trajectories must agree to tight numerical tolerance.
  * ``ssa`` -- BioNetGen SSA.  Stochastic; the two backends use different
    RNGs, so they agree only in distribution.
  * ``nf``  -- network-free.  Subprocess runs the standalone NFsim app;
    in-process runs BNGsim RuleMonkey (a different exact stochastic
    engine -- in-process NFsim is blocked, issue #44).

ODE rows are listed by ascending free-parameter count so ``run_jobs.py``
exercises the cheap models first and the table reads small-to-large.

Effort tiers
------------
Each problem carries an ``effort`` tier -- ``low`` / ``medium`` / ``high``
-- so the benchmark can be run as a cheap subset rather than the full
multi-hour sweep.  The tier is the *combined* subprocess + in-process
cost of the row, measured by ``harness/calibrate.py`` (an iters=3 short
run) and recorded in ``results/_calib_runs.json``.  ``run_jobs.py`` and
``consistency_check.py`` take ``--effort {low,medium,high}`` with
*cumulative* semantics: ``low`` runs only low-tier rows, ``medium`` runs
low + medium, ``high`` runs everything (the default).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent

# Effort tiers in ascending cost order. ``--effort X`` runs every tier at
# or below X (cumulative), so this tuple's order defines the semantics.
EFFORT_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class Problem:
    slug: str  # benchmark directory name, e.g. "prob05_threestep"
    label: str  # table row label, e.g. "P5 three-step cascade"
    n_params: int  # free parameters fit
    family: str  # "ode" | "ssa" | "nf"
    effort: str  # "low" | "medium" | "high" -- see "Effort tiers" above

    @property
    def dir(self) -> Path:
        return BENCH_DIR / self.slug

    @property
    def subprocess_conf(self) -> Path:
        return self.dir / "conf" / "subprocess.conf"

    @property
    def bngsim_conf(self) -> Path:
        return self.dir / "conf" / "bngsim.conf"


# ---------------------------------------------------------------------------
# The benchmark is ODE-only. The BioNetGen-ODE problems of Table 2 are
# 2, 5, 6, 7, 10, 13, 15, 18, 19, 20, 24, 31; this is the subset that
# runs clean on *both* PyBNF backends (subprocess and in-process bngsim).
#
# The other candidates were dropped after smoke testing -- they hit
# PyBNF / bngsim-bridge integration bugs, not benchmark-design issues:
#   * prob19/20 (RAFi) -- bngsim crashes on parameter_scan + steady_state
#                         (internal #45).
#   * prob15/06        -- bngsim bridge rejects interleaved
#                         parameter_scan / setConcentration action blocks
#                         (internal #46).
#   * prob13           -- PyBNF v1.3.0 find_t_length parser fails on the
#                         model, affects both backends (lanl/PyBNF #390).
# SSA and network-free probe rows were also dropped -- ODE-only benchmark.
#
# Ordered by ascending free-parameter count: run_jobs.py exercises the
# cheap models first and the table reads small-to-large.
#
# The ``effort`` tier of each row is assigned from the iters=3 calibration
# in results/_calib_runs.json (combined subprocess + in-process wall clock).
# ---------------------------------------------------------------------------
PROBLEMS = [
    # effort tier <- combined subprocess+bngsim wall clock from the iters=3
    # calibration (results/_calib_runs.json):
    #   prob05  15.6 s  | prob07  31.7 s            -> low
    #   prob18 116.9 s  | prob24 142.9 s            -> medium
    #   kozer_egfr -- 913-species network, ~8 min network generation per
    #                 job; not calibrated, tiered high by fiat.
    Problem("prob05_threestep", "P5 three-step cascade", 3, "ode", "low"),
    Problem("kozer_egfr", "P10 Kozer EGFR", 9, "ode", "high"),
    Problem("prob07_egg", "P7 egg-shaped curve", 10, "ode", "low"),
    Problem("prob24_jnk", "P24 Jnk cascade", 12, "ode", "medium"),
    Problem("prob18_mapk", "P18 MAPK scaffold", 13, "ode", "medium"),
]


def problems_at_effort(effort: str):
    """Return the problems at or below a cumulative effort threshold.

    ``low`` -> only low-tier problems; ``medium`` -> low + medium;
    ``high`` -> every problem.  Order within ``PROBLEMS`` is preserved.
    """
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")
    cutoff = EFFORT_LEVELS.index(effort)
    return [p for p in PROBLEMS if EFFORT_LEVELS.index(p.effort) <= cutoff]
