"""
_netbench.py
------------
Shared machinery for the ``.net``-based benchmark suites that compare
BNGsim's in-process engine against BioNetGen's ``run_network`` subprocess
(``suites/ssa/``, ``suites/psa/``).

Every such suite runs two gates per model:

* **correctness** -- an ensemble-mean *z*-test.  Each engine simulates an
  ensemble of replicate trajectories; the two ensemble means are compared
  with a two-sample *z*-statistic at every ``(time, species)`` cell.  The
  model passes when the largest ``|z|`` stays under a tolerance.  Both
  engines simulate the *same* ``.net``, so species columns line up; the
  replicates are independent draws (a different RNG per engine), which is
  exactly the two-sample setting the test assumes.

* **timing** -- a warmup + timed-run wall-clock comparison, median
  reported.  Per the suite design rule, timing is only meaningful for a
  model that *passed* correctness, so a runner skips timing otherwise.

This module holds the primitives; each suite runner owns its model
registry, effort tiers, and report layout.  ``run_network`` / ``BNG2.pl``
are located via ``BNGPATH`` / ``RUN_NETWORK`` (see ``benchmarks/README.md``).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------

BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
RUN_NETWORK = os.environ.get("RUN_NETWORK", os.path.join(BNGPATH, "bin", "run_network"))

RN_TIMEOUT = 600  # seconds -- generous for large stochastic models

# Default ensemble size for the correctness gate, and the warmup/timed-run
# counts for the timing gate.  Runners expose these as CLI overrides.
DEFAULT_REPLICATES = 20
DEFAULT_WARMUP = 2
DEFAULT_RUNS = 5

# Largest |z| tolerated by the exact-vs-exact correctness gate.  The
# statistic is a max over every (time, species) cell, so the tolerance must
# clear the extreme-value spread of that many draws: for the biggest model
# here (~3700 species x ~100 times) the H0 expectation of max|z| is ~5.1,
# so 6.0 leaves headroom without masking a genuine discrepancy.
DEFAULT_Z_TOL = 6.0

# Largest standardized mean difference tolerated by the
# approximation-vs-approximation gate.  d is measured in units of the
# process standard deviation; the gate reduces the per-cell d array by its
# 99.9th percentile (see _gate_verdict), so this tolerance reads as
# "99.9% of populated cells agree within 6 process-sigma".  Two correct but
# different PSA scalings sit inside that envelope; a gross modelling error
# shifts essentially every cell and drives the percentile far past it.
DEFAULT_D_TOL = 6.0

# Denominator floor for both gates.  A near-constant species has ~zero
# spread, so the standard error / process sigma it contributes is ~zero;
# dividing the mean gap by that bare number would turn output-format
# floating-point dust into a spurious failure.  The floor is the larger of
# a tiny absolute term and a relative term scaled to the cell magnitude:
# the relative term imposes a 0.1%-of-magnitude deadband on effectively
# deterministic species (it binds only when the coefficient of variation
# is below ~0.1%), while leaving genuinely stochastic cells untouched.
_ABS_FLOOR = 1e-9
_REL_FLOOR = 1e-3

# Cells whose species mean is below this count in *both* engines are
# excluded from the gating maximum.  A rare species (mean << 1) is
# dominated by sampling and method noise -- it carries no meaningful
# correctness signal -- so it is reported in the detail string but does
# not drive the pass/fail verdict.
DEFAULT_COUNT_FLOOR = 1.0


# ---------------------------------------------------------------------------
# Machine info
# ---------------------------------------------------------------------------


def machine_info() -> dict:
    """Collect machine + tool versions for benchmark reproducibility."""
    import bngsim

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "bngsim_version": getattr(bngsim, "__version__", "unknown"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
        if proc.returncode == 0:
            info["git_commit"] = proc.stdout.strip()
    except Exception:
        pass
    try:
        proc = subprocess.run([RUN_NETWORK], capture_output=True, text=True, timeout=5)
        info["run_network_version"] = (proc.stdout + proc.stderr).split("\n")[0].strip()
    except Exception:
        info["run_network_version"] = "unknown"
    return info


# ---------------------------------------------------------------------------
# Single-run engines
# ---------------------------------------------------------------------------


def run_bngsim(net_path, method, t_end, n_steps, seed, poplevel=0):
    """One BNGsim stochastic run.

    ``method`` is ``"ssa"`` or ``"psa"``.  Model loading + clone + reset are
    excluded from ``wall_time`` (they are amortized by PyBNF), so the timing
    covers ``sim.run()`` only -- matching ``run_network``'s simulate phase.

    Returns a dict with ``species`` (ndarray, n_times x n_species),
    ``wall_time``, ``steps``, and ``error`` (None on success).
    """
    import bngsim

    try:
        model = bngsim.Model.from_net(str(net_path))
        m = model.clone()
        m.reset()
        if method == "psa":
            sim = bngsim.Simulator(m, method="psa", poplevel=poplevel)
        else:
            sim = bngsim.Simulator(m, method="ssa")
    except Exception as e:  # noqa: BLE001
        return {"species": None, "wall_time": -1.0, "steps": -1, "error": f"load: {e}"}

    try:
        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
        elapsed = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return {"species": None, "wall_time": -1.0, "steps": -1, "error": f"run: {e}"}

    return {
        "species": np.asarray(result.species, dtype=float),
        "wall_time": elapsed,
        "steps": int(result.solver_stats.get("n_steps", 0)),
        "error": None,
    }


def run_run_network(net_path, method, t_end, n_steps, seed, poplevel=0):
    """One ``run_network`` stochastic run.

    ``wall_time`` includes the full subprocess overhead (spawn, ``.net``
    parse, file I/O) -- the real per-simulation cost PyBNF pays.  PSA is
    requested as ``-p ssa --poplevel Nc`` (run_network's heterogeneous
    adaptive scaling); its ``.cdat`` is written back at full population
    scale, so it is directly comparable to the SSA ``.cdat``.

    Returns the same dict shape as :func:`run_bngsim`.
    """
    net_path = str(net_path)
    sample_time = t_end / n_steps

    with tempfile.TemporaryDirectory(prefix="netbench_rn_") as tmpdir:
        prefix = os.path.join(tmpdir, "out")
        cmd = [RUN_NETWORK, "-p", "ssa"]
        if method == "psa" and poplevel > 0:
            cmd += ["--poplevel", str(int(poplevel))]
        cmd += [
            "-h",
            str(seed),
            "-g",
            net_path,
            "-o",
            prefix,
            net_path,
            f"{sample_time:.15g}",
            str(n_steps),
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {
                "species": None,
                "wall_time": float(RN_TIMEOUT),
                "steps": -1,
                "error": "timeout",
            }
        elapsed = time.perf_counter() - t0

        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            return {"species": None, "wall_time": elapsed, "steps": -1, "error": output[:400]}

        steps = -1
        m = re.search(r"TOTAL STEPS:\s*(\d+)", output)
        if m:
            steps = int(m.group(1))

        cdat = prefix + ".cdat"
        if not os.path.exists(cdat):
            return {"species": None, "wall_time": elapsed, "steps": steps, "error": "no .cdat"}
        try:
            data = np.loadtxt(cdat, comments="#")
            species = np.atleast_2d(data)[:, 1:]  # column 0 is time
        except Exception as e:  # noqa: BLE001
            return {"species": None, "wall_time": elapsed, "steps": steps, "error": f"cdat: {e}"}

        return {"species": species, "wall_time": elapsed, "steps": steps, "error": None}


# ---------------------------------------------------------------------------
# Correctness gates -- ensemble comparison statistics
# ---------------------------------------------------------------------------
#
# Two engines simulating the same stochastic model produce two ensembles of
# replicate trajectories.  How to compare them depends on whether the
# engines run the *same* algorithm:
#
# * Two exact-SSA engines must agree in distribution -- the only difference
#   is sampling noise -- so the right test is the two-sample *z*-test
#   (:func:`zscore_gate`).  It divides the mean gap by the standard error
#   of the mean, so its power grows with the replicate count: with enough
#   replicates it resolves an arbitrarily small genuine bias.
#
# * Two *approximation* engines (e.g. BNGsim partial-scaling PSA vs
#   run_network heterogeneous-adaptive-scaling PSA) carry a real, non-zero
#   method bias by construction.  A z-test would eventually flag that bias
#   as a failure once the replicate count is large enough -- it is the
#   wrong tool.  :func:`effect_size_gate` instead divides the mean gap by
#   the *process* standard deviation (a standardized mean difference, akin
#   to Cohen's d).  That denominator does not shrink with the replicate
#   count, so the statistic is replicate-count-stable: a small method bias
#   stays a small effect size, while a gross modelling error -- the thing a
#   correctness gate must catch -- still blows up.


def _ensemble_stats(bng_list, rn_list):
    """Per-cell means / variances / magnitude scale for two ensembles.

    Returns ``(mean_b, mean_r, var_b, var_r, scale, nb, nr)`` or raises
    ``ValueError`` on an empty or mismatched ensemble.  ``scale`` is the
    per-cell ``max(|mean_b|, |mean_r|)`` used both for the relative
    denominator floor and the rare-species gating mask.
    """
    if not bng_list or not rn_list:
        raise ValueError("empty ensemble")
    bng = np.stack(bng_list)  # (Nb, n_t, n_sp)
    rn = np.stack(rn_list)  # (Nr, n_t, n_sp)
    if bng.shape[1:] != rn.shape[1:]:
        raise ValueError(f"shape mismatch: BNGsim {bng.shape[1:]} vs RN {rn.shape[1:]}")
    nb, nr = bng.shape[0], rn.shape[0]
    mean_b, mean_r = bng.mean(axis=0), rn.mean(axis=0)
    var_b = bng.var(axis=0, ddof=1) if nb > 1 else np.zeros_like(mean_b)
    var_r = rn.var(axis=0, ddof=1) if nr > 1 else np.zeros_like(mean_r)
    scale = np.maximum(np.abs(mean_b), np.abs(mean_r))
    return mean_b, mean_r, var_b, var_r, scale, nb, nr


def _gate_verdict(stat, mean_b, mean_r, scale, tol, count_floor, label, reduce):
    """Reduce a per-cell statistic array to a ``(passed, value, detail)`` verdict.

    Only cells whose species is populated (``scale >= count_floor`` in at
    least one engine) drive the verdict -- rare species are sampling- and
    method-noise dominated and carry no correctness signal; they fall back
    in only if no cell qualifies.  ``reduce`` selects how the gated cells
    collapse to one number:

    * ``"max"`` -- the worst single cell.  Appropriate for the exact-vs-
      exact *z*-test, where any cell breaching the extreme-value threshold
      is a genuine statistical anomaly.
    * ``"p999"`` -- the 99.9th percentile.  Appropriate for the
      approximation-vs-approximation effect-size gate: the single worst
      cell out of tens of thousands is extreme-value noise, while a
      whole-species discrepancy still spans far more than 0.1% of cells.

    The detail string always also names the worst single cell, for triage.
    """
    mask = scale >= count_floor
    gated = mask if mask.any() else np.ones(mask.shape, dtype=bool)
    gated_vals = stat[gated]

    if reduce == "p999":
        value = float(np.percentile(gated_vals, 99.9))
    else:
        value = float(np.max(gated_vals))

    worst = np.where(gated, stat, -np.inf)
    idx = np.unravel_index(int(np.argmax(worst)), stat.shape)
    detail = (
        f"{label}: gated={value:.2f} ({'p99.9' if reduce == 'p999' else 'max'}), "
        f"worst cell {label}={float(stat[idx]):.2f} at t_idx={idx[0]}, sp_idx={idx[1]} "
        f"(BNGsim mean={mean_b[idx]:.4g}, RN mean={mean_r[idx]:.4g}); "
        f"{int(mask.sum())}/{mask.size} cells gated"
    )
    return value <= tol, value, detail


def zscore_gate(bng_list, rn_list, z_tol=DEFAULT_Z_TOL, count_floor=DEFAULT_COUNT_FLOOR):
    """Two-sample ensemble-mean *z*-test on species trajectories.

    ``bng_list`` / ``rn_list`` are lists of per-replicate species arrays
    (each n_times x n_species).  For every ``(time, species)`` cell the
    statistic is ``|mean_bng - mean_rn| / sqrt(var_bng/Nb + var_rn/Nr)``,
    the standard-error denominator floored relative to the cell magnitude.
    The verdict is the worst-cell ``max|z|``.  Returns
    ``(passed, max_z, detail)``.  Use this only when both engines run the
    *same* algorithm (exact-vs-exact).
    """
    try:
        mean_b, mean_r, var_b, var_r, scale, nb, nr = _ensemble_stats(bng_list, rn_list)
    except ValueError as e:
        return False, float("inf"), str(e)

    se = np.sqrt(var_b / nb + var_r / nr)
    se = np.maximum(se, np.maximum(_ABS_FLOOR, _REL_FLOOR * scale))
    z = np.abs(mean_b - mean_r) / se
    return _gate_verdict(z, mean_b, mean_r, scale, z_tol, count_floor, "|z|", reduce="max")


def effect_size_gate(bng_list, rn_list, d_tol=DEFAULT_D_TOL, count_floor=DEFAULT_COUNT_FLOOR):
    """Standardized-mean-difference gate on species trajectories.

    For every ``(time, species)`` cell the statistic is the mean gap
    divided by the pooled *process* standard deviation,
    ``|mean_bng - mean_rn| / sqrt((var_bng + var_rn) / 2)``.  Unlike the
    *z*-test this does not shrink with the replicate count, so it tolerates
    the bounded method bias between two approximation engines while still
    catching a gross discrepancy.  The verdict is the 99.9th percentile of
    the per-cell d over populated cells -- robust to the extreme-value
    noise of a single worst cell.  Returns ``(passed, d999, detail)``.
    """
    try:
        mean_b, mean_r, var_b, var_r, scale, nb, nr = _ensemble_stats(bng_list, rn_list)
    except ValueError as e:
        return False, float("inf"), str(e)

    pooled_sd = np.sqrt((var_b + var_r) / 2.0)
    pooled_sd = np.maximum(pooled_sd, np.maximum(_ABS_FLOOR, _REL_FLOOR * scale))
    d = np.abs(mean_b - mean_r) / pooled_sd
    return _gate_verdict(d, mean_b, mean_r, scale, d_tol, count_floor, "d", reduce="p999")


_GATES = {"zscore": zscore_gate, "effect_size": effect_size_gate}


def correctness_gate(
    net_path,
    method,
    t_end,
    n_steps,
    *,
    replicates=DEFAULT_REPLICATES,
    poplevel=0,
    statistic="zscore",
    tol=None,
    seed_base=2000,
):
    """Run the ensemble correctness gate for one model.

    Simulates ``replicates`` trajectories with each engine (same seed
    sequence, independent RNGs) and applies the chosen comparison
    ``statistic`` -- ``"zscore"`` (exact-vs-exact) or ``"effect_size"``
    (approximation-vs-approximation).  ``tol`` defaults to the statistic's
    module default.  Returns a dict: ``passed``, ``statistic``, ``stat``
    (the max value), ``detail``, ``replicates``, ``error``.
    """
    if statistic not in _GATES:
        raise ValueError(f"statistic must be one of {sorted(_GATES)}, got {statistic!r}")
    if tol is None:
        tol = DEFAULT_Z_TOL if statistic == "zscore" else DEFAULT_D_TOL

    bng_list, rn_list = [], []
    for i in range(replicates):
        seed = seed_base + i
        rb = run_bngsim(net_path, method, t_end, n_steps, seed, poplevel)
        if rb["error"]:
            return _gate_error(statistic, replicates, f"BNGsim: {rb['error']}")
        rr = run_run_network(net_path, method, t_end, n_steps, seed, poplevel)
        if rr["error"]:
            return _gate_error(statistic, replicates, f"run_network: {rr['error']}")
        bng_list.append(rb["species"])
        rn_list.append(rr["species"])

    passed, stat, detail = _GATES[statistic](bng_list, rn_list, tol)
    return {
        "passed": passed,
        "statistic": statistic,
        "stat": stat,
        "detail": detail,
        "replicates": replicates,
        "error": None,
    }


def _gate_error(statistic, replicates, msg):
    return {
        "passed": False,
        "statistic": statistic,
        "stat": None,
        "detail": "",
        "replicates": replicates,
        "error": msg,
    }


# ---------------------------------------------------------------------------
# Timing gate -- warmup + timed-run median
# ---------------------------------------------------------------------------


def timing_compare(
    net_path,
    method,
    t_end,
    n_steps,
    *,
    warmup=DEFAULT_WARMUP,
    runs=DEFAULT_RUNS,
    poplevel=0,
    seed_base=1000,
):
    """Median wall-clock comparison of the two engines for one model.

    Each engine runs ``warmup`` discarded passes then ``runs`` timed passes;
    the medians and their ratio (``speedup``, RN/BNGsim) are returned.
    """
    result = {"method": method, "poplevel": poplevel}

    bng_times, bng_steps = [], []
    for i in range(warmup + runs):
        r = run_bngsim(net_path, method, t_end, n_steps, seed_base + i, poplevel)
        if r["error"]:
            result["bngsim"] = {"median_wall_time": -1.0, "error": r["error"]}
            result["run_network"] = {"median_wall_time": -1.0}
            result["speedup"] = None
            return result
        if i >= warmup:
            bng_times.append(r["wall_time"])
            bng_steps.append(r["steps"])

    rn_times, rn_steps = [], []
    for i in range(warmup + runs):
        r = run_run_network(net_path, method, t_end, n_steps, seed_base + i, poplevel)
        if i >= warmup:
            rn_times.append(r["wall_time"])
            rn_steps.append(r["steps"])

    bng_med = median(bng_times)
    rn_med = median(rn_times) if rn_times else -1.0
    bng_st = int(median(bng_steps)) if bng_steps else -1
    rn_st = int(median(rn_steps)) if rn_steps and all(s >= 0 for s in rn_steps) else -1

    result["bngsim"] = {
        "median_wall_time": bng_med,
        "median_steps": bng_st,
        "all_times": bng_times,
    }
    result["run_network"] = {
        "median_wall_time": rn_med,
        "median_steps": rn_st,
        "all_times": rn_times,
    }
    result["speedup"] = (rn_med / bng_med) if (rn_med > 0 and bng_med > 0) else None
    return result


def geometric_mean(values):
    """Geometric mean of a list of positive numbers (empty -> None)."""
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    return float(np.exp(np.mean(np.log(vals))))
