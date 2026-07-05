#!/usr/bin/env python3
"""KINSOL steady-state dose-response benchmark (Table S8).

Compares the wall-clock cost of computing dose-response curves under five
distinct steady-state strategies:

  A. ``run_network`` long-time integration (BNG2.pl subprocess).
  B. BNGsim CVODE long-time integration (in-process).
  C. BNGsim KINSOL — Newton-first with simulation fallback. Try strict
     Newton from cold start; on failure charge the column with the
     fallback CVODE-to-t_end* time as well. This is the user-stated
     intent for what the "KINSOL" column should report and is the same
     policy that BNGsim's ``method="auto"`` implements internally
     (see "Two-tier semantics" below).
  D. BNGsim KINSOL — integration burst then Newton refinement. Always
     do a brief CVODE burst first, then hand off to strict Newton from
     the burst's final state. Wall time includes both phases. The
     burst uses BNGsim's adaptive exponential time-stepping schedule
     (see "Two-tier semantics") with a relaxed convergence tolerance
     so it terminates as soon as the trajectory is "close enough" for
     Newton to take over.

Per-dose reference quantities:

  1. y_ss target. Computed with strict Newton when it converges,
     otherwise from the long-horizon CVODE tail. The target value
     drives the settling-time analysis but does NOT appear in any
     timed wall-clock measurement.

  2. Settling time t_s = the smallest t* such that for every species i
     and every t >= t*, |y_i(t) - y_ss[i]| <= eps_rel * |y_ss[i]| +
     eps_abs (strict "enter and never leave" definition). A dose is
     censored if the dense reference trajectory never enters and
     stays inside the band by the characterization horizon T_char.

  3. All long-time engines (A, B, and the fallback paths of C and D)
     integrate to ``t_end = t_s``. This is the theoretical minimum
     horizon the long-integration approach needs to land inside the
     settling band; a real modeler would over-integrate for safety,
     so the reported wall times are a lower bound on the long-time
     cost. Doses where t_s could not be determined within T_char
     (= 10 × t_end_hint) are censored and excluded from the totals.

  4. The model-level summary reports the mean of t_s over non-censored
     doses. Per-dose t_s appears in the JSON output under
     ``models[i].doses[j].t_s``.

Two-tier semantics (BNGsim ``method="auto"``):

    BNGsim's ``find_steady_state`` (src/steady_state.cpp:626) implements
    ``method="auto"`` as Newton-first with integration fallback:
    cold-start KINSOL Newton is attempted first; on failure, control
    passes to ``solve_by_integration``, which integrates with adaptive
    exponential time-stepping (t = 10, 100, 1000, ... up to ``max_time``,
    default 1e6 model-time units) and terminates the moment the residual
    ``||f(y)||`` falls below ``tol`` (default 1e-9). Strategy C above is
    a manually-assembled equivalent of this auto path, except that the
    fallback uses CVODE-to-t_end* instead of BNGsim's residual-based
    early termination so that all "long integration" columns share an
    apples-to-apples horizon. Strategy D is the *opposite* ordering
    (integration first, Newton second) and has to be assembled manually
    because BNGsim does not expose it as a method= choice.

Usage:
    python bench_kinsol_dose_response.py
    python bench_kinsol_dose_response.py --quick 5
    python bench_kinsol_dose_response.py --model RAFi

Output:
    results/bench_kinsol_dose_response.json
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    BENCHMARKS_DIR,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    RUN_NETWORK,
    add_bngsim_timeout_arg,
    format_bngsim_timeout,
    get_machine_info,
    save_results,
)

# ── Settling-band parameters ──────────────────────────────────────────────
# 99.73% band corresponds to a relative tolerance of 1 - 0.9973 = 2.7e-3.
# Absolute floor protects species whose y_ss is at or near zero from
# being knocked out of the band by ordinary integrator noise.
EPS_REL = 1.0 - 0.9973
EPS_ABS = 1e-9

# ── Model definitions ─────────────────────────────────────────────────────
# t_end_hint is a per-model guess at the settling time. The dense
# reference trajectory is integrated to T_char = 10 × t_end_hint to
# safely capture overshoots; a dose is censored if no settling is found
# within T_char. The wall-time integration horizon for the engines is
# t_s itself (the per-dose measured settling time), not t_end_hint.

MODELS = [
    {
        "name": "RAFi_ground",
        "net_file": "ode/RAFi_ground.net",
        "scan_param": "Ifree",
        "par_min": 1e-4,
        "par_max": 1e2,
        "n_scan_pts": 20,
        "log_scale": True,
        "t_end_hint": 1e5,
    },
    {
        "name": "Scaff_22_ground",
        "net_file": "ode/Scaff_22_ground.net",
        "scan_param": "S",
        "par_min": 1e-3,
        "par_max": 1.0,
        "n_scan_pts": 20,
        "log_scale": True,
        "t_end_hint": 1e3,
    },
    {
        "name": "mwc",
        "net_file": "ode/mwc.net",
        "scan_param": "log_P_ox",
        "par_min": 0.0,
        "par_max": 1.0,
        "n_scan_pts": 20,
        "log_scale": False,
        "t_end_hint": 10.0,
    },
    {
        "name": "wofsy_goldstein",
        "net_file": "ode/wofsy_goldstein.net",
        "scan_param": "logConcLig",
        "par_min": -10.0,
        "par_max": -6.0,
        "n_scan_pts": 20,
        "log_scale": False,
        "t_end_hint": 100.0,
    },
]

# T_char (characterization horizon for trajectory analysis) is set to
# T_CHAR_FACTOR × t_end_hint per model. If t_s is not found within
# T_char the dose is censored.
T_CHAR_FACTOR = 10.0

# Dense sampling for the reference trajectory used to compute t_s.
N_DENSE_POINTS = 4001


# ── Helpers ───────────────────────────────────────────────────────────────


def make_scan_values(par_min, par_max, n_pts, log_scale):
    if log_scale:
        return np.logspace(math.log10(par_min), math.log10(par_max), n_pts)
    return np.linspace(par_min, par_max, n_pts)


def settling_time(t: np.ndarray, y: np.ndarray, y_ss: np.ndarray) -> float | None:
    """Return t_s = first time after which every species stays in the band.

    Strict "enter and never leave" definition: t_s is the largest time at
    which any species exits the band, plus an epsilon to land just inside.
    Returns None if the trajectory ends with at least one species outside
    the band (i.e. settling has not yet occurred by t[-1]).

    Parameters
    ----------
    t : (n_t,) model-time array, monotonically increasing.
    y : (n_t, n_sp) species concentrations along ``t``.
    y_ss : (n_sp,) target steady-state concentrations.
    """
    band = EPS_REL * np.abs(y_ss) + EPS_ABS  # (n_sp,)
    outside = np.abs(y - y_ss[None, :]) > band[None, :]  # (n_t, n_sp)
    # Final sample must be inside the band, else not settled.
    if outside[-1, :].any():
        return None
    any_out = outside.any(axis=1)  # (n_t,)
    if not any_out.any():
        return float(t[0])
    last_out_idx = int(np.where(any_out)[0][-1])
    if last_out_idx + 1 >= t.size:
        return None
    # Linearly interpolate between last "outside" sample and the next
    # (inside) sample so t_s is reported on the dense grid scale rather
    # than snapping to the next sampled point.
    return float(0.5 * (t[last_out_idx] + t[last_out_idx + 1]))


# ── Engine: run_network ───────────────────────────────────────────────────
#
# The previous version of this script tried to pass ``--par name=val`` to
# run_network and silently dropped it after run_network rejected the flag
# (run_network has no parameter-override CLI option at all). The fix is
# to rewrite the parameters block in a temporary .net file before invoking
# run_network. Derived ConstantExpressions are re-evaluated by run_network
# at load time, so overriding a primary parameter propagates correctly.

_PARAM_LINE_RE = re.compile(r"^(\s*\d+\s+)([A-Za-z_]\w*)(\s+)(\S.*?)(\s*(?:#.*)?)$")


def _override_net_param(net_text: str, param_name: str, param_val: float) -> str:
    """Return the .net file text with ``param_name`` set to ``param_val``.

    Matches the indexed parameters block ``begin parameters ... end parameters``;
    only the line whose name field equals ``param_name`` is modified.
    """
    lines = net_text.splitlines(keepends=True)
    in_params = False
    found = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("begin parameters"):
            in_params = True
            continue
        if stripped.startswith("end parameters"):
            in_params = False
            continue
        if not in_params:
            continue
        m = _PARAM_LINE_RE.match(line.rstrip("\n"))
        if not m:
            continue
        if m.group(2) != param_name:
            continue
        prefix, name, sep, _value, suffix = m.groups()
        # Preserve the trailing newline that splitlines(keepends=True) kept.
        nl = "\n" if line.endswith("\n") else ""
        lines[i] = f"{prefix}{name}{sep}{param_val:.15g}{suffix}{nl}"
        found = True
        break
    if not found:
        raise KeyError(f"parameter {param_name!r} not found in parameters block")
    return "".join(lines)


def run_rn_to_horizon(
    net_path: str, param_name: str, param_val: float, t_end: float, n_steps: int
) -> dict:
    """Run BNG2.pl run_network to t_end with the dose parameter overridden.

    The override is applied by writing a modified copy of the .net file
    (run_network has no per-invocation parameter-override CLI option).
    """
    sample_time = t_end / max(n_steps, 1)
    with tempfile.TemporaryDirectory(prefix="rn_ss_") as tmpdir:
        try:
            with open(net_path) as f:
                net_text = f.read()
            patched = _override_net_param(net_text, param_name, param_val)
        except Exception as e:
            return {"wall_time": -1.0, "error": f"net rewrite: {str(e)[:280]}"}
        patched_path = os.path.join(tmpdir, "patched.net")
        with open(patched_path, "w") as f:
            f.write(patched)
        prefix = os.path.join(tmpdir, "out")
        cmd = [
            RUN_NETWORK,
            "-o",
            prefix,
            "-g",
            patched_path,
            patched_path,
            f"{sample_time:.15g}",
            str(n_steps),
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return {"wall_time": -1.0, "error": "timeout"}
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            return {
                "wall_time": elapsed,
                "error": (proc.stdout + proc.stderr)[:300],
            }
        cdat = prefix + ".cdat"
        if not os.path.exists(cdat):
            return {"wall_time": elapsed, "error": "no .cdat output"}
        try:
            data = np.loadtxt(cdat, comments="#")
            return {"wall_time": elapsed, "final_conc": data[-1, 1:]}
        except Exception as e:
            return {"wall_time": elapsed, "error": str(e)[:300]}


# ── Engine: BNGsim CVODE long-time integration ────────────────────────────


def _bngsim_imported():
    import bngsim  # local import so --help works without the extension built

    return bngsim


def _make_sim(net_path: str, param_name: str, param_val: float):
    bngsim = _bngsim_imported()
    model = bngsim.Model.from_net(net_path)
    model.set_param(param_name, param_val)
    return bngsim.Simulator(model, method="ode"), model


def run_bngsim_cvode_to_horizon(
    net_path: str,
    param_name: str,
    param_val: float,
    t_end: float,
    n_points: int,
    bngsim_timeout: float | None = None,
) -> dict:
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
        t0 = time.perf_counter()
        result = sim.run(t_span=(0.0, float(t_end)), n_points=n_points, timeout=bngsim_timeout)
        elapsed = time.perf_counter() - t0
        species = np.asarray(result.species)
        return {
            "wall_time": elapsed,
            "final_conc": species[-1, :],
        }
    except Exception as e:
        return {"wall_time": -1.0, "error": str(e)[:300]}


def run_bngsim_dense_trajectory(
    net_path: str,
    param_name: str,
    param_val: float,
    t_end: float,
    n_points: int,
    bngsim_timeout: float | None = None,
) -> dict:
    """Untimed; used for trajectory analysis (and as fallback target)."""
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
        result = sim.run(t_span=(0.0, float(t_end)), n_points=n_points, timeout=bngsim_timeout)
        return {
            "time": np.asarray(result.time, dtype=float),
            "species": np.asarray(result.species, dtype=float),
        }
    except Exception as e:
        return {"error": str(e)[:300]}


# ── Engine: BNGsim KINSOL (strict Newton) ────────────────────────────────


def run_bngsim_kinsol_strict(net_path: str, param_name: str, param_val: float) -> dict:
    """Strict Newton attempt only. Returns y_ss on success, error on failure.

    The wall_time field is the elapsed time of the Newton attempt itself
    regardless of outcome — fallback bookkeeping is the caller's
    responsibility.
    """
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
        t0 = time.perf_counter()
        try:
            res = sim.steady_state(method="newton")
            elapsed = time.perf_counter() - t0
            converged = bool(getattr(res, "converged", True))
            species = np.asarray(
                res.species[-1, :] if hasattr(res, "species") else res, dtype=float
            )
            if not converged:
                return {"wall_time": elapsed, "error": "newton did not converge"}
            return {"wall_time": elapsed, "final_conc": species}
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return {"wall_time": elapsed, "error": str(e)[:300]}
    except Exception as e:  # construction failure is not "wall time of KINSOL"
        return {"wall_time": -1.0, "error": f"setup: {str(e)[:280]}"}


# ── Engine: BNGsim KINSOL (integration burst, then Newton refinement) ────
#
# Tier 1 uses ``method="integration"`` with a relaxed tolerance, which
# leverages BNGsim's adaptive exponential time-stepping schedule
# (t = 10, 100, 1000, ... up to ``BURST_MAX_TIME``) and terminates as
# soon as ``||f(y)|| < BURST_TOL``. The model state is left at the
# burst's final concentrations, so the subsequent ``method="newton"``
# call (Tier 2) starts from a warm guess instead of cold initial
# conditions.

BURST_TOL = 1e-3  # relaxed convergence tolerance for the integration burst
BURST_MAX_TIME = 1e6  # passes through to ``solve_by_integration`` max_time


def run_bngsim_kinsol_two_tier(net_path: str, param_name: str, param_val: float) -> dict:
    """Integration burst followed by strict Newton refinement.

    Both phases are timed and summed. On Newton refinement failure, returns
    an error so the caller can decide whether to charge the column with a
    further fallback (``benchmark_dose`` does so to keep the policy honest).
    """
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
    except Exception as e:
        return {"wall_time": -1.0, "error": f"setup: {str(e)[:280]}"}

    burst_t = 0.0
    refine_t = 0.0
    try:
        t0 = time.perf_counter()
        burst = sim.steady_state(method="integration", tol=BURST_TOL, max_time=BURST_MAX_TIME)
        burst_t = time.perf_counter() - t0
        if not bool(getattr(burst, "converged", True)):
            return {
                "wall_time": burst_t,
                "burst": burst_t,
                "refine": 0.0,
                "error": "integration burst did not reach BURST_TOL",
            }
        t0 = time.perf_counter()
        refined = sim.steady_state(method="newton")
        refine_t = time.perf_counter() - t0
        if not bool(getattr(refined, "converged", True)):
            return {
                "wall_time": burst_t + refine_t,
                "burst": burst_t,
                "refine": refine_t,
                "error": "newton refinement did not converge",
            }
        species = np.asarray(
            refined.species[-1, :] if hasattr(refined, "species") else refined,
            dtype=float,
        )
        return {
            "wall_time": burst_t + refine_t,
            "burst": burst_t,
            "refine": refine_t,
            "final_conc": species,
        }
    except Exception as e:
        return {
            "wall_time": burst_t + refine_t,
            "burst": burst_t,
            "refine": refine_t,
            "error": str(e)[:300],
        }


# ── Timing wrapper ────────────────────────────────────────────────────────


def timed_median(
    fn, *, warmup: int = DEFAULT_WARMUP, runs: int = DEFAULT_RUNS
) -> tuple[float, dict | None]:
    """Run fn (warmup + runs) times; median wall time of the timed runs.

    fn must return a dict with a ``wall_time`` field. If any timed run
    surfaces an error, return (-1, last_result).
    """
    times: list[float] = []
    last: dict | None = None
    for i in range(warmup + runs):
        result = fn()
        if i >= warmup:
            if "error" in result:
                return -1.0, result
            times.append(float(result["wall_time"]))
            last = result
    if not times:
        return -1.0, last
    return median(times), last


# ── Per-dose orchestration ────────────────────────────────────────────────


@dataclass
class DoseRecord:
    param_value: float
    t_s: float | None = None
    censored: bool = False
    censor_reason: str = ""
    kinsol_failed: bool = False  # strict Newton (column C) failed at this dose

    rn_time: float = -1.0
    cvode_time: float = -1.0
    # Column C: Newton-first with simulation fallback (matches BNGsim auto).
    kinsol_first_time: float = -1.0
    kinsol_first_strict_time: float = -1.0
    kinsol_first_fallback_time: float = -1.0
    # Column D: integration burst, then Newton refinement.
    kinsol_two_tier_time: float = -1.0
    kinsol_two_tier_burst_time: float = -1.0
    kinsol_two_tier_refine_time: float = -1.0

    notes: list[str] = field(default_factory=list)


def benchmark_dose(
    mdef: dict,
    pval: float,
    *,
    warmup: int,
    runs: int,
    n_steps_long: int,
    bngsim_timeout: float | None = None,
) -> DoseRecord:
    rec = DoseRecord(param_value=float(pval))
    net_path = str(BENCHMARKS_DIR / mdef["net_file"])
    pname = mdef["scan_param"]
    T_char = T_CHAR_FACTOR * mdef["t_end_hint"]

    # 1. Reference KINSOL attempt → y_ss (untimed run, just for analysis).
    ks_attempt = run_bngsim_kinsol_strict(net_path, pname, pval)
    y_ss: np.ndarray | None = None
    if "final_conc" in ks_attempt:
        y_ss = np.asarray(ks_attempt["final_conc"], dtype=float)
    else:
        rec.kinsol_failed = True
        rec.notes.append(f"newton: {ks_attempt.get('error', '?')[:120]}")

    # 2. Dense reference trajectory to T_char (untimed). Doubles as fallback
    #    target for y_ss when KINSOL did not converge.
    traj = run_bngsim_dense_trajectory(
        net_path,
        pname,
        pval,
        T_char,
        N_DENSE_POINTS,
        bngsim_timeout=bngsim_timeout,
    )
    if "error" in traj:
        rec.censored = True
        rec.censor_reason = f"reference trajectory failed: {traj['error']}"
        return rec
    t_dense = traj["time"]
    y_dense = traj["species"]
    if y_ss is None:
        # KINSOL fallback: trust the long-horizon trajectory's tail.
        y_ss = y_dense[-1, :].copy()
        rec.notes.append("y_ss from CVODE-fallback (KINSOL did not converge)")

    # 3. Settling time on the user's "enter and never leave" definition.
    t_s = settling_time(t_dense, y_dense, y_ss)
    if t_s is None:
        rec.censored = True
        rec.censor_reason = f"no settling within T_char={T_char:g}"
        return rec
    rec.t_s = t_s

    # 4. Wall-time engines at t_end = t_s. This is the theoretical
    #    minimum horizon needed to land inside the settling band; a
    #    real modeler would use t_end > t_s for safety, so the wall
    #    times here are a lower bound on the long-integration cost.
    rec.rn_time, _ = timed_median(
        lambda: run_rn_to_horizon(net_path, pname, pval, t_s, n_steps_long),
        warmup=warmup,
        runs=runs,
    )

    rec.cvode_time, _ = timed_median(
        lambda timeout=bngsim_timeout: run_bngsim_cvode_to_horizon(
            net_path,
            pname,
            pval,
            t_s,
            n_steps_long + 1,
            bngsim_timeout=timeout,
        ),
        warmup=warmup,
        runs=runs,
    )

    # 5. Column C: Newton-first with simulation fallback to t_s.
    def kinsol_first_iteration():
        a = run_bngsim_kinsol_strict(net_path, pname, pval)
        strict_t = float(a.get("wall_time", 0.0))
        if "final_conc" in a:
            return {"wall_time": strict_t, "strict": strict_t, "fallback": 0.0}
        b = run_bngsim_cvode_to_horizon(
            net_path,
            pname,
            pval,
            t_s,
            n_steps_long + 1,
            bngsim_timeout=bngsim_timeout,
        )
        if "error" in b:
            return {
                "wall_time": -1.0,
                "error": f"kinsol-first fallback failed: {b['error']}",
            }
        fb_t = float(b["wall_time"])
        return {"wall_time": strict_t + fb_t, "strict": strict_t, "fallback": fb_t}

    times: list[float] = []
    strict_acc: list[float] = []
    fb_acc: list[float] = []
    for i in range(warmup + runs):
        out = kinsol_first_iteration()
        if "error" in out:
            rec.kinsol_first_time = -1.0
            rec.notes.append(f"kinsol-first timing aborted: {out['error']}")
            break
        if i >= warmup:
            times.append(out["wall_time"])
            strict_acc.append(out["strict"])
            fb_acc.append(out["fallback"])
    else:
        rec.kinsol_first_time = median(times) if times else -1.0
        rec.kinsol_first_strict_time = median(strict_acc) if strict_acc else -1.0
        rec.kinsol_first_fallback_time = median(fb_acc) if fb_acc else 0.0

    # 6. Column D: integration burst, then Newton refinement. On Newton
    #    refinement failure, charge a CVODE-to-t_s fallback for honesty.
    def two_tier_iteration():
        a = run_bngsim_kinsol_two_tier(net_path, pname, pval)
        burst_t = float(a.get("burst", 0.0))
        refine_t = float(a.get("refine", 0.0))
        if "final_conc" in a:
            return {
                "wall_time": burst_t + refine_t,
                "burst": burst_t,
                "refine": refine_t,
                "fallback": 0.0,
            }
        b = run_bngsim_cvode_to_horizon(
            net_path,
            pname,
            pval,
            t_s,
            n_steps_long + 1,
            bngsim_timeout=bngsim_timeout,
        )
        if "error" in b:
            return {
                "wall_time": -1.0,
                "error": f"two-tier fallback failed: {b['error']}",
            }
        fb_t = float(b["wall_time"])
        return {
            "wall_time": burst_t + refine_t + fb_t,
            "burst": burst_t,
            "refine": refine_t,
            "fallback": fb_t,
        }

    tt_times: list[float] = []
    tt_burst: list[float] = []
    tt_refine: list[float] = []
    for i in range(warmup + runs):
        out = two_tier_iteration()
        if "error" in out:
            rec.kinsol_two_tier_time = -1.0
            rec.notes.append(f"two-tier timing aborted: {out['error']}")
            break
        if i >= warmup:
            tt_times.append(out["wall_time"])
            tt_burst.append(out["burst"])
            tt_refine.append(out["refine"])
    else:
        rec.kinsol_two_tier_time = median(tt_times) if tt_times else -1.0
        rec.kinsol_two_tier_burst_time = median(tt_burst) if tt_burst else -1.0
        rec.kinsol_two_tier_refine_time = median(tt_refine) if tt_refine else -1.0

    return rec


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="KINSOL steady-state dose-response benchmark (Table S8)"
    )
    ap.add_argument("--quick", type=int, default=0, help="Limit scan points per model.")
    ap.add_argument("--model", type=str, default="", help="Substring filter on model name.")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument(
        "--n-steps-long",
        type=int,
        default=100,
        help="Number of output steps for long-time wall-time runs.",
    )
    add_bngsim_timeout_arg(ap, default=None)
    args = ap.parse_args()

    info = get_machine_info()
    all_results = []

    print("=" * 78)
    print("  KINSOL Steady-State Dose-Response Benchmark (Table S8)")
    print(f"  Settling band: {EPS_REL * 100:.2f}% rel + {EPS_ABS:g} abs")
    print(f"  Protocol: {args.warmup}w + {args.runs}t per (engine, dose), median")
    print(f"  Long-time engines integrate to t_end = t_s (T_char = {T_CHAR_FACTOR:g}× t_end_hint)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(args.bngsim_timeout)}")
    print("=" * 78)

    for mdef in MODELS:
        name = mdef["name"]
        if args.model and args.model.lower() not in name.lower():
            continue
        net_path = str(BENCHMARKS_DIR / mdef["net_file"])
        if not Path(net_path).exists():
            print(f"\n  SKIP {name}: {net_path} not found")
            continue

        try:
            bngsim = _bngsim_imported()
            n_sp = bngsim.Model.from_net(net_path).n_species
        except Exception:
            n_sp = "?"

        n_pts = mdef["n_scan_pts"]
        if args.quick > 0:
            n_pts = min(args.quick, n_pts)
        scan_vals = make_scan_values(mdef["par_min"], mdef["par_max"], n_pts, mdef["log_scale"])

        print(f"\n{'─' * 78}")
        print(
            f"  {name} ({n_sp} sp, scan {mdef['scan_param']}: "
            f"{mdef['par_min']}→{mdef['par_max']}, {n_pts} pts, "
            f"hint={mdef['t_end_hint']:g})"
        )
        print(f"{'─' * 78}")
        print(
            f"  {'Param':>12} {'t_s':>10} "
            f"{'rn(s)':>9} {'CVODE(s)':>9} "
            f"{'KS-1st(s)':>10} {'KS-2tier(s)':>12}  flags"
        )
        print(f"  {'-' * 12} {'-' * 10} {'-' * 9} {'-' * 9} {'-' * 10} {'-' * 12}")

        per_dose = []
        for pval in scan_vals:
            rec = benchmark_dose(
                mdef,
                pval,
                warmup=args.warmup,
                runs=args.runs,
                n_steps_long=args.n_steps_long,
                bngsim_timeout=args.bngsim_timeout,
            )
            per_dose.append(rec)

            def fmt(x):
                return f"{x:.4g}" if (x is not None and x >= 0) else "—"

            flags = []
            if rec.censored:
                flags.append(f"CENSOR({rec.censor_reason[:40]})")
            if rec.kinsol_failed:
                flags.append("KS1_FALLBACK")
            print(
                f"  {pval:12.4g} {fmt(rec.t_s):>10} "
                f"{fmt(rec.rn_time):>9} {fmt(rec.cvode_time):>9} "
                f"{fmt(rec.kinsol_first_time):>10} "
                f"{fmt(rec.kinsol_two_tier_time):>12}  {' '.join(flags)}"
            )

        # Aggregates over non-censored doses; wall-time totals also exclude
        # censored doses.
        good = [r for r in per_dose if not r.censored]
        n_good = len(good)
        t_s_arr = np.array([r.t_s for r in good], dtype=float) if good else np.array([])
        t_s_mean = float(t_s_arr.mean()) if t_s_arr.size else None

        rn_total = sum(r.rn_time for r in good if r.rn_time > 0)
        cv_total = sum(r.cvode_time for r in good if r.cvode_time > 0)
        ks1_total = sum(r.kinsol_first_time for r in good if r.kinsol_first_time > 0)
        ks1_strict_total = sum(
            r.kinsol_first_strict_time for r in good if r.kinsol_first_strict_time > 0
        )
        ks1_fb_total = sum(r.kinsol_first_fallback_time for r in good)
        ks2_total = sum(r.kinsol_two_tier_time for r in good if r.kinsol_two_tier_time > 0)
        ks2_burst_total = sum(
            r.kinsol_two_tier_burst_time for r in good if r.kinsol_two_tier_burst_time > 0
        )
        ks2_refine_total = sum(
            r.kinsol_two_tier_refine_time for r in good if r.kinsol_two_tier_refine_time > 0
        )
        n_ks_fallback = sum(1 for r in good if r.kinsol_failed)

        model_result = {
            "name": name,
            "species": n_sp,
            "scan_param": mdef["scan_param"],
            "n_scan_pts": n_pts,
            "n_good": n_good,
            "n_kinsol_first_fallback": n_ks_fallback,
            "t_s_mean_model_time": t_s_mean,
            "wall_time_s": {
                "run_network_total": rn_total,
                "bngsim_cvode_total": cv_total,
                "bngsim_kinsol_first_total": ks1_total,
                "bngsim_kinsol_first_strict_total": ks1_strict_total,
                "bngsim_kinsol_first_fallback_total": ks1_fb_total,
                "bngsim_kinsol_two_tier_total": ks2_total,
                "bngsim_kinsol_two_tier_burst_total": ks2_burst_total,
                "bngsim_kinsol_two_tier_refine_total": ks2_refine_total,
            },
            "doses": [r.__dict__ for r in per_dose],
        }
        if ks1_total > 0 and rn_total > 0:
            model_result["speedup_kinsol_first_vs_run_network"] = rn_total / ks1_total
        if ks2_total > 0 and rn_total > 0:
            model_result["speedup_kinsol_two_tier_vs_run_network"] = rn_total / ks2_total

        print(
            f"\n  Totals over {n_good}/{n_pts} non-censored doses "
            f"({n_ks_fallback} kinsol-first fallbacks):"
        )
        print(f"    avg t_s={'—' if t_s_mean is None else f'{t_s_mean:.4g}'} model-time")
        print(
            f"    wall: run_net={rn_total * 1000:.1f}ms  "
            f"CVODE={cv_total * 1000:.1f}ms  "
            f"KS-1st={ks1_total * 1000:.1f}ms "
            f"(strict={ks1_strict_total * 1000:.1f}ms, fb={ks1_fb_total * 1000:.1f}ms)  "
            f"KS-2tier={ks2_total * 1000:.1f}ms "
            f"(burst={ks2_burst_total * 1000:.1f}ms, refine={ks2_refine_total * 1000:.1f}ms)"
        )

        all_results.append(model_result)

    print(f"\n{'=' * 78}")
    print("  SUMMARY (model-time columns: avg over non-censored doses; wall times in ms)")
    print("  KS-1st = Newton-first with sim fallback; KS-2tier = integration burst then Newton.")
    print(f"{'=' * 78}")
    hdr = (
        f"  {'Model':<20} {'avg t_s':>10} "
        f"{'KS-1st(ms)':>11} {'KS-2tier(ms)':>13} "
        f"{'CVODE(ms)':>10} {'run_net(ms)':>12}"
    )
    print(hdr)
    print(f"  {'-' * 20} {'-' * 10} {'-' * 11} {'-' * 13} {'-' * 10} {'-' * 12}")
    for r in all_results:
        ts = r["t_s_mean_model_time"]
        w = r["wall_time_s"]
        print(
            f"  {r['name']:<20} "
            f"{('—' if ts is None else f'{ts:.4g}'):>10} "
            f"{w['bngsim_kinsol_first_total'] * 1000:11.2f} "
            f"{w['bngsim_kinsol_two_tier_total'] * 1000:13.2f} "
            f"{w['bngsim_cvode_total'] * 1000:10.2f} "
            f"{w['run_network_total'] * 1000:12.2f}"
        )

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "n_steps_long": args.n_steps_long,
            "bngsim_timeout": args.bngsim_timeout,
            "settling_band": {"eps_rel": EPS_REL, "eps_abs": EPS_ABS},
            "t_char_factor": T_CHAR_FACTOR,
            "n_dense_points": N_DENSE_POINTS,
            "settling_definition": (
                "t_s = first model time after which all species remain inside "
                "the 99.73% band of y_ss for the rest of the trajectory; "
                "censored if not satisfied within T_char = T_CHAR_FACTOR × "
                "t_end_hint. Long-time engines integrate to t_end = t_s, "
                "the theoretical minimum horizon for the long-integration "
                "approach."
            ),
            "kinsol_first": {
                "policy": (
                    "strict KINSOL Newton from cold start; on failure, "
                    "fall back to CVODE integration to t_end* and charge "
                    "the column with both phases."
                ),
                "matches_bngsim_method_auto": True,
            },
            "kinsol_two_tier": {
                "policy": (
                    "integration burst (BNGsim method='integration', "
                    "tol=BURST_TOL, max_time=BURST_MAX_TIME) followed by "
                    "strict Newton refinement; both phases timed and summed."
                ),
                "burst_tol": BURST_TOL,
                "burst_max_time": BURST_MAX_TIME,
                "burst_schedule": (
                    "BNGsim solve_by_integration uses adaptive exponential "
                    "time-stepping at t = 10, 100, 1000, ... up to "
                    "burst_max_time; terminates as soon as ||f(y)|| < burst_tol."
                ),
            },
        },
        "models": all_results,
    }
    save_results(output, "bench_kinsol_dose_response")


if __name__ == "__main__":
    main()
