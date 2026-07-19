#!/usr/bin/env python3
"""``steady_state`` suite runner — KINSOL dose-response correctness + timing.

Promoted from ``harness/comparison/bench_kinsol_dose_response.py``. Computes
dose-response curves under five distinct steady-state strategies and compares
their wall-clock cost (paper Supplementary Table S7; see the table-number note
in ``_dev/phase4_plan.md``):

  A. ``run_network`` long-time integration (BNG2.pl subprocess).
  B. BNGsim CVODE long-time integration (in-process).
  C. BNGsim KINSOL — Newton-first with simulation fallback (matches
     BNGsim ``method="auto"``).
  D. BNGsim KINSOL — integration burst then Newton refinement.

Two gates per dose:

1. correctness — every engine that reports a steady state is cross-checked
   against the ``y_ss`` reference (strict Newton, or the long-horizon CVODE
   tail when Newton does not converge). A dose passes when all engines agree
   within ``rtol`` (just above the 99.73% settling band). Timing is only
   reported for doses that pass — a wall time is meaningless if the engines
   landed on different steady states.
2. timing — warmup + timed-run wall-clock comparison, median reported.

Per-dose reference quantities (settling time ``t_s``, censoring) carry over
from the predecessor script unchanged; see the long-form notes inline.

Usage:
    python run.py                     # both gates, full sweep
    python run.py --mode correctness  # cross-check only
    python run.py --mode timing       # timing only
    python run.py --effort low        # cheap subset (cumulative tiers)
    python run.py --quick 5           # limit scan points per model
    python run.py --model RAFi        # substring filter on model name

Output (git-ignored ``results/``):
    steady_state_results.json + steady_state_results.md
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg, filter_by_effort  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RUN_NETWORK = nb.RUN_NETWORK
DEFAULT_WARMUP = nb.DEFAULT_WARMUP
DEFAULT_RUNS = nb.DEFAULT_RUNS

# ── Settling-band parameters ──────────────────────────────────────────────
# 99.73% band corresponds to a relative tolerance of 1 - 0.9973 = 2.7e-3.
# Absolute floor protects species whose y_ss is at or near zero from
# being knocked out of the band by ordinary integrator noise.
EPS_REL = 1.0 - 0.9973
EPS_ABS = 1e-9

# ── Correctness-gate tolerances ───────────────────────────────────────────
# An engine's reported steady state is cross-checked against y_ss with
# numpy-allclose semantics |a - b| <= XCHECK_ATOL + XCHECK_RTOL*|b|. The
# relative tolerance sits just above the 99.73% settling band so a dose
# that genuinely settled is not failed by ordinary band-edge noise.
XCHECK_RTOL = 1e-2
XCHECK_ATOL = 1e-6

# ── Model definitions ─────────────────────────────────────────────────────
# t_end_hint is a per-model guess at the settling time. The dense
# reference trajectory is integrated to T_char = 10 × t_end_hint to
# safely capture overshoots; a dose is censored if no settling is found
# within T_char. The wall-time integration horizon for the engines is
# t_s itself (the per-dose measured settling time), not t_end_hint.
#
# ``effort`` buckets the model by cost for the --effort tiers: mwc settles
# fastest (t_end_hint 10), RAFi_ground slowest (t_end_hint 1e5).

# Six published, real-world models whose parameter_scan readout is a
# steady-state dose-response curve (arXiv Table 9/10). Nets are the
# generated networks vendored under models/net/ode/ (copied from
# suites/ode_fullnet/nets/, RuleHub commit 479d6d62). Scan ranges follow
# each model's native parameter_scan; n_scan_pts standardized to 20.
# ``bistable`` flags the two designed switches (toggle, lac operon) whose
# f(y)=0 has multiple roots — see the reference-branch handling in
# benchmark_dose().
MODELS = [
    {
        "name": "kinetic_proofreading",
        "net_file": "models/net/ode/kinetic_proofreading.net",
        "scan_param": "koff",
        "par_min": 1e-3,
        "par_max": 1.0,
        "n_scan_pts": 20,
        "log_scale": True,
        "t_end_hint": 1e3,
        "effort": "low",
        "bistable": False,
    },
    {
        "name": "genetic_switch",
        "net_file": "models/net/ode/genetic_switch.net",
        "scan_param": "alpha_2",
        "par_min": 1.0,
        "par_max": 500.0,
        "n_scan_pts": 20,
        "log_scale": False,
        "t_end_hint": 1e2,
        "effort": "low",
        "bistable": True,
    },
    {
        "name": "lac_operon",
        "net_file": "models/net/ode/lac_operon.net",
        "scan_param": "l_ext",
        "par_min": 1e-2,
        "par_max": 1e6,
        "n_scan_pts": 20,
        "log_scale": True,
        "t_end_hint": 1e3,
        "effort": "low",
        "bistable": True,
    },
    {
        "name": "Kocieniewski_2012",
        "net_file": "models/net/ode/Kocieniewski_2012.net",
        "scan_param": "Stot",
        "par_min": 1e1,
        "par_max": 1e6,
        "n_scan_pts": 20,
        "log_scale": True,
        "t_end_hint": 2e3,
        "effort": "medium",
        "bistable": False,
    },
    {
        "name": "Barua_2007",
        "net_file": "models/net/ode/Barua_2007.net",
        "scan_param": "Rtot",
        "par_min": 0.05,
        "par_max": 0.5,
        "n_scan_pts": 20,
        "log_scale": False,
        "t_end_hint": 1e3,
        "effort": "medium",
        "bistable": False,
    },
    {
        "name": "Barua_2013",
        "net_file": "models/net/ode/Barua_2013.net",
        "scan_param": "APCtot",
        "par_min": 3.2e4,
        "par_max": 3.2e6,
        "n_scan_pts": 20,
        "log_scale": True,
        "t_end_hint": 2.5e5,
        "effort": "high",
        "bistable": False,
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
    # (inside) sample so t_s is reported on the dense grid scale.
    return float(0.5 * (t[last_out_idx] + t[last_out_idx + 1]))


def max_rel_err(approx: np.ndarray, ref: np.ndarray) -> float:
    """Worst per-species relative error of ``approx`` against ``ref``.

    numpy-allclose semantics: an entry's "error" is
    ``|a - r| / (XCHECK_ATOL + XCHECK_RTOL*|r|) * XCHECK_RTOL`` reduced
    to a plain relative number, so a return value <= XCHECK_RTOL means
    every entry passes ``np.allclose(approx, ref, rtol, atol)``.
    """
    approx = np.asarray(approx, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if approx.shape != ref.shape:
        return float("inf")
    denom = XCHECK_ATOL + XCHECK_RTOL * np.abs(ref)
    scaled = np.abs(approx - ref) / denom  # <= 1 iff allclose
    return float(np.max(scaled) * XCHECK_RTOL)


# ── Engine: run_network ───────────────────────────────────────────────────
#
# run_network has no parameter-override CLI option, so the dose parameter
# is applied by rewriting the parameters block in a temporary .net file.
# Derived ConstantExpressions are re-evaluated by run_network at load time,
# so overriding a primary parameter propagates correctly.

_PARAM_LINE_RE = re.compile(r"^(\s*\d+\s+)([A-Za-z_]\w*)(\s+)(\S.*?)(\s*(?:#.*)?)$")


def _override_net_param(net_text: str, param_name: str, param_val: float) -> str:
    """Return the .net file text with ``param_name`` set to ``param_val``."""
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
    """Run BNG2.pl run_network to t_end with the dose parameter overridden."""
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
            return {"wall_time": elapsed, "error": (proc.stdout + proc.stderr)[:300]}
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


# Patched-.net cache: BNGsim and run_network must apply the dose the *same*
# way. ``Model.set_param()`` updates the parameter value but does NOT
# re-derive ConstantExpression parameters that feed species initial
# concentrations (e.g. mwc's ``_InitialConc1 = (C_ox*NA)*reacvol`` where
# ``C_ox`` chains back to the scanned ``log_P_ox``). Loading from a patched
# .net makes BNGsim re-derive those at parse time, exactly as run_network
# does — so both engines integrate the same dose-perturbed problem. The
# predecessor script used ``set_param`` and so silently ran a dose-
# independent BNGsim curve for IC-coupled scan parameters; see
# ``_dev/phase4_plan.md``.
_PATCH_TMPDIR: str | None = None
_PATCHED_NET: dict[tuple, str] = {}


def _patched_net_path(net_path: str, param_name: str, param_val: float) -> str:
    """Write (once, cached) a .net with the dose parameter overridden."""
    global _PATCH_TMPDIR
    key = (net_path, param_name, repr(float(param_val)))
    cached = _PATCHED_NET.get(key)
    if cached is not None:
        return cached
    if _PATCH_TMPDIR is None:
        _PATCH_TMPDIR = tempfile.mkdtemp(prefix="steady_state_net_")
        atexit.register(lambda: shutil.rmtree(_PATCH_TMPDIR, ignore_errors=True))
    with open(net_path) as f:
        patched = _override_net_param(f.read(), param_name, param_val)
    out = os.path.join(_PATCH_TMPDIR, f"patched_{len(_PATCHED_NET)}.net")
    with open(out, "w") as f:
        f.write(patched)
    _PATCHED_NET[key] = out
    return out


def _make_sim(net_path: str, param_name: str, param_val: float):
    bngsim = _bngsim_imported()
    model = bngsim.Model.from_net(_patched_net_path(net_path, param_name, param_val))
    return bngsim.Simulator(model, method="ode"), model


def run_bngsim_integration_ss(
    net_path: str, param_name: str, param_val: float, max_time: float
) -> dict:
    """Physical steady state via CVODE integration to the BNG2.pl parity
    criterion ``||f(y)||_2 / n < tol``. This is the ground-truth reference for
    the correctness gate: unlike Newton-from-IC it always returns the steady
    state the dynamics actually reach, with no spurious f(y)=0 roots and no
    NaN. Untimed."""
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
        res = sim.steady_state(method="integration", max_time=float(max_time))
        conc = np.asarray(res.concentrations, dtype=float)
        if not bool(getattr(res, "converged", True)) or not np.all(np.isfinite(conc)):
            return {"error": "integration reference did not converge"}
        return {"final_conc": conc}
    except Exception as e:
        return {"error": str(e)[:300]}


def run_bngsim_cvode_to_horizon(
    net_path: str, param_name: str, param_val: float, t_end: float, n_points: int
) -> dict:
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
        t0 = time.perf_counter()
        result = sim.run(t_span=(0.0, float(t_end)), n_points=n_points)
        elapsed = time.perf_counter() - t0
        species = np.asarray(result.species)
        return {"wall_time": elapsed, "final_conc": species[-1, :]}
    except Exception as e:
        return {"wall_time": -1.0, "error": str(e)[:300]}


def run_bngsim_dense_trajectory(
    net_path: str, param_name: str, param_val: float, t_end: float, n_points: int
) -> dict:
    """Untimed; used for trajectory analysis (and as fallback target)."""
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
        result = sim.run(t_span=(0.0, float(t_end)), n_points=n_points)
        return {
            "time": np.asarray(result.time, dtype=float),
            "species": np.asarray(result.species, dtype=float),
        }
    except Exception as e:
        return {"error": str(e)[:300]}


# ── Engine: BNGsim KINSOL (strict Newton) ────────────────────────────────


def run_bngsim_kinsol_strict(net_path: str, param_name: str, param_val: float) -> dict:
    """Strict Newton attempt only. Returns y_ss on success, error on failure."""
    try:
        sim, _ = _make_sim(net_path, param_name, param_val)
        t0 = time.perf_counter()
        try:
            res = sim.steady_state(method="newton")
            elapsed = time.perf_counter() - t0
            converged = bool(getattr(res, "converged", True))
            species = np.asarray(res.concentrations, dtype=float)
            # Guard: KINSOL can diverge into unphysical space (negative conc →
            # NaN in Hill/power rate laws, e.g. the genetic toggle). The C++
            # solver currently mislabels a NaN-residual result as converged
            # (`NaN >= tol` is false), so its integration fallback never fires
            # — detect the non-finite result here and force the harness
            # fallback to CVODE. See lanl/bngsim steady-state NaN-convergence bug.
            if not converged or not np.all(np.isfinite(species)):
                return {"wall_time": elapsed, "error": "newton did not converge (non-finite)"}
            return {"wall_time": elapsed, "final_conc": species}
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return {"wall_time": elapsed, "error": str(e)[:300]}
    except Exception as e:  # construction failure is not "wall time of KINSOL"
        return {"wall_time": -1.0, "error": f"setup: {str(e)[:280]}"}


# ── Engine: BNGsim KINSOL (integration burst, then Newton refinement) ────

BURST_TOL = 1e-3  # relaxed convergence tolerance for the integration burst
BURST_MAX_TIME = 1e6  # passes through to ``solve_by_integration`` max_time


def run_bngsim_kinsol_two_tier(net_path: str, param_name: str, param_val: float) -> dict:
    """Integration burst followed by strict Newton refinement."""
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
        species = np.asarray(refined.concentrations, dtype=float)
        if not bool(getattr(refined, "converged", True)) or not np.all(np.isfinite(species)):
            return {
                "wall_time": burst_t + refine_t,
                "burst": burst_t,
                "refine": refine_t,
                "error": "newton refinement did not converge (non-finite)",
            }
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
    """Run fn (warmup + runs) times; median wall time of the timed runs."""
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

    # Correctness gate: each engine's steady state vs the y_ss reference.
    correctness_ok: bool = False
    correctness_err: dict = field(default_factory=dict)  # engine -> max rel err
    correctness_detail: str = ""

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


def _correctness_check(
    net_path: str, pname: str, pval: float, t_s: float, n_steps_long: int, y_ss: np.ndarray
) -> tuple[bool, dict, str]:
    """Run each engine once (untimed) and cross-check its steady state vs y_ss.

    Returns (ok, {engine: max_rel_err}, detail). A dose passes when every
    engine that produced a steady state agrees with y_ss to XCHECK_RTOL.
    """
    errs: dict[str, float] = {}
    detail_parts: list[str] = []

    probes = {
        "run_network": run_rn_to_horizon(net_path, pname, pval, t_s, n_steps_long),
        "cvode": run_bngsim_cvode_to_horizon(net_path, pname, pval, t_s, n_steps_long + 1),
        "kinsol_first": run_bngsim_kinsol_strict(net_path, pname, pval),
        "kinsol_two_tier": run_bngsim_kinsol_two_tier(net_path, pname, pval),
    }
    # Both KINSOL strategies are fallback-capable — their timing columns
    # charge a CVODE fallback when the native solve fails (Newton diverges,
    # the burst misses BURST_TOL). A KINSOL strategy that falls back is
    # therefore *not* a correctness failure: its result is then CVODE's,
    # which is checked on its own row. Only run_network and CVODE must
    # always produce a steady state.
    _FALLBACK_ENGINES = {"kinsol_first", "kinsol_two_tier"}
    ok = True
    for engine, out in probes.items():
        if "final_conc" not in out:
            if engine in _FALLBACK_ENGINES:
                errs[engine] = float("nan")
                detail_parts.append(f"{engine}:fallback")
                continue
            ok = False
            errs[engine] = float("inf")
            detail_parts.append(f"{engine}:ERR({out.get('error', '?')[:50]})")
            continue
        e = max_rel_err(np.asarray(out["final_conc"], dtype=float), y_ss)
        errs[engine] = e
        if e > XCHECK_RTOL:
            ok = False
            detail_parts.append(f"{engine}:{e:.2e}>rtol")
        else:
            detail_parts.append(f"{engine}:{e:.2e}")
    return ok, errs, "  ".join(detail_parts)


def benchmark_dose(
    mdef: dict,
    pval: float,
    *,
    mode: str,
    warmup: int,
    runs: int,
    n_steps_long: int,
) -> DoseRecord:
    rec = DoseRecord(param_value=float(pval))
    net_path = str(_BENCH_ROOT / mdef["net_file"])
    pname = mdef["scan_param"]
    T_char = T_CHAR_FACTOR * mdef["t_end_hint"]

    # 1. Reference y_ss = the PHYSICAL steady state (CVODE integration to the
    #    parity criterion). Newton-from-IC is deliberately NOT the reference:
    #    seeded at the initial condition it can converge to a spurious root of
    #    f(y)=0 that the dynamics never reach (kinetic_proofreading at small
    #    koff: reldiff 1e3 vs the integrated state) or diverge to NaN (genetic
    #    toggle), so a Newton "reference" would silently mis-anchor the gate.
    ref = run_bngsim_integration_ss(net_path, pname, pval, T_char)
    y_ss: np.ndarray | None = None
    if "final_conc" in ref:
        y_ss = np.asarray(ref["final_conc"], dtype=float)

    # Record whether strict Newton (the KINSOL-first strategy) reaches the
    # physical state; it is checked against y_ss in the correctness gate below.
    ks_attempt = run_bngsim_kinsol_strict(net_path, pname, pval)
    if "final_conc" not in ks_attempt:
        rec.kinsol_failed = True
        rec.notes.append(f"newton: {ks_attempt.get('error', '?')[:120]}")

    # 2. Dense reference trajectory to T_char (untimed) for the settling time.
    #    Doubles as the y_ss fallback if the integration reference did not
    #    converge within T_char.
    traj = run_bngsim_dense_trajectory(net_path, pname, pval, T_char, N_DENSE_POINTS)
    if "error" in traj:
        rec.censored = True
        rec.censor_reason = f"reference trajectory failed: {traj['error']}"
        return rec
    t_dense = traj["time"]
    y_dense = traj["species"]
    if y_ss is None:
        y_ss = y_dense[-1, :].copy()
        rec.notes.append("y_ss from dense-trajectory tail (integration ref did not converge)")

    # 3. Settling time on the "enter and never leave" definition.
    t_s = settling_time(t_dense, y_dense, y_ss)
    if t_s is None:
        rec.censored = True
        rec.censor_reason = f"no settling within T_char={T_char:g}"
        return rec
    rec.t_s = t_s

    # 4. Correctness gate — every engine must reach the same steady state.
    if mode in ("correctness", "both"):
        ok, errs, detail = _correctness_check(net_path, pname, pval, t_s, n_steps_long, y_ss)
        rec.correctness_ok = ok
        rec.correctness_err = errs
        rec.correctness_detail = detail
    else:
        rec.correctness_ok = True  # timing-only mode does not gate

    # 5. Timing — only for doses that passed correctness (suite design rule).
    if mode == "correctness" or not rec.correctness_ok:
        if mode == "both" and not rec.correctness_ok:
            rec.notes.append("timing skipped: failed correctness gate")
        return rec

    # 5a. Wall-time engines at t_end = t_s.
    rec.rn_time, _ = timed_median(
        lambda: run_rn_to_horizon(net_path, pname, pval, t_s, n_steps_long),
        warmup=warmup,
        runs=runs,
    )
    rec.cvode_time, _ = timed_median(
        lambda: run_bngsim_cvode_to_horizon(net_path, pname, pval, t_s, n_steps_long + 1),
        warmup=warmup,
        runs=runs,
    )

    # 5b. Column C: Newton-first with simulation fallback to t_s.
    def kinsol_first_iteration():
        a = run_bngsim_kinsol_strict(net_path, pname, pval)
        strict_t = float(a.get("wall_time", 0.0))
        if "final_conc" in a:
            return {"wall_time": strict_t, "strict": strict_t, "fallback": 0.0}
        b = run_bngsim_cvode_to_horizon(net_path, pname, pval, t_s, n_steps_long + 1)
        if "error" in b:
            return {"wall_time": -1.0, "error": f"kinsol-first fallback failed: {b['error']}"}
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

    # 5c. Column D: integration burst, then Newton refinement.
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
        b = run_bngsim_cvode_to_horizon(net_path, pname, pval, t_s, n_steps_long + 1)
        if "error" in b:
            return {"wall_time": -1.0, "error": f"two-tier fallback failed: {b['error']}"}
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


# ── Reporting ─────────────────────────────────────────────────────────────


def generate_markdown(payload: dict, outpath: Path) -> None:
    info = payload["machine_info"]
    lines = [
        "# `steady_state` suite results",
        "",
        f"- mode: `{payload['mode']}`  effort: `{payload['effort']}`",
        f"- host: {info.get('hostname', '?')} / {info.get('platform', '?')}",
        f"- cpu: {info.get('cpu', '?')}",
        "",
        "KINSOL dose-response: five steady-state strategies, wall-clock cost.",
        "Timing is reported only for doses that passed the correctness gate",
        f"(all engines agree on the steady state within rtol={XCHECK_RTOL:g}).",
        "",
        "| Model | sp | doses | correct | censored | KS-1st (ms) | KS-2tier (ms) | CVODE (ms) | run_net (ms) |",
        "|-------|----|-------|---------|----------|-------------|---------------|------------|--------------|",
    ]
    for r in payload["models"]:
        w = r["wall_time_s"]
        lines.append(
            f"| {r['name']} | {r['species']} | {r['n_scan_pts']} | "
            f"{r['n_correct']}/{r['n_good']} | {r['n_censored']} | "
            f"{w['bngsim_kinsol_first_total'] * 1000:.2f} | "
            f"{w['bngsim_kinsol_two_tier_total'] * 1000:.2f} | "
            f"{w['bngsim_cvode_total'] * 1000:.2f} | "
            f"{w['run_network_total'] * 1000:.2f} |"
        )
    lines.append("")
    outpath.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="steady_state suite — KINSOL dose-response correctness + timing"
    )
    ap.add_argument(
        "--mode",
        choices=("correctness", "timing", "both"),
        default="both",
        help="Which gates to run (default: both).",
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
    add_effort_arg(ap)
    args = ap.parse_args()

    models = filter_by_effort(MODELS, args.effort, key=lambda m: m["effort"])
    if args.model:
        models = [m for m in models if args.model.lower() in m["name"].lower()]

    info = nb.machine_info()
    all_results = []

    print("=" * 78)
    print("  steady_state suite — KINSOL dose-response (Table S7)")
    print(f"  mode={args.mode}  effort={args.effort}: {len(models)} of {len(MODELS)} models")
    print(f"  Settling band: {EPS_REL * 100:.2f}% rel + {EPS_ABS:g} abs")
    print(f"  Protocol: {args.warmup}w + {args.runs}t per (engine, dose), median")
    print("=" * 78)

    for mdef in models:
        name = mdef["name"]
        net_path = str(_BENCH_ROOT / mdef["net_file"])
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
            f"  {'Param':>12} {'t_s':>10} {'correct':>8} "
            f"{'rn(s)':>9} {'CVODE(s)':>9} "
            f"{'KS-1st(s)':>10} {'KS-2tier(s)':>12}  flags"
        )

        per_dose = []
        for pval in scan_vals:
            rec = benchmark_dose(
                mdef,
                pval,
                mode=args.mode,
                warmup=args.warmup,
                runs=args.runs,
                n_steps_long=args.n_steps_long,
            )
            per_dose.append(rec)

            def fmt(x):
                return f"{x:.4g}" if (x is not None and x >= 0) else "—"

            flags = []
            if rec.censored:
                flags.append(f"CENSOR({rec.censor_reason[:40]})")
            if rec.kinsol_failed:
                flags.append("KS1_FALLBACK")
            correct = "—" if rec.censored else ("ok" if rec.correctness_ok else "FAIL")
            print(
                f"  {pval:12.4g} {fmt(rec.t_s):>10} {correct:>8} "
                f"{fmt(rec.rn_time):>9} {fmt(rec.cvode_time):>9} "
                f"{fmt(rec.kinsol_first_time):>10} "
                f"{fmt(rec.kinsol_two_tier_time):>12}  {' '.join(flags)}"
            )

        # Aggregates over non-censored doses; wall-time totals exclude
        # censored doses and doses that failed the correctness gate.
        good = [r for r in per_dose if not r.censored]
        n_good = len(good)
        n_censored = sum(1 for r in per_dose if r.censored)
        n_correct = sum(1 for r in good if r.correctness_ok)
        timed = [r for r in good if r.correctness_ok]
        t_s_arr = np.array([r.t_s for r in good], dtype=float) if good else np.array([])
        t_s_mean = float(t_s_arr.mean()) if t_s_arr.size else None

        rn_total = sum(r.rn_time for r in timed if r.rn_time > 0)
        cv_total = sum(r.cvode_time for r in timed if r.cvode_time > 0)
        ks1_total = sum(r.kinsol_first_time for r in timed if r.kinsol_first_time > 0)
        ks1_strict_total = sum(
            r.kinsol_first_strict_time for r in timed if r.kinsol_first_strict_time > 0
        )
        ks1_fb_total = sum(r.kinsol_first_fallback_time for r in timed)
        ks2_total = sum(r.kinsol_two_tier_time for r in timed if r.kinsol_two_tier_time > 0)
        ks2_burst_total = sum(
            r.kinsol_two_tier_burst_time for r in timed if r.kinsol_two_tier_burst_time > 0
        )
        ks2_refine_total = sum(
            r.kinsol_two_tier_refine_time for r in timed if r.kinsol_two_tier_refine_time > 0
        )
        n_ks_fallback = sum(1 for r in good if r.kinsol_failed)

        model_result = {
            "name": name,
            "species": n_sp,
            "scan_param": mdef["scan_param"],
            "n_scan_pts": n_pts,
            "n_good": n_good,
            "n_censored": n_censored,
            "n_correct": n_correct,
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
            "doses": [asdict(r) for r in per_dose],
        }
        if ks1_total > 0 and rn_total > 0:
            model_result["speedup_kinsol_first_vs_run_network"] = rn_total / ks1_total
        if ks2_total > 0 and rn_total > 0:
            model_result["speedup_kinsol_two_tier_vs_run_network"] = rn_total / ks2_total

        print(
            f"\n  {n_correct}/{n_good} non-censored doses passed correctness "
            f"({n_censored} censored, {n_ks_fallback} kinsol-first fallbacks)"
        )
        if args.mode != "correctness":
            print(
                f"    wall (correct doses only): run_net={rn_total * 1000:.1f}ms  "
                f"CVODE={cv_total * 1000:.1f}ms  "
                f"KS-1st={ks1_total * 1000:.1f}ms  "
                f"KS-2tier={ks2_total * 1000:.1f}ms"
            )

        all_results.append(model_result)

    payload = {
        "machine_info": info,
        "mode": args.mode,
        "effort": args.effort,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "n_steps_long": args.n_steps_long,
            "settling_band": {"eps_rel": EPS_REL, "eps_abs": EPS_ABS},
            "correctness_xcheck": {"rtol": XCHECK_RTOL, "atol": XCHECK_ATOL},
            "t_char_factor": T_CHAR_FACTOR,
            "n_dense_points": N_DENSE_POINTS,
        },
        "models": all_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "steady_state_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    generate_markdown(payload, RESULTS_DIR / "steady_state_results.md")
    print(f"\nResults: {json_path}")
    print(f"Report:  {RESULTS_DIR / 'steady_state_results.md'}")


if __name__ == "__main__":
    main()
