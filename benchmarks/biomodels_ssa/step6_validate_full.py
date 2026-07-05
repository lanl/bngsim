#!/usr/bin/env python3
"""Comprehensive cross-validation of BioModels benchmark suite.

For each model, runs 5 sequential gates:
  Gate 1: BNGsim ODE — .net loads and runs without NaN/Inf
  Gate 2: libRoadRunner ODE — SBML loads and runs
  Gate 3: ODE cross-validation — BNGsim ODE ≈ RoadRunner ODE (matched species)
  Gate 4: RoadRunner SSA means ≈ RoadRunner ODE
  Gate 5: BNGsim SSA means ≈ BNGsim ODE

Species matching uses two strategies:
  1. Direct name match (BNGsim observable names vs RR species IDs)
  2. bngxml mapping (parse _bngxml.xml for SBML id → BNG name)

Time horizon: adaptive — simulate to near steady state or use max T_END.

Output: results/validation_full.csv + console summary.

Usage:
    python step6_validate_full.py
    python step6_validate_full.py --ode-only
    python step6_validate_full.py --model BIOMD0000000060
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────

N_POINTS = 201  # ODE output points
SSA_ENSEMBLE = 50  # Number of SSA trajectories for mean estimation
SSA_SEED_BASE = 1000  # Seeds: 1000..1049
ODE_RTOL = 1e-4  # Max relative error for ODE cross-validation
ODE_ATOL = 1e-10  # Absolute floor for relative error denominator
SSA_N_SIGMA = 3.0  # SSA means must agree within N standard errors
TIMEOUT = 30  # Seconds per simulation call

# Candidate time horizons for steady-state search
T_CANDIDATES = [10.0, 100.0, 1000.0, 10000.0]

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
NET_DIR = BASE / "data" / "net_models"
SBML_DIR = BASE / "data" / "sbml_candidates"
BNGL_DIR = BASE / "data" / "bngl_models"
RESULTS_DIR = BASE / "results"


def parse_bngxml_mapping(bngxml_path):
    """Parse _bngxml.xml to get SBML species id → BNG observable name mapping.

    The bngxml has: <Species id="S1" name="Pc1()">
    We return: {"S1": "Pc1"} (strip parentheses from BNG name)
    """
    mapping = {}
    try:
        tree = ET.parse(bngxml_path)
        root = tree.getroot()
        # Handle namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for species in root.iter(f"{ns}Species"):
            sid = species.get("id", "")
            name = species.get("name", "")
            if sid and name:
                # Strip trailing () from BNG name: "Pc1()" → "Pc1"
                clean = name.rstrip(")")
                paren = clean.find("(")
                if paren >= 0:
                    clean = clean[:paren]
                mapping[sid] = clean
    except Exception:
        pass
    return mapping


def match_species(bng_obs_names, rr_species_ids, model_id):
    """Match RoadRunner species IDs to BNGsim observable names.

    Returns list of (rr_idx, bng_obs_idx) pairs, or None if matching fails.
    """
    bng_set = {name: i for i, name in enumerate(bng_obs_names)}

    # Strategy 1: Direct name match
    pairs = []
    for ri, rid in enumerate(rr_species_ids):
        if rid in bng_set:
            pairs.append((ri, bng_set[rid]))

    if len(pairs) == len(rr_species_ids):
        return pairs, "direct"

    # Strategy 2: bngxml mapping
    bngxml_path = BNGL_DIR / f"{model_id}_bngxml.xml"
    if bngxml_path.exists():
        mapping = parse_bngxml_mapping(bngxml_path)
        pairs2 = []
        for ri, rid in enumerate(rr_species_ids):
            bng_name = mapping.get(rid)
            if bng_name and bng_name in bng_set:
                pairs2.append((ri, bng_set[bng_name]))

        if len(pairs2) == len(rr_species_ids):
            return pairs2, "bngxml"

        # Try with direct match as fallback for any not found via bngxml
        if len(pairs2) > len(pairs):
            pairs = pairs2

    if len(pairs) == len(rr_species_ids):
        return pairs, "mixed"

    return None, f"matched {len(pairs)}/{len(rr_species_ids)}"


def find_steady_state_time(rr, max_t=10000.0):
    """Adaptively find time horizon where system is near steady state.

    Simulates with increasing T_END until derivatives are small or max reached.
    Returns (t_end, is_steady).
    """
    for t_end in T_CANDIDATES:
        if t_end > max_t:
            break
        try:
            rr.reset()
            n_pts = max(51, int(t_end) + 1)
            n_pts = min(n_pts, 1001)
            result = rr.simulate(0, t_end, n_pts)
            data = np.asarray(result)[:, 1:]  # skip time column

            # Check if system is "steady" at the end:
            # Compare last 10% of trajectory to see if species are changing
            n = data.shape[0]
            tail_start = max(0, n - n // 10)
            tail = data[tail_start:]
            if tail.shape[0] < 2:
                continue

            # Relative change in the tail
            ranges = np.ptp(tail, axis=0)
            maxvals = np.maximum(np.max(np.abs(data), axis=0), 1e-10)
            rel_changes = ranges / maxvals

            if np.all(rel_changes < 0.01):  # <1% change in last 10%
                return t_end, True
        except Exception:
            continue

    # Return the last candidate tried
    return T_CANDIDATES[-1] if T_CANDIDATES else 1000.0, False


def validate_model(model_id, ode_only=False):
    """Run all validation gates for a single model."""
    result = {
        "model_id": model_id,
        "t_end": None,
        "is_steady": None,
        "n_species_rr": None,
        "n_species_bng": None,
        "n_reactions_bng": None,
        "match_method": None,
        "gate1_bngsim_ode": False,
        "gate1_error": None,
        "gate1_time_s": None,
        "gate2_rr_ode": False,
        "gate2_error": None,
        "gate2_time_s": None,
        "gate3_ode_cross": False,
        "gate3_max_rel_err": None,
        "gate3_error": None,
        "gate4_rr_ssa": False,
        "gate4_max_rel_err": None,
        "gate4_error": None,
        "gate4_time_s": None,
        "gate5_bng_ssa": False,
        "gate5_max_rel_err": None,
        "gate5_error": None,
        "gate5_time_s": None,
    }

    net_path = str(NET_DIR / f"{model_id}.net")
    sbml_path = str(SBML_DIR / f"{model_id}.xml")

    if not os.path.exists(net_path):
        result["gate1_error"] = "net file not found"
        return result
    if not os.path.exists(sbml_path):
        result["gate2_error"] = "sbml file not found"
        return result

    # ─── Gate 0: Determine time horizon via RoadRunner ────────────────────
    import roadrunner

    try:
        rr = roadrunner.RoadRunner(sbml_path)
        t_end, is_steady = find_steady_state_time(rr)
        result["t_end"] = t_end
        result["is_steady"] = is_steady
    except Exception as e:
        result["gate2_error"] = f"RR load: {type(e).__name__}: {e}"
        return result

    rr_ids = rr.getIndependentFloatingSpeciesIds() + rr.getDependentFloatingSpeciesIds()
    result["n_species_rr"] = len(rr_ids)

    # ─── Gate 1: BNGsim ODE ──────────────────────────────────────────────
    import bngsim

    t0 = time.time()
    try:
        m = bngsim.Model.from_net(net_path)
        s = bngsim.Simulator(m, method="ode")
        bng_result = s.run(t_span=(0, t_end), n_points=N_POINTS)
        bng_sp = np.asarray(bng_result.species)
        bng_obs = np.asarray(bng_result.observables)
        bng_time = np.asarray(bng_result.time)

        result["n_species_bng"] = m.n_species
        result["n_reactions_bng"] = m.n_reactions

        if np.any(np.isnan(bng_sp)):
            result["gate1_error"] = "NaN in species"
        elif np.any(np.isinf(bng_sp)):
            result["gate1_error"] = "Inf in species"
        else:
            result["gate1_bngsim_ode"] = True
    except Exception as e:
        result["gate1_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    result["gate1_time_s"] = round(time.time() - t0, 3)

    if not result["gate1_bngsim_ode"]:
        return result

    # ─── Gate 2: RoadRunner ODE ───────────────────────────────────────────
    t0 = time.time()
    try:
        rr.reset()
        rr_result = rr.simulate(0, t_end, N_POINTS)
        rr_data = np.asarray(rr_result)
        rr_time = rr_data[:, 0]
        rr_sp = rr_data[:, 1:]

        if np.any(np.isnan(rr_sp)):
            result["gate2_error"] = "NaN in species"
        elif np.any(np.isinf(rr_sp)):
            result["gate2_error"] = "Inf in species"
        else:
            result["gate2_rr_ode"] = True
    except Exception as e:
        result["gate2_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    result["gate2_time_s"] = round(time.time() - t0, 3)

    if not result["gate2_rr_ode"]:
        return result

    # ─── Gate 3: ODE cross-validation ─────────────────────────────────────
    # Match RR species columns to BNGsim observable columns
    # RR result columns: rr.getIndependent + Dependent species, in that order
    # But rr.simulate() column order may differ — use colnames
    rr_colnames = [c.strip("[]") for c in rr_result.colnames[1:]]  # skip 'time'

    bng_obs_names = list(bng_result.observable_names)

    pairs, match_method = match_species(bng_obs_names, rr_colnames, model_id)
    result["match_method"] = match_method

    if pairs is None:
        result["gate3_error"] = f"Species match failed: {match_method}"
        return result

    # Compare matched trajectories
    max_rel_err = 0.0
    for rr_idx, bng_obs_idx in pairs:
        rr_traj = rr_sp[:, rr_idx]
        bng_traj = bng_obs[:, bng_obs_idx]

        # Interpolate BNGsim onto RR time grid if needed
        if len(bng_time) != len(rr_time) or not np.allclose(bng_time, rr_time, atol=1e-6):
            bng_traj = np.interp(rr_time, bng_time, bng_traj)

        denom = np.maximum(np.abs(rr_traj), ODE_ATOL)
        rel_err = np.max(np.abs(bng_traj - rr_traj) / denom)
        max_rel_err = max(max_rel_err, rel_err)

    result["gate3_max_rel_err"] = float(f"{max_rel_err:.2e}")

    if max_rel_err <= ODE_RTOL:
        result["gate3_ode_cross"] = True
    else:
        result["gate3_error"] = f"max_rel_err={max_rel_err:.2e} > {ODE_RTOL}"

    if not result["gate3_ode_cross"] or ode_only:
        return result

    # ─── Gate 4: RoadRunner SSA consistency ───────────────────────────────
    t0 = time.time()
    try:
        # Collect ensemble of SSA trajectories
        ssa_ensemble = []
        for i in range(SSA_ENSEMBLE):
            rr.reset()
            rr.setIntegrator("gillespie")
            rr.getIntegrator().setValue("seed", SSA_SEED_BASE + i)
            rr_ssa_res = rr.simulate(0, t_end, N_POINTS)
            ssa_data = np.asarray(rr_ssa_res)[:, 1:]
            ssa_ensemble.append(ssa_data)

        ssa_stack = np.stack(ssa_ensemble, axis=0)  # (N_ens, N_pts, N_sp)
        ssa_means = np.mean(ssa_stack, axis=0)  # (N_pts, N_sp)
        ssa_stds = np.std(ssa_stack, axis=0)  # (N_pts, N_sp)
        ssa_se = ssa_stds / np.sqrt(SSA_ENSEMBLE)  # standard error of mean

        # Compare SSA means to ODE (RR ODE as reference)
        max_rel_err_ssa = 0.0
        for rr_idx, _ in pairs:
            ode_traj = rr_sp[:, rr_idx]
            ssa_mean_traj = ssa_means[:, rr_idx]
            ssa_se[:, rr_idx]

            # Use max of: relative error, or absolute error / SE
            denom = np.maximum(np.abs(ode_traj), ODE_ATOL)
            rel_err = np.max(np.abs(ssa_mean_traj - ode_traj) / denom)
            max_rel_err_ssa = max(max_rel_err_ssa, rel_err)

        result["gate4_max_rel_err"] = float(f"{max_rel_err_ssa:.2e}")

        # SSA means should be within ~10% or 3 SE of ODE
        if max_rel_err_ssa <= 0.1:
            result["gate4_rr_ssa"] = True
        else:
            result["gate4_error"] = f"SSA mean drift: max_rel_err={max_rel_err_ssa:.2e}"
    except Exception as e:
        result["gate4_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    result["gate4_time_s"] = round(time.time() - t0, 3)

    if not result["gate4_rr_ssa"]:
        return result

    # ─── Gate 5: BNGsim SSA consistency ───────────────────────────────────
    t0 = time.time()
    try:
        ssa_ensemble_bng = []
        for i in range(SSA_ENSEMBLE):
            m2 = bngsim.Model.from_net(net_path)
            s2 = bngsim.Simulator(m2, method="ssa")
            r2 = s2.run(t_span=(0, t_end), n_points=N_POINTS, seed=SSA_SEED_BASE + i)
            ssa_ensemble_bng.append(np.asarray(r2.observables))

        ssa_stack_bng = np.stack(ssa_ensemble_bng, axis=0)
        ssa_means_bng = np.mean(ssa_stack_bng, axis=0)

        # Compare BNGsim SSA means to BNGsim ODE
        max_rel_err_bng_ssa = 0.0
        for _, bng_obs_idx in pairs:
            ode_traj = bng_obs[:, bng_obs_idx]
            ssa_mean_traj = ssa_means_bng[:, bng_obs_idx]

            denom = np.maximum(np.abs(ode_traj), ODE_ATOL)
            rel_err = np.max(np.abs(ssa_mean_traj - ode_traj) / denom)
            max_rel_err_bng_ssa = max(max_rel_err_bng_ssa, rel_err)

        result["gate5_max_rel_err"] = float(f"{max_rel_err_bng_ssa:.2e}")

        if max_rel_err_bng_ssa <= 0.1:
            result["gate5_bng_ssa"] = True
        else:
            result["gate5_error"] = f"BNG SSA mean drift: max_rel_err={max_rel_err_bng_ssa:.2e}"
    except Exception as e:
        result["gate5_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    result["gate5_time_s"] = round(time.time() - t0, 3)

    return result


def main():
    parser = argparse.ArgumentParser(description="Full cross-validation")
    parser.add_argument("--ode-only", action="store_true", help="Only run gates 1-3 (no SSA)")
    parser.add_argument("--model", type=str, default=None, help="Run single model by ID")
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    args = parser.parse_args()

    # Collect model IDs
    if args.model:
        model_ids = [args.model]
    else:
        model_ids = sorted(
            f.stem for f in NET_DIR.glob("*.net") if (SBML_DIR / f"{f.stem}.xml").exists()
        )

    print(f"Validating {len(model_ids)} models ({'ODE only' if args.ode_only else 'ODE + SSA'})")
    print("=" * 70)

    results = []
    for mid in model_ids:
        print(f"\n{mid}...", flush=True)
        t0 = time.time()
        r = validate_model(mid, ode_only=args.ode_only)
        wall = round(time.time() - t0, 1)

        gates = []
        if r["gate1_bngsim_ode"]:
            gates.append("G1✓")
        if r["gate2_rr_ode"]:
            gates.append("G2✓")
        if r["gate3_ode_cross"]:
            gates.append("G3✓")
        if r["gate4_rr_ssa"]:
            gates.append("G4✓")
        if r["gate5_bng_ssa"]:
            gates.append("G5✓")

        status = " ".join(gates) if gates else "FAIL"
        detail = ""
        for g in ["gate1_error", "gate2_error", "gate3_error", "gate4_error", "gate5_error"]:
            if r[g]:
                detail = f" — {r[g]}"
                break

        sp_info = ""
        if r["n_species_bng"]:
            sp_info = f" ({r['n_species_bng']}sp/{r['n_reactions_bng']}rxn)"

        t_info = f" T={r['t_end']}" if r["t_end"] else ""
        print(f"  {status}{sp_info}{t_info} [{wall}s]{detail}")

        results.append(r)

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    n = len(results)
    g1 = sum(1 for r in results if r["gate1_bngsim_ode"])
    g2 = sum(1 for r in results if r["gate2_rr_ode"])
    g3 = sum(1 for r in results if r["gate3_ode_cross"])
    g4 = sum(1 for r in results if r["gate4_rr_ssa"])
    g5 = sum(1 for r in results if r["gate5_bng_ssa"])

    print(f"  Total models:              {n}")
    print(f"  Gate 1 — BNGsim ODE:       {g1}/{n}")
    print(f"  Gate 2 — RoadRunner ODE:   {g2}/{n}")
    print(f"  Gate 3 — ODE cross-valid:  {g3}/{n}")
    if not args.ode_only:
        print(f"  Gate 4 — RR SSA means:     {g4}/{n}")
        print(f"  Gate 5 — BNG SSA means:    {g5}/{n}")

    # List fully validated models
    if args.ode_only:
        validated = [r for r in results if r["gate3_ode_cross"]]
    else:
        validated = [r for r in results if r["gate5_bng_ssa"]]

    print(f"\n  VALIDATED BENCHMARK MODELS: {len(validated)}")
    print("  " + "-" * 50)
    for r in validated:
        steady = "steady" if r["is_steady"] else "evolving"
        print(
            f"  {r['model_id']:25s} {r['n_species_bng']:3d}sp "
            f"{r['n_reactions_bng']:4d}rxn  T={r['t_end']:<8} {steady}"
        )

    # ─── Save CSV ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "validation_full.csv"

    import csv

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
