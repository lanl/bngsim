#!/usr/bin/env python3
"""Validate converted .net models for SSA benchmarking.

Each model is validated in a SUBPROCESS so that stuck C++
simulations (GIL released during SSA) can be killed via
SIGKILL when they exceed the timeout.

Levels:
  1: BNGsim load — .net parses without error
  2: BNGsim ODE — no NaN/Inf/crash
  3: BNGsim SSA self-consistency — same seed → identical
  4: ODE cross-validation — BNGsim vs libRoadRunner

Usage:
    python step4_validate.py
    python step4_validate.py --ode-only
    python step4_validate.py --timeout 30
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import config
import pandas as pd
import utils
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Single-model validator (runs in subprocess) ────

VALIDATE_ONE_SCRIPT = """
import json, sys, numpy as np

net_path = sys.argv[1]
sbml_path = sys.argv[2]
ode_only = sys.argv[3] == "1"
output_path = sys.argv[4]

row = {
    "load_ok": False, "load_error": None,
    "ode_ok": False, "ode_error": None,
    "ssa_consistent": False, "ssa_consistency_error": None,
    "ode_cross_ok": False, "ode_cross_error": None,
}

T_END = 10.0
N_PTS = 101

try:
    import bngsim
    model = bngsim.Model.from_net(net_path)
    if model.n_species == 0:
        row["load_error"] = "0 species"
        raise SystemExit
    row["load_ok"] = True
except SystemExit:
    pass
except Exception as e:
    row["load_error"] = f"{type(e).__name__}: {e}"

if row["load_ok"]:
    # Level 2: ODE
    try:
        m = bngsim.Model.from_net(net_path)
        s = bngsim.Simulator(m, method="ode")
        r = s.run(t_span=(0, T_END), n_points=N_PTS)
        sp = np.asarray(r.species)
        if np.any(np.isnan(sp)):
            row["ode_error"] = "NaN"
        elif np.any(np.isinf(sp)):
            row["ode_error"] = "Inf"
        elif np.any(sp < -1e-6):
            row["ode_error"] = "Negative species"
        else:
            row["ode_ok"] = True
    except Exception as e:
        row["ode_error"] = f"{type(e).__name__}: {e}"

if row["load_ok"] and not ode_only:
    # Level 3: SSA consistency
    try:
        m1 = bngsim.Model.from_net(net_path)
        s1 = bngsim.Simulator(m1, method="ssa")
        r1 = s1.run(t_span=(0, T_END), n_points=N_PTS, seed=42)

        m2 = bngsim.Model.from_net(net_path)
        s2 = bngsim.Simulator(m2, method="ssa")
        r2 = s2.run(t_span=(0, T_END), n_points=N_PTS, seed=42)

        if np.array_equal(r1.species, r2.species):
            row["ssa_consistent"] = True
        else:
            d = np.max(np.abs(r1.species - r2.species))
            row["ssa_consistency_error"] = f"max_diff={d}"
    except Exception as e:
        row["ssa_consistency_error"] = f"{type(e).__name__}: {e}"

if row.get("ssa_consistent") and not ode_only:
    # Level 4: ODE cross-validation
    import os
    if os.path.exists(sbml_path):
        try:
            import roadrunner
            m = bngsim.Model.from_net(net_path)
            s = bngsim.Simulator(m, method="ode")
            br = s.run(t_span=(0, T_END), n_points=N_PTS)
            bng_sp = np.asarray(br.species)

            rr = roadrunner.RoadRunner(sbml_path)
            rr_res = rr.simulate(0, T_END, N_PTS)
            rr_sp = np.asarray(rr_res)[:, 1:]

            bng_n = [x.lower().replace("()", "")
                     for x in br.species_names]
            rr_ids = (rr.getIndependentFloatingSpeciesIds()
                      + rr.getDependentFloatingSpeciesIds())
            rr_n = [x.lower().replace("[", "").replace("]", "")
                    for x in rr_ids]

            pairs = []
            for bi, bn in enumerate(bng_n):
                for ri, rn in enumerate(rr_n):
                    if bn == rn:
                        pairs.append((bi, ri))
                        break
            if not pairs:
                nc = min(bng_sp.shape[1], rr_sp.shape[1])
                pairs = [(i, i) for i in range(nc)] if nc else []

            if not pairs:
                row["ode_cross_error"] = "No species match"
            else:
                max_re = 0.0
                for bi, ri in pairs:
                    d = np.maximum(np.abs(rr_sp[:, ri]), 1e-10)
                    re = np.max(np.abs(bng_sp[:, bi] - rr_sp[:, ri]) / d)
                    max_re = max(max_re, re)
                if max_re > 1e-4:
                    row["ode_cross_error"] = f"max_rel_err={max_re:.2e}"
                else:
                    row["ode_cross_ok"] = True
        except Exception as e:
            row["ode_cross_error"] = f"{type(e).__name__}: {e}"

with open(output_path, "w") as f:
    json.dump(row, f)
"""


def validate_one_model(
    model_id: str,
    net_path: str,
    sbml_path: str,
    ode_only: bool = False,
    timeout_sec: int = 30,
) -> dict:
    """Validate a single model in a subprocess.

    The subprocess can be killed if it exceeds timeout,
    even when bngsim's C++ code has the GIL released.
    """
    row = {
        "model_id": model_id,
        "load_ok": False,
        "load_error": None,
        "ode_ok": False,
        "ode_error": None,
        "ssa_consistent": False,
        "ssa_consistency_error": None,
        "ode_cross_ok": False,
        "ode_cross_error": None,
        "ssa_cross_ok": False,
        "ssa_cross_error": None,
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                VALIDATE_ONE_SCRIPT,
                net_path,
                sbml_path,
                "1" if ode_only else "0",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        # Read results from temp file
        try:
            with open(tmp_path) as f:
                sub_result = json.load(f)
            row.update(sub_result)
        except (FileNotFoundError, json.JSONDecodeError):
            if result.returncode != 0:
                err = result.stderr.strip()[:200]
                row["load_error"] = f"Subprocess error: {err}"

    except subprocess.TimeoutExpired:
        row["ssa_consistency_error"] = f"Timeout ({timeout_sec}s)"

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return row


def validate_all_models(
    ode_only: bool = False,
    timeout_sec: int = 30,
) -> pd.DataFrame:
    """Validate all .net models using subprocesses."""
    net_dir = Path(config.NET_MODELS_DIR)
    if not net_dir.exists():
        logger.error(f"Net directory not found: {net_dir}")
        return pd.DataFrame()

    net_files = list(net_dir.glob("*.net"))
    model_ids = [f.stem for f in net_files]

    if not model_ids:
        logger.error("No .net models found")
        return pd.DataFrame()

    logger.info(f"Validating {len(model_ids)} models (timeout={timeout_sec}s per model)...")

    results = []
    for model_id in tqdm(model_ids, desc="Validating"):
        net_path = str(net_dir / f"{model_id}.net")
        sbml_path = str(Path(config.SBML_CANDIDATES_DIR) / f"{model_id}.xml")

        row = validate_one_model(
            model_id,
            net_path,
            sbml_path,
            ode_only=ode_only,
            timeout_sec=timeout_sec,
        )
        results.append(row)

    return pd.DataFrame(results)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Validate .net models (subprocess)")
    parser.add_argument(
        "--ode-only",
        action="store_true",
        help="Only run load + ODE tests (fast)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.VALIDATION_TIMEOUT,
        help="Timeout per model in seconds",
    )

    args = parser.parse_args()

    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("Model Validation (subprocess isolation)")
    logger.info("=" * 60)
    logger.info(f"Timeout: {args.timeout}s per model")

    df = validate_all_models(
        ode_only=args.ode_only,
        timeout_sec=args.timeout,
    )

    if df.empty:
        logger.error("No models to validate.")
        return

    log_path = Path(config.VALIDATION_LOG_CSV)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_path, index=False)
    logger.info(f"Saved to {log_path}")

    total = len(df)
    n_load = int(df["load_ok"].sum())
    n_ode = int(df["ode_ok"].sum())
    n_ssa = int(df["ssa_consistent"].sum())
    n_cross = int(df["ode_cross_ok"].sum())
    n_timeout = df["ssa_consistency_error"].astype(str).str.contains("Timeout").sum()

    print()
    print("Validation Summary")
    print("=" * 50)
    print(f"  Total .net models:        {total}")
    print(f"  Level 1 — Load OK:        {n_load}")
    print(f"  Level 2 — ODE OK:         {n_ode}")
    if not args.ode_only:
        print(f"  Level 3 — SSA consistent: {n_ssa}")
        print(f"  Level 4 — ODE cross-OK:   {n_cross}")
        print(f"  Timeouts:                 {n_timeout}")
    print()

    logger.info("Validation complete!")


if __name__ == "__main__":
    main()
