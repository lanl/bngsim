#!/usr/bin/env python3
"""Validation: BNGsim ODE vs run_network ODE (27 models).

Cross-validates BNGsim CVODE output against run_network (BNG 2.9.3)
for all 27 models in suite_ode.json. Criterion: max_rel_err < 1e-5.

Usage:
    python validate_ode.py [--quick N]

Output:
    results/validate_ode.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    SUITE_ODE,
    cross_validate_trajectories,
    get_machine_info,
    load_suite,
    run_bngsim_ode,
    run_rn_ode,
    save_results,
)

RTOL = 1e-5  # cross-validation tolerance


def main():
    parser = argparse.ArgumentParser(description="Validate BNGsim ODE vs run_network")
    parser.add_argument("--quick", type=int, default=0, help="Limit to first N models")
    args = parser.parse_args()

    print("=" * 70)
    print("  Validation: BNGsim ODE vs run_network")
    print("  Criterion: max_rel_err < 1e-5")
    print("=" * 70)

    info = get_machine_info()
    models = load_suite(SUITE_ODE)

    if args.quick > 0:
        models = models[: args.quick]

    results = []
    n_pass = n_fail = n_error = 0

    for i, m in enumerate(models):
        name = m["name"]
        net_path = str(BENCHMARKS_DIR / m["net_file"])
        t_end = m["t_end"]
        n_steps = m["n_steps"]
        n_sp = m["species"]
        n_rxn = m["reactions"]

        print(f"  [{i + 1}/{len(models)}] {name} ({n_sp} sp, {n_rxn} rxn)...", end=" ", flush=True)

        # BNGsim
        bng = run_bngsim_ode(net_path, t_end, n_steps)
        if "error" in bng:
            print(f"FAIL (BNGsim: {bng['error'][:60]})")
            results.append({"model": name, "status": "bngsim_error", "error": bng["error"]})
            n_error += 1
            continue

        # run_network
        rn = run_rn_ode(net_path, t_end, n_steps)
        if "error" in rn:
            print(f"FAIL (run_network: {rn['error'][:60]})")
            results.append({"model": name, "status": "rn_error", "error": rn["error"]})
            n_error += 1
            continue

        # Cross-validate
        passed, max_err, detail = cross_validate_trajectories(
            bng["species"],
            rn["species"],
            rtol=RTOL,
        )

        if passed:
            print(f"PASS  (err={max_err:.2e})")
            results.append({"model": name, "status": "pass", "max_rel_err": max_err})
            n_pass += 1
        else:
            print(f"FAIL  ({detail[:60]})")
            results.append(
                {"model": name, "status": "fail", "max_rel_err": max_err, "detail": detail}
            )
            n_fail += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_error}  Total: {len(models)}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "criterion": f"max_rel_err < {RTOL}",
        "summary": {"total": len(models), "pass": n_pass, "fail": n_fail, "error": n_error},
        "results": results,
    }
    save_results(output, "validate_ode")


if __name__ == "__main__":
    main()
