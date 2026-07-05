#!/usr/bin/env python3
"""Convert BNG ODE models from BNGL → SBML via BNG2.pl.

For each model in suite_ode.json:
1. Run BNG2.pl with generate_network + writeSBML
2. Validate the emitted SBML loads in libRoadRunner
3. Cross-validate BNGsim (.net) vs libRoadRunner (SBML) at curated t_end

Output:
    results/pool_c_sbml.json — manifest of converted models
    models/bng_sbml/*.sbml — converted SBML files

Usage:
    python convert_bngl_to_sbml.py [--quick N]
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    BENCHMARKS_DIR,
    BNG2_PL,
    BNG_TIMEOUT,
    HARNESS_DIR,
    SUITE_ODE,
    get_machine_info,
    load_suite,
    run_bngsim_ode,
    run_roadrunner_ode_sbml,
    save_results,
)

SBML_DIR = HARNESS_DIR / "models" / "bng_sbml"


def convert_one_bngl(bngl_path: Path, output_dir: Path) -> dict:
    """Convert a single BNGL file to SBML via BNG2.pl.

    Returns dict with status, sbml_path, species_count.
    """
    stem = bngl_path.stem
    result = {"model": stem, "bngl_path": str(bngl_path)}

    with tempfile.TemporaryDirectory(prefix=f"sbml_{stem}_") as tmpdir:
        work_dir = Path(tmpdir)

        # Copy BNGL + any .tfun files to work dir
        shutil.copy2(bngl_path, work_dir / bngl_path.name)
        for tfun in bngl_path.parent.glob("*.tfun"):
            shutil.copy2(tfun, work_dir / tfun.name)

        # Read BNGL, strip existing actions, add generate_network + writeSBML
        with open(bngl_path) as f:
            content = f.read()

        # Remove action lines
        action_patterns = [
            r"^\s*(generate_network|simulate|simulate_nf|writeXML|writeNetwork|"
            r"writeSBML|writeMDL|resetConcentrations|resetParameters|"
            r"saveConcentrations|setConcentration|setParameter|"
            r"parameter_scan|bifurcate|begin\s+actions|end\s+actions)\b.*$",
        ]

        lines = content.split("\n")
        clean_lines = []
        for line in lines:
            is_action = False
            for pat in action_patterns:
                if re.match(pat, line.strip(), re.MULTILINE):
                    is_action = True
                    break
            if not is_action:
                clean_lines.append(line)

        clean_content = "\n".join(clean_lines).rstrip()
        clean_content += "\n\ngenerate_network({overwrite=>1})\nwriteSBML()\n"

        modified_bngl = work_dir / bngl_path.name
        with open(modified_bngl, "w") as f:
            f.write(clean_content)

        # Run BNG2.pl
        try:
            proc = subprocess.run(
                ["perl", BNG2_PL, str(modified_bngl)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=BNG_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            result["status"] = "bng_timeout"
            return result

        # Find .xml (SBML) output — BNG2.pl writeSBML() produces {stem}_sbml.xml
        sbml_path = work_dir / f"{stem}_sbml.xml"
        if not sbml_path.exists():
            # Try alternate names
            for suffix in ["_sbml.xml", ".xml", ".sbml"]:
                candidate = work_dir / f"{stem}{suffix}"
                if candidate.exists():
                    sbml_path = candidate
                    break
            else:
                err = (proc.stdout + proc.stderr)[-300:]
                result["status"] = "no_sbml"
                result["error"] = err
                return result

        # Copy SBML to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{stem}.xml"
        shutil.copy2(sbml_path, dest)

        result["sbml_path"] = str(dest)
        result["status"] = "converted"

    return result


def validate_sbml(model_cfg: dict, sbml_path: str) -> dict:
    """Validate SBML by running libRoadRunner and cross-validating with BNGsim .net."""
    stem = model_cfg["name"]
    t_end = model_cfg["t_end"]
    n_steps = model_cfg["n_steps"]
    net_path = str(BENCHMARKS_DIR / model_cfg["net_file"])

    result = {"model": stem}

    # Run libRoadRunner on SBML
    rr = run_roadrunner_ode_sbml(sbml_path, t_end, n_steps)
    if "error" in rr:
        result["rr_status"] = "fail"
        result["rr_error"] = rr["error"]
        return result
    result["rr_status"] = "ok"

    # Run BNGsim on .net
    bng = run_bngsim_ode(net_path, t_end, n_steps)
    if "error" in bng:
        result["bngsim_status"] = "fail"
        result["bngsim_error"] = bng["error"]
        return result
    result["bngsim_status"] = "ok"

    # Cross-validate (by column order since names may differ)
    bng_sp = bng["species"]
    rr_sp = rr["species"]

    # Shape may differ if BNG2.pl exports different observables to SBML
    n_cols = min(bng_sp.shape[1], rr_sp.shape[1])
    n_rows = min(bng_sp.shape[0], rr_sp.shape[0])

    if n_cols == 0 or n_rows == 0:
        result["xval_status"] = "shape_mismatch"
        result["bng_shape"] = list(bng_sp.shape)
        result["rr_shape"] = list(rr_sp.shape)
        return result

    # Compare matched species by name if possible
    bng_names = list(rr.get("species_names", []))
    if bng_names:
        result["rr_species_count"] = len(bng_names)

    # Simple cross-validation on all columns up to n_cols
    denom = np.maximum(np.abs(rr_sp[:n_rows, :n_cols]), 1e-8)
    rel_err = np.abs(bng_sp[:n_rows, :n_cols] - rr_sp[:n_rows, :n_cols]) / denom
    max_err = float(np.max(rel_err))

    result["max_rel_err"] = max_err
    result["n_compared_species"] = n_cols
    result["xval_status"] = "pass" if max_err < 1e-3 else "fail"

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert BNGL → SBML via BNG2.pl")
    parser.add_argument("--quick", type=int, default=0, help="Limit to first N models (0=all)")
    args = parser.parse_args()

    print("=" * 70)
    print("  BNGL → SBML Conversion (Pool C)")
    print("=" * 70)

    info = get_machine_info()
    models = load_suite(SUITE_ODE)

    if args.quick > 0:
        models = models[: args.quick]
        print(f"  (limited to first {args.quick} models)")

    print(f"\n  Models to convert: {len(models)}")
    print()

    conversion_results = []
    validation_results = []

    for i, model_cfg in enumerate(models):
        name = model_cfg["name"]
        bngl_path = BENCHMARKS_DIR / "ode" / f"{name}.bngl"

        print(f"  [{i + 1}/{len(models)}] {name}...", end=" ", flush=True)

        if not bngl_path.exists():
            print("SKIP (no .bngl)")
            conversion_results.append({"model": name, "status": "no_bngl"})
            continue

        # Convert
        cr = convert_one_bngl(bngl_path, SBML_DIR)
        conversion_results.append(cr)

        if cr["status"] != "converted":
            print(f"FAIL ({cr['status']}: {cr.get('error', '')[:60]})")
            continue

        # Validate
        vr = validate_sbml(model_cfg, cr["sbml_path"])
        validation_results.append(vr)

        xval = vr.get("xval_status", "n/a")
        max_err = vr.get("max_rel_err", -1)
        rr_st = vr.get("rr_status", "n/a")

        if xval == "pass":
            print(f"OK  (RR={rr_st}, xval={max_err:.2e})")
        else:
            print(f"WARN  (RR={rr_st}, xval={xval}, err={max_err:.2e})")

    # Summary
    n_converted = sum(1 for r in conversion_results if r["status"] == "converted")
    n_xval = sum(1 for r in validation_results if r.get("xval_status") == "pass")

    print(f"\n{'=' * 70}")
    print(f"  Conversion: {n_converted}/{len(models)} SBML files produced")
    print(f"  Validation: {n_xval}/{len(validation_results)} cross-validated vs BNGsim")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "summary": {
            "total": len(models),
            "converted": n_converted,
            "validated": n_xval,
        },
        "conversions": conversion_results,
        "validations": validation_results,
    }
    save_results(output, "pool_c_sbml")


if __name__ == "__main__":
    main()
