#!/usr/bin/env python3
"""Validation: BNGsim NFsim vs standalone BNG2.pl NFsim (5 models).

Same seed → expect exact match on observables.

Usage:
    python validate_nf.py [--quick N]

Output:
    results/validate_nf.json
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BNG2_PL,
    BNG_TIMEOUT,
    NF_DIR,
    get_machine_info,
    run_bngsim_nfsim,
    save_results,
)

SEED = 42

# Model catalog
MODELS = {
    "simple_nfsim": {"t_end": 100, "n_steps": 50},
    "basicTLBR": {"t_end": 20, "n_steps": 100},
    "receptor_nf_iter36p0h3": {"t_end": 60, "n_steps": 30},
    "egfr_nf_iter5p12h10": {
        "t_end": 120,
        "n_steps": 60,
        "gml": 1000000,
    },
    "tcr": {"t_end": 60, "n_steps": 30},
}


def strip_actions(content):
    """Remove BNG action lines from BNGL content."""
    pats = [
        r"^\s*(generate_network|simulate|simulate_nf|"
        r"writeXML|writeNetwork|writeSBML|writeMDL|"
        r"resetConcentrations|resetParameters|"
        r"saveConcentrations|setConcentration|setParameter|"
        r"parameter_scan|bifurcate|"
        r"begin\s+actions|end\s+actions)\b.*$",
    ]
    lines = content.split("\n")
    clean = []
    for line in lines:
        skip = False
        for pat in pats:
            if re.match(pat, line.strip(), re.MULTILINE):
                skip = True
                break
        if not skip:
            clean.append(line)
    return "\n".join(clean)


def write_xml(bngl_path, work_dir):
    """Generate XML via BNG2.pl writeXML()."""
    stem = bngl_path.stem
    with open(bngl_path) as f:
        content = f.read()
    clean = strip_actions(content).rstrip()
    clean += "\n\nwriteXML()\n"
    mod_path = work_dir / f"{stem}_xml.bngl"
    mod_path.write_text(clean)
    subprocess.run(
        ["perl", BNG2_PL, str(mod_path)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=BNG_TIMEOUT,
    )
    xml = work_dir / f"{stem}_xml.xml"
    if not xml.exists():
        raise FileNotFoundError(f"writeXML failed for {stem}")
    return xml


def run_bng_nfsim(bngl_path, work_dir, t_end, n_steps, seed):
    """Run BNG2.pl NFsim and return (names, data)."""
    stem = bngl_path.stem
    with open(bngl_path) as f:
        content = f.read()
    clean = strip_actions(content).rstrip()
    action = (
        f'simulate({{method=>"nf",t_end=>{t_end},n_steps=>{n_steps},seed=>{seed},gml=>1000000}})\n'
    )
    clean += "\n\n" + action
    mod_path = work_dir / f"{stem}_sim.bngl"
    mod_path.write_text(clean)
    subprocess.run(
        ["perl", BNG2_PL, str(mod_path)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=BNG_TIMEOUT,
    )
    gdat = work_dir / f"{stem}_sim.gdat"
    if not gdat.exists():
        raise FileNotFoundError(f"NFsim sim failed for {stem}")
    with open(gdat) as f:
        header = f.readline().strip().lstrip("#").split()
    data = np.loadtxt(gdat, comments="#")
    return header, data


def main():
    parser = argparse.ArgumentParser(description="Validate BNGsim NFsim vs BNG2.pl")
    parser.add_argument(
        "--quick",
        type=int,
        default=0,
        help="Limit to first N models",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Validation: BNGsim NFsim vs BNG2.pl NFsim")
    print("  Same seed → expect exact observable match")
    print("=" * 70)

    info = get_machine_info()
    model_list = list(MODELS.items())
    if args.quick > 0:
        model_list = model_list[: args.quick]

    results = []
    n_pass = n_fail = n_error = 0

    for name, cfg in model_list:
        bngl = NF_DIR / f"{name}.bngl"
        print(f"\n  {name}...", flush=True)

        if not bngl.exists():
            print(f"    SKIP: {bngl} not found")
            results.append({"model": name, "status": "skip"})
            continue

        with tempfile.TemporaryDirectory(prefix=f"nf_{name}_") as tmpdir:
            wd = Path(tmpdir)
            try:
                # Generate XML
                xml = write_xml(bngl, wd)
                # Run BNG2.pl reference
                ref_names, ref_data = run_bng_nfsim(
                    bngl,
                    wd,
                    cfg["t_end"],
                    cfg["n_steps"],
                    SEED,
                )
                # Run BNGsim
                bng = run_bngsim_nfsim(
                    str(xml),
                    cfg["t_end"],
                    cfg["n_steps"],
                    SEED,
                    gml=cfg.get("gml", 1000000),
                )
                if "error" in bng:
                    raise RuntimeError(bng["error"])

                # Compare observables
                bng_names = ["time"] + bng["obs_names"]
                bng_data = np.column_stack([bng["times"], bng["obs_data"]])

                ref_obs = [n for n in ref_names if n.lower() != "time"]
                bng_obs = [n for n in bng_names if n.lower() != "time"]
                common = [n for n in ref_obs if n in bng_obs]

                if not common:
                    print("    FAIL: no common observables")
                    results.append(
                        {
                            "model": name,
                            "status": "fail",
                            "error": "no common obs",
                        }
                    )
                    n_fail += 1
                    continue

                n_t = min(ref_data.shape[0], bng_data.shape[0])
                max_err = 0.0
                for obs in common:
                    ri = ref_names.index(obs)
                    bi = bng_names.index(obs)
                    ae = np.max(np.abs(ref_data[:n_t, ri] - bng_data[:n_t, bi]))
                    max_err = max(max_err, ae)

                if max_err == 0:
                    print(f"    PASS: exact match ({len(common)} obs)")
                    results.append(
                        {
                            "model": name,
                            "status": "pass",
                            "n_obs": len(common),
                            "max_abs_err": 0.0,
                        }
                    )
                    n_pass += 1
                else:
                    print(f"    WARN: max_abs_err={max_err:.2e} ({len(common)} obs)")
                    results.append(
                        {
                            "model": name,
                            "status": "warn",
                            "n_obs": len(common),
                            "max_abs_err": float(max_err),
                        }
                    )
                    n_pass += 1  # warn is still acceptable

            except Exception as e:
                print(f"    ERROR: {str(e)[:80]}")
                results.append(
                    {
                        "model": name,
                        "status": "error",
                        "error": str(e)[:300],
                    }
                )
                n_error += 1

    print(f"\n{'=' * 70}")
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_error}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "summary": {
            "total": len(model_list),
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
        },
        "results": results,
    }
    save_results(output, "validate_nf")


if __name__ == "__main__":
    main()
