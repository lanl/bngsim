#!/usr/bin/env python3
"""Triage BNGL files from all available sources into ODE/NFsim/SSA categories.

Scans multiple directories for .bngl files, reads actions blocks to determine
simulation method, and outputs a classified inventory.

Usage:
    python _dev/triage_bngl.py
"""

import json
import os
import re
from pathlib import Path

# Model-corpus roots. Each is overridable via an env var so the triage can
# scan models stored anywhere; defaults assume the standard local layout.
MODELS2_DIR = os.environ.get(
    "BNG_MODELS2", os.path.expanduser("~/Simulations/bionetgen/bng2/Models2")
)
PYBNF_EXAMPLES = os.environ.get("PYBNF_EXAMPLES", os.path.expanduser("~/Code/PyBNF/examples"))
RULEHUB_DIR = os.environ.get("RULEHUB_DIR", "/tmp/RuleHub")
RULEBENDER_WS = os.environ.get(
    "RULEBENDER_WS", os.path.expanduser("~/Simulations/RuleBender-workspace")
)
RULEBENDER_OLD_WS = os.environ.get(
    "RULEBENDER_OLD_WS",
    os.path.expanduser("~/Simulations/OLD_SAVE/RuleBender-workspace copy"),
)
BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories to scan — top-level BNGL files only (skip results/ subdirs)
SCAN_DIRS = [
    # BNG2 Models2 (published, distributed with BioNetGen)
    MODELS2_DIR,
    # PyBNF examples (published)
    *(
        os.path.join(PYBNF_EXAMPLES, d)
        for d in (
            "egfr_benchmark",
            "egfr_ode",
            "receptor",
            "degranulation",
            "Degranulation_aMCMC",
            "constraint_raf",
            "igf1r",
            "fceri_gamma",
            "demo",
            "LinearRegression_aMCMC",
        )
    ),
    # RuleBender workspace (select published/interesting models)
    *(
        os.path.join(RULEBENDER_WS, d)
        for d in (
            "Boris_RAF_CellRep",
            "BRAFV600E",
            "CaM",
            "DNP-BSA",
            "Filament",
            "IGF1R",
            "IgE_restruct",
            "LigRec",
            "MWC",
            "RAS_oscillations",
            "van_der_Pol",
            "SIR_model",
            "linear_ODEs",
            "Bruna",
            "RAF-KSR-MEK",
            "ATG",
        )
    ),
    # OLD_SAVE unique models
    os.path.join(RULEBENDER_OLD_WS, "RAS_oscillations"),
    os.path.join(RULEBENDER_OLD_WS, "PD-1"),
    # RuleHub (cloned to /tmp)
    RULEHUB_DIR,
    # Existing SSA benchmarks (for reference)
    BENCHMARKS_DIR,
]

# Skip patterns for directories
SKIP_DIR_PATTERNS = ["/results/", "/build/", "/.git/", "__pycache__", "/biomodels/"]


def classify_bngl(filepath):
    """Read a BNGL file and classify its simulation method."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception as e:
        return {"error": str(e)}

    methods = set()

    # Check for simulate({method=>"xxx",...}) commands
    for m in re.finditer(r"simulate\s*\(\s*\{([^}]*)\}", text):
        block = m.group(1)
        mm = re.search(r'method\s*=>\s*"(\w+)"', block)
        if mm:
            methods.add(mm.group(1))

    # Check for simulate_ode / simulate_nf / simulate_ps
    if re.search(r"simulate_ode", text):
        methods.add("ode")
    if re.search(r"simulate_nf", text):
        methods.add("nf")

    # Check for writeXML (NFsim indicator)
    has_writexml = bool(re.search(r"writeXML", text))

    # Check for generate_network
    has_gennet = bool(re.search(r"generate_network", text))

    # Check for Sat/Hill rate laws
    has_sat = bool(re.search(r"\bSat\b", text))

    # Check for TFUN
    has_tfun = bool(re.search(r"tfun\(", text, re.IGNORECASE))

    # Check for __FREE (PyBNF free parameters)
    has_free_params = bool(re.search(r"__FREE", text))

    # Check for compartments
    has_compartments = bool(re.search(r"begin\s+compartments", text))

    # Check for energy patterns (cBNGL)
    has_energy = bool(re.search(r"energy_BNG|energy_example|begin\s+energy", text, re.IGNORECASE))

    return {
        "methods": sorted(methods),
        "has_gennet": has_gennet,
        "has_writexml": has_writexml,
        "has_sat": has_sat,
        "has_tfun": has_tfun,
        "has_free_params": has_free_params,
        "has_compartments": has_compartments,
        "has_energy": has_energy,
    }


def main():
    seen_names = set()
    ode_candidates = []
    nfsim_candidates = []
    ssa_candidates = []
    other = []

    for base_dir in SCAN_DIRS:
        if not os.path.isdir(base_dir):
            print(f"  SKIP (not found): {base_dir}")
            continue

        for root, _dirs, files in os.walk(base_dir):
            # Skip unwanted directories
            skip = False
            for pat in SKIP_DIR_PATTERNS:
                if pat in root:
                    skip = True
                    break
            if skip:
                continue

            for f in sorted(files):
                if not f.endswith(".bngl"):
                    continue

                fpath = os.path.join(root, f)
                name = os.path.splitext(f)[0]

                # Skip duplicates by filename
                if name in seen_names:
                    continue
                seen_names.add(name)

                info = classify_bngl(fpath)
                if "error" in info:
                    continue

                record = {
                    "name": name,
                    "path": fpath,
                    **info,
                }

                methods = info["methods"]

                if "nf" in methods or info["has_writexml"]:
                    # NFsim takes priority (even if generate_network present)
                    nfsim_candidates.append(record)
                elif "ode" in methods:
                    ode_candidates.append(record)
                elif "ssa" in methods:
                    ssa_candidates.append(record)
                elif info["has_gennet"] and not methods:
                    # generate_network but no explicit method -> likely ODE
                    ode_candidates.append(record)
                else:
                    other.append(record)

    # Print results
    print("\n=== ODE CANDIDATES ===")
    for c in sorted(ode_candidates, key=lambda x: x["name"]):
        flags = []
        if c["has_sat"]:
            flags.append("Sat!")
        if c["has_tfun"]:
            flags.append("TFUN")
        if c["has_free_params"]:
            flags.append("FREE")
        if c["has_compartments"]:
            flags.append("COMP")
        if c["has_energy"]:
            flags.append("ENERGY")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {c['name']}{flag_str}: {c['path']}")

    print("\n=== NFSIM CANDIDATES ===")
    for c in sorted(nfsim_candidates, key=lambda x: x["name"]):
        flags = []
        if c["has_free_params"]:
            flags.append("FREE")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {c['name']}{flag_str}: {c['path']}")

    print("\n=== SSA (already covered in SSA benchmark suite) ===")
    for c in sorted(ssa_candidates, key=lambda x: x["name"]):
        print(f"  {c['name']}: {c['path']}")

    print("\n=== OTHER (no simulate found) ===")
    for c in sorted(other, key=lambda x: x["name"]):
        print(f"  {c['name']}: {c['path']}")

    print(
        f"\nTotals: ODE={len(ode_candidates)}, NFsim={len(nfsim_candidates)}, "
        f"SSA={len(ssa_candidates)}, Other={len(other)}"
    )

    # Save to JSON for next pipeline step
    output = {
        "ode": [
            {
                "name": c["name"],
                "path": c["path"],
                "flags": {k: v for k, v in c.items() if k.startswith("has_")},
            }
            for c in sorted(ode_candidates, key=lambda x: x["name"])
        ],
        "nfsim": [
            {
                "name": c["name"],
                "path": c["path"],
                "flags": {k: v for k, v in c.items() if k.startswith("has_")},
            }
            for c in sorted(nfsim_candidates, key=lambda x: x["name"])
        ],
        "ssa": [
            {"name": c["name"], "path": c["path"]}
            for c in sorted(ssa_candidates, key=lambda x: x["name"])
        ],
    }

    outpath = Path(__file__).parent / "triage_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
