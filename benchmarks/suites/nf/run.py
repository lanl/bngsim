#!/usr/bin/env python3
"""``nf`` suite runner — NFsim correctness: BNGsim NfsimSimulator vs BNG2.pl.

For each supported BNGL model in ``models/bngl/nf/``:
  1. Generate XML via BNG2.pl writeXML()
  2. Run BNG2.pl simulate({method=>"nf",...}) → reference .gdat
  3. Run BNGsim NfsimSimulator (in-process) → BNGsim observables
  4. Compare: same seed → expect exact match

The core models run by default; the experimental models (known-
incomplete standalone-NFsim canaries — see EXPERIMENTAL.md) are opt-in
via BNGSIM_NF_INCLUDE_EXPERIMENTAL=1.

Usage:
    python run.py
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parents[1]  # bngsim/benchmarks

# BioNetGen 2.9.3 install. Set BNGPATH to the install root; BNG2_PL overrides
# the BNG2.pl path individually. Default = canonical ~/Simulations install.
BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
BNG2PL = os.environ.get("BNG2_PL", os.path.join(BNGPATH, "BNG2.pl"))
# The corpus (core + experimental models) is vendored in-repo, so this
# is a fixed path -- no env-var override, unlike the external corpora.
NF_DIR = BENCH_ROOT / "models" / "bngl" / "nf"
NF_EXPERIMENTAL_DIR = NF_DIR
SEED = 42
TIMEOUT = 120  # seconds per BNG2.pl run
INCLUDE_EXPERIMENTAL = os.environ.get("BNGSIM_NF_INCLUDE_EXPERIMENTAL", "0") == "1"
NFSIM_V1143_COMPAT = os.environ.get("BNGSIM_NF_V1143_COMPAT", "0") == "1"
_CONNECTIVITY_ENV = os.environ.get("BNGSIM_NF_CONNECTIVITY", "").strip().lower()
if _CONNECTIVITY_ENV in {"", "default", "auto"}:
    NFSIM_CONNECTIVITY = None
elif _CONNECTIVITY_ENV in {"0", "off", "false", "no"}:
    NFSIM_CONNECTIVITY = False
elif _CONNECTIVITY_ENV in {"1", "on", "true", "yes"}:
    NFSIM_CONNECTIVITY = True
else:
    raise ValueError("BNGSIM_NF_CONNECTIVITY must be one of: auto/default, on/true/1, off/false/0")

# Model catalog: name → simulation parameters
# CORE_MODELS are expected to work with current standalone NFsim.
# EXPERIMENTAL_MODELS are known-incomplete external NFsim canaries and are
# opt-in via BNGSIM_NF_INCLUDE_EXPERIMENTAL=1.
CORE_MODELS = {
    "simple_nfsim": {
        "desc": "BNG2 Models2: Simple NFsim test model",
        "t_end": 100,
        "n_steps": 50,
    },
    "basicTLBR": {
        "desc": "TLBR model (basic variant with functions)",
        "t_end": 20,
        "n_steps": 100,
    },
    "receptor_nf_iter36p0h3": {
        "desc": "Simple receptor (fitted params, Mitra & Bhatt)",
        "t_end": 60,
        "n_steps": 30,
    },
    "egfr_nf_iter5p12h10": {
        "desc": "Kozer et al 2014: EGFR (fitted params)",
        "t_end": 120,
        "n_steps": 60,
        "gml": 1000000,
    },
    "tcr": {
        "desc": "Chylek et al: TCR signaling (published)",
        "t_end": 60,
        "n_steps": 30,
    },
    "t3": {
        "desc": "NFsim DOR single-reactant local-function test",
        "t_end": 10,
        "n_steps": 40,
    },
    "localfunc": {
        "desc": "BioNetGen local-function synthesis benchmark",
        "t_end": 10,
        "n_steps": 40,
    },
    "isingspin_localfcn": {
        "desc": "BioNetGen NFsim local-function Ising spin benchmark",
        "t_end": 20,
        "n_steps": 40,
    },
}

EXPERIMENTAL_MODELS = {
    "t_dor2": {
        "desc": "NFsim DOR2 two-reactant local-function canary",
        "t_end": 10,
        "n_steps": 40,
        "path": NF_EXPERIMENTAL_DIR / "t_dor2.bngl",
        "reason": (
            "standalone NFsim rejects this two-reactant DOR case, so it is "
            "kept out of the default correctness suite"
        ),
    },
    "test_compartment_XML": {
        "desc": "BioNetGen cBNGL compartment canary",
        "t_end": 10,
        "n_steps": 40,
        "path": NF_EXPERIMENTAL_DIR / "test_compartment_XML.bngl",
        "reason": (
            "standalone NFsim compartment support remains incomplete, so it is "
            "kept out of the default correctness suite"
        ),
    },
}

MODELS = dict(CORE_MODELS)
if INCLUDE_EXPERIMENTAL:
    MODELS.update(EXPERIMENTAL_MODELS)


def format_experimental_skip_note() -> str:
    """Explain why experimental NFsim canaries are excluded by default."""
    details = "; ".join(
        f"{name}: {config.get('reason', config['desc'])}"
        for name, config in EXPERIMENTAL_MODELS.items()
    )
    return (
        f"  Note: skipping {len(EXPERIMENTAL_MODELS)} experimental NFsim "
        "canaries from the default correctness suite "
        "(set BNGSIM_NF_INCLUDE_EXPERIMENTAL=1 to include them): "
        f"{details}."
    )


def parse_gdat(path: Path) -> tuple:
    """Parse a BNG .gdat file → (header_names, data_array).

    Returns:
        header: list of observable names (excluding 'time')
        data: numpy array of shape (n_times, 1 + n_obs) — column 0 is time
    """
    with open(path) as f:
        header_line = f.readline().strip()
    # Header starts with '#' and contains column names
    names = header_line.lstrip("#").split()
    data = np.loadtxt(path, comments="#")
    if data.size == 0:
        data = np.empty((0, len(names)))
    elif data.ndim == 1:
        data = data.reshape(1, -1)
    return names, data


def write_xml_bngl(bngl_path: Path, work_dir: Path) -> Path:
    """Generate XML from a BNGL file by writing a minimal BNGL that just calls writeXML().

    Returns path to the generated XML file.
    """
    stem = bngl_path.stem
    # Read the original BNGL, strip all actions, add writeXML()
    with open(bngl_path) as f:
        content = f.read()

    # Remove everything after 'end model' or after last 'end ...' block
    # Strategy: find the model body, then replace all actions
    # Simple approach: remove lines that are BNG actions
    action_patterns = [
        r"^\s*generate_network\b.*$",
        r"^\s*simulate\b.*$",
        r"^\s*simulate_nf\b.*$",
        r"^\s*simulate_ssa\b.*$",
        r"^\s*simulate_ode\b.*$",
        r"^\s*writeXML\b.*$",
        r"^\s*writeNetwork\b.*$",
        r"^\s*writeSBML\b.*$",
        r"^\s*writeMDL\b.*$",
        r"^\s*writeMfile\b.*$",
        r"^\s*writeMexfile\b.*$",
        r"^\s*resetConcentrations\b.*$",
        r"^\s*resetParameters\b.*$",
        r"^\s*saveConcentrations\b.*$",
        r"^\s*setConcentration\b.*$",
        r"^\s*setParameter\b.*$",
        r"^\s*parameter_scan\b.*$",
        r"^\s*bifurcate\b.*$",
        r"^\s*begin\s+actions\b.*$",
        r"^\s*end\s+actions\b.*$",
    ]

    lines = content.split("\n")
    clean_lines = []
    in_multiline_action = False
    for line in lines:
        stripped = line.strip()
        # Skip empty/comment lines in actions region
        if in_multiline_action:
            if stripped.endswith("})") or stripped.endswith("})"):
                in_multiline_action = False
            continue

        is_action = False
        for pat in action_patterns:
            if re.match(pat, line, re.MULTILINE):
                is_action = True
                # Check if it's a multi-line action (ends with \)
                if stripped.endswith("\\"):
                    in_multiline_action = True
                break

        if not is_action:
            clean_lines.append(line)

    # Add writeXML() at the end
    clean_content = "\n".join(clean_lines).rstrip() + "\n\nwriteXML()\n"

    # Write to work directory
    xml_bngl = work_dir / f"{stem}_xml.bngl"
    with open(xml_bngl, "w") as f:
        f.write(clean_content)

    # Run BNG2.pl
    result = subprocess.run(
        ["perl", BNG2PL, str(xml_bngl)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )

    xml_path = work_dir / f"{stem}_xml.xml"
    if not xml_path.exists():
        raise RuntimeError(
            f"writeXML failed for {stem}: "
            f"stdout={result.stdout[-500:]}, stderr={result.stderr[-500:]}"
        )

    return xml_path


def run_bng_nfsim(
    bngl_path: Path, work_dir: Path, t_end: float, n_steps: int, seed: int, gml: int = 1_000_000
) -> Path:
    """Run BNG2.pl with method=>'nf' and return path to .gdat."""
    stem = bngl_path.stem

    # Read original BNGL, strip actions, add our standardized simulate
    with open(bngl_path) as f:
        content = f.read()

    action_patterns = [
        r"^\s*generate_network\b.*$",
        r"^\s*simulate\b.*$",
        r"^\s*simulate_nf\b.*$",
        r"^\s*simulate_ssa\b.*$",
        r"^\s*simulate_ode\b.*$",
        r"^\s*writeXML\b.*$",
        r"^\s*writeNetwork\b.*$",
        r"^\s*writeSBML\b.*$",
        r"^\s*writeMDL\b.*$",
        r"^\s*writeMfile\b.*$",
        r"^\s*writeMexfile\b.*$",
        r"^\s*resetConcentrations\b.*$",
        r"^\s*resetParameters\b.*$",
        r"^\s*saveConcentrations\b.*$",
        r"^\s*setConcentration\b.*$",
        r"^\s*setParameter\b.*$",
        r"^\s*parameter_scan\b.*$",
        r"^\s*bifurcate\b.*$",
        r"^\s*begin\s+actions\b.*$",
        r"^\s*end\s+actions\b.*$",
    ]

    lines = content.split("\n")
    clean_lines = []
    in_multiline_action = False
    for line in lines:
        stripped = line.strip()
        if in_multiline_action:
            if stripped.endswith("})") or stripped.endswith("})"):
                in_multiline_action = False
            continue

        is_action = False
        for pat in action_patterns:
            if re.match(pat, line, re.MULTILINE):
                is_action = True
                if stripped.endswith("\\"):
                    in_multiline_action = True
                break

        if not is_action:
            clean_lines.append(line)

    # Add standardized NFsim simulation
    sim_action = (
        f'simulate({{method=>"nf",t_end=>{t_end},n_steps=>{n_steps},'
        f"seed=>{seed},gml=>{int(gml)}}})\n"
    )
    clean_content = "\n".join(clean_lines).rstrip() + "\n\n" + sim_action

    sim_bngl = work_dir / f"{stem}_sim.bngl"
    with open(sim_bngl, "w") as f:
        f.write(clean_content)

    result = subprocess.run(
        ["perl", BNG2PL, str(sim_bngl)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )

    gdat_path = work_dir / f"{stem}_sim.gdat"
    if result.returncode != 0 or not gdat_path.exists():
        raise RuntimeError(
            f"NFsim simulation failed for {stem}: "
            f"stdout={result.stdout[-500:]}, stderr={result.stderr[-500:]}"
        )

    return gdat_path


def run_bngsim_nfsim(
    xml_path: Path,
    t_end: float,
    n_steps: int,
    seed: int,
    gml: int = 1_000_000,
    connectivity: bool | None = None,
    nfsim_v1143_compat: bool = False,
) -> tuple:
    """Run BNGsim NfsimSimulator and return (obs_names, data_array)."""
    import bngsim

    # Create a dummy model (NfsimSimulator doesn't use it)
    # We need a minimal .net file for the Model constructor
    dummy_net = xml_path.parent / "_dummy.net"
    if not dummy_net.exists():
        dummy_net.write_text(
            "begin parameters\n  1 k 1\nend parameters\n"
            "begin species\n  1 A() 1\nend species\n"
            "begin reactions\nend reactions\n"
            "begin groups\n  1 A 1\nend groups\n"
        )
    model = bngsim.Model.from_net(str(dummy_net))

    sim = bngsim.Simulator(
        model,
        method="nfsim",
        xml_path=str(xml_path),
        gml=int(gml),
        connectivity=connectivity,
        nfsim_v1143_compat=nfsim_v1143_compat,
    )

    result = sim.run(t_span=(0.0, t_end), n_points=n_steps + 1, seed=seed)

    obs_names = result.observable_names
    times = result.time
    obs_data = result.observables  # (n_times, n_obs)

    # Combine time + observables like .gdat format
    full_data = np.column_stack([times, obs_data])
    all_names = ["time"] + list(obs_names)

    return all_names, full_data


def compare_gdat(ref_names, ref_data, bng_names, bng_data, model_name):
    """Compare two .gdat-style datasets. Same seed → expect exact match."""

    # Match observable names (BNG may have 'time' as first column)
    ref_obs = [n for n in ref_names if n.lower() != "time"]
    bng_obs = [n for n in bng_names if n.lower() != "time"]

    # Find common observables
    common = [n for n in ref_obs if n in bng_obs]
    if not common:
        return {
            "status": "FAIL",
            "error": f"No common observables. ref={ref_obs}, bng={bng_obs}",
        }

    # Extract matching columns
    ref_time_col = ref_names.index("time") if "time" in ref_names else 0
    bng_time_col = bng_names.index("time") if "time" in bng_names else 0

    # Check time vectors match
    ref_times = ref_data[:, ref_time_col]
    bng_times = bng_data[:, bng_time_col]

    n_ref = len(ref_times)
    n_bng = len(bng_times)
    n_compare = min(n_ref, n_bng)

    if n_compare == 0:
        return {"status": "FAIL", "error": "Empty data"}

    # Compare each common observable
    max_abs_err = 0.0
    max_rel_err = 0.0
    worst_obs = ""
    n_mismatches = 0

    for obs in common:
        ref_col = ref_names.index(obs)
        bng_col = bng_names.index(obs)

        ref_vals = ref_data[:n_compare, ref_col]
        bng_vals = bng_data[:n_compare, bng_col]

        abs_err = np.abs(ref_vals - bng_vals)
        max_ae = np.max(abs_err)

        if max_ae > 0:
            n_mismatches += 1
            denom = np.maximum(np.abs(ref_vals), 1.0)
            rel_err = abs_err / denom
            max_re = np.max(rel_err)

            if max_re > max_rel_err:
                max_rel_err = max_re
                worst_obs = obs
            max_abs_err = max(max_abs_err, max_ae)

    return {
        "status": "PASS" if max_abs_err == 0 else "WARN",
        "n_common_obs": len(common),
        "n_times": n_compare,
        "max_abs_err": float(max_abs_err),
        "max_rel_err": float(max_rel_err),
        "n_mismatches": n_mismatches,
        "worst_obs": worst_obs,
        "note": "exact match"
        if max_abs_err == 0
        else "stochastic mismatch (different RNG paths?)",
    }


def main():
    print("=" * 70)
    print("  NFsim Correctness Sweep: BNGsim vs BNG2.pl NFsim")
    print("=" * 70)
    print(f"  BNG2.pl: {BNG2PL}")
    print(f"  BNGsim NFsim v1.14.3 compatibility: {NFSIM_V1143_COMPAT}")
    print(f"  BNGsim NFsim connectivity: {NFSIM_CONNECTIVITY}")
    if not INCLUDE_EXPERIMENTAL and EXPERIMENTAL_MODELS:
        print(format_experimental_skip_note())
    elif INCLUDE_EXPERIMENTAL and EXPERIMENTAL_MODELS:
        print(
            "  Note: including experimental NFsim canaries; failures there do "
            "not change the supported-suite baseline."
        )
    print()

    results = []

    for model_name, config in MODELS.items():
        print(f"--- {model_name} ({config['desc']}) ---")
        bngl_path = config.get("path", NF_DIR / f"{model_name}.bngl")

        if not bngl_path.exists():
            print(f"  SKIP: {bngl_path} not found")
            results.append({"model": model_name, "status": "SKIP", "error": "BNGL not found"})
            continue

        with tempfile.TemporaryDirectory(prefix=f"nf_{model_name}_") as tmpdir:
            work_dir = Path(tmpdir)
            gml = int(config.get("gml", 1_000_000))

            # Step 1: Generate XML
            try:
                print("  Generating XML...", end=" ", flush=True)
                xml_path = write_xml_bngl(bngl_path, work_dir)
                print(f"OK ({xml_path.name})")
            except Exception as e:
                print(f"FAIL: {e}")
                results.append(
                    {"model": model_name, "status": "FAIL", "stage": "writeXML", "error": str(e)}
                )
                continue

            # Step 2: Run BNG2.pl NFsim reference
            try:
                print("  Running BNG2.pl NFsim...", end=" ", flush=True)
                t0 = time.time()
                ref_gdat = run_bng_nfsim(
                    bngl_path,
                    work_dir,
                    config["t_end"],
                    config["n_steps"],
                    SEED,
                    gml=gml,
                )
                dt_ref = time.time() - t0
                ref_names, ref_data = parse_gdat(ref_gdat)
                print(f"OK ({ref_data.shape}, {dt_ref:.1f}s)")
            except Exception as e:
                print(f"FAIL: {e}")
                results.append(
                    {"model": model_name, "status": "FAIL", "stage": "bng_nfsim", "error": str(e)}
                )
                continue

            # Step 3: Run BNGsim NfsimSimulator
            try:
                print("  Running BNGsim NFsim...", end=" ", flush=True)
                t0 = time.time()
                bng_names, bng_data = run_bngsim_nfsim(
                    xml_path,
                    config["t_end"],
                    config["n_steps"],
                    SEED,
                    gml=gml,
                    connectivity=NFSIM_CONNECTIVITY,
                    nfsim_v1143_compat=NFSIM_V1143_COMPAT,
                )
                dt_bng = time.time() - t0
                print(f"OK ({bng_data.shape}, {dt_bng:.1f}s)")
            except Exception as e:
                print(f"FAIL: {e}")
                results.append(
                    {
                        "model": model_name,
                        "status": "FAIL",
                        "stage": "bngsim_nfsim",
                        "error": str(e),
                    }
                )
                continue

            # Step 4: Compare
            cmp = compare_gdat(ref_names, ref_data, bng_names, bng_data, model_name)
            cmp["model"] = model_name
            cmp["desc"] = config["desc"]
            cmp["bng_time"] = dt_ref
            cmp["bngsim_time"] = dt_bng
            cmp["speedup"] = dt_ref / dt_bng if dt_bng > 0 else float("inf")
            cmp["bngsim_nfsim_v1143_compat"] = NFSIM_V1143_COMPAT
            cmp["bngsim_connectivity"] = NFSIM_CONNECTIVITY

            status = cmp["status"]
            if status == "PASS":
                print(
                    f"  PASS: Exact match ({cmp['n_common_obs']} observables, {cmp['n_times']} time points)"
                )
            elif status == "WARN":
                print(
                    f"  WARN: max_abs_err={cmp['max_abs_err']:.2e}, max_rel_err={cmp['max_rel_err']:.2e} ({cmp['worst_obs']})"
                )
            else:
                print(f"  FAIL: {cmp.get('error', 'unknown')}")

            if dt_bng > 0:
                print(
                    f"  Timing: BNG2.pl={dt_ref:.2f}s, BNGsim={dt_bng:.2f}s, speedup={cmp['speedup']:.1f}×"
                )

            results.append(cmp)

        print()

    # Summary
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    print(
        f"  PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}  SKIP: {n_skip}  Total: {len(results)}"
    )

    # Save results
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "nf_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
