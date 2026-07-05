#!/usr/bin/env python3
"""SBML Test Suite harness: BNGsim, libRoadRunner, AMICI (Table S8).

Runs the official SBML semantic test suite (1824 cases) against up to 3
engines: BNGsim, libRoadRunner, and AMICI. Produces a feature-by-feature
compatibility report with pass/fail/skip counts per engine.

Usage:
    python run_sbml_test_suite.py                    # all 1824 cases, BNGsim only
    python run_sbml_test_suite.py --engines all      # BNGsim + RR + AMICI
    python run_sbml_test_suite.py --engines bngsim,rr  # BNGsim + RR
    python run_sbml_test_suite.py --quick 50         # first 50 cases
    python run_sbml_test_suite.py --case 00001       # single case

The test suite must be at SUITE_DIR (see below).

Output:
    sbml_test_suite_results.json  (or via --output)
"""

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SUITE_DIR = Path(
    os.environ.get(
        "SBML_TEST_SUITE_DIR",
        os.path.expanduser("~/Code/sbml-test-suite/cases/semantic"),
    )
)

# BioModels candidate-coverage pool (the "Candidates" column): the in-repo
# BioModels SBML corpus that the sbml-events benchmark also uses. Resolved
# relative to the repo so the scan is reproducible on any checkout (the prior
# hardcoded, machine-local path is gone). Override via SBML_CANDIDATES_DIR.
CANDIDATES_DIR = Path(
    os.environ.get(
        "SBML_CANDIDATES_DIR",
        str(Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events"),
    )
)

# SBML levels to try, in preference order (newest first)
SBML_PREF = [
    "l3v2",
    "l3v1",
    "l2v5",
    "l2v4",
    "l2v3",
    "l2v2",
    "l2v1",
    "l1v2",
]


def parse_settings(settings_path):
    """Parse an SBML Test Suite settings file."""
    s = {}
    with open(settings_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                s[key.strip()] = val.strip()
    return {
        "start": float(s.get("start", "0")),
        "duration": float(s.get("duration", "1")),
        "steps": int(s.get("steps", "50")),
        "variables": [v.strip() for v in s.get("variables", "").split(",") if v.strip()],
        "absolute": float(s.get("absolute", "1e-7")),
        "relative": float(s.get("relative", "1e-4")),
        "amount": [v.strip() for v in s.get("amount", "").split(",") if v.strip()],
        "concentration": [v.strip() for v in s.get("concentration", "").split(",") if v.strip()],
    }


def parse_model_desc(model_m_path):
    """Parse the .m model description for tags and test type."""
    tags = {
        "componentTags": [],
        "testTags": [],
        "testType": "TimeCourse",
    }
    if not model_m_path.exists():
        return tags
    text = model_m_path.read_text()
    for key in ("componentTags", "testTags", "testType"):
        m = re.search(rf"{key}:\s*(.+?)$", text, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if key == "testType":
                tags[key] = val
            else:
                tags[key] = [t.strip() for t in val.split(",") if t.strip()]
    return tags


def find_sbml_file(case_dir, case_id):
    """Find the best available SBML file for a test case."""
    for level in SBML_PREF:
        path = case_dir / f"{case_id}-sbml-{level}.xml"
        if path.exists():
            return path
    return None


def parse_results_csv(csv_path):
    """Parse expected results CSV. Returns (times, {var: values})."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        data = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row, strict=False):
                data[h].append(float(v))
    times = np.array(data.get("time", []))
    var_data = {k: np.array(v) for k, v in data.items() if k != "time"}
    return times, var_data


def compare_results(actual, expected, atol, rtol):
    """Check if actual matches expected within tolerances.

    Returns (pass, max_err) where max_err is the maximum relative error.
    """
    if len(actual) != len(expected):
        return False, float("inf")
    diffs = np.abs(actual - expected)
    # SBML test suite uses: |actual - expected| <= atol + rtol * |expected|
    tol = atol + rtol * np.abs(expected)
    within = diffs <= tol
    if np.all(within):
        # Compute a relative error metric for reporting
        denom = np.maximum(np.abs(expected), atol)
        rel = np.max(diffs / denom)
        return True, float(rel)
    else:
        denom = np.maximum(np.abs(expected), atol)
        rel = np.max(diffs / denom)
        return False, float(rel)


# ── Multi-engine runners ─────────────────────────────────────────────────


def run_case_roadrunner(case_dir, case_id, settings, sbml_path, exp_data):
    """Run a single case with libRoadRunner. Returns status dict."""
    try:
        import roadrunner
    except ImportError:
        return {"status": "skipped", "error": "roadrunner not installed"}

    try:
        rr = roadrunner.RoadRunner(str(sbml_path))
    except Exception as e:
        return {"status": "load_fail", "error": str(e)[:200]}

    t_start = settings["start"]
    t_end = t_start + settings["duration"]
    n_points = settings["steps"] + 1

    try:
        rr.integrator.absolute_tolerance = 1e-12
        rr.integrator.relative_tolerance = 1e-8
        result = rr.simulate(t_start, t_end, n_points)
    except Exception as e:
        return {"status": "sim_fail", "error": str(e)[:200]}

    data = np.array(result)
    colnames = result.colnames

    all_pass = True
    max_err_overall = 0.0

    for var_name in settings["variables"]:
        if var_name not in exp_data:
            continue
        expected = exp_data[var_name]

        # Find variable in RR output
        actual = None
        for ci, cn in enumerate(colnames):
            clean = cn[1:-1] if cn.startswith("[") else cn
            if clean == var_name:
                actual = data[:, ci]
                break

        if actual is None:
            return {"status": "var_missing", "error": f"variable '{var_name}' not found"}

        if len(actual) != len(expected):
            return {
                "status": "shape_mismatch",
                "error": f"got {len(actual)}, expected {len(expected)}",
            }

        ok, max_err = compare_results(actual, expected, settings["absolute"], settings["relative"])
        max_err_overall = max(max_err_overall, max_err)
        if not ok:
            all_pass = False

    if all_pass:
        return {"status": "pass", "max_err": max_err_overall}
    else:
        return {
            "status": "value_mismatch",
            "error": f"max_err={max_err_overall:.6g}",
            "max_err": max_err_overall,
        }


def run_case_amici(case_dir, case_id, settings, sbml_path, exp_data):
    """Run a single case with AMICI. Returns status dict."""
    try:
        import tempfile

        import amici
    except ImportError:
        return {"status": "skipped", "error": "amici not installed"}

    t_start = settings["start"]
    t_end = t_start + settings["duration"]
    n_points = settings["steps"] + 1

    try:
        with tempfile.TemporaryDirectory(prefix="amici_sts_") as tmpdir:
            model_name = f"case_{case_id}"
            importer = amici.SbmlImporter(str(sbml_path))
            importer.sbml2amici(model_name, tmpdir)

            model_module = amici.import_model_module(model_name, tmpdir)
            model = model_module.getModel()
            solver = model.getSolver()

            model.setTimepoints(np.linspace(t_start, t_end, n_points))
            solver.setAbsoluteTolerance(1e-12)
            solver.setRelativeTolerance(1e-8)

            rdata = amici.runAmiciSimulation(model, solver)
    except Exception as e:
        return {"status": "load_fail", "error": str(e)[:200]}

    if rdata.x is None:
        return {"status": "sim_fail", "error": "AMICI returned None"}

    state_ids = list(model.getStateIds())

    all_pass = True
    max_err_overall = 0.0

    for var_name in settings["variables"]:
        if var_name not in exp_data:
            continue
        expected = exp_data[var_name]

        actual = None
        if var_name in state_ids:
            idx = state_ids.index(var_name)
            actual = rdata.x[:, idx]

        if actual is None:
            return {"status": "var_missing", "error": f"variable '{var_name}' not found"}

        if len(actual) != len(expected):
            return {
                "status": "shape_mismatch",
                "error": f"got {len(actual)}, expected {len(expected)}",
            }

        ok, max_err = compare_results(actual, expected, settings["absolute"], settings["relative"])
        max_err_overall = max(max_err_overall, max_err)
        if not ok:
            all_pass = False

    if all_pass:
        return {"status": "pass", "max_err": max_err_overall}
    else:
        return {
            "status": "value_mismatch",
            "error": f"max_err={max_err_overall:.6g}",
            "max_err": max_err_overall,
        }


def run_single_case(case_dir, case_id, timeout=30):
    """Run a single SBML test suite case. Returns result dict."""
    import bngsim

    result = {
        "case": case_id,
        "status": "unknown",
        "error": "",
        "tags": {},
        "max_err": 0.0,
    }

    # Parse metadata
    model_m = case_dir / f"{case_id}-model.m"
    tags = parse_model_desc(model_m)
    result["tags"] = tags

    # Only handle TimeCourse tests (skip SteadyState for now)
    if tags["testType"] != "TimeCourse":
        result["status"] = "skipped"
        result["error"] = f"testType={tags['testType']}"
        return result

    # Parse settings
    settings_path = case_dir / f"{case_id}-settings.txt"
    if not settings_path.exists():
        result["status"] = "skipped"
        result["error"] = "no settings file"
        return result
    settings = parse_settings(settings_path)

    # Find SBML file
    sbml_path = find_sbml_file(case_dir, case_id)
    if sbml_path is None:
        result["status"] = "skipped"
        result["error"] = "no SBML file found"
        return result

    # Parse expected results
    csv_path = case_dir / f"{case_id}-results.csv"
    if not csv_path.exists():
        result["status"] = "skipped"
        result["error"] = "no results CSV"
        return result
    exp_times, exp_data = parse_results_csv(csv_path)

    # Load model
    try:
        model = bngsim.Model.from_sbml(str(sbml_path))
    except Exception as e:
        result["status"] = "load_fail"
        result["error"] = str(e)[:200]
        return result

    # Snapshot the model's t=0 compartment volumes BEFORE running the sim.
    # For non-constant compartments (rate rule on the compartment, or AR
    # over rate-ruled variables) the parameter slot is updated during the
    # integration; we want the t=0 value here.
    comp_vol_at_t0 = {}
    with contextlib.suppress(Exception):
        for pn in model.param_names:
            with contextlib.suppress(Exception):
                comp_vol_at_t0[pn] = model.get_param(pn)

    # Simulate
    t_start = settings["start"]
    t_end = t_start + settings["duration"]
    n_points = settings["steps"] + 1

    # Parameter-only models (no species): skip simulation, use
    # constant parameter / assignment-rule values directly.
    if model.n_species == 0:
        sim_result = None
    else:
        try:
            sim = bngsim.Simulator(model, method="ode")
            # Match libRoadRunner's default integrator tolerances. The SBML
            # Test Suite expects absolute precision down to 1e-8 on small
            # species concentrations (~1e-5) where BNGsim's default 1e-8/1e-8
            # leaves no headroom. Tightening to 1e-12/1e-8 brings borderline
            # event-timing cases (e.g. 00652-00657) inside tolerance without
            # affecting cases that already pass.
            sim_result = sim.run(
                t_span=(t_start, t_end),
                n_points=n_points,
                rtol=1e-8,
                atol=1e-12,
            )
        except Exception as e:
            result["status"] = "sim_fail"
            result["error"] = str(e)[:200]
            return result

    # Extract variables and compare
    species_names = model.species_names
    param_names = model.param_names
    obs_names = model.observable_names

    # Build species → compartment volume map from SBML for amount conversion
    comp_vols = {}
    species_comps = {}
    species_hosu = {}
    try:
        import libsbml

        doc = libsbml.readSBMLFromFile(str(sbml_path))
        sbml_m = doc.getModel()
        if sbml_m:
            for ci in range(sbml_m.getNumCompartments()):
                c = sbml_m.getCompartment(ci)
                comp_vols[c.getId()] = c.getSize() if c.isSetSize() else 1.0
            for si in range(sbml_m.getNumSpecies()):
                s = sbml_m.getSpecies(si)
                species_comps[s.getId()] = s.getCompartment()
                species_hosu[s.getId()] = s.getHasOnlySubstanceUnits()
    except Exception:
        pass

    # When a compartment's volume is overridden by an initialAssignment or
    # an assignmentRule, the loader's t=0 species concentrations divide by
    # the AR/IA-resolved volume — not the raw XML size — so the harness has
    # to use the matching value here. (e.g. 00140: AR sets compartment=1
    # while raw size=5.) The model parameter holds the AR-resolved value
    # IF queried before sim runs; afterwards a non-constant compartment's
    # parameter slot has been updated to the end-of-sim value, which is
    # not what we want.
    for cid in list(comp_vols.keys()):
        try:
            v = comp_vol_at_t0[cid]
        except Exception:
            v = None
        if v is not None:
            comp_vols[cid] = v

    all_pass = True
    max_err_overall = 0.0

    for var_name in settings["variables"]:
        if var_name not in exp_data:
            continue
        expected = exp_data[var_name]

        # Find the variable in BNGsim output
        actual = None
        is_species = False
        lookup_name = var_name

        # Check species (try direct name and _safe_name variant)
        if sim_result is not None:
            for try_name in [var_name, f"_ant_{var_name}"]:
                if try_name in species_names:
                    idx = species_names.index(try_name)
                    actual = sim_result.species[:, idx]
                    is_species = True
                    lookup_name = var_name
                    break

        # Check observables (including _obs_ prefixed assignment-rule obs)
        if actual is None and sim_result is not None:
            for try_name in [
                var_name,
                f"_ant_{var_name}",
                f"_obs_{var_name}",
                f"_obs__ant_{var_name}",
            ]:
                if try_name in obs_names:
                    idx = obs_names.index(try_name)
                    actual = sim_result.observables[:, idx]
                    is_species = True
                    lookup_name = var_name
                    break

        # Check parameters (for compartment volumes, constants, AR params)
        if actual is None:
            for try_name in [var_name, f"_ant_{var_name}"]:
                if try_name in param_names:
                    val = model.get_param(try_name)
                    actual = np.full(n_points, val)
                    break

        if actual is None:
            result["status"] = "var_missing"
            result["error"] = f"variable '{var_name}' not found in output"
            return result

        # Amount conversion: BNGsim tracks concentrations, but the test
        # may expect amounts. If var is in the 'amount' list and it's a
        # species with a non-unity compartment, multiply by volume.
        if is_species and var_name in settings["amount"] and lookup_name in species_comps:
            comp_id = species_comps[lookup_name]
            vol = comp_vols.get(comp_id, 1.0)
            if vol != 1.0:
                actual = actual * vol

        if len(actual) != len(expected):
            result["status"] = "shape_mismatch"
            result["error"] = (
                f"var '{var_name}': got {len(actual)} points, expected {len(expected)}"
            )
            return result

        ok, max_err = compare_results(actual, expected, settings["absolute"], settings["relative"])
        max_err_overall = max(max_err_overall, max_err)
        if not ok:
            all_pass = False

    result["max_err"] = max_err_overall
    if all_pass:
        result["status"] = "pass"
    else:
        result["status"] = "value_mismatch"
        result["error"] = f"max_err={max_err_overall:.6g}"

    return result


def _run_multi_engine_case(case_dir, case_id, engines):
    """Run a single case across all requested engines.

    Returns a dict with BNGsim result + per-engine sub-results.
    """
    # BNGsim result (always run)
    bngsim_result = run_single_case(case_dir, case_id)

    # Parse common metadata for other engines
    settings_path = case_dir / f"{case_id}-settings.txt"
    sbml_path = find_sbml_file(case_dir, case_id)
    csv_path = case_dir / f"{case_id}-results.csv"

    can_run_others = (
        settings_path.exists()
        and sbml_path is not None
        and csv_path.exists()
        and bngsim_result["tags"].get("testType") == "TimeCourse"
    )

    if can_run_others:
        settings = parse_settings(settings_path)
        _, exp_data = parse_results_csv(csv_path)
    else:
        settings = None
        exp_data = None

    # RoadRunner
    if "rr" in engines and can_run_others:
        rr_res = run_case_roadrunner(case_dir, case_id, settings, sbml_path, exp_data)
        bngsim_result["rr_status"] = rr_res["status"]
        bngsim_result["rr_error"] = rr_res.get("error", "")
        bngsim_result["rr_max_err"] = rr_res.get("max_err", 0.0)
    elif "rr" in engines:
        bngsim_result["rr_status"] = "skipped"
        bngsim_result["rr_error"] = "case not runnable"

    # AMICI
    if "amici" in engines and can_run_others:
        amici_res = run_case_amici(case_dir, case_id, settings, sbml_path, exp_data)
        bngsim_result["amici_status"] = amici_res["status"]
        bngsim_result["amici_error"] = amici_res.get("error", "")
        bngsim_result["amici_max_err"] = amici_res.get("max_err", 0.0)
    elif "amici" in engines:
        bngsim_result["amici_status"] = "skipped"
        bngsim_result["amici_error"] = "case not runnable"

    return bngsim_result


def _print_engine_summary(engine_name, results, key_prefix):
    """Print summary for a single engine."""
    counts = Counter()
    for r in results:
        st = r.get(f"{key_prefix}_status", "not_run")
        counts[st] += 1

    n_total = len(results)
    n_pass = counts["pass"]
    n_skip = counts["skipped"] + counts.get("not_run", 0)
    n_tested = n_total - n_skip
    n_load_fail = counts["load_fail"]
    n_sim_fail = counts["sim_fail"]
    n_val_mismatch = counts["value_mismatch"]
    n_var_missing = counts["var_missing"]

    print(f"\n  {engine_name}:")
    print(f"    Tested:         {n_tested}")
    if n_tested > 0:
        print(f"    PASS:           {n_pass} ({100 * n_pass / n_tested:.1f}%)")
    print(f"    Load failures:  {n_load_fail}")
    print(f"    Sim failures:   {n_sim_fail}")
    print(f"    Value mismatch: {n_val_mismatch}")
    print(f"    Var missing:    {n_var_missing}")

    return {
        "tested": n_tested,
        "pass": n_pass,
        "load_fail": n_load_fail,
        "sim_fail": n_sim_fail,
        "value_mismatch": n_val_mismatch,
        "var_missing": n_var_missing,
    }


def main():
    parser = argparse.ArgumentParser(description="SBML Test Suite harness: BNGsim + RR + AMICI")
    parser.add_argument("--quick", type=int, default=0, help="Run only first N cases")
    parser.add_argument("--case", type=str, default="", help="Run a single case (e.g., 00001)")
    parser.add_argument(
        "--engines", type=str, default="bngsim", help="Engines: bngsim, bngsim,rr, all"
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Also score the in-repo BioModels SBML coverage pool",
    )
    parser.add_argument(
        "--candidates-quick", type=int, default=0, help="Limit candidate models to first N"
    )
    parser.add_argument(
        "--output", type=str, default="sbml_test_suite_results.json", help="Output JSON file"
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="",
        help=(
            "Comma-separated tag-prefix filter. Only run cases whose component "
            "or test tags include at least one tag starting with one of the "
            "given prefixes (e.g., 'Event' for the SBML L3 event subset)."
        ),
    )
    args = parser.parse_args()

    # Parse engines
    if args.engines == "all":
        engines = {"bngsim", "rr", "amici"}
    else:
        engines = set(e.strip() for e in args.engines.split(","))

    if not SUITE_DIR.exists():
        print(f"ERROR: Suite not found at {SUITE_DIR}")
        print(f"  Expected: {SUITE_DIR}")
        print("  Get the pinned checkout: python fetch_semantic_suite.py")
        print("  (pin: SUITE_PIN.json — sbmlteam/sbml-test-suite @473e119d)")
        sys.exit(1)

    # Collect case directories
    if args.case:
        case_dirs = [SUITE_DIR / args.case]
    else:
        case_dirs = sorted(d for d in SUITE_DIR.iterdir() if d.is_dir() and d.name.isdigit())
        if args.tag_prefix:
            # Filter to cases with at least one tag whose name starts with
            # one of the comma-separated prefixes (e.g. "Event"). The model
            # description (.m) file holds componentTags / testTags; cheap
            # parse via parse_model_desc.
            prefixes = [p.strip() for p in args.tag_prefix.split(",") if p.strip()]
            filtered = []
            for d in case_dirs:
                m_path = d / f"{d.name}-model.m"
                if not m_path.exists():
                    continue
                tags = parse_model_desc(m_path)
                all_tags = (tags.get("componentTags", []) or []) + (tags.get("testTags", []) or [])
                if any(t.startswith(p) for t in all_tags for p in prefixes):
                    filtered.append(d)
            case_dirs = filtered
        if args.quick > 0:
            case_dirs = case_dirs[: args.quick]

    print(f"Running {len(case_dirs)} SBML test suite cases...")
    print(f"Engines: {', '.join(sorted(engines))}")

    results = []
    status_counts = Counter()
    tag_pass = defaultdict(int)
    tag_fail = defaultdict(int)
    tag_total = defaultdict(int)

    extra_engines = engines - {"bngsim"}

    t0 = time.time()
    for i, case_dir in enumerate(case_dirs):
        case_id = case_dir.name

        if extra_engines:
            r = _run_multi_engine_case(case_dir, case_id, extra_engines)
        else:
            r = run_single_case(case_dir, case_id)

        results.append(r)
        status_counts[r["status"]] += 1

        # Track per-tag statistics (BNGsim)
        all_tags = r["tags"].get("componentTags", []) + r["tags"].get("testTags", [])
        for tag in all_tags:
            tag_total[tag] += 1
            if r["status"] == "pass":
                tag_pass[tag] += 1
            elif r["status"] in ("load_fail", "sim_fail", "value_mismatch", "var_missing"):
                tag_fail[tag] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_fail = (
                status_counts.get("load_fail", 0)
                + status_counts.get("sim_fail", 0)
                + status_counts.get("value_mismatch", 0)
            )
            print(
                f"  {i + 1}/{len(case_dirs)} done "
                f"({elapsed:.1f}s, "
                f"pass={status_counts['pass']}, fail={n_fail})"
            )

    elapsed = time.time() - t0

    # Summary — BNGsim
    n_total = len(results)
    n_pass = status_counts["pass"]
    n_skip = status_counts["skipped"]
    n_tested = n_total - n_skip
    n_load_fail = status_counts["load_fail"]
    n_sim_fail = status_counts["sim_fail"]
    n_val_mismatch = status_counts["value_mismatch"]
    n_var_missing = status_counts["var_missing"]
    n_shape = status_counts["shape_mismatch"]

    print(f"\n{'=' * 60}")
    print("SBML Test Suite Results")
    print(f"{'=' * 60}")
    print(f"Total cases:      {n_total}")

    print("\n  BNGsim:")
    print(f"    Skipped:        {n_skip}")
    print(f"    Tested:         {n_tested}")
    if n_tested:
        print(f"    PASS:           {n_pass} ({100 * n_pass / n_tested:.1f}%)")
    print(f"    Load failures:  {n_load_fail}")
    print(f"    Sim failures:   {n_sim_fail}")
    print(f"    Value mismatch: {n_val_mismatch}")
    print(f"    Var missing:    {n_var_missing}")
    print(f"    Shape mismatch: {n_shape}")

    engine_summaries = {
        "bngsim": {
            "total": n_total,
            "skipped": n_skip,
            "tested": n_tested,
            "pass": n_pass,
            "load_fail": n_load_fail,
            "sim_fail": n_sim_fail,
            "value_mismatch": n_val_mismatch,
            "var_missing": n_var_missing,
            "elapsed_s": elapsed,
        }
    }

    if "rr" in extra_engines:
        engine_summaries["rr"] = _print_engine_summary("libRoadRunner", results, "rr")
    if "amici" in extra_engines:
        engine_summaries["amici"] = _print_engine_summary("AMICI", results, "amici")

    print(f"\n  Time: {elapsed:.1f}s")

    # Feature tag report (BNGsim only, for brevity)
    print(f"\n{'=' * 60}")
    print("Feature Tag Report — BNGsim (pass/tested)")
    print(f"{'=' * 60}")
    sorted_tags = sorted(tag_total.keys(), key=lambda t: tag_total[t], reverse=True)
    for tag in sorted_tags[:20]:  # top 20 tags
        total = tag_total[tag]
        passed = tag_pass[tag]
        failed = tag_fail[tag]
        tested = passed + failed
        pct = 100 * passed / tested if tested else 0
        print(f"  {tag:40s}  {passed:4d}/{tested:4d} ({pct:5.1f}%)  [total={total}]")
    if len(sorted_tags) > 20:
        print(f"  ... and {len(sorted_tags) - 20} more tags (see JSON output)")

    # ── Candidates scoring (BioModels SBML coverage, Table S8 column 3) ───
    candidates_results = {}
    if args.candidates:
        if not CANDIDATES_DIR.exists():
            print(f"\n  WARNING: Candidates dir not found: {CANDIDATES_DIR}")
        else:
            cand_files = sorted(CANDIDATES_DIR.glob("*.xml"))
            if args.candidates_quick > 0:
                cand_files = cand_files[: args.candidates_quick]

            print(f"\n{'=' * 60}")
            print(f"  BioModels SBML Candidates ({len(cand_files)} models)")
            print(f"{'=' * 60}")

            # For each engine, try load + simulate at t=10, xval pairwise
            T_END_CAND = 10.0
            N_STEPS_CAND = 100

            for eng_name in sorted(engines):
                n_load = n_sim = 0
                cand_details = []

                for ci, sbml_path in enumerate(cand_files):
                    mid = sbml_path.stem
                    entry = {"model": mid, "status": "unknown"}

                    # Try loading + simulating in this engine
                    try:
                        if eng_name == "bngsim":
                            import bngsim

                            m = bngsim.Model.from_sbml(str(sbml_path))
                            sim = bngsim.Simulator(m, method="ode")
                            res = sim.run(t_span=(0, T_END_CAND), n_points=N_STEPS_CAND + 1)
                            list(res.species_names)
                            traj = np.asarray(res.species)
                            n_load += 1
                            if not np.any(np.isnan(traj)):
                                n_sim += 1
                                entry["status"] = "sim_ok"
                            else:
                                entry["status"] = "nan"
                        elif eng_name == "rr":
                            import roadrunner

                            rr = roadrunner.RoadRunner(str(sbml_path))
                            rr.integrator.absolute_tolerance = 1e-12
                            rr.integrator.relative_tolerance = 1e-8
                            result = rr.simulate(0, T_END_CAND, N_STEPS_CAND + 1)
                            data = np.array(result)
                            n_load += 1
                            if not np.any(np.isnan(data)):
                                n_sim += 1
                                entry["status"] = "sim_ok"
                            else:
                                entry["status"] = "nan"
                        elif eng_name == "amici":
                            import tempfile

                            import amici

                            with tempfile.TemporaryDirectory(prefix="amici_c_") as td:
                                mn = mid.replace("-", "_").replace(".", "_")
                                imp = amici.SbmlImporter(str(sbml_path))
                                imp.sbml2amici(mn, td)
                                mm = amici.import_model_module(mn, td)
                                am = mm.getModel()
                                sol = am.getSolver()
                                am.setTimepoints(np.linspace(0, T_END_CAND, N_STEPS_CAND + 1))
                                sol.setAbsoluteTolerance(1e-12)
                                sol.setRelativeTolerance(1e-8)
                                rd = amici.runAmiciSimulation(am, sol)
                                n_load += 1
                                if rd.x is not None and not np.any(np.isnan(rd.x)):
                                    n_sim += 1
                                    entry["status"] = "sim_ok"
                                else:
                                    entry["status"] = "sim_fail"
                    except Exception as e:
                        entry["status"] = "fail"
                        entry["error"] = str(e)[:100]

                    cand_details.append(entry)

                    if (ci + 1) % 200 == 0:
                        print(
                            f"    {eng_name}: {ci + 1}/{len(cand_files)}, "
                            f"loaded={n_load}, simulated={n_sim}"
                        )

                # "pass" = loaded and simulated without NaN
                pct = 100 * n_sim / len(cand_files) if cand_files else 0
                print(f"  {eng_name:15s}: {n_sim}/{len(cand_files)} ({pct:.1f}%) loaded+simulated")

                candidates_results[eng_name] = {
                    "total": len(cand_files),
                    "loaded": n_load,
                    "simulated": n_sim,
                    "pass_rate": pct,
                    "details": cand_details,
                }

    # Save JSON
    output = {
        "engines": sorted(engines),
        "summary": engine_summaries,
        "candidates": candidates_results if candidates_results else None,
        "tags": {
            tag: {
                "total": tag_total[tag],
                "pass": tag_pass[tag],
                "fail": tag_fail[tag],
            }
            for tag in sorted_tags
        },
        "cases": results,
    }

    out_path = Path(__file__).parent / args.output
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
