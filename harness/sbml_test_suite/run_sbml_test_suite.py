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
import json
import os
import re
import sys
import time
import traceback
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


# ── Grading: the shared kernel of benchmarks/suites/sbml_test_suite ──────────
#
# That suite replaced this script for Table S8 (GH #225; harness/jobs.yaml).
# This script kept its own copy of the grading, and the copy fell behind bngsim.
# It indexed a Result's species block with the MODEL's species list, but a
# parameter or compartment that an event assigns is promoted to integrator
# state without being a species column of the Result (GH #71; it is reported as
# a same-named observable, GH #202), so 79 cases died with an IndexError and
# stopped the run. It converted concentrations to amounts with each
# compartment's t=0 volume, so after an event resized a compartment it reported
# the concentration as the amount. And its resolution and comparison rules had
# drifted from the kernel's in other ways: graded by the copy, bngsim passed
# 1391 cases; graded by the kernel, 1576, including all 79. Two graders that
# disagree cannot both be the answer, so the kernel is now the only one. This
# script keeps its CLI, its report and the --candidates pool, and scores each
# case with the same run_case() the suite runs.
_KERNEL_RUN = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "suites" / "sbml_test_suite" / "run.py"
)
_kernel_module = None


def _kernel():
    """The suite's run.py, loaded once (it puts its own directory on sys.path
    for its _grading / _engines / _effort imports)."""
    global _kernel_module
    if _kernel_module is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_sbml_suite_kernel", _KERNEL_RUN)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _kernel_module = module
    return _kernel_module


def _graded(case_dir, case_id, engines):
    """Score one case for BNGsim and *engines* through the kernel, in this
    script's result shape (``tags`` as a dict, per-engine ``<eng>_*`` keys)."""
    r = _kernel().run_case(case_dir, case_id, ["bngsim", *engines])
    out = {
        "case": case_id,
        "status": r["status"],
        "error": r["error"],
        "tags": parse_model_desc(case_dir / f"{case_id}-model.m"),
        "max_err": r["max_err"],
    }
    if "traceback" in r:
        out["traceback"] = r["traceback"]
    for eng in engines:
        for key in ("status", "error", "max_err", "traceback"):
            if f"{eng}_{key}" in r:
                out[f"{eng}_{key}"] = r[f"{eng}_{key}"]
    return out


def _harness_error(case_id, engines, exc):
    """A case this script could not score, in its result shape. The kernel
    records its own failures as ``harness_error``; this catches what is left
    (reading the model description, loading the kernel), so one bad case
    cannot stop the run before any result is written (GH #700)."""
    error = f"{type(exc).__name__}: {exc}"[:200]
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    out = {
        "case": case_id,
        "status": "harness_error",
        "error": error,
        "traceback": tb,
        "tags": {"componentTags": [], "testTags": [], "testType": "unknown"},
        "max_err": 0.0,
    }
    for eng in engines:
        out[f"{eng}_status"] = "harness_error"
        out[f"{eng}_error"] = error
        out[f"{eng}_traceback"] = tb
        out[f"{eng}_max_err"] = 0.0
    return out


# What counts as a failure in the tag report and the progress line: the kernel's
# FAIL_STATUSES. The tag report used to leave out shape_mismatch, which then
# counted neither as passed nor as failed in a tag's denominator, and the
# progress line var_missing as well.
_FAIL_STATUSES = (
    "load_fail",
    "sim_fail",
    "value_mismatch",
    "var_missing",
    "shape_mismatch",
    "harness_error",
)


def run_single_case(case_dir, case_id):
    """Run a single SBML test suite case for BNGsim. Returns result dict."""
    return _graded(case_dir, case_id, [])


def _run_multi_engine_case(case_dir, case_id, engines):
    """Run a single case for BNGsim plus *engines* (``rr``, ``amici``)."""
    return _graded(case_dir, case_id, sorted(engines))


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
    n_harness = counts["harness_error"]

    print(f"\n  {engine_name}:")
    print(f"    Tested:         {n_tested}")
    if n_tested > 0:
        print(f"    PASS:           {n_pass} ({100 * n_pass / n_tested:.1f}%)")
    print(f"    Load failures:  {n_load_fail}")
    print(f"    Sim failures:   {n_sim_fail}")
    print(f"    Value mismatch: {n_val_mismatch}")
    print(f"    Var missing:    {n_var_missing}")
    print(f"    Harness errors: {n_harness}")

    return {
        "tested": n_tested,
        "pass": n_pass,
        "load_fail": n_load_fail,
        "sim_fail": n_sim_fail,
        "value_mismatch": n_val_mismatch,
        "var_missing": n_var_missing,
        "harness_error": n_harness,
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

        try:
            if extra_engines:
                r = _run_multi_engine_case(case_dir, case_id, extra_engines)
            else:
                r = run_single_case(case_dir, case_id)
        except Exception as e:  # one bad case must not stop the run
            r = _harness_error(case_id, sorted(extra_engines), e)
        if r["status"] == "harness_error":
            print(f"  {case_id}: harness_error\n{r.get('traceback', r['error'])}", file=sys.stderr)

        results.append(r)
        status_counts[r["status"]] += 1

        # Track per-tag statistics (BNGsim)
        all_tags = r["tags"].get("componentTags", []) + r["tags"].get("testTags", [])
        for tag in all_tags:
            tag_total[tag] += 1
            if r["status"] == "pass":
                tag_pass[tag] += 1
            elif r["status"] in _FAIL_STATUSES:
                tag_fail[tag] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_fail = sum(status_counts[s] for s in _FAIL_STATUSES)
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
    n_harness = status_counts["harness_error"]

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
    print(f"    Harness errors: {n_harness}")

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
            "shape_mismatch": n_shape,
            "harness_error": n_harness,
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
