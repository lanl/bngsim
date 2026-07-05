#!/usr/bin/env python3
"""``sbml_test_suite`` suite runner — SBML semantic-suite compatibility.

Runs the official SBML semantic test suite (1824 cases) against up to four
engines — BNGsim, libRoadRunner, AMICI, COPASI — and emits a feature-by-feature
pass/fail/skip report (paper Table S8).

**Fair-by-construction (GH #225).** Every engine is graded through ONE shared
path (:mod:`_grading`): identical solver tolerances, identical SBML-truth
variable classification, identical species→compartment→parameter resolution
ladder, identical ``amount = conc(t)·vol(t)`` conversion, identical comparison /
shape / ``var_missing`` semantics. The per-engine code (:mod:`_engines`) is only
the irreducible "load the model, integrate, read value X" primitive. The
denominator is identical too: a case is *in scope* iff it is a ``TimeCourse``
with its settings / SBML / results files present — computed once and applied to
every engine, so load/sim/var-missing failures count the same for all. This
makes it structurally impossible for one engine to be graded more generously
than another; a "bngsim wins" result is therefore an engine result, not an
accounting artifact.

Unlike the other suites this one **vendors no models**: the corpus is the
external SBML Test Suite, read from ``SBML_TEST_SUITE_DIR`` (see ``SUITE_DIR``).

Two gates:

1. correctness — each case is run and its trajectory compared to the suite's
   reference CSV within the per-case tolerances; the pass/fail/skip table and
   the per-feature-tag breakdown are the compatibility result.
2. timing — per-case wall time, totalled per engine over the cases that engine
   got right (a timing number is only meaningful for a case it got right).

Usage:
    python run.py                       # correctness + timing, all cases, BNGsim
    python run.py --engines all         # BNGsim + RR + AMICI + COPASI
    python run.py --engines bngsim,rr   # BNGsim + RR
    python run.py --engines bngsim,copasi
    python run.py --mode correctness    # compatibility table only
    python run.py --mode timing         # per-engine wall-time only
    python run.py --effort low          # cheap representative subset
    python run.py --quick 50            # first 50 cases (after filters)
    python run.py --case 00001          # single case
    python run.py --tag-prefix Event    # SBML L3 event subset

Output (git-ignored ``results/``):
    sbml_test_suite_results.json + sbml_test_suite_results.md
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this suite dir (_grading/_engines)
from _effort import add_effort_arg, effort_allows  # noqa: E402
from _engines import ADAPTERS  # noqa: E402
from _grading import parse_results_csv, parse_settings, read_sbml_entities  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

# The SBML Test Suite is an external corpus; point SBML_TEST_SUITE_DIR at a
# local checkout of github.com/sbmlteam/sbml-test-suite (the runner never
# writes to it).
SUITE_DIR = Path(
    os.environ.get(
        "SBML_TEST_SUITE_DIR",
        os.path.expanduser("~/Code/sbml-test-suite/cases/semantic"),
    )
)

# SBML levels to try, in preference order (newest first)
SBML_PREF = ["l3v2", "l3v1", "l2v5", "l2v4", "l2v3", "l2v2", "l2v1", "l1v2"]

# Engine display labels + the canonical "all" set (order = report order).
ENGINE_LABELS = {
    "bngsim": "BNGsim",
    "rr": "libRoadRunner",
    "amici": "AMICI",
    "copasi": "COPASI",
}
ALL_ENGINES = ["bngsim", "rr", "amici", "copasi"]

# Per-engine failure statuses that count against an engine's denominator (a real
# fail), versus "skipped" which means out-of-scope or engine-not-installed and is
# excluded from every engine's denominator identically.
FAIL_STATUSES = ("load_fail", "sim_fail", "value_mismatch", "var_missing", "shape_mismatch")


def effort_tier(case_id: str) -> str:
    """Bucket a numbered case into a cumulative --effort tier (deterministic
    strided sample): ``low`` = case % 20 == 0 (~5%), ``medium`` = case % 4 == 0
    (~25%, superset of low), ``high`` = every case."""
    try:
        n = int(case_id)
    except ValueError:
        return "high"
    if n % 20 == 0:
        return "low"
    if n % 4 == 0:
        return "medium"
    return "high"


def parse_model_desc(model_m_path):
    """Parse the .m model description for tags and test type. ``tag_list`` is the
    flat union of component + test tags (what the per-tag report and emit.py's
    event filter consume)."""
    out = {"componentTags": [], "testTags": [], "testType": "TimeCourse"}
    if model_m_path.exists():
        text = model_m_path.read_text()
        for key in ("componentTags", "testTags", "testType"):
            m = re.search(rf"{key}:\s*(.+?)$", text, re.MULTILINE)
            if m:
                val = m.group(1).strip()
                if key == "testType":
                    out[key] = val
                else:
                    out[key] = [t.strip() for t in val.split(",") if t.strip()]
    out["tag_list"] = out["componentTags"] + out["testTags"]
    return out


def find_sbml_file(case_dir, case_id):
    """Find the best available SBML file for a test case."""
    for level in SBML_PREF:
        path = case_dir / f"{case_id}-sbml-{level}.xml"
        if path.exists():
            return path
    return None


def run_case(case_dir, case_id, engines) -> dict:
    """Run one suite case across the requested engines. Returns a result dict
    with a flat ``tags`` list (for emit) plus, per engine, ``<eng>_status`` /
    ``<eng>_error`` / ``<eng>_max_err`` / ``<eng>_wall_s``. ``bngsim`` is also
    mirrored to the bare ``status`` / ``error`` / ``max_err`` keys (primary
    subject) so the existing report consumers keep working.

    Scope is decided ONCE here and applied to every engine: a case that is not a
    runnable ``TimeCourse`` is ``skipped`` for all engines and excluded from every
    denominator — the identical-denominator guarantee.
    """
    desc = parse_model_desc(case_dir / f"{case_id}-model.m")
    result = {
        "case": case_id,
        "tags": desc["tag_list"],
        "test_type": desc["testType"],
        "status": "unknown",
        "error": "",
        "max_err": 0.0,
    }

    settings_path = case_dir / f"{case_id}-settings.txt"
    sbml_path = find_sbml_file(case_dir, case_id)
    csv_path = case_dir / f"{case_id}-results.csv"

    # ── Shared scope gate (engine-independent) ──────────────────────────────
    skip_reason = None
    if desc["testType"] != "TimeCourse":
        skip_reason = f"testType={desc['testType']}"
    elif not settings_path.exists():
        skip_reason = "no settings file"
    elif sbml_path is None:
        skip_reason = "no SBML file found"
    elif not csv_path.exists():
        skip_reason = "no results CSV"

    if skip_reason is not None:
        for eng in engines:
            result[f"{eng}_status"] = "skipped"
            result[f"{eng}_error"] = skip_reason
            result[f"{eng}_max_err"] = 0.0
            result[f"{eng}_wall_s"] = 0.0
        result["status"] = "skipped"
        result["error"] = skip_reason
        return result

    settings = parse_settings(settings_path)
    _, exp_data = parse_results_csv(csv_path, settings=settings)
    ent = read_sbml_entities(sbml_path)  # one libSBML classification, shared by all engines

    for eng in engines:
        t0 = time.perf_counter()
        try:
            res = ADAPTERS[eng](case_id, sbml_path, settings, exp_data, ent)
        except Exception as e:  # crash isolation: one bad engine must not abort the case
            res = {
                "status": "sim_fail",
                "error": f"uncaught {type(e).__name__}: {e}"[:200],
                "max_err": 0.0,
            }
        wall = time.perf_counter() - t0
        result[f"{eng}_status"] = res["status"]
        result[f"{eng}_error"] = res.get("error", "")
        result[f"{eng}_max_err"] = res.get("max_err", 0.0)
        result[f"{eng}_wall_s"] = wall

    # Mirror bngsim to the bare keys for backward-compatible report consumers.
    if "bngsim" in engines:
        result["status"] = result["bngsim_status"]
        result["error"] = result["bngsim_error"]
        result["max_err"] = result["bngsim_max_err"]
    return result


def summarize_engine(results, eng) -> dict:
    """Per-engine summary over the SAME in-scope denominator. ``tested`` = cases
    this engine actually ran (in-scope and not engine-unavailable); ``pass`` =
    cases it got right. An engine whose library is absent has every case
    ``skipped`` ⇒ tested 0 ⇒ rendered TBD downstream, never a misleading 0%."""
    counts = Counter(r.get(f"{eng}_status", "not_run") for r in results)
    n_skip = counts["skipped"] + counts.get("not_run", 0)
    n_tested = len(results) - n_skip
    return {
        "tested": n_tested,
        "pass": counts["pass"],
        "load_fail": counts["load_fail"],
        "sim_fail": counts["sim_fail"],
        "value_mismatch": counts["value_mismatch"],
        "var_missing": counts["var_missing"],
        "shape_mismatch": counts["shape_mismatch"],
        "wall_s_passing": sum(
            float(r.get(f"{eng}_wall_s", 0.0)) for r in results if r.get(f"{eng}_status") == "pass"
        ),
        "wall_s_total": sum(float(r.get(f"{eng}_wall_s", 0.0)) for r in results),
        "available": n_tested > 0,
    }


def print_engine_summary(label, s):
    print(f"\n  {label}:")
    print(f"    Tested:         {s['tested']}")
    if s["tested"] > 0:
        print(f"    PASS:           {s['pass']} ({100 * s['pass'] / s['tested']:.1f}%)")
    else:
        print("    PASS:           — (engine unavailable)")
    print(f"    Load failures:  {s['load_fail']}")
    print(f"    Sim failures:   {s['sim_fail']}")
    print(f"    Value mismatch: {s['value_mismatch']}")
    print(f"    Var missing:    {s['var_missing']}")
    print(f"    Shape mismatch: {s['shape_mismatch']}")


def generate_markdown(payload: dict, outpath: Path) -> None:
    """Emit the compatibility + timing report as markdown."""
    lines = [
        "# `sbml_test_suite` suite results",
        "",
        f"- mode: `{payload['mode']}`  effort: `{payload['effort']}`",
        f"- engines: {', '.join(payload['engines'])}",
        f"- cases run: {payload['n_cases']}  (in-scope TimeCourse denominator, identical per engine)",
        "",
        "## Engine compatibility (correctness gate)",
        "",
        "Every engine is graded through the shared kernel (`_grading.py`) — same",
        "tolerances, same variable resolution, same amount conversion, same",
        "denominator. Differences below are engine differences, not harness ones.",
        "",
        "| Engine | tested | pass | pass % | load_fail | sim_fail | value_mismatch | var_missing | shape_mismatch |",
        "|--------|--------|------|--------|-----------|----------|----------------|-------------|----------------|",
    ]
    for eng in payload["engines"]:
        s = payload["summary"].get(eng)
        if not s:
            continue
        tested = s["tested"]
        pct = f"{100 * s['pass'] / tested:.1f}%" if tested else "—"
        lines.append(
            f"| {ENGINE_LABELS.get(eng, eng)} | {tested} | {s['pass']} | {pct} | "
            f"{s['load_fail']} | {s['sim_fail']} | {s['value_mismatch']} | "
            f"{s['var_missing']} | {s['shape_mismatch']} |"
        )
    lines += [
        "",
        "## Timing gate (wall time over each engine's passing cases)",
        "",
        "| Engine | passing cases | wall over passing (s) | wall total (s) |",
        "|--------|---------------|-----------------------|----------------|",
    ]
    for eng in payload["engines"]:
        s = payload["summary"].get(eng)
        if not s:
            continue
        lines.append(
            f"| {ENGINE_LABELS.get(eng, eng)} | {s['pass']} | "
            f"{s['wall_s_passing']:.1f} | {s['wall_s_total']:.1f} |"
        )
    lines += [
        "",
        "## Top feature tags — BNGsim (pass / tested)",
        "",
        "| Tag | pass | tested | pass % |",
        "|-----|------|--------|--------|",
    ]
    tags = payload["tags"]
    for tag in sorted(tags, key=lambda x: tags[x]["total"], reverse=True)[:20]:
        info = tags[tag]
        tested = info["pass"] + info["fail"]
        pct = 100 * info["pass"] / tested if tested else 0.0
        lines.append(f"| {tag} | {info['pass']} | {tested} | {pct:.1f}% |")
    lines.append("")
    outpath.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="sbml_test_suite suite — fair SBML semantic-suite compatibility + timing"
    )
    parser.add_argument(
        "--mode",
        choices=("correctness", "timing", "both"),
        default="both",
        help="Which gates to report (default: both). Cases are always run.",
    )
    parser.add_argument("--quick", type=int, default=0, help="Run only first N cases")
    parser.add_argument("--case", type=str, default="", help="Run a single case (e.g., 00001)")
    parser.add_argument(
        "--engines",
        type=str,
        default="bngsim",
        help="Comma list of bngsim,rr,amici,copasi — or 'all' for every engine.",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="",
        help="Comma-separated tag-prefix filter (e.g. 'Event' for the L3 event subset).",
    )
    add_effort_arg(parser)
    args = parser.parse_args()

    # Parse engines (preserve canonical report order; bngsim always first).
    if args.engines == "all":
        engines = list(ALL_ENGINES)
    else:
        requested = {e.strip() for e in args.engines.split(",") if e.strip()}
        unknown = requested - set(ADAPTERS)
        if unknown:
            print(f"ERROR: unknown engine(s): {', '.join(sorted(unknown))}")
            print(f"  known: {', '.join(ALL_ENGINES)}")
            sys.exit(2)
        engines = [e for e in ALL_ENGINES if e in requested]

    if not SUITE_DIR.exists():
        print(f"ERROR: SBML Test Suite not found at {SUITE_DIR}")
        print("  Set SBML_TEST_SUITE_DIR to a checkout's cases/semantic. For the pinned")
        print("  commit (reproduces our numbers): harness/sbml_test_suite/fetch_semantic_suite.py")
        print("  (pin: harness/sbml_test_suite/SUITE_PIN.json).")
        sys.exit(1)

    # Collect case directories
    if args.case:
        case_dirs = [SUITE_DIR / args.case]
    else:
        case_dirs = sorted(d for d in SUITE_DIR.iterdir() if d.is_dir() and d.name.isdigit())
        if args.tag_prefix:
            prefixes = [p.strip() for p in args.tag_prefix.split(",") if p.strip()]
            filtered = []
            for d in case_dirs:
                m_path = d / f"{d.name}-model.m"
                if not m_path.exists():
                    continue
                all_tags = parse_model_desc(m_path)["tag_list"]
                if any(t.startswith(p) for t in all_tags for p in prefixes):
                    filtered.append(d)
            case_dirs = filtered
        case_dirs = [d for d in case_dirs if effort_allows(args.effort, effort_tier(d.name))]
        if args.quick > 0:
            case_dirs = case_dirs[: args.quick]

    print(
        f"Running {len(case_dirs)} SBML test suite cases (mode={args.mode}, effort={args.effort})"
    )
    print(f"Engines: {', '.join(ENGINE_LABELS.get(e, e) for e in engines)}")

    results = []
    tag_pass = defaultdict(int)
    tag_fail = defaultdict(int)
    tag_total = defaultdict(int)

    _checkpoint_warned = False
    t0 = time.time()
    for i, case_dir in enumerate(case_dirs):
        case_id = case_dir.name
        try:
            r = run_case(case_dir, case_id, engines)
        except Exception as e:  # crash isolation: one bad case must not abort the sweep
            import traceback as _tb

            _tb.print_exc()
            r = {
                "case": case_id,
                "tags": [],
                "test_type": "TimeCourse",
                "status": "sim_fail",
                "error": f"uncaught {type(e).__name__}: {e}"[:200],
                "max_err": 0.0,
            }
            for eng in engines:
                r.setdefault(f"{eng}_status", "sim_fail")
        results.append(r)

        # Per-tag stats track the primary subject (bngsim) — the feature-gap view.
        for tag in r.get("tags", []):
            tag_total[tag] += 1
            if r.get("status") == "pass":
                tag_pass[tag] += 1
            elif r.get("status") in FAIL_STATUSES:
                tag_fail[tag] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_pass = sum(1 for x in results if x.get("status") == "pass")
            n_fail = sum(1 for x in results if x.get("status") in FAIL_STATUSES)
            print(
                f"  {i + 1}/{len(case_dirs)} done ({elapsed:.1f}s, pass={n_pass}, fail={n_fail})"
            )
            try:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                _partial = RESULTS_DIR / "sbml_test_suite_results.json.partial"
                _tmp = RESULTS_DIR / "sbml_test_suite_results.json.partial.tmp"
                _tmp.write_text(
                    json.dumps(
                        {"completed": i + 1, "total": len(case_dirs), "cases": results},
                        default=str,
                    )
                )
                _tmp.replace(_partial)
            except Exception as e:
                if not _checkpoint_warned:
                    print(f"  [checkpoint skipped: {type(e).__name__}: {e}]")
                    _checkpoint_warned = True

    elapsed = time.time() - t0

    # ── Per-engine summaries over the identical in-scope denominator ────────
    summary = {eng: summarize_engine(results, eng) for eng in engines}

    n_scope = next((s["tested"] for s in summary.values() if s["available"]), 0)
    n_skip = len(results) - n_scope
    print(f"\n{'=' * 64}")
    print("SBML Test Suite Results (fair / shared-grading; GH #225)")
    print(f"{'=' * 64}")
    print(f"Total cases:    {len(results)}")
    print(f"Out of scope:   {n_skip}  (non-TimeCourse / missing files — excluded for ALL engines)")
    print(f"In-scope denom: {n_scope}  (identical for every engine)")
    for eng in engines:
        print_engine_summary(ENGINE_LABELS.get(eng, eng), summary[eng])
    print(f"\n  Total sweep wall time: {elapsed:.1f}s")

    sorted_tags = sorted(tag_total.keys(), key=lambda t: tag_total[t], reverse=True)
    if args.mode in ("correctness", "both"):
        print(f"\n{'=' * 64}")
        print("Feature Tag Report — BNGsim (pass/tested)")
        print(f"{'=' * 64}")
        for tag in sorted_tags[:20]:
            passed, failed = tag_pass[tag], tag_fail[tag]
            tested = passed + failed
            pct = 100 * passed / tested if tested else 0
            print(f"  {tag:40s}  {passed:4d}/{tested:4d} ({pct:5.1f}%)  [total={tag_total[tag]}]")
        if len(sorted_tags) > 20:
            print(f"  ... and {len(sorted_tags) - 20} more tags (see JSON output)")

    if args.mode in ("timing", "both"):
        print(f"\n{'=' * 64}")
        print("Timing Gate (wall over each engine's passing cases)")
        print(f"{'=' * 64}")
        for eng in engines:
            s = summary[eng]
            print(
                f"  {ENGINE_LABELS.get(eng, eng):16s} {s['pass']:5d} passing  "
                f"{s['wall_s_passing']:8.1f}s over passing  {s['wall_s_total']:8.1f}s total"
            )

    payload = {
        "mode": args.mode,
        "effort": args.effort,
        "engines": engines,
        "n_cases": len(results),
        "n_in_scope": n_scope,
        "n_out_of_scope": n_skip,
        "summary": summary,
        "tags": {
            tag: {"total": tag_total[tag], "pass": tag_pass[tag], "fail": tag_fail[tag]}
            for tag in sorted_tags
        },
        "cases": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "sbml_test_suite_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    (RESULTS_DIR / "sbml_test_suite_results.json.partial").unlink(missing_ok=True)
    generate_markdown(payload, RESULTS_DIR / "sbml_test_suite_results.md")
    print(f"\nResults: {json_path}")
    print(f"Report:  {RESULTS_DIR / 'sbml_test_suite_results.md'}")


if __name__ == "__main__":
    main()
