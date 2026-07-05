#!/usr/bin/env python3
"""bngsim's authoritative-equivalent SBML Test Suite score (GH #241).

The official SBML Test Suite runner grades a tool by (1) invoking a wrapper to
produce a CSV per case, (2) comparing that CSV to the reference within the
per-case tolerances, and (3) classifying each case into a :class:`ResultType`
using the tool's declared *unsupported tags*. Building the runner's Java/JNI GUI
locally is infeasible, so this script reimplements steps 2–3 faithfully in
Python and drives bngsim through the *same* shared load+integrate+resolve code
the committed wrapper uses (``build_bngsim_series`` + ``resolve_columns``). The
number it reports is therefore the local equivalent of what the official runner
would show for bngsim.

Ported faithfully from the runner sources:

* **compare** — ``CompareResultSets.compare`` with ``requireAllColumns=true``:
  a delivered CSV missing any expected (non-time) column is a ``NoMatch``; the
  per-element test is ``|e-g| > absErr + relErr*|e|``; ``mismatchedBadValues``
  handles INF/NaN. The time axis (any column named case-insensitively "time") is
  copied, not compared.
* **outcome enum** — ``WrapperConfig.getResultTypeInternal`` /
  ``run``: no CSV + unsupported-tag match ⇒ ``Unsupported``; no CSV + no match ⇒
  ``Error``; CSV + tag match ⇒ ``CannotSolve``; CSV + no match ⇒ ``Match`` /
  ``NoMatch``.
* **tag match** — ``TestCase.matches`` / ``prefixMatch``: a declared bare tag
  matches an exact case tag OR any case tag whose ``prefix:suffix`` prefix
  equals it (so declaring ``fbc`` covers every ``fbc:*`` tag).

The unsupported-tag set is read from the committed manifest
(``bngsim-unsupported-tags.txt``), which a unit test pins to the SSOT
``bngsim._sbml_unsupported``.

Usage::

    python score.py                       # full suite (all cases)
    python score.py --case 00042          # one case
    python score.py --quick 100           # first 100 cases
    python score.py --effort low          # cheap strided subset
    python score.py --reconcile results/sbml_test_suite_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Shared grading kernel + engine adapter live one directory up; the fair-harness
# effort helper two more up (bngsim/benchmarks).
_SUITE_DIR = Path(__file__).resolve().parent.parent
_BENCH_ROOT = _SUITE_DIR.parents[1]  # bngsim/benchmarks
for _p in (str(_SUITE_DIR), str(_BENCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _effort import add_effort_arg, effort_allows  # noqa: E402
from _engines import BngsimStageError, build_bngsim_series  # noqa: E402
from _grading import (  # noqa: E402
    parse_results_csv,
    parse_settings,
    read_sbml_entities,
    resolve_columns,
)
from bngsim import _sbml_unsupported  # noqa: E402

DEFAULT_SUITE_DIR = Path(
    os.environ.get(
        "SBML_TEST_SUITE_DIR",
        os.path.expanduser("~/Code/sbml-test-suite/cases/semantic"),
    )
)
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "bngsim-unsupported-tags.txt"

# SBML levels newest-first — the runner's "Highest supported" selection.
SBML_PREF = ["l3v2", "l3v1", "l2v5", "l2v4", "l2v3", "l2v2", "l2v1", "l1v2"]

# ResultType labels (mirroring the official enum) plus the harness-only
# SKIPPED/UNAVAILABLE bookkeeping used when a case has no gradable files.
MATCH = "Match"
NOMATCH = "NoMatch"
CANNOTSOLVE = "CannotSolve"
UNSUPPORTED = "Unsupported"
ERROR = "Error"
UNAVAILABLE = "Unavailable"
SKIPPED = "Skipped"


# ── Case metadata (ported from run.py's parse_model_desc / find_sbml_file) ────


def parse_model_desc(model_m_path: Path) -> dict:
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
    out["tags"] = out["componentTags"] + out["testTags"]
    return out


def find_sbml_file(case_dir: Path, case_id: str):
    for level in SBML_PREF:
        path = case_dir / f"{case_id}-sbml-{level}.xml"
        if path.exists():
            return path
    return None


# ── Faithful ports of the runner's tag-match and compare ──────────────────────


def matches_unsupported(case_tags, unsupported_tags) -> bool:
    """``TestCase.matches`` + ``prefixMatch``: a declared tag matches an exact
    case tag, or any case tag whose ``prefix:suffix`` prefix equals it."""
    prefixes = {t.split(":", 1)[0] for t in case_tags if ":" in t}
    case_set = set(case_tags)
    return any(tag in case_set or tag in prefixes for tag in unsupported_tags)


def _mismatched_bad(a: float, b: float) -> bool:
    """``CompareResultSets.mismatchedBadValues``: exactly one of a/b is INF, or
    exactly one is NaN."""
    return (
        (np.isinf(a) and not np.isinf(b))
        or (np.isinf(b) and not np.isinf(a))
        or (np.isnan(a) and not np.isnan(b))
        or (np.isnan(b) and not np.isnan(a))
    )


def compare_official(exp_data: dict, delivered: dict, n_rows: int, abs_err: float, rel_err: float):
    """Faithful port of ``CompareResultSets.compare`` (requireAllColumns=true).

    ``exp_data`` maps expected column name → values (time axis already excluded
    by ``parse_results_csv``); ``delivered`` maps resolved var → values. Returns
    ``(is_match, max_rel_err)``. A missing column, too few delivered rows, a
    bad-value mismatch, or an out-of-tolerance element makes it a NoMatch.
    """
    match = True
    max_rel = 0.0
    for col, e_vals in exp_data.items():
        # The runner copies (does not compare) any column named "time"
        # case-insensitively — the independent axis.
        if col.lower() == "time":
            continue
        if col not in delivered:  # requireAllColumns
            return False, float("inf")
        d_vals = delivered[col]
        if len(d_vals) < n_rows:  # delivered has fewer rows than expected
            return False, float("inf")
        e = np.asarray(e_vals, dtype=float)[:n_rows]
        g = np.asarray(d_vals, dtype=float)[:n_rows]
        diff = np.abs(e - g)
        # rel-err metric for the report (never the verdict)
        denom = np.maximum(np.abs(e), abs_err)
        finite = np.isfinite(e) & np.isfinite(g)
        if np.any(finite):
            max_rel = max(max_rel, float(np.max(diff[finite] / denom[finite])))
        for i in range(n_rows):
            if _mismatched_bad(e[i], g[i]) or diff[i] > (abs_err + rel_err * abs(e[i])):
                match = False
                break
        if not match:
            break
    return match, max_rel


# ── Per-case scoring ──────────────────────────────────────────────────────────


def score_case(case_dir: Path, case_id: str, unsupported_tags) -> dict:
    """Run + classify one case into a :class:`ResultType`-equivalent outcome."""
    desc = parse_model_desc(case_dir / f"{case_id}-model.m")
    tags = desc["tags"]
    out = {
        "case": case_id,
        "test_type": desc["testType"],
        "tags": tags,
        "matches_unsupported": matches_unsupported(tags, unsupported_tags),
        "outcome": None,
        "max_err": 0.0,
        "detail": "",
    }

    sbml_path = find_sbml_file(case_dir, case_id)
    settings_path = case_dir / f"{case_id}-settings.txt"
    csv_path = case_dir / f"{case_id}-results.csv"

    if sbml_path is None:
        out["outcome"] = UNAVAILABLE
        out["detail"] = "no SBML file at any level"
        return out

    # bngsim is a time-course ODE engine: it does not attempt non-TimeCourse
    # test types (e.g. FluxBalanceSteadyState). No delivered result ⇒ the
    # outcome enum classifies by tag (all fbc cases carry an fbc:* tag).
    if desc["testType"] != "TimeCourse":
        out["outcome"] = UNSUPPORTED if out["matches_unsupported"] else ERROR
        out["detail"] = f"testType={desc['testType']} (not attempted)"
        return out

    if not settings_path.exists() or not csv_path.exists():
        out["outcome"] = SKIPPED
        out["detail"] = "missing settings or results CSV"
        return out

    settings = parse_settings(settings_path)
    n_rows = settings["steps"] + 1

    # Deliver bngsim's result through the SAME shared path as the wrapper.
    delivered = None
    try:
        series, _ = build_bngsim_series(sbml_path, settings)
        ent = read_sbml_entities(sbml_path)
        delivered = resolve_columns(settings, series, ent)
    except BngsimStageError as e:
        out["detail"] = f"{e.stage}: {str(e)[:150]}"
    except Exception as e:  # noqa: BLE001 — any internal error ⇒ fail closed
        out["detail"] = f"error: {type(e).__name__}: {str(e)[:150]}"

    if delivered is None:  # no CSV would have been written (fail closed)
        out["outcome"] = UNSUPPORTED if out["matches_unsupported"] else ERROR
        return out

    # CSV present. Tag-excluded cases are ignored (CannotSolve) even if numbers
    # would match, mirroring the official runner.
    if out["matches_unsupported"]:
        out["outcome"] = CANNOTSOLVE
        return out

    try:
        _, exp_data = parse_results_csv(csv_path, settings=settings)
    except Exception as e:  # unparseable/mismatched reference ⇒ Error (official)
        out["outcome"] = ERROR
        out["detail"] = f"reference parse: {type(e).__name__}: {str(e)[:120]}"
        return out

    is_match, max_rel = compare_official(
        exp_data, delivered, n_rows, settings["absolute"], settings["relative"]
    )
    out["max_err"] = max_rel
    out["outcome"] = MATCH if is_match else NOMATCH
    return out


# ── Reconciliation against the fair harness JSON ──────────────────────────────


def reconcile(results: list[dict], fair_json_path: Path) -> dict:
    """Compare this scorer's pass/fail axis to the fair harness's bngsim verdict.

    Both should agree on whether bngsim gets a case right. Divergences are
    expected only where the *comparison* rules differ (e.g. the official runner
    treats a sign-flipped infinity as a match, and skips any column named "time")
    and are listed so they can be inspected, not silently absorbed.
    """
    fair = json.loads(fair_json_path.read_text())
    fair_status = {c["case"]: c.get("bngsim_status", c.get("status")) for c in fair["cases"]}
    score_pass = {r["case"]: (r["outcome"] == MATCH) for r in results}

    agree = disagree = 0
    only_score_pass = []
    only_fair_pass = []
    for case, sp in score_pass.items():
        if case not in fair_status:
            continue
        fp = fair_status[case] == "pass"
        if sp == fp:
            agree += 1
        else:
            disagree += 1
            (only_score_pass if sp else only_fair_pass).append(case)
    return {
        "compared": agree + disagree,
        "agree": agree,
        "disagree": disagree,
        "only_score_pass": sorted(only_score_pass),
        "only_fair_pass": sorted(only_fair_pass),
    }


# ── Orchestration ─────────────────────────────────────────────────────────────


def effort_tier(case_id: str) -> str:
    try:
        n = int(case_id)
    except ValueError:
        return "high"
    if n % 20 == 0:
        return "low"
    if n % 4 == 0:
        return "medium"
    return "high"


def main() -> int:
    ap = argparse.ArgumentParser(description="Authoritative-equivalent SBML Test Suite score for bngsim")
    ap.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--case", type=str, default="", help="Score a single case (e.g. 00001)")
    ap.add_argument("--quick", type=int, default=0, help="First N cases after filters")
    ap.add_argument("--reconcile", type=Path, default=None, help="Fair-harness results JSON to diff against")
    ap.add_argument("--json", type=Path, default=Path(__file__).resolve().parent / "score_results.json")
    add_effort_arg(ap)
    args = ap.parse_args()

    if not args.suite_dir.exists():
        print(f"ERROR: SBML Test Suite not found at {args.suite_dir}")
        print("  Set SBML_TEST_SUITE_DIR or pass --suite-dir.")
        return 1
    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest} (run gen_manifest.py)")
        return 1

    unsupported_tags = _sbml_unsupported.parse_manifest(args.manifest.read_text())
    print(f"Unsupported tags (from manifest): {', '.join(unsupported_tags)}")

    if args.case:
        case_dirs = [args.suite_dir / args.case]
    else:
        case_dirs = sorted(d for d in args.suite_dir.iterdir() if d.is_dir() and d.name.isdigit())
        case_dirs = [d for d in case_dirs if effort_allows(args.effort, effort_tier(d.name))]
        if args.quick > 0:
            case_dirs = case_dirs[: args.quick]

    print(f"Scoring {len(case_dirs)} cases (effort={args.effort}) …")
    results = []
    t0 = time.time()
    for i, case_dir in enumerate(case_dirs):
        results.append(score_case(case_dir, case_dir.name, unsupported_tags))
        if (i + 1) % 100 == 0:
            n_match = sum(1 for r in results if r["outcome"] == MATCH)
            print(f"  {i + 1}/{len(case_dirs)}  ({time.time() - t0:.0f}s, Match={n_match})")

    counts = Counter(r["outcome"] for r in results)
    # Denominators. "Attempted" = cases bngsim actually ran and was not excused
    # from (Match + NoMatch + Error). In-scope TimeCourse mirrors the fair
    # harness denominator for a like-for-like reconciliation.
    attempted = counts[MATCH] + counts[NOMATCH] + counts[ERROR]
    timecourse = [r for r in results if r["test_type"] == "TimeCourse" and r["outcome"] != SKIPPED]
    tc_match = sum(1 for r in timecourse if r["outcome"] == MATCH)

    print(f"\n{'=' * 64}")
    print("bngsim — SBML Test Suite score (local official-equivalent; GH #241)")
    print(f"{'=' * 64}")
    print(f"Cases scored:    {len(results)}")
    for label in (MATCH, NOMATCH, CANNOTSOLVE, UNSUPPORTED, ERROR, UNAVAILABLE, SKIPPED):
        print(f"  {label:12s} {counts.get(label, 0)}")
    if attempted:
        print(f"\n  Correct among attempted (Match/(Match+NoMatch+Error)): "
              f"{counts[MATCH]}/{attempted} = {100 * counts[MATCH] / attempted:.1f}%")
    if timecourse:
        print(f"  In-scope TimeCourse Match: {tc_match}/{len(timecourse)} "
              f"= {100 * tc_match / len(timecourse):.1f}%")

    # Error cases are the ones to look at first: bngsim failed to produce a
    # correct result AND had no unsupported-tag excuse.
    errors = [r for r in results if r["outcome"] == ERROR]
    if errors:
        print(f"\n  {len(errors)} Error case(s) (no CSV / unparseable, no tag excuse):")
        for r in errors[:25]:
            print(f"    {r['case']}  {r['detail']}")
        if len(errors) > 25:
            print(f"    … and {len(errors) - 25} more")

    payload = {
        "suite_dir": str(args.suite_dir),
        "manifest": str(args.manifest),
        "unsupported_tags": unsupported_tags,
        "effort": args.effort,
        "counts": dict(counts),
        "attempted": attempted,
        "match": counts[MATCH],
        "timecourse_total": len(timecourse),
        "timecourse_match": tc_match,
        "cases": results,
    }

    if args.reconcile:
        if not args.reconcile.exists():
            print(f"\n  [reconcile skipped: {args.reconcile} not found]")
        else:
            rec = reconcile(results, args.reconcile)
            payload["reconcile"] = rec
            print(f"\n{'=' * 64}")
            print(f"Reconciliation vs fair harness ({args.reconcile.name})")
            print(f"{'=' * 64}")
            print(f"  compared: {rec['compared']}  agree: {rec['agree']}  disagree: {rec['disagree']}")
            if rec["only_score_pass"]:
                print(f"  Match here but fail in fair harness ({len(rec['only_score_pass'])}): "
                      f"{', '.join(rec['only_score_pass'][:20])}")
            if rec["only_fair_pass"]:
                print(f"  Pass in fair harness but not Match here ({len(rec['only_fair_pass'])}): "
                      f"{', '.join(rec['only_fair_pass'][:20])}")

    args.json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResults: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
