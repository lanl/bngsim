#!/usr/bin/env python3
"""Diff a fresh SSA-screen run against the committed baseline (GH #108).

The regression the screen tracks is a model crossing from an EXPECTED state into
a scoring DIFF/EXCEPTION — the PASS↔FAIL move, with ``rr_known`` /
``diff_not_bngsim`` / ``too_slow`` (TIMEOUT) / a RoadRunner-side
``REFERENCE_FAILED`` all treated as expected classes (see
:func:`ssa_attribution.is_expected`). Every comparison keys on ``model_id`` and
uses :func:`ssa_attribution.is_regression`, so this stays a thin policy-driven
diff rather than a second verdict.

Categories reported:

  * ``regressions``         expected in the baseline, not expected now — FAILS.
  * ``improvements``        not expected in the baseline, expected now — suggests
                            a deliberate re-baseline (``make_ssa_baseline.py``).
  * ``changed_attribution`` both expected, but (outcome, subclass) changed — green
                            churn (e.g. pass → diff_not_bngsim); informational.
  * ``new_models``          present in the fresh run, absent from the baseline.
  * ``not_in_fresh``        in the baseline, absent from the fresh run — a partial
                            run (a slice/effort filter) or a shrunk corpus.

Only ``regressions`` set a non-zero exit code. ``--fresh`` accepts either a
``_core`` report (``runs/ssa_report.json``) or a native screen JSON
(``runs/ssa_screen.json`` / a legacy result), detected by shape.

Usage:
    python diff_ssa_baseline.py [--baseline ssa_baseline.json]
                                [--fresh runs/ssa_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so `import ssa_attribution` resolves
import ssa_attribution as sa  # noqa: E402
from _core import JobResult  # noqa: E402


def load_results(path: str | Path) -> list[JobResult]:
    """Load a list[JobResult] from a _core report OR a native screen JSON."""
    data = json.loads(Path(path).read_text())
    if "results" in data:  # a _core report
        return [JobResult.from_dict(r) for r in data["results"]]
    if "cases" in data:  # a native screen payload
        _meta, results = sa.payload_to_report(data)
        return results
    raise SystemExit(f"unrecognized report shape (no 'results' or 'cases'): {path}")


def _state(r: JobResult) -> str:
    return r.outcome + (f"/{r.subclass}" if r.subclass else "")


def diff_reports(baseline: list[JobResult], fresh: list[JobResult]) -> dict:
    """Categorize a fresh run against a baseline. Pure — no I/O, unit-testable."""
    b = {r.model_id: r for r in baseline}
    f = {r.model_id: r for r in fresh}

    regressions, improvements, churn = [], [], []
    for mid in sorted(b.keys() & f.keys()):
        br, fr = b[mid], f[mid]
        b_exp = sa.is_expected(br.outcome, br.subclass)
        f_exp = sa.is_expected(fr.outcome, fr.subclass)
        if b_exp and not f_exp:
            regressions.append(
                {
                    "model_id": mid,
                    "from": _state(br),
                    "to": _state(fr),
                    "why": fr.comment or fr.exception,
                }
            )
        elif not b_exp and f_exp:
            improvements.append({"model_id": mid, "from": _state(br), "to": _state(fr)})
        elif (br.outcome, br.subclass) != (fr.outcome, fr.subclass):
            churn.append({"model_id": mid, "from": _state(br), "to": _state(fr)})

    return {
        "regressions": regressions,
        "improvements": improvements,
        "changed_attribution": churn,
        "new_models": sorted(f.keys() - b.keys()),
        "not_in_fresh": sorted(b.keys() - f.keys()),
    }


def _print_report(d: dict) -> None:
    def section(title, items, render):
        if not items:
            return
        print(f"\n{title} ({len(items)}):")
        for it in items:
            print(f"  {render(it)}")

    print("=" * 72)
    print("  SSA screen regression diff (fresh vs committed baseline)")
    print("=" * 72)
    section(
        "REGRESSIONS — newly broken (FAIL)",
        d["regressions"],
        lambda r: f"{r['model_id']}: {r['from']} -> {r['to']}  {r['why'][:80]}",
    )
    section(
        "improvements — newly green (consider re-baselining)",
        d["improvements"],
        lambda r: f"{r['model_id']}: {r['from']} -> {r['to']}",
    )
    section(
        "attribution churn — green->green (informational)",
        d["changed_attribution"],
        lambda r: f"{r['model_id']}: {r['from']} -> {r['to']}",
    )
    if d["new_models"]:
        print(
            f"\nnew models (not in baseline) ({len(d['new_models'])}): {', '.join(d['new_models'])}"
        )
    if d["not_in_fresh"]:
        print(f"\nnot in fresh run ({len(d['not_in_fresh'])}) — partial run or shrunk corpus")
    print("\n" + "=" * 72)
    verdict = "REGRESSION" if d["regressions"] else "clean"
    print(
        f"  {verdict}: {len(d['regressions'])} regression(s), {len(d['improvements'])} improvement(s)"
    )
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=HERE / "ssa_baseline.json")
    ap.add_argument("--fresh", type=Path, default=HERE / "runs" / "ssa_report.json")
    args = ap.parse_args()

    for p in (args.baseline, args.fresh):
        if not p.exists():
            raise SystemExit(f"missing report: {p}")

    d = diff_reports(load_results(args.baseline), load_results(args.fresh))
    _print_report(d)
    return 1 if d["regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
