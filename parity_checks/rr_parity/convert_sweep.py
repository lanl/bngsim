#!/usr/bin/env python3
"""GH #224 converter re-sweep: measure the cBNGL+actions faithful-lift over the
ODE-verified rr_parity corpus, and audit refused-by-both over-refusals.

For every model that PASSED the bngsim↔RoadRunner ODE gate (report_ode.json), we
attempt BOTH serialization targets and record representability:

  .net  : sbml_to_net(strict=False, validate="L2") — the in-tree RHS-identity
          self-check is authoritative (#223). faithful = no capability `lossy`
          notes AND rhs_faithful is True.
  cBNGL : sbml_to_bngl(strict=False) — there is no in-tree cBNGL reader, so the
          BNG2.pl round-trip gate lives in the test suite; here "accepted" = the
          capability check passed (no `lossy`). These are *projected* faithful
          (the writer logic is BNG2.pl-proven per model class, not round-tripped
          individually in this sweep).

Headline metric: combined faithful = .net-faithful OR cBNGL-accepted, vs the
flat-.net-only baseline, over the same denominator.

Usage:
    .venv/bin/python parity_checks/rr_parity/convert_sweep.py \
        [--report runs/report_ode.json] [--out runs/convert_sweep.json] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent


def classify_net(rep) -> tuple[str, str]:
    """Return (status, reason) for the .net target."""
    if rep.lossy:
        return "refused", "; ".join(rep.lossy)
    if getattr(rep, "rhs_faithful", None) is False:
        return "unfaithful", "rhs self-check diverged"
    return "faithful", ""


def classify_bngl(rep) -> tuple[str, str]:
    """Return (status, reason) for the cBNGL target."""
    if rep.lossy:
        return "refused", "; ".join(rep.lossy)
    return "accepted", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(HERE / "runs" / "report_ode.json"))
    ap.add_argument("--out", default=str(HERE / "runs" / "convert_sweep.json"))
    ap.add_argument("--jobs", default=str(HERE / "ode_jobs.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from bngsim.convert import sbml_to_net, sbml_to_bngl

    report = json.load(open(args.report))["results"]
    jobs = {j["model_id"]: j for j in json.load(open(args.jobs))["jobs"]}

    # The ODE-verified subset = models bngsim and RoadRunner agree on.
    passed = [r["model_id"] for r in report if r["outcome"] == "PASS"]
    if args.limit:
        passed = passed[: args.limit]

    rows = []
    for i, mid in enumerate(passed):
        job = jobs.get(mid)
        if job is None:
            continue
        sbml = HERE / job["model"]
        if not sbml.exists():
            rows.append({"model_id": mid, "net": "missing", "bngl": "missing"})
            continue

        row = {"model_id": mid}
        # .net target
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = sbml_to_net(sbml, strict=False, validate="L2")
            st, reason = classify_net(rep)
            row["net"] = st
            row["net_reason"] = reason
        except Exception as e:  # load fail / unexpected
            row["net"] = "error"
            row["net_reason"] = f"{type(e).__name__}: {e}"
        # cBNGL target
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = sbml_to_bngl(sbml, strict=False)
            st, reason = classify_bngl(rep)
            row["bngl"] = st
            row["bngl_reason"] = reason
        except Exception as e:
            row["bngl"] = "error"
            row["bngl_reason"] = f"{type(e).__name__}: {e}"
        rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(passed)}", file=sys.stderr, flush=True)

    # ── headline tallies ──
    n = len(rows)
    net_faithful = [r for r in rows if r.get("net") == "faithful"]
    bngl_ok = [r for r in rows if r.get("bngl") == "accepted"]
    combined = [
        r
        for r in rows
        if r.get("net") == "faithful" or r.get("bngl") == "accepted"
    ]
    refused_both = [
        r
        for r in rows
        if r.get("net") in ("refused", "unfaithful", "error")
        and r.get("bngl") in ("refused", "error")
    ]

    summary = {
        "denominator": n,
        "net_faithful": len(net_faithful),
        "net_faithful_pct": round(100 * len(net_faithful) / n, 1) if n else 0,
        "bngl_accepted": len(bngl_ok),
        "bngl_accepted_pct": round(100 * len(bngl_ok) / n, 1) if n else 0,
        "combined_faithful": len(combined),
        "combined_faithful_pct": round(100 * len(combined) / n, 1) if n else 0,
        "refused_both": len(refused_both),
        "recovered_by_cbngl_only": len(
            [
                r
                for r in rows
                if r.get("bngl") == "accepted" and r.get("net") != "faithful"
            ]
        ),
    }

    out = {"summary": summary, "refused_both": refused_both, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
