#!/usr/bin/env python3
"""SSA roundtrip: ``Model.from_net`` vs ``Model.from_sbml`` distributional parity.

For each model in ``benchmarks/ssa/<name>.net`` paired with
``benchmarks/suites/sbml_roundtrip/<name>.xml``, run the same seed schedule on both
sides for ``--n`` replicates (default 200) and apply
``scipy.stats.ks_2samp`` per (species column, time point). A model
passes when ``min(p) > --alpha`` (default 0.01) across all columns and
time points.

Species are matched **by column index**: BNG's ``writeSBML`` preserves
the .net species ordering, and the loader smoke test in
``benchmarks/suites/sbml_roundtrip/load_check.json`` plus a t=0 initial-value
equality check confirm columns align across the tracked corpus.

Usage:
    cd bngsim && uv run python harness/run_ssa_roundtrip.py
                                                    [--n 200] [--alpha 0.01]
                                                    [--include-large] [--models simple_system,...]

Output:
    results/ssa_roundtrip.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BENCHMARKS_DIR,
    SUITE_SSA,
    get_machine_info,
    load_suite,
    save_results,
)

# Skipped under the default invocation. Two reasons land here:
#   - benchmarks/suites/sbml_roundtrip/.gitignore — generated SBML exceeds the
#     repo's 500 KB pre-commit cap (egfr_net, fceri_gamma,
#     multisite_phos, prion_aggregation); regenerate via
#     benchmarks/suites/sbml_roundtrip/run.py before --include-large.
#   - exact-SSA wall time — erk_activation has populations up to 3e6,
#     so a single replicate is O(billions of events).
LARGE_MODELS = {
    "egfr_net",
    "fceri_gamma",
    "multisite_phos",
    "prion_aggregation",
    "erk_activation",
}

DEFAULT_N = 200
DEFAULT_SEED_BASE = 2000
DEFAULT_ALPHA = 0.01


def _replicate_array(
    loader,  # callable -> bngsim.Model
    t_end: float,
    n_steps: int,
    n_replicates: int,
    seed_base: int,
) -> tuple[np.ndarray, list[str], int]:
    """Return (species [n_rep, n_time, n_species], species_names, n_completed)."""
    import bngsim

    model = loader()
    n_time = n_steps + 1

    out: list[np.ndarray] = []
    species_names: list[str] = []
    for rep in range(n_replicates):
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ssa")
        r = sim.run(t_span=(0, t_end), n_points=n_time, seed=seed_base + rep)
        s = np.asarray(r.species)
        if not species_names:
            species_names = list(r.species_names)
        out.append(s)
    return np.stack(out, axis=0), species_names, len(out)


def _run_one(model_meta: dict, n_replicates: int, seed_base: int, alpha: float) -> dict:
    import bngsim
    from scipy.stats import ks_2samp

    name = model_meta["name"]
    t_end = float(model_meta["t_end"])
    n_steps = int(model_meta["n_steps"])

    net_path = (BENCHMARKS_DIR / model_meta["net_file"]).resolve()
    xml_path = (BENCHMARKS_DIR / "suites" / "sbml_roundtrip" / f"{name}.xml").resolve()

    entry: dict = {
        "name": name,
        "net_file": str(net_path.relative_to(BENCHMARKS_DIR)),
        "sbml_file": str(xml_path.relative_to(BENCHMARKS_DIR)) if xml_path.exists() else None,
        "t_end": t_end,
        "n_steps": n_steps,
        "species_declared": int(model_meta.get("species", -1)),
        "n_replicates_completed_net": 0,
        "n_replicates_completed_sbml": 0,
        "elapsed_sec_net": 0.0,
        "elapsed_sec_sbml": 0.0,
        "elapsed_sec_ks": 0.0,
        "ks_min_p": None,
        "n_ks_failures": 0,
        "ks_failures": [],
        "status": "skip",
        "error": None,
    }

    if not xml_path.exists():
        entry["error"] = f"missing xml: {xml_path}; run benchmarks/suites/sbml_roundtrip/run.py"
        return entry

    try:
        t0 = time.perf_counter()
        net_arr, net_names, n_net = _replicate_array(
            lambda: bngsim.Model.from_net(str(net_path)),
            t_end,
            n_steps,
            n_replicates,
            seed_base,
        )
        entry["elapsed_sec_net"] = round(time.perf_counter() - t0, 3)
        entry["n_replicates_completed_net"] = n_net
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"net side: {type(exc).__name__}: {exc}"[:300]
        return entry

    try:
        t0 = time.perf_counter()
        sbml_arr, sbml_names, n_sbml = _replicate_array(
            lambda: bngsim.Model.from_sbml(str(xml_path)),
            t_end,
            n_steps,
            n_replicates,
            seed_base,
        )
        entry["elapsed_sec_sbml"] = round(time.perf_counter() - t0, 3)
        entry["n_replicates_completed_sbml"] = n_sbml
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"sbml side: {type(exc).__name__}: {exc}"[:300]
        return entry

    if net_arr.shape[1:] != sbml_arr.shape[1:]:
        entry["status"] = "error"
        entry["error"] = f"shape mismatch net={net_arr.shape} sbml={sbml_arr.shape}"
        return entry

    n_species_axis = net_arr.shape[2]
    n_time_axis = net_arr.shape[1]

    # Sanity: t=0 columns should be element-equal (initial conditions
    # come from the same model; SSA dispatch starts at t=0 with no
    # advance until the first sample step). If not, position alignment
    # is invalid and KS results are meaningless.
    t0_net = net_arr[0, 0, :]
    t0_sbml = sbml_arr[0, 0, :]
    if not np.array_equal(t0_net, t0_sbml):
        entry["status"] = "error"
        entry["error"] = (
            f"t=0 column mismatch (positional alignment broken); "
            f"net[0,0]={t0_net[:5].tolist()} vs sbml[0,0]={t0_sbml[:5].tolist()}"
        )
        return entry

    # KS test per (time_idx, species_idx) — skip t=0 (degenerate).
    t_ks0 = time.perf_counter()
    failures: list[dict] = []
    min_p = 1.0
    for ti in range(1, n_time_axis):
        for si in range(n_species_axis):
            a = net_arr[:, ti, si]
            b = sbml_arr[:, ti, si]
            # If both columns are identically constant, KS is degenerate
            # and ks_2samp returns p=1.0; that is fine.
            stat, p = ks_2samp(a, b)
            if p < min_p:
                min_p = float(p)
            if p < alpha:
                failures.append(
                    {
                        "t_idx": int(ti),
                        "species_idx": int(si),
                        "species_name_net": net_names[si],
                        "species_name_sbml": sbml_names[si],
                        "ks_stat": float(stat),
                        "p": float(p),
                    }
                )
    entry["elapsed_sec_ks"] = round(time.perf_counter() - t_ks0, 3)
    entry["ks_min_p"] = min_p
    entry["n_ks_failures"] = len(failures)
    # Cap stored failures to keep results.json bounded; record total in n_ks_failures.
    entry["ks_failures"] = failures[:50]
    entry["status"] = "pass" if len(failures) == 0 else "fail"
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n", type=int, default=DEFAULT_N, help="Replicate count per side (default 200)."
    )
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE, help="Seed schedule base.")
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Per-(time,species) KS p-value threshold; pass = min(p) > alpha.",
    )
    p.add_argument(
        "--include-large",
        action="store_true",
        help=(
            "Also test the four .gitignored large models "
            "(egfr_net, fceri_gamma, multisite_phos, prion_aggregation). "
            "Run benchmarks/suites/sbml_roundtrip/run.py first."
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names to run (overrides --include-large).",
    )
    args = p.parse_args()

    models = load_suite(SUITE_SSA)

    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m["name"] in wanted]
    elif not args.include_large:
        models = [m for m in models if m["name"] not in LARGE_MODELS]

    print("=" * 70)
    print("  SSA roundtrip: from_net vs from_sbml (KS_2samp)")
    print("=" * 70)
    print(f"  models:      {len(models)}")
    print(f"  replicates:  {args.n} per side")
    print(f"  seed_base:   {args.seed_base}")
    print(f"  alpha:       {args.alpha}")
    print()

    info = get_machine_info()
    started_total = time.perf_counter()
    cases = []
    for i, m in enumerate(models, 1):
        print(
            f"  [{i}/{len(models)}] {m['name']} (sp={m.get('species', '?')}, t_end={m['t_end']})"
        )
        entry = _run_one(m, n_replicates=args.n, seed_base=args.seed_base, alpha=args.alpha)
        if entry["status"] == "error":
            print(f"      ERR: {entry['error']}")
        elif entry["status"] == "skip":
            print(f"      SKIP: {entry['error']}")
        else:
            print(
                f"      {entry['status'].upper()} (min_p={entry['ks_min_p']:.3g}, "
                f"ks_fail={entry['n_ks_failures']}, "
                f"net={entry['elapsed_sec_net']}s, sbml={entry['elapsed_sec_sbml']}s)"
            )
        cases.append(entry)

    elapsed_total = time.perf_counter() - started_total
    n_pass = sum(1 for c in cases if c["status"] == "pass")
    n_fail = sum(1 for c in cases if c["status"] == "fail")
    n_error = sum(1 for c in cases if c["status"] == "error")
    n_skip = sum(1 for c in cases if c["status"] == "skip")

    print()
    print("=" * 70)
    print(
        f"  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_error}  SKIP: {n_skip}  "
        f"elapsed: {elapsed_total:.1f}s"
    )
    print("=" * 70)

    payload = {
        "machine_info": info,
        "config": {
            "n_replicates": args.n,
            "seed_base": args.seed_base,
            "ks_alpha": args.alpha,
            "include_large": args.include_large,
            "match_strategy": "by_column_index",
        },
        "summary": {
            "total": len(cases),
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "skip": n_skip,
            "elapsed_sec": round(elapsed_total, 2),
        },
        "cases": {c["name"]: c for c in cases},
    }
    save_results(payload, "ssa_roundtrip")
    return 0 if (n_fail + n_error) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
