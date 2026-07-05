#!/usr/bin/env python3
"""bngsim SSA vs libroadrunner gillespie: per-time mean z-score parity.

For each model in ``benchmarks/suites/sbml_roundtrip/<name>.xml`` (the 7 SSA-roundtrip
SBML benchmark fixtures), run ``--n`` replicates per backend at the same
seed schedule and check that, at every (time, species) coordinate with
t > 0, the absolute z-score of the difference of sample means

    mean_z = |mu_bn - mu_rr| / sqrt(var_bn/N + var_rr/N)

does not exceed ``--mean-z-tol`` (default 5.0). 5σ corresponds to a
~3e-7 per-test probability under H0, comfortably surviving Bonferroni
over the per-model number of (time, species) cells we test.

Mirrors ``tests/test_bngsim_ssa_replaces_rr.py``'s mean-z-score gate but
operates from the bngsim side directly (no PyBNF wrapper) so this
harness can run inside the bngsim repo's venv.

Usage:
    cd bngsim && uv run python harness/run_rr_ssa_trajectory_parity.py
                                  [--n 200] [--mean-z-tol 5.0]
                                  [--models simple_system,...]
                                  [--include-large]

Output:
    results/rr_ssa_trajectory_parity.json
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

# Same exclusion list as run_ssa_roundtrip.py; large populations or
# .gitignored generated SBML.
LARGE_MODELS = {
    "egfr_net",
    "fceri_gamma",
    "multisite_phos",
    "prion_aggregation",
    "erk_activation",
}

# Models the user wants to start with: the four small ones with t_end
# small enough that N=200 replicates per side fits in interactive time.
SMALL_MODELS = {
    "gene_expression_hill",
    "simple_system",
    "flagellar_motor",
    "oscillatory_system",
}

DEFAULT_N = 200
DEFAULT_SEED_BASE = 2000
DEFAULT_MEAN_Z_TOL = 5.0
DEFAULT_MAX_FAILURES_RECORDED = 50


def _bn_replicates(
    xml_path: str,
    t_end: float,
    n_steps: int,
    n_replicates: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Run ``n_replicates`` bngsim SSA replicates.

    Returns ``(times[n_time], species[n_rep, n_time, n_species], names, ssa_diag)``.
    ``ssa_diag`` aggregates the per-replicate GH #110 SSA boundary diagnostics
    over the replicate set: ``n_reverse_fires`` / ``n_negative_crossings`` summed,
    plus the first non-empty offending reaction / species. ``n_reverse_fires > 0``
    means a rate law evaluated negative somewhere on the trajectory and bngsim ran
    the channel backward with propensity ``|rate|`` (mean-faithful) -- i.e. the
    model is *runtime* sign-indefinite (GH #109), which the static t0 corpus gate
    cannot see.
    """
    import warnings

    import bngsim

    model = bngsim.Model.from_sbml(xml_path)
    out: list[np.ndarray] = []
    species_names: list[str] = []
    times: np.ndarray | None = None
    ssa_diag = {
        "n_reverse_fires": 0,
        "first_reverse_reaction": "",
        "n_negative_crossings": 0,
        "first_negative_species": "",
    }
    for rep in range(n_replicates):
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ssa")
        # The GH #110 SsaBoundaryWarning fires once per replicate on a
        # sign-indefinite model (30x noise per case). We capture the same signal
        # structurally via r.ssa_diagnostics below and surface it once, cleanly,
        # as the GH #109 annotation -- so silence the raw warning here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
            r = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed_base + rep)
        if times is None:
            times = np.asarray(r.time)
            species_names = list(r.species_names)
        out.append(np.asarray(r.species))
        d = r.ssa_diagnostics
        ssa_diag["n_reverse_fires"] += int(d.get("n_reverse_fires", 0))
        ssa_diag["n_negative_crossings"] += int(d.get("n_negative_crossings", 0))
        if not ssa_diag["first_reverse_reaction"] and d.get("first_reverse_reaction"):
            ssa_diag["first_reverse_reaction"] = d["first_reverse_reaction"]
        if not ssa_diag["first_negative_species"] and d.get("first_negative_species"):
            ssa_diag["first_negative_species"] = d["first_negative_species"]
    return times, np.stack(out, axis=0), species_names, ssa_diag


def _rr_replicates(
    xml_path: str,
    t_end: float,
    n_steps: int,
    n_replicates: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (times[n_time], species[n_rep, n_time, n_species], names).

    Both engines report in **concentration** units: bngsim's reported
    species values are amount/volume, and RR's default output columns are
    concentrations (``[S]``). Comparing concentration-vs-concentration is
    dimensionally consistent for any compartment volume. (Selecting RR
    *amounts* instead would mismatch bngsim on every V != 1 compartment.)
    Strips RR's 'time' column and the surrounding ``[brackets]`` from the
    species names so the layout matches bngsim's output exactly.
    """
    import roadrunner

    rr = roadrunner.RoadRunner(xml_path)
    rr.integrator = "gillespie"
    rr.integrator.variable_step_size = False

    out: list[np.ndarray] = []
    species_names: list[str] = []
    times: np.ndarray | None = None
    for rep in range(n_replicates):
        rr.reset()
        rr.integrator.seed = int(seed_base + rep)
        res = rr.simulate(0, t_end, n_steps + 1)
        arr = np.asarray(res)
        if times is None:
            times = arr[:, 0].copy()
            cols = [c[1:-1] if c.startswith("[") else c for c in res.colnames]
            species_names = cols[1:]
        out.append(arr[:, 1:])
    return times, np.stack(out, axis=0), species_names


def _align_columns(
    bn_names: list[str], rr_names: list[str]
) -> tuple[list[int], list[int], list[str]] | None:
    """Return (bn_idx[k], rr_idx[k], common_names[k]) or None if disjoint.

    Both bngsim's from_sbml and roadrunner read the same SBML file, so
    the species ID set must match — but we don't assume column order.
    """
    bn_map = {n: i for i, n in enumerate(bn_names)}
    rr_map = {n: i for i, n in enumerate(rr_names)}
    common = sorted(set(bn_map) & set(rr_map))
    if not common:
        return None
    bn_idx = [bn_map[n] for n in common]
    rr_idx = [rr_map[n] for n in common]
    return bn_idx, rr_idx, common


def _run_one(
    model_meta: dict,
    n_replicates: int,
    seed_base: int,
    mean_z_tol: float,
    max_failures_recorded: int,
) -> dict:
    name = model_meta["name"]
    t_end = float(model_meta["t_end"])
    n_steps = int(model_meta["n_steps"])

    xml_path = (BENCHMARKS_DIR / "suites" / "sbml_roundtrip" / f"{name}.xml").resolve()

    entry: dict = {
        "name": name,
        "sbml_file": str(xml_path.relative_to(BENCHMARKS_DIR)) if xml_path.exists() else None,
        "t_end": t_end,
        "n_steps": n_steps,
        "species_declared": int(model_meta.get("species", -1)),
        "n_replicates": n_replicates,
        "elapsed_sec_bn": 0.0,
        "elapsed_sec_rr": 0.0,
        "elapsed_sec_compare": 0.0,
        "n_compared": 0,
        "max_mean_z": None,
        "n_z_failures": 0,
        "z_failures": [],
        "status": "skip",
        "error": None,
    }

    if not xml_path.exists():
        entry["error"] = f"missing sbml: {xml_path}"
        return entry

    try:
        t0 = time.perf_counter()
        bn_times, bn_arr, bn_names, _bn_ssa_diag = _bn_replicates(
            str(xml_path), t_end, n_steps, n_replicates, seed_base
        )
        entry["elapsed_sec_bn"] = round(time.perf_counter() - t0, 3)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"bngsim side: {type(exc).__name__}: {exc}"[:300]
        return entry

    try:
        t0 = time.perf_counter()
        rr_times, rr_arr, rr_names = _rr_replicates(
            str(xml_path), t_end, n_steps, n_replicates, seed_base
        )
        entry["elapsed_sec_rr"] = round(time.perf_counter() - t0, 3)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"roadrunner side: {type(exc).__name__}: {exc}"[:300]
        return entry

    if not np.allclose(bn_times, rr_times, rtol=0, atol=1e-9):
        entry["status"] = "error"
        entry["error"] = (
            f"time grid mismatch: bn[:3]={bn_times[:3].tolist()} vs rr[:3]={rr_times[:3].tolist()}"
        )
        return entry

    align = _align_columns(bn_names, rr_names)
    if align is None:
        entry["status"] = "error"
        entry["error"] = f"no common species names: bn={bn_names[:5]}... rr={rr_names[:5]}..."
        return entry
    bn_idx, rr_idx, common = align
    entry["n_species_compared"] = len(common)

    n_time = bn_arr.shape[1]
    failures: list[dict] = []
    max_z = 0.0

    t_cmp0 = time.perf_counter()
    n_compared = 0
    for ti in range(1, n_time):
        for k, sp in enumerate(common):
            bn_col = bn_arr[:, ti, bn_idx[k]]
            rr_col = rr_arr[:, ti, rr_idx[k]]
            n = bn_col.shape[0]
            bn_mu = float(bn_col.mean())
            rr_mu = float(rr_col.mean())
            bn_var = float(bn_col.var(ddof=1))
            rr_var = float(rr_col.var(ddof=1))
            se = float(np.sqrt(max(bn_var / n + rr_var / n, 1e-18)))
            # Both columns identically constant → SE=0; treat means as
            # equal (z=0). If they differ exactly, SE-floor below
            # surfaces it as a finite z so it doesn't silently pass.
            z = 0.0 if se <= 1e-9 and bn_mu == rr_mu else abs(bn_mu - rr_mu) / max(se, 1e-9)
            n_compared += 1
            if z > max_z:
                max_z = z
            if z > mean_z_tol:
                failures.append(
                    {
                        "t_idx": int(ti),
                        "t": float(bn_times[ti]),
                        "species": sp,
                        "mean_z": float(z),
                        "bn_mu": bn_mu,
                        "rr_mu": rr_mu,
                        "bn_se": float(np.sqrt(bn_var / n)),
                        "rr_se": float(np.sqrt(rr_var / n)),
                    }
                )
    entry["elapsed_sec_compare"] = round(time.perf_counter() - t_cmp0, 3)
    entry["n_compared"] = n_compared
    entry["max_mean_z"] = float(max_z)
    entry["n_z_failures"] = len(failures)
    failures.sort(key=lambda f: -f["mean_z"])
    entry["z_failures"] = failures[:max_failures_recorded]
    entry["status"] = "pass" if not failures else "fail"
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n", type=int, default=DEFAULT_N, help="Replicate count per side (default 200)."
    )
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE, help="Seed schedule base.")
    p.add_argument(
        "--mean-z-tol",
        type=float,
        default=DEFAULT_MEAN_Z_TOL,
        help="Per-(time,species) mean z-score threshold; pass = max(z) <= tol.",
    )
    p.add_argument(
        "--include-large",
        action="store_true",
        help=(
            "Also test the .gitignored / heavy LARGE_MODELS set "
            "(egfr_net, fceri_gamma, multisite_phos, prion_aggregation, "
            "erk_activation). Run benchmarks/suites/sbml_roundtrip/run.py first."
        ),
    )
    p.add_argument(
        "--small-only",
        action="store_true",
        help=(
            "Only run the 4 small models "
            "(gene_expression_hill, simple_system, flagellar_motor, "
            "oscillatory_system). Useful for first-shape validation."
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names (overrides --include-large/--small-only).",
    )
    args = p.parse_args()

    models = load_suite(SUITE_SSA)

    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m["name"] in wanted]
    elif args.small_only:
        models = [m for m in models if m["name"] in SMALL_MODELS]
    elif not args.include_large:
        models = [m for m in models if m["name"] not in LARGE_MODELS]

    print("=" * 70)
    print("  bngsim SSA vs roadrunner gillespie: per-time mean z-score parity")
    print("=" * 70)
    print(f"  models:      {len(models)}")
    print(f"  replicates:  {args.n} per side")
    print(f"  seed_base:   {args.seed_base}")
    print(f"  mean_z_tol:  {args.mean_z_tol}")
    print()

    info = get_machine_info()
    started_total = time.perf_counter()
    cases = []
    for i, m in enumerate(models, 1):
        print(
            f"  [{i}/{len(models)}] {m['name']} "
            f"(sp={m.get('species', '?')}, t_end={m['t_end']}, n_steps={m['n_steps']})"
        )
        entry = _run_one(
            m,
            n_replicates=args.n,
            seed_base=args.seed_base,
            mean_z_tol=args.mean_z_tol,
            max_failures_recorded=DEFAULT_MAX_FAILURES_RECORDED,
        )
        if entry["status"] == "error":
            print(f"      ERR: {entry['error']}")
        elif entry["status"] == "skip":
            print(f"      SKIP: {entry['error']}")
        else:
            print(
                f"      {entry['status'].upper()} "
                f"(max_z={entry['max_mean_z']:.2f}, "
                f"z_fail={entry['n_z_failures']}/{entry['n_compared']}, "
                f"bn={entry['elapsed_sec_bn']}s, rr={entry['elapsed_sec_rr']}s)"
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
            "mean_z_tol": args.mean_z_tol,
            "include_large": args.include_large,
            "small_only": args.small_only,
            "match_strategy": "by_species_name",
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
    save_results(payload, "rr_ssa_trajectory_parity")
    return 0 if (n_fail + n_error) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
