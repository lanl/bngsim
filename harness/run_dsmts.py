#!/usr/bin/env python3
"""DSMTS sweep: BNGsim SSA vs DSMTS published mean/sd CSVs.

Iterates the 39 ``is_dsmts_proper`` cases in
``harness/sbml_test_suite/dsmts/dsmts_index.json``, runs ``Model.from_sbml``
+ ``Simulator(method="ssa")`` for ``--n`` replicates per case (default
1000), and applies the DSMTS Z/Y tolerance from
``stochastic/README.md``::

    Z_t = sqrt(N) * (X_obs - mu_exp) / sigma_exp     in mean_range
    Y_t = sqrt(N/2) * (S_obs^2 / sigma_exp^2 - 1)    in sd_range

The README's stated minimum is N=1000 with N=10000 recommended for
published baselines and N=100000 for "subtle implementation errors".
Per-case status uses the README's tolerant rule: ≤1 Z exceedance and
≤1 Y exceedance per variable count as ``pass``. The strict (zero
exceedance) verdict is recorded separately as ``strict_pass``.

The 39 cases (one SBML level + the mean/sd CSVs each, ~140 KB) are
vendored under ``harness/sbml_test_suite/dsmts/cases/`` and used by
default, so the gate is hermetic — no external sbml-test-suite checkout
needed. ``--suite-root`` / ``$SBML_TEST_SUITE_DIR`` still override (e.g.
to re-verify against newer upstream data); regenerate the vendored copy
with ``harness/sbml_test_suite/dsmts/vendor_dsmts_cases.py``.

Usage:
    cd bngsim && uv run python harness/run_dsmts.py [--n 1000] [--quick K]
                                                    [--cases 00001,00009]

Output:
    results/dsmts.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HARNESS_DIR,
    get_machine_info,
    save_results,
)

DSMTS_DIR = HARNESS_DIR / "sbml_test_suite" / "dsmts"
DSMTS_INDEX = DSMTS_DIR / "dsmts_index.json"
# In-repo copy of the 39 DSMTS-proper cases (one SBML level + mean/sd CSVs +
# settings each, ~140 KB), produced by ``vendor_dsmts_cases.py``. Used as the
# default suite root so the gate is hermetic — runnable on a bare clone and in
# CI without an external sbml-test-suite checkout. Overridable via --suite-root
# or $SBML_TEST_SUITE_DIR (e.g. to re-verify against newer upstream data).
VENDORED_SUITE_ROOT = DSMTS_DIR / "cases"
LEVEL_PREFERENCE = ["l3v2", "l3v1", "l2v5", "l2v4", "l2v3", "l2v2", "l2v1"]
DEFAULT_N = 1000
DEFAULT_SEED_BASE = 1000


def _read_dsmts_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse DSMTS mean/sd CSV. Header: ``"time",<var>,<var>...``"""
    with path.open() as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r]
    header = [c.strip().strip('"') for c in rows[0]]
    data = np.array([[float(c) for c in r] for r in rows[1:]])
    return header, data


def _resolve_sbml_path(suite_root: Path, case_id: str, levels: list[str]) -> Path:
    for lvl in LEVEL_PREFERENCE:
        if lvl not in levels:
            continue
        p = suite_root / case_id / f"{case_id}-sbml-{lvl}.xml"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No SBML file found for case {case_id} at preferences {LEVEL_PREFERENCE}"
    )


def _which_level(path: Path) -> str:
    # filename pattern: NNNNN-sbml-lXvY.xml
    return path.stem.split("-sbml-")[-1]


def _run_case(
    case_id: str,
    case: dict,
    suite_root: Path,
    n_replicates: int,
    seed_base: int,
    verbose: bool,
) -> dict:
    """Simulate one DSMTS case, compute Z/Y, return result entry."""
    import bngsim  # local import: keeps --help cheap

    settings = case["settings"]
    duration = float(settings["duration"])
    n_steps = int(settings["steps"])
    variables = list(settings["variables"])
    mean_range = tuple(float(v) for v in settings["mean_range"])
    sd_range = tuple(float(v) for v in settings["sd_range"])

    sbml_path = _resolve_sbml_path(suite_root, case_id, case["sbml_levels_present"])
    sbml_level = _which_level(sbml_path)

    started = time.perf_counter()
    y_test_noise_dominates = bool(case.get("y_test_noise_dominates", False))
    z_test_oracle_offset = bool(case.get("z_test_oracle_offset", False))
    entry: dict = {
        "case_id": case_id,
        "synopsis": case.get("synopsis", ""),
        "test_tags": case.get("test_tags", []),
        "component_tags": case.get("component_tags", []),
        "sbml_level_used": sbml_level,
        "sbml_path": str(sbml_path.relative_to(suite_root)),
        "y_test_noise_dominates": y_test_noise_dominates,
        "z_test_oracle_offset": z_test_oracle_offset,
        "settings": {
            "start": float(settings["start"]),
            "duration": duration,
            "steps": n_steps,
            "variables": variables,
            "mean_range": list(mean_range),
            "sd_range": list(sd_range),
        },
    }

    try:
        model = bngsim.Model.from_sbml(str(sbml_path))
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"from_sbml: {type(exc).__name__}: {exc}"[:300]
        entry["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return entry

    # Establish species name → column index mapping using one bare clone.
    try:
        m0 = model.clone()
        m0.reset()
        sim0 = bngsim.Simulator(m0, method="ssa")
        # Touch run to populate species_names (a single zero-time call).
        # We actually need species_names before running, but the Result
        # is what carries them — run with n_points=2 and seed 0.
        r0 = sim0.run(t_span=(0, duration), n_points=n_steps + 1, seed=seed_base)
        species_names = list(r0.species_names)
        observable_names = list(r0.observable_names)
        # DSMTS expected values are in amount (substance) units regardless of
        # hOSU; BNGsim's SBML loader stores everything as concentration
        # (`amount/V_c`). Multiply each species column by its volume_factor
        # (= V_c, set by the loader) to recover amount. Defaults to 1.0 for
        # V=1 cases (no-op).
        sp_data = model._core.codegen_data()["species"]
        species_amount_scale = np.array(
            [s.get("volume_factor", 1.0) for s in sp_data], dtype=float
        )
        species_fixed = [bool(s.get("fixed", False)) for s in sp_data]
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"warmup: {type(exc).__name__}: {exc}"[:300]
        entry["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return entry

    # Resolve each DSMTS variable to either a species column or an observable
    # column. Prefer species when it exists and is not fixed — the loader
    # always emits a same-named observable for each species as a mirror, but
    # reading the species directly avoids needing per-observable amount
    # scaling. Fall back to the observable when the species is fixed (which
    # is the linear-on-species AssignmentRule case, e.g. DSMTS 00019's
    # ``y = 2*X`` where the observable carries the time-varying weighted sum
    # while the species shadow stays pinned at IC).
    # ``var_sources[i]`` is ``("species", col)`` or ``("obs", col)``.
    var_sources: list[tuple[str, int]] = []
    for v in variables:
        sp_col = species_names.index(v) if v in species_names else -1
        if sp_col >= 0 and not species_fixed[sp_col]:
            var_sources.append(("species", sp_col))
        elif v in observable_names:
            var_sources.append(("obs", observable_names.index(v)))
        elif sp_col >= 0:
            # Fixed species with no observable fallback — read the species
            # column anyway (will be constant).
            var_sources.append(("species", sp_col))
        else:
            entry["status"] = "error"
            entry["error"] = (
                f"variable {v!r} not in species {species_names} or observables {observable_names}"
            )[:300]
            entry["elapsed_sec"] = round(time.perf_counter() - started, 3)
            return entry

    # Replicate sweep. We re-use the warmup r0 as replicate 0.
    n_time = n_steps + 1
    n_var = len(variables)
    # accumulators for streaming mean/M2 (Welford) per (time, var)
    sum_x = np.zeros((n_time, n_var))
    sum_x2 = np.zeros((n_time, n_var))
    n_done = 0

    def _gather_columns(result) -> np.ndarray:
        species_arr = np.asarray(result.species)
        obs_arr = np.asarray(result.observables) if "obs" in (s for s, _ in var_sources) else None
        cols = np.empty((species_arr.shape[0], len(var_sources)), dtype=float)
        for vi, (kind, col) in enumerate(var_sources):
            if kind == "species":
                # Convert storage (=amount/V_c) → amount by multiplying by V_c.
                cols[:, vi] = species_arr[:, col] * species_amount_scale[col]
            else:
                cols[:, vi] = obs_arr[:, col]
        return cols

    def _accum(result):
        nonlocal n_done
        cols = _gather_columns(result)
        sum_x[:] += cols
        sum_x2[:] += cols * cols
        n_done += 1

    _accum(r0)
    last_log = time.perf_counter()
    for rep in range(1, n_replicates):
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ssa")
        try:
            r = sim.run(t_span=(0, duration), n_points=n_steps + 1, seed=seed_base + rep)
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"rep {rep}: {type(exc).__name__}: {exc}"[:300]
            entry["elapsed_sec"] = round(time.perf_counter() - started, 3)
            entry["n_replicates_completed"] = n_done
            return entry
        _accum(r)
        if verbose and (time.perf_counter() - last_log) > 5.0:
            print(f"      [{case_id}] rep {n_done}/{n_replicates}", flush=True)
            last_log = time.perf_counter()

    obs_mean = sum_x / n_done
    # Sample variance (unbiased)
    obs_var = (sum_x2 - n_done * obs_mean * obs_mean) / max(n_done - 1, 1)
    obs_var = np.maximum(obs_var, 0.0)  # guard against tiny FP negatives

    # Read expected mean/sd CSVs, align time/columns.
    mean_hdr, mean_data = _read_dsmts_csv(suite_root / case["dsmts_mean_csv"])
    sd_hdr, sd_data = _read_dsmts_csv(suite_root / case["dsmts_sd_csv"])
    # Column 0 is time. Validate time alignment loosely.
    t_obs = np.linspace(settings["start"], duration, n_time)
    t_exp = mean_data[:, 0]
    if t_exp.shape[0] != n_time or not np.allclose(t_exp, t_obs, rtol=0, atol=1e-6):
        entry["status"] = "error"
        entry["error"] = f"time grid mismatch: expected {t_exp.shape[0]} pts vs {n_time}"
        entry["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return entry

    var_pass = True
    var_strict_pass = True
    per_variable: dict = {}
    for var in variables:
        if var not in mean_hdr or var not in sd_hdr:
            entry["status"] = "error"
            entry["error"] = (
                f"variable {var!r} missing from DSMTS CSV headers (mean={mean_hdr}, sd={sd_hdr})"
            )
            entry["elapsed_sec"] = round(time.perf_counter() - started, 3)
            return entry
        ci_obs = variables.index(var)
        ci_mean = mean_hdr.index(var)
        ci_sd = sd_hdr.index(var)
        mu_exp = mean_data[:, ci_mean]
        sd_exp = sd_data[:, ci_sd]
        x_obs = obs_mean[:, ci_obs]
        s2_obs = obs_var[:, ci_obs]

        # Z and Y; per README, sigma_exp = 0 → division undefined.
        # Treat sd_exp == 0 as "no test at this point" (DSMTS does not
        # define Z when sigma=0). Skip those points and record n_skipped.
        mask_z = sd_exp > 0
        z = np.full(n_time, np.nan)
        y = np.full(n_time, np.nan)
        z[mask_z] = np.sqrt(n_done) * (x_obs[mask_z] - mu_exp[mask_z]) / sd_exp[mask_z]
        var_exp = sd_exp * sd_exp
        mask_y = var_exp > 0
        y[mask_y] = np.sqrt(n_done / 2.0) * (s2_obs[mask_y] / var_exp[mask_y] - 1.0)

        z_failures = []
        y_failures = []
        for i in range(n_time):
            if mask_z[i] and (z[i] < mean_range[0] or z[i] > mean_range[1]):
                z_failures.append(
                    {
                        "t_idx": int(i),
                        "t": float(t_obs[i]),
                        "z": float(z[i]),
                        "obs_mean": float(x_obs[i]),
                        "exp_mean": float(mu_exp[i]),
                        "exp_sd": float(sd_exp[i]),
                    }
                )
            if mask_y[i] and (y[i] < sd_range[0] or y[i] > sd_range[1]):
                y_failures.append(
                    {
                        "t_idx": int(i),
                        "t": float(t_obs[i]),
                        "y": float(y[i]),
                        "obs_var": float(s2_obs[i]),
                        "exp_var": float(var_exp[i]),
                    }
                )

        per_variable[var] = {
            "z_min": float(np.nanmin(z)) if mask_z.any() else None,
            "z_max": float(np.nanmax(z)) if mask_z.any() else None,
            "y_min": float(np.nanmin(y)) if mask_y.any() else None,
            "y_max": float(np.nanmax(y)) if mask_y.any() else None,
            "n_z_skipped_sigma0": int(np.sum(~mask_z)),
            "n_y_skipped_sigma0": int(np.sum(~mask_y)),
            "n_z_failures": len(z_failures),
            "n_y_failures": len(y_failures),
            "z_failures": z_failures,
            "y_failures": y_failures,
        }
        # Y-test noise gate: when the case has been annotated as
        # ``y_test_noise_dominates`` (heavy-tail extinction regime where the
        # DSMTS ±5 Y-stat threshold falls below the natural σ_Y noise floor —
        # see dev/investigations/sbml_ssa_phase5d_2026-05-07.md), exclude Y
        # failures from the strict-pass calculation. Z failures still count.
        # Y-failure detail is preserved so the data is auditable.
        effective_y_failures = [] if y_test_noise_dominates else y_failures
        # Z-test oracle gate: when the case has been annotated as
        # ``z_test_oracle_offset`` (DSMTS analytical expected mean has a
        # known O(1/N) approximation error vs. exact discrete SSA — see
        # dev/investigations/sbml_ssa_phase5e_2026-05-07.md), exclude Z
        # failures from the strict-pass calculation. Y failures still count.
        # Z-failure detail is preserved so the data is auditable.
        effective_z_failures = [] if z_test_oracle_offset else z_failures
        if len(effective_z_failures) > 1 or len(effective_y_failures) > 1:
            var_pass = False
        if effective_z_failures or effective_y_failures:
            var_strict_pass = False

    entry["n_replicates_completed"] = n_done
    entry["per_variable"] = per_variable
    entry["status"] = "pass" if var_pass else "fail"
    entry["strict_pass"] = var_strict_pass
    entry["error"] = None
    entry["elapsed_sec"] = round(time.perf_counter() - started, 3)
    return entry


def _aggregate_by_tag(cases: list[dict]) -> dict:
    out: dict = {}
    for c in cases:
        for t in c.get("test_tags", []) + c.get("component_tags", []):
            slot = out.setdefault(
                t, {"total": 0, "pass": 0, "fail": 0, "error": 0, "strict_pass": 0}
            )
            slot["total"] += 1
            slot[c["status"]] = slot.get(c["status"], 0) + 1
            if c.get("strict_pass"):
                slot["strict_pass"] += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=(
            "Replicate count per case (default 1000). DSMTS README floor is "
            "N=1000; N=10000 is the published baseline; bump if Z exceedances "
            "look noise-dominated."
        ),
    )
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE, help="Seed schedule base.")
    p.add_argument(
        "--quick",
        type=int,
        default=0,
        help="Limit to first K is_dsmts_proper cases (debug aid).",
    )
    p.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated case ids (e.g. 00001,00009). Overrides --quick.",
    )
    p.add_argument(
        "--suite-root",
        type=Path,
        default=None,
        help="Override path to .../cases/stochastic (e.g. a full upstream "
        "checkout). Defaults to $SBML_TEST_SUITE_DIR/cases/stochastic if set, "
        "else the in-repo vendored copy (hermetic).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-rep progress.")
    args = p.parse_args()

    with DSMTS_INDEX.open() as f:
        index = json.load(f)

    if args.suite_root is not None:
        suite_root = args.suite_root.expanduser().resolve()
    else:
        env = os.environ.get("SBML_TEST_SUITE_DIR")
        if env:
            suite_root = (Path(env).expanduser() / "cases" / "stochastic").resolve()
        else:
            # Default: the in-repo vendored copy (hermetic). The legacy
            # _meta.suite_root (a machine-specific upstream path) is provenance
            # only and no longer load-bearing.
            suite_root = VENDORED_SUITE_ROOT.resolve()
    if not suite_root.is_dir():
        sys.exit(
            f"DSMTS suite root not found: {suite_root}\n"
            "The vendored copy should exist at "
            f"{VENDORED_SUITE_ROOT}; regenerate it with "
            "harness/sbml_test_suite/dsmts/vendor_dsmts_cases.py, or pass "
            "--suite-root / set $SBML_TEST_SUITE_DIR to a full upstream checkout."
        )

    proper_ids = [cid for cid, c in index["cases"].items() if c.get("is_dsmts_proper")]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        proper_ids = [cid for cid in proper_ids if cid in wanted]
    elif args.quick > 0:
        proper_ids = proper_ids[: args.quick]

    print("=" * 70)
    print("  DSMTS sweep: BNGsim SSA vs published mean/sd")
    print("=" * 70)
    print(f"  cases:       {len(proper_ids)}")
    print(f"  replicates:  {args.n}")
    print(f"  seed_base:   {args.seed_base}")
    print(f"  suite_root:  {suite_root}")
    print()

    info = get_machine_info()
    started_total = time.perf_counter()
    cases: list[dict] = []
    for i, cid in enumerate(proper_ids, 1):
        case = index["cases"][cid]
        synopsis = case.get("synopsis", "")
        print(f"  [{i}/{len(proper_ids)}] {cid}: {synopsis[:54]}")
        entry = _run_case(
            cid,
            case,
            suite_root,
            n_replicates=args.n,
            seed_base=args.seed_base,
            verbose=not args.quiet,
        )
        flag = (
            "PASS"
            if entry["status"] == "pass"
            else ("FAIL" if entry["status"] == "fail" else "ERR")
        )
        if entry["status"] == "error":
            print(f"      {flag}: {entry.get('error', '')}")
        else:
            zsum = sum(v.get("n_z_failures", 0) for v in entry["per_variable"].values())
            ysum = sum(v.get("n_y_failures", 0) for v in entry["per_variable"].values())
            print(
                f"      {flag} (z_fail={zsum}, y_fail={ysum}, "
                f"strict={'Y' if entry.get('strict_pass') else 'N'}, "
                f"{entry['elapsed_sec']}s)"
            )
        cases.append(entry)

    elapsed_total = time.perf_counter() - started_total
    n_pass = sum(1 for c in cases if c["status"] == "pass")
    n_fail = sum(1 for c in cases if c["status"] == "fail")
    n_error = sum(1 for c in cases if c["status"] == "error")
    n_strict = sum(1 for c in cases if c.get("strict_pass"))

    print()
    print("=" * 70)
    print(
        f"  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_error}  "
        f"strict_pass: {n_strict}/{len(cases)}  elapsed: {elapsed_total:.1f}s"
    )
    print("=" * 70)

    payload = {
        "machine_info": info,
        "config": {
            "n_replicates": args.n,
            "seed_base": args.seed_base,
            "preferred_levels": LEVEL_PREFERENCE,
            "z_tolerance_source": (
                "stochastic/README.md (per-case mean_range/sd_range; pass = "
                "≤1 Z exceedance and ≤1 Y exceedance per variable, "
                "strict_pass = 0 exceedances)"
            ),
            "suite_root": str(suite_root),
        },
        "summary": {
            "total": len(cases),
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "strict_pass": n_strict,
            "elapsed_sec": round(elapsed_total, 2),
            "by_test_tag": _aggregate_by_tag(cases),
        },
        "cases": {c["case_id"]: c for c in cases},
    }
    save_results(payload, "dsmts")
    return 0 if (n_fail + n_error) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
