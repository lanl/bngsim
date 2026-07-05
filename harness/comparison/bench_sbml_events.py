#!/usr/bin/env python3
"""SBML event models 3-engine benchmark (Figure S2).

214 SBML models with discrete events from BioModels (benchmarks/sbml_events/).
Three-phase pipeline with checkpointing:
  Phase 1: Load — BNGsim, libRoadRunner, AMICI
  Phase 2: Cross-validate — pairwise comparison between engines
  Phase 3: Time — 2w+5t median, head-to-head on xval-passing models

Four engine configs:
  1. BNGsim ExprTk (default)
  2. BNGsim codegen (BNGSIM_CODEGEN_THRESHOLD=0)
  3. libRoadRunner
  4. AMICI

Produces: 2×2 scatter plot → bngsim/dev/paper/fig_sbml_events_4panel.png

Usage:
    python bench_sbml_events.py                    # full run
    python bench_sbml_events.py --quick 20         # first 20 models
    python bench_sbml_events.py --phase 1          # load only
    python bench_sbml_events.py --resume           # resume from checkpoint

Output:
    results/bench_sbml_events.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    RESULTS_DIR,
    AmiciApiError,
    add_bngsim_timeout_arg,
    amici_prepare_model,
    amici_simulate,
    amici_state_names,
    cross_validate_by_name,
    format_bngsim_timeout,
    get_machine_info,
    save_results,
)

# ── Paths ─────────────────────────────────────────────────────────────────

EVENTS_DIR = BENCHMARKS_DIR / "sbml_events"

# ── Constants ─────────────────────────────────────────────────────────────

XVAL_RTOL = 1e-3
N_STEPS = 200
T_END_DEFAULT = 100.0
ADAPTIVE_HORIZONS = [100.0, 10.0, 1.0, 0.1]
_WARMUP = DEFAULT_WARMUP
_RUNS = DEFAULT_RUNS


# ── Helpers ───────────────────────────────────────────────────────────────


def geometric_mean(values):
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in values) / len(values)))


def discover_sbml_files():
    if not EVENTS_DIR.exists():
        print(f"ERROR: SBML events directory not found: {EVENTS_DIR}")
        sys.exit(1)
    return sorted(EVENTS_DIR.glob("*.xml"))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_checkpoint(data, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Checkpoint: {path}")
    return path


def load_checkpoint(name):
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Engine runners ────────────────────────────────────────────────────────


def _run_bngsim_sbml(sbml_path, t_end, n_steps, codegen=False, bngsim_timeout=None):
    """Run BNGsim on SBML file. Returns (names, traj) or raises."""
    import bngsim

    if codegen:
        old = os.environ.get("BNGSIM_CODEGEN_THRESHOLD")
        os.environ["BNGSIM_CODEGEN_THRESHOLD"] = "0"

    try:
        model = bngsim.Model.from_sbml(str(sbml_path))
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)
        names = list(result.species_names)
        traj = np.asarray(result.species)
        return names, traj
    finally:
        if codegen:
            if old is None:
                os.environ.pop("BNGSIM_CODEGEN_THRESHOLD", None)
            else:
                os.environ["BNGSIM_CODEGEN_THRESHOLD"] = old


def _run_rr_sbml(sbml_path, t_end, n_steps):
    """Run libRoadRunner on SBML file."""
    import roadrunner

    rr = roadrunner.RoadRunner(str(sbml_path))
    rr.integrator.absolute_tolerance = 1e-12
    rr.integrator.relative_tolerance = 1e-8
    result = rr.simulate(0, t_end, n_steps + 1)
    data = np.array(result)
    names = []
    for c in result.colnames[1:]:
        names.append(c[1:-1] if c.startswith("[") else c)
    return names, data[:, 1:]


def _run_amici_sbml(sbml_path, t_end, n_steps):
    """Run AMICI on SBML file (AMICI >= 1.0 snake_case API). Returns (names, states).

    Compiles the model in a scratch dir, then drives it through the shared
    fail-loud runner (GH #227): an API drift raises AmiciApiError, a non-success
    CVODES status raises RuntimeError.
    """
    import tempfile

    import amici

    with tempfile.TemporaryDirectory(prefix="amici_") as tmpdir:
        model_name = Path(sbml_path).stem.replace("-", "_").replace(".", "_")
        importer = amici.SbmlImporter(str(sbml_path))
        importer.sbml2amici(model_name, tmpdir)

        # Load compiled model, then run via the shared snake_case driver.
        model_module = amici.import_model_module(model_name, tmpdir)
        model, solver = amici_prepare_model(model_module)
        tspan = np.linspace(0, t_end, n_steps + 1)
        rdata, _ = amici_simulate(model, solver, tspan)
        return amici_state_names(model), rdata.x


# ── Timed runner ──────────────────────────────────────────────────────────


def _time_engine(runner_fn, warmup, runs):
    """Time a runner function. Returns median time or -1 on error."""
    times = []
    for i in range(warmup + runs):
        try:
            t0 = time.perf_counter()
            runner_fn()
            elapsed = time.perf_counter() - t0
        except AmiciApiError:
            raise  # API drift must crash loudly, never degrade to -1 (GH #227)
        except Exception:
            return -1
        if i >= warmup:
            times.append(elapsed)
    return median(times)


# ── Phase 1: Load ────────────────────────────────────────────────────────


def phase1_load(sbml_files, resume_data=None):
    """Try loading each model in all engines."""
    print("\n" + "=" * 70)
    print("  Phase 1: Load Gate")
    print(f"  Models: {len(sbml_files)}")
    print("=" * 70)

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r

    results = []
    counts = {"bngsim": 0, "rr": 0, "amici": 0}

    for i, sbml_path in enumerate(sbml_files):
        mid = sbml_path.stem
        if mid in done:
            results.append(done[mid])
            for eng in ("bngsim", "rr", "amici"):
                if done[mid].get(f"{eng}_ok"):
                    counts[eng] += 1
            continue

        entry = {"model": mid, "sbml_path": str(sbml_path)}

        # BNGsim
        try:
            import bngsim

            m = bngsim.Model.from_sbml(str(sbml_path))
            entry["bngsim_ok"] = True
            entry["n_species"] = m.n_species
            counts["bngsim"] += 1
        except Exception as e:
            entry["bngsim_ok"] = False
            entry["bngsim_error"] = str(e)[:150]

        # RoadRunner
        try:
            import roadrunner

            roadrunner.RoadRunner(str(sbml_path))
            entry["rr_ok"] = True
            counts["rr"] += 1
        except Exception as e:
            entry["rr_ok"] = False
            entry["rr_error"] = str(e)[:150]

        # AMICI
        try:
            import tempfile

            import amici

            with tempfile.TemporaryDirectory(prefix="amici_") as tmpdir:
                mn = mid.replace("-", "_").replace(".", "_")
                imp = amici.SbmlImporter(str(sbml_path))
                imp.sbml2amici(mn, tmpdir)
            entry["amici_ok"] = True
            counts["amici"] += 1
        except Exception as e:
            entry["amici_ok"] = False
            entry["amici_error"] = str(e)[:150]

        results.append(entry)

        if (i + 1) % 50 == 0 or i < 3:
            print(
                f"  [{i + 1}/{len(sbml_files)}] {mid}: "
                f"BNG={'✓' if entry.get('bngsim_ok') else '✗'} "
                f"RR={'✓' if entry.get('rr_ok') else '✗'} "
                f"AMICI={'✓' if entry.get('amici_ok') else '✗'}"
            )

        if (i + 1) % 100 == 0:
            save_checkpoint(results, "bench_sbml_events_phase1")

    print(
        f"\n  Phase 1: BNGsim={counts['bngsim']}, "
        f"RR={counts['rr']}, AMICI={counts['amici']} / {len(sbml_files)}"
    )
    save_checkpoint(results, "bench_sbml_events_phase1")
    return results


# ── Phase 2: Cross-validate ──────────────────────────────────────────────


def phase2_xval(phase1_results, bngsim_timeout=None, resume_data=None):
    """Cross-validate pairwise between engines."""
    print("\n" + "=" * 70)
    print("  Phase 2: Cross-Validation (pairwise)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(bngsim_timeout)}")
    print("=" * 70)

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r

    results = []
    pairs = [
        ("bngsim", "rr"),
        ("bngsim_cg", "rr"),
        ("bngsim", "amici"),
        ("bngsim_cg", "amici"),
    ]
    pair_counts = {f"{a}_vs_{b}": 0 for a, b in pairs}

    for i, p1 in enumerate(phase1_results):
        mid = p1["model"]
        sbml_path = p1.get("sbml_path", "")

        if mid in done:
            results.append(done[mid])
            for a, b in pairs:
                key = f"{a}_vs_{b}_xval"
                if done[mid].get(key):
                    pair_counts[f"{a}_vs_{b}"] += 1
            continue

        entry = {"model": mid, "sbml_path": sbml_path, "n_species": p1.get("n_species", 0)}

        # Run engines for xval (try adaptive horizons)
        for t_end in ADAPTIVE_HORIZONS:
            trajs = {}

            if p1.get("bngsim_ok"):
                try:
                    n, t = _run_bngsim_sbml(
                        sbml_path,
                        t_end,
                        N_STEPS,
                        codegen=False,
                        bngsim_timeout=bngsim_timeout,
                    )
                    if not np.any(np.isnan(t)):
                        trajs["bngsim"] = (n, t)
                except Exception:
                    pass

            if p1.get("bngsim_ok"):
                try:
                    n, t = _run_bngsim_sbml(
                        sbml_path,
                        t_end,
                        N_STEPS,
                        codegen=True,
                        bngsim_timeout=bngsim_timeout,
                    )
                    if not np.any(np.isnan(t)):
                        trajs["bngsim_cg"] = (n, t)
                except Exception:
                    pass

            if p1.get("rr_ok"):
                try:
                    n, t = _run_rr_sbml(sbml_path, t_end, N_STEPS)
                    if not np.any(np.isnan(t)):
                        trajs["rr"] = (n, t)
                except Exception:
                    pass

            if p1.get("amici_ok"):
                try:
                    n, t = _run_amici_sbml(sbml_path, t_end, N_STEPS)
                    if t is not None and not np.any(np.isnan(t)):
                        trajs["amici"] = (n, t)
                except AmiciApiError:
                    raise  # API drift must crash loudly (GH #227)
                except Exception:
                    pass

            # Check each pair
            any_pass = False
            for a_key, b_key in pairs:
                pair_name = f"{a_key}_vs_{b_key}"
                if a_key in trajs and b_key in trajs:
                    na, ta = trajs[a_key]
                    nb, tb = trajs[b_key]
                    ok, merr, nm, detail = cross_validate_by_name(na, ta, nb, tb, rtol=XVAL_RTOL)
                    if ok:
                        entry[f"{pair_name}_xval"] = True
                        entry[f"{pair_name}_t_end"] = t_end
                        entry[f"{pair_name}_err"] = merr
                        pair_counts[pair_name] += 1
                        any_pass = True

            if any_pass:
                break  # found a good horizon

        results.append(entry)

        if (i + 1) % 50 == 0 or i < 3:
            xv = sum(1 for k, v in entry.items() if k.endswith("_xval") and v)
            print(f"  [{i + 1}/{len(phase1_results)}] {mid}: {xv}/4 pairs passed")

        if (i + 1) % 100 == 0:
            save_checkpoint(results, "bench_sbml_events_phase2")

    print("\n  Phase 2 pair counts:")
    for pn, cnt in pair_counts.items():
        print(f"    {pn}: {cnt}")
    save_checkpoint(results, "bench_sbml_events_phase2")
    return results


# ── Phase 3: Timing ──────────────────────────────────────────────────────


def phase3_timing(phase2_results, warmup, runs, bngsim_timeout=None, resume_data=None):
    """Time head-to-head on xval-passing models."""
    print("\n" + "=" * 70)
    print(f"  Phase 3: Timing ({warmup}w + {runs}t, median)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(bngsim_timeout)}")
    print("=" * 70)

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r

    results = []
    pairs = [
        ("bngsim", "rr"),
        ("bngsim_cg", "rr"),
        ("bngsim", "amici"),
        ("bngsim_cg", "amici"),
    ]
    speedup_lists = {f"{a}_vs_{b}": [] for a, b in pairs}

    for i, p2 in enumerate(phase2_results):
        mid = p2["model"]
        sbml_path = p2.get("sbml_path", "")

        if mid in done:
            results.append(done[mid])
            for a, b in pairs:
                pn = f"{a}_vs_{b}"
                sp = done[mid].get(f"{pn}_speedup")
                if sp and sp > 0:
                    speedup_lists[pn].append(sp)
            continue

        entry = {"model": mid, "sbml_path": sbml_path, "n_species": p2.get("n_species", 0)}

        for a_key, b_key in pairs:
            pn = f"{a_key}_vs_{b_key}"
            if not p2.get(f"{pn}_xval"):
                continue

            t_end = p2.get(f"{pn}_t_end", T_END_DEFAULT)
            codegen = a_key.endswith("_cg")

            # Time engine A (BNGsim variant)
            ta = _time_engine(
                lambda sp=sbml_path, te=t_end, cg=codegen: _run_bngsim_sbml(
                    sp, te, N_STEPS, codegen=cg, bngsim_timeout=bngsim_timeout
                ),
                warmup=warmup,
                runs=runs,
            )
            entry[f"{a_key}_time"] = ta

            # Time engine B
            if b_key == "rr":
                tb = _time_engine(
                    lambda sp=sbml_path, te=t_end: _run_rr_sbml(sp, te, N_STEPS),
                    warmup=warmup,
                    runs=runs,
                )
            else:
                tb = _time_engine(
                    lambda sp=sbml_path, te=t_end: _run_amici_sbml(sp, te, N_STEPS),
                    warmup=warmup,
                    runs=runs,
                )
            entry[f"{b_key}_time_{pn}"] = tb

            if ta > 0 and tb > 0:
                speedup = tb / ta
                entry[f"{pn}_speedup"] = speedup
                speedup_lists[pn].append(speedup)

        results.append(entry)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(phase2_results)}] {mid}")
            save_checkpoint(results, "bench_sbml_events_phase3")

    print("\n  Phase 3 results:")
    for pn, slist in speedup_lists.items():
        if slist:
            gm = geometric_mean(slist)
            nw = sum(1 for s in slist if s > 1)
            print(f"    {pn}: {len(slist)} models, geomean={gm:.2f}x, BNGsim faster={nw}")

    save_checkpoint(results, "bench_sbml_events_phase3")
    return results, speedup_lists


# ── Plot ──────────────────────────────────────────────────────────────────


def make_scatter_plot(phase3_results, speedup_lists):
    """Generate 2×2 scatter plot → bngsim/dev/paper/fig_sbml_events_4panel.png."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    pairs = [
        ("bngsim_vs_rr", "BNGsim ExprTk", "libRoadRunner"),
        ("bngsim_vs_amici", "BNGsim ExprTk", "AMICI"),
        ("bngsim_cg_vs_rr", "BNGsim codegen", "libRoadRunner"),
        ("bngsim_cg_vs_amici", "BNGsim codegen", "AMICI"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()

    for ax, (pn, a_label, b_label) in zip(axes, pairs, strict=False):
        xs, ys, colors = [], [], []
        a_key = pn.split("_vs_")[0]
        b_key = pn.split("_vs_")[1]

        for r in phase3_results:
            ta = r.get(f"{a_key}_time")
            tb = r.get(f"{b_key}_time_{pn}")
            if ta and ta > 0 and tb and tb > 0:
                xs.append(tb * 1000)  # ms
                ys.append(ta * 1000)
                nsp = r.get("n_species", 0)
                if nsp <= 10:
                    colors.append("tab:blue")
                elif nsp <= 50:
                    colors.append("tab:orange")
                else:
                    colors.append("tab:red")

        if not xs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{a_label} vs {b_label}")
            continue

        ax.scatter(xs, ys, c=colors, alpha=0.5, s=20, edgecolors="none")
        lo = min(min(xs), min(ys)) * 0.5
        hi = max(max(xs), max(ys)) * 2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{b_label} (ms)")
        ax.set_ylabel(f"{a_label} (ms)")
        ax.set_aspect("equal")

        slist = speedup_lists.get(pn, [])
        if slist:
            gm = geometric_mean(slist)
            nw = sum(1 for s in slist if s > 1)
            ax.set_title(f"{a_label} vs {b_label}\ngeomean {gm:.2f}×, BNG wins {nw}/{len(slist)}")
        else:
            ax.set_title(f"{a_label} vs {b_label}")

    fig.tight_layout()
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    out_path = repo_root / "paper" / "fig_sbml_events_4panel.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="SBML event models 3-engine benchmark")
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--warmup", type=int, default=_WARMUP)
    ap.add_argument("--runs", type=int, default=_RUNS)
    add_bngsim_timeout_arg(ap)
    args = ap.parse_args()

    info = get_machine_info()
    sbml_files = discover_sbml_files()
    if args.quick > 0:
        sbml_files = sbml_files[: args.quick]
    print(f"\nTotal SBML event models: {len(sbml_files)}")

    # Phase 1
    p1r = load_checkpoint("bench_sbml_events_phase1") if args.resume else None
    if args.phase in (0, 1):
        p1 = phase1_load(sbml_files, resume_data=p1r)
    else:
        p1 = load_checkpoint("bench_sbml_events_phase1")
        if p1 is None:
            print("ERROR: Phase 1 checkpoint needed")
            sys.exit(1)
    if args.phase == 1:
        return

    # Phase 2
    p2r = load_checkpoint("bench_sbml_events_phase2") if args.resume else None
    if args.phase in (0, 2):
        p2 = phase2_xval(p1, bngsim_timeout=args.bngsim_timeout, resume_data=p2r)
    else:
        p2 = load_checkpoint("bench_sbml_events_phase2")
        if p2 is None:
            print("ERROR: Phase 2 checkpoint needed")
            sys.exit(1)
    if args.phase == 2:
        return

    # Phase 3
    p3r = load_checkpoint("bench_sbml_events_phase3") if args.resume else None
    if args.phase in (0, 3):
        p3, speedup_lists = phase3_timing(
            p2,
            args.warmup,
            args.runs,
            bngsim_timeout=args.bngsim_timeout,
            resume_data=p3r,
        )
    else:
        p3 = load_checkpoint("bench_sbml_events_phase3")
        speedup_lists = {}
        if p3 is None:
            print("ERROR: Phase 3 checkpoint needed")
            sys.exit(1)

    # Generate plot
    make_scatter_plot(p3, speedup_lists)

    # Save final results
    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "xval_rtol": XVAL_RTOL,
            "n_steps": N_STEPS,
            "adaptive_horizons": ADAPTIVE_HORIZONS,
            "bngsim_timeout": args.bngsim_timeout,
        },
        "summary": {
            "n_total": len(sbml_files),
            "n_bngsim_loaded": sum(1 for r in p1 if r.get("bngsim_ok")),
            "n_rr_loaded": sum(1 for r in p1 if r.get("rr_ok")),
            "n_amici_loaded": sum(1 for r in p1 if r.get("amici_ok")),
            "speedup_geomeans": {pn: geometric_mean(sl) for pn, sl in speedup_lists.items() if sl},
        },
        "phase1_load": p1,
        "phase2_xval": p2,
        "phase3_timing": p3,
    }
    save_results(output, "bench_sbml_events")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
