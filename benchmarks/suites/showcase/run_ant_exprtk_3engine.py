#!/usr/bin/env python3
"""Benchmark hand-crafted Antimony models across BNGsim ExprTK, RR, and AMICI.

This showcase benchmark:
1. Uses @SIM tags in each .ant file for (t_start, t_end, n_steps)
2. Cross-validates trajectories by species name
3. Times only consistency-passing model/engine pairs
4. Generates a single-panel scatter figure for main-text use
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root (the bngsim/ tree)
SHOWCASE_ROOT = Path(__file__).resolve().parent
ANT_MODELS_DIR = REPO_ROOT / "benchmarks" / "models" / "antimony" / "ssys"
RESULTS_ROOT = SHOWCASE_ROOT / "results"
PAPER_LATEX = REPO_ROOT / "dev" / "paper" / "latex"
FIGURE_DEFAULT = REPO_ROOT / "dev" / "paper" / "fig_ant_exprtk_rr_amici.png"
FIGURE_SNIPPET = PAPER_LATEX / "generated" / "ant_exprtk.tex"
AMICI_CACHE = REPO_ROOT / "harness" / "models" / "amici_cache"


def _safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
    except Exception:
        return "not installed"
    return str(getattr(mod, "__version__", "unknown"))


def _machine_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "bngsim_version": _safe_version("bngsim"),
        "roadrunner_version": _safe_version("roadrunner"),
        "amici_version": _safe_version("amici"),
        "antimony_version": _safe_version("antimony"),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = "unknown"
    return info


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def parse_sim_tag(ant_path: Path) -> tuple[float, float, int]:
    """Parse @SIM metadata from Antimony comments."""
    t_start = 0.0
    t_end = 20.0
    n_steps = 200
    try:
        for line in ant_path.read_text().splitlines():
            if "@SIM" not in line:
                continue
            fields = {
                m.group(1).upper(): m.group(2)
                for m in re.finditer(r"(\w+)\s*=\s*([0-9.eE+\-]+)", line)
            }
            if "T_START" in fields:
                t_start = float(fields["T_START"])
            if "T_END" in fields:
                t_end = float(fields["T_END"])
            if "N_STEPS" in fields:
                n_steps = int(float(fields["N_STEPS"]))
            break
    except Exception:
        pass
    return t_start, t_end, n_steps


def ant_to_sbml(ant_path: Path) -> str:
    import antimony

    antimony.clearPreviousLoads()
    rc = antimony.loadAntimonyFile(str(ant_path))
    if rc < 0:
        raise RuntimeError(antimony.getLastError())
    module_names = antimony.getModuleNames()
    if not module_names:
        raise RuntimeError("antimony did not return module names")
    sbml = antimony.getSBMLString(module_names[-1])
    if not sbml:
        raise RuntimeError("empty SBML returned from antimony")
    return sbml


def canonical_species_name(name: str, strip_compartment: bool) -> str:
    n = str(name).strip()
    if n.startswith("[") and n.endswith("]"):
        n = n[1:-1]
    if n.startswith("_ant_"):
        n = n[5:]
    # AMICI-generated state IDs are commonly prefixed (e.g., amici_X).
    if n.startswith("amici_"):
        n = n[6:]
    if strip_compartment and "." in n:
        n = n.split(".")[-1]
    return n


def cross_validate_by_name(
    names_a: list[str],
    traj_a: np.ndarray,
    names_b: list[str],
    traj_b: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float, int, str]:
    """Compare trajectories by species name, trying two name-normalization passes."""

    def _run_once(strip_compartment: bool) -> tuple[bool, float, int, str]:
        norm_a = [canonical_species_name(n, strip_compartment) for n in names_a]
        norm_b = [canonical_species_name(n, strip_compartment) for n in names_b]
        map_a = {n: i for i, n in enumerate(norm_a)}
        map_b = {n: i for i, n in enumerate(norm_b)}
        common = sorted(set(map_a) & set(map_b))
        if not common:
            return False, float("inf"), 0, "no common species names"

        n_time = min(traj_a.shape[0], traj_b.shape[0])
        max_rel_err = 0.0
        worst = ""

        for name in common:
            a_col = traj_a[:n_time, map_a[name]]
            b_col = traj_b[:n_time, map_b[name]]
            abs_diff = np.abs(a_col - b_col)
            # Mixed relative/absolute normalization:
            #   |a-b| / max(|a|, |b|, atol)
            # avoids near-zero blowups while preserving scale awareness.
            scale = np.maximum(np.maximum(np.abs(a_col), np.abs(b_col)), atol)
            rel = float(np.max(abs_diff / scale))
            if rel > max_rel_err:
                max_rel_err = rel
                worst = name

        passed = max_rel_err <= rtol
        detail = f"max_rel_err={max_rel_err:.2e} ({worst}), matched={len(common)}"
        return passed, max_rel_err, len(common), detail

    first = _run_once(strip_compartment=False)
    if first[2] > 0:
        return first
    return _run_once(strip_compartment=True)


def setup_bngsim_exprtk(ant_path: Path, atol: float, rtol: float):
    import bngsim

    old = os.environ.get("BNGSIM_NO_CODEGEN")
    os.environ["BNGSIM_NO_CODEGEN"] = "1"
    try:
        model = bngsim.Model.from_antimony(str(ant_path))
    finally:
        if old is None:
            os.environ.pop("BNGSIM_NO_CODEGEN", None)
        else:
            os.environ["BNGSIM_NO_CODEGEN"] = old
    sim = bngsim.Simulator(model, method="ode")
    sim.set_tolerances(rtol=rtol, atol=atol)
    return model, sim


def run_bngsim_once(model, sim, t_start: float, t_end: float, n_steps: int):
    model.reset()
    out = sim.run(t_span=(t_start, t_end), n_points=n_steps + 1)
    names = [str(n) for n in out.species_names]
    species = np.asarray(out.species, dtype=float)
    return names, species


def time_bngsim(
    model, sim, t_start: float, t_end: float, n_steps: int, warmup: int, runs: int
) -> float:
    times: list[float] = []
    for i in range(warmup + runs):
        model.reset()
        t0 = time.perf_counter()
        sim.run(t_span=(t_start, t_end), n_points=n_steps + 1)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return float(median(times))


def setup_roadrunner(sbml_str: str, atol: float, rtol: float):
    import roadrunner

    rr = roadrunner.RoadRunner(sbml_str)
    rr.integrator.absolute_tolerance = atol
    rr.integrator.relative_tolerance = rtol
    return rr


def run_roadrunner_once(rr, t_start: float, t_end: float, n_steps: int):
    rr.reset()
    out = rr.simulate(t_start, t_end, n_steps + 1)
    arr = np.asarray(out, dtype=float)
    names = [str(c).replace("[", "").replace("]", "") for c in out.colnames[1:]]
    species = arr[:, 1:]
    return names, species


def time_roadrunner(
    rr, t_start: float, t_end: float, n_steps: int, warmup: int, runs: int
) -> float:
    times: list[float] = []
    for i in range(warmup + runs):
        rr.reset()
        t0 = time.perf_counter()
        rr.simulate(t_start, t_end, n_steps + 1)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return float(median(times))


def compile_amici_module(sbml_str: str, model_name: str):
    import importlib

    import amici

    # AMICI-generated model modules expect amici.sim.sundials to be available.
    importlib.import_module("amici.sim")
    importlib.import_module("amici.sim.sundials")

    digest = hashlib.sha256(sbml_str.encode()).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", model_name)
    module_name = f"amici_ant_showcase_{safe_name}_{digest}"
    model_dir = AMICI_CACHE / module_name

    if model_dir.exists():
        try:
            return amici.import_model_module(module_name, str(model_dir))
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)

    AMICI_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        importer = amici.SbmlImporter(sbml_str, from_file=False)
        importer.sbml2amici(module_name, str(model_dir), verbose=False)
        return amici.import_model_module(module_name, str(model_dir))
    except Exception:
        shutil.rmtree(model_dir, ignore_errors=True)
        return None


def setup_amici_runtime(
    amici_module, t_start: float, t_end: float, n_steps: int, atol: float, rtol: float
) -> dict:
    import importlib

    if hasattr(amici_module, "getModel"):
        # AMICI <=0.x API
        model = amici_module.getModel()
        solver = model.getSolver()
        solver.setAbsoluteTolerance(atol)
        solver.setRelativeTolerance(rtol)
        model.setTimepoints(np.linspace(t_start, t_end, n_steps + 1))
        return {"api": "legacy", "model": model, "solver": solver}

    if hasattr(amici_module, "get_model"):
        # AMICI >=1.0 API
        model = amici_module.get_model()
        solver = model.create_solver()
        solver.set_absolute_tolerance(atol)
        solver.set_relative_tolerance(rtol)
        model.set_timepoints(np.linspace(t_start, t_end, n_steps + 1))
        sundials = importlib.import_module("amici.sim.sundials")
        return {"api": "modern", "model": model, "solver": solver, "sundials": sundials}

    raise RuntimeError("AMICI model module has neither getModel() nor get_model()")


def run_amici_once(runtime: dict):
    import amici

    model = runtime["model"]
    solver = runtime["solver"]
    api = runtime["api"]

    if api == "legacy":
        rdata = amici.runAmiciSimulation(model, solver)
        if getattr(rdata, "status", amici.AMICI_SUCCESS) != amici.AMICI_SUCCESS:
            raise RuntimeError(f"AMICI status={getattr(rdata, 'status', 'unknown')}")
        names = [str(n) for n in model.getStateIds()]
        species = np.asarray(rdata["x"], dtype=float)
    else:
        sundials = runtime["sundials"]
        rdata = sundials.run_simulation(model, solver)
        if getattr(rdata, "status", sundials.AMICI_SUCCESS) != sundials.AMICI_SUCCESS:
            raise RuntimeError(f"AMICI status={getattr(rdata, 'status', 'unknown')}")
        names = [str(n) for n in model.get_state_ids()]
        species = np.asarray(rdata.x, dtype=float)

    if species.ndim == 1:
        species = species.reshape((-1, 1))
    return names, species


def time_amici(runtime: dict, warmup: int, runs: int) -> float:
    times: list[float] = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        run_amici_once(runtime)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return float(median(times))


def geometric_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v > 0]
    if not vals:
        return None
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def build_figure_tex(
    fig_rel_path: str,
    total_n: int,
    rr_pass_n: int,
    amici_pass_n: int,
    rr_plot_n: int,
    amici_plot_n: int,
    rr_geomean: float | None,
    amici_geomean: float | None,
    rtol: float,
) -> str:
    rr_txt = "---" if rr_geomean is None else f"{rr_geomean:.2f}x"
    am_txt = "---" if amici_geomean is None else f"{amici_geomean:.2f}x"
    caption = (
        "Consistency-filtered cross-engine benchmark on hand-crafted Antimony models. "
        f"Pairwise cross-engine validation (max relative error <= {rtol:.0e}) passed for "
        f"{rr_pass_n}/{total_n} models (BNGsim vs libRoadRunner) and "
        f"{amici_pass_n}/{total_n} models (BNGsim vs AMICI). "
        f"Points shown are the consistency-passing models with successful timing: "
        f"libRoadRunner n={rr_plot_n}, AMICI n={amici_plot_n}. "
        f"Geometric mean ratios: RR/BNGsim={rr_txt}, AMICI/BNGsim={am_txt}."
    )
    return "\n".join(
        [
            "\\begin{figure}[!t]",
            "\\centering",
            f"\\includegraphics[width=\\columnwidth]{{{fig_rel_path}}}",
            f"\\caption{{{caption}}}",
            "\\label{fig:ant-exprtk-3engine}",
            "\\end{figure}",
        ]
    )


def generate_plot(results: list[dict], figure_path: Path, title: str | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rr_pts = [
        (r["bngsim_exprtk_time_s"] * 1e3, r["rr_time_s"] * 1e3)
        for r in results
        if r.get("rr_use_in_plot")
        and r.get("bngsim_exprtk_time_s", -1) > 0
        and r.get("rr_time_s", -1) > 0
    ]
    am_pts = [
        (r["bngsim_exprtk_time_s"] * 1e3, r["amici_time_s"] * 1e3)
        for r in results
        if r.get("amici_use_in_plot")
        and r.get("bngsim_exprtk_time_s", -1) > 0
        and r.get("amici_time_s", -1) > 0
    ]

    if not rr_pts and not am_pts:
        raise RuntimeError("no consistency-passing timing points to plot")

    fig, ax = plt.subplots(1, 1, figsize=(6.3, 5.0))

    if rr_pts:
        ax.scatter(
            [p[0] for p in rr_pts],
            [p[1] for p in rr_pts],
            s=22,
            alpha=0.70,
            marker="o",
            color="#1f77b4",
            edgecolors="none",
            label=f"libRoadRunner (n={len(rr_pts)})",
        )
    if am_pts:
        ax.scatter(
            [p[0] for p in am_pts],
            [p[1] for p in am_pts],
            s=28,
            alpha=0.75,
            marker="^",
            color="#ff7f0e",
            edgecolors="none",
            label=f"AMICI (n={len(am_pts)})",
        )

    all_vals = [v for p in rr_pts + am_pts for v in p]
    lo = max(min(all_vals) * 0.5, 1e-3)
    hi = max(all_vals) * 2.0

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9, color="black", alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("BNGsim ExprTK wall-clock time (ms)")
    ax.set_ylabel("Reference engine wall-clock time (ms)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    pdf_path = figure_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(figure_path),
        "pdf": str(pdf_path),
        "n_rr_points": len(rr_pts),
        "n_amici_points": len(am_pts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Showcase: Antimony BNGsim ExprTK vs RR/AMICI benchmark."
    )
    parser.add_argument("--quick", type=int, default=0, help="Limit to first N models.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per model/engine.")
    parser.add_argument(
        "--runs", type=int, default=5, help="Timed runs per model/engine (median reported)."
    )
    parser.add_argument(
        "--consistency-rtol", type=float, default=1e-3, help="Max relative error threshold."
    )
    parser.add_argument(
        "--consistency-atol",
        type=float,
        default=1e-8,
        help="Absolute floor for relative-error denom.",
    )
    parser.add_argument(
        "--solver-rtol", type=float, default=1e-8, help="ODE solver relative tolerance."
    )
    parser.add_argument(
        "--solver-atol", type=float, default=1e-8, help="ODE solver absolute tolerance."
    )
    parser.add_argument("--skip-amici", action="store_true", help="Skip AMICI entirely.")
    parser.add_argument(
        "--run-name", type=str, default=None, help="Custom run folder under showcase/results."
    )
    parser.add_argument(
        "--figure-path", type=str, default=str(FIGURE_DEFAULT), help="PNG path for output figure."
    )
    parser.add_argument("--skip-plot", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    run_name = args.run_name or f"ant_exprtk_3engine_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root = RESULTS_ROOT / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    have_amici = False
    if not args.skip_amici:
        try:
            import amici  # noqa: F401

            have_amici = True
        except Exception as exc:
            print(f"[warn] AMICI unavailable in current env: {exc}")
            print("[warn] continuing with BNGsim ExprTK vs libRoadRunner only")
            have_amici = False

    models = sorted(ANT_MODELS_DIR.glob("*.ant"))
    if args.quick > 0:
        models = models[: args.quick]

    print(f"[setup] models={len(models)} warmup={args.warmup} runs={args.runs} amici={have_amici}")

    rows: list[dict] = []

    for idx, ant_path in enumerate(models, start=1):
        name = ant_path.stem
        t_start, t_end, n_steps = parse_sim_tag(ant_path)
        row = {
            "model": name,
            "ant_path": str(ant_path),
            "t_start": t_start,
            "t_end": t_end,
            "n_steps": n_steps,
            "bngsim_exprtk_time_s": None,
            "rr_time_s": None,
            "amici_time_s": None,
            "rr_consistent": False,
            "amici_consistent": False,
            "rr_max_rel_err": None,
            "amici_max_rel_err": None,
            "rr_n_matched": 0,
            "amici_n_matched": 0,
            "rr_detail": "",
            "amici_detail": "",
            "rr_use_in_plot": False,
            "amici_use_in_plot": False,
            "error": "",
        }

        try:
            sbml_str = ant_to_sbml(ant_path)
            bng_model, bng_sim = setup_bngsim_exprtk(
                ant_path, atol=args.solver_atol, rtol=args.solver_rtol
            )
            bng_names, bng_species = run_bngsim_once(bng_model, bng_sim, t_start, t_end, n_steps)
            row["n_species"] = len(bng_names)

            rr = setup_roadrunner(sbml_str, args.solver_atol, args.solver_rtol)
            rr_names, rr_species = run_roadrunner_once(rr, t_start, t_end, n_steps)
            rr_pass, rr_err, rr_matched, rr_detail = cross_validate_by_name(
                bng_names,
                bng_species,
                rr_names,
                rr_species,
                rtol=args.consistency_rtol,
                atol=args.consistency_atol,
            )
            row["rr_consistent"] = rr_pass
            row["rr_max_rel_err"] = rr_err
            row["rr_n_matched"] = rr_matched
            row["rr_detail"] = rr_detail

            amici_mod = None
            amici_runtime = None
            if have_amici:
                amici_mod = compile_amici_module(sbml_str, name)
                if amici_mod is not None:
                    amici_runtime = setup_amici_runtime(
                        amici_mod, t_start, t_end, n_steps, args.solver_atol, args.solver_rtol
                    )
                    am_names, am_species = run_amici_once(amici_runtime)
                    am_pass, am_err, am_matched, am_detail = cross_validate_by_name(
                        bng_names,
                        bng_species,
                        am_names,
                        am_species,
                        rtol=args.consistency_rtol,
                        atol=args.consistency_atol,
                    )
                    row["amici_consistent"] = am_pass
                    row["amici_max_rel_err"] = am_err
                    row["amici_n_matched"] = am_matched
                    row["amici_detail"] = am_detail
                else:
                    row["amici_detail"] = "compile/import failed"

            if row["rr_consistent"] or row["amici_consistent"]:
                row["bngsim_exprtk_time_s"] = time_bngsim(
                    bng_model, bng_sim, t_start, t_end, n_steps, args.warmup, args.runs
                )

            if row["rr_consistent"]:
                row["rr_time_s"] = time_roadrunner(
                    rr, t_start, t_end, n_steps, args.warmup, args.runs
                )
                row["rr_use_in_plot"] = row["rr_time_s"] is not None and row["rr_time_s"] > 0

            if row["amici_consistent"] and amici_runtime is not None:
                row["amici_time_s"] = time_amici(amici_runtime, args.warmup, args.runs)
                row["amici_use_in_plot"] = (
                    row["amici_time_s"] is not None and row["amici_time_s"] > 0
                )

        except Exception as exc:
            row["error"] = str(exc)

        rows.append(row)

        parts = [f"[{idx}/{len(models)}] {name}"]
        if row.get("error"):
            parts.append(f"error={row['error']}")
        else:
            rr_tag = "pass" if row["rr_consistent"] else "fail"
            parts.append(
                f"rr={rr_tag}({row['rr_max_rel_err']:.2e})"
                if row["rr_max_rel_err"] is not None
                else "rr=na"
            )
            if have_amici:
                am_tag = "pass" if row["amici_consistent"] else "fail"
                parts.append(
                    f"amici={am_tag}({row['amici_max_rel_err']:.2e})"
                    if row["amici_max_rel_err"] is not None
                    else "amici=na"
                )
            if row["bngsim_exprtk_time_s"] is not None:
                parts.append(f"bng={row['bngsim_exprtk_time_s'] * 1e3:.2f}ms")
            if row["rr_time_s"] is not None:
                parts.append(f"rr_t={row['rr_time_s'] * 1e3:.2f}ms")
            if row["amici_time_s"] is not None:
                parts.append(f"am_t={row['amici_time_s'] * 1e3:.2f}ms")
        print("  " + " ".join(parts))

    rr_ratios = [
        r["rr_time_s"] / r["bngsim_exprtk_time_s"]
        for r in rows
        if r.get("rr_use_in_plot") and r.get("rr_time_s") and r.get("bngsim_exprtk_time_s")
    ]
    amici_ratios = [
        r["amici_time_s"] / r["bngsim_exprtk_time_s"]
        for r in rows
        if r.get("amici_use_in_plot") and r.get("amici_time_s") and r.get("bngsim_exprtk_time_s")
    ]

    summary = {
        "n_models_total": len(rows),
        "n_rr_crossval_pass": sum(1 for r in rows if r.get("rr_consistent")),
        "n_amici_crossval_pass": sum(1 for r in rows if r.get("amici_consistent")),
        "n_both_crossval_pass": sum(
            1 for r in rows if r.get("rr_consistent") and r.get("amici_consistent")
        ),
        "n_rr_consistent_and_timed": len(rr_ratios),
        "n_amici_consistent_and_timed": len(amici_ratios),
        "rr_over_bng_geomean": geometric_mean(rr_ratios),
        "amici_over_bng_geomean": geometric_mean(amici_ratios),
        "consistency_rtol": args.consistency_rtol,
        "consistency_atol": args.consistency_atol,
    }

    figure_info = None
    figure_path = Path(args.figure_path).resolve()
    if not args.skip_plot:
        try:
            figure_info = generate_plot(
                rows,
                figure_path,
                title=None,
            )
        except RuntimeError as exc:
            figure_info = {
                "error": str(exc),
                "png": str(figure_path),
                "pdf": str(figure_path.with_suffix(".pdf")),
            }
            print(f"[warn] plot skipped: {exc}")

    output = {
        "meta": _machine_info(),
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "solver_rtol": args.solver_rtol,
            "solver_atol": args.solver_atol,
            "consistency_rtol": args.consistency_rtol,
            "consistency_atol": args.consistency_atol,
            "amici_enabled": have_amici,
            "ant_models_dir": str(ANT_MODELS_DIR),
            "figure_path": str(figure_path),
        },
        "summary": summary,
        "figure": figure_info,
        "results": rows,
    }

    out_json = run_root / "results.json"
    out_json.write_text(json.dumps(output, indent=2, default=_json_default))

    figure_tex = None
    if figure_info is not None and "error" not in figure_info:
        fig_rel = os.path.relpath(figure_path, PAPER_LATEX).replace("\\", "/")
        figure_tex = build_figure_tex(
            fig_rel,
            total_n=summary["n_models_total"],
            rr_pass_n=summary["n_rr_crossval_pass"],
            amici_pass_n=summary["n_amici_crossval_pass"],
            rr_plot_n=summary["n_rr_consistent_and_timed"],
            amici_plot_n=summary["n_amici_consistent_and_timed"],
            rr_geomean=summary["rr_over_bng_geomean"],
            amici_geomean=summary["amici_over_bng_geomean"],
            rtol=args.consistency_rtol,
        )
        FIGURE_SNIPPET.parent.mkdir(parents=True, exist_ok=True)
        FIGURE_SNIPPET.write_text(figure_tex + "\n")

    print(f"[done] wrote {out_json}")
    if figure_info is not None and "error" not in figure_info:
        print(f"[done] wrote {figure_info['png']}")
        print(f"[done] wrote {figure_info['pdf']}")
        print(f"[done] wrote {FIGURE_SNIPPET}")
    elif figure_info is not None and "error" in figure_info:
        print(f"[done] no figure generated ({figure_info['error']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
