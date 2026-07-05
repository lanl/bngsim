#!/usr/bin/env python3
"""PyBNF fitting benchmark: run_network vs BNGsim auto.

This benchmark executes PyBNF example fitting jobs with simulation paths:
  1) BioNetGen subprocess path (run_network): PYBNF_NO_BNGSIM=1
  2) BNGsim auto path: default bngsim in-process execution (auto engine selection)

Only non-MCMC configurations are included by default. Results include per-engine
total wall time, final best objective score, median completed objective-evaluation
count (from PyBNF algorithm pickles), and median ``sorted_params_final`` row count.

Default job list: Mitra (2019) Table~2 **B-ode + scatter search** (12 models) via
``Mitra2019_models/table2_b_ode_ss_scatter_manifest.json`` (falls back to
``--examples-dir`` if that file is absent). Pass ``--examples`` to scan the LANL
PyBNF ``examples/`` tree instead. Legacy **DE** configs:
``table2_b_ode_benchmark_manifest.json``. Manifests may set ``preserve_fit_type``;
CLI scales ``max_iterations`` / ``population_size`` unless overridden.

Scatter search is stochastic: by default the harness injects ``seed=...`` into
rewritten configs so **run_network** and **bngsim_auto** share the same PyBNF
RNG state for each (model, replicate). Default ``max_iterations`` is 100 for
comparable budgets across models (use ``--quick`` for short smoke tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import contextlib

from common import REPO_ROOT, RESULTS_DIR, get_machine_info, save_results

DEFAULT_PYBNF_MANIFEST_REL = "Mitra2019_models/table2_b_ode_ss_scatter_manifest.json"

MCMC_FIT_TYPES = {"mh", "pt", "sa", "dream", "am", "bmc"}
DEFAULT_ENGINES = ["run_network", "bngsim_auto"]


@dataclass
class ConfigCase:
    config_path: Path
    model_dir: Path
    model_id: str
    fit_type: str
    model_decl: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def pybnf_benchmark_seed(base_seed: int, model_id: str, replicate_index: int) -> int:
    """Integer seed shared by both engines for (model_id, replicate_index).

    Derived from a SHA-256 digest so different models/replicates do not collide.
    PyBNF applies this via ``numpy.random.seed`` / ``random.seed`` at startup.
    """
    msg = f"{int(base_seed)}|{model_id}|{int(replicate_index)}".encode()
    h = hashlib.sha256(msg).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 2) + 1


def parse_fit_type(conf_text: str) -> str:
    m = re.search(r"(?im)^\s*fit_type\s*=\s*([A-Za-z0-9_]+)\s*$", conf_text)
    return m.group(1).strip().lower() if m else "de"


def parse_model_decl(conf_text: str) -> str:
    m = re.search(r"(?im)^\s*model\s*=\s*(.+?)\s*$", conf_text)
    return m.group(1).strip() if m else ""


def parse_model_file_from_decl(model_decl: str) -> str:
    if not model_decl:
        return ""
    lhs = model_decl.split(":", 1)[0].strip()
    return lhs


def parse_data_type(model_decl: str) -> str:
    """Classify data inputs in the model declaration."""
    if ":" not in model_decl:
        return "unknown"
    rhs = model_decl.split(":", 1)[1]
    toks = [t.strip().lower() for t in rhs.split(",") if t.strip()]
    has_exp = any(t.endswith(".exp") for t in toks)
    has_prop = any(t.endswith(".prop") for t in toks)
    if has_exp and has_prop:
        return "quant+qual"
    if has_exp:
        return "quant"
    if has_prop:
        return "qual"
    return "unknown"


def is_ss_convertible_fit_type(fit_type: str) -> bool:
    """Return True for fit types that can be safely coerced to scatter-search."""
    ft = (fit_type or "").strip().lower()
    return ft not in MCMC_FIT_TYPES and ft not in {"sim"}


def _load_pybnf_alg_success_fail(pickle_path: Path) -> tuple[int, int] | None:
    """Return (success_count, fail_count) from a PyBNF algorithm pickle."""
    try:
        with open(pickle_path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    alg = obj[0] if isinstance(obj, tuple) else obj
    try:
        s = int(getattr(alg, "success_count", 0) or 0)
        fc = int(getattr(alg, "fail_count", 0) or 0)
    except (TypeError, ValueError):
        return None
    return (s, fc)


def parse_objective_eval_counts(output_dir: Path) -> dict | None:
    """Count completed objective evaluations from PyBNF's saved algorithm state.

    Uses ``success_count`` + ``fail_count`` on the unpickled ``Algorithm`` object
    (every returned simulation job, successful or failed), summed over main and
    optional refinement phases. Falls back to ``alg_backup.bp`` when no
    ``alg_finished.bp`` exists (e.g. timeout mid-run).
    """
    total_s = 0
    total_f = 0
    phases: list[dict] = []
    for fname, phase in (
        ("alg_finished.bp", "main"),
        ("alg_refine_finished.bp", "refine"),
    ):
        p = output_dir / fname
        if not p.is_file():
            continue
        pair = _load_pybnf_alg_success_fail(p)
        if not pair:
            continue
        s, fc = pair
        phases.append({"phase": phase, "success": s, "fail": fc, "total": s + fc})
        total_s += s
        total_f += fc

    if total_s + total_f == 0:
        pb = output_dir / "alg_backup.bp"
        if pb.is_file():
            pair = _load_pybnf_alg_success_fail(pb)
            if pair:
                s, fc = pair
                if s + fc > 0:
                    phases.append({"phase": "backup", "success": s, "fail": fc, "total": s + fc})
                    total_s, total_f = s, fc

    if total_s + total_f == 0:
        return None
    return {
        "success": total_s,
        "fail": total_f,
        "total": total_s + total_f,
        "phases": phases,
    }


def aggregate_objective_eval_counts(output_dirs: list[Path]) -> dict | None:
    """Median (and range) of total eval counts across replicate/engine output dirs."""
    totals: list[int] = []
    details: list[dict] = []
    for d in output_dirs:
        c = parse_objective_eval_counts(d)
        if c and c.get("total", 0) > 0:
            totals.append(int(c["total"]))
            details.append(c)
    if not totals:
        return None
    return {
        "median_total": float(statistics.median(totals)),
        "min_total": min(totals),
        "max_total": max(totals),
        "per_output_dir": details,
    }


def count_sorted_params_final_rows(output_dir: Path) -> int | None:
    """Non-comment data lines in ``Results/sorted_params_final.txt`` (ranked bests)."""
    p = output_dir / "Results" / "sorted_params_final.txt"
    if not p.is_file():
        return None
    n = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        n += 1
    return n if n > 0 else None


def aggregate_sorted_params_final_rows(output_dirs: list[Path]) -> int | None:
    """Median count of reported rows in ``sorted_params_final.txt`` across dirs."""
    vals = [count_sorted_params_final_rows(p) for p in output_dirs]
    vals = [v for v in vals if isinstance(v, int) and v > 0]
    if not vals:
        return None
    return int(round(statistics.median(vals)))


def discover_cases(examples_dir: Path, include_mcmc: bool = False) -> list[ConfigCase]:
    cases: list[ConfigCase] = []
    for conf in sorted(examples_dir.glob("*/*.conf")):
        text = _read_text(conf)
        fit_type = parse_fit_type(text)
        if (not include_mcmc) and fit_type in MCMC_FIT_TYPES:
            continue
        model_decl = parse_model_decl(text)
        if not model_decl:
            continue
        model_file = parse_model_file_from_decl(model_decl)
        if not model_file.lower().endswith(".bngl"):
            # Keep this benchmark focused on BNGL/BNG2.pl simulation path.
            continue
        model_id = f"{conf.parent.name}/{conf.stem}"
        cases.append(
            ConfigCase(
                config_path=conf,
                model_dir=conf.parent,
                model_id=model_id,
                fit_type=fit_type,
                model_decl=model_decl,
            )
        )
    return cases


def load_manifest_cases(repo_root: Path, manifest_path: Path) -> tuple[list[ConfigCase], dict]:
    """Load explicit PyBNF jobs from a JSON manifest (repo-root-relative paths)."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = raw.get("entries") or raw.get("configs")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no entries: {manifest_path}")

    meta = {k: raw[k] for k in ("description", "reference", "preserve_fit_type") if k in raw}
    cases: list[ConfigCase] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        rel = e.get("config") or e.get("path")
        if not rel:
            raise ValueError(f"Manifest entry missing config path: {e}")
        conf_path = (repo_root / str(rel)).resolve()
        if not conf_path.is_file():
            raise FileNotFoundError(f"Manifest config not found: {conf_path}")
        text = _read_text(conf_path)
        fit_type = parse_fit_type(text)
        model_decl = parse_model_decl(text)
        if not model_decl:
            raise ValueError(f"No model= line in {conf_path}")
        model_file = parse_model_file_from_decl(model_decl)
        if not model_file.lower().endswith(".bngl"):
            raise ValueError(
                f"Manifest job must use a .bngl model (got {model_file!r}): {conf_path}"
            )
        model_id = str(e.get("model_id") or f"mitra2019/{conf_path.parent.name}/{conf_path.stem}")
        cases.append(
            ConfigCase(
                config_path=conf_path,
                model_dir=conf_path.parent,
                model_id=model_id,
                fit_type=fit_type,
                model_decl=model_decl,
            )
        )
    return cases, meta


def rewrite_conf_text(
    original: str,
    output_dir: str,
    bng_command: str,
    max_iterations: int,
    population_size: int,
    parallel_count: int,
    verbosity: int,
    force_fit_type: str = "",
    seed: int | None = None,
) -> str:
    key_vals = {
        "output_dir": output_dir,
        "bng_command": bng_command,
        "max_iterations": str(max_iterations),
        "population_size": str(population_size),
        "parallel_count": str(parallel_count),
        "verbosity": str(verbosity),
        "delete_old_files": "1",
        "backup_every": "1",
        "output_every": "5",
        # Dask 2025+/2026 no longer permits the old Future usage pattern
        # used by PyBNF worker objective evaluation; force objective on driver.
        "local_objective_eval": "1",
        # Use nearest independent-variable matching. Some in-process parameter_scan
        # paths produce floating grids that differ slightly from exp file literals.
        "ind_var_rounding": "1",
    }
    if force_fit_type:
        key_vals["fit_type"] = force_fit_type
    if seed is not None and seed >= 0:
        key_vals["seed"] = str(int(seed))
    present = {k: False for k in key_vals}
    out_lines: list[str] = []

    for line in original.splitlines():
        replaced = False
        for k, v in key_vals.items():
            if re.match(rf"^\s*{re.escape(k)}\s*=", line, flags=re.IGNORECASE):
                out_lines.append(f"{k}={v}")
                present[k] = True
                replaced = True
                break
        if not replaced:
            out_lines.append(line)

    for k, v in key_vals.items():
        if not present[k]:
            out_lines.append(f"{k}={v}")

    return "\n".join(out_lines).rstrip() + "\n"


def parse_best_objective(output_dir: Path) -> float | None:
    candidates = [
        output_dir / "Results" / "sorted_params_final.txt",
        output_dir / "Results" / "sorted_params.txt",
        output_dir / "Results" / "sorted_params_backup.txt",
    ]
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 2:
                continue
            try:
                return float(parts[1])
            except Exception:
                continue
    return None


def run_one_engine(
    py_exec: Path,
    repo_root: Path,
    case: ConfigCase,
    engine: str,
    conf_path: Path,
    output_dir: Path,
    timeout_s: int,
    bngpath_dir: Path,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["BNGPATH"] = str(bngpath_dir)

    # Engine switches.
    env.pop("PYBNF_NO_BNGSIM", None)
    env.pop("BNGSIM_NO_CODEGEN", None)
    if engine == "run_network":
        env["PYBNF_NO_BNGSIM"] = "1"
        env["BNGSIM_NO_CODEGEN"] = "1"
    elif engine == "bngsim_auto":
        pass
    else:
        return {"status": "error", "error": f"unknown_engine_{engine}"}

    cmd = [str(py_exec), "-m", "pybnf", "-c", str(conf_path), "-o"]

    model_rel = parse_model_file_from_decl(case.model_decl)
    run_cwd = case.model_dir
    if model_rel and not Path(model_rel).is_absolute():
        # Some Mitra configs live in subfolders (e.g., fit_ss/) while model files
        # are one level up. Walk upward until the declared model path resolves.
        probe = run_cwd
        for _ in range(4):
            if (probe / model_rel).exists():
                run_cwd = probe
                break
            if probe == repo_root:
                break
            probe = probe.parent

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(run_cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        wall_s = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        wall_s = time.perf_counter() - t0
        # Even on timeout, PyBNF may have written backup/final parameter files.
        # Keep best-so-far objective so expensive models still contribute signal.
        objective = parse_best_objective(output_dir)
        out = {
            "status": "timeout_with_partial" if objective is not None else "timeout",
            "wall_s": wall_s,
            "objective": objective,
            "error": f"timeout_{timeout_s}s",
        }
        err_src = exc.stderr if exc.stderr is not None else exc.stdout
        if isinstance(err_src, bytes):
            err_src = err_src.decode("utf-8", errors="replace")
        err_tail = (err_src or "")[-1000:]
        if err_tail.strip():
            out["error"] = f"{out['error']} | {err_tail.strip().replace(chr(10), ' | ')}"
        return out

    objective = parse_best_objective(output_dir)
    status = "ok" if proc.returncode == 0 else "error"
    out = {
        "status": status,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "objective": objective,
    }
    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "")[-1000:]
        out["error"] = err_tail.strip().replace("\n", " | ")
    return out


def resolve_bng_command(repo_root: Path) -> tuple[Path, str]:
    candidates = [
        repo_root / "BioNetGen-2.9.3" / "BNG2.pl",
        repo_root / "bionetgen-2.9.3" / "BNG2.pl",
    ]
    for p in candidates:
        if p.exists():
            return p.parent, str(p)
    raise FileNotFoundError("Could not find local BNG2.pl under repo root")


def main():
    ap = argparse.ArgumentParser(
        description="PyBNF fitting benchmark (BNGsim auto vs run_network)"
    )
    ap.add_argument("--examples-dir", default=str(REPO_ROOT / "examples"))
    ap.add_argument(
        "--examples",
        action="store_true",
        help=(
            "Discover configs under --examples-dir (LANL PyBNF layout) instead "
            "of the default Mitra Table~2 scatter manifest."
        ),
    )
    ap.add_argument(
        "--manifest",
        default=DEFAULT_PYBNF_MANIFEST_REL,
        help=(
            "JSON manifest of configs (repo-root-relative), default: Mitra 12-job "
            "B-ode scatter list. Use table2_b_ode_benchmark_manifest.json for the "
            "DE variant. If this file is missing, falls back to --examples-dir discovery. "
            "Ignored when --examples is set."
        ),
    )
    ap.add_argument("--main-count", type=int, default=12)
    ap.add_argument(
        "--max-models", type=int, default=0, help="Limit number of eligible models (0=all)"
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="PyBNF max_iterations injected into rewritten configs (default: 100)",
    )
    ap.add_argument("--population-size", type=int, default=8)
    ap.add_argument(
        "--parallel-count",
        type=int,
        default=8,
        help="Override PyBNF parallel_count to avoid unstable worker explosions",
    )
    ap.add_argument("--verbosity", type=int, default=0)
    ap.add_argument(
        "--timeout-s",
        type=int,
        default=7200,
        help="Subprocess wall-clock limit per PyBNF run in seconds (default: 7200)",
    )
    ap.add_argument(
        "--pybnf-base-seed",
        type=int,
        default=20190219,
        help="Base integer for deterministic per-(model,replicate) PyBNF RNG seeds (default: 20190219)",
    )
    ap.add_argument(
        "--unseeded",
        action="store_true",
        help="Do not inject seed= into configs (legacy uncontrolled stochastic search; blocked unless --allow-unseeded)",
    )
    ap.add_argument(
        "--allow-unseeded",
        action="store_true",
        help="Allow --unseeded runs (not publication-ready)",
    )
    ap.add_argument(
        "--allow-low-iterations",
        action="store_true",
        help="Allow non-quick runs with max_iterations < 100 (not publication-ready)",
    )
    ap.add_argument("--include-mcmc", action="store_true")
    ap.add_argument(
        "--fit-types",
        default="ss,de,pso,sim,ade",
        help="Comma-separated fit types to include (default: ss,de,pso,sim,ade)",
    )
    ap.add_argument("--model-filter", default="", help="Substring filter on model_id")
    ap.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    ap.add_argument(
        "--force-fit-type",
        default=None,
        metavar="TYPE",
        help="Rewrite configs to this fit_type. Default: ss for examples/ discovery; "
        "for manifests with preserve_fit_type, keep config unless this flag is set.",
    )
    ap.add_argument(
        "--preserve-config-fit-type",
        action="store_true",
        help="Do not override fit_type in .conf (still applies max_iterations/population/etc.)",
    )
    ap.add_argument("--quick", action="store_true", help="Very small smoke run")
    ap.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of stochastic repeats per engine/model (minimum enforced: 3)",
    )
    args = ap.parse_args()

    repo_root = REPO_ROOT
    py_exec = repo_root / ".venv" / "bin" / "python"
    if not py_exec.exists():
        raise FileNotFoundError(f"Missing venv python: {py_exec}")

    manifest_meta: dict = {}
    manifest_path: Path | None = None
    if args.examples:
        examples_dir = Path(args.examples_dir).resolve()
        if not examples_dir.exists():
            raise FileNotFoundError(f"Examples directory not found: {examples_dir}")
        cases = discover_cases(examples_dir, include_mcmc=args.include_mcmc)
    else:
        mp = (args.manifest or "").strip()
        manifest_path = Path(mp)
        if not manifest_path.is_absolute():
            manifest_path = (repo_root / manifest_path).resolve()
        if manifest_path.is_file():
            cases, manifest_meta = load_manifest_cases(repo_root, manifest_path)
        elif mp == DEFAULT_PYBNF_MANIFEST_REL:
            print(
                f"Note: default manifest not found ({manifest_path}); "
                "using --examples-dir discovery instead.",
                file=sys.stderr,
            )
            examples_dir = Path(args.examples_dir).resolve()
            if not examples_dir.exists():
                raise FileNotFoundError(f"Examples directory not found: {examples_dir}")
            cases = discover_cases(examples_dir, include_mcmc=args.include_mcmc)
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    bngpath_dir, bng_command = resolve_bng_command(repo_root)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    max_iterations = args.max_iterations
    population_size = args.population_size
    parallel_count = max(1, int(args.parallel_count))
    timeout_s = args.timeout_s
    max_models = args.max_models
    # Scatter-search fitting is stochastic; enforce replicates for stable reporting.
    replicates = max(3, int(args.replicates))
    if args.quick:
        max_iterations = min(max_iterations, 3)
        population_size = min(population_size, 4)
        timeout_s = min(timeout_s, 300)
        if max_models <= 0:
            max_models = 2
        # Keep quick runs short but still stochastic-aware.
        replicates = min(replicates, 3)
    elif max_iterations < 100 and not args.allow_low_iterations:
        raise RuntimeError(
            "Publication protocol requires max_iterations >= 100 for non-quick runs. "
            "Use --allow-low-iterations to override for diagnostics."
        )

    if args.unseeded and not args.allow_unseeded:
        raise RuntimeError(
            "--unseeded is disabled by default for publication readiness. "
            "Pass --allow-unseeded to run legacy uncontrolled mode."
        )

    allowed_fit_types = {t.strip().lower() for t in args.fit_types.split(",") if t.strip()}
    if allowed_fit_types:
        cases = [c for c in cases if c.fit_type in allowed_fit_types]
    if args.preserve_config_fit_type:
        force_fit_type_effective = ""
    elif args.force_fit_type is not None:
        force_fit_type_effective = args.force_fit_type.strip()
    elif manifest_meta.get("preserve_fit_type"):
        force_fit_type_effective = ""
    else:
        force_fit_type_effective = "ss"

    if force_fit_type_effective:
        cases = [c for c in cases if is_ss_convertible_fit_type(c.fit_type)]
    if args.model_filter:
        q = args.model_filter.lower()
        cases = [c for c in cases if q in c.model_id.lower()]
    if max_models > 0:
        cases = cases[:max_models]

    if not cases:
        raise RuntimeError("No eligible example configs found for benchmark")

    main_ids = {c.model_id for c in cases[: min(args.main_count, len(cases))]}

    run_root = RESULTS_DIR / "pybnf_fit_runs"
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PyBNF fitting benchmark: run_network vs BNGsim auto")
    print(f"Eligible cases: {len(cases)}")
    print(f"Main table models: {min(args.main_count, len(cases))}")
    print(f"Engines: {', '.join(engines)}")
    print(f"Using BNG2.pl: {bng_command}")
    print("=" * 90)

    results = []
    for i, case in enumerate(cases, start=1):
        group = "main" if case.model_id in main_ids else "supplement"
        print(f"\n[{i}/{len(cases)}] {case.model_id} ({case.fit_type}, {group})")

        model_result = {
            "model_id": case.model_id,
            "config_path": str(case.config_path),
            "fit_type": case.fit_type,
            "data_type": parse_data_type(case.model_decl),
            "group": group,
            "engines": {},
        }

        for eng in engines:
            rep_runs = []
            for rep in range(replicates):
                out_dir = run_root / f"{case.model_id.replace('/', '__')}__{eng}__r{rep + 1}"
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.parent.mkdir(parents=True, exist_ok=True)

                conf_text = _read_text(case.config_path)
                run_seed = (
                    None
                    if args.unseeded
                    else pybnf_benchmark_seed(args.pybnf_base_seed, case.model_id, rep)
                )
                patched = rewrite_conf_text(
                    conf_text,
                    output_dir=str(out_dir),
                    bng_command=bng_command,
                    max_iterations=max_iterations,
                    population_size=population_size,
                    parallel_count=parallel_count,
                    verbosity=args.verbosity,
                    force_fit_type=force_fit_type_effective,
                    seed=run_seed,
                )
                temp_conf = case.model_dir / f".bench_pybnf_{eng}_r{rep + 1}.conf"
                temp_conf.write_text(patched, encoding="utf-8")

                try:
                    run = run_one_engine(
                        py_exec=py_exec,
                        repo_root=repo_root,
                        case=case,
                        engine=eng,
                        conf_path=temp_conf,
                        output_dir=out_dir,
                        timeout_s=timeout_s,
                        bngpath_dir=bngpath_dir,
                    )
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        temp_conf.unlink()
                rep_runs.append({"replicate": rep + 1, **run})

            ok_runs = [r for r in rep_runs if r.get("status") == "ok"]
            wall_vals = [
                float(r["wall_s"]) for r in ok_runs if isinstance(r.get("wall_s"), (int, float))
            ]
            obj_vals = [
                float(r["objective"])
                for r in ok_runs
                if isinstance(r.get("objective"), (int, float))
            ]
            agg = {
                "status": "ok" if ok_runs else "error",
                "wall_s": statistics.median(wall_vals) if wall_vals else None,
                "objective": min(obj_vals) if obj_vals else None,
                "replicates": rep_runs,
            }
            if not ok_runs:
                errs = [r.get("error", "") for r in rep_runs if r.get("error")]
                agg["error"] = " | ".join(errs[-3:]) if errs else "all_replicates_failed"
            model_result["engines"][eng] = agg
            print(
                f"  - {eng:<14} {agg['status']:<8} "
                f"wall_med={agg.get('wall_s')} obj_best={agg.get('objective')} "
                f"(n={replicates})"
            )

        out_dirs = [
            run_root / f"{case.model_id.replace('/', '__')}__{eng}__r{rep + 1}"
            for eng in engines
            for rep in range(replicates)
        ]
        ev = aggregate_objective_eval_counts(out_dirs)
        if ev:
            model_result["objective_evaluations"] = int(round(ev["median_total"]))
            model_result["objective_evaluations_detail"] = ev
        spr = aggregate_sorted_params_final_rows(out_dirs)
        if spr is not None:
            model_result["sorted_params_final_rows_median"] = spr

        results.append(model_result)

    if args.unseeded:
        seed_policy = (
            "Uncontrolled: no seed= line injected; PyBNF RNG state depends on process environment."
        )
    else:
        seed_policy = (
            "Deterministic per (model_id, replicate_index): SHA-256 digest from "
            f"base_seed={args.pybnf_base_seed}|model_id|replicate → integer seed; "
            "same seed for run_network and bngsim_auto so numpy.random matches at PyBNF startup."
        )
    summary = {
        "n_cases": len(cases),
        "n_main": len(main_ids),
        "engines": engines,
        "max_iterations": max_iterations,
        "population_size": population_size,
        "parallel_count": parallel_count,
        "timeout_s": timeout_s,
        "replicates": replicates,
        "force_fit_type_applied": force_fit_type_effective or None,
        "pybnf_base_seed": None if args.unseeded else int(args.pybnf_base_seed),
        "unseeded": bool(args.unseeded),
        "seed_policy": seed_policy,
        "objective_interpretation": (
            "Per-engine objective is min over successful replicates of parse_best_objective (best row in "
            "sorted_params_final.txt). With matched seeds, scatter-search proposal sequences align across "
            "engines when the optimizer code path is identical; objective values can still differ because "
            "chi-squared depends on simulated trajectories and run_network vs BNGsim ODE paths are not "
            "bit-identical. Default benchmark budget uses max_iterations=100 (not full production Mitra fits)."
        ),
        "publication_protocol_ok": bool(
            (not args.unseeded) and (not args.quick) and (max_iterations >= 100)
        ),
    }
    output = {
        "machine_info": get_machine_info(),
        "summary": summary,
        "results": results,
    }
    if manifest_path is not None:
        output["manifest"] = {
            "path": str(manifest_path),
            **{k: manifest_meta[k] for k in manifest_meta if k != "preserve_fit_type"},
            "preserve_fit_type": manifest_meta.get("preserve_fit_type"),
        }
    save_results(output, "bench_pybnf_fitting")


if __name__ == "__main__":
    main()
