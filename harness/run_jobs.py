#!/usr/bin/env python3
"""Run benchmark jobs defined in jobs.yaml.

Reads the job manifest and executes enabled jobs in dependency order.
Each job is a Python script invoked as a subprocess with optional CLI flags.

Usage:
    python run_jobs.py                          # all enabled jobs
    python run_jobs.py --job bench_ssa          # single job by id
    python run_jobs.py --phase validation       # all validation jobs
    python run_jobs.py --phase comparison       # all comparison jobs
    python run_jobs.py --list                   # show job summary
    python run_jobs.py --dry-run                # show what would run

Environment:
    Must be run from repo root with the root .venv active.
    See bngsim/dev/skills/env-setup-uv.md for setup instructions.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
JOBS_YAML = HARNESS_DIR / "jobs.yaml"
RESULTS_DIR = HARNESS_DIR / "results"


def load_manifest() -> dict:
    """Load and parse jobs.yaml."""
    with open(JOBS_YAML) as f:
        return yaml.safe_load(f)


def get_jobs(manifest: dict) -> list[dict]:
    """Extract job list from manifest."""
    return manifest.get("jobs", [])


def resolve_dependencies(jobs: list[dict]) -> list[dict]:
    """Sort jobs respecting depends_on ordering."""
    by_id = {j["id"]: j for j in jobs}
    visited = set()
    order = []

    def visit(job_id):
        if job_id in visited:
            return
        visited.add(job_id)
        job = by_id.get(job_id)
        if not job:
            return
        for dep in job.get("depends_on", []):
            visit(dep)
        order.append(job)

    for j in jobs:
        visit(j["id"])
    return order


def check_environment():
    """Verify we're in the correct environment."""
    errors = []

    venv = os.environ.get("VIRTUAL_ENV", "")
    if not venv.endswith("/.venv"):
        errors.append(
            f"VIRTUAL_ENV={venv!r} — expected to end with /.venv\n"
            f"  Run from repo root with direnv active."
        )

    bngsim_venv = REPO_ROOT / "bngsim" / ".venv"
    if bngsim_venv.exists():
        errors.append(f"bngsim/.venv exists — delete it: rm -rf {bngsim_venv}")

    try:
        import bngsim  # noqa: F401
    except ImportError:
        errors.append(
            "Cannot import bngsim. Install with: uv pip install --no-build-isolation -e ./bngsim"
        )

    return errors


def capture_metadata(run_dir: Path) -> dict:
    """Write meta.json to the run directory."""
    meta = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }

    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        meta["git_short"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        pass

    try:
        import bngsim

        meta["bngsim_version"] = getattr(bngsim, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import roadrunner

        meta["roadrunner_version"] = getattr(roadrunner, "__version__", "unknown")
    except ImportError:
        pass

    meta_path = run_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def run_job(job: dict, extra_flags: list[str] | None = None) -> dict:
    """Execute a single benchmark job. Returns result dict."""
    job_id = job["id"]
    script = job["script"]
    script_path = HARNESS_DIR / script

    if not script_path.exists():
        return {
            "id": job_id,
            "status": "error",
            "error": f"Script not found: {script_path}",
        }

    cmd = [sys.executable, str(script_path)]

    # Add CLI flags from manifest
    cli_flags = job.get("cli_flags", "")
    if cli_flags:
        cmd.extend(cli_flags.split())

    # Add extra flags (e.g., --quick N)
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"\n{'═' * 70}")
    print(f"  JOB: {job_id}")
    print(f"  Script: {script}")
    print(f"  Phase: {job.get('phase', '?')}")
    print(f"  Estimated: {job.get('estimated_time', '?')}")
    print(f"{'═' * 70}\n")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=HARNESS_DIR,
            text=True,
            # Stream output live
            stdout=None,
            stderr=None,
        )
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else "failed"
        return {
            "id": job_id,
            "status": status,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "id": job_id,
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(time.time() - t0, 1),
        }


def print_job_list(jobs: list[dict]):
    """Print a summary table of all jobs."""
    print(f"\n{'ID':<30} {'Phase':<14} {'Enabled':<9} {'Est. Time':<15} Description")
    print("─" * 100)
    for j in jobs:
        enabled = "✓" if j.get("enabled", True) else "✗"
        desc = j.get("description", "")[:40].strip().replace("\n", " ")
        print(
            f"{j['id']:<30} {j.get('phase', '?'):<14} "
            f"{enabled:<9} {j.get('estimated_time', '?'):<15} {desc}…"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Run benchmark jobs from jobs.yaml")
    parser.add_argument(
        "--job",
        "-j",
        help="Run a single job by id",
    )
    parser.add_argument(
        "--phase",
        "-p",
        choices=["validation", "comparison", "calibration", "tables"],
        help="Run all jobs in a phase",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all jobs and exit",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would run without executing",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip environment validation",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip timestamped results directory creation",
    )
    parser.add_argument(
        "extra_flags",
        nargs="*",
        help="Extra flags passed to each job script (e.g., --quick 10)",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    all_jobs = get_jobs(manifest)

    if args.list:
        print_job_list(all_jobs)
        return

    # Filter jobs
    if args.job:
        jobs = [j for j in all_jobs if j["id"] == args.job]
        if not jobs:
            ids = [j["id"] for j in all_jobs]
            print(f"ERROR: Job '{args.job}' not found. Available: {ids}")
            sys.exit(1)
    elif args.phase:
        jobs = [j for j in all_jobs if j.get("phase") == args.phase and j.get("enabled", True)]
    else:
        jobs = [j for j in all_jobs if j.get("enabled", True)]

    # Resolve dependencies
    jobs = resolve_dependencies(jobs)

    if not jobs:
        print("No jobs to run.")
        return

    if args.dry_run:
        print("\nDry run — would execute these jobs:\n")
        for j in jobs:
            print(f"  [{j.get('phase', '?')}] {j['id']}")
            print(f"    → {j['script']}")
            if j.get("cli_flags"):
                print(f"    flags: {j['cli_flags']}")
        return

    # Environment check
    if not args.skip_env_check:
        errors = check_environment()
        if errors:
            print("❌ Environment check failed:\n")
            for e in errors:
                print(f"  • {e}")
            print("\nFix these issues or use --skip-env-check to override.")
            sys.exit(1)
        print("✅ Environment check passed")

    # Create timestamped results directory
    run_dir = None
    if not args.no_metadata:
        try:
            git_short = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
        except Exception:
            git_short = "unknown"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RESULTS_DIR / f"{timestamp}_{git_short}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = capture_metadata(run_dir)
        print(f"📁 Results directory: {run_dir}")
        print(f"   Git: {meta.get('git_short', '?')} ({meta.get('git_branch', '?')})")

    # Execute jobs
    results = []
    n_ok = 0
    n_fail = 0
    total_start = time.time()

    for job in jobs:
        result = run_job(job, args.extra_flags or None)
        results.append(result)
        if result["status"] == "ok":
            n_ok += 1
            print(f"  ✅ {job['id']} ({result.get('elapsed_sec', '?')}s)")
        else:
            n_fail += 1
            print(f"  ❌ {job['id']}: {result.get('error', result['status'])}")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY: {n_ok} passed, {n_fail} failed, {round(total_elapsed, 1)}s total")
    if run_dir:
        # Archive results
        import shutil

        for f in RESULTS_DIR.glob("*.json"):
            if f.parent == RESULTS_DIR:  # top-level only
                shutil.copy2(f, run_dir / f.name)
        print(f"  Archived to: {run_dir}")

        # Save run log
        log_path = run_dir / "run_log.json"
        with open(log_path, "w") as f:
            json.dump(
                {
                    "jobs": results,
                    "total_elapsed_sec": round(total_elapsed, 1),
                    "n_ok": n_ok,
                    "n_fail": n_fail,
                },
                f,
                indent=2,
            )
    print(f"{'═' * 70}")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
