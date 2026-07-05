#!/usr/bin/env python3
"""Validate that jobs.yaml, scripts, suites, and paper artifacts are in sync.

Checks:
  1. Every job's script exists on disk
  2. Every job's suite file exists (for file-path suites)
  3. Every paper artifact file path is valid (figures exist, LaTeX files exist)
  4. No orphan scripts (scripts in comparison/ or validation/ not in any job)
  5. Package prerequisites are importable
  6. External tool prerequisites are on PATH or at known locations
  7. Job IDs are unique
  8. depends_on references valid job IDs
  9. Output paths don't collide

Usage:
    python validate_jobs.py          # full validation
    python validate_jobs.py --quick  # skip import checks (fast)

Exit code: 0 if all checks pass, 1 if any fail.
"""

import argparse
import importlib
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BNG2_PL, NFSIM_BIN, RUN_NETWORK  # noqa: E402

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
JOBS_YAML = HARNESS_DIR / "jobs.yaml"
BENCHMARKS_DIR = HARNESS_DIR.parent / "benchmarks"

# Known external tool paths, env-resolved by common.py ($BNG2_PL / $RUN_NETWORK /
# $NFSIM, each defaulting under $BNGPATH → REPO_ROOT/BioNetGen-2.9.3). Keys match
# the tool names used in jobs.yaml `prerequisites.external`.
EXTERNAL_TOOLS = {
    "run_network": RUN_NETWORK,
    "BNG2.pl": BNG2_PL,
    "NFsim": NFSIM_BIN,
}


class Checker:
    """Accumulates pass/warn/fail results."""

    def __init__(self):
        self.passes = []
        self.warnings = []
        self.failures = []

    def ok(self, msg: str):
        self.passes.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  ⚠️  {msg}")

    def fail(self, msg: str):
        self.failures.append(msg)
        print(f"  ❌ {msg}")

    def section(self, title: str):
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

    def summary(self):
        print(f"\n{'═' * 60}")
        print(
            f"  RESULTS: {len(self.passes)} pass, "
            f"{len(self.warnings)} warn, {len(self.failures)} fail"
        )
        if self.failures:
            print("\n  Failures:")
            for f in self.failures:
                print(f"    • {f}")
        if self.warnings:
            print("\n  Warnings:")
            for w in self.warnings:
                print(f"    • {w}")
        print(f"{'═' * 60}")
        return len(self.failures) == 0


def load_manifest() -> dict:
    with open(JOBS_YAML) as f:
        return yaml.safe_load(f)


def check_scripts(jobs: list[dict], c: Checker):
    """Check that every job's script exists."""
    c.section("Script existence")
    for job in jobs:
        script = job.get("script", "")
        path = HARNESS_DIR / script
        if path.exists():
            c.ok(f"{job['id']}: {script}")
        else:
            c.fail(f"{job['id']}: script not found: {path}")


def check_orphan_scripts(jobs: list[dict], c: Checker):
    """Find scripts not referenced by any job."""
    c.section("Orphan script detection")
    referenced = {job["script"] for job in jobs}

    for subdir in ["comparison", "validation"]:
        d = HARNESS_DIR / subdir
        if not d.exists():
            continue
        for py_file in sorted(d.glob("*.py")):
            rel = f"{subdir}/{py_file.name}"
            if rel not in referenced:
                # Skip __init__.py, __pycache__, and utility files
                if py_file.name.startswith("_") or py_file.name == "plot_1013_4panel.py":
                    continue
                c.warn(f"Orphan script not in any job: {rel}")

    # Also check top-level harness scripts
    for py_file in sorted(HARNESS_DIR.glob("*.py")):
        if py_file.name in ("run_jobs.py", "validate_jobs.py", "common.py", "__init__.py"):
            continue
        rel = py_file.name
        if rel not in referenced:
            c.warn(f"Orphan script not in any job: {rel}")


def check_suites(jobs: list[dict], c: Checker):
    """Check that suite file references resolve."""
    c.section("Suite file existence")
    for job in jobs:
        suite = job.get("suite")
        if suite is None or not isinstance(suite, str):
            continue
        # Skip narrative descriptions (not file paths).
        # Heuristic: narrative descriptions contain spaces but don't
        # look like relative paths (no / and no file extension).
        is_narrative = (
            suite.startswith("Pool")
            or suite.startswith("Pools")
            or "/" not in suite
            and " " in suite
        )
        if is_narrative:
            c.ok(f"{job['id']}: suite is narrative description (OK)")
            continue

        # Try as relative to bngsim/
        suite_path = HARNESS_DIR.parent / suite
        if suite_path.exists():
            c.ok(f"{job['id']}: {suite}")
        else:
            # Try as relative to harness/
            suite_path2 = HARNESS_DIR / suite
            if suite_path2.exists():
                c.ok(f"{job['id']}: {suite}")
            else:
                c.fail(f"{job['id']}: suite not found: {suite}")


def check_paper_artifacts(jobs: list[dict], c: Checker):
    """Check that paper artifact file paths exist."""
    c.section("Paper artifact paths")
    for job in jobs:
        artifacts = job.get("paper_artifacts", [])
        for art in artifacts:
            if not isinstance(art, dict):
                continue
            # Check figure files
            fig_file = art.get("file")
            if fig_file:
                path = REPO_ROOT / fig_file
                if path.exists():
                    c.ok(f"{job['id']}: {fig_file}")
                else:
                    c.warn(
                        f"{job['id']}: figure not found: {fig_file} (may need to run job first)"
                    )
            # Check LaTeX file references
            latex_ref = art.get("latex", "")
            if latex_ref:
                # Extract file path before " §" or " line"
                tex_path = latex_ref.split(" §")[0].split(" line")[0].strip()
                path = REPO_ROOT / tex_path
                if path.exists():
                    c.ok(f"{job['id']}: LaTeX ref {tex_path}")
                else:
                    c.fail(f"{job['id']}: LaTeX file not found: {tex_path}")
            # Check generated table files
            files = art.get("files", [])
            for f in files:
                # Generated files may not exist yet — warn, don't fail
                path = HARNESS_DIR / f
                if path.exists():
                    c.ok(f"{job['id']}: {f}")
                else:
                    c.warn(f"{job['id']}: generated file not found: {f} (run job to create)")


def check_unique_ids(jobs: list[dict], c: Checker):
    """Check job IDs are unique."""
    c.section("Job ID uniqueness")
    seen = {}
    for job in jobs:
        jid = job["id"]
        if jid in seen:
            c.fail(f"Duplicate job ID: {jid}")
        else:
            seen[jid] = True
    if len(seen) == len(jobs):
        c.ok(f"All {len(jobs)} job IDs are unique")


def check_dependencies(jobs: list[dict], c: Checker):
    """Check depends_on references valid job IDs."""
    c.section("Dependency references")
    all_ids = {j["id"] for j in jobs}
    for job in jobs:
        deps = job.get("depends_on", [])
        for dep in deps:
            if dep in all_ids:
                c.ok(f"{job['id']} → {dep}")
            else:
                c.fail(f"{job['id']}: depends_on '{dep}' not found")


def check_output_collisions(jobs: list[dict], c: Checker):
    """Check output paths don't collide."""
    c.section("Output path uniqueness")
    outputs = {}
    for job in jobs:
        out = job.get("output")
        if out is None:
            continue
        out_list = [out] if isinstance(out, str) else out
        for o in out_list:
            if o in outputs:
                c.fail(f"Output collision: {o} claimed by '{outputs[o]}' and '{job['id']}'")
            else:
                outputs[o] = job["id"]
    if not c.failures:
        c.ok(f"All {len(outputs)} output paths are unique")


def check_packages(jobs: list[dict], c: Checker, quick: bool):
    """Check Python package prerequisites are importable."""
    c.section("Package prerequisites")
    if quick:
        c.ok("Skipped (--quick mode)")
        return

    all_packages = set()
    for job in jobs:
        pkgs = job.get("prerequisites", {}).get("packages", [])
        all_packages.update(pkgs)

    for pkg in sorted(all_packages):
        try:
            importlib.import_module(pkg)
            c.ok(f"import {pkg}")
        except ImportError:
            c.warn(f"Cannot import {pkg} (needed by some jobs)")


def check_external_tools(jobs: list[dict], c: Checker, quick: bool):
    """Check external tool prerequisites."""
    c.section("External tool prerequisites")
    if quick:
        c.ok("Skipped (--quick mode)")
        return

    all_tools = set()
    for job in jobs:
        tools = job.get("prerequisites", {}).get("external", [])
        all_tools.update(tools)

    for tool in sorted(all_tools):
        known_path = EXTERNAL_TOOLS.get(tool)
        if known_path and Path(known_path).exists():
            c.ok(f"{tool}: {known_path}")
        elif shutil.which(tool):
            c.ok(f"{tool}: found on PATH")
        else:
            c.warn(f"{tool}: not found (needed by some jobs)")


def main():
    parser = argparse.ArgumentParser(
        description="Validate jobs.yaml sync with scripts, suites, and paper"
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Skip import and tool checks (fast mode)",
    )
    args = parser.parse_args()

    print(f"Validating {JOBS_YAML}\n")

    manifest = load_manifest()
    jobs = manifest.get("jobs", [])
    print(f"Found {len(jobs)} jobs ({sum(1 for j in jobs if j.get('enabled', True))} enabled)")

    c = Checker()

    check_unique_ids(jobs, c)
    check_dependencies(jobs, c)
    check_scripts(jobs, c)
    check_orphan_scripts(jobs, c)
    check_suites(jobs, c)
    check_paper_artifacts(jobs, c)
    check_output_collisions(jobs, c)
    check_packages(jobs, c, quick=args.quick)
    check_external_tools(jobs, c, quick=args.quick)

    passed = c.summary()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
