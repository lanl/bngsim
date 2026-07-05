#!/usr/bin/env python3
"""Set up a reproducible dependency environment for showcase benchmarks.

Examples:
  uv run --active python bngsim/benchmarks/suites/showcase/setup_showcase_env.py
  uv run --active python bngsim/benchmarks/suites/showcase/setup_showcase_env.py --mode latest
  uv run --active python bngsim/benchmarks/suites/showcase/setup_showcase_env.py --keep-dask
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PINNED_FILE = Path(__file__).resolve().parent / "requirements_showcase_pinned.txt"
ANTIMONY_WHEEL_RUN_URL = "https://github.com/sys-bio/antimony/actions/runs/22922160697/"
LATEST_PACKAGES = [
    "libroadrunner",
    "amici",
    "antimony",
    "matplotlib",
    "dask",
    "distributed",
]


def run_cmd(cmd: list[str], *, check: bool = True) -> int:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def module_version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        return f"ERROR ({exc})"
    return str(getattr(mod, "__version__", "unknown"))


def verify_imports(expect_dask: bool) -> int:
    print("\n[verify] import checks")
    checks = [
        ("bngsim", "bngsim"),
        ("roadrunner", "libroadrunner"),
        ("amici", "amici"),
        ("antimony", "antimony"),
        ("matplotlib", "matplotlib"),
    ]

    failures = 0
    for module_name, label in checks:
        version = module_version(module_name)
        ok = not version.startswith("ERROR")
        status = "OK" if ok else "FAIL"
        print(f"  {label:14s} {status:4s} {version}")
        if not ok:
            failures += 1
        if module_name == "antimony" and ok and version == "3.1.1":
            print(
                "  antimony note    WARN 3.1.1 loaded; for long SBML conversion"
                f" loops install 3.1.2 wheel from {ANTIMONY_WHEEL_RUN_URL}"
            )

    has_dask = importlib.util.find_spec("dask") is not None
    has_distributed = importlib.util.find_spec("distributed") is not None
    if expect_dask:
        if has_dask:
            print(f"  dask            OK   {module_version('dask')}")
        else:
            print("  dask            FAIL missing")
            failures += 1
        if has_distributed:
            print(f"  distributed     OK   {module_version('distributed')}")
        else:
            print("  distributed     FAIL missing")
            failures += 1
    else:
        if has_dask or has_distributed:
            print("  dask/distributed WARN installed (remove mode requested)")
        else:
            print("  dask/distributed OK   not installed")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install/refresh dependencies for showcase benchmarks."
    )
    parser.add_argument(
        "--mode",
        choices=["pinned", "latest"],
        default="pinned",
        help="Dependency mode: pinned (paper-reproducible) or latest (rolling).",
    )
    parser.add_argument(
        "--remove-dask",
        action="store_true",
        help="Uninstall dask/distributed after setup (only if you want AMICI-only workflows).",
    )
    parser.add_argument(
        "--skip-bngsim-reinstall",
        action="store_true",
        help="Skip `uv pip install -e ./bngsim`.",
    )
    args = parser.parse_args()

    print(f"[setup] repo={REPO_ROOT}")
    print(f"[setup] mode={args.mode}")

    if not args.skip_bngsim_reinstall:
        run_cmd(["uv", "pip", "install", "--no-build-isolation", "-e", "./bngsim"])

    if args.mode == "pinned":
        run_cmd(["uv", "pip", "install", "--upgrade", "-r", str(PINNED_FILE)])
    else:
        run_cmd(["uv", "pip", "install", "--upgrade", *LATEST_PACKAGES])

    if args.remove_dask:
        run_cmd(["uv", "pip", "uninstall", "dask", "distributed"], check=False)

    failures = verify_imports(expect_dask=not args.remove_dask)
    if failures:
        print(f"\n[done] setup finished with {failures} import failure(s)")
        return 1

    print("\n[done] showcase environment is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
