#!/usr/bin/env python3
"""Turnkey: stand up (or verify) a parity/benchmark env that drives bngsim through
PyBioNetGen — identically on every machine, whether it has run bngsim before or not.

What it guarantees:
  * PyBioNetGen pinned to the exact RuleWorld commit carrying the merged BNGsim
    bridge (``../requirements-pybionetgen.txt`` → RuleWorld/PyBioNetGen@5109a46) —
    no local checkout, no PR-branch, no PYTHONPATH dance.
  * bngsim installed from THIS repo's wheel (``scripts/ship_wheel.py`` builds it;
    bngsim is not on PyPI). Pass ``--bngsim-wheel`` for an explicit wheel, or
    ``--build-bngsim`` to build one for the current interpreter.
  * the BNGsim backend is then PROVEN live in the new env (``bngsim_backend``):
    bngsim importable + version-compatible + a trivial model actually simulates
    via bngsim. A machine that can't run bngsim fails HERE, loudly, not silently
    mid-sweep on the legacy stack.

Usage:
    python bootstrap_parity_env.py --venv .venv-parity --build-bngsim
    python bootstrap_parity_env.py --venv .venv-parity --bngsim-wheel dist/bngsim-*.whl
    python bootstrap_parity_env.py --check-only            # verify the ACTIVE interpreter
    python bootstrap_parity_env.py --check-only --python /path/to/venv/bin/python3

Requires ``uv`` on PATH. PyBioNetGen's setup.py shells out to ``pip install numpy``
and downloads BNG2.pl at build time, so the build needs pip+setuptools+numpy in the
target env and ``--no-build-isolation`` — both handled here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARITY_ROOT = HERE.parent  # .../parity_checks
REPO_BNGSIM = PARITY_ROOT.parent  # .../bngsim
REQ_PYBIONETGEN = PARITY_ROOT / "requirements-pybionetgen.txt"
SHIP_WHEEL = REPO_BNGSIM / "scripts" / "ship_wheel.py"


def _run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _uv():
    from shutil import which

    if not which("uv"):
        sys.exit("ABORT: `uv` is required (the repo's package manager) but is not on PATH.")
    return "uv"


def _build_bngsim_wheel() -> Path:
    """Build a bngsim wheel for the current interpreter via the repo's protocol."""
    if not SHIP_WHEEL.exists():
        sys.exit(f"ABORT: --build-bngsim but {SHIP_WHEEL} is missing.")
    wheelhouse = Path(tempfile.mkdtemp(prefix="bngsim_wheel_"))
    print(f"building bngsim wheel via ship_wheel.py into {wheelhouse} ...")
    _run([sys.executable, str(SHIP_WHEEL), "--build-only", "--wheelhouse", str(wheelhouse)])
    wheels = sorted(wheelhouse.glob("bngsim-*.whl"))
    if not wheels:
        sys.exit(f"ABORT: ship_wheel.py built no wheel in {wheelhouse}.")
    return wheels[-1]


def _resolve_bngsim_wheel(args) -> Path:
    if args.bngsim_wheel:
        p = Path(args.bngsim_wheel).expanduser().resolve()
        if not p.exists():
            sys.exit(f"ABORT: --bngsim-wheel does not exist: {p}")
        return p
    if args.build_bngsim:
        return _build_bngsim_wheel()
    # Fall back to the newest wheel in the conventional wheelhouse-local.
    wheelhouse = (REPO_BNGSIM.parent / "wheelhouse-local").resolve()
    wheels = sorted(wheelhouse.glob("bngsim-*.whl")) if wheelhouse.exists() else []
    if wheels:
        print(f"using newest existing wheel: {wheels[-1]}")
        return wheels[-1]
    sys.exit(
        "ABORT: no bngsim wheel. Pass --bngsim-wheel <path>, or --build-bngsim to build "
        f"one (bngsim is not on PyPI), or drop a wheel in {wheelhouse}."
    )


def _verify(python_exe: str) -> int:
    """Run the backend self-check in the TARGET interpreter and report."""
    print("\nverifying BNGsim backend in the target env ...")
    # Run bngsim_backend's __main__ self-check (backend_status) in the target env.
    r = subprocess.run([python_exe, str(HERE / "bngsim_backend.py")])
    if r.returncode != 0:
        print(
            "\nFAIL: the target env reports the BNGsim backend is NOT live. A parity/"
            "benchmark sweep there would error or silently use the legacy stack.",
            file=sys.stderr,
        )
        return r.returncode
    print("\nOK: BNGsim backend is live in the target env. Use this interpreter for sweeps:")
    print(f"   {python_exe}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--venv", default="", help="venv dir to create/populate (e.g. .venv-parity)")
    ap.add_argument(
        "--python",
        default="",
        help="for --check-only: the interpreter to verify (default: this one)",
    )
    ap.add_argument("--bngsim-wheel", default="", help="explicit bngsim wheel to install")
    ap.add_argument(
        "--build-bngsim",
        action="store_true",
        help="build a bngsim wheel for the current interpreter via scripts/ship_wheel.py",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="don't install anything; just verify the BNGsim backend in --python (or the active env)",
    )
    args = ap.parse_args()

    if args.check_only:
        return _verify(args.python or sys.executable)

    if not args.venv:
        ap.error("--venv is required unless --check-only")
    uv = _uv()
    venv = Path(args.venv).expanduser().resolve()
    python_exe = str(venv / ("Scripts" if os.name == "nt" else "bin") / "python3")

    print(f"=== bootstrapping parity env at {venv} ===")
    _run([uv, "venv", str(venv)])
    # Build prerequisites PyBioNetGen's setup.py needs (it shells `pip install numpy`).
    _run([uv, "pip", "install", "--python", python_exe, "pip", "setuptools", "wheel", "numpy"])
    # Pinned PyBioNetGen from RuleWorld (no build isolation: setup.py runs in this env).
    if not REQ_PYBIONETGEN.exists():
        sys.exit(f"ABORT: pin file missing: {REQ_PYBIONETGEN}")
    print(f"installing pinned PyBioNetGen ({REQ_PYBIONETGEN.name}) ...")
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python_exe,
            "--no-build-isolation",
            "-r",
            str(REQ_PYBIONETGEN),
        ]
    )
    # bngsim from this repo's wheel.
    wheel = _resolve_bngsim_wheel(args)
    print(f"installing bngsim wheel: {wheel}")
    _run([uv, "pip", "install", "--python", python_exe, str(wheel)])

    return _verify(python_exe)


if __name__ == "__main__":
    raise SystemExit(main())
