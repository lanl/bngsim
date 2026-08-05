#!/usr/bin/env python3
"""Turnkey: stand up (or verify) a parity/benchmark env that drives bngsim through
PyBioNetGen — identically on every machine, whether it has run bngsim before or not.

What it guarantees:
  * PyBioNetGen pinned to the exact RuleWorld commit carrying the merged BNGsim
    bridge (``../requirements-pybionetgen.txt`` → RuleWorld/PyBioNetGen@43b09a5) —
    no local checkout, no PR-branch, no PYTHONPATH dance.
  * a bngsim chosen EXPLICITLY, never resolved by chance. Three sources, in the
    order you'd reach for them: ``--build-bngsim`` builds this working tree
    (``scripts/ship_wheel.py``) — what you want when validating a change;
    ``--bngsim-wheel`` installs a wheel you already have; ``--bngsim-pypi X.Y.Z``
    installs the published release from PyPI — the faithful route for a *consumer
    reproducing a published golden*, since that golden's engine WAS the PyPI wheel.
    With none of the three, the newest wheel in ``../wheelhouse-local`` is used.
  * the BNGsim backend is then PROVEN live in the new env (``bngsim_backend``):
    bngsim importable + version-compatible + a trivial model actually simulates
    via bngsim. A machine that can't run bngsim fails HERE, loudly, not silently
    mid-sweep on the legacy stack. That check also PRINTS which bngsim it proved —
    ``bngsim_build_commit`` (the commit its compiled extension was built from) and
    ``bngsim_install`` (index / wheel / editable). Since bngsim went to PyPI the
    version string alone no longer identifies an artifact: it bumps only at
    release, so every commit between two releases reports the same one (GH #163).

Usage:
    python bootstrap_parity_env.py --venv .venv-parity --build-bngsim
    python bootstrap_parity_env.py --venv .venv-parity --bngsim-wheel dist/bngsim-*.whl
    python bootstrap_parity_env.py --venv .venv-parity --bngsim-pypi 0.12.2
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


def _pypi_requirement(value: str) -> str:
    """``0.12.2`` -> ``bngsim==0.12.2``; an explicit specifier passes through.

    Reproducing a published golden wants an EXACT version, so a bare version is
    pinned with ``==`` rather than left to resolve to whatever is newest. A value
    that already starts with a comparison operator (``>=0.12,<0.13``) is taken as
    written; anything else is rejected rather than guessed at.
    """
    v = value.strip()
    if v[:1].isdigit():
        return f"bngsim=={v}"
    if v[:1] in "=<>!~":
        return f"bngsim{v}"
    sys.exit(
        "ABORT: --bngsim-pypi wants a version (e.g. 0.12.2) or a specifier "
        f"(e.g. '>=0.12,<0.13'), got {value!r}."
    )


def _resolve_bngsim_source(args) -> str:
    """The bngsim to install, as a single ``uv pip install`` argument.

    Either a local wheel path or a PyPI requirement — both are just an argument,
    so the install step does not branch. Explicit sources win in the order the
    flags are mutually exclusive in; with none of them, fall back to the newest
    wheel in the conventional ``wheelhouse-local``.
    """
    if args.bngsim_wheel:
        p = Path(args.bngsim_wheel).expanduser().resolve()
        if not p.exists():
            sys.exit(f"ABORT: --bngsim-wheel does not exist: {p}")
        return str(p)
    if args.build_bngsim:
        return str(_build_bngsim_wheel())
    if args.bngsim_pypi:
        return _pypi_requirement(args.bngsim_pypi)
    # Fall back to the newest wheel in the conventional wheelhouse-local.
    wheelhouse = (REPO_BNGSIM.parent / "wheelhouse-local").resolve()
    wheels = sorted(wheelhouse.glob("bngsim-*.whl")) if wheelhouse.exists() else []
    if wheels:
        print(f"using newest existing wheel: {wheels[-1]}")
        return str(wheels[-1])
    sys.exit(
        "ABORT: no bngsim to install. Pick a source:\n"
        "  --build-bngsim          build a wheel from THIS working tree (validating a change)\n"
        "  --bngsim-wheel <path>   install a wheel you already have\n"
        "  --bngsim-pypi <version> install the published release (reproducing a golden)\n"
        f"or drop a wheel in {wheelhouse}."
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--venv", default="", help="venv dir to create/populate (e.g. .venv-parity)")
    ap.add_argument(
        "--python",
        default="",
        help="interpreter version for the new venv (`uv venv --python`, e.g. 3.12); "
        "defaults to the version running this script, which is the one --build-bngsim "
        "builds the wheel for. With --check-only: the interpreter to verify instead.",
    )
    # Exactly one bngsim source (or none, for the wheelhouse-local fallback) —
    # naming two would silently mean "whichever this function checks first".
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bngsim-wheel", default="", help="explicit bngsim wheel to install")
    src.add_argument(
        "--build-bngsim",
        action="store_true",
        help="build a bngsim wheel for the current interpreter via scripts/ship_wheel.py",
    )
    src.add_argument(
        "--bngsim-pypi",
        default="",
        metavar="VERSION",
        help="install the published bngsim from PyPI (e.g. 0.12.2) instead of a local "
        "wheel — the faithful route when reproducing a published golden, whose engine "
        "was that release. If PyPI has no wheel for this interpreter, pip falls back to "
        "the sdist and builds from source (slow, and needs a toolchain).",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="don't install anything; just verify the BNGsim backend in --python (or the active env)",
    )
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if args.check_only:
        return _verify(args.python or sys.executable)

    if not args.venv:
        ap.error("--venv is required unless --check-only")
    uv = _uv()
    venv = Path(args.venv).expanduser().resolve()
    python_exe = str(venv / ("Scripts" if os.name == "nt" else "bin") / "python3")

    # The venv's interpreter VERSION is load-bearing: bngsim ships as an ABI-tagged
    # wheel (cp312-...), so a venv on a version the wheel has no tag for fails at the
    # LAST step, after the whole PyBioNetGen build. Default to THIS interpreter's
    # version — that is the one `--build-bngsim` builds the wheel for — instead of
    # whatever `uv venv` would pick on its own. `--python` overrides.
    py_version = args.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    # `--allow-existing` keeps re-running the bootstrap idempotent: current uv aborts
    # on an existing venv rather than reusing it, which would make a re-bootstrap
    # after a pin bump fail on step one.
    print(f"=== bootstrapping parity env at {venv} (python {py_version}) ===")
    _run([uv, "venv", "--allow-existing", "--python", py_version, str(venv)])
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
    # bngsim: a local wheel (built here or handed to us) or the published release.
    source = _resolve_bngsim_source(args)
    print(f"installing bngsim: {source}")
    _run([uv, "pip", "install", "--python", python_exe, source])

    return _verify(python_exe)


if __name__ == "__main__":
    raise SystemExit(main())
