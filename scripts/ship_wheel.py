#!/usr/bin/env python3
"""Build a bngsim wheel and install it into downstream consumer venvs.

This is the supported one-command path for the "rebuild → ship" loop. It

  1. builds a wheel for the *current* interpreter using the canonical command
     (``pip wheel . --no-build-isolation --no-deps``; note that ``python -m
     build`` is unreliable here because the importable ``build`` package in
     the dev venv is not pypa/build), pinning ``MACOSX_DEPLOYMENT_TARGET`` per
     build architecture (10.15 on x86_64, the ``wheelhouse-local`` convention;
     11.0 on arm64, which has no valid 10.x tag), then
  2. force-installs that wheel into each downstream consumer venv, handling
     the pip-vs-uv split automatically (PyBioNetGen's venv has no pip), and
  3. verifies the installed version in each.

Consumers are discovered, not hardcoded (the project forbids baked
``/Users/...`` paths):

  * ``BNGSIM_WHEEL_CONSUMERS`` — if set, an ``os.pathsep``-separated list of
    venv directories (each optionally ``name=path``). This is the explicit
    override.
  * otherwise, auto-discovery scans ``<workspace>/*/.venv`` (workspace = the
    directory two levels above this repo, e.g. ``~/Code``) and treats a venv
    as a consumer when its interpreter already has a *registered* bngsim
    distribution (``importlib.metadata.version("bngsim")`` succeeds). The
    bngsim source venv itself is excluded. A venv where ``import bngsim``
    half-works but no distribution is registered (a stale/namespace shim) is
    skipped, so broken installs are not silently overwritten.

Usage:
    python scripts/ship_wheel.py            # build + install into all consumers
    python scripts/ship_wheel.py --list     # show consumers, don't change anything
    python scripts/ship_wheel.py --build-only
    python scripts/ship_wheel.py --no-build           # reuse newest matching wheel
    python scripts/ship_wheel.py --consumer PyBNF      # limit (repeatable)
    python scripts/ship_wheel.py --wheelhouse DIR
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]


def _py_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, env=env)


def _pyproject_version() -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (SOURCE_DIR / "pyproject.toml").read_text(),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"No version line in {SOURCE_DIR / 'pyproject.toml'}")
    return match.group(1)


def _default_wheelhouse() -> Path:
    env = os.environ.get("BNGSIM_WHEELHOUSE")
    if env:
        return Path(env).expanduser().resolve()
    # Existing convention: wheelhouse-local as a sibling of the source dir.
    return (SOURCE_DIR.parent / "wheelhouse-local").resolve()


# ─── consumer discovery ─────────────────────────────────────────────────


@dataclass
class Consumer:
    name: str
    venv: Path

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"


def _venv_bngsim_version(python: Path) -> str | None:
    """Return the registered bngsim dist version in ``python``'s env, or None.

    None means no bngsim *distribution* is registered there — either it isn't
    installed at all, or only a stale/namespace shim is importable. Either way
    we don't treat it as a ship target.
    """
    if not python.is_file():
        return None
    try:
        proc = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.metadata as m; print(m.version('bngsim'))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _consumers_from_env(spec: str) -> list[Consumer]:
    consumers: list[Consumer] = []
    for entry in spec.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, _, raw = entry.partition("=")
        else:
            raw = entry
            name = ""
        venv = Path(raw).expanduser().resolve()
        consumers.append(Consumer(name=name or venv.parent.name, venv=venv))
    return consumers


def _discover_consumers() -> list[Consumer]:
    env_spec = os.environ.get("BNGSIM_WHEEL_CONSUMERS")
    if env_spec:
        return _consumers_from_env(env_spec)

    workspace = SOURCE_DIR.parent.parent  # e.g. ~/Code
    own_venv = SOURCE_DIR / ".venv"
    found: list[Consumer] = []
    for child in sorted(workspace.iterdir()):
        if not child.is_dir():
            continue
        venv = child / ".venv"
        if venv.resolve() == own_venv.resolve():
            continue
        if _venv_bngsim_version(venv / "bin" / "python") is not None:
            found.append(Consumer(name=child.name, venv=venv.resolve()))
    return found


def _detect_manager(consumer: Consumer) -> str:
    """Return "pip" if the venv has pip, else "uv" if uv is on PATH."""
    proc = subprocess.run(
        [str(consumer.python), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return "pip"
    if shutil.which("uv"):
        return "uv"
    raise RuntimeError(
        f"{consumer.name}: venv has no pip and uv is not on PATH; cannot install the wheel."
    )


# ─── build + install ─────────────────────────────────────────────────────


def _macos_deployment_target() -> str:
    """Lowest macOS a wheel built HERE can claim, per build architecture.

    x86_64 keeps 10.15 (the Intel build box's convention — broad compatibility).
    arm64 must not: Apple Silicon starts at macOS 11.0, so pip/uv never generate
    a ``macosx_10_*_arm64`` compatibility tag and a wheel carrying one is
    installable NOWHERE — including on the machine that built it. Keying off
    ``platform.machine()`` is what makes one script correct on both boxes; it
    also reports ``x86_64`` under Rosetta, which is the right answer there.
    """
    return "10.15" if platform.machine() == "x86_64" else "11.0"


def _build_wheel(wheelhouse: Path, version: str) -> Path:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if sys.platform == "darwin" and not env.get("MACOSX_DEPLOYMENT_TARGET"):
        env["MACOSX_DEPLOYMENT_TARGET"] = _macos_deployment_target()
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(SOURCE_DIR),
            "--no-build-isolation",
            "--no-deps",
            "-w",
            str(wheelhouse),
        ],
        env=env,
    )
    return _find_wheel(wheelhouse, version)


def _find_wheel(wheelhouse: Path, version: str) -> Path:
    tag = _py_tag()
    matches = [p for p in wheelhouse.glob(f"bngsim-{version}-*.whl") if f"-{tag}-" in p.name]
    if not matches:
        raise FileNotFoundError(
            f"No bngsim-{version} wheel for {tag} in {wheelhouse}. "
            "Build one first (drop --no-build)."
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def _install(consumer: Consumer, wheel: Path) -> None:
    manager = _detect_manager(consumer)
    if manager == "pip":
        _run(
            [
                str(consumer.python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ]
        )
    else:  # uv
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(consumer.venv)
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(consumer.python),
                "--reinstall",
                "--no-deps",
                str(wheel),
            ],
            env=env,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show consumers and exit")
    parser.add_argument("--build-only", action="store_true", help="build the wheel, don't install")
    parser.add_argument(
        "--no-build", action="store_true", help="reuse the newest matching wheel in the wheelhouse"
    )
    parser.add_argument(
        "--consumer",
        action="append",
        default=[],
        metavar="NAME",
        help="limit to this consumer (repeatable; matches the discovered name)",
    )
    parser.add_argument(
        "--wheelhouse", type=Path, default=None, help="output/lookup dir for wheels"
    )
    args = parser.parse_args()

    version = _pyproject_version()
    wheelhouse = (
        args.wheelhouse.expanduser().resolve() if args.wheelhouse else _default_wheelhouse()
    )
    consumers = _discover_consumers()
    if args.consumer:
        wanted = set(args.consumer)
        consumers = [c for c in consumers if c.name in wanted]

    if args.list:
        print(f"bngsim {version} (interpreter tag {_py_tag()})")
        print(f"wheelhouse: {wheelhouse}")
        if not consumers:
            print("consumers: (none discovered)")
        for c in consumers:
            cur = _venv_bngsim_version(c.python)
            mgr = _detect_manager(c) if c.python.is_file() else "?"
            print(f"  - {c.name}: {c.venv}  [{mgr}]  currently bngsim=={cur}")
        return 0

    if args.no_build:
        wheel = _find_wheel(wheelhouse, version)
        print(f"reusing wheel: {wheel}", flush=True)
    else:
        wheel = _build_wheel(wheelhouse, version)
        print(f"built wheel: {wheel}", flush=True)

    if args.build_only:
        return 0

    if not consumers:
        print("No consumer venvs discovered; nothing to install.", flush=True)
        print("Set BNGSIM_WHEEL_CONSUMERS or run with --list to debug.", flush=True)
        return 0

    failures: list[str] = []
    for c in consumers:
        print(f"\n=== {c.name} ({c.venv}) ===", flush=True)
        try:
            _install(c, wheel)
            installed = _venv_bngsim_version(c.python)
            status = "OK" if installed == version else f"MISMATCH (got {installed})"
            print(f"{c.name}: bngsim=={installed} [{status}]", flush=True)
            if installed != version:
                failures.append(c.name)
        except (subprocess.CalledProcessError, RuntimeError) as err:
            print(f"{c.name}: FAILED — {err}", flush=True)
            failures.append(c.name)

    if failures:
        print(f"\nShipped {version} with failures: {', '.join(failures)}", flush=True)
        return 1
    print(f"\nShipped {version} to: {', '.join(c.name for c in consumers)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
