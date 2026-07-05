"""Local CI orchestrator for bngsim wheel build + smoke validation.

Cross-platform (macOS x86_64 / macOS arm64 / Linux x86_64 / Windows AMD64).
`uv` is required; we deliberately do not fall back to pip + venv so that every
reporter's run is byte-comparable.

Subcommands:
    wheel        Build one wheel for a target Python (default: current uv
                 interpreter) and smoke-test it.
    matrix       Build wheels for Python 3.10, 3.11, 3.12, 3.13 in sequence
                 and smoke-test each. Skips Python versions that uv cannot
                 provision on this platform.
    smoke        Smoke-test an already-built wheel against a target Python.

Output:
    Each run writes a Markdown report to
        scripts/local_ci_report-<platform>-<arch>-<pyver>.md
    summarizing pass/fail per check. The matrix subcommand additionally
    writes
        scripts/local_ci_report-<platform>-<arch>-matrix.md
    aggregating the per-Python results into one table.

Usage (after `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`):
    uv run python scripts/local_ci.py wheel
    uv run python scripts/local_ci.py wheel --python 3.13
    uv run python scripts/local_ci.py matrix
    uv run python scripts/local_ci.py smoke --wheel path/to/wheel.whl --python 3.12

Note: paths resolve from this file, so it runs from anywhere; the ``uv run``
invocations above assume the bngsim repository root as the working directory.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

BNGSIM_DIR = Path(__file__).resolve().parents[1]  # repo root (the bngsim/ tree)
REPO_ROOT = BNGSIM_DIR.parent  # workspace holding the repo (+ wheelhouse-local, build venvs)
DATA_DIR = BNGSIM_DIR / "tests" / "data"
ANTIMONY_FIXTURE_DIR = BNGSIM_DIR / "benchmarks" / "models" / "antimony" / "ssys"
SCRIPTS_DIR = BNGSIM_DIR / "scripts"
SMOKE_SCRIPT = SCRIPTS_DIR / "local_ci_smoke.py"
WHEELHOUSE = REPO_ROOT / "wheelhouse-local"

PYTHONS = ("3.10", "3.11", "3.12", "3.13")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return f"[{status}] {self.name} {self.detail}".rstrip()


@dataclass
class WheelReport:
    pyver: str
    wheel: Path | None = None
    build_ok: bool = False
    build_detail: str = ""
    smoke_checks: list[CheckResult] = field(default_factory=list)

    @property
    def smoke_ok(self) -> bool:
        return bool(self.smoke_checks) and all(c.ok for c in self.smoke_checks)


# ---------------------------------------------------------------------------
# uv plumbing
# ---------------------------------------------------------------------------


def ensure_uv() -> str:
    """Locate uv on PATH or abort with install hint. Returns absolute path."""
    uv = shutil.which("uv")
    if uv is not None:
        return uv
    sys.stderr.write(
        "local_ci.py requires `uv` to be installed and on PATH.\n"
        "Install (macOS/Linux):  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "Install (Windows PS):   irm https://astral.sh/uv/install.ps1 | iex\n"
        "Then re-run this script.\n"
    )
    sys.exit(2)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Thin subprocess wrapper that echoes the command before running."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def uv_python_install(uv: str, pyver: str) -> bool:
    """Make sure uv has cpython-<pyver> available locally. Returns True on success."""
    try:
        run([uv, "python", "install", pyver])
    except subprocess.CalledProcessError as e:
        print(f"  uv python install {pyver} failed: {e}")
        return False
    return True


# ---------------------------------------------------------------------------
# Build + smoke
# ---------------------------------------------------------------------------

# The per-Python build/test venvs are throwaway: each run rmtree's and recreates
# them. Remove them afterward too so they don't pile up as stale scratch (this
# was ~1.3 GB of leftover .venv-build-cp*/.venv-test-cp* dirs). Pass --keep-venvs
# to any subcommand to retain them for debugging a failed build or smoke run.
KEEP_VENVS = False


def _teardown_venv(venv: Path) -> None:
    if KEEP_VENVS:
        print(f"  keeping venv {venv.name} (--keep-venvs)")
        return
    shutil.rmtree(venv, ignore_errors=True)


def build_wheel(uv: str, pyver: str) -> Path | None:
    """Build a bngsim wheel for the given target Python. Returns the wheel path."""
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    venv = REPO_ROOT / f".venv-build-cp{pyver.replace('.', '')}"
    if venv.exists():
        shutil.rmtree(venv)
    run([uv, "venv", str(venv), "--python", pyver])
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    try:
        # uv venv ships pip-compatible package management; uv pip install handles deps.
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "--upgrade",
                "pip",
                "build",
                "cmake",
                "ninja",
            ]
        )

        env = {}
        if platform.system() == "Darwin":
            env["CMAKE_ARGS"] = "-DBNGSIM_ENABLE_KLU=OFF"
            # 11.0 for arm64, 10.15 for Intel.
            env["MACOSX_DEPLOYMENT_TARGET"] = "11.0" if platform.machine() == "arm64" else "10.15"

        print(f"=== building wheel for python {pyver} ===")
        try:
            run(
                [str(py), "-m", "build", "--wheel", "--outdir", str(WHEELHOUSE), str(BNGSIM_DIR)],
                env=env,
            )
        except subprocess.CalledProcessError as e:
            print(f"  build failed: {e}")
            return None

        # Find the wheel just produced for this pyver.
        tag = f"cp{pyver.replace('.', '')}"
        matches = sorted(
            WHEELHOUSE.glob(f"bngsim-*-{tag}-{tag}-*.whl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return matches[0] if matches else None
    finally:
        _teardown_venv(venv)


def smoke_wheel(uv: str, wheel: Path, pyver: str) -> list[CheckResult]:
    """Install the wheel into a fresh venv and run local_ci_smoke.py."""
    venv = REPO_ROOT / f".venv-test-cp{pyver.replace('.', '')}"
    if venv.exists():
        shutil.rmtree(venv)
    run([uv, "venv", str(venv), "--python", pyver])
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    # The [antimony] extra is opt-in: try to install, accept failure (e.g. on
    # Pythons where no antimony wheel exists for this platform). On failure
    # the smoke script reports Antimony as "extra not available" rather than
    # blocking the whole run.
    try:
        spec = f"{wheel}[pandas]"
        run([uv, "pip", "install", "--python", str(py), spec])
        try:
            run([uv, "pip", "install", "--python", str(py), "antimony>=3.1.1"])
        except subprocess.CalledProcessError:
            print("  antimony not available on this platform/python; smoke will skip it")

        report_dir = SCRIPTS_DIR / "_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report = report_dir / f"smoke-cp{pyver.replace('.', '')}.json"

        try:
            run(
                [
                    str(py),
                    str(SMOKE_SCRIPT),
                    "--data-dir",
                    str(DATA_DIR),
                    "--antimony-fixture-dir",
                    str(ANTIMONY_FIXTURE_DIR),
                    "--report",
                    str(report),
                ]
            )
        except subprocess.CalledProcessError as e:
            print(f"  smoke run exited non-zero: {e}")
            # fall through; report file may still exist with partial info

        if not report.exists():
            return [CheckResult("smoke", False, "no report produced")]

        import json

        data = json.loads(report.read_text())
        return [
            CheckResult(name=k, ok=v["ok"], detail=v.get("detail", "")) for k, v in data.items()
        ]
    finally:
        _teardown_venv(venv)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _platform_tag() -> str:
    sys_ = platform.system().lower()
    arch = platform.machine().lower()
    return f"{sys_}-{arch}"


def write_single_report(report: WheelReport) -> Path:
    plat = _platform_tag()
    out = SCRIPTS_DIR / f"local_ci_report-{plat}-cp{report.pyver.replace('.', '')}.md"
    lines = [
        f"# local_ci report: {plat} / Python {report.pyver}",
        "",
        f"- bngsim source: `{BNGSIM_DIR}`",
        f"- platform: `{platform.platform()}`",
        f"- machine: `{platform.machine()}`",
        f"- python (target): `{report.pyver}`",
        f"- wheel: `{report.wheel.name if report.wheel else '(none)'}`",
        f"- build: **{'PASS' if report.build_ok else 'FAIL'}**{(' — ' + report.build_detail) if report.build_detail else ''}",
        "",
        "## smoke checks",
        "",
    ]
    if report.smoke_checks:
        lines.append("| check | result | detail |")
        lines.append("|---|---|---|")
        for c in report.smoke_checks:
            lines.append(f"| {c.name} | {'PASS' if c.ok else 'FAIL'} | {c.detail} |")
    else:
        lines.append("(no smoke checks ran — build failed)")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return out


def write_matrix_report(reports: list[WheelReport]) -> Path:
    plat = _platform_tag()
    out = SCRIPTS_DIR / f"local_ci_report-{plat}-matrix.md"
    lines = [
        f"# local_ci matrix report: {plat}",
        "",
        f"- bngsim source: `{BNGSIM_DIR}`",
        f"- platform: `{platform.platform()}`",
        f"- machine: `{platform.machine()}`",
        "",
        "| Python | wheel | build | smoke |",
        "|---|---|---|---|",
    ]
    for r in reports:
        wheel = r.wheel.name if r.wheel else "(none)"
        build = "PASS" if r.build_ok else "FAIL"
        if not r.build_ok:
            smoke = "(skipped)"
        else:
            failing = [c.name for c in r.smoke_checks if not c.ok]
            smoke = "PASS" if not failing else "FAIL: " + ", ".join(failing)
        lines.append(f"| {r.pyver} | `{wheel}` | {build} | {smoke} |")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def cmd_wheel(args: argparse.Namespace) -> int:
    uv = ensure_uv()
    pyver = args.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    if not uv_python_install(uv, pyver):
        return 1
    report = WheelReport(pyver=pyver)
    report.wheel = build_wheel(uv, pyver)
    report.build_ok = report.wheel is not None
    if report.build_ok:
        report.smoke_checks = smoke_wheel(uv, report.wheel, pyver)
    write_single_report(report)
    return 0 if report.build_ok and report.smoke_ok else 1


def cmd_matrix(args: argparse.Namespace) -> int:
    uv = ensure_uv()
    reports: list[WheelReport] = []
    for pyver in PYTHONS:
        print(f"\n############ Python {pyver} ############")
        if not uv_python_install(uv, pyver):
            reports.append(
                WheelReport(
                    pyver=pyver, build_ok=False, build_detail=f"uv could not provision {pyver}"
                )
            )
            continue
        report = WheelReport(pyver=pyver)
        report.wheel = build_wheel(uv, pyver)
        report.build_ok = report.wheel is not None
        if report.build_ok:
            report.smoke_checks = smoke_wheel(uv, report.wheel, pyver)
        reports.append(report)
        write_single_report(report)
    write_matrix_report(reports)
    all_pass = all(r.build_ok and r.smoke_ok for r in reports)
    return 0 if all_pass else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    uv = ensure_uv()
    wheel = Path(args.wheel).resolve()
    if not wheel.is_file():
        sys.stderr.write(f"wheel not found: {wheel}\n")
        return 2
    pyver = args.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    if not uv_python_install(uv, pyver):
        return 1
    report = WheelReport(pyver=pyver, wheel=wheel, build_ok=True, build_detail="(supplied)")
    report.smoke_checks = smoke_wheel(uv, wheel, pyver)
    write_single_report(report)
    return 0 if report.smoke_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_wheel = sub.add_parser("wheel", help="build + smoke one wheel")
    p_wheel.add_argument("--python", help="target Python, e.g. 3.12")
    p_wheel.set_defaults(func=cmd_wheel)
    p_matrix = sub.add_parser("matrix", help="build + smoke cp310-cp313")
    p_matrix.set_defaults(func=cmd_matrix)
    p_smoke = sub.add_parser("smoke", help="smoke-test an existing wheel")
    p_smoke.add_argument("--wheel", required=True)
    p_smoke.add_argument("--python", help="target Python, e.g. 3.12")
    p_smoke.set_defaults(func=cmd_smoke)
    for p in (p_wheel, p_matrix, p_smoke):
        p.add_argument(
            "--keep-venvs",
            action="store_true",
            help="retain the throwaway build/test venvs (default: delete them after use)",
        )
    args = ap.parse_args()
    global KEEP_VENVS
    KEEP_VENVS = getattr(args, "keep_venvs", False)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
