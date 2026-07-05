#!/usr/bin/env python3
"""``run_all.py`` -- one-command orchestrator for the bngsim benchmark suites.

For each enabled suite under ``suites/<name>/``, this script:

1. Runs the suite's precheck (a callable returning ``None`` or a
   ``"reason: missing X"`` string).  Non-``None`` -> suite ``skipped``.
2. Runs ``<python> suites/<name>/<run_cmd>``, redirecting stdout/stderr
   to per-suite logs.
3. If step 2 succeeded AND the suite has an ``emit_cmd``, runs
   ``<python> suites/<name>/<emit_cmd>`` to refresh
   ``benchmarks/reports/generated/<name>.md``.
4. Records timing, exit code, status (``ok`` / ``failed`` / ``skipped``)
   in ``results/run_all_<UTC>/summary.{json,md}``.

Designed for a *walk-away* workflow: by default it continues past a
failed suite (use ``--halt-on-failure`` to opt out), skips suites whose
precheck reports missing engines / corpora, and produces a final
``<ran>/<failed>/<skipped>`` header so an unattended run is auditable
in one glance.

Cross-suite ordering is the registry order with ``depends_on`` honored
(topological sort, ties broken by registry order).  Default sweep:

    ode, ssa, psa, nf, showcase, antimony, python_ssa, python_ode,
    steady_state, sbml_test_suite, forward_sens, biomodels, fitting,
    sbml_roundtrip

Sharding workflow (run on multiple machines, merge by committing the
shared benchmark result/report artifacts):

    # Machine A
    python run_all.py --only ode,ssa,psa,nf,fitting

    # Machine B
    python run_all.py --only biomodels,sbml_test_suite,forward_sens

Quick report rebuild from already-collected results (no benchmark run):

    python run_all.py --emit-only

Set ``BNGSIM_BENCH_LATEX_DIR`` when a private paper repository needs raw
LaTeX fragments mirrored alongside the public Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Semaphore

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parents[1]
SUITES_DIR = BENCH_ROOT / "suites"
RESULTS_ROOT = BENCH_ROOT / "results"

#: Default interpreter; ``--python`` and ``${BENCH_PYTHON}`` override.
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv-biomodels" / "bin" / "python"

# ---------------------------------------------------------------------------
# Precheck helpers
# ---------------------------------------------------------------------------


def _check_bng2pl() -> str | None:
    """``BNG2.pl`` resolvable on PATH or via ``$BNG2_PL`` / ``$BNGPATH``?"""
    if os.environ.get("BNG2_PL"):
        if Path(os.environ["BNG2_PL"]).exists():
            return None
        return f"BNG2_PL set but {os.environ['BNG2_PL']} does not exist"
    bngpath = os.environ.get("BNGPATH")
    if bngpath and (Path(bngpath) / "BNG2.pl").exists():
        return None
    if shutil.which("BNG2.pl"):
        return None
    # Final fallback: the canonical local install (matches the playbook).
    canonical = Path.home() / "Simulations" / "BioNetGen-2.9.3" / "BNG2.pl"
    if canonical.exists():
        return None
    return "BNG2.pl not found (BNG2_PL/BNGPATH unset, not on PATH, no canonical install)"


def _check_nfsim() -> str | None:
    if rc := _check_bng2pl():
        return rc
    # NFsim shipped with BNG2.pl distribution; same root is sufficient.
    return None


def _check_amici() -> str | None:
    try:
        import amici  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"amici not importable: {exc!r}"
    return None


def _check_roadrunner() -> str | None:
    try:
        import roadrunner  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"libroadrunner not importable: {exc!r}"
    return None


def _check_copasi() -> str | None:
    try:
        import COPASI  # noqa: F401  (python-copasi bindings)
    except Exception as exc:  # noqa: BLE001
        return f"python-copasi not importable: {exc!r}"
    return None


def _check_antimony() -> str | None:
    try:
        import antimony  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"antimony not importable: {exc!r}"
    return None


def _check_libsbml() -> str | None:
    try:
        import libsbml  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"libsbml not importable: {exc!r}"
    return None


def _check_gillespy2() -> str | None:
    try:
        import gillespy2  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"gillespy2 not importable: {exc!r}"
    return None


def _check_pybnf() -> str | None:
    if shutil.which(os.environ.get("PYBNF_CMD", "pybnf")):
        return None
    return "pybnf CLI not on PATH and PYBNF_CMD not set (or invalid)"


def _check_sbml_test_suite_dir() -> str | None:
    p = os.environ.get("SBML_TEST_SUITE_DIR")
    if not p:
        canonical = Path.home() / "Code" / "sbml-test-suite"
        if canonical.exists():
            return None
        return (
            "SBML_TEST_SUITE_DIR unset and ~/Code/sbml-test-suite not found "
            "-- clone https://github.com/sbmlteam/sbml-test-suite and set the env var"
        )
    if not Path(p).expanduser().exists():
        return f"SBML_TEST_SUITE_DIR={p} does not exist"
    return None


def _check_biomodels_corpus() -> str | None:
    if rc := _check_libsbml():
        return rc
    sbml_dir = SUITES_DIR / "biomodels" / "data" / "sbml_downloads"
    if not sbml_dir.exists():
        return f"biomodels SBML corpus not at {sbml_dir} -- run suites/biomodels/fetch.py first"
    manifest = SUITES_DIR / "biomodels" / "manifest.csv"
    if not manifest.exists():
        return (
            f"biomodels manifest.csv not found at {manifest} -- "
            "run suites/biomodels/filter.py first"
        )
    return None


def _check_chain(*checks: Callable[[], str | None]) -> Callable[[], str | None]:
    """Return a callable that runs ``checks`` in order, returning the first non-None."""

    def chained() -> str | None:
        for c in checks:
            if reason := c():
                return reason
        return None

    return chained


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteSpec:
    """One row of the suite registry."""

    name: str
    #: argv tail after the interpreter; ``None`` for emit-only suites.
    run_cmd: tuple[str, ...] | None
    #: argv tail for the emitter; ``None`` for run-only suites or suites
    #: where the runner writes a generated report itself.
    emit_cmd: tuple[str, ...] | None
    precheck: Callable[[], str | None] = field(default=lambda: None)
    #: whether to forward ``--effort X`` to ``run_cmd``.
    accepts_effort: bool = True
    #: other suites that must run first (topological sort).
    depends_on: tuple[str, ...] = ()
    #: whether this suite's ``run.py`` shells out to BNG2.pl (gates
    #: parallel mode with the BNG2.pl semaphore).
    uses_bng2pl: bool = False
    #: directory holding ``run.py`` / ``emit.py`` (relative to SUITES_DIR).
    #: Defaults to ``<name>``; overridden only when a suite is split.
    subdir: str | None = None


REGISTRY: tuple[SuiteSpec, ...] = (
    SuiteSpec(
        name="ode",
        run_cmd=("run.py", "--mode", "both"),
        emit_cmd=("emit.py",),
        precheck=_check_bng2pl,
        uses_bng2pl=True,
    ),
    SuiteSpec(
        name="ssa",
        run_cmd=("run.py", "--mode", "both"),
        emit_cmd=("emit.py",),
        precheck=_check_bng2pl,
        uses_bng2pl=True,
    ),
    SuiteSpec(
        name="psa",
        run_cmd=("run.py", "--mode", "both"),
        emit_cmd=("emit.py",),
        precheck=_check_bng2pl,
        uses_bng2pl=True,
    ),
    SuiteSpec(
        name="nf",
        run_cmd=("run.py",),
        emit_cmd=("emit.py",),
        precheck=_check_nfsim,
        accepts_effort=False,
        uses_bng2pl=True,
    ),
    SuiteSpec(
        name="showcase",
        # The showcase suite contains several runner scripts; run_all.py
        # invokes the one that produces generated/ant_exprtk.md + the
        # antimony source JSON.  The other showcase
        # script (run_ode_trf_fit_from_net.py) stays user-driven.
        run_cmd=("run_ant_exprtk_3engine.py",),
        emit_cmd=None,  # script writes generated/ant_exprtk.md directly
        precheck=_check_chain(_check_antimony, _check_roadrunner),
        accepts_effort=False,
    ),
    SuiteSpec(
        name="antimony",
        run_cmd=("run.py",),
        emit_cmd=("emit.py",),
        precheck=_check_chain(_check_antimony, _check_roadrunner),
        accepts_effort=False,
        depends_on=("showcase",),  # antimony/emit.py reads showcase results
    ),
    SuiteSpec(
        name="python_ssa",
        run_cmd=("run.py", "--mode", "both"),
        emit_cmd=None,  # python_ode owns the joint S6 emitter
        precheck=_check_gillespy2,
    ),
    SuiteSpec(
        name="python_ode",
        run_cmd=("run.py", "--mode", "both"),
        emit_cmd=("emit.py",),
        # diffrax is a soft requirement (engine skipped if absent), so
        # no hard precheck.
        precheck=lambda: None,
        depends_on=("python_ssa",),  # joint S6 emit reads both result JSONs
    ),
    SuiteSpec(
        name="steady_state",
        run_cmd=("run.py", "--mode", "both"),
        emit_cmd=("emit.py",),
        precheck=_check_bng2pl,
        uses_bng2pl=True,
    ),
    SuiteSpec(
        name="sbml_test_suite",
        run_cmd=("run.py", "--mode", "both", "--engines", "all"),
        emit_cmd=("emit.py",),
        precheck=_check_chain(
            _check_sbml_test_suite_dir,
            _check_libsbml,
            _check_roadrunner,
            _check_amici,
            _check_copasi,
        ),
    ),
    SuiteSpec(
        name="forward_sens",
        run_cmd=("run.py", "--mode", "both"),
        emit_cmd=("emit.py",),
        precheck=_check_amici,
    ),
    SuiteSpec(
        name="biomodels",
        run_cmd=("run.py",),  # default --engines = all three
        emit_cmd=("emit.py",),
        precheck=_check_biomodels_corpus,
    ),
    SuiteSpec(
        name="fitting",
        run_cmd=("run.py", "--mode", "both"),  # thin driver
        emit_cmd=("emit.py",),
        precheck=_check_chain(_check_pybnf, _check_bng2pl),
        uses_bng2pl=True,
    ),
    SuiteSpec(
        name="sbml_roundtrip",
        run_cmd=("run.py",),
        emit_cmd=None,  # no report artifact; tracked gate
        precheck=_check_bng2pl,
        accepts_effort=False,
        uses_bng2pl=True,
    ),
)

REGISTRY_BY_NAME: dict[str, SuiteSpec] = {s.name: s for s in REGISTRY}


# ---------------------------------------------------------------------------
# Topological sort with registry-order tiebreak
# ---------------------------------------------------------------------------


def topo_order(selected_names: list[str]) -> list[str]:
    """Return ``selected_names`` sorted so deps come before dependents.

    Ties broken by registry order.  Raises if the selection includes a
    cycle (impossible with the current registry) or breaks a dep edge
    (the caller wants to know about that).
    """
    selected_set = set(selected_names)
    # Carry over the registry-order index as the tiebreak key.
    reg_index = {s.name: i for i, s in enumerate(REGISTRY)}
    ordered: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise RuntimeError(f"cycle detected at {name!r}")
        visiting.add(name)
        spec = REGISTRY_BY_NAME[name]
        for dep in spec.depends_on:
            if dep in selected_set:
                visit(dep)
        visiting.discard(name)
        visited.add(name)
        ordered.append(name)

    for name in sorted(selected_names, key=lambda n: reg_index[n]):
        visit(name)
    return ordered


def warn_missing_deps(selected: list[str]) -> list[str]:
    """Return human-readable warnings for selected suites whose deps are unselected."""
    warnings = []
    sel = set(selected)
    for name in selected:
        spec = REGISTRY_BY_NAME[name]
        for dep in spec.depends_on:
            if dep not in sel:
                warnings.append(
                    f"selection {name!r} depends on {dep!r} which is not in the run; "
                    f"emit.py may see stale {dep!r} data"
                )
    return warnings


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class SuiteResult:
    name: str
    status: str  # "ok" | "failed" | "skipped"
    reason: str = ""  # skip reason or short error summary
    run_rc: int | None = None
    emit_rc: int | None = None
    run_wall_s: float | None = None
    emit_wall_s: float | None = None
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "run_rc": self.run_rc,
            "emit_rc": self.emit_rc,
            "run_wall_s": self.run_wall_s,
            "emit_wall_s": self.emit_wall_s,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _suite_dir(spec: SuiteSpec) -> Path:
    return SUITES_DIR / (spec.subdir or spec.name)


def build_run_argv(spec: SuiteSpec, python: str, effort: str | None) -> list[str]:
    """Compose the argv for ``spec.run_cmd``."""
    if spec.run_cmd is None:
        return []
    tail = list(spec.run_cmd)
    if effort and spec.accepts_effort:
        tail.extend(["--effort", effort])
    return [python, str(_suite_dir(spec) / tail[0]), *tail[1:]]


def build_emit_argv(spec: SuiteSpec, python: str) -> list[str]:
    if spec.emit_cmd is None:
        return []
    tail = list(spec.emit_cmd)
    return [python, str(_suite_dir(spec) / tail[0]), *tail[1:]]


def _ts_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_suite(
    spec: SuiteSpec,
    *,
    python: str,
    effort: str | None,
    out_dir: Path,
    bng2pl_lock: Semaphore | None,
    skip_run: bool,
    skip_emit: bool,
    dry_run: bool,
) -> SuiteResult:
    """Run one suite.  Pure I/O; no return-code coupling to the caller."""
    suite_out = out_dir / spec.name
    if not dry_run:
        suite_out.mkdir(parents=True, exist_ok=True)
    res = SuiteResult(name=spec.name, status="ok", started_at=_ts_utc())

    # ---- precheck -----------------------------------------------------------
    if not skip_run:  # skip precheck when --emit-only (saves spurious skips)
        try:
            reason = spec.precheck()
        except Exception as exc:  # noqa: BLE001
            reason = f"precheck raised {exc!r}"
        if reason:
            res.status = "skipped"
            res.reason = reason
            res.finished_at = _ts_utc()
            return res

    # ---- run.py -------------------------------------------------------------
    if not skip_run and spec.run_cmd is not None:
        argv = build_run_argv(spec, python, effort)
        if dry_run:
            print(f"[run_all] DRY {spec.name}: {' '.join(argv)}")
        else:
            print(f"[run_all] RUN {spec.name}: {' '.join(argv)}", flush=True)
            t0 = time.monotonic()
            with (
                open(suite_out / "run.stdout.log", "wb") as so,
                open(suite_out / "run.stderr.log", "wb") as se,
            ):
                lock_ctx = bng2pl_lock if (bng2pl_lock and spec.uses_bng2pl) else None
                if lock_ctx is not None:
                    lock_ctx.acquire()
                try:
                    proc = subprocess.run(
                        argv,
                        stdout=so,
                        stderr=se,
                        check=False,
                        cwd=_suite_dir(spec),
                    )
                finally:
                    if lock_ctx is not None:
                        lock_ctx.release()
            res.run_wall_s = time.monotonic() - t0
            res.run_rc = proc.returncode
            if proc.returncode != 0:
                res.status = "failed"
                res.reason = f"run.py exit {proc.returncode}; see {suite_out}/run.stderr.log"
                res.finished_at = _ts_utc()
                return res

    # ---- emit.py ------------------------------------------------------------
    if not skip_emit and spec.emit_cmd is not None:
        argv = build_emit_argv(spec, python)
        if dry_run:
            print(f"[run_all] DRY {spec.name} emit: {' '.join(argv)}")
        else:
            print(f"[run_all] EMIT {spec.name}: {' '.join(argv)}", flush=True)
            t0 = time.monotonic()
            with (
                open(suite_out / "emit.stdout.log", "wb") as so,
                open(suite_out / "emit.stderr.log", "wb") as se,
            ):
                proc = subprocess.run(
                    argv,
                    stdout=so,
                    stderr=se,
                    check=False,
                    cwd=_suite_dir(spec),
                )
            res.emit_wall_s = time.monotonic() - t0
            res.emit_rc = proc.returncode
            if proc.returncode != 0:
                res.status = "failed"
                res.reason = f"emit.py exit {proc.returncode}; see {suite_out}/emit.stderr.log"

    res.finished_at = _ts_utc()
    return res


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def _count(results: list[SuiteResult]) -> tuple[int, int, int]:
    return (
        sum(1 for r in results if r.status == "ok"),
        sum(1 for r in results if r.status == "failed"),
        sum(1 for r in results if r.status == "skipped"),
    )


def render_summary_md(
    results: list[SuiteResult], *, started: str, finished: str, dry_run: bool
) -> str:
    n_ok, n_failed, n_skipped = _count(results)
    verb = "planned" if dry_run else "ran"
    lines = [
        f"# run_all summary -- {started} to {finished}",
        "",
        f"**{n_ok} {verb} / {n_failed} failed / {n_skipped} skipped**",
        "" if not dry_run else "(dry-run: no commands were executed)",
        "" if dry_run else "",
        "| Suite | Status | run (s) | emit (s) | Reason |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        run_s = f"{r.run_wall_s:.1f}" if r.run_wall_s is not None else "-"
        emit_s = f"{r.emit_wall_s:.1f}" if r.emit_wall_s is not None else "-"
        reason = r.reason.replace("|", "\\|") if r.reason else ""
        lines.append(f"| {r.name} | {r.status} | {run_s} | {emit_s} | {reason} |")
    return "\n".join(lines) + "\n"


def render_summary_console(results: list[SuiteResult], *, dry_run: bool) -> str:
    n_ok, n_failed, n_skipped = _count(results)
    verb = "planned" if dry_run else "ran"
    out = [f"\n{n_ok} {verb} / {n_failed} failed / {n_skipped} skipped"]
    if dry_run:
        out.append("(dry-run: no commands were executed)")
    out.append("")
    for r in results:
        run_s = f"{r.run_wall_s:6.1f}s" if r.run_wall_s is not None else "    -  "
        emit_s = f"{r.emit_wall_s:5.1f}s" if r.emit_wall_s is not None else "   -  "
        out.append(f"  {r.name:<18} {r.status:<8} run={run_s} emit={emit_s}  {r.reason}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _resolve_python(arg: str | None) -> str:
    if arg:
        return str(Path(arg).expanduser())
    env = os.environ.get("BENCH_PYTHON")
    if env:
        return str(Path(env).expanduser())
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    # Fall back to the currently-running interpreter as a last resort.
    return sys.executable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--effort",
        choices=("low", "medium", "high"),
        default="high",
        help="Cumulative effort tier forwarded to suites that accept it (default: high).",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated suite subset to run (default: every registered suite).",
    )
    ap.add_argument(
        "--skip",
        default="",
        help="Comma-separated suite subset to drop.",
    )
    ap.add_argument(
        "--emit-only",
        action="store_true",
        help="Skip run.py for every suite; just refresh generated/*.md reports.",
    )
    ap.add_argument(
        "--run-only",
        action="store_true",
        help="Skip emit.py for every suite; produce results JSON but no report fragments.",
    )
    ap.add_argument(
        "--python",
        default=None,
        help="Interpreter (default: $BENCH_PYTHON or .venv-biomodels/bin/python).",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Opt-in concurrency (default 1 -- serial). Suites that shell out "
            "to BNG2.pl are gated by a single semaphore even under "
            "--parallel >1."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and exit; do not run anything.",
    )
    ap.add_argument(
        "--halt-on-failure",
        action="store_true",
        help="Opt-in: halt on the first non-zero exit (default = continue).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print the suite registry and exit.",
    )
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'name':<18} {'effort':<6} {'BNG2.pl':<7} {'deps':<28} run_cmd / emit_cmd")
        for s in REGISTRY:
            run_str = "(none)" if s.run_cmd is None else " ".join(s.run_cmd)
            emit_str = "(none)" if s.emit_cmd is None else " ".join(s.emit_cmd)
            deps = ",".join(s.depends_on) or "-"
            print(
                f"  {s.name:<16} {('yes' if s.accepts_effort else 'no'):<6} "
                f"{('yes' if s.uses_bng2pl else 'no'):<7} "
                f"{deps:<28} {run_str}  |  {emit_str}"
            )
        return 0

    if args.emit_only and args.run_only:
        ap.error("--emit-only and --run-only are mutually exclusive")

    # Resolve the suite selection.
    only = _split_csv(args.only)
    skip = _split_csv(args.skip)
    unknown = [n for n in only + skip if n not in REGISTRY_BY_NAME]
    if unknown:
        ap.error(f"unknown suite(s): {', '.join(unknown)}")
    selected = only if only else [s.name for s in REGISTRY]
    selected = [n for n in selected if n not in skip]
    if not selected:
        ap.error("empty selection after --only/--skip filters")

    # Warn about skipped deps (does not abort -- the user might rely on
    # stale results intentionally).
    for w in warn_missing_deps(selected):
        print(f"[run_all] WARN: {w}", file=sys.stderr)

    order = topo_order(selected)
    python = _resolve_python(args.python)
    bng2pl_lock = Semaphore(1) if args.parallel > 1 else None

    started_at = _ts_utc()
    out_dir = RESULTS_ROOT / f"run_all_{started_at.replace(':', '-')}"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_all] python={python}")
    print(f"[run_all] effort={args.effort}  parallel={args.parallel}  out={out_dir}")
    print(f"[run_all] suites ({len(order)}): {', '.join(order)}\n")

    # --- execution -----------------------------------------------------------
    results: list[SuiteResult] = []
    if args.parallel <= 1:
        for name in order:
            res = run_suite(
                REGISTRY_BY_NAME[name],
                python=python,
                effort=args.effort,
                out_dir=out_dir,
                bng2pl_lock=None,
                skip_run=args.emit_only,
                skip_emit=args.run_only,
                dry_run=args.dry_run,
            )
            results.append(res)
            if args.halt_on_failure and res.status == "failed":
                print(f"[run_all] HALT: {name} failed ({res.reason})", file=sys.stderr)
                break
    else:
        # Parallel mode: respect deps via a wait-on-prerequisites pattern.
        # Each suite waits for its `depends_on` predecessors before launching.
        done_evts: dict[str, Event] = {}
        from threading import Event

        for name in order:
            done_evts[name] = Event()

        lock = Lock()  # protects ``results`` appends

        def _worker(spec: SuiteSpec) -> None:
            for dep in spec.depends_on:
                if dep in done_evts:
                    done_evts[dep].wait()
            r = run_suite(
                spec,
                python=python,
                effort=args.effort,
                out_dir=out_dir,
                bng2pl_lock=bng2pl_lock,
                skip_run=args.emit_only,
                skip_emit=args.run_only,
                dry_run=args.dry_run,
            )
            with lock:
                results.append(r)
            done_evts[spec.name].set()

        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [ex.submit(_worker, REGISTRY_BY_NAME[n]) for n in order]
            for f in futures:
                f.result()
        # Preserve registry order in summary
        results.sort(key=lambda r: order.index(r.name))

    finished_at = _ts_utc()

    # --- summary -------------------------------------------------------------
    summary_console = render_summary_console(results, dry_run=args.dry_run)
    print(summary_console)
    if not args.dry_run:
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "python": python,
                    "effort": args.effort,
                    "parallel": args.parallel,
                    "selection": order,
                    "results": [r.to_dict() for r in results],
                },
                indent=2,
            )
        )
        (out_dir / "summary.md").write_text(
            render_summary_md(results, started=started_at, finished=finished_at, dry_run=False)
        )

    # Exit non-zero if anything failed.  Skipped suites do not affect rc.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
