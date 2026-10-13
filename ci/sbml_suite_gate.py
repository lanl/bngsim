#!/usr/bin/env python3
"""Per-pull-request gate on the SBML semantic test suite (GH #700).

Runs the pinned SBML Test Suite's semantic cases through
``benchmarks/suites/sbml_test_suite/run.py`` (1,823 cases, 1,789 of them scored;
about 5 s on an M-series Mac) and
holds bngsim's outcome on every scored case to the committed baseline,
``parity_checks/nightly/baselines/sbml_semantic.json``. That is the file the
nightly (GH #702) compares against too, so there is one record of where bngsim
stands on the suite, kept current by the pull requests that change it.

Any change of a case's outcome fails, in either direction:

* fix it: a case that was good is not, a refused case now gets a wrong answer,
  or the harness could not score a case (``harness_error``);
* record it: an improvement, a change between two failing outcomes, or a case
  that appeared or left. Re-run with ``--update-baseline`` and commit the file in
  the same pull request, so the change shows up in review one line per case.

The expected values are the suite's own, never bngsim's, so the baseline only
records how far bngsim is from an independent answer. Unlike a golden-trajectory
baseline, it cannot end up defending a wrong answer bngsim once produced.

    uv run --no-sync python ci/sbml_suite_gate.py                    # check
    uv run --no-sync python ci/sbml_suite_gate.py --update-baseline  # record

The corpus is not vendored. Fetch the pinned checkout once with
``harness/sbml_test_suite/fetch_semantic_suite.py``. ``--suite`` (or
``$SBML_TEST_SUITE_DIR``) takes either the checkout or its ``cases/semantic``.

Exit status: 0 the run matches the baseline (or the baseline was recorded),
1 it does not, 2 the gate could not run (no corpus, the wrong corpus, a crash).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "benchmarks" / "suites" / "sbml_test_suite" / "run.py"
VERDICTS = REPO / "parity_checks" / "nightly" / "verdicts.py"
BASELINE = REPO / "parity_checks" / "nightly" / "baselines" / "sbml_semantic.json"
PIN = REPO / "harness" / "sbml_test_suite" / "SUITE_PIN.json"
SUITE = "sbml_semantic"
UPDATE = "uv run --no-sync python ci/sbml_suite_gate.py --update-baseline"


# The variables that tell git which repository it is in (not, say, which config).
_GIT_REPO_VARS = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
    }
)


class GateError(Exception):
    """The gate could not run, as distinct from a run that does not match."""


def _verdicts():
    spec = importlib.util.spec_from_file_location("_sbml_gate_verdicts", VERDICTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_suite(arg: str | None) -> Path:
    """The ``cases/semantic`` directory to run, from ``--suite``, then
    ``$SBML_TEST_SUITE_DIR``, then ``~/Code/sbml-test-suite``. A checkout root is
    accepted too, since ``run.py`` given one finds no cases and reports nothing."""
    raw = arg or os.environ.get("SBML_TEST_SUITE_DIR") or "~/Code/sbml-test-suite"
    path = Path(raw).expanduser()
    if (path / "cases" / "semantic").is_dir():
        path = path / "cases" / "semantic"
    if not path.is_dir() or not any(d.is_dir() and d.name.isdigit() for d in path.iterdir()):
        raise GateError(
            f"no SBML Test Suite semantic cases at {path}.\n"
            "  Fetch the pinned checkout:\n"
            "    uv run --no-sync python harness/sbml_test_suite/fetch_semantic_suite.py "
            "--sparse --dest ~/Code/sbml-test-suite\n"
            "  then pass --suite <checkout> or set SBML_TEST_SUITE_DIR."
        )
    return path


def check_pin(semantic: Path, expected: str) -> str:
    """Refuse a corpus that is not a git checkout at *expected*: a baseline is
    only meaningful against the corpus it was recorded on. Returns the commit."""
    root = semantic.parent.parent
    # An exported GIT_DIR makes git ignore -C and answer for that repository.
    env = {k: v for k, v in os.environ.items() if k not in _GIT_REPO_VARS}
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
        capture_output=True,
        text=True,
        env=env,
    )
    top, _, head = proc.stdout.strip().partition("\n")
    # A copy inside some other repository would report that repository's HEAD.
    if proc.returncode != 0 or not head or Path(top).resolve() != root.resolve():
        raise GateError(
            f"{root} is not a git checkout, so its version cannot be checked against "
            f"the pin ({expected[:12]}). Fetch it with fetch_semantic_suite.py, or pass "
            "--allow-unpinned for a copy you know matches."
        )
    if head != expected:
        raise GateError(
            f"{root} is at {head[:12]}, but the pin is {expected[:12]} "
            "(harness/sbml_test_suite/SUITE_PIN.json). Check out the pin, or fetch it "
            "afresh with fetch_semantic_suite.py."
        )
    return head


def run_suite(semantic: Path, results_dir: Path) -> dict:
    """Run every semantic case through the suite runner; returns its report."""
    env = {**os.environ, "SBML_TEST_SUITE_DIR": str(semantic)}
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mode",
        "correctness",
        "--results-dir",
        str(results_dir),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    report = results_dir / "sbml_test_suite_results.json"
    if proc.returncode != 0 or not report.is_file():
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        raise GateError(f"the suite runner failed (exit {proc.returncode}):\n{tail}")
    payload = json.loads(report.read_text())
    # stderr is mostly bngsim's own warnings; a harness_error's traceback is
    # what someone reading the log needs.
    for row in payload.get("cases", []):
        if row.get("status") == "harness_error":
            print(f"{row['case']}: harness_error\n{row.get('traceback', '')}", file=sys.stderr)
    return payload


def bngsim_commit() -> str | None:
    """The commit of the checkout that ran, for a recorded baseline's header."""
    env = {k: v for k, v in os.environ.items() if k not in _GIT_REPO_VARS}
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, env=env
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def unrecordable(fresh: dict) -> str:
    """Why *fresh* must not become the baseline, or ``""``. A recorded
    harness_error matches itself on every later run, so the gate would stop
    seeing it; a run with no cases records nothing at all."""
    if not fresh["cases"]:
        return "Not recording: no cases ran."
    n = fresh["tally"].get("harness_error", 0)
    if n:
        return (
            f"Not recording: the harness could not score {n} case(s). Fix the harness "
            "first; a baseline that holds a harness_error stops the gate seeing it."
        )
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--suite", help="the SBML Test Suite checkout, or its cases/semantic")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="write this run's verdicts over the baseline instead of checking against it",
    )
    ap.add_argument("--fresh-out", type=Path, help="also write this run's verdicts here")
    ap.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="run a corpus that is not a git checkout at the pinned commit",
    )
    args = ap.parse_args(argv)

    V = _verdicts()
    try:
        semantic = resolve_suite(args.suite)
        pin = json.loads(PIN.read_text())["commit"]
        commit = None if args.allow_unpinned else check_pin(semantic, pin)
        with tempfile.TemporaryDirectory(prefix="sbml_suite_gate_") as tmp:
            report_path = Path(tmp) / "sbml_test_suite_results.json"
            run_suite(semantic, Path(tmp))
            prov = {
                "recorded_by": "ci/sbml_suite_gate.py",
                "bngsim_commit": bngsim_commit(),
                "suite_commit": commit,
            }
            fresh = V.extract("sbml-suite", [report_path], SUITE, prov)
    except GateError as exc:
        print(f"SBML suite gate could not run: {exc}", file=sys.stderr)
        return 2

    if args.fresh_out:
        args.fresh_out.parent.mkdir(parents=True, exist_ok=True)
        V.dump_verdicts(fresh, args.fresh_out)

    refusal = unrecordable(fresh) if args.update_baseline else ""
    if not args.baseline.exists():
        if refusal:
            print(refusal, file=sys.stderr)
            return 1
        if args.update_baseline:
            V.dump_verdicts(fresh, args.baseline)
            print(f"Recorded {fresh['n_cases']} cases {fresh['tally']} -> {args.baseline}")
            return 0
        print(f"SBML suite gate: no baseline at {args.baseline}. Record one: {UPDATE}")
        return 2

    result = V.gate(V.load_verdicts(args.baseline), fresh)
    md = V.render_gate_md(result)
    print(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(md + "\n")

    if args.update_baseline:
        if refusal:
            print(refusal, file=sys.stderr)
            return 1
        V.dump_verdicts(fresh, args.baseline)
        print(f"Recorded {fresh['n_cases']} cases {fresh['tally']} -> {args.baseline}")
        if result["n_fix"]:
            print(
                f"{result['n_fix']} of the recorded changes are regressions (above). "
                "Say why in the pull request; the baseline diff is the review."
            )
        return 0

    if result["ok"]:
        return 0
    if result["empty"]:
        print("No cases ran, so nothing was checked.", file=sys.stderr)
        return 1
    if result["n_fix"]:
        print(
            "The cases under 'harness could not score', 'regressed' and 'refused ... a "
            "wrong answer' are what to fix. Re-baseline one only if the change is "
            "deliberate, and say so in the pull request."
        )
    if result["n_record"]:
        shown = (
            args.baseline.relative_to(REPO)
            if args.baseline.is_relative_to(REPO)
            else args.baseline
        )
        print(
            "The other changes are not regressions, but the baseline must record them, "
            "so that a later pull request that undoes them fails here. Record them in "
            f"this pull request:\n\n    {UPDATE}\n\nand commit {shown}."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
