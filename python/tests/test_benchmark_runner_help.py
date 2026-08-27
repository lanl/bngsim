"""Every benchmark runner must answer ``--help`` instead of starting to measure.

A suite runner is a script that takes minutes to hours and writes a *fixed*
results path. ``--help`` is what someone types at one they do not recognize --
which is exactly the moment they cannot know that the file underneath is one
they should not clobber. A runner that parses no ``argv`` answers that question
by running: no usage text, no refusal for a mistyped flag, and a results JSON
left where the finished sweep's goes -- a probe's numbers under the sweep's
name, or a truncated file if the write is interrupted. Either way the report is
indistinguishable from a complete one by existence, which is what a resume check
tends to ask.

That was GH #488, found against ``suites/psa/run_bngsim_timing.py`` (~20 minutes,
writing ``results/psa_bngsim_timing.json``). ``suites/nf/run.py`` -- which
``run_all.py`` itself invokes -- had the identical defect, so this was a
one-of-N-sites drift rather than a single script's oversight: every *other*
runner in the registry already parsed argv, and nothing failed when two did not.

The contract pinned here is deliberately weak and mechanical -- exit 0 with a
usage line for ``--help``, exit 2 for an unrecognized flag -- because that is
what an ``ArgumentParser`` gives for free and what "does not measure" reduces to
from the outside. The family is derived from ``run_all.py``'s REGISTRY rather
than listed here, so a suite added to the orchestrator is covered without
touching this file.

Out of scope, and covered by GH #489: the ``suites/`` scripts the orchestrator
never invokes. Some are spawned workers or wrappers already reading ``sys.argv``
positionally (``ssa_table5/_ssa_cell.py``, the ``jacobian`` probes, the SBML
test-suite wrapper), where a parser would change a contract rather than add a
guard; the rest are hand-run helpers. Whether each is an entry point is a
per-script judgement, not a rule this file can state.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "benchmarks"
SUITES_DIR = BENCH_ROOT / "suites"
RUN_ALL = BENCH_ROOT / "run_all.py"

#: Generous on purpose. A correct ``--help`` returns in well under a second;
#: this bound only has to be shorter than the sweep it must not have started.
HELP_TIMEOUT = 60

#: Runners the orchestrator does not invoke, so REGISTRY cannot name them.
#: ``run_bngsim_timing.py`` is the psa suite's BNGsim-only timing companion --
#: the script GH #488 was filed against.
COMPANIONS = ("psa/run_bngsim_timing.py",)

pytestmark = pytest.mark.skipif(
    not RUN_ALL.exists(),
    reason="benchmarks/ is not in this checkout (installed package)",
)


def _load_run_all() -> types.ModuleType:
    """Import ``benchmarks/run_all.py`` by path -- benchmarks/ is not a package.

    Registered in ``sys.modules`` first: ``SuiteSpec`` is a dataclass, and
    ``@dataclass`` resolves its own annotations through the defining module.
    """
    name = "_bench_run_all"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUN_ALL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _family() -> list[Path]:
    """Every script ``run_all.py`` can invoke, plus the listed companions."""
    run_all = _load_run_all()
    scripts: list[Path] = []
    for suite in run_all.REGISTRY:
        suite_dir = SUITES_DIR / (suite.subdir or suite.name)
        for cmd in (suite.run_cmd, suite.emit_cmd):
            if cmd is not None:
                scripts.append(suite_dir / cmd[0])
    scripts.extend(SUITES_DIR / rel for rel in COMPANIONS)
    return scripts


def _rel(path: Path) -> str:
    return str(path.relative_to(SUITES_DIR))


FAMILY = _family() if RUN_ALL.exists() else []


def _invoke(script: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``script`` from its own directory -- runners resolve siblings by cwd."""
    return subprocess.run(
        [sys.executable, script.name, *args],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=HELP_TIMEOUT,
    )


def test_registry_scripts_all_exist():
    """A registry entry naming a script that is not there would silently pass."""
    missing = [_rel(s) for s in FAMILY if not s.exists()]
    assert not missing, f"registry names scripts that do not exist: {missing}"
    assert len(FAMILY) > 10, "the family collapsed -- REGISTRY did not load"


@pytest.mark.parametrize("script", FAMILY, ids=_rel)
def test_help_answers_instead_of_measuring(script: Path):
    """``--help`` prints usage and exits 0, well inside the timeout."""
    proc = _invoke(script, "--help")
    assert proc.returncode == 0, f"{_rel(script)} --help exited {proc.returncode}\n{proc.stderr}"
    assert "usage:" in proc.stdout, f"{_rel(script)} --help printed no usage line"


@pytest.mark.parametrize("script", FAMILY, ids=_rel)
def test_unrecognized_flag_is_refused(script: Path):
    """A mistyped flag is refused (argparse's exit 2), never silently ignored."""
    proc = _invoke(script, "--not-a-real-flag")
    assert proc.returncode == 2, (
        f"{_rel(script)} accepted --not-a-real-flag (exit {proc.returncode})"
    )


def test_psa_timing_refuses_an_empty_timed_protocol():
    """``--runs 0`` would median an empty list; argparse refuses it up front."""
    script = SUITES_DIR / "psa" / "run_bngsim_timing.py"
    proc = _invoke(script, "--runs", "0")
    assert proc.returncode == 2, f"--runs 0 was accepted (exit {proc.returncode})"
    assert "--runs must be >= 1" in proc.stderr


def test_psa_timing_out_redirects_away_from_the_tracked_path(tmp_path):
    """``--out`` plus a scoped protocol gives a probe that clobbers nothing.

    The second half of GH #488: even once ``--help`` is safe, a diagnostic run
    had no way to avoid ``results/psa_bngsim_timing.json``. The payload records
    the subset it covers, because the default path cannot distinguish one.
    """
    script = SUITES_DIR / "psa" / "run_bngsim_timing.py"
    default_out = script.parent / "results" / "psa_bngsim_timing.json"
    before = default_out.read_bytes() if default_out.exists() else None

    out = tmp_path / "probe.json"
    proc = _invoke(script, "--effort", "low", "--warmup", "0", "--runs", "1", "--out", str(out))
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(out.read_text())
    assert payload["effort"] == "low"
    assert (payload["warmup"], payload["runs"]) == (0, 1)
    assert payload["results"], "a scoped run still has to measure something"

    after = default_out.read_bytes() if default_out.exists() else None
    assert after == before, "--out did not keep the run away from the default path"
