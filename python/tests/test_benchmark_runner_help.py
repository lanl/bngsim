"""Every benchmark script must answer ``--help`` instead of starting to measure.

A suite runner is a script that takes minutes to hours and writes a *fixed*
results path. ``--help`` is what someone types at one they do not recognize --
which is exactly the moment they cannot know that the file underneath is one
they should not clobber. A script that parses no ``argv`` answers that question
by running: no usage text, no refusal for a mistyped flag, and a results file
left where the finished sweep's goes -- a probe's numbers under the sweep's
name, or a truncated file if the write is interrupted. Either way the report is
indistinguishable from a complete one by existence, which is what a resume check
tends to ask.

That was GH #488, found against ``suites/psa/run_bngsim_timing.py`` (~20 minutes,
writing ``results/psa_bngsim_timing.json``). ``suites/nf/run.py`` -- which
``run_all.py`` itself invokes -- had the identical defect, so it was never one
script's oversight: every *other* runner in the orchestrator's registry already
parsed argv, and nothing failed when two did not.

GH #489 then swept the rest of the tree, where the hazard turned out not to be
hypothetical. ``jacobian/probe_attach.py`` read ``--help`` as a model id and
overwrote the *committed* ``results/attach_probe.json`` with the resulting
one-row failure -- 46 lines of measurement replaced by two, by a command typed
to find out what the script does. ``ode_engines_s4_sbml/check_sbml_engine_
agreement.py`` ran all three engines before overwriting its committed report.
``ode_fullnet/recover_s4_points.py`` reached furthest of all: regenerate two
networks through BNG2.pl, rewrite the characterization, then copy both files
into a *different repository's* ``latex/generated/``.

**Discovery is by structure, not by a table.** The family is every ``suites/``
script with a ``__main__`` guard, so a script added to any suite inherits the
rule without an edit here. The contract pinned on it is deliberately weak and
mechanical -- exit 0 with a usage line for ``--help``, exit 2 for an
unrecognized flag -- because that is what an ``ArgumentParser`` gives for free
and what "does not measure" reduces to from the outside.

The one exemption is a script a *caller* launches on a fixed argv:
``ssa_table5/_ssa_cell.py`` (the per-cell subprocess of Table 5's serial
measurement pass) and the SBML test suite's ``bngsim_wrapper.py``. Those are
left parser-free by decision, at the paper side's explicit ask -- an
``ArgumentParser`` there changes a caller's contract to buy a usage line for an
invocation nobody makes. ``SPAWNED_WORKERS`` is that exemption list, and the
tests on it are what keep "no parser" readable as intent rather than as the
defect it otherwise looks like.

``jacobian/diagnose_divergence.py`` is both: a driver a person types, which
re-enters itself as ``--worker`` on an argv it composes. Only the driver half is
parsed, and only the driver half is what this file exercises.
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

#: Scripts a *caller* launches on a fixed argv, mapped to the caller that does
#: it. Exempt from the argv contract by decision (GH #489); the value is the
#: file whose name the worker must still point at, so the exemption is legible
#: from the worker itself and not only from here.
SPAWNED_WORKERS = {
    "ssa_table5/_ssa_cell.py": "run_ssa_timing.py",
    "sbml_test_suite/testrunner/bngsim_wrapper.py": "bngsim_wrapper.sh",
}

#: Script -> the *committed* artifact its bare invocation rewrites. The two
#: places GH #489's hazard was demonstrated rather than argued: probing these
#: destroyed a file in the repository. ``--help`` must leave each byte-identical.
TRACKED_ARTIFACTS = {
    "jacobian/probe_attach.py": "jacobian/results/attach_probe.json",
    "ode_engines_s4_sbml/check_sbml_engine_agreement.py": (
        "ode_engines_s4_sbml/report_ode_engines_s4_agreement.json"
    ),
}

_MAIN_GUARD = 'if __name__ == "__main__"'

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


def _registry_scripts() -> list[str]:
    """Every script ``run_all.py`` can invoke, suite-relative."""
    out = []
    for suite in _load_run_all().REGISTRY:
        sub = suite.subdir or suite.name
        for cmd in (suite.run_cmd, suite.emit_cmd):
            if cmd is not None:
                out.append(f"{sub}/{cmd[0]}")
    return out


def _family() -> list[Path]:
    """Every ``suites/`` script with a ``__main__`` guard, minus the exemptions."""
    return sorted(
        p
        for p in SUITES_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
        and _MAIN_GUARD in p.read_text(encoding="utf-8")
        and str(p.relative_to(SUITES_DIR)) not in SPAWNED_WORKERS
    )


def _rel(path: Path) -> str:
    return str(path.relative_to(SUITES_DIR))


FAMILY = _family() if SUITES_DIR.is_dir() else []


def _invoke(script: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``script`` from its own directory -- runners resolve siblings by cwd."""
    return subprocess.run(
        [sys.executable, script.name, *args],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=HELP_TIMEOUT,
    )


def test_discovery_found_the_tree():
    """A glob that quietly matched nothing would make every test below vacuous."""
    assert len(FAMILY) > 40, f"the family collapsed to {len(FAMILY)} scripts"


def test_every_orchestrated_script_is_in_the_family():
    """The registry cannot name a script the argv contract does not reach."""
    family = {_rel(s) for s in FAMILY}
    named = _registry_scripts()
    missing = [rel for rel in named if not (SUITES_DIR / rel).exists()]
    assert not missing, f"registry names scripts that do not exist: {missing}"
    unreached = [rel for rel in named if rel not in family]
    assert not unreached, f"registry scripts outside the argv contract: {unreached}"


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


@pytest.mark.parametrize("rel", sorted(TRACKED_ARTIFACTS), ids=lambda r: r)
def test_help_leaves_a_committed_artifact_alone(rel: str):
    """Probing a harness must not destroy the report it would have described."""
    artifact = SUITES_DIR / TRACKED_ARTIFACTS[rel]
    if not artifact.exists():
        pytest.skip(f"{TRACKED_ARTIFACTS[rel]} is not in this checkout")
    before = artifact.read_bytes()
    proc = _invoke(SUITES_DIR / rel, "--help")
    assert proc.returncode == 0, proc.stderr
    assert artifact.read_bytes() == before, f"{rel} --help rewrote {TRACKED_ARTIFACTS[rel]}"


@pytest.mark.parametrize("rel", sorted(SPAWNED_WORKERS), ids=lambda r: r)
def test_spawned_workers_keep_their_positional_contract(rel: str):
    """An exempt worker gets no parser, and says whose argv it is reading.

    Both halves matter. The first is the exemption itself -- "add an
    ``ArgumentParser`` everywhere" is the tempting uniform answer to #489 and it
    is the wrong one for a file only a spawner invokes. The second is what keeps
    the exemption from reading as the very defect #489 swept for: the worker has
    to name its caller, so the next reader can see there is a contract here.
    """
    src = (SUITES_DIR / rel).read_text(encoding="utf-8")
    assert "argparse" not in src, (
        f"{rel} grew a parser -- it is spawned by {SPAWNED_WORKERS[rel]} on a fixed argv"
    )
    assert SPAWNED_WORKERS[rel] in src, f"{rel} does not name its caller ({SPAWNED_WORKERS[rel]})"


def test_spawned_worker_exemptions_are_all_real_files():
    """An exemption for a file that moved would silently widen to nothing."""
    gone = [rel for rel in SPAWNED_WORKERS if not (SUITES_DIR / rel).exists()]
    assert not gone, f"SPAWNED_WORKERS names files that are not there: {gone}"


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
