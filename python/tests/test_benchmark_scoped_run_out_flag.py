"""A scoped benchmark run must be able to avoid the committed report.

Issue #493, the other half of #488/#489. Those settled *whether* a script parses
``argv``. These seven parse it perfectly well -- and then write a **tracked**
artifact at a path fixed in the source, with no way to aim a run anywhere else.
Each of them also accepts a flag that narrows what it measures (``--engines``,
``--models``, ``--limit``, ``--effort``, ``--quick``, ``--stride``,
``--max-models``, or a different ``T``/``N``). Put together, a legitimate scoped
run silently replaces the full sweep's numbers with its own, under the same name.

That is not hypothetical either. Two committed cross-engine reports were
replaced by ``--engines bngsim`` runs in a single morning::

    "engines": [
   -  "run_network", "bngsim", "roadrunner", "amici_klu", "amici_dense", "copasi"
   +  "bngsim"
    ],

Both write through ``tmp.write_text(...)`` then ``tmp.replace(OUT)``, so nothing
is ever truncated -- the failure mode is a clean, complete overwrite, which is
the one that reads as a finished run. You find out at ``git status`` time, or
from a figure regenerated off a one-engine table.

What is checked here, and what is not. Actually *running* any of these needs
BNG2.pl, run_network, RoadRunner, AMICI or COPASI, so a test cannot drive one to
completion and diff the artifact. What it can do is pin the surface: the flag
exists, its default is the committed path (so the tracked file stays what an
unflagged run produces), and the artifact really is tracked (so the table below
cannot quietly become a list of files nobody commits). ``resolve_out`` is unit
tested directly, because ``run_forced.py`` is the one script whose default
*derives* from another flag and so cannot be an argparse default -- the case
where "``--out`` reaches every use" is easiest to get wrong.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = REPO_ROOT / "benchmarks" / "suites"

#: script -> (tracked artifact it writes, substring its ``--help`` must name).
#: The artifact is what makes each row falsifiable: a path that stops being
#: tracked fails the first test rather than silently weakening the rest.
SCOPED_WRITERS = {
    "ode_engines_s3/run_s3_timing.py": (
        "ode_engines_s3/report_ode_engines_s3.json",
        "report_ode_engines_s3.json",
    ),
    "ode_engines_s4_sbml/run_s4_timing.py": (
        "ode_engines_s4_sbml/report_ode_engines_s4.json",
        "report_ode_engines_s4.json",
    ),
    "ode_fullnet/run_timing.py": (
        "ode_fullnet/report_ode_timing_fullnet.json",
        "report_ode_timing_fullnet.json",
    ),
    "ode_fullnet/run_forced.py": (
        "ode_fullnet/report_ode_timing_forced_auto.json",
        "report_ode_timing_forced_<mode>.json",
    ),
    "jacobian/run.py": (
        "jacobian/results/jacobian_results.json",
        "results/jacobian_results.json",
    ),
    "jacobian/robustness_sweep.py": (
        "jacobian/results/robustness_sweep.json",
        "results/robustness_sweep.json",
    ),
    "jacobian/diagnose_divergence.py": (
        "jacobian/results/diag_MODEL9089538076.json",
        "results/diag_<MODEL_ID>.json",
    ),
}

HELP_TIMEOUT = 60

pytestmark = pytest.mark.skipif(
    not SUITES_DIR.is_dir(),
    reason="benchmarks/ is not in this checkout (installed package)",
)


def _help(rel: str) -> str:
    script = SUITES_DIR / rel
    proc = subprocess.run(
        [sys.executable, script.name, "--help"],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=HELP_TIMEOUT,
    )
    assert proc.returncode == 0, f"{rel} --help exited {proc.returncode}\n{proc.stderr}"
    return proc.stdout


@pytest.mark.parametrize("rel", sorted(SCOPED_WRITERS), ids=lambda r: r)
def test_the_artifact_this_row_protects_is_tracked(rel: str):
    """A row naming an untracked file protects nothing -- fail rather than pass."""
    artifact = SUITES_DIR / SCOPED_WRITERS[rel][0]
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(artifact)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 and not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert proc.returncode == 0, f"{SCOPED_WRITERS[rel][0]} is not tracked; this row is stale"


@pytest.mark.parametrize("rel", sorted(SCOPED_WRITERS), ids=lambda r: r)
def test_a_scoped_run_can_be_aimed_somewhere_else(rel: str):
    """``--out`` exists, and its help names the committed path as the default.

    Both halves matter. Without the flag there is no probe that does not
    clobber; with the flag defaulted somewhere *other* than the committed path,
    an ordinary unflagged run stops producing the file the repository holds.
    """
    text = _help(rel)
    assert "--out" in text, f"{rel} has no --out: a scoped run cannot avoid its committed report"
    expected = SCOPED_WRITERS[rel][1]
    flat = " ".join(text.split())
    assert expected in flat, f"{rel} --help does not name {expected} as the --out default"


@pytest.mark.parametrize(
    "rel",
    ["ode_engines_s3/run_s3_timing.py", "ode_engines_s4_sbml/run_s4_timing.py"],
    ids=lambda r: r,
)
def test_a_cross_engine_report_says_what_it_holds(rel: str):
    """``engines_present`` is derived from the rows, not from ``--engines``.

    The subtler half of the same defect. Without ``--redo`` a scoped
    ``--engines bngsim`` run re-times nothing -- every model is already in the
    report and is skipped -- and yet the pass still rewrote ``_meta.engines`` to
    ``["bngsim"]`` while every row kept all six engines' numbers. The file then
    described the *request* rather than its own contents, which is worse than a
    narrower file: it reads as a one-engine report that happens to carry six.

    Structural, because producing a real one needs run_network, RoadRunner,
    AMICI and COPASI. What is checkable is that the new field is computed from
    ``ordered`` and that the old one is left alone, since anything already
    reading ``engines`` must keep the meaning it had.
    """
    src = (SUITES_DIR / rel).read_text()
    assert '"engines_present": engines_present,' in src, f"{rel} does not record engines_present"
    assert 'present = {e for r in ordered for e in (r.get("engines") or {})}' in src, (
        f"{rel} derives engines_present from something other than the rows it wrote"
    )
    assert '"engines": want_engines,' in src, (
        f"{rel} changed the meaning of _meta.engines; it must stay the request"
    )


def _load(rel: str, name: str):
    """Import a benchmark script by path, leaving ``sys.path`` as we found it."""
    path = SUITES_DIR / rel
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = saved


@pytest.fixture(scope="module")
def run_forced():
    mod = _load("ode_fullnet/run_forced.py", "run_forced_under_test")
    yield mod
    sys.modules.pop("run_forced_under_test", None)


@pytest.mark.parametrize("mode", ["auto", "dense", "sparse"])
def test_the_derived_default_is_still_per_mode(run_forced, mode: str):
    """No ``--out`` means the report this mode has always written."""
    args = argparse.Namespace(mode=mode, out=None)
    assert run_forced.resolve_out(args) == run_forced.out_path(mode)
    assert run_forced.resolve_out(args).name == f"report_ode_timing_forced_{mode}.json"


def test_the_override_beats_the_derived_default(run_forced, tmp_path):
    """``--out`` wins for every mode, which is what makes a probe possible.

    ``run_forced.py`` resolves this in two places -- the resume read at the head
    of the pass and the write at the end -- and a fix that reached only the
    second would read the committed report and write the scratch one, quietly
    seeding a probe with the full sweep's rows.
    """
    scratch = tmp_path / "probe.json"
    for mode in ("auto", "dense", "sparse"):
        args = argparse.Namespace(mode=mode, out=scratch)
        assert run_forced.resolve_out(args) == scratch


def test_run_forced_resolves_the_same_path_for_resume_and_write(run_forced):
    """One resolver, so the two sites cannot disagree.

    Structural, deliberately: proving it by running the pass needs run_network
    and the .net cache. What is checkable is that neither site re-derives the
    path on its own -- ``out_path(args.mode)`` appears only inside
    ``resolve_out``.
    """
    src = (SUITES_DIR / "ode_fullnet/run_forced.py").read_text()
    assert src.count("out_path(args.mode)") == 1, (
        "run_forced.py derives its output path outside resolve_out(); "
        "--out would reach some uses and not others"
    )
    call_sites = [
        ln.strip()
        for ln in src.splitlines()
        if "resolve_out(" in ln and not ln.lstrip().startswith("def ")
    ]
    assert len(call_sites) == 2, (
        f"expected resolve_out() at the resume read and the write, got {call_sites}"
    )
