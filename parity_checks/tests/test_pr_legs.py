"""Issue #1387: a pull request runs only the nightly-parity legs its files feed.

Every PR that touched one of the workflow's PR paths ran all of its legs,
about 1.5 hours on ~10 runners. ``parity_checks/nightly/pr_legs.py`` maps the
PR's changed paths to the legs they feed, and the plan job runs those. These
pin the mapping on the issue's cases, measure each runner's imports again so
the table cannot drift from the code, and check that the workflow asks for
what the script answers.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "nightly-parity.yml"

_spec = importlib.util.spec_from_file_location(
    "pr_legs", REPO / "parity_checks/nightly/pr_legs.py"
)
pr_legs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr_legs)

# Where the legs' code lives, for a module only a runtime sys.path finds.
CODE_ROOTS = ("parity_checks", "benchmarks", "harness")

ALL = set(pr_legs.LEGS)
PARITY = ALL - {"sbml"}
DIFF = {"differential", "differential-nf"}

# The script each leg's job runs (the workflow's own command lines).
RUNNERS = {
    "sbml": ["benchmarks/suites/sbml_test_suite/run.py", "harness/run_dsmts.py"],
    "rr-ode": ["parity_checks/rr_parity/rr_run.py"],
    "bng-ode": ["parity_checks/bng_parity/bng_ode_run.py"],
    "rr-ssa": ["parity_checks/rr_parity/ssa_screen.py"],
    "fd-sens": ["parity_checks/fd_sens/fd_sens_run.py"],
    "differential": ["parity_checks/differential/differential_run.py"],
    "differential-nf": ["parity_checks/differential/differential_run.py"],
    "bng-stoch": ["parity_checks/bng_parity/bng_stoch_run.py"],
    "amici-sens": ["parity_checks/amici_parity/amici_sens_run.py"],
}


def _run(paths: list[str], *args: str) -> dict[str, bool]:
    done = subprocess.run(
        [sys.executable, str(REPO / "parity_checks/nightly/pr_legs.py"), *args],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=True,
    )
    return {k: v == "true" for k, v in (line.split("=") for line in done.stdout.split())}


@pytest.mark.parametrize(
    ("paths", "legs", "corpus"),
    [
        (["parity_checks/differential/_differential.py"], DIFF, True),
        (["parity_checks/fd_sens/fd_sens_jobs.json"], {"fd-sens", "differential"}, True),
        (["parity_checks/fd_sens/fd_sens_run.py"], {"fd-sens"}, False),
        (
            ["parity_checks/amici_parity/_amici_sens.py"],
            {"amici-sens", "fd-sens", *DIFF},
            True,
        ),
        (["parity_checks/amici_parity/amici_sens_run.py"], {"amici-sens"}, True),
        (["parity_checks/amici_parity/amici_ode_jobs.json"], {"amici-sens"}, True),
        (
            ["parity_checks/bng_parity/jobs.json"],
            {"bng-ode", "bng-stoch", "differential-nf"},
            False,
        ),
        (["parity_checks/bng_parity/bng_ode_run.py"], {"bng-ode"}, False),
        # differential's NF arm imports bng_stoch_run, after a sys.path insert.
        (["parity_checks/bng_parity/net_gillespie.py"], {"bng-stoch", *DIFF}, True),
        (["parity_checks/_core/work.py"], PARITY, True),
        (["parity_checks/rr_parity/_rr_common.py"], PARITY, True),
        (["parity_checks/rr_parity/ssa_screen.py"], {"rr-ssa"}, True),
        (["parity_checks/rr_parity/rr_run.py"], {"rr-ode"}, True),
        # Data no import shows: differential's SSA arm runs rr_ssa's jobs.
        (["parity_checks/rr_parity/ssa_jobs.json"], {"differential"}, True),
        # The corpus's manifest: every leg that reads the corpus, and more.
        (["parity_checks/rr_parity/ode_jobs.json"], PARITY, True),
        (["benchmarks/suites/biomodels/biomodels_provenance.json"], pr_legs.CORPUS_LEGS, True),
        ([".github/workflows/nightly-parity.yml"], ALL, True),
        (["parity_checks/nightly/verdicts.py"], ALL, True),
        (["harness/run_dsmts.py"], {"sbml"}, False),
        (["parity_checks/nightly/baselines/differential.json"], set(), False),
        # Documentation, and the golden gate's trajectories, feed no leg.
        (["parity_checks/rr_parity/README.md", "parity_checks/nightly/README.md"], set(), False),
        (["parity_checks/bng_parity/golden/golden.json"], set(), False),
        # Beside a baseline, the code change decides.
        (
            ["parity_checks/nightly/baselines/fd_sens.json", "parity_checks/fd_sens/x.py"],
            {"fd-sens", "differential"},
            True,
        ),
        # A PR's other files run no leg here, as a PR of only those runs none.
        (["changelog.d/1387.changed.md", "python/bngsim/_simulator.py"], set(), False),
        (
            ["parity_checks/differential/x.py", "changelog.d/1.fixed.md", "python/tests/t.py"],
            DIFF,
            True,
        ),
    ],
    ids=[
        "differential only",
        "fd_sens's models",
        "fd_sens's runner",
        "amici_parity's scheduler",
        "amici_sens's runner",
        "amici_sens's jobs",
        "bng_parity's jobs",
        "bng_ode's runner",
        "bng_stoch's own module",
        "_core",
        "rr_parity's _rr_common",
        "the SSA screen",
        "rr_ode's runner",
        "rr_ssa's SSA jobs",
        "the corpus manifest",
        "the corpus provenance",
        "the workflow",
        "a nightly script",
        "harness",
        "a baseline only",
        "documentation only",
        "the golden trajectories",
        "a baseline beside code",
        "only files outside the nightly",
        "differential beside a changelog and a test",
    ],
)
def test_a_pull_requests_paths_run_the_legs_they_feed(paths, legs, corpus):
    out = _run(paths)
    assert {leg for leg in pr_legs.LEGS if out[leg]} == legs
    assert out["corpus"] is corpus


def test_a_scheduled_or_dispatched_run_runs_every_leg():
    out = _run([], "--all")
    assert all(out[leg] for leg in pr_legs.LEGS) and out["corpus"]


def _loaded_files(script: str) -> set[str]:
    """The repo files ``script`` loads, loaded as its job runs it (its own
    directory first on sys.path), without running main(): the modules it
    imports, and those an import inside a function would load, which only
    reading the loaded modules' import statements finds. A name the probe's
    sys.path cannot find, where a function puts its directory on sys.path
    before importing it, is looked up among the legs' code by its file name:
    every module of that name counts."""
    probe = textwrap.dedent(f"""
        import ast, importlib.util, json, pathlib, sys
        repo = pathlib.Path({str(REPO)!r})
        script = repo / {script!r}
        index = {{}}

        def in_repo(name):
            if not index:
                for root in {CODE_ROOTS!r}:
                    for f in (repo / root).rglob("*.py"):
                        stem = f.parent.name if f.name == "__init__.py" else f.stem
                        index.setdefault(stem, []).append(str(f))
            return index.get(name, [])

        sys.path.insert(0, str(script.parent))
        spec = importlib.util.spec_from_file_location("runner_under_test", script)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ModuleNotFoundError as e:
            top = (e.name or "").split(".")[0]
            print(json.dumps({{"broken" if in_repo(top) else "missing": e.name}}))
            raise SystemExit(0)

        def ours(f):
            if not f:
                return None
            try:
                rel = pathlib.Path(f).resolve().relative_to(repo)
            except ValueError:
                return None
            # Not bngsim (python/), and not an environment inside the checkout
            # (CI's .venv/ is), whose packages are no leg's code.
            if rel.parts[0] == "python" or rel.parts[0].startswith("."):
                return None
            if "site-packages" in rel.parts:
                return None
            return rel

        def spec_of(name):
            try:
                found = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                return None
            return found if found is not None and found.has_location else None

        todo = [str(script)] + [getattr(m, "__file__", None) for m in list(sys.modules.values())]
        seen = set()
        while todo:
            rel = ours(todo.pop())
            if rel is None or rel in seen:
                continue
            seen.add(rel)
            if rel.suffix != ".py":
                continue
            for node in ast.walk(ast.parse((repo / rel).read_text())):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    names = [node.module] + [f"{{node.module}}.{{a.name}}" for a in node.names]
                else:
                    continue
                for name in names:
                    # The top-level name alone, unless it is the repo's: finding
                    # a dotted name imports its parents.
                    top = spec_of(name.split(".")[0])
                    if top is None:
                        todo += in_repo(name.split(".")[0])
                        continue
                    if ours(top.origin) is None:
                        continue
                    found = spec_of(name) if "." in name else top
                    if found is not None:
                        todo.append(found.origin)
        print(json.dumps({{"files": sorted(map(str, seen))}}))
    """)
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO, timeout=300
    )
    assert done.returncode == 0, (script, done.stderr.strip().splitlines()[-5:])
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert "broken" not in out, f"{script} cannot import the repo's {out.get('broken')}"
    if "missing" in out:
        pytest.skip(f"{script} needs {out['missing']}, which is not installed here")
    return set(out["files"])


@pytest.mark.parametrize(
    ("leg", "script"), [(leg, s) for leg, scripts in RUNNERS.items() for s in scripts]
)
def test_every_file_a_runner_loads_feeds_its_leg(leg, script):
    """The table is measured, not guessed: a runner that comes to load
    another file fails here until the table says that file feeds its leg."""
    files = _loaded_files(script)
    assert script in files
    for f in files:
        assert leg in pr_legs.legs_for([f]), (script, f)


def _jobs() -> dict[str, str]:
    """Each job's block of the workflow, by its id."""
    body = WORKFLOW.read_text().split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_each_job_runs_the_runners_its_leg_names():
    """RUNNERS is what each leg's job runs (the nightly's own scripts aside,
    which feed every leg), and only a leg's job runs one: a job that came to
    run another script would go unmeasured, and one that is no leg would run
    on every pull request."""
    jobs = _jobs()
    assert set(RUNNERS) == ALL and set(jobs) >= ALL
    for job, block in jobs.items():
        run = set(re.findall(r"uv run --no-sync python (\S+\.py)", block))
        run = {s for s in run if not s.startswith("parity_checks/nightly/")}
        assert run == set(RUNNERS.get(job, ())), job


def test_every_corpus_input_feeds_every_corpus_leg():
    """The corpus job's cache key names what the corpus is built from: a change
    to any of them rebuilds what every corpus leg reads."""
    key = re.search(r"key=rr-corpus-v\d+-\$\{\{ hashFiles\((.*)\) \}\}", WORKFLOW.read_text())
    inputs = re.findall(r"'([^']+)'", key.group(1))
    assert inputs
    for pattern in inputs:
        path = pattern.removesuffix("**") + ("any.xml" if pattern.endswith("**") else "")
        assert pr_legs.legs_for([path]) >= pr_legs.CORPUS_LEGS, pattern


def _workflow_paths(negated: bool) -> list[str]:
    """pull_request.paths: its exclusions ("!..."), or the rest. Quoted or not."""
    text = WORKFLOW.read_text()
    block = text.split("  pull_request:\n    paths:\n", 1)[1].split("\n\n", 1)[0]
    items = [
        re.sub(r"\s+#.*", "", i).strip("\"'") for i in re.findall(r"^\s+- (.+)$", block, re.M)
    ]
    return (
        [i[1:] for i in items if i.startswith("!")]
        if negated
        else [i for i in items if not i.startswith("!")]
    )


def _workflow_triggers() -> list[str]:
    return _workflow_paths(negated=False)


def test_what_the_workflow_excludes_feeds_no_leg():
    """A path the workflow is not triggered on alone asks for no leg beside
    other changes either."""
    excluded = _workflow_paths(negated=True)
    assert "**/*.md" in excluded
    for pattern in excluded:
        # A path the glob matches, under a leg's directory.
        path = pattern.replace("**/", "parity_checks/rr_parity/").replace("**", "x.json")
        assert pr_legs.legs_for([path.replace("*", "README")]) == set(), pattern


def test_the_script_reads_the_paths_the_workflow_is_triggered_on():
    """TRIGGERS is the workflow's pull_request.paths: a trigger the script did
    not read would run its PR with no leg."""
    as_prefixes = [p.removesuffix("**") for p in _workflow_triggers()]
    assert as_prefixes and sorted(as_prefixes) == sorted(pr_legs.TRIGGERS)


def test_an_unruled_trigger_runs_every_leg(monkeypatch):
    """When unsure, run the leg."""
    monkeypatch.setattr(pr_legs, "TRIGGERS", (*pr_legs.TRIGGERS, "parity_checks/copasi_parity/"))
    assert pr_legs.legs_for(["parity_checks/copasi_parity/copasi_run.py"]) == ALL


def test_every_path_the_workflow_triggers_on_has_a_rule():
    """A trigger the table does not name would run every leg, unmeasured."""
    triggers = _workflow_triggers()
    assert triggers
    for pattern in triggers:
        path = pattern.removesuffix("/**") + ("/any.py" if pattern.endswith("/**") else "")
        named = [prefix for prefix, _ in pr_legs.RULES if path.startswith(prefix)]
        assert named, pattern


def test_the_workflow_runs_each_leg_on_its_flag():
    """Each leg's job is gated on the plan job's output of the same name, and
    the plan job outputs every one the script prints."""
    text = WORKFLOW.read_text()
    for name in [*pr_legs.LEGS, "corpus"]:
        assert f"      {name}: ${{{{ steps.legs.outputs.{name} }}}}" in text, name
        assert f"    if: needs.plan.outputs.{name} == 'true'" in text, name
    assert "python3 parity_checks/nightly/pr_legs.py --all" in text
