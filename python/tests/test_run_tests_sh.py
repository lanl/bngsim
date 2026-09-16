"""``run_tests.sh`` must run the suite, and run it against the real repo.

The script exists for one reason: the source tree's ``python/bngsim/`` can
shadow the *installed* package, so the tests are relocated away from it. Getting
that wrong has now cost twice.

Issue #578: the copy took only ``test_*.py`` + ``conftest.py``, so
``__init__.py`` (which ``test_sbml_psa.py``'s relative imports need) and the
sibling helpers ``_ci_workflow.py`` / ``_extrande_reference.py`` never arrived.
Six modules raised at collection, and pytest aborts the run on a collection
error — the script executed none of the 5775 tests it had counted.

Issue #590: the relocated tests were put in a bare temp directory, which hid far
more than ``python/bngsim/``. Sixty-nine modules resolve ``tests/data``,
``benchmarks/`` or ``scripts/`` by walking up from ``__file__``, and that
walk-up left the repo entirely: 64 failures, 181 errors and 359 spurious skips
reported as *"not in this checkout (installed package)"* while the source tree
sat right there. The layout is now a stand-in for the repo — ``python/tests``
copied, every other top-level entry symlinked — so a walk-up lands somewhere
that *is* the repo, while ``python/`` still holds nothing but ``tests``.

What is pinned here:

* the copy takes every ``.py`` in ``python/tests``, and lands at
  ``python/tests`` — both halves of the name are load-bearing, ``tests`` because
  ``__init__.py`` makes pytest derive the package name from the directory, and
  ``python`` because that is the directory pytest then puts on ``sys.path``;
* everything else is linked, the caches are not (a run here must not write into
  the developer's own), and ``python/`` is not;
* the shadowing guarantee itself: the stand-in's ``python/`` contains ``tests``
  and nothing else, so there is no ``bngsim`` for ``sys.path`` to find;
* a ``parents[2]`` walk-up from a copied module reaches the repo's real
  ``tests/data``, ``benchmarks/`` and ``scripts/``;
* end to end: the stand-in collects *exactly* what an in-place run collects, for
  the ten modules #578 broke, both exiting 0. Equality is the assertion — a
  count that merely stopped erroring would still hide a quiet shrink;
* ``_source_root.bngsim_source_root`` resolves from either env var with no
  usable ``__file__``, refuses a directory that is not a bngsim checkout, and
  returns ``None`` rather than a guess.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root()
_SCRIPT = None if _ROOT is None else _ROOT / "run_tests.sh"
if _SCRIPT is None or not _SCRIPT.is_file():
    pytest.skip("run_tests.sh not in this checkout", allow_module_level=True)

# POSIX-specific, on two counts: the thing under test is a bash script, and the
# stand-in it builds is a symlink farm, which Windows only permits under
# Developer Mode or an elevated token. The full suite runs on ubuntu and macOS
# (python-tests.yml); the Windows legs run curated file lists that do not include
# this module, so this guard is belt-and-braces rather than load-bearing today.
if os.name == "nt":  # pragma: no cover - the CI legs that run this are POSIX
    pytest.skip(
        "POSIX-specific: run_tests.sh is bash and builds a symlink farm", allow_module_level=True
    )

_TESTS_DIR = _ROOT / "python" / "tests"

#: Directories the script must NOT link, so a run in the stand-in cannot write
#: into the developer's own caches (GH #372 gave the suite its own for that).
EXCLUDED = ("python", ".pytest_cache", ".mypy_cache", ".ruff_cache")

#: The modules issue #578 broke. The first six raised at collection; the last
#: four silently lost parametrized cases or skipped themselves whole.
AFFECTED = (
    "test_ci_run_list_coverage.py",  # _ci_workflow
    "test_windows_klu_ci_coverage.py",  # _ci_workflow
    "test_windows_nfsim_ci_coverage.py",  # _ci_workflow
    "test_ssa_variable_volume.py",  # _extrande_reference
    "test_sbml_psa.py",  # relative import, needs __init__.py
    "test_sbml_suite_grading.py",  # _grading, outside python/tests
    "test_benchmark_runner_help.py",  # 127 cases -> 11
    "test_codegen_structural_key.py",  # 37 cases -> 24
    "test_ssa_baseline_regression.py",  # whole module skipped
    "test_ssa_biomodels_parity_zgate.py",  # whole module skipped
)


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _build_standin(dest: Path) -> Path:
    """Reproduce the script's layout: copy python/tests, link everything else."""
    (dest / "python" / "tests").mkdir(parents=True)
    for src in sorted(_TESTS_DIR.glob("*.py")):
        shutil.copy2(src, dest / "python" / "tests" / src.name)
    for entry in sorted(_ROOT.iterdir()):
        if entry.name in EXCLUDED:
            continue
        (dest / entry.name).symlink_to(entry)
    return dest


# --------------------------------------------------------------------------- #
# the script's contract
# --------------------------------------------------------------------------- #
def test_copy_takes_every_module_in_the_tests_package():
    """The cp glob must cover `python/tests` entirely, not just `test_*.py`.

    Asserted as set equality against the directory rather than by matching the
    literal glob, so any narrowing — a new prefix convention, an added exclude —
    fails here whatever form it takes.
    """
    assert 'cp "${TEST_SRC}"/*.py' in _script_text(), "the copy no longer takes every .py"

    copied = {p.name for p in _TESTS_DIR.glob("*.py")}
    present = {p.name for p in _TESTS_DIR.iterdir() if p.is_file()}
    assert copied == present, f"non-.py files in python/tests are not copied: {present - copied}"
    # The three classes the #578 glob dropped, named so a future narrowing has
    # to argue with them individually.
    assert "__init__.py" in copied
    assert {"_ci_workflow.py", "_extrande_reference.py", "_source_root.py"} <= copied
    assert "conftest.py" in copied


def test_copy_lands_at_python_slash_tests():
    """Both halves of the path are load-bearing.

    `tests`, because `__init__.py` makes pytest derive the package name from the
    directory — a raw mktemp name like `tmp.AbC123` is not a legal identifier.
    `python`, because that is the directory pytest then puts on `sys.path`, and
    it is the one whose `bngsim/` this whole exercise exists to hide.
    """
    text = _script_text()
    assert 'mkdir -p "$STANDIN/python/tests"' in text
    assert 'cp "${TEST_SRC}"/*.py "$STANDIN/python/tests/"' in text
    assert '-m pytest "$STANDIN/python/tests"' in text


def test_everything_but_python_and_the_caches_is_linked():
    """The link farm is what makes a `__file__` walk-up land in the real repo."""
    text = _script_text()
    assert 'ln -s "$entry" "$STANDIN/$name"' in text
    for name in EXCLUDED:
        assert name in text, f"{name} is no longer excluded from the link farm"


def test_the_working_directory_variable_is_not_named_tmpdir():
    """Assigning `TMPDIR` in the script redirects every CHILD's temp files.

    It is already exported on macOS and honoured by Python's `tempfile`, so the
    stand-in would become pytest's `tmp_path` root — putting every temp
    directory inside a tree whose root carries bngsim's `pyproject.toml`. A
    walk-up out of any of them then "finds" a source root that is a link farm,
    and the per-run artifact cache lands inside `rootdir`, which
    `test_artifact_cache_isolation.py` asserts against. Three tests failed that
    way before this was renamed.
    """
    body = "\n".join(ln for ln in _script_text().splitlines() if not ln.lstrip().startswith("#"))
    assert "TMPDIR=" not in body, "the script assigns TMPDIR, hijacking every child's tempdir"


def test_the_script_refuses_a_shell_that_cannot_make_symlinks(tmp_path):
    """A silent deep copy would be far worse than a refusal.

    `ln -s` copies instead of linking under Git Bash/MSYS without Developer Mode.
    The link farm spans the whole repo, so a copy there would duplicate `build/`,
    `third_party/`, `.git` and `.venv` — minutes and gigabytes, where the point
    of linking was that it costs nothing. The script checks one entry and exits
    non-zero with the fix rather than proceeding.
    """
    shimdir = tmp_path / "bin"
    shimdir.mkdir()
    # An `ln` that makes a regular file, which is what MSYS does to a symlink it
    # is not allowed to create. Touching rather than copying keeps this cheap;
    # the guard looks at the file TYPE, which is the part MSYS gets wrong.
    shim = shimdir / "ln"
    shim.write_text('#!/usr/bin/env bash\n: > "${!#}"\n', encoding="utf-8")
    shim.chmod(0o755)

    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    env["PATH"] = f"{shimdir}{os.pathsep}{env['PATH']}"
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "--collect-only", "-q"],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, f"expected a refusal, got {proc.returncode}:\n{out[-2000:]}"
    assert "is not a symlink" in out, out[-2000:]
    assert "winsymlinks:nativestrict" in out, "the refusal must name the Windows fix"
    assert "tests collected" not in out, "it ran the suite anyway"


def test_rig_exports_both_source_locating_env_vars():
    """For the modules that consult them instead of walking up."""
    text = _script_text()
    assert 'BNGSIM_TEST_DATA="$DATA_DIR"' in text
    assert 'BNGSIM_SOURCE_ROOT="$SCRIPT_DIR"' in text


# --------------------------------------------------------------------------- #
# the layout, built and inspected
# --------------------------------------------------------------------------- #
def test_the_standin_hides_the_source_package_and_nothing_else(tmp_path):
    """The one guarantee the script exists for, asserted directly.

    `python/` must hold `tests` and nothing else — no `bngsim` for `sys.path` to
    find ahead of the installed one. Everything the repo has *besides* `python/`
    must be reachable, or we are back to #590.
    """
    standin = _build_standin(tmp_path / "standin")

    assert {p.name for p in (standin / "python").iterdir()} == {"tests"}
    assert not (standin / "python" / "bngsim").exists()

    linked = {p.name for p in standin.iterdir()}
    expected = {p.name for p in _ROOT.iterdir() if p.name not in EXCLUDED} | {"python"}
    assert linked == expected
    for cache in (".pytest_cache", ".mypy_cache", ".ruff_cache"):
        assert not (standin / cache).exists(), f"{cache} must not be reachable from the stand-in"


def test_a_walk_up_from_a_copied_module_reaches_the_real_repo(tmp_path):
    """`Path(__file__).resolve().parents[2]` is what 69 modules rely on.

    Under the #590 layout it left the repo; under this one it lands on the
    stand-in, whose entries resolve through to the real trees.
    """
    standin = _build_standin(tmp_path / "standin")
    copied = standin / "python" / "tests" / "conftest.py"
    walked_up = copied.resolve().parents[2]

    assert walked_up == standin.resolve()
    for rel in ("tests/data", "benchmarks", "scripts", "pyproject.toml"):
        assert (walked_up / rel).exists(), f"{rel} unreachable from the walk-up"
    assert (walked_up / "tests" / "data").resolve() == (_ROOT / "tests" / "data").resolve()


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def _collect(paths: list[Path], cwd: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["BNGSIM_TEST_DATA"] = str(_ROOT / "tests" / "data")
    env["BNGSIM_SOURCE_ROOT"] = str(_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + [str(p) for p in paths],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _collected_count(output: str) -> int:
    for line in reversed(output.splitlines()):
        if " collected" in line and ("test" in line):
            for token in line.replace("=", " ").split():
                if token.isdigit():
                    return int(token)
    raise AssertionError(f"no collection summary in:\n{output[-2000:]}")


def test_standin_collects_exactly_what_an_in_place_run_collects(tmp_path):
    """The whole point of the script, measured rather than asserted about.

    Equality, not "greater than zero": before #578 six of these modules errored
    (taking the session down with them) and four collected a strict subset of
    their cases. Only the first failure is loud.
    """
    standin = _build_standin(tmp_path / "standin")
    dest = standin / "python" / "tests"

    rc_copy, out_copy = _collect([dest / name for name in AFFECTED], standin)
    rc_tree, out_tree = _collect([_TESTS_DIR / name for name in AFFECTED], _ROOT)

    assert rc_tree == 0, f"in-place collection failed:\n{out_tree[-2000:]}"
    assert rc_copy == 0, f"collection from the stand-in failed:\n{out_copy[-2000:]}"
    assert _collected_count(out_copy) == _collected_count(out_tree)


#: Modules that the #590 layout broke by *execution*, not collection: each
#: reaches the repo by walking up from ``__file__``, and each is cheap (text and
#: file-existence checks, no solves). Under the old layout all 17 of these tests
#: failed or errored; a -k probe over them is the whole defect in ~11s.
WALK_UP_PROBE = "changelog_structure or vendoring or ship_wheel_build_deps"


def test_the_script_itself_runs_walk_up_dependent_tests_green():
    """Run the real script, not a reimplementation of it, and RUN rather than collect.

    Everything above builds the stand-in the way the script is *supposed* to, so
    it pins the layout's properties but would pass against a script that built
    something else. And collection alone proves nothing here: #591 already got
    collection to parity, and #590 is the *execution* gap it left behind — these
    modules collected fine and then failed on a path that had left the repo.

    So this invokes ``run_tests.sh`` on modules that resolve the repo by walking
    up from ``__file__``, and demands they pass. Before this change every one of
    them failed or errored there while passing in place.
    """
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "-q", "-p", "no:cacheprovider", "-k", WALK_UP_PROBE],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"run_tests.sh exited {proc.returncode}:\n{out[-4000:]}"

    summary = next(
        (ln for ln in reversed(out.splitlines()) if re.search(r"\d+ (passed|failed)", ln)), ""
    )
    assert summary, f"no pytest summary line:\n{out[-3000:]}"
    assert "failed" not in summary and "error" not in summary, summary

    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 15, (
        f"the -k probe has gone blind — only {summary!r}; it must actually run these modules"
    )
    # The tell-tale of the #590 layout: a source-tree guard reporting the tree
    # absent while the script is being run from inside it.
    assert "not in this checkout" not in out, (
        f"a module still cannot see the source tree from the stand-in:\n{out[-3000:]}"
    )


# --------------------------------------------------------------------------- #
# the source-root resolver
# --------------------------------------------------------------------------- #
def _resolver_from(tmp_path: Path):
    """Import a copy of `_source_root` that sits OUTSIDE any bngsim checkout.

    Importing the in-tree module would let its own `__file__` walk-up answer
    every question, which is precisely the route the env vars exist to replace.
    """
    import importlib.util

    shutil.copy2(_TESTS_DIR / "_source_root.py", tmp_path / "_source_root_copy.py")
    spec = importlib.util.spec_from_file_location(
        "_source_root_copy", tmp_path / "_source_root_copy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bngsim_source_root


@pytest.mark.parametrize("var", ["BNGSIM_TEST_DATA", "BNGSIM_SOURCE_ROOT"])
def test_source_root_resolves_from_either_env_var_alone(tmp_path, monkeypatch, var):
    """Each var on its own finds the repo, with no usable `__file__` to fall back on."""
    resolve = _resolver_from(tmp_path)
    monkeypatch.delenv("BNGSIM_TEST_DATA", raising=False)
    monkeypatch.delenv("BNGSIM_SOURCE_ROOT", raising=False)
    value = _ROOT / "tests" / "data" if var == "BNGSIM_TEST_DATA" else _ROOT
    monkeypatch.setenv(var, str(value))
    assert resolve() == _ROOT


def test_source_root_returns_none_rather_than_a_guess(tmp_path, monkeypatch):
    """Nothing points at a checkout ⇒ `None`, which callers turn into a skip.

    A wrong answer here is worse than none: it would send a module globbing an
    unrelated directory and reporting the empty result as a pass.
    """
    resolve = _resolver_from(tmp_path)
    monkeypatch.delenv("BNGSIM_TEST_DATA", raising=False)
    monkeypatch.delenv("BNGSIM_SOURCE_ROOT", raising=False)
    assert resolve() is None


def test_source_root_refuses_a_directory_that_is_not_a_bngsim_checkout(tmp_path, monkeypatch):
    """A `pyproject.toml` is not enough — it has to be bngsim's."""
    resolve = _resolver_from(tmp_path)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "pyproject.toml").write_text('[project]\nname = "not-bngsim"\n', encoding="utf-8")
    monkeypatch.delenv("BNGSIM_TEST_DATA", raising=False)
    monkeypatch.setenv("BNGSIM_SOURCE_ROOT", str(decoy))
    assert resolve() is None
