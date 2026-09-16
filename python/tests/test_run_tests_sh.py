"""``run_tests.sh`` must actually run the suite (issue #578).

The script copies the tests to a temp directory so the source tree's
``python/bngsim/`` cannot shadow the *installed* package it exists to exercise.
It copied only ``test_*.py`` + ``conftest.py``, which left out three things the
suite needs and two the temp dir's own name broke:

* ``__init__.py`` — ``python/tests`` is a package, and ``test_sbml_psa.py``
  imports its fixtures with ``from .test_ssa_psa_volume import ...``;
* the sibling helpers ``_ci_workflow.py`` and ``_extrande_reference.py``, which
  five modules import at module scope;
* ``_grading.py``'s location, which ``test_sbml_suite_grading.py`` reached by a
  ``__file__`` walk-up that lands outside the repo once the file has moved.

Six modules therefore raised at collection time, and pytest aborts the whole run
on a collection error — so the script could not execute a single one of the 5775
tests it had just collected. Four more modules did not raise: they resolved a
repo path from ``__file__``, came up empty and quietly collected *fewer*
parametrized cases (or skipped themselves whole), which is the worse failure —
a green run that dropped 168 tests without saying so.

What is pinned here:

* the copy takes every ``.py`` in ``python/tests``, into a directory named
  ``tests`` — the name matters, because with ``__init__.py`` present pytest
  derives the package name from the directory, and ``tmp.AbC123`` is not a legal
  one;
* the rig exports both ``BNGSIM_TEST_DATA`` and ``BNGSIM_SOURCE_ROOT``, which is
  how a relocated module finds the repo at all;
* end to end: the copy collects *exactly* what an in-place run collects, for the
  ten modules the bug touched, both exiting 0. Equality is the assertion — a
  count that merely stopped erroring would still hide the quiet shrink;
* ``_source_root.bngsim_source_root`` resolves from either env var with no
  usable ``__file__``, refuses a directory that is not a bngsim checkout, and
  returns ``None`` rather than a guess when there is nothing to find.

Scoped to those ten modules on purpose: collecting all 296 twice costs ~12s and
would pin the same property no more firmly for the code that regressed.
"""

from __future__ import annotations

import os
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

_TESTS_DIR = _ROOT / "python" / "tests"

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


# --------------------------------------------------------------------------- #
# the script's contract
# --------------------------------------------------------------------------- #
def test_copy_takes_every_module_in_the_tests_package():
    """The cp glob must cover `python/tests` entirely, not just `test_*.py`.

    Asserted as set equality against the directory rather than by matching the
    literal glob, so any narrowing — a new prefix convention, an added exclude —
    fails here whatever form it takes.
    """
    text = _script_text()
    assert 'cp "${TEST_SRC}"/*.py' in text, "the copy no longer takes every .py"

    copied = {p.name for p in _TESTS_DIR.glob("*.py")}
    present = {p.name for p in _TESTS_DIR.iterdir() if p.is_file()}
    assert copied == present, f"non-.py files in python/tests are not copied: {present - copied}"
    # The three classes the old glob dropped, named so a future narrowing has to
    # argue with them individually.
    assert "__init__.py" in copied
    assert {"_ci_workflow.py", "_extrande_reference.py", "_source_root.py"} <= copied
    assert "conftest.py" in copied


def test_copy_lands_in_a_directory_named_tests():
    """With `__init__.py` present, the directory name IS the package name.

    pytest derives it from the directory and puts the PARENT on `sys.path`. A
    raw mktemp name (`tmp.AbC123`) is not a legal identifier, and the parent it
    would add is the temp root rather than a stand-in for `python/` — so the
    name is load-bearing twice over.
    """
    text = _script_text()
    assert 'mkdir "$TMPDIR/tests"' in text
    assert 'cp "${TEST_SRC}"/*.py "$TMPDIR/tests/"' in text
    assert '-m pytest "$TMPDIR/tests"' in text


def test_rig_exports_both_source_locating_env_vars():
    """A relocated module cannot walk up to the repo; the env vars are the only route."""
    text = _script_text()
    assert 'BNGSIM_TEST_DATA="$DATA_DIR"' in text
    assert 'BNGSIM_SOURCE_ROOT="$SCRIPT_DIR"' in text


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
        if "tests collected" in line or "test collected" in line:
            return int(line.split()[-4] if "error" in line else line.split("test")[0].split()[-1])
    raise AssertionError(f"no collection summary in:\n{output[-2000:]}")


def test_copied_suite_collects_exactly_what_an_in_place_run_collects(tmp_path):
    """The whole point of the script, measured rather than asserted about.

    Equality, not "greater than zero": before the fix six of these modules
    errored (taking the session down with them) and four collected a strict
    subset of their cases. Only the first failure is loud.
    """
    dest = tmp_path / "tests"
    dest.mkdir()
    for src in sorted(_TESTS_DIR.glob("*.py")):
        shutil.copy2(src, dest / src.name)

    rc_copy, out_copy = _collect([dest / name for name in AFFECTED], tmp_path)
    rc_tree, out_tree = _collect([_TESTS_DIR / name for name in AFFECTED], _ROOT)

    assert rc_tree == 0, f"in-place collection failed:\n{out_tree[-2000:]}"
    assert rc_copy == 0, f"collection from the copy failed:\n{out_copy[-2000:]}"
    assert _collected_count(out_copy) == _collected_count(out_tree)


# --------------------------------------------------------------------------- #
# the source-root resolver
# --------------------------------------------------------------------------- #
def _resolver_from(tmp_path: Path):
    """Import a copy of `_source_root` that sits OUTSIDE any bngsim checkout.

    Importing the in-tree module would let its own `__file__` walk-up answer
    every question, which is precisely the route the env vars exist to replace.
    """
    shutil.copy2(_TESTS_DIR / "_source_root.py", tmp_path / "_source_root_copy.py")
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        spec = importlib.util.spec_from_file_location(
            "_source_root_copy", tmp_path / "_source_root_copy.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.bngsim_source_root
    finally:
        sys.path.remove(str(tmp_path))


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
