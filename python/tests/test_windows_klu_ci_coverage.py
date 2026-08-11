"""The Windows sparse/KLU leg's own coverage is the thing under test.

Issue #296. Every Windows leg built ``-DBNGSIM_ENABLE_KLU=OFF`` — deliberately,
and for the same stated reason on each: to skip the from-source SuiteSparse
build that would dominate the job. ``wheels.yml`` does build a Windows+KLU
artifact, and its ``test-command`` asserts ``bngsim.HAS_KLU`` and nothing about
what it computes. So ``{Windows} x {KLU on} x {runs pytest}`` was empty, and the
one codegen entry point that is resolved *by name* out of the built DLL — the
half whose failure mode is platform-specific, and the class of defect that made
``windows-tail.yml`` exist — had never executed on Windows.

``windows-klu.yml`` is that intersection. This file guards the two ways it can
stop being it, neither of which any runtime test reads:

* **The build flag.** Every test the leg runs is behind a KLU gate, so a leg
  that lost ``ENABLE_KLU=ON`` does not fail, it skips twenty tests and reports
  green. That is issue #296 restored, in the job written to close it.
* **The file list.** A new entirely-KLU-gated test file is invisible on Windows
  the day it lands unless somebody remembers this workflow — the recurring shape
  behind #291 and #295. So the required list is computed from the tests
  directory rather than written down here.

Text assertions over the workflow, in the shape of
``test_windows_nfsim_ci_coverage.py`` and ``test_lapack_dense_ci_coverage.py``:
the functional proof lives in CI, and this is what keeps the proof from being
deleted. The cross-workflow property — that a leg either runs a file it fires on
or says in the workflow why it does not — is #295's
``test_ci_run_list_coverage.py``, which picks this workflow up by shape with no
edit; the two overlap here by design.
"""

from __future__ import annotations

import ast
import os
import re
import sys

import pytest

# Sibling helper import under pytest's importlib mode, as test_ssa_variable_volume.py
# does it. The parsers are shared with the #291/#295 coverage tests; private copies
# of them would be the drift these files exist to catch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ci_workflow import (  # noqa: E402
    REPO_ROOT,
    TESTS_DIR,
    WORKFLOWS,
    expand,
    paths_filter_selectors,
    pytest_selectors,
    strip_comments,
)

WORKFLOW = WORKFLOWS / "windows-klu.yml"

#: Any mention of KLU, in a skip condition or a decorator. Case-insensitive and
#: deliberately loose: the two gates in the tree spell it ``bngsim.HAS_KLU`` and
#: ``_klu_available()``, and a third will spell it a third way.
KLU = re.compile(r"klu", re.I)

#: The source files whose contents decide what this leg is testing. A leg
#: registered only under its own test filenames fires when the tests change and
#: not when the code they guard does — the one time it cannot catch a regression
#: (the #167 gate). Kept to the two ends of the mechanism the leg exists for:
#: the emitter that writes ``bngsim_codegen_jac_sparse`` and the call site that
#: resolves it out of the built DLL.
REQUIRED_SOURCE_TRIGGERS = (
    "python/bngsim/_codegen.py",
    "src/cvode_simulator.cpp",
    "src/steady_state.cpp",
)

#: Known members of the computed list below. The list is derived, so it can go
#: empty — a renamed decorator, an ``ast`` shape this scanner does not model —
#: and an empty required-set passes every assertion it feeds. Found by mutation
#: while writing #295's parser: the failure that matters is the check going
#: blind, not the check failing.
KNOWN_GATED = frozenset({"test_codegen_jacobian_sparse.py", "test_steady_state_linear_solver.py"})

pytestmark = pytest.mark.skipif(not WORKFLOW.exists(), reason=".github/ not in this checkout")


def _body() -> str:
    return strip_comments(WORKFLOW.read_text(encoding="utf-8"))


def _klu_gate_names(tree: ast.Module) -> set[str]:
    """Module-level names bound to a KLU-conditioned ``skipif`` mark.

    ``needs_klu = pytest.mark.skipif(not (_CC and _klu_available()), ...)`` —
    the name is what the tests below are decorated with, so the decorators have
    to be matched against it rather than against the word KLU.
    """
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        text = ast.unparse(node.value)
        if "skipif" in text and KLU.search(text):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _module_is_klu_gated(tree: ast.Module, gates: set[str]) -> bool:
    """``pytestmark = skipif(not bngsim.HAS_KLU, ...)`` or a module-level skip."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            text = ast.unparse(node.value)
            if "pytestmark" in targets and (KLU.search(text) or any(g in text for g in gates)):
                return True
        if isinstance(node, ast.Expr):
            text = ast.unparse(node)
            if "pytest.skip" in text and KLU.search(text):
                return True
    return False


def _every_test_is_klu_gated(tree: ast.Module, gates: set[str]) -> bool:
    """Every top-level test function/class carries a KLU gate as a decorator.

    ``all()`` over a non-empty set, so a single ungated test takes the file out
    of the required list. That is the safe direction: the assertion this feeds
    is "must be run by the leg", and a file with a case that runs without KLU
    has a home on a leg that is not this one.
    """
    items = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
        or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
    ]
    if not items:
        return False
    return all(
        any(
            KLU.search(text) or any(text == gate or text.startswith(gate + "(") for gate in gates)
            for text in (ast.unparse(decorator) for decorator in node.decorator_list)
        )
        for node in items
    )


def entirely_klu_gated() -> set[str]:
    """Test files in which nothing can run without a KLU build."""
    gated = set()
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        gates = _klu_gate_names(tree)
        if _module_is_klu_gated(tree, gates) or _every_test_is_klu_gated(tree, gates):
            gated.add(path.name)
    return gated


class TestTheLegRunsEveryFileThatNeedsIt:
    def test_the_scanner_still_sees_the_files_it_was_written_for(self):
        """Guard the derivation before asserting on it.

        Every check below is driven by ``entirely_klu_gated()``. If the scanner
        stopped recognising a gate — a renamed decorator, a mark applied a way
        this does not model — the required set would empty and the file would
        report green while asserting nothing. That is the failure mode, not a
        missing file.
        """
        gated = entirely_klu_gated()
        assert gated, (
            "no python/tests file scanned as entirely KLU-gated. This module's own "
            "premise is broken -- the gate is spelled a way _klu_gate_names / "
            "_module_is_klu_gated do not recognise, and every assertion below is "
            "now vacuous."
        )
        missing = sorted(KNOWN_GATED - gated)
        assert not missing, (
            f"{missing} used to scan as entirely KLU-gated and no longer does. Either "
            "the file grew a case that runs without KLU (fine -- update KNOWN_GATED) "
            "or the scanner went blind (not fine)."
        )

    def test_every_entirely_klu_gated_file_is_executed(self):
        """The regression itself, in the form it would come back.

        Asserted against the filesystem rather than against the workflow's own
        paths filter, so a new KLU-only test file fails here on the day it lands
        instead of the day somebody notices Windows never ran it. #291 and #295
        were both this, one workflow at a time.
        """
        executed = expand(pytest_selectors(_body()))
        missing = sorted(entirely_klu_gated() - executed)
        assert not missing, (
            f"windows-klu.yml does not run {missing}. Nothing in those files can "
            "execute without a KLU build, and this is the only leg anywhere that "
            "builds one on Windows -- so they run on Linux and macOS and nowhere "
            "else (issue #296). Add them to the pytest call and to the paths filter."
        )

    def test_the_trigger_and_the_run_step_cannot_disagree(self):
        """Everything the filter fires on is something the run step runs.

        The converse is fine: running a file the filter ignores costs a little
        wall clock and never a false green.
        """
        body = _body()
        triggered = expand(paths_filter_selectors(body))
        assert triggered, "parsed no python/tests entries from windows-klu.yml's paths filter"
        unrun = sorted(triggered - expand(pytest_selectors(body)))
        assert not unrun, f"windows-klu.yml fires on {unrun} but does not run them."

    def test_the_leg_fires_on_the_code_it_guards(self):
        """A leg registered only under its own test filenames runs when the
        tests change and not when the emitter or the resolve does -- i.e. never
        when the thing it guards breaks. The #167 gate, on this workflow.
        """
        body = _body()
        missing = [path for path in REQUIRED_SOURCE_TRIGGERS if path not in body]
        assert not missing, (
            f"windows-klu.yml does not trigger on {missing}. Those are the two ends of "
            "the mechanism the leg exists for -- the emit of bngsim_codegen_jac_sparse "
            "and its by-name resolve out of the built DLL -- so a change to either one "
            "would not summon the only job that runs it on Windows."
        )


class TestTheLegCannotGoGreenHavingRunNothing:
    def test_the_build_requires_klu(self):
        """The flag is the leg.

        Every test it runs is KLU-gated, so ``ENABLE_KLU=OFF`` here does not
        fail the job -- it skips the cluster and reports green, which is exactly
        the state #296 describes. ``REQUIRE_KLU=ON`` is asserted alongside so a
        SuiteSparse that stops being discoverable fails at configure with the
        CMake fix-it message rather than at pytest with twenty skips.
        """
        body = _body()
        assert "BNGSIM_ENABLE_KLU=ON" in body, (
            "windows-klu.yml no longer builds with BNGSIM_ENABLE_KLU=ON. It is the "
            "only Windows leg that does; without it the cluster skips and the job "
            "stays green (issue #296)."
        )
        assert "BNGSIM_REQUIRE_KLU=ON" in body, (
            "windows-klu.yml no longer passes BNGSIM_REQUIRE_KLU=ON, so a build that "
            "cannot find SuiteSparse degrades to a green run of twenty skips instead "
            "of failing at configure."
        )
        assert "BNGSIM_ENABLE_KLU=OFF" not in body, (
            "windows-klu.yml passes BNGSIM_ENABLE_KLU=OFF somewhere -- probably copied "
            "from windows-tail.yml, where it is deliberate. Here it empties the job."
        )

    def test_a_build_without_klu_fails_the_leg(self):
        """Belt to the flag's braces: the flag says what was asked for, this
        says what was linked. Same shape as windows-nfsim.yml's HAS_NFSIM
        assertion and python-tests.yml's HAS_KLU one.
        """
        assert re.search(r"sys\.exit\([^)]*HAS_KLU", _body()), (
            "windows-klu.yml prints HAS_KLU but does not fail on it. A build that "
            "configured with KLU on and linked it anyway would report a green run "
            "of skips."
        )

    def test_a_missing_compiler_fails_the_leg(self):
        """``test_codegen_jacobian_sparse.py``'s KLU mark is
        ``skipif(not (_CC and _klu_available()))`` -- one reason string for two
        conditions, and it names the solver. So a runner that lost cc/clang/gcc
        would take 13 of the 20 tests out under a reason that blames the build.
        The zero-skip guard catches it either way; this makes the log say which.
        """
        body = _body()
        assert re.search(r"which\('cc'\)|which\(\"cc\"\)", body), (
            "windows-klu.yml no longer checks for a C compiler on PATH. Two thirds of "
            "the cluster gates on `shutil.which('cc') or ... clang ... gcc` and reports "
            "its absence as a KLU skip."
        )

    def test_a_shrinking_run_fails_the_leg(self):
        """The count guard backstops what a text assertion cannot express -- a
        rename that orphans a file, a conftest that deselects, a gate that grows
        a second condition. On a twenty-test single-purpose cluster the skip
        count is the sharp half: every reachable skip reason here is a false
        green, and a pass floor with the usual slack cannot see a seven-test
        file stop being collected (#291).
        """
        body = _body()
        assert "PASS_FLOOR" in body, (
            "windows-klu.yml no longer floors its passed count, so a file that stopped "
            "being collected would not fail the leg."
        )
        assert re.search(r'count skipped|"\$skipped"', body), (
            "windows-klu.yml no longer checks its skip count. On this leg a skip is a "
            "false green -- see the comment on that check."
        )


class TestTheSuiteSparseRecipeIsTheWheelsOne:
    """One script, one prefix, so the KLU under test is the KLU that ships.

    The CMakeLists auto-build (``_bngsim_autobuild_klu``) is gated
    ``if(WIN32) ... not supported``, so this leg has to build SuiteSparse
    itself -- and the moment there are two ways to build it on Windows, they
    drift, which is what ``test_suitesparse_build_scripts.py`` already pins for
    the ``.sh``/``.ps1`` pair.
    """

    def test_the_leg_uses_the_shipped_build_script(self):
        assert "ci/build_suitesparse.ps1" in _body(), (
            "windows-klu.yml no longer builds SuiteSparse with ci/build_suitesparse.ps1. "
            "A second recipe on the same platform is the drift test_suitesparse_build_"
            "scripts.py exists to prevent, one layer up."
        )

    def test_the_prefix_matches_the_wheel_build(self):
        """``-Prefix`` here and cibuildwheel's ``before-all`` are the same path,
        and so are the ``SUITESPARSE_ROOT`` values that point the bngsim build
        at it. A leg that installed SuiteSparse somewhere else would still pass
        every assertion above while testing a differently-built KLU.
        """
        tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        windows = pyproject["tool"]["cibuildwheel"]["windows"]
        prefix = re.search(r"-Prefix\s+(\S+)", windows["before-all"])
        assert prefix, "cibuildwheel's Windows before-all no longer passes -Prefix"
        body = _body()
        assert f"-Prefix {prefix.group(1)}" in body, (
            f"windows-klu.yml does not build SuiteSparse into {prefix.group(1)}, which is "
            "where the wheel build puts it."
        )
        assert f"SUITESPARSE_ROOT={windows['environment']['SUITESPARSE_ROOT']}" in body, (
            "windows-klu.yml points the bngsim build at a different SUITESPARSE_ROOT than "
            "the wheel does."
        )
