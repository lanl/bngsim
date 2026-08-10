"""The optional BLAS dense solver's CI coverage is itself the thing under test.

Issue #269. The LAPACK dense backend (GH #84) reached a state where every gate
that was supposed to cover it reported green while executing none of it:

- ``tests/test_lapack_dense_linsol.cpp`` early-returned ``0`` from four of its
  six cases when no BLAS was linked, and ``RUN_TEST`` counts ``0`` as a pass, so
  a bare ``ubuntu-latest`` printed ``6/6 passed`` — the same line a host with a
  backend prints. ``ctest`` suppresses stdout on success, so the one tell
  (``lapack_dense_available = no``) never reached the log.
- After #265 no CI job anywhere built a BLAS on Linux, leaving macOS Accelerate
  as the only implementation under test — and Accelerate versus reference
  LAPACK/OpenBLAS is precisely the pair this integration is most likely to
  disagree about (symbol decoration, LP64 vs ILP64).

Both halves were fixed by editing files no runtime test reads, which is how they
would come back. These are text assertions over the C++ test source and the two
workflows, in the shape of ``test_suitesparse_build_scripts.py``: the functional
proof lives in CI, and this file is what keeps the proof from being deleted.

The one thing NOT asserted here is that a BLAS leg exists on any particular
runner — a workflow is free to change runner images. What is asserted is that a
no-BLAS build can never again report itself as fully covered, and that a leg
which claims a backend fails when it does not have one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_TEST = REPO_ROOT / "tests" / "test_lapack_dense_linsol.cpp"
NATIVE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "native-tests.yml"
PYTHON_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "python-tests.yml"

AVAILABILITY_GUARD = "!bngsim::lapack_dense_available()"


def _strip_comments(text: str, marker: str) -> str:
    """Drop whole-line comments, so prose about a flag is not read as the flag.

    Line-based like the #178 test's helper, and for the same reason: both
    workflows and this C++ file carry their rationale in whole-line comments
    that name the exact tokens being asserted on.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(marker))


class TestCppSkipsAreLoud:
    """A skipped C++ case must not be able to look like a passing one."""

    @pytest.mark.skipif(not CPP_TEST.exists(), reason="tests/ not in this checkout")
    def test_no_availability_guard_returns_silently(self):
        """Every ``if (!lapack_dense_available())`` early-out goes through SKIP().

        This is the exact regression: a bare ``return 0`` there is counted as a
        pass by ``RUN_TEST`` and is indistinguishable, in the summary line and
        in the exit code, from the case having run. The one guard allowed to not
        skip is the gate-predicate test, whose no-backend branch opens a block
        (``{``) because it asserts something real about that configuration —
        that opting in with no backend still refuses the BLAS path.
        """
        lines = CPP_TEST.read_text(encoding="utf-8").splitlines()
        offenders = []
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if AVAILABILITY_GUARD not in line or stripped_line.startswith("//"):
                continue
            if stripped_line.endswith("{"):
                continue  # block form: a branch that asserts, not an early-out
            nxt = next(
                (
                    stripped
                    for stripped in (candidate.strip() for candidate in lines[i + 1 :])
                    if stripped
                ),
                "",
            )
            if not (nxt.startswith("SKIP(") or nxt.startswith("{")):
                offenders.append(f"line {i + 1}: guard followed by {nxt!r}")
        assert not offenders, (
            "A no-backend early-out in test_lapack_dense_linsol.cpp does not go "
            "through SKIP(): " + "; ".join(offenders) + ". A bare `return 0` "
            "there is counted as a PASS, which is how this file reported 6/6 on "
            "a host that ran none of the BLAS path (issue #269)."
        )

    @pytest.mark.skipif(not CPP_TEST.exists(), reason="tests/ not in this checkout")
    def test_summary_reports_the_skip_count(self):
        """The summary line has to say SKIPPED, because ctest will not."""
        body = _strip_comments(CPP_TEST.read_text(encoding="utf-8"), "//")
        assert "tests_skipped" in body
        assert '"SKIPPED' in body, (
            "The summary line no longer prints a skip count. ctest hides stdout "
            "on success, so this string is the only place a no-BLAS host says "
            "what it did not run."
        )

    @pytest.mark.skipif(not CPP_TEST.exists(), reason="tests/ not in this checkout")
    def test_skips_do_not_fail_the_binary(self):
        """A host with no BLAS is supported; skipping there must still exit 0.

        Otherwise the stock ``native-tests.yml`` leg — which is the
        configuration the manylinux and Windows wheels ship — goes permanently
        red for a build option it never asked for, and a permanently-red suite
        is worse than a smaller green one (#28/#36).
        """
        body = _strip_comments(CPP_TEST.read_text(encoding="utf-8"), "//")
        assert re.search(r"tests_passed \+ tests_skipped == tests_run", body), (
            "The exit code no longer counts skips as non-failures."
        )


class TestNativeWorkflowRunsTheBackend:
    @pytest.mark.skipif(not NATIVE_WORKFLOW.exists(), reason=".github/ not in this checkout")
    def test_has_a_leg_with_a_real_blas(self):
        body = _strip_comments(NATIVE_WORKFLOW.read_text(encoding="utf-8"), "#")
        assert "liblapack-dev" in body, (
            "native-tests.yml installs no BLAS on any leg, so its "
            "test_lapack_dense_linsol run executes none of the dgetrf path "
            "(issue #269). This is the C++ half of the Linux gap."
        )
        assert 'expect_lapack: "yes"' in body and 'expect_lapack: "no"' in body, (
            "Both C++ configurations must stay in the matrix: no-BLAS is what "
            "the manylinux/Windows wheels ship, with-BLAS is what a source "
            "install on a LAPACK box gets."
        )

    @pytest.mark.skipif(not NATIVE_WORKFLOW.exists(), reason=".github/ not in this checkout")
    def test_guard_asserts_the_backend_and_the_skip_count(self):
        """The leg that installs a BLAS must fail if any case skipped.

        Without this the matrix entry is decoration: a build that resolved no
        LAPACK would run the same five no-ops and report the same green.
        """
        body = _strip_comments(NATIVE_WORKFLOW.read_text(encoding="utf-8"), "#")
        assert "lapack_dense_available" in body, (
            "The guard no longer reads the binary's own availability line."
        )
        assert "SKIPPED" in body, (
            "The guard no longer inspects the skip count, so a leg that lost "
            "its BLAS between configure and run would pass."
        )

    @pytest.mark.skipif(not NATIVE_WORKFLOW.exists(), reason=".github/ not in this checkout")
    def test_klu_stays_off(self):
        """Why adding a BLAS here cannot weaken issue #178's proof.

        That proof is about the SuiteSparse autobuild on a host with no BLAS.
        This job never reaches the autobuild, and this assertion is the reason
        the package could be added here rather than to python-tests.yml's bare
        Linux leg.
        """
        body = _strip_comments(NATIVE_WORKFLOW.read_text(encoding="utf-8"), "#")
        assert "-DBNGSIM_ENABLE_KLU=OFF" in body


class TestPythonWorkflowKeepsBothConfigurations:
    @pytest.mark.skipif(not PYTHON_WORKFLOW.exists(), reason=".github/ not in this checkout")
    def test_bare_linux_leg_survives(self):
        """The #178 proof leg must still install nothing and expect no backend.

        The LAPACK leg is an ADDITION, never a replacement. A leg that installs
        one package is no longer the bare host an sdist install lands on, which
        is the entire reason #265 removed the apt step.
        """
        body = _strip_comments(PYTHON_WORKFLOW.read_text(encoding="utf-8"), "#")
        assert 'expect_lapack: "false"' in body, (
            "No python-tests leg expects a BLAS-free build any more. One Linux "
            "leg must stay bare — see this workflow's header on issue #178."
        )

    @pytest.mark.skipif(not PYTHON_WORKFLOW.exists(), reason=".github/ not in this checkout")
    def test_lapack_leg_exists_and_is_asserted(self):
        body = _strip_comments(PYTHON_WORKFLOW.read_text(encoding="utf-8"), "#")
        assert "liblapack-dev" in body, (
            "No python-tests leg builds the LAPACK dense solver on Linux, so "
            "the seven LAPACK-gated tests run only against macOS Accelerate "
            "(issue #269)."
        )
        assert "HAS_LAPACK_DENSE" in body, (
            "The legs no longer assert which dense backend they got, so one "
            "quietly losing (or gaining) a BLAS would still report green — the "
            "same shape as the HAS_KLU guard next to it."
        )

    @pytest.mark.skipif(not PYTHON_WORKFLOW.exists(), reason=".github/ not in this checkout")
    def test_lapack_leg_installs_only_that_one_package(self):
        """One package, and specifically not SuiteSparse.

        The LAPACK leg still has to take BNGSIM_KLU_AUTOBUILD, or it stops being
        a *second* data point on the default configuration and becomes a
        different job that happens to run the same tests.
        """
        body = _strip_comments(PYTHON_WORKFLOW.read_text(encoding="utf-8"), "#")
        installed = set(re.findall(r"apt: (\S+)", body)) - {'""'}
        assert installed == {"liblapack-dev"}, (
            f"python-tests.yml legs install {sorted(installed)}; only "
            "liblapack-dev is allowed, and only on the leg added for issue #269."
        )
