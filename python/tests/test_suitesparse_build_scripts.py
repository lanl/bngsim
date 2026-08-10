"""The two SuiteSparse/KLU build scripts are one recipe written twice.

``ci/build_suitesparse.sh`` (macOS + the ``BNGSIM_KLU_AUTOBUILD`` sdist fallback)
and ``ci/build_suitesparse.ps1`` (Windows wheels) build the same pinned KLU
subset for the same consumer — SUNDIALS' KLU TPL — from two files that share no
code. That is the classic paired-site shape: when one is edited and the other is
not, nothing fails, the platforms just quietly stop shipping the same library.

The specific drift this file exists to catch is issue #178. Both scripts have to
take ``SuiteSparseBLAS.cmake``'s user-supplied-BLAS early return by defining
``BLAS_LIBRARIES``, because that module ends in ``find_package(BLAS REQUIRED)``
and *nothing* in the five libraries built here links a BLAS. Without the opt-out
the configure dies inside ``SuiteSparse_config`` before KLU is reached on any
host with no BLAS, which — with ``BNGSIM_REQUIRE_KLU=ON`` set for every
scikit-build-core build — turned a source ``pip install`` on a stock Linux box
into a hard build error. The flag looks unused (CMake even reports it as such,
since SuiteSparse only *reads* it when ``BLAS_FOUND``), so it is exactly the kind
of line a tidy-up deletes; on macOS and on a runner with a system BLAS, deleting
it changes nothing visible.

These are text assertions over build scripts, which is all a Python test can be
here: actually running either script needs a network clone and minutes of
compilation. The functional proof is CI — ``python-tests.yml`` installs no
SuiteSparse on either leg, so both take the from-source autobuild and assert
``HAS_KLU``. The last test below is what keeps that proof from being deleted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SH = REPO_ROOT / "ci" / "build_suitesparse.sh"
PS1 = REPO_ROOT / "ci" / "build_suitesparse.ps1"
PYTHON_TESTS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "python-tests.yml"

# Flags that decide *what library comes out*, so both platforms must pass them.
# SUITESPARSE_USE_FORTRAN is deliberately absent: it exists only to keep a
# dev-box gfortran out of the macOS dylibs and has no Windows counterpart.
SHARED_CMAKE_DEFINES = {
    "SUITESPARSE_ENABLE_PROJECTS": "suitesparse_config;amd;colamd;btf;klu",
    "KLU_USE_CHOLMOD": "OFF",
    "BUILD_SHARED_LIBS": "ON",
    "BUILD_STATIC_LIBS": "OFF",
    "SUITESPARSE_USE_OPENMP": "OFF",
    "SUITESPARSE_USE_CUDA": "OFF",
    "SUITESPARSE_DEMOS": "OFF",
    # Issue #178. Empty on purpose: DEFINED-ness is the whole signal, and an
    # empty value means anything that did try to link it gets nothing.
    "BLAS_LIBRARIES": "",
}

_DEFINE = re.compile(r"""-D([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|[^\s"'`\\]*)""")

pytestmark = pytest.mark.skipif(
    not (SH.exists() and PS1.exists()),
    reason="ci/ build scripts are not in this checkout (installed package)",
)


def _cmake_defines(path: Path) -> dict[str, str]:
    """Collect ``-DNAME=VALUE`` from a script, ignoring comment lines.

    Comment-stripping is line-based (``#`` first on the line) rather than a real
    shell/PowerShell parse: both files keep their rationale in whole-line
    comments, and a token like ``-DFOO=BAR`` quoted inside prose would otherwise
    register as a flag that is not actually passed.
    """
    defines: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        for name, value in _DEFINE.findall(line):
            defines[name] = value.strip('"')
    return defines


@pytest.mark.parametrize("script", [SH, PS1], ids=["sh", "ps1"])
def test_blas_probe_is_opted_out(script: Path):
    """Neither script may let SuiteSparse_config run its BLAS probe (#178)."""
    defines = _cmake_defines(script)
    assert "BLAS_LIBRARIES" in defines, (
        f"{script.relative_to(REPO_ROOT)} no longer defines BLAS_LIBRARIES. That "
        "flag is what skips SuiteSparse_config's find_package(BLAS REQUIRED) "
        "(issue #178); without it this script cannot configure on a host with no "
        "BLAS, which is every stock Linux box doing a source pip install. CMake "
        "reports it as an unused variable — that is expected, SuiteSparse only "
        "reads it when BLAS_FOUND, the branch being skipped."
    )
    assert defines["BLAS_LIBRARIES"] == "", (
        "BLAS_LIBRARIES must stay empty: nothing in the KLU subset links a BLAS, "
        "so naming a real library here would add a dependency the built libs do "
        f"not need. Found {defines['BLAS_LIBRARIES']!r}."
    )


@pytest.mark.parametrize(("name", "value"), sorted(SHARED_CMAKE_DEFINES.items()))
def test_both_scripts_agree_on_what_gets_built(name: str, value: str):
    """A flag changed on one platform and not the other ships two libraries."""
    sh, ps1 = _cmake_defines(SH), _cmake_defines(PS1)
    assert sh.get(name) == value, (
        f"ci/build_suitesparse.sh: -D{name} is {sh.get(name)!r}, expected {value!r}"
    )
    assert ps1.get(name) == value, (
        f"ci/build_suitesparse.ps1: -D{name} is {ps1.get(name)!r}, expected {value!r}"
    )


def test_both_scripts_pin_the_same_suitesparse():
    """The tag *and* the asserted commit, so a bump cannot land on one platform."""
    sh_text = SH.read_text(encoding="utf-8")
    ps1_text = PS1.read_text(encoding="utf-8")

    sh_version = re.search(r'^SS_VERSION="([^"]+)"', sh_text, re.M)
    sh_commit = re.search(r'^SS_COMMIT="([^"]+)"', sh_text, re.M)
    ps1_version = re.search(r'^\$SsVersion\s*=\s*"([^"]+)"', ps1_text, re.M)
    ps1_commit = re.search(r'^\$SsCommit\s*=\s*"([^"]+)"', ps1_text, re.M)
    assert sh_version and sh_commit and ps1_version and ps1_commit, (
        "could not find the SuiteSparse tag/commit pins in both scripts — if they "
        "were renamed, update this test rather than dropping the check"
    )

    assert sh_version.group(1) == ps1_version.group(1), (
        f"SuiteSparse tag differs: sh={sh_version.group(1)} ps1={ps1_version.group(1)}"
    )
    assert sh_commit.group(1) == ps1_commit.group(1), (
        f"SuiteSparse commit differs: sh={sh_commit.group(1)} ps1={ps1_commit.group(1)}"
    )


@pytest.mark.skipif(
    not PYTHON_TESTS_WORKFLOW.exists(),
    reason=".github/workflows is not in this checkout (installed package)",
)
def test_python_tests_workflow_provisions_no_suitesparse():
    """This job's *lack* of a SuiteSparse install is the #178 acceptance proof.

    Re-adding one would make the job green again while the bug is back, which is
    how #178 survived in the first place: the Linux leg installed
    ``libsuitesparse-dev`` as a workaround and nothing then exercised the
    autobuild on Linux. System-SuiteSparse discovery stays covered by
    cibuildwheel's Linux leg (``dnf install suitesparse-devel``).
    """
    # Comment lines dropped first: the workflow's header explains at length why
    # the apt step is gone, naming the package it used to install.
    body = "\n".join(
        line
        for line in PYTHON_TESTS_WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    for package in ("libsuitesparse-dev", "suitesparse-devel", "suite-sparse"):
        assert package not in body, (
            f"python-tests.yml installs {package}. Both legs must run on a host "
            "with no SuiteSparse so BNGSIM_KLU_AUTOBUILD is exercised — that is "
            "the standing proof for issue #178. A red build here means the "
            "from-source fallback regressed; fix the fallback, not the job."
        )
