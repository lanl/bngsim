"""The wheel's KLU smoke test is the thing under test.

Issue #303. The cibuildwheel ``test-command`` was
``assert bngsim.HAS_KLU`` — a **compile-time** flag, True because the extension
was *built* against SuiteSparse and unmoved by whether the libraries that ended
up inside the wheel work. That is a thin check anywhere; it is a thin check with
teeth here, because every wheel is *repaired* and repair rewrites the linkage
(delvewheel content-hashes the DLL names and injects a loader patch, delocate
rewrites install names to ``@loader_path``, auditwheel mangles SONAMEs and
patches RPATH). None of that is exercised by a source build, so
``windows-klu.yml`` (#296) does not cover it.

``ci/wheel_smoke_klu.py`` replaces it. This file guards it two ways:

* **Functionally.** The script is executable Python with no test dependencies,
  so a local KLU build can just run it. That is a real assertion, not a text
  one — unlike the workflow-parsing coverage tests, this can fail because the
  code is wrong rather than because a config drifted.
* **Textually.** The script only protects the wheels if the wheels actually run
  it. ``test-command`` lives in ``pyproject.toml``, which no runtime test reads,
  and reverting it to a one-line ``python -c`` import check would be invisible.

Deliberately NOT asserted: that the script passes on a build without KLU. It
exits 1 there by design — that is the failure it exists to report.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import bngsim
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "ci" / "wheel_smoke_klu.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _cibuildwheel_test_command() -> str:
    """The global ``[tool.cibuildwheel] test-command``.

    Read with tomllib rather than by regex so a per-platform override cannot be
    mistaken for the global one; the caller supplies the 3.10 skip.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return config["tool"]["cibuildwheel"]["test-command"]


class TestTheSmokeScriptWorks:
    @pytest.mark.skipif(not bngsim.HAS_KLU, reason="KLU not compiled")
    def test_it_passes_on_a_klu_build(self):
        """The functional half, and the reason this file is not text-only.

        Run from a directory that is not the repo root, the way cibuildwheel
        runs it: from ``{project}`` the script would find bngsim's source tree
        on ``sys.path`` and could pass against something other than the
        installed artifact.
        """
        proc = subprocess.run(
            [sys.executable, str(SMOKE)],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"ci/wheel_smoke_klu.py failed on a build with HAS_KLU=True.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        assert "OK:" in proc.stdout, f"unexpected output:\n{proc.stdout}"

    def test_it_needs_nothing_the_wheel_does_not_already_have(self):
        """No ``test-requires`` backs this command, so every import in the
        script has to come from bngsim's own runtime dependencies. numpy is one;
        pytest, scipy and antimony are not, and reaching for one would make the
        wheel legs error on collection instead of testing the wheel.
        """
        allowed = {"bngsim", "numpy", "sys", "os", "pathlib", "subprocess", "__future__"}
        source = SMOKE.read_text(encoding="utf-8")
        imported = {
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and not line.startswith("from .")
        }
        assert imported <= allowed, (
            f"ci/wheel_smoke_klu.py imports {sorted(imported - allowed)}, which the "
            "cibuildwheel test venv does not install (there is no test-requires). "
            "Add it to [project] dependencies, add a test-requires, or do without."
        )


class TestTheWheelsActuallyRunIt:
    def test_test_command_invokes_the_smoke_script(self):
        """A script nothing runs protects nothing.

        The specific regression: reverting ``test-command`` to the one-line
        ``python -c "import bngsim; ... assert bngsim.HAS_KLU"`` it used to be.
        That reads as a reasonable simplification, passes every other test in
        the tree, and puts issue #303 back.
        """
        command = _cibuildwheel_test_command()
        assert "ci/wheel_smoke_klu.py" in command, (
            f"[tool.cibuildwheel] test-command is {command!r}, which does not run "
            "ci/wheel_smoke_klu.py. The wheels are then back to asserting only that "
            "HAS_KLU is True, which is a compile-time flag (issue #303)."
        )
        assert "{project}" in command, (
            f"test-command is {command!r}: the script path must go through "
            "cibuildwheel's {project} substitution, since the command runs from a "
            "temporary directory and on Linux from inside the manylinux container."
        )

    def test_the_smoke_script_is_shipped_where_the_command_looks_for_it(self):
        """``{project}/ci/...`` has to exist in the tree cibuildwheel mounts,
        and ``ci/`` must not be excluded from the sdist — an sdist-built wheel
        runs the same command."""
        assert SMOKE.is_file(), f"{SMOKE} is missing but test-command runs it"
        tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
        config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        excluded = config["tool"]["scikit-build"].get("sdist", {}).get("exclude", [])
        assert not any(entry.strip("/") == "ci" for entry in excluded), (
            "sdist.exclude drops ci/, so an sdist-built wheel would run a "
            "test-command whose script is not there."
        )

    def test_the_wheels_leg_fires_when_the_script_changes(self):
        """``ci/**`` is already in wheels.yml's paths filter; assert it, because
        the trigger is what makes editing the script measurable at all (#167).
        """
        workflow = REPO_ROOT / ".github" / "workflows" / "wheels.yml"
        if not workflow.is_file():
            pytest.skip(".github/ not in this checkout")
        body = workflow.read_text(encoding="utf-8")
        assert '"ci/**"' in body or "- ci/**" in body, (
            "wheels.yml no longer triggers on ci/**, so a change to "
            "ci/wheel_smoke_klu.py would not run the job that uses it."
        )
