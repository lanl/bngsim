"""GH #166 — codegen compiles must not leak orphaned backend processes.

A compiler driver (``cc``/``clang``/``cl``) execs a backend (``clang -cc1``,
MSVC ``c1``/``c2``) as a separate process. ``subprocess.run(..., timeout=...)``
SIGKILLs only the immediate driver on timeout, so the backend is reparented to
PID 1 and keeps compiling — pegging a core for tens of minutes on a genome-scale
source. ``_run_compile`` launches each compile in its own session / process
group and tears the whole group down on timeout or abort.

These tests use a synthetic "driver" shell script that spawns a long-lived
grandchild (standing in for ``clang -cc1``). The acceptance criterion from the
issue: after the compile call returns/raises, no descendant survives.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest
from bngsim import _codegen as cg

# The orphan-leak failure mode and its fix are POSIX process-group semantics;
# the Windows path uses a different mechanism (taskkill /T) not exercised here.
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="process-group reaping is POSIX-specific"
)


def _spawn_grandchild_script(tmp_path, pidfile, sleep_s: float = 300.0):
    """A 'driver' that launches a long-lived grandchild (records its PID) and
    then blocks — mirroring ``cc`` waiting on its ``clang -cc1`` backend."""
    script = tmp_path / "fake_cc.sh"
    script.write_text(
        "#!/bin/sh\n"
        f"sleep {sleep_s} &\n"
        f'echo $! > "{pidfile}"\n'
        "wait\n"
    )
    script.chmod(0o755)
    return ["/bin/sh", str(script)]


def _pid_alive(pid: int) -> bool:
    """Signal 0 probes liveness without affecting the process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else (shouldn't happen here)
    return True


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    """Poll until ``pid`` disappears (the kill + reap is asynchronous)."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.02)
    return not _pid_alive(pid)


def _read_pid(pidfile, timeout: float = 5.0) -> int:
    """The grandchild writes its PID asynchronously; wait for the file."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            text = pidfile.read_text().strip()
        except FileNotFoundError:
            text = ""
        if text:
            return int(text)
        time.sleep(0.02)
    raise AssertionError("grandchild never recorded its PID")


# ─── Timeout reaps the backend grandchild ─────────────────────────────


def test_run_compile_timeout_reaps_grandchild(tmp_path):
    """On timeout, _run_compile kills the whole process group — the grandchild
    (the stand-in for ``clang -cc1``) must not survive the call."""
    pidfile = tmp_path / "grandchild.pid"
    cmd = _spawn_grandchild_script(tmp_path, pidfile)

    with pytest.raises(subprocess.TimeoutExpired):
        cg._run_compile(cmd, timeout=0.5)

    grandchild = _read_pid(pidfile)
    assert _wait_dead(grandchild), (
        f"grandchild PID {grandchild} survived a timed-out compile "
        "(orphaned backend leak, GH #166)"
    )


def test_run_compile_abort_reaps_grandchild(tmp_path):
    """A KeyboardInterrupt mid-compile must tear the group down too, not just a
    timeout — start_new_session detaches the children from the terminal's
    foreground group, so our explicit kill is the only reaper."""
    pidfile = tmp_path / "grandchild.pid"

    # A driver that aborts via a raised KeyboardInterrupt while communicate()
    # is blocked: monkeypatch Popen.communicate to raise once the grandchild
    # is up, exercising the `except BaseException` cleanup path.
    cmd = _spawn_grandchild_script(tmp_path, pidfile)

    real_communicate = subprocess.Popen.communicate
    state = {"raised": False}

    def fake_communicate(self, *args, **kwargs):
        # First call (the timed wait) → simulate an abort; later reap call runs.
        if not state["raised"]:
            state["raised"] = True
            _read_pid(pidfile)  # ensure the grandchild exists before we bail
            raise KeyboardInterrupt
        return real_communicate(self, *args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(subprocess.Popen, "communicate", fake_communicate):
        with pytest.raises(KeyboardInterrupt):
            cg._run_compile(cmd, timeout=30.0)

    grandchild = _read_pid(pidfile)
    assert _wait_dead(grandchild), (
        f"grandchild PID {grandchild} survived an aborted compile (GH #166)"
    )


# ─── Normal CompletedProcess semantics preserved ──────────────────────


def test_run_compile_success_returns_completed_process(tmp_path):
    """The happy path is a drop-in for subprocess.run: CompletedProcess with
    returncode 0 and captured stdout."""
    result = cg._run_compile(["/bin/sh", "-c", "printf hello"], timeout=10.0)
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
    assert result.stdout == "hello"


def test_run_compile_failure_surfaces_returncode_and_stderr(tmp_path):
    """A non-zero exit is returned (not raised) with stderr captured, matching
    how the callers test ``result.returncode`` / ``result.stderr``."""
    result = cg._run_compile(
        ["/bin/sh", "-c", "printf boom >&2; exit 3"], timeout=10.0
    )
    assert result.returncode == 3
    assert "boom" in result.stderr
