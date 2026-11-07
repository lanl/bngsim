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
import signal
import subprocess
import sys
import textwrap
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
    script.write_text(f'#!/bin/sh\nsleep {sleep_s} &\necho $! > "{pidfile}"\nwait\n')
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

    with (
        mock.patch.object(subprocess.Popen, "communicate", fake_communicate),
        pytest.raises(KeyboardInterrupt),
    ):
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
    result = cg._run_compile(["/bin/sh", "-c", "printf boom >&2; exit 3"], timeout=10.0)
    assert result.returncode == 3
    assert "boom" in result.stderr


# ─── Issue #1552: a compile does not outlive its caller ───────────────
#
# The compile runs in a session of its own, so only Python could kill it, and a
# caller killed by a signal (a harness's SIGTERM at its wall cap, a SIGKILL)
# ran none of that. The compile ran on to the end, wrote a library nobody
# installed, and left its .c behind. It now runs under a supervisor that kills
# it, and removes those files, when the caller's end of a pipe closes.


def _a_caller(code: str, env: dict | None = None) -> subprocess.Popen:
    """A Python process that compiles, to be killed while it does."""
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**os.environ, **(env or {})},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM], ids=["SIGKILL", "SIGTERM"])
def test_a_killed_caller_takes_its_compile_and_files_with_it(tmp_path, sig):
    pidfile = tmp_path / "grandchild.pid"
    source, output = tmp_path / "x.c", tmp_path / "x.partial.so"
    source.write_text("int x;\n")
    script = tmp_path / "fake_cc.sh"
    script.write_text('#!/bin/sh\necho partial > "$2"\nsleep 300 &\necho $! > "$1"\nwait\n')
    script.chmod(0o755)
    caller = _a_caller(
        f"""
        from pathlib import Path
        import bngsim._codegen as cg
        cg._run_compile(
            ["/bin/sh", {str(script)!r}, {str(pidfile)!r}, {str(output)!r}],
            abandon=(Path({str(source)!r}), Path({str(output)!r})),
        )
        """
    )
    try:
        grandchild = _read_pid(pidfile, timeout=60.0)
    finally:
        caller.send_signal(sig)
        caller.wait(timeout=30)
    assert _wait_dead(grandchild), f"the compile (PID {grandchild}) outlived its caller"
    deadline = time.perf_counter() + 5.0
    while (source.exists() or output.exists()) and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert not source.exists(), "the compile's source was left behind"
    assert not output.exists(), "the compile's partial library was left behind"


def test_compile_rhs_leaves_nothing_in_the_cache_when_its_caller_is_killed(tmp_path):
    """End to end, through compile_rhs and the cache's own census: a stand-in
    compiler writes its output, starts a backend, and waits, and the caller is
    sent the SIGTERM a harness sends at its wall cap."""
    from bngsim import cache as cache_mod

    cache = tmp_path / "codegen"
    pidfile = tmp_path / "backend.pid"
    script = tmp_path / "slow_cc.sh"
    script.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            out=""
            while [ $# -gt 0 ]; do
                if [ "$1" = "-o" ]; then out="$2"; shift; fi
                shift
            done
            echo partial > "$out"
            sleep 300 &
            echo $! > "$BNGSIM_TEST_BACKEND_PID"
            wait
            """
        )
    )
    script.chmod(0o755)
    caller = _a_caller(
        """
        import bngsim._codegen as cg
        cg.compile_rhs("int bngsim_rhs(void) { return 0; }\\n", "0123456789abcdef")
        """,
        env={
            "BNGSIM_CODEGEN_CACHE_DIR": str(cache),
            "CC": str(script),
            "BNGSIM_TEST_BACKEND_PID": str(pidfile),
        },
    )
    try:
        backend = _read_pid(pidfile, timeout=60.0)
        assert any(p.suffix == ".c" for p in cache.iterdir()), "the compile is in flight"
    finally:
        caller.send_signal(signal.SIGTERM)
        caller.wait(timeout=30)
    assert _wait_dead(backend), f"the compiler's backend (PID {backend}) outlived its caller"
    deadline = time.perf_counter() + 5.0
    while list(cache.iterdir()) and time.perf_counter() < deadline:
        time.sleep(0.02)
    info = cache_mod.codegen_cache_info(cache)
    assert info.entries == (), [e.path.name for e in info.entries]


def test_a_compiler_that_cannot_be_started_raises_as_popen_did(tmp_path):
    """compile_rhs reads a FileNotFoundError as a missing toolchain."""
    with pytest.raises(FileNotFoundError):
        cg._run_compile([str(tmp_path / "no-such-cc"), "-v"], timeout=10.0)
    not_executable = tmp_path / "cc"
    not_executable.write_text("")
    with pytest.raises(PermissionError):
        cg._run_compile([str(not_executable)], timeout=10.0)


def test_a_compiler_killed_by_a_signal_reports_it(tmp_path):
    """The signal that ended the compiler is the supervisor's too, so the caller
    sees the returncode it would have seen from the compiler itself."""
    result = cg._run_compile(["/bin/sh", "-c", "kill -TERM $$"], timeout=10.0)
    assert result.returncode == -signal.SIGTERM


def test_without_an_interpreter_to_supervise_it_still_reaps_on_timeout(tmp_path, monkeypatch):
    """A frozen or embedded Python has no interpreter to run the supervisor
    with, and compiles as before."""
    monkeypatch.setattr(cg, "_compile_supervisor_python", lambda: None)
    pidfile = tmp_path / "grandchild.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        cg._run_compile(_spawn_grandchild_script(tmp_path, pidfile), timeout=0.5)
    assert _wait_dead(_read_pid(pidfile))
    assert cg._run_compile(["/bin/sh", "-c", "printf ok"], timeout=10.0).stdout == "ok"


def test_a_caller_with_many_descriptors_open_still_compiles(tmp_path):
    """The supervisor waits on the caller's pipe with select(), which takes no
    descriptor past FD_SETSIZE (1024), and the pipe keeps the caller's number.
    A caller with that many files open must still get its compile."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1200:
        pytest.skip(f"the descriptor limit ({hard}) is below FD_SETSIZE")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 1200), hard))
    held = []
    try:
        while not held or held[-1] < 1100:
            held.append(os.open(os.devnull, os.O_RDONLY))
        result = cg._run_compile(["/bin/sh", "-c", "printf ok"], timeout=30.0)
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert (result.returncode, result.stdout) == (0, "ok")
