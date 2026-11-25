"""The ``BNGSIM_SSA_PROP_CC=1`` propensity library is built for this process's
machine and cached under a name that carries it (issue #1927).

``bngsim::CcJit`` (``include/bngsim/cc_jit.hpp``) ran ``cc -O3 -shared -fPIC``
with no ``-arch`` and cached the result as ``ssa_prop_<fnv1a>.so``. Under
Rosetta 2, Apple's ``cc`` built arm64 for an x86_64 process, which ``dlopen``
refused, and the SSA fell back to interpreted propensities: the ablation measured
nothing. A native and a Rosetta process sharing a cache named one library alike,
so whichever compiled first won; and the name held no format version, so a
library an older bngsim built was reused. #1759 fixed the same for the Python
codegen; this is the one compile C++ runs itself.

Oracles: the library's own architecture as ``lipo`` reads it, the propensity
backend the run reports, and, under Rosetta, a small x86_64 driver of the header
that must load and call what it built.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import bngsim
import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER_DIR = ROOT / "include"

NET = """begin parameters
    1 k 0.01
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1,1 2 k
end reactions
begin groups
    1 Aobs 1
end groups
"""

on_posix = pytest.mark.skipif(os.name == "nt", reason="POSIX-specific: CcJit has no Windows build")
requires_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler available")
on_macos = pytest.mark.skipif(
    platform.system() != "Darwin" or shutil.which("lipo") is None,
    reason="macOS only: Apple's -arch and lipo",
)


def _libraries(cache: Path) -> list[str]:
    return sorted(p.name for p in cache.iterdir() if p.name.startswith("ssa_prop_"))


@pytest.fixture
def cc_run(tmp_path, monkeypatch):
    """Run the SSA with the cc propensity ablation into a fresh cache, with HOME
    elsewhere so a write to the old fixed location would show."""
    for flag in ("BNGSIM_SSA_NO_CODEGEN", "BNGSIM_SSA_PROP_JIT", "CC"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("BNGSIM_SSA_PROP_CC", "1")
    cache, home = tmp_path / "cache", tmp_path / "home"
    cache.mkdir()
    home.mkdir()
    monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", str(cache))
    monkeypatch.setenv("HOME", str(home))
    net = tmp_path / "dimer.net"
    net.write_text(NET)

    def run():
        model = bngsim.Model.from_net(str(net))
        return bngsim.Simulator(model, method="ssa").run(t_span=(0.0, 1.0), n_points=3, seed=1)

    return run, cache, home


@on_posix
@requires_cc
def test_the_library_is_named_for_this_processs_machine_and_format(cc_run):
    run, cache, home = cc_run
    result = run()
    assert result.ssa_diagnostics["propensity_backend"] == "cc"
    (name,) = _libraries(cache)
    assert name.startswith(f"ssa_prop_{platform.machine()}_v2_")
    assert name.endswith(".so")
    assert not (home / ".cache").exists(), "CcJit ignored BNGSIM_CODEGEN_CACHE_DIR"
    # A second run is a cache hit on the same file.
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert _libraries(cache) == [name]


@on_posix
@requires_cc
def test_another_compiler_gets_its_own_library(cc_run, monkeypatch):
    """The compile command, CC included, is in the name's hash: a library one
    compiler built is not taken for another's."""
    run, cache, _ = cc_run
    run()
    monkeypatch.setenv("CC", "cc -Wall")
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(cache)) == 2


@on_posix
@requires_cc
@on_macos
def test_the_library_holds_this_processs_machine(cc_run):
    run, cache, _ = cc_run
    run()
    (name,) = _libraries(cache)
    out = subprocess.run(["lipo", "-archs", str(cache / name)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [platform.machine()]


# ── The header itself, built as an x86_64 process under Rosetta ─────────────

_DRIVER = r"""
#include "bngsim/cc_jit.hpp"
#include <cstdio>
int main() {
    const std::string src = "double twice(double x) { return 2.0 * x; }\n";
    try {
        bngsim::CcJit jit(src, "ssa_prop");
        auto twice = jit.symbol<double (*)(double)>("twice");
        std::printf("%s %s %g\n", bngsim::CcJit::machine(), jit.so_path().c_str(), twice(3.0));
        return 0;
    } catch (const std::exception &e) {
        std::printf("%s THROW %s\n", bngsim::CcJit::machine(), e.what());
        return 1;
    }
}
"""


def _xcrun(*args: str) -> str | None:
    try:
        out = subprocess.run(["xcrun", *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


@pytest.fixture
def rosetta_toolchain():
    """``(c++, clang, sdk)`` to build and run an x86_64 driver on an Apple
    Silicon Mac with Rosetta, or a skip. The driver calls the real ``clang`` (an
    arm64 binary that runs from a translated process) because the ``/usr/bin/cc``
    shim loads an arm64-only libxcrun from the Command Line Tools."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("needs Rosetta 2 on an Apple Silicon Mac")
    if not os.path.exists("/Library/Apple/usr/libexec/oah/libRosettaRuntime"):
        pytest.skip("needs Rosetta 2")
    cxx, clang, sdk = _xcrun("-f", "clang++"), _xcrun("-f", "clang"), _xcrun("--show-sdk-path")
    if not (cxx and clang and sdk):
        pytest.skip("needs Rosetta 2 and Apple's clang and an SDK")
    return cxx, clang, sdk


@on_macos
def test_under_rosetta_the_header_builds_x86_64_and_keeps_arm64_apart(rosetta_toolchain, tmp_path):
    """The issue's case: an x86_64 process builds, loads and calls an x86_64
    library, and a native arm64 process sharing the cache builds its own."""
    cxx, clang, sdk = rosetta_toolchain
    driver = tmp_path / "driver.cpp"
    driver.write_text(_DRIVER)
    cache = tmp_path / "cache"
    cache.mkdir()
    env = {**os.environ, "CC": clang, "SDKROOT": sdk, "BNGSIM_CODEGEN_CACHE_DIR": str(cache)}
    seen = {}
    for machine in ("x86_64", "arm64"):
        exe = tmp_path / f"driver_{machine}"
        subprocess.run(
            [cxx, "-std=c++17", "-arch", machine, f"-I{HEADER_DIR}", str(driver), "-o", str(exe)],
            check=True, capture_output=True, env=env, timeout=300,
        )  # fmt: skip
        out = subprocess.run([str(exe)], capture_output=True, text=True, env=env, timeout=300)
        assert out.returncode == 0, out.stdout + out.stderr
        reported, so, value = out.stdout.split()
        assert (reported, value) == (machine, "6")
        assert Path(so).name.startswith(f"ssa_prop_{machine}_v2_")
        seen[machine] = so
    for machine, so in seen.items():
        out = subprocess.run(["lipo", "-archs", so], capture_output=True, text=True, check=True)
        assert out.stdout.split() == [machine]
    assert len(_libraries(cache)) == 2


@on_posix
@requires_cc
def test_the_threads_of_one_run_replicates_all_run_cc_on_a_cold_cache(cc_run, tmp_path):
    """The temps were named by process alone, so on a cold cache one thread
    removed the source another was compiling, and about half the replicates fell
    back to interpreted propensities (review of #1927)."""
    model = bngsim.Model.from_net(str(tmp_path / "dimer.net"))
    sim = bngsim.Simulator(model, method="ssa")
    runs = sim.run_replicates(16, t_span=(0.0, 1.0), n_points=3, seed=1, num_processors=8)
    assert [r.ssa_diagnostics["propensity_backend"] for r in runs] == ["cc"] * 16
    _, cache, _ = cc_run
    assert len(_libraries(cache)) == 1


@on_posix
@requires_cc
def test_a_cache_dir_with_a_quote_still_compiles(cc_run, tmp_path, monkeypatch):
    run, _, _ = cc_run
    cache = tmp_path / "it's a cache"
    cache.mkdir()
    monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", str(cache))
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(cache)) == 1


@on_posix
@requires_cc
def test_a_cache_dir_under_home_is_expanded(cc_run, monkeypatch):
    run, _, home = cc_run
    monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", "  ~/cc-cache  ")
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(home / "cc-cache")) == 1


@on_posix
@requires_cc
def test_a_blank_cc_is_an_unset_one(cc_run, monkeypatch):
    run, cache, _ = cc_run
    run()
    monkeypatch.setenv("CC", "   ")
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(cache)) == 1


@on_posix
@requires_cc
def test_a_relative_cache_dir_starting_with_a_dash_is_made_absolute(cc_run, tmp_path, monkeypatch):
    """``-x`` reached the compiler as an option, not a path (review of #1927)."""
    run, _, _ = cc_run
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", "-x")
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(tmp_path / "-x")) == 1


@on_posix
@requires_cc
def test_a_cc_whose_path_holds_a_blank_is_one_word(cc_run, tmp_path, monkeypatch):
    run, cache, _ = cc_run
    tools = tmp_path / "my tools"
    tools.mkdir()
    (tools / "cc").symlink_to(shutil.which("cc"))
    monkeypatch.setenv("CC", str(tools / "cc"))
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(cache)) == 1


# ── The Python codegen reads the variable and the home the same way ─────────


def _python_cache_dir(env: dict[str, str]) -> str:
    code = "import bngsim._codegen as c; print(c.CACHE_DIR)"
    base = {k: v for k, v in os.environ.items() if k not in ("HOME", "BNGSIM_CODEGEN_CACHE_DIR")}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**base, **env},
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@on_posix
def test_an_unknown_user_in_the_cache_dir_does_not_break_import(tmp_path):
    """``Path.expanduser()`` raised for ``~nosuchuser``, and ``import bngsim`` with
    it; :func:`os.path.expanduser` leaves the path as written, as CcJit does."""
    got = _python_cache_dir(
        {"HOME": str(tmp_path), "BNGSIM_CODEGEN_CACHE_DIR": "~nosuchuser-bngsim/c"}
    )
    assert got.endswith("/~nosuchuser-bngsim/c") and os.path.isabs(got)


@on_posix
def test_an_empty_home_is_the_password_databases(tmp_path):
    """``Path.home()`` is ``/`` for an empty HOME, which put the cache at the
    read-only ``/.cache``."""
    import pwd

    got = _python_cache_dir({"HOME": ""})
    assert got == f"{pwd.getpwuid(os.getuid()).pw_dir}/.cache/bngsim/codegen"
    got = _python_cache_dir({"HOME": str(tmp_path), "BNGSIM_CODEGEN_CACHE_DIR": " ~/c "})
    assert got == f"{tmp_path}/c"


@on_posix
@requires_cc
@pytest.mark.parametrize("dirname", ["back\\slash", "a dir"])
def test_a_cc_path_the_shell_would_read_is_one_word(cc_run, tmp_path, monkeypatch, dirname):
    run, cache, _ = cc_run
    tools = tmp_path / dirname
    tools.mkdir()
    (tools / "cc").symlink_to(shutil.which("cc"))
    monkeypatch.setenv("CC", str(tools / "cc"))
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(cache)) == 1


@on_posix
@requires_cc
def test_a_cc_on_path_whose_name_holds_a_blank_is_one_word(cc_run, tmp_path, monkeypatch):
    run, cache, _ = cc_run
    tools = tmp_path / "bin"
    tools.mkdir()
    # A wrapper, not a symlink: Apple's /usr/bin/cc picks its tool by the name it
    # is run as.
    wrapper = tools / "my cc"
    wrapper.write_text(f'#!/bin/sh\nexec "{shutil.which("cc")}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CC", "my cc")
    assert run().ssa_diagnostics["propensity_backend"] == "cc"
    assert len(_libraries(cache)) == 1


_NO_PASSWD_ENTRY = """
import pwd
def _no_entry(uid):
    raise KeyError(uid)
pwd.getpwuid = _no_entry
import bngsim._codegen as c
print(c.CACHE_DIR)
"""


@on_posix
@pytest.mark.parametrize("home", [None, ""])
def test_no_home_and_no_password_entry_is_tmp(home):
    """A container run as a UID with no password entry and no HOME failed
    ``import bngsim`` (``Path.home()`` raised), or with an empty HOME put the
    cache at ``/.cache``; CcJit falls back to ``/tmp``, and so does Python."""
    env = {k: v for k, v in os.environ.items() if k not in ("HOME", "BNGSIM_CODEGEN_CACHE_DIR")}
    if home is not None:
        env["HOME"] = home
    out = subprocess.run(
        [sys.executable, "-c", _NO_PASSWD_ENTRY],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "/tmp/.cache/bngsim/codegen"


@on_posix
def test_a_relative_cache_dir_in_a_removed_working_directory_still_imports(tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    code = (
        "import os, sys; os.chdir(sys.argv[1]); os.rmdir(sys.argv[1]); "
        "import bngsim._codegen as c; print(c.CACHE_DIR)"
    )
    env = {**os.environ, "BNGSIM_CODEGEN_CACHE_DIR": "relcache"}
    out = subprocess.run(
        [sys.executable, "-c", code, str(gone)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "relcache"


def test_the_cache_command_reads_its_directory_as_the_variable_is(tmp_path, monkeypatch):
    """``bngsim.cache -C '~user/x'`` raised for an unknown user (review of #1927)."""
    from bngsim.cache import codegen_cache_dir

    monkeypatch.chdir(tmp_path)
    got = codegen_cache_dir("~nosuchuser-bngsim/x")
    assert (
        got == tmp_path.resolve() / "~nosuchuser-bngsim" / "x"
        or got == tmp_path / "~nosuchuser-bngsim" / "x"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    assert codegen_cache_dir("~/x") == tmp_path / "x"
