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

on_posix = pytest.mark.skipif(os.name == "nt", reason="CcJit is POSIX-only")
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
        pytest.skip("needs an Apple Silicon Mac")
    if not os.path.exists("/Library/Apple/usr/libexec/oah/libRosettaRuntime"):
        pytest.skip("needs Rosetta 2")
    cxx, clang, sdk = _xcrun("-f", "clang++"), _xcrun("-f", "clang"), _xcrun("--show-sdk-path")
    if not (cxx and clang and sdk):
        pytest.skip("needs Apple's clang and an SDK")
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
