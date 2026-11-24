"""Issue #1486 — the codegen cache key names the architecture.

The cache names each compiled RHS ``rhs_<key>_<hash>``, and the key was
``_CODEGEN_VERSION`` and a digest of the emitters, nothing about the machine. Two
builds of one bngsim on two architectures that share a cache directory (a native
arm64 Python and an x86_64 one under Rosetta on one Mac, or the x86-64 and
aarch64 nodes of a cluster on one home directory) therefore named the same
model's library alike, and whichever ran second loaded the other's: "dlopen
failed ... incompatible architecture (have 'arm64', need 'x86_64')".

The key's last field is now the architecture of the running interpreter, so the
two name their libraries apart. ``bngsim.cache`` reads an artifact that differs
from this install's key in that field alone as the other architecture's build of
the same source, not as an orphan (test_cache_cli.py), so an orphan sweep from
one does not take the other's cache.

Issue #1759 — the library is built for that architecture. Apple's ``cc`` builds
for the hardware it runs on, not for the process that starts it, so an x86_64
Python under Rosetta compiled arm64 under its x86-64 name and failed at dlopen.
Every compile and link on macOS now carries ``-arch`` with the interpreter's
machine. The command tests below force Darwin and a machine, so they run on any
OS; the ones that build force the other machine on a real Mac and read what was
built with ``lipo`` and ``nm``, and the last one loads it into an x86_64 Python
under Rosetta and calls it. ``test_codegen_toolchain_verdict.py`` has the
refusal of a library a compiler built for the wrong machine all the same.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import bngsim
import bngsim._codegen as cg
import bngsim._codegen_toolchain as tc
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _codegen_backend import under_jit_no_cc_artifact  # noqa: E402

NET = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 1.0
    2 B() 0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 A_tot 1
end groups
"""


@pytest.mark.parametrize(
    "machine, tag",
    [
        ("arm64", "arm64"),
        ("x86_64", "x86-64"),
        ("aarch64", "aarch64"),
        ("AMD64", "amd64"),
        ("ARM64", "arm64"),
        ("i686", "i686"),
        ("ppc64le", "ppc64le"),
        ("", "unknown"),
    ],
)
def test_the_tag_is_a_key_field_safe_spelling_of_the_machine(machine, tag):
    assert cg._codegen_arch_tag(machine) == tag
    assert cg._artifact_key_field(f"1+0+{tag}") == f"1+0+{tag}", "no escaping needed"


def test_the_key_ends_with_this_processs_architecture():
    import platform

    assert cg._codegen_arch_tag(platform.machine()) == cg._CODEGEN_ARCH
    assert cg._codegen_cache_key_for(cg._CODEGEN_ARCH) == cg._CODEGEN_CACHE_KEY
    assert cg._CODEGEN_CACHE_KEY.rsplit("+", 1)[1] == cg._CODEGEN_ARCH
    assert cg._artifact_stem("0123456789abcdef").startswith(f"rhs_{cg._CODEGEN_CACHE_KEY}_")


def _other_arch() -> str:
    return "x86-64" if cg._CODEGEN_ARCH != "x86-64" else "arm64"


def test_two_architectures_compute_two_keys():
    """One source, two machines: the same version and digest, two keys."""
    mine, theirs = cg._CODEGEN_CACHE_KEY, cg._codegen_cache_key_for(_other_arch())
    assert mine != theirs
    assert mine.rsplit("+", 1)[0] == theirs.rsplit("+", 1)[0]


def _run(path) -> float:
    with contextlib.redirect_stderr(io.StringIO()):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=True)
        result = sim.run(t_span=(0.0, 1.0), n_points=3)
    return float(np.asarray(result.species)[-1, 0])


def _libraries(cache) -> set:
    suffix = cg._shared_lib_suffix()
    return {p for p in cache.iterdir() if p.name.endswith(suffix)}


@under_jit_no_cc_artifact
def test_another_architectures_library_is_never_loaded(tmp_path, monkeypatch):
    """The issue's reproducer, on one machine: a run on the other architecture
    leaves its library in a shared cache, and a run here must build its own
    rather than dlopen that one. The other architecture's library is planted by
    running under its key and then overwriting what was built with bytes no
    loader takes, standing in for a binary of another architecture."""
    cache = tmp_path / "codegen"
    monkeypatch.setattr(cg, "CACHE_DIR", cache)
    net = tmp_path / "m.net"
    net.write_text(NET)

    with monkeypatch.context() as m:
        m.setattr(cg, "_CODEGEN_CACHE_KEY", cg._codegen_cache_key_for(_other_arch()))
        _run(net)
    (planted,) = _libraries(cache)
    assert f"+{_other_arch()}_" in planted.name
    planted.write_bytes(b"built for another architecture; loading this must never happen\n")

    a1 = _run(net)

    assert a1 == pytest.approx(np.exp(-0.5), rel=1e-6)
    (built,) = _libraries(cache) - {planted}
    assert f"+{cg._CODEGEN_ARCH}_" in built.name
    assert planted.read_bytes().startswith(b"built for another architecture")


# ── Issue #1759: the library is built for this process's machine ─────────────


def _recording_compiler(monkeypatch) -> list[list[str]]:
    """Stand in for every compiler call: record the command and write the file
    its ``-o`` names, so a sharded build runs to its end with nothing compiled."""
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, timeout=None, env=None, abandon=()):
        calls.append(list(cmd))
        out = Path(cmd[cmd.index("-o") + 1])
        (Path(cwd) / out if cwd is not None else out).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cg, "_run_compile", fake_run)
    return calls


def _arch_words(cmd: list[str]) -> list[str]:
    """Every value an ``-arch`` in ``cmd`` names."""
    return [cmd[i + 1] for i, word in enumerate(cmd) if word == "-arch"]


def _all_three_commands(monkeypatch, tmp_path, compiler) -> dict[str, list[str]]:
    """The serial compile, a shard's object compile and the shard link, as
    ``compiler`` would be run on this (patched) platform."""
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    serial = cg._build_compile_cmd(tmp_path / "m.c", tmp_path / "m.dylib", "-O2", compiler)
    calls = _recording_compiler(monkeypatch)
    cg._compile_sharded(
        "int d(void){return 0;}", ["int u(void){return 1;}"], tmp_path / "s.dylib",
        "-O2", 2, None, compiler,
    )  # fmt: skip
    objects = [c for c in calls if "-c" in c]
    (link,) = [c for c in calls if "-c" not in c]
    assert len(objects) == 2
    return {"compile": serial, "object": objects[0], "link": link}


@pytest.mark.parametrize("machine", ["arm64", "x86_64"])
def test_every_macos_compile_and_link_asks_for_the_interpreters_machine(
    machine, monkeypatch, tmp_path
):
    """The issue's fix: ``-arch`` with the machine this process runs as, once, on
    the serial compile and on both halves of a sharded build. An object compiled
    for one machine does not link into a library for another, so a sharded build
    needs it on every step."""
    monkeypatch.setattr(cg.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", machine, raising=False)
    for step, cmd in _all_three_commands(monkeypatch, tmp_path, ["cc"]).items():
        assert _arch_words(cmd) == [machine], (step, cmd)
        assert cmd[:3] == ["cc", "-arch", machine], (step, cmd)


def test_cc_arguments_come_before_the_architecture(monkeypatch, tmp_path):
    """``CC``'s own words stay first, as the compiler and its arguments, and an
    ``-arch`` it names is kept: clang builds one library holding both."""
    monkeypatch.setattr(cg.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", "arm64", raising=False)
    compiler = ["ccache", "cc", "-arch", "x86_64"]
    for step, cmd in _all_three_commands(monkeypatch, tmp_path, compiler).items():
        assert cmd[:4] == compiler, (step, cmd)
        assert _arch_words(cmd) == ["x86_64", "arm64"], (step, cmd)


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_other_platforms_name_no_architecture(system, monkeypatch, tmp_path):
    """Linux's and MinGW's gcc build for their own machine and do not know
    ``-arch``; GCC on Linux refuses it."""
    monkeypatch.setattr(cg.platform, "system", lambda: system)
    for step, cmd in _all_three_commands(monkeypatch, tmp_path, ["gcc"]).items():
        assert "-arch" not in cmd, (step, cmd)


def test_msvc_names_no_architecture(monkeypatch, tmp_path):
    monkeypatch.setattr(cg.platform, "system", lambda: "Windows")
    cmd = cg._build_compile_cmd(tmp_path / "m.c", tmp_path / "m.dll", "/O2", ["cl"])
    assert cmd[:2] == ["cl", "/O2"] and "-arch" not in cmd


@pytest.mark.parametrize("machine", ["", "i386", "ppc64", "aarch64"])
def test_a_machine_apples_compilers_have_no_name_for_adds_nothing(machine, monkeypatch, tmp_path):
    """What ``-arch`` would not take is left to the compiler, as before."""
    monkeypatch.setattr(cg.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", machine, raising=False)
    assert cg._arch_flags() == []
    for step, cmd in _all_three_commands(monkeypatch, tmp_path, ["cc"]).items():
        assert "-arch" not in cmd, (step, cmd)


def test_the_flag_and_the_key_name_one_machine():
    """The library is named for the machine it is built for: the key's last field
    and the ``-arch`` value come from one reading of ``platform.machine()``."""
    machine = cg._CODEGEN_MACHINE
    assert platform.machine() == machine
    assert cg._codegen_arch_tag(machine) == cg._CODEGEN_ARCH
    if platform.system() == "Darwin" and cg._CODEGEN_MACHINE in ("arm64", "x86_64"):
        assert cg._arch_flags() == ["-arch", cg._CODEGEN_MACHINE]
    else:
        assert cg._arch_flags() == []


def _thin(cpu: int, *, bits: int = 64, order: str = "little") -> bytes:
    magic = 0xFEEDFACF if bits == 64 else 0xFEEDFACE
    return magic.to_bytes(4, order) + cpu.to_bytes(4, order) + bytes(24)  # type: ignore[arg-type]


def _fat(cpus: list[int], *, wide: bool = False) -> bytes:
    head = (0xCAFEBABF if wide else 0xCAFEBABE).to_bytes(4, "big") + len(cpus).to_bytes(4, "big")
    record = 32 if wide else 20
    return head + b"".join(c.to_bytes(4, "big") + bytes(record - 4) for c in cpus)


X86_64, ARM64 = 0x01000007, 0x0100000C


@pytest.mark.parametrize(
    "data, held",
    [
        (_thin(X86_64), {"x86_64"}),
        (_thin(ARM64), {"arm64"}),
        (_thin(7, bits=32), {"i386"}),
        (_thin(18, bits=32, order="big"), {"ppc"}),
        (_fat([X86_64, ARM64]), {"x86_64", "arm64"}),
        (_fat([ARM64], wide=True), {"arm64"}),
        (_thin(0x0100002A), {"cpu type 0x100002a"}),
        (b"\x7fELF\x02\x01\x01" + bytes(57), None),
        (b"MZ" + bytes(62), None),
        (_thin(ARM64)[:7], None),
        (_fat([]), None),
        (_fat([ARM64, X86_64])[:-1], None),
        (_fat([ARM64] * 65), None),
        (_fat([X86_64] * 64), {"x86_64"}),
    ],
    ids=[
        "x86_64", "arm64", "i386", "ppc-big-endian", "universal", "universal-64",
        "unnamed-cpu", "elf", "pe", "short", "no-slices", "cut-table", "too-many-slices",
        "most-slices",
    ],
)  # fmt: skip
def test_the_mach_o_reader(data, held, tmp_path):
    lib = tmp_path / "lib"
    lib.write_bytes(data)
    assert tc.macho_architectures(lib) == (None if held is None else frozenset(held))


def test_the_mach_o_reader_on_a_missing_file(tmp_path):
    assert tc.macho_architectures(tmp_path / "gone") is None


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_nothing_is_foreign_off_macos(system, monkeypatch, tmp_path):
    """Only macOS asks: a Mach-O file elsewhere is no library of this process."""
    lib = tmp_path / "lib"
    lib.write_bytes(_thin(ARM64))
    monkeypatch.setattr(tc.platform, "system", lambda: system)
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", "x86_64")
    assert tc.foreign_library(lib) is None


@pytest.mark.parametrize(
    "machine, data, foreign",
    [
        ("x86_64", _thin(ARM64), {"arm64"}),
        ("arm64", _thin(X86_64), {"x86_64"}),
        ("x86_64", _thin(X86_64), None),
        ("x86_64", _fat([ARM64, X86_64]), None),
        ("x86_64", b"not a library", None),
        ("ppc64", _thin(ARM64), None),
    ],
    ids=["arm64-in-x86_64", "x86_64-in-arm64", "own", "universal", "not-mach-o", "unnamed"],
)
def test_a_library_is_foreign_when_it_holds_no_code_for_this_machine(
    machine, data, foreign, monkeypatch, tmp_path
):
    lib = tmp_path / "lib"
    lib.write_bytes(data)
    monkeypatch.setattr(tc.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", machine)
    assert tc.foreign_library(lib) == (None if foreign is None else frozenset(foreign))


@pytest.mark.parametrize("translated", [False, True])
def test_the_reason_names_both_machines_and_rosetta(translated, monkeypatch):
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", "x86_64")
    monkeypatch.setattr(tc, "_translated", lambda: translated)
    reason = tc.foreign_library_reason(frozenset({"arm64"}))
    assert "-arch x86_64" in reason and "built arm64 code" in reason
    assert "Set CC to a compiler that builds x86_64 code" in reason
    assert ("under Rosetta 2" in reason) is translated
    assert ("native arm64 Python" in reason) is translated


def test_rosetta_is_read_from_the_process_not_the_machine():
    """Under Rosetta only when this process is x86_64 on an Apple Silicon Mac
    (``hw.optional.arm64``, which a translated process reads as 1 too); an Intel
    Mac and another OS have no ``sysctl.proc_translated`` to read."""
    on_apple_silicon = False
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True)
        on_apple_silicon = out.returncode == 0 and out.stdout.strip() == "1"
    assert tc._translated() is (on_apple_silicon and platform.machine() == "x86_64")


# ── On a real Mac: build for the other machine and read what was built ───────

on_macos = pytest.mark.skipif(
    platform.system() != "Darwin" or shutil.which("lipo") is None or shutil.which("nm") is None,
    reason="macOS only: Apple's -arch, lipo and nm (issue #1759); the command tests "
    "run everywhere",
)
requires_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler available")


def _other_machine() -> str:
    """The machine this Mac is not, which its compiler can build for."""
    return "x86_64" if platform.machine() == "arm64" else "arm64"


def _lipo(path) -> set[str]:
    out = subprocess.run(["lipo", "-archs", str(path)], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def _exports(path, machine) -> set[str]:
    out = subprocess.run(
        ["nm", "-arch", machine, "-gU", str(path)], capture_output=True, text=True, check=True
    )
    return {line.split()[-1] for line in out.stdout.splitlines() if line.strip()}


@pytest.fixture
def built_for_the_other_machine(tmp_path, monkeypatch):
    """A codegen library of :data:`NET` that ``compile_rhs`` built, in a fresh
    cache, as a process running as the other machine would build it."""
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(tc, "_verdicts", {})
    monkeypatch.setattr(tc, "_warned", set())
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cache")
    net = tmp_path / "m.net"
    net.write_text(NET)
    src, _ = cg.generate_combined_from_model(bngsim.Model.from_net(str(net)))
    machine = _other_machine()
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", machine, raising=False)
    return machine, cg.compile_rhs(src, "17590000aaaabbbb"), src


@on_macos
@requires_cc
def test_compile_rhs_builds_for_the_interpreters_machine(built_for_the_other_machine):
    """The issue's failure, from this side: what the compile wrote holds the
    machine the process runs as, here forced to the one this Mac is not, and
    nothing else. Two readers that share no code with bngsim say so."""
    machine, so, _ = built_for_the_other_machine
    assert _lipo(so) == {machine}
    assert "_bngsim_codegen_rhs" in _exports(so, machine)
    assert tc.macho_architectures(so) == frozenset({machine})


@on_macos
@requires_cc
def test_a_sharded_build_compiles_and_links_for_the_interpreters_machine(tmp_path, monkeypatch):
    """Objects for one machine and a link for another give a library that holds
    none of their code, so the symbols are read as well as the architecture."""
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    machine = _other_machine()
    monkeypatch.setattr(cg, "_CODEGEN_MACHINE", machine, raising=False)
    out = tmp_path / "sharded.dylib"
    cg._compile_sharded(
        "int bngsim_unit_0(void);\nint bngsim_driver(void) { return bngsim_unit_0(); }\n",
        ["int bngsim_unit_0(void) { return 1759; }\n"],
        out, "-O2", 2, None, cg._find_c_compiler(),
    )  # fmt: skip
    assert _lipo(out) == {machine}
    assert {"_bngsim_driver", "_bngsim_unit_0"} <= _exports(out, machine)


@functools.lru_cache(maxsize=1)
def _x86_64_python() -> list[str] | None:
    """A command that runs Python as x86_64 on this Mac: Apple's universal
    ``/usr/bin/python3`` under Rosetta 2, which only an Apple Silicon Mac with
    Rosetta installed has."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    cmd = ["arch", "-x86_64", "/usr/bin/python3", "-I"]
    try:
        out = subprocess.run(
            [*cmd, "-c", "import platform; print(platform.machine())"],
            capture_output=True, text=True, timeout=60,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return cmd if out.returncode == 0 and out.stdout.strip() == "x86_64" else None


_CALL_RHS = r"""
import ctypes, json, platform, sys
class UD(ctypes.Structure):
    _fields_ = [("param_values", ctypes.POINTER(ctypes.c_double)),
                ("tfun_ctx", ctypes.c_void_p), ("tfun_eval", ctypes.c_void_p)]
try:
    lib = ctypes.CDLL(sys.argv[1])
except OSError as err:
    print(json.dumps({"machine": platform.machine(), "error": str(err)}))
    sys.exit(0)
fn = lib.bngsim_codegen_rhs
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double),
               ctypes.POINTER(ctypes.c_double), ctypes.c_void_p]
k = (ctypes.c_double * 1)(0.5)
y, ydot = (ctypes.c_double * 2)(2.0, 0.0), (ctypes.c_double * 2)()
rc = fn(0.0, y, ydot, ctypes.byref(UD(k, None, None)))
print(json.dumps({"machine": platform.machine(), "rc": rc, "ydot": list(ydot)}))
"""


def _call_in(python: list[str], so) -> dict:
    out = subprocess.run(
        [*python, "-c", _CALL_RHS, str(so)], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@on_macos
@requires_cc
@pytest.mark.skipif(
    _x86_64_python() is None,
    reason="needs Rosetta 2 and Apple's universal /usr/bin/python3 for an x86_64 Python "
    "to load the library in (issue #1759)",
)
def test_an_x86_64_python_under_rosetta_loads_and_calls_the_library(built_for_the_other_machine):
    """End to end, the issue's case: a library ``compile_rhs`` built for x86_64
    on an Apple Silicon Mac loads into an x86_64 Python under Rosetta, and its
    right-hand side is ``dA/dt = -k A`` at ``k = 0.5, A = 2``."""
    machine, so, _ = built_for_the_other_machine
    assert machine == "x86_64"
    python = _x86_64_python()
    assert python is not None

    called = _call_in(python, so)
    assert called == {"machine": "x86_64", "rc": 0, "ydot": [-1.0, 1.0]}


_COMPILE_HERE = r"""
import subprocess, sys
subprocess.run(sys.argv[1:], check=True)
"""


@on_macos
@requires_cc
@pytest.mark.skipif(
    _x86_64_python() is None,
    reason="needs Rosetta 2 and Apple's universal /usr/bin/python3 for an x86_64 Python "
    "to compile and load in (issue #1759)",
)
def test_from_inside_rosetta_the_old_command_builds_arm64_and_the_new_one_x86_64(
    built_for_the_other_machine, tmp_path, monkeypatch
):
    """The issue's premise, from a translated process: the compile command
    without ``-arch`` (main's), run by an x86_64 Python under Rosetta, builds
    the Mac's arm64, which that Python cannot load; the command with it builds
    x86_64, which it loads and calls."""
    _, _, source = built_for_the_other_machine
    python = _x86_64_python()
    assert python is not None
    src = tmp_path / "rhs.c"
    src.write_text(source)
    command = cg._build_compile_cmd(src, tmp_path / "new.dylib", "-O2", ["cc"])
    assert command[1:3] == ["-arch", "x86_64"]
    before = [word for i, word in enumerate(command) if i not in (1, 2)]
    before[before.index("-o") + 1] = str(tmp_path / "old.dylib")

    for cmd in (before, command):
        out = subprocess.run(
            [*python, "-c", _COMPILE_HERE, *cmd], capture_output=True, text=True, timeout=300
        )
        assert out.returncode == 0, out.stderr
    assert _lipo(tmp_path / "old.dylib") == {"arm64"}
    assert _lipo(tmp_path / "new.dylib") == {"x86_64"}
    assert "incompatible architecture" in _call_in(python, tmp_path / "old.dylib")["error"]
    assert _call_in(python, tmp_path / "new.dylib")["ydot"] == [-1.0, 1.0]


@on_macos
@pytest.mark.skipif(
    _x86_64_python() is None,
    reason="needs Rosetta 2 and Apple's universal /usr/bin/python3 for an x86_64 Python "
    "to read sysctl.proc_translated in (issue #1759)",
)
def test_an_x86_64_python_under_rosetta_reads_as_translated():
    """The check that puts "under Rosetta 2" in the message, run in such a process
    (the function's own source, since that Python cannot import this bngsim)."""
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(tc._translated))
    script = f"import platform\n{source}\nprint(_translated())\n"
    out = subprocess.run(
        [*_x86_64_python(), "-c", script], capture_output=True, text=True, timeout=60
    )
    assert (out.returncode, out.stdout.strip()) == (0, "True"), out.stderr
