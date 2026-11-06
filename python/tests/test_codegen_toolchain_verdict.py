"""Issue #1544: a C toolchain that cannot build any codegen library.

On macOS with the version 27 SDK, conda-forge's ld64-956.6 cannot read the
SDK's ``libSystem.tbd`` (it lists an ``arm64e.x1`` target), so every codegen link
failed on ``_bzero``, whatever the model. Auto-codegen fell back to the
interpreter with a warning per model, and every ``codegen=True`` construction
and every SSA propensity build compiled again to fail again: a 23-hour benchmark
pass built no library, and one fit failed 7,676 compiles in 43 minutes.

Now a failed compile is followed by a probe: the same command on a trivial
source. When that fails too, the toolchain is broken, the verdict is logged
once, and later builds fail at once on it. ``BNGSIM_CODEGEN_STRICT`` turns the
automatic fallback into an error.

The broken toolchain here is a ``/bin/sh`` compiler that fails the way #1544's
did, so these run on any POSIX machine. Where it must build something, it hands
the command to the machine's own compiler.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
from pathlib import Path

import bngsim
import bngsim._codegen as cg
import bngsim._codegen_toolchain as tc
import bngsim._simulator as simmod
import pytest
from bngsim._bngsim_core import ModelBuilder

N_AUTO = 300  # above the default 256-species auto-codegen threshold

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="POSIX-specific: the /bin/sh fake compiler has no Windows equivalent"
)


def _real_cc() -> str | None:
    """The machine's own gcc/clang, for the fake compiler to hand a call to."""
    cc = os.environ.get("CC")
    if cc and shutil.which(cc):
        return shutil.which(cc)
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _compiler_present() -> bool:
    """Whether the compiler codegen would use is there (MSVC's cl on Windows)."""
    try:
        return shutil.which(cg._find_c_compiler()[0]) is not None
    except Exception:
        return False


REAL_CC = _real_cc()
requires_cc = pytest.mark.skipif(not _compiler_present(), reason="no C compiler available")
# The fake compiler hands calls to gcc/clang, so it needs one; on Windows the
# POSIX-only mark beside it is the reason these do not run.
requires_posix_cc = pytest.mark.skipif(
    os.name != "nt" and REAL_CC is None, reason="no C compiler available"
)

_FAKE_CC = r"""#!/bin/sh
# A compiler for issue #1544's tests: logs each call, then fails as #1544's
# links did, unless FAKECC_MODE says to hand the call to the real compiler.
kind=model
for a in "$@"; do
  case "$a" in
    -print-prog-name=*) echo "query" >> "$FAKECC_LOG"; echo "fakecc: no linker" >&2; exit 1 ;;
    *.c) if grep -q bngsim_toolchain_probe "$a" 2>/dev/null; then kind=probe; fi ;;
  esac
done
echo "$kind ${SDKROOT:-}" >> "$FAKECC_LOG"
delegate() { unset SDKROOT; exec "$REAL_CC" "$@"; }
case "$FAKECC_MODE" in
  model) if [ "$kind" = probe ]; then delegate "$@"; fi ;;
  sdk) case "${SDKROOT:-}" in *Good.sdk) delegate "$@" ;; esac ;;
  slowprobe) if [ "$kind" = probe ]; then sleep 5; fi ;;
esac
echo "ld: warning: ignoring file ${SDKROOT:-}/usr/lib/libSystem.tbd, malformed file" >&2
echo "Undefined symbols for architecture arm64:" >&2
echo '  "_bzero", referenced from:' >&2
echo "ld: symbol(s) not found for architecture arm64" >&2
exit 1
"""


class FakeCC:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str):
        self.path = tmp_path / "fakecc"
        self.path.write_text(_FAKE_CC)
        self.path.chmod(self.path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = tmp_path / "fakecc.log"
        self.log.write_text("")
        monkeypatch.setenv("CC", str(self.path))
        monkeypatch.setenv("FAKECC_LOG", str(self.log))
        monkeypatch.setenv("FAKECC_MODE", mode)
        monkeypatch.setenv("REAL_CC", REAL_CC or "/nonexistent/cc")

    def calls(self, kind: str | None = None) -> list[str]:
        lines = [ln.split(" ", 1)[0] for ln in self.log.read_text().splitlines() if ln]
        lines = [k for k in lines if k != "query"]
        return [k for k in lines if kind is None or k == kind]


@pytest.fixture
def clean(monkeypatch, tmp_path):
    """A fresh cache, no verdicts, no failure memo, and the cc backend chosen."""
    monkeypatch.setattr(tc, "_verdicts", {})
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(simmod, "_AUTO_CODEGEN_FAILURES", {})
    monkeypatch.setattr(simmod, "_AUTO_CODEGEN_SKIP_LOGGED", set())
    # Off unless a test turns it on: the SDK suggestion probes every installed
    # SDK on a real Mac, which would add calls to the fake compiler's log.
    monkeypatch.setattr(tc, "_is_macos", lambda: False)
    for var in (
        "BNGSIM_CODEGEN_JIT",
        "BNGSIM_NO_CODEGEN",
        "BNGSIM_CODEGEN_THRESHOLD",
        "BNGSIM_CODEGEN_STRICT",
        "BNGSIM_CODEGEN_JOBS",
        "BNGSIM_CODEGEN_CHUNK",
        "BNGSIM_SSA_NO_CODEGEN",
        "BNGSIM_SSA_PROP_CC",
        "BNGSIM_SSA_PROP_JIT",
        "SDKROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _chain(n: int) -> bngsim.Model:
    b = ModelBuilder()
    for i in range(n):
        b.add_parameter(f"k{i}", 0.1 + 0.001 * i, "", False)
        b.add_species(f"S{i}", 1.0, False, 1.0)
    for i in range(n - 1):
        b.add_reaction([i], [i + 1], "elementary", f"k{i}", 1.0, True)
    return bngsim.Model(_core=b.build())


def _verdict_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "cannot build a codegen library" in r.getMessage()
    ]


def _other_codegen_warnings(caplog) -> list[str]:
    needles = ("Model codegen failed", "Skipping automatic codegen", "SSA propensity codegen")
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and any(n in r.getMessage() for n in needles)
    ]


@posix_only
def test_codegen_true_names_the_toolchain_and_does_not_compile_again(clean, monkeypatch, caplog):
    """Three models with codegen=True: the first compiles once and probes once,
    and is told the toolchain is to blame; the other two fail at once."""
    cc = FakeCC(clean, monkeypatch, "broken")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        for i, n in enumerate((5, 6, 7)):
            with pytest.raises(bngsim.CodegenToolchainError) as err:
                bngsim.Simulator(_chain(n), method="ode", codegen=True)
            assert "codegen=True requested" in str(err.value)
            assert "not the model's" in str(err.value)
            assert str(cc.path) in str(err.value)
            assert "_bzero" in str(err.value)
            if i == 0:
                assert cc.calls() == ["model", "probe"]
    assert cc.calls() == ["model", "probe"]
    assert len(_verdict_warnings(caplog)) == 1
    assert _other_codegen_warnings(caplog) == []


@posix_only
def test_auto_codegen_falls_back_with_one_warning_and_one_compile(clean, monkeypatch, caplog):
    """Two large models, each constructed twice: one compile and one probe in
    all, one warning, and every run on the interpreter. Once the verdict is in,
    no model's source is even generated."""
    cc = FakeCC(clean, monkeypatch, "broken")
    generated = []
    real_generate = cg.generate_combined_from_model

    def counting_generate(model, **kwargs):
        generated.append(model.n_species)
        return real_generate(model, **kwargs)

    monkeypatch.setattr(cg, "generate_combined_from_model", counting_generate)
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        for n in (N_AUTO, N_AUTO + 1):
            for _ in range(2):
                sim = bngsim.Simulator(_chain(n), method="ode")
                assert sim.codegen_backend == "exprtk"
    assert cc.calls() == ["model", "probe"]
    assert generated == [N_AUTO]
    assert len(_verdict_warnings(caplog)) == 1
    assert _other_codegen_warnings(caplog) == []


@posix_only
@requires_posix_cc
def test_auto_codegen_tries_again_once_the_environment_is_fixed(clean, monkeypatch):
    """The verdict is the toolchain's, not the model's: once SDKROOT names an
    SDK that links, the same model compiles on its next construction."""
    cc = FakeCC(clean, monkeypatch, "sdk")
    monkeypatch.setenv("SDKROOT", str(clean / "Bad.sdk"))
    m = _chain(N_AUTO)
    assert bngsim.Simulator(m, method="ode").codegen_backend == "exprtk"
    assert cc.calls() == ["model", "probe"]
    monkeypatch.setenv("SDKROOT", str(clean / "Good.sdk"))
    assert bngsim.Simulator(m, method="ode").codegen_backend == "cc"


@posix_only
def test_strict_makes_the_automatic_fallback_an_error(clean, monkeypatch):
    """BNGSIM_CODEGEN_STRICT: the build that reaches the verdict raises, and so
    does the next model, refused on the verdict alone."""
    cc = FakeCC(clean, monkeypatch, "broken")
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    for n in (N_AUTO, N_AUTO + 1):
        with pytest.raises(bngsim.CodegenToolchainError, match="BNGSIM_CODEGEN_STRICT"):
            bngsim.Simulator(_chain(n), method="ode")
    assert cc.calls() == ["model", "probe"]


@posix_only
def test_strict_off_values_leave_the_fallback(clean, monkeypatch):
    """Control. ``0`` and an empty value are off, as the SSA flags read them."""
    FakeCC(clean, monkeypatch, "broken")
    for value in ("0", ""):
        monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", value)
        assert bngsim.Simulator(_chain(N_AUTO), method="ode").codegen_backend == "exprtk"


def _failing_prepare(monkeypatch, calls):
    def fail(model):
        calls.append(model)
        cg._record_codegen_error(RuntimeError("Codegen compilation timed out after 600 s"))
        return None

    monkeypatch.setattr(cg, "prepare_model_codegen", fail)
    monkeypatch.setattr(cg, "prepare_model_codegen_source", fail)


def test_strict_raises_on_a_model_failure_and_on_its_memo(clean, monkeypatch):
    """A failure of the model's own build (not the toolchain's) raises under
    strict, both when it happens and when the failure memo would skip it."""
    calls: list = []
    _failing_prepare(monkeypatch, calls)
    m = _chain(N_AUTO)
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    with pytest.raises(RuntimeError, match="BNGSIM_CODEGEN_STRICT.*timed out") as err:
        bngsim.Simulator(m, method="ode")
    assert not isinstance(err.value, bngsim.CodegenToolchainError)
    monkeypatch.delenv("BNGSIM_CODEGEN_STRICT")
    assert bngsim.Simulator(m, method="ode").codegen_backend == "exprtk"  # remembered
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    with pytest.raises(RuntimeError, match="BNGSIM_CODEGEN_STRICT.*its build failed at"):
        bngsim.Simulator(m, method="ode")
    assert len(calls) == 2  # the memo's skip still built nothing


def test_strict_leaves_a_decline_alone(clean, monkeypatch):
    """Control. A decline is a statement about the model, not a failed build:
    the model runs on the interpreter under strict too."""

    def decline(model):
        cg._record_codegen_error(None)
        cg._record_codegen_decline("declined for the test")
        return None

    monkeypatch.setattr(cg, "prepare_model_codegen", decline)
    monkeypatch.setattr(cg, "prepare_model_codegen_source", decline)
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    assert bngsim.Simulator(_chain(N_AUTO), method="ode").codegen_backend == "exprtk"


@posix_only
def test_ssa_propensity_library_is_not_rebuilt_on_a_broken_toolchain(clean, monkeypatch, caplog):
    """The SSA propensity library compiled again on every construction of the
    same model; after the verdict it is not compiled at all."""
    cc = FakeCC(clean, monkeypatch, "broken")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        sim = bngsim.Simulator(_chain(8), method="ssa")
        assert sim._ssa_propensity_lib == ""
        assert cc.calls() == ["model", "probe"]
        for _ in range(2):
            bngsim.Simulator(_chain(8), method="ssa")
    assert cc.calls() == ["model", "probe"]
    assert len(_verdict_warnings(caplog)) == 1
    assert _other_codegen_warnings(caplog) == []


@posix_only
def test_strict_ssa_propensity_failure_raises(clean, monkeypatch):
    FakeCC(clean, monkeypatch, "broken")
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    for _ in range(2):  # the build that reaches the verdict, then one refused on it
        with pytest.raises(bngsim.CodegenToolchainError, match="SSA propensity library"):
            bngsim.Simulator(_chain(8), method="ssa")


def test_strict_ssa_propensity_emit_failure_raises(clean, monkeypatch):
    """The emit step's failure, which falls back to the interpreted propensities
    unasked, raises under strict too."""
    import bngsim._bngsim_core as core

    def boom(_core):
        raise RuntimeError("emit failed for the test")

    monkeypatch.setattr(core, "emit_ssa_propensity_source_structure", boom)
    assert bngsim.Simulator(_chain(8), method="ssa")._ssa_propensity_lib == ""
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    with pytest.raises(RuntimeError, match="BNGSIM_CODEGEN_STRICT.*emit failed for the test"):
        bngsim.Simulator(_chain(8), method="ssa")


@posix_only
@requires_posix_cc
def test_a_model_failure_is_not_blamed_on_the_toolchain(clean, monkeypatch, caplog):
    """The probe builds, so the model's own source is what failed: the error is
    the compiler's, every model is still compiled, and the probe runs once."""
    cc = FakeCC(clean, monkeypatch, "model")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        for n in (5, 6):
            with pytest.raises(RuntimeError, match="Codegen compilation failed") as err:
                bngsim.Simulator(_chain(n), method="ode", codegen=True)
            assert not isinstance(err.value, bngsim.CodegenToolchainError)
    assert cc.calls() == ["model", "probe", "model"]
    assert _verdict_warnings(caplog) == []


@posix_only
def test_a_probe_that_proves_nothing_keeps_no_verdict(clean, monkeypatch):
    """Control. A probe that runs out of time says nothing about the toolchain:
    the model's error stands, and the next build is attempted."""
    cc = FakeCC(clean, monkeypatch, "slowprobe")
    monkeypatch.setattr(tc, "_PROBE_TIMEOUT", 0.5)
    for n in (5, 6):
        with pytest.raises(RuntimeError, match="Codegen compilation failed") as err:
            bngsim.Simulator(_chain(n), method="ode", codegen=True)
        assert not isinstance(err.value, bngsim.CodegenToolchainError)
    assert cc.calls() == ["model", "probe", "model", "probe"]


@posix_only
@requires_posix_cc
def test_a_cached_library_is_still_used(clean, monkeypatch):
    """Control. A library built before the toolchain broke is served from the
    cache, verdict or not, and nothing is compiled."""
    warm = bngsim.Simulator(_chain(5), method="ode", codegen=True)
    assert warm.codegen_backend == "cc"
    cc = FakeCC(clean, monkeypatch, "broken")
    with pytest.raises(bngsim.CodegenToolchainError):
        bngsim.Simulator(_chain(6), method="ode", codegen=True)
    assert cc.calls() == ["model", "probe"]
    sim = bngsim.Simulator(_chain(5), method="ode", codegen=True)
    assert sim.codegen_backend == "cc"
    assert cc.calls() == ["model", "probe"]


@posix_only
@requires_posix_cc
def test_the_verdict_names_the_sdk_and_an_sdk_that_links(clean, monkeypatch):
    """On macOS the verdict names the SDK and tries the other installed ones;
    setting the one it names gets a fresh attempt, which builds."""
    cc = FakeCC(clean, monkeypatch, "sdk")
    bad = clean / "SDKs" / "MacOSX99.0.sdk"
    other = clean / "SDKs" / "MacOSX98.sdk"
    good = clean / "SDKs" / "Good.sdk"
    for d in (bad, other, good):
        d.mkdir(parents=True)
    monkeypatch.setenv("SDKROOT", str(bad))
    monkeypatch.setattr(tc, "_is_macos", lambda: True)
    monkeypatch.setattr(tc, "_macos_sdk_candidates", lambda current: [str(other), str(good)])
    with pytest.raises(bngsim.CodegenToolchainError) as err:
        bngsim.Simulator(_chain(5), method="ode", codegen=True)
    text = str(err.value)
    assert f"SDK:      {bad} (SDKROOT)" in text
    assert f"SDKROOT={good} builds it" in text
    assert "linker:" not in text  # the fake answers no linker, and none is made up
    # The model, the probe, then one probe per candidate until one builds.
    assert [ln.split(" ", 1) for ln in cc.log.read_text().splitlines() if ln != "query"] == [
        ["model", str(bad)],
        ["probe", str(bad)],
        ["probe", str(other)],
        ["probe", str(good)],
    ]
    monkeypatch.setenv("SDKROOT", str(good))
    assert bngsim.Simulator(_chain(5), method="ode", codegen=True).codegen_backend == "cc"


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX-specific: the SDK links are symlinks, which Windows restricts"
)
def test_sdk_candidates_newest_first_one_name_each(tmp_path):
    """The candidates leave out the SDK in use and the unversioned link, give
    each SDK once under its shortest name, and put the newest first."""
    root = tmp_path / "SDKs"
    root.mkdir()
    for name in ("MacOSX27.0.sdk", "MacOSX26.5.sdk", "MacOSX14.4.sdk"):
        (root / name).mkdir()
    (root / "MacOSX.sdk").symlink_to("MacOSX27.0.sdk")
    (root / "MacOSX27.sdk").symlink_to("MacOSX27.0.sdk")
    (root / "MacOSX26.sdk").symlink_to("MacOSX26.5.sdk")
    (root / "MacOSX13.sdk").symlink_to("MacOSX13.3.sdk")  # dangling
    got = tc._macos_sdk_candidates(str(root / "MacOSX.sdk"), roots=[str(root)])
    assert got == [str(root / "MacOSX26.sdk"), str(root / "MacOSX14.4.sdk")]


@posix_only
def test_a_missing_compiler_is_a_verdict_said_once(clean, monkeypatch, caplog):
    monkeypatch.setenv("CC", str(clean / "no-such-cc"))
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        for n in (5, 6):
            with pytest.raises(bngsim.CodegenToolchainError, match="C compiler not found"):
                bngsim.Simulator(_chain(n), method="ode", codegen=True)
        assert bngsim.Simulator(_chain(N_AUTO), method="ode").codegen_backend == "exprtk"
    found = [r for r in caplog.records if "C compiler not found" in r.getMessage()]
    assert len(found) == 1
    assert _other_codegen_warnings(caplog) == []


@posix_only
def test_a_sharded_build_reaches_the_verdict(clean, monkeypatch):
    """The parallel path's compile and link failures are put to the probe too."""
    cc = FakeCC(clean, monkeypatch, "broken")
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "4")
    monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "2")
    m = _chain(40)
    src, _ = cg.generate_combined_from_model(m)
    assert cg._split_sharded_source(src) is not None
    with pytest.raises(bngsim.CodegenToolchainError):
        cg.compile_rhs(src, "shardedhash")
    assert cc.calls("probe") == ["probe"]
    n_before = len(cc.calls())
    with pytest.raises(bngsim.CodegenToolchainError):
        cg.compile_rhs(src, "shardedhash2")
    assert len(cc.calls()) == n_before


def test_the_probe_includes_every_header_generated_code_does():
    """A sysroot that has lost its headers fails every model on ``<math.h>``
    (an Xcode update that removed the SDK a build was configured with did that
    here); the probe has to fail on it too, or that failure is blamed on the
    model and compiled again for every one."""
    emitted = set(
        re.findall(r'_emit\("#include <([^>]+)>"\)', Path(cg.__file__).read_text("utf-8"))
    )
    assert emitted, "no emitted #include found; the pattern above has gone stale"
    probed = set(re.findall(r"#include <([^>]+)>", tc._PROBE_SOURCE))
    assert emitted <= probed


@requires_cc
def test_the_probe_builds_with_this_machines_compiler(clean):
    """The probe source has to build wherever a model does, MSVC included."""
    compiler = cg._find_c_compiler()
    opt = cg._resolve_opt_flag(Path(compiler[0]).stem.lower(), 0)
    assert tc._probe(compiler, opt) == (True, "")


def test_the_sdk_comes_from_sdkroot_or_xcrun(monkeypatch):
    """The SDK line reads SDKROOT when it is set, and xcrun otherwise."""
    monkeypatch.setenv("SDKROOT", "/some/MacOSX1.sdk")
    assert tc._current_sdk() == ("/some/MacOSX1.sdk", "SDKROOT")
    monkeypatch.delenv("SDKROOT")
    monkeypatch.setattr(
        tc, "_run", lambda cmd, **kw: "/x/MacOSX2.sdk\n" if cmd[0] == "xcrun" else ""
    )
    assert tc._current_sdk() == ("/x/MacOSX2.sdk", "xcrun --show-sdk-path")
