"""Whether the C toolchain codegen compiles with can build a library at all
(issue #1544).

A codegen compile that fails says nothing, by itself, about whose failure it is.
On macOS with the version 27 SDK, conda-forge's ld64-956.6 cannot parse the
SDK's ``libSystem.tbd`` and ``libm.tbd`` (they list an ``arm64e.x1`` target it
does not know), so it links no library at all, whatever the model. Auto-codegen
then ran every large model on the interpreted RHS with a warning per model, and
every ``codegen=True`` construction paid a full compile to fail again: a 23-hour
benchmark pass built no library, and one fit compiled and failed 7,676 times in
43 minutes.

When a compile fails, the same command is run on a trivial source that includes
what generated code includes and calls what it calls from libc and libm. If that
fails too, the toolchain is broken, not the model. The verdict is kept, keyed on
the compiler and the environment variables that steer it, and a later build
that would compile is refused on it at once (a library already in the cache is
still used). The first verdict is logged once, with the SDK and the linker named
and, on macOS, an installed SDK that links. A probe that passes means the
model's own source is what failed, and the failure is reported as before.

A verdict is good for :data:`_VERDICT_TTL` seconds. After that a broken one is
probed again before it refuses anything, so a toolchain that was broken for a
while (a full disk, an SDK link being swapped by an update) is used again once
it builds; the probe costs a fraction of a second, a model's compile can cost
minutes. A re-probe that fails, or proves nothing (it times out), renews the
verdict.

A toolchain is first found broken only after a real compile has failed, never on
a probe alone, so one that can build a model is not refused because a probe
could not. Once it is, a re-probe decides: that one refuses without a model's
compile, since the toolchain has already failed a real build.

While a verdict holds, :func:`bngsim._codegen.prepare_model_codegen` refuses a
model after its cache lookup and before its source is generated, so a model that
codegen would have declined while generating it is refused as the toolchain's
failure instead.

This module is not one of ``_CODEGEN_SOURCE_MODULES``: nothing here changes the
emitted C, so editing it leaves the artifact cache alone.
"""

from __future__ import annotations

import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from bngsim._exceptions import CodegenToolchainError

logger = logging.getLogger("bngsim")

#: The environment variables that decide what the compiler driver and the linker
#: do, besides ``CC`` (which picks the compiler and is part of the key through
#: it). A verdict is kept per value of all of them, so fixing one in-process
#: (``os.environ["SDKROOT"] = ...``) gets a fresh attempt.
_TOOLCHAIN_ENV = (
    "PATH",
    "SDKROOT",
    "DEVELOPER_DIR",
    "TOOLCHAINS",
    "MACOSX_DEPLOYMENT_TARGET",
    "CONDA_BUILD_SYSROOT",
    "CPATH",
    "C_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "INCLUDE",
    "LIB",
)

#: What generated sources include, and what they need from the C library and
#: libm: an array cleared with ``memset`` (clang lowers it to ``_bzero`` on macOS,
#: the first symbol #1544's links lost) and the elementary functions rate laws
#: call. The headers are what a sysroot that has lost them fails on. Plain C89
#: apart from the export marker, so gcc, clang and MSVC all build it.
_PROBE_SOURCE = """\
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define BNGSIM_PROBE_EXPORT __declspec(dllexport)
#else
#define BNGSIM_PROBE_EXPORT
#endif

BNGSIM_PROBE_EXPORT double bngsim_toolchain_probe(double *y, int n, double p)
{
    double s;
    int i;
    memset(y, 0, (size_t)n * sizeof(double));
    s = exp(p) + log(p + 2.0) + pow(p, 1.5) + sqrt(p + 1.0) + fmod(p, 3.0);
    for (i = 0; i < n; ++i) {
        s += y[i];
    }
    return s;
}
"""

#: Seconds a probe may take. A trivial source builds in well under one; a probe
#: that runs out (a machine loaded with parallel compiles) proves nothing either
#: way, and no verdict is kept.
_PROBE_TIMEOUT = 60.0

#: Seconds a verdict holds before it is probed again.
_VERDICT_TTL = 300.0

#: How many other installed macOS SDKs a broken verdict tries, newest first, to
#: suggest one that links, and how long it may spend on them in all.
_MAX_SDK_CANDIDATES = 6
_SDK_SCAN_BUDGET = 30.0

#: How many lines of the probe's stderr the verdict quotes.
_STDERR_LINES = 12

_lock = threading.Lock()
#: Verdicts by toolchain key, with the time each was reached: the message of a
#: broken toolchain, or ``None`` for one a probe found working.
_verdicts: dict[tuple, tuple[str | None, float]] = {}
#: Keys whose broken verdict has been logged, so it is logged once.
_warned: set[tuple] = set()

_FALLBACK = (
    "Models at or above BNGSIM_CODEGEN_THRESHOLD species run on the interpreted "
    "RHS, and codegen=True and forward sensitivities raise CodegenToolchainError. "
    "Set BNGSIM_CODEGEN_STRICT=1 to make that fallback an error too, or "
    "BNGSIM_NO_CODEGEN=1 to skip codegen."
)


def codegen_strict() -> bool:
    """Whether ``BNGSIM_CODEGEN_STRICT`` asks for an automatic codegen build that
    fails to raise, instead of falling back to the interpreter (issue #1544).
    Unset, empty, ``0``, ``false``, ``no`` and ``off`` are off."""
    value = os.environ.get("BNGSIM_CODEGEN_STRICT", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def toolchain_key(compiler: list[str] | None) -> tuple:
    """The key a verdict is kept under: the compiler command (``None`` when none
    can be named) and the environment variables in :data:`_TOOLCHAIN_ENV`."""
    return (
        tuple(compiler) if compiler is not None else None,
        tuple(os.environ.get(name) for name in _TOOLCHAIN_ENV),
    )


def _keep_broken(key: tuple, message: str) -> bool:
    """Keep a broken verdict (the caller holds the lock); whether it is the first
    for this key, and so the one to log."""
    _verdicts[key] = (message, time.monotonic())
    first = key not in _warned
    _warned.add(key)
    return first


def check() -> tuple[list[str] | None, str | None]:
    """The compiler a build would use now, and the broken-toolchain verdict on it
    (``None`` when there is none). Where no compiler can be named at all (Windows
    with neither ``cl`` nor ``gcc``), the compiler is ``None`` and that is the
    verdict, logged once."""
    from bngsim import _codegen as cg

    try:
        compiler = cg._find_c_compiler()
    except RuntimeError as err:
        message = f"{err} {_FALLBACK}"
        with _lock:
            first = _keep_broken(toolchain_key(None), message)
        if first:
            logger.warning("%s", message)
        return None, message
    return compiler, known_broken(compiler)


def known_broken(compiler: list[str]) -> str | None:
    """The message of the broken-toolchain verdict for ``compiler`` in the current
    environment, or ``None`` when there is none (not probed, or probed working).

    An expired broken verdict is probed again first: it stands, renewed, unless
    the probe now builds."""
    key = toolchain_key(compiler)
    entry = _verdicts.get(key)
    if entry is None or entry[0] is None:
        return None
    if time.monotonic() - entry[1] < _VERDICT_TTL:
        return entry[0]
    with _lock:
        entry = _verdicts.get(key)
        if entry is None or entry[0] is None:
            return None
        if time.monotonic() - entry[1] < _VERDICT_TTL:
            return entry[0]  # another thread renewed it while this one waited
        ok, _ = _probe(compiler)
        if ok:
            del _verdicts[key]
            _warned.discard(key)
            logger.warning("The C toolchain builds a codegen library again; codegen resumes.")
            return None
        _verdicts[key] = (entry[0], time.monotonic())
        return entry[0]


def record_missing(compiler: list[str]) -> str | None:
    """When ``compiler`` is not an executable where ``subprocess`` would look for
    it, keep that as the toolchain's verdict, log it once, and return it;
    ``None`` when it is there."""
    if shutil.which(compiler[0]) is not None:
        return None
    message = (
        f"C compiler not found: {compiler[0]!r} is not an executable on PATH"
        f"{' (from CC)' if os.environ.get('CC') else ''}. Install Xcode Command Line "
        "Tools (macOS), gcc (Linux), or Visual Studio Build Tools (Windows), or set CC. "
        f"{_FALLBACK}"
    )
    with _lock:
        first = _keep_broken(toolchain_key(compiler), message)
    if first:
        logger.warning("%s", message)
    return message


def strict_refusal(what: str, cause: BaseException | str) -> RuntimeError:
    """The error ``BNGSIM_CODEGEN_STRICT`` makes of an automatic codegen build
    that failed (``what`` names it), in place of the fallback to the interpreter.
    A :class:`CodegenToolchainError` when the toolchain is what failed."""
    detail = f"{type(cause).__name__}: {cause}" if isinstance(cause, BaseException) else cause
    message = (
        f"BNGSIM_CODEGEN_STRICT is set, and {what} failed to build ({detail}). Without "
        "it the run falls back to the interpreter; unset it to allow that (issue #1544)."
    )
    if isinstance(cause, CodegenToolchainError):
        return CodegenToolchainError(message)
    return RuntimeError(message)


def _run(cmd: list[str], *, timeout: float = 10.0) -> str:
    """stdout and stderr of a short diagnostic command (``ld -v`` prints to
    stderr), or ``""`` when it cannot run or fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    if out.returncode != 0:
        return ""
    return (out.stdout or "") + (out.stderr or "")


def _probe(
    compiler: list[str],
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[bool | None, str]:
    """Build :data:`_PROBE_SOURCE` with the command every codegen library is built
    with. Returns ``(True, "")`` when it builds, ``(False, stderr)`` when the
    compiler or linker refuses it, and ``(None, why)`` when the attempt proves
    nothing (it timed out, or the scratch directory could not be written)."""
    from bngsim import _codegen as cg

    limit = _PROBE_TIMEOUT if timeout is None else timeout
    opt_flag = cg._resolve_opt_flag(Path(compiler[0]).stem.lower(), len(_PROBE_SOURCE))
    try:
        with tempfile.TemporaryDirectory(prefix="bngsim_toolchain_probe_") as work:
            c_path = Path(work) / "probe.c"
            so_path = Path(work) / f"probe{cg._shared_lib_suffix()}"
            c_path.write_text(_PROBE_SOURCE, encoding="utf-8")
            cmd = cg._build_compile_cmd(c_path, so_path, opt_flag, compiler)
            # From the scratch directory, so what a compiler writes beside its
            # output (MSVC's probe.obj) goes away with it.
            result = cg._run_compile(cmd, cwd=Path(work), timeout=limit, env=env)
    except subprocess.TimeoutExpired:
        return None, f"the probe build ran past {limit:g} s"
    except OSError as err:
        # A compiler that is not there is caught before any probe
        # (record_missing); anything else here is the scratch directory's.
        return None, f"the probe could not be written or run ({err})"
    if result.returncode == 0:
        return True, ""
    return False, (result.stderr or "") + (result.stdout or "")


def _excerpt(stderr: str) -> str:
    lines = [line.rstrip() for line in stderr.splitlines() if line.strip()]
    shown = lines[:_STDERR_LINES]
    if len(lines) > len(shown):
        shown.append(f"... ({len(lines) - len(shown)} more lines)")
    return "\n".join(f"    {line}" for line in shown)


def _linker(compiler: list[str]) -> str:
    """The linker the compiler drives and its version line, as far as they can be
    asked; ``""`` for MSVC or when they cannot."""
    if Path(compiler[0]).stem.lower() == "cl":
        return ""
    name = _run([*compiler, "-print-prog-name=ld"]).strip().splitlines()
    ld = name[0].strip() if name else ""
    if not ld:
        return ""
    resolved = shutil.which(ld) or ld
    version = next((ln.strip() for ln in _run([resolved, "-v"]).splitlines() if ln.strip()), "")
    return f"{resolved} ({version})" if version else resolved


def _current_sdk() -> tuple[str, str]:
    """The macOS SDK a compile uses, and where that came from."""
    sdkroot = os.environ.get("SDKROOT", "")
    if sdkroot:
        return sdkroot, "SDKROOT"
    return _run(["xcrun", "--show-sdk-path"]).strip(), "xcrun --show-sdk-path"


def _sdk_version(path: str) -> tuple[int, ...]:
    """The version an SDK's name says, from the name itself or the directory it
    points at, whichever says more (Xcode's ``MacOSX27.sdk`` points at an
    unversioned ``MacOSX.sdk``; the CLT's ``MacOSX26.sdk`` at ``MacOSX26.5.sdk``)."""

    def parse(name: str) -> tuple[int, ...]:
        match = re.search(r"MacOSX(\d+(?:\.\d+)*)\.sdk", name)
        return tuple(int(x) for x in match.group(1).split(".")) if match else ()

    return max(parse(os.path.basename(path)), parse(os.path.basename(os.path.realpath(path))))


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _sdk_roots() -> list[str]:
    """Where macOS SDKs are installed: the Command Line Tools, the selected
    developer directory, and the default Xcode."""
    roots = ["/Library/Developer/CommandLineTools/SDKs"]
    developer = _run(["xcode-select", "-p"]).strip()
    if developer:
        roots.append(f"{developer}/Platforms/MacOSX.platform/Developer/SDKs")
    roots.append(
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs"
    )
    return roots


def _macos_sdk_candidates(current: str, roots: list[str] | None = None) -> list[str]:
    """Installed macOS SDKs other than ``current``, newest first, one name per SDK
    (the shortest, so ``MacOSX26.sdk`` rather than the ``MacOSX26.5.sdk`` it
    points at: the major-version link survives a point update)."""
    if roots is None:
        roots = _sdk_roots()
    current_real = os.path.realpath(current) if current else ""
    by_target: dict[str, str] = {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "MacOSX*.sdk"))):
            if os.path.basename(path) == "MacOSX.sdk":
                continue  # the unversioned name tracks the newest, current or not
            real = os.path.realpath(path)
            if real == current_real or not os.path.isdir(real):
                continue
            kept = by_target.get(real)
            if kept is None or len(os.path.basename(path)) < len(os.path.basename(kept)):
                by_target[real] = path
    return sorted(by_target.values(), key=_sdk_version, reverse=True)


def _sdk_suggestion(compiler: list[str], current: str) -> str:
    """An installed SDK the probe builds with, as advice; ``""`` when none is
    found within :data:`_SDK_SCAN_BUDGET` seconds."""
    candidates = _macos_sdk_candidates(current)[:_MAX_SDK_CANDIDATES]
    if not candidates:
        return ""
    deadline = time.monotonic() + _SDK_SCAN_BUDGET
    tried = []
    for sdk in candidates:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        tried.append(sdk)
        ok, _ = _probe(compiler, env=dict(os.environ, SDKROOT=sdk), timeout=remaining)
        if ok:
            return (
                f"SDKROOT={sdk} builds it: export that before starting Python (or set "
                "os.environ['SDKROOT'], which retries)."
            )
    if not tried:
        return ""
    return f"None of the other installed SDKs tried builds it either ({', '.join(tried)})."


def _broken_message(compiler: list[str], stderr: str) -> str:
    """The broken-toolchain verdict: what failed, with what, and what to do."""
    exe = shutil.which(compiler[0]) or compiler[0]
    lines = [
        "The C toolchain cannot build a codegen library: a trivial C file fails to "
        "build with the command every codegen library is built with, so the failure "
        "is the environment's, not the model's (issue #1544).",
        f"  compiler: {exe}{' (CC)' if os.environ.get('CC') else ''}",
    ]
    linker = _linker(compiler)
    if linker:
        lines.append(f"  linker:   {linker}")
    suggestion = ""
    if _is_macos():
        sdk, source = _current_sdk()
        if sdk:
            target = os.path.basename(os.path.realpath(sdk))
            shown = sdk if target == os.path.basename(sdk) else f"{sdk} -> {target}"
            lines.append(f"  SDK:      {shown} ({source})")
            suggestion = _sdk_suggestion(compiler, sdk)
    lines.append("  error:")
    lines.append(_excerpt(stderr) or "    (the compiler printed nothing)")
    if suggestion:
        lines.append(suggestion)
    lines.append(
        "No codegen library is compiled while the compiler and its environment (CC, "
        "PATH, SDKROOT, DEVELOPER_DIR, ...) stay as they are; libraries already in "
        f"the cache are still used, and the toolchain is probed again after "
        f"{_VERDICT_TTL:g} s. {_FALLBACK}"
    )
    return "\n".join(lines)


def classify_failure(compiler: list[str]) -> str | None:
    """After a codegen compile has failed, whether the toolchain is to blame.

    Returns the broken-toolchain message, keeping it as the verdict for this
    toolchain and logging it once, when the probe fails too; ``None`` when the
    probe builds (the model's source is what failed) or proves nothing."""
    key = toolchain_key(compiler)
    with _lock:
        entry = _verdicts.get(key)
        if entry is not None and time.monotonic() - entry[1] < _VERDICT_TTL:
            return entry[0]
        ok, detail = _probe(compiler)
        if ok is None:
            logger.debug("codegen toolchain probe inconclusive: %s", detail)
            return None
        if ok:
            _verdicts[key] = (None, time.monotonic())
            return None
        message = _broken_message(compiler, detail)
        first = _keep_broken(key, message)
    if first:
        logger.warning("%s", message)
    return message
