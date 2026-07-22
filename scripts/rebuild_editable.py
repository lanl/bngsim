#!/usr/bin/env python3
"""Reconfigure, rebuild, and reinstall the editable bngsim extension.

Editable installs keep the Python package in-tree but import the compiled
extension from the active environment's ``site-packages/bngsim``. Since
``editable.rebuild = false`` in ``pyproject.toml``, C++ changes do not
refresh that installed extension automatically on import; this helper is the
supported path for rebuilding and reinstalling it for the current interpreter.

On macOS, older editable build directories may cache a stale universal
``CMAKE_OSX_ARCHITECTURES=x86_64;arm64`` setting. Reusing that cache on an
arm64-only Homebrew setup makes the Python rebuild fail when test executables
or the extension try to link against x86_64 KLU slices that do not exist.
This helper reconfigures the build for the current interpreter architecture,
keeps Python mode on, keeps tests off, and rebuilds only ``_bngsim_core``
before reinstalling it. It then regenerates the ``_bngsim_core.pyi`` type stub
from the freshly built module (via pybind11-stubgen) so the stub mypy checks
against never drifts from the bindings.

Two things differ under ``uv`` (the environment manager CONTRIBUTING.md
documents), and both used to make this script unusable there:

* uv builds the package in an **ephemeral isolated venv**, so scikit-build-core
  records a ``python_executable`` under ``~/.cache/uv/builds-v0/.tmpXXXX/`` that
  is deleted the moment the build finishes — and ``CMakeCache.txt`` caches the
  same dead path in ``Python_EXECUTABLE``. Matching build metadata on that
  recorded interpreter can therefore never succeed. ``_load_build_info`` falls
  back to selecting by extension ABI, and the configure line pins
  ``Python_EXECUTABLE`` to the running interpreter so the stale cache entry is
  overridden rather than trusted.
* a uv-created venv has **no ``pip``**, so ``python -m pip install -e`` fails
  outright. The install steps route through ``uv pip`` in that case (see
  ``_editable_install_cmd``). Note the happy path needs no installer at all —
  it is pure cmake — so this only matters for the bootstrap and version-drift
  branches.
"""

from __future__ import annotations

import contextlib
import errno
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

# Default upper bound on how long to wait for the editable_rebuild.lock.
# 10 minutes is comfortably longer than a clean rebuild on a slow box and
# short enough that a deadlock surfaces as a clear failure rather than a
# silent hang.
_LOCK_TIMEOUT_SECONDS = 600


@contextlib.contextmanager
def _editable_rebuild_lock(build_dir: Path, *, timeout: float) -> Iterator[None]:
    """Acquire scikit-build-core's editable rebuild lock convention.

    ``editable.rebuild`` is currently disabled, so this is mostly defensive:
    it still prevents accidental overlap with any future import-time rebuild
    hook, or with another manual helper invocation that uses the same lock
    path. We hold the flock for the duration of our cmake invocations so
    concurrent rebuild attempts do not race in the same build directory.

    No-op on non-Unix platforms (where ``fcntl`` is unavailable) — the
    scikit-build-core lock convention is Unix-only there as well.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return

    lock_path = build_dir / "editable_rebuild.lock"
    build_dir.mkdir(parents=True, exist_ok=True)

    flags = os.O_RDWR | os.O_TRUNC
    if not lock_path.exists():
        flags |= os.O_CREAT
    fd = os.open(str(lock_path), flags, 0o644)
    try:
        deadline = time.monotonic() + timeout
        last_log = 0.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout:.0f}s waiting for editable rebuild lock "
                    f"at {lock_path}. Another rebuild (likely an import-time auto-rebuild "
                    f"from a parallel `import bngsim`) is holding the lock. If you are "
                    f"sure no real rebuild is in progress, remove the lock file and retry."
                )
            if now - last_log > 30:
                last_log = now
                remaining = max(0, deadline - now)
                print(
                    f"Waiting for editable rebuild lock at {lock_path} "
                    f"({remaining:.0f}s remaining)...",
                    flush=True,
                )
            time.sleep(0.1)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _resolve_macos_sdkroot() -> str | None:
    if sys.platform != "darwin":
        return None

    for env_name in ("CMAKE_OSX_SYSROOT", "SDKROOT"):
        candidate = os.environ.get(env_name, "").strip()
        if candidate and Path(candidate).exists():
            return candidate

    try:
        proc = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    candidate = proc.stdout.strip()
    if candidate and Path(candidate).exists():
        return candidate
    return None


def _cmake_env() -> dict[str, str] | None:
    sdkroot = _resolve_macos_sdkroot()
    if sdkroot is None:
        return None
    env = os.environ.copy()
    env["SDKROOT"] = sdkroot
    env["CMAKE_OSX_SYSROOT"] = sdkroot
    return env


def _requested_macos_architectures() -> str | None:
    if sys.platform != "darwin":
        return None

    for env_name in ("BNGSIM_CMAKE_OSX_ARCHITECTURES", "CMAKE_OSX_ARCHITECTURES"):
        candidate = os.environ.get(env_name, "").strip()
        if candidate:
            return candidate

    arch = platform.machine().strip().lower()
    if arch in {"arm64", "x86_64"}:
        return arch
    return None


def _editable_install_cmd(source_dir: Path) -> list[str]:
    """Build the argv that (re)registers the editable install for this interpreter.

    ``python -m pip`` is the historical path and stays preferred: where pip
    exists the project's build deps are expected alongside it, so
    ``--no-build-isolation`` reuses the already-configured build tree instead of
    paying for a from-scratch one.

    A uv-created venv ships no pip at all, which made every call site here die
    with ``No module named pip``. Fall back to ``uv pip`` against this exact
    interpreter. Build isolation is deliberately left ON for that branch: uv
    venvs do not carry scikit-build-core or pybind11 (they live only in the
    transient build env), so ``--no-build-isolation`` would fail to find the
    backend. The isolated build records a dead ``python_executable``, which is
    precisely the case ``_load_build_info`` now tolerates.
    """
    if importlib.util.find_spec("pip") is not None:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(source_dir),
        ]

    uv = shutil.which("uv")
    if uv is not None:
        return [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "-e",
            str(source_dir),
        ]

    raise RuntimeError(
        "Cannot register the editable install: this interpreter has no pip "
        f"({sys.executable}) and no `uv` was found on PATH. Install one of them, "
        "or provision the environment directly with "
        "`uv sync --extra dev --reinstall-package bngsim`."
    )


def _bootstrap_editable(source_dir: Path, *, env: dict[str, str] | None) -> None:
    _run(_editable_install_cmd(source_dir), env=env)


def _load_build_info(source_dir: Path) -> dict[str, str]:
    """Locate the build tree belonging to the running interpreter.

    Preferred key is the interpreter scikit-build-core recorded at build time.
    That is exact when the build ran in the target environment (pip with
    ``--no-build-isolation``), and useless when it ran under build isolation:
    uv builds in a throwaway venv beneath ``~/.cache/uv/builds-v0/`` and records
    that path, so the recorded interpreter is both unequal to ours and gone from
    disk by the time we look.

    The fallback keys on what actually has to match — the extension ABI. A tree
    holding a built ``_bngsim_core`` with this interpreter's ``EXT_SUFFIX`` is
    loadable by this interpreter; the newest such tree is the one whose artifact
    the environment is currently running. That is a heuristic, not a proof: two
    venvs on the same CPython version share an ``EXT_SUFFIX``, so this can pick a
    sibling venv's tree. Retargeting is harmless for *us* — main() pins
    ``Python_EXECUTABLE`` and installs into our own platlib — but it does leave
    the sibling needing a rebuild.
    """
    build_root = source_dir / "build"
    current_python = Path(sys.executable).resolve()

    infos: list[dict[str, str]] = []
    for candidate in sorted(build_root.glob("*/.skbuild-info.json")):
        data = json.loads(candidate.read_text())
        if Path(data["source_dir"]).resolve() == source_dir.resolve():
            infos.append(data)

    for data in infos:
        if Path(data["python_executable"]).resolve() == current_python:
            return data

    ext_suffix = _ext_suffix()
    abi_matches: list[tuple[float, dict[str, str]]] = []
    for data in infos:
        built = Path(data["build_dir"]) / f"_bngsim_core{ext_suffix}"
        if built.is_file():
            abi_matches.append((built.stat().st_mtime, data))
    if abi_matches:
        newest = max(abi_matches, key=lambda pair: pair[0])[1]
        print(
            f"build metadata: no tree recorded for {sys.executable}; "
            f"selected {newest['build_dir']} by extension ABI ({ext_suffix})",
            flush=True,
        )
        return newest

    if len(infos) == 1:
        return infos[0]

    raise FileNotFoundError(
        "No editable build metadata found for this interpreter. "
        "Bootstrap the editable install first."
    )


def _install_prefix() -> Path:
    platlib = sysconfig.get_path("platlib")
    if not platlib:
        raise RuntimeError("Could not determine platlib path for the current interpreter")
    return Path(platlib).resolve() / "bngsim"


def _pyproject_version(source_dir: Path) -> str | None:
    """Parse the single ``version = "X.Y.Z"`` literal from pyproject.toml."""
    pyproject = source_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject.read_text(),
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _installed_metadata_version() -> str | None:
    """Return the version recorded in this interpreter's bngsim dist-info.

    Reads fresh metadata (invalidating importlib's caches) so a refresh
    performed earlier in the same process is observed.
    """
    importlib.invalidate_caches()
    try:
        return importlib.metadata.version("bngsim")
    except importlib.metadata.PackageNotFoundError:
        return None


def _refresh_editable_metadata(source_dir: Path, *, env: dict[str, str] | None) -> None:
    """Re-register the editable install's dist-info for the current interpreter.

    ``cmake --install`` refreshes the compiled extension in place but leaves
    the dist-info METADATA untouched, so ``importlib.metadata.version`` (and
    therefore ``bngsim.__version__``) keeps reporting the pre-bump version
    after a ``pyproject.toml`` version change. Re-running the editable install
    with ``--no-build-isolation --no-deps`` re-registers the metadata cheaply
    (it reuses the already-built extension; no from-scratch C++ rebuild).

    On the ``uv pip`` fallback (see ``_editable_install_cmd``) the build is
    isolated, so this costs a real rebuild rather than a metadata-only refresh.
    It runs only on a detected version drift, so that is a rare price.
    """
    _run(_editable_install_cmd(source_dir), env=env)


def _ext_suffix() -> str:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Could not determine Python extension suffix")
    return str(ext_suffix)


def _regenerate_stub(source_dir: Path, *, env: dict[str, str] | None) -> None:
    """Regenerate the committed ``_bngsim_core.pyi`` from the freshly built module.

    The stub is the type contract mypy checks against, and pybind11 does not
    emit it — so without regeneration it silently drifts out of date whenever
    the C++ bindings gain a member, and the missing symbols surface later as
    spurious ``attr-defined`` mypy errors on whatever Python file happens to use
    them. Running pybind11-stubgen here, right after the extension is rebuilt and
    reinstalled, keeps the stub in lockstep with the bindings.

    Opt out with ``BNGSIM_SKIP_STUBGEN=1``. Best-effort on a missing generator:
    pybind11-stubgen ships in the ``dev`` extra, but a plain rebuild without it
    warns and skips rather than failing (the binary is already built by now).
    """
    if os.environ.get("BNGSIM_SKIP_STUBGEN", "") not in ("", "0"):
        print("stubgen=skipped (BNGSIM_SKIP_STUBGEN)", flush=True)
        return
    if importlib.util.find_spec("pybind11_stubgen") is None:
        print(
            "stubgen=skipped (pybind11-stubgen not installed; install the "
            "bngsim[dev] extra or `pip install pybind11-stubgen` to enable)",
            flush=True,
        )
        return

    stub_dest = source_dir / "python" / "bngsim" / "_bngsim_core.pyi"
    stub_env = dict(env) if env is not None else os.environ.copy()
    # We just built the extension; let the generator import it without the
    # staleness guard tripping on an mtime race between install and import.
    stub_env["BNGSIM_ALLOW_STALE_CORE"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        _run(
            [
                sys.executable,
                "-m",
                "pybind11_stubgen",
                "bngsim._bngsim_core",
                # SolverOptions()/SteadyStateOptions() default args render as raw
                # C++ object reprs the generator cannot parse; fall back to
                # ``= ...`` for those instead of erroring on them.
                "--ignore-invalid-expressions",
                "<.*>",
                "--output-dir",
                tmp,
            ],
            env=stub_env,
        )
        generated = Path(tmp) / "bngsim" / "_bngsim_core.pyi"
        if not generated.is_file():
            raise FileNotFoundError(f"pybind11-stubgen did not produce {generated}")
        shutil.copyfile(generated, stub_dest)
    print(f"stub_regenerated={stub_dest}", flush=True)


def main() -> int:
    source_dir = Path(__file__).resolve().parents[1]
    cmake_env = _cmake_env()
    cmake_sdkroot = cmake_env.get("CMAKE_OSX_SYSROOT") if cmake_env is not None else None

    try:
        build_info = _load_build_info(source_dir)
    except FileNotFoundError:
        _bootstrap_editable(source_dir, env=cmake_env)
        build_info = _load_build_info(source_dir)

    build_dir = Path(build_info["build_dir"]).resolve()
    install_prefix = _install_prefix()
    macos_architectures = _requested_macos_architectures()

    # Tell any child process that ends up importing bngsim during the
    # build (rare but possible — e.g. a CMake test that links against
    # the extension) to skip its own auto-rebuild. The hook recognizes
    # the build_dir path appearing in SKBUILD_EDITABLE_SKIP via os.pathsep.
    inherited_skip = os.environ.get("SKBUILD_EDITABLE_SKIP", "")
    skip_value = (
        os.pathsep.join((inherited_skip, str(build_dir))) if inherited_skip else str(build_dir)
    )
    if cmake_env is None:
        cmake_env = os.environ.copy()
    cmake_env["SKBUILD_EDITABLE_SKIP"] = skip_value

    timeout = float(os.environ.get("BNGSIM_REBUILD_LOCK_TIMEOUT", _LOCK_TIMEOUT_SECONDS))
    with _editable_rebuild_lock(build_dir, timeout=timeout):
        configure_cmd = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DBNGSIM_BUILD_PYTHON=ON",
            "-DBNGSIM_BUILD_TESTS=OFF",
            # Pin the interpreter we are building *for*. Without this, FindPython
            # reuses the cache, and a tree produced under build isolation cached a
            # Python_EXECUTABLE inside a build venv that no longer exists (uv wipes
            # ~/.cache/uv/builds-v0/.tmpXXXX after each build). Passing it also
            # makes the ABI-selected fallback in _load_build_info safe: whichever
            # tree we reuse is retargeted at this interpreter before it is built.
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if cmake_sdkroot:
            configure_cmd.append(f"-DCMAKE_OSX_SYSROOT={cmake_sdkroot}")
        if macos_architectures:
            configure_cmd.append(f"-DCMAKE_OSX_ARCHITECTURES={macos_architectures}")
        # Carry the GH #78 MIR micro-JIT opt-in through to the configure so a
        # reconfigure doesn't silently turn the prototype backend off. Default
        # OFF (matches the CMake option); set BNGSIM_ENABLE_MIR=1 to build it.
        if os.environ.get("BNGSIM_ENABLE_MIR", "").strip().lower() in ("1", "on", "true", "yes"):
            configure_cmd.append("-DBNGSIM_ENABLE_MIR=ON")

        _run(
            configure_cmd,
            env=cmake_env,
        )

        _run(
            ["cmake", "--build", str(build_dir), "--target", "_bngsim_core"],
            env=cmake_env,
        )
        _run(
            ["cmake", "--install", str(build_dir), "--prefix", str(install_prefix)],
            env=cmake_env,
        )

    installed_extension = install_prefix / f"_bngsim_core{_ext_suffix()}"
    print(f"build_dir={build_dir}", flush=True)
    print(f"installed_extension={installed_extension}", flush=True)

    # Keep the committed type stub in lockstep with the just-built bindings.
    _regenerate_stub(source_dir, env=cmake_env)

    # `cmake --install` refreshes the compiled extension but not the editable
    # install's dist-info, so after a pyproject version bump bngsim.__version__
    # (which reads importlib.metadata) keeps reporting the old version and
    # test_version_consistency fails. Detect that drift and re-register the
    # metadata. Opt out with BNGSIM_SKIP_METADATA_REFRESH=1.
    if os.environ.get("BNGSIM_SKIP_METADATA_REFRESH", "") not in ("", "0"):
        print("metadata_refresh=skipped (BNGSIM_SKIP_METADATA_REFRESH)", flush=True)
        return 0

    pyproject_version = _pyproject_version(source_dir)
    installed_version = _installed_metadata_version()
    if pyproject_version is not None and pyproject_version != installed_version:
        print(
            f"metadata drift: installed={installed_version} "
            f"pyproject={pyproject_version}; refreshing dist-info",
            flush=True,
        )
        _refresh_editable_metadata(source_dir, env=cmake_env)
        installed_version = _installed_metadata_version()
    print(f"metadata_version={installed_version}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
