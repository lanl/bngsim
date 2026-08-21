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
  outright. The install steps route through ``uv pip`` in that case, and a
  ``uv venv --seed`` interpreter — pip but no build backend — takes pip's own
  build isolation (see ``_editable_install_cmd``). Note the happy path needs no
  installer at all — it is pure cmake — so this only matters for the bootstrap
  and version-drift branches.
* pybind11 is a ``[build-system] requires`` entry, which uv supplies in a
  transient isolated build env and **never installs into ``.venv``**. The pure
  cmake happy path has no such env, so ``find_package(pybind11)`` has to succeed
  from the environment this script is running in. ``_pybind11_cmake_dir`` asks
  the running interpreter and pins ``pybind11_DIR`` when it can answer; see
  that function for what happens when it cannot (GH #229).

One more thing about that shared tree. Every build for one interpreter and
platform uses the same CMake build directory (``build-dir = "build/{wheel_tag}"``
in ``pyproject.toml``), and the wheel tag records the Python version and the
platform — not the virtual environment being installed into, and not the options
passed. A second install with different options therefore rewrites the cache
this script builds from, and any option this script leaves unspecified comes
back from whatever that other build left behind. That is how a working editable
install silently starts producing a binary with different features compiled in:
the staleness guard compares timestamps, and the binary really is fresh — it is
just configured differently (GH #459). So this script names **every** feature
option on the configure line rather than inheriting any of them
(:data:`_FEATURE_OPTION_DEFAULTS`), and refuses to build on a tree whose cache
disagrees with what it is about to ask for (:func:`_configuration_conflicts`).
Setting a same-named environment variable to a CMake boolean overrides one
option for one rebuild — ``BNGSIM_ENABLE_MIR=1`` is the long-standing spelling of
that, and it is now how every option in the table is asked for.
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
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

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


def _pybind11_cmake_dir() -> str | None:
    """This interpreter's pybind11 CMake package directory, or ``None``.

    ``find_package(pybind11 CONFIG REQUIRED)`` in CMakeLists.txt has to resolve
    from whatever environment the configure runs in. Under uv that environment
    is the project ``.venv``, and pybind11 is declared only in
    ``[build-system] requires`` — uv installs it into a transient isolated build
    env, never into ``.venv``. So the one interpreter that is guaranteed to be
    *right* (the one we are building for) is also the one CMake cannot ask
    without help. We are already running in it, so we ask it directly and pin
    the answer with ``-Dpybind11_DIR``. GH #229.

    Returning ``None`` is deliberately non-fatal: CMake can still find a
    system-wide pybind11 (Homebrew ships one under ``/opt/homebrew``), and that
    path works today on plenty of machines. Refusing here would break them for
    a dependency they demonstrably do not need. What the caller does instead is
    say so up front and, if the configure then fails, name the remedy — see
    :data:`_PYBIND11_MISSING_NOTE`.
    """
    try:
        import pybind11
    except ImportError:
        return None
    try:
        cmake_dir = Path(str(pybind11.get_cmake_dir()))
    except Exception:
        return None
    # Pinning a directory with no config file in it turns one CMake error into a
    # different, more confusing one ("pybind11_DIR was set to a directory not
    # containing..."). Both spellings, matching the CMakeLists.txt prelude.
    if not any(
        (cmake_dir / name).is_file() for name in ("pybind11Config.cmake", "pybind11-config.cmake")
    ):
        return None
    # Forward slashes: a Windows site-packages path carries backslashes, and a
    # backslash in a -D value is an escape to CMake. Identical to str() on POSIX.
    return cmake_dir.as_posix()


#: What to tell someone whose configure just died for want of pybind11.
#:
#: The stale-binary guard names this script as *the* remedy
#: (``_build_provenance.py``), and the only other thing it offers is
#: ``BNGSIM_ALLOW_STALE_CORE=1`` — run against a binary that does not match the
#: source, which is precisely what the guard exists to prevent. So a raw CMake
#: error here does not just fail; it pushes the reader toward the unsafe escape
#: hatch, on the path designed to keep them off it. GH #229.
_PYBIND11_MISSING_NOTE = """\
The cmake configure above failed, and pybind11 is not importable in this
interpreter. So if the error is

    Could not find a package configuration file provided by "pybind11"

then that is why, and here is the fix. (If cmake failed for some other reason,
read its message above — this note does not apply.)

pybind11 is a [build-system] requirement, which uv installs into a transient
isolated build env and never into .venv — so a venv only carries it if some
extra declares it. The `dev` extra does:

    uv sync --extra dev

Or rebuild through uv's own isolated build env, which needs no local pybind11
(name every extra you want — `uv sync` prunes the ones you omit):

    uv sync --extra dev --reinstall-package bngsim\
"""


#: The CMake options that decide which features get compiled into
#: ``_bngsim_core``, each mapped to its ``option()`` default in CMakeLists.txt.
#:
#: None of these may be inherited from the build tree. The cache in
#: ``build/{wheel_tag}`` is only ever a record of whichever build ran last, and
#: that build may have been an install into an entirely different virtual
#: environment with different ``--config-settings`` (GH #459). Naming each option
#: on the configure line is what makes a rebuild produce the configuration it was
#: asked for instead of the one the tree happens to be carrying.
#:
#: ``test_rebuild_editable_feature_options.py`` holds these defaults to
#: CMakeLists.txt, and fails if an option is added there without being
#: classified here or in :data:`_PINNED_OPTION_VALUES`.
_FEATURE_OPTION_DEFAULTS: dict[str, str] = {
    "BNGSIM_ENABLE_KLU": "ON",
    "BNGSIM_REQUIRE_KLU": "OFF",
    "BNGSIM_KLU_AUTOBUILD": "ON",
    "BNGSIM_USE_SYSTEM_SUNDIALS": "OFF",
    "BNGSIM_BUILD_NFSIM": "ON",
    "BNGSIM_BUILD_RULEMONKEY": "ON",
    "BNGSIM_ENABLE_MIR": "OFF",
    "BNGSIM_ENABLE_LAPACK_DENSE": "ON",
}

#: Options whose value follows from what this script *is* — an incremental
#: rebuild of the Python bindings and nothing else — rather than from how anyone
#: configured the project. Not overridable, and not part of the cache comparison:
#: a tree that also carries the C++ unit tests is not misconfigured, it just
#: builds more than we ask it to.
_PINNED_OPTION_VALUES: dict[str, str] = {
    "BNGSIM_BUILD_PYTHON": "ON",
    "BNGSIM_BUILD_TESTS": "OFF",
}

#: scikit-build-core's own default, for a pyproject.toml that names no build
#: type. Inheriting the build type has the same failure mode as inheriting a
#: feature option, and a quieter one: a tree another install configured ``Debug``
#: rebuilds as ``Debug`` here, with nothing but the compile lines to say so.
_DEFAULT_BUILD_TYPE = "Release"

_ORIGIN_DEFAULT = "default"
_ORIGIN_PYPROJECT = "pyproject.toml"
_ORIGIN_ENVIRONMENT = "environment"


class _Setting(NamedTuple):
    """One resolved CMake setting, and what asked for it."""

    value: str
    origin: str


def _as_cmake_bool(raw: str) -> str | None:
    """``ON``/``OFF`` for a boolean spelling CMake accepts, else ``None``.

    Deliberately a subset of CMake's own truthiness rules (which also take
    ``IGNORE``, ``NOTFOUND`` and any non-zero number): these values are read from
    a human-typed environment variable and from the two files in this repo that
    set them, so a spelling outside the subset is far more likely to be a typo
    than an intention. ``None`` lets the caller say so by name.
    """
    value = raw.strip().lower()
    if value in ("1", "on", "true", "yes", "y"):
        return "ON"
    if value in ("0", "off", "false", "no", "n"):
        return "OFF"
    return None


def _toml_table_body(text: str, header: str) -> str | None:
    """The lines under ``[header]``, up to the next table header or EOF."""
    match = re.search(
        rf"^\[{re.escape(header)}\]\s*$(?P<body>.*?)(?=^\[|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return None if match is None else match.group("body")


def _pyproject_cmake_settings(source_dir: Path) -> dict[str, str]:
    """The CMake settings ``pyproject.toml`` hands to every scikit-build-core build.

    That is ``cmake.build-type``, the ``-D`` entries in ``cmake.args``, and the
    ``[tool.scikit-build.cmake.define]`` table — the configuration a plain
    ``pip install .`` produces, and therefore the configuration an editable
    rebuild should reproduce when nothing asks for anything else.

    Read out of the file rather than copied into a constant for the reason
    :func:`_build_dep_modules` gives: a second copy is how a paired site starts
    drifting. Regex rather than ``tomllib`` because this script supports Python
    3.10, where there is no ``tomllib``.
    """
    pyproject = source_dir / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    text = pyproject.read_text()
    settings: dict[str, str] = {}

    skbuild = _toml_table_body(text, "tool.scikit-build")
    if skbuild is not None:
        build_type = re.search(r'^cmake\.build-type\s*=\s*"([^"]*)"', skbuild, re.MULTILINE)
        if build_type is not None:
            settings["CMAKE_BUILD_TYPE"] = build_type.group(1)
        args = re.search(
            r"^cmake\.args\s*=\s*\[(?P<items>[^\]]*)\]",
            skbuild,
            re.MULTILINE | re.DOTALL,
        )
        if args is not None:
            # Comments inside the array are prose, same as in [build-system].
            items = re.sub(r"#[^\n]*", "", args.group("items"))
            for entry in re.findall(r"""['"]([^'"]+)['"]""", items):
                define = re.fullmatch(
                    r"-D([A-Za-z_][A-Za-z0-9_]*)(?::[A-Za-z]+)?=(.*)", entry.strip()
                )
                if define is not None:
                    settings[define.group(1)] = define.group(2)

    # Last, so it wins over cmake.args: scikit-build-core applies cmake.define
    # after the raw args, and a project that sets one option in both places means
    # the table entry.
    define_table = _toml_table_body(text, "tool.scikit-build.cmake.define")
    if define_table is not None:
        for name, value in re.findall(
            r"""^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['"]([^'"]*)['"]""",
            define_table,
            re.MULTILINE,
        ):
            settings[name] = value
    return settings


def _resolve_cmake_settings(
    source_dir: Path, environ: Mapping[str, str] | None = None
) -> dict[str, _Setting]:
    """Every configuration-carrying setting this rebuild will pass, and its source.

    Resolution order, weakest first: the CMakeLists.txt ``option()`` default, the
    value ``pyproject.toml`` declares, then a same-named environment variable.
    The build tree is not in that list — that is the whole point (GH #459).
    """
    environ = os.environ if environ is None else environ
    declared = _pyproject_cmake_settings(source_dir)
    resolved: dict[str, _Setting] = {}

    build_type = declared.get("CMAKE_BUILD_TYPE")
    if build_type is None:
        resolved["CMAKE_BUILD_TYPE"] = _Setting(_DEFAULT_BUILD_TYPE, _ORIGIN_DEFAULT)
    else:
        resolved["CMAKE_BUILD_TYPE"] = _Setting(build_type, _ORIGIN_PYPROJECT)
    requested_build_type = environ.get("CMAKE_BUILD_TYPE", "").strip()
    if requested_build_type:
        resolved["CMAKE_BUILD_TYPE"] = _Setting(requested_build_type, _ORIGIN_ENVIRONMENT)

    for name, default in _FEATURE_OPTION_DEFAULTS.items():
        setting = _Setting(default, _ORIGIN_DEFAULT)
        if name in declared:
            value = _as_cmake_bool(declared[name])
            if value is None:
                raise SystemExit(
                    f"pyproject.toml sets {name} = {declared[name]!r}, which is not a "
                    f"CMake boolean. Use ON or OFF."
                )
            setting = _Setting(value, _ORIGIN_PYPROJECT)
        requested = environ.get(name, "").strip()
        if requested:
            value = _as_cmake_bool(requested)
            if value is None:
                raise SystemExit(
                    f"{name}={requested!r} in the environment is not a CMake boolean. "
                    f"Use ON/OFF, 1/0, true/false, or yes/no."
                )
            setting = _Setting(value, _ORIGIN_ENVIRONMENT)
        resolved[name] = setting

    return resolved


def _cmake_cache_values(build_dir: Path, names: Iterable[str]) -> dict[str, str]:
    """The cached value of each of ``names``, for the entries the cache has.

    Booleans come back normalized to ``ON``/``OFF`` so a cache written ``TRUE``
    compares equal to a configure line asking for ``ON``. A missing cache (a tree
    that has never been configured) is an empty result, not an error, and so is an
    entry with no value: CMake writes a bare ``CMAKE_BUILD_TYPE:STRING=`` into any
    tree configured by hand without one, and a setting the tree never decided is
    not a setting it disagrees about — the configure line pins it either way.
    """
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        return {}
    wanted = set(names)
    values: dict[str, str] = {}
    # Anchored on both ends so the parallel `NAME-ADVANCED:INTERNAL=1` entries,
    # whose names carry a dash, cannot match.
    entry = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):([A-Za-z]+)=(.*)$")
    for line in cache_path.read_text(errors="replace").splitlines():
        match = entry.match(line.strip())
        if match is None:
            continue
        name, kind, value = match.group(1), match.group(2).upper(), match.group(3).strip()
        if name not in wanted:
            continue
        if not value:
            continue
        if kind == "BOOL":
            value = _as_cmake_bool(value) or value
        values[name] = value
    return values


def _configuration_conflicts(
    cached: Mapping[str, str], settings: Mapping[str, _Setting]
) -> list[tuple[str, str, str]]:
    """``(option, cached, wanted)`` for every setting the tree disagrees with.

    Settings this invocation asked for by environment variable are excluded: a
    disagreement there is the requested change, not drift, and reconfiguring for
    it is exactly what ``BNGSIM_ENABLE_MIR=1 python scripts/rebuild_editable.py``
    has always meant. What is left is a tree that says something nobody in this
    run asked for, which is the GH #459 signature.
    """
    conflicts: list[tuple[str, str, str]] = []
    for name, setting in settings.items():
        if setting.origin == _ORIGIN_ENVIRONMENT:
            continue
        cached_value = cached.get(name)
        if cached_value is None or cached_value == setting.value:
            continue
        conflicts.append((name, cached_value, setting.value))
    return conflicts


def _configuration_conflict_message(build_dir: Path, conflicts: list[tuple[str, str, str]]) -> str:
    """What to say to someone whose build tree was reconfigured behind their back."""
    rows = "\n".join(
        f"    {name}: cache has {cached}, this rebuild asks for {wanted}"
        for name, cached, wanted in conflicts
    )
    keep = " ".join(f"{name}={cached}" for name, cached, _ in conflicts)
    return f"""\
The CMake cache in

    {build_dir}

was configured with a different feature set than this rebuild asks for:

{rows}

Every build for this interpreter and platform shares that one directory
(build-dir = "build/{{wheel_tag}}" in pyproject.toml). Its name records the wheel
tag and nothing about the configuration, so a second install — into another
virtual environment, or with different --config-settings — rewrites the cache
this rebuild would otherwise build from. Building on it produces an extension
with features you did not ask for, and nothing else says so: the staleness guard
compares timestamps, and the binary really would be fresh (GH #459).

Reconfiguring in place is not enough, because a tree that has already been built
carries link settings from the configuration it was built with. Delete it and
reinstall:

    rm -rf {build_dir}
    uv pip install -e . --no-deps --force-reinstall

If the cached configuration is the one you want, ask for it and this rebuild
will keep it:

    {keep} python scripts/rebuild_editable.py

Giving a differently configured install its own tree is what stops this
happening again:

    uv pip install ... --config-settings=build-dir=/tmp/otherbuild/{{wheel_tag}} .\
"""


def _configure_cmd(
    source_dir: Path,
    build_dir: Path,
    *,
    settings: Mapping[str, _Setting],
    pybind11_cmake_dir: str | None,
    sdkroot: str | None,
    macos_architectures: str | None,
) -> list[str]:
    """The cmake configure argv for an editable rebuild of ``_bngsim_core``."""
    cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        # Pin the interpreter we are building *for*. Without this, FindPython
        # reuses the cache, and a tree produced under build isolation cached a
        # Python_EXECUTABLE inside a build venv that no longer exists (uv wipes
        # ~/.cache/uv/builds-v0/.tmpXXXX after each build). Passing it also
        # makes the ABI-selected fallback in _load_build_info safe: whichever
        # tree we reuse is retargeted at this interpreter before it is built.
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    # Pin the bindings to *this* interpreter's pybind11 when it has one. Beyond
    # making the configure work at all where nothing else supplies pybind11,
    # this is what keeps the two rebuild paths agreeing: uv's isolated build
    # resolves pybind11 from [build-system] requires, and without this flag a
    # cmake-only rebuild silently compiles against whatever unrelated copy the
    # system happens to ship instead. GH #229.
    if pybind11_cmake_dir is not None:
        cmd.append(f"-Dpybind11_DIR={pybind11_cmake_dir}")
    if sdkroot:
        cmd.append(f"-DCMAKE_OSX_SYSROOT={sdkroot}")
    if macos_architectures:
        cmd.append(f"-DCMAKE_OSX_ARCHITECTURES={macos_architectures}")
    # Every option that decides what ends up in the binary, named outright. What
    # is *not* here is inherited from CMakeCache.txt, and the cache belongs to
    # whichever build ran last in this tree rather than to this environment
    # (GH #459) — so nothing configuration-carrying may be left out.
    cmd += [f"-D{name}={value}" for name, value in _PINNED_OPTION_VALUES.items()]
    cmd += [f"-D{name}={setting.value}" for name, setting in settings.items()]
    return cmd


def _can_import(module: str) -> bool:
    """Whether ``module`` is importable in *this* interpreter.

    ``find_spec`` rather than a real import: these are probes, and importing the
    build backend to find out whether it exists would cost more than the answer
    is worth. It raises rather than returns ``None`` for a few shapes of broken
    entry (a ``None`` parked in ``sys.modules``, a module with no ``__spec__``);
    every one of them means "not usable from here", which is the answer.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _build_dep_modules(source_dir: Path) -> tuple[str, ...] | None:
    """Import names for pyproject's ``[build-system] requires``, or ``None``.

    Read out of pyproject rather than copied into a constant here. There is
    already one copy of this list — ``scripts/ship_wheel.py:BUILD_DEP_MODULES``,
    which has to hardcode it because it probes *foreign* interpreters — and a
    second copy is how a paired site starts drifting. This script only ever
    targets the interpreter it is running in, and it is always run from a source
    checkout, so it can read the requirement itself.

    ``None`` means the table could not be parsed, which the caller treats as
    "assume the deps are absent". That is the safe direction: an isolated build
    always works and merely costs time, while assuming deps that are not there
    fails inside pip with a traceback naming neither the module nor the fix.
    """
    pyproject = source_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    table = _toml_table_body(pyproject.read_text(), "build-system")
    if table is None:
        return None
    requires = re.search(
        r"^requires\s*=\s*\[(?P<items>[^\]]*)\]",
        table,
        re.MULTILINE | re.DOTALL,
    )
    if requires is None:
        return None
    # Comments inside the array are prose about what is deliberately *not*
    # required (ninja), so strip them before reading the quoted entries.
    items = re.sub(r"#[^\n]*", "", requires.group("items"))
    dists = re.findall(r"""['"]([^'"]+)['"]""", items)
    return tuple(
        re.split(r"[<>=!~;\[\s]", d, maxsplit=1)[0].strip().replace("-", "_") for d in dists
    )


def _has_build_deps(source_dir: Path) -> bool:
    """Whether this interpreter can run a PEP 517 build without isolation."""
    modules = _build_dep_modules(source_dir)
    if not modules:
        return False
    return all(_can_import(m) for m in modules)


def _editable_install_cmd(source_dir: Path) -> list[str]:
    """Build the argv that (re)registers the editable install for this interpreter.

    ``python -m pip install --no-build-isolation`` is the fast form and stays
    preferred: it reuses the already-configured build tree instead of paying for
    a from-scratch one. What it needs is not pip, though — it is pip **plus the
    PEP 517 backend importable in this same interpreter**, and the two come
    apart in both directions (GH #271):

    * A uv-created venv ships **no pip at all**, which made every call site here
      die with ``No module named pip``.
    * A ``uv venv --seed`` interpreter ships pip but **no scikit-build-core**,
      because the backend lives only in ``[build-system] requires`` and uv puts
      that in a transient isolated build env (the same fact behind GH #229).
      Asking only about pip sent that interpreter down the unisolated path,
      where it died with ``BackendUnavailable: Cannot import
      'scikit_build_core.build'`` — a traceback naming neither the missing dist
      nor the fix. ``scripts/ship_wheel.py:_has_build_deps`` exists to reject
      exactly this inference; this function had not learned it.

    So probe the precondition that is load-bearing, and when it fails let the
    installer supply the backend itself rather than refusing: pip's own build
    isolation reaches the same backend, so no environment that used to work
    stops working and no new hard failure is introduced. Only "no pip and no uv"
    is unrecoverable, and that one still raises.

    Both isolated branches record a dead ``python_executable`` in the build
    metadata, which is precisely the case ``_load_build_info`` tolerates — and
    they cost a real from-scratch build, which is why they are the fallback and
    not the default.
    """
    if _can_import("pip"):
        skip_isolation = ["--no-build-isolation"] if _has_build_deps(source_dir) else []
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            *skip_isolation,
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

    On either isolated branch (see ``_editable_install_cmd``: no pip, or pip
    without the build backend) the build is isolated, so this costs a real
    rebuild rather than a metadata-only refresh. It runs only on a detected
    version drift, so that is a rare price.
    """
    _run(_editable_install_cmd(source_dir), env=env)


def _ext_suffix() -> str:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Could not determine Python extension suffix")
    return str(ext_suffix)


#: The values in the generated stub that describe *the build that produced it*
#: rather than the API, and the placeholder each is rewritten to. CMake stamps
#: `__build_commit__` with the current git commit plus a `+dirty` suffix when the
#: tree has uncommitted changes, and `__pybind11_version__` with whichever
#: pybind11 that configure resolved (GH #288); pybind11-stubgen faithfully copies
#: whatever the just-built module reports — so a stub regenerated mid-change
#: carries one developer's commit, or worse a `+dirty` marker, into a committed
#: file. That is a spurious diff on every rebuild and a claim about a build
#: nobody else made; PR #70 merged an `e61f83d57358+dirty` stamp exactly this
#: way. The pybind11 stamp is the same hazard by construction: it differs between
#: two developers whose environments resolved different pybind11 versions, which
#: is the very condition #288 exists to make visible — visible in the *binary*,
#: not as a committed diff that flips back and forth.
#:
#: `__version__` is stamped from `BNGSIM_VERSION_STR`, so it is not
#: machine-specific — it is *release*-specific, which goes stale on a cadence all
#: its own. A release bumps `pyproject.toml` and nothing rebuilds, so the
#: committed stub keeps describing the previous release until the next person to
#: run this script finds a one-line diff they did not make: 0.13.0 shipped with
#: the stub still reading `'0.12.2'`. Since #31 `pyproject.toml` is the single
#: source of truth for the version and every other anchor derives from it; a
#: snapshot in a generated file is not a derivation, it is a fifth anchor that
#: can only ever be right by coincidence.
#:
#: "unknown" is CMake's own default when the underlying fact is unavailable
#: (CMakeLists.txt), which is precisely the stub's situation: a type stub cannot
#: know what any given build came from. The value in a `.pyi` is documentation —
#: mypy checks the *type* — so nothing downstream reads it, and
#: `test_build_provenance.py` asserts on the runtime attributes of the compiled
#: module, never on these lines. `test_version_consistency.py` still holds the
#: *runtime* `_bngsim_core.__version__` to `pyproject.toml`, which is where that
#: invariant belongs.
_STUB_PER_BUILD_STAMPS = ("__build_commit__", "__pybind11_version__", "__version__")
_STUB_STAMP_PLACEHOLDER = "unknown"
_STUB_STAMP_RE = re.compile(
    rf"^((?:{'|'.join(_STUB_PER_BUILD_STAMPS)}): str = )'[^']*'", re.MULTILINE
)

#: The module's ``HAS_*`` capability flags are per-build facts by the same
#: argument, and the ones GH #459 watched flip: they record which features that
#: particular configure compiled in, so a developer who builds with
#: ``-DBNGSIM_ENABLE_MIR=ON`` regenerates the stub with ``HAS_MIR: bool = True``
#: and is left holding a tracked file no other build agrees with. Keep the
#: declaration — mypy checks the *type*, and nothing reads a stub's value — and
#: drop the value, which is the ``= ...`` every hand-written stub uses. Matched
#: by prefix rather than listed, so a flag added to the bindings is covered the
#: day it appears.
_STUB_CAPABILITY_RE = re.compile(r"^(HAS_[A-Z0-9_]*: bool = )(?:True|False)$", re.MULTILINE)


def _normalize_stub_build_stamps(stub_text: str) -> str:
    """Replace the generated build-describing values with stable placeholders.

    Keeps the committed stub reproducible across machines, commits, dirty working
    trees, build environments, releases — and build configurations. A no-op for
    any stamp or flag the module did not report at all.
    """
    normalized = _STUB_STAMP_RE.sub(
        rf"\1'{_STUB_STAMP_PLACEHOLDER}'",
        stub_text,
    )
    return _STUB_CAPABILITY_RE.sub(r"\1...", normalized)


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

    The generated values that describe the build rather than the API — the
    ``__build_commit__``/``__pybind11_version__``/``__version__`` stamps and the
    ``HAS_*`` capability flags — are normalized away before the stub lands; see
    :func:`_normalize_stub_build_stamps`.
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
        stub_dest.write_text(_normalize_stub_build_stamps(generated.read_text()))
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

    pybind11_cmake_dir = _pybind11_cmake_dir()
    if pybind11_cmake_dir is None:
        # Said before the configure, not after, so the correlation is visible
        # even when cmake goes on to succeed against a system pybind11 — in
        # which case this line is also the only notice that the bindings were
        # built against a copy the project never pinned.
        print(
            "pybind11: not importable in this interpreter; leaving discovery to CMake (GH #229)",
            flush=True,
        )
    else:
        print(f"pybind11: {pybind11_cmake_dir}", flush=True)

    # Said before the configure for the same reason the pybind11 line is: this
    # is the configuration the binary is about to get, and a rebuild that changes
    # what is compiled in should never be something you find out from a diff.
    settings = _resolve_cmake_settings(source_dir)
    print(
        "cmake configuration: "
        + " ".join(f"{name}={setting.value}" for name, setting in settings.items()),
        flush=True,
    )
    for name, setting in settings.items():
        if setting.origin == _ORIGIN_ENVIRONMENT:
            print(f"  {name}={setting.value} (requested via the environment)", flush=True)

    timeout = float(os.environ.get("BNGSIM_REBUILD_LOCK_TIMEOUT", _LOCK_TIMEOUT_SECONDS))
    with _editable_rebuild_lock(build_dir, timeout=timeout):
        # Under the lock, so what we read is what we are about to build from.
        conflicts = _configuration_conflicts(_cmake_cache_values(build_dir, settings), settings)
        if conflicts:
            raise SystemExit(_configuration_conflict_message(build_dir, conflicts))

        configure_cmd = _configure_cmd(
            source_dir,
            build_dir,
            settings=settings,
            pybind11_cmake_dir=pybind11_cmake_dir,
            sdkroot=cmake_sdkroot,
            macos_architectures=macos_architectures,
        )

        try:
            _run(
                configure_cmd,
                env=cmake_env,
            )
        except subprocess.CalledProcessError:
            if pybind11_cmake_dir is not None:
                raise
            # Only reachable with pybind11 absent, so the note is a candidate
            # diagnosis rather than a verdict — it says which failure it
            # explains, and cmake's own error is directly above it.
            raise SystemExit(_PYBIND11_MISSING_NOTE) from None

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
