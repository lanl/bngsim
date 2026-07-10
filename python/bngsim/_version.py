"""Single source of truth for ``bngsim.__version__`` (issue #31).

The literal version string lives in ``pyproject.toml``. This module
exposes it at runtime by asking ``importlib.metadata`` for the
installed package version, falling back to parsing ``pyproject.toml``
directly when the package is being imported from a source tree
without a wheel install (e.g. ``PYTHONPATH=python`` test rigs).
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


def _read_pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        return "0.0.0+unknown"
    match = re.search(
        r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+(?:[-+.][a-zA-Z0-9.+-]+)?)"\s*$',
        pyproject.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else "0.0.0+unknown"


try:
    __version__: str = _pkg_version("bngsim")
except PackageNotFoundError:
    __version__ = _read_pyproject_version()
