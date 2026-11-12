# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import importlib.util
import os
import sys

import sphinx.util.requests

# Make the package importable for autodoc even when bngsim isn't pip-installed
# in the build environment (e.g. local `sphinx-build`, or the RTD mock path).
sys.path.insert(0, os.path.abspath(os.path.join("..", "python")))

# -- Project information -----------------------------------------------------
project = "bngsim"
author = "William S. Hlavacek"
copyright = "Los Alamos National Laboratory"

# bngsim's own _version.py, loaded by path: it asks the installed package's
# metadata and falls back to pyproject.toml, which is where Read the Docs and
# lint.yml's docs job find it, since neither installs bngsim. Importing the
# bngsim package here instead would import the compiled core, which neither
# builds. Without the fallback the title read "bngsim 0.0".
_version_spec = importlib.util.spec_from_file_location(
    "_bngsim_version", os.path.abspath(os.path.join("..", "python", "bngsim", "_version.py"))
)
_version_module = importlib.util.module_from_spec(_version_spec)
_version_spec.loader.exec_module(_version_module)
release = _version_module.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # Markdown (MyST) source pages
    "sphinx.ext.autodoc",  # pull docstrings from the package
    "sphinx.ext.napoleon",  # NumPy/Google-style docstrings
    "sphinx.ext.intersphinx",  # cross-link to NumPy/pandas/etc. docs
    "sphinx.ext.viewcode",  # "[source]" links
    "sphinx_autodoc_typehints",  # render type hints in signatures
    "sphinx_copybutton",  # copy button on code blocks
]

# the logo dir is assets, not doc pages.
exclude_patterns = ["bngsim-coyote-logo/*", "_build", "requirements.txt"]

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "substitution", "tasklist"]
# NB: dollarmath is intentionally OFF — pages contain shell `$VARS` and `$(...)`
# that must not be parsed as math.
myst_heading_anchors = 4  # auto-anchor h1-h4 so cross-page #links resolve

# -- autodoc -----------------------------------------------------------------
# Read the Docs and lint.yml's docs job install no bngsim, so the compiled C++
# core is absent there and is mocked; only the binary is mocked, so the
# pure-Python wrappers still import for autodoc (see .readthedocs.yaml). A
# local build in a venv with bngsim installed imports the real core.
autodoc_mock_imports = []
try:
    import bngsim._bngsim_core  # noqa: F401
except Exception:  # pragma: no cover - docs-build fallback
    autodoc_mock_imports = ["bngsim._bngsim_core"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
always_document_param_types = True

# An "Attributes" section renders as :ivar: fields, not as `.. attribute::`
# directives: autodoc already documents each of those attributes from the class,
# and a second directive for the same name is a duplicate-object warning now
# that the API page is indexed (issue #586).
napoleon_use_ivar = True

# -- cross-references --------------------------------------------------------
# Warn on a reference that resolves to nothing, so `-W` fails on a :meth: that
# names a renamed API (issue #586). nitpick-ignore.txt lists the references that
# cannot resolve (words of a NumPy-style type, undocumented modules).
nitpicky = True
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nitpick-ignore.txt")) as f:
    nitpick_ignore = [
        tuple(line.split(None, 1))
        for line in (raw.strip() for raw in f)
        if line and not line.startswith("#")
    ]

# -- intersphinx -------------------------------------------------------------
_INTERSPHINX_URLS = {
    "python": "https://docs.python.org/3/",
    "numpy": "https://numpy.org/doc/stable/",
    "pandas": "https://pandas.pydata.org/docs/",
    "scipy": "https://docs.scipy.org/doc/scipy/",
}
intersphinx_timeout = 30


def _inventory(name: str, url: str) -> str | None:
    """Where intersphinx reads ``name``'s inventory from.

    With BNGSIM_INTERSPHINX_CACHE unset, ``None``: intersphinx fetches ``url``
    itself. Set to a directory, as lint.yml's docs job sets it, a copy there,
    refreshed from ``url`` and kept as it was if the host does not answer.
    Under -W an unreachable inventory is a failed build, so without the copy an
    outage at numpy.org would fail every pull request (issue #586).
    """
    cache = os.environ.get("BNGSIM_INTERSPHINX_CACHE")
    if not cache:
        return None
    path = os.path.join(os.path.expanduser(cache), f"{name}.inv")
    try:
        # Sphinx's own client, as intersphinx fetches with: numpy.org refuses
        # urllib's default User-Agent.
        with sphinx.util.requests.get(url + "objects.inv", timeout=intersphinx_timeout) as r:
            r.raise_for_status()
            data = r.content
        if not data.startswith(b"# Sphinx inventory version"):
            raise ValueError("not a Sphinx inventory")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as out:
            out.write(data)
        os.replace(path + ".tmp", path)
    except (OSError, ValueError) as exc:  # requests' errors are OSErrors
        state = "keeping the cached copy" if os.path.exists(path) else "no cached copy"
        print(f"intersphinx: {name}: {url} failed ({exc}); {state}")
    return path if os.path.exists(path) else None


intersphinx_mapping = {
    name: (url, _inventory(name, url)) for name, url in _INTERSPHINX_URLS.items()
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"bngsim {version}"
html_logo = "bngsim-coyote-logo/bngsim-logo.svg"
html_favicon = "bngsim-coyote-logo/bngsim-favicon.svg"
html_static_path = []
