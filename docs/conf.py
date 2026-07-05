# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import importlib.metadata
import os
import sys

# Make the package importable for autodoc even when bngsim isn't pip-installed
# in the build environment (e.g. local `sphinx-build`, or the RTD mock path).
sys.path.insert(0, os.path.abspath(os.path.join("..", "python")))

# -- Project information -----------------------------------------------------
project = "bngsim"
author = "William S. Hlavacek"
copyright = "Los Alamos National Laboratory"

try:
    release = importlib.metadata.version("bngsim")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
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
myst_heading_anchors = 3  # auto-anchor h1-h3 so cross-page #links resolve

# -- autodoc -----------------------------------------------------------------
# On Read the Docs the compiled C++ core is built and installed (see
# .readthedocs.yaml). If for some reason it is unavailable, mock only the
# binary so the pure-Python wrappers still import for autodoc.
autodoc_mock_imports = []
try:
    import bngsim._bngsim_core  # noqa: F401
except Exception:  # pragma: no cover - docs-build fallback
    autodoc_mock_imports = ["bngsim._bngsim_core"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
always_document_param_types = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"bngsim {version}"
html_logo = "bngsim-coyote-logo/bngsim-logo.svg"
html_favicon = "bngsim-coyote-logo/bngsim-favicon.svg"
html_static_path = []
