"""Parsing helpers for the workflow files that CI-coverage tests assert over.

Two test modules read ``.github/workflows/*.yml`` as text and ask the same
questions of it: what does this job fire on, what does it actually run, and can
what it runs go green having executed nothing. They lived as two private copies
of the same five parsers, which is the exact drift shape those tests exist to
catch — so the parsers are here, once.

Text, not a YAML parse, and deliberately: the properties under test are about
*comments* (a paths filter carries its rationale in them, and issue #295's
exclusion list is expressed in them) and about the shell continuation lines of a
``run:`` block, neither of which survives a structural load.
"""

from __future__ import annotations

import re
from pathlib import Path

try:  # 3.10 is still supported and has no tomllib
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
TESTS_DIR = Path(__file__).resolve().parent

#: Distributions whose import name differs from their PyPI name. Only the ones
#: this project actually depends on; a general resolver would need the metadata
#: of an installed wheel, which is not available for a package CI never installs.
_IMPORT_NAMES = {"python-libsbml": "libsbml", "pytest-cov": "pytest_cov"}


def strip_comments(text: str) -> str:
    """Drop whole-line ``#`` comments.

    Line-based, like the #178 and #269 tests' helpers, and for the same reason:
    these workflows carry their rationale in whole-line comments that name the
    exact filenames and tokens being asserted on, so a substring search over the
    raw text would read the prose as configuration.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def continued_block(lines: list[str], start: int) -> list[str]:
    """``lines[start]`` plus every line joined to it by a trailing backslash."""
    block = [lines[start]]
    while block[-1].rstrip().endswith("\\") and start + len(block) <= len(lines) - 1:
        block.append(lines[start + len(block)])
    return block


def paths_filter_selectors(body: str) -> list[str]:
    """The ``python/tests/`` entries of the ``paths:`` filter (YAML list items).

    Quotes are optional because YAML's are: every entry in these files is
    quoted today, and a parser that required it would read an unquoted one as
    absent -- silently shrinking its own view of what a leg fires on, which is
    the one failure mode a coverage test cannot afford. Found by mutation.
    """
    return re.findall(r"""^\s*-\s*["']?(python/tests/[^"'\s]+)["']?\s*$""", body, re.M)


def pytest_selectors(body: str) -> list[str]:
    """The ``python/tests/`` arguments the run step hands to pytest."""
    lines = body.splitlines()
    start = next((i for i, line in enumerate(lines) if "python -m pytest" in line), None)
    assert start is not None, "workflow no longer invokes pytest"
    return [
        token
        for line in continued_block(lines, start)
        for token in line.split()
        if token.startswith("python/tests/")
    ]


def expand(selectors: list[str]) -> set[str]:
    """Resolve selectors (literal or globbed) against the checkout, as bash would."""
    resolved: set[str] = set()
    for selector in selectors:
        parent, _, pattern = selector.rpartition("/")
        resolved.update(path.name for path in (REPO_ROOT / parent).glob(pattern))
    return resolved


def pip_install_text(body: str) -> str:
    """Every ``pip install`` command in the workflow, line continuations included."""
    lines = body.splitlines()
    return "\n".join(
        line
        for i, _ in enumerate(lines)
        if "pip install" in lines[i]
        for line in continued_block(lines, i)
    )


def _pyproject() -> dict:
    """``pyproject.toml``, parsed.

    Callers reach the project metadata only from the provisioning checks, and
    those guard with ``pytest.importorskip("tomllib")`` -- so the 3.10 path is a
    declared skip of one test rather than an import error that takes this whole
    module (and every filename assertion in it) out of collection.
    """
    assert tomllib is not None, "tomllib is 3.11+; guard the caller with importorskip"
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _optional_dependencies() -> dict[str, list[str]]:
    return _pyproject().get("project", {}).get("optional-dependencies", {})


def _requirement_name(spec: str) -> str:
    """``"antimony>=3.1.3"`` -> ``antimony``; ``"bngsim[test,jax]"`` -> ``bngsim``."""
    return re.split(r"[<>=!~\[;\s]", spec.strip().strip("\"'"), maxsplit=1)[0].lower()


def provisioned_roots(body: str) -> set[str]:
    """Import roots a workflow's ``pip install`` lines make available.

    Extras are expanded from ``pyproject.toml`` rather than matched as text:
    ``pip install ".[test]"`` provisions scipy and antimony without either name
    appearing in the workflow, and a check that missed that would demand a
    redundant explicit install to stay quiet.
    """
    text = pip_install_text(body)
    extras = _optional_dependencies()
    roots: set[str] = set()
    pending = [
        stripped
        for group in re.findall(r"\[([A-Za-z0-9_,\s-]+)\]", text)
        for name in group.split(",")
        if (stripped := name.strip()) in extras
    ]
    seen: set[str] = set()
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for spec in extras[extra]:
            name = _requirement_name(spec)
            if name == "bngsim":
                pending += [e.strip() for e in re.findall(r"\[([^\]]+)\]", spec)[0].split(",")]
            else:
                roots.add(_IMPORT_NAMES.get(name, name))
    # Bare names on the pip command line, e.g. `pip install "scipy>=1.15"`.
    for spec in re.findall(r'["\']([A-Za-z0-9_.-]+(?:[<>=!~][^"\']*)?)["\']', text):
        name = _requirement_name(spec)
        if name:
            roots.add(_IMPORT_NAMES.get(name, name))
    return roots


def base_roots() -> set[str]:
    """Import roots present without the workflow installing anything.

    bngsim's own ``[project] dependencies`` (every job pip-installs bngsim) plus
    bngsim and pytest. Read from ``pyproject.toml`` so promoting an extra to a
    base dependency does not need an edit here to stay accurate.
    """
    roots = {"bngsim", "pytest"}
    for spec in _pyproject().get("project", {}).get("dependencies", []):
        name = _requirement_name(spec)
        roots.add(_IMPORT_NAMES.get(name, name))
    return roots


#: A module-scope ``pytest.importorskip`` — no leading whitespace, so it takes
#: the whole file out of collection rather than costing one visible test.
MODULE_IMPORTORSKIP = re.compile(r'^[^\s#].*importorskip\(\s*["\']([\w.]+)["\']', re.M)


def module_level_optional_imports(filename: str) -> list[str]:
    """Root modules ``filename`` importorskips at module scope."""
    text = (TESTS_DIR / filename).read_text(encoding="utf-8")
    return [name.split(".")[0] for name in MODULE_IMPORTORSKIP.findall(text)]
