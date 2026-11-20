"""Write ``llms.txt`` and ``llms-full.txt`` at the root of the HTML build (issue #1699).

An agent with a network reads documentation through a fetch tool, and the pages
Sphinx renders are built for a browser. ``llms.txt`` (https://llmstxt.org) is
the index an agent reads instead: each page, one line on what it answers, and
a link to its Markdown. ``llms-full.txt`` is every page but the optional ones
in one file, for an agent that would rather load it all.

Both are built from the pages themselves, so they cannot drift from them:

* the sections and their order are the root page's toctrees and captions,
  and the captions ``llms_txt_optional_captions`` names make up the spec's
  ``Optional`` section, which an agent short of context may skip;
* each page's line is the ``summary`` of its front matter, and a page without
  one is a warning, which ``sphinx -W`` makes a failed build;
* each page's Markdown is written next to its HTML as ``<page>.md``, with
  MyST's front matter removed and an ``{include}`` replaced by the file it
  includes (the agent guide's page is one: its text is the package's
  ``agent_guide.md``).

Read the Docs serves a project's ``llms.txt`` and ``llms-full.txt`` at the root
of its domain, from the default version's build, so the links are absolute:
``READTHEDOCS_CANONICAL_URL`` where Read the Docs sets it, ``llms_txt_base_url``
otherwise.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

logger = logging.getLogger(__name__)

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)
_INCLUDE = re.compile(r"^```\{include\}\s+(?P<path>\S+)\s*\n(?:[^`\n]*\n)*?```\s*$", re.MULTILINE)


def _summary(env: BuildEnvironment, docname: str) -> str:
    return str(env.metadata.get(docname, {}).get("summary", "")).strip()


def _check_summaries(app: Sphinx, env: BuildEnvironment) -> None:
    for docname in sorted(env.found_docs):
        if not _summary(env, docname):
            logger.warning(
                "page has no 'summary' in its front matter, the line llms.txt gives it",
                location=docname,
            )


def _read(path: Path) -> str:
    """A file's text with its line endings as ``\n``, as a CRLF checkout has them too."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _markdown(env: BuildEnvironment, docname: str) -> str:
    """The page's Markdown as an agent should read it."""
    source = Path(env.doc2path(docname))
    text = _FRONT_MATTER.sub("", _read(source), count=1)

    def include(match: re.Match[str]) -> str:
        target = (source.parent / match.group("path")).resolve()
        return _FRONT_MATTER.sub("", _read(target), count=1).rstrip("\n")

    return _INCLUDE.sub(include, text).rstrip("\n") + "\n"


def _sections(app: Sphinx) -> list[tuple[str, list[str]]]:
    """The root page's toctrees: (caption, docnames), in page order."""
    env = app.env
    root = app.config.root_doc
    sections: list[tuple[str, list[str]]] = [("Overview", [root])]
    for toctree in env.get_doctree(root).findall(addnodes.toctree):
        caption = toctree.get("caption") or "Pages"
        sections.append((caption, [ref for _, ref in toctree["entries"] if ref in env.found_docs]))
    return sections


def _title(env: BuildEnvironment, docname: str) -> str:
    title = env.titles.get(docname)
    return title.astext() if isinstance(title, nodes.Node) else docname


def _base_url(app: Sphinx) -> str:
    base = os.environ.get("READTHEDOCS_CANONICAL_URL") or app.config.llms_txt_base_url
    return base if base.endswith("/") else base + "/"


def _write(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return
    env, out = app.env, Path(app.outdir)
    base = _base_url(app)
    optional = set(app.config.llms_txt_optional_captions)
    sections = _sections(app)

    markdown = {docname: _markdown(env, docname) for docname in env.found_docs}
    for docname, text in markdown.items():
        path = out / f"{docname}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    root = app.config.root_doc
    summary = app.config.llms_txt_summary or _summary(env, root)
    index = [f"# {app.config.project}", "", f"> {summary}", ""]
    index += [*app.config.llms_txt_intro.strip().splitlines(), ""]
    full = [f"# {app.config.project} documentation, every page in one file", ""]
    # The spec's "Optional" section, which an agent short of context may skip,
    # gathers the optional captions' pages, and comes last.
    headings: dict[str, list[str]] = {}
    seen: set[str] = set()
    for caption, docnames in sections:
        lines = headings.setdefault("Optional" if caption in optional else caption, [])
        for docname in docnames:
            if docname in seen:
                continue
            seen.add(docname)
            lines.append(
                f"- [{_title(env, docname)}]({base}{docname}.md): {_summary(env, docname)}"
            )
            if caption not in optional:
                full += [f"<!-- {base}{docname}.md -->", "", markdown[docname]]
    if "Optional" in headings:
        headings["Optional"] = headings.pop("Optional")
    for heading, lines in headings.items():
        index += [f"## {heading}", "", *lines, ""]
    (out / "llms.txt").write_text("\n".join(index).rstrip("\n") + "\n", encoding="utf-8")
    (out / "llms-full.txt").write_text("\n".join(full).rstrip("\n") + "\n", encoding="utf-8")


def setup(app: Sphinx) -> dict[str, object]:
    app.add_config_value("llms_txt_base_url", "", "html", types=frozenset({str}))
    app.add_config_value("llms_txt_summary", "", "html", types=frozenset({str}))
    app.add_config_value("llms_txt_intro", "", "html", types=frozenset({str}))
    app.add_config_value("llms_txt_optional_captions", (), "html", types=frozenset({tuple, list}))
    app.connect("env-check-consistency", _check_summaries)
    app.connect("build-finished", _write)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
