"""Shared report helpers for the bngsim benchmark suites.

Each ``suites/<name>/emit.py`` reads its own ``results/*.json`` and
writes a neutral Markdown report fragment to
``bngsim/benchmarks/reports/generated/<name>.md``.  Set
``BNGSIM_BENCH_LATEX_DIR`` to mirror the legacy LaTeX fragment into a
private paper checkout.

The contract is intentionally narrow:

* missing or non-finite numeric → :data:`TBD` (``\\textit{TBD}``);
* emitters never touch paper source files directly;
* output is overwritten atomically; an emitter must always produce a report
  even when ``results/*.json`` is empty or missing.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

#: ``bngsim/benchmarks/``
BENCH_ROOT = Path(__file__).resolve().parent

#: repo root (bngsim/)
REPO_ROOT = BENCH_ROOT.parents[0]

#: Public report fragments generated from suite results.
REPORT_DIR = BENCH_ROOT / "reports" / "generated"

#: Optional private-paper export destination.  When set, ``write_generated``
#: mirrors the raw LaTeX fragment to ``$BNGSIM_BENCH_LATEX_DIR/<name>.tex``.
LATEX_DIR_ENV = "BNGSIM_BENCH_LATEX_DIR"

# ---------------------------------------------------------------------------
# Placeholder + formatting helpers
# ---------------------------------------------------------------------------

#: Standard placeholder for a cell whose data is missing.
TBD = r"\textit{TBD}"


def _is_missing(v) -> bool:
    if v is None:
        return True
    return isinstance(v, float) and not math.isfinite(v)


def fmt_time(v, *, decimals: int = 3) -> str:
    """Format a wall-clock value (seconds) or return :data:`TBD`."""
    if _is_missing(v):
        return TBD
    if v < 0.001:
        return r"$<$0.001"
    return f"{v:.{decimals}f}"


def fmt_obj(v) -> str:
    """Format an objective-function value (3 sig figs) or :data:`TBD`."""
    if _is_missing(v):
        return TBD
    return f"{v:.3g}"


def fmt_speedup(t_ref, t_new) -> str:
    """``t_ref / t_new`` rendered as ``N.NN$\\times$``; :data:`TBD` if either side missing."""
    if _is_missing(t_ref) or _is_missing(t_new) or t_new == 0:
        return TBD
    return f"{t_ref / t_new:.2f}$\\times$"


def fmt_pct(v) -> str:
    """Render a 0..100 percentage as ``NN.N\\%``; :data:`TBD` if missing."""
    if _is_missing(v):
        return TBD
    return f"{v:.1f}\\%"


def fmt_int(v) -> str:
    if _is_missing(v):
        return TBD
    return f"{int(v)}"


def geomean(xs) -> float | None:
    """Return the geometric mean of ``xs``, or ``None`` if empty / non-positive."""
    pos = [x for x in xs if x is not None and isinstance(x, (int, float)) and x > 0]
    if not pos:
        return None
    return math.exp(sum(math.log(x) for x in pos) / len(pos))


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?")
_LATEX_BRACED_COMMAND_RE = re.compile(r"\\(?:textit|textbf|emph|mathrm)\{([^{}]*)\}")
_MULTICOLUMN_RE = re.compile(r"\\multicolumn\{\d+\}\{[^{}]*\}\{(.+)\}$")
_CAPTION_RE = re.compile(r"\\caption\{(.+?)\}", re.DOTALL)
_LABEL_RE = re.compile(r"\\label\{(.+?)\}")
_INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{(.+?)\}")


def _strip_latex_cell(text: str) -> str:
    """Best-effort cleanup for LaTeX table cells rendered into Markdown."""
    text = text.strip()
    if not text:
        return ""
    text = _MULTICOLUMN_RE.sub(lambda m: m.group(1), text)
    prev = None
    while prev != text:
        prev = text
        text = _LATEX_BRACED_COMMAND_RE.sub(lambda m: m.group(1), text)
    replacements = {
        r"\_": "_",
        r"\%": "%",
        r"\ ": " ",
        r"\times": "x",
        r"\le": "<=",
        r"\ge": ">=",
        r"\pm": "+/-",
        r"\\": "",
        "~": " ",
        "$": "",
        "`": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = _LATEX_COMMAND_RE.sub("", text)
    text = text.replace("{", "").replace("}", "")
    return " ".join(text.split()).strip()


def _split_latex_row(line: str) -> list[str] | None:
    line = line.strip()
    if "&" not in line:
        return None
    if line.endswith(r"\\"):
        line = line[:-2]
    line = line.rstrip("\\").strip()
    cells = [_strip_latex_cell(c) for c in line.split("&")]
    return cells if cells else None


def _latex_table_to_markdown(body: str, *, title: str) -> str:
    rows = []
    notes = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith(r"\multicolumn") and "&" not in line:
            notes.append(_strip_latex_cell(line.rstrip("\\")))
            continue
        cells = _split_latex_row(line)
        if cells is not None and any(cells) and cells[0]:
            rows.append(cells)

    if not rows:
        return f"# {title}\n\n_No tabular rows were rendered._\n"

    ncols = max(len(r) for r in rows)
    normalized = [r + [""] * (ncols - len(r)) for r in rows]
    header = normalized[0]
    data = [row for row in normalized[1:] if row != header]
    lines = [f"# {title}", ""]
    if notes:
        lines.extend(f"**{note}**" for note in notes)
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in data)
    return "\n".join(lines) + "\n"


def _latex_figure_to_markdown(body: str, *, title: str) -> str:
    image = _INCLUDE_RE.search(body)
    caption = _CAPTION_RE.search(body)
    label = _LABEL_RE.search(body)
    image_path = _strip_latex_cell(image.group(1)) if image else ""
    caption_text = _strip_latex_cell(caption.group(1)) if caption else title
    lines = [f"# {title}", ""]
    if image_path:
        lines.append(f"![{caption_text}]({image_path})")
        lines.append("")
    lines.append(caption_text)
    if label:
        lines.extend(["", f"Label: `{label.group(1)}`"])
    return "\n".join(lines) + "\n"


def latex_fragment_to_markdown(name: str, body: str) -> str:
    title = name.replace("_", " ").title()
    if r"\begin{figure}" in body:
        return _latex_figure_to_markdown(body, title=title)
    return _latex_table_to_markdown(body, title=title)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_results(path: Path):
    """Parse ``path`` as JSON; return ``{}`` (or ``[]`` for list-shaped
    inputs is the caller's responsibility) when the file is absent.

    Missing-results is a normal state in Phase 5 -- the suite has not been
    run yet, and the emitter is expected to render a TBD-only fragment.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_generated(name: str, body: str) -> Path:
    """Write a Markdown report and optionally mirror the raw LaTeX fragment.

    Returns the public Markdown path.  If ``BNGSIM_BENCH_LATEX_DIR`` is set,
    the unmodified LaTeX fragment is also written to ``<dir>/<name>.tex`` for
    private paper tooling.
    """
    if not body.endswith("\n"):
        body = body + "\n"

    report = latex_fragment_to_markdown(name, body)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{name}.md"
    out.write_text(report)

    latex_dir = os.environ.get(LATEX_DIR_ENV)
    if latex_dir:
        tex_out = Path(latex_dir).expanduser() / f"{name}.tex"
        tex_out.parent.mkdir(parents=True, exist_ok=True)
        tex_out.write_text(body)
    return out


# ---------------------------------------------------------------------------
# Per-row audience filtering (paper_role)
# ---------------------------------------------------------------------------

#: Audience values accepted by :func:`filter_by_role` and the suites'
#: ``emit.py --audience`` flag.  Order is documentation-only.
AUDIENCES = ("main", "supp", "all")

#: Filename suffix added by :func:`audience_suffix` for each audience.
#: ``"all"`` -> no suffix (the default Phase-5 filename), so emitters
#: that never get a non-default audience produce byte-identical output.
_AUDIENCE_SUFFIX = {"all": "", "main": "_main", "supp": "_supp"}


def audience_suffix(audience: str) -> str:
    """Return the filename suffix for ``audience`` (``""`` for ``"all"``).

    Used by emitters to compose the output stem -- ``write_generated(name +
    audience_suffix(args.audience), body)`` -- so the default audience writes
    ``generated/<name>.md`` and the non-default variants land next to it as
    ``generated/<name>_main.md`` / ``<name>_supp.md``.
    """
    if audience not in _AUDIENCE_SUFFIX:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    return _AUDIENCE_SUFFIX[audience]


def _row_role(row) -> str:
    """Return the ``paper_role`` of a row, defaulting to ``"supp"``.

    ``row`` is any dict-like (a MODELS entry, a parsed results-JSON
    record) **or** an attribute-bearing object (a dataclass instance --
    fitting's ``Problem``).  A missing or ``None`` role is treated as
    ``"supp"`` so the supplementary table is the default audience.
    """
    role = row.get("paper_role") if hasattr(row, "get") else getattr(row, "paper_role", None)
    return role if role else "supp"


def _keep_role(role: str, audience: str) -> bool:
    """Whether a row tagged ``role`` is included for ``audience``."""
    if role == "skip":
        return False
    if audience == "all":
        return True
    if audience == "main":
        return role == "main"
    return role in ("main", "supp")  # audience == "supp"


def filter_by_role(rows, audience: str) -> list:
    """Filter ``rows`` by ``paper_role`` for the given ``audience``.

    ``audience`` is one of :data:`AUDIENCES`:

    * ``"all"``  -- every row whose ``paper_role`` is not ``"skip"``.
      This is the default and reproduces the Phase-5 behavior whenever
      no row is tagged.
    * ``"main"`` -- rows whose ``paper_role == "main"``.
    * ``"supp"`` -- rows whose ``paper_role`` is ``"main"`` or
      ``"supp"`` (supp is a *superset* of main -- the full
      supplementary table includes the main-text subset).

    A row with no ``paper_role`` (or ``None``) is treated as ``"supp"``.
    Rows tagged ``"skip"`` are always dropped, regardless of audience.

    The input may be any iterable of dict-likes (or dataclass-likes);
    the return is a list preserving the input order.
    """
    if audience not in _AUDIENCE_SUFFIX:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    return [row for row in rows if _keep_role(_row_role(row), audience)]


def filter_items_by_role(items, audience: str) -> list:
    """Like :func:`filter_by_role`, but for ``(key, value)`` sequences.

    Reads ``paper_role`` off each value (matching dict-shaped registries
    like ``nf/run.py``'s ``CORE_MODELS = {name: spec, ...}``).  Returns
    the kept ``(key, value)`` pairs in input order.
    """
    if audience not in _AUDIENCE_SUFFIX:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    return [(k, v) for k, v in items if _keep_role(_row_role(v), audience)]


def add_audience_arg(parser, *, default: str = "all") -> None:
    """Register the standard ``--audience {main,supp,all}`` option on ``parser``.

    Phase-6 contract: every table-rendering ``emit.py`` accepts the flag.
    The default is ``"all"`` -- the Phase-5 behavior, byte-identical
    output when no row is tagged.  ``"main"`` / ``"supp"`` write to a
    filename-suffixed variant alongside the default file.
    """
    if default not in _AUDIENCE_SUFFIX:
        raise ValueError(f"default must be one of {AUDIENCES}, got {default!r}")
    parser.add_argument(
        "--audience",
        choices=AUDIENCES,
        default=default,
        help=(
            "Which paper_role rows to render. 'all' (default) keeps every "
            "row not tagged 'skip' and writes generated/<name>.md; 'main' "
            "keeps only paper_role='main' rows and writes generated/<name>_main.md; "
            "'supp' keeps 'main' + 'supp' rows and writes generated/<name>_supp.md."
        ),
    )
