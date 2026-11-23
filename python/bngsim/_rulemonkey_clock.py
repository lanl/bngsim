"""Run a clock-reading model on RuleMonkey on elapsed time (issue #1063).

Since #1022, ``method="rm"`` runs a model that reads the clock (``time()``, a
bare ``time``, or a table function indexed by time) on absolute model time, as
``ode`` and ``ssa`` do. RuleMonkey 3.11.0 placed each event at ``t + tau`` in
doubles, so from a large ``t_start`` the waiting times rounded away once the
total propensity times |t| neared 1e15, and the run was biased with no error: from
``t_start = 1e12``, ``A(s~0) -> A(s~1)`` at 0.5 from 1e5 copies left 22743 at
``1e12 + 2`` against the binomial mean 36788 (z = -291). A model that reads no
clock runs on elapsed time and is exact (#1060).

So a model that reads the clock through ``time()`` runs on elapsed time too,
and gets absolute time back through the model. :func:`offset_clock_xml` writes
a copy of the BNG XML in which every function reads the clock as
``(time()+__bngsim_t0)``, and the caller sets ``__bngsim_t0`` to the run's
``t_start`` before it runs. RuleMonkey resolves a parameter from the model's
parameter table wherever an expression names it (``<ListOfReferences>`` lists
only observables for it). Output functions are evaluated through the same
rewritten expressions, so they report on absolute time as before.

The clock is rewritten wherever RuleMonkey reads it and nowhere else
(:func:`~bngsim._network_free_checks.rulemonkey_clock_ref`): ``time()``, a bare
``time``, and its alias ``t``, except that a bare ``time`` or ``t`` that names a
model symbol reads the symbol. BNG2.pl accepts an observable named ``time``, and
turning that into the clock changed the rates of a model that reads no clock.

A table function indexed by the clock is not rewritten, and a model with one
runs on model time (the caller checks
:func:`~bngsim._network_free_checks.time_indexed_table_functions`). It needs no
offset: since 3.12.0 RuleMonkey places its events, and its panels for a rate
that reads the clock, on the time elapsed since each call starts
(richardposner/RuleMonkey#110), so such a run keeps its waiting times from a
large ``t_start`` (issue #1063). Offsetting it would have indexed the table
through a function, which 3.11.0 could not evaluate
(richardposner/RuleMonkey#111).
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from xml.sax.saxutils import escape

from bngsim._exceptions import ModelError
from bngsim._network_free_checks import (
    _function_elements,
    _indexed_by_time,
    _local,
    rulemonkey_clock_ref,
    rulemonkey_symbols,
)

# The parameter the run's t_start is written to.
OFFSET_PARAM = "__bngsim_t0"

# A <Function> element with a body; a self-closing one has nothing to rewrite.
_FUNCTION = re.compile(r"<Function\b(?P<attrs>[^>]*)(?<!/)>(?P<body>.*?)</Function>", re.DOTALL)
_EXPRESSION = re.compile(r"(<Expression>)(.*?)(</Expression>)", re.DOTALL)
_ID = re.compile(r"""\bid\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_FILE = re.compile(r"""\bfile\s*=\s*(?:"[^"]*"|'[^']*')""")


def _table_file(xml_dir: str, file_attr: str) -> str:
    """The table's data file as RuleMonkey finds it beside the original XML
    (``<xml_dir>/<file>``, then ``<xml_dir>/../<file>``, each normalized
    lexically, as load_model in simulator.cpp does), made absolute so the copy
    written elsewhere finds the same one."""
    if os.path.isabs(file_attr):
        return file_attr
    first = os.path.normpath(os.path.join(xml_dir, file_attr))
    second = os.path.normpath(os.path.join(xml_dir, "..", file_attr))
    return second if not os.path.exists(first) and os.path.exists(second) else first


def offset_clock_text(xml: str, xml_dir: str | Path) -> str:
    """*xml* with the model's clock read as ``time()+__bngsim_t0``, its table
    files found from *xml_dir*. No table function in *xml* may be indexed by the
    clock."""
    root = ET.fromstring(xml.encode("utf-8", "surrogateescape"))
    symbols = rulemonkey_symbols(root)
    if OFFSET_PARAM in symbols:
        raise ModelError(
            f"the BNG XML already has a symbol named {OFFSET_PARAM!r}, which bngsim "
            "uses to run a model that reads the clock on RuleMonkey. "
            "Rename it."
        )
    functions = {f.get("id", ""): f for f in _function_elements(root)}
    if any(_indexed_by_time(f) for f in functions.values()):
        raise ValueError("a table function indexed by the clock is not rewritten")
    clock = rulemonkey_clock_ref(symbols)
    offset = f"(time()+{OFFSET_PARAM})"
    xml_dir = os.path.abspath(xml_dir)

    def function(m: re.Match[str]) -> str:
        attrs, body = m.group("attrs"), m.group("body")
        fid = _ID.search(attrs)
        el = functions.get(fid.group(1) or fid.group(2)) if fid else None
        file_attr = "" if el is None else (el.get("file") or "").strip()
        if file_attr and el is not None and (el.get("type") or "").lower() == "tfun":
            # The copy is written elsewhere: its data file, where RuleMonkey
            # found it beside the original.
            path = escape(_table_file(xml_dir, file_attr), {'"': "&quot;"})
            attrs = _FILE.sub(lambda _: f'file="{path}"', attrs, count=1)
        body = _EXPRESSION.sub(
            lambda e: e.group(1) + clock.sub(offset, e.group(2)) + e.group(3), body
        )
        return f"<Function{attrs}>{body}</Function>"

    xml = _FUNCTION.sub(function, xml)

    parameter = f'<Parameter id="{OFFSET_PARAM}" type="Constant" value="0" expr="0"/>'
    if "<ListOfParameters>" in xml:
        xml = xml.replace("<ListOfParameters>", f"<ListOfParameters>\n      {parameter}", 1)
    elif "<ListOfParameters/>" in xml:
        listed = f"<ListOfParameters>\n      {parameter}\n    </ListOfParameters>"
        xml = xml.replace("<ListOfParameters/>", listed, 1)
    else:
        raise ModelError("the BNG XML has no <ListOfParameters> to add the clock's offset to")
    return xml


def evaluate_symbols(xml_path: str | Path) -> set[str]:
    """The names RuleMonkey's ``evaluate_expression`` binds ahead of its clock
    in *xml_path*: a parameter, observable or function of the name, except that
    a parameter named ``t`` leaves ``t`` the clock there, since the engine binds
    the clock's alias after the parameters (Engine::evaluate_expression in
    engine.cpp)."""
    root = ET.parse(str(xml_path)).getroot()
    symbols = rulemonkey_symbols(root)
    if not any(
        (el.get("name") if _local(el.tag) == "Observable" else el.get("id")) == "t"
        for el in root.iter()
        if _local(el.tag) in ("Observable", "Function")
    ):
        symbols.discard("t")  # a parameter t, if any, does not hide the clock here
    return symbols


# A RuleMonkey state file: its format marker, then the molecule-type schema's
# fingerprint, then the clock (Engine::write_state in engine.cpp).
_STATE_MARKER = "RM_STATE_V2"
_STATE_CLOCK_LINE = 2


def state_to_model_time(path: str | Path, t0: float) -> None:
    """Write the clock of the state file at *path*, saved on time elapsed since
    *t0*, as the model's time, as a session on model time saves it."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="surrogateescape").split("\n")
    if len(lines) <= _STATE_CLOCK_LINE or lines[0].strip() != _STATE_MARKER:
        raise RuntimeError(f"unexpected RuleMonkey state file format in {path}")
    lines[_STATE_CLOCK_LINE] = repr(t0 + float(lines[_STATE_CLOCK_LINE]))
    path.write_text("\n".join(lines), encoding="utf-8", errors="surrogateescape")


@contextlib.contextmanager
def state_at_elapsed_zero(path: str | Path) -> Iterator[tuple[str, float]]:
    """The state file at *path*, with its clock moved to 0 in a temporary copy,
    and the model time it held: a session loads the copy on elapsed time from
    there. A file this cannot read is handed through as it is, at 0, for
    RuleMonkey to reject in its own words."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="surrogateescape").split("\n")
        if len(lines) <= _STATE_CLOCK_LINE or lines[0].strip() != _STATE_MARKER:
            raise ValueError
        t0 = float(lines[_STATE_CLOCK_LINE])
    except (OSError, ValueError):
        yield str(path), 0.0
        return
    lines[_STATE_CLOCK_LINE] = "0"
    with tempfile.TemporaryDirectory(prefix="bngsim-rm-state-") as tmp:
        out = Path(tmp) / "state"
        out.write_text("\n".join(lines), encoding="utf-8", errors="surrogateescape")
        yield str(out), t0


@contextlib.contextmanager
def offset_clock_xml(xml_path: str | Path) -> Iterator[str]:
    """The path of :func:`offset_clock_text` of *xml_path*, written to a
    temporary directory that is removed on exit. RuleMonkey reads the XML and
    its table files once, when it is built, so build it inside the block."""
    source = Path(xml_path)
    # Undecodable bytes pass through as they are; RuleMonkey reads bytes.
    text = source.read_bytes().decode("utf-8", "surrogateescape")
    text = offset_clock_text(text, os.path.dirname(os.path.abspath(source)))
    with tempfile.TemporaryDirectory(prefix="bngsim-rm-clock-") as tmp:
        out = Path(tmp) / source.name
        out.write_bytes(text.encode("utf-8", "surrogateescape"))
        yield str(out)
