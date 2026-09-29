"""Checks a BNG XML must pass before a network-free backend is handed it.

Issue #862: a rate law that reads ``time()`` is wrong or refused on both
network-free backends. RuleMonkey treats a total propensity of 0 as absorbing,
so ``if(time()>=6, k, 0)`` never fires and the run returns the initial state at
every row with no warning (richardposner/RuleMonkey#86). NFsim cannot prepare
such a function and fails at setup with ``Error preparing function f in class
GlobalFunction``, which names neither the cause nor an alternative. Both are
refused here, before the backend is built, naming the functions. The clock is
read under either spelling, ``time()`` or a bare ``time``, and both are refused.

A table function indexed by time (``type="TFUN"`` with ``ctrName="time"``)
keeps its clock in an attribute, and its ``<Expression>`` reads a placeholder.
RuleMonkey reads it through the same zero-propensity trap, so RuleMonkey refuses
it too. NFsim runs it, but refreshes its value only when a reaction fires, so a
rate that is 0 at the start is not seen either (issue #892). That is left to
the fix there, since refusing it would take away a table function NFsim reads
correctly whenever reactions fire often.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bngsim._exceptions import ModelError

# `time()` or a bare `time`: the engine binds the clock under both spellings, and
# RuleMonkey returned the same flat run for `if(time>=6,k,0)` as for `time()`.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?![A-Za-z0-9_])")

# The counter names RuleMonkey reads as the clock for a table function, checked
# before any parameter or observable of that name (resolve_tfun_counter in
# third_party/rulemonkey/cpp/rulemonkey/simulator.cpp).
_RULEMONKEY_CLOCK_COUNTERS = frozenset({"time", "t", "time()", "t()"})

_BACKENDS = {
    "rulemonkey": (
        "RuleMonkey (method='rm' / 'nf_exact')",
        "RuleMonkey treats a total propensity of 0 as absorbing, so a rate that is 0 at "
        "the start and turns on later never fires, and the run returns the initial "
        "state with no warning (richardposner/RuleMonkey#86)",
    ),
    "nfsim": (
        "NFsim (method='nf' / 'nf_reject')",
        "NFsim cannot prepare a function that reads time() and fails at setup",
    ),
}


def _functions(xml_path: str | Path) -> list[ET.Element] | None:
    """Every ``<Function>`` element in *xml_path*, or ``None`` when the file
    cannot be parsed: the backend then reports its own parse error, which is
    more specific than anything these checks could say."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    return [
        el
        for el in root.iter()
        if el.tag.endswith("Function") and not el.tag.endswith("ListOfFunctions")
    ]


def _reads_time(func: ET.Element) -> bool:
    return any(
        expr.tag.endswith("Expression") and _TIME_REF.search(expr.text or "") for expr in func
    )


def _indexed_by_time(func: ET.Element) -> bool:
    return (func.get("type") or "").lower() == "tfun" and (
        (func.get("ctrName") or "").strip() in _RULEMONKEY_CLOCK_COUNTERS
    )


def time_reading_functions(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* whose expression reads the clock,
    as ``time()`` or as a bare ``time``. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _reads_time(f)]


def time_indexed_table_functions(xml_path: str | Path) -> list[str]:
    """The ids of the table functions in *xml_path* whose counter RuleMonkey
    reads as the clock. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _indexed_by_time(f)]


def refuse_time_dependent_rate_laws(xml_path: str | Path, backend: str) -> None:
    """Raise :class:`ModelError` if *xml_path* has a function reading the clock,
    or, for RuleMonkey, a table function indexed by it.

    *backend* is ``"rulemonkey"`` or ``"nfsim"``.
    """
    funcs = _functions(xml_path) or []
    reads = [f.get("id", "?") for f in funcs if _reads_time(f)]
    tables = []
    if backend == "rulemonkey":
        tables = [f.get("id", "?") for f in funcs if _indexed_by_time(f) and not _reads_time(f)]
    if not reads and not tables:
        return
    label, why = _BACKENDS[backend]
    parts = []
    if reads:
        parts.append("functions read time(): " + ", ".join(f"'{n}'" for n in reads))
    if tables:
        parts.append("table functions are indexed by time: " + ", ".join(f"'{n}'" for n in tables))
    raise ModelError(
        f"{label} cannot run a model whose {' and whose '.join(parts)}. {why}. "
        "Simulate it with method='ode' or method='ssa' on the generated network, which "
        "read time() as the model time (issue #862)."
    )
