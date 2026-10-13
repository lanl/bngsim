"""Checks on a BNG XML before a network-free backend is handed it.

**NFsim** cannot prepare a function that reads ``time()`` and fails at setup
with ``Error preparing function f in class GlobalFunction``, which names
neither the cause nor an alternative (issue #862). It is refused here, before
the backend is built, naming the functions. The clock is read under either
spelling, ``time()`` or a bare ``time``, and both are refused. A table function
indexed by time (``type="TFUN"`` with ``ctrName="time"``) keeps its clock in an
attribute, and its ``<Expression>`` reads a placeholder, so NFsim can prepare
it and it is not refused. NFsim refreshes its value only when a reaction fires,
so a rate that is 0 at the start is not seen (issue #892, still open for
NFsim).

**RuleMonkey** runs both since 3.11.0 (issue #1022): a rate that reads the clock
is integrated between events, and a total propensity of 0 is no longer final
(richardposner/RuleMonkey#86). Nothing is refused when a simulator or session
is built. What is left is one question a session asks,
:func:`clock_reading_functions`: whether the model reads the clock at all. A
session segment that does not start at the session's clock is run as elapsed
time only when it does not.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bngsim._exceptions import ModelError

# `time()` or a bare `time`: NFsim fails to prepare a function under either
# spelling.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?![A-Za-z0-9_])")

# The names RuleMonkey's loader counts as the clock in a function's expression:
# `time` (the call `time()` included) and the bare alias `t`. It counts them even
# where a model symbol has taken the name (the walk under "8b" in load_model,
# third_party/rulemonkey/cpp/rulemonkey/simulator.cpp), and so does this.
_RULEMONKEY_CLOCK_REF = re.compile(r"(?<![A-Za-z0-9_])(?:time|t)(?![A-Za-z0-9_])")

# The counter names RuleMonkey reads as the clock for a table function, checked
# before any parameter or observable of that name (resolve_tfun_counter in
# third_party/rulemonkey/cpp/rulemonkey/simulator.cpp).
_RULEMONKEY_CLOCK_COUNTERS = frozenset({"time", "t", "time()", "t()"})

_BACKENDS = {
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


def clock_reading_functions(xml_path: str | Path) -> list[str] | None:
    """The ids of the functions in *xml_path* through which RuleMonkey reads the
    clock: an expression naming ``time`` or the bare alias ``t``, or a table
    function indexed by either. ``None`` when the file cannot be parsed here,
    which a caller has to read as "may read the clock".

    Every function counts, not only the ones a rate law uses: a function that is
    only printed reads the clock into its output column all the same.
    """
    funcs = _functions(xml_path)
    if funcs is None:
        return None
    return [
        f.get("id", "?")
        for f in funcs
        if _indexed_by_time(f)
        or any(
            expr.tag.endswith("Expression") and _RULEMONKEY_CLOCK_REF.search(expr.text or "")
            for expr in f
        )
    ]


def refuse_time_dependent_rate_laws(xml_path: str | Path, backend: str) -> None:
    """Raise :class:`ModelError` if *xml_path* has a function reading the clock.

    *backend* is ``"nfsim"``, the one backend that cannot run such a function.
    """
    reads = time_reading_functions(xml_path)
    if not reads:
        return
    label, why = _BACKENDS[backend]
    raise ModelError(
        f"{label} cannot run a model whose functions read time(): "
        f"{', '.join(repr(n) for n in reads)}. {why}. "
        "Simulate it with method='rm' on the same XML, or with method='ode' or "
        "method='ssa' on the generated network, all of which read time() as the model "
        "time (issue #862)."
    )
