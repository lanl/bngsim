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

**RuleMonkey** integrates a rate that reads the clock between events since
3.11.0 (issue #1022, richardposner/RuleMonkey#86). It does so in steps of up to
a thousandth of a call, and a step can lie across a short window in a rate and
see none of it: a pulse of area 1 that is 0.1 wide fires nothing in a call of
1000. So a clock-reading model is run only where bngsim can say where its rates
change with time, and every engine call is then cut at those instants, so that
no call has one inside it. :func:`clock_profile` reads the XML for that. A
function reads the clock in one of three supported ways:

* **smoothly**: sums, products and whole-number powers of the clock, parameters
  and observables, through ``exp``, ``sin``, ``cos``, ``tanh``, ``atan`` and
  the like, divided only by things that do not move with the clock. Nothing to
  cut at;
* **as a switch**: the clock compared with a number or a parameter
  (``time() >= 6``, ``(time()<t_off)``), in an expression that reads it no other
  way. The rate changes at each threshold;
* **through a table** indexed by the clock. The rate changes at each of the
  table's points.

A function may also be built from functions of those kinds. Anything else that
reads the clock (``floor(time())``, ``abs(time() - 6) < w``, a threshold that is
an expression, a division by the clock) is refused with :class:`ModelError`
when the simulator or the session is built: bngsim cannot say where such a rate
changes, so RuleMonkey could step over it.

The reading follows RuleMonkey's loader, not the XML at large: the first
``ListOfParameters``, ``ListOfObservables`` and ``ListOfFunctions`` under
``<model>``, each named exactly so, the first ``<Expression>`` of a function, a
table file beside the path as given. An element the loader ignores must not
change what a name means here. A file that is not well-formed XML is refused,
even where RuleMonkey's own parser would read it: bngsim cannot say what such a
file holds.

What the cuts do not cover is a smooth rate with a feature much shorter than a
thousandth of a call: a Gaussian 0.01 wide in a call of 1000, or a rate that
dips below zero, which RuleMonkey reads as 0, and is positive only that
briefly. Nothing marks where such a feature is.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.parsers import expat

from bngsim._exceptions import ModelError

# `time()` or a bare `time`: NFsim fails to prepare a function under either
# spelling.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?![A-Za-z0-9_])")

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


def _parse(data: bytes) -> tuple[ET.Element | None, str]:
    """*data* as a document, every element named as it is written, prefix and
    all; or ``None`` and the parser's reason when it is not well-formed XML.

    Namespaces are not worked out: RuleMonkey's loader compares names as
    written, so ``q:ListOfFunctions`` is not ``ListOfFunctions`` to it, and must
    not be here. Nothing is repaired either. RuleMonkey's own parser reads some
    files that are not well-formed (a comment holding ``--``, a repeated
    attribute), but a file patched up to parse is not the file it reads.
    """
    builder = ET.TreeBuilder()
    parser = expat.ParserCreate()
    parser.buffer_text = True
    parser.StartElementHandler = builder.start
    parser.EndElementHandler = builder.end
    parser.CharacterDataHandler = builder.data
    try:
        parser.Parse(data, True)
        return builder.close(), ""
    except (expat.ExpatError, AssertionError, ValueError) as why:
        return None, str(why)


def _root(xml_path: str | Path) -> ET.Element | None:
    """The parsed document, or ``None`` when the file cannot be read or parsed."""
    try:
        return _parse(Path(xml_path).read_bytes())[0]
    except OSError:
        return None


def _is_function(el: ET.Element) -> bool:
    return el.tag.endswith("Function") and not el.tag.endswith("ListOfFunctions")


def _functions(xml_path: str | Path) -> list[ET.Element] | None:
    """Every ``<Function>`` element in *xml_path*, or ``None`` when the file
    cannot be parsed: the backend then reports its own parse error, which is
    more specific than anything these checks could say."""
    root = _root(xml_path)
    return None if root is None else [el for el in root.iter() if _is_function(el)]


def _child(parent: ET.Element, name: str) -> ET.Element | None:
    """The first direct child written as *name*, which is the one RuleMonkey's
    loader takes."""
    return next((el for el in parent if el.tag == name), None)


def _listed(parent: ET.Element | None, list_name: str, item: str) -> list[ET.Element]:
    """The *item* elements of the first *list_name* directly under *parent*."""
    holder = None if parent is None else _child(parent, list_name)
    return [] if holder is None else [el for el in holder if el.tag == item]


def _expression(func: ET.Element) -> str:
    """The text of the function's first ``<Expression>``, with whatever markup
    it holds read through, as the backends read it."""
    expression = _child(func, "Expression")
    return "" if expression is None else "".join(expression.itertext()).strip()


def _reads_time(func: ET.Element) -> bool:
    return bool(_TIME_REF.search(_expression(func)))


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


# ── How a RuleMonkey model reads the clock ────────────────────────────────

# A number is read before a name, as ExprTk reads `2t` and `1e0t`: the number,
# then the name, multiplied.
_TOKEN = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|==|!=|&&|\|\||\S))"
)

# Entire functions of one argument: no pole, no edge of a domain, no kink.
_SMOOTH_CALLS = frozenset({"exp", "sin", "cos", "tanh", "sinh", "cosh", "atan", "erf", "erfc"})
_ORDERINGS = frozenset({"<", "<=", ">", ">="})
_COMPARISONS = _ORDERINGS | {"==", "!="}
# What may stand on the outer side of a comparison for it to be a whole operand.
_OPENERS = frozenset({"(", ",", "&&", "||"})
_CLOSERS = frozenset({")", ",", "&&", "||"})
# The placeholder a table function's wrapper reads its table through, in the two
# spellings BNG2.pl has written.
_TABLE_VALUE = frozenset({"__TFUN_VAL__", "__TFUN__VAL__"})

_FREE, _SWITCH, _SMOOTH = "free", "switch", "smooth"
# What an expression can hold that moves with the clock.
_MOVING = frozenset({"clock", _SWITCH, _SMOOTH})


class _Refused(Exception):
    """A function reads the clock in a way this cannot place. ``function`` is
    the one that does, which need not be the one whose shape was asked for."""

    function: str | None = None


@dataclass(frozen=True)
class _Shape:
    """How one function's value moves with the clock, the state held still.

    ``free``: not at all. ``switch``: constant between the breakpoints.
    ``smooth``: smooth between them. A breakpoint is a number, or the name of a
    parameter whose value is one.
    """

    kind: str = _FREE
    breakpoints: frozenset = frozenset()


@dataclass(frozen=True)
class ClockProfile:
    """How a BNG XML reads the clock, as RuleMonkey will run it.

    ``readers``
        The functions whose value moves with the clock, in document order.
        Empty for a model that sees only elapsed time.
    ``thresholds``
        Where a rate changes with the clock: numbers, and names of parameters
        whose values are such instants. See :meth:`breakpoints`.
    ``refusal``
        Why RuleMonkey cannot run the model, or ``None``.
    """

    readers: tuple[str, ...] = ()
    thresholds: tuple[float | str, ...] = ()
    refusal: str | None = None

    @property
    def reads_clock(self) -> bool:
        return bool(self.readers)

    def breakpoints(self, parameter: Callable[[str], float]) -> list[float]:
        """The instants at which a rate changes with the clock, sorted, with
        each parameter threshold read through *parameter* (so an override made
        before the run is the one that counts)."""
        values = {float(parameter(x)) if isinstance(x, str) else float(x) for x in self.thresholds}
        return sorted(v for v in values if math.isfinite(v))


def _tokens(text: str) -> list[tuple[str, str]]:
    """*text* as ``(kind, text)`` pairs, kind one of ``num``, ``name``, ``op``."""
    out = []
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup
        if kind is not None:
            out.append((kind, m.group(kind)))
    return out


def _table_points(func: ET.Element, xml_dir: Path, files: list[Path]) -> list[float]:
    """The x values of a table function, inline or from its file, searched for
    where RuleMonkey searches: beside the XML, then one directory up. Every
    place looked in is added to *files*."""
    file_attr = (func.get("file") or "").strip()
    try:
        if not file_attr:
            return [float(x) for x in (func.get("xData") or "").split(",")]
        # Normalised by its text, as the loader does: `link/..` is the link's
        # parent, not its target's.
        for joined in (xml_dir / file_attr, xml_dir / ".." / file_attr):
            candidate = Path(os.path.normpath(joined))
            # Kept whether or not it is there: a file that turns up at the
            # first place later is the one the engine will then read.
            files.append(candidate)
            if candidate.exists():
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
                return [float(s.split()[0]) for s in map(str.strip, lines) if s and s[0] != "#"]
    except (ValueError, IndexError, OSError):
        pass
    raise _Refused("its table could not be read here to learn where it changes")


class _Reader:
    """Works out the :class:`_Shape` of every function in one document."""

    def __init__(self, root: ET.Element, xml_dir: Path):
        self.xml_dir = xml_dir
        self.files: list[Path] = []  # the table files read along the way
        # What RuleMonkey's loader reads, and nothing else: <model> itself or
        # directly under the root, and the first list of each kind under it.
        model = root if root.tag == "model" else _child(root, "model")
        self.function_list = _listed(model, "ListOfFunctions", "Function")
        # A later function takes the name from an earlier one, as in the loader.
        self.functions = {f.get("id", "?"): f for f in self.function_list}
        self.parameters = {
            p.get("id") for p in _listed(model, "ListOfParameters", "Parameter") if p.get("id")
        }
        self.observables = {
            o.get("name")
            for o in _listed(model, "ListOfObservables", "Observable")
            if o.get("name")
        }
        # A model symbol takes a bare `time` or `t` from the clock. The call
        # `time()` stays the clock whatever is named `time`.
        self.symbols = self.parameters | self.observables | set(self.functions)
        self.shapes: dict[str, _Shape] = {}
        self.walking: set[str] = set()

    def shape(self, name: str) -> _Shape:
        if name in self.shapes:
            return self.shapes[name]
        if name in self.walking:
            raise _Refused(f"it refers back to itself through {name!r}")
        self.walking.add(name)
        try:
            found = self._function(self.functions[name])
        except _Refused as why:
            if why.function is None:
                why.function = name
            raise
        finally:
            self.walking.discard(name)
        self.shapes[name] = found
        return found

    def _function(self, func: ET.Element) -> _Shape:
        arguments = {a.get("id") for a in _listed(func, "ListOfArguments", "Argument")}
        text = _expression(func)
        if (func.get("type") or "").lower() != "tfun":
            return self._read(text, arguments, None)
        counter = (func.get("ctrName") or "").strip()
        if counter in _RULEMONKEY_CLOCK_COUNTERS:
            step = (func.get("method") or "linear").strip().lower() == "step"
            table = _Shape(
                _SWITCH if step else _SMOOTH,
                frozenset(_table_points(func, self.xml_dir, self.files)),
            )
        elif counter in self.functions and self.shape(counter).kind != _FREE:
            raise _Refused(
                f"it is a table indexed by {counter!r}, which reads the clock, so where the "
                "table's value changes cannot be read off the table"
            )
        else:
            table = _Shape()
        if not text:
            return table
        # A wrapper that leaves the placeholder out still belongs to a function
        # the engine counts as moving with its table.
        wrapper = self._read(text, arguments, table)
        kinds = (wrapper.kind, table.kind)
        kind = _SMOOTH if _SMOOTH in kinds else _SWITCH if _SWITCH in kinds else _FREE
        return _Shape(kind, wrapper.breakpoints | table.breakpoints)

    def _read(self, text: str, arguments: set, table: _Shape | None) -> _Shape:
        """The shape of an expression. *table* is what its table placeholder
        stands for, when it is a table function's wrapper."""
        tokens = _tokens(text)
        items: list[tuple[str, str]] = []  # (what, text)
        parts: list[_Shape] = []  # the shape of every function or table it reads
        i = 0
        while i < len(tokens):
            kind, tok = tokens[i]
            called = tokens[i + 1 : i + 3] == [("op", "("), ("op", ")")]
            tagged = (
                len(tokens) > i + 3
                and tokens[i + 1] == ("op", "(")
                and tokens[i + 2][0] == "name"
                and tokens[i + 3] == ("op", ")")
            )
            if kind != "name":
                items.append((kind, tok))
            elif tok == "time" and called:
                items.append(("clock", tok))
                i += 2
            elif tok in ("time", "t") and tok not in self.symbols:
                items.append(("clock", tok))
                i += 2 if called else 0
            elif tok in _TABLE_VALUE and table is not None:
                items.append((table.kind, tok))
                parts.append(table)
            elif tok in self.functions:
                part = self.shape(tok)
                items.append((part.kind, tok))
                parts.append(part)
                i += 2 if called else 3 if tagged else 0
            elif tok in self.observables:
                items.append((_FREE, tok))
                i += 3 if tagged else 0
            elif tok in self.parameters:
                items.append(("parameter", tok))
            elif tok in arguments:
                items.append((_FREE, tok))
            elif tok in _SMOOTH_CALLS and tokens[i + 1 : i + 2] == [("op", "(")]:
                items.append(("call", tok))
            else:
                items.append(("other", tok))
            i += 1

        breakpoints: set = set().union(*(p.breakpoints for p in parts))
        items = self._fold(items)
        whats = [what for what, _ in items]
        if not _MOVING.intersection(whats):
            return _Shape()
        rough = self._not_smooth(items)
        if not rough:
            moving = "clock" in whats or _SMOOTH in whats
            return _Shape(_SMOOTH if moving else _SWITCH, frozenset(breakpoints))
        culprit = rough[0][1]
        # Not smooth, so it may hold anything: a step, a kink, a comparison.
        # Then the clock may enter only as a switch, which leaves the value
        # constant between the thresholds whatever else is done to it.
        if _SMOOTH in whats:
            smooth = next(tok for what, tok in items if what == _SMOOTH)
            raise _Refused(
                f"it applies {culprit!r} where it also reads {smooth!r}, which moves "
                "smoothly with the clock"
            )
        clocks = [j for j, (what, _) in enumerate(items) if what == "clock"]
        thresholds = [self._threshold(items, j) for j in clocks]
        if None not in thresholds:
            return _Shape(_SWITCH, frozenset(breakpoints.union(thresholds)))
        # The readings that are not a whole side of a good comparison: are they
        # in a comparison at all?
        compared = any(
            what == "op" and tok in _COMPARISONS
            for j, found in zip(clocks, thresholds, strict=True)
            if found is None
            for what, tok in (items[j - 1] if j else ("edge", ""), *items[j + 1 : j + 2])
        )
        if any(x is not None for x in thresholds) and not compared:
            raise _Refused(
                "it reads the clock both in a comparison with a number or a parameter and "
                "outside one. Write the switch and the smooth part as two functions and "
                "multiply them"
            )
        # Name the rough thing nearest a reading of the clock, which is more
        # likely the one applied to it than the first in the expression, and a
        # function before an operator.
        named = [found for found in rough if found[1][0].isalpha() or found[1][0] == "_"]
        nearest = min(named or rough, key=lambda found: min(abs(found[0] - j) for j in clocks))[1]
        raise _Refused(self._why_not_smooth("<" if compared else nearest))

    @staticmethod
    def _fold(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """*items* with every parenthesised group that holds nothing moving with
        the clock, and the name it is an argument list of, made one constant:
        ``ln(2)``, ``(60*24)``, ``if(A0>100,k,0)``."""
        items = list(items)
        folded = True
        while folded:
            folded = False
            opened: list[int] = []
            for j, (what, tok) in enumerate(items):
                if (what, tok) == ("op", "("):
                    opened.append(j)
                elif (what, tok) == ("op", ")") and opened:
                    start = opened.pop()
                    inner = items[start + 1 : j]
                    if _MOVING.intersection(w for w, _ in inner) or ("op", "(") in inner:
                        continue  # a group left standing inside it holds something moving
                    head = (
                        start - 1 if start and items[start - 1][0] in ("other", "call") else start
                    )
                    items[head : j + 1] = [(_FREE, "(...)")]
                    folded = True
                    break
        return items

    @staticmethod
    def _why_not_smooth(culprit: str) -> str:
        """What stands between an expression and being smooth in the clock."""
        if culprit == "/":
            return (
                "it divides by something that moves with the clock (a pole wherever that "
                "passes 0), or by something that is not a number, a parameter or a "
                "constant in parentheses"
            )
        if culprit == "^":
            return (
                "it raises to a power that is not a whole number written as a number, and "
                "the base is not a positive number: the value can bend or leave the real "
                "numbers as the clock moves"
            )
        if culprit in _COMPARISONS:
            return (
                "it has the clock in a comparison this cannot place: time() has to stand "
                "alone on one side of <, <=, > or >=, with a number or a parameter alone on "
                "the other (give a threshold that is a sum a parameter of its own)"
            )
        return (
            f"it holds {culprit!r}, which can make the value jump, bend or leave the real "
            "numbers as the clock moves, and that is not one of the smooth functions "
            "bngsim knows (exp, sin, cos, tanh, sinh, cosh, atan, erf, erfc)"
        )

    @staticmethod
    def _not_smooth(items: list[tuple[str, str]]) -> list[tuple[int, str]]:
        """Where in *items* something could make the value jump, bend, or leave
        the numbers as the clock moves, and what it is. Empty when nothing could."""
        found: list[tuple[int, str]] = []
        for j, (what, tok) in enumerate(items):
            before = items[j - 1] if j else ("edge", "")
            after = items[j + 1] if j + 1 < len(items) else ("edge", "")
            beyond = items[j + 2] if j + 2 < len(items) else ("edge", "")
            if what == "other":
                found.append((j, tok))
                continue
            if what != "op" or tok in "+-*()":
                continue
            # Divided by something that does not move with the clock.
            if tok == "/" and after[0] in ("num", "parameter", _FREE) and beyond[1] != "^":
                continue
            # A whole power of anything, or any power of a positive number.
            if tok == "^" and after[0] == "num" and after[1].isdigit() and beyond[1] != "^":
                continue
            base_is_exponent = j >= 2 and items[j - 2] == ("op", "^")
            if tok == "^" and before[0] == "num" and float(before[1]) > 0 and not base_is_exponent:
                continue
            found.append((j, tok))
        return found

    @staticmethod
    def _threshold(items: list[tuple[str, str]], j: int) -> float | str | None:
        """What the clock at *items[j]* is compared with, when it is one whole
        side of an ordering whose other side is a number or a parameter;
        ``None`` when it is not."""

        def at(k: int) -> tuple[str, str]:
            return items[k] if 0 <= k < len(items) else ("edge", "")

        def constant(k: int, sign: float) -> float | str | None:
            what, tok = at(k)
            if what == "num":
                return sign * float(tok)
            return tok if what == "parameter" and sign > 0 else None

        def opens(k: int) -> bool:
            return at(k)[0] == "edge" or (at(k)[0] == "op" and at(k)[1] in _OPENERS)

        def closes(k: int) -> bool:
            return at(k)[0] == "edge" or (at(k)[0] == "op" and at(k)[1] in _CLOSERS)

        if at(j + 1)[0] == "op" and at(j + 1)[1] in _ORDERINGS and opens(j - 1):
            minus = at(j + 2) == ("op", "-")
            value = constant(j + 3 if minus else j + 2, -1.0 if minus else 1.0)
            if value is not None and closes(j + 4 if minus else j + 3):
                return value
        if at(j - 1)[0] == "op" and at(j - 1)[1] in _ORDERINGS and closes(j + 1):
            minus = at(j - 3) == ("op", "-") and opens(j - 4)
            value = constant(j - 2, -1.0 if minus else 1.0)
            if value is not None and (minus or opens(j - 3)):
                return value
        return None


def _read_profile(path: Path, data: bytes) -> tuple[ClockProfile | None, list[Path]]:
    root = _parse(data)[0]
    if root is None:
        return None, []
    reader = _Reader(root, path.parent)
    readers: list[str] = []
    thresholds: set = set()
    for func in reader.function_list:
        name = func.get("id", "?")
        try:
            # A function a later one took the name from is still evaluated.
            named = reader.functions.get(name) is func
            shape = reader.shape(name) if named else reader._function(func)
        except _Refused as why:
            culprit = why.function or name
            return ClockProfile(
                readers=(culprit,),
                refusal=(
                    f"function {culprit!r} reads the clock in a way RuleMonkey cannot be "
                    f"trusted with: {why}. RuleMonkey integrates a rate that reads the "
                    "clock in steps of up to a thousandth of a call and can step over a "
                    "short window in it, so bngsim runs one only where it can say when the "
                    "rate changes: where the clock is read smoothly, compared with a number "
                    "or a parameter, or read through a table. Simulate this model with "
                    "method='ssa' or method='ode' on the generated network (issue #1022)."
                ),
            ), reader.files
        if shape.kind != _FREE:
            if name not in readers:
                readers.append(name)
            thresholds |= shape.breakpoints
    numbers = sorted(x for x in thresholds if not isinstance(x, str))
    names = sorted(x for x in thresholds if isinstance(x, str))
    return ClockProfile(readers=tuple(readers), thresholds=(*numbers, *names)), reader.files


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _file_digest(path: Path) -> str | None:
    try:
        return _digest(path.read_bytes())
    except OSError:
        return None


# One reading per file, kept while the bytes of the file and of the table files
# it names are unchanged: a fit builds a session from the same XML for every
# evaluation. Keyed on the bytes, not on a size and a time, which a rewrite can
# leave as they were.
_PROFILES: dict[str, tuple[str, tuple[tuple[Path, str | None], ...], ClockProfile | None]] = {}


def clock_profile(xml_path: str | Path) -> ClockProfile | None:
    """How the model in *xml_path* reads the clock under RuleMonkey, or ``None``
    when the file cannot be read or is not well-formed XML.

    Every function counts, not only the ones a rate law uses: a function that is
    only printed reads the clock into its output column all the same.
    """
    path = Path(xml_path)
    try:
        data = path.read_bytes()
    except OSError:
        return None
    digest = _digest(data)
    kept = _PROFILES.get(str(path))
    if kept is not None and kept[0] == digest and all(_file_digest(f) == d for f, d in kept[1]):
        return kept[2]
    profile, files = _read_profile(path, data)
    if len(_PROFILES) >= 256:
        _PROFILES.clear()
    _PROFILES[str(path)] = (digest, tuple((f, _file_digest(f)) for f in files), profile)
    return profile


def rulemonkey_clock_profile(xml_path: str | Path) -> ClockProfile:
    """:func:`clock_profile`, raising :class:`ModelError` for a model RuleMonkey
    cannot be trusted with, and for one this cannot read."""
    profile = clock_profile(xml_path)
    if profile is None:
        try:
            why = _parse(Path(xml_path).read_bytes())[1] or "no reason given"
        except OSError as e:
            why = str(e)
        raise ModelError(
            f"RuleMonkey (method='rm' / 'nf_exact'): {str(xml_path)!r} could not be read "
            f"to see how its rate laws read the clock ({why}), so it cannot be run. bngsim "
            "reads the file as XML and repairs nothing; RuleMonkey's own parser also reads "
            "some files that are not well-formed, such as one with -- inside a comment or "
            "an attribute given twice. Correct the file (issue #1022)."
        )
    if profile.refusal is not None:
        raise ModelError(
            f"RuleMonkey (method='rm' / 'nf_exact') cannot run this model: {profile.refusal}"
        )
    return profile
