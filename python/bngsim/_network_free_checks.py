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
  the like, divided only by numbers and parameters. Nothing to cut at;
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
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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


_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _root(xml_path: str | Path) -> ET.Element | None:
    """The parsed document, or ``None`` when the file cannot be parsed.

    A comment that holds ``--`` is not well-formed XML, but both backends read
    past it, and hand-annotated files have them. So a file that does not parse
    is tried again with its comments taken out.
    """
    try:
        text = Path(xml_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for candidate in (text, _COMMENT.sub("", text)):
        try:
            return ET.fromstring(candidate)
        except ET.ParseError:
            continue
    return None


def _is_function(el: ET.Element) -> bool:
    return el.tag.endswith("Function") and not el.tag.endswith("ListOfFunctions")


def _functions(xml_path: str | Path) -> list[ET.Element] | None:
    """Every ``<Function>`` element in *xml_path*, or ``None`` when the file
    cannot be parsed: the backend then reports its own parse error, which is
    more specific than anything these checks could say."""
    root = _root(xml_path)
    return None if root is None else [el for el in root.iter() if _is_function(el)]


def _expression(func: ET.Element) -> str:
    """The text of the function's ``<Expression>``, with whatever markup it
    holds read through, as the backends read it."""
    return " ".join(
        "".join(child.itertext()) for child in func if child.tag.endswith("Expression")
    ).strip()


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
# What may stand on the outer side of a comparison for it to be a whole operand.
_OPENERS = frozenset({"(", ",", "&&", "||"})
_CLOSERS = frozenset({")", ",", "&&", "||"})
# The placeholder a table function's wrapper reads its table through, in the two
# spellings BNG2.pl has written.
_TABLE_VALUE = frozenset({"__TFUN_VAL__", "__TFUN__VAL__"})

_FREE, _SWITCH, _SMOOTH = "free", "switch", "smooth"


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
    where RuleMonkey searches: beside the XML, then one directory up. A file
    that is read is added to *files*."""
    file_attr = (func.get("file") or "").strip()
    try:
        if not file_attr:
            return [float(x) for x in (func.get("xData") or "").split(",")]
        for candidate in (xml_dir / file_attr, xml_dir / ".." / file_attr):
            if candidate.exists():
                files.append(candidate)
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
        self.functions = {f.get("id", "?"): f for f in root.iter() if _is_function(f)}
        self.parameters = {
            el.get("id") for el in root.iter() if el.tag.endswith("Parameter") and el.get("id")
        }
        observables = {
            el.get("name")
            for el in root.iter()
            if el.tag.endswith("Observable") and el.get("name")
        }
        self.observables = {name for name in observables if name is not None}
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
        arguments = {
            el.get("id") for el in func.iter() if el.tag.endswith("Argument") and el.get("id")
        }
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
        whats = [what for what, _ in items]
        if "clock" not in whats and _SMOOTH not in whats and _SWITCH not in whats:
            return _Shape()
        culprit = self._not_smooth(items)
        if culprit is None:
            moving = "clock" in whats or _SMOOTH in whats
            return _Shape(_SMOOTH if moving else _SWITCH, frozenset(breakpoints))
        # Not smooth, so it may hold anything: a step, a kink, a comparison.
        # Then the clock may enter only as a switch, which leaves the value
        # constant between the thresholds whatever else is done to it.
        if _SMOOTH in whats:
            smooth = next(tok for what, tok in items if what == _SMOOTH)
            raise _Refused(
                f"it applies {culprit!r} where it also reads {smooth!r}, which moves "
                "smoothly with the clock"
            )
        for j, (what, _) in enumerate(items):
            if what == "clock":
                breakpoints.add(self._threshold(items, j, culprit))
        return _Shape(_SWITCH, frozenset(breakpoints))

    @staticmethod
    def _not_smooth(items: list[tuple[str, str]]) -> str | None:
        """The first thing in *items* that could make the value jump, bend, or
        leave the numbers as the clock moves; ``None`` when nothing could."""
        for j, (what, tok) in enumerate(items):
            after = items[j + 1] if j + 1 < len(items) else ("end", "")
            beyond = items[j + 2] if j + 2 < len(items) else ("end", "")
            if what == "other":
                return tok
            if what != "op" or tok in "+-*()":
                continue
            if tok == "/" and after[0] in ("num", "parameter") and beyond[1] != "^":
                continue
            if tok == "^" and after[0] == "num" and after[1].isdigit() and beyond[1] != "^":
                continue
            return tok
        return None

    @staticmethod
    def _threshold(items: list[tuple[str, str]], j: int, culprit: str) -> float | str:
        """What the clock at *items[j]* is compared with: it has to be one whole
        side of an ordering whose other side is a number or a parameter."""

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
        raise _Refused(
            f"it reads the clock outside a comparison with a number or a parameter "
            f"(next to {culprit!r})"
        )


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _read_profile(path: Path) -> tuple[ClockProfile | None, list[Path]]:
    root = _root(path)
    if root is None:
        return None, []
    reader = _Reader(root, path.resolve().parent)
    readers: list[str] = []
    thresholds: set = set()
    for name in reader.functions:
        try:
            shape = reader.shape(name)
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
            readers.append(name)
            thresholds |= shape.breakpoints
    numbers = sorted(x for x in thresholds if not isinstance(x, str))
    names = sorted(x for x in thresholds if isinstance(x, str))
    return ClockProfile(readers=tuple(readers), thresholds=(*numbers, *names)), reader.files


# One reading per file, kept while the file and the table files it names are
# unchanged: a fit builds a session from the same XML for every evaluation.
_PROFILES: dict[str, tuple[tuple, ClockProfile | None]] = {}


def clock_profile(xml_path: str | Path) -> ClockProfile | None:
    """How the model in *xml_path* reads the clock under RuleMonkey, or ``None``
    when the file cannot be parsed here.

    Every function counts, not only the ones a rate law uses: a function that is
    only printed reads the clock into its output column all the same.
    """
    path = Path(xml_path)
    kept = _PROFILES.get(str(path))
    if kept is not None and all(_stamp(Path(f)) == stamp for f, stamp in kept[0]):
        return kept[1]
    profile, files = _read_profile(path)
    if len(_PROFILES) >= 256:
        _PROFILES.clear()
    _PROFILES[str(path)] = (tuple((str(f), _stamp(f)) for f in (path, *files)), profile)
    return profile


def rulemonkey_clock_profile(xml_path: str | Path) -> ClockProfile:
    """:func:`clock_profile`, raising :class:`ModelError` for a model RuleMonkey
    cannot be trusted with, and for one this cannot read."""
    profile = clock_profile(xml_path)
    if profile is None:
        raise ModelError(
            f"RuleMonkey (method='rm' / 'nf_exact'): {str(xml_path)!r} could not be read "
            "to see how its rate laws read the clock, so it cannot be run (issue #1022)."
        )
    if profile.refusal is not None:
        raise ModelError(
            f"RuleMonkey (method='rm' / 'nf_exact') cannot run this model: {profile.refusal}"
        )
    return profile
