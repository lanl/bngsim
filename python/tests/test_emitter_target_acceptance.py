"""Issue #464 — ask the target whether it accepts what the emitter wrote.

``bngsim._jacobian`` has two emitters. ``sympy_to_exprtk`` writes the analytic
Jacobian and the zero-logarithm guard as text for the engine's own expression
parser, and ``sympy_to_c`` writes the same derivatives as C for the generated
source. Both are built out of a function-name table plus a handful of printer
methods, and until now nothing checked that the text either one produces is text
the thing on the other side will actually take.

Issue #460 is what that costs. ``Max(k1, k2)`` had been in the ExprTk emitter's
name table from the start, the table entry was never once read, and sympy's own
printer wrote the class name instead. The engine's parser is case sensitive, so
it answered "Undefined symbol: 'Max'" and the model dropped to the
finite-difference Jacobian without saying so. Nothing failed, so nobody looked.
The fix was one printer method. Finding it took a reading of sympy's printer
source, because there was no test that would have said anything.

This file is that test. Every construct the two emitters claim to support gets a
row in ``CASES`` below. Each row is emitted, handed to the engine to compile and
to the system C compiler to compile, evaluated at several points on both, and
compared against sympy's own value for the same expression. A construct the
target rejects fails loudly here, and so does one the target accepts but computes
differently, which is the worse of the two cases and the one a string comparison
cannot see.

Two guards keep the table honest, because a table nobody adds to stops meaning
anything:

* ``test_every_emittable_function_has_a_case`` reads the emitters' own name
  tables. A new function mapped there with no row here fails.
* ``test_every_printer_method_is_exercised`` runs the whole table through
  instrumented copies of both printers and collects which ``_print_*`` methods
  ran. A new printer method with no row that reaches it fails. This is the guard
  that would have caught #460: ``Max`` had a table entry and no method, so the
  row for it would have gone through sympy's fallback and the engine would have
  refused the text.

The last section is about the family #460 left behind. ``Xor``, ``Implies`` and
``Equivalent`` are boolean nodes the emitters have no spelling for, and ``Xor``
is the dangerous one, because sympy prints it as infix ``^`` and the engine
reads ``^`` as exponentiation. That would be a wrong number rather than a
refusal. They are refused today. What was not established is whether anything can
produce one, so ``test_no_rate_law_produces_a_boolean_node_with_no_spelling``
asks that mechanically instead of by reading code: it puts every boolean idiom a
rate law may be written in through the loader's own parse and collects the
boolean node types that come out.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import bngsim
import pytest

sp = pytest.importorskip("sympy")

from bngsim import _jacobian as jac  # noqa: E402
from bngsim._codegen import _CODEGEN_PRELUDE_LINES  # noqa: E402
from bngsim._jacobian import (  # noqa: E402
    _SYMPY_FUNC_TO_C,
    _SYMPY_FUNC_TO_EXPRTK,
    _TIME_SYM,
    sympy_to_c,
    sympy_to_exprtk,
)

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler (cc/clang/gcc) on PATH")


# ─── the alphabet every case is written in ───────────────────────────────────
#
# Plain real symbols rather than positive ones: `positive=True` would let sympy
# fold `Abs(x)` to `x` and `sign(x)` to 1 before the emitter ever saw them, which
# is the opposite of what this file is for.

x0, x1, x2 = (sp.Symbol(name, real=True) for name in ("x0", "x1", "x2"))

# The placeholder the loader puts in for the engine's `time()`, and the alias it
# puts in for a parameter whose name is a Python keyword. Both are printed by
# `_print_Symbol` on paths an ordinary symbol does not reach.
TIME = sp.Symbol(_TIME_SYM)
KEYWORD_PARAM = sp.Symbol(jac._alias_keyword_param("lambda"))

mratio = jac.engine_sympy_bindings(sp)["mratio"]

# (time, (x0, x1, x2)). x0 stays positive so a logarithm and a square root are
# defined on it, x1/10 stays inside asin/acos's domain, and x2 changes sign and
# passes through zero so sign, abs, floor and ceil are asked about all three
# cases rather than one.
POINTS = (
    (0.0, (2.0, 0.5, -3.0)),
    (1.5, (0.25, 3.0, 4.0)),
    (7.0, (9.0, 0.75, -0.5)),
    (3.0, (1.0, 1.0, 0.0)),
)


@dataclass(frozen=True)
class Case:
    """One construct, and what the two targets have to make of it."""

    name: str
    expr: object
    #: Which emitters are supposed to have a spelling. `mratio_dz` is the one
    #: construct only C can render — the engine has no such function — so its
    #: row asserts the ExprTk emitter declines rather than writes something.
    targets: tuple[str, ...] = ("exprtk", "c")
    #: An ExprTk text whose value, asked of the engine, is the reference for
    #: this row. Set on the two rows sympy cannot value: it has no numeric
    #: mratio. Everything else is held to sympy.
    reference: str = ""
    points: tuple = POINTS
    notes: str = ""
    #: Which entries of the emitters' name tables this row is the case for.
    #: Read by `test_every_emittable_function_has_a_case`.
    covers: frozenset = field(default_factory=frozenset)


def _f(*names: str) -> frozenset:
    return frozenset(names)


CASES: tuple[Case, ...] = (
    # ── the name tables ──────────────────────────────────────────────────
    Case("exp", sp.exp(x1), covers=_f("exp")),
    Case("log", sp.log(x0), covers=_f("log")),
    Case("sqrt", sp.sqrt(x0), covers=_f("sqrt")),
    Case("abs", sp.Abs(x2), covers=_f("Abs")),
    Case("sign", sp.sign(x2), covers=_f("sign")),
    Case("sin", sp.sin(x2), covers=_f("sin")),
    Case("cos", sp.cos(x2), covers=_f("cos")),
    Case("tan", sp.tan(x1), covers=_f("tan")),
    Case("asin", sp.asin(x1 / 10), covers=_f("asin")),
    Case("acos", sp.acos(x1 / 10), covers=_f("acos")),
    Case("atan", sp.atan(x2), covers=_f("atan")),
    Case("sinh", sp.sinh(x1), covers=_f("sinh")),
    Case("cosh", sp.cosh(x1), covers=_f("cosh")),
    Case("tanh", sp.tanh(x2), covers=_f("tanh")),
    Case("floor", sp.floor(x2), covers=_f("floor")),
    Case("ceiling", sp.ceiling(x2), covers=_f("ceiling")),
    Case("min", sp.Min(x0, x1), covers=_f("Min")),
    Case("max", sp.Max(x0, x1), covers=_f("Max")),
    Case(
        "min of three",
        sp.Min(x0, x1, x2),
        notes="folded into nested binary calls; neither target takes an n-ary min",
    ),
    Case("max of three", sp.Max(x0, x1, x2)),
    # ── powers, one row per branch of `_print_Pow` ───────────────────────
    Case("power one half", x0 ** sp.Rational(1, 2), notes="emitted as sqrt, not as a power"),
    Case("power minus one half", x0 ** sp.Rational(-1, 2)),
    Case("power minus one", x0**-1),
    Case("power of three", x0**3, notes="C spells a small whole power as repeated multiplication"),
    Case("power of minus three", x0**-3),
    Case("power of seven", x0**7, notes="above the repeated-multiplication cutoff, so C uses pow"),
    Case("fractional power", x0 ** sp.Rational(5, 2)),
    Case("symbolic power", x0**x1),
    # ── conditions ───────────────────────────────────────────────────────
    Case("piecewise", sp.Piecewise((x0, x1 > 1), (x2, True))),
    Case("equal", sp.Piecewise((x0, sp.Eq(x1, 3.0)), (x2, True))),
    Case("not equal", sp.Piecewise((x0, sp.Ne(x1, 3.0)), (x2, True))),
    Case("less than", sp.Piecewise((x0, x1 < 1), (x2, True))),
    Case("less or equal", sp.Piecewise((x0, x1 <= 1), (x2, True))),
    Case("greater than", sp.Piecewise((x0, x1 > 1), (x2, True))),
    Case("greater or equal", sp.Piecewise((x0, x1 >= 1), (x2, True))),
    Case("and", sp.Piecewise((x0, sp.And(x1 > 1, x2 > 0)), (x2, True))),
    Case("or", sp.Piecewise((x0, sp.Or(x1 > 1, x2 > 0)), (x2, True))),
    Case(
        "not",
        sp.Piecewise((x0, sp.Not(sp.And(x1 > 1, x2 > 0))), (x2, True)),
        notes="negating one comparison is not enough: sympy flips it to the "
        "opposite comparison and no `not` is ever printed",
    ),
    Case(
        "if then else",
        sp.Piecewise((x0, sp.logic.boolalg.ITE(x1 > 1, x2 > 0, x2 < 0)), (x2, True)),
        notes="rewritten to and/or/not before printing; nothing prints an ITE",
    ),
    Case(
        "two conditions",
        sp.Piecewise((x0, x1 > 1), (x1, x2 > 0), (x2, True)),
        notes="nested, so the second condition is only asked when the first fails",
    ),
    # ── literals and named constants ─────────────────────────────────────
    Case("decimal literal", sp.Float(2.5) * x0),
    Case("whole number literal", sp.Integer(3) + x0),
    Case("fraction literal", sp.Rational(2, 3) + x0),
    Case("very small literal", sp.Float(1e-300) * x0, notes="a normal double, and the C twin"),
    Case("very large literal", sp.Float(1e300) * x0),
    Case("e", sp.E * x0),
    Case("pi", sp.pi * x0, notes="reachable from asin(1) and friends, which fold to pi/2"),
    # ── symbols ──────────────────────────────────────────────────────────
    Case("time", TIME * x0, notes="the engine's time(), carried through sympy as a placeholder"),
    Case(
        "parameter named after a python keyword",
        KEYWORD_PARAM * x0,
        notes="the ExprTk emitter has to undo the alias the loader applied; the C "
        "emitter resolves every symbol through its caller and never sees it",
    ),
    # ── the engine's own function ────────────────────────────────────────
    Case(
        "mratio",
        mratio(-3.0, 5.0, -x0),
        reference="mratio(-3.0,5.0,-x0)",
        notes="sympy has no numeric mratio, so the engine is the reference. That "
        "leaves the ExprTk half of this row asking only that the engine compiles "
        "the text and answers a real number, which is still the half that would "
        "have caught #460. The C half is a real comparison: the generated "
        "source carries its own copy of mratio, and this holds it to the "
        "engine's. test_codegen_mratio.py is where that copy is held to the "
        "C++ it was ported from.",
        covers=_f("mratio"),
    ),
    Case(
        "the derivative of mratio",
        sp.diff(mratio(-3.0, 5.0, -x0), x0),
        targets=("c",),
        # Kummer's identity, which is what the helper computes:
        # dR/dz (a,b,z) = R(a,b,z)·[(a+1)/(b+1)·R(a+1,b+1,z) - (a/b)·R(a,b,z)],
        # here with a = -3, b = 5, z = -x0, so the chain rule adds the minus.
        reference=(
            "-(mratio(-3.0,5.0,-x0)*((-2.0/6.0)*mratio(-2.0,6.0,-x0)"
            " - (-3.0/5.0)*mratio(-3.0,5.0,-x0)))"
        ),
        notes="only C has this one: it is emitted as a helper the generated "
        "source carries, and the engine has no such function. So the reference "
        "is the identity the helper is supposed to compute, written out in the "
        "engine's own mratio calls.",
        covers=_f("mratio_dz"),
    ),
)


# ─── the two targets ─────────────────────────────────────────────────────────

#: A model with the three observables and the keyword-named parameter every case
#: is written against. The rate law is the emitted text, so building the model is
#: what asks the engine to compile it.
NET = """begin parameters
    1 lambda 1.5  # Constant
end parameters
begin observables
    1 Molecules x0 A
    2 Molecules x1 B
    3 Molecules x2 C
end observables
begin functions
    1 law() {body}
end functions
begin species
    1 A() 1.0
    2 B() 1.0
    3 C() 1.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 x0                    1
    2 x1                    2
    3 x2                    3
end groups
"""


def engine_values(tmp_path, text: str, points, tag: str) -> list[float]:
    """Compile ``text`` with the engine's own expression parser and evaluate it.

    Loading the model is the compile: a rate law the parser will not take makes
    ``Model.from_net`` raise, which is the loud half of what this file checks.
    """
    net = tmp_path / f"{tag}.net"
    net.write_text(NET.format(body=text))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(net))
    core = model._core if hasattr(model, "_core") else model
    return [core._eval_functions(t, list(conc))["law"] for t, conc in points]


#: What ``sympy_to_c`` resolves each free symbol to. The C emitter has no symbol
#: names of its own — every one goes through this callback — so this is the
#: harness standing in for the generated source's ``y[i]`` / parameter lookups.
C_SYMBOLS = {
    "x0": "x0",
    "x1": "x1",
    "x2": "x2",
    _TIME_SYM: "t",
    str(KEYWORD_PARAM): "kw_lambda",
}


def compiler_values(text: str, points) -> list[float]:
    """Compile ``text`` with the system C compiler and run it at each point.

    The generated source's own prelude is included, because that is where the
    helpers for the engine's functions are defined and a rate law reaching one
    of those would not compile without it.
    """
    source = (
        "#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        + "\n".join(_CODEGEN_PRELUDE_LINES)
        + "\nint main(int argc, char **argv) {\n"
        "  double t = atof(argv[1]);\n"
        "  double x0 = atof(argv[2]), x1 = atof(argv[3]), x2 = atof(argv[4]);\n"
        "  double kw_lambda = 1.5;\n"
        f'  printf("%.17g\\n", (double)({text}));\n'
        "  return 0;\n}\n"
    )
    work = tempfile.mkdtemp()
    c_file, program = os.path.join(work, "case.c"), os.path.join(work, "case")
    with open(c_file, "w") as handle:
        handle.write(source)
    built = subprocess.run(
        [_CC, "-O2", "-o", program, c_file, "-lm"], capture_output=True, text=True
    )
    assert built.returncode == 0, (
        f"the C compiler rejected what sympy_to_c wrote:\n{text}\n\n{built.stderr}"
    )
    out = []
    for t, conc in points:
        run = subprocess.run(
            [program, repr(t), *[repr(v) for v in conc]],
            capture_output=True,
            text=True,
            check=True,
        )
        out.append(float(run.stdout.strip()))
    return out


def reference_values(tmp_path, case: Case) -> list[float]:
    """What the expression is worth, asked of neither emitter. Both targets are
    held to this."""
    if case.reference:
        return engine_values(tmp_path, case.reference, case.points, tag=f"ref_{_tag(case)}")
    values = []
    for t, conc in case.points:
        bound = case.expr.subs(
            {x0: conc[0], x1: conc[1], x2: conc[2], TIME: t, KEYWORD_PARAM: 1.5}
        )
        if bound in (sp.true, sp.false):
            values.append(1.0 if bound == sp.true else 0.0)
        else:
            values.append(float(sp.N(bound)))
    return values


def _tag(case: Case) -> str:
    return case.name.replace(" ", "_")


def _emit_exprtk(case: Case) -> str | None:
    return sympy_to_exprtk(case.expr)


def _emit_c(case: Case) -> str | None:
    return sympy_to_c(case.expr, C_SYMBOLS.get)


_IDS = [case.name for case in CASES]


# ─── the table ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_the_engine_takes_what_the_exprtk_emitter_wrote(tmp_path, case):
    text = _emit_exprtk(case)
    if "exprtk" not in case.targets:
        assert text is None, (
            f"{case.name}: the ExprTk emitter is not supposed to have a spelling for "
            f"this, but it wrote {text!r}. A spelling the engine does not have is "
            f"worse than a refusal, because the model runs and the number is wrong."
        )
        return
    assert text is not None, f"{case.name}: the ExprTk emitter declined a construct it claims"
    got = engine_values(tmp_path, text, case.points, tag=_tag(case))
    assert all(math.isfinite(v) for v in got), f"{case.name}: the engine answered {got}"
    assert got == pytest.approx(reference_values(tmp_path, case), rel=1e-12, abs=1e-300), (
        f"{case.name}: the engine compiled {text!r} and computed something else"
    )


@needs_cc
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_the_c_compiler_takes_what_the_c_emitter_wrote(tmp_path, case):
    text = _emit_c(case)
    if "c" not in case.targets:
        assert text is None, f"{case.name}: the C emitter is not supposed to spell this"
        return
    assert text is not None, f"{case.name}: the C emitter declined a construct it claims"
    got = compiler_values(text, case.points)
    assert all(math.isfinite(v) for v in got), f"{case.name}: the compiled C answered {got}"
    assert got == pytest.approx(reference_values(tmp_path, case), rel=1e-12, abs=1e-300), (
        f"{case.name}: the compiled C for {text!r} computed something else"
    )


# ─── keeping the table honest ────────────────────────────────────────────────


def test_every_emittable_function_has_a_case():
    """The emitters' name tables are their statement of what they support, so
    every entry needs a row here.

    Some of those entries are read by a printer method rather than by the table
    lookup itself — ``Min``, ``Max`` and ``Abs`` all have their own method — and
    they stay in the tables because a third place reads them: ``_is_emittable``
    uses the two tables together to decide whether a derivative is renderable at
    all. Either way, an entry is a claim, and a claim gets a row.
    """
    claimed = set(_SYMPY_FUNC_TO_EXPRTK) | set(_SYMPY_FUNC_TO_C)
    covered = set().union(*(case.covers for case in CASES))
    assert claimed - covered == set(), (
        "these functions are mapped by an emitter and have no case in this file: "
        f"{sorted(claimed - covered)}. Add one — a name in the table with no case "
        "is exactly how Max(k1, k2) went unnoticed (issue #460)."
    )
    assert covered - claimed == set(), (
        f"these cases claim to cover a name no emitter maps: {sorted(covered - claimed)}"
    )


#: ``_print_LatticeOp`` is on both printers to refuse, not to write anything, so
#: no case can reach it and produce text. test_exprtk_minmax_spelling.py holds
#: that one.
REFUSAL_ONLY_METHODS = frozenset({"_print_LatticeOp"})


def _instrumented(make_printer):
    """A fresh printer class that records which ``_print_*`` methods ran."""
    printer_class = make_printer()
    ran: set[str] = set()

    for name in [attr for attr in vars(printer_class) if attr.startswith("_print_")]:
        original = getattr(printer_class, name)

        def recorded(self, expr, *args, _original=original, _name=name, **kwargs):
            ran.add(_name)
            return _original(self, expr, *args, **kwargs)

        setattr(printer_class, name, recorded)
    return printer_class, ran


def test_every_printer_method_is_exercised(monkeypatch):
    """No printer method may go untested, in either emitter.

    This is the guard for the shape of #460 rather than for its instance. ``Max``
    was in the name table and had no printer method, so sympy's own fallback
    printed the class name and the engine refused the text. A method with no case
    is the same hole one step along: it is code nothing has ever asked the target
    about.
    """
    exprtk_class, exprtk_ran = _instrumented(jac._make_printer)
    c_class, c_ran = _instrumented(jac._make_c_printer)
    monkeypatch.setattr(jac, "_printer_cache", [exprtk_class()])
    monkeypatch.setattr(jac._c_printer_local, "printer", c_class(), raising=False)

    for case in CASES:
        _emit_exprtk(case)
        _emit_c(case)

    for label, printer_class, ran in (
        ("ExprTk", exprtk_class, exprtk_ran),
        ("C", c_class, c_ran),
    ):
        defined = {attr for attr in vars(printer_class) if attr.startswith("_print_")}
        missed = sorted(defined - ran - REFUSAL_ONLY_METHODS)
        assert not missed, (
            f"the {label} emitter's {missed} were not reached by any case in this file. "
            f"Add a case that produces one, so the target is asked whether it accepts "
            f"what they write."
        )


# ─── the boolean family issue #460 left behind ───────────────────────────────

#: Every way a rate law may spell a condition, each one text the engine's own
#: expression parser accepts. The four word operators at the end are the ones
#: that matter: the parser has `xor`, `nand`, `nor` and `xnor` as operators, so a
#: model may legally be written with any of them, and each is a boolean neither
#: emitter has a spelling for.
CONDITION_TEXTS = (
    "if(x0>1,k1,0)",
    "if(x0>1 and x1>1,k1,0)",
    "if(x0>1 or x1>1,k1,0)",
    "if(not(x0>1),k1,0)",
    "if(not(x0>1 and x1>1),k1,0)",
    "if(x0>1 && x1>1,k1,0)",
    "if(x0>1 || x1>1,k1,0)",
    "if(x0==1,k1,0)",
    "if(x0!=1,k1,0)",
    "if((x0>1)==(x1>1),k1,0)",
    "if((x0>1)!=(x1>1),k1,0)",
    "if((x0>1 and not(x1>1)) or (not(x0>1) and x1>1),k1,0)",
    "if(if(x0>1,1,0)==1,k1,0)",
    "if(x0>1,if(x1>1,k1,0),0)",
    "if(x0>1 xor x1>1,k1,0)",
    "if(x0>1 xnor x1>1,k1,0)",
    "if(x0>1 nand x1>1,k1,0)",
    "if(x0>1 nor x1>1,k1,0)",
)

#: The rate laws above that the loader's sympy parse cannot read at all. All four
#: are the word operators: nothing rewrites them on the way into sympy, so the
#: parse raises and the model keeps the finite-difference Jacobian. That is the
#: fail-safe answer, and it is also why no `Xor` has ever reached an emitter.
#:
#: Pinned rather than ignored. Teaching the parse one of these is the change that
#: makes `Xor` reachable, and this is what says so.
CONDITIONS_THE_PARSE_DECLINES = frozenset(
    {
        "if(x0>1 xor x1>1,k1,0)",
        "if(x0>1 xnor x1>1,k1,0)",
        "if(x0>1 nand x1>1,k1,0)",
        "if(x0>1 nor x1>1,k1,0)",
    }
)


def test_no_rate_law_produces_a_boolean_node_with_no_spelling():
    """The reachability question #460 left open, asked mechanically.

    ``Xor``, ``Implies`` and ``Equivalent`` are refused by ``_is_emittable``, and
    that guard was written from a reading of sympy's printer rather than from
    anything that reaches it. ``Xor`` is the reason it exists: sympy prints it
    infix as ``^``, the engine reads ``^`` as exponentiation, and the result is a
    wrong Jacobian entry instead of a refusal.

    So rather than assert nothing reaches them, put every condition a rate law
    may be written in through the loader's own parse and look at what comes out.
    Today every one of these gives back ``And``, ``Or``, ``Not``, ``Eq`` or
    ``Ne``, all five of which are spelled. The four word operators parse to
    nothing at all, which is the fail-safe answer: the model keeps running on the
    finite-difference Jacobian.

    The day one of them starts producing an ``Xor``, this fails and says to give
    it a spelling first.
    """
    from sympy.logic.boolalg import BooleanFunction

    spelled = set(jac._EMITTABLE_BOOLEAN_FUNCS)
    produced: dict[str, set[str]] = {}
    declined = set()
    for text in CONDITION_TEXTS:
        try:
            parsed = jac._exprtk_to_sympy(text)
        except Exception:
            parsed = None
        if parsed is None:
            declined.add(text)
            continue
        for node in parsed.atoms(BooleanFunction):
            produced.setdefault(type(node).__name__, set()).add(text)

    unspelled = {name: sorted(texts) for name, texts in produced.items() if name not in spelled}
    assert not unspelled, (
        f"these rate laws now produce boolean nodes no emitter can spell: {unspelled}. "
        f"Give them a spelling in both emitters before this becomes reachable — the "
        f"engine takes sympy's `^` for Xor as exponentiation and computes a wrong "
        f"number rather than refusing (issue #464)."
    )
    # The three connectives every condition is built from do come out, so the
    # sweep is not passing by reading nothing.
    assert {"And", "Or", "Not"} <= set(produced)
    # And the ones that read nothing are the ones that are supposed to.
    assert declined == CONDITIONS_THE_PARSE_DECLINES, (
        "the set of rate laws the parse cannot read has moved. Anything that "
        "started being read may now be building a boolean node the emitters have "
        "no spelling for, so check what it produces before accepting this."
    )


@pytest.mark.parametrize("name", ["Xor", "Implies", "Equivalent"])
def test_a_boolean_node_with_no_spelling_reaches_neither_target(name):
    """The other half of the same question. If one ever is produced, both
    emitters have to decline it rather than write something the target reads as
    arithmetic."""
    from sympy.logic import boolalg

    node = getattr(boolalg, name)(x0 > 0, x1 > 0)
    assert type(node).__name__ == name, "sympy no longer builds this node"
    assert sympy_to_exprtk(node) is None
    assert sympy_to_c(node, C_SYMBOLS.get) is None
