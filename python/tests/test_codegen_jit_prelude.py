"""GH #85 — the MIR JIT prelude must supply every libc name the codegen emits.

``MirJit`` does not hand c2mir the generated C source unchanged. c2mir cannot
parse the platform SDK's ``<math.h>`` / ``<stdlib.h>`` / ``<string.h>`` (on macOS
it stops at "Unsupported compiler detected"), so ``make_jit_source`` strips every
``#include <…>`` line and prepends a prelude that re-declares the libc/libm
*functions* the RHS can call.

Stripping a header also strips its **types and macros**, and the prelude declared
none of those. #85 is the first one that mattered: ``bngsim_codegen_output_sens``
(GH #198) casts its column index with ``(size_t)``, and with ``size_t`` unknown
as a typedef name ``(size_t)_c`` stops being a cast — the whole declaration fails
to parse and c2mir reports the *next* token, "syntax error on double (expected
'<statement>')". So a Functional model built with ``sensitivity_params`` could
not run on the JIT backend at all, on any platform.

``NULL`` and ``NAN`` are the same hole one step behind it, and they are not
theoretical: the no-observables function-block call args pass ``NULL`` for
``obs``, and #198 emits ``NAN`` as the sentinel for a function it cannot
differentiate. Both were unreachable only because ``size_t`` failed first. That
is what this file guards — the *class*, not the one instance.

The prelude is only half of the deal. ``mir_jit.hpp`` says the other half out
loud: "M_PI/M_E are still provided by the generated source's own #ifndef/#define
blocks." Nothing checked that, and the sensitivity source was not doing it (GH
#470), so :class:`TestEachSourceDefinesTheMathMacrosItUses` checks it here, one
generated source at a time.

Two levels, because the end-to-end one only bites where the JIT actually runs:

* :class:`TestPreludeSuppliesWhatTheCodegenEmits` compares emitted C against
  ``mir_jit.hpp`` as text, so it runs on every backend and on a build with no
  MIR at all — including under the pre-push hook, which is the only automated
  gate ``python/tests`` has (CI runs this file only in the MIR job).
* :class:`TestTheGeneratedSourceCompiles` puts each of the three through a real
  ``Simulator``: a formality under ``cc``, the actual #85 reproducer under
  ``BNGSIM_CODEGEN_JIT=mir``.

The prelude also bounds the other direction: it is the whole list of libc/libm
names a JIT'd source may import. c2mir accepts a call to an undeclared function
(an implicit ``int name()``) where cc refuses it, so a bare call the emitter has
no C for used to reach MIR_link, and either ended the process or was bound to an
unrelated global of the same name. :class:`TestACallThePreludeDoesNotDeclareIsRefused`
pins the refusal ``MirJit`` now makes instead.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"
BNGSIM_ROOT = Path(__file__).resolve().parents[2]
MIR_JIT_HPP = BNGSIM_ROOT / "include" / "bngsim" / "mir_jit.hpp"


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# One model per name the prelude has to supply. They are separate models on
# purpose: ``size_t`` is emitted unconditionally by the #198 block, so a single
# model would have masked NULL and NAN behind it exactly the way #85 did.

# size_t: a Functional rate law + sensitivity_params is the #85 reproducer.
SIR = """\
begin parameters
    1 S0     2e7  # Constant
    2 I0     1  # Constant
    3 beta   1/S0  # ConstantExpression
    4 gamma  1/7  # Constant
end parameters
begin functions
    1 betaI() beta*I
end functions
begin species
    1 person(state~S) S0
    2 person(state~I) I0
    3 person(state~R) 0
end species
begin reactions
    1 1 2 betaI #_R1
    2 2 3 gamma #_R2
end reactions
begin groups
    1 S                    1
    2 I                    2
    3 R                    3
end groups
"""

# NULL: a function that reads no observable, in a model that declares none, so
# both function blocks take their `obs` argument as a null pointer.
NO_OBS = """\
begin parameters
    1 k      0.3  # Constant
    2 A0     10  # Constant
end parameters
begin functions
    1 kf() k*2
end functions
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf #_R1
end reactions
begin groups
end groups
"""

# NAN: the #198 sentinel for a function whose output sensitivity is refused.
# The shared fixture from test_expression_output_sensitivities.py — if(), abs(),
# floor() and friends, all of which the *value* codegen still compiles.
UNSUPPORTED_NET = DATA_DIR / "expr_sens_unsupported.net"


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _jit_source_input(model) -> str:
    """The C source ``MirJit`` is constructed from — i.e. what reaches
    ``make_jit_source``. Same entry point ``_auto_codegen_for_sensitivity`` uses
    on the JIT backend (for a .net model too, since #803), with the
    output-sensitivity block switched on the way ``sensitivity_params`` switches
    it on."""
    model._want_output_sens = True
    return cg.prepare_model_codegen_source(model)


_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def _names_used(c_source: str, names) -> set[str]:
    """Which of ``names`` the generated source actually references. Comments are
    stripped first — the ``CodegenUserData`` typedef carries a "may be NULL"
    comment that would otherwise report every model as using ``NULL``."""
    code = _COMMENTS.sub(" ", c_source)
    return {
        n for n in names if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", code)
    }


def _jit_prelude_text() -> str:
    """The prelude string ``MirJit::jit_prelude()`` returns, reassembled from the
    header's concatenated C string literals."""
    src = _COMMENTS.sub(" ", MIR_JIT_HPP.read_text(encoding="utf-8"))
    start = src.index("static const char *jit_prelude()")
    body = src[start : src.index("\n    }", start)]
    chunks = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    assert chunks, "could not find the prelude string literals in mir_jit.hpp"
    return "".join(chunks).replace("\\n", "\n").replace('\\"', '"')


# Names the generated C can only get from a system header, so every one of them
# is a candidate #85. Deliberately wider than what is emitted today: the point is
# to fail the day the codegen starts emitting a *new* one.
#
# M_PI and M_E are deliberately NOT here, even though they belong to the same
# family. This check reads the COMBINED source, which always opens with the RHS,
# and the RHS always carries a define for both — so listing them here would
# report every model as covered no matter which piece of the source actually
# names them. They get their own per-source check further down (GH #470).
HEADER_NAMES = (
    "size_t",
    "ptrdiff_t",
    "NULL",
    "NAN",
    "INFINITY",
    "HUGE_VAL",
    "DBL_MAX",
    "DBL_MIN",
    "DBL_EPSILON",
    "SIZE_MAX",
)

# Names c2mir can supply from a header it bundles in memory, which the prelude
# may therefore satisfy with a plain #include. Only <stddef.h> qualifies — it is
# in every target's TARGET_STD_INCLUDES, whereas <math.h>/<stdlib.h>/<string.h>
# have no bundled copy and fall through to the unparseable platform SDK.
#
# NULL is NOT listed, even though <stddef.h> nominally defines it: c2mir's copy
# omits NULL on macOS-x86_64 and on Windows (see mirc_x86_64_stddef.h), so an
# #include alone would leave two of the four MIR CI legs broken. It has to be
# #define'd in the prelude.
BUNDLED_HEADER = {"size_t": "stddef.h", "ptrdiff_t": "stddef.h"}


def _defines(c_source: str, name: str) -> bool:
    """Whether ``c_source`` carries a ``#define`` for ``name``."""
    return bool(re.search(rf"^\s*#\s*define\s+{re.escape(name)}\b", c_source, re.M))


def _prelude_supplies(prelude: str, name: str) -> bool:
    if _defines(prelude, name):
        return True
    if re.search(rf"\btypedef\b[^;\n]*\b{re.escape(name)}\s*;", prelude):
        return True
    header = BUNDLED_HEADER.get(name)
    return bool(header and re.search(rf"^\s*#\s*include\s*<{re.escape(header)}>", prelude, re.M))


@pytest.fixture(scope="module")
def prelude() -> str:
    return _jit_prelude_text()


class TestPreludeSuppliesWhatTheCodegenEmits:
    """Text-level, so it holds on a build without MIR — and fails on the
    pre-#85 prelude for all three names at once."""

    @pytest.mark.parametrize("fixture", ["sir", "no_obs", "unsupported"])
    def test_every_header_name_in_the_emitted_source(self, tmp_path, prelude, fixture):
        if fixture == "unsupported":
            model = bngsim.Model.from_net(str(UNSUPPORTED_NET))
        else:
            model = _model(tmp_path, SIR if fixture == "sir" else NO_OBS)
        used = _names_used(_jit_source_input(model), HEADER_NAMES)
        missing = sorted(n for n in used if not _prelude_supplies(prelude, n))
        assert not missing, (
            f"{fixture}: the generated source names {missing}, which MirJit's prelude does "
            f"not supply. Every `#include <…>` is stripped on the JIT path, so this is a "
            f"c2mir compile failure of the whole module, not a decline of one feature "
            f"(GH #85). Add it to jit_prelude() in include/bngsim/mir_jit.hpp."
        )

    def test_the_fixtures_still_cover_the_three_names(self, tmp_path):
        """Keeps the guard above honest. If the codegen stops emitting one of
        these, the parametrized test keeps passing while covering less — so say
        so here instead, and pick a fixture that does exercise it."""
        used: set[str] = set()
        for model in (
            _model(tmp_path, SIR),
            _model(tmp_path, NO_OBS, name="noobs.net"),
            bngsim.Model.from_net(str(UNSUPPORTED_NET)),
        ):
            used |= _names_used(_jit_source_input(model), HEADER_NAMES)
        assert {"size_t", "NULL", "NAN"} <= used, (
            f"fixtures cover {sorted(used)}; they are supposed to reach size_t (#198 index "
            f"casts), NULL (no-observables function-block args) and NAN (the #198 "
            f"unsupported-function sentinel)."
        )


# ─── the other half of the contract (GH #470) ───────────────────────────────
#
# A rate law written over the engine's ``_pi`` or ``_e`` reaches the generated C
# as ``M_PI`` / ``M_E``, and ``<math.h>`` is not a dependable place to get either
# one: MSVC defines neither without ``_USE_MATH_DEFINES``, and glibc hides both
# in strict C mode. On the JIT path the include is stripped outright, so a
# ``#define`` in the source is the only supplier there is. Every generated source
# that means to compile on its own therefore has to define what it uses.
MATH_H_MACROS = ("M_PI", "M_E")

# A derived rate constant written over both constants. The chain rule for it is
# what lands in the sensitivity source, so this is the model that puts the two
# names in *that* source; the RHS reads the finished value out of p[] and names
# neither.
DERIVED_CONSTANT = """\
begin parameters
    1 k1   0.3  # Constant
    2 kpe  k1*_pi*_e
end parameters
begin species
    1 A() 10.0
    2 B() 0.0
end species
begin reactions
    1 1 2 kpe #_R1
end reactions
begin groups
    1 Aobs                    1
end groups
"""

# A functional rate law over both constants. The RHS evaluates the function
# inline, so this is the model that puts the two names in the RHS. It could not
# also serve the sensitivity sources while the .net sensitivity emitter (retired
# by #803) declined every Functional reaction, which is why the two fixtures are
# separate.
FUNCTIONAL_CONSTANT = """\
begin parameters
    1 k1   0.3  # Constant
end parameters
begin functions
    1 kpe() k1*_pi*_e*Aobs
end functions
begin species
    1 A() 10.0
    2 B() 0.0
end species
begin reactions
    1 1 2 kpe #_R1
end reactions
begin groups
    1 Aobs                    1
end groups
"""

# The generated sources that open with their own ``#include <math.h>``, each
# paired with a model that reaches it. The Jacobian, the output evaluator and the
# output-sensitivity block emit no include of their own and only ever compile
# behind an RHS, so they are outside this contract by construction.
STANDALONE_SOURCES = [
    (
        "generate_rhs_from_model",
        FUNCTIONAL_CONSTANT,
        lambda net, model: cg.generate_rhs_from_model(model),
    ),
    (
        "generate_sens_from_model",
        DERIVED_CONSTANT,
        lambda net, model: cg.generate_sens_from_model(model),
    ),
]

# A source's own guard lines mention a macro without using it. Drop them before
# asking what the source reads, or a source that defines M_PI and never touches
# it again would report as using it and the fixture check below would pass on
# nothing.
_MACRO_GUARD_LINE = re.compile(
    r"^[ \t]*#[ \t]*(?:ifndef|define)[ \t]+(?:" + "|".join(MATH_H_MACROS) + r")\b.*$", re.M
)


def _math_macros_used(c_source: str) -> set[str]:
    return _names_used(_MACRO_GUARD_LINE.sub(" ", c_source), MATH_H_MACROS)


class TestEachSourceDefinesTheMathMacrosItUses:
    """One source at a time, never the combined text.

    The combined source is what hid this: both places that assemble it put the
    RHS first and append the sensitivity source to it, so the RHS's defines
    covered the whole translation unit and the sensitivity source was never asked
    to carry its own. Asking each source on its own is the only form of the
    question that can fail.
    """

    @pytest.mark.parametrize(
        ("name", "net_text", "generate"),
        STANDALONE_SOURCES,
        ids=[s[0] for s in STANDALONE_SOURCES],
    )
    def test_it_defines_every_math_macro_it_uses(self, tmp_path, name, net_text, generate):
        model = _model(tmp_path, net_text, name="constants.net")
        c_source = generate(model._net_path, model)
        assert c_source is not None, f"{name} declined this model, so the case checks nothing"
        assert "#include <math.h>" in c_source, (
            f"{name} no longer emits an `#include <math.h>` of its own, so it is a fragment "
            f"appended to someone else's translation unit rather than a source that stands "
            f"alone. Take it out of STANDALONE_SOURCES above."
        )
        used = _math_macros_used(c_source)
        assert used == set(MATH_H_MACROS), (
            f"{name} uses {sorted(used)} on this model, not {sorted(MATH_H_MACROS)}. The case "
            f"covers less than it claims to — give it a model that reaches both names."
        )
        missing = sorted(n for n in used if not _defines(c_source, n))
        assert not missing, (
            f"{name} writes {missing} into its own translation unit and defines nothing for "
            f"it. Today that compiles only because this source is concatenated after the RHS, "
            f"which does define both. Add the same `#ifndef M_PI` / `#ifndef M_E` block the "
            f"RHS emitters open with (GH #470)."
        )

    def test_no_further_standalone_source_has_appeared(self):
        """The table above is written by hand, so it cannot notice a new source.
        The emitters can: a source that means to stand on its own writes its own
        ``#include <math.h>``. Two emitters, two entry points: the RHS and the
        sensitivity RHS. (Three emitters and four entry points until #803 retired
        the ``.net`` RHS and sensitivity generators.)"""
        emitters = len(re.findall(r'_emit\("#include <math\.h>"\)', inspect.getsource(cg)))
        assert emitters == 2, (
            f"_codegen.py emits `#include <math.h>` from {emitters} places, not the 2 the "
            f"table above covers. If a new generated source now stands on its own, give it "
            f"an entry there and its own M_PI / M_E defines (GH #470). If the two headers "
            f"were merged into one emitter instead, drop the entry that no longer exists "
            f"and lower this count."
        )


class TestTheGeneratedSourceCompiles:
    """End-to-end on whichever backend is selected. Under
    ``BNGSIM_CODEGEN_JIT=mir`` each of these raised
    ``MirJit: c2mir failed to compile generated RHS source`` before the fix."""

    @pytest.mark.parametrize(
        ("fixture", "params"),
        [("sir", ["gamma"]), ("no_obs", ["k"]), ("unsupported", ["k1"])],
        ids=["size_t", "NULL", "NAN"],
    )
    def test_a_sensitivity_run_compiles_and_integrates(self, tmp_path, fixture, params):
        if fixture == "unsupported":
            model = bngsim.Model.from_net(str(UNSUPPORTED_NET))
        else:
            model = _model(tmp_path, SIR if fixture == "sir" else NO_OBS)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
        run = sim.run(t_span=(0.0, 5.0), n_points=6)
        sens = np.asarray(run.sensitivities)
        assert sens.shape == (6, model._core.n_species, len(params))
        # The NaN sentinel lives in the *function* sensitivities, which this
        # selector does not return; the state sensitivities are always real.
        assert np.isfinite(sens).all()

    @pytest.mark.parametrize("base", [10, 2], ids=["M_LN10", "M_LN2"])
    def test_a_log_parameterised_rate_constant_has_its_sensitivity(self, tmp_path, base):
        """``k = base^logk``, the usual fitting parameterisation. sympy writes
        dk/dlogk as ``pow(base, logk)*M_LN10`` (``M_LN2``), and no generated
        source defined either, so under ``BNGSIM_CODEGEN_JIT=mir`` c2mir stopped
        at "undeclared identifier M_LN10". A(t) = A0 exp(-k t) gives the oracle:
        dA/dlogk = -A0 t k ln(base) exp(-k t)."""
        model = _model(tmp_path, LOG_PARAMETERISED.format(base=base))
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["logk"])
        run = sim.run(t_span=(0.0, 1.0), n_points=2)
        k = float(base) ** -0.3
        exact = -2.0 * k * np.log(base) * np.exp(-k)
        got = np.asarray(run.sensitivities)[-1, 0, 0]
        assert got == pytest.approx(exact, rel=1e-5)

    @pytest.mark.parametrize(("base", "power"), [(10, 20), (6, 30)])
    def test_an_integer_past_64_bits_in_a_derivative_is_a_double(self, tmp_path, base, power):
        """``k = p*base^power`` with ``p = base^-power``: sympy prints dk/dp as
        the integer base^power, which is past 2^64. Printed as a bare integer
        literal, cc refused to build the source and c2mir read it as the 64-bit
        maximum, so dA/dp came back 82% (10^20) or 99.99% (6^30) small with no
        warning. k = 1, so dA/dp = -A0 t base^power exp(-t)."""
        text = LOG_PARAMETERISED.replace("logk -0.3", f"p {float(base) ** -power!r}")
        text = text.replace("{base}^logk", f"p*{base}^{power}")
        model = _model(tmp_path, text)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["p"])
        run = sim.run(t_span=(0.0, 1.0), n_points=2)
        exact = -2.0 * float(base) ** power * np.exp(-1.0)
        assert np.asarray(run.sensitivities)[-1, 0, 0] == pytest.approx(exact, rel=1e-5)

    def test_an_integer_exponent_that_is_not_exactly_a_double_is_refused(self):
        """A double literal would round such an exponent, and its parity with
        it: d/dp p^(10^20) at p = -1 printed as pow(p, 1e20) is +1 where
        (-1)^(10^20 - 1) is -1. The printer refuses it, which declines the
        analytic derivative, rather than print it."""
        from bngsim import _term_order

        sympy = pytest.importorskip("sympy")
        p = sympy.Symbol("p")
        assert _term_order.ccode(p**1000) == "pow(p, 1000.0)"
        assert _term_order.ccode(p ** (2**53)) == f"pow(p, {2**53}.0)"
        for n in (2**53 + 1, 10**20 - 1):
            with pytest.raises(ValueError, match="not exactly a double"):
                _term_order.ccode(p**n)

    def test_infinity_in_an_untaken_branch_compiles(self, tmp_path):
        """sympy prints oo as ``INFINITY``, a <math.h> macro the JIT strips;
        ``k = if(p > 1, p, p*abs(1/0))`` puts it in the branch not taken. k = p
        = 2, so dA/dp = -A0 t exp(-2 t)."""
        text = LOG_PARAMETERISED.replace("logk -0.3", "p 2.0")
        text = text.replace("{base}^logk", "if(p>1,p,p*abs(1/0))")
        model = _model(tmp_path, text)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["p"])
        run = sim.run(t_span=(0.0, 1.0), n_points=2)
        exact = -2.0 * np.exp(-2.0)
        assert np.asarray(run.sensitivities)[-1, 0, 0] == pytest.approx(exact, rel=1e-5)


# The model for the tests above: one first-order decay at k = base^logk.
LOG_PARAMETERISED = """\
begin parameters
    1 logk -0.3
    2 k    {base}^logk
end parameters
begin species
    1 A() 2.0
    2 B() 0
end species
begin reactions
    1 1 2 k #_R1
end reactions
begin groups
    1 Atot 1
end groups
"""


class TestEachSourceDefinesEveryConstantSympyCanPrint:
    """Text-level, so it holds on every backend. M_PI and M_E were the only
    constants a source defined (GH #470), but sympy's C printer has a dozen
    (``sympy.printing.c.get_math_macros``) and the sensitivity source is printed
    by it. Each standalone source has to define every one, with the double
    <math.h> would have given."""

    @pytest.mark.parametrize(
        ("name", "net_text", "generate"),
        STANDALONE_SOURCES,
        ids=[s[0] for s in STANDALONE_SOURCES],
    )
    def test_with_the_value_math_h_gives(self, tmp_path, name, net_text, generate):
        sympy = pytest.importorskip("sympy")
        from sympy.printing.c import get_math_macros

        # Several expressions map to one macro, and the sympy.codegen spellings
        # (Sqrt(2)) do not evaluate; any one that does gives the value.
        exact = {}
        for expr, macro in get_math_macros().items():
            value = sympy.N(expr, 40)
            if value.is_real:
                exact.setdefault(macro, value)
        assert set(exact) == set(get_math_macros().values())
        c_source = generate(None, _model(tmp_path, net_text, name="constants.net"))
        for macro, value in exact.items():
            found = re.search(rf"^\s*#\s*define\s+{macro}\s+(\S+)", c_source, re.M)
            assert found, f"{name} does not define {macro}, which sympy's C printer writes"
            assert float(found.group(1)) == float(value), (
                f"{name} defines {macro} as {found.group(1)}, not the double nearest {value}"
            )


# ─── a call the prelude does not declare ───────────────────────────────────
#
# The emitter used to write some ExprTk functions through as bare calls it had no
# C for (frac, inrange, iclamp, roundn, root, logn, sinc, equal, a table
# function's name). cc refuses such a source as a call to an undeclared function.
# c2mir instead takes the call as an implicit ``int name()``, the C89 rule, so on
# the JIT backend the name reached MIR_link as an import of its own. One the
# resolver could not find, MIR's default error function reported by exit(1),
# which ended the Python process mid-run. One dlsym could find was bound to
# whatever global has that spelling: on macOS a private framework loaded into
# the Python process exports an ``equal``, and a run of ``k*(1 + equal(k, 0.5))``
# came back with B(1) = 0.787 where the answer is 1.264, with no warning. MirJit
# now refuses every import outside the prelude's table before it links.
#
# Since issue #1212 the emitter writes C for each of those, and declines any call
# it has none for before a source exists, so no model's own source carries such
# a call to the JIT any more. The JIT's refusal is still its last line, and is
# pinned here on a source made by hand: the one generated for a law that calls
# exp, with the call renamed.

_UNDECLARED_NET = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 2.0
    2 B() 0
end species
begin functions
    1 f() k*(1 + exp(k))
end functions
begin reactions
    1 1 2 f #_R1
end reactions
begin groups
    1 Atot 1
end groups
"""

# name -> the call that replaces exp(p[0]) in the generated source.
_UNDECLARED_CALLS = {
    "equal": "equal(p[0], 0.5)",
    "frac": "frac(p[0] + 2.0)",
}

_EXP_CALL = "exp(p[0])"

# Run in a child: before the fix the frac case exit(1)ed, which would have taken
# the pytest worker with it rather than failing this test. The source is handed
# over on the model, which is where automatic codegen takes a JIT source from.
_RUN_IN_A_CHILD = """
import sys
import bngsim
from bngsim import _codegen as cg
model = bngsim.Model.from_net(sys.argv[1])
model._codegen_c_source = cg.prepare_model_codegen_source(model).replace(sys.argv[2], sys.argv[3])
sim = bngsim.Simulator(model, method="ode")
try:
    sim.run(t_span=(0.0, 5.0), n_points=6)
except Exception as e:
    print(f"refused: {e}")
else:
    print("ran")
"""


class TestACallThePreludeDoesNotDeclareIsRefused:
    @pytest.mark.parametrize("name", list(_UNDECLARED_CALLS))
    def test_the_emitter_writes_c_for_it_now(self, tmp_path, prelude, name):
        """What makes a hand-built source necessary: the emitter's own source for
        the function no longer calls it bare (issue #1212)."""
        law = {"equal": "k*(1 + equal(k, 0.5))", "frac": "k*(1 + frac(k + 2))"}[name]
        model = _model(tmp_path, _UNDECLARED_NET.replace("k*(1 + exp(k))", law))
        code = _COMMENTS.sub(" ", cg.prepare_model_codegen_source(model))
        assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(", code)
        assert re.search(rf"\bbngsim_{re.escape(name)}\s*\(", code)

    @pytest.mark.parametrize("name", list(_UNDECLARED_CALLS))
    def test_the_hand_built_source_calls_it_bare(self, tmp_path, prelude, name):
        """Keeps the end-to-end case honest: it covers the hole only while the
        source calls ``name`` without defining it, and the prelude does not
        declare it."""
        model = _model(tmp_path, _UNDECLARED_NET)
        source = cg.prepare_model_codegen_source(model)
        assert _EXP_CALL in source
        code = _COMMENTS.sub(" ", source.replace(_EXP_CALL, _UNDECLARED_CALLS[name]))
        call = rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\("
        assert re.search(call, code), f"the source does not call {name}()"
        assert not re.search(rf"\b(?:double|int)\s+{re.escape(name)}\s*\(", code)
        assert not re.search(call, prelude)

    @pytest.mark.skipif(
        not getattr(bngsim, "HAS_MIR", False),
        reason="bngsim built without the MIR backend (configure with -DBNGSIM_ENABLE_MIR=ON)",
    )
    @pytest.mark.parametrize("name", list(_UNDECLARED_CALLS))
    def test_the_jit_refuses_it_by_name(self, tmp_path, name):
        net = tmp_path / "m.net"
        net.write_text(_UNDECLARED_NET)
        child = subprocess.run(
            [sys.executable, "-c", _RUN_IN_A_CHILD, str(net), _EXP_CALL, _UNDECLARED_CALLS[name]],
            env={**os.environ, "BNGSIM_CODEGEN_JIT": "mir"},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert child.returncode == 0, child.stderr[-2000:]
        assert f"refers to {name}," in child.stdout, child.stdout + child.stderr[-2000:]
