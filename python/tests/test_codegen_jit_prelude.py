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
"""

from __future__ import annotations

import inspect
import os
import re
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
    for a .net model on the JIT backend, with the output-sensitivity block
    switched on the way ``sensitivity_params`` switches it on."""
    model._want_output_sens = True
    return cg.prepare_codegen_source(model._net_path, model, emit_jac=True)


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
# inline, so this is the model that puts the two names in the RHS. It cannot also
# serve the sensitivity sources: a Functional reaction is exactly what the .net
# sensitivity path declines, which is why the two fixtures are separate.
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

# The four generated sources that open with their own ``#include <math.h>``, each
# paired with a model that reaches it. The Jacobian, the output evaluator and the
# output-sensitivity block emit no include of their own and only ever compile
# behind an RHS, so they are outside this contract by construction.
STANDALONE_SOURCES = [
    ("generate_rhs_c", FUNCTIONAL_CONSTANT, lambda net, model: cg.generate_rhs_c(str(net))),
    (
        "generate_rhs_from_model",
        FUNCTIONAL_CONSTANT,
        lambda net, model: cg.generate_rhs_from_model(model),
    ),
    (
        "generate_sens_rhs_c",
        DERIVED_CONSTANT,
        lambda net, model: cg.generate_sens_rhs_c(str(net)),
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
        ``#include <math.h>``. Three emitters, four entry points — the two
        sensitivity generators share ``_emit_sens_rhs_body``."""
        emitters = len(re.findall(r'_emit\("#include <math\.h>"\)', inspect.getsource(cg)))
        assert emitters == 3, (
            f"_codegen.py emits `#include <math.h>` from {emitters} places, not the 3 the "
            f"table above covers. If a new generated source now stands on its own, give it "
            f"an entry there and its own M_PI / M_E defines (GH #470). If the three headers "
            f"were merged into one emitter instead, drop the entries that no longer exist "
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
