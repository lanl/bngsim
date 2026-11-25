"""A chain of ``if()``s is one ``?:`` chain in the generated C (issue #1814).

A table of values over time is a chain of nested ``if()``s, one link a day:
``if(time()<=1, v1, if(time()<=2, v2, ... 0))``. The Mallela 2024 ``phoenix``
model's ``v_rate()`` is 340 of them. The C of the rate law (``_replace_if_calls``)
and of its derivative (the C printer's ``Piecewise``) wrote each link inside the
one before it, two brackets deeper, and passed clang's limit of 256 brackets at
128 links. A forward-sensitivity run, which needs the compiled right-hand side,
failed to build with "bracket nesting level exceeded maximum of 256"; a plain
run fell back to the interpreter. GCC has no such limit, so the C is measured
here as well as compiled.

An ``if()`` that is all of another's false branch is now the next link of one C
chain. The sympy side is left as it was: the parse text nests and the
``Piecewise`` it builds is the right-nested one. A flat ``Piecewise`` is not the
same thing to sympy (``_codegen._if_call_args`` says why), and the last section
holds a model of each way it differs to the answers the nested one gives.

These tests cover:

- the bracket depth of the C does not grow with the chain, and the C reads as
  the nested form did: the same text but for the brackets, and the same value at
  every link, compiled;
- the parse keeps the nested ``Piecewise``;
- a chain of 1000 links is walked without a Python frame per link, in C, in the
  parse text and in the scan for conditions. The models here stop at 340 links:
  each of those walks reads the rest of the chain again at every link;
- a forward-sensitivity run of a 128-link model builds and returns
  dA(5)/dk = -1.488792, under clang where there is one;
- window chains, a NaN condition and gated ramps run as the nested form does.
"""

from __future__ import annotations

import math
import re
import shlex
import shutil
import sys
import types

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim import _jacobian as J  # noqa: E402
from bngsim import _jax_rhs as jr  # noqa: E402
from bngsim._switch_sensitivity import _iter_if_conditions  # noqa: E402

_CLANG = shutil.which("clang")
needs_clang = pytest.mark.skipif(_CLANG is None, reason="no clang on PATH")


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


needs_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

# clang's default -fbracket-depth.
CLANG_BRACKET_LIMIT = 256


def _day_value(i: int) -> float:
    return 0.001 * (i % 7 + 1)


def _chain(n: int) -> str:
    """The issue's ``v_rate()``: a value for each of *n* days, as nested if()s."""
    return "".join(f"if((time()<={i + 1}),{_day_value(i)}," for i in range(n)) + "0" + ")" * n


def _net(n: int) -> str:
    return f"""begin parameters
    1 k  0.5
    2 A0 100
end parameters
begin functions
    1 v_rate() {_chain(n)}
    2 r() k*v_rate()
end functions
begin species
    1 A() A0
end species
begin reactions
    1 1 0 r
end reactions
begin groups
    1 Atot 1
end groups
"""


def _load(tmp_path, text: str, name: str = "m") -> bngsim.Model:
    path = tmp_path / f"{name}.net"
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _model(tmp_path, n: int) -> bngsim.Model:
    return _load(tmp_path, _net(n), f"chain_{n}")


def _bracket_depth(src: str) -> int:
    """The deepest nesting of (), [] and {} in C source, outside comments and
    string literals."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    depth = deepest = 0
    for ch in src:
        if ch in "([{":
            depth += 1
            deepest = max(deepest, depth)
        elif ch in ")]}":
            depth -= 1
    return deepest


# A(5) and dA(5)/dk: on [0, 5] only the first five days are read, so every chain
# of five or more has A' = -k·v(t)·A with ∫v = 0.015.
A5 = 100.0 * math.exp(-0.5 * 0.015)
S5 = -0.015 * A5


# ── The C of an if() ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expr, expect",
    [
        ("if(a,b,c)", "((a) ? (b) : (c))"),
        ("if(a,b,if(c,d,e))", "((a) ? (b) : (c) ? (d) : (e))"),
        ("if(a, b, if (c, d, if(x, y, z)) )", "((a) ? (b) : (c) ? (d) : (x) ? (y) : (z))"),
        ("1+if(a,b,if(c,d,e))*if(p,q,r)", "1+((a) ? (b) : (c) ? (d) : (e))*((p) ? (q) : (r))"),
        # In brackets that hold nothing else.
        ("if(a,b,(if(c,d,e)))", "((a) ? (b) : (c) ? (d) : (e))"),
        ("if(a,b, ( (if(c,d,(if(x,y,z)))) ) )", "((a) ? (b) : (c) ? (d) : (x) ? (y) : (z))"),
    ],
)
def test_an_if_that_is_all_of_the_false_branch_continues_the_chain(expr, expect):
    assert cg._replace_if_calls(expr) == expect


@pytest.mark.parametrize(
    "expr, expect",
    [
        # In the true branch, or in the condition.
        ("if(a,if(c,d,e),f)", "((a) ? (((c) ? (d) : (e))) : (f))"),
        ("if(if(a,b,c),d,e)", "((((a) ? (b) : (c))) ? (d) : (e))"),
        # Part of the false branch, not all of it.
        ("if(a,b,if(c,d,e)*2)", "((a) ? (b) : (((c) ? (d) : (e))*2))"),
        ("if(a,b,if(c,d,e)+if(p,q,r))", "((a) ? (b) : (((c) ? (d) : (e))+((p) ? (q) : (r))))"),
        ("if(a,b,(if(c,d,e))*2)", "((a) ? (b) : ((((c) ? (d) : (e)))*2))"),
        ("if(a,b,(if(c,d,e))+(x))", "((a) ? (b) : ((((c) ? (d) : (e)))+(x)))"),
    ],
)
def test_any_other_if_in_an_if_is_nested_as_before(expr, expect):
    assert cg._replace_if_calls(expr) == expect


def test_a_chain_reads_its_comparisons_as_the_nested_form_did():
    # A condition is left an int, a value written as a double (issue #1495),
    # at every link: the C is what it was but for the brackets.
    c = cg._expr_to_c("if(k, A>thr, if(k>1, B>thr, A>0))/k", ["thr", "k"], [], ["A", "B"], [])
    assert c == (
        "((p[1]) ? ((obs[0]>p[0]) ? 1.0 : 0.0) : (p[1]>1.0) ? ((obs[1]>p[0]) ? 1.0 : 0.0)"
        " : ((obs[0]>0.0) ? 1.0 : 0.0))/p[1]"
    )


def _bracketed_chain(n: int) -> str:
    """The issue's table with each false branch in brackets of its own."""
    return "".join(f"if((time()<={i + 1}),{_day_value(i)},(" for i in range(n)) + "0" + "))" * n


@pytest.mark.parametrize("chain", [_chain, _bracketed_chain])
@pytest.mark.parametrize("n", [128, 340])
def test_the_c_of_a_chain_is_no_deeper_than_of_two_ifs(n, chain):
    def depth(links):
        return _bracket_depth(cg._expr_to_c(f"k*({chain(links)})", ["k"], [], [], []))

    assert depth(n) == depth(2) < CLANG_BRACKET_LIMIT


# ── The C of a Piecewise ─────────────────────────────────────────────────────

t, a, b, d = sp.symbols("t a b d")


def _nested_piecewise(n: int):
    """A table of *n* days as the parse builds it: each Piecewise the default
    of the one before."""
    out = sp.Integer(0)
    for i in reversed(range(n)):
        out = sp.Piecewise((sp.Float(_day_value(i)), t <= i + 1), (out, True))
    return out


@pytest.mark.parametrize(
    "pw, expect",
    [
        (sp.Piecewise((a, t <= 1), (d, True)), "(((t <= 1.0)) ? (a) : (d))"),
        (
            sp.Piecewise((a, t <= 1), (b, t <= 2), (d, True)),
            "(((t <= 1.0)) ? (a) : ((t <= 2.0)) ? (b) : (d))",
        ),
        # A Piecewise that is all of the default joins the chain.
        (
            sp.Piecewise((a, t <= 1), (sp.Piecewise((b, t <= 2), (d, True)), True)),
            "(((t <= 1.0)) ? (a) : ((t <= 2.0)) ? (b) : (d))",
        ),
        # No default: 0.0, as before.
        (
            sp.Piecewise((a, t <= 1), (b, t <= 2)),
            "(((t <= 1.0)) ? (a) : ((t <= 2.0)) ? (b) : (0.0))",
        ),
    ],
)
def test_a_piecewise_prints_to_c_as_one_chain(pw, expect):
    assert J.sympy_to_c(pw, str) == expect


@pytest.mark.parametrize("n", [98, 128])
def test_the_c_of_a_nested_piecewise_is_no_deeper_than_of_two(n):
    def depth(pieces):
        return _bracket_depth(J.sympy_to_c(_nested_piecewise(pieces), str))

    assert depth(n) == depth(2) < CLANG_BRACKET_LIMIT


# ── The parse ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expr, expect",
    [
        (
            "if(a,b,if(c, d, if (x,y,z)))",
            "Piecewise((b, a), (Piecewise((d, c), (Piecewise((y, x), (z, True)), True)), True))",
        ),
        # Its brackets too.
        (
            "if(a,b,( if(c,d,(if(x,y,z))) ))",
            "Piecewise((b, a), (( Piecewise((d, c), "
            "((Piecewise((y, x), (z, True))), True)) ), True))",
        ),
    ],
)
def test_the_parse_text_of_a_chain_nests_as_it_always_has(expr, expect):
    assert cg._translate_bngl_if_to_piecewise(expr) == expect


def _window(n: int, value) -> str:
    return (
        "".join(f"if((time()>={i})&&(time()<{i + 1}),{value(i)}," for i in range(n))
        + "0"
        + ")" * n
    )


_PARSED_CHAINS = [
    _chain(20),
    _window(14, lambda i: f"k{i}"),
    _window(16, lambda i: "k"),
    "if(Aobs/Bobs<1,k1,if(Aobs/Bobs>=1,k2,k3))",
    "if(time()<1,0,if(Aobs>thr,kb*(Aobs-thr),0))",
    "if(mode>0.5,0,if(Aobs>thr,kb*(Aobs-thr),0))",
    "if(x<1,a,if(x<2,a,if(x<3,b,c)))",
]


@pytest.mark.parametrize("text", _PARSED_CHAINS)
def test_the_parse_of_a_chain_is_the_right_nested_piecewise(text):
    """What sympy is given is the nested Piecewise, link by link: a flat one
    merges links of one value and drops links it takes to be unreachable."""

    def nested(expr: str):
        # Read without the code under test: the link is if(cond, value, rest).
        if not expr.startswith("if(") or cg._find_close_paren_strict(expr, 2) != len(expr) - 1:
            return J._exprtk_to_sympy(expr)
        cond, value, rest = (arg.strip() for arg in cg._split_top_level_commas(expr[3:-1]))
        return J._piecewise_of_if(sp)(
            (J._exprtk_to_sympy(value), J._exprtk_to_sympy(cond)), (nested(rest), True)
        )

    parsed = J._exprtk_to_sympy(text)
    assert parsed is not None
    assert sp.srepr(parsed) == sp.srepr(nested(text))


def test_a_chain_of_1000_links_is_walked_without_a_frame_for_each():
    def chain(n):
        return "".join(f"if(time()<{i + 1},{i % 7}," for i in range(n)) + "0" + ")" * n

    def bracketed(n):
        return "".join(f"if(time()<{i + 1},{i % 7},(" for i in range(n)) + "0" + "))" * n

    def depth(c):
        return _bracket_depth(cg._expr_to_c(c, [], [], [], []))

    for written in (chain, bracketed):
        text = written(1000)
        assert list(_iter_if_conditions(text)) == [f"time()<{i + 1}" for i in range(1000)]
        assert cg._translate_bngl_if_to_piecewise(text).count("Piecewise(") == 1000
        assert depth(text) == depth(written(2))


# ── Models ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [98, 340])
def test_the_generated_source_of_a_chain_model_is_no_deeper_than_of_two_links(tmp_path, n):
    """The compiler-independent half. At 98 links the analytic sensitivity RHS
    is built, so its derivative is in the source too; at 99 or more the parse
    of the rate law passes Python's limit of 200 nested parentheses and the run
    takes CVODES' difference quotient, as before."""
    src, has_sens = cg.generate_combined_from_model(_model(tmp_path, n), emit_sens_rhs=True)
    assert has_sens == (n < 99)
    two, _ = cg.generate_combined_from_model(_model(tmp_path, 2), emit_sens_rhs=True)
    assert _bracket_depth(src) == _bracket_depth(two) < CLANG_BRACKET_LIMIT


@needs_cc
def test_a_compiled_chain_takes_the_branch_of_every_link(tmp_path, monkeypatch):
    # Through all 340 days and past the end of the table. Day i decays A by
    # exp(-k·v_i), so a link that took its neighbour's value would be off by
    # at least 5e-4 from that day on.
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")
    n = 340
    sim = bngsim.Simulator(_model(tmp_path, n), method="ode", codegen=True)
    assert sim.codegen_backend != "exprtk"
    result = sim.run(t_span=(0, n + 5), n_points=n + 6, rtol=1e-10, atol=1e-12)
    decay = np.concatenate([[0.0], np.cumsum([_day_value(i) for i in range(n)] + [0.0] * 5)])
    assert np.asarray(result.species)[:, 0] == pytest.approx(
        100.0 * np.exp(-0.5 * decay), rel=1e-6
    )


def _sensitivity(model):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    result = sim.run(t_span=(0, 5), n_points=6, rtol=1e-8, atol=1e-10)
    return float(result.species[-1][0]), float(result.sensitivities[-1, 0, 0]), sim


@needs_cc
def test_forward_sensitivity_of_a_128_link_chain_is_right(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")
    a5, s5, _sim = _sensitivity(_model(tmp_path, 128))
    assert a5 == pytest.approx(A5, rel=1e-7)
    assert s5 == pytest.approx(S5, rel=1e-6)


@pytest.fixture
def clang(monkeypatch, tmp_path):
    """clang compiling into a cold cache, so the library is clang's and not one
    gcc left there, and the serial cc build whatever the run sets: the nightly
    forced-JIT run's BNGSIM_CODEGEN_JIT=mir would take every model to MIR and
    never call clang."""
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")
    for var in (
        "BNGSIM_CODEGEN_JIT",
        "BNGSIM_NO_CODEGEN",
        "BNGSIM_CODEGEN_THRESHOLD",
        "BNGSIM_CODEGEN_STRICT",
        "BNGSIM_CODEGEN_JOBS",
        "BNGSIM_CODEGEN_CHUNK",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CC", _CLANG)


@needs_clang
@pytest.mark.parametrize("n", [98, 128])
def test_forward_sensitivity_of_a_long_chain_builds_under_clang(tmp_path, clang, n):
    # At 98 links the analytic sensitivity RHS is compiled with the rate law.
    a5, s5, sim = _sensitivity(_model(tmp_path, n))
    assert sim.codegen_backend == "cc"
    assert a5 == pytest.approx(A5, rel=1e-7)
    assert s5 == pytest.approx(S5, rel=1e-6)


@needs_clang
def test_a_plain_run_of_a_128_link_chain_compiles_under_clang(tmp_path, clang, monkeypatch):
    # It fell back to the interpreter before; strict, it raised.
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    sim = bngsim.Simulator(_model(tmp_path, 128), method="ode", codegen=True)
    assert sim.codegen_backend == "cc"
    result = sim.run(t_span=(0, 5), n_points=6, rtol=1e-8, atol=1e-10)
    assert float(result.species[-1][0]) == pytest.approx(A5, rel=1e-7)


# ── A source nested deep some other way ──────────────────────────────────────


def _true_chain(n: int) -> str:
    """*n* if()s, each in the true branch of the one before: v_i from 0.01·i
    until 0.01·(i+1), and 0 after."""
    text = "0"
    for i in reversed(range(n)):
        text = f"if((time()>={0.01 * (i + 1):g}),{text},{_day_value(i)})"
    return text


def _mixed_chain(n: int) -> str:
    """The same function as :func:`_true_chain`, its links nested in the true
    and the false branch by turns."""
    text = "0"
    for i in reversed(range(n)):
        cut = f"{0.01 * (i + 1):g}"
        if i % 2:
            text = f"if((time()<{cut}),{_day_value(i)},{text})"
        else:
            text = f"if((time()>={cut}),{text},{_day_value(i)})"
    return text


def _deep_net(v_rate: str) -> str:
    return _net(0).replace("1 v_rate() 0", f"1 v_rate() {v_rate}")


def _deep_expected(n: int) -> tuple[float, float]:
    """A(5) and dA(5)/dk for a table of *n* hundredths of a time unit."""
    integral = sum(0.01 * _day_value(i) for i in range(n))
    a5 = 100.0 * math.exp(-0.5 * integral)
    return a5, -integral * a5


def test_the_bracket_depth_is_counted_per_kind_outside_comments_and_literals():
    src = "double f(void){ /* ((((( */ const char *s = \"((((\"; return ((a[(b)])) + '('; }"
    assert cg._c_bracket_depth(src) == 3


def test_only_clang_is_given_a_deeper_limit_and_only_for_a_source_past_it(monkeypatch):
    shallow = f"double r(double t) {{ return {cg._expr_to_c(_chain(340), [], [], [], [])}; }}"
    deep = f"double r(double t) {{ return {cg._expr_to_c(_true_chain(130), [], [], [], [])}; }}"
    depth = cg._c_bracket_depth(deep)
    assert cg._c_bracket_depth(shallow) <= cg._CLANG_BRACKET_DEPTH < depth

    def not_asked(compiler):
        raise AssertionError("the compiler was asked what it is")

    monkeypatch.setattr(cg._toolchain, "is_clang", not_asked)
    assert cg._bracket_depth_flags(shallow, ["cc"]) == []
    assert cg._bracket_depth_flags(deep, ["cl"]) == []
    monkeypatch.setattr(cg._toolchain, "is_clang", lambda compiler: False)
    assert cg._bracket_depth_flags(deep, ["cc"]) == []
    monkeypatch.setattr(cg._toolchain, "is_clang", lambda compiler: True)
    assert cg._bracket_depth_flags(deep, ["cc"]) == [f"-fbracket-depth={depth + 64}"]
    # Within the margin of the limit as well, where the preprocessor's brackets
    # (NAN's) can take a source over it.
    for nested, flagged in ((192, False), (193, True), (256, True)):
        src = "double f(void){ return " + "(" * nested + "1" + ")" * nested + "; }"
        assert cg._c_bracket_depth(src) == nested
        assert cg._bracket_depth_flags(src, ["cc"]) == (
            [f"-fbracket-depth={nested + 64}"] if flagged else []
        )


@needs_clang
@pytest.mark.parametrize("nested", [255, 256])
def test_nan_at_the_deepest_point_of_a_source_just_under_the_limit_builds(tmp_path, clang, nested):
    # glibc's NAN is (__builtin_nanf ("")): two brackets deeper than the source
    # shows, which takes 255 to 257.
    body = "(" * nested + "NAN" + ")" * nested
    src = f"#include <math.h>\ndouble f(void) {{ return {body}; }}\n"
    assert cg._c_bracket_depth(src) == nested
    assert cg.compile_rhs(src, f"nan_{nested}").exists()


def test_a_compiler_that_cannot_say_is_not_taken_for_clang(monkeypatch, tmp_path):
    monkeypatch.setattr(cg._toolchain, "_clang", {})
    assert cg._toolchain.is_clang([str(tmp_path / "no-such-cc")]) is False


def test_a_compiler_that_did_not_answer_is_asked_again(monkeypatch):
    # A first `cc -dM -E` past its timeout on a cold file system said nothing;
    # kept, every deep build in the process went without the flag.
    monkeypatch.setattr(cg._toolchain, "_clang", {})
    answers = iter(["", "#define __clang__ 1\n", "#define __GNUC__ 13\n"])
    asked = []

    def run(cmd, **kwargs):
        asked.append(cmd[0])
        return next(answers)

    monkeypatch.setattr(cg._toolchain, "_run", run)
    assert cg._toolchain.is_clang(["cc"]) is False
    assert cg._toolchain.is_clang(["cc"]) is True
    assert cg._toolchain.is_clang(["cc"]) is True  # kept, not asked
    assert cg._toolchain.is_clang(["gcc"]) is False
    assert cg._toolchain.is_clang(["gcc"]) is False  # kept, not asked
    assert asked == ["cc", "cc", "gcc"]


@needs_clang
def test_clang_is_told_from_its_macros(monkeypatch):
    monkeypatch.setattr(cg._toolchain, "_clang", {})
    assert cg._toolchain.is_clang([_CLANG]) is True


@needs_clang
@pytest.mark.parametrize("v_rate, n", [(_true_chain, 130), (_mixed_chain, 260)])
def test_a_source_nested_past_clangs_limit_builds_under_clang(
    tmp_path, clang, monkeypatch, v_rate, n
):
    # Nested in the true branch, an if() is two brackets deeper than the last,
    # flat C or not: past 256 at 130 links. The sensitivity run declined the
    # analytic RHS for the parse's own limit, then failed to build the RHS.
    model = _load(tmp_path, _deep_net(v_rate(n)), f"deep_{n}")
    src, _has_sens = cg.generate_combined_from_model(model, emit_sens_rhs=True)
    assert cg._c_bracket_depth(src) > cg._CLANG_BRACKET_DEPTH
    a5_exact, s5_exact = _deep_expected(n)
    a5, s5, sim = _sensitivity(model)
    assert sim.codegen_backend == "cc"
    assert a5 == pytest.approx(a5_exact, rel=1e-7)
    # CVODES' difference quotient, across a step of the rate every 0.01.
    assert s5 == pytest.approx(s5_exact, rel=1e-4)
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg_plain")
    runs = {}
    for codegen in (True, False):
        sim = bngsim.Simulator(
            _load(tmp_path, _deep_net(v_rate(n)), f"deep_{n}"), method="ode", codegen=codegen
        )
        assert sim.codegen_backend == ("cc" if codegen else "exprtk")
        result = sim.run(t_span=(0, 5), n_points=6, rtol=1e-8, atol=1e-10)
        runs[codegen] = float(result.species[-1][0])
    # The compiled run is the interpreted one; both are a few 1e-6 off across
    # the jumps, which the sensitivity run stops at and the plain one does not.
    assert runs[True] == pytest.approx(runs[False], rel=1e-12)
    assert runs[True] == pytest.approx(a5_exact, rel=1e-5)


@needs_clang
def test_a_failed_build_is_called_the_compiled_right_hand_side(tmp_path, clang, monkeypatch):
    # erf() has no analytic derivative here, so the build is of the right-hand
    # side alone. A limit of 24 is below the 40 brackets of its rate law, and
    # above what the system headers nest (20 with _FORTIFY_SOURCE in glibc's
    # <stdlib.h>), which the toolchain probe built after a failure includes: a
    # probe that failed too would make the failure the toolchain's.
    monkeypatch.setattr(cg._toolchain, "_verdicts", {})
    monkeypatch.setenv("CC", f"{shlex.quote(_CLANG)} -fbracket-depth=24")
    law = "k*erf(Atot)*" + "(" * 40 + "Atot" + ")" * 40
    model = _load(tmp_path, _deep_net("1").replace("k*v_rate()", law))
    with pytest.raises(bngsim.CodegenBuildError) as err:
        _sensitivity(model)
    message = str(err.value)
    assert message.startswith("Failed to build the compiled right-hand side forward sensitivity")
    assert "analytical sensitivity RHS" not in message


# ── The JAX right-hand side ──────────────────────────────────────────────────

try:  # JAX is an optional extra
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    JAX_AVAILABLE = False


@pytest.fixture
def jnp(monkeypatch):
    """jax.numpy, or numpy standing in for it where JAX is not installed: the
    translated text and its helpers ask it for where and logical_and, which
    mean the same in both."""
    if JAX_AVAILABLE:
        jr.jax_available()  # 64-bit, as the RHS runs
        import jax.numpy

        return jax.numpy
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(numpy=np))
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    return np


_JAX_OBS = ("a", "b")


def _jax_text(expr: str) -> str:
    return jr._translate_expr_jax(
        expr, {"k": 0}, {n: i for i, n in enumerate(_JAX_OBS)}, set(), []
    )


def _jax_value(text: str, jnp, t: float, a: float = 0.0, b: float = 0.0) -> float:
    names = {**jr._JAX_HELPERS, "jnp": jnp, "t": t, "params": np.array([2.0])}
    names.update(obs=np.array([a, b]), y=np.zeros(1))
    return float(eval(compile(text, "<jax>", "eval"), {"__builtins__": {}}, names))  # noqa: S307


def test_the_jax_translation_of_a_340_link_chain_compiles_and_takes_every_link(jnp):
    # Nested, each link a jnp.where inside the last: "too many nested
    # parentheses" from Python's compiler past 199.
    text = _jax_text(f"k*({_chain(340)})")
    for i in range(340):
        assert _jax_value(text, jnp, i + 0.5) == 2.0 * _day_value(i), i
    assert _jax_value(text, jnp, 340.5) == 0.0


@pytest.mark.parametrize(
    "expr",
    [
        "if(a>1,k,if(b>1,2*k,0))",
        "if(a>1 && b<2,k,(if(b>1,a>2,0)))",
        "if(a<1,k,if(a<2,a>b,if(b<1,2,-1)))+if(b>0,1,if(a>0,2,3))",
        "if(if(a>1,1,if(b>1,2,3))>1,k,if(a,b,a+b))",
    ],
)
def test_a_jax_chain_is_the_nested_where_calls_it_was(expr, jnp, monkeypatch):
    flat = _jax_text(expr)
    assert "__bngsim_where_chain__" in flat
    monkeypatch.setattr(jr, "_flatten_if_chains", lambda c: c)
    nested = _jax_text(expr)
    for a in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        for b in (-1.0, 0.0, 0.5, 1.0, 2.5):
            assert _jax_value(flat, jnp, 0.0, a, b) == _jax_value(nested, jnp, 0.0, a, b), (a, b)


def test_a_jax_chains_conditions_are_truth_values_and_its_values_numbers():
    # A comparison read as a number is a double (issue #1495), a condition not.
    assert _jax_text("if(a>1 && b<2,k,(if(b>1,a>2,0)))") == (
        "__bngsim_where_chain__(jnp.logical_and(obs[0] > 1.0, obs[1] < 2.0), params[0], "
        "obs[1] > 1.0, jnp.where(obs[0] > 2.0, 1.0, 0.0), 0.0)"
    )
    assert _jax_text("if(a>1,k,0)") == "jnp.where(obs[0]>1.0,params[0],0.0)"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_the_jax_jacobian_runs_a_340_link_chain_as_the_default_does(tmp_path):
    def run(n, **kwargs):
        sim = bngsim.Simulator(_model(tmp_path, n), method="ode", **kwargs)
        return np.asarray(sim.run(t_span=(0, 5), n_points=6, rtol=1e-8, atol=1e-10).species)[:, 0]

    with_jax = run(340, jacobian="jax")
    assert with_jax == pytest.approx(run(340), rel=1e-6)
    assert with_jax[-1] == pytest.approx(A5, rel=1e-7)
    assert np.array_equal(with_jax, run(128, jacobian="jax"))


# ── Chains a flat Piecewise reads differently ────────────────────────────────


def _window_net(chain: str, params: dict[str, float]) -> str:
    lines = "\n".join(
        f"    {i + 1} {name} {value}" for i, (name, value) in enumerate(params.items())
    )
    return f"""begin parameters
{lines}
end parameters
begin functions
    1 r() {chain}
end functions
begin species
    1 X() 0
end species
begin reactions
    1 0 1 r
end reactions
begin groups
    1 Xobs 1
end groups
"""


def _window_sensitivity(tmp_path, chain, params, asked, t_end):
    text = _window_net(chain, params)
    _src, has_sens = cg.generate_combined_from_model(_load(tmp_path, text, "windows"))
    sim = bngsim.Simulator(
        _load(tmp_path, text, "windows"), method="ode", sensitivity_params=asked
    )
    result = sim.run(t_span=(0, t_end), n_points=3, rtol=1e-8, atol=1e-10)
    return has_sens, np.asarray(result.sensitivities)[-1, 0]


@needs_cc
def test_a_chain_of_windows_builds_its_analytic_sensitivity_rhs(tmp_path):
    # Fourteen windows [i, i+1), each its own rate: X gains k_i over window i,
    # so dX/dk_i is 1 once it has passed. The derivative of each column is 0 on
    # thirteen windows, which a flat Piecewise merges into one Or of windows in
    # normal form, doubling with each: the build ran past its budget and the
    # analytic RHS was declined.
    chain = _window(14, lambda i: f"k{i}")
    parsed = J._exprtk_to_sympy(chain)
    assert sp.count_ops(sp.diff(parsed, sp.Symbol("k0"))) <= 2 * 14
    params = {f"k{i}": 1.0 + 0.1 * i for i in range(14)}
    has_sens, s = _window_sensitivity(tmp_path, chain, params, ["k0", "k1", "k2"], 15)
    assert has_sens
    assert s == pytest.approx([1.0, 1.0, 1.0], rel=1e-6)


@needs_cc
def test_a_chain_of_windows_of_one_value_parses_small_and_runs(tmp_path):
    chain = _window(16, lambda i: "k")
    parsed = J._exprtk_to_sympy(chain)
    assert sp.count_ops(parsed) <= 2 * 16  # merged into one Or, 786k
    has_sens, s = _window_sensitivity(tmp_path, chain, {"k": 0.5}, ["k"], 20)
    assert has_sens
    assert s == pytest.approx([16.0], rel=1e-6)


NAN_NET = """begin parameters
    1 k1 1
    2 k2 2
    3 k3 3
end parameters
begin functions
    1 r() if(Aobs/Bobs<1,k1,if(Aobs/Bobs>=1,k2,k3))
end functions
begin species
    1 A() 0
    2 B() 0
    3 C() 0
end species
begin reactions
    1 0 3 r
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
    3 Cobs 3
end groups
"""


@needs_cc
def test_a_nan_condition_takes_the_last_link_in_the_sensitivity_too(tmp_path):
    # Aobs/Bobs is 0/0, NaN, so neither comparison holds and C is made at k3.
    # A flat Piecewise took the second condition for the complement of the
    # first and dropped k3: dC/dk was [0, 2, 0] beside C = 6.
    sim = bngsim.Simulator(
        _load(tmp_path, NAN_NET), method="ode", sensitivity_params=["k1", "k2", "k3"]
    )
    result = sim.run(t_span=(0, 2), n_points=3, rtol=1e-8, atol=1e-10)
    assert float(result.species[-1][2]) == pytest.approx(6.0, rel=1e-9)
    assert np.asarray(result.sensitivities)[-1, 2] == pytest.approx([0.0, 0.0, 2.0], abs=1e-9)


def _ramp_net(gate: str) -> str:
    # A made at rate k crosses thr = 1 at t = 1/k; after the gate opens B is made
    # at kb·(A - thr). The erf() law has no analytic derivative here, so the
    # whole model takes CVODES' difference quotient.
    return f"""begin parameters
    1 k 1
    2 kb 0.5
    3 thr 1
    4 mode 0
end parameters
begin functions
    1 r() if({gate},0,if(Aobs>thr,kb*(Aobs-thr),0))
    2 s() 0.1*erf(Cobs)
end functions
begin species
    1 A() 0
    2 B() 0
    3 C() 1
end species
begin reactions
    1 0 1 k
    2 0 2 r
    3 0 3 s
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
    3 Cobs 3
end groups
"""


@needs_cc
@pytest.mark.parametrize("gate", ["time()<1", "mode>0.5"])
@pytest.mark.parametrize("functional_sens_rhs", [True, False])
def test_a_gated_ramp_is_proved_a_bend_for_the_difference_quotient(
    tmp_path, monkeypatch, gate, functional_sens_rhs
):
    # The ramp bends at A = thr and does not jump, which the quotient is right
    # across; it is proved for a two-way if() only, so a flat Piecewise of three
    # links was refused. B(5) = ∫_1^5 kb·(k·t - 1) dt = 6k - 2.
    if not functional_sens_rhs:
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
    sim = bngsim.Simulator(
        _load(tmp_path, _ramp_net(gate)), method="ode", sensitivity_params=["k"]
    )
    result = sim.run(t_span=(0, 5), n_points=3, rtol=1e-8, atol=1e-10)
    assert float(result.species[-1][1]) == pytest.approx(4.0, rel=1e-7)
    assert float(result.sensitivities[-1, 1, 0]) == pytest.approx(6.0, rel=1e-5)
