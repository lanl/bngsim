"""A chain of ``if()``s is written as one chain, however long it is (issue #1814).

A table of values over time is a chain of nested ``if()``s, one link a day:
``if(time()<=1, v1, if(time()<=2, v2, ... 0))``. The Mallela 2024 ``phoenix``
model's ``v_rate()`` is 340 of them. Every emitter wrote each ``if()`` of the
chain inside the one before it, two brackets deeper, so the nesting grew with
the table:

- the C of the rate law (``_replace_if_calls``), and of its derivative (the C
  printer's ``Piecewise``), passed clang's limit of 256 brackets at 128 links.
  A forward-sensitivity run, which needs the compiled right-hand side, failed to
  build with "bracket nesting level exceeded maximum of 256"; a plain run fell
  back to the interpreter. GCC has no such limit, so the C is measured here as
  well as compiled;
- the ``Piecewise`` text the rate law is parsed from passed Python's limit of
  200 parentheses at 99 links, so the analytic sensitivity RHS was declined for
  CVODES' difference quotient.

An ``if()`` that is all of another's false branch is now the next link of one
chain in all three. These tests cover:

- the bracket depth of the C and of the parse text does not grow with the chain;
- the chain reads as the nested form did: the same C but for the brackets, the
  same branch taken at every link, compiled and parsed;
- a forward-sensitivity run of a 128-link model builds its analytic RHS and
  returns dA(5)/dk = -1.488792, under clang where there is one.
"""

from __future__ import annotations

import math
import re
import shutil

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim import _jacobian as J  # noqa: E402

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


def _model(tmp_path, n: int) -> bngsim.Model:
    path = tmp_path / f"chain_{n}.net"
    path.write_text(_net(n))
    return bngsim.Model.from_net(str(path))


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


@pytest.mark.parametrize("n", [128, 340])
def test_the_c_of_a_chain_is_no_deeper_than_of_two_ifs(n):
    def depth(links):
        return _bracket_depth(cg._expr_to_c(f"k*({_chain(links)})", ["k"], [], [], []))

    assert depth(n) == depth(2) < CLANG_BRACKET_LIMIT


# ── The C of a Piecewise ─────────────────────────────────────────────────────

t, a, b, d = sp.symbols("t a b d")


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


@pytest.mark.parametrize("n", [128, 340])
def test_the_c_of_a_piecewise_is_no_deeper_than_of_two_pieces(n):
    def depth(pieces):
        pw = sp.Piecewise(*((_day_value(i), t <= i + 1) for i in range(pieces)), (0, True))
        return _bracket_depth(J.sympy_to_c(pw, str))

    assert depth(n) == depth(2) < CLANG_BRACKET_LIMIT


# ── The parse ────────────────────────────────────────────────────────────────


def test_the_parse_text_of_a_chain_is_no_deeper_than_of_two_ifs():
    def depth(links):
        return _bracket_depth(J._preprocess_exprtk(f"k*({_chain(links)})"))

    # Python's tokenizer refuses 200 nested parentheses.
    assert depth(340) == depth(2) < 200


@pytest.mark.parametrize("n", [20, 340])
def test_a_chain_parses_to_the_branch_the_engine_takes_at_every_link(n):
    # Twenty links parsed before, nested; 340 did not parse at all.
    parsed = J._exprtk_to_sympy(f"k*({_chain(n)})")
    assert parsed is not None
    clock, k = sp.Symbol(J._TIME_SYM), sp.Symbol("k")
    rate = sp.lambdify((clock, k), parsed)
    for i in range(n):
        assert rate(i + 0.5, 2.0) == 2.0 * _day_value(i), i
        assert rate(i + 1.0, 2.0) == 2.0 * _day_value(i), i  # `<=`: day i's last instant
    assert rate(n + 0.5, 2.0) == 0.0


def test_a_chain_parses_to_what_the_nested_piecewise_is():
    n = 20
    clock = sp.Symbol(J._TIME_SYM)
    nested = sp.Integer(0)
    for i in reversed(range(n)):
        nested = sp.Piecewise((sp.Float(_day_value(i)), clock <= i + 1), (nested, True))
    assert sp.piecewise_fold(nested) == J._exprtk_to_sympy(_chain(n))


# ── Models ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [128, 340])
def test_a_chain_model_gets_its_analytic_sensitivity_rhs_in_c_clang_can_read(tmp_path, n):
    """The compiler-independent half: the whole generated source, RHS and
    sensitivity RHS, nests as deep as it does for a chain of two."""
    src, has_sens = cg.generate_combined_from_model(_model(tmp_path, n), emit_sens_rhs=True)
    assert has_sens, "the analytic sensitivity RHS was declined"
    two, _ = cg.generate_combined_from_model(_model(tmp_path, 2), emit_sens_rhs=True)
    assert _bracket_depth(src) == _bracket_depth(two) < CLANG_BRACKET_LIMIT


def test_the_version_leaves_the_libraries_without_a_sensitivity_rhs_behind():
    """An install whose source digest is empty (bytecode only) keys on the
    version alone, so only the version can leave behind a v44 library of a
    chain of 99 or more, which has no analytic sensitivity RHS."""
    assert int(cg._CODEGEN_VERSION) >= 45


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
    return float(result.species[-1][0]), float(result.sensitivities[-1, 0, 0])


@needs_cc
def test_forward_sensitivity_of_a_128_link_chain_is_right(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")
    a5, s5 = _sensitivity(_model(tmp_path, 128))
    assert a5 == pytest.approx(A5, rel=1e-7)
    assert s5 == pytest.approx(S5, rel=1e-6)


@needs_clang
def test_forward_sensitivity_of_a_128_link_chain_builds_under_clang(tmp_path, monkeypatch):
    # A cold cache, so the library is clang's and not one gcc left there.
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")
    monkeypatch.setenv("CC", _CLANG)
    a5, s5 = _sensitivity(_model(tmp_path, 128))
    assert a5 == pytest.approx(A5, rel=1e-7)
    assert s5 == pytest.approx(S5, rel=1e-6)


@needs_clang
def test_a_plain_run_of_a_128_link_chain_compiles_under_clang(tmp_path, monkeypatch):
    # It fell back to the interpreter before; strict, it raised.
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")
    monkeypatch.setenv("CC", _CLANG)
    monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", "1")
    sim = bngsim.Simulator(_model(tmp_path, 128), method="ode", codegen=True)
    assert sim.codegen_backend == "cc"
    result = sim.run(t_span=(0, 5), n_points=6, rtol=1e-8, atol=1e-10)
    assert float(result.species[-1][0]) == pytest.approx(A5, rel=1e-7)
