"""A derived initial condition's sensitivity seed reads rint, mod and frac as the
engine does (issue #1800).

The seed's partial of ``X0 = rint(p)*k`` with respect to k is ``rint(p)``, and it
was evaluated with numpy's functions: ``rint`` rounds a half to even, ``mod`` is
floored and sympy's ``frac`` is ``x - floor(x)``. The engine, which computes the
start every run uses, reads ``rint`` as BNG's ``floor(x + 0.5)``, ``mod`` as
``std::fmod`` and ``frac`` as ``x - trunc(x)``. Where they disagree the column
was wrong from t = 0 on, with no error: ``rint(2.5)*k`` seeded dX/dk = 2 for 3.

The oracle is the engine itself. X decays at kd, so dX(t)/dk = X(0)/k * exp(-kd t),
and X(0) is what a plain run of the same model starts from.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _derived_expr_value_numeric

KD = 0.5
TIMES = [0.0, 1.0, 2.0]


def _model(tmp_path, ic: str, p: float, k: float = 1.0) -> bngsim.Model:
    net = tmp_path / "seed.net"
    net.write_text(
        f"""begin parameters
    1 p {p!r}
    2 k {k!r}
    3 kd {KD}
    4 X0 {ic}
end parameters
begin species
    1 X() X0
end species
begin reactions
    1 1 0 kd #_R1
end reactions
""",
        encoding="utf-8",
    )
    return bngsim.Model.from_net(str(net))


# (initial condition, p, the engine's value of the factor beside k, numpy's)
CASES = [
    ("rint(p)*k", 2.5, 3.0, 2.0),  # a half rounds up, not to even
    ("rint(p)*k", 0.5, 1.0, 0.0),
    ("rint(p)*k", -1.5, -1.0, -2.0),
    ("mod(p,2)*k", -0.5, -0.5, 1.5),  # the sign of the first argument
    ("mod(p,-2)*k", 3.0, 1.0, -1.0),
    ("frac(p)*k", -0.5, -0.5, 0.5),  # x - trunc(x), not x - floor(x)
    ("frac(p)*k", -1.25, -0.25, 0.75),
    ("(p % 2)*k", -0.5, -0.5, 1.5),  # ExprTk's % is fmod too, not sympy's floored Mod
    ("frac(-2.5)*k", 0.0, -0.5, 0.5),  # a constant argument: sympy folded it at the parse
]


@pytest.mark.parametrize("ic,p,engine,other", CASES)
def test_the_seed_reads_the_start_the_engine_computes(tmp_path, ic, p, engine, other):
    assert engine != other  # each case sits where the two readings part
    model = _model(tmp_path, ic, p)
    start = bngsim.Simulator(model, method="ode").run(sample_times=TIMES).species[0, 0]
    assert start == engine  # k = 1: the plain run starts from the engine's value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bngsim.Simulator(
            _model(tmp_path, ic, p), method="ode", sensitivity_params=["k"]
        ).run(sample_times=TIMES)
    column = result.sensitivities[:, 0, 0]
    expected = [engine * math.exp(-KD * t) for t in TIMES]
    np.testing.assert_allclose(column, expected, rtol=1e-6, atol=1e-9)


def test_a_partial_without_these_calls_is_unchanged(tmp_path):
    # The control: no rint, mod or frac, so nothing is rewritten.
    model = _model(tmp_path, "2.5*p*k", 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
            sample_times=TIMES
        )
    np.testing.assert_allclose(
        result.sensitivities[:, 0, 0], [5.0 * math.exp(-KD * t) for t in TIMES], rtol=1e-6
    )


# A derived parameter's value, as a crossing's threshold reads it: sympy's evalf,
# not numpy, so the same three are written out in sympy's own functions there.
# On main frac read x - floor(x) (1.5 for 0.5 below), and rint and mod did not
# evaluate at all (None, which refuses the crossing).
@pytest.mark.parametrize(
    "expr,p,q,value",
    [
        ("frac(p)+1", -0.5, 0.0, 0.5),
        ("frac(p)*2", -1.25, 0.0, -0.5),
        ("rint(p)+1", -2.5, 0.0, -1.0),
        ("rint(p)", 2.5, 0.0, 3.0),
        ("mod(p,q)+5", -3.0, 2.0, 4.0),
        ("mod(p,q)", 3.0, -2.0, 1.0),
        # fmod is exact where a - b*trunc(a/b) is not: 1.0/0.1 rounds to 10.
        ("mod(p,q)", 1.0, 0.1, math.fmod(1.0, 0.1)),
        ("p % q", -0.5, 2.0, -0.5),
        ("frac(-2.5) + p", 0.0, 0.0, -0.5),
    ],
)
def test_a_derived_value_reads_them_as_the_engine_does(expr, p, q, value):
    got = _derived_expr_value_numeric(expr, {"p", "q"}, set(), {"p": 0, "q": 1}, [p, q])
    assert got == value


# The other two numeric routes. The switch analysis's double-exact lambdify
# wrote sympy's frac as x % 1 (0.5 for -0.5); it now writes x - trunc(x). The
# kink check reads all three as the engine does.
@pytest.mark.parametrize(
    "make,args,value",
    [
        (lambda sp, x, y: sp.frac(x), (-0.5, 0.0), -0.5),
        (lambda sp, x, y: sp.frac(x), (-1.25, 0.0), -0.25),
        (lambda sp, x, y: sp.frac(x), (2.75, 0.0), 0.75),
        (lambda sp, x, y: x / 24 + 1, (1.0, 0.0), 1.0 / 24 + 1),  # unchanged
    ],
)
def test_the_switch_analysis_reads_them_as_the_engine_does(make, args, value):
    import sympy as sp
    from bngsim._parameter_kinks import _values
    from bngsim._switch_sensitivity import _lambdify_doubles

    x, y = sp.symbols("x y")
    assert _lambdify_doubles([x, y], make(sp, x, y))(*args) == value
    assert _values([make(sp, x, y)], dict(zip((x, y), args, strict=True)), sp) == [value]


def test_the_kink_check_reads_rint_as_the_engine_does():
    import sympy as sp
    from bngsim._parameter_kinks import _values

    x = sp.Symbol("x")
    assert _values([sp.Function("rint")(x)], {x: 2.5}, sp) == [3.0]
    assert _values([sp.Function("rint")(x)], {x: -1.5}, sp) == [-1.0]


def test_the_switch_analysis_still_refuses_what_it_cannot_read():
    # rint, mod and round stay calls this route cannot print, as on main: it
    # reads only what it can read exactly as the engine does.
    import sympy as sp
    from bngsim._switch_sensitivity import _lambdify_doubles

    x, y = sp.symbols("x y")
    for expr in (sp.Function("rint")(x), sp.Function("mod")(x, y), sp.Function("round")(x)):
        with pytest.raises(Exception, match="(?i)unsupported|not.*print"):
            _lambdify_doubles([x, y], expr)


def test_the_kink_check_reads_mod_as_fmod():
    import sympy as sp
    from bngsim._parameter_kinks import _values

    x, y = sp.symbols("x y")
    assert _values([sp.Function("mod")(x, y)], {x: -3.0, y: 2.0}, sp) == [-1.0]


def test_the_switch_analysis_reads_a_non_finite_frac_as_nan_not_a_raise():
    # x % 1 gave nan here; a truncation through int() would raise instead.
    import sympy as sp
    from bngsim._switch_sensitivity import _lambdify_doubles

    x = sp.Symbol("x")
    f = _lambdify_doubles([x], sp.frac(x))
    assert all(math.isnan(f(v)) for v in (math.inf, -math.inf, math.nan))


# From the review of 9cd18bd3.


@pytest.mark.parametrize("name", ["frac", "mod", "rint"])
def test_a_parameter_named_like_one_of_them_stays_a_parameter(name):
    # The engine reserves none of the three as a parameter name: `frac*k` is a
    # product of two parameters, d/dfrac = k and d/dk = frac.
    from bngsim._codegen import _direct_derived_partials_numeric

    partials, _ = _direct_derived_partials_numeric(
        f"{name}*k", {name, "k"}, set(), {name: 0, "k": 1}, [2.0, 0.3], False
    )
    assert partials == pytest.approx({name: 0.3, "k": 2.0})


@pytest.mark.parametrize(
    "expr,p,value",
    [
        ("-2.5 % 2 + p", 0.0, -0.5),  # sympy folded the literals as a floored Mod: 1.5
        ("(p + 2) % 2", -0.5, 1.5),  # sympy simplified it to Mod(p, 2): -0.5
        ("p*3 % 4 - 1", 2.0, 1.0),  # % binds like * and /, tighter than -
    ],
)
def test_percent_is_spelled_as_fmod_before_the_parse(expr, p, value):
    assert _derived_expr_value_numeric(expr, {"p"}, set(), {"p": 0}, [p]) == value


def test_a_long_literal_is_read_as_the_engine_reads_it():
    # strtod reads 8192.499999999999 as 8192.499999999998, so rint gives 8192;
    # float() of sympy's 56-bit Float rounds twice, onto the tie, and gave 8193.
    got = _derived_expr_value_numeric("rint(8192.499999999999) + p", {"p"}, set(), {"p": 0}, [0.0])
    assert got == 8192.0


def test_rint_of_an_infinity_is_that_infinity():
    # floor(inf + 0.5) is inf in the engine; only frac of an infinity is nan.
    assert _derived_expr_value_numeric("rint(p)", {"p"}, set(), {"p": 0}, [math.inf]) == math.inf
    assert _derived_expr_value_numeric("rint(p)", {"p"}, set(), {"p": 0}, [-math.inf]) == -math.inf


def test_the_switch_printer_prints_a_nested_frac_once_per_level():
    import sympy as sp
    from bngsim._switch_sensitivity import _double_printer

    x = sp.Symbol("x")
    text = _double_printer()().doprint(sp.frac(sp.frac(sp.frac(x))))
    assert text.count("x") == 1  # was 27: each level printed its argument three times


def test_the_kink_check_keeps_mains_reading_of_a_percent_the_rate_law_parse_simplified():
    # The rate-law parse reads % as sympy's Mod and simplifies (p + 2) % 2 to
    # Mod(p, 2) before the kink check sees it. Rewritten to fmod after that, it
    # read fmod(-0.5, 2) = -0.5 where the engine has fmod(1.5, 2) = 1.5, and main
    # 1.5: a regression. The kink check leaves sympy's Mod as main does (#1815).
    import sympy as sp
    from bngsim._jacobian import _exprtk_to_sympy
    from bngsim._parameter_kinks import _values

    p = sp.Symbol("p")
    assert _values([_exprtk_to_sympy("(p + 2) % 2", bends=True)], {p: -0.5}, sp) == [1.5]


def test_percent_beside_a_parameter_named_mod_parses():
    # % is spelled as a private call, so a parameter named mod is not called.
    from bngsim._codegen import _prepare_derived_expr

    prep, why = _prepare_derived_expr("a % b", {"a", "b", "mod"}, [])
    assert prep is not None, why
    assert (
        _derived_expr_value_numeric(
            "a % b", {"a", "b", "mod"}, set(), {"a": 0, "b": 1, "mod": 2}, [-0.5, 2.0, 7.0]
        )
        == -0.5
    )


def test_the_clock_parser_reads_percent_as_main_does():
    # The clock parser shares the derived-expression preprocessing; the spelling
    # of % is the derived parse's alone, so the clock solvers are unchanged.
    import sympy as sp
    from bngsim._switch_sensitivity import _parse_clock_expr

    expr, _ = _parse_clock_expr("_bng_clock_t % 24 - 6")
    assert expr.has(sp.Mod)
    expr, _ = _parse_clock_expr("mod - _bng_clock_t % 3")  # a parameter named mod beside it
    assert expr.has(sp.Mod)


def test_an_observable_named_like_one_of_them_is_refused_as_a_model_symbol():
    # Read bare, frac / mod / rint is a model symbol, which a derived parameter
    # cannot read: the refusal says so, as main's did, not a TypeError.
    from bngsim._codegen import _prepare_derived_expr

    for name in ("frac", "mod", "rint"):
        prep, why = _prepare_derived_expr(f"time() > k*{name}", {"k"}, [], allow_no_reference=True)
        assert prep is None and "unresolved symbol" in why, why


def test_the_compiled_chain_rule_names_no_private_function():
    from bngsim import _codegen

    for expr in ("k*rint(p)", "k*(p % 2)", "k*frac(p)"):
        _, why = _codegen._direct_derived_partials(expr, {"p", "k"}, set(), {"p": 0, "k": 1}, None)
        assert why and "__bngsim" not in why, why
