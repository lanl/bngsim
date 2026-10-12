"""GH #565 — `log10()` and its siblings must translate, or say so up front.

The JAX translator's table stopped at seventeen names. ``log10`` was not among
them, and the ``log`` rule could not stand in: ``\\blog\\b`` does not match
inside ``log10``, because ``1`` is a word character and the boundary fails. So
``log10(...)`` reached the generated source untranslated, and the RHS evaluates
each body with ``{"__builtins__": {}}`` and a namespace of
``jnp``/``t``/``params``/``obs``/``y`` — no ``log10`` anywhere in it. The model
died mid-solve with ``NameError: name 'log10' is not defined``, on a model the
ODE backend and ``jacobian="auto"`` both run.

Two halves here. Every reserved engine name that jax.numpy spells directly is
now mapped — the logarithms, the hyperbolics and their inverses, the roundings,
``sign``/``sgn``, the constants ``_pi``/``_e``, and the bare ``time``. And a
name that is still not mapped is refused while the expression is in hand,
naming it, rather than reaching a solve-time ``NameError`` that names a symbol
the caller never wrote in Python. ``erf``, ``erfc``, ``tgamma``, ``clamp``,
``avg``, ``sum``, ``mratio`` and the table functions are in that second group:
none of them worked before either.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from bngsim._jax_rhs import _translate_expr_jax

try:  # JAX is an optional extra
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    JAX_AVAILABLE = False

PARAMS = {"k1": 0}
OBS = {"B": 0}


def _t(expr: str) -> str:
    return _translate_expr_jax(expr, PARAMS, OBS, set(), [])


# ── The names that now translate ─────────────────────────────────────────────


def test_the_issue_expression():
    assert _t("k1*log10(1+B)") == "params[0]*jnp.log10(1.0+obs[0])"


@pytest.mark.parametrize(
    "expr, want",
    [
        ("log10(B)", "jnp.log10(obs[0])"),
        ("log2(B)", "jnp.log2(obs[0])"),
        ("sinh(B)", "jnp.sinh(obs[0])"),
        ("cosh(B)", "jnp.cosh(obs[0])"),
        ("tanh(B)", "jnp.tanh(obs[0])"),
        ("asinh(B)", "jnp.arcsinh(obs[0])"),
        ("acosh(B)", "jnp.arccosh(obs[0])"),
        ("atanh(B)", "jnp.arctanh(obs[0])"),
        ("round(B)", "__bngsim_round__(obs[0])"),
        ("trunc(B)", "jnp.trunc(obs[0])"),
        ("sign(B)", "jnp.sign(obs[0])"),
        ("sgn(B)", "jnp.sign(obs[0])"),
    ],
)
def test_each_added_function(expr, want):
    assert _t(expr) == want


@pytest.mark.parametrize(
    "expr, want",
    [
        ("_pi*B", "jnp.pi*obs[0]"),
        ("_e*B", "jnp.e*obs[0]"),
        # the bare clock, which the C path maps the same way
        ("k1*time", "params[0]*t"),
        ("k1*time()", "params[0]*t"),
    ],
)
def test_the_constants_and_the_bare_clock(expr, want):
    assert _t(expr) == want


def test_every_reserved_name_translates_or_is_one_of_the_refused():
    """The refused list is measured against the engine's own, not written from
    memory: every reserved constant translates, and of the reserved functions
    exactly these seven are refused. `time` and `if` are rewritten structurally
    and covered above and in the #564 file."""
    import bngsim

    names = bngsim.reserved_names()
    for const in names["constants"]:
        _t(f"{const}*B")  # raises if refused
    refused = set()
    for fn in set(names["functions"]) - {"time", "if"}:
        try:
            _t(f"{fn}(B)")
        except ValueError:
            refused.add(fn)
    assert refused == {"erf", "erfc", "tgamma", "clamp", "avg", "sum", "mratio"}


@pytest.mark.parametrize(
    "expr, want",
    [
        ("max(B,k1,2)", "jnp.maximum(jnp.maximum(obs[0],params[0]),2.0)"),
        ("min(B,k1,2,3)", "jnp.minimum(jnp.minimum(jnp.minimum(obs[0],params[0]),2.0),3.0)"),
        ("max(B)", "(obs[0])"),
        (
            "max(B, min(k1,B,3))",
            "jnp.maximum(obs[0], jnp.minimum(jnp.minimum(params[0],obs[0]),3.0))",
        ),
        # a binary call is left exactly as written, spaces and all
        ("max(B, k1)", "jnp.maximum(obs[0], params[0])"),
    ],
)
def test_a_variadic_max_or_min_is_folded_to_binary_calls(expr, want):
    """The engine's max/min are ExprTk's, which take any number of arguments;
    jnp.maximum takes two, so `max(a, b, c)` was a TypeError mid-solve."""
    assert _t(expr) == want


def test_the_logarithms_do_not_collide():
    """`log10` is not `log` with a stray `10`, and `ln` is still `jnp.log`."""
    assert _t("ln(B)+log(B)+log10(B)+log2(B)") == (
        "jnp.log(obs[0])+jnp.log(obs[0])+jnp.log10(obs[0])+jnp.log2(obs[0])"
    )


def test_the_hyperbolics_beat_their_shorter_names():
    """`sinh` must not be read as `sin` plus an `h`, nor `asinh` as `asin`."""
    assert _t("sin(B)+sinh(B)+asin(B)+asinh(B)") == (
        "jnp.sin(obs[0])+jnp.sinh(obs[0])+jnp.arcsin(obs[0])+jnp.arcsinh(obs[0])"
    )


# ── The names that still do not, and now say so ──────────────────────────────


@pytest.mark.parametrize("fn", ["erf", "erfc", "tgamma", "clamp", "avg", "sum", "tfun"])
def test_an_unmapped_name_is_refused_with_its_name(fn):
    """It raised NameError deep in the solve before; the message now arrives
    while the expression is still in hand, and names the function."""
    with pytest.raises(ValueError, match=fn):
        _t(f"k1*{fn}(B)")


def test_the_refusal_points_at_the_working_alternative():
    with pytest.raises(ValueError, match="jacobian='auto'"):
        _t("erf(B)")


def test_a_quoted_file_name_is_not_read_as_identifiers():
    """A table function is refused for `tfun`, not for the words inside its
    file name."""
    with pytest.raises(ValueError) as exc:
        _t("tfun('dose_response.tfun', B)")
    # The expression is echoed in full (it is the useful part of the message);
    # what must not grow is the list of names, which is everything before
    # " in the expression".
    names = str(exc.value).split(" in the expression")[0]
    assert "'tfun'" in names
    assert "dose_response" not in names


def test_a_translatable_expression_is_not_refused():
    """The guard must stay quiet on everything that does work — the table, an
    if(), a function reference and the operators."""
    out = _translate_expr_jax(
        "if(B>1 && k1<2, log10(B)+sinh(k1)+other(), _pi*B^2)",
        PARAMS,
        OBS,
        {"other"},
        ["other"],
    )
    assert "jnp.where(" in out and "func_other" in out


# ── End to end ───────────────────────────────────────────────────────────────

NET = """begin parameters
    1 k1      0.5  # Constant
end parameters
begin functions
    1 law() k1*{body}
end functions
begin species
    1 B() 10
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 B                    1
end groups
"""


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("body", ["log10(1+B)", "log2(1+B)", "tanh(B)", "_pi*B"])
def test_the_jax_jacobian_matches_the_default(tmp_path, body):
    """The failure the issue reports: this raised NameError inside the solve
    while the same model ran on the default Jacobian."""
    import bngsim

    net = tmp_path / "m.net"
    net.write_text(NET.format(body=body))

    def _run(**kw):
        with contextlib.redirect_stderr(io.StringIO()):
            model = bngsim.Model.from_net(str(net))
            return bngsim.Simulator(model, method="ode", **kw).run(t_span=(0.0, 5.0), n_points=6)

    default = np.asarray(_run().species)[:, 0]
    with_jax = np.asarray(_run(jacobian="jax").species)[:, 0]
    assert with_jax == pytest.approx(default, rel=1e-6, abs=1e-8)
    assert default[-1] < default[0]


PROBE_NET = """begin parameters
    1 k1      1.0  # Constant
    2 h       0.0  # Constant
end parameters
begin functions
    1 law() k1*{body}
end functions
begin species
    1 B() 1
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 B                    1
end groups
"""


def _jax_and_engine_rhs(tmp_path, body, h):
    """dB/dt from the JAX RHS and from the engine's interpreted one, at B = 1."""
    import bngsim
    import jax.numpy as jnp
    from bngsim._jax_rhs import generate_jax_rhs

    net = tmp_path / "m.net"
    net.write_text(PROBE_NET.format(body=body))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(net))
    model.set_param("h", h)
    engine = model.rhs(np.array([1.0]))[0]
    jax_rhs = generate_jax_rhs(str(net))
    return float(jax_rhs(jnp.array([1.0]), 0.0, jnp.array([1.0, h]))[0]), float(engine)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("fn", ["round", "rint"])
@pytest.mark.parametrize(
    "h",
    # the halves, where jnp.round (half to even) disagreed and where C's round
    # (away from zero below 0) and BNG's rint (floor(x + 0.5), #771) part;
    # then the edges where the addition in floor(x + 0.5) itself rounds, which
    # rint keeps and round must not (#684)
    [
        2.5,
        -2.5,
        0.5,
        -0.5,
        1.5,
        2.4,
        -0.3,
        0.49999999999999994,
        -0.49999999999999994,
        4503599627370497.0,
        -4503599627370497.0,
    ],
)
def test_the_roundings_match_the_engine_exactly(tmp_path, fn, h):
    """`round` and `rint` must mean what the engine means. Mapped onto
    jnp.round, both rounded a half to even — round(2.5) was 2, not 3."""
    got, want = _jax_and_engine_rhs(tmp_path, f"{fn}(h)", h)
    assert got == want


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("body", ["max(h,1,2)", "min(h,1,2)", "max(h)", "min(3,max(h,-1,0),2)"])
@pytest.mark.parametrize("h", [-2.0, 0.5, 1.5, 4.0])
def test_a_variadic_max_or_min_matches_the_engine(tmp_path, body, h):
    got, want = _jax_and_engine_rhs(tmp_path, body, h)
    assert got == want


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("const", ["_pi", "_e", "_kB", "_NA", "_R", "_h", "_F"])
def test_the_constants_match_the_engine_exactly(tmp_path, const):
    got, want = _jax_and_engine_rhs(tmp_path, const, 0.0)
    assert got == want


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_an_unmapped_model_fails_at_build_with_the_name(tmp_path):
    """And the refusal reaches the caller where they can act on it — building
    the Simulator — rather than mid-solve."""
    import bngsim

    net = tmp_path / "m.net"
    net.write_text(NET.format(body="erf(B)"))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(net))
        with pytest.raises(ValueError, match="erf"):
            bngsim.Simulator(model, method="ode", jacobian="jax")
