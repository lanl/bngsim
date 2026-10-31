"""Issue #1212 — codegen writes C for what ExprTk means, or declines.

The C emitter used to copy parts of a rate law into the generated C that C has
no meaning for:

* ExprTk functions <math.h> has no name for (``frac``, ``inrange``, ``iclamp``,
  ``roundn``, ``root``, ``logn``, ``sinc``, ``equal``, ``cot``, ``sec``, ``csc``,
  ``ncdf``, ``deg2rad``, ``rad2deg`` and the rest of exprtk.hpp's
  ``base_function_list``), and two it has under the same name with other
  arithmetic (``hypot``, ``log2``);
* a table read as ``tfun_kf`` or ``tfun_kf ()`` rather than ``tfun_kf()``, a name
  with a dot in it (``k.f``), and ExprTk's implicit products (``2tfun_kf()/2``,
  ``k(x + 1)``, ``(a)(b)``) and square and curly brackets;
* a parameter named ``e`` beside a literal such as ``2e-3``, which was written
  into its exponent;
* the physical constants other than ``_pi`` and ``_e`` (``_R``, ``_NA``...).

cc refused such a source, after the compile automatic codegen then fell back
from had been paid for, and ``codegen=True`` refused with the compiler's error.
The MIR JIT refused it by name, so under automatic codegen a model the cc path
ran fell over instead.

Each function is now C transcribed from its ``*_impl`` in exprtk.hpp, the names
resolve, and anything the translator still has no C for declines while the
expression is in hand, before any compiler sees it. The oracle throughout is the
interpreter (``codegen=False``): at the level of one expression, bit for bit.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim._bngsim_core import ModelBuilder

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
_EXE = ".exe" if os.name == "nt" else ""
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH for the codegen leg")
needs_mir = pytest.mark.skipif(
    not getattr(bngsim, "HAS_MIR", False),
    reason="bngsim built without the MIR backend (configure with -DBNGSIM_ENABLE_MIR=ON)",
)

NAN = float("nan")
INF = float("inf")

# name -> (expression over observables A, B, C, the argument tuples to try).
# Every argument set is chosen to land on each branch of the transcription: a
# negative and a positive side, the edges of a range, a half-way value, the
# region a guard tests for, and a NaN where the comparisons decide the answer.
FUNCTIONS: dict[str, tuple[str, tuple[tuple[float, ...], ...]]] = {
    # Not past 2**63: built by MSVC, which reports __cplusplus as 199711L,
    # ExprTk's trunc is its pre-C++11 (long long)v, undefined out there.
    "frac": ("frac(A)", ((2.75,), (-2.75,), (0.0,), (-0.0,), (5.0,), (4503599627370495.5,),
                         (-4503599627370495.5,), (NAN,))),
    "inrange": (
        "inrange(A, B, C)",
        ((0.0, 0.5, 1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, -0.1, 1.0), (0.0, 1.1, 1.0),
         (1.0, 0.5, 0.0), (0.0, NAN, 1.0)),
    ),
    "iclamp": (
        "iclamp(A, B, C)",
        ((0.0, 0.25, 1.0), (0.0, 0.75, 1.0), (0.0, 0.5, 1.0), (0.0, -2.0, 1.0), (0.0, 3.0, 1.0),
         (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.0, NAN, 1.0)),
    ),
    "roundn": (
        "roundn(A, B)",
        ((2.345, 2.0), (-2.345, 2.0), (2.5, 0.0), (-2.5, 0.0), (1.23456, 2.9), (1.23456, -3.0),
         (1.23456, 40.0), (0.125, 2.0), (-0.125, 2.0)),
    ),
    "root": (
        "root(A, B)",
        ((8.0, 3.0), (-8.0, 3.0), (-8.0, 2.0), (-8.0, 2.5), (16.0, 4.0), (0.0, 3.0), (2.0, -2.0),
         (-27.0, -3.0)),
    ),
    "logn": ("logn(A, B)", ((8.0, 2.0), (100.0, 10.0), (0.5, 3.0), (3.0, 0.5))),
    "sinc": (
        "sinc(A)",
        ((0.0,), (1e-17,), (-1e-17,), (2.220446049250313e-16,), (1e-15,), (0.5,), (-3.0,), (NAN,)),
    ),
    "equal": (
        "equal(A, B)",
        ((0.5, 0.5), (1.0, 1.0 + 5e-11), (1.0, 1.0 + 2e-10), (1e12, 1e12 + 50.0),
         (1e12, 1e12 + 500.0), (0.0, 1e-11), (NAN, NAN), (-3.0, 3.0)),
    ),
    "not_equal": (
        "not_equal(A, B)",
        ((0.5, 0.5), (1.0, 1.0 + 5e-11), (1.0, 1.0 + 2e-10), (1e12, 1e12 + 500.0), (NAN, 1.0)),
    ),
    "cot": ("cot(A)", ((0.5,), (-1.2,), (3.0,))),
    "sec": ("sec(A)", ((0.5,), (-1.2,), (3.0,))),
    "csc": ("csc(A)", ((0.5,), (-1.2,), (3.0,))),
    "ncdf": ("ncdf(A)", ((0.0,), (1.0,), (-2.5,), (8.0,), (-40.0,))),
    "deg2rad": ("deg2rad(A)", ((180.0,), (-45.0,), (1.0,), (123.456,))),
    "rad2deg": ("rad2deg(A)", ((math.pi,), (-0.5,), (1.0,), (2.0,))),
    "deg2grad": ("deg2grad(A)", ((90.0,), (-1.0,), (7.3,))),
    "grad2deg": ("grad2deg(A)", ((100.0,), (-1.0,), (7.3,))),
    "hypot": ("hypot(A, B)", ((3.0, 4.0), (1e200, 1e200), (1e-200, 1e-200), (-2.0, 0.5))),
    "log2": ("log2(A)", ((8.0,), (10.0,), (0.3,), (12345.678,), (3.0,), (7.0,))),
    "mul": ("mul(A, B, C)", ((2.0, 3.0, 4.0), (0.1, 0.2, 0.3), (-1.5, 2.0, NAN))),
    "mand": ("mand(A, B, C)",
             ((1.0, 2.0, 3.0), (1.0, 0.0, 3.0), (NAN, 1.0, 1.0), (0.0, NAN, 1.0))),
    "mor": ("mor(A, B, C)",
            ((0.0, 0.0, 0.0), (0.0, -0.0, 2.0), (NAN, 0.0, 0.0), (0.0, 0.0, -1.0))),
}  # fmt: skip

OBS = ("A", "B", "C")


def _interpreter(expr: str, args: tuple[float, ...], t: float = 0.0, param: str = "k") -> float:
    """``expr`` as the engine computes it: the derivative of ``0 -> PROBE`` at that
    rate, from the same ``compute_derivs`` CVODE reads. The one parameter is 0.5."""
    builder = ModelBuilder()
    builder.add_parameter(param, 0.5)
    values = (*args, *(1.0,) * (len(OBS) - len(args)))
    for idx, (name, value) in enumerate(zip(OBS, values, strict=False)):
        builder.add_species(f"S_{name}", value)
        builder.add_observable(name, [(idx, 1.0)])
    probe = builder.add_species("PROBE", 0.0)
    builder.add_function("probe_rate", expr)
    builder.add_reaction([], [probe], "functional", "probe_rate")
    model = builder.build()
    state = np.array([*values, 0.0], dtype=np.float64)
    return float(model.compute_derivs(t, state)[probe])


_C_PROGRAM = """#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>
{prelude}
int main(void) {{
    static const double rows[][3] = {{{rows}}};
    double p[] = {{0.5}};
    double t = 0.0;
    (void)p; (void)t;
    for (unsigned r = 0; r < sizeof(rows) / sizeof(rows[0]); ++r) {{
        const double *obs = rows[r];
        double v = (double)({body});
        unsigned long long bits;
        memcpy(&bits, &v, sizeof bits);
        printf("%llu\\n", bits);
    }}
    return 0;
}}
"""


def _c_double(x: float) -> str:
    if math.isnan(x):
        return "NAN"
    if math.isinf(x):
        return "INFINITY" if x > 0 else "-INFINITY"
    return "-0.0" if x == 0.0 and math.copysign(1.0, x) < 0 else repr(x)


def _compiled(expr: str, rows: list[tuple[float, ...]], param: str = "k") -> list[float]:
    """``expr`` through the codegen translator, compiled with the generated
    source's own prelude, at each row of observable values."""
    body = cg._expr_to_c(expr, [param], [], list(OBS), [])
    padded = [(*r, *(1.0,) * (len(OBS) - len(r))) for r in rows]
    source = _C_PROGRAM.format(
        prelude="\n".join(cg._CODEGEN_PRELUDE_LINES),
        rows=", ".join("{" + ", ".join(_c_double(v) for v in r) + "}" for r in padded),
        body=body,
    )
    with tempfile.TemporaryDirectory() as tmp:
        c_path = Path(tmp) / "probe.c"
        exe = Path(tmp) / f"probe{_EXE}"
        c_path.write_text(source)
        assert _CC is not None
        built = subprocess.run(
            [_CC, "-O2", "-ffp-contract=off", str(c_path), "-o", str(exe), "-lm"],
            capture_output=True,
            text=True,
        )
        assert built.returncode == 0, (
            f"{expr!r} emitted {body!r}, which does not compile:\n{built.stderr}"
        )
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    return [struct.unpack("<d", struct.pack("<Q", int(line)))[0] for line in out.split()]


def _same(a: float, b: float) -> bool:
    """Equal as doubles, NaN equal to NaN; a zero's sign counts."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


# ── One expression, bit for bit ──────────────────────────────────────────────


@needs_cc
@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_the_compiled_function_is_the_interpreters_bit_for_bit(name):
    """Each transcription computes what ExprTk computes, at every branch."""
    expr, rows = FUNCTIONS[name]
    compiled = _compiled(expr, list(rows))
    for args, got in zip(rows, compiled, strict=False):
        want = _interpreter(expr, args)
        assert _same(got, want), f"{expr} at {args}: compiled {got!r}, interpreter {want!r}"


@needs_cc
@pytest.mark.parametrize("name", ["hypot", "log2"])
def test_c_s_own_function_of_the_same_name_is_not_the_engines(name):
    """Why hypot and log2 are transcribed although C has both: at one of the
    rows above, C's gives another double than ExprTk's. (Negative control: if
    this stops holding the transcription is still right, but no longer needed.)"""
    expr, rows = FUNCTIONS[name]
    c_name = {"hypot": "hypot(obs[0], obs[1])", "log2": "log2(obs[0])"}[name]
    source = _C_PROGRAM.format(
        prelude="",
        rows=", ".join(
            "{" + ", ".join(_c_double(v) for v in (*r, *(1.0,) * (3 - len(r)))) + "}" for r in rows
        ),
        body=c_name,
    )
    with tempfile.TemporaryDirectory() as tmp:
        c_path, exe = Path(tmp) / "c.c", Path(tmp) / f"c{_EXE}"
        c_path.write_text(source)
        subprocess.run([_CC, "-O0", str(c_path), "-o", str(exe), "-lm"], check=True)
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    libm = [struct.unpack("<d", struct.pack("<Q", int(v)))[0] for v in out.split()]
    assert any(not _same(c, _interpreter(expr, a)) for a, c in zip(rows, libm, strict=False))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_every_function_is_one_the_engine_accepts(name):
    """The rows are only a reference if ExprTk compiles the expression."""
    expr, rows = FUNCTIONS[name]
    assert any(math.isfinite(_interpreter(expr, a)) for a in rows)


def test_the_list_covers_every_exprtk_function_c_does_not_have():
    """Every name in exprtk.hpp's base_function_list is either here, rewritten
    by an older rule, or a <math.h> call of the same meaning. A function added
    upstream without C fails this rather than reaching a compiler."""
    header = Path(__file__).resolve().parents[2] / "third_party" / "exprtk" / "exprtk.hpp"
    if not header.is_file():
        pytest.skip("exprtk.hpp not found beside the tests")
    text = header.read_text(errors="replace")
    start = text.index("static const std::string base_function_list[]")
    names = set(re.findall(r'"(\w+)"', text[start : text.index("};", start)]))
    handled = (
        set(FUNCTIONS)
        | set(cg._ENGINE_CALL_NAMES)
        | cg._C_CALLS_AS_WRITTEN
        | {name for name, (rep, _) in cg._BUILTIN_IDENT_MAP.items() if rep != name}
    )
    # Not numeric: a string match and an assignment, which no rate law holds.
    assert names - handled == {"like", "swap"}


# ── Whole models, through each backend ───────────────────────────────────────

_NET = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 2.0
    2 B() 0
end species
begin functions
    1 g() Atot + 1
    2 f() {law}
end functions
begin reactions
    1 1 2 f #_R1
end reactions
begin groups
    1 Atot 1
end groups
"""

# A rate law that reads each function where it changes along the run, so a wrong
# transcription moves the trajectory.
_LAWS = {
    "frac": "k*(1 + frac(3*Atot - 4.5))",
    "inrange": "k*(1 + inrange(0.5, Atot, 1.5))",
    "iclamp": "k*(1 + iclamp(0.5, Atot, 1.5))",
    "roundn": "k*(1 + roundn(Atot - 1.3, 1))",
    "root": "k*(1 + root(Atot - 1.5, 3))",
    "logn": "k*(1 + logn(Atot + 1, 3))",
    "sinc": "k*(1 + sinc(3*Atot - 3))",
    "equal": "k*(1 + equal(k, 0.5))",
    "not_equal": "k*(1 + not_equal(Atot, 2))",
    "cot": "k*(1 + cot(Atot + 0.3))",
    "sec": "k*(1 + sec(Atot - 1.4))",
    "csc": "k*(1 + csc(Atot + 0.3))",
    "ncdf": "k*(1 + ncdf(4*Atot - 5))",
    "deg2rad": "k*(1 + sin(deg2rad(90*Atot)))",
    "rad2deg": "k*(1 + rad2deg(Atot)/100)",
    "deg2grad": "k*(1 + deg2grad(Atot)/10)",
    "grad2deg": "k*(1 + grad2deg(Atot)/10)",
    "hypot": "k*hypot(Atot, k)",
    "log2": "k*(1 + log2(Atot + 1))",
    "mul": "mul(k, Atot, 0.5)",
    "mand": "k*(1 + mand(Atot > 1, k))",
    "mor": "k*(1 + mor(Atot > 1.5, 0))",
    "_R": "k*_R*Atot/8",
    "_NA": "k*_NA*Atot*1e-24",
    "_kB": "k*_kB*Atot*1e23",
    "_h": "k*_h*Atot*1e34",
    "_F": "k*_F*Atot*1e-5",
    # The parser's products: a bracket after a group, an observable, a function
    # of the model, its empty call and a constant; a number ending in its dot;
    # square and curly brackets.
    "group_bracket": "k*(Atot)(k + 1)",
    "observable_bracket": "k*Atot(k + 1)",
    "function_bracket": "k*g(Atot)/2",
    "empty_call_bracket": "k*g()(Atot)/2",
    "constant_bracket": "k*_pi(Atot)/3",
    "number_dot": "2.k*Atot",
    "square_curly": "k*[Atot + 1]{k}",
}

# The runs below are at rtol=1e-10, atol=1e-12. On macOS arm64 the compiled and
# interpreted runs of every law here agree bit for bit, under cc and MIR. The
# bounds allow for a platform whose compiled law rounds a bit apart, which CVODE
# carries into the trajectory at about its own tolerance: 10x its rtol, its atol.
_RUN_IN_A_CHILD = """
import sys, json
import bngsim
out = {}
for cg in (False, True):
    sim = bngsim.Simulator(bngsim.Model.from_net(sys.argv[1]), method="ode", codegen=cg)
    r = sim.run(t_span=(0.0, 3.0), n_points=7, rtol=1e-10, atol=1e-12)
    out[str(cg)] = r.species[:, 1].tolist()
print(json.dumps(out))
"""


def _runs(tmp_path, law: str, jit: str | None, template: str = _NET) -> dict:
    net = tmp_path / "m.net"
    net.write_text(template.format(law=law))
    env = {k: v for k, v in os.environ.items() if k != "BNGSIM_CODEGEN_JIT"}
    if jit:
        env["BNGSIM_CODEGEN_JIT"] = jit
    child = subprocess.run(
        [sys.executable, "-c", _RUN_IN_A_CHILD, str(net)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert child.returncode == 0, child.stderr[-3000:]
    import json

    return json.loads(child.stdout.strip().splitlines()[-1])


@needs_cc
@pytest.mark.parametrize("name", list(_LAWS))
def test_codegen_true_runs_it_and_matches_the_interpreter(tmp_path, name):
    """``codegen=True`` refused every one of these with the compiler's error."""
    out = _runs(tmp_path, _LAWS[name], None)
    np.testing.assert_allclose(out["True"], out["False"], rtol=1e-9, atol=1e-12)


@needs_mir
def test_the_mir_jit_runs_each_and_matches_the_interpreter(tmp_path):
    """Under MIR they were refused by name, so automatic codegen fell over where
    cc fell back. One test over every law, so a build without MIR skips once."""
    for name, law in _LAWS.items():
        out = _runs(tmp_path, law, "mir")
        np.testing.assert_allclose(out["True"], out["False"], rtol=1e-9, atol=1e-12, err_msg=name)


# ── Implicit products and names with a dot ───────────────────────────────────


@pytest.mark.parametrize(
    ("text", "written"),
    [
        ("2k", "2*k"),
        ("2.0tfun_kf()/2", "2.0*tfun_kf()/2"),
        ("2(k + 1)", "2*(k + 1)"),
        ("(k + 1)2", "(k + 1)*2"),
        ("(k)A", "(k)*A"),
        ("k 2", "k *2"),
        ("k A", "k *A"),
        ("1.5e3k", "1.5e3*k"),
        ("2k.f/2", "2*k.f/2"),
        ("(k)(A)", "(k)*(A)"),
        ("[k]{A}", "(k)*(A)"),
        ("2[k]", "2*(k)"),
        ("2.k", "2.*k"),
        ("2.(k)", "2.*(k)"),
    ],
)
def test_an_implicit_product_is_written_out(text, written):
    assert cg._insert_implicit_products(text) == written


@pytest.mark.parametrize(
    "text",
    [
        "k*A + 2",
        "1e-3*k",
        "2.5E-3*A",
        "if(A > 1 and k < 2, 1, 0)",
        "if((A > 1) or (k < 2), 1, 0)",
        "not(A > 1)",
        "k.f + x1.y2",
        "sin (A)",
        "time ()",
        "1 and 0",
    ],
)
def test_text_with_no_implicit_product_is_left_alone(text):
    assert cg._insert_implicit_products(text) == text


# What a name is decides what a bracket after it is: a product after a variable
# (a scalar of the model, or a constant), a call after anything else.
_LOOKUP = cg._build_ident_lookup_model(
    {"k": "p[0]"}, {}, {"A": "obs[0]", "B": "obs[1]"}, {"g": "func[0]"}
)


@pytest.mark.parametrize(
    ("text", "written"),
    [
        ("k(A)", "k*(A)"),
        ("k (A)", "k *(A)"),
        ("A[k]", "A*(k)"),
        ("g(A)", "g*(A)"),
        ("g()(A)", "g()*(A)"),
        ("_pi(A)", "_pi*(A)"),
        ("k(A)(B)", "k*(A)*(B)"),
    ],
)
def test_a_bracket_after_a_variable_is_a_product(text, written):
    assert cg._insert_implicit_products(text, _LOOKUP) == written


@pytest.mark.parametrize(
    "text",
    ["exp(A)", "sin (A)", "max(A, B)", "g()", "g() + k", "time()", "if(A > 1, k, B)", "k*(A)"],
)
def test_a_bracket_after_anything_else_is_left_alone(text):
    assert cg._insert_implicit_products(text, _LOOKUP) == text


@pytest.mark.parametrize(
    "text",
    [
        "exp(A)(B)",
        "k*exp(-A)(B)",
        "if(A > 1, 1, 2)(B)",
        "not(A > 1)(B)",
        "time() (B)",
        "max(A, k)[B]",
    ],
)
def test_a_bracket_after_a_call_declines(text):
    """The engine reads the bracket as a second statement, and gives its value
    for the whole expression: each of these is B. There is no C for that here."""
    assert _interpreter(text, (2.5, 0.75)) == 0.75
    with pytest.raises(cg.CodegenDeclined, match="second statement"):
        cg._insert_implicit_products(text, _LOOKUP)


def test_a_bracket_after_a_table_declines():
    """A table is a call to the engine, as ``time`` is: ``2*tfun_kf()(Atot)`` is
    Atot, the second statement, though the lookup holds the table's name."""
    model = _table_model("tfun_kf()(Atot)")
    assert model._core.compute_derivs(0.0, np.array([0.0, 1.0]))[0] == 1.0
    with pytest.raises(cg.CodegenDeclined, match="'tfun_kf'.*second statement"):
        cg.generate_rhs_from_model(model)


# The parser's products besides the lexer's (post_variable_process and
# post_bracket_process in exprtk.hpp), each with the precedence of a written `*`
# (`2/A(B)` is 2/A times B); square and curly brackets; a number ending in its
# dot. And what is not a product: a call, and a number's exponent, which stays
# in its number.
_PRODUCTS = [
    "(A)(B)", "(A + 1)(B + 1)", "2(A)(B)", "(A)(B)C", "A(B)", "A(B + 1)", "A (B)", "(A) (B)",
    "A()(B)", "k(A)", "_pi(A)", "A(B)(C)", "A(B)^2", "2/A(B)", "A^2(B)", "-A(B)", "sin(A(B))",
    "A[B]", "(A)[B](C)", "{A}[B]", "2[A]", "[A + B]k", "2.A", "2. A", "2.(A)",
    "exp(A)B", "max(A, B)C", "sin (A)", "frac(A)B",
    "2.e3*A", "2.e-3A", "1.e5*A", ".5e2A", "3.E+2*A", "(A > 1) and (B)(C)",
]  # fmt: skip


# The C library's own functions, which a product may call: compiled, they are
# the platform's libm, and the engine's are whatever its own build linked. On
# Windows the two give sin(2.5) an ulp apart, so a law calling one is held to a
# few ulp; how it parses is what these ask, and a wrong parse is off by more.
_LIBM_CALL = re.compile(r"\b(sin|cos|tan|exp|log|sqrt|pow)\s*\(")


def _same_or_libm(got: float, want: float, expr: str) -> bool:
    if _same(got, want) or not _LIBM_CALL.search(expr):
        return _same(got, want)
    return math.isfinite(want) and abs(got - want) <= 4 * math.ulp(want)


@needs_cc
@pytest.mark.parametrize(
    "expr", ["2k*A", "2(A + 1)", "(A + 1)2", "(A)B", "k 2", "A B", "3A^2", *_PRODUCTS]
)
def test_an_implicit_product_computes_what_the_engine_does(expr):
    rows = [(2.5, 0.75, 1.0), (-0.4, 3.2, 1.0)]
    for args, got in zip(rows, _compiled(expr, rows), strict=False):
        assert _same_or_libm(got, _interpreter(expr, args), expr), expr


def test_a_libm_call_is_held_to_ulps_and_a_misparse_is_not_hidden():
    """Control for the tolerance: a law that calls libm passes an ulp apart, and
    sin(A*B), what a misparse of sin(A)B gives, is far outside it."""
    want = math.sin(2.5) * 0.75
    assert _same_or_libm(math.nextafter(want, math.inf), want, "sin(A)B")
    assert not _same_or_libm(math.sin(2.5 * 0.75), want, "sin(A)B")
    assert not _same_or_libm(math.nextafter(want, math.inf), want, "(A)(B)")


# A parameter named after an exponent's letter: the `e` of `2e-3` is the number's,
# and the parameter is read outside it. A .net keeps the literal as written, and
# the SBML loader writes a number as repr(v), in e-notation below 1e-4 and from
# 1e16: BIOMD0000001042 has `1e-10` beside a parameter `e`.
_EXPONENT_LETTERS = [
    ("e", "2e-3*e*A"), ("e", "2.e-3*e*A"), ("E", "2E-3*E*A"), ("e", "1.5e3*e*A"),
    ("e", "e*2e3*A"), ("e", "2E-3*e*A"), ("E", "2e-3*E*A"),
    ("e", "e^2*A"), ("e", "2*e*A"), ("e", "(e)*A"), ("e", "e(A)"),
]  # fmt: skip


@needs_cc
@pytest.mark.parametrize(("param", "expr"), _EXPONENT_LETTERS)
def test_a_parameter_named_like_an_exponent_is_read_outside_the_number(param, expr):
    rows = [(2.5, 0.75, 1.0)]
    (got,) = _compiled(expr, rows, param)
    assert _same(got, _interpreter(expr, rows[0], param=param)), expr


@pytest.mark.parametrize(
    "jit", [pytest.param(None, marks=needs_cc), pytest.param("mir", marks=needs_mir)]
)
def test_codegen_true_runs_a_net_law_beside_a_parameter_named_e(tmp_path, jit):
    """``2e-3*e*Atot`` was written ``2p[1]-3*p[1]*obs[0]``, which no compiler takes."""
    template = _NET.replace("1 k 0.5", "1 k 0.5\n    2 e 0.5")
    out = _runs(tmp_path, "k*(1 + 2e-3*e*Atot)", jit, template)
    np.testing.assert_allclose(out["True"], out["False"], rtol=1e-9, atol=1e-12)


# A -> B at k*e*(A + 1e-10), the literal as BIOMD0000001042's MathML writes it.
_SBML_E = (
    '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
    '<model id="m"><listOfCompartments><compartment id="c" size="1" constant="true"/>'
    "</listOfCompartments><listOfSpecies>"
    '<species id="A" compartment="c" initialConcentration="2" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
    '<species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
    '</listOfSpecies><listOfParameters><parameter id="k" value="0.5" constant="true"/>'
    '<parameter id="e" value="0.5" constant="true"/></listOfParameters>'
    '<listOfReactions><reaction id="r" reversible="false">'
    '<listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/>'
    "</listOfReactants><listOfProducts>"
    '<speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>'
    '<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci>'
    '<ci>e</ci><apply><plus/><ci>A</ci><cn type="e-notation">1<sep/>-10</cn></apply></apply>'
    "</math></kineticLaw></reaction></listOfReactions></model></sbml>"
)


@needs_cc
def test_codegen_true_runs_an_sbml_law_beside_a_parameter_named_e():
    """The law was written ``(p[1]*p[2]*(obs[0]+1p[2]-10))``, and the build failed."""
    runs = {
        cg_on: bngsim.Simulator(
            bngsim.Model.from_sbml_string(_SBML_E), method="ode", codegen=cg_on
        ).run(t_span=(0.0, 3.0), n_points=7, rtol=1e-10, atol=1e-12)
        for cg_on in (False, True)
    }
    np.testing.assert_allclose(
        np.asarray(runs[True].species), np.asarray(runs[False].species), rtol=1e-9, atol=1e-12
    )


# ── Tables, by whatever spelling reads them ──────────────────────────────────


def _table_model(call: str, table: str = "kf") -> bngsim.Model:
    """T' = 2*<call>, where <call> reads a table indexed by the function ix().
    ``h`` is declared first, so only the dependency on ix puts it after: read
    before ix is written, the table is read at an unset index."""
    b = ModelBuilder()
    b.add_parameter("ix", 0.0)
    t = b.add_species("T", 0.0)
    a = b.add_species("A", 1.0)
    b.add_observable("Atot", [(a, 1.0)])
    b.add_function("h", f"2*{call}")
    b.add_function("ix", "Atot")
    b.add_function(table, "0")
    b.add_inline_table_function_spec(table, [0.0, 1.0, 2.0], [0.0, 3.0, 4.0], "ix", "linear")
    b.add_reaction([], [t], "functional", "h")
    b.add_function("decay", "0.5*Atot")
    b.add_reaction([a], [], "functional", "decay")
    return bngsim.Model(_core=b.build())


_TABLE_CALLS = [
    ("tfun_kf()", "kf"),
    ("tfun_kf ()", "kf"),
    ("tfun_kf\t()", "kf"),
    ("tfun_kf", "kf"),
    ("2tfun_kf()/2", "kf"),
    ("2.0tfun_kf()/2", "kf"),
    ("kf", "kf"),
    ("k.f", "k.f"),
    ("tfun_k.f()", "k.f"),
    ("2k.f/2", "k.f"),
]


@pytest.mark.parametrize(("call", "table"), _TABLE_CALLS)
def test_each_spelling_reads_the_table_after_its_index(call, table):
    """The emitted statement reads the table, and comes after the statement for
    the function its index is: `tfun_kf ()` was copied through as an undeclared
    call, and was ordered before `ix` too, which would have read it unset."""
    data = _table_model(call, table)._core.codegen_data()
    names = [f["name"] for f in data["functions"]]
    calls = cg._tfun_call_map(data)
    order = cg._topological_function_order(data["functions"], calls)
    assert order.index(names.index("h")) > order.index(names.index("ix"))
    src = cg.generate_rhs_from_model(_table_model(call, table))
    line = next(ln for ln in src.splitlines() if ln.rstrip().endswith("/* h */") and "func[" in ln)
    assert "data->tfun_eval(" in line or f"func[{names.index(table)}]" in line, line
    assert "tfun_k" not in line.split("=", 1)[1].replace("data->tfun_eval", ""), line


@needs_cc
@pytest.mark.parametrize(("call", "table"), _TABLE_CALLS)
def test_each_spelling_runs_compiled_as_the_call_runs_interpreted(call, table):
    """Each spelling, compiled, against the interpreter on ``tfun_kf()``: the
    same model, read by a spelling the interpreter orders right.

    Not against the interpreter on the same spelling. In a model built through
    ``ModelBuilder`` with ``h`` declared first, where ``h`` reads the table by
    the function's name (``kf``), or by any spelling once the table's name holds
    a dot (``k.f``), the interpreter reads the table one evaluation late, which
    moves this run by 6e-7 at rtol 1e-10. A ``.net`` with the same functions in
    the same order is evaluated right: that is ``ModelBuilder``'s ordering, a
    separate bug, and not the codegen's."""
    reference = bngsim.Simulator(_table_model("tfun_kf()", "kf"), method="ode", codegen=False).run(
        t_span=(0.0, 2.0), n_points=5, rtol=1e-10, atol=1e-12
    )
    compiled = bngsim.Simulator(_table_model(call, table), method="ode", codegen=True).run(
        t_span=(0.0, 2.0), n_points=5, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(compiled.species), np.asarray(reference.species), rtol=1e-9, atol=1e-12
    )


# ── What it still has no C for ───────────────────────────────────────────────

# Engine-valid, and with no C the translator writes: ExprTk's word operators
# other than and/or/not, and a bracket after a call, which the engine reads as a
# second statement (this law is Atot).
_NO_C = {
    "xor": "k*(1 + if(Atot > 1 xor k > 1, 1, 0))",
    "nand": "k*(1 + if(Atot > 1 nand k > 1, 1, 0))",
    "exp": "k*exp(-Atot)(Atot)",
}


@pytest.mark.parametrize("name", list(_NO_C))
def test_a_name_with_no_c_declines_before_any_compiler_sees_it(tmp_path, monkeypatch, name):
    """Automatic codegen falls back to the interpreter, and no compile is paid
    for: the decline is made while the expression is in hand."""
    net = tmp_path / "m.net"
    net.write_text(_NET.format(law=_NO_C[name]))
    model = bngsim.Model.from_net(str(net))

    def no_compile(*a, **k):
        raise AssertionError("a source with no C for it reached the compiler")

    monkeypatch.setattr(cg, "compile_rhs", no_compile)
    assert cg.prepare_model_codegen(model) is None
    assert cg.last_codegen_error() is None
    assert repr(name) in (cg.last_codegen_decline() or "")
    assert cg.prepare_model_codegen_source(model) is None
    assert repr(name) in (cg.last_codegen_decline() or "")


@pytest.mark.parametrize("name", list(_NO_C))
def test_codegen_true_refuses_it_by_name(tmp_path, name):
    net = tmp_path / "m.net"
    net.write_text(_NET.format(law=_NO_C[name]))
    with pytest.raises(Exception, match=repr(name)):
        bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode", codegen=True)


@pytest.mark.parametrize("name", list(_NO_C))
def test_the_interpreter_runs_what_codegen_declines(tmp_path, name):
    net = tmp_path / "m.net"
    net.write_text(_NET.format(law=_NO_C[name]))
    r = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
        t_span=(0, 1), n_points=3
    )
    assert np.isfinite(np.asarray(r.species)).all()
