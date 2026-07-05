"""Compile-and-run oracle tests for the sympy→C Jacobian emitter (GH #76, Task 4).

``bngsim._jacobian.sympy_to_c`` emits the same differentiated rate-law sympy
expressions as C that ``sympy_to_exprtk`` emits as ExprTk — the codegen
counterpart of the interpreted analytical Jacobian. Each test emits C for a
derivative, compiles it with the system C compiler, runs it at several points,
and compares to sympy's own numeric evaluation (``subs`` + ``evalf``) — a real
oracle, not a string check. Skips cleanly when sympy or a C compiler is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim._jacobian import sympy_to_c  # noqa: E402

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler (cc/clang/gcc) on PATH")

# positive=True: rate-law observables/params are non-negative, and it lets sqrt /
# fractional powers evaluate on the substrate axis without complex branches.
S, Km, Vmax, k, n = sp.symbols("S Km Vmax k n", positive=True)


def _c_eval(expr, points):
    """Emit C for ``expr``, compile+run it at each point; return the C values
    (or ``None`` if the emitter declined). Symbols map positionally to ``x0..``."""
    names = sorted({str(s) for s in expr.free_symbols})
    idx = {nm: i for i, nm in enumerate(names)}
    c = sympy_to_c(expr, lambda nm: (f"x{idx[nm]}" if nm in idx else None))
    if c is None:
        return None
    decl = "".join(f"  double x{i}=atof(argv[{i + 1}]);\n" for i in range(len(names)))
    src = (
        "#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        "int main(int argc,char**argv){\n" + decl + f'  printf("%.17g\\n",(double)({c}));\n'
        "  return 0;\n}\n"
    )
    d = tempfile.mkdtemp()
    cf, ex = os.path.join(d, "t.c"), os.path.join(d, "t")
    with open(cf, "w") as fh:
        fh.write(src)
    subprocess.run([_CC, "-O2", "-o", ex, cf, "-lm"], check=True, capture_output=True)
    out = []
    for pt in points:
        args = [repr(float(pt[nm])) for nm in names]
        r = subprocess.run([ex, *args], capture_output=True, text=True, check=True)
        out.append(float(r.stdout.strip()))
    return out


def _oracle(expr, points):
    return [
        float(expr.subs({sp.Symbol(nm, positive=True): pt[nm] for nm in pt}).evalf())
        for pt in points
    ]


_P3 = [
    {"S": 2.0, "Km": 5.0, "Vmax": 3.0},
    {"S": 0.1, "Km": 0.7, "Vmax": 9.0},
    {"S": 40.0, "Km": 1.0, "Vmax": 2.0},
]
# Each case feeds the emitter the *derivative* expression it actually sees.
_CASES = [
    ("mm_dS", sp.diff(Vmax * S / (Km + S), S), _P3),
    ("mm_dKm", sp.diff(Vmax * S / (Km + S), Km), _P3),
    (
        "hill_dS",
        sp.diff(S**n / (Km**n + S**n), S),
        [
            {"S": 2.0, "Km": 5.0, "n": 2.0},
            {"S": 1.5, "Km": 0.4, "n": 3.0},
            {"S": 9.0, "Km": 2.0, "n": 2.5},
        ],
    ),
    (
        "rat_sqrt",
        sp.Rational(3, 2) * sp.sqrt(S) + sp.Rational(1, 3) * S,
        [{"S": 4.0}, {"S": 0.25}, {"S": 100.0}],
    ),
    ("exp_dS", sp.diff(sp.exp(-k * S), S), [{"S": 1.0, "k": 0.5}, {"S": 3.0, "k": 2.0}]),
    ("log_dS", sp.diff(sp.log(S + 1), S), [{"S": 1.0}, {"S": 9.0}]),
    ("abs_dS", sp.diff(sp.Abs(S - Km), S), [{"S": 7.0, "Km": 3.0}, {"S": 1.0, "Km": 4.0}]),
    ("min_SKm", sp.Min(S, Km), [{"S": 2.0, "Km": 5.0}, {"S": 8.0, "Km": 3.0}]),
    ("max_SKm", sp.Max(S, Km), [{"S": 2.0, "Km": 5.0}, {"S": 8.0, "Km": 3.0}]),
    (
        "piecewise",
        sp.Piecewise((Vmax * S, Km < S), (Vmax * Km, True)),
        [{"S": 7.0, "Km": 3.0, "Vmax": 2.0}, {"S": 1.0, "Km": 4.0, "Vmax": 2.0}],
    ),
    ("intpow", S**3 - 4 * S**2 + 2 * S, [{"S": 1.5}, {"S": 3.0}]),
]


@needs_cc
@pytest.mark.parametrize("expr,points", [(c[1], c[2]) for c in _CASES], ids=[c[0] for c in _CASES])
def test_c_emitter_matches_sympy_numerically(expr, points):
    cvals = _c_eval(expr, points)
    assert cvals is not None, "emitter unexpectedly declined an emittable expression"
    for cv, pv in zip(cvals, _oracle(expr, points), strict=False):
        assert abs(cv - pv) <= 1e-9 * (abs(pv) + 1e-12) or abs(cv - pv) <= 1e-9, (cv, pv)


@needs_cc
def test_c_emitter_removes_zero_axis_power_singularity():
    # d/dS of a Hill term has removable S^n/S factors. The C emitter must not
    # preserve the raw quotient, or codegen Jacobians evaluate NaN at S=0.
    expr = sp.diff((S / Km) ** n / (1 + (S / Km) ** n), S)
    vals = _c_eval(
        expr,
        [
            {"S": 0.0, "Km": 5.0, "n": 4.0},
            {"S": 0.0, "Km": 5.0, "n": 1.0},
        ],
    )
    assert vals is not None
    assert vals[0] == pytest.approx(0.0, abs=1e-15)
    assert vals[1] == pytest.approx(0.2, rel=1e-12)


def test_unresolvable_symbol_returns_none():
    # A free symbol the resolver cannot map must fail the whole emission rather
    # than reference an undefined C variable.
    assert sympy_to_c(sp.Symbol("ZZZ"), lambda nm: None) is None


def test_special_function_returns_none():
    # erf has no math.h / emitter mapping → FD fallback, never wrong C.
    assert sympy_to_c(sp.erf(S), lambda nm: "x0") is None


def test_resolver_drives_symbol_names():
    # The emitter owns only syntax; symbol→lvalue mapping is the caller's, so the
    # same derivative can target y[i] / param_values[idx] verbatim.
    c = sympy_to_c(
        sp.diff(Vmax * S / (Km + S), S),
        lambda nm: {
            "S": "y[0]",
            "Km": "data->param_values[1]",
            "Vmax": "data->param_values[2]",
        }.get(nm),
    )
    assert c is not None
    assert "y[0]" in c and "data->param_values[1]" in c and "data->param_values[2]" in c
