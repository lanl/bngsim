"""Issue #507: ``max`` / ``min`` / ``abs`` over a differentiation variable are
differentiated as the piecewise functions they are.

sympy differentiates ``Max`` to ``Heaviside`` and ``Abs`` (over the plain symbols
the parser creates) to ``re`` / ``im``; no emitter spells either, so
``differentiate_rate_law`` declined the rate law and the whole model ran on the
finite-difference Jacobian — 28 corpus models, a Lorenz attractor among them.
Now ``abs`` differentiates to ``sign(a)·a'`` through a twin of ``Abs`` with that
``fdiff``, and the ``Heaviside`` factors of a ``max`` / ``min`` derivative are
spelled ``if(u >= 0, 1, 0)`` afterwards — both forms the emitters print, and
neither copies the argument the way a Piecewise rewrite of the value would.
"""

from __future__ import annotations

import glob
from pathlib import Path

import bngsim
import numpy as np
import pytest
import sympy as sp
from bngsim import _jacobian as J

FIXTURE = "jac_kinks_max_min_abs.net"


def _at(expr, **vals):
    return float(expr.subs({sp.Symbol(k): v for k, v in vals.items()}))


# ─── The symbolic core ───────────────────────────────────────────────────────


def test_max_over_two_observables_differentiates_branchwise():
    dd = J.differentiate_rate_law("k*max(A,B)", {}, {"A", "B"}, {"k"})
    assert dd is not None and set(dd) == {"A", "B"}
    assert (_at(dd["A"], A=2, B=1, k=3), _at(dd["B"], A=2, B=1, k=3)) == (3.0, 0.0)
    assert (_at(dd["A"], A=1, B=2, k=3), _at(dd["B"], A=1, B=2, k=3)) == (0.0, 3.0)
    for e in dd.values():
        s = J.sympy_to_exprtk(e)
        assert s is not None and "if(" in s and "Heaviside" not in s, s


def test_min_against_a_constant_differentiates_branchwise():
    dd = J.differentiate_rate_law("k*min(A,1.0)", {}, {"A"}, {"k"})
    assert dd is not None and set(dd) == {"A"}
    assert _at(dd["A"], A=0.5, k=3) == 3.0
    assert _at(dd["A"], A=1.5, k=3) == 0.0


def test_abs_differentiates_to_sign_times_the_inner_derivative():
    dd = J.differentiate_rate_law("k*abs(A-C)", {}, {"A", "C"}, {"k"})
    assert dd is not None and set(dd) == {"A", "C"}
    assert (_at(dd["A"], A=2, C=1, k=3), _at(dd["C"], A=2, C=1, k=3)) == (3.0, -3.0)
    assert (_at(dd["A"], A=1, C=2, k=3), _at(dd["C"], A=1, C=2, k=3)) == (-3.0, 3.0)
    s = J.sympy_to_exprtk(dd["A"])
    assert s is not None and "sign(" in s and "re(" not in s and "im(" not in s, s
    # the derivative does not copy the argument: one sign(), no if() branches
    assert "if(" not in s, s


def test_n_ary_max_and_nested_kinks_emit():
    dd = J.differentiate_rate_law("k*max(A,B,C)", {}, {"A", "B", "C"}, {"k"})
    assert dd is not None and set(dd) == {"A", "B", "C"}
    assert _at(dd["C"], A=1, B=2, C=3, k=1) == 1.0 and _at(dd["A"], A=1, B=2, C=3, k=1) == 0.0
    nested = J.differentiate_rate_law("k*max(abs(A),B)", {}, {"A", "B"}, {"k"})
    assert nested is not None
    assert _at(nested["A"], A=-3, B=1, k=1) == -1.0  # |A| > B, on the negative side
    assert _at(nested["A"], A=0.5, B=1, k=1) == 0.0  # B wins
    assert all(J.sympy_to_exprtk(e) is not None for e in nested.values())


def test_kinks_over_parameters_only_are_not_rewritten():
    """``abs(k)`` over a parameter is a value factor of the derivative and prints
    as ``abs(k)``, not as an ``if()`` — only a differentiation variable under the
    function needs the rewrite."""
    terms = J.build_per_observable_terms("abs(k)*max(k,Km)*A", {}, {"A"}, {"k", "Km"})
    assert terms == [("A", "abs(k)*max(k,Km)")] or (
        terms is not None and "if(" not in terms[0][1] and "abs(k)" in terms[0][1]
    ), terms


def test_prepare_leaves_piecewise_conditions_and_parameter_abs_alone():
    A, k = sp.Symbol("A"), sp.Symbol("k")
    absval = J._kink_bindings(sp)["absval"]
    expr = sp.Piecewise((k * sp.Abs(A) * sp.Abs(k), sp.Abs(A) > k), (sp.Max(A, k), True))
    out = J._prepare_kinks(expr, {"A"})
    (v1, c1), (v2, _c2) = out.args
    assert c1.has(sp.Abs) and not c1.has(absval), "the condition's Abs must survive"
    assert v1.has(absval) and v1.has(sp.Abs(k)), "Abs(A) is twinned, Abs(k) is not"
    assert isinstance(v2, sp.Max), "Max is left for sympy to differentiate"
    d = J._finish_kinks(sp.diff(out, A))
    assert not d.has(absval) and not d.has(sp.Heaviside), d
    assert J.sympy_to_exprtk(d) is not None


# ─── End to end ──────────────────────────────────────────────────────────────


def _central(m: bngsim.Model, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    n = y.size
    D = np.zeros((n, n))
    for j in range(n):
        s = h * max(abs(y[j]), 1.0)
        yp, ym = y.copy(), y.copy()
        yp[j] += s
        ym[j] -= s
        D[:, j] = (np.asarray(m.rhs(yp)) - np.asarray(m.rhs(ym))) / (2.0 * s)
    return D


def test_fixture_attaches_and_matches_finite_differences_on_every_branch(data_dir: Path):
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    assert m.analytical_jacobian_status == "complete"
    names = list(m.species_names)
    iA, iB, iC = (next(i for i, n in enumerate(names) if n.startswith(x)) for x in "ABC")
    rng = np.random.default_rng(507)
    branches = set()
    for _ in range(12):
        y = rng.uniform(0.05, 2.0, len(names))
        branches.add((y[iA] > y[iB], y[iA] > y[iC], y[iA] > 1.0))
        Jy = m.jacobian(y)
        assert Jy.source == "analytical"
        np.testing.assert_allclose(np.asarray(Jy), _central(m, y), rtol=1e-6, atol=1e-9)
    assert len(branches) >= 4, f"too few kink branches sampled: {branches}"


_LORENZ = glob.glob("benchmarks/suites/ode_fullnet/nets/*ph_lorenz_attractor.bngl.net")


@pytest.mark.skipif(not _LORENZ, reason="benchmark ode_fullnet corpus not present")
def test_lorenz_attractor_derives_and_attaches_off_its_seed_surface():
    """The issue's reproducer. Its ``max(X_raw, 0.001)`` guards used to stop the
    derivation outright; they derive now. What remains is the model's own
    construction: each signed rate is split into an ``if(c > 0, c, 0)`` and an
    ``if(c < 0, -c, 0)`` reaction, and the seed (x = y = z = 1) sits exactly on
    ``c = 0`` for all three, where each half's Piecewise derivative is 0 while
    the summed rate is smooth with slope σ. That is a point defect of the
    analytical formula, and the C++ gate is right to refuse it at that state —
    the reason it records says so. One step off the surface the Jacobian
    attaches and is exact, with the textbook 8 of 9 nonzeros."""
    m = bngsim.Model.from_net(_LORENZ[0])
    assert m.prepare_analytical_jacobian() is False
    assert m.analytical_jacobian_status.startswith("declined: the C++ attach declined")

    m = bngsim.Model.from_net(_LORENZ[0])
    m.set_concentration("LX()", 52.0)  # x = 2: no rate's switching argument is 0
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    rng = np.random.default_rng(1)
    n = len(m.species_names)
    y = rng.uniform(50.5, 52.0, n)
    Jy = m.jacobian(y)
    assert Jy.source == "analytical"
    assert int((np.abs(np.asarray(Jy)) > 0).sum()) == 8
    np.testing.assert_allclose(np.asarray(Jy), _central(m, y), rtol=1e-6, atol=1e-9)
