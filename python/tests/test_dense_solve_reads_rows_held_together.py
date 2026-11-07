"""The dense Newton solve keeps species at 0 that feed each other at exactly 0
(issue #1485).

#1471 reads a row with nothing off its diagonal as its own equation. Species at
0 that feed each other are not held alone: in a chain, Q' = kQ + P, Q's row
reads P's, and in a cycle, P' = Q' = k(P + Q), P's and Q's rows read each other.
But their rows read nothing else and their right-hand side is 0, so their
components are exactly 0. Partial pivoting does not see that. With the species
written V first, P's and Q's columns come last and they stay at 0; written P and
Q first, V's row, whose slope in them is 1e8, becomes their pivot, and the
back-solve leaves about 1e-16 of rounding there. The 0 is unstable (an e-fold
per unit of time), so by t = 40 the chain's V was 6.4e-3 off and the cycle's P
had reached 9.6e5. KLU's block-triangular form solves those rows by themselves,
and keeps them at 0 in any order; the dense solve now gives them 0 too.

Oracle: the closed form. P = Q = 0 and V = 1 - exp(-t) from 0.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_needs_klu = pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")

LAWS = {
    "chain": ("k*P", "k*Q + P", "1 + c*Q"),
    "cycle": ("k*(P + Q)", "k*(P + Q)", "1 + c*(P + Q)"),
}
RUN = {"t_span": (0, 40), "n_points": 41, "rtol": 1e-8, "atol": 1e-10}


def _model(shape: str, order: str) -> bngsim.Model:
    fp, fq, fv = LAWS[shape]
    decl = ", ".join(f"{s} = 0" for s in order)
    return bngsim.Model.from_antimony_string(
        f"model m\n  species {decl};\n  k = 1; c = 1e8;\n"
        f"  J1: -> P; {fp};\n  J2: -> Q; {fq};\n  J3: -> V; {fv};\n  J4: V -> ; V;\nend"
    )


def _species(model, **simulator) -> dict[str, np.ndarray]:
    r = bngsim.Simulator(model, method="ode", **simulator).run(**RUN)
    names = list(r.species_names)
    return {n: np.asarray(r.species)[:, names.index(n)] for n in names} | {"t": np.asarray(r.time)}


@pytest.mark.parametrize("order", ["VPQ", "QPV", "PQV"])
@pytest.mark.parametrize("shape", ["chain", "cycle"])
def test_species_at_zero_that_feed_each_other_stay_there(shape, order):
    """QPV and PQV: V off by 6.4e-3 (chain) and 6.4e13 (cycle) on main."""
    model = _model(shape, order)
    assert list(model.species_names) == list(order)
    x = _species(model, force_dense_linear_solver=True)
    np.testing.assert_array_equal(x["P"], 0.0)
    np.testing.assert_array_equal(x["Q"], 0.0)
    # 100 times the run's rtol: V is 4.7e-8 off here, and 6.4e-3 or more where the
    # rows at 0 leave it.
    np.testing.assert_allclose(x["V"], 1.0 - np.exp(-x["t"]), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("jacobian", ["analytical", "fd"])
@pytest.mark.parametrize("shape", ["chain", "cycle"])
def test_each_jacobian_route_reads_them(shape, jacobian):
    x = _species(_model(shape, "PQV"), force_dense_linear_solver=True, jacobian=jacobian)
    np.testing.assert_array_equal(x["P"], 0.0)
    np.testing.assert_array_equal(x["Q"], 0.0)


@_needs_klu
@pytest.mark.parametrize("shape", ["chain", "cycle"])
def test_dense_agrees_with_klu_on_the_species_at_zero(shape):
    """The control: KLU kept them at 0 before the fix as well."""
    x = _species(_model(shape, "PQV"), force_sparse_linear_solver=True)
    np.testing.assert_array_equal(x["P"], 0.0)
    np.testing.assert_array_equal(x["Q"], 0.0)
