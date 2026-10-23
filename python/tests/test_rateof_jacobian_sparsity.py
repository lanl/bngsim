"""The Jacobian sparsity pattern holds the entries a ``rate_of__X`` read adds
(issue #1239).

A law that reads ``rate_of__X``, the engine's live dX/dt, depends on every
species X's rate depends on: row X of the pattern. ``build_jac_sparsity`` gave
a functional rate law the species it reads through observables and functions,
and none through a rate. Wherever X's rate read a species the reading law did
not, the entry had no slot:

* the analytic Jacobian, once #1143 writes ``rate_of__X`` out, declines its
  attach ("lands in ∂f[S]/∂[P], which has no slot");
* and the colored finite-difference Jacobian of the sparse (KLU) route, which
  is colored from the pattern, dropped the entry. Its trajectories stayed
  within the tolerance, but stepped differently from the dense route's.

Oracle: every entry a central difference of the right-hand side finds nonzero,
at the start and at two perturbed states, has a slot in the pattern.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

MODELS = {
    # the issue's: S's size reads [P], whose dilution reads rate_of__Q
    "nested sizes": """
compartment C0 = 1; compartment C2; compartment C1; r = 0.5; q = 0.2;
C2 := 1 + q*Q; C1 := 2 + r*P;
species Q in C0 = 1; species P in C2 = 1; species S in C1 = 5;
Jq: Q -> ; 0.3*Q*C0; J1: P -> ; 0.25*P*C2; J2: S -> ; 0.2*S*C1;
""",
    # the issue's: a rate rule over rateOf
    "a rate rule over rateOf": """
compartment C0 = 1; k1 = 0.3; g = 0.7;
species P in C0 = 1; species S in C0 = 0;
J1: 2P -> ; k1*P*C0;
S' = g*rateOf(P);
""",
    # S reads rateOf(P) and P's rate reads rateOf(Q): the edges are a fixed point
    "a chain of rates": """
compartment C0 = 1; k = 0.4; g = 0.7; h = 0.3;
species Q in C0 = 2; species P in C0 = 1; species S in C0 = 0; species R in C0 = 1;
JQ: Q -> ; k*Q*R; JR: R -> ; 0.1*R;
P' = h*rateOf(Q) - 0.2*P;
S' = g*rateOf(P);
""",
    "rateOf through a function": """
compartment C0 = 1; k1 = 0.3; g = 0.7;
species P in C0 = 1; species S in C0 = 0; species W in C0 = 2;
J1: P -> ; k1*P*W; JW: W -> ; 0.05*W;
function f(x) x*x end
S' = g*f(rateOf(P));
""",
}
CONTROL = """
compartment C0 = 1; k1 = 0.3;
species P in C0 = 1; species S in C0 = 0;
J1: P -> S; k1*P*C0;
"""


def _missing(core) -> list[tuple[int, int]]:
    sp = core.jacobian_sparsity
    n = sp["n"]
    have = {
        (int(i), j)
        for j in range(n)
        for i in sp["row_indices"][sp["col_ptrs"][j] : sp["col_ptrs"][j + 1]]
    }
    x0 = core.get_state()
    rng = np.random.default_rng(0)
    states = [x0] + [x0 * (1 + 0.3 * rng.random(n)) + 0.1 for _ in range(2)]
    missing = set()
    for x in states:
        for j in range(n):
            h = 1e-6 * max(1.0, abs(x[j]))
            e = np.zeros(n)
            e[j] = h
            col = (core.compute_derivs(0.0, x + e) - core.compute_derivs(0.0, x - e)) / (2 * h)
            for i in np.nonzero(np.abs(col) > 1e-9 * max(1.0, np.abs(col).max()))[0]:
                if (int(i), j) not in have:
                    missing.add((int(i), j))
    return sorted(missing)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_entry_a_rate_read_adds_has_a_slot(name):
    model = bngsim.Model.from_antimony_string(MODELS[name])
    names = list(model.species_names)
    assert [(names[i], names[j]) for i, j in _missing(model._core)] == []


def test_a_model_with_no_rate_read_keeps_its_pattern():
    model = bngsim.Model.from_antimony_string(CONTROL)
    sp = model._core.jacobian_sparsity
    assert _missing(model._core) == []
    assert sp["nnz"] == 2


@pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")
@pytest.mark.parametrize("name", list(MODELS))
def test_the_sparse_route_steps_as_the_dense_one_does(name):
    """The colored finite-difference Jacobian is colored from the pattern. With
    every entry in it, KLU's run is the dense run to rounding, step for step."""
    runs = []
    for kw in ({}, {"force_sparse_linear_solver": True, "jacobian": "fd"}):
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(MODELS[name]), method="ode", **kw)
        runs.append(sim.run(t_span=(0, 20), n_points=21, rtol=1e-8, atol=1e-12))
    dense, klu = runs
    assert klu.solver_stats["linear_solver"] == 1  # LINEAR_SOLVER_KLU
    x_dense, x_klu = np.asarray(dense.species), np.asarray(klu.species)
    # to rounding, against the largest value the run reaches
    assert np.abs(x_klu - x_dense).max() <= 1e-12 * max(1.0, np.abs(x_dense).max())
    assert klu.solver_stats["n_steps"] == dense.solver_stats["n_steps"]
