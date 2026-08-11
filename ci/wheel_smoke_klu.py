"""Wheel smoke test: the KLU vendored into this wheel actually factorizes.

Issue #303. The cibuildwheel ``test-command`` used to be

    python -c "import bngsim; print(bngsim.__version__); assert bngsim.HAS_KLU"

and ``HAS_KLU`` is a **compile-time** flag: it is True because the extension was
*built* against SuiteSparse, and it stays True whether or not the libraries that
ended up inside the wheel can do anything. That gap matters here more than it
would for an ordinary dependency, because every wheel is *repaired* and repair
rewrites the linkage — delvewheel copies the DLLs into a mangled directory,
content-hashes their names and injects a loader patch; delocate rewrites install
names to ``@loader_path``; auditwheel mangles SONAMEs and patches RPATH. None of
that is exercised by a source build, so ``windows-klu.yml`` (issue #296) does not
cover it: that leg tests a ``pip install .`` with the DLLs sitting where the
build left them.

What this asserts instead is that the sparse solve *computes the right answer*,
cross-checked against a factorization that does not touch SuiteSparse at all.

Deliberately self-contained. It builds its model through ``ModelBuilder`` rather
than reading a fixture, so it needs no repo data files; it needs no C compiler
(codegen is not involved — the interpreted sparse path drives the same
``klu_factor``/``klu_solve`` the vendored library provides); and it imports
nothing beyond bngsim and numpy, both of which any working wheel already has.
That is what lets it run as the ``test-command`` on all four legs and all four
CPythons without a ``test-requires``.

Run: ``python ci/wheel_smoke_klu.py`` (exits non-zero, loudly, on any failure).
"""

from __future__ import annotations

import sys

import bngsim
import numpy as np
from bngsim._bngsim_core import ModelBuilder

#: ``LinearSolverKind`` (include/bngsim/result.hpp), as reported by
#: ``Result.solver_stats["linear_solver"]``. Mirrored as literals the same way
#: python/tests/test_force_sparse_linear_solver.py mirrors them.
#:   0 = built-in dense LU, 1 = KLU sparse, 2 = BLAS dgetrf dense
LS_KLU = 1

#: Past ``SPARSE_THRESHOLD`` (50 species) and under ``SPARSE_DENSITY_MAX`` (10%),
#: so ``choose_linear_solver_kind`` routes it to KLU with no forcing flag. A
#: first-order chain of this length lands at density ~2/N.
N_CHAIN = 60
INITIAL_AMOUNT = 100.0


def _chain(n: int):
    """``S0 -> S1 -> ... -> S(n-1)``, first order, no drain.

    No drain is the point: total mass is conserved exactly, which gives an
    absolute invariant to check alongside the dense/sparse cross-check. A
    factorization that returns finite nonsense fails that; one that returns the
    same nonsense as the dense path could not, which is why both are here.
    """
    builder = ModelBuilder()
    species = []
    for i in range(n):
        builder.add_parameter(f"k{i}", 0.1 * (i + 1))
        species.append(builder.add_species(f"S{i}", INITIAL_AMOUNT if i == 0 else 0.0))
    for i in range(n - 1):
        builder.add_reaction([species[i]], [species[i + 1]], "elementary", f"k{i}")
    return builder.build()


def _run(core, **kwargs):
    sim = bngsim.Simulator(bngsim.Model(_core=core), method="ode", **kwargs)
    return sim.run(t_span=(0.0, 20.0), n_points=21, rtol=1e-10, atol=1e-12)


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    print(f"bngsim {bngsim.__version__} on {sys.version}")
    print(f"HAS_KLU {bngsim.HAS_KLU}")
    if not bngsim.HAS_KLU:
        _fail("HAS_KLU is False — this wheel was not built against SuiteSparse/KLU")

    # Guard the premise before trusting the result. If the auto-routing
    # thresholds move, this model could drift out of the KLU band and every
    # assertion below would still pass while testing the dense solver twice.
    probe = _chain(N_CHAIN)
    density = float(probe.codegen_jacobian_plan()["density"])
    print(f"model: n_species={probe.n_species} density={density:.4f}")
    if probe.n_species < 50 or density >= 0.10:
        _fail(
            f"fixture no longer sits in the auto-KLU band (n={probe.n_species}, "
            f"density={density:.4f}); it would silently route to the dense solver"
        )

    # A FRESH model per run, never a shared one. ``run()`` advances the core's
    # species concentrations, so a second run against the same object starts
    # from the first one's final state -- the control arm silently stops being a
    # control. Caught here by the cross-check below reporting a 100% mismatch
    # whose first row was not the initial condition;
    # test_force_sparse_linear_solver.py builds a fresh chain per run for the
    # same reason. Do not hoist this.
    sparse = _run(_chain(N_CHAIN))
    kind = sparse.solver_stats["linear_solver"]
    print(f"auto-routed linear_solver={kind} (KLU is {LS_KLU})")
    if kind != LS_KLU:
        _fail(
            f"the sparse-band model did not route to KLU (linear_solver={kind}). "
            "The wheel reports HAS_KLU but the solver did not use it."
        )

    # The oracle: the built-in dense LU factorizes the same problem and touches
    # no SuiteSparse code. Agreement to solver tolerance is what says the
    # vendored KLU computed rather than merely loaded.
    dense = _run(_chain(N_CHAIN), force_dense_linear_solver=True)
    if dense.solver_stats["linear_solver"] == LS_KLU:
        _fail("force_dense_linear_solver did not leave the KLU path — no independent oracle")

    if not np.all(np.isfinite(sparse.species)):
        _fail("the KLU trajectory contains non-finite values")

    try:
        np.testing.assert_allclose(sparse.species, dense.species, rtol=1e-8, atol=1e-10)
    except AssertionError as exc:
        _fail(f"KLU and dense factorizations disagree:\n{exc}")

    # Absolute invariant, independent of both solvers: the chain has no drain.
    mass = np.asarray(sparse.species, dtype=float).sum(axis=1)
    drift = float(np.abs(mass - INITIAL_AMOUNT).max())
    print(f"mass conservation: worst drift {drift:.3e}")
    if drift > 1e-6:
        _fail(f"KLU trajectory does not conserve mass (worst drift {drift:.3e})")

    print("OK: vendored SuiteSparse/KLU factorizes and agrees with the dense solver")


if __name__ == "__main__":
    main()
