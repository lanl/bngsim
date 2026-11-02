"""A structurally empty Jacobian is not sent to KLU by the automatic routing.

A model of 50 or more species whose Jacobian has no structural nonzero (no
reaction moves any species; an event assigns them) passed the size and density
test of ``route_to_sparse_linear_solver`` with a density of 0, and went to KLU.
KLU then has nothing to factor, and under ``jacobian="fd"`` nothing to color,
and the run was refused with "force_sparse_linear_solver: this model cannot
supply a sparse Jacobian ... Drop the flag": a flag the caller never set (SBML
Test Suite 01530-01533, 01564; found by #1257's ``ajac_vs_fdjac`` check).

The routing now sends such a model to the dense solver, which differences the
full matrix. Forcing KLU still asks for it, and is still refused by name. The
codegen emitter reads the same rule (``routes_sparse``), and its fallback for an
older core already required a nonzero: the two now agree.
"""

from __future__ import annotations

import shutil

import bngsim
import pytest

antimony = pytest.importorskip("antimony")

N = 60  # past SPARSE_THRESHOLD (50)

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")


def _sbml(body):
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(f"model m\n{body}\nend") >= 0, antimony.getLastError()
    return antimony.getSBMLString("m")


@pytest.fixture(scope="module")
def empty():
    """N species no reaction moves, each assigned S_i + 1 by an event at t = 5."""
    return _sbml(
        "".join(f"species S{i} = {i}; " for i in range(N))
        + "at (time >= 5): "
        + ", ".join(f"S{i} = S{i} + 1" for i in range(N))
        + ";"
    )


@pytest.fixture(scope="module")
def diagonal():
    """N species each decaying alone: a diagonal pattern, density 1/N, sparse."""
    return _sbml("".join(f"species A{i} = 1; J{i}: A{i} -> ; 0.1*A{i}; " for i in range(N)))


def _model(sbml):
    return bngsim.Model.from_sbml_string(sbml)


def _sim(sbml, **kw):
    return bngsim.Simulator(_model(sbml), method="ode", **kw)


def _run(sim):
    return sim.run(t_span=(0.0, 10.0), n_points=11, rtol=1e-8, atol=1e-10)


def _assert_the_closed_form(result):
    """S_i holds at i until the event at t = 5, then at i + 1. Exact, so ==: f is
    identically 0, so the solver never moves the state, and the event adds 1 to an
    integer."""
    x = result.species
    for i in range(N):
        assert x[4][i] == i  # t = 4
        assert x[-1][i] == i + 1  # t = 10


def _assert_not_klu(result):
    """The route the run took, which is what the fix changes: 1 is KLU."""
    assert result.solver_stats["linear_solver"] != 1


def test_the_empty_pattern_is_empty(empty):
    sp = _model(empty)._core.jacobian_sparsity
    assert sp["n"] == N and sp["nnz"] == 0


@pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")
def test_an_empty_jacobian_routes_dense(empty):
    assert _model(empty)._core.codegen_jacobian_plan()["routes_sparse"] is False


@pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")
def test_a_sparse_jacobian_still_routes_to_klu(diagonal):
    """The control: a diagonal pattern past the threshold is what KLU is for."""
    assert _model(diagonal)._core.codegen_jacobian_plan()["routes_sparse"] is True


@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_the_run_is_the_closed_form(empty, jacobian):
    """Before, the run went to KLU, and with jacobian="fd" it was refused."""
    result = _run(_sim(empty, jacobian=jacobian))
    _assert_not_klu(result)
    _assert_the_closed_form(result)


@needs_cc
@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_the_codegen_run_is_the_closed_form(empty, jacobian):
    """Codegen reads the same rule, through codegen_jacobian_plan's routes_sparse.
    Before, its run went to KLU too, and with jacobian="fd" it was refused."""
    sim = _sim(empty, codegen=True, jacobian=jacobian)
    assert sim.codegen_backend in ("cc", "mir")
    result = _run(sim)
    _assert_not_klu(result)
    _assert_the_closed_form(result)


@pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")
def test_forcing_klu_is_still_refused_by_name(empty):
    """The caller who asks for KLU on an empty pattern gets the refusal, and
    its advice to drop the flag is now the caller's own flag."""
    refusal = "force_sparse_linear_solver: this model cannot"
    with pytest.raises(bngsim.SimulationError, match=refusal):
        _run(_sim(empty, jacobian="fd", force_sparse_linear_solver=True))
