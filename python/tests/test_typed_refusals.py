"""Two refusals bngsim makes by name, each of its own type (issue #1313).

``force_sparse_linear_solver`` on a model with no sparse Jacobian to hand KLU,
and ``codegen=True`` on a model codegen declines, were a ``SimulationError``
and a ``RuntimeError`` told apart from a crash only by their text: a caller
that had to (the path-agreement harness's ``_sparse_declined`` and
``_compiled``) matched the message, and a reworded message turned a refusal
into a crash. Each subclasses what was raised before, so an existing
``except`` still catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bngsim
import pytest
from bngsim._bngsim_core import ModelBuilder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_root import bngsim_source_root  # noqa: E402

_NO_REACTIONS = (
    "begin parameters\n 1 k 1.0\nend parameters\n"
    "begin species\n 1 A() 1\n 2 B() 2\nend species\n"
    "begin reactions\nend reactions\n"
)


def _no_reactions(tmp_path) -> bngsim.Model:
    """Species and no reactions: a structurally empty Jacobian, with no column to
    perturb and nothing for KLU to factorize."""
    path = tmp_path / "no_reactions.net"
    path.write_text(_NO_REACTIONS)
    return bngsim.Model.from_net(str(path))


def _cycle_model():
    """Two functions that read each other: codegen declines the model (#621)."""
    b = ModelBuilder()
    b.add_parameter("k", 1.0, "", False)
    b.add_species("S", 1.0, False, 1.0)
    b.add_function("f", "0.5*g + 1")
    b.add_function("g", "0.5*f + 1")
    b.add_reaction([0], [], "functional", "f", 1.0, True)
    return bngsim.Model(_core=b.build())


def _sparse(model, **kw):
    return bngsim.Simulator(
        model, method="ode", force_sparse_linear_solver=True, jacobian="fd", **kw
    )


def test_both_are_exported_and_subclass_what_was_raised():
    assert issubclass(bngsim.SparseJacobianUnavailableError, bngsim.SimulationError)
    assert issubclass(bngsim.CodegenDeclinedError, bngsim.BngsimError)
    assert issubclass(bngsim.CodegenDeclinedError, RuntimeError)
    assert not issubclass(bngsim.CodegenDeclinedError, bngsim.SimulationError)
    assert {"SparseJacobianUnavailableError", "CodegenDeclinedError"} <= set(bngsim.__all__)


def test_no_sparse_jacobian_for_klu_is_its_own_type(tmp_path):
    if not bngsim._bngsim_core.HAS_KLU:
        pytest.skip("KLU not compiled in")
    with pytest.raises(bngsim.SparseJacobianUnavailableError) as refused:
        _sparse(_no_reactions(tmp_path)).run(t_span=(0, 1), n_points=3)
    assert "cannot supply a sparse Jacobian" in str(refused.value)
    # what was raised before still catches it
    assert isinstance(refused.value, bngsim.SimulationError)


def test_no_sparse_jacobian_reaches_a_batch_by_its_type(tmp_path):
    """run_batch's threads wrap a point's failure as "Batch simulation i
    failed": a refusal by name is the model's, whichever point meets it."""
    if not bngsim._bngsim_core.HAS_KLU:
        pytest.skip("KLU not compiled in")
    with pytest.raises(bngsim.SparseJacobianUnavailableError):
        _sparse(_no_reactions(tmp_path)).run_batch(
            t_span=(0, 1), n_points=3, params=[{"k": 1.0}, {"k": 2.0}], num_processors=2
        )


def test_a_solver_failure_is_not_the_refusal(tmp_path):
    """A run that fails for any other reason is still a plain SimulationError."""
    path = tmp_path / "blowup.net"
    path.write_text(
        "begin parameters\n 1 k 1.0\nend parameters\n"
        "begin species\n 1 A() 1\nend species\n"
        "begin reactions\n 1 1,1 1,1,1 k\nend reactions\n"
    )
    with pytest.raises(bngsim.SimulationError) as failed:
        bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
            t_span=(0, 10), n_points=3
        )
    assert not isinstance(failed.value, bngsim.SparseJacobianUnavailableError)


def test_codegen_declining_a_model_is_its_own_type():
    with pytest.raises(bngsim.CodegenDeclinedError) as refused:
        bngsim.Simulator(_cycle_model(), method="ode", codegen=True)
    assert "declined this model" in str(refused.value)
    assert isinstance(refused.value, RuntimeError)


def test_a_codegen_build_failure_is_not_a_decline(tmp_path, monkeypatch):
    """A build that failed is no statement about the model: a plain
    RuntimeError, never CodegenDeclinedError."""
    import bngsim._codegen as cg

    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated build failure")

    monkeypatch.setattr(cg, "generate_combined_from_model", fail)
    path = tmp_path / "decay.net"
    path.write_text(
        "begin parameters\n 1 k 1.0\nend parameters\n"
        "begin species\n 1 A() 1\nend species\n"
        "begin reactions\n 1 1 0 k\nend reactions\n"
    )
    with pytest.raises(RuntimeError, match="simulated build failure") as failed:
        bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=True)
    assert not isinstance(failed.value, bngsim.CodegenDeclinedError)


def _harness():
    root = bngsim_source_root() or Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "parity_checks" / "differential"))
    import _differential as D

    return D


def test_the_harness_reads_both_by_type_not_by_text(tmp_path, monkeypatch):
    """The harness's SKIP / UNSUPPORTED for these refusals no longer hangs on the
    wording: reworded, each is still read as the refusal it is."""
    D = _harness()
    reworded = bngsim.SparseJacobianUnavailableError("KLU has nothing to factorize here")
    assert D._sparse_declined(reworded)
    assert not D._sparse_declined(bngsim.SimulationError("CVODE integration failed"))

    def declines(job, **kw):
        raise bngsim.CodegenDeclinedError("this model is not for compiling")

    monkeypatch.setattr(D, "run", declines)
    job = D.Job("m", tmp_path / "m.net")
    with pytest.raises(D._Declined):
        D._compiled(job)


def test_a_batch_row_takes_the_linear_solver_the_simulator_was_asked_for(tmp_path):
    """A batch row dropped force_dense_linear_solver and force_sparse_linear_solver
    and ran whatever the auto rule picked: on a 60-species chain, KLU, under a
    Simulator asked for the dense solver."""
    if not bngsim._bngsim_core.HAS_KLU:
        pytest.skip("KLU not compiled in")
    n = 60
    path = tmp_path / "chain.net"
    path.write_text(
        "begin parameters\n 1 k 1.0\nend parameters\nbegin species\n"
        + "".join(f" {i + 1} S{i}() {1.0 if i == 0 else 0.0}\n" for i in range(n))
        + "end species\nbegin reactions\n"
        + "".join(f" {i + 1} {i + 1} {i + 2} k\n" for i in range(n - 1))
        + "end reactions\n"
    )

    def kinds(**flags):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", **flags)
        rows = sim.run_batch(t_span=(0, 1), n_points=3, params=[{"k": 1.0}, {"k": 2.0}])
        return {r.solver_stats["linear_solver"] for r in rows}

    assert kinds() == {1}  # the auto rule: KLU
    assert kinds(force_dense_linear_solver=True) == {0}
