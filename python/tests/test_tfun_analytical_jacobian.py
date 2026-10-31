"""A table function that does not read the state is a constant to the Jacobian.

A table indexed by time, or by a constant parameter, has a value at each moment
and no dependence on any species, so ``∂f/∂y`` treats it exactly as it treats
``time()``: its current value is a coefficient and it is never differentiated.
The engine registers such a table as the zero-argument function ``tfun_<name>``,
and before this change the symbolic core did not know that name, so every rate
law that read one — directly or through a function — declined the analytical
Jacobian and the whole model fell back to finite differences (one RHS
evaluation per species per Jacobian on the dense route).

A table indexed by an observable depends on the state through its index, and
still declines. The compiled Jacobian reads a whole-body table's value from the
``func[]`` slot the function recomputation already fills; a table embedded in
arithmetic has no slot and leaves the compiled Jacobian declined, which the
interpreted analytical one then covers.

The network is small on purpose: every derivative is checked entrywise against a
central difference of the RHS at a time inside the table, and every strategy's
trajectory against ``jacobian="fd"``.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _codegen_backend import selected_codegen_backend  # noqa: E402

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

#: A table that is 1 at t=0, rises to 4 at t=5, falls to 2 at t=10 and ends at 3
#: at t=20; at t=7.5 it is 3.0, between breakpoints.
_TIME_TABLE = "tfun([0,5,10,20],[1,4,2,3],time)"

#: Each form a rate law can read a table in. ``kept_factor`` is the one whose
#: derivative keeps the table as a factor, ``F()*K/(K+Btot)^2``, so it is where
#: the symbolic core must accept the table's name and the emitted Jacobian must
#: supply its value.
FORMS = {
    # A -> B at F()*A: the derivative is the function value itself.
    "time_species_factor": ([f"F() {_TIME_TABLE}"], ["1 2 F"]),
    "time_kept_factor": ([f"F() {_TIME_TABLE}", "G() F()*Btot/(Km+Btot)"], ["1 2 F", "3 4 G"]),
    "time_embedded": ([f"G() {_TIME_TABLE}*Btot/(Km+Btot)"], ["3 4 G", "1 2 kc"]),
    "parameter_kept_factor": (
        ["F() tfun([0,1,2],[1,3,5],kp)", "G() F()*Btot/(Km+Btot)"],
        ["3 4 G", "1 2 kc"],
    ),
}
#: Indexed by an observable: ∂F/∂B is the table's slope, which is not what a
#: constant-coefficient treatment would give, so the model stays on FD.
OBSERVABLE_INDEXED = (
    ["F() tfun([0,5,10],[1,3,5],Btot)", "G() F()*Ctot/(Km+Ctot)"],
    ["3 4 G", "1 2 kc"],
)

_Y = np.array([6.0, 2.5, 7.0, 1.0])


def _net_text(functions: list[str], reactions: list[str], n_chain: int = 0) -> str:
    """A <-> B, C -> D, plus ``reactions``; ``n_chain`` extra first-order species
    X1 -> X2 -> ... pad the model past SPARSE_THRESHOLD (50) at a Jacobian density
    under 10%, so the solver — and the compiled Jacobian — take the sparse route."""
    species = ["1 A() 10", "2 B() 1", "3 C() 8", "4 D() 0"]
    rxns = ["2 1 kb", *reactions]
    for i in range(n_chain):
        species.append(f"{5 + i} X{i}() {5.0 if i == 0 else 0.0}")
        if i:
            rxns.append(f"{4 + i} {5 + i} kx")
    body = textwrap.dedent(
        """
        begin parameters
          1 kb 0.3
          2 Km 2.0
          3 kp 1.5
          4 kc 0.2
          5 kx 0.5
        end parameters
        begin species
        SPECIES
        end species
        begin functions
        FUNCS
        end functions
        begin reactions
        RXNS
        end reactions
        begin groups
          1 Btot 2
          2 Ctot 3
        end groups
        """
    ).strip()
    fill = {
        "SPECIES": [f"  {s}" for s in species],
        "FUNCS": [f"  {i} {f}" for i, f in enumerate(functions, start=1)],
        "RXNS": [f"  {i} {r}" for i, r in enumerate(rxns, start=1)],
    }
    for key, lines in fill.items():
        body = body.replace(key, "\n".join(lines))
    return body + "\n"


def _load(tmp_path: Path, functions, reactions, *, n_chain: int = 0) -> bngsim.Model:
    path = tmp_path / "tfun_jac.net"
    path.write_text(_net_text(functions, reactions, n_chain))
    return bngsim.Model.from_net(str(path))


def _central_fd(model: bngsim.Model, y: np.ndarray, t: float) -> np.ndarray:
    n = y.size
    jac = np.zeros((n, n))
    for j in range(n):
        h = 1e-6 * max(abs(y[j]), 1.0)
        up, dn = y.copy(), y.copy()
        up[j] += h
        dn[j] -= h
        jac[:, j] = (np.asarray(model.rhs(up, t)) - np.asarray(model.rhs(dn, t))) / (2 * h)
    return jac


def _trajectory(model: bngsim.Model, **kwargs):
    sim = bngsim.Simulator(model, method="ode", **kwargs)
    r = sim.run(t_span=(0, 25), n_points=26, rtol=1e-10, atol=1e-12)
    return sim, np.asarray(r.species)


@pytest.mark.parametrize("form", sorted(FORMS))
class TestStateFreeTablesAreAnalytical:
    def test_analytical_strategy_and_entrywise_derivative(self, tmp_path, form):
        """The C++ FD self-check at attach time would decline a derivative it
        cannot evaluate (an unknown ``tfun_<name>`` in the ExprTk text) silently,
        so the strategy is the assertion that the name was accepted end to end."""
        model = _load(tmp_path, *FORMS[form])
        sim = bngsim.Simulator(model, method="ode")
        assert sim.jacobian_strategy == "analytical"
        for t in (2.0, 7.5):
            jac = model.jacobian(_Y, t)
            assert jac.source == "analytical"
            np.testing.assert_allclose(jac, _central_fd(model, _Y, t), rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("codegen", [False, pytest.param(True, marks=needs_cc)])
    def test_trajectory_matches_fd(self, tmp_path, form, codegen):
        sim, analytical = _trajectory(_load(tmp_path, *FORMS[form]), codegen=codegen)
        assert sim.jacobian_strategy == "analytical"
        _, fd = _trajectory(_load(tmp_path, *FORMS[form]), codegen=codegen, jacobian="fd")
        np.testing.assert_allclose(analytical, fd, rtol=1e-6, atol=1e-9)


@needs_cc
class TestCompiledJacobianReadsTheTable:
    """``generate_jacobian_from_model`` is where a derivative that keeps the table
    as a factor needs the table's value in C: the whole-body table's ``func[]``
    slot, evaluated at ``t`` by the recomputation the Jacobian runs anyway."""

    @pytest.mark.parametrize(
        ("n_chain", "symbol"), [(0, "bngsim_codegen_jac("), (60, "bngsim_codegen_jac_sparse(")]
    )
    def test_emits_and_matches_fd(self, tmp_path, n_chain, symbol):
        from bngsim._codegen import generate_jacobian_from_model

        model = _load(tmp_path, *FORMS["time_kept_factor"], n_chain=n_chain)
        model.prepare_analytical_jacobian()
        src = generate_jacobian_from_model(model)
        assert src is not None, "the compiled Jacobian declined a time-indexed table"
        assert symbol in src
        # The slot is filled before the derivative that reads it.
        fill = src.index("func[0] = data->tfun_eval(")
        assert fill < src.index("func[0]/(")

        sim, compiled = _trajectory(
            _load(tmp_path, *FORMS["time_kept_factor"], n_chain=n_chain), codegen=True
        )
        # cc, or the codegen JIT a forced run selects (issue #1217): either
        # compiles this source, and the trajectory below is what is at stake.
        assert sim.codegen_backend == selected_codegen_backend()
        _, fd = _trajectory(
            _load(tmp_path, *FORMS["time_kept_factor"], n_chain=n_chain), jacobian="fd"
        )
        np.testing.assert_allclose(compiled, fd, rtol=1e-6, atol=1e-9)


def test_observable_indexed_table_still_declines(tmp_path):
    model = _load(tmp_path, *OBSERVABLE_INDEXED)
    sim = bngsim.Simulator(model, method="ode")
    assert sim.jacobian_strategy == "fd"
    assert "tfun_F" not in model._core.functional_jacobian_context()["constant_names"]


@needs_cc  # a sensitivity run always builds the compiled RHS (GH #214)
def test_sensitivity_derivative_is_not_given_the_table_as_a_constant(tmp_path):
    """The table's name is a constant for ``∂f/∂y`` only. A parameter-indexed
    table depends on its parameter, so ``∂f/∂p`` must not treat it as one: the
    compiled sensitivity RHS still declines it, and the forward sensitivity to
    the index parameter matches a finite difference of the trajectory."""
    functions, reactions = FORMS["parameter_kept_factor"]
    sim = bngsim.Simulator(
        _load(tmp_path, functions, reactions), method="ode", sensitivity_params=["kp"]
    )
    r = sim.run(t_span=(0, 10), n_points=11, rtol=1e-10, atol=1e-12)
    sens = np.asarray(r.sensitivities)[:, :, 0]  # (time, species) for kp

    def final(kp: float) -> np.ndarray:
        model = _load(tmp_path, functions, reactions)
        model.set_param("kp", kp)
        s = bngsim.Simulator(model, method="ode")
        return np.asarray(s.run(t_span=(0, 10), n_points=11, rtol=1e-11, atol=1e-13).species)

    # A central difference of two trajectories carries their integration error
    # divided by 2h on top of its O(h²) truncation. At h = 1e-5 that noise was
    # the size of atol on A and B, which do not depend on kp (true value 0):
    # 6.8e-7 on macOS, 6.1e-6 on Linux CI. The table is linear in kp on [1, 2],
    # so a wider step adds no truncation from the table; at 1e-3 the A/B noise
    # is ~20x smaller and C/D still match to ~1e-6 relative (1e-2 is too coarse).
    h = 1e-3
    fd = (final(1.5 + h) - final(1.5 - h)) / (2 * h)
    assert np.max(np.abs(fd)) > 1e-3  # premise: kp moves the trajectory
    np.testing.assert_allclose(sens, fd, rtol=1e-4, atol=1e-6)
