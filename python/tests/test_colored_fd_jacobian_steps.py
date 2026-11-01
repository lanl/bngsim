"""KLU's colored FD Jacobian takes each column's step on the error test's scale
(issue #1254).

A model on the KLU route without an analytic Jacobian has its Jacobian filled
by colored finite differences (``colored_fd_jacobian``), CVODE having no
difference quotient of its own for a sparse matrix. Each column's step was
``sqrt(eps)*max(|y_j|, 1)``: a floor of 1 in absolute terms, so a species of
1e-9 was moved by fifteen times itself, and the matrix KLU factored was the
Jacobian somewhere else. CVODE's own dense difference quotient, on the dense
route, takes ``max(sqrt(eps)*|y_j|, min_inc/ewt_j)``: the error weights' scale.
The colored one now does too.

* #1254's equality, ``if(Bobs==0, kb, 0)``: Y(10) = 2 ln 5 exactly. KLU with
  the FD Jacobian was 7.7e-5 to 1.5e-2 off it, 145 to 598 times dense's error
  with the same Jacobian. It now takes dense's steps and reads dense's value.
* A dimerization chain at nanomolar scale: right on KLU either way, but the
  wrong Jacobian cost 781 steps and 100 Jacobians, where dense took 299 and 40.
* The compiled right-hand side: at 256+ species CVODE's f(y) comes from codegen,
  while the columns are the interpreted RHS. Over a step of min_inc/ewt_j their
  last bits read as slope, and fceri_ji's run moved 1.6e-10 of its peak off the
  interpreted one; the quotient's base is now the interpreted f(y) as well.
* The steady-state march's colored quotient took the same absolute step: a
  nanomolar reversible chain marched 3574 steps on KLU without converging, where
  dense converged in 197. It now steps on the march's error weights too.

Found by #1257's ``klu_vs_dense`` check, which also had four BioModels fail or
read 3% off on this route (BIOMD0000000827, MODEL1006230034, MODEL2002030001,
MODEL1912090001); those now agree with dense to 2e-4 or better.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")

_EQUALITY = """begin parameters
    1 a  0.5
    2 q  2.0
    3 kb 1.0
    4 kp 1.0
end parameters
begin species
    1 A() 10
    2 B() 0
    3 Y() 0
end species
begin functions
    1 fB() kp*if(q-Aobs>0,q-Aobs,0)
    2 fY() if(Bobs==0,kb,0)
end functions
begin reactions
    1 1 0 a
    2 0 2 fB
    3 0 3 fY
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
end groups
"""

KLU_FD = {"jacobian": "fd", "force_sparse_linear_solver": True}


def _net(tmp_path, text):
    path = tmp_path / "m.net"
    path.write_text(text)
    return str(path)


def _y10(path, rtol, n_points, **kw):
    r = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", **kw).run(
        t_span=(0.0, 10.0), n_points=n_points, rtol=rtol, atol=rtol / 100
    )
    return r.species[-1][list(r.species_names).index("Y()")]


# Not rtol 1e-10 at 1001 samples, which the PR also ran: there dense with the
# analytic Jacobian is 3.4e-3 off, as far as KLU with the wrong step was, so the
# case cannot tell a wrong matrix from where the steps happened to fall.
@pytest.mark.parametrize(("rtol", "n_points"), [(1e-8, 101), (1e-8, 1001), (1e-10, 101)])
def test_the_equality_on_klu_is_as_right_as_on_dense(tmp_path, rtol, n_points):
    """B is 0 exactly until A falls below q at t* = 2 ln 5, and Y grows at kb
    while B == 0, so Y(10) = t*. KLU with the FD Jacobian was 382 to 598 times
    as far off it as dense with the FD Jacobian (7.7e-5 to 1.5e-2, where dense
    was 1.3e-7 to 3.9e-5); it now reads dense's value in all three cases.

    Not asked: that either is exact. An equality on a state has no root to
    stop the solver at, and where its switch flips depends on where the steps
    fall, on every route. The two routes now difference on one step rule but
    still factor and mend differently, and their step counts can part (202 and
    214 at rtol 1e-8 and 1001 samples, on the same Y(10)), so the bound leaves
    10x of room against that, and main still fails it by 38x or more."""
    path = _net(tmp_path, _EQUALITY)
    exact = 2 * math.log(5)
    dense = abs(_y10(path, rtol, n_points, jacobian="fd") - exact)
    klu = abs(_y10(path, rtol, n_points, **KLU_FD) - exact)
    assert klu <= max(10 * dense, 1e-9 * exact), (klu, dense)


# 2 A_i -> A_{i+1} at 1e9 /M/s from A_0 = 1 nM: O(1) dynamics in a second
_CHAIN = (
    "begin parameters\n    1 k 1e9\nend parameters\nbegin species\n"
    + "".join(f"    {i + 1} A{i}() {'1e-9' if i == 0 else '0'}\n" for i in range(6))
    + "end species\nbegin reactions\n"
    + "".join(f"    {i + 1} {i + 1},{i + 1} {i + 2} k\n" for i in range(5))
    + "end reactions\n"
)


def _run(path, **kw):
    return bngsim.Simulator(bngsim.Model.from_net(path), method="ode", **kw).run(
        t_span=(0.0, 20.0), n_points=21, rtol=1e-8, atol=1e-20
    )


def test_a_nanomolar_model_takes_the_steps_dense_takes(tmp_path):
    """The step on the error weights' scale gives the Jacobian dense's
    difference quotient gives, and Newton converges as it does there. Before,
    a species of 1e-9 was stepped by 1.5e-8: right in the end, at 2.6x the
    steps and 2.5x the Jacobians."""
    path = _net(tmp_path, _CHAIN)
    dense = _run(path, jacobian="fd")
    klu = _run(path, **KLU_FD)
    np.testing.assert_allclose(klu.species, dense.species, rtol=1e-6, atol=1e-20)
    assert klu.solver_stats["n_steps"] <= 1.2 * dense.solver_stats["n_steps"]
    assert klu.solver_stats["n_jac_evals"] <= 1.2 * dense.solver_stats["n_jac_evals"]


def test_the_chain_on_klu_with_the_closed_form_agrees_with_dense(tmp_path):
    """The fixture's control, not a guard of the step (the closed form never
    reads it): with an exact Jacobian KLU and dense agree on the chain, so what
    the test above measures is the difference quotient, not the linear solver."""
    path = _net(tmp_path, _CHAIN)
    a = _run(path, force_sparse_linear_solver=True)
    b = _run(path, jacobian="auto")
    np.testing.assert_allclose(a.species, b.species, rtol=1e-6, atol=1e-20)


_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"


def test_a_compiled_rhs_on_klu_gives_the_interpreted_runs_matrix():
    """fceri_ji, 354 species: codegen compiles its right-hand side, so CVODE's
    f(y) is the compiled one while the colored columns are the interpreted one.
    Differenced over CVODE's step, their last bits were part of the matrix: the
    compiled run took 241 steps where the interpreted one took 251, and read
    1.6e-10 of the peak away from it. With the interpreted f(y) as the base
    both runs form one matrix: 251 steps each, 4e-15 apart.

    A guard of that base under the new step, not of #1254 itself: on the old
    step, 1.5e-8 and wider, the last bits were too small a part of each
    difference to show, and main passes it (256 steps each, 2.4e-15 apart)."""
    path = _NETS_DIR / "fceri_ji.net"
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    runs = {}
    for codegen in (False, True):
        sim = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", codegen=codegen, **KLU_FD
        )
        if codegen and sim.codegen_backend == "none":
            pytest.skip("no C compiler: codegen built no right-hand side")
        runs[codegen] = np.asarray(sim.run(t_span=(0.0, 100.0), n_points=51, rtol=1e-8).species)
    peak = float(np.max(np.abs(runs[False])))
    assert np.max(np.abs(runs[True] - runs[False])) <= 1e-12 * peak


# 2 A_i <-> A_{i+1} at 1e9 /M/s and 1 /s from A_0 = 1 nM
_REVERSIBLE_CHAIN = (
    "begin parameters\n    1 kf 1e9\n    2 kr 1.0\nend parameters\nbegin species\n"
    + "".join(f"    {i + 1} A{i}() {'1e-9' if i == 0 else '0'}\n" for i in range(6))
    + "end species\nbegin reactions\n"
    + "".join(
        f"    {2 * i + 1} {i + 1},{i + 1} {i + 2} kf\n    {2 * i + 2} {i + 2} {i + 1},{i + 1} kr\n"
        for i in range(5)
    )
    + "end reactions\n"
)


def test_the_steady_state_march_on_klu_takes_the_steps_dense_takes(tmp_path):
    """The march's colored quotient had the same absolute step. On this chain
    KLU marched 3574 steps and did not converge, where dense converged in 197;
    it now steps on the march's error weights, as the dense quotient does, and
    converges where dense does."""
    path = _net(tmp_path, _REVERSIBLE_CHAIN)

    def solve(**kw):
        sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", jacobian="fd", **kw)
        return sim.steady_state(tol=1e-9, max_time=1e6)

    dense = solve(force_dense_linear_solver=True)
    klu = solve(force_sparse_linear_solver=True)
    assert dense.converged and klu.converged, (dense.n_steps, klu.n_steps)
    assert klu.linear_solver == "klu"
    assert klu.n_steps <= 1.2 * dense.n_steps, (klu.n_steps, dense.n_steps)
    scale = float(np.max(np.abs(dense.concentrations)))
    np.testing.assert_allclose(
        klu.concentrations, dense.concentrations, rtol=1e-6, atol=1e-7 * scale
    )
