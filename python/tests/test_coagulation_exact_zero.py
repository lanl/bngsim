"""Wajima 2009's coagulation models stay on their exact, unstable 0 (issue #1304).

BIOMD0000000338 (the aPTT test) and BIOMD0000000339 (the PT test) start XIIa,
kallikrein (K) and the contact activator (CA) at 0. The right-hand sides of all
three are exactly 0 wherever all three are 0, so XIIa = K = CA = 0 for all t is
the exact solution. It is unstable: the XIIa <-> K loop grows any departure.

KLU keeps it exactly. Until #1485 the dense LU's rounding left it, which way and
when depending on the CPU and the output grid (#1304's measurements): on macOS
arm64, 338's dense run clotted (Fg(2) = 0.73 against KLU's 311) and 339's read
1.8% of the peak from KLU's; on Linux, 338's failed at t = 0.72; on macOS
x86-64, it was refused at t = 0.069, with XIIa carried to r4's pole at -1.
libRoadRunner leaves the 0 too, by its own rounding: upward, XIIa spikes and the
cascade clots; downward, XIIa passes the pole and its step collapses. So the
rr_parity overrides for both models name RoadRunner, not bngsim, as the engine
that diverges, and the second test below is their evidence.

Both tests are controls: they pass on main, where #1485 is. With #1485's zeroing
of the rows at 0 switched off (macOS arm64), every dense case fails, and the
default runs, on KLU, pass.

Oracles:

- the property itself: XIIa, K and CA are exactly 0 on every row, on either
  linear solver and every Jacobian;
- for every other species, RoadRunner on the same SBML with XIIa, K and CA made
  constant at 0. That problem has no unstable 0 to leave, and its solution is
  the model's.

The models are BioModels' (CC0), copied from the rr_parity corpus, so CI reads
them. 338's dense runs are also read in test_clamp_far_below_zero.py.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
MODELS = {m: DATA / f"{m}.xml" for m in ("BIOMD0000000338", "BIOMD0000000339")}
# Each model's rr_parity job: its window, grid and tolerances.
RUN = {"t_span": (0.0, 2.0), "n_points": 101, "rtol": 1e-6, "atol": 1e-9}
ZERO = ("XIIa", "K", "CA")
# Of each species' peak. The routes agree within 7.5e-6 of it, and RoadRunner on
# the pinned copy within 1.6e-5 (macOS arm64). A run that leaves the 0 is off by
# about 1: RoadRunner's run of the model as written is, on 338.
BOUND = 1e-4

_needs_klu = pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")


def _run(model_id: str, **simulator):
    model = bngsim.Model.from_sbml(str(MODELS[model_id]))
    result = bngsim.Simulator(model, method="ode", **simulator).run(**RUN)
    return (
        np.asarray(result.time),
        np.asarray(result.species, dtype=float),
        list(result.species_names),
    )


def _gap(x: np.ndarray, ref: np.ndarray) -> float:
    """The largest difference, as a share of each species' peak in ``ref``."""
    return float((np.abs(x - ref) / (np.abs(ref).max(axis=0) + RUN["atol"])).max())


@_needs_klu
@pytest.mark.parametrize("jacobian", ["fd", "auto", "analytical"])
@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_dense_keeps_the_exact_zero_and_agrees_with_klu(model_id, jacobian):
    """Control. With #1485's zeroing switched off (arm64), 338's dense run is
    refused at t = 0.08, XIIa carried to r4's pole, on every Jacobian; 339's
    spikes XIIa to 72.7 (fd and auto: Fg(2) = 0.611038 against KLU's 0.611371)
    or is refused at t = 0.06 (analytical)."""
    _, klu, names = _run(model_id, jacobian=jacobian, force_sparse_linear_solver=True)
    _, dense, dense_names = _run(model_id, jacobian=jacobian, force_dense_linear_solver=True)
    assert dense_names == names
    for name in ZERO:
        np.testing.assert_array_equal(klu[:, names.index(name)], 0.0, err_msg=f"KLU {name}")
        np.testing.assert_array_equal(dense[:, names.index(name)], 0.0, err_msg=f"dense {name}")
    assert _gap(dense, klu) <= BOUND


@pytest.mark.parametrize(
    "solver",
    [{}, {"force_dense_linear_solver": True}],
    ids=["default", "dense"],
)
@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_roadrunner_with_the_zero_held_agrees(model_id, solver):
    """Control. RoadRunner cannot leave a 0 that is held constant, so on the
    pinned copy it solves the model's own problem. bngsim's run of the model as
    written agrees with it on every species they share. With #1485's zeroing
    switched off (arm64), both dense cases fail, as above."""
    roadrunner = pytest.importorskip("roadrunner")
    import libsbml  # a base dependency

    doc = libsbml.readSBMLFromFile(str(MODELS[model_id]))
    for name in ZERO:
        species = doc.getModel().getSpecies(name)
        assert species.getInitialConcentration() == 0.0, name
        species.setBoundaryCondition(True)
        species.setConstant(True)
    rr = roadrunner.RoadRunner(libsbml.writeSBMLToString(doc))
    rr.integrator.relative_tolerance = RUN["rtol"]
    rr.integrator.absolute_tolerance = RUN["atol"]

    t, x, names = _run(model_id, **solver)
    ids = set(rr.model.getFloatingSpeciesIds()) | set(rr.model.getBoundarySpeciesIds())
    shared = [name for name in names if name in ids]
    assert len(shared) >= 50 and set(ZERO) <= set(shared), shared
    rr.timeCourseSelections = ["time"] + [f"[{name}]" for name in shared]
    ref = np.asarray(rr.simulate(RUN["t_span"][0], RUN["t_span"][1], RUN["n_points"]))
    np.testing.assert_allclose(ref[:, 0], t, rtol=0, atol=1e-12)
    for name in ZERO:
        np.testing.assert_array_equal(x[:, names.index(name)], 0.0, err_msg=name)
    assert _gap(x[:, [names.index(name) for name in shared]], ref[:, 1:]) <= BOUND
