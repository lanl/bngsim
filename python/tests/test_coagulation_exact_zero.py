"""Wajima 2009's coagulation models stay on their exact, unstable 0 (issue #1304).

BIOMD0000000338 (the aPTT test) and BIOMD0000000339 (the PT test) start XIIa,
kallikrein (K) and the contact activator (CA) at 0. Each of the six reactions
that change them is proportional to one of the three, and no rule, initial
assignment or event sets them, so XIIa = K = CA = 0 for all t is the exact
solution. It is unstable: the XIIa <-> K loop grows any departure.

KLU keeps it exactly. Until #1485 the dense LU's rounding left it, which way and
when depending on the CPU and the output grid (#1304's measurements): on macOS
arm64, 338's dense run clotted (Fg(2) = 0.73 against KLU's 311) and 339's read
1.8% of the peak from KLU's; on Linux, 338's failed at t = 0.72; on macOS
x86-64, it was refused at t = 0.069, with XIIa carried to r4's pole at -1.
libRoadRunner leaves the 0 too, by its own rounding: upward, XIIa spikes and the
cascade clots; downward, XIIa passes the pole and its step collapses. So the
rr_parity overrides for both models name RoadRunner, not bngsim, as the engine
that diverges, and this test is their evidence. That dense keeps the 0 and
agrees with KLU is test_differential_shapes.py's #1304 test, on the same files.

Oracle: RoadRunner on the same SBML with XIIa, K and CA made constant at 0. That
problem has no unstable 0 to leave, and its solution is the model's.

The models are BioModels' (CC0), copied from the rr_parity corpus, so CI reads
them.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
MODELS = {m: DATA / f"{m}.xml" for m in ("BIOMD0000000338", "BIOMD0000000339")}
# Each model's rr_parity window and grid, at the tolerances rr_run forces on both
# engines (its manifest's rtol 1e-6 is SED-ML provenance only).
RUN = {"t_span": (0.0, 2.0), "n_points": 101, "rtol": 1e-9, "atol": 1e-12}
ZERO = ("XIIa", "K", "CA")
# Of each species' peak after t = 0: the dilution event at t > 0 thins every
# species threefold, and some are consumed before t = 0.02, so a peak that read
# the initial row would loosen the bound for them (338's VIII is 0.7 at t = 0 and
# at most 9.2e-3 after it). Measured on macOS arm64: 1.03e-6 (338, VIII) and
# 4.7e-7 (339, V). A run that leaves the 0 is off by about 1.
BOUND = 1e-5


@pytest.mark.parametrize(
    "solver",
    [{}, {"force_dense_linear_solver": True}],
    ids=["default", "dense"],
)
@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_roadrunner_with_the_zero_held_agrees(model_id, solver):
    """Control: passes on main, where #1485 is. RoadRunner cannot leave a 0 that
    is held constant, so on the pinned copy it solves the model's own problem,
    and bngsim's run of the model as written agrees with it on every species
    they share. With #1485's zeroing of the rows at 0 switched off (arm64), both
    dense cases fail: the dense runs spike XIIa to 74 (338, which then clots:
    Fg(2) = 0.738 against 310.989) and 71 (339). The default runs, on KLU, pass
    either way."""
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

    model = bngsim.Model.from_sbml(str(MODELS[model_id]))
    result = bngsim.Simulator(model, method="ode", **solver).run(**RUN)
    t, x, names = np.asarray(result.time), np.asarray(result.species), list(result.species_names)
    for name in ZERO:
        np.testing.assert_array_equal(x[:, names.index(name)], 0.0, err_msg=name)

    ids = set(rr.model.getFloatingSpeciesIds()) | set(rr.model.getBoundarySpeciesIds())
    shared = [name for name in names if name in ids]
    assert len(shared) >= 50 and set(ZERO) <= set(shared), shared
    rr.timeCourseSelections = ["time"] + [f"[{name}]" for name in shared]
    ref = np.asarray(rr.simulate(RUN["t_span"][0], RUN["t_span"][1], RUN["n_points"]))
    np.testing.assert_allclose(ref[:, 0], t, rtol=0, atol=1e-12)
    ours, theirs = x[1:, [names.index(name) for name in shared]], ref[1:, 1:]
    gap = np.abs(ours - theirs) / (np.abs(theirs).max(axis=0) + RUN["atol"])
    worst = np.unravel_index(gap.argmax(), gap.shape)
    assert gap.max() <= BOUND, (shared[worst[1]], t[1 + worst[0]], gap.max())
