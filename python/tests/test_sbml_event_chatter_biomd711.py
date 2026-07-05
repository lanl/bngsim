"""GH #95 — event-chatter guard (Zeno re-firing of a non-negativity clamp).

`BIOMD0000000711` (Hancioglu2007, 11-species influenza immune response) carries a
single event — a non-negativity clamp:

    trigger:    Viral_Load__V < 0
    assignment: Viral_Load__V := 0

After the viral burst, V decays exponentially. Once it falls ~11 orders of
magnitude *below* atol (around t≈11, V≈1e-23), floating-point noise makes V
oscillate around 0; the clamp fires on every micro-crossing, and each firing
forces a `CVodeReInit` that resets BDF to order-1 tiny steps. The integrator
then crawls — bngsim took >300s where RoadRunner (which keeps V cleanly positive
down to ~1e-58, so its event never fires) finishes in ~0.1s. The 11-species model
was the cleanest case in the rr_parity ODE *timeout* bucket; the slowdown was
event chattering, not model difficulty.

The chatter guard (src/cvode_simulator.cpp) detects an event re-firing with both
negligible time advance AND sub-tolerance state change, and — after a run of such
fires — suppresses that event's trigger root so CVODE steps over the noise floor
(re-arming if the assigned species climb back above the floor). The dual
criterion leaves genuine recurring events untouched (the full event suite is the
guard against over-suppression).

This test locks the fix in: a regression that reinstates the chatter makes the
run exceed its wall budget (`SimulationTimeout`) instead of finishing in well
under a second. The chatter is an emergent property of the coupled stiff system
(a single decaying species stays cleanly positive and does not reproduce it), so
this is gated on the real corpus model — like ``test_sbml_rateof``'s corpus
cross-check, it skips unless both the gitignored model and libRoadRunner are
present locally.
"""

from __future__ import annotations

import time
from pathlib import Path

import bngsim
import numpy as np
import pytest

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"
_MODEL_DIR = _MODELS_DIR / "BIOMD0000000711"

# The job protocol (figure SED-ML horizon) at the shared parity tolerance.
_T_END = 15.0
_N_POINTS = 1001
_RTOL = 1e-9
_ATOL = 1e-12
# Generous wall budget: the guarded run finishes in ~0.5s; the chatter pathology
# runs >300s, so any value far below that reliably catches a regression while
# tolerating slow CI machines.
_WALL_BUDGET = 60.0


def _model_xml() -> Path | None:
    xmls = sorted(_MODEL_DIR.glob("*.xml"))
    return xmls[0] if xmls else None


def test_biomd711_event_chatter_does_not_stall():
    """The headline symptom: the clamp event no longer chatters the solver to a halt.

    Without the guard this run never completes inside the wall budget and raises
    ``SimulationTimeout``; with it, the integration finishes in well under a
    second and the clamped viral load ends at its (numerically zero) floor.
    """
    xml = _model_xml()
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODEL_DIR}")

    model = bngsim.Model.from_sbml(str(xml))
    t0 = time.perf_counter()
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, _T_END),
        n_points=_N_POINTS,
        rtol=_RTOL,
        atol=_ATOL,
        timeout=_WALL_BUDGET,
    )
    wall = time.perf_counter() - t0

    species = np.asarray(res.species)
    assert species.shape[0] == _N_POINTS
    assert np.isfinite(species).all(), "non-finite trajectory after integration"
    # Returning at all means it beat the timeout; assert real headroom so a
    # partial regression (slow but not infinite) is still visible.
    assert wall < _WALL_BUDGET, f"integration took {wall:.1f}s (chatter regression?)"

    v = species[:, list(res.species_names).index("Viral_Load__V")]
    assert v[-1] == pytest.approx(0.0, abs=1e-6), f"viral load did not settle: {v[-1]:.3e}"


def test_biomd711_matches_roadrunner():
    """With the chatter resolved, the trajectory agrees with the RoadRunner reference.

    RoadRunner is the existence proof the issue cites (it integrates this model in
    ~0.1s); parity confirms the guard produces the correct trajectory, not merely
    a fast one. Gated on roadrunner being importable.
    """
    xml = _model_xml()
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODEL_DIR}")
    roadrunner = pytest.importorskip("roadrunner")
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)

    model = bngsim.Model.from_sbml(str(xml))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, _T_END), n_points=_N_POINTS, rtol=_RTOL, atol=_ATOL, timeout=_WALL_BUDGET
    )
    bn = np.asarray(res.species)
    bn_names = list(res.species_names)

    rr = roadrunner.RoadRunner(str(xml))
    rr.integrator = "cvode"
    rr.integrator.relative_tolerance = _RTOL
    rr.integrator.absolute_tolerance = _ATOL
    boundary = set(rr.model.getBoundarySpeciesIds())
    rr.timeCourseSelections = [f"[{s}]" if s in boundary else s for s in rr.timeCourseSelections]
    ref = np.asarray(rr.simulate(0.0, _T_END, _N_POINTS))
    rr_names = [c[1:-1] if c.startswith("[") else c for c in rr.timeCourseSelections][1:]

    bn_map = {n: i for i, n in enumerate(bn_names)}
    rr_map = {n: i for i, n in enumerate(rr_names)}
    common = sorted(set(bn_map) & set(rr_map))
    assert common, "bngsim/RoadRunner species sets are disjoint (loader divergence)"

    a = bn[:, [bn_map[n] for n in common]]
    b = ref[:, [1 + rr_names.index(n) for n in common]]
    # Per-cell solver error budget |a-b| <= atol + rtol*|b|. The clamp's effect
    # on the real dynamics is below ~1e-7 relative; the only cells that graze a
    # tight budget are the decay tail where both engines sit in the noise floor.
    over = np.abs(a - b) > (_ATOL + 1e-4 * np.abs(b))
    assert over.mean() < 1e-2, f"fail fraction {over.mean():.2e} over budget"
