"""GH #93 — BIOMD0000000451 ODE integrability (CVODE convergence robustness).

The rr_parity auto-derived taxonomy once flagged BIOMD0000000451 (Carbo2013,
"Cytokine driven CD4+ T cell differentiation and phenotype plasticity"; 94
species, 2 compartments, 16 COPASI Hill-type functionDefinitions, 2 assignment
rules, no events) as the lone ODE-corpus EXCEPTION — "bngsim raised, RoadRunner
ran with a finite reference":

    SimulationError: CVODE integration failed at t=0.010000 with flag=-4

flag=-4 is ``CV_CONV_FAILURE`` (the nonlinear solver's convergence test failed
repeatedly). It surfaced at t=0.01 of a 0->10 horizon while libRoadRunner — also
the SUNDIALS CVODE family at the same tolerance — integrated the same SBML to
completion with 100% finite output, so the IVP is integrable and the divergence
was bngsim-side (RHS / functionDefinition handling / initial-step selection),
not an intrinsically unintegrable problem.

The EXCEPTION reflected a pre-fix corpus-survey state; the SBML loader and CVODE
hot-path rework that landed afterward resolved it. bngsim now integrates this
model to full RoadRunner parity. This test locks that in: a regression that
reintroduces the early-step convergence failure makes bngsim raise here again.

The rr_parity BioModels corpus is gitignored (the ~270 MB keep-set is never
committed), so — like ``test_sbml_rateof``'s corpus cross-check — this skips
unless both the model file and libRoadRunner are present locally.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"
_MODEL = _MODELS_DIR / "BIOMD0000000451" / "BIOMD0000000451_url.xml"

# The issue's exact repro: shared parity integration tolerance over the full
# horizon. flag=-4 struck at t=0.01, so any sample beyond the first proves the
# integrator cleared the early steps it used to fail on.
_T_END = 10.0
_N_POINTS = 1001
_RTOL = 1e-9
_ATOL = 1e-12


def _rr_concentrations(roadrunner, path: Path):
    """RoadRunner ODE run with every species forced to concentration ``[id]``.

    Mirrors rr_parity's adapter: floating species already report ``[X]``;
    rewriting boundary species (which default to *amount*) to ``[id]`` keeps the
    comparison concentration-vs-concentration, as bngsim always reports.
    """
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)
    rr = roadrunner.RoadRunner(str(path))
    rr.integrator = "cvode"
    rr.integrator.relative_tolerance = _RTOL
    rr.integrator.absolute_tolerance = _ATOL
    # At atol=1e-12 on this stiff 16-Hill-function model, CVODE's default
    # initial-step heuristic picks h~1e-13 at t=0; the corrector then can't
    # converge and the oracle bottoms out at hmin with CV_CONV_FAILURE
    # (deterministic, at t=1.96e-07). It is purely a step-zero pathology — the
    # IVP is integrable at this tolerance once the first step is sane (bngsim
    # clears it unaided). Seed RoadRunner with a reasonable initial step (and a
    # generous step cap) so the reference completes; the parity tolerance is
    # unchanged.
    rr.integrator.setValue("initial_time_step", 1e-6)
    rr.integrator.setValue("maximum_num_steps", 200_000)
    boundary = set(rr.model.getBoundarySpeciesIds())
    sel = [f"[{s}]" if s in boundary else s for s in rr.timeCourseSelections]
    rr.timeCourseSelections = sel
    res = np.asarray(rr.simulate(0.0, _T_END, _N_POINTS))
    names = [c[1:-1] if c.startswith("[") else c for c in rr.timeCourseSelections][1:]
    return res[:, 1:], names


def test_biomd451_ode_integrates_finite():
    """The headline symptom: bngsim ODE no longer raises CV_CONV_FAILURE.

    Loads the model and integrates over the issue's exact protocol. A flag=-4
    regression turns this into a ``SimulationError`` at load+run; the assertions
    also guard against a silent non-finite trajectory.
    """
    if not _MODEL.exists():
        pytest.skip(f"rr_parity corpus model not present: {_MODEL}")

    model = bngsim.Model.from_sbml(str(_MODEL))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, _T_END), n_points=_N_POINTS, rtol=_RTOL, atol=_ATOL
    )

    species = np.asarray(res.species)
    assert species.shape[0] == _N_POINTS
    assert np.isfinite(species).all(), "non-finite trajectory after integration"
    assert np.asarray(res.time)[-1] == pytest.approx(_T_END)


def test_biomd451_ode_matches_roadrunner():
    """bngsim ODE agrees with the libRoadRunner reference across the horizon.

    RoadRunner is the existence proof the issue cites; parity confirms the model
    is not merely *finite* but *correct*. Gated on roadrunner being importable.
    """
    if not _MODEL.exists():
        pytest.skip(f"rr_parity corpus model not present: {_MODEL}")
    roadrunner = pytest.importorskip("roadrunner")

    model = bngsim.Model.from_sbml(str(_MODEL))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, _T_END), n_points=_N_POINTS, rtol=_RTOL, atol=_ATOL
    )
    bn = np.asarray(res.species)
    bn_names = list(res.species_names)

    rr_vals, rr_names = _rr_concentrations(roadrunner, _MODEL)

    bn_map = {n: i for i, n in enumerate(bn_names)}
    rr_map = {n: i for i, n in enumerate(rr_names)}
    common = sorted(set(bn_map) & set(rr_map))
    assert common, "bngsim/RoadRunner species sets are disjoint (loader divergence)"

    bi = [bn_map[n] for n in common]
    ri = [rr_map[n] for n in common]
    a = bn[:, bi]
    b = rr_vals[:, ri]
    # Solver error model: |a-b| <= atol + rtol*|b|, per-cell. A handful of
    # cells may graze it on two independent SUNDIALS builds; gate on the
    # fraction over budget, the same shape rr_parity's differ uses.
    over = np.abs(a - b) > (_ATOL + 1e-4 * np.abs(b))
    assert over.mean() < 1e-3, f"fail fraction {over.mean():.2e} over budget"
