"""GH #95 — build-time analytical-Jacobian derivation budget (the ODE "timeout" half).

The chatter half of #95 (`test_sbml_event_chatter_biomd711.py`) was a slow *solve*.
This is the other half: a slow *build*. The rr_parity harness times
``Model.from_sbml()`` **and** ``run()`` against one wall cap, so a slow build reads
as an ODE "timeout."

The #76 analytical Jacobian symbolically differentiates every Functional rate law
with sympy at model-build time. On a few large BioModels the derivation runs tens
of seconds to over a minute while the ODE solve is already sub-second under a
finite-difference Jacobian — the bet (pay derivation to speed the solve) is
guaranteed to lose because the solve was never Jacobian-bound:

  * ``BIOMD0000000496`` (295 species, 333 functional reactions, rate laws inlining
    to ~5 kB): derivation ~41 s, solve ~0.25 s. FD solve is 0.25 s too.
  * ``BIOMD0000000628`` (139 species, 210 functional reactions whose 18-char rate
    laws each inline to ~21 kB): derivation ~75 s, solve ~0.1 s. FD solve matches.

The fix (``bngsim._jacobian.attach_functional_jacobian``) bounds the derivation
wall-time (``BNGSIM_JAC_DERIV_BUDGET_S``): a model that derives under budget keeps
the analytical Jacobian, one that does not falls back to the finite-difference
Jacobian instead of hanging.

Two things are locked in here:

  * **The losers fall back and stay correct** (``test_large_functional_*``): the
    build collapses and the FD-fallback trajectory still matches RoadRunner.
  * **A model that genuinely needs the analytical Jacobian is not starved**
    (``test_needs_analytical_*``): ``BIOMD0000000457`` is stiff enough that its FD
    solve *fails* at the 1e-9/1e-12 parity tolerance, yet it derives in only ~12 s,
    so the default budget keeps it on the analytical path. This is the regression
    the budget value was chosen to avoid — a budget too small would turn its PASS
    into a solver failure.

Like the chatter test, these are gated on both the gitignored corpus model and
libRoadRunner being present locally.
"""

from __future__ import annotations

import time
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._jacobian import _DEFAULT_DERIVATION_BUDGET_S

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"

# (model_id, t_end, n_points) — the rr_parity ODE job horizon for each model.
_LOSER_CASES = [
    ("BIOMD0000000496", 10.0, 1001),
    ("BIOMD0000000628", 10.0, 1001),
]
# The model whose solve genuinely needs the analytical Jacobian.
_NEEDS_ANALYTICAL = ("BIOMD0000000457", 10.0, 1001)

# The parity sweep forces these (tight) tolerances on both engines.
_RTOL = 1e-9
_ATOL = 1e-12

# A small derivation budget makes the loser build deterministic and fast (~6 s)
# regardless of the shipping default, isolating the test from machine speed: the
# fixed path is well under this wall cap, the unbounded derivation (40-80 s) is
# well over it.
_TEST_BUDGET_S = "3"
_LOSER_WALL_CAP = 25.0


def _model_xml(model_id: str) -> Path | None:
    xmls = sorted((_MODELS_DIR / model_id).glob("*.xml"))
    return xmls[0] if xmls else None


@pytest.mark.parametrize("model_id,t_end,n_points", _LOSER_CASES)
def test_large_functional_build_solve_under_budget(model_id, t_end, n_points, monkeypatch):
    """Build + solve completes well under the wall cap, on the FD fallback.

    With the derivation budget the build is bounded and the model integrates on the
    finite-difference Jacobian; without it the build alone runs 40-80 s. The
    ``analytical_jacobian_complete is False`` assertion pins that the budget
    actually engaged (a regression that derives fully would flip it to True *and*
    blow the wall cap).
    """
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")
    monkeypatch.setenv("BNGSIM_JAC_DERIV_BUDGET_S", _TEST_BUDGET_S)

    t0 = time.perf_counter()
    model = bngsim.Model.from_sbml(str(xml))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=_RTOL, atol=_ATOL, timeout=_LOSER_WALL_CAP
    )
    wall = time.perf_counter() - t0

    species = np.asarray(res.species)
    assert species.shape[0] == n_points
    assert np.isfinite(species).all(), "non-finite trajectory after integration"
    assert wall < _LOSER_WALL_CAP, (
        f"{model_id} build+solve took {wall:.1f}s — derivation-budget regression? "
        "(unbounded derivation runs 40-80s)"
    )
    assert model._core.analytical_jacobian_complete is False, (
        f"{model_id} attached the analytical Jacobian under a {_TEST_BUDGET_S}s budget — "
        "the build-time budget did not engage"
    )


@pytest.mark.parametrize("model_id,t_end,n_points", _LOSER_CASES)
def test_large_functional_fd_matches_roadrunner(model_id, t_end, n_points, monkeypatch):
    """The FD-fallback trajectory agrees with the RoadRunner reference.

    Proves the budget fallback produces the correct trajectory, not merely a fast
    build. Gated on roadrunner being importable.
    """
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")
    roadrunner = pytest.importorskip("roadrunner")
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)
    monkeypatch.setenv("BNGSIM_JAC_DERIV_BUDGET_S", _TEST_BUDGET_S)

    model = bngsim.Model.from_sbml(str(xml))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=_RTOL, atol=_ATOL, timeout=_LOSER_WALL_CAP
    )
    bn = np.asarray(res.species)
    bn_names = list(res.species_names)

    rr = roadrunner.RoadRunner(str(xml))
    rr.integrator = "cvode"
    rr.integrator.relative_tolerance = _RTOL
    rr.integrator.absolute_tolerance = _ATOL
    boundary = set(rr.model.getBoundarySpeciesIds())
    rr.timeCourseSelections = [f"[{s}]" if s in boundary else s for s in rr.timeCourseSelections]
    ref = np.asarray(rr.simulate(0.0, t_end, n_points))
    rr_names = [c[1:-1] if c.startswith("[") else c for c in rr.timeCourseSelections][1:]

    bn_map = {n: i for i, n in enumerate(bn_names)}
    common = sorted(set(bn_map) & set(rr_names))
    assert common, "bngsim/RoadRunner species sets are disjoint (loader divergence)"

    a = bn[:, [bn_map[n] for n in common]]
    b = ref[:, [1 + rr_names.index(n) for n in common]]
    # Cross-engine solver error budget |a-b| <= atol + rtol*|b|, with a loose
    # relative term (the engines differ in step control, not in the dynamics).
    over = np.abs(a - b) > (_ATOL + 1e-3 * np.abs(b) + 1e-9)
    assert over.mean() < 1e-2, f"{model_id} fail fraction {over.mean():.2e} over budget"


def test_default_budget_covers_needs_analytical_models():
    """The shipping default must clear the slowest needs-analytical derivation.

    ``BIOMD0000000457`` derives in ~12 s even under worker contention and *needs*
    the analytical Jacobian (its FD solve fails at the parity tolerance, see below).
    The default budget must exceed that with margin, or loading the model on the
    default would silently strand it on a failing FD solve. Machine-independent
    guard against lowering the default into the danger zone.
    """
    assert _DEFAULT_DERIVATION_BUDGET_S >= 15.0, (
        f"default derivation budget {_DEFAULT_DERIVATION_BUDGET_S}s is below the "
        "~12s needs-analytical derivation of BIOMD0000000457 (+ margin)"
    )


def test_needs_analytical_model_keeps_analytical_and_solves(monkeypatch):
    """A stiff model that FD cannot solve keeps its analytical Jacobian.

    ``BIOMD0000000457``'s finite-difference solve fails at the 1e-9/1e-12 parity
    tolerance; only the analytical Jacobian integrates it. With a generous budget
    (so the result does not depend on machine speed) the derivation completes,
    ``analytical_jacobian_complete`` is True, and the solve succeeds — the
    regression the budget value exists to prevent.
    """
    model_id, t_end, n_points = _NEEDS_ANALYTICAL
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")
    # Generous budget: deterministically derive regardless of machine/contention.
    monkeypatch.setenv("BNGSIM_JAC_DERIV_BUDGET_S", "90")

    model = bngsim.Model.from_sbml(str(xml))
    # GH #145: the derivation is deferred off the load path — warm it (the
    # ODE-solve setup below would trigger the same) before asserting completeness.
    model.prepare_analytical_jacobian()
    assert model._core.analytical_jacobian_complete is True, (
        f"{model_id} did not attach the analytical Jacobian even with a 90s budget"
    )
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=_RTOL, atol=_ATOL, timeout=60.0
    )
    species = np.asarray(res.species)
    assert species.shape[0] == n_points
    assert np.isfinite(species).all(), "analytical-Jacobian solve produced a non-finite trajectory"
