"""GH #95 — build-time analytical-Jacobian derivation budget (the ODE "timeout" half).

The chatter half of #95 (`test_sbml_event_chatter_biomd711.py`) was a slow *solve*.
This is the other half: a slow *build*. The rr_parity harness times
``Model.from_sbml()`` **and** ``run()`` against one wall cap, so a slow build reads
as an ODE "timeout." (This module used to mirror that single wall and no longer
does — see ``_SOLVE_WALL_CAP`` for what went wrong with it.)

The #76 analytical Jacobian symbolically differentiates every Functional rate law
with sympy at model-build time. On a few large BioModels the derivation ran tens
of seconds to over a minute while the ODE solve was already sub-second under a
finite-difference Jacobian — the bet (pay derivation to speed the solve) is
guaranteed to lose because the solve was never Jacobian-bound:

  * ``BIOMD0000000496`` (295 species, 333 functional reactions, rate laws inlining
    to ~5 kB): derivation ~41 s when #95 was written, **3.2 s today**. Solve
    ~0.04 s either way.
  * ``BIOMD0000000628`` (139 species, 210 functional reactions whose 18-char rate
    laws each inline to ~21 kB): derivation ~75 s then, **18.5 s today**. Solve
    ~0.03 s either way.

The fix (``bngsim._jacobian.attach_functional_jacobian``) bounds the derivation
wall-time (``BNGSIM_JAC_DERIV_BUDGET_S``): a model that derives under budget keeps
the analytical Jacobian, one that does not falls back to the finite-difference
Jacobian instead of hanging.

**The fixtures are no longer losers at the shipping budget, and that is why this
module forces a small one** (issue #190). #96's printer fix and the work after it
took 496's derivation from ~41 s to 3.2 s and 628's from ~75 s to 18.5 s, so both
models now derive *completely* inside the 20 s default — the bet they were picked
to lose, they now win. What is still
worth pinning is the mechanism: when the budget does engage, the fallback is whole
(``attach_functional_jacobian`` discards every term it derived) and the model still
integrates correctly. ``_TEST_BUDGET_S`` is therefore chosen from each fixture's
*measured* derivation cost with a wide margin, not from anything the shipping
default resembles. When the assertion below flips again, the fix is to re-measure
and lower the budget, or to re-pick the fixture — not to relax the claim.

Two things are locked in here:

  * **The losers fall back and stay correct** (``test_large_functional_*``): the
    build collapses and the FD-fallback trajectory still matches RoadRunner.
  * **A model that genuinely needs the analytical Jacobian is not starved**
    (``test_needs_analytical_*``): ``BIOMD0000000457`` was stiff enough that its FD
    solve *failed* at the 1e-9/1e-12 parity tolerance, yet it derived in only ~12 s,
    so the default budget keeps it on the analytical path. This is the regression
    the budget value was chosen to avoid — a budget too small would turn its PASS
    into a solver failure. **Both halves of that rationale have since gone stale**
    the same way the loser half did (issue #210, and see
    ``test_default_budget_covers_*``); the assertions below are still true and
    still pass, but the fixture no longer demonstrates what it was picked to
    demonstrate.

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

# The forced budget, and the margin that makes it a *decision* rather than a coin
# flip. Measured on this corpus, unbounded (``BNGSIM_JAC_DERIV_BUDGET_S=inf``),
# two reps, idle machine — ``Model.from_sbml`` + ``prepare_analytical_jacobian``:
#
#   BIOMD0000000496   3.2-3.5 s     13x this budget
#   BIOMD0000000628  17.5-18.7 s    74x this budget
#
# #190 is what happens without that margin: at the old 3 s, 496's 3.2 s derivation
# left a 1.07x margin, so whether "the budget engaged" held came down to machine
# speed and load — it held when the test was written and stopped holding when the
# derivation got faster. A margin this wide costs nothing (the cut-off state is
# identical either way, see below) and needs another 13x speed-up to rot.
_TEST_BUDGET_S = "0.25"

# Two caps over two separate phases, because they fail for different reasons and
# only one of them is this module's subject.
#
# _ATTACH_WALL_CAP covers load + the Jacobian attach: the phase the #95 budget
# actually bounds, with no codegen and no .so cache in it. Observed 0.4-1.2 s at
# the budget above — the overshoot is real and expected, since the deadline is
# checked *between* reactions, so one 21 kB-inlined 628 rate law can run past it.
# The cap is what pins that overshoot to a single rate law rather than an
# unbounded derivation (an unbudgeted 628 attach is 18.5 s and trips it).
#
# _SOLVE_WALL_CAP covers ``run()`` alone. It deliberately does NOT span
# ``Simulator()``: on a Functional model that size the constructor is codegen, and
# codegen cost is decided by the .so cache, not by anything #95 bounds — 0.08 s
# warm against 35 s cold for 496, of which ~17 s is a speculative ∂f/∂p emission a
# plain ODE run never calls (issue #209, #190's second half).
# The old single 25 s wall spanned all of it, which is why it read as ~1 s of
# margin from a warm checkout and blew the cap outright from a cold one.
_ATTACH_WALL_CAP = 8.0
_SOLVE_WALL_CAP = 25.0


def _model_xml(model_id: str) -> Path | None:
    xmls = sorted((_MODELS_DIR / model_id).glob("*.xml"))
    return xmls[0] if xmls else None


@pytest.mark.parametrize("model_id,t_end,n_points", _LOSER_CASES)
def test_large_functional_build_solve_under_budget(model_id, t_end, n_points, monkeypatch):
    """The budget engages, the fallback is whole, and the model still integrates.

    Three claims, in the order they can fail:

    * the derivation phase stays bounded (``_ATTACH_WALL_CAP``);
    * the budget *engaged* — ``analytical_jacobian_complete is False``, the direct
      and machine-independent signal, and the one #190 was filed about;
    * the model integrates on the finite-difference Jacobian it fell back to, in a
      finite trajectory of the right shape.

    ``attach_functional_jacobian`` discards ``all_terms`` wholesale when the
    deadline fires, so the resulting model state does not depend on *where* in the
    reaction list the cut-off landed. That is what lets the budget be set for
    margin (see ``_TEST_BUDGET_S``) instead of for realism: a 0.25 s cut-off and a
    3 s one leave the same model behind.
    """
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")
    monkeypatch.setenv("BNGSIM_JAC_DERIV_BUDGET_S", _TEST_BUDGET_S)

    # GH #145 defers the derivation off the load path; warm it explicitly so the
    # attach is timed on its own rather than inside the Simulator constructor,
    # which would fold codegen into the measurement. Simulator() below then finds
    # it already prepared, exactly as a second Simulator on one model would.
    t0 = time.perf_counter()
    model = bngsim.Model.from_sbml(str(xml))
    model.prepare_analytical_jacobian()
    attach_wall = time.perf_counter() - t0

    assert attach_wall < _ATTACH_WALL_CAP, (
        f"{model_id} load+Jacobian-attach took {attach_wall:.1f}s under a "
        f"{_TEST_BUDGET_S}s derivation budget — the budget stopped bounding the "
        "derivation (unbudgeted: 3.2s for 496, 18.5s for 628)"
    )
    assert model._core.analytical_jacobian_complete is False, (
        f"{model_id} attached the analytical Jacobian under a {_TEST_BUDGET_S}s budget — "
        "the build-time budget did not engage. If the derivation simply got faster "
        "again (it is 13-74x this budget as of #190), re-measure it unbudgeted and "
        "lower _TEST_BUDGET_S, or re-pick the fixture; do not relax this claim."
    )

    # Construction is codegen on a model this size; it is timed by neither cap.
    sim = bngsim.Simulator(model, method="ode")
    t0 = time.perf_counter()
    res = sim.run(
        t_span=(0.0, t_end), n_points=n_points, rtol=_RTOL, atol=_ATOL, timeout=_SOLVE_WALL_CAP
    )
    solve_wall = time.perf_counter() - t0

    species = np.asarray(res.species)
    assert species.shape[0] == n_points
    assert np.isfinite(species).all(), "non-finite trajectory after integration"
    assert solve_wall < _SOLVE_WALL_CAP, (
        f"{model_id} FD-fallback solve took {solve_wall:.1f}s (measured: ~0.04s)"
    )


@pytest.mark.parametrize("model_id,t_end,n_points", _LOSER_CASES)
def test_large_functional_fd_matches_roadrunner(model_id, t_end, n_points, monkeypatch):
    """The FD-fallback trajectory agrees with the RoadRunner reference.

    Proves the budget fallback produces the correct trajectory, not merely a fast
    build. Gated on roadrunner being importable.

    Until #190 this case was mis-titled for ``BIOMD0000000496``: at the old 3 s
    budget that model derived *completely*, so what was compared against RoadRunner
    was the analytical Jacobian, not the FD fallback the name promises. The budget
    that makes the sibling test's claim true fixes this one's subject too.
    """
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")
    roadrunner = pytest.importorskip("roadrunner")
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)
    monkeypatch.setenv("BNGSIM_JAC_DERIV_BUDGET_S", _TEST_BUDGET_S)

    model = bngsim.Model.from_sbml(str(xml))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=_RTOL, atol=_ATOL, timeout=_SOLVE_WALL_CAP
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

    **Both measurements behind that reasoning are stale** (found while fixing
    #190). Today ``BIOMD0000000457`` derives in **0.26 s**, not ~12 s, and its FD
    solve no longer fails: against RoadRunner at the parity tolerance it scores a
    1.7e-4 fail fraction, well inside the 1e-2 budget the sibling parity test
    allows (analytical scores 0.0). So the 15 s floor asserted here is no longer
    justified by anything measured — it is not *wrong*, and it is not being
    relaxed here, but the corpus no longer contains the evidence for it. Re-picking
    the fixture, or deriving the floor from whatever the slowest genuinely
    needs-analytical model is today, is issue #210.
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
