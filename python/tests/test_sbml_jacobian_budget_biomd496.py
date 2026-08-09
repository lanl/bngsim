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
    to ~5 kB): derivation ~41 s when #95 was written, 3.2 s / 10.9 s today. Solve
    ~0.04 s either way.
  * ``BIOMD0000000628`` (139 species, 210 functional reactions whose 18-char rate
    laws each inline to ~21 kB): derivation ~75 s then, 18.5 s / 58.5 s today.
    Solve ~0.03 s either way.

**Every derivation second quoted in this module is machine-scoped, and the two
figures above are why the pair is quoted.** The left number is what #190 measured;
the right is the same model, same corpus, same commit (``_jacobian.py`` has not
changed since #191) re-measured for #210 on a different development machine. The
ratio is a uniform 3.1-3.3x across both fixtures, so what travels between machines
is the *ratio* of a derivation to the budget, never the seconds. Every margin
below is stated as a ratio for that reason.

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
    (``test_needs_analytical_*``): ``BIOMD0000000457`` is stiff enough that its FD
    solve *fails* at the 1e-9/1e-12 parity tolerance, so the default budget must be
    large enough to keep it on the analytical path. This is the regression the
    budget value was chosen to avoid — a budget too small would turn its PASS into
    a solver failure. That claim is no longer prose: the ``fails_on_the_fd_jacobian``
    case below runs it. What *had* gone stale (issue #210) is the derivation cost
    the floor was sized against — ~12 s when #95 picked it, **0.51 s** now. See
    ``test_default_budget_covers_*`` for the re-derived floor and the corpus sweep
    behind it.

Like the chatter test, these are gated on both the gitignored corpus model and
libRoadRunner being present locally.
"""

from __future__ import annotations

import time
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SimulationError, SimulationTimeout
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
#   BIOMD0000000496   3.2-3.5 s / 10.9 s      13x / 43x this budget
#   BIOMD0000000628  17.5-18.7 s / 58.5 s     74x / 234x this budget
#
# (Two machines, ~3.3x apart; see the module docstring. The slower one only widens
# the margin, so 13x is the one to reason with.)
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

# The floor ``test_default_budget_covers_needs_analytical_models`` holds the
# shipping default to, and the measurement it is derived from (issue #210).
#
# #95 set the floor at 15 s to clear a ~12 s derivation for the one model whose
# solve needs the analytical Jacobian. The 12 s is gone — the #96-and-after
# speed-ups took it to 0.51 s — so the floor was left standing on a number that no
# longer exists anywhere in the corpus. Re-derived by re-running the classification
# #95 did, on the whole corpus:
#
#   * 1319 rr_parity ODE models materialized; 1286 load (32 fail to load, 1 ran
#     past a 400 s probe cap). The analytical Jacobian attaches on 1214 and
#     declines on 72 — a decline is on FD at *every* budget, so no budget can
#     starve it, and those 72 drop out of the question.
#   * The 68 attaching models whose unbudgeted derivation costs >= 0.5 s were run
#     three ways at the parity tolerance on their own rr_parity horizon, one fresh
#     process each: analytical, FD forced, and RoadRunner.
#   * **Exactly one needs the analytical Jacobian: BIOMD0000000457.** On FD its
#     solve does not merely drift, it fails — CVODE returns -3 at t~3.36 with
#     h~1e-42. Analytical solves and scores a 0.0 fail fraction against RoadRunner
#     (max abs diff 1.1e-10). Of the other 67: 63 solve on FD to a 0.0 fail
#     fraction against their own analytical trajectory, 2 fail in *both* modes (no
#     Jacobian saves them), and 2 are declared rr_parity artifacts where the
#     engines legitimately diverge (376 oscillator phase drift — its FD run is
#     *closer* to RoadRunner than its analytical one; MODEL1112050001 exponentially
#     ill-conditioned past t~10). None of the four is a needs-analytical model.
#   * 457 sits near the bottom of that band, which is what makes 0.51 s a *bound*
#     and not just the largest value seen: all 53 models that derive more slowly
#     are accounted for above, and every model excluded from the band derives in
#     under 0.5 s — so nothing left untested can push the slowest needs-analytical
#     derivation above 457's.
#
# The margin, composed from measured spreads rather than picked: 3.3x for machine
# speed (the #190-vs-#210 ratio in the module docstring), 1.3x for worker
# contention (0.51 s serial against 0.66 s under a 4-worker sweep), and 2x of
# ordinary headroom — ~10x over 0.51 s.
#
# This LOWERS a tripwire on a constant; it does not lower the constant. The
# shipping default is unchanged at 20 s and is not this module's to re-tune. What
# the old value did do is forbid the re-tuning: at 15 s the default could never be
# moved down toward the loser band (10.9 s for the fastest loser here) however
# strong the evidence, on the strength of a 12 s measurement that had evaporated.
_SLOWEST_NEEDS_ANALYTICAL_S = 0.51
_NEEDS_ANALYTICAL_FLOOR_S = 5.0


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
        "derivation (unbudgeted: 3.2-10.9s for 496, 18.5-58.5s for 628, the range "
        "being two development machines)"
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

    ``BIOMD0000000457`` *needs* the analytical Jacobian — on FD its solve fails
    outright, which ``test_needs_analytical_model_fails_on_the_fd_jacobian`` runs
    rather than asserting in prose. A default below its derivation cost would
    silently strand it on that failing solve, so the default is floored at ~10x the
    cost with the margin composed from measured spreads. See
    ``_NEEDS_ANALYTICAL_FLOOR_S`` for the corpus sweep the number comes from, and
    for why the old 15 s floor (a ~12 s derivation that is now 0.51 s) had to go.

    The guard is on the constant, so it needs no corpus and no reference engine.
    What it protects is the corpus fixture two tests below.
    """
    assert _DEFAULT_DERIVATION_BUDGET_S >= _NEEDS_ANALYTICAL_FLOOR_S, (
        f"default derivation budget {_DEFAULT_DERIVATION_BUDGET_S}s is below the "
        f"{_NEEDS_ANALYTICAL_FLOOR_S}s needs-analytical floor — BIOMD0000000457 "
        f"derives in {_SLOWEST_NEEDS_ANALYTICAL_S}s here and cannot solve without "
        "the analytical Jacobian, and the floor carries ~10x for machine speed and "
        "contention. Re-measure the corpus before lowering this, and record what "
        "the slowest needs-analytical derivation became."
    )


def test_needs_analytical_model_fails_on_the_fd_jacobian():
    """The premise under the floor, run instead of asserted in prose.

    Everything ``_NEEDS_ANALYTICAL_FLOOR_S`` claims rests on this fixture actually
    *needing* the analytical Jacobian. That was a docstring sentence while the
    corpus moved under it (issue #210), so it is a test now: force FD and the solve
    does not come back. Here that is a ``SimulationError`` — CVODE returns -3 at
    t~3.36 with h~1e-42 — and a machine on which it grinds instead of giving up
    fails the same way through ``timeout`` (``SimulationTimeout``), which is the
    other half of "FD is not a viable path here."

    Two ways to accidentally not test this, both of which report a comfortable PASS:

    * Reuse a ``Model`` that has already been run. ``run()`` leaves the model's
      species at the end state (that is what ``Model.reset()`` is for), so a second
      ``Simulator`` on it starts from t_end and never reaches the stiff transient at
      t~3.36 that FD fails on. **This is what made #210 report that the FD solve had
      stopped failing** — it read `fd: OK` off a model the preceding analytical run
      had already carried to t=10. Load a fresh model, as below.
    * Let the analytical Jacobian attach anyway. ``jacobian="fd"`` is the switch;
      ``BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0`` and a sub-derivation budget reach the
      same solve, and all three fail here.

    If this ever stops raising, the floor above has lost its subject: re-run the
    corpus classification and re-pick the fixture from whatever needs analytical
    then. Do not delete the assertion and keep the floor.
    """
    model_id, t_end, n_points = _NEEDS_ANALYTICAL
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")

    model = bngsim.Model.from_sbml(str(xml))
    sim = bngsim.Simulator(model, method="ode", jacobian="fd")
    assert model._core.analytical_jacobian_complete is False, (
        f'{model_id} attached the analytical Jacobian under jacobian="fd" — this '
        "test is not measuring the FD solve"
    )
    with pytest.raises((SimulationError, SimulationTimeout)):
        sim.run(
            t_span=(0.0, t_end),
            n_points=n_points,
            rtol=_RTOL,
            atol=_ATOL,
            timeout=_SOLVE_WALL_CAP,
        )


def test_needs_analytical_model_keeps_analytical_and_solves(monkeypatch):
    """A stiff model that FD cannot solve keeps its analytical Jacobian.

    The other half of the test above: ``BIOMD0000000457``'s finite-difference solve
    fails at the 1e-9/1e-12 parity tolerance, and only the analytical Jacobian
    integrates it. With a generous budget (so the result does not depend on machine
    speed) the derivation completes, ``analytical_jacobian_complete`` is True, and
    the solve succeeds — the regression the budget value exists to prevent.
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
