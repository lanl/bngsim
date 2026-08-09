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
  * **No corpus model needs the analytical Jacobian, so the budget has no
    correctness floor left** (``test_no_corpus_model_needs_*``, issue #249). This
    replaces the claim that stood here through #95, #210 and #244: that
    ``BIOMD0000000457``'s FD solve *fails* at the parity tolerance, so the default
    budget had to stay large enough to keep it on the analytical path. It does not
    fail. Re-running #244's own classification over the whole corpus finds **zero**
    needs-analytical models — see ``_NEEDS_ANALYTICAL_SWEEP`` for the numbers and
    ``_DEFAULT_BUDGET_IS_A_PERFORMANCE_KNOB`` for what now guards the constant.

**Why the old claim went unnoticed for a merge.** It was true when #95 wrote it
and it is corpus-gated, so it *skips* in CI (``ssss.ss`` for this file, under
``model corpus absent from this checkout``). Only a developer checkout with the
corpus materialized runs it, where it went red the moment #244 landed. A
corpus-gated assertion is worth having, but the tick on a green PR is not evidence
about any of them.

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
# The model #95 picked as needing the analytical Jacobian. It does not any more
# (issue #249) — kept as the canary the sweep below is re-opened by.
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

# ─── The floor is retired: it has no subject left (issue #249) ───────────────
#
# #95 floored the shipping default at 15 s to clear a ~12 s derivation for the one
# model whose solve was said to need the analytical Jacobian; #210/#244 lowered
# that to 5 s after re-measuring the derivation at 0.51 s. Both rested on
# ``BIOMD0000000457``'s FD solve failing. It does not fail, and the assertion that
# claimed it did was red on ``main`` from the moment #244 merged — invisible
# because it is corpus-gated and CI has no corpus.
#
# It is not marginal and it is not this machine. 457's FD solve returns in 0.07 s
# with 626 steps and a finite trajectory, and it survives the whole ladder either
# side of the parity pair — rtol 1e-6/1e-12 through 1e-12/1e-15, dense and sparse
# linear solvers, interpreted and codegen RHS. Nothing resembling the "CVODE
# returns -3 at t~3.36 with h~1e-42" the old docstring describes appears anywhere
# in that grid. Its derivation is 0.283 s, which puts it *below* the 0.5 s band
# #244 selected fixtures from, so its own recipe would not pick it today either.
# (The two loser fixtures re-measure at 3.486 s and 19.821 s here, against #244's
# 3.2-3.5 s and 17.5-18.7 s — so this is the same machine class its numbers came
# from, not a fast outlier.)
#
# _NEEDS_ANALYTICAL_SWEEP — #244's classification, re-run over the whole corpus
# rather than a slow band, one fresh process and one fresh model per arm:
#
#   * 1,323 rr_parity ODE models probed; 1,291 load (32 do not). The analytical
#     Jacobian attaches on 1,218 and declines on 73 — a decline is on FD at *every*
#     budget, so no budget can starve it, and those 73 drop out of the question.
#   * All 1,218 attaching models were then forced onto FD (``jacobian="fd"`` costs
#     no derivation, so the whole set is affordable — there is no band and nothing
#     is sampled). **1,198 solve.** 15 fail, and every one of the 15 fails
#     identically *with* the analytical Jacobian — no Jacobian saves them. 5 cannot
#     be simulated at all (``fast="true"`` reactions the loader declines).
#   * **Needs-analytical models: zero.**
#   * Nor is there a weaker property to re-anchor on. Over the 23 models whose
#     derivation is slow enough (>= 0.5 s) to be starved by any plausible budget,
#     the worst FD-vs-analytical trajectory difference is 1.9e-6 relative — solver
#     noise at rtol 1e-9, not a correctness gap.
#
# So the budget is a *performance* knob on this corpus, not a correctness one, and
# a floor derived from a needs-analytical derivation would be the exact mistake
# #244 called out: a tripwire standing on a measurement that has evaporated. What
# replaces it is a pin on the constant (below) plus a canary on 457, so the
# question re-opens by itself if a needs-analytical model ever reappears.
#
# The sweep behind the retired floor, kept for provenance:
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
# Two of that sweep's readings do not reproduce, and the difference is the whole
# of #249: the band is 23 models here rather than 68 (its threshold appears to
# have been applied to load+derive, 39 by that reading, the rest being 4-worker
# contention it says it measured 1.3x of), and 457's FD arm solves. The #249 sweep
# above needs neither judgement, because it runs the FD arm on *every* attaching
# model instead of a band drawn by a timing threshold.
#
# The shipping default stays 20 s: dropping the floor removes a requirement, it
# does not license a change, and re-tuning the default is not this module's to do.
# What guards it now is an equality pin rather than an inequality, because with no
# correctness requirement left there is no inequality that means anything — see
# ``test_default_budget_is_a_performance_knob``.
_DEFAULT_BUDGET_S = 20.0


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


def test_default_budget_is_a_performance_knob():
    """The default is pinned, because no corpus model needs it to be anything.

    This used to be an inequality — the default had to clear the slowest
    *needs-analytical* derivation, so a smaller one would strand a model on a
    failing FD solve. Issue #249 retired that floor: forcing FD on all 1,218
    models the analytical Jacobian attaches to, none of them needs it (see
    ``_NEEDS_ANALYTICAL_SWEEP``), and the worst FD-vs-analytical difference in the
    slow band is 1.9e-6 relative, which is solver noise at rtol 1e-9. With no
    correctness requirement there is no meaningful inequality left to assert: any
    floor would be a number chosen to look like evidence.

    An equality pin says the honest thing instead. Moving the default is now a
    *performance* decision — how much derivation a build is willing to pay for and
    discard — and this test makes it a deliberate one: change the constant here,
    and say in the commit what the new value was measured against. It needs no
    corpus and no reference engine, which is the point, since everything that does
    need a corpus skips in CI.
    """
    assert _DEFAULT_DERIVATION_BUDGET_S == _DEFAULT_BUDGET_S, (
        f"the default derivation budget moved from {_DEFAULT_BUDGET_S}s to "
        f"{_DEFAULT_DERIVATION_BUDGET_S}s. That is allowed — no corpus model needs "
        "the analytical Jacobian (issue #249), so nothing here forbids it — but it "
        "is a performance decision, not a free one: a budget below a model's "
        "derivation cost makes the build pay the derivation and then discard it. "
        "Update _DEFAULT_BUDGET_S with the new value and record what it was "
        "measured against; do not delete this pin."
    )


def test_no_corpus_model_needs_the_analytical_jacobian():
    """The canary that re-opens ``_NEEDS_ANALYTICAL_SWEEP`` if it stops being true.

    Through #95, #210 and #244 this asserted the opposite — that ``BIOMD0000000457``
    is stiff enough that its FD solve *fails* at the parity tolerance, which is
    what floored the shipping budget. It does not fail (issue #249). Forcing FD on
    all 1,218 models the analytical Jacobian attaches to finds no model that needs
    it, so what is worth running per-commit is the one model the claim was made
    about: FD solves it, and to the same trajectory the analytical Jacobian does.

    A full re-sweep costs about an hour, so this stands in for it. **If this test
    fails, do not adjust it** — re-run the classification (``jacobian="fd"`` over
    every attaching model; it needs no derivation, so the whole corpus is
    affordable) and, if a needs-analytical model has appeared, restore a floor
    derived from *its* derivation cost.

    Two ways to accidentally not test this, both of which report a comfortable PASS
    (kept from the original, because both still apply):

    * Reuse a ``Model`` that has already been run. ``run()`` leaves the model's
      species at the end state (that is what ``Model.reset()`` is for), so a second
      ``Simulator`` on it starts from t_end and never reaches the transient at
      t~3.36 the old claim turned on. **This is what made #210 report that the FD
      solve had stopped failing.** Load a fresh model per arm, as below.
    * Let the analytical Jacobian attach anyway, so the "FD" arm is not one.
      ``analytical_jacobian_complete`` stays True on a model whose Jacobian needs
      no sympy derivation at all, so it is asserted False here rather than used as
      the FD signal generally — 77 corpus models are all-mass-action and would fail
      that reading.
    """
    model_id, t_end, n_points = _NEEDS_ANALYTICAL
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")

    run_kw = dict(
        t_span=(0.0, t_end),
        n_points=n_points,
        rtol=_RTOL,
        atol=_ATOL,
        timeout=_SOLVE_WALL_CAP,
    )

    fd_model = bngsim.Model.from_sbml(str(xml))
    fd_sim = bngsim.Simulator(fd_model, method="ode", jacobian="fd")
    assert fd_model._core.analytical_jacobian_complete is False, (
        f'{model_id} attached the analytical Jacobian under jacobian="fd" — this '
        "test is not measuring the FD solve"
    )
    try:
        fd = np.asarray(fd_sim.run(**run_kw).species)
    except (SimulationError, SimulationTimeout) as exc:  # pragma: no cover - the canary
        pytest.fail(
            f"{model_id}'s FD solve failed ({type(exc).__name__}) — it succeeded in "
            "0.07s over rtol 1e-6..1e-12 and both linear solvers when #249 retired "
            "the needs-analytical floor. Re-run the corpus classification: if a "
            "needs-analytical model exists again, restore a floor from its "
            "derivation cost rather than editing this test."
        )
    assert np.isfinite(fd).all(), f"{model_id} FD trajectory is non-finite"

    # ...and the analytical Jacobian buys nothing on it: same trajectory, so the
    # budget cannot strand this model however small it gets.
    an_model = bngsim.Model.from_sbml(str(xml))
    an = np.asarray(bngsim.Simulator(an_model, method="ode").run(**run_kw).species)
    scale = float(np.maximum(np.abs(an), np.abs(fd)).max())
    rel = float(np.abs(an - fd).max() / scale) if scale else 0.0
    # Measured 8.5e-6 — two Jacobians take different step sequences, and this is
    # the largest such gap anywhere in the corpus (the 23-model slow band tops out
    # at 1.9e-6). The threshold carries ~120x over it, because what it has to
    # separate is solver noise from the analytical Jacobian actually changing the
    # answer, and only the second is worth re-opening the sweep for.
    assert rel < 1e-3, (
        f"{model_id} FD and analytical trajectories differ by {rel:.2e} relative "
        "(measured 8.5e-6 when #249 retired the floor) — both solve, but not to "
        "the same answer, so the analytical Jacobian is doing something here after "
        "all. Re-run the #249 classification."
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
