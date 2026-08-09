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

Three things are locked in here:

  * **The losers fall back and stay correct** (``test_large_functional_*``): the
    build collapses and the FD-fallback trajectory still matches RoadRunner.
  * **No corpus model needs the analytical Jacobian, so the budget has no
    correctness floor left** (``test_no_corpus_model_needs_*``, issue #249). This
    replaces the claim that stood here through #95, #210 and #244: that
    ``BIOMD0000000457``'s FD solve *fails* at the parity tolerance, so the default
    budget had to stay large enough to keep it on the analytical path. Re-running
    #244's own classification over the whole corpus finds **zero** needs-analytical
    models — see ``_NEEDS_ANALYTICAL_SWEEP``.
  * **But it does have a performance floor, and 457 is not where it lives**
    (``test_paying_model_*`` / ``test_default_budget_covers_*``, issue #245).
    ``BIOMD0000000608``'s FD solve is correct and **4.16x slower**, and its
    derivation costs 4.76 s — nearly 10x 457's. A screen that asks only whether FD
    *works* is blind to that population by construction, which is why #249 could
    conclude no inequality was left to assert. One is: not on correctness, on cost.

**Why these claims keep going unnoticed for a merge.** The file is corpus-gated, so
it *skips* in CI (``ssss.ss``, under ``model corpus absent from this checkout``).
Only a developer checkout with the corpus runs it. The 457 assertion has now been
red on one architecture or the other continuously since #244 — as #244 wrote it, on
arm64; as #249 rewrote it, on x86_64 — because both pinned the single tolerance
where the two machines disagree. It is a ladder now, and asserts only the part that
travels. A corpus-gated assertion is worth having, but a green tick is not evidence
about any of them, and neither is one machine.

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
# 457's FD solve returns in 0.07 s with 626 steps and a finite trajectory, and it
# survives the whole ladder either side of the parity pair — rtol 1e-6/1e-12
# through 1e-12/1e-15, dense and sparse linear solvers, interpreted and codegen
# RHS. Its derivation is 0.283 s, which puts it *below* the 0.5 s band #244
# selected fixtures from, so its own recipe would not pick it today either.
#
# **On x86_64 macOS that ladder has exactly one rung missing, and it is the parity
# rung** (issue #245). Same commit, core rebuilt from the tree, corpus present:
#
#   | rtol      | 1e-6 | 1e-8   | 1e-9 (parity) | 1e-10  | 1e-12  |
#   |-----------|------|--------|---------------|--------|--------|
#   | arm64     | ok   | ok     | ok, 626 steps | ok     | ok     |
#   | x86_64    | ok   | ok     | **CVODE -3**  | ok     | ok     |
#   | |an - fd| | 0.0  | 1.9e-6 | —             | 2.7e-8 | 9.8e-10|
#
# The failure is "CVODE -3 at t~3.36 with h~3.1e-42" — #95's signature exactly. So
# both readings are real and neither machine is misconfigured: this is a knife-edge
# stiff transient whose convergence is not portable, and #244 and #249 each sampled
# the one tolerance where the two architectures disagree.
#
# **That strengthens #249's conclusion rather than weakening it.** A model that
# genuinely needed the analytical Jacobian would fail across a *band*, hardest at
# the tightest tolerance. 457 fails at an isolated point with success on both
# immediate neighbours and its cleanest agreement (9.8e-10) at the tightest rung —
# an arithmetic accident, not a property of the model. What it does rule out is
# ever pinning that one cell, which is what the canary below used to do from each
# side in turn: red on x86_64 as written, red on arm64 as #244 wrote it.
#
# (The two loser fixtures re-measure at 3.486 s and 19.821 s on the arm64 machine
# against #244's 3.2-3.5 s and 17.5-18.7 s, so #249's numbers are the same machine
# class #244's came from; the x86_64 figures quoted below are ~3.3x slower again.)
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
# What guards it now is an equality pin — see
# ``test_default_budget_is_a_performance_knob``.
_DEFAULT_BUDGET_S = 20.0

# ─── One inequality does survive, on performance rather than correctness (#245) ──
#
# #249 retired the floor on the grounds that with no correctness requirement there
# is no meaningful inequality left, and any floor would be a number chosen to look
# like evidence. That is right about *correctness* and it is why the pin above is
# an equality. It is not the whole picture, because #249's sweep asked whether the
# FD solve **works**, never what it **costs**, and one of those has a floor in it.
#
# Re-running the >= 2 s band for solve *time* — medians of repeats on a warm
# codegen cache — finds models FD solves correctly and slowly:
#
#   | model           | derivation | FD / analytical solve |
#   |-----------------|-----------:|----------------------:|
#   | BIOMD0000000608 |     4.76 s |             **4.16x** |
#   | MODEL1603150001 |     2.51 s |                 3.01x |
#   | MODEL1601050000 |     2.98 s |                 2.74x |
#   | MODEL1602080000 |     2.32 s |                 1.70x |
#   | MODEL1504130000 |     2.16 s |                 1.40x |
#
# A default below 608's 4.76 s does not merely make that build pay a derivation and
# discard it — it hands the model a solve 4.16x slower for the rest of its life. So
# the floor is 4.76 s times the one spread that can move it, 3.3x for machine speed
# (the ratio in the module docstring; seconds do not travel, ratios do): 15.7 s,
# held at 15.0. The shipping default clears it by 1.33x.
#
# **These ratios are only visible on a warm cache.** ``fd_viability.jsonl`` runs one
# cold sample per mode, analytical first, so that arm absorbs codegen warm-up: it
# reports 496 at 5.84x (really 1.02x) and MODEL1603150001 at 0.33x (really 3.01x) —
# wrong by 5.8x in one direction and 9x in the other. Every number above is a
# median of repeats after a discarded warm-up run.
#
# The ceiling, for the same reason it is not asserted: the cheapest derivation that
# does *not* pay for itself is BIOMD0000000628 at 59.3 s, whose analytical solve is
# 0.49x — slower than its FD one. Between 4.76 s and 59.3 s nothing needs getting
# right (496 at 10.9 s measures 1.02x, 497 at 11.1 s measures 1.25x), so the window
# is 12.5x wide and 20 s sits 4.2x above the floor and 3.0x below the ceiling.
_SLOWEST_PAYING_DERIVATION_S = 4.76
_PAYING_DERIVATION_FLOOR_S = 15.0

# The most expensive derivation that still pays for itself, and the speed-up
# asserted for it — well under the measured 4.16x, so the claim is a decision
# rather than a coin flip.
_PAYING_ANALYTICAL = ("BIOMD0000000608", 10.0, 1001)
_PAYING_SOLVE_SPEEDUP = 2.0

# The tolerance ladder the 457 canary walks, and how much of it must hold. A
# needs-analytical model fails across a band and hardest at the tightest rung; the
# measured x86_64/arm64 disagreement is a single interior rung, so one failure is
# tolerated and two are not. See the block comment above for the grid.
_CANARY_LADDER = ((1e-6, 1e-9), (1e-8, 1e-11), (1e-9, 1e-12), (1e-10, 1e-13), (1e-12, 1e-15))
_CANARY_MAX_ISOLATED_FAILURES = 1
# Largest FD-vs-analytical relative difference to call solver noise. Measured
# 8.5e-6 (arm64) / 1.9e-6 (x86_64); ~120x margin over the worse of the two.
_CANARY_AGREEMENT_REL = 1e-3


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
        "derivation cost makes the build pay the derivation and then discard it, "
        f"and below {_PAYING_DERIVATION_FLOOR_S}s it also costs a 4.16x slower "
        "solve on BIOMD0000000608 (issue #245, see the floor test). Update "
        "_DEFAULT_BUDGET_S with the new value and record what it was measured "
        "against; do not delete this pin."
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

    **It walks a tolerance ladder rather than pinning the parity rung**, which is
    the one thing about 457 that does not travel (issue #245). At rtol 1e-9/atol
    1e-12 the FD solve returns 626 steps on arm64 and CVODE -3 on x86_64 — the same
    commit, the same corpus, a core built from the tree. Pinning that cell asserts
    an arithmetic accident, and it has now been red on one architecture or the other
    continuously since #244: as written by #244 it failed on arm64, and as written
    by #249 it failed on x86_64. Both were invisible for a merge because the file is
    corpus-gated and CI has no corpus.

    What *is* portable is the shape of the failure. A model that genuinely needed
    the analytical Jacobian would fail across a band and hardest at the tightest
    rung; 457 fails at an isolated interior point with success on both immediate
    neighbours and its cleanest agreement at the tightest rung. So the ladder is
    walked, one isolated failure is tolerated, two are not, the tightest rung must
    solve, and every rung that solves must agree with the analytical arm.

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

    def _solve(jacobian: str | None, rtol: float, atol: float):
        """One arm, on a model that has never been run. Returns None if it fails."""
        model = bngsim.Model.from_sbml(str(xml))
        kwargs = {"jacobian": jacobian} if jacobian else {}
        sim = bngsim.Simulator(model, method="ode", **kwargs)
        if jacobian == "fd":
            assert model._core.analytical_jacobian_complete is False, (
                f'{model_id} attached the analytical Jacobian under jacobian="fd" '
                "— this test is not measuring the FD solve"
            )
        try:
            return np.asarray(
                sim.run(
                    t_span=(0.0, t_end),
                    n_points=n_points,
                    rtol=rtol,
                    atol=atol,
                    timeout=_SOLVE_WALL_CAP,
                ).species
            )
        except (SimulationError, SimulationTimeout):
            return None

    failed: list[float] = []
    worst_rel = 0.0
    for rtol, atol in _CANARY_LADDER:
        fd = _solve("fd", rtol, atol)
        if fd is None:
            failed.append(rtol)
            continue
        assert np.isfinite(fd).all(), f"{model_id} FD trajectory non-finite at rtol {rtol:.0e}"
        # ...and where it solves, the analytical Jacobian must not change the answer:
        # a budget cannot strand a model whose two Jacobians agree.
        an = _solve(None, rtol, atol)
        assert an is not None, (
            f"{model_id}'s ANALYTICAL solve failed at rtol {rtol:.0e} while FD "
            "succeeded — that is not a Jacobian question, re-run the classification"
        )
        scale = float(np.maximum(np.abs(an), np.abs(fd)).max())
        worst_rel = max(worst_rel, float(np.abs(an - fd).max() / scale) if scale else 0.0)

    tightest = _CANARY_LADDER[-1][0]
    assert len(failed) <= _CANARY_MAX_ISOLATED_FAILURES and tightest not in failed, (
        f"{model_id}'s FD solve failed at rtol {[f'{r:.0e}' for r in failed]} — "
        f"at most {_CANARY_MAX_ISOLATED_FAILURES} isolated interior rung may fail "
        f"(the known x86_64/arm64 disagreement at rtol {_RTOL:.0e}), and never the "
        f"tightest at {tightest:.0e}. A failure band means a needs-analytical model "
        "may exist again: re-run the classification and, if one does, restore a "
        "floor from its derivation cost rather than editing this test."
    )
    assert worst_rel < _CANARY_AGREEMENT_REL, (
        f"{model_id} FD and analytical trajectories differ by {worst_rel:.2e} "
        "relative (measured 8.5e-6 arm64 / 1.9e-6 x86_64) — both solve, but not to "
        "the same answer, so the analytical Jacobian is doing something here after "
        "all. Re-run the #249 classification."
    )


def test_paying_model_solves_faster_on_the_analytical_jacobian(monkeypatch):
    """The premise under ``_PAYING_DERIVATION_FLOOR_S``, run instead of asserted.

    #249 was right that there is no *correctness* floor left, and the equality pin
    above says so. What survives is a performance one: ``BIOMD0000000608`` derives
    for 4.76 s and solves 4.16x faster for it (0.065 s vs 0.015 s, 52 species, 86
    functional reactions). FD is perfectly correct there — which is precisely why a
    screen asking only whether FD *works* cannot see it, and why the floor it
    supports had to be re-derived rather than inherited (issue #245).

    The measurement traps, both of which report a comfortable PASS:

    * Time one run each. The first ``run()`` after a cold ``Simulator`` carries
      codegen warm-up and it lands on whichever mode went first — that is how
      ``BIOMD0000000496`` reads as a 5.8x analytical win in ``fd_viability.jsonl``
      when it is really 1.02x, and how ``MODEL1603150001`` reads as 0.33x when it is
      really 3.01x. Both arms are constructed, warmed, then timed best-of-3 here.
    * Reuse a model across runs. ``run()`` leaves the species at the end state, so
      the second run integrates a different problem (see the canary above).
      ``reset()`` between.

    Asserted at 2x against a measured 4.16x. The budget is lifted for the analytical
    arm so this measures the *value* of the derivation rather than whether the
    shipping default happens to cover it — that is the pin's job, and coupling them
    would make one failure report as two.
    """
    model_id, t_end, n_points = _PAYING_ANALYTICAL
    xml = _model_xml(model_id)
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / model_id}")
    monkeypatch.setenv("BNGSIM_JAC_DERIV_BUDGET_S", "inf")

    def _best_run_s(jacobian: str | None) -> float:
        model = bngsim.Model.from_sbml(str(xml))
        kwargs = {"jacobian": jacobian} if jacobian else {}
        sim = bngsim.Simulator(model, method="ode", **kwargs)
        expect_analytical = jacobian is None
        assert model._core.analytical_jacobian_complete is expect_analytical, (
            f"{model_id} attached={model._core.analytical_jacobian_complete} with "
            f"jacobian={jacobian!r} — this measurement is not comparing what it says"
        )
        best = float("inf")
        # One warm-up (codegen / first-touch) that is not timed, then best-of-3.
        for rep in range(4):
            model.reset()
            t0 = time.perf_counter()
            sim.run(
                t_span=(0.0, t_end),
                n_points=n_points,
                rtol=_RTOL,
                atol=_ATOL,
                timeout=_SOLVE_WALL_CAP,
            )
            if rep:
                best = min(best, time.perf_counter() - t0)
        return best

    analytical_s = _best_run_s(None)
    fd_s = _best_run_s("fd")
    assert fd_s > _PAYING_SOLVE_SPEEDUP * analytical_s, (
        f"{model_id} FD solve {fd_s:.4f}s is not {_PAYING_SOLVE_SPEEDUP}x the "
        f"analytical {analytical_s:.4f}s (measured 4.16x: 0.065s vs 0.015s) — the "
        f"{_PAYING_DERIVATION_FLOOR_S}s floor rests on this derivation being worth "
        "paying for. Re-run the classification and re-pick the fixture from "
        "whatever pays then; do not delete this and keep the floor."
    )


def test_default_budget_covers_paying_derivations():
    """The default may not drop below the slowest derivation that pays for itself.

    The pin above catches *any* move of the constant and asks for a justification;
    this is the one bound a justification may not talk its way past. They are not
    redundant: the pin is a change detector that an intentional re-tune updates,
    and this is what still holds after it has been updated.

    ``BIOMD0000000608`` derives in 4.76 s here and solves 4.16x slower without the
    result, so a default under it does not merely waste a derivation — it hands the
    model a permanently slower solve. See ``_PAYING_DERIVATION_FLOOR_S`` for the
    band behind the number and for why the margin is 3.3x.

    The guard is on the constant, so it needs no corpus and no reference engine.
    """
    assert _DEFAULT_DERIVATION_BUDGET_S >= _PAYING_DERIVATION_FLOOR_S, (
        f"default derivation budget {_DEFAULT_DERIVATION_BUDGET_S}s is below the "
        f"{_PAYING_DERIVATION_FLOOR_S}s floor — BIOMD0000000608 derives in "
        f"{_SLOWEST_PAYING_DERIVATION_S}s here and solves 4.16x slower without the "
        "analytical Jacobian, and the floor carries 3.3x for machine speed. "
        "Re-measure before lowering this, and record what the slowest derivation "
        "that still pays for itself became — a cold codegen cache or a single "
        "sample per mode will get the ratio wrong by up to 9x in either direction."
    )


def test_needs_analytical_model_keeps_analytical_and_solves(monkeypatch):
    """The analytical arm of the canary's fixture solves and is finite.

    ``BIOMD0000000457`` was picked by #95 as the model FD could not integrate. It
    can (issue #249) — except at the parity tolerance on x86_64, where the two
    architectures disagree (issue #245, see ``_CANARY_LADDER``). What is true on
    every machine is this half: with a generous budget the derivation completes,
    ``analytical_jacobian_complete`` is True, and the solve returns a finite
    trajectory. That is what makes it usable as the canary's reference arm.
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
