"""Issue #375 — telling apart two switch times that flip at the same instant.

A switch-time parameter's whole gradient is the jump ``s⁺ = s⁻ + (f⁻−f⁺)·∂t*/∂p``
at its crossing (issue #48), and the core reads ``f⁻``/``f⁺`` by nudging the
*clock* a few ulp either side of the threshold. That reads the whole RHS, so
every condition thresholding that clock at that value flips together and the
difference is their **sum**.

#375 is what that costs when two switch times are set to the same number. The
detector keyed crossings on the threshold's *value*, so it merged them into one
record whose ``∂t*/∂p`` was the union of both, and the core then charged each
parameter with the other's jump. On two independent ramps:

    X: if(time() >= tA, 0, kA)        Y: if(time() >= tB, 0, kB)

    tA = tB = 1.0  ->  [[2, 2], [5, 5]]     (exact: [[2, 0], [0, 5]])
    tA=1.0, tB=1.3 ->  [[2, 0], [0, 5]]     exact

Nothing was logged, and on the corpus (`BIOMD0000000075`, `BIOMD0000000161`,
which each ship three stimulus onsets at `tau = 0.05`) the spurious entries
largely cancel along the conservation chain — so a sum over species looks right
while individual columns are two orders of magnitude out.

**The fix is in what "distinct" means and in how the jump is measured.**

* Crossings are keyed on ``∂threshold/∂primary``, not on the threshold's value.
  One threshold gating six rate laws still collapses to one crossing — that merge
  is load-bearing, or its jump is applied six times — while two thresholds that
  merely share a number stay apart.
* Splitting the records is not enough on its own, and this is the part worth
  keeping in mind: each would still read the same clock-nudged ``f⁻ − f⁺`` and
  get the same combined jump. So a coinciding crossing is separated by moving its
  *threshold* instead: raise it by a hair with the clock held on the after side,
  and that condition alone falls back to its before-branch. The difference is
  that crossing's own jump, and the core's existing per-instant sum is correct.
* The parameter raised has to be one no coinciding threshold reads. Where none
  exists the crossings are genuinely inseparable this way and bngsim refuses.

Crossings that no *requested* parameter moves are detected and kept too. They
emit no column, but they still flip at that instant and still contaminate
``f⁻``, which is why #375 reproduced even with a single parameter requested.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder
from bngsim._exceptions import SensitivityUnsupportedError
from bngsim._switch_sensitivity import _ISOLATION_MIN_ULP, _q, compute_switch_time_sens

# ─── N independent ramps, each stopping at its own switch time ───────────────
# dXi/dt = if(time() >= ti, 0, ki)  ⇒  Xi(t) = ki·min(t, ti)
#
# Nothing couples them, so past every switch the exact sensitivity matrix is
# diagonal: ∂Xi/∂tj = ki·δij. Any off-diagonal entry is one ramp being charged
# with another's jump, which is #375 exactly and needs no tolerance to see.
_RATES = (2.0, 5.0, 11.0)
_RUN = dict(rtol=1e-12, atol=1e-14, max_steps=10**7)
_TS = [0.0, 0.5, 1.5, 2.0]


def _ramps(*switch_times):
    b = ModelBuilder()
    for i, t in enumerate(switch_times):
        b.add_parameter(f"t{i}", t)
        b.add_parameter(f"k{i}", _RATES[i])
    idx = [b.add_species(f"X{i}()", 0.0) for i in range(len(switch_times))]
    for i in range(len(switch_times)):
        b.add_function(f"rate{i}", f"if(time()>=t{i},0,k{i})")
        b.add_reaction([], [idx[i]], "functional", f"rate{i}")
    return bngsim.Model(_core=b.build()), idx


def _matrix(*switch_times, params=None):
    """Final ``∂Xi/∂tj`` over the requested parameter columns."""
    model, idx = _ramps(*switch_times)
    cols = params if params is not None else [f"t{i}" for i in range(len(switch_times))]
    r = bngsim.Simulator(model, method="ode", sensitivity_params=cols).run(
        sample_times=_TS, **_RUN
    )
    S = np.asarray(r.sensitivities)
    return np.array([[S[-1, i, c] for c in range(len(cols))] for i in idx])


def _records(*switch_times, params=None):
    model, _ = _ramps(*switch_times)
    cols = params if params is not None else [f"t{i}" for i in range(len(switch_times))]
    return compute_switch_time_sens(model._core, cols, 0.0, 2.0)[0]


class TestTheReportedMatrix:
    """#375's reproducer, and the control arm that always worked."""

    def test_coincident_switch_times_are_diagonal(self):
        # Was [[2, 2], [5, 5]].
        np.testing.assert_allclose(_matrix(1.0, 1.0), np.diag(_RATES[:2]), atol=1e-9)

    def test_distinct_switch_times_are_diagonal(self):
        # The path #375 never touched; it must stay byte-for-byte as good.
        np.testing.assert_allclose(_matrix(1.0, 1.3), np.diag(_RATES[:2]), atol=1e-9)

    def test_three_coincident_switch_times_are_diagonal(self):
        # Each crossing is isolated against BOTH of its neighbours, so the two
        # off-diagonal entries per row have to vanish independently — a fix that
        # only ever subtracted one coinciding jump would pass the pair above and
        # fail here.
        np.testing.assert_allclose(_matrix(1.0, 1.0, 1.0), np.diag(_RATES), atol=1e-9)

    def test_a_partial_coincidence_is_diagonal(self):
        # Two switches share an instant, the third does not: the isolated and
        # non-isolated paths run in the same integration.
        np.testing.assert_allclose(_matrix(1.0, 1.0, 1.4), np.diag(_RATES), atol=1e-9)

    def test_one_requested_parameter_is_not_charged_the_other_jump(self):
        # The case that made #375 reproduce on a single-parameter request:
        # `t1`'s crossing emits no column, so an implementation that only looked
        # at the requested columns would see no coincidence at all — while the
        # RHS difference it flips is still in f⁻.
        col = _matrix(1.0, 1.0, params=["t0"])
        assert col[0, 0] == pytest.approx(_RATES[0], rel=1e-9)
        assert col[1, 0] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("switch_times", [(1.0, 1.0), (1.0, 1.3)])
    def test_matches_a_central_finite_difference(self, switch_times):
        # An oracle that shares no code with the sensitivity path. max_step keeps
        # a plain run from stepping over a switch it gets no stop for.
        h = 1e-3
        cols = []
        for j in range(len(switch_times)):
            hi, lo = list(switch_times), list(switch_times)
            hi[j] += h
            lo[j] -= h
            ends = []
            for pt in (hi, lo):
                model, idx = _ramps(*pt)
                r = bngsim.Simulator(model, method="ode").run(
                    sample_times=_TS, max_step=0.01, **_RUN
                )
                ends.append(np.asarray(r.species)[-1, idx])
            cols.append((ends[0] - ends[1]) / (2 * h))
        np.testing.assert_allclose(_matrix(*switch_times), np.stack(cols, axis=1), atol=1e-6)


class TestWhatCountsAsOneCrossing:
    """Keyed on ∂threshold/∂primary, so a shared *value* no longer merges."""

    def test_coincident_crossings_emit_separate_one_hot_records(self):
        recs = _records(1.0, 1.0)
        assert len(recs) == 2
        assert [r.dtstar for r in recs] == [[1.0, 0.0], [0.0, 1.0]]
        # ...and each carries the bump that takes it alone off the instant.
        assert all(len(r.isolate_param_idx0) == 1 for r in recs)
        assert {r.isolate_param_idx0[0] for r in recs} == {
            0,
            2,
        }  # t0 and t1's parameter slots
        assert all(d > 0.0 for r in recs for d in r.isolate_delta)

    def test_distinct_crossings_carry_no_isolation(self):
        # The untouched path: nothing coincides, so nothing is perturbed and the
        # core reads the plain f⁻ − f⁺ exactly as issue #48 shipped it.
        recs = _records(1.0, 1.3)
        assert len(recs) == 2
        assert all(r.isolate_param_idx0 == [] and r.isolate_delta == [] for r in recs)

    def test_one_threshold_in_many_rate_laws_stays_one_crossing(self):
        # The merge #375's fix must NOT undo: `t>=t0` gating six functions in
        # Lin2021 is one crossing, stopped at and jumped across once. Splitting
        # it would apply the jump twice, which is the failure the value-keyed
        # merge existed to prevent.
        b = ModelBuilder()
        for name, value in (("t0", 1.0), ("kA", 2.0), ("kB", 5.0)):
            b.add_parameter(name, value)
        x_idx = b.add_species("X()", 0.0)
        y_idx = b.add_species("Y()", 0.0)
        b.add_function("rateX", "if(time()>=t0,0,kA)")
        b.add_function("rateY", "if(time()>=t0,0,kB)")
        b.add_reaction([], [x_idx], "functional", "rateX")
        b.add_reaction([], [y_idx], "functional", "rateY")
        model = bngsim.Model(_core=b.build())

        recs = compute_switch_time_sens(model._core, ["t0"], 0.0, 2.0)[0]
        assert len(recs) == 1
        assert recs[0].isolate_param_idx0 == []
        # X = kA·min(t,t0), Y = kB·min(t,t0): one threshold, both columns.
        r = bngsim.Simulator(model, method="ode", sensitivity_params=["t0"]).run(
            sample_times=_TS, **_RUN
        )
        S = np.asarray(r.sensitivities)
        assert S[-1, x_idx, 0] == pytest.approx(2.0, rel=1e-9)
        assert S[-1, y_idx, 0] == pytest.approx(5.0, rel=1e-9)

    def test_a_hard_coded_twin_still_turns_isolation_on(self):
        # The coinciding condition need not be fitted at all. A literal `1.0`
        # threshold has no partials, so it emits no record — but it flips at the
        # same instant and its jump is in f⁻ just the same.
        b = ModelBuilder()
        for name, value in (("tA", 1.0), ("kA", 2.0), ("kB", 5.0)):
            b.add_parameter(name, value)
        x_idx = b.add_species("X()", 0.0)
        y_idx = b.add_species("Y()", 0.0)
        b.add_function("rateX", "if(time()>=tA,0,kA)")
        b.add_function("rateY", "if(time()>=1.0,0,kB)")
        b.add_reaction([], [x_idx], "functional", "rateX")
        b.add_reaction([], [y_idx], "functional", "rateY")
        model = bngsim.Model(_core=b.build())

        recs = compute_switch_time_sens(model._core, ["tA"], 0.0, 2.0)[0]
        assert len(recs) == 1
        assert recs[0].isolate_param_idx0 == [0]
        r = bngsim.Simulator(model, method="ode", sensitivity_params=["tA"]).run(
            sample_times=_TS, **_RUN
        )
        S = np.asarray(r.sensitivities)
        assert S[-1, x_idx, 0] == pytest.approx(2.0, rel=1e-9)
        assert S[-1, y_idx, 0] == pytest.approx(0.0, abs=1e-9)

    def test_a_coinciding_switch_with_no_jump_of_its_own_is_fine(self):
        # A condition can flip with no effect on the RHS at all: `if(t>=tB, 0,
        # kB*Zobs)` at a crossing where Zobs is still zero has both branches
        # evaluating to 0. Its isolated difference is then exactly zero — the
        # right answer, since that crossing genuinely contributes no jump, while
        # the one it coincides with contributes a real one.
        #
        # Pinned because the first draft of the isolation read an unchanged RHS
        # as a failed bump and raised, which turned this ordinary model into a
        # crashed run — a worse answer than the merged jump #375 was filed for.
        b = ModelBuilder()
        for name, value in (("tA", 1.0), ("tB", 1.0), ("kA", 2.0), ("kB", 5.0)):
            b.add_parameter(name, value)
        x_idx = b.add_species("X()", 0.0)
        z_idx = b.add_species("Z()", 0.0)
        b.add_observable("Zobs", [(z_idx, 1.0)])
        b.add_function("rateX", "if(time()>=tA,0,kA)")
        b.add_function("rateZ", "if(time()>=tB,0,kB*Zobs)")
        b.add_reaction([], [x_idx], "functional", "rateX")
        b.add_reaction([], [z_idx], "functional", "rateZ")
        model = bngsim.Model(_core=b.build())

        r = bngsim.Simulator(model, method="ode", sensitivity_params=["tA", "tB"]).run(
            sample_times=_TS, **_RUN
        )
        S = np.asarray(r.sensitivities)
        assert S[-1, x_idx, 0] == pytest.approx(2.0, rel=1e-9)  # X keeps its own jump
        assert S[-1, x_idx, 1] == pytest.approx(0.0, abs=1e-9)  # and not tB's
        # Z never leaves zero, so nothing moves it either way.
        assert S[-1, z_idx, 0] == pytest.approx(0.0, abs=1e-9)
        assert S[-1, z_idx, 1] == pytest.approx(0.0, abs=1e-9)


class TestTheIsolationStep:
    """Which parameter is raised, and how far."""

    def test_the_step_clears_a_neighbouring_threshold(self):
        # A third switch 1e-9 away is a DIFFERENT instant, and the bump must not
        # reach it: raising t0 past 1+1e-9 would flip a condition t0 does not
        # own, re-introducing the contamination from the other side. The step is
        # capped at a quarter of the gap, so the ceiling is that quarter and not
        # the default 1e-6·span it would otherwise take.
        neighbour = 1.0 + 1e-9
        quarter_gap = 0.25 * (neighbour - 1.0)
        recs = _records(1.0, 1.0, neighbour)
        bumps = [r.isolate_delta[0] for r in recs if r.isolate_delta]
        assert len(bumps) == 2, "the coincident crossings carry no isolation step"
        assert all(0.0 < d <= quarter_gap for d in bumps), (bumps, quarter_gap)

    def test_the_quantisation_leaves_room_for_the_isolation_step(self):
        # _isolation_bump refuses when the step would sink into the threshold's
        # last bits. That branch is unreachable only because _q groups thresholds
        # on 12 significant digits: the closest one it calls *different* is far
        # enough away that a quarter of the gap still clears the floor. Widening
        # _q would silently start producing steps that flip nothing, and a jump
        # read from two identical RHS evaluations is an exact zero — so the
        # margin is asserted here rather than left as a property nobody checks.
        span = 1.0
        gap = next(d for d in (10.0**-e for e in range(6, 17)) if _q(span + d) != _q(span))
        floor = _ISOLATION_MIN_ULP * np.finfo(float).eps * span
        assert 0.25 * gap > floor, (
            f"_q resolves thresholds {gap:g} apart, whose quarter-gap isolation step "
            f"{0.25 * gap:g} no longer clears the {floor:g} floor"
        )

    def test_a_coincidence_with_no_private_parameter_is_refused(self):
        # `t0` and `t0 + gap` with gap = 0 land together, and every parameter t0's
        # threshold depends on is also read by the other. Raising t0 moves both,
        # so there is no step that isolates it — bngsim says so rather than
        # returning the merged jump #375 reported.
        b = ModelBuilder()
        for name, value in (("t0", 1.0), ("gap", 0.0), ("kA", 2.0), ("kB", 5.0)):
            b.add_parameter(name, value)
        x_idx = b.add_species("X()", 0.0)
        y_idx = b.add_species("Y()", 0.0)
        b.add_function("rateX", "if(time()>=t0,0,kA)")
        b.add_function("rateY", "if(time()>=(t0+gap),0,kB)")
        b.add_reaction([], [x_idx], "functional", "rateX")
        b.add_reaction([], [y_idx], "functional", "rateY")
        model = bngsim.Model(_core=b.build())

        with pytest.raises(SensitivityUnsupportedError, match="issue #375"):
            compute_switch_time_sens(model._core, ["t0", "gap"], 0.0, 2.0)

    def test_the_same_pair_is_supported_once_the_switch_times_differ(self):
        # The refusal above is about the coincidence, not about the shape: give
        # `gap` a non-zero value and the identical model is answered exactly.
        b = ModelBuilder()
        for name, value in (("t0", 1.0), ("gap", 0.3), ("kA", 2.0), ("kB", 5.0)):
            b.add_parameter(name, value)
        x_idx = b.add_species("X()", 0.0)
        y_idx = b.add_species("Y()", 0.0)
        b.add_function("rateX", "if(time()>=t0,0,kA)")
        b.add_function("rateY", "if(time()>=(t0+gap),0,kB)")
        b.add_reaction([], [x_idx], "functional", "rateX")
        b.add_reaction([], [y_idx], "functional", "rateY")
        model = bngsim.Model(_core=b.build())

        r = bngsim.Simulator(model, method="ode", sensitivity_params=["t0", "gap"]).run(
            sample_times=_TS, **_RUN
        )
        S = np.asarray(r.sensitivities)
        # X stops at t0; Y stops at t0+gap, so it moves with both.
        np.testing.assert_allclose(
            [[S[-1, x_idx, 0], S[-1, x_idx, 1]], [S[-1, y_idx, 0], S[-1, y_idx, 1]]],
            [[2.0, 0.0], [5.0, 5.0]],
            atol=1e-9,
        )
