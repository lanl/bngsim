"""Issue #182 — ``solver_stats`` must count the whole run, not its last segment.

Every ``CVodeGetNum*`` counter counts from CVODE's last (re-)initialization, and
the event path re-initializes at each fire. ``record_solver_stats`` sampled them
once at the end, so what it reported was the segment after the LAST fire: an
under-count on any model with events, and a flat ``n_steps=0`` when the last
fire lands on the final instant, which is what the issue was found on
(``Smith_BMCSystBiol2013`` with its dose time moved onto ``t_end``). The
trajectory was correct throughout — only the counters moved, which is what made
it expensive to diagnose.

The model here is exponential decay with bolus doses, whose step count is set by
the decay time constant rather than by the state's magnitude, so adding doses
does not by itself make the integration cheaper: every counter for a dosed run
is expected to stay at or above the same model's undosed count.
"""

from __future__ import annotations

import pytest
from bngsim._bngsim_core import CvodeSimulator, ModelBuilder, NetworkModel, TimeSpec

# Counters sourced from the CVodeGetNum* family. err_test_fails and
# nonlin_conv_fails come from it too, but a well-behaved run may legitimately
# report zero of either, so they carry no positivity claim.
COUNTERS = ("n_steps", "n_rhs_evals", "n_jac_evals", "n_nonlin_iters")

T_END = 10.0

# Past T_END: the events are built and rooted exactly as in the firing runs,
# they simply never trigger. That keeps the comparison about the counters
# rather than about the presence of a root function.
NEVER = 20.0


def _decay_with_doses(dose_times: tuple[float, ...]) -> NetworkModel:
    """dS/dt = -k·S from S(0)=100, plus a ``S := S + 50`` event per dose time."""
    b = ModelBuilder()
    b.add_parameter("k", 0.5)
    s = b.add_species("S", 100.0)
    b.add_reaction([s], [], "elementary", "k")
    for i, t in enumerate(dose_times):
        b.add_event(f"dose{i}", f"time() >= {t}", [(s, "S + 50")])
    return b.build()


def _run(sim: CvodeSimulator, t_end: float = T_END) -> dict[str, float]:
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = 11
    return sim.run(ts).solver_stats.to_dict()


def _stats(dose_times: tuple[float, ...], t_end: float = T_END) -> dict[str, float]:
    return _run(CvodeSimulator(_decay_with_doses(dose_times)), t_end)


def test_a_fire_at_the_final_instant_still_counts_the_run():
    """The issue's reproduction: the last fire closes an empty segment.

    With the trigger at ``t_end`` the post-fire segment is a single instant, so
    the end-of-run sample read zeros out of CVODE — for a run that had just
    integrated the entire span. The run before the fire is the whole run here,
    so its counts must match the never-firing model's.
    """
    fired = _stats((T_END,))
    never = _stats((NEVER,))

    for name in COUNTERS:
        assert fired[name] > 0, f"{name} reads {fired[name]} for a run that integrated to t_end"
        # The two integrate the same trajectory over the same span, and measure
        # identical here (127 steps each). The tolerance is for a root located a
        # hair inside t_end elsewhere, which splits off a short second segment.
        assert fired[name] == pytest.approx(never[name], rel=0.2)


def test_the_work_before_every_fire_is_counted():
    """Four mid-run doses: the reported cost may not fall below the undosed run.

    Sampling once at the end returned only the span after the last dose: 51 of
    the 252 steps this run takes, which reads as a cheap run rather than as a
    broken counter. Each restart drops CVODE to first order and costs steps, so
    the dosed run is if anything the more expensive of the two — 252 against the
    undosed 127.
    """
    dosed = _stats((2.0, 4.0, 6.0, 8.0))
    undosed = _stats((NEVER,))

    for name in COUNTERS:
        assert dosed[name] >= undosed[name], (
            f"{name}: dosed run reports {dosed[name]} against {undosed[name]} undosed — "
            "the segments before the last fire are missing"
        )


def test_the_counters_do_not_carry_across_runs():
    """The banked segments are per-run: a repeated run repeats its counts.

    They live on the simulator, so a missing reset would have each run report
    its own cost plus every earlier run's. ``reset()`` puts the species back on
    their initial concentrations, which is what makes the second run the same
    integration as the first (a run leaves the model on its final state).
    """
    model = _decay_with_doses((2.0, 4.0, 6.0, 8.0))
    sim = CvodeSimulator(model)
    first = _run(sim)
    model.reset()
    second = _run(sim)

    for name in COUNTERS:
        assert second[name] == first[name]
