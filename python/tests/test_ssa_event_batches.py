"""SSA/PSA same-instant event batches follow SBML L3v2 §4.11.6 (issues #761, #755).

#761: an event whose trigger another event's assignment turned true never fired
under SSA/PSA — not at that instant, not later — because the batch was a fixed
list and every caller then recorded the new rising edge as "already true".

#755: among simultaneous events of equal priority the lowest index always ran
first, so the last-declared writer won every replicate, where §4.11.6 (and
bngsim's ODE engine) pick at random.

The oracle is the semantics itself, checked against the ODE engine, which
implements the same drain (GH #242) and shares no event code with the SSA.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

METHODS = [
    pytest.param("ssa", {}, id="ssa"),
    pytest.param("psa", {"poplevel": 100}, id="psa"),
]

# A reaction keeps the SSA on its reaction path; without one it takes the
# zero-propensity idle path, which has its own call into the batch.
WITH_REACTION = "species A = 100; J: A => ; 0.01*A;\n"
IDLE = ""


def _final(ant, method, kw, t_end=3.0, seed=1):
    m = bngsim.Model.from_antimony_string(ant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
        r = bngsim.Simulator(m, method=method, **kw).run(t_span=(0, t_end), n_points=4, seed=seed)
    names = list(r.species_names)
    return {n: float(np.asarray(r.species)[-1, names.index(n)]) for n in names}


@pytest.mark.parametrize(("method", "kw"), METHODS)
@pytest.mark.parametrize("reaction", [WITH_REACTION, IDLE], ids=["reaction", "idle"])
def test_an_assignment_triggers_a_cascade(method, kw, reaction):
    """E1 at t = 1 sets x, which triggers E2, which triggers E3. Under SSA E2
    and E3 never fired (y = z = 0); the ODE fires all three."""
    ant = reaction + (
        "species x = 0; species y = 0; species z = 0;\n"
        "E1: at (time >= 1): x = 1;\n"
        "E2: at (x > 0.5): y = 1;\n"
        "E3: at (y > 0.5): z = 1;\n"
    )
    for seed in range(5):
        got = _final(ant, method, kw, seed=seed)
        assert (got["x"], got["y"], got["z"]) == (1.0, 1.0, 1.0), got
    ode = _final(ant, "ode", {})
    assert (ode["x"], ode["y"], ode["z"]) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_a_cascade_at_t_start(method, kw):
    """An initialValue=false event true at t = 0 fires in the t = 0 batch, and
    what it sets triggers the next one in the same batch."""
    ant = WITH_REACTION + (
        "species x = 0; species y = 0;\n"
        "E1: at (time >= 0), t0 = false: x = 1;\n"
        "E2: at (x > 0.5): y = 1;\n"
    )
    got = _final(ant, method, kw)
    assert (got["x"], got["y"]) == (1.0, 1.0), got


# Five on/off toggles of `sw` at one instant, highest priority first. Each "on"
# is a rising edge for the three watchers; each "off" is a falling edge.
TOGGLES = (
    "species sw = 0; species p = 0; species q = 0; species r = 0;\n"
    + "".join(
        f"at (time >= 0.99), priority = {pri}: sw = {2 if pri % 2 == 0 else 0};\n"
        for pri in range(10, 1, -1)
    )
    + (
        # persistent, values at execution: runs once per rising edge
        "at (sw > 1), priority = 1, persistent = true, fromTrigger = false: p = p + 1;\n"
        # persistent, values at trigger time: five runs, each writes 0 + 1
        "at (sw > 1), priority = 1, persistent = true, fromTrigger = true: q = q + 1;\n"
        # not persistent: cancelled by every "off" but the last "on"
        "at (sw > 1), priority = 1, persistent = false, fromTrigger = false: r = r + 3;\n"
    )
)


@pytest.mark.parametrize(("method", "kw"), METHODS)
@pytest.mark.parametrize("reaction", [WITH_REACTION, IDLE], ids=["reaction", "idle"])
def test_rising_and_falling_edges_within_one_instant(method, kw, reaction):
    """The watchers run p = 5, q = 1, r = 3, as the ODE does (the shape of SBML
    suite case 00978, which SSA answered 0, 0, 0)."""
    got = _final(reaction + TOGGLES, method, kw)
    assert (got["p"], got["q"], got["r"]) == (5.0, 1.0, 3.0), got
    ode = _final(reaction + TOGGLES, "ode", {})
    assert (ode["p"], ode["q"], ode["r"]) == (5.0, 1.0, 3.0), ode


TIE = WITH_REACTION + (
    "species z = 0;\n"
    "E1: at (time >= 1), priority = 1: z = 1;\n"
    "E2: at (time >= 1), priority = 1: z = 2;\n"
)


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_equal_priorities_are_ordered_at_random(method, kw):
    """Two events at one instant and one priority write z. The last to run
    wins, and which that is is a fair coin. SSA gave z = 2 on every seed."""
    n = 200
    wins = sum(_final(TIE, method, kw, seed=s)["z"] == 2.0 for s in range(n))
    # Binomial(200, 1/2): 4.5 standard deviations either side of 100.
    assert 68 <= wins <= 132, wins


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_the_tie_break_is_reproducible_per_seed(method, kw):
    runs = [[_final(TIE, method, kw, seed=s)["z"] for s in range(12)] for _ in range(2)]
    assert runs[0] == runs[1]
    assert set(runs[0]) == {1.0, 2.0}


DISTINCT = WITH_REACTION + (
    "species z = 0;\n"
    "E1: at (time >= 1), priority = 2: z = 1;\n"
    "E2: at (time >= 1), priority = 1: z = 2;\n"
)


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_distinct_priorities_stay_deterministic(method, kw):
    """With distinct priorities the order is fixed (higher first), whatever the
    seed: the random draw happens only at a genuine tie."""
    assert {_final(DISTINCT, method, kw, seed=s)["z"] for s in range(20)} == {2.0}


def _trajectory(ant, method, kw, seed):
    m = bngsim.Model.from_antimony_string(ant)
    r = bngsim.Simulator(m, method=method, **kw).run(t_span=(0, 3.0), n_points=31, seed=seed)
    return np.asarray(r.species)[:, list(r.species_names).index("A")]


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_a_tie_does_not_move_the_reaction_stream(method, kw):
    """The tie-break draws from a stream of its own: the reacting species run
    exactly as they do in the same model without the tie."""
    for seed in range(8):
        np.testing.assert_array_equal(
            _trajectory(TIE, method, kw, seed), _trajectory(DISTINCT, method, kw, seed)
        )


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_a_three_way_tie_is_uniform(method, kw):
    ant = WITH_REACTION + (
        "species z = 0;\n"
        "E1: at (time >= 1), priority = 1: z = 1;\n"
        "E2: at (time >= 1), priority = 1: z = 2;\n"
        "E3: at (time >= 1), priority = 1: z = 3;\n"
    )
    n = 300
    last = [_final(ant, method, kw, seed=s)["z"] for s in range(n)]
    for z in (1.0, 2.0, 3.0):
        k = last.count(z)
        # Binomial(300, 1/3): 4.5 standard deviations either side of 100.
        assert 63 <= k <= 137, (z, k)


@pytest.mark.parametrize(("method", "kw"), METHODS + [pytest.param("ode", {}, id="ode")])
def test_priorities_are_reevaluated_before_each_pick(method, kw):
    """E1 runs first and drops E2's priority from 3 to 0, below E3's 1, so E3
    runs before E2 and z ends at 1. Priorities read once at the start would run
    E2 before E3 and end at 2."""
    ant = WITH_REACTION + (
        "species s = 1; species z = 0;\n"
        "E1: at (time >= 1), priority = 5: s = 0;\n"
        "E2: at (time >= 1), priority = 3*s: z = 1;\n"
        "E3: at (time >= 1), priority = 1: z = 2;\n"
    )
    assert _final(ant, method, kw)["z"] == 1.0


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_a_reaction_triggers_a_cascade(method, kw):
    """The post-reaction sweep: A decaying past 50 fires E1, whose assignment
    fires E2 at the same instant."""
    ant = (
        "species A = 100; J: A => ; 0.5*A;\n"
        "species x = 0; species y = 0;\n"
        "E1: at (A < 50): x = 1;\n"
        "E2: at (x > 0.5): y = 1;\n"
    )
    for seed in range(5):
        got = _final(ant, method, kw, t_end=10.0, seed=seed)
        assert (got["x"], got["y"]) == (1.0, 1.0), got


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_a_trigger_on_a_rate_rule_target_is_not_swallowed(method, kw):
    """No reaction: the SSA jumps to the next event its time probe finds (t = 1,
    where E1's condition reads y frozen at 0). E2 rose on the way, because y
    moved, and was recorded as already true, so it never fired. It fires now.
    When it fires (at t = 1 rather than 0.5) is issue #751's."""
    ant = (
        "y' = 1; y = 0; species a = 0; species b = 0;\n"
        "E1: at (time >= 1 && y < 0.8): a = 1;\n"
        "E2: at (y > 0.5): b = 1;\n"
    )
    for seed in range(3):
        assert _final(ant, method, kw, seed=seed)["b"] == 1.0


LOOP = "species x = 0;\nE1: at (x < 0.5), t0 = false: x = 1;\nE2: at (x > 0.5): x = 0;\n"


@pytest.mark.parametrize(("method", "kw"), METHODS + [pytest.param("ode", {}, id="ode")])
def test_an_algebraic_loop_of_events_is_refused(method, kw):
    """Two events that re-arm each other at one instant never settle."""
    m = bngsim.Model.from_antimony_string(WITH_REACTION + LOOP)
    with pytest.raises(bngsim.SimulationError, match="CASCADE_LIMIT"):
        bngsim.Simulator(m, method=method, **kw).run(t_span=(0, 1.0), n_points=2, seed=1)


@pytest.mark.parametrize(("method", "kw"), METHODS + [pytest.param("ode", {}, id="ode")])
def test_a_nan_priority_is_refused(method, kw):
    """A priority that evaluates to NaN compares false with everything, and the
    order fell back to declaration order."""
    ant = WITH_REACTION + (
        "species z = 0; q = 0;\n"
        "E1: at (time >= 1), priority = q/q: z = 1;\n"
        "E2: at (time >= 1), priority = 5: z = 2;\n"
    )
    m = bngsim.Model.from_antimony_string(ant)
    with pytest.raises(bngsim.SimulationError, match="priority that is NaN"):
        bngsim.Simulator(m, method=method, **kw).run(t_span=(0, 3.0), n_points=4, seed=1)
