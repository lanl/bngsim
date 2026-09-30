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
    n = 60
    wins = sum(_final(TIE, method, kw, seed=s)["z"] == 2.0 for s in range(n))
    # Binomial(60, 1/2): 4.5 standard deviations either side of 30.
    assert 13 <= wins <= 47, wins


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_the_tie_break_is_reproducible_per_seed(method, kw):
    runs = [[_final(TIE, method, kw, seed=s)["z"] for s in range(12)] for _ in range(2)]
    assert runs[0] == runs[1]
    assert set(runs[0]) == {1.0, 2.0}


def test_distinct_priorities_stay_deterministic():
    """With distinct priorities the order is fixed (higher first), whatever the
    seed: the random draw happens only at a genuine tie."""
    ant = WITH_REACTION + (
        "species z = 0;\n"
        "E1: at (time >= 1), priority = 2: z = 1;\n"
        "E2: at (time >= 1), priority = 1: z = 2;\n"
    )
    assert {_final(ant, "ssa", {}, seed=s)["z"] for s in range(20)} == {2.0}
