"""The read-ahead at each output row evaluates only what the triggers read
(issue #1278).

#1133's read-ahead reads the event triggers half a root tolerance past every
output row, and synced the whole model there to do it: every observable and
every function, for a handful of triggers. On a 20001-row grid that was
1.15-1.34x the run, and almost no row's read-ahead fires. It now refreshes the
observables and functions the triggers read, directly or through one another,
and syncs the model whole only where a trigger rises, before anything fires.

These are the rows where a trigger does cross, as in
``test_state_trigger_on_a_sample.py``, with the trigger reading the state each
way the refresh has to follow: through a chain of functions, through an
observable, through rateOf (which takes the whole sync), and an event whose
assignment reads a function the trigger does not.

Oracle: q = t, so the event is at t = 2.5, a few doubles after the row CVODE
reaches; the row holds the event's state.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_GRID = [0.0, 1.0, 2.0, 2.5, 3.0]


def _at_row(text: str, name: str, t: float = 2.5, **kw) -> float:
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(sample_times=_GRID, **kw)
    return float(np.asarray(r.species)[_GRID.index(t), list(r.species_names).index(name)])


@pytest.mark.parametrize(
    "trigger",
    [
        pytest.param("q >= 2.5", id="the state itself"),
        pytest.param("g >= 5", id="a chain of functions"),
        pytest.param("2*h >= 5", id="one function"),
        pytest.param("q + 0*rateOf(q) >= 2.5", id="rateOf (the whole sync)"),
    ],
)
def test_the_row_on_the_crossing_holds_the_event(trigger):
    """Read through what the trigger reads, the crossing a few doubles after
    the row still fires at it."""
    text = (
        "species S = 3; q = 0; q' = 1; h := q; g := 2*h;"
        f" pad1 := 7*S; pad2 := pad1 + 1; E1: at ({trigger}): S = 1;"
    )
    assert _at_row(text, "S") == 1.0


def test_a_trigger_read_through_an_observable():
    """A species fed by a reaction: in an SBML model its token is the
    same-named observable's total, not the concentration, so the refresh
    has to sum that observable."""
    text = "species Q = 0; species S = 3; J1: -> Q; 1; E1: at (Q >= 2.5): S = 1;"
    assert _at_row(text, "S") == 1.0


def test_an_assignment_reads_what_the_trigger_does_not():
    """``S = w`` reads w := 10*q, which the trigger does not: where the trigger
    rises, the model is synced whole before the event fires, so w is read at
    the crossing (25), not left as the last root evaluation had it."""
    text = "species S = 3; q = 0; q' = 1; w := 10*q; E1: at (q >= 2.5): S = w;"
    assert _at_row(text, "S") == pytest.approx(25.0, rel=1e-12)


def test_a_window_read_through_functions_is_read_fresh():
    """A trigger true only in a narrow window around the crossing, read
    through a chain of functions. Their values as the last evaluation left
    them (where CVODE's step ended, past the row) read false; only the chain
    refreshed at the state ahead, h before g, reads true, and fires at the row.
    The window is narrower than a step, so no later root search finds it."""
    text = (
        "species S = 3; q = 0; q' = 1; h := q; g := 2*h; E1: at (g >= 5 && g <= 5.000001): S = 1;"
    )
    assert _at_row(text, "S") == 1.0


def test_an_assignment_reads_a_function_the_rates_read_too():
    """``S = w``, with w := 10*q also read by a rate law, so it is no deferred
    function an event refreshes for itself: where the trigger rises, the
    model is synced whole first, and w is read at the crossing (25)."""
    text = (
        "species S = 3; species X = 0; q = 0; q' = 1; w := 10*q;"
        " J1: -> X; 1e-30*w; E1: at (q >= 2.5): S = w;"
    )
    assert _at_row(text, "S") == pytest.approx(25.0, rel=1e-12)


def test_no_row_fires_where_no_trigger_crosses():
    """Control: the triggers read false at every row ahead, as before."""
    text = "species S = 3; q = 0; q' = 1; h := q; g := 2*h; E1: at (g >= 50): S = 1;"
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 10), n_points=2001)
    assert np.all(np.asarray(r.species)[:, list(r.species_names).index("S")] == 3.0)


# --------------------------------------------------------------------------- #
# the plan itself, through the core's test hooks
# --------------------------------------------------------------------------- #
_CHAIN = (
    "species S = 3; q = 0; q' = 1; h := q; g := 2*h; pad1 := 7*S; pad2 := pad1 + 1;"
    " E1: at (g >= 5): S = 1;"
)


def _core(text: str):
    return bngsim.Model.from_antimony_string(text)._core


def test_the_plan_follows_a_chain_of_functions_and_nothing_else():
    """g reads h: both are refreshed. pad1 and pad2 are not read by any
    trigger, and the read-ahead leaves them to the row's own refresh."""
    plan = _core(_CHAIN)._trigger_read_plan()
    assert plan["full"] is False
    assert sorted(plan["functions"]) == ["g", "h"]


def test_the_plans_values_are_a_whole_evaluations():
    """At a state none of them were last evaluated at, every function the plan
    refreshes reads what a whole evaluation gives there."""
    core = _core(_CHAIN)
    plan = core._trigger_read_plan()
    names = list(core.function_names)
    state = [11.0, 2.75]  # S, q
    planned, whole = core._trigger_read_plan_values(1.25, state)
    for name in plan["functions"]:
        i = names.index(name)
        assert planned[i] == whole[i], name
    assert planned[names.index("g")] == 5.5


def test_a_trigger_on_a_species_plans_its_observable():
    core = _core("species Q = 0; species S = 3; J1: -> Q; 1; E1: at (Q >= 2.5): S = 1;")
    plan = core._trigger_read_plan()
    assert plan["full"] is False and plan["observables"] == ["Q"]


def test_a_rateof_read_takes_the_whole_sync():
    """A rateOf accessor is refreshed only by a derivative probe."""
    plan = _core(
        "species S = 3; q = 0; q' = 1; E1: at (q + 0*rateOf(q) >= 2.5): S = 1;"
    )._trigger_read_plan()
    assert plan["full"] is True


def test_a_table_function_takes_the_whole_sync():
    """A table function reads its index out of sight of the references."""
    core = _core(_CHAIN)
    core.add_table_function_arrays("tab", [0.0, 10.0], [0.0, 1.0])
    assert core._trigger_read_plan()["full"] is True
