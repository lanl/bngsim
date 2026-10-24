"""An event whose trigger on the state crosses on an output sample fires in
that sample's row (issue #1124, item 2).

``q' = 1; at (q >= 2.5): S = 1`` with a sample at 2.5: CVODE reaches
q(2.5) = 2.4999999999999996, a few doubles short, so the trigger read false
at the row, the root was found a few doubles after it, and the row held the
state from before the event ([S] = 3; libRoadRunner and SSA give 1). The root
search places a crossing no closer than its own tolerance, so a root that
near is not after the row by anything the run can tell. The triggers at an
output are now also read with the state half that tolerance ahead, and the
clock at the row.

A run with sensitivity columns does the same (issue #1182), and takes the
event's jump in its columns there as at a located crossing: the row is t*,
within half a root tolerance of it. It had waited for the next step's root
search, and recorded the row before the event that the run without columns
recorded after it.

A trigger on the clock is exact, and keeps #950's answer: an event a double
after a sample leaves that row before it.

Oracles: closed forms. q = t, so the event is at t = 2.5; its count of
executions; the columns' closed forms through t* = 2.5/k; and for the clock,
the order of two doubles.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

_RISE = "species S = 3; q = 0; q' = 1; E1: at (q {op} 2.5): S = 1;"

_GRIDS = {
    "a_sample_on_it": [0.0, 1.0, 2.0, 2.5, 3.0],
    "only_it": [0.0, 2.5],
    "half_steps": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "linspace": list(np.linspace(0.0, 5.0, 11)),
}


def _column(text, times, name, params=None, **kw):
    model = bngsim.Model.from_antimony_string(text)
    for key, value in (params or {}).items():
        model.set_param(key, value)
    r = bngsim.Simulator(model, method="ode").run(sample_times=times, **kw)
    return [float(v) for v in np.asarray(r.species)[:, list(r.species_names).index(name)]]


def _row(times, t):
    return [float(x) for x in times].index(t)


@pytest.mark.parametrize("grid", sorted(_GRIDS))
def test_the_row_on_the_crossing_holds_the_event(grid):
    """The issue's repro, on four grids. Each was 3 at t = 2.5."""
    times = _GRIDS[grid]
    s = _column(_RISE.format(op=">="), times, "S")
    at = _row(times, 2.5)
    assert s[:at] == [3.0] * at
    assert s[at:] == [1.0] * (len(times) - at)


def test_it_agrees_with_ssa():
    times = _GRIDS["a_sample_on_it"]
    model = bngsim.Model.from_antimony_string(_RISE.format(op=">="))
    r = bngsim.Simulator(model, method="ssa").run(sample_times=times, seed=1)
    ssa = [float(v) for v in np.asarray(r.species)[:, list(r.species_names).index("S")]]
    assert _column(_RISE.format(op=">="), times, "S") == ssa


def test_it_fires_once():
    """Fired at the row, the root a few doubles later is behind the run."""
    text = "species n = 0; q = 0; q' = 1; E1: at (q >= 2.5): n = n + 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0, 4.0], "n") == [0, 0, 0, 1, 1, 1]


def test_a_falling_trigger():
    text = "species S = 3; q = 5; q' = -1; E1: at (q <= 2.5): S = 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S") == [3, 3, 3, 1, 1]


def test_a_delayed_execution_runs_its_delay_from_the_row():
    text = "species S = 3; q = 0; q' = 1; E1: at 1 after (q >= 2.5): S = 1;"
    assert _column(text, [0.0, 2.5, 3.4, 3.5, 4.0], "S") == [3, 3, 3, 1, 1]


def test_at_a_tight_tolerance_as_well():
    s = _column(_RISE.format(op=">="), _GRIDS["a_sample_on_it"], "S", rtol=1e-10, atol=1e-12)
    assert s == [3, 3, 3, 1, 1]


def test_an_event_on_the_row_that_turns_the_state_back_is_read_ahead_from():
    """An event on a stop at the sample turns q back before it reaches 2.5.

    q tops out a few doubles short of 2.5 and falls, so E1 never fires. Read
    ahead on the derivative from before E0 (q' = +1), q crossed 2.5 and E1 fired
    in the row.
    """
    text = (
        "species S = 3; q = 0; k = 1; q' = k;"
        " E0: at (time >= 2.5): k = -1; E1: at (q >= 2.5): S = 1;"
    )
    for times in ([0.0, 1.0, 2.0, 2.5, 3.0], [0.0, 2.5, 3.0]):
        assert _column(text, times, "S") == [3.0] * len(times)
        assert max(_column(text, times, "q")) < 2.5


def test_a_crossing_after_the_sample_by_more_than_rounding_is_after_it():
    """1e-9 after the sample is far past the root tolerance: the row is 3."""
    text = "species S = 3; q = 0; q' = 1; E1: at (q >= 2.5 + 1e-9): S = 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S") == [3, 3, 3, 3, 1]


@pytest.mark.parametrize("ulps", [1, 8])
def test_a_clock_trigger_a_double_after_the_sample_is_after_it(ulps):
    """Only the state is read ahead: #950's row before the event stands."""
    tau = 2.5
    for _ in range(ulps):
        tau = math.nextafter(tau, math.inf)
    text = "species S = 3; tau = 0; E1: at (time >= tau): S = 1;"
    s = _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S", params={"tau": tau})
    assert s == [3, 3, 3, 3, 1]


# A run with sensitivity columns fires at the row too, with the jump of a
# located crossing (issue #1182). Before, its fire waited for the next step's
# root search, a few doubles later, and the row held the state from before the
# event. `S' = 1` and `S = S/2` at q = k·t = 2.5 give closed forms: t* = 2.5/k,
# so past it S = (3 + t*)/2 + (t - t*), dS/dk = 2.5/(2k²) = 1.25 at k = 1,
# dS/dq0 = 0.5 (t* = (2.5 - q0)/k) and dS/dS0 = 0.5. Before it, 0, 0 and 1.
_HALVE = "species S = 3; q = 0; k = 1; q' = k; S' = 1; E1: at (q {op} 2.5): S = S/2;"


def _sens_run(text, times, **kw):
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode", **kw).run(sample_times=times)
    j = list(r.species_names).index("S")
    s = [float(v) for v in np.asarray(r.species)[:, j]]
    return r, j, s


@pytest.mark.parametrize("grid", sorted(_GRIDS))
def test_a_sensitivity_run_agrees_with_a_plain_run_at_the_row(grid):
    """The same model and grid, with and without sensitivities, at the crossing row."""
    text = "species S = 3; q = 0; k = 1; q' = k; E1: at (q >= 2.5): S = 1;"
    times = _GRIDS[grid]
    _, _, s = _sens_run(text, times, sensitivity_params=["k"])
    assert s == _column(text, times, "S")
    assert s == [3.0 if t < 2.5 else 1.0 for t in times]


@pytest.mark.parametrize("op", [">=", ">"])
def test_a_sensitivity_run_jumps_its_columns_at_the_row(op):
    text = _HALVE.format(op=op)
    times = [0.0, 1.0, 2.0, 2.5, 3.0]
    r, j, s = _sens_run(text, times, sensitivity_params=["k"], sensitivity_ic=["q", "S"])
    assert s == _column(text, times, "S")
    np.testing.assert_allclose(s, [3, 4, 5, 2.75, 3.25], rtol=1e-6)
    after = np.array(times) >= 2.5
    dk = np.asarray(r.sensitivities)[:, j, 0]
    np.testing.assert_allclose(dk, np.where(after, 1.25, 0.0), atol=1e-6)
    dic = np.asarray(r.sensitivities_ic)[:, j, :]
    np.testing.assert_allclose(dic[:, 0], np.where(after, 0.5, 0.0), atol=1e-6)
    np.testing.assert_allclose(dic[:, 1], np.where(after, 0.5, 1.0), atol=1e-6)


def test_a_sensitivity_run_from_a_later_start_fires_at_the_row():
    """From t = 1 the crossing is at 3.5: t* = 1 + 2.5/k."""
    times = [1.0, 2.0, 3.5, 4.0]
    r, j, s = _sens_run(
        _HALVE.format(op=">="), times, sensitivity_params=["k"], sensitivity_ic=["q"]
    )
    assert s == _column(_HALVE.format(op=">="), times, "S")
    np.testing.assert_allclose(s, [3, 4, 2.75, 3.25], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(r.sensitivities)[:, j, 0], [0, 0, 1.25, 1.25], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(r.sensitivities_ic)[:, j, 0], [0, 0, 0.5, 0.5], atol=1e-6
    )


def test_two_events_on_one_crossing_at_the_row_fire_together():
    """Both rise on q = 2.5; E2 reads S from before the batch (3 + 2.5)."""
    text = (
        _HALVE.format(op=">=").replace("species S = 3;", "species S = 3, R = 1;")
        + " E2: at (q >= 2.5): R = R + S;"
    )
    times = [0.0, 2.5, 3.0]
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(sample_times=times)
    names = list(r.species_names)
    x = np.asarray(r.species)
    plain = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode").run(
        sample_times=times
    )
    np.testing.assert_array_equal(x, np.asarray(plain.species))
    np.testing.assert_allclose(x[:, names.index("R")], [1, 6.5, 6.5], rtol=1e-6)
    # R = 1 + 3 + t*: dR/dk = dt*/dk = -2.5
    np.testing.assert_allclose(
        np.asarray(r.sensitivities)[:, names.index("R"), 0], [0, -2.5, -2.5], atol=1e-6
    )


def test_a_falling_trigger_at_the_row_agrees_with_a_plain_run():
    text = "species S = 3; q = 5; k = 1; q' = -k; S' = 1; E1: at (q <= 2.5): S = S/2;"
    times = [0.0, 1.0, 2.5, 3.0]
    r, j, s = _sens_run(text, times, sensitivity_params=["k"])
    # (Off the crossing the two runs step apart by rounding: CVODE's error
    # test reads the columns too.)
    np.testing.assert_allclose(s, _column(text, times, "S"), rtol=1e-12)
    np.testing.assert_allclose(s, [3, 4, 2.75, 3.25], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(r.sensitivities)[:, j, 0], [0, 0, 1.25, 1.25], atol=1e-6)
