"""Issue #1673: a row on a counter's threshold reads the same at every run length.

``x(0) = 0.1``, ``x' = 1``, ``S = 3`` and the event ``at (x >= 2.6): S = 1``, with
a row at t = 2.5. The stop for the counter's threshold comes from a two-point
solve of the residual ``(t + 0.1) - 2.6`` at t = 0 and at t = ``scale``, the power
of two at or above the run's end. In doubles that answer moves with the scale:
2.5000000000000004 at a scale of 4 or 64, 2.5 at 8 or 32. Where it was one double
after 2.5 the row was taken first, so the row read S = 3 for ``t_end`` in
(2.5, 4] and (32, 64], and S = 1 for (4, 32], with nothing else changed.

The residual's zero, as the doubles evaluate it, is a property of the condition
alone: ``(2.5 + 0.1) - 2.6`` is exactly 0. The solve's answer is now moved to the
nearest double where the residual is closest to 0, the first of them on a tie, so
the stop is 2.5 at every scale and the row reads the state after the event, as a
row on an event's instant does (#1124 item 2).

The models are SBML text, so the file needs no Antimony.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

M = "http://www.w3.org/1998/Math/MathML"
TIME = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
T_ENDS = [3.0, 4.0, 4.5, 8.0, 32.0, 40.0, 64.0, 65.0, 300.0]


def _doc(trigger: str, x0: str = "0.1", counter: bool = True) -> str:
    # S = 3 until the event sets it to 1; x is a counter, x' = 1 from x0.
    rule = (
        f'<listOfRules><rateRule variable="x"><math xmlns="{M}"><cn>1</cn></math>'
        "</rateRule></listOfRules>"
        if counter
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies><species id="S" compartment="C" initialConcentration="3"
 hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="x" value="{x0}" constant="false"/></listOfParameters>
{rule}
<listOfEvents><event id="E1" useValuesFromTriggerTime="true">
<trigger initialValue="true" persistent="true"><math xmlns="{M}">{trigger}</math></trigger>
<listOfEventAssignments><eventAssignment variable="S"><math xmlns="{M}"><cn>1</cn></math>
</eventAssignment></listOfEventAssignments></event></listOfEvents>
</model></sbml>"""


def _geq(lhs: str, threshold: str) -> str:
    return f"<apply><geq/>{lhs}<cn>{threshold}</cn></apply>"


def _row(doc: str, row: float, t_end: float, *, grid: bool = False, **tol) -> float:
    """[S] on the row at `row`, for a run to `t_end`."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(doc), method="ode")
    if grid:
        result = sim.run(t_span=(0.0, t_end), n_points=int(round(t_end / 0.5)) + 1, **tol)
    else:
        result = sim.run(sample_times=[0.0, row, t_end], **tol)
    at = list(np.asarray(result.time)).index(row)
    return float(np.asarray(result.species)[at, list(result.species_names).index("S")])


@pytest.mark.parametrize("t_end", T_ENDS)
def test_the_issues_row_reads_after_the_event_at_every_run_length(t_end):
    # On main: 3 at t_end = 3, 4, 40 and 64; 1 at 4.5, 8, 32, 65 and 300.
    assert _row(_doc(_geq("<ci>x</ci>", "2.6")), 2.5, t_end) == 1.0


@pytest.mark.parametrize(
    "kw", [{"grid": True}, {"rtol": 1e-10, "atol": 1e-12}], ids=["grid", "tight"]
)
def test_the_row_reads_the_same_on_a_grid_and_at_any_tolerance(kw):
    doc = _doc(_geq("<ci>x</ci>", "2.6"))
    assert {_row(doc, 2.5, t_end, **kw) for t_end in (4.0, 8.0, 64.0)} == {1.0}


@pytest.mark.parametrize(
    ("x0", "row"),
    [("0.3", 1.7), ("0.3", 0.9), ("0.1", 0.9), ("0.7", 0.9)],
    ids=lambda v: str(v),
)
def test_other_offsets_read_the_same_at_every_run_length(x0, row):
    # The threshold is the double x0 + row rounds to, so the trigger is met on
    # the row as the doubles evaluate it. On main the first three read 3 at
    # some run lengths and 1 at others; 0.7 + 0.9 read 3 at every one, its
    # stop always a double after the row.
    threshold = repr(float(x0) + row)
    doc = _doc(_geq("<ci>x</ci>", threshold), x0=x0)
    assert {_row(doc, row, t_end) for t_end in T_ENDS} == {1.0}


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [("2.600000001", 3.0), ("2.599999999", 1.0)],
    ids=["a hair above the row", "a hair below it"],
)
def test_a_threshold_off_the_row_reads_as_it_did(threshold, expected):
    # Not met on the row, or met before it: the state before, or after, at
    # every run length, on main and here.
    doc = _doc(_geq("<ci>x</ci>", threshold))
    assert {_row(doc, 2.5, t_end) for t_end in T_ENDS} == {expected}


def test_the_same_event_on_the_clock_agrees_with_the_counter():
    # `time + 0.1 >= 2.6` is the counter's trigger written on the clock.
    clock = _doc(
        f"<apply><geq/><apply><plus/>{TIME}<cn>0.1</cn></apply><cn>2.6</cn></apply>", counter=False
    )
    counter = _doc(_geq("<ci>x</ci>", "2.6"))
    for t_end in (4.0, 8.0, 64.0):
        assert _row(clock, 2.5, t_end) == _row(counter, 2.5, t_end) == 1.0


# ── The snap itself ─────────────────────────────────────────────────────────


def _snap_to_residual_zero(at, t_star):
    # Imported here, so the run-length tests above still run where it is absent.
    from bngsim._switch_sensitivity import _snap_to_residual_zero as snap

    return snap(at, t_star)[0]


def test_an_answer_an_ulp_off_the_zero_is_moved_onto_it():
    def at(t):
        return (t + 0.1) - 2.6

    assert at(2.5) == 0.0
    assert _snap_to_residual_zero(at, math.nextafter(2.5, math.inf)) == 2.5
    assert _snap_to_residual_zero(at, math.nextafter(2.5, -math.inf)) == 2.5


def test_an_answer_on_the_first_zero_is_returned_as_it_was():
    def at(t):
        return t - 10.0

    assert _snap_to_residual_zero(at, 10.0) == 10.0


def test_a_run_of_zeros_is_placed_at_its_first_double():
    # At 2^54 the doubles are 4 apart, and t // 16 * 16 - 2^54 is 0 for the
    # four of them from 2^54 to 2^54 + 12: the condition `>= 0` first holds at
    # 2^54, wherever in the run the solve landed.
    base = 2.0**54

    def at(t):
        return (t // 16) * 16 - base

    assert _snap_to_residual_zero(at, base + 12.0) == base
    assert _snap_to_residual_zero(at, base) == base


def test_a_run_of_zeros_longer_than_the_window_is_followed_to_its_first_double():
    # Zero for the 64 doubles from 2^54 to 2^54 + 252, the solve landing 50 of
    # them in: the run is followed down to its first double.
    base = 2.0**54

    def at(t):
        return (t // 256) * 256 - base

    assert at(base + 252.0) == 0.0 and at(base - 4.0) != 0.0
    assert _snap_to_residual_zero(at, base + 200.0) == base


@pytest.mark.parametrize("off", [5, -5], ids=["above", "below"])
def test_an_answer_several_doubles_off_is_walked_to_the_zero(off):
    # A solve that lands five doubles from the zero, on either side.
    def at(t):
        return t - 2.6

    start = 2.6
    for _ in range(abs(off)):
        start = math.nextafter(start, math.inf if off > 0 else -math.inf)
    assert _snap_to_residual_zero(at, start) == 2.6


def test_the_residual_it_returns_is_the_one_at_the_double_it_returns():
    # The linearity check reads it; it must be the snapped double's, not the solve's.
    from bngsim._switch_sensitivity import _snap_to_residual_zero as snap

    def at(t):
        return (t + 0.1) - 2.6

    above = math.nextafter(2.5, math.inf)
    t, r = snap(at, above, at(above))
    assert (t, r) == (2.5, 0.0) and at(above) != 0.0


def test_a_tie_between_two_doubles_goes_to_the_earlier():
    # t - (1 + 2^-53): the zero is half way between 1 and the next double, and
    # the residual is 2^-53 in size at both.
    def at(t):
        return (t - 1.0) - 2.0**-53

    above = math.nextafter(1.0, 2.0)
    assert abs(at(1.0)) == abs(at(above)) == 2.0**-53
    assert _snap_to_residual_zero(at, above) == 1.0
    assert _snap_to_residual_zero(at, 1.0) == 1.0


def test_no_zero_is_placed_where_the_residual_is_smallest():
    # 1/3 has no double: the residual is smallest at the double nearest it.
    def at(t):
        return 3.0 * t - 1.0

    got = _snap_to_residual_zero(at, math.nextafter(1.0 / 3.0, math.inf))
    assert abs(at(got)) == min(
        abs(at(t)) for t in (math.nextafter(got, -1), got, math.nextafter(got, 2))
    )


def test_a_residual_it_cannot_read_leaves_the_answer_alone():
    assert _snap_to_residual_zero(lambda t: None, 2.5) == 2.5
    assert _snap_to_residual_zero(lambda t: float("nan"), 2.5) == 2.5
    assert math.isinf(_snap_to_residual_zero(lambda t: t, math.inf))
