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
    [("0.3", 1.7), ("0.3", 0.9), ("0.1", 0.9), ("0.7", 0.9), ("0.7", 2.37)],
    ids=lambda v: str(v),
)
def test_other_offsets_read_the_same_at_every_run_length(x0, row):
    # The threshold is the double x0 + row rounds to, so the trigger is met on
    # the row as the doubles evaluate it. On main the first three read 3 at
    # some run lengths and 1 at others; 0.7 + 0.9 and 0.7 + 2.37 read 3 at
    # every one, the stop always a double after the row. 0.7 + 2.37 is the
    # double 3.0700000000000003, a threshold of 17 digits.
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


@pytest.mark.parametrize(
    ("x0", "threshold", "row"),
    [("3.0", "3.9", 0.9), ("1000.0", "1002.37", 2.37)],
    ids=["3.9 from 3", "1002.37 from 1000"],
)
def test_a_first_zero_before_the_row_leaves_a_counter_s_row_before_the_event(x0, threshold, row):
    # x0 + t is the threshold in doubles a double or more before the row as
    # well, and the stop lands on such a double (0.8999999999999999, and
    # 2.369999999999999), where the integrated counter has not reached the
    # threshold: the event is taken after the row, which reads the state before
    # it at every run length. The trigger written on the clock reads the state
    # after it. The user guide says so.
    counter = _doc(_geq("<ci>x</ci>", threshold), x0=x0)
    clock = _doc(
        f"<apply><geq/><apply><plus/>{TIME}<cn>{x0}</cn></apply><cn>{threshold}</cn></apply>",
        counter=False,
    )
    assert {_row(counter, row, t_end) for t_end in T_ENDS} == {3.0}
    assert {_row(clock, row, t_end) for t_end in T_ENDS} == {1.0}


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


# ── Where the walk does not apply (a review of the first commit found) ─────


def _stops(monkeypatch, doc: str, row: float, t_end: float) -> set[float]:
    """The crossing times the solve returns during a run, and [S] on the row."""
    from bngsim import _switch_sensitivity as ss

    seen: list[float | None] = []
    solve = ss._crossing_time_of_condition

    def spy(*args, **kwargs):
        answer = solve(*args, **kwargs)
        seen.append(answer)
        return answer

    monkeypatch.setattr(ss, "_crossing_time_of_condition", spy)
    _row(doc, row, t_end)
    monkeypatch.setattr(ss, "_crossing_time_of_condition", solve)
    return {t for t in seen if t is not None}


def _strict(lhs: str, threshold: str) -> str:
    return f"<apply><gt/>{lhs}<cn>{threshold}</cn></apply>"


@pytest.mark.parametrize("t_end", [4.0, 8.0, 64.0])
def test_a_strict_trigger_is_not_moved_onto_the_zero(t_end):
    # In doubles 0.7 + 0.9 is 1.6, so `x > 1.6` is not met at t = 0.9 and the
    # row reads the state before the event, as it did on main at every run
    # length. Walked onto the zero, the stop was 0.9 and the row read 1. The
    # `>=` of the same residual, solved first in the same process, still is.
    geq, gt = (
        _doc(_geq("<ci>x</ci>", "1.6"), x0="0.7"),
        _doc(_strict("<ci>x</ci>", "1.6"), x0="0.7"),
    )
    assert _row(geq, 0.9, t_end) == 1.0
    assert _row(gt, 0.9, t_end) == 3.0


def test_a_falling_trigger_met_on_its_zero_is_moved_onto_it(monkeypatch):
    # `2.7 - time <= 0.33000000000000007`: in doubles 2.7 - 2.37 is that
    # threshold, so a falling `<=` is first met at t = 2.37, the residual's
    # first zero. On main the stop was 2.37 for some run lengths and
    # 2.3700000000000006 for others (4 and 8 against 40 and 64).
    lhs = f"<apply><minus/><cn>2.7</cn>{TIME}</apply>"
    doc = _doc(f"<apply><leq/>{lhs}<cn>0.33000000000000007</cn></apply>", counter=False)
    for t_end in (4.0, 8.0, 40.0, 64.0):
        assert _stops(monkeypatch, doc, 2.37, t_end) == {2.37}
        assert _row(doc, 2.37, t_end) == 1.0


@pytest.mark.parametrize("t_end", [4.0, 64.0])
def test_a_crossing_on_the_start_instant_stays_on_it(monkeypatch, t_end):
    # `x >= 0.1` from x = 0.1: (t + 0.1) - 0.1 is 0 for every t below about
    # 7e-18 in size, far more doubles than the walk follows. The stop is the
    # solve's 0, not -3.2e-322, 64 doubles below it.
    stops = _stops(monkeypatch, _doc(_geq("<ci>x</ci>", "0.1")), 1.0, t_end)
    assert stops and all(t == 0.0 for t in stops)


@pytest.mark.parametrize("t_end", [4.0, 8.0, 64.0])
def test_a_run_of_zeros_longer_than_the_walk_leaves_the_solve_alone(monkeypatch, t_end):
    # (t + 1e6) - 1000000.5 is 0 for millions of doubles around 0.5. Its first
    # is out of reach, and a stop 64 doubles down from the solve, at
    # 0.49999999999999645, was a double of no meaning: the solve's 0.5 stands.
    doc = _doc(_geq("<ci>x</ci>", "1000000.5"), x0="1000000")
    assert _stops(monkeypatch, doc, 0.5, t_end) == {0.5}
    assert _row(doc, 0.5, t_end) == 1.0


def test_a_walk_that_reaches_its_bound_returns_the_answer_as_it_was():
    from bngsim._switch_sensitivity import _CROSSING_SNAP_ULPS, _CROSSING_ZERO_RUN

    base = 2.0**54  # doubles 4 apart

    def zeros(t):  # 0 for 4 * _CROSSING_ZERO_RUN doubles from base
        return 0.0 if base <= t < base + 16.0 * _CROSSING_ZERO_RUN else t - base

    start = base + 8.0 * _CROSSING_ZERO_RUN
    assert _snap_to_residual_zero(zeros, start) == start

    def level(t):  # the same size for many doubles, then smaller
        return 1.0 if t > base else 0.5

    start = base + 4.0 * (4 * _CROSSING_SNAP_ULPS)
    assert _snap_to_residual_zero(level, start) == start


# ── A trigger written with abs() or a square ────────────────────────────────
#
# Such a comparison is registered as its two edges, and the stop reads each
# edge's operator. They were written as strict `<` and `>` whatever the model
# wrote, so `abs(x - 3.1) <= 0.5` was placed by the strict rule and its row
# moved with the run length; each edge now keeps the operator written. Each
# spelling below is met on its zero, exactly on the row at t = 2.5, so the row
# reads the state after the event (S = 1) at every run length.

_U = "<apply><minus/><ci>x</ci><cn>3.1</cn></apply>"
_ABS = f"<apply><abs/>{_U}</apply>"
_SQUARE = f"<apply><power/>{_U}<cn>2</cn></apply>"
_FALL = "<apply><minus/><cn>2.6</cn><ci>x</ci></apply>"


def _cmp(op: str, lhs: str, rhs: str) -> str:
    return f"<apply><{op}/>{lhs}{rhs}</apply>"


SPELLINGS = {
    "abs(x - 3.1) <= 0.5": _cmp("leq", _ABS, "<cn>0.5</cn>"),
    "0.5 >= abs(x - 3.1)": _cmp("geq", "<cn>0.5</cn>", _ABS),
    "(x - 3.1)^2 <= 0.25": _cmp("leq", _SQUARE, "<cn>0.25</cn>"),
    # A control the first commit already placed: `<=` on a falling residual.
    "2.6 - x <= 0 (falling)": _cmp("leq", _FALL, "<cn>0</cn>"),
}


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_every_spelling_met_on_its_zero_reads_the_same_at_every_run_length(spelling):
    """On main each read 3 for some run lengths and 1 for others; before the
    edges kept their operator, the abs() and square ones still did."""
    doc = _doc(SPELLINGS[spelling])
    assert [_row(doc, 2.5, t_end) for t_end in T_ENDS] == [1.0] * len(T_ENDS)


def test_a_run_of_zeros_costs_a_few_evaluations_not_one_per_double():
    """Each evaluation is a sympy parse of the residual, about 0.25 ms. A run of
    zeros was followed a double at a time, up to 64 of them, and a run longer
    than that used all 64 to give the solve's answer back: about 16 ms more per
    solve than main, paid again at each new parameter set. The next double is
    tried first, then the bound, then halves."""
    from bngsim._switch_sensitivity import _CROSSING_ZERO_RUN
    from bngsim._switch_sensitivity import _snap_to_residual_zero as snap

    base = 2.0**54  # doubles 4 apart
    calls = []

    def counted(zero_until):
        def at(t):
            calls.append(t)
            return 0.0 if base <= t < zero_until else t - base

        return at

    snap(counted(base + 16.0 * _CROSSING_ZERO_RUN), base + 8.0 * _CROSSING_ZERO_RUN, 0.0)
    assert len(calls) <= 3  # longer than the bound: the next double, the bound
    calls.clear()
    assert snap(counted(base + 4.0 * 50), base + 4.0 * 25, 0.0)[0] == base
    assert len(calls) <= 10  # 25 doubles down to the first zero
    calls.clear()
    assert snap(counted(base + 4.0), base, 0.0)[0] == base
    assert len(calls) <= 1  # the common case: the double below is not 0


def _solve(cond: str, t_end: float) -> float | None:
    from bngsim import _switch_sensitivity as sw

    scope = sw.SwitchConditionScope(
        core=None,
        clocks={},
        clock_symbols=frozenset({"time", "time()"}),
        param_names=(),
        param_pats={},
        primary_names=frozenset(),
        param_idx={},
        values=(),
        derived_exprs={},
        run_constants=frozenset(),
        function_names=frozenset(),
    )
    sw._CROSSING_CACHE.clear()
    return sw._crossing_time_of_condition(cond, scope, 0.0, t_end)


@pytest.mark.parametrize("op", [">", "==", "!="])
def test_a_strict_comparison_and_an_equality_keep_the_solve_s_answer(op):
    """A strict stop moved past the last zero, where its truth changes, is no
    longer on the instant the engine fires a strict event at, and `==` holds
    only on the run: these keep the solve's answer. Solved again over the
    answer's own scale, that answer is the double after 2.5 at every run length
    (over the window's scale alone it was 2.5 for a run to 8, and the double
    after it for 4 or 64)."""
    cond = f"((time()+(0.1)){op}2.6)"
    assert {_solve(cond, t_end) for t_end in T_ENDS} == {math.nextafter(2.5, math.inf)}


# ── The answer's own scale (Ely's review of 93daf254) ──────────────────────


def test_a_strict_row_reads_the_same_at_every_run_length():
    # `x > 2.6` is first true after 2.5, where `(2.5 + 0.1) - 2.6` is 0, so the
    # row at 2.5 reads the state before the event. Over the window's scale the
    # stop was 2.5 for some run lengths, and the row then read 1.
    doc = _doc(_strict("<ci>x</ci>", "2.6"))
    assert {_row(doc, 2.5, t_end) for t_end in T_ENDS} == {3.0}


def test_a_tie_the_walk_broke_downward_reads_the_same_at_every_run_length():
    # x' = 1 from 1.71, `x >= 4.53`, a row at 2.8200000000000003. Over a scale
    # of 512 (a run to 300) the solve gave 2.82, where the residual is as small
    # as a double below it, and the walk went down to 2.8199999999999994: the
    # row read 3 for that run, and 1 for shorter ones.
    row = 2.8200000000000003
    assert {_solve("((time()+(1.71))>=4.53)", t_end) for t_end in T_ENDS} == {row}
    doc = _doc(_geq("<ci>x</ci>", "4.53"), x0="1.71")
    assert {_row(doc, row, t_end) for t_end in T_ENDS} == {1.0}


def test_a_crossing_on_a_power_of_two_is_solved_over_one_scale():
    # The answer is nudged up before its power of two is read, so 2.0 and the
    # double after it are solved over the same scale (4), not 2 and 4.
    after = math.nextafter(2.0, math.inf)
    assert {_solve("((time()+(0.1))>2.1)", t_end) for t_end in T_ENDS} == {after}
    assert {_solve("((time()+(0.1))>=2.1)", t_end) for t_end in T_ENDS} == {2.0}
    doc = _doc(_strict("<ci>x</ci>", "2.1"))
    assert {_row(doc, 2.0, t_end) for t_end in T_ENDS} == {3.0}


@pytest.mark.parametrize("cond", ["(((time()-2)^2)<=0)", "((time()*time())>=4)"])
def test_a_residual_that_is_not_linear_is_still_declined(cond):
    # The second solve comes after the linearity check: a line through a curve
    # can land on its root, and `(time-2)^2 <= 0` was resolved to 2 when it
    # came first.
    assert {_solve(cond, t_end) for t_end in T_ENDS} == {None}


def test_the_solve_for_a_trigger_met_on_its_zero_is_the_same_at_every_run_length():
    assert {_solve("((time()+(0.1))>=2.6)", t) for t in T_ENDS} == {2.5}
