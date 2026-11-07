"""The sample on a strict time trigger's onset is not judged against AMICI (#1557).

At t = c a trigger ``time > c`` is false: the event has not fired yet, and bngsim
and RoadRunner report the state before it there. AMICI's importer turns ``>`` and
``>=`` into one root function, ``time - c``, so it fires at c and reports the state
after it. From the next sample on, the three agree. BIOMD0000000104, 125, 563, 860
and 862 each read DIFF against AMICI on that one cell and nowhere else.

``_rr_common.strict_onsets`` finds the onsets from the SBML; ``_amici_common``
names the job-grid samples on one (``onset_samples``, and ``amici_ode``'s
``strict_onset_samples``); ``amici_run._compare_ode`` forgives their rows and
says what that changed. A closed comparison (``time >= c``) is not an onset:
every engine reports the state after the event at c, so that sample is still
judged. rr_parity is unchanged, so RoadRunner still checks bngsim's reading of
the onset sample.

The derivation and comparison tests need only libsbml and numpy. The ones that
run AMICI skip where it is not installed; the bngsim one always runs.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libsbml")

import _amici_common as ac  # noqa: E402
import _rr_common as rc  # noqa: E402
import amici_run as ar  # noqa: E402
import generate_amici_matrix as gam  # noqa: E402

T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _cn(v) -> str:
    return f"<cn>{v}</cn>"


def _rel(op: str, *args: str) -> str:
    return f"<apply><{op}/>{''.join(args)}</apply>"


def _apply(op: str, *args: str) -> str:
    return f"<apply><{op}/>{''.join(args)}</apply>"


def _param(pid: str, value: float, constant: bool = True) -> str:
    return f'<parameter id="{pid}" value="{value}" constant="{str(constant).lower()}"/>'


def _event(eid: str, trigger: str, target: str) -> str:
    return (
        f'<event id="{eid}" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math {MATH}>{trigger}</math></trigger>'
        f'<listOfEventAssignments><eventAssignment variable="{target}">'
        f"<math {MATH}><cn>1</cn></math></eventAssignment></listOfEventAssignments></event>"
    )


def _sbml(events: dict[str, str], params: str = "", extra: str = "") -> str:
    """An L3V2 model: one species per key of ``events``, each at 0 with no
    reactions, set to 1 by an event on its trigger (a MathML body)."""
    species = "".join(
        f'<species id="{s}" compartment="C" initialConcentration="0" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        for s in events
    )
    evs = "".join(_event(f"E_{s}", trig, s) for s, trig in events.items())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
  </listOfCompartments>
  <listOfSpecies>{species}</listOfSpecies>
  <listOfParameters>{params}</listOfParameters>
  {extra}
  <listOfEvents>{evs}</listOfEvents>
 </model>
</sbml>
"""


def _onsets(trigger: str, t_start=0.0, t_end=10.0, params="", extra="") -> list[float]:
    return rc.strict_onsets(_sbml({"S": trigger}, params, extra), t_start, t_end)


# --------------------------------------------------------------------------- #
# strict_onsets: which boundaries are onsets
# --------------------------------------------------------------------------- #
_TWO_T = _apply("times", _cn(2), T)
_TEN_MINUS_T = _apply("minus", _cn(10), T)


@pytest.mark.parametrize(
    "trigger, expected",
    [
        (_rel("gt", T, _cn(2)), [2.0]),  # time > 2
        (_rel("lt", _cn(2), T), [2.0]),  # 2 < time
        (_rel("gt", _apply("minus", _TWO_T, _cn(1)), _cn(0)), [0.5]),  # 2t - 1 > 0
        (_rel("lt", _TEN_MINUS_T, _cn(5)), [5.0]),  # (10 - t) < 5: true after 5
        (_rel("gt", _TWO_T, _apply("plus", T, _cn(3))), [3.0]),  # time on both sides
        (_rel("geq", T, _cn(2)), []),  # closed: true at 2 already
        (_rel("leq", _cn(2), T), []),
        (_rel("lt", T, _cn(2)), []),  # strict, but it turns false
        (_rel("gt", _cn(2), T), []),
        (_rel("gt", _TEN_MINUS_T, _cn(5)), []),  # (10 - t) > 5: false after 5
        (_rel("gt", _cn(2), _cn(1)), []),  # no clock
    ],
    ids=[
        "gt",
        "lt-reversed",
        "affine",
        "falling-affine-lt",
        "time-both-sides",
        "geq",
        "leq-reversed",
        "lt-turns-false",
        "gt-reversed-turns-false",
        "falling-affine-gt",
        "constant",
    ],
)
def test_an_onset_is_a_strict_comparison_that_turns_true_just_after_c(trigger, expected):
    assert _onsets(trigger) == expected


def test_a_window_has_an_onset_where_it_opens_not_where_it_closes():
    window = _apply("and", _rel("gt", T, _cn(1)), _rel("lt", T, _cn(3)))
    assert _onsets(window) == [1.0]
    # n-ary: 1 < time < 3 is the same window
    assert _onsets(_rel("lt", _cn(1), T, _cn(3))) == [1.0]


def test_a_rem_has_an_onset_where_it_rises_through_the_level_not_where_it_wraps():
    """rem(time, 4) > 1 is false at 1 + 4k and true just after; at each wrap
    (4k) rem is already 0, its value just after, so a wrap is never an onset."""
    rem = _apply("rem", T, _cn(4))
    assert _onsets(_rel("gt", rem, _cn(1))) == [1.0, 5.0, 9.0]
    assert _onsets(_rel("lt", _cn(1), rem)) == [1.0, 5.0, 9.0]
    assert _onsets(_rel("lt", rem, _cn(1))) == []  # true at each wrap, false after 1


def test_a_comparison_is_read_as_written_under_not():
    """Documented limit: ``not(time <= 2)`` is ``time > 2``, but the closed
    comparison inside is what is read. AMICI imports no trigger with ``not``."""
    assert _onsets(_apply("not", _rel("leq", T, _cn(2)))) == []


def test_a_constant_parameter_places_an_onset_and_a_changing_one_does_not():
    assert _onsets(_rel("gt", T, "<ci>tc</ci>"), params=_param("tc", 3.0)) == [3.0]
    rule = (
        f'<listOfRules><assignmentRule variable="tc"><math {MATH}><cn>3</cn></math>'
        "</assignmentRule></listOfRules>"
    )
    trig = _rel("gt", T, "<ci>tc</ci>")
    assert _onsets(trig, params=_param("tc", 3.0, constant=False), extra=rule) == []


def test_a_species_comparison_gives_no_onset():
    assert _onsets(_rel("gt", "<ci>S</ci>", _cn(0.5))) == []


def test_only_onsets_inside_the_run_after_its_start_are_kept():
    """At t_start the start rule decides, not the comparison (BIOMD0000000285's
    class); t_end itself is in the run (BIOMD0000000563's onset is its last
    sample)."""
    trig = _apply("or", *(_rel("gt", T, _cn(c)) for c in (0, 5, 10, 11)))
    assert _onsets(trig) == [5.0, 10.0]
    assert _onsets(_rel("gt", T, _cn(2)), t_start=2.0) == []


def test_event_instants_still_reads_a_strict_onset_as_before():
    """``event_instants`` and ``strict_onsets`` share one trigger reader; the
    instant it gives an inequality is still a hair after its boundary."""
    (got,) = rc.event_instants(_sbml({"S": _rel("gt", T, _cn(2))}), 0.0, 10.0)
    assert 2.0 < got < 2.0 + 1e-8


# --------------------------------------------------------------------------- #
# onset_samples: which job-grid times sit on an onset
# --------------------------------------------------------------------------- #
def test_onset_samples_are_the_grid_times_within_1e_12_of_an_onset():
    grid = np.linspace(0.0, 1.0, 11)
    assert ac.onset_samples(grid, [0.5]) == [0.5]
    assert ac.onset_samples(grid, [0.5 + 5e-13, 0.33, 2.0]) == [0.5]  # absolute below 1
    assert ac.onset_samples(grid, [0.5 + 2e-12]) == []
    assert ac.onset_samples(grid, []) == []
    wide = np.array([0.0, 3600.0, 7200.0])
    assert ac.onset_samples(wide, [3600.0 + 0.5e-12 * 3600]) == [3600.0]  # relative above
    assert ac.onset_samples(wide, [3600.0 + 2e-12 * 3600]) == []


# --------------------------------------------------------------------------- #
# _compare_ode forgives the onset rows, and says so
# --------------------------------------------------------------------------- #
_T = np.linspace(0.0, 1.0, 11)


def _step(at: float, before: float, after: float, *, closed: bool) -> np.ndarray:
    """A species stepping from ``before`` to ``after`` at ``at``: the value at
    ``at`` is ``after`` when ``closed`` (AMICI's reading) and ``before`` otherwise
    (the SBML's for ``time > at``)."""
    return np.where((at <= _T) if closed else (at < _T), after, before)


def _pair(*, bn_col: np.ndarray, am_col: np.ndarray):
    flat = np.linspace(1.0, 2.0, len(_T))
    bn = (_T, np.column_stack([bn_col, flat]), ["S", "R"])
    am = (_T, np.column_stack([am_col, flat]), ["S", "R"])
    return bn, am


def test_the_onset_row_is_forgiven_and_the_comment_says_it_was_decisive():
    bn, am = _pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    status, value, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[0.5])
    assert (status, value) == ("pass", 0.0)
    assert comment.endswith(" | strict onset not judged at t = 0.5 (rescued 1 DECISIVE; GH #1557)")
    # The premise of the forgiveness: without it this is the #1557 DIFF.
    status, value, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12)
    assert (status, value) == ("diff", pytest.approx(0.6))
    assert " | " not in comment and "fail 1/22 (hard 1" in comment


def test_a_difference_off_the_onset_row_is_still_a_diff():
    """Only the sample on the onset is forgiven: an engine that fired late,
    after the next sample, still fails there."""
    bn, am = _pair(
        bn_col=_step(0.65, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    status, _, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[0.5])
    assert status == "diff"
    assert "(rescued 1;" in comment and "DECISIVE" not in comment


def test_a_one_sided_non_finite_onset_cell_is_never_forgiven():
    bn_col = _step(0.5, 1.0, 0.4, closed=False)
    bn_col[5] = np.nan
    bn, am = _pair(bn_col=bn_col, am_col=_step(0.5, 1.0, 0.4, closed=True))
    status, value, _, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[0.5])
    assert status == "diff" and value == np.inf


def test_without_an_onset_on_the_grid_the_comparison_is_the_one_it_was():
    bn, am = _pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=True), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    plain = ar._compare_ode(bn, am, atol=1e-12)
    assert ar._compare_ode(bn, am, atol=1e-12, onsets=[]) == plain
    assert ar._compare_ode(bn, am, atol=1e-12, onsets=[0.55]) == plain  # between samples
    assert " | " not in plain[2]


def test_the_instants_note_stays_with_the_tallies_before_the_onset_note():
    bn, am = _pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    comment = ar._compare_ode(bn, am, atol=1e-12, n_instants=2, onsets=[0.5])[2]
    head, onset = comment.split(" | ")
    assert head.endswith("; AMICI through 2 event instants")
    assert onset.startswith(ar.ONSET_NOTE_PREFIX)


def test_amici_run_reads_the_onsets_out_of_amicis_timing(monkeypatch):
    """The worker hands ``amici_ode``'s ``strict_onset_samples`` to the
    comparison and keeps them out of the report's timing block. Both engines are
    stand-ins, so this runs without AMICI."""
    bn_col = _step(0.5, 1.0, 0.4, closed=False)
    am_col = _step(0.5, 1.0, 0.4, closed=True)

    def fake_amici_ode(xml, t_start, t_end, n_points, rtol, atol, **kw):
        return _T, am_col[:, None], ["S"], {"event_instants": 1, "strict_onset_samples": [0.5]}

    monkeypatch.setattr(ac, "measure_warmup", lambda: {})
    monkeypatch.setattr(ac, "set_amici_quiet", lambda: None)
    monkeypatch.setattr(ac, "bn_ode", lambda *a, **k: (_T, bn_col[:, None], ["S"], {}))
    monkeypatch.setattr(ac, "amici_ode", fake_amici_ode)
    spec = {
        "key": "M|ode",
        "model_id": "M",
        "method": "ode",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": "unused.xml",
        "params": {"t_start": 0.0, "t_end": 1.0, "n_points": 11, "rtol": 1e-9, "atol": 1e-12},
    }
    out: list[dict] = []

    class _Queue:
        put = staticmethod(out.append)

    ar._worker(spec, _Queue())
    (res,) = out
    assert res["status"] == "pass"
    assert (
        "; AMICI through 1 event instants | strict onset not judged at t = 0.5" in res["comment"]
    )
    assert res["timing"]["amici"] == {"event_instants": 1}


# --------------------------------------------------------------------------- #
# The AMICI matrix says it in plain English
# --------------------------------------------------------------------------- #
def _row_comment(v: dict, times=(1.0,)) -> str:
    head = "5 sp; fail 1/5005 (hard 0, soft 0, forgiven 0); err/movement 6.2e-09"
    return f"{head} | {ar.onset_note(list(times), v)}"


def test_the_matrix_names_a_decisive_onset():
    v = {"passed": True, "passed_without_forgive": False, "n_forgive_rescued": 1}
    note = gam.humanize_comment({"comment": _row_comment(v), "outcome": "PASS"})
    assert "Not judged at t = 1, where a strict time trigger (time > c) fires" in note
    assert note.endswith("That cleared 1 failing point; without it this row would be a DIFF.")


def test_the_matrix_is_quiet_about_an_onset_that_cleared_nothing_on_a_pass():
    v = {"passed": True, "passed_without_forgive": True, "n_forgive_rescued": 0}
    comment = _row_comment(v, (2.0, 5.0)).replace("fail 1/5005", "fail 0/5005")
    assert gam.humanize_comment({"comment": comment, "outcome": "PASS"}) == ""
    # ... and still says it on a DIFF, where every unjudged sample is worth knowing
    v = {"passed": False, "passed_without_forgive": False, "n_forgive_rescued": 0}
    note = gam.humanize_comment({"comment": _row_comment(v, (2.0, 5.0)), "outcome": "DIFF"})
    assert "Not judged at t = 2, 5," in note and "cleared" not in note


# --------------------------------------------------------------------------- #
# The premise, on the engines
# --------------------------------------------------------------------------- #
_GRID = (0.0, 1.0, 11)  # samples every 0.1, one on 0.5
_PREMISE = _sbml({"S": _rel("gt", T, _cn(0.5)), "Q": _rel("geq", T, _cn(0.5))})


def _col(run, name):
    t, v, names = run[:3]
    names = ac.normalize_amici_names(list(names))
    return v[:, names.index(name)]


def test_bngsim_reports_the_state_before_a_strict_onset_and_after_a_closed_one():
    """bngsim's half of the premise: at t = 0.5, ``time > 0.5`` is false, so S
    is still 0; ``time >= 0.5`` is true, so Q is 1. RoadRunner reads both the
    same way, which rr_parity checks on every corpus model."""
    run = rc.bn_ode(_PREMISE, *_GRID, 1e-9, 1e-12)
    i = int(np.flatnonzero(run[0] == 0.5)[0])
    s, q = _col(run, "S"), _col(run, "Q")
    assert (s[i], s[i + 1]) == (0.0, 1.0)
    assert (q[i - 1], q[i]) == (0.0, 1.0)


def test_amici_reads_a_strict_onset_as_true_and_names_its_sample():
    """AMICI's half: it reports S = 1 at 0.5, after the event, as it does for
    Q. ``amici_ode`` names 0.5 as the onset sample, and the comparison against
    bngsim is a PASS only because that row is not judged. If AMICI starts
    reading ``>`` strictly, the first assertion fails and the rule can go."""
    pytest.importorskip("amici")
    am = ac.amici_ode(_PREMISE, *_GRID, 1e-9, 1e-12, through_event_instants=True)
    i = int(np.flatnonzero(am[0] == 0.5)[0])
    assert (_col(am, "S")[i - 1], _col(am, "S")[i]) == (0.0, 1.0)
    assert (_col(am, "Q")[i - 1], _col(am, "Q")[i]) == (0.0, 1.0)
    assert am[3]["strict_onset_samples"] == [0.5]

    bn = rc.bn_ode(_PREMISE, *_GRID, 1e-9, 1e-12)[:3]
    onsets = am[3]["strict_onset_samples"]
    status, _, comment, _, _ = ar._compare_ode(bn, am[:3], atol=1e-12, onsets=onsets)
    assert status == "pass" and "(rescued 1 DECISIVE;" in comment
    assert ar._compare_ode(bn, am[:3], atol=1e-12)[0] == "diff"
