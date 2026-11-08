"""The sample on a strict time trigger's onset is not judged against AMICI (#1557).

At t = c a trigger ``time > c`` is false: the event has not fired yet, and bngsim
and RoadRunner report the state before it there. AMICI's importer turns ``>`` and
``>=`` into one root function, ``time - c``, so it fires at c and reports the state
after it. From the next sample on, the three agree. BIOMD0000000104, 125, 563, 860
and 862 each read DIFF against AMICI on that one cell and nowhere else.

``_rr_common.strict_onsets`` finds the onsets from the SBML; ``_amici_common``
names the job-grid samples on one (``onset_samples``, and ``amici_ode``'s
``strict_onset_samples``); ``amici_run._compare_ode`` leaves their rows out of the
verdict and says what that changed. A closed comparison (``time >= c``) is not an onset:
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
import amici_sens_run as asr  # noqa: E402
import generate_amici_matrix as gam  # noqa: E402

T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _cn(v) -> str:
    return f"<cn>{v}</cn>"


def _apply(op: str, *args: str) -> str:
    """``<apply><op/>args</apply>``: a relational (``gt``, ``lt``, ...) or any other operator."""
    return f"<apply><{op}/>{''.join(args)}</apply>"


_rel = _apply


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
# _compare_ode leaves the onset rows out, and says what that changed
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
    # The tallies are over the judged rows: 10 of 11 samples, 2 species.
    assert comment.startswith("2 sp; fail 0/20 (hard 0, soft 0, forgiven 0)")
    assert comment.endswith(" | strict onset not judged at t = 0.5 (rescued 1 DECISIVE; GH #1557)")
    # The premise of the forgiveness: without it this is the #1557 DIFF.
    status, value, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12)
    assert (status, value) == ("diff", pytest.approx(0.6))
    assert " | " not in comment and "fail 1/22 (hard 1" in comment


def test_an_onset_where_the_engines_agree_is_named_but_not_decisive():
    """A sample on an onset is left out even where nothing differs there (an
    event that sets a parameter, say): the note names it, rescued 0."""
    bn, am = _pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=True), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    status, _, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[0.5])
    assert status == "pass"
    assert comment.endswith(" | strict onset not judged at t = 0.5 (rescued 0; GH #1557)")


def test_a_difference_off_the_onset_row_is_still_a_diff():
    """Only the sample on the onset is forgiven: an engine that fired late,
    after the next sample, still fails there."""
    bn, am = _pair(
        bn_col=_step(0.65, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    status, _, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[0.5])
    assert status == "diff"
    assert "(rescued 1;" in comment and "DECISIVE" not in comment


def test_every_species_on_the_onset_row_is_left_out_and_counted():
    """The whole sample goes, not just the cells that jump: here both species
    step at the onset, and both cells are counted as rescued."""
    bn_col, am_col = _step(0.5, 1.0, 0.4, closed=False), _step(0.5, 1.0, 0.4, closed=True)
    bn = (_T, np.column_stack([bn_col, 3 * bn_col]), ["S", "R"])
    am = (_T, np.column_stack([am_col, 3 * am_col]), ["S", "R"])
    status, _, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[0.5])
    assert status == "pass" and "(rescued 2 DECISIVE;" in comment


def test_a_one_sided_non_finite_onset_cell_is_never_forgiven():
    """No reading of a jump puts a NaN on one engine, so that row is judged."""
    bn_col = _step(0.5, 1.0, 0.4, closed=False)
    bn_col[5] = np.nan
    bn, am = _pair(bn_col=bn_col, am_col=_step(0.5, 1.0, 0.4, closed=True))
    status, value, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[0.5])
    assert status == "diff" and value == np.inf
    assert " | " not in comment  # nothing was left out


def test_movement_only_at_an_onset_is_no_signal():
    """BIOMD0000000563's shape: the onset is the last sample, and the jump there
    is all that moves on AMICI's grid. Left out, nothing moves in the rows that
    are judged, and the row says NO SIGNAL as it would without the onset."""
    flat = np.ones(len(_T))
    bn = (_T, np.column_stack([flat * 0.0, flat]), ["S", "R"])
    am_s = np.where(_T >= 1.0, 1.0, 0.0)
    am = (_T, np.column_stack([am_s, flat]), ["S", "R"])
    status, _, comment, _, _ = ar._compare_ode(bn, am, atol=1e-12, onsets=[1.0])
    assert status == "pass"
    assert "NO SIGNAL" in comment and "(rescued 1 DECISIVE;" in comment
    note = gam.humanize_comment({"comment": comment, "outcome": "PASS"})
    assert note.startswith("No species moves in either engine")


def test_a_grid_or_species_mismatch_keeps_the_instants_note():
    bn, am = _pair(bn_col=np.zeros(len(_T)), am_col=np.zeros(len(_T)))
    short = (am[0][:-1], am[1][:-1], am[2])
    assert ar._compare_ode(bn, short, n_instants=3)[2].endswith("; AMICI through 3 event instants")
    other = (am[0], am[1], ["X", "Y"])
    assert ar._compare_ode(bn, other, n_instants=3)[2].endswith("; AMICI through 3 event instants")


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
def _row_comment(rescued: int, decisive: bool, times=(1.0,), fail: int = 0) -> str:
    head = f"5 sp; fail {fail}/5000 (hard {fail}, soft 0, forgiven 0); err/movement 6.2e-09"
    return f"{head} | {ar.onset_note(list(times), rescued, decisive)}"


def test_the_matrix_names_a_decisive_onset():
    note = gam.humanize_comment({"comment": _row_comment(1, True), "outcome": "PASS"})
    assert note.startswith("Not judged at t = 1, the c of a strict time comparison (time > c)")
    assert note.endswith("That cleared 1 failing point; without it this row would be a DIFF.")


def test_the_matrix_is_quiet_about_an_onset_that_cleared_nothing_on_a_pass():
    comment = _row_comment(0, False, (2.0, 5.0))
    assert gam.humanize_comment({"comment": comment, "outcome": "PASS"}) == ""
    # ... and still says it on a DIFF, where every unjudged sample is worth knowing
    comment = _row_comment(0, False, (2.0, 5.0), fail=9)
    note = gam.humanize_comment({"comment": comment, "outcome": "DIFF"})
    assert "Not judged at t = 2, 5," in note and "cleared" not in note


# --------------------------------------------------------------------------- #
# The premise, on the engines
# --------------------------------------------------------------------------- #
_GRID = (0.0, 1.0, 11)  # samples every 0.1, one on 0.5
_PREMISE = _sbml({"S": _rel("gt", T, _cn(0.5)), "Q": _rel("geq", T, _cn(0.5))})
# ``time > tc`` with tc = 0.3 in the SBML and 0.7 after its initialAssignment:
# amici_ode must place the onset where AMICI does, at 0.7.
_ASSIGNED = _sbml(
    {"P": _rel("gt", T, "<ci>tc</ci>")},
    params=_param("tc", 0.3),
    extra=(
        '<listOfInitialAssignments><initialAssignment symbol="tc">'
        f"<math {MATH}><cn>0.7</cn></math></initialAssignment></listOfInitialAssignments>"
    ),
)


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


def test_amici_ode_places_an_onset_by_amicis_own_parameter_values():
    pytest.importorskip("amici")
    am = ac.amici_ode(_ASSIGNED, *_GRID, 1e-9, 1e-12, through_event_instants=True)
    (at,) = am[3]["strict_onset_samples"]
    assert at == pytest.approx(0.7, abs=1e-15)
    assert rc.strict_onsets(_ASSIGNED, 0.0, 1.0) == [0.3]  # the SBML attribute, unassigned


# --------------------------------------------------------------------------- #
# The sensitivity job leaves the onset rows out too (issue #1571)
# --------------------------------------------------------------------------- #
def _sens_pair(*, bn_col, am_col, bn_s=None, am_s=None):
    """``(t, x, sx, names)`` for both engines: species S (``*_col``) and a flat
    R, one parameter, whose sensitivities are ``*_s`` for S (S's own column by
    default) and a ramp for R."""
    flat = np.linspace(1.0, 2.0, len(_T))
    ramp = np.linspace(0.1, 0.3, len(_T))

    def run(col, s):
        x = np.column_stack([col, flat])
        sx = np.stack([col if s is None else s, ramp], axis=1)[:, :, None]
        return _T, x, sx, ["S", "R"]

    return run(bn_col, bn_s), run(am_col, am_s)


_SENS = {"param_values": [1.0], "atol": 1e-12, "param_ids": ["k"]}


def test_the_sens_job_leaves_the_onset_row_out_of_both_verdicts():
    """BIOMD0000000104's shape: at the onset AMICI's state and sensitivities
    are after the event and bngsim's before it. Left out, the state is ok and
    the row passes; the note counts the failing cells on that row for each."""
    bn, am = _sens_pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    got = asr._compare(bn, am, onsets=[0.5], **_SENS)
    assert (got["status"], got["value"], got["state_passed"]) == ("pass", 0.0, True)
    assert got["comment"].startswith("2 sp x 1 par; 1/1 columns judged: fail 0/20 (hard 0")
    assert got["comment"].endswith(
        " | strict onset not judged at t = 0.5 (state rescued 1 DECISIVE;"
        " sensitivities rescued 1 DECISIVE; GH #1571)"
    )
    # The premise: judged on every sample, this is the #1571 row.
    got = asr._compare(bn, am, **_SENS)
    assert (got["status"], got["state_passed"]) == ("diff", False)
    assert " | " not in got["comment"]


def test_a_state_only_onset_cell_is_the_state_verdicts_alone():
    """104, 125, 860 and 862 read 'state DIFF' with the sensitivity verdict PASS:
    the jump is in the state, not in its sensitivities. Only the state is
    DECISIVE then."""
    smooth = np.linspace(0.2, 0.8, len(_T))
    bn, am = _sens_pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=False),
        am_col=_step(0.5, 1.0, 0.4, closed=True),
        bn_s=smooth,
        am_s=smooth,
    )
    assert asr._compare(bn, am, **_SENS)["status"] == "pass"
    got = asr._compare(bn, am, onsets=[0.5], **_SENS)
    assert (got["status"], got["state_passed"]) == ("pass", True)
    assert got["comment"].endswith("(state rescued 1 DECISIVE; sensitivities rescued 0; GH #1571)")


def test_the_sens_job_still_judges_a_difference_off_the_onset_row():
    bn, am = _sens_pair(
        bn_col=_step(0.65, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    got = asr._compare(bn, am, onsets=[0.5], **_SENS)
    assert (got["status"], got["state_passed"]) == ("diff", False)
    assert "(state rescued 1;" in got["comment"] and "DECISIVE" not in got["comment"]


def test_the_sens_job_judges_an_onset_row_with_a_one_sided_non_finite_cell():
    """In the state or in a sensitivity, as ``amici_run`` does for the state."""
    bn_col, am_col = _step(0.5, 1.0, 0.4, closed=False), _step(0.5, 1.0, 0.4, closed=True)
    bn_s = bn_col.copy()
    bn_s[5] = np.nan
    bn, am = _sens_pair(bn_col=bn_col, am_col=am_col, bn_s=bn_s)
    got = asr._compare(bn, am, onsets=[0.5], **_SENS)
    assert got["status"] == "diff" and " | " not in got["comment"]


def test_without_an_onset_on_the_grid_the_sens_comparison_is_the_one_it_was():
    bn, am = _sens_pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    plain = asr._compare(bn, am, **_SENS)
    assert asr._compare(bn, am, onsets=[], **_SENS) == plain
    assert asr._compare(bn, am, onsets=[0.55], **_SENS) == plain  # between samples
    # Without the floor's inputs, too: the tensor-wide verdict alone.
    assert asr._compare(bn, am, onsets=[0.55]) == asr._compare(bn, am)


def test_the_sens_job_audits_the_onset_without_the_floor_inputs():
    """``_compare`` without ``param_values``/``atol`` judges the tensor as a
    whole; the rescued count is taken over that verdict then."""
    bn, am = _sens_pair(
        bn_col=_step(0.5, 1.0, 0.4, closed=False), am_col=_step(0.5, 1.0, 0.4, closed=True)
    )
    got = asr._compare(bn, am, onsets=[0.5])
    assert got["status"] == "pass"
    assert "(state rescued 1 DECISIVE; sensitivities rescued 1 DECISIVE;" in got["comment"]


def test_both_runners_place_and_leave_out_onsets_through_one_helper():
    """The ODE and sensitivity jobs must not drift on which samples they leave
    out: both call ``_amici_common.onset_rows``, and their notes share a prefix."""
    assert asr.asens.onset_rows is ac.onset_rows
    assert ar.ONSET_NOTE_PREFIX == asr.asens.ONSET_NOTE_PREFIX == ac.ONSET_NOTE_PREFIX


def test_amici_sens_records_the_onset_samples_by_amicis_own_values(tmp_path):
    """The sensitivity run names the sample on 0.7, where AMICI places the
    initially assigned ``tc``, as ``amici_ode`` does; the SBML attribute says 0.3."""
    # The runner's warmup imports this before any model loads; a compiled AMICI
    # model reads amici.sim.sundials at import and fails if nothing has.
    pytest.importorskip("amici.sim.sundials")
    import _amici_sens as asens

    xml = tmp_path / "assigned.xml"
    xml.write_text(_ASSIGNED)
    ids, built = asens.amici_free_parameter_ids(str(xml))
    out = asens.amici_sens(built, *_GRID, 1e-9, 1e-12, ids[:1], "staggered", sbml=_ASSIGNED)
    (at,) = out[4]["strict_onset_samples"]
    assert at == pytest.approx(0.7, abs=1e-15)
    # Without the text, nothing is placed.
    out = asens.amici_sens(built, *_GRID, 1e-9, 1e-12, ids[:1], "staggered")
    assert out[4]["strict_onset_samples"] == []
