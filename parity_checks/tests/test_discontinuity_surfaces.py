"""The discontinuity suite's coincident surfaces and events (issue #1915): two
switches on one instant, a switch beside an event, and events that set or move
the state. Their closed forms are checked here against finite differences of
their own species and against their specs integrated branch by branch, resets
included; so are the frame's additions: resets in the branch-by-branch
reference, a refusal a point is permitted, and the exact and blurred readings of
an instant."""

from __future__ import annotations

import dataclasses
import math

import _closed_forms as CF
import _sweep as S
import numpy as np
import pytest
from _branchwise import Branchwise, Reset, Spec, Surface, _sides, reset_saltation, saltation
from _case import CaseError, Uniform, point
from _cases import BY_NAME, _flips
from _cases import coincident as C
from _frozen import case_fingerprint
from _reference import Owed, Permitted, ReferenceUnavailable, window_peak

SPAN = (0.0, 30.0)
UP, DOWN = math.nextafter(25.0, math.inf), math.nextafter(25.0, 0.0)

# ── Each column is the derivative of its own species ────────────────────────


def _times(cf, p, n=61):
    cross = np.asarray(cf.crossings(p, SPAN), float)
    t = np.linspace(*SPAN, n)
    for lo, hi in zip(cross, cross[1:], strict=False):
        if hi - lo > 1e-6:
            t = np.concatenate([t, lo + (hi - lo) * np.array([0.25, 0.5, 0.75])])
    if len(cross):
        near = np.min(np.abs(t[:, None] - cross[None, :]), axis=1)
        t = t[near > 2e-3]
    return np.sort(t)


def _fd_check(cf, p, cols, h_rel=1e-6):
    t = _times(cf, p)
    x, got = cf.fn(dict(p), t, "right")
    for q in cols:
        h = h_rel * max(abs(p[q]), 1e-2)
        up, down = dict(p, **{q: p[q] + h}), dict(p, **{q: p[q] - h})
        xu, xd = cf.fn(up, t, "right")[0], cf.fn(down, t, "right")[0]
        for s in x:
            fd = (np.asarray(xu[s]) - np.asarray(xd[s])) / (2 * h)
            col = np.broadcast_to(np.asarray(got[(s, q)], float), t.shape)
            scale = float(np.abs(fd).max())
            if scale == 0:
                assert np.abs(col).max() <= 1e-12, f"d{s}/d{q} should be 0"
                continue
            np.testing.assert_allclose(col, fd, rtol=0, atol=1e-5 * scale, err_msg=f"d{s}/d{q}")


def _row(name, **over):
    case = BY_NAME[name]
    return case.reference, case.values(tuple(sorted(over.items())))


FD_ROWS = [
    ("coincident_sum", {}, C.SUM_COLS),  # the tie: every column exists
    ("coincident_sum", {"c2": 25.5}, C.SUM_COLS),
    ("coincident_and", {"c2": 25.5}, C.AND_COLS),
    ("coincident_and", {"c2": 24.5}, C.AND_COLS),
    ("coincident_and", {}, C.AND_RATES),  # the tie: the rate columns only
    ("coincident_state", {"th2": 1.0}, C.STATE_COLS),
    ("coincident_state", {"th2": 2.7}, C.STATE_COLS),
    ("coincident_state", {"th1": 2.0}, C.STATE_COLS),  # th2 crossed first
    ("coincident_state", {}, C.STATE_RATES),
    ("switch_event", {"te": 25.5}, C.SE_COLS),
    ("switch_event", {"te": 24.5}, C.SE_COLS),
    ("switch_event", {"a": 0.5, "d": 2.0, "te": 12.0}, C.SE_COLS),
    ("switch_event", {"a": 0.0, "d": 5.0}, C.SE_RATES),
    ("event_state_reset", {}, ("k", "th", "a", "d")),
    ("event_state_reset", {"a": 0.0, "d": 9.0}, ("k", "th", "a", "d")),
    ("event_state_reset", {"k": 0.3, "d": 2.0}, ("k", "th", "a", "d")),
    ("event_state_reset", {"a": 0.5, "d": 6.0}, ("k", "th", "a", "d")),
    ("event_cross_switch", {}, ("k1", "k2", "th", "tau", "xr")),
    ("event_cross_switch", {"xr": 2.0}, ("k1", "k2", "th", "tau", "xr")),
    ("event_cross_switch", {"tau": 1.0, "xr": 2.0}, ("k1", "k2", "th", "tau", "xr")),
    ("event_cross_switch", {"tau": 25.0}, ("k1", "k2", "th", "tau", "xr")),
    ("event_clock_reset", {}, ("kx", "kc", "c", "w", "tau")),
    ("event_clock_reset", {"tau": 15.02}, ("kx", "kc", "c", "w", "tau")),
    ("event_clock_reset", {"c": 18.0, "tau": 16.0}, ("kx", "kc", "c", "w", "tau")),
    ("event_clock_reset", {"w": 4.0}, ("kx", "kc", "c", "w", "tau")),
]


@pytest.mark.parametrize(("name", "over", "cols"), FD_ROWS, ids=[f"{n}{o}" for n, o, _ in FD_ROWS])
def test_each_column_is_the_derivative_of_its_species(name, over, cols):
    cf, p = _row(name, **over)
    _fd_check(cf, p, cols)


# ── Each closed form against its spec integrated branch by branch ───────────
#
# On rows whose instants differ, a double apart included: at 30 digits the two
# roots of switches a double apart are two roots, and each sample on one reads
# that one's limits, as the closed forms' exact indicators do.

BRANCHWISE_ROWS = [
    ("coincident_sum", {"c2": 25.5}),
    ("coincident_sum", {"c2": UP}),
    ("coincident_and", {"c2": 25.5}),
    ("coincident_and", {"c2": DOWN}),
    ("coincident_state", {"th2": 1.0}),
    ("coincident_state", {"th2": 3.0 * (1 - 1e-4)}),
    ("switch_event", {"te": 25.5}),
    ("switch_event", {"te": UP}),
    ("switch_event", {"a": 0.5, "d": 2.0, "te": 12.0}),
    ("event_state_reset", {}),
    ("event_state_reset", {"a": 0.0, "d": 9.0}),
    ("event_cross_switch", {}),
    ("event_cross_switch", {"tau": 1.0, "xr": 2.0}),
    ("event_clock_reset", {}),
    ("event_clock_reset", {"tau": 15.02}),
    ("event_clock_reset", {"c": 18.0, "tau": 16.0}),
]


@pytest.mark.parametrize(
    ("name", "over"), BRANCHWISE_ROWS, ids=[f"{n}{o}" for n, o in BRANCHWISE_ROWS]
)
def test_each_closed_form_agrees_with_its_spec_integrated_branch_by_branch(name, over):
    case = BY_NAME[name]
    values = case.values(tuple(sorted(over.items())))
    cf = case.reference
    cross = cf.crossings(values, SPAN)
    cols = case.axes.sens[-1]
    bw = Branchwise(case.spec, expect=len(cross))
    np.testing.assert_allclose(bw.crossings(values, SPAN), cross, rtol=1e-13)
    t = np.array(sorted(set(np.linspace(*SPAN, 31)) | {float(c) for c in cross}))
    a = bw.solve(values, SPAN, t, case.observe, cols)
    b = cf.solve(values, SPAN, t, case.observe, cols)
    # Each column against its peak over the window: a column that is 0 at
    # every sample (a pulse's w column past it) is judged by its size inside.
    wp = dict(
        zip(("species", "sens"), window_peak(cf, values, SPAN, case.observe, cols), strict=True)
    )
    for side in ("species", "species_left", "sens", "sens_left"):
        x, y = getattr(a, side).lo, getattr(b, side).lo
        peak = np.maximum(np.maximum(np.abs(x), np.abs(y)).max(axis=0), wp[side.split("_")[0]])
        gap = np.abs(x - y).max(axis=0)
        np.testing.assert_array_less(gap, 1e-12 * peak + 1e-20, err_msg=side)


# ── Resets in the branch-by-branch reference ────────────────────────────────


def test_a_reset_that_moves_nothing_but_the_state_is_saltation():
    """With R the identity, the reset's formula is ``saltation()``."""
    s = [[0.3, -1.0], [2.0, 0.5]]
    fm, fp = [1.0, -2.0], [0.5, 4.0]
    gx, gt, gp = [1.0, -0.5], 0.25, [0.0, 2.0]
    eye = [[1.0, 0.0], [0.0, 1.0]]
    got = reset_saltation(s, fm, fp, eye, [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]], gx, gt, gp)
    want = saltation(s, fm, fp, gx, gt, gp)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-15)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-15)


def test_an_event_that_adds_to_a_decaying_state_jumps_its_columns_by_hand():
    """X' = -kX, and at t = te the event sets X = a*X + d. Just past te:
    dX/dte = k*(a*X- + d) - a*k*X- = k*d, dX/da = X-, dX/dd = 1, and dX/dk keeps
    a times its value before."""
    k, te, a, d, xm = 0.2, 3.0, 0.5, 2.0, 1.7
    sm = [[-te * xm, 0.0, 0.0, 0.0]]  # columns k, te, a, d just before te
    fm, fp = [-k * xm], [-k * (a * xm + d)]
    Rx, Rt, Rp = [[a]], [0.0], [[0.0, 0.0, xm, 1.0]]
    gx, gt, gp = [0.0], 1.0, [0.0, -1.0, 0.0, 0.0]  # g = t - te
    s_plus, tau = reset_saltation(sm, fm, fp, Rx, Rt, Rp, gx, gt, gp)
    assert tau == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert s_plus[0] == pytest.approx([a * -te * xm, k * d, xm, 1.0])


def _decay_spec(resets, surfaces=()):
    return Spec(
        states=("X",),
        params=("k", "th"),
        rhs=lambda x, s, t, p: {"X": -p["k"] * x["X"]},
        surfaces=surfaces,
        x0=lambda p: {"X": 10},
        resets=resets,
    )


@pytest.mark.parametrize(
    ("spec", "why"),
    [
        (
            _decay_spec(
                (Reset("E", lambda x, t, p: t - 1, lambda x, t, p: {"X": p["th"]}),),
                (Surface("s", lambda x, t, p: x["X"] - p["th"]),),
            ),
            "on a surface",
        ),
        (
            _decay_spec((Reset("E", lambda x, t, p: 20 - x["X"], lambda x, t, p: {"X": 1}),)),
            "holds at the start",
        ),
        (
            _decay_spec(
                (Reset("E", lambda x, t, p: p["th"] - x["X"], lambda x, t, p: {"X": x["X"] / 2}),)
            ),
            "leaves its own",
        ),
        (
            _decay_spec(
                (
                    Reset("E", lambda x, t, p: t - 1, lambda x, t, p: {"X": 1}),
                    Reset("F", lambda x, t, p: p["th"] - x["X"], lambda x, t, p: {"X": 9}),
                )
            ),
            "fires another",
        ),
    ],
    ids=["lands-on-a-surface", "true-at-start", "leaves-its-trigger-true", "fires-another"],
)
def test_the_branchwise_reference_refuses_a_reset_it_cannot_follow(spec, why):
    with pytest.raises(ReferenceUnavailable, match=why):
        Branchwise(spec, expect=1).crossings({"k": 0.2, "th": 3.0}, (0.0, 10.0))


def test_a_sample_on_the_second_of_two_instants_a_double_apart_reads_that_one():
    assert _sides((25.0, UP), 25.0) == (0, 1)
    assert _sides((25.0, UP), UP) == (1, 2)
    assert _sides((25.0, UP), 26.0) == (2, 2)


def test_a_spec_with_a_reset_must_name_its_switch_and_set_only_states():
    case = BY_NAME["event_cross_switch"]
    stray = Spec(
        case.spec.states,
        case.spec.params,
        case.spec.rhs,
        case.spec.surfaces,
        case.spec.x0,
        resets=(Reset("E9", lambda x, t, p: t - 1, lambda x, t, p: {"X": 1}),),
    )
    with pytest.raises(CaseError, match="E9"):
        stray.check_case(case)
    with pytest.raises(ValueError, match="not states"):
        Spec(
            ("X",),
            ("k",),
            lambda x, s, t, p: {"X": 0},
            (),
            lambda p: {"X": 1},
            resets=(Reset("E", lambda x, t, p: t - 1, lambda x, t, p: {"Q": 1}),),
        ).symbolic()


def test_a_record_of_a_case_whose_reset_changed_is_stale():
    case = BY_NAME["event_cross_switch"]
    spec = case.spec
    moved = Spec(
        spec.states,
        spec.params,
        spec.rhs,
        spec.surfaces,
        spec.x0,
        resets=(Reset("E1", lambda x, t, p: t - p["tau"], lambda x, t, p: {"X": 2 * p["xr"]}),),
    )
    assert case_fingerprint(case) != case_fingerprint(dataclasses.replace(case, spec=moved))


# ── Ties: owed, permitted, or neither ───────────────────────────────────────


@pytest.mark.parametrize(
    ("cf", "tie", "near", "apart", "cols", "rates"),
    [
        (
            CF.coincident_time(True),
            {"c1": 25.0, "c2": 25.0},
            {"c1": 25.0, "c2": UP},
            {"c1": 25.0, "c2": 25.5},
            C.AND_COLS,
            C.AND_RATES,
        ),
        (
            CF.coincident_state(),
            {"th1": 3.0, "th2": 3.0},
            {"th1": 3.0, "th2": math.nextafter(3.0, 0)},
            {"th1": 3.0, "th2": 2.7},
            C.STATE_COLS,
            C.STATE_RATES,
        ),
        (
            CF.switch_event(),
            {"c": 25.0, "te": 25.0},
            {"c": 25.0, "te": DOWN},
            {"c": 25.0, "te": 25.5},
            C.SE_COLS,
            C.SE_RATES,
        ),
    ],
    ids=["and", "state", "switch_event"],
)
def test_a_tie_owes_the_refusal_of_its_columns_and_an_ulp_from_it_permits_one(
    cf, tie, near, apart, cols, rates
):
    assert isinstance(cf.owed(tie, cols), Owed) and cf.permitted(tie, cols) is None
    assert cf.owed(tie, rates) is None and cf.owed(tie, ()) is None
    assert cf.owed(near, cols) is None and isinstance(cf.permitted(near, cols), Permitted)
    assert cf.permitted(near, rates) is None
    assert cf.owed(apart, cols) is None and cf.permitted(apart, cols) is None


def test_the_additive_pair_owes_and_permits_nothing_at_its_tie():
    cf = CF.coincident_time(False)
    for p in ({"c1": 25.0, "c2": 25.0}, {"c1": 25.0, "c2": UP}):
        assert cf.owed(p, C.SUM_COLS) is None and cf.permitted(p, C.SUM_COLS) is None


def test_two_state_crossings_within_a_few_ulps_are_one_instant_on_a_row():
    """A state crossing is located only to the run's accuracy: on a row at
    either of two crossings an ulp apart, the limits are before both and after
    both, never the stretch between them."""
    cf, p = _row("coincident_state", th2=math.nextafter(3.0, 0))
    cross = cf.crossings(p, SPAN)
    sol = cf.solve(p, SPAN, np.asarray(cross), ("X()",), ("d2",))
    before = -np.asarray(cross) * 3.0  # dX/dd2 = -t*X above both, X = 3 there
    after = CF.coincident_state().fn(dict(p, th2=3.0), np.asarray([cross[0]]), "right")[1][
        ("X()", "d2")
    ]
    np.testing.assert_allclose(sol.sens_left.lo[:, 0, 0], before, rtol=1e-12)
    np.testing.assert_allclose(sol.sens.lo[:, 0, 0], np.broadcast_to(after, (2,)), rtol=1e-9)


# ── A refusal permitted, in the driver ──────────────────────────────────────


def test_a_permitted_refusal_passes_and_says_so_in_its_subclass():
    case = BY_NAME["switch_event"]
    row = S.run_point(case, point(1e-8, Uniform(31), C.SE_COLS, te=UP), strict=True)
    assert row.outcome == "PASS" and row.subclass == "refused:SensitivityRefusedForParametersError"
    assert "refused as permitted" in row.note


def _permitting(case, refusal="SensitivityUnsupportedError", scale=1.0):
    ref = case.reference
    fn = ref.fn

    def scaled(p, t, side):
        x, cols = fn(p, t, side)
        return x, {k: v * scale for k, v in cols.items()}

    return dataclasses.replace(
        case,
        reference=dataclasses.replace(
            ref, fn=scaled, permits=lambda p, s: Permitted("test", refusal) if s else None
        ),
    )


def test_a_permitted_point_that_answers_is_judged_and_a_wrong_answer_fails():
    case = BY_NAME["coincident_sum"]
    p = point(1e-8, Uniform(31), C.SUM_COLS, c2=25.5)
    assert S.run_point(_permitting(case), p).outcome == "PASS"
    assert S.run_point(_permitting(case), p).subclass is None
    assert S.run_point(_permitting(case, scale=1.001), p).outcome == "DIFF"


def test_a_refusal_of_another_class_than_permitted_is_not_excused():
    case = BY_NAME["switch_event"]
    other = _permitting(case, refusal="CodegenDeclinedError")
    other = dataclasses.replace(other, reference=dataclasses.replace(other.reference, owes=None))
    p = point(1e-8, Uniform(31), C.SE_COLS, te=UP)
    assert S.run_point(other, p).outcome == "UNSUPPORTED"
    with pytest.raises(Exception, match="Forward sensitivity"):
        S.run_point(other, p, strict=True)


def test_a_point_that_may_refuse_is_in_no_convergence_group():
    case = BY_NAME["coincident_and"]
    pts = [point(r, Uniform(31), C.AND_COLS, c2=UP) for r in (1e-6, 1e-8)]
    assert case.groups(pts) == {}
    pts = [point(r, Uniform(31), C.AND_COLS, c2=25.5) for r in (1e-6, 1e-8)]
    assert len(case.groups(pts)) == 1


# ── #1915's flips ───────────────────────────────────────────────────────────


def test_every_1915_flip_is_left_out_of_its_case_with_its_count():
    assert len(_flips.FLIPS_1915) >= 40
    for key, (passed, kind) in _flips.FLIPS_1915.items():
        case = BY_NAME[key.split("|", 1)[0].split("[", 1)[0]]
        why = case.exclude[key]
        assert f"{passed} of {_flips.RUNS_1915}" in why or f"all {_flips.RUNS_1915}" in why
        assert key not in case.known and key not in {case.key(p) for p in case.slice}
        assert kind in _flips.WHY or len(kind) > 40


def test_a_case_names_its_events_as_switches_and_its_spec_holds_them():
    for name in ("switch_event", "event_state_reset", "event_cross_switch", "event_clock_reset"):
        case = BY_NAME[name]
        assert {r.switch for r in case.spec.resets} <= {s.name for s in case.switches}
        assert case.spec.resets, name


def test_an_antimony_case_sets_its_instants_as_doubles():
    """Antimony keeps 15 digits of a literal, so a row a double from 25 must
    reach the model through set_param, which the driver does for every row."""
    case = BY_NAME["switch_event"]
    m = S.prepare(case, case.values((("te", UP),)))
    assert m.get_param("te") == UP


def test_a_case_without_a_spec_reset_named_for_an_event_is_refused():
    case = BY_NAME["event_state_reset"]
    bare = dataclasses.replace(case.spec, resets=())
    with pytest.raises(CaseError, match="no surface"):
        bare.check_case(case)
