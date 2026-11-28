"""The discontinuity suite's families (issue #1914): pulses, steps, sawtooths and
bends, each swept. Their closed forms are checked here: against finite
differences of their own species, against the closed form they generalize,
against the branch-by-branch reference, and outside their domain. So is the
window peak's scan of a stretch a reference names."""

from __future__ import annotations

import _closed_forms as CF
import numpy as np
import pytest
from _reference import window_peak

SPAN = (0.0, 30.0)

# ── The general exposure against the closed form it generalizes ─────────────


@pytest.mark.parametrize("shape", ["step", "triangle", "sawtooth"])
@pytest.mark.parametrize(("a", "b"), [(3.0, 1.0), (2.5, 0.5), (1.0, -1.0), (3.0, 0.05)])
def test_the_general_exposure_is_clock_exposure_on_its_residual(shape, a, b):
    """``pulse_exposure(shape, "coupled")`` gives #1776's closed form back, both
    limits at every crossing included, on the seed's and #1779's rows."""
    p = {"kx": 1.0, "k_clock": 2.0, "a": a, "b": b, "c": 25 * (a - b * 2.0), "w": 0.003}
    old, new = CF.clock_exposure(shape), CF.pulse_exposure(shape, "coupled")
    cross = old.crossings(p, SPAN)
    assert new.crossings(p, SPAN) == pytest.approx(cross, rel=1e-15)
    t = np.unique(np.concatenate([np.linspace(*SPAN, 301), cross, np.linspace(24.99, 25.01, 401)]))
    for side in ("left", "right"):
        xa, ca = old.fn(p, t, side)
        xb, cb = new.fn(p, t, side)
        for s in xa:
            np.testing.assert_allclose(xb[s], xa[s], rtol=1e-15, atol=1e-15 * np.abs(xa[s]).max())
        for key in ca:
            scale = np.abs(ca[key]).max()
            np.testing.assert_allclose(cb[key], ca[key], rtol=0, atol=1e-15 * scale + 1e-300)


# ── Each column is the derivative of its own species ────────────────────────
#
# Central differences of the closed form's species, at times away from every
# crossing, judged against each column's own size: a sign or a factor written
# wrongly into a column fails, however small the column is.

PULSE_ROWS = {
    "time": {"kx": 1.0, "k_clock": 2.0, "c": 25.0, "w": 0.1},
    "clock": {"kx": 1.0, "k_clock": 2.0, "c": 0.05, "w": 0.1, "T0": -49.95},
    "coupled": {"kx": 1.0, "k_clock": 2.0, "a": 2.5, "b": 0.5, "c": 37.5, "w": 0.1},
}
STEP_ROWS = {
    "time": {"k0": 0.1, "k1": 1.0, "k_clock": 2.0, "c": 25.0},
    "clock": {"k0": 0.3, "k1": 0.02, "k_clock": 0.5, "c": 2.5, "T0": -10.0},
}


def _times(cf, p, n=61):
    """Uniform times, with the crossings' neighbourhoods left out, and a few
    inside each pulse: a column is checked where it changes fastest too."""
    cross = np.asarray(cf.crossings(p, SPAN), float)
    t = np.linspace(*SPAN, n)
    for lo, hi in cf.peaks(p, SPAN):
        t = np.concatenate([t, np.linspace(lo, hi, 17)])
    for lo, hi in zip(cross, cross[1:], strict=False):
        t = np.concatenate([t, lo + (hi - lo) * np.array([0.25, 0.5, 0.75])])
    if len(cross):
        near = np.min(np.abs(t[:, None] - cross[None, :]), axis=1)
        t = t[near > 1e-3 * max(1e-3, np.ptp(cross) or 1.0)]
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


@pytest.mark.parametrize("shape", ["window", "triangle", "sawtooth", "gauss"])
@pytest.mark.parametrize("reads", ["time", "clock", "coupled"])
def test_each_pulse_column_is_the_derivative_of_its_species(shape, reads):
    p = PULSE_ROWS[reads]
    _fd_check(
        CF.pulse_exposure(shape, reads),
        p,
        [q for q in p if q not in ("a", "b")] + (["a", "b"] if reads == "coupled" else []),
    )


@pytest.mark.parametrize("reads", ["time", "clock"])
def test_each_step_column_is_the_derivative_of_its_species(reads):
    p = STEP_ROWS[reads]
    _fd_check(CF.level_step(reads), p, list(p))


@pytest.mark.parametrize(
    ("cf", "p"),
    [
        (CF.state_step(), {"k1": 0.2, "k2": 1.0, "th": 3.0}),
        (CF.state_step(), {"k1": 0.05, "k2": 0.02, "th": 5.0}),
        (CF.min_bend(), {"vmax": 1.0, "k": 0.5, "kd": 0.2}),
        (CF.min_bend(), {"vmax": 1.0, "k": 0.09, "kd": 0.2}),  # never crosses
        (CF.abs_bend(), {"kx": 1.0, "th": 3.0, "kd": 0.2}),
        (CF.abs_bend(), {"kx": 0.7, "th": 9.0, "kd": 2.0}),
    ],
)
def test_each_state_column_is_the_derivative_of_its_species(cf, p):
    _fd_check(cf, p, list(p))


def test_a_state_step_jumps_its_own_column_by_the_saltation_term():
    """At t* the k1 column jumps from -t*·th to -t*·th·k2/k1."""
    cf, p = CF.state_step(), {"k1": 0.2, "k2": 1.0, "th": 3.0}
    (ts,) = cf.crossings(p, SPAN)
    sol = cf.solve(p, SPAN, np.array([ts]), ("X()",), ("k1",))
    assert sol.sens_left.lo[0, 0, 0] == pytest.approx(-ts * 3.0, rel=1e-12)
    assert sol.sens.lo[0, 0, 0] == pytest.approx(-ts * 3.0 * 1.0 / 0.2, rel=1e-12)


@pytest.mark.parametrize(
    ("cf", "p"),
    [
        (CF.min_bend(), {"vmax": 1.0, "k": 0.5, "kd": 0.2}),
        (CF.abs_bend(), {"kx": 1.0, "th": 3.0, "kd": 0.2}),
        (CF.pulse_exposure("triangle", "time"), PULSE_ROWS["time"]),
    ],
)
def test_a_bend_leaves_every_column_continuous(cf, p):
    cross = cf.crossings(p, SPAN)
    sol = cf.solve(p, SPAN, np.asarray(cross), ("Y()",), tuple(q for q in p if q != "k_clock"))
    np.testing.assert_allclose(sol.sens.lo, sol.sens_left.lo, rtol=1e-12, atol=1e-12)


# ── The domain ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cf", "p", "message"),
    [
        (CF.pulse_exposure("window", "time"), {"kx": 1.0, "c": -0.01, "w": 0.1}, "ahead of t = 0"),
        (
            CF.pulse_exposure("triangle", "clock"),
            {"kx": 1.0, "k_clock": 2.0, "c": 0.0, "w": 0.1},
            "ahead of t = 0",
        ),
        (
            CF.pulse_exposure("sawtooth", "clock"),
            {"kx": 1.0, "k_clock": -1.0, "c": 5.0, "w": 0.1},
            "move forward",
        ),
        (CF.pulse_exposure("gauss", "time"), {"kx": 1.0, "c": 5.0, "w": 0.0}, "w > 0"),
        (CF.level_step("time"), {"k0": 1.0, "k1": 2.0, "c": -1.0}, "ahead of t = 0"),
    ],
)
def test_the_exposures_refuse_outside_their_domain(cf, p, message):
    with pytest.raises(ValueError, match=message):
        cf.fn(p, np.linspace(*SPAN, 5), "right")


# ── The window peak's named stretches ───────────────────────────────────────


def test_the_window_peak_finds_a_gaussian_narrower_than_its_uniform_scan():
    """A Gaussian of s = 0.00075 lies between two of the 2001 uniform scan
    times, and has no crossing to anchor it. The stretch the closed form names
    around its middle is what finds its peak: the scan without it reads the
    columns at their end values, 1e4 times too small."""
    cf = CF.pulse_exposure("gauss", "clock")
    p = {"kx": 1.0, "k_clock": 2.0, "c": 0.05, "w": 0.003, "T0": -49.95}
    obs, cols = ("Y()",), ("kx", "k_clock", "c", "w")
    _, named = window_peak(cf, p, SPAN, obs, cols)
    bare = CF.ClosedForm(cf.fn, cf.crossings_fn)
    _, blind = window_peak(bare, p, SPAN, obs, cols)
    middle = cf.solve(p, SPAN, np.array([25.0]), obs, cols).sens.lo[0]
    assert np.all(named >= np.abs(middle) * (1 - 1e-12))
    assert named[0, 1] > 1e3 * blind[0, 1]


def test_a_stretch_is_clipped_to_the_window():
    cf = CF.pulse_exposure("gauss", "time")
    p = {"kx": 1.0, "c": 29.99, "w": 0.1}
    ((lo, hi),) = cf.peaks(p, SPAN)
    assert (lo, hi) == (pytest.approx(29.79), 30.0)


# ── Each case's closed form against its spec, integrated branch by branch ────
#
# The spec is held to the model text by the slice's spec check; here the
# closed form is held to the spec, both limits at every crossing included, so
# a closed form and its model cannot drift apart. Rows wide enough for the
# branchwise reference's sign scan (400 times over the window) to see inside
# every pulse.

from _branchwise import Branchwise  # noqa: E402
from _cases import BY_NAME  # noqa: E402

BRANCHWISE_ROWS = [
    ("pulse_time", {"w": 4.0}),
    ("pulse_clock", {"w": 4.0}),
    ("gauss_time", {"w": 4.0}),
    ("gauss_clock", {"w": 4.0}),
    ("step_time", {}),
    ("step_clock", {"k1": 0.01}),
    ("step_state", {}),
    ("sawtooth_time", {"w": 4.0}),
    ("sawtooth_clock", {"w": 4.0}),
    ("sawtooth_coupled", {"w": 4.0, "a": 2.5, "b": 0.5}),
    ("triangle_time", {"w": 4.0}),
    ("triangle_clock", {"w": 4.0}),
    ("min_state", {}),
    ("abs_state", {}),
]


@pytest.mark.parametrize(("name", "row"), BRANCHWISE_ROWS, ids=[n for n, _ in BRANCHWISE_ROWS])
def test_each_family_closed_form_agrees_with_its_spec_integrated_branch_by_branch(name, row):
    case = BY_NAME[name]
    values = case.values(tuple(sorted(row.items())))
    span = case.window(values)
    cf = case.reference
    cross = cf.crossings(values, span)
    cols = case.axes.sens[-1]
    bw = Branchwise(case.spec, expect=len(cross))
    np.testing.assert_allclose(bw.crossings(values, span), cross, rtol=1e-13)
    t = np.array(sorted(set(np.linspace(*span, 31)) | {float(c) for c in cross}))
    a = bw.solve(values, span, t, case.observe, cols)
    b = cf.solve(values, span, t, case.observe, cols)
    for side in ("species", "species_left", "sens", "sens_left"):
        x, y = getattr(a, side).lo, getattr(b, side).lo
        peak = np.maximum(np.abs(x), np.abs(y)).max(axis=0)
        gap = np.abs(x - y).max(axis=0)
        # A column that is 0 throughout (the clock's, by kx) comes back from 30
        # digits as 1e-36 or so, which the absolute floor covers.
        np.testing.assert_array_less(gap, 1e-12 * peak + 1e-20, err_msg=side)


# ── The family cases' marks ─────────────────────────────────────────────────

import dataclasses  # noqa: E402

from _case import CaseError  # noqa: E402
from _cases import _families as F  # noqa: E402


def test_a_mark_names_a_point_the_sweep_has():
    case = BY_NAME["pulse_time"]
    cols = F.columns("window", "time")
    (k,) = F.keys(case, [({}, cols, ("1e-10",), ("u4",))])
    assert k == "pulse_time|rtol=1e-10,grid=u4,sens=kx+c+w"
    with pytest.raises(CaseError, match="no point"):
        F.keys(case, [({"w": 0.5}, cols, ("1e-10",), ("u4",))])


def test_a_clock_row_puts_its_crossing_at_the_time_it_names():
    for kw in ({}, {"k_clock": 8.0}, {"T0": -49.95}, {"t": 25.37}):
        row = F.at25("clock", **kw)
        values = {**F.base_params("window", "clock"), **row}
        (lead, _) = CF.pulse_exposure("window", "clock").crossings(values, SPAN)
        assert lead == pytest.approx(kw.get("t", 25.0), rel=1e-15)


def test_every_family_case_sweeps_every_grid_tolerance_and_column_set():
    for name, _ in BRANCHWISE_ROWS:
        case = BY_NAME[name]
        assert {g.label for g in case.axes.grids} == set(F.ALL_GRIDS)
        assert tuple(case.axes.rtol) == (1e-6, 1e-8, 1e-10, 1e-12)
        assert len(case.axes.sens) == 2 and case.axes.sens[0] == ()


def test_a_mark_at_a_key_the_sweep_has_not_is_refused():
    case = BY_NAME["step_time"]
    stray = {"step_time|rtol=1e-12,grid=u4,sens=k0+k1+c,engine=diffrax": None}
    with pytest.raises(CaseError):
        dataclasses.replace(case, known={**case.known, **stray}).validate()


@pytest.mark.parametrize("reads", ["time", "clock"])
def test_a_gaussian_begun_before_its_tail_has_fallen_takes_the_tail_off(reads):
    """A Gaussian whose middle is 0.5 from t = 0 and s = 1 has a fifth of its
    density there: E counts it from t = 0, and the columns of c and w carry
    the tail's own term (``h0``), which a pulse far from t = 0 never sees."""
    p = {"kx": 1.0, "c": 0.5, "w": 4.0}
    if reads == "clock":
        p.update({"k_clock": 1.0, "T0": 0.0})
    cf = CF.pulse_exposure("gauss", reads)
    x, _ = cf.fn(p, np.array([0.0]), "right")
    assert x["X()"][0] == 10.0  # nothing before t = 0
    _fd_check(cf, p, [q for q in p if q != "T0"])


def test_the_state_steps_offset_lies_outside_its_crossings_uncertainty():
    """Offset's default hair, 1e-9 of t*, lies inside a state switch's own
    crossing uncertainty at rtol 1e-6 (about rtol*X/|X'|), so a correct run can
    read either side there and its th column, which jumps by 4, fails; the
    case's 1e-4 lies outside it."""
    import _sweep as S
    from _case import Offset, point

    case = BY_NAME["step_state"]
    cols = ("k1", "k2", "th")
    p_case = point(1e-6, case.axes.grids[-1], cols, k2=1.0)
    hair = dataclasses.replace(
        case,
        axes=dataclasses.replace(case.axes, grids=(*case.axes.grids[:-1], Offset(31))),
        slice=(),
        known={},
    )
    assert S.run_point(case, p_case).outcome == "PASS"
    assert S.run_point(hair, point(1e-6, Offset(31), cols, k2=1.0)).outcome == "DIFF"
