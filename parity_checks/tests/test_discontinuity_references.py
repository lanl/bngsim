"""The discontinuity suite's references (issue #1912): each checked against
another, against hand values, and against the test they were lifted from."""

from __future__ import annotations

import dataclasses
import importlib
import json
import subprocess
import sys
from pathlib import Path

import _closed_forms as CF
import numpy as np
import pytest
from _branchwise import Branchwise, Spec, Surface, saltation
from _cases import BY_NAME
from _frozen import Frozen, tool_section, write_record
from _reference import Band, ReferenceUnavailable, past

PC = Path(__file__).resolve().parents[1]
SPAN = (0.0, 30.0)


def _grid(cross):
    return np.array(sorted(set(np.linspace(*SPAN, 31)) | {float(c) for c in cross}))


def _close(a: Band, b: Band, rel: float) -> float:
    """The largest gap between two point bands, over each column's peak."""
    peak = np.maximum(np.abs(a.lo), np.abs(b.lo)).max(axis=0)
    gap = np.abs(a.lo - b.lo).max(axis=0)
    worst = float(np.max(np.where(peak > 0, gap / np.where(peak > 0, peak, 1), gap)))
    assert worst <= rel, worst
    return worst


CLOSED = {
    "time_window": CF.window_decay("X()", "Y()", 10.0),
    "clock_step": CF.clock_exposure("step"),
    "decay_bend": CF.decay_bend(),
}


@pytest.mark.parametrize(
    ("name", "row"),
    [
        ("time_window", ()),
        ("time_window", (("t_off", 7.31),)),
        # Both switches on scan times (multiples of 30/400): each restart sits
        # on a scan time where g is exactly 0.
        ("time_window", (("t_off", 15.0), ("t_on", 7.5))),
        ("clock_step", ()),
        ("clock_step", (("a", 1.0), ("b", -1.0))),
        ("decay_bend", ()),
    ],
)
def test_the_branchwise_reference_reproduces_each_closed_form(name, row):
    """Both limits at every crossing included, so the saltation jumps are
    checked as well as the smooth stretches."""
    case = BY_NAME[name]
    values = case.values(row)
    cols = tuple(case.spec.params)
    cf = CLOSED[name]
    bw = Branchwise(case.spec, expect=len(cf.crossings(values, SPAN)))
    cross = cf.crossings(values, SPAN)
    np.testing.assert_allclose(bw.crossings(values, SPAN), cross, rtol=4e-16)
    t = _grid(cross)
    a = bw.solve(values, SPAN, t, case.observe, cols)
    b = cf.solve(values, SPAN, t, case.observe, cols)
    assert a.at_crossing.sum() == b.at_crossing.sum() == len(cross)
    for side in ("species", "species_left", "sens", "sens_left"):
        _close(getattr(a, side), getattr(b, side), 2e-14)
    if name != "decay_bend":
        # The columns do jump at these crossings, so the two limits differ.
        assert np.abs(a.sens.lo - a.sens_left.lo).max() > 1e-3


def test_the_clock_closed_forms_agree_with_the_test_they_were_lifted_from():
    sys.path.insert(0, str(PC.parent / "python" / "tests"))
    src = importlib.import_module("test_clock_carried_crossing")
    t = np.array([src.T_END])
    for shape in ("triangle", "sawtooth", "step"):
        for a, b in [(3.0, 1.0), (2.5, 0.5), (1.0, -1.0), (5.5, 2.0)]:
            c = 25.0 * (a - b * src.K_CLOCK)
            p = {"kx": 1.0, "k_clock": src.K_CLOCK, "a": a, "b": b, "c": c, "w": 0.003}
            sol = CF.clock_exposure(shape).solve(p, SPAN, t, ("Y()",), ("kx", "k_clock"))
            got = (sol.species.lo[0, 0], sol.sens.lo[0, 0, 0], sol.sens.lo[0, 0, 1])
            np.testing.assert_allclose(got, src._exact(shape, a, b, c), rtol=1e-12)


def test_saltation_matches_hand_values():
    # A time switch turning on X' = -k X at t_on: dX/dt_on jumps from 0 to k X.
    k, x = 0.7, 3.0
    s_plus, tau = saltation([[0.0]], [0.0], [-k * x], gx=[0.0], gt=1.0, gp=[-1.0])
    assert tau == [1.0] and s_plus == [[k * x]]
    # A state switch: y' = b starts as x' = -a x falls through th. With
    # t* = ln(x0/th)/a and y = b (t - t*), dy/dth = b/(a th) at once.
    a, b, th = 0.2, 0.5, 3.0
    s_plus, tau = saltation(
        [[0.0], [0.0]], [-a * th, 0.0], [-a * th, b], gx=[-1.0, 0.0], gt=0.0, gp=[1.0]
    )
    assert tau == pytest.approx([-1 / (a * th)])
    assert s_plus == [[0.0], [pytest.approx(b / (a * th))]]
    with pytest.raises(ZeroDivisionError):
        saltation([[0.0]], [0.0], [1.0], gx=[1.0], gt=0.0, gp=[0.0])


def _one_state(rhs, g, x0=2.0, params=("k",)):
    return Spec(
        states=("X()",),
        params=params,
        rhs=lambda x, s, t, p: {"X()": rhs(x["X()"], s, t, p)},
        surfaces=tuple(Surface(f"s{i}", gi) for i, gi in enumerate(g)),
        x0=lambda p: {"X()": x0},
    )


@pytest.mark.parametrize(
    ("spec", "expect", "message"),
    [
        # X' = -1 above 1 and +1 below it: both flows point into X = 1.
        (
            _one_state(lambda x, s, t, p: p["k"] * (1 - 2 * s[0]), [lambda x, t, p: x["X()"] - 1]),
            1,
            "a slide",
        ),
        (
            _one_state(lambda x, s, t, p: -p["k"] * x, [lambda x, t, p: x["X()"] - 2]),
            0,
            "starts exactly on",
        ),
        (
            _one_state(lambda x, s, t, p: -p["k"] * x * s[0], [lambda x, t, p: t - 5]),
            2,
            "expects 2",
        ),
        (
            _one_state(
                lambda x, s, t, p: -p["k"] * x * (s[0] + s[1]),
                [lambda x, t, p: t - 5, lambda x, t, p: 5 - t],
            ),
            2,
            "at one instant",
        ),
    ],
)
def test_the_branchwise_reference_refuses_what_it_cannot_follow(spec, expect, message):
    with pytest.raises(ReferenceUnavailable, match=message):
        Branchwise(spec, expect=expect).solve({"k": 1.0}, SPAN, [0.0, 30.0], ("X()",), ())


def test_thirty_digits_are_enough_for_a_double():
    case = BY_NAME["decay_bend"]
    values, t = case.values(), np.linspace(*SPAN, 31)
    cols = case.spec.params
    a = Branchwise(case.spec, expect=1).solve(values, SPAN, t, case.observe, cols)
    b = Branchwise(case.spec, expect=1, dps=40).solve(values, SPAN, t, case.observe, cols)
    _close(a.species, b.species, 4e-16)
    _close(a.sens, b.sens, 4e-16)


def test_a_time_at_a_crossing_reads_either_side_and_one_beside_it_reads_its_own():
    t_star = 7.3
    t = np.array([t_star, np.nextafter(t_star, 0), t_star + 1e-9, t_star - 1e-9])
    assert past(t, t_star, "right").tolist() == [True, True, True, False]
    assert past(t, t_star, "left").tolist() == [False, False, True, False]


# ── Frozen records ──────────────────────────────────────────────────────────


@pytest.fixture
def window_record(tmp_path):
    """A record of time_window built from its closed form, as two tools would
    give it: each band the value widened by its own margin."""
    case = BY_NAME["time_window"]
    values = case.values()
    times = np.linspace(*SPAN, 7)
    sol = CLOSED["time_window"].solve(values, SPAN, times, case.observe, ("kx",))

    def section(tool_margin, crossings_margin):
        sp = Band(sol.species.lo - tool_margin, sol.species.hi + tool_margin)
        se = Band(sol.sens.lo - tool_margin, sol.sens.hi + tool_margin)
        cr = [
            ("on", 7.3 - crossings_margin, 7.3 + crossings_margin),
            ("off", 12.1 - 1e-12, 12.1 + 1e-12),
        ]
        return tool_section("pin", sp, se, cr)

    path = tmp_path / "time_window.json"
    row = {"params": values, "observe": case.observe, "sens": ["kx"], "times": times}
    write_record(
        path,
        case,
        [{**row, "tools": {"ariadne": section(1e-9, 1e-12), "capd": section(2e-9, 1e-11)}}],
    )
    return case, values, times, sol, path


def _frozen_for(case, path):
    ref = Frozen(path)
    ref.check_case(case)
    return ref


def test_a_record_answers_with_the_tools_bands_intersected(window_record):
    case, values, times, sol, path = window_record
    got = _frozen_for(case, path).solve(values, SPAN, times, case.observe, ("kx",))
    np.testing.assert_allclose(got.species.lo, sol.species.lo - 1e-9)
    np.testing.assert_allclose(got.species.hi, sol.species.hi + 1e-9)
    assert np.all(got.sens.distance(sol.sens.lo) == 0)
    assert "ariadne+capd" in got.note


def test_a_record_certifies_the_crossing_structure(window_record):
    case, values, *_, path = window_record
    ref = _frozen_for(case, path)
    assert ref.intervals(values) == [
        (7.3 - 1e-12, 7.3 + 1e-12, "on"),
        (12.1 - 1e-12, 12.1 + 1e-12, "off"),
    ]
    ref.certify(values, [(7.3, "on"), (12.1, "off")])
    with pytest.raises(ReferenceUnavailable, match="not the certified one"):
        ref.certify(values, [(7.3 + 1e-10, "on"), (12.1, "off")])
    with pytest.raises(ReferenceUnavailable, match="certifies 2"):
        ref.certify(values, [(7.3, "on")])


def test_a_record_refuses_when_its_tools_disagree(window_record, tmp_path):
    case, values, times, sol, _ = window_record
    sp = tool_section("pin", sol.species)
    shifted = tool_section("pin", Band(sol.species.lo + 1e-6, sol.species.hi + 1e-6))
    path = tmp_path / "split.json"
    row = {"params": values, "observe": case.observe, "sens": [], "times": times}
    write_record(path, case, [{**row, "tools": {"ariadne": sp, "capd": shifted}}])
    with pytest.raises(ReferenceUnavailable, match="do not overlap"):
        _frozen_for(case, path).solve(values, SPAN, times, case.observe, ())


def test_a_record_refuses_a_changed_case_an_unrecorded_time_and_an_unknown_tool(window_record):
    case, values, times, _, path = window_record
    changed = dataclasses.replace(case, model=case.model.replace("10", "11"))
    with pytest.raises(ReferenceUnavailable, match="stale"):
        _frozen_for(changed, path).solve(values, SPAN, times, case.observe, ())
    with pytest.raises(ReferenceUnavailable, match="no enclosure recorded"):
        _frozen_for(case, path).solve(values, SPAN, [0.0, 1.0], case.observe, ())
    with pytest.raises(ReferenceUnavailable, match="no record for these parameters"):
        _frozen_for(case, path).solve({**values, "kx": 2.0}, SPAN, times, case.observe, ())
    partial = path.parent / "partial.json"
    rec = json.loads(path.read_text())
    rec["rows"][0]["params"] = {"kx": 1.0}
    partial.write_text(json.dumps(rec))
    with pytest.raises(ReferenceUnavailable, match="no record for these parameters"):
        _frozen_for(case, partial).solve(values, SPAN, times, case.observe, ())
    with pytest.raises(ValueError, match="unknown tools"):
        write_record(
            path,
            case,
            [{"params": {}, "observe": [], "sens": [], "times": [], "tools": {"flowstar": {}}}],
        )


def test_a_branchwise_reference_checks_its_crossings_against_a_certificate(
    window_record, tmp_path
):
    case, values, times, sol, path = window_record
    good = Branchwise(case.spec, expect=2, certificate=Frozen(path))
    good.check_case(case)
    assert good.crossings(values, SPAN) == (7.3, 12.1)
    wrong = tmp_path / "wrong.json"
    cr = [("on", 7.4, 7.5), ("off", 12.1, 12.1)]
    row = {"params": values, "observe": case.observe, "sens": [], "times": times}
    write_record(
        wrong, case, [{**row, "tools": {"ariadne": tool_section("pin", sol.species, None, cr)}}]
    )
    bad = Branchwise(case.spec, expect=2, certificate=Frozen(wrong))
    bad.check_case(case)
    with pytest.raises(ReferenceUnavailable, match="not the certified one"):
        bad.crossings(values, SPAN)


# ── Independence ────────────────────────────────────────────────────────────


def test_no_reference_module_imports_bngsim():
    """A reference must not share code with the engine it judges."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import _case, _reference, _closed_forms, _branchwise, _frozen, _cases;"
        "print(sorted(m for m in sys.modules if m == 'bngsim' or m.startswith('bngsim.')))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code, str(PC / "discontinuity")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("shape", "change", "message"),
    [
        ("triangle", {"c": 1.0, "w": 3.0}, "ahead of t = 0"),  # started before t = 0
        ("triangle", {"w": 0.0}, "needs w > 0"),
        ("sawtooth", {"w": 0.0}, "needs w > 0"),
        ("step", {"b": 2.0}, "r > 0"),
    ],
)
def test_the_clock_closed_forms_refuse_outside_their_domain(shape, change, message):
    p = {"kx": 1.0, "k_clock": 2.0, "a": 3.0, "b": 1.0, "c": 25.0, "w": 0.003, **change}
    with pytest.raises(ValueError, match=message):
        CF.clock_exposure(shape).solve(p, SPAN, np.array([30.0]), ("Y()",), ())
