"""The discontinuity suite's case format, keys, grids and slice rules (issue #1912)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import _sweep as S
import pytest
from _case import SLICE_CAP, Aligned, Axes, CaseError, Known, Offset, Uniform, point
from _cases import BY_NAME, CASES

BASELINE = Path(__file__).resolve().parents[1] / "nightly" / "baselines" / "discontinuity.json"


def _with(**change):
    return dataclasses.replace(BY_NAME["time_window"], **change)


def test_every_case_validates_and_no_two_keys_collide():
    keys = [k for c in CASES for k in c.all_keys()]
    assert len(keys) == len(set(keys))
    for c in CASES:
        c.validate()


def test_keys_read_as_the_nightly_files_them():
    c = BY_NAME["clock_step"]
    p = point(1e-10, Uniform(31), ("kx", "c"), b=-1.0, a=1.0)
    assert c.key(p) == "clock_step[a=1.0,b=-1.0]|rtol=1e-10,grid=u31,sens=kx+c"
    assert c.key(point(1e-6, Aligned(31))) == "clock_step|rtol=1e-06,grid=aligned31,sens=none"
    assert c.converge_key(p.group) == "clock_step[a=1.0,b=-1.0]|converge,grid=u31,sens=kx+c"


def test_the_sweep_is_the_product_of_its_axes_and_groups_span_the_tolerances():
    c = BY_NAME["time_window"]
    a = c.axes
    pts = c.points()
    assert len(pts) == len(set(pts)) == len(a.params) * len(a.sens) * len(a.grids) * len(a.rtol)
    groups = c.groups(pts)
    assert len(groups) == len(a.params) * len(a.sens) * len(a.grids)
    for ps in groups.values():
        assert [p.rtol for p in ps] == sorted(a.rtol, reverse=True)
    assert c.all_keys() == {c.key(p) for p in pts} | {c.converge_key(g) for g in groups}


def test_a_derived_parameter_follows_its_row():
    c = BY_NAME["clock_step"]
    assert c.values((("a", 2.5), ("b", 0.5)))["c"] == 25.0 * (2.5 - 0.5 * 2.0)


def test_grids_sample_on_and_beside_every_crossing_inside_the_window():
    span, cross = (0.0, 30.0), (7.3, 12.1, 31.0, 0.0)
    assert Uniform(4).request(span, cross) == {"n_points": 4}
    assert Aligned(4).request(span, cross) == {"sample_times": [0.0, 7.3, 10.0, 12.1, 20.0, 30.0]}
    times = Offset(4).request(span, cross)["sample_times"]
    for c in (7.3, 12.1):
        d = 1e-9 * c
        assert c - d in times and c + d in times and c not in times
    assert max(times) == 30.0 and min(times) == 0.0


def _slice_of(n):
    c = BY_NAME["time_window"]
    pts = list(c.slice)
    return tuple(pts + [p for p in c.points() if p not in pts][: n - len(pts)])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(name="a|b"), "may not hold"),
        (dict(family="spike"), "family"),
        (dict(fmt="bngl"), "fmt"),
        (dict(switches=()), "names no switch"),
        (dict(t_span=(5.0, 1.0)), "forward window"),
        (dict(observe=()), "observes no species"),
        (dict(axes=Axes(params=({"zz": 1.0},))), "not in params"),
        (dict(derive=lambda p: {"zz": 1.0}), "derive sets"),
        (dict(axes=Axes(sens=(("zz",),))), "are not parameters"),
        (dict(axes=Axes(sens=(("kx", "kx"),))), "names a column twice"),
        (dict(axes=Axes(rtol=(1e-8, 2.0))), "must lie in"),
        (dict(axes=Axes(grids=(Uniform(4), Uniform(4)))), "once each"),
        (dict(slice=()), "no point in the PR slice"),
        (dict(slice=_slice_of(SLICE_CAP + 1)), "past the cap"),
        (dict(slice=(point(1e-9, Uniform(4)),)), "outside its sweep"),
        (dict(slice=BY_NAME["time_window"].slice[:2]), "never takes rtol"),
        (dict(known={"nope|rtol=1e-08": Known(1)}), "known failures at keys"),
        (
            dict(known={BY_NAME["time_window"].key(BY_NAME["time_window"].slice[0]): Known(0)}),
            "naming its issue",
        ),
        (dict(observe=("X()", "Z()")), "its spec does not hold"),
    ],
)
def test_a_case_written_wrongly_is_refused(change, message):
    with pytest.raises(CaseError, match=message):
        _with(**change).validate()


def test_a_known_failure_is_a_strict_expected_failure_in_the_slice():
    base = BY_NAME["time_window"]
    marked_key = base.key(base.slice[0])
    case = _with(known={marked_key: Known(1799, AssertionError, "a row recorded short")})
    params = S.slice_params([case])
    marked = [p for p in params if p.marks]
    assert [p.id for p in marked] == [marked_key]
    (mark,) = marked[0].marks
    assert mark.name == "xfail"
    assert mark.kwargs["strict"] is True and mark.kwargs["raises"] is AssertionError
    assert mark.kwargs["reason"].startswith("#1799")


def known_rows_passing(cases, baseline: dict) -> list[str]:
    """The known failures the committed baseline records as passing."""
    rows = baseline["cases"]
    return sorted(
        k for c in cases for k in c.known if rows.get(k, {}).get("outcome", "PASS") == "PASS"
    )


def test_the_rule_names_a_known_failure_the_baseline_records_as_passing():
    case = BY_NAME["clock_step"]
    (k1, k2) = sorted(case.known)
    baseline = {"cases": {k1: {"outcome": "DIFF"}, k2: {"outcome": "PASS"}}}
    assert known_rows_passing([case], baseline) == [k2]
    assert known_rows_passing([case], {"cases": {}}) == [k1, k2]


@pytest.mark.skipif(not BASELINE.exists(), reason="no committed discontinuity baseline yet")
def test_every_known_failure_fails_in_the_committed_baseline():
    """A fixed bug's row turns PASS in the nightly; re-baselining it must take
    its mark out too, or this fails."""
    baseline = json.loads(BASELINE.read_text())
    assert known_rows_passing(CASES, baseline) == []
