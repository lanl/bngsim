"""The discontinuity suite's PR slice (issue #1912, phase 0 of #1781).

``parity_checks/discontinuity`` holds models whose rate laws switch, each with
a reference that does not go through bngsim's switch machinery: closed forms,
or the model integrated branch by branch at 30 digits. The nightly sweeps
every case over its parameters, tolerances, output grids and sensitivity
columns. This runs the few points each case picks for every pull request:

* each slice point, judged against its reference by the case's rule;
* each case's spec (the model written out by hand, with explicit branch
  variables) against its model text, by bngsim's own right-hand side;
* the convergence rule on every slice group that holds two tolerances.

A point that fails on main for a filed bug is a strict expected failure
naming the issue, so the fix turns this red until the mark comes out
(``Case.known``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "parity_checks" / "discontinuity"))

import _sweep as S  # noqa: E402
from _cases import CASES  # noqa: E402

_RUNS: dict = {}


def _run(case, point):
    key = (case.name, point)
    if key not in _RUNS:
        try:
            _RUNS[key] = S.run_point(case, point, strict=True)
        except S.EngineUnavailable as e:  # diffrax without JAX
            pytest.skip(str(e))
    return _RUNS[key]


@pytest.mark.parametrize(("case", "point"), S.slice_params(CASES))
def test_a_slice_point_matches_its_reference(case, point):
    row = _run(case, point)
    assert row.outcome == "PASS", f"{row.key}: {row.note}"


@pytest.mark.parametrize("case", [c for c in CASES if c.spec is not None], ids=lambda c: c.name)
def test_the_spec_says_what_the_model_text_says(case):
    assert S.check_spec(case) == []


@pytest.mark.parametrize(("case", "group", "points"), S.slice_groups(CASES))
def test_a_slice_group_converges(case, group, points):
    row = S.converge_row(case, group, [(p, _run(case, p)) for p in points])
    assert row.outcome == "PASS", f"{row.key}: {row.note}"
