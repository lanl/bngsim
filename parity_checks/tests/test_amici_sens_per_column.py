"""amici_sens judges each parameter column on its own scale (issue #1261).

``sens_verdict`` on the whole (time, species, parameter) tensor hands
``_core.differ`` one array, whose significance gate counts a (species,
parameter) column only past 1e-3 of the array's peak. A parameter column's
sensitivities are in that parameter's units, so a column of dx/dk2 that is a
thousandth of dx/dk1 was never looked at: on the nightly's 82 compared
amici_sens cases, 2,702 of 4,272 parameter columns, in 56 cases, could be
zeroed, negated or multiplied by 1000 and still read PASS.

``_compare`` now judges each column with ``_amici_sens.column_verdicts``, the
rule fd_sens adopted in #1167 and now shares: a column scaled to its own peak,
judged where that peak clears its ``atol/|p|`` floor by ``JUDGE_FACTOR``, and
named on the row as not judged where it does not. On the same 82 cases that
judges 3,448 columns, every zeroed, negated or x1000 one among them reads DIFF,
and no case's outcome moves.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import _amici_sens as asens
import amici_sens_run as run
import numpy as np
import pytest

ATOL = 1e-12
T = np.linspace(0.0, 10.0, 11)
NAMES = ["A", "B"]
X = np.stack([np.exp(-0.1 * T), 1.0 - np.exp(-0.1 * T)], axis=1)


def _tensor(small_peak):
    """dx/dk1 peaks at 1e3, dx/dk2 at ``small_peak``: two parameters in their
    own units, which is what the tensor-wide gate cannot tell from noise."""
    shape = np.sin(0.3 * T + 0.2)[:, None] * np.array([1.0, -0.5])[None, :]
    return np.stack([1e3 * shape, small_peak * shape], axis=2)


def _compare(bn_sx, am_sx, param_values=(1.0, 1.0)):
    return run._compare(
        (T, X, bn_sx, NAMES),
        (T, X.copy(), am_sx, NAMES),
        param_values=list(param_values),
        atol=ATOL,
    )


@pytest.mark.parametrize("corrupt", ["zero", "negate", "x1000"])
def test_a_small_column_gone_wrong_reads_diff(corrupt):
    """dx/dk2 is 1e-7 of dx/dk1 (1e-4 even at x1000, still under the 1e-3
    gate) and 1e6 over its floor. Corrupted, it read PASS on the tensor as a
    whole; judged alone, it reads DIFF."""
    am = _tensor(1e-4)
    bn = am.copy()
    bn[..., 1] = {"zero": 0.0, "negate": -bn[..., 1], "x1000": 1000.0 * bn[..., 1]}[corrupt]
    assert asens.sens_verdict(bn, am, param_values=[1.0, 1.0], atol=ATOL)["passed"], (
        "precondition: the tensor-wide verdict cannot see it"
    )
    assert _compare(bn, am)["status"] == "diff"


def test_a_tensor_under_differs_absolute_floor_is_still_judged():
    """Every cell under differ's absolute 1e-12, as nanomolar_dimer's 6e-20
    tensor is in fd_sens, but far over its atol/|p| floor at atol 1e-30: each
    column is scaled to its own peak, so a negated one reads DIFF."""
    am = _tensor(1.0) * 1e-17
    bn = am.copy()
    bn[..., 1] = -bn[..., 1]
    res = run._compare(
        (T, X, bn, NAMES), (T, X.copy(), am, NAMES), param_values=[1.0, 1.0], atol=1e-30
    )
    assert res["status"] == "diff"


def test_the_failing_column_is_named():
    am = _tensor(0.1)
    bn = am.copy()
    bn[..., 1] = 0.0
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["columns_diff"] == ["k2"]
    assert res["n_columns_judged"] == 2


def test_agreeing_columns_pass():
    am = _tensor(0.1)
    res = _compare(am.copy(), am)
    assert res["status"] == "pass"
    assert res["columns_not_judged"] == [] and res["columns_diff"] == []


def test_a_column_under_its_floor_is_named_and_not_judged():
    """dx/dk2 peaks at 1e-10, under 20x its 1e-10 floor (100*atol/|p|): the
    solver does not resolve it, so the row names it rather than passing it.
    Zeroing it is then not the verdict's to see, by design."""
    am = _tensor(1e-10)
    bn = am.copy()
    bn[..., 1] = 0.0
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["status"] == "pass"
    assert res["columns_not_judged"] == ["k2"]
    assert res["n_columns_judged"] == 1


@pytest.mark.parametrize("over_floor", [5.0, 15.0])
@pytest.mark.parametrize("corrupt", ["zero", "negate"])
def test_a_column_under_20x_its_floor_gone_wrong_still_reads_diff(over_floor, corrupt):
    """dx/dk2 at 5x or 15x its 1e-10 floor is not judged alone, but zeroed or
    negated it is that far past the floor, and 1e-2 of the tensor's peak:
    the tensor-wide verdict main gave reads DIFF, and so must this one. Judged
    per column only, it read PASS."""
    shape = np.sin(0.3 * T + 0.2)[:, None] * np.array([1.0, -0.5])[None, :]
    am = np.stack([1e-7 * shape, over_floor * 1e-10 * shape], axis=2)
    bn = am.copy()
    bn[..., 1] = 0.0 if corrupt == "zero" else -bn[..., 1]
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["columns_not_judged"] == ["k2"], "precondition: k2 is not judged alone"
    assert res["status"] == "diff"
    assert res["columns_diff"] == ["k2"]


def test_a_model_wide_silent_zero_under_20x_is_a_diff_not_a_bad_test():
    """Every column at 10x its floor and bngsim returning zeros: nothing is
    judged alone, but the engines disagree far past the floor. That is a DIFF
    (scoring), as main called it, not a non-scoring BAD_TEST."""
    shape = np.sin(0.3 * T + 0.2)[:, None] * np.array([1.0, -0.5])[None, :]
    am = np.stack([1e-9 * shape, 1e-9 * shape], axis=2)
    res = _compare(np.zeros_like(am), am)
    assert res["n_columns_judged"] == 0
    assert res["status"] == "diff"
    assert res["columns_diff"] == [0, 1]
    # The tensor's own measure, as main reported it: a silent zero is 1.0.
    assert res["value"] == pytest.approx(1.0)


def test_a_row_with_no_column_to_judge_is_a_bad_test():
    """#328's vacuous pass, now refused rather than reported as one, with a
    key a census can group it by (#324)."""
    am = np.full((len(T), 2, 2), 1e-14)
    res = _compare(am.copy(), am)
    assert res["status"] == "bad_test"
    assert res["n_columns_judged"] == 0
    assert res["exception_class"] == "columns:none-judged"
    assert res["exception"]


def test_below_noise_floor_counts_the_whole_tensor():
    """n_below_noise_floor says how much of the tensor is under solver
    resolution (#316): a column left out of the per-column verdict is still
    part of the tensor."""
    am = _tensor(1e-10)
    res = _compare(am.copy(), am)
    assert res["columns_not_judged"] == [1]
    assert res["n_below_noise_floor"] == am.size


def test_no_parameter_judges_no_column():
    assert asens.judged_columns(np.zeros((3, 2, 0)), np.zeros((3, 2, 0)), [], ATOL) == []


def test_without_parameter_values_the_tensor_is_judged_whole():
    """No floor to judge a column against: the verdict sens_verdict gives
    without one, unchanged."""
    am = _tensor(0.1)
    res = run._compare((T, X, am.copy(), NAMES), (T, X.copy(), am, NAMES))
    assert res["status"] == "pass"
    # Not assessed, rather than "all judged, none failed".
    assert res["n_columns_judged"] is None
    assert res["columns_not_judged"] is None and res["columns_diff"] is None


def _fd_sens():
    path = Path(__file__).resolve().parents[1] / "fd_sens" / "fd_sens_run.py"
    spec = importlib.util.spec_from_file_location("fd_sens_run", path)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    return F


def _peaking_at(peak):
    """dx/dk1 at 1e3 and dx/dk2 peaking at exactly ``peak``."""
    am = _tensor(1.0)
    am[..., 1] *= peak / np.abs(am[..., 1]).max()
    return am


def test_fd_sens_judges_on_the_larger_of_its_two_floors():
    """fd_sens shares the rule and adds the difference's own floor to it: a
    column 20x over its atol/|p| floor but not over its difference's is not
    judged there, and is just over both."""
    F = _fd_sens()
    assert F.JUDGE_FACTOR == asens.JUDGE_FACTOR
    floor = max(F.fd_noise_floor(1.0, 1.0), 100.0 * ATOL)
    assert F.fd_noise_floor(1.0, 1.0) > 100.0 * ATOL, "precondition: the difference's floor rules"
    under, over = _peaking_at(15.0 * floor), _peaking_at(25.0 * floor)
    assert F.judgeable_columns(under, under, 1.0, [1.0, 1.0], ATOL) == [0]
    assert F.judgeable_columns(over, over, 1.0, [1.0, 1.0], ATOL) == [0, 1]


def test_fd_sens_judges_a_nan_valued_parameter_on_its_atol_floor():
    """A NaN parameter value makes the difference's floor NaN; fd_sens judged
    the column on its atol/|p| floor then (Python's max ignores the NaN), and
    still does."""
    F = _fd_sens()
    ones = np.ones((3, 2, 2))
    assert F.judgeable_columns(ones, ones, 1.0, [1.0, float("nan")], ATOL) == [0, 1]


def test_a_column_only_bngsim_makes_large_is_judged_on_bngsims_peak():
    """AMICI's dx/dk2 is under its floor, bngsim's 1e-7 of dx/dk1: either
    side's peak counts, so the column is judged, and reads DIFF. On AMICI's
    peak alone it would be left out, and the tensor cannot see it."""
    am = _tensor(1e-12)
    bn = am.copy()
    bn[..., 1] *= 1e8
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["status"] == "diff"
    assert res["columns_diff"] == ["k2"]


def test_a_column_just_over_20x_its_floor_is_judged():
    """dx/dk2 peaks at 22x its 1e-10 floor, 2e-12 of dx/dk1: judged alone,
    negated, it reads DIFF, worst column 2.0 of its peak. A factor over 22
    would leave it out, and an atol not scaled down with the column would
    forgive the whole of it."""
    am = _peaking_at(22.0 * 1e-10)
    bn = am.copy()
    bn[..., 1] = -bn[..., 1]
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["columns_not_judged"] == []
    assert res["status"] == "diff"
    assert res["columns_diff"] == ["k2"]
    assert res["value"] == pytest.approx(2.0)


def _noisy_moving_column(k1_peak, k2_over_floor):
    """dx/dk2 at ``k2_over_floor`` x its 1e-10 floor, and bngsim off by 0.9 of
    the floor on every cell of species A: solver noise, which the floor
    forgives, and which the movement gate (1% of a column's range) or the
    relative ceiling would fail without it."""
    am = _tensor(1.0)
    am[..., 0] *= k1_peak / np.abs(am[..., 0]).max()
    am[..., 1] *= k2_over_floor * 1e-10 / np.abs(am[..., 1]).max()
    bn = am.copy()
    bn[:, 0, 1] += 0.9e-10
    return bn, am


def test_the_floor_survives_a_columns_scaling():
    """A column judged alone is scaled to a peak of 1 and its atol with it, so
    the floor forgives the same noise: the row passes, and the floor decided
    it. An atol left unscaled would shrink the floor by the column's peak."""
    bn, am = _noisy_moving_column(1e3, 30.0)
    res = _compare(bn, am)
    assert res["columns_not_judged"] == []
    assert res["status"] == "pass"
    assert res["noise_decisive"] is True
    assert res["n_noise_rescued"] > 0


def test_a_floor_decisive_in_the_tensor_alone_is_decisive():
    """dx/dk2 at 5x its floor is not judged alone; the tensor's verdict passes
    it only on the floor. The row's PASS rested on the floor."""
    bn, am = _noisy_moving_column(1e-7, 5.0)
    res = _compare(bn, am)
    assert res["columns_not_judged"] == [1]
    assert res["status"] == "pass"
    assert res["noise_decisive"] is True
    # The judged column rescued nothing; the tensor's floor did, and says so.
    assert res["n_noise_rescued"] == 0 and res["n_noise_rescued_whole"] > 0
    assert "whole tensor ok" in res["comment"] and "DECISIVE" in res["comment"]


def test_a_column_left_out_is_named_only_if_the_tensor_fails_it():
    """dx/dk1 zeroed (judged, DIFF) beside a dx/dk2 at 5x its floor that is
    off past the floor on one cell, at 1e-12 of the tensor's peak, which the
    tensor forgives: only k1 is named."""
    am = _tensor(1.0)
    am[..., 1] *= 5e-10 / np.abs(am[..., 1]).max()
    bn = am.copy()
    bn[..., 0] = 0.0
    bn[3, 0, 1] += 1.5e-10
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["status"] == "diff"
    assert res["columns_diff"] == ["k1"]


@pytest.mark.parametrize("side", ["bngsim", "amici"])
def test_a_non_finite_cell_elsewhere_names_no_other_column(side):
    """One side's NaN or inf in dx/dk1 (judged, DIFF), beside a dx/dk2 at 5x
    its floor off past the floor on one cell, at 1e-12 of the tensor's peak:
    only k1 is named. differ never forgives a one-sided non-finite cell, so
    a k2 judged with k1 merely forgiven would be named too."""
    am = _tensor(1.0)
    am[..., 1] *= 5e-10 / np.abs(am[..., 1]).max()
    bn = am.copy()
    if side == "bngsim":
        bn[4, 1, 0] = np.nan
    else:
        am[4, 1, 0] = np.inf
    bn[0, 0, 1] += 1.5e-10
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["status"] == "diff"
    assert res["columns_diff"] == ["k1"]


def test_a_non_finite_cell_peaks_at_inf():
    sx = np.ones((2, 1, 3))
    sx[1, 0, 0] = np.nan
    sx[0, 0, 1] = -np.inf
    assert list(asens.column_peaks(sx)) == [np.inf, np.inf, 1.0]


def test_a_bad_test_row_leads_with_its_key_in_the_matrix(tmp_path):
    import json

    import generate_amici_sens_matrix as gsm

    row = {
        "model_id": "tiny",
        "method": "sens/staggered",
        "outcome": "BAD_TEST",
        "value": 0.0,
        "comment": "no parameter column carries a sensitivity 20x over its floor",
        "exception": "no parameter column carries a sensitivity 20x over its floor",
        "exception_class": "columns:none-judged",
        "extra": {"sens_method": "staggered"},
    }
    report = tmp_path / "report_sens.json"
    report.write_text(json.dumps({"_meta": {"tally": {"BAD_TEST": 1}}, "results": [row]}))
    gsm.generate_html(report, tmp_path / "sens.html")
    assert "[columns:none-judged] no parameter column" in (tmp_path / "sens.html").read_text()
