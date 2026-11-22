"""``Result.digest()`` and ``Result.save(".npz")`` (issue #1704).

A caller that cannot look at a plot (an AI agent) needs the dozen numbers it
would read off one: first and last, the extremes and when, whether a column
turns or has settled, and whether anything is NaN, negative or stuck. These
tests hold each number to the arrays it is read from, and to the closed forms
of the bundled examples where they have one. The thresholds are the
steady-state code's (``criterion="scaled"``), so a column the digest calls
settled is one ``steady_state()`` agrees has arrived, and one it calls unsettled
is still moving by more than the run's tolerance.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import warnings

import bngsim
import numpy as np
import pytest
from bngsim import Result, Simulator, UsageError, _digest, examples

# The keys of each kind of column, which #1708's --json and #1709's tool server
# return: a key removed or renamed breaks their callers (DIGEST_VERSION).
RUN_KEYS = [
    "digest_version",
    "method",
    "stochastic",
    "n_runs",
    "t_start",
    "t_end",
    "n_points",
    "n_sensitivity_params",
    "n_sensitivity_ic_species",
    "n_notices",
    "notices",
    "kinds",
    "rtol",
    "atol",
]
FLAGS = ["non_finite", "t_non_finite", "negative", "constant"]
DETERMINISTIC_KEYS = [
    "name",
    "first",
    "last",
    "min",
    "t_min",
    "max",
    "t_max",
    "monotone",
    "n_interior_extrema",
    "tail_from",
    "tail_change",
    "settled",
    *FLAGS,
]
STOCHASTIC_KEYS = ["name", "first", "last", "min", "max", *FLAGS]
STACK_KEYS = [
    "name",
    "last_n_finite",
    "last_mean",
    "last_std",
    "last_min",
    "last_q05",
    "last_q25",
    "last_median",
    "last_q75",
    "last_q95",
    "last_max",
    "min",
    "max",
    *FLAGS,
]


def _ode(name: str, t_end: float = 20.0, n_points: int = 201) -> Result:
    return Simulator(examples.load(name), method="ode").run(t_span=(0, t_end), n_points=n_points)


def _row(d: dict, name: str, kind: str = "observables") -> dict:
    (row,) = [r for r in d[kind]["columns"] if r["name"] == name]
    return row


def _synthetic(
    columns: dict[str, np.ndarray],
    t: np.ndarray,
    *,
    method: str | None = "ode",
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> Result:
    """A Result of given observable columns whose record names *method* and
    tolerances, as a file's record does."""
    names = list(columns)
    obs = np.column_stack([columns[n] for n in names]) if names else np.zeros((t.size, 0))
    r = Result(
        _time=t,
        _species=obs.copy(),
        _observables=obs,
        _species_names=names,
        _observable_names=names,
    )
    if method is not None:
        r._record_frozen = {"run": {"method": method, "rtol": rtol, "atol": {"scalar": atol}}}
    return r


def _stack(t: np.ndarray, block: np.ndarray) -> Result:
    """A stack of runs of one observable ``y``, with no record."""
    return Result(
        _time=t, _species=block, _observables=block, _species_names=["y"], _observable_names=["y"]
    )


# ─── The shape of the dict, which callers build on ───────────────────────────


def test_a_deterministic_run_has_every_key_in_order_and_strict_json():
    d = _ode("chain").digest(species=True)
    assert list(d)[: len(RUN_KEYS)] == RUN_KEYS
    assert d["digest_version"] == 1
    assert d["kinds"] == ["observables", "species"]
    assert list(d)[len(RUN_KEYS) :] == ["observables", "species"]
    for kind in d["kinds"]:
        assert list(d[kind]) == ["n_columns", "n_left_out", "columns"]
        for row in d[kind]["columns"]:
            assert list(row) == DETERMINISTIC_KEYS
    # Strict JSON (no NaN), as #1708's --json prints it, and back unchanged.
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert isinstance(d, dict)


def test_the_digest_is_deterministic():
    r = _ode("chain")
    assert r.digest(species=True) == r.digest(species=True)
    assert str(r.digest()) == str(r.digest())


def test_the_run_block_says_the_window_method_and_points():
    d = _ode("decay", t_end=7.5, n_points=16).digest()
    assert d["method"] == "ode"
    assert d["stochastic"] is False
    assert d["n_runs"] is None
    assert (d["t_start"], d["t_end"], d["n_points"]) == (0.0, 7.5, 16)
    assert d["kinds"] == ["observables"]
    assert (d["rtol"], d["atol"]) == (1e-8, 1e-8)
    assert "species" not in d


# ─── Each number against the arrays and the closed forms ─────────────────────


def test_first_last_and_extremes_are_the_arrays_own():
    r = _ode("chain")
    d = r.digest(species=True)
    t = r.time
    for kind, block, names in (
        ("observables", r.observables, r.observable_names),
        ("species", r.species, r.species_names),
    ):
        arr = np.asarray(block)
        for j, name in enumerate(names):
            row = _row(d, name, kind)
            col = arr[:, j]
            assert row["first"] == col[0]
            assert row["last"] == col[-1]
            assert row["min"] == col.min()
            assert row["max"] == col.max()
            assert row["t_min"] == t[np.argmin(col)]
            assert row["t_max"] == t[np.argmax(col)]


def test_values_match_the_closed_form():
    t_end = 20.0
    r = _ode("chain", t_end=t_end)
    d = r.digest()
    exact = examples.exact("chain", np.array([0.0, t_end]))
    for name in ("A", "B", "C"):
        row = _row(d, name)
        assert row["first"] == pytest.approx(exact[name][0], rel=1e-6, abs=1e-9)
        assert row["last"] == pytest.approx(exact[name][-1], rel=1e-5, abs=1e-6)
    # B = 100·k1/(k2-k1)·(e^{-k1 t} - e^{-k2 t}) peaks once, at
    # t* = ln(k1/k2)/(k1-k2) = 2 ln 2 for k1 = 1, k2 = 0.5.
    b = _row(d, "B")
    t_star = 2 * math.log(2)
    assert abs(b["t_max"] - t_star) <= 0.1 / 2 + 1e-12  # one sample of 0.1
    assert b["n_interior_extrema"] == 1 and b["monotone"] is False
    assert _row(d, "A")["monotone"] is True and _row(d, "A")["n_interior_extrema"] == 0
    assert _row(d, "C")["monotone"] is True


def test_turns_are_counted_on_a_sinusoid():
    # x = cos t, y = -sin t: turns at kπ and at π/2 + kπ.
    m = bngsim.Model.from_antimony_string("x' = y; y' = -x; x = 1; y = 0")
    for t_end, nx, ny in ((10.0, 3, 3), (12.0, 3, 4), (3.0, 0, 1)):
        d = (
            Simulator(m.clone() if hasattr(m, "clone") else m, method="ode")
            .run(t_span=(0, t_end), n_points=1001)
            .digest()
        )
        assert _row(d, "x")["n_interior_extrema"] == nx, t_end
        assert _row(d, "y")["n_interior_extrema"] == ny, t_end
        assert _row(d, "x")["negative"] is (t_end > math.pi / 2)


def test_a_dose_is_two_turns():
    # bolus: A decays, and a dose at t = 5 lifts it: a minimum just before the
    # dose and a maximum at it.
    d = _ode("bolus").digest()
    a = _row(d, "A")
    assert a["n_interior_extrema"] == 2
    assert a["t_max"] == pytest.approx(5.0)


def test_tail_change_is_the_largest_move_in_the_last_tenth_over_the_range():
    r = _ode("decay", t_end=20.0, n_points=201)
    row = _row(r.digest(), "A")
    col = np.asarray(r.observables)[:, 0]
    tail = r.time >= 18.0
    expected = np.abs(col[tail] - col[-1]).max() / (col.max() - col.min())
    assert row["tail_change"] == pytest.approx(expected, rel=1e-12)
    assert row["settled"] is False


# ─── "Settled" is the steady-state code's criterion ──────────────────────────


def test_settled_agrees_with_steady_state():
    # schlogl rests at X = 100; decay is still moving at t = 20, by about
    # 9% of its last value over the last tenth, far above rtol = 1e-8.
    r = _ode("schlogl")
    row = _row(r.digest(), "X")
    assert row["settled"] is True
    assert row["last"] == pytest.approx(100.0, rel=1e-6)
    ss = Simulator(examples.load("schlogl"), method="ode").steady_state()
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(row["last"], rel=1e-6)
    assert _row(_ode("decay").digest(), "A")["settled"] is False


def test_a_decay_below_the_tolerance_floor_is_settled():
    # A = 100 e^{-t} integrated at atol 1e-8: by t = 100 it moves by under
    # 1e-9, and by t = 200 it sits near -1e-10, which is above
    # -min(atol, rtol·largest) = -1e-8 and so not "negative".
    row = _row(_ode("decay", t_end=100.0, n_points=501).digest(), "A")
    assert row["settled"] is True
    assert row["monotone"] is True
    r = _ode("decay", t_end=200.0, n_points=501)
    row = _row(r.digest(), "A")
    assert row["settled"] is True and row["negative"] is False
    assert np.asarray(r.observables).min() < 0  # what "negative" lets pass


def test_the_tolerance_is_the_runs_own():
    # One column moving 1e-9 relative over the last tenth: settled at the
    # default rtol of 1e-8, not at 1e-12 (floored at 1e-10, still under it).
    t = np.linspace(0, 10, 101)
    y = 1.0 + 1e-9 * (t >= 9.5)
    loose = _synthetic({"y": y}, t, rtol=1e-8, atol=1e-30)
    tight = _synthetic({"y": y}, t, rtol=1e-12, atol=1e-30)
    assert _row(loose.digest(), "y")["settled"] is True
    assert _row(tight.digest(), "y")["settled"] is False
    # rtol is floored at the criterion's 1e-10: a 1e-11 move is within it.
    y2 = 1.0 + 1e-11 * (t >= 9.5)
    assert _row(_synthetic({"y": y2}, t, rtol=1e-14, atol=1e-30).digest(), "y")["settled"] is True


def test_the_tail_spans_the_whole_last_tenth_however_the_grid_falls():
    # Five points over [0, 10]: the last tenth starts at 9, so the tail is read
    # from 7.5, the last sample at or before it.
    r = _ode("decay", t_end=10.0, n_points=5)
    col = np.asarray(r.observables)[:, 0]
    row = _row(r.digest(), "A")
    assert row["tail_change"] == pytest.approx((col[3] - col[4]) / (col[0] - col[4]), rel=1e-12)
    assert row["settled"] is False and row["monotone"] is True and row["tail_from"] == 7.5
    # Two samples a nanosecond apart at the end of [0, 3] are no tail: the
    # column, at 22 and falling, is read from t = 2.
    sim = Simulator(examples.load("decay"), method="ode")
    r = sim.run(sample_times=[0, 1, 2, 3 - 1e-9, 3])
    col = np.asarray(r.observables)[:, 0]
    row = _row(r.digest(), "A")
    assert row["settled"] is False and row["tail_from"] == 2.0
    assert row["tail_change"] == pytest.approx((col[2] - col[4]) / (col[0] - col[4]), rel=1e-12)
    # A grid of 19 intervals reads as one of 10 or 100 does, to within the
    # one sample before the cut it adds.
    by_grid = [
        _row(_ode("decay", t_end=10.0, n_points=n).digest(), "A")["tail_change"]
        for n in (11, 20, 101)
    ]
    assert by_grid[0] == pytest.approx(by_grid[2], rel=1e-5)
    assert by_grid[1] >= by_grid[2]
    # A non-zero start: the tenth is of the window, not of t.
    r = sim.run(t_span=(100, 110), n_points=11, start="initial")
    t = r.time
    assert t[0] == 100
    col = np.asarray(r.observables)[:, 0]
    row = _row(r.digest(), "A")
    assert row["tail_change"] == pytest.approx(abs(col[9] - col[10]) / np.ptp(col), rel=1e-12)


def test_a_model_in_small_units_is_read_at_its_own_scale():
    # Issue #1001's trap: a model whose values sit near 1e-9 is held to
    # rtol·largest, not to the scalar atol of 1e-8, which is larger than the
    # whole column. Its turns count, and a move of 1e-12 is not "settled".
    t = np.linspace(0, 10, 1001)
    y = 1e-9 * (1.0 + 0.5 * np.sin(t))
    row = _row(_synthetic({"y": y}, t, atol=1e-8).digest(), "y")
    assert row["n_interior_extrema"] == 3 and row["settled"] is False
    assert row["constant"] is False
    still = 1e-9 + 1e-12 * (t >= 9.5)
    row = _row(_synthetic({"y": still}, t, atol=1e-8).digest(), "y")
    assert row["settled"] is False and row["constant"] is False
    # Large values: a move of 1e-4 at 1e6 is under rtol·|y| = 1e-2.
    big = 1e6 + 1e-4 * np.sin(37 * t)
    row = _row(_synthetic({"y": big}, t, atol=1e-8).digest(), "y")
    assert row["constant"] is True and row["monotone"] is True and row["settled"] is True
    # At rtol 1e-12 (read as 1e-10) and atol 1e-3, the floor is
    # rtol·largest = 1e-4 and the tolerance at 1e6 about 2e-4: a wobble of
    # 2e-5 is none, one of 2e-3 is many.
    for amp, flat in ((1e-5, True), (1e-3, False)):
        loose = _synthetic({"y": 1e6 + amp * np.sin(5 * t)}, t, rtol=1e-12, atol=1e-3)
        assert _row(loose.digest(), "y")["monotone"] is flat, amp


def test_a_species_is_held_to_its_own_atol_and_an_observable_to_the_runs_rule():
    # A species is held to its per-species atol as the run applied it; an
    # observable, whose species the record does not name, to what the run's
    # rule gives a species of its size: min(atol, rtol·max(its largest,
    # 1e-6·largest)). A column at 1e-12 beside one at 1 is held to 1e-14.
    t = np.linspace(0, 10, 1001)
    wobble = 1e-12 * np.sin(5 * t)
    r = _synthetic({"big": np.ones_like(t), "y": wobble, "z": wobble}, t, atol=1e-8)
    r._record_frozen["run"]["atol"]["per_species"] = [1e-8, 1e-14, 1e-8]
    d = r.digest(species=True)
    assert _row(d, "y", "species")["n_interior_extrema"] == 16  # 5t = π/2 + kπ, k = 0..15
    assert _row(d, "y", "species")["negative"] is True
    assert _row(d, "z", "species")["monotone"] is True  # its own 1e-8
    for name in ("y", "z"):
        assert _row(d, name, "observables")["n_interior_extrema"] == 16
    # A vector atol a caller chose does not reach the observables: the
    # observable of a species held to 1e-30 is still held to the run's rule.
    r = _synthetic({"big": np.ones_like(t), "y": 1e-15 * np.sin(5 * t)}, t, atol=1e-8)
    r._record_frozen["run"]["atol"]["per_species"] = [1e-30, 1e-30]
    d = r.digest(species=True)
    assert _row(d, "y", "species")["n_interior_extrema"] == 16
    assert _row(d, "y", "observables")["monotone"] is True  # rtol·1e-6·largest = 1e-14


def test_a_vector_atol_does_not_leak_into_other_observables():
    # chain at atol [1e-3, 1e-6, 1e-30]: observable A, which is species A
    # alone, reads as species A does, not at C's 1e-30.
    r = Simulator(examples.load("chain"), method="ode").run(
        t_span=(0, 60), n_points=601, atol=[1e-3, 1e-6, 1e-30]
    )
    d = r.digest(species=True)
    a = _row(d, "A")
    assert a["monotone"] is True and a["negative"] is False and a["settled"] is True
    assert _row(d, "A()", "species")["monotone"] is True


def test_an_observable_that_is_one_species_reads_as_the_species():
    # A model in small units, whose per-species atol the run tightened: an
    # observable of A must not read "never moves" where the species does not.
    m = bngsim.Model.from_antimony_string(
        "compartment c=1e-6; species A in c, B in c, $S in c; A=1e-9; B=0; S=1;"
        " J: A->B; k*A; k=1; S' = 0.1"
    )
    r = Simulator(m, method="ode").run(t_span=(0, 10), n_points=101)
    d = r.digest(species=True)
    for name in ("A", "B"):
        obs, sp = _row(d, name), _row(d, name, "species")
        assert obs["constant"] is False and sp["constant"] is False
        assert {k: obs[k] for k in ("monotone", "settled", "negative")} == {
            k: sp[k] for k in ("monotone", "settled", "negative")
        }


def test_tail_change_reads_the_whole_tail():
    # The last tenth starts and ends at 1 and swings to 1.5 in between.
    t = np.linspace(0, 10, 101)
    y = np.ones_like(t)
    y[95] = 1.5
    y[0] = 0.0
    row = _row(_synthetic({"y": y}, t).digest(), "y")
    assert row["tail_change"] == pytest.approx(0.5 / 1.5)
    assert row["settled"] is False


# ─── Noise on a deterministic column is not a turn ───────────────────────────


def test_wobble_under_the_tolerance_is_no_turn():
    # A column at rest at 1 with an integrator's wobble: under rtol·|y| +
    # floor = 2e-8 it is no turn; a thousand times larger, it is many.
    t = np.linspace(0, 10, 1001)
    row = _row(_synthetic({"y": 1.0 + 1e-10 * np.sin(37 * t)}, t).digest(), "y")
    assert row["monotone"] is True and row["n_interior_extrema"] == 0
    assert row["settled"] is True and row["constant"] is True
    row = _row(_synthetic({"y": 1.0 + 1e-7 * np.sin(37 * t)}, t).digest(), "y")
    assert row["n_interior_extrema"] == 118  # 37t = π/2 + kπ, k = 0..117
    assert row["monotone"] is False and row["settled"] is False and row["constant"] is False
    # A rise that the wobble rides on stays monotone while the rise is steeper.
    rising = 1.0 - np.exp(-t)
    assert _row(_synthetic({"y": rising + 1e-6 * np.sin(37 * t)}, t).digest(), "y")["monotone"]


def test_the_turn_count_does_not_depend_on_the_phase_the_column_starts_in():
    # Swings of 1.8 against a tolerance of 1: started at 0 or at 0.9, the
    # extremes reached and left by more than 1 are the same four (the last
    # -0.9 is left by 0.9 alone).
    sine = np.array([0, 0.9, -0.9, 0.9, -0.9, 0.9, -0.9, 0.0])
    cosine = np.array([0.9, -0.9, 0.9, -0.9, 0.9, -0.9, 0.0])
    assert _digest._n_turns(sine, 0.0, 1.0) == 4
    assert _digest._n_turns(cosine, 0.0, 1.0) == 4
    assert _digest._n_turns(-sine, 0.0, 1.0) == 4
    t = np.linspace(0, 20, 2001)
    for y in (100 + 0.09 * np.sin(t), 100 + 0.09 * np.cos(t)):
        row = _row(_synthetic({"y": y}, t, rtol=1e-3, atol=1e-8).digest(), "y")
        assert row["monotone"] is False and row["n_interior_extrema"] in (5, 6)


def test_a_turn_is_counted_where_the_swing_back_just_exceeds_the_tolerance():
    # Up to 1, back by d, up again: one turn back is a maximum and the next a
    # minimum, so 2 interior extrema, where d exceeds rtol·1 + min(atol,
    # rtol·largest) = 2e-8; none where d is under it.
    t = np.arange(5.0)
    for d_back, n in ((3e-8, 2), (1e-8, 0)):
        y = np.array([0.0, 1.0, 1.0 - d_back, 1.0 - d_back, 2.0])
        assert _row(_synthetic({"y": y}, t).digest(), "y")["n_interior_extrema"] == n, d_back
    assert _digest._n_turns(np.array([0.0, 1.0, 0.0, 1.0, 0.0]), 1e-8, 1e-8) == 3
    assert _digest._n_turns(np.array([1.0, 1.0, 1.0]), 1e-8, 0.0) == 0
    assert _digest._n_turns(np.array([0.0, 2.0, 1.0]), 0.0, 0.5) == 1
    assert _digest._n_turns(np.array([0.0, 2.0, 1.6]), 0.0, 0.5) == 0
    # The first move is under the tolerance, then a fall: no turn at the start.
    assert _digest._n_turns(np.array([1.0, 1.0 + 1e-9, 0.0]), 1e-8, 1e-8) == 0
    # A fall then a rise, from below: one minimum.
    assert _digest._n_turns(np.array([1.0, 0.0, 1.0]), 1e-8, 1e-8) == 1
    # The extreme a turn is measured from follows the column past a dip too
    # small to count: from 3 back to 2.2 is a turn of 0.8, though 2.2 is above 1.
    assert _digest._n_turns(np.array([0.0, 1.0, 0.9, 3.0, 2.2]), 0.0, 0.5) == 1
    assert _digest._n_turns(np.array([0.0, -1.0, -0.9, -3.0, -2.2]), 0.0, 0.5) == 1


# ─── Flags ───────────────────────────────────────────────────────────────────


def test_flags_non_finite_negative_and_constant():
    t = np.linspace(0, 4, 5)
    cols = {
        "nan_at_2": np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
        "inf_at_3": np.array([1.0, 2.0, 3.0, np.inf, 5.0]),
        "all_nan": np.full(5, np.nan),
        "dust_negative": np.array([1.0, 0.5, -1e-12, 0.2, 0.1]),
        "truly_negative": np.array([1.0, 0.5, -0.01, 0.2, 0.1]),
        "constant": np.full(5, 3.0),
        "moves_by_dust": 3.0 + np.array([0, 1e-12, 0, 0, 0]),
        "constant_but_nan": np.array([3.0, 3.0, np.nan, 3.0, 3.0]),
        "minus_inf_at_1": np.array([1.0, -np.inf, 3.0, 4.0, 5.0]),
    }
    d = _synthetic(cols, t).digest()
    json.dumps(d, allow_nan=False)
    row = _row(d, "nan_at_2")
    assert row["non_finite"] is True and row["t_non_finite"] == 2.0
    assert (row["min"], row["max"], row["t_min"], row["t_max"]) == (1.0, 5.0, 0.0, 4.0)
    assert row["monotone"] is None and row["settled"] is None and row["constant"] is False
    row = _row(d, "inf_at_3")
    assert row["non_finite"] is True and row["t_non_finite"] == 3.0 and row["max"] == 5.0
    row = _row(d, "all_nan")
    assert row["first"] is None and row["min"] is None and row["t_min"] is None
    assert row["t_non_finite"] == 0.0 and row["negative"] is False
    assert _row(d, "dust_negative")["negative"] is False
    assert _row(d, "truly_negative")["negative"] is True
    assert _row(d, "constant")["constant"] is True
    assert _row(d, "constant")["settled"] is True
    assert _row(d, "constant")["tail_change"] == 0.0
    assert _row(d, "moves_by_dust")["constant"] is True
    assert _row(d, "truly_negative")["constant"] is False
    assert _row(d, "constant_but_nan")["constant"] is False
    row = _row(d, "minus_inf_at_1")
    assert (row["min"], row["t_min"], row["t_non_finite"]) == (1.0, 0.0, 1.0)
    assert row["negative"] is False
    lines = str(d).splitlines()
    assert [x for x in lines if x.startswith("  truly_negative:")][0].endswith("FLAGS: negative")
    assert [x for x in lines if x.startswith("  constant:")][0].endswith("FLAGS: never moves")
    assert "FLAGS: non-finite from t=2" in str(d)
    bad = ("nan_at_2", "inf_at_3", "all_nan", "constant_but_nan", "minus_inf_at_1")
    for name in cols:
        assert _row(d, name)["non_finite"] is (name in bad)


def test_a_stochastic_run_flags_any_negative_and_any_move():
    t = np.linspace(0, 4, 5)
    d = _synthetic(
        {
            "neg": np.array([1.0, 0.0, -1e-12, 0.0, 1.0]),
            "moves": 3 + np.array([0, 1e-12, 0, 0, 0]),
        },
        t,
        method="ssa",
    ).digest()
    assert _row(d, "neg")["negative"] is True
    assert _row(d, "moves")["constant"] is False


# ─── The size bound ──────────────────────────────────────────────────────────


def test_past_max_columns_the_largest_ranges_are_listed_in_column_order():
    t = np.linspace(0, 1, 11)
    rng = np.random.default_rng(7)
    ranges = rng.permutation(np.arange(1.0, 51.0))
    cols = {f"c{j:02d}": ranges[j] * t for j in range(50)}
    cols["c10"] = cols["c10"].copy()
    cols["c10"][4] = np.nan  # a non-finite column comes first whatever its range
    cols["c10"] *= 1e-6
    d = _synthetic(cols, t).digest(max_columns=5)
    block = d["observables"]
    assert block["n_columns"] == 50 and block["n_left_out"] == 45
    names = [r["name"] for r in block["columns"]]
    finite_ranges = {f"c{j:02d}": ranges[j] for j in range(50) if j != 10}
    top4 = sorted(finite_ranges, key=lambda n: -finite_ranges[n])[:4]
    assert names == sorted(["c10", *top4])
    assert "observables: 5 of 50 (45 left out; listed are any with a non-finite" in str(d)
    # The default bound is 20, and 0 lists none but still counts them.
    assert len(_synthetic(cols, t).digest()["observables"]["columns"]) == 20
    zero = _synthetic(cols, t).digest(max_columns=0)["observables"]
    assert zero["columns"] == [] and zero["n_left_out"] == 50
    assert len(_synthetic(cols, t).digest(max_columns=500)["observables"]["columns"]) == 50


def test_ties_in_range_go_to_the_earlier_column():
    t = np.linspace(0, 1, 3)
    d = _synthetic({"a": t, "b": t, "c": 2 * t, "d": t}, t).digest(max_columns=2)
    assert [r["name"] for r in d["observables"]["columns"]] == ["a", "c"]


def test_a_large_result_stays_small_and_cheap():
    t = np.linspace(0, 100, 1001)
    k = np.linspace(0.01, 3, 300)
    cols = {f"s{j}": np.exp(-k[j] * t) * np.cos(k[j] * t) for j in range(300)}
    r = _synthetic(cols, t)
    start = time.process_time()
    d = r.digest()
    spent = time.process_time() - start
    assert len(json.dumps(d)) < 12_000
    assert len(str(d)) < 8_000
    assert spent < 2.0, spent  # about 2 ms here; a loose bound for slow runners


# ─── No reading of noise ─────────────────────────────────────────────────────


def test_one_stochastic_run_gets_values_and_flags_only():
    m = examples.load("schlogl")
    r = Simulator(m, method="ssa").run(t_span=(0, 20), n_points=201, seed=3)
    d = r.digest(species=True)
    assert d["stochastic"] is True and d["method"] == "ssa"
    col = np.asarray(r.observables)[:, 0]
    for kind in d["kinds"]:
        for row in d[kind]["columns"]:
            assert list(row) == STOCHASTIC_KEYS
    row = _row(d, "X")
    assert (row["first"], row["last"], row["min"], row["max"]) == (
        col[0],
        col[-1],
        col.min(),
        col.max(),
    )
    text = str(d)
    assert "extrem" not in text and "settled" not in text and "monotone" not in text


def test_a_result_whose_method_is_unknown_reads_nothing_of_its_shape():
    t = np.linspace(0, 1, 11)
    d = _synthetic({"y": t}, t, method=None).digest()
    assert d["method"] is None and d["stochastic"] is None
    assert list(_row(d, "y")) == STOCHASTIC_KEYS


# ─── A stack of runs ─────────────────────────────────────────────────────────


def test_a_replicate_stack_gives_statistics_at_the_last_time():
    sim = Simulator(examples.load("birth_death"), method="ssa")
    runs = sim.run_replicates(t_span=(0, 10), n_points=11, n_replicates=24, seed=11)
    stack = Result.squeeze(runs)
    d = stack.digest()
    assert d["n_runs"] == 24 and d["stochastic"] is True
    row = _row(d, "A")
    assert list(row) == STACK_KEYS
    last = np.array([np.asarray(r.observables)[-1, 0] for r in runs])
    assert row["last_n_finite"] == 24
    assert row["last_mean"] == pytest.approx(last.mean(), rel=1e-12)
    assert row["last_std"] == pytest.approx(last.std(ddof=1), rel=1e-12)
    assert row["last_min"] == last.min() and row["last_max"] == last.max()
    for key, q in (("q05", 0.05), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q95", 0.95)):
        assert row[f"last_{key}"] == pytest.approx(np.quantile(last, q), rel=1e-12)
    every = np.asarray(stack.observables)[:, :, 0]
    assert row["min"] == every.min() and row["max"] == every.max()
    # The mean is the closed form's, within the replicates' standard error.
    exact = examples.exact("birth_death", np.array([10.0]))["A"][0]
    assert abs(row["last_mean"] - exact) < 5 * row["last_std"] / math.sqrt(24)
    assert "over 24 finite runs" in str(d)


def test_a_batch_stack_of_deterministic_rows_gives_statistics_too():
    sim = Simulator(examples.load("decay"), method="ode")
    ks = [0.5, 1.0, 2.0]
    b = sim.run_batch(t_span=(0, 2), n_points=5, params=[{"k": k} for k in ks], squeeze=True)
    d = b.digest()
    row = _row(d, "A")
    assert list(row) == STACK_KEYS and d["stochastic"] is False and d["n_runs"] == 3
    exact = np.array([examples.exact("decay", np.array([2.0]), k=k)["A"][0] for k in ks])
    assert row["last_mean"] == pytest.approx(exact.mean(), rel=1e-6)
    assert row["last_median"] == pytest.approx(exact[1], rel=1e-6)


def test_a_stack_of_one_has_no_standard_deviation_and_counts_non_finite_runs():
    t = np.linspace(0, 1, 3)
    one = np.array([[[1.0], [2.0], [3.0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "degrees of freedom" warning
        row = _row(_stack(t, one).digest(), "y")
    assert row["last_std"] is None and row["last_mean"] == 3.0
    two = np.array([[[1.0], [2.0], [3.0]], [[1.0], [np.nan], [np.nan]]])
    d = _stack(t, two).digest()
    json.dumps(d, allow_nan=False)
    row = _row(d, "y")
    assert row["last_n_finite"] == 1 and row["non_finite"] is True and row["t_non_finite"] == 0.5
    three = np.array([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [np.inf]], [[1.0], [2.0], [5.0]]])
    row = _row(_stack(t, three).digest(), "y")
    assert row["last_n_finite"] == 2 and row["last_mean"] == 4.0 and row["last_max"] == 5.0
    assert row["t_non_finite"] == 1.0


# ─── What else it says, and what it refuses ──────────────────────────────────


def test_notices_are_the_results():
    m = examples.load("decay")
    sim = Simulator(m, method="ode")
    sim.run(t_span=(0, 1), n_points=3)
    with pytest.warns(bngsim.LiveStartWarning):
        r = sim.run(t_span=(0, 1), n_points=3)
    d = r.digest()
    assert d["notices"] == r.notices and d["notices"]
    assert f"notice {r.notices[0]['code']}:" in str(d)


def test_notices_are_bounded_and_counted():
    from bngsim._notices import Notice

    t = np.linspace(0, 1, 11)
    r = _synthetic({"y": t}, t)
    r._notices = tuple(Notice("Warn", f"c{i:02d}", "x" * 2000) for i in range(30))
    d = r.digest()
    assert d["n_notices"] == 30 and len(d["notices"]) == _digest.MAX_NOTICES == 20
    assert [n["code"] for n in d["notices"]] == [f"c{i:02d}" for i in range(20)]
    assert all(len(n["message"]) == _digest.MAX_NOTICE_CHARS for n in d["notices"])
    assert "10 more notices in Result.notices" in str(d)
    assert len(json.dumps(d)) < 14_000
    assert len(r.notices[0]["message"]) == 2000  # the result keeps them whole


def test_a_stack_of_rows_at_different_tolerances_judges_nothing_by_one():
    sim = Simulator(examples.load("chain"), method="ode")
    rows = [
        sim.run(t_span=(0, 5), n_points=6, rtol=rtol, atol=1e-3, start="initial")
        for rtol in (1e-3, 1e-4)
    ]
    d = Result.squeeze(rows).digest()
    assert d["rtol"] is None and d["atol"] is None and d["stochastic"] is False
    assert "rtol" not in str(d).splitlines()[0]
    same = Result.squeeze([rows[0], rows[0]]).digest()
    assert (same["rtol"], same["atol"]) == (1e-3, 1e-3)


def test_a_digest_survives_a_file(tmp_path):
    pytest.importorskip("h5py")
    sim = Simulator(examples.load("chain"), method="ode")
    r = sim.run(t_span=(0, 10), n_points=101, atol=[1e-9, 1e-10, 1e-11])
    r.save(tmp_path / "r.h5")
    back = Result.load(tmp_path / "r.h5")
    assert back.digest(species=True) == r.digest(species=True)
    stack = Result.squeeze([r, sim.run(t_span=(0, 10), n_points=101, start="initial")])
    stack.save(tmp_path / "s.h5")
    assert Result.load(tmp_path / "s.h5").digest() == stack.digest()


def test_a_model_with_no_observables_lists_its_species():
    t = np.linspace(0, 1, 3)
    sp = np.column_stack([t, 2 * t])
    r = Result(_time=t, _species=sp, _observables=np.zeros((3, 0)), _species_names=["a", "b"])
    d = r.digest()
    assert d["kinds"] == ["species"]
    assert [x["name"] for x in d["species"]["columns"]] == ["a", "b"]
    assert r.digest(species=True)["kinds"] == ["species"]


def test_an_empty_result_has_a_run_block_and_no_columns():
    r = Result(_time=np.zeros(0), _species=np.zeros((0, 0)), _observables=np.zeros((0, 0)))
    d = r.digest()
    assert d["n_points"] == 0 and d["t_start"] is None
    json.dumps(d, allow_nan=False)
    assert "no time points" in str(d)


def test_sensitivities_are_counted():
    r = _ode("decay")
    assert r.digest()["n_sensitivity_params"] == 0
    assert "sensitivities" not in str(r.digest())


@pytest.mark.parametrize(
    "kwargs",
    [{"max_columns": -1}, {"max_columns": 2.5}, {"max_columns": True}, {"species": "A"}],
)
def test_bad_arguments_are_refused(kwargs):
    with pytest.raises(UsageError):
        _ode("decay", n_points=3).digest(**kwargs)


def test_the_text_form_is_numbers_and_one_line_per_column():
    d = _ode("chain").digest(species=True)
    text = str(d)
    assert text == d.text()
    lines = text.splitlines()
    assert lines[0] == "ode run, t 0 to 20, 201 points, rtol 1e-08, atol 1e-08"
    assert lines[1] == "notices: none"
    assert len(lines) == 2 + 2 + 3 + 3  # a header per kind, a line per column
    assert lines[3].startswith("  A: first 100, last ")
    assert "1 interior extremum" in text
    for word in ("oscillat", "bistab", "steady state", "converge"):
        assert word not in text.lower()


def test_digest_costs_a_run_nothing():
    # Nothing of the digest is imported, let alone computed, by a run.
    code = (
        "import sys, bngsim\n"
        "from bngsim import examples\n"
        "sim = bngsim.Simulator(examples.load('decay'), method='ode')\n"
        "r = sim.run(t_span=(0, 1), n_points=3)\n"
        "r.record; r.summary(); repr(r)\n"
        "assert 'bngsim._digest' not in sys.modules, 'imported by a run'\n"
        "r.digest()\n"
        "assert 'bngsim._digest' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


# ─── .npz with NumPy alone ───────────────────────────────────────────────────


def test_save_npz_holds_the_arrays_names_and_record(tmp_path):
    r = _ode("chain")
    path = tmp_path / "run.npz"
    r.save(path)
    with np.load(path, allow_pickle=False) as f:
        assert np.array_equal(f["time"], r.time)
        assert np.array_equal(f["species"], np.asarray(r.species))
        assert np.array_equal(f["observables"], np.asarray(r.observables))
        assert f["expressions"].shape == np.asarray(r.expressions).shape
        assert list(f["species_names"]) == r.species_names
        assert list(f["observable_names"]) == r.observable_names
        record = json.loads(str(f["record"]))
        assert "sensitivity_params" not in f.files
    assert record == json.loads(json.dumps(r.record))
    assert record["run"]["method"] == "ode"


def test_save_npz_keeps_each_block_apart(tmp_path):
    t = np.linspace(0, 1, 4)
    sp = np.arange(8.0).reshape(4, 2)
    ob = 100 + np.arange(4.0).reshape(4, 1)
    ex = -np.arange(12.0).reshape(4, 3)
    r = Result(
        _time=t,
        _species=sp,
        _observables=ob,
        _expressions=ex,
        _species_names=["a", "b"],
        _observable_names=["o"],
        _expression_names=["e1", "e2", "e3"],
    )
    r.save(tmp_path / "r.npz")
    with np.load(tmp_path / "r.npz", allow_pickle=False) as f:
        assert np.array_equal(f["species"], sp)
        assert np.array_equal(f["observables"], ob)
        assert np.array_equal(f["expressions"], ex)
        assert list(f["expression_names"]) == ["e1", "e2", "e3"]
        assert list(f["species_names"]) == ["a", "b"]


def test_save_npz_needs_no_h5py(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    r = _ode("decay", n_points=5)
    r.save(tmp_path / "run.npz")
    with pytest.raises(ImportError):
        r.save(tmp_path / "run.h5")
    with np.load(tmp_path / "run.npz", allow_pickle=False) as f:
        assert np.array_equal(f["observables"], np.asarray(r.observables))


def test_save_npz_of_a_stack_and_an_upper_case_suffix(tmp_path):
    sim = Simulator(examples.load("birth_death"), method="ssa")
    stack = Result.squeeze(sim.run_replicates(t_span=(0, 1), n_points=3, n_replicates=3, seed=2))
    path = tmp_path / "STACK.NPZ"
    stack.save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["STACK.NPZ"]
    with np.load(path, allow_pickle=False) as f:
        assert f["observables"].shape == (3, 3, 1)
        assert np.array_equal(f["observables"], np.asarray(stack.observables))
        assert json.loads(str(f["record"]))["rows_differ"]


def test_save_npz_keeps_sensitivities(tmp_path):
    try:
        sim = Simulator(examples.load("decay"), method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 1), n_points=3)
    except bngsim.BngsimError as e:  # no compiler on this machine
        pytest.skip(f"forward sensitivities unavailable here: {e}")
    r.save(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz", allow_pickle=False) as f:
        assert list(f["sensitivity_params"]) == ["k"]
        assert np.array_equal(f["sensitivities_species"], r.sensitivities)
        if r.has_sensitivities_observables:
            assert np.array_equal(f["sensitivities_observables"], r.sensitivities_observables)
    assert r.digest()["n_sensitivity_params"] == 1


@pytest.mark.parametrize("name", ["out.csv", "out.TSV"])
def test_save_to_a_delimited_text_name_is_refused_with_to_csv(tmp_path, name):
    r = _ode("decay", n_points=3)
    with pytest.raises(UsageError, match="to_csv") as info:
        r.save(tmp_path / name)
    assert info.value.remedies[0]["call"] == "Result.to_csv"
    assert not (tmp_path / name).exists()


def test_load_of_an_npz_names_numpy_load(tmp_path):
    r = _ode("decay", n_points=3)
    r.save(tmp_path / "r.npz")
    with pytest.raises(UsageError, match="numpy.load") as info:
        Result.load(tmp_path / "r.npz")
    assert info.value.remedies[0]["call"] == "numpy.load"


def test_to_csv_needs_nothing_beyond_numpy(tmp_path, monkeypatch):
    # The issue's "a base install can neither tabulate nor save a Result" is
    # half stale: to_csv needs neither pandas nor h5py.
    monkeypatch.setitem(sys.modules, "pandas", None)
    monkeypatch.setitem(sys.modules, "h5py", None)
    r = _ode("chain", n_points=5)
    r.to_csv(tmp_path / "o.csv")
    back = np.loadtxt(tmp_path / "o.csv", delimiter=",", skiprows=1)
    assert np.allclose(back[:, 1:], np.asarray(r.observables), rtol=1e-11)
