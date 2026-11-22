"""``Result.digest(sensitivities=True)`` (issue #1883).

Which parameters move a column most, ranked so that the units of a parameter do
not decide it: by ``|p·dY/dp|``, which within one column is the order of the
scaled sensitivity ``(p/y)·dY/dp``. The oracles are the closed form of the
``decay`` example, ``A = A0·exp(-k·t)`` (``dln A/dln k = -k·t``, ``dln A/dln A0
= 1``), and synthetic results whose sensitivity arrays are given outright.
"""

from __future__ import annotations

import json
import math

import bngsim
import numpy as np
import pytest
from bngsim import Result, Simulator, UsageError, examples

ROW_KEYS = [
    "y",
    "resolved",
    "parameters",
    "n_parameters_left_out",
    "n_parameters_non_finite",
    "initial_amounts",
    "n_initial_amounts_left_out",
    "n_initial_amounts_non_finite",
]
ENTRY_KEYS = ["name", "dy_dp", "dy_dlnp", "dlny_dlnp", "one_sided"]
HEAD_KEYS = [
    "t",
    "max_parameters",
    "reason",
    "n_parameters",
    "n_initial_amounts",
    "n_at_zero",
    "at_zero",
    "n_unvalued",
    "unvalued",
    "n_initial_at_zero",
    "initial_at_zero",
    "n_initial_unvalued",
    "initial_unvalued",
    "n_one_sided",
    "one_sided",
]


def _decay(t_end: float = 10.0, k: float | None = None, **kw) -> Result:
    model = examples.load("decay")
    if k is not None:
        model.set_param("k", k)
    try:
        sim = Simulator(model, method="ode", **kw)
        return sim.run(t_span=(0, t_end), n_points=11)
    except bngsim.CodegenToolchainError:
        pytest.skip("no C compiler for forward sensitivities")


def _sens(
    y_last: list[float],
    d_last: list[list[float]],
    params: dict[str, float | None],
    *,
    ic: dict[str, float] | None = None,
    ic_last: list[list[float]] | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    record_params: bool = True,
) -> Result:
    """A two-point result of observables ``y0, y1, ...`` (also its species)
    whose last values are *y_last* and whose ``dY/dp`` at the last time is
    *d_last* (a row per column, a column per parameter of *params*)."""
    names = [f"y{i}" for i in range(len(y_last))]
    t = np.array([0.0, 1.0])
    vals = np.vstack([np.ones(len(y_last)), y_last])
    if ic:
        # the species' first value is its initial amount
        vals[0] = list(ic.values())
        names = list(ic)
    block = np.zeros((2, len(y_last), len(params)))
    block[-1] = np.array(d_last, dtype=float).reshape(len(y_last), len(params))
    kw = {}
    if ic is not None:
        icb = np.zeros((2, len(y_last), len(ic)))
        icb[-1] = np.array(ic_last, dtype=float)
        kw = dict(
            _sensitivities_ic=icb,
            _sensitivity_ic_species=list(ic),
            _observable_sensitivities_ic=icb.copy(),
        )
    r = Result(
        _time=t,
        _species=vals.copy(),
        _observables=vals,
        _species_names=names,
        _observable_names=names,
        _sensitivities=block,
        _sensitivity_params=list(params),
        _observable_sensitivities=block.copy(),
        **kw,
    )
    run = {"method": "ode", "rtol": rtol, "atol": {"scalar": atol}}
    if record_params:
        run["parameters"] = dict(params)
    r._record_frozen = {"run": run}
    return r


def _col(d: dict, name: str, kind: str = "observables") -> dict:
    (row,) = [r for r in d[kind]["columns"] if r["name"] == name]
    return row["sensitivity"]


# ─── Against the closed form ─────────────────────────────────────────────────


def test_decay_matches_the_closed_form():
    k, t = 0.5, 10.0
    r = _decay(t, sensitivity_params=["k", "A0"], sensitivity_ic=["A()"])
    d = r.digest(sensitivities=True, species=True)
    json.dumps(d, allow_nan=False)
    a = 100.0 * math.exp(-k * t)
    for kind, name in (("observables", "A"), ("species", "A()")):
        s = _col(d, name, kind)
        assert list(s) == ROW_KEYS
        assert s["resolved"] is True
        assert s["y"] == pytest.approx(a, rel=1e-6)
        (dk, da0) = s["parameters"]
        assert list(dk) == ENTRY_KEYS
        assert dk["name"] == "k" and da0["name"] == "A0"
        assert dk["dlny_dlnp"] == pytest.approx(-k * t, rel=1e-5)
        assert dk["dy_dp"] == pytest.approx(-t * a, rel=1e-5)
        assert dk["dy_dlnp"] == pytest.approx(-k * t * a, rel=1e-5)
        assert da0["dlny_dlnp"] == pytest.approx(1.0, rel=1e-6)
        (x0,) = s["initial_amounts"]
        assert x0["name"] == "A()" and x0["dlny_dlnp"] == pytest.approx(1.0, rel=1e-6)
        assert x0["one_sided"] is None
        assert s["n_parameters_left_out"] == s["n_parameters_non_finite"] == 0
    head = d["sensitivity"]
    assert list(head) == HEAD_KEYS
    assert head["t"] == t and head["reason"] is None
    assert head["n_parameters"] == 2 and head["n_initial_amounts"] == 1


def test_early_on_the_rank_turns_over_as_the_closed_form_says():
    # |dln A/dln k| = k·t, against 1 for A0: k leads past t = 1/k = 2.
    early = _col(_decay(1.0, sensitivity_params=["A0", "k"]).digest(sensitivities=True), "A")
    late = _col(_decay(10.0, sensitivity_params=["A0", "k"]).digest(sensitivities=True), "A")
    assert [e["name"] for e in early["parameters"]] == ["A0", "k"]
    assert [e["name"] for e in late["parameters"]] == ["k", "A0"]


def test_a_parameter_at_zero_is_listed_apart_not_ranked():
    # k = 0: A stays at A0 and dA/dk = -t·A0, but there is no dA/dln k.
    r = _decay(10.0, k=0.0, sensitivity_params=["k", "A0"])
    d = r.digest(sensitivities=True)
    s = _col(d, "A")
    assert [e["name"] for e in s["parameters"]] == ["A0"]
    assert d["sensitivity"]["at_zero"] == ["k"] and d["sensitivity"]["n_at_zero"] == 1
    assert r.sensitivities[-1, 0, 0] == pytest.approx(-1000.0, rel=1e-6)
    assert "parameters at 0, not ranked: k" in str(d)


def test_the_digest_survives_a_file(tmp_path):
    pytest.importorskip("h5py")
    r = _decay(10.0, sensitivity_params=["k", "A0"], sensitivity_ic=["A()"])
    r.save(tmp_path / "r.h5")
    back = Result.load(tmp_path / "r.h5")
    assert back.digest(sensitivities=True) == r.digest(sensitivities=True)


# ─── The rank and its bounds, on arrays given outright ───────────────────────


def test_the_rank_is_by_p_times_dy_dp_not_by_units():
    # Raw dY/dp ranks b (1e3) over a (10); weighed by p, a (10·100) leads b (1e3·1e-3).
    r = _sens([2.0], [[10.0, 1e3]], {"a": 100.0, "b": 1e-3})
    s = _col(r.digest(sensitivities=True), "y0")
    assert [e["name"] for e in s["parameters"]] == ["a", "b"]
    assert [e["dy_dlnp"] for e in s["parameters"]] == [1000.0, 1.0]
    assert [e["dlny_dlnp"] for e in s["parameters"]] == [500.0, 0.5]


def test_ranking_is_by_magnitude_and_keeps_the_sign():
    r = _sens([1.0], [[-5.0, 3.0, 4.0]], {"a": 1.0, "b": 1.0, "c": 1.0})
    s = _col(r.digest(sensitivities=True), "y0")
    assert [(e["name"], e["dlny_dlnp"]) for e in s["parameters"]] == [
        ("a", -5.0),
        ("c", 4.0),
        ("b", 3.0),
    ]


def test_ties_go_to_the_earlier_parameter():
    r = _sens([1.0], [[2.0, -2.0, 2.0]], {"c": 1.0, "a": 1.0, "b": 1.0})
    s = _col(r.digest(sensitivities=True), "y0")
    assert [e["name"] for e in s["parameters"]] == ["c", "a", "b"]


def test_past_max_parameters_the_rest_are_counted():
    params = {f"p{i}": 1.0 for i in range(8)}
    r = _sens([1.0], [[float(i) for i in range(8)]], params)
    s = _col(r.digest(sensitivities=True, max_parameters=3), "y0")
    assert [e["name"] for e in s["parameters"]] == ["p7", "p6", "p5"]
    assert s["n_parameters_left_out"] == 5
    s0 = _col(r.digest(sensitivities=True, max_parameters=0), "y0")
    assert s0["parameters"] == [] and s0["n_parameters_left_out"] == 8
    assert "(5 more)" in str(r.digest(sensitivities=True, max_parameters=3))


def test_a_value_within_its_tolerance_of_zero_has_no_scaled_value_but_a_rank():
    # y0's atol by the run's rule is min(1e-8, 1e-6·1) = 1e-8: y0 = 1e-9 is under it.
    r = _sens([1e-9, 1.0], [[1.0, -3.0], [1.0, 1.0]], {"a": 1.0, "b": 1.0})
    d = r.digest(sensitivities=True)
    s = _col(d, "y0")
    assert s["resolved"] is False
    assert [e["name"] for e in s["parameters"]] == ["b", "a"]
    assert all(e["dlny_dlnp"] is None for e in s["parameters"])
    assert [e["dy_dlnp"] for e in s["parameters"]] == [-3.0, 1.0]
    assert "y within its tolerance of 0" in str(d)
    assert _col(d, "y1")["resolved"] is True


def test_just_past_the_floor_is_resolved():
    r = _sens([2e-8, 1.0], [[1.0], [1.0]], {"a": 1.0})
    assert _col(r.digest(sensitivities=True), "y0")["resolved"] is True
    r = _sens([1e-8, 1.0], [[1.0], [1.0]], {"a": 1.0})
    assert _col(r.digest(sensitivities=True), "y0")["resolved"] is False


def test_a_species_is_resolved_against_its_own_atol():
    # A species held to atol 1e-3 at 1e-5 was not told from 0, though the
    # digest's floor min(atol, rtol·largest) = 1e-6 lies under it; one held to
    # 1e-8 was.
    r = _sens([1e-5, 1e-5, 1.0], [[1.0], [1.0], [1.0]], {"a": 1.0})
    r._record_frozen["run"]["atol"] = {
        "given": [1e-3, 1e-8, 1e-8],
        "scalar": 1e-8,
        "per_species": [1e-3, 1e-8, 1e-8],
    }
    d = r.digest(sensitivities=True, species=True)
    assert _col(d, "y0", "species")["resolved"] is False
    assert _col(d, "y1", "species")["resolved"] is True


def test_a_non_finite_derivative_is_counted_not_ranked():
    # An assignment-rule species whose row is refused holds NaN.
    r = _sens([1.0], [[np.nan, 2.0, np.inf]], {"a": 1.0, "b": 1.0, "c": 1.0})
    d = r.digest(sensitivities=True)
    json.dumps(d, allow_nan=False)
    s = _col(d, "y0")
    assert [e["name"] for e in s["parameters"]] == ["b"]
    assert s["n_parameters_non_finite"] == 2
    assert "2 not finite" in str(d)
    # neither is it one left out past the bound
    one = _col(r.digest(sensitivities=True, max_parameters=1), "y0")
    assert one["n_parameters_left_out"] == 0 and one["n_parameters_non_finite"] == 2


def test_a_parameter_the_record_gives_no_value_is_listed_apart():
    r = _sens([1.0], [[1.0, 2.0]], {"a": None, "b": 1.0})
    d = r.digest(sensitivities=True)
    assert [e["name"] for e in _col(d, "y0")["parameters"]] == ["b"]
    assert d["sensitivity"]["unvalued"] == ["a"]
    r = _sens([1.0], [[1.0, 2.0]], {"a": 1.0, "b": 1.0}, record_params=False)
    d = r.digest(sensitivities=True)
    assert _col(d, "y0")["parameters"] == []
    assert d["sensitivity"]["n_unvalued"] == 2


def test_the_lists_apart_are_bounded_and_counted():
    params = dict.fromkeys((f"p{i}" for i in range(30)), 0.0)
    r = _sens([1.0], [[1.0] * 30], params)
    head = r.digest(sensitivities=True)["sensitivity"]
    assert head["n_at_zero"] == 30 and len(head["at_zero"]) == 20
    assert "and 10 more" in str(r.digest(sensitivities=True))


def test_a_one_sided_column_is_ranked_with_its_side():
    r = _sens([1.0], [[1.0, 2.0]], {"a": 1.0, "b": 1.0})
    r._one_sided_sensitivities = [(0.5, {"a": "increasing"})]
    d = r.digest(sensitivities=True)
    s = _col(d, "y0")
    assert [(e["name"], e["one_sided"]) for e in s["parameters"]] == [
        ("b", None),
        ("a", "increasing"),
    ]
    assert d["sensitivity"]["one_sided"] == {"a": "increasing"}
    assert "a 1 (increasing side)" in str(d)


def test_initial_amounts_are_scaled_by_the_first_value_where_no_start_is_held():
    # x0 = 4 for s0 and 0 for s1: s1 has no dY/dln x0.
    r = _sens(
        [2.0, 1.0],
        [[0.0], [0.0]],
        {"a": 1.0},
        ic={"s0": 4.0, "s1": 0.0},
        ic_last=[[0.5, 7.0], [0.25, 3.0]],
    )
    d = r.digest(sensitivities=True)
    s = _col(d, "s0")
    assert [(e["name"], e["dy_dlnp"], e["dlny_dlnp"]) for e in s["initial_amounts"]] == [
        ("s0", 2.0, 1.0)
    ]
    assert d["sensitivity"]["initial_at_zero"] == ["s1"]


# ─── Where there is nothing to read ──────────────────────────────────────────


def test_a_run_without_sensitivities_says_so():
    r = Simulator(examples.load("decay"), method="ode").run(t_span=(0, 1), n_points=3)
    d = r.digest(sensitivities=True)
    assert d["sensitivity"]["reason"] == "the run computed no sensitivities"
    assert _col(d, "A") is None
    assert "sensitivities: none read" in str(d)
    json.dumps(d, allow_nan=False)


def test_a_stack_says_so():
    r = _sens([1.0], [[1.0]], {"a": 1.0})
    stack = Result.squeeze([r, r])
    d = stack.digest(sensitivities=True)
    assert d["sensitivity"]["reason"].startswith("a stack of runs")
    assert _col(d, "y0") is None


def test_a_column_whose_block_is_missing_has_none_for_it():
    r = _sens([1.0], [[1.0]], {"a": 1.0})
    r._observable_sensitivities = np.empty((0, 0, 0))
    d = r.digest(sensitivities=True, species=True)
    assert _col(d, "y0")["parameters"] is None
    assert _col(d, "y0", "species")["parameters"][0]["name"] == "a"
    assert "parameters: no block for this column" in str(d)


def test_without_the_option_nothing_changes():
    r = _sens([1.0], [[1.0]], {"a": 1.0})
    d = r.digest()
    assert "sensitivity" not in d
    assert all("sensitivity" not in c for c in d["observables"]["columns"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sensitivities": 1},
        {"max_parameters": -1},
        {"max_parameters": 2.0},
        {"max_parameters": True},
    ],
)
def test_bad_arguments_are_refused(kwargs):
    with pytest.raises(UsageError):
        _sens([1.0], [[1.0]], {"a": 1.0}).digest(**kwargs)


def test_initial_amounts_are_scaled_by_the_start_state():
    # The start state holds the species in order, then a promoted symbol.
    r = _sens(
        [2.0, 1.0],
        [[0.0], [0.0]],
        {"a": 1.0},
        ic={"s0": 4.0, "s1": 0.0},
        ic_last=[[0.5, 7.0], [0.25, 3.0]],
    )
    r._start_state = np.array([8.0, 1.0, 99.0])
    d = r.digest(sensitivities=True)
    s = _col(d, "s0")
    assert [(e["name"], e["dy_dlnp"]) for e in s["initial_amounts"]] == [("s1", 7.0), ("s0", 4.0)]
    assert d["sensitivity"]["n_initial_at_zero"] == 0


def test_an_event_at_the_first_time_does_not_move_the_initial_amount():
    # A = 10 declared; an event at t=0 adds 5, so A(t) = (A0 + 5)·exp(-k·t):
    # dA/dA0 = exp(-k·t), and the scale is the declared 10, not the 15 the
    # first sample holds.
    try:
        model = bngsim.Model.from_antimony_string(
            "species A; A = 10; k = 0.5; J1: A -> ; k*A;\nE: at (time >= 0), t0=false: A = A + 5;"
        )
        r = Simulator(model, method="ode", sensitivity_ic=["A"]).run(
            t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12
        )
    except (ImportError, bngsim.CodegenToolchainError) as e:
        pytest.skip(f"needs antimony and a C compiler: {e}")
    assert r.species[0, 0] == pytest.approx(15.0)
    (x0,) = _col(r.digest(sensitivities=True, species=True), "A", "species")["initial_amounts"]
    e2 = math.exp(-2.0)
    assert x0["dy_dp"] == pytest.approx(e2, rel=1e-6)
    assert x0["dy_dlnp"] == pytest.approx(10.0 * e2, rel=1e-6)
    assert x0["dlny_dlnp"] == pytest.approx(10.0 / 15.0, rel=1e-6)


def test_observables_and_species_read_their_own_blocks():
    try:
        m = examples.load("binding")
        r = Simulator(
            m,
            method="ode",
            sensitivity_params=["kf", "kr", "A0", "B0"],
            sensitivity_ic=["A(b)", "B(a)"],
        ).run(t_span=(0, 5), n_points=6)
    except bngsim.CodegenToolchainError:
        pytest.skip("no C compiler for forward sensitivities")
    d = r.digest(sensitivities=True, species=True, max_parameters=10)
    blocks = {
        "observables": (
            r.observable_names,
            r.sensitivities_observables,
            r.sensitivities_observables_ic,
        ),
        "species": (r.species_names, r.sensitivities, r.sensitivities_ic),
    }
    seen = 0
    for kind, (names, par, ic) in blocks.items():
        for row in d[kind]["columns"]:
            j = names.index(row["name"])
            s = row["sensitivity"]
            for e in s["parameters"]:
                assert e["dy_dp"] == par[-1, j, r.sensitivity_params.index(e["name"])]
                seen += 1
            for e in s["initial_amounts"]:
                assert e["dy_dp"] == ic[-1, j, r.sensitivity_ic_species.index(e["name"])]
                seen += 1
    assert seen > 10
    # an observable that is no species reads otherwise than any species row
    assert set(r.observable_names).isdisjoint(r.species_names)


def test_a_kink_past_the_last_time_does_not_make_a_column_one_sided():
    # A result cut short keeps the whole run's kinks.
    r = _sens([1.0], [[1.0, 2.0]], {"a": 1.0, "b": 1.0})
    r._one_sided_sensitivities = [(5.0, {"a": "increasing"})]
    d = r.digest(sensitivities=True)
    assert all(e["one_sided"] is None for e in _col(d, "y0")["parameters"])
    assert d["sensitivity"]["one_sided"] == {} and d["sensitivity"]["n_one_sided"] == 0


def test_kinks_on_opposite_sides_are_mixed_and_the_same_side_stays():
    r = _sens([1.0], [[1.0, 2.0]], {"a": 1.0, "b": 1.0})
    r._one_sided_sensitivities = [
        (0.2, {"a": "increasing", "b": "decreasing"}),
        (0.7, {"a": "decreasing", "b": "decreasing"}),
    ]
    d = r.digest(sensitivities=True)
    assert d["sensitivity"]["one_sided"] == {"a": "mixed", "b": "decreasing"}
    assert {e["name"]: e["one_sided"] for e in _col(d, "y0")["parameters"]} == {
        "a": "mixed",
        "b": "decreasing",
    }


def test_the_one_sided_list_is_bounded_and_counted():
    params = {f"p{i}": 1.0 for i in range(30)}
    r = _sens([1.0], [[1.0] * 30], params)
    r._one_sided_sensitivities = [(0.5, dict.fromkeys(params, "increasing"))]
    head = r.digest(sensitivities=True)["sensitivity"]
    assert head["n_one_sided"] == 30 and len(head["one_sided"]) == 20


def test_a_product_that_overflows_is_counted_not_ranked():
    r = _sens([1.0], [[1e300, 2.0]], {"a": 1e300, "b": 1.0})
    d = r.digest(sensitivities=True)
    json.dumps(d, allow_nan=False)
    s = _col(d, "y0")
    assert [e["name"] for e in s["parameters"]] == ["b"]
    assert s["n_parameters_non_finite"] == 1


def test_a_missing_block_leaves_the_other_ranked():
    r = _sens(
        [2.0, 1.0],
        [[3.0], [1.0]],
        {"a": 1.0},
        ic={"s0": 4.0, "s1": 1.0},
        ic_last=[[0.5, 7.0], [0.25, 3.0]],
    )
    r._observable_sensitivities_ic = np.empty((0, 0, 0))
    d = r.digest(sensitivities=True)
    s = _col(d, "s0")
    assert [e["name"] for e in s["parameters"]] == ["a"]
    assert s["initial_amounts"] is None and s["n_initial_amounts_left_out"] is None
    assert "initial amounts: no block for this column" in str(d)


def test_a_result_with_no_record_of_its_parameters_ranks_none_and_says_so():
    try:
        r = Simulator(examples.load("chain")).compute_all_sensitivities(
            t_span=(0, 10), n_points=11
        )
    except bngsim.CodegenToolchainError:
        pytest.skip("no C compiler for forward sensitivities")
    d = r.digest(sensitivities=True)
    assert d["sensitivity"]["n_unvalued"] == len(r.sensitivity_params) > 0
    text = str(d)
    assert "none ranked" in text and "parameters of no recorded value" in text


def test_no_sensitivities_says_so_once():
    r = Simulator(examples.load("chain"), method="ode").run(t_span=(0, 1), n_points=3)
    text = str(r.digest(sensitivities=True, species=True))
    assert text.count("sensitivities: none read") == 1
    assert "none for this column" not in text
