"""The discontinuity suite's frame, as #1913 extends it: Antimony models, exact
uniform grids, engines, steady states, refusals owed, failure types by name, a
window that follows its row, the rows a spec check reads, and the window peak.
Every rule is shown to fail on the defect it is there to see."""

from __future__ import annotations

import dataclasses

import _closed_forms as CF
import _judge as J
import _sweep as S
import bngsim
import numpy as np
import pytest
from _case import (
    Axes,
    CaseError,
    Judge,
    Known,
    Point,
    Sampled,
    SteadyState,
    Uniform,
    converge_method,
    point,
)
from _cases import BY_NAME
from _reference import ClosedForm, Limit, Owed, ReferenceUnavailable, window_peak


def _with(name, **change):
    return dataclasses.replace(BY_NAME[name], **change)


# ── Keys and the sweep ──────────────────────────────────────────────────────


def test_keys_name_an_engine_only_where_it_is_not_the_default():
    c = BY_NAME["diffrax_gaussian"]
    d = point(1e-8, Uniform(3), engine="diffrax")
    assert c.key(d) == "diffrax_gaussian|rtol=1e-08,grid=u3,sens=none,engine=diffrax"
    assert c.key(point(1e-8, Uniform(3))) == "diffrax_gaussian|rtol=1e-08,grid=u3,sens=none"
    assert converge_method(d.group) == "converge,grid=u3,sens=none,engine=diffrax"
    assert converge_method(point(1e-8, Uniform(3)).group) == "converge,grid=u3,sens=none"


@pytest.mark.parametrize("name", ["diffrax_gaussian", "cube_switch", "steady_counter_window"])
def test_the_sweep_is_the_product_of_every_axis_engines_included(name):
    c = BY_NAME[name]
    a = c.axes
    pts = c.points()
    n = len(a.params) * len(a.sens) * len(a.engines) * len(a.grids) * len(a.rtol)
    assert len(pts) == len(set(pts)) == n
    assert {p.engine for p in pts} == set(a.engines)


def test_a_window_follows_its_row():
    c = BY_NAME["late_switch"]
    for row in c.rows():
        sigma = dict(row)["sigma"]
        assert c.window(c.values(row)) == (0.0, sigma + 2.0)
    assert c.window() == (0.0, c.params["sigma"] + 2.0)


def test_the_spec_check_reads_the_rows_a_case_names():
    c = BY_NAME["cube_switch"]
    assert c.spec_check_rows() == [tuple(sorted(c.spec_rows[0].items()))]
    assert BY_NAME["time_window"].spec_check_rows() == BY_NAME["time_window"].rows()


# ── Grids ───────────────────────────────────────────────────────────────────


def test_a_sampled_grid_asks_for_uniform_times_and_adds_none():
    assert Sampled(4).request((0.0, 6.0), (2.5, 3.0)) == {"sample_times": [0.0, 2.0, 4.0, 6.0]}
    assert SteadyState("newton").request((0.0, 1.0), (0.5,)) == {"method": "newton"}


def test_a_sampled_grid_sees_a_row_moved_inside_it_where_a_uniform_one_cannot():
    """#1799's row: asked for at 3e-11 and recorded from 2.95e-11."""
    span = (0.0, 1e-10)
    asked = Sampled(11).request(span, ())
    got = list(asked["sample_times"])
    got[3] = 2.95e-11
    assert "asked for at t=3e-11 came back at t=2.95e-11" in S.grid_mismatch(asked, got, span)
    assert S.grid_mismatch(Uniform(11).request(span, ()), got, span) is None


# ── Validation ──────────────────────────────────────────────────────────────


def _steady(**axes):
    base = BY_NAME["steady_counter_window"].axes
    return dataclasses.replace(base, **axes)


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("time_window", dict(axes=Axes(engines=("ode", "nope"))), "not among"),
        ("time_window", dict(axes=Axes(engines=("ode", "ode"))), "once each"),
        ("time_window", dict(axes=Axes(engines=())), "once each"),
        (
            "diffrax_gaussian",
            dict(axes=Axes(grids=(Uniform(3),), sens=((), ("kx",)), engines=("ode", "diffrax"))),
            "plain runs only",
        ),
        (
            "diffrax_gaussian",
            dict(axes=Axes(grids=(Sampled(3),), engines=("ode", "diffrax"))),
            "plain runs only",
        ),
        (
            "steady_counter_window",
            dict(axes=_steady(grids=(SteadyState("integration"), Uniform(4)))),
            "mixes steady states",
        ),
        ("steady_counter_window", dict(axes=_steady(sens=(("kc",),))), "plain runs"),
        ("steady_counter_window", dict(axes=_steady(engines=("diffrax",))), "plain runs"),
        (
            "steady_counter_window",
            dict(axes=_steady(grids=(SteadyState("bisection"),))),
            "not one of",
        ),
        ("steady_counter_window", dict(reference=CF.table_pulse()), "Limit reference"),
        ("steady_counter_window", dict(judge=Judge()), "convergence rows off"),
        ("steady_counter_window", dict(steady_mask=("A()",)), "must be in steady_mask"),
        ("time_window", dict(reference=Limit(lambda p: {})), "SteadyState grids only"),
        ("time_window", dict(steady_mask=("X()", "Y()")), "SteadyState grids only"),
        ("time_window", dict(judge=Judge(peak="max")), "judge peak"),
        ("cube_switch", dict(spec_rows=({"c": 2.0},)), "not rows it sweeps"),
        ("cube_switch", dict(spec_rows=()), "not rows it sweeps"),
        ("late_switch", dict(t_span=lambda p: (5.0, 1.0)), "forward window"),
        ("diffrax_gaussian", dict(note=""), "names no switch"),
        (
            "steady_counter_window",
            dict(judge=Judge(converge=False, peak="window", why="x")),
            "no window peak",
        ),
        ("time_window", dict(judge=Judge(k_sens=10.0)), "does not say why"),
        ("time_window", dict(judge=Judge(converge=False)), "does not say why"),
        ("time_window", dict(exclude={"nope|rtol=1e-08": "x"}), "which its sweep has not"),
        (
            "clock_step_weak",
            dict(exclude={next(iter(BY_NAME["clock_step_weak"].exclude)): ""}),
            "without saying why",
        ),
    ],
)
def test_a_case_written_wrongly_is_refused(name, change, message):
    with pytest.raises(CaseError, match=message):
        _with(name, **change).validate()


@pytest.mark.parametrize(
    ("raises", "message"), [("not a name", "class name"), (3, "exception type")]
)
def test_a_known_failure_names_an_exception_class(raises, message):
    base = BY_NAME["periodic_window"]
    key = next(iter(base.known))
    with pytest.raises(CaseError, match=message):
        _with("periodic_window", known={key: Known(1805, raises)}).validate()


# ── Failure types by name ───────────────────────────────────────────────────


def test_a_name_resolves_in_bngsim_then_in_the_builtins():
    assert S.resolve("SimulationError") is bngsim.SimulationError
    assert S.resolve("SensitivityUnsupportedError") is bngsim.SensitivityUnsupportedError
    assert S.resolve("AssertionError") is AssertionError
    assert S.resolve(bngsim.SimulationTimeout) is bngsim.SimulationTimeout
    for bad in ("NoSuchError", "Simulator", "__version__"):
        with pytest.raises(ValueError, match="names no exception class"):
            S.resolve(bad)


def test_a_known_failure_named_by_its_class_marks_the_slice_with_that_class():
    params = S.slice_params([BY_NAME["periodic_window"]])
    marked = [p for p in params if p.marks]
    assert marked
    for p in marked:
        (mark,) = p.marks
        assert mark.kwargs["raises"] is bngsim.SensitivityUnsupportedError
        assert mark.kwargs["strict"] is True and mark.kwargs["reason"].startswith("#1805")


# ── Refusals owed ───────────────────────────────────────────────────────────


def _owing(name, refusal="SensitivityUnsupportedError"):
    base = BY_NAME[name].reference
    return ClosedForm(
        base.fn, base.crossings_fn, owes=lambda p, sens: Owed("no derivative", refusal)
    )


def test_a_run_that_returns_where_a_refusal_is_owed_is_a_diff():
    case = _with("time_window", reference=_owing("time_window"))
    row = S.run_point(case, case.slice[0])
    assert row.outcome == "DIFF"
    assert "SensitivityUnsupportedError was expected: no derivative" in row.note


def test_a_run_that_raises_the_refusal_owed_passes_and_another_does_not():
    case = _with("periodic_window", reference=_owing("periodic_window"), known={})
    p = point(1e-10, Uniform(2), ("kd",))
    row = S.run_point(case, p)
    assert row.outcome == "PASS" and "refused as owed (no derivative)" in row.note
    narrow = _with(
        "periodic_window", reference=_owing("periodic_window", "CodegenDeclinedError"), known={}
    )
    assert S.run_point(narrow, p).outcome == "UNSUPPORTED"
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        S.run_point(narrow, p, strict=True)


def test_an_unowed_refusal_is_unsupported_and_strict_raises_its_type():
    case = BY_NAME["periodic_window"]
    p = point(1e-10, Uniform(2), ("c",))
    assert S.run_point(case, p).outcome == "UNSUPPORTED"
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        S.run_point(case, p, strict=True)


def test_the_event_closed_forms_owe_a_refusal_only_at_their_tie():
    tie, kink = CF.event_touch_tie(), CF.table_kink_switch_time()
    assert tie.owed({"k": 0.1, "tau": 2.5}, ("tau",)) is not None
    assert tie.owed({"k": 0.1, "tau": 2.5}, ()) is None
    assert tie.owed({"k": 0.1, "tau": 2.5 + 1e-9}, ("tau",)) is None
    assert kink.owed({"r": 1.0, "tau": 3.0}, ("tau",)) is not None
    assert kink.owed({"r": 1.0, "tau": 3.3}, ("tau",)) is None


# ── Steady states ───────────────────────────────────────────────────────────


def _fake(converged, species):
    def execute(case, model, p, request, atol):
        return S.Ran(
            times=np.array([np.inf]),
            species=np.asarray([species], dtype=float),
            names=list(case.observe),
            sens=None,
            sens_params=[],
            converged=converged,
            work={},
            backend=None,
        )

    return execute


def test_a_steady_state_must_be_right_and_say_it_converged(monkeypatch):
    case = BY_NAME["steady_counter_window"]
    p = point(1e-8, SteadyState("newton"), kc=1.0001)
    limit = case.reference.fn(case.values(p.row))
    right = [limit[s] for s in case.observe]
    monkeypatch.setattr(S, "execute", _fake(True, right))
    assert S.run_point(case, p).outcome == "PASS"
    monkeypatch.setattr(S, "execute", _fake(False, right))
    row = S.run_point(case, p)
    assert row.outcome == "DIFF" and "converged=False" in row.note
    wrong = [right[0] * (1 + 1e-3), right[1], right[2]]
    monkeypatch.setattr(S, "execute", _fake(True, wrong))
    assert S.run_point(case, p).outcome == "DIFF"


def test_a_limit_answers_at_infinity_and_gives_no_columns():
    lim = Limit(lambda p: {"A": 2.0, "B": p["k"]})
    sol = lim.solve({"k": 3.0}, (0.0, 1.0), [np.inf], ("B", "A"), ())
    assert sol.species.lo.tolist() == [[3.0, 2.0]] and sol.times.tolist() == [np.inf]
    assert lim.crossings({}, (0.0, 1.0)) == () and lim.owed({}, ()) is None
    with pytest.raises(ReferenceUnavailable, match="no sensitivity columns"):
        lim.solve({"k": 3.0}, (0.0, 1.0), [np.inf], ("A",), ("k",))
    with pytest.raises(ReferenceUnavailable, match="no value for"):
        lim.solve({"k": 3.0}, (0.0, 1.0), [np.inf], ("C",), ())
    with pytest.raises(ReferenceUnavailable, match="not finite"):
        Limit(lambda p: {"A": np.nan}).solve({}, (0.0, 1.0), [np.inf], ("A",), ())


# ── Engines ─────────────────────────────────────────────────────────────────


def test_a_diffrax_point_without_jax_skips_in_the_slice_and_is_a_harness_row(monkeypatch):
    import bngsim._jax_rhs as jr

    monkeypatch.setattr(jr, "jax_available", lambda: False)
    case = BY_NAME["diffrax_gaussian"]
    p = point(1e-8, Uniform(3), engine="diffrax")
    with pytest.raises(S.EngineUnavailable, match=S.NO_JAX):
        S.run_point(case, p, strict=True)
    row = S.run_point(case, p)
    assert row.outcome == "EXCEPTION" and row.subclass == "harness:EngineUnavailable"


def test_each_ode_engine_runs_as_it_says(monkeypatch):
    seen = []
    real = bngsim.Simulator

    def spy(*a, **k):
        seen.append(k.get("codegen"))
        return real(*a, **k)

    monkeypatch.setattr(S.bngsim, "Simulator", spy)
    case = BY_NAME["cube_switch"]
    for engine in ("ode-interpreted", "ode-compiled"):
        S.run_point(case, point(1e-10, Uniform(2), engine=engine, eps=1e-2, c=1.0612079999999993))
    S.run_point(BY_NAME["time_window"], BY_NAME["time_window"].slice[0])
    assert seen == [False, True, None]


# ── The window peak ─────────────────────────────────────────────────────────

TRIANGLE = {"kx": 1.0, "k_clock": 2.0, "a": 3.0, "b": 1.0, "c": 25.0, "w": 0.003}
COLS = ("kx", "k_clock")


def test_the_window_peak_catches_a_column_that_peaks_between_samples():
    cf = CF.clock_exposure("triangle")
    t = np.linspace(0.0, 30.0, 4)
    sol = cf.solve(TRIANGLE, (0.0, 30.0), t, ("Y()",), COLS)
    sp, se = window_peak(cf, TRIANGLE, (0.0, 30.0), ("Y()",), COLS)
    sampled = np.abs(sol.sens.lo).max(axis=0)
    # The clock-rate column runs about 3e4 times its end value at the apex,
    # which no sample of this grid holds; the apex is a crossing, so the scan
    # holds it however narrow the pulse.
    assert se[0, 1] > 1e4 * sampled[0, 1]
    assert sp[0] == pytest.approx(np.abs(sol.species.lo).max(), rel=1e-12)
    apex = cf.solve(TRIANGLE, (0.0, 30.0), np.array([25.0]), ("Y()",), COLS)
    assert se[0, 1] >= np.abs(apex.sens.lo[0, 0, 1]) * (1 - 1e-12)


@pytest.mark.parametrize(("peak", "outcome"), [("samples", "DIFF"), ("window", "PASS")])
def test_a_case_judged_by_its_window_peak_allows_rtol_of_that_peak(peak, outcome):
    cf = CF.clock_exposure("triangle")
    rtol, t = 1e-9, np.linspace(0.0, 30.0, 4)
    sol = cf.solve(TRIANGLE, (0.0, 30.0), t, ("Y()",), COLS)
    S_ = sol.sens.lo.copy()
    S_[-1, 0, 1] += 1e4 * rtol * abs(S_[-1, 0, 1])
    wp = window_peak(cf, TRIANGLE, (0.0, 30.0), ("Y()",), COLS) if peak == "window" else None
    v = J.judge(
        sol,
        sol.species.lo,
        S_,
        rtol=rtol,
        atol=rtol * 1e-2,
        observe=("Y()",),
        sens=COLS,
        params=TRIANGLE,
        k_state=1e3,
        k_sens=1e3,
        window_peak=wp,
    )
    assert v.outcome == outcome


def test_only_the_cases_that_say_so_are_judged_by_their_window_peak():
    window = {c.name for c in BY_NAME.values() if c.judge.peak == "window"}
    assert window == {"clock_triangle_weak", "clock_sawtooth_weak", "clock_sawtooth_literal"}


def test_a_point_is_a_dataclass_whose_engine_defaults_to_the_ode():
    assert Point((), 1e-8, Uniform(4), ()).engine == "ode"


def test_the_spec_check_runs_only_the_rows_it_is_given(monkeypatch):
    case = BY_NAME["slow_threshold_run"]
    seen = []
    real = S.prepare

    def spy(c, values):
        seen.append(values["r"])
        return real(c, values)

    monkeypatch.setattr(S, "prepare", spy)
    assert S.check_spec(case) == []
    assert set(seen) == {case.spec_rows[0]["r"]}


# ── Points left out, and convergence rows ───────────────────────────────────


def test_an_excluded_point_is_not_swept_reported_or_sliced():
    c = BY_NAME["clock_step_weak"]
    assert c.exclude
    for k in c.exclude:
        assert k not in {c.key(p) for p in c.points()} and k not in c.all_keys()
        assert k not in c.known
    first = next(iter(c.exclude))
    marked = _with("clock_step_weak", known={**c.known, first: Known(1779)})
    with pytest.raises(CaseError, match="both excluded and marked"):
        marked.validate()
    p = next(p for p in c._product() if c.key(p) == first)
    with pytest.raises(CaseError, match="outside its sweep"):
        _with("clock_step_weak", slice=(*c.slice[:3], p)).validate()


def test_a_group_holding_a_known_failure_has_no_convergence_row():
    base = BY_NAME["time_window"]
    p = base.slice[0]
    assert base.converge_key(p.group) in base.all_keys()
    marked = _with("time_window", known={base.key(p): Known(1, AssertionError)})
    assert marked.converge_key(p.group) not in marked.all_keys()
    assert all(
        base.key(q) not in marked.known
        for ps in marked.groups(marked.points()).values()
        for q in ps
    )


def test_every_case_that_changes_its_judge_says_why():
    for c in BY_NAME.values():
        assert c.judge.is_default() or c.judge.why, c.name


# ── What each engine is given ───────────────────────────────────────────────


def test_a_time_course_gets_the_cases_timeout_and_its_tolerances(monkeypatch):
    seen = {}
    real = bngsim.Simulator.run

    def spy(self, *a, **k):
        seen.update(k)
        return real(self, *a, **k)

    monkeypatch.setattr(bngsim.Simulator, "run", spy)
    case = dataclasses.replace(BY_NAME["time_window"], timeout=17.0)
    S.run_point(case, case.slice[0])
    assert seen["timeout"] == 17.0 and seen["rtol"] == case.slice[0].rtol
    assert seen["atol"] == case.slice[0].rtol * case.atol_ratio


def test_a_steady_state_gets_its_tolerances_method_and_mask(monkeypatch):
    seen = {}
    real = bngsim.Simulator.steady_state

    def spy(self, *a, **k):
        seen.update(k)
        return real(self, *a, **k)

    monkeypatch.setattr(bngsim.Simulator, "steady_state", spy)
    case = BY_NAME["steady_counter_window"]
    p = point(1e-8, SteadyState("newton"), kc=1.0001)
    S.run_point(case, p)
    assert seen == {
        "method": "newton",
        "rtol": 1e-8,
        "atol": 1e-10,
        "mask": list(case.steady_mask),
    }


def test_a_diffrax_point_gets_its_tolerances_and_grid(monkeypatch):
    import bngsim._diffrax_solver as D
    import bngsim._jax_rhs as jr

    seen = {}

    def fake(model, **k):
        seen.update(k)
        t = np.linspace(k["t_start"], k["t_end"], k["n_points"])
        return {
            "time": t,
            "species": np.zeros((len(t), 1)),
            "species_names": ["X()"],
            "n_steps": 1,
        }

    monkeypatch.setattr(jr, "jax_available", lambda: True)
    monkeypatch.setattr(D, "run_diffrax", fake)
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda name: object())
    case = BY_NAME["diffrax_gaussian"]
    S.run_point(case, point(1e-8, Uniform(3), engine="diffrax"))
    assert (seen["rtol"], seen["atol"], seen["n_points"]) == (1e-8, 1e-8, 3)
    assert (seen["t_start"], seen["t_end"]) == case.window()


# ── The spec check ──────────────────────────────────────────────────────────


def test_the_spec_check_reports_a_plain_run_that_fails(monkeypatch):
    def fail(self, *a, **k):
        raise bngsim.SimulationError("no progress")

    monkeypatch.setattr(bngsim.Simulator, "run", fail)
    problems = S.check_spec(BY_NAME["time_window"])
    assert problems and all("the plain run the check reads failed" in p for p in problems[:-2])


def test_the_spec_check_sees_inside_a_pulse_narrower_than_its_offsets():
    """The literal sawtooth's pulse is 1e-3 wide in time, and the check's
    offsets sit 2.5e-3 from each crossing: only the midpoint between them
    lies inside it. A spec with half the pulse's rate is caught there."""
    case = BY_NAME["clock_sawtooth_literal"]
    spec = case.spec
    half = dataclasses.replace(
        spec,
        rhs=lambda x, s, t, p: {
            k: (v / 2 if k != "T()" else v) for k, v in spec.rhs(x, s, t, p).items()
        },
    )
    assert S.check_spec(case) == []
    assert S.check_spec(dataclasses.replace(case, spec=half))


# ── Every name resolves ─────────────────────────────────────────────────────


def test_every_marks_class_and_every_owed_refusal_resolve_in_bngsim():
    for c in BY_NAME.values():
        for k in c.known.values():
            S.resolve(k.raises)
        owed = getattr(c.reference, "owed", None)
        if owed is None:
            continue
        for p in c.points():
            o = owed(c.values(p.row), p.sens)
            if o is not None:
                assert issubclass(S.resolve(o.refusal), bngsim.BngsimError), c.name


# ── Instants a double apart ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("at_03", "outcome"),
    [((1.0, 0.0, 0.0), "PASS"), ((0.0, 0.0, 0.0), "PASS"), ((1.0, 1.0, 2.0), "DIFF")],
)
def test_a_correct_engine_passes_on_a_row_at_one_of_two_instants_a_double_apart(at_03, outcome):
    """#1627's u101 grid has a row on 0.3 exactly: G and F have fired there,
    H and E (on 0.30000000000000004) not yet. Read with the usual 4-ulp blur,
    the reference took H and E to have fired too, so a correct engine failed."""
    case = BY_NAME["event_double_apart"]
    values = case.values()
    t = np.linspace(0.0, 1.0, 101)
    assert t[30] == 0.3
    sol = case.reference.solve(values, case.window(), t, case.observe, ())
    x = sol.species.lo.copy()
    x[30] = at_03
    v = J.judge(
        sol,
        x,
        None,
        rtol=1e-8,
        atol=1e-10,
        observe=case.observe,
        sens=(),
        params=values,
        k_state=1e3,
        k_sens=1e3,
    )
    assert v.outcome == outcome


def test_a_group_whose_points_owe_a_refusal_has_no_convergence_row():
    """Once a bug like #1618 is fixed and its marks come out, its owed points
    pass as refused, with no ratio. A row over them could only SKIP, and the
    slice would fail it with nothing wrong in the engine."""
    base = BY_NAME["periodic_window"].reference
    columns_owe = ClosedForm(
        base.fn, base.crossings_fn, owes=lambda p, sens: Owed("no derivative") if sens else None
    )
    case = _with("periodic_window", reference=columns_owe, known={}, judge=Judge())
    pts = case.points()
    owed_groups = {p.group for p in pts if p.sens}
    assert owed_groups and not owed_groups & set(case.groups(pts))
    plain = {p.group for p in pts if not p.sens}
    assert plain and plain <= set(case.groups(pts))
