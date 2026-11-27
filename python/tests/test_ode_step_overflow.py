"""Issue #1838: an ODE run over a span CVODE cannot step through.

``A -> B`` at k = 1 over ``(0, 7e307)`` never returned, and ``timeout=5`` did
not stop it. ``(1e308, 1.7e308)`` is the same run once #1753 moves it onto its
own clock; for a model that reads the clock it is not moved, and came back as
``[1, 0, nan, nan, nan]`` with no error, beside a ``RoundoffStepWarning`` that
blamed steps too small to move the clock.

CVODE is given no largest step, and grows its step up to tenfold a step (ten
thousandfold after its first). Near the largest double the step overflows to
inf before the run reaches its end. With h = inf, the test that asks the Newton
iteration for a fresh Jacobian reads inf / inf, a NaN, and never asks, so the
iteration retries for ever inside one step: 16.7 million calls of the
right-hand side in 8 s, the step count fixed at 171. ``max_steps`` counts steps
and could not see it, and the wall-clock budget was read only between CVode
calls. Where the step stayed finite and t + h passed the end at inf, CVODE kept
the step and interpolated the rows from it, and they were NaN.

Where it happens turns on the model as much as on the span: ``A' = 1`` returns
over ``(0, 1e306)``, and ``A -> B`` at k = 0, whose steps CVODE grows the
fastest, never returned there, nor over ``(-7e307, 0)``. So no ceiling on the
span tells the two apart. A run is ended, as ``UsageError``, where CVODE's step
is no longer finite, or as a CVode call returns with its clock at inf; a step
attempt whose t + h is inf that CVODE throws away and retries shorter is let
be, as is a run whose first-step routine tries a time of inf. A grid of output
times that cannot be finite is refused before it starts. Every run whose
clock stayed finite returns the rows it did. And the wall-clock budget is
read from the right-hand side, so a run that spends it inside one CVode call
stops there as ``SimulationTimeout``.

A run that starts too far from t = 0 for its model is refused before it
steps: that is where the clock-reading run over (1e308, 1.7e308) now ends, on
either RHS, where a compiled RHS returned A = B = 0 (it reads 0*time() at inf
as NaN, which the interpreted one folds to 0, and so was refused only at a kept
step to inf). Issue #1943 (a first output far past the model's time scale)
is not this file's.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest


def _net(law="k", k=1.0):
    # A -> B at the rate f()*A, from A = 1.
    return (
        f"begin parameters\n 1 k {k!r}\nend parameters\n"
        f"begin functions\n 1 f() {law}\nend functions\n"
        "begin species\n 1 A() 1\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 f\nend reactions\n"
        "begin groups\n 1 Atot 1\n 2 Btot 2\nend groups\n"
    )


# $S -> $S + A at k = 1: A' = 1, so A = t - t_start, which BDF integrates
# exactly whatever its step.
CLOCK_NET = (
    "begin parameters\n 1 k 1\nend parameters\n"
    "begin species\n 1 A() 0\n 2 $S() 1\nend species\n"
    "begin reactions\n 1 2 1,2 k\nend reactions\n"
    "begin groups\n 1 Atot 1\nend groups\n"
)

READS_THE_CLOCK = "k*(1+0*time())"  # not moved onto the run's own clock


def _sim(tmp_path, text, **kw):
    net = tmp_path / "m.net"
    net.write_text(text)
    return bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", **kw)


def _roundoff_warnings(caught):
    return [w for w in caught if issubclass(w.category, bngsim.RoundoffStepWarning)]


def _decays_from_1(r, k=1.0):
    # A(t) = e^-kt from A = 1, after a run inside reach on the same Simulator.
    np.testing.assert_allclose(
        np.asarray(r.species)[:, 0], np.exp(-k * np.arange(6.0)), rtol=1e-5, atol=1e-7
    )


def _span(a, b):
    return {"t_span": (a, b), "n_points": 5}


# Each hung or returned NaN rows before, with no error.
OVERFLOWS = [
    pytest.param("k", 1.0, _span(0.0, 7e307), id="the issue's run that never returned"),
    pytest.param(READS_THE_CLOCK, 1.0, _span(0.0, 7e307), id="it, reading the clock"),
    pytest.param("k", 1.0, _span(1e308, 1.7e308), id="moved onto (0, 7e307)"),
    pytest.param("k", 1.0, _span(-1.7e308, -1e308), id="a far run below 0"),
    pytest.param("k", 0.0, _span(0.0, 1e306), id="a model at rest"),
    pytest.param("k", 0.0, _span(-7e307, 0.0), id="a model at rest, toward 0"),
]


@pytest.mark.parametrize(("law", "k", "times"), OVERFLOWS)
def test_a_run_whose_step_overflows_ends_there(tmp_path, capfd, law, k, times):
    sim = _sim(tmp_path, _net(law, k))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # timeout=30 only so that an overflow that goes unseen fails here as
        # a timeout rather than hanging the suite.
        with pytest.raises(bngsim.UsageError, match="largest double") as info:
            sim.run(**times, timeout=30)
    assert isinstance(info.value, ValueError)
    assert info.value.code == "usage.span_out_of_reach"
    assert "larger unit" in str(info.value)
    # (1e308, 1.7e308) of the model that reads the clock warned of 155 steps
    # where t + h == t beside its nan rows: the far span's doing, which the
    # error says.
    assert _roundoff_warnings(caught) == []
    assert all(n["code"] != "solver.roundoff_steps" for n in info.value.notices)
    # SUNDIALS' own line for the call ("the right-hand side routine failed in
    # an unrecoverable manner") is not written beside it. Captured at the file
    # descriptor: SUNDIALS writes to the C stderr.
    assert "[ERROR]" not in capfd.readouterr().err
    # And it moved nothing: the same Simulator runs a span inside reach.
    assert sim.current_time == 0.0
    _decays_from_1(sim.run((0.0, 5.0), 6), k)


@pytest.mark.parametrize("n_points", [3, 5])
def test_a_grid_whose_times_cannot_be_finite_is_refused_before_it_runs(tmp_path, n_points):
    # 2e308 is past the largest double, so the grid's spacing is inf: its
    # first time was -1e308 + 0 * inf, a NaN, and every model tried failed
    # "at t=inf", with max_step or without.
    sim = _sim(tmp_path, _net())
    with pytest.raises(bngsim.UsageError, match="too long") as info:
        sim.run((-1e308, 1e308), n_points, max_step=1e306, timeout=30)
    assert info.value.code == "usage.span_out_of_reach"
    assert "output times" in str(info.value)
    assert "larger unit" in str(info.value)
    assert sim.current_time == 0.0
    _decays_from_1(sim.run((0.0, 5.0), 6))


# A step CVODE keeps at t + h = inf, NaN rows before, is refused as each kind of
# CVode call returns.
KEPT_AT_INF = [
    # Steps of 1e307 from 0: the one past 1.79e308 is kept, in the warm loop's
    # call, and with max_steps=5 in a batch retry_while_advancing runs.
    pytest.param("clock", {}, {"max_step": 1e307}, id="the warm loop"),
    pytest.param("clock", {}, {"max_step": 1e307, "max_steps": 5}, id="a retried batch"),
    # An event takes the run down the cold loop, and so do sensitivities: from
    # 0, as a run that starts far from 0 is refused before it steps.
    pytest.param("event", {}, {"max_step": 1e307}, id="the cold loop"),
    pytest.param(
        "event",
        {"sensitivity_params": ["k"]},
        {"max_step": 1e307},
        id="the cold loop, sensitivities",
    ),
]
EVENT = "species A = 1, B = 0; k = 1; J1: A -> B; k*A; E1: at time > 1e10: A = 1;"


@pytest.mark.parametrize(("model", "sim_kw", "run_kw"), KEPT_AT_INF)
def test_a_step_kept_at_inf_is_refused_as_its_call_returns(tmp_path, model, sim_kw, run_kw):
    span, n_points = (0.0, 1.79e308), 2
    if model == "clock":
        sim = _sim(tmp_path, CLOCK_NET, **sim_kw)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), "ode", **sim_kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(bngsim.UsageError, match="largest double") as info:
            sim.run(span, n_points, timeout=30, **run_kw)
    assert info.value.code == "usage.span_out_of_reach"


@pytest.mark.parametrize("num_processors", [1, 2], ids=["sequential", "threads"])
def test_a_batch_row_or_scan_point_that_overflows_is_named(tmp_path, num_processors):
    # Over (0, 3e306) A -> B at k = 1 returns, and at k = 1e-300 its step
    # passes the largest double: whether a row gets there turns on its values,
    # so the error names the row, as a refusal partway does.
    sim = _sim(tmp_path, _net())
    with pytest.raises(bngsim.UsageError, match=r"^run_batch row 1: .*largest double") as info:
        sim.run_batch(
            t_span=(0.0, 3e306),
            n_points=5,
            params=[{"k": 1.0}, {"k": 1e-300}],
            num_processors=num_processors,
            timeout=30,
        )
    assert info.value.code == "usage.span_out_of_reach"
    with pytest.raises(bngsim.UsageError, match=r"^scan point 1 \(k = 1e-300\): ") as info:
        sim.parameter_scan("k", par_scan_vals=[1.0, 1e-300], t_span=(0.0, 3e306), n_points=5)
    assert info.value.code == "usage.span_out_of_reach"


def test_compute_all_sensitivities_ends_it_as_it_is():
    # It wrapped the refusal as "Sensitivity chunk failed", a SimulationError,
    # retried each column alone and advised loosening rtol and atol.
    text = "species A, B; A = 1; B = 0; k = 1; k2 = 1\nJ1: A -> B; k*A\nJ2: B -> ; k2*B\n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode")
    for span in [(0.0, 7e307), (-1e308, 1e308)]:
        with pytest.raises(bngsim.UsageError) as info:
            sim.compute_all_sensitivities(
                t_span=span, n_points=5, params=["k", "k2"], chunk_size=2
            )
        assert "larger unit" in str(info.value)
        assert "rtol" not in str(info.value)


# Each returned these rows before, right in every column, and returns them
# still.
def test_far_runs_that_returned_still_return(tmp_path):
    # A' = 1: A = t - t_start, exactly.
    r = _sim(tmp_path, CLOCK_NET).run((0.0, 1e300), 5)
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], np.asarray(r.time), rtol=1e-12)
    # Toward 0 its steps stay finite, though they are too small to move the
    # clock out there, and it says so.
    with pytest.warns(bngsim.RoundoffStepWarning):
        r = _sim(tmp_path, CLOCK_NET).run((-7e307, 0.0), 5)
    np.testing.assert_allclose(
        np.asarray(r.species)[:, 0], np.asarray(r.time) + 7e307, rtol=1e-12, atol=1e295
    )
    # A' = 1/(1 + t/1e308): CVODE tries two steps whose t + h is inf, throws
    # them away on its error test and goes on from its own finite clock. The
    # rows are 1e308 ln(1 + t/1e308), to 3e-8.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = bngsim.Model.from_antimony_string(
            "species A = 0; k = 1; J1: -> A; k/(1+time/1e308);"
        )
    for times in [
        {"t_span": (0.0, 1.79e308), "n_points": 5},
        {"sample_times": [0.0, 1.0, 1.79e308]},
    ]:
        r = bngsim.Simulator(log, "ode").run(**times, start="initial")
        t = np.asarray(r.time)
        np.testing.assert_allclose(
            np.asarray(r.species)[:, 0], 1e308 * np.log1p(t / 1e308), rtol=1e-6, atol=1e-6
        )
    # Far from 0 and short, A' = 1 is moved onto its own clock, as from 0.
    far = _sim(tmp_path, CLOCK_NET).run((2.0**520, 2.0**520 + 2.0**470), 5)
    near = _sim(tmp_path, CLOCK_NET).run((0.0, 2.0**470), 5)
    assert np.array_equal(np.asarray(far.species), np.asarray(near.species))
    np.testing.assert_allclose(np.asarray(far.species)[:, 0], np.asarray(near.time), rtol=1e-12)


@pytest.mark.parametrize("max_step", [1e305, 1e306, 1e307])
def test_a_run_longer_than_the_largest_double_whose_steps_stay_finite_returns(tmp_path, max_step):
    # Sample times 2e308 apart, max_step bounding every step: CVODE never takes
    # a step, or reaches a time, past the largest double. (Its first steps at
    # -1e308 are too small to move the clock, and it says so; the rows are
    # right.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin = bngsim.Model.from_antimony_string("species A = 0; J1: -> A; time/1e308;")
        rest = _sim(tmp_path, _net(k=0.0)).run(
            sample_times=[-1e308, 0.0, 1e308], max_step=max_step, timeout=30
        )
        r = bngsim.Simulator(lin, "ode").run(
            sample_times=[-1e308, 0.0, 1e308], max_step=max_step, timeout=30
        )
    assert np.array_equal(np.asarray(rest.species), [[1.0, 0.0]] * 3)
    # A' = t/1e308: A = (t^2 - 1e616)/2e308.
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], [0.0, -5e307, 0.0], atol=1e300)


def test_a_first_step_cvode_only_tries_at_inf_is_let_be(tmp_path):
    # At rest, and with output rows 2.5e299 apart, CVODE's first-step routine
    # multiplies two times past the largest double and tries t_0 + inf. It goes
    # on from the step it settles on, its own clock finite throughout.
    r = _sim(tmp_path, _net(k=0.0)).run((0.0, 1e300), 5)
    assert np.array_equal(np.asarray(r.species), [[1.0, 0.0]] * 5)


# One CVode call: two rows, a step of at most 1e-4, and max_steps out of the
# way, so the run is ten million steps that no budget read between calls sees.
ONE_CALL = {"n_points": 2, "max_step": 1e-4, "max_steps": 2**31 - 1}
DECAY = "species A, B; A = 1; B = 0; k = 1\nJ1: A -> B; k*A\n"


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"codegen": False}, id="interpreted"),
        pytest.param({"codegen": True}, id="compiled"),
        # Forward sensitivities take the run down the cold path.
        pytest.param({"sensitivity_params": ["k"]}, id="with sensitivities"),
    ],
)
def test_a_timeout_stops_a_run_inside_one_cvode_call(capfd, kw):
    # On main each of these ran to its end, 10 to 12 s, past a budget of
    # 0.5 s, and returned with no error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(DECAY), "ode", **kw)
    with pytest.raises(bngsim.SimulationTimeout) as info:
        sim.run((0.0, 1e3), timeout=0.5, **ONE_CALL)
    assert info.value.timeout == 0.5
    assert info.value.elapsed > 0.5
    # SUNDIALS' own account of the call the budget ended, "the right-hand
    # side routine failed in an unrecoverable manner", is not written to
    # standard error beside it.
    assert "[ERROR]" not in capfd.readouterr().err
    # The next run on the same Simulator is the run it would have been.
    _decays_from_1(sim.run((0.0, 5.0), 6, start="initial"))


@pytest.mark.filterwarnings("ignore")
def test_a_failure_after_a_timeout_is_still_logged(capfd):
    # S' = S^2 from 1 has no solution past t = 1: CVODE's account of that
    # failure still reaches standard error, from a run on the thread whose
    # last run spent its budget inside a call.
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(DECAY), "ode")
    with pytest.raises(bngsim.SimulationTimeout):
        sim.run((0.0, 1e3), timeout=0.2, **ONE_CALL)
    assert "[ERROR]" not in capfd.readouterr().err
    blowup = bngsim.Model.from_antimony_string("species S = 1; J: -> S; S^2;")
    with pytest.raises(bngsim.SimulationError):
        bngsim.Simulator(blowup, "ode").run((0.0, 2.0), 3)
    assert "[ERROR]" in capfd.readouterr().err


FAR_STARTS = [
    # A clock-reading law is not moved onto its own clock (#1753): over
    # (1e308, 1.7e308) it came back [1, 0, nan, nan, nan], and with a compiled
    # RHS, which reads 0*time() at inf as NaN where the interpreted one folds
    # it, A = B = 0 where B is 1, beside a RoundoffStepWarning alone.
    pytest.param(READS_THE_CLOCK, 1.0, _span(1e308, 1.7e308), id="the issue's nan rows"),
    pytest.param(READS_THE_CLOCK, 1.0, _span(2e26, 3e26), id="where CVODE failed"),
    pytest.param(READS_THE_CLOCK, 1.0, _span(1e30, 1.5e30), id="past 1e26 time scales"),
    pytest.param(READS_THE_CLOCK, 1e3, _span(1e27, 1.5e27), id="a faster model, sooner"),
    pytest.param("k", 1.0, {"sample_times": [-1e308, 0.0, 1e308]}, id="sample times across 2e308"),
]


@pytest.mark.parametrize(("law", "k", "times"), FAR_STARTS)
def test_a_run_too_far_from_zero_for_its_model_is_refused(tmp_path, law, k, times):
    # The doubles there are so far apart that CVODE's smallest step outruns the
    # model: a clock-reading A -> B at k = 1 is right from t0 = 1e26 down,
    # fails in CVODE from 1.05e26, and from about 1e55 loses B. Refused before
    # it steps, on either RHS, and the same Simulator runs a span inside reach.
    sim = _sim(tmp_path, _net(law, k))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(bngsim.UsageError, match="too far from t = 0") as info:
            sim.run(**times, timeout=30)
    assert info.value.code == "usage.span_out_of_reach"
    assert "larger unit" in str(info.value)
    assert _roundoff_warnings(caught) == []
    assert sim.current_time == 0.0
    _decays_from_1(sim.run((0.0, 5.0), 6), k)


@pytest.mark.parametrize("t0", [1e16, 1e20, 1e25])
def test_a_far_start_inside_reach_still_returns(tmp_path, t0):
    # The doubles near 1e16 are 2 apart, past the model's time scale of 1, and
    # the run is right: a refusal there would turn good runs away.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _sim(tmp_path, _net(READS_THE_CLOCK)).run(t_span=(t0, 1.5 * t0), n_points=5)
    s = np.asarray(r.species)
    np.testing.assert_allclose(s[1:, 1], 1.0, rtol=1e-6)
    np.testing.assert_allclose(s.sum(axis=1), 1.0, rtol=1e-6)


def test_a_large_species_at_rest_does_not_stretch_the_time_scale():
    # The time scale is how fast the rates change along the path: C = 1e6,
    # which nothing changes, beside A -> B at k = 1, leaves it at 1, so a run
    # from 1e30 is refused as A -> B alone is. Its largest species over its
    # largest rate would read 1e6, and let the run go on to fail.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = bngsim.Model.from_antimony_string(
            "species A = 1, B = 0, C = 1e6; k = 1; J1: A -> B; k*(1 + 0*time)*A;"
        )
    with pytest.raises(bngsim.UsageError, match="too far from t = 0") as info:
        bngsim.Simulator(m, "ode").run(t_span=(1e30, 1.5e30), n_points=5, start="initial")
    assert info.value.code == "usage.span_out_of_reach"


@pytest.mark.parametrize("t0", [1e28, 1e30, 1e60])
def test_a_species_far_under_its_tolerance_does_not_set_the_time_scale(tmp_path, t0):
    # A = 1e-20 beside B = 1 moves 1e-20 on its time scale of 1, far under its
    # error weight, and the run is right: [0, 1] at every row after the first.
    net = _net(READS_THE_CLOCK).replace(" 1 A() 1\n 2 B() 0\n", " 1 A() 1e-20\n 2 B() 1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _sim(tmp_path, net).run(t_span=(t0, 1.5 * t0), n_points=5)
    np.testing.assert_allclose(np.asarray(r.species)[1:, 1], 1.0, rtol=1e-12)


def test_a_slow_species_sets_its_own_time_scale_beside_a_residue():
    # C -> D at 1e-25 from C = 1 has a time scale of 1e25, so a run from 1e30
    # is inside reach; a residue A = 1e-18 moving at 1 beside it does not set
    # the scale. C = e^-0.5 and e^-2 at 0.5e25 and 2e25 past the start.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = bngsim.Model.from_antimony_string(
            "species A = 1e-18, B = 0, C = 1, D = 0; k = 1; ks = 1e-25;"
            " J1: A -> B; k*(1 + 0*time)*A; J2: C -> D; ks*(1 + 0*time)*C;"
        )
        r = bngsim.Simulator(m, "ode").run(
            sample_times=[1e30, 1e30 + 0.5e25, 1e30 + 2e25], start="initial"
        )
    c = np.asarray(r.species)[:, r.species_names.index("C")]
    np.testing.assert_allclose(c[1:], np.exp([-0.5, -2.0]), rtol=1e-5)
