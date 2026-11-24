"""Issue #1838: an ODE run over a span CVODE cannot step through.

``A -> B`` at k = 1 over ``(0, 7e307)`` never returned, and ``timeout=5`` did
not stop it. ``(1e308, 1.7e308)`` is the same run once #1753 moves it onto its
own clock; for a model that reads the clock it is not moved, and came back as
``[1, 0, nan, nan, nan]`` with no error, beside a ``RoundoffStepWarning`` that
blamed steps too small to move the clock.

CVODE is given no largest step, and grows its step up to tenfold a step (ten
thousandfold after its first). Near the largest double the step overflows to
inf before the run reaches its end, and the clock with it. With h = inf, the
test that asks the Newton iteration for a fresh Jacobian reads inf / inf, a
NaN, and never asks, so the iteration retries for ever inside one step: 16.7
million calls of the right-hand side in 8 s, the step count fixed at 171.
``max_steps`` counts steps and could not see it, and the wall-clock budget was
read only between CVode calls. Where the clock passed the end at inf instead,
CVODE interpolated the rows there, and they were NaN.

Where it happens turns on the model as much as on the span: ``A -> B`` returns
over ``(-7e307, 0)``, and at k = 0, whose steps CVODE grows the fastest, it
never returned there, nor over ``(0, 1e306)``. So no ceiling on the span tells
the two apart. The right-hand side ends the run, as ``UsageError``, at the
first call after CVODE's own clock is no longer finite; a run whose length is
itself past the largest double is refused before it starts. Every run that
returned before returns the rows it did. And the wall-clock budget is read from
the right-hand side, so a run that spends it inside one CVode call stops there
as ``SimulationTimeout``.
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


# Each hung or returned NaN rows before, with no error.
OVERFLOWS = [
    pytest.param("k", 1.0, (0.0, 7e307), id="the issue's run that never returned"),
    pytest.param(READS_THE_CLOCK, 1.0, (0.0, 7e307), id="it, reading the clock"),
    pytest.param("k", 1.0, (1e308, 1.7e308), id="moved onto (0, 7e307)"),
    pytest.param(READS_THE_CLOCK, 1.0, (1e308, 1.7e308), id="the issue's nan rows"),
    pytest.param("k", 1.0, (-1.7e308, -1e308), id="a far run below 0"),
    pytest.param("k", 0.0, (0.0, 1e306), id="a model at rest"),
    pytest.param("k", 0.0, (-7e307, 0.0), id="a model at rest, toward 0"),
]


@pytest.mark.parametrize(("law", "k", "span"), OVERFLOWS)
def test_a_run_whose_step_overflows_ends_there(tmp_path, capfd, law, k, span):
    sim = _sim(tmp_path, _net(law, k))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # timeout=30 only so that an overflow that goes unseen fails here as
        # a timeout rather than hanging the suite.
        with pytest.raises(bngsim.UsageError, match="largest double") as info:
            sim.run(span, 5, timeout=30)
    assert isinstance(info.value, ValueError)
    assert info.value.code == "usage.invalid_argument"
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


@pytest.mark.parametrize(
    "times",
    [
        pytest.param({"t_span": (-1e308, 1e308), "n_points": 5}, id="t_span"),
        pytest.param({"sample_times": [-1e308, 0.0, 1e308]}, id="sample_times"),
    ],
)
def test_a_run_whose_length_overflows_is_refused_before_it_runs(tmp_path, times):
    # 2e308 is past the largest double: the grid's first time was
    # -1e308 + 0 * inf, a NaN, and every model tried failed "at t=inf".
    sim = _sim(tmp_path, _net())
    with pytest.raises(bngsim.UsageError, match="too long") as info:
        sim.run(**times, timeout=30)
    assert "larger unit" in str(info.value)
    assert sim.current_time == 0.0
    _decays_from_1(sim.run((0.0, 5.0), 6))


def test_run_batch_ends_it_too(tmp_path):
    sim = _sim(tmp_path, _net())
    with pytest.raises(bngsim.UsageError, match="largest double"):
        sim.run_batch(t_span=(0.0, 7e307), n_points=5, params=[{"k": 1.0}, {"k": 2.0}], timeout=30)


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


# Each returned these rows before, and returns them still.
def test_far_runs_that_returned_still_return(tmp_path):
    exp_rows = [1.0, 0.0, 0.0, 0.0, 0.0]
    # A' = 1: A = t - t_start, exactly.
    r = _sim(tmp_path, CLOCK_NET).run((0.0, 1e300), 5)
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], np.asarray(r.time), rtol=1e-12)
    # A -> B, its first step far longer than its time scale: A is at 0 by the
    # first row.
    for span in [(0.0, 1e200), (1e200, 2e200)]:
        r = _sim(tmp_path, _net()).run(span, 5)
        np.testing.assert_allclose(np.asarray(r.species)[:, 0], exp_rows, atol=1e-12)
    # Toward 0 its steps stay finite, though they are too small to move the
    # clock out there, and it says so.
    with pytest.warns(bngsim.RoundoffStepWarning):
        r = _sim(tmp_path, _net()).run((-7e307, 0.0), 5)
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], exp_rows, atol=1e-12)
    # Far from 0 and short, A' = 1 is moved onto its own clock, as from 0.
    far = _sim(tmp_path, CLOCK_NET).run((2.0**520, 2.0**520 + 2.0**470), 5)
    near = _sim(tmp_path, CLOCK_NET).run((0.0, 2.0**470), 5)
    assert np.array_equal(np.asarray(far.species), np.asarray(near.species))
    np.testing.assert_allclose(np.asarray(far.species)[:, 0], np.asarray(near.time), rtol=1e-12)


def test_a_first_step_cvode_only_tries_at_inf_is_let_be(tmp_path):
    # At rest, and with output rows 2.5e299 apart, CVODE's first-step routine
    # multiplies two times past the largest double and tries t_0 + inf. It goes
    # on from the step it settles on, its own clock finite throughout.
    r = _sim(tmp_path, _net(k=0.0)).run((0.0, 1e300), 5)
    assert np.array_equal(np.asarray(r.species)[:, 0], np.ones(5))


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
