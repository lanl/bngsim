"""Output times at, or within CVODE's roundoff of, the start of a plain ODE run
(issue #1501).

Every ODE run with no roots, crossing stops, state-switch conditions or
sensitivities takes ``Impl::run_warm``, whose output loop asked CVode for each
output time from the second on. On the first call after a (re)init, CVODE
refuses a ``tout`` equal to ``t0`` or within ``2*eps*max(|t0|, |tout|)`` of it
(``CV_TOO_CLOSE``), and one so close the step underflows (``CV_ILL_INPUT``). So
replicate samples at the start, ``sample_times=[0, 0, 0, 5, 5, 5, 10, 10, 10]``,
were refused on the default path alone. The event-capable loop already took
such an output as reached (its ``reached_output``, from #737 and #950). The
GH #176 retry then caught the refusal as a Jacobian failure, logged a warning
blaming the rate law, and failed the same way.

``run_warm`` now takes an output within CVODE's roundoff of the start as
reached, before its first step, and records the start state there at the time
asked for, as the event-capable loop does. Once a step is taken, CVODE
interpolates a repeated time, and those rows are what they were.

Oracles: the closed form exp(-k (t - t0)), the event-capable loop
(``BNGSIM_NO_WARM_CVODE=1``) on the same grid, and the warm path's own rows on
the grid without the repeats.
"""

from __future__ import annotations

import contextlib
import io
import logging
import shutil

import bngsim
import numpy as np
import pytest

DECAY = "species x; x = 1; k = 0.1; J1: x -> ; k*x;\n"
RULES = DECAY + "species y; y := 2*x; tt := time*3;\n"
TRIPLICATE = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0]

# Against exp(-k (t - t0)): the integration's own error. At the default
# rtol = atol = 1e-8 the rows come back within 6.0e-08 of it on every grid here,
# and 1e-6 leaves room for other platforms and Jacobian routes. A row read at
# another of a grid's distinct times (1 or more apart here) is off by 0.09 or
# more.
CLOSED_FORM_ATOL = 1e-6
# The two loops take different steps, each within that error of the closed
# form, so they agree to well inside 1e-6 of each other.
LOOPS_RTOL = 1e-6

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH for the codegen leg")


def _run(times, text=DECAY, *, cold=False, monkeypatch=None, **kw):
    if cold:
        monkeypatch.setenv("BNGSIM_NO_WARM_CVODE", "1")
    elif monkeypatch is not None:
        monkeypatch.delenv("BNGSIM_NO_WARM_CVODE", raising=False)
    with contextlib.redirect_stderr(io.StringIO()):
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode", **kw)
        return sim.run(sample_times=times)


def _closed_form(times, k=0.1, t0=None):
    t = np.asarray(times, dtype=float)
    return np.exp(-k * (t - (t[0] if t0 is None else t0)))


def _held_start_bound(k, t0):
    """How far a row of a 1e-9-long grid at |t0| = 1e6 may be from the closed
    form. A row taken as reached holds the start state over less than CVODE's
    roundoff distance, 2·eps·|t0| (4.4e-10 here), so it is off by less than k
    times that; the rows CVode integrates to are nearer (within 3.1e-12). The
    largest error measured is a reached row three doubles from the start,
    3.5e-11 to 1.05e-10 for k = 0.1 to 0.3: 0.63 of this bound, a ratio set by
    the grid's rounding alone. A quarter over, for roundoff in the closed form.
    A run that held the start state on every row is off by k·1e-9 at the last,
    1.9 times this bound, so the test can tell the two apart; a fixed 1e-9
    could not, since x moves by only k·1e-9 over the whole grid."""
    return 1.25 * k * 2.0 * np.finfo(float).eps * abs(t0)


@pytest.fixture(autouse=True)
def _warm(monkeypatch):
    # Every case below is about the default path; a stray env var would move
    # it onto the event-capable loop.
    monkeypatch.delenv("BNGSIM_NO_WARM_CVODE", raising=False)


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({}, id="auto"),
        pytest.param({"jacobian": "fd"}, id="fd"),
        pytest.param({"jacobian": "analytical"}, id="analytical"),
        pytest.param({"codegen": True}, id="codegen", marks=needs_cc),
    ],
)
def test_replicates_at_the_start_are_the_closed_form(kw):
    """The issue's grid: CV_TOO_CLOSE on main, on every route of the warm path."""
    x = np.asarray(_run(TRIPLICATE, **kw).species)[:, 0]
    np.testing.assert_allclose(x, _closed_form(TRIPLICATE), rtol=0, atol=CLOSED_FORM_ATOL)
    assert np.all(x[:3] == x[0])


def test_the_rows_after_the_start_are_the_plain_grid_s_bit_for_bit():
    """Taking the repeats at the start as reached asks CVode nothing there: the
    run takes the steps it takes on [0, 5, 10], and its rows are those."""
    a, b = _run(TRIPLICATE), _run([0.0, 5.0, 10.0])
    xa, xb = np.asarray(a.species)[:, 0], np.asarray(b.species)[:, 0]
    assert np.array_equal(xa[[0, 3, 6]], xb)
    assert np.all(xa[3:6] == xa[3]) and np.all(xa[6:] == xa[6])
    assert a.solver_stats["n_steps"] == b.solver_stats["n_steps"]
    assert np.asarray(a.time).tolist() == TRIPLICATE


def test_repeats_after_the_first_step_are_read_as_before():
    """Control: a time repeated after the first step was read already, CVODE
    interpolating it, and its rows are left as they were."""
    times = [0.0, 5.0, 5.0, 5.0, 10.0, 10.0]
    x = np.asarray(_run(times).species)[:, 0]
    np.testing.assert_allclose(x, _closed_form(times), rtol=0, atol=CLOSED_FORM_ATOL)
    assert np.all(x[1:4] == x[1]) and x[4] == x[5]


def test_the_same_rows_as_the_event_capable_loop(monkeypatch):
    """The warm path and the event-capable loop agree on the grid, to the
    integration tolerance (they take different steps)."""
    warm = np.asarray(_run(TRIPLICATE, monkeypatch=monkeypatch).species)
    cold = np.asarray(_run(TRIPLICATE, cold=True, monkeypatch=monkeypatch).species)
    np.testing.assert_allclose(warm, cold, rtol=LOOPS_RTOL, atol=1e-9)


@pytest.mark.parametrize(
    "times",
    [
        [5.0, float(np.nextafter(5.0, 6.0)), 10.0],
        [0.0, 5e-324, 1.0],
        [0.0, 0.0, 1e-300, 1.0],
        [
            1e6,
            float(np.nextafter(1e6, 2e6)),
            float(np.nextafter(np.nextafter(1e6, 2e6), 2e6)),
            1e6 + 1.0,
        ],
    ],
    ids=[
        "a double after 5 (CV_TOO_CLOSE)",
        "5e-324 (CV_ILL_INPUT)",
        "1e-300",
        "two doubles after 1e6",
    ],
)
def test_a_time_within_roundoff_of_the_start(times):
    """Not only an exact repeat: CVODE refuses anything within its roundoff of
    t0, and the test that takes it as reached is the event-capable loop's own
    (with its denormal floor, for 5e-324)."""
    x = np.asarray(_run(times).species)[:, 0]
    np.testing.assert_allclose(x, _closed_form(times), rtol=0, atol=CLOSED_FORM_ATOL)


def test_a_row_taken_as_reached_carries_the_time_asked_for():
    """A double after the start is not the start: its row is labelled with the
    time asked for, and a function of the clock is read there, as the
    event-capable loop records it."""
    t1 = float(np.nextafter(5.0, 6.0))
    r = _run([5.0, t1, 10.0], DECAY + "tt := time*3;\n")
    assert np.asarray(r.time).tolist() == [5.0, t1, 10.0]
    assert np.asarray(r.expressions["tt"])[1] == 3.0 * t1 != 15.0


@pytest.mark.parametrize(
    "times",
    [
        [-2.0, -2.0, -2.0, 3.0, 3.0],
        [-5.0, float(np.nextafter(-5.0, 0.0)), 0.0],
        [-1e6, -1e6, -1e6 + 1.0],
    ],
    ids=["replicates at -2", "a double after -5", "replicates at -1e6"],
)
def test_a_run_that_starts_below_zero(times):
    """CVODE's distance is relative to |t0|, and so is the test that takes an
    output as reached: a run that starts before t = 0 (#1538 measured one)
    reads its replicates as one that starts at 0 does. CV_TOO_CLOSE on main."""
    r = _run(times)
    x = np.asarray(r.species)[:, 0]
    np.testing.assert_allclose(x, _closed_form(times), rtol=0, atol=CLOSED_FORM_ATOL)
    assert np.asarray(r.time).tolist() == times


def test_a_uniform_grid_that_ends_below_zero():
    """The uniform-grid case, mirrored: t_span=(-1e6 - 1e-9, -1e6)."""
    with contextlib.redirect_stderr(io.StringIO()):
        r = bngsim.Simulator(bngsim.Model.from_antimony_string(DECAY), method="ode").run(
            t_span=(-1e6 - 1e-9, -1e6), n_points=101
        )
    x = np.asarray(r.species)[:, 0]
    assert len(x) == 101
    t = np.asarray(r.time)
    np.testing.assert_allclose(
        x, _closed_form(t, t0=t[0]), rtol=0, atol=_held_start_bound(0.1, t[0])
    )


def test_a_uniform_grid_whose_first_step_rounds_onto_the_start():
    """t_span=(1e6, 1e6 + 1e-9) with 101 points: the first outputs round onto
    the start, and run, run_batch and parameter_scan all reach run_warm."""
    span = {"t_span": (1e6, 1e6 + 1e-9), "n_points": 101}

    def sim():
        return bngsim.Simulator(bngsim.Model.from_antimony_string(DECAY), method="ode")

    with contextlib.redirect_stderr(io.StringIO()):
        runs = [sim().run(**span)]
        runs += sim().run_batch(**span, params=[{"k": 0.2}])
        scan = sim().parameter_scan("k", [0.3], **span)
    runs += list(getattr(scan, "results", scan))
    for r, k in zip(runs, (0.1, 0.2, 0.3), strict=True):
        x = np.asarray(r.species)[:, 0]
        assert len(x) == 101
        np.testing.assert_allclose(
            x,
            _closed_form(np.asarray(r.time), k=k, t0=1e6),
            rtol=0,
            atol=_held_start_bound(k, 1e6),
        )


def test_a_rule_species_and_a_clock_function_at_the_repeats(monkeypatch):
    """A row taken as reached is recorded as a stepped row is: the assignment
    rule's species copied in, and a function of the clock read at the time
    asked for."""
    r = _run(TRIPLICATE, RULES, monkeypatch=monkeypatch)
    names = list(r.species_names)
    y = np.asarray(r.species)[:, names.index("y")]
    tt = np.asarray(r.expressions["tt"])
    # y is twice x, and so is its error.
    np.testing.assert_allclose(y, 2 * _closed_form(TRIPLICATE), rtol=0, atol=2 * CLOSED_FORM_ATOL)
    assert tt.tolist() == [3.0 * t for t in TRIPLICATE]
    cold = _run(TRIPLICATE, RULES, cold=True, monkeypatch=monkeypatch)
    np.testing.assert_allclose(np.asarray(r.species), np.asarray(cold.species), rtol=LOOPS_RTOL)
    assert tt.tolist() == np.asarray(cold.expressions["tt"]).tolist()


def test_a_net_model(tmp_path):
    """The same through the .net loader, which BNGL models take."""
    net = tmp_path / "decay.net"
    net.write_text(
        "begin parameters\n 1 k 0.1\nend parameters\n"
        "begin species\n 1 X() 1\nend species\n"
        "begin reactions\n 1 1 0 k\nend reactions\n"
        "begin groups\n 1 Xtot 1\nend groups\n"
    )
    with contextlib.redirect_stderr(io.StringIO()):
        r = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
            sample_times=TRIPLICATE
        )
    np.testing.assert_allclose(
        np.asarray(r.species)[:, 0], _closed_form(TRIPLICATE), rtol=0, atol=CLOSED_FORM_ATOL
    )


def test_a_sensitivity_run_keeps_cvode_s_refusal_without_blaming_the_rate_law(caplog):
    """A sensitivity run never takes run_warm, and one whose first sample is a
    double after the start keeps CVODE's refusal on purpose: its row would come
    back as zeros (test_event_on_a_stop.py). It is a refusal of the time grid,
    so it is no longer routed through the GH #176 retry, whose warning blamed a
    discontinuous rate law and whose FD retry failed the same way."""
    with (
        caplog.at_level(logging.WARNING, logger="bngsim"),
        pytest.raises(bngsim.SimulationError, match=r"CV_TOO_CLOSE"),
    ):
        _run([5.0, float(np.nextafter(5.0, 6.0)), 10.0], sensitivity_params=["k"])
    assert not [r for r in caplog.records if "GH#176" in r.getMessage()]


def test_a_steady_state_stop_on_a_replicate_at_the_start(monkeypatch):
    """``steady_state=True`` tests ||f(t, y)|| after each row it records, a row
    taken as reached at the start among them (the rule TimeSpec states, and
    run_network -c's). A model at rest, x = 0, has f = 0 exactly, so the first
    row after row 0 settles: on this grid that is the second sample at t = 0,
    and the run stops there, as the event-capable loop stops. Exact equality,
    no tolerance: the rows are the initial state, and f is 0, not small."""
    rest = "species x; x = 0; k = 0.1; J1: x -> ; k*x;\n"
    for cold in (False, True):
        if cold:
            monkeypatch.setenv("BNGSIM_NO_WARM_CVODE", "1")
        with contextlib.redirect_stderr(io.StringIO()):
            r = bngsim.Simulator(bngsim.Model.from_antimony_string(rest), method="ode").run(
                sample_times=[0.0, 0.0, 0.0, 5.0], steady_state=True
            )
        assert np.asarray(r.time).tolist() == [0.0, 0.0]
        assert np.asarray(r.species)[:, 0].tolist() == [0.0, 0.0]
        assert r.solver_stats["steady_state_reached"]
