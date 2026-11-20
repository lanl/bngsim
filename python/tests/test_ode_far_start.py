"""Issue #1681: an ODE run far from t = 0, of a model that reads no clock.

CVODE advances its clock by ``t + h``, and far from 0 the doubles are coarse:
0.125 apart at 1e15, 16 apart at 1e17. ``A -> B`` at k = 1 over
``(1e15, 1e15 + 10)`` came back as A(+5) = 0.00144 for e^-5 = 0.00674, as
7.97e-7 at rtol 1e-10, and over ``(1e17, 1e17 + 64)`` A stayed at 1, each with
no error.

A model that reads no clock cannot tell where its run starts, so a run whose end
nearer 0 is at 2^20 or beyond is integrated on the time elapsed since that end,
and reports the times asked for: it is the run from 0, bit for bit, where that
end is on the spacing of the doubles at the far end, and as accurate where it is
not. A run nearer 0 and a model that reads the clock are run as before; the stop
and the switch time of a switch on a counter clock move with the run.

The models are ``.net`` text. Which clock a run was on is read from a failing
run's message, which names its times: a law that leaves through infinity at
pi/2 fails at that time on the run's own clock, and says so.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

FAR = 2.0**20  # where a run starts counting its own clock (elapsed_clock_origin)


def _net(law):
    # A -> B at the rate f()*A.
    return (
        "begin parameters\n 1 k 1\nend parameters\n"
        f"begin functions\n 1 f() {law}\nend functions\n"
        "begin species\n 1 A() 1\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 f\nend reactions\n"
        "begin groups\n 1 Atot 1\n 2 Btot 2\nend groups\n"
    )


M = "http://www.w3.org/1998/Math/MathML"
# X' = 1 + X^2 from 0, so X = tan(t), beside an event that never fires (S stays
# at 0). A model with an event is run on the model's clock, which its trigger
# and its delay are read on.
EVENT_SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies><species id="S" compartment="C" initialConcentration="0"
 hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="X" value="0" constant="false"/></listOfParameters>
<listOfRules><rateRule variable="X"><math xmlns="{M}"><apply><plus/><cn>1</cn>
<apply><times/><ci>X</ci><ci>X</ci></apply></apply></math></rateRule></listOfRules>
<listOfEvents><event id="E" useValuesFromTriggerTime="true">
<trigger initialValue="true" persistent="true"><math xmlns="{M}"><apply><geq/>
<ci>S</ci><cn>1</cn></apply></math></trigger>
<listOfEventAssignments><eventAssignment variable="X"><math xmlns="{M}"><cn>0</cn></math>
</eventAssignment></listOfEventAssignments></event></listOfEvents>
</model></sbml>"""


def _sim(tmp_path, law="k", **kw):
    net = tmp_path / "far.net"
    net.write_text(_net(law))
    return bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", **kw)


def _species(result):
    return np.asarray(result.species, dtype=float)


@pytest.mark.parametrize("t0", [FAR, 1e9, 1e12, 1e15, 1e17])
def test_a_far_run_is_the_run_from_0(tmp_path, t0):
    # 160 long, so the eleven rows are doubles at 1e17 too, where they are 16
    # apart. On main A(+80) from 1e12 was 6e-4 of e^-80's scale off, and from
    # 1e17 A never moved.
    far = _sim(tmp_path).run((t0, t0 + 160.0), 11)
    near = _sim(tmp_path).run((0.0, 160.0), 11)
    assert np.array_equal(_species(far), _species(near))
    assert np.array_equal(np.asarray(far.observables), np.asarray(near.observables))
    assert np.array_equal(np.asarray(far.time), np.linspace(t0, t0 + 160.0, 11))


@pytest.mark.parametrize("span", [(1e15, 1e15 + 10.0), (-1e15 - 10.0, -1e15)])
def test_a_far_run_follows_the_closed_form(tmp_path, span):
    # A(t0 + s) = e^-s. A run at negative times counts from its end, the end
    # nearer 0, so it is the run over (-10, 0). bngsim's defaults
    # (rtol = atol = 1e-8): 3e-6 of A's start is what a run from 0 is held to.
    result = _sim(tmp_path).run(span, 11)
    elapsed = np.asarray(result.time) - span[0]
    assert np.abs(_species(result)[:, 0] - np.exp(-elapsed)).max() < 3e-6


def test_each_row_is_the_state_at_the_time_it_reports(tmp_path):
    # Eleven rows over (1e15, 1e15 + 1) are 0.1 apart as asked, and the doubles
    # there are 0.125 apart, so the times reported are the doubles nearest, and
    # each row is the state at the time it reports: A = e^-(t - t0). Drawn again
    # on the run's clock, the rows would be 0.1 apart under times 0.125 apart,
    # 2e-2 of A off.
    t0 = 1e15
    result = _sim(tmp_path).run((t0, t0 + 1.0), 11)
    elapsed = np.asarray(result.time) - t0
    assert set(np.round(np.diff(elapsed), 12)) <= {0.0, 0.125, 0.25} and elapsed[-1] == 1.0
    assert np.abs(_species(result)[:, 0] - np.exp(-elapsed)).max() < 3e-6


def test_a_far_run_of_a_state_switch_is_the_run_from_0(tmp_path):
    # A law that switches on the state, not the clock: its root is found on the
    # run's clock, at A = 0.5.
    law = "if(Atot>0.5,k,2*k)"
    far = _sim(tmp_path, law).run((1e15, 1e15 + 10.0), 11)
    near = _sim(tmp_path, law).run((0.0, 10.0), 11)
    assert np.array_equal(_species(far), _species(near))
    # A is e^-t down to 0.5, at ln 2, and 0.5 e^(-2 (t - ln 2)) from there.
    assert abs(_species(far)[2, 0] / (0.5 * math.exp(-2 * (2 - math.log(2)))) - 1) < 1e-5


@pytest.mark.parametrize(
    "run",
    [
        lambda sim, t: sim.run((t, t + 10.0), 2),
        lambda sim, t: sim.run(sample_times=[t, t + 1.0, t + 2.5, t + 10.0]),
        lambda sim, t: sim.run_batch((t, t + 10.0), 11, params=[{"k": 2.0}])[0],
        lambda sim, t: sim.parameter_scan("k", [2.0], t_span=(t, t + 10.0), n_points=11)[0],
    ],
    ids=["two rows", "sample_times", "run_batch", "parameter_scan"],
)
def test_each_way_of_asking_for_a_far_run_gives_the_run_from_0(tmp_path, run):
    far = run(_sim(tmp_path), 1e15)
    near = run(_sim(tmp_path), 0.0)
    assert np.array_equal(_species(far), _species(near))
    assert np.array_equal(np.asarray(far.time) - 1e15, np.asarray(near.time))


@pytest.mark.parametrize("kw", [{"codegen": False}, {"jacobian": "fd"}], ids=["interpreted", "fd"])
def test_a_far_run_is_the_run_from_0_on_each_route(tmp_path, kw):
    law = "k*Atot/(1+Atot)"
    far = _sim(tmp_path, law, **kw).run((1e15, 1e15 + 10.0), 11)
    near = _sim(tmp_path, law, **kw).run((0.0, 10.0), 11)
    assert np.array_equal(_species(far), _species(near))


def test_a_far_sensitivity_run_is_the_run_from_0(tmp_path):
    far = _sim(tmp_path, sensitivity_params=["k"]).run((1e15, 1e15 + 10.0), 11)
    near = _sim(tmp_path, sensitivity_params=["k"]).run((0.0, 10.0), 11)
    assert np.array_equal(np.asarray(far.sensitivities), np.asarray(near.sensitivities))
    # dA/dk = -s e^(-k s)
    s = np.asarray(near.time)
    column = np.asarray(far.sensitivities)[:, 0, 0]
    assert np.abs(column - (-s * np.exp(-s))).max() < 1e-5


def test_a_far_run_to_a_steady_state_reports_the_rows_it_reached(tmp_path):
    # The early stop keeps fewer rows than asked for; each is at its asked time.
    t0 = 1e15
    far = _sim(tmp_path).run((t0, t0 + 1000.0), 101, steady_state=True)
    near = _sim(tmp_path).run((0.0, 1000.0), 101, steady_state=True)
    assert 1 < len(far.time) < 101
    assert np.array_equal(np.asarray(far.time) - t0, np.asarray(near.time))
    assert np.array_equal(_species(far), _species(near))


def test_a_far_run_continues_from_where_the_last_ended(tmp_path):
    # Two legs, the second from the time the first reported: e^-10 at the end,
    # and the next leg may start there.
    t0 = 1e15
    sim = _sim(tmp_path)
    first = sim.run((t0, t0 + 5.0), 6)
    assert first.time[-1] == t0 + 5.0
    second = sim.run((t0 + 5.0, t0 + 10.0), 6)
    assert second.time[-1] == t0 + 10.0
    assert abs(_species(second)[-1, 0] - math.exp(-10.0)) < 3e-6


# ── Which clock a run was on ────────────────────────────────────────────────
#
# B' = k + B^2 from 0 is B = tan(t), with no value from pi/2 on: a source of B
# at the rate f(), which reads B alone.
NOTE = "The model reads no clock"


def _blowup_net(law="k+Btot*Btot"):
    return (
        "begin parameters\n 1 k 1\nend parameters\n"
        f"begin functions\n 1 f() {law}\nend functions\n"
        "begin species\n 1 B() 0\nend species\n"
        "begin reactions\n 1 0 1 f\nend reactions\n"
        "begin groups\n 1 Btot 1\nend groups\n"
    )


def _blowup_sim(tmp_path, law="k+Btot*Btot"):
    net = tmp_path / "blowup.net"
    net.write_text(_blowup_net(law))
    return bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode")


def _failure(sim, t0):
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run((t0, t0 + 3.0), 4)
    return str(info.value)


def test_a_far_runs_failure_names_its_own_clock_and_says_so(tmp_path):
    msg = _failure(_blowup_sim(tmp_path), 1e15)
    assert "non-finite value at t=1.5708" in msg, msg
    assert NOTE in msg and "elapsed since t = 1000000000000000:" in msg, msg


@pytest.mark.parametrize("t0", [0.0, 1000.0, FAR - 1.0])
def test_a_run_nearer_0_keeps_the_clock_it_had(tmp_path, t0):
    # Below 2^20 the doubles are within 1.2e-10 of each other, and the run is
    # the one it was: its failure names the absolute time.
    msg = _failure(_blowup_sim(tmp_path), t0)
    assert NOTE not in msg, msg
    assert f"non-finite value at t={t0 + math.pi / 2:.6g}" in msg, msg


@pytest.mark.parametrize("reader", ["a function of time()", "an event"])
def test_a_model_that_reads_the_clock_keeps_the_absolute_one(tmp_path, reader):
    # Its laws are evaluated at t, so the run is not moved, far or not.
    if reader == "an event":
        sim = bngsim.Simulator(bngsim.Model.from_sbml_string(EVENT_SBML), "ode")
    else:
        sim = _blowup_sim(tmp_path, "k+Btot*Btot+0*time()")
    msg = _failure(sim, FAR)
    assert NOTE not in msg, msg
    assert f"t={FAR + math.pi / 2:.6g}" in msg, msg


def test_a_far_runs_timeout_is_still_a_timeout(tmp_path):
    with pytest.raises(bngsim.SimulationTimeout) as info:
        _sim(tmp_path).run((1e15, 1e15 + 10.0), 11, timeout=1e-9)
    assert NOTE not in str(info.value)


# ── After review ────────────────────────────────────────────────────────────

# A unit-rate counter C from c0 = 0.5, and X' = k once C >= sigma: the crossing
# is at t0 + 2.5, and dX(t0 + 5)/dsigma = -k = -2 (issue #725's model).
CLOCK_NET = """begin parameters
    1 c0     0.5
    2 sigma  3
    3 k      2
    4 rc     1
end parameters
begin functions
    1 rate_X() if(t>=sigma,k,0)
end functions
begin species
    1 C() c0
    2 X() 0
end species
begin reactions
    1 0 1 rc
    2 0 2 rate_X
end reactions
begin groups
    1 t 1
end groups
"""


def test_a_far_runs_switch_time_moves_with_it(tmp_path, monkeypatch):
    # A switch time is a time to stop at and jump across. Left on the absolute
    # clock while the run moved onto its own, it was never reached: dX/dsigma
    # came back 0 for -2. The stops are dropped here so the switch time is all
    # the run is given.
    monkeypatch.setattr(
        bngsim._bngsim_core.SolverOptions, "set_crossing_stops", lambda self, *a: None
    )
    net = tmp_path / "clock.net"
    net.write_text(CLOCK_NET)
    t0 = 2.0**30
    run = bngsim.Simulator(
        bngsim.Model.from_net(str(net)), "ode", sensitivity_params=["sigma"]
    ).run(sample_times=[t0 + i for i in range(6)], rtol=1e-10, atol=1e-12)
    x = list(run.species_names).index("X()")
    assert np.asarray(run.sensitivities)[-1, x, 0] == pytest.approx(-2.0, abs=1e-6)


@pytest.mark.parametrize("sens", [False, True], ids=["plain", "sensitivity"])
@pytest.mark.parametrize("t0", [1e15, -(2.0**30) - 5.0], ids=["far", "far negative"])
def test_a_far_run_of_a_switch_on_a_counter_is_the_run_from_its_near_end(tmp_path, t0, sens):
    # CLOCK_NET reads no clock: its switch is on the counter C, and its crossing
    # is handed to the run as a stop (and, with sigma asked for, a switch time)
    # at t_start + (sigma - c0). Kept on the absolute clock for its stops, the
    # run from 1e15 came back as X(t0 + 5) = 2.12 for 5, 2.47 at rtol 1e-10,
    # with no error. The stops move with the run, so it is the run from its end
    # nearer 0, to the bit: from 0, or over (-5, 0) for the negative one. Run
    # with each stop left at its absolute time, it is not: no stop is ever
    # reached, and the step goes over the jump.
    net = tmp_path / "clock.net"
    net.write_text(CLOCK_NET)
    kw = {"sensitivity_params": ["sigma"]} if sens else {}

    def run(start):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", **kw)
        return sim.run(sample_times=[start + i for i in range(6)])

    far, near = run(t0), run(0.0 if t0 > 0 else -5.0)
    assert np.array_equal(_species(far), _species(near))
    x = list(far.species_names).index("X()")
    assert _species(far)[-1, x] == pytest.approx(5.0, abs=1e-6)
    if sens:
        assert np.array_equal(np.asarray(far.sensitivities), np.asarray(near.sensitivities))


def _clock_run(tmp_path, start, sens=(), c0=None, ic=()):
    net = tmp_path / "clock.net"
    net.write_text(
        CLOCK_NET if c0 is None else CLOCK_NET.replace("1 c0     0.5", f"1 c0     {c0}")
    )
    kw = {"sensitivity_params": list(sens)} if sens else {}
    if ic:
        kw["sensitivity_ic"] = list(ic)
    sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", **kw)
    # Rows 16 apart: the doubles' spacing at 1e17, so each is its own time there.
    return sim.run(sample_times=[start, start + 16.0, start + 32.0])


def test_a_counter_crossing_that_rounds_onto_a_far_start_is_refused(tmp_path):
    # Issue #1828: CLOCK_NET's crossing is 2.5 after the start. From 1e17, where
    # the doubles are 16 apart, its time rounds onto the start, the crossing is
    # dropped, and dX/dsigma came back 0 for -2 with no error, where a build
    # before #1681 refused the run (CV_TOO_CLOSE). It is refused, naming the issue.
    with pytest.raises(
        bngsim.SensitivityUnsupportedError, match=r"crosses 2\.5 after the start.*issue #1828"
    ):
        _clock_run(tmp_path, 1e17, sens=["sigma"])


def test_a_crossing_rounded_onto_a_nearer_start_is_refused_too(tmp_path):
    # The rounding depends on the spacing at the start and the crossing's
    # distance, not on how far out the start is: from 1e6 (1.2e-10 apart) a
    # crossing 4e-11 after it rounds onto it, and the column was 0 for -2.
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="issue #1828"):
        _clock_run(tmp_path, 1e6, sens=["sigma"], c0=repr(3 - 4e-11))


def test_a_counter_crossing_moved_only_by_the_counter_s_start_is_refused(tmp_path):
    # The clock's own sensitivity moves the crossing too (issue #725): with the
    # counter's initial value as the only column, it has the same jump to lose.
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="issue #1828"):
        _clock_run(tmp_path, 1e17, ic=["C()"])


def test_a_column_the_crossing_does_not_move_is_run(tmp_path):
    # k scales X past the switch but does not move it: there is no jump to lose,
    # and dX/dk from 1e17 is the run from 0's (13.5 and 29.5 at +16 and +32).
    far, near = _clock_run(tmp_path, 1e17, sens=["k"]), _clock_run(tmp_path, 0.0, sens=["k"])
    x = list(far.species_names).index("X()")
    assert np.allclose(
        np.asarray(far.sensitivities)[:, x, 0], np.asarray(near.sensitivities)[:, x, 0], atol=1e-6
    )
    assert np.asarray(far.sensitivities)[-1, x, 0] == pytest.approx(29.5, abs=1e-6)


def test_the_same_crossing_from_a_start_where_it_does_not_round_is_run(tmp_path):
    # From 1e15 the doubles are 0.125 apart: the crossing is placed, and the
    # column is the run from 0's.
    far, near = (
        _clock_run(tmp_path, 1e15, sens=["sigma"]),
        _clock_run(tmp_path, 0.0, sens=["sigma"]),
    )
    x = list(far.species_names).index("X()")
    assert np.asarray(far.sensitivities)[-1, x, 0] == pytest.approx(-2.0, abs=1e-6)
    assert np.array_equal(np.asarray(far.sensitivities), np.asarray(near.sensitivities))


def test_a_plain_run_from_1e17_is_not_refused(tmp_path):
    # No column for the lost switch to zero: X is right (2 (32 - 2.5) = 59).
    run = _clock_run(tmp_path, 1e17)
    x = list(run.species_names).index("X()")
    assert _species(run)[-1, x] == pytest.approx(59.0, abs=1e-6)


def test_a_counter_crossing_placed_off_its_time_is_refused(tmp_path):
    # Issue #1828, form 2: from 1e17 (16 apart) a crossing 20 after the start
    # is placed at +16, and its jump taken there: 0, -2, -2 for 0, 0, -2, with
    # no error, where a build before #1681 refused the run (CV_TOO_CLOSE).
    with pytest.raises(
        bngsim.SensitivityUnsupportedError,
        match=r"is placed 16 after the start, 4 off its time.*issue #1828",
    ):
        _clock_run(tmp_path, 1e17, sens=["sigma"], c0="-17")


@pytest.mark.parametrize("c0", ["-33", "-37"], ids=["36 after", "40 after"])
def test_a_crossing_past_the_end_that_rounds_onto_it_makes_no_jump(tmp_path, c0):
    # From 1e17 a crossing 36 or 40 after the start, past a run of 32, is placed
    # on the end, and the run made its jump in the last row: 0, 0, -2 for
    # 0, 0, 0. By its elapsed time, which does not round, it is past the end and
    # no crossing of the run: the column is the run from 0's.
    far, near = (_clock_run(tmp_path, t, sens=["sigma"], c0=c0) for t in (1e17, 0.0))
    x = list(far.species_names).index("X()")
    assert np.array_equal(np.asarray(far.sensitivities)[:, x, 0], [0.0, 0.0, 0.0])
    assert np.array_equal(np.asarray(far.sensitivities), np.asarray(near.sensitivities))


def test_the_refusal_names_the_earliest_crossing_lost(tmp_path):
    # A period-5 schedule on the counter, on while its phase is at least 3, the
    # counter from 0.5, with the period requested: the off-edge 4.5 after the
    # start and the on-edge 7.5 after it both move with P, and from 1e17 both
    # round onto the start. The message named the first found, 7.5.
    text = CLOCK_NET.replace("4 rc     1\n", "4 rc     1\n    5 P      5\n").replace(
        "if(t>=sigma,k,0)", "if(t-P*floor(t/P)>=sigma,k,0)"
    )
    assert "5 P      5" in text and "floor(t/P)" in text
    net = tmp_path / "schedule.net"
    net.write_text(text)
    sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", sensitivity_params=["P"])
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crosses 4.5 after the start"):
        sim.run(sample_times=[1e17, 1e17 + 16.0, 1e17 + 32.0])


def test_a_crossing_placed_off_by_a_large_share_of_the_run_is_refused(tmp_path):
    # From 1e16 (2 apart) a crossing 2.5 after the start is placed at +2, 0.5
    # off in a run of 32: where the jump depends on when it is taken, the
    # column was 0.27 for 0.16; a build before #1681 refused it.
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="0.5 off its time"):
        _clock_run(tmp_path, 1e16, sens=["sigma"])


def test_a_crossing_placed_off_by_a_small_share_of_the_run_is_run(tmp_path):
    # From 1e15 (0.125 apart) a crossing 2.53 after the start is placed at 2.5,
    # 0.03 of a run of 32 (under 0.1%): run, and X' = k's column is the run
    # from 0's (-2 wherever the jump is taken).
    far = _clock_run(tmp_path, 1e15, sens=["sigma"], c0=repr(3 - 2.53))
    x = list(far.species_names).index("X()")
    assert np.asarray(far.sensitivities)[-1, x, 0] == pytest.approx(-2.0, abs=1e-6)


def test_a_crossing_exactly_on_the_run_s_end_is_run(tmp_path):
    # The window is half-open: a crossing on t_end makes its jump into the last
    # row. 32 after the start is a double at 1e17, so it is placed where it is,
    # and the run is the run from 0's.
    far, near = (_clock_run(tmp_path, t, sens=["sigma"], c0="-29") for t in (1e17, 0.0))
    x = list(far.species_names).index("X()")
    # The value itself, in both: the same wrong window in both would agree.
    assert np.array_equal(np.asarray(far.sensitivities)[:, x, 0], [0.0, 0.0, -2.0])
    assert np.array_equal(np.asarray(near.sensitivities)[:, x, 0], [0.0, 0.0, -2.0])


def test_a_crossing_past_the_run_s_end_is_not_refused(tmp_path):
    # Dropped by the same window, but for being after the end, not for
    # rounding: c0 = -100 puts it 103 after the start, past +32. X stays 0.
    run = _clock_run(tmp_path, 1e17, sens=["sigma"], c0="-100")
    x = list(run.species_names).index("X()")
    assert _species(run)[-1, x] == 0.0
    assert np.asarray(run.sensitivities)[-1, x, 0] == 0.0


@pytest.mark.parametrize("c0", ["3", "40"], ids=["on the start", "past the end"])
def test_a_counter_with_no_crossing_inside_the_run_is_not_refused(tmp_path, c0):
    # c0 = sigma: the switch is on at the start; c0 = 40: the counter is past
    # sigma throughout. Neither has a crossing inside the run to lose.
    run = _clock_run(tmp_path, 1e17, sens=["sigma"], c0=c0)
    x = list(run.species_names).index("X()")
    assert _species(run)[-1, x] == pytest.approx(64.0, abs=1e-6)


def test_a_schedule_edge_that_rounds_onto_the_start_is_refused(tmp_path):
    # The schedule path drops an edge by the same window. A schedule on the
    # counter, period 5, on while its phase is at least sigma = 3, the counter
    # 4e-11 below the first on-edge: on-edges 4e-11, 5, 10, 15, ... after the
    # start, each moved by sigma, so dX/dsigma is -k = -2 per edge passed, -8 at
    # +16 and -14 at +32. From 0 the first edge is placed. From 1e6 it rounds
    # onto the start, and the columns were -6 and -12 with no error.
    law = "if(t-P*floor(t/P)>=sigma,k,0)"
    text = (
        CLOCK_NET.replace("1 c0     0.5", f"1 c0     {3 - 4e-11!r}")
        .replace("4 rc     1\n", "4 rc     1\n    5 P      5\n")
        .replace("if(t>=sigma,k,0)", law)
    )
    assert law in text and "5 P      5" in text
    net = tmp_path / "schedule.net"
    net.write_text(text)

    def run(start):
        sim = bngsim.Simulator(
            bngsim.Model.from_net(str(net)), "ode", sensitivity_params=["sigma"]
        )
        return sim.run(sample_times=[start, start + 16.0, start + 32.0])

    near = run(0.0)
    x = list(near.species_names).index("X()")
    # The jumps are exact multiples of k; 1e-6 is the file's bound on the column.
    assert np.asarray(near.sensitivities)[1:, x, 0] == pytest.approx([-8.0, -14.0], abs=1e-6)
    with pytest.raises(
        bngsim.SensitivityUnsupportedError, match=r"crosses 4e-11 after the start.*issue #1828"
    ):
        run(1e6)


# Two counters, C from 0.5 and D from 1.5, crossing 3 and 4 at one instant,
# t0 + 2.5: the run is given one stop there and a probe for each crossing
# (issue #767), and every one moves with the run.
TWO_COUNTERS_NET = """begin parameters
    1 c0     0.5
    2 d0     1.5
    3 sigma  3
    4 s2     4
    5 k      2
    6 k2     3
    7 rc     1
end parameters
begin functions
    1 rate_X() if(t>=sigma,k,0)
    2 rate_Y() if(u>=s2,k2,0)
end functions
begin species
    1 C() c0
    2 D() d0
    3 X() 0
    4 Y() 0
end species
begin reactions
    1 0 1 rc
    2 0 2 rc
    3 0 3 rate_X
    4 0 4 rate_Y
end reactions
begin groups
    1 t 1
    2 u 2
end groups
"""


@pytest.mark.parametrize("sens", [False, True], ids=["plain", "sensitivity"])
def test_a_far_run_of_two_crossings_on_one_instant_is_the_run_from_0(tmp_path, sens):
    net = tmp_path / "two.net"
    net.write_text(TWO_COUNTERS_NET)
    kw = {"sensitivity_params": ["sigma"]} if sens else {}

    def run(start):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", **kw)
        return sim.run(sample_times=[start + i for i in range(6)])

    far, near = run(1e15), run(0.0)
    assert np.array_equal(_species(far), _species(near))
    # X = k (s - 2.5) and Y = k2 (s - 2.5) past the instant: 5 and 7.5 at s = 5.
    x, y = (list(far.species_names).index(name) for name in ("X()", "Y()"))
    assert _species(far)[-1, x] == pytest.approx(5.0, abs=1e-6)
    assert _species(far)[-1, y] == pytest.approx(7.5, abs=1e-6)
    if sens:
        assert np.array_equal(np.asarray(far.sensitivities), np.asarray(near.sensitivities))
        assert np.asarray(far.sensitivities)[-1, x, 0] == pytest.approx(-2.0, abs=1e-6)


# S' = amp below lvl = 1 and -amp above it, from S = 0: at t = 1 it slides
# along S = 1, which a sensitivity run refuses (issue #926), naming the time.
SLIDE_NET = """begin parameters
    1 amp 1.0
    2 lvl 1.0
end parameters
begin functions
    1 f() if(Sobs<lvl,amp,-amp)
end functions
begin species
    1 S() 0.0
end species
begin reactions
    1 0 1 f
end reactions
begin groups
    1 Sobs 1
end groups
"""


def test_a_far_sensitivity_refusal_names_its_own_clock_and_keeps_its_type(tmp_path):
    net = tmp_path / "slide.net"
    net.write_text(SLIDE_NET)
    sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", sensitivity_params=["amp"])
    with pytest.raises(bngsim.SensitivityRefusedDuringRunError) as info:
        sim.run(sample_times=[1e15, 1e15 + 0.5, 1e15 + 1.5], rtol=1e-6, atol=1e-8, timeout=10.0)
    msg = str(info.value)
    assert "at t=1 " in msg and "issue #926" in msg, msg
    assert NOTE in msg and "elapsed since t = 1000000000000000:" in msg, msg


def test_a_far_failure_that_names_no_time_says_nothing_of_the_clock(tmp_path):
    # The message is the one the same request gives nearer 0.
    messages = []
    for t0 in (0.0, 1e15):
        with pytest.raises(bngsim.SimulationError) as info:
            _sim(tmp_path, jacobian="bogus").run((t0, t0 + 10.0), 11)
        messages.append(str(info.value))
    assert "bogus" in messages[0] and messages[1] == messages[0], messages


def test_a_run_past_2_72_is_moved_once_and_says_so_once(tmp_path):
    # Past 2^72 the doubles are over 2^20 apart, and what is left of the start
    # once it is rounded to them can be 2^20 again: from 2^73 - 2^20 to 2^73 +
    # 2^30 the spacing at the far end is 2^21, and 2^20 is left. That run was
    # moved again, and its failure carried two notes with two times to add.
    t0 = 2.0**73 - 2.0**20
    with pytest.raises(bngsim.SimulationError) as info:
        _blowup_sim(tmp_path).run((t0, 2.0**73 + 2.0**30), 2)
    msg = str(info.value)
    assert "non-finite value at t=1.5708" in msg, msg
    assert msg.count(NOTE) == 1 and f"elapsed since t = {t0:.17g}:" in msg, msg


def test_a_far_run_whose_start_is_not_on_the_far_ends_spacing(tmp_path):
    # From 2^53 - 1 the doubles at the far end are 2 apart, so the run starts at
    # what is left, 1, on its own clock: not the run from 0 to the bit, but as
    # accurate. A(t0 + s) = e^-s, at the rows asked for.
    t0 = 2.0**53 - 1.0
    result = _sim(tmp_path).run(sample_times=[t0, t0 + 1.0, t0 + 5.0, t0 + 101.0])
    assert np.array_equal(np.asarray(result.time), [t0, t0 + 1.0, t0 + 5.0, t0 + 101.0])
    elapsed = np.array([0.0, 1.0, 5.0, 101.0])
    assert np.abs(_species(result)[:, 0] - np.exp(-elapsed)).max() < 3e-6
