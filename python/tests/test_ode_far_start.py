"""Issue #1681: an ODE run far from t = 0, of a model that reads no clock.

CVODE advances its clock by ``t + h``, and far from 0 the doubles are coarse:
0.125 apart at 1e15, 16 apart at 1e17. ``A -> B`` at k = 1 over
``(1e15, 1e15 + 10)`` came back as A(+5) = 0.00144 for e^-5 = 0.00674, as
7.97e-7 at rtol 1e-10, and over ``(1e17, 1e17 + 64)`` A stayed at 1, each with
no error.

A model that reads no clock cannot tell where its run starts, so a run whose end
nearer 0 is at 2^20 or beyond is integrated on the time elapsed since that end,
and reports the times asked for: it is the run from 0, bit for bit. A run nearer
0, and a model that reads the clock, are run as before.

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
