"""A state come to rest on a state-switch surface is refused by name (issue #1548).

l-type-calcium-channel-dynamics.bngl gives v_rec() = if(-70 + Voltage_Level < -20,
0.5, 0.05), and Voltage() relaxes as 50 - 49e^(-t): the threshold is that steady
state. The exact solution never reaches it, so v_rec() stays 0.5 and Calcium()
settles at 957.956. A run is within its tolerance of the threshold by t ~ 20 and
then rests there, and the branch the rate law reads from there is whichever side
rounding left the state on: main returned 957.956 on 301 outputs and 187.682, the
other branch's steady state, on 151, with nothing said. run_network does the
same on 301 outputs.

A run is now refused by name where a switch's residual has been off its surface,
is now within the band the run's tolerance leaves it, the flow at the state and
on each side of the surface moves it less than that band over the rest of the
run, and the rate law jumps across the surface by enough to matter. The models
here are built so that each answer is a closed form: A' = k·(thr − A) relaxes A
onto thr from either side, A = thr + (A0 − thr)·e^(−k·t).

The run is asked at the start, at each output, and between outputs: where CVODE
returns at a root or a stop, and every few steps (lanl/bngsim#1589). Asked at
outputs alone, a rest that began and ended between two of them was never seen,
and whether a run was refused depended on its output grid. A plain run of an
SBML model is asked about its thresholds as a sensitivity run of it is
(lanl/bngsim#1588); it was not asked at all.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest

REFUSED = "come to rest on the surface.*issue #1548"

# A relaxes to `rest`; W's rate law switches on A against `thr`.
RELAX = """begin parameters
    1 A0 {a0!r}
    2 k 1
    3 rest 2
    4 thr {thr!r}
    5 kb 3
    6 kc 5
end parameters
begin functions
    1 src() k*rest
    2 fW() {law}
end functions
begin species
    1 A() A0
    2 W() 0
end species
begin reactions
    1 0 1 src
    2 1 0 k
    3 0 2 fW
end reactions
begin groups
    1 Aobs 1
end groups
"""
JUMP = "if(Aobs<thr,kb,0.25*kb)"
CLAMP = "if(Aobs<thr,kc*(thr-Aobs),0)"  # continuous at its surface


def _sim(tmp_path: Path, law: str, a0: float, thr: float = 2.0, **kw) -> bngsim.Simulator:
    path = tmp_path / "m.net"
    path.write_text(RELAX.format(a0=a0, thr=thr, law=law))
    return bngsim.Simulator(bngsim.Model.from_net(path), method="ode", **kw)


def _w(result: bngsim.Result) -> float:
    return float(np.asarray(result.species)[-1, list(result.species_names).index("W()")])


def _refused_at(info: pytest.ExceptionInfo) -> float:
    """The time a refusal names."""
    found = re.search(r"At t=(\S+) the state has come to rest", str(info.value))
    assert found is not None, str(info.value)
    return float(found.group(1))


@pytest.mark.parametrize("a0", [10.0, 0.0], ids=["from-above", "from-below"])
@pytest.mark.parametrize(("rtol", "n_points"), [(1e-8, 301), (1e-8, 151), (1e-6, 41), (1e-11, 7)])
def test_a_state_relaxing_onto_the_threshold_of_a_jump_is_refused(tmp_path, a0, rtol, n_points):
    """On the exact solution W gathers on the branch A started on for ever,
    but the run cannot say which side of thr A is on once it rests there."""
    sim = _sim(tmp_path, JUMP, a0)
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(t_span=(0.0, 60.0), n_points=n_points, rtol=rtol, atol=rtol)
    assert "'(Aobs)-(thr)'" in str(info.value)
    assert "dW()/dt" in str(info.value)


@pytest.mark.parametrize(
    "kw", [{"jacobian": "fd"}, {"jacobian": "analytical"}, {"codegen": True}], ids=str
)
def test_every_jacobian_and_backend_refuses_it(tmp_path, kw):
    sim = _sim(tmp_path, JUMP, 10.0, **kw)
    with pytest.raises(bngsim.SimulationError, match=REFUSED):
        sim.run(t_span=(0.0, 60.0), n_points=121, rtol=1e-8, atol=1e-8)


def test_a_run_asked_for_its_two_ends_alone_is_refused(tmp_path):
    """Asked over the stretch since the output before and the rest of the run:
    a run asked only for t_end has the whole of it to speak for. It is refused
    in the rest, which A enters at t ~ 17.7, and not at t_end, where the rest
    was first seen before the run was asked between outputs (issue #1589)."""
    sim = _sim(tmp_path, JUMP, 10.0)
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(t_span=(0.0, 60.0), n_points=2, rtol=1e-8, atol=1e-8)
    assert 17.0 < _refused_at(info) < 30.0


def test_a_sensitivity_run_is_refused_the_same_way(tmp_path):
    sim = _sim(tmp_path, JUMP, 10.0, sensitivity_params=["kb", "thr"])
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(t_span=(0.0, 60.0), n_points=61, rtol=1e-8, atol=1e-10)
    assert "#926" not in str(info.value)


@pytest.mark.parametrize("n_points", [121, 2])
@pytest.mark.parametrize("a0", [10.0, 0.0], ids=["from-above", "from-below"])
def test_a_threshold_the_run_can_tell_from_the_rest_runs(tmp_path, a0, n_points):
    """thr = 2 + 1e-4, far outside the band where A rests: from below A never
    reaches it, and from above it crosses it once, at t* = ln((A0 − 2)/1e-4).
    A crosses at 1e-4 a unit of time there, so an error e in A moves t* by
    e/1e-4: run at 1e-10, where that is 2e-6. On two outputs the crossing is
    asked about where it is a root, between them (issue #1589), and is carried
    across by the flow it comes on."""
    thr = 2.0 + 1e-4
    result = _sim(tmp_path, JUMP, a0, thr=thr).run(
        t_span=(0.0, 60.0), n_points=n_points, rtol=1e-10, atol=1e-10
    )
    if a0 < thr:
        want = 3.0 * 60.0
    else:
        t_star = math.log((a0 - 2.0) / 1e-4)
        want = 0.75 * t_star + 3.0 * (60.0 - t_star)
    assert _w(result) == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("n_points", [121, 2])
@pytest.mark.parametrize("a0", [10.0, 0.0], ids=["from-above", "from-below"])
def test_a_continuous_switch_resting_on_its_surface_runs(tmp_path, a0, n_points):
    """The clamp kc·(thr − A)⁺ has no jump at thr, so where A rests there is no
    branch to be wrong about: W = kc·∫(thr − A)⁺ = kc·2·(1 − e^(−T)) from below,
    and 0 from above."""
    result = _sim(tmp_path, CLAMP, a0).run(
        t_span=(0.0, 60.0), n_points=n_points, rtol=1e-8, atol=1e-8
    )
    want = 5.0 * 2.0 * (1.0 - math.exp(-60.0)) if a0 < 2.0 else 0.0
    assert _w(result) == pytest.approx(want, rel=1e-6, abs=1e-6)


PUSH = "if(Aobs<thr,0,c)"  # a push up, above thr only


def _pushed(tmp_path: Path, c: float, a0: float = 0.0, **kw) -> bngsim.Simulator:
    """RELAX, with A pushed up at c once it is above thr = 2, where it rests."""
    text = RELAX.format(a0=a0, thr=2.0, law=JUMP)
    text = text.replace("    6 kc 5\n", f"    6 kc 5\n    7 c {c!r}\n")
    text = text.replace("end functions\n", f"    3 push() {PUSH}\nend functions\n")
    text = text.replace("    3 0 2 fW\n", "    3 0 2 fW\n    4 0 1 push\n")
    path = tmp_path / "pushed.net"
    path.write_text(text)
    return bngsim.Simulator(bngsim.Model.from_net(path), method="ode", **kw)


@pytest.mark.parametrize("c", [1e-3, 1.0])
@pytest.mark.parametrize(("rtol", "n_points"), [(1e-8, 301), (1e-8, 151), (1e-6, 41), (1e-8, 76)])
def test_a_state_resting_against_a_side_that_pushes_it_off_is_refused(tmp_path, c, rtol, n_points):
    """A relaxes onto thr = 2 from below, and above it A is pushed further up
    at c. On the exact solution A never reaches thr and W gathers kb = 3 for
    ever, 180 by t = 60. Once rounding puts A above thr, though, it is carried
    off at c and W gathers 0.75: a first cut of this refusal asked the flow on
    both sides to be small, and c on the far side let these runs return W of 80
    to 127, with nothing said, on every one of these grids."""
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        _pushed(tmp_path, c).run(t_span=(0.0, 60.0), n_points=n_points, rtol=rtol, atol=rtol)
    assert "'(Aobs)-(thr)'" in str(info.value)


@pytest.mark.parametrize("c", [1e-3, 1.0])
@pytest.mark.parametrize("n_points", [2, 3, 5])
def test_a_rest_between_two_outputs_is_refused(tmp_path, c, n_points):
    """lanl/bngsim#1589. The pushed case on grids with no output from where A
    comes within the band of thr (t ~ 16.3) to where rounding first put it
    across. Asked at outputs alone, the run returned W(60) = 86 for 180 on
    two outputs, with nothing said, and was refused on every denser grid
    tried. It is now refused in the rest, between outputs."""
    grid = np.linspace(0.0, 60.0, n_points)
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        _pushed(tmp_path, c).run(t_span=(0.0, 60.0), n_points=n_points, rtol=1e-8, atol=1e-8)
    t = _refused_at(info)
    assert 16.0 < t < 30.0
    assert np.min(np.abs(grid - t)) > 1e-3


def test_a_sensitivity_run_is_refused_between_outputs_too(tmp_path):
    """The same, with sensitivity columns: a run that already takes its steps
    in short batches for its slides is asked after each of them as well."""
    sim = _pushed(tmp_path, 1.0, sensitivity_params=["kb"])
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(t_span=(0.0, 60.0), n_points=2, rtol=1e-8, atol=1e-10)
    assert 16.0 < _refused_at(info) < 30.0
    assert "#926" not in str(info.value)


# RELAX, with A then sent off to 10 by a switch in time at t = 30.
LEFT = RELAX.format(a0=0.0, thr=2.0, law=JUMP).replace(
    "    1 src() k*rest\n", "    1 src() if(time()<30,k*rest,k*10)\n"
)


@pytest.mark.parametrize("n_points", [2, 3, 7, 61])
def test_a_rest_a_later_input_ends_is_refused_on_every_grid(tmp_path, n_points):
    """A rests on thr from t ~ 16 until a switch in time at t = 30 sends it off
    to 10, so no output after the rest sees it there. Asked at outputs alone,
    the run was refused on 7 outputs or more and ran on 2 and 3, where W(60)
    came back as the exact 3·30 + 0.75·30 = 112.5 only because rounding had
    left A under thr throughout the rest."""
    path = tmp_path / "left.net"
    path.write_text(LEFT)
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(t_span=(0.0, 60.0), n_points=n_points, rtol=1e-8, atol=1e-8)
    assert 16.0 < _refused_at(info) <= 30.0


# S slides on S = 1 under flows of ±e^(−10t) from either side, decaying to
# nothing; W's rate law is {law}.
DECAYING_SLIDE = """begin parameters
    1 kb 3
end parameters
begin functions
    1 a() exp(-10*time())
    2 f() if(Sobs<1,a(),-a())
    3 fW() {law}
end functions
begin species
    1 S() 0.95
    2 W() 0
end species
begin reactions
    1 0 1 f
    2 0 2 fW
end reactions
begin groups
    1 Sobs 1
end groups
"""
DECAY_GRIDS = [[0.0, 3.5], [0.0, 1.4, 1.45, 1.5, 3.5], list(np.linspace(0.0, 3.5, 36))]


@pytest.mark.parametrize("times", DECAY_GRIDS, ids=len)
def test_a_slide_whose_flows_decay_runs(tmp_path, times):
    """S reaches 1 at t = ln 2/10 and is held there from both sides by flows
    that decay to nothing. Once they are under the band, the flows read as at
    rest, and the jump across the surface, 2e^(−10t) in dS/dt, read as one
    that moves S by its unit over the rest of the run. But both sides carry S
    onto the surface, so that jump only takes S back onto it from whichever
    side rounding left it: S stays at 1. Main refused this run on the grid
    with outputs at 1.4 to 1.5 and ran it on the others; asked between outputs
    it would have refused it on every grid (issue #1589). That part of the
    jump is not counted, and the run is held to S = 1."""
    path = tmp_path / "slide.net"
    path.write_text(DECAYING_SLIDE.format(law="0"))
    result = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        sample_times=times, rtol=1e-6, atol=1e-8
    )
    s = np.asarray(result.species)[:, list(result.species_names).index("S()")]
    assert s[-1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("times", DECAY_GRIDS, ids=len)
def test_a_slide_whose_flows_decay_beneath_a_gated_rate_is_refused(tmp_path, times):
    """Control: the same slide, with W gathering kb under the surface and
    kb/4 over it. Once the flows that hold S on the surface are under its
    band, which side S is on, and so what W gathers, is rounding's call, and
    the jump in dW/dt is not one the surface's flows take back."""
    path = tmp_path / "slide.net"
    path.write_text(DECAYING_SLIDE.format(law="if(Sobs<1,kb,0.25*kb)"))
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(sample_times=times, rtol=1e-6, atol=1e-8)
    assert "dW()/dt" in str(info.value)


@pytest.mark.parametrize("n_points", [41, 2])
def test_a_species_held_at_its_cap_from_one_side_runs(tmp_path, n_points):
    """Control: A is made at 1 while it is under cap = 5 and not at all above
    it, so it reaches cap at t = 5 and stays. The flow under the surface points
    up onto it and puts A back whenever rounding takes it under, so its branch
    is the model's: W gathers kb = 3 until t = 5 and 0.75 from there."""
    text = """begin parameters
    1 cap 5
    2 kb 3
end parameters
begin functions
    1 make() if(Aobs<cap,1,0)
    2 fW() if(Aobs<cap,kb,0.25*kb)
end functions
begin species
    1 A() 0
    2 W() 0
end species
begin reactions
    1 0 1 make
    2 0 2 fW
end reactions
begin groups
    1 Aobs 1
end groups
"""
    path = tmp_path / "cap.net"
    path.write_text(text)
    result = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, 20.0), n_points=n_points, rtol=1e-8, atol=1e-8
    )
    assert _w(result) == pytest.approx(3.0 * 5.0 + 0.75 * 15.0, rel=1e-6)


@pytest.mark.parametrize("sens", [["k"], None], ids=["sensitivity", "plain"])
def test_a_species_an_event_sets_onto_its_gate_runs(sens):
    """S decays until an event sets it to 0 at t = 5, where nothing moves it
    again, and ``S > 0`` gates W's rate. S was off its surface before the event,
    but the event puts it exactly on it, so the gate reads the model's branch
    (0 > 0 is false), not rounding's: W = 5·k and dW/dk = 5. A first cut
    refused this run as come to rest. The plain case holds the event hook to
    a run with no sensitivity record of its fires, now that a plain run of an
    SBML model is asked (lanl/bngsim#1588)."""
    text = (
        "species S, W; S = 1; W = 0; k = 3; kd = 0.1\n"
        "Jd: S -> ; kd*S\nE1: at (time > 5): S = 0\nJ: -> W; piecewise(k, S > 0, 0)\n"
    )
    extra = {} if sens is None else {"sensitivity_params": sens}
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode", **extra)
    result = sim.run(t_span=(0.0, 10.0), n_points=11, rtol=1e-8, atol=1e-8)
    w = list(result.species_names).index("W")
    assert float(np.asarray(result.species)[-1, w]) == pytest.approx(15.0, rel=1e-6)
    if sens is not None:
        assert float(np.asarray(result.sensitivities)[-1, w, 0]) == pytest.approx(5.0, rel=1e-6)


# A sits at 0 until an event sets it a hair over its gate at t = 5, where c
# carries it up and off at 1e-6 a unit of time.
SET_OVER = (
    "species A, W; A = 0; W = 0; thr = 2; kb = 3; c = 1e-6\n"
    "Jp: -> A; piecewise(c, A > thr, 0)\nJw: -> W; piecewise(kb, A < thr, 0.25*kb)\n"
    "E1: at (time >= 5): A = thr + 1e-12\n"
)


@pytest.mark.parametrize(
    "times", [[0.0, 5.1, 10.0], [0.0, 10.0], list(np.linspace(0.0, 10.0, 101))], ids=len
)
def test_an_event_that_sets_a_state_over_its_gate_runs(times):
    """A is under thr until the event and over it after, so W = 3·5 + 0.75·5.
    For 0.17 time units after the event A is within the band of thr, and the
    run is asked about it there, between outputs (issue #1589). Nothing
    carries it onto thr from either side, but c carries it off the side the
    event put it on, so it is not at rest."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(SET_OVER), method="ode")
    result = sim.run(sample_times=times, rtol=1e-8, atol=1e-8)
    w = list(result.species_names).index("W")
    assert float(np.asarray(result.species)[-1, w]) == pytest.approx(18.75, rel=1e-9)


def test_a_species_that_never_leaves_its_surface_runs(tmp_path):
    """S is 0 at the start and nothing makes it: its gate ``S > 0`` reads the
    model, not rounding, and the run is not asked about it."""
    text = """begin parameters
    1 kb 3
end parameters
begin functions
    1 fW() if(Sobs>0,kb,0)
end functions
begin species
    1 S() 0
    2 W() 0
end species
begin reactions
    1 0 2 fW
end reactions
begin groups
    1 Sobs 1
end groups
"""
    path = tmp_path / "s.net"
    path.write_text(text)
    result = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, 10.0), n_points=11, rtol=1e-8, atol=1e-8
    )
    assert _w(result) == 0.0


@pytest.mark.parametrize("n_points", [21, 2])
def test_a_transversal_crossing_runs(tmp_path, n_points):
    """A rises at 1 a unit of time through thr = 5, so W switches at t = 5. On
    two outputs the run is asked at the crossing's root, where A has just
    crossed, and the flow it came on carries it across (issue #1589)."""
    text = RELAX.format(a0=0.0, thr=5.0, law=JUMP).replace("1 0 1 src\n    2 1 0 k\n", "1 0 1 k\n")
    path = tmp_path / "t.net"
    path.write_text(text.replace("    3 0 2 fW\n", "    2 0 2 fW\n"))
    result = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, 10.0), n_points=n_points, rtol=1e-8, atol=1e-8
    )
    assert _w(result) == pytest.approx(3.0 * 5.0 + 0.75 * 5.0, rel=1e-6)


@pytest.mark.parametrize("n_points", [301, 151, 76])
def test_the_upstream_model_is_refused_on_every_grid(data_dir, tmp_path, n_points):
    """The rulehub model itself, k_v_stim = 50: refused on the grid where main
    was right (301 outputs) and on those where it was 0.633 off (151, 76)."""
    text = (data_dir / "ltype_calcium_discontinuous_jacobian.net").read_text()
    old = "   16 k_v_stim        49.99999999999  # Constant\n"
    assert text.count(old) == 1
    path = tmp_path / "upstream.net"
    path.write_text(text.replace(old, "   16 k_v_stim        50.0  # Constant\n"))
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(t_span=(0.0, 150.0), n_points=n_points, rtol=1e-8, atol=1e-8)
    assert "Voltage_Level" in str(info.value)


@pytest.mark.parametrize("times", [[0.0, 2.0, 4.0, 8.0, 12.0], [0.0, 12.0]], ids=len)
def test_a_plain_sbml_run_resting_on_its_gate_is_refused(times):
    """lanl/bngsim#1588. B decays under its guard ``B > 1e-9`` and rests there,
    within the absolute tolerance of 1e-8, so when Y stops gathering is
    rounding's call: the plain run returned Y(12) = 2.267 for k·ln(1e9)/lam =
    2.072, with nothing said. A sensitivity run of it was refused
    (test_state_switch_on_a_clock_instant.py). The plain run handed the core
    its thresholds only as conditions to watch for a pinned state, which the
    rest refusal did not read, and they are the ones it asks about now. On
    the run's two ends alone it is asked between them (lanl/bngsim#1589)."""
    text = (
        "species B, X, Y; B = 1; X = 0; Y = 0; lam = 5; k = 0.5; r = 1.5; tau = 8\n"
        "Jb: B -> ; lam*B\nJy: -> Y; piecewise(k, B > 1e-9, 0)\n"
        "Jx: -> X; piecewise(r, time >= tau, 0)\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(sample_times=times, rtol=1e-8, atol=1e-8)
    assert "'(B)-(1e-09)'" in str(info.value)


# S never moves (it is made and used by nothing), and thr() falls past nothing:
# from 1.01 to 1 around t = 30.03, settling 0.0022 above S. Y's rate is 10
# while S < thr(), so the whole run (#1597's stiff sweep, case 308).
STILL_BESIDE = """begin parameters
    1 k   10.0
    2 tp  30.028149168667564
    3 kf  1000000.0
    4 kr  500000.0
end parameters
begin functions
    1 thr() 1+0.005*(1+tanh(-(time()-tp)/0.0005))
    2 r() if(Sobs<thr(),k,0)
end functions
begin species
    1 S() 0.9977951597346537
    2 Y() 0
    3 P() 1.0
    4 Q() 0.0
end species
begin reactions
    1 0 2 r
    2 3 4 kf
    3 4 3 kr
end reactions
begin groups
    1 Sobs 1
end groups
"""


def test_a_species_that_never_moves_beside_a_settled_threshold_runs(tmp_path):
    """At rtol 1e-3, S is 0.0022 below thr() once it settles, inside the bar
    of 0.009 the tolerance leaves the residual, and neither side's flow moves
    it: the run was refused as at rest on the surface. But S never moves, and
    BDF integrates a species at one constant rate exactly, so its side is the
    model's to say and not rounding's. Such a species is read at its rounding
    in the bar (Impl::rate_constant, issue #1623), and the run returns Y =
    10.5. A species the run does move, resting there, is refused as before."""
    path = tmp_path / "still.net"
    path.write_text(STILL_BESIDE)
    result = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        sample_times=[30.0, 31.05], rtol=1e-3, atol=1e-5
    )
    assert float(result.species[-1][1]) == pytest.approx(10.5, rel=1e-6)
