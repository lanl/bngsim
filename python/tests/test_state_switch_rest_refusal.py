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
"""

from __future__ import annotations

import math
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
    """Asked at each output, over the stretch since the one before and the rest
    of the run: a run asked only for t_end has the whole of it to speak for."""
    sim = _sim(tmp_path, JUMP, 10.0)
    with pytest.raises(bngsim.SimulationError, match=r"At t=60 .*issue #1548"):
        sim.run(t_span=(0.0, 60.0), n_points=2, rtol=1e-8, atol=1e-8)


def test_a_sensitivity_run_is_refused_the_same_way(tmp_path):
    sim = _sim(tmp_path, JUMP, 10.0, sensitivity_params=["kb", "thr"])
    with pytest.raises(bngsim.SimulationError, match=REFUSED) as info:
        sim.run(t_span=(0.0, 60.0), n_points=61, rtol=1e-8, atol=1e-10)
    assert "#926" not in str(info.value)


@pytest.mark.parametrize("a0", [10.0, 0.0], ids=["from-above", "from-below"])
def test_a_threshold_the_run_can_tell_from_the_rest_runs(tmp_path, a0):
    """thr = 2 + 1e-4, far outside the band where A rests: from below A never
    reaches it, and from above it crosses it once, at t* = ln((A0 − 2)/1e-4).
    A crosses at 1e-4 a unit of time there, so an error e in A moves t* by
    e/1e-4: run at 1e-10, where that is 2e-6."""
    thr = 2.0 + 1e-4
    result = _sim(tmp_path, JUMP, a0, thr=thr).run(
        t_span=(0.0, 60.0), n_points=121, rtol=1e-10, atol=1e-10
    )
    if a0 < thr:
        want = 3.0 * 60.0
    else:
        t_star = math.log((a0 - 2.0) / 1e-4)
        want = 0.75 * t_star + 3.0 * (60.0 - t_star)
    assert _w(result) == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("a0", [10.0, 0.0], ids=["from-above", "from-below"])
def test_a_continuous_switch_resting_on_its_surface_runs(tmp_path, a0):
    """The clamp kc·(thr − A)⁺ has no jump at thr, so where A rests there is no
    branch to be wrong about: W = kc·∫(thr − A)⁺ = kc·2·(1 − e^(−T)) from below,
    and 0 from above."""
    result = _sim(tmp_path, CLAMP, a0).run(t_span=(0.0, 60.0), n_points=121, rtol=1e-8, atol=1e-8)
    want = 5.0 * 2.0 * (1.0 - math.exp(-60.0)) if a0 < 2.0 else 0.0
    assert _w(result) == pytest.approx(want, rel=1e-6, abs=1e-6)


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


def test_a_transversal_crossing_runs(tmp_path):
    """A rises at 1 a unit of time through thr = 5, so W switches at t = 5."""
    text = RELAX.format(a0=0.0, thr=5.0, law=JUMP).replace("1 0 1 src\n    2 1 0 k\n", "1 0 1 k\n")
    path = tmp_path / "t.net"
    path.write_text(text.replace("    3 0 2 fW\n", "    2 0 2 fW\n"))
    result = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, 10.0), n_points=21, rtol=1e-8, atol=1e-8
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
