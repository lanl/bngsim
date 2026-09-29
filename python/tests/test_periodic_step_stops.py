"""Issue #869: a periodic dose written with floor() or mod() of time() is not
stepped over.

A `.net` pulse train has a zero rate law between pulses, so CVODE's step grows
past a whole pulse unless something stops it at the edges. Only one spelling,
``(time()-t0) - P*floor((time()-t0)/P) < w``, was recognized as a schedule and
stopped at; the difference of two floors (bare or compared) and ``mod()`` were
not, and each returned X = 0 where 11 pulses were given. Every edge of a
floor/ceil/rint/mod of time is now placed as a crossing stop.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import fixed_crossing_stops, fixed_time_crossings

_NET = """begin parameters
    1 P {P}
    2 w {w}
    3 t0 {t0}
end parameters
begin species
    1 X() 0
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
begin groups
    1 Xtot 1
end groups
"""

# The issue's table: every spelling is a unit pulse on [t0 + nP, t0 + nP + w).
DOSES = {
    "floor_difference": "floor((time()-t0)/P) - floor((time()-t0-w)/P)",
    "floor_difference_compared": "if(floor((time()-t0)/P) - floor((time()-t0-w)/P) > 0.5, 1, 0)",
    "mod_compared": "if(mod(time()-t0+10*P, P) < w, 1, 0)",
    "ceil_difference": "ceil((time()-t0)/P) - ceil((time()-t0-w)/P)",
    "remainder_compared": "if((time()-t0) - P*floor((time()-t0)/P) < w, 1, 0)",
}

# (P, w, t0): daily, daily from t = 0, weekly, and fast.
SCHEDULES = [(24, 1, 5), (24, 1, 0), (168, 1, 5), (10, 0.5, 5)]
T_END = 252.0


def _model(tmp_path, dose: str, P: float, w: float, t0: float) -> bngsim.Model:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(P=P, w=w, t0=t0, dose=dose), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _pulses_begun(P: float, t0: float) -> int:
    return math.floor((T_END - t0) / P) + 1


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
@pytest.mark.parametrize("P, w, t0", SCHEDULES)
@pytest.mark.parametrize("dose", DOSES.values(), ids=DOSES.keys())
def test_every_pulse_is_integrated(tmp_path, dose, P, w, t0, codegen):
    sim = bngsim.Simulator(_model(tmp_path, dose, P, w, t0), method="ode", codegen=codegen)
    if codegen:
        assert sim.codegen_backend in ("cc", "mir")
    r = sim.run(t_span=(0.0, T_END), n_points=int(T_END) + 1)
    x_end = float(np.asarray(r.species)[-1, 0])
    assert x_end == pytest.approx(w * _pulses_begun(P, t0), rel=1e-6)


def test_mod_of_a_negative_argument_keeps_fmods_sign(tmp_path):
    # fmod(time()-5, 24) is negative, so below w, for the whole of [0, 5): a
    # sixth hour of dose before the first pulse, which the stops must also see.
    m = _model(tmp_path, "if(mod(time()-t0, P) < w, 1, 0)", 24, 1, 5)
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, T_END), n_points=253)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(5 + 11, rel=1e-6)


@pytest.mark.parametrize("dose", DOSES.values(), ids=DOSES.keys())
def test_the_stops_are_the_pulse_edges(tmp_path, dose):
    m = _model(tmp_path, dose, 24, 1, 5)
    stops = fixed_time_crossings(m._core, 0.0, 60.0, m.time_discontinuity_conditions())
    assert stops == pytest.approx([5.0, 6.0, 29.0, 30.0, 53.0, 54.0], abs=1e-9)


def test_a_condition_that_never_changes_places_no_stop(tmp_path):
    # `rem >= 0` holds for every t. Its floor jumps once a period, but the
    # truth does not, and a stop where nothing changes only perturbs stepping.
    m = _model(tmp_path, "if(time() - P*floor(time()/P) >= 0, 1, 0)", 24, 1, 5)
    assert fixed_time_crossings(m._core, 0.0, 252.0, m.time_discontinuity_conditions()) == []


def test_a_step_of_state_is_not_a_stop(tmp_path):
    # floor(X) jumps at a time no one knows before the run. (+2, not +1: a
    # species fed at rate 1 would read as a counter clock, issue #443.)
    m = _model(tmp_path, "floor(Xtot/2) + 2", 24, 1, 5)
    assert m.time_discontinuity_conditions() == ()


# A BNGL model reads time through a counter species far more often than through
# time() (issue #443). C starts at 3, so the counter reads t + 3 and every edge
# comes 3 earlier than it does on time().
_COUNTER_NET = """begin parameters
    1 P {P}
    2 w {w}
    3 t0 {t0}
    4 kc 1
end parameters
begin species
    1 X() 0
    2 C() 3
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
    2 0 2 kc #_R2
end reactions
begin groups
    1 Xtot 1
    2 tc 2
end groups
"""


@pytest.mark.parametrize("dose", DOSES.values(), ids=DOSES.keys())
def test_a_pulse_on_a_counter_clock_is_stopped_at(tmp_path, dose):
    path = tmp_path / "c.net"
    text = _COUNTER_NET.format(P=24, w=1, t0=5, dose=dose.replace("time()", "tc"))
    path.write_text(text, encoding="utf-8")
    m = bngsim.Model.from_net(str(path))
    stops = fixed_crossing_stops(m._core, 0.0, 60.0, m.time_discontinuity_conditions())
    assert [s.time for s in stops] == pytest.approx([2.0, 3.0, 26.0, 27.0, 50.0, 51.0], abs=1e-9)
    assert {s.clock_species_idx for s in stops} == {1}
    assert [s.threshold for s in stops] == pytest.approx([5.0, 6.0, 29.0, 30.0, 53.0, 54.0])
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, T_END), n_points=int(T_END) + 1)
    # Pulses begin at t = 2 + 24n, the last of them at 242.
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(11.0, rel=1e-6)


@pytest.mark.parametrize("cond", ["mod(time(), P) < P", "floor(time()/P)*P + P - time() > 0"])
def test_a_condition_that_fails_only_at_its_jumps_places_no_stop(tmp_path, cond):
    # Each is true except at the instant of a jump, where its residual is exactly
    # 0. An instant carries no dose, so there is nothing to stop at, and the
    # residual's value AT the jump must not be read as the value just before it.
    m = _model(tmp_path, f"if({cond}, 1, 0)", 24, 1, 5)
    assert fixed_time_crossings(m._core, 0.0, 252.0, m.time_discontinuity_conditions()) == []


def test_two_models_spelling_one_atom_over_different_derived_names(tmp_path):
    # The stop cache outlives a model. `onset` is t0 + 1 in one model and t0 * 5
    # in the other, and both read t0 = 2, so the atom as written and the leaf
    # values agree, and only the derived definition tells their edges apart.
    def model(name: str, onset: str) -> bngsim.Model:
        path = tmp_path / f"{name}.net"
        path.write_text(
            "begin parameters\n    1 t0 2\n    2 onset " + onset + "\nend parameters\n"
            "begin species\n    1 X() 0\nend species\n"
            "begin functions\n    1 dose() floor(time()/onset)\nend functions\n"
            "begin reactions\n    1 0 1 dose #_R1\nend reactions\n",
            encoding="utf-8",
        )
        return bngsim.Model.from_net(str(path))

    a, b = model("a", "t0+1"), model("b", "t0*5")
    assert fixed_time_crossings(a._core, 0.0, 35.0, a.time_discontinuity_conditions()) == (
        pytest.approx([3.0 * n for n in range(1, 12)])
    )
    assert fixed_time_crossings(b._core, 0.0, 35.0, b.time_discontinuity_conditions()) == (
        pytest.approx([10.0, 20.0, 30.0])
    )


# A step inside another step's argument. The circadian light functions of the
# Pokhilko and Adams models (BIOMD0000000412, 445, 476) keep their phase with
# `floor(floor(time()+phase)/cyclePeriod)`, a day counter whose inner floor jumps
# every hour. Each of those hours was a stop, and a CVODE restart, where the rate
# law does not change: 4x the Jacobian evaluations on those models. The stops are
# the outer step's jumps.
_LIGHT = (
    "0.5*(1+tanh(P*(time()/P-floor(floor(time())/P))/0.05))"
    " - 0.5*(1+tanh((P*(time()/P-floor(floor(time())/P))-12)/0.05))"
    " + 0.5*(1+tanh((P*(time()/P-floor(floor(time())/P))-P)/0.05))"
)
NESTED = {
    "day_counter": ("floor(floor(time()+t0)/P)", 100.0, [19.0, 43.0, 67.0, 91.0]),
    "circadian_light": (_LIGHT, 100.0, [24.0, 48.0, 72.0, 96.0]),
    # floor(n/2.5) for n = floor(t) steps up at n = 3, 5, 8, 10, 13, 15.
    "ratio_not_whole": ("floor(floor(time())/2.5)", 16.0, [3.0, 5.0, 8.0, 10.0, 13.0, 15.0]),
    # Between the inner jumps the outer argument t*n/10 still moves: it reaches 1
    # at t = 10/3, inside [3, 4), and jumps from 1.99 to 2.5 at t = 5.
    "outer_argument_moves": ("floor(time()*floor(time())/10)", 6.0, [10.0 / 3.0, 5.0]),
}


@pytest.mark.parametrize("dose, t_end, edges", NESTED.values(), ids=NESTED.keys())
def test_a_step_inside_a_step_stops_where_the_outer_one_jumps(tmp_path, dose, t_end, edges):
    m = _model(tmp_path, dose, 24, 1, 5)
    conditions = m.time_discontinuity_conditions()
    assert len(conditions) == 1  # the outer call, and never the inner one on its own
    stops = fixed_time_crossings(m._core, 0.0, t_end, conditions)
    assert stops == pytest.approx(edges, abs=1e-9)


def test_a_pulse_through_a_remainder_of_a_remainder_is_integrated(tmp_path):
    # The resolver read no step inside a step, so this pulse had no stops and
    # CVODE stepped over every one after the first: X = 1 where 5 were given.
    m = _model(tmp_path, "if(mod(mod(time(), P), 12) < w, 1, 0)", 24, 1, 5)
    stops = fixed_time_crossings(m._core, 0.0, 60.0, m.time_discontinuity_conditions())
    assert stops == pytest.approx([1.0, 12.0, 13.0, 24.0, 25.0, 36.0, 37.0, 48.0, 49.0])
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, 60.0), n_points=61)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(5.0, rel=1e-6)


@pytest.mark.parametrize(
    "dose",
    [
        # A step this cannot read the jumps of, around one it can.
        "round(floor(time())/3)",
        # A step of state around a step of time. (+2: see test_a_step_of_state...)
        "floor(floor(time()) + Xtot/100) + 2",
    ],
)
def test_an_outer_step_that_cannot_be_read_keeps_the_inner_ones_stops(tmp_path, dose):
    # Its jumps are among the inner step's, so those are still stopped at: a
    # restart where nothing changes, but never a jump stepped over.
    m = _model(tmp_path, dose, 24, 1, 5)
    stops = fixed_time_crossings(m._core, 0.0, 6.0, m.time_discontinuity_conditions())
    assert stops == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
