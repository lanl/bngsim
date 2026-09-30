"""Issue #733 — a unit-rate clock is decided from structure, not two RHS probes.

A species is a clock (dc/dt = 1 for the whole run, so a threshold on it is a
threshold on time) only when no event assigns it, no reaction consumes it, and
every reaction producing it is zeroth-order with a constant rate law. Two RHS
probes agreeing on 1 also admitted a rate gated in time, an event-reset SBML
timer, and A <-> B with kf = kr. Their thresholds were then jumped once, at a
crossing time computed as though c = c0 + t, and never rooted.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim import _switch_sensitivity as sw

AB = """\
begin parameters
    1 kf   1
    2 kr   1
    3 thr  0.4
    4 kin  2
    5 kd   0.3
end parameters
begin functions
    1 pulse() if(Bo>=thr,kin,0)
end functions
begin species
    1 A() 1
    2 B() 0
    3 Y() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
    3 0 3 pulse
    4 3 0 kd
end reactions
begin groups
    1 Bo                   2
end groups
"""

COUNTER = """\
begin parameters
    1 one  1
    2 Toff 10
end parameters
begin functions
    1 gated() if(time()<Toff,1,0)
end functions
begin species
    1 counter() 0
    2 fake() 0
end species
begin reactions
    1 0 1 one
    2 0 2 gated
end reactions
begin groups
    1 tc                   1
    2 tf                   2
end groups
"""


def _net(tmp_path, text, name):
    p = tmp_path / name
    p.write_text(text)
    return bngsim.Model.from_net(p)


def test_a_real_counter_is_still_a_clock_and_the_impostors_are_not(tmp_path):
    clocks = sw._unit_rate_clock_species(_net(tmp_path, COUNTER, "c.net")._core)
    assert clocks.get("counter()") == 0 and clocks.get("tc") == 0
    assert "fake()" not in clocks and "tf" not in clocks  # gated in time
    assert sw._unit_rate_clock_species(_net(tmp_path, AB, "ab.net")._core) == {}


def test_a_clock_that_catalyses_a_reaction_is_still_a_clock(tmp_path):
    """``Tc() -> Tc() + X()`` reads the clock without changing it: net
    stoichiometry 0, so Tc's slope is still 1 and it stays a clock."""
    text = COUNTER.replace("    2 0 2 gated\n", "    2 0 2 gated\n    3 1 1,2 one\n")
    clocks = sw._unit_rate_clock_species(_net(tmp_path, text, "cat.net")._core)
    assert clocks.get("counter()") == 0


def test_a_counter_driven_by_a_table_function_of_time_is_not_a_clock(tmp_path):
    """The rate reads ``tf()``, a table function indexed by time: 1 up to t = 1,
    then down to 0 by t = 2. Its text names no clock, but it is gated in time."""
    text = """\
begin parameters
    1 thr 3
end parameters
begin functions
    1 tf() tfun([0,1,2,10],[1,1,0,0],time,method=>"linear")
end functions
begin species
    1 C() 0
end species
begin reactions
    1 0 1 tf
end reactions
begin groups
    1 Cnt                  1
end groups
"""
    assert sw._unit_rate_clock_species(_net(tmp_path, text, "tf.net")._core) == {}


def test_a_kinetic_law_that_is_the_whole_flux_can_make_a_clock():
    """``J0: S -> x; 1`` lists S as a reactant, but an SBML kinetic law is the
    whole flux and does not multiply by it: x rises at exactly 1."""
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string("species S = 100; species x = 0; J0: S -> x; 1")
    assert "x" in sw._unit_rate_clock_species(model._core)


def test_an_event_reset_timer_is_not_a_clock():
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string("x' = 1; x = 0; P = 100; E1: at (x >= P): x = 0")
    assert sw._unit_rate_clock_species(model._core) == {}


def test_a_shift_invariant_species_gets_its_real_crossing(tmp_path):
    """B(t) = (1 − e^(−2t))/2 reaches thr = 0.4 at t* = −ln(1 − 2·thr)/2 = 0.8047,
    with dt*/dthr = 1/(1 − 2·thr) = 5, so dY/dthr = −kin·5·e^(−kd·(t − t*)) past
    t* and 0 before it. Read as a clock, the jump landed at t = 0.4, 5x small."""
    t = np.array([0.5, 1.0, 5.0])
    run = bngsim.Simulator(
        _net(tmp_path, AB, "ab2.net"), method="ode", sensitivity_params=["thr"]
    ).run(sample_times=[0.0, *t], rtol=1e-10, atol=1e-12)
    y = list(run.species_names).index("Y()")
    got = np.asarray(run.sensitivities)[1:, y, 0]
    t_star = -math.log(1 - 2 * 0.4) / 2
    want = np.where(t > t_star, -2.0 * 5.0 * np.exp(-0.3 * (t - t_star)), 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-8)


def test_every_window_of_an_event_reset_timer_counts():
    """The timer restarts every P = 100, so the window [a, a + w) opens once per
    period and dA/dw = kin per completed window: 10, 20, 30 at t = 50, 150, 250.
    Read as a clock, only the first window was compensated."""
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "x' = 1; x = 0; P = 100; E1: at (x >= P): x = 0\n"
        "species A; A = 0; kin = 10; a = 30; w = 0.1\n"
        "J0: -> A; piecewise(kin, (x >= a) && (x < a + w), 0)\n"
    )
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["w"]).run(
        sample_times=[0.0, 50.0, 150.0, 250.0], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    got = np.asarray(run.sensitivities)[1:, names.index("A"), 0]
    np.testing.assert_allclose(got, [10.0, 20.0, 30.0], rtol=1e-5)
