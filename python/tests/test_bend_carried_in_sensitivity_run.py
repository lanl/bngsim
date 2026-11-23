"""A sensitivity run is carried over a bend it stalls at, not refused (issue #1229).

The issue #952 hold check ends a run with sensitivities whose state is held
within a few ulp of a state-switch surface, because at a jump there is a
saltation term to apply and no crossing time to take it from. It asked that
of every surface, and a bend has neither: where a triangle pulse returns to 0
the rate law is continuous, and only its slope changes, by ~1/w². At a tight
tolerance that slope change fails the error test on the step to it, and the
run was refused for a jump that does not exist. The same pulse written with
``if()`` was refused the same way.

A stalled surface is now read a step either side along the flow (the #150
probe), and one the right-hand side is continuous across, or that no rate law
reads (a bend stop), is carried over as a plain run's is, the columns
unchanged. A stalled surface it jumps across is refused as before; see
``test_pinned_state_switch_sensitivity.py``. So is a carry that would take
something else across with it, a jump spelled apart from the pulse or an
event's trigger on the same surface: neither has a crossing time there.

The model is ``test_bend_sensitivity_rhs.py``'s: a counter T at k_clock = 2, and
X -> Y at a triangle of area 0.5 in T, of width w, centred on 50. Exact
dY/dk_clock = -0.973501 and Y = 10(1 - e^(-1/4)).
"""

from __future__ import annotations

import logging
import math

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

EXACT = -10 * 0.25 * math.exp(-0.25) / 2  # -0.973501
Y_EXACT = 10 * (1 - math.exp(-0.25))
LAWS = {
    "max/abs": "kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))",
    "if()": (
        "kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c,(Tobs-c+w/2)/(w/2),"
        "if(Tobs<c+w/2,(c+w/2-Tobs)/(w/2),0)))"
    ),
}


def _run(tmp_path, law, w, rtol, beside=None):
    """Y and dY/dk_clock at t = 30; *beside* is the rate law of a second
    reaction, Z -> W from Z = 10."""
    funcs, species, rxns = f" 1 f() {law}\n", "", ""
    if beside is not None:
        funcs += f" 2 g() {beside}\n"
        species = " 4 Z() 10\n 5 W() 0\n"
        rxns = " 3 4 5 g #_FUNC\n"
    path = tmp_path / "m.net"
    path.write_text(
        f"begin parameters\n 1 kx 1\n 2 k_clock 2\n 3 c 50\n 4 w {w!r}\n 5 kz 0.1\n"
        "end parameters\n"
        f"begin species\n 1 T() 0\n 2 X() 10\n 3 Y() 0\n{species}end species\n"
        f"begin functions\n{funcs}end functions\n"
        f"begin reactions\n 1 0 1 k_clock\n 2 2 3 f #_FUNC\n{rxns}end reactions\n"
        "begin groups\n 1 Tobs 1\nend groups\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["kx", "k_clock"]
    )
    run = sim.run(t_span=(0, 30), n_points=3, rtol=rtol, atol=rtol / 100)
    return float(np.asarray(run.species)[-1, 2]), float(np.asarray(run.sensitivities)[-1, 2, 1])


# The cells the issue lists as refused on main.
@pytest.mark.parametrize("spelling", list(LAWS))
@pytest.mark.parametrize(
    ("w", "rtol", "rel"),
    [(0.01, 1e-10, 2e-5), (0.01, 1e-11, 1e-5), (0.03, 1e-11, 1e-6)],
)
def test_a_run_that_stalls_at_a_bend_returns_the_exact_column(tmp_path, spelling, w, rtol, rel):
    y, dy = _run(tmp_path, LAWS[spelling], w, rtol)
    assert y == pytest.approx(Y_EXACT, rel=1e-8)
    assert dy == pytest.approx(EXACT, rel=rel)


@pytest.mark.parametrize("spelling", list(LAWS))
@pytest.mark.parametrize("rtol", [1e-9, 1e-10, 1e-11])
def test_the_narrowest_pulse_runs_too(tmp_path, spelling, rtol):
    """w = 0.003 was refused at every tolerance. Its column is now within what
    the analytic path gives a pulse that narrow where main runs it (9e-5 at w =
    0.005, rtol 1e-9; 9e-4 at w = 0.003, rtol 1e-8), not to the tolerance."""
    y, dy = _run(tmp_path, LAWS[spelling], 0.003, rtol)
    assert y == pytest.approx(Y_EXACT, rel=1e-8)
    assert dy == pytest.approx(EXACT, rel=5e-4)


def test_a_jump_on_the_same_surface_is_refused_not_carried(tmp_path):
    """Z -> W switches on where the pulse returns to 0, written
    ``Tobs-c >= w/2`` beside the pulse's ``Tobs < c+w/2``: another switch on
    the same surface, read by another reaction, so the pulse's own probe did
    not see its jump. Carried over with the bend, it was crossed with no root
    and no saltation, and dW/dk_clock came back 0 for 7.58 with no error."""
    with pytest.raises(
        bngsim.SimulationError,
        match="no crossing time for the sensitivities to move with.*A plain run is carried across",
    ):
        _run(tmp_path, LAWS["if()"], 0.01, 1e-11, beside="kz*if(Tobs-c>=w/2,1,0)")


def _beside_a_decay(law, w, trigger=None, **kw):
    """The pulse model with Z -> 0 at kd = 0.1 from Z = 10 beside it, and an
    event adding 5 to Z on *trigger* where one is given."""
    b = ModelBuilder()
    for name, value in (("kx", 1.0), ("k_clock", 2.0), ("c", 50.0), ("w", w), ("kd", 0.1)):
        b.add_parameter(name, value)
    t, x, y, z = (b.add_species(s, v) for s, v in (("T", 0), ("X", 10), ("Y", 0), ("Z", 10)))
    b.add_observable("Tobs", [(t, 1.0)])
    b.add_reaction([], [t], "elementary", "k_clock")
    b.add_function("f", law)
    b.add_reaction([x], [y], "functional", "f")
    b.add_reaction([z], [], "elementary", "kd")
    if trigger is not None:
        b.add_event("dose", trigger, [(z, "Z + 5")])
    return bngsim.Simulator(
        bngsim.Model(b.build()), method="ode", sensitivity_params=["kx", "k_clock"], **kw
    )


@pytest.mark.parametrize("spelling", list(LAWS))
def test_an_event_on_the_same_surface_is_refused_not_carried(spelling):
    """An event adds 5 to a decaying Z where the pulse returns to 0. Carried
    over with the bend, its trigger turned with no root: the event did not
    fire until the run's last output, and Z came back 5.4979 for 3.5313."""
    sim = _beside_a_decay(LAWS[spelling], 0.01, trigger="Tobs > c + w/2")
    with pytest.raises(bngsim.SimulationError, match="event 'dose'"):
        sim.run(t_span=(0, 30), n_points=3, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize(
    ("spelling", "w", "jacobian", "rel"),
    [("max/abs", 0.01, "auto", 2e-5), ("if()", 0.003, "fd", 5e-4)],
)
def test_a_stall_the_time_cannot_move_out_of_is_carried(caplog, spelling, w, jacobian, rel):
    """Beside the decaying Z, the step at the pulse's foot fell under half an
    ulp of t = 25.0025 (h = 1.1e-15), so the time the hold check waits to see
    pass never did, and the state was neither carried nor refused (issue
    #1283). The batch ran out on the no-progress guard: with the analytic
    Jacobian after 20,000 steps and a GH#176 retry on the difference quotient,
    and with jacobian="fd", where there is nothing to retry with, as a failed
    run. The foot is continuous, and is carried."""
    sim = _beside_a_decay(LAWS[spelling], w, jacobian=jacobian)
    with caplog.at_level(logging.WARNING):
        run = sim.run(t_span=(0, 30), n_points=3, rtol=1e-10, atol=1e-12)
    assert not [r for r in caplog.records if "no progress" in r.getMessage()]
    assert float(np.asarray(run.species)[-1, 2]) == pytest.approx(Y_EXACT, rel=1e-8)
    assert float(np.asarray(run.sensitivities)[-1, 2, 1]) == pytest.approx(EXACT, rel=rel)
