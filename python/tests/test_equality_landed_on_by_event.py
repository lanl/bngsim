"""A state an event puts on a rate-law equality is refused in a sensitivity
run where it leaves the surface by a switch nothing jumps, not differentiated to
0 (issue #1279).

#1228's model with an SBML washout: A = 10·e^(-a·t), B made only once A falls
to q, at t* = ln(10/q)/a, and ``Y' = piecewise(kb, B == 0, 0)``. B starts at 1,
so the run does not start on the surface, and an event at t = 1 sets B = 0.
The equality then holds until t*, so Y = kb·(t* - 1), and the exact columns are
dY/da = -kb·ln(10/q)/a² = -6.437752, dY/dq = -kb/(a·q) = -1 and dY/dkb =
t* - 1 = 2.218876. #1240 read the equality at the start alone, and the run came
back [0, 0, 2.218876] with no warning: leaving the surface is a switch nothing
locates. The solver now roots where the state leaves, wherever it got on, and
refuses there.

An event that leaves the state off the surface runs as before, and so does one
that writes nothing the equality reads, one that leaves it on the surface for
the instant the flow takes it away at a time no column moves, and a stretch
that outlasts the run or only an event ends.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SensitivityUnsupportedError

pytest.importorskip("antimony")

A, Q, KB = 0.5, 2.0, 1.0
PARAMS = ["a", "q", "kb"]


def _model(event: str, also_y: str = "") -> bngsim.Model:
    return bngsim.Model.from_antimony_string(
        f"""
        model m
          A = 10; B = 1; Y = 0
          a = {A}; q = {Q}; kb = {KB}; kp = 1
          R1: A -> ; a*A
          R2: -> B; kp*piecewise(q - A, q - A > 0, 0)
          R3: -> Y; piecewise(kb, B == 0, 0){also_y}
          E1: at (time >= {event}
        end
        """
    )


def _columns(model, t_span=(0, 8)):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=PARAMS)
    result = sim.run(t_span=t_span, n_points=9, rtol=1e-10, atol=1e-12)
    return np.asarray(result.sensitivities)[-1, list(result.species_names).index("Y"), :]


@pytest.mark.parametrize(
    "event", ["1): B = 0", "0), t0=false: B = 0"], ids=["mid-run", "at the start"]
)
def test_a_state_an_event_puts_on_the_equality_is_refused_where_it_leaves(event):
    """On main, mid-run: [0, 0, 2.218876] for [-6.4378, -1, 2.2189] by
    difference. An event that fires at t = 0 does so after the start was read,
    and is the same case: [0, 0, 3.218873] for [-6.4378, -1, 3.2189]."""
    with pytest.raises(
        SensitivityUnsupportedError, match=r"'B==0.*leaves it at t=3\.2188.*issue #1279"
    ):
        _columns(_model(event))


def test_a_state_on_the_equality_for_an_instant_at_a_moving_time_is_refused():
    """The event's jump reads Y's flow f⁺ on the surface, where B leaves it at
    once: main gave dY/dte = -1 for 0 by difference (te = 4, after A < q)."""
    model = bngsim.Model.from_antimony_string(
        f"""
        model m
          A = 10; B = 1; Y = 0
          a = {A}; q = {Q}; kb = {KB}; kp = 1; te = 4
          R1: A -> ; a*A
          R2: -> B; kp*piecewise(q - A, q - A > 0, 0)
          R3: -> Y; piecewise(kb, B == 0, 0) + kb*A
          E1: at (time >= te): B = 0
        end
        """
    )
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["te", "a", "kb"])
    with pytest.raises(SensitivityUnsupportedError, match=r"'B==0.*leaves it at t=4"):
        sim.run(t_span=(0, 8), n_points=9, rtol=1e-10, atol=1e-12)


def test_a_stretch_an_event_starts_that_outlasts_the_run_runs():
    """Over (0, 3), before t*: Y = kb·(3 - 1). Refused by the first fix for this
    issue, which judged at the event by the start's first-order test."""
    np.testing.assert_allclose(
        _columns(_model("1): B = 0"), t_span=(0, 3)), [0.0, 0.0, 2.0], atol=1e-9
    )


def test_a_stretch_only_an_event_ends_runs():
    """Nothing makes B, so B = 0 holds until the event at t = 4 sets it to 1:
    Y = 4·kb. #1240 refused it at the start."""
    model = bngsim.Model.from_antimony_string(
        f"""
        model m
          A = 10; B = 0; Y = 0
          a = {A}; q = {Q}; kb = {KB}
          R1: A -> ; a*A
          R3: -> Y; piecewise(kb, B == 0, 0)
          E1: at (time >= 4): B = 1
        end
        """
    )
    np.testing.assert_allclose(_columns(model), [0.0, 0.0, 4.0], atol=1e-9)


# Y also gains kb·A, so the columns are those of kb·∫A over (0, 8) and not 0.
_ALSO_Y = " + kb*A"
_INT_A = 10 * (1 - math.exp(-8 * A)) / A
_EXACT_ALSO = [
    KB * 10 * (8 * math.exp(-8 * A) / A - (1 - math.exp(-8 * A)) / A**2),
    0.0,
    _INT_A,
]


@pytest.mark.parametrize(
    "event",
    ["1): B = 5", "4): B = 0", "1): A = 10*exp(-a*time)", "1): Y = Y"],
    ids=[
        "off the surface",
        "on it for an instant: A is below q at t = 4, so B is made at once",
        "writes nothing the equality reads",
        "writes the equality's reader",
    ],
)
def test_an_event_that_leaves_the_state_off_the_equality_runs(event):
    np.testing.assert_allclose(_columns(_model(event, _ALSO_Y)), _EXACT_ALSO, rtol=1e-6)


def test_an_equality_only_events_move_runs():
    """A flag no reaction changes is left only at an event, whose own jump is
    right, as in BIOMD0000000301's `pulse_flag == 1`: Y = kb·(t_off - t_on),
    so dY/d[t_on, t_off, kb] = [-kb, kb, t_off - t_on]."""
    model = bngsim.Model.from_antimony_string(
        f"""
        model m
          Y = 0; flag = 0; kb = {KB}; t_on = 1; t_off = 3.5
          R1: -> Y; piecewise(kb, flag == 1, 0)
          E1: at (time >= t_on): flag = 1
          E2: at (time >= t_off): flag = 0
        end
        """
    )
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["t_on", "t_off", "kb"])
    result = sim.run(t_span=(0, 8), n_points=9, rtol=1e-10, atol=1e-12)
    columns = np.asarray(result.sensitivities)[-1, list(result.species_names).index("Y"), :]
    np.testing.assert_allclose(columns, [-KB, KB, 2.5], rtol=1e-6)


def test_an_equality_over_a_boundary_species_only_events_move_runs():
    """The same, with the flag a boundary species a reaction reads: the
    reaction names it but never changes it."""
    model = bngsim.Model.from_antimony_string(
        f"""
        model m
          species $F = 0; Y = 0; kb = {KB}; t_on = 1; t_off = 3.5
          R1: $F -> Y; piecewise(kb, F == 1, 0)
          E1: at (time >= t_on): F = 1
          E2: at (time >= t_off): F = 0
        end
        """
    )
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["t_on", "t_off", "kb"])
    result = sim.run(t_span=(0, 8), n_points=9, rtol=1e-10, atol=1e-12)
    columns = np.asarray(result.sensitivities)[-1, list(result.species_names).index("Y"), :]
    np.testing.assert_allclose(columns, [-KB, KB, 2.5], rtol=1e-6)


def test_a_run_without_sensitivities_is_unaffected():
    result = bngsim.Simulator(_model("1): B = 0"), method="ode").run(
        t_span=(0, 8), n_points=9, rtol=1e-10, atol=1e-12
    )
    t_star = math.log(10 / Q) / A
    y = float(np.asarray(result.species)[-1, list(result.species_names).index("Y")])
    assert y == pytest.approx(KB * (t_star - 1), rel=1e-6)


@pytest.mark.parametrize(
    "entry", ["run_batch", "run_batch in threads", "compute_all_sensitivities"]
)
def test_every_entry_point_refuses_it_typed(entry):
    """A chunk of compute_all_sensitivities wrapped the solver's refusal in a
    SimulationError, which also sent it to the column-by-column retry."""
    sim = bngsim.Simulator(_model("1): B = 0"), method="ode", sensitivity_params=PARAMS)
    with pytest.raises(SensitivityUnsupportedError, match=r"'B==0.*issue #1279"):
        if entry.startswith("run_batch"):
            sim.run_batch(
                t_span=(0, 8),
                n_points=9,
                params=[{"a": A}, {"a": 0.6}],
                num_processors=2 if "threads" in entry else 1,
            )
        else:
            sim.compute_all_sensitivities(
                t_span=(0, 8), n_points=9, params=PARAMS, chunk_size=1, n_workers=1
            )
