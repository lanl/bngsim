"""An event on ``time == c``, where an event writes c, fires on every grid
(issue #1519).

An event-written parameter is held in a species slot, so it is no run
constant, and the stop that makes ``time == c`` fire (issue #1015's, for a
constant c) was not placed: the resolver declined the equality, and the
per-stretch reading of #1244 left event triggers out. No root finds an
instant either, so the event fired only where an output sample, or a point
of the SSA's trigger-probe lattice, landed on it. A dose counter
``at (time == c && c < 91): c = c + 1`` from c = 0.5 gave all 10 doses over
(0, 10) at ``n_points=101`` and none at 100 or 102, with nothing said;
BIOMD0000000620 (Palmer2014) is that model.

The instants are now placed stretch by stretch between the executions of the
events that write c, the trigger's own included: where those events fire at
a fixed time or on such an instant, and set values known before the run.
Where they do not, the run is refused by name.

Oracle: the count of executions, from the trigger's meaning. ``time == c``
turns true at t = c, so the counter's instants are 0.5, 1.5, ..., 9.5, and
the count does not depend on the output grid or the engine.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_SPAN = (0.0, 10.0)
# No sample on any instant; the grids either side of the one with a sample on
# every half-integer, which was main's only right answer; an uneven one.
_GRIDS = {
    "two_points": {"t_span": _SPAN, "n_points": 2},
    "n_points_100": {"t_span": _SPAN, "n_points": 100},
    "n_points_102": {"t_span": _SPAN, "n_points": 102},
    "uneven": {"sample_times": [0.0, 1.9, 3.3, 7.04, 10.0]},
}
# The SSA probes the triggers on a lattice of a thousandth of the run, which
# over (0, 10) lands on every half-integer; over (0, 10.3) on none.
_SSA_SPAN = (0.0, 10.3)
_SSA_GRIDS = {
    "two_points": {"t_span": _SSA_SPAN, "n_points": 2},
    "n_points_100": {"t_span": _SSA_SPAN, "n_points": 100},
    "uneven": {"sample_times": [0.0, 1.9, 3.3, 7.04, 10.3]},
}

_COUNTER = "E: at (time == c && c < 91): n = n + 1, c = c + 1;"

# (model body, executions of E by t = 10)
_CASES = {
    "the_issue_s_counter": ("var c = 0.5; " + _COUNTER, 10),
    # c is written only after its one instant, by an event at a fixed time.
    "written_after_its_instant": (
        "var c = 0.5037; F: at (time >= 5): c = 0.25; E: at (time == c): n = n + 1;",
        1,
    ),
    # Read through an assignment rule.
    "through_an_assignment_rule": (
        "var c = 0.5; b := c; E: at (time == b && c < 91): n = n + 1, c = c + 1;",
        10,
    ),
    "through_a_scaled_rule": (
        "var c = 0.25; b := 2*c; E: at (time == b): n = n + 1, c = c + 0.5;",
        10,
    ),
    # A species held by a rate rule of 0 is no more integrated than a parameter.
    "a_species_held_by_c_prime_zero": ("species c = 0.5; c' = 0; " + _COUNTER, 10),
    # Fired at 0.6, 1.6, ...: the instant is the trigger's, not the execution's.
    "delayed": (
        "var c = 0.5; E: at 0.1 after (time == c && c < 91): n = n + 1, c = c + 1;",
        10,
    ),
    # Delayed past the next instant: the trigger at 0.5 sets c = 1.5 at 2.5,
    # after 1.5, so the equality never holds again.
    "delayed_past_the_next_instant": (
        "var c = 0.5; E: at 2 after (time == c): n = n + 1, c = c + 1;",
        1,
    ),
    # The next dose read from the clock at the instant.
    "next_dose_from_the_clock": (
        "var c = 0.5; E: at (time == c): n = n + 1, c = time + 1;",
        10,
    ),
    "next_dose_from_the_clock_frozen_at_the_trigger": (
        "var c = 0.5; E: at 0.1 after (time == c): n = n + 1, c = time + 1;",
        10,
    ),
    # c + 0.1 rounds at every step; each instant is the double the sum makes.
    "a_step_that_rounds": (
        "var c = 0.5037; E: at (time == c): n = n + 1, c = c + 0.1;",
        95,
    ),
    # The guard reads a second parameter, which an event sets at 5.2.
    "a_guard_on_another_written_parameter": (
        "var c = 0.5; var d = 0; F: at (time >= 5.2): d = 5; "
        "E: at (time == c && d < 3): n = n + 1, c = c + 1;",
        5,
    ),
    # True from the start, false at c alone, and true again: one rise each time
    # c is passed, at 0.5 and at 7.
    "not_equal": (
        "var c = 0.5; E: at (time != c): n = n + 1; F: at (time >= 5): c = 7;",
        2,
    ),
    # Fires at the start too: at 0, 1, ..., 10.
    "from_the_start": (
        "var c = 0; E: at (time == c), t0=false: n = n + 1, c = c + 1;",
        11,
    ),
    # A reaction beside it: the SSA's probe windows start at random times.
    "beside_a_reaction": ("species X = 0; J: -> X; 5; var c = 0.5; " + _COUNTER, 10),
    # A rate law that reads c, stiffly: c stays the double the event set.
    "read_by_a_stiff_rate_law": (
        "species X = 1; J: X -> ; 1000*c*X + 0.5*X; var c = 0.5; " + _COUNTER,
        10,
    ),
}


def _count(body: str, method: str = "ode", **grid) -> float:
    model = bngsim.Model.from_antimony_string(f"model m\n species n = 0\n {body}\nend")
    kw = {"seed": 1} if method == "ssa" else {}
    r = bngsim.Simulator(model, method=method).run(**grid, **kw)
    return float(np.asarray(r.species)[-1, list(r.species_names).index("n")])


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize(("body", "fires"), _CASES.values(), ids=_CASES.keys())
def test_it_fires_on_every_grid(body, fires, grid):
    """Main fired none of these off the grid that lands on the instants."""
    assert _count(body, **_GRIDS[grid]) == fires


@pytest.mark.parametrize(("body", "fires"), _CASES.values(), ids=_CASES.keys())
def test_control_a_grid_on_the_instants(body, fires):
    """Control: a sample on every half-integer fires each instant on main
    too, and the count is the same as off it."""
    assert _count(body, t_span=_SPAN, n_points=101) == fires


@pytest.mark.parametrize("grid", _SSA_GRIDS)
@pytest.mark.parametrize("name", _CASES)
def test_a_stochastic_run_fires_it_too(name, grid):
    """The SSA's breakpoints come from the same resolver. Off its probe
    lattice main gave none of these."""
    body, fires = _CASES[name]
    # c = 0.5037 + 0.1k up to 10.3 rather than 10.
    fires = 98 if name == "a_step_that_rounds" else fires
    assert _count(body, method="ssa", **_SSA_GRIDS[grid]) == fires


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_a_span_off_the_probe_lattice(method):
    """Over (0, 10.3) main gave 0 of 10 in both engines."""
    assert _count("var c = 0.5; " + _COUNTER, method, t_span=(0.0, 10.3), n_points=2) == 10


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_control_a_constant_side(method):
    """Control: with c a constant, main fired it on every grid already."""
    body = "c = 0.5037; E: at (time == c): n = n + 1;"
    assert _count(body, method, t_span=_SPAN, n_points=100) == 1


@pytest.mark.parametrize("legs", [[(0.0, 4.75), (4.75, 10.0)], [(0.0, 5.5), (5.5, 10.0)]])
def test_a_continued_run_places_the_rest(legs):
    """A run that continues the last one reads c as the last one left it."""
    model = bngsim.Model.from_antimony_string(
        f"model m\n species n = 0\n var c = 0.5; {_COUNTER}\nend"
    )
    sim = bngsim.Simulator(model, method="ode")
    for lo, hi in legs:
        r = sim.run(t_span=(lo, hi), n_points=2)
    assert float(np.asarray(r.species)[-1, 0]) == 10.0
    assert model.get_concentration("c") == 10.5


@pytest.mark.parametrize(("n_points", "rtol"), [(100, 1e-6), (102, 1e-6), (100, 1e-10)])
def test_biomd0000000620_gives_every_dose(n_points, rtol):
    """Palmer2014's daily anakinra dose, ``time == counter && counter < 91``
    from 0.5: main gave Anakinrasc(10) = 0 and the counter 0.5 off the grid
    of 101, at rtol 1e-10 too. Its 14222.31 there is the oracle."""
    path = Path(__file__).resolve().parents[2] / "benchmarks/sbml_events/BIOMD0000000620.xml"

    def at_ten(n, tol):
        r = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode").run(
            t_span=(0, 10), n_points=n, rtol=tol, atol=1e-9
        )
        x = np.asarray(r.species, float)[-1, list(r.species_names).index("Anakinrasc")]
        names = list(r.observable_names)
        counter = np.asarray(r.observables, float)[-1, names.index("Anakinra_dose_counter")]
        return x, counter

    oracle, _ = at_ten(101, rtol)
    dose, counter = at_ten(n_points, rtol)
    assert counter == 10.5
    assert dose == pytest.approx(oracle, rel=1e-5)
    assert oracle == pytest.approx(14222.3, rel=1e-4)


# ── Where the instants cannot be placed: refused by name ────────────────────


@pytest.mark.parametrize("method", ["ode", "ssa"])
@pytest.mark.parametrize(
    ("body", "why"),
    [
        (
            "species X = 0; X' = 1; var c = 100; F: at (X > 2): c = time + 1; "
            "E: at (time == c): n = n + 1;",
            "event F on X>2 sets it, a trigger that is no fixed time",
        ),
        (
            "species X = 1; X' = 1; var c = 0.5; E: at (time == c): n = n + 1, c = c + X;",
            "sets it to (c+X), which is not known before the run",
        ),
        (
            "var c = 0.5; E: at (time == c): n = n + 1, c = c + 1; F: at (time >= 2.5): c = 4.5;",
            "with another event at t=2.5, in an order not known before the run",
        ),
        (
            "var c = 0.5; E: at 0.1 after (time == c), persistent=false: n = n + 1, c = c + 1;",
            "after a delay, on an instant, and it is not persistent",
        ),
        # E sets d to the instant it fires on, which makes G's trigger rise
        # on that same instant.
        (
            "var c = 0.5; var d = 100; E: at (time == c): n = n + 1, d = time; "
            "G: at (time == d): c = c + 1;",
            "which makes an equality hold on its own instant",
        ),
    ],
    ids=[
        "a-state-trigger",
        "a-value-of-the-state",
        "two-writers-at-one-instant",
        "not-persistent",
        "a-cascade-on-the-instant",
    ],
)
def test_a_schedule_not_known_before_the_run_is_refused(body, why, method):
    """Main fired none of these off the instants, with nothing said."""
    with pytest.raises(bngsim.SimulationError, match="issue #1519") as e:
        _count(body, method, t_span=_SPAN, n_points=2)
    assert why in str(e.value)
    assert "and only events set c" in str(e.value)


def test_past_the_stretch_budget_it_is_refused_and_legs_are_not(monkeypatch):
    """More executions than the run places stretches for is refused, with the
    way out, and the same run in legs is not."""
    from bngsim import _switch_sensitivity

    monkeypatch.setattr(_switch_sensitivity, "_HELD_STRETCHES_MOST", 6)
    body = "var c = 0.5; " + _COUNTER
    with pytest.raises(bngsim.SimulationError, match="more than 6 times in the run") as e:
        _count(body, t_span=_SPAN, n_points=2)
    assert "shorter legs" in str(e.value)
    model = bngsim.Model.from_antimony_string(f"model m\n species n = 0\n {body}\nend")
    sim = bngsim.Simulator(model, method="ode")
    for lo, hi in ((0.0, 3.0), (3.0, 6.0), (6.0, 10.0)):
        r = sim.run(t_span=(lo, hi), n_points=2)
    assert float(np.asarray(r.species)[-1, 0]) == 10.0


def test_an_instant_no_double_meets_is_refused_as_for_a_constant():
    """``time - 1 == c`` with c set to 1.3 at 0.5: 2.3 - 1 is
    1.2999999999999998 and the next double gives 1.3000000000000003, so the
    event cannot fire, as for ``time - 1 == 1.3`` (issue #1015). Main ran it,
    and it fired no time, with nothing said."""
    body = "var c = 0; F: at (time >= 0.5): c = 1.3; E: at (time - 1 == c): n = n + 1;"
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        _count(body, t_span=_SPAN, n_points=2)
