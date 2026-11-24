"""A run that took steps too small to move CVODE's clock says so (issue #1837).

Before each step CVODE checks whether ``t + h == t`` in double precision, and
where it is, it takes the step anyway: the state moves by ``h`` and the clock
does not. It printed "t + h = t on the next step" to standard error ten times
and then not again, and the run returned a wrong answer with no error and no
Python warning. Each run now counts them (CVODES' own count, read through
``src/cvodes_roundoff_steps.c``), reports the count in
``solver_stats["n_roundoff_steps"]``, warns :class:`bngsim.RoundoffStepWarning`
naming the first row after one and the spacing of the doubles there, and keeps
the warning as the notice ``solver.roundoff_steps``, on the error of a run
that fails.

The oracles are closed forms, the issue's two examples:

- ``A' = -k A`` with a rate that reads ``time()`` (so the run keeps the
  absolute clock), from 1e15, where the doubles are 0.125 apart:
  ``A(t0 + 5) = exp(-5 k)``;
- the late switch of issue #1836: a counter ``C' = 1`` from 0.5, and
  ``X' = k (1 - X)`` once ``C >= sigma``, with ``sigma = span - 2``, over
  ``span = 2^46``, where the doubles are 2^-6 apart: ``X(span) = 1 - exp(-5)``
  and ``dX(span)/dsigma = -k exp(-5)``.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

FAR = 1e15
#: The spacing of the doubles at 1e15, and at 2^46.
SPACING_FAR = 0.125
SPACING_2_46 = 2.0**-6

DECAY_NET = """begin parameters
 1 k 1
end parameters
begin functions
 1 f() k+0*time()
end functions
begin species
 1 A() 1
 2 B() 0
end species
begin reactions
 1 1 2 f
end reactions
begin groups
 1 Atot 1
end groups
"""

LATE_SWITCH_NET = """begin parameters
    1 c0     0.5
    2 sigma  %r
    3 k      2
    4 rc     1
end parameters
begin functions
    1 rate_X() if(t>=sigma,k*(1-x),0)
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
    2 x 2
end groups
"""


def _decay(tmp_path: Path) -> bngsim.Model:
    path = tmp_path / "decay.net"
    path.write_text(DECAY_NET)
    return bngsim.Model.from_net(str(path))


def _decay_built(event: bool = False) -> bngsim.Model:
    """The same decay, built in memory; with *event*, A := 0 at 1e15 + 6."""
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    a = b.add_species("A", 1.0)
    b.add_function("f", "k + 0*time()")
    b.add_reaction([a], [], "functional", "f")
    b.add_observable("Aobs", [(a, 1.0)])
    if event:
        b.add_event("evt", f"time() >= {FAR!r} + 6", [(a, "0")])
    return bngsim.Model(_core=b.build())


def _roundoff(caught) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, bngsim.RoundoffStepWarning)]


def _run(sim: bngsim.Simulator, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(**kw)
    return result, caught


@pytest.mark.parametrize("cold", [False, True], ids=["default", "cold-path"])
def test_a_decay_far_out_that_reads_the_clock_warns_of_its_steps(tmp_path, monkeypatch, cold):
    """The issue's second example. The run is wrong, as the guide says a far
    run that keeps the absolute clock can be, and now says so: the warning
    names the row and the spacing of the doubles there."""
    if cold:
        # The plain run takes the warm path; this one the cold path's loop.
        monkeypatch.setenv("BNGSIM_NO_WARM_CVODE", "1")
    sim = bngsim.Simulator(_decay(tmp_path), "ode")
    r, caught = _run(sim, sample_times=[FAR, FAR + 5.0, FAR + 10.0])

    a5 = float(np.asarray(r.species)[1, 0])
    # The answer the warning is about: an order of magnitude off exp(-5).
    assert abs(a5 / math.exp(-5.0) - 1.0) > 0.5
    steps = r.solver_stats["n_roundoff_steps"]
    assert steps > 10  # CVODE stops printing after ten

    found = _roundoff(caught)
    assert len(found) == 1
    message = str(found[0].message)
    assert f"CVODE took {steps} steps too small to move its clock" in message
    assert f"by t = {FAR + 5.0!r}" in message and f"after t = {FAR!r}" in message
    assert f"the doubles there are {SPACING_FAR:.3g} apart" in message
    assert f"at most {SPACING_FAR / 2:.3g}" in message
    assert f"the rows from t = {FAR + 5.0!r} on can be wrong" in message
    # Named at the caller's line, not inside bngsim.
    assert found[0].filename == __file__

    notes = [n for n in r.notices if n["code"] == "solver.roundoff_steps"]
    assert notes == [
        {"category": "RoundoffStepWarning", "code": "solver.roundoff_steps", "message": message}
    ]
    assert r.record["notices"][-1]["code"] == "solver.roundoff_steps"


def test_the_same_decay_near_zero_takes_no_such_step(tmp_path):
    """The control: from 0 the run is right, takes no step where t + h == t,
    and says nothing."""
    sim = bngsim.Simulator(_decay(tmp_path), "ode")
    r, caught = _run(sim, sample_times=[0.0, 5.0, 10.0])
    assert float(np.asarray(r.species)[1, 0]) == pytest.approx(math.exp(-5.0), rel=1e-4)
    assert r.solver_stats["n_roundoff_steps"] == 0
    assert _roundoff(caught) == []
    assert not any(n["code"] == "solver.roundoff_steps" for n in r.notices)


def test_a_decay_far_out_that_reads_no_clock_takes_no_such_step():
    """A model that reads no clock is integrated on a clock counted from the
    run's start (issue #1681), so the same decay from 1e15 is right and takes
    no such step."""
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    a = b.add_species("A", 1.0)
    b.add_reaction([a], [], "elementary", "k")
    sim = bngsim.Simulator(bngsim.Model(_core=b.build()), "ode")
    r, caught = _run(sim, sample_times=[FAR, FAR + 5.0, FAR + 10.0])
    assert float(np.asarray(r.species)[1, 0]) == pytest.approx(math.exp(-5.0), rel=1e-4)
    assert r.solver_stats["n_roundoff_steps"] == 0
    assert _roundoff(caught) == []


def test_a_late_switch_over_2_46_warns_of_its_steps(tmp_path):
    """The issue's first example, with forward sensitivities: dX/dsigma comes
    back a third of -k exp(-5), and the run now says why."""
    span = 2.0**46
    path = tmp_path / "late.net"
    path.write_text(LATE_SWITCH_NET % (span - 2.0,))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), "ode", sensitivity_params=["sigma"])
    r, caught = _run(sim, sample_times=[0.0, span], rtol=1e-10, atol=1e-12)

    s = float(np.asarray(r.sensitivities)[-1, 1, 0])
    assert abs(s / (-2.0 * math.exp(-5.0)) - 1.0) > 0.5
    steps = r.solver_stats["n_roundoff_steps"]
    assert steps > 10
    found = _roundoff(caught)
    assert len(found) == 1
    message = str(found[0].message)
    assert f"CVODE took {steps} steps" in message
    assert f"by t = {span!r}" in message
    assert f"the doubles there are {SPACING_2_46:.3g} apart" in message
    assert any(n["code"] == "solver.roundoff_steps" for n in r.notices)


def test_the_steps_before_an_event_are_counted_after_it():
    """A restart at an event zeroes CVODE's own count; the run's total keeps
    the steps of the segment before it. After A := 0 at 1e15 + 6 the run takes
    no such step, so every one counted is from before the event."""
    plain, _ = _run(
        bngsim.Simulator(_decay_built(), "ode"), sample_times=[FAR, FAR + 5.0, FAR + 10.0]
    )
    r, caught = _run(
        bngsim.Simulator(_decay_built(event=True), "ode"),
        sample_times=[FAR, FAR + 5.0, FAR + 10.0],
    )
    assert float(np.asarray(r.species)[-1, 0]) == 0.0  # the event fired
    assert r.solver_stats["n_roundoff_steps"] == plain.solver_stats["n_roundoff_steps"] > 10
    assert len(_roundoff(caught)) == 1


def test_a_run_that_fails_keeps_the_notice_on_its_error(tmp_path):
    """A run that raises has no Result; the steps it took go on its error
    (issue #1857), and no warning is raised beside the error."""
    sim = bngsim.Simulator(_decay(tmp_path), "ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(bngsim.SimulationError) as info:
            sim.run(sample_times=[FAR, FAR + 5.0, FAR + 10.0], max_steps=5)
    notes = [n for n in info.value.notices if n["code"] == "solver.roundoff_steps"]
    assert len(notes) == 1
    assert notes[0]["category"] == "RoundoffStepWarning"
    assert "too small to move its clock" in notes[0]["message"]
    assert "before the run failed" in notes[0]["message"]
    assert _roundoff(caught) == []
    assert any(n["code"] == "solver.roundoff_steps" for n in info.value.to_dict()["notices"])


def test_a_batch_puts_each_rows_notice_on_its_own_result(tmp_path):
    """A row that took such steps says so on its own Result; a row that took
    none says nothing. k = 1e-9 barely moves A, so CVODE's steps are long."""
    sim = bngsim.Simulator(_decay(tmp_path), "ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = sim.run_batch(
            t_span=(FAR, FAR + 10.0), n_points=3, params=[{"k": 1.0}, {"k": 1e-9}]
        )
    taken = [r.solver_stats["n_roundoff_steps"] for r in rows]
    assert taken[0] > 10 and taken[1] == 0
    codes = [[n["code"] for n in r.notices] for r in rows]
    assert "solver.roundoff_steps" in codes[0]
    assert "solver.roundoff_steps" not in codes[1]
    assert len(_roundoff(caught)) >= 1


def test_a_promoted_warning_is_the_runs_error(tmp_path):
    """Under ``-W error`` the run raises it, as for any warning."""
    sim = bngsim.Simulator(_decay(tmp_path), "ode")
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.RoundoffStepWarning)
        with pytest.raises(bngsim.RoundoffStepWarning, match="too small to move its clock"):
            sim.run(sample_times=[FAR, FAR + 5.0, FAR + 10.0])


def test_the_count_is_zero_on_a_stochastic_run():
    """The key is on every Result, as the other counters are."""
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    a = b.add_species("A", 100.0)
    b.add_reaction([a], [], "elementary", "k")
    r = bngsim.Simulator(bngsim.Model(_core=b.build()), "ssa").run(
        t_span=(0, 1), n_points=3, seed=1
    )
    assert r.solver_stats["n_roundoff_steps"] == 0
