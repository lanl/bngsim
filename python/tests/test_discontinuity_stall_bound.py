"""Issue #54 — a collapsed step size at a discontinuity must fail, not hang.

At an ``if(t >= sigma)`` rate jump CVODE drives the step size to ~1e-15 until
``t + h == t`` and returns ``CV_TOO_MUCH_WORK``. That return is ordinarily
recoverable — ``max_steps`` is a batch size per output point, not a ceiling on
the run, so the integrator's state is intact and calling ``CVode`` again just
continues. The retry loop had no exit for the case where a whole batch buys no
progress, so the run never returned: ``max_steps=1_000_000`` changed nothing and
only the wall-clock ``timeout`` ever ended it. In a PyBNF fit with
``wall_time_sim = 60`` each such trial burned a full minute before being scored
``inf``.

The retry now stops the moment a batch fails to advance the internal time, which
is that stall and no other case — a model that legitimately needs many steps
advances every batch, however slowly.

Since issue #443 this fixture no longer wedges when it is run as written. Its
switch thresholds a counter species, bngsim now works out that the counter
reaches ``sigma`` at t = sigma and stops the step exactly there, and the run
completes — which is the better fix and is asserted at the bottom of this file.
So the three tests that need the stall stand the crossing stop down first, by
emptying the conditions the model derived. That is not an artificial state: it
is what every model whose crossing time bngsim cannot resolve still looks like —
a threshold on live state, a residual that is not linear in time — and those are
the models the bounded retry is there for.
"""

from __future__ import annotations

import time
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SimulationError

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "data"
STALL_NET = DATA_DIR / "switch_discontinuity_stall.net"

# sigma in the fixture — where the rate jump is and where the step size collapses.
SIGMA = 68.3718

# Comfortably longer than the milliseconds the bounded path needs, and far shorter
# than the minute the unbounded loop burned. A pass here is "failed fast", not
# "failed eventually".
_FAST = 30.0


@pytest.fixture
def stall_model():
    """The fixture with its crossing stop stood down, so it still wedges.

    See the module docstring: the model as written is repaired by issue #443,
    and what is under test here is the retry bound rather than the crossing.
    Emptying the derived conditions is how a model whose crossing time cannot be
    resolved reaches the integrator, which is the population this guard covers.
    """
    assert STALL_NET.exists(), f"test data not found: {STALL_NET}"
    model = bngsim.Model.from_net(str(STALL_NET))
    assert model.time_discontinuity_conditions(), (
        "the fixture no longer derives a crossing at all, so standing it down "
        "proves nothing; check what changed in the scan"
    )
    model._derived_time_disc_conditions = ()
    return model


def test_stall_raises_quickly_instead_of_hanging(stall_model):
    """The headline: default tolerances must fail fast, not spin."""
    sim = bngsim.Simulator(stall_model, method="ode")
    t0 = time.monotonic()
    with pytest.raises(SimulationError):
        sim.run(t_span=(0.0, 648.0), n_points=649)
    elapsed = time.monotonic() - t0

    assert elapsed < _FAST, (
        f"took {elapsed:.1f}s to give up; the retry loop is not bounded by progress "
        f"and only `timeout` is ending the run (issue #54)"
    )


def test_error_names_the_t_and_h_it_wedged_at(stall_model):
    """The issue asks for a *diagnosable* failure. The message must carry the
    integrator state, and point at the discontinuity rather than blaming the
    model in the abstract — ``t`` should be sigma, where the rate jumps."""
    sim = bngsim.Simulator(stall_model, method="ode")
    with pytest.raises(SimulationError) as exc:
        sim.run(t_span=(0.0, 648.0), n_points=649)

    msg = str(exc.value)
    assert "no progress" in msg
    assert "step size" in msg and "h=" in msg
    assert "t=" in msg

    reported_t = float(msg.split("t=")[1].split()[0])
    assert reported_t == pytest.approx(SIGMA, abs=1e-3), (
        f"reported t={reported_t} is not the discontinuity at sigma={SIGMA}; "
        f"the message would send a user looking in the wrong place"
    )


def test_raising_max_steps_does_not_reintroduce_the_hang(stall_model):
    """``max_steps=1_000_000`` was one of the configurations that never returned.
    A bigger batch is still a batch: it must still terminate."""
    sim = bngsim.Simulator(stall_model, method="ode")
    t0 = time.monotonic()
    with pytest.raises(SimulationError):
        sim.run(t_span=(0.0, 648.0), n_points=649, max_steps=1_000_000)
    assert time.monotonic() - t0 < _FAST


def test_loose_tolerances_still_integrate_normally(stall_model):
    """No false positives, and the reason this is bounded by *progress* rather
    than by a step count: at rtol=atol=1e-7 the same model integrates fine, and
    the issue's own table says so. A step-count ceiling would have failed this."""
    sim = bngsim.Simulator(stall_model, method="ode")
    r = sim.run(t_span=(0.0, 648.0), n_points=649, rtol=1e-7, atol=1e-7)

    species = np.asarray(r.species)
    assert species.shape[0] == 649
    assert np.all(np.isfinite(species))
    # The run really did cross the switch rather than stopping short of it.
    assert float(np.asarray(r.time)[-1]) == pytest.approx(648.0)
    assert float(np.asarray(r.time)[-1]) > SIGMA


def test_the_switch_is_stopped_at_rather_than_wedged_on():
    """Issue #443: the model this fixture came from now integrates.

    The stall it was written for is a step size collapsing at a rate jump the
    integrator tried to step over. The counter species the jump thresholds
    advances at exactly rate 1, so the crossing is at t = sigma and bngsim stops
    the step there, takes the whole approach on the branch that is ending, and
    restarts on the other one. Nothing is left for the error test to fail on.

    The same run with the crossing stood down is the `stall_model` fixture
    above, and it still raises — so this asserts the repair rather than a
    quietly loosened tolerance.
    """
    model = bngsim.Model.from_net(str(STALL_NET))
    assert model.time_discontinuity_conditions() == ("t>=sigma",)

    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 648.0), n_points=649)

    species = np.asarray(r.species)
    assert species.shape[0] == 649
    assert np.all(np.isfinite(species))
    assert float(np.asarray(r.time)[-1]) == pytest.approx(648.0)
    # The switch turns transfer on at sigma, so S_P has to leave zero after it
    # and stay at zero before it. Column order is the species block: S_M, S_P,
    # E, counter.
    times = np.asarray(r.time)
    assert species[times < SIGMA, 1].max() == 0.0
    assert species[-1, 1] > 0.0


def test_a_slow_but_advancing_model_is_not_refused():
    """The guard must not fire on an ordinary stiff model that simply needs many
    steps. A tiny per-output-point batch forces repeated CV_TOO_MUCH_WORK returns
    on a model with no discontinuity at all; every one of those batches advances
    t, so the run must complete normally."""
    net = DATA_DIR / "simple_decay.net"
    if not net.exists():
        pytest.skip(f"test data not found: {net}")

    m = bngsim.Model.from_net(str(net))
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, 10.0), n_points=11, max_steps=2)

    assert np.all(np.isfinite(np.asarray(r.species)))
    assert float(np.asarray(r.time)[-1]) == pytest.approx(10.0)
