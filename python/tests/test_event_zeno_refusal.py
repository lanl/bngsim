"""Issue #1158 — an event that re-arms its own trigger as it fires is refused.

``at (B > 0): B = 0, F = F + 1`` with ``B' = 1``: each fire puts B back on its
threshold, the flow makes the trigger true again at once, and the event fires
infinitely often at one instant (Zeno behaviour). No trajectory continues past
that instant, in bngsim or anywhere else; libRoadRunner stops at F = 2, which is
where its own event handling gives up, not an answer.

bngsim fired at every root CVODE reported, each one root tolerance (100 ulps of
|t| + |h|) after the last, about 4e5 fires a second, and ``run`` never returned:
over t in [0, 1e-12] as over [0, 2]. The chatter guard (GH #95) did not see it,
since each fire changes the state (F, or B and A) by more than the tolerance.

Those fires are the same instant to the solver, so a run of them is the
same-instant algebraic loop ``CASCADE_LIMIT`` (100000) already refuses, one
rounding at a time: a root batch within the root tolerance of the batch before
(whichever events fired in it, so a ring of events re-arming each other counts
too) is counted, and past the limit the run is refused, naming every event that
fired in the streak. A pulse train spaced by more than the root tolerance, which
CVODE resolves, never counts.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_ZENO = {
    "reset B, first order": "species B = 0; B' = 1;\nF = 0; E: at (B > 0): F = F + 1, B = 0;\n",
    "reset B, second order": (
        "species B = 0, A = 0; A' = 1; B' = A;\nF = 0; E: at (B > 0): F = F + 1, B = 0;\n"
    ),
    "reset B and A": (
        "species B = 0, A = 0; A' = 1; B' = A;\nF = 0; E: at (B > 0): F = F + 1, B = 0, A = 0;\n"
    ),
    "put on the threshold mid-run": (
        "species B = -1, A = 0; A' = 1; B' = A;\nF = 0;\n"
        "E0: at (time > 0.5): B = 0, A = 0;\nE: at (B > 0): F = F + 1, B = 0, A = 0;\n"
    ),
    "a clamp with a counter": "species X = 1; X' = -1; c = 0;\nE: at (X < 0): X = 0, c = c + 1;\n",
}


def _run(text, t_span, n_points=5):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode").run(
        t_span=t_span, n_points=n_points, rtol=1e-10, atol=1e-12
    )
    last = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))
    last.update(zip(run.observable_names, np.asarray(run.observables)[-1], strict=True))
    return last


@pytest.mark.parametrize("shape", sorted(_ZENO))
def test_an_event_that_rearms_itself_as_it_fires_is_refused_by_name(shape):
    """Each of these hung: `run` did not return. Now it is refused, naming E
    (and E0, which put B on its threshold, where it fired in the streak)."""
    with pytest.raises(bngsim.SimulationError, match=r"Events fired 100000 times") as err:
        _run(_ZENO[shape], (0.0, 2.0))
    assert "'E'" in str(err.value)
    assert "Zeno behaviour" in str(err.value)
    assert "the run has no continuation past it" in str(err.value)


def _ring(n, on_at=None):
    """Ei, `at (Xi > 0)`, resets X(i+1 mod n) to 0, all Xi' = 1, X0 = 0: each
    event re-arms the next, and the ring fires infinitely often at one
    instant (libRoadRunner stops at F = 2)."""
    lines = [f"species X{i} = {0 if i == 0 else 1}; X{i}' = 1;" for i in range(n)]
    lines.append("species F = 0;")
    lines += [f"E{i}: at (X{i} > 0): X{(i + 1) % n} = 0, F = F + 1;" for i in range(n)]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("n", [2, 3, 5])
def test_a_ring_of_events_rearming_each_other_is_refused_naming_them_all(n):
    """Counted per event, a ring of three or more never came within the window
    of its own previous fire and still hung."""
    with pytest.raises(bngsim.SimulationError, match="Zeno behaviour") as err:
        _run(_ring(n), (0.0, 2.0))
    for i in range(n):
        assert f"'E{i}'" in str(err.value)


def test_an_event_that_shares_the_trigger_is_named_with_the_one_that_resets_it():
    """Watch only counts; Reset re-arms the trigger they share. Counted per
    event the refusal named Watch alone and told the user to change it."""
    text = (
        "species B = 0; B' = 1; species F = 0, G = 0;\n"
        "Watch: at (B > 0): G = G + 1;\nReset: at (B > 0): F = F + 1, B = 0;\n"
    )
    with pytest.raises(bngsim.SimulationError, match=r"\('Watch', 'Reset'\)"):
        _run(text, (0.0, 2.0))


@pytest.mark.parametrize("t0", [300.0, 1e6])
def test_a_self_reset_is_refused_from_any_start(t0):
    """From a start that is not 0 the fires are as close, to that time's own
    rounding."""
    with pytest.raises(bngsim.SimulationError, match="Zeno behaviour"):
        _run(_ZENO["reset B, first order"], (t0, t0 + 2.0))


@pytest.mark.parametrize(
    "t_start, period",
    [(0.0, 1e-6), (1.0, 2e-9), (1.0, 2.887e-14)],
    ids=["1e-6 from 0", "2e-9 at t=1", "1.3 root tolerances at t=1"],
)
def test_a_fast_pulse_train_past_the_limit_is_not_refused(t_start, period):
    """More than CASCADE_LIMIT fires of one event, each a real period after the
    last: a train, not a loop. The last is 1.3 root tolerances apart at t = 1
    (100 ulps of 1 is 2.2e-14), which CVODE resolves; a window of twice the
    tolerance refused it."""
    text = (
        f"species Z = 1; Z' = -Z; n = 0; tn = {t_start + period!r};\n"
        f"P: at (time >= tn && n < 105000), t0=false: tn = tn + {period!r}, n = n + 1;\n"
    )
    last = _run(text, (t_start, t_start + 110000 * period), 3)
    assert last["n"] == 105000.0


def test_a_decaying_clamp_is_still_stepped_over_by_the_chatter_guard():
    """`at (X < 0): X = 0` with X' = -X fires on noise once X is far below
    atol, changing it by less than the tolerance: the chatter guard (GH #95)
    makes it dormant long before the Zeno guard would count to its limit.
    X(3) = e^-3 as in libRoadRunner (0.0497873)."""
    last = _run("species X = 1; X' = -X;\nC: at (X < 0): X = 0;\n", (0.0, 3.0), 4)
    assert last["X"] == pytest.approx(np.exp(-3.0), rel=1e-5)
