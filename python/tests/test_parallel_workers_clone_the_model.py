"""Every parallel worker integrates its OWN model clone — the invariant #201 rests on.

Issue #201 fixed a race on the one object clones genuinely share, the ExprTk parser.
The audit that accompanied it found no second race, but the reason was
circumstantial: ``NetworkModel::state_switch`` and ``event_trigger_residual_expr``
are lazy memos — ``const`` methods that compile into ``impl_->evaluator`` on first
use — and they are safe *only* because every thread fan-out in the library hands
each worker its own ``clone()``. That is a property of today's call sites, not
something the type system holds.

It is worth stating the invariant in its general form rather than as a fact about
those two memos, because the memos are not the fragile part. An integration writes
``impl_->species`` concentrations and ``impl_->current_time`` straight back onto the
model. Two threads integrating one ``NetworkModel`` corrupt each other's *state*
long before anyone reaches a lazy compile. So:

    A NetworkModel is not thread-safe. Any fan-out must give each worker its own
    clone.

This test asserts exactly that, at runtime, over every parallel entry point the
library exposes — rather than asserting the syntactic proxy "the source contains a
.clone() call", which can pass while being wrong. It wraps the C++ simulator
constructors (imported at call time in ``_simulator.py``, so the module attribute is
the seam) and records which thread constructed a simulator over which model core. If
any core object is seen from two different threads, the invariant is broken.

Adding a new parallel entry point that forgets to clone fails here, which is the
point: the failure mode it would otherwise reintroduce is a data race, and a race
found by CI is worth a great deal more than a race found in a user's fit.
"""

from __future__ import annotations

import threading
from collections import defaultdict

import bngsim
import bngsim._bngsim_core as core
import pytest


def _model():
    """Two parameters a scan can vary, over a reversible pair. Deliberately tiny:
    this test is about which thread touches which object, not about numerics."""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("kf", 0.4)
    b.add_parameter("kr", 0.1)
    a = b.add_species("A", 10.0)
    c = b.add_species("B", 0.0)
    b.add_reaction([a], [c], "elementary", "kf")
    b.add_reaction([c], [a], "elementary", "kr")
    b.add_observable("Atot", [(a, 1.0)])
    return bngsim.Model(b.build())


class _Tracker:
    """Records (core object identity) -> {thread ids that integrated it}.

    Keys are ``id()``, which is only meaningful while the object is alive — a
    freed clone's address is promptly reused, and two distinct cores landing on
    one address would be reported as a shared model that never existed. Observed
    while writing this: a phantom "one core, two threads" in a run that was in
    fact correct. So the tracker holds a strong reference to every core it has
    seen, which is what makes the key stable for the length of the test.
    """

    def __init__(self):
        self.seen: dict[int, set[int]] = defaultdict(set)
        self._keepalive: list = []
        self.lock = threading.Lock()

    def note(self, model_core):
        with self.lock:
            self._keepalive.append(model_core)
            self.seen[id(model_core)].add(threading.get_ident())

    def shared(self) -> dict[int, set[int]]:
        return {k: v for k, v in self.seen.items() if len(v) > 1}

    def threads(self) -> set[int]:
        return {tid for tids in self.seen.values() for tid in tids}


@pytest.fixture
def tracker(monkeypatch):
    t = _Tracker()
    # Every C++ entry point a worker can integrate through. `find_steady_state` is
    # a free function rather than a simulator class, and leaving it out made the
    # steady_state_batch case instrument nothing but the constructor — which the
    # "did the fan-out fan out" guard caught, and which a bare "no core crossed
    # threads" assertion would have reported as a pass.
    for name in ("CvodeSimulator", "SsaSimulator", "find_steady_state"):
        original = getattr(core, name)

        def make(orig):
            def wrapper(model_core, *args, **kwargs):
                t.note(model_core)
                return orig(model_core, *args, **kwargs)

            return wrapper

        monkeypatch.setattr(core, name, make(original))
    return t


def _exercise(tracker, what: str, call):
    """Run a parallel entry point, then check the invariant either way.

    A fan-out that shares its model does not politely reach the assertion below:
    the workers scribble on each other's species vector and CVODE dies first
    (measured — reverting run_batch's clone gives "CVODE integration failed
    flag=-3", not a clean result). So an exception is caught here and re-reported
    against the tracker, or the real cause is a crash whose message says nothing
    about threading.
    """
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - re-raised below if not our diagnosis
        if tracker.shared():
            _assert_no_core_crossed_threads(tracker, f"{what} (and it then failed: {exc})")
        raise
    _assert_no_core_crossed_threads(tracker, what)


def _assert_no_core_crossed_threads(tracker, what: str):
    shared = tracker.shared()
    assert tracker.seen, f"{what}: nothing was integrated — the wrapper never fired"
    # Without this the whole assertion is vacuous: "no core was seen from two
    # threads" is also true when the fan-out quietly ran on one. Requiring real
    # concurrency is what makes a green result mean the invariant HELD rather
    # than that it was never exercised.
    assert len(tracker.threads()) > 1, (
        f"{what}: every integration ran on one thread, so this proves nothing about "
        f"per-worker cloning. The fan-out did not fan out."
    )
    assert not shared, (
        f"{what}: {len(shared)} NetworkModel core(s) were integrated from more than one "
        f"thread. A NetworkModel is not thread-safe — an integration writes species "
        f"concentrations and current_time onto it, and its lazy expression memos "
        f"(state_switch, event_trigger_residual_expr) compile into a shared evaluator. "
        f"Every parallel worker must integrate its own model.clone() (issue #201)."
    )


def test_run_batch_workers_each_get_their_own_clone(tracker):
    sim = bngsim.Simulator(_model(), method="ode")
    _exercise(
        tracker,
        "run_batch(num_processors=4)",
        lambda: sim.run_batch(
            (0.0, 1.0),
            3,
            params=[{"kf": 0.3}, {"kf": 0.5}, {"kf": 0.7}, {"kf": 0.9}],
            num_processors=4,
        ),
    )


def test_compute_all_sensitivities_chunks_each_get_their_own_clone(tracker):
    sim = bngsim.Simulator(_model(), method="ode")
    _exercise(
        tracker,
        "compute_all_sensitivities(n_workers=2)",
        lambda: sim.compute_all_sensitivities(
            t_span=(0.0, 1.0), n_points=3, params=["kf", "kr"], chunk_size=1, n_workers=2
        ),
    )


def test_steady_state_batch_workers_each_get_their_own_clone(tracker):
    sim = bngsim.Simulator(_model(), method="ode")
    _exercise(
        tracker,
        "steady_state_batch(n_workers=4)",
        lambda: sim.steady_state_batch(
            [{"kf": 0.3}, {"kf": 0.5}, {"kf": 0.7}, {"kf": 0.9}], n_workers=4, max_time=1e4
        ),
    )


def test_the_tracker_would_actually_catch_a_shared_model(tracker):
    """The test's own negative control.

    Everything above passes when the wrapper fires and sees no sharing — which is
    also what it does when the wrapper is broken and sees nothing at all. This
    drives one core from two threads on purpose and requires the tracker to say so,
    so a green suite above means the invariant held rather than that nothing was
    watched.
    """
    m = _model()
    barrier = threading.Barrier(2)

    def integrate():
        barrier.wait()
        core.CvodeSimulator(m._core)

    threads = [threading.Thread(target=integrate) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.shared(), "the tracker did not notice one core used by two threads"
