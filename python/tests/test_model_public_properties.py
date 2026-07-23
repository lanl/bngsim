"""Guard the public `Model` surface that the user guide documents.

`Model` is a thin wrapper over the pybind11 `NetworkModel` (`self._core`), and a
property that exists on the core type is *not* automatically reachable on the
wrapper — there is no `__getattr__` forwarding. That gap is silent: the docs read
fine, the C++ binding is present, the tests pass because they reach through
`._core`, and only a user following the guide verbatim sees the `AttributeError`.

That is exactly how `Model.conservation_laws` (lanl/bngsim #31) and
`Model.n_events` (documented in `docs/user-guide/events.md`) were both advertised
for months while raising on the documented call. These tests assert the public
path specifically — reaching through `._core` here would defeat their purpose.
"""

from __future__ import annotations

import bngsim
import pytest

_DECAY_WITH_EVENT = """
    S = 100; D = 0;
    k_decay = 0.1;
    J1: S -> ; k_decay * S;

    at (time > 10): D = 50;
    at (time > 50): D = 0;
"""


@pytest.fixture
def event_model():
    """The two-event Antimony model from `docs/user-guide/events.md`."""
    return bngsim.Model.from_antimony_string(_DECAY_WITH_EVENT)


def test_n_events_is_public(event_model):
    """`model.n_events` — used by the first example in the events guide."""
    assert event_model.n_events == 2


def test_n_events_zero_for_eventless_model():
    model = bngsim.Model.from_antimony_string("S = 100; k = 0.1; J1: S -> ; k * S;")
    assert model.n_events == 0


def test_conservation_laws_is_public():
    """`model.conservation_laws` — the example in the conservation-laws guide.

    A → B conserves A + B, so exactly one law over two species.
    """
    model = bngsim.Model.from_antimony_string("A = 100; B = 0; k = 0.1; J1: A -> B; k * A;")
    laws = model.conservation_laws

    # Every key the user guide tells users to read.
    assert laws["n_laws"] == 1
    assert laws["n_species"] == 2
    assert len(laws["dependent"]) == 1
    assert len(laws["independent"]) == 1
    assert abs(abs(laws["constants"][0]) - 100.0) < 1e-10
    assert len(laws["coefficients"]) == 1
    assert len(laws["coefficients"][0]) == 2


def test_public_properties_agree_with_core(event_model):
    """The wrapper must forward, not reimplement."""
    assert event_model.n_events == event_model._core.n_events
    assert event_model.conservation_laws == event_model._core.conservation_laws
