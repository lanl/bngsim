"""Issue #643 — what ``copy.copy`` means for an extension-backed object.

``copy.copy(model)`` used to hand back a ``Model`` sharing the original's engine
handle. That is textbook shallow-copy behaviour — ``copy.copy`` is documented to
insert references to the objects found in the original — but all of a ``Model``'s
state lives behind that one handle, so there was nothing left for the copy to
own. The result was not a shallow copy in any useful sense; it was an alias with
a different ``id()``::

    c = copy.copy(m)          # m has k = 0.1
    c.set_param("k", 5.0)
    m.get_param("k")          # 5.0 — the write landed on m
    sim.run(...)              # and m now integrates to a different answer

Nothing raised, and the two things a caller reaches for next did not rescue
them: ``copy.deepcopy`` raises (it cannot pickle the handle), and ``clone()``,
which does give an independent model, is invisible from the copy path.

The call taken here: ``Model`` and ``Simulator`` refuse and name the remedy —
loud where the alias was silent, and it cannot make anyone's numbers quietly
wrong. ``Result`` keeps its copy: it has no setter and no mutating method, so
no call on the copy writes into the original. This file pins that split, and
pins the premise the Result half rests on, so that adding a setter to ``Result``
fails here rather than silently re-opening the trap.
"""

from __future__ import annotations

import copy
import inspect

import bngsim
import numpy as np
import pytest
from bngsim._foreign_names import _ForeignAttribute

from .test_pickle_refusals import DECAY_SBML  # k = 0.1, X(0) = 100, X' = -k X


@pytest.fixture
def model():
    return bngsim.Model.from_sbml_string(DECAY_SBML)


def _x_at_1(m: bngsim.Model) -> float:
    """Where the model as it stands would integrate X to at t = 1.

    Solved on a clone, because a run writes the end-of-span concentrations back
    into the model it solved (issue #553) — probing the original directly would
    move the very state under test.
    """
    probe = bngsim.Simulator(m.clone(), method="ode").run(t_span=(0, 1), n_points=2)
    return float(np.asarray(probe.species)[-1, 0])


# ── The trap, closed ─────────────────────────────────────────────────


def test_copying_a_model_refuses_and_names_clone(model):
    with pytest.raises(TypeError) as excinfo:
        copy.copy(model)

    message = str(excinfo.value)
    assert "cannot copy a Model with copy.copy()" in message
    assert "clone()" in message
    # The reason, in terms of what would have gone wrong, not a C++ type name.
    assert "share" in message
    assert "_bngsim_core" not in message


def test_copying_a_simulator_refuses_and_names_what_to_rebuild(model):
    sim = bngsim.Simulator(model, method="ode")

    with pytest.raises(TypeError) as excinfo:
        copy.copy(sim)

    message = str(excinfo.value)
    assert "cannot copy a Simulator with copy.copy()" in message
    assert "model.clone()" in message
    assert "_bngsim_core" not in message


def test_the_remedy_the_message_names_is_real(model):
    """An error message that recommends a method is a claim about the API."""
    assert hasattr(model, "clone")
    assert hasattr(bngsim.Simulator(model, method="ode"), "requested_method")


# ── ...and the remedy really is independent, which is the whole point ─


@pytest.mark.parametrize(
    ("write", "read"),
    [
        (lambda m: m.set_param("k", 5.0), lambda m: m.get_param("k")),
        (
            lambda m: m.set_concentration("X", 7.0),
            lambda m: m.get_concentration("X"),
        ),
    ],
    ids=["set_param", "set_concentration"],
)
def test_a_write_through_a_clone_leaves_the_original_alone(model, write, read):
    """Both writes the issue names, through the operation the refusal offers."""
    before = read(model)
    solves_to = _x_at_1(model)

    clone = model.clone()
    write(clone)

    assert read(model) == before
    assert read(clone) != before
    # The part that made the alias a bug rather than a curiosity: the original
    # went on simulating differently. It must not here.
    assert _x_at_1(model) == pytest.approx(solves_to)


def test_the_original_still_integrates_as_loaded_after_a_refused_copy(model):
    """A refusal that half-happened would be worse than the alias."""
    solves_to = _x_at_1(model)
    with pytest.raises(TypeError):
        copy.copy(model)
    assert model.get_param("k") == pytest.approx(0.1)
    assert _x_at_1(model) == pytest.approx(solves_to)
    assert _x_at_1(model) == pytest.approx(np.exp(-0.1) * 100.0, rel=1e-6)


# ── Result keeps its copy, and why ───────────────────────────────────


def test_copying_a_result_still_works(model):
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)

    duplicate = copy.copy(result)

    assert duplicate is not result
    assert type(duplicate) is type(result)
    np.testing.assert_array_equal(np.asarray(duplicate.species), np.asarray(result.species))
    assert duplicate.n_times == result.n_times


def test_result_exposes_no_setter_for_the_sharing_to_carry(model):
    """The premise the carve-out rests on, pinned.

    ``Result`` copies while ``Model`` refuses only because nothing in a
    ``Result``'s API writes. The first setter or mutator added to it turns
    ``copy.copy(result)`` into the same trap ``Model.__copy__`` now refuses, so
    fail here and make that a decision rather than an accident.
    """
    mutating_prefixes = ("set_", "add_", "update", "insert", "append", "remove", "clear", "pop")
    offenders = []
    for name in dir(bngsim.Result):
        if name.startswith("_"):
            continue
        if isinstance(inspect.getattr_static(bngsim.Result, name), _ForeignAttribute):
            continue  # another tool's name, there only to raise naming bngsim's (issue #1700)
        attribute = getattr(bngsim.Result, name)
        if isinstance(attribute, property) and attribute.fset is not None:
            offenders.append(f"{name} (property setter)")
        elif callable(attribute) and name.startswith(mutating_prefixes):
            offenders.append(f"{name}()")

    assert not offenders, (
        "Result has grown a way to write through it: "
        + ", ".join(offenders)
        + ". copy.copy(result) shares the original's state, so this re-opens issue #643 — "
        "either refuse the copy as Model does, or give Result its own clone()."
    )


def test_a_result_copy_shares_its_arrays_which_numpy_already_did(model):
    """The limit of the carve-out, stated rather than assumed.

    In-place writes DO cross between a Result and its copy. That aliasing is
    numpy's, not the copy's: each accessor returns the stored array itself, so
    the same write reaches the result through the original name too.
    """
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)

    assert result.species is result.species  # the same live view every time
    assert copy.copy(result).species is result.species

    np.asarray(copy.copy(result).species)[0, 0] = 999.0
    assert float(np.asarray(result.species)[0, 0]) == 999.0
