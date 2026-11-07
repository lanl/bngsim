"""Network-free sessions: a constructor that raised, and copies of a live session.

``NfsimSession`` and ``RuleMonkeySession`` keep their state in ``__slots__`` and
hold a handle into the engine, which ``destroy()`` (and so ``__del__``) releases.

- A constructor that raised before it set ``_destroyed`` (an unknown keyword,
  which Python refuses before ``__init__`` runs, or a build without the engine)
  left ``__del__`` reading the unset slot: ``AttributeError``, reported through
  ``sys.unraisablehook`` ("Exception ignored in: ...__del__") beside the real
  error the caller got.
- ``copy.copy(session)`` shared the handle, and the copy's ``__del__`` destroyed
  it: dropping the copy left the original's next ``simulate()`` failing with "No
  active session". Copying and pickling are now refused, as for ``Model`` and
  ``Simulator`` (issue #643).
"""

from __future__ import annotations

import copy
import gc
import pickle
import sys
from pathlib import Path

import bngsim
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
COMPARTMENT_XML = DATA / "compartment_bimolecular_size2.xml"
SWITCH_XML = DATA / "first_order_switch.xml"
SESSIONS = [
    pytest.param("NfsimSession", "HAS_NFSIM", id="nfsim"),
    pytest.param("RuleMonkeySession", "HAS_RULEMONKEY", id="rulemonkey"),
]


def _need(flag: str, cls_name: str) -> None:
    if getattr(bngsim._bngsim_core, flag):
        return
    if cls_name == "NfsimSession":
        pytest.skip("NFsim not compiled in")
    pytest.skip("RuleMonkey not compiled in")


@pytest.fixture
def unraisable(monkeypatch):
    """What Python reports from ``__del__`` (and any other unraisable error),
    from a collector installed after earlier tests' garbage is gone."""
    gc.collect()
    seen: list = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


@pytest.fixture
def deleted(monkeypatch):
    """Count each session's ``__del__`` calls, so a test that sees nothing
    reported knows the destructor did run inside it."""
    calls: dict[str, int] = {}

    for name in ("NfsimSession", "RuleMonkeySession"):
        cls = getattr(bngsim, name)
        original = cls.__del__

        def counted(self, _original=original, _name=name):
            calls[_name] = calls.get(_name, 0) + 1
            _original(self)

        monkeypatch.setattr(cls, "__del__", counted)
    return calls


def _reported(seen: list) -> list[str]:
    gc.collect()
    return [f"{type(u.exc_value).__name__}: {u.exc_value}" for u in seen]


@pytest.mark.parametrize("cls_name, flag", SESSIONS)
@pytest.mark.parametrize(
    "args, kwargs",
    [
        pytest.param(("model.xml",), {"seed": 1}, id="unknown-keyword"),
        pytest.param((), {}, id="missing-path"),
    ],
)
def test_a_call_python_refuses_raises_typeerror_and_nothing_else(
    cls_name, flag, args, kwargs, unraisable, deleted
):
    """seed= is initialize()'s keyword, not the constructor's, and the path is
    required: the caller gets TypeError, and the half-made session reports
    nothing on its way out."""
    with pytest.raises(TypeError):
        getattr(bngsim, cls_name)(*args, **kwargs)
    assert _reported(unraisable) == []
    assert deleted.get(cls_name) == 1


@pytest.mark.parametrize("cls_name, flag", SESSIONS)
def test_a_build_without_the_engine_raises_runtimeerror_and_nothing_else(
    cls_name, flag, unraisable, deleted, monkeypatch
):
    """The engine check is the constructor's first line, before _destroyed is
    set: the caller gets the RuntimeError that says the engine is absent, and
    nothing more."""
    monkeypatch.setattr(bngsim._bngsim_core, flag, False)
    with pytest.raises(RuntimeError, match="support is not present"):
        getattr(bngsim, cls_name)("model.xml")
    assert _reported(unraisable) == []
    assert deleted.get(cls_name) == 1


@pytest.mark.parametrize("cls_name, flag", SESSIONS)
def test_a_constructor_that_refuses_the_model_still_reports_nothing(
    cls_name, flag, unraisable, deleted
):
    """Neither engine runs compartments, and each refuses them after _destroyed
    is set, so __del__ calls destroy() on a session with no core. That was quiet
    before this fix too; it is held here so the fix trades no noise for another."""
    _need(flag, cls_name)
    with pytest.raises(bngsim.ModelError, match="compartment"):
        getattr(bngsim, cls_name)(str(COMPARTMENT_XML))
    assert _reported(unraisable) == []
    assert deleted.get(cls_name) == 1


@pytest.mark.parametrize("cls_name, flag", SESSIONS)
def test_a_half_made_session_still_has_a_repr(cls_name, flag):
    """A traceback that shows locals reprs the session a raising constructor
    left; that read the unset slots too."""
    half = object.__new__(getattr(bngsim, cls_name))
    assert repr(half) == f"{cls_name}(xml_path=None, state=destroyed)"


@pytest.mark.parametrize("cls_name, flag", SESSIONS)
@pytest.mark.parametrize(
    "how, refusal",
    [
        pytest.param(copy.copy, "cannot copy a", id="copy"),
        pytest.param(copy.deepcopy, "cannot pickle a", id="deepcopy"),
        pytest.param(pickle.dumps, "cannot pickle a", id="pickle"),
    ],
)
def test_a_live_session_is_not_copied_and_runs_on_after_the_attempt(
    cls_name, flag, how, refusal, unraisable
):
    """A copy shared the engine handle, and dropping it destroyed the original's
    session. Each way of copying is refused by name, and the original, set up
    before the attempt, simulates after it."""
    _need(flag, cls_name)
    session = getattr(bngsim, cls_name)(str(SWITCH_XML))
    try:
        session.initialize(seed=1)
        with pytest.raises(TypeError, match=f"{refusal} {cls_name}"):
            how(session)
        gc.collect()
        result = session.simulate(0, 1, 3)
        assert result.observables.shape == (3, 1)
    finally:
        session.destroy()
    assert _reported(unraisable) == []
