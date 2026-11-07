"""What objects backed by the compiled extension do under pickle and copy.

:class:`bngsim.Result`, :class:`bngsim.Model` and :class:`bngsim.Simulator` each
hold a handle into ``bngsim._bngsim_core``. A pybind11 object has no pickle
support, so none of the three can be serialized — which is fine, and is not the
problem. The problem is what a caller was told: ``cannot pickle
'bngsim._bngsim_core.ResultCore' object`` names a type they have never seen and
says nothing about what to do instead, while the caller is a fitting or scanning
harness returning work from a process pool and the way through is to convert
first (issue #636).

Pickling and copying share one hook, so both live here: ``copy.copy`` falls back
to ``__reduce_ex__``, which defers to ``__reduce__`` when a class overrides it,
so a refusal written there refuses a *shallow* copy too. Each class therefore
says under ``__copy__`` what a shallow copy of it should mean, and the answer is
not the same for all three (issue #643):

- A ``Model`` and a ``Simulator`` **refuse**. Every scrap of a ``Model``'s state
  lives behind the one handle, so a copy that shares it is not a shallow copy in
  any useful sense — it is an alias with a different ``id()``, and a
  ``set_param`` through it changes the original, which then simulates
  differently with nothing raised. A ``Simulator`` shares its ``Model`` outright
  and aliases by the same route. :func:`raise_cannot_shallow_copy` names
  ``clone()``, which does give an independent model.
- A ``Result`` still copies, through :func:`shallow_copy`. It shares its core
  and its arrays the same way, but it exposes no setter and no mutating method —
  only readers and exporters — so the API offers no write for the sharing to
  carry. An in-place numpy write into a shared array (``copy.copy(r).species[0]
  = ...``) does reach the original, and is pinned as such; that is the aliasing
  numpy already has, though, not one the copy introduces: every access hands
  back the same live view, so the write reaches the original through the
  original too.

``copy.deepcopy`` reaches the pickle refusal on all three and is meant to: it
failed before too, with the message this module exists to replace.

:class:`bngsim.NfsimSession` and :class:`bngsim.RuleMonkeySession` refuse both
as well. A shallow copy shared the session's handle into the network-free
engine, and the copy's ``__del__`` destroyed it: dropping ``copy.copy(session)``
left the original's next ``simulate()`` failing with "No active session".
"""

from __future__ import annotations

from typing import Any, NoReturn, TypeVar

T = TypeVar("T")


def raise_cannot_pickle(what: str, remedy: str) -> NoReturn:
    """Refuse to serialize an extension-backed object, and say what to do.

    *remedy* is the class's own sentence: which conversion carries the part of
    it that a worker process actually needs.
    """
    raise TypeError(
        f"cannot pickle a {what}: it holds a handle into bngsim's compiled extension, so "
        f"it cannot cross a process boundary or be deep-copied. {remedy}"
    )


def raise_cannot_shallow_copy(what: str, shared: str, remedy: str) -> NoReturn:
    """Refuse a shallow copy that would only alias the original, and say what to do.

    *shared* names what the copy would go on sharing; *remedy* is the class's own
    sentence naming the operation that does hand back an independent object.
    """
    raise TypeError(
        f"cannot copy a {what} with copy.copy(): the copy would share {shared}, so a write "
        f"through the copy changes the original, which then simulates differently with "
        f"nothing raised. {remedy}"
    )


def shallow_copy(obj: T) -> T:
    """A shallow copy of a ``__slots__`` object, as the default ``copy.copy`` made.

    The copy shares every value the original holds, the extension handle and
    the arrays included. Used by :class:`bngsim.Result`, whose API is read-only,
    so no method of its can write through the sharing; the classes whose API can
    refuse instead (see the module docstring, and issue #643).
    """
    new = object.__new__(type(obj))
    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            try:
                value: Any = getattr(obj, slot)
            except AttributeError:
                continue  # an unset slot stays unset, as it would in a default copy
            setattr(new, slot, value)
    return new
