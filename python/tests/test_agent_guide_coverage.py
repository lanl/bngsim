"""Every public bngsim name is in the agent guide or left out of it on purpose (#1699).

``test_agent_guide.py`` holds the guide to the code: every block runs and
every name it spells exists. That stops the guide saying something wrong. It
does not stop the guide falling silent: a new method, keyword, simulation
method, capability or extra is public the day it merges, and nothing would
ask whether an agent reading the guide should hear of it.

This test asks. The public surface is:

* ``bngsim:<name>`` for each name in ``bngsim.__all__``;
* ``<Class>:<name>`` for each public attribute of ``Model``, ``Simulator``,
  ``Result`` and ``SteadyStateResult`` (not the wrong names bngsim answers
  with an error, which the guide's table covers);
* ``<call>(<keyword>)`` for each keyword of ``Simulator()``, the run calls
  and ``steady_state``;
* ``method:<name>``, ``feature:<name>`` and ``extra:<name>`` for each
  ``Simulator.METHODS`` value, ``capabilities()`` feature and install extra.

A name counts as in the guide when it appears as code there: in a block, in
backticks or in the generated table. The rest are listed, one per line, in
:data:`LEFT_OUT`, and the two must cover the surface exactly. So a new public
name fails this test until its author either uses it in the guide or adds it
to the list, a one-line decision a reviewer sees, and a name that goes away or
that the guide takes up fails it until its line is dropped.

The list is not a to-do list. The guide is a page held to a budget, and most
of the surface (solver diagnostics, the sensitivity fields of a steady state,
the codegen cache calls) is for ``help()``, not for a first read.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import re
from pathlib import Path

import bngsim
from bngsim._foreign_names import RUN_CALLS, _ForeignAttribute

#: The names the guide leaves out on purpose, one ``owner:name`` per line.
LEFT_OUT = Path(__file__).with_name("agent_guide_left_out.txt")

_CLASSES = (bngsim.Model, bngsim.Simulator, bngsim.Result, bngsim.SteadyStateResult)
_KEYWORD_CALLS = ("Simulator", *RUN_CALLS, "steady_state")

_FENCE = re.compile(r"^```\w*\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _keywords(call: str) -> list[str]:
    target = bngsim.Simulator if call == "Simulator" else getattr(bngsim.Simulator, call)
    return [
        name
        for name, p in inspect.signature(target).parameters.items()
        if name != "self"
        and not name.startswith("_")
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _surface() -> dict[str, str]:
    """Each public name's key, mapped to the identifier the guide would spell."""
    surface = {f"bngsim:{name}": name for name in bngsim.__all__}
    for cls in _CLASSES:
        for name in dir(cls):
            if name.startswith("_"):
                continue
            if isinstance(inspect.getattr_static(cls, name, None), _ForeignAttribute):
                continue
            surface[f"{cls.__name__}:{name}"] = name
    for call in _KEYWORD_CALLS:
        for name in _keywords(call):
            surface[f"{call}({name})"] = name
    for name in bngsim.Simulator.METHODS:
        surface[f"method:{name}"] = name
    for name in bngsim.capabilities()["features"]:
        surface[f"feature:{name}"] = name
    for name in importlib.metadata.metadata("bngsim").get_all("Provides-Extra") or ():
        surface[f"extra:{name}"] = name
    return surface


def _code() -> str:
    """The guide's code: its blocks and its backticked spans, the table's included."""
    text = bngsim.agent_guide()
    blocks = _FENCE.findall(text)
    spans = re.findall(r"`([^`\n]+)`", _FENCE.sub("", text))
    return "\n".join(blocks + spans)


def _left_out() -> set[str]:
    lines = LEFT_OUT.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_every_public_name_is_in_the_guide_or_left_out_on_purpose():
    code = _code()
    surface = _surface()
    used = {key for key, name in surface.items() if re.search(rf"\b{re.escape(name)}\b", code)}
    should_be_listed = set(surface) - used
    listed = _left_out()
    new = sorted(should_be_listed - listed)
    stale = sorted(listed - should_be_listed)
    assert not new and not stale, (
        "the agent guide's coverage changed. For each new public name, use it in "
        "python/bngsim/agent_guide.md if an agent meeting bngsim should know it, or "
        f"add its line to {LEFT_OUT.name}; drop each stale line.\n"
        f"new, in neither: {new}\nstale lines: {stale}"
    )


def test_the_left_out_list_is_sorted_and_has_no_duplicates():
    lines = [
        line for line in LEFT_OUT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]  # fmt: skip
    assert lines == sorted(set(lines))
