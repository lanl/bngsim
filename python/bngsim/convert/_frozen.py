"""Parameters folded at load, in a network a converter writes (issue #1792).

:attr:`bngsim.Model.frozen_params` lists the parameters whose value went into
a number when an SBML model was built: a stoichiometry, a compartment's size,
an initial value an initialAssignment sets. ``set_param`` refuses to change
one, because the write would move nothing it was folded into. A written
``.net`` or ``.bngl`` holds that number too, and held the parameter like any
other, so the model loaded from it took the write: without effect where
nothing else read the parameter, and with half an effect where a rate law
still did.

The two cases are told apart on the text that is written:

* Nothing else in the file names the parameter. It is left out. ``set_param``
  on it then fails as a name the model has not, and so does a
  ``setParameter``, in bngsim and in BNG2.pl alike.
* Something still reads it (a rate law, a function, another parameter, a
  species' initial value). It stays, with ``folded_at_load=<what it was folded
  into>`` in a comment on its line, which ``Model.from_net`` and
  ``Model.from_bngl`` read back into ``frozen_params``. Only bngsim reads the
  comment: any other reader of the file takes the parameter as free.

A parameter BNG2.pl folded into the rate constants of a compartmental network
(a ``unit_conversion=`` comment, issue #711) is not handled here: it is
written as before.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

MARKER = "folded_at_load="
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_INDEX = re.compile(r"(\s*)\d+(?=\s)")


def _reasons(model: Any) -> dict[str, str]:
    """What each frozen parameter this module handles was folded into."""
    frozen = getattr(model, "_frozen_params", None) or {}
    return {str(n): str(w) for n, w in frozen.items() if "unit_conversion=" not in str(w)}


def has_folded(model: Any) -> bool:
    """Whether *model* has a parameter this module writes otherwise."""
    return bool(_reasons(model))


def as_names(given: Iterable[str] | str) -> frozenset[str]:
    """*given*, the names actions read, as a set: read once, so that an
    iterator serves every use of it, and a ``str`` taken for the one name it
    is, not for its letters."""
    return frozenset((given,)) if isinstance(given, str) else frozenset(given)


def names_read(protocol: Any) -> frozenset[str]:
    """The names the actions of *protocol* read when run: every name
    in a value that a ``setConcentration``, an ``addConcentration`` or a
    ``setParameter`` is given as an expression, and in any other text an
    experiment carries (a stop condition). What an action sets or scans is no
    read. A parameter among these has to be in the network the actions are run
    on, so :func:`fold` keeps it."""
    from bngsim._model import _names_in

    def texts(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for inner in value.values():
                yield from texts(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                yield from texts(inner)

    found: set[str] = set()
    for step in getattr(protocol, "steps", None) or ():
        for text in (
            *texts(getattr(step, "value_expr", None)),
            *texts(getattr(step, "extra", None)),
        ):
            found.update(_names_in(text))
    return frozenset(found)


def fold(model: Any, text: str, also_read: Iterable[str] = ()) -> tuple[str, list[str], list[str]]:
    """*text*, a ``.net`` or a BNGL model block with every parameter of *model*
    written, as it is to be written: ``(text, left out, marked)``, the two
    lists the names of *model*'s frozen parameters in the order of the file.

    A parameter is read when a line other than its own names it, comments
    aside, or when it is among *also_read*, the names that actions written
    beside the file read (:func:`names_read`). One that only parameters left
    out read is left out as well. Any use of the name counts, also where it is
    a species, an observable or a molecule type of that name: the parameter
    then stays, marked, which errs on the side of refusing a write. A name
    counts as the network's own reader finds it: after a number (``2f`` is
    ``2*f``), not as the exponent of one (``1e-30``). The lines that open and
    close a block (``begin species``) name nothing.
    """
    reasons = _reasons(model)
    if not reasons:
        return text, [], []
    from bngsim._model import _FOLDED_READ_BACK, _names_in

    kept_for_actions = as_names(also_read)
    lines = text.split("\n")
    block: list[int] = []  # the lines of the parameters block
    at: dict[str, int] = {}  # a frozen parameter's own line
    inside = False
    for i, line in enumerate(lines):
        words = line.partition("#")[0].split()
        if not words:
            continue
        if words[0] in ("begin", "end"):
            inside = words[:2] == ["begin", "parameters"]
            continue
        if inside and len(words) >= 3 and words[0].isdigit():
            block.append(i)
            if words[1] in reasons:
                at.setdefault(words[1], i)
    codes = []
    names = []
    for line in lines:
        code = line.partition("#")[0]
        # A block's first and last line name the block, and nothing it reads.
        marks_a_block = code.split()[:1] in (["begin"], ["end"])
        codes.append("" if marks_a_block else code)
        names.append(set() if marks_a_block else set(_names_in(code)))

    def named(n: str, i: int) -> bool:
        # A name that is no identifier (a dot in it) is looked for as it stands.
        return n in names[i] if _IDENTIFIER.fullmatch(n) else n in codes[i]

    left_out: set[str] = set()
    while True:
        gone = {at[n] for n in left_out}
        more = {
            n
            for n, own in at.items()
            if n not in left_out
            and n not in kept_for_actions
            and not any(named(n, i) for i in range(len(lines)) if i != own and i not in gone)
        }
        if not more:
            break
        left_out |= more

    gone = {at[n] for n in left_out}
    for n, i in at.items():
        if n not in left_out:
            what = " ".join(reasons[n].removesuffix(_FOLDED_READ_BACK).split())
            lines[i] = f"{lines[i]}  # {MARKER}{what}"
    number = 0
    for i in block:
        if i not in gone:
            number += 1
            lines[i] = _INDEX.sub(rf"\g<1>{number}", lines[i], count=1)
    kept = [line for i, line in enumerate(lines) if i not in gone]
    order = sorted(at, key=at.__getitem__)
    return (
        "\n".join(kept),
        [n for n in order if n in left_out],
        [n for n in order if n not in left_out],
    )


def notes(left_out: list[str], marked: list[str]) -> tuple[list[str], list[str]]:
    """``(dropped, bngsim_only)`` notes for a capability report."""
    dropped: list[str] = []
    bngsim_only: list[str] = []
    if left_out:
        dropped.append(
            f"{len(left_out)} parameter(s) folded into a number when the model was "
            f"loaded, which nothing in the written network reads ({', '.join(left_out)}): "
            "a write to one moved nothing, and is now refused as a name the model "
            "has not (Model.frozen_params of the model written)"
        )
    if marked:
        bngsim_only.append(
            f"{len(marked)} parameter(s) folded into a number when the model was "
            "loaded, which the written network or its actions still read "
            f"({', '.join(marked)}), carry a "
            f"`{MARKER}` comment: bngsim reads it and refuses a write to one, as it does "
            "on the model written, but any other reader takes the write, which moves "
            "what still reads the parameter and not the number it was folded into"
        )
    return dropped, bngsim_only
