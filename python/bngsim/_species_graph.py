"""A BNGL species as a graph, to find the species of a network that a name spells.

BNG2.pl's ``setConcentration("A(y,x)", 50)`` reads its first argument as one
complete species and sets the species of the network that is the same graph,
however the name writes it: the components of a molecule in another order, the
molecules of a complex in another order, other bond labels, the bond of a
symmetric dimer on the other of two equal sites, a compartment as ``@c:A()``,
``@c::A()`` or ``A()@c``. A network names each species one way, so a name
written another way was "not found".

A species is read here as its molecules, each with a compartment and its
components, and the bonds between components. Two names spell one species when
some pairing of the two names' molecules and components keeps every name, state,
compartment and bond. That pairing is searched for, not computed from a
canonical order: species are small, and a search is easy to see right.

What is no complete species matches nothing, as BNG2.pl refuses it: a bond
wildcard (``!+``, ``!?``), a state wildcard (``~?``), a bond label written once
or three times. A component or a state left out makes another graph, which no
species of the network is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_NAME = r"[A-Za-z_]\w*"
_FIXED = re.compile(r"\s*\$")
_PREFIX = re.compile(rf"\s*@\s*({_NAME})\s*::?")
# A molecule: its name, what its parentheses hold, its own compartment.
_MOLECULE = re.compile(rf"\s*({_NAME})\s*(?:\(([^()]*)\))?\s*(?:@\s*({_NAME})\s*)?")
# A component: its name, then states and bonds. BNGL writes a state and a bond
# label as words or numbers (``~P``, ``~0``, ``!1``); a wildcard is neither.
_COMPONENT = re.compile(rf"\s*({_NAME})\s*((?:[~!]\s*\w+\s*)*)")
_MARK = re.compile(r"([~!])\s*(\w+)")
_DOT = re.compile(r"\s*\.")
# The molecules' names, straight from the text: each follows the start of the
# name, a ``.``, the ``:`` of a compartment or a ``$``. A component follows a
# ``(`` or a ``,``, and a compartment an ``@``.
_MOLECULE_NAMES = re.compile(rf"(?:^|[.:$])\s*({_NAME})")

# What must be equal for two nodes to be paired: for a molecule its name and
# compartment, for a component its name and state ("" where there is none).
_Label = tuple[str, str, str]
_Key = tuple[tuple[_Label, int], ...]
_Rough = tuple[tuple[str, ...], int, int]


@dataclass(frozen=True)
class SpeciesGraph:
    """A species: a node for each molecule and each component. A molecule is
    joined to its components, and a component to the one it is bonded to."""

    labels: tuple[_Label, ...]
    edges: tuple[frozenset[int], ...]  # edges[i]: the nodes joined to node i
    key: _Key  # each node's label and degree, sorted: equal for one species


def parse(text: str) -> SpeciesGraph | None:
    """*text* as a species graph, or ``None`` if it is not one complete species."""
    pos = 0
    compartment = ""
    m = _FIXED.match(text, pos)  # a fixed species' $, before or after the compartment
    if m:
        pos = m.end()
    m = _PREFIX.match(text, pos)
    if m:
        compartment, pos = m.group(1), m.end()
        m = _FIXED.match(text, pos)
        if m:
            pos = m.end()

    labels: list[_Label] = []
    joined: list[set[int]] = []
    bonds: dict[str, list[int]] = {}
    while True:
        m = _MOLECULE.match(text, pos)
        if not m:
            return None
        name, inside, own = m.groups()
        pos = m.end()
        molecule = len(labels)
        labels.append(("molecule", name, own or compartment))
        joined.append(set())
        for part in inside.split(",") if inside else ():
            if not part.strip():
                continue  # A() and, as BNG2.pl and the replay always took it, A(x,,y)
            c = _COMPONENT.fullmatch(part)
            if not c:
                return None  # a wildcard, or anything else that is no component
            state = ""
            component = len(labels)
            for mark, word in _MARK.findall(c.group(2)) if c.group(2) else ():
                if mark == "!":
                    bonds.setdefault(word, []).append(component)
                elif state:
                    return None  # two states
                else:
                    state = word
            labels.append(("component", c.group(1), state))
            joined.append({molecule})
            joined[molecule].add(component)
        m = _DOT.match(text, pos)
        if not m:
            break
        pos = m.end()
    if text[pos:].strip():
        return None
    for pair in bonds.values():
        if len(pair) != 2 or pair[0] == pair[1] or pair[1] in joined[pair[0]]:
            return None  # a dangling bond, a third end, or one pair bonded twice
        joined[pair[0]].add(pair[1])
        joined[pair[1]].add(pair[0])
    edges = tuple(frozenset(s) for s in joined)
    key = tuple(sorted((label, len(edges[i])) for i, label in enumerate(labels)))
    return SpeciesGraph(tuple(labels), edges, key)


def same_species(a: SpeciesGraph, b: SpeciesGraph) -> bool:
    """Whether *a* and *b* are one species: some pairing of the two graphs' nodes keeps
    every label and every edge."""
    if a.key != b.key:
        return False
    n = len(a.labels)
    # Breadth first, so that each node after the first of its connected piece
    # is joined to one already placed, which leaves it few places to go.
    order: list[int] = []
    seen: set[int] = set()
    for start in range(n):
        if start in seen:
            continue
        seen.add(start)
        queue = [start]
        for i in queue:  # grows as it is read
            order.append(i)
            for j in sorted(a.edges[i]):
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
    # For each node, the nodes placed before it that it is joined to. The
    # neighbours of the image of one such node are the only places it can go.
    position = {i: at for at, i in enumerate(order)}
    before = [sorted(k for k in a.edges[i] if position[k] < at) for at, i in enumerate(order)]
    image: dict[int, int] = {}
    used: set[int] = set()

    def fits(at: int, j: int) -> bool:
        i = order[at]
        if a.labels[i] != b.labels[j] or len(a.edges[i]) != len(b.edges[j]):
            return False
        # Joined to the images of the placed nodes it is joined to, and to no
        # other placed node: each of those images is a neighbour, and the
        # neighbours already used are no more than those.
        return all(image[k] in b.edges[j] for k in before[at]) and sum(
            1 for k in b.edges[j] if k in used
        ) == len(before[at])

    # A search with its own stack: a species of hundreds of molecules is
    # deeper than Python lets a function call itself.
    choices: list[Iterable[int]] = []
    at = 0
    while at < n:
        if at == len(choices):
            places = b.edges[image[before[at][0]]] if before[at] else range(n)
            choices.append(iter(sorted(places)))
        for j in choices[at]:
            if j not in used and fits(at, j):
                image[order[at]] = j
                used.add(j)
                at += 1
                break
        else:  # nowhere left for this node: move the one before it
            choices.pop()
            if at == 0:
                return False
            at -= 1
            used.discard(image.pop(order[at]))
    return True


def _rough(text: str) -> _Rough:
    """What every spelling of one species shares, read from the text alone: its
    molecules' names, and how many bond ends and states it has. It sorts a
    network's names into small heaps, so that a name is compared with few
    names."""
    names = tuple(sorted(_MOLECULE_NAMES.findall(text)))
    return names, text.count("!"), text.count("~")


class SpeciesIndex:
    """The species of a network by structure, so that a name written
    another way finds the one it spells."""

    def __init__(self, names: Iterable[str]) -> None:
        self._heaps: dict[_Rough, list[str]] = {}
        for name in names:
            # A network writes every species with its parentheses. A name with
            # none is an SBML id, which is no BNGL and matches only exactly.
            if "(" in name:
                self._heaps.setdefault(_rough(name), []).append(name)
        # A heap's names as graphs, read when a name first falls in it.
        self._graphs: dict[_Rough, dict[_Key, list[tuple[str, SpeciesGraph]]]] = {}
        self._found: dict[str, str | None] = {}

    def find(self, text: str) -> str | None:
        """The network's name for the species *text* spells; ``None`` if it
        spells no species of the network, or is no complete species."""
        if text in self._found:
            return self._found[text]
        graph = parse(text)
        found = None
        if graph is not None:
            rough = _rough(text)
            if rough not in self._graphs:
                by_key: dict[_Key, list[tuple[str, SpeciesGraph]]] = {}
                for name in self._heaps.get(rough, ()):
                    other = parse(name)
                    if other is not None:
                        by_key.setdefault(other.key, []).append((name, other))
                self._graphs[rough] = by_key
            alike = self._graphs[rough].get(graph.key, ())
            same = [name for name, other in alike if same_species(graph, other)]
            # Two names of one network for one species would be a network
            # BNG2.pl does not write: neither is picked.
            found = same[0] if len(same) == 1 else None
        if len(self._found) < 4096:  # a caller's spellings are few; this is no leak
            self._found[text] = found
        return found
