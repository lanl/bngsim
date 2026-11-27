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
some pairing of the two names' molecules, components and bonds keeps every
name, state, compartment and bond. That pairing is searched for, not computed
from a canonical order: species are small, and a search is easy to see right.

What is no complete species matches nothing, as BNG2.pl refuses it: a bond
wildcard (``!+``, ``!?``), a state wildcard (``~?``), a bond label written once
or three times. A component or a state left out makes another graph, which no
species of the network is. A compartment written before the name is the
species' own: one that the molecules' compartments contradict (``@c:A()@d``)
finds nothing, as BNG2.pl aborts on it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

_NAME = r"[A-Za-z_]\w*"
_FIXED = re.compile(r"\s*\$")
_PREFIX = re.compile(rf"\s*@({_NAME})::?")
# A molecule: its name, what its parentheses hold, its own compartment.
_MOLECULE = re.compile(rf"\s*({_NAME})\s*(?:\(([^()]*)\))?\s*(?:@({_NAME})\s*)?")
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
# compartment, for a component its name and state ("" where there is none);
# every bond has the one label.
_Label = tuple[str, str, str]
_BOND: _Label = ("bond", "", "")
_Key = tuple[tuple[_Label, int], ...]
_Rough = tuple[tuple[str, ...], int, int]


@dataclass(frozen=True)
class SpeciesGraph:
    """A species: a node for each molecule, each component and each bond. A
    molecule is joined to its components, and a bond to the two components it
    joins (so two bonds between the same two sites are two nodes)."""

    labels: tuple[_Label, ...]
    edges: tuple[frozenset[int], ...]  # edges[i]: the nodes joined to node i
    key: _Key  # each node's label and degree, sorted: equal for one species
    prefix: str  # the compartment written before the species, "" for none


def parse(text: str) -> SpeciesGraph | None:
    """*text* as a species graph, or ``None`` if it is not one complete species."""
    pos = 0
    prefix = ""
    m = _FIXED.match(text, pos)  # a fixed species' $, before or after the compartment
    if m:
        pos = m.end()
    m = _PREFIX.match(text, pos)
    if m:
        prefix, pos = m.group(1), m.end()
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
        labels.append(("molecule", name, own or prefix))
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
    for ends in bonds.values():
        if len(ends) != 2 or ends[0] == ends[1]:
            return None  # a dangling bond, a third end, or a site bonded to itself
        bond = len(labels)
        labels.append(_BOND)
        joined.append(set(ends))
        for end in ends:
            joined[end].add(bond)
    edges = tuple(frozenset(s) for s in joined)
    key = tuple(sorted((label, len(edges[i])) for i, label in enumerate(labels)))
    return SpeciesGraph(tuple(labels), edges, key, prefix)


def _refined(
    a: SpeciesGraph, b: SpeciesGraph, ca: list[int], cb: list[int], rounds: int
) -> tuple[list[int], list[int], bool]:
    """The two graphs' colours, each node's refined *rounds* times by the
    colours next to it, in one palette for both; and whether a round changed
    nothing more. Two nodes that some pairing of the graphs matches keep one
    colour through every round, so a node may only go where its colour is."""
    for _ in range(rounds):
        sa = [(c, tuple(sorted(ca[k] for k in a.edges[i]))) for i, c in enumerate(ca)]
        sb = [(c, tuple(sorted(cb[k] for k in b.edges[i]))) for i, c in enumerate(cb)]
        palette = {s: n for n, s in enumerate(sorted(set(sa) | set(sb)))}
        na, nb = [palette[s] for s in sa], [palette[s] for s in sb]
        if len(set(na)) == len(set(ca)) and len(set(nb)) == len(set(cb)):
            return ca, cb, True
        ca, cb = na, nb
    return ca, cb, False


def _pairing(
    a: SpeciesGraph, b: SpeciesGraph, ca: list[int], cb: list[int], budget: int | None
) -> bool | None:
    """Whether some pairing of the nodes keeps every colour and every edge;
    ``None`` if *budget* places were tried without an answer."""
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
        if ca[i] != cb[j] or len(a.edges[i]) != len(b.edges[j]):
            return False
        # Joined to the images of the placed nodes it is joined to, and to no
        # other placed node: each of those images is a neighbour, and the
        # neighbours already used are no more than those.
        return all(image[k] in b.edges[j] for k in before[at]) and sum(
            1 for k in b.edges[j] if k in used
        ) == len(before[at])

    # A search with its own stack: a species of hundreds of molecules is
    # deeper than Python lets a function call itself.
    choices: list[Iterator[int]] = []
    at = 0
    while at < n:
        if at == len(choices):
            places: Iterable[int] = b.edges[image[before[at][0]]] if before[at] else range(n)
            choices.append(iter(sorted(places)))
        for j in choices[at]:
            if budget is not None:
                budget -= 1
                if budget < 0:
                    return None
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


def same_species(a: SpeciesGraph, b: SpeciesGraph) -> bool:
    """Whether *a* and *b* are one species: some pairing of the two graphs'
    nodes keeps every label and every edge."""
    if a.key != b.key:
        return False
    palette = {label: n for n, label in enumerate(sorted(set(a.labels)))}
    ca, cb = [palette[x] for x in a.labels], [palette[x] for x in b.labels]
    # The plain search first: a chain, a ring or a small species is done in a
    # few places a node. A branched species whose sites share a name can send
    # it down a wrong branch at every fork; when it has tried too many places,
    # the colours are refined, which tells such branches apart, and it starts
    # again. Colours that no round changes are as fine as they get, and the
    # search then runs to its end.
    budget, rounds, settled = 32 * len(a.labels) + 256, 8, False
    while True:
        found = _pairing(a, b, ca, cb, None if settled else budget)
        if found is not None:
            return found
        ca, cb, settled = _refined(a, b, ca, cb, rounds)
        if Counter(ca) != Counter(cb):
            return False
        budget, rounds = 2 * budget, 4 * rounds


def _rough(text: str) -> _Rough:
    """What every spelling of one species shares, read from the text alone: its
    molecules' names, and how many bond ends and states it has. It sorts a
    network's names into small bins, so that a name is compared with few
    names."""
    names = tuple(sorted(_MOLECULE_NAMES.findall(text)))
    return names, text.count("!"), text.count("~")


def _prefix_fits(graph: SpeciesGraph, other: SpeciesGraph) -> bool:
    """Whether the compartment written before *graph*'s name, if there is one,
    is the compartment of *other*, a species of the network: BNG2.pl refuses
    another. That is the one *other*'s name states before it, as BNG2.pl
    writes every species of a compartmental network; or, in a network written
    with the compartment on the molecules alone (``L(r)@EC``), the one every
    molecule of *other* is in. Molecules in two compartments and no prefix
    say nothing of the species' own, which depends on the compartments'
    dimensions: a prefix finds nothing there."""
    if graph.prefix in ("", other.prefix):
        return True
    return not other.prefix and all(
        compartment == graph.prefix for kind, _, compartment in other.labels if kind == "molecule"
    )


class SpeciesIndex:
    """The species of a network by structure, so that a name written
    another way finds the one it spells."""

    def __init__(self, names: Iterable[str]) -> None:
        self._bins: dict[_Rough, list[str]] = {}
        for name in names:
            # A network writes every species with its parentheses. A name with
            # none is an SBML id, which is no BNGL and matches only exactly.
            if "(" in name:
                self._bins.setdefault(_rough(name), []).append(name)
        # A bin's names as graphs, read when a name first falls in it.
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
                for name in self._bins.get(rough, ()):
                    other = parse(name)
                    if other is not None:
                        by_key.setdefault(other.key, []).append((name, other))
                self._graphs[rough] = by_key
            same = [
                name
                for name, other in self._graphs[rough].get(graph.key, ())
                if _prefix_fits(graph, other) and same_species(graph, other)
            ]
            # Two names of one network for one species would be a network
            # BNG2.pl does not write: neither is picked.
            found = same[0] if len(same) == 1 else None
        if len(self._found) < 4096:  # a caller's spellings are few; this is no leak
            self._found[text] = found
        return found
