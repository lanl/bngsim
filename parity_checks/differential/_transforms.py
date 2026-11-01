"""Metamorphic transforms of a model file, for the differential suite (issue #1305).

Every path ``_differential.py`` compares is built from one loaded model: one
loader, one model build, one expression parser, so a bug there makes every
path agree on the wrong answer. These transform the *file* instead, in ways
whose effect on the answer is known exactly, so the loader reads each a second
time, differently:

* ``reorder``: species and reactions in another order (a ``.net`` renumbered,
  SBML's ``listOf*`` reversed). Trajectories by name are unchanged.
* ``rename``: parameters, species and observables to fresh names. Unchanged,
  by the renaming.
* ``split``: each reaction into two identical ones at half the rate.
  Unchanged.
* ``rescale_time``: every rate times ``TIME_SCALE`` in a model that does not
  read the clock, and the window divided by it: ``x'(t) = x(c·t)``.

Each returns a :class:`Transformed`, or raises :class:`Inapplicable` where the
transform does not keep the answer for this model (a reaction whose rate is
read by name cannot be split; a model with events or the clock cannot be
rescaled). Nothing here judges: ``_differential.py`` runs the file and
compares it with the base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TRANSFORMS = ("reorder", "rename", "split", "rescale_time")

# Not a power of two, so a rescaled rate is not the base's bits shifted.
TIME_SCALE = 3.0
SPLIT_SHARE = 0.5


class Inapplicable(Exception):
    """The transform does not keep the answer for this model; the reason says why."""


@dataclass
class Transformed:
    path: Path
    # A transformed species name -> the base's name for it (absent: the same).
    names: dict[str, str] = field(default_factory=dict)
    # The window and every rate were scaled by this (1: they were not).
    time_scale: float = 1.0
    note: str = ""


def transform(kind: str, path: Path, workdir: Path) -> Transformed:
    path = Path(path)
    out = Path(workdir) / f"{kind}{path.suffix}"
    if path.suffix == ".net":
        return _NET[kind](path.read_text(), out)
    return _SBML[kind](path, out)


# ── SBML ────────────────────────────────────────────────────────────────────


def _read(path: Path):
    import libsbml

    doc = libsbml.readSBMLFromFile(str(path))
    if doc.getModel() is None:
        raise Inapplicable("libsbml reads no model from the file")
    return doc


def _write(doc, out: Path) -> Path:
    import libsbml

    if not libsbml.writeSBMLToFile(doc, str(out)):
        raise RuntimeError(f"libsbml could not write {out}")
    return out


def _elements(model):
    """Every element of the model. libsbml's getListOfAllElements leaves out a
    Level 2 stoichiometryMath (BIOMD0000000040), which reads ids like any law."""
    out = list(model.getListOfAllElements())
    for r in model.getListOfReactions():
        for sr in list(r.getListOfReactants()) + list(r.getListOfProducts()):
            if sr.isSetStoichiometryMath():
                out.append(sr.getStoichiometryMath())
    return out


def _maths(model):
    """Every math element in the model, with what holds it."""
    for el in _elements(model):
        if hasattr(el, "isSetMath") and el.isSetMath():
            yield el, el.getMath()


def _walk(node):
    yield node
    for i in range(node.getNumChildren()):
        yield from _walk(node.getChild(i))


def _names_read(model) -> set[str]:
    return {
        n.getName()
        for _, m in _maths(model)
        for n in _walk(m)
        if n.isName() and n.getName() is not None
    }


def _times(factor: float, math):
    import libsbml

    node = libsbml.ASTNode(libsbml.AST_TIMES)
    c = libsbml.ASTNode(libsbml.AST_REAL)
    c.setValue(factor)
    node.addChild(c)
    node.addChild(math.deepCopy())
    return node


def _reverse(lst) -> None:
    items = [lst.remove(0) for _ in range(lst.size())]
    for it in reversed(items):
        lst.appendAndOwn(it)


def _sbml_reorder(path: Path, out: Path) -> Transformed:
    doc = _read(path)
    m = doc.getModel()
    if m.getNumSpecies() + m.getNumReactions() < 2:
        raise Inapplicable("fewer than two species and reactions to reorder")
    for lst in (m.getListOfSpecies(), m.getListOfReactions(), m.getListOfParameters()):
        _reverse(lst)
    for r in m.getListOfReactions():
        _reverse(r.getListOfReactants())
        _reverse(r.getListOfProducts())
    return Transformed(_write(doc, out), note="species, reactions and parameters reversed")


def _fresh(prefix: str, taken: set[str]):
    i = 0
    while True:
        i += 1
        name = f"{prefix}{i}"
        if name not in taken:
            taken.add(name)
            yield name


def _sbml_rename(path: Path, out: Path) -> Transformed:
    doc = _read(path)
    m = doc.getModel()
    taken = {el.getId() for el in m.getListOfAllElements() if el.isSetId()}
    # A kinetic law's local parameter shadows a global of its name, and a
    # renamed reference inside that law would leave the local one.
    local = {
        p.getId()
        for r in m.getListOfReactions()
        if r.isSetKineticLaw()
        for p in list(r.getKineticLaw().getListOfLocalParameters())
        + list(r.getKineticLaw().getListOfParameters())
    }
    plan = []
    for lst, prefix in (
        (m.getListOfSpecies(), "mt_s"),
        (m.getListOfParameters(), "mt_p"),
        (m.getListOfCompartments(), "mt_c"),
    ):
        fresh = _fresh(prefix, taken)
        plan += [(el, el.getId(), next(fresh)) for el in lst if el.getId() not in local]
    if not plan:
        raise Inapplicable("nothing to rename")
    everything = _elements(m)
    for el, old, new in plan:
        el.setId(new)
        m.renameSIdRefs(old, new)
        for other in everything:
            other.renameSIdRefs(old, new)
    # Every id, not only the species': a Result reports a rate rule's parameter
    # as a column too.
    names = {new: old for el, old, new in plan}
    return Transformed(_write(doc, out), names=names, note=f"{len(plan)} ids renamed")


def _sbml_split(path: Path, out: Path) -> Transformed:
    doc = _read(path)
    m = doc.getModel()
    read = _names_read(m)
    taken = {el.getId() for el in m.getListOfAllElements() if el.isSetId()}
    fresh = _fresh("mt_r", taken)
    split = 0
    for r in list(m.getListOfReactions()):
        if not r.isSetKineticLaw() or not r.getKineticLaw().isSetMath():
            continue
        refs = list(r.getListOfReactants()) + list(r.getListOfProducts())
        if (
            r.getId() in read  # its rate is read by name elsewhere
            or any(sr.isSetId() for sr in refs)  # a stoichiometry read by name
            or (r.isSetFast() and r.getFast())
        ):
            continue
        twin = r.clone()
        twin.setId(next(fresh))
        for x in (r, twin):
            x.getKineticLaw().setMath(_times(SPLIT_SHARE, x.getKineticLaw().getMath()))
        m.addReaction(twin)
        split += 1
    if not split:
        raise Inapplicable("no reaction whose rate is not read by name elsewhere")
    return Transformed(_write(doc, out), note=f"{split} reactions split")


def _sbml_rescale(path: Path, out: Path) -> Transformed:
    import libsbml

    doc = _read(path)
    m = doc.getModel()
    if m.getNumEvents():
        raise Inapplicable("events: their times are not rescaled")
    clock = {libsbml.AST_NAME_TIME, libsbml.AST_FUNCTION_DELAY}
    rate_of = getattr(libsbml, "AST_FUNCTION_RATE_OF", None)
    for _, math in _maths(m):
        for n in _walk(math):
            if n.getType() in clock:
                raise Inapplicable("reads the clock or a delay")
            if rate_of is not None and n.getType() == rate_of:
                raise Inapplicable("reads a rate of change, which the rescale scales too")
    # A reaction's id in math is its rate, which the rescale has already scaled:
    # a rate rule that reads one would be scaled twice (BIOMD0000000556's APP
    # came out 9x), and a law that reads one is not linear in the factor.
    read = {r.getId() for r in m.getListOfReactions()} & _names_read(m)
    if read:
        raise Inapplicable(f"math reads a reaction's rate ({sorted(read)[0]})")
    for r in m.getListOfReactions():
        kl = r.getKineticLaw() if r.isSetKineticLaw() else None
        if kl is not None and kl.isSetMath():
            kl.setMath(_times(TIME_SCALE, kl.getMath()))
    for rule in m.getListOfRules():
        if rule.isRate():
            rule.setMath(_times(TIME_SCALE, rule.getMath()))
    return Transformed(
        _write(doc, out),
        time_scale=TIME_SCALE,
        note=f"rates x{TIME_SCALE:g}, window /{TIME_SCALE:g}",
    )


_SBML = {
    "reorder": _sbml_reorder,
    "rename": _sbml_rename,
    "split": _sbml_split,
    "rescale_time": _sbml_rescale,
}


# ── .net ────────────────────────────────────────────────────────────────────

_BLOCK = re.compile(r"^\s*begin\s+(\w+)\s*$")
_END = re.compile(r"^\s*end\s+(\w+)\s*$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A name in an expression, and not the exponent of a number: in 2e1 or 1E0 a
# parameter named e1 or E0 is no part of the literal.
_NAME = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class _Net:
    """A ``.net`` file as its blocks: ``parts`` interleaves verbatim text (str)
    and block names (a 1-tuple), ``blocks`` maps a name to its entry lines."""

    parts: list
    blocks: dict[str, list[str]]

    @classmethod
    def parse(cls, text: str) -> _Net:
        parts, blocks, cur = [], {}, None
        for line in text.splitlines():
            if cur is None:
                m = _BLOCK.match(line)
                if m:
                    cur = m.group(1)
                    blocks[cur] = []
                    parts.append((cur,))
                else:
                    parts.append(line)
            else:
                m = _END.match(line)
                if m and m.group(1) == cur:
                    cur = None
                else:
                    blocks[cur].append(line)
        return cls(parts, blocks)

    def text(self) -> str:
        out = []
        for p in self.parts:
            if isinstance(p, tuple):
                out += [f"begin {p[0]}", *self.blocks[p[0]], f"end {p[0]}"]
            else:
                out.append(p)
        return "\n".join(out) + "\n"

    def entries(self, block: str) -> list[tuple[list[str], str]]:
        """``(fields, comment)`` for each non-blank line of ``block``."""
        out = []
        for line in self.blocks.get(block, []):
            body, _, comment = line.partition("#")
            fields = body.split()
            if fields:
                out.append((fields, comment))
        return out

    def set_entries(self, block: str, rows: list[tuple[list[str], str]]) -> None:
        self.blocks[block] = ["    " + " ".join(f) + (f" #{c}" if c else "") for f, c in rows]


def _subst(expr: str, names: dict[str, str]) -> str:
    return _NAME.sub(lambda m: names.get(m.group(0), m.group(0)), expr)


def _net_reorder(text: str, out: Path) -> Transformed:
    net = _Net.parse(text)
    species = net.entries("species")
    if len(species) < 2:
        raise Inapplicable("fewer than two species to reorder")
    n = len(species)
    new_of = {int(f[0]): n - k for k, (f, _) in enumerate(species)}

    def remap(lst: str) -> str:
        if lst == "0":
            return lst
        return ",".join(str(new_of[int(i)]) for i in lst.split(","))

    net.set_entries(
        "species", [([str(new_of[int(f[0])]), *f[1:]], c) for f, c in reversed(species)]
    )
    reactions = net.entries("reactions")
    net.set_entries(
        "reactions",
        [
            ([str(k + 1), remap(f[1]), remap(f[2]), *f[3:]], c)
            for k, (f, c) in enumerate(reversed(reactions))
        ],
    )
    groups = []
    for f, c in net.entries("groups"):
        members = []
        for term in f[2].split(",") if len(f) > 2 else []:
            w, star, i = term.rpartition("*")
            members.append(f"{w}{star}{new_of[int(i)]}")
        groups.append(([f[0], f[1], ",".join(members), *f[3:]], c))
    net.set_entries("groups", groups)
    out.write_text(net.text())
    return Transformed(out, note=f"{n} species and {len(reactions)} reactions reversed")


def _net_rename(text: str, out: Path) -> Transformed:
    net = _Net.parse(text)
    params = [f[1] for f, _ in net.entries("parameters")]
    groups = [f[1] for f, _ in net.entries("groups")]
    taken = set(_IDENT.findall(text))
    names = {}
    for old, new in zip(params, _fresh("mt_p", taken), strict=False):
        names[old] = new
    for old, new in zip(groups, _fresh("mt_g", taken), strict=False):
        names[old] = new
    for old, new in zip(sorted(_function_names(net)), _fresh("mt_f", taken), strict=False):
        names[old] = new
    if not names:
        raise Inapplicable("nothing to rename")
    # From the name on: a value is an expression, and may hold spaces
    # (ic_derived_compound: if((sel>=1)&&(sel<10), R0, 0.5*R0)).
    for block, start in (("parameters", 1), ("groups", 1), ("species", 2)):
        rows = []
        for f, c in net.entries(block):
            rows.append(([*f[:start], *(_subst(x, names) for x in f[start:])], c))
        net.set_entries(block, rows)
    rows = []
    for f, c in net.entries("reactions"):
        rows.append(([*f[:3], *(_subst(x, names) for x in f[3:])], c))
    net.set_entries("reactions", rows)
    rows = []
    for f, c in net.entries("functions"):
        rows.append(([f[0], *(_subst(x, names) for x in f[1:])], c))
    net.set_entries("functions", rows)
    # Species names are labels (no expression reads one), so they are renamed
    # too, where they carry no compartment the name itself declares.
    species = net.entries("species")
    species_names = {}
    if not any("@" in f[1] for f, _ in species):
        rows = []
        for k, (f, c) in enumerate(species):
            fixed = "$" if f[1].startswith("$") else ""
            new = f"{fixed}MtS{k + 1}()"
            species_names[new.lstrip("$")] = f[1].lstrip("$")
            rows.append(([f[0], new, *f[2:]], c))
        net.set_entries("species", rows)
    out.write_text(net.text())
    return Transformed(
        out, names=species_names, note=f"{len(names)} names, {len(species_names)} species"
    )


# BNG rate laws written as a keyword and its arguments, each linear in its first
# (MM kcat Km is kcat·E·S/(Km + S); Sat k K and Hill k K n likewise in k).
LINEAR_IN_FIRST = ("MM", "Sat", "Hill")


def _scaled_rate(net: _Net, f: list[str], name: str, factor: float) -> list[str] | None:
    """The reaction's fields with its rate scaled by ``factor`` through a new
    parameter ``name``, or None where the rate is not one that can be."""
    if f[3] in LINEAR_IN_FIRST:
        if len(f) < 5:
            return None
        _net_new_param(net, name, f"{factor!r}*({f[4]})")
        return [*f[:4], name, *f[5:]]
    _net_new_param(net, name, f"{factor!r}*({' '.join(f[3:])})")
    return [*f[:3], name]


def _function_names(net: _Net) -> set[str]:
    return {f[1].split("(")[0] for f, _ in net.entries("functions")}


def _net_new_param(net: _Net, name: str, expr: str) -> None:
    rows = net.entries("parameters")
    rows.append(([str(len(rows) + 1), name, expr], ""))
    net.set_entries("parameters", rows)


def _net_split(text: str, out: Path) -> Transformed:
    net = _Net.parse(text)
    funcs = _function_names(net)
    taken = set(_IDENT.findall(text))
    fresh = _fresh("mt_split", taken)
    halves: dict[str, str] = {}
    rows, split = [], 0
    for f, c in net.entries("reactions"):
        rate = " ".join(f[3:])
        if set(_IDENT.findall(rate)) & funcs:
            if rate not in funcs:  # a function inside an expression: left whole
                rows.append((list(f), c))
                continue
            # A function's rate: halved by a function that calls it.
            name = halves.setdefault(rate, next(fresh))
            rows += [([*f[:3], name], c), ([*f[:3], name], c)]
        else:
            scaled = _scaled_rate(net, list(f), next(fresh), SPLIT_SHARE)
            if scaled is None:
                rows.append((list(f), c))
                continue
            rows += [(scaled, c), (list(scaled), c)]
        split += 1
    if not split:
        raise Inapplicable("every reaction's rate is a function inside an expression")
    net.set_entries("reactions", [([str(k + 1), *f[1:]], c) for k, (f, c) in enumerate(rows)])
    if halves:
        frows = net.entries("functions")
        for old, new in halves.items():
            frows.append(([str(len(frows) + 1), f"{new}()", f"{SPLIT_SHARE!r}*{old}()"], ""))
        net.set_entries("functions", frows)
    out.write_text(net.text())
    return Transformed(out, note=f"{split} reactions split")


def _net_rescale(text: str, out: Path) -> Transformed:
    net = _Net.parse(text)
    if "time" in set(_IDENT.findall(text)) or net.entries("tfuns") or "tfun" in text:
        raise Inapplicable("reads the clock")
    funcs = _function_names(net)
    taken = set(_IDENT.findall(text))
    fresh = _fresh("mt_scaled", taken)
    fresh_f = _fresh("mt_scaledf", taken)
    scaled_f: dict[str, str] = {}
    rows = []
    for f, c in net.entries("reactions"):
        rate = " ".join(f[3:])
        used = set(_IDENT.findall(rate)) & funcs
        if used:
            if rate not in funcs:
                raise Inapplicable(f"a rate {rate!r} that is a function within an expression")
            if rate not in scaled_f:
                scaled_f[rate] = next(fresh_f)
            rows.append(([*f[:3], scaled_f[rate]], c))
            continue
        scaled = _scaled_rate(net, list(f), next(fresh), TIME_SCALE)
        if scaled is None:
            raise Inapplicable(f"a rate {rate!r} this cannot scale")
        rows.append((scaled, c))
    net.set_entries("reactions", rows)
    if scaled_f:
        frows = net.entries("functions")
        for old, new in scaled_f.items():
            frows.append(([str(len(frows) + 1), f"{new}()", f"{TIME_SCALE!r}*{old}()"], ""))
        net.set_entries("functions", frows)
    out.write_text(net.text())
    return Transformed(
        out, time_scale=TIME_SCALE, note=f"rates x{TIME_SCALE:g}, window /{TIME_SCALE:g}"
    )


_NET = {
    "reorder": _net_reorder,
    "rename": _net_rename,
    "split": _net_split,
    "rescale_time": _net_rescale,
}
