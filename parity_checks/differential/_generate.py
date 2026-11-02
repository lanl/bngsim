"""Generated models for the differential suite, and a shrinker for what they find
(issue #1306).

``_differential.py`` checks fixed models: the corpus, and shapes written by
hand. Its checks are oracles for whole classes of bug (``cc_vs_interp`` above
all: the compiled C against the ExprTk interpreter), but only over the
expressions those models happen to hold. This draws models instead, from the
grammar bngsim accepts, as Csmith and EMI do for C compilers:

* :func:`generate` draws a small reaction network whose rate laws are random
  expression trees, seeded and reproducible, rendered as a ``.net`` or as
  SBML;
* :func:`shrink` delta-debugs a model that a check finds something in: it
  drops reactions and replaces subexpressions by their children or by
  constants while the same check still finds the same thing, and returns the
  smallest model it reached.

The laws are kept finite by construction, so a finding is a disagreement
between paths and not a domain error: a logarithm or a square root reads
``1 + x²``, a power has a base of at least 1, a quotient a denominator of at
least 1, an exponential a non-positive argument; and each law ``g`` enters its
reaction as ``k·g²/(1 + g²)``, which is bounded and still reads every node of
``g``. The guards are squares, not ``abs()``: bngsim declines the analytic
sensitivity RHS for a law that holds ``abs()`` under a call, and a guard in
every law would keep the sensitivity checks off the analytic path everywhere.
``abs`` is in the grammar as itself.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

FORMATS = ("net", "sbml")


@dataclass
class Node:
    """An expression: ``op`` is a grammar entry's name or a leaf's kind
    (``num``, ``par``, ``obs``, ``time``, ``rateof``); a leaf's ``value`` is
    its number or the index it names."""

    op: str
    args: list[Node] = field(default_factory=list)
    value: float | int | None = None


@dataclass
class Reaction:
    reactants: list[int]
    products: list[int]
    law: Node


@dataclass
class Model:
    fmt: str
    species: list[float]  # initial amounts, S1..Sn
    params: list[float]  # k1..km; the first len(reactions) are the reactions' scales
    reactions: list[Reaction]
    # SBML only: a species driven by mass action alone, whose rateOf the laws
    # may read (a law reading its own species' rateOf would be an algebraic loop).
    driver: bool = False


# ── The grammar ─────────────────────────────────────────────────────────────
#
# Each entry: arity, its .net spelling and its SBML (L3 formula) spelling, as
# functions of the rendered arguments; None where the format has no such
# function. Guards are part of the spelling, so a shrunk node keeps them.

_CMP = (">", "<", ">=", "<=", "==", "!=")


def _bin(op):
    return lambda a, b: f"({a} {op} {b})"


GRAMMAR: dict[str, tuple] = {
    "add": (2, _bin("+"), _bin("+")),
    "sub": (2, _bin("-"), _bin("-")),
    "mul": (2, _bin("*"), _bin("*")),
    "div": (2, lambda a, b: f"({a} / (1 + ({b})^2))", lambda a, b: f"({a} / (1 + ({b})^2))"),
    "pow2": (1, lambda a: f"((1 + ({a})^2)^2)", lambda a: f"((1 + ({a})^2)^2)"),
    "powh": (1, lambda a: f"((1 + ({a})^2)^0.5)", lambda a: f"((1 + ({a})^2)^0.5)"),
    "powm": (1, lambda a: f"((1 + ({a})^2)^-1.5)", lambda a: f"((1 + ({a})^2)^-1.5)"),
    "exp": (1, lambda a: f"exp(-({a})^2)", lambda a: f"exp(-({a})^2)"),
    "log": (1, lambda a: f"log(1 + ({a})^2)", lambda a: f"ln(1 + ({a})^2)"),
    "ln": (1, lambda a: f"ln(1 + ({a})^2)", None),
    "log10": (1, lambda a: f"log10(1 + ({a})^2)", lambda a: f"log10(1 + ({a})^2)"),
    "log2": (1, lambda a: f"log2(1 + ({a})^2)", None),
    "sqrt": (1, lambda a: f"sqrt(1 + ({a})^2)", lambda a: f"sqrt(1 + ({a})^2)"),
    "abs": (1, lambda a: f"abs({a})", lambda a: f"abs({a})"),
    "floor": (1, lambda a: f"floor({a})", lambda a: f"floor({a})"),
    "ceil": (1, lambda a: f"ceil({a})", lambda a: f"ceil({a})"),
    "round": (1, lambda a: f"round({a})", None),
    "rint": (1, lambda a: f"rint({a})", None),
    "trunc": (1, lambda a: f"trunc({a})", None),
    "sign": (1, lambda a: f"sign({a})", None),
    "tanh": (1, lambda a: f"tanh({a})", lambda a: f"tanh({a})"),
    "sin": (1, lambda a: f"sin({a})", lambda a: f"sin({a})"),
    "cos": (1, lambda a: f"cos({a})", lambda a: f"cos({a})"),
    "atan": (1, lambda a: f"atan({a})", lambda a: f"arctan({a})"),
    "erf": (1, lambda a: f"erf({a})", None),
    "min": (2, lambda a, b: f"min({a}, {b})", lambda a, b: f"min({a}, {b})"),
    "max": (2, lambda a, b: f"max({a}, {b})", lambda a, b: f"max({a}, {b})"),
    "avg": (2, lambda a, b: f"avg({a}, {b})", None),
    "clamp": (1, lambda a: f"clamp(0.5, {a}, 2.5)", None),
    **{
        f"if{op}": (
            4,
            (lambda o: lambda a, b, c, d: f"if({a} {o} {b}, {c}, {d})")(op),
            (lambda o: lambda a, b, c, d: f"piecewise({c}, {a} {o} {b}, {d})")(op),
        )
        for op in _CMP
    },
    "ifand": (
        4,
        lambda a, b, c, d: f"if(({a} > 1) && ({b} < 2), {c}, {d})",
        lambda a, b, c, d: f"piecewise({c}, ({a} > 1) && ({b} < 2), {d})",
    ),
    "ifor": (
        4,
        lambda a, b, c, d: f"if(({a} > 1) || ({b} < 2), {c}, {d})",
        lambda a, b, c, d: f"piecewise({c}, ({a} > 1) || ({b} < 2), {d})",
    ),
}

_NUMS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 0.25, 2.5)


def _ops(fmt: str) -> list[str]:
    col = 1 if fmt == "net" else 2
    return [name for name, e in GRAMMAR.items() if e[col] is not None]


def _leaf(rng: random.Random, model: Model) -> Node:
    kinds = ["num", "par", "obs", "obs", "time"]
    if model.fmt == "sbml" and model.driver:
        kinds.append("rateof")
    kind = rng.choice(kinds)
    if kind == "num":
        return Node("num", value=rng.choice(_NUMS))
    if kind == "par":
        return Node("par", value=rng.randrange(len(model.params)))
    if kind == "obs":
        return Node("obs", value=rng.randrange(len(model.species)))
    return Node(kind)


def _tree(rng: random.Random, model: Model, depth: int) -> Node:
    if depth == 0 or rng.random() < 0.25:
        return _leaf(rng, model)
    op = rng.choice(_ops(model.fmt))
    return Node(op, [_tree(rng, model, depth - 1) for _ in range(GRAMMAR[op][0])])


def generate(seed: int, fmt: str = "net", depth: int = 3) -> Model:
    """A random network: 2 to 4 species, 1 to 4 reactions with a law each, all
    drawn from ``random.Random(seed)``, so a seed is its model."""
    rng = random.Random(seed)
    n_sp = rng.randint(2, 4)
    n_rx = rng.randint(1, 4)
    model = Model(
        fmt=fmt,
        species=[round(rng.uniform(0.5, 5.0), 3) for _ in range(n_sp)],
        params=[round(rng.uniform(0.1, 2.0), 3) for _ in range(n_rx + 2)],
        reactions=[],
        driver=fmt == "sbml" and rng.random() < 0.5,
    )
    for _ in range(n_rx):
        # No species on both sides, and none twice on one: mass action then
        # means the same in a .net and in SBML.
        pool = list(range(n_sp))
        rng.shuffle(pool)
        k_in, k_out = rng.choice([(0, 1), (1, 0), (1, 1), (2, 1), (1, 2)])
        reactants, products = pool[:k_in], pool[k_in : k_in + k_out]
        model.reactions.append(Reaction(reactants, products, _tree(rng, model, depth)))
    return model


# ── Rendering ───────────────────────────────────────────────────────────────


def render_expr(node: Node, fmt: str) -> str:
    if node.op == "num":
        return repr(float(node.value))
    if node.op == "par":
        return f"k{node.value + 1}"
    if node.op == "obs":
        return f"O{node.value + 1}" if fmt == "net" else f"S{node.value + 1}"
    if node.op == "time":
        return "time()" if fmt == "net" else "time"
    if node.op == "rateof":
        return "rateOf(D)"
    spell = GRAMMAR[node.op][1 if fmt == "net" else 2]
    return spell(*(render_expr(a, fmt) for a in node.args))


def _rate(model: Model, r: int) -> str:
    g = render_expr(model.reactions[r].law, model.fmt)
    return f"k{r + 1}*({g})^2/(1 + ({g})^2)"


def render_net(model: Model) -> str:
    lines = ["begin parameters"]
    lines += [f"    {i + 1} k{i + 1} {v!r}" for i, v in enumerate(model.params)]
    lines += ["end parameters", "begin species"]
    lines += [f"    {i + 1} S{i + 1}() {v!r}" for i, v in enumerate(model.species)]
    lines += ["end species", "begin functions"]
    lines += [f"    {r + 1} f{r + 1}() {_rate(model, r)}" for r in range(len(model.reactions))]
    lines += ["end functions", "begin reactions"]
    for r, rx in enumerate(model.reactions):
        side = lambda ix: ",".join(str(i + 1) for i in ix) or "0"  # noqa: E731
        lines.append(f"    {r + 1} {side(rx.reactants)} {side(rx.products)} f{r + 1}")
    lines += ["end reactions", "begin groups"]
    lines += [f"    {i + 1} O{i + 1} {i + 1}" for i in range(len(model.species))]
    lines += ["end groups"]
    return "\n".join(lines) + "\n"


def render_sbml(model: Model) -> str:
    import libsbml

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId("generated")
    c = m.createCompartment()
    c.setId("C")
    c.setSize(1.0)
    c.setConstant(True)
    names = [f"S{i + 1}" for i in range(len(model.species))] + (["D"] if model.driver else [])
    inits = list(model.species) + ([1.0] if model.driver else [])
    for sid, v in zip(names, inits, strict=True):
        s = m.createSpecies()
        s.setId(sid)
        s.setCompartment("C")
        s.setInitialConcentration(v)
        s.setHasOnlySubstanceUnits(False)
        s.setBoundaryCondition(False)
        s.setConstant(False)
    for i, v in enumerate(model.params):
        p = m.createParameter()
        p.setId(f"k{i + 1}")
        p.setValue(v)
        p.setConstant(True)

    def reaction(rid, reactants, products, law):
        r = m.createReaction()
        r.setId(rid)
        r.setReversible(False)
        for sid in reactants:
            sr = r.createReactant()
            sr.setSpecies(sid)
            sr.setStoichiometry(1.0)
            sr.setConstant(True)
        for sid in products:
            sr = r.createProduct()
            sr.setSpecies(sid)
            sr.setStoichiometry(1.0)
            sr.setConstant(True)
        math = libsbml.parseL3Formula(law)
        if math is None:
            raise ValueError(f"libsbml cannot parse {law!r}: {libsbml.getLastParseL3Error()}")
        r.createKineticLaw().setMath(math)

    for r, rx in enumerate(model.reactions):
        # A .net's functional rate is multiplied by its reactants' amounts;
        # an SBML kinetic law is the whole flux.
        mass = "".join(f" * S{i + 1}" for i in rx.reactants)
        reaction(
            f"R{r + 1}",
            [f"S{i + 1}" for i in rx.reactants],
            [f"S{i + 1}" for i in rx.products],
            f"C * {_rate(model, r)}{mass}",
        )
    if model.driver:
        n = len(model.params)
        reaction("Din", [], ["D"], f"C * k{n}")
        reaction("Dout", ["D"], [], f"C * k{n - 1} * D")
    return libsbml.writeSBMLToString(doc)


def write(model: Model, path: Path) -> Path:
    path = Path(path).with_suffix(".net" if model.fmt == "net" else ".xml")
    path.write_text(render_net(model) if model.fmt == "net" else render_sbml(model))
    return path


# ── Shrinking ───────────────────────────────────────────────────────────────


def _nodes(node: Node, path=()):
    yield path, node
    for i, a in enumerate(node.args):
        yield from _nodes(a, (*path, i))


def _replace(law: Node, path: tuple, new: Node) -> Node:
    if not path:
        return new
    law = copy.deepcopy(law)
    at = law
    for i in path[:-1]:
        at = at.args[i]
    at.args[path[-1]] = new
    return law


def _candidates(model: Model):
    """Smaller models, the largest cut first."""
    if len(model.reactions) > 1:
        for r in range(len(model.reactions)):
            m = copy.deepcopy(model)
            del m.reactions[r]
            yield m
    for r, rx in enumerate(model.reactions):
        for path, node in _nodes(rx.law):
            if node.op in ("num", "par", "obs", "time", "rateof"):
                if node.op != "num":
                    m = copy.deepcopy(model)
                    m.reactions[r].law = _replace(rx.law, path, Node("num", value=1.0))
                    yield m
                continue
            for child in node.args:
                m = copy.deepcopy(model)
                m.reactions[r].law = _replace(rx.law, path, copy.deepcopy(child))
                yield m
            m = copy.deepcopy(model)
            m.reactions[r].law = _replace(rx.law, path, Node("num", value=1.0))
            yield m


def size(model: Model) -> int:
    return sum(len(list(_nodes(rx.law))) for rx in model.reactions) + len(model.reactions)


def _named_leaves(model: Model) -> int:
    return sum(
        1
        for rx in model.reactions
        for _, n in _nodes(rx.law)
        if n.op in ("par", "obs", "time", "rateof")
    )


def _smaller(a: Model, b: Model) -> bool:
    """``a`` is smaller than ``b``: fewer nodes, or as many and fewer leaves
    that name something (a parameter, a species, the clock), which a number in
    its place leaves the same size."""
    return (size(a), _named_leaves(a)) < (size(b), _named_leaves(b))


def shrink(model: Model, holds: Callable[[Model], bool], budget: int = 400) -> Model:
    """The smallest model, by :func:`size` and then by the leaves that name
    something, that ``holds`` still holds of, reached by taking the first
    candidate cut that keeps it, until no cut does or ``budget`` evaluations
    are spent. ``holds(model)`` is assumed true."""
    spent = 0
    while spent < budget:
        for cand in _candidates(model):
            spent += 1
            if spent > budget:
                break
            if _smaller(cand, model) and holds(cand):
                model = cand
                break
        else:
            return model
    return model
