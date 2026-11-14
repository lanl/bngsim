"""write_sbml keeps ExprTk's grouping of a comparison of a comparison
(issue #1499).

ExprTk, and with it bngsim, BNG2.pl and run_network, puts all six relationals
on one left-associative level and reads ``(0<x)<1`` as a comparison of the
truth value of ``0<x``. libsbml's ``parseL3Formula`` drops the brackets around
a relational's relational operand: ``(0<x)<1`` came back as ``lt(0, x, 1)``,
that is ``0<x && x<1``, and ``(x>1)==0`` as ``(x>1) && (1==0)``. BNG2.pl writes
both forms, fully bracketed, from ``if(0 < x < 1, k, 0)`` and the "not" idiom
``if((x > 1) == 0, k, 0)``; the SBML written from that ``.net`` encoded other
conditions, and nothing was said. The writer now makes ExprTk's grouping
explicit and writes a truth value a relational compares as the number it is,
``piecewise(1, cond, 0)``, which libsbml keeps.

The ``.net`` files are written by hand with the text BNG2.pl writes (checked
with BNG2.pl 2.5.1; the issue's 2.9.3 writes the same), so no test needs
BNG2.pl. Each oracle is the ExprTk reading written out in Python with explicit
grouping, and values are chosen so the flat reading differs. Truth values are
exact; trajectories run at rtol 1e-10 / atol 1e-12 and are compared with
their closed forms at rel 1e-7.
"""

from __future__ import annotations

import itertools
import math
import re
import warnings

import bngsim
import libsbml
import numpy as np
import pytest
from bngsim.convert import net_to_sbml, write_sbml
from bngsim.convert._sbml_writer import _exprtk_to_ast, _normalize_exprtk

TOL = {"rtol": 1e-10, "atol": 1e-12}
REL = 1e-7


def load_net(tmp_path, text, name="m.net"):
    path = tmp_path / name
    path.write_text(text)
    return path, bngsim.Model.from_net(str(path))


def from_sbml(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Model.from_sbml_string(text)


def at(model, names, t1):
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, t1), n_points=2, **TOL)
    return [float(np.asarray(r.observables[n])[-1]) for n in names]


# ── The issue's functions, as BNG2.pl writes them ────────────────────────────

BNG2_FUNCTIONS = """begin parameters
    1 x 0.5
    2 k 1
end parameters
begin functions
    1 f1() if(((0<x)<1),k,0)
    2 f2() if(((x>1)==0),k,0)
end functions
begin species
    1 A() 10
    2 B() 10
end species
begin reactions
    1 1 0 f1
    2 2 0 f2
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""


def test_bng2_pls_bracketed_functions_keep_their_grouping(tmp_path):
    """f1 = ((0<0.5)<1) = (1<1) = 0, so A stays 10; f2 = ((0.5>1)==0) = 1, so
    B(1) = 10/e. The written SBML gave A(1) = 3.679 and B(1) = 10. (A run
    carries the model's state forward, so the SBML is written from a model
    that has not run.)"""
    path, model = load_net(tmp_path, BNG2_FUNCTIONS)
    text = write_sbml(model)
    doc = libsbml.readSBMLFromString(text)  # held: the rules live in it
    rules = {
        r.getVariable(): libsbml.formulaToL3String(r.getMath())
        for r in doc.getModel().getListOfRules()
    }
    assert rules["f1"] == "piecewise(k, piecewise(1, 0 < x, 0) < 1, 0)"
    assert rules["f2"] == "piecewise(k, piecewise(1, x > 1, 0) == 0, 0)"
    assert at(model, ["Atot", "Btot"], 1.0) == pytest.approx([10.0, 10 / math.e], rel=REL)
    assert at(from_sbml(text), ["A", "B"], 1.0) == pytest.approx([10.0, 10 / math.e], rel=REL)


def test_a_band_on_an_observable_the_l1_gate_passed(tmp_path):
    """f = ((0<Atot)<20) is 1 at every A, so A grows by k = 10: A(2) = 30. The
    written SBML read 0<A<20 and stopped at 20, with net_to_sbml's two-state
    gate reporting ok."""
    net = """begin parameters
    1 k 10
end parameters
begin functions
    1 f() if(((0<Atot)<20),k,0)
end functions
begin species
    1 A() 10
end species
begin reactions
    1 0 1 f
end reactions
begin groups
    1 Atot 1
end groups
"""
    path, _ = load_net(tmp_path, net)
    report = net_to_sbml(path, tmp_path / "m.xml")
    assert report.ok
    written = bngsim.Model.from_sbml(str(tmp_path / "m.xml"))
    assert at(written, ["A"], 2.0) == pytest.approx([30.0], rel=REL)


# ── Every relational chain, as ExprTk groups it ──────────────────────────────

# name: (ExprTk text, its ExprTk reading, the flat reading libsbml gave, or None
# for a control libsbml always kept). At these values each pair differs, which
# the test checks first, so a chain cannot pass for the wrong reason.
VALUES = {"x": 0.5, "y": 2.0, "a": 2.0, "b": 3.0, "k": 1.0, "u": 0.0, "v": 0.0, "w": 2.0}
CHAINS = {
    # BNG2.pl's own brackets
    "z_band": (
        "((0<x)<1)",
        lambda p: int(int(p["x"] > 0) < 1),
        lambda p: int(0 < p["x"] < 1),
    ),
    "z_not": (
        "((x>1)==0)",
        lambda p: int(int(p["x"] > 1) == 0),
        lambda p: int(p["x"] > 1 and 1 == 0),
    ),
    "z_pair": (
        "((u>1)==(v>1))",
        lambda p: int(int(p["u"] > 1) == int(p["v"] > 1)),
        lambda p: int(p["u"] > 1 and p["v"] == 1 and p["v"] > 1),
    ),
    # unbracketed chains, as a hand-written .net has them
    "z_lt_lt": ("1<x<2", lambda p: int(int(p["x"] > 1) < 2), lambda p: int(1 < p["x"] < 2)),
    "z_lt_eq": (
        "x<y==1",
        lambda p: int(int(p["x"] < p["y"]) == 1),
        lambda p: int(p["x"] < p["y"] and p["y"] == 1),
    ),
    "z_eq_lt": (
        "a==b<1",
        lambda p: int(int(p["a"] == p["b"]) < 1),
        lambda p: int(p["a"] == p["b"] and p["b"] < 1),
    ),
    "z_ne_gt": (
        "k!=b>1",
        lambda p: int(int(p["k"] != p["b"]) > 1),
        lambda p: int(p["k"] != p["b"] and p["b"] > 1),
    ),
    "z_ge_eq": (
        "x>=y==0",
        lambda p: int(int(p["x"] >= p["y"]) == 0),
        lambda p: int(p["x"] >= p["y"] and p["y"] == 0),
    ),
    "z_if": (
        "if(0<x<1,k,0)",
        lambda p: p["k"] if int(int(p["x"] > 0) < 1) else 0.0,
        lambda p: p["k"] if 0 < p["x"] < 1 else 0.0,
    ),
    # a logical compared: libsbml read it as one more conjunct
    "z_and": (
        "((w<1)&&(v<1))==0",
        lambda p: int(int(p["w"] < 1 and p["v"] < 1) == 0),
        lambda p: int(p["w"] < 1 and p["v"] < 1 and 1 == 0),
    ),
    # libsbml always kept these: a relational right operand, arithmetic. Their
    # value was right; the first compared a truth value.
    "z_right": ("0<(x<1)", lambda p: int(int(p["x"] < 1) > 0), None),
    "z_arith": ("(y>1)*k", lambda p: int(p["y"] > 1) * p["k"], None),
}


def chains_net():
    names = list(VALUES) + list(CHAINS)
    lines = [f"    {i + 1} {n} {VALUES[n]!r}" for i, n in enumerate(VALUES)]
    lines += [f"    {len(VALUES) + i + 1} {n} {CHAINS[n][0]}" for i, n in enumerate(CHAINS)]
    assert len(set(names)) == len(names)
    return (
        "begin parameters\n"
        + "\n".join(lines)
        + "\nend parameters\nbegin species\n    1 A() 1\nend species\n"
        + "begin reactions\n    1 1 0 k\nend reactions\n"
    )


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    path = tmp_path_factory.mktemp("chains") / "c.net"
    path.write_text(chains_net())
    model = bngsim.Model.from_net(str(path))
    return model, write_sbml(model)


@pytest.mark.parametrize("name", CHAINS)
def test_a_chain_reads_as_exprtk_groups_it(chains, name):
    """The .net's own value and the written SBML's both equal the ExprTk
    reading; at these values the flat reading would differ."""
    model, text = chains
    _, grouped, flat = CHAINS[name]
    want = grouped(VALUES)
    if flat is not None:
        assert flat(VALUES) != want
    assert model.get_param(name) == want
    assert from_sbml(text).get_param(name) == want


def _maths(model):
    """Every formula a written model holds."""
    for rule in model.getListOfRules():
        yield rule.getVariable(), rule.getMath()
    for ia in model.getListOfInitialAssignments():
        yield ia.getSymbol(), ia.getMath()
    for rxn in model.getListOfReactions():
        yield rxn.getId(), rxn.getKineticLaw().getMath()


def _relational_operands_that_are_truth_values(node, out):
    if node is None:
        return out
    for i in range(node.getNumChildren()):
        child = node.getChild(i)
        if node.isRelational() and (child.isRelational() or child.isLogical()):
            out.append(libsbml.formulaToL3String(node))
        _relational_operands_that_are_truth_values(child, out)
    return out


def test_every_relational_compares_numbers(chains):
    """The written document is well typed: no relational in it has a relational
    or a logical operand, each such truth value being written as
    ``piecewise(1, cond, 0)``, and libsbml's consistency check reports no error.
    On main ``0<(x<1)`` came out as ``lt(0, lt(x, 1))``."""
    _, text = chains
    doc = libsbml.readSBMLFromString(text)
    found = []
    for symbol, formula in _maths(doc.getModel()):
        found += [
            f"{symbol}: {f}" for f in _relational_operands_that_are_truth_values(formula, [])
        ]
    assert found == []
    doc.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    doc.checkConsistency()
    errors = [
        doc.getError(i).getMessage()
        for i in range(doc.getNumErrors())
        if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
    ]
    assert errors == []


def test_the_derived_parameters_initial_assignment_keeps_the_grouping(chains):
    """``z if(0<x<1,k,0)`` is 0 in bngsim; the written initialAssignment for z,
    which overrides its value, says the same."""
    _, text = chains
    doc = libsbml.readSBMLFromString(text)
    ia = {i.getSymbol(): i.getMath() for i in doc.getModel().getListOfInitialAssignments()}
    assert libsbml.formulaToL3String(ia["z_if"]) == "piecewise(k, piecewise(1, 0 < x, 0) < 1, 0)"


def test_a_numbered_truth_value_is_not_numbered_again():
    """The spelling the writer produces reads back unchanged."""
    ast = _exprtk_to_ast("piecewise(1, 0 < x, 0) < 1", where="t")
    assert libsbml.formulaToL3String(ast) == "piecewise(1, 0 < x, 0) < 1"


# ── A function's condition, at every corner of a grid ────────────────────────

GRID_NET = """begin parameters
    1 x {x}
    2 y {y}
    3 k 1
end parameters
begin functions
    1 f() if({cond},k,0)
end functions
begin species
    1 A() 10
end species
begin reactions
    1 1 0 f
end reactions
begin groups
    1 Atot 1
end groups
"""

# A function's condition becomes an assignment rule (the chains above are
# initialAssignments). On main the written rate differed from ExprTk's at some
# corner of x, y in {0.5, 1.5} for every condition here but seven controls
# whose value libsbml kept: `x<>y>0` and the `or` form of the compared logical
# (equal at these corners), the relational right operand `0<(x<1)`, and the
# last four (a lone relational, logical operands, an arithmetic operand).
GRID = [
    "(0<x)<1",
    "((0<x)<1)",
    "0<x<1",
    "(x>1)==0",
    "(x>1) = 0",
    "(x<y)==1",
    "(x>1)==(y>1)",
    "x<y==1",
    "x==y<1",
    "x!=y>1",
    "x>=y==0",
    "x<>y>0",
    "0<(x<1)",
    "((0<x)<1)<1",
    "1<x<2<3",
    "((x<1) && (y<1)) == 0",
    "((x<1) and (y<1))==0",
    "((x<1) or (y>1))==0",
    "(x<1)==(y<1)",
    "x<1",
    "(x<1) and (y<1)",
    "(x<1) or (y>1)",
    "((x<1)+(y<1))>1",
]


@pytest.mark.parametrize("cond", GRID)
def test_a_functions_condition_reads_as_exprtk_groups_it(tmp_path, cond):
    """The written model's rate equals the .net model's, which ExprTk computes,
    at every corner: k = 1 and A = 10, so each rate is exactly 0 or 10."""
    for x, y in itertools.product((0.5, 1.5), repeat=2):
        _, model = load_net(tmp_path, GRID_NET.format(x=x, y=y, cond=cond))
        back = from_sbml(write_sbml(model))
        # Each rate is exactly 0 or -10; abs 1e-12 is the representation's slack.
        assert float(back.rhs([10.0])[0]) == pytest.approx(
            float(model.rhs([10.0])[0]), abs=1e-12
        ), (x, y)


# ── What the pass leaves alone ───────────────────────────────────────────────


def test_a_ternary_operand_is_a_number_not_a_truth_value():
    """``(x<1 ? 2 : 3) > 1`` compares the ternary's number; it is not written as
    ``piecewise(1, ..., 0)`` (parseL3Formula then refuses ``?:`` as before)."""
    assert "piecewise(1," not in _normalize_exprtk("(x<1 ? 2 : 3) > 1", frozenset())


@pytest.mark.parametrize(
    ("expr", "says"),
    [
        ("(x<1", r"function 'f': could not parse expression"),
        ("tfun('a(b.dat', x) < 1", r"function 'f': expression uses function call\(s\) .*'tfun'"),
    ],
    ids=["unbalanced", "a-bracket-in-a-tfun-file-name"],
)
def test_an_expression_the_pass_cannot_read_fails_as_it_did(expr, says):
    """Unbalanced text, or a bracket inside a quoted file name, is left to the
    errors that name where it is: the parse error, and the tfun refusal."""
    with pytest.raises(bngsim.ConversionError, match=says):
        _exprtk_to_ast(expr, where="function 'f'")


# ── The SED-ML writer goes through the same translator ───────────────────────


def test_a_sedml_value_expression_keeps_the_grouping(tmp_path):
    """write_sedml_protocol writes a setParameter's expression with the same
    translator: ``if(0<x<1,2,3)`` is ExprTk's ``if((0<x)<1, 2, 3)``, which its
    math must say. On main it said ``0 < x < 1``. The math is looked for
    anywhere in the document, so the test holds wherever the writer puts it."""
    import xml.etree.ElementTree as ET

    from bngsim.convert import ProtocolSpec, write_sedml_protocol
    from bngsim.convert._protocol import Experiment, StateChange

    net = GRID_NET.format(x=0.5, y=0.5, cond="x<1")
    _, model = load_net(tmp_path, net)
    spec = ProtocolSpec(
        steps=(
            StateChange(kind="set_parameter", target="k", value_expr="if(0<x<1,2,3)"),
            Experiment(kind="simulate", method="ode", t_span=(0.0, 1.0), n_points=2),
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sedml = write_sedml_protocol(spec, sbml=write_sbml(model))
    formulas = [
        libsbml.formulaToL3String(
            libsbml.readMathMLFromString(ET.tostring(el, encoding="unicode"))
        )
        for el in ET.fromstring(sedml).iter()
        if el.tag.endswith("}math")
    ]
    # A math libsbml cannot read gives None, kept out so a failure stays readable.
    conditions = [f for f in formulas if f and f.startswith("piecewise(2,")]
    assert len(conditions) == 1, formulas
    # The SED-ML names x by a variable of its own, so the name is free.
    assert re.fullmatch(r"piecewise\(2, piecewise\(1, 0 < \w+, 0\) < 1, 3\)", conditions[0])
