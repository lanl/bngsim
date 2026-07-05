"""GH #111 — ``Model.from_sbml`` must load large networks at Python's default
recursion limit.

BNG-roundtripped SBML encodes each observable as an assignment rule whose math
is a sum over hundreds–thousands of species. libsbml's MathML reader binarizes
an n-ary ``<plus/>`` into a deep *left-leaning* chain ``(((s0+s1)+s2)+…)``, so a
recursive AST walk costs one Python frame per operand and overflows the default
1000-frame limit — surfacing as ``ModelError: … maximum recursion depth
exceeded`` on ``multisite_phos`` (~1026 species) and ``fceri_gamma`` (3744).

The fix walks every full-subtree scanner iteratively (``_iter_ast_subtree`` /
``_flatten_assoc_chain``) instead of bumping ``sys.setrecursionlimit`` — the same
approach already used on the numeric-eval and ExprTk-translation paths. These
tests pin that contract: the converted scanners, and an end-to-end load, must
succeed on an AST deeper than the recursion limit **without touching the limit**.
"""

from __future__ import annotations

import sys

import bngsim
import libsbml
from bngsim._sbml_loader import (
    _ast_has_rateof,
    _ast_references_time,
    _ast_to_exprtk_with_funcdefs,
    _classify_linear_species_rule,
    _collect_state_discontinuity_conditions,
    _collect_time_discontinuity_conditions,
    _iter_ast_subtree,
    _walk_name_refs,
    _walk_species_refs,
)

# Deeper than the default 1000-frame limit so a per-operand recursive walk would
# overflow — the whole point of the regression.
_N_TERMS = 1500


def _deep_plus_ast(n: int = _N_TERMS) -> libsbml.ASTNode:
    """An ``s0 + s1 + … + s{n-1}`` AST as libsbml's MathML reader builds it:
    a left-leaning binary ``<plus/>`` chain of depth ~``n``."""
    terms = "".join(f"<ci>s{i}</ci>" for i in range(n))
    mml = f'<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/>{terms}</apply></math>'
    ast = libsbml.readMathMLFromString(mml)
    assert ast is not None
    return ast


def _left_spine_depth(node: libsbml.ASTNode) -> int:
    depth = 0
    n = node
    while n is not None and n.getNumChildren() > 0:
        depth += 1
        n = n.getChild(0)
    return depth


def test_deep_chain_is_actually_deeper_than_the_recursion_limit():
    """Guard the guard: if libsbml ever stops binarizing n-ary plus, this AST
    would shrink and the recursion tests below would silently stop testing
    anything. Assert the shape we depend on."""
    ast = _deep_plus_ast()
    assert _left_spine_depth(ast) > sys.getrecursionlimit()


def test_iter_ast_subtree_visits_every_node_iteratively():
    ast = _deep_plus_ast()
    names = {n.getName() for n in _iter_ast_subtree(ast) if n.getType() == libsbml.AST_NAME}
    assert names == {f"s{i}" for i in range(_N_TERMS)}


def test_iter_ast_subtree_handles_none():
    assert list(_iter_ast_subtree(None)) == []


def test_ast_has_rateof_no_overflow_on_deep_chain():
    # No rateOf anywhere in a pure species sum.
    assert _ast_has_rateof(_deep_plus_ast(), set()) is False


def test_ast_references_time_no_overflow_on_deep_chain():
    assert _ast_references_time(_deep_plus_ast()) is False


def test_walk_name_refs_finds_targets_in_deep_chain():
    out: set[str] = set()
    targets = {"s0", "s7", f"s{_N_TERMS - 1}"}
    _walk_name_refs(_deep_plus_ast(), targets, out)
    assert out == targets


def test_walk_species_refs_finds_targets_in_deep_chain():
    out: set[str] = set()
    _walk_species_refs(_deep_plus_ast(), {"s42"}, out)
    assert out == {"s42"}


def test_collect_time_discontinuity_conditions_no_overflow_on_deep_chain():
    # No inequalities ⇒ no conditions, but the walk must still reach the bottom.
    conditions: list[str] = []
    _collect_time_discontinuity_conditions(_deep_plus_ast(), {}, conditions)
    assert conditions == []


def test_collect_state_discontinuity_conditions_no_overflow_on_deep_chain():
    # GH #244's floor/ceiling scanner (added later) must walk iteratively too:
    # a deep species sum has no floor/ceiling, but the walk must still reach the
    # bottom without overflowing the recursion limit.
    conditions: list[str] = []
    _collect_state_discontinuity_conditions(_deep_plus_ast(), {}, {}, set(), conditions)
    assert conditions == []


def test_classify_linear_species_rule_flattens_deep_chain():
    species_idx = {f"s{i}": i for i in range(_N_TERMS)}
    entries = _classify_linear_species_rule(_deep_plus_ast(), species_idx)
    assert entries is not None
    # Every summand is a unit-weight species term.
    assert sorted(entries) == [(i, 1.0) for i in range(_N_TERMS)]


def test_ast_to_exprtk_translates_deep_chain():
    ex = _ast_to_exprtk_with_funcdefs(_deep_plus_ast(), {})
    assert ex == "(" + "+".join(f"s{i}" for i in range(_N_TERMS)) + ")"


def test_recursion_limit_is_left_untouched_by_the_scanners():
    """The fix is iterative, not a ``setrecursionlimit`` bump — assert the
    process-global limit is the same before and after a full scan sweep."""
    before = sys.getrecursionlimit()
    ast = _deep_plus_ast()
    _ast_has_rateof(ast, set())
    _ast_references_time(ast)
    _collect_time_discontinuity_conditions(ast, {}, [])
    _classify_linear_species_rule(ast, {f"s{i}": i for i in range(_N_TERMS)})
    _ast_to_exprtk_with_funcdefs(ast, {})
    assert sys.getrecursionlimit() == before


def test_from_sbml_string_loads_deep_assignment_rule_at_default_limit():
    """End-to-end guard for the exact path that overflowed in #30's screen:
    ``_build_model_from_sbml_doc`` pre-scans every rule's math (rateOf detection,
    time-discontinuity collection) before building. A deep observable-style sum
    must load at the default limit. Summing parameters (not species) keeps the
    network trivial so the test stays fast while still walking a >1000-deep AST."""
    before = sys.getrecursionlimit()
    n = _N_TERMS
    params = "".join(f'<parameter id="p{i}" value="1" constant="true"/>' for i in range(n))
    terms = "".join(f"<ci>p{i}</ci>" for i in range(n))
    rule_math = (
        f'<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/>{terms}</apply></math>'
    )
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
        'level="3" version="2"><model>'
        '<listOfCompartments><compartment id="c" size="1" constant="true"/>'
        "</listOfCompartments>"
        '<listOfSpecies><species id="X" compartment="c" initialConcentration="0" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        "</listOfSpecies>"
        f'<listOfParameters>{params}<parameter id="obs" constant="false"/>'
        "</listOfParameters>"
        f'<listOfRules><assignmentRule variable="obs">{rule_math}</assignmentRule>'
        "</listOfRules></model></sbml>"
    )
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.n_species >= 1
    assert sys.getrecursionlimit() == before
