"""GH #108 — which of the two parameter-name alias maps a site is allowed to use.

The package carries two aliasing conventions for a BNGL parameter whose name
would not survive a sympy round trip:

* ``_sympy_symbol_alias_map`` — Python keywords **and** C reserved words.
* an open-coded ``_alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n``
  — Python keywords only, at six sites.

That looks like a hazard, because ``sp.ccode`` really does rename a symbol whose
name is a C reserved word (``Symbol("const")`` prints as ``const_``), and the
name-keyed rewrite back to ``p[idx]`` then misses it — *use of undeclared
identifier* ``const_``, which is what PR #69 fixed for
``ode/pulses_demo_fixed.bngl``.

It is not a hazard, and the rule is: ``sp.ccode`` is reached from exactly one
place, and that place uses the wide map. Everything else prints through
``_jacobian.sympy_to_c``, whose ``resolve`` callback maps each symbol to a C
reference itself and never lets sympy print a name at all. So the narrow sites
are correct rather than lucky — but nothing asserted it, so nothing failed if
someone widened a narrow site, narrowed a wide one, or (the one that would
actually bite) added a *new* emission path that reached for ``sp.ccode`` and
copied the keyword-only ternary sitting next to it.

Kept in its own ``test_codegen*`` file rather than beside the round-trip
regressions in ``test_derived_param_jacobian.py`` for a mundane reason: that file
is in no CI job's list, and both the MIR-JIT matrix and the Windows tail — the
two environments where a leaked reserved word actually fails to compile — run
files by name.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

pytest.importorskip("sympy")


class _CallSiteVisitor(ast.NodeVisitor):
    """Record the dotted enclosing function of every call to a named callee."""

    def __init__(self, module: str, callee_names: set[str], out: set[tuple[str, str]]):
        self.module = module
        self.callee_names = callee_names
        self.out = out
        self.stack: list[str] = []

    def visit_FunctionDef(self, node):  # noqa: N802
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):  # noqa: N802
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name in self.callee_names:
            self.out.add((self.module, ".".join(self.stack) or "<module>"))
        self.generic_visit(node)


def _call_sites(callee_names: set[str]) -> set[tuple[str, str]]:
    """``(module stem, dotted enclosing function)`` for every call to any name in
    ``callee_names``, across the whole package.

    Reads the source with ``ast`` rather than importing and introspecting: the
    question is which *call sites exist*, and a site on a branch this test does
    not execute still has to obey the rule.
    """
    import bngsim

    out: set[tuple[str, str]] = set()
    for path in sorted(pathlib.Path(bngsim.__file__).parent.glob("*.py")):
        _CallSiteVisitor(path.stem, callee_names, out).visit(ast.parse(path.read_text()))
    return out


_ALIAS_TWIN_NET = """\
# Created by BioNetGen 2.9.3
begin parameters
    1 kf     2.0  # Constant
    2 {name}     3.0  # Constant
    3 Km     5.0  # Constant
    4 kd     {name}*{name}  # ConstantExpression
end parameters
begin functions
    1 rateLawF() (({name}*{name})*Aobs)/(Km+Aobs)
    2 report() {name}*Bobs
end functions
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 kd #_R1
    2 2 1 rateLawF #_R2
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
end groups
"""

# Deliberately not a name any emitter could produce on its own, so the rename
# below cannot collide with C the generator wrote for another reason — and in
# particular not with the real ``const`` of ``const double* p``.
_ORDINARY_TWIN_NAME = "kzz"


def _emit_every_path(tmp_path, name: str) -> dict[str, str | None]:
    """Every C-emitting entry point in ``_codegen``, for a model whose one
    parameter of interest is named ``name``.

    Covers both aliasing conventions at once: ``kd = <name>*<name>`` is a derived
    rate constant, so its chain rule goes through the *wide* map and ``sp.ccode``,
    while ``rateLawF``/``report`` are differentiated through ``sympy_to_c`` and
    the *narrow* one.
    """
    import bngsim
    from bngsim import _codegen as cg

    path = tmp_path / f"alias_{name}.net"
    path.write_text(_ALIAS_TWIN_NET.format(name=name))
    model = bngsim.Model.from_net(str(path))
    return {
        "rhs (from net)": cg.generate_rhs_c(str(path)),
        "rhs (from model)": cg.generate_rhs_from_model(model),
        "jacobian": cg.generate_jacobian_from_model(model),
        "sens rhs (from net)": cg.generate_sens_rhs_c(str(path)),
        "sens rhs (elementary)": cg.generate_sens_from_model(model, functional=False),
        "sens rhs (functional)": cg.generate_sens_from_model(model, functional=True),
        "outputs": cg.generate_outputs_from_model(model),
        "output sens": cg.generate_output_sens_from_model(model),
        "combined": cg.generate_combined_from_model(model, emit_output_sens=True)[0],
    }


def _differing_lines(ordinary: str | None, twin: str | None, name: str):
    """Line pairs where the reserved-name model's C differs from its ordinary
    twin's, after renaming the ordinary one."""
    assert (ordinary is None) == (twin is None), (
        f"one arm emitted and the other declined: ordinary={ordinary is None}, "
        f"{name}={twin is None}"
    )
    if ordinary is None:
        return []
    renamed = re.sub(rf"\b{_ORDINARY_TWIN_NAME}\b", name, ordinary)
    a, b = renamed.splitlines(), twin.splitlines()
    assert len(a) == len(b), f"line count differs for {name!r}: {len(a)} vs {len(b)}"
    return [(x, y) for x, y in zip(a, b, strict=False) if x != y]


def _eval_c_arithmetic(expr: str, p: list[float], obs: list[float]) -> float:
    """Evaluate a generated C scalar expression as Python arithmetic.

    Only ``p[i]``/``obs[i]`` reads and ``+ - * /`` are admitted — anything else
    fails the guard rather than being silently ``eval``-ed.
    """
    e = re.sub(r"\bp\[(\d+)\]", lambda m: repr(p[int(m.group(1))]), expr)
    e = re.sub(r"\bobs\[(\d+)\]", lambda m: repr(obs[int(m.group(1))]), e)
    assert re.fullmatch(r"[-+*/(). 0-9eE]+", e), f"not plain arithmetic: {e!r}"
    return eval(e)  # noqa: S307 - guarded to arithmetic on literals above


class TestIssue108TheAliasingRule:
    """The rule, and what it buys — see the module docstring for why it holds."""

    def test_every_ccode_call_takes_its_symbols_from_the_wide_alias_map(self):
        """The rule, as a relationship rather than a list of names: a function
        that calls ``sp.ccode`` must also be the one that prepared the
        expression, and preparation is where the wide map lives."""
        ccode_sites = _call_sites({"ccode"})
        prepared_sites = _call_sites({"_prepare_derived_expr"})
        assert ccode_sites, "no sp.ccode call site found at all — the scan broke, not the rule"
        assert ccode_sites <= prepared_sites, (
            f"{sorted(ccode_sites - prepared_sites)} print through sp.ccode without going "
            "through _prepare_derived_expr, so a parameter named with a C reserved word "
            "will be printed as `<name>_` and no rewrite will map it back to p[idx]. "
            "Either prepare the expression there too, or print through "
            "_jacobian.sympy_to_c, which never prints a symbol name."
        )

    def test_the_wide_alias_map_has_a_single_entry_point(self):
        """#108's consolidation, asserted: the three derived-parameter sites run
        one preparation sequence, so a fix to it cannot land on one twin and miss
        the others (that is #53/#56, #69, PR #71 and #105, four times over)."""
        wide_sites = _call_sites({"_sympy_symbol_alias_map"})
        assert wide_sites == {("_codegen", "_prepare_derived_expr")}, (
            "the keyword+C-reserved alias map is reached from more than one place "
            f"({sorted(wide_sites)}); route the new site through _prepare_derived_expr instead"
        )

    def test_the_narrow_sites_never_print_through_ccode(self):
        """The other half of the rule: keyword-only aliasing is safe exactly
        where sympy is never allowed to print a symbol name."""
        ccode_sites = _call_sites({"ccode"})
        narrow = _call_sites({"_alias_keyword_param"}) - {("_codegen", "_sympy_symbol_alias_map")}
        assert narrow, "no narrow alias site found at all — the scan broke, not the rule"
        assert not (narrow & ccode_sites), (
            f"{sorted(narrow & ccode_sites)} alias Python keywords only but print "
            "through sp.ccode, which also renames C reserved words"
        )

    @pytest.mark.parametrize("name", ["const", "restrict", "int", "double"])
    def test_a_c_reserved_name_emits_identical_c_on_every_path(self, tmp_path, name):
        """What the rule buys, measured rather than argued: rename one parameter
        to a C reserved word and every emitter produces byte-identical C."""
        ordinary = _emit_every_path(tmp_path, _ORDINARY_TWIN_NAME)
        twin = _emit_every_path(tmp_path, name)
        for path_name in ordinary:
            diff = _differing_lines(ordinary[path_name], twin[path_name], name)
            assert not diff, f"{path_name} differs for a parameter named {name!r}: {diff[:2]}"

    @pytest.mark.parametrize("name", ["lambda", "del", "class"])
    def test_a_python_keyword_name_differs_only_where_a_differentiator_defers(
        self, tmp_path, name
    ):
        """A Python keyword is the one name that can change the emitted C, and
        for a reason that has nothing to do with aliasing:
        ``_saturable_jacobian.differentiate_rate_law_native`` declines a rate law
        carrying one on purpose (only the sympy parser aliases them), so that
        rate law's Jacobian is spelled by sympy instead of by the closed-form
        differentiator. Both are correct — this pins that the difference stays
        confined to that spelling and never reaches the numbers.
        """
        ordinary = _emit_every_path(tmp_path, _ORDINARY_TWIN_NAME)
        twin = _emit_every_path(tmp_path, name)
        p = [2.0, 3.0, 5.0, 9.0, 0.0, 0.0]
        obs = [10.0, 4.0]
        for path_name in ordinary:
            for line_a, line_b in _differing_lines(ordinary[path_name], twin[path_name], name):
                m_a = re.fullmatch(r"\s*double d\d+ = (.+);", line_a)
                m_b = re.fullmatch(r"\s*double d\d+ = (.+);", line_b)
                assert m_a and m_b, (
                    f"{path_name} differs for {name!r} outside the closed-form "
                    f"differentiator's own line:\n  ordinary: {line_a}\n  {name}: {line_b}"
                )
                assert _eval_c_arithmetic(m_a.group(1), p, obs) == pytest.approx(
                    _eval_c_arithmetic(m_b.group(1), p, obs)
                ), f"{path_name}: the two spellings of {line_a.strip()} disagree numerically"

    @pytest.mark.parametrize("name", ["const", "lambda"])
    def test_the_reserved_name_model_still_compiles(self, tmp_path, name):
        """Identity to a twin only means anything if the twin's C is real C.
        Constructing the Simulator builds *and compiles* the sensitivity RHS, so
        a reserved word that leaked into an identifier fails here — which is why
        this file is on the MIR-JIT and MSVC legs rather than only the pre-push
        hook's clang run."""
        import bngsim

        path = tmp_path / f"compile_{name}.net"
        path.write_text(_ALIAS_TWIN_NET.format(name=name))
        model = bngsim.Model.from_net(str(path))
        bngsim.Simulator(model, method="ode", sensitivity_params=[name, "kf", "Km"])
