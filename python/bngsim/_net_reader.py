"""bngsim._net_reader — Pure-Python .net file parser for ModelBuilder.

Parses a BNG .net file into a dictionary of model components that can
be fed into ``ModelBuilder`` for programmatic model construction. This
enables users to load .net files into ModelBuilder, inspect/modify the
structure, and build models — all without requiring the C++ net file
loader.

This is the **recommended** pattern for users who want to:
- Load a .net file and modify it before building
- Extract model structure for analysis
- Use .net models as templates for programmatic construction

Example
-------
>>> from bngsim._net_reader import parse_net_file
>>> from bngsim._bngsim_core import ModelBuilder
>>> parsed = parse_net_file("model.net")
>>> builder = ModelBuilder()
>>> for name, value in parsed["parameters"]:
...     builder.add_parameter(name, value)
>>> # ... add species, reactions, etc.
>>> model = builder.build()

Or use the convenience function:

>>> from bngsim import build_model_from_parsed, parse_net_file
>>> model = build_model_from_parsed(parse_net_file("model.net"))
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from bngsim._codegen import _strip_fixed_marker

#: A name and nothing else. Tells a misspelled parameter reference in a species
#: IC column apart from an arithmetic one: `B0_typo` should have resolved,
#: `2*A0` is an expression to evaluate (issue #600).
_BARE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _check_synthetic_rate_expr(expr: str) -> None:
    t = expr.strip()
    if re.search(r"[\+\-\*/\^]\s*$", t):
        raise ValueError(f"invalid rate expression {expr!r}: ends with an operator")
    if re.match(r"^\s*[\*/\^]", t):
        raise ValueError(f"invalid rate expression {expr!r}: starts with invalid operator")
    depth = 0
    for c in t:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"invalid rate expression {expr!r}: unmatched ')'")
    if depth != 0:
        raise ValueError(f"invalid rate expression {expr!r}: unmatched '('")


def parse_net_file(path: str | Path) -> dict[str, Any]:
    """Parse a BNG .net file into a structured dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the .net file.

    Returns
    -------
    dict
        Structured contents of the ``.net`` file, with keys::

            parameters   : list of (name, value, expression, is_expression)
            species      : list of (name, init_conc, is_fixed)
            species_ic_params : list of (species_idx0, param_name) for each
                           species whose IC column names a parameter
            observables  : list of (name, entries), entries = [(sp_idx0, factor), ...]
            functions    : list of (name, expression)
            reactions    : list of dict with keys reactants, products (0-based
                           species indices), type ("elementary"/"functional"),
                           rate_law (parameter or function name), stat_factor
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Parse each block
    parameters = _parse_parameters(text)
    species, species_ic_params, lifted_params = _parse_species(text, parameters)
    # An expression-valued IC is carried as a synthetic `_InitialConc<N>`
    # parameter, exactly as BNG2.pl writes one, so it reaches ModelBuilder with
    # the declared parameters and the species resolves against it (issue #600).
    parameters = parameters + lifted_params
    observables = _parse_observables(text)
    functions = _parse_functions(text)
    reactions = _parse_reactions(text, functions)

    return {
        "parameters": parameters,
        "species": species,
        "species_ic_params": species_ic_params,
        "observables": observables,
        "functions": functions,
        "reactions": reactions,
    }


def build_model_from_parsed(parsed: dict[str, Any]):
    """Build a NetworkModel from parsed .net data via ModelBuilder.

    Parameters
    ----------
    parsed : dict
        Output of ``parse_net_file()``.

    Returns
    -------
    bngsim.Model
        The constructed model.
    """
    from bngsim._bngsim_core import ModelBuilder
    from bngsim._model import Model

    builder = ModelBuilder()

    # Parameters
    param_map = {}  # name -> value (for resolving species ICs)
    for name, value, expr, is_expr in parsed["parameters"]:
        builder.add_parameter(name, value, expr, is_expr)
        param_map[name] = value

    # Species
    for name, init_conc, is_fixed in parsed["species"]:
        builder.add_species(name, init_conc, is_fixed)

    # A species IC written as a parameter name is handed to the builder as a
    # reference rather than as the number resolved above, so build() re-resolves
    # it from the compiled parameter — the step that makes this model's initial
    # state the one Model.from_net produces (issue #554) — and records the
    # (species, parameter) pair the forward-sensitivity seeding reads.
    for species_idx0, param_name in parsed.get("species_ic_params", ()):
        builder.add_species_param_ref(species_idx0, param_name)

    # Observables
    for name, entries in parsed["observables"]:
        builder.add_observable(name, entries)

    # Functions (track names for reaction rate resolution)
    func_names: set[str] = set()
    for name, expression in parsed["functions"]:
        builder.add_function(name, expression)
        func_names.add(name)

    # Reactions
    for i, rxn in enumerate(parsed["reactions"]):
        rtype = rxn["type"]
        rate_law = rxn["rate_law"]
        if rtype == "elementary" and rate_law not in param_map:
            if not rate_law.strip():
                raise ValueError(
                    "elementary reaction has empty or whitespace-only rate_law "
                    "(not a parameter name)"
                )
            if rate_law in func_names:
                rtype = "functional"
            else:
                _check_synthetic_rate_expr(rate_law)
                collision_idx = i
                auto_func = f"__net_reader_func_{collision_idx}"
                while auto_func in func_names:
                    collision_idx += 1
                    auto_func = f"__net_reader_func_{collision_idx}"
                builder.add_function(auto_func, rate_law)
                func_names.add(auto_func)
                rate_law = auto_func
                rtype = "functional"
        builder.add_reaction(
            rxn["reactants"],
            rxn["products"],
            rtype,
            rate_law,
            rxn["stat_factor"],
        )

    core = builder.build()
    return Model(_core=core)


# ─── Internal parsers ─────────────────────────────────────────────────


def _extract_block(text: str, block_name: str) -> str:
    """Extract content between 'begin <block>' and 'end <block>'."""
    pattern = rf"begin\s+{block_name}\s*\n(.*?)end\s+{block_name}"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1) if m else ""


def _parse_parameters(text: str) -> list[tuple[str, float, str, bool]]:
    """Parse parameters block.

    Returns list of (name, value, expression, is_expression).
    """
    block = _extract_block(text, "parameters")
    # Two-pass: first collect all, then evaluate expressions
    raw_params = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: [index] name value_or_expr  # comment. The leading index is
        # optional, as in the species block: `generate_network` writes
        # `1 A0 100`, `writeFile({format=>"net"})` writes `A0 100`. Requiring it
        # discarded an unindexed parameters block silently (issue #600).
        parts = line.split("#")[0].strip().split()
        if not parts:
            continue
        field = 1 if (parts[0].isascii() and parts[0].isdigit()) else 0
        if field + 1 >= len(parts):
            raise ValueError(f"parameter line {line!r} has no name and value to read")
        name = parts[field]
        expr = " ".join(parts[field + 1 :])
        raw_params.append((name, expr))

    # Split literals from expressions the same way net_file_loader.cpp does:
    # a value the numeric parse consumes whole is a constant, anything else is
    # an expression. `1/7` and `10^2` land on the expression side in both.
    decls: list[tuple[str, str, float, bool]] = []
    for name, expr in raw_params:
        try:
            decls.append((name, expr, float(expr), False))
        except ValueError:
            decls.append((name, expr, 0.0, True))

    values = _evaluate_parameter_exprs(decls)
    return [
        (name, values[i], expr, is_expr) for i, (name, expr, _literal, is_expr) in enumerate(decls)
    ]


def _evaluate_parameter_exprs(decls: list[tuple[str, str, float, bool]]) -> list[float]:
    """Evaluate declared .net parameters with the engine's own expression evaluator.

    ``decls`` is ``(name, expression, literal_value, is_expression)`` in
    declaration order; the return is the value of each, positionally.

    The evaluation runs through a parameters-only ``ModelBuilder``, whose
    ``build()`` compiles and evaluates every expression exactly as the C++ .net
    loader's does — so ``parse_net_file`` reports the number
    ``Model.from_net`` would put in the same slot. Evaluating these in Python
    instead cannot be made to agree: BNGL spells exponentiation ``^``, which
    Python reads as bitwise XOR (``10^2`` → 8, ``1e-4^3`` → ``TypeError``), and
    BNGL's ``if(c,t,f)``, ``&&``/``||`` and names like ``lambda`` are not Python
    at all. Issue #554 — every one of those used to be swallowed into a silent
    0.0 that then seeded species initial conditions.

    An install with no compiled extension falls through to
    ``_evaluate_parameter_exprs_without_engine``, which keeps this function's
    contract as far as arithmetic goes and refuses the rest.
    """
    if not decls:
        return []

    try:
        from bngsim._bngsim_core import ModelBuilder
    except ImportError:
        return _evaluate_parameter_exprs_without_engine(decls)

    builder = ModelBuilder()
    for name, expr, literal, is_expr in decls:
        # 0.0 is the pre-evaluation seed net_file_loader.cpp uses. Nothing is
        # left holding it any more: since #602 `build()` refuses an expression
        # it cannot compile rather than leaving the seed in place, so a value
        # that comes back is one the evaluator actually produced.
        builder.add_parameter(name, 0.0 if is_expr else literal, expr, is_expr)
    model = builder.build()
    return [model.get_param(name) for name, _, _, _ in decls]


# Namespace for the engine-free fallback below — the one the reader has always
# had, kept as it was so nothing that evaluated before stops evaluating.
_FALLBACK_NS: dict[str, Any] = {
    "__builtins__": {},
    "pi": math.pi,
    "e": math.e,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pow": pow,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
}


def _evaluate_parameter_exprs_without_engine(
    decls: list[tuple[str, str, float, bool]],
) -> list[float]:
    """Evaluate parameter expressions with no compiled extension available.

    ``parse_net_file`` is documented as working without one — its dict is the
    interchange format for handing a ``.net`` model to scipy, gillespy2 or a
    hand-written RHS — so this path keeps that promise. What it cannot keep is
    BNGL: it evaluates ordinary arithmetic, reading ``^`` as exponentiation
    (the one operator Python spells the same and means differently), and
    *refuses* anything beyond that instead of substituting a number.
    ``if(c,t,f)``, ``&&``/``||`` and a parameter named for a Python keyword need
    the engine's evaluator. Silence is what made the old fallback dangerous
    (issue #554): 0.0 is a perfectly plausible rate constant.
    """
    ns = dict(_FALLBACK_NS)
    values = []
    for name, expr, literal, is_expr in decls:
        if not is_expr:
            value = literal
        else:
            try:
                value = float(eval(expr.replace("^", "**"), ns))  # noqa: S307
            except Exception as exc:
                raise ValueError(
                    f"cannot evaluate .net parameter {name} = {expr!r}: {exc}. "
                    f"bngsim._bngsim_core is unavailable, so parse_net_file is "
                    f"evaluating plain arithmetic only; BNGL's if(), && / || and "
                    f"names Python reserves need the engine's expression "
                    f"evaluator, which a built bngsim provides."
                ) from exc
        ns[name] = value
        values.append(value)
    return values


def _parse_species(
    text: str,
    parameters: list[tuple[str, float, str, bool]],
) -> tuple[
    list[tuple[str, float, bool]],
    list[tuple[int, str]],
    list[tuple[str, float, str, bool]],
]:
    """Parse species block.

    Returns ``(species, ic_param_refs, lifted_params)``. ``species`` is a list of
    (name, init_conc, is_fixed) and ``ic_param_refs`` pairs the 0-based index of
    each species whose IC column names a parameter with that parameter's name.
    The names are what lets ``build_model_from_parsed`` hand the reference to
    ``ModelBuilder.add_species_param_ref``, which is where the .net loader gets
    both its re-resolved IC and its forward-sensitivity seed.

    ``lifted_params`` carries the synthetic ``_InitialConc<N>`` parameters minted
    for expression-valued initial concentrations, in the same shape as the
    parameters block, for the caller to append (issue #600).
    """
    block = _extract_block(text, "species")
    species: list[tuple[str, float, bool]] = []
    ic_param_refs: list[tuple[int, str]] = []
    param_map = {name: val for name, val, _, _ in parameters}
    taken = set(param_map)
    lifted: list[tuple[str, str]] = []  # (synthetic name, expression)
    lifted_slots: list[int] = []  # species rows whose IC the lift will fill

    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("#")[0].strip().split()
        if not parts:
            continue

        # A species line is `[<index>] <species> [<concentration>]`: the leading
        # index is OPTIONAL. BNG2.pl strips one only when present
        # (Perl2/SpeciesList.pm, `s/^\s*\d+\s+//`), and writes both shapes into a
        # .net — `generate_network` indexed, `writeFile({format=>"net"})` bare.
        # Requiring three fields dropped every line of an unindexed block
        # silently, leaving a model with no species at all (issue #600).
        # `isascii()` keeps this identical to the C++ `std::isdigit` check: bare
        # `isdigit()` also accepts characters like "²", which is not an index.
        field = 1 if (parts[0].isascii() and parts[0].isdigit()) else 0
        if field >= len(parts):
            raise ValueError(f"species line {line!r} has an index but no species pattern")

        # `$` clamp marker may sit at index 0 or after a `@<compartment>::`
        # prefix (BNG2.pl emits the latter for cBNGL models).
        name, is_fixed = _strip_fixed_marker(parts[field])
        # The concentration is the rest of the line, joined so a spaced
        # expression survives; omitted means zero, as in BNG2.pl's reader.
        ic_str = " ".join(parts[field + 1 :])
        if not ic_str:
            species.append((name, 0.0, is_fixed))
            continue
        try:
            init_conc = float(ic_str)
        except ValueError:
            if ic_str in param_map:
                ic_param_refs.append((len(species), ic_str))
                init_conc = param_map[ic_str]
            elif _BARE_IDENTIFIER.fullmatch(ic_str):
                # A name and nothing else, naming no declared parameter: a typo,
                # or a parameter declared after the species block. Seeding 0.0
                # made that a silently wrong trajectory (issue #571), and
                # net_file_loader.cpp refuses it with this same message (#554).
                raise ValueError(
                    f"species {name!r} initial concentration {ic_str!r} "
                    "is neither a number nor a declared parameter"
                ) from None
            else:
                # Not a number and not a name, so an expression — and
                # run_network evaluates one. Lift it into a synthetic parameter
                # the way BNG2.pl's generate_network lifts a BNGL seed-species
                # expression, and point the species at that (issue #600). A
                # symbol the model never declares still fails loudly, because
                # the parameter compile refuses it (issue #602).
                n = 1
                while f"_InitialConc{n}" in taken:
                    n += 1
                synthetic = f"_InitialConc{n}"
                taken.add(synthetic)
                lifted.append((synthetic, ic_str))
                lifted_slots.append(len(species))
                ic_param_refs.append((len(species), synthetic))
                init_conc = 0.0  # replaced below, once the lift is evaluated
        species.append((name, init_conc, is_fixed))

    if not lifted:
        return species, ic_param_refs, []

    # Evaluate the lifted expressions alongside the declared parameters, in the
    # one place that can: the engine's evaluator, through the same
    # parameters-only build the parameters block uses. Doing it here rather than
    # in `_parse_parameters` is what lets a lift reference any declared
    # parameter regardless of where the species block sits in the file.
    decls = [(name, expr, value, is_expr) for name, value, expr, is_expr in parameters]
    decls += [(name, expr, 0.0, True) for name, expr in lifted]
    values = _evaluate_parameter_exprs(decls)

    lifted_params = [
        (name, values[len(parameters) + i], expr, True) for i, (name, expr) in enumerate(lifted)
    ]
    lifted_value = {name: value for name, value, _, _ in lifted_params}
    for row in lifted_slots:
        sp_name, _seed, sp_fixed = species[row]
        ref = dict(ic_param_refs)[row]
        species[row] = (sp_name, lifted_value[ref], sp_fixed)

    return species, ic_param_refs, lifted_params


def _parse_observables(text: str) -> list[tuple[str, list[tuple[int, float]]]]:
    """Parse groups (observables) block.

    Returns list of (name, [(species_idx_0based, factor), ...]).
    """
    block = _extract_block(text, "groups")
    observables = []

    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("#")[0].strip().split()
        if not parts:
            continue
        # `[<index>] <name> [<entries>]` — the index is optional, and
        # run_network reads an unindexed groups block. Taking the name from a
        # fixed second field read the *entry* as the name (issue #600).
        field = 1 if (parts[0].isascii() and parts[0].isdigit()) else 0
        if field >= len(parts):
            raise ValueError(f"group line {line!r} has an index but no observable name")
        name = parts[field]
        # Entries are optional: an observable matching no species has none.
        entries = []
        for token in parts[field + 1 :]:
            for sub in token.split(","):
                sub = sub.strip()
                if not sub:
                    continue
                if "*" in sub:
                    factor_s, idx_s = sub.split("*", 1)
                    entries.append((int(idx_s) - 1, float(factor_s)))
                else:
                    entries.append((int(sub) - 1, 1.0))
        observables.append((name, entries))

    return observables


def _parse_functions(text: str) -> list[tuple[str, str]]:
    """Parse functions block.

    Returns list of (name, expression).
    """
    block = _extract_block(text, "functions")
    functions = []

    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: index name() expression
        # or: index name() expression  #comment
        line = line.split("#")[0].strip()
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        parts[0]
        name_with_parens = parts[1]
        expression = parts[2]
        # Strip trailing () from name
        name = name_with_parens.rstrip("()")
        functions.append((name, expression))

    return functions


def _parse_reactions(
    text: str,
    functions: list[tuple[str, str]],
) -> list[dict]:
    """Parse reactions block.

    Returns list of dicts with reactants, products, type, rate_law, stat_factor.
    """
    block = _extract_block(text, "reactions")
    func_names = {name for name, _ in functions}
    reactions = []

    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#")[0].strip()
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 4:
            # A reaction line's index is NOT optional: run_network refuses a
            # reactions block written without it. Skipping a short line built a
            # model with no reactions at all, which loaded clean and integrated
            # a flat trajectory while reporting success (issue #600).
            raise ValueError(
                f"reaction line {line!r} needs an index, reactants, products and a rate law"
            )
        # Find the rate law — it's the last token before any comment
        # Format: idx reactants products rate_law [stat_factor]
        # reactants and products are comma-separated species indices
        # We need to parse: idx r1,r2 p1,p2 rate_law

        # The reactant and product fields are 1-based species indices
        reactant_str = parts[1]
        product_str = parts[2]
        rate_law = parts[3]

        # Parse stat_factor if present (not common)
        stat_factor = 1.0

        # Parse reactants (1-based → 0-based, 0 means null/creation)
        reactants = []
        for tok in reactant_str.split(","):
            tok = tok.strip()
            if tok and tok != "0":
                reactants.append(int(tok) - 1)

        # Parse products (1-based → 0-based, 0 means null/degradation)
        products = []
        for tok in product_str.split(","):
            tok = tok.strip()
            if tok and tok != "0":
                products.append(int(tok) - 1)

        # Determine type: if rate_law is a function name → functional
        rtype = "functional" if rate_law in func_names else "elementary"

        reactions.append(
            {
                "reactants": reactants,
                "products": products,
                "type": rtype,
                "rate_law": rate_law,
                "stat_factor": stat_factor,
            }
        )

    return reactions
