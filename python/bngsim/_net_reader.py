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

>>> from bngsim import Model
>>> model = Model.from_net_via_builder("model.net")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bngsim._codegen import _strip_fixed_marker


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
    species = _parse_species(text, parameters)
    observables = _parse_observables(text)
    functions = _parse_functions(text)
    reactions = _parse_reactions(text, functions)

    return {
        "parameters": parameters,
        "species": species,
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
    params = []
    # Two-pass: first collect all, then evaluate expressions
    raw_params = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: index name value_or_expr  # comment
        parts = line.split("#")[0].strip().split()
        if len(parts) < 3:
            continue
        _idx_str, name = parts[0], parts[1]
        expr = " ".join(parts[2:])
        raw_params.append((name, expr))

    # Evaluate parameters in order (later ones can reference earlier ones)
    ns: dict[str, Any] = {"__builtins__": {}}
    import math

    ns.update(
        {
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
    )

    for name, expr in raw_params:
        try:
            value = float(eval(expr, ns))
        except Exception:
            value = 0.0
        ns[name] = value
        # Check if it's a pure number or an expression
        try:
            float(expr)
            is_expr = False
        except ValueError:
            is_expr = True
        params.append((name, value, expr, is_expr))

    return params


def _parse_species(
    text: str,
    parameters: list[tuple[str, float, str, bool]],
) -> list[tuple[str, float, bool]]:
    """Parse species block.

    Returns list of (name, init_conc, is_fixed).
    """
    block = _extract_block(text, "species")
    species = []
    param_map = {name: val for name, val, _, _ in parameters}

    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("#")[0].strip().split()
        if len(parts) < 3:
            continue
        parts[0]
        # `$` clamp marker may sit at index 0 or after a `@<compartment>::`
        # prefix (BNG2.pl emits the latter for cBNGL models).
        name, is_fixed = _strip_fixed_marker(parts[1])
        ic_str = parts[2]
        try:
            init_conc = float(ic_str)
        except ValueError:
            # May be a parameter name
            init_conc = param_map.get(ic_str, 0.0)
        species.append((name, init_conc, is_fixed))

    return species


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
        if len(parts) < 2:
            continue
        parts[0]
        name = parts[1]
        entries = []
        for token in parts[2:]:
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
        if len(parts) < 4:
            continue

        parts[0]
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
