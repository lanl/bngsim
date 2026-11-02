"""Checks a BNG XML must pass before a network-free backend is handed it.

Issue #862: NFsim cannot prepare a function that reads ``time()`` and fails at
setup with ``Error preparing function f in class GlobalFunction``, which names
neither the cause nor an alternative. It is refused here, before the backend is
built, naming the functions. The clock is read under either spelling,
``time()`` or a bare ``time``, and both are refused.

RuleMonkey runs such a function since 3.11.0, which integrates a rate that reads
the clock between events (richardposner/RuleMonkey#86), and it runs on absolute
model time (issue #1022). So does a table function indexed by time
(``type="TFUN"`` with a clock ``ctrName``), which keeps its clock in an
attribute while its ``<Expression>`` reads a placeholder. NFsim runs one on
elapsed time, BNG2.pl's simulate_nf semantics, and reads it again at each of
its x values whether or not a reaction fires, which is exact for a step table
(issue #892). Where a linear one ramps between them, NFsim places each firing
where the propensity integrated from the last one reaches an exponential draw
(issue #1179), so neither kind is refused.

A RuleMonkey session uses :func:`clock_readers` to refuse a segment whose labels
are not the clock the model reads.

An Arrhenius (energy-based) rate law is refused on NFsim (issue #854's review).
The vendored NFsim expands energy rules at load with three silent defects
(RuleWorld/nfsim#102): it divides every energy by an ``RT`` that BioNetGen does
not apply, 2.478 unless the model has a parameter named ``RT``; it gives a
one-way binding rule a reverse reaction the model does not have; and it drops a
one-way unbinding rule and every state-change rule. Drop this refusal when the
vendored NFsim carries the upstream fix (gated issue #1057). Energy patterns
alone change no rate, so only the rate law is refused. So is a rate-law type
NFsim does not implement: BNG2.pl writes ``Hill`` into the XML, and NFsim drops
such a rule with a message only its captured log holds, so the rule never fires.

A compartment model is refused on NFsim (issue #1067): NFsim ignores pattern-level
compartments, misses volume factors and compartment adjacency, misplaces seed
complexes and created molecules, and mishandles MoveConnected, and BNG2.pl's
writeXML drops NumberPerQuantityUnit (RuleWorld/nfsim#106, #108, #109, #110,
RuleWorld/bionetgen#348). A narrower envelope kept missing cases, one of them
invisible in the XML, so every compartment model is refused.

A function with an argument that reads, outside it, an observable that can
change, directly or through a function it calls, is refused on NFsim too
(issue #1062): NFsim re-evaluates such a function only on the complexes a
reaction produces, so that term goes stale on every other molecule
(RuleWorld/nfsim#104).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bngsim._exceptions import ModelError

# `time()` or a bare `time`: the engine binds the clock under both spellings.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?![A-Za-z0-9_])")

# The counter names RuleMonkey reads as the clock for a table function, checked
# before any parameter or observable of that name (resolve_tfun_counter in
# third_party/rulemonkey/cpp/rulemonkey/simulator.cpp).
_RULEMONKEY_CLOCK_COUNTERS = frozenset({"time", "t", "time()", "t()"})

_BACKENDS = {
    "nfsim": (
        "NFsim (method='nf' / 'nf_reject')",
        "NFsim cannot prepare a function that reads time() and fails at setup",
    ),
}


def _function_elements(root: ET.Element) -> list[ET.Element]:
    return [
        el
        for el in root.iter()
        if el.tag.endswith("Function") and not el.tag.endswith("ListOfFunctions")
    ]


def _functions(xml_path: str | Path) -> list[ET.Element] | None:
    """Every ``<Function>`` element in *xml_path*, or ``None`` when the file
    cannot be parsed: the backend then reports its own parse error, which is
    more specific than anything these checks could say."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    return _function_elements(root)


def _reads_time(func: ET.Element, clock: re.Pattern[str] = _TIME_REF) -> bool:
    return any(expr.tag.endswith("Expression") and clock.search(expr.text or "") for expr in func)


def rulemonkey_symbols(root: ET.Element) -> set[str]:
    """The names RuleMonkey binds in an expression ahead of its clock:
    parameters and functions by id, observables by name (init_eval_layout in
    third_party/rulemonkey/cpp/rulemonkey/engine.cpp)."""
    names: set[str] = set()
    for el in root.iter():
        tag = _local(el.tag)
        if tag in ("Parameter", "Function"):
            names.add(el.get("id", ""))
        elif tag == "Observable":
            names.add(el.get("name", ""))
    return names


def rulemonkey_clock_ref(symbols: set[str]) -> re.Pattern[str]:
    """The clock as RuleMonkey reads it in an expression of a model with these
    *symbols* (init_eval_layout and strip_layout_call_args in engine.cpp, issue
    #87 there): ``time()`` always; a bare or spaced ``time``, and the alias
    ``t`` or ``t()``, unless the model has a symbol of that name, which RuleMonkey
    reads instead. BNG2.pl writes an observable named ``time`` but never the
    alias, which only hand-written XML has."""
    clock = [r"time\(\)"]
    if "time" not in symbols:
        clock.append(r"time(?:\s*\(\s*\))?")
    if "t" not in symbols:
        clock.append(r"t(?:\s*\(\s*\))?")
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{'|'.join(clock)})(?![A-Za-z0-9_(])")


def _indexed_by_time(func: ET.Element) -> bool:
    return (func.get("type") or "").lower() == "tfun" and (
        (func.get("ctrName") or "").strip() in _RULEMONKEY_CLOCK_COUNTERS
    )


def time_reading_functions(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* whose expression reads the clock,
    as ``time()`` or as a bare ``time``. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _reads_time(f)]


def time_indexed_table_functions(xml_path: str | Path) -> list[str]:
    """The ids of the table functions in *xml_path* whose counter RuleMonkey
    reads as the clock. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _indexed_by_time(f)]


def clock_readers(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* that read the clock as RuleMonkey
    does (:func:`rulemonkey_clock_ref`), or through a table indexed by time.
    Empty when the file cannot be parsed."""
    return rulemonkey_clock(xml_path)[0]


def rulemonkey_clock(xml_path: str | Path) -> tuple[list[str], list[str]]:
    """:func:`clock_readers` of *xml_path*, and the ids of those that are
    tables indexed by time, from one parse."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return [], []
    clock = rulemonkey_clock_ref(rulemonkey_symbols(root))
    functions = _function_elements(root)
    return (
        [f.get("id", "?") for f in functions if _reads_time(f, clock) or _indexed_by_time(f)],
        [f.get("id", "?") for f in functions if _indexed_by_time(f)],
    )


# The rate-law types the vendored NFsim implements (NFinput.cpp,
# initReactionRulePermutation). BNG2.pl writes three more: Arrhenius, refused
# below for its own reasons; Sat, which NFsim refuses at load; and Hill, which
# NFsim announces only in a log bngsim captures and then drops the rule, so it
# never fires. An unknown type takes the same path as Hill.
_NFSIM_RATE_LAWS = frozenset({"Ele", "Function", "FunctionProduct", "MM"})


def _rate_law_functions(xml_path: str | Path) -> list[tuple[str, str, str]]:
    """``(rule name, rate-law type, rate-law name attribute)`` for every rule."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return []
    return [
        (rule.get("name") or rule.get("id") or "?", law.get("type") or "", law.get("name") or "")
        for rule in root.iter()
        if rule.tag.endswith("ReactionRule")
        for law in rule
        if law.tag.endswith("RateLaw")
    ]


def _rate_laws(xml_path: str | Path) -> list[tuple[str, str]]:
    """``(rule name, rate-law type)`` for every reaction rule in *xml_path*.
    Empty when the file cannot be parsed."""
    return [(rule, kind) for rule, kind, _ in _rate_law_functions(xml_path)]


def arrhenius_rules(xml_path: str | Path) -> list[str]:
    """The names of the reaction rules in *xml_path* whose rate law is
    ``Arrhenius``. Empty when the file cannot be parsed."""
    return [name for name, kind in _rate_laws(xml_path) if kind == "Arrhenius"]


def nfsim_unimplemented_rate_laws(xml_path: str | Path) -> list[tuple[str, str]]:
    """``(rule name, rate-law type)`` for each rule whose rate law the vendored
    NFsim does not implement, Arrhenius aside. Empty when the file cannot be parsed."""
    return [
        (name, kind)
        for name, kind in _rate_laws(xml_path)
        if kind != "Arrhenius" and kind not in _NFSIM_RATE_LAWS
    ]


def refuse_nfsim_rate_laws(xml_path: str | Path) -> None:
    """Raise :class:`ModelError` if *xml_path* has a rule NFsim would run wrong or
    not at all: an Arrhenius rate law (RuleWorld/nfsim#102), or a type NFsim does
    not implement."""
    arrhenius = arrhenius_rules(xml_path)
    other = nfsim_unimplemented_rate_laws(xml_path)
    if not arrhenius and not other:
        return
    label, _ = _BACKENDS["nfsim"]
    parts = []
    if arrhenius:
        parts.append(
            "Arrhenius (energy-based) rate laws: rules "
            + ", ".join(f"'{n}'" for n in arrhenius)
            + ". NFsim's expansion of energy rules divides every energy by an RT that "
            "BioNetGen does not apply (2.478 unless the model defines a parameter RT), "
            "gives a one-way binding rule a reverse reaction, and drops one-way unbinding "
            "and state-change rules, all without an error (RuleWorld/nfsim#102; this "
            "refusal is lifted under issue #1057)"
        )
    if other:
        parts.append(
            "rate laws NFsim does not implement: "
            + ", ".join(f"rule '{n}' ({kind or 'no type'})" for n, kind in other)
            + ". NFsim implements Ele, MM and function rate laws, and drops a rule of "
            "any other type without an error (Hill) or stops the load (Sat)"
        )
    raise ModelError(
        f"{label} cannot run a model with {'; and with '.join(parts)}. Simulate it with "
        "method='ode' or method='ssa' on the network BNG2.pl generates."
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) == name]


def compartments(xml_path: str | Path) -> list[str]:
    """The ids of the compartments *xml_path* declares. Empty when the file
    cannot be parsed."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return []
    return [
        c.get("id") or "?"
        for lst in root.iter()
        if _local(lst.tag) == "ListOfCompartments"
        for c in lst
        if _local(c.tag).lower() == "compartment"
    ]


def refuse_rulemonkey_compartments(xml_path: str | Path) -> None:
    """Raise :class:`ModelError` if *xml_path* declares compartments.

    RuleMonkey does not scale rates by compartment volume or keep molecules in
    their compartments, and means to refuse such a model at load, but its check
    looks for ``<Compartment>`` where BNG2 writes ``<compartment>``, so it ran
    one with the wrong rates and no error (issue #1022).
    """
    found = compartments(xml_path)
    if not found:
        return
    raise ModelError(
        "RuleMonkey (method='rm' / 'nf_exact') cannot run a model with compartments: "
        + ", ".join(f"'{n}'" for n in found)
        + ". It does not scale rates by compartment volume or keep molecules in their "
        "compartments, so the rates would be wrong. Simulate it with method='ode' or "
        "method='ssa' on the generated network (issue #1022)."
    )


def _molecule_names(el: ET.Element) -> set[str]:
    """Molecule-type names of every ``<Molecule>`` below *el*."""
    return {m.get("name", "") for m in el.iter() if _local(m.tag) == "Molecule"}


def _constant_observables(root: ET.Element) -> set[str]:
    """Names of the observables in *root* whose count no rule can change.

    A pattern is constant when it is inert or a conserved total:

    * inert: no rule names any molecule type in it, and every seed species that
      holds one of those types holds only types no rule names. No rule can then
      bind, free, move, delete, merge or split anything in those complexes.
      Species observables and multi-molecule patterns are covered only this way,
      since a merge elsewhere in a complex changes them without naming their
      types.
    * a conserved total: a ``Molecules`` observable of one bare molecule ``X()``
      with no compartment, where no rule adds an X, deletes an X by name, or
      deletes a whole species that could hold an X, and no fixed (``$``) seed
      species holds an X. Two types can share a complex
      only if they share a seed species or a rule pattern, so a whole-species
      delete whose pattern's types never meet X's cannot take an X with it.
    """
    rules = [el for el in root.iter() if _local(el.tag) == "ReactionRule"]
    named: set[str] = set()
    mol_type: dict[str, str] = {}
    pattern_types: dict[str, set[str]] = {}
    for rule in rules:
        named |= _molecule_names(rule)
        for el in rule.iter():
            kind = _local(el.tag)
            if kind == "Molecule" and el.get("id"):
                mol_type[el.get("id", "")] = el.get("name", "")
            elif kind in ("ReactantPattern", "ProductPattern") and el.get("id"):
                pattern_types[el.get("id", "")] = _molecule_names(el)
    seed_species = [
        sp
        for lst in root.iter()
        if _local(lst.tag) == "ListOfSpecies"
        for sp in lst
        if _local(sp.tag) == "Species"
    ]
    seeds = [_molecule_names(sp) for sp in seed_species]
    # Types that can share a complex: joined through any seed species or pattern.
    partners: dict[str, set[str]] = {}
    for group in [*seeds, *pattern_types.values()]:
        for t in group:
            partners.setdefault(t, set()).update(group)

    def complex_mates(x: str) -> set[str]:
        seen, todo = {x}, [x]
        while todo:
            for u in partners.get(todo.pop(), ()):
                if u not in seen:
                    seen.add(u)
                    todo.append(u)
        return seen

    added: set[str] = set()
    deleted: set[str] = set()
    # A fixed ($) seed species is topped back up whenever it is used, so its
    # types are added (and, bound away, effectively removed) with no Add.
    for sp in seed_species:
        if sp.get("Fixed") == "1":
            added |= _molecule_names(sp)
            deleted |= _molecule_names(sp)
    species_deletes: list[set[str] | None] = []  # None = could not resolve
    for rule in rules:
        for el in rule.iter():
            kind = _local(el.tag)
            if kind == "Add":
                added.add(mol_type.get(el.get("id", ""), ""))  # "" = unknown: any type
            elif kind == "Delete":
                ref = el.get("id", "")
                if el.get("DeleteMolecules") == "1":
                    deleted |= pattern_types.get(ref) or {mol_type.get(ref, "")}
                else:
                    species_deletes.append(pattern_types.get(ref))

    def inert(types: set[str]) -> bool:
        return not (types & named) and all(not (seed & named) for seed in seeds if seed & types)

    def conserved_total(obs: ET.Element, pattern: ET.Element) -> bool:
        mols = [m for m in pattern.iter() if _local(m.tag) == "Molecule"]
        if (
            obs.get("type") != "Molecules"
            or len(mols) != 1
            or len(mols[0]) != 0
            or pattern.get("compartment")
            or mols[0].get("compartment")
        ):
            return False
        x = mols[0].get("name", "")
        if {x, ""} & added or {x, ""} & deleted:
            return False
        mates = complex_mates(x)
        return all(types is not None and not (types & mates) for types in species_deletes)

    constant = set()
    for obs in root.iter():
        if _local(obs.tag) != "Observable":
            continue
        patterns = [p for p in obs.iter() if _local(p.tag) == "Pattern"]
        if patterns and all(
            inert(_molecule_names(p)) or conserved_total(obs, p) for p in patterns
        ):
            constant.add(obs.get("name", ""))
    return constant


def _refs(f: ET.Element, kind: str) -> list[str]:
    """The names *f* lists among its references with type *kind*."""
    return [
        r.get("name", "")
        for lst in _children(f, "ListOfReferences")
        for r in lst
        if r.get("type") == kind
    ]


def _calls(f: ET.Element, funcs: dict[str, ET.Element]) -> list[str]:
    """The functions *f* reads: those it references, and its counter when that is
    a function."""
    ctr = f.get("ctrName", "")
    return _refs(f, "Function") + ([ctr] if ctr in funcs else [])


def _rate_law_reach(root: ET.Element, funcs: dict[str, ET.Element]) -> set[str]:
    """The functions a rule's rate law uses, directly or through the functions
    they call."""
    used: set[str] = set()
    todo = [
        name
        for rule in root.iter()
        if _local(rule.tag) == "ReactionRule"
        for law in rule
        if _local(law.tag) == "RateLaw"
        for key in ("name", "name1", "name2")
        if (name := law.get(key)) in funcs
    ]
    while todo:
        name = todo.pop()
        if name not in used:
            used.add(name)
            todo += [c for c in _calls(funcs[name], funcs) if c in funcs]
    return used


def _outside_reads(
    xml_path: str | Path,
) -> tuple[list[tuple[str, str]], set[str], dict[str, set[str]]] | None:
    """``(reads, constant, observable types)``: every ``(function, observable)``
    read outside an argument by a function with an argument that a rate law
    uses, the observables no rule can change, and each observable's molecule
    types. ``None`` when the file cannot be parsed."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    funcs = {f.get("id", ""): f for f in _functions(xml_path) or []}
    observables = {o.get("name", "") for o in root.iter() if _local(o.tag) == "Observable"}

    def args(f: ET.Element) -> list[str]:
        return [a.get("id", "") for lst in _children(f, "ListOfArguments") for a in lst]

    def counter_reads(f: ET.Element) -> set[str]:
        """What a table function's counter reads, if it changes: an observable,
        or the clock as NFsim reads it, which takes any case of ``time`` or
        ``t`` before a model symbol of that name (:func:`_nfsim_clock`)."""
        ctr = f.get("ctrName", "").strip()
        return {ctr} if ctr in observables or _nfsim_clock(ctr) else set()

    def outside(f: ET.Element) -> set[str]:
        """Observables *f* reads itself other than through its argument."""
        fargs = args(f)
        expr = " ".join(e.text or "" for e in _children(f, "Expression"))
        found = counter_reads(f)
        for name in _refs(f, "Observable"):
            uses = re.finditer(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])\s*(\(\s*([A-Za-z_]\w*)?\s*\))?",
                expr,
            )
            if any(m.group(1) is None or m.group(2) not in fargs for m in uses):
                found.add(name)
        return found

    def through_calls(f: ET.Element, seen: frozenset[str]) -> set[str]:
        """Observables *f* reaches outside any argument through the functions it calls."""
        found: set[str] = set()
        for name in _calls(f, funcs):
            g = funcs.get(name)
            if g is None or name in seen:
                continue
            if not args(g):
                found |= outside(g)  # a global function: every read is outside
            found |= through_calls(g, seen | {name})
        return found

    # Only functions a rate law uses can run stale; follow calls from there.
    used = _rate_law_reach(root, funcs)

    found = []
    for name, f in funcs.items():
        if name not in used or not args(f):
            continue  # unused, or global: re-evaluated whenever its observables change
        # A composite's own direct reads are NFsim's to refuse; its counter is not.
        reads = (
            through_calls(f, frozenset({name})) | counter_reads(f)
            if _calls(f, funcs)
            else outside(f)
        )
        found += [(name, o) for o in sorted(reads)]
    obs_types = {
        o.get("name", ""): _molecule_names(o) for o in root.iter() if _local(o.tag) == "Observable"
    }
    return found, _constant_observables(root), obs_types


def stale_local_function_reads(xml_path: str | Path) -> list[tuple[str, str]]:
    """``(function, observable)`` for each function with an argument that a rule's
    rate law uses in *xml_path* and that reads, outside its argument, an
    observable whose count can change. Empty when the file cannot be parsed.

    NFsim re-evaluates a function with an argument only on the complexes a
    reaction produces, so such a term goes stale on every other molecule
    (RuleWorld/nfsim#104). The read can be direct (``MAB`` or ``MAB()`` beside
    ``AC(x)`` in a local function), through a function it calls (``fB(x)*g()``
    with ``g() = 1+0.01*MAB``), or through a table function's counter
    (``tfun(..., MAB)`` or ``tfun(..., time)``, kept in ``ctrName``). A composite
    function's own direct read of an observable is left to NFsim, which refuses
    it at load; its table counter is not, since NFsim does not check that. Constant observables are
    allowed; see :func:`_constant_observables`, which errs toward calling a
    count changing.
    """
    out = _outside_reads(xml_path)
    if out is None:
        return []
    reads, constant, _ = out
    return [(f, o) for f, o in reads if o not in constant]


def constant_read_types(xml_path: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Molecule type -> the ``(function, observable)`` reads that
    :func:`stale_local_function_reads` lets run only because no rule can change
    that observable. A session edit of such a type's count would change it and
    leave those reads stale, so :class:`~bngsim.NfsimSession` refuses one."""
    out = _outside_reads(xml_path)
    if out is None:
        return {}
    reads, constant, obs_types = out
    guarded: dict[str, list[tuple[str, str]]] = {}
    for f, o in reads:
        if o in constant:
            for t in obs_types.get(o, ()):
                guarded.setdefault(t, []).append((f, o))
    return guarded


def refuse_nfsim_stale_local_functions(xml_path: str | Path) -> None:
    """Raise :class:`ModelError` if a local function in *xml_path* reads, outside
    its argument, an observable whose count can change. NFsim only."""
    reads = stale_local_function_reads(xml_path)
    if not reads:
        return
    label, _ = _BACKENDS["nfsim"]
    # BNG2.pl names the function behind an inline rate law _rateLawN; say whose.
    rule_of: dict[str, list[str]] = {}
    for rule, kind, law_name in _rate_law_functions(xml_path):
        if kind == "Function":
            rule_of.setdefault(law_name, []).append(rule)

    def who(f: str) -> str:
        rules = rule_of.get(f, [])
        return (
            f"'{f}' (the rate law of rule {', '.join(repr(r) for r in rules)})"
            if (f.startswith("_rateLaw") and rules)
            else f"'{f}'"
        )

    raise ModelError(
        f"{label} cannot run a model whose functions with an argument read an observable "
        "outside it, directly or through a function they call: "
        + ", ".join(f"{who(f)} reads '{o}'" for f, o in reads)
        + ". NFsim re-evaluates such a function only on the complexes a reaction "
        "produces, so that term goes stale on every other molecule, without an error "
        "(RuleWorld/nfsim#104; this refusal is lifted under issue #1062). Simulate the "
        "network BNG2.pl generates with method='ode' or method='ssa'."
    )


def refuse_nfsim_compartments(xml_path: str | Path) -> None:
    """Raise :class:`ModelError` if *xml_path* declares compartments. NFsim only:
    its compartment support is wrong in ways no check of the XML can bound."""
    comps = compartments(xml_path)
    if not comps:
        return
    label, _ = _BACKENDS["nfsim"]
    raise ModelError(
        f"{label} cannot run a compartment model (compartments "
        + ", ".join(f"'{c}'" for c in comps)
        + "). NFsim's compartment support runs such models wrong without an error: it "
        "ignores a species-level compartment on a pattern (@EC:A()), applies no volume "
        "factor to a rule with two or more reactants, gives no compartment to a molecule "
        "a rule creates with none written, puts every molecule of a seed complex in the "
        "species' compartment, binds across compartments that do not touch, and moves a "
        "whole complex on MoveConnected; and BNG2.pl's writeXML drops "
        "NumberPerQuantityUnit (RuleWorld/nfsim#106, #108, #109, #110, "
        "RuleWorld/bionetgen#348; this refusal is lifted under issue #1067). Simulate the "
        "network BNG2.pl generates with method='ode' or method='ssa'."
    )


def refuse_time_dependent_rate_laws(xml_path: str | Path, backend: str) -> None:
    """Raise :class:`ModelError` if *xml_path* has a function reading the clock.

    *backend* is ``"nfsim"``, the one network-free backend that cannot run one.
    """
    reads = time_reading_functions(xml_path)
    if not reads:
        return
    label, why = _BACKENDS[backend]
    raise ModelError(
        f"{label} cannot run a model whose functions read time(): "
        + ", ".join(f"'{n}'" for n in reads)
        + f". {why}. Simulate it with method='rm', or with method='ode' or "
        "method='ssa' on the generated network, which read time() as the model "
        "time (issue #862)."
    )


def _nfsim_clock(ctr: str) -> bool:
    """Whether NFsim reads a table function's counter as the clock: it trims and
    lowercases the name and checks it before any model symbol of that name
    (tfun_is_time_name in third_party/nfsim/src/NFinput/parseFuncXML.cpp)."""
    return ctr.strip().lower() in _RULEMONKEY_CLOCK_COUNTERS
