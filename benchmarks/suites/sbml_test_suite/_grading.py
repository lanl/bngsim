"""Shared, engine-agnostic grading kernel for the SBML semantic test suite.

Every engine adapter — bngsim, libRoadRunner, AMICI, COPASI — funnels its raw
output through *this* module, so the grading path is provably identical across
engines:

* the same solver tolerances (:data:`SOLVER_ATOL` / :data:`SOLVER_RTOL`),
* the same SBML-truth variable classification (species vs compartment vs
  parameter, read once from libSBML and shared),
* the same resolution ladder (:func:`resolve`),
* the same amount<->concentration conversion (``amount = conc(t) * vol(t)``,
  using the species' own compartment trajectory),
* the same comparison / shape / ``var_missing`` semantics (:func:`grade`).

The *only* per-engine code is the irreducible "read value X out of this engine's
native output" primitive set — an object exposing ``species_conc(id)`` and
``entity_value(id)`` (see :class:`EngineSeries`). Because the resolution ladder,
the conversion, and the comparison all live here and are called identically for
every engine, it is structurally impossible for one engine to be graded through
a more generous path than another (GH #225).
"""

from __future__ import annotations

import csv
import math
from typing import Protocol

import numpy as np

# ── Solver tolerances — ONE tuned pair, applied to every engine ──────────────
# The SBML Test Suite needs ~1e-8 absolute precision on species whose values can
# be ~1e-5, where a 1e-8 absolute tolerance leaves no headroom; 1e-12/1e-8 pulls
# borderline event-timing cases (e.g. 00652-00657) inside tolerance. The point
# for fairness is not the exact pair but that it is *identical* across engines —
# no engine is run at a tolerance the others are not. Defined here so all four
# adapters import the same constants.
SOLVER_ATOL = 1e-12
SOLVER_RTOL = 1e-8


def parse_settings(settings_path) -> dict:
    """Parse an SBML Test Suite ``*-settings.txt`` file."""
    s: dict[str, str] = {}
    with open(settings_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                s[key.strip()] = val.strip()
    return {
        "start": float(s.get("start", "0")),
        "duration": float(s.get("duration", "1")),
        "steps": int(s.get("steps", "50")),
        "variables": [v.strip() for v in s.get("variables", "").split(",") if v.strip()],
        "absolute": float(s.get("absolute", "1e-7")),
        "relative": float(s.get("relative", "1e-4")),
        "amount": [v.strip() for v in s.get("amount", "").split(",") if v.strip()],
        "concentration": [v.strip() for v in s.get("concentration", "").split(",") if v.strip()],
    }


def parse_results_csv(csv_path, settings: dict | None = None):
    """Parse an expected-results CSV. Returns ``(times, {var: values})``.

    The SBML Test Suite identifies the independent time axis by column position:
    it is the first column, commonly named ``time`` but sometimes ``Time``. Model
    output ids are case-sensitive and may themselves be named ``time``, ``Time``,
    or ``TIME``. Keep the axis separate from the remaining exact-case headers so
    those ids do not collide with the axis label.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        var_headers = header[1:]
        seen = set()
        duplicates = set()
        for h in var_headers:
            if h in seen:
                duplicates.add(h)
            seen.add(h)
        duplicates = sorted(duplicates)
        if duplicates:
            dup_list = ", ".join(repr(h) for h in duplicates)
            raise ValueError(f"duplicate expected-results variable column(s): {dup_list}")

        times: list[float] = []
        data: dict[str, list[float]] = {h: [] for h in var_headers}
        for row in reader:
            if not row:
                continue
            times.append(float(row[0]))
            for h, v in zip(var_headers, row[1:], strict=False):
                data[h].append(float(v))
    times_arr = np.array(times)
    if settings is not None:
        expected_times = np.linspace(
            settings["start"],
            settings["start"] + settings["duration"],
            settings["steps"] + 1,
        )
        if len(times_arr) != len(expected_times) or not np.allclose(
            times_arr, expected_times, rtol=1e-9, atol=1e-9
        ):
            raise ValueError("expected-results first column does not match the settings time grid")

    return times_arr, {k: np.array(v) for k, v in data.items()}


def read_sbml_entities(sbml_path) -> dict:
    """Classify model ids from libSBML — the SBML-truth used to route every
    engine's variable resolution identically.

    Returns a dict with ``species`` / ``compartments`` / ``parameters`` (sets of
    ids) and ``species_comp`` ({species_id: compartment_id}). When libSBML is
    unavailable or the parse fails the sets are empty; :func:`resolve` then falls
    back to trying both rungs, so a missing classification never silently routes
    one engine differently from another.
    """
    ent = {
        "species": set(),
        "compartments": set(),
        "parameters": set(),
        "species_comp": {},
        "comp_size": {},  # static t=0 compartment size — last-resort amount-conversion volume
        # Reaction-flux support (GH #249): an output variable may be a reaction id
        # whose expected value is the reaction *rate* (kinetic law). ``reactions``
        # maps each (comp-flattened) reaction id to a picklable mini-AST of its
        # kinetic law plus its local parameters; ``flat_species_hosu`` /
        # ``flat_species_comp`` are the flattened-model species facts the flux
        # evaluator needs to read a species reference at amount-vs-concentration.
        "reactions": {},
        "flat_species_hosu": {},
        "flat_species_comp": {},
        # Constant symbols an engine may not surface in its trajectory but that a
        # kinetic law can reference by id — currently constant <speciesReference>
        # stoichiometries (e.g. J1_sr in 01387). Static value ⇒ safe to read from
        # the model; time-varying speciesReferences are deliberately omitted.
        "flat_const_sym": {},
    }
    try:
        import libsbml

        doc = libsbml.readSBMLFromFile(str(sbml_path))
        m = doc.getModel()
        if m is None:
            return ent
        for i in range(m.getNumCompartments()):
            c = m.getCompartment(i)
            ent["compartments"].add(c.getId())
            ent["comp_size"][c.getId()] = c.getSize() if c.isSetSize() else 1.0
        for i in range(m.getNumParameters()):
            ent["parameters"].add(m.getParameter(i).getId())
        for i in range(m.getNumSpecies()):
            sp = m.getSpecies(i)
            ent["species"].add(sp.getId())
            ent["species_comp"][sp.getId()] = sp.getCompartment()
        _extract_reactions(doc, sbml_path, ent)
    except Exception:
        pass
    return ent


# ── Reaction-flux extraction + evaluation (GH #249) ──────────────────────────
# A handful of suite cases (the comp ``SubmodelOutput`` cluster, e.g. 01360)
# request a *reaction id* as an output variable; the expected value is the
# reaction rate, i.e. the kinetic law evaluated over the trajectory. No engine's
# native output surfaces the rate of a reaction that is not otherwise referenced,
# so the shared kernel derives it — identically for every engine — from that
# engine's own reported species / parameter / compartment trajectories. This is
# fair by construction (one evaluator, one code path) and fails *closed*: any law
# the small evaluator does not support resolves to ``None`` → an honest
# ``var_missing``, never a guessed value.


class _FluxUnsupported(Exception):
    """The kinetic law (or a symbol in it) is outside the flux evaluator's
    supported subset — grade the variable as ``var_missing``, never a guess."""


def _ast_to_mini(node):
    """Convert a libSBML kinetic-law ``ASTNode`` into a picklable nested tuple.

    Fork-safe (COPASI runs in a child) and libSBML-free at eval time. Numeric
    constants (``e`` / ``pi`` / avogadro) fold to numbers; anything the evaluator
    cannot handle becomes an ``("unsupported", ...)`` leaf that raises on eval.
    """
    import libsbml

    t = node.getType()
    if t == libsbml.AST_INTEGER:
        return ("num", float(node.getInteger()))
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
        return ("num", float(node.getReal()))
    if t == libsbml.AST_NAME_AVOGADRO:
        return ("num", 6.02214076e23)
    if t == libsbml.AST_CONSTANT_E:
        return ("num", math.e)
    if t == libsbml.AST_CONSTANT_PI:
        return ("num", math.pi)
    if t == libsbml.AST_CONSTANT_TRUE:
        return ("num", 1.0)
    if t == libsbml.AST_CONSTANT_FALSE:
        return ("num", 0.0)
    if t == libsbml.AST_NAME_TIME:
        return ("time",)
    if t == libsbml.AST_NAME:
        return ("sym", node.getName())
    op = _AST_OPS.get(t)
    if op is None:
        return ("unsupported", int(t))
    return ("call", op, [_ast_to_mini(node.getChild(i)) for i in range(node.getNumChildren())])


def _build_ast_ops():
    import libsbml

    return {
        libsbml.AST_PLUS: "+",
        libsbml.AST_MINUS: "-",
        libsbml.AST_TIMES: "*",
        libsbml.AST_DIVIDE: "/",
        libsbml.AST_POWER: "^",
        libsbml.AST_FUNCTION_POWER: "^",
        libsbml.AST_FUNCTION_EXP: "exp",
        libsbml.AST_FUNCTION_LN: "ln",
        libsbml.AST_FUNCTION_LOG: "log",
        libsbml.AST_FUNCTION_ROOT: "root",
        libsbml.AST_FUNCTION_ABS: "abs",
        libsbml.AST_FUNCTION_PIECEWISE: "piecewise",
        libsbml.AST_RELATIONAL_LT: "lt",
        libsbml.AST_RELATIONAL_LEQ: "leq",
        libsbml.AST_RELATIONAL_GT: "gt",
        libsbml.AST_RELATIONAL_GEQ: "geq",
        libsbml.AST_RELATIONAL_EQ: "eq",
        libsbml.AST_RELATIONAL_NEQ: "neq",
    }


try:  # built once; libSBML is optional (empty map ⇒ only literals/symbols supported)
    _AST_OPS = _build_ast_ops()
except Exception:
    _AST_OPS = {}


def _extract_reactions(doc, sbml_path, ent) -> None:
    """Populate ``ent['reactions']`` with each reaction id → kinetic-law mini-AST.

    Comp models are flattened first (the engines and the expected CSV both use
    flattened, submodel-prefixed reaction ids such as ``A__J2``); a flatten
    failure just leaves submodel reactions unfound, i.e. a benign var_missing.
    Flattening mutates the document, so comp models are re-read into a fresh doc
    to leave the caller's ``doc`` intact; the common non-comp case reuses it.
    """
    import libsbml

    if doc.getPlugin("comp") is not None:
        doc = libsbml.readSBMLFromFile(str(sbml_path))
        props = libsbml.ConversionProperties()
        props.addOption("flatten comp", True, "flatten comp")
        if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
            return
    m = doc.getModel()
    if m is None:
        return
    for i in range(m.getNumSpecies()):
        sp = m.getSpecies(i)
        ent["flat_species_hosu"][sp.getId()] = bool(sp.getHasOnlySubstanceUnits())
        ent["flat_species_comp"][sp.getId()] = sp.getCompartment()
    for i in range(m.getNumCompartments()):
        c = m.getCompartment(i)
        ent["comp_size"].setdefault(c.getId(), c.getSize() if c.isSetSize() else 1.0)
    for i in range(m.getNumReactions()):
        rx = m.getReaction(i)
        # Named constant <speciesReference> ids resolve to their (fixed)
        # stoichiometry when used as a symbol — e.g. A__J2 = J1_sr in 01387.
        for refs in (rx.getListOfReactants(), rx.getListOfProducts()):
            for sr in refs:
                if sr.isSetId() and sr.getConstant() and sr.isSetStoichiometry():
                    ent["flat_const_sym"][sr.getId()] = sr.getStoichiometry()
        kl = rx.getKineticLaw()
        if kl is None or kl.getMath() is None:
            continue
        local = {}
        for j in range(kl.getNumLocalParameters()):
            lp = kl.getLocalParameter(j)
            local[lp.getId()] = lp.getValue()
        for j in range(kl.getNumParameters()):  # L2 spelling of local parameters
            p = kl.getParameter(j)
            local.setdefault(p.getId(), p.getValue())
        ent["reactions"][rx.getId()] = {"ast": _ast_to_mini(kl.getMath()), "local": local}


def _flux_eval(mini, resolve_sym, n, times):
    """Vectorised evaluation of a mini-AST to an ``np.ndarray`` of length ``n``.

    Unsupported nodes raise :class:`_FluxUnsupported`; the caller turns that into
    ``None`` (var_missing) so an un-evaluable law is never scored as a guess.
    """
    kind = mini[0]
    if kind == "num":
        return np.full(n, mini[1], dtype=float)
    if kind == "time":
        return np.asarray(times, dtype=float)
    if kind == "sym":
        return resolve_sym(mini[1])
    if kind == "unsupported":
        raise _FluxUnsupported(f"AST type {mini[1]}")
    op, children = mini[1], mini[2]
    a = [_flux_eval(c, resolve_sym, n, times) for c in children]
    with np.errstate(all="ignore"):
        if op == "+":
            return sum(a) if a else np.zeros(n)
        if op == "*":
            out = np.ones(n)
            for x in a:
                out = out * x
            return out
        if op == "-":
            return -a[0] if len(a) == 1 else a[0] - sum(a[1:])
        if op == "/":
            return a[0] / a[1]
        if op == "^":
            return np.power(a[0], a[1])
        if op == "exp":
            return np.exp(a[0])
        if op == "ln":
            return np.log(a[0])
        if op == "log":  # 1 child ⇒ log10; 2 children ⇒ log base a[0] of a[1]
            return np.log10(a[0]) if len(a) == 1 else np.log(a[1]) / np.log(a[0])
        if op == "root":  # 2 children ⇒ a[0]-th root of a[1]
            return np.power(a[1], 1.0 / a[0]) if len(a) == 2 else np.sqrt(a[0])
        if op == "abs":
            return np.abs(a[0])
        if op in ("lt", "leq", "gt", "geq", "eq", "neq"):
            cmp = {
                "lt": np.less,
                "leq": np.less_equal,
                "gt": np.greater,
                "geq": np.greater_equal,
                "eq": np.equal,
                "neq": np.not_equal,
            }[op]
            return cmp(a[0], a[1]).astype(float)
        if op == "piecewise":  # value0, cond0, value1, cond1, ..., [default]
            out = np.full(n, np.nan)
            i = 0
            while i + 1 < len(a):
                out = np.where(a[i + 1] != 0.0, a[i], out)
                i += 2
            if len(a) % 2 == 1:  # trailing otherwise-value
                out = np.where(np.isnan(out), a[-1], out)
            return out
    raise _FluxUnsupported(f"operator {op}")


def resolve_reaction_flux(var, series, ent, times, _stack=None):
    """Reaction rate of ``var`` over the trajectory, or ``None`` if ``var`` is not
    a reaction or its law falls outside the evaluator's supported subset.

    A kinetic law may reference another reaction's rate by id (01351) or a
    constant ``<speciesReference>`` id (01387); those are resolved recursively /
    from the model, so the whole family is graded through one uniform path. A
    reaction that (transitively) references itself raises ``_FluxUnsupported`` and
    is graded ``var_missing`` rather than looping.
    """
    rinfo = ent.get("reactions", {}).get(var)
    if rinfo is None:
        return None
    n = len(times)
    local = rinfo["local"]
    hosu = ent.get("flat_species_hosu", {})
    stack = (_stack or frozenset()) | {var}

    def resolve_sym(name):
        if name in local:
            return np.full(n, float(local[name]))
        if name in hosu:  # a species: kinetic laws read amount iff hasOnlySubstanceUnits
            conc = series.species_conc(name)
            if conc is None:
                raise _FluxUnsupported(f"species {name} not reported")
            conc = np.asarray(conc, dtype=float)
            if not hosu[name]:
                return conc
            comp = ent.get("flat_species_comp", {}).get(name)
            vol = series.entity_value(comp) if comp else None
            if vol is None:
                vol = np.full(n, ent.get("comp_size", {}).get(comp, 1.0))
            return conc * np.asarray(vol, dtype=float)
        # Prefer the engine's own reported value (parameter / compartment / an
        # engine-materialised rate symbol) so we grade its real trajectory.
        v = series.entity_value(name)
        if v is not None:
            v = np.asarray(v, dtype=float)
            return np.full(n, float(v)) if v.ndim == 0 else v
        # A referenced reaction's rate the engine does not surface: recurse.
        if name in ent.get("reactions", {}):
            if name in stack:
                raise _FluxUnsupported(f"reaction {name} references itself")
            sub = resolve_reaction_flux(name, series, ent, times, _stack=stack)
            if sub is None:
                raise _FluxUnsupported(f"reaction {name} rate unresolved")
            return sub
        # A constant speciesReference stoichiometry referenced as a symbol.
        if name in ent.get("flat_const_sym", {}):
            return np.full(n, float(ent["flat_const_sym"][name]))
        raise _FluxUnsupported(f"symbol {name} unresolved")

    try:
        arr = _flux_eval(rinfo["ast"], resolve_sym, n, times)
    except _FluxUnsupported:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    return arr if len(arr) == n else None


def interpretation(var: str, settings: dict) -> str:
    """Whether the suite expects ``var`` as an ``amount``, a ``concentration``,
    or a bare ``raw`` value (parameters / compartments appear in neither list)."""
    if var in settings["amount"]:
        return "amount"
    if var in settings["concentration"]:
        return "concentration"
    return "raw"


def compare_series(actual, expected, atol, rtol):
    """Check ``actual`` against ``expected`` within the suite's per-case
    tolerances. Returns ``(passed, max_rel_err)``.

    Finite expected values use the SBML Test Suite rule
    ``|actual - expected| <= atol + rtol*|expected|``. The reported error is a
    relative-error metric for the report, never used for the verdict itself
    (which is the tolerance test above).

    Non-finite expected values (``INF`` / ``-INF`` / ``NaN``) are *exact
    sentinels*, not numbers to compare within a tolerance (GH #247, cases
    00950/00951/01811/01813). The tolerance rule cannot express them: an
    expected ``inf`` makes the tolerance ``inf`` but ``actual - inf`` is ``NaN``
    (``inf - inf``), and every comparison against ``NaN`` is False, so a
    correct engine would be graded as a mismatch. Grade each such element by an
    exact IEEE match instead — an expected ``NaN`` requires an actual ``NaN``;
    an expected ``+inf`` / ``-inf`` requires the actual to be that same-signed
    infinity. This is value-based, not column-name based, so an id merely
    *spelled* ``INF`` / ``NaN`` (01811/01813) whose data is finite is still
    graded by the normal tolerance test.
    """
    if len(actual) != len(expected):
        return False, float("inf")
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    finite = np.isfinite(expected)

    # Sentinel elements: exact IEEE match. ``==`` gives +inf==+inf / -inf==-inf
    # True and +inf==-inf / value-vs-inf False; NaN is matched via isnan since
    # ``NaN == NaN`` is False.
    exp_nf = expected[~finite]
    act_nf = actual[~finite]
    sentinel_ok = bool(np.all(np.where(np.isnan(exp_nf), np.isnan(act_nf), act_nf == exp_nf)))

    # Finite elements: the standard suite tolerance test.
    a = actual[finite]
    e = expected[finite]
    diffs = np.abs(a - e)
    tol = atol + rtol * np.abs(e)
    finite_ok = bool(np.all(diffs <= tol))

    within = sentinel_ok and finite_ok
    if e.size:
        denom = np.maximum(np.abs(e), atol)
        rel = float(np.max(diffs / denom))
    else:
        rel = 0.0
    return within, rel


class EngineSeries(Protocol):
    """Per-engine primitive set the shared grader is allowed to call.

    An adapter constructs one of these *after* it has run its engine; the shared
    :func:`resolve` / :func:`grade` call ONLY these two methods, so no engine can
    reach a value the contract does not also offer the others.
    """

    def species_conc(self, sbml_id: str):
        """Concentration trajectory of species ``sbml_id`` (``np.ndarray`` of
        length ``n_points``), or ``None`` if this engine cannot report it."""
        ...

    def entity_value(self, sbml_id: str):
        """Trajectory of a non-species entity (compartment / parameter / other
        assignment target) named ``sbml_id``, or ``None``. For a constant the
        adapter returns a broadcast array, so the grader treats every value as a
        time series uniformly."""
        ...


def resolve(var: str, series: EngineSeries, ent: dict, times):
    """The shared resolution ladder. Returns ``(array_or_None, is_species)``.

    SBML-truth classification (``ent``) routes the lookup so every engine follows
    the same order:

    1. if ``var`` is a species → its concentration via ``species_conc``;
    2. else (compartment / parameter / assignment target) → ``entity_value``;
    3. last resort, if classification was unavailable → try ``species_conc``;
    4. reaction id → its kinetic-law rate over this engine's own trajectory.

    ``is_species`` gates the amount conversion in :func:`grade` — only true
    species are multiplied by their compartment volume.
    """
    if var in ent["species"]:
        a = series.species_conc(var)
        if a is not None:
            return a, True
    a = series.entity_value(var)
    if a is not None:
        return a, False
    # Classification unavailable (empty ent) or entity path empty: try the
    # species rung even if not classified, so a missing libSBML parse never
    # routes one engine differently from another.
    a = series.species_conc(var)
    if a is not None:
        return a, True
    # Reaction flux (GH #249). Placed last so an engine that natively surfaces a
    # referenced reaction's rate symbol (e.g. bngsim materialises it as an
    # expression) is still graded through its own reported value; this covers the
    # pure-output reactions no engine column exposes. Derived identically for
    # every engine from its own trajectory, so it cannot favour one over another.
    a = resolve_reaction_flux(var, series, ent, times)
    if a is not None:
        return a, False
    return None, False


def _time_grid(settings: dict):
    """The suite output time grid (``start … start+duration`` in ``steps+1``
    points), or ``None`` if the settings dict omits any of the three keys.

    Needed by the reaction-flux rung (time csymbol / broadcasting a constant rate
    to the trajectory). A caller that passes a minimal settings dict without the
    grid keys gets ``None`` and the callers fall back below.
    """
    if all(k in settings for k in ("start", "duration", "steps")):
        return np.linspace(
            settings["start"],
            settings["start"] + settings["duration"],
            settings["steps"] + 1,
        )
    return None


def _resolve_graded(var: str, settings: dict, series: EngineSeries, ent: dict, times):
    """Resolve one variable to its final graded column, or ``None`` if the engine
    cannot report it.

    This is the single resolve + amount-conversion primitive shared by
    :func:`grade` (which then compares) and :func:`resolve_columns` (which writes
    a CSV): the resolution ladder and the ``amount = conc(t)·vol(t)`` conversion
    live here once, so a value the comparator grades and a value the test-runner
    wrapper delivers are computed through byte-identical code.
    """
    arr, is_species = resolve(var, series, ent, times)
    if arr is None:
        return None

    # Amount conversion — IDENTICAL for every engine: a species expected as an
    # amount is its concentration times its own compartment's volume trajectory,
    # vol(t). For a constant compartment this is conc * size; for a rate-ruled /
    # assignment-driven compartment it is the elementwise product, which is why we
    # use the volume *trajectory* and not a t=0 scalar (the latter silently
    # mis-converts time-varying-volume cases — e.g. 00051 — and did so
    # asymmetrically before #225).
    if is_species and interpretation(var, settings) == "amount":
        comp = ent["species_comp"].get(var)
        vol = series.entity_value(comp) if comp else None
        if vol is None and comp:
            # The engine did not surface this compartment (e.g. a constant
            # compartment some engines omit from their output). Fall back to the
            # static SBML size so the amount conversion still happens —
            # identically for every engine — rather than silently comparing a
            # concentration against an expected amount.
            vol = np.full(len(arr), ent["comp_size"].get(comp, 1.0))
        if vol is not None:
            arr = arr * vol

    return np.asarray(arr, dtype=float)


def resolve_columns(settings: dict, series: EngineSeries, ent: dict) -> dict:
    """Resolve every requested ``variables:`` entry to its final graded column.

    Returns ``{var: np.ndarray}`` for the variables this engine can produce; a
    variable the engine cannot resolve is **omitted, never fabricated**, so a
    delivered CSV built from this dict is missing exactly the columns the engine
    genuinely could not report. The SBML Test Suite test-runner wrapper (GH #241)
    builds its CSV from this dict and relies on the official grader's
    ``requireAllColumns`` to turn an omitted column into a ``NoMatch`` — which is
    the honest outcome, identical to :func:`grade`'s ``var_missing``.
    """
    grid = _time_grid(settings)
    if grid is None and settings.get("steps") is not None:
        grid = np.arange(settings["steps"] + 1, dtype=float)
    columns: dict[str, np.ndarray] = {}
    for var in settings["variables"]:
        times = grid if grid is not None else np.zeros(0, dtype=float)
        arr = _resolve_graded(var, settings, series, ent, times)
        if arr is not None:
            columns[var] = arr
    return columns


def grade(settings: dict, exp_data: dict, series: EngineSeries, ent: dict) -> dict:
    """Grade one engine's run against the expected CSV. Engine-independent.

    Walks the suite's requested ``variables`` in order; for each it resolves the
    value through the shared :func:`_resolve_graded` primitive (resolution ladder
    + amount conversion) and compares within the per-case tolerances. Returns
    ``{status, max_err, error}`` where ``status`` is one of ``pass`` /
    ``value_mismatch`` / ``var_missing`` / ``shape_mismatch``.
    """
    all_pass = True
    max_err_overall = 0.0
    grid = _time_grid(settings)

    for var in settings["variables"]:
        if var not in exp_data:
            continue
        expected = exp_data[var]

        times = grid if grid is not None else np.arange(len(expected), dtype=float)
        arr = _resolve_graded(var, settings, series, ent, times)
        if arr is None:
            return {
                "status": "var_missing",
                "error": f"variable '{var}' not found in output",
                "max_err": max_err_overall,
            }

        if len(arr) != len(expected):
            return {
                "status": "shape_mismatch",
                "error": f"var '{var}': got {len(arr)} points, expected {len(expected)}",
                "max_err": max_err_overall,
            }

        ok, max_err = compare_series(arr, expected, settings["absolute"], settings["relative"])
        max_err_overall = max(max_err_overall, max_err)
        if not ok:
            all_pass = False

    if all_pass:
        return {"status": "pass", "max_err": max_err_overall, "error": ""}
    return {
        "status": "value_mismatch",
        "error": f"max_err={max_err_overall:.6g}",
        "max_err": max_err_overall,
    }
