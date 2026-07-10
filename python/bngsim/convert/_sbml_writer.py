"""bngsim.convert._sbml_writer — serialize an in-memory model to SBML (GH #216).

The reverse of :mod:`bngsim._sbml_loader`: take a loaded :class:`bngsim.Model`
(typically produced by ``Model.from_net``) and emit a Level-3 Version-2 SBML
document that ``Model.from_sbml`` reads back. Drives off
``model._core.codegen_data()`` (the authoritative in-memory schema) plus
``model.get_state()`` (species initial values, which ``codegen_data`` omits).

Scope is the **network channel** only — species, reactions, parameters,
observables, functions, compartments. Unlike the ``.net`` text format
(:mod:`bngsim.convert._net_writer`), SBML *can* express per-species
amount/concentration semantics and non-unit compartment volumes, so net2sbml is
the more faithful half of the round-trip: a :class:`bngsim.Model` that still
carries ``volume_factor``/``amount_valued`` (e.g. one loaded straight from
SBML) is reconstructed with the right compartments and unit kinds. Constructs
SBML cannot carry (events, live/time-varying volumes, ``tfun`` table-function
calls) are surfaced via :func:`sbml_capability_report` and raise
:class:`bngsim.ConversionError` (or warn under ``strict=False``).

Design notes
------------
* **Symbol namespaces.** In the engine a species is a BNGL *pattern*
  (``A()``, ``LTCC(b,g~C,...)``) and is referenced by *index* only — by
  reactions and by observable entries — never by name in any expression
  string. Function/observable expressions reference *observables* and
  *parameters* by bare name. So species SBML ids are purely internal: we
  sanitize them to valid ``SId`` tokens (originals preserved in the ``name``
  attribute) and wire reactions/observables to them by index. Parameters,
  observables and functions share the engine's one symbol namespace, so their
  ids map straight through (after ``SId`` sanitization for the rare
  pattern-charactered name).

* **Function shadows.** The ``.net`` builder auto-creates a constant
  *shadow parameter* per function (and the engine reports both). We collapse
  each function to a single SBML ``<parameter>`` + ``<assignmentRule>`` and
  drop the shadow constant so the namespace stays single-valued.

* **Expression dialect.** Function/observable bodies are engine *ExprTk*
  strings. We translate them to MathML via a small normalize-then-parse step
  (:func:`_exprtk_to_ast`): rewrite ``if(c,a,b)`` → ``piecewise(a,c,b)``,
  ``time()`` → the SBML time csymbol, ``_pi`` → ``pi``, ExprTk ``log`` (natural
  log) → SBML ``ln`` (SBML ``log`` is base-10!), and the base-2/``log1p`` calls
  that have no MathML primitive to their exact ``ln`` identities
  (``log2(x)`` → ``ln(x)/ln(2)``, ``log1p(x)`` → ``ln(1+x)``), then
  ``libsbml.parseL3Formula``. Calls outside the supported set (notably ``tfun``)
  fail loud.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bngsim._exceptions import ConversionError, ConversionWarning

if TYPE_CHECKING:
    from bngsim._model import Model

# SBML Level/Version we emit. L3v2 has the richest MathML (piecewise, min/max)
# and is what the loader is happiest reading back.
_SBML_LEVEL = 3
_SBML_VERSION = 2

# Sentinel name standing in for the SBML time csymbol while we go through the
# L3 infix parser (which has no infix spelling for it). Chosen so it cannot
# collide with a real model symbol.
_TIME_SENTINEL = "__bngsim_csymbol_time__"

# libsbml's parseL3Formula recognizes the bareword ``pi`` — *case-insensitively*
# (``pi``/``pI``/``PI``/``Pi``) — as the π constant, silently collapsing a model
# symbol so named into ``<pi/>`` (a wrong-RHS we must never emit). In BNGL/ExprTk
# π is spelled ``_pi``, so after normalization a real π only ever arrives via this
# sentinel name; any π *constant* the parser then produces is a misparsed symbol
# we revert back to a name reference. (See :func:`_walk_fix_pi`.)
_PI_SENTINEL = "__bngsim_const_pi__"

# ExprTk function/operator-keyword names we can translate to MathML. Anything
# else spelled as ``name(`` in an expression is refused (fail-loud) — most
# importantly ``tfun`` (table-function interpolation), which has no SBML form.
_EXPRTK_SUPPORTED_CALLS = frozenset(
    {
        # arithmetic / power
        "pow",
        "sqrt",
        "abs",
        "exp",
        "ln",  # appears only post-rewrite, but allow it
        "log",  # ExprTk natural log → rewritten to ln before the scan
        "log10",
        # trig / hyperbolic (loader emits only this direct set; exotic funcs
        # are pre-expanded by the loader into these + sqrt/ln)
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "floor",
        "ceil",
        "ceiling",
        # L3v2 n-ary
        "min",
        "max",
        # logical / conditional (post-rewrite forms)
        "and",
        "or",
        "not",
        "piecewise",
        # time csymbol (zero-arg)
        "time",
    }
)


# ─── SId sanitization ──────────────────────────────────────────────────────


_SID_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SID_BAD = re.compile(r"[^A-Za-z0-9_]")

# Barewords that ``libsbml.parseL3Formula`` reads as a MathML constant / csymbol
# rather than a symbol reference — *case-insensitively* (``pi``/``PI``/``Pi`` all
# collapse to ``<pi/>``, ``time`` to the time csymbol, ``avogadro`` to the
# Avogadro csymbol, ``nan``/``inf``/``true``/``false``/``exponentiale`` to their
# literals). A model symbol whose sanitized ``SId`` equals one of these would be
# silently mis-lowered wherever it is referenced by that ``SId`` in an
# assignment-rule sum or kinetic-law formula (GH #8: a species ``pi()`` →
# observable ``Ptot`` ≡ 3.14159 instead of the population count). We bump such an
# ``SId`` (``pi`` → ``pi_2``) at allocation so no emitted formula token can ever
# resolve to a constant, fixing every downstream reference at once. Compared
# lowercased. (``_walk_fix_pi`` still handles genuine π from the ExprTk ``_pi``
# sentinel; this is the complementary guard for the SId-reference paths.)
_MATHML_RESERVED_SIDS = frozenset(
    {
        "pi",
        "exponentiale",
        "avogadro",
        "time",
        "nan",
        "notanumber",
        "inf",
        "infinity",
        "true",
        "false",
    }
)


def _dedup_functions(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse functions that repeat the same name *and* expression to one.

    A ``.net`` synthesized from a cross-compartment volume-scaled model can carry
    the same helper function (e.g. ``_vd_reaction_1_compartment_1``) many times
    over — the engine keys functions by index and tolerates the duplication, but
    SBML keys by ``SId``, so emitting each as its own parameter produces a
    document with duplicate ids (which bngsim's own ``from_sbml`` then rejects).
    Functions share one symbol namespace keyed by name, so two entries with the
    same name *must* denote the same function; we keep the first and drop exact
    repeats. A name reused with a *different* expression is a genuine conflict —
    left in place so it surfaces downstream rather than being silently merged.
    """
    seen: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for f in functions:
        nm = f["name"]
        expr = f.get("expression", "")
        if nm in seen:
            if seen[nm] == expr:
                continue  # exact repeat — already emitted
        else:
            seen[nm] = expr
        out.append(f)
    return out


def _unique_name(name: str, used: set[str]) -> str:
    """Return ``name`` made unique against ``used`` by appending ``_2``, ``_3``, …

    SBML element ``name`` is a free-text display label (the math references the
    ``SId``, never the name), so a collision is cosmetic for SBML itself — but
    bngsim's own ``from_sbml`` keys symbols by name, so two parameters sharing a
    name make the emitted document fail to reload. A best-effort ``.net`` (e.g.
    one carrying synthesized wrapper symbols that duplicate a name) must still
    round-trip through the reader; for already-unique names this is a no-op.
    """
    out = name
    n = 2
    while out in used:
        out = f"{name}_{n}"
        n += 1
    used.add(out)
    return out


def _sanitize_sid(name: str, used: set[str], *, fallback: str) -> str:
    """Return a valid, unique SBML ``SId`` derived from ``name``.

    Valid ``SId`` is ``[A-Za-z_][A-Za-z0-9_]*``. Pattern characters
    (``() ~ , . !`` …) are replaced with ``_``; a name that starts with a digit
    (or empties out) is prefixed with ``fallback``. Uniqueness is enforced by
    appending ``_2``, ``_3``, … on collision, and the same bump avoids the MathML
    reserved constant barewords (:data:`_MATHML_RESERVED_SIDS`) case-insensitively
    so no ``SId`` can be mis-read as ``<pi/>`` / the time csymbol / etc. (GH #8).
    """
    base = _SID_BAD.sub("_", name).strip("_")
    if not base or not base[0].isalpha() and base[0] != "_":
        base = f"{fallback}{base}" if base else fallback
    if not _SID_OK.match(base):  # paranoia; e.g. all-underscore collapse
        base = fallback
    sid = base
    n = 2
    while sid in used or sid.lower() in _MATHML_RESERVED_SIDS:
        sid = f"{base}_{n}"
        n += 1
    used.add(sid)
    return sid


# ─── ExprTk → MathML ───────────────────────────────────────────────────────


def _split_top_level(s: str) -> list[str]:
    """Split ``s`` on commas that sit at bracket-depth 0."""
    args: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur or args:
        args.append("".join(cur))
    return args


def _match_paren(s: str, open_idx: int) -> int:
    """Index of the ``)`` matching the ``(`` at ``s[open_idx]``."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ConversionError(f"unbalanced parentheses in expression: {s!r}")


_IF_TOKEN = re.compile(r"(?<![A-Za-z0-9_])if\s*\(")


def _rewrite_if_to_piecewise(expr: str) -> str:
    """Rewrite every ExprTk ``if(cond, a, b)`` to L3 ``piecewise(a, cond, b)``.

    Handles nesting by always rewriting the *first* ``if(`` and recursing on
    the rewritten tail, so inner ``if``s (now inside the emitted ``piecewise``
    arguments) are picked up on subsequent passes.
    """
    m = _IF_TOKEN.search(expr)
    if not m:
        return expr
    open_idx = m.end() - 1
    close_idx = _match_paren(expr, open_idx)
    inner = expr[open_idx + 1 : close_idx]
    parts = _split_top_level(inner)
    if len(parts) != 3:
        raise ConversionError(f"if(...) expects 3 arguments, got {len(parts)} in {expr!r}")
    cond, a, b = (p.strip() for p in parts)
    piece = f"piecewise(({a}), ({cond}), ({b}))"
    rebuilt = expr[: m.start()] + piece + expr[close_idx + 1 :]
    return _rewrite_if_to_piecewise(rebuilt)


# token: identifier immediately followed by '(' → a function call
_CALL_TOKEN = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
# ExprTk natural-log `log(` but NOT log10(/log1p(/log2( ; replace with ln(
_LOG_TOKEN = re.compile(r"(?<![A-Za-z0-9_])log\s*\(")
_PI_TOKEN = re.compile(r"(?<![A-Za-z0-9_])_pi(?![A-Za-z0-9_])")
_TIME_CALL = re.compile(r"(?<![A-Za-z0-9_])time\s*\(\s*\)")


_ZERO_ARG_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)")


def _strip_zero_arg_calls(expr: str, scalar_names: frozenset[str]) -> str:
    """Rewrite ``obs()`` → ``obs`` for any name that is a scalar model symbol.

    BNGL accepts an observable (or other scalar) referenced as a zero-arg call
    (``Atot()``) anywhere the bareword is valid (BNG issue #28); the engine
    stores the call verbatim in a function body. SBML/MathML has no zero-arg
    call form, so collapse it to the bare symbol. ``time()`` is left alone (it
    is the SBML time csymbol, handled separately).
    """

    def _repl(m: re.Match) -> str:
        name = m.group(1)
        return name if name in scalar_names else m.group(0)

    return _ZERO_ARG_CALL.sub(_repl, expr)


_LOG2_TOKEN = re.compile(r"(?<![A-Za-z0-9_])log2\s*\(")
_LOG1P_TOKEN = re.compile(r"(?<![A-Za-z0-9_])log1p\s*\(")


def _rewrite_unary_call(expr: str, token: re.Pattern, build: Callable[[str], str]) -> str:
    """Rewrite every ``name(arg)`` unary call matched by ``token`` via ``build``.

    Paren-matched (so a nested-parenthesized ``arg`` is captured whole) and
    recursive over both the argument and the tail, so nested same-name calls are
    all rewritten. ``build(inner)`` returns the replacement infix.
    """
    m = token.search(expr)
    if not m:
        return expr
    open_idx = m.end() - 1
    close_idx = _match_paren(expr, open_idx)
    inner = _rewrite_unary_call(expr[open_idx + 1 : close_idx], token, build)
    tail = _rewrite_unary_call(expr[close_idx + 1 :], token, build)
    return expr[: m.start()] + build(inner) + tail


def _normalize_exprtk(expr: str, scalar_names: frozenset[str]) -> str:
    """ExprTk infix → L3-parseable infix (pure string transforms)."""
    # 0) obs() → obs for scalar model symbols (BNG zero-arg-call syntax)
    out = _strip_zero_arg_calls(expr, scalar_names)
    # 0b) ExprTk infix booleans → L3 infix: ``and``/``or`` are keyword operators
    #     in ExprTk but parseL3Formula only accepts ``&&``/``||`` (the bareword
    #     ``and``/``or`` is a syntax error). Word-boundary-anchored so a symbol that
    #     merely *contains* the letters (``ligand``, ``factor``) is untouched, and
    #     so the ``and`` inside ``nand`` is not split. Other ExprTk boolean
    #     keywords (nand/nor/xor/not) are left to fail-loud at parse rather than be
    #     silently mistranslated. Common in time-gated forcing functions
    #     (``if((time()>=a) and (time()<=b), …)``).
    out = re.sub(r"(?<![A-Za-z0-9_])and(?![A-Za-z0-9_])", "&&", out)
    out = re.sub(r"(?<![A-Za-z0-9_])or(?![A-Za-z0-9_])", "||", out)
    # 1) if(...) → piecewise(...)
    out = _rewrite_if_to_piecewise(out)
    # 1b) base-2 / log1p have no MathML primitive but reduce to ln exactly:
    #     log2(x) → ln(x)/ln(2);  log1p(x) → ln(1+x). Done before the log(→ln(
    #     step (these contain "log" but the trailing digit/`1p` shields them from
    #     _LOG_TOKEN, so order is for clarity, not correctness).
    out = _rewrite_unary_call(out, _LOG2_TOKEN, lambda a: f"(ln({a}) / ln(2))")
    out = _rewrite_unary_call(out, _LOG1P_TOKEN, lambda a: f"ln(1 + ({a}))")
    # 2) protect log10(, then ExprTk log( (natural) → ln(
    out = re.sub(r"(?<![A-Za-z0-9_])log10\s*\(", "__LOG10__(", out)
    out = _LOG_TOKEN.sub("ln(", out)
    out = out.replace("__LOG10__(", "log10(")
    # 3) _pi → a sentinel name (restored to the π constant after parsing, so the
    #    parser never confuses genuine π with a pi-spelled model symbol).
    out = _PI_TOKEN.sub(_PI_SENTINEL, out)
    # 4) time() → sentinel name (converted to the time csymbol after parsing)
    out = _TIME_CALL.sub(_TIME_SENTINEL, out)
    return out


def _assert_supported_calls(normalized: str, *, where: str) -> None:
    """Raise on any function call outside the supported set (fail-loud)."""
    bad = sorted(
        {name for name in _CALL_TOKEN.findall(normalized) if name not in _EXPRTK_SUPPORTED_CALLS}
    )
    if bad:
        joined = ", ".join(repr(b) for b in bad)
        hint = ""
        if "tfun" in bad:
            hint = " — table (interpolation) functions have no SBML representation; "
        raise ConversionError(
            f"{where}: expression uses function call(s) {joined} that net2sbml "
            f"cannot translate to MathML.{hint}"
        )


def _walk_set_time(node: Any) -> None:
    """Convert every sentinel-named AST_NAME to the SBML time csymbol."""
    import libsbml

    if node is None:
        return
    if node.getType() == libsbml.AST_NAME and node.getName() == _TIME_SENTINEL:
        node.setType(libsbml.AST_NAME_TIME)
        node.setName("time")
    for i in range(node.getNumChildren()):
        _walk_set_time(node.getChild(i))


def _walk_fix_pi(node: Any) -> None:
    """Reconcile π so a pi-spelled model symbol survives the round-trip.

    ``parseL3Formula`` reads the bareword ``pi`` (case-insensitively) as the π
    constant, so a model symbol named ``pi``/``pI``/``PI``/``Pi`` would serialize
    to ``<pi/>`` — a silently wrong RHS. After normalization genuine π only
    arrives as :data:`_PI_SENTINEL` (from the ExprTk ``_pi`` constant), so:

    * a sentinel ``AST_NAME`` becomes the π constant, and
    * any *other* π-constant node is a misparsed symbol — revert it to an
      ``AST_NAME`` keeping its original spelling (``pI`` → ``<ci>pI</ci>``),
      which the usual SId remapping then resolves to the right parameter.
    """
    import libsbml

    if node is None:
        return
    t = node.getType()
    if t == libsbml.AST_NAME and node.getName() == _PI_SENTINEL:
        node.setType(libsbml.AST_CONSTANT_PI)
        node.setName("pi")  # canonical spelling; serializes as <pi/>
    elif t == libsbml.AST_CONSTANT_PI:
        # Misparsed model symbol (genuine π came through the sentinel above).
        node.setType(libsbml.AST_NAME)
    for i in range(node.getNumChildren()):
        _walk_fix_pi(node.getChild(i))


def _exprtk_to_ast(expr: str, *, where: str, scalar_names: frozenset[str] = frozenset()) -> Any:
    """Translate an engine ExprTk expression to a libsbml AST node.

    ``scalar_names`` are model symbols (parameters/observables/functions) that
    may appear as zero-arg calls ``sym()`` and must collapse to the bareword.

    Raises :class:`ConversionError` on an unsupported construct or a parse
    failure (fail-loud — never a silently-wrong RHS).
    """
    import libsbml

    if not expr or not expr.strip():
        raise ConversionError(f"{where}: empty expression")
    normalized = _normalize_exprtk(expr, scalar_names)
    _assert_supported_calls(normalized, where=where)
    ast = libsbml.parseL3Formula(normalized)
    if ast is None:
        err = libsbml.getLastParseL3Error()
        raise ConversionError(
            f"{where}: could not parse expression {expr!r} (normalized {normalized!r}): {err}"
        )
    _walk_set_time(ast)
    _walk_fix_pi(ast)
    return ast


# ─── Capability analysis ───────────────────────────────────────────────────


# ExprTk operator/keyword names that are *operations*, not value symbols — they
# never make an expression time-varying on their own (their arguments might).
# ``time`` is excluded: the time csymbol DOES make an expression time-varying.
_EXPR_OPERATORS = (_EXPRTK_SUPPORTED_CALLS | {"if"}) - {"time"}
_IDENT_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _symbol_time_invariant(
    name: str,
    params: set[str],
    funcs: dict[str, str],
    obs: set[str],
    species: set[str],
    cache: dict[str, bool],
) -> bool:
    """Whether ``name`` evaluates to a time-invariant constant.

    A parameter is constant; a species, observable, or the ``time`` csymbol
    varies; a function is constant iff every symbol it references is. Unknown
    symbols are treated as varying (conservative — never call a varying volume
    constant). Functions are memoized, with an in-progress ``False`` to break any
    reference cycle on the safe side.

    Functions are resolved *before* parameters: the ``.net`` builder gives every
    function a constant *shadow parameter* of the same name, so a time-varying
    function (e.g. an assignment-rule compartment ``cell = 1 + 0.1·time``) also
    appears as a constant parameter — the function is the live definition.
    """
    if name in cache:
        return cache[name]
    if name in species or name in obs or name == "time":
        cache[name] = False
        return False
    if name in funcs:
        cache[name] = False  # tentative — breaks cycles as "varying"
        result = all(
            _symbol_time_invariant(t, params, funcs, obs, species, cache)
            for t in set(_IDENT_TOKEN.findall(funcs[name]))
            if t not in _EXPR_OPERATORS
        )
        cache[name] = result
        return result
    if name in params or name in ("_pi", "pi"):
        cache[name] = True
        return True
    cache[name] = False  # unknown symbol — conservatively varying
    return False


def _time_varying_ar_volumes(model: Model, data: dict[str, Any]) -> list[str]:
    """Assignment-rule compartment volumes (``_varvol_ar_conc_map``) that vary in
    time — i.e. transitively reference a species, observable, or the time csymbol.

    Empty when every assignment-rule compartment volume is a constant expression
    (the common PBPK ``bodyweight · fraction`` case), which round-trips as a plain
    static compartment.
    """
    vac = getattr(model, "_varvol_ar_conc_map", None) or {}
    if not vac:
        return []
    params = {p["name"] for p in data["parameters"]}
    funcs = {f["name"]: f["expression"] for f in data["functions"]}
    obs = {o["name"] for o in data["observables"]}
    species = {s["name"] for s in data["species"]}
    cache: dict[str, bool] = {}
    comps = {comp for comp, _ in vac.values()}
    return sorted(
        c for c in comps if not _symbol_time_invariant(c, params, funcs, obs, species, cache)
    )


def sbml_capability_report(model: Model) -> dict[str, list[str]]:
    """Inspect a model for constructs the SBML network channel cannot carry.

    Returns ``{"dropped": [...], "lossy": [...]}`` of plain-English notes.
    ``dropped`` is non-fatal (events — they belong to a SED-ML protocol
    sidecar); ``lossy`` is fatal under ``strict=True``: a compartment volume that
    moves in time (rate-rule-driven, event-resized, or a *time-varying*
    assignment-rule expression), ``tfun`` table-function calls, and the rare
    cross-compartment uniform-propensity reaction. *Static* non-unit volumes are
    **not** lossy — including constant assignment-rule compartment volumes (the
    PBPK ``bodyweight · fraction`` case): the writer scales each kinetic law by
    its reaction's compartment volume so both the dynamics and the static
    amount/concentration reporting round-trip exactly. Pure; no I/O, no warnings.
    """
    core = model._core
    data = core.codegen_data()
    species = data["species"]
    functions = data["functions"]
    reactions = data["reactions"]

    dropped: list[str] = []
    lossy: list[str] = []

    def _eg(names: list[str]) -> str:
        head = ", ".join(repr(n) for n in names[:3])
        return head + (", …" if len(names) > 3 else "")

    n_events = int(getattr(core, "n_events", 0) or 0)
    if n_events:
        dropped.append(
            f"{n_events} event(s) — events define the simulation protocol, not "
            "the network channel; emit them as a SED-ML sidecar instead"
        )

    # Time-varying / rule-driven compartment volumes. Static non-unit volumes
    # round-trip exactly (kinetics *and* the static amount/concentration
    # reporting), but a volume that moves in time — driven by a rate rule or an
    # event resize — cannot be carried by the static compartment a single SBML
    # document declares. The loader records every species whose reported
    # concentration/amount it rescales against such a live volume in a Python-side
    # map on the Model, so a non-empty map is the exact signal that the reported
    # columns would drift (the stored ODE trajectory stays correct — only the
    # volume-relative reporting differs). Refuse rather than emit a document whose
    # reported output silently diverges.
    #
    # The assignment-rule *species* report redirect (``_ar_report_map``, GH #205)
    # is deliberately not gated here: it is a reporting concern orthogonal to
    # volume, and unit-volume models that carry it already convert — net2sbml
    # treats unit and non-unit such models identically.
    varvol_maps = (
        (
            "_varvol_conc_map",
            "track a time-varying (rate-rule/event-driven) "
            "compartment volume in their reported concentration",
        ),
        ("_varvol_amount_map", "track a time-varying compartment volume in their reported amount"),
        ("_varvol_event_resize_map", "sit in an event-resized compartment"),
    )
    for attr, why in varvol_maps:
        names = sorted((getattr(model, attr, None) or {}).keys())
        if names:
            lossy.append(
                f"{len(names)} species {why} (e.g. {_eg(names)}) — the kinetics "
                "convert faithfully but a static SBML document cannot carry a "
                "time-varying compartment volume"
            )

    # Assignment-rule compartments (``_varvol_ar_conc_map``, GH #87) are gated
    # only when their volume genuinely *varies in time*. The common case — a
    # PBPK organ volume defined as ``bodyweight · fraction`` from constant
    # parameters — is time-invariant: the report-time rescale is a no-op
    # (× V_static/V_static) and the species round-trips as a plain static
    # compartment (verified: BIOMD 1027/1028/1029/1039/856 are RHS- and
    # initial-state-exact). Only an AR volume that transitively references a
    # species, an observable, or the time csymbol is lossy.
    tv_ar = _time_varying_ar_volumes(model, data)
    if tv_ar:
        vac = getattr(model, "_varvol_ar_conc_map", None) or {}
        tv_set = set(tv_ar)
        affected = sorted(sp for sp, (comp, _) in vac.items() if comp in tv_set)
        lossy.append(
            f"{len(affected)} species sit in an assignment-rule compartment whose "
            f"volume varies in time (e.g. {_eg(affected)}; volume symbol(s) "
            f"{_eg(tv_ar)}) — the kinetics convert faithfully but a static SBML "
            "document cannot carry a time-varying compartment volume"
        )
    # Belt-and-suspenders: a live ODE volume on the dynamics side (should be
    # subsumed by ``_varvol_conc_map``/``_varvol_amount_map`` above).
    live_vol = [s["name"] for s in species if int(s.get("ode_live_volume_idx0", -1)) >= 0]
    if live_vol and not any("time-varying" in note for note in lossy):
        lossy.append(
            f"{len(live_vol)} species use a live (time-varying) compartment volume "
            f"(e.g. {_eg(live_vol)}) — net2sbml emits static compartments only"
        )

    # Static non-unit compartment volumes are faithful (#216 follow-up): the
    # writer scales each kinetic law by the reaction's compartment volume so the
    # SBML d[conc]/dt = kineticLaw/V semantics reconstruct the engine's stored
    # concentration rate (see :func:`_reaction_volume_factor`). The one shape no
    # single kinetic law can express is a *uniform*-propensity reaction
    # (``per_species_volume_scaling=False``) whose dynamic species span more than
    # one compartment volume — there SBML would divide each species's term by its
    # own V while the engine applied one rate to all. In practice empty (the
    # loader routes genuine cross-compartment reactions through
    # per_species_volume_scaling, which is emitted faithfully), but guarded so a
    # future loader shape can never silently mis-scale.
    cross_vol = [
        n + 1
        for n, r in enumerate(reactions)
        if not bool(r.get("per_species_volume_scaling", False))
        and len(_dynamic_volumes(r, species)) > 1
    ]
    if cross_vol:
        lossy.append(
            f"{len(cross_vol)} reaction(s) (e.g. #{cross_vol[0]}) couple species "
            "across compartments of different volume with a single "
            "concentration-rate propensity — no single SBML kinetic law "
            "reproduces the per-species volume scaling"
        )

    tfun_fns = [f["name"] for f in functions if "tfun" in f.get("expression", "")]
    if tfun_fns:
        lossy.append(
            f"{len(tfun_fns)} function(s) call a table (interpolation) function "
            f"(e.g. {_eg(tfun_fns)}) — tfun has no SBML/MathML representation"
        )

    return {"dropped": dropped, "lossy": lossy}


def _apply_capability_policy(
    report: dict[str, list[str]], *, strict: bool, model_name: str
) -> None:
    """Warn on dropped constructs; raise (strict) or warn (lenient) on lossy ones."""
    for note in report["dropped"]:
        warnings.warn(f"{model_name}: dropping {note}", ConversionWarning, stacklevel=3)
    if report["lossy"]:
        joined = "; ".join(report["lossy"])
        if strict:
            raise ConversionError(
                f"{model_name}: cannot convert faithfully to SBML — {joined}. "
                "Pass strict=False (--allow-lossy) to emit a best-effort document anyway."
            )
        for note in report["lossy"]:
            warnings.warn(
                f"{model_name}: lossy conversion — {note}", ConversionWarning, stacklevel=3
            )


# ─── Writer ────────────────────────────────────────────────────────────────


def _check(rc: int, what: str) -> None:
    """Raise if a libsbml setter returned a non-OK status code."""
    import libsbml

    if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ConversionError(f"libsbml rejected {what} (status {rc})")


def write_sbml(
    model: Model,
    out_path: str | Path | None = None,
    *,
    strict: bool = True,
    model_id: str | None = None,
) -> str:
    """Serialize ``model`` to SBML (Level 3 Version 2) text.

    Parameters
    ----------
    model : bngsim.Model
        A loaded model (e.g. from ``Model.from_net`` or ``Model.from_sbml``).
    out_path : str | Path | None
        If given, the SBML text is also written here.
    strict : bool
        When True (default), raise :class:`bngsim.ConversionError` on constructs
        SBML cannot represent faithfully. When False, downgrade to a
        :class:`bngsim.ConversionWarning` and emit a best-effort document.
    model_id : str | None
        SBML model id (sanitized to a valid SId). Defaults to ``model.name`` or
        ``"bngsim_model"``.

    Returns
    -------
    str
        The SBML document text.
    """
    import libsbml

    core = model._core
    data = core.codegen_data()
    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    functions = _dedup_functions(data["functions"])
    reactions = data["reactions"]

    raw_name = model_id or getattr(model, "name", None) or "bngsim_model"
    _apply_capability_policy(
        sbml_capability_report(model), strict=strict, model_name=str(raw_name)
    )

    init_state = list(model.get_state())
    func_names = {f["name"] for f in functions}
    obs_names = {o["name"] for o in observables}
    # Symbols that may appear as a bareword OR a zero-arg call inside a function
    # body (BNG issue #28): all parameters, observables and functions.
    scalar_names = frozenset({p["name"] for p in params} | obs_names | func_names)

    # ── document / model skeleton ─────────────────────────────────────────
    doc = libsbml.SBMLDocument(_SBML_LEVEL, _SBML_VERSION)
    m = doc.createModel()
    used_sids: set[str] = set()
    # Display names are keyed by bngsim's own from_sbml, so they must stay unique
    # even when a best-effort .net carries duplicate-named synthesized symbols.
    used_names: set[str] = set()
    _check(m.setId(_sanitize_sid(str(raw_name), used_sids, fallback="model")), "model id")

    # ── id allocation (symbols first, then species) ────────────────────────
    # Allocate SIds for the engine's symbol namespace — parameters (minus the
    # function/observable *shadows* the .net builder adds), observables, and
    # functions — so each keeps its natural id. Only THEN allocate species ids,
    # which sanitize away from pattern characters and dodge every reserved
    # symbol name. This separation is what lets a species ``S1()`` and an
    # observable ``S1`` coexist (the species becomes e.g. ``S1_2``) and keeps
    # an observable's own rule (``obs = species``) from self-referencing.
    shadow = func_names | obs_names
    sym_sid: dict[str, str] = {}  # engine symbol name → SId (params/obs/funcs)
    param_emit: list[tuple[dict[str, Any], str]] = []
    for p in params:
        nm = p["name"]
        if nm in shadow:
            continue  # shadow constant; the function/observable emission covers it
        sid = _sanitize_sid(nm, used_sids, fallback="p")
        sym_sid[nm] = sid
        param_emit.append((p, sid))
    for o in observables:
        sym_sid[o["name"]] = _sanitize_sid(o["name"], used_sids, fallback="obs")
    for f in functions:
        sym_sid[f["name"]] = _sanitize_sid(f["name"], used_sids, fallback="func")
    species_sid: list[str] = [
        _sanitize_sid(s["name"], used_sids, fallback=f"s{i + 1}") for i, s in enumerate(species)
    ]

    # ── compartments (reconstructed from per-species volume_factor) ────────
    # Group species by their compartment volume V_c (=volume_factor). The .net
    # source is uniformly V=1, but an SBML-sourced Model carries real volumes;
    # emitting one compartment per distinct volume keeps that faithful.
    vol_to_comp: dict[float, str] = {}

    def _comp_for(vol: float) -> str:
        key = float(vol)
        if key not in vol_to_comp:
            cid = _sanitize_sid(
                f"compartment_{len(vol_to_comp) + 1}", used_sids, fallback="compartment"
            )
            c = m.createCompartment()
            _check(c.setId(cid), "compartment id")
            _check(c.setConstant(True), "compartment constant")
            _check(c.setSize(key), "compartment size")
            _check(c.setSpatialDimensions(3.0), "compartment dims")
            vol_to_comp[key] = cid
        return vol_to_comp[key]

    # ── species ────────────────────────────────────────────────────────────
    # Engine stores `amount / V_c` uniformly (loader §3). So:
    #   * concentration species → initialConcentration = stored value
    #   * amount species (hasOnlySubstanceUnits) → initialAmount = stored * V_c
    for i, s in enumerate(species):
        vol = float(s.get("volume_factor", 1.0))
        comp = _comp_for(vol)
        sid = species_sid[i]
        sp = m.createSpecies()
        _check(sp.setId(sid), "species id")
        if sp.getId() != s["name"]:
            sp.setName(s["name"])  # preserve the original BNGL pattern
        _check(sp.setCompartment(comp), "species compartment")
        stored = float(init_state[i]) if i < len(init_state) else 0.0
        amount_valued = bool(s.get("amount_valued", False))
        _check(sp.setHasOnlySubstanceUnits(amount_valued), "species hOSU")
        if amount_valued:
            _check(sp.setInitialAmount(stored * vol), "species initialAmount")
        else:
            _check(sp.setInitialConcentration(stored), "species initialConcentration")
        # A `fixed` ($) species is held constant; mark it a boundary species.
        fixed = bool(s.get("fixed", False))
        _check(sp.setBoundaryCondition(fixed), "species boundaryCondition")
        _check(sp.setConstant(fixed), "species constant")

    # ── parameters ─────────────────────────────────────────────────────────
    for p, sid in param_emit:
        pp = m.createParameter()
        _check(pp.setId(sid), "parameter id")
        disp = _unique_name(p["name"], used_names)
        if pp.getId() != disp:
            pp.setName(disp)
        _check(pp.setConstant(True), "parameter constant")
        _check(pp.setValue(float(p["value"])), "parameter value")

    # ── observables → parameter + linear assignmentRule over species ───────
    for o in observables:
        sid = sym_sid[o["name"]]
        pp = m.createParameter()
        _check(pp.setId(sid), "observable parameter id")
        disp = _unique_name(o["name"], used_names)
        if pp.getId() != disp:
            pp.setName(disp)
        _check(pp.setConstant(False), "observable parameter constant")
        terms = []
        for idx0, factor in o["entries"]:
            sp_id = species_sid[int(idx0)]
            fac = float(factor)
            terms.append(sp_id if fac == 1.0 else f"{fac!r} * {sp_id}")
        formula = " + ".join(terms) if terms else "0"
        ast = libsbml.parseL3Formula(formula)
        if ast is None:
            raise ConversionError(
                f"observable {o['name']!r}: could not build sum formula {formula!r}"
            )
        rule = m.createAssignmentRule()
        _check(rule.setVariable(sid), "observable rule variable")
        _check(rule.setMath(ast), "observable rule math")

    # ── functions → parameter + assignmentRule (translated MathML) ─────────
    for f in functions:
        sid = sym_sid[f["name"]]
        pp = m.createParameter()
        _check(pp.setId(sid), "function parameter id")
        disp = _unique_name(f["name"], used_names)
        if pp.getId() != disp:
            pp.setName(disp)
        _check(pp.setConstant(False), "function parameter constant")
        ast = _exprtk_to_ast(
            f["expression"],
            where=f"function {f['name']!r}",
            scalar_names=scalar_names,
        )
        rule = m.createAssignmentRule()
        _check(rule.setVariable(sid), "function rule variable")
        _check(rule.setMath(ast), "function rule math")

    # Function/observable bodies reference other symbols by their *sanitized*
    # SId. Rebuild each rule's math against sym_sid so a sanitized reference
    # (e.g. an observable whose name held pattern characters) resolves. Done
    # after all ids are known so forward references work.
    _remap_rule_symbols(m, sym_sid)

    # ── reactions ──────────────────────────────────────────────────────────
    for n, r in enumerate(reactions):
        rid = _sanitize_sid(f"r{n + 1}", used_sids, fallback="r")
        rxn = m.createReaction()
        _check(rxn.setId(rid), "reaction id")
        _check(rxn.setReversible(False), "reaction reversible")
        # reactant/product species refs, with stoichiometry from index repeats
        for idx0, count in _multiset(r["reactants"]).items():
            ref = rxn.createReactant()
            _check(ref.setSpecies(species_sid[idx0]), "reactant species")
            _check(ref.setStoichiometry(float(count)), "reactant stoich")
            _check(ref.setConstant(True), "reactant constant")
        for idx0, count in _multiset(r["products"]).items():
            ref = rxn.createProduct()
            _check(ref.setSpecies(species_sid[idx0]), "product species")
            _check(ref.setStoichiometry(float(count)), "product stoich")
            _check(ref.setConstant(True), "product constant")
        formula = _kinetic_formula(r, species_sid, sym_sid, params, species, idx=n)
        ast = libsbml.parseL3Formula(formula)
        if ast is None:
            raise ConversionError(
                f"reaction #{n + 1}: could not build kinetic-law formula {formula!r}: "
                f"{libsbml.getLastParseL3Error()}"
            )
        # A rate symbol spelled pi/pI/PI/Pi would have parsed to the π constant;
        # revert it to a name reference (kinetic laws carry no genuine π sentinel).
        _walk_fix_pi(ast)
        kl = rxn.createKineticLaw()
        _check(kl.setMath(ast), "kinetic-law math")

    # ── serialize ──────────────────────────────────────────────────────────
    text = libsbml.writeSBMLToString(doc)
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


def _remap_rule_symbols(m: Any, sym_sid: dict[str, str]) -> None:
    """Rewrite AST_NAME references in every assignment rule to sanitized SIds.

    Function/observable bodies reference other symbols by their *engine* name;
    when sanitization rewrote that name (rare — only pattern-charactered
    symbols), the parsed reference would dangle. Walk each rule's math and
    rename any AST_NAME whose engine name maps to a different SId. A no-op for
    the common case where every symbol name is already a valid SId.
    """
    import libsbml

    rewrite = {k: v for k, v in sym_sid.items() if k != v}
    if not rewrite:
        return

    def _walk(node: Any) -> None:
        if node is None:
            return
        if node.getType() == libsbml.AST_NAME:
            nm = node.getName()
            if nm in rewrite:
                node.setName(rewrite[nm])
        for i in range(node.getNumChildren()):
            _walk(node.getChild(i))

    for i in range(m.getNumRules()):
        _walk(m.getRule(i).getMath())


def _multiset(indices: list[int]) -> dict[int, int]:
    """Count repeated 0-based species indices (stoichiometry > 1)."""
    out: dict[int, int] = {}
    for i in indices:
        out[int(i)] = out.get(int(i), 0) + 1
    return out


def _fmt(x: float) -> str:
    return repr(float(x))


def _dynamic_volumes(rxn: dict[str, Any], species: list[dict[str, Any]]) -> set[float]:
    """Distinct compartment volumes of a reaction's *dynamic* (non-fixed) species.

    Fixed (``$``/boundary) species carry no ODE derivative, so their volume does
    not constrain the kinetic-law volume scaling (and they appear in the law as
    constants, already folded into the propensity). Reactant and product indices
    are combined — a reaction is single-compartment iff this set has ≤1 element.
    """
    return {
        float(species[i].get("volume_factor", 1.0))
        for i in list(rxn["reactants"]) + list(rxn["products"])
        if not bool(species[i].get("fixed", False))
    }


def _reaction_volume_factor(
    rxn: dict[str, Any], species: list[dict[str, Any]], *, idx: int
) -> float:
    """Volume scale converting the engine propensity to an SBML extent rate.

    The engine stores every species as a concentration (``amount / V_c``) and
    computes one per-reaction propensity ``P``. SBML instead defines a reaction
    *extent* rate ``L`` and integrates ``d[S_i]/dt = stoich_i · L / V(S_i)`` for
    concentration species (and ``d(amount_i)/dt = stoich_i · L`` for amount
    species — same ``/V_c`` once mapped back to the engine's stored
    concentration). Two engine conventions therefore map differently:

    * ``per_species_volume_scaling=True`` — the engine already divides each
      species's accumulation by *its own* ``V_c``, so ``P`` plays the role of
      ``L`` directly. This is exactly the SBML per-species rule and carries
      genuinely cross-compartment reactions faithfully. Factor = ``1``.

    * ``per_species_volume_scaling=False`` (the default) — ``P`` is a
      concentration rate applied uniformly (``d[S_i]/dt = ±P``). SBML would give
      ``±L / V(S_i)``, so we emit ``L = P · V_c``. This is exact iff every
      dynamic species shares one compartment volume ``V_c`` (the elementary /
      single-compartment functional case); a unit ``V_c`` folds out.

    Raises :class:`ConversionError` for the unrepresentable shape: a uniform
    (``per_species_volume_scaling=False``) reaction whose dynamic species span
    more than one volume — no single kinetic law reproduces the per-species
    division. ``sbml_capability_report`` flags this up front under ``strict``.
    """
    if bool(rxn.get("per_species_volume_scaling", False)):
        return 1.0
    vols = _dynamic_volumes(rxn, species)
    if len(vols) > 1:
        raise ConversionError(
            f"reaction #{idx + 1}: couples dynamic species across compartments "
            f"of different volume {sorted(vols)} with a single concentration-rate "
            "propensity; no single SBML kinetic law reproduces the per-species "
            "volume scaling"
        )
    return next(iter(vols)) if vols else 1.0


def _kinetic_formula(
    rxn: dict[str, Any],
    species_sid: list[str],
    sym_sid: dict[str, str],
    params: list[dict[str, Any]],
    species: list[dict[str, Any]],
    *,
    idx: int,
) -> str:
    """Build the L3-infix kinetic-law formula reproducing the engine propensity.

    The engine works in concentration space and computes a per-reaction
    propensity ``P``:

    * elementary  → ``stat_factor * k * Π(reactant species)``
      (the engine always applies the reactant mass-action factor for
      elementary reactions)
    * functional  → ``stat_factor * f`` × ``Π(reactant species)`` only when
      ``apply_species_factor`` is set
    * mm          → the tQSSA closed form (see :func:`_mm_formula`); the rate
      *is* the full propensity, so the reactant factor is not applied

    The emitted kinetic law is the SBML *extent* rate ``P · V_c``, where ``V_c``
    is :func:`_reaction_volume_factor` (``1`` for unit-volume or already
    per-species-scaled reactions, so this is a no-op for ``.net``-sourced and
    most SBML-sourced models).
    """
    typ = rxn["type"]
    if typ == "mm":
        base = _mm_formula(rxn, species_sid, sym_sid, params, idx=idx)
    else:
        fname = rxn["function_name"]
        sf = float(rxn["stat_factor"])
        asf = bool(rxn["apply_species_factor"])
        reactants = list(rxn["reactants"])
        rate_sym = sym_sid.get(fname, fname)

        factors: list[str] = []
        if sf != 1.0:
            factors.append(_fmt(sf))
        factors.append(rate_sym)
        if typ == "elementary" or asf:
            factors.extend(species_sid[i] for i in reactants)
        base = " * ".join(f"({t})" for t in factors) if factors else "0"

    vc = _reaction_volume_factor(rxn, species, idx=idx)
    if vc != 1.0:
        return f"({_fmt(vc)}) * ({base})"
    return base


def _mm_formula(
    rxn: dict[str, Any],
    species_sid: list[str],
    sym_sid: dict[str, str],
    params: list[dict[str, Any]],
    *,
    idx: int,
) -> str:
    """Closed-form Michaelis–Menten (tQSSA) kinetic law, matching ``model.cpp``.

    With E = first reactant, S = second reactant, and (kcat, Km) the two rate
    parameters::

        sFree = 0.5 * ((S - Km - E) + sqrt((S - Km - E)^2 + 4*Km*S))
        rate  = kcat * stat_factor * max(sFree, 0) * E / (Km + max(sFree, 0))

    (The engine floors a negative ``sFree`` to 0; ``max`` reproduces that. SBML
    expresses the whole law explicitly — net2sbml is the faithful half.)
    """
    reactants = list(rxn["reactants"])
    rate_idx = list(rxn.get("rate_param_indices", []))
    if len(reactants) < 2 or len(rate_idx) < 2:
        raise ConversionError(
            f"reaction #{idx + 1}: Michaelis–Menten rate law needs 2 reactants "
            f"and 2 rate parameters (got {len(reactants)} / {len(rate_idx)})"
        )
    E = species_sid[reactants[0]]
    S = species_sid[reactants[1]]
    kcat_name = params[rate_idx[0]]["name"]
    km_name = params[rate_idx[1]]["name"]
    kcat = sym_sid.get(kcat_name, kcat_name)
    Km = sym_sid.get(km_name, km_name)
    sf = float(rxn["stat_factor"])

    delta = f"(({S}) - ({Km}) - ({E}))"
    s_free = f"(0.5 * ({delta} + sqrt({delta}^2 + 4 * ({Km}) * ({S}))))"
    s_free = f"max({s_free}, 0)"
    pre = f"({_fmt(sf)} * " if sf != 1.0 else "("
    return f"{pre}({kcat}) * ({s_free}) * ({E}) / (({Km}) + ({s_free})))"
