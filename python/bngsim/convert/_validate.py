"""bngsim.convert._validate — the conversion-validation framework (GH #217).

A single entry point, :func:`validate_conversion`, grades a format conversion
(SBML⇄``.net``) at five escalating levels of rigor and returns a structured
:class:`ConversionValidationReport` artifact. The acceptance bar (GH #211c) is
**L0–L3 as hard gates, plus best-effort L4**:

* **L0 — syntactic validity** (hard gate). The converted output passes the
  *target* format's own validator: libsbml's consistency checks for SBML; the
  ``.net`` reader accepting the file for ``.net``.
* **L1 — structural equivalence** (hard gate). Source and conversion carry the
  same #species, #reactions and per-reaction reactant/product topology.
* **L2 — round-trip identity** (hard gate). ``X → Y → X`` reproduces the source
  model graph (counts, topology and ODE right-hand side) after a canonical,
  loader-level normalization that is robust to benign reordering/relabelling.
* **L3 — numerical/semantic equivalence** (hard gate). Source and conversion
  are simulated on a shared time grid and compared trajectory-by-trajectory
  under a **scale-aware** per-cell tolerance (the #214 verdict, vendored here so
  the shipped converter carries no parity-suite dependency).
* **L4 — symbolic/algebraic equivalence** (best-effort, **non-gating**). The
  per-species ODE right-hand side is reconstructed symbolically from each
  representation and compared with ``sympy.simplify``. Reports
  **equal / not-equal / inconclusive**; it never blocks a conversion and is
  allowed to punt on Michaelis–Menten/volume-scaled/transcendental kinetics it
  cannot reconstruct faithfully.

Both conversion directions are gated: pass an ``.xml``/``.sbml`` source to grade
``sbml2net``, or a ``.net`` source to grade ``net2sbml``.
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bngsim.convert._net_writer import FLUX_HELPER_PREFIX, SYNTHETIC_PREFIX

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from bngsim._model import Model


# ─── Shared structural / RHS primitives ────────────────────────────────────
# These low-level helpers live here (not in convert.__init__) so the public
# convert surface and this framework share one definition without an import
# cycle: __init__ imports them from this module.


_FLUX_FRAGMENT_RE = re.compile(rf"^{re.escape(FLUX_HELPER_PREFIX)}(\d+)_\d+$")


def _net_stoich_sig(net: dict[int, int]) -> dict[str, list[int]]:
    """(reactants, products) multiset from a per-species net-stoichiometry map,
    dropping catalysts (net 0)."""
    reactants: list[int] = []
    products: list[int] = []
    for sp, s in net.items():
        if s < 0:
            reactants.extend([sp] * -s)
        elif s > 0:
            products.extend([sp] * s)
    return {"reactants": reactants, "products": products}


def _folded_reactions(data: dict[str, Any]) -> list[dict[str, list[int]]]:
    """Per-reaction **net** stoichiometry, with signed-flux fragments folded back.

    The ``.net`` writer re-encodes an ``asf=False`` functional reaction as one
    zero-reactant ``0 -> Molᵢ`` fragment per *net-affected* species, rate function
    ``__flux{n}_{k}() = sᵢ * (P)`` (see ``_net_writer._expand_functional_reactions``).
    That re-encoding is RHS-identical but (a) changes the reaction count/topology and
    (b) drops catalysts — a species appearing on both sides nets to zero and gets no
    fragment. So a structural comparison must reconstruct source reaction *n* from
    its fragments (negative factor ⇒ reactant, positive ⇒ product) **and** reduce
    every reaction to its net stoichiometry, so an explicit catalyst ``A -> A + B``
    and its folded flux form ``-> B`` compare equal. Applied symmetrically to source
    and ``.net``; a model with no fragments is reduced to net stoichiometry only.
    """
    funcs = {f["name"]: f.get("expression", "") for f in data.get("functions", [])}
    out: list[dict[str, list[int]]] = []
    groups: dict[str, dict[int, int]] = {}
    for r in data["reactions"]:
        m = _FLUX_FRAGMENT_RE.match(str(r.get("function_name", "")))
        factor = None
        if m and len(r["products"]) == 1 and not r["reactants"]:
            try:
                factor = int(round(float(funcs[r["function_name"]].split("*", 1)[0].strip())))
            except (KeyError, ValueError, IndexError):
                factor = None
        if m is not None and factor is not None:
            # fragment: species = the sole product; accumulate into its source group
            grp = groups.setdefault(m.group(1), {})
            sp = int(r["products"][0])
            grp[sp] = grp.get(sp, 0) + factor
        else:
            net: dict[int, int] = {}
            for i in r["reactants"]:
                net[i] = net.get(i, 0) - 1
            for i in r["products"]:
                net[i] = net.get(i, 0) + 1
            out.append(_net_stoich_sig(net))
    for net in groups.values():
        out.append(_net_stoich_sig(net))
    return out


def _stoich_signatures(model: Model) -> Counter:
    """Multiset of per-reaction (sorted reactant idxs, sorted product idxs).

    Signed-flux fragments are folded back to their source reaction first, so the
    signature reflects the source topology rather than the ``.net`` re-encoding.
    """
    sigs: Counter = Counter()
    for r in _folded_reactions(model._core.codegen_data()):
        sigs[(tuple(sorted(r["reactants"])), tuple(sorted(r["products"])))] += 1
    return sigs


def _dynamic_stoich_signatures(model: Model) -> Counter:
    """Per-reaction (sorted reactant idxs, sorted product idxs) over *non-fixed*
    species — the topology that must be invariant across loaders even when a
    constant boundary species is folded into the rate law. Signed-flux fragments
    are folded back to their source reaction first (see :func:`_folded_reactions`)."""
    data = model._core.codegen_data()
    fixed = {i for i, s in enumerate(data["species"]) if s.get("fixed", False)}
    sigs: Counter = Counter()
    for r in _folded_reactions(data):
        rr = tuple(sorted(i for i in r["reactants"] if i not in fixed))
        pp = tuple(sorted(i for i in r["products"] if i not in fixed))
        sigs[(rr, pp)] += 1
    return sigs


def _effective_n_reactions(model: Model) -> int:
    """Reaction count with signed-flux fragments folded back to their source."""
    return len(_folded_reactions(model._core.codegen_data()))


def _max_rhs_delta(a_model: Model, b_model: Model) -> float:
    """Largest absolute difference between the two models' ODE RHS.

    Compared at the shared initial state and a nonlinear-exercising perturbation
    of it — a trajectory-free equivalence probe that stays valid even when one
    loader synthesized extra reporting observables (those never enter the RHS).
    Returns a scale-relative delta (``max|Δ| / max(1, max|rhs|)``).
    """
    import numpy as np

    y0 = np.asarray(a_model.get_state(), dtype=np.float64)
    worst = 0.0
    for t, y in ((0.0, y0), (1.0, y0 * 1.37 + 0.05)):
        a = np.asarray(a_model._core._eval_rhs(t, y.tolist()), dtype=np.float64)
        b = np.asarray(b_model._core._eval_rhs(t, y.tolist()), dtype=np.float64)
        if a.shape != b.shape:
            return float("inf")
        scale = max(float(np.abs(a).max(initial=0.0)), 1.0)
        worst = max(worst, float(np.max(np.abs(a - b)) / scale))
    return worst


def _ar_report_delta(model: Model) -> float:
    """How much an AssignmentRule-target species' *reported* value varies with state.

    An AR-target species is emitted ``fixed`` (its ODE derivative is zeroed) and
    ``Simulator.run`` overwrites its reported column with the rule's live value —
    a report-time transform (``_apply_ar_report_map``) that the plain ``.net``
    cannot carry, so a round-tripped network reports these species *frozen at their
    initial value*. That loss is invisible to :func:`_max_rhs_delta` (the frozen
    species has ``dy/dt = 0`` in both models). It is a faithfulness loss **iff the
    rule's value actually varies**: a constant rule (``EGF := 30``) freezes to the
    same value the source reports, so it round-trips faithfully.

    We measure that variation directly on the source model, at the same two states
    :func:`_max_rhs_delta` probes: the shared initial state and a nonlinear
    perturbation. The rule's live value is its bare-name observable (linear-on-
    species rules) or expression/function (everything else). Returns the largest
    scale-relative change of any AR-target's live value between the two states —
    ``0.0`` when the model has no AR-target species or all their rules are
    constant, ``inf`` when a rule cannot be evaluated (conservatively unfaithful).
    """
    import numpy as np

    amap = getattr(model, "_ar_report_map", None)
    if not amap:
        return 0.0
    core = model._core
    data = core.codegen_data()
    obs_entries = {o["name"]: o["entries"] for o in data["observables"]}
    y0 = np.asarray(model.get_state(), dtype=np.float64)
    states = [(0.0, y0), (1.0, y0 * 1.37 + 0.05)]
    fn_by_state: list[dict | None] = []
    for _t, y in states:
        try:
            fn_by_state.append(core._eval_functions(_t, y.tolist()))
        except Exception:  # noqa: BLE001 — out-of-domain probe leaves this state undecided
            fn_by_state.append(None)

    def _live_value(kind: str, src: str, y: np.ndarray, fns: dict | None) -> float | None:
        if kind == "observable" and src in obs_entries:
            return float(sum(float(f) * float(y[int(i)]) for i, f in obs_entries[src]))
        if kind == "expression" and fns is not None and src in fns:
            try:
                return float(fns[src])
            except (TypeError, ValueError):
                return None
        return None

    worst = 0.0
    for entry in amap.values():
        kind, src = entry[0], entry[1]
        v0 = _live_value(kind, src, states[0][1], fn_by_state[0])
        v1 = _live_value(kind, src, states[1][1], fn_by_state[1])
        if v0 is None or v1 is None:
            return float("inf")  # can't prove constant → treat as unfaithful
        scale = max(abs(v0), 1.0)
        worst = max(worst, abs(v1 - v0) / scale)
    return worst


# ─── Scale-aware numerical verdict (vendored from the #214 parity kernel) ───
# A self-contained port of parity_checks/_core/differ.deterministic_verdict,
# trimmed to the gate this framework needs (no ensemble path, no injected
# forgive-mask, no soft-fail budget). The shipped converter must not import the
# repo-only parity suite, so the scale-aware tolerance lives here.

_REL_TOL = 1e-4  # relative term (rtol·|y|)
_ABS_TOL_COL = 1e-6  # absolute term scaled by the per-column (per-variable) peak
_ABS_TOL_FILE = 1e-9  # absolute term scaled by the file-wide peak (sub-scale cols)
_NEAR_ZERO_FLOOR_REL = 1e-12  # denom floor for the reported relative diff
_HARD_REL_CEILING = 0.05  # a magnitude-carrying column past this is a real divergence
_SIGNIF_FLOOR = 1e-3  # column peak below this fraction of the file peak is sub-scale


def _scale_aware_verdict(a: np.ndarray, b: np.ndarray) -> dict:
    """Verdict for two aligned deterministic trajectories ``a`` vs ``b``.

    ``a``/``b`` are ``(n_time, n_var)`` arrays in a shared column order. The
    per-cell tolerance ``|a-b| <= ABS_FILE·scale + ABS_COL·col_peak + REL·|y|``
    is combined with a peak-relative dynamic-range gate: a failing cell in a
    column that is either below the model's dynamic range (peak < SIGNIF_FLOOR·
    file-peak) or never diverges past the relative ceiling at its own peak scale
    is forgiven, matching the #214 scale-aware verdict. A cell where exactly one
    side is non-finite is an unconditional hard fail.
    """
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return {
            "passed": False,
            "reason": f"shape {a.shape} vs {b.shape}",
            "max_rel": float("inf"),
            "max_abs": float("inf"),
            "n_fail": -1,
        }

    both_nan = np.isnan(a) & np.isnan(b)
    one_side_nonfinite = np.isfinite(a) != np.isfinite(b)
    absd = np.where(both_nan, 0.0, np.abs(a - b))
    absd_clean = np.where(np.isnan(absd), np.inf, absd)  # one-side NaN → flag

    finite_mag = np.concatenate([np.abs(a).ravel(), np.abs(b).ravel()])
    finite_mag = finite_mag[np.isfinite(finite_mag)]
    scale = float(finite_mag.max()) if finite_mag.size else 1.0
    zero_floor = max(1e-12, scale * _NEAR_ZERO_FLOOR_REL)

    colmag = np.where(
        np.isfinite(np.maximum(np.abs(a), np.abs(b))), np.maximum(np.abs(a), np.abs(b)), 0.0
    )
    col_peak = colmag.max(axis=0) if colmag.size else np.zeros(0)
    col_peak_denom = np.maximum(col_peak, zero_floor)

    cell_tol = _ABS_TOL_FILE * scale + _ABS_TOL_COL * col_peak[np.newaxis, :] + _REL_TOL * colmag
    fail_mask = absd_clean > cell_tol

    reld_peak = np.where(np.isnan(absd), np.inf, absd_clean / col_peak_denom[np.newaxis, :])
    col_reldiv_peak = reld_peak.max(axis=0) if reld_peak.size else np.zeros(0)
    col_significant = col_peak > _SIGNIF_FLOOR * scale
    col_is_real = (col_reldiv_peak > _HARD_REL_CEILING) & col_significant

    below_dyn_range = fail_mask & (~col_is_real)[np.newaxis, :] & ~one_side_nonfinite
    effective_fail = (fail_mask & ~below_dyn_range) | one_side_nonfinite
    n_fail = int(np.sum(effective_fail))

    max_rel = float(np.max(reld_peak[effective_fail])) if n_fail else 0.0
    max_abs = float(np.max(absd_clean[effective_fail])) if n_fail else 0.0
    return {
        "passed": n_fail == 0,
        "max_rel": max_rel,
        "max_abs": max_abs,
        "n_fail": n_fail,
        "n_cells": int(effective_fail.size),
        "scale": scale,
    }


# ─── Symbolic per-species RHS reconstruction (L4) ───────────────────────────


def _symbolic_rhs(model: Model):
    """Reconstruct the per-*dynamic*-species ODE right-hand side symbolically.

    Returns ``(rhs, n_species)`` where ``rhs`` is a dict mapping each *non-fixed*
    species index to its sympy RHS expression, or ``None`` to signal that a
    faithful reconstruction is out of reach (→ *inconclusive* at L4). The shared
    symbol space is species by **index** (``_s0, _s1, …``) and time as ``_t``;
    every constant parameter **and every fixed (``$``) species** is folded to its
    numeric value, so a benign parameter rename — or a constant boundary species
    that one loader keeps as a clamped species and the other folds into the rate
    law — does not read as an algebraic difference. Observables are inlined to
    their species sums; user functions are inlined and translated through the
    Jacobian's ExprTk→sympy bridge.

    Punts (``None``) on a reaction kind other than elementary/functional/MM, any
    non-unit/live/per-species compartment volume, an amount-valued species, a
    table function, an unresolved identifier, or a flux containing a non-smooth
    construct (``Max``/``Min``/``Piecewise``/``Abs``/``floor``/``ceiling``) that
    ``sympy.simplify`` cannot reliably prove equal — exactly the
    piecewise/transcendental/custom kinetics L4 is allowed to punt on.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    from bngsim._jacobian import _exprtk_to_sympy, _inline_functions

    data = model._core.codegen_data()
    species = data["species"]
    params = data["parameters"]
    observables = data["observables"]
    functions = data["functions"]
    reactions = data["reactions"]

    if data.get("table_functions"):
        return None
    for s in species:
        if s.get("volume_factor", 1.0) != 1.0 or s.get("amount_valued", False):
            return None
        if s.get("ode_live_volume_idx0", -1) != -1:
            return None

    n_sp = len(species)
    state = list(model.get_state())
    fixed = {i for i, s in enumerate(species) if s.get("fixed", False)}
    sp_sym = [sp.Symbol(f"_s{i}") for i in range(n_sp)]
    t_sym = sp.Symbol("_t")
    # A fixed species is a constant held at its initial value: fold it to a
    # number so a dynamic species' RHS reads identically whether the loader kept
    # the boundary species symbolic or folded it into the rate constant.
    fixed_subs = {sp_sym[i]: sp.Float(state[i]) for i in fixed}

    param_val = {p["name"]: p["value"] for p in params if p.get("is_const", True)}
    obs_expr = {}
    for o in observables:
        acc = sp.Integer(0)
        for idx0, factor in o["entries"]:
            acc += sp.Float(factor) * sp_sym[idx0]
        obs_expr[o["name"]] = acc
    func_map = {f["name"]: f["expression"] for f in functions}

    time_symbol_name = "_bngsim_time_csymbol"  # sympy csymbol for ExprTk time()
    _NONSMOOTH = (sp.Max, sp.Min, sp.Abs, sp.floor, sp.ceiling, sp.Piecewise)

    from sympy.core.function import AppliedUndef

    def _resolve(expr_str: str):
        inlined = _inline_functions(expr_str, func_map)
        if inlined is None:
            return None
        ex = _exprtk_to_sympy(inlined)
        if ex is None:
            return None
        subs = {}
        for s in ex.free_symbols:
            name = str(s)
            if name == time_symbol_name:
                subs[s] = t_sym
            elif name in obs_expr:
                subs[s] = obs_expr[name]
            elif name in param_val:
                subs[s] = sp.Float(param_val[name])
            else:
                return None
        # Zero-arg observable/parameter calls (BNG ``Atot()`` == bareword
        # ``Atot``) parse as undefined nullary functions, not symbols — resolve
        # them the same way so they don't survive as opaque atoms.
        for f in ex.atoms(AppliedUndef):
            name = f.func.__name__
            if name in obs_expr:
                subs[f] = obs_expr[name]
            elif name in param_val:
                subs[f] = sp.Float(param_val[name])
            else:
                return None
        return ex.xreplace(subs)

    # Fold *derived* (non-const) parameters that are nonetheless time-invariant —
    # e.g. ``m1 = 5/MEK`` with ``MEK`` constant — to their numeric value, by
    # fixpoint over the const params already known. A derived parameter that
    # stays symbolic (references time or an observable) is genuinely
    # time-varying and is left out, so a rate referencing it punts to
    # inconclusive rather than folding a wrong constant.
    pending = [p for p in params if not p.get("is_const", True)]
    progressed = True
    while pending and progressed:
        progressed = False
        still = []
        for p in pending:
            ex = _resolve(p["expression"])
            if ex is not None and ex.is_number:
                param_val[p["name"]] = float(ex)
                progressed = True
            else:
                still.append(p)
        pending = still

    rhs = {i: sp.Integer(0) for i in range(n_sp) if i not in fixed}
    for r in reactions:
        if r.get("per_species_volume_scaling", False):
            return None
        if r.get("ssa_live_volume_idx0", -1) != -1:
            return None
        kind = r["type"]
        sf = sp.Float(r.get("stat_factor", 1.0))
        reactants = list(r["reactants"])
        products = list(r["products"])
        apply_factor = r.get("apply_species_factor", True)

        if kind == "elementary":
            name = r["function_name"]
            if name not in param_val:
                return None
            flux = sf * sp.Float(param_val[name])
        elif kind == "functional":
            body = _resolve(r["function_name"])
            if body is None:
                return None
            flux = sf * body
        elif kind == "mm":
            if len(reactants) < 2:
                return None
            rate_params = r.get("rate_param_indices") or []
            if len(rate_params) < 2:
                return None
            kcat = sp.Float(params[rate_params[0]]["value"])
            km = sp.Float(params[rate_params[1]]["value"])
            E = sp_sym[reactants[0]]
            S = sp_sym[reactants[1]]
            M = sp.Rational(1, 2) * ((S - km - E) + sp.sqrt((S - km - E) ** 2 + 4 * km * S))
            flux = sf * kcat * M * E / (km + M)
        else:
            return None

        if apply_factor and kind != "mm":
            for ri in reactants:
                flux = flux * sp_sym[ri]

        flux = flux.xreplace(fixed_subs)
        if flux.has(*_NONSMOOTH):
            return None

        for ri in reactants:
            if ri not in fixed:
                rhs[ri] = rhs[ri] - flux
        for pi in products:
            if pi not in fixed:
                rhs[pi] = rhs[pi] + flux

    return rhs, n_sp


# A symbolic RHS difference whose coefficients are all this far below the RHS's
# own coefficient scale is floating-point round-off, not a real difference: the
# BNGL-float → MathML → back round-trip reassociates arithmetic and leaves
# ~1e-16-scale coefficient dust that sympy.simplify() will not crush to 0. The
# 1e-9 relative floor sits ~6 orders above that dust and far below any
# dynamically-relevant term, so it forgives the dust without masking a genuine
# (magnitude-carrying) difference. Cross-checked against L2 (RHS identity) on the
# whole bng_parity corpus: every forgiven case also passes the L2 numeric gate.
_L4_FP_RTOL = 1e-9


def _max_numeric_coeff(expr) -> float | None:
    """Largest ``|numeric coefficient|`` over the expanded terms of ``expr``.

    Returns ``0.0`` when ``expr`` expands to 0, or ``None`` when any term carries
    a non-numeric (still-symbolic) coefficient. A cheap *first* screen only: a
    small max-coefficient guarantees the residual is dust, but a large one does
    NOT prove a difference — terms can carry huge coefficients that cancel
    numerically (catastrophic cancellation), which is why an unforgiven residual
    is then adjudicated numerically by :func:`_residual_numerically_zero`.
    """
    import sympy as sp

    e = sp.expand(expr)
    if e == 0:
        return 0.0
    worst = 0.0
    for term in e.as_ordered_terms():
        coeff, _rest = term.as_coeff_Mul()
        if not coeff.is_number:
            return None
        try:
            worst = max(worst, abs(float(coeff)))
        except (TypeError, ValueError):
            return None
    return worst


# Numeric tolerance for adjudicating a symbolic residual the coefficient screen
# did not clear: |Δ| ≤ this · max(|rhs|, 1) at sample states ⇒ round-off, not a
# real difference. ~1e-7 sits far below any dynamically-relevant term and well
# above evaluation round-off.
_L4_NUM_RTOL = 1e-7


def _residual_numerically_zero(diff, rhs_ref, n_sp: int, state) -> bool | None:
    """Whether the symbolic residual ``diff`` evaluates to ~0 at sample states.

    Robust to the term cancellation that makes the per-coefficient magnitude a
    poor proxy: substitutes the species symbols (``_s0…``) and time at the shared
    initial state and a nonlinear-exercising perturbation, and compares ``|Δ|``
    against the RHS magnitude. Returns ``True`` (round-off), ``False`` (a genuine,
    numerically-confirmed difference), or ``None`` (could not be evaluated — e.g.
    a complex intermediate from a real branch).
    """
    import numpy as np
    import sympy as sp

    syms = [sp.Symbol(f"_s{i}") for i in range(n_sp)]
    t_sym = sp.Symbol("_t")
    y0 = np.asarray(state, dtype=float)
    if y0.shape[0] != n_sp:
        return None
    for t_val, y in ((0.0, y0), (1.0, y0 * 1.37 + 0.05)):
        subs = {syms[i]: float(y[i]) for i in range(n_sp)}
        subs[t_sym] = float(t_val)
        try:
            dv = complex(diff.xreplace(subs))
            rv = complex(rhs_ref.xreplace(subs))
        except (TypeError, ValueError):
            return None
        if abs(dv.imag) > 1e-12 or abs(rv.imag) > 1e-12:
            return None
        if abs(dv.real) > _L4_NUM_RTOL * max(abs(rv.real), 1.0):
            return False
    return True


def _symbolic_verdict(a_model: Model, b_model: Model) -> tuple[str, str]:
    """Compare two models' symbolic per-species RHS. Returns (status, detail).

    ``status`` is ``"equal"``, ``"not-equal"`` or ``"inconclusive"``. Only the
    dynamic species both models agree on are compared (a species fixed in one
    representation and folded in the other carries no comparable RHS). A residual
    Δ whose coefficients are all below :data:`_L4_FP_RTOL` of the RHS's own
    coefficient scale is forgiven as floating-point round-off (reported ``equal``,
    noted in the detail), since the conversion round-trip leaves coefficient dust
    sympy cannot crush to 0 even when the math is identical; a magnitude-carrying
    Δ is still ``not-equal``.
    """
    try:
        import sympy as sp
    except ImportError:
        return "inconclusive", "sympy not installed"

    ra = _symbolic_rhs(a_model)
    rb = _symbolic_rhs(b_model)
    if ra is None or rb is None:
        which = "source" if ra is None else "conversion"
        return (
            "inconclusive",
            f"symbolic RHS not reconstructible from the {which} "
            "(MM/volume-scaled/table/piecewise/transcendental kinetics — "
            "best-effort L4 punts)",
        )
    rhs_a, n_a = ra
    rhs_b, n_b = rb
    if n_a != n_b:
        return "not-equal", f"species count {n_a} != {n_b}"

    shared = sorted(set(rhs_a) & set(rhs_b))
    state = a_model.get_state()
    forgiven = 0
    worst_resid = 0.0
    unprovable = 0
    for i in shared:
        diff = sp.simplify(rhs_a[i] - rhs_b[i])
        if diff == 0:
            continue
        # Cheap screen: an all-small-coefficient residual is unambiguous dust.
        resid = _max_numeric_coeff(diff)
        if resid is not None:
            scale = max(
                _max_numeric_coeff(rhs_a[i]) or 0.0,
                _max_numeric_coeff(rhs_b[i]) or 0.0,
                1.0,
            )
            if resid <= _L4_FP_RTOL * scale:
                forgiven += 1
                worst_resid = max(worst_resid, resid)
                continue
        # The screen did not clear it — a large coefficient may still cancel
        # numerically (catastrophic cancellation), so adjudicate by evaluation.
        verdict = _residual_numerically_zero(diff, rhs_a[i], n_a, state)
        if verdict is False:
            return "not-equal", (
                f"species index {i} RHS differs (numerically confirmed): simplify(Δ) = {diff}"
            )
        # True (numerically ~0 via cancellation) or None (uncomputable) → simplify
        # could not symbolically reduce it; we do not claim algebraic equality.
        unprovable += 1

    skipped = n_a - len(shared)
    if unprovable:
        return "inconclusive", (
            f"{unprovable} species' RHS residual could not be symbolically "
            "reduced to 0 (numerically consistent with equality — likely "
            "round-off/cancellation simplify cannot crack; best-effort L4 punts)"
        )
    note = f" ({skipped} fixed/boundary species excluded)" if skipped else ""
    if forgiven:
        note += (
            f"; {forgiven} equal only up to floating-point round-off "
            f"(max residual coeff {worst_resid:.1e})"
        )
    return "equal", (
        f"all {len(shared)} dynamic per-species ODE RHS expressions are algebraically equal{note}"
    )


# ─── Result types ──────────────────────────────────────────────────────────


@dataclass
class LevelResult:
    """Outcome of one validation level."""

    level: str  # "L0".."L4"
    name: str  # short human name
    gating: bool  # True for L0–L3, False for the best-effort L4
    status: str  # "pass"/"fail"/"skip" (gating) or "equal"/"not-equal"/"inconclusive" (L4)
    detail: str  # plain-English explanation
    metrics: dict = field(default_factory=dict)

    @property
    def ran(self) -> bool:
        return self.status != "skip"

    @property
    def ok(self) -> bool:
        """Whether this level *permits* the conversion to pass overall.

        A gating level must be ``pass`` (or ``skip``); a non-gating level (L4)
        never blocks — it is informational, so any status is ``ok``.
        """
        if not self.gating:
            return True
        return self.status in ("pass", "skip")

    def summary(self) -> str:
        return f"{self.level} {self.name}: {self.status.upper()} — {self.detail}"


@dataclass
class ConversionValidationReport:
    """Structured artifact from :func:`validate_conversion` (GH #217)."""

    source: str
    direction: str  # "sbml2net" | "net2sbml"
    target_path: str | None
    levels: list[LevelResult] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    lossy: list[str] = field(default_factory=list)

    def level(self, name: str) -> LevelResult | None:
        for lv in self.levels:
            if lv.level == name:
                return lv
        return None

    @property
    def ok(self) -> bool:
        """True when every *gating* level that ran passed (L4 never blocks)."""
        return all(lv.ok for lv in self.levels)

    @property
    def gates_passed(self) -> bool:
        return all(lv.ok for lv in self.levels if lv.gating)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "direction": self.direction,
            "target_path": self.target_path,
            "ok": self.ok,
            "dropped": list(self.dropped),
            "lossy": list(self.lossy),
            "levels": [
                {
                    "level": lv.level,
                    "name": lv.name,
                    "gating": lv.gating,
                    "status": lv.status,
                    "detail": lv.detail,
                    "metrics": lv.metrics,
                }
                for lv in self.levels
            ],
        }

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        lines = [
            f"Conversion validation {verdict}: {self.direction}  ({self.source}"
            + (f" → {self.target_path}" if self.target_path else "")
            + ")",
        ]
        for note in self.dropped:
            lines.append(f"  dropped: {note}")
        for note in self.lossy:
            lines.append(f"  lossy:   {note}")
        for lv in self.levels:
            lines.append("  " + lv.summary())
        return "\n".join(lines)


# ─── Direction plumbing ────────────────────────────────────────────────────


_SBML_SUFFIXES = {".xml", ".sbml"}


def _detect_direction(source: Path) -> str:
    suffix = source.suffix.lower()
    if suffix in _SBML_SUFFIXES:
        return "sbml2net"
    if suffix == ".net":
        return "net2sbml"
    raise ValueError(
        f"cannot infer conversion direction from {source.name!r}: expected a "
        "'.xml'/'.sbml' (sbml2net) or '.net' (net2sbml) source; pass "
        "direction= explicitly"
    )


@dataclass
class _Plan:
    """The format-specific callables for one conversion direction."""

    load_source: Callable[..., Model]  # (path) -> Model
    write_target: Callable[..., str]  # (model, path, strict) -> str
    load_target: Callable[..., Model]  # (path) -> Model
    write_back: Callable[..., str]  # (model, path, strict) -> str  (target → source format)
    load_back: Callable[..., Model]  # (path) -> Model
    capability: Callable[..., dict]  # (model) -> {"dropped": [...], "lossy": [...]}
    target_suffix: str
    back_suffix: str
    target_is_sbml: bool


def _build_plan(direction: str) -> _Plan:
    from bngsim._model import Model
    from bngsim.convert._net_writer import capability_report, write_net
    from bngsim.convert._sbml_writer import sbml_capability_report, write_sbml

    def _w_net(model, path, strict):
        return write_net(model, path, strict=strict)

    def _w_sbml(model, path, strict):
        return write_sbml(model, path, strict=strict)

    if direction == "net2sbml":
        return _Plan(
            load_source=Model.from_net,
            write_target=_w_sbml,
            load_target=Model.from_sbml,
            write_back=_w_net,
            load_back=Model.from_net,
            capability=sbml_capability_report,
            target_suffix=".xml",
            back_suffix=".net",
            target_is_sbml=True,
        )
    if direction == "sbml2net":
        return _Plan(
            load_source=Model.from_sbml,
            write_target=_w_net,
            load_target=Model.from_net,
            write_back=_w_sbml,
            load_back=Model.from_sbml,
            capability=capability_report,
            target_suffix=".net",
            back_suffix=".xml",
            target_is_sbml=False,
        )
    raise ValueError(f"unknown direction {direction!r}; expected 'sbml2net' or 'net2sbml'")


# ─── Per-level checks ──────────────────────────────────────────────────────


def _check_l0_sbml(text: str) -> LevelResult:
    """L0 for an SBML target: libsbml read + consistency, gating on ERROR/FATAL."""
    import libsbml

    doc = libsbml.readSBMLFromString(text)
    doc.checkConsistency()
    fatals: list[str] = []
    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        sev = err.getSeverity()
        if sev in (libsbml.LIBSBML_SEV_ERROR, libsbml.LIBSBML_SEV_FATAL):
            fatals.append(f"line {err.getLine()}: {err.getShortMessage()}")
    if fatals:
        return LevelResult(
            "L0",
            "syntactic validity",
            True,
            "fail",
            "libsbml reported " + str(len(fatals)) + " error(s): " + "; ".join(fatals[:5]),
            {"n_errors": len(fatals)},
        )
    return LevelResult(
        "L0",
        "syntactic validity",
        True,
        "pass",
        "the converted SBML passes libsbml's consistency checks (no error/fatal diagnostics)",
        {"n_errors": 0},
    )


def _check_l0_net(target_model: Model | None, load_error: str | None) -> LevelResult:
    """L0 for a ``.net`` target: the ``.net`` reader accepted a usable network."""
    if load_error is not None:
        return LevelResult(
            "L0",
            "syntactic validity",
            True,
            "fail",
            f"the .net reader rejected the converted output: {load_error}",
            {},
        )
    # load_error is None ⟺ the load succeeded and target_model is set.
    assert target_model is not None
    n_sp = target_model.n_species
    n_rxn = target_model.n_reactions
    if n_sp > 0 and n_rxn > 0:
        return LevelResult(
            "L0",
            "syntactic validity",
            True,
            "pass",
            f"the .net reader accepts the converted output ({n_sp} species, {n_rxn} reactions)",
            {},
        )
    return LevelResult(
        "L0",
        "syntactic validity",
        True,
        "fail",
        f"the converted .net parsed to an empty network ({n_sp} species, {n_rxn} reactions)",
        {},
    )


def _structural_metrics(src: Model, dst: Model) -> dict:
    def _real_params(m: Model) -> int:
        return sum(
            1 for n in m.param_names if not n.startswith((SYNTHETIC_PREFIX, FLUX_HELPER_PREFIX))
        )

    return {
        "n_species": (src.n_species, dst.n_species),
        # Fold signed-flux fragments back to their source reaction so the count
        # reflects the original topology, not the RHS-identical re-encoding.
        "n_reactions": (_effective_n_reactions(src), _effective_n_reactions(dst)),
        "n_parameters": (_real_params(src), _real_params(dst)),
        "n_observables": (src.n_observables, dst.n_observables),
    }


def _check_l1(source_model: Model, target_model: Model) -> LevelResult:
    """L1 structural equivalence: species/reaction counts + reaction topology.

    Parameter and observable counts are *recorded but not gated*: the two
    loaders label these differently (``from_sbml`` auto-reports every species as
    an observable and stores summing rules as time-varying parameters; the
    ``.net`` writer may synthesize helper functions), so a strict count match
    across formats would flag a benign labelling difference, not a defect. The
    invariant that must hold — and is gated — is the reaction reactant/product
    topology over the dynamic (non-fixed) species.
    """
    metrics = _structural_metrics(source_model, target_model)
    mismatches: list[str] = []
    if metrics["n_species"][0] != metrics["n_species"][1]:
        mismatches.append("species count {} != {}".format(*metrics["n_species"]))
    if metrics["n_reactions"][0] != metrics["n_reactions"][1]:
        mismatches.append("reactions count {} != {}".format(*metrics["n_reactions"]))

    sig_a = _dynamic_stoich_signatures(source_model)
    sig_b = _dynamic_stoich_signatures(target_model)
    if sig_a != sig_b:
        only_a = sig_a - sig_b
        only_b = sig_b - sig_a
        if only_a:
            mismatches.append(f"{sum(only_a.values())} reaction topolog(ies) only in source")
        if only_b:
            mismatches.append(f"{sum(only_b.values())} reaction topolog(ies) only in conversion")

    if mismatches:
        return LevelResult(
            "L1", "structural equivalence", True, "fail", "; ".join(mismatches), metrics
        )
    return LevelResult(
        "L1",
        "structural equivalence",
        True,
        "pass",
        "species/reaction counts and reaction reactant→product topology match "
        "(parameter/observable counts recorded, not gated — they differ benignly "
        "by loader convention)",
        metrics,
    )


def _check_l2(source_model: Model, back_model: Model, *, rhs_tol: float) -> LevelResult:
    """L2 round-trip identity: ``X → Y → X`` reproduces the source model graph.

    Identity is judged at the loaded-model level — species/reaction counts, the
    dynamic reaction topology, and the ODE right-hand side — rather than by a
    text diff, which is fragile to benign reordering and relabelling. The RHS
    check is the substantive identity gate: it confirms the round-trip preserved
    the actual dynamics, not just the shape.
    """
    metrics = _structural_metrics(source_model, back_model)
    mismatches: list[str] = []
    if metrics["n_species"][0] != metrics["n_species"][1]:
        mismatches.append("species count {} != {}".format(*metrics["n_species"]))

    # The ODE RHS is the substantive identity gate. Reaction topology is *not*
    # hard-gated here: when the forward conversion re-encoded a functional law as
    # per-species signed flux, that re-encoding does not survive a second hop
    # (X→Y→X) in a form the fold-back can recognize — a ``C→0`` degradation comes
    # back as a ``0→C`` reaction carrying a negative-signed rate — so the stoich
    # signature legitimately differs while the dynamics are identical. A topology
    # change with a *matching* RHS is a benign re-encoding (recorded, not failed);
    # a real corruption shows up as an RHS divergence.
    delta = None
    topo_reencoded = False
    if not mismatches:
        delta = _max_rhs_delta(source_model, back_model)
        metrics["max_rhs_delta"] = delta
        if not (delta <= rhs_tol):
            mismatches.append(
                f"ODE RHS differs (max scale-relative |Δ| = {delta:.2e} > {rhs_tol:.0e})"
            )
        else:
            topo_reencoded = _dynamic_stoich_signatures(source_model) != (
                _dynamic_stoich_signatures(back_model)
            )

    if mismatches:
        return LevelResult(
            "L2", "round-trip identity", True, "fail", "; ".join(mismatches), metrics
        )
    note = (
        " (reactions re-encoded as per-species signed flux; RHS-identical)"
        if topo_reencoded
        else ""
    )
    return LevelResult(
        "L2",
        "round-trip identity",
        True,
        "pass",
        f"X→Y→X reproduces the source ODE RHS (max scale-relative |Δ dy/dt| = {delta:.2e}){note}",
        metrics,
    )


def _simulate_species(model: Model, t_span, n_points) -> np.ndarray:
    import numpy as np

    import bngsim

    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=t_span, n_points=n_points)
    return np.asarray(result.species, dtype=float)


def _check_l3(source_model: Model, target_model: Model, *, t_span, n_points) -> LevelResult:
    """L3 numerical equivalence: simulate both, compare species trajectories.

    Species order is preserved through the conversion, so the two ``(n_time,
    n_species)`` arrays are index-aligned and compared cell-by-cell under the
    scale-aware verdict. Observables are *not* compared — the loaders report
    different observable sets, while the species state is the conserved,
    apples-to-apples quantity.
    """
    try:
        a = _simulate_species(source_model, t_span, n_points)
        b = _simulate_species(target_model, t_span, n_points)
    except Exception as exc:  # integration blew up on one side → genuine failure
        return LevelResult(
            "L3",
            "numerical equivalence",
            True,
            "fail",
            f"simulation failed: {type(exc).__name__}: {exc}",
            {},
        )

    if a.shape != b.shape:
        return LevelResult(
            "L3",
            "numerical equivalence",
            True,
            "fail",
            f"species-trajectory shape mismatch {a.shape} vs {b.shape}",
            {},
        )

    verdict = _scale_aware_verdict(a, b)
    metrics = {
        "max_rel": verdict["max_rel"],
        "max_abs": verdict["max_abs"],
        "n_fail": verdict["n_fail"],
        "n_cells": verdict.get("n_cells"),
        "t_span": list(t_span),
        "n_points": n_points,
    }
    if verdict["passed"]:
        return LevelResult(
            "L3",
            "numerical equivalence",
            True,
            "pass",
            f"source and conversion agree on all {verdict.get('n_cells')} "
            f"species·time cells under the scale-aware tolerance "
            f"(max rel |Δ| = {verdict['max_rel']:.2e})",
            metrics,
        )
    return LevelResult(
        "L3",
        "numerical equivalence",
        True,
        "fail",
        f"{verdict['n_fail']} species·time cell(s) diverge beyond the scale-aware "
        f"tolerance (max rel |Δ| = {verdict['max_rel']:.2e}, max abs = {verdict['max_abs']:.2e})",
        metrics,
    )


def _check_l4(source_model: Model, target_model: Model) -> LevelResult:
    """L4 symbolic equivalence (best-effort, non-gating)."""
    status, detail = _symbolic_verdict(source_model, target_model)
    return LevelResult("L4", "symbolic equivalence", False, status, detail, {})


# ─── Entry point ───────────────────────────────────────────────────────────


_ALL_LEVELS = ("L0", "L1", "L2", "L3", "L4")


def _resolve_levels(levels: tuple[str, ...] | str) -> list[str]:
    """Normalize a ``levels`` request to an ordered ``["L0", …]`` list."""
    if levels == "all":
        return list(_ALL_LEVELS)
    want = [lv.upper() for lv in levels]
    unknown = [lv for lv in want if lv not in _ALL_LEVELS]
    if unknown:
        raise ValueError(f"unknown level(s) {unknown}; valid: {_ALL_LEVELS}")
    return want


def _protocol_horizon(protocol) -> tuple[tuple[float, float], int] | None:
    """The L3 simulation grid from a parsed BNGL protocol, or ``None``.

    Returns the primary experiment's ``(t_span, n_points)`` when it carries a
    real horizon (``t_end > t_start``) — the model's *own* integrable time range,
    which both avoids the blanket-horizon stiff-hang and exercises the trajectory
    over the interval the modeller actually simulated. The method is irrelevant:
    L3 proves the conversion preserved the dynamics, which is a deterministic
    property checked ODE-vs-ODE, so even an SSA protocol's horizon is usable.
    """
    if protocol is None:
        return None
    exp = protocol.primary_experiment()
    if exp is None or not (exp.t_span[1] > exp.t_span[0]):
        return None
    return exp.t_span, exp.n_points


def _grade_target(
    plan: _Plan,
    source_model: Model,
    target_path: Path,
    *,
    want: list[str],
    work: Path,
    source_stem: str,
    rhs_tol: float,
    t_span: tuple[float, float],
    n_points: int,
    strict: bool,
    caps: dict,
    source: str,
    direction: str,
    report_target_path: str | None,
    protocol=None,
) -> ConversionValidationReport:
    """Grade an already-written ``target_path`` at the requested L0–L4 levels.

    The shared kernel: it reloads the converted target once, runs each requested
    level (L0 syntactic, L1 structural, L2 reverse round-trip, L3 numerical, L4
    symbolic), and returns the :class:`ConversionValidationReport`. It never runs
    the *forward* conversion — the caller supplies the target — so wiring this
    into the converter does not convert twice.
    """
    from bngsim._exceptions import ConversionError

    # Reload the target once; L0–L4 reuse it.
    target_model = None
    load_error = None
    try:
        target_model = plan.load_target(target_path)
    except Exception as exc:  # noqa: BLE001 — surfaced as L0 failure
        load_error = f"{type(exc).__name__}: {exc}"

    report = ConversionValidationReport(
        source=source,
        direction=direction,
        target_path=report_target_path,
        dropped=list(caps.get("dropped", [])),
        lossy=list(caps.get("lossy", [])),
    )

    # ── L0 ───────────────────────────────────────────────────────────
    if "L0" in want:
        if plan.target_is_sbml:
            if load_error is not None:
                report.levels.append(
                    LevelResult(
                        "L0",
                        "syntactic validity",
                        True,
                        "fail",
                        f"the SBML reader rejected the converted output: {load_error}",
                        {},
                    )
                )
            else:
                report.levels.append(_check_l0_sbml(target_path.read_text()))
        else:
            report.levels.append(_check_l0_net(target_model, load_error))

    # If the target did not load, the remaining model-level levels cannot
    # run; record them as failed gates so .ok reflects the broken conversion.
    if target_model is None:
        for lv in ("L1", "L2", "L3", "L4"):
            if lv in want:
                gating = lv != "L4"
                report.levels.append(
                    LevelResult(
                        lv,
                        _LEVEL_NAMES[lv],
                        gating,
                        "fail" if gating else "inconclusive",
                        "skipped: the converted output did not reload",
                        {},
                    )
                )
        return report

    # ── L1 ───────────────────────────────────────────────────────────
    if "L1" in want:
        report.levels.append(_check_l1(source_model, target_model))

    # ── L2 ───────────────────────────────────────────────────────────
    if "L2" in want:
        try:
            back_path = work / (source_stem + "_roundtrip" + plan.back_suffix)
            plan.write_back(target_model, back_path, strict)
            back_model = plan.load_back(back_path)
            report.levels.append(_check_l2(source_model, back_model, rhs_tol=rhs_tol))
        except ConversionError as exc:
            report.levels.append(
                LevelResult(
                    "L2",
                    "round-trip identity",
                    True,
                    "fail",
                    f"the reverse conversion refused the model: {exc}",
                    {},
                )
            )
        except Exception as exc:  # noqa: BLE001
            report.levels.append(
                LevelResult(
                    "L2",
                    "round-trip identity",
                    True,
                    "fail",
                    f"round-trip failed: {type(exc).__name__}: {exc}",
                    {},
                )
            )

    # ── L3 ───────────────────────────────────────────────────────────
    if "L3" in want:
        horizon = _protocol_horizon(protocol)
        l3_t_span, l3_n_points = horizon if horizon is not None else (t_span, n_points)
        result = _check_l3(source_model, target_model, t_span=l3_t_span, n_points=l3_n_points)
        if horizon is not None:
            # The protocol is a .bngl simulate action (net2sbml) or a SED-ML time
            # course (sbml2net); both arrive as a ProtocolSpec horizon here.
            result.metrics["horizon_source"] = (
                "sedml protocol" if direction == "sbml2net" else "bngl protocol"
            )
        report.levels.append(result)

    # ── L4 ───────────────────────────────────────────────────────────
    if "L4" in want:
        report.levels.append(_check_l4(source_model, target_model))

    return report


def grade_conversion(
    direction: str,
    source_model: Model,
    target_text: str,
    *,
    levels: tuple[str, ...] | str = "all",
    rhs_tol: float = 1e-6,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
    strict: bool = True,
    caps: dict | None = None,
    source: str = "<memory>",
    source_stem: str = "model",
    work: str | Path | None = None,
    report_target_path: str | None = None,
    protocol=None,
) -> ConversionValidationReport:
    """Grade an **already-converted** model pair at levels L0–L4 (GH #217).

    The shared core behind :func:`validate_conversion` and the ``validate="full"``
    gate of :func:`bngsim.convert.net_to_sbml` / :func:`sbml_to_net`. The caller
    has already produced ``target_text`` (the conversion output) from
    ``source_model``; this reloads that text, runs the requested levels
    (including the L2 reverse round-trip), and returns the
    :class:`ConversionValidationReport`. Re-serializing the supplied text to a
    work file is cheap — the expensive forward conversion is **not** repeated, so
    the converter can gate itself without converting twice.

    Parameters
    ----------
    direction : {"net2sbml", "sbml2net"}
        Which conversion produced ``target_text``.
    source_model : Model
        The already-loaded source model the conversion came from.
    target_text : str
        The converted output (SBML or ``.net`` text) to grade.
    levels, rhs_tol, t_span, n_points, strict :
        As for :func:`validate_conversion`.
    caps : dict | None
        Capability report (``{"dropped": [...], "lossy": [...]}``) to carry into
        the verdict; defaults to empty.
    source : str
        Label for the report's ``source`` field.
    source_stem : str
        Base name for the work artifacts written under ``work``.
    work : str | Path | None
        Directory for the reloaded target + L2 round-trip artifacts. ``None``
        uses a temp directory cleaned up on return.
    report_target_path : str | None
        Value for the report's ``target_path`` field (the kept artifact path, or
        ``None`` when artifacts are ephemeral).
    protocol : ProtocolSpec | None
        A parsed BNGL simulation protocol. When given, L3 simulates over the
        protocol's real horizon (the model's own integrable time range) instead
        of the blanket ``t_span``/``n_points`` — avoiding the stiff-model hang and
        exercising the trajectory the modeller actually ran.

    Returns
    -------
    ConversionValidationReport
    """
    plan = _build_plan(direction)
    want = _resolve_levels(levels)
    caps = caps if caps is not None else {"dropped": [], "lossy": []}

    import contextlib

    with contextlib.ExitStack() as stack:
        if work is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        else:
            work = Path(work)
            work.mkdir(parents=True, exist_ok=True)

        target_path = work / (source_stem + plan.target_suffix)
        target_path.write_text(target_text)

        return _grade_target(
            plan,
            source_model,
            target_path,
            want=want,
            work=work,
            source_stem=source_stem,
            rhs_tol=rhs_tol,
            t_span=t_span,
            n_points=n_points,
            strict=strict,
            caps=caps,
            source=source,
            direction=direction,
            report_target_path=report_target_path,
            protocol=protocol,
        )


def validate_conversion(
    source_path: str | Path,
    *,
    direction: str | None = None,
    levels: tuple[str, ...] | str = "all",
    strict: bool = True,
    rhs_tol: float = 1e-6,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
    out_dir: str | Path | None = None,
) -> ConversionValidationReport:
    """Grade a format conversion at levels **L0–L4** (GH #217).

    Converts ``source_path`` to the opposite format and validates the result at
    escalating rigor: L0 syntactic validity, L1 structural equivalence, L2
    round-trip identity, L3 numerical equivalence (all hard gates), and L4
    symbolic equivalence (best-effort, non-gating). The direction is inferred
    from the source suffix (``.xml``/``.sbml`` → ``sbml2net``; ``.net`` →
    ``net2sbml``) unless given explicitly.

    This is the path-driven entry point: it loads and converts ``source_path``
    once, then delegates the grading to :func:`grade_conversion` (which the
    converter calls directly on the artifact it just produced, avoiding a second
    conversion).

    Parameters
    ----------
    source_path : str | Path
        Source model to convert and validate.
    direction : {"sbml2net", "net2sbml"} | None
        Conversion direction. ``None`` infers it from the source suffix.
    levels : tuple[str, ...] | "all"
        Which levels to run (subset of ``("L0","L1","L2","L3","L4")``).
        ``"all"`` runs every level.
    strict : bool
        Forwarded to the writers: raise on constructs the target format cannot
        carry faithfully (``False`` emits a best-effort document instead). Note
        that an unfaithful conversion will still be *caught* by L1–L3.
    rhs_tol : float
        Scale-relative tolerance for the L2 ODE-RHS identity check.
    t_span, n_points :
        Time grid for the L3 simulation comparison.
    out_dir : str | Path | None
        Where to keep the converted artifacts. ``None`` uses a temp directory
        that is cleaned up on return (the report still carries the in-memory
        verdicts).

    Returns
    -------
    ConversionValidationReport
    """
    from bngsim._exceptions import ConversionError

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"no such source model: {source_path}")
    direction = direction or _detect_direction(source_path)
    plan = _build_plan(direction)
    want = _resolve_levels(levels)

    import contextlib

    with contextlib.ExitStack() as stack:
        if out_dir is not None:
            work = Path(out_dir)
            work.mkdir(parents=True, exist_ok=True)
        else:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory()))

        source_model = plan.load_source(source_path)
        caps = plan.capability(source_model)

        target_path = work / (source_path.stem + plan.target_suffix)
        try:
            target_text = plan.write_target(source_model, target_path, strict)
        except ConversionError as exc:
            # The converter declined to emit an unfaithful conversion (strict
            # mode). There is no artifact to grade, so return a verdict marking
            # every requested gating level failed with the refusal reason —
            # never crash the suite/CLI; the caller can retry with strict=False.
            report = ConversionValidationReport(
                source=str(source_path),
                direction=direction,
                target_path=None,
                dropped=list(caps.get("dropped", [])),
                lossy=list(caps.get("lossy", [])),
            )
            for lv in want:
                gating = lv != "L4"
                report.levels.append(
                    LevelResult(
                        lv,
                        _LEVEL_NAMES[lv],
                        gating,
                        "fail" if gating else "inconclusive",
                        f"conversion refused under strict mode: {exc}",
                        {},
                    )
                )
            return report

        return grade_conversion(
            direction,
            source_model,
            target_text,
            levels=tuple(want),
            rhs_tol=rhs_tol,
            t_span=t_span,
            n_points=n_points,
            strict=strict,
            caps=caps,
            source=str(source_path),
            source_stem=source_path.stem,
            work=work,
            report_target_path=str(target_path) if out_dir is not None else None,
        )


_LEVEL_NAMES = {
    "L0": "syntactic validity",
    "L1": "structural equivalence",
    "L2": "round-trip identity",
    "L3": "numerical equivalence",
    "L4": "symbolic equivalence",
}
