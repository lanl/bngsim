#!/usr/bin/env python3
"""Independent ``.net``-faithful Gillespie SSA — the bng_parity SECOND oracle.

When an SSA-track job's LEGACY reference (``run_network``) fails — most often the
``edgepop`` molecule/edge-population observable crash (GH: reference-side bug) —
bngsim ran but there was no oracle to score it, so the row is REFERENCE_FAILED
(unscored). This module supplies an INDEPENDENT direct-method Gillespie on the SAME
BNG ``.net`` so the bngsim SSA ensemble can be validated against a second engine.

Independence
    A from-scratch reimplementation: its own ``.net`` parser, its own propensity
    kernel, its own RNG (NumPy PCG64), its own stepping. It shares NO code with
    bngsim's SSA, so agreement is meaningful.

Why it sidesteps ``edgepop``
    It reads the ``.net``'s pre-resolved ``groups`` block for observables — the
    network generator already reduced each observable to a weighted sum of species
    indices — so it computes exactly the observables that crash run_network's
    pattern/bond ``edgepop`` path WITHOUT any pattern matching.

Faithfulness (matches bngsim/BNG exactly)
    For an elementary mass-action reaction the propensity is
        a = K * prod_i  falling_factorial(X_i, m_i)
    where ``m_i`` is species i's reactant multiplicity and ``K`` is the full ``.net``
    rate coefficient — the rate expression (e.g. ``3*_rateLaw1``) already folds in
    the statistical factor, so ``eval(rate_expr) == rate_param * stat_factor`` which
    equals bngsim's ``K`` for a count-based (ssa_volume_factor == 1) model. This is
    the same falling-factorial convention as ``NetworkModel::compute_propensity``.

Supported-gate (``parse_net`` returns ``None`` when unsupported)
    Elementary constant-rate mass-action, count-based:
      * every reaction rate coefficient must evaluate to a finite constant using ONLY
        the parameters block (a rate that references an observable / function / time
        raises → unsupported);
      * seed-species amounts must be non-negative integers (counts, not
        concentrations — the ssa_volume_factor == 1 regime).
    Functional / time-dependent / Michaelis-Menten / concentration-unit models are
    REFUSED (never mis-scored) — those need a functional-rate oracle or COPASI.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter as _now

import numpy as np

# ── Safe constant-expression evaluator (BNG ExprTk subset) ──────────────────
_FUNCS = {
    "if": lambda c, a, b: a if c else b,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "rint": round,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
}
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_CMPOPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class _UnsupportedExpr(Exception):
    """Raised when an expression references a non-constant symbol (observable /
    function / time) or an unwhitelisted construct — the model is not supported."""


def _eval_expr(expr: str, env: dict[str, float]) -> float:
    """Evaluate a BNG constant expression against ``env`` (param name -> value).

    Raises ``_UnsupportedExpr`` for any name not in ``env`` (an observable/function/
    time reference) or any non-arithmetic construct. This IS the supported-gate for
    rate laws: a functional rate references a symbol outside the parameters block and
    therefore raises here.
    """
    try:
        node = ast.parse(expr.replace("^", "**"), mode="eval").body
    except SyntaxError as e:
        raise _UnsupportedExpr(f"unparseable: {expr!r}") from e

    def ev(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise _UnsupportedExpr(f"non-numeric constant {n.value!r}")
        if isinstance(n, ast.Name):
            if n.id in env:
                return float(env[n.id])
            raise _UnsupportedExpr(f"non-constant symbol {n.id!r}")
        if isinstance(n, ast.BinOp):
            op = _BINOPS.get(type(n.op))
            if op is None:
                raise _UnsupportedExpr(f"op {type(n.op).__name__}")
            return float(op(ev(n.left), ev(n.right)))
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +v
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.Not):
                return float(not v)
            raise _UnsupportedExpr("unary")
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            op = _CMPOPS.get(type(n.ops[0]))
            if op is None:
                raise _UnsupportedExpr("compare")
            return float(op(ev(n.left), ev(n.comparators[0])))
        if isinstance(n, ast.BoolOp) and len(n.values) == 2:
            lhs, rhs = ev(n.values[0]), ev(n.values[1])
            if isinstance(n.op, ast.And):
                return float(bool(lhs) and bool(rhs))
            if isinstance(n.op, ast.Or):
                return float(bool(lhs) or bool(rhs))
            raise _UnsupportedExpr("boolop")
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise _UnsupportedExpr(f"construct {type(n).__name__}")

    return ev(node)


# ── .net model ──────────────────────────────────────────────────────────────
@dataclass
class NetModel:
    x0: np.ndarray  # initial counts, species order (0-based)
    reactions: list  # (reactants:list[int0], products:list[int0], k:float)
    obs_names: list  # group names, in file order
    obs_terms: list  # per-obs list of (weight:float, species_idx0)
    mults: list = None  # per-rxn tuple of (species_idx0, multiplicity)
    dep_rxns: list = None  # per-species list of rxn indices it is a reactant of

    def precompute(self):
        ns = len(self.x0)
        self.mults = [tuple(Counter(reac).items()) for reac, _p, _k in self.reactions]
        dep = [[] for _ in range(ns)]
        for r, (reac, _p, _k) in enumerate(self.reactions):
            for si in set(reac):
                dep[si].append(r)
        self.dep_rxns = dep
        return self


# Cost gates for a pure-Python SSA. The reaction cap bounds the O(n_rxn)
# direct-method selection; the two event caps bound trajectory and ensemble cost.
# All three are deterministic: the seeds are fixed, so whether an ensemble fits is
# the same on every machine, and so is which oracle scores the row (issue #881).
MAX_REACTIONS = 1_500
MAX_EVENTS = 5_000_000
# The ensemble cap for a 10-reaction network. A larger network gets fewer events,
# because an event costs a fixed overhead plus a linear scan over the reactions:
# measured 3.8 us at 10 reactions and 51.5 us at 1,500 on an Apple M-series, which
# is (110 + n_reactions) scan-steps of about 0.032 us each. Capping events alone
# admitted 20 M events on a 1,500-reaction network, ~1,000 s there, which a fast
# runner finishes and a slow one stops, so the verdict would follow runner speed
# again. See ensemble_event_cap().
MAX_ENSEMBLE_EVENTS = 20_000_000
_EVENT_FIXED_COST = 110  # the fixed part of one event, in reaction-scan steps
# A SAFETY STOP only, never the thing that decides support. It used to be 90 s and
# also chose the oracle: v08's 10-replicate ensemble (5.2 M events, ~30 s on an
# Apple M-series) ran out of it on the slower nightly runners, which then scored the
# row against RoadRunner instead, so the verdict followed runner speed (issue #881).
# An ensemble the caps admit runs ~75 s at most on an Apple M-series, at any network
# size; this leaves several-fold headroom for a slow, loaded runner. Hitting it is
# reported as such.
DEFAULT_WALL_BUDGET_SEC = 900.0


def ensemble_event_cap(n_reactions: int) -> int:
    """The most events an ensemble on an *n_reactions* network may fire: the same
    cost as ``MAX_ENSEMBLE_EVENTS`` events on a 10-reaction network. It depends on
    the network alone, so whether an ensemble fits is the same on every machine."""
    return MAX_ENSEMBLE_EVENTS * (_EVENT_FIXED_COST + 10) // (_EVENT_FIXED_COST + n_reactions)


class _TooCostly(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


@dataclass
class NetGillespieOutcome:
    """What :func:`run_net_gillespie` produced, and if nothing, why.

    ``reason`` is ``"ok"`` (``values`` holds the ensemble), ``"unsupported"`` (the
    ``.net`` is outside the supported gate), ``"event_cap"`` (a deterministic event cap
    was exceeded), ``"wall_budget"`` (the safety stop fired) or ``"error"`` (the oracle
    raised). The first three are the same on every machine; ``"wall_budget"`` is not,
    which is why a caller must not let it pick a different oracle.
    """

    values: tuple | None
    reason: str
    detail: str = ""
    elapsed_sec: float = 0.0
    events: int = 0

    def describe(self) -> str:
        """A clause for a row comment: why this oracle produced nothing."""
        if self.reason == "unsupported":
            return f"does not support this .net ({self.detail})"
        if self.reason == "event_cap":
            return f"exceeded its deterministic event cap ({self.detail})"
        if self.reason == "wall_budget":
            return (
                f"hit its {self.detail} safety stop after {self.elapsed_sec:.0f} s and "
                f"{self.events} events"
            )
        if self.reason == "error":
            return f"raised {self.detail}"
        return "ran"


_SECTION = re.compile(r"begin\s+(\w+)|end\s+(\w+)")


def parse_net(path: str | Path) -> NetModel | None:
    """Parse a BNG ``.net`` into a ``NetModel``; return ``None`` if UNSUPPORTED
    (functional/time-dependent rate, or non-integer seed counts). Never guesses."""
    return parse_net_with_reason(path)[0]


def parse_net_with_reason(path: str | Path) -> tuple[NetModel | None, str]:
    """:func:`parse_net`, plus why a ``None`` is unsupported (``""`` when supported)."""
    text = Path(path).read_text()
    params: dict[str, float] = {}
    raw_params: list[tuple[str, str]] = []
    species_amt: list[str] = []
    rxn_lines: list[str] = []
    grp_lines: list[str] = []
    section = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _SECTION.match(s)
        if m:
            section = m.group(1) if m.group(1) else None
            continue
        body = s.split("#", 1)[0].strip()
        if not body:
            continue
        if section == "parameters":
            parts = body.split(None, 2)  # idx name value
            if len(parts) >= 3:
                raw_params.append((parts[1], parts[2].strip()))
        elif section == "species":
            parts = body.split()  # idx pattern... amount  (amount is last)
            species_amt.append(parts[-1])
        elif section == "reactions":
            rxn_lines.append(body)
        elif section == "groups":
            grp_lines.append(body)

    # Resolve parameters (they may reference earlier params); iterate to a fixpoint.
    pending = dict(raw_params)
    for _ in range(len(pending) + 2):
        progressed = False
        for name, expr in list(pending.items()):
            try:
                params[name] = _eval_expr(expr, params)
                del pending[name]
                progressed = True
            except _UnsupportedExpr:
                pass
        if not pending or not progressed:
            break
    # Unresolved params that a RATE needs will surface as _UnsupportedExpr below.

    # Seed species initial amounts must be integer counts.
    x0 = []
    for amt in species_amt:
        try:
            v = _eval_expr(amt, params)
        except _UnsupportedExpr as e:
            return None, f"seed amount {amt!r} is not a constant: {e}"
        if not float(v).is_integer() or v < 0:
            # concentration units / non-count → refuse
            return None, f"seed amount {amt!r} = {v:g} is not a non-negative integer count"
        x0.append(int(round(v)))

    # Reactions:  idx  reactant_idxs  product_idxs  rate_expr
    if len(rxn_lines) > MAX_REACTIONS:
        return None, f"{len(rxn_lines)} reactions > MAX_REACTIONS ({MAX_REACTIONS})"
    reactions = []
    for rl in rxn_lines:
        parts = rl.split()
        if len(parts) < 4:
            return None, f"malformed reaction line {rl!r}"
        reac = [] if parts[1] == "0" else [int(i) - 1 for i in parts[1].split(",")]
        prod = [] if parts[2] == "0" else [int(i) - 1 for i in parts[2].split(",")]
        rate_expr = parts[3]
        try:
            k = _eval_expr(rate_expr, params)  # folds in the statistical factor
        except _UnsupportedExpr as e:
            # functional / time-dependent rate → refuse
            return None, f"rate {rate_expr!r} is not a constant: {e}"
        if not math.isfinite(k) or k < 0:
            return None, f"rate {rate_expr!r} = {k:g} is not a finite non-negative constant"
        reactions.append((reac, prod, k))

    # Groups (observables): name  [w*]idx,[w*]idx,...
    obs_names, obs_terms = [], []
    for gl in grp_lines:
        parts = gl.split(None, 2)  # idx name terms
        if len(parts) < 2:
            return None, f"malformed group line {gl!r}"
        name = parts[1]
        terms = []
        if len(parts) == 3:
            for tok in parts[2].split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if "*" in tok:
                    w, i = tok.split("*")
                    terms.append((float(w), int(i) - 1))
                else:
                    terms.append((1.0, int(tok) - 1))
        obs_names.append(name)
        obs_terms.append(terms)

    return NetModel(np.array(x0, dtype=np.int64), reactions, obs_names, obs_terms), ""


# ── Direct-method Gillespie ──────────────────────────────────────────────────
def _prop(k, mult, x):
    """K * falling_factorial(X_si, m) over reactant multiplicities."""
    ar = k
    for si, c in mult:
        xi = x[si]
        for m in range(c):
            ar *= xi - m
        if ar <= 0.0:
            return 0.0
    return ar


def _simulate_one(
    net: NetModel,
    t_grid: np.ndarray,
    rng: np.random.Generator,
    deadline: float | None = None,
    tally: list[int] | None = None,
) -> np.ndarray:
    """One exact SSA trajectory; returns observables sampled at ``t_grid`` — shape
    (n_time, n_obs). Uses a species→reaction dependency graph so only the propensities
    touched by the fired reaction are recomputed (O(degree) per step, not O(nr)).

    ``tally``, when given, is a one-element list holding the events the ensemble has
    fired so far; it is advanced here and checked against :func:`ensemble_event_cap`
    for this network."""
    x = net.x0.copy().astype(np.int64)
    rxns = net.reactions
    ensemble_cap = ensemble_event_cap(len(rxns))
    mults = net.mults
    dep = net.dep_rxns
    nr = len(rxns)
    nt = len(t_grid)
    obs_terms = net.obs_terms
    out = np.empty((nt, len(net.obs_names)), dtype=np.float64)

    def record(row):
        for j, terms in enumerate(obs_terms):
            out[row, j] = sum(w * x[i] for w, i in terms) if terms else 0.0

    a = [_prop(rxns[r][2], mults[r], x) for r in range(nr)]
    a0 = math.fsum(a)
    t = float(t_grid[0])
    k_out = 0
    events = 0
    while k_out < nt:
        if a0 <= 0.0:
            while k_out < nt:  # no reaction can fire → hold state
                record(k_out)
                k_out += 1
            break
        tau = rng.exponential(1.0 / a0)
        t_next = t + tau
        while k_out < nt and t_grid[k_out] <= t_next:  # emit outputs crossed by this step
            record(k_out)
            k_out += 1
        t = t_next
        # select reaction (linear scan; nr is small for these networks)
        thresh = rng.random() * a0
        cum = 0.0
        chosen = nr - 1
        for r in range(nr):
            cum += a[r]
            if cum >= thresh:
                chosen = r
                break
        reac, prod, _k = rxns[chosen]
        for si in reac:
            x[si] -= 1
        for si in prod:
            x[si] += 1
        # refresh only propensities affected by the changed species
        touched = set()
        for si in reac:
            touched.update(dep[si])
        for si in prod:
            touched.update(dep[si])
        for r in touched:
            a[r] = _prop(rxns[r][2], mults[r], x)
        a0 = math.fsum(a)  # re-sum (kept exact; nr small)
        events += 1
        if events > MAX_EVENTS:
            raise _TooCostly("event_cap", f"one trajectory passed MAX_EVENTS = {MAX_EVENTS}")
        if tally is not None:
            tally[0] += 1
            if tally[0] > ensemble_cap:
                raise _TooCostly(
                    "event_cap",
                    f"the ensemble passed {ensemble_cap} events, MAX_ENSEMBLE_EVENTS = "
                    f"{MAX_ENSEMBLE_EVENTS} scaled to {nr} reactions",
                )
        if deadline is not None and (events & 0xFFFF) == 0 and _now() > deadline:
            raise _TooCostly("wall_budget", "wall budget exceeded")
    return out


def run_net_gillespie(
    net_path, t_grid, n_rep: int, seed_base: int, wall_budget_sec: float = DEFAULT_WALL_BUDGET_SEC
) -> NetGillespieOutcome:
    """Independent SSA ensemble on ``net_path`` sampled at ``t_grid``, never raising.

    On success ``values`` is ``(t_grid, values, obs_names)`` with ``values`` shape
    (n_rep, n_time, n_obs), the same layout the harness's bngsim ensemble uses. On
    failure ``values`` is ``None`` and ``reason`` says why (see
    :class:`NetGillespieOutcome`), so a report can tell "unsupported" from "too
    costly" from "ran out of time" from "raised" (issue #881).
    """
    start = _now()
    tally = [0]
    try:
        net, why = parse_net_with_reason(net_path)
        if net is None:
            return NetGillespieOutcome(None, "unsupported", why, _now() - start)
        net.precompute()
        t_grid = np.asarray(t_grid, dtype=np.float64)
        vals = np.empty((n_rep, len(t_grid), len(net.obs_names)), dtype=np.float64)
        deadline = start + wall_budget_sec
        for rep in range(n_rep):
            rng = np.random.default_rng(seed_base + rep)  # independent PCG64 stream
            vals[rep] = _simulate_one(net, t_grid, rng, deadline=deadline, tally=tally)
    except _TooCostly as e:
        detail = f"{wall_budget_sec:g} s" if e.reason == "wall_budget" else str(e)
        return NetGillespieOutcome(None, e.reason, detail, _now() - start, tally[0])
    except Exception as e:  # noqa: BLE001 - reported on the row, never raised
        return NetGillespieOutcome(
            None, "error", f"{type(e).__name__}: {e}", _now() - start, tally[0]
        )
    return NetGillespieOutcome((t_grid, vals, net.obs_names), "ok", "", _now() - start, tally[0])


def net_gillespie_ensemble(
    net_path, t_grid, n_rep: int, seed_base: int, wall_budget_sec: float = DEFAULT_WALL_BUDGET_SEC
):
    """:func:`run_net_gillespie`'s ensemble, or ``None`` for any reason it has none."""
    return run_net_gillespie(net_path, t_grid, n_rep, seed_base, wall_budget_sec).values


if __name__ == "__main__":
    import sys

    net = parse_net(sys.argv[1])
    if net is None:
        print("UNSUPPORTED (.net has functional/time-dependent rates or non-count seeds)")
        sys.exit(2)
    print(f"supported: {len(net.reactions)} reactions, {len(net.x0)} species, obs={net.obs_names}")
    tg = np.linspace(0.0, float(sys.argv[2]) if len(sys.argv) > 2 else 10.0, 51)
    _, v, names = net_gillespie_ensemble(
        sys.argv[1], tg, n_rep=int(sys.argv[3]) if len(sys.argv) > 3 else 20, seed_base=1
    )
    print(
        "final-time ensemble mean:",
        dict(zip(names, v[:, -1, :].mean(axis=0).round(3), strict=False)),
    )
