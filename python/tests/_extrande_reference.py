"""Extrande reference SSA for time-dependent propensities (GH #81 oracle).

Why this exists
---------------
The rate-rule-under-SSA work (GH #81) needs an *independent* reference engine to
validate bngsim's hybrid sampler on models whose propensities vary in time
because a rate rule drives a species/parameter/compartment that feeds a kinetic
law. The obvious off-the-shelf references do not cover this:

* **RoadRunner ``gillespie``** categorically refuses rate rules
  (``RuntimeError: the gillespie integrator is unable to simulate a model with
  rate rules``).
* **COPASI**'s stochastic (direct/next-reaction) Time-Course likewise fails on a
  rate-rule-driven propensity; only its *hybrid* LSODA/RK methods will run such a
  model, and those are themselves an ODE+SSA approximation of the same kind we are
  validating — a sanity cross-check on the mean, not an exact oracle.

So we hand-roll **Extrande** (Voliotis, Thomas, Grima & Bowsher, *PLoS Comput.
Biol.* 2016), the thinning/rejection sampler purpose-built for time-varying
propensities. It is *exact* (statistically equivalent to the time-dependent SSA),
needs no per-channel propensity integration, and only needs a propensity upper
bound over a short look-ahead — trivial here because the time variation is smooth.
This module is deliberately a *different algorithm* (rejection) from bngsim's
sub-step direct method, so agreement between them is strong evidence of
correctness, not a shared-bug coincidence.

It is also fully self-contained — it takes an explicit Python model spec
(stoichiometries + propensity callables + continuous-variable ODEs), not a bngsim
model — so it cannot inherit a bngsim modeling bug.

Model spec
----------
``ReactionSpec(stoich, propensity)``
    ``stoich``: dict ``species_name -> int`` net change when the reaction fires
    (negative = consumed). ``propensity``: ``state -> float`` (amount/time).

``RefModel``
    ``species``: ordered discrete-species names (integer molecule counts).
    ``x0``: initial counts (amounts).
    ``cont``: dict ``name -> (state -> dc/dt)`` deterministic continuous variables
    (rate-rule targets: species/parameter/volume). ``c0``: their initial values.
    ``reactions``: list of ``ReactionSpec``.

A ``state`` passed to every callable is a plain dict mapping each discrete
species and continuous variable to its current value, plus ``"t"`` = current time.
Continuous variables are integrated with classical RK4 between events (the
discrete state is constant there, so ``dc/dt`` is an ordinary ODE in ``t``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

State = dict


@dataclass
class ReactionSpec:
    stoich: dict[str, int]
    propensity: Callable[[State], float]


@dataclass
class RefModel:
    species: list[str]
    x0: dict[str, float]
    reactions: list[ReactionSpec]
    cont: dict[str, Callable[[State], float]] = field(default_factory=dict)
    c0: dict[str, float] = field(default_factory=dict)


def _make_state(model: RefModel, x: np.ndarray, c: dict[str, float], t: float) -> State:
    s = {name: x[i] for i, name in enumerate(model.species)}
    s.update(c)
    s["t"] = t
    return s


def _rk4_step(model: RefModel, x: np.ndarray, c: dict[str, float], t: float, dt: float) -> dict:
    """Advance the continuous variables one RK4 step of length ``dt``.

    The discrete counts ``x`` are held fixed (no reaction fires inside the step),
    so each continuous variable obeys an ordinary ODE ``dc/dt = g(x, c, t)``.
    """
    if not model.cont or dt == 0.0:
        return dict(c)
    names = list(model.cont)

    def deriv(cc: dict[str, float], tt: float) -> dict[str, float]:
        st = _make_state(model, x, cc, tt)
        return {n: model.cont[n](st) for n in names}

    k1 = deriv(c, t)
    c2 = {n: c[n] + 0.5 * dt * k1[n] for n in names}
    k2 = deriv(c2, t + 0.5 * dt)
    c3 = {n: c[n] + 0.5 * dt * k2[n] for n in names}
    k3 = deriv(c3, t + 0.5 * dt)
    c4 = {n: c[n] + dt * k3[n] for n in names}
    k4 = deriv(c4, t + dt)
    return {n: c[n] + (dt / 6.0) * (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]) for n in names}


def _total_propensity(model: RefModel, x: np.ndarray, c: dict[str, float], t: float) -> float:
    st = _make_state(model, x, c, t)
    return sum(max(0.0, r.propensity(st)) for r in model.reactions)


def simulate(
    model: RefModel,
    t_out: np.ndarray,
    rng: np.random.Generator,
    *,
    look_ahead: float | None = None,
    bound_grid: int = 16,
    bound_inflate: float = 1.25,
) -> np.ndarray:
    """One Extrande trajectory; returns an array ``(len(t_out), n_species)``.

    ``look_ahead`` (L) is the thinning window; defaults to ``span / 2000``. Over
    each window the continuous variables are integrated by RK4 and the propensity
    upper bound ``B`` is the grid-max of the total propensity inflated by
    ``bound_inflate``. A genuine upper bound is required for exactness; the code
    asserts the realized propensity never exceeds ``B`` (and would surface a
    violation rather than bias silently).
    """
    t_out = np.asarray(t_out, dtype=float)
    t_start, t_end = float(t_out[0]), float(t_out[-1])
    span = t_end - t_start
    L = look_ahead if look_ahead is not None else span / 2000.0

    x = np.array([float(model.x0[s]) for s in model.species], dtype=float)
    c = {n: float(model.c0.get(n, 0.0)) for n in model.cont}
    t = t_start

    n_sp = len(model.species)
    out = np.zeros((len(t_out), n_sp), dtype=float)
    next_out = 0

    def record_until(t_now: float):
        nonlocal next_out
        while next_out < len(t_out) and t_out[next_out] <= t_now + 1e-15:
            out[next_out] = x
            next_out += 1

    record_until(t)  # t == t_start sample

    max_iter = 50_000_000
    it = 0
    while next_out < len(t_out):
        it += 1
        if it > max_iter:
            raise RuntimeError("Extrande reference exceeded iteration budget")

        win = min(L, t_end - t)
        if win <= 0.0:
            # Numerically at the end; flush remaining samples at current state.
            record_until(t_end)
            break

        # Propensity bound B over [t, t+win]: integrate c across a grid and take
        # the inflated max. Discrete x is constant across the window.
        cg = dict(c)
        tg = t
        b = _total_propensity(model, x, cg, tg)
        grid_dt = win / bound_grid
        for _ in range(bound_grid):
            cg = _rk4_step(model, x, cg, tg, grid_dt)
            tg += grid_dt
            b = max(b, _total_propensity(model, x, cg, tg))
        B = b * bound_inflate
        if B <= 0.0:
            # No propensity anywhere in the window: advance deterministically.
            c = _rk4_step(model, x, c, t, win)
            t += win
            record_until(t)
            continue

        tau = rng.exponential(1.0 / B)
        if tau >= win:
            # No (real or thinned) event in the window: advance to its end.
            c = _rk4_step(model, x, c, t, win)
            t += win
            record_until(t)
            continue

        # Candidate event at t+tau: integrate c there, then accept/thin.
        c = _rk4_step(model, x, c, t, tau)
        t += tau
        st = _make_state(model, x, c, t)
        props = np.array([max(0.0, r.propensity(st)) for r in model.reactions])
        a0 = props.sum()
        if a0 > B * (1.0 + 1e-9):
            raise AssertionError(
                f"Extrande bound violated at t={t:.6g}: a0={a0:.6g} > B={B:.6g}. "
                "Reduce look_ahead or raise bound_inflate."
            )
        u = rng.random() * B
        record_until(t)  # samples strictly before the event keep pre-event state
        if u <= a0:
            # Real event: pick the reaction proportional to its propensity.
            k = int(np.searchsorted(np.cumsum(props), u))
            if k >= len(props):
                k = len(props) - 1
            for sp, dv in model.reactions[k].stoich.items():
                x[model.species.index(sp)] += dv
        # else: thinned (extra) event — no state change. Either way time advanced.

    return out


def simulate_batch(
    model: RefModel,
    t_out: np.ndarray,
    n_reps: int,
    seed: int,
    **kwargs,
) -> np.ndarray:
    """``n_reps`` Extrande trajectories; returns ``(n_reps, len(t_out), n_species)``."""
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(n_reps)]
    return np.stack([simulate(model, t_out, r, **kwargs) for r in rngs], axis=0)
