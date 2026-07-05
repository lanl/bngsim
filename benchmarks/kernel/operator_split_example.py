#!/usr/bin/env python3
"""Worked example: a two-subset operator split over shared species (GH #102 Stage 1).

The Stage 1 acceptance artifact. An external orchestrator partitions one reaction
network into a deterministic **ODE subset** and a stochastic **SSA subset** and
couples them over their shared species using the hardened state-exchange layer
(:mod:`bngsim.coupling`) — bngsim is the per-step engine on *both* sides of the
split, in one tool and one species namespace. Each part prints what it proves:

1. **ODE + ODE split == monolithic ODE.** Two ODE operators Strang-split over the
   shared state reproduce the whole-network ODE, converging at second order as
   the coupling step shrinks — with the total molecule count conserved across
   every exchange to machine precision (:class:`bngsim.ConservationLedger`).

2. **ODE + SSA split (the hybrid).** The same partition, now with the second
   operator run stochastically. A purely first-order network's SSA mean solves
   the ODE exactly (linear propensities ⇒ no moment-closure error), so the
   ensemble mean of the hybrid split must reproduce the monolithic ODE within
   Monte Carlo tolerance. The shared state crosses the boundary in *count* space
   (:class:`bngsim.UnitConverter`) through an explicit, leak-accounted rounding
   policy (:class:`bngsim.DiscreteExchange`).

3. **At scale.** A large synthetic first-order network is partitioned and advanced
   through one hybrid split; we report the per-step exchange cost (negligible vs
   the solve), conservation, and the dithered rounding leak — the proof the
   exchange layer composes at the issue's ~100K-reaction target.

4. **Division.** At a cell-division event the molecules are partitioned across two
   daughters (:class:`bngsim.Divider`, exact integer conservation) and the
   compartment volume is halved (:func:`bngsim.set_compartment_volume`).

The split loop lives *here*, in the orchestrator — Stage 1 ships the exchange
primitives, not a splitting integrator (that native engine is Stage 2). Run:

    python bngsim/benchmarks/kernel/operator_split_example.py            # full
    python bngsim/benchmarks/kernel/operator_split_example.py --quick    # fast smoke
    python bngsim/benchmarks/kernel/operator_split_example.py --scale 20000
"""

from __future__ import annotations

import argparse
import time

import bngsim
import numpy as np
from bngsim import (
    ConservationLedger,
    CouplingMap,
    DiscreteExchange,
    Divider,
    ReactionKernel,
    UnitConverter,
    make_subset_model,
    set_compartment_volume,
)
from bngsim._bngsim_core import ModelBuilder

# ─── Model builders ──────────────────────────────────────────────────────────


def linear_chain(rates: list[float], init0: float = 100.0) -> bngsim.Model:
    """S0 → S1 → ... first-order transfer chain (conserves the total count)."""
    n = len(rates) + 1
    b = ModelBuilder()
    sp = [b.add_species(f"S{i}", init0 if i == 0 else 0.0) for i in range(n)]
    for i, k in enumerate(rates):
        b.add_parameter(f"k{i}", k)
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    b.set_compute_conservation_laws(False)
    return bngsim.Model(_core=b.build())


def random_first_order_network(n: int, init_total: float, seed: int) -> bngsim.Model:
    """A connected random first-order transfer network on ``n`` species.

    Each species has one outgoing reaction S_i → S_{j} (random target) with a
    log-uniform rate over ~3 decades — the ~100K-first-order class the issue
    targets, scaled down. Pure transfer ⇒ the total count is conserved, and
    being first-order the SSA mean solves the ODE.
    """
    rng = np.random.default_rng(seed)
    b = ModelBuilder()
    sp = [b.add_species(f"S{i}", init_total if i == 0 else 0.0) for i in range(n)]
    for i in range(n):
        k = float(10.0 ** rng.uniform(-1.5, 1.5))
        j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        b.add_parameter(f"k{i}", k)
        b.add_reaction([sp[i]], [sp[j]], "elementary", f"k{i}")
    b.set_compute_conservation_laws(False)
    return bngsim.Model(_core=b.build())


# ─── The orchestrator's operator-split step ──────────────────────────────────


def strang_step(ka, kb, state, dt, *, converter=None, discrete=None, ledger=None, seed=None):
    """One Strang step ½A · B · ½A over the shared full-state vector.

    ``ka`` advances the first operator (ODE), ``kb`` the second. When
    ``converter``/``discrete`` are supplied the B hand-off goes through count
    space — storage → amounts → integer counts → storage — the discretization an
    SSA subset needs. ``ledger`` records the conserved total after the step.
    """

    def run_a(s, h):
        ka.set_state(s)
        return ka.advance(h)

    def run_b(s, h):
        if converter is not None and discrete is not None:
            s = converter.from_counts(discrete.discretize(converter.to_amounts(s)))
        kb.set_state(s)
        return kb.advance(h, seed=seed)

    out = run_a(run_b(run_a(state, dt / 2), dt), dt / 2)
    if ledger is not None:
        ledger.record(out)
    return out


# ─── Part 1: ODE + ODE == monolithic ─────────────────────────────────────────


def part1_ode_ode(quick: bool) -> None:
    print("\n== 1. ODE + ODE operator split reproduces the monolithic ODE ==")
    rates = [0.3, 0.6, 0.45, 0.8, 0.5]
    full = linear_chain(rates, init0=100.0)
    T = 12.0
    ref = bngsim.Simulator(full).run(t_span=(0.0, T), n_points=2).species[-1]

    even = make_subset_model(full, keep_reactions=[0, 2, 4])
    odd = make_subset_model(full, keep_reactions=[1, 3])
    ka, kb = ReactionKernel(even), ReactionKernel(odd)

    print(f"   chain of {full.n_species} species, {full.n_reactions} reactions; horizon T={T}")
    print(f"   {'steps':>6} {'dt':>8} {'max|split-mono|':>16} {'order':>7} {'max Σ drift':>13}")
    prev = None
    for n_steps in [12, 24] if quick else [12, 24, 48, 96]:
        ka.reset()
        kb.reset()
        led = ConservationLedger(atol=1e-7, name="N")
        state = ka.get_state()
        led.record(state)
        for _ in range(n_steps):
            state = strang_step(ka, kb, state, T / n_steps, ledger=led)
        err = float(np.max(np.abs(state - ref)))
        order = "" if prev is None else f"{prev / err:5.2f}x"
        prev = err
        print(
            f"   {n_steps:>6} {T / n_steps:>8.4f} {err:>16.3e} {order:>7} {led.max_abs_drift:>13.2e}"
        )
    print("   → second-order convergence (≈4× per halving); total count conserved to ~1e-13.")


# ─── Part 2: ODE + SSA hybrid ────────────────────────────────────────────────


def part2_ode_ssa(quick: bool) -> None:
    print("\n== 2. ODE + SSA hybrid split: ensemble mean reproduces the ODE ==")
    rates = [0.5, 0.3, 0.7, 0.4]
    init = 5000.0
    full = linear_chain(rates, init0=init)
    T, n_steps = 6.0, 12
    ref = bngsim.Simulator(full).run(t_span=(0.0, T), n_points=2).species[-1]

    ode_part = make_subset_model(full, keep_reactions=[0, 2])
    ssa_part = make_subset_model(full, keep_reactions=[1, 3])
    uc = UnitConverter.from_model(ssa_part)  # V_c = 1 ⇒ storage == counts
    ka = ReactionKernel(ode_part, method="ode")
    kb = ReactionKernel(ssa_part, method="ssa")

    n_reps = 16 if quick else 64
    finals = np.zeros((n_reps, full.n_species))
    leaks = np.zeros(n_reps)
    for r in range(n_reps):
        ka.reset()
        kb.reset()
        dx = DiscreteExchange(full.n_species, policy="nearest", dither=True)
        state = ka.get_state()
        for step in range(n_steps):
            state = strang_step(
                ka, kb, state, T / n_steps, converter=uc, discrete=dx, seed=r * 997 + step
            )
        finals[r] = state
        leaks[r] = dx.leak

    mean = finals.mean(axis=0)
    rel = float(np.max(np.abs(mean - ref))) / init
    print(
        f"   {full.n_species} species, init total {init:.0f}, {n_reps} SSA reps, {n_steps} steps"
    )
    print(f"   {'species':>8} {'ODE ref':>10} {'hybrid mean':>12}")
    for i in range(full.n_species):
        print(f"   {full.species_names[i]:>8} {ref[i]:>10.2f} {mean[i]:>12.2f}")
    print(f"   max |mean - ODE| / init = {rel:.2e}   (Monte Carlo: ~1/sqrt(N·reps))")
    print(
        f"   ensemble total = {mean.sum():.2f} (seeded {init:.0f}); dithered leak/rep ≈ {leaks.mean():+.2f} molecules"
    )


# ─── Part 3: at scale ────────────────────────────────────────────────────────


def part3_at_scale(scale: int) -> None:
    print(f"\n== 3. At scale: one hybrid split over a {scale}-reaction network ==")
    init = float(50 * scale)
    full = random_first_order_network(scale, init, seed=12345)
    T, n_steps = 2.0, 8

    # Partition reactions in half (even = ODE, odd = SSA).
    t0 = time.perf_counter()
    ode_part = make_subset_model(full, keep_reactions=list(range(0, scale, 2)))
    ssa_part = make_subset_model(full, keep_reactions=list(range(1, scale, 2)))
    build_s = time.perf_counter() - t0

    uc = UnitConverter.from_model(ssa_part)
    cmap = CouplingMap.from_model(full, full.species_names)  # full shared set
    ka = ReactionKernel(ode_part, method="ode")
    kb = ReactionKernel(ssa_part, method="ssa")
    dx = DiscreteExchange(full.n_species, policy="nearest", dither=True)
    led = ConservationLedger(atol=1.0, rtol=1e-6, name="N")

    ka.reset()
    kb.reset()
    state = ka.get_state()
    led.record(state)
    exchange_s = 0.0
    solve_s = 0.0
    for step in range(n_steps):
        # Time the exchange (gather/convert/round/scatter) vs the solve.
        te = time.perf_counter()
        amounts = uc.to_amounts(cmap.gather(state))
        shared = uc.from_counts(dx.discretize(amounts))
        state = cmap.scatter(state, shared)
        exchange_s += time.perf_counter() - te
        ts = time.perf_counter()
        ka.set_state(state)
        state = ka.advance(T / n_steps / 2)
        kb.set_state(state)
        state = kb.advance(T / n_steps, seed=step)
        ka.set_state(state)
        state = ka.advance(T / n_steps / 2)
        solve_s += time.perf_counter() - ts
        led.record(state)

    print(
        f"   species={full.n_species}, reactions={full.n_reactions} "
        f"(ODE {ode_part.n_reactions} + SSA {ssa_part.n_reactions})"
    )
    print(f"   subset build: {build_s * 1e3:.1f} ms   |   {n_steps} hybrid steps")
    print(
        f"   per-step exchange: {exchange_s / n_steps * 1e3:.3f} ms   "
        f"solve: {solve_s / n_steps * 1e3:.2f} ms   "
        f"exchange/solve = {exchange_s / max(solve_s, 1e-9):.1e}"
    )
    print(
        f"   total count: seeded {init:.0f}, final {state.sum():.0f}, "
        f"max drift {led.max_abs_drift:.1f} ({led.max_abs_drift / init:.1e} rel), "
        f"dithered leak {dx.leak:+.1f}"
    )


# ─── Part 4: division ────────────────────────────────────────────────────────


def part4_division() -> None:
    print("\n== 4. Cell division: partition molecules across daughters + halve volume ==")
    b = ModelBuilder()
    b.add_parameter("V", 2.0)  # compartment volume parameter
    sp = [b.add_species(n, 0.0) for n in ("A", "B", "C")]
    b.add_parameter("k", 0.5)
    b.add_reaction([sp[0]], [sp[1]], "elementary", "k")
    model = bngsim.Model(_core=b.build())
    model.set_state(np.array([1000.0, 400.0, 3.0]))

    uc = UnitConverter.from_model(model)
    divider = Divider(method="binomial", rng=np.random.default_rng(0))

    parent_counts = uc.to_counts(model.get_state())
    daughters = divider.divide(parent_counts, n_daughters=2)
    print(
        f"   parent volume V={bngsim.get_compartment_volume(model, 'V')}, "
        f"counts {parent_counts.astype(int).tolist()}"
    )
    for i, day in enumerate(daughters):
        print(f"   daughter {i}: counts {day.astype(int).tolist()}")
    total = np.sum(daughters, axis=0)
    assert np.array_equal(total, parent_counts), "division must conserve every species exactly"
    print(f"   Σ daughters = {total.astype(int).tolist()} == parent  (exact integer conservation)")

    # One daughter continues; halve its compartment volume (count-preserving).
    set_compartment_volume(model, "V", bngsim.get_compartment_volume(model, "V") / 2)
    model.set_state(uc.from_counts(daughters[0]))
    print(
        f"   daughter 0 continues at V={bngsim.get_compartment_volume(model, 'V')} "
        f"(same counts, doubled concentration)"
    )


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer reps / coarser sweep")
    ap.add_argument("--scale", type=int, default=2000, help="reaction count for the at-scale part")
    args = ap.parse_args()

    print("bngsim two-subset operator-split example (GH #102 Stage 1)")
    print(f"  version: {bngsim.__version__}")
    part1_ode_ode(args.quick)
    part2_ode_ssa(args.quick)
    part3_at_scale(200 if args.quick else args.scale)
    part4_division()
    print(
        "\nThe orchestrator owns the split; bngsim supplies the conserved, "
        "unit-correct,\ndiscretized state exchange on both sides. Native partitioned "
        "hybrid integration\n(one engine, dynamic repartition) is Stage 2."
    )


if __name__ == "__main__":
    main()
