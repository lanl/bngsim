# Reaction-kernel MVP demo (GH #102)

Driving bngsim **as a pluggable, codegen-fast reaction kernel** from an external
orchestrator — first and foremost a hand-rolled hybrid SSA/ODE splitting loop
that today glues a stochastic engine to AMICI-as-an-ODE-solver. The orchestrator
owns the split; bngsim is the per-step engine on either side.

`mvp_kernel_demo.py` is the **go/no-go on scale** demo. It drives bngsim per-step
through the bulk state-vector exchange API on a synthetic ~100K first-order
network and measures setup cost, bulk-state throughput, and the
marshalling-vs-solve ratio.

```bash
python bngsim/benchmarks/kernel/mvp_kernel_demo.py          # default sweep, to 100K
python bngsim/benchmarks/kernel/mvp_kernel_demo.py --quick  # fast smoke
python bngsim/benchmarks/kernel/mvp_kernel_demo.py --force-dense  # dense vs KLU
```

`--force-dense` runs the ODE sweep with CVODE's dense linear solver instead of
the auto-selected sparse KLU (via `Simulator(..., force_dense_linear_solver=True)`),
for a same-model dense-vs-sparse comparison. It caps the sweep below ~4K species,
since the dense path is ~O(n^2.5+)/step.

The demo auto-detects whether the build links a sparse (KLU) linear solver and
adapts its ODE sweep; it prints the KLU status at the top.

## What landed (this MVP)

- **Bulk state-vector API** (`Model.get_state()` / `Model.set_state(arr)`, plus
  `Simulator` delegators) — the per-step exchange primitive. One Python call
  marshals the entire live concentration vector to/from a contiguous `float64`
  numpy array, ordered like `species_names()`. O(n_species), no per-name hash
  lookups. This is the hard prerequisite from the issue.
- **`ModelBuilder.set_compute_conservation_laws(False)`** — opt out of
  conservation-law detection, which is dense O(n_species³) Gaussian elimination
  consumed only by the steady-state solver. ODE-only large models skip it so
  build stays O(reactions).
- **Sparse Jacobian-sparsity construction** — `build_jac_sparsity` no longer
  allocates a dense n×n bool matrix (O(n²) memory/time); it builds per-column
  adjacency and emits byte-identical CSC.
- **Sparse linear solver enabled** — the build now links SuiteSparse/KLU, so
  CVODE uses a sparse direct solve for large, low-density Jacobians instead of a
  dense n×n factorization. This is the unlock that makes ODE integration reach
  100K (see below).

## Findings: GO at 100K

Measured on the synthetic random linear network (one Intel Mac, KLU enabled):

| N (species) | build/setup | full-state get / set | solve / step | marshal / solve |
|------------:|------------:|---------------------:|-------------:|----------------:|
| 1,000       | ~19 ms      | ~2 / ~2 µs           | ~2 ms        | ~5e-3           |
| 10,000      | ~71 ms      | ~16 / ~18 µs         | ~27 ms       | ~2e-3           |
| 50,000      | ~418 ms     | ~77 / ~106 µs        | ~189 ms      | ~2e-3           |
| 100,000     | ~912 ms     | ~196 / ~278 µs       | ~380 ms      | ~2e-3           |

- ✅ **The 100K-reaction network advances per-step with overhead dominated by
  the solve.** Per-step marshalling is ~0.2% of the solve and the ratio is flat
  in N. The state-marshalling binding risk the issue flags at 100K is resolved.
- ✅ **Bulk state-exchange is O(n) and sub-millisecond at 100K** (~0.25 ms for
  the full 0.8 MB vector, one Python call each way).
- ✅ **Setup/build reaches 100K in well under a second** (~0.9 s) — the
  low-setup-overhead advantage over AMICI the issue rests on.
- ✅ **Round-trip equality**: a step-wise kernel loop (`get_state` →
  `run_until` → `set_state`) reproduces a standalone `run()` to ~1e-7 relative.
- ✅ The sparse solve scales near-linearly (≈13× wall for 10× the species from
  10K→100K), because the first-order Jacobian is ~2 nonzeros/column (density
  2e-5 at 100K).

## The linear solver (KLU) and the dense fallback

ODE *integration* at scale needs a sparse linear solver. With KLU the per-step
solve scales near-linearly (above). **Without** KLU, CVODE falls back to a dense
solver whose Jacobian/factorization is O(n²) memory and ~O(n^2.5+) time per step
(measured: 10 ms at N=500 → ~2.1 s at N=4000), capping integrable N at a few
thousand — independent of the marshalling cost and the build-path fixes above.

KLU is **on by default** in the build (`BNGSIM_ENABLE_KLU=ON`); it auto-disables
only when SuiteSparse is not installed, degrading to dense with a configure-time
warning. To enable it:

```bash
brew install suite-sparse          # provides include/suitesparse/klu.h
# then reconfigure + rebuild so the KLU discovery re-runs:
pip install -e . --no-build-isolation
```

Verify with `bngsim._bngsim_core` — `ModelBuilder().build().codegen_jacobian_plan()["has_klu"]`
is `True` when the sparse path is available. Note that even with KLU compiled in,
bngsim still auto-selects the *dense* solver for small / high-density models
(`SPARSE_THRESHOLD`, `SPARSE_DENSITY_MAX` in `cvode_simulator.cpp`), so the dense
path is always exercised where it wins.

## Synthetic model

Random linear (first-order) network: each species `S_i` has one outgoing
reaction `S_i -> S_{j}` (random target `j`) with a log-uniform rate over ~3
decades, so N species ≈ N reactions. A real ~100K model is unavailable; this is
the class the driving use case targets and is reproducible from a seed.

## Stage 0: the kernel adapter (`ReactionKernel` + warm path + Vivarium)

The MVP proved scale on the raw bulk-state API. Stage 0 turns that into a small,
hardened, framework-agnostic facade plus the warm hot path underneath it.

### `bngsim.ReactionKernel` — the direct-call API

A thin object wrapping a `Model` + `Simulator` that an external orchestrator
drives per step:

```python
import bngsim
kernel = bngsim.ReactionKernel(model, method="ode")   # or "ssa" / "psa"
state = kernel.get_state()        # bulk O(n) pull of the coupling species
state[i] += influx                # the orchestrator's contribution
kernel.set_state(state)           # inject it back
state = kernel.advance(dt)        # integrate one coupling step; returns new state
kernel.observables()              # derived readouts at the current state
```

`advance(dt)` is method-agnostic over the stateful backends (`ode` / `ssa` /
`psa`); the network-free XML backends (`nfsim` / `rulemonkey`) are wrappable for
introspection but raise on `advance` (no stateful per-step API).
`from_simulator()` adopts an already-configured `Simulator` (custom tolerances,
codegen `.so`, …). The kernel never touches `snapshot`/`restore` on the hot path.

**Worked example:** `kernel_example.py` — a runnable tutorial (acceptance
invariant, the per-step drive loop, observable readout):

```bash
python bngsim/benchmarks/kernel/kernel_example.py
```

### Warm CVODE hot path

`CvodeSimulator::run()` used to rebuild all SUNDIALS state every call —
SUNContext, the N_Vector, CVODE memory, and (most expensively) the KLU sparse
solver with a fresh symbolic factorization. The warm path keeps those objects
alive on the simulator and re-enters via `CVodeReInit`, which reuses the
allocations and keeps the linear solver attached, so KLU does **not** redo its
symbolic factorization (only a cheap numeric refactor runs per step). It is
transparent: every ODE `run`/`run_until` with no events, no discontinuity-trigger
roots (GH #72), no forward sensitivities, and a non-JAX Jacobian takes it
automatically; everything else takes the unchanged cold path. The warm path is
byte-identical to the cold rebuild — the saved per-call re-init cost grows with N
(skipped symbolic factorization) and, being fixed per call, dominates as the
coupling `dt` shrinks (the many-small-steps regime).

```bash
python bngsim/benchmarks/kernel/warm_path_bench.py          # parity + speedup sweep
python bngsim/benchmarks/kernel/warm_path_bench.py --quick
```

`BNGSIM_NO_WARM_CVODE=1` forces the old cold rebuild path (used by the bench to
measure the win, and as a safety valve).

### Optional Vivarium shell

`bngsim.vivarium.BngsimProcess` exposes the kernel as a vivarium-core `Process`
(`ports_schema` / `next_update` / `calculate_timestep`) so a bngsim network can
be operator-split against other processes over a shared species store. `species`
updates are returned as **deltas** (`accumulate` updater) so they compose
additively under splitting; `observables` are `set` each step. It is an optional
extra — `pip install 'bngsim[vivarium]'` — and the base package never imports it
(check `bngsim.HAS_VIVARIUM`).

### Acceptance

Advancing a model step-wise through the kernel reproduces a single standalone
`Simulator.run` over the same horizon, to integrator tolerance — pinned by
`python/tests/test_kernel.py::TestAdvanceRoundTrip` (parametrized over step
counts). The warm path's byte-identical contract is pinned by
`test_warm_cvode.py`, and the Vivarium shell's Engine-vs-kernel parity by
`test_vivarium.py`.

## Stage 1: hardened state exchange (`bngsim.coupling`)

Stage 0 drives a *single* network per step. Stage 1 is the **exchange layer** that
makes a *real two-subset hybrid split* correct and ergonomic — when an orchestrator
couples a deterministic ODE subset and a stochastic SSA subset over shared species,
the raw storage vectors crossing that boundary need units, addressing,
discretization, and conservation handled explicitly. All of it is Python over the
bulk-state primitive (no engine change); import from the package root:

```python
from bngsim import (
    UnitConverter, CouplingMap, DiscreteExchange, ConservationLedger,
    Divider, make_subset_model, set_compartment_volume,
)
```

- **`UnitConverter`** — bulk **count ↔ concentration ↔ amount** over the whole
  state vector via each species's `volume_factor` (`V_c`). `amount = storage·V_c`
  is **volume-invariant**, so subsets exchange in count/amount space and never need
  a live volume; the live-volume override is only on the *concentration* views
  (`to_concentrations`/`from_concentrations`), for a framework reporting mol/L into
  a compartment it is itself growing.
- **`CouplingMap`** — shared-species **name ↔ index** `gather`/`scatter`/`read`/
  `write`, so the orchestrator addresses only the coupling subset by name across
  two subsets ordered differently.
- **`DiscreteExchange` / `round_to_counts`** — an explicit, inspectable **rounding
  policy** at the SSA/NFsim hand-off (`nearest`/`floor`/`ceil`/`stochastic`) with
  **leak accounting**: a dithered carry feeds the fractional residual forward so
  repeated continuous→discrete round-trips don't shed mass, and `.leak` surfaces
  the silent drift that implicit SSA-entry rounding would hide.
- **`ConservationLedger` / `moiety_total`** — **no-leak checks** across the
  exchange boundary (baseline, drift, `assert_conserved`).
- **`Divider`** — molecule **partitioning at cell division** (binomial /
  multinomial / deterministic), exact integer conservation, plus
  `get_/set_compartment_volume` for framework volume-growth → compartment coupling.
- **`make_subset_model`** — the **ODE-subset-as-model** helper: reconstruct a
  reaction-subset model (`keep_reactions=`) with the other operator's species
  marked `fixed=True` (`fixed_species=`), so a *static* partition needs no native
  subset integration. Reconstructs the mass-action + functional class (the issue's
  ~100K first-order target); refuses events / table functions / MM / `amount_valued`
  rather than reconstruct them unfaithfully.

### Design resolutions (the two tensions the kickoff flagged)

- **(a) SSA `volume_factor` is load-time static.** The exchange currency is
  amount/count, which is volume-invariant, so the converters need *no* live-volume
  override there; the override lives only on the concentration views, for a
  framework that speaks mol/L. Tracking live volume inside the SSA *propensities*
  is Stage 2 — the exchange layer is correct in count space regardless.
- **(b) Where rounding lives.** In the exchange layer, as an explicit, leak-accounted
  `DiscreteExchange` — not relying on the SSA engine's implicit entry rounding, so
  the orchestrator controls *when* discretization happens and *sees* the residual.

### Worked example + acceptance

`operator_split_example.py` is the headline artifact — a two-subset operator split
(bngsim ODE subset + bngsim SSA subset over shared species) reproducing a reference
trajectory within tolerance, at scale:

```bash
python bngsim/benchmarks/kernel/operator_split_example.py            # full
python bngsim/benchmarks/kernel/operator_split_example.py --quick    # fast smoke
python bngsim/benchmarks/kernel/operator_split_example.py --scale 20000
```

It shows (1) an **ODE+ODE** Strang split reproducing the monolithic ODE at second
order with the total conserved to ~1e-13; (2) the **ODE+SSA** hybrid whose ensemble
mean reproduces the monolithic ODE within Monte Carlo tolerance (a first-order
network's SSA mean solves the ODE exactly), exchanging in count space through the
leak-accounted rounding; (3) the exchange cost staying ~1% of the solve **at scale**;
and (4) a **division** that partitions counts across daughters and halves the volume.

The numerical acceptance is pinned by
`python/tests/test_coupling.py::TestOperatorSplit` (the ODE+ODE second-order
convergence + machine-precision conservation, and the ODE+SSA mean matching the ODE
to ~0.3 %), with `make_subset_model` validated by integrating bit-for-bit like the
model it reconstructs, the `Divider` by exact integer conservation, and the rounding
by its dithered leak bound. The orchestrator owns the split loop — Stage 1 ships the
exchange primitives, not a splitting integrator (the native partitioned hybrid engine
is Stage 2).
