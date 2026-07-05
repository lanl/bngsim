# Benchmarks & validation

BNGsim's SSA (Gillespie direct method with dependency graph + Fenwick tree) and
PSA (partial-scaling approximation) were benchmarked against `run_network` 3.0
(BioNetGen 2.9.3) across 10 SSA models and 3 PSA models (6 configurations).
BNGsim timing covers `sim.run()` only; run_network timing includes full subprocess
overhead. Protocol: 2 warmup + 5 timed runs, median reported.

## SSA results (BNGsim wins every model)

| Model | Species | Reactions | BNGsim | run_network | Speedup |
|-------|---------|-----------|--------|-------------|---------|
| gene_expression_hill | 2 | 4 | 0.0001 s | 0.046 s | **547×** |
| simple_system | 4 | 4 | 0.002 s | 0.045 s | **20×** |
| flagellar_motor | 4 | 4 | 0.0008 s | 0.045 s | **54×** |
| gene_expr_3stage | 6 | 6 | 0.50 s | 1.02 s | **2.1×** |
| oscillatory_system | 5 | 8 | 0.25 s | 0.34 s | **1.3×** |
| gene_expression | 10 | 14 | 0.004 s | 0.053 s | **15×** |
| tcr_signaling | 37 | 97 | 0.68 s | 1.69 s | **2.5×** |
| egfr_net | 356 | 3749 | 0.086 s | 0.44 s | **5.2×** |
| multisite_phos | 1026 | 7680 | 0.36 s | 1.75 s | **4.9×** |
| fceri_gamma | 3744 | 58276 | 2.24 s | 64.5 s | **28.8×** |

**Geometric mean speedup: 8.7×**

## PSA results (BNGsim wins every configuration)

| Model | Nc | BNGsim | run_network | Speedup |
|-------|----|--------|-------------|---------|
| tcr_signaling | 10 | 0.015 s | 0.098 s | **6.5×** |
| tcr_signaling | 100 | 0.039 s | 0.173 s | **4.4×** |
| prion_aggregation | 10 | 0.46 s | 1.33 s | **2.9×** |
| prion_aggregation | 100 | 0.45 s | 1.35 s | **3.0×** |
| erk_activation | 10 | 0.14 s | 6.38 s | **44.8×** |
| erk_activation | 100 | 1.70 s | 10.9 s | **6.4×** |

**Geometric mean PSA speedup: 6.8×**

BNGsim's advantage comes from: (1) zero subprocess overhead (in-process execution),
(2) O(log N) Fenwick tree reaction selection (vs O(N) linear scan in run_network),
and (3) pre-computed propensity and dependency-graph data — including the
per-reaction affected-set used after each fire — with zero heap allocation, sort,
or dedup in the SSA hot loop.
The advantage grows with network size — 28.8× on fceri_gamma (58K reactions).

## PSA on SBML / Antimony models

`Simulator(model, method="psa", poplevel=N_c)` accepts SBML or Antimony models
loaded via `Model.from_sbml(...)` / `Model.from_antimony(...)` — the same
dispatch as `.net`, sharing the `validate_for_ssa` gate at construct time
(`reversible_non_mass_action`, `non_integer_stoichiometry`, etc. raise
`SsaValidationError` before the simulator is built). PSA semantics for `N_c`
are identical to the `.net` path; choose per [Lin, Feng, Hlavacek
(2019)](https://doi.org/10.1063/1.5095032) — accuracy degrades when `N_c` is
too small relative to per-species populations (e.g. `erk_activation` requires
`N_c ≳ 100` to track the network reference).

PSA's leaping advantage carries fully through the SBML loader. On
`tcr_signaling` loaded via `Model.from_sbml(...)` (3 timed runs, median;
t_end=300, n_steps=1000), PSA beats SSA by **8.6× at N_c=300**, **15.3× at
N_c=100**, **34.8× at N_c=30**, and **46.7× at N_c=10**, matching the
SSA→PSA speedup ranges measured on the `.net` path.

## ODE parity vs RoadRunner (results ship in-repo)

bngsim's SBML ODE engine is cross-checked against
[libRoadRunner](https://github.com/sys-bio/roadrunner) over the **full BioModels ODE
corpus (1,323 models)** by the [`parity_checks/rr_parity`](https://github.com/lanl/bngsim/tree/HEAD/parity_checks/rr_parity)
suite: both engines integrate each model at a shared tolerance and their trajectories
are compared cell-by-cell. The latest released result is committed **in this
repository** (so it's there the moment you clone or install) as a gzipped report under
[`parity_checks/rr_parity/snapshots/`](https://github.com/lanl/bngsim/tree/HEAD/parity_checks/rr_parity/snapshots/) — see that
directory's README for the convention. Regenerate the interactive HTML matrix
(per-model verdicts, three-tier timing breakdown, BNGsim-vs-RoadRunner cost scatter,
and win-by-species tables) from a snapshot with:

```bash
cd parity_checks/rr_parity
gunzip -c snapshots/report_ode_<version>.json.gz > runs/report_ode.json
python generate_matrix.py            # writes runs/parity_matrix.html
```

As of the latest snapshot: **1,237/1,323 models agree** to ~1e-4 relative error; on
warm per-integration cost BNGsim is **~1.65× faster by geometric mean** and wins the
majority of models, with RoadRunner pulling ahead on the high-species tail (the
crossover is around 20–35 species).

## SBML semantic test suite — capability boundary and intentional deviations

Against the [SBML semantic test suite](https://github.com/sbmlteam/sbml-test-suite)
(v3.3.0, 1823 cases), graded through a faithful local port of the **official** test-runner
grading, bngsim scores **1577 `Match`** (1577/1789 in-scope time-course cases = 88.1%).
The full breakdown is 1577 `Match`, 242 `Unsupported` (the declared capability boundary
below), 3 `NoMatch` (the intentional deviation below), 1 `CannotSolve`, and **0 `Error`**
— no case is ever a wrong-but-plausible answer. In a shared-tolerance head-to-head bngsim
also passes more cases than RoadRunner (1578 vs 1529 under the in-repo fair harness). The
cases bngsim does **not** match are all deliberate — a declared capability boundary and
one intentional deviation, not open bugs.

That score is reproducible and reviewable: [`benchmarks/suites/sbml_test_suite/testrunner/`](https://github.com/lanl/bngsim/tree/HEAD/benchmarks/suites/sbml_test_suite/testrunner/)
ships a fail-closed wrapper for the official runner, a **committed unsupported-tags
manifest** (`bngsim-unsupported-tags.txt`, pinned by a unit test to the shipped source of
truth [`bngsim._sbml_unsupported`](https://github.com/lanl/bngsim/blob/HEAD/python/bngsim/_sbml_unsupported.py)), and `score.py` — a
line-for-line port of the runner's comparison, outcome enum, and tag matching. See that
directory's README for the runbook and the full per-category breakdown.

**Unsupported constructs (refused at load — never silently approximated).** bngsim is
an ODE/SSA kinetic engine with no differential-algebraic, delay, or fast-equilibrium
solver. Where such a construct appears in the equations that define the system, bngsim
fails closed with a `ModelError` naming it — rather than integrate a *different*
mathematical system and return a confident-but-wrong trajectory (opt into the legacy
silent approximation with `BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`):

| Construct (tag) | Why unsupported | Trivial form still simulated |
|-----------------|-----------------|------------------------------|
| `AlgebraicRule` (non-empty) | DAE constraint — no algebraic/DAE solver | an empty `<algebraicRule/>` (no constraint) |
| `csymbol delay` / `delay(x, τ≠0)` | delay-differential equation — no DDE solver | `delay(x, 0)` = identity |
| `FastReaction` (`fast="true"`) under ODE | fast-equilibrium constraint — no constraint solver | — |

The suite's flux-balance test type (`fbc` / `FluxBalanceSteadyState`, 34 cases) is
likewise declared unsupported — it is a steady-state constraint problem, not a
time-course, and bngsim is a time-course ODE/SSA engine. All of the above are recorded in
the committed manifest, so the official grading reports them as `Unsupported` (an honest
capability boundary) rather than as failures.

This is a genuine capability boundary shared with RoadRunner, which declares the same
constructs unsupported in the SBML Test Suite Database. bngsim's declared-unsupported
set is *narrower*, though: it fully or partially supports several tags others declare
unsupported — e.g. `VolumeConcentrationRates` (time-varying compartment-volume dilution,
15/19 suite cases) and `AssignedVariableStoichiometry` (named / rate-ruled
`<speciesReference>` stoichiometry, 27/73 — ahead of RoadRunner's 20).

**Intentional deviations (simulated, but deliberately differ).** Three cases where
bngsim runs the model but does not match the suite's reference on purpose:

- **Avogadro constant (3 cases: 00960, 00961, 01323).** bngsim uses the SI-exact
  Avogadro number `Nₐ = 6.02214076e23` (fixed by the 2019 redefinition of the SI
  mole) everywhere the `avogadro` csymbol appears. The suite's analytic reference
  predates that redefinition and uses the older measured value, so bngsim differs in
  the last few digits. We match the current SI definition on purpose.

**Reproducible random event ordering.** When several events fire at the same instant
with equal (or unset) priority, SBML L3v2 §4.11.6 chooses their order at random. bngsim
implements this with a **per-run seed** (`SolverOptions.event_seed`, exposed as
`Simulator(method="ode").run(seed=…)`): equal-priority ties are broken by a
`std::mt19937_64`, so the `RandomEventExecution` suite family (00962, 01588, 01590,
01591, 01599, 01605, 01627 and the T0/no-delay variants) passes, while a *fixed default
seed* keeps every run reproducible out of the box — the hard requirement for the
parameter-fitting workloads bngsim is built for (PyBNF). The RNG is consumed **only** at
a genuine equal-priority tie, so any model without such simultaneity is byte-identical
regardless of the seed, and an event-free ODE run stays seed-less (`Result.seed is
None`). Pass an explicit `seed` for an independent event-ordering realization.

All other suite gaps are tracked as ordinary compliance work.
