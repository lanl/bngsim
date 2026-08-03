# Use with PyBNF

When `bngsim` is installed, [PyBNF](https://github.com/lanl/PyBNF) automatically
uses it for BNGL model fitting instead of spawning `run_network` subprocesses:

```bash
pip install bngsim
cd tests/full_tests
pybnf -c T1-ssprop/polynomial.conf
# T1 benchmark: 3:16 (was 3+ hours without bngsim)
```

No configuration changes needed — PyBNF auto-detects `bngsim` at import time.

## Scheduler-free cluster evaluation

A fitting run distributes thousands of independent evaluations across a cluster.
BNGsim is the **stateless single-evaluation kernel** for that workload — the
frontend (PyBNF) owns the scheduler: multistart, bootstrap, profile likelihood,
Slurm/MPI fan-out. BNGsim adds **no scheduler code**; it exposes the raw output +
sensitivity *primitives* (the `(n_times, n_outputs, n_param)` tensor), never a
pre-baked loss — objective/noise/normalization composition stays in the frontend.

**Statelessness / re-entrancy.** Every evaluation runs against an independent
model clone with no shared mutable state (the C++ engine is instance-based with no
globals, no file I/O, no stdout). `run_batch` clones the model per row, so many
threads — or many processes — evaluate concurrently against the **one** read-only
compiled `.so` without interfering. For a fixed `(model, θ, sensitivity set,
solver options)` the result is deterministic, and batch rows are returned in input
order regardless of worker count.

**Shared compiled artifact.** The codegen cache is content-addressed by model
hash and updated atomically (compile to a process-unique temp file, then
`os.replace` into the cache), so concurrent jobs racing on the same model never
observe a partial `.so`. Point `BNGSIM_CODEGEN_CACHE_DIR` at node-local scratch,
or pre-warm it once on a login node and have worker jobs reuse the artifact:

```bash
# 1. Compile once on the login node into a shared/staged cache.
export BNGSIM_CODEGEN_CACHE_DIR=/scratch/$USER/bngsim_cache
python -c "import bngsim; bngsim.Simulator(bngsim.Model.from_net('model.net'), codegen=True)"

# 2. Every worker job inherits the env var and reuses the cached .so (no compile).
sbatch --export=ALL fit_job.slurm
```

**Local batch over a parameter matrix.** A `sensitivity_params`-configured
`Simulator` yields the full per-row output-sensitivity tensor from `run_batch`,
reusing the one shared artifact:

```python
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
rows = sim.run_batch(
    t_span=(0, 100), n_points=101,
    params=[{"kf": kf, "kr": kr} for kf, kr in theta_matrix],
    num_processors=8,           # independent clones, one shared .so
)
for r in rows:                  # deterministic input order
    g = r.output_sensitivities(["observable:Atot"])   # (n_times, 1, n_param)
```

**The `parameter` axis is a total derivative — do not sum it with the `ic`
axis.** `output_sensitivities(axis="parameter")` carries every path by which θ
reaches the trajectory, the right-hand side *and* the initial-condition seeding
`∂x(0)/∂θ` (a `.net` `R() R0` / `R() Rtot`, or an SBML `<initialAssignment>`
over constant parameters). `axis="ic"` is the companion basis `∂y(t)/∂x_k(0)`
with the initial value held independent. The two overlap:

```text
d_param[θ] = (right-hand-side path) + Σ_k (∂x_k(0)/∂θ)·d_ic[x_k]
```

A gradient path that routes one fitted parameter to every native column it
reaches and **sums** them therefore double-counts any seeded initial condition.
The rule is not "drop one axis" — it is *add an `ic` term only for the part of
`∂x(0)/∂θ` bngsim does not already carry*. `Model.effective_ic_sensitivity()`
reports exactly that, from model structure alone, with **no integration**, so
routing can be built once at setup rather than per evaluation:

```python
# once, at fit setup — no simulation needed
seed = model.effective_ic_sensitivity(free_params)   # {'R(r)': {'R0': 1.0}}

def route(theta, d_param, d_ic):                     # per evaluation: tensor reads
    g = d_param[theta]                               # already a total derivative
    for species, coeff in my_own_ic_map.get(theta, {}).items():
        if theta not in seed.get(species, {}):       # absent ⇒ not carried...
            g = g + coeff * d_ic[species]            # ...so supply it
    return g
```

Two distinctions the accessor is careful about, because both have bitten
consumers:

- **Absent is not zero.** A *present* entry valued `0.0` means "seeded, and the
  coefficient is zero at this state" — a chain-rule factor that vanishes here
  but need not at the next point, so the column must stay routed. An *absent*
  entry means there is no seeding path at all. Only the second one is your
  signal to add an `ic` term.
- **Ids are the ones you wrote.** A compound `<initialAssignment>` is lowered
  internally onto a synthetic `_ic_<species>` parameter; the report always names
  the original symbols, matching how the sensitivity tensor is labelled.

The overlap is easy to miss because its commonest case is degenerate: a
parameter appearing *only* in an initial condition, with coefficient 1, gives a
`parameter` column **bit-identical** to the seeded species' `ic` column, so
summing reports that column at exactly 2×. Comparing the two columns is not a
usable test — it detects nothing once the coefficient differs from 1 (a compound
`<initialAssignment>`) or the parameter also drives the rate laws.

`Result.ic_sensitivity_seed` is the per-run record of the same matrix. Prefer
the `Model` accessor for routing; reach for the `Result` one to see what a
*particular* run used, which is the only correct answer for a batch or scan over
a nonlinear derived initial condition (`Rtot = R0*scale`), where the
coefficients move from point to point.

The alternative integration is to **push instead of subtract**: hand your own
rows to `Model.declare_ic_sensitivity({species: {param: ∂x(0)/∂θ}})` and read
`axis="parameter"` alone, never touching the `ic` axis. A declared species' row
is taken as fully specified — parameters not named get `∂x_k(0)/∂θ = 0` — so it
*replaces* the model's own row rather than adding to it, matching the semantics
a PEtab condition-table override already has against an `<initialAssignment>`.
The cost is that declarations are model *state*, applied per condition and
carried to workers, where reading is a pure mapping.

Gate on the capability rather than a version string:
`bngsim.capabilities()["features"]["effective_ic_sensitivity"]`. On a build
without it the seed matrix is not knowable, and every answer a consumer could
guess is silently wrong — refusing the gradient fit is the honest behaviour.

On `SteadyStateResult` none of this applies: that `parameter` axis is the
implicit-function derivative of the algebraic system (`J·∂x*/∂p = −∂f/∂p`), it
has no seeding term to double-count, and `axis="ic"` raises.

**Checkpoint / restart.** `EvaluationSpec` is a frozen, JSON-serializable record
of one evaluation — model source (+ optional SHA-256 integrity guard), θ vector,
time grid, sensitivity set, solver options, and output selectors. Ship it to a
worker or write it to a checkpoint; `evaluate()` reconstructs the simulator and
runs deterministically. Pair it with the compact `Result.summary()` for cheap
indexing/logging without re-reading every full HDF5 payload:

```python
spec = bngsim.EvaluationSpec(
    model_source="model.net", model_format="net",
    t_span=(0, 100), n_points=101,
    params={"kf": 0.5}, sensitivity_params=("kf",),
    outputs=("observable:Atot",),
)
blob = spec.to_json()                              # checkpoint / send to worker
result = bngsim.EvaluationSpec.from_json(blob).evaluate()
json.dump(result.summary(), open("eval_0001.json", "w"))   # compact index entry
result.save("eval_0001.h5")                        # full arrays (HDF5)
```

`spec.with_params(theta_row)` stamps a θ row onto a base spec, so a sweep
serializes one spec plus a matrix rather than thousands of near-duplicates.
