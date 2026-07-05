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
