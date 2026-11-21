---
summary: "Using BNGsim as PyBNF's simulator for parameter fitting: setup, cluster evaluation, and the capability checks a fitting tool should make."
---

# Use with PyBNF

When BNGsim is installed, [PyBNF](https://github.com/lanl/PyBNF) uses it to
fit BNGL models instead of starting `run_network` subprocesses:

```bash
pip install bngsim
cd tests/full_tests
pybnf -c T1-ssprop/polynomial.conf
```

No configuration is needed. PyBNF finds `bngsim` when it is imported.

## Scheduler-free cluster evaluation

A fit spreads thousands of independent evaluations over a cluster. BNGsim runs
one evaluation at a time and has no scheduler. The fitting tool (PyBNF) owns
multistart, bootstrap, profile likelihood and the Slurm or MPI fan-out. BNGsim
returns outputs and their sensitivities, the `(n_times, n_outputs, n_param)`
tensor, never a finished loss. The objective, noise model and normalization
stay in the fitting tool.

### Evaluations are independent

Every evaluation runs on its own model clone, with no shared mutable state. The
C++ engine has no globals, no file I/O and no stdout. A model and its clones
share one cache of the analysis of their functions, on the Python side; an entry
is read-only once made, and the cache is changed under a lock. `run_batch`
clones the model for each row, so many threads or processes can run at once
against the one read-only compiled `.so`.

For a fixed model, θ, sensitivity set and solver options, plus a `seed` for a
stochastic method, the result is deterministic. Batch rows come back in input
order, whatever the number of workers.

### One compiled artifact for every job

The codegen cache is keyed by the model's hash and written atomically (compiled
to a temporary file, then moved into place), so jobs racing on one model never
see a partial `.so`. Point `BNGSIM_CODEGEN_CACHE_DIR` at node-local scratch, or
fill it once on a login node and let the worker jobs reuse it:

```bash
# 1. Compile once on the login node into a shared cache.
export BNGSIM_CODEGEN_CACHE_DIR=/scratch/$USER/bngsim_cache
python -c "import bngsim; bngsim.Simulator(bngsim.Model.from_net('model.net'), codegen=True)"

# 2. Every worker job inherits the variable and reuses the cached .so.
sbatch --export=ALL fit_job.slurm
```

### A batch over a parameter matrix

A `Simulator` built with `sensitivity_params` gives each row's
output-sensitivity tensor from `run_batch`:

```python
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
rows = sim.run_batch(
    t_span=(0, 100), n_points=101,
    params=[{"kf": kf, "kr": kr} for kf, kr in theta_matrix],
    num_processors=8,           # independent clones, one shared .so
)
for r in rows:                  # in input order
    g = r.output_sensitivities(["observable:Atot"])   # (n_times, 1, n_param)
```

### Do not add the `parameter` axis to the `ic` axis

`output_sensitivities(axis="parameter")` is a total derivative. It includes
every way θ reaches the trajectory: the right-hand side and the initial
condition, `∂x(0)/∂θ` (a `.net` `R() R0` or `R() Rtot`, or an SBML
`<initialAssignment>` over constant parameters). `axis="ic"` is the derivative
with respect to the initial values themselves, `∂y(t)/∂x_k(0)`. The two
overlap:

```text
d_param[θ] = (right-hand-side path) + Σ_k (∂x_k(0)/∂θ)·d_ic[x_k]
```

So summing every column a fitted parameter reaches counts a seeded initial
condition twice. Add an `ic` term only for the part of `∂x(0)/∂θ` that BNGsim
does not already include. `Model.effective_ic_sensitivity()` tells you that
part, from the model alone, with no simulation, so you can set up the routing
once:

```python
# once, at fit setup; no simulation needed
seed = model.effective_ic_sensitivity(free_params)   # {'R(r)': {'R0': 1.0}}

def route(theta, d_param, d_ic):                     # per evaluation
    g = d_param[theta]                               # already a total derivative
    for species, coeff in my_own_ic_map.get(theta, {}).items():
        if theta not in seed.get(species, {}):       # not included...
            g = g + coeff * d_ic[species]            # ...so add it
    return g
```

Two things to get right:

- **Absent is not zero.** An entry of `0.0` means the initial condition is
  seeded and its coefficient is zero at this state. It may not be zero at the
  next point, so keep that column routed. Only an absent entry means there is
  no seeding, and only then add an `ic` term.
- **Names are the ones you wrote.** A compound `<initialAssignment>` is stored
  on an internal `_ic_<species>` parameter, but the report names the original
  symbols, as the sensitivity tensor does.

The double count is easy to miss. When a parameter appears only in an initial
condition, with coefficient 1, its `parameter` column equals the species' `ic`
column exactly, so the sum is exactly twice the right value. Comparing the two
columns does not catch it in general: they differ once the coefficient is not 1
or the parameter also appears in the rate laws.

`Result.ic_sensitivity_seed` records the same matrix for one run. Use the
`Model` call for routing. Use the `Result` one to see what a particular run
used. For a batch or scan over a nonlinear derived initial condition
(`Rtot = R0*scale`), the coefficients change from point to point, and the
`Result` one is the only correct answer.

The other way is to supply the rows yourself:
`Model.declare_ic_sensitivity({species: {param: ∂x(0)/∂θ}})`, then read
`axis="parameter"` alone and never touch `axis="ic"`. A declared species' row
replaces the model's own, and parameters it does not name get
`∂x_k(0)/∂θ = 0`. This matches how a PEtab condition table overrides an
`<initialAssignment>`. The cost is that declarations are model state, applied
per condition and sent to the workers.

Check `bngsim.capabilities()["features"]["effective_ic_sensitivity"]`, not a
version string. On a build without it, the seed matrix cannot be known, and a
gradient fit should be refused.

None of this applies to a `SteadyStateResult`. Its `parameter` axis is the
implicit-function derivative (`J·∂x*/∂p = −∂f/∂p`), with no initial-condition
term, and `axis="ic"` raises.

### Checkpoint and restart

`EvaluationSpec` is a frozen, JSON-serializable record of one evaluation: the
model source (with an optional SHA-256 check), θ, the time grid, the
sensitivity set, solver options, output selectors, and a `seed` for a
stochastic method. Send it to a worker or write it to a checkpoint;
`evaluate()` rebuilds the simulator and runs it.

A `method="ssa"` spec without a `seed` draws a fresh one on each `evaluate()`,
as `Simulator.run` does, and reports it on `Result.seed`. So
`dataclasses.replace(spec, seed=result.seed)` reruns that trajectory. SED-ML
carries the seed as `KISAO:0000488`.

Pair a spec with `Result.summary()` for a small index entry, without reading
back the full HDF5 file:

```python
spec = bngsim.EvaluationSpec(
    model_source="model.net", model_format="net",
    t_span=(0, 100), n_points=101,
    params={"kf": 0.5}, sensitivity_params=("kf",),
    outputs=("observable:Atot",),
)
blob = spec.to_json()                              # checkpoint, or send to a worker
result = bngsim.EvaluationSpec.from_json(blob).evaluate()
json.dump(result.summary(), open("eval_0001.json", "w"))   # small index entry
result.save("eval_0001.h5")                        # full arrays (HDF5)
```

`spec.with_params(theta_row)` puts a θ row on a base spec, so a sweep stores one
spec and a matrix instead of thousands of near-copies.

## Gate on what the build computes, not on its version

Before a long gradient fit, a fitting tool should check that this install
computes the derivatives it will trust. `capabilities()["features"]` answers
this for compiled backends and optional packages. It also has behavior keys for
derivative behaviors that a version string or `hasattr` cannot reveal:

```python
caps = bngsim.capabilities()
caps["features"]["event_sensitivities"]              # across a discrete event
caps["features"]["cross_compartment_sensitivities"]  # analytic ∂f/∂p, not difference quotients
caps["features"]["per_species_atol"]                 # run(atol=[...])
caps["features"]["tracking_atol"]                    # run(atol=TrackingAtol())
```

Each key is `True` or `False` on every build, so `False` is an answer you can
act on. A version string does not tell you: BNGsim raises `__version__` at the
start of a release cycle, so a source install made mid-cycle reports the same
version as the release. And `hasattr` does not tell you, because these keys
describe what a build computes, not what it exposes.

They matter because a build without one of these does not refuse the case. It
returns a finite tensor with a term missing, and a fit on it converges to the
wrong answer with no sign of trouble.

`caps["build"]` says which build answered:

```python
caps["build"]["commit"]   # what the compiled extension was built from
caps["build"]["stale"]    # True: that extension is behind the C++ beside it
```

Log the commit when a job starts. It is the only thing that tells apart two
installs with the same `version`. Check `stale` too. In a source checkout, the
compiled extension is built separately and does not rebuild on import, so it can
lag the Python code while every version and feature check passes. BNGsim warns
about this at import, but that may be before your package has set up logging.
Reading the key lets you raise it again where someone will see it.

## Ask each model whether its gradient is analytic

Whether a gradient uses BNGsim's analytic `∂f/∂p` or CVODES' internal
difference quotient is decided per model, when its code is generated. Two
models on the same install can get different answers.

CVODES takes one sensitivity right-hand side for all columns, so one rate law
BNGsim cannot differentiate turns off the analytic derivative for the whole
model. The difference quotient costs an extra right-hand-side evaluation per
column per step, so an N-parameter fit pays about N times the sensitivity cost.
On a long fit, that can be the difference between finishing and timing out.

Ask the Simulator, since the answer is per model:

```python
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
sim.has_analytic_sens_rhs     # True: the analytic ∂f/∂p. False: difference quotient.
sim.sens_rhs_decline_reason   # why, or None
```

Ask a Simulator built for a sensitivity run. A plain
`Simulator(model, method="ode")` compiles no analytic `∂f/∂p` because none was
asked for. It reports `False`, which is not a decline.

`has_analytic_sens_rhs` is read from the compiled artifact, so it is right
whatever the codegen cache did. The reason is worked out when the code is
generated and saved beside the artifact, so a cache hit replays it. A `None`
reason means either nothing to report or nothing recorded;
`has_analytic_sens_rhs` tells the two apart.

The reason tells you what to change. For example, "uses unsupported construct:
`abs()`" names the thing to rewrite. One case is worse than slow: a rate law
that switches at a crossing whose time depends on the parameters would get a
gradient that is wrong at and after the crossing, so BNGsim refuses that run.
