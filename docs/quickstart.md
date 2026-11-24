---
summary: "Load a .net, SBML, Antimony or BNGL model, run it as ODEs, by SSA or by PSA, change parameters, and reproduce a stochastic run with a seed."
---

# Quickstart

## Load a model and simulate

The block below runs as written after `pip install bngsim`. It writes its own
model first, a two-species network in BioNetGen's `.net` format. With a `.net`
file of your own, start at `from_net`.

```python
from pathlib import Path

import bngsim

# A -> B at rate k, in BioNetGen's network (.net) format.
Path("model.net").write_text(
    """\
begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 A 1
    2 B 2
end groups
""",
    encoding="utf-8",
)

# Load a BioNetGen .net file
model = bngsim.Model.from_net("model.net")

# Create an ODE simulator
sim = bngsim.Simulator(model, method="ode")

# Run simulation
result = sim.run(t_span=(0, 10), n_points=101)

# Access results as NumPy arrays
print(result.time.shape)          # (101,)
print(result.species.shape)       # (101, 2): one column per species, in model.species_names order
print(result.observables.shape)   # (101, 2): one column per observable, "A" and "B"
print(round(float(result.observables["A"][-1]), 4))  # 0.6738, which is 100 * exp(-5)
```

## Other model formats

SBML, Antimony and BNGL load through the same `Model` class. The small models
in `bngsim.examples` come in each format, so these lines run as written; pass
your own file's path instead:

```python
sbml_model = bngsim.Model.from_sbml(bngsim.examples.path("binding", "sbml"))
ant_model = bngsim.Model.from_antimony(bngsim.examples.path("binding", "antimony"))  # pip install bngsim[antimony]
```

```python
bngl_model = bngsim.Model.from_bngl(bngsim.examples.path("binding", "bngl"))  # pip install bngsim[bngl]
```

BNGsim simulates reaction networks, and BNGL describes rules. So `from_bngl`
hands the model to `BNG2.pl generate_network` and loads the `.net` it writes.
That needs BNG2.pl: the `bngl` extra supplies it, or set `$BNGPATH` to your own
BioNetGen install. Generated networks are cached, so loading an unchanged model
again does not rerun BNG2.pl.

See [Loading models](user-guide/loading-models.md) for more.

## Parameter sweeps

```python
# Set a parameter in memory. A run goes on from the state the last one
# ended in; start="initial" runs this one from the initial conditions.
model.set_param("k", 1.0)
result2 = sim.run(t_span=(0, 100), n_points=101, start="initial")

# Reset species to initial conditions
model.reset()

# Set several parameters at once from a dict (this model has one, k)
model.set_params({"k": 0.25})
```

## Stochastic simulation (SSA)

```python
ssa = bngsim.Simulator(model, method="ssa")
result = ssa.run(t_span=(0, 100), n_points=101, seed=42)
```

## Stochastic simulation (PSA)

PSA, the partial-scaling approximation, speeds up runs with large populations.
It leaps reaction firings whose smallest reactant population is above
`poplevel`. It keeps the ensemble means but inflates the fluctuations of the
scaled channels. Use it for means. For variances, distributions or rare events,
use `method="ssa"`.

PSA takes the same models and checks as SSA, so the call is the same for
`.net`, SBML and Antimony models:

```python
import bngsim

# .net: birth and death, 0 -> A -> 0, which settles near 10,000 copies of A
net_model = bngsim.Model.from_net(bngsim.examples.path("birth_death", "net"))
psa = bngsim.Simulator(net_model, method="psa", poplevel=100.0)
result = psa.run(t_span=(0, 300), n_points=301, seed=42)

# SBML or Antimony: the same model
sbml_model = bngsim.Model.from_sbml(bngsim.examples.path("birth_death", "sbml"))
psa = bngsim.Simulator(sbml_model, method="psa", poplevel=100.0)
result = psa.run(t_span=(0, 300), n_points=301, seed=42)
```

Choose `poplevel` as in [Lin, Feng, Hlavacek (2019)](https://doi.org/10.1063/1.5096774).
Accuracy drops when it is small next to the species' populations. An SBML or
Antimony rate law that SSA cannot take (for example a reversible
non-mass-action law, or a non-integer stoichiometry) raises
`SsaValidationError` when the `Simulator` is built.

## Seeds for stochastic runs

Every stochastic call takes `seed=`:

- `seed=None`, the default: BNGsim draws a fresh 31-bit seed from system
  entropy on each call, so two calls give independent trajectories.
- `seed=N`, an integer from 0 to 2**64 - 1: BNGsim passes `N` to the engine
  as is. `run()` reseeds on every call, so the same seed, starting state and
  run arguments give the same trajectory.
  A seed outside that range raises `ValueError`.
- NFsim (`method="nf"`, `NfsimSession`) takes seeds from 0 to 2**32 - 1 only,
  and raises `ValueError` on a larger one. SSA, PSA and RuleMonkey use all 64
  bits.

`Result.seed` is the seed a run used, whether you gave it or BNGsim drew it:

```python
import bngsim

sim = bngsim.Simulator(model, method="ssa")

# Default: a fresh seed each call (each run from the initial conditions).
r1 = sim.run(t_span=(0, 100), n_points=101, start="initial")
r2 = sim.run(t_span=(0, 100), n_points=101, start="initial")
assert r1.seed != r2.seed                        # seeds differ

# Explicit, reproducible:
sim_a = bngsim.Simulator(model.clone(), method="ssa")
sim_b = bngsim.Simulator(model.clone(), method="ssa")
ra = sim_a.run(t_span=(0, 100), n_points=101, seed=42, start="initial")
rb = sim_b.run(t_span=(0, 100), n_points=101, seed=42, start="initial")
assert ra.seed == rb.seed == 42                  # exact reproduction
assert (ra.species == rb.species).all()
```

An `NfsimSession` or `RuleMonkeySession` takes its seed once, in
`initialize(seed)`. A second `simulate()` continues the random stream, so to
repeat a session's trajectory, call `initialize(seed)` again.

`run_batch`, `run_replicates` and `parameter_scan` take one base seed. Row 0
runs at the base seed itself, so `seed=rows[0].seed` reruns the whole call,
even when BNGsim drew the base. Rows 1, 2, … get distinct seeds derived from
the base with numpy's `SeedSequence(base_seed).generate_state(...)`, each
masked to 31 bits and none equal to the base. Each row's seed is on its
`Result.seed`, so `run(seed=r.seed)` repeats that row. Nearby base seeds give
unrelated rows.

`Result.seed` is `None` for an ODE run, and it survives `Result.save()` and
`Result.load()`. A session's seed is `session.seed`, and it is on every
`Result` its `simulate()` returns.

## Where a run starts

Reproducing a stochastic run takes the same starting state and the same seed.
The seed alone is not enough, because the state lives in the `Model`, and every
run leaves its end state there. That is what lets a run continue the last one,
as BioNetGen's `simulate({continue=>1})` does.

```python
sim = bngsim.Simulator(model, method="ssa")

# From the initial conditions, with seed 42. start="initial" resets the
# model first, whatever the blocks above left it holding.
r1 = sim.run(t_span=(0, 100), n_points=101, seed=42, start="initial")

# The same seed, but starting from r1's end state, so a different
# trajectory. start="live" says the continuation is meant; without it,
# run() warns with bngsim.LiveStartWarning here.
r2 = sim.run(t_span=(0, 100), n_points=101, seed=42, start="live")
assert r2.species[0].tolist() == r1.species[-1].tolist()  # state continues
assert (r1.started_from, r2.started_from) == ("initial", "live")

# To repeat r1, start from the initial state again (or call model.reset()).
r3 = sim.run(t_span=(0, 100), n_points=101, seed=42, start="initial")
assert (r3.species == r1.species).all()  # exactly reproduces r1
```

A second `Simulator` on the same model also starts where the last run ended.
To run two trajectories from the initial state, pass `start="initial"`, call
`model.reset()` between them, or give each its own `model.clone()` made before
the first run.

`Result.started_from` is `"initial"` or `"live"`, and `summary()` shows it.
`run()` warns with `bngsim.LiveStartWarning` before it runs when it would start
from exactly the state the last run left, at a time before that run ended:
running `t_span=(0, 5)` twice, for example. Continuing the clock
(`t_span=(5, 10)` next) or changing the state in between does not warn.
`run_until`, `run_replicates`, the scans and `run_protocol` warn the same way.
