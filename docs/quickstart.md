# Quickstart



## Load a model and simulate

```python
import bngsim

# Load a BioNetGen .net file
model = bngsim.Model.from_net("model.net")

# Create an ODE simulator
sim = bngsim.Simulator(model, method="ode")

# Run simulation
result = sim.run(t_span=(0, 100), n_points=101)

# Access results as NumPy arrays
print(result.time.shape)          # (101,)
print(result.species.shape)       # (101, n_species)
print(result.observables.shape)   # (101, n_observables)
```

## Parameter sweeps

```python
# Update parameters in-memory (no file I/O)
model.set_param("kf", 0.5)
model.set_param("kr", 0.1)
result2 = sim.run(t_span=(0, 100), n_points=101)

# Reset species to initial conditions
model.reset()

# Set multiple parameters at once
model.set_params({"kf": 1.0, "kr": 0.2, "kcat": 5.0})
```

## Stochastic simulation (SSA)

```python
ssa = bngsim.Simulator(model, method="ssa")
result = ssa.run(t_span=(0, 100), n_points=101, seed=42)
```

## Stochastic simulation (PSA)

PSA (partial-scaling approximation) accelerates large-population runs by
leaping reaction firings whose minimum reactant population exceeds
`poplevel = N_c`, while preserving Gillespie statistics. The dispatch and
the `validate_for_ssa` gate are shared with SSA, so the same call shape
works for `.net`, SBML, and Antimony models:

```python
import bngsim

# .net
net_model = bngsim.Model.from_net("psa/tcr_signaling.net")
psa = bngsim.Simulator(net_model, method="psa", poplevel=100.0)
result = psa.run(t_span=(0, 300), n_points=301, seed=42)

# SBML / Antimony — same dispatch
sbml_model = bngsim.Model.from_sbml("ssa_sbml/tcr_signaling.xml")
psa = bngsim.Simulator(sbml_model, method="psa", poplevel=100.0)
result = psa.run(t_span=(0, 300), n_points=301, seed=42)
```

Choose `N_c` per [Lin, Feng, Hlavacek (2019)](https://doi.org/10.1063/1.5095032);
accuracy degrades when `N_c` is too small relative to per-species populations.
SBML/Antimony models with kineticLaw shapes that SSA cannot handle (e.g.
`reversible_non_mass_action`, `non_integer_stoichiometry`) raise
`SsaValidationError` at construct time.

## Seed semantics for stochastic methods

The `seed=` keyword on every stochastic entry point follows one contract:

- `seed=None` (the default, or omitting `seed=`): bngsim draws a fresh
  31-bit seed from system entropy on each call. Two consecutive calls
  produce independent trajectories.
- `seed=N` (any integer): bngsim passes `N` down to the backend
  verbatim. A fresh `Simulator` (or `NfsimSession` / `RuleMonkeySession`)
  initialized with the same `seed=N` reproduces the same trajectory.

`Result.seed` exposes the integer that was actually used, so
caller-generated and bngsim-drawn seeds are equally inspectable:

```python
import bngsim

sim = bngsim.Simulator(model, method="ssa")

# Default — fresh seed each call:
r1 = sim.run(t_span=(0, 100), n_points=101)
r2 = sim.run(t_span=(0, 100), n_points=101)
assert r1.seed != r2.seed                        # seeds differ

# Explicit, reproducible:
sim_a = bngsim.Simulator(model.clone(), method="ssa")
sim_b = bngsim.Simulator(model.clone(), method="ssa")
ra = sim_a.run(t_span=(0, 100), n_points=101, seed=42)
rb = sim_b.run(t_span=(0, 100), n_points=101, seed=42)
assert ra.seed == rb.seed == 42                  # exact reproduction
```

The same contract applies to `Simulator.run_batch`,
`NfsimSession.initialize`, and `RuleMonkeySession.initialize`. For
`run_batch`, the resolved value is the **base** seed; per-sim seeds are
`base_seed + i` and stamped onto each `Result.seed`.

`Result.seed` is `None` for ODE results and round-trips through HDF5
`Result.save()` / `Result.load()`. For sessions, the seed used by
`initialize()` is exposed as `session.seed` and stamped onto every
`Result` returned by `simulate()`.

**Reproducibility unit.** The reproducibility contract is:

> **same starting model state + same `seed=N` → same trajectory**

The Simulator object itself is not the unit. The C++ SSA/PSA backends
construct a fresh `std::mt19937_64` from the passed seed on every
`.run()` call, so passing the same explicit `seed=` always seeds the
RNG identically. What persists across `.run()` calls on the same
`Simulator` is the *model state* (species concentrations) — that's
what makes `simulate(...); simulate({continue=>1, ...})` multi-segment
protocols work. So:

```python
sim = bngsim.Simulator(model, method="ssa")

# First run: starts from model's initial state, seed=42.
r1 = sim.run(t_span=(0, 100), n_points=101, seed=42)

# Second run on the same sim with seed=42 *continues from r1's end
# state*, then advances with the seed=42 RNG stream. r2 is correctly a
# different trajectory because it starts from a different state — not
# because of any RNG issue.
r2 = sim.run(t_span=(0, 100), n_points=101, seed=42)
assert r2.species[0].tolist() == r1.species[-1].tolist()  # state continues

# To re-run the same trajectory: reset the model first.
model.reset()
r3 = sim.run(t_span=(0, 100), n_points=101, seed=42)
assert (r3.species == r1.species).all()  # exactly reproduces r1
```

Two fresh `Simulator` instances with the same seed also reproduce —
they just both start from the model's initial state, which is the
same as calling `model.reset()` between runs.
