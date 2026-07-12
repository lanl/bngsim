# Running simulations

## Batch parameter sweeps

```python
import numpy as np

# Sweep over parameter values (GIL released, thread-parallel)
param_sets = [{"kf": v} for v in np.linspace(0.1, 10, 100)]
results = sim.run_batch(
    t_span=(0, 100),
    n_points=101,
    params=param_sets,
    num_processors=4,       # parallel threads
)
# results is a list of 100 Result objects

# Or get a single Result with 3D arrays
batch = sim.run_batch(
    t_span=(0, 100), n_points=101,
    params=param_sets, squeeze=True,
)
print(batch.species.shape)  # (100, 101, n_species)
```

`run_batch` clones the model and resets every point to the `.net` seed initial
conditions — correct for a dose–response from a fresh start.

## Parameter scans that carry a pre-equilibrated state

A `run_batch` re-syncs each point to the seed, which is wrong for a multi-phase
protocol: pre-equilibrate, intervene (wash / bolus), then scan from the
*post-intervention* state. `parameter_scan` matches BNG2.pl's `reset_conc`
semantics — each point resets to the state **at scan invocation** (or to a named
snapshot), not the seed:

```python
# Phase 1 — pre-incubate off the seed, then snapshot the post-wash state.
sim.run_until(t=7200)                    # load receptors
sim.model.set_concentration("L(r,label~hot)", 0.0)   # wash out free hot ligand
sim.save_concentrations("start_competition")

# Phase 2 — dose-response that resets each point to the saved snapshot.
results = sim.parameter_scan(
    "cold_conc",
    par_scan_vals=[2.3e-10, 6.6e-9, 3.9e-8, 4.9e-7],
    t_span=(0, 1200), n_points=13,
    reset_conc=True,                     # BNG reset_conc=>1
    reset_to="start_competition",        # reset each point to the snapshot
    # Apply a coupled setConcentration that tracks the scanned value:
    on_point=lambda model, v: model.set_concentration(
        "L(r,label~cold)", v * NA * Vecf
    ),
)
# Each Result carries custom_attrs["scan_value"]; the bound-hot readout starts
# near the pre-wash amount and falls as cold competes it off.

# bifurcate() is the continuation sibling (reset_conc=>0): each point continues
# from the previous point's end-state — sweep up then down to trace hysteresis.
branch = sim.bifurcate("stimulus", par_min=0, par_max=10, n_scan_pts=50,
                       t_span=(0, 500), n_points=2)
```

## Interactive simulation

```python
# Run to a time point, inspect/modify, continue
sim.run_until(t=50)

# Knock out a reaction mid-simulation
sim.intervene({"kf": 0.0})

# Continue from current state
result = sim.run_until(t=100)

# Save and restore snapshots
snap = sim.snapshot()
sim.run_until(t=200)
sim.restore(snap)  # back to t=50
```

## Species manipulation

```python
# Save current concentrations as new initial state
model.save_concentrations()

# Perturb a single species (e.g., bolus addition)
model.set_concentration("A(b)", 500.0)

# Check concentration
print(model.get_concentration("A(b)"))

# Reset restores to save_concentrations() snapshot
model.reset()

# Named states coexist — save two, restore either (BNG saveConcentrations("name"))
model.save_concentrations("t=0")
# ... advance / intervene ...
model.save_concentrations("start_competition")
model.restore_concentrations("t=0")            # back to the first named state
model.restore_concentrations("start_competition")
```

## Stop conditions

```python
# Expression-based: stop when observable drops below threshold
sim.add_stop_condition("A_total < 10", label="low_A")

# Callable-based: arbitrary Python logic
sim.add_stop_condition(
    lambda r: r.species[-1, 0] < 5.0,
    label="very_low_A",
)

try:
    result = sim.run(t_span=(0, 1000), n_points=1001)
except bngsim.StopConditionMet as e:
    print(f"Stopped at t={e.result.time[-1]}: {e.condition}")
    partial = e.result  # truncated result up to stop point
```
