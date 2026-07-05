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
