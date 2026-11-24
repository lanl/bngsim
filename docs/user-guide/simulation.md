---
summary: "Parameter sweeps with run_batch, parameter scans from a saved state, bifurcate for continuation, runs in steps, saved states, and stop conditions."
---

# Running simulations

This page covers the calls beyond a single `run()`: sweeps, scans, runs in
steps, saved states and stop conditions.

The examples share one model, a radioligand competition assay: a labeled
(hot) and an unlabeled (cold) form of a ligand `L` bind the same receptor `R`.
Amounts are molecules per cell, and the doses are molar concentrations in the
medium, converted with `NA` and the medium's volume per cell, `Vecf`:

```python
from pathlib import Path

import numpy as np

import bngsim

Path("model.net").write_text(
    """\
begin parameters
    1 NA        6.022e23      # molecules per mole
    2 Vecf      1e-9          # medium volume per cell, L
    3 kf        1e7           # binding, /M/s
    4 kr        1e-3          # unbinding, /s
    5 R0        5e4           # receptors per cell
    6 hot_conc  1e-9          # hot ligand, M
    7 cold_conc 0             # cold ligand, M
    8 kon       kf/(NA*Vecf)  # binding, per molecule per second
end parameters
begin species
    1 L(r,label~hot)           hot_conc*NA*Vecf
    2 L(r,label~cold)          cold_conc*NA*Vecf
    3 R(l)                     R0
    4 L(r!1,label~hot).R(l!1)  0
    5 L(r!1,label~cold).R(l!1) 0
end species
begin reactions
    1 1,3 4 kon
    2 4 1,3 kr
    3 2,3 5 kon
    4 5 2,3 kr
end reactions
begin groups
    1 Bound_hot  4
    2 Bound_cold 5
    3 R_free     3
end groups
""",
    encoding="utf-8",
)
model = bngsim.Model.load("model.net")
sim = bngsim.Simulator(model, method="ode")
```

## Batch parameter sweeps

`run_batch` runs one point per parameter set, in parallel threads:

```python
param_sets = [{"kf": v} for v in np.linspace(1e6, 1e8, 100)]
results = sim.run_batch(
    t_span=(0, 100),
    n_points=101,
    params=param_sets,
    num_processors=4,       # parallel threads
)
# results is a list of 100 Result objects

# Or one Result with 3D arrays
batch = sim.run_batch(
    t_span=(0, 100), n_points=101,
    params=param_sets, squeeze=True,
)
print(batch.species.shape)  # (100, 101, 5): points, time points, species
```

Watch out:

- `run_batch` clones the model and starts every point from the model's initial
  conditions. That is right for a dose-response from a fresh start. If the
  model is not at its initial conditions (after a `set_concentration`, or a
  run), it warns that the points do not start where the model is.
- `run_replicates` starts every replicate where the model is, as `run` does.

## Parameter scans from a saved state

Use `parameter_scan` when every point should start from a state you prepared:
equilibrate, intervene (a wash or a bolus), then scan. Like BNG2.pl's
`reset_conc`, each point resets to the state at the call, or to a named saved
state:

```python
# Phase 1: pre-incubate, wash, and save the state.
sim.run_until(t=7200)                    # load receptors
sim.model.set_concentration("L(r,label~hot)", 0.0)   # wash out free hot ligand
sim.save_concentrations("start_competition")

# Phase 2: a dose-response that starts each point from the saved state.
NA, Vecf = model.get_param("NA"), model.get_param("Vecf")
results = sim.parameter_scan(
    "cold_conc",
    par_scan_vals=[2.3e-10, 6.6e-9, 3.9e-8, 4.9e-7],
    t_span=(0, 1200), n_points=13,
    reset_conc=True,                     # BNG reset_conc=>1
    reset_to="start_competition",        # start each point from this state
    # Set a dose that follows the scanned value:
    on_point=lambda model, v: model.set_concentration(
        "L(r,label~cold)", v * NA * Vecf
    ),
)
# The hook does the dosing. A scan from a saved state cannot move a species
# through the parameter that sets its starting amount, and warns if you try.
# For a dose scan from the initial conditions, call model.reset() first.
# Each Result has custom_attrs["scan_value"].

# bifurcate() is the continuation scan (reset_conc=>0): each point starts
# from the previous point's end state. Sweep up, then down, to trace
# hysteresis: on the bundled Schlogl model, bistable for k3 in about 870-1130,
# the two sweeps end on different branches there.
switch = bngsim.Simulator(bngsim.examples.load("schlogl"))
up = switch.bifurcate("k3", par_min=700, par_max=1300, n_scan_pts=50,
                      t_span=(0, 500), n_points=2)
down = switch.bifurcate("k3", par_min=1300, par_max=700, n_scan_pts=50,
                        t_span=(0, 500), n_points=2)
```

## Running in steps

```python
# Start over: the model at its initial conditions, a new Simulator at t = 0.
# (A run_until target must lie past the Simulator's clock.)
model.reset()
sim = bngsim.Simulator(model, method="ode")

# Run to a time, inspect or change the model, continue
sim.run_until(t=50)

# Knock out a reaction mid-run
sim.intervene({"kf": 0.0})

# Continue from the current state
result = sim.run_until(t=100)

# Save and restore snapshots
snap = sim.snapshot()
sim.run_until(t=200)
sim.restore(snap)  # back to t=100, where the snapshot was taken
```

Each step continues the species, the clock and the events. An event whose
trigger is still true at a step boundary does not fire again there. A delayed
event pending across a boundary fires in the next step. A change between steps
that makes a trigger true fires it at the start of the next step.

`snapshot()` and `restore()` save and rewind the events with the state.
`set_state(x, time=t)` and `set_time(t)` also rewind the events, if `t` is
where one of the last 64 steps ended. Rolling back further warns and starts the
events afresh; take a `snapshot()` there instead. A run from any other start
time, or after `model.reset()`, starts each trigger from its `initialValue`.

A `run_until` step runs from the Simulator's own clock. If that clock is behind
the time of the model's state, as for a second `Simulator`'s first step after
another one's run, the step warns with `bngsim.LiveStartWarning`. Call
`set_time(t)` to put the clock at the state's time first.

### Rolling a step back

`restore` rebuilds the solver. That is fine for an occasional checkpoint, but
slow inside a loop. A predictor-corrector coupling loop redoes every step:
advance, correct the coupling input, then advance the same interval again.
`set_state(state, time=t)` moves the state and the clock together and keeps the
ODE solver warm:

```python
dt = 10.0                                # the coupling step
idx = model.species_names.index("R(l)")  # the species the coupling sets
coupling_value = 4.0e3                   # its corrected value

saved, t0 = sim.get_state().copy(), sim.current_time
sim.run_until(t0 + dt)                   # predictor
corrected = saved.copy()
corrected[idx] = coupling_value          # e.g. from the other half of the split
sim.set_state(corrected, time=t0)        # back to t0, with the correction
sim.run_until(t0 + dt)                   # corrector, over the same interval
```

Set the clock too whenever the model reads time: a time-indexed table function,
or `time()` in a rate law or function. Rolling back the state alone leaves
`current_time` a step ahead, and the redone step sees its inputs at the wrong
times. `sim.set_time(t)` sets the clock alone. Both calls check their input
first: a non-finite time raises `ValueError`, and a refused call changes
neither the state nor the clock.

`ReactionKernel.set_state(state, time=t)` does the same for a kernel.

A `run_until` step keeps the ODE solver's linear-solver state from the step
before (see [Linear solver](solvers.md#linear-solver)). A `run()` sets the
solver up afresh. Its species still start where the last run left them, unless
it passes `start="initial"`.

## Setting and saving species

```python
# Save the current concentrations as the new initial state
model.save_concentrations()

# Change one species (a bolus, say)
model.set_concentration("L(r,label~hot)", 500.0)

# Read one
print(model.get_concentration("L(r,label~hot)"))  # 500.0

# reset() goes back to the state save_concentrations() saved
model.reset()

# Named states, as BNG's saveConcentrations("name"): save two, restore either
model.save_concentrations("t=0")
# ... run, change the model ...
model.save_concentrations("start_competition")
model.restore_concentrations("t=0")
model.restore_concentrations("start_competition")
```

## Stop conditions

```python
# The model is where the steps above left it: binding knocked out (kf = 0),
# so the hot ligand comes off the receptor.

# An expression: stop when an observable drops below a threshold
sim.add_stop_condition("Bound_hot < 10000", label="low_hot")

# A callable: any Python logic (column 3 is the hot ligand-receptor complex)
sim.add_stop_condition(
    lambda r: r.species[-1, 3] < 5000.0,
    label="very_low_hot",
)

try:
    result = sim.run(t_span=(0, 1000), n_points=1001)
except bngsim.StopConditionMet as e:
    print(f"Stopped at t={e.result.time[-1]}: {e.condition}")
    partial = e.result  # the result cut at the stop
```

A condition is met at the first sample where it is true. It is checked on the
finished run, so it does not interrupt the solve. The run covers the whole
interval and leaves the model at its end. `e.result` is that run cut at the
first row where the condition held, and `sim.current_time` is the end of the
interval, not the stop time. A following `run_until` continues from the end of
the interval.

To continue from where a step started instead, take a snapshot first:

```python
snap = sim.snapshot()
try:
    sim.run_until(t=2000)
except bngsim.StopConditionMet as e:
    sim.restore(snap)   # back to where the step started
```

### What an expression can read

An expression reads observables by name, species by name with `(` and `)`
written as `_` (`A()` is `A__`), and the time as `t` or `time`. Nothing else,
not even Python builtins. A network-free run has no species; it reads the
observables its XML declares. A number is met where it is > 0.

`add_stop_condition` raises `ValueError` for a name it cannot read (a
misspelled observable, a parameter, a model function, `abs`), for `^` (Python's
bitwise XOR; write `**` for a power), and for a tuple.

During a run:

- An expression that raises at a sample raises `ValueError` from the run.
- Arithmetic numpy warns about (0/0, division by zero, overflow) takes numpy's
  value: a NaN is not met, and `+inf` is met where the condition reads `> 0`.
  Each run issues one `RuntimeWarning` naming the condition and the error.
  Your `np.errstate` settings are kept, so an error set to raise makes the
  expression raise.
- On Python 3.13 and later, any other warning raised inside a condition (such
  as numpy's `ComplexWarning`) cannot be shown from there, so the run raises a
  `ValueError` that says so.

A callable is met where it returns a true value. An exception it raises
propagates.

### Stop conditions in batch calls

`run_batch`, `run_replicates`, `parameter_scan` and `bifurcate` check every
row and raise nothing when a condition is met. A row that meets one comes back
cut there, next to the rows that ran to the end. `Result.stop_condition` names
the condition it met, or is `None`. Conditions are checked in the order they
were added, and the first one met cuts the row. An expression that raises at a
sample still raises `ValueError` from these calls.

`squeeze=True` is refused while a condition is registered, since a cut row is
shorter than the rest. `Result.squeeze(rows)` stacks the list when no row was
cut.

```python
# Each row starts from the state reset() returns to, the one save_concentrations()
# set above, with hot ligand bound. Binding is still knocked out (kf = 0), so the
# complex only comes apart: fast at kr = 0.1, hardly at all at kr = 1e-4.
model.reset()
rows = sim.run_batch(params=[{"kr": 0.1}, {"kr": 1e-4}], t_span=(0, 100))
[r.stop_condition for r in rows]   # ['low_hot', None]
```
