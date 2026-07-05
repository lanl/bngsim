# Events

## Events (SBML / Antimony)

BNGsim supports SBML events — discrete state changes triggered by boolean
conditions during simulation. Events are used for drug dosing protocols,
cell division, threshold switches, and other discontinuous interventions.

Events are automatically parsed from SBML and Antimony models:

```python
import bngsim

# Antimony: bolus dose at t=10, washout at t=50
model = bngsim.Model.from_antimony_string("""
    S = 100; D = 0;
    k_decay = 0.1;
    J1: S -> ; k_decay * S;

    at (time > 10): D = 50;     # inject drug at t=10
    at (time > 50): D = 0;      # wash out at t=50
""")

sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=201)
print(f"Model has {model.n_events} events")
```

```python
# SBML: events are parsed from <listOfEvents> automatically
model = bngsim.Model.from_sbml("model_with_events.xml")
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=201)
```

### How events work

Events fire when a trigger expression transitions from false to true (edge
detection). When an event fires, its assignment expressions are evaluated and
applied to the state vector, and the ODE integrator is restarted.

BNGsim's event implementation uses CVODE rootfinding (`CVodeRootInit`) for
precise event detection — the integrator finds the exact time point where the
trigger crosses zero, applies the assignments, and restarts.

### Supported event features

| Feature | Status | Notes |
|---------|--------|-------|
| Trigger expressions | ✅ | Any boolean expression on species/time/params |
| Event assignments | ✅ | Assign to species, parameters, or compartments |
| Multiple events | ✅ | Any number of events per model |
| `persistent` attribute | ✅ | SBML L3: trigger must stay true through delay |
| `initialValue` attribute | ✅ | Events that can fire at t=0 |
| Delay | ✅ | Constant delay (queued with cancellation) |
| Parameter → species promotion | ✅ | Parameters targeted by events auto-promoted |
| Priority ordering | ✅ | Real-valued priority; equal-priority ties broken by seed-keyed RNG (§4.11.6) |
| Same-instant cascade | ✅ | Delay-0 events an assignment triggers join the batch (§4.11.6) |

### Delayed events

Events with a delay queue their assignments and apply them after the delay
period has elapsed. Non-persistent events whose trigger reverts to false
before the delay expires are cancelled.

```python
# Antimony delayed event example
model = bngsim.Model.from_antimony_string("""
    S = 0; k = 0.1;
    J1: -> S; k;
    at (time > 5), delay=3: S = 100;  # trigger at t=5, apply at t=8
""")
```

### Known limitations

- **ODE only**: Events work with `method="ode"`. SSA events are not yet supported.
- **`useValuesFromTriggerTime`**: Not implemented — values are computed at
  application time, not trigger time.
- **Cross-validated**: 62/153 (40.5%) BioModels event models pass vs RoadRunner
  at max_rel_err < 1e-3. Including marginal matches: 79/138 = 57.2%.
  Remaining failures are due to event timing precision differences between
  CVODE rootfinding and RoadRunner's event detection algorithm.

### Events via the ModelBuilder API

For programmatic model construction, use `builder.add_event()`:

```python
from bngsim._bngsim_core import ModelBuilder

builder = ModelBuilder()
sp_S = builder.add_species("S", 100.0)
builder.add_parameter("k", 0.1)
builder.add_observable("S", [(sp_S, 1.0)])
builder.add_function("decay", "k*S")
builder.add_reaction([], [sp_S], "functional", "decay")

# Add event: when time > 10, set S = 200
builder.add_event(
    "bolus",                    # event id
    "time()>10",                # trigger expression
    [(sp_S, "200")],            # [(species_idx, value_expr), ...]
    delay=0.0,
    priority=0,
    persistent=True,
    initial_value=True,
)

model = builder.build()
```
