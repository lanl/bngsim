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
| `useValuesFromTriggerTime` | ✅ | SBML L3 default (`True`): assignment RHS frozen at trigger time. `False` evaluates at firing time |
| SSA / PSA events | ✅ | Non-delayed events only — see limitations below |

### Delayed events

Events with a delay queue their assignments and apply them after the delay
period has elapsed. Non-persistent events whose trigger reverts to false
before the delay expires are cancelled.

```python
# Antimony delayed event example
model = bngsim.Model.from_antimony_string("""
    S = 0; k = 0.1;
    J1: -> S; k;
    at 3 after (time > 5): S = 100;  # trigger at t=5, apply at t=8
""")
```

### Events under SSA and PSA

Events also fire in the stochastic engines. Triggers are checked at every
reaction-fire boundary, and a time-only trigger is honored at its exact crossing
time rather than at the next τ-step — so a bolus at `time > 10` lands at `t=10`,
not somewhere in the interval that straddles it.

```python
model = bngsim.Model.from_antimony_string("""
    S = 100; k_decay = 0.1;
    J1: S -> ; k_decay * S;

    at (time > 10): S = 200;   # bolus, no delay — supported under SSA
""")

sim = bngsim.Simulator(model, method="ssa")
result = sim.run(t_span=(0, 20), n_points=21, seed=42)
```

Delay-bearing events are the one gap: `method="ssa"` and `method="psa"` reject
them at run entry with an error naming the offending event. Use `method="ode"`
for models that need delayed events.

### Known limitations

- **Delays are ODE-only**: `method="ode"` supports the full event feature set.
  `method="ssa"` and `method="psa"` support events *without* delays; a
  delay-bearing event is rejected at run entry with an error naming the event id.
  Use `method="ode"` or drop the `delay` attribute.
- **`rateOf` triggers are ODE-only**: models using the SBML `rateOf` csymbol are
  rejected under SSA/PSA — the instantaneous dx/dt it refers to has no
  well-defined value in a discrete stochastic trajectory.
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
    use_values_from_trigger_time=True,  # SBML L3 default; freeze RHS at trigger
)

model = builder.build()
```
