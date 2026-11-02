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

#### A trigger that holds at one instant

`at (time == 2)` is true at a single time. No root finder brackets it, since
the trigger reads false on both sides of any step across it, so the run stops
on the instant instead and the event fires there, on any output grid, in the
ODE and stochastic engines alike. The instant is the time at which the model's
evaluator reads the equality as true: for `time == tau/7` that is the
evaluator's own rounding of `tau/7`, and `time == pi` or `time^2 == 4` is found
the same way. An equality of things that hold their values, such as
`floor(time/24) == 3`, is true for a stretch and is an ordinary window.

Two cases are outside this:

- An equality that no time satisfies in double arithmetic, such as
  `time - 1 == 1.3` (2.3 − 1 rounds to 1.2999999999999998, and the next
  double gives 1.3000000000000003). Where that equality is the whole trigger
  the event could never fire, and the run is refused with an error that names
  it. Write the instant on one side (`time == 2.3`), or use `>=`. Inside a
  larger trigger (`time - 1 == 1.3 || time >= 4`) the event can fire through
  the rest, and nothing is refused.
- An equality on a model variable rather than on `time` (`at (x == 2)` with a
  rate rule `x' = 1`, or on a species). The variable is integrated and does not
  land on the value exactly; use an ordering.

#### Triggers that start on their threshold

A trigger reading false whose *residual* is exactly zero at the start time is
not below its threshold, it is sitting on it — `S > 0` with `S(t_start) = 0`.
Whether the first move off that surface is a crossing depends on where the
trajectory goes. bngsim decides it by evaluating the trigger itself a small
step along the flow from the start time:

- **It leaves into the true side** — the crossing is real and located at the
  start time. `time > 0` is this case, and so is a species with a nonzero
  production rate there, and `A < 10` with `A(t_start) = 10` and A decaying.
  The event fires, however the comparison is written.
- **It does not leave** — nothing crosses. A root reported at the start time is
  the initial condition being re-read, and its instant would be set by the
  integrator's first step rather than by the model, so the event does not fire.
  A later, genuine crossing of the same trigger is unaffected.

The step is first order in the flow. A trigger that leaves its threshold only at
second order, `B > 0` where `dB/dt` is itself 0 at the start, is read as not
leaving (issue #934).

This mirrors what SUNDIALS does with a root function that is identically zero at
`t0`. Declare `initialValue="true"` on the trigger if you want an event
suppressed at the start time regardless of the flow.

#### An event beside a state-dependent rate-law switch

A forward-sensitivity run stops at a rate-law switch, `if(A < thr, ...)` in a
rate law, and restarts a step of about 1e-13 of the time scale past it. An event
whose trigger rises within that step, such as `at (A < thr)` on the same
threshold, has no located crossing of its own. The run is refused rather than
the event being dropped. A trigger that falls there is recorded, so its next
rise fires. Runs without sensitivities are unaffected.

### Supported event features

| Feature | Status | Notes |
|---------|--------|-------|
| Trigger expressions | ✅ | Any boolean expression on species/time/params |
| Event assignments | ✅ | Assign to species, parameters, or compartments |
| Multiple events | ✅ | Any number of events per model |
| `persistent` attribute | ✅ | SBML L3: trigger must stay true through delay |
| `initialValue` attribute | ✅ | Events that can fire at t=0; a `run_until` leg continues the previous leg's trigger state instead (issue #693) |
| Delay | ✅ | Constant delay (queued with cancellation) |
| Parameter → species promotion | ✅ | Parameters targeted by events auto-promoted |
| Priority ordering | ✅ | Real-valued priority, read when each event executes; equal-priority ties broken by seed-keyed RNG (§4.11.6) |
| Same-instant cascade | ✅ | Delay-0 events an assignment triggers join the batch (§4.11.6) |
| `useValuesFromTriggerTime` | ✅ | SBML L3 default (`True`): assignment RHS frozen at trigger time. `False` evaluates at firing time |
| SSA / PSA events | ✅ | Delayed events included (issue #526) |

### Delayed events

Events with a delay queue their assignments and apply them after the delay
period has elapsed. Non-persistent events whose trigger reverts to false
before the delay expires are cancelled.

Everything that executes at one instant is one batch, whether it was triggered
just then or a delay ago: the event of highest priority executes next, and the
priorities are read again after each execution, so a priority that changed
during a delay counts at its value when the event executes. Two events are on
one instant when their execution times are the same number. Delayed events of
equal priority execute in the order they were triggered in.

A strict comparison and the non-strict one with the same two sides are the
exception where the run crosses their threshold. `time/3 > tau` beside
`time/3 >= tau` (or `tau <= time/3`) can turn true at one read, but the
non-strict one is true on the crossing and the strict one only past it. So the
strict one executes after the non-strict one and everything else of the
instant, and its trigger, values and delay are read at the state they left.
Where an event of the instant moves either side, the strict one is raised there
instead, as a cascade, and executes where its priority puts it.
That holds in ODE and SSA runs, whatever the rounding of the threshold (issue
#947). Where the two sides jump past each other instead (a reaction firing in
SSA, a `piecewise` that switches), both turn true at one instant and they are
one batch. A forward-sensitivity run refuses such a pair by name where the
strict event reads or writes a species the first ones wrote.

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

    at (time > 10): S = 200;              # bolus
    at 2 after (time > 15): S = S + 50;  # a delayed one
""")

sim = bngsim.Simulator(model, method="ssa")
result = sim.run(t_span=(0, 20), n_points=21, seed=42)
```

Delayed events run under `method="ssa"` and `method="psa"` as under
`method="ode"` (issue #526). The delay is read when the event triggers, and
with `useValuesFromTriggerTime` (the default) so are the assigned values. The
execution then waits in a queue and is taken at its own time, which stops the
reaction step that would have passed it. A non-persistent event whose trigger
goes false before then does not execute. A row sampled at the execution time
reads its effect. Executions still waiting at the end of a `run_until` leg
carry into the next.

### Known limitations

- **`rateOf` triggers are ODE-only**: models using the SBML `rateOf` csymbol are
  rejected under SSA/PSA — the instantaneous dx/dt it refers to has no
  well-defined value in a discrete stochastic trajectory.
- **An event that re-arms its own trigger as it fires is refused (Zeno)**: an
  assignment that puts the trigger back where the flow makes it true again at
  once — `at (B > 0): B = 0, n = n + 1` with `B' = 1`, or a ring of events each
  re-arming the next — makes the events fire infinitely often at one instant,
  and no trajectory continues past it. `method="ode"` stops with an error naming
  the events after 100000 fires in a row, each within CVODE's root tolerance of
  the last (issue #1158). Change the assignment or the trigger so the event
  leaves its trigger false for a time, for example by moving the state past the
  threshold rather than onto it. Fast event trains spaced by more than the root
  tolerance are not affected. A clamp like `at (X < 0): X = 0` on a decaying X,
  whose fires change X by less than the integration tolerance, is not refused:
  once X has decayed far below `atol` bngsim stops re-firing it instead.
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
