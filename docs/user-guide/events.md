---
summary: "SBML and Antimony events: triggers, delays, priorities and assignments, in ODE and stochastic runs, and what BNGsim refuses."
---

# Events

An event changes the model's state at once when its trigger turns true: a drug
dose, a washout, a cell division, a threshold switch. BNGsim reads events from
SBML and Antimony models, and from models built with `ModelBuilder`.

## Running a model with events

Nothing extra is needed. Load the model and run it:

```python
import bngsim

# A bolus at t = 10 and a washout at t = 50.
model = bngsim.Model.from_antimony_string("""
    S = 100; D = 0;
    k_decay = 0.1;
    J1: S -> ; k_decay * S;

    at (time > 10): D = 50;
    at (time > 50): D = 0;
""")

sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=201)
print(model.n_events)  # 2
```

An SBML file's `<listOfEvents>` is read the same way:

```python
model = bngsim.Model.from_sbml("model_with_events.xml")
result = bngsim.Simulator(model, method="ode").run(t_span=(0, 100), n_points=201)
```

An event fires when its trigger goes from false to true. Its assignments are
applied to the state, and the integrator restarts from there. In an ODE run,
CVODE's root finder locates the exact time the trigger turns true.

What to watch out for:

- **A trigger on a model variable should be an ordering, not an equality.**
  `at (x == 2)` on an integrated variable or a species almost never lands on 2
  exactly. Use `x >= 2`.
- **An event that re-arms its own trigger is refused.** If an assignment puts
  the trigger back where the flow makes it true again at once, such as
  `at (B > 0): B = 0, n = n + 1` with `B' = 1`, the event would fire forever at
  one instant. An ODE run stops with an error naming the events after 100,000
  fires in a row, each within the root tolerance of the last. Move the state
  past the threshold, not onto it. A clamp like `at (X < 0): X = 0` on a
  decaying X is not refused: once X is far below `atol`, BNGsim stops re-firing
  it.
- **`rateOf` is ODE-only.** A model that uses SBML's `rateOf` is refused under
  SSA and PSA, since a stochastic trajectory has no derivative to read.

## What BNGsim supports

| Feature | Notes |
|---|---|
| Triggers | any boolean expression of species, parameters and time |
| Assignments | to species, parameters or compartments; a parameter an event sets becomes part of the state |
| Delays | a constant or an expression, read when the event triggers |
| `persistent` | the trigger must stay true through the delay, or the event is canceled |
| `initialValue` | the trigger's value just before the run starts; a `run_until` leg carries on the previous leg's trigger state instead |
| `useValuesFromTriggerTime` | default `True`: assigned values are read when the event triggers; `False` reads them when it executes |
| Priority | read when each event executes; ties are broken by a random draw keyed to the seed |
| Same-instant cascades | a delay-0 event that an assignment triggers joins the same batch |
| SSA and PSA | all of the above, delayed events included |

## Delayed events

A delayed event reads its delay, and by default its assigned values, when it
triggers, then waits and executes at its time. A non-persistent event whose
trigger turns false before then does not execute.

```python
model = bngsim.Model.from_antimony_string("""
    S = 0; k = 0.1;
    J1: -> S; k;
    at 3 after (time > 5): S = 100;  # triggers at t = 5, executes at t = 8
""")
```

Everything that executes at one instant is one batch, whether it triggered
then or a delay ago. The event with the highest priority executes next, and
priorities are read again after each execution, so a priority that changed
during a delay counts at its value when the event executes. Two events are at
one instant when their execution times are the same number. Delayed events of
equal priority execute in the order they triggered.

A strict comparison beside its non-strict twin is the one exception. `time/3 >
tau` and `time/3 >= tau` can turn true at the same reading, but the non-strict
one is true on the crossing and the strict one only past it. So the strict one
executes after everything else at that instant, and reads its trigger, values
and delay from the state they left. If an event at that instant moves either
side, the strict one is raised there instead, as a cascade, and executes where
its priority puts it. Where the two sides jump past each other instead (an SSA
reaction firing, a `piecewise` that switches), both turn true at once and are
one batch. A forward-sensitivity run refuses such a pair, by name, where the
strict event reads or writes a species the first ones wrote.

## Events in stochastic runs

Events fire under `method="ssa"` and `method="psa"` too. Triggers are checked
after every reaction, and a trigger on time alone fires at its exact time, so a
bolus at `time > 10` lands at t = 10. A delayed execution stops the reaction
step that would pass it. A sampled row at the execution time shows its effect.
Executions still waiting at the end of a `run_until` leg carry into the next.

```python
model = bngsim.Model.from_antimony_string("""
    S = 100; k_decay = 0.1;
    J1: S -> ; k_decay * S;

    at (time > 10): S = 200;             # a bolus
    at 2 after (time > 15): S = S + 50;  # a delayed one
""")

sim = bngsim.Simulator(model, method="ssa")
result = sim.run(t_span=(0, 20), n_points=21, seed=42)
```

## Events in a model built in Python

`ModelBuilder.add_event` adds an event:

```python
import bngsim

b = bngsim.ModelBuilder()
b.add_parameter("k", 0.1)
b.add_species("S", 100)
b.add_reaction(["S"], [], rate="k")

# When time reaches 10, set S to 200.
b.add_event("time() >= 10", {"S": "200"}, name="bolus")

model = b.build()
```

## Details

### A trigger that holds at one instant

`at (time == 2)` is true at a single time, so no root finder can bracket it.
Instead the run stops on that instant and the event fires there, on any output
grid, in ODE and stochastic runs alike. The instant is where the model's
evaluator reads the equality as true: for `time == tau/7`, its own rounding of
`tau/7`. `time == pi` and `time^2 == 4` work the same way. An equality of things
that hold their values, such as `floor(time/24) == 3`, is true for a stretch and
is an ordinary trigger.

The other side may be a parameter that events set, as in a dose counter:

```
var c = 0.5;
E: at (time == c && c < 91): dose = dose + 100, c = c + 1;
```

The run stops on each instant in turn: 0.5, 1.5, 2.5 and so on. This also works
where `c` is read through an assignment rule, after a delay, with `!=`, and for
a first dose or a later one (`time == t0 || (time == c + t0 && time < t_end)`).
In a stochastic run, a value within rounding of a whole number is kept on that
number.

For this, each event that sets `c` must fire at a fixed time (`time >= t1`), on
such an instant, or on a flag only events set (`on == 1`) at the instant another
event raises it. And it must set a value known before the run: numbers,
constants, parameters events set, or the instant's time. Several such events at
one instant run together where none reads or sets what another sets.

Otherwise the run is refused, with an error naming the parameter and why. It is
refused when:

- an event on the state sets the parameter, or reads such a parameter;
- two events at one instant each set what the other reads, or a third reads
  what both set;
- a value is frozen at a trigger on the instant another event sets what it
  reads;
- an event on `time > t1`, which fires at a root just after t1, makes a trigger
  rise, or executes beside an instant where their order matters;
- in an ODE run, an instant falls within rounding of another instant's stop, or
  of a fixed-time event that reads or sets what it does, since the integrator
  merges them (every 8 h beside every 12 h, in days, puts 1/3 + 1/3 + ... at
  1.9999999999999998 beside 2.0; a stochastic run takes both);
- the events execute more than 10,000 times in one run. Run it in shorter legs.

An equality that no double satisfies, such as `time - 1 == 1.3`, can never fire.
Where it is the whole trigger, the run is refused, naming it. Write the instant
on one side (`time == 2.3`), or use `>=`. Inside a larger trigger
(`time - 1 == 1.3 || time >= 4`) the rest can still fire it, and nothing is
refused.

### A trigger that starts on its threshold

A trigger that reads false at the start can sit exactly on its threshold: `S >
0` with `S = 0`. BNGsim follows the SBML start rule, as libRoadRunner does. The
event fires at the start if the trajectory leaves the threshold into the
trigger's true side, at whatever order it leaves, and never if it stays on the
threshold or leaves into the false side.

- `time > 0` fires at once, and so does a species with a nonzero production
  rate, `A < 10` with `A = 10` and A decaying, and `B > 0` where B rises at second
  order (`B = t²/2`).
- A trigger that stays on the threshold, or leaves into the false side, does not
  fire. A later real crossing still fires.

`initialValue` gives the trigger's value just before the start, so it decides
only a trigger that is already true there. `S > 0` at `S = 0` reads false
whatever `initialValue` is; `S >= 0` there reads true, and `initialValue`
decides it.

Where `t_start` is not 0, the first steps of the run are too small to read
against the rounding of the time. So a trigger on its threshold that the flow
does not carry into its true side at first order is held false for
1e4·eps·|t_start| (1.6e-12 from 0.7), then read as it is. From `t_start = 0`
there is no hold.

A forward-sensitivity run refuses, by name, an event that fires at the start
this way without first-order motion, since its fire time has no derivative
there.

### An event beside a rate-law switch

A forward-sensitivity run stops at a state-dependent switch in a rate law,
`if(A < thr, ...)`, and restarts about 1e-13 of the time scale past it. An event
whose trigger rises inside that step, such as `at (A < thr)` on the same
threshold, has no crossing of its own to locate, so the run is refused rather
than the event dropped. A trigger that falls there is recorded, so its next
rise fires. Runs without sensitivities are not affected.
