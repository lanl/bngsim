# Network-free simulation (NFsim & RuleMonkey)

## Network-free simulation (NFsim and RuleMonkey)

For rule-based models with combinatorial complexity, use `Simulator` with a
network-free method token. BNGsim runs both vendored network-free backends
in-process; the `xml_path` argument points to the BNG-generated XML consumed by
NFsim or RuleMonkey.

```python
import bngsim

model = bngsim.Model.from_net("model.net")

# Rejection/null-event network-free simulation via NFsim:
sim = bngsim.Simulator(model, method="nf", xml_path="model.xml")
sim = bngsim.Simulator(model, method="nf_reject", xml_path="model.xml")
sim = bngsim.Simulator(model, method="nfsim", xml_path="model.xml")

# Exact non-local network-free simulation via RuleMonkey:
rm = bngsim.Simulator(model, method="rm", xml_path="model.xml")
rm = bngsim.Simulator(model, method="nf_exact", xml_path="model.xml")

result = rm.run(t_span=(0, 100), n_points=101, seed=42)
```

Each run uses the model's parameters. NFsim and RuleMonkey read the XML, so a
run passes them every parameter set on the model with `Model.set_param` since it
was loaded, before or after the `Simulator` was built. Parameters that depend on
it, and seed amounts that read it, follow, except a derived parameter written on
the model, which keeps the value written there as it does on the model. A model
whose parameters are all back at their loaded values runs as the XML has it.
`run_batch` does not run a network-free method: scan point by point with
`Model.set_param` and `Simulator.run`.

It starts from the model's species amounts the same way. A run passes the
backend every species amount changed since the model was loaded, whether by
`Model.set_concentration` or by an SSA or ODE run on the model (a network-free
run leaves the model as it was). So a network-free run after an SSA run on the
same model starts where that run ended, as a second SSA run would: call
`Model.reset()` first, or give the SSA run `model.clone()`, to start both from
the initial amounts. Where a run passes any parameter, it passes every seed
species' amount too. Each amount is rounded half-up to a count, as every seed
is. It takes precedence over the parameter that species' seed amount reads. The
XML has to list the species as a seed species, under the name the `.net` gives
it. If it doesn't, a nonzero amount is refused, because the backend would start
that species at 0.

### NFsim connectivity option

BNGsim's in-process NFsim wrapper defaults to `connectivity=False`.

- `connectivity=False` uses the conservative full membership-update path.
- `connectivity=True` enables NFsim's inferred dependency-graph path.

Use `connectivity=True` only as an explicit opt-in:

```python
sim = bngsim.Simulator(
    model,
    method="nfsim",
    xml_path="model.xml",
    connectivity=True,
)
```

Current guidance:

- `connectivity=true` is correctness-clean on the supported NF benchmark suite.
- It is not a general timing win across that suite, so the wrapper default remains `False`.
- Prefer the default unless you have validated `connectivity=True` on your model and workload.

### Network-free method tokens

BNGsim uses algorithm-based method names (not tool brands) following the taxonomy
of Chylek et al. (2013) and Suderman et al. (2019):

| Token | Canonical | Algorithm | Status | Backend |
|-------|-----------|-----------|--------|---------|
| `"nf"` | `nf_reject` | Default network-free policy | ✅ Available | NFsim |
| `"nf_reject"` | `nf_reject` | Rejection/null-event (Yang et al.) | ✅ Available | NFsim |
| `"nf_exact"` | `nf_exact` | Exact non-local network-free | ✅ Available when built with RuleMonkey | RuleMonkey |

**Compatibility aliases** (accepted, normalized internally):

| Alias | Resolves to | Notes |
|-------|-------------|-------|
| `"nfsim"` | `nf_reject` | Legacy NFsim token |
| `"rulemonkey"` | `nf_exact` | Legacy RuleMonkey token |
| `"rm"` | `nf_exact` | Short RuleMonkey token |

Retired experimental network-free tokens such as `"nf_fixed"`, `"dynstoc"`, and
`"ds"` raise clear errors if requested. No silent fallback occurs.

You can inspect the normalization programmatically:

```python
from bngsim import normalize_method

canonical, dispatch = normalize_method("nf")
# canonical = "nf_reject", dispatch = "nfsim"

canonical, dispatch = normalize_method("nfsim")
# canonical = "nf_reject", dispatch = "nfsim"

canonical, dispatch = normalize_method("rm")
# canonical = "nf_exact", dispatch = "rulemonkey"
```

## How a rule's rate is counted

Both network-free engines have to answer the same question before they can fire
a rule: **how many distinct reactions does this rule represent right now?** Two
model features make that answer non-obvious, and getting either wrong changes a
trajectory by an integer factor rather than by a little. Both are settled the
same way in NFsim and RuleMonkey, so `method="nf"` and `method="rm"` agree.

### The symmetry factor

A reactant pattern with a non-trivial automorphism matches the same reaction
more than once. `A(b) + A(b) -> A(b!1).A(b!1)` matches each unordered pair of
molecules twice, once in each order, so a naive count doubles the propensity.

BNG2.pl computes the correction as `1/automorphisms/context-permutations` and
writes it into the XML as `symmetry_factor`, **independently of the rate law
type** — a symmetric rule carries `symmetry_factor="0.5"` with `type="Ele"`,
`type="Function"` and `type="MM"` alike. BNGsim applies it on all of them:
elementary rate constants, global functions, local functions, function products
and Michaelis-Menten (issue #195).

For Michaelis-Menten the factor goes on the **substrate count inside the law**,
not on the finished propensity (issue #282). The factor corrects a match
multiplicity, so the law is being handed `2N` where `N` exists, and
`σ·MM(2N, E)` equals `MM(σ·2N, E)` only where the law is linear. Above
saturation the two placements genuinely differ. Only the substrate needs it: an
MM rule does not transform its enzyme, and BNG counts reaction-centre
automorphisms, so an enzyme-side symmetry arrives as `symmetry_factor="1"`.

### Reactant patterns with no reaction centre

A reactant pattern that the rule never modifies is pure *context* — a
multi-subunit scaffold a rule reads but does not touch. Such a pattern is
counted **once per complex**, not once per way it embeds into that complex
(issues #281, #298, #300). Counting embeddings runs a multi-subunit catalyst two
or three times too fast.

### `TotalRate`

Under `TotalRate` the rate law states the whole propensity of the rule outright.
There is no match count to correct, so **the symmetry factor is not applied at
all** (issue #426). Applying it ran a `TotalRate` rule with a symmetric reaction
centre at half the rate the model asks for.

Two cautions on `TotalRate` generally. BioNetGen does not implement it for
network simulations — `RxnRule.pm` carries the note that it is implemented only
for XML network-free output — so `generate_network` writes the rate law into the
`.net` as an ordinary rate constant and the ODE integrates plain mass action.
A `TotalRate` model therefore has no BNG2 result to check a network-free run
against. It is also rejected by BNG2.pl on saturable, Michaelis-Menten, Hill and
Arrhenius laws, and on local functions.

### Rate laws that read `time()`

`method="rm"` runs a function that reads `time()` (or a bare `time`), and a
table function indexed by time (`tfun(..., time)`), on **absolute model time**,
as `method="ode"` and `method="ssa"` do: `run(t_span=(5, 7))` starts from the
initial state at t = 5, and `time()` reads 5 there. RuleMonkey integrates a rate
that reads the clock between events, so a rate such as `if(time()>=6, k, 0)`,
which is 0 at the start, turns on at t = 6 (issue #1022).

RuleMonkey reads the clock as `time()`, a bare `time`, or its alias `t` (or
`t()`). A bare `time` or `t` that names a parameter, observable or function is
that symbol instead: BNG2.pl accepts an observable named `time`, and `k*time`
then reads its count, while `time()` is still the clock. bngsim reads a model
the same way when it decides whether the model reads the clock.

NFsim cannot prepare a function that reads `time()`, so `method="nf"` refuses
one when the simulator or session is built, naming the function (issue #862).
Use `method="rm"`, or `method="ode"` or `method="ssa"` on the generated network.
NFsim runs a table function indexed by time, on **time elapsed since
`t_start`**, as BNG2.pl's `simulate_nf` does. It reads the table again at each
of the table's x values, whether or not a reaction has fired, and draws the next
waiting time from there. That is exact for a step table (`method=>"step"`): a
rate that is 0 at the start turns on at the step that turns it on (issue #892).
A linear table changes between those reads, and NFsim would hold the value it
last read, so where little fires the run would lag the ramp: with 100 `X(p~1)`
leaving at a rate that rises from 0 at t = 0 to 10 at t = 1, none would have
left by t = 1, where about 70 have left by t = 0.5. So `method="nf"` refuses a
model whose rate laws read a linear table indexed by time, directly or through
the functions they call, and names the tables (issue #1179). Use `method="rm"`,
which integrates the ramp, or `method="ssa"` on the generated network. To stay
on NFsim, make the table a step on a grid fine enough for the ramp, each step
holding the ramp's value at its middle: steps of a tenth of the ramp matched the
exact mean within sampling noise. A linear table that no rate law reads is
not refused; NFsim reads it correctly at each output time.

A model that reads neither `time()` nor a time-indexed table cannot tell the
two clocks apart, and runs on elapsed time on both backends, which keeps event
times small at a large `t_start`. So does a `method="rm"` run or a
`RuleMonkeySession` of a model that reads the clock in its functions: wherever
RuleMonkey reads the clock, it reads `time()` plus the run's or the session's
start, and its events are placed on elapsed time, while every time it takes or
reports is model time (issue #1063). Before that its waiting times were rounded
once the total propensity times |t| neared 1e15, and from t = 1e12 a run or a
session was biased with no error.

A model with a table indexed by time is not offset, and runs on model time.
Since RuleMonkey 3.12.0 that keeps its waiting times too: RuleMonkey places its
events, and the panels on which it integrates a rate that reads the clock, on
the time elapsed since each call starts (richardposner/RuleMonkey#110), and only
the clock a rate reads is rounded to the doubles near it. From t = 1e13 a step
table's run matches its closed form, where bngsim used to refuse a span that
short (issue #1063). `method="ssa"` keeps the rounding limit on a rate that
reads `time()` (issue #1059).

### Arrhenius (energy) rules

`method="rm"` gives an energy rule BNG2's rate, `exp(-(Ea0 + phi*dG))` with `dG`
the free-energy change of the reaction the rule writes, and `1-phi` for the
reverse half of a reversible rule. Energies are in units of RT: there is no RT
in the formula, and a parameter named `RT` is an ordinary parameter, which a
model in kJ/mol divides by where it uses it (issue #1022).

RuleMonkey runs an energy rule that makes one bond between two reactant
patterns, or breaks one within one, between molecules of two types, with no
other operation. It expands the rule at load where that is exact, as for pair
and cooperativity patterns, and everywhere else counts `dG` at each firing as
BNG2's network does per reaction: an energy pattern with a state, a site that
must be free, wildcard bonds, a site named twice or context further out, and
context in the rule's own reactant pattern all run at BNG2's rate
(richardposner/RuleMonkey#109; `method="rm"` refused these until issue #1058).
Counting at each firing costs more per event, and more the larger the complexes
an event touches.

RuleMonkey refuses at load, naming the rule and the reason, an energy rule that
changes a state or adds or deletes a molecule, makes or breaks more than one
bond, closes a ring, binds two molecules of one type, has more than two
reactants, or carries include or exclude constraints, and a rule it cannot
expand whose bond can reach an energy pattern joined only by `.`. Simulate such
a model with `method="ode"` or `method="ssa"` on the generated network. BNG2
writes the reverse half of a reversible rule `X` as a rule named `_reverse_X`,
so two unidirectional rules, `X` and a mirror image named `_reverse_X` with the
same `phi` and `Ea0` parameters, read as one reversible rule, and RuleMonkey
gives the second `1-phi` where BNG2 gives it `phi`.

### Compartments

RuleMonkey does not scale rates by compartment volume or keep molecules in
their compartments, so `method="rm"` refuses a model that declares
compartments (issue #1022). Use `method="ode"` or `method="ssa"` on the
generated network.

### RuleMonkey's warnings

RuleMonkey refuses a model it cannot honor, naming what it cannot, and warns of
constructs it reads in a way that is easy to miss: a rule whose products can
never be what it writes (BioNetGen generates no reaction for it, and RuleMonkey
never fires it), a Michaelis-Menten law over a pattern that matches several
species, `TotalRate`, and the like. Each warning is a
`bngsim.RuleMonkeyWarning`, emitted when the simulator or session is built.

### Rate laws NFsim cannot run

NFsim refuses, when the simulator or session is built, a model with an
`Arrhenius` (energy-based) rate law or with a rate-law type it does not
implement, naming each rule. NFsim's expansion of energy rules divides every
energy by an `RT` that BioNetGen does not apply (2.478 unless the model defines
a parameter `RT`), gives a one-way binding rule a reverse reaction, and drops
one-way unbinding and state-change rules, all without an error
(RuleWorld/nfsim#102; the refusal is lifted under issue #1057). NFsim implements `Ele`, `MM` and function rate laws only:
it dropped a `Hill` rule silently, and it stops the load on `Sat`. Energy
patterns with no `Arrhenius` rule change no rate and still run. Simulate such a
model's generated network with `method="ode"` or `method="ssa"`.

### Compartment models

NFsim refuses a model that declares compartments, naming them. Its compartment
support ran such models wrong without an error, in ways no check of the XML can
bound: it ignored a species-level compartment on a pattern (`@EC:A()` matched
every compartment, RuleWorld/nfsim#106); applied no volume factor to a rule with
two or more reactants or to synthesis spanning compartments (#108); gave no
compartment to a molecule a rule creates with none written (#109); put every
molecule of a seed complex in the species' compartment, bound molecules in
compartments that do not touch, and moved a whole complex on `MoveConnected`
(#110). BNG2.pl's `writeXML` also drops `NumberPerQuantityUnit`
(RuleWorld/bionetgen#348), so even a careful check could not see it. Issue #1067
lifts the refusal once the fixes are vendored; until then simulate a compartment
model's generated network with `method="ode"` or `method="ssa"`.

### Functions that read an observable outside their argument

In `fA(x) = k*(1+AC(x))*(1+0.01*MAB)`, `AC(x)` is read through the function's
argument and follows the molecule, but `MAB` is read outside it. NFsim
re-evaluates a function with an argument only on the complexes a reaction
produces, so on every other molecule that term keeps the value it had at the
molecule's last event: on a small binding model the bound count came out about
10 % low at t = 10, without an error (RuleWorld/nfsim#104). A rate law such as
`fB(x)*g()`, whose global `g()` reads `MAB`, goes stale the same way, and so does
a table function counted by an observable or the clock, `fB(x)*tfun(..., MAB)` or
`fB(x)*tfun(..., time)`. NFsim
therefore refuses such a read in a function a rate law uses, naming the function
(or the rule whose rate law it is) and the observable, unless the observable can
never change: each of its patterns either names only types that no rule names,
held in seed complexes of such types only, or is a bare `X()` with no
compartment, counted by a `Molecules` observable, that no rule adds, deletes, or
can delete along with a whole species, and that no fixed (`$`) seed species
holds. Because such a count is constant only while the rules alone act on it,
an `NfsimSession` also refuses `add_molecules`, `add_species`, `remove_species`
or `set_species_count` on a type it counts. The check errs toward refusing (issue #1062 lifts
it once the upstream fix is vendored). A rate law that is a global function of
its own, such as `g()` reading `MAB`, is re-evaluated whenever `MAB` changes and
runs. Otherwise simulate the model's generated network with `method="ode"` or
`method="ssa"`.

### Stateful network-free sessions

For advanced workflows that mutate parameters or live particle counts between
segments, use the stateful session APIs:

```python
from bngsim import NfsimSession, RuleMonkeySession

with NfsimSession("model.xml") as nf:
    nf.set_param("kp1", 0.5)
    nf.initialize(seed=42)
    result = nf.simulate(0, 100, n_points=101)

with RuleMonkeySession("model.xml") as rm:
    rm.set_param("kp1", 0.5)
    rm.initialize(seed=42)
    result = rm.simulate(0, 100, n_points=101)
```

A RuleMonkey session's clock is model time: `current_time`, `step_to`, the
clock read through `evaluate`, and the time a `save_state` snapshot holds are
all model time, though its events are placed on elapsed time where the model
allows (see "Rate laws that read `time()`" above). `initialize(seed,
t_start=5)` puts the seed state at t = 5; left unset, the first `simulate` sets it from its own
start for a model that reads the clock, provided nothing has moved the session
first. A segment that starts at `current_time` runs on the clock. One that
starts elsewhere continues from `current_time` and is labelled with the times
asked for, which a model that reads the clock refuses, since its rows would not
be labelled with the times it read. An NFsim session's clock is elapsed time.
On both, a result's `solver_stats["n_steps"]` counts the events since
`initialize`, not the segment's alone.

On NFsim, a `set_param` reaches every rate constant, Michaelis-Menten `kcat` and
`Km`, and function, before or after `initialize()`. NFsim refuses a compartment
model (see "Compartment models" above); once that refusal is lifted, note that
a write to a compartment-size parameter does not reach the model either, since
BioNetGen writes the size into the XML as a number (issue #1061, waiting on
RuleWorld/bionetgen#347 and RuleWorld/nfsim#105).

**PyBNF integration**: When `bngsim` is installed, `method=>"nf"` / `"nfsim"`
routes through the in-process NFsim backend, and `method=>"rm"` / `"rulemonkey"`
/ `"nf_exact"` routes through RuleMonkey when RuleMonkey support is compiled in.
