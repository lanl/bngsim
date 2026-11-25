---
summary: "Network-free simulation of rule-based models with NFsim and RuleMonkey: the methods, what each refuses, sessions, and how a rule's rate is counted."
---

# Network-free simulation (NFsim and RuleMonkey)

A rule-based model can have too many species to write out as a network.
BNGsim runs it network-free, with NFsim or RuleMonkey, in your Python process.
Both are in the wheels; a build from source can leave RuleMonkey out, and
`bngsim.HAS_RULEMONKEY` says whether it is there.

## Running a model network-free

Pass a network-free method and `xml_path`, the model's BioNetGen XML. BNG2.pl's
`writeXML()` action writes it from the `.bngl`.

```python
import bngsim

# The bundled binding model, A(b) + B(a) <-> A(b!1).B(a!1): the .net and the
# BioNetGen XML that BNG2.pl wrote from its .bngl.
xml_path = bngsim.examples.path("binding", "bng-xml")
model = bngsim.Model.from_net(bngsim.examples.path("binding", "net"))

# NFsim: rejection (null-event) sampling.
nf = bngsim.Simulator(model, method="nf", xml_path=xml_path)

# RuleMonkey: exact network-free simulation.
rm = bngsim.Simulator(model, method="rm", xml_path=xml_path)

result = rm.run(t_span=(0, 100), n_points=101, seed=42)
```

What to watch out for:

- **The run uses the model's parameters.** Every parameter you set with
  `Model.set_param` since loading is passed to the engine, whether you set it
  before or after building the `Simulator`. Parameters that depend on it follow,
  and so do seed amounts that read it. A derived parameter you wrote directly
  keeps the value you wrote. A model with all its parameters at their loaded
  values runs as the XML has it.
- **The run starts from the model's species amounts.** Every amount changed
  since loading, by `Model.set_concentration` or by an SSA or ODE run on the
  model, is passed to the engine. So a network-free run after an SSA run on
  the same model starts where the SSA run ended. Call `model.reset()` first, or
  give the SSA run `model.clone()`, to start both from the initial amounts. A
  network-free run leaves the model's state as it was.
- **Amounts are rounded to counts,** half up, as every seed is. A changed amount
  takes precedence over the parameter its seed amount reads. Where a run passes
  any parameter, it passes every seed species' amount too.
- **The species must be a seed species in the XML,** under the name the `.net`
  gives it. If it isn't, a nonzero amount is refused, because the engine would
  start that species at 0.
- **`run_batch` does not run a network-free method.** Scan point by point with
  `Model.set_param` and `Simulator.run`.

## The method names

| Method | Same as | Engine | Algorithm |
|---|---|---|---|
| `"nf"` | `"nf_reject"`, `"nfsim"` | NFsim | rejection (null-event) |
| `"rm"` | `"nf_exact"`, `"rulemonkey"` | RuleMonkey | exact, non-local |

Retired names such as `"nf_fixed"`, `"dynstoc"` and `"ds"` raise an error.
`normalize_method` shows how a name resolves:

```python
from bngsim import normalize_method

canonical, dispatch = normalize_method("nf")  # ("nf_reject", "nfsim")
canonical, dispatch = normalize_method("rm")  # ("nf_exact", "rulemonkey")
```

PyBNF uses the same engines in process: `method=>"nf"` or `"nfsim"` runs
NFsim, and `method=>"rm"`, `"rulemonkey"` or `"nf_exact"` runs RuleMonkey.

### NFsim's connectivity option

NFsim's `connectivity` option defaults to `False`, the conservative full
membership update. `connectivity=True` turns on NFsim's inferred dependency
graph. It gives correct results on our test models but is not reliably faster,
so use it only after checking it on your own model.

```python
sim = bngsim.Simulator(model, method="nf", xml_path=xml_path, connectivity=True)
```

## What each engine refuses

Each engine refuses a model it would run wrong, when the simulator or session
is built, with an error that names the rule, function or compartment. The way
out is to use the other engine, where the table shows it runs, or to simulate
the generated network with `method="ode"` or `method="ssa"`.

| Model feature | NFsim (`nf`) | RuleMonkey (`rm`) |
|---|---|---|
| Compartments | refused | refused |
| A function that reads `time()` | refused | runs |
| A table function indexed by time | runs, except inside a function with an argument | runs |
| `Arrhenius` (energy) rate law | refused | runs, with limits (below) |
| A function with an argument that reads an observable outside it | refused (below) | runs |
| A Michaelis-Menten law, or a local function read on one molecule (`A%x()`), on a whole-complex deletion that may match twice in a complex | refused (below) | runs |
| `MatchOnce` on a reactant the rule acts on that may match twice in a complex | runs | refused (below) |

NFsim also refuses any rate-law type it does not implement, such as `Hill` or
`Sat`: it implements `Ele`, `MM` and function rate laws only. Energy patterns
with no `Arrhenius` rule change no rate and run on both engines.

RuleMonkey also warns, with a `bngsim.RuleMonkeyWarning` when the simulator or
session is built, about constructs that are easy to misread: a rule whose
products can never be what it writes (BioNetGen makes no reaction for it, and
RuleMonkey never fires it), a Michaelis-Menten law over a pattern that matches
several species, `TotalRate`, and the like.

### Functions that read an observable outside their argument

In `fA(x) = k*(1+AC(x))*(1+0.01*MAB)`, `AC(x)` is read through the argument and
follows the molecule, but `MAB` is read outside it. NFsim updates such a
function only on the complexes a reaction produces, so everywhere else that term
goes stale. The same goes for `fB(x)*g()` where `g()` reads `MAB`, and for a
table function counted by an observable or the clock inside such a rate law.

So NFsim refuses such a read, naming the function or rule and the observable,
unless the observable can never change. It can't change when each of its
patterns is one of these:

- a pattern that names only types no rule names, held only in seed complexes of
  such types;
- a bare `X()` with no compartment, counted by a `Molecules` observable, that no
  rule adds or deletes (alone or with a whole species) and no fixed (`$`) seed
  species holds.

Because that count stays constant only while
the rules alone act on it, an `NfsimSession` refuses `add_molecules`,
`add_species`, `remove_species` and `set_species_count` on a type it counts.
The check errs toward refusing. A rate law that is a global function of its own,
such as `g()` reading `MAB`, is updated whenever `MAB` changes and runs.

### Arrhenius (energy) rules on RuleMonkey

RuleMonkey gives an energy rule BioNetGen's rate, `exp(-(Ea0 + phi*dG))`, where
`dG` is the free-energy change of the reaction the rule writes, and `1-phi` for
the reverse half of a reversible rule. Energies are in units of RT. There is no
RT in the formula, and a parameter named `RT` is an ordinary parameter. A model
in kJ/mol divides by its own `RT` where it uses it.

RuleMonkey runs an energy rule that makes one bond between two reactant
patterns, or breaks one within one, between molecules of two types, and does
nothing else. Where it can, it expands the rule at load. Elsewhere it computes
`dG` at each firing, as BioNetGen's network does per reaction, which costs more
per event the larger the complexes an event touches.

It refuses, naming the rule and the reason, an energy rule that:

- changes a state, or adds or deletes a molecule;
- makes or breaks more than one bond, or closes a ring;
- binds two molecules of one type;
- has more than two reactants, or include or exclude constraints;
- can't be expanded and has a bond that can reach an energy pattern joined only
  by `.`.

BioNetGen writes the reverse half of a reversible rule `X` as `_reverse_X`. So
two one-way rules named `X` and `_reverse_X`, mirror images with the same `phi`
and `Ea0`, read as one reversible rule, and RuleMonkey gives the second `1-phi`
where BioNetGen gives it `phi`.

## Rate laws that read `time()`

**RuleMonkey** runs a function that reads the clock, and a table function
indexed by time, on absolute model time, as `method="ode"` and `method="ssa"` do.
`run(t_span=(5, 7))` starts from the initial state at t = 5, and `time()` reads 5
there. RuleMonkey integrates a clock-dependent rate between events, so a rate
like `if(time()>=6, k, 0)` turns on at t = 6.

RuleMonkey reads the clock as `time()`, a bare `time`, or `t` (or `t()`). A bare
`time` or `t` that names a parameter, observable or function means that symbol
instead. BNG2.pl accepts an observable named `time`, so `k*time` then reads its
count, while `time()` is still the clock. BNGsim reads a model the same way when
it decides whether the model reads the clock.

**NFsim** refuses a function that reads `time()`. It runs a table function
indexed by time on the time elapsed since `t_start`, as BNG2.pl's `simulate_nf`
does. It reads the table again at each of the table's x values, so a step table
(`method=>"step"`) is exact. For a linear table that ramps, NFsim places each
firing where the propensity integrated since the last firing reaches an
exponential draw, as `rm` and `ssa` do. While a table ramps, each firing costs a
few readings of every propensity instead of one.

**Large start times.** A model that reads neither `time()` nor a time-indexed
table runs on elapsed time on both engines, so event times stay small at a
large `t_start`. RuleMonkey does the same for a model whose functions read the
clock: it places events on elapsed time and reads the clock as `time()` plus the
start, while every time it takes or reports is model time. On RuleMonkey, a
model with a table indexed by time runs on model time, and RuleMonkey still places its events on
the time elapsed since each call, so its waiting times keep their precision.
`method="ssa"` has a rounding limit on a rate that reads `time()` at very large
times.

## How a rule's rate is counted

Both engines must count how many distinct reactions a rule stands for right
now. Getting it wrong changes a rate by a whole-number factor. NFsim and
RuleMonkey count the same way, so `nf` and `rm` agree.

**Symmetry factor.** A reactant pattern with a symmetry matches the same
reaction more than once: `A(b) + A(b) -> A(b!1).A(b!1)` matches each pair of
molecules in both orders. BNG2.pl writes the correction into the XML as
`symmetry_factor` for every rate-law type, and BNGsim applies it to elementary
rate constants, global and local functions, function products and
Michaelis-Menten laws. For Michaelis-Menten the factor goes on the substrate
count inside the law, not on the finished rate, since the law is not linear.
An enzyme-side symmetry needs no factor: BNG2.pl writes 1 there.

**Context patterns.** A reactant pattern the rule never modifies, such as a
multi-subunit scaffold, is counted once per complex, not once per way it
matches inside that complex.

**`TotalRate`.** The rate law gives the rule's whole rate, so no symmetry factor
is applied. Two cautions: BioNetGen does not implement `TotalRate` for networks,
so `generate_network` writes the law into the `.net` as an ordinary rate
constant, and there is no network result to check a network-free run against.
BNG2.pl also rejects `TotalRate` on saturable, Michaelis-Menten, Hill and
Arrhenius laws, and on local functions.

**Deleting a whole complex, and `MatchOnce`.** A rule that deletes a whole
species without `DeleteMolecules` (`A(d!1).A(d!1) -> 0`, or `M() -> 0` on a
complex holding two `M`) removes the complex once, at the rule's rate, however
many matches it holds. That is what BioNetGen's network does, and what
`{MatchOnce}` asks for. NFsim gets it by firing such a pick with probability
1/m, where m is the number of matches in the picked complex. A `TotalRate` rule
is not thinned. RuleMonkey counts a whole-complex deletion once per complex.

A complex may hold two matches when the pattern has two or more molecules, or
is one molecule that does not list free every component some bond in the model
touches. There, some rules are refused:

- `nf` refuses a Michaelis-Menten law, which is not linear so the thinning can't
  correct it, and a local function read on one molecule (`A%x()`), whose value
  can differ between the matches. A local function read on the whole species
  (`%x:A()`) runs.
- `rm` reads no `MatchOnce`, so it refuses a `MatchOnce` reactant the rule acts
  on. A `MatchOnce` reactant the rule only reads is already counted once per
  complex, and runs.

One case is not handled: a `MatchOnce` pattern that matches one molecule twice
through a repeated component. `{MatchOnce}S(b~0) -> S(b~1) k` on
`S(b~0~1,b~0~1)` runs at 2k on both engines, where BioNetGen's network runs it
at k.

## Sessions

For work that changes parameters or molecule counts between segments, use a
session:

```python
from bngsim import NfsimSession, RuleMonkeySession

with NfsimSession(xml_path) as nf:
    nf.set_param("kf", 0.02)
    nf.initialize(seed=42)
    result = nf.simulate(0, 100, n_points=101)

with RuleMonkeySession(xml_path) as rm:
    rm.set_param("kf", 0.02)
    rm.initialize(seed=42)
    result = rm.simulate(0, 100, n_points=101)
```

- **NFsim's session clock is elapsed time.** A `set_param` reaches every rate
  constant, Michaelis-Menten `kcat` and `Km`, and function, before or after
  `initialize()`.
- **RuleMonkey's session clock is model time.** `current_time`, `step_to`, the
  clock read through `evaluate`, and the time in a `save_state` snapshot are all
  model time. `initialize(seed, t_start=5)` puts the seed state at t = 5. Left
  unset, the first `simulate` sets it from its own start, for a model that reads
  the clock, if nothing has moved the session first. A segment that starts at
  `current_time` runs on the clock. One that starts elsewhere continues from
  `current_time` and is labeled with the times you asked for. A model that reads
  the clock refuses that, since its rows would not match the times it read.
- On both, `solver_stats["n_steps"]` counts the events since `initialize`, not
  the segment's alone.

## What an NFsim run changes in your process

NFsim runs inside your Python process, so a few of its effects are process-wide:

- **The working directory is not changed.** BNGsim makes each relative TFUN
  `file=` path absolute against the XML's directory, so a table beside the XML
  loads wherever you run from. A failed load names the file.
- **C++ `std::cout` and `std::cerr` are redirected during an NFsim call,** so
  BNGsim can use NFsim's output in its error messages. Both are restored when
  the call ends, however it ends. C++ output from another thread during the call
  is captured too. C `printf` and Python's `print` are not affected.
- **One random-number stream is shared.** Which molecule a rule fires on is
  drawn from NFsim's process-wide generator, which every `initialize()` and
  `run()` reseeds; the rest of a run draws from its own session's stream. So a
  session stepped over several `simulate()` calls draws
  differently if another session is prepared between them. This shows in
  aggregation models (BLBR, TLBR). For same-seed reproducibility, run sessions
  one after another or in separate processes.
