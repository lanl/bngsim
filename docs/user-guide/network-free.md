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

**RuleMonkey runs them**, on absolute model time (issue #1022).
`run(t_span=(5, 7))` starts from the initial state with the clock at 5, and
`time()` reads 5 to 7, as under `method="ode"` and `"ssa"`. A rate such as
`if(time()>=6, k, 0)`, which is 0 at the start, fires when it turns on.

RuleMonkey integrates such a rate between events in steps of up to a thousandth
of a call, and a step can lie across a short window in the rate and see none of
it. So bngsim works out from the XML when the rates change with the clock and
ends an engine call on each of those instants. A pulse is then seen however
short it is and however long the run. For that, a function has to read the clock
in one of these ways:

- **smoothly**: sums, products and whole-number powers of `time()`, parameters
  and observables, through `exp`, `sin`, `cos`, `tanh`, `sinh`, `cosh`, `atan`,
  `erf` and `erfc`, a positive number raised to any power (`2^(-time()/half)`),
  and division by anything that does not read the clock (`time()/(60*24)`);
- **as a switch**: `time()` compared with a number or a parameter, as in
  `time() >= 6` or `(time() < t_off)`, in an expression that reads the clock no
  other way;
- **through a table** indexed by time (`tfun(..., time)`), step or linear;
- or be built from functions of those kinds.

Anything else that reads the clock is refused with `ModelError` when the
simulator or session is built, naming the function: `floor(time()/5)`,
`abs(time() - 6) < w`, a threshold that is an expression (`time() >= 6 + w`;
give the sum a parameter of its own), a division by the clock, `sqrt(time())`,
or a switch and a smooth term in one expression (write them as two functions and
multiply). bngsim cannot say where such a rate changes, so RuleMonkey could step
over it. Use `method="ssa"` or `"ode"` on the generated network for those.

Two limits remain. A smooth rate with a feature much narrower than a thousandth
of the run can be stepped over, because nothing marks where it is: a Gaussian
pulse 0.01 wide in a run of 1000 (`ssa` misses that one too), or a rate that
dips below zero, which RuleMonkey reads as 0, and is positive only that
briefly. And the clock is a double: at t = 1e12 it cannot resolve a wait
shorter than 0.0001, under `ssa` as well.

Each instant a rate changes at ends one engine call and starts another, which
costs about a third of a millisecond, and RuleMonkey's steps shrink with the
call. A table of 1,000 points adds about 0.3 s to a run that crosses all of
them, and one of 10,000 points about 3.5 s.

A model that does not read the clock sees only elapsed time, and is run on
elapsed time whatever `t_span` it is given, exactly as before.

**NFsim does not run them.** It cannot prepare a function that reads `time()`
(or a bare `time`), so `method="nf"` refuses one when the simulator or session
is built, naming the function (issue #862). Use `method="rm"` on the same XML,
or `method="ode"` or `"ssa"` on the generated network. NFsim runs a table
function indexed by time, but refreshes its value only when a reaction fires. So
a rate that is 0 at the start, or that changes while nothing fires, is not seen
until something does, and until then the run is flat (issue #892).

NFsim also keeps a different clock. A `method="nf"` run over `t_span=(5, 7)`
starts from the initial state and runs for 2 time units with NFsim's clock going
from 0 to 2, as BNG2.pl's `simulate_nf` does, and bngsim labels the rows 5 to 7.
The difference from RuleMonkey shows only where NFsim reads the clock, which is
through a time-indexed table function.

### What RuleMonkey flags without refusing

RuleMonkey refuses an XML it cannot honor. Some things it runs and flags
instead, and bngsim raises each as a `bngsim.RuleMonkeyWarning` when the
`method="rm"` simulator or the session is built. Among them:

- a rule whose products can never be what it writes, such as
  `A(s~0) + B(a) -> A(s~1).B(a)`, which puts A and B in one complex but bonds
  nothing. BioNetGen generates no reaction for such a rule, and RuleMonkey never
  fires it;
- a keyword it ignores (a rule `priority`, `MoveConnected`);
- a construct it reads differently from BioNetGen's network (`TotalRate`,
  `reactant_N()` outside `TotalRate`, a Michaelis-Menten pattern that matches
  several species).

```python
import warnings, bngsim
warnings.simplefilter("error", bngsim.RuleMonkeyWarning)   # promote
```

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

A RuleMonkey session runs on the model's clock. `initialize(seed, t_start=0.0)`
puts the initial state at `t_start`, `current_time` reads the clock, and a
segment that starts at `current_time` continues on it:

```python
with RuleMonkeySession("model.xml") as rm:
    rm.initialize(seed=42, t_start=5.0)      # time() reads 5 here
    first = rm.simulate(5, 7, n_points=3)
    second = rm.simulate(7, 10, n_points=4)  # continues at 7
```

A segment that starts anywhere else, such as a second `simulate(0, 10, ...)` on
a session already at 10, is run from the current state over the same span and
labelled with the times asked for. That is exact for a model that does not read
the clock, where only elapsed time matters. For a model that does, the rates
would see times the rows do not carry, so the segment is refused with
`SimulationError`.

A session may be advanced in as many calls as needed: a run cut into short
segments has the same statistics as one call. It does not spend the same random
numbers, so the two are not identical row for row. `solver_stats["n_steps"]` on
a session `Result` is the session's running event count, on both backends.

`current_time` is the time of the last thing the engine did. It stops short of a
segment's end where nothing more can fire, or the molecule limit is reached,
after `simulate` and after `step_to` alike, and the next segment of a model that
reads the clock starts there. `step_to(t)` takes a time on the session's clock
and does not go back.

**PyBNF integration**: When `bngsim` is installed, `method=>"nf"` / `"nfsim"`
routes through the in-process NFsim backend, and `method=>"rm"` / `"rulemonkey"`
/ `"nf_exact"` routes through RuleMonkey when RuleMonkey support is compiled in.
