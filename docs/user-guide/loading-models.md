---
summary: "Load a model from a file, a string or BioModels: .net, BNGL (through BNG2.pl), SBML and Antimony. What each loader needs and refuses, Model.describe, and reading a .net file as a dict."
---

# Loading models

This page covers how to load a model, how to see what you loaded, and what
each format needs.

## Load a model

`Model.load` picks the loader from the file's suffix:

```python
import bngsim

model = bngsim.Model.load("model.net")    # Model.from_net
model = bngsim.Model.load("model.xml")    # Model.from_sbml (.sbml too)
model = bngsim.Model.load("model.ant")    # Model.from_antimony
model = bngsim.Model.load("model.bngl")   # Model.from_bngl (needs BNG2.pl)
```

The suffix match ignores case. An unknown suffix raises `ModelError`, which
lists the suffixes BNGsim knows. If the suffix doesn't match the contents, such
as an SBML file saved as `.txt`, call the loader for the format yourself.
`Model.load` passes `defer_jacobian=` on to the loader.

SBML and Antimony also load from a string, and BioModels entries load by id:

```python
model = bngsim.Model.from_antimony_string("""
    S1 = 100; S2 = 0;
    k1 = 0.1; k2 = 0.05;
    J1: S1 -> S2; k1 * S1;
    J2: S2 -> S1; k2 * S2;
""")
model = bngsim.Model.from_sbml_string(sbml_xml_text)
model = bngsim.Model.from_biomodels("BIOMD0000000010")
```

Every model runs the same way after it loads:

```python
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=101)
```

Watch out for:

- **BNGL needs BioNetGen.** BNGsim runs BNG2.pl to turn rules into a network,
  so `.bngl` files need BNG2.pl and Perl. See [BNGL models](#bngl-models).
- **Antimony needs an extra:** `pip install "bngsim[antimony]"`. SBML needs
  `python-libsbml>=5.20`, which comes with BNGsim.
- **`Model.load` reads files only.** `Model.load("BIOMD0000000010")` fetches
  nothing. Its error tells you to call `Model.from_biomodels`.
- **`compartment_sizes=`** sets compartment volumes for SBML and Antimony
  models. It is refused for `.net` and `.bngl`, because BNG2.pl folds the
  volumes into the rate constants. Change those in the BNGL source.

## See what you loaded: `Model.describe()`

`Model.describe()` tells you where the model came from, what its species and
parameters are, what each species' number means, and which methods it can
run. It returns a dict you can pass to `json.dumps`, and `print` shows it as a
report. It runs nothing and doesn't change the model.

```python
import bngsim
from pathlib import Path

Path("decay.net").write_text(
    "begin parameters\n 1 k 0.5\nend parameters\n"
    "begin species\n 1 A() 10\n 2 B() 0\nend species\n"
    "begin reactions\n 1 1 2 k\nend reactions\n"
    "begin groups\n 1 Atot 1\nend groups\n"
)
model = bngsim.Model.from_net("decay.net")
print(model.describe())

d = model.describe()
d["methods"]["ssa"]["status"]  # 'ok'
d["species"][0]                # {'name': 'A()', 'initial': 10.0, 'fixed': False}
```

prints

```text
Model from net: /path/to/decay.net (sha256 e2e0d0740901)
Size: 2 species, 1 reaction, 1 observable, 1 parameter
Next run starts from: the initial state
Species values are as the net file gives them; it states no units.
Methods:
  ode, ssa, psa: ok
  nf, rm: needs: [needs_bng_xml] The network-free methods read the model's rules from a BNG-XML file, passed as xml_path=..., not from the loaded network. BNG2.pl writes one from the .bngl a network came from, with the writeXML() action.
  forward sensitivities: undecided (backend: cc): nothing refuses them before a run; whether the rate laws differentiate is decided when a sensitivity run builds its code
  steady state: ok, if a solve converges (the call also checks the conservation laws)
Analytical Jacobian: complete
Time: nothing reads the clock
Species (2):
  A()  10
  B()  0
Parameters (1):
  k = 0.5  primary
Observables (1):
  Atot = A()
Reactions (1):
  A() -> B()  elementary  k
```

The main fields:

- **`source`**: the format, the absolute path and the SHA-256 of the file (or
  of the text, for a string load). A `.bngl` load adds the generated `.net`; a
  BioModels load adds the id and version. A model built in Python has none.
- **`species_values`**: what a species' number means. For SBML and Antimony it
  is a concentration, the amount divided by the compartment size, as
  `get_concentration` and `Result.species` give it. A species declared as an
  amount also has `quantity: "amount"` and its `initial_amount`. A `.net` or
  BNGL file states no units, so this is `None`, and its species have no
  `quantity` or `compartment` key: they are absent, not guessed.
- **`reactions`**, **`parameters`** and **`functions`**: as the file declares
  them. A local parameter is named `_lp_<reaction>_<id>`, and a function
  definition a law calls is listed under `function_definitions`. BNGsim may run one reaction as several channels; `Model.n_reactions`
  counts those. A size or parameter that a rule or event changes is listed
  under `other_states`.
- **`methods`**: a `status` and `reasons` for `ode`, `ssa`, `psa`, `nf`, `rm`
  and `steady_state`: `"ok"`, `"refused"` (each construct, with a `code` and
  the message the run would raise), `"needs_input"` or `"unavailable"`. The
  reasons come from the same checks the Simulator makes. A run can still fail
  on its arguments or starting values: a rate that is infinite or NaN at the
  start fails a stochastic run, though `describe()` says `"ok"`.
- **`forward_sensitivities`** is never `"ok"`. It is `"refused"` for a delayed
  event, `"unavailable"` without a C compiler, and otherwise `"undecided"`,
  because a rate law that can't be differentiated, such as one with `floor()`,
  is found only when the run builds its code. `min`, `max`, `abs` and
  comparisons do differentiate. `refused_parameters` lists the parameters a
  sensitivity run always refuses: those folded at load, the compartment sizes
  `set_param` can't write, and the parameters a rule or function sets. Any
  other compartment size gets a d/dV column.
- **`steady_state`** is `"refused"` for a model with an event, a compartment
  size that jumps, a state that reads time, or no species. `"ok"` doesn't
  promise that a solve converges.
- **`next_run_starts_from`**: `"live"` after a run or a `set_concentration`,
  when the next `run()` continues from the current state.

Lists longer than `limit` (default 50) are cut, and `omitted` says how many
were left out. An SSA refusal is listed once per kind, with a count, and an
expression is cut at 200 characters. `describe(limit=None)` lists everything.

## BNGL models

BNGsim simulates reaction networks. BNGL describes rules. BNG2.pl turns rules
into a network, so `Model.from_bngl` runs BNG2.pl:

```python
model = bngsim.Model.from_bngl("egfr.bngl")
```

You need BNG2.pl and Perl. The extra installs PyBioNetGen, which bundles
BNG2.pl:

```bash
pip install 'bngsim[bngl]'
```

Or point to a BioNetGen you already have:

```bash
export BNGPATH=/path/to/BioNetGen-2.9.3
```

BNGsim looks in this order: the `bng2_pl=` argument, `$BNG2_PL`, `$BNGPATH`,
`BNG2.pl` on `PATH`, then an installed PyBioNetGen. An environment variable
wins over an installed package. `perl` must be on `PATH`. macOS and most Linux
systems have it; stock Windows doesn't, so BNGL loading is not available there.
`bngsim.HAS_BNGL` checks both BNG2.pl and Perl when you ask. When it is `False`,
`bngsim.capabilities()["missing"]["bngl"]` lists every place BNGsim looked:

```python
if not bngsim.HAS_BNGL:
    print(bngsim.capabilities()["missing"]["bngl"])
```

### The actions block

`Model.from_bngl` doesn't run the file's experiment. It runs the actions up to
the first `generate_network`, with that action's own `max_iter`, `max_agg` and
`max_stoich`, and stops. If there is no `generate_network` before the first
run, it stops at that run.

- Actions before that point are applied, as BNG2.pl applies them. A
  `setParameter` or `setConcentration` there changes the network you get. File
  writes, `setOutputDir` and `quit` are skipped.
- Actions from that point on are dropped.

If the dropped actions change a parameter or a concentration before the first
run (`simulate`, `parameter_scan` or `bifurcate`), the model you get is not the
one the author first simulated. `from_bngl` warns with a `ConversionWarning`
that names those actions. It names only actions that change the starting
state; a value set and then set back is not one. Pass `protocol=True` to get
the actions as a `ProtocolSpec`:

```python
model, spec = bngsim.Model.from_bngl("egfr.bngl", protocol=True)
[e.t_span for e in spec.experiments]   # the simulate(...) calls
```

See [Interchange](interchange.md) for what a `ProtocolSpec` holds.

### Run the file's own experiment

`bngsim.run_bngl` loads a file and runs its actions in order, as BNG2.pl does.
`bngsim.run_protocol(model, spec)` runs a spec you already have on a copy of
the model:

```python
out = bngsim.run_bngl("egfr_path.bngl")   # equilibrate without EGF, then add it
out[1].result.observables["Sos_act"]      # the second run
out.model                                 # the model in the state the file leaves it
```

There is one entry per `simulate`, `parameter_scan` or `bifurcate`. Index by
position or by a run's `suffix`. A scan's points are in `scan_results`; a
bifurcation's sweep back down is in `backward_results`. The runs follow
BNG2.pl:

- Each run starts from the state the last one left. With `continue=>1` it also
  starts at the time the last one ended; otherwise at its `t_start`, 0 by
  default.
- A parameter set to an expression, `setParameter("k2","k1*3")`, follows its
  inputs until it is set to a number. So does a species whose amount is a
  parameter or an expression, until a run makes it a number. Setting a number
  ends an expression, one the network declares too, even when the number is
  what the expression gives: with `k2 k1*0.5` declared, `setParameter("k2",0.5)`
  keeps k2 at 0.5 when `k1` moves, as BNG2.pl does. A parameter defined through
  itself is refused; an expression a number ended doesn't count for that.
- `out.model` keeps the values the actions left. `set_param` on it follows its
  own rule: a derived parameter that holds the number its expression gives
  still follows that expression, and one set to an expression holds a number.
- `saveConcentrations`/`resetConcentrations` and
  `saveParameters`/`resetParameters` keep one saved state per label. A reset to a declared value
  that an action before `generate_network` changed is refused.
- `parameter_scan` resets before each point, and again at the end, when
  `reset_conc` is on (the default). The scanned parameter keeps its last value. `bifurcate` never
  resets.
- ODE runs use 1e-8 absolute and relative tolerance unless the action or the
  call sets others. `seed=` seeds stochastic runs.
- A file with no model block, such as the actions file `omex_to_net` writes,
  runs every action on the network you give it.

These are refused before anything runs: a network-free (NFsim) step; PLA;
run arguments such as `stop_if` or `max_sim_steps`; an action that changes
results but has no step to run, such as `setVolume`; and an argument the
protocol can only hold approximately. `allow_lossy=True` runs the last two as
closely as it can, with a warning. A run with no `method` runs as ODE, and a
`steady_state=>1` run that doesn't converge warns and goes on; BNG2.pl refuses
both.

### The network cache

BNGsim keeps each generated `.net` under `~/.cache/bngsim/networks` (set
`$BNGSIM_BNGL_CACHE_DIR` to move it) and reuses it when you load the same model
again. The cache key covers the model text and the BioNetGen that made it, so
editing the model or upgrading BioNetGen regenerates the network. Editing only
the actions after generation reuses it. A model that reads another file before
`generate_network` (`readFile`, `readNFspecies`) is generated fresh every time,
because the key can't see that file. A relative path it reads is resolved
from the model's directory.

To keep the `.net` yourself, pass `net_out=`:

```python
model = bngsim.Model.from_bngl("egfr.bngl", net_out="egfr.net")
```

`cache=False` generates into a temporary directory for this process.

### Compartments and errors

Compartmental BNGL (cBNGL) models load. BNG2.pl builds each compartment's
volume into the rate constants. So `set_param` refuses to change a volume, and
a sensitivity for one is refused: the parameter is in the model, but no rate
reads it. `Model.frozen_params` lists these parameters.

A model whose network never stops growing hits `timeout=` (600 s by default)
and raises `ModelError`. A model BNG2.pl rejects raises `ModelError` with the
end of BNG2.pl's output, which is where a BNGL syntax error is reported.

## SBML and Antimony models

BNGsim reads SBML with libsbml and Antimony with libantimony. It handles
compartments, boundary species, initial assignments, piecewise functions and
function definitions.

**Antimony with several modules.** BNGsim loads the main module, the one
libantimony names and RoadRunner runs: the model marked `model *m()`, else the
top level if a statement sits outside every model, else the last model
defined. A module instantiated at the top level (`A: sub()`) loads with its
instance's names: `A_S1` for `sub`'s `S1`. If a stray top-level statement,
such as a `unit` line, makes the top level the main module, and nothing there
changes over time (no reaction, event or rate rule, and no rule that reads
time), BNGsim loads the last model defined instead. A `ConversionWarning` names anything the top
level held that was left out.

**SBML Level 3 packages.** A document that marks a package
`required="true"` is refused by name unless BNGsim supports that package.
`comp` loads; the model is flattened first. `distrib` loads, and its random
draws are refused one by one. `multi`, `qual`, `spatial` and the rest are
refused, because reading only the core species and reactions would build a
different model. Packages marked `required="false"`, such as `layout`,
`render` and `fbc`, load as if they weren't there.

## Models from BioModels

`Model.from_biomodels` loads a BioModels entry by its id:

```python
model = bngsim.Model.from_biomodels("BIOMD0000000010")
```

BNGsim downloads the entry's main SBML file and checks it against the sha256
BioModels publishes. It keeps a copy in `~/.cache/bngsim/biomodels` (or
`$BNGSIM_BIOMODELS_CACHE_DIR`), so the next call needs no network.

- **Versions.** You get the latest version, or the latest one already cached.
  `refresh=True` fetches again and updates the cache. `version=2` pins a
  version.
- **No cache.** `cache=False` fetches the file and keeps nothing.
- **Checking.** If BioModels publishes no sha256 for a file, BNGsim only checks
  that it is SBML.
- **Source.** `model.describe()["source"]` records the id, version, file name
  and sha256.
- **No network.** If BioModels can't be reached and nothing is cached, the
  error says so and names the cache directory. BioModels refuses some cloud
  hosts. There, download the SBML yourself and use `Model.from_sbml(path)`.

## `.net` files

`Model.from_net` reads BioNetGen's network format. It refuses a file it can't
read correctly, rather than guess:

- **Species index.** A species line is `[<index>] <species> [<concentration>]`.
  The index may be left off. If it's there, it must equal the line's position:
  1, 2, 3, ... BioNetGen's two readers disagree when it doesn't, so BNGsim
  refuses the file.
- **Reaction lines.** A line is `<index> <reactants> <products> <rate law>`.
  Only `Sat`, `Hill` and `MM` take more tokens, their constants. Anything else
  after the rate law is refused, as `run_network` refuses it. BNGsim reads no
  per-reaction modifiers, such as a delay. A factor before the type
  (`2*MM kcat Km`) is read as the reaction's statistical factor.
- **`MM` reactions** take two reactants, an enzyme and a substrate. More are
  refused. To model several enzymes on one substrate, write a functional rate
  law over the enzymes' total, such as `S -> P` at
  `kcat*Etot*r()/(Km+r()*Sobs)`, where `r()` is the free fraction of the
  substrate, `2*Km/(sqrt((Sobs-Km-Etot)^2+4*Km*Sobs)-(Sobs-Km-Etot))`. Writing
  it with the fraction avoids 0/0 when `S` is 0.
- **`Sat` and `Hill`** are deprecated. BNGsim rewrites them as functional rate
  laws with a warning, adding a one-species observable for each reactant the
  function reads. A repeated reactant (`A() + A()`) is read position by
  position, as `run_network` reads it.
- **`Hill` with 0 < h < 1.** The function `Vmax*S^(h-1)/(Kh^h + S^h)` is
  infinite at S = 0 when 0 < h < 1. BNGsim guards it so the rate is 0 there,
  whatever h you set later. `Model.state_switch_root_conditions()` names
  `S > 0` at every h, and a model with events keeps every root. `run_diffrax`
  refuses such a model while h is in (0, 1).
- **Empty files.** A file with none of the parameters, species, functions,
  reactions and groups blocks is refused, as `run_network` refuses it. An empty
  block is fine.
- **Line endings.** LF and CRLF both work. A bare carriage return (classic Mac
  OS) is read as a space, as BioNetGen reads it, so the file is one line and is
  refused with an explanation.
- **Encoding.** The file must be UTF-8. A line in the five blocks that isn't,
  such as a Latin-1 `µ` in a species name, is refused, with the byte shown as
  `\xb5`. Save the file as UTF-8. A bad byte in a comment or action line is
  ignored. `write_net` always writes UTF-8.

### Read a `.net` file as a dict

`parse_net_file` returns a `.net` file's contents as a Python dict you can
inspect, change, build, or hand to another tool. It is the same reading
`Model.from_net` uses, with the values the built model holds, and it refuses
the same files. It needs BNGsim's compiled extension.

```python
import bngsim

parsed = bngsim.parse_net_file("model.net")

print(parsed["parameters"])   # [(name, value, expr, is_expr), ...]
print(parsed["species"])      # [(name, init_conc, is_fixed), ...]
print(parsed["species_ic_params"])  # [(sp_idx, param_name), ...]: initial
                              #   values written as a parameter name
print(parsed["observables"])  # [(name, [(sp_idx, factor), ...]), ...]
print(parsed["functions"])    # [(name, expression), ...]
print(parsed["reactions"])    # [{"reactants": [...], "products": [...],
                              #   "type": "elementary"|"functional"|"mm",
                              #   "rate_law": "k1", "legacy_constants": [],
                              #   "stat_factor": 1.0}, ...]
print(parsed["net_file_dir"]) # the file's directory, which a relative
                              #   tfun('...') path is read from
```

A reaction's `"type"` says how to read its `"rate_law"`:

- `"elementary"`: a parameter name. A numeric prefix such as `0.5*k1`, a
  symmetry factor or a unit conversion, comes back as `"rate_law": "k1"` and
  `"stat_factor": 0.5`. The rate constant is `stat_factor` times the
  parameter.
- `"functional"`: a function name. The function is a rate constant too: the
  rate is `stat_factor` times the function times the reactant amounts.
- `"mm"`: `"<kcat>,<Km>"`, with `"legacy_constants": [kcat, Km]`. A factor
  before the type, `2*MM kcat Km`, comes back as `"stat_factor": 2.0` and
  multiplies the whole rate. `Sat` and `Hill` come back rewritten as
  `"functional"`, keeping their factor the same way.

Values are numbers: a parameter's value is the one `Model.from_net` uses, and a
species whose initial value names a parameter carries that parameter's value.
An initial value written as an expression comes back as a parameter named
`_InitialConc<N>`, as BNG2.pl writes it.

**Build a model from the dict:**

```python
model = bngsim.build_model_from_parsed(parsed)
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=101)
```

An unchanged dict builds the same model `Model.from_net` loads, table
functions and rewritten `Sat` and `Hill` laws included. When you change one:

- A species in `species_ic_params` takes its initial value from that
  parameter. Change the parameter, or drop the entry, not the number in
  `species`.
- A parameter whose `is_expression` is true is computed from its expression.
  Change the expression, or set `is_expression` to false and give a value.
- A rate law that names a function in `functions` is functional; anything else
  is elementary (`"mm"` aside), whatever `"type"` says.

**Use it with SciPy:**

```python
import numpy as np
from scipy.integrate import solve_ivp

parsed = bngsim.parse_net_file("model.net")
y0 = np.array([ic for _, ic, _ in parsed["species"]])
pvals = {n: v for n, v, _, _ in parsed["parameters"]}
fixed = [i for i, (_, _, is_fixed) in enumerate(parsed["species"]) if is_fixed]

# Your own right-hand side, for mass-action reactions only
def rhs(t, y):
    dydt = np.zeros(len(y))
    for rxn in parsed["reactions"]:
        if rxn["type"] != "elementary":
            raise NotImplementedError(rxn["type"])
        rate = rxn["stat_factor"] * pvals[rxn["rate_law"]]
        for ri in rxn["reactants"]:
            rate *= y[ri]
        for ri in rxn["reactants"]:
            dydt[ri] -= rate
        for pi in rxn["products"]:
            dydt[pi] += rate
    dydt[fixed] = 0.0  # a `$`-fixed species holds its value
    return dydt

sol = solve_ivp(rhs, (0, 100), y0, method='LSODA')
```
