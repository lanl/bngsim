# Loading models

## `Model.load` — dispatch on the file suffix

When you already have a path, `Model.load` picks the right factory for you:

```python
model = bngsim.Model.load("model.ant")    # -> Model.from_antimony
model = bngsim.Model.load("model.xml")    # -> Model.from_sbml  (.sbml too)
model = bngsim.Model.load("model.net")    # -> Model.from_net
model = bngsim.Model.load("model.bngl")   # -> Model.from_bngl  (needs BNG2.pl)
```

Matching is case-insensitive, and `defer_jacobian` is forwarded to the selected
factory. An unrecognized suffix raises `ModelError` listing the loadable ones.
The format-specific factories below remain the explicit route when the suffix
does not match the contents (for example an SBML document saved as `.txt`).

## BNGL model loading

BNGsim simulates reaction *networks*; BNGL describes *rules*. Turning one into
the other is network generation — BNG2.pl's job — so `Model.from_bngl` runs it
rather than parsing BNGL itself:

```python
model = bngsim.Model.from_bngl("egfr.bngl")
```

That needs BNG2.pl on the machine. Either install the extra, which brings
PyBioNetGen (BNG2.pl is bundled with it):

```bash
pip install 'bngsim[bngl]'
```

...or point at a BioNetGen you already have. The resolution order is
`bng2_pl=` argument → `$BNG2_PL` → `$BNGPATH` → `BNG2.pl` on `PATH` → installed
PyBioNetGen, so an environment variable always overrides an installed package:

```bash
export BNGPATH=/path/to/BioNetGen-2.9.3
```

BNG2.pl is a Perl script, so `perl` must also be on `PATH` — macOS and most
Linux distributions ship one; stock Windows does not, and BNGL loading is simply
unavailable there. `bngsim.HAS_BNGL` is the probe for the whole arrangement
(both halves, checked at the moment you ask), and when it is `False`,
`bngsim.capabilities()["missing"]["bngl"]` names every location that was
searched:

```python
if not bngsim.HAS_BNGL:
    print(bngsim.capabilities()["missing"]["bngl"])
```

### The actions block is not executed

A `.bngl` in the wild ends in `simulate({...})` or `parameter_scan({...})`.
Running the file as written would run the author's entire experiment just to
obtain a network. So the actions stop at the source's first `generate_network`,
which runs with the source's own `max_iter` / `max_agg` / `max_stoich`, since
those are what make an unbounded rule set finite. A file with no
`generate_network` before its first run stops at that run, the state the run
starts from.

- **Before that point:** actions are applied, as BNG2.pl applies them. A
  `setParameter` or `setConcentration` there seeds the network. Only file writes,
  `setOutputDir` and `quit` are left out.
- **At and after it:** everything is dropped, as the author's experiment.

If that experiment sets a parameter or a concentration before its first run
(`simulate`, `parameter_scan` or `bifurcate`), the model you get is not the one
the author first simulated. `from_bngl` warns with a `ConversionWarning` naming
those actions. It names only the actions that change the state the run starts
from: a `setConcentration` to the amount the species already holds is not one,
and neither is a value set and then set back, or a `saveConcentrations()`
followed by `resetConcentrations()`. Pass `protocol=True` to recover the actions
instead of losing them:

```python
model, spec = bngsim.Model.from_bngl("egfr.bngl", protocol=True)
[e.t_span for e in spec.experiments]   # the simulate(...) calls, as a ProtocolSpec
```

See [Interchange](interchange.md) for what a `ProtocolSpec` carries.

### Running the file's own experiment

`bngsim.run_bngl` loads a file and runs its actions in order, as BNG2.pl runs
the file. `bngsim.run_protocol(model, spec)` runs a spec you already hold on a
copy of the model:

```python
out = bngsim.run_bngl("egfr_path.bngl")   # equilibrate without EGF, then add it
out[1].result.observables["Sos_act"]      # the 120 s stimulation, as BNG2.pl runs it
out.model                                 # the model in the state the file leaves it
```

There is one entry per `simulate`, `parameter_scan` or `bifurcate`. You can
index the result by position, or by a run's `suffix`. A scan's points are in
`scan_results`, one `Result` per value. A bifurcation's sweep back down is in
`backward_results`. The steps follow BNG2.pl:

- **State between runs.** Each run starts from the state the last one left.
  `continue=>1` also starts at the time it left; any other run starts at its
  `t_start`, 0 by default.
- **Values.** A species whose amount is a parameter or an expression follows it
  until a run makes it a number. `setParameter`, `setConcentration` and
  `addConcentration` change the live model.
- **Saved states.** `saveConcentrations`/`resetConcentrations` and
  `saveParameters`/`resetParameters` keep one saved state per label.
- **Scans.** `parameter_scan` resets to its starting state before each point when
  `reset_conc` is on (the default), and again at the end. The scanned parameter
  keeps its last value. `bifurcate` never resets.
- **Tolerances.** ODE runs use `run_network`'s tolerances, 1e-8 absolute and
  relative, unless the action or the call gives others. `seed=` seeds the
  stochastic runs.

What bngsim cannot run as written is refused before anything runs:

- a network-free (NFsim) step, which needs a live NFsim session (issue #1482);
- PLA, which bngsim does not have;
- a run argument such as `stop_if` or `max_sim_steps`;
- an action that changes results but has no step to run, such as `setVolume`;
- an argument the protocol holds only approximately.

`allow_lossy=True` runs the last two as nearly as the protocol holds them, with
a warning. Two things BNG2.pl refuses run here instead: a run that names no
`method` runs as ODE, and a `steady_state=>1` run that does not reach a steady
state is warned about while the protocol goes on, where BNG2.pl stops it.

### Generated networks are cached

Network generation is the expensive step, so the emitted `.net` is kept under
`~/.cache/bngsim/networks` (override with `$BNGSIM_BNGL_CACHE_DIR`) and reused
when the same BNGL is loaded again. The key is a digest of the flattened model
text *and* the BioNetGen that produced it (BNG2.pl, and the `Perl2` library and
`VERSION` it loads, from `$BNGPATH` when that is set), so the cache cannot go
stale: editing the model — or upgrading BioNetGen — regenerates, while editing
only the actions block correctly reuses. A model that reads another file before
`generate_network` — a parameter set with `readFile`, seed amounts with
`readNFspecies`, in the actions or inside `begin model` — is generated afresh on
every load, since its network depends on that file and the key cannot see it
(issue #1494). A relative path it reads resolves against the model's directory.

`Model.from_net` records the path of the network it loaded, so the file is kept
rather than deleted with a scratch directory. To keep the `.net` as an artifact
of your own, pass `net_out=`:

```python
model = bngsim.Model.from_bngl("egfr.bngl", net_out="egfr.net")
```

`cache=False` regenerates into a per-process directory instead.

### Compartments and errors

Compartmental (cBNGL) models load: BNG2.pl bakes each compartment's volume into
the generated rate constants, exactly as for a hand-generated `.net`. For that
reason `Model.load` refuses `compartment_sizes=` on `.bngl` as it does on
`.net` — the volume has to change in the BNGL source, before generation.
`set_param` refuses to change a volume for the same reason, and a sensitivity
column for one is refused (issue #711): the parameter is still in the model, and
no rate reads it. `Model.frozen_params` lists them.

A model whose network is unbounded hits `timeout=` (600 s by default) and gets a
`ModelError` saying so; a model BNG2.pl rejects gets one carrying the tail of
BNG2.pl's own output, which is the only thing that can localize a BNGL syntax
error.

## Antimony and SBML model loading

BNGsim can load models from Antimony (`.ant`) and SBML (`.xml`) files in addition
to BNG `.net` files. This uses libantimony for parsing and libsbml for correct
SBML semantics (compartments, boundary species, initial assignments, piecewise
functions, function definitions).

```python
# Load from Antimony file
model = bngsim.Model.from_antimony("model.ant")

# Load from Antimony string
model = bngsim.Model.from_antimony_string("""
    S1 = 100; S2 = 0;
    k1 = 0.1; k2 = 0.05;
    J1: S1 -> S2; k1 * S1;
    J2: S2 -> S1; k2 * S2;
""")

# Load from SBML file
model = bngsim.Model.from_sbml("model.xml")

# Load from SBML XML string
model = bngsim.Model.from_sbml_string(sbml_xml_text)

# All model types support the same simulation API
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=101)
```

Antimony loading requires `bngsim[antimony]`. Direct SBML loading requires
`python-libsbml>=5.20` (installed automatically with the base package).

**SBML Level 3 packages.** A document that declares a package `required="true"`
— SBML's way of saying the package changes the mathematical meaning of the model
— is refused by name unless bngsim accounts for that package. `comp` does load
(the composition is flattened first); so does `distrib`, whose random draws the
math translator refuses individually. `multi`, `qual`, `spatial` and the rest are
refused, because reading only the core `<listOfSpecies>` / `<listOfReactions>` of
such a document builds a *different* model — a `multi` document's core species are
rule templates, and a `qual` document's core layer is typically empty. A
presentation-only package (`layout`, `render`, `fbc`) declares `required="false"`
and loads untouched.

## `.net` files as a dict (`parse_net_file`)

`parse_net_file` returns the components of a `.net` file as a plain Python
dict, which you can inspect, modify and build, or hand to another engine. It is
the C++ loader's own reading of the file, the one `Model.from_net` builds, with
the values the built model holds, so the two cannot disagree about what a line
means, and a file `Model.from_net` refuses is refused here too. It needs
bngsim's compiled extension.

```python
import bngsim

parsed = bngsim.parse_net_file("model.net")

print(parsed["parameters"])   # [(name, value, expr, is_expr), ...]
print(parsed["species"])      # [(name, init_conc, is_fixed), ...]
print(parsed["species_ic_params"])  # [(sp_idx, param_name), ...] — ICs written
                              #   as a parameter name rather than a number
print(parsed["observables"])  # [(name, [(sp_idx, factor), ...]), ...]
print(parsed["functions"])    # [(name, expression), ...]
print(parsed["reactions"])    # [{"reactants": [...], "products": [...],
                              #   "type": "elementary"|"functional"|"mm",
                              #   "rate_law": "k1", "legacy_constants": [],
                              #   "stat_factor": 1.0}, ...]
print(parsed["net_file_dir"]) # the file's own directory — what a relative
                              #   tfun('...') path resolves against
```

A reaction's `"type"` says how to read its `"rate_law"`:

- `"elementary"`: a parameter name. BNG2.pl writes a numeric prefix on the
  rate constant, `0.5*k1`, for a symmetry factor or, in a compartmental model,
  a unit conversion folded into it (`1.6605503e-12*kf`); the dict carries that
  as `"rate_law": "k1"` and `"stat_factor": 0.5`, so the rate constant is
  `stat_factor` times the parameter.
- `"functional"`: a name from the `functions` block. The function's value is a
  rate constant too: the rate is `stat_factor` times the function times the
  reactant amounts, as for an elementary reaction (`0.5*f1` comes back as
  `"rate_law": "f1"`, `"stat_factor": 0.5`).
- `"mm"`: `"<kcat>,<Km>"` for a `MM kcat Km` rate column, with
  `"legacy_constants": [kcat, Km]`. BNG2.pl writes a symmetry factor ahead of
  the type, `2*MM kcat Km`, and it comes back as `"stat_factor": 2.0`, which
  multiplies the whole rate (likewise `2*Sat k K` and `2*Hill V K h`, whose
  factor stays the rewritten reaction's `"stat_factor"`).

Values are evaluated: a parameter's is the number `Model.from_net` puts in its
slot, and a species whose initial concentration names a parameter carries that
parameter's value. An initial concentration written as an expression comes back
as a synthetic `_InitialConc<N>` parameter, as BNG2.pl writes one, named by the
species.

A species line is `[<index>] <species> [<concentration>]`. The index may be
left off, and when it is there it must be the line's own number, 1, 2, 3, ...
in the order the species are listed (issue #1512). Reactions and groups name a
species by that number, and BioNetGen's two readers disagree on a file whose
index is not its line: run_network places the species at its index, BNG2.pl's
`readFile` at its line. `parse_net_file` and `Model.from_net` refuse such a
file rather than pick one reading.

The deprecated `Sat` and `Hill` rate-law tokens come back rewritten, as
`Model.from_net` rewrites them, with the same warning: the reaction is
`"functional"`, driven by an explicit function, and a single-species observable
is added for each reactant the function reads. A factor ahead of the type
(`2*Sat k K`, `2*Hill V K h`) stays the reaction's `"stat_factor"`.

`Hill Vmax Kh h` is the law `Vmax*S^h/(Kh^h + S^h)`, and the reaction
multiplies its function by S, so the function is
`Vmax*S^(h-1)/(Kh^h + S^h)`. For 0 < h < 1 that is infinite at S = 0, where
the law is 0. So when h is in (0, 1) at load, or is an expression parameter,
the function is written `if((S > 0) or (h <= 0) or (h >= 1), ..., 0)`. It is 0
for S <= 0 while h is in (0, 1), and the run roots on `S > 0`. An h of 1 or
more, or of 0 or less, at load keeps the unguarded function and adds no root.
A model whose h is moved into (0, 1) after loading still fails at S = 0. That
includes `set_param`, and a BNGL `setParameter` or `parameter_scan` after
`generate_network`. Give h a value in (0, 1) before the network is loaded: the
guard reads h when the model runs, so later values of 1 or more keep the
unguarded function's values.

A reaction line is `<index> <reactants> <products> <rate law>`, and only a
`Sat`, `Hill` or `MM` rate law takes more tokens: its constants.
`parse_net_file` and `Model.from_net` refuse a line with anything else after
the rate law, as run_network does, rather than load it without that part.
bngsim reads no per-reaction modifiers, such as a delay (issue #505). A stat
factor ahead of the rate law's type (`2*MM kcat Km`) is read off the first
token before the type is matched, as run_network reads it.

An `MM` reaction reads two reactants, an enzyme and a substrate, and one with
more is refused (issue #1502): bngsim left the others out of its rate, so the
answer changed with the order the enzymes were listed in. BNG2.pl refuses such a
rule, and run_network reads the first reactant as the substrate and the total of
the others as the enzyme. To run that, write the reaction with a functional rate
law over the enzymes' total, such as `S -> P` at
`kcat*sFree()*Etot/(Km+sFree())/Sobs`, with `sFree()` the free substrate
`0.5*((Sobs-Km-Etot)+sqrt((Sobs-Km-Etot)^2+4*Km*Sobs))`.

A file with none of the parameters, species, functions, reactions and groups
blocks is refused, as run_network refuses it: a 0-byte file, one of comments or
actions only (issue #1511). A block that is there and empty still loads. Lines
end at a newline, LF or CRLF. A bare carriage return is read as BNG2.pl and
run_network read it, as a blank inside its line, so a file saved with classic
Mac OS line endings is one line, and is refused, with a word on why.

**Use with BNGsim** (fastest path — C++ CVODE/SSA):

```python
model = bngsim.build_model_from_parsed(parsed)
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=101)
```

`build_model_from_parsed` makes the `ModelBuilder` calls the loader makes, so an
unmodified dict builds the model `Model.from_net` loads, table functions and
rewritten `Sat`/`Hill` rate laws included; a test pins that over every `.net`
file in the repository. A modified dict builds the modified model. Three things
to know when modifying one:

- A species listed in `species_ic_params` takes its initial concentration from
  that parameter, so change the parameter, or drop the entry, rather than the
  number in `species`.
- A parameter whose `is_expression` is true is evaluated from its expression, so
  change the expression, or set `is_expression` false with a value.
- A reaction's rate law is read by what it names: a function in `functions`
  makes it functional, anything else elementary (`"mm"` aside), whatever its
  `"type"` says.

**Use with scipy**:

```python
import numpy as np
from scipy.integrate import solve_ivp

parsed = bngsim.parse_net_file("model.net")
y0 = np.array([ic for _, ic, _ in parsed["species"]])
pvals = {n: v for n, v, _, _ in parsed["parameters"]}
fixed = [i for i, (_, _, is_fixed) in enumerate(parsed["species"]) if is_fixed]

# Build your own RHS from the parsed data (mass-action reactions only)
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
    dydt[fixed] = 0.0  # a `$`-clamped species holds its value
    return dydt

sol = solve_ivp(rhs, (0, 100), y0, method='LSODA')
```

**Use with gillespy2** (Python SSA):

```python
import gillespy2

parsed = bngsim.parse_net_file("model.net")
m = gillespy2.Model(name="my_model")
for name, val, _, _ in parsed["parameters"]:
    m.add_parameter(gillespy2.Parameter(name=name, expression=str(val)))
for name, ic, _ in parsed["species"]:
    m.add_species(gillespy2.Species(name=name, initial_value=int(ic)))
# ... add reactions from parsed["reactions"], each at stat_factor * rate_law
```
