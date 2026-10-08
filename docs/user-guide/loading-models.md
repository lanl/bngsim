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
obtain a network, so only `generate_network` is run — carrying the source's own
`max_iter` / `max_agg` / `max_stoich`, since those are what make an unbounded
rule set finite. Pass `protocol=True` to recover the actions instead of losing
them:

```python
model, spec = bngsim.Model.from_bngl("egfr.bngl", protocol=True)
[e.t_span for e in spec.experiments]   # the simulate(...) calls, as a ProtocolSpec
```

See [Interchange](interchange.md) for what a `ProtocolSpec` carries.

### Generated networks are cached

Network generation is the expensive step, so the emitted `.net` is kept under
`~/.cache/bngsim/networks` (override with `$BNGSIM_BNGL_CACHE_DIR`) and reused
when the same BNGL is loaded again. The key is a digest of the flattened model
text *and* the BNG2.pl that produced it, so the cache cannot go stale: editing
the model — or upgrading BioNetGen — regenerates, while editing only the actions
block correctly reuses.

Keeping the network is not only a speed choice. `Model.from_net` records the
path, and codegen prefers that file because a BNG2.pl network carries derived
rate-constant parameters whose chain rules the in-memory path does not
reconstruct. To keep the `.net` as an artifact of your own, pass `net_out=`:

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
  `"legacy_constants": [kcat, Km]`.

Values are evaluated: a parameter's is the number `Model.from_net` puts in its
slot, and a species whose initial concentration names a parameter carries that
parameter's value. An initial concentration written as an expression comes back
as a synthetic `_InitialConc<N>` parameter, as BNG2.pl writes one, named by the
species.

The deprecated `Sat` and `Hill` rate-law tokens come back rewritten, as
`Model.from_net` rewrites them, with the same warning: the reaction is
`"functional"`, driven by an explicit function, and a single-species observable
is added for each reactant the function reads.

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
