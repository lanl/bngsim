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

## Universal `.net` reader (`parse_net_file`)

BNGsim includes a pure-Python `.net` file parser that produces engine-agnostic
model data. This lets you use BNG `.net` files with **any** Python simulation
engine — BNGsim, scipy, gillespy2, or your own solver — without requiring
the BNGsim C++ extension for the parsing step.

```python
import bngsim

# Parse a .net file into a plain Python dict (no C++ needed)
parsed = bngsim.parse_net_file("model.net")

# Inspect the parsed data
print(parsed["parameters"])   # [(name, value, expr, is_expr), ...]
print(parsed["species"])      # [(name, init_conc, is_fixed), ...]
print(parsed["observables"])  # [(name, [(sp_idx, factor), ...]), ...]
print(parsed["functions"])    # [(name, expression), ...]
print(parsed["reactions"])    # [{"reactants": [...], "products": [...],
                              #   "type": "elementary"|"functional",
                              #   "rate_law": "k1", "stat_factor": 1.0}, ...]
```

**Use with BNGsim** (fastest path — C++ CVODE/SSA):

```python
model = bngsim.build_model_from_parsed(parsed)
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 100), n_points=101)
```

**Use with scipy** (pure Python, no C++ extension needed):

```python
import numpy as np
from scipy.integrate import solve_ivp

parsed = bngsim.parse_net_file("model.net")
y0 = np.array([ic for _, ic, _ in parsed["species"]])
pvals = {n: v for n, v, _, _ in parsed["parameters"]}

# Build your own RHS from the parsed data
def rhs(t, y):
    dydt = np.zeros(len(y))
    for rxn in parsed["reactions"]:
        rate = pvals[rxn["rate_law"]]
        for ri in rxn["reactants"]:
            rate *= y[ri]
        for ri in rxn["reactants"]:
            dydt[ri] -= rate
        for pi in rxn["products"]:
            dydt[pi] += rate
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
# ... add reactions from parsed["reactions"]
```

The parsed dict is the **universal interchange format** between `.net` files
and any Python-based simulation framework.
