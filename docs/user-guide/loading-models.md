# Loading models

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
