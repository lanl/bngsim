# bngsim

**Embeddable simulation engine for BioNetGen reaction networks.**

`bngsim` is a high-performance C++ simulation kernel with Python bindings that
replaces BioNetGen's subprocess-based `run_network` driver. It loads `.net`,
Antimony, and SBML models, runs ODE (CVODE) and stochastic (Gillespie/PSA)
simulations in-process, and returns results as NumPy arrays — no file I/O, no
subprocess spawning, no Perl dependency.

```python
import bngsim

model = bngsim.Model.from_net_file("model.net")
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_end=100.0, n_steps=1000)

print(result.times)              # (1001,) NumPy array
print(result["A"])               # trajectory of observable "A"
```

New here? Start with {doc}`installation` and the {doc}`quickstart`.

```{toctree}
:caption: Getting started
:maxdepth: 2

installation
quickstart
```

```{toctree}
:caption: User guide
:maxdepth: 1

user-guide/loading-models
user-guide/simulation
user-guide/network-free
user-guide/results
user-guide/events
user-guide/table-functions
user-guide/solvers
user-guide/codegen
user-guide/sensitivities
user-guide/steady-state
user-guide/conservation-laws
user-guide/interchange
user-guide/pybnf
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/api
reference/expressions
```

```{toctree}
:caption: About & development
:maxdepth: 1

about/architecture
about/benchmarks
development/building
development/extending
```
