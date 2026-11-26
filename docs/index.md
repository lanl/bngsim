---
summary: "What BNGsim is, and a model loaded and run in one block."
---

# BNGsim

**A simulator for biochemical reaction networks, as a Python library.**

BNGsim simulates models written in SBML, Antimony, and BNGL. It does ODE
integration, stochastic simulation (SSA, PSA), and network-free simulation
(NFsim, RuleMonkey). It also finds steady states, runs parameter scans, and
computes forward sensitivities. Models are simulated in-process, and results
are NumPy arrays. For BioNetGen users, it replaces `run_network`.

This runs as written after `pip install bngsim`: it writes a two-species
network to `model.net`, loads it, and integrates it.

```python
from pathlib import Path

import bngsim

# A -> B at rate k, in BioNetGen's network (.net) format.
Path("model.net").write_text(
    """\
begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 A 1
    2 B 2
end groups
""",
    encoding="utf-8",
)

model = bngsim.Model.from_net("model.net")
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0.0, 10.0), n_points=101)

print(result.time.shape)          # (101,): one entry per time point
print(result.observables["A"])    # trajectory of observable "A"
print(round(float(result.observables["A"][-1]), 4))  # 0.6738, which is 100 * exp(-5)
```

New here? Start with {doc}`installation` and the {doc}`quickstart`. An AI
agent, or anyone who would rather read rules than a tutorial, can start with
the {doc}`agent-guide`, which `bngsim.agent_guide()` also returns.

```{toctree}
:caption: Getting started
:maxdepth: 2

installation
quickstart
choosing
features
agent-guide
```

```{toctree}
:caption: User guide
:maxdepth: 1

user-guide/loading-models
user-guide/simulation
user-guide/network-free
user-guide/results
user-guide/checking
user-guide/events
user-guide/table-functions
user-guide/solvers
user-guide/codegen
user-guide/sensitivities
user-guide/steady-state
user-guide/conservation-laws
user-guide/evaluators
user-guide/interchange
user-guide/pybnf
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/api
reference/expressions
reference/error-codes
reference/command-line
```

```{toctree}
:caption: About & development
:maxdepth: 1

about/why
about/architecture
about/benchmarks
about/limitations
development/building
development/extending
development/sensitivity-terms
```
