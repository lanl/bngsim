---
summary: "What BNGsim does: formats, methods, analyses, results and interchange; what sets it apart; its engines; and the tools that use it."
---

# What BNGsim does

| | |
|---|---|
| **Model formats** | SBML and Antimony; BNGL, through BNG2.pl; BioNetGen's `.net` networks. The network-free methods read the XML that BioNetGen writes, passed as `xml_path` |
| **Methods** | ODE integration (SUNDIALS CVODE, with a dense or a sparse KLU linear solver); exact stochastic simulation (`ssa`); the partial-scaling approximation (`psa`); network-free simulation with NFsim (`nf`) and RuleMonkey (`rm`) |
| **Analyses** | forward sensitivities to parameters and to initial conditions, including through events (CVODES); Fisher information; steady states, with their sensitivities; conservation laws; parameter scans |
| **Running** | thread-parallel batches (`run_batch`) and replicates with the GIL released; stepwise runs (`run_until`) with state changes between steps; stop conditions; a wall-clock limit on runs, batches and scans; a BNGL file's own actions (`run_bngl`) |
| **Results** | NumPy arrays, with access by observable name; pandas, xarray and HDF5 through extras |
| **Interchange** | `.net` to and from SBML; SBML to compartmental BNGL; SED-ML protocols; COMBINE (`.omex`) archives. A conversion can be validated by a round-trip ladder, which is a check and not a proof ([interchange](https://bngsim.readthedocs.io/en/latest/user-guide/interchange.html)) |

## Engines

BNGsim's engines are written in C++. ODE integration and steady-state solving
use CVODE and KINSOL from SUNDIALS. Stochastic simulation uses BNGsim's own SSA
and PSA. Network-free simulation uses NFsim and RuleMonkey.

## Who uses it

[PyBioNetGen](https://github.com/RuleWorld/PyBioNetGen) and
[PyBNF](https://github.com/lanl/PyBNF) use BNGsim. PyBNF uses it for
parameterization, uncertainty quantification, and model checking
([guide](user-guide/pybnf.md)).

## What sets it apart

- **One library for deterministic, stochastic and network-free simulation.** A
  model loaded once can be integrated, run stochastically and solved for its
  steady state through one API, where the model admits each, with no
  translation between tools. For BNGL models, both network-free engines (NFsim
  and RuleMonkey) run in the same process as the rest.
- **It refuses what it cannot simulate faithfully.** A model with an algebraic
  rule, a `delay()` term, a fast reaction, or an SBML package that changes its meaning
  raises an error that names the construct. BNGsim does not drop the construct
  and integrate a different system. The same holds for sensitivities: where a
  derivative cannot be computed correctly, the run says so.
- **Its results are compared with other simulators' every night.** BNGL models
  are compared with BioNetGen's `run_network`, SBML models with libRoadRunner,
  sensitivities with AMICI and finite differences, and SBML semantics with the
  SBML test suite. A regression raises an alert. Results will be in
  [bngsim-benchmarks](https://github.com/lanl/bngsim-benchmarks)
  ([how BNGsim is checked](about/benchmarks.md)).
- **It is built to be embedded.** A run starts no subprocess and writes no
  result files. The ODE and stochastic engines hold no global state, so many
  models and simulators coexist in one process, and the GIL is released during
  a run. Some steps do reach outside a run. BNG2.pl expands a `.bngl` file into
  a network, which is cached on disk. The C compiler builds compiled code, which
  is cached too. An NFsim run has effects on the whole process, among them one
  shared random-number stream
  ([what an NFsim run changes](https://bngsim.readthedocs.io/en/latest/user-guide/network-free.html#what-an-nfsim-run-changes-in-your-process)).
  [PyBNF](https://github.com/lanl/PyBNF) and
  [PyBioNetGen](https://github.com/RuleWorld/PyBioNetGen) use BNGsim this way
  ([PyBNF guide](https://bngsim.readthedocs.io/en/latest/user-guide/pybnf.html)).
