# Architecture

```
┌─────────────────────────────────────────────┐
│  Python API (bngsim)                        │  pip install bngsim
│  Model, Simulator, Result                   │  NumPy arrays, logging
│  codegen, symbolic Jacobian, sensitivities  │  sympy, emitted C
├─────────────────────────────────────────────┤
│  pybind11 Binding Layer                     │  GIL release during sim
│  _bngsim_core.cpp                           │  Exception translation
├─────────────────────────────────────────────┤
│  C++ Engine (libbngsim)                     │  Re-entrant, instance-based
│  NetworkModel                               │  No globals, no file I/O
│  CvodeSimulator   (ODE, forward sens)       │  No stdout, no exit()
│  SsaSimulator     (SSA, PSA)                │
│  NfsimSimulator   (network-free)            │
├─────────────────────────────────────────────┤
│  SUNDIALS v7.x  (vendored)   CVODE, CVODES  │  adaptive BDF, KLU sparse
│  ExprTk         (vendored)   rate laws      │  header-only, bytecode
│  NFsim          (vendored)   network-free   │  rejection / null-event
│  RuleMonkey     (vendored)   network-free   │  exact non-local
│  MIR            (vendored)   JIT            │  generated-C fast path
└─────────────────────────────────────────────┘
```

## Key design decisions

1. **SUNDIALS v7.x** with `SUNContext` for re-entrancy, rather than the 2.4.0
   bundled in BNG.
2. **ExprTk** replaces muParser. Header-only and bytecode-compiled, with
   backward-compatible aliases for BNG's spellings.
3. **Instance-based state.** No globals, so multiple models and simulators
   coexist safely in one process and the GIL can be released during a run.
4. **Static linking.** The wheel is self-contained, with SUNDIALS linked
   statically into the extension.
5. **Pluggable loaders.** `.net`, SBML (`.xml`), Antimony (`.ant`) and BNGL
   (`.bngl`) all build through `ModelBuilder`, so every backend sees one model
   representation whatever the source format was.
6. **Both network-free engines run in-process.** NFsim and RuleMonkey are
   vendored and linked, not shelled out to. Neither writes files or calls
   `exit()`, and both reuse the host expression evaluator rather than carrying
   their own, so a rate law means the same thing everywhere.
7. **Gradients are analytic where they can be.** A symbolic Jacobian and an
   analytic `∂f/∂p` are derived from the model and emitted as C. Where a
   construct cannot be differentiated the run says so and falls back to CVODES'
   difference quotient rather than returning a quietly wrong number.

## Vendored dependencies

Everything under `third_party/` is generated source with its own `VENDOR.json`
recording the upstream commit, and a matching `scripts/*_VENDORING.md` describing
the refresh. Edits made directly to those trees are lost at the next refresh, so
a change that needs to survive is landed upstream first.

NFsim additionally carries a small explicit queue of local patches, listed in
both `scripts/vendor_nfsim.py` and its `VENDOR.json`. Each entry is either a
required local adaptation, such as throwing instead of calling `exit()`, or is
marked as bound for an upstream pull request and dropped once it lands.
