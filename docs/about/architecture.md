# Architecture

```
┌───────────────────────────────────────┐
│  Python API (bngsim)                  │  pip install bngsim
│  Model, Simulator, Result             │  NumPy arrays, logging
├───────────────────────────────────────┤
│  pybind11 Binding Layer               │  GIL release during sim
│  _bngsim_core.cpp                     │  Exception translation
├───────────────────────────────────────┤
│  C++ Engine (libbngsim)               │  Re-entrant, instance-based
│  NetworkModel, CvodeSimulator         │  No globals, no file I/O
│  SsaSimulator                         │  No stdout, no exit()
├───────────────────────────────────────┤
│  SUNDIALS v7.x (vendored)             │  CVODE (adaptive BDF)
│  ExprTk (vendored, header-only)       │  Rate law expressions
└───────────────────────────────────────┘
```

## Key design decisions

1. **SUNDIALS v7.x** with `SUNContext` for re-entrancy (not the ancient 2.4.0 bundled in BNG)
2. **ExprTk** replaces muParser — header-only, bytecode-compiled, 5 backward-compat aliases
3. **Instance-based state** — no globals. Multiple models/simulators coexist safely
4. **Static linking** — the wheel is self-contained (SUNDIALS linked statically into the extension)
5. **Pluggable loaders** — `.net`, Antimony (`.ant`), and SBML (`.xml`) via `ModelBuilder`
