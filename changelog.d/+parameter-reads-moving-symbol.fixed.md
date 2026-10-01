- **A parameter defined in terms of something a run moves is refused.** A
  derived parameter is evaluated only when a parameter is set, so one that read
  `time()`, a function, an observable or a rate accessor held its build-time
  value for the whole run: with `kf = c + time()` a function and `k2 = 0.5*kf`
  a parameter, `0 -> A` at `k2` ran at 0 under ODE and SSA alike, where A(4) is
  6. `ModelBuilder.build()` now refuses it, naming the symbol, and says to
  define the quantity as a function. No model in the SBML test suite, the
  BioModels corpus or the `.net` files in the tree is affected.
