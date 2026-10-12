- **A model whose rates of change are written in terms of each other's `rateOf`
  is refused by name.** `X' = rateOf(Y); Y' = rateOf(X)`, or `X' = k*rateOf(X)`,
  is an algebraic loop: each rate is defined by the others at the same instant.
  bngsim integrated such a model with the rates read a pass late (1.75 for 2 in
  one example, 1.5 with `codegen=True`). An ODE `Simulator` of it, and
  `Model.rhs`, now raise `ModelError` naming the species (lanl/bngsim#991), as
  libRoadRunner, Antimony and libSBML refuse it. A species that stands on both
  sides of a reaction (a catalyst) is in no such loop, whatever its reaction's
  rate law reads: its derivative holds nothing of that rate.
