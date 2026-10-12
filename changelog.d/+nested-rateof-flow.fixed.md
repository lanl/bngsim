- **A rate that reads a nested `rateOf` is integrated.** In
  `S' = rateOf(A); T' = k*rateOf(S)` the right-hand side read `rateOf(S)` a
  probe behind: T(1) came out 0.44 for 0.216 at the default tolerances, and was
  right only from `atol=1e-10` down. With `codegen=True`, and in any run with
  sensitivity columns, it was read as zero: T stayed 0 and every dT/dp was 0.
  The right-hand side, interpreted and generated, now makes one probe more for
  each level a derivative reads a nested `rateOf` at. A rate rule on a `rateOf`
  that nothing takes the `rateOf` of (`Y' = rateOf(X)`) costs what it did
  (lanl/bngsim#991).
