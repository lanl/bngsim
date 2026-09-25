- **The sensitivity decline warning no longer calls the fallback correct across
  a step in the state.** `floor(Atot)` jumps each time `Atot` crosses an
  integer, at a time the rate constants move. The analytic sensitivity RHS
  declines it, and CVODES' difference quotient then drops those jumps: on
  `k1*floor(Atot)` it is 59 % off a central difference by t = 0.5. The warning
  now names such a call as a moving crossing. The same holds for `ceil`,
  `round`, `sign`, `mod` and the other stepping builtins, unless the argument
  reads only run-constants, or only the clock and literals.
