- **An event assignment far larger than what it reads keeps its
  sensitivity.** The event jump differenced the assigned value over a
  millionth of each state and parameter. `X = X + D` with `D = 100` moves by
  2e-15 over that step at `X = 1e-9`, under one ulp of 100, so the derivative
  came back 0 and the sensitivity X carried into the event was dropped: a dose
  into a species a zero-order process had run down gave −3 for −6. A parameter
  whose term is a small part of the value (`D + q*Y`, `q = 1e-9`) gave 0 for
  `Y`. Where the narrow difference keeps fewer than nine digits it is now taken
  again over wider steps, from a millionth of the value down, and the widest
  one the value is straight across is kept. A value that bends inside every
  wider step (`D + max(0, X - K)` near `K`, a saturating law) keeps the narrow
  difference, as before.
