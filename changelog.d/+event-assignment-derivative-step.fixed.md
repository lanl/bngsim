- **An event assignment far larger than what it reads keeps its
  sensitivity.** The event jump differenced the assigned value over a
  millionth of each state and parameter. `X = X + D` with `D = 100` moves by
  2e-15 over that step at `X = 1e-9`, under one ulp of 100, so the derivative
  came back 0 and the sensitivity X carried into the event was dropped: a dose
  into a species a zero-order process had run down gave −3 for −6. A parameter
  whose term is a small part of the value (`D + q*Y`, `q = 1e-9`) gave 0 for
  `Y`. Where the narrow difference keeps fewer than nine digits it is now taken
  again over wider steps, up to a millionth of the value, for as long as the
  value is straight across each, the difference over half of it is the same,
  and it agrees with the one before. The step kept is the last a wider one
  agreed with. A value that bends inside every wider step
  (`D + max(0, X - K)` near `K`, a saturating law) keeps the narrow
  difference, as before, and so does a staircase with a tread over about 16
  ulp of the value. A finer staircase is read as its slope.
