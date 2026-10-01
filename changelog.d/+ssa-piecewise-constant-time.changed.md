- **SSA and PSA run a rate that is a step in time as a constant between
  breakpoints.** A rate that reads the clock only inside comparisons of the bare
  clock with a fixed threshold (`if((time() > 10) && (time() < 150), 2.6, 0)`)
  stays in the selection tree, and the loop re-reads it once it reaches each
  crossing, instead of integrating it panel by panel. BIOMD0000000558 runs 1.6×
  faster than with #933 and 1.7× faster than before it. A rate that reads the
  clock anywhere else (arithmetic, a step call such as `mod` or `floor`, a
  schedule) is integrated as before. At every output row a guard checks that
  such a rate has not moved since it was set, and refuses the run if it has.
