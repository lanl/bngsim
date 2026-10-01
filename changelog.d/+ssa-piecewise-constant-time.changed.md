- **SSA and PSA run a rate that is a step in time as a constant between
  breakpoints.** A rate that reads the clock only inside conditions resolved to
  fixed crossing times (`if((time() > 10) && (time() < 150), 2.6, 0)`, a dosing
  schedule) stays in the selection tree, and the loop re-reads it at each
  crossing instead of integrating it panel by panel. BIOMD0000000558 runs 1.6×
  faster than with #933 and 1.7× faster than before it. A rate that reads the
  clock anywhere else is integrated as before. A guard refuses a run in which
  such a rate moved between breakpoints.
