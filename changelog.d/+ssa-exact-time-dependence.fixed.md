- **SSA and PSA are exact in time for rates that move between firings.** A rate
  that reads time, or a rate-rule target, was held constant over sub-steps of
  (t_end − t_start)/1000, so the answer depended on the horizon: synthesis at
  `5*(1 + sin(time))` over (0, 5000) gave E[N(995)] = 2707 where the exact mean
  is 5002 (#719). Rate rules were integrated by forward Euler on that grid, so
  `x' = -30*x` over (0, 100) went to 1e33 (#753). With no firing reaction they
  only moved at output times (#751). The loop now integrates each time-varying
  propensity over adaptive panels and fires at the root of the integrated
  hazard, with rate rules on an L-stable Rosenbrock method. It never steps
  across a time at which a rate jumps (a resolved `if()` condition, a
  time-indexed table's knot), so a pulse narrower than a step is no longer
  skipped. A function of time that feeds no rate no longer moves a model off
  the discrete loop.
