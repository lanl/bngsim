- **A step call whose argument is within its own rounding of a level it crosses
  is refused by name.** `sign`, `floor`, `round` and the other step calls of an
  argument that is a small difference of large terms (a polynomial of the clock
  written out term by term, near a root or a turn) read both of their values by
  turns around the crossing, for millions of doubles or billions. There is no
  time at which such a call changes, and a stop placed for it anyway left runs
  that stepped over the window it bounds. The run now stops with an error
  naming the call and the stretch, where before it ran with no stop for the
  call. Write the argument in a form that does not cancel (`(time() - 5)^3`);
  no `max_step` stands in. A flicker shorter than a billionth of the run is one
  instant and has its stop, as before. An argument that does not flicker but
  stays within a few of its last places of a level for more of the run than can
  be read (`floor(1e9 + 3e-7*sin(time()))`, a `tanh` creeping onto 1) is
  refused the same way, and so is a `mod` or `frac` of a quotient beyond 2^48. See "Rate laws that switch on time" in the solver guide.
