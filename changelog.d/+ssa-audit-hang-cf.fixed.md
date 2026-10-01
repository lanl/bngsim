- **Three SSA/PSA defects from a correctness audit.**
  - A run with a tiny total propensity beside a trigger that reads the clock
    probed the trigger out to its next firing, far past `t_end`: it hung and
    ignored `timeout=`. A `time > T` event just past a leg's end also fired in
    both legs.
  - An SBML conversionFactor other than 1 is refused under SSA/PSA (code
    `conversion_factor`): folded into the rate it keeps the mean but not the
    noise. The ODE dropped the factor when the stoichiometry came from an
    initialAssignment (A(10) = 40 where 70 is right).
  - A `run_until` leg rounded a fractional count an event had written in the
    previous leg, so legs differed from the run whole; a run that continues
    the previous one rounds nothing.
