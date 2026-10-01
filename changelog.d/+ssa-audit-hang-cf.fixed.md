- **Three SSA/PSA defects from a correctness audit.**
  - A run with a tiny total propensity beside a trigger that reads the clock
    probed the trigger out to its next firing, far past `t_end`: it hung and
    ignored `timeout=`.
  - An SBML conversionFactor other than 1 is now refused under SSA/PSA (code
    `conversion_factor`). It was folded into the rate, which keeps the mean
    but halves the variance at cf = 2: each firing must move the species by
    cf × stoichiometry.
  - An event that writes a fractional molecule count rounds it there, with the
    #718 warning, so a run split into legs matches the run whole.
