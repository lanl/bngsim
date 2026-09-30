- **`make_subset_model` keeps what a parameter write does to the model.** A
  subset rebuilt its parameters without their compartment-size and
  write-refused flags, and its species without their `initialAmount`, their
  initial-value parameter or their ODE live-volume divide. After
  `set_param("C", 10)` a subset kept a 10-molecule species at 10/5 = 2 where the
  model re-divides it to 1. `A() A0` then `set_param("A0", 80)` left the
  subset's A at 50. A compartment write the model refuses went through on the
  subset. The subset also dropped a rule-assigned species' continuous flag, the
  SSA propensity's live compartment size and the SSA falling factorial above.
