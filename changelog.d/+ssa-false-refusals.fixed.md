- **SSA/PSA no longer refuse three kinds of SBML reaction they run exactly.**
  - A reaction flagged reversible whose law has no difference in it
    (`Vm*A/(Km+A)*c`, as COPASI flags by default) was refused as a net flux. It
    runs as the same law flagged irreversible does, trajectory for trajectory.
  - In a variable-volume compartment, `k*A*c` ran but `0.3*A*c`, `3*c` and
    `k2*A*c` with `k2` an assignment rule over time were refused. The
    volume correction depends on the species alone.
  - A zeroth-order synthesis of an amount-valued species (`=> H; k`), which
    does not depend on the volume, was refused.
