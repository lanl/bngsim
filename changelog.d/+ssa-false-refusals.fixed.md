- **SSA/PSA no longer refuse three kinds of SBML reaction they run exactly.**
  - A reaction flagged reversible whose law has no difference in it
    (`Vm*A/(Km+A)*c`, as COPASI flags by default) runs as the same law
    flagged irreversible does, trajectory for trajectory.
  - In a variable-volume compartment, `0.3*A*c`, `3*c` and `k2*A*c` with `k2`
    an assignment rule over time were refused where `k*A*c` ran.
  - A zeroth-order synthesis of an amount-valued species (`=> H; k`).
