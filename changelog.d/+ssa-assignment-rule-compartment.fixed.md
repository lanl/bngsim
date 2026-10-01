- **SSA refuses a model that reads a concentration in a compartment whose size
  changes in a way it does not follow.** That covers:
  * a reaction touching such a concentration in a compartment an assignment
    rule resizes;
  * a read of one in an event- or rate-rule-resized compartment from outside
    it, through an assignment rule, or by a cross-compartment law the volume
    correction does not fit (a modifier, `X*X`);
  * a rate rule or an event reading one.

  All ran at the load-time size: A(4) at 3× the ODE's, z = +16 to +50, and a
  trigger that never fired. A rule over constants (`C := 2`), and amount-valued
  species, still run.
