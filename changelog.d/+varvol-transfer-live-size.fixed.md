- **A transfer between compartments divides by the size the compartment has,
  whatever its rate law.** A species stored as a concentration in a compartment
  whose size follows a rate rule or is reset by an event obeys
  `d[S]/dt = stoich*law/V(t) - [S]*V'/V`. The transfer term used the size the
  compartment had when the model was loaded, unless the reaction was an
  irreversible mass-action monomial. A reaction flagged reversible (the SBML
  Level 2 default, and what Antimony's `->` writes), a reversible difference
  `k*A - k2*B` and a saturating law were all integrated with the load-time
  size, with no warning: with `A -> B; k*A` into a compartment growing from 2
  at rate 1, [B](5) came back 0.284 for 0.196. Every such reaction now divides
  by the live size under ODE. Under SSA they are refused as before.
