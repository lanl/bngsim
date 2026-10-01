- **A transfer between compartments is divided by the size each compartment
  has, for every rate law.** A species stored as a concentration in a
  compartment whose size follows a rate rule or is reset by an event obeys
  `d[S]/dt = stoich*law/V(t) - [S]*V'/V`. For any reaction but an irreversible
  mass-action monomial, the transfer term was divided by the size at load, or
  by one compartment's size for every species. A reaction flagged reversible
  (the SBML Level 2 default, and Antimony's `->`), a reversible difference
  `k*A - k2*B` and a saturating law were all wrong, with no warning: with
  `A -> B; k*A` into a compartment growing from 2 at rate 1, [B](5) came back
  0.284 for 0.196. Each changed species' row is now over its own compartment's
  live size under ODE, also beside an assignment-rule compartment and with
  mixed conversion factors. Under SSA these reactions are refused as before.
  Still wrong: a transfer between assignment-rule and static compartments, and
  the irreversible monomial beside an assignment-rule compartment (#745); some
  arrangements of a substance-only species beside a concentration (#787).
