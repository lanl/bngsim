- **An SBML reaction that changes an amount and a concentration in one
  compartment whose size changes ran a wrong ODE.** The two are divided by the
  size at load and the live size, and one divide served both: `H => B; k*H`
  with H an amount in a growing compartment made B(6) 40.55 molecules out of 40,
  where RoadRunner gives 33.39. A concentration changed beside an amount-valued
  catalyst was divided by the size at load as well. 45 of 54 such shapes were
  wrong; each row now takes its own divide.
