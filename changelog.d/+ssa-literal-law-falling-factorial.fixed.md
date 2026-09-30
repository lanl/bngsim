- **SSA and PSA take the falling factorial for SBML laws evaluated as written,
  too.** A kinetic law the loader cannot read as mass action (a boundary
  reactant, reactants in compartments of different sizes, an assignment-rule
  compartment, a law divided by a volume) was evaluated literally, so a
  species it reads m times contributed n^m: `2B -> P` at `k*B*B` kept firing
  with one B left, where the same reaction from a `.net` fires at 0, and a
  lone A in `2A + B -> P` across two compartments went to -9. Such a law now
  takes n(n−1)…(n−m+1) when it is a product of its species. A law that writes
  its own combinatorics (`k*X*(X-1)/2`) is left as written. ODE is unchanged.
