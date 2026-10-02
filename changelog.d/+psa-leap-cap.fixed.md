- **A PSA leap no longer takes more of a species than it holds.** With a
  stoichiometry above `poplevel` (`3A ->` at `poplevel=2`), a leap of
  floor(n/poplevel) fires consumed more than was present and drove the count
  negative. The leap is now also capped by each such species' population over
  its stoichiometry; where `poplevel` covers every stoichiometry, the usual
  case, runs are unchanged. A firing that takes and gives back one species
  (`3A -> A + B`) no longer reports a negative crossing it never made, and
  `poplevel=nan` is refused instead of running exact SSA.
