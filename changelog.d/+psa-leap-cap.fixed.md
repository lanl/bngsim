- **A PSA leap no longer takes more of a species than it holds.** With a
  stoichiometry above `poplevel` (`3A ->` at `poplevel=2`), a leap of
  floor(n/poplevel) fires consumed more than was present and drove the count
  negative (warned). The leap is now also capped by each such species'
  population over its stoichiometry. Where `poplevel` is at least every
  stoichiometry, the usual case, runs are unchanged, matching run_network.
