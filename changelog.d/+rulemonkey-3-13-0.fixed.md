- **`method="rm"` runs on RuleMonkey 3.13.0.** A rate that counts the
  complexes matching a reactant pattern now follows a bond made or broken far
  from the rule's own molecules at once; it lagged until the rule next fired,
  so a chain cut and rejoined fast read 233 against BNG2's 259. Passing
  `block_same_complex_binding=False` with `method="rm"` or to a
  `RuleMonkeySession` is deprecated upstream: RuleMonkey prints one warning to
  stderr per process, and 4.0 removes the mode.
