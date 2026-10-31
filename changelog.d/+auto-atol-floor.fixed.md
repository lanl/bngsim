- **`atol="auto"` no longer holds a species at 0 to the scale of the large
  ones.** A species with no magnitude of its own (one that starts at 0) was
  given the smallest positive start as its scale. In a model whose every start
  is large, that is no scale for it. MODEL2009230001's species at 0 beside one
  at 1.1e10 were held to an absolute 110 and came out 100% off, and
  BIOMD0000000661 and BIOMD0000000845 were 2.2e2 and 1.3 off. The default floor
  of `auto_atol`, `derive_atol` and a `TrackingAtol` ceiling is now the smallest
  positive start, but not above 1: one molecule, in a model of counts. Those
  three models are now 8.5e-6, 1.9e-4 and 6.5e-8 off. A model whose smallest
  start is at most 1 gets the same tolerances as before.
