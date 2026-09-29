- **A failed ODE run now leaves the model where it found it, and the GH #176
  retry starts where the failed attempt did.** A run continues from the model's
  live state, and every event or switch stop writes that state back. So a run
  that failed part-way left the model at the failure, and the next run on it
  started there, labelled `t_start`, with no warning. The finite-difference
  retry did the same within one call: on a derived pulse onset under
  sensitivities it reported `X(0) = X(3)` and a sensitivity of -641 before the
  pulse, where explicit `jacobian="fd"` raised. Both now restore the starting
  state and any carried-over sensitivity seed.
