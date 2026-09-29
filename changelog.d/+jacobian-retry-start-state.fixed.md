- **The finite-difference Jacobian retry now starts where the failed attempt
  started (GH #176).** When the analytical Jacobian failed part-way through a
  run, the model still held the state at the failure, because every event or
  switch stop writes the state back. The retry integrated from that state,
  labelled it `t_start`, and returned it, with only the retry warning to show for
  it. On a derived pulse onset under sensitivities, this gave `X(0) = X(3)` and a
  sensitivity of -641 before the pulse opened, where explicit `jacobian="fd"`
  raised. The retry now restores the starting state and any carried-over
  sensitivity seed first.
