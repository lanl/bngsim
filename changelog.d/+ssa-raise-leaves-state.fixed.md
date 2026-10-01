- **An SSA or PSA run that raises leaves the model as it found it.** The
  stochastic loop writes its running state into the model as it goes (the
  species it syncs for observables, functions and event triggers, and the
  clock), so a run stopped part way by a timeout or a refusal left a mid-run
  state behind under a clock and events still at the run's start. The next run
  continued from it with no error. The species and the clock are now put back
  before the error propagates, as the ODE path already did.
