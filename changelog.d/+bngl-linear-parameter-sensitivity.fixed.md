- **`Model.from_bngl` ran a `LinearParameterSensitivity` that came before the
  file's `generate_network`.** BNG2.pl got it with the rest of the model, and
  it runs ODE simulations of copies read from a `.net`, writing their output
  beside that file, during the load. It is now left out, like the file's runs,
  without stopping the actions after it.
