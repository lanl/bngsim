- **How fast a crossing's residual moves is read correctly where it reads a
  species far smaller than the others.** The gradient of an event trigger or a
  state-switch residual is a central difference over a millionth of each
  species. For `(LECs + Capillaries) - L` with LECs at 1e4 and Capillaries at
  1e-10, just after its rate law turns on, that step is under an ulp of the
  residual: the residual read as not moving with a species that carries it at
  504 a unit of time, and its flow came out -90 for +414. A difference lost in
  the rounding of the largest species the residual reads is now retaken over
  wider steps.
