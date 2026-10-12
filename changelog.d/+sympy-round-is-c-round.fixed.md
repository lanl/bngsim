- **`round()` of a number means C's round in the symbolic paths too, so a
  derived parameter's sensitivity and a clock threshold through one are right.**
  The sympy readers left `round` to Python's, which rounds a half to even:
  `round(2.5)` became 2 where the engine computes 3 (issue #684). A derived
  parameter `round(2.5)*k` got a forward sensitivity 2/3 of the true one, a
  clock condition `floor(time()) >= round(2.5)` got its stop at t = 2 instead of
  3, and a rate law with `round(2.5)` lost its analytic Jacobian to the
  self-check. A `round` of a parameter is evaluated as C's round where its value
  is needed and declines where it would have to be differentiated.
