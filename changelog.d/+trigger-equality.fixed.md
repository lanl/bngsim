- **An event on `time == c` fires on any output grid.** `at (time == 2)` is
  true at one double, so it had no crossing stop and no root, and fired only
  where an output sample landed on 2: with `n_points=6` over (0, 5) it fired,
  with `n_points=2` it did not, and the run returned the state of a model with
  no event. The run now stops on the instant, taken as the double at which the
  model's evaluator reads the equality as true (so `time == tau/7`, `time ==
  pi` and `time^2 == 4` fire too), for `==` and `!=`, alone or inside a
  compound trigger. An equality that no double satisfies, such as
  `time - 1 == 1.3`, is refused with an error naming the trigger where it is
  the whole trigger. An equality on an integrated variable (`x == 2` under
  `x' = 1`) is unchanged: use an ordering.
