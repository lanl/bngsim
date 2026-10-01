- **A bare `time` is the clock unless the model declares a scalar of that
  name.** ExprTk calls `time` without its parentheses, so `if(time > 5, k, 0)`
  in a `.net` function reads the clock. The SSA took such a rate as constant,
  and the builder let a parameter `2*time` through. It held 0 for the whole run.
