- **A narrow window of a sinusoid or a polynomial in time is no longer
  stepped over** (the time-only cases of #714, in ODE, sensitivity, SSA and
  PSA runs). `sin(10*time) > 0.99` or `(time - 3)^2 < 4e-4` is true on windows
  a few hundredths wide. No resolver placed their crossings, so an ODE step, an
  SSA panel or the SSA's event probe could lie across a window unseen. An event
  then fired 0–63 times where 159 is right, a pulse ran 0 or 1.27 where 4.5 is
  right, and the result depended on the output grid. The crossings are now
  solved in closed form (a polynomial's exactly), for literal time and for a
  BNGL counter clock, and every engine stops a few ulps past each one.
