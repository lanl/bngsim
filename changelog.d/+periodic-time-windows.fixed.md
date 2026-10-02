- **A narrow window of a sinusoid or a polynomial in time is no longer
  stepped over** (#714; ODE, sensitivity, SSA and PSA runs). `sin(10*time) >
  0.99` or `(time - 3)^2 < 4e-4` is true on windows a few hundredths wide. No
  resolver placed their crossings, so an ODE step, an SSA panel or the SSA's
  event probe could lie across a window unseen. An event fired 0–63 times where
  159 is right, a pulse ran 0 or 1.27 where 4.5 is right, and SBML suite 00936
  gave 0, 2 or 1 for 3, depending on the output grid. The crossings are now
  solved in closed form (a polynomial's exactly), on literal time, a BNGL
  counter clock or an `if()` branch, and every engine stops just past each.
