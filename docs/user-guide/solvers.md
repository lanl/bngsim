# Solver configuration

## Solver configuration

```python
sim.set_tolerances(rtol=1e-10, atol=1e-10)
sim.set_max_steps(50000)

# Per-run overrides
result = sim.run(
    t_span=(0, 100), n_points=101,
    rtol=1e-12, atol=1e-12, max_steps=100000,
)

# Solver diagnostics
print(result.solver_stats)
# {'n_steps': 1247, 'n_rhs_evals': 2891, 'n_jac_evals': 43, ...}
```

Those counters cover the **whole run**. CVODE's own counters restart at every
re-initialization — an event fire, a rate-law switch crossing and a chatter
re-arm each force one — so bngsim banks each segment as it closes rather than
sampling the counters once at the end, which would report only the span after
the last restart (issue #182).

## Per-species absolute tolerance

`atol` also takes one value **per species**, ordered like
`model.species_names`. That is `CVodeSVtolerances` rather than
`CVodeSStolerances`, and it exists because a scalar `atol` is one number asked
to mean the same thing for every state variable:

| species | `y(0)` | the atol it needs alone (`rtol·y`) |
|---|---:|---:|
| `IRp` | 1.763e-09 | **1.8e-17** |
| `IRSiP` | 1.33e-01 | 1.3e-09 |
| `X` | 1.0e+01 | **1.0e-07** |

Ten decades of species, and a scalar has to pick one number for all of them.
Pinned near the tight end the model stops integrating; pinned near the loose
end the small species sits under the noise floor for the entire run and its
trajectory means nothing — it can come back negative. There is no scalar
between them that is a tolerance for both.

```python
# Explicit — one value per species, in species_names order.
result = sim.run(t_span=(0, 100), atol=[1e-17, 1e-9, 1e-7])

# Derived from the model's own state: rtol * max(|y_i|, floor).
result = sim.run(t_span=(0, 100), atol="auto")

# ...or derive, inspect, adjust, then pass it back.
atol = sim.auto_atol()
atol[model.species_names.index("IRp")] *= 10
result = sim.run(t_span=(0, 100), atol=atol)
```

The vector is positional and its length is checked, not adjusted: a wrong
length raises `ValueError` rather than broadcasting or truncating, because the
consequence of guessing is species *i* held to the number written for species
*j*, with a plausible-looking trajectory to show for it.

`atol=` accepts the same three forms on `set_tolerances`, `run_batch`,
`parameter_scan`, `bifurcate`, `run_until`, `compute_all_sensitivities`,
`steady_state`, `steady_state_batch`, and `EvaluationSpec`. Anything that runs
many points (a batch, a scan, a chunked sensitivity job) resolves `atol` **once**
up front, including `"auto"` — every point is held to the same tolerance, so the
points can be compared with each other.

Two things worth knowing:

- **The sensitivity columns follow the state axis.** `atolS` for
  ∂x_i/∂θ is built from species *i*'s own absolute tolerance, so a per-species
  state tolerance is not collapsed back onto one number for the derivatives.
- **`steady_state_tol` still reads the scalar.** The early-stop criterion is
  `||f(t,y)||₂ / n_species`, one norm over every species with no per-species
  reading to take, so it falls back to the *scalar* `atol` even when a vector is
  in force — and to this `Simulator`'s own scalar, not to anything derived from
  the vector, because there is no honest single number to derive from one.
  **Pass `steady_state_tol` explicitly whenever you pass a vector.** Left
  unset it silently reverts to the default `1e-8`, and on a model whose states
  are themselves ~1e-8 that criterion is already satisfied at *t* = 0: the
  relaxation returns the initial state and calls it the steady state. That reads
  as a modelling problem rather than a tolerance one, which is why it is worth
  one kwarg to rule out.

`atol="auto"` derives `rtol * max(|y_i|, floor)`, where `floor` defaults to the
smallest strictly positive species value in the model — a species sitting at
zero has no magnitude of its own to scale, so it is treated as living at the
smallest scale the model actually exhibits. Pass `floor=` to `auto_atol` to
choose differently. Being built from *initial* values, this cannot see a species
that starts at order one and decays to something tiny; that is a within-species,
over-time mismatch, and the CVODE construct for it is `CVodeWFtolerances`, which
bngsim does not expose yet (issue #213). The corollary is worth stating plainly:
on a model whose species all start at similar magnitudes there is no
cross-species spread to compromise over, so a per-species vector is elementwise
the scalar it replaces and changes nothing. It is not a general tolerance
improvement — it is the fix for models spanning decades.

### Which state the tolerance comes from

`atol="auto"` and `sim.auto_atol()` read the model's **live** state — the one
the next `run()` would start from. That is what you want for a one-off run, and
it is the wrong thing for a **parameter fit that moves initial conditions**:
the vector would be re-derived at every evaluation, so `atol` becomes a function
of the fit point rather than of the model. The objective then steps wherever the
derivation crosses a rounding boundary — invisible in the usual way, since the
objective still looks correct and the finite-difference gradient check still
passes, and only the search behaves oddly.

`bngsim.derive_atol` is the same rule against a state **you** supply, so the
result can be a constant of the model. Derive it once, hold it, pass it every
time:

```python
import bngsim

model = bngsim.Model.load("model.xml")
sim = bngsim.Simulator(model, method="ode")

RTOL = 1e-8
nominal = model.get_state()                          # before anything is fitted
atol = bngsim.derive_atol(nominal, RTOL)             # a constant of the model

for theta in search:
    model.set_params(theta)                          # may move initial conditions
    result = sim.run(t_span=(0, 100), rtol=RTOL, atol=atol)
```

The rule, the `floor` default, and the length/order contract are identical to
`auto_atol`'s; the only difference is which state is read. If you assemble the
vector yourself instead — clamping per species, reading it off a table —
`bngsim.normalize_atol_vector(vec, model.n_species, model.species_names)`
applies the same length and position check `run()` would, so the mismatch
surfaces where the vector was built rather than at the first evaluation.

### Detecting the capability

The version string will not tell you whether an install has any of this — the
checkout that first carried it still declared `0.12.2`. Feature-detect instead:

```python
if hasattr(bngsim, "AUTO"):
    atol = bngsim.derive_atol(nominal, RTOL)
else:
    atol = scalar_fallback(nominal, RTOL)
```

## Jacobian strategy

The ODE solver (CVODE BDF/Newton) needs the Jacobian matrix ∂f/∂y at each
Newton iteration. BNGsim supports three strategies, selectable via the
`jacobian` keyword:

```python
# Default: auto-select best available strategy
sim = bngsim.Simulator(model, method="ode")  # jacobian="auto"

# Force analytical Jacobian (error if model has Functional/MM rates)
sim = bngsim.Simulator(model, method="ode", jacobian="analytical")

# Force finite-difference Jacobian (baseline, for benchmarking)
sim = bngsim.Simulator(model, method="ode", jacobian="fd")
```

| Strategy | Description | Cost per Jacobian | Availability |
|----------|-------------|-------------------|--------------|
| `"auto"` (default) | Analytical if available, else finite-difference; falls back to FD if the analytical attempt fails to integrate | — | All models |
| `"analytical"` | Exact derivatives from mass-action stoichiometry | O(nnz) ops, zero RHS evals | All-Elementary models only |
| `"fd"` | Finite-difference approximation (SUNDIALS DQ for dense, colored FD for sparse) | O(N) or O(n_colors) RHS evals | All models |

After a run, `sim.jacobian_strategy` reports the strategy that actually produced
the result (`"analytical"`, `"fd"`, or `"jax"`) — including `"fd"` when an
`"auto"` run fell back (see below).

**When to use `"fd"`**: For A/B benchmarking, or if you suspect the analytical
Jacobian is causing issues. The `"fd"` option uses SUNDIALS' internal
difference-quotient (DQ) approximation for dense models, and graph-colored
finite differences for sparse (KLU) models.

**Analytical Jacobian**: Available automatically when all reactions use
Elementary (mass-action) rate laws. For models with Functional or
Michaelis-Menten rates, the solver falls back to finite-difference. The
analytical Jacobian provides ~10-27% speedup on large models by eliminating
O(N) RHS evaluations per Jacobian update.

**Auto-fallback on a solver failure (GH #176)**: `jacobian="auto"` is a bet —
an analytical Jacobian is a strict speedup where it integrates, but it is not
guaranteed to. A rate law that is *discontinuous* in a state variable (e.g. an
`if()` whose condition crosses a threshold the state sits at) has an exact
derivative that omits the jump, which can de-stabilize CVODE's implicit
corrector even though the Jacobian is mathematically correct. Under `"auto"`,
such a CVODE failure is caught and the integration is transparently retried once
with the finite-difference Jacobian (which straddles the step and regularizes
the corrector), so the default config still integrates the model. An explicit
`jacobian="analytical"` is **not** second-guessed — it surfaces the failure.

**Build-time derivation budget (GH #95 / #187)**: for models with Functional or
Michaelis-Menten rate laws, the analytical Jacobian terms are derived once
(symbolically) at model load. That derivation is wall-clock-budgeted so a
pathological *small* model cannot hang the load — it simply falls back to the FD
Jacobian, which is just as fast to solve at small scale. The budget **scales with
species count** and becomes **unbounded** on genome-scale models
(≥ 20,000 species): there an FD Jacobian needs ~`n_species` RHS evaluations per
Newton step and is not a viable solver path, so the analytical Jacobian is
treated as mandatory and is always derived to completion. Override the budget with
`BNGSIM_JAC_DERIV_BUDGET_S` (seconds, or `inf`/`none`/`0` to disable it entirely —
the manual genome-scale escape hatch).

**Sensitivity derivation budget (GH #90)**: a sensitivity run derives a second
set of terms at build time — both halves of the compiled sensitivity RHS's
`ySdot = J·yS + ∂f/∂p` — and that derivation has its own budget,
`BNGSIM_SENS_DERIV_BUDGET_S`. Only a *sensitivity* run pays it: a plain
`Simulator(model, method="ode")` used to run the same derivation for a solve that
never installs the result, which on a large Functional model was most of the build
(issue #209). It
shares the Jacobian budget's base and its per-species scaling, and takes the same
values (seconds, or `inf`/`none`/`0` for unbounded), but the two are independent:
raising one does not raise the other.

One difference is deliberate. The sensitivity budget **never becomes unbounded**,
at any model size. Where a dropped analytical Jacobian falls back to something
that may not converge at all, a declined sensitivity RHS falls back to CVODES'
internal difference quotient, which is correct at every scale — so there is never
a reason to let this derivation run without a bound, and a genome-scale model with
Functional rate laws gets a decline it can read rather than a build that appears
to hang.

The fallback is correct but not cheap: measured 9–37x per sensitivity column, and
on a stiff model at a tight `rtol` the difference quotient's own `~sqrt(rtol)`
accuracy collapses the step size (it can exhaust `max_steps`). So the decline is
logged as a warning naming this variable — if a model you care about reports it,
raise the budget rather than accept the slower path:

```bash
BNGSIM_SENS_DERIV_BUDGET_S=inf python fit.py
```

**Expression output sensitivities (GH #97)**: a sensitivity run on a model with
global functions derives a *third* set of terms at build time — the chain rule
`d func/dθ` behind `Result.output_sensitivities("expression:…")` — and it reads
the same `BNGSIM_SENS_DERIV_BUDGET_S`. Two differences from the phase above, both
deliberate:

- It resolves its **own** deadline rather than sharing that build's, so a slow
  `∂f/∂p` cannot starve it (and vice versa). One knob, two clocks.
- Its budget scales with the number of **derivation steps** the analysis is about
  to run — one per expression parsed, one per derivative taken — rather than with
  species count, because that is what drives the cost here: a model can carry
  thousands of global functions on a few hundred species.

An expiry does **not** decline anything. Output sensitivities are per function, so
every function derived before the deadline keeps working, and the rest are marked
unsupported: selecting one raises an error naming the budget and this variable,
the same way selecting a function containing a non-differentiable construct does.
Observable and species sensitivities are unaffected.

## Logging

```python
import logging
bngsim.configure_logging(logging.DEBUG)
# Now all bngsim operations produce log output
```
