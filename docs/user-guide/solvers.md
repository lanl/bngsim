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

## The default tolerances, and models in small units

The defaults are `rtol=1e-8` and a scalar `atol=1e-8`, BNG2.pl's, chosen for
species held as molecule counts. A species that stays below `atol` for the whole
run has no error control beyond the tolerance itself. That happens in a model in
molar or other small units, which is common in SBML and BioModels. Such a
trajectory can be off by its own size, or go negative, and CVODE reports no
error: a binding model of species near 1e-9 M ended at -8.7e-10 for 4.1e-10.

`run()`, `run_until()`, `run_batch()`, `parameter_scan()`,
`compute_all_sensitivities()` and `bngsim.jax.differentiable_solve()` raise
`bngsim.AtolScaleWarning` in that case,
once per call and naming the species, where you gave no `atol` and a species
the solver integrates stays below it while reaching at least 1e-6 of the run's
largest. A species stuck at rounding in a count model does not reach that, and
a `$`-fixed species or an assignment-rule target is not integrated; neither is
warned. The remedy is `atol="auto"` (next section), or a scalar `atol` well
below those species (issue #1361).

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

## Tracking absolute tolerance

A vector fixes the tolerance *per species*. It does not fix it *per instant*:
whatever number species `i` gets, it keeps for the whole run. So a species that
starts at order one and **decays** to something tiny crosses under its own
`atol` partway through and stops being error-controlled from there on — the
same failure as above, reached through time rather than through the species
list. No initial-value heuristic can see it coming, because at `t = 0` there is
nothing to see.

`atol="tracking"` hands CVODE an error-weight function (`CVodeWFtolerances`)
that re-evaluates the vector against the state actually being integrated:

```
atol_i(y) = clamp(rtol * |y_i|, ceiling_i * 10**-decades, ceiling_i)
```

```python
result = sim.run(t_span=(0, 30), atol="tracking")                     # default depth
result = sim.run(t_span=(0, 30), atol=bngsim.TrackingAtol(decades=6))  # shallower
result = sim.run(t_span=(0, 30), atol=bngsim.TrackingAtol(ceiling=my_vector))
```

`ceiling` is the vector from the previous section (`"auto"` by default), so
this is a strict extension of it — at `decades=0` the rule *is* that vector,
and the clamp's upper end means tracking is never looser than it, only tighter.

`tests/data/deep_decay.net` is the worked example: two species that both start
at exactly `1.0`, so `atol="auto"` derives the same `1e-8` for both and the
per-species vector is provably a no-op. One of them decays sixteen decades.
Worst relative error in that species against the analytical `exp(-t)`:

| `atol` | rel. err. | steps |
|---|---:|---:|
| `1e-8` (scalar) | 1.8e+04 | 162 |
| `"auto"` (vector) | 1.8e+04 | 162 |
| `TrackingAtol(decades=6)` | 9.5e-02 | 366 |
| `TrackingAtol()` (12) | 2.6e-06 | 605 |

Two things worth knowing before reaching for it:

- **It costs steps** — roughly 4x here. That is the price of resolving a decay
  over twelve decades, and it is why the mode is opt-in rather than the default.
- **It cannot conjure digits that are not there.** A species whose value is a
  *difference* of large fluxes carries absolute roundoff of order `eps × flux`;
  asking for relative accuracy below that collapses the step size instead of
  sharpening the answer. `decades` is the bound on how far it will try, and 12
  is where the measured accuracy above stops improving.

Swept over the first 400 `rr_parity` SBML models (391 integrate at the default
tolerance; 37 of them have a species that falls six or more decades below its
own initial value, so this is not an exotic shape):

| `decades` | still integrate | lost | gained | steps vs default |
|---|---:|---:|---:|---:|
| 3 | 393 | 0 | 2 | 1.18 |
| 6 | 392 | 1 | 2 | 1.24 |
| 12 (default) | 387 | 6 | 2 | 1.58 |

So six models that integrate at the default tolerance do not at the default
depth. That failure is loud — the solver error names the depth and tells you to
lower it — and **lowering `decades` is the first thing to try**. (Two models
integrate *only* under tracking, which is the same effect from the other side.)

On the **sensitivity** axis nothing changes: `atolS` is still built from the
ceiling, so turning tracking on leaves the sensitivity tolerances where the same
vector would have put them. Measured on the fixture above, that costs nothing —
the sensitivity column comes back resolved anyway, because the state axis is
what drives the step size.

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
finite differences for sparse (KLU) models. Against a system SUNDIALS of
another release, see [Installation](../installation.md#from-source).

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
Jacobian, which at small scale is never wrong and usually solves just as fast.
Not always, though — on some small models the analytical Jacobian is genuinely
several times faster to solve with, so the default is set above the slowest
derivation measured to buy such a speed-up rather than at the point waste stops
(GH #245); raising it is rarely necessary. The budget **scales with
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
internal difference quotient, which converges at every scale — so there is never
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

## Linear solver

Each Newton iteration solves a linear system in `I - γJ`, and how that matrix is
factored is chosen per model, separately from how `J` is computed.
`result.solver_stats["linear_solver"]` reports the choice: `0` for the built-in
dense LU, `1` for sparse KLU, `2` for the BLAS dense factor.

| Route | When (default options) |
|---|---|
| KLU (sparse) | at least 50 species, a Jacobian under 10% dense with a structural nonzero to factor, and an LU factor that stays sparse (below) |
| BLAS dense factor | a model KLU would take whose LU factor fills in, on a build with BLAS |
| built-in dense LU | everything else |

**The LU-fill test.** A sparse Jacobian does not mean a sparse factor. On
rule-derived networks a few hub species (a free enzyme, an adaptor, a ligand)
couple to most complexes, and the LU of a Jacobian that is 1–7% dense fills
20–35% of `n²` or more. KLU has no BLAS, so there it spends most of a run
refactoring, and the BLAS dense factor is several times faster. A model of
256–5,000 species whose estimated fill is at least 10% therefore goes dense,
when all of these hold:

- the build links a BLAS dense factor (`bngsim.HAS_LAPACK_DENSE`: the macOS
  wheels, and source builds that find a LAPACK);
- the analytical Jacobian is in use (`jacobian="auto"` or `"analytical"`, and
  `sim.jacobian_strategy == "analytical"`). With `"fd"` the dense route would
  difference every column, where KLU colors its differences, so KLU stays.

| model (species, Jacobian density, LU fill) | KLU | BLAS dense |
|---|---|---|
| `fceri_ji` (354, 6.8%, 0.35) | 0.111 s | **0.025 s** |
| `multisite_phos` (1026, 1.2%, 0.21) | 2.37 s | **0.66 s** |
| `fceri_gamma` (3744, 0.9%, 0.30) | 47.4 s | **4.68 s** |
| `metapop_sir_100` (300, 1%, 0.02) | **0.025 s** | 0.159 s — stays KLU |

`sim.model._core.jacobian_sparsity["lu_fill_estimate"]` reports the estimate,
which comes from KLU's own symbolic analysis. `force_sparse_linear_solver=True`
keeps a model on KLU whatever its fill, and `force_dense_linear_solver=True`
takes the built-in dense LU (or, with `BNGSIM_LAPACK_DENSE=1`, the adaptive BLAS
solver below). The route changes the cost, not the answer: trajectories agree to
solver tolerance. `steady_state()` routes its march by the same rule and reports
it as `"lapack-dense"`.

**The opt-in BLAS solver (`BNGSIM_LAPACK_DENSE=1`).** Every other dense model
stays on the built-in LU unless this variable is set. The solver it enables
starts on the built-in factor and switches to BLAS after five factorizations, so
a short run pays no BLAS overhead. That count belongs to an *integration*: the
steps of a coupling loop (`run_until`, `ReactionKernel.advance`) continue it, and
an independent `run()` restarts it. `solver_stats["n_dense_blas_factorizations"]`
counts the run's own BLAS factorizations.

## Rate laws that switch on time

A rate law that is flat on both sides of a window, `if(time() >= 100 &&
time() <= 100.5, k, 0)`, gives an adaptive integrator no error to see, so a
step can span the whole window and the run returns as if it were never open.
bngsim finds the times at which a rate law changes branch on the clock before
the run, stops the integration on each, and restarts there. That covers:

- a comparison of the clock with constants, inside an `if()` or used as a
  number (`k*((time()>=100)*(time()<=140))`);
- `floor`, `ceil`, `trunc`, `round`, `rint`, `frac`, `sign` and `mod` of the
  clock, compared or bare, and a repeating schedule written with them;
- a window written with `abs`, `min`, `max`, a power, an exponential or a
  sinusoid, and a pulse that only bends (`max(0, 1 - abs(time()-t0)/w)`);
- the breakpoints of a table function over time;
- the same conditions written on a unit-rate counter species.

A condition is first read exactly where it has a shape bngsim knows (linear in
time, a repeating schedule, a step of a linear argument, a sinusoid, a
polynomial). Any other is evaluated over the run window with the model's own
evaluator, and a stop is placed where its value changes. The evaluation is done
on a grid and again on one twice as fine, until both find the same number of
changes; what an exact reading placed is checked against the evaluator the same
way.

What this does not cover:

- **A smooth pulse with no switch in it**, such as a narrow Gaussian
  `exp(-((time()-t0)/s)^2)` feeding a model that is otherwise at rest. Nothing
  changes branch, so there is nothing to stop at. Pass `max_step` no larger
  than the pulse's width.
- **A window narrower than 1/8,192 of the run that no sample has any sign
  of**: one made by a smooth residual that is flat to the last bit outside the
  window (`exp(-((time()-t0)/s)^2) > 0.5` with `s` a millionth of the run), or
  by a discontinuity bngsim does not know the levels of (the `%` operator).
  Windows made by the calls listed above, by a residual that leans toward its
  turn, or at the edge of a `sqrt` or `ln` are found at any width.
- **A condition whose changes cannot be counted.** `sin(1/(time()-t0)) > 0.5`
  changes more often on every finer grid, up to two million cells. The run is
  refused with an error that names it.
- **A step call whose argument is within its own rounding of a level it
  crosses.** `sign(time()^3 - 15*time()^2 + 75*time() - 125)`, the cube of
  `time() - 5` written out term by term, reads -1 and 1 by turns on the doubles
  around 5, for as long as the terms cancel to less than their own last places:
  0.00004 time units either side of 5 for this cube, a twentieth of a time unit
  for a seventh power. There is no time at which such a call changes, so no
  stop can be placed for it, and the run is refused with an error that names
  the call and the stretch. Write the argument so that it is not a small
  difference of large terms (`(time() - 5)^3`). A `max_step` does not stand in
  here as it does for a condition that cannot be counted: what the rate law's
  other switches are is not known where a call of it flickers, so no bound can
  be checked against them. A flicker shorter than a billionth of the run, which
  any sum of terms can show around a crossing, is one instant of the run and is
  stopped at as usual. The refusal is for the call, whatever reads it:
  `ceil(p) > -7751` for a parabola `p` written out is refused for the flicker
  of `ceil(p)` at its top, which the comparison does not read.

  An argument that does not flicker can still be too close to a level for too
  long. `floor(1e9 + 3e-7*sin(time()))` moves by two and a half last places of
  1e9 in all, and `floor(tanh(u))` creeps onto 1 one last place at a time: each
  is a staircase of its own last places, and nothing says the next step of such
  a staircase is not back across the level. So the call is read, at every
  quarter of an instant (a billionth of the run), for as far as its argument is
  within eight of its last places of a level it crosses. Up to two million
  readings that is done and every change has its stop. Past that the run is
  refused in the same words, naming the call and the longest such stretch. A
  run that is short against its clock (a leg of a few time units at t = 1e9) is
  read whole. A staircase made by step calls and comparisons inside the
  argument (`mod(floor(time()), 24)`) is none of this: where those jump is
  known. Nor is an argument that is a line in time by its own spelling (the
  clock, numbers and parameters that nothing writes during the run, `+`, `-`,
  `*`, and `/` by one of those), however coarse: `ceil(t0 + time())` with
  `t0 = 1e12` moves once in 500 instants of a run 240 long, but a sum or a
  product with a constant never steps back, so it crosses each whole number
  once and is read as any other.

  Far from zero, such a line holds each of its doubles for a stretch of time,
  a tread (1.2e-4 at `1e12 + time()`), and a rate law that reads the clock
  once, inside such a line, is a staircase, however the rest of it is written
  (`floor((1e10 + time())/(0.01))`, as SBML's `quotient` loads, is read through
  `1e10 + time()`). Where the reading above refuses such a law, or leaves it
  without stops, the law is read once on each tread instead, and stopped on
  the last double before each change (issue #1018): between two stops the
  rate is constant, and integrated exactly. So it is with
  `floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)`, a pulse 1.5e-4 wide before
  each wrap, too many to read every quarter instant; with a `mod` or `frac`
  used as a number (not under a comparison) whose argument is coarser than the
  clock and whose period is at most 1,024 doubles of its argument, as in
  `100*mod(1.1e12 + time(), 0.0048828125)` (20); and with one that takes off a
  whole number beyond 2^48 (2.8e14), as `100*mod(1e12 + time(), 0.0015)` does.
  That last returns its argument's rounding, a few steps up to each wrap, and
  not a sawtooth: where its stops are placed, the run is made as the model
  reads it, and warned. Take the offset out of what the call divides for a
  sawtooth.

  A sawtooth with more doubles to its period than that is a staircase of steps
  under a thousandth of the period, the line it nearly is, and is integrated
  as before: `100*mod(1.7e9 + time(), 86400)/86400`, a time of day, over 0.2 is
  14 steps, where a stop on each of its 838,861 treads would be 1.7 million.
  Up to 4.2 million treads of each law are read in a run, and 1,000,000 stops
  placed. A law the reading above refuses that needs more stops than that is
  refused by name, whatever `max_step` is passed, with a leg that fits (past
  the 4.2 million treads, it is refused as it was): a step bound
  in place of its stops left `100*mod(1e12 + time(), 12u)` 0.4% off at the
  default tolerances, and 4.5% from X(0) = 1e4, where in legs each change has
  its stop and the answer is exact to rounding. A sawtooth it leaves without
  stops has its stops placed after every other switch of the run has its own,
  and where they do not fit, or are more than can be read, it is integrated as
  before, with the error test finding each change, and a warning (once a
  session) to run it in legs that fit. Nor is anything read so whose changes
  are closer together than 3.5e-13 of the time (16 times the 2.2e-14 of it
  within which CVODE takes a stop as reached and integrates nothing more):
  late in a run its stops would each be reached at the one before, and the
  stretch between never integrated. Such a law is refused where it was, and
  integrated as before where it was not. So is a rate law that reads such a
  line outside its conditions too, as SBML's `rem` does, written out as
  `x - P*floor(x/P)`: a stop on each change of the `floor` is not a stop on
  each step of `x`. So is a run with two laws read so, whose stops can fall a
  double apart and be taken as one, or with one beside a condition past the
  budget, whose step bound its stops would cut. Runs with sensitivity
  columns, an event's trigger, a counter clock, a function no rate law reads
  and the stochastic methods are read as before.
- **More than 1,000,000 switches in one run.** A repeating schedule or a
  sinusoid that is the only thing in the model that switches is then integrated
  under a step bound of half its shorter stretch. Anything else is refused with
  an error that names the condition: run it in shorter legs with `run_until`, or pass
  `max_step` no larger than half the shortest stretch between two switches of
  the rate law, which then stands for the stops. A `max_step` wider than a
  stretch bngsim knows of is tightened to half of it, and a bound that would
  take more than 1e8 steps is refused rather than started.

A condition that is read by evaluating it costs about 2 ms each time its run
window or a parameter it reads changes (the answer is kept otherwise), which a
loop of short `run_until` legs or a fit over such a parameter pays on every
pass. A condition of a shape bngsim knows costs a few microseconds.

Each stop is a restart of the integrator, and each restart spends about one
unit of the local error tolerance. With tens of thousands of stops in a run
that adds up: a quantity of order `n_stops × atol` is integrated to no better
than that. `mod(time(), 0.0003)` over 10 time units (33,000 stops) accumulates
1.5e-3, which the default `atol` of 1e-8 does not resolve and `atol=1e-14`
does to 2e-8. Tighten `atol` where a run has many stops and small quantities.

A stop sits on the first double at which the model's own arithmetic reads
the rate law's new value, found by asking the evaluator: an edge worked out
from a period and an offset can be a double away from it, and with an offset
in the argument (`mod(1e12 + time(), 0.3)`) billions of doubles of the clock
(issue #891). The compiled right-hand side is built without fused
multiply-add (issue #963), so it changes on the same double as the
interpreter. (A law read tread by tread is stopped on the last double before
each change: above.) A condition an event's trigger reads keeps its stop where it
was put, as does every condition in a run with sensitivity columns. At an
`rtol` looser than 1e-4, each stop the resolvers or the scan put on a change
sits on the last double before the change instead (issue #1226); one on a
counter clock stays where it was put, since the integrated counter lands a
few doubles past it. A step ends on a stop, and the implicit step reads the
rate at its end: on the change, the step that reaches it is taken at the next
stretch's rate, and at a tolerance loose enough for that step to span the
stretch, `if(mod(time(), 24) < 1, 1, 0)` over 1,000 days came to 14,627 for
1,000 at `rtol` 1e-2 (each day's 23 hours off at the rate on, its hour on at
the rate off). A double earlier it is exact to rounding, where the rate law
holds between its changes. One that reads the clock outside every condition
that holds between its changes (a sawtooth, written with `mod`, `frac` or
`floor`, a ramp, a pulse times the clock) keeps its stops where they were put:
spanned by one step, its stretch is read at one end, and the end a double
before the change is no better than the other (`mod(time(), 0.7)` from
X(0) = 1e4 at `rtol` 1e-3 was 19.3 over a double before each change, and 0.45
as the stops were put). A condition that comes in parentheses, as every one an
SBML or Antimony model loads does (`((time() - 24*floor(time()/24)) < 1)`), is
not read as one that holds, and keeps its stops where they were put: an SBML
pulse schedule is main's at every tolerance. Tighter, the step on the change fails on the jump and
the run walks the stretch before it, which finds a pulse inside the stretch
that a step from the double before would not. In a run where a moved stop
would sit a double or two from another stop (two spellings of one schedule,
or another switch on one of its edges), every stop the resolvers placed is
where they put it: the run takes the second of two such as reached where it
restarts on the first, and neither alone is right for both conditions.

CVODE takes a stop as reached once a step ends within 100 doubles' worth of
its time, about 2.2e-14*t, and does not integrate what is left. Where a rate
jumps on a stop by so much that that time's worth of the jump would be a tenth
of the tolerance or more, a run with no sensitivity columns stops CVODE on the
double before the stop, integrates that double on the branch that ends, and
takes the stop from there: stopped on the stop itself, a dose schedule
started at t = 1e6 lost 2e-8 at each stop where it switched off, 180 times
rtol at 1e-9 (issue #891). Elsewhere the step that reaches a stop reads the
new branch and, at a tight tolerance, fails on the jump, so the integrator
walks the stretch before it, which is how a pulse inside it is found.

Where the rate law is flat at a restart, the integrator's first step has
nothing to measure its error by, and the steps after it grow as far as the
integrator lets them (10,000 times the first, then ten times each) until one
takes most of the stretch to the next stop at once. That step reads the law at
its end alone, and on the stop a law that repeats reads there what it read at
the restart, so the step passes with nothing added for the stretch. So, until
a stretch has been looked at, a run watches its steps, and before one that
grew with nothing measured and would take half or more of what is left, it
reads the right-hand side where that step ends, against what the integrator's
own history predicts there. Where the step's error there is well past what the
integrator's error test allows, nothing is done: it fails the step and walks
the stretch itself. Otherwise the step is read at a quarter, a half and three
quarters of its length, and where a step would fail at one of them the run
restarts there, so the step ends on it (issue #998). Before, a dose train that
begins each period flat lost every period from about the 72nd stop on, and
with output rows inside the periods from the first, with no error. A run with
sensitivity columns steps as before.

What this does not see, and can still step over (issue #1192): a pulse that
is nothing at both ends of a step and at the three places inside it; steps
the integrator takes on its way to an output row, where a step that would
cross the row fails and is retried; a stretch while a delayed event is
waiting to fire; the stretch after a stop that restarts nothing (a table
function's breakpoint); and, in a model with a stiff species, a step the look
takes for one that fails. For these, as for a narrow pulse in a run with no
stops, pass `max_step` no larger than the pulse.

In a run with sensitivity columns the same stops are placed for a condition
that reads none of the requested parameters, and for a pulse that only bends.
A step call outside any condition whose step times move with a requested
parameter (`floor((time()-w)/P)` with `w` or `P` requested) is refused by name:
each step is a jump in that column, which is not computed.

A condition that reads the model's state (`if(S > 5, ...)`) switches at a time
nothing knows before the run. Its residual is a CVODE root instead; see
[Events](events.md) and the sensitivities guide for what that does and does
not resolve.

## When a solve fails on a rate law's domain

bngsim evaluates a rate law literally. A logarithm, a `sqrt` or a fractional
power of a negative concentration is `nan`, and that `nan` is kept rather than
papered over — the same policy the SSA's negative-count warning states. What the
integrator does with it is give up, historically with nothing but a number:

```
CVODE integration failed at t=1.000000 with flag=-4
```

Such a failure now names what it found (GH #336):

```
CVODE integration failed at t=1.000000 with flag=-4 (CV_CONV_FAILURE). The
compiled RHS returned a non-finite value at t=1. Non-finite there:
logterm() = ln(1 - Atot) -> nan. Species below zero there: C() = -1. bngsim
evaluates a rate law literally, so a state outside its domain ...
```

The first sentence says which callback saw it and when — `The RHS`, `The
compiled RHS`, `The Jacobian`, `The compiled sensitivity RHS` — which separates
a right-hand side that leaves the domain from a *derivative* that does. What
follows is the state read back at that point: the rate laws that evaluate
non-finite there, and any species below zero.

The check costs a healthy solve nothing and says nothing about one. CVODE
routinely takes a step that dips a zero-pinned species slightly negative and
recovers from it, and a run that finishes prints no diagnostic at all. Only a
run that *fails* replays the offending state, and it does so through the
interpreted evaluator — so a model's own instrumentation
(`bngsim: warning: 'ln(-1e-09)' returned nan`) appears on stderr alongside the
message, even for a compiled run that never touched the interpreted right-hand
side.

The fix is in the model: constrain the species, or write the law so it is
defined where the solver will look. `ln(abs(S))` — identical to `ln(S)` for
every `S > 0` — is the cheapest spelling. An `if()` guard works too but is not
free: a state-dependent condition in a *declared* rate law reads as a state
switch whose crossing time moves, which the forward-sensitivity path declines
(the same reason the GH #333 zero-base rewrite lands on a function's evaluation
expression and never on its declared one).

## Logging

```python
import logging
bngsim.configure_logging(logging.DEBUG)
# Now all bngsim operations produce log output
```
