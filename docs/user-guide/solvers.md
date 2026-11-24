---
summary: "ODE solver settings: tolerances and their defaults, per-species and tracking atol, the Jacobian, dense and sparse (KLU) linear solvers, rate laws that switch on time, runs far from t = 0, failed solves, and logging."
---

# Solver configuration

This page covers the settings of the ODE solver (CVODE): tolerances, step
limits, the Jacobian, the linear solver, and what to expect from rate laws that
switch on time, runs that start far from 0, and solves that fail.

The examples share one small model whose species span ten decades, as they do
in a signaling model in molar units (see
[Per-species absolute tolerance](#per-species-absolute-tolerance)):

```python
from pathlib import Path

import bngsim

Path("model.net").write_text(
    """\
begin parameters
    1 k1    1e-8      # IRSiP -> IRp, /s
    2 k2    1.0       # IRp -> IRSiP, /s
    3 kx    0.1       # X -> 0, /s
    4 IRp0  1.763e-9  # initial amounts
    5 IRS0  0.133
    6 X0    10
end parameters
begin species
    1 IRp    IRp0
    2 IRSiP  IRS0
    3 X      X0
end species
begin reactions
    1 2 1 k1
    2 1 2 k2
    3 3 0 kx
end reactions
begin groups
    1 IRp    1
    2 IRSiP  2
    3 X      3
end groups
""",
    encoding="utf-8",
)
model = bngsim.Model.load("model.net")
sim = bngsim.Simulator(model, method="ode")
```

## Set tolerances and step limits

```python
sim.set_tolerances(rtol=1e-10, atol=1e-10)
sim.set_max_steps(50000)

# Or for one run only
result = sim.run(
    t_span=(0, 100), n_points=101,
    rtol=1e-12, atol=1e-12, max_steps=100000,
)

print(result.solver_stats)
# e.g. {'n_steps': 690, 'n_rhs_evals': 722, 'n_jac_evals': 55, ...}
```

Things to know:

- The defaults are `rtol=1e-8` and `atol=1e-8`.
- A scalar `atol` is a **ceiling**, not a fixed value. A species much smaller
  than `atol` is held to a tighter tolerance of its own. See below.
- `atol` also takes one value per species, `"auto"`, `"tracking"`, or
  `bngsim.ExactAtol(value)` to hold every species to exactly that number.
- `max_step=` bounds the size of a single step. Use it for a narrow pulse the
  solver could step over.

### Solver statistics

`result.solver_stats` covers the whole run, across every restart of the
integrator (events, rate-law switches).

- `n_rhs_evals` counts the integration's own right-hand-side evaluations.
- `n_jac_rhs_evals` counts the evaluations spent forming a Jacobian by
  difference quotients. It is 0 when every Jacobian is closed-form.
- `n_jac_evals` counts the linear solver's setups, which can reuse the last
  Jacobian. `n_jac_formations` counts the Jacobians the solver asked for.
- `linear_solver` is `0` for the built-in dense LU, `1` for sparse KLU and `2`
  for the BLAS dense factor.

## The default tolerances, and models in small units

A scalar `atol`, the default or one you pass, is used as is where it is small
next to a species. A species smaller than that is held to `rtol` of its own
start instead:

```
atol_i = min(atol, rtol * |x_i(0)|)
```

`x(0)` is the state the run starts from. A species that starts at 0, or too
small for `rtol` of it to be resolved beside the largest species, has no scale
of its own. It uses the floor in place of `|x_i(0)|`:

```
floor = max(smallest start of a species the solver integrates, 1e-6 * the largest)
```

This matters for models in molar or other small units, common in SBML. Held to
`1e-8`, a species near `1e-9` M has no error control: its trajectory can be off
by its own size or go negative, and the solver reports nothing.

- **It is never looser than the scalar.** A species the scalar already resolves
  keeps it.
- **A model of counts is unchanged.** Its species start at 0 or at least 1, so
  every one gets the scalar.
- **The floor** is the scale given to a species that starts at 0. It is never
  below 1e-6 of the largest species. Where every integrated species starts at
  0, the floor is the smallest start of any species, a `$`-fixed one included.
  Where every species starts at 0, the default is the scalar.
- **Sensitivities on CVODES' difference quotient keep the scalar.** When
  BNGsim cannot derive `∂f/∂p` for a model (`has_analytic_sens_rhs` is
  `False`), a tighter tolerance makes the difference quotient worse, not
  better, so such a run uses the scalar as is.
- **It reads the state each call starts from.** A batch, a scan or a chunked
  sensitivity job derives it once for all its runs. A `run_until` step (and so
  `ReactionKernel.advance`) keeps the one its trajectory started with. For a
  fit that moves initial conditions, pass a fixed vector from
  `bngsim.derive_atol` (below).

The default cannot scale a species that starts at 0 in a model where no species
starts small. When you pass no `atol` and an integrated species stays below its
tolerance while reaching at least 1e-6 of the run's largest species, `run()`,
`run_until()`, `run_batch()`, `parameter_scan()`, `compute_all_sensitivities()`
and `bngsim.jax.differentiable_solve()` warn with `bngsim.AtolScaleWarning`,
once per call, naming the species. `$`-fixed species and assignment-rule
targets are not integrated and are not warned about. The fix is
`atol="tracking"` (below), or an `atol` well below those species.

### A scalar `atol` you pass, and holding it exactly

A scalar `atol` you pass is read the same way, as a ceiling. That includes one
given to `set_tolerances`, and a SED-ML absolute tolerance (`KISAO:0000211`)
that `bngsim.convert.read_sedml` reads.

To hold every species to exactly the number you give, say so:

```python
result = sim.run(t_span=(0, 100), rtol=1e-6, atol=bngsim.ExactAtol(1e-9), start="initial")
sim.set_tolerances(rtol=1e-8, atol=bngsim.ExactAtol(1e-8))

# A SED-ML file, here one BNGsim writes, read back with its atol held exactly
bngsim.convert.write_sedml(bngsim.EvaluationSpec("model.net", atol=1e-9), "simulation.sedml")
spec = bngsim.convert.read_sedml("simulation.sedml", exact_atol=True)
```

- `ExactAtol` works wherever `atol` does: `run`, `run_until`, `run_batch`,
  `parameter_scan`, `compute_all_sensitivities`, `set_tolerances` and
  `steady_state`.
- An `EvaluationSpec` stores it in JSON as `{"exact": value}`.
- SED-ML cannot say "exactly", so `write_sedml` writes the number, and
  `read_sedml(..., exact_atol=True)` reads it back as exact.
- A vector of `n_species` values is always used exactly as given.
- `steady_state(atol=...)` uses a scalar as given. Under `criterion="scaled"`
  it is also the steady-state convergence test's tolerance.

Use `ExactAtol` to compare with a tool that holds one absolute tolerance for
every species.

## Per-species absolute tolerance

`atol` takes one value **per species**, in `model.species_names` order. Use it
when species span many decades and no single number is a tolerance for all of
them:

| species | `y(0)` | the atol it needs (`rtol·y`) |
|---|---:|---:|
| `IRp` | 1.763e-09 | **1.8e-17** |
| `IRSiP` | 1.33e-01 | 1.3e-09 |
| `X` | 1.0e+01 | **1.0e-07** |

```python
# Three ways to give the vector. start="initial" runs each one from the
# initial conditions, not from where the previous run ended.

# Explicit: one value per species, in species_names order.
result = sim.run(t_span=(0, 100), atol=[1e-17, 1e-9, 1e-7], start="initial")

# Derived from the model's state: rtol * max(|y_i|, floor).
result = sim.run(t_span=(0, 100), atol="auto", start="initial")

# Or derive, adjust and pass it back.
model.reset()
atol = sim.auto_atol()
atol[model.species_names.index("IRp")] *= 10
result = sim.run(t_span=(0, 100), atol=atol, start="initial")
```

- **The length is checked.** A vector of the wrong length raises `ValueError`.
  It is never broadcast or cut.
- **Where it is accepted:** `run`, `set_tolerances`, `run_batch`,
  `parameter_scan`, `bifurcate`, `run_until`, `compute_all_sensitivities`,
  `steady_state`, `steady_state_batch` and `EvaluationSpec` all take a vector,
  `"auto"` or a tracking `atol`.
- **One tolerance for many points.** A batch, a scan or a chunked sensitivity
  job resolves the tolerance once, so every point is held to the same one. In
  `run_batch`, `parameter_scan` and `steady_state_batch`, `"auto"` takes each
  species' smallest scale over the points' starts. With a scalar, each
  `run_batch` row and `parameter_scan` point reads the scale both from the
  model and from its own start, and takes the tighter of the two. A row that
  doses a species far below the model's value is held to that species' scale.
- **`"auto"`'s floor.** `"auto"` gives a species at 0 the smallest strictly
  positive species value in the model, but not more than 1. Pass `floor=` to
  `auto_atol` to choose another.
- **It does not help species of similar size.** If every species starts at
  about the same magnitude, the vector equals the scalar it replaces.
- **Sensitivity tolerances follow it.** The tolerance for ∂x_i/∂θ is built from
  species *i*'s own absolute tolerance.
- **`steady_state_tol` still reads the scalar.** The early-stop test in `run()`
  is one norm over all species, so it uses the `Simulator`'s scalar `atol` even
  when you pass a vector. **Pass `steady_state_tol` whenever you pass a
  vector.** Otherwise, on a model whose states are near `1e-8`, the test passes
  at *t* = 0 and the run returns the initial state as the steady state.

## Tracking absolute tolerance

A vector is fixed for the whole run. A species that starts at order one and
decays to something tiny falls below its own `atol` partway through and loses
error control from there on. `atol="tracking"` re-reads the tolerance from the
current state as the run goes:

```
atol_i(y) = clamp(rtol * |y_i|, ceiling_i * 10**-decades, ceiling_i)
```

```python
# Alternatives, each from the initial conditions.
result = sim.run(t_span=(0, 30), atol="tracking", start="initial")  # decades=12
result = sim.run(t_span=(0, 30), atol=bngsim.TrackingAtol(decades=6), start="initial")
result = sim.run(t_span=(0, 30), atol=bngsim.TrackingAtol(ceiling=atol), start="initial")  # the vector above
```

`ceiling` defaults to `"auto"`. Tracking is never looser than its ceiling, and
at `decades=0` it is the ceiling.

`tests/data/deep_decay.net` is a worked example: two species start at 1.0, and
one decays sixteen decades. Against the exact `exp(-t)`, its worst relative
error is 1.8e+04 with the scalar or `"auto"`, 9.5e-02 with `decades=6` and
2.6e-06 with the default 12.

What to know:

- **It costs more steps.** That is why it is not the default.
- **It cannot add digits that are not there.** A species that is a difference
  of large fluxes carries roundoff of order `eps × flux`. Asking for more
  shrinks the step instead of improving the answer. `decades` bounds how far it
  tries.
- **Some models that integrate at the default tolerance fail at the default
  depth.** The error names the depth. **Lower `decades` first.**
- **Sensitivity tolerances do not track.** They are built from the ceiling, as
  for the vector.

### Which state the tolerance comes from

`atol="auto"` and `sim.auto_atol()` read the model's **live** state, the one
the next `run()` starts from. In a parameter fit that moves initial conditions,
that makes `atol` change with each fit point, and the objective jumps where the
tolerance changes. Derive the vector once with `bngsim.derive_atol` and pass it
every time:

```python
import bngsim

model = bngsim.Model.load("model.net")
sim = bngsim.Simulator(model, method="ode")

RTOL = 1e-8
nominal = model.get_state()                          # before anything is fitted
atol = bngsim.derive_atol(nominal, RTOL)             # fixed for the whole fit

search = [{"X0": x0, "k2": k2} for x0 in (5.0, 20.0) for k2 in (0.5, 2.0)]
for theta in search:
    model.set_params(theta)                          # may move initial conditions
    result = sim.run(t_span=(0, 100), rtol=RTOL, atol=atol,
                     start="initial")                # each point from its own start
```

The rule and the `floor` default are the same as `auto_atol`'s. If you build
the vector yourself,
`bngsim.normalize_atol_vector(vec, model.n_species, model.species_names)` runs
the same length check `run()` would, so a mistake shows up where the vector
was built.

## Jacobian strategy

Choose how the Jacobian ∂f/∂y is computed with `jacobian=`:

```python
sim = bngsim.Simulator(model, method="ode")                         # jacobian="auto"
sim = bngsim.Simulator(model, method="ode", jacobian="analytical")  # exact
sim = bngsim.Simulator(model, method="ode", jacobian="fd")          # finite differences
```

| Strategy | What it does |
|---|---|
| `"auto"` (default) | Analytical where available, else finite differences. If the analytical run fails to integrate, it retries once with finite differences. |
| `"analytical"` | Exact derivatives. A failure is raised, not retried. A model with no complete analytical Jacobian (its derivation ran out of budget, or `BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0` is set) runs on finite differences. |
| `"fd"` | Finite differences: SUNDIALS' difference quotient for dense models, graph-colored differences for sparse (KLU) ones. |
| `"jax"` | The Jacobian by JAX autodiff, for research: much slower than `"auto"`, and it warns so. Needs `jax` and `jaxlib`; ODE only. |

After a run, `sim.jacobian_strategy` reports what was used: `"analytical"`,
`"fd"` or `"jax"`. It reports `"fd"` whenever finite differences were used,
including an `"auto"` run that fell back.

The retry under `"auto"` exists because a rate law that jumps in a state
variable (an `if()` on a threshold the state sits at) has an exact derivative
that ignores the jump, which can stop CVODE's corrector. Use `"fd"` to compare
against the analytical Jacobian, or if you suspect it. For a system SUNDIALS of
another release, see [Installation](../installation.md#from-source).

### Derivation budgets

For a model with Functional or Michaelis-Menten rate laws, the analytical terms
are derived when the model loads. That derivation has a time budget so a
pathological model cannot hang the load:

- **`BNGSIM_JAC_DERIV_BUDGET_S`** bounds the Jacobian derivation. When it runs
  out, the model uses the finite-difference Jacobian. The budget grows with the
  number of species, and has no limit at 20,000 species or more, where a
  finite-difference Jacobian is not practical. Set it in seconds, or to `inf`,
  `none` or `0` for no limit.
- **`BNGSIM_SENS_DERIV_BUDGET_S`** bounds the sensitivity derivation, which
  only a sensitivity run does. It scales the same way and takes the same
  values, but it is independent. Unlike the Jacobian budget, it stays bounded at
  any model size.
  When it runs out, the run falls back to CVODES' difference quotient, which is
  much slower per sensitivity column and, on a stiff model at a tight `rtol`,
  can exhaust `max_steps`. The fallback is logged as a warning that names this
  variable. If it hits a model you care about, raise the budget:

  ```bash
  BNGSIM_SENS_DERIV_BUDGET_S=inf python fit.py
  ```

- **Expression output sensitivities** (`Result.output_sensitivities("expression:…")`)
  are derived under the same variable, with a deadline of their own and a budget that
  grows with the number of expressions and derivatives to take. When it runs
  out, the expressions already derived still work. Selecting one that was not
  raises an error naming the budget. Species and observable sensitivities are
  not affected.

## Linear solver

Each Newton iteration solves a linear system. How that system is factored is
chosen per model, separately from how the Jacobian is computed.
`result.solver_stats["linear_solver"]` reports the choice.

| Route | When (default options) |
|---|---|
| KLU (sparse) | at least 50 species, a Jacobian under 10% dense with a structural nonzero to factor, and an LU factor that stays sparse (below) |
| BLAS dense factor | a model KLU would take whose LU factor fills in, on a build with BLAS |
| built-in dense LU | everything else |

**The LU-fill test.** A sparse Jacobian can still have a dense LU factor. In
rule-derived networks, a few hub species (a free enzyme, an adaptor) couple to
most complexes, and the factor fills in. KLU then spends most of the run
refactoring. So a model of 256 to 5,000 species whose estimated fill is at
least 10% takes the BLAS dense factor instead, when:

- the build has a BLAS dense factor (`bngsim.HAS_LAPACK_DENSE`: the macOS
  wheels, and source builds that find LAPACK); and
- the analytical Jacobian is in use (`sim.jacobian_strategy == "analytical"`).
  With `"fd"`, KLU stays, because it colors its differences.

`sim.model._core.jacobian_sparsity["lu_fill_estimate"]` reports the estimate,
from KLU's own symbolic analysis.

Overrides:

- `force_sparse_linear_solver=True` keeps a model on KLU.
- `force_dense_linear_solver=True` takes the built-in dense LU, or with
  `BNGSIM_LAPACK_DENSE=1`, the adaptive BLAS solver.

The route changes the cost, not the answer: trajectories agree to the solver
tolerance. `steady_state()` picks its route by the same rule and reports the
BLAS route as `"lapack-dense"`.

**`BNGSIM_LAPACK_DENSE=1`** turns on the BLAS solver for other dense models. It
starts on the built-in factor and switches to BLAS after five factorizations,
so a short run pays nothing extra. The count runs per integration: the steps of
`run_until` and `ReactionKernel.advance` continue it, and a new `run()` starts
it over. `solver_stats["n_dense_blas_factorizations"]` counts the BLAS
factorizations.

## Rate laws that switch on time

A rate law that is flat on both sides of a short window, such as
`if(time() >= 100 && time() <= 100.5, k, 0)`, shows an adaptive solver no
error, so a step can jump over the window entirely. BNGsim finds the times
where such a rate law changes before the run, stops the integration at each,
and restarts there. It handles:

- comparisons of the clock with constants, in an `if()` or as a number
  (`k*((time()>=100)*(time()<=140))`);
- `floor`, `ceil`, `trunc`, `round`, `rint`, `frac`, `sign` and `mod` of the
  clock, and repeating schedules written with them;
- windows written with `abs`, `min`, `max`, powers, exponentials or sinusoids,
  and pulses that only bend (`max(0, 1 - abs(time()-t0)/w)`);
- the breakpoints of a table function over time;
- the same conditions on a counter species fed at a constant rate;
- smooth bumps: `exp` of a quadratic in the clock with a negative leading term
  (also written `_e^` or `pow(_e, ...)`), and `cosh` or `tanh` of a linear
  function of it, such as `1/cosh(u)^2`. BNGsim stops at each bump's center. A
  bump centered just past the end of the run gets a stop at the end.

A condition of a shape BNGsim knows is solved exactly. Any other is evaluated
over the run on finer and finer grids until the number of changes agrees, and
a stop is placed at each. That evaluation is redone when the run window or a
parameter it reads changes, which adds up in a loop of short `run_until` legs
or a fit.

**Many stops cost accuracy.** Each restart spends about one unit of the local
error tolerance. With tens of thousands of stops, a quantity of order
`n_stops × atol` is lost: `mod(time(), 0.0003)` over 10 time units (33,000
stops) accumulates 1.5e-3 at `atol=1e-8`, and 2e-8 at `atol=1e-14`. Tighten
`atol` for runs with many stops and small quantities.

Between stops, after a restart where the rate law is flat, the run checks each
long step at its quarter points before taking it, so a pulse at the start of a
stretch is not stepped over. It cannot see a pulse that is zero at all those
points, or one inside a step shorter than a quarter of what is left of its
stretch, and runs with sensitivity columns skip the check. For these, and for a
narrow pulse in a run with no stops, pass `max_step` no larger than the pulse.

### What it does not cover

- **A smooth pulse of another shape**, such as `exp(-u)/(1+exp(-u))^2`, or a
  bump on a counter whose feed reads the state or is changed by an event.
  Pass `max_step` no larger than the pulse's width. A counter that an event
  resets gets a stop only on the first bump, and a warning says so.
- **A window narrower than 1/8,192 of the run that no sample can see**, such as
  `exp(-((time()-t0)/s)^2) > 0.5` with a tiny `s`, or one made with the `%`
  operator. Windows made by the calls listed above are found at any width.
- **A condition whose changes cannot be counted**, such as
  `sin(1/(time()-t0)) > 0.5`. The run is refused with an error that names it.
  Run it in shorter legs with `run_until`, or pass `max_step` no larger than
  half the shortest stretch between switches, which then stands in for the
  stops.
- **A step call whose argument rounds back and forth across a level.**
  `sign(time()^3 - 15*time()^2 + 75*time() - 125)` flips between -1 and 1 near
  5, because the terms cancel below their rounding. The run is refused with an
  error naming the call and the stretch. Write it as `(time() - 5)^3`.
  `max_step` does not help here.
- **A step call whose argument stays within rounding of a level.**
  `floor(1e9 + 3e-7*sin(time()))` and `floor(tanh(u))` creep along their last
  places near a level. BNGsim reads such a call finely for up to two million
  evaluations and places a stop at every change. Past that, the run is refused
  with an error naming the call and the stretch.
- **More than 1,000,000 switches in one run.** A repeating schedule or sinusoid
  that is the only switch in the model is then run under a step bound of half
  its shortest stretch. Anything else is refused with an error naming the
  condition. Run it in shorter legs with `run_until`, or pass `max_step` no
  larger than half the shortest stretch between switches. A `max_step` wider
  than a stretch BNGsim knows of is tightened to half of it. A bound that would
  take more than 1e8 steps is refused.

### Step calls on a clock far from 0

Far from 0, a line such as `1e12 + time()` holds each double for a stretch of
time, so a step call on it (`floor((1e10 + time())/0.01)`, a `mod` of
`1e12 + time()`) is a staircase. Where the reading above refuses such a law or
leaves it without stops, BNGsim reads it one double of its argument at a time
instead, with a stop on each change. Between stops the rate is constant and is
integrated exactly.

- **Up to 1,000,000 stops.** A law that needs more is refused by name, whatever
  `max_step` you pass, because a step bound does not keep it within tolerance.
  The error gives a leg length: run it in `run_until` legs no longer than that.
- **A slow sawtooth** such as a time of day,
  `100*mod(1.7e9 + time(), 86400)/86400`, is integrated as the near-straight
  line it is. One whose stops do not fit is integrated without them, with a
  warning to run it in legs that fit.
- **A `mod` whose argument is more than 2^48 periods from 0** returns its
  argument's rounding, not a sawtooth. The run uses it as the model reads it,
  and warns. Take the offset out of what the call divides to get a sawtooth.
- **SBML's `quotient`** on such a clock is refused by name.
- **`x - P*floor(x/P)` and SBML's `rem`** on such a clock are not read this
  way: they are refused where the reading above refuses them, and otherwise run
  without stops.
- **Not read this way:** changes closer than 3.5e-13 of the time apart, two
  such laws in one run, one beside a condition past the stop budget, more than
  4.2 million treads in a law, and runs with sensitivity columns, an event
  trigger, a counter clock, or a stochastic method.

### With sensitivities

In a run with sensitivity columns, the same stops are placed for conditions
that read none of the requested parameters, and for pulses that only bend. A
step call whose step times move with a requested parameter
(`floor((time()-w)/P)` with `w` or `P` requested) is refused by name, since the
jump in that column is not computed.

A condition that reads the model's state (`if(S > 5, ...)`) switches at a time
not known before the run. It is handled as a CVODE root instead; see
[Events](events.md) and the sensitivities guide.

## A run that starts far from t = 0

The doubles are coarse far from 0: 1.2e-4 apart at `t = 1e12`, 0.125 apart at
`1e15`. A step shorter than that spacing is rounded, and the trajectory goes
wrong.

**A model that reads no clock** is moved. An ODE run (`method="ode"`) whose end
nearer 0 is at 2^20 (1,048,576) or beyond is integrated on the time since its
start, and reports the times you asked for. The start is first rounded toward 0
to the spacing of the doubles at the run's far end. Where the start is already
on that spacing, the run from the far start is the run from 0, to the bit:

```python
far = sim.run(t_span=(1e15, 1e15 + 10), n_points=11, start="initial")
near = sim.run(t_span=(0, 10), n_points=11, start="initial")
# far.species equals near.species
```

Otherwise it is the run from a start less than that spacing, as accurate but
not identical in every bit.

Exceptions:

- A run below 0 that starts more than twice as far from 0 as it ends is counted
  from its end, so its first rows are only as fine as the doubles there. Start
  such a run at 0 or above, with the offset written into the model.
- A run whose end nearer 0 is within 2^20 of 0, or that crosses 0, keeps its
  clock. One that starts far below 0 has coarse first rows.
- `run_diffrax` keeps the clock it is given. From `1e15` it ends at its step
  limit with an error.
- If a far run fails with a message naming a time, that time is on the run's
  own clock, and the message says which time to add to it.

**A model that reads the clock** keeps it, far or not: a function of `time()`,
a table function over time, a discontinuity trigger, or any event. Its steps are only as fine as the
doubles there, with no error. A decay whose rate reads `time()` is 1.8e-3 off
at `t = 1e12` and off by a factor of 11 at `1e15`. Run such a model from near 0,
with the offset written into the model. SSA and PSA count their own clock from
a far start for every model.

**A switch on a counter clock** (an `if()` on a species with `C' = 1`) is
placed from the counter, so a far start does not move it. Its stop still sits
on the run's clock, so from `1e16` on a far run can differ slightly from the
run from 0. Late switches in a very long run are taken where the clock is
coarse, and their sensitivities can be badly wrong, or lost, with no error. A
sensitivity run is refused with `SensitivityRefusedForParametersError` (a
`SensitivityUnsupportedError`) if a counter crossing inside the run, which
moves a requested column, rounds onto the start or past the run's end. A
column the crossing does not move can still be run alone.

## When a solve fails on a rate law's domain

BNGsim evaluates a rate law literally. A logarithm, `sqrt` or fractional power
of a negative concentration is `nan`, and BNGsim keeps it. The solver then
fails, and the error says why:

```
CVODE integration failed at t=1.000000 with flag=-4 (CV_CONV_FAILURE). The
compiled RHS returned a non-finite value at t=1. Non-finite there:
logterm() = ln(1 - Atot) -> nan. Species below zero there: C() = -1. bngsim
evaluates a rate law literally, so a state outside its domain ...
```

The first sentence names the callback (`The RHS`, `The compiled RHS`,
`The Jacobian`, `The compiled sensitivity RHS`) and the time. Then come the
rate laws that are not finite there and any species below zero. A run that
succeeds pays nothing for this. A failed run replays the state through the
interpreted evaluator, so a model's own warnings (`bngsim: warning:
'ln(-1e-09)' returned nan`) can appear on stderr even for a compiled run.

A species whose own law drives it to minus infinity in finite time also fails
this way. `X' = -(1 + X^2)` from `X = 1` fails at `t = 3*pi/4`, naming the law.
Some such failures can name only the state.

Fix it in the model: constrain the species, or write the law so it is defined
where the solver looks. `ln(abs(S))` equals `ln(S)` for every `S > 0` and is
the cheapest fix. An `if()` guard also works, but a state-dependent condition
in a declared rate law is a switch that forward sensitivities refuse.

Before it gives up, a run whose right-hand side or Jacobian is not finite tries
once more with concentrations below 0 read as 0. The solver's own error often
puts an emptied species a hair below 0, where a `sqrt` of it is `nan`. Only
the species the failing laws read are moved, and only ones that are
concentrations: a species declared with a negative initial value, a fixed
species and a rate-rule target keep their values. A species further below 0
than any rounding of 0 (10^6 of its atol) keeps its value where the law stays
finite without the change, and once it has gone that far, it keeps its value at
any depth for the rest of the run. So a voltage at -61 keeps its value beside a
`sqrt` of another species. At a loose atol this threshold can reach a species'
own range: at `ExactAtol(1e-4)` it is 100, so a voltage that crosses 0 on its
own is read as 0 until it first passes -100.

## Logging

```python
import logging
bngsim.configure_logging(logging.DEBUG)
```

### SUNDIALS' own lines

SUNDIALS writes its own lines straight to the process's streams. They do not
pass through `sys.stdout`, `sys.stderr`, `warnings` or the `bngsim` logger, so
`contextlib.redirect_stdout`, a warnings filter and a logging level do not
catch them.

```
[WARNING][rank 0][...cvodes.c:3482][CVode] Internal t = 1e+06 and h = 1.49012e-13
are such that t + h = t on the next step. The solver will continue anyway.
```

Both `[ERROR]` and `[WARNING]` lines go to **standard error**. A warning can
come from a run that is right; the one above is. Two SUNDIALS environment
variables redirect them, and BNGsim respects them when set:

```bash
SUNLOGGER_WARNING_FILENAME=/dev/null python script.py   # drop the warnings (NUL on Windows)
SUNLOGGER_WARNING_FILENAME=stdout    python script.py   # send them to standard output
SUNLOGGER_ERROR_FILENAME=/dev/null   python script.py   # drop the error lines
python script.py 2> sundials.log                        # both, with the rest of stderr, to a file
```

- **Set them before the `Simulator` first runs.** They are read when a solver
  context is made. A `Simulator` running a model with no events, switches or
  sensitivities makes one on its first run and reuses it. Other runs, each
  `run_batch` row and each steady-state solve make their own, and read the
  variables again.
- **A file name is overwritten, not appended.** Each solver context empties
  the file, and `run_batch` rows on several threads write over each other. Use
  `stdout`, `stderr` or the null device, and redirect the stream to keep a log.
- **A name SUNDIALS cannot open** (a file in a missing directory) stops every
  run and steady-state solve, with an error that names the variable.

BNGsim drops the warnings itself for runs whose rate laws switch on state, or
on time in an SBML model, since they restart at each switch, and for KINSOL
solves, unless `SUNLOGGER_WARNING_FILENAME` is set. A BNGL rate law that
switches on time keeps its warnings. The `[ERROR]` lines of a solve BNGsim
expects to fail and recovers from are always dropped.
