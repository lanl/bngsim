---
summary: "Find a steady state, choose a method, read what converged means, find every steady state, sweep doses, and compute steady-state sensitivities."
---

# Steady states

A steady state is a state where nothing changes: `f(y) = 0`. This page shows
how to find one, how to sweep it over doses, and how to get its derivatives
with respect to parameters.

## Find a steady state

```python
import bngsim

model = bngsim.examples.load("binding")      # A + B <-> C
sim = bngsim.Simulator(model, method="ode")

ss = sim.steady_state()
print(ss.converged)                          # True
print(ss.method_used)                        # integration
print(round(float(ss["A(b!1).B(a!1)"]), 2))  # 62.98
print(ss.concentrations.shape)               # (3,)
print(ss.to_dict())                          # {name: value} for every species
```

The solve starts from the model's current state. A conserved total, such as
`A + C` here, is the one that state holds.

Watch out for:

- **ODE only.** `steady_state()` on an `"ssa"` or other stochastic Simulator
  raises `ValueError`.
- **A model with an event, or one that reads the time, is refused** with
  `SimulationError`. The solvers read the rates at `t = 0` and fire no event,
  so the answer would be wrong. A `time()` call, or a table function indexed
  by time, in a rate law or in a rule that a species or compartment size is
  reported through, counts as reading the time. `rateOf` does not. Integrate
  such a model with `run()` over a span long enough for it to settle.
- **A function that reads the time but only reports,** such as
  `minutes := time/60`, is allowed. The state does not depend on it. Its value
  in the result, and its sensitivity row, is NaN, and `output_sensitivities`
  refuses it.
- **`converged=False` with a nonempty `ss.unconverged_pure_sinks`** means an
  accumulator species is holding the solve up. Use `mask=` (see
  [Accumulator species](#accumulator-species-mask)). An empty list means the
  cause is something else, and `max_time`, `tol` or `max_steps` are worth
  trying.
- **A conservation law the model's own rates do not keep** is refused with
  `SimulationError`. The solvers hold every law the model reports.

The common options:

```python
ss = sim.steady_state(
    method="integration",  # or "newton"; "kinsol" is an alias for "newton"
    tol=1e-9,              # the residual test, ||f||_2 / n_species < tol
    max_time=1e6,          # the longest the march integrates, and the run test's run
    rtol=1e-8,             # the march's relative tolerance, and the run test's
    atol="auto",           # a scalar, one value per species, or "auto"
    max_steps=50000,       # CVODE steps per output point
    criterion="scaled",    # or "residual", BNG2.pl's test alone
)
```

## Choose a method

- **`method="integration"` (the default)** integrates with CVODE until the
  convergence test passes, or until `max_time`. For a model that does not read
  the time, it returns the steady state the dynamics reach.
- **`method="newton"`** runs the same integration first, then polishes the
  state with KINSOL's Newton solver. The polish is kept only if two Newton
  solves from successively tighter starts land on the same state, and only if
  the root is stable (see [Unstable roots](#unstable-roots-saddles)).
  Otherwise integration goes on. So a Newton solve always meets the same
  convergence test. In a model with conservation laws, Newton solves on the
  reduced system automatically.
- **`method="kinsol"`** is an alias for `"newton"`. `ss.method_used` always
  says `"newton"` or `"integration"`.

Use the default. `"newton"` only adds work on top of it. Use `"newton"` when:

- you want the root resolved far below `tol` (Newton reaches a residual near
  1e-13, where integration stops as soon as it crosses `tol`), for example
  when the steady state feeds a stiff solve; or
- you have cut `max_time` well below its default. Newton can reach `tol` from
  a state integration has not yet brought under it.

## What converged means

`ss.converged` is decided by one test on every path. Under the default,
`criterion="scaled"`, a state passes when it passes both:

- **The residual test,** `||f(y)||_2 / n_species < tol`. This is BNG2.pl's
  `run_network -c` test. `ss.residual` is its value.
- **The run test.** A run taken on from the state for `max_time` (at most 1e6)
  moves no species by more than its tolerance, `rtol·|y_i| + atol_i`.
  `ss.continued_run_distance` is the run's largest move, in tolerances. At
  most 1 passes.

The residual is a rate, in the model's units. A model in molar units, or a
slow one, can pass `tol` far from its steady state. The run test catches that.

`criterion="residual"` keeps the residual test alone, for comparisons with
`run_network`. It takes no run and no Newton step, and
`ss.continued_run_distance` and `ss.newton_distance` are NaN. It is also the
test `run(steady_state=True)` uses.

```python
ss = sim.steady_state()
ss.criterion               # "scaled"
ss.residual                # the residual test's value
ss.continued_run_distance  # the run's largest move, in tolerances
ss.newton_distance         # the Newton step to the root, in tolerances
```

### Where the solve stops

Up to 512 unknowns, the solve uses the Newton step from the state to the
root, `δ = −J⁻¹f`, measured in the same tolerances, to decide where to stop.
The march stops at a state that passes `tol` and whose step is within
tolerance. KINSOL then polishes that state. The polish is kept if it passes
the residual test, is nearer the root, and moves no species more than a
thousand tolerances. The step decides where the solve stops. It does not
decide `converged`.

Sometimes the step cannot be passed: on a continuum of steady states, where
the Jacobian is singular, or where a species goes to 0 as 1/t. The march then
goes on to `max_time`. If it ends at a state that fails the residual test, the
last state that passed it is returned.

Above 512 unknowns there is no step. The state is polished directly if the
polish converges, lowers the residual and moves no species more than its
tolerance. Otherwise a run is taken on from the state first, for `max_time` or
1e6, whichever is shorter, and the polish starts where that run ends and moves
no species more than a thousand tolerances. If the run cannot reach its time,
the unpolished state is returned. A state that comes from such a run has
moved on in time with every species, so an accumulator left out by `mask=`
has gone on growing.

### Tolerances

`rtol` defaults to the Simulator's `rtol`, 1e-8 unless `set_tolerances`
changed it. For the run test it is at least 1e-10, because a run
of 1e6 cannot tell a smaller move from its own error.

`atol_i` is the march's absolute tolerance. It is never more than `rtol` times
the largest species the solve starts with or reaches. If neither `atol=` nor
`set_tolerances` gives one, each species takes the smaller of what
`atol="auto"` gives it and 1e-8. `"auto"` is based on the smallest species in
the starting state that is not `$`-fixed, but no finer than double precision
allows beside the largest. If every such species starts at 0, a `$`-fixed
species sets it.

The run test's run is integrated at the march's `atol`, but never coarser than
`rtol` times the largest species the solve starts with (or, where every species
starts at 0, the largest of the state). A scalar `atol=` is taken as given, so
it can be coarser than the whole model. Then the march controls no species'
error and stops where that error leaves it, and a run at the same `atol` does
not see the state move. BIOMD0000000609's species are all below 1.4e-13. At its
SED-ML file's `atol=1e-9` the march stopped on a root of `f` with two species
near -1.5e-14, which a run from it leaves at once, and the solve said
`converged=True` (issue #1895). The run held to the start's scale refuses it.
With no `atol`, the default is never that coarse and the run is unchanged. To
solve such a model, leave `atol` unset or pass one at its scale.

If a run moves the state, it is taken again at a tenth of the tolerances, a
hundredth, and (up to 512 unknowns) a thousandth. A move of more than 10
tolerances that does not halve at a tenth of them is a real drift, and the
state is refused.

### What the test does not catch

- **A drift too slow to move a species by its tolerance** within `max_time`
  (or 1e6) passes.
- **A species below zero** is not refused for being there. Some models' rate
  laws take a species below zero, and a run rests there too. A state the march
  reached only through its own error, which a run leaves, fails the run test.
- **An unstable root that a run from it does not leave** passes, as it passes
  the residual test. A state exactly on the middle root of a switch is one.
  A march started near such a root, or near the boundary between two basins,
  can stop at it or go to either side. The Newton path labels an unstable
  root (`root_stability="unstable"`). Above 512 unknowns it labels none
  (`"undetermined"`), and Newton goes wherever KINSOL goes.
- **A species `mask=` leaves out that overflows within the run** fails the
  run test, and the solve is refused. Pass a `max_time` short enough for it
  to stay finite.
- **A species `mask=` leaves out is held during the polish.** If it shares a
  total with other species that the solver does not know is conserved, that
  total can be off by as much as the polish moves the rest, up to a thousand
  tolerances. For example, `X -> 0` and `0 -> Y` at the same rate keep
  `X + Y` fixed; with `Y` left out, the polish can move `X` and leave the sum
  short.
  `converged` speaks only for the species the mask keeps.
- **A species that approaches 0 as 1/t** (`A + B -> C` at equal totals,
  `2A -> C`) is refused if, at `max_time`, it is still more than about 500
  tolerances from 0. A longer `max_time` takes it on.
- **A species that starts at 0 and settles below `rtol` times the largest
  species** is resolved only to that level.
- **If every species starts at 0,** the march uses the scalar 1e-8, and a
  model whose species settle far below that is resolved only to it.
- **Above 512 unknowns,** a run whose own error moves the state past
  tolerance is refused.
- **A Newton solve with no integration first** (`steady_states()`, or
  `steady_state_batch(..., newton_burst=False)`) takes no run. It accepts a
  root by the residual test, and `ss.criterion` says `"residual"`.

## The Jacobian and the linear solver

These choices change the cost, not the answer. The root must pass the same
test either way.

- **The right-hand side** is whatever the Simulator's
  [`codegen_backend`](codegen.md) uses. `ss.rhs_backend` says which ran:
  `"codegen-so"`, `"codegen-jit"` or `"exprtk"`.
- **The Jacobian.** The march and the polish both use the model's closed-form
  Jacobian when it has one and `jacobian=` allows it. `jacobian="fd"` uses
  difference quotients, one right-hand-side evaluation per unknown.
  `ss.solver_jacobian_source` says which: `"codegen"`, `"analytical"` or
  `"finite-difference"`.
- **The retry.** A rate law with a jump in a state variable can make the
  march fail with the exact Jacobian. Under `jacobian="auto"` the march is
  then retried once on difference quotients, as `run()` does, and
  `ss.solver_jacobian_retried` is `True`. After one such retry, the
  Simulator uses difference quotients on its later steady-state solves too.
  `jacobian="analytical"` does not retry, so the solve comes back with
  `converged=False`.
- **The linear solver.** The march picks sparse KLU or a dense LU by the same
  rule as `run()` (see [Linear solver](solvers.md#linear-solver)).
  `force_sparse_linear_solver` and `force_dense_linear_solver` on the
  Simulator apply here too. `ss.linear_solver` says which ran: `"klu"`,
  `"dense"` or `"lapack-dense"`. With `jacobian="fd"` on KLU, the march uses a
  colored difference quotient. `jacobian="jax"` always uses a dense matrix.
  The KINSOL polish and the sensitivity solve always factor dense.

## Accumulator species (`mask=`)

A species that some reaction makes and none consumes, such as a "degraded"
or "secreted" pool that counts total flux, grows at a constant rate forever.
Its derivative never reaches 0, so the residual has a floor above `tol` and
the solve fails however long it runs.

`mask=` restricts the test to the species that do have a steady state.
`model.is_pure_sink()` finds the accumulators, so you do not have to list
them:

```python
model.pure_sink_species()     # the names of the accumulators

ss = sim.steady_state(method="newton", mask=~model.is_pure_sink())
ss.converged                  # True
ss.n_residual_species         # how many species the test covered
ss.excluded_species           # indices of the species left out
```

`mask` takes a boolean array of length `n_species` (`True` keeps a species)
or a list of the species names to keep. Integer indices raise `TypeError`,
because `[0, 1]` could mean either. A boolean array of the wrong length, an
unknown name, or a mask that leaves out every species raises `ValueError`.

What the mask changes:

- Every species is still integrated. Excluded species come back in
  `ss.concentrations`.
- The residual is averaged over the kept species, so `tol` keeps its meaning.
- Newton and the sensitivity solve leave the excluded species out. They must:
  an accumulator makes both systems singular. Excluded species are held where
  integration left them.
- An excluded species has a NaN sensitivity row. It has no steady value, so
  it has no steady-state derivative. An observable or function that reads it
  is NaN too.

`model.pure_sink_species()` is structural. A species is listed if some
reaction makes it, none consumes it, no other species' rate reads it, and it
is not `$`-fixed. A listed species does not mean the solve will fail: in
`A -> B` with nothing feeding `A`, `B` is a pure sink, and the model converges
because the flux dies out. `ss.unconverged_pure_sinks` names the accumulators
that actually held a failed solve up. A failed solve also logs this as a
warning.

`steady_state_batch(mask=...)` applies one mask to every entry.
`run(steady_state=True)` takes no mask: on an accumulator model it runs its
full span.

## Species set by an SBML assignment rule

A species defined by an `<assignmentRule>` has no equation of its own, so it
would make the system singular. It is left out of the solved species on
every solve, automatically, and listed in `ss.excluded_species` together with
anything your own `mask=` leaves out. Its reported value is the rule evaluated
at the returned state. Its sensitivity row is the chain rule through the
rule:

```python
ss = sim.steady_state(sensitivity_params=["ks"])
ss["S"]                                   # the rule's value at the steady state
ss.output_sensitivities(["observable:S"]) # the same derivative as its sensitivity row
```

## Unstable roots (saddles)

Not every root of `f(y) = 0` is a state the system can rest at. A bistable
model has three roots: two stable ones and a saddle between them. Newton can
land on the saddle, because a trajectory slows down as it passes near it.

So before a Newton polish is kept, BNGsim takes the eigenvalues of the
Jacobian over the species the polish solved for. If any has a positive real
part, the root is rejected and integration goes on.

```python
ss = sim.steady_state(method="newton")
ss.root_stability             # "stable"
ss.n_unstable_roots_rejected  # how many saddles were rejected on the way
ss.eigenvalues                # the spectrum the verdict was read from
```

- `root_stability` is `"stable"`, `"unstable"` or `"undetermined"` for a root
  Newton returned, and `""` for an integration result. A trajectory cannot
  come to rest on an unstable root, so integration needs no check.
- `"undetermined"` means no verdict: more than 512 unknowns, a Jacobian
  whose spectrum could not be computed, or a Jacobian that is all zeros
  (whose zero eigenvalues are still reported). The root is kept.
- `"unstable"` is returned only when your initial condition was already on
  that root. Integration would return the same state, so there is nothing to
  fall back to.
- `n_unstable_roots_rejected > 0` with `method_used == "integration"` explains
  why the polish did not answer.
- `eigenvalues` has one entry per unknown: `n_species` less one per
  conservation law, less the masked species. It is sorted by descending real
  part, with each conjugate pair adjacent, `+imag` first. The full system's
  spectrum adds one zero per conservation law;
  `np.linalg.eigvals(model.jacobian(ss.concentrations))` shows them (see
  [Model evaluators](evaluators.md)). It is empty when no spectrum was
  computed: for every integration result, and above 512 unknowns. A Hopf
  bifurcation shows up as the leading conjugate pair crossing the imaginary
  axis.

This check says whether the returned root is stable. It does not say how many
other roots exist. For that, use `steady_states()`.

## Every steady state: `steady_states()`

`sim.steady_states()` looks for all the roots of one parameter set. It runs
Newton from many starting states, with no integration first, and groups the
roots it reaches. Every root is kept and labeled.

```python
roots = sim.steady_states(n_starts=200, seed=0)
roots.states          # one row per distinct root
roots.stability       # "stable" or "unstable" for each root
roots.eigenvalues     # the spectrum at each root
roots.multiplicity    # how many starts reached each root: a rough basin size
roots.basins          # the root each start reached; -1 where Newton failed
roots.n_failed        # how many starts failed
```

**Starting states.** Each species is drawn log-uniformly over `decades`
decades (default 4) up to a bound. A species in no conservation law is
bounded by `upper`, which defaults to ten times the largest value in the
model's state, its law totals and one ordinary steady state (10 if all are
0). Species in a law are drawn below the largest total, then moved together
onto the law at the totals the model's state holds. Members of a law start
within about `decades` decades of each other; widen `decades` to reach a
root where one is much further below its total. A `$`-fixed species, and one
`mask=` leaves out, keeps its value in every start.

**Accepting a root.** A root is accepted by the residual test, with `tol` as
in `steady_state()`. A model so small that most starts already pass `tol` is
refused, since the residual says nothing about where the roots are. With law
totals above about 1e7, rounding alone can exceed `tol`, and such a root
counts as a failed start.

**Failed starts.** Newton does not converge from every start. Failed starts
count in `n_failed` and have basin -1. An accumulator makes every start fail;
leave it out with `mask=~model.is_pure_sink()`.

**Telling roots apart.** Two roots are the same when every species agrees to
`dedupe_rtol` (default 1e-6) of its size. Near 0 a small floor applies: 1e-9
of the largest value the species takes at any converged start, or double precision
beside the largest value seen. A species in a conservation law also agrees
to 1e-9 of its smallest law's scale. A continuum of steady states (a zero
eigenvalue at every root, as on SIR's line of disease-free states) is not a
set of separate roots: each start lands somewhere on it and counts as its own
root.

**Your own starts.** `steady_state_batch(states=..., method="newton",
newton_burst=False)` runs the same Newton solves from starting states you
give. With `params=` as well, the two pair up row by row. `certify=False`
skips the stability check, and `certify="start"` takes no step and checks
each starting state as it is.

This is a search, not a proof. A root with a small basin, or a saddle Newton
never lands on, can be missed.

## Stop a time course at steady state

`steady_state()` returns only the end point. To get the trajectory up to the
steady state, and stop as soon as it is reached, pass `steady_state=True` to
`run()`. This matches BNG2.pl's `simulate({steady_state=>1})`. After each
output point the run checks `||f(t,y)||_2 / n_species` and stops once it is
under `steady_state_tol`. The `Result` holds only the rows it integrated.

```python
r = sim.run(t_span=(0, 1000), n_points=101, steady_state=True, start="initial")
print(len(r.time))                             # 5: fewer than 101, it stopped early
print(r.solver_stats["steady_state_reached"])  # 1

# steady_state_tol defaults to atol, as in BNG2.pl.
r = sim.run(
    t_span=(0, 1000), n_points=101, steady_state=True, steady_state_tol=1e-9, start="initial"
)
```

- ODE only, and no mask.
- In a model with an event or one that reads the time, the run does not stop
  early. It runs to the end of its span, and `steady_state_reached` stays 0.
  A function that reads the time but only reports does not block the stop.
- With `sensitivity_params` or `sensitivity_ic`, the run also waits for every
  sensitivity column to stop changing, by the same test with `dS/dt` in place
  of `f`, held to the solver's `atol` (or `steady_state_tol`, if tighter). A
  looser `steady_state_tol` does not loosen the columns. A column far larger
  than `atol` may never pass; the run then goes to the end of its span with
  `steady_state_reached = 0`.

## Dose-response sweeps: `steady_state_batch()`

`steady_state_batch()` solves for the steady state of many parameter sets, in
parallel threads:

```python
model.reset()
doses = [50.0, 100.0, 200.0]
results = sim.steady_state_batch(params=[{"A0": d} for d in doses], n_workers=4)
print([round(float(r["A(b!1).B(a!1)"]), 2) for r in results])  # [40.0, 62.98, 74.11]
```

Each entry clones the model, sets its parameters, resets the clone to its
initial conditions and solves. It takes the same solver options as
`steady_state()`, but not `sensitivity_params`. It also takes `states=`,
`n_workers=`, `newton_burst=` and `certify=` (see
[Every steady state](#every-steady-state-steady_states)).

**Each entry starts from the initial conditions,** not from the model's
current state. The reset is what makes a dose over an initial-condition
parameter like `A0` take effect. If the current state differs from the
initial conditions (after a `set_concentration` or a run), the call warns.
To start every entry from the current state instead, pass
`states=[model.get_state()] * len(params)`, or call
`model.save_concentrations()` first so that the reset returns to it. Either
way, a dose over an initial-condition parameter is then dropped. If you want
the doses, call `model.reset()` first, as above.

## Steady-state sensitivities

Pass `sensitivity_params` to get the derivative of the steady state with
respect to parameters, `dY_ss/dp = -J⁻¹·∂f/∂p`:

```python
ss = sim.steady_state(sensitivity_params=["kf", "A0"])
print(ss.sensitivity.shape)   # (3, 2): (n_species, n_params)
print(ss.sensitivity_params)  # ['kf', 'A0']

grad = ss.output_sensitivities(["observable:C"])   # (n_selectors, n_params)
```

- **It needs code generation,** as forward sensitivities do, and so a C
  compiler (see [Code generation](codegen.md)). The derivative code is built
  automatically. `codegen=False`, or `BNGSIM_NO_CODEGEN`, raises.
- **The derivatives are exact.** `J` and `∂f/∂p` come from compiled closed
  forms where the model has them. Where it does not (Michaelis–Menten, a
  condition such as `if()`, a nonsmooth function such as `abs` or `max`, or
  `jacobian="fd"`), they are taken by automatic differentiation through the
  interpreter. Only a model that cannot be followed that way (functions that
  read each other in a ring, or `rateOf`) falls back to difference quotients,
  checked at more than one step.
- **A state exactly where a rate law changes branch** is differentiated from
  each side, and refused where the two sides differ.
- **Conservation laws** are handled for you. The solve uses the reduced
  system and rebuilds the dependent species.
- `ss.rhs_backend`, `ss.sens_jacobian_source`, `ss.sens_dfdp_source` and
  `ss.sens_output_source` say how each piece was computed.
- The initial-condition axis is not computed, and
  `output_sensitivities(..., axis="ic")` raises.
- Without `sensitivity_params`, `steady_state()` returns the state the solver
  stopped at and runs none of the checks below.

### Observable and expression sensitivities

`ss.sensitivity` is per species. `output_sensitivities` gives the
derivatives of observables and expressions, as on a time-course
[`Result`](sensitivities.md):

```python
grad = ss.output_sensitivities(["observable:C"])   # add "expression:name" rows too
ss.observable_names            # rows of ss.sensitivities_observables
ss.expression_names            # rows of ss.sensitivities_expressions
ss.sensitivities_observables   # (n_observables, n_params)
```

Observables are an exact linear map of the species. For an SBML species with
`hasOnlySubstanceUnits="true"`, the map includes its compartment size, since
its observable is an amount. Functions get the full chain rule: their
dependence on the state and their own direct dependence on the parameter.
`ss.sens_output_source` says what answered the functions: `"codegen"`,
`"automatic-differentiation"`, `"codegen+automatic-differentiation"`,
`"partial"` or `"none"`, and `""` when the model has no functions. A function
row with no answer is NaN, and `output_sensitivities` refuses it by name, with
the reason. A function is never differenced at a steady state.

### Initial conditions and conservation laws

A stable steady state forgets its initial conditions, except for conserved
totals. In `A <-> B`, a parameter that sets the initial amount of `A` moves
the steady state through the total `A + B`, and its column says so: `[1/3,
2/3]` for `A0` at `kf = 1`, `kr = 0.5`. A `$`-fixed species set by a
parameter, `$A() A0`, moves what reads it in the same way.

The seed is read from the state the solve starts at
(`model.effective_ic_sensitivity`). A species moved off its initial condition
by hand contributes nothing.

These requests raise `SensitivityUnsupportedError`:

- A parameter that sets the initial amount of a conserved species, when a
  `run()` has advanced the state (also when `save_concentrations()` saved that
  state). The total is still what the parameter made it, but the state no
  longer says so. Call `reset()` first.
- A compartment size, in a model with a conservation law.
- Any parameter, when a law spans compartments of different size and a
  compartment is sized by an assignment rule that can change: one that reads
  a species, the time, or a parameter you asked for. A rule over constants
  alone, such as `c2 := 2*p`, is fine.

A law across compartments of different size is a total of amounts, so it
carries the sizes: `A + 2*B` for `A` in a compartment of size 1 and `B` in
one of size 2. Otherwise its columns are computed like any other law's.

A species set by an assignment rule that the chain rule cannot answer has a
NaN row, and a compartment sized by such a rule refuses the solve.

### When the columns are refused

`-J⁻¹·∂f/∂p` is the derivative of the steady state only at an isolated root
the system rests at, and only when the returned state is on it. It is not
the derivative when the steady states form a continuum (an epidemic that
burns out ends at whatever susceptible level it leaves), when one species is
used up by two irreversible reactions to different products (`A -> B` and
`A -> C`), or when a total is conserved but not by a
linear law. BNGsim checks, and raises `SimulationError` if a check fails.

To check, it takes one Newton step from the returned state, and runs on for
`max_time` from that state moved by up to a millionth. Each column entry is
measured against its species' own size, `ss.sens_species_scale`, so units
and compartment sizes do not change the verdict. The checks:

| On the result | What it measures | Refused when |
| --- | --- | --- |
| `sens_root_determinant_ratio` | The Jacobian's determinant after the step over the one before. 1 at an isolated root, near 0 at a singular one, 1/2 at a double root. | outside 0.6 to 1.67 |
| `sens_root_pivot_share` | The smallest pivot of the factorization, relative to the terms it came from. | below 1e-13 |
| `sens_root_condition` | How much a relative error in the Jacobian is magnified in the columns. | above 1e13 |
| `sens_root_column_shift` | How far a column moves in one Newton step, as a fraction of its largest entry (and of no less than `1e-3/\|p\|`). If a column or the state (`sens_root_state_shift`) moves by more than 1%, the state is stepped on, up to ten times (`sens_root_newton_steps`). | the column or the state still moves more than 0.01 after the steps |
| `sens_root_hold_shift` | How far the columns move after a run of `max_time` from a millionth beside the state. `sens_root_hold_time` is the time the run reached in `max_steps` steps. | above 0.01, NaN, or the run falls short of `max_time` |
| `sens_root_growth_rate` | The largest real part of the eigenvalues, up to 512 unknowns; above that, of the eigenvalues among the species at 0. `sens_root_spectral_radius` is the largest in size. | times `max_time`, above 0.01 |
| `sens_root_relaxation` | How much of a column a run of `max_time` would leave unsettled. | above 0.01 |

If one column fails, the whole request is refused, so which parameters you ask
for together can matter.

What each failure usually means:

- **Determinant ratio, pivot share or condition number:** the steady state is
  not an isolated root, or the model's rates are too far apart for the columns
  to be reliable: more than about 1e12 apart with a closed-form Jacobian, or
  1e5 with a difference quotient, where the pivot and condition limits are
  1e-6 and 1e6. A ratio below 0
  or above 1.67 can also mean the state is far from its root (a smaller `tol`
  helps) or a rate law jumps between the two states.
- **Column shift:** the state is too far from the root. `tol` bounds the
  residual, not the distance to the root. When a step moves a column by more
  than 1%, the solve steps on and returns the stepped state, so
  `ss.concentrations`, `ss.residual` and `ss.sensitivity` are then those of
  the stepped state. A small species is solved to its own steady value this
  way. A stepped root must agree, within 1%, with where the hold run ends. A
  column that is essentially 0 (under `1e-5/|p|`) is returned as 0.
- **Hold shift or growth rate:** a run leaves the state. It may be the middle
  root of a switch, the fixed point inside a limit cycle, or a state that a
  species at 0 would grow from (an invader the model does not start
  with). Some of these columns would be right, but telling them apart is not
  possible here. If the run uses up its `max_steps` before `max_time`, give it
  more steps.
- **Relaxation:** the state is a steady state the model does not reach in
  `max_time`, such as a species whose turnover is switched off. Raise
  `max_time`, or take the column from a time course.

For such a model, take the columns from a time course with forward
sensitivities, run to the steady state:

```python
sim = bngsim.Simulator(model, sensitivity_params=["kf", "A0"])
result = sim.run(t_span=(0, 1e4), n_points=2, start="initial")
result.sensitivities[-1]      # (n_species, n_params) at the last time
```

`ss.sens_jacobian_rcond`, `min|U|/max|U|` of the LU, is reported but decides
nothing. It depends on the units.

### Sensitivities with a mask

With `mask=`, the columns are solved on the equations of the kept species and
the conservation laws among them alone. Where a masked-out sink drains a law
(`A <-> B -> P`: `A` and `B` end at 0 whatever the total), the columns are
computed. Where part of the total stays out of the sink, the root is not
isolated and the columns are refused.

A masked-out species is held where the solve left it. That is right only if
no kept species' equation reads it, as none reads a pure sink. A mask that
leaves out a species a kept one reads is refused. `ss.sens_mask_held_species`
and `ss.sens_mask_reader_species` name them.

## Pre-equilibration: `carry_sensitivities=True`

A pre-equilibration protocol runs the model to steady state, changes a
parameter, then measures. The two runs use the same Simulator with no reset
between them, so the measurement starts from `x_ss(θ)`. Its sensitivity seed
is then `dx_ss/dθ`, not 0. Pass `carry_sensitivities=True` on the
measurement run to use it:

```python
# A model with the parameters k_prod, k_deg and Ligand_isPresent, and an observable R_active.
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k_prod", "k_deg"])

# Phase 1: equilibrate, with the same sensitivity_params, so dx_ss/dθ is kept.
sim.run(t_span=(0, 1e6), n_points=2, steady_state=True)

# Change a parameter. The species state carries over.
model.set_param("Ligand_isPresent", 1)

# Phase 2: measure, seeded from phase 1's dx_ss/dθ.
r = sim.run(t_span=(0, 60), n_points=61, carry_sensitivities=True)
grad = r.output_sensitivities("observable:R_active")
```

In a model with an event or one that reads the time, phase 1 does not stop
early. Give it the span the pre-condition is meant to last.

These raise rather than give a wrong derivative:

- A sensitivity run on a carried-over state without
  `carry_sensitivities=True`.
- `carry_sensitivities=True` with no matching seed: phase 1 used other
  `sensitivity_params`, a run without sensitivities advanced the state, an
  SSA or PSA run advanced it, or a `reset()` cleared it.

A run marks the model's state as carried over, and that mark stays until a
`reset()` to a baseline no run advanced. `set_state()` and
`set_concentration()` do not clear it. To treat a carried state as fixed data
instead, with `∂x(0)/∂θ = 0`, call `reset()` first, then `set_state()` the
state. If `save_concentrations()` made
an advanced state the baseline, `reset()` returns to that state, so set the
values on a fresh `Model` instead. After an SSA or PSA run, set
`result.species[-1]`, not the model's state, which in a compartment whose
size changes holds counts over the load-time size. The `reset()` also sets the
clock to 0 and drops pending events.

After a run of any method has advanced the state, two more things hold:
`set_param()` of a parameter that names an initial condition does not re-dose
that species, even if `set_state()` put it back on its declared value
(`reset()` first, or use `set_concentration()`), and
`steady_state(sensitivity_params=...)` of such a parameter, when it sets the
total of a conservation law, is refused.

Limits: the equilibration is a steady state and the change is an absolute
parameter value. Finite-time equilibration is not covered. `sensitivity_ic`
across the boundary raises, unless `save_concentrations()` made the carried
state the initial condition and no run has moved it since. A model with
events warns.

### Into a parameter scan

A dose-response experiment equilibrates once, then scans, and every point
starts from the same equilibrated state with the same `dx_ss/dθ`. The
snapshot calls carry the derivative with the state:

- `save_concentrations()` (unlabeled) makes the current state the baseline,
  with its `dx/dθ`, and `reset()` restores both. A baseline saved from a state
  no run advanced is literal: no parameter seeds it. A baseline saved from a
  state a run advanced without tracking `dx/dθ` keeps the carried-over mark,
  and a sensitivity run from it raises.
- `save_concentrations(label=...)` and `restore_concentrations(label)` do the
  same for a named snapshot.
- `parameter_scan` and `bifurcate` restore the reset target's state and
  `dx/dθ` at each point, run it with `carry_sensitivities=True`, and leave the
  model as they found it.

```python
# A model with the parameters kf, kr, kdeg and L_0, and an observable pReceptor.
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr", "kdeg"])
sim.run(t_span=(0, 1e6), n_points=2, steady_state=True)   # equilibrate once

# The hook does the dosing: in a sensitivity scan, changing L_0 moves no
# species, so without the hook every point would start the ligand where the
# equilibration left it, and the scan would warn that the dose is dropped.
points = sim.parameter_scan(
    "L_0", par_min=1e-3, par_max=1e2, n_scan_pts=12, log_scale=True,
    t_span=(0, 600), n_points=61, steady_state=True,
    on_point=lambda m, dose: m.set_concentration("L(r)", dose),
)
grads = [p.output_sensitivities("observable:pReceptor") for p in points]
```

A continuation scan (`bifurcate` with `reset_conc=False`) carries each point's
state and `dx/dθ` on to the next.

A sensitivity scan raises when:

| Situation | Why |
| --- | --- |
| the reset target carries no matching `dx/dθ` | nothing to seed from: phase 1 was not run on this Simulator with these `sensitivity_params`, or a plain run, `set_concentration()` or `set_state()` dropped it |
| the scanned parameter is in `sensitivity_params` | each point overwrites it |
| `sensitivity_ic` is asked for | the point starts from a snapshot, not the initial conditions |
| an `on_point` hook changes a differentiated parameter | the same problem as scanning one |

For a sweep where every point starts from the model's own initial conditions,
use `run_batch`. It clones and resets each row.

### The dose an on_point hook applies

An `on_point` hook sets the initial condition of the species it writes, so
their seed comes from the hook's arithmetic, not from the carried
derivative. Each row of a point's seed comes from the first of these that
applies:

1. a seed the hook installed itself, with
   `model._core.set_pending_sensitivity_seed(...)` after its writes;
2. a row declared with `model.declare_ic_sensitivity({species: {param: value}})`;
3. a row measured through the hook: BNGsim calls the hook at perturbed inputs
   and differences what it assigns;
4. otherwise, the carried row, exactly.

So a literal dose needs nothing; it measures 0. A dose computed from a fitted
parameter, or an increment of the carried pool (`x_k + dose`), gets its true
derivative.

```python
# NA is Avogadro's number; Vecf is a fitted volume, and L(r) a species of the model.
def on_point(model, dose_nM):
    v = dose_nM * 1e-9 * NA * model.get_param("Vecf")   # Vecf is fitted
    model.set_concentration("L(r)", v)
    # Optional: declaring the row skips measuring it.
    model.declare_ic_sensitivity({"L(r)": {"Vecf": v / model.get_param("Vecf")}})
```

Measuring calls the hook several extra times per point, so the hook must be a
deterministic function of `(model, value)`. BNGsim checks this by running it
again. It also measures at two step sizes and raises rather than difference a
jump: a dose rounded to whole molecules is not differentiable, so declare its
row.

`declare_ic_sensitivity` works on a plain `run()` too. If you set an initial
condition by hand from a parameter, declare its derivative this way: outside
a hook there is nothing to measure.
