# Steady-state solver

## Steady-state solver

BNGsim includes a steady-state solver for finding f(y) = 0 — the equilibrium
where all species concentrations stop changing. This is essential for
dose-response curves, bifurcation analysis, and fitting steady-state data.

All paths share **one** convergence criterion, matching BNG2.pl's
`run_network -c`: the parity residual `||f(y)||_2 / n_species < tol`. This is
the same quantity `run(steady_state=True)` checks (see below).

**`method="integration"` (default)**: CVODE BDF integration that marches
forward one step at a time and stops when the parity residual
`||f(y)||_2 / n_species` drops below `tol` (capped at `max_time`). This is the
strict BNG2.pl-parity path, and it always returns the steady state the
dynamics actually reach.

**`method="newton"`**: the two-tier integrate-first solver. Tier 1 is the
*same* CVODE burst as `"integration"`, carrying the state into the physical
root's basin; tier 2 is a KINSOL Newton polish using the analytical Jacobian
when available (all-Elementary models) or KINSOL's internal finite
differences. For models with conservation laws, BNGsim automatically uses a
reduced-space Newton formulation (see [Conservation laws](#conservation-laws)).
The polish is accepted only once it is *seed-stable* — two Newton solves from
successively tighter bursts landing on the same state — otherwise integration
simply continues, so a `method="newton"` call always honors the parity
criterion.

**`method="kinsol"`**: accepted alias for `"newton"` (the canonical name is
always echoed in `ss.method_used`).

> **Which one?** Since tier 1 *is* the integration path, `"newton"` can only
> add work on top of `"integration"` — and on six published dose-response
> models it added 1.4–3.9× (geometric mean 2.5×) of it (issue #28). Use the
> default unless one of these applies:
>
> - **You want the root resolved far below `tol`.** Newton lands at a residual
>   around `1e-13` where integration stops the moment it crosses `tol`
>   (~`1e-9`). That headroom matters mainly when the steady state feeds a stiff
>   downstream solve.
> - **You have cut `max_time` well below its default.** Newton reaches `tol`
>   from a looser burst than integration needs on its own, so under a tight
>   time budget it can converge where integration runs out of horizon. At the
>   default `max_time=1e6` no model in the benchmark corpus shows this; at
>   `max_time=1e3` several do.

> The old `method="auto"` and the `max|f|` / geometric-time-horizon Tier-1
> criterion were removed: `"newton"` already means integrate-then-polish-with-
> fallback, and every integration path now uses the single `||f||_2/n` rule.

Every path evaluates whatever RHS the Simulator's
[`codegen_backend`](codegen.md) reports — the compiled `.so`, the in-process MIR
JIT, or the ExprTk interpreter — and `ss.rhs_backend` echoes which one ran.
(Before issue #63 the steady-state solver read no codegen option at all, so a
Simulator built with `codegen=True` still solved interpreted.)

```python
import bngsim

model = bngsim.Model.from_net("model.net")
sim = bngsim.Simulator(model, method="ode")

# Basic steady-state (default method="integration", BNG2.pl parity criterion)
ss = sim.steady_state()
print(ss.converged)          # True
print(ss.method_used)        # "integration"
print(ss.residual)           # ||f(y)||_2 / n_species at convergence, e.g. 8.6e-10

# Access species by name (dict-like)
print(ss["A(b)"])            # steady-state concentration of A(b)
print(ss.concentrations)     # full array, shape (n_species,)
print(ss.to_dict())          # {"A(b)": 50.0, "B(a)": 25.0, ...}

# Force a specific method
ss = sim.steady_state(method="integration")  # CVODE parity early-stop (default)
ss = sim.steady_state(method="newton")       # burst, then Newton polish
ss = sim.steady_state(method="kinsol")       # alias for "newton"

# Custom tolerances
ss = sim.steady_state(
    tol=1e-12,         # convergence tolerance on ||f||_2/n
    max_time=1e8,      # max integration time (integration path)
    rtol=1e-10,        # CVODE relative tolerance
    atol=1e-10,        # CVODE absolute tolerance
    max_steps=50000,   # max CVODE internal steps
)
```

### Write-only accumulator species (`mask=`, issue #74)

Counting cumulative flux with a "degraded" / "produced" / "secreted" pool is a
common BNGL idiom: some reaction produces the species and none consumes it. Such
a **pure sink** has a constant non-zero derivative for as long as its producing
reactions fire, so `||f(y)||_2 / n_species` has a floor above `tol` and
`steady_state()` reports failure however long it integrates — even when every
other species has settled. On `beta_catenin_destruction_complex_barua2013`
(409 species, four pure sinks) the residual does not move across two decades of
`max_time`, while the state grows linearly:

| `max_time` | converged | residual   | `max\|y\|` |
| ---------- | --------- | ---------- | -------- |
| 2.5e6      | False     | 7.4990e-03 | 7.49e+06 |
| 2.5e7      | False     | 7.4990e-03 | 7.49e+07 |
| 2.5e8      | False     | 7.4990e-03 | 7.49e+08 |

That is a constant derivative, not a slow tail. `mask=` restricts the
convergence test to the subspace that *does* have a steady state, and
`Model.is_pure_sink()` finds the accumulators structurally, so nothing has to be
hand-listed:

```python
model.pure_sink_species()
# ['bCat(ARM34,ARM59,s33s37~U,s45~U,ss~d)', ... ]   # 4 of 409

ss = sim.steady_state(method="newton", mask=~model.is_pure_sink())
ss.converged                # True
ss.residual                 # 9.10e-10
ss.n_residual_species       # 405 — how many species entered the norm
ss.excluded_species         # [11, 151, 289, 359]
```

`mask` also takes the species names to keep, if you would rather be explicit:

```python
ss = sim.steady_state(mask=[n for n in model.species_names if not n.endswith("ss~d)")])
```

Integer indices are rejected: `[0, 1]` is ambiguous between a two-species 0/1
mask and "keep species 0 and 1", and guessing would be a silent wrong answer on
exactly the long species lists where you cannot eyeball it.

**What the mask changes.** Everything still integrates — the excluded species'
equations stay in the RHS and their trajectories come back in
`ss.concentrations`. What is restricted is:

- the **residual norm**, over `n_included` rather than `n_species`, so `tol`
  keeps its meaning as a per-species residual scale no matter how many species
  you dropped;
- the **KINSOL unknown set** on `method="newton"`, and the **`dY_ss/dp` linear
  system**. Those two have to follow: an accumulator contributes a structurally
  zero Jacobian *column*, so leaving it in makes both systems singular at every
  seed. Excluded species are held at the values integration left them at, which
  is exact because nothing else's derivative reads them.

Excluded species come back with a **NaN** `dY_ss/dp` row. A species with no
steady value has no steady-state gradient, and `0.0` would be a confident wrong
answer a fitter would read as "this parameter does not matter". Any observable
or expression sensitivity that sums such a species is NaN for the same reason.

`steady_state_batch(mask=...)` applies one mask to every entry — the pure-sink
set is structural, so it does not move with the parameter set, which is what
makes a single mask correct for a whole dose scan.

**When a solve fails**, the result now says whether the cause was structural:

```python
ss = sim.steady_state()             # no mask
ss.converged                        # False
ss.unconverged_pure_sinks           # ['bCat(...ss~d)', ...] — 4 names
```

and the same is logged at WARNING level. An empty list means the failure was
*not* an accumulator, so `max_time` / `tol` / `max_steps` are worth trying.

`pure_sink_species()` is purely structural — a species qualifies when it is a
product of at least one reaction, a reactant of none, read by no other species'
derivative, and not a `$`-fixed boundary condition. The third clause is not
implied by the first two (an Elementary rate law reads only its reactants, but a
Functional one reads observables) and is what makes excluding the species
provably harmless to the rest of the system.

Detection is not a convergence verdict: `A -> B` with nothing feeding `A` makes
`B` a textbook pure sink, and that model converges perfectly well because the
flux dies out on its own. `pure_sink_species()` answers "can this be dropped
from the test without changing the problem"; `unconverged_pure_sinks` answers
"is this what held the solve up".

`run(steady_state=True)` — the time-course early stop — keeps the unrestricted
BNG2.pl criterion and takes no mask. On an accumulator model it simply never
fires early and you get the full `t_span`, which is a complete and correct
trajectory rather than a reported failure.

### Time course that stops at steady state (`run(steady_state=True)`)

`steady_state()` above returns just the equilibrium point. If instead you
want the **trajectory up to** equilibrium — and want it to stop as soon as
the network equilibrates rather than integrating the full `t_span` — pass
`steady_state=True` to `run()`. This mirrors BNG2.pl's
`simulate({steady_state=>1})` (`run_network -c`): after recording each
output point the integrator checks `||f(t,y)||_2 / n_species` and stops once
it drops below the tolerance, returning a `Result` truncated to only the
rows it integrated.

```python
# Stop early once the network equilibrates (ODE only)
r = sim.run(t_span=(0, 1000), n_points=101, steady_state=True)
print(len(r.time))                                 # < 101 if it equilibrated early
print(r.solver_stats["steady_state_reached"])      # 1 if the criterion fired, else 0

# steady_state_tol defaults to atol (matching BNG2.pl); override explicitly:
r = sim.run(t_span=(0, 1000), n_points=101, steady_state=True, steady_state_tol=1e-9)
```

### Dose-response sweeps (parallel)

`steady_state_batch()` computes steady states across multiple parameter sets
in parallel — ideal for dose-response curves:

```python
import numpy as np

# Sweep ligand concentration over 4 orders of magnitude
doses = np.logspace(-2, 2, 50)
param_sets = [{"L_0": d} for d in doses]

# Parallel steady-state sweep (8 threads)
results = sim.steady_state_batch(
    params=param_sets,
    n_workers=8,
    tol=1e-10,
)

# Extract dose-response curve
response = np.array([r["R_bound"] for r in results])

import matplotlib.pyplot as plt
plt.semilogx(doses, response)
plt.xlabel("Ligand concentration")
plt.ylabel("Bound receptor at steady state")
```

Each batch entry clones the model (thread-safe deep copy), applies the
parameter set, and runs an independent steady-state solve. The GIL is
released during C++ KINSOL/CVODE integration, so threads achieve real
parallelism.

### Steady-state sensitivity

BNGsim computes the steady-state sensitivity matrix `dY_ss/dp` via the
implicit function theorem: `dY_ss/dp = -J⁻¹ · (∂f/∂p)`.

Both factors are taken in closed form where the model supports it:

- **J** — the analytical Jacobian at the steady state, compiled when the codegen
  artifact carries one and interpreted otherwise. This is the same
  "analytical when complete, finite differences otherwise" rule
  `jacobian="auto"` applies everywhere else, and the same matrix the
  `method="newton"` polish uses; `jacobian="fd"` pins the difference quotient.
- **∂f/∂p** — the analytical parameter derivative the code-generated
  sensitivity RHS emits, the same one CVODES integrates against on the
  time-course path. Since issue #67 this covers Functional rate laws too, as long
  as they are smooth algebra. What has no such derivative to emit is
  Michaelis–Menten, and any Functional law carrying a condition (`if()`, a
  comparison, a logical — issue #68) or a non-smooth builtin
  (`abs`/`min`/`max`/`floor`/`ceil`/`round`); for those the factor is still
  finite-differenced — with a warning, and `ss.sens_dfdp_source` says so.

Because the analytical `∂f/∂p` comes from codegen, `sensitivity_params` **requires
code generation**, exactly as `Simulator(..., sensitivity_params=...)` and
`compute_all_sensitivities()` do since GH #214: a request that cannot get one is
refused rather than answered from `sqrt(eps)`-noisy difference quotients. The
analytical RHS is built automatically via `cc` or the in-process MIR JIT, so this
does not require a system compiler — but `codegen=False` and `BNGSIM_NO_CODEGEN`
now raise here.

`ss.rhs_backend`, `ss.sens_jacobian_source` and `ss.sens_dfdp_source` report which
path each piece actually took.

```python
ss = sim.steady_state(
    sensitivity_params=["kf", "kr", "kcat"],
)
print(ss.rhs_backend)             # "codegen-so" | "codegen-jit" | "exprtk"
print(ss.sens_jacobian_source)    # "codegen" | "analytical" | "finite-difference"
print(ss.sens_dfdp_source)        # "codegen" | "finite-difference"
print(ss.sens_output_source)      # "codegen" | "mixed" | "finite-difference"

# Sensitivity matrix: (n_species, n_params)
print(ss.sensitivity.shape)       # (50, 3)
print(ss.sensitivity_params)      # ["kf", "kr", "kcat"]

# How does species "P" change with respect to kf?
p_idx = ss.species_names.index("P")
kf_idx = ss.sensitivity_params.index("kf")
print(ss.sensitivity[p_idx, kf_idx])
```

For models with conservation laws where the full Jacobian is singular,
BNGsim automatically builds a reduced Jacobian on the independent species
subspace, solves the non-singular reduced system, and reconstructs the
dependent species sensitivities from the conservation constraints.

#### Observable / expression output sensitivities

`ss.sensitivity` is species-level. To read `∂(observable)/∂θ` or
`∂(expression)/∂θ` directly — without re-deriving the output Jacobian yourself —
use `output_sensitivities`, exactly as on a CVODE
[`Result`](sensitivities.md):

```python
ss = sim.steady_state(sensitivity_params=["kf", "kr", "kcat"])

# (n_selectors, n_params), one row per selector — no time axis at steady state.
grad = ss.output_sensitivities(["observable:P_tot", "expression:activity"])

ss.observable_names            # rows of ss.sensitivities_observables
ss.expression_names            # rows of ss.sensitivities_expressions
ss.sensitivities_observables   # (n_observables, n_params) bulk array
```

BNGsim projects `dY_ss/dp` internally: observables use the exact linear group
map, and global functions use the full total derivative — the state-chain term
`(∂func/∂x)·dY_ss/dp` **plus** the function's explicit parameter dependence
`∂func/∂p` (e.g. a rate-law function `k3/(K4+G)` differentiated w.r.t. `k3`). A
downstream gradient consumer can reuse its existing CVODE
`output_sensitivities` code path unchanged.

Since issue #75 that total derivative is not merely *matching* the CVODES codegen
chain rule — it **is** that chain rule: the compiled `bngsim_codegen_output_sens`
evaluator, fed the solved `dY_ss/dp` columns, so a steady-state gradient and a
converged long-run gradient come from one derivation. `ss.sens_output_source`
reports which path the expression block took:

- `"codegen"` — every function came from the compiled chain rule.
- `"mixed"` — some did; the rest were finite-differenced. That is a function the
  codegen declines to differentiate (a table function, or a non-smooth builtin
  such as `abs`/`min`/`max`/`floor` — the same constructs listed for `∂f/∂p`
  above), or an auto-generated `_rateLawN` intermediate outside the
  user-selectable set.
- `"finite-difference"` — none did, because the model has no compiled
  output-sensitivity evaluator at all (a `rateOf` model, an embedded
  table-function wrapper, or a model with no user-selectable global functions).

The observable block is the exact linear projection either way, and is not
covered by this field. Its weights are the group factors times the amount-valued
volume factor for an SBML `hasOnlySubstanceUnits="true"` species — whose
observable denotes an *amount*, not the stored concentration — matching what
`update_observables` uses for the value and what the CVODE `run()` path uses for
its derivative (issue #119).

A stable steady state forgets its initial conditions (`∂x*/∂x(0) = 0`), so the
initial-condition axis is structurally zero and is not computed;
`output_sensitivities(..., axis="ic")` raises rather than return zeros.

### Pre-equilibration / carry-over output sensitivities (`carry_sensitivities=True`)

A **pre-equilibration** protocol equilibrates the system to steady state under
a pre-condition (unmeasured), then perturbs a parameter and measures — running
the **same persistent `Simulator` across two `run()` calls with no reset
between them**, so the equilibration steady state `x_ss(θ)` *is* the
measurement phase's initial condition (the receptor dimerizes before ligand is
added — the equilibration is not a no-op). Because the measurement phase starts
from `x_ss(θ)`, its forward-sensitivity seed is `∂x(0)/∂θ = dx_ss/dθ` — the
steady-state sensitivity of phase 1 — **not** the fresh-start zero. Pass
**`carry_sensitivities=True`** on the measurement run to seed it correctly:

```python
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k_prod", "k_deg"])

# Phase 1 — equilibrate under the pre-condition, unmeasured. Run with the
# sensitivity_params so the engine captures dx_ss/dθ at the steady state.
sim.run(t_span=(0, 1e6), n_points=2, steady_state=True)

# Apply the measurement-phase perturbation (an absolute setParameter — the
# species state carries over; no reset).
model.set_param("Ligand_isPresent", 1)

# Phase 2 — measure. carry_sensitivities=True seeds yS(0) from phase 1's
# dx_ss/dθ, so output_sensitivities() is correct across the boundary.
r = sim.run(t_span=(0, 60), n_points=61, carry_sensitivities=True)
grad = r.output_sensitivities("observable:R_active")   # correct across the boundary
```

**No silent wrong derivatives.** Requesting sensitivities on a carried-over
state *without* `carry_sensitivities=True` **raises** (fresh seeding would
silently assume `∂x(0)/∂θ = 0`). So does `carry_sensitivities=True` when no
matching seed is available — e.g. the equilibration phase was not run with the
same `sensitivity_params`, a plain (non-sensitivity) run advanced the state
without tracking `dx/dθ`, or a `reset()` (as an SBML/RoadRunner every-action
reset would do) returned to a θ-independent IC baseline and so wiped the
carry-over. A fresh single sensitivity run is unaffected.

Scope (matching the new-era pre-equilibration surface): the equilibration is a
**steady state** (PEtab `time = -inf`) and the perturbation is an **absolute**
(`=`) `setParameter` — the species state carries over, only a parameter
changes. Finite-time equilibration and initial-condition–axis (`sensitivity_ic`)
sensitivities across the boundary are out of scope (the latter raises); a model
with **events** warns, since event-time sensitivity discontinuities are handled
separately. The carried seed is model-level state alongside the concentrations,
introspectable via `model._core.ic_state_dirty` /
`model._core.has_pending_sensitivity_seed`.

### …and into a parameter scan (dose-response, issue #81)

A dose-response experiment pre-equilibrates **once** and then scans, and every
scan point starts from that same equilibrated state — so each point's seed is
the *same* `dx_ss/dθ`. Snapshot / restore primitives therefore carry the
derivative with the state:

* `save_concentrations()` (unlabeled) redefines the IC baseline to the current
  state, so the new baseline **inherits** its `dx/dθ` — the state did not change,
  so neither did its derivative — and `reset()` restores both. A baseline saved
  with no carried derivative is θ-independent literal ICs, i.e. fresh-start
  seeding as before.
* `save_concentrations(label=...)` / `restore_concentrations(label)` capture and
  restore a named snapshot's `dx/dθ` the same way.
* `Simulator.parameter_scan` / `bifurcate` restore the reset target's state
  **and** its `dx/dθ` per point and integrate each point with
  `carry_sensitivities=True`, then leave the model (state, parameter, carried
  derivative) exactly as they found it.

```python
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr", "kdeg"])

# Pre-equilibrate once, with the sensitivity_params, so dx_ss/dθ is captured.
sim.run(t_span=(0, 1e6), n_points=2, steady_state=True)

# Scan the dose. Each point resets to the equilibrated state *and* its dx_ss/dθ.
points = sim.parameter_scan(
    "L_0", par_min=1e-3, par_max=1e2, n_scan_pts=12, log_scale=True,
    t_span=(0, 600), n_points=61, steady_state=True,
)
grads = [p.output_sensitivities("observable:pReceptor") for p in points]
```

A continuation scan (`bifurcate`, `reset_conc=False`) instead carries each
point's state *and* `dx/dθ` from the previous point, making the whole sweep one
differentiable protocol.

**Still no silent wrong derivatives.** A sensitivity scan raises — rather than
re-seeding a point fresh — when:

| Situation | Why it cannot be answered |
| --- | --- |
| the reset target carries no matching `dx/dθ` | nothing to seed from; the equilibration was not run on this `Simulator` with these `sensitivity_params`, or a plain run / `set_concentration()` / `set_state()` dropped it |
| the scanned parameter is a `sensitivity_params` entry | each point overwrites it, so the derivative carried *into* the point was taken at a different value of the same symbol |
| `sensitivity_ic` is requested | the point starts from a snapshot, not the model's ICs, so `∂y/∂y_k(0)` has no meaning across the boundary |
| an `on_point` hook moves a differentiated parameter | same composition problem as scanning one |

#### The dose an `on_point` hook applies (issue #111)

An `on_point` hook *assigns* the initial condition its point starts from, so for
the species it writes, `∂x_k(0)/∂θ` is whatever the hook's own arithmetic
implies — not the carried equilibration derivative. Each row of the point's seed
is resolved by the most specific thing available:

1. a row the hook installed wholesale
   (`model._core.set_pending_sensitivity_seed(...)` after its writes);
2. a row **declared** with `model.declare_ic_sensitivity({species: {param: value}})`;
3. a row **measured through the hook** — bngsim calls the hook at perturbed
   inputs and differences the initial condition it assigns;
4. otherwise the carried row, bit-exact.

So the ordinary literal dose needs nothing: it measures `0`. A dose computed
*from* a fitted parameter — nM converted to molecules through a fitted volume —
measures its true derivative, and so does an *increment* of the carried pool
(`x_k + dose`), which comes back as the carried row plus the dose's derivative.

```python
def on_point(model, dose_nM):
    v = dose_nM * 1e-9 * NA * model.get_param("Vecf")   # Vecf is fitted
    model.set_concentration("L(r)", v)
    # Optional: declaring the row skips its measurement (exact, and it is the way
    # out for an expensive hook or one with a non-differentiable dose).
    model.declare_ic_sensitivity({"L(r)": {"Vecf": v / model.get_param("Vecf")}})
```

Measuring invokes the hook several extra times per point on the live model with
perturbed inputs, so the hook must be a **deterministic** function of
`(model, value)`; bngsim verifies that by re-running it and comparing. It also
checks the measurement at two step sizes and **raises** rather than reporting a
difference quotient of a jump — a dose rounded to whole molecules is not
differentiable, and such a row must be declared.

`declare_ic_sensitivity` is honoured on a plain `run()` too, which is the way to
give a **hand-assigned** θ-dependent initial condition its derivative: outside a
hook there is nothing to probe, and the parameter-graph seeding differentiates
the `.net` IC *expression*, which a `set_concentration` has replaced.

For a sweep whose points start from the model's own seed initial conditions (no
pre-equilibration), use `run_batch` — it clones and resets each row, so
fresh-start seeding is the correct one.
