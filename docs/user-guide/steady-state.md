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
strict BNG2.pl-parity path, and for a model whose right-hand side does not read
the time it returns the steady state the dynamics reach.

**A model with an event, or one that reads the time, is refused** by
`steady_state()` and `steady_state_batch()` (issue #710). Both solvers look for
a root of `f(y)` with everything read at `t = 0`, and fire no event: `A -> B` at
`k*(1 - exp(-time()))`, whose rate is 0 at `t = 0`, came back at its initial
state as converged, and a model whose event switches a production rate on came
back at the state from before it. A reported quantity that reads the time has
no steady value either: an assignment-rule species `S := B*(1 - exp(-time))`
came back 0 for 2/3. A `time()` call or a table function indexed by time, in a
rate law, a rule or a function that is only reported, is a read of the time;
`rateOf` is not.

Integrate such a model with `run()` over a span long enough for the trajectory
to settle. `run(..., steady_state=True)` does not stop early for it: the
criterion is `‖f(t, y)‖` at an output point, and a rate that is switched on at
`t = 5` is 0 at `t = 1`. The run goes to the end of its span, and
`steady_state_reached` stays 0.

**`method="newton"`**: the two-tier integrate-first solver. Tier 1 is the
*same* CVODE burst as `"integration"`, carrying the state into the physical
root's basin; tier 2 is a KINSOL Newton polish. For models with conservation
laws, BNGsim automatically uses a reduced-space Newton formulation (see
[Conservation laws](#conservation-laws)).
The polish is accepted only once it is *seed-stable* — two Newton solves from
successively tighter bursts landing on the same state — **and** only once the
root is *dynamically stable* (see [Unstable roots](#unstable-roots-saddles)),
otherwise integration simply continues, so a `method="newton"` call always
honors the parity criterion.

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

### The Jacobian the solver uses

Both tiers factor the model's **closed-form Jacobian** when it has one and
`jacobian=` asks for it (issue #127): the march installs it with
`CVodeSetJacFn`, the polish with `KINSetJacFn` — projected onto the polish's own
unknown set, since the reduced Newton solves for the conservation-law
independents rather than for every species. `ss.solver_jacobian_source` reports
which matrix was factored: `"codegen"` (the compiled `bngsim_codegen_jac`),
`"analytical"` (the interpreted fill), or `"finite-difference"`.

`jacobian="fd"` pins the difference quotient, as it does everywhere else in the
library, and a model with no complete closed form differences whatever the
option says. That difference quotient costs **one RHS evaluation per unknown per
Jacobian setup** — which is what both tiers paid unconditionally before #127,
including on codegen-backed solves that had the compiled Jacobian loaded in the
same object.

This is a cost, not a correctness knob: the accepted root has to satisfy the
same `||f||_2/n < tol` either way, so what the Jacobian changes is the step
sequence taken to reach it. The same option still selects how the *stability
certificate* below and `dY_ss/dp` build their matrix.

**What it is worth.** On the 585-model `.net` corpus, 560 models have a closed
form to install; over them a solve makes a median 1.09× fewer RHS evaluations
(`method="newton"`: 1.10×, up to 33×). Wall clock follows the size of the
Jacobian setup relative to the rest of the solve:

| model (species) | `jacobian="fd"` | `"auto"` | interpreted | codegen |
|---|---|---|---|---|
| `SHP2_base_model` (149) | 9.9 ms | 8.9 ms | **1.11×** | 0.89× |
| `Barua_2007` (149) | 11.1 ms | 10.3 ms | **1.08×** | 0.87× |
| `egfr_ground` (356) | 337 ms | 261 ms | **1.29×** | 1.06× |
| `fceri_fyn` (1281) | — | — | 1.04× | **1.19×** |

(`method="integration"`; the interpreted column is the timing shown, the codegen
column the same A/B with `codegen=True`.)

Where the RHS is interpreted the saving is real everywhere. Where it is
*compiled*, a difference-quotient column is cheap and the dense factorization is
not: a profile of the 149-species case puts ~80% of CVODE's time in
`SUNDlsMat_denseGETRF` and none of it in the Jacobian fill, so removing the
columns does not pay for itself below a few hundred species and the solve can
come out slightly slower. That dense factorization is what issue #128 addressed,
below.

### The linear solver the march factors with

The march routes its Newton matrix to **sparse KLU or a dense LU by the same
rule `run()` uses** (issue #128): KLU when the model has at least
`SPARSE_THRESHOLD` = 50 species, a Jacobian sparsity density under
`SPARSE_DENSITY_MAX` = 10%, and a structural nonzero to factor — unless the
LU-fill test moves it to the BLAS dense factor, as it does for `run()` (see
[Linear solver](solvers.md#linear-solver)). `ss.linear_solver` reports which one
ran — `"klu"`, `"dense"` or `"lapack-dense"`.

`force_sparse_linear_solver` and `force_dense_linear_solver` on the Simulator
override the size and density gates in either direction and now reach
`steady_state()` and `steady_state_batch()` as well as `run()`. Before #128 the
steady-state paths had no sparse option at all, so a Simulator built with
`force_sparse_linear_solver=True` still got a dense factorization out of
`steady_state()` without saying so.

This changes the cost, not the answer. Measured before and after over the
585-model `.net` corpus under both methods: the models the routing leaves dense
are byte-identical, `converged` flips on none of the routed models, and over
every converged model the state moves by at most 2.0e-8 of the model's own
scale (median exactly 0). What KLU changes is the factorization and, with it,
the step sequence taken to reach the same root — the `n_steps` ratio on routed
models is a median of 1.000 (range 0.93–1.07):

| model (species, density) | dense | KLU |
|---|---|---|
| `egfr_ground` (356, 6.7%) | 0.19 s | **0.18 s** |
| `BaruaBCR_2012` (1122, 2.5%) | 5.59 s | **1.78 s** (3.1×) |
| `fceri_fyn` (1281, 2.3%) | 13.25 s | **6.92 s** (1.9×) |

(`method="integration"`, interpreted RHS, `tol=1e-9`. The cutoff earns its
place: below a few hundred species KLU's setup and indexing cost roughly cancel
its factorization win, which is why the rule does not simply always choose it.)

`jacobian="fd"` stays on the sparse route rather than reverting to dense. On a
KLU-routed march that is a requirement and not a preference — CVODE's built-in
difference quotient supports dense and banded matrices only — so the march fills
the CSC values with a Curtis-Powell-Reid *colored* difference quotient, at one
RHS evaluation per color (typically 5–20) instead of one per species. Only
`jacobian="jax"` forces the dense route, since a JAX Jacobian only ever fills a
dense matrix.

The **KINSOL polish and the `dY_ss/dp` solve are deliberately not routed**. Both
factor the *reduced* system, whose matrix is the model's Jacobian projected
through the conservation-law reconstruction — the projection fills in entries the
model's sparsity pattern does not have, so the reduced pattern is a different
object that would have to be derived. Both also factor once per solve rather than
once per integration step, which is where it matters least.

**When the closed form is called off.** An exact Jacobian omits the jump in a
rate law that is genuinely discontinuous in a state variable, so CVODE's
corrector meets a step it was not warned about, the local error test fails
repeatedly, and the march collapses to `hmin`. Under `jacobian="auto"` a march
that fails that way is retried once on difference quotients, exactly as
`run()` does (GH #176) and for the same models — one of the 585.
`ss.solver_jacobian_retried` records it, and an explicit
`jacobian="analytical"` surfaces the failure instead of retrying.

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

# atol also takes one value per species, or "auto" (issue #196) — the accuracy
# the march is held to on the way to the root. `tol` above is the separate test
# for having arrived, and stays a single norm over all species.
ss = sim.steady_state(atol="auto")
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

### SBML species set by an `<assignmentRule>` (issue #247)

A species an assignment rule defines is the mirror image of an accumulator, and
it is handled the same way — automatically, because it is structural rather than
a judgement call. Such a species has no equation of its own: BNGsim emits its
slot `fixed` and its value comes from the rule, so its RHS row, and therefore its
Jacobian row, is identically zero. An accumulator contributes a zero *column*; a
rule target contributes a zero *row*; either one makes the system singular.

So rule targets are excluded from the solved subspace on every solve, and
`ss.excluded_species` lists them alongside anything your own `mask=` dropped —
the two are intersected, never overridden. Their reported value is the rule
evaluated at the returned state, so it agrees with what `run()` reports late in a
trajectory, and their `dY_ss/dp` row is the chain rule through the assignment:

```python
ss = sim.steady_state(sensitivity_params=["ks"])
ss["S"]                      # the rule's value, not the frozen initial one
ss.sensitivity[i_S]          # d(rule)/dks, the same number as
ss.output_sensitivities(["observable:S"])
```

Before this, the value was whatever the slot was seeded with at t=0 — a factor of
ten out on a model whose rule doubles a species that grows tenfold — and a single
assignment rule made the sensitivity solve refuse the whole model, gradients of
ordinary integrated species included, with a message that blamed a
conservation-law continuum.

### Unstable roots (saddles)

Not every root of `f(y) = 0` is a steady state the system can occupy. A bistable
model has three: two attractors and, between them, a **saddle** on the
separatrix — an equilibrium that satisfies `f(y) = 0` exactly and that no
trajectory can rest on, because any perturbation grows.

`method="newton"` used to be able to return one (issue #78). The trajectory
*slows down* near a separatrix, so two successively tighter bursts hand KINSOL
almost the same seed, both polish to the saddle, and the seed-stability test —
which asks whether refining the seed moves the root — is satisfied precisely
where it should not be. On the Gardner 2000 toggle at `alpha_2 = 53.53` that
returned `[28.245, 1.830]` with `converged=True` and a residual of `2.8e-10`,
while `method="integration"` returned the correct branch.

The polish is now certified before it is accepted: BNGsim takes the eigenvalues
of the Jacobian **restricted to the species the polish solved for** and rejects
the root when any has a positive real part, continuing to integrate instead. The
saddle above has eigenvalues `+0.406` and `-2.406`; the branch integration
reaches has `-0.863` and `-1.137`.

Three fields report it:

```python
ss = sim.steady_state(method="newton")
ss.root_stability             # "stable" — the returned root is an attractor
ss.n_unstable_roots_rejected  # 1 — a saddle was discarded on the way there
ss.eigenvalues                # array([-0.863+0.j, -1.137+0.j]) — the spectrum it was read off
```

- `root_stability` is `"stable"`, `"undetermined"`, or `"unstable"` for a root a
  Newton polish returned, and `""` for the integration path — a trajectory
  cannot come to rest on an unstable equilibrium, so there is nothing to certify.
- `"undetermined"` means the certificate declined and the root was accepted as
  before: the restricted system has **more than 512 unknowns** (a full spectrum
  is O(n³) and would cost more than the solve it is checking — 2.1 s at 1281
  species), or its Jacobian is entirely zero. Six of the 585-model `ode_fullnet`
  corpus land here, all on the size limit.
- `"unstable"` is returned only when *your own initial condition* was already
  that root. There is nowhere to fall back to then — integration would return the
  same state — so the verdict is reported rather than acted on.
- `n_unstable_roots_rejected > 0` on a result whose `method_used` is
  `"integration"` explains why the polish did not answer.
- `eigenvalues` is the spectrum the verdict was read off (issue #523): the
  Jacobian **restricted to the species the polish solved for** — one entry per
  unknown, so `n_species` less one per conservation law less the masked
  species — sorted by descending real part, a conjugate pair adjacent with its
  `+imag` member first. The full system's spectrum is this plus one zero per
  conservation law; `np.linalg.eigvals(model.jacobian(ss.concentrations))`
  shows them (see [Model evaluators](evaluators.md)). It is empty when no
  spectrum was computed — every integration result, and a root the certificate
  declined on before the eigensolver, which is the size limit above. A
  continuation that tracks a branch reads this field at each step: a Hopf
  bifurcation is the leading conjugate pair crossing the imaginary axis.

Rejecting a saddle does not cost the tighter residual the polish buys: the burst
leaves the saddle's neighborhood on its own, and a later rung polishes the
attractor. On the toggle dose above, `method="newton"` returns the correct branch
at a residual of `7.5e-14` against integration's `3.2e-10`.

The certificate is *not* a bifurcation analysis — it says whether the root that
was returned is linearly stable, not how many other roots exist. For the full
picture, scan the parameter and compare `method="integration"` from different
initial conditions.

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
  `jacobian="auto"` applies everywhere else, and the same matrix the stability
  certificate reads — and, since issue #127, the same one the march and the
  polish factor; `jacobian="fd"` pins the difference quotient for all four.
- **∂f/∂p** — the analytical parameter derivative the code-generated
  sensitivity RHS emits, the same one CVODES integrates against on the
  time-course path. Since issue #67 this covers Functional rate laws too, as long
  as they are smooth algebra. What has no such derivative to emit is
  Michaelis–Menten, and any Functional law carrying a condition (`if()`, a
  comparison, a logical — issue #68) or a non-smooth builtin
  (`abs`/`min`/`max`/`floor`/`ceil`/`round`); for those the factor is still
  finite-differenced — with a warning, and `ss.sens_dfdp_source` says so.

Where a factor *is* differenced, the one-sided step is **relative** to what it
perturbs (issue #76): `sqrt(eps)·|p|` for a parameter, and `sqrt(eps)·max(|y_j|,
max|y|)` for a species — floored at the state's own magnitude, so a species at
zero still gets a probe on the scale of the model it belongs to. Before #76 both
were floored at an absolute `sqrt(eps)`, which is a small probe only for a
quantity of order 1: a rate constant of 1e-9 was moved by 1500% of itself, and a
model carrying molecule counts was probed at 1e-14 of its state.

A step relative to the parameter has its own failure, at the other end: when a
parameter's own term is a small fraction of the derivative it sits in, that step
moves the derivative by less than its roundoff and the quotient is noise rather
than a gradient. So the parameter probe takes **two** steps — the relative one
and an absolute `sqrt(eps)` — and each component of `∂f/∂p` keeps the quotient
whose response cleared that component's own roundoff floor (issue #123). Nothing
about this is tunable, and it is not a substitute for a closed form: a model
whose gradient depends on the step size wants the analytical `∂f/∂p`, which is
what the refusal below is for.

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

#### Where the columns are returned: an isolated root (issue #995)

`-J⁻¹·∂f/∂p` is the derivative of the steady state only where the steady state
is an isolated root that the system rests at, and the state the solve returned
is on it. Where the steady states form a continuum, the one a run
ends at depends on the path it took, and that dependence is not in the root
equations. (A root of higher order, which a species nears as 1/t, has a
singular Jacobian too, and no derivative with respect to a parameter that would
move it off zero.) An epidemic that burns out is a continuum: `S + I -> 2 I`, `I -> R` ends at
whatever S it left, every state with I = 0 being a steady state, and the solve
returned dS*/dg = 26.25 where the final-size relation gives 66.05. An
irreversible branch to two products is another, and so is a total that is
conserved but not by a linear law.

The solve checks, and raises `SimulationError` where a check fails. It takes
one Newton step from the state it returned and factors its system again there,
and it takes a run on for `max_time` from that state with every concentration
moved by a millionth of itself. What it reads is each a ratio of two quantities
in the same units. An entry of a column is taken over its species' own
concentration, `ss.sens_species_scale`, the larger of the returned value and
the corrected one, so that the units of a species or the size of its
compartment change nothing that is asked of it.

A species at a zero has no concentration to be taken over. It is one that the
corrected state has at nothing and that nothing left there makes: its rate is
zero with every such species set to zero, and from next to nothing it does not
grow. Its entries are taken over where it has been and the largest among the
species it is coupled to, no more than a conserved total it belongs to allows.
A species that is small and has a steady value, 1e-12 beside another at 1, is
not at a zero, and neither is one that makes itself and is falling towards
what its surroundings carry. Each keeps its own scale, and has to be solved to
its steady value before its entries are returned (the column shift, below),
which the solve does.

| On the result | What it is | Refused |
| --- | --- | --- |
| `sens_root_determinant_ratio` | The determinant of the system at the corrected state over the one at the returned state. 1, to the accuracy of the solve, at an isolated root; next to nothing where the Jacobian is singular at the steady state the solve was approaching; 1/2 at a root of higher order; negative or large where the state is far from its root, or a rate law is discontinuous between the two. | outside 0.6 to 1.67 |
| `sens_root_pivot_share` | The least a pivot of the factorization is of the terms it was computed from. 1e-16 where a pivot is what rounding left of a zero: a Jacobian that is singular whatever the state, as two products of one irreversible branch make it. | below 1e-10 |
| `sens_root_condition` | The componentwise condition number of the system, the Perron root of `\|A⁻¹\|·\|A\|`: how many times a relative error in each entry of the Jacobian is magnified in the columns. 2e16 for a set of species that exchange among themselves and are produced and never consumed. | above 1e12 |
| `sens_root_column_shift` | The largest move of a column when it is solved again a Newton step on, as a fraction of its largest entry. Above 0.01 the state is stepped on, up to six times (`sens_root_newton_steps`), until it is not. | still above 0.01 after six steps |
| `sens_root_hold_shift` | The same where a run ends that is taken on for `max_time` from a millionth beside the returned state. Not a number where that run failed. `sens_root_hold_time` is the time it reached, with the `max_steps` steps it has. | above 0.01, not a number, or a run short of `max_time` |
| `sens_root_growth_rate` | The largest real part among the eigenvalues of the system, up to 512 unknowns, beside `sens_root_spectral_radius`, the largest eigenvalue in size. | above 1e-6 of the spectral radius |
| `sens_root_relaxation` | The most of a column that a run of `max_time` would leave unestablished, as a fraction of its largest entry. | above 0.01 |

The request is refused whole where one column fails, so which parameters are
asked for together can decide it.

The first three say the steady state is not an isolated root. (A determinant
ratio that is negative, or above 1.67, is also what a state far from an
isolated root gives, where the rates are so small that `tol` passes it, and
what a rate law that is discontinuous between the two states gives; a smaller
`tol` settles the first.) The columns of such a model come from a time course
with forward sensitivities, run to the steady state:

```python
sim = bngsim.Simulator(model, sensitivity_params=["g", "I0"])
result = sim.run(t_span=(0, 1e4), n_points=2)
result.sensitivities[-1]          # (n_species, n_params) at the last time
```

The column shift says how far the state the solver stopped at is from the
steady state, in what matters here: its columns. `tol` bounds the residual
`||f(y)||₂/n` and not the distance to the root, and a model whose
concentrations are 1e-6 passes `tol=1e-9` a long way off: BIOMD0000000002 is
accepted 0.02% from its steady state, where every column is 5.9% from the
derivative. Where a column moves by more than 1% in one Newton step, the state
is stepped on, a Newton step at a time, until none does, and what is returned
is the last: `ss.concentrations`, `ss.residual` and `ss.sensitivity` are then
those of the stepped state, which is the root to what a step still moves it
by, and `ss.sens_root_newton_steps` says how many steps that took. Where the
first step moves no column, it is 0 and the result is the solver's own, to the
last bit. A root that was stepped to has to be where the run below ends, within
1% in every species: Newton can step to a root the system leaves. The solve
refuses where the columns have not settled in six steps:
the state is far from a root for the size of its rates, which a smaller `tol`
mends, or the steady state is not an isolated root. A column whose every
entry, over its species' scale, is below `1e-3/|p|` is measured against that
instead of its own largest entry: a species that moves by less than a
thousandth of its scale when the parameter doubles.

A species that is small beside the rest is solved to its own steady value by
the same steps. A made at `5e-10·G` and removed at 5e-3 has a steady value of
1e-7 beside G at 1, and the residual is under `tol=1e-9` where A starts, at
nothing: dA*/dkd came back -5.8e-9 for -2e-5. Its entries are taken over its
own concentration, the column is seen to move, and two steps take A to 1e-7.

The hold shift says the state is not one a run stays at. An integration stops
at the first state whose residual is under `tol`, and that says where the run
is, not where it is going: BIOMD0000000407 is returned as it starts, beside a
stable root, and a run takes one of its species from 2.448 to 3e-4. A smaller
`tol` runs past such a state. And a root can be one the system leaves, however
exactly the state is on it: the middle root of a bistable switch, or the fixed
point inside a limit cycle, which `method="newton"` finds and a model can be
started on. `-J⁻¹·∂f/∂p` there is how the root moves, not where a run ends. The
run starts a millionth beside the state because one started on such a root
stays on it. A species that is absent is not moved: a state is not asked
whether it would last the arrival of something the model does not start with.

The run has `max_steps` steps. Where it uses them short of `max_time`, it was
not seen to stay, and the columns are refused: an oscillation about the state
does this where it is slow to grow or to die away. Give a `max_time` such a
run reaches, or more steps.

The growth rate says the same of a state a run of `max_time` is too short to
leave. A millionth grows to a hundredth in `9.2/rate`, and an oscillation that
grows more slowly than that leaves the run where it started; the relaxation
below does not see it either. The eigenvalues do, where the growth is above
what they are themselves known to, a millionth of the largest of them. What
neither shows is returned: a state the system leaves by an oscillation that
grows more slowly than `9.2/max_time` and than a millionth of the fastest rate
in the model, or one in a system of more than 512 unknowns.

Without `sensitivity_params`, `steady_state()` returns what it did: the state
the solver stopped at.

The relaxation says the column is that of a steady state the model does not reach
in the time the solve was given. A species whose turnover is switched off at
the steady state stays where it started, and the state is a root to the last
bit; its column is the ratio of a production and a removal that are both next
to nothing, 12,500 in MODEL1607210000 where no run moves the species at all.
Raise `max_time` to ask about the state such a run does reach, or take the
column from a time course over the times that matter.

`sens_jacobian_rcond`, `min|U|/max|U|` of the LU, is still reported, and
decides nothing. It depends on the units: for an isolated root it falls with
the size ratio between two compartments, to 4e-16 at 1e8 where
`sens_root_condition` stays at 6.2, and it reads 1.0 for the epidemic above
with its sink masked out. The warning that was logged below 1e-8 is gone.

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

A stable steady state keeps nothing of its initial conditions but the conserved
totals. In a model with a conservation law, `A <-> B` with `A + B` fixed by
where the run starts, a parameter that sets an initial amount moves the steady
state through the total, and its `dY_ss/dp` column carries that: `[1/3, 2/3]`
for `A0` at `kf = 1`, `kr = 0.5` (issue #704). The seeding is the one a time
course starts from (`Model.effective_ic_sensitivity`), read from the state the
solve starts at: a species that has been moved off its initial condition
contributes nothing, and neither does any species once `save_concentrations()`
has made the state its own baseline. A fixed species that a parameter sets,
`$A() A0`, moves what reads it by the same seed.

Three requests are refused, with `SensitivityUnsupportedError`:

- a parameter that sets the initial amount of a conserved species, on a state a
  `run()` has advanced. The total is still what the parameter made it, and the
  state no longer says so (a time course refuses sensitivities there too).
  `reset()` first;
- a compartment size, in a model with a conservation law;
- any parameter, where a law spans compartments of different size and an
  assignment rule sets the size of a compartment. The right-hand side divides
  by the size the model loaded at where the rule gives another (issue #745),
  so the steady state is that of another system.

A law across compartments of different size is a total of amounts, and carries
the sizes: `A + 2*B` for `A` in a compartment of size 1 exchanged with `B` in
one of size 2 (issue #758). Outside the cases above its columns are computed
like any other law's, where the steady state is an isolated root. (Such a
model was refused wherever `min|U|/max|U|` was below 1e-8, which an isolated
root reaches with compartments 1e5 apart in size. It is asked what every model
is asked now.)

With `mask=`, the columns are solved on the equations of the species the mask
kept (issue #995), with the laws that hold among those species alone. A
masked-out species is taken out of the laws that have it by one of them: that
law is given up, and the species it was solved for is an unknown instead, with
its own equation; the other laws are kept as the combinations that leave the
species out (`A + P` and `B + P` with P masked out leave `B - A`). A masked-out
species that a law is solved for is given by that law, as before. Where
the masked sink drains the law (`A <-> B -> P`: A and B end at 0 whatever the
total) the columns are what they were. Where a share of the total stays out of
the sink, which no steady-state solve can know, they are refused as those of a
root that is not isolated: `A <-> B -> P` beside `A -> C <-> D` returned 0 for
every column of C and D, with the mask, where C + D ends at the share of A
that took the second branch.

A masked-out species is held where the solve left it, and that is its part in
`dY_ss/dp` only where no equation of the kept species reads it, as none reads
a pure sink. A mask that leaves out a species one of them does read is refused
(`ss.sens_mask_held_species`, `ss.sens_mask_reader_species`): `A <-> M` with M
left out has dA*/dkf = -4/9, and -2/3 with M held.

The initial-condition axis itself is not computed, and
`output_sensitivities(..., axis="ic")` raises.

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

(An equilibration run with `steady_state=True` stops early only for a model
that reads no time and has no event, issue #710. For one that does, phase 1
runs to the end of its span, through any event in it: give it the span the
pre-condition is meant to last.)

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
  with no carried derivative is θ-independent literal ICs: no parameter seeds
  it, where a fresh start is seeded from the model's own initial-condition
  expressions (issue #704).
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
