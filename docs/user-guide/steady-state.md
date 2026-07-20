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
implicit function theorem: `dY_ss/dp = -J⁻¹ · (∂f/∂p)`, where J is the
Jacobian at steady state and `∂f/∂p` is computed by finite differences.

```python
ss = sim.steady_state(
    sensitivity_params=["kf", "kr", "kcat"],
)

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
`∂func/∂p` (e.g. a rate-law function `k3/(K4+G)` differentiated w.r.t. `k3`) —
matching the CVODES codegen chain rule. A downstream gradient consumer can reuse
its existing CVODE `output_sensitivities` code path unchanged.

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
same `sensitivity_params`, or a `reset()` (as an SBML/RoadRunner every-action
reset would do) wiped the carry-over. A fresh single sensitivity run is
unaffected, and `reset()` returns to fresh-start seeding.

Scope (matching the new-era pre-equilibration surface): the equilibration is a
**steady state** (PEtab `time = -inf`) and the perturbation is an **absolute**
(`=`) `setParameter` — the species state carries over, only a parameter
changes. Finite-time equilibration and initial-condition–axis (`sensitivity_ic`)
sensitivities across the boundary are out of scope (the latter raises); a model
with **events** warns, since event-time sensitivity discontinuities are handled
separately. The carried seed is model-level state alongside the concentrations,
introspectable via `model._core.ic_state_dirty` /
`model._core.has_pending_sensitivity_seed`.
