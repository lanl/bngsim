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
`BNGSIM_SENS_DERIV_BUDGET_S`. It
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

## Logging

```python
import logging
bngsim.configure_logging(logging.DEBUG)
# Now all bngsim operations produce log output
```
