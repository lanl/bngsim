# Sensitivity analysis & gradients

## Forward sensitivity analysis (CVODES)

BNGsim integrates CVODES forward sensitivity analysis to compute ∂Y/∂p —
how species trajectories change with respect to parameters. This enables
parameter identifiability analysis, Fisher information, and gradient-based
optimization.

```python
# Compute sensitivities for specific parameters
sim = bngsim.Simulator(
    model, method="ode",
    sensitivity_params=["kf", "kr"],
)
result = sim.run(t_span=(0, 100), n_points=101)

# Sensitivity tensor: (n_times, n_species, n_params)
print(result.sensitivities.shape)  # (101, 5, 2)
print(result.sensitivity_params)   # ["kf", "kr"]
print(result.has_sensitivities)    # True

# dA/dkf at the last time point
print(result.sensitivities[-1, 0, 0])
```

### Parameters that set an initial condition

When a species' initial condition names a parameter — `R() R0` in the `.net`, or
`R() Rtot` with `Rtot = R0` — that parameter reaches the trajectory through
`x_R(0)`, so the seed `∂x_R(0)/∂R0` is part of the answer. BNGsim differentiates
the IC expression (through nested derived parameters) and seeds it automatically.

That seed describes the initial condition the *model declares*, which is what
`reset()` returns the state to. If you **assign** the species instead —
`set_concentration("R()", 7.0)`, a bulk `set_state`, an externally injected state —
the parameter no longer reaches its initial condition, and the row is dropped: a
literal assignment has `∂x_R(0)/∂θ = 0`, which is also what `set_concentration`
documents about itself. Nothing changes for a model you have not assigned into
(across the 585-model `.net` corpus and 120 SBML models, every one loads with its
live state equal to its baseline).

If the value you assign *does* depend on a fitted parameter — a dose in molecules
computed through a fitted volume — say so, because only you know:

```python
v = dose_nM * 1e-9 * NA * model.get_param("Vecf")
model.set_concentration("L(r)", v)
model.declare_ic_sensitivity({"L(r)": {"Vecf": v / model.get_param("Vecf")}})
```

A declaration is the most specific statement available and wins over both the
expression-derived row and the drop; declaring `{}` pins a row to zero explicitly.
Inside a `parameter_scan` `on_point` hook the derivative is *measured* through the
hook instead, so a dose there needs no declaration — see
[Steady state](steady-state.md#the-dose-an-on_point-hook-applies-issue-111).

### SBML species set by an `<assignmentRule>`

A species an assignment rule defines — `IRS_total`, `InR_active`, the *reported*
quantities of a published SBML model — has no ODE of its own. BNGsim emits its
state slot `fixed` and overwrites the reported column each step from the rule's
live value: an observable when the rule is linear in species, a function
otherwise. Its **sensitivity row is the chain rule through that assignment**, and
`result.sensitivities[:, i, :]` carries it, so it lines up with
`result.species[:, i]` the way every other row does and `Result.gradient` can
score it (issue #221). `output_sensitivities("species:<name>")` returns the same
numbers.

Where the chain rule is not available the row is `NaN`, never `0.0` — a
structural zero is indistinguishable from a measured one, and an optimizer reads
it as a flat objective rather than as a missing term. The run warns naming the
species, and `result.ar_sensitivity_refused` lists them:

```python
result.ar_sensitivity_refused          # frozenset({"stimulus"})
result.output_sensitivities("species:stimulus")
# ValueError: ... has no output sensitivity — uses unsupported construct: if() conditional
```

Two things put a species there: codegen declined the rule's own output
sensitivity (a `piecewise` lowers to an `if()`, which #198 refuses rather than
guess at), or the reported value carries a time-varying volume rescale the
redirect does not model. A `NaN` row does **not** cost you the rest of the
gradient: `gradient`, `sse_gradient`, `chi2_gradient` and
`neg_log_likelihood_gradient` drop these rows wherever your `dL/dY` is exactly
zero, so a fit that never scores that species still gets a number. Weight one and
the gradient is `NaN`, which is the honest answer.

## Parameters that set *when*, not *how fast*

Some fitted parameters never appear in a rate. They set the **time at which the
dynamics change** — the onset of an intervention, the start of a dose, the
moment a promoter flips. BNGL and `.net` models spell this as an `if()` over a
clock:

```
begin functions
    1 rate_X()  if((t>=sigma), k, 0)
end functions
```

where `t` is an observable on a unit-rate counter species (`dT/dt = 1`), which
is how BNG models read simulation time inside a rate law.

`sigma` is reached only through the *condition*, so `∂f/∂sigma` is a clean `0`
inside each branch — sympy drops the boundary delta when a parameter appears
nowhere else. A forward sensitivity that used only the variational source term
would therefore report a flat zero for a parameter the trajectory obviously
depends on. The whole gradient is a finite jump at the crossing `t*`:

```
s(t*⁺) = s(t*⁻) + (f⁻ − f⁺) · ∂t*/∂p
```

BNGsim locates the crossings, computes `∂t*/∂p`, stops the integration *on*
`t*`, and applies the jump. Nothing needs declaring: `.net` and BNGL models
register no discontinuity triggers — only the SBML loader does — so the
conditions are recovered from the function bodies themselves.

```python
res = sim.run(t_span=(0, 10), n_points=101, sensitivity_params=["sigma", "k"])
res.sensitivities[:, x_idx, 0]   # ∂X/∂sigma: exactly 0 before t*, −k after
```

A model with `if()` conditions but **no fitted switch time** yields no records
at all, and its stepping stays bit-for-bit identical to a run without
sensitivities.

### Clock forms that are recognised

The crossing has to be solved for from the condition. These forms are handled:

| Condition shape | Example | Notes |
|---|---|---|
| Affine in the clock | `t >= sigma` | The common case. `∂t*/∂p = ∂(threshold)/∂p`. |
| A monomial in the clock | `t^3 >= thresh` | Any degree. `c·clock^n` is monotonic over a non-negative clock, so it crosses once. |
| A quadratic in the clock | `a*t^2 + b*t >= c` | The first shape with genuinely two crossings, and both are compensated (issue #421). A *general* polynomial of degree 3 or more is declined: a cubic with three real roots has no usable radical form, and from degree 5 there is none at all. |
| Repeating schedule | `rem(t, P) >= d` | A `floor()`-periodic dosing schedule (issue #436). |
| libSBML's `rem()` expansion | `if(sign(a)!=sign(b), …)` | libSBML does not emit `rem()`; it expands it into a sign test over two remainders. Read back to the same schedule (issue #465), so a model gets the same gradient whichever tool wrote its SBML. |
| A guard spelled as a comparison | `(t < 0) == 0` | Resolved the way the equivalent boolean is (issue #473). |

A crossing time that comes out non-real is treated as a crossing that never
happens, rather than crashing the run.

### Derived switch times

A threshold built from other parameters is chain-ruled to the fitted primaries.
With `sigma = t0 + t_delta` and `tau1 = sigma + t_delta2`, a `t0` jump is placed
at **both** the `sigma` and `tau1` crossings, while `t_delta2` gets a jump only
at `tau1`. A threshold the run never reaches contributes an exactly zero column.

### Two switches at the same instant

When two crossings share an instant, each must be charged only its own jump. The
before-branch is read with the other switch's threshold bumped away, so a
coincident pair does not merge into a single doubled jump (issue #375). Models
with distinct switch times are unaffected and read the plain `f⁻ − f⁺`.

### Landing on the crossing

A discontinuity root alone cannot catch these. CVODE tests for a root only on a
step it *accepts*, so where the jump is large enough that the error test rejects
every step containing the crossing, `t` creeps to the last double below `t*` and
wedges at `t + h == t` without the root ever firing (issue #305). BNGsim
therefore registers the crossing times as explicit stops, so the integrator
lands on each one and restarts on the far side. This applies to a rate law
switching on simulation time (issue #445) and to one switching on a counter
species (issue #443), where the counter is landed exactly on its threshold so
the restart reads the after-branch.

These stops are added for any model carrying such a switch, sensitivities or
not, because stepping over the discontinuity was never correct.

### What is declined

A condition whose crossing time moves with the *state* rather than with a
parameter is a different problem, and the analytic path declines it rather than
returning a gradient that silently omits the saltation term. So does a
conjunction, a negation, and a comparison whose sides are themselves
comparisons. A parameter that both sets a switch time and acts inside a branch
is rejected rather than answered with the jump alone.

A decline is never silent. Ask the Simulator directly:

```python
sim.has_analytic_sens_rhs      # False when the run falls back
sim.sens_rhs_decline_reason    # why, in words, or None
```

The fallback is CVODES' own difference quotient, which is correct and slower.
See the [PyBNF guide](pybnf.md#ask-each-model-whether-its-gradient-is-analytic)
for using this to triage a fit.

## Parallel sensitivity computation

For models with many parameters (Np), computing all sensitivities serially
is expensive (O(Np) overhead per CVODE step). `compute_all_sensitivities()`
splits parameters into chunks and runs them in parallel via thread pool:

```python
sim = bngsim.Simulator(model, method="ode")

# Compute full sensitivity tensor using parallel chunks
result = sim.compute_all_sensitivities(
    t_span=(0, 100),
    n_points=101,
    chunk_size=2,     # 2 params per CVODES job (optimal)
    n_workers=8,      # parallel threads
)

# Full tensor: (n_times, n_species, n_params)
print(result.sensitivities.shape)  # (101, 149, 40)
print(result.sensitivity_params)   # == model.primary_param_names
```

Each chunk clones the model (thread-safe deep copy) and runs an independent
CVODES instance. The GIL is released during C++ CVODE integration, so threads
achieve real parallelism. Near-linear speedup from 1→2→4→8 workers.

### What the default column set is, and why it is narrower than `param_names`

`params=None` means *every independent knob*, which is
`model.primary_param_names` — not `model.param_names` (issue #203).
`result.sensitivity_params` is always the authority on what the columns are, and
anything dropped is named in a warning. Two classes come out:

- **Derived (expression-backed) parameters** — `_rateLaw1 = chi*kon` from a
  compound BNGL rate law, `_rateLaw_R16_fwd = alpha*konBT` from an SBML kinetic
  law. Such a parameter reaches the trajectory only through the primaries it is
  built from, and *their* columns are total derivatives through it. So the two
  columns are the same physical effect twice, in exact proportion
  `d(derived)/d(primary)`, and `Result.gradient` contracts the whole parameter
  axis into one vector an optimizer then steps along in every coordinate at
  once. Roughly one SBML model in five carries some (279 of the 1,291 loadable
  rr_parity models, 9,524 parameters in total).
- **Synthesized slots** (`model.param_is_internal`) — parameter slots bngsim
  created for its own bookkeeping rather than ones the model declared.
  `set_param` refuses a value-changing write to either kind, so neither is a
  coordinate that can move on its own. `_V0_<comp>` is bngsim's record of a
  compartment's size at load, which the rate constants in that compartment are
  normalised against — differentiate the compartment size itself, an ordinary
  writable parameter. And every **function** has one, holding the value it last
  evaluated to; the engine rewrites it from the function's own expression before
  every derivative evaluation, so that column is identically zero (issue #227) —
  differentiate the parameters the function's expression reads.

Naming a derived parameter — or `_V0_<comp>` — in `params=[...]` still returns
its column: an explicit ask is a statement that you want that derivative *on its
own terms*, treating the parameter as a free axis. That is exactly what
`bngsim.jax.differentiable_solve(..., flat=True)` asks for, and why the default
here (`flat=False`'s list) and that opt-in now agree end to end. (`flat=True`
leaves out the synthesized slots too, for the same reason: its vector is one
`set_param` per name, and those writes are refused.)

**A function's slot is the exception, and it raises** (issue #329). "On its own
terms" needs terms to exist: a derived parameter has them, because
`set_param(..., force_override=True)` pins it and the pin survives. A function's
slot has none — the next `evaluate_functions()` overwrites it whatever you do,
which is why `set_param` refuses it outright and `force_override` does not help.
So the column can only ever be identically zero, and asking for it by name in
`sensitivity_params=`, `params=[...]`, or `steady_state(sensitivity_params=...)`
is refused with a `ValueError` naming the expression to differentiate instead.

This matters most on **assignment-rule-driven SBML**, where the rule targets can
outnumber the real knobs several to one — 38 of 46 parameters in
`BIOMD0000000126`, 35 of 38 in `BIOMD0000000266`. Passing `model.param_names`
wholesale used to return a tensor that was mostly structural zeros with nothing
marking which columns those were. Pass `model.primary_param_names`, or let
`params=None` pick it.

The chain rule *through* an `<assignmentRule>` is unaffected: a parameter whose
only route to the right-hand side is a rule target differentiates normally, since
bngsim lowers the rule to a function the emitted sensitivity RHS differentiates
like any other expression. It is the rule's **target** that is not a column, not
the parameters underneath it.

What is *not* dropped is a constant written as arithmetic — `gamma 1/7`,
`pi 2*asin(1)`, `c6 ln(2)/120`. Those name nothing, so there is no primary
underneath them carrying their effect, and they are ordinary knobs (issue #227).

## Fisher Information Matrix

The Fisher Information Matrix (FIM) quantifies how much information observed
species trajectories carry about each parameter — the foundation for
parameter identifiability analysis and experimental design.

```python
# Compute FIM from sensitivity data
fim = result.fisher_information(sigma=0.1)  # scalar noise σ
print(fim.shape)  # (n_params, n_params)

# Per-species noise
sigma_per_species = np.array([0.1, 0.5, 1.0, ...])
fim = result.fisher_information(sigma=sigma_per_species)

# Identifiability diagnostics
print(np.linalg.cond(fim))        # condition number
eigvals = np.linalg.eigvalsh(fim)
print(eigvals[:3])                 # smallest eigenvalues → least identifiable
```

The FIM is the Cramér–Rao lower bound on parameter covariance:
Cov(p̂) ≥ FIM⁻¹. Large diagonal entries indicate identifiable parameters;
near-zero eigenvalues indicate practical non-identifiability.

That last reading is only sound if the columns are independent, which is what
the default column set above guarantees. `Sᵀ Σ⁻¹ S` over a parameter axis that
contains both a derived parameter and a primary underneath it is rank-deficient
**by construction** — the two columns are exactly proportional, so there is a
null direction that says nothing about the model or the data. Before #203 that
was the default on any model with derived parameters.

## Parameter gradients for optimization

`Result.gradient()` computes ∇_p L from the sensitivity tensor and a
user-supplied loss function, enabling gradient-based optimization:

```python
import numpy as np
from scipy.optimize import minimize

data = np.load("experimental_data.npy")  # (n_times, n_species)

# The fitted vector is the default column set, so `grad` lines up with `p_vec`.
names = model.primary_param_names

def objective(p_vec):
    # Set parameters and simulate with sensitivities
    model.set_params(dict(zip(names, p_vec)))
    model.reset()
    result = sim.compute_all_sensitivities(
        t_span=(0, 100), n_points=101,
        n_workers=8,
    )
    assert result.sensitivity_params == names

    # Compute loss and gradient
    loss = np.sum((result.species - data) ** 2)
    grad = result.gradient(
        lambda species, time: 2 * (species - data)
    )
    return loss, grad

# L-BFGS-B optimization with analytical gradients
opt = minimize(objective, x0=[model.get_param(n) for n in names],
               method='L-BFGS-B', jac=True)
```

`gradient()` sums over time and species but *not* over parameters — the
double-counting hazard is downstream, in the optimizer, which steps along every
coordinate of the returned vector at once. That is only meaningful if the
coordinates are independent, which is what the default column set above
guarantees and a hand-written `sensitivity_params=` list does not: if such a
list names both `_rateLaw1` and the `kon` underneath it, do not hand the
resulting vector to an optimizer over both.

The gradient computation is O(n_times × n_species × n_params) — a single
matrix multiply per time point. Combined with parallel
`compute_all_sensitivities()`, the total cost of loss + gradient is
dominated by the CVODES solve, not the gradient algebra.

**SBML compartment sizes are writable and differentiable** (issues #164, #170).
On an SBML model `model.param_names` includes the compartments. A
compartment size is now an ordinary writable parameter — `set_param("Liver", v)`
re-derives everything the volume decides (the amount↔concentration conversion,
an amount-declared initial condition, the mass-action scalar, the SSA propensity
volume, and any `<initialAssignment>` that reads the size — the PBPK idiom of
copying each compartment into its own parameter) and reproduces *reloading the
model at that size*, bit for bit. A volume scan or a gradient-free fit needs
nothing special. The generated C reads the
volume from `p[]` rather than baking it, so the emitted source does not depend on
the load-time size and the write lands on `codegen=True` and on an
already-compiled `.so` too — including a write that arrives mid-scan, after the
source was generated.

The **gradient** followed in stage 3: `d/dV` now carries the storage half as well
as the kinetic-law half — including the initial-condition seed, which is
`-amount/V²` for an amount-declared species rather than zero — so `Liver` is an
ordinary column of `compute_all_sensitivities()` and `sensitivity_params=["Liver"]`
is accepted. `_V0_Liver` is not: that is bngsim's record of the load-time size
rather than the volume, `set_param` refuses to move it, and it is one of the
things the `params=None` default drops (see
[the default column set](#what-the-default-column-set-is-and-why-it-is-narrower-than-param_names)).

A handful of compartments still cannot be written, and those keep the original
refusal — `compute_all_sensitivities()` skips the column with a warning and
`sensitivity_params=["Liver"]` raises, because a column is exactly as trustworthy
as the write is. `model.unwritable_compartment_size_params` lists them and the
error names the reason per size. Reload at the size instead:

```python
m = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v})
```

...or difference over the rebuild, which is exact:

```python
def dloss_dV(v, h):
    up = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v + h})
    dn = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v - h})
    return (loss(up) - loss(dn)) / (2 * h)
```

## Differentiable ODE solving with JAX

BNGsim provides a JAX-traceable ODE solver via `bngsim.jax.differentiable_solve`.
This registers CVODE as a JAX custom primitive with a `custom_jvp` rule that
dispatches to CVODES forward sensitivities — combining SUNDIALS-quality stiff ODE
solving (0.1ms) with JAX's composable automatic differentiation (`jax.grad`,
`jax.value_and_grad`, `jax.jacfwd`).

```python
import jax
import jax.numpy as jnp
from bngsim.jax import differentiable_solve

model = bngsim.Model.from_net("model.net")

# Differentiate over primary parameters only (default). Derived
# ConstantExpression parameters such as BNG2.pl-emitted ``_rateLaw{N}``
# (for compound BNGL rate laws like ``chi*kon``) are recomputed from
# their primaries automatically, so ``jax.grad`` returns gradients
# with respect to ``model.primary_param_names`` with the chain rule
# through derived expressions correctly applied.
p0 = jnp.array(
    [model.get_param(n) for n in model.primary_param_names]
)

# Forward solve (no differentiation)
Y = differentiable_solve(model, p0, (0, 100), 101)

# Gradient of a loss function w.r.t. primary parameters
data = jnp.load("experimental_data.npy")

def loss(p):
    Y = differentiable_solve(model, p, (0, 100), 101)
    return jnp.sum((Y - data) ** 2)

grad = jax.grad(loss)(p0)                    # parameter gradient
val, grad = jax.value_and_grad(loss)(p0)     # loss + gradient

# Full sensitivity matrix via jacfwd
def solve_flat(p):
    return differentiable_solve(model, p, (0, 100), 101).ravel()

J = jax.jacfwd(solve_flat)(p0)  # (n_times*n_species, n_primary_params)

# Legacy / advanced: treat every parameter (including derived
# ``_rateLaw{N}``) as an independent coordinate. Use only when you
# really want to vary derived parameters independently of their
# defining expression. The vector is ``param_names`` minus the
# synthesized slots ``set_param`` refuses to write (issues #170, #227).
internal = [n for n, f in zip(model.param_names, model.param_is_internal) if f]
p_flat = jnp.array(
    [model.get_param(n) for n in model.param_names if n not in internal]
)
Y_flat = differentiable_solve(model, p_flat, (0, 100), 101, flat=True)
```

Requires: `pip install 'bngsim[jax]'`

**How it works**: The `@jax.custom_jvp` rule runs CVODES once per JVP call,
computing the primal solution and forward sensitivities simultaneously (single
solve, not two). The sensitivity tensor is contracted with the tangent vector
via `jnp.einsum('tsp,p->ts', sens, dp)`.

**Performance**: ~1.2× overhead vs plain ODE solve for large models — 23,000×
faster than Diffrax in internal benchmarking. Each call clones the model internally for
thread safety. Solver options (`rtol`, `atol`, `max_steps`) are passed through
as keyword arguments.

**When to use**: For JAX ecosystem integration (`optax`, `numpyro`, `blackjax`,
`scipy.optimize`). For non-JAX gradient computation, use `Result.gradient()`
which is lower-overhead and doesn't require JAX.

## Built-in objective gradients

BNGsim provides built-in gradient methods for the most common parameter
estimation objectives, eliminating the need to manually derive `dL/dY`:

```python
result = sim.compute_all_sensitivities(
    t_span=(0, 100), n_points=101, chunk_size=2, n_workers=8,
)

# Sum of squared errors (most common)
loss, grad = result.sse_gradient(data)

# Chi-squared (weighted by measurement noise)
loss, grad = result.chi2_gradient(data, sigma=0.1)
loss, grad = result.chi2_gradient(data, sigma=per_species_sigma)

# Negative Gaussian log-likelihood (includes constant term)
nll, grad = result.neg_log_likelihood_gradient(data, sigma=0.1)

# Partial observation (only fit species 0 and 2)
loss, grad = result.sse_gradient(
    data_subset, species_indices=[0, 2]
)

# Direct use with scipy L-BFGS-B
from scipy.optimize import minimize
def objective(p_vec):
    model.set_params(dict(zip(param_names, p_vec)))
    model.reset()
    result = sim.compute_all_sensitivities(...)
    return result.sse_gradient(data)  # returns (loss, grad)
opt = minimize(objective, x0, method='L-BFGS-B', jac=True)
```

All methods return `(loss_value, gradient_vector)` — the format expected by
`scipy.optimize.minimize(..., jac=True)`. For custom objectives not covered
by the built-ins, use `Result.gradient(loss_fn)` with a user-supplied
`dL/dY` function, or the JAX bridge for automatic differentiation.

### Adding a new built-in objective (Developer Guide)

The pattern for adding a new objective gradient method to `Result` is:

1. **Derive `dL/dY`** — the partial derivative of your loss function with
   respect to each species value at each time point. This is a
   `(n_times, n_species)` array.

2. **Add a method** to the `Result` class in `bngsim/python/bngsim/_result.py`.

3. **Contract with sensitivity tensor** — the parameter gradient is
   `∇_p L = Σ_t (dY/dp)^T · (dL/dY)_t`, computed as a loop over time points.

**Worked example: negative binomial log-likelihood** (for count data in
epidemiological models where `Y` is the expected count and `D` is observed):

```python
def negbinom_gradient(
    self,
    data: NDArray[np.float64],
    r: Union[float, NDArray[np.float64]],
    *,
    species_indices: Optional[list[int]] = None,
) -> tuple[float, NDArray[np.float64]]:
    """Negative binomial NLL and parameter gradient.

    NLL = -Σ_{t,i} [D*log(p) + r*log(1-p)]  (up to constants)
    where p = Y/(Y+r), Y = model prediction, D = observed count.

    dL/dY = (D - Y*r/(Y+r)) * (-r/(Y+r)^2)
          = r*(D - Y) / (Y*(Y+r))
    """
    if not self.has_sensitivities:
        raise ValueError("No sensitivity data.")

    data = np.asarray(data, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    Y = self._species
    sens = self._sensitivities

    if species_indices is not None:
        Y = Y[:, species_indices]
        sens = sens[:, species_indices, :]

    # p = Y / (Y + r)
    p = Y / (Y + r_arr)
    p = np.clip(p, 1e-15, 1 - 1e-15)  # numerical safety

    # NLL (negative log-likelihood, dropping constant terms)
    nll = -float(np.sum(
        data * np.log(p) + r_arr * np.log(1 - p)
    ))

    # dL/dY = r * (Y - D) / (Y * (Y + r))
    dL_dY = r_arr * (Y - data) / (Y * (Y + r_arr) + 1e-30)

    # Contract with sensitivity tensor
    nt = sens.shape[0]
    np_ = sens.shape[2]
    grad = np.zeros(np_, dtype=np.float64)
    for t in range(nt):
        grad += sens[t].T @ dL_dY[t]

    return nll, grad
```

**Key rules:**
- The method must check `self.has_sensitivities` and raise `ValueError` if missing.
- Support `species_indices` for partial observation.
- Return `(loss, gradient)` tuple — both are always computed together.
- The gradient contraction loop `for t in range(nt): grad += sens[t].T @ dL_dY[t]`
  is the same for ALL objectives — only `dL_dY` changes.
- Add tests in `test_objective_gradients.py` that verify:
  (a) shape, (b) zero-residual gradient is zero, (c) consistency with
  `Result.gradient()` using the same `dL/dY` manually, (d) error handling.
