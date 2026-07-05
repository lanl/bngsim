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
print(result.sensitivity_params)   # all 40 param names
```

Each chunk clones the model (thread-safe deep copy) and runs an independent
CVODES instance. The GIL is released during C++ CVODE integration, so threads
achieve real parallelism. Near-linear speedup from 1→2→4→8 workers.

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

## Parameter gradients for optimization

`Result.gradient()` computes ∇_p L from the sensitivity tensor and a
user-supplied loss function, enabling gradient-based optimization:

```python
import numpy as np
from scipy.optimize import minimize

data = np.load("experimental_data.npy")  # (n_times, n_species)

def objective(p_vec):
    # Set parameters and simulate with sensitivities
    model.set_params(dict(zip(param_names, p_vec)))
    model.reset()
    result = sim.compute_all_sensitivities(
        t_span=(0, 100), n_points=101,
        n_workers=8,
    )

    # Compute loss and gradient
    loss = np.sum((result.species - data) ** 2)
    grad = result.gradient(
        lambda species, time: 2 * (species - data)
    )
    return loss, grad

# L-BFGS-B optimization with analytical gradients
opt = minimize(objective, x0=initial_params,
               method='L-BFGS-B', jac=True)
```

The gradient computation is O(n_times × n_species × n_params) — a single
matrix multiply per time point. Combined with parallel
`compute_all_sensitivities()`, the total cost of loss + gradient is
dominated by the CVODES solve, not the gradient algebra.

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
# defining expression.
p_flat = jnp.array([model.get_param(n) for n in model.param_names])
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
