"""bngsim.jax — Differentiable ODE solving with JAX integration.

This submodule provides JAX-traceable wrappers around BNGsim's CVODE
solver. The key function, :func:`differentiable_solve`, is registered
with ``jax.custom_jvp`` so that JAX's automatic differentiation
dispatches to CVODES forward sensitivities for exact gradients.

This combines SUNDIALS-quality stiff ODE solving (0.1ms per solve)
with JAX's composable AD system (``jax.grad``, ``jax.value_and_grad``,
``jax.jacfwd``).

**Requires**: ``pip install 'bngsim[jax]'``

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> from bngsim.jax import differentiable_solve
>>>
>>> model = bngsim.Model.from_net("model.net")
>>> p0 = jnp.array([model.get_param(n) for n in model.param_names])
>>>
>>> # Gradient of a loss function
>>> def loss(p):
...     Y = differentiable_solve(model, p, (0, 100), 101)
...     return jnp.sum((Y - data) ** 2)
>>> grad = jax.grad(loss)(p0)
"""

from __future__ import annotations

try:
    import jax as _jax  # noqa: F401

    _jax.config.update("jax_enable_x64", True)
except ImportError:
    raise ImportError(
        "JAX is required for the bngsim.jax submodule. Install with: pip install 'bngsim[jax]'"
    ) from None

from bngsim._jax_bridge import differentiable_solve

__all__ = ["differentiable_solve"]
