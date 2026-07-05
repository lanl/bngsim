"""bngsim._jax_bridge — JAX custom_jvp bridge for differentiable ODE solving.

Registers BNGsim's CVODE ODE solver as a JAX custom primitive with a
``custom_jvp`` rule that dispatches the forward-mode Jacobian-vector
product to CVODES forward sensitivities. This gives:

- **CVODE-quality numerics** (0.1ms per solve, 4 orders of magnitude
  faster than Diffrax)
- **Exact sensitivities** (CVODES forward method, same integration
  direction, no adjoint brittleness)
- **JAX composability** (automatic chain rule, ``jax.grad``,
  ``jax.value_and_grad``, ``jax.jacfwd``)

Architecture:
  1. ``differentiable_solve()`` — public API, accepts keyword solver opts
  2. ``_solve_core()`` — ``@jax.custom_jvp`` primitive (only ``params``
     is differentiable; model, t_span, n_points, opts are non-diff)
  3. ``_solve_core_jvp()`` — JVP rule: runs CVODES once (primal +
     sensitivity simultaneously), contracts tangent via einsum

Optional dependency: ``pip install bngsim[jax]``.
"""

from __future__ import annotations

import functools
import logging

import numpy as np

logger = logging.getLogger("bngsim")


def _check_jax():
    """Import JAX and enable 64-bit precision. Raises ImportError if missing."""
    try:
        import jax  # noqa: F401

        jax.config.update("jax_enable_x64", True)
        import jax.numpy  # noqa: F401
    except ImportError:
        raise ImportError(
            "JAX is required for bngsim.jax.differentiable_solve(). "
            "Install with: pip install 'bngsim[jax]'"
        ) from None


def differentiable_solve(
    model,
    params,
    t_span: tuple[float, float],
    n_points: int,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 10000,
    chunk_size: int = 0,
    n_workers: int = 0,
    flat: bool = False,
):
    """JAX-traceable ODE solve via CVODE + CVODES forward sensitivities.

    This function wraps BNGsim's C++ CVODE integrator as a JAX primitive.
    The forward pass runs CVODE; the JVP rule runs CVODES to compute
    exact forward sensitivities, enabling ``jax.grad``,
    ``jax.value_and_grad``, and ``jax.jacfwd`` to differentiate through
    the ODE solve.

    Parameters
    ----------
    model : bngsim.Model
        The model to simulate. Must be an ODE-compatible model
        (loaded from ``.net``, Antimony, or SBML).
        **Not differentiated** — passed through unchanged.
    params : jnp.ndarray
        Parameter values. **This is the differentiable argument.**

        - When ``flat=False`` (default): shape ``(n_primary_params,)``,
          ordered to match ``model.primary_param_names``. Derived
          ``ConstantExpression`` parameters (e.g., BNG2.pl-emitted
          ``_rateLaw{N} = chi*kon``) are recomputed automatically from
          the primaries each call, so ``jax.grad`` returns gradients
          with respect to the primary parameters with the chain rule
          through derived expressions correctly applied.
        - When ``flat=True``: shape ``(n_params,)``, ordered to match
          ``model.param_names``. Every parameter (primary and derived)
          is treated as an independent coordinate. Use this only if you
          really want to vary derived parameters independently of their
          defining expression — most users do not.
    t_span : tuple[float, float]
        ``(t_start, t_end)`` time interval. Not differentiated.
    n_points : int
        Number of output time points. Not differentiated.
    rtol : float
        Relative tolerance for CVODE. Default ``1e-8``.
    atol : float
        Absolute tolerance for CVODE. Default ``1e-8``.
    max_steps : int
        Maximum internal solver steps per output point. Default ``10000``.
    chunk_size : int
        When > 0, split sensitivity computation into parallel chunks
        of this size (number of parameters per CVODES job). Each chunk
        runs on a separate cloned model. Benchmarks on large models found
        ``chunk_size=2`` to be a good default: 2-parameter CVODES jobs add
        only ~1.2× overhead.
        Default ``0`` means all parameters in a single CVODES call.
    n_workers : int
        Number of parallel threads for chunked sensitivity. Each thread
        runs an independent CVODES instance (GIL released during C++).
        Default ``0`` means ``min(n_chunks, os.cpu_count())``.
        Only used when ``chunk_size > 0``.
    flat : bool
        If ``True``, use the legacy flat-vector semantics where every
        parameter (primary *and* derived) is an independent input. If
        ``False`` (default), differentiate over primary parameters only;
        derived parameters are propagated via the model's set_param
        chain so gradients reflect the chain rule. See ``params``.

    Returns
    -------
    jnp.ndarray, shape ``(n_points, n_species)``
        Species concentrations at each output time point.

    Raises
    ------
    ImportError
        If JAX is not installed.
    ValueError
        If ``params`` length doesn't match ``model.n_parameters``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from bngsim.jax import differentiable_solve
    >>>
    >>> model = bngsim.Model.from_net("model.net")
    >>> p0 = jnp.array(
    ...     [model.get_param(n) for n in model.primary_param_names]
    ... )
    >>>
    >>> # Forward solve (no differentiation)
    >>> Y = differentiable_solve(model, p0, (0, 100), 101)
    >>>
    >>> # Gradient with parallel chunked sensitivity (40 params, 8 cores)
    >>> def loss(p):
    ...     Y = differentiable_solve(
    ...         model, p, (0, 100), 101,
    ...         chunk_size=2, n_workers=8,
    ...     )
    ...     return jnp.sum((Y - data) ** 2)
    >>> grad = jax.grad(loss)(p0)
    >>>
    >>> # Value and gradient simultaneously
    >>> val, grad = jax.value_and_grad(loss)(p0)
    >>>
    >>> # Full Jacobian (sensitivity matrix)
    >>> def solve_flat(p):
    ...     return differentiable_solve(model, p, (0, 100), 101).ravel()
    >>> J = jax.jacfwd(solve_flat)(p0)

    Notes
    -----
    **How it works**: The function is registered with ``jax.custom_jvp``.
    When JAX needs the Jacobian-vector product (JVP) for differentiation,
    it calls the JVP rule which runs CVODES forward sensitivity analysis.
    CVODES computes the primal solution *and* sensitivities simultaneously
    in a single solve — no double computation.

    **Parallel chunking**: For models with many parameters (Np > 10),
    use ``chunk_size=2, n_workers=N`` to split sensitivities into
    ⌈Np/2⌉ parallel CVODES jobs. Each job adds only ~1.2× overhead,
    so with enough cores the full gradient costs ~1.2× a plain ODE solve.
    Without chunking, CVODES runs all Np sensitivities serially in one
    call (O(Np) overhead per step).

    **Performance**: The overhead is ~1.2× a plain ODE solve for large
    models. For stiff systems this can be dramatically faster than Diffrax
    because CVODE runs in compiled C++, not Python-traced XLA.

    **Thread safety**: Each call clones the model internally, so
    concurrent ``jax.grad`` calls are safe.
    """
    _check_jax()
    import jax.numpy as jnp

    # Decide which parameter names this call is differentiating over.
    # ``flat=True`` (legacy) treats every parameter as independent;
    # ``flat=False`` (default) takes only primary parameters and lets
    # ``set_param`` propagate them to derived ConstantExpression
    # parameters (e.g., ``_rateLaw{N} = chi*kon`` from BNG2.pl).
    diff_param_names = tuple(model.param_names) if flat else tuple(model.primary_param_names)

    params = jnp.asarray(params, dtype=jnp.float64)
    if int(params.shape[0]) != len(diff_param_names):
        raise ValueError(
            f"params length ({int(params.shape[0])}) doesn't match "
            f"the {'flat parameter set' if flat else 'primary parameter set'} "
            f"({len(diff_param_names)}). "
            f"Expected order: {list(diff_param_names)}"
        )

    # Package solver options + parameter selection as a hashable tuple
    # (non-diff arg). Tuple position is consumed positionally in the
    # internal helpers below.
    opts = (
        float(rtol),
        float(atol),
        int(max_steps),
        int(chunk_size),
        int(n_workers),
        diff_param_names,
    )

    return _solve_core(model, params, t_span, n_points, opts)


# ─── JAX custom_jvp primitive ───────────────────────────────────────────


def _make_solve_core():
    """Factory to create the custom_jvp-wrapped solve function.

    We use a factory so that the import of JAX happens lazily
    (only when differentiable_solve is actually called).
    """
    import jax
    import jax.numpy as jnp

    @functools.partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3, 4))
    def solve_core(model, params, t_span, n_points, opts):
        """Run CVODE ODE solve (opaque to JAX tracer).

        Only ``params`` participates in differentiation.
        """
        Y_np = _run_primal(model, params, t_span, n_points, opts)
        return jnp.array(Y_np, dtype=jnp.float64)

    @solve_core.defjvp
    def solve_core_jvp(model, t_span, n_points, opts, primals, tangents):
        """JVP rule: run CVODES once for primal + sensitivities.

        CVODES computes Y and dY/dp simultaneously (single solve).
        The tangent is contracted: dY = einsum('tsp,p->ts', sens, dp).
        """
        (params,) = primals
        (dp,) = tangents

        # Run CVODES once — returns (Y, sensitivity_tensor)
        Y_np, sens_np = _run_with_sensitivity(model, params, t_span, n_points, opts)

        Y = jnp.array(Y_np, dtype=jnp.float64)
        sens = jnp.array(sens_np, dtype=jnp.float64)

        # Contract: dY[t, s] = sum_p sens[t, s, p] * dp[p]
        dY = jnp.einsum("tsp,p->ts", sens, dp)

        return Y, dY

    return solve_core


# Lazy singleton — created on first call
_SOLVE_CORE = None


def _solve_core(model, params, t_span, n_points, opts):
    """Dispatch to the lazily-created custom_jvp function."""
    global _SOLVE_CORE
    if _SOLVE_CORE is None:
        _SOLVE_CORE = _make_solve_core()
    return _SOLVE_CORE(model, params, t_span, n_points, opts)


# ─── Internal helpers (pure Python, no JAX tracing) ─────────────────────


def _run_primal(model, params_jnp, t_span, n_points, opts):
    """Run a plain CVODE solve (no sensitivities).

    Parameters
    ----------
    model : bngsim.Model
        The model (will be cloned).
    params_jnp : jnp.ndarray
        Parameter values, ordered to match ``opts[5]`` (the
        differentiable parameter name list selected by
        ``differentiable_solve``).
    t_span : tuple[float, float]
        Time span.
    n_points : int
        Number of output points.
    opts : tuple
        ``(rtol, atol, max_steps, chunk_size, n_workers, diff_param_names)``.

    Returns
    -------
    np.ndarray, shape (n_points, n_species)
        Species concentrations.
    """
    from bngsim._bngsim_core import CvodeSimulator, SolverOptions, TimeSpec

    params_np = np.asarray(params_jnp, dtype=np.float64)
    diff_param_names = opts[5]

    # Clone for thread safety
    clone = model.clone()

    # Set the requested parameters in order. ``set_param`` re-evaluates
    # remaining ``is_expression`` parameters so derived parameters track
    # primaries automatically when ``flat=False``.
    for i, name in enumerate(diff_param_names):
        clone._core.set_param(name, float(params_np[i]))
    clone.reset()

    # Set up solver
    sim = CvodeSimulator(clone._core)
    ts = TimeSpec()
    ts.t_start = t_span[0]
    ts.t_end = t_span[1]
    ts.n_points = n_points

    rtol, atol, max_steps = opts[:3]
    solver_opts = SolverOptions()
    solver_opts.rtol = rtol
    solver_opts.atol = atol
    solver_opts.max_steps = max_steps

    core_result = sim.run(ts, solver_opts)
    return np.array(core_result.species_data, dtype=np.float64)


def _run_with_sensitivity(model, params_jnp, t_span, n_points, opts):
    """Run CVODES with all-parameter forward sensitivities.

    Returns both the primal solution and the full sensitivity tensor.
    When ``chunk_size > 0`` in opts, uses parallel chunked sensitivity
    via ``Simulator.compute_all_sensitivities()``, which splits Np
    parameters into ⌈Np/chunk_size⌉ parallel CVODES jobs.
    Otherwise, runs a single CVODES call with all parameters.

    Parameters
    ----------
    model : bngsim.Model
        The model (will be cloned internally).
    params_jnp : jnp.ndarray
        Parameter values.
    t_span : tuple[float, float]
        Time span.
    n_points : int
        Number of output points.
    opts : tuple
        (rtol, atol, max_steps, chunk_size, n_workers).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Y, sens) where Y is (n_points, n_species) and
        sens is (n_points, n_species, n_params).
    """
    params_np = np.asarray(params_jnp, dtype=np.float64)
    rtol, atol, max_steps = opts[:3]
    chunk_size = opts[3] if len(opts) > 3 else 0
    n_workers = opts[4] if len(opts) > 4 else 0
    diff_param_names = opts[5]

    if chunk_size > 0:
        # ── Parallel chunked path ──
        # Uses Simulator.compute_all_sensitivities() which handles
        # cloning, chunking, ThreadPoolExecutor, and stitching.
        from bngsim._simulator import Simulator

        # Create a temporary model with the JAX params applied. We feed
        # primaries in order via set_param so derived parameters track.
        clone = model.clone()
        for i, name in enumerate(diff_param_names):
            clone._core.set_param(name, float(params_np[i]))
        clone.reset()

        sim = Simulator(clone, method="ode")

        # Compute actual n_workers if not specified
        effective_workers = n_workers if n_workers > 0 else None

        result = sim.compute_all_sensitivities(
            t_span=t_span,
            n_points=n_points,
            params=list(diff_param_names),
            chunk_size=chunk_size,
            n_workers=effective_workers,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )

        Y = np.asarray(result.species, dtype=np.float64)
        sens = np.asarray(result.sensitivities, dtype=np.float64)
        return Y, sens

    # ── Single CVODES call (all selected params at once) ──
    from bngsim._bngsim_core import (
        CvodeSimulator,
        SolverOptions,
        TimeSpec,
    )

    clone = model.clone()
    for i, name in enumerate(diff_param_names):
        clone._core.set_param(name, float(params_np[i]))
    clone.reset()

    sim = CvodeSimulator(clone._core)
    ts = TimeSpec()
    ts.t_start = t_span[0]
    ts.t_end = t_span[1]
    ts.n_points = n_points

    solver_opts = SolverOptions()
    solver_opts.rtol = rtol
    solver_opts.atol = atol
    solver_opts.max_steps = max_steps
    solver_opts.set_sensitivity_params(list(diff_param_names))

    core_result = sim.run(ts, solver_opts)

    Y = np.array(core_result.species_data, dtype=np.float64)
    sens = np.array(core_result.sensitivity_data, dtype=np.float64)

    return Y, sens
