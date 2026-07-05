"""bngsim._diffrax_solver — Diffrax ODE solver (pure JAX, no CVODE).

Runs the entire ODE solve in JAX/XLA — RHS, Jacobian (AD), and the
adaptive stepper are all JIT-compiled into a single fused computation.
No Python↔C++ boundary crossings during the solve.

Uses Kvaerno5 (5th-order implicit Runge-Kutta, L-stable) for stiff
biochemical systems, with PID step size control.

Optional dependency: ``pip install diffrax``.
"""

from __future__ import annotations

import logging

import numpy as np

from bngsim._jax_rhs import (
    _parse_net_file,
    generate_jax_rhs,
    jax_available,
)

logger = logging.getLogger("bngsim")

# ─── Availability check ────────────────────────────────────────────

_DIFFRAX_AVAILABLE: bool | None = None


def diffrax_available() -> bool:
    """Check if diffrax is importable (cached)."""
    global _DIFFRAX_AVAILABLE
    if _DIFFRAX_AVAILABLE is None:
        if not jax_available():
            _DIFFRAX_AVAILABLE = False
        else:
            try:
                import diffrax  # noqa: F401

                _DIFFRAX_AVAILABLE = True
            except ImportError:
                _DIFFRAX_AVAILABLE = False
    return _DIFFRAX_AVAILABLE


# ─── Diffrax solver ─────────────────────────────────────────────────


def run_diffrax(
    net_path: str,
    param_dict: dict[str, float],
    t_start: float = 0.0,
    t_end: float = 100.0,
    n_points: int = 101,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 100000,
) -> dict:
    """Run an ODE simulation using Diffrax (pure JAX).

    Parameters
    ----------
    net_path : str
        Path to the .net file.
    param_dict : dict[str, float]
        Parameter name → value mapping (all params).
    t_start, t_end : float
        Time interval.
    n_points : int
        Number of output time points.
    rtol, atol : float
        Solver tolerances.
    max_steps : int
        Maximum internal solver steps.

    Returns
    -------
    dict
        Keys: 'time' (n_points,), 'species' (n_points, n_sp),
        'species_names' list[str], 'n_steps' int.
    """
    if not diffrax_available():
        raise ImportError(
            "Diffrax is required for method='diffrax'. Install with: pip install diffrax"
        )

    import diffrax
    import jax
    import jax.numpy as jnp

    # Parse model and build JAX RHS
    model = _parse_net_file(net_path)
    rhs = generate_jax_rhs(net_path)

    # Build parameter array in .net file order
    param_names_ordered = [name for _, name, _, _ in model["parameters"]]
    params = jnp.array(
        [param_dict[n] for n in param_names_ordered],
        dtype=jnp.float64,
    )

    # Initial species concentrations
    y0_list = []
    for _, _name, conc_str, _is_fixed in model["species"]:
        try:
            val = float(conc_str)
        except (ValueError, TypeError):
            # Expression — look up in param_dict
            val = param_dict.get(conc_str, 0.0)
        y0_list.append(val)
    y0 = jnp.array(y0_list, dtype=jnp.float64)

    # Species names
    sp_names = [name for _, name, _, _ in model["species"]]

    # Output times
    ts = jnp.linspace(t_start, t_end, n_points)

    # Diffrax RHS adapter: f(t, y, args) where args = params
    def diffrax_rhs(t, y, args):
        return rhs(y, t, args)

    # Build solver
    term = diffrax.ODETerm(diffrax_rhs)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # JIT-compile and run
    @jax.jit
    def _solve(y0, params):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_start,
            t1=t_end,
            dt0=min((t_end - t_start) / 100.0, 0.01),
            y0=y0,
            args=params,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return sol.ys, sol.stats["num_steps"]

    species_out, n_steps = _solve(y0, params)

    return {
        "time": np.asarray(ts),
        "species": np.asarray(species_out),
        "species_names": sp_names,
        "n_steps": int(n_steps),
    }
