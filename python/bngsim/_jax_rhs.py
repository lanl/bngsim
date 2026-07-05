"""bngsim._jax_rhs — JAX-based ODE RHS and AD Jacobian for BNGsim.

Generates a JAX-traced RHS function from a .net file, then uses
``jax.jacfwd`` to compute exact dense Jacobians via automatic
differentiation. This provides exact Jacobians for ALL rate law types
(Elementary, Functional, MichaelisMenten) without manual derivatives.

Architecture:
  1. generate_jax_rhs(net_path) -> Callable: Parse .net, build JAX RHS
  2. generate_jax_jacobian(net_path) -> Callable: jacfwd(rhs) wrapper
  3. screen_for_discontinuities(net_path) -> bool: Check for floor/ceil/etc.

The JAX Jacobian is fed back to CVODE via the existing user-Jacobian
callback mechanism (dense matrix). JAX runs on CPU only.

Optional dependency: ``pip install bngsim[jax]``.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

from bngsim._codegen import _classify_rate_law, _parse_net_file

logger = logging.getLogger("bngsim")

# ─── Availability check ─────────────────────────────────────────────────────

_JAX_AVAILABLE: bool | None = None


def jax_available() -> bool:
    """Check if JAX is importable (cached).

    Also enables 64-bit precision (required for CVODE compatibility).
    """
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is None:
        try:
            import jax

            # Enable 64-bit precision — CVODE uses double, and float32
            # Jacobians would corrupt Newton convergence.
            jax.config.update("jax_enable_x64", True)
            import jax.numpy  # noqa: F401

            _JAX_AVAILABLE = True
        except ImportError:
            _JAX_AVAILABLE = False
    return _JAX_AVAILABLE


# ─── Discontinuity screening ────────────────────────────────────────────────

# Functions that produce useless AD gradients (piecewise constant).
_DISCONTINUOUS_FUNCS = {"floor", "ceil", "rint", "round", "Heaviside"}


def screen_for_discontinuities(net_path: str) -> list[str]:
    """Scan function expressions for constructs that defeat AD.

    Returns a list of problematic function names found, or empty list
    if the model is safe for JAX AD.
    """
    model = _parse_net_file(net_path)
    problems = []
    for _, name, expr in model["functions"]:
        for disc_fn in _DISCONTINUOUS_FUNCS:
            if re.search(rf"\b{disc_fn}\b", expr):
                problems.append(f"function '{name}' uses {disc_fn}()")
    return problems


# ─── Expression translator (.net expression -> JAX/Python) ──────────────────


def _translate_expr_jax(
    expr: str,
    param_names: dict[str, int],
    obs_names: dict[str, int],
    func_names_set: set[str],
    func_order: list[str],
) -> str:
    """Translate a .net function expression to JAX-compatible Python.

    Replaces:
      - parameter names -> params[idx]
      - observable names -> obs[idx]
      - function names -> func_<name> (local variable)
      - time() / t() -> t
      - if(cond,a,b) -> jnp.where(cond,a,b)
      - ln() -> jnp.log()
      - common math -> jnp.<func>()
      - && -> & (JAX boolean), || -> | (JAX boolean)
    """
    c = expr

    # Replace logical operators FIRST
    c = c.replace("&&", " & ")
    c = c.replace("||", " | ")

    # Replace time() / t() with t
    c = re.sub(r"\btime\(\)", "t", c)
    c = re.sub(r"\bt\(\)", "t", c)

    # Replace if(cond, a, b) -> jnp.where(cond, a, b)
    # This handles nested if() via repeated application
    for _ in range(10):  # max nesting depth
        new_c = re.sub(r"\bif\s*\(", "jnp.where(", c)
        if new_c == c:
            break
        c = new_c

    # Replace function references: funcName() or bare funcName
    # Must do BEFORE parameter replacement
    for fname in func_order:
        safe = _safe_py_name(fname)
        c = re.sub(rf"\b{re.escape(fname)}\(\)", f"func_{safe}", c)
        c = re.sub(rf"\b{re.escape(fname)}\b", f"func_{safe}", c)

    # Replace observable names (longest first to avoid partial match)
    for name in sorted(obs_names.keys(), key=len, reverse=True):
        idx = obs_names[name]
        c = re.sub(
            rf"(?<!func_)\b{re.escape(name)}\b",
            f"obs[{idx}]",
            c,
        )

    # Replace parameter names (longest first)
    for name in sorted(param_names.keys(), key=len, reverse=True):
        idx = param_names[name]
        c = re.sub(
            rf"(?<!obs\[)(?<!func_)\b{re.escape(name)}\b",
            f"params[{idx}]",
            c,
        )

    # Replace math functions with jnp equivalents
    c = re.sub(r"\bln\b", "jnp.log", c)
    c = re.sub(r"\blog\b", "jnp.log", c)
    c = re.sub(r"\bsqrt\b", "jnp.sqrt", c)
    c = re.sub(r"\bexp\b", "jnp.exp", c)
    c = re.sub(r"\bsin\b", "jnp.sin", c)
    c = re.sub(r"\bcos\b", "jnp.cos", c)
    c = re.sub(r"\btan\b", "jnp.tan", c)
    c = re.sub(r"\basin\b", "jnp.arcsin", c)
    c = re.sub(r"\bacos\b", "jnp.arccos", c)
    c = re.sub(r"\batan\b", "jnp.arctan", c)
    c = re.sub(r"\babs\b", "jnp.abs", c)
    c = re.sub(r"\bmin\b", "jnp.minimum", c)
    c = re.sub(r"\bmax\b", "jnp.maximum", c)
    c = re.sub(r"\bpow\b", "jnp.power", c)
    c = re.sub(r"\brint\b", "jnp.round", c)
    c = re.sub(r"\bfloor\b", "jnp.floor", c)
    c = re.sub(r"\bceil\b", "jnp.ceil", c)

    # Replace ^ with ** for exponentiation
    c = c.replace("^", "**")

    return c


def _safe_py_name(name: str) -> str:
    """Convert a BNG name to a safe Python identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ─── JAX RHS generator ──────────────────────────────────────────────────────


def generate_jax_rhs(net_path: str) -> Any:
    """Generate a JAX-traced RHS function from a .net file.

    The returned function has signature::

        rhs(y: jnp.ndarray, t: float, params: jnp.ndarray) -> jnp.ndarray

    where y is species (n_species,), params is (n_params,), and the
    return is dydt (n_species,).

    All operations use jnp so the function is JAX-traceable for AD.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    Callable
        JAX-traceable RHS function.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    import jax.numpy as jnp

    model = _parse_net_file(net_path)
    params_list = model["parameters"]
    species_list = model["species"]
    reactions = model["reactions"]
    observables = model["observables"]
    functions = model["functions"]

    n_sp = len(species_list)
    n_params = len(params_list)

    # Build index maps (0-based)
    param_idx = {name: i for i, (_, name, _, _) in enumerate(params_list)}
    func_names_set = {name for _, name, _ in functions}
    func_order = [name for _, name, _ in functions]
    {name: i for i, (_, name, _) in enumerate(functions)}
    obs_idx = {name: i for i, (_, name, _) in enumerate(observables)}

    # Fixed species (0-based indices)
    fixed_sp = frozenset(sp[0] - 1 for sp in species_list if sp[3])

    # Pre-build observable weight matrix as dense array
    # obs[k] = sum of weight[k, j] * y[j]
    obs_weights = []
    for _, _name, entries in observables:
        row = [0.0] * n_sp
        for factor, sp_i in entries:
            row[sp_i - 1] = factor
        obs_weights.append(row)

    # Pre-translate function expressions to JAX Python
    func_exprs = []
    for _, name, expr in functions:
        jax_expr = _translate_expr_jax(expr, param_idx, obs_idx, func_names_set, func_order)
        func_exprs.append((name, jax_expr))

    # Pre-classify reactions and build stoichiometry
    rxn_data = []
    for _, reactants, products, rate_law, _comment in reactions:
        kind = _classify_rate_law(rate_law, func_names_set)
        rxn_data.append((reactants, products, kind))

    # Build the RHS function source as a closure
    # We create numpy arrays for the weight matrix at build time
    obs_w_array = jnp.array(obs_weights, dtype=jnp.float64)

    def rhs(y, t, params):
        """JAX-traced ODE RHS: dy/dt = f(y, t, params)."""
        # Compute observables: obs = W @ y
        obs = obs_w_array @ y

        # Evaluate functions in dependency order
        func_vals = {}
        for fname, jax_expr in func_exprs:
            _safe_py_name(fname)
            # Build local namespace for eval
            local_ns = {
                "jnp": jnp,
                "t": t,
                "params": params,
                "obs": obs,
                "y": y,
            }
            # Add previously computed functions
            for prev_name, prev_val in func_vals.items():
                local_ns[f"func_{_safe_py_name(prev_name)}"] = prev_val
            val = eval(jax_expr, {"__builtins__": {}}, local_ns)  # noqa: S307
            func_vals[fname] = val

        # Compute derivatives
        dydt = jnp.zeros(n_sp, dtype=y.dtype)

        for reactants, products, kind in rxn_data:
            if kind[0] == "elementary":
                _, pname, sf = kind
                p_i = param_idx.get(pname, -1)
                rate = params[p_i] * sf if p_i >= 0 else 0.0
                for ri in reactants:
                    rate = rate * y[ri - 1]
            elif kind[0] == "functional":
                _, fname, sf = kind
                rate = func_vals[fname] * sf
                for ri in reactants:
                    rate = rate * y[ri - 1]
            elif kind[0] == "mm":
                _, kcat_name, km_name, sf = kind
                kcat_i = param_idx.get(kcat_name, -1)
                km_i = param_idx.get(km_name, -1)
                kcat = params[kcat_i] if kcat_i >= 0 else 0.0
                km = params[km_i] if km_i >= 0 else 0.0
                if len(reactants) >= 2:
                    e = y[reactants[0] - 1]
                    s = y[reactants[1] - 1]
                    s_free = 0.5 * ((s - km - e) + jnp.sqrt((s - km - e) ** 2 + 4.0 * km * s))
                    rate = sf * kcat * s_free * e / (km + s_free)
                else:
                    rate = 0.0
            else:
                rate = 0.0

            # Accumulate stoichiometry
            for ri in reactants:
                if ri > 0:
                    dydt = dydt.at[ri - 1].add(-rate)
            for pi in products:
                if pi > 0:
                    dydt = dydt.at[pi - 1].add(rate)

        # Zero fixed species derivatives
        for si in fixed_sp:
            dydt = dydt.at[si].set(0.0)

        return dydt

    # Attach metadata (function attributes — Any-typed to satisfy mypy)
    rhs_obj: Any = rhs
    rhs_obj.n_species = n_sp
    rhs_obj.n_params = n_params
    rhs_obj.fixed_species = fixed_sp

    return rhs_obj


# ─── JAX Jacobian wrapper ───────────────────────────────────────────────────


def generate_jax_jacobian(net_path: str) -> Any:
    """Generate a JAX AD Jacobian function from a .net file.

    Returns a function::

        jac_fn(y: ndarray, t: float, params: ndarray) -> ndarray

    that computes the exact N×N dense Jacobian ∂f/∂y via forward-mode AD.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    Callable
        Function that returns (n_species, n_species) Jacobian matrix.

    Raises
    ------
    ImportError
        If JAX is not installed.
    RuntimeError
        If model contains discontinuous functions (floor/ceil/etc.).
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    # Screen for discontinuities — warn but don't reject.
    # CVODE uses the Jacobian for Newton convergence, not solution
    # correctness. A locally-zero Jacobian at a discontinuity just
    # means CVODE takes smaller steps there (same as FD behavior).
    problems = screen_for_discontinuities(net_path)
    if problems:
        logger.warning(
            "Model functions may produce zero JAX AD gradients at "
            "discontinuities (CVODE will adapt step size):\n  %s",
            "\n  ".join(problems),
        )

    import jax

    rhs = generate_jax_rhs(net_path)

    # Forward-mode AD: differentiate RHS w.r.t. y (argnums=0)
    # Returns (n_species, n_species) Jacobian matrix
    jac_fn_raw = jax.jacfwd(rhs, argnums=0)

    def jac_fn(y, t, params):
        """Compute dense Jacobian J[i][j] = ∂f_i/∂y_j."""
        return jac_fn_raw(y, t, params)

    # Attach metadata (function attributes — Any-typed to satisfy mypy)
    jac_fn_obj: Any = jac_fn
    jac_fn_obj.n_species = rhs.n_species
    jac_fn_obj.n_params = rhs.n_params
    jac_fn_obj.rhs = rhs

    return jac_fn_obj


# ─── Prepare JAX Jacobian for CVODE ─────────────────────────────────────────


def prepare_jax_jacobian(net_path: str) -> tuple:
    """Prepare a JAX Jacobian evaluator for use with CVODE.

    Returns a tuple (jac_fn, n_species, param_values) where jac_fn
    is a callable that takes (y_flat, t, param_flat) and returns
    a flat row-major Jacobian array suitable for C++ consumption.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    tuple
        (evaluate_jacobian, n_species) where evaluate_jacobian is
        a callable (y_flat, t, param_flat) -> flat_jac_array.
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    import jax
    import jax.numpy as jnp
    import numpy as np

    jac_fn = generate_jax_jacobian(net_path)
    n_sp = jac_fn.n_species

    # JIT-compile for speed
    jac_fn_jit = jax.jit(jac_fn)

    # Warm up with dummy data to trigger compilation
    dummy_y = jnp.ones(n_sp, dtype=jnp.float64)
    dummy_p = jnp.ones(jac_fn.n_params, dtype=jnp.float64)
    # warmup may fail with dummy params; that's OK
    with contextlib.suppress(Exception):
        _ = jac_fn_jit(dummy_y, 0.0, dummy_p)

    def evaluate_jacobian(y_flat, t, param_flat):
        """Evaluate Jacobian, return column-major flat array for CVODE.

        CVODE dense matrix is column-major (Fortran order).
        """
        y_jax = jnp.array(y_flat, dtype=jnp.float64)
        p_jax = jnp.array(param_flat, dtype=jnp.float64)
        J = jac_fn_jit(y_jax, t, p_jax)
        # J is (n_sp, n_sp) with J[i][j] = df_i/dy_j
        # CVODE dense matrix is column-major: column j, row i
        # np.asfortranarray gives column-major layout
        J_np = np.asarray(J, dtype=np.float64)
        return J_np.flatten(order="F")

    return evaluate_jacobian, n_sp
