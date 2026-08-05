"""bngsim.Simulator — Unified simulation interface.

``bngsim.Simulator(model, method="ode")`` exposes ODE, SSA, PSA, and
network-free simulation through a single Python interface.

Supported features include batch execution, stop conditions, logging,
interactive stepping for stateful solvers, and forward sensitivity analysis.

Network-free method normalization uses canonical algorithm tokens:
- ``nf_reject``: rejection/null-event handling (NFsim-style, Yang et al.)
- ``nf_exact``: exact non-local network-free token (RuleMonkey)
- ``nf_fixed``: retired fixed-step network-free token
- ``nf``: umbrella token, currently routes to ``nf_reject``
Legacy aliases (``nfsim``, ``rulemonkey``, ``rm``, ``dynstoc``, ``ds``)
are accepted and normalized to canonical tokens before dispatch.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import threading
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from bngsim._codegen import _codegen_jit_backend, last_codegen_cache_hit, last_codegen_sec
from bngsim._exceptions import (
    DenseSolverFallbackWarning,
    ModelError,
    SimulationError,
    SimulationTimeout,
    SsaBoundaryWarning,
    SsaValidationError,
    StopConditionMet,
)
from bngsim._model import Model
from bngsim._result import Result, _as_selector_list, _resolve_output_selector
from bngsim._seed import _DEFAULT_EVENT_SEED, _resolve_seed
from bngsim._ssa_validation import validate_for_ssa

logger = logging.getLogger("bngsim")

try:
    from bngsim._bngsim_core import HAS_RULEMONKEY as _HAS_RULEMONKEY
except (ImportError, AttributeError):
    _HAS_RULEMONKEY = False

# SuiteSparse/KLU availability (GH #209). Read straight from the C++ extension so
# this is independent of bngsim package import order. When False, the ODE backend
# has only the dense linear solver.
try:
    from bngsim._bngsim_core import HAS_KLU as _HAS_KLU
except (ImportError, AttributeError):
    _HAS_KLU = False

# Species-count threshold above which a dense-only-because-no-KLU ODE run emits
# the one-time DenseSolverFallbackWarning. At ~2000 species the sparse KLU path
# starts to matter; below it the dense solver is fine and the notice would be
# noise. Matches the trigger suggested in GH #209.
_DENSE_FALLBACK_WARN_NSPECIES = 2000

# Process-wide one-shot guard for the dense-fallback notice, so a run_batch over
# many large models (or repeated run() calls) warns at most once, not per run.
_dense_fallback_warned = False

# ─── on_point initial-condition sensitivity probe (issue #111) ────────
# A parameter_scan on_point hook assigns the initial conditions its point starts
# from, so the point's ∂x(0)/∂θ for those species is whatever the hook's own
# arithmetic implies. bngsim measures it by calling the hook at perturbed inputs
# (Simulator._probe_on_point_ic_sens). These set the perturbation and how much
# the two step sizes may disagree before the hook is declared non-differentiable
# there. EPS is relative (to |θ| for the parameter path, to ‖x‖∞ for the state
# path), so a 1e-9 rate constant is probed at 1e-9 scale, not an absolute floor
# (the mistake issue #76 reports on the steady-state FD). A dose formula is
# almost always linear in each parameter, where a central difference is exact to
# roundoff and the two step sizes agree to ~1e-10 — TOL is loose enough for a
# genuinely nonlinear-but-smooth formula (O(h²) truncation) and far tighter than
# the O(1/h) blow-up a jump produces.
_IC_SENS_PROBE_EPS = 1e-6
_IC_SENS_PROBE_COARSE = 4.0
_IC_SENS_PROBE_TOL = 1e-4


# ─── Network-free method normalization ───────────────────────────────

# Canonical method tokens (algorithm-based, not tool-branded):
#   nf_reject — rejection/null-event handling (NFsim-style, Yang et al.)
#   nf_exact  — exact non-local network-free token (RuleMonkey)
#   nf_fixed  — retired fixed-step network-free token
#   nf        — umbrella token, routes to nf_reject (current default)
#
# Legacy/compatibility aliases map to canonical tokens:
_NF_METHOD_ALIASES: dict[str, str] = {
    # Umbrella token → current default implementation
    "nf": "nf_reject",
    # Canonical (identity)
    "nf_reject": "nf_reject",
    "nf_exact": "nf_exact",
    "nf_fixed": "nf_fixed",
    # Legacy tool-branded aliases
    "nfsim": "nf_reject",
    "rulemonkey": "nf_exact",
    "rm": "nf_exact",
    "dynstoc": "nf_fixed",
    "ds": "nf_fixed",
}

_available_nf_methods = {"nf_reject"}
_unavailable_nf_methods: dict[str, str] = {}

if _HAS_RULEMONKEY:
    _available_nf_methods.add("nf_exact")
else:
    _unavailable_nf_methods["nf_exact"] = (
        "method='nf_exact' (exact non-local network-free) is recognized "
        "but RuleMonkey is not present in this bngsim install. "
        "The vendored RuleMonkey backend at third_party/rulemonkey/ is "
        "built by default; this install was either configured with "
        "-DBNGSIM_BUILD_RULEMONKEY=OFF or installed from a wheel that "
        "excludes RuleMonkey."
    )

_unavailable_nf_methods["nf_fixed"] = (
    "method='nf_fixed' (fixed-step network-free) is "
    "recognized but unavailable in this environment. "
    "This experimental backend is not part of the current bngsim release."
)

# Methods with usable dispatch in this runtime.
_AVAILABLE_NF_METHODS: frozenset[str] = frozenset(_available_nf_methods)

# Recognized canonical methods that are unavailable in this runtime.
_UNAVAILABLE_NF_METHODS: dict[str, str] = _unavailable_nf_methods

# The in-process codegen JIT backend selector (BNGSIM_CODEGEN_JIT=mir, GH #78)
# is defined in bngsim._codegen and imported above, so the SBML loader and the
# sensitivity auto-codegen path here share one source of truth.


def normalize_method(requested: str) -> tuple[str, str]:
    """Normalize a user-requested method token to its canonical form.

    Parameters
    ----------
    requested : str
        The method string as provided by the user.

    Returns
    -------
    tuple[str, str]
        ``(canonical, dispatch)`` where *canonical* is the normalized
        algorithm name (e.g. ``"nf_reject"``) and *dispatch* is the
        internal backend key used for simulator creation (e.g.
        ``"nfsim"``).

    Raises
    ------
    ValueError
        If the method token is not recognized at all, or if it maps
        to an unavailable backend.
    """
    lower = requested.strip().lower()

    # Non-NF methods pass through unchanged
    if lower in ("ode", "ssa", "psa"):
        return lower, lower

    # Check NF alias map
    canonical = _NF_METHOD_ALIASES.get(lower)
    if canonical is None:
        # Build helpful error with all known tokens
        all_known = sorted({"ode", "ssa", "psa"} | set(_NF_METHOD_ALIASES.keys()))
        raise ValueError(f"Unknown method '{requested}'. Supported: {all_known}")

    # Check availability
    if canonical in _UNAVAILABLE_NF_METHODS:
        raise ValueError(_UNAVAILABLE_NF_METHODS[canonical])

    assert canonical in _AVAILABLE_NF_METHODS
    dispatch = {
        "nf_reject": "nfsim",
        "nf_exact": "rulemonkey",
    }[canonical]

    return canonical, dispatch


class Simulator:
    """Unified simulation interface for ODE, SSA, PSA, and network-free methods.

    Parameters
    ----------
    model : Model
        The model to simulate.
    method : str
        Simulation method:

        **Deterministic / network-based:**

        - ``"ode"`` — CVODE adaptive BDF integrator (deterministic)
        - ``"ssa"`` — Variant of Gillespie's direct method (exact stochastic)
        - ``"psa"`` — Partial Scaling Algorithm (approximate stochastic).
          Lin, Feng, Hlavacek, J. Chem. Phys. 150, 244101 (2019).
          Requires ``poplevel`` keyword argument.

        **Network-free (canonical tokens):**

        - ``"nf"`` — Network-free simulation (default policy; currently
          routes to ``nf_reject``).
        - ``"nf_reject"`` — Rejection/null-event algorithm (NFsim-style).
          Requires ``xml_path`` keyword argument.
        - ``"nf_exact"`` — Exact non-local network-free algorithm
          (RuleMonkey). Requires ``xml_path`` keyword argument.
        - ``"nf_fixed"`` — Legacy compatibility token.
          Recognized but unavailable in this build.

        **Legacy aliases** (accepted for compatibility):

        - ``"nfsim"`` → ``"nf_reject"``
        - ``"rulemonkey"`` / ``"rm"`` → ``"nf_exact"``
        - ``"dynstoc"`` / ``"ds"`` → unavailable compatibility alias

    poplevel : float, optional
        Critical population size N_c for PSA. Required when
        ``method="psa"``. Must be > 1. Larger values are more
        conservative (less acceleration, less approximation error).
        Typical values: 100–1000.

    connectivity : bool, optional
        Only used for ``method="nf"`` / ``"nf_reject"`` / ``"nfsim"``.
        Controls NFsim's reaction-connectivity optimization at XML
        initialization. ``False`` uses the conservative full membership
        update path; ``True`` enables the inferred dependency-graph path.
        If omitted, the underlying NFsim wrapper default is used
        (currently ``False``).

    nfsim_v1143_compat : bool, optional
        Only used for ``method="nf"`` / ``"nf_reject"`` / ``"nfsim"``.
        When true, preserve NFsim v1.14.3's extra selector draw for
        same-seed trajectory compatibility with the standalone CLI.

    block_same_complex_binding : bool, optional
        Only used for ``method="nf"`` / ``"nf_reject"`` / ``"nfsim"``.
        NFsim ``-bscb``: when True, two reactant patterns in a bimolecular
        rule cannot match molecules in the same complex.
        Default: ``True`` in bngsim — NFsim CLI defaults it off, but bngsim
        defaults it on for correctness on BLBR/aggregation models. Pass
        ``False`` to allow same-complex binding (BNG2.pl ``complex=>1``).
        This governs only the binding policy; complex bookkeeping for
        ``Species``-typed observable counting is enabled automatically when
        the model declares such an observable, independent of this flag.

    traversal_limit : int, optional
        Only used for ``method="nf"`` / ``"nf_reject"`` / ``"nfsim"``.
        NFsim ``-utl N``: universal traversal limit. ``None`` (default)
        lets NFsim auto-compute a suggested limit from the XML.

    codegen : bool, optional
        Only used for ``method="ode"``. When true, use a compiled C RHS.
        Models loaded from BioNetGen ``.net`` files use the ``.net``
        codegen path. SBML, Antimony, and other already-built models use
        model-based codegen. For SBML/Antimony models, pass
        ``codegen=True`` without ``net_path``.

    net_path : str, optional
        BioNetGen ``.net`` path for the ``.net`` codegen path. This is not
        a generic model path and should not point to SBML XML. Models loaded
        with :meth:`Model.from_net` remember their source path, so most
        callers do not need to pass this manually.

    sensitivity_params : list[str], optional
        Parameter names to integrate forward sensitivities for, alongside
        the state ODEs. The result then carries a
        ``(n_times, n_species, n_params)`` ``sensitivities`` tensor whose
        ``(t, i, k)`` entry is ``∂x_i(t) / ∂p_k`` evaluated at the
        baseline parameter values. Only valid for ``method="ode"``.

    sensitivity_ic : list[str], optional
        Species names to integrate forward initial-condition
        sensitivities for. The result carries a
        ``(n_times, n_species, n_ic)`` ``sensitivities_ic`` tensor whose
        ``(t, i, k)`` entry is ``∂x_i(t) / ∂x_k(0)``. Useful when fitting
        IC parameters via chain rule from a Python-side reparameterization
        (e.g., ``model.set_concentration("Epo", 10**theta)``) without
        a corresponding model parameter to differentiate against.
        Requires the codegen sensitivity RHS path; codegen is auto-enabled
        for any sensitivity workflow. Only valid for ``method="ode"``.

    strict_ssa : bool, optional
        Only used for ``method="ssa"`` / ``"psa"``. Default ``True``.

        SBML loader records SSA-compatibility issues at load time
        (e.g. ``reversible_non_mass_action``,
        ``assignment_rule_on_reactant``). When ``True`` the Simulator
        raises :class:`SsaValidationError` on any error-severity issue —
        this is the safe default that prevents fitting workflows from
        silently consuming wrong dynamics on broken-under-SSA constructs.

        Pass ``False`` to downgrade most error-severity issues to
        warnings (logged via :mod:`logging`) and let the Simulator
        construct anyway. This mirrors roadrunner's warn-and-run
        behavior under ``gillespie`` integration. Useful when comparing
        bngsim against roadrunner on the same model, or when the user
        understands that the dynamics under SSA will be approximate
        for these constructs.

        Two issue codes remain non-overridable even with
        ``strict_ssa=False``: ``non_integer_stoichiometry`` (SSA
        requires ±1 fire deltas) and ``fast_reaction`` (no
        fast-equilibrium constraint solver).

    sensitivity_method : {"staggered", "simultaneous"}, optional
        CVODES corrector strategy for the coupled state + sensitivity
        system. Both modes integrate state and *all* sensitivity ODEs
        as one extended ODE in a single CVODES pass; they differ only
        in how each integration step's nonlinear solve is structured:

        - ``"staggered"`` (default, ``CV_STAGGERED``): advance the
          state first, then — with the new state in hand — advance the
          sensitivity ODEs as a separate solve. Two smaller nonlinear
          solves per step instead of one big one. Often more robust
          on stiff or large systems; this is CVODES' / BNGsim's
          default.
        - ``"simultaneous"`` (``CV_SIMULTANEOUS``): solve state and
          all sensitivity variables together as one coupled
          nonlinear system at every step. Often a touch faster per
          step on small / well-conditioned problems; the per-step
          solve is larger so it can struggle on stiff or large
          systems. This is **AMICI's default**, so this is the value
          to use when you want apples-to-apples timing against AMICI.

        CVODES has a third mode (``CV_STAGGERED1``, one parameter at a
        time) that BNGsim does not currently expose.

    force_dense_linear_solver : bool, optional
        Only used for ``method="ode"``. Default ``False``. Force CVODE's
        dense direct linear solver even for large, low-density models that
        would otherwise auto-select sparse KLU. This is orthogonal to
        ``jacobian`` (which selects how the Jacobian is *computed*) — it
        overrides only the linear-solver *kind*. Intended for benchmarking
        the dense path against KLU on the same model; it has no effect in a
        build compiled without KLU (already always dense).

        Since issue #128 this reaches :meth:`steady_state` and
        :meth:`steady_state_batch` too — specifically their CVODE march, which
        routes by this same rule and reports the outcome as
        :attr:`SteadyStateResult.linear_solver`. The KINSOL polish and the
        ``dY_ss/dp`` solve factor a *reduced* matrix, whose sparsity pattern is
        a different object from the model's, and stay dense either way.

    force_sparse_linear_solver : bool, optional
        Only used for ``method="ode"``. Default ``False``. The mirror image of
        ``force_dense_linear_solver``: force sparse KLU even on a model the auto
        rule would send to the dense solver for being too small (``n_species <
        50``) or too dense (Jacobian density ``>= 10%``). Only those two gates
        are bypassed — KLU still needs a real sparsity pattern and a non-JAX
        Jacobian — so it is likewise a no-op in a build without KLU. Passing
        both force flags raises :class:`ValueError`.

        Intended for measuring the auto-selection rule against its own
        alternative: forced-dense shows what KLU buys on large sparse networks,
        forced-sparse shows KLU's setup and indexing overhead on the small dense
        ones. A model that is *both* too dense to have been graph-colored and
        without a usable analytical Jacobian has no way to fill a sparse matrix
        at all; ``run()`` raises there rather than quietly reverting to dense.

        Applies to :meth:`steady_state` as well (issue #128) — with one
        difference: a model whose Jacobian has *no structural nonzero* is left on
        the dense solver there instead of raising, because the steady-state auto
        rule can route such a model to KLU on its own (density 0 < 10%) and
        f(y) ≡ 0 makes it a steady state that used to solve immediately.

    Examples
    --------
    >>> model = bngsim.Model.from_net("model.net")
    >>> sim = bngsim.Simulator(model, method="ode")
    >>> result = sim.run(t_span=(0, 100), n_points=101)
    >>> result.time.shape
    (101,)

    >>> ssa = bngsim.Simulator(model, method="ssa")
    >>> result = ssa.run(t_span=(0, 100), n_points=101, seed=42)

    >>> psa = bngsim.Simulator(model, method="psa", poplevel=300)
    >>> result = psa.run(t_span=(0, 100), n_points=101, seed=42)

    >>> # Network-free (all equivalent):
    >>> nf1 = bngsim.Simulator(model, method="nf", xml_path="m.xml")
    >>> nf2 = bngsim.Simulator(model, method="nf_reject", xml_path="m.xml")
    >>> nf3 = bngsim.Simulator(model, method="nfsim", xml_path="m.xml")
    >>> rm = bngsim.Simulator(model, method="rm", xml_path="m.xml")
    """

    # All tokens accepted by the constructor (used for documentation;
    # actual validation is done by normalize_method()).
    METHODS = {
        "ode",
        "ssa",
        "psa",
        "nf",
        "nf_reject",
        "nf_exact",
        "nf_fixed",
        "nfsim",
        "rulemonkey",
        "rm",
        "dynstoc",
        "ds",
    }

    __slots__ = (
        "_model",
        "_method",
        "_canonical_method",
        "_requested_method",
        "_sim",
        "_rtol",
        "_atol",
        "_max_steps",
        "_stop_conditions",
        # Interactive simulation state
        "_current_time",
        "_snapshot_stack",
        # NFsim-specific
        "_xml_path",
        # PSA-specific
        "_poplevel",
        # ODE Jacobian strategy
        "_jacobian",
        # GH #176: once the auto analytical Jacobian fails to integrate and the FD
        # retry succeeds, skip the doomed analytical attempt on subsequent runs.
        "_ode_jacobian_fell_back",
        # Issue #127: the same memo for the steady-state march, whose retry is
        # decided in C++ (a failed march is a flag, not an exception).
        "_ss_jacobian_fell_back",
        # Force dense linear solver over auto-selected sparse KLU (benchmarking)
        "_force_dense_linear_solver",
        # ...and the mirror flag, forcing KLU past the size/density gates (GH #29)
        "_force_sparse_linear_solver",
        # Code-generated RHS support
        "_codegen",
        "_codegen_so_path",
        "_codegen_c_source",
        "_net_path",
        # JAX AD Jacobian support
        "_jax_jac_evaluator",
        # CVODES forward sensitivities
        "_sensitivity_params",
        "_sensitivity_ic",
        "_sensitivity_method",
        # Per-species V_c cache for Result.as_roadrunner; lazily filled.
        "_volume_factors_cache",
        # GH #198 — memoized expression output-sensitivity support map; lazily filled.
        "_expr_sens_support_memo",
    )

    def __init__(
        self,
        model: Model,
        method: str = "ode",
        *,
        xml_path: str = "",
        poplevel: float | None = None,
        gml: int | None = None,
        connectivity: bool | None = None,
        nfsim_v1143_compat: bool = False,
        block_same_complex_binding: bool = True,
        traversal_limit: int | None = None,
        jacobian: str = "auto",
        force_dense_linear_solver: bool = False,
        force_sparse_linear_solver: bool = False,
        codegen: bool | None = None,
        net_path: str = "",
        sensitivity_params: list[str] | None = None,
        sensitivity_ic: list[str] | None = None,
        sensitivity_method: str = "staggered",
        strict_ssa: bool = True,
    ) -> None:
        # Normalize the user-facing method token before backend dispatch.
        # normalize_method() validates the token, checks availability,
        # and returns (canonical, dispatch) where dispatch is the
        # internal backend key (e.g. "nfsim").
        canonical, dispatch = normalize_method(method)

        self._model = model
        # GH #198: stash whether expression output sensitivities will be needed,
        # BEFORE the codegen prep below (which runs in this __init__, ahead of the
        # self._sensitivity_* assignments) reads it to decide whether to emit the
        # bngsim_codegen_output_sens evaluator. Its build-time differentiation is
        # expensive on large functional models, so a non-sensitivity run must not
        # pay it. The .so cache key carries this flag (prepare_codegen), so a
        # non-sensitivity .so is never reused for a sensitivity run.
        model._want_output_sens = bool(sensitivity_params or sensitivity_ic)
        self._requested_method = method  # original user token
        self._method = dispatch  # internal dispatch key
        self._canonical_method = canonical
        self._xml_path = xml_path
        self._poplevel: float = 0.0

        # Log normalization when it changes the token.
        if method != dispatch:
            logger.debug(
                "Method normalized: '%s' → canonical='%s', dispatch='%s'",
                method,
                canonical,
                dispatch,
            )

        # Validate PSA-specific options
        if dispatch == "psa":
            if poplevel is None:
                raise ValueError(
                    "method='psa' requires poplevel=N_c (critical "
                    "population size). Typical values: 100–1000. "
                    "See Lin, Feng, Hlavacek, J. Chem. Phys. 150, "
                    "244101 (2019)."
                )
            if poplevel <= 1.0:
                raise ValueError(
                    f"poplevel must be > 1 for PSA. Got {poplevel}. "
                    "For exact stochastic simulation, use method='ssa'."
                )
            self._poplevel = float(poplevel)
        elif poplevel is not None:
            raise ValueError(
                f"poplevel is only valid for method='psa', "
                f"not method='{method}'. Use method='psa' to "
                "enable partial scaling."
            )

        # Create the appropriate C++ simulator based on dispatch key
        # Typed as Any because it's runtime-dispatched: CvodeSimulator,
        # SsaSimulator, or NfsimSimulator depending on `dispatch`.
        self._sim: Any
        if dispatch == "ode":
            from bngsim._bngsim_core import CvodeSimulator

            # GH #113: fast="true" reactions declare a fast-equilibrium
            # constraint bngsim has no solver for. Under SSA this is caught by
            # validate_for_ssa; under ODE the kinetic law would otherwise
            # integrate as an ordinary reaction, silently ignoring the
            # constraint. The loader already recorded a fast_reaction SsaIssue
            # (kept loadable so the SSA validate/override contract is intact),
            # so surface it here. Refuse by default;
            # BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1 restores the silent
            # approximation (cf. the delay/AlgebraicRule load-time gate).
            fast_issues = [
                i for i in getattr(model, "_ssa_issues", None) or [] if i.code == "fast_reaction"
            ]
            if fast_issues and os.environ.get("BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS") != "1":
                locs = ", ".join(i.location for i in fast_issues if i.location)
                raise ModelError(
                    'Model contains fast="true" reaction(s) '
                    f"[{locs}], a fast-equilibrium constraint bngsim cannot "
                    "honor under ODE — the kinetic law would integrate as an "
                    "ordinary reaction, silently ignoring the constraint "
                    "(RoadRunner refuses the same model). To restore the legacy "
                    "silent-approximation behavior, set "
                    "BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1."
                )

            self._sim = CvodeSimulator(model._core)
        elif dispatch in ("ssa", "psa"):
            # SBML-loaded models carry a list of SsaIssue records (populated
            # by _sbml_loader). Run validation BEFORE constructing the
            # C++ simulator: errors abort with SsaValidationError; warnings
            # are logged and execution continues. .net / builder models
            # have an empty list and pass through unchanged.
            #
            # ``strict_ssa=False`` lets callers run SSA on models with
            # known-broken kineticLaw shapes (e.g. reversible_non_mass_action,
            # AR-on-reactant) — the default of True matches bngsim's
            # cautious-at-SSA philosophy; the override mirrors roadrunner's
            # warn-and-run UX for users who understand the limitations and
            # want to do bngsim↔rr comparisons. Two issue codes remain
            # non-overridable because they violate SSA's discrete-fire model
            # at the kernel level: non_integer_stoichiometry (no fractional
            # ±N firing) and fast_reaction (no fast-equilibrium constraint
            # solver between fires).
            ssa_issues = validate_for_ssa(model)
            ssa_errors = [i for i in ssa_issues if i.severity == "error"]
            ssa_warnings = [i for i in ssa_issues if i.severity == "warning"]
            if ssa_errors:
                if strict_ssa:
                    raise SsaValidationError(ssa_issues)
                hard_errors = [
                    i for i in ssa_errors if i.code in SsaValidationError.NON_OVERRIDABLE_CODES
                ]
                if hard_errors:
                    raise SsaValidationError(hard_errors, override_attempted=True)
                for e in ssa_errors:
                    loc = f" [{e.location}]" if e.location else ""
                    logger.warning(
                        "SSA validation (strict_ssa=False, downgraded): %s%s — %s",
                        e.code,
                        loc,
                        e.message,
                    )
            for w in ssa_warnings:
                loc = f" [{w.location}]" if w.location else ""
                logger.warning("SSA validation: %s%s — %s", w.code, loc, w.message)

            # PSA shares SsaSimulator; dispatch happens at run time.
            # Event-with-delay rejection happens C++-side at run() entry.
            from bngsim._bngsim_core import SsaSimulator

            self._sim = SsaSimulator(model._core)

            # GH #190 — for exact SSA, hand the C++ simulator a cc-compiled
            # value-specialized propensity .so so eligible small mass-action
            # models take the RR-parity recompute-all + flat-scan loop by
            # DEFAULT (no MIR). The C++ side makes the final eligibility/size
            # decision and ignores it for PSA / events / functional / large-nr
            # models; this just provides the artifact, cached on disk so an
            # ensemble compiles once. Skipped for PSA (recompute-all needs exact
            # SSA), when codegen=False, when BNGSIM_SSA_NO_CODEGEN is set, or when
            # a BNGSIM_SSA_PROP_{CC,JIT} env override selects an in-process
            # backend instead (then the C++ side compiles the source itself).
            if (
                dispatch == "ssa"
                and codegen is not False
                and not os.environ.get("BNGSIM_SSA_NO_CODEGEN")
                and not os.environ.get("BNGSIM_SSA_PROP_CC")
                and not os.environ.get("BNGSIM_SSA_PROP_JIT")
            ):
                try:
                    from bngsim._codegen import prepare_ssa_propensity_lib

                    _ssa_lib = prepare_ssa_propensity_lib(model)
                    if _ssa_lib:
                        self._sim.set_propensity_library(_ssa_lib)
                        logger.debug("SSA propensity .so ready: %s", _ssa_lib)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("SSA propensity codegen skipped", exc_info=True)
        elif dispatch == "nfsim":
            from bngsim._bngsim_core import HAS_NFSIM

            if not HAS_NFSIM:
                raise RuntimeError(
                    "NFsim support is not present in this bngsim install. "
                    "The vendored NFsim backend at third_party/nfsim/ is "
                    "built by default; this install was either configured "
                    "with -DBNGSIM_BUILD_NFSIM=OFF or installed from a "
                    "wheel that excludes NFsim."
                )
            if not xml_path:
                raise ValueError(
                    f"method='{method}' requires xml_path=... pointing to a BNG XML file."
                )
            from bngsim._bngsim_core import NfsimSimulator

            self._sim = NfsimSimulator(xml_path)
            if gml is not None:
                self._sim.set_molecule_limit(int(gml))
            if connectivity is not None:
                self._sim.set_connectivity(bool(connectivity))
            if nfsim_v1143_compat:
                self._sim.set_nfsim_v1143_compat(True)
            # Always propagate so explicit False reaches C++ (default is True
            # on both sides).
            self._sim.set_block_same_complex_binding(bool(block_same_complex_binding))
            if traversal_limit is not None:
                self._sim.set_traversal_limit(int(traversal_limit))
        elif dispatch == "rulemonkey":
            from bngsim._bngsim_core import HAS_RULEMONKEY

            if not HAS_RULEMONKEY:
                raise RuntimeError(
                    "RuleMonkey support is not present in this bngsim "
                    "install. The vendored RuleMonkey backend at "
                    "third_party/rulemonkey/ is built by default; this "
                    "install was either configured with "
                    "-DBNGSIM_BUILD_RULEMONKEY=OFF or installed from a "
                    "wheel that excludes RuleMonkey."
                )
            if not xml_path:
                raise ValueError(
                    f"method='{method}' requires xml_path=... pointing to a BNG XML file."
                )
            from bngsim._bngsim_core import RuleMonkeySimulator

            self._sim = RuleMonkeySimulator(xml_path)
            if gml is not None:
                self._sim.set_molecule_limit(int(gml))
            self._sim.set_block_same_complex_binding(bool(block_same_complex_binding))
        # Default solver options (ODE only)
        self._rtol = 1e-8
        self._atol = 1e-8
        self._max_steps = 10000
        self._jacobian = jacobian
        self._ode_jacobian_fell_back = False
        self._ss_jacobian_fell_back = False
        # GH #29: the two pins contradict each other, and a benchmark that got
        # auto-selected numbers back under a "forced" label would be worse than
        # one that failed. Refuse at construction rather than letting either win.
        if force_dense_linear_solver and force_sparse_linear_solver:
            raise ValueError(
                "force_dense_linear_solver and force_sparse_linear_solver are "
                "mutually exclusive; pass at most one. Omit both for the "
                "size/density auto-selection."
            )
        self._force_dense_linear_solver = bool(force_dense_linear_solver)
        self._force_sparse_linear_solver = bool(force_sparse_linear_solver)
        self._jax_jac_evaluator = None
        self._volume_factors_cache: list[float] | None = None

        # Registered stop conditions.
        self._stop_conditions: list[_StopCondition] = []

        # Interactive simulation state
        self._current_time: float = 0.0
        self._snapshot_stack: list[dict] = []

        # Code-generated RHS support, including model-based codegen reuse.
        self._codegen = codegen
        self._codegen_so_path = ""
        # In-process MIR micro-JIT source (GH #78). When the JIT backend is
        # selected (BNGSIM_CODEGEN_JIT=mir), the codegen C source is JIT-compiled
        # in C++ instead of being built into a .so by `cc` and dlopen'd. Carries
        # the generated source string; mutually exclusive with _codegen_so_path.
        self._codegen_c_source = ""
        jit_backend = _codegen_jit_backend()
        net_path_str = str(net_path) if net_path else ""
        self._net_path = net_path_str

        # ── Lazy analytical Functional Jacobian + large-model auto-codegen ────
        # (GH #145) Both are consumed ONLY by ODE solves, so they are deferred
        # off the model-load path and triggered here, at ODE-solve setup. Non-ODE
        # dispatch (SSA/PSA/NFsim/RuleMonkey) never reaches this branch, so it
        # never pays the SymPy derivation or the codegen compile.
        if dispatch == "ode":
            # Derive the analytical Functional Jacobian (GH #76) on first need.
            # prepare_analytical_jacobian() is once-only per model (its sentinel),
            # so repeated solves / repeated Simulators on one model derive at most
            # once, and a warmed parent passes the derived terms to clones with no
            # re-derive (warm-before-clone, GH #145 §3). jacobian="fd" needs no
            # analytical terms; "jax" uses autodiff — both skip the derivation.
            # The eager escape hatch (defer_jacobian=False / BNGSIM_EAGER_JACOBIAN
            # =1) already warmed the model at load, so this is then a no-op.
            if jacobian in ("auto", "analytical"):
                model.prepare_analytical_jacobian()

            # Large-model auto-codegen, relocated from the SBML loader (GH #145
            # §4). Native C RHS only wins above ~150-300 species (ExprTk is faster
            # below), so it triggers at/above BNGSIM_CODEGEN_THRESHOLD (256).
            # Ordered AFTER the Jacobian attach: the codegen analytical-Jacobian
            # emitter (generate_jacobian_from_model) declines unless
            # analytical_jacobian_complete is set, so the attach must populate it
            # first — the load-time "attach before codegen" invariant, preserved.
            # Scope matches the loader's original step 12: SBML / builder models
            # only (a .net model carries _net_path and codegens via its own .net
            # path on explicit codegen=True, never the model-based path here — that
            # keeps issue #15's derived-parameter chain rules). Skipped when the
            # caller set codegen explicitly (True is handled below; False opts
            # out), when BNGSIM_NO_CODEGEN is set, when the model already prepared
            # codegen (a prior Simulator — amortized, like the load-time path was),
            # or below threshold. Writes onto the model so the reuse block below
            # inherits it and a warm clone carries it, exactly as the loader did.
            if (
                codegen is None
                and not net_path_str
                and not getattr(model, "_net_path", "")
                and not getattr(model, "_codegen_so_path", "")
                and not getattr(model, "_codegen_c_source", "")
                and not os.environ.get("BNGSIM_NO_CODEGEN")
                and model.n_species >= int(os.environ.get("BNGSIM_CODEGEN_THRESHOLD", "256"))
            ):
                try:
                    if jit_backend:
                        from bngsim._codegen import prepare_model_codegen_source

                        _cg_src = prepare_model_codegen_source(model)
                        if _cg_src is not None:
                            model._codegen_c_source = _cg_src
                    else:
                        from bngsim._codegen import prepare_model_codegen

                        _cg_so = prepare_model_codegen(model)
                        if _cg_so is not None:
                            model._codegen_so_path = str(_cg_so)
                except Exception as e:
                    logger.debug("Auto-codegen skipped: %s", e)

        if codegen and dispatch == "ode":
            model_net_path = getattr(model, "_net_path", "")
            explicit_net_path = bool(net_path_str) and Path(net_path_str).suffix.lower() == ".net"
            use_net = explicit_net_path or (model_net_path and not net_path_str)
            codegen_path = net_path_str if explicit_net_path else model_net_path

            # Pass the built model to the .net codegen so the .so also carries the
            # compiled callbacks reconstructed from the (fully-populated) model:
            #   * GH #162 — the analytical Jacobian (dense / sparse CSC), but ONLY
            #     when an analytical Jacobian is wanted (prepared at L625 above);
            #     "fd"/"jax" keep the .net RHS Jacobian-free.
            #   * GH #163 — the compiled output evaluator (bngsim_codegen_outputs),
            #     emitted whenever the model qualifies (obs/func, no rateOf),
            #     INDEPENDENT of the Jacobian strategy — "fd"/"jax" record
            #     observables too, so the model is passed unconditionally and the
            #     emit_jac flag (not model=None) gates the Jacobian.
            # prepare_codegen declines each callback cleanly when it does not apply.
            emit_jac = jacobian in ("auto", "analytical")

            if jit_backend:
                # JIT path: generate the C source string; the C++ MirJit backend
                # compiles it in-process. No `cc` subprocess, no .so, no dlopen.
                if use_net:
                    from bngsim._codegen import prepare_codegen_source

                    self._codegen_c_source = prepare_codegen_source(
                        codegen_path, model, emit_jac=emit_jac
                    )
                    self._net_path = codegen_path
                    # .net-path prepares record only to the thread-local (no
                    # Model arg); surface codegen time + cache-hit on the model.
                    model._codegen_sec = last_codegen_sec()
                    model._codegen_cache_hit = last_codegen_cache_hit()
                else:
                    from bngsim._codegen import prepare_model_codegen_source

                    self._net_path = ""
                    src = prepare_model_codegen_source(model)
                    if src is None:
                        raise RuntimeError(
                            "codegen=True requested, but model-based codegen failed. "
                            "For .net models, pass net_path=... pointing to the .net file."
                        )
                    self._codegen_c_source = src
                if hasattr(model, "_codegen_c_source"):
                    model._codegen_c_source = self._codegen_c_source
                logger.info(
                    "Codegen JIT (%s) source ready: %d chars",
                    jit_backend,
                    len(self._codegen_c_source),
                )
            else:
                if use_net:
                    from bngsim._codegen import prepare_codegen

                    # prepare_codegen returns Path; the else-branch's
                    # prepare_model_codegen returns Path | None (None-checked
                    # below), so the variable must carry the union (pre-existing
                    # mypy gap).
                    so_path: Path | None = prepare_codegen(codegen_path, model, emit_jac=emit_jac)
                    self._net_path = codegen_path
                    model._codegen_sec = last_codegen_sec()  # T0.3 (see above)
                    model._codegen_cache_hit = last_codegen_cache_hit()
                else:
                    from bngsim._codegen import prepare_model_codegen

                    self._net_path = ""
                    so_path = prepare_model_codegen(model)
                    if so_path is None:
                        raise RuntimeError(
                            "codegen=True requested, but model-based codegen failed. "
                            "For .net models, pass net_path=... pointing to the .net file."
                        )
                self._codegen_so_path = str(so_path)
                if hasattr(model, "_codegen_so_path"):
                    model._codegen_so_path = self._codegen_so_path
                logger.info("Codegen .so ready: %s", self._codegen_so_path)
        elif codegen and dispatch != "ode":
            raise ValueError("codegen=True is only supported for method='ode'.")

        # Reuse model-based codegen output when the model already prepared it.
        # Prefer the JIT source when the JIT backend is active and the model
        # carries one; otherwise inherit the .so path.
        if (
            jit_backend
            and not self._codegen_c_source
            and dispatch == "ode"
            and hasattr(model, "_codegen_c_source")
            and model._codegen_c_source
        ):
            self._codegen_c_source = model._codegen_c_source
            logger.debug(
                "Auto-codegen JIT source from model: %d chars", len(self._codegen_c_source)
            )
        elif (
            not self._codegen_so_path
            and not self._codegen_c_source
            and dispatch == "ode"
            and hasattr(model, "_codegen_so_path")
            and model._codegen_so_path
        ):
            self._codegen_so_path = model._codegen_so_path
            logger.debug(
                "Auto-codegen from model: %s",
                self._codegen_so_path,
            )

        # CVODES forward sensitivities.
        self._sensitivity_params = sensitivity_params or []
        self._sensitivity_ic = sensitivity_ic or []
        # GH #198 — lazily computed (memoized) expression output-sensitivity
        # support map; None until first needed by a sensitivity run.
        self._expr_sens_support_memo: dict[str, str | None] | None = None
        if self._sensitivity_params and dispatch != "ode":
            raise ValueError("sensitivity_params is only supported for method='ode'.")
        if self._sensitivity_ic and dispatch != "ode":
            raise ValueError("sensitivity_ic is only supported for method='ode'.")
        self._raise_if_compartment_size_params(self._sensitivity_params)

        # Forward sensitivity REQUIRES an analytical codegen sensitivity RHS
        # (GH #214 follow-up): the interpreted path finite-differences the whole
        # sensitivity RHS and silently fails at tight tolerances, so
        # _auto_codegen_for_sensitivity now builds codegen unconditionally and
        # RAISES (codegen=False / BNGSIM_NO_CODEGEN / no backend / a
        # non-differentiable rate law) rather than degrading. It is a no-op only
        # when codegen was already provided/inherited. compute_all_sensitivities
        # reuses the same helper (GH #204) so its parallel chunk path matches
        # this single-shot path exactly.
        if (self._sensitivity_params or self._sensitivity_ic) and dispatch == "ode":
            self._auto_codegen_for_sensitivity(jit_backend=jit_backend)

        # Validate and store sensitivity method
        if sensitivity_method not in ("staggered", "simultaneous"):
            raise ValueError(
                f"sensitivity_method must be 'staggered' or 'simultaneous', "
                f"got '{sensitivity_method}'"
            )
        self._sensitivity_method = sensitivity_method

        # JAX AD Jacobian setup.
        if jacobian == "jax" and dispatch == "ode":
            warnings.warn(
                "jacobian='jax' is 2-80x slower than 'auto' due "
                "to Python-C++ callback overhead per Jacobian "
                "evaluation. Use jacobian='auto' for production. "
                "jacobian='jax' is intended for AD research only.",
                stacklevel=2,
            )
            if not net_path:
                raise ValueError(
                    "jacobian='jax' requires net_path=... pointing "
                    "to the .net file used to load the model."
                )
            from bngsim._jax_rhs import (
                jax_available,
                prepare_jax_jacobian,
            )

            if not jax_available():
                raise ImportError(
                    "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
                )
            eval_fn, n_sp = prepare_jax_jacobian(net_path)
            self._jax_jac_evaluator = eval_fn
            logger.info("JAX AD Jacobian ready for %d species", n_sp)
        elif jacobian == "jax" and dispatch != "ode":
            raise ValueError("jacobian='jax' is only supported for method='ode'.")

        logger.debug(
            "Created Simulator(method='%s', dispatch='%s'%s) for %r",
            method,
            dispatch,
            ", codegen=True" if codegen else "",
            model,
        )

    def _get_volume_factors(self) -> list[float]:
        """Return per-species V_c, cached on the simulator.

        Used to stamp every public-facing :class:`Result` so
        :meth:`Result.as_roadrunner` can convert stored concentrations
        back to amounts when an `X` selector is requested. Returns an
        empty list if the model can't expose codegen_data (extremely
        unlikely; .net and SBML loaders both populate it).
        """
        if self._volume_factors_cache is None:
            try:
                # T7: narrow C++ accessor returns V_c for reported species in
                # reported-species order — the same list the old
                # codegen_data()["species"] filter produced, but without
                # building a full per-parameter/species/observable/function
                # Python dict just to read one field. The reported filter
                # (GH #71) lives in the accessor so the V_c list aligns with the
                # projected Result.species columns; `reported` defaults True so
                # .net and ordinary SBML models are unaffected.
                self._volume_factors_cache = [
                    float(v) for v in self._model._core.reported_volume_factors()
                ]
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("volume_factors unavailable: %s", e)
                self._volume_factors_cache = []
        return self._volume_factors_cache

    def _compartment_size_params(self) -> set[str]:
        """Names of the model's SBML compartment-size parameters (issue #164).

        Empty for ``.net`` models, for every model built through
        :class:`ModelBuilder` directly, and for a compartment the SBML loader
        promoted to a species (rate-rule / event-resized) — that one is genuine
        live state, not a baked constant. Degrades to "none" against a core
        built before the flag existed, which loses the refusal rather than
        breaking the run.
        """
        try:
            flags = self._model._core.param_is_compartment_size
        except AttributeError:  # pragma: no cover - defensive
            return set()
        # strict=: see Model.compartment_size_params — same invariant.
        return {n for n, f in zip(self._model.param_names, flags, strict=True) if f}

    def _raise_if_compartment_size_params(self, param_names: list[str]) -> None:
        """Refuse a forward-sensitivity column for a compartment size.

        Issue #164 — the reported column is wrong in **both** directions, and
        every oracle reachable from inside the process agrees with it. A
        compartment's value is folded at load into constants no derivative here
        differentiates: per-species volume factors, amount-declared initial
        conditions, the mass-action scalar's ``Π V^n / V_storage``, the SSA
        propensity volume, and the emitted RHS. What CVODES differentiates is
        the leftover ``p[]`` reference in whichever rate laws still carry one —
        so on issue #164's model ``dA/dC1`` is reported as 36.6 where the truth
        is 0 (``A`` is exactly ``C1``-invariant), and ``dB/dC2`` as 0 where the
        truth is 2.30. Both errors survive a finite-difference check, because a
        re-solve at ``p ± h`` moves the parameter through ``set_param`` and
        inherits the same staleness; only rebuilding from source disagrees.

        So refuse the column. The gradient is available by rebuilding at
        ``V ± h`` (``Model.from_sbml(..., compartment_sizes=...)``), which is
        what the tests use as the oracle. Issue #170 tracks the analytic column.
        """
        if not param_names:
            return
        bad = sorted(set(param_names) & self._compartment_size_params())
        if bad:
            raise ValueError(
                f"Forward sensitivity is not supported for compartment size(s) "
                f"{bad}: an SBML compartment's value is folded at load into constants "
                f"the sensitivity RHS does not differentiate (per-species volume "
                f"factors, amount-declared initial conditions, mass-action rate "
                f"constants, SSA propensity volumes), so the column would be wrong in "
                f"both directions — an invented gradient where the true one is zero, "
                f"and a silent zero where it is not. bngsim refuses rather than return "
                f"a confidently wrong derivative (issue #164). A finite difference "
                f"through a rebuild is the available gradient: reload with "
                f"Model.from_sbml(..., compartment_sizes={{...}}) at V ± h. Issue #170 "
                f"tracks making a compartment size differentiable."
            )

    def _raise_if_event_sensitivities(self, param_names: list[str] | None = None) -> None:
        """Refuse output sensitivities only for unsupported event subclasses.

        Originally a blanket refusal on any model with events (GH #205): the
        integrator reinitialises state at an event (``CVodeReInit``) but the
        CVODES forward-sensitivity vectors were never reinitialised, so the
        columns went silently stale at and after the first fire.

        GH #212 lifted the refusal for the **fixed-time subclass** (``g = time −
        T``, the dosing/stimulation pattern), where the event-time sensitivity
        ``∂t*/∂p = 0`` and the core applies the sensitivity jump
        ``s⁺ = J_h·s⁻ + ∂h/∂p`` plus ``CVodeSensReInit`` at each fire. Issue #49
        added the events whose *threshold* is a fitted constant, resolving
        ``∂t*/∂p`` before the run. Issue #144 added the **state-dependent**
        triggers — ``v > 30``, whose crossing time moves with every parameter
        through the trajectory — by differentiating the crossing at each fire.
        What still raises: execution delays, and any trigger that does not
        reduce to a single relational comparison (a conjunction, a negation, an
        equality), whose crossing has no single differentiable surface.

        The classification is delegated to the core
        (:func:`NetworkModel.event_sensitivity_unsupported_reason`), which knows
        each event's persistence/delay and — via the trigger's referenced
        variables — whether its crossing time moves, and whether it can be
        differentiated. "Moves" is judged against every address that carries
        live state, not just species concentrations: an observable total or a
        rateOf accessor moves with the trajectory too, so a trigger reading one
        has a parameter-dependent crossing time even though it names no
        parameter (issue #52). ``param_names`` is the set of parameters whose
        sensitivities this call requests (defaults to
        ``self._sensitivity_params``); an IC-only request passes an empty list,
        which still exercises the persistence/delay checks — and, since issue
        #144, still gets a crossing shift on its IC columns.

        Discontinuity triggers (GH #72 forcing pulses) do not jump state and are
        not events, so they are unaffected.
        """
        if self._model._core.n_events <= 0:
            return
        names = list(param_names) if param_names is not None else list(self._sensitivity_params)
        compensated, detail, blocked = self._event_time_compensation(names)
        reason = self._model._core.event_sensitivity_unsupported_reason(names, compensated)
        # An event the solver differentiates at the fire (issue #144) is not
        # blocked, whatever the ahead-of-run detector made of its trigger: the
        # detector only recognizes clock thresholds, so it reports `S < S_max`
        # as unresolvable even when `S_max` is requested. Filtering here rather
        # than teaching the detector keeps the core the single authority on
        # which crossings are differentiable.
        if blocked:
            runtime = set(self._model._core.events_with_runtime_event_time_sens())
            blocked = {ei: msg for ei, msg in blocked.items() if ei not in runtime}
        if reason is None and blocked:
            # The core tests the trigger's *bound addresses*, which cannot see
            # through an assignment-rule parameter: in `time >= t_rule` with
            # `t_rule = t_first + …`, the trigger binds to `t_rule`'s address and
            # never to `t_first`'s, so a requested `t_first` looks absent. The
            # detector inlines the rule and knows better (issue #49).
            reason = sorted(blocked.values())[0] + "."
            detail = ""
        if reason:
            raise ValueError(
                "Output sensitivities are not supported for this model's events: "
                + reason
                + detail
                + " bngsim refuses rather than return silently-stale derivatives "
                "(GH #205). Forward sensitivity is supported through fixed-time events, "
                "through events whose trigger thresholds a fitted constant (issue #49), "
                "and through state-dependent triggers that reduce to a single relational "
                "comparison (issue #144); the remaining subclasses are tracked in issue "
                "#144."
            )

    def _event_time_compensation(self, names: list[str]) -> tuple[list[int], str, dict]:
        """Which events carry a known ``∂t*/∂p``, and why the others do not.

        Compensation is a property of the trigger alone, not of the run window —
        :func:`compute_event_time_sens` decides it before it looks at ``t_star``
        — so this is safe to ask before ``t_span`` is resolved. Detection
        failure degrades to "nothing compensated", i.e. the pre-#49 refusal.
        """
        from bngsim._switch_sensitivity import compute_event_time_sens

        try:
            res = compute_event_time_sens(self._model._core, names, float("-inf"), float("inf"))
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Event-time sensitivity detection failed (%s); refusing as before", e)
            return [], "", {}
        detail = ""
        if res.reasons:
            detail = " Detail: " + "; ".join(sorted(res.reasons.values())) + "."
        return list(res.compensated), detail, dict(res.blocked)

    def _apply_event_time_sens(self, opts, core, t_start, t_end, param_names=None) -> None:
        """Inject each event's ``∂t*/∂p`` (issue #49).

        An event whose trigger thresholds a fitted constant — ``time >= T0``
        with ``T0`` requested — fires at a time that moves with the parameter,
        so its forward-sensitivity jump carries two terms the GH #212 state jump
        does not: ``s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p``. This is
        the plumbing that hands the detector's result to the solver; a no-op
        unless some trigger threshold actually moves with a requested parameter,
        which leaves every fixed-time event model byte-identical.

        Unlike the issue #48 switch path this needs no ``CVodeSetStopTime`` and
        no parameter pinning: an event trigger is not part of ``f``, so ``f`` is
        smooth right up to the root (CVODE's root finder already stops exactly
        at ``t*``) and ``∂f/∂T0`` is genuinely zero without help.
        """
        names = list(param_names) if param_names is not None else list(self._sensitivity_params)
        if not names or core.n_events <= 0:
            return
        from bngsim._switch_sensitivity import compute_event_time_sens

        try:
            res = compute_event_time_sens(core, names, float(t_start), float(t_end))
        except Exception as e:  # pragma: no cover - defensive
            # The guard above already refused anything whose ∂t*/∂p is needed
            # but unavailable, so reaching here means detection worked once and
            # failed now. Warn rather than silently zero the event-time column.
            logger.warning(
                "Event-time sensitivity detection failed (%s); any event-time "
                "parameter's gradient will be zero (issue #49).",
                e,
            )
            return
        if res.records:
            logger.info(
                "Event-time forward sensitivity: %d event(s) with a moving crossing (issue #49)",
                len(res.records),
            )
            opts.set_event_time_sens(res.records)

    def _apply_ic_param_sens_seed(
        self, opts, model: Model, param_names: Sequence[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Inject ∂x_i(0)/∂p initial-condition sensitivity seeds (issue #43).

        When a species initial condition is a parameter reference — directly
        (``R() R0``) or through a derived ConstantExpression (``R() Rtot`` with
        ``Rtot = R0``) — the forward-sensitivity seed yS_i(0) must carry the IC
        Jacobian column ∂(IC)/∂p. The C++ seeding cannot differentiate a derived
        IC, so the coefficients are computed from the model's parameter graph via
        the sympy chain rule and passed through ``SolverOptions``. A no-op for the
        common model with no parameter-referenced species ICs, and for IC-only
        sensitivity (no ``sensitivity_params``), where param columns don't exist.

        This is the plumbing only. The derivation — including the issue #113
        retirement of a species moved off its declared initial condition and the
        issue #111 :meth:`Model.declare_ic_sensitivity` overlay — lives in
        :meth:`Model._ic_sensitivity_triples`, which is also what
        :meth:`Model.effective_ic_sensitivity` reports from, so the matrix a
        caller reads cannot drift from the one the solver was seeded with.

        ``param_names`` is the sensitivity-parameter column order this run will
        use — the chunked path differentiates a subset per chunk, so it cannot be
        assumed to be ``self._sensitivity_params``. It selects which rows are
        *reported*; the C++ seeding applies the same filter itself.

        Returns that effective matrix for :attr:`Result.ic_sensitivity_seed` to
        carry alongside the run (issue #155).
        """
        names = list(param_names) if param_names is not None else list(self._sensitivity_params)
        if not names:
            return {}
        triples, injected = model._ic_sensitivity_triples()
        if injected:
            # A zero coefficient seeds nothing, so it is dropped from the list the
            # core receives; the *report* keeps it, because "seeded, zero here" and
            # "no seeding path" are different answers to a consumer (issue #155).
            opts.set_ic_param_sens([t for t in triples if t[2] != 0.0] or [(-1, 0, 0.0)])
        return model.effective_ic_sensitivity(names)

    def _apply_switch_time_sens(self, opts, core, t_start, t_end, param_names=None) -> None:
        """Inject the switch-time crossings and their ∂t*/∂p (issue #48).

        A *switch time* is a fitted parameter that sets **when** a step in the
        dynamics occurs — the ``if(t>=sigma, ...)`` onset times of the Lin2021
        COVID model. ``∂f/∂sigma`` is a clean ``0`` inside each smooth branch, so
        the variational source term carries no information about it and the whole
        gradient is a jump ``s⁺ = s⁻ + (f⁻−f⁺)·∂t*/∂p`` at the crossing, which the
        core applies after stopping there with ``CVodeSetStopTime``.

        Detection (which ``if()`` conditions threshold a unit-rate clock) and the
        chain rule from each threshold to its fitted primaries live in
        :mod:`bngsim._switch_sensitivity`; this is the plumbing that hands the
        result to the solver. ``param_names`` is the column order this run will
        use — the chunked path differentiates a subset per chunk, so it cannot be
        assumed to be ``self._sensitivity_params``.

        A no-op unless some ``if()`` threshold actually moves with a requested
        parameter, which leaves every other model's integration untouched.
        """
        names = list(param_names) if param_names is not None else list(self._sensitivity_params)
        if not names:
            return
        from bngsim._switch_sensitivity import compute_switch_time_sens

        try:
            records, pinned = compute_switch_time_sens(core, names, float(t_start), float(t_end))
        except ValueError:
            # An unsupported switch parameter (one that also acts in-branch) is a
            # refusal the caller must see, not a detection hiccup to swallow.
            raise
        except Exception as e:  # pragma: no cover - defensive
            # Detection is best-effort: failing it leaves the pre-#48 behavior
            # (a switch-time column of zeros), so degrade rather than break a run
            # that may not have a switch time at all. Warn, though — for a model
            # that DOES fit one, this is the difference between a gradient and a
            # silent zero.
            logger.warning(
                "Switch-time sensitivity detection failed (%s); any switch-time "
                "parameter's gradient will be zero (issue #48).",
                e,
            )
            return
        if records:
            logger.info(
                "Switch-time forward sensitivity: %d crossing(s) at t=%s (issue #48)",
                len(records),
                ", ".join(f"{r[0]:.6g}" for r in records),
            )
            opts.set_switch_time_sens(records)
            opts.set_switch_pinned_params(pinned)

    def _apply_state_switch_sens(self, opts, core) -> None:
        """Register the state-dependent rate-law switches to jump at (issue #150).

        The rate-law twin of :meth:`_apply_event_time_sens`. A condition that
        reads the state — ``piecewise(0, Virus < 1, Virus*rho_V)`` — flips a
        branch of ``f`` at a crossing whose time moves with every parameter
        through the trajectory, so ``∂x/∂θ`` jumps there by the saltation term
        ``(f⁻−f⁺)·dt*/dθ``. Neither the analytic sensitivity RHS nor CVODES'
        internal difference quotient carries it; handing the conditions to the
        core is what registers each crossing as a root and applies the jump.

        Unlike the switch-time and event-time detectors this takes no parameter
        list: a *state* crossing moves with every column, initial conditions
        included, so there is no subset it can be skipped for.

        A no-op for any model with no conditional rate law, which leaves the
        root set — and the whole integration — untouched.
        """
        from bngsim._switch_sensitivity import state_switch_conditions

        try:
            conditions = state_switch_conditions(core)
        except Exception as e:  # pragma: no cover - defensive
            # Detection is best-effort: failing it leaves the pre-#150 behavior
            # (columns short by the crossing jump), which the codegen gate still
            # warns about. Degrading beats breaking a run that may have no state
            # switch at all — but say so, because for a model that DOES cross one
            # this is the difference between a gradient and a wrong gradient.
            logger.warning(
                "State-switch sensitivity detection failed (%s); a rate-law condition "
                "that reads model state will not have its crossing jump applied "
                "(issue #150).",
                e,
            )
            return
        if conditions:
            logger.info(
                "State-switch forward sensitivity: %d condition(s) rooted and jumped "
                "(issue #150): %s",
                len(conditions),
                ", ".join(repr(c) for c in conditions),
            )
            opts.set_state_switch_conditions(conditions)

    def _auto_codegen_for_sensitivity(
        self, *, jit_backend: str, n_sens_dirs: int | None = None
    ) -> None:
        """Build & attach a code-generated RHS for a sensitivity workflow.

        Shared by the constructor (when ``sensitivity_params`` /
        ``sensitivity_ic`` are given) and :meth:`compute_all_sensitivities`. A
        no-op only when a codegen ``.so`` / JIT source is already present or
        inherited.

        ``n_sens_dirs`` is accepted for call-site compatibility but unused: there
        is no longer a size gate (it is ``del``-d below).

        HARD REQUIREMENT (GH #214 follow-up). Forward sensitivity now *requires*
        an analytical codegen sensitivity RHS — the size gate and the silent
        interpreted fallback were retired. Rationale: without a codegen sens
        function CVODES finite-differences the entire sensitivity RHS
        (``∂f/∂y·s + ∂f/∂p``); that ~sqrt(eps) noise cannot support tight
        tolerances, so the error test silently micro-steps to a halt (the
        preequilibration model hangs at rtol=1e-11 — ~92M steps). The old
        docstring's claim that the interpreted path is "numerically identical" is
        true for the state RHS ``f(x)`` but FALSE for the sensitivity RHS.

        This raises rather than degrading:
          * ``codegen=False`` or ``BNGSIM_NO_CODEGEN`` + sensitivities → raise;
          * no codegen backend (no C compiler and no JIT) → raise;
          * the model's rate laws cannot be differentiated to closed form (a
            non-smooth construct, ``rateOf()`` in a rate law, an unparseable
            expression) → raise.
        The analytical RHS builds via cc, or the in-process MIR JIT where no
        compiler exists, so requiring it does not require a system compiler.
        """
        model = self._model
        del n_sens_dirs  # retained for call-site compatibility; no size gate now
        # Already have a codegen RHS (explicitly provided or inherited from the
        # model) — nothing to build.
        if self._codegen_so_path or self._codegen_c_source:
            return

        # Hard requirement (GH #214 follow-up). Forward sensitivity needs an
        # ANALYTICAL sensitivity RHS. With no codegen sens function CVODES
        # finite-differences the *entire* sensitivity RHS (∂f/∂y·s + ∂f/∂p), and
        # that ~sqrt(eps) noise cannot support tight tolerances — the error test
        # silently micro-steps to a halt (the preequilibration model hangs at
        # rtol=1e-11: ~92M steps). So codegen is REQUIRED; we refuse rather than
        # fall back to the finite-difference path. The analytical RHS is built via
        # cc, or the in-process MIR JIT where no compiler exists, so this does not
        # require a system compiler.
        if self._codegen is False:
            raise ValueError(
                "Forward sensitivity analysis requires code generation, but "
                "codegen=False was passed. The analytical sensitivity RHS is "
                "built automatically — remove codegen=False, or drop the "
                "sensitivity request. (The interpreted finite-difference "
                "sensitivity path was retired because it silently fails at tight "
                "tolerances; GH #214.)"
            )
        if os.environ.get("BNGSIM_NO_CODEGEN"):
            raise ValueError(
                "Forward sensitivity analysis requires code generation, but "
                "BNGSIM_NO_CODEGEN is set. Unset it for sensitivity runs (the "
                "interpreted finite-difference sensitivity path was retired "
                "because it silently fails at tight tolerances; GH #214)."
            )
        # Prefer the .net path when the model carries a net_path
        # (Model.from_net stashes it). The .net codegen handles derived-parameter
        # chain rules (e.g., ``_rateLaw{N} = chi*kon``) that the model-based path
        # does not (issue #15). Falls through to model-based codegen for
        # from_sbml / from_antimony / from_builder.
        model_net_path = getattr(model, "_net_path", "")
        # GH #163 appends the compiled output evaluator whenever the model
        # qualifies; GH #162 appends the analytical Jacobian when one is wanted.
        emit_jac = self._jacobian in ("auto", "analytical")
        auto_src: str | None = None
        auto_so: Path | None = None
        try:
            if jit_backend:
                # In-process MIR micro-JIT backend (GH #78): generate the same
                # combined RHS + sensitivity-RHS C source string the cc path
                # compiles, and hand it to the C++ MirJit instead of building a
                # .so. Numerically identical RHS either way.
                if model_net_path:
                    from bngsim._codegen import prepare_codegen_source

                    auto_src = prepare_codegen_source(model_net_path, model, emit_jac=emit_jac)
                    self._net_path = model_net_path
                    model._codegen_sec = last_codegen_sec()  # T0.3 (.net path)
                    model._codegen_cache_hit = last_codegen_cache_hit()
                else:
                    from bngsim._codegen import prepare_model_codegen_source

                    auto_src = prepare_model_codegen_source(model)
            else:
                if model_net_path:
                    from bngsim._codegen import prepare_codegen

                    auto_so = prepare_codegen(model_net_path, model, emit_jac=emit_jac)
                    self._net_path = model_net_path
                    model._codegen_sec = last_codegen_sec()  # T0.3 (.net path)
                    model._codegen_cache_hit = last_codegen_cache_hit()
                else:
                    from bngsim._codegen import prepare_model_codegen

                    auto_so = prepare_model_codegen(model)
        except Exception as e:
            # A backend/compile failure (e.g. no C compiler and no JIT) — not a
            # differentiability issue. Refuse loudly rather than silently fall to
            # the finite-difference sensitivity path (GH #214).
            raise RuntimeError(
                "Failed to build the analytical sensitivity RHS required for "
                f"forward sensitivity ({type(e).__name__}: {e}). This needs a "
                "codegen backend: a C compiler, or BNGSIM_CODEGEN_JIT=mir for the "
                "in-process MIR JIT."
            ) from e

        # A None return means codegen DECLINED — the model's rate laws could not
        # be differentiated to closed form. Refuse rather than return unreliable
        # finite-difference sensitivities (GH #214).
        diff_err = (
            "Could not generate an analytical sensitivity RHS for this model: its "
            "rate laws could not be differentiated to closed form (e.g. a "
            "non-smooth construct such as min/max/abs/floor, rateOf() inside a "
            "rate law, or an unparseable expression). Forward sensitivity needs a "
            "differentiable RHS; bngsim refuses rather than return unreliable "
            "finite-difference derivatives (GH #214). If the rate law is smooth "
            "but unsupported here, please file a codegen issue."
        )
        if jit_backend:
            if auto_src is None:
                raise ValueError(diff_err)
            self._codegen_c_source = auto_src
            if hasattr(model, "_codegen_c_source"):
                model._codegen_c_source = self._codegen_c_source
            logger.info(
                "Auto-enabled codegen JIT (%s) for sensitivity workflow: %d chars",
                jit_backend,
                len(self._codegen_c_source),
            )
        else:
            if auto_so is None:
                raise ValueError(diff_err)
            self._codegen_so_path = str(auto_so)
            if hasattr(model, "_codegen_so_path"):
                model._codegen_so_path = self._codegen_so_path
            logger.info(
                "Auto-enabled codegen for sensitivity workflow: %s",
                self._codegen_so_path,
            )

    def _prepare_output_sens_codegen(self) -> None:
        """Attach a codegen artifact that CARRIES ``bngsim_codegen_output_sens``.

        :meth:`_auto_codegen_for_sensitivity` alone is not enough for any entry
        point that takes ``sensitivity_params`` as a *method* argument. The GH #198
        output-sensitivity evaluator is emitted only when the model carries
        ``_want_output_sens``, which :meth:`__init__` sets from its own
        ``sensitivity_params``; a method-argument request leaves it False, so the
        artifact arrives with no symbol to call and the d(output)/dθ consumer
        silently drops to finite differences (:meth:`compute_all_sensitivities`) or
        an empty block (GH #205).

        Two construction-time wrinkles to clear past:

        * .net models never auto-codegen at construction (the species-threshold
          attach is SBML/builder-only), so the helper below always fires fresh.
        * an SBML/builder model CAN already carry a plain-RHS codegen ``.so`` /
          source from construction (species-threshold attach, explicit
          ``codegen=True``, or inherited) — built WITHOUT output sens because
          ``_want_output_sens`` was then False. :meth:`_auto_codegen_for_sensitivity`
          no-ops on an already-attached codegen, so that plain artifact would shadow
          the sensitivity one. ``_want_output_sens`` doubles as the "the attached
          codegen already has output sens" signal: when it was False, clear the
          plain artifact so the helper regenerates with output sens (the result is a
          superset; the ``.so`` cache keeps a repeat cheap), restoring it if
          regeneration produces nothing so the RHS speed-up survives. When it was
          already True (a ``sensitivity_params``-built sim, or a second call here),
          skip the clear so a large model is not needlessly re-generated.

        A function-free model needs none of this: ``_codegen_emit_flags`` gates the
        evaluator on ``n_functions``, so the source is byte-identical with or
        without the flag and an inherited plain-RHS codegen is already right.
        """
        model = self._model
        if model._core.n_functions > 0 and not model._want_output_sens:
            model._want_output_sens = True
            prev_so, prev_src = self._codegen_so_path, self._codegen_c_source
            self._codegen_so_path = ""
            self._codegen_c_source = ""
            try:
                self._auto_codegen_for_sensitivity(jit_backend=_codegen_jit_backend())
            except Exception:
                # Dropping a WORKING artifact to regenerate it must not be able to
                # turn a call that used to succeed into a refusal: the helper
                # no-ops on an attached artifact, so before the drop this could
                # not raise at all. Put the old one back and let the consumer take
                # its finite-difference fallback. With nothing to put back the
                # refusal is the real one (GH #214) and propagates.
                if not prev_so and not prev_src:
                    raise
                self._codegen_so_path, self._codegen_c_source = prev_so, prev_src
                logger.info(
                    "Output-sensitivity codegen regeneration failed; keeping the "
                    "previously attached artifact (d(output)/dp falls back to "
                    "finite differences)."
                )
            if not self._codegen_so_path and not self._codegen_c_source:
                self._codegen_so_path, self._codegen_c_source = prev_so, prev_src
        else:
            self._auto_codegen_for_sensitivity(jit_backend=_codegen_jit_backend())

    def _expression_sens_support(self) -> dict[str, str | None]:
        """Memoized ``{function_name: unsupported_reason_or_None}`` for GH #198
        expression output sensitivities, from the model's codegen analysis.

        Computed once per Simulator (the function bodies do not change across runs,
        so a fitting loop pays the sympy analysis a single time). A failure to
        analyze (e.g. a model that cannot expose ``codegen_data``) degrades to an
        empty map, so output_sensitivities still raises the generic empty-block
        error rather than crashing here.
        """
        memo = self._expr_sens_support_memo
        if memo is None:
            try:
                from bngsim._codegen import output_sens_support

                memo = output_sens_support(self._model)
            except Exception:  # pragma: no cover - defensive; analysis is best-effort
                memo = {}
            self._expr_sens_support_memo = memo
        return memo

    def _stamp(
        self,
        result: Result,
        *,
        seed: int | None = None,
        ic_seed: dict[str, dict[str, float]] | None = None,
    ) -> Result:
        """Attach per-species V_c, stochastic seed and ∂x(0)/∂θ to *result*."""
        vf = self._get_volume_factors()
        if vf:
            result._species_volume_factors = vf
        if seed is not None:
            result._seed = seed
        if ic_seed is not None:
            result._ic_sensitivity_seed = ic_seed
        self._apply_ar_report_map(result)
        self._apply_varvol_conc_map(result)
        self._apply_varvol_ar_conc_map(result)
        self._apply_varvol_event_resize_map(result)
        ar_map, ar_blocked = self._ar_sensitivity_metadata()
        result._ar_sens_map = ar_map
        result._ar_sens_blocked = ar_blocked
        return result

    def _ar_sensitivity_metadata(self) -> tuple[dict[str, tuple[str, str, float]], frozenset[str]]:
        """AR-species output-sensitivity redirect map + blocked set (GH #205).

        The redirect map is the same ``_ar_report_map`` the value path uses to
        overwrite a frozen AssignmentRule-target species column with its rule's
        live value: ``species_name → (kind, src, vdiv)``, where ``kind`` is
        ``"observable"`` (linear-on-species rule, GH #197) or ``"expression"``
        (everything else, GH #198). ``Result.output_sensitivities`` redirects a
        ``species:<ar>`` selector through it so the derivative follows the
        assignment expression rather than the raw frozen-state ``yS``.

        The blocked set is AR species whose reported value *also* carries a
        time-varying volume rescale (``_varvol_conc_map`` / ``_varvol_ar_conc_map``,
        GH #85/#87): the redirect scales only by the constant ``vdiv``, so those
        species' output sensitivities are refused rather than returned subtly
        wrong. Both are empty for .net and non-AR models (no redirect).
        """
        amap = getattr(self._model, "_ar_report_map", None) or {}
        if not amap:
            return {}, frozenset()
        vc = getattr(self._model, "_varvol_conc_map", None) or {}
        vac = getattr(self._model, "_varvol_ar_conc_map", None) or {}
        blocked = frozenset(name for name in amap if name in vc or name in vac)
        return dict(amap), blocked

    def _apply_ar_report_map(self, result: Result) -> None:
        """Report AssignmentRule-target species at their live rule value.

        An AR-target species is emitted ``fixed`` (the loader zeroes its ODE
        derivative), so the integrator leaves the species column frozen at its
        initial value. The rule's true time-varying value — what RR reports by
        re-evaluating the rule each step — is carried under the same bare name
        as an observable (linear-on-species rules) or an expression/function
        (everything else). Overwrite the frozen species column with that live
        column. No-op for .net and non-AR models (empty map).
        """
        amap = getattr(self._model, "_ar_report_map", None)
        if not amap:
            return
        species_names = result._species_names
        if not species_names:
            return
        # Only the 2D (n_times, n_species) layout (single run / PSA mean) is
        # column-addressable here. squeezed run_batch results are 3D
        # (n_reps, n_times, n_species); skip the cosmetic report-remap there —
        # the dynamics fix (classifier reroute) already applies per replicate.
        if result._species.ndim != 2:
            return
        sp_idx = {n: i for i, n in enumerate(species_names)}
        obs_idx = {n: i for i, n in enumerate(result._observable_names)}
        expr_idx = {n: i for i, n in enumerate(result._expression_names)}
        # Copy once so we never mutate a buffer aliasing C++-owned memory.
        sp = np.array(result._species, dtype=np.float64, copy=True)
        changed = False
        for name, entry in amap.items():
            # entry is (kind, src, vdiv). vdiv (GH #75) is V_c(target) when the
            # AR target is an hOSU=true V≠1 species — the rule's observable /
            # expression yields the target's amount, and bngsim reports stored
            # concentration = amount / V_c(target). 1.0 for V=1 / hOSU=false /
            # legacy 2-tuples ⇒ no-op (byte-identical reporting).
            kind, src = entry[0], entry[1]
            vdiv = entry[2] if len(entry) > 2 else 1.0
            j = sp_idx.get(name)
            if j is None:
                continue
            col = None
            if kind == "observable" and src in obs_idx:
                col = result._observables[:, obs_idx[src]]
            elif kind == "expression" and src in expr_idx:
                col = result._expressions[:, expr_idx[src]]
            if col is None:
                continue
            sp[:, j] = col / vdiv if vdiv != 1.0 else col
            changed = True
        if changed:
            result._species = sp

    def _apply_varvol_conc_map(self, result: Result) -> None:
        """Report species in variable-volume compartments at amount/V_live(t).

        bngsim stores every species as ``amount / V_static`` (the compartment
        size at load, carried as ``volume_factor``). That equals the true
        concentration only while the compartment is static; for a species whose
        compartment is driven by a rate rule (or resized by an event) the live
        size V(t) diverges, so the reported concentration is stale by exactly
        ``V_static / V_live(t)``. The integrated amounts are already correct —
        the dynamics divide Functional rates by the live compartment symbol
        (GH #74) — so this rescales only the reported concentration column,
        reading V_live(t) from the compartment's own promoted-species column
        (``volume_factor`` 1.0, so its stored value *is* the live size).

        Records the live-volume column index per rescaled species in
        ``result._varvol_live_vol`` so :meth:`Result.as_roadrunner` can recover
        the amount (``conc * V_live``) for a bare-id selector instead of the now
        meaningless ``conc * V_static``. No-op for .net and static models (empty
        map) and for the 3-D batch layout (the dynamics fix already applies per
        replicate; the cosmetic report-remap, like the AR remap, only addresses
        the 2-D single-run / PSA-mean layout). GH #85.
        """
        vmap = getattr(self._model, "_varvol_conc_map", None)
        amap = getattr(self._model, "_varvol_amount_map", None)
        if not vmap and not amap:
            return
        species_names = result._species_names
        if not species_names or result._species.ndim != 2:
            return
        sp_idx = {n: i for i, n in enumerate(species_names)}
        vf = self._get_volume_factors()
        # SSA/PSA preserve molecule counts across a volume change (the ODE
        # dilution / event concentration-rescale are ``ode_only`` and skipped),
        # so a stochastic result stores ``amount/V_static`` where the ODE result
        # stores the live ``amount/V_live`` for the same species. The reporting
        # rescale below is therefore method-dependent. GH #131.
        stochastic = self._method in ("ssa", "psa")
        # Copy once so we never mutate a buffer aliasing C++-owned memory.
        sp = np.array(result._species, dtype=np.float64, copy=True)
        live_vol: dict[int, int] = {}
        conc_factor: dict[int, np.ndarray] = {}
        changed = False

        # hOSU=true species in a rate-rule compartment (vmap). Stored as
        # amount/V_static under BOTH methods, but the V_static→V_live correction
        # is applied differently:
        #   • ODE reports in concentration space, so rescale the raw column in
        #     place (sp[:,j] *= V_static/V_live) and record the live-volume column
        #     so as_roadrunner recovers the amount as conc·V_live.
        #   • SSA keeps the raw column as the conserved molecule count
        #     (amount/V_static) — every SSA test and the engine's own state read
        #     it that way — so leave sp[:,j] untouched and record the per-sample
        #     concentration factor V_static/V_live instead; as_roadrunner applies
        #     it to the [S] selector only, while the bare amount selector recovers
        #     amount as raw·V_static via the volume factor. GH #131.
        for s_name, c_name in (vmap or {}).items():
            j = sp_idx.get(s_name)
            k = sp_idx.get(c_name)
            # k missing ⇒ the compartment column is unreported (e.g. an
            # event-promoted compartment hidden per GH #71); without V_live(t)
            # we cannot rescale, so leave the stale amount/V_static rather than
            # guess. j missing ⇒ unreported species. Either way, skip.
            if j is None or k is None:
                continue
            v_static = vf[j] if j < len(vf) else 1.0
            v_live = sp[:, k]
            # V_live(0) == V_static, so factor(0) == 1 and t0 reporting is
            # unchanged. Guard the (physically impossible) zero-volume sample
            # rather than emit inf/nan.
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(v_live != 0.0, v_static / v_live, 1.0)
            if stochastic:
                conc_factor[j] = factor
            else:
                sp[:, j] = sp[:, j] * factor
                live_vol[j] = k
            changed = True

        # GH #86: hOSU=false species in a rate-rule compartment.
        #
        # Under ODE the #86 dilution term ``-[S]·V̇/V`` is integrated, so the
        # stored concentration is already the live ``amount/V_live(t)``; the
        # column is correct and only the bare-id amount selector needs the
        # live-volume column (``conc·V_live``, not the stale ``conc·V_static``).
        #
        # Under SSA/PSA that dilution reaction is ``ode_only`` and skipped (the
        # molecule count is conserved by construction; the live volume's effect
        # on propensities is carried by the engine's ``(V_static/V_live)^…``
        # correction). The stored value therefore stays ``amount/V_static`` —
        # exactly the hOSU=true (vmap) situation above — so the [S] selector
        # needs the same V_static/V_live concentration factor. GH #131 finding 1.
        for s_name, c_name in (amap or {}).items():
            j = sp_idx.get(s_name)
            k = sp_idx.get(c_name)
            if j is None or k is None:
                continue
            if stochastic:
                v_static = vf[j] if j < len(vf) else 1.0
                v_live = sp[:, k]
                with np.errstate(divide="ignore", invalid="ignore"):
                    conc_factor[j] = np.where(v_live != 0.0, v_static / v_live, 1.0)
            else:
                live_vol[j] = k
            changed = True

        if changed:
            result._species = sp
            if live_vol:
                result._varvol_live_vol = live_vol
            if conc_factor:
                result._varvol_conc_factor = conc_factor

    def _apply_varvol_ar_conc_map(self, result: Result) -> None:
        """Report species in ASSIGNMENT-RULE compartments at amount/V_live(t).

        Companion to :meth:`_apply_varvol_conc_map` for compartments whose size
        is set by an assignment rule (e.g. ``tV := mV + dV``) rather than a rate
        rule. After the AR-report and rate-rule-varvol passes, every amount-valued
        species in such a compartment holds ``amount / V_static`` (a plain species
        stores that directly; an AR-target species was set to it by
        :meth:`_apply_ar_report_map` via ``vdiv = V_static``). The true reported
        concentration is ``amount / V_live(t)``, so rescale uniformly by
        ``V_static / V_live(t)``.

        Unlike the rate-rule map, the live volume is NOT a promoted-species
        column — an AR compartment has no ODE state. It is read from the
        compartment's own assignment-rule **expression** column (the loader emits
        a function named after the compartment). No-op for .net and models without
        an assignment-rule compartment (empty map), and for the 3-D batch layout.
        GH #87.
        """
        amap = getattr(self._model, "_varvol_ar_conc_map", None)
        # (#234) hOSU=false counterpart: a diluted species' stored column is already
        # amount/V_live, so only its bare-id amount selector needs V_live (read from
        # the AR expression column) — no column rescale. Handled in the same pass.
        amount_map = getattr(self._model, "_varvol_ar_amount_map", None)
        if not amap and not amount_map:
            return
        species_names = result._species_names
        if not species_names or result._species.ndim != 2:
            return
        expr_names = result._expression_names
        if not expr_names:
            return
        sp_idx = {n: i for i, n in enumerate(species_names)}
        expr_idx = {n: i for i, n in enumerate(expr_names)}
        # Copy once so we never mutate a buffer aliasing C++-owned memory.
        sp = np.array(result._species, dtype=np.float64, copy=True)
        changed = False
        for s_name, (comp_name, v_static) in (amap or {}).items():
            j = sp_idx.get(s_name)
            k = expr_idx.get(comp_name)
            # k missing ⇒ the compartment's AR expression is unreported; without
            # V_live(t) we cannot rescale, so leave the stale amount/V_static.
            if j is None or k is None:
                continue
            v_live = result._expressions[:, k]
            # V_live(0) == V_static ⇒ factor(0) == 1, so t0 reporting is
            # unchanged. Guard the (physically impossible) zero-volume sample.
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(v_live != 0.0, v_static / v_live, 1.0)
            sp[:, j] = sp[:, j] * factor
            changed = True
        if changed:
            result._species = sp

        # (#234) Record V_live(t) per diluted hOSU=false species so as_roadrunner's
        # bare-id amount selector reports conc·V_live(t), not the stale conc·V_static
        # the volume factor would give. The concentration column is left untouched.
        if amount_map:
            amount_factor = result._varvol_amount_factor or {}
            recorded = False
            for s_name, comp_name in amount_map.items():
                j = sp_idx.get(s_name)
                k = expr_idx.get(comp_name)
                if j is None or k is None:
                    continue
                amount_factor[j] = result._expressions[:, k]
                recorded = True
            if recorded:
                result._varvol_amount_factor = amount_factor

    def _apply_varvol_event_resize_map(self, result: Result) -> None:
        """Report species in EVENT-RESIZED compartments at amount/V_live(t).

        An event assignment changes a compartment's size discretely. The right
        report-time correction depends on hOSU and method, because the raw column
        holds different things (RoadRunner reports ``[X]`` = amount/V_live and the
        bare ``X`` = amount for every species):

          * hOSU=true, BOTH methods — raw is ``amount/V_static`` (the amount is
            conserved across the resize; the injected ``V_old/V_new`` rescale only
            touches hOSU=false concentration columns). ``[X]`` is stale by
            ``V_static/V_live`` → record a concentration factor; the bare ``X`` is
            already correct as ``raw·V_static`` via the volume factor.
          * hOSU=false, SSA/PSA — that injected rescale is ``ode_only`` and skipped
            to preserve counts, so raw is again ``amount/V_static``: same
            concentration factor; bare ``X`` correct via the volume factor.
          * hOSU=false, ODE — the rescale ran, so raw is the live ``amount/V_live``
            and ``[X]`` is already correct, but the bare ``X`` amount must be
            ``raw·V_live`` (RoadRunner's amount), NOT the ``raw·V_static`` the
            volume factor gives → record a live-volume amount factor.

        The event-promoted compartment is hidden from species output (GH #71) but
        is emitted as a same-named OBSERVABLE, so V_live(t) is read from there.
        Neither path rescales the raw column. No-op for .net / static /
        event-resize-free models (empty map) and for the 3-D batch layout. GH #131.
        """
        emap = getattr(self._model, "_varvol_event_resize_map", None)
        if not emap:
            return
        species_names = result._species_names
        if not species_names or result._species.ndim != 2:
            return
        obs_names = result._observable_names
        if not obs_names:
            return
        stochastic = self._method in ("ssa", "psa")
        sp_idx = {n: i for i, n in enumerate(species_names)}
        obs_idx = {n: i for i, n in enumerate(obs_names)}
        conc_factor = result._varvol_conc_factor or {}
        amount_factor = result._varvol_amount_factor or {}
        changed = False
        for s_name, (comp_name, v_static, hosu) in emap.items():
            j = sp_idx.get(s_name)
            k = obs_idx.get(comp_name)
            # k missing ⇒ the compartment observable is unreported; without
            # V_live(t) leave the stale reporting rather than guess.
            if j is None or k is None:
                continue
            v_live = result._observables[:, k]
            # V_live(0) == V_static ⇒ factor(0) == 1, so t0 reporting is unchanged.
            if hosu or stochastic:
                with np.errstate(divide="ignore", invalid="ignore"):
                    conc_factor[j] = np.where(v_live != 0.0, v_static / v_live, 1.0)
            else:
                amount_factor[j] = v_live
            changed = True
        if changed:
            if conc_factor:
                result._varvol_conc_factor = conc_factor
            if amount_factor:
                result._varvol_amount_factor = amount_factor

    def _require_interactive_backend_support(self) -> None:
        """Reject high-level interactive flows for stateless XML backends."""
        if self._method not in ("nfsim", "rulemonkey"):
            return

        raise NotImplementedError(
            "Interactive simulation helpers are not supported for XML-backed "
            "network-free backends. Use run() for independent trajectories, "
            "or use the low-level session APIs on NfsimSimulator directly."
        )

    def _recreate_interactive_sim(self) -> None:
        """Rebuild the C++ backend simulator from the (possibly mutated) model.

        The persistent CvodeSimulator / SsaSimulator is constructed once from
        ``model._core``; after a parameter change some backend state must be
        rebuilt to pick it up (the SSA value-specialized propensity library
        bakes rate-constant values; the ODE path also drops any cached
        integrator workspace). ``intervene``, ``restore``, and the scan
        primitives all re-derive the backend through this single helper so the
        recreation rule lives in one place.
        """
        if self._method == "ode":
            from bngsim._bngsim_core import CvodeSimulator

            self._sim = CvodeSimulator(self._model._core)
        elif self._method in ("ssa", "psa"):
            from bngsim._bngsim_core import SsaSimulator

            self._sim = SsaSimulator(self._model._core)

    # ─── Run ────────────────────────────────────────────────────────

    def _resolve_max_step(self, max_step: float | None) -> float | None:
        """Resolve the effective integrator step bound (GH #88).

        An explicit ``max_step`` wins (a value ``<= 0`` disables the bound,
        returning ``None``). Otherwise fall back to the per-model bound the
        SBML loader derived for a periodic floor()/modulo dosing schedule, if
        any. ``None`` means leave the step unconstrained.
        """
        if max_step is not None:
            return float(max_step) if max_step > 0.0 else None
        pd = getattr(self._model, "_periodic_disc_max_step", None)
        return float(pd) if pd is not None and pd > 0.0 else None

    def _run_ode_with_jacobian_fallback(self, times, opts):
        """Run the CVODE integration, falling back to the finite-difference
        Jacobian if the analytical Jacobian fails (GH #176).

        ``jacobian="auto"`` (the default) is a *bet*: an analytical Jacobian is a
        strict speedup where it integrates, but it is not guaranteed to. A rate
        law that is genuinely discontinuous in a state variable — e.g.
        l-type-calcium-channel-dynamics' ``v_rec = if((-70+V)<-20, 0.5, 0.05)``
        with the state ``V`` asymptotically approaching the threshold 50 at
        t≈25 — has an exact derivative that omits the jump, so the analytical
        Jacobian cannot warn CVODE's implicit corrector about the step. The BDF
        predictor overshoots the discontinuity, the corrector meets an
        unanticipated jump, the local error test fails repeatedly and the step
        collapses to hmin (flag=-3). The finite-difference Jacobian instead
        straddles the step and supplies a regularizing slope, which is why FD and
        legacy run_network (always FD) integrate the same model cleanly.

        So under ``auto`` we honour the meaning of "auto": try the analytical
        Jacobian, and on a solver failure transparently retry once with the FD
        Jacobian (which ``opts.jacobian="fd"`` selects even when analytical terms
        are attached to the model). An explicit ``jacobian="analytical"`` is the
        user's deliberate choice and is *not* second-guessed — it surfaces the
        failure. ``"fd"`` / ``"jax"`` never had analytical terms to fall back
        from. The compiled-codegen Jacobian path is excluded: its derivative is
        baked into the ``.so`` and is not re-selectable at run time.
        """
        eligible = (
            self._jacobian == "auto"
            and not self._codegen_so_path
            and bool(getattr(self._model._core, "analytical_jacobian_complete", False))
        )
        if not eligible:
            return self._sim.run(times, opts)
        if self._ode_jacobian_fell_back:
            # A prior run on this Simulator already proved the analytical attempt
            # is doomed for this model — go straight to FD, no wasted attempt.
            opts.jacobian = "fd"
            return self._sim.run(times, opts)
        try:
            return self._sim.run(times, opts)
        except RuntimeError as e:
            logger.warning(
                "GH#176 analytical Jacobian: CVODE integration failed (%s); "
                "retrying with the finite-difference Jacobian. The rate law is "
                "likely discontinuous in a state variable (e.g. an if() whose "
                "condition crosses a threshold), which the exact Jacobian cannot "
                "represent. Pass jacobian='fd' to skip this attempt, or "
                "jacobian='analytical' to surface the failure.",
                e,
            )
            opts.jacobian = "fd"
            result = self._sim.run(times, opts)
            # Only memoize once FD has actually succeeded — a model that fails on
            # both (genuinely unintegrable) keeps surfacing its error every run.
            self._ode_jacobian_fell_back = True
            return result

    def run(
        self,
        t_span: tuple[float, float] = (0.0, 100.0),
        n_points: int = 101,
        *,
        sample_times: list[float] | None = None,
        seed: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        max_step: float | None = None,
        timeout: float | None = None,
        steady_state: bool = False,
        steady_state_tol: float | None = None,
        carry_sensitivities: bool = False,
    ) -> Result:
        """Run a simulation.

        Parameters
        ----------
        t_span : tuple[float, float]
            ``(t_start, t_end)`` time interval.
        n_points : int
            Number of output time points (including t_start).
        sample_times : list[float], optional
            Explicit output time points. When provided, overrides
            ``t_span`` and ``n_points``. Must contain at least 3
            values. Values are sorted automatically.
        seed : int, optional
            Random seed for stochastic methods. When omitted (or
            ``None``), bngsim draws a fresh seed from system entropy
            so consecutive ``run()`` calls produce independent
            trajectories. Pass an explicit integer for reproducibility.
            The actual seed used is exposed via ``Result.seed``.
            Ignored for ``method="ode"``.
        rtol : float, optional
            Relative tolerance for ODE solver. Default ``1e-8``.
        atol : float, optional
            Absolute tolerance for ODE solver. Default ``1e-8``.
        max_steps : int, optional
            Max internal solver steps per output point.
            Default ``10000``.
        max_step : float, optional
            ODE-only. Upper bound on a single internal integrator step
            (time units). ``None`` (default) leaves the step
            unconstrained, except that a model loaded from SBML with a
            periodic ``floor()``/modulo dosing schedule auto-applies a
            bound that keeps the integrator from stepping over a narrow
            dose pulse (GH #88). Pass an explicit value to override that
            (or to bound any model); ``<= 0`` disables the bound.
        timeout : float, optional
            Wall-clock budget in seconds. When set (and positive), the
            simulator raises :class:`bngsim.SimulationTimeout` if
            elapsed wall-clock time exceeds this limit. ``None`` or
            ``<= 0`` disables the budget. Supported on every backend
            (ODE/SSA/PSA/NFsim/RuleMonkey); RuleMonkey polls every
            ~1024 SSA events via its upstream cancellation hook.
            Partial results are not attached to the timeout exception.
        steady_state : bool, optional
            ODE-only. When ``True``, the integrator checks
            ``||f(t,y)||_2 / n_species`` after recording each output
            point and stops once it falls below ``steady_state_tol``.
            The returned :class:`Result` is truncated to only the rows
            actually integrated (BNG2.pl ``simulate({steady_state=>1})``
            parity, i.e. ``run_network -c``). Default ``False``.
            ``Result.solver_stats["steady_state_reached"]`` reports
            whether the criterion fired before ``t_end``.
        steady_state_tol : float, optional
            Tolerance for the ``steady_state`` check above. ``None`` or
            ``<= 0`` falls back to ``atol`` (matching BNG2.pl, which
            reuses the integrator atol as the steady-state cutoff).
        carry_sensitivities : bool, optional
            ODE-only, pre-equilibration (GH #210, ADR-0052). When ``True``
            and this run continues a carried-over species state from a
            prior ``run()`` on the same persistent ``Simulator`` (a
            two-phase equilibrate-then-measure protocol with no reset
            between phases), the forward-sensitivity initial conditions
            ``yS(0)`` are seeded from the prior phase's final
            steady-state sensitivity ``dx_ss/dθ`` instead of a fresh
            start. This makes ``output_sensitivities()`` correct across
            the pre-equilibration boundary: the measurement phase's IC is
            ``x_ss(θ)``, so ``∂x(0)/∂θ`` is the equilibration
            sensitivity, not zero. Requires the equilibration phase to
            have been run on the same ``Simulator`` with the same
            ``sensitivity_params`` (and no reset). Requesting
            sensitivities on a carried-over state **without** this flag
            raises (no silent wrong derivatives); a fresh single run is
            unaffected. Default ``False``.

        Returns
        -------
        Result
            Simulation results with time, species, observables.

        Raises
        ------
        SimulationError
            If the solver fails.
        SimulationTimeout
            If ``timeout`` is set and the wall-clock budget is exceeded.
        StopConditionMet
            If a stop condition triggers (partial result attached).
        ValueError
            If t_span or n_points are invalid, or if output sensitivities
            were requested (``sensitivity_params`` / ``sensitivity_ic``,
            including the ``carry_sensitivities`` path) on a model that
            contains events. Events reinitialise the CVODE state
            discontinuously without a matching forward-sensitivity
            reinitialisation, so derivatives go silently stale at and after an
            event; bngsim refuses rather than return wrong numbers (GH #205).
            Discontinuity triggers (forcing pulses / piecewise-time dosing)
            do not jump state and are unaffected.
        """
        from bngsim._bngsim_core import TimeSpec

        times = TimeSpec()

        if sample_times is not None:
            sorted_times = sorted(float(t) for t in sample_times)
            if len(sorted_times) < 2:
                raise ValueError(
                    f"sample_times must contain at least 2 points, got {len(sorted_times)}"
                )
            times.sample_times = sorted_times
            times.t_start = sorted_times[0]
            times.t_end = sorted_times[-1]
            times.n_points = len(sorted_times)
            t_start = times.t_start
            t_end = times.t_end
            n_points = times.n_points
        else:
            t_start, t_end = t_span
            if t_end <= t_start:
                raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
            if n_points < 2:
                raise ValueError(f"n_points ({n_points}) must be >= 2")
            times.t_start = t_start
            times.t_end = t_end
            times.n_points = n_points

        # Normalize the timeout kwarg. None or non-positive disables the
        # wall-clock budget (C++ side reads 0.0 as inactive).
        timeout_seconds: float = 0.0
        if timeout is not None:
            timeout_seconds = float(timeout)
            if timeout_seconds < 0.0:
                raise ValueError(f"timeout must be non-negative or None, got {timeout!r}")

        if steady_state and self._method != "ode":
            raise ValueError(
                "steady_state=True is only supported for method='ode' "
                f"(got method='{self._method}'). BNG2.pl ties the steady_state "
                "early-stop to the CVODE integrator only."
            )
        ss_tol_value: float = 0.0
        if steady_state_tol is not None:
            ss_tol_value = float(steady_state_tol)
            if ss_tol_value < 0.0:
                raise ValueError(
                    f"steady_state_tol must be non-negative or None, got {steady_state_tol!r}"
                )

        # GH #205 — event-time output-sensitivity correctness. Events
        # reinitialise the CVODE state discontinuously but the forward-
        # sensitivity vectors are never reinitialised, so derivatives go
        # silently stale at/after an event. Refuse on every sensitivity entry
        # point (single-shot and the carry-over path below). This upgrades GH
        # #210's narrow carry-over warning to a unified hard raise.
        if self._sensitivity_params or self._sensitivity_ic:
            self._raise_if_event_sensitivities()

        # GH #210 — pre-equilibration / carry-over output sensitivities. Only
        # meaningful for the ODE forward-sensitivity path; validate early.
        if carry_sensitivities:
            if self._method != "ode":
                raise ValueError(
                    "carry_sensitivities=True is only supported for method='ode' "
                    f"(got method='{self._method}'). Pre-equilibration output "
                    "sensitivities ride the CVODES forward-sensitivity path (GH #210)."
                )
            if not self._sensitivity_params:
                raise ValueError(
                    "carry_sensitivities=True requires sensitivity_params on the "
                    "Simulator: there are no sensitivity columns to seed across the "
                    "pre-equilibration boundary (GH #210)."
                )

        # Resolve the run seed. Stochastic methods draw a fresh seed from entropy
        # when the caller omits one; the ODE path is deterministic except for
        # random tie-breaking among simultaneous equal-priority events (GH #242),
        # so it uses a FIXED default when unset (reproducible out of the box) and
        # honors an explicit seed for an independent event-ordering realization.
        used_seed: int
        if self._method == "ode":
            used_seed = _DEFAULT_EVENT_SEED if seed is None else int(seed)
        else:
            used_seed = _resolve_seed(seed)

        # The seed affects the result — and is worth surfacing / stamping — for any
        # stochastic method, and for an ODE model WITH events (it breaks equal-
        # priority event ties, GH #242). An event-free ODE run is fully
        # deterministic, so its seed is neither logged nor stamped.
        seed_is_meaningful = self._method != "ode" or self._model._core.n_events > 0

        if seed_is_meaningful:
            logger.info(
                "Running %s simulation: t=[%.3g, %.3g], n_points=%d, seed=%d",
                self._method.upper(),
                t_start,
                t_end,
                n_points,
                used_seed,
            )
        else:
            logger.info(
                "Running %s simulation: t=[%.3g, %.3g], n_points=%d",
                self._method.upper(),
                t_start,
                t_end,
                n_points,
            )

        core_result = None
        # The ∂x(0)/∂θ rows this run seeds, captured where they are built and
        # stamped onto the Result (issue #155). None ⇒ no parameter-sensitivity
        # request, so there is no `parameter` axis to describe.
        ic_seed: dict[str, dict[str, float]] | None = None
        try:
            if self._method == "ode":
                from bngsim._bngsim_core import SolverOptions

                # GH #209: warn once if a large model is about to run dense-only
                # purely because this install lacks KLU (not user-forced dense).
                self._maybe_warn_dense_fallback()

                opts = SolverOptions()
                opts.rtol = rtol if rtol is not None else self._rtol
                opts.atol = atol if atol is not None else self._atol
                opts.max_steps = max_steps if max_steps is not None else self._max_steps
                opts.jacobian = self._jacobian
                opts.force_dense_linear_solver = self._force_dense_linear_solver
                opts.force_sparse_linear_solver = self._force_sparse_linear_solver
                opts.timeout_seconds = timeout_seconds
                opts.steady_state = bool(steady_state)
                opts.steady_state_tol = ss_tol_value
                opts.carry_sensitivities = bool(carry_sensitivities)
                # Seed for random equal-priority event tie-breaking (GH #242).
                # Inert unless the model has simultaneous equal-priority events.
                opts.event_seed = used_seed
                eff_max_step = self._resolve_max_step(max_step)
                if eff_max_step is not None:
                    opts.max_step_size = eff_max_step
                if self._codegen_so_path:
                    opts.codegen_so_path = self._codegen_so_path
                if self._codegen_c_source:
                    opts.codegen_c_source = self._codegen_c_source

                # Pass the requested sensitivity parameter / IC species lists to CVODES.
                if self._sensitivity_params:
                    opts.set_sensitivity_params(self._sensitivity_params)
                    ic_seed = self._apply_ic_param_sens_seed(opts, self._model)
                    if carry_sensitivities:
                        # GH #210/#81: this run seeds from the PRIOR phase's
                        # dx/dθ and the engine discards the IC-parameter rows
                        # entirely — the carried state is not at the model's
                        # initial conditions. Reporting the rows just computed
                        # would describe a seed that was never used. There is no
                        # double-count to guard against here either: an `ic`
                        # axis across a carry boundary is refused outright.
                        ic_seed = None
                    self._apply_switch_time_sens(opts, self._model._core, t_start, t_end)
                    self._apply_event_time_sens(opts, self._model._core, t_start, t_end)
                if self._sensitivity_ic:
                    opts.set_sensitivity_ic(self._sensitivity_ic)
                if self._sensitivity_params or self._sensitivity_ic:
                    opts.set_sensitivity_method(self._sensitivity_method)
                    # Outside the parameter guard on purpose: a *state* crossing
                    # moves every column, initial conditions included (issue
                    # #150 / #144), so an IC-only request needs the jump too.
                    self._apply_state_switch_sens(opts, self._model._core)

                # Install the Python callback used for the JAX Jacobian path.
                if self._jacobian == "jax" and self._jax_jac_evaluator is not None:
                    jax_eval = self._jax_jac_evaluator
                    # Build contiguous param array from model
                    model_core = self._model._core
                    param_names = model_core.param_names
                    param_vals = np.array(
                        [model_core.get_param(n) for n in param_names],
                        dtype=np.float64,
                    )

                    def _jax_callback(t, y_arr):
                        """Python callback for CVODE Jacobian.

                        Called from C++ with GIL acquired.
                        Returns flat column-major Jacobian.
                        """
                        return jax_eval(
                            np.asarray(y_arr),
                            t,
                            param_vals,
                        )

                    opts.set_jax_jac_fn(_jax_callback)

                core_result = self._run_ode_with_jacobian_fallback(times, opts)
            elif self._method == "ssa":
                core_result = self._sim.run(times, used_seed, timeout_seconds)
            elif self._method == "psa":
                core_result = self._sim.run_psa(times, used_seed, self._poplevel, timeout_seconds)
            elif self._method == "nfsim" or self._method == "rulemonkey":
                core_result = self._sim.run(times, used_seed, timeout_seconds)
            else:
                raise ValueError(f"Unknown method: {self._method}")
        except SimulationTimeout:
            # Already a typed bngsim exception (raised via the C++ translator)
            # — pass through unchanged so callers can classify wall-clock
            # terminations distinctly from solver errors.
            raise
        except RuntimeError as e:
            raise SimulationError(f"Simulation failed: {e}") from e

        # Stamp the seed on the Result when it identifies the realization (any
        # stochastic method) or drives ODE equal-priority event tie-breaking
        # (GH #242). An event-free ODE run stays seed-less (Result.seed is None),
        # preserving the "ODE is deterministic" contract (test_ode_seed_is_none).
        result = self._stamp(
            Result(core_result),
            seed=used_seed if seed_is_meaningful else None,
            ic_seed=ic_seed,
        )

        # GH #198 — attach the expression output-sensitivity support map so a
        # selector for an unsupported global function raises the specific reason
        # (unsupported construct / deferred table function) rather than a bare
        # empty-block error. Only meaningful on a sensitivity run.
        if self._sensitivity_params or self._sensitivity_ic:
            result._expression_sens_support = self._expression_sens_support()

        # GH #110 — surface SSA literal-rate-law boundary events as one warning
        # each (filterable via bngsim.SsaBoundaryWarning). The structured counts
        # stay on result.ssa_diagnostics regardless of warning filters. No-op on
        # non-SSA backends (counts are zero there).
        self._warn_ssa_boundary(result)

        # Check stop conditions on the result
        if self._stop_conditions:
            self._check_stop_conditions(result)

        logger.info(
            "Simulation complete: %d steps, %d RHS evals",
            result.solver_stats.get("n_steps", 0),
            result.solver_stats.get("n_rhs_evals", 0),
        )

        return result

    # ─── Batch ──────────────────────────────────────────────────────

    def run_batch(
        self,
        t_span: tuple[float, float] = (0.0, 100.0),
        n_points: int = 101,
        *,
        params: Sequence[dict[str, float]] | None = None,
        seed: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        max_step: float | None = None,
        num_processors: int | None = None,
        squeeze: bool = False,
        timeout: float | None = None,
        steady_state: bool = False,
        steady_state_tol: float | None = None,
    ) -> list[Result] | Result:
        """Run a batch of simulations over parameter sets.

        For each parameter set:
        1. Clone the model (independent copy)
        2. Apply parameters via ``set_params``
        3. Reset species to initial conditions
        4. Run the simulation (GIL released during each run)
        5. Collect the result

        Parameters
        ----------
        t_span : tuple[float, float]
            ``(t_start, t_end)`` time interval for each simulation.
        n_points : int
            Number of output time points per simulation.
        params : sequence of dict[str, float]
            Parameter sets. Each dict maps parameter names to values.
        seed : int, optional
            Base random seed for stochastic methods. Simulation *i*
            uses ``base_seed + i``. When omitted (or ``None``),
            ``base_seed`` is drawn fresh from system entropy on each
            call so consecutive batches produce independent
            trajectories. The actual per-sim seed is exposed via
            ``Result.seed`` on each result.
        rtol : float, optional
            Relative tolerance for ODE solver.
        atol : float, optional
            Absolute tolerance for ODE solver.
        max_steps : int, optional
            Maximum internal solver steps per output point.
        num_processors : int, optional
            Number of threads for parallel execution. Default
            ``None`` (sequential). The GIL is released during
            each simulation, so threads parallelize effectively.
        squeeze : bool
            If ``True``, return a single Result with 3D arrays
            ``(n_sims, n_times, n_cols)`` instead of a list.
        steady_state : bool, optional
            ODE-only. When ``True``, every simulation in the batch stops
            early once ``||f(t,y)||_2 / n_species`` falls below
            ``steady_state_tol`` and its :class:`Result` is truncated to
            the rows actually integrated (BNG2.pl
            ``simulate({steady_state=>1})`` / ``run_network -c`` parity,
            applied per parameter point). Default ``False``. Because each
            point truncates independently, the per-Result row counts may
            differ; use ``squeeze=False`` (the default) when mixing
            steady-state early-stop with heterogeneous equilibration times.
        steady_state_tol : float, optional
            Tolerance for the ``steady_state`` check above. ``None`` or
            ``<= 0`` falls back to ``atol`` (matching BNG2.pl).

        Returns
        -------
        list[Result] or Result
            One Result per parameter set (list), or a single
            squeezed Result with 3D arrays if ``squeeze=True``.

        Raises
        ------
        SimulationError
            If any simulation fails.
        ValueError
            If params is empty or t_span/n_points are invalid.

        Examples
        --------
        >>> param_sets = [{"k1": v} for v in [0.1, 1.0, 10.0]]
        >>> results = sim.run_batch(
        ...     t_span=(0, 100), n_points=101,
        ...     params=param_sets, num_processors=4,
        ... )
        >>> len(results)
        3

        >>> batch = sim.run_batch(
        ...     t_span=(0, 100), n_points=101,
        ...     params=param_sets, squeeze=True,
        ... )
        >>> batch.species.shape
        (3, 101, n_species)
        """
        if params is None or len(params) == 0:
            raise ValueError("params must be a non-empty sequence of dicts")

        t_start, t_end = t_span
        if t_end <= t_start:
            raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
        if n_points < 2:
            raise ValueError(f"n_points ({n_points}) must be >= 2")

        if steady_state and self._method != "ode":
            raise ValueError(
                "steady_state=True is only supported for method='ode' "
                f"(got method='{self._method}'). BNG2.pl ties the steady_state "
                "early-stop to the CVODE integrator only."
            )
        if steady_state and squeeze:
            raise ValueError(
                "run_batch(steady_state=True, squeeze=True) is not supported: "
                "each parameter point truncates to its own equilibration row "
                "count, so the per-Result trajectories cannot be stacked into a "
                "single 3D array. Use squeeze=False (the default)."
            )
        ss_tol_value: float = 0.0
        if steady_state_tol is not None:
            ss_tol_value = float(steady_state_tol)
            if ss_tol_value < 0.0:
                raise ValueError(
                    f"steady_state_tol must be non-negative or None, got {steady_state_tol!r}"
                )

        # GH #203/#205 — a sensitivity-configured Simulator now computes per-row
        # output sensitivities in the batch (the ODE path carries sensitivity_params
        # through to each clone). Sensitivities are unsupported across event-time
        # discontinuities, so refuse the whole batch up front (model-structural
        # check, hoisted out of the per-row loop) rather than return stale
        # derivatives — same policy as single-shot run().
        if self._sensitivity_params or self._sensitivity_ic:
            self._raise_if_event_sensitivities()

        n_sims = len(params)
        logger.info(
            "Starting batch: %d simulations, num_processors=%s",
            n_sims,
            num_processors or "sequential",
        )

        effective_rtol = rtol if rtol is not None else self._rtol
        effective_atol = atol if atol is not None else self._atol
        effective_max_steps = max_steps if max_steps is not None else self._max_steps
        # Integrator step bound (GH #88): an explicit max_step, else the
        # per-model periodic-dosing bound. None ⇒ unconstrained. Resolved once
        # for the whole batch (the loader's bound is schedule-structural, not
        # per-parameter-point).
        effective_max_step = self._resolve_max_step(max_step)
        # Per-simulation wall-clock budget. None / non-positive disables.
        # Applied independently to each sim (not a shared batch budget).
        effective_timeout: float = 0.0
        if timeout is not None:
            effective_timeout = float(timeout)
            if effective_timeout < 0.0:
                raise ValueError(f"timeout must be non-negative or None, got {timeout!r}")

        # Resolve the base seed once per batch. ODE doesn't use a seed,
        # but resolving anyway keeps the per-sim derivation deterministic
        # for any future hybrid path; the value is simply ignored when
        # method='ode' on the per-sim path.
        base_seed = _resolve_seed(seed) if self._method != "ode" else 0

        def _run_one(i: int) -> Result:
            """Run simulation i (thread-safe, GIL released)."""
            return self._run_single_batch(
                i,
                params[i],
                t_span,
                n_points,
                base_seed,
                effective_rtol,
                effective_atol,
                effective_max_steps,
                effective_timeout,
                steady_state=bool(steady_state),
                steady_state_tol=ss_tol_value,
                max_step=effective_max_step,
            )

        if num_processors is not None and num_processors > 1:
            # Parallel execution via ThreadPoolExecutor
            # GIL is released during C++ simulation, so threads
            # provide real parallelism.
            with ThreadPoolExecutor(max_workers=num_processors) as executor:
                futures = [executor.submit(_run_one, i) for i in range(n_sims)]
                results = []
                for i, future in enumerate(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        raise SimulationError(f"Batch simulation {i} failed: {e}") from e
        else:
            # Sequential execution
            results = [_run_one(i) for i in range(n_sims)]

        logger.info("Batch complete: %d results", len(results))

        if squeeze:
            return self._stamp(Result.squeeze(results))
        return [self._stamp(r) for r in results]

    def run_replicates(
        self,
        n_replicates: int,
        t_span: tuple[float, float] = (0.0, 100.0),
        n_points: int = 101,
        *,
        seed: int | None = None,
        timeout: float | None = None,
        num_processors: int | None = None,
        squeeze: bool = False,
    ) -> list[Result] | Result:
        """Run ``n_replicates`` stochastic replicates of the *same* model.

        Unlike :meth:`run_batch` — a parameter scan that clones the model per
        point — replicates share identical parameters and differ only in RNG
        seed, so this reuses a single simulator and calls ``reset()`` between
        replicates instead of cloning + reconstructing one each time. Reusing
        the simulator also reuses its cached SSA dependency graph (built once),
        so on low-activity models the per-replicate cost collapses to the actual
        trajectory work rather than the fixed clone + graph-rebuild overhead.

        Replicate *i* uses ``seed_base + i`` (``seed_base`` resolved once from
        ``seed``, or from system entropy when ``seed is None``; each value is
        exposed via the corresponding ``Result.seed``), matching the seed
        schedule :meth:`run_batch` uses across parameter points.

        Sequential execution (``num_processors`` ``None`` or ``1``) reuses this
        simulator directly. Parallel execution clones the model **once per worker
        thread** (not per replicate) for thread-safety, each thread reusing its
        clone across the replicates it handles.

        SSA/PSA only — ODE has no replicate concept; use :meth:`run_batch` for
        ODE parameter scans.

        Parameters
        ----------
        n_replicates : int
            Number of replicate trajectories (``>= 1``).
        t_span, n_points, seed, timeout, num_processors, squeeze
            As in :meth:`run` / :meth:`run_batch`.

        Returns
        -------
        list[Result] or Result
            One :class:`Result` per replicate, or a single squeezed Result with
            3D arrays ``(n_replicates, n_times, n_cols)`` when ``squeeze=True``.

        Examples
        --------
        >>> ssa = bngsim.Simulator(model, method="ssa")
        >>> reps = ssa.run_replicates(30, t_span=(0, 100), n_points=101, seed=0)
        >>> len(reps)
        30
        """
        if self._method not in ("ssa", "psa"):
            raise ValueError(
                "run_replicates is for stochastic methods (method='ssa' or "
                f"'psa'); got method={self._method!r}. Use run_batch for ODE "
                "parameter scans."
            )
        if n_replicates < 1:
            raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
        t_start, t_end = t_span
        if t_end <= t_start:
            raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
        if n_points < 2:
            raise ValueError(f"n_points ({n_points}) must be >= 2")

        eff_timeout: float = 0.0
        if timeout is not None:
            eff_timeout = float(timeout)
            if eff_timeout < 0.0:
                raise ValueError(f"timeout must be non-negative or None, got {timeout!r}")

        base_seed = _resolve_seed(seed)

        from bngsim._bngsim_core import SsaSimulator, TimeSpec

        def _make_times() -> Any:
            ts = TimeSpec()
            ts.t_start = t_start
            ts.t_end = t_end
            ts.n_points = n_points
            return ts

        def _run_one(sim: Any, model: Model, times: Any, i: int) -> Result:
            # reset() restores species to initial conditions and zeroes time;
            # the simulator (hence the cached dependency graph) is reused. A
            # fresh seed per replicate makes each trajectory independent.
            model.reset()
            used = base_seed + i
            if self._method == "psa":
                cr = sim.run_psa(times, used, self._poplevel, eff_timeout)
            else:
                cr = sim.run(times, used, eff_timeout)
            r = self._stamp(Result(cr), seed=used)
            self._warn_ssa_boundary(r)
            return r

        if num_processors is not None and num_processors > 1:
            # One clone per worker thread (not per replicate): thread-local
            # state keyed off this call's fresh `local` object, so each thread
            # builds its clone + simulator once and reuses them across its chunk.
            local = threading.local()

            def _worker(i: int) -> Result:
                sim = getattr(local, "sim", None)
                if sim is None:
                    local.model = self._model.clone()
                    local.sim = SsaSimulator(local.model._core)
                    local.times = _make_times()
                    sim = local.sim
                return _run_one(sim, local.model, local.times, i)

            results: list[Result] = []
            with ThreadPoolExecutor(max_workers=num_processors) as executor:
                futures = [executor.submit(_worker, i) for i in range(n_replicates)]
                for i, future in enumerate(futures):
                    try:
                        results.append(future.result())
                    except SimulationTimeout:
                        raise
                    except Exception as e:
                        raise SimulationError(f"Replicate {i} failed: {e}") from e
        else:
            times = _make_times()
            results = [_run_one(self._sim, self._model, times, i) for i in range(n_replicates)]

        if squeeze:
            return self._stamp(Result.squeeze(results))
        return results

    # ─── Parameter scan / bifurcation (issue #11) ──────────────────

    @staticmethod
    def _resolve_scan_values(
        par_scan_vals: Sequence[float] | None,
        par_min: float | None,
        par_max: float | None,
        n_scan_pts: int | None,
        log_scale: bool,
    ) -> list[float]:
        """Resolve the ordered list of scanned parameter values.

        Accepts either an explicit ``par_scan_vals`` list or the BNG
        ``par_min`` / ``par_max`` / ``n_scan_pts`` (+ ``log_scale``) triple.
        ``n_scan_pts`` is the number of points, inclusive of both endpoints
        (``np.linspace`` / ``np.geomspace`` convention).
        """
        if par_scan_vals is not None:
            vals = [float(v) for v in par_scan_vals]
            if not vals:
                raise ValueError("par_scan_vals must be a non-empty sequence")
            return vals
        if par_min is None or par_max is None or n_scan_pts is None:
            raise ValueError(
                "Provide either par_scan_vals, or all of par_min, par_max, and n_scan_pts."
            )
        n = int(n_scan_pts)
        if n < 1:
            raise ValueError(f"n_scan_pts must be >= 1, got {n}")
        if n == 1:
            return [float(par_min)]
        if log_scale:
            if par_min <= 0.0 or par_max <= 0.0:
                raise ValueError(
                    "log_scale=True requires positive par_min and par_max "
                    f"(got par_min={par_min}, par_max={par_max})."
                )
            return [float(v) for v in np.geomspace(par_min, par_max, n)]
        return [float(v) for v in np.linspace(par_min, par_max, n)]

    # ─── Scan-boundary sensitivity carry (issue #81) ────────────────────────

    def _resolve_scan_sens_carry(
        self, parameter: str, reset_conc: bool, reset_to: str | None
    ) -> tuple[np.ndarray | None, list[str]]:
        """Resolve the per-point ``∂x(0)/∂θ`` seed for a sensitivity scan.

        A scan point's initial condition is the reset snapshot (or the previous
        point's end-state), *not* the model's seed initial conditions. After a
        pre-equilibration that snapshot is ``x_ss(θ)``, so the point's
        forward-sensitivity seed is the equilibration's ``dx_ss/dθ``; seeding it
        fresh would discard the equilibration's contribution and give derivatives
        that are wrong rather than merely approximate (issue #81, and the reason
        the scan used to refuse outright).

        Returns ``(seed, names)`` — the ``(n_species, n_params)`` matrix each
        point seeds from and the parameter names labeling its columns — or
        ``(None, [])`` when this Simulator has no sensitivity columns and the scan
        takes the plain path. Raises when a sensitivity scan cannot be made
        correct, rather than returning a wrong gradient.
        """
        if not (self._sensitivity_params or self._sensitivity_ic):
            return None, []

        names = list(self._sensitivity_params)
        if self._sensitivity_ic:
            # The IC axis (∂y/∂y_k(0)) is meaningless across the boundary: the
            # snapshot each point starts from is no longer the model's IC, so e_k
            # is not a seed. Same posture as the single-run carry path in C++.
            raise ValueError(
                "parameter_scan / bifurcate do not support sensitivity_ic "
                "(initial-condition axis) sensitivities: each point starts from a "
                "snapshot rather than the model's initial conditions, so ∂y/∂y_k(0) "
                "has no meaning across that boundary. Use sensitivity_params only, "
                "or run_batch (which resets each point to the seed) for the IC axis."
            )
        if parameter in names:
            # The scan overwrites the parameter per point, so the snapshot's
            # ∂x/∂(that parameter) was taken at a different value of the same
            # symbol — the two cannot be composed into one derivative.
            raise ValueError(
                f"parameter_scan / bifurcate cannot scan {parameter!r}: it is also a "
                "sensitivity_params entry, and each point overwrites it. The carried "
                f"∂x/∂{parameter} was accumulated at the pre-scan value, so composing it "
                "with a point that pins the parameter to a scan value would mix two "
                "values of one symbol. Scan a parameter you are not differentiating "
                "(the usual dose / condition), or use run_batch for a sweep of a "
                "differentiated parameter (each row re-seeds from the model seed)."
            )

        core = self._model._core
        if reset_conc and reset_to is not None:
            seeded = self._model._named_sens_seeds.get(str(reset_to))
            seed, seed_names = seeded if seeded is not None else (None, [])
            source = f"the saved state {reset_to!r}"
            fix = (
                f"save_concentrations({reset_to!r}) while the equilibrated dx/dθ is "
                "pending (i.e. right after the equilibration run)"
            )
        else:
            seed = core.pending_sensitivity_seed() if core.has_pending_sensitivity_seed else None
            seed_names = list(core.pending_sensitivity_seed_param_names)
            source = "the model's live state at scan invocation"
            fix = (
                "run the equilibration phase on this same Simulator with these "
                "sensitivity_params and no reset in between"
            )
        if seed is None or list(seed_names) != names:
            have = ", ".join(seed_names) if seed_names else "(none)"
            raise ValueError(
                "parameter_scan / bifurcate can only carry output sensitivities "
                f"into a scan when {source} carries a matching forward-sensitivity "
                "matrix dx/dθ: each point starts from that state, so its ∂x(0)/∂θ is "
                "the carried derivative and re-seeding it fresh would be wrong, not "
                f"approximate (issue #81). Requested sensitivity_params: {names}; "
                f"carried columns: {have}. To fix, {fix} — note that a plain "
                "(non-sensitivity) run, set_concentration(), or set_state() in "
                "between drops the carried dx/dθ. For a sweep whose points start "
                "from the model's own seed initial conditions, use run_batch."
            )
        return np.array(seed, dtype=np.float64), names

    def _pending_scan_sens_seed(self, names: list[str], point_index: int) -> np.ndarray:
        """The dx/dθ a continuation point inherits from the previous point's run."""
        core = self._model._core
        if not core.has_pending_sensitivity_seed or list(
            core.pending_sensitivity_seed_param_names
        ) != list(names):
            raise SimulationError(
                f"Continuation scan point {point_index} lost its carried "
                "forward-sensitivity matrix dx/dθ: point "
                f"{point_index - 1}'s run left no matching seed (columns "
                f"{list(core.pending_sensitivity_seed_param_names)} vs requested "
                f"{list(names)}). A continuation scan (reset_conc=False / bifurcate) "
                "carries each point's ∂x(0)/∂θ from the previous point, so the chain "
                "cannot be broken mid-scan (issue #81)."
            )
        return np.array(core.pending_sensitivity_seed(), dtype=np.float64)

    def _install_scan_sens_seed(self, seed: np.ndarray, names: list[str]) -> None:
        """Make ``seed`` the pending ∂x(0)/∂θ for the next per-point run."""
        core = self._model._core
        core.set_pending_sensitivity_seed(seed, names)
        # The point's initial condition is a θ-dependent snapshot, so fresh-start
        # seeding must stay refused: keep the carry-over arm engaged.
        core.ic_state_dirty = True

    def _run_on_point_with_ic_sens(
        self,
        on_point: Callable[[Model, float], None],
        value: float,
        seed: np.ndarray,
        names: list[str],
    ) -> np.ndarray:
        """Run an ``on_point`` hook and resolve the point's ``∂x(0)/∂θ`` around it.

        ``on_point`` exists to apply coupled ``setConcentration`` overrides — the
        ligand dose of a dose-response scan — so the initial condition a scan point
        actually starts from is *the hook's output*, and the seed the point needs is
        ``d(post-hook x)/dθ``. For the species the hook assigns, that is **not** the
        carried equilibration derivative: it is whatever the hook's own arithmetic
        implies (issue #111).

        Each row of the point's seed is resolved by the most specific thing
        available:

        1. **The hook installed a whole seed** (``_core.set_pending_sensitivity_seed``
           after its writes) — taken verbatim, the pre-#111 escape hatch.
        2. **The row is declared** via :meth:`Model.declare_ic_sensitivity` — used
           verbatim, and not probed.
        3. **The hook assigned this species** — the row is *measured* through the
           hook (see :meth:`_probe_on_point_ic_sens`): a literal dose comes back
           ``0``, a dose computed from a differentiated parameter comes back with
           its true ``∂x_k(0)/∂θ``, and an increment of the carried pool comes back
           with the carried row plus the dose's derivative.
        4. **Otherwise** — the carried row, bit-exact (never routed through the
           measurement, which would only add noise to a known-exact number).

        A hook that moves a differentiated parameter's *value* is refused: the
        carried dx/dθ was accumulated at the pre-hook value, so the two cannot be
        composed (the same argument as scanning a differentiated parameter).
        """
        model = self._model
        core = model._core
        params_before = [model.get_param(p) for p in names]
        base_state = np.asarray(model.get_state(), dtype=np.float64)

        model._declared_ic_sens.clear()
        model._ic_write_log = set()
        try:
            on_point(model, float(value))
        finally:
            written = model._ic_write_log or set()
            model._ic_write_log = None
        declared = {k: dict(v) for k, v in model._declared_ic_sens.items()}
        model._declared_ic_sens.clear()

        for name, before in zip(names, params_before, strict=True):
            if model.get_param(name) != before:
                raise ValueError(
                    f"on_point changed the sensitivity parameter {name!r} "
                    f"({before!r} → {model.get_param(name)!r}). The carried "
                    f"∂x/∂{name} was accumulated at the pre-hook value, so a point that "
                    "overwrites it would mix two values of one symbol (issue #81). "
                    "Use on_point for the point's conditions (doses, non-fitted "
                    "parameters) only."
                )

        # (1) The hook took over the whole matrix.
        if core.has_pending_sensitivity_seed and list(
            core.pending_sensitivity_seed_param_names
        ) == list(names):
            resolved = np.array(core.pending_sensitivity_seed(), dtype=np.float64)
            return self._apply_declared_ic_sens(resolved, names, declared)

        post_state = np.asarray(model.get_state(), dtype=np.float64)
        rows = self._assigned_ic_rows(written, base_state, post_state)
        # A species whose row is declared needs no measurement.
        declared_rows = {model.species_names.index(sp) for sp in declared}
        probe_rows = sorted(rows - declared_rows)

        resolved = seed.copy()
        if probe_rows:
            measured = self._probe_on_point_ic_sens(
                on_point, value, seed, names, base_state, post_state, params_before, probe_rows
            )
            resolved[probe_rows, :] = measured[probe_rows, :]
        return self._apply_declared_ic_sens(resolved, names, declared)

    def _assigned_ic_rows(
        self, written: set[str], base_state: np.ndarray, post_state: np.ndarray
    ) -> set[int]:
        """Species indices whose initial condition an ``on_point`` hook assigned.

        The write log is exact for the documented API (``set_concentration`` /
        ``set_state`` / ``reset`` / ``restore_concentrations`` on the :class:`Model`);
        the value diff is the belt-and-braces half that also catches a hook writing
        through ``model._core`` directly. Their union is what gets measured rather
        than carried.
        """
        names = self._model.species_names
        if "*" in written:
            return set(range(len(names)))
        rows = {names.index(sp) for sp in written if sp in names}
        rows.update(int(i) for i in np.nonzero(post_state != base_state)[0])
        return rows

    def _apply_declared_ic_sens(
        self, seed: np.ndarray, names: list[str], declared: dict[str, dict[str, float]]
    ) -> np.ndarray:
        """Overwrite the rows a hook declared via ``declare_ic_sensitivity``."""
        if not declared:
            return seed
        species = self._model.species_names
        col = {name: j for j, name in enumerate(names)}
        for sp, row in declared.items():
            i = species.index(sp)
            # A declared row is fully specified: unnamed params are 0 (the hook
            # said what this initial condition depends on).
            seed[i, :] = 0.0
            for p, d in row.items():
                if p in col:
                    seed[i, col[p]] = d
        return seed

    def _probe_on_point_ic_sens(
        self,
        on_point: Callable[[Model, float], None],
        value: float,
        seed: np.ndarray,
        names: list[str],
        base_state: np.ndarray,
        post_state: np.ndarray,
        nominal_params: list[float],
        probe_rows: list[int],
    ) -> np.ndarray:
        """Measure ``d(post-hook x)/dθ`` through an ``on_point`` hook (issue #111).

        The hook is a map ``H: (x, θ) → x'`` from the point's pre-hook state to the
        state it actually integrates from, so the chain rule wants

            ``dx'/dθ_i = ∂H/∂θ_i + (∂H/∂x)·s_i``

        with ``s_i`` the carried ``∂x/∂θ_i``. Both terms are obtained by calling the
        hook at perturbed inputs — a central difference in θ_i for the first, and a
        central difference along the carried column for the second (which is what
        makes an *increment* of the carried pool come out right rather than being
        mistaken for a literal). Each term is computed at two step sizes and the
        two must agree, or the hook is not differentiable here (a dose rounded to
        whole molecules, say) and this refuses rather than reporting a difference
        quotient of a jump.

        The hook is therefore invoked several extra times per point, on the live
        model with a perturbed input, and must be a deterministic function of
        ``(model, value)``. The model is restored — state, parameters, and a final
        nominal hook call — before this returns, so what the point integrates is
        exactly what the nominal call produced. Declaring a row with
        :meth:`Model.declare_ic_sensitivity` skips its measurement entirely, which
        is the way out for an expensive or side-effecting hook.
        """
        model = self._model
        ns = base_state.size
        measured = np.zeros((ns, len(names)), dtype=np.float64)
        x_scale = max(float(np.max(np.abs(base_state))), 1.0)
        # Determinism first, on the nominal inputs: a hook that answers differently
        # every call would otherwise fail below as "not differentiable", which is
        # the wrong diagnosis for the wrong hook.
        self._replay_on_point(on_point, value, base_state, names, nominal_params)
        again = np.asarray(model.get_state(), dtype=np.float64)
        if not np.array_equal(again, post_state):
            raise ValueError(
                "on_point is not a deterministic function of (model, value): re-running "
                "it on the same inputs produced a different initial condition. bngsim "
                "calls the hook at perturbed inputs to measure ∂x(0)/∂θ for the initial "
                "conditions it assigns (issue #111), which requires determinism. Make "
                "the hook depend only on its arguments, or declare its rows with "
                "model.declare_ic_sensitivity() inside it to skip the measurement."
            )
        for j, name in enumerate(names):
            theta = nominal_params[j]
            col = np.asarray(seed[:, j], dtype=np.float64)
            # ∂H/∂θ_i — perturb the parameter, hold the pre-hook state.
            h_t = _IC_SENS_PROBE_EPS * (abs(theta) if theta != 0.0 else 1.0)
            d_theta = self._ic_sens_central(
                on_point, value, base_state, name, theta, h_t, None, probe_rows, f"∂/∂{name}"
            )
            # (∂H/∂x)·s_i — perturb the state along the carried column, scaled so
            # the state perturbation stays small relative to the state itself
            # however large the carried derivative is. Zero column ⇒ zero term.
            col_scale = float(np.max(np.abs(col)))
            d_state = np.zeros(ns, dtype=np.float64)
            if col_scale > 0.0:
                h_x = _IC_SENS_PROBE_EPS * x_scale / col_scale
                d_state = self._ic_sens_central(
                    on_point,
                    value,
                    base_state,
                    name,
                    theta,
                    h_x,
                    col,
                    probe_rows,
                    f"∂/∂x along ∂x/∂{name}",
                )
            measured[:, j] = d_theta + d_state

        # Re-establish the nominal point: restore the inputs and re-run the hook,
        # so a parameter the hook set *from* a perturbed θ is recomputed at the
        # nominal one (restoring `post_state` alone would leave it perturbed).
        self._replay_on_point(on_point, value, base_state, names, nominal_params)
        return measured

    def _replay_on_point(
        self,
        on_point: Callable[[Model, float], None],
        value: float,
        base_state: np.ndarray,
        names: list[str],
        nominal_params: list[float],
    ) -> None:
        """Re-run the hook on the nominal inputs, leaving no declarations behind."""
        model = self._model
        model.set_state(base_state)
        for name, val in zip(names, nominal_params, strict=True):
            model.set_param(name, val)
        model._declared_ic_sens.clear()
        on_point(model, float(value))
        model._declared_ic_sens.clear()

    def _ic_sens_central(
        self,
        on_point: Callable[[Model, float], None],
        value: float,
        base_state: np.ndarray,
        name: str,
        theta: float,
        h: float,
        direction: np.ndarray | None,
        probe_rows: list[int],
        what: str,
    ) -> np.ndarray:
        """One central difference of the hook, validated across two step sizes."""
        fine, fine_mag = self._ic_sens_diff(on_point, value, base_state, name, theta, h, direction)
        coarse, coarse_mag = self._ic_sens_diff(
            on_point, value, base_state, name, theta, h * _IC_SENS_PROBE_COARSE, direction
        )
        for i in probe_rows:
            # Roundoff floor of the difference quotient at the finer step.
            floor = 8.0 * float(np.finfo(float).eps) * max(fine_mag[i], coarse_mag[i]) / (2.0 * h)
            tol = _IC_SENS_PROBE_TOL * max(abs(fine[i]), abs(coarse[i])) + floor
            if abs(fine[i] - coarse[i]) > tol:
                raise ValueError(
                    f"on_point's initial condition for {self._model.species_names[i]!r} is "
                    f"not differentiable in {name!r} at this scan point: the measured "
                    f"{what} is {fine[i]:.6g} at step {h:.3g} but {coarse[i]:.6g} at step "
                    f"{h * _IC_SENS_PROBE_COARSE:.3g}, so the hook has a jump or a kink "
                    "there (a dose rounded to whole molecules does this). bngsim will "
                    "not report a difference quotient of a jump as a derivative "
                    "(issue #111) — declare the row explicitly with "
                    "model.declare_ic_sensitivity({species: {param: value}}) inside the "
                    "hook (an intentionally θ-independent assignment declares 0)."
                )
        return fine

    def _ic_sens_diff(
        self,
        on_point: Callable[[Model, float], None],
        value: float,
        base_state: np.ndarray,
        name: str,
        theta: float,
        h: float,
        direction: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(central difference, |value| scale per row)`` for one step size."""
        model = self._model
        out = []
        for sign in (1.0, -1.0):
            if direction is None:
                model.set_state(base_state)
                model.set_param(name, theta + sign * h)
            else:
                model.set_state(base_state + (sign * h) * direction)
                model.set_param(name, theta)
            try:
                on_point(model, float(value))
            except Exception as e:
                raise ValueError(
                    f"on_point raised while bngsim measured d(initial condition)/d{name} "
                    f"through it ({type(e).__name__}: {e}). The hook is called with "
                    "perturbed inputs to obtain ∂x(0)/∂θ for the initial conditions it "
                    "assigns (issue #111), so it must tolerate a nearby parameter value "
                    "and state — or declare its rows with model.declare_ic_sensitivity() "
                    "inside the hook, which skips the measurement."
                ) from e
            finally:
                # The parameter goes back to nominal even when the hook raised, so
                # the scan's own restore never has to guess what a probe left behind.
                model.set_param(name, theta)
            model._declared_ic_sens.clear()
            out.append(np.asarray(model.get_state(), dtype=np.float64))
        plus, minus = out
        return (plus - minus) / (2.0 * h), np.maximum(np.abs(plus), np.abs(minus))

    def _capture_carryover_state(self) -> tuple[np.ndarray | None, list[str], bool]:
        """Snapshot the model's carry-over sensitivity state (issue #81)."""
        core = self._model._core
        seed = (
            np.array(core.pending_sensitivity_seed(), dtype=np.float64)
            if core.has_pending_sensitivity_seed
            else None
        )
        return seed, list(core.pending_sensitivity_seed_param_names), bool(core.ic_state_dirty)

    def _restore_carryover_state(
        self, snapshot: tuple[np.ndarray | None, list[str], bool]
    ) -> None:
        """Put back what :meth:`_capture_carryover_state` captured."""
        seed, names, dirty = snapshot
        core = self._model._core
        if seed is None:
            core.set_pending_sensitivity_seed(np.zeros((0, 0), dtype=np.float64), [])
        else:
            core.set_pending_sensitivity_seed(seed, names)
        core.ic_state_dirty = dirty

    def parameter_scan(
        self,
        parameter: str,
        par_scan_vals: Sequence[float] | None = None,
        *,
        par_min: float | None = None,
        par_max: float | None = None,
        n_scan_pts: int | None = None,
        log_scale: bool = False,
        t_span: tuple[float, float] = (0.0, 100.0),
        n_points: int = 101,
        reset_conc: bool = True,
        reset_to: str | None = None,
        on_point: Callable[[Model, float], None] | None = None,
        seed: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        max_step: float | None = None,
        timeout: float | None = None,
        steady_state: bool = False,
        steady_state_tol: float | None = None,
        squeeze: bool = False,
    ) -> list[Result] | Result:
        """Sweep one parameter, running a simulation per value (BNG ``parameter_scan``).

        This is the native scan primitive whose ``reset_conc`` semantics match
        BNG2.pl (issue #11). Unlike a hand-rolled loop that re-derives every
        point's species from the ``.net`` seed initializers, each point here
        resets to the state **at scan invocation** (or to a named snapshot) —
        so a pre-equilibrate → intervene → scan protocol carries its
        post-intervention state into the sweep faithfully, instead of discarding
        it.

        For each scanned value the model is: reset to the snapshot (when
        ``reset_conc``), assigned the scanned ``parameter``, passed through the
        optional ``on_point`` hook (for coupled ``setConcentration`` overrides
        that track the scanned parameter), then integrated over ``t_span``.

        Supported on the stateful model-backed backends (ODE / SSA / PSA) only;
        the XML network-free backends have no in-process scan path.

        Parameters
        ----------
        parameter : str
            Name of the parameter to scan. Must exist in the model.
        par_scan_vals : sequence of float, optional
            Explicit values to scan. Mutually exclusive with the
            ``par_min`` / ``par_max`` / ``n_scan_pts`` triple.
        par_min, par_max : float, optional
            Endpoints of a generated scan range (inclusive).
        n_scan_pts : int, optional
            Number of points in the generated range (>= 1).
        log_scale : bool
            When generating a range, space points geometrically (requires
            positive endpoints) rather than linearly. Default ``False``.
        t_span : tuple[float, float]
            ``(t_start, t_end)`` for each per-point simulation.
        n_points : int
            Output time points per simulation (including ``t_start``).
        reset_conc : bool
            BNG ``reset_conc``. When ``True`` (default), every point resets to
            the snapshot before applying the scanned parameter — points are
            independent. When ``False``, points are *not* reset between values;
            each continues from the previous point's end-state (a continuation
            scan — see :meth:`bifurcate`).
        reset_to : str, optional
            Name of a saved concentration snapshot
            (``Model.save_concentrations(label=...)``) to reset each point to.
            When ``None`` (default), the reset target is the model's live state
            captured at the moment this method is called. Only consulted when
            ``reset_conc=True``.
        on_point : callable, optional
            ``on_point(model, value)`` invoked after the reset + scanned-parameter
            assignment and before integration, for each point. Use it to apply
            coupled ``setConcentration`` overrides whose value tracks the scanned
            parameter (e.g. a ligand species whose count is
            ``value * NA * Vecf``) — the model-specific bookkeeping the primitive
            cannot infer on its own.
        seed : int, optional
            Base seed for stochastic methods; point *i* uses ``seed_base + i``
            (drawn fresh from entropy when ``None``). Ignored for ODE.
        rtol, atol, max_steps, max_step, timeout, steady_state, steady_state_tol
            Per-simulation solver options, forwarded to :meth:`run`.
        squeeze : bool
            When ``True``, stack the per-point results into a single
            :class:`Result` with 3-D arrays (like :meth:`run_batch`); otherwise
            return a list of per-point results.

        Returns
        -------
        list[Result] or Result
            One :class:`Result` per scanned value (in order), each carrying
            ``custom_attrs["scan_parameter"]`` and ``custom_attrs["scan_value"]``.
            A squeezed :class:`Result` when ``squeeze=True``.

        Notes
        -----
        The persistent model + backend simulator are left as they were before
        the call: the scanned parameter, the reset-target concentrations and the
        carried sensitivity state are restored afterward, so a
        :class:`Simulator` can be scanned repeatedly (and the returned
        trajectories, not the live model, are the product).

        **Output sensitivities across the scan boundary (issue #81).** On a
        :class:`Simulator` built with ``sensitivity_params``, each point's
        forward-sensitivity seed ``∂x(0)/∂θ`` is the *carried* ``dx/dθ`` of the
        state it starts from — the pre-equilibration's steady-state sensitivity —
        because that state, not the model's seed initial conditions, is the
        point's initial condition. So a scan is run with sensitivities only when
        that state carries a matching ``dx/dθ``:

        * ``reset_conc=True`` — every point restores the reset target's state
          *and* its ``dx/dθ``, then integrates with
          ``carry_sensitivities=True``. The target must carry one: the live state
          at invocation does after an equilibration run on this same
          ``Simulator``, and a ``reset_to`` snapshot does when it was saved while
          that ``dx/dθ`` was pending.
        * ``reset_conc=False`` (:meth:`bifurcate`) — each point continues the
          previous point's state *and* its ``dx/dθ``, so the whole continuation
          is one differentiable protocol.

        An ``on_point`` hook assigns the initial condition its point starts from,
        so for the species it writes the seed is *its* derivative, not the carried
        one (issue #111). Row by row, the most specific thing available wins: a
        row the hook installed wholesale, then one declared with
        :meth:`Model.declare_ic_sensitivity`, then one **measured through the
        hook** (bngsim calls the hook at perturbed inputs, so a literal dose
        measures ``0``, a dose computed from a differentiated parameter measures
        its true derivative, and an increment of the carried pool measures the
        carried row plus the dose's), and otherwise the carried row unchanged.
        Measuring calls the hook several extra times per point, so it must be a
        deterministic function of ``(model, value)``; declaring a row skips its
        measurement.

        The scanned parameter must not be a ``sensitivity_params`` entry (each
        point overwrites it, which cannot be composed with the derivative carried
        into the point), ``sensitivity_ic`` is not supported across the boundary,
        an ``on_point`` hook may not move a differentiated parameter, and a hook
        whose assigned initial condition is not differentiable in a parameter (a
        dose rounded to whole molecules) must declare that row rather than have a
        difference quotient of the jump reported as a derivative. Every one of
        those is a raise, never a silently-reseeded gradient. For a sweep whose
        points start from the model's own seed initial conditions, use
        :meth:`run_batch`.
        """
        self._require_interactive_backend_support()
        if steady_state and self._method != "ode":
            raise ValueError(
                "steady_state=True is only supported for method='ode' "
                f"(got method='{self._method}')."
            )
        values = self._resolve_scan_values(par_scan_vals, par_min, par_max, n_scan_pts, log_scale)

        # Validate the parameter and capture its pre-scan value so the model can
        # be left pristine (get_param raises a clean ParameterError if unknown).
        original_value = self._model.get_param(parameter)

        # Determine — and validate — the per-point reset target up front.
        use_named = reset_to is not None
        if use_named and not self._model.has_saved_concentrations(reset_to):
            known = ", ".join(self._model.saved_concentration_labels) or "(none)"
            raise ValueError(
                f"reset_to={reset_to!r}: no saved concentration state by that "
                f"name. Saved states: {known}. Call save_concentrations({reset_to!r}) "
                "before the scan."
            )
        # Snapshot the live state at invocation as the reset target (and as the
        # post-scan restore point). Captured even for reset_conc=False so the
        # model can be rewound afterward.
        invocation_state = self._model.get_state()

        # Each point's forward-sensitivity seed ∂x(0)/∂θ (issue #81). A scan point
        # starts from the snapshot, not the model seed, so re-seeding it fresh
        # would be wrong; ``None`` = this Simulator has no sensitivity columns and
        # the scan runs the plain path. Resolved after the reset-target validation
        # above so a bad ``reset_to`` still reports itself, not a missing seed.
        carry_seed, carry_names = self._resolve_scan_sens_carry(parameter, reset_conc, reset_to)
        # ...and the carry-over state to put back afterward, so a scan still
        # leaves the model exactly as it found it (the runs below both consume and
        # overwrite the pending seed).
        invocation_sens = self._capture_carryover_state()

        def _reset_point() -> None:
            # Rewind the scanned parameter FIRST (issue #79). When it names a
            # species initial condition, the previous point left that species'
            # IC baseline at the previous point's value; restoring only the live
            # concentrations below would leave the two disagreeing, and the
            # `set_param` that follows would decline to move a species it no
            # longer sees sitting on its baseline — so every point after the
            # first would silently run at the invocation dose. A no-op for the
            # parameter that names no IC, which is nearly all of them.
            self._model.set_param(parameter, original_value)
            if use_named:
                self._model.restore_concentrations(reset_to)
            else:
                self._model.set_state(invocation_state)
            # Issue #141: the restore above declares this point a fresh start from
            # the invocation baseline, but neither set_state nor
            # restore_concentrations clears GH #210's carry-over marker — and the
            # IC rebuild now reads that marker to refuse to overwrite a state some
            # run advanced. Left set by the PREVIOUS point's run, it would freeze
            # an IC-naming parameter at the invocation dose from point 1 on: the
            # #79 trap again, one level down. Only the reset_conc=True path lands
            # here; the pre-equilibration carry (#81) is reset_conc=False and must
            # keep its marker.
            self._model._core.ic_state_dirty = False

        base_seed = _resolve_seed(seed) if self._method != "ode" else 0

        results: list[Result] = []
        try:
            for i, value in enumerate(values):
                if reset_conc:
                    _reset_point()
                # A continuation point (reset_conc=False) starts from the previous
                # point's end-state, whose dx/dθ that run left pending — read it
                # now, before anything below can clear it.
                pt_sens_seed = carry_seed
                if carry_seed is not None and not reset_conc and i > 0:
                    pt_sens_seed = self._pending_scan_sens_seed(carry_names, i)
                if pt_sens_seed is not None:
                    # Restore it now: the per-point reset above (set_state /
                    # restore_concentrations) drops the pending seed, and the
                    # on_point hook needs to see the same state the run will.
                    self._install_scan_sens_seed(pt_sens_seed, carry_names)
                self._model.set_param(parameter, float(value))
                if on_point is not None:
                    if pt_sens_seed is None:
                        on_point(self._model, float(value))
                    else:
                        # The hook assigns the initial condition this point starts
                        # from, so ∂x(0)/∂θ for the species it writes is its own
                        # arithmetic's — declared, measured through the hook, or
                        # carried, row by row (issue #111).
                        pt_sens_seed = self._run_on_point_with_ic_sens(
                            on_point, float(value), pt_sens_seed, carry_names
                        )
                        self._install_scan_sens_seed(pt_sens_seed, carry_names)
                # Rebuild the backend so the scanned parameter (and any on_point
                # rate-constant change) is picked up; run() then seeds from the
                # model's current live concentrations.
                self._recreate_interactive_sim()
                point_seed = None if self._method == "ode" else base_seed + i
                result = self.run(
                    t_span=t_span,
                    n_points=n_points,
                    seed=point_seed,
                    rtol=rtol,
                    atol=atol,
                    max_steps=max_steps,
                    max_step=max_step,
                    timeout=timeout,
                    steady_state=steady_state,
                    steady_state_tol=steady_state_tol,
                    carry_sensitivities=pt_sens_seed is not None,
                )
                result.custom_attrs["scan_parameter"] = parameter
                result.custom_attrs["scan_value"] = float(value)
                results.append(result)
        finally:
            # Leave the persistent model + simulator as we found them.
            self._model.set_param(parameter, original_value)
            self._model.set_state(invocation_state)
            self._restore_carryover_state(invocation_sens)
            self._recreate_interactive_sim()

        if squeeze:
            return self._stamp(Result.squeeze(results))
        return results

    def bifurcate(
        self,
        parameter: str,
        par_scan_vals: Sequence[float] | None = None,
        *,
        par_min: float | None = None,
        par_max: float | None = None,
        n_scan_pts: int | None = None,
        log_scale: bool = False,
        t_span: tuple[float, float] = (0.0, 100.0),
        n_points: int = 101,
        seed: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        max_step: float | None = None,
        timeout: float | None = None,
        steady_state: bool = False,
        steady_state_tol: float | None = None,
        squeeze: bool = False,
    ) -> list[Result] | Result:
        """Continuation scan of one parameter (BNG ``bifurcate``, ``reset_conc=0``).

        A :meth:`parameter_scan` sibling that does **not** reset concentrations
        between points: each point continues from the previous point's
        end-state, so the sweep traces a branch of steady states as the
        parameter is stepped. Sweep ``par_scan_vals`` up then down (two calls)
        to expose hysteresis. The first point starts from the model's live state
        at invocation.

        Accepts the same arguments as :meth:`parameter_scan` except
        ``reset_conc`` (pinned to ``False``), ``reset_to``, and ``on_point``
        (which pertain to the per-point reset that continuation omits). See
        :meth:`parameter_scan` for the shared parameters.
        """
        return self.parameter_scan(
            parameter,
            par_scan_vals,
            par_min=par_min,
            par_max=par_max,
            n_scan_pts=n_scan_pts,
            log_scale=log_scale,
            t_span=t_span,
            n_points=n_points,
            reset_conc=False,
            seed=seed,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            max_step=max_step,
            timeout=timeout,
            steady_state=steady_state,
            steady_state_tol=steady_state_tol,
            squeeze=squeeze,
        )

    def _run_single_batch(
        self,
        index: int,
        pset: dict[str, float],
        t_span: tuple[float, float],
        n_points: int,
        base_seed: int,
        rtol: float,
        atol: float,
        max_steps: int,
        timeout_seconds: float = 0.0,
        steady_state: bool = False,
        steady_state_tol: float = 0.0,
        max_step: float | None = None,
    ) -> Result:
        """Run a single simulation in a batch (thread-safe)."""
        from bngsim._bngsim_core import TimeSpec

        clone = self._model.clone()
        clone.set_params(pset)
        clone.reset()

        times = TimeSpec()
        times.t_start = t_span[0]
        times.t_end = t_span[1]
        times.n_points = n_points

        # This row's ∂x(0)/∂θ rows (issue #155); None for an IC-only request,
        # which has no `parameter` axis to describe.
        row_ic_seed: dict[str, dict[str, float]] | None = None
        try:
            if self._method == "ode":
                from bngsim._bngsim_core import (
                    CvodeSimulator,
                    SolverOptions,
                )

                sim: Any = CvodeSimulator(clone._core)
                opts = SolverOptions()
                opts.rtol = rtol
                opts.atol = atol
                opts.max_steps = max_steps
                opts.jacobian = self._jacobian
                opts.timeout_seconds = timeout_seconds
                opts.steady_state = steady_state
                opts.steady_state_tol = steady_state_tol
                if max_step is not None and max_step > 0.0:
                    opts.max_step_size = max_step
                # GH #203 — the HPC contract: every batch row reuses the ONE
                # compiled artifact this Simulator already prepared (large/codegen
                # models would otherwise run interpreted per row, reusing nothing),
                # and a Simulator built with sensitivity_params yields the full
                # per-row output-sensitivity tensor — mirroring single-shot run()
                # (codegen + sensitivity option-building) so a θ-matrix batch and a
                # per-θ loop of run() produce identical results.
                if self._codegen_so_path:
                    opts.codegen_so_path = self._codegen_so_path
                if self._codegen_c_source:
                    opts.codegen_c_source = self._codegen_c_source
                if self._sensitivity_params:
                    opts.set_sensitivity_params(self._sensitivity_params)
                    # Seed ∂x_i(0)/∂p from the CLONE's params (this row's point):
                    # a nonlinear derived IC (e.g. Rtot = R0*scale) has a
                    # param-dependent coefficient, so it must track set_params.
                    # That is also why each row's Result carries its OWN matrix.
                    row_ic_seed = self._apply_ic_param_sens_seed(opts, clone)
                    # Likewise the switch times: this row's t0/sigma set where the
                    # crossings are, so they must be detected on the clone.
                    self._apply_switch_time_sens(opts, clone._core, t_span[0], t_span[1])
                    self._apply_event_time_sens(opts, clone._core, t_span[0], t_span[1])
                if self._sensitivity_ic:
                    opts.set_sensitivity_ic(self._sensitivity_ic)
                if self._sensitivity_params or self._sensitivity_ic:
                    opts.set_sensitivity_method(self._sensitivity_method)
                    # See the note at the single-shot site: keyed on "any
                    # sensitivity at all", not on a parameter request.
                    self._apply_state_switch_sens(opts, clone._core)
                core_result = sim.run(times, opts)

            elif self._method in ("ssa", "psa"):
                from bngsim._bngsim_core import SsaSimulator

                sim = SsaSimulator(clone._core)
                if self._method == "psa":
                    core_result = sim.run_psa(
                        times, base_seed + index, self._poplevel, timeout_seconds
                    )
                else:
                    core_result = sim.run(times, base_seed + index, timeout_seconds)
            else:
                raise ValueError(f"Unknown method: {self._method}")
        except SimulationTimeout:
            raise
        except RuntimeError as e:
            raise SimulationError(f"Batch simulation {index} failed: {e}") from e

        result = Result(core_result)
        if self._method != "ode":
            result._seed = base_seed + index
        # GH #203/#198 — on a sensitivity batch, carry the expression
        # output-sensitivity support map so an unsupported expression selector
        # raises its specific reason on each row's Result, exactly as run() does.
        if self._method == "ode" and (self._sensitivity_params or self._sensitivity_ic):
            result._expression_sens_support = self._expression_sens_support()
            result._ic_sensitivity_seed = row_ic_seed
        return result

    # ─── Parallel sensitivity ───────────────────────────────────────

    def compute_all_sensitivities(
        self,
        t_span: tuple[float, float] = (0.0, 100.0),
        n_points: int = 101,
        *,
        params: list[str] | None = None,
        chunk_size: int = 2,
        n_workers: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
    ) -> Result:
        """Compute full sensitivity tensor via parallel chunked CVODES jobs.

        Splits ``Np`` parameters into ``⌈Np/chunk_size⌉`` independent
        CVODES forward-sensitivity jobs, runs them in parallel via
        ``model.clone()`` + ``ThreadPoolExecutor`` (GIL released during
        C++ CVODE integration), and stitches the partial sensitivity
        arrays into a complete ``(n_times, n_species, n_params)`` tensor.

        Benchmarks showed that 2-parameter sensitivity chunks add only
        ~1.2× overhead for large models (593–1281 species). With ``⌈Np/2⌉``
        parallel jobs, the full sensitivity tensor costs ~1.2× wall-clock
        of a plain ODE solve — making gradients nearly free with cores.

        Parameters
        ----------
        t_span : tuple[float, float]
            ``(t_start, t_end)`` time interval.
        n_points : int
            Number of output time points (including t_start).
        params : list[str], optional
            Parameter names to compute sensitivities for.
            Default: all model parameters.
        chunk_size : int
            Number of sensitivity parameters per CVODES job.
            Default 2, which benchmarking found to work well for large
            models because 2-parameter chunks add only ~1.2× overhead.
        n_workers : int, optional
            Number of parallel threads. Default:
            ``min(⌈Np/chunk_size⌉, os.cpu_count())``.
            Set to 1 for serial execution (debugging/profiling).
        rtol : float, optional
            Relative tolerance for ODE solver.
        atol : float, optional
            Absolute tolerance for ODE solver.
        max_steps : int, optional
            Max internal solver steps per output point.

        Returns
        -------
        Result
            Simulation result with full ``sensitivities`` tensor
            of shape ``(n_times, n_species, n_params)``.
            The ``sensitivity_params`` attribute lists all parameter
            names in the order they appear in the tensor.
            Species trajectories are from the first chunk's ODE solve
            (all chunks produce identical trajectories since they share
            the same model and parameters).

        Raises
        ------
        ValueError
            If method is not 'ode', or params contains unknown names, or the
            analytical sensitivity RHS every chunk needs cannot be built —
            ``codegen=False`` / ``BNGSIM_NO_CODEGEN``, no codegen backend, or
            rate laws that do not differentiate to closed form. That is the same
            hard requirement ``Simulator(..., sensitivity_params=...)`` carries
            (GH #214): the interpreted finite-difference sensitivity path is not
            reliable at tight tolerances, so it is refused rather than used.
        SimulationError
            If any chunk simulation fails.

        Examples
        --------
        >>> model = bngsim.Model.from_net("model.net")
        >>> sim = bngsim.Simulator(model, method="ode")
        >>> result = sim.compute_all_sensitivities(
        ...     t_span=(0, 100), n_points=101,
        ...     chunk_size=2, n_workers=8,
        ... )
        >>> result.sensitivities.shape  # (101, n_species, n_params)
        (101, 149, 40)
        >>> fim = result.fisher_information(sigma=0.1)
        >>> grad = result.gradient(
        ...     lambda species, time: np.sum((species - data)**2)
        ... )

        Notes
        -----
        **Architecture**: Each chunk clones the model (deep copy,
        thread-safe), creates a fresh ``CvodeSimulator``, runs
        CVODES with ``chunk_size`` sensitivity parameters, and
        returns its partial ``(n_times, n_species, chunk_size)``
        sensitivity tensor. The main thread stitches these along
        axis 2 (the parameter axis).

        **Why ThreadPoolExecutor works**: The GIL is released during
        C++ CVODE integration (``py::call_guard<py::gil_scoped_release>``),
        so threads achieve true parallelism for the compute-intensive
        portion. Python overhead is negligible (model clone + setup).

        **Optimal chunk_size**: Benchmarks show that ``chunk_size=2``
        minimizes per-chunk overhead while keeping
        thread count manageable. ``chunk_size=1`` works but doubles
        the number of threads needed.
        """
        import math
        import os

        if self._method != "ode":
            raise ValueError(
                "compute_all_sensitivities() is only supported "
                f"for method='ode', not method='{self._method}'."
            )

        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        # Determine parameter list
        all_param_names = self._model.param_names
        if params is not None:
            # Validate requested params exist
            known = set(all_param_names)
            unknown = set(params) - known
            if unknown:
                raise ValueError(
                    f"Unknown parameter(s): {sorted(unknown)}. Known: {sorted(known)}"
                )
            target_params = list(params)
            # Issue #164 — an explicit ask gets a hard answer.
            self._raise_if_compartment_size_params(target_params)
        else:
            # ...but "every parameter" is a request for everything computable,
            # and on an SBML model that list leads with the compartments. Raising
            # would make this method unusable on any SBML model for the sake of
            # columns nobody asked for by name; silently including them would put
            # a wrong column in the tensor. So drop them and say so (issue #164).
            skipped = sorted(set(all_param_names) & self._compartment_size_params())
            target_params = [p for p in all_param_names if p not in set(skipped)]
            if skipped:
                warnings.warn(
                    f"compute_all_sensitivities: skipping compartment size(s) {skipped} "
                    f"— an SBML compartment's value is folded into load-time constants "
                    f"the sensitivity RHS does not differentiate, so its column would be "
                    f"wrong in both directions (issue #164). The returned tensor has "
                    f"{len(target_params)} parameter columns; result.sensitivity_params "
                    f"lists them. Pass params=[...] to make the refusal explicit, or "
                    f"finite-difference through a rebuild "
                    f"(Model.from_sbml(..., compartment_sizes={{...}}) at V ± h).",
                    stacklevel=2,
                )

        n_params = len(target_params)
        if n_params == 0:
            raise ValueError("No parameters to compute sensitivities for.")

        # GH #205 — events: allowed only for the subclasses whose ∂t*/∂p is
        # known (fixed-time; issue #49 thresholds; issue #144 state-dependent
        # triggers). Classified against this call's actual target parameters.
        self._raise_if_event_sensitivities(target_params)

        # Split into chunks
        n_chunks = math.ceil(n_params / chunk_size)
        chunks: list[list[str]] = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_params)
            chunks.append(target_params[start:end])

        # Determine worker count
        if n_workers is None:
            cpu_count = os.cpu_count() or 1
            n_workers = min(n_chunks, cpu_count)
        n_workers = max(1, n_workers)

        logger.info(
            "compute_all_sensitivities: %d params, %d chunks (chunk_size=%d), %d workers",
            n_params,
            n_chunks,
            chunk_size,
            n_workers,
        )

        # GH #204 — expression output-sensitivities (GH #198) are evaluated by
        # the compiled codegen output-sensitivity ABI, which each chunk inherits
        # through ``self._codegen_so_path`` / ``self._codegen_c_source``. When
        # this Simulator was built WITHOUT sensitivity_params (the normal
        # compute_all_sensitivities entry point), the constructor's sensitivity
        # auto-codegen never fired, so the chunks would run interpreted and every
        # chunk's expression-sensitivity block would come back empty. Trigger the
        # SAME auto-codegen the single-shot sensitivity path uses. The helper is a
        # no-op if a codegen .so/JIT source is already attached (e.g. a large-model
        # or explicit codegen build), so this never double-compiles.
        #
        # GH #62 — that attach is UNCONDITIONAL, exactly as in the constructor.
        # It used to be gated on ``n_functions > 0`` ("expression-free models stay
        # on the interpreted path unchanged"), which is right for the *expression*
        # output block but wrong for the state-sensitivity RHS every chunk needs:
        # a function-free model never reached the helper, so each chunk finite-
        # differenced the whole sensitivity RHS interpreted. That is the path GH
        # #214 retired for the single-shot solve — unreliable at tight tolerances,
        # and its FD noise degrades step-size control on top of costing extra RHS
        # evaluations. One chunk over the same parameters as the coupled
        # ``Simulator(sensitivity_params=...)`` solve therefore ran up to 49x
        # slower on large function-free networks (fceri_fyn 768 s vs 15.7 s,
        # 40202 internal steps vs 656), making parameter sharding a pessimization
        # exactly where it should help. Gating what is expression-
        # specific stays below: only the output-sens *rebuild* looks at
        # ``n_functions``, because ``_codegen_emit_flags`` emits the GH #198
        # evaluator only when the model has functions — for a function-free model
        # the source is byte-identical with or without ``_want_output_sens``, so
        # there is nothing to rebuild and an inherited plain-RHS codegen is
        # already the right artifact.
        #
        # GH #205 — the GH #198 output-sensitivity evaluator is emitted only when
        # ``_want_output_sens`` is set (both the .net and model-based codegen paths
        # gate on it), which the constructor does for sensitivity_params runs but
        # this entry point (built without them) does not. compute_all_sensitivities
        # always wants the output blocks, so run the re-prep dance that marks the
        # flag and regenerates a shadowing plain-RHS artifact. Shared with
        # steady_state(), which has the same constructor-vs-method-argument gap
        # (issue #75) — see _prepare_output_sens_codegen for the wrinkles.
        self._prepare_output_sens_codegen()

        # Effective solver options
        effective_rtol = rtol if rtol is not None else self._rtol
        effective_atol = atol if atol is not None else self._atol
        effective_max_steps = max_steps if max_steps is not None else self._max_steps

        def _run_chunk(chunk_idx: int) -> Result:
            """Run one sensitivity chunk (thread-safe)."""
            chunk_params = chunks[chunk_idx]
            return self._run_sensitivity_chunk(
                chunk_params,
                t_span,
                n_points,
                effective_rtol,
                effective_atol,
                effective_max_steps,
            )

        # Run chunks (parallel or serial)
        if n_workers > 1 and n_chunks > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_run_chunk, i) for i in range(n_chunks)]
                chunk_results: list[Result] = []
                for i, future in enumerate(futures):
                    try:
                        chunk_results.append(future.result())
                    except Exception as e:
                        raise SimulationError(
                            f"Sensitivity chunk {i} (params={chunks[i]}) failed: {e}"
                        ) from e
        else:
            chunk_results = [_run_chunk(i) for i in range(n_chunks)]

        # Stitch sensitivity tensors along param axis
        stitched = self._stitch_sensitivity_results(chunk_results, target_params)
        # Carry the #198 support map so an unsupported expression selector raises
        # the specific reason on the stitched result too.
        stitched._expression_sens_support = self._expression_sens_support()
        return stitched

    def _run_sensitivity_chunk(
        self,
        sens_params: list[str],
        t_span: tuple[float, float],
        n_points: int,
        rtol: float,
        atol: float,
        max_steps: int,
    ) -> Result:
        """Run a single sensitivity chunk (thread-safe).

        Clones the model, creates a fresh CvodeSimulator, runs
        CVODES with the given sensitivity parameters, and returns
        a Result with partial sensitivity data.
        """
        from bngsim._bngsim_core import (
            CvodeSimulator,
            SolverOptions,
            TimeSpec,
        )

        clone = self._model.clone()

        sim = CvodeSimulator(clone._core)

        ts = TimeSpec()
        ts.t_start = t_span[0]
        ts.t_end = t_span[1]
        ts.n_points = n_points

        opts = SolverOptions()
        opts.rtol = rtol
        opts.atol = atol
        opts.max_steps = max_steps
        opts.jacobian = self._jacobian
        if self._codegen_so_path:
            opts.codegen_so_path = self._codegen_so_path
        if self._codegen_c_source:
            opts.codegen_c_source = self._codegen_c_source
        opts.set_sensitivity_params(sens_params)
        # Report this chunk's OWN subset of ∂x(0)/∂θ; the stitch takes the union
        # over chunks, which is the full requested set (issue #155).
        chunk_ic_seed = self._apply_ic_param_sens_seed(opts, clone, sens_params)
        # This chunk differentiates only `sens_params`, so the ∂t*/∂p columns
        # must be built against that subset rather than the full request.
        self._apply_switch_time_sens(opts, clone._core, t_span[0], t_span[1], sens_params)
        self._apply_event_time_sens(opts, clone._core, t_span[0], t_span[1], sens_params)
        self._apply_state_switch_sens(opts, clone._core)
        opts.set_sensitivity_method(self._sensitivity_method)

        try:
            core_result = sim.run(ts, opts)
        except RuntimeError as e:
            raise SimulationError(f"Sensitivity chunk failed (params={sens_params}): {e}") from e

        return self._stamp(Result(core_result), ic_seed=chunk_ic_seed)

    @staticmethod
    def _stitch_sensitivity_results(
        chunk_results: list[Result],
        all_param_names: list[str],
    ) -> Result:
        """Stitch partial sensitivity results into a full Result.

        Takes the ODE solution (time, species, observables) from the
        first chunk, and concatenates sensitivity tensors from all
        chunks along axis 2 (parameter axis).

        Parameters
        ----------
        chunk_results : list[Result]
            Results from each sensitivity chunk.
        all_param_names : list[str]
            Full ordered list of parameter names.

        Returns
        -------
        Result
            Combined result with full sensitivity tensor.
        """
        if not chunk_results:
            raise ValueError("No chunk results to stitch")

        # Use first chunk for ODE solution
        base = chunk_results[0]

        # Concatenate sensitivity tensors along param axis (axis=2)
        sens_parts = []
        for r in chunk_results:
            if r.has_sensitivities:
                sens_parts.append(r.sensitivities)
            else:
                raise SimulationError("Chunk result missing sensitivity data")

        full_sens = np.concatenate(sens_parts, axis=2)

        # Build combined result from raw arrays
        result = Result(
            core=None,
            _time=base._time.copy(),
            _species=base._species.copy(),
            _observables=base._observables.copy(),
            _expressions=base._expressions.copy()
            if base._expressions.size > 0
            else base._expressions,
            _species_names=base._species_names,
            _observable_names=base._observable_names,
            _expression_names=base._expression_names,
            _solver_stats=base._solver_stats,
        )
        # Inject full sensitivity tensor + param names
        result._sensitivities = full_sens
        result._sensitivity_params = list(all_param_names)

        # Issue #155: each chunk reported only its own parameter subset, so the
        # stitched matrix is their union — disjoint by construction, since the
        # chunks partition the parameter list.
        merged_ic_seed: dict[str, dict[str, float]] = {}
        for r in chunk_results:
            for sp, row in (r._ic_sensitivity_seed or {}).items():
                merged_ic_seed.setdefault(sp, {}).update(row)
        result._ic_sensitivity_seed = merged_ic_seed

        # GH #196/#197/#198 — carry the observable/expression *parameter* output
        # sensitivities through the same param-axis stitch. They are chunked on
        # the identical parameter axis as the species block, so they concatenate
        # along axis 2. Both blocks are populated at simulation time now (GH #197
        # observable runtime chain rule; GH #198 expression codegen evaluator), so
        # these are real concatenations whenever every chunk computed them. The IC
        # output blocks are not parameter-chunked, so they are left at their
        # __init__ default (empty), matching how the species IC block is handled in
        # this param-stitching path.
        def _concat_param_block(attr: str) -> np.ndarray:
            parts = [getattr(r, attr) for r in chunk_results]
            empty = [p.size == 0 for p in parts]
            if all(empty):
                # Legitimately empty for *every* chunk: an interpreted run (no
                # codegen) or a model with no expression outputs. Matches the
                # species IC block, which is not parameter-chunked either.
                return np.empty((0, 0, 0))
            if any(empty):
                # Some chunks computed this output block and some did not — a real
                # inconsistency (e.g. codegen attached for only part of the run),
                # not "nobody computed it". Be as loud as the species path (which
                # raises on a chunk missing its sensitivity tensor) rather than
                # silently dropping the block to (0, 0, 0).
                missing = [i for i, e in enumerate(empty) if e]
                raise SimulationError(
                    f"Inconsistent '{attr}' across sensitivity chunks: "
                    f"chunk(s) {missing} are missing this output-sensitivity block "
                    "while others computed it. All chunks must produce the same "
                    "output-sensitivity blocks (check codegen is enabled uniformly)."
                )
            return np.concatenate(parts, axis=2)

        result._observable_sensitivities = _concat_param_block("_observable_sensitivities")
        result._expression_sensitivities = _concat_param_block("_expression_sensitivities")

        # Propagate volume_factors from the chunk results (all chunks
        # came from the same model, so any non-None field is correct).
        for r in chunk_results:
            if r._species_volume_factors is not None:
                result._species_volume_factors = r._species_volume_factors
                break

        # GH #205 — carry the AR-species output-sensitivity redirect map (and
        # its blocked set) from a stamped chunk so species:<ar> selectors follow
        # the assignment expression on the stitched result too.
        result._ar_sens_map = base._ar_sens_map
        result._ar_sens_blocked = base._ar_sens_blocked

        return result

    # --- Steady-state solver ------------------------------------

    def steady_state(
        self,
        *,
        tol: float = 1e-9,
        max_time: float = 1e6,
        method: str = "integration",
        rtol=None,
        atol=None,
        max_steps=None,
        sensitivity_params=None,
        mask=None,
    ):
        """Find the steady state of the ODE system f(y) = 0.

        Solver methods:

        - ``"integration"`` (default): CVODE BDF integrated until the BNG2.pl
          parity criterion ``||f(y)||_2 / n_species < tol``
          (``run_network -c``).
        - ``"newton"``: two-tier integrate-first solver. The *same* CVODE burst
          carries the state into the physical root's basin, then KINSOL
          polishes. The polish is accepted only once it is *seed-stable*
          (agrees across two successively tighter bursts) **and** carries no
          eigenvalue in the right half-plane (issue #78, see
          :attr:`SteadyStateResult.root_stability`); otherwise integration
          continues. Seeding Newton at the raw initial condition instead can
          converge to a spurious root of ``f(y)=0`` the trajectory never
          reaches, or walk a species negative into ``NaN`` (GH #27) — hence the
          burst.
        - ``"kinsol"``: accepted alias for ``"newton"``.

        Both tiers factor the model's closed-form Jacobian when it has one and
        ``jacobian=`` asks for it (issue #127): ``CVodeSetJacFn`` on the march,
        ``KINSetJacFn`` on the polish, the latter projected onto the polish's own
        unknown set. ``jacobian="fd"`` pins the difference quotient instead — one
        RHS evaluation per unknown per Jacobian setup, which is what both tiers
        paid unconditionally before #127. ``ss.solver_jacobian_source`` reports
        which matrix was factored. The same option still selects how the
        *stability certificate* and ``dY_ss/dp`` build theirs.

        Because ``"newton"`` integrates first and only then polishes, it is
        ``"integration"`` plus extra work: across six published dose-response
        models it cost 1.4-3.9x more wall clock (GH #28), which is why
        ``"integration"`` is the default. Two things still argue for
        ``"newton"``:

        - **A much tighter root.** The polish lands near a residual of ~1e-13
          where integration stops the moment it crosses ``tol`` (~1e-9) — worth
          having when the steady state feeds a stiff downstream solve.
        - **A tight ``max_time`` budget.** Newton reaches ``tol`` from a
          *looser* burst than integration needs on its own, so when ``max_time``
          is cut well below the default it can converge where integration runs
          out of time. At the default ``max_time=1e6`` this does not happen on
          any model in the benchmark corpus, but at ``max_time=1e3`` several
          models flip.

        Parameters
        ----------
        tol : float
            Convergence tolerance on ``||f(y)||_2 / n_species``. Default 1e-9.
        max_time : float
            Max integration time for the integration path. Default 1e6.
        method : str
            ``"integration"`` (default), ``"newton"``, or ``"kinsol"``
            (alias for ``"newton"``).
        sensitivity_params : list[str], optional
            Parameter names for dY_ss/dp sensitivity. Requires code generation,
            exactly as :meth:`run` and :meth:`compute_all_sensitivities` do —
            see the sensitivity note below.
        mask : array-like of bool, or sequence of str, optional
            Which species the convergence test covers (issue #74). A boolean
            array of length ``n_species`` (``True`` = keep), or the species
            names to keep. ``None`` (default) tests every species, which is the
            BNG2.pl parity criterion. See the accumulator note below.

        Returns
        -------
        SteadyStateResult

        Notes
        -----
        **Write-only accumulators (issue #74).** A species some reaction
        produces and none consumes — a ``degraded`` / ``produced`` /
        ``secreted`` pool counting cumulative flux, a common BNGL idiom — has a
        constant non-zero derivative forever. ``||f(y)||_2 / n_species`` then has
        a floor above ``tol`` and this method reports ``converged=False`` however
        far it integrates, even when every other species is settled to 1e-10.
        ``mask`` restricts the test to the subspace that *does* have a steady
        state; :meth:`Model.is_pure_sink` finds the accumulators structurally, so
        the recipe needs no hand-listed species::

            ss = sim.steady_state(method="newton", mask=~model.is_pure_sink())

        Everything still integrates — the excluded species' equations stay in
        the RHS and their trajectories come back in ``ss.concentrations``. What
        the mask restricts is the residual norm (over ``n_included``, so ``tol``
        keeps its meaning), the KINSOL polish's unknown set, and the
        ``dY_ss/dp`` linear system; those last two have to follow, because an
        accumulator contributes a structurally zero Jacobian *column* and makes
        both systems singular at every seed. Excluded species come back with a
        NaN ``dY_ss/dp`` row: no steady value, no steady-state gradient.

        When a solve fails, ``ss.unconverged_pure_sinks`` names any accumulator
        that was in the test and is carrying flux — that is a structural floor,
        not a slow tail, and no ``max_time`` will move it. Detection is not a
        convergence verdict, though: ``A -> B`` with nothing feeding ``A`` has a
        textbook pure sink and converges anyway, because the flux dies out. The
        time-course early stop, ``run(steady_state=True)``, keeps the
        unrestricted criterion and takes no mask.

        **Codegen (issue #63).** The solve runs whatever RHS this Simulator's
        :attr:`codegen_backend` reports — the compiled ``.so``, the MIR JIT, or
        the ExprTk interpreter. Until #63 the steady-state path read no codegen
        option at all, so a Simulator built with ``codegen=True`` still solved
        interpreted; ``ss.rhs_backend`` now reports which backend actually ran.

        **Sensitivity requires codegen.** ``dY_ss/dp = -J⁻¹·(∂f/∂p)`` used to
        build *both* factors from finite differences. It now prefers the model's
        analytical Jacobian and the analytical ``∂f/∂p`` the codegen sensitivity
        RHS emits, and — like :meth:`run` and :meth:`compute_all_sensitivities`
        since GH #214 — refuses rather than silently degrading when codegen is
        unavailable. ``ss.sens_jacobian_source`` / ``ss.sens_dfdp_source`` report
        which path each factor took. Since GH #67 a Functional model whose rate
        laws are smooth algebra has an analytical ``∂f/∂p`` here too — it is the
        same emitted ``bngsim_codegen_sens_rhs``, read at ``yS = 0``. What still
        differences that factor, with a warning, is Michaelis-Menten and the
        Functional laws carrying a condition or a non-smooth builtin.

        The expression block ``d(func)/dp`` prefers compiled code the same way
        (issue #75): it is evaluated by ``bngsim_codegen_output_sens``, the GH #198
        chain rule a CVODES forward-sensitivity ``run()`` already uses, fed the
        solved ``dY_ss/dp`` columns — so a steady-state gradient and a long-run
        one now come from the same evaluator. Finite differences remain the
        per-function fallback for what that evaluator declines (a table function,
        a non-smooth construct, a whole-model ``rateOf`` decline);
        ``ss.sens_output_source`` reports ``"codegen"``, ``"mixed"``, or
        ``"finite-difference"``. The *observable* block is an exact linear
        projection through the group factors and never differences anything.
        """
        if self._method != "ode":
            raise ValueError(
                f"steady_state() is only supported for method='ode', not method='{self._method}'."
            )

        # Issue #74 — resolve the convergence-test subspace before anything
        # expensive runs, so a bad mask is an immediate error rather than a solve
        # that quietly tested the wrong species.
        mask_selector = _resolve_ss_mask(mask, self._model)

        # GH #205 — dY_ss/dp on event models: allowed only for the subclasses
        # whose ∂t*/∂p is known (see _raise_if_event_sensitivities),
        # classified against this call's requested sensitivity_params.
        if sensitivity_params:
            self._raise_if_event_sensitivities(sensitivity_params)
            # Issue #164 — and the same refusal the constructor applies: a
            # compartment size is not differentiable here either, and dY_ss/dp
            # would inherit the wrong column through the linear solve.
            self._raise_if_compartment_size_params(list(sensitivity_params))
            # Issue #63 — the same hard codegen requirement run() and
            # compute_all_sensitivities() apply (GH #214): dY_ss/dp wants the
            # analytical ∂f/∂p the codegen sensitivity RHS emits, so a request
            # that cannot get one is refused rather than quietly answered from
            # √eps-noisy difference quotients.
            #
            # Issue #75 — and the artifact must carry bngsim_codegen_output_sens
            # too, or d(func)/dp falls back to the finite-difference block in
            # compute_ss_output_sensitivity. Bare _auto_codegen_for_sensitivity
            # never emits that symbol from here: it is gated on the model's
            # _want_output_sens, which the CONSTRUCTOR sets from its own
            # sensitivity_params, while this is a METHOD argument. A no-op when a
            # codegen artifact with output sens is already attached.
            self._prepare_output_sens_codegen()

        from bngsim._bngsim_core import (
            SteadyStateOptions,
            find_steady_state,
        )

        opts = SteadyStateOptions()
        opts.tol = tol
        opts.max_time = max_time
        opts.method = method
        opts.rtol = rtol if rtol is not None else self._rtol
        opts.atol = atol if atol is not None else self._atol
        opts.max_steps = max_steps if max_steps is not None else self._max_steps
        # A previous solve on this Simulator already proved the closed-form
        # Jacobian makes CVODE give up on this model, and paid a doomed march to
        # find out (issue #127, the GH #176 memo for `run`). Go straight to the
        # difference quotient rather than re-paying it at every point of a scan.
        opts.jacobian = "fd" if self._ss_jacobian_fell_back else self._jacobian
        # Issue #128 — the Simulator's dense/sparse override reaches the march,
        # which routes by the same rule run() does. Before this, a Simulator
        # built with force_sparse_linear_solver=True still factored densely here.
        opts.force_dense_linear_solver = self._force_dense_linear_solver
        opts.force_sparse_linear_solver = self._force_sparse_linear_solver
        if self._codegen_so_path:
            opts.codegen_so_path = self._codegen_so_path
        if self._codegen_c_source:
            opts.codegen_c_source = self._codegen_c_source
        if sensitivity_params:
            opts.sensitivity_params = list(sensitivity_params)
        if mask_selector is not None:
            opts.steady_state_mask = mask_selector

        logger.info(
            "Finding steady state: method=%s, tol=%.1e",
            method,
            tol,
        )

        try:
            core_result = find_steady_state(self._model._core, opts)
        except RuntimeError as e:
            raise SimulationError(f"Steady-state computation failed: {e}") from e

        result = SteadyStateResult(core_result)

        logger.info(
            "Steady state %s: method=%s, backend=%s, residual=%.2e, steps=%d, species_tested=%d",
            "converged" if result.converged else "FAILED",
            result.method_used,
            result.rhs_backend,
            result.residual,
            result.n_steps,
            result.n_residual_species,
        )
        self._note_ss_jacobian_retry(result)
        self._warn_about_pure_sinks(result)
        self._warn_about_ss_sensitivity(result)
        return result

    def _note_ss_jacobian_retry(self, result) -> None:
        """Say so when the solver had to call off the closed-form Jacobian.

        The steady-state half of GH #176 (issue #127): the solver installed the
        model's analytical Jacobian, CVODE gave up on the march, and it retried
        on difference quotients. Worth a line — the usual cause is a rate law
        that is discontinuous in a state variable, which is a fact about the
        model — and worth remembering, so a scan does not re-pay the doomed
        attempt at every point.
        """
        if not getattr(result, "solver_jacobian_retried", False):
            return
        if not self._ss_jacobian_fell_back:
            logger.warning(
                "GH#176 analytical Jacobian: the steady-state march failed with the "
                "closed-form Jacobian installed; retried with the finite-difference "
                "one. The rate law is likely discontinuous in a state variable (e.g. "
                "an if() whose condition crosses a threshold), which the exact "
                "Jacobian cannot represent. Pass jacobian='fd' to skip this attempt, "
                "or jacobian='analytical' to surface the failure."
            )
        self._ss_jacobian_fell_back = True

    @staticmethod
    def _warn_about_pure_sinks(result: SteadyStateResult) -> None:
        """Name the structural cause of a failed solve (issue #74).

        ``converged=False`` on its own reads as "needs more time", which is the
        one remedy that cannot work when a write-only accumulator has put a floor
        under the residual. Locating the species by hand on a 409-species network
        is an afternoon; the solver already knows, so it says so.
        """
        sinks = result.unconverged_pure_sinks
        if result.converged or not sinks:
            return
        shown = ", ".join(sinks[:4]) + (", ..." if len(sinks) > 4 else "")
        logger.warning(
            "Steady state FAILED for a structural reason, not a numerical one: "
            "%d write-only accumulator species (produced by some reaction, "
            "consumed by none) carry a non-zero derivative at the returned state, "
            "so ||f(y)||_2/n cannot reach tol however long the solve integrates. "
            "Species: %s. Exclude them from the convergence test to solve on the "
            "subspace that does have a steady state: "
            "sim.steady_state(mask=~model.is_pure_sink()).",
            len(sinks),
            shown,
        )

    #: Below this ``min|U_jj| / max|U_jj|`` the steady-state sensitivity system is
    #: reported as badly conditioned. It gates a WARNING and deliberately not a
    #: refusal: the full 585-model ``ode_fullnet`` sweep says no threshold on this
    #: ratio — or on ``1/κ₁``, or on ``σ_min/σ_max`` — can support one.
    #:
    #: Method: solve for ``dY_ss/dp`` the way ``compute_ss_sensitivity`` does, then
    #: check it against a central difference of the steady state itself (re-solve
    #: at ``p ± h`` from the same initial conditions), keeping only probes that
    #: converge in the step size. Of 308 models where the reduced solve returns a
    #: finite answer, 286 are right and 22 are wrong — and the two populations are
    #: not separable:
    #:
    #:   * Correct gradients sit arbitrarily low. ``ode/simplifications_v1``
    #:     measures 1.5e-42 here and is accurate to 7e-7; ``RBM_covid_v2`` (n=112)
    #:     measures 1.1e-13 and is accurate to 1.2e-6. Six correct results fall
    #:     below 1e-8.
    #:   * Wrong gradients sit arbitrarily high. Six of the 22 have a *perfectly*
    #:     conditioned reduced Jacobian — ``NativeTutorials/ABpapprox`` and
    #:     ``ode/temp`` both measure exactly 1.0 and are wrong by more than 100%.
    #:     Those are not conditioning failures and no conditioning number can see
    #:     them.
    #:
    #: The best single cut on this ratio (4.3e-9) still misclassifies 10; the
    #: shipped 1e-8 discards 6 correct results and lets 6 wrong ones through.
    #: ``1/κ₁`` and ``σ_min/σ_max`` do no better (9 and 10 errors at their best
    #: cuts). Refusal is instead gated on the one unambiguous signal — the solve
    #: produced a non-finite gradient — which needs no threshold at all.
    _SS_SENS_RCOND_FLOOR = 1e-8

    @classmethod
    def _warn_about_ss_sensitivity(cls, result: SteadyStateResult) -> None:
        """Surface the ways a dY_ss/dp can be less than it appears (issue #63).

        All of these used to be invisible — the result came back looking like
        every other sensitivity result.
        """
        # 1. ∂f/∂p had to be differenced. Reaching here means codegen IS attached
        #    (steady_state refuses otherwise) but the artifact carries no
        #    bngsim_codegen_sens_rhs. Since GH #67 that is a narrower set than
        #    "not all Elementary": a Functional model whose rate laws are smooth
        #    algebra does get one, and what is left declining is Michaelis-Menten
        #    and the rate laws carrying a condition or a non-smooth builtin.
        #    Still the best available answer, but it rests on a ~sqrt(eps)
        #    difference quotient.
        if result.sens_dfdp_source == "finite-difference":
            logger.warning(
                "Steady-state dY_ss/dp: no analytical ∂f/∂p is available for this "
                "model (the codegen sensitivity RHS declined its rate laws — "
                "Michaelis-Menten, or a Functional law with a condition or a "
                "non-smooth builtin; see issues #55/#68), so ∂f/∂p was "
                "computed by finite differences at a ~sqrt(eps) step, relative "
                "to each parameter (issue #76). The "
                "Jacobian factor used the %s path.",
                result.sens_jacobian_source,
            )

        # 2. The Jacobian at the root is badly conditioned. The solve returned
        #    finite numbers (case 0 above catches the ones that did not), but they
        #    may still be meaningless.
        #
        #    This deliberately does NOT claim the steady state is a continuum, as
        #    it used to. Measured on the corpus, roughly half the models this
        #    fires on return a gradient that is in fact correct — one at
        #    min|U|/max|U| = 1.5e-42 is accurate to 7e-7 — because the ratio is a
        #    heuristic read off the LU diagonal, not a rank test. Say what was
        #    measured and what to do about it, not what it implies.
        rcond = result.sens_jacobian_rcond
        if result.sensitivity is not None and 0.0 <= rcond < cls._SS_SENS_RCOND_FLOOR:
            logger.warning(
                "Steady-state dY_ss/dp may not be reliable for this model: the "
                "Jacobian at the steady state is badly conditioned (min|U|/max|U| "
                "= %.2e from its LU, versus ~1e-4 or better for a typical "
                "well-posed system). The solve returned finite numbers, but if the "
                "root is not isolated they are not a gradient. This ratio is a "
                "heuristic, not a rank test, and it is wrong in both directions on "
                "real models — verify against a finite difference of the steady "
                "state (re-solve at p ± h and difference) before trusting or "
                "discarding this result. Read ss.sens_jacobian_rcond to test it in "
                "code.",
                rcond,
            )

        # 3. The reduced solve failed outright: a zero pivot put NaN/inf in the
        #    result. SUNDIALS' dense LU has no least-squares fallback, so this is
        #    not an ill-conditioned answer to round off — it is no answer at all,
        #    and dY_ss/dp genuinely does not exist at this root. Refuse rather than
        #    return a NaN matrix a fitter will quietly turn into a non-update.
        #
        #    This is the ONLY refusal the corpus supports, and it is deliberately
        #    not a threshold: see _SS_SENS_RCOND_FLOOR for why no cut on the
        #    conditioning can separate right answers from wrong ones. Checked last
        #    so the diagnostics above are still emitted on the way out. Driving the
        #    real entry point over the 585-model corpus: 395 return a gradient, 31
        #    are refused, and no NaN reaches the caller.
        sens = result.sensitivity
        if sens is not None and not np.all(np.isfinite(sens)):
            arr = np.asarray(sens)
            finite_rows = np.all(np.isfinite(arr), axis=1)
            # Issue #74 — a masked-out species' row is NaN *by construction*: it
            # has no steady value, so it has no steady-state gradient, and 0.0
            # would be a confident wrong answer. That is the caller's own
            # decision, not a failed solve, so it is exempt from the refusal
            # below (the species-level NaN is left in place, and the warning
            # above it says so). Every other row still has to be finite.
            for i in result.excluded_species:
                finite_rows[i] = True
            if result.excluded_species:
                logger.warning(
                    "Steady-state dY_ss/dp: %d species were excluded from the "
                    "convergence test by mask=, so they have no steady value and "
                    "their rows are NaN (first: %s). Any observable or expression "
                    "sensitivity summing one of them is NaN for the same reason.",
                    len(result.excluded_species),
                    ", ".join(result.species_names[i] for i in list(result.excluded_species)[:3]),
                )
            if np.all(finite_rows):
                return
            bad = [n for n, ok in zip(result.species_names, finite_rows, strict=True) if not ok]
            raise SimulationError(
                "Steady-state dY_ss/dp does not exist for this model: the Jacobian "
                "at the steady state is singular on the reduced (conservation-law) "
                "subspace, so -J⁻¹·(∂f/∂p) has no solution and the linear solve "
                f"returned non-finite values for {len(bad)} of {arr.shape[0]} "
                f"species (first: {', '.join(bad[:3])}"
                f"{', ...' if len(bad) > 3 else ''}). The steady state is a "
                "continuum rather than an isolated point — there is a direction you "
                "can move along without leaving equilibrium — so no unique gradient "
                "exists to report. A common cause is two species that are only "
                "produced and never consumed, fed from a common irreversible step: "
                "that makes the equilibrium set a line. Check "
                "Model.pure_sink_species() — if it names them, they have no steady "
                "value at all, and steady_state(mask=~model.is_pure_sink()) solves "
                "for the gradient of the species that do (issue #74). "
                f"ss.sens_jacobian_rcond is {result.sens_jacobian_rcond:.2e}."
            )

    def steady_state_batch(
        self,
        params,
        *,
        tol: float = 1e-9,
        max_time: float = 1e6,
        method: str = "integration",
        rtol=None,
        atol=None,
        max_steps=None,
        n_workers=None,
        mask=None,
    ):
        """Compute steady states for multiple parameter sets.

        Parameters
        ----------
        params : sequence of dict[str, float]
            Parameter sets.
        method : str
            ``"integration"`` (default), ``"newton"``, or ``"kinsol"``
            (alias for ``"newton"``). See :meth:`steady_state`.
        n_workers : int, optional
            Number of parallel threads.
        mask : array-like of bool, or sequence of str, optional
            Which species the convergence test covers, applied identically to
            every entry (issue #74). See :meth:`steady_state`. The species set is
            structural, so it does not move with the parameter set — which is
            what makes one mask correct for a whole dose scan.

        Returns
        -------
        list[SteadyStateResult]

        Notes
        -----
        Every entry runs this Simulator's :attr:`codegen_backend`, the same as
        :meth:`steady_state` (issue #63). Each entry resolves the artifact for
        itself — one ``dlopen`` of the shared path, or one JIT compile of the
        shared source, per solve — so a dose-response sweep pays that per entry
        rather than once for the batch. ``r.rhs_backend`` reports what ran.
        """
        if self._method != "ode":
            raise ValueError(
                "steady_state_batch() is only supported for "
                f"method='ode', not method='{self._method}'."
            )
        if not params:
            raise ValueError("params must be non-empty")

        mask_selector = _resolve_ss_mask(mask, self._model)

        from bngsim._bngsim_core import (
            SteadyStateOptions,
            find_steady_state,
        )

        eff_rtol = rtol if rtol is not None else self._rtol
        eff_atol = atol if atol is not None else self._atol
        eff_max_steps = max_steps if max_steps is not None else self._max_steps

        def _run_one(i):
            clone = self._model.clone()
            clone.set_params(params[i])
            clone.reset()
            opts = SteadyStateOptions()
            opts.tol = tol
            opts.max_time = max_time
            opts.method = method
            opts.rtol = eff_rtol
            opts.atol = eff_atol
            opts.max_steps = eff_max_steps
            # Read (and, below, set) the same memo `steady_state()` keeps, so a
            # sweep over a model whose closed-form Jacobian CVODE cannot use pays
            # the doomed march once rather than once per entry (issue #127).
            opts.jacobian = "fd" if self._ss_jacobian_fell_back else self._jacobian
            # Issue #128 — same dense/sparse routing as steady_state().
            opts.force_dense_linear_solver = self._force_dense_linear_solver
            opts.force_sparse_linear_solver = self._force_sparse_linear_solver
            if self._codegen_so_path:
                opts.codegen_so_path = self._codegen_so_path
            if self._codegen_c_source:
                opts.codegen_c_source = self._codegen_c_source
            if mask_selector is not None:
                opts.steady_state_mask = mask_selector
            try:
                core_result = find_steady_state(clone._core, opts)
            except RuntimeError as e:
                raise SimulationError(f"Batch {i} failed: {e}") from e
            result = SteadyStateResult(core_result)
            self._note_ss_jacobian_retry(result)
            self._warn_about_pure_sinks(result)
            return result

        if n_workers is not None and n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_run_one, i) for i in range(len(params))]
                results = []
                for i, future in enumerate(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        raise SimulationError(f"Batch {i} failed: {e}") from e
        else:
            results = [_run_one(i) for i in range(len(params))]

        return results

    # ─── Stop conditions ────────────────────────────────────────────

    def add_stop_condition(
        self,
        condition: str | Callable,
        *,
        label: str = "",
    ) -> None:
        """Add a stop condition checked after each simulation.

        Parameters
        ----------
        condition : str or callable
            - **str**: An expression string evaluated at each time
              point using observable names as variables. Simulation
              stops when the expression becomes true (> 0).
              Example: ``"A_tot < 10"``
            - **callable**: A function ``f(result) -> bool`` called
              after the simulation. If it returns True, the stop
              condition is triggered.
        label : str
            Human-readable label for the condition.

        Examples
        --------
        >>> sim.add_stop_condition("A_tot < 10", label="low_A")
        >>> sim.add_stop_condition(
        ...     lambda r: r.species[-1, 0] < 5,
        ...     label="very_low_A",
        ... )
        """
        sc: _StopCondition
        if isinstance(condition, str):
            sc = _ExpressionStopCondition(condition, label=label)
        elif callable(condition):
            sc = _CallableStopCondition(condition, label=label)
        else:
            raise TypeError(
                "condition must be a string expression or callable, "
                f"got {type(condition).__name__}"
            )
        self._stop_conditions.append(sc)
        logger.debug(
            "Added stop condition: %s",
            label or repr(condition),
        )

    def clear_stop_conditions(self) -> None:
        """Remove all stop conditions."""
        self._stop_conditions.clear()
        logger.debug("Cleared all stop conditions")

    def _maybe_warn_dense_fallback(self) -> None:
        """Warn once if a large ODE model runs dense-only for lack of KLU.

        GH #209. When this install was built without SuiteSparse/KLU, the ODE
        backend can only use the dense linear solver, which factorizes the full
        N×N Jacobian at O(N³). For a large/sparse model that is the difference
        between minutes and hours — and it is silent, because the build
        "succeeded" dense-only. We surface it as a one-time
        :class:`bngsim.DenseSolverFallbackWarning` at ``run()`` when, and only
        when, the dense solver is forced by the *missing build*, not by a
        deliberate user choice:

        * ``HAS_KLU`` is False (no sparse solver compiled in), and
        * the user did not request ``force_dense_linear_solver`` or
          ``jacobian="jax"`` (both legitimately dense), and
        * the model is large enough for sparsity to matter
          (``n_species >= _DENSE_FALLBACK_WARN_NSPECIES``).

        A KLU-enabled install never reaches the warning. The notice fires at
        most once per process (see ``_dense_fallback_warned``).
        """
        global _dense_fallback_warned
        if _dense_fallback_warned or _HAS_KLU:
            return
        if self._force_dense_linear_solver or self._jacobian == "jax":
            return
        try:
            n_species = int(self._model.n_species)
        except Exception:  # pragma: no cover - defensive; never block a run
            return
        if n_species < _DENSE_FALLBACK_WARN_NSPECIES:
            return

        _dense_fallback_warned = True
        warnings.warn(
            f"This bngsim install was built WITHOUT SuiteSparse/KLU, so this "
            f"{n_species:,}-species ODE model will run on the DENSE linear "
            f"solver — it factorizes the full N×N Jacobian at O(N³), which for "
            f"a large/sparse network can be orders of magnitude slower (minutes "
            f"→ hours) and far more memory-hungry than the sparse KLU solver. "
            f"Rebuild bngsim with KLU to fix this: install SuiteSparse "
            f"(brew install suite-sparse / apt-get install libsuitesparse-dev / "
            f"conda install -c conda-forge suitesparse) and reinstall from "
            f"source; if it lives on a non-standard prefix pass "
            f"-DCMAKE_PREFIX_PATH or -DKLU_ROOT. Verify with "
            f"bngsim.capabilities()['features']['klu']. See GH #209. (Silence: "
            f"warnings.simplefilter('ignore', bngsim.DenseSolverFallbackWarning).)",
            DenseSolverFallbackWarning,
            stacklevel=3,
        )

    @staticmethod
    def _warn_ssa_boundary(result: Result) -> None:
        """Emit one ``SsaBoundaryWarning`` per literal-rate-law boundary event.

        GH #110. The exact SSA does not floor species at zero and fires a
        negative-rate reaction in reverse so its mean tracks the ODE; both are
        surfaced here instead of being silent. No-op on non-SSA backends (the
        diagnostic counts are zero there). The structured counts remain on
        ``result.ssa_diagnostics`` regardless of warning filters.
        """
        diag = result.ssa_diagnostics
        if diag["n_negative_crossings"] > 0:
            sp = diag["first_negative_species"]
            sp_txt = f" (first: {sp})" if sp else ""
            warnings.warn(
                f"SSA: a species count went negative {diag['n_negative_crossings']} "
                f"time(s){sp_txt}. The SSA evaluates rate laws literally and does "
                "not floor counts at zero (matching the ODE/CVODE path); enforce "
                "non-negativity in the rate law itself, e.g. piecewise(X<=0, 0, k). "
                "See result.ssa_diagnostics.",
                SsaBoundaryWarning,
                stacklevel=3,
            )
        if diag["n_reverse_fires"] > 0:
            rx = diag["first_reverse_reaction"]
            rx_txt = f" (first: {rx})" if rx else ""
            warnings.warn(
                f"SSA: a reaction fired in reverse {diag['n_reverse_fires']} "
                f"time(s){rx_txt} because its rate law evaluated negative. The "
                "reaction is run backward with propensity |rate| so the SSA mean "
                "tracks the ODE; if this is unintended, fix the rate-law sign or "
                "split the reaction into explicit forward/reverse channels. See "
                "result.ssa_diagnostics.",
                SsaBoundaryWarning,
                stacklevel=3,
            )

    def _check_stop_conditions(self, result: Result) -> None:
        """Check stop conditions against a completed result.

        If any condition triggers, raises StopConditionMet with
        the result truncated to the trigger point.
        """
        for sc in self._stop_conditions:
            trigger_idx = sc.check(result)
            if trigger_idx is not None:
                # Truncate result to the trigger point
                trunc = self._truncate_result(result, trigger_idx + 1)
                label = sc.label or str(sc)
                logger.info(
                    "Stop condition triggered at t=%.6g: %s",
                    result.time[trigger_idx],
                    label,
                )
                raise StopConditionMet(
                    f"Stop condition '{label}' triggered at t={result.time[trigger_idx]:.6g}",
                    result=trunc,
                    condition=label,
                )

    @staticmethod
    def _truncate_result(result: Result, n: int) -> Result:
        """Return a result truncated to the first n time points."""
        return Result(
            core=None,
            _time=result._time[:n].copy(),
            _species=result._species[:n].copy(),
            _observables=result._observables[:n].copy(),
            _expressions=result._expressions[:n].copy()
            if result._expressions.size > 0
            else result._expressions,
            _species_names=result._species_names,
            _observable_names=result._observable_names,
            _expression_names=result._expression_names,
            _solver_stats=result._solver_stats,
            _species_volume_factors=result._species_volume_factors,
        )

    # ─── Interactive simulation ─────────────────────────────────────

    def run_until(
        self,
        t: float,
        *,
        n_points: int | None = None,
        seed: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
    ) -> Result:
        """Run simulation from current time to *t*.

        This enables interactive simulation: run to a time point,
        inspect or modify the model, then continue. Supported for
        stateful model-backed solvers (ODE / SSA / PSA) only.

        Parameters
        ----------
        t : float
            Target time. Must be > current time.
        n_points : int, optional
            Number of output points. Default: max(2, int(dt)+1).
        seed : int, optional
            Random seed for stochastic methods. ``None`` (default)
            draws a fresh seed; pass an integer for reproducibility.
            See ``Simulator.run`` for the full contract.
        rtol, atol, max_steps : optional
            Solver options (ODE only).

        Returns
        -------
        Result
            Simulation result for the [current_time, t] interval.

        Examples
        --------
        >>> sim.run_until(t=50)        # simulate to t=50
        >>> sim.intervene({"k1": 0.0}) # knock out a reaction
        >>> result = sim.run_until(t=100)  # continue to t=100
        """
        self._require_interactive_backend_support()

        if t <= self._current_time:
            raise ValueError(f"Target time ({t}) must be > current time ({self._current_time})")

        dt = t - self._current_time
        if n_points is None:
            n_points = max(2, int(dt) + 1)

        logger.info(
            "run_until: t=%.6g → %.6g (%d points)",
            self._current_time,
            t,
            n_points,
        )

        result = self.run(
            t_span=(self._current_time, t),
            n_points=n_points,
            seed=seed,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )

        # Update current time
        self._current_time = t

        # The model already holds the final state from the
        # simulation (CVODE/SSA write back final concentrations)
        return result

    def intervene(self, params: dict[str, float]) -> None:
        """Apply a perturbation (parameter change) mid-simulation.

        Use between ``run_until()`` calls to modify the model
        during an interactive simulation session.

        Parameters
        ----------
        params : dict[str, float]
            Parameter name → value mapping.

        Examples
        --------
        >>> sim.run_until(t=50)
        >>> sim.intervene({"k1": 0.0})  # knock out reaction
        >>> sim.run_until(t=100)        # continue
        """
        self._require_interactive_backend_support()

        logger.info(
            "Intervening at t=%.6g: %s",
            self._current_time,
            params,
        )
        self._model.set_params(params)

        # Recreate the C++ simulator to pick up parameter changes
        self._recreate_interactive_sim()

    def save_concentrations(self, label: str | None = None) -> None:
        """Snapshot the model's current concentrations (BNG ``saveConcentrations``).

        Thin delegator to :meth:`Model.save_concentrations` on the underlying
        model. With no ``label`` this overwrites the default slot (a later
        :meth:`Model.reset` returns here); with a ``label`` it stores a named
        snapshot that :meth:`parameter_scan` can reset each point to via
        ``reset_to=label``. Use it to capture a post-intervention state between
        ``run_until`` phases and a following scan (issue #11).
        """
        self._require_interactive_backend_support()
        self._model.save_concentrations(label)

    def restore_concentrations(self, label: str | None = None) -> None:
        """Restore the model's concentrations from a snapshot (BNG ``resetConcentrations``).

        Thin delegator to :meth:`Model.restore_concentrations`. With no ``label``
        this restores the default slot (identical to :meth:`Model.reset`); with a
        ``label`` it restores that named snapshot. The backend simulator is
        rebuilt so a subsequent :meth:`run` / :meth:`run_until` seeds from the
        restored state.
        """
        self._require_interactive_backend_support()
        self._model.restore_concentrations(label)
        # Species state changed wholesale; rebuild the backend so it seeds from
        # the restored concentrations on the next run.
        self._recreate_interactive_sim()

    def snapshot(self) -> dict:
        """Capture the current simulation state.

        Returns a dict that can be passed to ``restore()`` to
        return to this point.

        Returns
        -------
        dict
            Opaque snapshot of model + simulator state.

        Examples
        --------
        >>> sim.run_until(t=50)
        >>> snap = sim.snapshot()
        >>> sim.run_until(t=100)
        >>> sim.restore(snap)  # back to t=50
        """
        self._require_interactive_backend_support()

        # Save model state: all species concentrations + params
        species_state = {
            name: self._model.get_concentration(name) for name in self._model.species_names
        }
        param_state = {name: self._model.get_param(name) for name in self._model.param_names}
        snap = {
            "current_time": self._current_time,
            "species": species_state,
            "params": param_state,
        }
        self._snapshot_stack.append(copy.deepcopy(snap))
        logger.debug(
            "Snapshot captured at t=%.6g",
            self._current_time,
        )
        return snap

    def restore(self, snapshot: dict | None = None) -> None:
        """Restore simulation state from a snapshot.

        Parameters
        ----------
        snapshot : dict, optional
            A snapshot from ``snapshot()``. If ``None``, restores
            the most recent snapshot from the internal stack.

        Examples
        --------
        >>> snap = sim.snapshot()
        >>> sim.run_until(t=100)
        >>> sim.restore(snap)  # back to snapshot point
        >>> sim.restore()      # same (uses internal stack)
        """
        self._require_interactive_backend_support()

        if snapshot is None:
            if not self._snapshot_stack:
                raise SimulationError("No snapshots available to restore")
            snapshot = self._snapshot_stack.pop()

        self._current_time = snapshot["current_time"]

        # Restore parameters (parameter may not exist after model changes)
        for name, value in snapshot["params"].items():
            with contextlib.suppress(Exception):
                self._model.set_param(name, value)

        # Restore species concentrations
        for name, value in snapshot["species"].items():
            with contextlib.suppress(Exception):
                self._model.set_concentration(name, value)

        # Recreate simulator with restored state
        self._recreate_interactive_sim()

        logger.info(
            "Restored to t=%.6g",
            self._current_time,
        )

    # ─── Bulk state exchange (GH #102) ─────────────────────────────

    def get_state(self) -> np.ndarray:
        """Bulk-copy the live species-concentration vector (GH #102).

        Thin delegator to :meth:`Model.get_state` on the underlying model. After
        a stateful ``run_until``/``run`` the model holds the final state (the
        ODE/SSA backends write it back), so this returns the post-step state.
        It is the per-step ``get`` half of driving bngsim as a reaction kernel
        from an external orchestrator; pair with :meth:`set_state`.
        """
        return self._model.get_state()

    def set_state(self, state: np.ndarray) -> None:
        """Bulk-assign the live species-concentration vector (GH #102).

        Thin delegator to :meth:`Model.set_state`. The C++ simulator reads the
        model's current concentrations as the initial condition at the start of
        the next ``run_until``/``run``, so a bulk ``set_state`` between steps is
        the per-step ``set`` half of the kernel exchange (e.g. injecting the
        SSA-subset coupling species before advancing the ODE subset).
        """
        self._model.set_state(state)

    # ─── Solver configuration (ODE) ────────────────────────────────

    def set_tolerances(self, rtol: float = 1e-8, atol: float = 1e-8) -> None:
        """Set ODE solver tolerances.

        Parameters
        ----------
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        """
        self._rtol = rtol
        self._atol = atol
        if self._method == "ode":
            self._sim.set_tolerances(rtol, atol)

    def set_max_steps(self, max_steps: int) -> None:
        """Set maximum internal solver steps per output point.

        Parameters
        ----------
        max_steps : int
            Maximum steps.
        """
        self._max_steps = max_steps
        if self._method == "ode":
            self._sim.set_max_steps(max_steps)

    # ─── Properties ─────────────────────────────────────────────────

    @property
    def method(self) -> str:
        """Internal dispatch method ('ode', 'ssa', 'psa', 'nfsim', or 'rulemonkey')."""
        return self._method

    @property
    def requested_method(self) -> str:
        """Original method token as provided by the user.

        Useful for logging and reproducibility metadata. For example,
        if the user passed ``method="nf"``, this returns ``"nf"`` while
        :attr:`method` returns the backend dispatch key (``"nfsim"`` for
        ``nf``/``nf_reject``).
        """
        return self._requested_method

    @property
    def model(self) -> Model:
        """The model being simulated."""
        return self._model

    @property
    def codegen_backend(self) -> str:
        """The RHS codegen backend this Simulator hands the ODE engine.

        Returns one of:

        - ``"mir"`` — in-process MIR micro-JIT (GH #78): the generated C
          source is JIT-compiled inside C++ (reached only when
          ``BNGSIM_CODEGEN_JIT=mir`` on a MIR-enabled build prepared codegen
          for this model).
        - ``"cc"`` — native C compiled to a ``.so`` by ``cc`` and ``dlopen``'d
          (auto-selected at/above ``BNGSIM_CODEGEN_THRESHOLD`` species — 256 by
          default — or when ``codegen=True`` was passed explicitly).
        - ``"exprtk"`` — the ExprTk bytecode interpreter, no native code
          (the default below the codegen threshold).

        This is the backend that *actually* runs, not a request: it reflects
        exactly what :meth:`run` passes the engine — a non-empty JIT source
        selects MIR, else a non-empty ``.so`` path selects cc, else ExprTk
        (mirroring the ``opts.codegen_*`` dispatch). Only meaningful for
        ``method="ode"``; other backends never codegen and report ``"exprtk"``.
        """
        if self._codegen_c_source:
            return "mir"
        if self._codegen_so_path:
            return "cc"
        return "exprtk"

    @property
    def jacobian_strategy(self) -> str:
        """The Jacobian strategy the ODE engine *actually* uses: ``"analytical"``,
        ``"fd"``, or ``"jax"``.

        This is the post-resolution strategy, not the requested ``jacobian=``
        mode. With ``jacobian="auto"`` (the default) the engine uses the
        analytical Jacobian when the model has one — every Elementary mass-action
        law, plus Functional laws whose derivatives were symbolically derived
        within the build-time budget (GH #76/#95) — and finite differences
        otherwise. So this reports ``"analytical"`` only when the analytical
        Jacobian is genuinely complete and not overridden:

        - ``jacobian="fd"`` → always ``"fd"``.
        - ``BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0`` (a Functional model loaded with
          this set never attaches its analytical terms) → ``"fd"``.
        - a derivation that blew the budget and fell back → ``"fd"``.
        - ``jacobian="auto"`` whose analytical attempt failed to integrate and
          fell back to FD at run time (GH #176) → ``"fd"``.
        - ``jacobian="jax"`` → ``"jax"``.

        Mirrors the engine's callback selection exactly (cvode_simulator.cpp):
        analytical iff the requested mode is auto/analytical *and*
        ``analytical_jacobian_complete``. Only meaningful for ``method="ode"``.
        """
        requested = self._jacobian
        if requested == "jax":
            return "jax"
        if self._ode_jacobian_fell_back:
            return "fd"
        if requested in ("auto", "analytical") and bool(
            self._model._core.analytical_jacobian_complete
        ):
            return "analytical"
        return "fd"

    @property
    def last_codegen_sec(self) -> float:
        """Wall seconds spent generating/compiling this model's codegen RHS.

        ``≈0.0`` for an ExprTk model (no codegen runs) or a codegen cache hit;
        the ``cc`` compile time on a cold ``"cc"`` model; the source-generation
        time on a ``"mir"`` model. Recorded once at setup by the
        ``bngsim._codegen.prepare_*`` entry points (whether codegen ran at model
        load or in this Simulator), so a single :meth:`run` exposes the codegen
        cost directly — no run-twice-and-subtract needed. Purely a setup-time
        figure; the per-step integration hot path is never instrumented.
        """
        return float(getattr(self._model, "_codegen_sec", 0.0))

    @property
    def last_libsbml_parse_sec(self) -> float:
        """Wall seconds the SBML loader spent in the libSBML parse phase
        (``readSBML*`` + document-level error check) — the shared C++ core both
        engines use. Recorded once at load; setup-time only, never the hot path.
        ``0.0`` for a model not loaded from SBML (e.g. a ``.net`` model)."""
        return float(getattr(self._model, "_libsbml_parse_sec", 0.0))

    @property
    def last_interpret_sec(self) -> float:
        """Wall seconds spent interpreting the parsed libSBML document into the
        internal ``_core`` model (bngsim's Python interpretation layer, including
        the ``builder.build()`` core construction; excludes libSBML parse, the
        analytical-Jacobian derivation, and codegen, which are timed separately).
        Recorded once at load; setup-time only."""
        return float(getattr(self._model, "_interpret_sec", 0.0))

    @property
    def last_jacobian_sec(self) -> float:
        """Wall seconds spent symbolically deriving this model's analytical
        Functional Jacobian (GH #76, ``sympy`` ``sp.diff``), with SymPy already
        imported — the one-time SymPy import is process-warmup, measured
        separately, not here. ``≈0.0`` for an all-Elementary model, an FD fallback,
        an over-budget derivation (GH #95), or ``BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0``.
        A bngsim-only per-model cost — RoadRunner uses a difference-quotient
        Jacobian and has no analog. Setup-time only; the per-step integration hot
        path is never instrumented. Recorded wherever the derivation runs (at load
        today; at first ODE-solve setup after the GH #145 lazy deferral), so this
        accessor is stable across that change."""
        return float(getattr(self._model, "_jac_derive_sec", 0.0))

    @property
    def codegen_cache_hit(self) -> bool | None:
        """Whether this model's compiled ``.so`` was reused from the on-disk
        codegen cache (``~/.cache/bngsim/codegen/``).

        - ``True`` — the ``.so`` was found in the cache and loaded without
          recompiling (the ``cc`` compile was skipped).
        - ``False`` — no cached ``.so`` matched, so it was compiled fresh.
        - ``None`` — no ``.so`` was involved at all: an ExprTk model (no codegen)
          or a MIR model (in-process JIT, which has no on-disk ``.so`` cache).

        This is the *definitive* cache signal recorded by the codegen pipeline at
        the ``get_cached_so`` / memo branch — not inferred from
        :attr:`last_codegen_sec` (a model-based cache hit still spends time on
        source generation, so a small wall time does not imply a cache hit).
        Only meaningful for ``method="ode"`` with the ``"cc"`` backend.
        """
        return getattr(self._model, "_codegen_cache_hit", None)

    @property
    def current_time(self) -> float:
        """Current time in interactive simulation."""
        return self._current_time

    def __repr__(self) -> str:
        return f"Simulator(method='{self._method}', model={self._model!r})"


# ─── Stop condition implementations ────────────────────────────────


class _StopCondition:
    """Base class for stop conditions."""

    def __init__(self, label: str = "") -> None:
        self.label = label

    def check(self, result: Result) -> int | None:
        """Check condition. Returns time index or None."""
        raise NotImplementedError


class _ExpressionStopCondition(_StopCondition):
    """Stop condition based on observable expression string.

    The expression is evaluated at each time point using
    observable names as variables. Returns the first time
    index where the expression evaluates to True (> 0).
    """

    def __init__(self, expression: str, *, label: str = "") -> None:
        super().__init__(label=label or expression)
        self._expression = expression
        self._code = compile(expression, "<stop_condition>", "eval")

    def check(self, result: Result) -> int | None:
        """Evaluate expression at each time point."""
        obs_names = result.observable_names
        obs_data = np.asarray(result.observables)
        time = result.time

        for t_idx in range(len(time)):
            # Build namespace for eval
            ns = {"time": time[t_idx], "t": time[t_idx]}
            for j, name in enumerate(obs_names):
                ns[name] = obs_data[t_idx, j]
            # Also add species
            sp_data = result.species
            for j, name in enumerate(result.species_names):
                # Sanitize species name for eval
                safe = name.replace("(", "_").replace(")", "_")
                ns[safe] = sp_data[t_idx, j]

            try:
                val = eval(  # noqa: S307
                    self._code, {"__builtins__": {}}, ns
                )
                if val:
                    return t_idx
            except Exception as exc:
                logger.debug(
                    "Stop condition '%s' eval failed at t=%s: %s",
                    self._expression,
                    time[t_idx],
                    exc,
                )
                continue

        return None


class _CallableStopCondition(_StopCondition):
    """Stop condition based on a Python callable.

    The callable receives the full Result and returns True
    if the condition is met.
    """

    def __init__(
        self,
        func: Callable,
        *,
        label: str = "",
    ) -> None:
        super().__init__(label=label or repr(func))
        self._func = func

    def check(self, result: Result) -> int | None:
        """Call the function; if True, return last time index."""
        try:
            if self._func(result):
                return len(result.time) - 1
        except Exception:
            pass
        return None


def _resolve_ss_mask(mask, model: Model) -> list[int] | None:
    """Normalize a ``steady_state(mask=...)`` argument to the core's selector.

    Issue #74. Two spellings, distinguished by element type so neither can be
    mistaken for the other:

    * a **boolean** array-like of length ``n_species`` — ``True`` keeps the
      species in the convergence test. This is the form
      ``~model.is_pure_sink()`` produces.
    * a **sequence of species names** — the species to keep, in any order.

    Integers are deliberately rejected: ``[0, 1]`` is ambiguous between a
    two-species 0/1 mask and "keep species 0 and 1", and guessing would be a
    silent wrong answer on exactly the kind of long species list where the
    caller cannot eyeball it.

    Returns ``None`` when ``mask`` is ``None``, so the caller leaves the core
    option unset and the BNG2.pl parity criterion applies unchanged.
    """
    if mask is None:
        return None

    names = model.species_names
    ns = len(names)

    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        if arr.ndim != 1 or arr.size != ns:
            raise ValueError(
                f"steady_state(mask=...): a boolean mask must have one entry per "
                f"species, got shape {arr.shape} for a model with {ns} species."
            )
        keep = [1 if v else 0 for v in arr.tolist()]
    elif arr.dtype.kind in "US" or (arr.dtype == object and all(isinstance(x, str) for x in arr)):
        wanted = [str(x) for x in arr.tolist()]
        index = {n: i for i, n in enumerate(names)}
        unknown = [n for n in wanted if n not in index]
        if unknown:
            raise ValueError(
                f"steady_state(mask=...): unknown species name(s) "
                f"{unknown[:5]}{', ...' if len(unknown) > 5 else ''}. Names must "
                f"match Model.species_names exactly."
            )
        keep = [0] * ns
        for n in wanted:
            keep[index[n]] = 1
    else:
        raise TypeError(
            "steady_state(mask=...) takes either a boolean array of length "
            f"n_species ({ns}) or a sequence of species names to keep, not "
            f"{arr.dtype!r}. Integer indices are rejected as ambiguous — build a "
            "boolean array instead, e.g. ~model.is_pure_sink()."
        )

    if not any(keep):
        raise ValueError(
            "steady_state(mask=...) excludes every species: there is no subspace "
            "left to solve f(y) = 0 on."
        )
    return keep


def _ss_output_sens_block(core: Any, attr: str) -> np.ndarray:
    """Read a 2-D steady-state output-sensitivity block off a C++ core (GH #12).

    The pybind accessors return a ``(n_rows, n_params)`` array — empty ``(0, 0)``
    when the block was never populated (no sensitivity run). A ``hasattr`` guard
    tolerates an older core built before the block existed.
    """
    if hasattr(core, attr):
        return np.asarray(getattr(core, attr), dtype=np.float64)
    return np.empty((0, 0))


class SteadyStateResult:
    """Result of a steady-state computation.

    Attributes
    ----------
    concentrations : ndarray, shape (n_species,)
        Species concentrations at steady state.
    species_names : list[str]
        Species names.
    residual : float
        ``max|f(y)|`` at convergence.
    method_used : str
        ``"integration"`` or ``"newton"``.
    converged : bool
        Whether steady state was found.
    n_steps : int
        Number of solver steps.
    n_rhs_evals : int
        Number of RHS evaluations.
    n_residual_species : int
        How many species entered ``||f||_2 / n`` — ``n_species`` unless
        ``steady_state(mask=...)`` restricted it (issue #74). Assert on this
        rather than inferring from a residual that moved.
    excluded_species : list[int]
        0-based indices the mask excluded, ascending; empty without a mask.
        Their concentrations are returned but their settling was never tested,
        and their ``sensitivity`` rows are NaN.
    unconverged_pure_sinks : list[str]
        On a *failed* solve, the write-only accumulator species that were in the
        convergence test and are carrying flux at the returned state — the
        structural reason the residual has a floor (issue #74). Empty otherwise,
        including on every converged solve. See :meth:`Model.pure_sink_species`.
    root_stability : str
        Whether the system can rest on the returned root (issue #78), from the
        eigenvalues of the Jacobian restricted to the species the Newton polish
        solved for: ``"stable"`` (every eigenvalue in the closed left half-plane),
        ``"undetermined"`` (the certificate declined — more than 512 unknowns, or
        a Jacobian whose spectrum it could not compute), or ``"unstable"``.
        ``""`` when the result came from integration, which needs no certificate:
        a trajectory cannot come to rest on an unstable equilibrium.
        ``"unstable"`` is returned only when the *initial condition itself* was
        that root; a polish that lands on one is discarded instead.
    n_unstable_roots_rejected : int
        How many candidate Newton roots this solve discarded as unstable (issue
        #78). Non-zero means the polish landed on a saddle the trajectory was
        merely passing near — on a bistable model the seed-stability guard cannot
        catch that, because near a separatrix the trajectory slows down and two
        successively tighter bursts hand Newton the same seed. Explains a
        ``method_used`` of ``"integration"`` from a ``method="newton"`` solve.
    sensitivity : ndarray or None
        Species ``dY_ss/dp`` matrix, shape ``(n_species, n_params)``. ``None``
        if no sensitivity was requested.
    rhs_backend : str
        Which RHS actually evaluated ``f(y)``: ``"exprtk"`` (interpreted),
        ``"codegen-so"`` (compiled + ``dlopen``'d), or ``"codegen-jit"``
        (in-process MIR). Issue #63 — before it, a steady-state solve ignored
        the Simulator's codegen artifact, so this was always effectively
        ``"exprtk"`` no matter what :attr:`Simulator.codegen_backend` reported.
    solver_jacobian_source : str
        Which Newton matrix the *solver tiers* factored (issue #127):
        ``"codegen"`` (the compiled analytical Jacobian), ``"analytical"`` (the
        interpreted fill), or ``"finite-difference"`` (CVODE's and KINSOL's own
        difference quotients, one RHS evaluation per unknown per Jacobian setup).
        Reported on every solve, and it covers both tiers — the CVODE march and
        the KINSOL polish take the same gate, so they cannot disagree. Before
        #127 neither installed a callback and this was always the difference
        quotient, on models whose closed form was already loaded.
    solver_jacobian_retried : bool
        Whether the closed-form Jacobian had to be *called off*: CVODE gave up on
        the march with it installed, and the solve was retried on difference
        quotients (issue #127, the steady-state half of GH #176). The usual cause
        is a rate law that is discontinuous in a state variable, whose exact
        derivative omits the jump. The returned answer is the retry's, so
        ``solver_jacobian_source`` reads ``"finite-difference"`` alongside this.
        Only ``jacobian="auto"`` retries; ``"analytical"`` surfaces the failure.
    linear_solver : str
        Which direct linear solver the CVODE **march** factored its Newton matrix
        with (issue #128): ``"klu"`` (sparse CSC), ``"dense"`` (the built-in dense
        LU) or ``"lapack-dense"`` (the GH #84 BLAS factor). Chosen by the same
        size/density/force-flag rule :meth:`Simulator.run` uses, so the two agree
        on a model. Before #128 the steady-state paths had no sparse route and
        this was always dense — 1.9x to 3.1x of wall clock on the 1000+ species
        corpus models, for the same answer. The KINSOL polish and the
        ``dY_ss/dp`` solve factor a *reduced* matrix and are always dense; this
        field does not describe them.
    sens_jacobian_source, sens_dfdp_source : str
        How each factor of ``dY_ss/dp = -J⁻¹·(∂f/∂p)`` was built.
        ``sens_jacobian_source`` is ``"codegen"`` (compiled analytical Jacobian),
        ``"analytical"`` (interpreted analytical), or ``"finite-difference"``;
        ``sens_dfdp_source`` is ``"codegen"`` or ``"finite-difference"``. Both
        are ``""`` when no sensitivity was requested. Issue #63 — both factors
        were unconditionally finite-differenced before it, and the result gave
        no way to tell.
    sens_output_source : str
        How the expression block ``d(func)/dp`` was built (issue #75):
        ``"codegen"`` (the compiled ``bngsim_codegen_output_sens`` chain rule —
        the same evaluator a CVODES forward-sensitivity ``run()`` uses — answered
        every function), ``"finite-difference"`` (it answered none: no compiled
        artifact, or the codegen declined the whole model), or ``"mixed"`` (it
        answered some and the rest were differenced, e.g. a table function).
        ``""`` when no sensitivity was requested or the model has no global
        functions. The *observable* block is an exact linear projection either
        way and is not covered by this field.
    sens_jacobian_rcond : float
        ``min|U_jj| / max|U_jj|`` from the LU of the (reduced) Jacobian that was
        inverted — how close to singular the sensitivity system was.
        ``dY_ss/dp`` exists only when that Jacobian has full rank; a steady state
        that is a *continuum* rather than an isolated point makes it
        rank-deficient, and the returned matrix is then meaningless. Well-posed
        corpus models measure 1e-4 to 1e-1; rank-deficient ones 1e-12 to 1e-9.
        ``0.0`` when no sensitivity was requested.

    Notes
    -----
    When ``sensitivity_params`` is passed to :meth:`Simulator.steady_state`, the
    result also carries the **observable-** and **expression-level** steady-state
    forward sensitivities (GH #12): read them by name with
    :meth:`output_sensitivities`, mirroring :meth:`bngsim.Result.output_sensitivities`
    on a CVODE run. These are the chain-rule projection of the species
    ``dY_ss/dp`` onto the model's observables and global functions, so a gradient
    consumer gets ``∂(observable)/∂θ`` directly without re-deriving the output
    Jacobian.

    Examples
    --------
    >>> ss = sim.steady_state()
    >>> ss.converged
    True
    >>> ss["A"]  # species A at steady state
    50.0
    >>> ss.concentrations
    array([50., 25., ...])
    >>> ss = sim.steady_state(sensitivity_params=["k_deg"])
    >>> ss.output_sensitivities(["observable:Stot"])  # (n_sel, n_params)
    array([[-1.25]])
    """

    __slots__ = (
        "_concentrations",
        "_species_names",
        "_name_to_idx",
        "residual",
        "method_used",
        "converged",
        "n_steps",
        "n_rhs_evals",
        "n_residual_species",
        "excluded_species",
        "unconverged_pure_sinks",
        "root_stability",
        "n_unstable_roots_rejected",
        "rhs_backend",
        "solver_jacobian_source",
        "solver_jacobian_retried",
        "linear_solver",
        "sens_jacobian_source",
        "sens_dfdp_source",
        "sens_output_source",
        "sens_jacobian_rcond",
        "_sensitivity",
        "_sens_param_names",
        "_observable_names",
        "_expression_names",
        "_observable_sensitivity",
        "_expression_sensitivity",
    )

    def __init__(self, core) -> None:
        import numpy as np

        self._concentrations = np.array(core.concentrations, dtype=np.float64)
        self._species_names = list(core.species_names)
        self._name_to_idx = {n: i for i, n in enumerate(self._species_names)}
        self.residual = core.residual
        self.method_used = core.method_used
        self.converged = core.converged
        self.n_steps = core.n_steps
        self.n_rhs_evals = core.n_rhs_evals
        # Issue #74 — what the convergence test covered, and why it failed.
        self.n_residual_species = getattr(core, "n_residual_species", len(self._species_names))
        self.excluded_species = list(getattr(core, "excluded_species", []))
        self.unconverged_pure_sinks = list(getattr(core, "unconverged_pure_sinks", []))
        # Issue #78 — the linear-stability verdict on a Newton root, and how many
        # candidate roots were thrown out for failing it.
        self.root_stability = getattr(core, "root_stability", "")
        self.n_unstable_roots_rejected = getattr(core, "n_unstable_roots_rejected", 0)
        # Issue #63 — which numerical path ran. getattr-guarded like the GH #12
        # blocks below so an older core stays loadable.
        self.rhs_backend = getattr(core, "rhs_backend", "exprtk")
        # Issue #127 — which Jacobian the march and the polish factored. An
        # older core installed neither callback, hence the default.
        self.solver_jacobian_source = getattr(core, "solver_jacobian_source", "finite-difference")
        self.solver_jacobian_retried = bool(getattr(core, "solver_jacobian_retried", False))
        # Issue #128 — which direct linear solver the march factored with. An
        # older core had no sparse route at all, hence the default.
        self.linear_solver = getattr(core, "linear_solver", "dense")
        self.sens_jacobian_source = getattr(core, "sens_jacobian_source", "")
        self.sens_dfdp_source = getattr(core, "sens_dfdp_source", "")
        # Issue #75 — how d(func)/dp was built: the compiled chain rule, the
        # finite-difference fallback, or "mixed" when it took both.
        self.sens_output_source = getattr(core, "sens_output_source", "")
        self.sens_jacobian_rcond = getattr(core, "sens_jacobian_rcond", 0.0)

        self._sensitivity: np.ndarray | None
        if core.n_sens_params > 0:
            self._sensitivity = np.array(core.sensitivity_data, dtype=np.float64)
            self._sens_param_names = list(core.sens_param_names)
        else:
            self._sensitivity = None
            self._sens_param_names = []

        # GH #12 — observable/expression output sensitivities d(output)/dp at the
        # steady state, populated on a sensitivity run (empty otherwise). The
        # names + blocks parallel Result's: observable_names/expression_names
        # label the rows, and expression_names is already filtered of the
        # auto-generated _rateLawN functions by the pybind layer. Guarded with
        # getattr for forward/backward compatibility with an older core.
        self._observable_names = list(getattr(core, "observable_names", []))
        self._expression_names = list(getattr(core, "expression_names", []))
        self._observable_sensitivity = _ss_output_sens_block(core, "observable_sensitivity_data")
        self._expression_sensitivity = _ss_output_sens_block(core, "expression_sensitivity_data")

    @property
    def concentrations(self):
        """Steady-state species concentrations."""
        return self._concentrations

    @property
    def species_names(self) -> list[str]:
        """Species names."""
        return self._species_names

    @property
    def sensitivity(self):
        """Sensitivity matrix dY_ss/dp, shape (n_species, n_params).

        None if no sensitivity was requested.
        """
        return self._sensitivity

    @property
    def sensitivity_params(self) -> list[str]:
        """Parameter names for sensitivity."""
        return self._sens_param_names

    # ─── Observable / expression output sensitivities (GH #12) ──────────

    @property
    def observable_names(self) -> list[str]:
        """Observable names labelling the observable output-sensitivity rows.

        Populated on a sensitivity run (``sensitivity_params=[...]``); empty
        otherwise. Provided for parity with :attr:`bngsim.Result.observable_names`.
        """
        return self._observable_names

    @property
    def expression_names(self) -> list[str]:
        """Expression (global-function) names for the expression rows.

        Bare, user-facing names (the auto-generated ``_rateLawN`` intermediates
        are filtered out, matching :attr:`bngsim.Result.expression_names`).
        Populated on a sensitivity run; empty otherwise.
        """
        return self._expression_names

    @property
    def sensitivities_observables(self) -> np.ndarray:
        """Observable steady-state sensitivities ``d(observable)/dp``.

        Shape ``(n_observables, n_params)`` on a sensitivity run, aligned with
        :attr:`observable_names` (rows) and :attr:`sensitivity_params` (columns);
        empty ``(0, 0)`` otherwise. This is the exact linear projection of the
        species :attr:`sensitivity` through each observable's group factors —
        including the amount-valued volume factor for an SBML
        ``hasOnlySubstanceUnits="true"`` species, whose observable denotes an
        amount rather than the stored concentration (issue #119). So for such a
        species this block is *not* simply ``sensitivity`` re-weighted by the
        group coefficients; it is the derivative of the observable's value.
        """
        return self._observable_sensitivity

    @property
    def sensitivities_expressions(self) -> np.ndarray:
        """Expression (global-function) steady-state sensitivities ``d(func)/dp``.

        Shape ``(n_expressions, n_params)`` on a sensitivity run, aligned with
        :attr:`expression_names` (rows) and :attr:`sensitivity_params` (columns);
        empty ``(0, 0)`` otherwise. Carries the function's full total derivative
        — the state-chain term ``(∂func/∂x)·dY_ss/dp`` plus the function's
        explicit parameter dependence ``∂func/∂p``.
        """
        return self._expression_sensitivity

    def resolve_outputs(self, selectors: str | Iterable[str]) -> list[dict[str, Any]]:
        """Resolve typed output selectors to structured metadata.

        Same selector grammar as :meth:`bngsim.Result.resolve_outputs`
        (``species:``/``observable:``/``expression:`` with ``state:``/``function:``
        aliases, ``()`` handling, and bare-name uniqueness). Observable and
        expression names resolve only on a sensitivity run.
        """
        return [self._resolve_one_output(sel) for sel in _as_selector_list(selectors)]

    def output_sensitivities(
        self,
        selectors: str | Iterable[str],
        *,
        axis: str = "parameter",
    ) -> np.ndarray:
        """Return steady-state ``d(named output)/dp`` for each selector, stacked.

        The steady-state analogue of :meth:`bngsim.Result.output_sensitivities`:
        resolves each selector and stacks the matching steady-state sensitivity
        row, so a gradient consumer reads ``∂(observable)/∂θ`` /
        ``∂(expression)/∂θ`` directly instead of re-deriving the output Jacobian.

        ``species:`` selectors read the species ``dY_ss/dp`` rows;
        ``observable:`` selectors the exact linear group projection;
        ``expression:`` selectors the finite-difference total derivative of the
        global function (state chain + explicit parameter dependence).

        Parameters
        ----------
        selectors : str or iterable of str
            Selectors accepted by :meth:`resolve_outputs`.
        axis : {"parameter"}, optional
            Only ``"parameter"`` (the default) is meaningful here. A stable
            steady state is independent of its initial conditions
            (``∂x*/∂x(0) = 0``), so the ``"ic"`` axis is structurally zero and is
            not computed; requesting it raises :class:`ValueError`.

        Returns
        -------
        ndarray
            Shape ``(n_selectors, n_params)``, one row per selector in input
            order (no time axis — a steady state is a single point). An empty
            selector list yields a ``(0, n_params)`` array.

        Raises
        ------
        ValueError
            If ``axis`` is not ``"parameter"``; if no parameter sensitivities
            were computed (run with ``sensitivity_params=[...]``); or if a
            selector names a kind whose sensitivities are unavailable.
        TypeError
            Propagated from :meth:`resolve_outputs`.

        Examples
        --------
        >>> ss = sim.steady_state(sensitivity_params=["k_deg"])
        >>> ss.output_sensitivities(["observable:Stot", "expression:foo"]).shape
        (2, 1)
        """
        if axis == "ic":
            raise ValueError(
                "output_sensitivities: the 'ic' (initial-condition) axis is not "
                "available on a steady-state result. A stable steady state forgets "
                "its initial conditions (∂x*/∂x(0) = 0), so IC-axis output "
                "sensitivities are structurally zero and are not computed."
            )
        if axis != "parameter":
            raise ValueError(f"output_sensitivities: axis must be 'parameter', got {axis!r}.")
        if self._sensitivity is None:
            raise ValueError(
                "output_sensitivities: no steady-state sensitivities were computed for "
                "this result. Enable them via sim.steady_state(sensitivity_params=[...])."
            )
        n_params = self._sensitivity.shape[1]
        meta = self.resolve_outputs(selectors)
        if not meta:
            return np.empty((0, n_params), dtype=np.float64)
        rows = [self._output_sensitivity_row(m) for m in meta]
        return np.stack(rows, axis=0)

    def _resolve_one_output(self, selector: str) -> dict[str, Any]:
        """Resolve one selector to its metadata dict via the shared resolver."""
        return _resolve_output_selector(
            selector,
            self._species_names,
            self._observable_names,
            self._expression_names,
        )

    def _output_sensitivity_row(self, meta: dict[str, Any]) -> np.ndarray:
        """``(n_params,)`` steady-state sensitivity row for one resolved output."""
        kind = meta["kind"]
        if kind == "species":
            # self._sensitivity is not None here (checked by output_sensitivities).
            return self._sensitivity[meta["index"], :]  # type: ignore[index]
        block = (
            self._observable_sensitivity if kind == "observable" else self._expression_sensitivity
        )
        if block.size == 0:
            raise ValueError(
                f"output_sensitivities: no {kind} sensitivities are available for "
                f"selector {meta['selector']!r} on this steady-state result."
            )
        return block[meta["index"], :]

    def __getitem__(self, key: str) -> float:
        """Get steady-state concentration by species name."""
        if key not in self._name_to_idx:
            raise KeyError(f"Species '{key}' not found. Available: {self._species_names}")
        return float(self._concentrations[self._name_to_idx[key]])

    def to_dict(self) -> dict[str, float]:
        """Return species concentrations as a dict."""
        return {n: float(self._concentrations[i]) for i, n in enumerate(self._species_names)}

    def __repr__(self) -> str:
        return (
            f"SteadyStateResult("
            f"converged={self.converged}, "
            f"method='{self.method_used}', "
            f"residual={self.residual:.2e}, "
            f"n_species={len(self._concentrations)})"
        )
