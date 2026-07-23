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

    def _raise_if_event_sensitivities(self, param_names: list[str] | None = None) -> None:
        """Refuse output sensitivities only for unsupported event subclasses.

        Originally a blanket refusal on any model with events (GH #205): the
        integrator reinitialises state at an event (``CVodeReInit``) but the
        CVODES forward-sensitivity vectors were never reinitialised, so the
        columns went silently stale at and after the first fire.

        GH #212 lifts the refusal for the **Phase-1 subclass** — fixed-time,
        persistent, no-delay events (``g = time − T``, the dosing/stimulation
        pattern). For that class the event-time sensitivity ``∂t*/∂p = 0`` and
        the core now applies the sensitivity jump ``s⁺ = J_h·s⁻ + ∂h/∂p`` and
        ``CVodeSensReInit`` at each fire. Everything still unsupported keeps
        raising: state-dependent triggers and parameter-valued trigger times
        (Phase 2), delays and non-persistent triggers (Phase 3).

        The classification is delegated to the core
        (:func:`NetworkModel.event_sensitivity_unsupported_reason`), which knows
        each event's persistence/delay and — via the trigger's referenced
        variables — whether it is fixed-time and whether its crossing time
        depends on a requested sensitivity parameter. ``param_names`` is the set
        of parameters whose sensitivities this call requests (defaults to
        ``self._sensitivity_params``); an IC-only request passes an empty list,
        which still exercises the persistence/delay/state-dependence checks.

        Discontinuity triggers (GH #72 forcing pulses) do not jump state and are
        not events, so they are unaffected.
        """
        if self._model._core.n_events <= 0:
            return
        names = list(param_names) if param_names is not None else list(self._sensitivity_params)
        reason = self._model._core.event_sensitivity_unsupported_reason(names)
        if reason:
            raise ValueError(
                "Output sensitivities are not supported for this model's events: "
                + reason
                + " bngsim refuses rather than return silently-stale derivatives "
                "(GH #205). Forward sensitivity through fixed-time / persistent / "
                "no-delay events is supported (GH #212 Phase 1); the remaining "
                "subclasses are tracked there."
            )

    def _apply_ic_param_sens_seed(self, opts, core) -> None:
        """Inject ∂x_i(0)/∂p initial-condition sensitivity seeds (issue #43).

        When a species initial condition is a parameter reference — directly
        (``R() R0``) or through a derived ConstantExpression (``R() Rtot`` with
        ``Rtot = R0``) — the forward-sensitivity seed yS_i(0) must carry the IC
        Jacobian column ∂(IC)/∂p. The C++ seeding cannot differentiate a derived
        IC, so the coefficients are computed from the model's parameter graph via
        the sympy chain rule and passed through ``SolverOptions``. A no-op for the
        common model with no parameter-referenced species ICs, and for IC-only
        sensitivity (no ``sensitivity_params``), where param columns don't exist.
        """
        if not self._sensitivity_params:
            return
        from bngsim._codegen import compute_ic_param_sens_seed

        seeds = compute_ic_param_sens_seed(core)
        if seeds:
            opts.set_ic_param_sens(seeds)

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

    def _stamp(self, result: Result, *, seed: int | None = None) -> Result:
        """Attach per-species V_c and stochastic seed to *result*."""
        vf = self._get_volume_factors()
        if vf:
            result._species_volume_factors = vf
        if seed is not None:
            result._seed = seed
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
                    self._apply_ic_param_sens_seed(opts, self._model._core)
                if self._sensitivity_ic:
                    opts.set_sensitivity_ic(self._sensitivity_ic)
                if self._sensitivity_params or self._sensitivity_ic:
                    opts.set_sensitivity_method(self._sensitivity_method)

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
        result = self._stamp(Result(core_result), seed=used_seed if seed_is_meaningful else None)

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
        the call: the scanned parameter and the reset-target concentrations are
        restored afterward, so a :class:`Simulator` can be scanned repeatedly
        (and the returned trajectories, not the live model, are the product).
        """
        self._require_interactive_backend_support()
        if steady_state and self._method != "ode":
            raise ValueError(
                "steady_state=True is only supported for method='ode' "
                f"(got method='{self._method}')."
            )
        # A scan resets each point to a snapshot (or carries the prior point's
        # state), so the per-point IC is not the model's seed. CVODES forward
        # sensitivities would be mis-seeded across that boundary (∂y(0)/∂θ ≠ 0),
        # so refuse rather than return silently-wrong derivatives — use run_batch
        # for a seed-reset sensitivity scan (it clones + resets each point).
        if self._sensitivity_params or self._sensitivity_ic:
            raise ValueError(
                "parameter_scan / bifurcate do not support output sensitivities: "
                "each point resets to a snapshot rather than the seed initial "
                "conditions, so the forward-sensitivity seed would be wrong across "
                "that boundary. Build a Simulator without sensitivity_params for "
                "the scan, or use run_batch (which resets each point to the seed) "
                "for a sensitivity parameter sweep."
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

        def _reset_point() -> None:
            if use_named:
                self._model.restore_concentrations(reset_to)
            else:
                self._model.set_state(invocation_state)

        base_seed = _resolve_seed(seed) if self._method != "ode" else 0

        results: list[Result] = []
        try:
            for i, value in enumerate(values):
                if reset_conc:
                    _reset_point()
                self._model.set_param(parameter, float(value))
                if on_point is not None:
                    on_point(self._model, float(value))
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
                )
                result.custom_attrs["scan_parameter"] = parameter
                result.custom_attrs["scan_value"] = float(value)
                results.append(result)
        finally:
            # Leave the persistent model + simulator as we found them.
            self._model.set_param(parameter, original_value)
            self._model.set_state(invocation_state)
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
                    self._apply_ic_param_sens_seed(opts, clone._core)
                if self._sensitivity_ic:
                    opts.set_sensitivity_ic(self._sensitivity_ic)
                if self._sensitivity_params or self._sensitivity_ic:
                    opts.set_sensitivity_method(self._sensitivity_method)
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
            If method is not 'ode', or params contains unknown names.
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
        else:
            target_params = list(all_param_names)

        n_params = len(target_params)
        if n_params == 0:
            raise ValueError("No parameters to compute sensitivities for.")

        # GH #205/#212 — events: allowed only for the fixed-time Phase-1
        # subclass, and only when no requested parameter is a trigger time.
        # Classified against this call's actual target parameters.
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
        # SAME auto-codegen the single-shot sensitivity path uses, but only when
        # the model actually has expression (function) outputs: species and
        # observable sensitivities need no codegen, so expression-free models stay
        # on the interpreted path unchanged. The helper is a no-op if a codegen
        # .so/JIT source is already attached (e.g. a large-model or explicit
        # codegen build), so this never double-compiles.
        #
        # GH #205 — the GH #198 output-sensitivity evaluator is emitted only when
        # ``_want_output_sens`` is set (both the .net and model-based codegen paths
        # gate on it), which the constructor does for sensitivity_params runs but
        # this entry point (built without them) does not. compute_all_sensitivities
        # always wants the output blocks, so mark the flag before generating.
        #
        # Two construction-time wrinkles to clear past:
        #   * .net models never auto-codegen at construction (the species-threshold
        #     attach is SBML/builder-only), so the helper below always fires fresh.
        #   * an SBML/builder model CAN already carry a plain-RHS codegen .so/source
        #     from construction (species-threshold attach, explicit codegen=True, or
        #     inherited) — built WITHOUT output sens because _want_output_sens was
        #     then False. The helper no-ops on an already-attached codegen, so that
        #     plain .so would shadow the sensitivity codegen and the expression
        #     block would come back empty. ``_want_output_sens`` (set once at
        #     construction, gating both paths) doubles as the "attached codegen
        #     already has output sens" signal: when it was False, clear the plain
        #     artifact so the helper regenerates with output sens (the result is a
        #     superset; the .so cache keeps a repeat cheap), restoring it if
        #     regeneration produces nothing so the RHS speed-up survives. When it
        #     was already True (a sensitivity_params-built sim), skip the clear so a
        #     large model is not needlessly re-generated.
        if self._model._core.n_functions > 0:
            if self._model._want_output_sens:
                self._auto_codegen_for_sensitivity(jit_backend=_codegen_jit_backend())
            else:
                self._model._want_output_sens = True
                prev_so, prev_src = self._codegen_so_path, self._codegen_c_source
                self._codegen_so_path = ""
                self._codegen_c_source = ""
                self._auto_codegen_for_sensitivity(jit_backend=_codegen_jit_backend())
                if not self._codegen_so_path and not self._codegen_c_source:
                    self._codegen_so_path, self._codegen_c_source = prev_so, prev_src

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
        self._apply_ic_param_sens_seed(opts, clone._core)
        opts.set_sensitivity_method(self._sensitivity_method)

        try:
            core_result = sim.run(ts, opts)
        except RuntimeError as e:
            raise SimulationError(f"Sensitivity chunk failed (params={sens_params}): {e}") from e

        return self._stamp(Result(core_result))

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
    ):
        """Find the steady state of the ODE system f(y) = 0.

        Solver methods:

        - ``"integration"`` (default): CVODE BDF integrated until the BNG2.pl
          parity criterion ``||f(y)||_2 / n_species < tol``
          (``run_network -c``).
        - ``"newton"``: two-tier integrate-first solver. The *same* CVODE burst
          carries the state into the physical root's basin, then KINSOL
          polishes with an analytical Jacobian; the polish is accepted only
          once it is *seed-stable* (agrees across two successively tighter
          bursts), otherwise integration continues. Seeding Newton at the raw
          initial condition instead can converge to a spurious root of
          ``f(y)=0`` the trajectory never reaches, or walk a species negative
          into ``NaN`` (GH #27) — hence the burst.
        - ``"kinsol"``: accepted alias for ``"newton"``.

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
            Parameter names for dY_ss/dp sensitivity.

        Returns
        -------
        SteadyStateResult
        """
        if self._method != "ode":
            raise ValueError(
                f"steady_state() is only supported for method='ode', not method='{self._method}'."
            )

        # GH #205/#212 — dY_ss/dp on event models: allowed only for the
        # fixed-time Phase-1 subclass with no parameter-valued trigger time,
        # classified against this call's requested sensitivity_params.
        if sensitivity_params:
            self._raise_if_event_sensitivities(sensitivity_params)

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
        opts.jacobian = self._jacobian
        if sensitivity_params:
            opts.sensitivity_params = list(sensitivity_params)

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
            "Steady state %s: method=%s, residual=%.2e, steps=%d",
            "converged" if result.converged else "FAILED",
            result.method_used,
            result.residual,
            result.n_steps,
        )
        return result

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

        Returns
        -------
        list[SteadyStateResult]
        """
        if self._method != "ode":
            raise ValueError(
                "steady_state_batch() is only supported for "
                f"method='ode', not method='{self._method}'."
            )
        if not params:
            raise ValueError("params must be non-empty")

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
            opts.jacobian = self._jacobian
            try:
                core_result = find_steady_state(clone._core, opts)
            except RuntimeError as e:
                raise SimulationError(f"Batch {i} failed: {e}") from e
            return SteadyStateResult(core_result)

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
    sensitivity : ndarray or None
        Species ``dY_ss/dp`` matrix, shape ``(n_species, n_params)``. ``None``
        if no sensitivity was requested.

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
        species :attr:`sensitivity` through each observable's group factors.
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
