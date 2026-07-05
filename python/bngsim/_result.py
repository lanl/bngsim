"""bngsim.Result — Rich simulation result container.

Result wraps the C++ Result and provides:
- NumPy arrays for time, species, observables
- Named observable/species access via dict-like indexing
- Optional pandas DataFrame property
- Solver diagnostics
- File export (.gdat, .cdat)
- HDF5 save/load (requires h5py)
- Squeeze semantics: single sim → 2D, batch → 3D arrays
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bngsim._bngsim_core import ResultCore

logger = logging.getLogger("bngsim")


@dataclass(frozen=True)
class IdentifiabilityReport:
    r"""Model-identifiability readout of a Fisher Information Matrix (GH #202).

    A pure function of the model + output-sensitivity tensor (no data, no
    residuals, no objective): the eigen-decomposition of the FIM and the
    practical-identifiability flags derived from it. This is sloppiness /
    practical-identifiability analysis of the *model*, independent of any fit.

    Produced by :meth:`Result.identifiability`.

    Attributes
    ----------
    fim : ndarray, shape ``(n, n)``
        The (symmetric, positive-semidefinite) Fisher Information Matrix the
        readout was computed from, over ``n`` parameters (or IC species).
    parameters : list of str
        Names labelling the FIM axes — the :attr:`Result.sensitivity_params`
        (``axis="parameter"``) or :attr:`Result.sensitivity_ic_species`
        (``axis="ic"``) in matrix order.
    eigenvalues : ndarray, shape ``(n,)``
        FIM eigenvalues in **ascending** order. Tiny negative values from
        round-off are clipped to ``0``. ``eigenvalues[0]`` is the smallest —
        the **sloppiest** (least-constrained) direction.
    eigenvectors : ndarray, shape ``(n, n)``
        Orthonormal eigenvectors as **columns**, aligned with
        :attr:`eigenvalues`: ``eigenvectors[:, i]`` is the parameter
        combination for ``eigenvalues[i]``. A near-zero eigenvalue marks its
        eigenvector as a practically non-identifiable parameter combination.
    rank : int
        Numerical rank — the number of eigenvalues above :attr:`threshold`.
    condition_number : float
        :math:`\lambda_\max / \lambda_\min`; ``inf`` when rank-deficient. A
        large value signals an ill-conditioned (sloppy) parameter set.
    identifiable : ndarray of bool, shape ``(n,)``
        Per-eigen-direction flag (aligned with :attr:`eigenvalues`): ``True``
        where the eigenvalue exceeds :attr:`threshold`, ``False`` for a
        practically non-identifiable / sloppy direction.
    non_identifiable_directions : list of int
        Indices into :attr:`eigenvalues` / :attr:`eigenvectors` columns that
        are flagged non-identifiable (the ``False`` entries of
        :attr:`identifiable`), smallest eigenvalue first.
    cramer_rao_bound : ndarray, shape ``(n, n)``
        :math:`\text{FIM}^{-1}` — the Cramér–Rao **lower bound** on parameter
        variance, an identifiability aid only (a property of the FIM). This is
        **not** a data/noise-weighted fit covariance. All-``NaN`` when the FIM
        is rank-deficient (the inverse is undefined; a warning is emitted).
    threshold : float
        Eigenvalue cutoff (``rtol * lambda_max``) below which a direction is
        flagged non-identifiable.
    """

    fim: NDArray[np.float64]
    parameters: list[str]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    rank: int
    condition_number: float
    identifiable: NDArray[np.bool_]
    non_identifiable_directions: list[int]
    cramer_rao_bound: NDArray[np.float64]
    threshold: float

    @property
    def is_identifiable(self) -> bool:
        """True when every direction is identifiable (full-rank FIM)."""
        return self.rank == self.fim.shape[0]

    def __repr__(self) -> str:  # noqa: D105 — concise readout, not full arrays
        n = self.fim.shape[0]
        cond = "inf" if not np.isfinite(self.condition_number) else f"{self.condition_number:.3g}"
        return (
            f"IdentifiabilityReport(n={n}, rank={self.rank}, "
            f"condition_number={cond}, "
            f"non_identifiable_directions={self.non_identifiable_directions})"
        )


class Result:
    """Simulation result container.

    Attributes
    ----------
    time : ndarray, shape (n_times,)
        Time points.
    species : ndarray, shape (n_times, n_species)
        Species concentrations at each time point.
    observables : ndarray, shape (n_times, n_observables)
        Observable values at each time point.
    solver_stats : dict
        Solver diagnostics (n_steps, n_rhs_evals, etc.).

    Examples
    --------
    >>> result = sim.run(t_span=(0, 100), n_points=101)
    >>> result.time.shape
    (101,)
    >>> result.observables.shape
    (101, 5)
    >>> result.observables["Atot"]  # named access
    array([100., 99.5, ...])
    """

    __slots__ = (
        "_time",
        "_species",
        "_observables",
        "_expressions",
        "_sensitivities",
        "_sensitivities_ic",
        "_observable_sensitivities",
        "_expression_sensitivities",
        "_observable_sensitivities_ic",
        "_expression_sensitivities_ic",
        "_expression_sens_support",
        "_species_names",
        "_observable_names",
        "_expression_names",
        "_sensitivity_params",
        "_sensitivity_ic_species",
        "_solver_stats",
        "_ssa_diagnostics",
        "_species_volume_factors",
        "_varvol_live_vol",
        "_varvol_conc_factor",
        "_varvol_amount_factor",
        "_ar_sens_map",
        "_ar_sens_blocked",
        "_seed",
        "_core",
        "custom_attrs",
    )

    def __init__(
        self,
        core: ResultCore | None = None,
        *,
        custom_attrs: dict[str, Any] | None = None,
        # Allow construction from raw arrays (for load / batch stacking)
        _time: NDArray | None = None,
        _species: NDArray | None = None,
        _observables: NDArray | None = None,
        _expressions: NDArray | None = None,
        _species_names: list[str] | None = None,
        _observable_names: list[str] | None = None,
        _expression_names: list[str] | None = None,
        _solver_stats: dict[str, int] | None = None,
        _species_volume_factors: list[float] | None = None,
        _seed: int | None = None,
        # Sensitivity blocks (load / batch / stitch). All optional; absent ⇒
        # the empty (0, 0, 0) block. The observable/expression blocks reuse the
        # species parameter / IC-species name lists (GH #196).
        _sensitivities: NDArray | None = None,
        _sensitivity_params: list[str] | None = None,
        _sensitivities_ic: NDArray | None = None,
        _sensitivity_ic_species: list[str] | None = None,
        _observable_sensitivities: NDArray | None = None,
        _expression_sensitivities: NDArray | None = None,
        _observable_sensitivities_ic: NDArray | None = None,
        _expression_sensitivities_ic: NDArray | None = None,
    ) -> None:
        self._core = core

        if core is not None:
            # Construct from C++ core Result
            self._time: NDArray[np.float64] = core.time
            self._species: NDArray[np.float64] = core.species_data
            self._observables: NDArray[np.float64] = core.observable_data
            self._species_names: list[str] = list(core.species_names)
            self._observable_names: list[str] = list(core.observable_names)

            # Expression data
            if hasattr(core, "expression_data") and core.n_expressions > 0:
                self._expressions: NDArray[np.float64] = core.expression_data
                self._expression_names: list[str] = list(core.expression_names)
            else:
                self._expressions = np.empty((self._time.shape[0], 0))
                self._expression_names = []

            # Sensitivity data (CVODES forward sensitivities)
            if hasattr(core, "n_sens_params") and core.n_sens_params > 0:
                self._sensitivities: NDArray[np.float64] = core.sensitivity_data
                self._sensitivity_params: list[str] = list(core.sens_param_names)
            else:
                self._sensitivities = np.empty((0, 0, 0))
                self._sensitivity_params = []

            # IC sensitivity data
            if hasattr(core, "n_sens_ic_species") and core.n_sens_ic_species > 0:
                self._sensitivities_ic: NDArray[np.float64] = core.sensitivity_ic_data
                self._sensitivity_ic_species: list[str] = list(core.sens_ic_species_names)
            else:
                self._sensitivities_ic = np.empty((0, 0, 0))
                self._sensitivity_ic_species = []

            # GH #196 — observable/expression output sensitivities. Storage only
            # (no computation path yet), so these are empty (0, 0, 0) on every
            # current run; the accessors return that shape until a later stage
            # populates the C++ blocks. The param/IC axes are shared with the
            # species blocks above (no separate name lists).
            self._observable_sensitivities: NDArray[np.float64] = _core_sens_block(
                core, "observable_sensitivity_data"
            )
            self._expression_sensitivities: NDArray[np.float64] = _core_sens_block(
                core, "expression_sensitivity_data"
            )
            self._observable_sensitivities_ic: NDArray[np.float64] = _core_sens_block(
                core, "observable_sensitivity_ic_data"
            )
            self._expression_sensitivities_ic: NDArray[np.float64] = _core_sens_block(
                core, "expression_sensitivity_ic_data"
            )

            # Solver stats. ``steady_state_reached`` is the int 0/1 marker
            # for whether the BNG-style early-stop fired (False on every
            # backend except CVODE with steady_state=True).
            stats = core.solver_stats
            self._solver_stats: dict[str, int] = {
                "n_steps": stats.n_steps,
                "n_rhs_evals": stats.n_rhs_evals,
                "n_jac_evals": stats.n_jac_evals,
                "n_err_test_fails": stats.n_err_test_fails,
                "n_nonlin_iters": stats.n_nonlin_iters,
                "n_nonlin_conv_fails": stats.n_nonlin_conv_fails,
                # Which direct linear solver the ODE engine actually used, as the
                # ``LinearSolverKind`` enum (include/bngsim/result.hpp). Pinned
                # mapping — DO NOT reinterpret as a SUNDIALS solver-id:
                #   0 = built-in dense LU   (SUNLinSol_Dense)
                #   1 = KLU sparse          (SUNLinSol_KLU)   ← NOT "Band"
                #   2 = BLAS dgetrf dense   (GH #84, opt-in)
                # Auto rule (cvode_simulator.cpp choose_linear_solver_kind):
                # KLU when N>=50 and Jacobian density<10%, else dense. Set in both
                # the cold and warm run paths and for both the sparse and dense
                # branches, so every ODE Result carries the real solver kind.
                "linear_solver": stats.linear_solver,
                # GH #132 adaptive gate: of the dense factorizations, how many took
                # the BLAS dgetrf path. 0 unless linear_solver==2 (LAPACK-dense)
                # AND the run crossed the K-factorization gate (K=5 default, N>=256)
                # — a short LAPACK-dense run stays on the built-in dense LU and
                # reports 0 here. Lets a benchmark tell "dgetrf actually ran" from
                # "LAPACK-dense mode but gate never crossed".
                "n_dense_blas_factorizations": stats.n_dense_blas_factorizations,
                "steady_state_reached": int(bool(stats.steady_state_reached)),
            }

            # GH #110 — SSA boundary diagnostics. Zero/empty on every non-SSA
            # backend (the C++ struct is default-constructed there).
            ssa_diag = core.ssa_diagnostics
            self._ssa_diagnostics: dict[str, Any] = {
                "n_negative_crossings": ssa_diag.n_negative_crossings,
                "first_negative_species": ssa_diag.first_negative_species,
                "n_reverse_fires": ssa_diag.n_reverse_fires,
                "first_reverse_reaction": ssa_diag.first_reverse_reaction,
                # GH #190 — how propensities were evaluated: "cc"/"mir"
                # (compiled recompute-all) or "interpreted".
                "propensity_backend": getattr(ssa_diag, "propensity_backend", "interpreted"),
            }
        else:
            # Construct from raw arrays (load / batch)
            self._time = _time if _time is not None else np.empty(0)
            self._species = _species if _species is not None else np.empty((0, 0))
            self._observables = _observables if _observables is not None else np.empty((0, 0))
            self._expressions = _expressions if _expressions is not None else np.empty((0, 0))
            self._species_names = _species_names or []
            self._observable_names = _observable_names or []
            self._expression_names = _expression_names or []
            # Sensitivity blocks (GH #196): honor the kwargs when given (load /
            # batch / stitch), else default to the empty (0, 0, 0) block.
            self._sensitivities = _empty_sens_block(_sensitivities)
            self._sensitivity_params = _sensitivity_params or []
            self._sensitivities_ic = _empty_sens_block(_sensitivities_ic)
            self._sensitivity_ic_species = _sensitivity_ic_species or []
            self._observable_sensitivities = _empty_sens_block(_observable_sensitivities)
            self._expression_sensitivities = _empty_sens_block(_expression_sensitivities)
            self._observable_sensitivities_ic = _empty_sens_block(_observable_sensitivities_ic)
            self._expression_sensitivities_ic = _empty_sens_block(_expression_sensitivities_ic)
            self._solver_stats = _solver_stats or {
                "n_steps": 0,
                "n_rhs_evals": 0,
                "n_jac_evals": 0,
                "n_err_test_fails": 0,
                "n_nonlin_iters": 0,
                "n_nonlin_conv_fails": 0,
                "linear_solver": 0,
                "n_dense_blas_factorizations": 0,
                "steady_state_reached": 0,
            }
            self._ssa_diagnostics = {
                "n_negative_crossings": 0,
                "first_negative_species": "",
                "n_reverse_fires": 0,
                "first_reverse_reaction": "",
                "propensity_backend": "interpreted",
            }

        # GH #198 — {function_name: unsupported_reason_or_None} for expression
        # output sensitivities, set by the Simulator after a sensitivity run from
        # codegen analysis. A reason string lets output_sensitivities() raise the
        # specific cause (unsupported construct, table function) instead of a
        # bare empty-block error. Empty ⇒ no info (e.g. a loaded result).
        self._expression_sens_support: dict[str, str | None] = {}

        # Per-species V_c, used by as_roadrunner() to convert stored
        # values (concentration = amount/V_c, per Phase 2's storage
        # convention) to amounts when the caller selects an `X` column.
        # Populated by Simulator.run; None means "treat as 1.0" which is
        # correct for V=1 SBML and all .net models.
        self._species_volume_factors: list[float] | None = _species_volume_factors

        # GH #85: species_idx → live-volume column idx for species whose
        # concentration column has been rescaled to amount/V_live(t) (a
        # variable-volume compartment). Populated by Simulator._stamp. Tells
        # as_roadrunner() to recover the amount as conc * V_live(t) rather than
        # conc * V_static for those species. None ⇒ no variable-volume species.
        self._varvol_live_vol: dict[int, int] | None = None

        # GH #131: species_idx → per-sample concentration factor V_static/V_live(t)
        # for a variable-volume species whose raw column is LEFT as the conserved
        # count amount/V_static (the stochastic convention) rather than rescaled
        # in place. as_roadrunner() multiplies it into the ``[S]`` (concentration)
        # selector so [S] reports amount/V_live, while the bare ``S`` (amount)
        # selector recovers the amount as raw·V_static via the volume factor.
        # Also carries the hOSU=true event-resize concentration correction, which
        # applies under ODE too (raw stays amount/V_static there as well). None ⇒
        # no such species.
        self._varvol_conc_factor: dict[int, np.ndarray] | None = None

        # GH #131: species_idx → per-sample live-volume V_live(t) for a species
        # whose raw column holds the live concentration amount/V_live (the ODE
        # convention for an hOSU=false EVENT-resized species — the #74 V_old/V_new
        # rescale already ran), so the bare ``X`` amount selector reports the
        # amount as raw·V_live rather than the stale raw·V_static the volume factor
        # would give. V_live is read from the event-promoted compartment's
        # observable column. Takes priority over the species-column live_vol and
        # the static volume factor in as_roadrunner. None ⇒ no such species.
        self._varvol_amount_factor: dict[int, np.ndarray] | None = None

        # GH #205: SBML AssignmentRule-target species → (kind, src, vdiv), the
        # same map Simulator._apply_ar_report_map uses to overwrite the frozen
        # species value column with its rule's live value. An AR species' OUTPUT
        # sensitivity must follow that assignment expression (the observable for
        # a linear-on-species rule, the function/expression otherwise), not the
        # raw integrated state sensitivity — output_sensitivities("species:<ar>")
        # redirects through this. The raw tensor stays as sensitivities_species.
        # Empty for .net and non-AR models (no redirect). Set by Simulator._stamp.
        self._ar_sens_map: dict[str, tuple[str, str, float]] = {}

        # GH #205: AR-species names whose reported value also carries a
        # time-varying volume rescale (variable-volume compartment, #85/#87). The
        # redirect above accounts only for the constant vdiv, so those species'
        # output sensitivities are refused rather than returned subtly wrong.
        self._ar_sens_blocked: frozenset[str] = frozenset()

        # Stochastic seed actually used for this simulation. None for
        # ODE results (deterministic) and for results loaded from older
        # files that pre-date seed exposure.
        self._seed: int | None = _seed

        self.custom_attrs: dict[str, Any] = custom_attrs or {}

    # ─── Core data ──────────────────────────────────────────────────

    @property
    def seed(self) -> int | None:
        """The integer RNG seed actually used for this simulation.

        ``None`` for deterministic (ODE) results and for results loaded
        from HDF5 files predating seed exposure. For stochastic methods
        (SSA, PSA, NFsim, RuleMonkey), this is the integer that was
        passed down to the backend — equal to the ``seed=`` keyword
        when the caller supplied one, or the freshly drawn integer
        when the caller passed ``seed=None`` (or omitted it).

        For squeezed batch results, this is the single seed if every
        underlying sim used the same one, otherwise ``None``; the
        per-sim seeds remain accessible on the individual ``Result``
        objects from ``Simulator.run_batch(..., squeeze=False)``.
        """
        return self._seed

    @property
    def time(self) -> NDArray[np.float64]:
        """Time points, shape ``(n_times,)``."""
        return self._time

    @property
    def species(self) -> NDArray[np.float64]:
        """Species concentrations, shape ``(n_times, n_species)``."""
        return self._species

    @property
    def observables(self) -> _ObservableAccessor:
        """Observable values with named access.

        Can be used as:
        - ``result.observables`` → full array ``(n_times, n_obs)``
        - ``result.observables["name"]`` → single column ``(n_times,)``
        """
        return _ObservableAccessor(self._observables, self._observable_names)

    @property
    def expressions(self) -> _ObservableAccessor:
        """Expression (function) values with named access."""
        return _ObservableAccessor(self._expressions, self._expression_names)

    # ─── Dimensions ─────────────────────────────────────────────────

    @property
    def n_times(self) -> int:
        """Number of time points."""
        return self._time.shape[0]

    @property
    def n_species(self) -> int:
        """Number of species."""
        if self._species.ndim == 2:
            return self._species.shape[1]
        return 0

    @property
    def n_observables(self) -> int:
        """Number of observables."""
        if self._observables.ndim == 2:
            return self._observables.shape[1]
        return 0

    @property
    def n_expressions(self) -> int:
        """Number of expressions."""
        if self._expressions.ndim == 2:
            return self._expressions.shape[1]
        return 0

    @property
    def has_simulation_data(self) -> bool:
        """True iff the simulator produced at least one numeric column.

        Returns ``False`` for runs that complete without any species,
        observable, or expression columns — e.g. a BNGL file with a
        ``simulate`` action but an empty model body. Used by batch
        harnesses to distinguish "ran and produced data" from "ran and
        produced nothing meaningful" without an exception path.
        """
        return self.n_species > 0 or self.n_observables > 0 or self.n_expressions > 0

    # ─── Names ──────────────────────────────────────────────────────

    @property
    def species_names(self) -> list[str]:
        """Species names."""
        return self._species_names

    @property
    def observable_names(self) -> list[str]:
        """Observable names."""
        return self._observable_names

    @property
    def expression_names(self) -> list[str]:
        """Expression (function) names (bare, in-memory column keys).

        Auto-generated ``_rateLawN`` rate-law intermediates are filtered
        out; recover them via :attr:`raw_expression_names`.
        """
        return self._expression_names

    @property
    def gdat_expression_names(self) -> list[str]:
        """Function-column labels for ``.gdat``/``.scan`` headers.

        bngsim emits **bare** function headers (no ``()`` suffix, issue
        #58), so this is identical to :attr:`expression_names`. It is
        retained for consumers that assemble a BNG-native file header
        (e.g. a consumer-built ``.scan``) and want an intent-revealing
        name.
        """
        return [n for n in self._expression_names if not _is_auto_rate_law(n)]

    @property
    def raw_expression_names(self) -> list[str]:
        """Unfiltered function names, including internal ``_rateLawN``.

        On a freshly-simulated result these include the auto-generated
        ``_rateLawN`` rate-law intermediates that :attr:`expression_names`
        filters out (the #58 recoverability fix). On a result loaded from
        HDF5 or assembled from raw arrays, only the filtered (bare) set was
        persisted, so this returns the same columns as
        :attr:`expression_names`.
        """
        if self._core is not None:
            return list(self._core.raw_expression_names)
        return list(self._expression_names)

    @property
    def raw_expressions(self) -> _ObservableAccessor:
        """Unfiltered function values, including internal ``_rateLawN``.

        Columns correspond to :attr:`raw_expression_names`. See that
        property for the freshly-simulated vs. loaded distinction.
        """
        if self._core is not None:
            return _ObservableAccessor(
                self._core.raw_expression_data, list(self._core.raw_expression_names)
            )
        return _ObservableAccessor(self._expressions, self._expression_names)

    @property
    def raw_n_expressions(self) -> int:
        """Number of unfiltered function columns (includes ``_rateLawN``)."""
        if self._core is not None:
            return int(self._core.raw_n_expressions)
        return self.n_expressions

    # ─── Output selectors (GH #195) ────────────────────────────────

    def resolve_outputs(
        self,
        selectors: str | Iterable[str],
    ) -> list[dict[str, Any]]:
        """Resolve typed output selectors to structured metadata.

        A *selector* names one output column of this result. Fitting
        frontends compare against *named* outputs (species, observables,
        global functions) rather than raw column indices; this is the
        unified, typed lookup that maps a selector string onto the column
        it refers to. No sensitivity or value computation happens here —
        it is pure name resolution.

        Parameters
        ----------
        selectors : str or iterable of str
            One selector or a sequence of them. A single string is treated
            as a one-element list (so ``resolve_outputs("observable:Atot")``
            and ``resolve_outputs(["observable:Atot"])`` are equivalent).

            Accepted forms:

            - ``"species:<name>"`` — a species column.
            - ``"observable:<name>"`` — an observable column.
            - ``"expression:<name>"`` — a global-function / expression
              column. A trailing ``()`` is stripped
              (``"expression:foo()"`` → ``"expression:foo"``), matching the
              BNG ``.gdat``/``.scan`` header convention while in-memory
              column keys stay bare (issue #58).
            - Aliases: ``"state:"`` → ``"species:"``,
              ``"function:"`` → ``"expression:"``.
            - A **bare** name (no prefix) resolves only if it is unique
              across species, observables, and expressions. A bare
              ``"foo()"`` that does not match any column verbatim is retried
              as the expression ``foo`` (the function-call convention).

        Returns
        -------
        list of dict
            One dict per input selector, in input order, each with keys:

            - ``"selector"`` — the canonical typed selector
              (``"<kind>:<name>"``), suitable as a stable downstream key.
            - ``"kind"`` — ``"species"``, ``"observable"`` or
              ``"expression"``.
            - ``"name"`` — the bare in-memory column name (the key used by
              the named accessors, e.g. ``result.observables[name]``).
            - ``"index"`` — column index within that kind's array.
            - ``"column_label"`` — the label as written to a
              ``.gdat``/``.cdat`` header. bngsim emits **bare** headers
              (issue #58), so this is currently identical to ``"name"``.

        Raises
        ------
        ValueError
            If a selector uses an unknown kind prefix, is empty, is
            unresolved, or is a bare name that matches more than one column.
            Ambiguity errors list every matching typed selector; unresolved
            errors list the available candidates.
        TypeError
            If a selector is not a string.

        Examples
        --------
        >>> result.resolve_outputs("observable:Atot")
        [{'selector': 'observable:Atot', 'kind': 'observable',
          'name': 'Atot', 'index': 0, 'column_label': 'Atot'}]
        >>> [m["selector"] for m in result.resolve_outputs(
        ...     ["species:A()", "function:scaled()", "Btot"])]
        ['species:A()', 'expression:scaled', 'observable:Btot']
        """
        return [self._resolve_one_output(sel) for sel in _as_selector_list(selectors)]

    def outputs(
        self,
        selectors: str | Iterable[str],
    ) -> NDArray[np.float64]:
        """Return the value columns named by *selectors*.

        Thin value accessor on top of :meth:`resolve_outputs`: resolves
        each selector and stacks the named columns into a single array.

        Parameters
        ----------
        selectors : str or iterable of str
            Selectors accepted by :meth:`resolve_outputs`.

        Returns
        -------
        ndarray
            Shape ``(n_times, n_outputs)`` for a single-simulation result,
            with one column per selector in input order. For a stacked
            batch result (3-D arrays) the shape is
            ``(n_sims, n_times, n_outputs)``. An empty selector list yields
            a ``(n_times, 0)`` array.

        Raises
        ------
        ValueError, TypeError
            Propagated from :meth:`resolve_outputs`.
        """
        meta = self.resolve_outputs(selectors)
        if not meta:
            return np.empty(self._time.shape + (0,), dtype=np.float64)
        cols = [self._array_for_kind(m["kind"])[..., m["index"]] for m in meta]
        return np.stack(cols, axis=-1)

    def output_sensitivities(
        self,
        selectors: str | Iterable[str],
        *,
        axis: str = "parameter",
    ) -> NDArray[np.float64]:
        """Return ``d(named output)/dθ`` for each selector, stacked.

        Selector-addressed companion to :meth:`outputs`: resolves each
        selector and stacks the matching sensitivity slice — the
        chain-rule derivative of that output column with respect to the
        sensitivity *parameters* (default) or the differentiated *initial
        conditions*.

        Observable sensitivities are computed at simulation time from the
        CVODES species sensitivities via the linear chain rule
        ``d obs_j/dθ = Σ_i c_ji·dx_i/dθ`` (GH #197); ``expression:`` selectors
        carry the codegen output-sensitivity chain rule for global functions
        (GH #198). ``species:`` selectors read the species sensitivities
        directly — *except* for an SBML AssignmentRule-target species, whose
        reported value is the rule's live value (its state slot is frozen), so
        its output sensitivity follows the assignment expression: the
        sensitivity of the rule's observable (linear-on-species rule) or
        function (everything else), GH #205. The raw integrated-state
        sensitivity for such a species stays available as the low-level
        :attr:`sensitivities_species` tensor.

        Parameters
        ----------
        selectors : str or iterable of str
            Selectors accepted by :meth:`resolve_outputs`.
        axis : {"parameter", "ic"}, optional
            Which sensitivity axis to return. ``"parameter"`` (default)
            gives ``d output/dp`` over :attr:`sensitivity_params`;
            ``"ic"`` gives ``d output/dY(0)`` over
            :attr:`sensitivity_ic_species`.

        Returns
        -------
        ndarray
            Shape ``(n_times, n_outputs, n_axis)`` for a single-simulation
            result (``(n_sims, n_times, n_outputs, n_axis)`` for a stacked
            batch), one slice per selector in input order. ``n_axis`` is
            the number of sensitivity parameters (or IC species). An empty
            selector list yields a ``(n_times, 0, n_axis)`` array.

        Raises
        ------
        ValueError
            If ``axis`` is invalid; if the requested sensitivity axis was
            not computed for this result; or if a selector names a kind
            whose sensitivities are unavailable (e.g. an ``expression:``
            selector before GH #198).
        TypeError
            Propagated from :meth:`resolve_outputs`.

        Examples
        --------
        >>> sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        >>> result = sim.run(t_span=(0, 10), n_points=11)
        >>> result.output_sensitivities("observable:Atot").shape
        (11, 1, 1)
        """
        if axis not in ("parameter", "ic"):
            raise ValueError(
                f"output_sensitivities: axis must be 'parameter' or 'ic', got {axis!r}."
            )
        names = self._sensitivity_params if axis == "parameter" else self._sensitivity_ic_species
        n_axis = len(names)
        if n_axis == 0:
            if axis == "parameter":
                raise ValueError(
                    "output_sensitivities: no parameter sensitivities were computed for this "
                    "result. Enable them via Simulator(..., sensitivity_params=[...])."
                )
            raise ValueError(
                "output_sensitivities: no initial-condition sensitivities were computed for this "
                "result. Enable them via Simulator(..., sensitivity_ic=[...])."
            )
        meta = self.resolve_outputs(selectors)
        if not meta:
            return np.empty(self._time.shape + (0, n_axis), dtype=np.float64)
        cols = [self._output_sensitivity_slice(m, axis) for m in meta]
        return np.stack(cols, axis=-2)

    def _resolve_one_output(self, selector: str) -> dict[str, Any]:
        """Resolve a single selector string to its metadata dict."""
        if not isinstance(selector, str):
            raise TypeError(
                f"Output selector must be a string, got {type(selector).__name__}: {selector!r}."
            )
        sel = selector.strip()
        if not sel:
            raise ValueError("Empty output selector.")

        if ":" in sel:
            prefix, name = sel.split(":", 1)
            prefix = prefix.strip().lower()
            name = name.strip()
            if prefix not in _SELECTOR_PREFIX_ALIASES:
                valid = ", ".join(f"{p}:" for p in _SELECTOR_KINDS)
                raise ValueError(
                    f"Unknown selector kind {prefix + ':'!r} in {selector!r}. "
                    f"Valid prefixes: {valid} "
                    f"(aliases: state:→species:, function:→expression:)."
                )
            kind = _SELECTOR_PREFIX_ALIASES[prefix]
            # `()` is the function-call header convention; strip it only for
            # expressions. Species names legitimately carry "()" (e.g. "A()"),
            # so stripping there would break resolution.
            if kind == "expression" and name.endswith("()"):
                name = name[:-2].strip()
            names = self._names_for_kind(kind)
            if name in names:
                return _output_meta(kind, name, names.index(name))
            raise ValueError(
                f"Unresolved selector {selector!r}: no {kind} named {name!r}. "
                f"Available {kind} names: {names}"
            )

        # Bare name: resolve only if unique across all kinds.
        matches = [
            (kind, self._names_for_kind(kind).index(sel))
            for kind in _SELECTOR_KINDS
            if sel in self._names_for_kind(kind)
        ]
        if len(matches) == 1:
            kind, idx = matches[0]
            return _output_meta(kind, sel, idx)
        if len(matches) > 1:
            typed = ", ".join(f"{kind}:{sel}" for kind, _ in matches)
            raise ValueError(
                f"Ambiguous output selector {sel!r}: matches {typed}. "
                f"Use a typed selector to disambiguate."
            )
        # No verbatim match. A bare "foo()" is the function-call convention:
        # retry the stripped name against expressions.
        if sel.endswith("()"):
            base = sel[:-2].strip()
            expr_names = self._expression_names
            if base in expr_names:
                return _output_meta("expression", base, expr_names.index(base))
        raise ValueError(
            f"Unresolved output selector {selector!r}: not found among "
            f"species, observables, or expressions.\n"
            f"  species:     {self._species_names}\n"
            f"  observables: {self._observable_names}\n"
            f"  expressions: {self._expression_names}"
        )

    def _names_for_kind(self, kind: str) -> list[str]:
        """Name list for a selector kind (species/observable/expression)."""
        if kind == "species":
            return self._species_names
        if kind == "observable":
            return self._observable_names
        return self._expression_names

    def _array_for_kind(self, kind: str) -> NDArray[np.float64]:
        """Value array for a selector kind (species/observable/expression)."""
        if kind == "species":
            return self._species
        if kind == "observable":
            return self._observables
        return self._expressions

    def _sensitivity_block_for(self, kind: str, axis: str) -> NDArray[np.float64]:
        """Sensitivity block for a selector kind and axis (parameter/ic)."""
        if axis == "parameter":
            if kind == "species":
                return self._sensitivities
            if kind == "observable":
                return self._observable_sensitivities
            return self._expression_sensitivities
        if kind == "species":
            return self._sensitivities_ic
        if kind == "observable":
            return self._observable_sensitivities_ic
        return self._expression_sensitivities_ic

    def _output_sensitivity_slice(self, meta: dict[str, Any], axis: str) -> NDArray[np.float64]:
        """``(..., n_times, n_axis)`` sensitivity slice for one resolved output."""
        kind = meta["kind"]
        # GH #205 — an SBML AssignmentRule-target species reports its rule's live
        # value (the species state slot is emitted ``fixed``, frozen at t=0, and
        # the value path overwrites the column from the rule's observable /
        # function). Its OUTPUT sensitivity must follow that assignment
        # expression — the sensitivity of the observable (linear-on-species rule,
        # GH #197) or the function/expression (everything else, GH #198) — not
        # the raw integrated ``yS`` of the frozen state (which is ~0). The raw
        # tensor stays available as ``Result.sensitivities_species``. Recurse on
        # the source selector so its support / empty-block errors propagate.
        if kind == "species":
            redirect = self._ar_sens_map.get(meta["name"])
            if redirect is not None:
                if meta["name"] in self._ar_sens_blocked:
                    raise ValueError(
                        f"output_sensitivities: assignment-rule species "
                        f"{meta['selector']!r} is reported with a time-varying "
                        "volume rescale (variable-volume compartment); its "
                        "output sensitivity is not supported (GH #205). Use "
                        "Result.sensitivities_species for the raw integrated "
                        "state sensitivity."
                    )
                src_kind, src_name = redirect[0], redirect[1]
                vdiv = redirect[2] if len(redirect) > 2 else 1.0
                src_names = self._names_for_kind(src_kind)
                if src_name in src_names:
                    src_meta = _output_meta(src_kind, src_name, src_names.index(src_name))
                    sl = self._output_sensitivity_slice(src_meta, axis)
                    return sl / vdiv if vdiv != 1.0 else sl
                # Rule source not reported (shouldn't happen for a loaded AR
                # model) — fall through to the raw species block below.
        # An unsupported expression has a NaN row (or none) — report WHY (the
        # specific construct / table function) rather than return NaN or a bare
        # empty-block error (GH #198 fails loudly).
        if kind == "expression":
            reason = self._expression_sens_support.get(meta["name"])
            if reason:
                raise ValueError(
                    f"output_sensitivities: expression {meta['name']!r} has no output "
                    f"sensitivity — {reason}."
                )
        block = self._sensitivity_block_for(kind, axis)
        if block.size == 0:
            extra = (
                " Expression/global-function output sensitivities require codegen (an "
                "interpreted run does not compute them); enable codegen and "
                "Simulator(..., sensitivity_params=[...])."
                if kind == "expression"
                else ""
            )
            raise ValueError(
                f"output_sensitivities: no {kind} sensitivities are available for selector "
                f"{meta['selector']!r}.{extra}"
            )
        # The species sensitivity axis is the full species list, but species
        # value columns/names are projected to the reported subset (GH #71).
        # When a projection is active the resolved index addresses the wrong
        # axis, so refuse rather than return a silently mismatched slice. The
        # observable/expression blocks are never projected, so they always pass.
        if block.shape[-2] != len(self._names_for_kind(kind)):
            raise ValueError(
                f"output_sensitivities: cannot address {kind} selector "
                f"{meta['selector']!r} by index because this result projects its "
                f"{kind} columns (GH #71); use the .sensitivities array directly."
            )
        return block[..., meta["index"], :]

    # ─── Sensitivity data (CVODES) ─────────────────────────────────

    def fisher_information(
        self,
        sigma: float | NDArray[np.float64] = 1.0,
        *,
        outputs: str | Iterable[str] | None = None,
        axis: str = "parameter",
    ) -> NDArray[np.float64]:
        r"""Compute the Fisher Information Matrix from sensitivity data.

        The FIM quantifies how much information the simulated output
        trajectories carry about each parameter. It is a pure function of the
        **model + output-sensitivity tensor** — no measurement data, no
        residuals, no objective — so it measures the *model's* practical
        identifiability (sloppiness), independent of any fit. It is computed as:

        .. math::
            \text{FIM} = \sum_t \left(\frac{\partial Y}{\partial \theta}\right)^T
            \Sigma^{-1}
            \left(\frac{\partial Y}{\partial \theta}\right)

        where :math:`\partial Y / \partial \theta` is the
        ``(n_outputs, n_params)`` sensitivity matrix at time *t*, and
        :math:`\Sigma` is the diagonal output-noise covariance.

        By default the FIM is built over **species** (back-compatible with the
        original species-only method). Pass *outputs* to build it over **named
        outputs** — any mix of ``species:``/``observable:``/``expression:``
        selectors (GH #202) — using the output-sensitivity tensor (GH
        #197/#198). For the richer eigenvalue / identifiability readout, see
        :meth:`identifiability`.

        Parameters
        ----------
        sigma : float or ndarray, optional
            Output-noise standard deviation used only to scale the FIM (it is
            **not** measurement data). Defaults to ``1.0`` (unscaled).

            - **scalar** — homogeneous: every output shares the same σ.
            - **1-D array** — per-output σ. Length must match the number of
              outputs: ``n_species`` when *outputs* is ``None``, otherwise the
              number of selectors.
        outputs : str or iterable of str, optional
            Output selectors (see :meth:`resolve_outputs`) to build the FIM
            over. ``None`` (default) uses every species — the original
            species-only behaviour.
        axis : {"parameter", "ic"}, optional
            Sensitivity axis: ``"parameter"`` (default) builds the FIM over
            :attr:`sensitivity_params`; ``"ic"`` over
            :attr:`sensitivity_ic_species`.

        Returns
        -------
        ndarray, shape ``(n, n)``
            Symmetric positive-semidefinite FIM over ``n`` parameters (or IC
            species, for ``axis="ic"``).

        Raises
        ------
        ValueError
            If the requested sensitivity axis was not computed (run with
            ``sensitivity_params`` / ``sensitivity_ic``); if *sigma*'s shape is
            wrong; if *axis* is invalid; or if this is a batch result (the FIM
            is defined per single simulation — iterate replicates).

        Notes
        -----
        The FIM is the Cramér–Rao lower bound on the inverse of the parameter
        covariance: :math:`\text{Cov}(\hat\theta) \geq \text{FIM}^{-1}`. Large
        diagonal entries indicate identifiable parameters; small entries
        indicate practical non-identifiability. ``np.linalg.cond(fim)`` (or
        :attr:`IdentifiabilityReport.condition_number`) gauges the overall
        identifiability of the parameter set.

        Examples
        --------
        >>> sim = Simulator(model, method="ode",
        ...                 sensitivity_params=["k1", "k2"])
        >>> result = sim.run(t_span=(0, 100), n_points=101)
        >>> fim = result.fisher_information(sigma=0.1)           # species
        >>> fim.shape
        (2, 2)
        >>> result.fisher_information(outputs=["observable:Atot"]).shape
        (2, 2)
        """
        block = self._fim_sensitivity_block(outputs, axis)
        return self._fim_from_block(block, sigma, outputs is None)

    def identifiability(
        self,
        sigma: float | NDArray[np.float64] = 1.0,
        *,
        outputs: str | Iterable[str] | None = None,
        axis: str = "parameter",
        rtol: float | None = None,
    ) -> IdentifiabilityReport:
        r"""Model-identifiability readout from the Fisher Information Matrix.

        Builds the FIM (see :meth:`fisher_information`) and returns its
        eigen-decomposition with practical-identifiability flags: which
        parameter combinations are well-constrained by the model and which are
        "sloppy" (near-zero eigenvalues / practically non-identifiable). This
        is sensitivity analysis of the *model* — no data, no residuals, no
        objective.

        Parameters
        ----------
        sigma : float or ndarray, optional
            Output-noise σ; see :meth:`fisher_information`. Defaults to ``1.0``.
        outputs : str or iterable of str, optional
            Output selectors to build the FIM over; ``None`` (default) uses
            every species.
        axis : {"parameter", "ic"}, optional
            Sensitivity axis; see :meth:`fisher_information`.
        rtol : float, optional
            Relative eigenvalue cutoff for flagging non-identifiable / sloppy
            directions: a direction is flagged when its eigenvalue is below
            ``rtol * lambda_max``. Defaults to ``n * eps`` (NumPy's
            numerical-rank tolerance), which flags only numerically singular
            directions; pass a larger value (e.g. ``1e-6``) to surface
            sloppy-but-nonzero directions.

        Returns
        -------
        IdentifiabilityReport
            Eigenvalues/eigenvectors, numerical rank, condition number,
            per-direction identifiability flags, and the Cramér–Rao bound
            ``FIM⁻¹`` (all-NaN + a warning when the FIM is rank-deficient).

        Examples
        --------
        >>> report = result.identifiability(sigma=0.1)
        >>> report.rank, report.condition_number
        (2, 3.4)
        >>> report.non_identifiable_directions
        []
        """
        fim = self.fisher_information(sigma, outputs=outputs, axis=axis)
        labels = list(
            self._sensitivity_params if axis == "parameter" else self._sensitivity_ic_species
        )
        return _identifiability_from_fim(fim, labels, rtol=rtol)

    def _fim_sensitivity_block(
        self,
        outputs: str | Iterable[str] | None,
        axis: str,
    ) -> NDArray[np.float64]:
        """``(n_times, n_rows, n_axis)`` sensitivity block backing the FIM.

        Species (``outputs=None``) read the raw species block directly —
        preserving the original species-only behaviour and sidestepping the
        column-projection guard (GH #71). Named outputs route through
        :meth:`output_sensitivities`, which stacks one slice per selector.
        """
        if axis not in ("parameter", "ic"):
            raise ValueError(
                f"fisher_information: axis must be 'parameter' or 'ic', got {axis!r}."
            )
        have = self.has_sensitivities if axis == "parameter" else self.has_sensitivities_ic
        if not have:
            which = "sensitivity_params=[...]" if axis == "parameter" else "sensitivity_ic=[...]"
            raise ValueError(
                "No sensitivity data available. Run the simulation "
                f"with {which} to compute sensitivities."
            )
        if self._species.ndim != 2:
            raise ValueError(
                "fisher_information is defined only for single-simulation results "
                f"(2-D arrays). Got species shape {self._species.shape}; for batch "
                "results, iterate replicates and combine the per-replicate FIMs."
            )
        if outputs is None:
            return self._sensitivities if axis == "parameter" else self._sensitivities_ic
        return self.output_sensitivities(outputs, axis=axis)

    @staticmethod
    def _fim_from_block(
        block: NDArray[np.float64],
        sigma: float | NDArray[np.float64],
        species_only: bool,
    ) -> NDArray[np.float64]:
        """FIM ``Σ_t Sᵀ diag(1/σ²) S`` from a ``(n_times, n_rows, n_axis)`` block."""
        nr = block.shape[1]
        row_label = "n_species" if species_only else "n_outputs"

        # Build the inverse-variance weight (1/σ²) per output row.
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        if sigma_arr.ndim == 0:
            if sigma_arr.item() <= 0:
                raise ValueError("sigma must be > 0.")
            inv_var = np.full(nr, 1.0 / (sigma_arr.item() ** 2))
        elif sigma_arr.ndim == 1:
            if sigma_arr.shape[0] != nr:
                raise ValueError(
                    f"sigma array length ({sigma_arr.shape[0]}) must match {row_label} ({nr})."
                )
            if np.any(sigma_arr <= 0):
                raise ValueError("All sigma values must be > 0.")
            inv_var = 1.0 / (sigma_arr**2)
        else:
            raise ValueError(f"sigma must be a scalar or 1-D array, got shape {sigma_arr.shape}.")

        # FIM = Σ_t  S_tᵀ diag(1/σ²) S_t = Σ_t (S_t·w)ᵀ S_t with w = 1/σ².
        # Weight every time point, then contract over time + rows via einsum.
        wS = block * inv_var[np.newaxis, :, np.newaxis]  # (nt, nr, n_axis)
        return np.einsum("tri,trj->ij", wS, block)  # (n_axis, n_axis)

    def gradient(
        self,
        loss_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        r"""Compute parameter gradient from sensitivity data and a loss function.

        Given forward sensitivities :math:`\partial Y / \partial p` and a
        user-supplied loss function that returns the per-species loss gradient
        :math:`\partial L / \partial Y` at each time point, computes the
        parameter gradient:

        .. math::
            \nabla_p L = \sum_t \left(\frac{\partial Y}{\partial p}\right)^T
            \frac{\partial L}{\partial Y}(t)

        This enables gradient-based optimization via
        ``scipy.optimize.minimize(method='L-BFGS-B')``.

        Parameters
        ----------
        loss_fn : callable
            A function with signature ``loss_fn(species, time) -> dL_dY``
            where:

            - ``species`` — ``(n_times, n_species)`` array of species values
            - ``time`` — ``(n_times,)`` array of time points
            - returns ``dL_dY`` — ``(n_times, n_species)`` array of
              :math:`\partial L / \partial Y_{i,t}`.

            Common example (sum-of-squares vs data):

            .. code-block:: python

                def loss_fn(species, time):
                    return 2 * (species - data)  # dL/dY for SSE

        Returns
        -------
        ndarray, shape ``(n_params,)``
            Parameter gradient :math:`\nabla_p L`.

        Raises
        ------
        ValueError
            If sensitivity data is not available.
        TypeError
            If ``loss_fn`` is not callable.

        Examples
        --------
        >>> # Sum-of-squares loss vs observed data
        >>> data = np.random.randn(101, 2)  # observed data
        >>> result = sim.compute_all_sensitivities(
        ...     t_span=(0, 10), n_points=101
        ... )
        >>> grad = result.gradient(
        ...     lambda species, time: 2 * (species - data)
        ... )
        >>> grad.shape
        (n_params,)

        >>> # Use with scipy L-BFGS-B
        >>> from scipy.optimize import minimize
        >>> def objective(p_vec):
        ...     model.set_params(dict(zip(param_names, p_vec)))
        ...     model.reset()
        ...     result = sim.compute_all_sensitivities(...)
        ...     loss = np.sum((result.species - data)**2)
        ...     grad = result.gradient(
        ...         lambda sp, t: 2 * (sp - data)
        ...     )
        ...     return loss, grad
        >>> minimize(objective, x0, method='L-BFGS-B', jac=True)

        Notes
        -----
        The gradient computation is O(n_times × n_species × n_params) —
        a single matrix multiply per time point. With parallel
        ``compute_all_sensitivities()``, the total cost of computing
        both the loss value and its gradient is ~1.2× a plain ODE solve
        (with sufficient cores).
        """
        if not self.has_sensitivities:
            raise ValueError(
                "No sensitivity data available. Run the simulation "
                "with sensitivity_params or compute_all_sensitivities() "
                "to compute sensitivities."
            )
        if not callable(loss_fn):
            raise TypeError(f"loss_fn must be callable, got {type(loss_fn).__name__}")

        sens = self._sensitivities  # (n_times, n_species, n_params)
        nt, ns, np_ = sens.shape

        # Evaluate loss gradient dL/dY at all time points
        dL_dY = np.asarray(
            loss_fn(self._species, self._time),
            dtype=np.float64,
        )
        if dL_dY.shape != (nt, ns):
            raise ValueError(
                f"loss_fn must return shape (n_times={nt}, n_species={ns}), got {dL_dY.shape}"
            )

        # ∇_p L = Σ_t (dY/dp)^T · (dL/dY)
        # Vectorized: contract over time and species via einsum.
        grad = np.einsum("tsi,ts->i", sens, dL_dY)  # (n_params,)

        return grad

    def sse_gradient(
        self,
        data: NDArray[np.float64],
        *,
        species_indices: list[int] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        r"""Compute SSE loss and parameter gradient in one call.

        Sum-of-squared-errors: :math:`L = \sum_{t,i} (Y_{t,i} - D_{t,i})^2`.

        This is the most common objective in parameter estimation. The
        analytical derivative is :math:`\partial L / \partial Y = 2(Y - D)`,
        so no user-supplied ``loss_fn`` is needed.

        Parameters
        ----------
        data : ndarray, shape ``(n_times, n_species)`` or ``(n_times, n_obs)``
            Observed data to compare against species trajectories.
            Must match the shape of ``self.species`` (or the selected
            subset if ``species_indices`` is given).
        species_indices : list[int], optional
            If given, only these species columns are used for the loss.
            Sensitivities for all parameters are still included.

        Returns
        -------
        tuple[float, ndarray]
            ``(loss, gradient)`` where *loss* is the scalar SSE and
            *gradient* is shape ``(n_params,)``.

        Raises
        ------
        ValueError
            If sensitivity data is missing or shapes don't match.

        Examples
        --------
        >>> result = sim.compute_all_sensitivities(...)
        >>> loss, grad = result.sse_gradient(observed_data)
        >>> # Use with scipy L-BFGS-B:
        >>> minimize(objective, x0, method='L-BFGS-B', jac=True)
        """
        if not self.has_sensitivities:
            raise ValueError(
                "No sensitivity data. Run with sensitivity_params or compute_all_sensitivities()."
            )
        data = np.asarray(data, dtype=np.float64)
        Y = self._species
        sens = self._sensitivities  # (nt, ns, np)

        if species_indices is not None:
            Y = Y[:, species_indices]
            sens = sens[:, species_indices, :]

        if data.shape != Y.shape:
            raise ValueError(f"data shape {data.shape} != species shape {Y.shape}")

        residual = Y - data
        loss = float(np.sum(residual**2))

        # dL/dY = 2 * (Y - data), then contract with sens
        # Vectorized: contract over time and species via einsum.
        dL_dY = 2.0 * residual  # (nt, ns_sel)
        grad = np.einsum("tsi,ts->i", sens, dL_dY)  # (n_params,)

        return loss, grad

    def chi2_gradient(
        self,
        data: NDArray[np.float64],
        sigma: float | NDArray[np.float64],
        *,
        species_indices: list[int] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        r"""Compute chi-squared loss and parameter gradient.

        :math:`\chi^2 = \sum_{t,i} \left(\frac{Y_{t,i} - D_{t,i}}{\sigma_i}\right)^2`

        Equivalent to weighted SSE with weights :math:`1/\sigma_i^2`.
        The derivative is :math:`\partial L / \partial Y_{t,i} = 2(Y - D)/\sigma_i^2`.

        Parameters
        ----------
        data : ndarray, shape ``(n_times, n_species)``
            Observed data.
        sigma : float or ndarray
            Measurement noise standard deviation.

            - **scalar** — same σ for all species.
            - **1-D array, shape ``(n_species,)``** — per-species σ.
            - **2-D array, shape ``(n_times, n_species)``** — per-point σ.

        species_indices : list[int], optional
            Subset of species columns to include.

        Returns
        -------
        tuple[float, ndarray]
            ``(chi2, gradient)`` where *chi2* is the scalar value and
            *gradient* is shape ``(n_params,)``.

        Examples
        --------
        >>> loss, grad = result.chi2_gradient(data, sigma=0.1)
        >>> loss, grad = result.chi2_gradient(data, sigma=per_species_sigma)
        """
        if not self.has_sensitivities:
            raise ValueError(
                "No sensitivity data. Run with sensitivity_params or compute_all_sensitivities()."
            )
        data = np.asarray(data, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        Y = self._species
        sens = self._sensitivities

        if species_indices is not None:
            Y = Y[:, species_indices]
            sens = sens[:, species_indices, :]

        if data.shape != Y.shape:
            raise ValueError(f"data shape {data.shape} != species shape {Y.shape}")

        # Build 1/sigma^2 array broadcastable to (nt, ns)
        inv_var: float | NDArray[np.float64]
        if sigma_arr.ndim == 0:
            inv_var = 1.0 / (sigma_arr.item() ** 2)
        elif sigma_arr.ndim == 1:
            inv_var = 1.0 / (sigma_arr**2)  # (ns,)
        elif sigma_arr.ndim == 2:
            inv_var = 1.0 / (sigma_arr**2)  # (nt, ns)
        else:
            raise ValueError(f"sigma must be scalar, 1-D, or 2-D, got shape {sigma_arr.shape}")

        residual = Y - data
        chi2 = float(np.sum(residual**2 * inv_var))

        # dL/dY = 2 * (Y - D) / sigma^2
        # Vectorized: contract over time and species via einsum.
        dL_dY = 2.0 * residual * inv_var
        grad = np.einsum("tsi,ts->i", sens, dL_dY)  # (n_params,)

        return chi2, grad

    def neg_log_likelihood_gradient(
        self,
        data: NDArray[np.float64],
        sigma: float | NDArray[np.float64],
        *,
        species_indices: list[int] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        r"""Compute negative Gaussian log-likelihood and gradient.

        :math:`-\ln \mathcal{L} = \frac{1}{2} \sum_{t,i}
        \left[\left(\frac{Y_{t,i} - D_{t,i}}{\sigma_i}\right)^2
        + \ln(2\pi\sigma_i^2)\right]`

        The constant term :math:`\ln(2\pi\sigma^2)` doesn't affect the
        gradient but is included in the loss value for correct log-likelihood.

        Parameters
        ----------
        data : ndarray, shape ``(n_times, n_species)``
            Observed data.
        sigma : float or ndarray
            Measurement noise standard deviation (scalar, per-species,
            or per-point).
        species_indices : list[int], optional
            Subset of species columns.

        Returns
        -------
        tuple[float, ndarray]
            ``(nll, gradient)`` — negative log-likelihood and gradient.

        Examples
        --------
        >>> nll, grad = result.neg_log_likelihood_gradient(data, sigma=0.1)
        """
        if not self.has_sensitivities:
            raise ValueError(
                "No sensitivity data. Run with sensitivity_params or compute_all_sensitivities()."
            )
        data = np.asarray(data, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        Y = self._species
        sens = self._sensitivities

        if species_indices is not None:
            Y = Y[:, species_indices]
            sens = sens[:, species_indices, :]

        if data.shape != Y.shape:
            raise ValueError(f"data shape {data.shape} != species shape {Y.shape}")

        # Build sigma^2 and inv_var
        var: float | NDArray[np.float64]
        inv_var: float | NDArray[np.float64]
        if sigma_arr.ndim == 0:
            var = sigma_arr.item() ** 2
            inv_var = 1.0 / var
        elif sigma_arr.ndim <= 2:
            var = sigma_arr**2
            inv_var = 1.0 / var
        else:
            raise ValueError(f"sigma must be scalar, 1-D, or 2-D, got shape {sigma_arr.shape}")

        residual = Y - data
        nt_d, ns_d = Y.shape
        # NLL = 0.5 * sum((Y-D)^2/var) + 0.5 * sum(log(2*pi*var))
        chi2_sum = np.sum(residual**2 * inv_var)
        # Constant term: need sum over all data points
        if sigma_arr.ndim == 0:
            const_sum = nt_d * ns_d * np.log(2 * np.pi * var)
        else:
            const_sum = np.sum(np.log(2 * np.pi * var))
        nll = float(0.5 * (chi2_sum + const_sum))

        # d(NLL)/dY = (Y - D) / sigma^2  (the 0.5 * 2 cancel)
        # Vectorized: contract over time and species via einsum.
        dL_dY = residual * inv_var
        grad = np.einsum("tsi,ts->i", sens, dL_dY)  # (n_params,)

        return nll, grad

    @property
    def sensitivities(self) -> NDArray[np.float64]:
        """Forward sensitivity data dY/dp.

        Shape ``(n_times, n_species, n_params)`` when sensitivities
        are available, or empty ``(0, 0, 0)`` otherwise.

        Example
        -------
        >>> sim = Simulator(model, method="ode",
        ...                 sensitivity_params=["k"])
        >>> result = sim.run(t_span=(0, 10), n_points=11)
        >>> result.sensitivities.shape
        (11, 2, 1)
        >>> result.sensitivities[-1, 0, 0]  # dA/dk at t=10
        """
        return self._sensitivities

    @property
    def has_sensitivities(self) -> bool:
        """Whether sensitivity data is available."""
        return self._sensitivities.size > 0

    @property
    def sensitivity_params(self) -> list[str]:
        """Parameter names for sensitivity analysis."""
        return self._sensitivity_params

    @property
    def sensitivities_ic(self) -> NDArray[np.float64]:
        """Forward IC sensitivity data dY/dY(0).

        Shape ``(n_times, n_species, n_ic_species)`` when IC
        sensitivities are available, or empty ``(0, 0, 0)`` otherwise.
        Column ``k`` is ``∂Y(t)/∂Y_k(0)`` where the ordering of ``k``
        matches ``sensitivity_ic_species``.
        """
        return self._sensitivities_ic

    @property
    def has_sensitivities_ic(self) -> bool:
        """Whether IC sensitivity data is available."""
        return self._sensitivities_ic.size > 0

    @property
    def sensitivity_ic_species(self) -> list[str]:
        """Species names whose ICs were differentiated against."""
        return self._sensitivity_ic_species

    # ─── Observable / expression output sensitivities (GH #196) ─────
    #
    # Storage + API only at this stage; no computation path populates these
    # yet (that lands in a later stage), so they are empty ``(0, 0, 0)`` on
    # every current run. The parameter axis reuses :attr:`sensitivity_params`
    # and the IC axis reuses :attr:`sensitivity_ic_species` — an observable or
    # expression is differentiated wrt the same parameters / initial
    # conditions as a species.

    @property
    def sensitivities_species(self) -> NDArray[np.float64]:
        """Alias for :attr:`sensitivities` (species forward sensitivities).

        Provided so the four output-sensitivity blocks have a parallel,
        self-describing name on the same object: ``sensitivities_species``,
        ``sensitivities_observables``, ``sensitivities_expressions`` (+ the
        ``_ic`` variants). Identical array to :attr:`sensitivities`.

        This is the **raw integrated state** tensor — for an SBML
        AssignmentRule-target species its slot is the frozen state's
        sensitivity (~0), not the rule's. The rule-following derivative is the
        one :meth:`output_sensitivities` returns for ``species:<name>``
        (GH #205); this attribute is the documented low-level escape hatch.
        """
        return self._sensitivities

    @property
    def sensitivities_observables(self) -> NDArray[np.float64]:
        """Observable parameter sensitivities ``d observable / dp``.

        Shape ``(n_times, n_observables, n_params)`` when computed, or empty
        ``(0, 0, 0)`` otherwise. The parameter axis matches
        :attr:`sensitivity_params`.
        """
        return self._observable_sensitivities

    @property
    def has_sensitivities_observables(self) -> bool:
        """Whether observable parameter sensitivities are available."""
        return self._observable_sensitivities.size > 0

    @property
    def sensitivities_expressions(self) -> NDArray[np.float64]:
        """Expression (function) parameter sensitivities ``d expression / dp``.

        Shape ``(n_times, n_expressions, n_params)`` when computed, or empty
        ``(0, 0, 0)`` otherwise. The parameter axis matches
        :attr:`sensitivity_params`.
        """
        return self._expression_sensitivities

    @property
    def has_sensitivities_expressions(self) -> bool:
        """Whether expression parameter sensitivities are available."""
        return self._expression_sensitivities.size > 0

    @property
    def sensitivities_observables_ic(self) -> NDArray[np.float64]:
        """Observable IC sensitivities ``d observable / dY(0)``.

        Shape ``(n_times, n_observables, n_ic_species)`` when computed, or
        empty ``(0, 0, 0)`` otherwise. The IC axis matches
        :attr:`sensitivity_ic_species`.
        """
        return self._observable_sensitivities_ic

    @property
    def has_sensitivities_observables_ic(self) -> bool:
        """Whether observable IC sensitivities are available."""
        return self._observable_sensitivities_ic.size > 0

    @property
    def sensitivities_expressions_ic(self) -> NDArray[np.float64]:
        """Expression (function) IC sensitivities ``d expression / dY(0)``.

        Shape ``(n_times, n_expressions, n_ic_species)`` when computed, or
        empty ``(0, 0, 0)`` otherwise. The IC axis matches
        :attr:`sensitivity_ic_species`.
        """
        return self._expression_sensitivities_ic

    @property
    def has_sensitivities_expressions_ic(self) -> bool:
        """Whether expression IC sensitivities are available."""
        return self._expression_sensitivities_ic.size > 0

    # ─── Solver diagnostics ─────────────────────────────────────────

    @property
    def solver_stats(self) -> dict[str, int]:
        """Solver diagnostics dict.

        Keys: ``n_steps``, ``n_rhs_evals``, ``n_jac_evals``,
        ``n_err_test_fails``, ``n_nonlin_iters``, ``n_nonlin_conv_fails``.
        """
        return self._solver_stats

    @property
    def ssa_diagnostics(self) -> dict[str, Any]:
        """SSA boundary diagnostics dict (GH #110).

        The exact SSA evaluates rate laws literally: it does not floor species
        at zero, and a reaction whose rate law evaluates negative fires in
        reverse (propensity ``|rate|``) so the SSA mean tracks the ODE. Both
        events are surfaced here instead of being silent.

        Keys:

        - ``n_negative_crossings`` (int): times a species count crossed from
          ``>= 0`` to ``< 0`` (no floor applied).
        - ``first_negative_species`` (str): name of the first species to go
          negative; ``""`` if none.
        - ``n_reverse_fires`` (int): reactions fired in reverse because their
          rate law was negative.
        - ``first_reverse_reaction`` (str): label of the first reaction
          reversed; ``""`` if none.

        All zero/empty on every non-SSA backend.
        """
        return self._ssa_diagnostics

    # ─── pandas DataFrame (optional dependency) ─────────────────────

    @property
    def dataframe(self):
        """Return a pandas DataFrame with time + observables.

        Requires pandas (``pip install bngsim[pandas]``).

        Returns
        -------
        pandas.DataFrame
            Columns: ``time``, then one per observable.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for Result.dataframe. Install with: pip install bngsim[pandas]"
            ) from None

        data = {"time": self._time}
        for i, name in enumerate(self._observable_names):
            data[name] = self._observables[:, i]
        return pd.DataFrame(data)

    # ─── xarray accessors (optional dependency) ─────────────────────

    @property
    def xr(self) -> _XarrayAccessor:
        """Per-field xarray ``DataArray`` accessor (AMICI-style).

        Each attribute access (``result.xr.species``, ``result.xr.observables``,
        etc.) builds a fresh :class:`xarray.DataArray` with labeled coords.
        Mirrors AMICI's ``rdata.xr.x`` / ``.y`` / ``.sx`` ergonomics so
        downstream code accustomed to that ecosystem can slice by name:

        >>> result.xr.species.sel(state="A").values
        >>> result.xr.observables.sel(observable="A_tot")
        >>> result.xr.sensitivities.sel(parameter="k1", state="A")

        Available fields (empty when the corresponding block has zero
        width):

        - ``species`` — dims ``(time, state)``
        - ``observables`` — dims ``(time, observable)``
        - ``expressions`` — dims ``(time, expression)``
        - ``sensitivities`` — dims ``(time, state, parameter)``
          (only when sensitivities were computed)
        - ``sensitivities_ic`` — dims ``(time, state, ic_state)``
          (only when IC sensitivities were computed)

        Requires the optional xarray dependency (``pip install xarray``).
        For a one-shot :class:`xarray.Dataset` of every field with a
        shared ``time`` coord, see :meth:`to_xarray`.

        Raises
        ------
        ImportError
            If xarray is not installed (raised on first attribute access).
        AttributeError
            If the requested field is not a known xarray-exposable
            block, or if it has zero width.
        """
        return _XarrayAccessor(self)

    def to_xarray(self):
        """Bundle every field into a single :class:`xarray.Dataset`.

        One-call complement to :attr:`xr`. The returned Dataset has a
        shared ``time`` coord and exposes ``species``, ``observables``,
        ``expressions``, and (when present) ``sensitivities`` /
        ``sensitivities_ic`` as data variables, each with the same
        labeled coords as the per-field DataArrays on :attr:`xr`.

        ``custom_attrs`` is mirrored onto ``ds.attrs``; the stochastic
        ``seed`` (when present) is stored as ``ds.attrs["seed"]``.

        Returns
        -------
        xarray.Dataset
            A Dataset with all populated trajectory fields as data vars.

        Raises
        ------
        ImportError
            If xarray is not installed.
        RuntimeError
            If the result holds 3-D batch arrays. Iterate per-replicate
            and call ``to_xarray`` on each ``Result``.

        Examples
        --------
        >>> ds = result.to_xarray()
        >>> ds["species"].sel(species="A").plot()
        >>> ds.to_netcdf("result.nc")  # archive via xarray's writer
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for Result.to_xarray() and Result.xr. "
                "Install with: pip install xarray"
            ) from None

        if self._species.ndim != 2:
            raise RuntimeError(
                "to_xarray is defined only for single-simulation results "
                f"(2-D arrays). Got species shape {self._species.shape}; "
                "for batch results, iterate replicates."
            )

        data_vars: dict[str, Any] = {}
        if self._species.size and self._species.shape[1] > 0:
            data_vars["species"] = (
                ("time", "state"),
                self._species,
            )
        if self._observables.size and self._observables.shape[1] > 0:
            data_vars["observables"] = (
                ("time", "observable"),
                self._observables,
            )
        if self._expressions.size and self._expressions.shape[1] > 0:
            data_vars["expressions"] = (
                ("time", "expression"),
                self._expressions,
            )
        if self.has_sensitivities:
            data_vars["sensitivities"] = (
                ("time", "state", "parameter"),
                self._sensitivities,
            )
        if self.has_sensitivities_ic:
            data_vars["sensitivities_ic"] = (
                ("time", "state", "ic_state"),
                self._sensitivities_ic,
            )
        # GH #196 — observable / expression output sensitivities.
        if self.has_sensitivities_observables:
            data_vars["sensitivities_observables"] = (
                ("time", "observable", "parameter"),
                self._observable_sensitivities,
            )
        if self.has_sensitivities_expressions:
            data_vars["sensitivities_expressions"] = (
                ("time", "expression", "parameter"),
                self._expression_sensitivities,
            )
        if self.has_sensitivities_observables_ic:
            data_vars["sensitivities_observables_ic"] = (
                ("time", "observable", "ic_state"),
                self._observable_sensitivities_ic,
            )
        if self.has_sensitivities_expressions_ic:
            data_vars["sensitivities_expressions_ic"] = (
                ("time", "expression", "ic_state"),
                self._expression_sensitivities_ic,
            )

        coords: dict[str, Any] = {"time": self._time}
        if self._species_names:
            coords["state"] = self._species_names
        if self._observable_names:
            coords["observable"] = self._observable_names
        if self._expression_names:
            coords["expression"] = self._expression_names
        if self._sensitivity_params:
            coords["parameter"] = self._sensitivity_params
        if self._sensitivity_ic_species:
            coords["ic_state"] = self._sensitivity_ic_species

        attrs: dict[str, Any] = dict(self.custom_attrs)
        if self._seed is not None:
            attrs["seed"] = int(self._seed)

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    # ─── RoadRunner-compatible output ───────────────────────────────

    def as_roadrunner(self, selections: list[str] | None = None) -> Any:
        """Return a :class:`bngsim.NamedArray` mimicking ``rr.simulate``.

        Drop-in shape replacement for libRoadRunner's ``simulate(...)``
        return value: a 2-D array whose columns are labeled per
        *selections*. Used to swap RR for BNGsim in PyBNF stochastic-
        fitting workflows without changing call sites.

        Parameters
        ----------
        selections : list[str], optional
            Column selectors to include, in order. Each entry is one of:

            - ``"time"`` — the time column.
            - ``"[X]"`` — the concentration of species ``X`` (BNGsim's
              stored species value, which equals ``amount/V_c``
              uniformly per Phase 2).
            - ``"X"`` — the amount of species ``X`` (concentration ×
              ``V_c``); equals the stored value when the species's
              compartment volume is 1 (the default and the .net case).

            When *None*, defaults to RoadRunner's
            ``["time"] + [f"[{s}]" for s in species_names]``.

        Returns
        -------
        :class:`bngsim.NamedArray`
            Shape ``(n_times, len(selections))``; columns ordered to
            match *selections*. The result subclasses
            :class:`numpy.ndarray`, so all numpy operations work.

        Raises
        ------
        ValueError
            If a selector is unrecognized or names a species not in
            the model.
        RuntimeError
            Only if the result was loaded from disk and the array is
            3-D batch-shaped (call :meth:`squeeze` per-row first).

        Examples
        --------
        Replace ``rr.simulate(0, 100, 101)`` (RoadRunner) with the
        BNGsim equivalent::

            sim = bngsim.Simulator(model, method="ssa")
            arr = sim.run(t_span=(0, 100), n_points=101).as_roadrunner()
            # arr has columns ["time", "[X]", "[Y]", ...]
            x_trace = arr["[X]"]

        Custom selections::

            arr = result.as_roadrunner(
                selections=["time", "[X]", "X", "[Y]"]
            )
        """
        # Lazy import to avoid a circular dep on bngsim.__init__.
        from bngsim._named_array import NamedArray

        if self._species.ndim != 2:
            raise RuntimeError(
                "as_roadrunner is defined only for single-simulation "
                "results (2-D species). Got shape "
                f"{self._species.shape}; for batch results, iterate "
                "rows or call as_roadrunner per-replicate."
            )

        if selections is None:
            selections = ["time"] + [f"[{s}]" for s in self._species_names]

        n_t = self._time.shape[0]
        sp_idx = {name: i for i, name in enumerate(self._species_names)}
        vf = self._species_volume_factors

        cols: list[NDArray[np.float64]] = []
        for sel in selections:
            if sel == "time":
                cols.append(self._time)
                continue
            if sel.startswith("[") and sel.endswith("]"):
                inner = sel[1:-1]
                if inner not in sp_idx:
                    raise ValueError(self._invalid_selection_message(sel))
                idx = sp_idx[inner]
                col = self._species[:, idx]
                cf = self._varvol_conc_factor
                if cf is not None and idx in cf:
                    # GH #131: the raw column is the conserved count amount/V_static
                    # (the stochastic convention, and the ODE convention for an
                    # hOSU=true event-resized species), so the live concentration
                    # amount/V_live = raw · (V_static/V_live) is reported by
                    # multiplying the stored per-sample factor.
                    col = col * cf[idx]
                cols.append(col)
                continue
            if sel in sp_idx:
                idx = sp_idx[sel]
                col = self._species[:, idx]
                lv = self._varvol_live_vol
                af = self._varvol_amount_factor
                if af is not None and idx in af:
                    # GH #131: this column holds amount/V_live(t) but its live
                    # volume lives in an OBSERVABLE column (an hOSU=false
                    # event-resized compartment, hidden from species output per
                    # #71), so the amount = conc · V_live(t) is recovered from the
                    # stored live-volume array rather than a species column.
                    col = col * af[idx]
                elif lv is not None and idx in lv:
                    # GH #85: this species column holds amount/V_live(t), so the
                    # amount is conc * V_live(t), not conc * V_static — recover
                    # it from the live compartment-volume column.
                    col = col * self._species[:, lv[idx]]
                elif vf is not None and idx < len(vf) and vf[idx] != 1.0:
                    col = col * vf[idx]
                cols.append(col)
                continue
            raise ValueError(self._invalid_selection_message(sel))

        data = np.empty((n_t, len(selections)), dtype=np.float64)
        for j, c in enumerate(cols):
            data[:, j] = c
        return NamedArray(data, selections)

    def _invalid_selection_message(self, sel: str) -> str:
        return (
            f"Invalid selection '{sel}'. Valid selections: "
            f"{['time'] + [f'[{s}]' for s in self._species_names] + list(self._species_names)}"
        )

    # ─── Compact summary (checkpoint indexing / logging) ────────────

    def summary(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable description of this result.

        The HPC scheduler-free contract (GH #203) pairs a serializable
        :class:`~bngsim.EvaluationSpec` (what was run) with a lightweight
        summary (what came back), so a cluster driver can index/log/checkpoint
        thousands of evaluations without re-reading every full HDF5 payload
        (use :meth:`save`/:meth:`load` for the complete arrays). The summary
        carries shapes, output names, sensitivity availability, solver
        diagnostics, and the seed — every value a built-in JSON-encodable type.

        Returns
        -------
        dict
            Keys: ``version`` (bngsim version), ``is_batch``, ``n_sims``
            (``None`` unless batch), ``n_times``, ``t_start``/``t_end``
            (``None`` for an empty time grid), ``shapes`` (per-array shape
            lists), ``species_names``/``observable_names``/``expression_names``,
            the ``has_sensitivities*`` flags, ``sensitivity_params``,
            ``sensitivity_ic_species``, ``solver_stats``, and ``seed``.

        Examples
        --------
        >>> import json
        >>> s = result.summary()
        >>> json.dumps(s)  # always succeeds
        '...'
        """
        from bngsim._version import __version__

        is_batch = self._species.ndim == 3
        t = self._time
        n_times = int(t.shape[-1]) if t.size else 0
        t_flat = t.ravel()
        t_start = float(t_flat[0]) if t_flat.size else None
        t_end = float(t_flat[-1]) if t_flat.size else None

        def _shape(arr: NDArray[np.float64]) -> list[int]:
            return [int(d) for d in arr.shape]

        return {
            "version": __version__,
            "is_batch": is_batch,
            "n_sims": int(self._species.shape[0]) if is_batch else None,
            "n_times": n_times,
            "t_start": t_start,
            "t_end": t_end,
            "shapes": {
                "species": _shape(self._species),
                "observables": _shape(self._observables),
                "expressions": _shape(self._expressions),
                "sensitivities_species": _shape(self._sensitivities),
                "sensitivities_observables": _shape(self._observable_sensitivities),
                "sensitivities_expressions": _shape(self._expression_sensitivities),
            },
            "species_names": list(self._species_names),
            "observable_names": list(self._observable_names),
            "expression_names": list(self._expression_names),
            "has_sensitivities": self.has_sensitivities,
            "has_sensitivities_observables": self.has_sensitivities_observables,
            "has_sensitivities_expressions": self.has_sensitivities_expressions,
            "has_sensitivities_ic": self.has_sensitivities_ic,
            "sensitivity_params": list(self._sensitivity_params),
            "sensitivity_ic_species": list(self._sensitivity_ic_species),
            "solver_stats": dict(self._solver_stats),
            "seed": self._seed,
        }

    # ─── HDF5 save / load ──────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save result to HDF5 file.

        Parameters
        ----------
        path : str or Path
            Output file path (e.g. ``"results.h5"``).

        Raises
        ------
        ImportError
            If h5py is not installed.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for Result.save(). Install with: pip install bngsim[hdf5]"
            ) from None

        path = Path(path)
        logger.info("Saving result to %s", path)

        from bngsim._version import __version__ as _bngsim_version

        with h5py.File(path, "w") as f:
            # Version for forward compatibility. v2 (GH #196) adds the optional
            # sensitivity blocks below; a v1 reader simply won't find them and a
            # v2 file without sensitivities is byte-equivalent to a v1 one.
            f.attrs["bngsim_version"] = _bngsim_version
            f.attrs["format_version"] = 2

            # Stochastic seed actually used (omitted for ODE results).
            if self._seed is not None:
                f.attrs["seed"] = int(self._seed)

            # Core data
            f.create_dataset("time", data=self._time)
            f.create_dataset("species", data=self._species)
            f.create_dataset("observables", data=self._observables)
            if self._expressions.size > 0:
                f.create_dataset("expressions", data=self._expressions)

            # Names (stored as variable-length strings)
            dt = h5py.string_dtype()
            f.create_dataset(
                "species_names",
                data=self._species_names,
                dtype=dt,
            )
            f.create_dataset(
                "observable_names",
                data=self._observable_names,
                dtype=dt,
            )
            if self._expression_names:
                f.create_dataset(
                    "expression_names",
                    data=self._expression_names,
                    dtype=dt,
                )

            # Sensitivity blocks (GH #196). Each is written only when populated,
            # so a result without sensitivities produces no sensitivity datasets
            # (and round-trips back to the empty (0, 0, 0) blocks on load). The
            # species blocks and the new observable/expression blocks share the
            # parameter / IC-species name axes, persisted once here.
            if self._sensitivities.size > 0:
                f.create_dataset("sensitivities", data=self._sensitivities)
            if self._sensitivities_ic.size > 0:
                f.create_dataset("sensitivities_ic", data=self._sensitivities_ic)
            if self._observable_sensitivities.size > 0:
                f.create_dataset("observable_sensitivities", data=self._observable_sensitivities)
            if self._expression_sensitivities.size > 0:
                f.create_dataset("expression_sensitivities", data=self._expression_sensitivities)
            if self._observable_sensitivities_ic.size > 0:
                f.create_dataset(
                    "observable_sensitivities_ic", data=self._observable_sensitivities_ic
                )
            if self._expression_sensitivities_ic.size > 0:
                f.create_dataset(
                    "expression_sensitivities_ic", data=self._expression_sensitivities_ic
                )
            if self._sensitivity_params:
                f.create_dataset("sensitivity_params", data=self._sensitivity_params, dtype=dt)
            if self._sensitivity_ic_species:
                f.create_dataset(
                    "sensitivity_ic_species", data=self._sensitivity_ic_species, dtype=dt
                )

            # Solver stats
            stats_grp = f.create_group("solver_stats")
            for key, val in self._solver_stats.items():
                stats_grp.attrs[key] = val

            # Custom attrs (only string/numeric values)
            if self.custom_attrs:
                ca_grp = f.create_group("custom_attrs")
                for key, val in self.custom_attrs.items():
                    try:
                        ca_grp.attrs[key] = val
                    except TypeError:
                        logger.warning(
                            "Skipping non-serializable custom_attr '%s'",
                            key,
                        )

        logger.info("Result saved (%d time points)", self.n_times)

    @classmethod
    def load(cls, path: str | Path) -> Result:
        """Load result from HDF5 file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        Result
            Loaded result.

        Raises
        ------
        ImportError
            If h5py is not installed.
        FileNotFoundError
            If the file does not exist.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for Result.load(). Install with: pip install bngsim[hdf5]"
            ) from None

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        logger.info("Loading result from %s", path)

        with h5py.File(path, "r") as f:
            time = np.array(f["time"])
            species = np.array(f["species"])
            observables = np.array(f["observables"])

            expressions = (
                np.array(f["expressions"]) if "expressions" in f else np.empty((time.shape[0], 0))
            )

            species_names = [
                s.decode() if isinstance(s, bytes) else s for s in f["species_names"][()]
            ]
            observable_names = [
                s.decode() if isinstance(s, bytes) else s for s in f["observable_names"][()]
            ]
            expression_names = []
            if "expression_names" in f:
                expression_names = [
                    s.decode() if isinstance(s, bytes) else s for s in f["expression_names"][()]
                ]

            # Solver stats
            solver_stats = {}
            if "solver_stats" in f:
                for key in f["solver_stats"].attrs:
                    solver_stats[key] = int(f["solver_stats"].attrs[key])

            # Custom attrs
            custom_attrs = {}
            if "custom_attrs" in f:
                for key in f["custom_attrs"].attrs:
                    custom_attrs[key] = f["custom_attrs"].attrs[key]
                    # Convert numpy scalars to Python types
                    if hasattr(custom_attrs[key], "item"):
                        custom_attrs[key] = custom_attrs[key].item()

            seed = int(f.attrs["seed"]) if "seed" in f.attrs else None

            # Sensitivity blocks (GH #196; format_version >= 2). Absent datasets
            # round-trip back to the empty (0, 0, 0) block via _empty_sens_block.
            def _sens(name: str) -> NDArray | None:
                return np.array(f[name]) if name in f else None

            def _names(name: str) -> list[str] | None:
                if name not in f:
                    return None
                return [s.decode() if isinstance(s, bytes) else s for s in f[name][()]]

            sensitivities = _sens("sensitivities")
            sensitivities_ic = _sens("sensitivities_ic")
            observable_sensitivities = _sens("observable_sensitivities")
            expression_sensitivities = _sens("expression_sensitivities")
            observable_sensitivities_ic = _sens("observable_sensitivities_ic")
            expression_sensitivities_ic = _sens("expression_sensitivities_ic")
            sensitivity_params = _names("sensitivity_params")
            sensitivity_ic_species = _names("sensitivity_ic_species")

        result = cls(
            core=None,
            _time=time,
            _species=species,
            _observables=observables,
            _expressions=expressions,
            _species_names=species_names,
            _observable_names=observable_names,
            _expression_names=expression_names,
            _solver_stats=solver_stats,
            _seed=seed,
            custom_attrs=custom_attrs,
            _sensitivities=sensitivities,
            _sensitivity_params=sensitivity_params,
            _sensitivities_ic=sensitivities_ic,
            _sensitivity_ic_species=sensitivity_ic_species,
            _observable_sensitivities=observable_sensitivities,
            _expression_sensitivities=expression_sensitivities,
            _observable_sensitivities_ic=observable_sensitivities_ic,
            _expression_sensitivities_ic=expression_sensitivities_ic,
        )
        logger.info(
            "Loaded result: %d time points, %d species, %d observables",
            result.n_times,
            result.n_species,
            result.n_observables,
        )
        return result

    # ─── File export ────────────────────────────────────────────────

    def to_gdat(
        self,
        path: str | Path,  # noqa: F821
        *,
        print_functions: bool = False,
        print_rate_laws: bool = False,
    ) -> None:
        """Export observables in BNG ``.gdat`` format.

        Function-column headers are always **bare** (no ``()`` suffix) and
        identical across every simulation method (issue #58).

        Parameters
        ----------
        path : str or Path
            Output file path.
        print_functions : bool
            When True, append the user-named function columns after the
            observables (auto-generated ``_rateLawN`` rate-law columns are
            still omitted). Default False keeps the observables-only output,
            matching BNG2.pl's behavior when ``print_functions=>1`` is not
            set.
        print_rate_laws : bool
            When True, additionally append the auto-generated ``_rateLawN``
            rate-law columns (bare names). These are internal rate-law
            intermediates, omitted by default. On a result loaded from HDF5
            or assembled from raw arrays they are unavailable (only the
            filtered set is persisted), so this flag is a no-op there.
            Default False.
        """
        if self._core is not None:
            self._core.to_gdat(str(path), print_functions, print_rate_laws)
            return

        # Manual export for loaded results (bare headers, no "()").
        names = list(self._observable_names)
        data = self._observables
        if (print_functions or print_rate_laws) and self._expression_names:
            keep = [
                i
                for i, n in enumerate(self._expression_names)
                if (_is_auto_rate_law(n) and print_rate_laws)
                or (not _is_auto_rate_law(n) and print_functions)
            ]
            if keep:
                names = names + [self._expression_names[i] for i in keep]
                data = np.concatenate([data, self._expressions[:, keep]], axis=1)
        _write_bng_file(path, self._time, data, names)

    def to_cdat(self, path: str | Path) -> None:  # noqa: F821
        """Export species in BNG ``.cdat`` format.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        if self._core is not None:
            self._core.to_cdat(str(path))
        else:
            _write_bng_file(
                path,
                self._time,
                self._species,
                self._species_names,
            )

    def to_csv(
        self,
        path: str | Path,
        *,
        kind: str = "observables",
        sep: str = ",",
        include_time: bool = True,
        header: bool = True,
    ) -> None:
        """Export the trajectory as a plain delimited text file.

        Unlike :meth:`to_gdat` / :meth:`to_cdat` (BNG-native, ``#``-prefixed,
        fixed-width space-padded), this writer produces a format SBML/
        RoadRunner/Tellurium users expect: a header row of column names
        followed by data rows, with a user-chosen separator (``,`` for
        CSV by default; ``"\\t"`` for TSV; any single character).

        The first column is ``time`` (unless ``include_time=False``); the
        remaining columns are the observable or species columns named
        exactly as they appear on the in-memory :class:`Result`. No
        ``#`` comment prefix is emitted, so the file can be loaded
        directly by ``pandas.read_csv`` or ``numpy.loadtxt``.

        Parameters
        ----------
        path : str or Path
            Output file path. The caller chooses the path; nothing is
            written to the current working directory implicitly.
        kind : {"observables", "species"}
            Which trajectory block to write. ``"observables"`` (default)
            matches what ``.gdat`` would contain; ``"species"`` matches
            ``.cdat``. ``Result.expressions`` is not currently exported
            via ``to_csv`` — use :meth:`Result.save` (HDF5) for a
            lossless capture.
        sep : str
            Column separator. Default ``","``. Use ``"\\t"`` for TSV.
            Must be a single character.
        include_time : bool
            Whether to prepend a ``time`` column. Default ``True``.
        header : bool
            Whether to write the header row of column names. Default
            ``True``.

        Raises
        ------
        ValueError
            If ``kind`` is not ``"observables"`` or ``"species"``, if ``sep``
            is not a single character, or if the trajectory is not a 2-D
            single-sim array (call this per-replicate on batch results).
        OSError
            If the file cannot be opened for writing.

        Notes
        -----
        Backend coverage: works on every method
        (ODE / SSA / PSA / NFsim / RuleMonkey) and on results loaded
        from HDF5. What is **not** exported by this writer:
        ``expressions``, ``sensitivities``, ``solver_stats``,
        ``custom_attrs``, ``seed``. Use HDF5 (:meth:`Result.save`) when
        any of those need to round-trip.

        Examples
        --------
        Plain CSV of observables, ready for ``pandas.read_csv``::

            sim = bngsim.Simulator(model, method="ode")
            result = sim.run(t_span=(0, 100), n_points=101)
            result.to_csv("out.csv")

        Tab-separated species amounts::

            result.to_csv("out.tsv", kind="species", sep="\\t")

        Headerless data block (e.g. to append to an existing file)::

            result.to_csv("data.csv", header=False)
        """
        if kind not in ("observables", "species"):
            raise ValueError(f"kind must be 'observables' or 'species', got {kind!r}")
        if len(sep) != 1:
            raise ValueError(f"sep must be a single character, got {sep!r}")

        if kind == "observables":
            data = self._observables
            names = self._observable_names
        else:
            data = self._species
            names = self._species_names

        if data.ndim != 2:
            raise ValueError(
                f"to_csv requires a 2-D single-sim result; got "
                f"{kind} with shape {data.shape}. For batch results "
                "(3-D), call to_csv on each replicate Result."
            )

        path = Path(path)
        n_t = self._time.shape[0]
        n_cols = data.shape[1]

        with open(path, "w") as f:
            if header:
                cols = (["time"] if include_time else []) + list(names)
                f.write(sep.join(cols) + "\n")
            for i in range(n_t):
                row: list[str] = []
                if include_time:
                    row.append(f"{self._time[i]:.12e}")
                for j in range(n_cols):
                    row.append(f"{data[i, j]:.12e}")
                f.write(sep.join(row) + "\n")

    # ─── Squeeze / stack helpers (for batch) ────────────────────────

    @staticmethod
    def squeeze(results: list[Result]) -> Result:
        """Stack a list of single-sim Results into one batch Result.

        The returned Result has 3D arrays:
        - ``species.shape == (n_sims, n_times, n_species)``
        - ``observables.shape == (n_sims, n_times, n_observables)``

        Any populated sensitivity block (species, observable, expression, and
        their IC variants — GH #196) is carried through with the sim axis
        prepended, e.g. ``sensitivities.shape == (n_sims, n_times, n_rows,
        n_cols)``; a block that is empty on the inputs stays empty. The
        parameter / IC-species name lists are taken from the first result.

        Parameters
        ----------
        results : list[Result]
            Results from individual simulations (all must have the
            same n_times, n_species, n_observables).

        Returns
        -------
        Result
            A batch result with 3D arrays.
        """
        if not results:
            raise ValueError("Cannot squeeze empty result list")
        if len(results) == 1:
            return results[0]

        time = results[0]._time
        species_names = results[0]._species_names
        observable_names = results[0]._observable_names
        expression_names = results[0]._expression_names

        species_3d = np.stack([r._species for r in results], axis=0)
        obs_3d = np.stack([r._observables for r in results], axis=0)

        # Expressions may or may not exist
        if results[0]._expressions.size > 0:
            expr_3d = np.stack([r._expressions for r in results], axis=0)
        else:
            expr_3d = np.empty((len(results), time.shape[0], 0))

        # Aggregate solver stats. Counters sum across the batch; categorical
        # fields keep their value only when every input agrees.
        agg_stats = {}
        for key in results[0]._solver_stats:
            values = [r._solver_stats.get(key, 0) for r in results]
            if key == "linear_solver":
                agg_stats[key] = values[0] if all(v == values[0] for v in values) else -1
            else:
                agg_stats[key] = sum(values)

        # If every input used the same seed, surface it; otherwise fall
        # back to None — the per-sim seeds remain on the unsqueezed
        # Result objects.
        seeds = {r._seed for r in results}
        squeeze_seed = next(iter(seeds)) if len(seeds) == 1 else None

        # GH #196 — carry every populated sensitivity block through, stacking on
        # a new leading sim axis. Empty-on-input blocks stay empty (0, 0, 0).
        def stack_sens(attr: str) -> NDArray[np.float64]:
            if getattr(results[0], attr).size == 0:
                return np.empty((0, 0, 0))
            return np.stack([getattr(r, attr) for r in results], axis=0)

        return Result(
            core=None,
            _time=time,
            _species=species_3d,
            _observables=obs_3d,
            _expressions=expr_3d,
            _species_names=species_names,
            _observable_names=observable_names,
            _expression_names=expression_names,
            _solver_stats=agg_stats,
            _species_volume_factors=results[0]._species_volume_factors,
            _seed=squeeze_seed,
            _sensitivities=stack_sens("_sensitivities"),
            _sensitivity_params=results[0]._sensitivity_params,
            _sensitivities_ic=stack_sens("_sensitivities_ic"),
            _sensitivity_ic_species=results[0]._sensitivity_ic_species,
            _observable_sensitivities=stack_sens("_observable_sensitivities"),
            _expression_sensitivities=stack_sens("_expression_sensitivities"),
            _observable_sensitivities_ic=stack_sens("_observable_sensitivities_ic"),
            _expression_sensitivities_ic=stack_sens("_expression_sensitivities_ic"),
        )

    # ─── Dunder methods ─────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Result(n_times={self.n_times}, "
            f"n_species={self.n_species}, "
            f"n_observables={self.n_observables})"
        )


class _ObservableAccessor:
    """Wrapper allowing both array access and named column access.

    ``result.observables`` returns the full array.
    ``result.observables["name"]`` returns a single column.
    ``result.observables[0]`` returns a single time-point row.
    """

    __slots__ = ("_data", "_names", "_name_to_idx")

    def __init__(self, data: NDArray[np.float64], names: list[str]) -> None:
        self._data = data
        self._names = names
        self._name_to_idx = {n: i for i, n in enumerate(names)}

    def __getitem__(self, key: str | int | slice) -> NDArray[np.float64]:
        if isinstance(key, str):
            if key not in self._name_to_idx:
                raise KeyError(f"Observable '{key}' not found. Available: {self._names}")
            return self._data[:, self._name_to_idx[key]]
        return self._data[key]

    def __array__(self, dtype=None, copy=None) -> NDArray[np.float64]:
        """Support ``np.asarray(result.observables)``.

        Implements the NumPy 2.x ``__array__`` protocol with proper
        ``copy`` semantics.
        """
        target_dtype = np.dtype(dtype) if dtype is not None else self._data.dtype
        needs_cast = target_dtype != self._data.dtype

        if copy is True:
            return np.array(self._data, dtype=target_dtype, copy=True)
        if copy is False:
            if needs_cast:
                raise ValueError(
                    f"Unable to avoid copy while creating an array "
                    f"with dtype {target_dtype} from array with "
                    f"dtype {self._data.dtype}."
                )
            return self._data
        # copy=None (default): copy only if needed
        if needs_cast:
            return np.array(self._data, dtype=target_dtype)
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ObservableAccessor(shape={self.shape}, names={self._names})"


class _XarrayAccessor:
    """Per-field xarray ``DataArray`` accessor.

    Built lazily by :attr:`Result.xr`. Each attribute returns a fresh
    ``DataArray`` with labeled coords. xarray is imported on first
    access so the import cost is paid only when actually used.
    """

    __slots__ = ("_result",)

    _FIELDS = (
        "species",
        "observables",
        "expressions",
        "sensitivities",
        "sensitivities_ic",
        "sensitivities_observables",
        "sensitivities_expressions",
        "sensitivities_observables_ic",
        "sensitivities_expressions_ic",
    )

    def __init__(self, result: Result) -> None:
        self._result = result

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._FIELDS:
            raise AttributeError(
                f"Result.xr has no field {name!r}. Available: {', '.join(self._FIELDS)}"
            )
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for Result.xr. Install with: pip install xarray"
            ) from None

        result = self._result
        if name == "species":
            data = result._species
            if data.ndim != 2 or data.shape[1] == 0:
                raise AttributeError("Result has no species data")
            return xr.DataArray(
                data,
                dims=("time", "state"),
                coords={
                    "time": result._time,
                    "state": result._species_names,
                },
                name="species",
            )
        if name == "observables":
            data = result._observables
            if data.ndim != 2 or data.shape[1] == 0:
                raise AttributeError("Result has no observables")
            return xr.DataArray(
                data,
                dims=("time", "observable"),
                coords={
                    "time": result._time,
                    "observable": result._observable_names,
                },
                name="observables",
            )
        if name == "expressions":
            data = result._expressions
            if data.ndim != 2 or data.shape[1] == 0:
                raise AttributeError("Result has no expressions")
            return xr.DataArray(
                data,
                dims=("time", "expression"),
                coords={
                    "time": result._time,
                    "expression": result._expression_names,
                },
                name="expressions",
            )
        if name == "sensitivities":
            if not result.has_sensitivities:
                raise AttributeError(
                    "Result has no sensitivities. Run with sensitivity_params=[...]."
                )
            return xr.DataArray(
                result._sensitivities,
                dims=("time", "state", "parameter"),
                coords={
                    "time": result._time,
                    "state": result._species_names,
                    "parameter": result._sensitivity_params,
                },
                name="sensitivities",
            )
        if name == "sensitivities_ic":
            if not result.has_sensitivities_ic:
                raise AttributeError(
                    "Result has no IC sensitivities. Run with sensitivity_ic=[...]."
                )
            return xr.DataArray(
                result._sensitivities_ic,
                dims=("time", "state", "ic_state"),
                coords={
                    "time": result._time,
                    "state": result._species_names,
                    "ic_state": result._sensitivity_ic_species,
                },
                name="sensitivities_ic",
            )
        # GH #196 — observable / expression output sensitivities. The parameter
        # axis reuses the species ``parameter`` coord; the IC axis reuses
        # ``ic_state``.
        if name == "sensitivities_observables":
            if not result.has_sensitivities_observables:
                raise AttributeError("Result has no observable sensitivities.")
            return xr.DataArray(
                result._observable_sensitivities,
                dims=("time", "observable", "parameter"),
                coords={
                    "time": result._time,
                    "observable": result._observable_names,
                    "parameter": result._sensitivity_params,
                },
                name="sensitivities_observables",
            )
        if name == "sensitivities_expressions":
            if not result.has_sensitivities_expressions:
                raise AttributeError("Result has no expression sensitivities.")
            return xr.DataArray(
                result._expression_sensitivities,
                dims=("time", "expression", "parameter"),
                coords={
                    "time": result._time,
                    "expression": result._expression_names,
                    "parameter": result._sensitivity_params,
                },
                name="sensitivities_expressions",
            )
        if name == "sensitivities_observables_ic":
            if not result.has_sensitivities_observables_ic:
                raise AttributeError("Result has no observable IC sensitivities.")
            return xr.DataArray(
                result._observable_sensitivities_ic,
                dims=("time", "observable", "ic_state"),
                coords={
                    "time": result._time,
                    "observable": result._observable_names,
                    "ic_state": result._sensitivity_ic_species,
                },
                name="sensitivities_observables_ic",
            )
        if name == "sensitivities_expressions_ic":
            if not result.has_sensitivities_expressions_ic:
                raise AttributeError("Result has no expression IC sensitivities.")
            return xr.DataArray(
                result._expression_sensitivities_ic,
                dims=("time", "expression", "ic_state"),
                coords={
                    "time": result._time,
                    "expression": result._expression_names,
                    "ic_state": result._sensitivity_ic_species,
                },
                name="sensitivities_expressions_ic",
            )
        raise AttributeError(name)

    def __dir__(self) -> list[str]:
        return list(self._FIELDS)

    def __repr__(self) -> str:
        return f"<Result.xr fields={list(self._FIELDS)}>"


def _empty_sens_block(arr: NDArray | None) -> NDArray[np.float64]:
    """A sensitivity block, defaulting an absent one to the empty ``(0, 0, 0)``.

    Centralizes the "no data" sentinel so every sensitivity block (species,
    observable, expression, and their IC variants — GH #196) shares one shape
    convention: a 3-D array that is empty (``size == 0``) when not computed.
    """
    return arr if arr is not None else np.empty((0, 0, 0))


def _core_sens_block(core: Any, attr: str) -> NDArray[np.float64]:
    """Read a 3-D sensitivity block off a C++ core, tolerating older cores.

    The pybind accessors (GH #196) always return a 3-D array — empty
    ``(0, 0, 0)`` when the block was never populated — so this is a thin
    ``getattr`` with a ``hasattr`` guard for forward/backward compatibility
    with a core built before the block existed.
    """
    if hasattr(core, attr):
        return getattr(core, attr)
    return np.empty((0, 0, 0))


def _is_auto_rate_law(name: str) -> bool:
    """True for BNG2.pl's auto-generated ``_rateLawN`` function names.

    These internal rate-law functions are filtered out of ``.gdat``/``.scan``
    output to match BNG2.pl, which never writes them to its result files.
    """
    return name.startswith("_rateLaw") and name[len("_rateLaw") :].isdigit()


# Output-selector kinds and the prefixes that map onto them (GH #195). The
# canonical prefixes are the kinds themselves; the rest are accepted aliases.
_SELECTOR_KINDS = ("species", "observable", "expression")
_SELECTOR_PREFIX_ALIASES = {
    "species": "species",
    "state": "species",
    "observable": "observable",
    "expression": "expression",
    "function": "expression",
}


def _as_selector_list(selectors: str | Iterable[str]) -> list[str]:
    """Normalize a selector argument to a list of strings.

    A bare string is wrapped as a one-element list; any other iterable is
    materialized in order. Validation of the individual entries (type,
    emptiness) is deferred to :meth:`Result._resolve_one_output`.
    """
    if isinstance(selectors, str):
        return [selectors]
    return list(selectors)


def _output_meta(kind: str, name: str, index: int) -> dict[str, Any]:
    """Build the metadata dict for one resolved output column (GH #195).

    ``column_label`` is the ``.gdat``/``.cdat`` header label; bngsim emits
    bare headers (issue #58), so it is identical to ``name``.
    """
    return {
        "selector": f"{kind}:{name}",
        "kind": kind,
        "name": name,
        "index": index,
        "column_label": name,
    }


def _identifiability_from_fim(
    fim: NDArray[np.float64],
    parameters: list[str],
    *,
    rtol: float | None = None,
) -> IdentifiabilityReport:
    """Eigen-readout + Cramér–Rao bound of a FIM (GH #202).

    The FIM is symmetric positive-semidefinite, so :func:`numpy.linalg.eigh`
    gives real, ascending eigenvalues and orthonormal eigenvectors. Directions
    with an eigenvalue at or below ``rtol * lambda_max`` are flagged
    practically non-identifiable ("sloppy"); when any exist the FIM is
    rank-deficient and its inverse (the Cramér–Rao bound) is undefined — we
    return all-NaN and warn rather than emit a garbage inverse.
    """
    n = fim.shape[0]
    eigvals, eigvecs = np.linalg.eigh(fim)  # ascending; columns are eigenvectors
    # The FIM is PSD; clip round-off negatives so reported eigenvalues stay ≥ 0.
    eigvals = np.clip(eigvals, 0.0, None)

    lam_max = float(eigvals[-1]) if eigvals.size else 0.0
    if rtol is None:
        # NumPy's matrix_rank tolerance (singular values == |eigenvalues| here).
        rtol = n * float(np.finfo(np.float64).eps)
    threshold = rtol * lam_max

    identifiable = eigvals > threshold
    rank = int(np.count_nonzero(identifiable))
    non_identifiable = [i for i in range(n) if not identifiable[i]]

    lam_min = float(eigvals[0]) if eigvals.size else 0.0
    condition_number = lam_max / lam_min if (rank == n and lam_min > 0.0) else float("inf")

    cramer_rao_bound: NDArray[np.float64]
    if rank < n:
        warnings.warn(
            f"Fisher information matrix is rank-deficient (rank {rank} < {n}): "
            f"{n - rank} practically non-identifiable / sloppy direction(s). The "
            "Cramér–Rao bound (FIM⁻¹) is undefined and returned as NaN.",
            RuntimeWarning,
            stacklevel=3,
        )
        cramer_rao_bound = np.full((n, n), np.nan, dtype=np.float64)
    else:
        cramer_rao_bound = np.linalg.inv(fim).astype(np.float64, copy=False)

    return IdentifiabilityReport(
        fim=fim,
        parameters=parameters,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        rank=rank,
        condition_number=condition_number,
        identifiable=identifiable,
        non_identifiable_directions=non_identifiable,
        cramer_rao_bound=cramer_rao_bound,
        threshold=threshold,
    )


def _write_bng_file(
    path: str | Path,
    time: NDArray,
    data: NDArray,
    names: list[str],
) -> None:
    """Write BNG-format (.gdat/.cdat) file from arrays."""
    path = Path(path)
    with open(path, "w") as f:
        # Header
        header = "# " + " ".join(["time"] + names)
        f.write(header + "\n")
        # Data rows
        for i in range(len(time)):
            row = [f"{time[i]:20.12e}"]
            for j in range(data.shape[1]):
                row.append(f"{data[i, j]:20.12e}")
            f.write(" ".join(row) + "\n")
