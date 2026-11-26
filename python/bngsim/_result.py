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

import contextlib
import hashlib
import json
import logging
import threading
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, overload

import numpy as np
from numpy.typing import NDArray

from bngsim import _foreign_names, _notices, _record
from bngsim._error_codes import pass_argument
from bngsim._exceptions import (
    ArgumentTypeError,
    MissingFileError,
    NotFoundError,
    NotSupportedError,
    UsageError,
)
from bngsim._extension_handle import raise_cannot_pickle, shallow_copy
from bngsim._notices import Notice

if TYPE_CHECKING:
    from bngsim._bngsim_core import ResultCore

logger = logging.getLogger("bngsim")


def _empty_ssa_diagnostics() -> dict[str, Any]:
    """SSA diagnostics dict for a result built from raw arrays: all zero/empty."""
    return {
        "n_negative_crossings": 0,
        "first_negative_species": "",
        "n_reverse_fires": 0,
        "first_reverse_reaction": "",
        "n_rounded_populations": 0,
        "first_rounded_species": "",
        # Nothing ran to say which backend evaluated the propensities;
        # "interpreted" would name one that was never used.
        "propensity_backend": "unknown",
    }


def _empty_psa_diagnostics() -> dict[str, Any]:
    """PSA diagnostics dict for a non-PSA (or non-SSA) run: inactive, all empty."""
    return {
        "active": False,
        "reaction_index": [],
        "time": 0.0,
        "mbar": [],
        "qexc": [],
        "mbar_integral": [],
        "qexc_integral": [],
        "exact_event_integral": 0.0,
        "scaled_event_integral": 0.0,
        "speedup": float("nan"),
        "peak_population": 0.0,
        "activation_crossed": False,
    }


def _psa_diagnostics_from_core(ssa_diag: Any) -> dict[str, Any]:
    """Build the PSA diagnostics dict from the C++ SsaDiagnostics struct (GH #15).

    Returns the empty/inactive dict when the run did not use partial scaling, so
    every non-PSA backend reports a stable, zero-valued shape.
    """
    if not getattr(ssa_diag, "psa_active", False):
        return _empty_psa_diagnostics()

    t = float(ssa_diag.psa_time)
    mbar_int = [float(x) for x in ssa_diag.psa_mbar_integral]
    qexc_int = [float(x) for x in ssa_diag.psa_qexc_integral]
    # Dwell-time averages m̄_r = (1/T)∫m_r dt and q̄_r = (1/T)∫(m_r−1)a_r dt. The
    # raw integrals are kept alongside so a covariance correction can form either
    # the stationary source (÷T) or a transient one.
    if t > 0.0:
        mbar = [v / t for v in mbar_int]
        qexc = [v / t for v in qexc_int]
    else:
        mbar = [1.0 for _ in mbar_int]
        qexc = [0.0 for _ in qexc_int]

    exact = float(ssa_diag.psa_exact_event_integral)
    scaled = float(ssa_diag.psa_scaled_event_integral)
    # ŝ = ∫Σa_r dt / ∫Σ(a_r/m_r) dt — a per-path event-rate ratio (would-be exact
    # events ÷ actual scaled events), computable from the scaled run alone. This
    # is NOT a wall-clock speedup: it ignores per-event overhead and the variance
    # of any excess correction (see docs/notes.tex §sampling-cost).
    speedup = exact / scaled if scaled > 0.0 else float("nan")

    return {
        "active": True,
        "reaction_index": [int(i) for i in ssa_diag.psa_reaction_index],
        "time": t,
        "mbar": mbar,
        "qexc": qexc,
        "mbar_integral": mbar_int,
        "qexc_integral": qexc_int,
        "exact_event_integral": exact,
        "scaled_event_integral": scaled,
        "speedup": speedup,
        "peak_population": float(ssa_diag.psa_peak_population),
        "activation_crossed": bool(ssa_diag.psa_activation_crossed),
    }


def _one_sided_from_core(core: Any, columns: list[str]) -> list[tuple[float, dict[str, str]]]:
    """The core's one-sided switch records, by parameter name (issue #1822)."""
    out: list[tuple[float, dict[str, str]]] = []
    for t, cols, directions in getattr(core, "one_sided_switches", ()):
        named = {
            columns[c]: "increasing" if d > 0 else "decreasing"
            for c, d in zip(cols, directions, strict=True)
            if 0 <= c < len(columns)
        }
        if named:
            out.append((float(t), named))
    return out


_HELD = threading.local()


@contextlib.contextmanager
def _one_sided_warnings_held() -> Iterator[None]:
    """Record one-sided entries on the results made in this thread, and warn of
    none of them: for runs made in worker threads, where no frame of the
    caller's is on the stack for a warning to name. Whoever holds them warns
    once the results are back on the calling thread (issue #1822). The steps
    a run took where t + h == t are held the same way (issue #1837)."""
    was = getattr(_HELD, "on", False)
    _HELD.on = True
    try:
        yield
    finally:
        _HELD.on = was


def _warnings_held() -> bool:
    """Whether this thread is in :func:`_one_sided_warnings_held`."""
    return bool(getattr(_HELD, "on", False))


def _merge_one_sided(results: Iterable[Any]) -> list[tuple[float, dict[str, str]]]:
    """The one-sided entries of several results of one run, by instant."""
    by_t: dict[float, dict[str, str]] = {}
    for r in results:
        for t, named in getattr(r, "_one_sided_sensitivities", ()):
            by_t.setdefault(t, {}).update(named)
    return sorted(by_t.items())


def _warn_one_sided(entries: list[tuple[float, dict[str, str]]], *, note: bool = True) -> None:
    """Warn of each one-sided entry, and record it with the run (issue #1703)
    unless *note* is False: a batch row held its warning and recorded it on
    its own thread, and the caller's thread gives the warning alone."""
    if not entries:
        return
    held = getattr(_HELD, "on", False)
    from bngsim._exceptions import OneSidedSensitivityWarning

    for t, named in entries:
        which = ", ".join(f"'{p}' {direction}" for p, direction in named.items())
        _notices.warn(
            f"Forward sensitivity: the switch time at t={t:.6g} shares its instant with "
            "another that no requested parameter moves alone, and a rate law composes the "
            "two, so the run sits on a kink in the parameters that move this switch apart "
            "from the other. Their columns are one-sided derivatives, each for a step in the "
            f"direction that delays this switch past the other: {which}. With sigma = t0 + "
            "t_delta on a bound t_delta >= 0 that is the feasible side; bngsim cannot see "
            "the bound, so check the direction against it. In the other direction the "
            "derivative is another number, which bngsim does not compute. "
            "Every other column is an ordinary derivative. "
            "Result.one_sided_sensitivities lists these.",
            OneSidedSensitivityWarning,
            code="sens.one_sided",
            stacklevel=_notices.OUTSIDE,
            emit=not held,
            record=note,
        )


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


def _state_digest(state: NDArray[np.float64]) -> str:
    """The SHA-256, in hex, of a state vector's values as little-endian float64
    bytes, with ``-0.0`` read as ``0.0`` and every NaN as one NaN, so that two
    vectors equal value for value (``np.array_equal(equal_nan=True)``) have one
    digest (issue #1701)."""
    canon = np.ascontiguousarray(state, dtype="<f8") + 0.0  # -0.0 + 0.0 is 0.0
    if not np.isfinite(canon).all():
        canon[np.isnan(canon)] = np.nan
    return hashlib.sha256(canon.tobytes()).hexdigest()


class Result:
    """Simulation result container.

    Attributes
    ----------
    time : ndarray, shape (n_times,)
        Time points.
    species : ndarray, shape (n_times, n_species)
        Species concentrations at each time point. ``result.species["A"]``
        is species ``A``'s column.
    observables : ndarray, shape (n_times, n_observables)
        Observable values at each time point. ``result.observables["Atot"]``
        is observable ``Atot``'s column.
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

    Notes
    -----
    A ``Result`` holds a handle into bngsim's compiled extension, so it cannot
    be pickled or deep-copied; :meth:`save` and :meth:`Result.load`, or one of
    the conversions (:meth:`as_roadrunner`, :meth:`to_xarray`), carry it across
    a process boundary instead. ``copy.copy(result)`` does work, and the copy
    shares that handle *and the result arrays* with the original. That is left
    working, where the same sharing makes ``copy.copy`` refuse on a
    :class:`Model` or a :class:`Simulator` (issue #643), because a ``Result``
    has no setter and no mutating method: no call you can make on the copy
    writes into the original. Writing into its arrays in place does —
    ``copy.copy(result).species[0] = ...`` changes ``result`` — but so does
    ``result.species[0] = ...``, because every access returns the same live
    view rather than a fresh array. Take ``.copy()`` of a block you mean to
    modify, whichever name you reach it through.
    """

    __slots__ = (
        "_stop_condition",
        "_started_from",
        "_start_state",
        "_start_state_sha256",
        # Issue #1703: what the record says of the run, the model and the
        # notices (bngsim._record builds Result.record from them).
        "_run_info",
        "_run_params",
        "_run_atol",
        "_model_info",
        "_record_frozen",
        "_record_differs",
        "_source_text",
        "_notices",
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
        "_reaction_firing_counts",
        "_reaction_propensity_integrals",
        "_reaction_labels",
        "_expression_sens_support",
        "_species_names",
        "_observable_names",
        "_expression_names",
        "_sensitivity_params",
        "_sensitivity_ic_species",
        "_one_sided_sensitivities",
        "_solver_stats",
        "_ssa_diagnostics",
        "_psa_diagnostics",
        "_psa_diagnostics_runs",
        "_species_volume_factors",
        "_varvol_live_vol",
        "_varvol_conc_factor",
        "_varvol_amount_factor",
        "_volume_record_missing",
        "_ar_sens_map",
        "_ar_sens_blocked",
        "_ar_sens_refused",
        "_seed",
        "_ic_sensitivity_seed",
        "_core",
        # Issue #1700: [the _species array and the _species_names list it was made
        # over, the _SpeciesArray view Result.species returns], so every access
        # returns the same view. A list, filled in place, so a copy.copy() shares
        # it, and the copy and the original return one view whichever asks first.
        "_species_view",
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
        _species_volume_factors: list[float] | list[list[float]] | None = None,
        _seed: int | None = None,
        _ic_sensitivity_seed: dict[str, dict[str, float]] | None = None,
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
        # GH #616 — SSA per-reaction event statistics (load / batch). Absent ⇒
        # the empty (0, 0) block and no labels.
        _reaction_firing_counts: NDArray | None = None,
        _reaction_propensity_integrals: NDArray | None = None,
        _reaction_labels: list[str] | None = None,
        # SSA diagnostics for a result built from raw arrays. ``squeeze`` passes
        # the aggregate over its inputs and ``load`` the saved dict. Absent (a
        # hand-built result, or a file that predates saving them) ⇒ the inert
        # zero dict, whose backend reads "unknown" because nothing ran to say.
        _ssa_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self._core = core
        self._species_view: list[Any] = []

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
            self._one_sided_sensitivities = _one_sided_from_core(core, self._sensitivity_params)
            _warn_one_sided(self._one_sided_sensitivities)

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

            # GH #616 — SSA per-reaction firing counts and propensity integrals,
            # recorded only when the run asked for them
            # (Simulator(..., reaction_stats=True)); (0, 0) otherwise and on every
            # other backend.
            if getattr(core, "n_reaction_stats", 0) > 0:
                self._reaction_firing_counts: NDArray[np.float64] = core.reaction_firing_counts
                self._reaction_propensity_integrals: NDArray[np.float64] = (
                    core.reaction_propensity_integrals
                )
                self._reaction_labels: list[str] = list(core.reaction_labels)
            else:
                self._reaction_firing_counts = np.empty((0, 0))
                self._reaction_propensity_integrals = np.empty((0, 0))
                self._reaction_labels = []

            # Solver stats. ``steady_state_reached`` is the int 0/1 marker
            # for whether the BNG-style early-stop fired (False on every
            # backend except CVODE with steady_state=True).
            stats = core.solver_stats
            self._solver_stats: dict[str, int] = {
                "n_steps": stats.n_steps,
                "n_rhs_evals": stats.n_rhs_evals,
                "n_jac_evals": stats.n_jac_evals,
                # RHS evaluations spent forming difference-quotient Jacobians,
                # which n_rhs_evals does not count (issue #1333).
                "n_jac_rhs_evals": stats.n_jac_rhs_evals,
                # Jacobians the solver asked for; n_jac_evals counts the
                # setups, which may reuse one (issue #1333).
                "n_jac_formations": stats.n_jac_formations,
                "n_err_test_fails": stats.n_err_test_fails,
                "n_nonlin_iters": stats.n_nonlin_iters,
                "n_nonlin_conv_fails": stats.n_nonlin_conv_fails,
                # The SENSITIVITY solve's own counters (GH #384). Separate in
                # CVODES, and separate here: the two above count the state solve,
                # so a run can reject steps on its sensitivity error test and
                # still report `n_err_test_fails == 0`. 0 with sensitivities off.
                "n_sens_err_test_fails": stats.n_sens_err_test_fails,
                "n_sens_nonlin_conv_fails": stats.n_sens_nonlin_conv_fails,
                # Steps CVODE took where t + h == t, which moved the state and
                # not the clock (issue #1837); a run that took one warns
                # RoundoffStepWarning. 0 on every other backend.
                "n_roundoff_steps": stats.n_roundoff_steps,
                # Which direct linear solver the ODE engine actually used, as the
                # ``LinearSolverKind`` enum (include/bngsim/result.hpp). Pinned
                # mapping — DO NOT reinterpret as a SUNDIALS solver-id:
                #   0 = built-in dense LU   (SUNDIALS' GETRF, bngsim's solver)
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
                # Issue #718 — fractional molecule counts rounded at run start.
                "n_rounded_populations": ssa_diag.n_rounded_populations,
                "first_rounded_species": ssa_diag.first_rounded_species,
                # GH #190 — how propensities were evaluated: "cc"/"mir"
                # (compiled recompute-all) or "interpreted".
                "propensity_backend": getattr(ssa_diag, "propensity_backend", "interpreted"),
            }
            # GH #15 — PSA partial-scaling diagnostics (inactive dict on non-PSA).
            self._psa_diagnostics: dict[str, Any] = _psa_diagnostics_from_core(ssa_diag)
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
            # Known only to the run that read the kink (issue #1822).
            self._one_sided_sensitivities = []
            self._sensitivities_ic = _empty_sens_block(_sensitivities_ic)
            self._sensitivity_ic_species = _sensitivity_ic_species or []
            self._observable_sensitivities = _empty_sens_block(_observable_sensitivities)
            self._expression_sensitivities = _empty_sens_block(_expression_sensitivities)
            self._observable_sensitivities_ic = _empty_sens_block(_observable_sensitivities_ic)
            self._expression_sensitivities_ic = _empty_sens_block(_expression_sensitivities_ic)
            self._reaction_firing_counts = _empty_2d_block(_reaction_firing_counts)
            self._reaction_propensity_integrals = _empty_2d_block(_reaction_propensity_integrals)
            self._reaction_labels = _reaction_labels or []
            self._solver_stats = _solver_stats or {
                "n_steps": 0,
                "n_rhs_evals": 0,
                "n_jac_evals": 0,
                "n_jac_rhs_evals": 0,
                "n_jac_formations": 0,
                "n_err_test_fails": 0,
                "n_nonlin_iters": 0,
                "n_nonlin_conv_fails": 0,
                "n_sens_err_test_fails": 0,
                "n_sens_nonlin_conv_fails": 0,
                "n_roundoff_steps": 0,
                "linear_solver": 0,
                "n_dense_blas_factorizations": 0,
                "steady_state_reached": 0,
            }
            self._ssa_diagnostics = _ssa_diagnostics or _empty_ssa_diagnostics()
            self._psa_diagnostics = _empty_psa_diagnostics()

        # Each run's PSA diagnostics, on a result squeeze stacked from several
        # runs; None on a single run, whose one report is _psa_diagnostics.
        self._psa_diagnostics_runs: list[dict[str, Any]] | None = None

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
        # correct for V=1 SBML and all .net models. A squeezed batch holds one
        # such list per row, since a row may have written a size (issue #743).
        self._species_volume_factors: list[float] | list[list[float]] | None = (
            _species_volume_factors
        )

        # GH #85: species_idx → live-volume column idx for species whose
        # concentration column holds amount/V_live(t) (an hOSU=false species in a
        # rate-rule compartment, whose dilution is integrated). Populated by
        # Simulator._stamp. Tells as_roadrunner() to recover the amount as
        # conc * V_live(t) rather than conc * V_static for those species. None ⇒
        # no such species.
        self._varvol_live_vol: dict[int, int] | None = None

        # GH #131: species_idx → per-sample concentration factor V_static/V_live(t)
        # for a variable-volume species whose raw column was LEFT as the conserved
        # count amount/V_static rather than rescaled in place. as_roadrunner()
        # multiplies it into the ``[S]`` (concentration) selector so [S] reports
        # amount/V_live, while the bare ``S`` (amount) selector recovers the
        # amount as raw·V_static via the volume factor. bngsim recorded it under
        # SSA, and for an hOSU=true event-resized species under ODE too, until
        # issue #1368, which rescales those columns in place under every method:
        # only a result loaded from a file an older bngsim saved carries one.
        # None ⇒ no such species.
        self._varvol_conc_factor: dict[int, np.ndarray] | None = None

        # GH #131: species_idx → per-sample live-volume V_live(t) for a species
        # whose raw column holds the live concentration amount/V_live (the ODE
        # convention for an hOSU=false EVENT-resized species — the #74 V_old/V_new
        # rescale already ran), so the bare ``X`` amount selector reports the
        # amount as raw·V_live rather than the stale raw·V_static the volume factor
        # would give. V_live is read from the event-promoted compartment's
        # observable column. Also carries the size for a column the
        # variable-volume passes rescaled to amount/V_live (V_static at a sample
        # where V_live is 0, which they leave at amount/V_static) and for a
        # rule-target concentration species. Takes priority over the
        # species-column live_vol and the static volume factor in as_roadrunner.
        # None ⇒ no such species.
        self._varvol_amount_factor: dict[int, np.ndarray] | None = None

        # Issue #787: which bngsim wrote the file this result was loaded from
        # ("bngsim 0.11.35"), when that file predates Result.save recording the four
        # volume fields above. Those fields are then None because the file does
        # not say, not because every compartment has size 1, so as_roadrunner
        # refuses a bare-id amount rather than report the stored concentration
        # as one. None for every other result, a file that records them included.
        self._volume_record_missing: str | None = None

        # GH #205: SBML AssignmentRule-target species → (kind, src, vdiv,
        # vdiv_param: the size vdiv is read from, or ""; issue #724), the
        # same map Simulator._apply_ar_report_map uses to overwrite the frozen
        # species value column with its rule's live value. An AR species' OUTPUT
        # sensitivity must follow that assignment expression (the observable for
        # a linear-on-species rule, the function/expression otherwise), not the
        # raw integrated state sensitivity — output_sensitivities("species:<ar>")
        # redirects through this. GH #221 applies the same redirect to the
        # sensitivity TENSOR rows, so the two agree by construction; the redirect
        # is kept here because it also carries the by-name refusals below.
        # Empty for non-AR models (no redirect); a .net model carries the map
        # from_net rebuilt (#515). Set by Simulator._stamp.
        self._ar_sens_map: dict[str, tuple[str, str, float, str]] = {}

        # GH #205: AR-species names whose reported value also carries a
        # time-varying volume rescale (variable-volume compartment, #85/#87; an
        # event-resized one since issue #1368). The
        # redirect above accounts only for the constant vdiv, so a selector for
        # one of them reads its own row of the species tensor, which is put
        # through the quotient rule with the size's row (issue #787).
        self._ar_sens_blocked: frozenset[str] = frozenset()

        # GH #221: AR-species names whose sensitivity TENSOR row carries NaN
        # because the redirect could not be honoured — blocked above, a source
        # output-sensitivity block this run never computed, or (the common one)
        # a source row that is itself NaN because codegen declined that
        # expression, GH #198. Subset of the map's keys; empty on every model
        # that resolves. A structural zero left in place would be
        # indistinguishable from a measured one, which is the failure GH #221 is
        # about. Set by Simulator._stamp.
        self._ar_sens_refused: frozenset[str] = frozenset()

        # Stochastic seed actually used for this simulation. None for
        # ODE results (deterministic) and for results loaded from older
        # files that pre-date seed exposure.
        self._seed: int | None = _seed

        # GH #155: the ∂x(0)/∂θ rows CVODES was seeded with for this run, as
        # {species_name: {param_name: coeff}}. Set by Simulator._stamp (and by
        # the batch/stitch paths); None means "not recorded for this result",
        # which is NOT the same statement as {} ("recorded, and nothing seeds").
        # The `parameter` axis is a TOTAL derivative that already contains
        # (this matrix) · (the `ic` axis), so a consumer composing its own
        # initial-condition chain rule reads this to know what not to add again.
        self._ic_sensitivity_seed: dict[str, dict[str, float]] | None = _ic_sensitivity_seed

        # The label of the stop condition this run ended at, which the result
        # was cut at; None where none was met (issue #1369).
        self._stop_condition: str | None = None

        # Issue #1701: where the run started, "initial" or "live", and the
        # state vector it started from, whose sha256 is computed on first read
        # (a run_until loop of short steps pays for no hash it never reads) and
        # is all a loaded file holds. Set by Simulator.run; None where nothing
        # recorded it (see started_from).
        self._started_from: str | None = None
        self._start_state: NDArray[np.float64] | None = None
        self._start_state_sha256: str | None = None

        # Issue #1703: what Result.record says, set by the Simulator call that
        # made the result (bngsim._record): what the run was asked and did, the
        # parameter values (names, values) and per-species atol it started
        # with, and what the model is. A stack or a file holds the parts as a
        # dict instead (_record_frozen). None where nothing recorded them.
        self._run_info: Any = None
        self._run_params: tuple[tuple[str, ...], Any] | None = None
        self._run_atol: Any = None
        self._model_info: Any = None
        self._record_frozen: dict[str, Any] | None = None
        # Of a stack (squeeze): the record's paths at which its rows differ,
        # each None in the stack's record. None for a single run.
        self._record_differs: tuple[str, ...] | None = None
        # The text the model was loaded from, where it came from a string, of
        # a result loaded from a file (a run's is on _model_info): what
        # rerun() builds the model again from.
        self._source_text: str | bytes | None = None
        # The warnings, fallbacks and complaints of the call that made it,
        # each once, in order (bngsim._notices).
        self._notices: tuple[Notice, ...] = ()

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
        when the caller passed ``seed=None`` (or omitted it). A seed is
        from 0 to 2**64 - 1, and under NFsim from 0 to 2**32 - 1, the
        width its generators read, so the seed passed is the seed used
        (issue #1515).

        For squeezed batch results, this is the single seed if every
        underlying sim used the same one, otherwise ``None``; the
        per-sim seeds remain accessible on the individual ``Result``
        objects from ``Simulator.run_batch(..., squeeze=False)``.
        """
        return self._seed

    @property
    def stop_condition(self) -> str | None:
        """The stop condition this run met, by its label, or ``None``.

        A condition registered with :meth:`Simulator.add_stop_condition` cuts
        the result at the first sample it holds at: :meth:`Simulator.run`
        raises :class:`StopConditionMet` with that result, and
        :meth:`Simulator.run_batch` and :meth:`Simulator.run_replicates`
        return it in their list, with this set, beside the rows that ran to
        the end (issue #1369). A condition registered without a label reads
        as its expression, or a callable as its ``repr()``. :meth:`save` and
        :meth:`to_xarray` carry it.

        A cut result holds the reported blocks only, so :attr:`state` refuses
        on it, as it does on the result :meth:`Simulator.run` raises with.
        """
        return self._stop_condition

    @property
    def started_from(self) -> str | None:
        """Where this run started — ``"initial"``, ``"live"``, or ``None``.

        The state lives in the :class:`Model`, and each run writes its end state
        back there, so a second :meth:`Simulator.run` starts where the first one
        ended unless the model is reset in between (issue #1701).

        - ``"initial"``: the run started from the model's initial conditions,
          the state :meth:`Model.reset` restores. After an unlabeled
          :meth:`Model.save_concentrations` that is the saved state.
        - ``"live"``: it started from any other state: the end of an earlier
          run, a :meth:`Model.set_concentration`, a restored snapshot, a
          :meth:`Simulator.run_until` step.
        - ``None``: not recorded. :meth:`Simulator.run` records it, and so do
          the calls made of it (``run_until``, ``parameter_scan``,
          ``bifurcate``, a protocol's runs), ``run_replicates``,
          ``compute_all_sensitivities`` and, since issue #1703, a
          ``run_batch`` row, which starts from the initial conditions by
          construction (``Model.reset``). A file an earlier bngsim saved does
          not; nor does a :meth:`squeeze` of rows that disagree on it.

        "Initial" means the start vector equals the initial conditions value
        for value. :attr:`start_state_sha256` tells two starts apart further.
        :meth:`summary`, :meth:`save` and :meth:`to_xarray` carry both. For a
        network-free method the start is the model's species vector, which
        the run hands its backend in place of the XML's seed amounts where
        they differ, and which the backend rounds to whole molecules.
        """
        return self._started_from

    @property
    def start_state_sha256(self) -> str | None:
        """The SHA-256, in hex, of the state vector this run started from.

        The digest of the model's full species vector
        (:meth:`Model.get_state`, ordered like :attr:`Model.species_names`) as
        little-endian float64 bytes, the moment before the run integrated,
        with ``-0.0`` read as ``0.0`` and every NaN as one. Two runs of one
        model with the same digest started from the same values; a digest that
        differs says the starts differ even where both are ``"live"`` (issue
        #1701). ``None`` where nothing recorded it, and in a :meth:`squeeze` of
        rows whose starts differ. A squeeze keeps each of the two where its
        rows agree on it, apart: rows from different states can all be
        ``"initial"``, and rows from one state can disagree on it, where it
        was the initial conditions for some (after a
        :meth:`Model.save_concentrations`) and not for others.
        """
        if self._start_state_sha256 is None and self._start_state is not None:
            self._start_state_sha256 = _state_digest(self._start_state)
        return self._start_state_sha256

    @property
    def record(self) -> dict[str, Any]:
        """What this result's numbers are and how its run went, in one
        JSON-serializable dict (issue #1703).

        Read it to say what a number is and how it was computed from the
        result, not from memory of the script: the quantity each column holds,
        the method, the tolerances as applied, where the run started, which
        code ran, what the run warned of, the model file and the build.
        :meth:`summary` returns it too, :meth:`save` keeps it, and
        :meth:`to_xarray` carries it as a JSON string. Each read builds a new
        dict.

        Keys:

        * ``schema_version``: 1. A change that removes or renames a key, or
          changes what one holds, increments it.
        * ``software``: ``bngsim``, the version, and ``build_commit``, the
          commit the compiled extension was built from (``None`` where the
          build did not record one).
        * ``model``: where the model came from, as :meth:`Model.describe`'s
          ``source`` gives it (``format``, ``path`` and the file's ``sha256``,
          ``None`` each for a model built in Python), and two keys
          ``describe()`` does not have: ``edits``, the calls that changed the
          model's equations or data after its load (``"add_table_function"``),
          which its source does not hold; and ``tables_sha256``, a hash of its
          table functions' data (``None`` where it has none), which a table
          read from a file of its own keeps out of the source's hash.
        * ``run``: ``method`` (``"ode"``, ``"ssa"``, ``"psa"``, ``"nfsim"``,
          ``"rulemonkey"``) and ``requested_method`` as passed (``"nf"``);
          ``t_start``, ``t_end`` and ``n_points`` the run was asked for, and
          ``sample_times`` where they were given; ``start``, the ``start=``
          the run was made with (``run_until`` and a scan's points make theirs
          with ``"live"``), and ``started_from``, ``start_state_sha256`` and
          ``start_state``, the state it began from (ordered like
          :attr:`Model.species_names`, whose unreported states it includes);
          ``parameters``, each parameter's value at the start, synthesized
          slots left out, and ``parameters_pinned``, the derived ones a
          ``set_param`` had pinned to a number so that they no longer follow
          their expressions; ``seed`` (as :attr:`seed`); for an ODE run
          ``rtol``, ``atol`` (``given``, what the call or
          :meth:`Simulator.set_tolerances` asked for, ``None`` for the
          default; ``scalar``, ``per_species`` and ``tracking_decades``, what
          the solver was given: ``per_species`` is ``None`` where the scalar
          held every species, and the default and ``"auto"`` scale it from the
          start, issue #1420), ``max_steps``, ``max_step``, ``steady_state``
          and ``steady_state_tol`` (``None``: the default, from ``atol``),
          ``carry_sensitivities``, ``sensitivity_params``,
          ``sensitivity_ic`` and ``sensitivity_method``; ``poplevel`` for
          PSA; ``xml``, the ``path`` and ``sha256`` of the BNG-XML file a
          network-free method ran; ``stop_conditions``, each registered one
          by its label (else its expression); ``timeout``; and
          ``wall_time_s``, the seconds the call took to make this result;
          ``start_carry``, the event state the run started from: ``None`` for
          a fresh start, else ``events`` (``None``, or the time ``t`` a run
          continues it at, each event's trigger truth there and its delayed
          executions not yet applied) and ``ssa_counts`` (the state is an SSA
          run's counts); ``declared_ic_sensitivity``, whether the run seeded
          its sensitivities from :meth:`Model.declare_ic_sensitivity`;
          ``strict_ssa`` and ``reaction_stats``, the Simulator's; and
          ``rerun_of``, of a :meth:`rerun`: the ``record_sha256`` of the
          record it repeated (as JSON with sorted keys) and the settings it
          ``changed``, else ``None``.
          A result a stop condition cut keeps the record of the run it was
          cut from: ``time`` and :attr:`stop_condition` say where it ends.
        * ``backend``: which code ran. ``rhs``: ``"interpreted"``, ``"cc"``
          (compiled C) or ``"mir"`` (the in-process JIT); ``jacobian``:
          ``"analytical"``, ``"difference_quotient"`` or ``"jax"``, as used;
          ``jacobian_requested``; ``jacobian_retried``: True where this run's
          analytical attempt failed and it was made again from its start with
          a difference quotient (GH #176; a later run on the Simulator uses
          the difference quotient from the start, and its notices say why);
          ``linear_solver``: ``"dense"``, ``"klu"`` or ``"lapack_dense"``, and
          ``linear_solver_requested`` (``"auto"``, ``"dense"``,
          ``"sparse"``); ``rhs_requested``, the Simulator's ``codegen=``
          (``None`` for the default); ``sensitivity_rhs``: ``"analytic"`` or
          ``"difference_quotient"`` on a sensitivity run; ``propensities``
          for SSA and PSA: ``"interpreted"``, ``"cc"`` or ``"mir"``. ``None``
          for a part a method does not have.
        * ``time``: ``units``, the SBML unit id of time where the file
          declares one.
        * ``columns``: for each ``species`` and ``observables`` column, its
          ``name``, ``quantity`` and ``compartment`` (and a species'
          ``substance_units``); and ``compartments``, each one's ``size`` for
          this run, ``size_varies`` (a rule or event moves it, and the column
          divides by the size at each time; ``size`` is then ``None``) and
          ``units``. ``quantity`` is ``"concentration"`` (an amount over its
          compartment's size), ``"amount"``, ``"count"`` (a number of
          molecules), or ``None`` where the model does not say. An SBML or
          Antimony model's species columns are concentrations, under every
          method: under a stochastic one, molecules over the size. Its
          observable of one species is the species as the file declares it,
          an amount (a count, under a stochastic method) where the file
          declares the species in amounts. A ``.net`` or BNGL model's columns
          are counts under a stochastic method, and ``None`` under ODE, whose
          numbers are the file's, in units it does not state. bngsim converts
          no unit.
        * ``rows_differ``: of a :meth:`squeeze`, each path (``"run.seed"``,
          ``"run.parameters.kf"``) whose value is ``None`` there because the
          rows differ; ``None`` for a single run.
        * ``notices``: :attr:`notices`.

        Recorded by :meth:`Simulator.run` and the calls that make their
        results with it (:meth:`~Simulator.run_until`,
        :meth:`~Simulator.parameter_scan`, :meth:`~Simulator.bifurcate`, a
        protocol's simulate actions), and by :meth:`~Simulator.run_batch` and
        :meth:`~Simulator.run_replicates`. ``None`` for a part nothing
        recorded: a :meth:`~Simulator.compute_all_sensitivities` result (which
        has its notices), a network-free session's, a protocol action that
        runs no time, a result built from arrays, or one loaded from a file an
        earlier bngsim saved. In a :meth:`squeeze`, each value the rows hold
        alike, and ``None`` where they differ, which ``rows_differ`` lists;
        each row's own record is on the rows ``squeeze=False`` returns.

        Values are Python floats, so a parameter of ``inf`` is written by
        ``json.dumps`` as ``Infinity``, which strict JSON does not have.
        """
        return _record.record(self)

    def rerun(self, **changes: Any) -> Result:
        """Make the run this result records again, and return its result
        (issue #1703).

        "Again" has one reading, the record's: the model built from its source
        (the recorded file, whose sha256 must still match, or the text a
        string load kept), the parameter values and the state the run started
        from, the event state it continued, the method, the seed, the sample
        times, the tolerances as the solver was given them, and the Jacobian,
        linear solver and right-hand side that ran. The model the run used is
        neither read nor changed, so a rerun inherits nothing a later run left
        behind (issue #1701). An ODE run comes back bit for bit, and so does a
        stochastic one on the same build.

        Parameters
        ----------
        **changes
            Settings to change, by the keyword :meth:`Simulator.run` or
            :class:`Simulator` takes: ``t_span``, ``n_points``,
            ``sample_times``, ``seed``, ``rtol``, ``atol``, ``max_steps``,
            ``max_step``, ``timeout``, ``steady_state``, ``steady_state_tol``,
            ``method``, ``poplevel``, ``jacobian``, ``codegen``; and
            ``linear_solver`` (an ODE run's: ``"dense"``, ``"klu"``, or
            ``"auto"`` for the Simulator's own rule; the Simulator's
            ``force_dense_linear_solver`` and ``force_sparse_linear_solver``:
            ``"klu"`` runs dense on a build without KLU or with a JAX
            Jacobian, with a ``rerun.different_backend`` notice, and raises
            :class:`SparseJacobianUnavailableError` for a model with no
            sparse Jacobian to give KLU). Every other setting is kept:
            ``rerun(rtol=1e-10)`` keeps the absolute tolerances the run was
            given. A new ``method`` drops the settings of the old one it does
            not take (an ODE run's tolerances under ``"ssa"``).

        Returns
        -------
        Result
            The new run's. Its record's ``run.rerun_of`` names the record it
            repeated, by sha256, and what was changed; its ``run.start`` is
            ``"live"``, as the start is given to it.

        Raises
        ------
        UsageError
            ``rerun.not_recorded`` where the record lacks a part the rerun
            needs (``details["missing"]``); ``rerun.unsupported`` for a run a
            record cannot make again (``details["reason"]``: a stack, a
            network-free run, stop conditions, carried or declared
            sensitivities, sensitivities the rebuilt model seeds otherwise, a
            model changed after its load or built in Python);
            ``rerun.unknown_setting`` for a setting it does not change.
        ModelError
            ``rerun.model_changed`` where the model file's sha256 differs from
            the record's, or the model built from the source is not the one the
            run used (its table data, parameters or species differ);
            ``rerun.model_missing`` where the file cannot be read.

        Notes
        -----
        A model loaded from a string keeps its text for this, for as long as
        the model or a result of it lives, and :meth:`save` writes the text
        into the file: a saved result of such a model carries the model.

        A rerun on another bngsim version or build than the run's, or one
        that ran other code (a compiled right-hand side where the run's was
        interpreted), runs and says so in its :attr:`notices`
        (``rerun.different_build``, ``rerun.different_backend``).

        Examples
        --------
        >>> again = result.rerun()  # the same numbers
        >>> tight = result.rerun(rtol=1e-10)  # the same run, at a tighter rtol
        >>> other = result.rerun(seed=7)  # a stochastic run's other realization
        >>> dense = result.rerun(linear_solver="dense")  # the same run, not on KLU
        """
        from bngsim._rerun import rerun

        return rerun(self, **changes)

    @property
    def notices(self) -> list[dict[str, str]]:
        """Every warning, fallback and solver complaint of the call that made
        this result, each once (issue #1703): the run's own in the order it
        gave them, then those of the call around it (a batch's, a scan's, a
        ``run_until`` step's).

        Each is a dict: ``category``, the warning's class name
        (``"AtolScaleWarning"``), ``"log"`` for one the ``bngsim`` logger
        gave, or ``"notice"`` for one given nowhere else; ``code``, a stable
        key (``"jacobian.difference_quotient_retry"``, ``"atol.below_scale"``,
        ``"start.continues_last_run"``, ...) whose wording does not change
        between releases where the message may; and ``message``. A notice is
        recorded here whether or not a warnings filter or a logging level let
        it through: a warning the default filter shows only once is on every
        result it concerns. A notice of a call about all its results (a
        batch's rows that start from the initial conditions) is on each; one
        about some of them (a row whose species stay below the tolerance) on
        those alone.

        Not here: what the :class:`Simulator` said when it was built (its SSA
        validation, a compiler it could not use), which :attr:`record`'s
        ``backend`` reflects; what a protocol says of its actions as it reads
        them; and the lines SUNDIALS writes to standard error itself. An empty
        list is a run that warned of nothing else. The error-code reference
        page lists every notice code with what to do about it, and a run that
        raises keeps its notices on the error, as :attr:`BngsimError.notices
        <bngsim.BngsimError.notices>`.
        """
        return [n.as_dict() for n in self._notices]

    @property
    def ic_sensitivity_seed(self) -> dict[str, dict[str, float]] | None:
        """``∂x(0)/∂θ`` — the initial-condition seeding this run's gradients carry.

        The **per-run record** of the matrix
        :meth:`Model.effective_ic_sensitivity` reports. Prefer that one to build
        gradient routing: it answers at setup, from model structure alone, with
        no simulation. Reach for this to see what a *particular* run used —
        the only correct answer for a batch or scan over a nonlinear derived
        initial condition (``Rtot = R0*scale``), whose coefficients move from
        point to point, and the one that reports ``None`` for a carried-seed
        phase (below).

        ``{species_name: {param_name: ∂x_k(0)/∂θ}}``, the forward-sensitivity
        seed CVODES was actually initialized with for this simulation: the
        parameter-graph derivative of each species' initial condition (issues
        #43/#147), *after* the issue #113 retirement of a species moved off its
        declared initial condition and the issue #111
        :meth:`Model.declare_ic_sensitivity` overlay.

        Read this to compose your own initial-condition chain rule without
        double-counting. :meth:`output_sensitivities` with ``axis="parameter"``
        is a **total** derivative:

        .. code-block:: text

            d_param[θ] = (right-hand-side path) + Σ_k (∂x_k(0)/∂θ)·d_ic[x_k]

        so for any ``θ`` appearing here, the seeding term is *already inside*
        ``d_param[θ]`` and adding an ``ic``-axis term for it counts it twice. A
        ``θ`` absent from this mapping has an RHS-only ``d_param`` column, and
        an initial-condition term your own code supplies is the missing piece
        (or hand it to :meth:`Model.declare_ic_sensitivity` and let the
        ``parameter`` axis carry it instead).

        Only parameters in :attr:`sensitivity_params` appear — a row the engine
        computed for a parameter you did not request has no column to describe.
        Coefficients are evaluated at *this* run's parameter values, so a batch
        or scan over a nonlinear derived initial condition
        (``Rtot = R0*scale``) reports a different matrix per point.

        Returns
        -------
        dict or None
            ``{}`` means recorded, and **nothing** seeds — every ``d_param``
            column is right-hand-side only. ``None`` means *not recorded*: no
            parameter sensitivities were requested; the result came from a
            non-ODE method or an HDF5 file predating this field; a squeezed
            batch whose rows disagreed (the per-row ``Result`` objects from
            ``run_batch(..., squeeze=False)`` still carry theirs); or the run
            used ``carry_sensitivities=True``, where the seed is the *prior*
            phase's ``dx/dθ`` and these rows are discarded by the engine
            (GH #210/#81 — and no double-count can arise there, because an
            ``ic`` axis across a carry boundary is refused outright). The two
            are deliberately distinct — an empty mapping is a positive
            statement.

        Examples
        --------
        >>> sim = Simulator(model, method="ode", sensitivity_params=["R0", "kf"])
        >>> r = sim.run(t_span=(0, 10), n_points=11)
        >>> r.ic_sensitivity_seed                       # doctest: +SKIP
        {'R(r)': {'R0': 1.0}}
        """
        return self._ic_sensitivity_seed

    @property
    def time(self) -> NDArray[np.float64]:
        """Time points, shape ``(n_times,)``."""
        return self._time

    @property
    def species(self) -> _SpeciesArray:
        """Species concentrations, shape ``(n_times, n_species)``.

        A numpy array, which also takes a species name where a column index
        goes, as :attr:`observables` does: ``result.species["A"]`` is the
        column of species ``A``, ``result.species[-1, "A"]`` its last value. On
        a squeezed batch the name selects along the last axis, as an index of
        :attr:`species_names` would. A fixed species may be named with its
        ``$`` or without, as :meth:`Model.set_concentration` takes it. Only
        this array takes a name: one numpy derives from it (a copy, a
        transpose, a sort) carries none, since its columns need not be these
        (issue #1700).
        """
        cached = self._species_view
        if cached and cached[0] is self._species and cached[1] is self._species_names:
            return cached[2]
        view = _SpeciesArray.over(self._species, self._species_names, self)
        cached[:] = (self._species, self._species_names, view)
        return view

    @property
    def state(self) -> NDArray[np.float64]:
        """The full integrator state trajectory, shape ``(n_times, n_state)``.

        Every core state entry, exactly as the integrator held it and in
        :attr:`bngsim.Model.species_names` order, so a row is a state
        :meth:`bngsim.Model.rhs` and :meth:`bngsim.Model.jacobian` accept as
        is (issue #508). It differs from :attr:`species` on two kinds of
        model. An SBML model whose event assigns a parameter or a compartment
        promotes the target to a state entry that :attr:`species` and
        :attr:`species_names` omit (GH #71), so this has more columns. A
        model whose reported species column is remapped after the solve — an
        assignment-rule target reported at the rule's live value, or a
        species in an event-resized or rate-ruled compartment reported at
        ``amount / V_live(t)`` (GH #85, #131) — reports the remapped value in
        :attr:`species` and the stored one here. Elsewhere the two are
        identical.

        Only a result returned by a solve carries it: a result loaded from
        disk or stacked from a batch holds the reported species block alone.

        Raises
        ------
        ValueError
            If this result was not produced directly by a solve.
        """
        return self._state_core("state").state_data

    @property
    def state_names(self) -> list[str]:
        """Names of the :attr:`state` columns: the producing model's
        :attr:`bngsim.Model.species_names`, promoted entries included."""
        return list(self._state_core("state_names").state_names)

    def _state_core(self, what: str) -> ResultCore:
        if self._core is None:
            # A row of a batch that a stop condition cut is a solve's too, so
            # the generic reason would misname it (issue #1369).
            how = (
                f"this one was cut at stop condition {self._stop_condition!r}, and a cut Result"
                if self._stop_condition is not None
                else "a loaded or stacked Result"
            )
            raise UsageError(
                f"Result.{what} is only available on a Result returned by a solve; "
                f"{how} carries only the reported species block."
            )
        return self._core

    @property
    def observables(self) -> _ObservableAccessor:
        """Observable values with named access.

        Can be used as:
        - ``result.observables`` → full array ``(n_times, n_obs)``
        - ``result.observables["name"]`` → single column ``(n_times,)``
        """
        return _ObservableAccessor(
            self._observables, self._observable_names, "observable", self._names_by_kind()
        )

    @property
    def expressions(self) -> _ObservableAccessor:
        """Expression (function) values with named access."""
        return _ObservableAccessor(
            self._expressions, self._expression_names, "expression", self._names_by_kind()
        )

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
                self._core.raw_expression_data,
                list(self._core.raw_expression_names),
                "expression",
                self._names_by_kind(),
                "result.raw_expression_names",
            )
        return _ObservableAccessor(
            self._expressions, self._expression_names, "expression", self._names_by_kind()
        )

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
        function (everything else), GH #205. Since GH #221 the
        :attr:`sensitivities` tensor carries that same redirected row, so this
        method and the tensor agree for those species; the redirect is kept here
        because it raises **by name** where the tensor can only hold ``NaN``
        (an expression whose output sensitivity a construct declines). A rule
        target in a compartment whose size changes is read from the tensor, whose
        row has the size's share in it (issue #787), and is refused by name where
        that row is ``NaN``.

        .. rubric:: The two axes are not independent (GH #155)

        ``axis="parameter"`` is the **total** derivative ``dy(t)/dθ``: it carries
        every path by which ``θ`` reaches the trajectory, the right-hand side
        **and** the initial-condition seeding ``∂x(0)/∂θ`` (issues #43/#147).
        ``axis="ic"`` is ``∂y(t)/∂x_k(0)`` with the initial value held as an
        *independent* variable. So for a parameter that seeds an initial
        condition the two overlap:

        .. code-block:: text

            d_param[θ] = (right-hand-side path) + Σ_k (∂x_k(0)/∂θ)·d_ic[x_k]

        **Do not sum the axes.** A consumer that routes a fitted parameter to
        several columns and adds them must add an ``ic`` term only for the part
        of ``∂x(0)/∂θ`` this engine does not already carry — which
        :attr:`ic_sensitivity_seed` reports exactly, per run. The common case
        that makes the overlap easy to miss: a parameter appearing *only* in the
        initial condition, with coefficient 1, gives a ``parameter`` column
        bit-identical to the seeded species' ``ic`` column (same seed vector,
        same variational equation), so summing reports it at exactly 2×.

        On :class:`SteadyStateResult` the question does not arise: that
        ``parameter`` axis is the implicit-function derivative of the algebraic
        system, there is no seeding term in it, and ``axis="ic"`` raises.

        Parameters
        ----------
        selectors : str or iterable of str
            Selectors accepted by :meth:`resolve_outputs`.
        axis : {"parameter", "ic"}, optional
            Which sensitivity axis to return. ``"parameter"`` (default)
            gives ``d output/dp`` over :attr:`sensitivity_params` — the total
            derivative, seeding included, see above; ``"ic"`` gives
            ``d output/dY(0)`` over :attr:`sensitivity_ic_species`.

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
            raise UsageError(
                f"output_sensitivities: axis must be 'parameter' or 'ic', got {axis!r}."
            )
        names = self._sensitivity_params if axis == "parameter" else self._sensitivity_ic_species
        n_axis = len(names)
        if n_axis == 0:
            if axis == "parameter":
                raise UsageError(
                    "output_sensitivities: no parameter sensitivities were computed for this "
                    "result. Enable them via Simulator(..., sensitivity_params=[...]).",
                    code="sens.not_computed",
                    remedies=[pass_argument("sensitivity_params")],
                )
            raise UsageError(
                "output_sensitivities: no initial-condition sensitivities were computed for this "
                "result. Enable them via Simulator(..., sensitivity_ic=[...]).",
                code="sens.not_computed",
                remedies=[pass_argument("sensitivity_ic")],
            )
        meta = self.resolve_outputs(selectors)
        if not meta:
            return np.empty(self._time.shape + (0, n_axis), dtype=np.float64)
        cols = [self._output_sensitivity_slice(m, axis) for m in meta]
        return np.stack(cols, axis=-2)

    def _resolve_one_output(self, selector: str) -> dict[str, Any]:
        """Resolve a single selector string to its metadata dict.

        Delegates to the shared :func:`_resolve_output_selector` so a
        :class:`Result` and a :class:`~bngsim._simulator.SteadyStateResult`
        accept selectors by identical rules (GH #12).
        """
        return _resolve_output_selector(
            selector,
            self._species_names,
            self._observable_names,
            self._expression_names,
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
        # the raw integrated ``yS`` of the frozen state (which is ~0). GH #221
        # puts that same redirected row in the species tensor, so this recursion
        # is no longer the only place it exists — it is kept because recursing on
        # the SOURCE selector propagates that source's support / empty-block
        # errors, which a NaN row in the tensor cannot state by name.
        #
        # A species in a compartment whose size moves is reported as its rule
        # over the live size, which the redirect below, a division by a
        # constant, does not give. Its own row of the species tensor is that
        # value's derivative (issue #787), so the selector reads it, and refuses
        # by name where the row is NaN. Checked before the redirect, which a
        # loaded result does not carry, so the answer is the same after a save
        # and load.
        live_sized = kind == "species" and meta["name"] in self._ar_sens_blocked
        if live_sized and meta["name"] in self._ar_sens_refused:
            raise UsageError(
                f"output_sensitivities: assignment-rule species "
                f"{meta['selector']!r} is reported over the live size of a "
                "compartment whose size changes, and its row of "
                "Result.sensitivities is NaN: it needs the output sensitivity "
                "of the rule and that of the size, and the run has not got "
                "both: codegen declined one of them, or built no "
                "output sensitivities for this model. "
                "Result.ar_sensitivity_refused lists every such species."
            )
        if kind == "species" and not live_sized:
            redirect = self._ar_sens_map.get(meta["name"])
            if redirect is not None:
                src_kind, src_name, vdiv, vdiv_param = redirect
                src_names = self._names_for_kind(src_kind)
                if src_name in src_names:
                    src_meta = _output_meta(src_kind, src_name, src_names.index(src_name))
                    sl = self._output_sensitivity_slice(src_meta, axis)
                    sl = sl / vdiv if vdiv != 1.0 else sl
                    if (
                        vdiv_param
                        and axis == "parameter"
                        and vdiv_param in self._sensitivity_params
                    ):
                        # Issue #724: the reported value is rule/V, and in the
                        # column of the size V it has the term −(rule/V)/V. The
                        # recursion above is of the rule's own selector, which
                        # does not have it. (A size that a parameter sets, issue
                        # #696, would bring a factor dV/dθ here.)
                        sl = np.array(sl, dtype=np.float64, copy=True)
                        value = self._species[..., meta["index"]]
                        for c, column in enumerate(self._sensitivity_params):
                            if column == vdiv_param:
                                sl[..., c] -= value / vdiv
                    return sl
                # Rule source not reported (shouldn't happen for a loaded AR
                # model) — fall through to the raw species block below.
        # An unsupported expression has a NaN row (or none) — report WHY (the
        # specific construct / table function) rather than return NaN or a bare
        # empty-block error (GH #198 fails loudly).
        if kind == "expression":
            reason = self._expression_sens_support.get(meta["name"])
            if reason:
                raise UsageError(
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
            raise UsageError(
                f"output_sensitivities: no {kind} sensitivities are available for selector "
                f"{meta['selector']!r}.{extra}"
            )
        # Every sensitivity block projects its row axis by the same rule its
        # name list uses (GH #202), so a resolved index addresses the matching
        # row. This is a defensive check on that invariant — the one way to
        # reach it now is an ``.h5`` written before GH #202, whose stored
        # species block still carries the unreported (promoted-parameter) rows
        # its ``species_names`` dropped. Refuse rather than return a silently
        # mismatched slice.
        if block.shape[-2] != len(self._names_for_kind(kind)):
            raise UsageError(
                f"output_sensitivities: cannot address {kind} selector "
                f"{meta['selector']!r} by index — the {kind} sensitivity block has "
                f"{block.shape[-2]} rows but this result names "
                f"{len(self._names_for_kind(kind))} {kind}s. Re-run the "
                "simulation, or use the .sensitivities array directly."
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

        Unlike the gradient methods, the FIM weights **every** output it is built
        over, so there is no unweighted row to drop: a model with a refused
        assignment-rule row (:attr:`ar_sensitivity_refused`, GH #221) returns a
        ``NaN`` matrix by default. Name the outputs you want in *outputs* to
        build it over the rows that are known.

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

        Species (``outputs=None``) read the species block directly — one row
        per :attr:`species_names` entry, so a per-species *sigma* vector is
        ``n_species`` long (GH #202). Named outputs route through
        :meth:`output_sensitivities`, which stacks one slice per selector.
        """
        if axis not in ("parameter", "ic"):
            raise UsageError(
                f"fisher_information: axis must be 'parameter' or 'ic', got {axis!r}."
            )
        have = self.has_sensitivities if axis == "parameter" else self.has_sensitivities_ic
        if not have:
            which = "sensitivity_params=[...]" if axis == "parameter" else "sensitivity_ic=[...]"
            raise UsageError(
                "No sensitivity data available. Run the simulation "
                f"with {which} to compute sensitivities.",
                code="sens.not_computed",
                remedies=[pass_argument(which.split("=")[0])],
            )
        if self._species.ndim != 2:
            raise UsageError(
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
                raise UsageError("sigma must be > 0.")
            inv_var = np.full(nr, 1.0 / (sigma_arr.item() ** 2))
        elif sigma_arr.ndim == 1:
            if sigma_arr.shape[0] != nr:
                raise UsageError(
                    f"sigma array length ({sigma_arr.shape[0]}) must match {row_label} ({nr})."
                )
            if np.any(sigma_arr <= 0):
                raise UsageError("All sigma values must be > 0.")
            inv_var = 1.0 / (sigma_arr**2)
        else:
            raise UsageError(f"sigma must be a scalar or 1-D array, got shape {sigma_arr.shape}.")

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

            - ``species`` — :attr:`species`, the ``(n_times, n_species)``
              array of species values
            - ``time`` — ``(n_times,)`` array of time points
            - returns ``dL_dY`` — ``(n_times, n_species)`` array of
              :math:`\partial L / \partial Y_{i,t}`.

            ``dL_dY`` is contracted against :attr:`sensitivities`, whose
            species axis is the same one ``species`` is indexed on (GH #202),
            so a ``dL_dY`` shaped like the ``species`` argument always fits.

            Common example (sum-of-squares vs data):

            .. code-block:: python

                def loss_fn(species, time):
                    return 2 * (species - data)  # dL/dY for SSE

        Returns
        -------
        ndarray, shape ``(n_params,)``
            Parameter gradient :math:`\nabla_p L`, one entry per column of
            :attr:`sensitivities`, ordered like :attr:`sensitivity_params`.
            That list — not :attr:`Model.param_names` — is what the returned
            vector is aligned with, and it is what an optimizer's parameter
            vector must be built from (issue #203; see Notes).

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

        >>> # Use with scipy L-BFGS-B. The fitted vector is the *default*
        >>> # sensitivity column set, so `grad` lines up with `p_vec`.
        >>> from scipy.optimize import minimize
        >>> names = model.primary_param_names
        >>> def objective(p_vec):
        ...     model.set_params(dict(zip(names, p_vec)))
        ...     model.reset()
        ...     result = sim.compute_all_sensitivities(...)
        ...     assert result.sensitivity_params == names
        ...     loss = np.sum((result.species - data)**2)
        ...     grad = result.gradient(
        ...         lambda sp, t: 2 * (sp - data)
        ...     )
        ...     return loss, grad
        >>> x0 = [model.get_param(n) for n in names]
        >>> minimize(objective, x0, method='L-BFGS-B', jac=True)

        Notes
        -----
        **The columns must be independent coordinates, and under the
        ``compute_all_sensitivities()`` default they are** (issue #203). This
        method contracts the *whole* parameter axis into one ``(n_params,)``
        vector, so it is only a gradient if no two columns describe the same
        physical effect. A **derived** parameter's column does describe an
        effect the primaries underneath it already carry — ``_rateLaw1 =
        chi*kon`` reaches the trajectory only through ``chi`` and ``kon``, whose
        own columns are total derivatives *through* it — so summing all three
        counts that effect twice. ``compute_all_sensitivities(params=None)``
        drops the derived columns for this reason and warns; a
        ``sensitivity_params=`` list you wrote yourself is your own to keep
        independent, and if it names both a derived parameter and a primary
        underneath it, do not hand this vector to an optimizer over both.
        (Same shape as issue #155 one level down: the ``parameter`` axis of an
        initial-condition sensitivity is a *total* derivative and must not be
        summed with the ``ic`` axis.)

        The gradient computation is O(n_times × n_species × n_params) —
        a single matrix multiply per time point. With parallel
        ``compute_all_sensitivities()``, the total cost of computing
        both the loss value and its gradient is ~1.2× a plain ODE solve
        (with sufficient cores).
        """
        if not self.has_sensitivities:
            raise UsageError(
                "No sensitivity data available. Run the simulation "
                "with sensitivity_params or compute_all_sensitivities() "
                "to compute sensitivities.",
                code="sens.not_computed",
                remedies=[pass_argument("sensitivity_params")],
            )
        if not callable(loss_fn):
            raise ArgumentTypeError(f"loss_fn must be callable, got {type(loss_fn).__name__}")

        sens = self._sensitivities  # (n_times, n_species, n_params)
        nt, ns, np_ = sens.shape

        # Evaluate loss gradient dL/dY at all time points
        dL_dY = np.asarray(
            loss_fn(self._species, self._time),
            dtype=np.float64,
        )
        if dL_dY.shape != (nt, ns):
            raise UsageError(
                f"loss_fn must return shape (n_times={nt}, n_species={ns}), got {dL_dY.shape}"
            )

        # ∇_p L = Σ_t (dY/dp)^T · (dL/dY)
        # Vectorized: contract over time and species via einsum.
        grad = self._contract_dL_dY(sens, dL_dY, self._refused_row_mask())  # (n_params,)

        return grad

    def _refused_row_mask(self) -> NDArray[np.bool_] | None:
        """Species-axis mask of the GH #221 NaN rows, or ``None`` if there are none.

        Aligned with :attr:`species_names`, so a caller that subsets the species
        axis (``species_indices=``) must subset this the same way.
        """
        if not self._ar_sens_refused:
            return None
        return np.array([n in self._ar_sens_refused for n in self._species_names], dtype=bool)

    @staticmethod
    def _contract_dL_dY(
        sens: NDArray[np.float64],
        dL_dY: NDArray[np.float64],
        refused: NDArray[np.bool_] | None,
    ) -> NDArray[np.float64]:
        """``Σ_t (dY/dp)^T·(dL/dY)``, with the GH #221 NaN rows carried correctly.

        A refused assignment-rule row is ``NaN`` on purpose — the derivative of
        that column is genuinely unknown, and reporting 0.0 is the failure #221
        is about. But IEEE makes ``0 · NaN`` NaN, so without this an unknown row
        the loss puts **no weight on** would turn every parameter of the gradient
        into NaN. Entries where ``dL/dY`` is exactly zero are therefore dropped
        before the contraction (they contributed exactly zero anyway, so a run
        with nothing refused takes the untouched ``einsum`` path). An entry the
        loss *does* weight keeps its NaN and poisons the gradient — correctly:
        that number is not known, and must not come back looking like one.
        """
        if refused is not None and refused.any():
            drop = refused[None, :] & (dL_dY == 0.0)
            if drop.any():
                sens = np.where(drop[:, :, None], 0.0, sens)
        return np.einsum("tsi,ts->i", sens, dL_dY)

    def _fit_target(
        self,
        target: str,
        species_indices: list[int] | None,
        observable_indices: list[int] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_] | None]:
        """``(Y, dY/dp, refused rows)`` of what a fit compares data with.

        ``target`` names it, so the shape of the data never decides it: data
        with as many observables as there are species was scored against the
        species, with their sensitivities, and sse_gradient returned a
        gradient of the wrong sign with no warning (issue #1363).
        """
        if not self.has_sensitivities:
            raise UsageError(
                "No sensitivity data. Run with sensitivity_params or compute_all_sensitivities().",
                code="sens.not_computed",
                remedies=[pass_argument("sensitivity_params")],
            )
        if target == "species":
            if observable_indices is not None:
                raise UsageError(
                    "observable_indices selects observables; pass target='observables'",
                    code="usage.conflicting_arguments",
                )
            Y = self._species
            sens = self._sensitivities  # (nt, ns, np)
            # The rows of the block are the columns of the values (GH #202). An
            # .h5 written before that carries rows species_names dropped, and
            # species_indices would pick rows from a block they do not address.
            if Y.shape[1] != sens.shape[1]:
                raise UsageError(
                    f"{Y.shape[1]} species but {sens.shape[1]} species sensitivity rows; "
                    "re-run the simulation"
                )
            refused = self._refused_row_mask()
            if species_indices is not None:
                Y = Y[:, species_indices]
                sens = sens[:, species_indices, :]
                if refused is not None:
                    refused = refused[species_indices]
            return Y, sens, refused
        if target == "observables":
            if species_indices is not None:
                raise UsageError(
                    "species_indices selects species; pass target='species'",
                    code="usage.conflicting_arguments",
                )
            if not self.has_sensitivities_observables:
                raise UsageError(
                    "No observable sensitivities in this result, so a fit to observable "
                    "data has no gradient. Run with sensitivity_params on a model with "
                    "observables.",
                    code="sens.not_computed",
                    remedies=[pass_argument("sensitivity_params")],
                )
            Y = self._observables
            sens = self._observable_sensitivities  # (nt, n_obs, np)
            if Y.shape[1] != sens.shape[1]:
                raise UsageError(
                    f"{Y.shape[1]} observables but {sens.shape[1]} observable sensitivity rows"
                )
            if observable_indices is not None:
                Y = Y[:, observable_indices]
                sens = sens[:, observable_indices, :]
            # A run writes no NaN observable row, a refused species (GH #221)
            # included: an observable reads the integrated state, not the value
            # an assignment rule writes over a species, and its row is the chain
            # rule through that state's rows, which the core refuses when they
            # are not finite (GH #384). A row made NaN by hand is carried as
            # _contract_dL_dY carries a refused species row.
            unknown = np.isnan(sens).any(axis=(0, 2))
            return Y, sens, unknown if unknown.any() else None
        raise UsageError(f"target must be 'species' or 'observables', got {target!r}")

    def _check_data_shape(
        self, data: NDArray[np.float64], Y: NDArray[np.float64], target: str
    ) -> None:
        if data.shape == Y.shape:
            return
        hint = ""
        if target == "species" and data.shape == self._observables.shape:
            hint = "; it is the shape of the observables: pass target='observables'"
        raise UsageError(f"data shape {data.shape} != {target} shape {Y.shape}{hint}")

    def sse_gradient(
        self,
        data: NDArray[np.float64],
        *,
        target: str = "species",
        species_indices: list[int] | None = None,
        observable_indices: list[int] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        r"""Compute SSE loss and parameter gradient in one call.

        Sum-of-squared-errors: :math:`L = \sum_{t,i} (Y_{t,i} - D_{t,i})^2`.

        This is the most common objective in parameter estimation. The
        analytical derivative is :math:`\partial L / \partial Y = 2(Y - D)`,
        so no user-supplied ``loss_fn`` is needed.

        Parameters
        ----------
        data : ndarray, shape ``(n_times, n_species)`` or ``(n_times, n_obs)``
            Observed data, of the shape of what ``target`` names (or of the
            columns ``species_indices`` / ``observable_indices`` select).
        target : {"species", "observables"}
            What the data is compared with: :attr:`species`, through
            :attr:`sensitivities`, or :attr:`observables`, through
            :attr:`sensitivities_observables` (issue #1363). Named, not read
            from the data's shape, which is the same for both when a model
            has as many observables as species.
        species_indices : list[int], optional
            If given, only these species columns are used for the loss
            (``target="species"``). Sensitivities for all parameters are
            still included.
        observable_indices : list[int], optional
            The same, over observables (``target="observables"``).

        Returns
        -------
        tuple[float, ndarray]
            ``(loss, gradient)`` where *loss* is the scalar SSE and
            *gradient* is shape ``(n_params,)``.

        Raises
        ------
        ValueError
            If sensitivity data is missing, ``target`` is neither, or shapes
            don't match.

        Examples
        --------
        >>> result = sim.compute_all_sensitivities(...)
        >>> loss, grad = result.sse_gradient(observed_data)
        >>> loss, grad = result.sse_gradient(observed_obs, target="observables")
        >>> # Use with scipy L-BFGS-B:
        >>> minimize(objective, x0, method='L-BFGS-B', jac=True)
        """
        data = np.asarray(data, dtype=np.float64)
        Y, sens, refused = self._fit_target(target, species_indices, observable_indices)
        self._check_data_shape(data, Y, target)

        residual = Y - data
        loss = float(np.sum(residual**2))

        # dL/dY = 2 * (Y - data), then contract with sens
        # Vectorized: contract over time and species via einsum.
        dL_dY = 2.0 * residual  # (nt, ns_sel)
        grad = self._contract_dL_dY(sens, dL_dY, refused)  # (n_params,)

        return loss, grad

    def chi2_gradient(
        self,
        data: NDArray[np.float64],
        sigma: float | NDArray[np.float64],
        *,
        target: str = "species",
        species_indices: list[int] | None = None,
        observable_indices: list[int] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        r"""Compute chi-squared loss and parameter gradient.

        :math:`\chi^2 = \sum_{t,i} \left(\frac{Y_{t,i} - D_{t,i}}{\sigma_i}\right)^2`

        Equivalent to weighted SSE with weights :math:`1/\sigma_i^2`.
        The derivative is :math:`\partial L / \partial Y_{t,i} = 2(Y - D)/\sigma_i^2`.

        Parameters
        ----------
        data : ndarray, shape ``(n_times, n_species)`` or ``(n_times, n_obs)``
            Observed data, of the shape of what ``target`` names.
        sigma : float or ndarray
            Measurement noise standard deviation.

            - **scalar** — same σ for all columns.
            - **1-D array, shape ``(n_columns,)``** — per-column σ.
            - **2-D array, shape ``(n_times, n_columns)``** — per-point σ.

        target : {"species", "observables"}
            What the data is compared with, as in :meth:`sse_gradient`.
        species_indices : list[int], optional
            Subset of species columns to include.
        observable_indices : list[int], optional
            Subset of observable columns to include.

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
        data = np.asarray(data, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        Y, sens, refused = self._fit_target(target, species_indices, observable_indices)
        self._check_data_shape(data, Y, target)

        # Build 1/sigma^2 array broadcastable to (nt, ns)
        inv_var: float | NDArray[np.float64]
        if sigma_arr.ndim == 0:
            inv_var = 1.0 / (sigma_arr.item() ** 2)
        elif sigma_arr.ndim == 1:
            inv_var = 1.0 / (sigma_arr**2)  # (ns,)
        elif sigma_arr.ndim == 2:
            inv_var = 1.0 / (sigma_arr**2)  # (nt, ns)
        else:
            raise UsageError(f"sigma must be scalar, 1-D, or 2-D, got shape {sigma_arr.shape}")

        residual = Y - data
        chi2 = float(np.sum(residual**2 * inv_var))

        # dL/dY = 2 * (Y - D) / sigma^2
        # Vectorized: contract over time and species via einsum.
        dL_dY = 2.0 * residual * inv_var
        grad = self._contract_dL_dY(sens, dL_dY, refused)  # (n_params,)

        return chi2, grad

    def neg_log_likelihood_gradient(
        self,
        data: NDArray[np.float64],
        sigma: float | NDArray[np.float64],
        *,
        target: str = "species",
        species_indices: list[int] | None = None,
        observable_indices: list[int] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        r"""Compute negative Gaussian log-likelihood and gradient.

        :math:`-\ln \mathcal{L} = \frac{1}{2} \sum_{t,i}
        \left[\left(\frac{Y_{t,i} - D_{t,i}}{\sigma_i}\right)^2
        + \ln(2\pi\sigma_i^2)\right]`

        The constant term :math:`\ln(2\pi\sigma^2)` doesn't affect the
        gradient but is included in the loss value for correct log-likelihood.

        Parameters
        ----------
        data : ndarray, shape ``(n_times, n_species)`` or ``(n_times, n_obs)``
            Observed data, of the shape of what ``target`` names.
        sigma : float or ndarray
            Measurement noise standard deviation (scalar, per-column,
            or per-point).
        target : {"species", "observables"}
            What the data is compared with, as in :meth:`sse_gradient`.
        species_indices : list[int], optional
            Subset of species columns.
        observable_indices : list[int], optional
            Subset of observable columns.

        Returns
        -------
        tuple[float, ndarray]
            ``(nll, gradient)`` — negative log-likelihood and gradient.

        Examples
        --------
        >>> nll, grad = result.neg_log_likelihood_gradient(data, sigma=0.1)
        """
        data = np.asarray(data, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        Y, sens, refused = self._fit_target(target, species_indices, observable_indices)
        self._check_data_shape(data, Y, target)

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
            raise UsageError(f"sigma must be scalar, 1-D, or 2-D, got shape {sigma_arr.shape}")

        residual = Y - data
        nt_d, ns_d = Y.shape
        # NLL = 0.5 * sum((Y-D)^2/var) + 0.5 * sum(log(2*pi*var))
        chi2_sum = np.sum(residual**2 * inv_var)
        # Constant term: need sum over all data points
        if sigma_arr.ndim == 0:
            const_sum = nt_d * ns_d * np.log(2 * np.pi * var)
        else:
            # Once per data point, as the sum over t and i says: a per-column
            # sigma counted it once per column, and the loss came out short by
            # (n_times - 1) of those terms (found with issue #1363).
            const_sum = np.sum(np.broadcast_to(np.log(2 * np.pi * var), Y.shape))
        nll = float(0.5 * (chi2_sum + const_sum))

        # d(NLL)/dY = (Y - D) / sigma^2  (the 0.5 * 2 cancel)
        # Vectorized: contract over time and species via einsum.
        dL_dY = residual * inv_var
        grad = self._contract_dL_dY(sens, dL_dY, refused)  # (n_params,)

        return nll, grad

    @property
    def sensitivities(self) -> NDArray[np.float64]:
        """Forward sensitivity data dY/dp.

        Shape ``(n_times, n_species, n_params)`` when sensitivities
        are available, or empty ``(0, 0, 0)`` otherwise.

        The species axis is the **same** axis as :attr:`species` /
        :attr:`species_names`: row *i* is ``d species_names[i]/dp``, so a
        ``dL/dY`` built from :attr:`species` contracts against this tensor
        row-for-row (:meth:`gradient`, :meth:`sse_gradient`, …). On a model
        where an SBML parameter or compartment is an event-assignment target,
        bngsim promotes that symbol to full integrator state but does **not**
        report it as a trajectory column (GH #71); its row is likewise absent
        here, and its derivative is
        ``output_sensitivities("observable:<name>")`` — the loader gives every
        promoted symbol a same-named observable (GH #202).

        Row *i* is the derivative of **the value in column *i***, which for an
        SBML AssignmentRule-target species is the rule's live value, not the
        frozen state slot the integrator carries (GH #221): those rows are the
        chain rule through the assignment, the same quantity
        ``output_sensitivities("species:<name>")`` returns. Where that chain rule
        is unavailable the row is ``NaN`` — never a structural zero, which no
        consumer could tell from a measured one — and the run warns naming the
        species (:attr:`ar_sensitivity_refused`).

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
    def one_sided_sensitivities(self) -> list[tuple[float, dict[str, str]]]:
        """The sensitivity columns that are one-sided derivatives (GH #1822).

        One ``(t, {parameter: direction})`` entry for each switch time at
        which the run sits on a kink: two rate-law switches share the instant
        ``t``, a rate law composes them, and no requested parameter moves one
        of them alone. ``direction`` is ``"increasing"`` or ``"decreasing"``:
        the column of that parameter in :attr:`sensitivities` is the derivative
        for a step of the parameter in that direction, the one that delays its
        switch past the other. In the other direction the derivative is another
        number. Every parameter not listed has an ordinary derivative.

        Empty for a run that sits on no such kink, which is nearly every run,
        and for a result that did not come from one (loaded from a file, sliced
        or stacked). A result cut short by a stop condition, or stitched from
        :meth:`Simulator.compute_all_sensitivities`' chunks, keeps the entries
        of the runs it was made from. A :class:`bngsim.OneSidedSensitivityWarning` is raised for
        each entry when the run returns.
        """
        return [(t, dict(columns)) for t, columns in self._one_sided_sensitivities]

    @property
    def sensitivities_ic(self) -> NDArray[np.float64]:
        """Forward IC sensitivity data dY/dY(0).

        Shape ``(n_times, n_species, n_ic_species)`` when IC
        sensitivities are available, or empty ``(0, 0, 0)`` otherwise.
        Column ``k`` is ``∂Y(t)/∂Y_k(0)`` where the ordering of ``k``
        matches ``sensitivity_ic_species``. The **row** axis is
        :attr:`species_names`, exactly as in :attr:`sensitivities` (GH #202).
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

    # ─── SSA per-reaction event statistics (GH #616) ─────────────────

    @property
    def reaction_firing_counts(self) -> NDArray[np.float64]:
        """Cumulative firings of each reaction at each output time, ``N_r(t)``.

        Shape ``(n_times, n_reactions)`` for a single run and
        ``(n_sims, n_times, n_reactions)`` for a stacked batch, or empty
        ``(0, 0)`` unless the run was made with
        ``Simulator(model, method="ssa", reaction_stats=True)``. Column ``r``
        is labelled by :attr:`reaction_labels`. Integer-valued, held as
        float64 so it subtracts directly from
        :attr:`reaction_propensity_integrals`: ``N_r(t) − ∫₀ᵗ a_r ds`` is the
        compensated counting process of reaction ``r``, whose mean over
        replicates is zero, and which ``bngsim.girsanov`` turns into
        likelihood-ratio scores for the rate constants.
        """
        return self._reaction_firing_counts

    @property
    def reaction_propensity_integrals(self) -> NDArray[np.float64]:
        """Integrated propensity of each reaction at each output time, ``∫₀ᵗ |a_r| ds``.

        Same shape and labelling as :attr:`reaction_firing_counts`. The
        integrand is the propensity magnitude the sampler used to draw the
        reaction — a reaction whose rate law goes negative fires in reverse at
        ``|a_r|`` (GH #110) — so the block describes the process as sampled.
        """
        return self._reaction_propensity_integrals

    @property
    def reaction_labels(self) -> list[str]:
        """Labels for the reaction axis of the per-reaction statistics.

        ``"R<index> (A + B -> C)"``: the ``.net`` file's 1-based reaction index,
        then the reaction written in species names, ``0`` standing for no
        reactant or no product. Empty unless the statistics were recorded.
        """
        return self._reaction_labels

    @property
    def has_reaction_stats(self) -> bool:
        """Whether per-reaction firing counts and propensity integrals were recorded."""
        return self._reaction_firing_counts.size > 0

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

        Every row is the derivative of the value the matching :attr:`species`
        column reports. For an SBML AssignmentRule-target species that is the
        rule's live value: GH #205 routed ``output_sensitivities("species:…")``
        through the rule, and GH #221 put the same row in this tensor, because
        the raw integrated ``yS`` of the frozen slot it used to hold is
        identically zero — not a low-level escape hatch, just a silent zero that
        :meth:`gradient` read as a flat objective.
        """
        return self._sensitivities

    @property
    def ar_sensitivity_refused(self) -> frozenset[str]:
        """Species whose sensitivity row carries ``NaN``: assignment-rule targets,
        and amounts in a compartment whose size's derivative is not to be had.

        An AR-target species' sensitivity row is the chain rule through its
        assignment (GH #221). Where that is unavailable the row is ``NaN`` rather
        than left at the structural zero no consumer could distinguish from a
        measurement, and the run warns naming these species. Two causes: codegen
        declined the rule's own output sensitivity (GH #198 refuses an ``if()``
        conditional or a table function rather than guess — the common one), or
        the target is an amount in a compartment whose size changes and codegen
        declined the output sensitivity of the size's own rule (issue #787).

        Also listed: an amount-valued species (``hasOnlySubstanceUnits``) that is
        no rule's target, in a compartment an assignment rule resizes, where the
        rule's own output sensitivity is not available. Its row is the quotient
        rule through the compartment's size (issue #742), and without that
        derivative it is ``NaN``. :meth:`output_sensitivities` returns that row,
        where for an assignment-rule target it raises.

        Membership means *some* entry of the row is ``NaN``, not necessarily all
        of them. :meth:`gradient` and the built-in objective gradients drop these
        rows wherever ``dL/dY`` is exactly zero, so a fit that never scores these
        species still gets a number; an entry the loss does weight stays ``NaN``.
        :meth:`output_sensitivities` raises for an assignment-rule target among
        them, with the specific reason.

        Empty on every model that resolves: 590 of the 639 AR rows reachable
        across the 215 rr_parity corpus models that run under sensitivities.
        """
        return self._ar_sens_refused

    @property
    def sensitivities_observables(self) -> NDArray[np.float64]:
        """Observable parameter sensitivities ``d observable / dp``.

        Shape ``(n_times, n_observables, n_params)`` when computed, or empty
        ``(0, 0, 0)`` otherwise. The row axis matches
        :attr:`observable_names` (the loader's internal network-rewrite
        observables are dropped here as well, GH #202) and the parameter axis
        matches :attr:`sensitivity_params`.
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
        empty ``(0, 0, 0)`` otherwise. The row axis matches
        :attr:`observable_names` (GH #202) and the IC axis matches
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
        ``n_jac_rhs_evals``, ``n_jac_formations``, ``n_err_test_fails``, ``n_nonlin_iters``,
        ``n_nonlin_conv_fails``, ``n_sens_err_test_fails``,
        ``n_sens_nonlin_conv_fails``, ``n_roundoff_steps``.

        ``n_rhs_evals`` is CVODE's count of the integration's own evaluations.
        ``n_jac_rhs_evals`` counts those spent forming Jacobians by difference
        quotients (issue #1333); it is 0 where every Jacobian is closed-form.
        ``n_roundoff_steps`` counts the steps CVODE took where ``t + h == t``,
        which a run warns of as :class:`~bngsim.RoundoffStepWarning` (issue
        #1837).
        ``n_jac_evals`` counts the linear solver's setups, which can reuse a
        Jacobian; ``n_jac_formations`` counts the Jacobians the solver asked
        for, one that a failed right-hand-side call cut short included.
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
        - ``n_rounded_populations`` (int): molecule counts that were not whole
          numbers when the run (or ``run_until`` leg) started and were rounded
          to one. Event-assigned parameters and compartments and rate-rule
          targets hold continuous values and are never rounded (issue #718).
        - ``first_rounded_species`` (str): name of the first rounded species;
          ``""`` if none.

        All zero/empty on every non-SSA backend.
        """
        return self._ssa_diagnostics

    @property
    def psa_diagnostics(self) -> dict[str, Any]:
        r"""PSA partial-scaling diagnostics dict (GH #15).

        A cheap audit layer for a partial-scaling run: per-reaction, time-
        integrated scaling statistics plus a measured event-rate speedup. These
        are *measured accept/reject signals* for a fixed PSA run, not a
        certification of accuracy. Inactive (``active=False``, all zero/empty) on
        exact SSA and every non-SSA backend.

        For reaction ``r``, the integer leap multiplier is
        ``m_r(N) = max(1, floor(N'_min / N_c))`` (``N'_min`` = smallest
        participating population, ``N_c`` = ``poplevel``), ``a_r`` its unscaled
        propensity, and ``T`` the simulated horizon.

        Keys:

        - ``active`` (bool): whether this run used partial scaling.
        - ``reaction_index`` (list[int]): 1-based reaction indices, parallel to
          every per-reaction list below (matches ``.net`` reaction numbering).
        - ``time`` (float): ``T``, the simulated horizon (integral denominator).
        - ``mbar`` (list[float]): dwell-time-averaged multiplier
          ``m̄_r = (1/T)∫ m_r dt``. A non-scaling reaction reports ``1.0``.
        - ``qexc`` (list[float]): propensity-weighted excess
          ``q̄_r = (1/T)∫ (m_r−1)·a_r dt`` — the correct per-reaction statistic
          for attributing variance inflation (``E[m·a] ≠ E[m]·a(E[N])``), and the
          source term for an affine/monomolecular covariance correction.
        - ``mbar_integral`` / ``qexc_integral`` (list[float]): the raw
          ``∫ m_r dt`` and ``∫ (m_r−1)·a_r dt`` (divide by ``time`` for the
          averages; kept so a transient correction can use the un-normalized
          integral).
        - ``exact_event_integral`` (float): ``∫ Σ_r a_r dt`` (would-be exact
          event count).
        - ``scaled_event_integral`` (float): ``∫ Σ_r a_r/m_r dt`` (actual scaled
          event count).
        - ``speedup`` (float): measured per-path event-rate ratio
          ``ŝ = exact/scaled ≥ 1``, computable from the scaled run alone. It is
          **not** a wall-clock speedup and ignores correction-estimator variance.
        - ``peak_population`` (float): running max over the populations (counts)
          participating in scaling-eligible reactions.
        - ``activation_crossed`` (bool): whether ``peak_population ≥ 2·poplevel``.
          Scaling can only engage past ``2·poplevel``; if this is ``False`` no
          channel ever scaled and the run was pathwise identical to exact SSA.

        To turn ``speedup`` into a cost/precision go/no-go, pair it with the
        observable's variance-inflation ratio ``ρ`` (from a calibration/anchor
        run) via :func:`bngsim.psa_cost_decision`.

        On a stacked batch (:meth:`squeeze`, ``squeeze=True``) the per-run
        statistics are not combined into one: ``active`` is whether any run
        used partial scaling, ``peak_population`` the largest and
        ``activation_crossed`` whether any run crossed, and the per-reaction
        lists, the integrals and ``speedup`` are left empty (NaN). Each run's
        own report is in :attr:`psa_diagnostics_per_run`.
        """
        return self._psa_diagnostics

    @property
    def psa_diagnostics_per_run(self) -> list[dict[str, Any]]:
        """Each run's :attr:`psa_diagnostics`, in the order the runs were stacked.

        One report for a single run; one per row for a stacked batch
        (:meth:`squeeze`), each exactly as that run reported it.
        """
        if self._psa_diagnostics_runs is None:
            return [self._psa_diagnostics]
        return list(self._psa_diagnostics_runs)

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
            from bngsim._error_codes import install
            from bngsim._exceptions import DependencyError

            raise DependencyError(
                "pandas is required for Result.dataframe. "
                "Install with: pip install bngsim[pandas]",
                remedies=[install("pandas", extra="pandas")],
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
        ``seed`` (when present) is stored as ``ds.attrs["seed"]``, and the
        :attr:`stop_condition` the result was cut at as
        ``ds.attrs["stop_condition"]``. :attr:`started_from` and
        :attr:`start_state_sha256`, where recorded, are
        ``ds.attrs["started_from"]`` and ``ds.attrs["start_state_sha256"]``,
        and :attr:`record`, as :meth:`summary` gives it (without the start
        vector), is ``ds.attrs["record"]``, a JSON string.

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
            from bngsim._error_codes import install
            from bngsim._exceptions import DependencyError

            raise DependencyError(
                "xarray is required for Result.to_xarray() and Result.xr. "
                "Install with: pip install 'bngsim[xarray]'",
                remedies=[install("xarray", extra="xarray")],
            ) from None

        if self._species.ndim != 2:
            raise NotSupportedError(
                "to_xarray is defined only for single-simulation results "
                f"(2-D arrays). Got species shape {self._species.shape}; "
                "for batch results, iterate replicates.",
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
        # GH #616 — SSA per-reaction event statistics.
        if self.has_reaction_stats:
            data_vars["reaction_firing_counts"] = (
                ("time", "reaction"),
                self._reaction_firing_counts,
            )
            data_vars["reaction_propensity_integrals"] = (
                ("time", "reaction"),
                self._reaction_propensity_integrals,
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
        if self._reaction_labels:
            coords["reaction"] = self._reaction_labels

        attrs: dict[str, Any] = dict(self.custom_attrs)
        if self._seed is not None:
            attrs["seed"] = int(self._seed)
        if self._stop_condition is not None:
            attrs["stop_condition"] = self._stop_condition
        if self._started_from is not None:
            attrs["started_from"] = self._started_from
        if self.start_state_sha256 is not None:
            attrs["start_state_sha256"] = self.start_state_sha256
        # A string: a netCDF attribute holds no dict (issue #1703).
        attrs["record"] = json.dumps(_record.record(self, start_state=False))

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
            Indexing carries the column names over — ``arr[:, 1:]`` is
            labeled with the columns the slice kept — and any other
            transform (arithmetic, a transpose, a reduction) returns an
            array with no names at all, so a lookup by name on it raises
            rather than answering with a column that has moved
            (issue #561). The names also survive a pickle round trip, so
            an array collected from a worker process arrives labeled
            (issue #629).

        Raises
        ------
        ValueError
            If a selector is unrecognized or names a species not in
            the model, or if it asks for an amount (``"X"``) from a
            result loaded from a file that does not record compartment
            sizes (one written before bngsim saved them, issue #787).
        RuntimeError
            If the result is a stacked batch (3-D species, from
            :meth:`squeeze` or ``run_batch(..., squeeze=True)``); call this
            on each row's Result from ``run_batch(..., squeeze=False)``.

        Warns
        -----
        UserWarning
            When a ``"[X]"`` column is asked of a result loaded from such a
            file: it is the stored value, which is not the concentration
            where a compartment's size changes during a stochastic run, or
            for an amount-valued species an event resizes.

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
            raise NotSupportedError(
                "as_roadrunner is defined only for single-simulation "
                "results (2-D species). Got shape "
                f"{self._species.shape}; for batch results, iterate "
                "rows or call as_roadrunner per-replicate.",
            )

        if selections is None:
            selections = ["time"] + [f"[{s}]" for s in self._species_names]

        n_t = self._time.shape[0]
        sp_idx = {name: i for i, name in enumerate(self._species_names)}
        vf = self._species_volume_factors
        if self._volume_record_missing is not None:
            self._check_missing_volume_record(selections, sp_idx)

        cols: list[NDArray[np.float64]] = []
        for sel in selections:
            if sel == "time":
                cols.append(self._time)
                continue
            if sel.startswith("[") and sel.endswith("]"):
                inner = sel[1:-1]
                if inner not in sp_idx:
                    raise UsageError(
                        self._invalid_selection_message(sel),
                        code="output.not_found",
                        details={"name": sel},
                    )
                idx = sp_idx[inner]
                col = self._species[:, idx]
                cf = self._varvol_conc_factor
                if cf is not None and idx in cf:
                    # GH #131: the raw column is the conserved count amount/V_static
                    # (what an older bngsim left under SSA, and under ODE for an
                    # hOSU=true event-resized species, before issue #1368), so the
                    # live concentration amount/V_live = raw · (V_static/V_live) is
                    # reported by multiplying the stored per-sample factor.
                    col = col * cf[idx]
                cols.append(col)
                continue
            if sel in sp_idx:
                idx = sp_idx[sel]
                col = self._species[:, idx]
                lv = self._varvol_live_vol
                af = self._varvol_amount_factor
                if af is not None and idx in af:
                    # GH #131: this column holds amount/V_live(t), and the size it
                    # is multiplied by is stored per sample: the live volume may
                    # live in an OBSERVABLE column (an hOSU=false event-resized
                    # compartment, hidden from species output per #71), and a
                    # sample where the size is 0 holds amount/V_static, so it is
                    # multiplied by V_static there.
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
            raise UsageError(
                self._invalid_selection_message(sel),
                code="output.not_found",
                details={"name": sel},
            )

        data = np.empty((n_t, len(selections)), dtype=np.float64)
        for j, c in enumerate(cols):
            data[:, j] = c
        return NamedArray(data, selections)

    def _check_missing_volume_record(self, selections: list[str], sp_idx: dict[str, int]) -> None:
        """Refuse an amount, and warn on a concentration, from a file without sizes.

        A file written before issue #787 holds the stored species values but not
        the compartment sizes or the per-sample volume factors, and nothing in it
        says whether a size was 1. An amount is the stored value times the size,
        so it is refused rather than reported at size 1: a species of 4 in a
        compartment of size 2 read 4 for 8. A concentration is the stored value
        except where a size changes under SSA or PSA, or an event resizes the
        compartment of an amount-valued species, so it is reported with a warning.
        """
        writer = self._volume_record_missing
        amounts = [sel for sel in selections if sel != "time" and sel in sp_idx]
        if amounts:
            name = amounts[0]
            raise UsageError(
                f"as_roadrunner: cannot report the amount {name!r}. This result was loaded "
                f"from a file written by {writer}, which did not save compartment "
                "sizes, and an amount is the stored concentration times its compartment's "
                f"size. Select '[{name}]' and multiply it by the size (where every size is "
                "1, as in a .net model, the two are equal), or run the model again and save "
                "the result with this version, which records the sizes."
            )
        if any(
            sel.startswith("[") and sel.endswith("]") and sel[1:-1] in sp_idx for sel in selections
        ):
            warnings.warn(
                f"as_roadrunner: this result was loaded from a file written by {writer}, "
                "which did not save the volume factors, so each '[X]' column is "
                "the stored species value. That is the concentration unless a compartment's "
                "size changes during a stochastic run, or an event resizes the compartment "
                "of an amount-valued species, where it is the amount over the starting size. "
                "Run the model again and save the result with this version to record them.",
                stacklevel=3,
            )

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
            ``sensitivity_ic_species``, ``solver_stats``, ``seed``,
            ``started_from`` and ``start_state_sha256`` (where the run started,
            issue #1701; see :attr:`started_from`), and ``record``
            (:attr:`record`, issue #1703), without ``run.start_state``, the
            start vector, which :meth:`save` keeps.

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
            "started_from": self._started_from,
            "start_state_sha256": self.start_state_sha256,
            "record": _record.record(self, start_state=False),
        }

    def digest(
        self,
        *,
        species: bool = False,
        max_columns: int = 20,
        sensitivities: bool = False,
        max_parameters: int = 5,
    ) -> dict[str, Any]:
        """A small, deterministic description of this result's values, for a
        caller that cannot look at a plot (issue #1704).

        A dict, JSON-ready (``json.dumps(d, allow_nan=False)`` writes it: every
        number is finite or ``None``); ``str()`` of it, or ``print()``, is its
        plain-text form, one line per column. It gives numbers and counts, and
        never a word such as "oscillates" or "bistable": a reader draws that
        from the counts.

        Parameters
        ----------
        species : bool
            Also list the species. The observables are listed always, or the
            species alone where the result has no observables.
        max_columns : int
            How many columns of each kind to list, at most (default 20). Past
            it the digest lists the columns with the largest range, any column
            holding a non-finite value first, in column order, and gives how
            many it left out.
        sensitivities : bool
            Also say, for each listed column of one run, which parameters and
            which initial amounts move it most at the last time (issue #1883;
            see the ``sensitivity`` keys below). Off by default.
        max_parameters : int
            How many parameters, and how many initial amounts, each column
            lists with ``sensitivities=True`` (default 5).

        Returns
        -------
        dict
            For the run: ``digest_version`` (1; a key removed or changed in
            meaning changes it, a key added does not), ``method``,
            ``stochastic``, ``n_runs`` (a stack's number of runs, else
            ``None``), ``t_start``, ``t_end``, ``n_points``,
            ``n_sensitivity_params``, ``n_sensitivity_ic_species``,
            ``n_notices``, ``notices`` (the first 20 of :attr:`notices`, each
            message cut at 500 characters), ``kinds``, the kinds listed, and
            ``rtol`` (floored at 1e-10) and ``atol``, the run's ceiling on any
            species' atol (its scalar, or the largest per-species value where
            the caller gave a vector, ``"auto"`` or a tracking atol), whence
            each column's tolerance (see Notes); ``None`` where the record
            gives none.
            Then for each kind (``"observables"``, ``"species"``)
            ``{"n_columns", "n_left_out", "columns"}``, where each column is a
            dict with its ``name`` and:

            - of one deterministic run: ``first``, ``last``, ``min`` and
              ``t_min``, ``max`` and ``t_max``, ``monotone``,
              ``n_interior_extrema``, ``tail_from`` (the last sample at or
              before the start of the last tenth of the window, so later than
              a tenth back where the grid is coarse), ``tail_change`` (how far
              any value from ``tail_from`` on is from the last value, as a
              share of the column's range) and ``settled``; ``t_min`` and
              ``t_max`` are the first times the extremes are reached;
            - of one stochastic run, or a run whose method or tolerances the
              result does not record: ``first``, ``last``, ``min`` and ``max``
              alone, as its shape is noise or cannot be judged;
            - of a stack of runs (:meth:`squeeze`, ``run_batch(squeeze=True)``):
              statistics across the runs at the last time, ``last_mean``,
              ``last_std`` (the sample standard deviation; ``None`` of one run),
              ``last_min``, ``last_q05``, ``last_q25``, ``last_median``,
              ``last_q75``, ``last_q95``, ``last_max`` and ``last_n_finite``
              (the runs whose last value is finite, which the statistics are
              over), with ``min`` and ``max`` over every run and time;
            - always, the flags ``non_finite`` (with ``t_non_finite``, the first
              time a value is NaN or infinite), ``negative`` and ``constant``
              (the column never moves);
            - with ``sensitivities=True``, ``sensitivity``: ``None`` where the
              result holds neither its parameter nor its initial-amount block
              for the column's kind, else ``y`` (its
              value at the last time), ``resolved`` (whether ``|y|`` exceeds
              the column's absolute tolerance, a species' own as applied and an
              observable's by the run's rule, so the run told it from 0;
              ``None`` where the tolerances are not recorded), ``parameters`` and
              ``initial_amounts``, each at most ``max_parameters`` entries,
              largest ``|dy_dlnp|`` first, of ``name``, ``dy_dp`` (the
              derivative as :attr:`sensitivities` holds it), ``dy_dlnp``
              (``p·dy_dp``, with ``p`` the parameter's value as the run started
              or the species' amount the run started from, before any event at
              the first time), ``dlny_dlnp`` (the
              scaled sensitivity ``(p/y)·dy_dp``, ``None`` unless ``resolved``)
              and ``one_sided`` (the side a one-sided column is taken from at
              the last time, by the kinks of :attr:`one_sided_sensitivities`
              at or before it; ``"mixed"`` where two took opposite sides; else
              ``None``); and ``n_parameters_left_out``,
              ``n_parameters_non_finite`` (a derivative, or its product with
              ``p``, that is not finite) and the same two of
              ``initial_amounts``. A block the result does not hold for the
              column's kind (an older file) gives ``None`` for its three keys.

            With ``sensitivities=True`` the run block has ``sensitivity`` too:
            ``t`` (the last time), ``max_parameters``, ``reason`` (why there is
            none, for a run without sensitivities or a stack, else ``None``),
            ``n_parameters``, ``n_initial_amounts``, and the parameters, and
            initial amounts, that are not ranked because their value is 0
            (``at_zero``, ``initial_at_zero``) or not recorded (``unvalued``,
            ``initial_unvalued``), each list at most 20 long with its count
            (``n_at_zero`` …), and ``one_sided``, ``{parameter: side}`` at the
            last time, at most 20, with ``n_one_sided``.

            ``min`` and ``max`` are over the finite values; a ``first`` or
            ``last`` that is not finite is ``None``, as are the readings of
            shape of a column holding a non-finite value.

        Notes
        -----
        What counts is decided by the tolerance of the steady-state code's
        ``criterion="scaled"`` (issue #1001), not a threshold of the
        digest's own: a move at a value ``y`` is within it when at most
        ``rtol·|y| + min(atol, rtol·largest)``, with the run's ``rtol`` (at
        least 1e-10) and absolute tolerance (a species' own as applied; an
        observable, whose species the record does not name, what the run's
        rule gives a species of its size: the smaller of the run's ``atol``
        ceiling and ``rtol`` times its largest magnitude, not under 1e-6 of
        ``largest``), and
        ``largest`` the largest magnitude of any column of that kind over the
        run. A column is ``settled`` when no value from ``tail_from`` on is
        further than that from the last value. That is a reading of the window the run had:
        a column still moving slower than its tolerance per tenth of the
        window reads as settled, and a longer run, or
        :meth:`Simulator.steady_state`, tells. An interior extremum is an
        extreme the column reaches by more than the tolerance and leaves by
        more than the tolerance at it, so an integrator's wobble about a value
        it has reached is no turn, and ``monotone`` is no turn. ``negative``
        is a minimum below ``-min(atol, rtol·largest)``, and ``constant`` a
        range within the tolerance. Of a stochastic run, or one whose method
        or tolerances are not recorded (a stack whose rows ran at different
        ones), ``negative`` is any value below 0 and ``constant`` a column
        whose values are all equal.

        Nothing is computed until this is called, and the readings of shape
        are made for the listed columns alone.

        Of sensitivities: raw ``dY/dp`` would rank parameters by their units (a
        rate constant in 1/s against one in 1/(M·s)), so the rank is by
        ``|dY/dln p| = |p·dY/dp|``, which within one column is the order of the
        scaled sensitivity ``(p/y)·dY/dp`` and needs no division by ``y``. A
        parameter at 0 has no ``dY/dln p`` however much it matters, and one
        whose value the record does not give (a hidden internal parameter, or
        a result with no record of its run's parameters, such as
        :meth:`Simulator.compute_all_sensitivities`' or a file written before
        the record held them) cannot be scaled: they are listed apart, not
        ranked.
        A derivative that is not finite (an assignment-rule species whose row
        is refused, :attr:`ar_sensitivity_refused`, or an observable summing
        one) is counted and not ranked. The parameter axis is the total
        derivative, as :attr:`sensitivities` holds it.

        Examples
        --------
        >>> d = result.digest()
        >>> d["observables"]["columns"][0]["n_interior_extrema"]
        0
        >>> print(d)  # the plain-text form, one line per column
        ode run, t 0 to 10, 101 points, rtol 1e-08, atol 1e-08
        notices: none
        observables: 2 of 2
          A: first 100, last 33.3702; min 33.3702 at t=10, max 100 at t=0; monotone; ...
        """
        from bngsim import _digest

        return _digest.digest(
            self,
            species=species,
            max_columns=max_columns,
            sensitivities=sensitivities,
            max_parameters=max_parameters,
        )

    def check(
        self,
        checks: str | list[str] | None = None,
        *,
        budget: float | None = None,
        replicates: int = 10,
    ) -> dict[str, Any]:
        """Make this run again with one setting changed, and say whether the
        answer moved (issue #1707).

        Each check is a :meth:`rerun` from the state this run started from, on
        a fresh model: the session's model is neither read nor changed. A
        disagreement is reported, not raised.

        Parameters
        ----------
        checks : None, "all", str or list of str
            Which checks to run. ``None`` (default) runs the cheap ones that
            catch the common case: ``"tolerance"`` and ``"grid"`` of an ODE
            run (and ``"sensitivity"`` of one that computed forward
            sensitivities), ``"replicates"`` of a stochastic one. ``"all"``
            runs every check that applies. The checks:

            - ``"tolerance"``: the run at ``rtol/10`` and ``atol/10`` (and at
              ``/100``, where the budget allows); passes when no species moves
              by more than 1e-3 of its own peak, past the atol the run applied
              to it. How far each moves is the run's own *drift*, which the
              other checks are judged by.
            - ``"grid"``: the run at 8x the output points, read at its own
              times; an output point is a stop the integrator lands on. Not
              for a run that stopped at a steady state, whose other checks
              compare the state it stopped at.
            - ``"jacobian"``: the run on a difference-quotient Jacobian.
            - ``"linear_solver"``: the run on the other linear solver, dense
              for a run on KLU and KLU for a dense one; ``skip`` with the
              reason where KLU cannot run it (a build without KLU, a model
              with no sparse Jacobian for it, a JAX Jacobian).
            - ``"backend"``: compiled against interpreted, where a C compiler
              is present.
            - ``"second_engine"``: the model run in libRoadRunner, where it is
              installed, for an SBML or Antimony model run from its initial
              state at the file's parameters. The other checks all read the
              model as bngsim loaded it; only this one does not.
            - ``"sensitivity"``: the forward sensitivities of the tolerance
              check's run, against this run's (issue #1893). Each series
              (one species, one parameter or initial amount, over time) is
              judged on its own peak, past the base absolute tolerance the
              solver set for it (``atol_i·max(|x_i(0)|, 1)/|θ_j|``; the
              solver's own roundoff floor can raise it during the run), and passes when
              it moves by at most 1e-3 of that peak. A series within 100 of
              those tolerances of zero is not resolved by the run, named in
              the verdict's ``not_resolved``, and judged on its column's
              largest peak. A series past the bound whose move from
              ``rtol/10`` to ``rtol/100`` is at least half its move at
              ``rtol/10`` is taken to be held by a floor and not the
              tolerance: it is named (``n_floor_held``), not failed, its move
              is read on its column's largest peak, and the quotient check
              allows it ten times that move. An error in such a series that does not
              shrink with the tolerance, and is small beside its column's
              peak, passes both checks; the name is the only sign of it.
            - ``"sensitivity_fd"``: the forward sensitivities against central
              difference quotients, each parameter (through ``set_param``, so
              an initial amount it sets follows) and initial amount moved by
              1e-3 and 2e-3 of itself, of runs at ``rtol/100`` clamped to
              1e-14 to 1e-8 (never looser than the run), atol scaled alike,
              with the sensitivities off. The quotient's own error is read
              cell by cell from its two steps: their Richardson difference,
              and what doubling the step leaves of its forward half less its
              backward, which widens it near a switch or event time the
              parameter moves; a cell where that is 5% of a series' peak is
              not judged. A series it resolves to no better than 5% of its
              peak, everywhere or over a stretch where its Richardson
              difference reaches that share (a long oscillation whose phase
              the step moves), is named in ``not_resolved`` and not judged
              there;
              ``allowance`` is the largest share of a peak it allows where
              it judges. A value at 0, and a derived
              parameter, are not judged (``not_judged``). It costs four runs
              per column and one more, within ``budget``; the columns the
              budget does not reach are named in ``timed_out``.
            - ``"sensitivity_second_engine"``: the forward sensitivities
              against AMICI's, where AMICI is installed (it is no dependency
              or extra of bngsim: it compiles C++ per model), for a run that
              ``"second_engine"`` could make again from the file. AMICI reads
              the file, so this is the one sensitivity check that does not
              read the model as bngsim loaded it. A model with an event, or
              with a switch in its math (a piecewise, a comparison, ``floor``,
              ``min``, ``max``, ...), is skipped: AMICI reads some of those
              differently from bngsim. AMICI runs at the run's tolerance and
              at a tenth of it; each parameter column AMICI has is judged
              against the tighter run series by series as a path is, on the
              larger of the two engines' peaks, with the series' move at the
              tighter tolerance as its drift (none where the series is not
              converged) or AMICI's own move between its two runs, the larger
              (a series AMICI moves past 1e-3 of its peak on has no
              reference and is not judged: ``no_reference``), and the
              states are compared alike; an initial amount is not judged
              (``not_judged``), as AMICI gives sensitivities to parameters
              only. A diff is judged again a decade tighter (bngsim at
              ``rtol/10`` against AMICI at ``rtol/100``):
              ``confirmed_tighter`` is True where it holds as large with
              both engines converged there (the reading's), False where it
              goes or halves (the tolerance's: ``diff_not_converged``), and
              None where that does not settle it. AMICI's build is cached in
              ``~/.cache/bngsim/amici`` (or ``$BNGSIM_AMICI_CACHE_DIR``).
            - ``"replicates"``: a stochastic run made again at ``replicates``
              other seeds; for each observable at the last time, the mean
              over them, its standard error, and how many standard errors this
              run is from it. It reports, and judges nothing: it passes when
              its reruns complete.
        budget : float, optional
            Seconds of wall time for all the bngsim reruns (a run's own
            recorded ``timeout`` still applies where it is shorter). A check
            that does not fit reports ``"timeout"``, none is dropped, and the
            report is then not ``ok``. Loading the model, running
            libRoadRunner, and building and running AMICI are not bounded.
        replicates : int
            The reruns of the replicates check (default 10, at least 2).

        Returns
        -------
        dict
            JSON-ready; ``str()`` of it is a plain-text form. ``check_version``
            (1), ``ok`` (a check passed, and none moved the answer past its
            bound, raised or ran out of budget), ``method``, ``stochastic``,
            ``rtol``, ``atol``, ``drift`` (the largest move of a species at
            the tighter tolerance), ``bound`` (1e-3) and ``converged``
            (whether every species is under it; a report on a run that is
            not converged is never ``ok``, whichever checks ran),
            ``n_not_resolved`` and ``not_resolved`` (the first 20 species
            within 100 times their atol of zero, which a check judges on
            the run's whole scale alone),
            ``checks``
            (in the order run, each with ``name``, ``status``, ``value``,
            ``worst_column``, ``changed`` (the setting changed) and
            ``note``), ``n_pass``, ``n_diff`` and
            ``next``, what to do. A status is ``"pass"``, ``"diff"`` (the
            answer moved past the bound), ``"skip"`` (with the reason),
            ``"unsupported"``, ``"exception"`` (the changed run raised),
            ``"no_base"`` (no species is converged at the run's own
            tolerance, so no path is judged) or ``"timeout"``. A path's
            verdict also gives ``margin`` (its largest difference over the
            allowance; at most 1 passes), ``n_not_judged`` and ``n_trace``.
            A sensitivity check's verdict names its worst series as
            ``"dA/dk"`` and also gives ``columns`` (each column's worst
            species and value, the worst columns first, at most 20),
            ``n_columns``, ``n_not_resolved``, ``not_resolved`` (series
            names) and ``n_refused`` (series the run reports as not finite
            by design, which are not judged); ``"sensitivity"`` also gives
            ``n_series`` and ``n_floor_held``, and ``"sensitivity_fd"``
            ``margin``, ``n_judged``, ``not_judged`` (columns, with the
            reason), ``timed_out``, ``allowance``, ``n_not_converged``
            (series the tolerance check finds not converged, judged on the
            tolerance alone) and ``diff_not_converged`` (a diff in those
            series alone, which ``next`` reads as the tolerance's);
            ``"sensitivity_second_engine"`` gives those of ``margin``,
            ``n_judged``, ``not_judged``, ``n_not_converged`` and
            ``diff_not_converged`` too, ``not_converged``,
            ``no_reference`` and ``floor_held`` (series names),
            ``confirmed_tighter``, and ``state_margin`` and
            ``state_diff`` (the states against AMICI's),
            ``species_not_compared`` (with the reason) and ``engine``
            (AMICI's ``version``, its ``commit`` where the install records
            one, and ``pinned``: whether that is the commit bngsim's AMICI
            parity suite pins).

        Raises
        ------
        UsageError
            On a stack of runs, or an argument it does not take.

        Notes
        -----
        Every comparison is made species by species, each on its own scale,
        so a species far smaller than the largest is judged as carefully. A
        path passes when, in every species converged at the run's tolerance
        (its move under 1e-3), its largest difference from this run is at
        most ``max(min(1e3·(atol_j + rtol·peak_j), max(0.01·peak_j,
        atol_j)), 100·atol_j, 10·d_j)``: a thousand times what the
        tolerances allow, but never more than 1% of the species' peak
        (unless it is within 100 atol of zero), or ten times how far it moves
        itself at the tighter tolerance (``d_j``). A species whose peak is
        within 100 times its atol, where that atol is larger than
        ``rtol·peak_j`` (the absolute tolerance, not the relative one, holds
        it), is not resolved by the run, and is named in ``not_resolved``
        (and ``next``); a path judges it on the run's whole scale alone, as a
        species at the largest peak and atol, so its noise near zero is no
        finding and a departure as large as the run's species is one. The
        factors and the rule are the developer-side differential harness's
        (issues #1257, #1887). A ``"diff"``
        between two bngsim paths on a model bngsim accepts is a bug worth
        reporting, with ``json.dumps(report)``.
        """
        from bngsim import _check

        return _check.check(self, checks=checks, budget=budget, replicates=replicates)

    def _save_npz(self, path: Path) -> None:
        """:meth:`save` to a ``.npz`` file, with NumPy alone (issue #1704)."""
        logger.info("Saving result to %s", path)
        arrays: dict[str, NDArray[Any]] = {
            "time": np.asarray(self._time),
            "species": np.asarray(self._species),
            "observables": np.asarray(self._observables),
            "expressions": np.asarray(self._expressions),
            "species_names": np.array(list(self._species_names), dtype=str),
            "observable_names": np.array(list(self._observable_names), dtype=str),
            "expression_names": np.array(list(self._expression_names), dtype=str),
        }
        blocks = (
            ("sensitivities_species", "sensitivities"),
            ("sensitivities_observables", "sensitivities_observables"),
            ("sensitivities_expressions", "sensitivities_expressions"),
            ("sensitivities_species_ic", "sensitivities_ic"),
            ("sensitivities_observables_ic", "sensitivities_observables_ic"),
            ("sensitivities_expressions_ic", "sensitivities_expressions_ic"),
        )
        for key, attr in blocks:
            has = "has_sensitivities" if attr == "sensitivities" else f"has_{attr}"
            if getattr(self, has):
                arrays[key] = np.asarray(getattr(self, attr))
        for key, names in (
            ("sensitivity_params", self._sensitivity_params),
            ("sensitivity_ic_species", self._sensitivity_ic_species),
        ):
            if names:
                arrays[key] = np.array(list(names), dtype=str)
        arrays["record"] = np.array(json.dumps(self.record))
        # A file object, so NumPy adds no second ".npz" to a path ending ".NPZ".
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    # ─── HDF5 save / load ──────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save result to an HDF5 file, or to a NumPy ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Output file path. A path ending ``.npz`` is written with NumPy
            alone (issue #1704); any other is HDF5 (e.g. ``"results.h5"``),
            which needs h5py. A path ending ``.csv`` or ``.tsv`` is refused:
            :meth:`to_csv` writes those, one kind of column at a time.

        Raises
        ------
        DependencyError
            If h5py is not installed and *path* is not ``.npz``. Also an
            :class:`ImportError`.
        UsageError
            If *path* ends ``.csv`` or ``.tsv``.

        Notes
        -----
        HDF5 keeps everything: :meth:`Result.load` reads it back as a Result
        whose record, notices, sensitivities and volume record are as saved.
        A ``.npz`` file is for ``numpy.load(path)``, with no pickled object in
        it (``allow_pickle=False`` reads it): ``time``, ``species``,
        ``observables`` and ``expressions`` (a stack's with a leading run
        axis), ``species_names``, ``observable_names`` and
        ``expression_names``, each sensitivity block the result holds
        (``sensitivities_species``, ``sensitivities_observables``,
        ``sensitivities_expressions`` and their ``_ic`` forms) with
        ``sensitivity_params`` and ``sensitivity_ic_species``, and ``record``,
        :attr:`record` and :attr:`notices` as one JSON string
        (``json.loads(str(f["record"]))``). :meth:`Result.load` does not read
        it back.
        """
        suffix = Path(path).suffix.lower()
        if suffix == ".npz":
            self._save_npz(Path(path))
            return
        if suffix in (".csv", ".tsv"):
            from bngsim._error_codes import use_call

            raise UsageError(
                f"Result.save({str(path)!r}) writes HDF5 or .npz, not a delimited "
                f"text file. Result.to_csv(path, kind='observables' or 'species') "
                f"writes {suffix}, one kind of column at a time; "
                f"Result.save('....npz') keeps every array with NumPy alone.",
                remedies=[use_call("Result.to_csv", str(path))],
            )
        try:
            import h5py
        except ImportError:
            from bngsim._error_codes import install
            from bngsim._exceptions import DependencyError

            raise DependencyError(
                "h5py is required for Result.save(). Install with: pip install bngsim[hdf5]",
                remedies=[install("h5py", extra="hdf5")],
            ) from None

        path = Path(path)
        logger.info("Saving result to %s", path)

        from bngsim._rerun import source_text
        from bngsim._version import __version__ as _bngsim_version

        # The model's text, where it was loaded from a string, which rerun()
        # builds it from: kept as the bytes it hashes to, a str as UTF-8, so
        # text of any encoding is written whole. Made before the file is opened.
        held_text = source_text(self)
        text_bytes = (
            None
            if held_text is None
            else np.frombuffer(
                held_text.encode("utf-8") if isinstance(held_text, str) else bytes(held_text),
                dtype=np.uint8,
            )
        )

        with h5py.File(path, "w") as f:
            # Version for forward compatibility. v2 (GH #196) adds the optional
            # sensitivity blocks below; a v1 reader simply won't find them and a
            # v2 file without sensitivities is byte-equivalent to a v1 one. v3
            # (issue #787) adds the volume record, and Result.load reads a file
            # below v3 as one that does not say what the sizes were.
            f.attrs["bngsim_version"] = _bngsim_version
            f.attrs["format_version"] = 3

            # Stochastic seed actually used (omitted for ODE results).
            if self._seed is not None:
                f.attrs["seed"] = int(self._seed)
            # The stop condition the result was cut at (issue #1369), written
            # only when one was met. Absent loads as None, as any older file
            # does, so the format version stands.
            if self._stop_condition is not None:
                f.attrs["stop_condition"] = self._stop_condition
            # Where the run started (issue #1701), written where recorded. Absent
            # loads as None, as any older file does, so the format version stands.
            if self._started_from is not None:
                f.attrs["started_from"] = self._started_from
            if self.start_state_sha256 is not None:
                f.attrs["start_state_sha256"] = self.start_state_sha256
            # The record (issue #1703): its parts and notices as JSON, in a
            # dataset, since an attribute holds at most 64 KiB and a large
            # network's columns and parameters pass that; and the start vector,
            # which summary() leaves out. Each written only where recorded, so
            # the format version stands.
            if self._start_state is not None:
                f.create_dataset("start_state", data=np.asarray(self._start_state, dtype="<f8"))
            if text_bytes is not None:
                ds = f.create_dataset("model_text", data=text_bytes)
                ds.attrs["type"] = "str" if isinstance(held_text, str) else "bytes"
            parts = _record.frozen_parts(self)
            if parts is not None or self._notices:
                f.create_dataset(
                    "record",
                    data=json.dumps(
                        {
                            "parts": parts,
                            "notices": [n.as_dict() for n in self._notices],
                            "rows_differ": (
                                None
                                if self._record_differs is None
                                else list(self._record_differs)
                            ),
                        }
                    ),
                    dtype=h5py.string_dtype(),
                )

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
            # GH #616 — SSA per-reaction event statistics, written only when recorded.
            if self._reaction_firing_counts.size > 0:
                f.create_dataset("reaction_firing_counts", data=self._reaction_firing_counts)
                f.create_dataset(
                    "reaction_propensity_integrals", data=self._reaction_propensity_integrals
                )
                f.create_dataset("reaction_labels", data=self._reaction_labels, dtype=dt)
            if self._sensitivity_ic_species:
                f.create_dataset(
                    "sensitivity_ic_species", data=self._sensitivity_ic_species, dtype=dt
                )
            # GH #221: which species rows above are NaN by refusal rather than by
            # a failed solve. Written only when there are any, so an ordinary
            # result's file is unchanged. Without it a reloaded result cannot
            # tell the two apart, and `gradient` would lose the exemption that
            # lets a loss which never weights those rows still return a number.
            if self._ar_sens_refused:
                f.create_dataset(
                    "ar_sensitivity_refused", data=sorted(self._ar_sens_refused), dtype=dt
                )
            # Issue #787: the assignment-rule species whose output sensitivity is
            # refused by name because a time-varying size rescales the value.
            # Without it a reloaded result handed back the NaN row as an answer.
            if self._ar_sens_blocked:
                f.create_dataset(
                    "ar_sensitivity_blocked", data=sorted(self._ar_sens_blocked), dtype=dt
                )
            # GH #155: the ∂x(0)/∂θ rows the parameter axis already carries, one
            # (species, parameter, coefficient) entry per element. Without it a
            # reloaded result reported None, "not recorded", and a caller adding
            # its own initial-condition term could not tell which parameter
            # columns already hold one. Written whenever it is recorded, empty
            # included, because {} ("nothing seeds") is a different answer.
            if self._ic_sensitivity_seed is not None:
                seed_rows = [
                    (sp, p, c)
                    for sp, row in self._ic_sensitivity_seed.items()
                    for p, c in row.items()
                ]
                f.create_dataset(
                    "ic_sensitivity_seed_species", data=[e[0] for e in seed_rows], dtype=dt
                )
                f.create_dataset(
                    "ic_sensitivity_seed_params", data=[e[1] for e in seed_rows], dtype=dt
                )
                f.create_dataset(
                    "ic_sensitivity_seed",
                    data=np.array([e[2] for e in seed_rows], dtype=np.float64),
                )

            # Volume record (issue #787). as_roadrunner turns a stored species value
            # into an amount with its compartment's size, and into a concentration
            # with a per-sample factor where the size changes, and neither is in the
            # arrays above: without them a species of 4 in a compartment of size 2
            # loaded back with an amount of 4 for 8. The factor maps are keyed by
            # species column; each per-sample array is one column of a
            # (time, entry) block, beside the column indices it belongs to. A
            # squeezed batch holds one record per row, so its sizes are a
            # (sim, species) block and its factors (sim, time, entry). Each is
            # written only when the result carries it, so one that carries none
            # (built from raw arrays) loads back the same.
            if self._species_volume_factors is not None:
                f.create_dataset(
                    "species_volume_factors",
                    data=np.asarray(self._species_volume_factors, dtype=np.float64),
                )
            if self._varvol_live_vol:
                f.create_dataset(
                    "varvol_live_vol",
                    data=np.array(sorted(self._varvol_live_vol.items()), dtype=np.int64),
                )
            for key, fmap in (
                ("varvol_conc_factor", self._varvol_conc_factor),
                ("varvol_amount_factor", self._varvol_amount_factor),
            ):
                if fmap:
                    cols = sorted(fmap)
                    f.create_dataset(f"{key}_species", data=np.array(cols, dtype=np.int64))
                    f.create_dataset(
                        key,
                        data=np.stack(
                            [np.asarray(fmap[j], dtype=np.float64) for j in cols], axis=-1
                        ),
                    )
            # A result loaded from a file without the record keeps saying so when
            # it is saved again; otherwise the new file would read as one whose
            # sizes are all 1.
            if self._volume_record_missing is not None:
                f.attrs["volume_record_missing"] = self._volume_record_missing

            # Solver stats
            stats_grp = f.create_group("solver_stats")
            for key, val in self._solver_stats.items():
                stats_grp.attrs[key] = val

            # SSA boundary diagnostics (GH #110), and the PSA diagnostics (GH #15)
            # of a run that used partial scaling; every other result carries the
            # inactive PSA dict, which load rebuilds. Without them a loaded SSA
            # result reported 0 rounded populations for 1 and the propensity
            # backend "unknown" for "cc", and a PSA run read as inactive. The
            # per-reaction PSA lists are datasets, since an attribute holds at
            # most 64 KiB, and the scalars attributes.
            ssa_grp = f.create_group("ssa_diagnostics")
            for key, val in self._ssa_diagnostics.items():
                ssa_grp.attrs[key] = val

            def _write_psa(grp: Any, diag: dict[str, Any]) -> None:
                for key, val in diag.items():
                    if isinstance(val, list):
                        grp.create_dataset(key, data=np.asarray(val))
                    else:
                        grp.attrs[key] = val

            if self._psa_diagnostics.get("active"):
                _write_psa(f.create_group("psa_diagnostics"), self._psa_diagnostics)
            # A stacked batch's runs, each under its index, when any used PSA.
            runs = self._psa_diagnostics_runs
            if runs is not None and any(run.get("active") for run in runs):
                runs_grp = f.create_group("psa_diagnostics_runs")
                for k, run in enumerate(runs):
                    _write_psa(runs_grp.create_group(str(k)), run)

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

        Notes
        -----
        A file written before bngsim saved compartment sizes (issue #787) loads,
        but does not say what the sizes were. :meth:`as_roadrunner` on it refuses
        a bare-id amount selector and warns on a ``"[X]"`` one; the arrays
        themselves are as they were saved.
        """
        if Path(path).suffix.lower() == ".npz":
            from bngsim._error_codes import use_call

            raise UsageError(
                f"Result.load({str(path)!r}) reads the HDF5 files Result.save writes; "
                f"a .npz file is for numpy.load(path), which gives its arrays, "
                f"names and record.",
                remedies=[use_call("numpy.load", str(path))],
            )
        try:
            import h5py
        except ImportError:
            from bngsim._error_codes import install
            from bngsim._exceptions import DependencyError

            raise DependencyError(
                "h5py is required for Result.load(). Install with: pip install bngsim[hdf5]",
                remedies=[install("h5py", extra="hdf5")],
            ) from None

        path = Path(path)
        if not path.exists():
            raise MissingFileError(f"Result file not found: {path}", path)

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
            ar_sens_refused = _names("ar_sensitivity_refused") or []
            ar_sens_blocked = _names("ar_sensitivity_blocked") or []
            # GH #616 — absent datasets round-trip to the empty (0, 0) block.
            reaction_firing_counts = _sens("reaction_firing_counts")
            reaction_propensity_integrals = _sens("reaction_propensity_integrals")
            reaction_labels = _names("reaction_labels")

            # Volume record (issue #787; format_version >= 3). In a v3 file an
            # absent dataset means the result carried none. A file below v3 never
            # wrote them, so it is marked as not saying, and as_roadrunner refuses
            # the amounts that need them.
            volume_factors = (
                np.asarray(f["species_volume_factors"][()], dtype=np.float64).tolist()
                if "species_volume_factors" in f
                else None
            )
            live_vol = (
                {int(j): int(k) for j, k in np.array(f["varvol_live_vol"])}
                if "varvol_live_vol" in f
                else None
            )

            def _factor_map(key: str) -> dict[int, NDArray[np.float64]] | None:
                if key not in f:
                    return None
                block = np.array(f[key], dtype=np.float64)
                cols = np.array(f[f"{key}_species"])
                return {int(j): block[..., c].copy() for c, j in enumerate(cols)}

            conc_factor = _factor_map("varvol_conc_factor")
            amount_factor = _factor_map("varvol_amount_factor")

            def _text(value: Any) -> str:
                return value.decode() if isinstance(value, bytes) else str(value)

            volume_record_missing: str | None = None
            if int(f.attrs.get("format_version", 1)) < 3:
                writer = f.attrs.get("bngsim_version")
                volume_record_missing = (
                    f"bngsim {_text(writer)}" if writer is not None else "an earlier bngsim"
                )
            elif "volume_record_missing" in f.attrs:
                volume_record_missing = _text(f.attrs["volume_record_missing"])
            stop_condition = (
                _text(f.attrs["stop_condition"]) if "stop_condition" in f.attrs else None
            )
            started_from = _text(f.attrs["started_from"]) if "started_from" in f.attrs else None
            start_state_sha256 = (
                _text(f.attrs["start_state_sha256"]) if "start_state_sha256" in f.attrs else None
            )
            # Issue #1703: the record, the start vector and the notices. Absent
            # in a file an earlier bngsim wrote, whose record then holds only
            # what the fields above give it.
            start_state = (
                np.asarray(f["start_state"][()], dtype=np.float64) if "start_state" in f else None
            )
            model_text: str | bytes | None = None
            if "model_text" in f:
                raw = bytes(np.asarray(f["model_text"][()], dtype=np.uint8))
                kind = _text(f["model_text"].attrs.get("type", "bytes"))
                model_text = raw.decode("utf-8") if kind == "str" else raw
            record_parts: dict[str, Any] | None = None
            notices: tuple[Notice, ...] = ()
            rows_differ: tuple[str, ...] | None = None
            if "record" in f:
                held = json.loads(_text(f["record"][()]))
                record_parts = held.get("parts")
                notices = tuple(Notice(**n) for n in held.get("notices", ()))
                if held.get("rows_differ") is not None:
                    rows_differ = tuple(held["rows_differ"])

            # GH #155: absent ⇒ None, not recorded, as in a file that predates
            # saving the seed.
            ic_seed: dict[str, dict[str, float]] | None = None
            if "ic_sensitivity_seed" in f:
                ic_seed = {}
                for sp, p, c in zip(
                    _names("ic_sensitivity_seed_species") or [],
                    _names("ic_sensitivity_seed_params") or [],
                    f["ic_sensitivity_seed"][()],
                    strict=True,
                ):
                    ic_seed.setdefault(sp, {})[p] = float(c)

            # SSA / PSA diagnostics. A file that predates saving them loads the zero
            # dicts a result built from raw arrays carries, and a key the file lacks
            # keeps its zero value.
            def _attr(value: Any) -> Any:
                return value.item() if isinstance(value, np.generic) else value

            ssa_diagnostics: dict[str, Any] | None = None
            if "ssa_diagnostics" in f:
                ssa_diagnostics = _empty_ssa_diagnostics()
                for key, val in f["ssa_diagnostics"].attrs.items():
                    ssa_diagnostics[key] = _attr(val)

            def _read_psa(grp: Any) -> dict[str, Any]:
                diag = _empty_psa_diagnostics()
                for key, val in grp.attrs.items():
                    diag[key] = _attr(val)
                for key in grp:
                    diag[key] = np.array(grp[key]).tolist()
                return diag

            psa_diagnostics = (
                _read_psa(f["psa_diagnostics"])
                if "psa_diagnostics" in f
                else _empty_psa_diagnostics()
            )
            psa_runs: list[dict[str, Any]] | None = None
            if "psa_diagnostics_runs" in f:
                runs_grp = f["psa_diagnostics_runs"]
                psa_runs = [_read_psa(runs_grp[k]) for k in sorted(runs_grp, key=int)]

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
            _species_volume_factors=volume_factors,
            _seed=seed,
            _ic_sensitivity_seed=ic_seed,
            custom_attrs=custom_attrs,
            _sensitivities=sensitivities,
            _sensitivity_params=sensitivity_params,
            _sensitivities_ic=sensitivities_ic,
            _sensitivity_ic_species=sensitivity_ic_species,
            _observable_sensitivities=observable_sensitivities,
            _expression_sensitivities=expression_sensitivities,
            _observable_sensitivities_ic=observable_sensitivities_ic,
            _expression_sensitivities_ic=expression_sensitivities_ic,
            _reaction_firing_counts=reaction_firing_counts,
            _reaction_propensity_integrals=reaction_propensity_integrals,
            _reaction_labels=reaction_labels,
            _ssa_diagnostics=ssa_diagnostics,
        )
        result._psa_diagnostics = psa_diagnostics
        result._psa_diagnostics_runs = psa_runs
        # GH #221 — the NaN rows this file records as refusals, not as a failed
        # solve, so `gradient` keeps its zero-weight exemption after a round trip.
        result._ar_sens_refused = frozenset(ar_sens_refused)
        result._ar_sens_blocked = frozenset(ar_sens_blocked)
        result._varvol_live_vol = live_vol
        result._varvol_conc_factor = conc_factor
        result._varvol_amount_factor = amount_factor
        result._volume_record_missing = volume_record_missing
        result._stop_condition = stop_condition
        result._started_from = started_from
        result._start_state_sha256 = start_state_sha256
        result._start_state = start_state
        result._record_frozen = record_parts
        result._record_differs = rows_differ
        result._source_text = model_text
        result._notices = notices
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

        The columns are :attr:`species`, the values this Result reports.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Raises
        ------
        ValueError
            For a squeezed ``run_batch`` or ``run_replicates`` Result, which
            holds one trajectory per row: write each row's ``.cdat`` from the
            unsqueezed results (``squeeze=False``).
        RuntimeError
            If the file cannot be opened for writing.
        """
        # The species this Result reports, which the solve's post-processing
        # corrects after the core's copy is written (an assignment-rule species,
        # a resized compartment's concentrations). Written from the core's copy,
        # the .cdat disagreed with Result.species, to_csv(kind="species") and
        # the .cdat the same Result wrote after save/load (issue #1505). The
        # core's writer keeps the layout: one .cdat format, with or without a
        # core.
        from bngsim._bngsim_core import _write_cdat

        if np.ndim(self._species) != 2:
            # A squeezed run_batch / run_replicates Result holds one trajectory
            # per row; a .cdat file holds one. (It used to fail inside string
            # formatting, with a TypeError that named neither.)
            raise UsageError(
                f"to_cdat writes one trajectory; this Result holds "
                f"{np.shape(self._species)[0]} (species of shape "
                f"{np.shape(self._species)}). Write each row's .cdat from the "
                f"unsqueezed results (squeeze=False)."
            )
        _write_cdat(
            str(path),
            np.ascontiguousarray(self._time, dtype=float),
            np.ascontiguousarray(self._species, dtype=float),
            list(self._species_names),
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
            raise UsageError(f"kind must be 'observables' or 'species', got {kind!r}")
        if len(sep) != 1:
            raise UsageError(f"sep must be a single character, got {sep!r}")

        if kind == "observables":
            data = self._observables
            names = self._observable_names
        else:
            data = self._species
            names = self._species_names

        if data.ndim != 2:
            raise UsageError(
                f"to_csv requires a 2-D single-sim result; got "
                f"{kind} with shape {data.shape}. For batch results "
                "(3-D), call to_csv on each replicate Result."
            )

        path = Path(path)
        n_t = self._time.shape[0]
        n_cols = data.shape[1]

        with open(path, "w", encoding="utf-8") as f:
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
        n_cols)``; a block that is empty on the inputs stays empty.
        Each row keeps its own compartment sizes and per-sample volume factors.

        Parameters
        ----------
        results : list[Result]
            Results from individual simulations. All must have the same time
            points, the same species, observable and expression names in the
            same order, the same sensitivity parameters and IC species, the
            same reaction labels, and the same blocks present: the stack has
            one time axis and one set of column names, which every row is
            read by (issue #1513).

        Returns
        -------
        Result
            A batch result with 3D arrays.

        Raises
        ------
        ValueError
            If ``results`` is empty, any of them met a stop condition
            (:attr:`stop_condition`, issue #1369), or a row's time points,
            names or blocks differ from the first row's (issue #1513).
        """
        if not results:
            raise UsageError("Cannot squeeze empty result list")
        stopped = [i for i, r in enumerate(results) if r._stop_condition is not None]
        if stopped:
            # A stack has one time axis and says of no row that it stopped
            # (issue #1369). A callable's row keeps every sample, so this is
            # not "cut short" of every row it names.
            rows = ", ".join(map(str, stopped[:10])) + (", ..." if len(stopped) > 10 else "")
            raise UsageError(
                f"Cannot squeeze results that met a stop condition ({len(stopped)} of "
                f"{len(results)}: rows {rows}): a stack holds rows of one length and "
                "does not say which stopped. Read them from the list, where "
                "Result.stop_condition names the condition."
            )
        if len(results) == 1:
            return results[0]
        _refuse_disagreeing_rows(results)

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
        # SSA boundary diagnostics (GH #110): counts add up across replicates,
        # the first offender is the first replicate's, and the propensity
        # backend is the one every replicate used, or "mixed". Dropping these
        # left a batch reporting the default dict's backend as if it had run.
        diags = [r._ssa_diagnostics for r in results]
        backends = {d.get("propensity_backend", "unknown") for d in diags}
        agg_ssa: dict[str, Any] = {
            "n_negative_crossings": sum(int(d.get("n_negative_crossings", 0)) for d in diags),
            "first_negative_species": next(
                (d["first_negative_species"] for d in diags if d.get("n_negative_crossings")),
                "",
            ),
            "n_reverse_fires": sum(int(d.get("n_reverse_fires", 0)) for d in diags),
            "first_reverse_reaction": next(
                (d["first_reverse_reaction"] for d in diags if d.get("n_reverse_fires")), ""
            ),
            "n_rounded_populations": sum(int(d.get("n_rounded_populations", 0)) for d in diags),
            "first_rounded_species": next(
                (d["first_rounded_species"] for d in diags if d.get("n_rounded_populations")),
                "",
            ),
            "propensity_backend": next(iter(backends)) if len(backends) == 1 else "mixed",
        }

        seeds = {r._seed for r in results}
        squeeze_seed = next(iter(seeds)) if len(seeds) == 1 else None
        # Issue #1701: the same rule for where the rows started, each part on
        # its own. A scan over an initial-condition parameter starts every row
        # from the initial conditions, and each from a state of its own.
        froms = {r._started_from for r in results}
        digests = {r.start_state_sha256 for r in results}
        squeeze_start = (
            next(iter(froms)) if len(froms) == 1 else None,
            next(iter(digests)) if len(digests) == 1 else None,
        )

        # GH #155: same rule for ∂x(0)/∂θ. A nonlinear derived IC makes the
        # coefficients point-dependent, so the rows can legitimately disagree
        # across a batch; None then says "ask the unsqueezed rows" rather than
        # picking one arbitrarily and reporting it as the batch's.
        ic_seeds = [r._ic_sensitivity_seed for r in results]
        squeeze_ic_seed = ic_seeds[0] if all(s == ic_seeds[0] for s in ic_seeds) else None

        # GH #196 — carry every populated sensitivity block through, stacking on
        # a new leading sim axis. Empty-on-input blocks stay empty (0, 0, 0).
        def stack_sens(attr: str) -> NDArray[np.float64]:
            if getattr(results[0], attr).size == 0:
                return np.empty((0, 0, 0))
            return np.stack([getattr(r, attr) for r in results], axis=0)

        # GH #616 — the per-reaction statistics stack the same way, (sims, times,
        # reactions); an ensemble's covariance of observable and score is exactly
        # what the stacked block is for.
        def stack_2d(attr: str) -> NDArray[np.float64]:
            if getattr(results[0], attr).size == 0:
                return np.empty((0, 0))
            return np.stack([getattr(r, attr) for r in results], axis=0)

        # Each row's volume record (issue #787): its compartment sizes, which a
        # batch row may have written (issue #743), one list per row, and its
        # per-sample factors on a new leading sim axis. The stack took row 0's
        # sizes for every row and none of the factors, so a batch over a size of
        # 2 and 5 held 2 for both. A row's live-volume column is stacked as its
        # amount factor, which as_roadrunner multiplies by in place of that
        # column, so the rows need not agree on which species have which; a row
        # without a factor another row has gets the one it reads instead: 1 for
        # a concentration, its static size for an amount.
        n_t = time.shape[0]
        vfs = [r._species_volume_factors for r in results]
        stack_vf = (
            None
            if all(v is None for v in vfs)
            else [
                np.asarray(v if v is not None else np.ones(len(species_names))).tolist()
                for v in vfs
            ]
        )

        def static_size(r: Result, j: int) -> NDArray[np.float64]:
            vf = r._species_volume_factors
            return np.full(n_t, vf[j] if vf is not None and j < len(vf) else 1.0, dtype=np.float64)

        def stack_factors(
            maps: list[dict[int, NDArray[np.float64]]],
            fill: Callable[[Result, int], NDArray[np.float64]],
        ) -> dict[int, NDArray[np.float64]] | None:
            keys = sorted(set().union(*maps))
            if not keys:
                return None
            return {
                j: np.stack(
                    [m[j] if j in m else fill(r, j) for r, m in zip(results, maps, strict=True)]
                )
                for j in keys
            }

        stack_conc = stack_factors(
            [r._varvol_conc_factor or {} for r in results], lambda r, j: np.ones(n_t)
        )
        stack_amount = stack_factors(
            [
                {
                    **{j: r._species[:, k] for j, k in (r._varvol_live_vol or {}).items()},
                    **(r._varvol_amount_factor or {}),
                }
                for r in results
            ],
            static_size,
        )

        out = Result(
            core=None,
            _time=time,
            _species=species_3d,
            _observables=obs_3d,
            _expressions=expr_3d,
            _species_names=species_names,
            _observable_names=observable_names,
            _expression_names=expression_names,
            _solver_stats=agg_stats,
            _species_volume_factors=stack_vf,
            _seed=squeeze_seed,
            _ic_sensitivity_seed=squeeze_ic_seed,
            _sensitivities=stack_sens("_sensitivities"),
            _sensitivity_params=results[0]._sensitivity_params,
            _sensitivities_ic=stack_sens("_sensitivities_ic"),
            _sensitivity_ic_species=results[0]._sensitivity_ic_species,
            _observable_sensitivities=stack_sens("_observable_sensitivities"),
            _expression_sensitivities=stack_sens("_expression_sensitivities"),
            _observable_sensitivities_ic=stack_sens("_observable_sensitivities_ic"),
            _expression_sensitivities_ic=stack_sens("_expression_sensitivities_ic"),
            _reaction_firing_counts=stack_2d("_reaction_firing_counts"),
            _reaction_propensity_integrals=stack_2d("_reaction_propensity_integrals"),
            _reaction_labels=results[0]._reaction_labels,
            _ssa_diagnostics=agg_ssa,
        )
        # The assignment-rule report metadata each row was stamped with (issues
        # #698, #743). A redirect is kept where every row agrees on it; a row
        # whose compartment size differs divides by its own size, and its row
        # of the stacked sensitivity tensor already holds that, so a selector
        # with no common redirect reads the tensor. A refusal of any row stands.
        maps = [r._ar_sens_map for r in results]
        out._ar_sens_map = {
            name: entry
            for name, entry in maps[0].items()
            if all(m.get(name) == entry for m in maps[1:])
        }
        out._ar_sens_blocked = frozenset().union(*(r._ar_sens_blocked for r in results))
        out._ar_sens_refused = frozenset().union(*(r._ar_sens_refused for r in results))
        out._started_from, out._start_state_sha256 = squeeze_start
        if squeeze_start[1] is not None:
            # One start for every row: its vector, for the record (issue #1703).
            out._start_state = next(
                (r._start_state for r in results if r._start_state is not None), None
            )
        # The record's parts the rows agree on, and every row's notices
        # (issue #1703).
        _record.squeeze_into(out, results)
        out._notices = _notices.unique(n for r in results for n in r._notices)
        out._varvol_conc_factor = stack_conc
        out._varvol_amount_factor = stack_amount
        # A row loaded from a file that does not record its sizes leaves the
        # stack's record incomplete, so the stack says so too.
        out._volume_record_missing = next(
            (r._volume_record_missing for r in results if r._volume_record_missing), None
        )
        # Each run's PSA report is kept as it was. The stack's own report says
        # only what the runs agree on without a choice of statistic: whether any
        # used partial scaling, the largest peak, whether any crossed the
        # activation level. Dropped, a stack of PSA runs read as inactive, as if
        # it had run exact SSA. A stack of stacks keeps every run's report.
        runs = [run for r in results for run in r.psa_diagnostics_per_run]
        out._psa_diagnostics_runs = runs
        if any(run.get("active") for run in runs):
            summary = _empty_psa_diagnostics()
            summary["active"] = True
            summary["peak_population"] = max(
                float(run.get("peak_population", 0.0)) for run in runs
            )
            summary["activation_crossed"] = any(run.get("activation_crossed") for run in runs)
            out._psa_diagnostics = summary
        return out

    # ─── Names that are not the API (issue #1700) ──────────────────

    def _names_by_kind(self) -> tuple[list[str], list[str], list[str]]:
        """The species, observable and expression names, for :func:`_where_is`."""
        return self._species_names, self._observable_names, self._expression_names

    def _where_is(self, name: str, besides: str | None = None) -> str | None:
        """Which of this result's blocks holds a column *name*, as the code
        that reads it, or ``None``. *besides* is a block not to look in."""
        return _where_is(name, self._names_by_kind(), besides)

    def __dir__(self) -> list[str]:
        # Without the names of bngsim._foreign_names' table (issue #1700).
        return _foreign_names.public_dir(self)

    if not TYPE_CHECKING:
        # Hidden from type checkers, which then refuse ``result["A"]`` as they did.

        def __getitem__(self, key: object) -> NoReturn:
            # ``result["A"]``, a pandas or RoadRunner habit, was "'Result' object
            # is not subscriptable", which says nothing of where "A" is.
            text = (
                "'Result' object is not subscriptable: it holds one array per kind, "
                "result.time, result.species, result.observables and result.expressions, "
                "and each of those is indexed"
            )
            if isinstance(key, str):
                where = self._where_is(key)
                if where is None:
                    names = [
                        *self._species_names,
                        *self._observable_names,
                        *self._expression_names,
                    ]
                    close = _foreign_names.near_matches(key, names)
                    where = (
                        f"No column is named {key!r}; close matches: "
                        f"{_foreign_names.quote_list(close)}."
                        if close
                        else f"No column is named {key!r}."
                    )
                raise ArgumentTypeError(f"{text} by name. {where}")
            shown = repr(key) if isinstance(key, int) else "..."
            raise ArgumentTypeError(
                f"{text}: result.species[{shown}], result.observables[{shown}]."
            )

        # With __getitem__ defined, Python would iterate a Result by calling it
        # with 0, 1, 2, ...; None keeps iter() and ``in`` refusing, as before.
        __iter__ = None

    # ─── Dunder methods ─────────────────────────────────────────────

    def __reduce__(self) -> NoReturn:
        raise_cannot_pickle(
            "Result",
            "Send what you need from it instead — as_roadrunner() for a labeled table "
            "(its column names survive the trip), to_xarray() for a Dataset, "
            ".time / .species / .observables for plain arrays — or save(path) here and "
            "Result.load(path) on the other side.",
        )

    def __copy__(self) -> Result:
        # Preserved explicitly: copy.copy() falls back to __reduce_ex__, which
        # defers to the refusal above, and a shallow copy works here and always
        # has. The copy shares this Result's core and arrays, as the default
        # copy did — and unlike Model and Simulator, which refuse for exactly
        # that sharing (issue #643), a Result can afford it: it has no setter
        # and no mutating method, so no call on the copy writes through. An
        # in-place numpy write into a shared array does reach the original, but
        # that aliasing is not the copy's doing — the accessors hand back the
        # stored array itself, so it is there through the original too. Adding a
        # setter WOULD make this a Model-shaped trap and move it to the
        # refusals; test_copy_refusals.py fails when one appears.
        return shallow_copy(self)

    def __repr__(self) -> str:
        return (
            f"Result(n_times={self.n_times}, "
            f"n_species={self.n_species}, "
            f"n_observables={self.n_observables})"
        )


def _unclamped(name: str) -> str:
    """*name* without a fixed species' ``$``, which the model's species names
    never hold (see :func:`bngsim._model._unclamped`)."""
    from bngsim._model import _unclamped

    return _unclamped(name)


_KINDS = (("species", "species"), ("observable", "observables"), ("expression", "expressions"))


def _where_is(
    name: str, names_by_kind: tuple[list[str], ...], besides: str | None = None
) -> str | None:
    """The block holding a column *name*, as the code that reads it, or
    ``None`` (issue #1700). *names_by_kind* is a Result's species, observable
    and expression names; *besides* a kind not to look in. A fixed species'
    ``$`` may be given, as the model's names never hold it."""
    for (kind, attr), names in zip(_KINDS, names_by_kind, strict=False):
        if kind == besides:
            continue
        if name in names or (kind == "species" and _unclamped(name) in names):
            lead = "a species" if kind == "species" else f"an {kind}"
            return f"{name!r} is {lead}: result.{attr}[{name!r}]."
    return None


class _SpeciesArray(np.ndarray[Any, np.dtype[np.float64]]):
    """:attr:`Result.species`: a numpy array that also takes a species name in
    the column position (issue #1700).

    ``result.species["A"]`` used to raise numpy's "only integers, slices ..."
    while ``result.observables["A"]`` worked. :attr:`Result.observables` is a
    wrapper around its array; making ``species`` one too would have taken its
    numpy methods (``.copy()``, ``.max()``, item assignment) away from every
    caller, so this is the array itself, with one more kind of key.

    A name is answered, read or written, only by the array :attr:`Result.species`
    returns. Every array numpy derives from it — a slice, a ufunc's output, a
    reduction (a number) — is a plain ``ndarray``, and the few methods that keep
    the class (``.copy()``, ``.T``, ``.reshape()``, ``.astype()``, ``np.sort``)
    make one that carries no names and refuses a name, because its columns need
    not be these: the trap issue #561 found in :class:`~bngsim.NamedArray`,
    where ``np.sort(a, axis=1)`` kept the labels of columns it had moved. A
    pickle round trip gives a plain ``ndarray``.

    Every key goes through Python, so indexing costs about 0.2 µs more than on
    the plain array (``result.species[i]``: 0.39 µs against 0.16 µs here);
    iteration and ufuncs hand the plain array to numpy and cost a call each.
    """

    # The view Result.species returns sets these; every other instance keeps
    # the class's None. ndarray subclasses carry attributes in a __dict__.
    _plain: NDArray[np.float64] | None = None
    _index: dict[str, int] | None = None
    _names_by_kind: tuple[list[str], ...] = ()

    @classmethod
    def over(cls, data: NDArray[np.float64], names: list[str], result: Any) -> _SpeciesArray:
        plain = np.asarray(data)
        view = plain.view(cls)
        # Indexed in its place, so a key that is no name costs one Python call
        # more than on the array itself, and what it returns is a plain ndarray.
        view._plain = plain
        view._index = {n: i for i, n in enumerate(names)}
        # The blocks' names, for the "is an observable" hint. Names, not the
        # Result: the Result holds this view, so a reference back would be a
        # cycle, and a large result would wait for the cyclic collector.
        view._names_by_kind = result._names_by_kind()
        return view

    def _column(self, name: str) -> int:
        index = self._index
        if index is None:
            raise NotFoundError(
                f"Species {name!r}: this array was derived from result.species, and only "
                "result.species itself takes a species name, since a copy, a transpose or "
                "a sort need not keep its columns. Index result.species by name first, "
                f"then derive: result.species[{name!r}].",
                details={"name": name},
            )
        if name in index:
            return index[name]
        bare = _unclamped(name)
        if bare in index:
            return index[bare]
        # A species the name matches loosely first: BNGL's Molecules S1 S1()
        # makes S1 an observable too, and here the caller wants the species.
        close = _foreign_names.near_matches(name, index)
        loose = [c for c in close if _foreign_names._bare(c) == _foreign_names._bare(name)]
        where = f"Close matches: {_foreign_names.quote_list(close)}." if loose else None
        if where is None:
            where = _where_is(name, self._names_by_kind, besides="species")
        if where is None and close:
            where = f"Close matches: {_foreign_names.quote_list(close)}."
        text = f"Species {name!r} not found."
        if where is not None:
            text += f" {where}"
        raise NotFoundError(
            f"{text} result.species_names lists all {len(index)}.",
            code="output.not_found",
            details={"name": name, "close_matches": list(close)},
        )

    def _whole(self) -> NDArray[np.float64]:
        plain = self._plain
        return plain if plain is not None else np.asarray(self)

    @overload  # type: ignore[override]
    def __getitem__(self, key: str) -> NDArray[np.float64]: ...
    @overload
    def __getitem__(self, key: Any) -> Any: ...

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._whole()[..., self._column(key)]
        if isinstance(key, tuple) and key and isinstance(key[-1], str):
            # (rows, name): the name's column, then the rest of the key on it, which
            # for basic and advanced keys alike selects what (rows, its index)
            # would on a 2-D array, and the leading axes of a batch.
            return self._whole()[..., self._column(key[-1])][key[:-1]]
        plain = self._plain
        if plain is not None:
            return plain[key]
        out = super().__getitem__(key)
        return out.view(np.ndarray) if isinstance(out, _SpeciesArray) else out

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        # A name writes where it reads, so ``species["A"] *= 2`` (a read, then a
        # write back under the same key) works as it does for an index.
        if isinstance(key, str):
            self._whole()[..., self._column(key)] = value
        elif isinstance(key, tuple) and key and isinstance(key[-1], str):
            column = self._whole()[..., self._column(key[-1])]
            column[key[:-1]] = value
        else:
            self._whole()[key] = value

    def __iter__(self) -> Any:
        # The plain array's own iterator: ndarray iteration indexes row by row,
        # which through __getitem__ above would cost a Python call per row.
        return iter(self._whole())

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        # A ufunc runs on the plain arrays, so its output is plain: a new array,
        # whose columns this class knows nothing about, and a reduction a number
        # on every numpy (1.x hands a subclass a 0-d array where it means one).
        # Written into ``out=`` (``species += 1`` is), it returns ``out`` as the
        # caller passed it, as numpy's own ufuncs do, so the name keeps the array.
        args = [np.asarray(x) if isinstance(x, _SpeciesArray) else x for x in inputs]
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(
                np.asarray(o) if isinstance(o, _SpeciesArray) else o for o in out
            )
        result = getattr(ufunc, method)(*args, **kwargs)
        if out is None or method == "at":
            return result
        return out[0] if len(out) == 1 else out

    def __reduce__(self) -> Any:
        return np.asarray(self).__reduce__()

    def __repr__(self) -> str:
        return repr(np.asarray(self))


class _ObservableAccessor:
    """Wrapper allowing both array access and named column access.

    ``result.observables`` returns the full array.
    ``result.observables["name"]`` returns a single column.
    ``result.observables[0]`` returns a single time-point row.

    A squeezed batch (``run_batch(..., squeeze=True)``) carries one more
    leading axis, ``(n_sims, n_times, n_cols)``. The named column lives on the
    LAST axis on both layouts, so the lookup indexes there (GH #560): a
    hard-coded ``[:, idx]`` took the *time* axis on a batch and returned
    ``(n_sims, n_cols)`` — one row per replicate, all of it from time row
    ``idx`` — which is the same rank and a plausible set of numbers, so
    nothing raised. An integer or slice key still indexes the leading axis,
    which is the time axis on a single run and the replicate axis on a batch,
    matching what the same key does on the underlying array.
    """

    __slots__ = ("_data", "_names", "_name_to_idx", "_kind", "_names_by_kind", "_listed")

    def __init__(
        self,
        data: NDArray[np.float64],
        names: list[str],
        kind: str = "observable",
        names_by_kind: tuple[list[str], ...] = (),
        listed: str | None = None,
    ) -> None:
        self._data = data
        self._names = names
        self._name_to_idx = {n: i for i, n in enumerate(names)}
        # For the KeyError (issue #1700): what a name is looked up as, the
        # Result's names to look in for one that is something else (the lists,
        # not the Result, so an accessor kept does not keep the Result's arrays
        # alive), and where the names are listed.
        self._kind = kind
        self._names_by_kind = names_by_kind
        self._listed = listed or f"result.{kind}_names"

    def __getitem__(self, key: str | int | slice) -> NDArray[np.float64]:
        if isinstance(key, str):
            if key not in self._name_to_idx:
                raise NotFoundError(
                    self._not_found(key), code="output.not_found", details={"name": key}
                )
            # Last axis, not the second: see the note above about batches.
            return self._data[..., self._name_to_idx[key]]
        return self._data[key]

    def _not_found(self, key: str) -> str:
        text = f"{self._kind.capitalize()} '{key}' not found."
        where = _where_is(key, self._names_by_kind, besides=self._kind)
        if where is None:
            close = _foreign_names.near_matches(key, self._names)
            where = f"Close matches: {_foreign_names.quote_list(close)}." if close else None
        if where is not None:
            text += f" {where}"
        if len(self._names) <= 50:
            return f"{text} Available: {self._names}"
        return f"{text} {self._listed} lists all {len(self._names)}."

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
        "reaction_firing_counts",
        "reaction_propensity_integrals",
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
            from bngsim._error_codes import install
            from bngsim._exceptions import DependencyError

            raise DependencyError(
                "xarray is required for Result.xr. Install with: pip install 'bngsim[xarray]'",
                remedies=[install("xarray", extra="xarray")],
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
        if name in ("reaction_firing_counts", "reaction_propensity_integrals"):
            # GH #616 — SSA per-reaction event statistics.
            if not result.has_reaction_stats:
                raise AttributeError(
                    "Result has no per-reaction statistics. Run with reaction_stats=True."
                )
            return xr.DataArray(
                getattr(result, name),
                dims=("time", "reaction"),
                coords={"time": result._time, "reaction": result._reaction_labels},
                name=name,
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


# What a row of a stack is read by, besides its time points: the stack keeps row
# 0's list of each and puts every row's columns under it by position.
_SQUEEZE_NAME_LISTS = (
    ("species", "_species_names"),
    ("observables", "_observable_names"),
    ("expressions", "_expression_names"),
    ("sensitivity parameters", "_sensitivity_params"),
    ("IC-sensitivity species", "_sensitivity_ic_species"),
    ("reaction labels", "_reaction_labels"),
)
# The blocks row 0 alone decided whether to stack.
_SQUEEZE_BLOCKS = (
    ("expression values", "_expressions"),
    ("species sensitivities", "_sensitivities"),
    ("species IC sensitivities", "_sensitivities_ic"),
    ("observable sensitivities", "_observable_sensitivities"),
    ("expression sensitivities", "_expression_sensitivities"),
    ("observable IC sensitivities", "_observable_sensitivities_ic"),
    ("expression IC sensitivities", "_expression_sensitivities_ic"),
    ("reaction firing counts", "_reaction_firing_counts"),
    ("reaction propensity integrals", "_reaction_propensity_integrals"),
)


def _names_text(names: list[str]) -> str:
    shown = ", ".join(repr(n) for n in names[:6])
    return f"[{shown}{', ...' if len(names) > 6 else ''}]"


def _refuse_disagreeing_rows(results: list[Result]) -> None:
    """Refuse a stack whose rows do not all share row 0's times, names and blocks.

    The stack has one time axis and one list of names for each kind of column,
    both row 0's, and every row is read through them by position. A row run on
    other times, or with its species in another order, was labelled with row
    0's and read wrong at every point, with nothing said (issue #1513). The
    comparison is exact: every stack bngsim builds itself (run_batch,
    run_replicates, parameter_scan) runs one grid on clones of one model.
    """
    first = results[0]
    for i, row in enumerate(results[1:], start=1):
        problem = None
        if row._time.shape != first._time.shape:
            problem = (
                f"has {row._time.shape[0]} time points where row 0 has {first._time.shape[0]}"
            )
        elif not np.array_equal(row._time, first._time):
            k = int(np.flatnonzero(row._time != first._time)[0])
            problem = (
                f"has time point {k} at t = {float(row._time[k])!r} where row 0 has it at "
                f"t = {float(first._time[k])!r}"
            )
        else:
            for what, attr in _SQUEEZE_NAME_LISTS:
                mine, theirs = list(getattr(row, attr)), list(getattr(first, attr))
                if mine != theirs:
                    problem = (
                        f"has {what} {_names_text(mine)} where row 0 has {_names_text(theirs)}"
                    )
                    break
            else:
                for what, attr in _SQUEEZE_BLOCKS:
                    mine, theirs = getattr(row, attr).size > 0, getattr(first, attr).size > 0
                    if mine != theirs:
                        problem = (
                            f"has {what} where row 0 has none"
                            if mine
                            else f"has no {what} where row 0 has them"
                        )
                        break
        if problem is not None:
            raise UsageError(
                f"Cannot squeeze these results: row {i} {problem}. A stack has one time "
                "axis and one set of column names, the first row's, and reads every row "
                "by them, so the rows must agree on both. Read them from "
                "the list, or rerun them on one time grid and one model."
            )


def _empty_sens_block(arr: NDArray | None) -> NDArray[np.float64]:
    """A sensitivity block, defaulting an absent one to the empty ``(0, 0, 0)``.

    Centralizes the "no data" sentinel so every sensitivity block (species,
    observable, expression, and their IC variants — GH #196) shares one shape
    convention: a 3-D array that is empty (``size == 0``) when not computed.
    """
    return arr if arr is not None else np.empty((0, 0, 0))


def _empty_2d_block(arr: NDArray | None) -> NDArray[np.float64]:
    """A ``(time, column)`` block, defaulting an absent one to the empty ``(0, 0)``.

    The 2-D counterpart of :func:`_empty_sens_block`, for the GH #616
    per-reaction statistics.
    """
    return arr if arr is not None else np.empty((0, 0))


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


def _resolve_output_selector(
    selector: str,
    species_names: list[str],
    observable_names: list[str],
    expression_names: list[str],
) -> dict[str, Any]:
    """Resolve one output selector against the given name lists (GH #195/#12).

    The single source of truth for output-selector semantics — typed prefixes
    and their aliases, the ``()`` function-call convention, and bare-name
    uniqueness — shared by :meth:`Result._resolve_one_output` and
    :meth:`~bngsim._simulator.SteadyStateResult._resolve_one_output` so both
    containers accept identical selectors. Returns the metadata dict from
    :func:`_output_meta`; see :meth:`Result.resolve_outputs` for the accepted
    forms and raised errors.
    """
    names_by_kind = {
        "species": species_names,
        "observable": observable_names,
        "expression": expression_names,
    }
    if not isinstance(selector, str):
        raise ArgumentTypeError(
            f"Output selector must be a string, got {type(selector).__name__}: {selector!r}."
        )
    sel = selector.strip()
    if not sel:
        raise UsageError("Empty output selector.")

    if ":" in sel:
        prefix, name = sel.split(":", 1)
        prefix = prefix.strip().lower()
        name = name.strip()
        if prefix not in _SELECTOR_PREFIX_ALIASES:
            valid = ", ".join(f"{p}:" for p in _SELECTOR_KINDS)
            raise UsageError(
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
        names = names_by_kind[kind]
        if name in names:
            return _output_meta(kind, name, names.index(name))
        raise UsageError(
            f"Unresolved selector {selector!r}: no {kind} named {name!r}. "
            f"Available {kind} names: {names}",
            code="output.not_found",
            details={"name": name},
        )

    # Bare name: resolve only if unique across all kinds.
    matches = [
        (kind, names_by_kind[kind].index(sel))
        for kind in _SELECTOR_KINDS
        if sel in names_by_kind[kind]
    ]
    if len(matches) == 1:
        kind, idx = matches[0]
        return _output_meta(kind, sel, idx)
    if len(matches) > 1:
        typed = ", ".join(f"{kind}:{sel}" for kind, _ in matches)
        raise UsageError(
            f"Ambiguous output selector {sel!r}: matches {typed}. "
            f"Use a typed selector to disambiguate."
        )
    # No verbatim match. A bare "foo()" is the function-call convention:
    # retry the stripped name against expressions.
    if sel.endswith("()"):
        base = sel[:-2].strip()
        if base in expression_names:
            return _output_meta("expression", base, expression_names.index(base))
    raise UsageError(
        f"Unresolved output selector {selector!r}: not found among "
        f"species, observables, or expressions.\n"
        f"  species:     {species_names}\n"
        f"  observables: {observable_names}\n"
        f"  expressions: {expression_names}",
        code="output.not_found",
        details={"name": sel},
    )


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
    with open(path, "w", encoding="utf-8") as f:
        # Header
        header = "# " + " ".join(["time"] + names)
        f.write(header + "\n")
        # Data rows
        for i in range(len(time)):
            row = [f"{time[i]:20.12e}"]
            for j in range(data.shape[1]):
                row.append(f"{data[i, j]:20.12e}")
            f.write(" ".join(row) + "\n")


_foreign_names.install(Result, "Result")
