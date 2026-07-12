"""bngsim.Model — High-level Python wrapper for NetworkModel.

This class delegates to the C++ ``NetworkModel`` and provides Python-friendly
helpers for loading models, updating parameters, and inspecting model state.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bngsim._exceptions import ModelError, ParameterError

if TYPE_CHECKING:
    from bngsim._bngsim_core import NetworkModel

logger = logging.getLogger("bngsim")


class Model:
    """A BioNetGen reaction network model.

    A Model holds species, reactions, observables, parameters, and functions.
    It can be loaded from ``.net`` files and, via the factory methods below,
    from Antimony and SBML inputs.

    Models are **not** thread-safe. For parallel workers, use :meth:`clone`
    to create independent copies.

    Parameters
    ----------
    _core : NetworkModel
        Internal C++ model object. Users should not construct this directly;
        use the factory methods instead.

    Examples
    --------
    >>> model = bngsim.Model.from_net("model.net")
    >>> model.n_species
    5
    >>> model.set_param("kf", 0.5)
    >>> model.get_param("kf")
    0.5
    >>> model.set_params({"kf": 1.0, "kr": 0.1})
    """

    __slots__ = (
        "_core",
        "_codegen_so_path",
        "_codegen_c_source",
        "_codegen_sec",
        "_codegen_cache_hit",
        "_libsbml_parse_sec",
        "_interpret_sec",
        "_jac_derive_sec",
        "_jac_attempted",
        "_net_path",
        "_ssa_issues",
        "_ar_report_map",
        "_varvol_conc_map",
        "_varvol_amount_map",
        "_varvol_ar_conc_map",
        "_varvol_ar_amount_map",
        "_varvol_event_resize_map",
        "_periodic_disc_max_step",
        "_want_output_sens",
        "_named_conc_states",
    )

    def __init__(self, _core: NetworkModel) -> None:
        self._core = _core
        self._codegen_so_path: str = ""
        # GH #198: whether codegen should emit the expression output-sensitivity
        # evaluator. Set by the Simulator before codegen prep (only a sensitivity
        # run needs it, since its build-time differentiation is expensive).
        self._want_output_sens: bool = False
        # In-process MIR micro-JIT codegen source (GH #78); set when the JIT
        # backend (BNGSIM_CODEGEN_JIT=mir) prepares codegen for this model.
        self._codegen_c_source: str = ""
        # Wall seconds the model's codegen prepare spent (T0.3). Set by the
        # _codegen.prepare_* entry points (~0 for ExprTk models that never
        # codegen or a cache hit; the cc compile time on a cold large model).
        # Read by Simulator.last_codegen_sec; surfaced by the rr_parity harness
        # so one run yields the setup cost without a run-twice-and-subtract.
        self._codegen_sec: float = 0.0
        # Whether the codegen .so was reused from the on-disk cache (True), freshly
        # compiled (False), or no .so was involved (None — ExprTk or MIR). Set by
        # the _codegen.prepare_* entry points; read by Simulator.codegen_cache_hit.
        # The definitive cache signal, not inferred from wall time.
        self._codegen_cache_hit: bool | None = None
        # Per-model setup wall seconds, each timed at its own boundary in the
        # SBML loader (read by Simulator.last_libsbml_parse_sec /
        # last_interpret_sec / last_jacobian_sec; surfaced by the rr_parity
        # harness). The per-step integration hot path is never instrumented.
        #   _libsbml_parse_sec — libSBML readSBML* + error check (shared C++ core).
        #   _interpret_sec     — doc → internal _core (bngsim Python interpretation).
        #   _jac_derive_sec    — analytical Functional Jacobian derivation (sympy
        #                        sp.diff, GH #76); 0 for all-Elementary models, an
        #                        FD fallback, or BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0.
        self._libsbml_parse_sec: float = 0.0
        self._interpret_sec: float = 0.0
        self._jac_derive_sec: float = 0.0
        # GH #145 once-only sentinel for the lazy analytical Functional Jacobian.
        # False until prepare_analytical_jacobian() has *attempted* the SymPy
        # derivation — set True regardless of whether it attached or fell back to
        # finite differences — so the ODE-solve trigger derives at most once per
        # model. analytical_jacobian_complete cannot be the sentinel: it is also
        # False for all-Elementary models and for legitimate FD fallbacks, which
        # would make a non-differentiable model re-run SymPy on every solve.
        self._jac_attempted: bool = False
        # Set by Model.from_net so downstream consumers (esp. the codegen
        # auto-trigger in Simulator) can route to the .net codegen path,
        # which handles derived-parameter chain rules that the model-based
        # path does not (issue #15).
        self._net_path: str = ""
        # Populated by the SBML loader (and only the SBML loader) with a
        # list of SsaIssue records for SSA-incompatible constructs. Empty
        # list means the model is SSA-clean as far as the loader can see.
        # See bngsim._ssa_validation.validate_for_ssa.
        self._ssa_issues: list = []
        # Populated by the SBML loader: maps a mangled AssignmentRule-target
        # species name to ``(kind, source_name)`` where kind is
        # "observable" or "expression". Simulator.run uses it to report the
        # rule's live value in the species column instead of the frozen
        # initial value (the species is emitted ``fixed``). Empty for .net
        # and non-AR models. See _sbml_loader.py section 11.
        self._ar_report_map: dict[str, tuple[str, str]] = {}
        # Populated by the SBML loader (GH #85): maps a mangled species name to
        # the mangled name of its variable-volume compartment (a rate-rule or
        # event-driven compartment, promoted to a species column). Simulator.run
        # uses it to rescale the reported concentration of that species from
        # ``amount / V_static`` to ``amount / V_live(t)``. Empty for .net,
        # static-compartment, and unit-volume models. See _sbml_loader.py.
        self._varvol_conc_map: dict[str, str] = {}
        # Populated by the SBML loader (GH #86): maps a mangled hOSU=false
        # species name to its rate-rule compartment, for the *amount* (bare-id)
        # report only. Such a species is integrated in concentration space and
        # its stored concentration is already correct (the dilution term is in
        # the dynamics), so — unlike _varvol_conc_map — its concentration column
        # is NOT rescaled; only as_roadrunner's bare-id selector must recover the
        # amount as ``conc * V_live(t)`` instead of ``conc * V_static``. Empty
        # for .net, static, and amount-valued-only models. See _sbml_loader.py.
        self._varvol_amount_map: dict[str, str] = {}
        # Populated by the SBML loader (GH #87): maps a mangled amount-valued
        # species name to ``(comp_expr_name, V_static)`` for a species in an
        # ASSIGNMENT-RULE compartment (e.g. ``tV := mV + dV``). Simulator.run
        # rescales its reported concentration from ``amount / V_static`` to
        # ``amount / V_live(t)``, reading V_live(t) from the compartment's own
        # assignment-rule *expression* column. Empty for .net, static, rate-rule-
        # only, and unit-volume models. See _sbml_loader.py.
        self._varvol_ar_conc_map: dict[str, tuple[str, float]] = {}
        # Populated by the SBML loader (GH #234): the hOSU=false counterpart of
        # _varvol_ar_conc_map. Maps a mangled species name that received the §8c
        # dilution term (in a time-varying ASSIGNMENT-RULE compartment) to the
        # compartment's expression-column name. Simulator._apply_varvol_ar_conc_map
        # records V_live(t) from that column so the bare-id amount selector reports
        # conc·V_live(t); the concentration column is already correct. Empty for
        # .net, static, and AR-compartment-free models. See _sbml_loader.py.
        self._varvol_ar_amount_map: dict[str, str] = {}
        # Populated by the SBML loader (GH #131): maps a mangled species name in
        # an EVENT-RESIZED compartment to ``(comp_obs_name, V_static, hOSU)``.
        # Simulator._apply_varvol_event_resize_map applies the report-time
        # concentration correction ``× V_static/V_live`` — for every hOSU=true
        # species (both ODE and SSA) and for hOSU=false species under SSA only —
        # reading V_live from the compartment's same-named observable column.
        # Empty for .net, static, and event-resize-free models. See _sbml_loader.py.
        self._varvol_event_resize_map: dict[str, tuple[str, float, bool]] = {}
        # Populated by the SBML loader (GH #88): a recommended integrator
        # ``max_step_size`` (float) for a model whose ODE RHS is forced by a
        # periodic floor()/modulo dosing schedule, so the adaptive integrator
        # cannot step over a narrow dose pulse. None (the default) for every
        # model without such a schedule — the integrator is then unconstrained,
        # byte-identical to before. Simulator.run applies it unless the caller
        # passes an explicit ``max_step``. See _sbml_loader.py.
        self._periodic_disc_max_step: float | None = None
        # Issue #11: named saved concentration states. Maps a user label to a
        # snapshot of the full live species-concentration vector (a copy of
        # get_state(), ordered like species_names). This is the multi-slot
        # analog of BNG2.pl's saveConcentrations("name") / resetConcentrations(
        # "name"): a block can save two distinct states and restore either one.
        # The *default* (unlabeled) slot is deliberately NOT stored here — it
        # continues to route through the C++ initial_conc mechanism (save_
        # concentrations()/reset()) so today's single-slot behavior is preserved
        # byte-for-byte. Carried through clone().
        self._named_conc_states: dict[str, np.ndarray] = {}

    # ─── Factory methods ──────────────────────────────────────────────────

    @classmethod
    def from_antimony(cls, path: str | Path) -> Model:
        """Load a model from an Antimony ``.ant`` file.

        Antimony is a human-readable model description language.
        Internally converts to SBML via libantimony, then loads
        via libsbml for correct SBML semantics.

        Requires: ``pip install antimony python-libsbml``

        Parameters
        ----------
        path : str or Path
            Path to the ``.ant`` file.

        Returns
        -------
        Model
            The loaded model.

        Raises
        ------
        ImportError
            If ``antimony`` or ``libsbml`` is not installed.
        FileNotFoundError
            If the file does not exist.
        ModelError
            If the file cannot be parsed.
        """
        from bngsim._sbml_loader import load_antimony_via_sbml

        try:
            return load_antimony_via_sbml(path)
        except (ImportError, FileNotFoundError):
            raise
        except Exception as e:
            raise ModelError(f"Failed to load Antimony file {path}: {e}") from e

    @classmethod
    def from_antimony_string(cls, text: str) -> Model:
        """Load a model from an Antimony string.

        Parameters
        ----------
        text : str
            Antimony model text.

        Returns
        -------
        Model
            The loaded model.
        """
        from bngsim._sbml_loader import load_antimony_string_via_sbml

        try:
            return load_antimony_string_via_sbml(text)
        except ImportError:
            raise
        except Exception as e:
            raise ModelError(f"Failed to load Antimony string: {e}") from e

    @classmethod
    def from_sbml(cls, path: str | Path, *, defer_jacobian: bool | None = None) -> Model:
        """Load a model from an SBML ``.xml`` file.

        Parameters
        ----------
        path : str or Path
            Path to the SBML file.
        defer_jacobian : bool, optional
            GH #145 escape hatch. The analytical Functional Jacobian (GH #76) is
            derived lazily at the first ODE-solve setup by default (``None``);
            pass ``defer_jacobian=False`` to derive it eagerly at load instead
            (the pre-#145 behavior, for A/B and safety). ``BNGSIM_EAGER_JACOBIAN=1``
            forces eager for every load path.

        Returns
        -------
        Model
            The loaded model.

        Raises
        ------
        ImportError
            If ``python-libsbml`` is not installed.
        FileNotFoundError
            If the file does not exist.
        ModelError
            If the file cannot be parsed.
        """
        from bngsim._sbml_loader import load_sbml

        try:
            model = load_sbml(path)
        except (ImportError, FileNotFoundError):
            raise
        except Exception as e:
            raise ModelError(f"Failed to load SBML file {path}: {e}") from e
        # GH #145 eager escape hatch: BNGSIM_EAGER_JACOBIAN=1 is honored inside the
        # loader for every SBML-family entry point; this restores derive-at-load
        # for the explicit ``defer_jacobian=False`` request. Default is lazy.
        if defer_jacobian is False:
            model.prepare_analytical_jacobian()
        return model

    @classmethod
    def from_sbml_string(cls, text: str, *, defer_jacobian: bool | None = None) -> Model:
        """Load a model from an SBML XML string.

        Parameters
        ----------
        text : str
            SBML XML text.
        defer_jacobian : bool, optional
            GH #145 escape hatch (see :meth:`from_sbml`). Default lazy; pass
            ``defer_jacobian=False`` (or set ``BNGSIM_EAGER_JACOBIAN=1``) to
            derive the analytical Functional Jacobian eagerly at load.

        Returns
        -------
        Model
            The loaded model.
        """
        from bngsim._sbml_loader import load_sbml_string

        try:
            model = load_sbml_string(text)
        except ImportError:
            raise
        except Exception as e:
            raise ModelError(f"Failed to load SBML string: {e}") from e
        if defer_jacobian is False:
            model.prepare_analytical_jacobian()
        return model

    @classmethod
    def from_net(cls, path: str | Path, *, defer_jacobian: bool | None = None) -> Model:
        """Load a model from a BNG ``.net`` file.

        Parameters
        ----------
        path : str or Path
            Path to the ``.net`` file.
        defer_jacobian : bool, optional
            GH #145 escape hatch. The analytical Functional Jacobian (GH #76) is
            derived lazily at the first ODE-solve setup by default (``None``);
            pass ``defer_jacobian=False`` (or set ``BNGSIM_EAGER_JACOBIAN=1``) to
            derive it eagerly at load instead (pre-#145 behavior, for A/B).

        Returns
        -------
        Model
            The loaded model.

        Raises
        ------
        ModelError
            If the file cannot be parsed.
        FileNotFoundError
            If the file does not exist.
        """
        from bngsim._bngsim_core import NetworkModel

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Net file not found: {path}")
        try:
            core = NetworkModel.from_net(str(path))
        except (ValueError, RuntimeError) as e:
            raise ModelError(f"Failed to load {path}: {e}") from e
        m = cls(_core=core)
        m._net_path = str(path)
        # GH #145: the analytical Functional Jacobian (GH #76) is consumed only by
        # ODE solves, so it is no longer derived here at load — it is deferred to
        # the first ODE-solve setup (Simulator.__init__ →
        # prepare_analytical_jacobian). A .net model run under SSA/PSA/NFsim/
        # RuleMonkey, or merely inspected, never pays the SymPy derivation.
        # (All-Elementary .net models carry the closed-form analytical Jacobian
        # from the C++ build regardless — there are no Functional reactions to
        # differentiate.) Eager escape hatch (A/B, safety): defer_jacobian=False
        # or BNGSIM_EAGER_JACOBIAN=1 restores the pre-#145 derive-at-load.
        from bngsim._jacobian import eager_jacobian_requested

        if eager_jacobian_requested(defer_jacobian):
            m.prepare_analytical_jacobian()
        return m

    # ─── Lazy analytical Jacobian (GH #145) ───────────────────────────────

    def prepare_analytical_jacobian(self) -> bool:
        """Derive and attach the analytical Functional Jacobian (GH #76), at
        most once.

        Idempotent (GH #145): the SymPy derivation runs only on the first call;
        later calls are no-ops guarded by the model's once-only sentinel. Returns
        whether the model now carries a *complete* analytical Jacobian (``False``
        if it fell back to finite differences, or was already FD / all-Elementary
        with the closed-form C++ Jacobian).

        This is the lazy-derivation entry point. The Jacobian is consumed only by
        ODE solves (CVODE's dense Jacobian, the steady-state Newton solver, and
        codegen's analytical-Jacobian emitter), so it is deferred off the model-
        load path (``from_sbml`` / ``from_net`` no longer derive) and triggered
        at ODE-solve setup. Call it directly to **warm a parent template before**
        :meth:`clone` fan-out: a warmed parent passes the derived terms to clones
        (which re-compile the derivative ExprTk strings with no SymPy), so
        parallel fitting derives once, not once per worker.
        """
        if self._jac_attempted:
            return bool(self._core.analytical_jacobian_complete)
        self._jac_attempted = True
        try:
            from bngsim._jacobian import attach_functional_jacobian

            t0 = time.perf_counter()
            attach_functional_jacobian(self._core)
            self._jac_derive_sec = time.perf_counter() - t0
        except Exception as e:
            # attach_functional_jacobian is contractually no-raise (it falls back
            # to FD and logs over-budget / unsupported cases itself, GH #95); this
            # guard only surfaces a genuinely unexpected error without re-deriving.
            logger.debug("Analytical Functional Jacobian skipped: %s", e)
        return bool(self._core.analytical_jacobian_complete)

    # ─── Load-phase timing accessors ──────────────────────────────────────
    # Public read-only views of the per-model setup timings the SBML loader
    # records, mirroring Simulator.last_libsbml_parse_sec / last_interpret_sec /
    # last_jacobian_sec for callers that hold only a Model (e.g. the rr_parity SSA
    # screen loads via Model.from_sbml and runs per-replicate Simulators, so the
    # parse/interpret cost lives on the Model, not on any one Simulator). Setup-
    # time only; never the integration hot path.

    @property
    def last_libsbml_parse_sec(self) -> float:
        """Wall seconds the SBML loader spent in the libSBML parse phase
        (``readSBML*`` + document-level error check). ``0.0`` for a non-SBML
        model (e.g. ``Model.from_net``). See
        :attr:`Simulator.last_libsbml_parse_sec`."""
        return float(self._libsbml_parse_sec)

    @property
    def last_interpret_sec(self) -> float:
        """Wall seconds spent interpreting the parsed libSBML document into the
        internal ``_core`` model (excludes libSBML parse, Jacobian derivation, and
        codegen). ``0.0`` for a non-SBML model. See
        :attr:`Simulator.last_interpret_sec`."""
        return float(self._interpret_sec)

    @property
    def last_jacobian_sec(self) -> float:
        """Wall seconds spent symbolically deriving this model's analytical
        Functional Jacobian (GH #76). ``0.0`` until the derivation runs (it is
        lazy since GH #145, and never runs on the SSA/PSA/NFsim paths). See
        :attr:`Simulator.last_jacobian_sec`."""
        return float(self._jac_derive_sec)

    # ─── Clone ────────────────────────────────────────────────────────────

    def clone(self) -> Model:
        """Deep copy the model for parallel workers.

        Each clone is fully independent — it has its own parameter values,
        species concentrations, and expression evaluator state.

        Returns
        -------
        Model
            An independent deep copy.
        """
        m = Model(_core=self._core.clone())
        m._net_path = self._net_path
        m._want_output_sens = self._want_output_sens
        m._codegen_so_path = self._codegen_so_path
        m._codegen_c_source = self._codegen_c_source
        m._codegen_sec = self._codegen_sec
        m._codegen_cache_hit = self._codegen_cache_hit
        # Carry the populated Jacobian + its derive time to clones (the existing
        # warm-clone path re-compiles the derivative ExprTk strings with NO sympy),
        # so a warmed parent yields cheap clones — the key invariant a future lazy
        # deferral (GH #145) relies on to avoid N× sympy in parallel fitting.
        m._libsbml_parse_sec = self._libsbml_parse_sec
        m._interpret_sec = self._interpret_sec
        m._jac_derive_sec = self._jac_derive_sec
        # GH #145: carry the once-only sentinel so a clone of a warmed parent does
        # NOT re-attempt the SymPy derivation. The C++ clone above already
        # re-compiles the parent's functional_jac into the clone's evaluator with
        # no SymPy, so a derived parent → cheap, already-warm clones; copying the
        # sentinel keeps the ODE-solve trigger a no-op on those clones (a clone of
        # an un-warmed parent inherits _jac_attempted=False and derives on first
        # solve — hence warm-before-clone for parallel fitting, GH #145 §3).
        m._jac_attempted = self._jac_attempted
        m._ssa_issues = list(self._ssa_issues)
        m._ar_report_map = dict(self._ar_report_map)
        m._varvol_conc_map = dict(self._varvol_conc_map)
        m._varvol_amount_map = dict(self._varvol_amount_map)
        m._varvol_ar_conc_map = dict(self._varvol_ar_conc_map)
        m._varvol_ar_amount_map = dict(self._varvol_ar_amount_map)
        m._varvol_event_resize_map = dict(self._varvol_event_resize_map)
        m._periodic_disc_max_step = self._periodic_disc_max_step
        # Issue #11: carry named concentration snapshots to the clone, each a
        # fresh copy so the clone's restore can never alias the parent's stored
        # vector. (The default slot lives in the C++ core, deep-copied above.)
        m._named_conc_states = {k: v.copy() for k, v in self._named_conc_states.items()}
        return m

    # ─── SSA validation ───────────────────────────────────────────────────

    def validate_for_ssa(self) -> list:
        """Return SSA-compatibility issues detected by the SBML loader.

        Returns
        -------
        list of :class:`bngsim.SsaIssue`
            One entry per detected construct; empty for SSA-clean models
            and for models loaded outside the SBML path
            (``Model.from_net``, builder).

        See Also
        --------
        bngsim.validate_for_ssa : module-level function with the same body.
        """
        from bngsim._ssa_validation import validate_for_ssa

        return validate_for_ssa(self)

    # ─── Parameter access ─────────────────────────────────────────────────

    def set_param(self, name: str, value: float) -> None:
        """Set a parameter value by name.

        Parameters
        ----------
        name : str
            Parameter name (e.g. "kf", "Km").
        value : float
            New value.

        Raises
        ------
        ParameterError
            If the parameter name is not found.
        """
        try:
            self._core.set_param(name, float(value))
        except (KeyError, RuntimeError) as e:
            raise ParameterError(f"Parameter '{name}' not found in model") from e

    def get_param(self, name: str) -> float:
        """Get a parameter value by name.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        float
            Current value.

        Raises
        ------
        ParameterError
            If the parameter name is not found.
        """
        try:
            return self._core.get_param(name)
        except (KeyError, RuntimeError) as e:
            raise ParameterError(f"Parameter '{name}' not found in model") from e

    def set_params(self, params: dict[str, float]) -> None:
        """Set multiple parameters from a dict.

        Parameters
        ----------
        params : dict[str, float]
            Parameter name → value mapping.

        Raises
        ------
        ParameterError
            If any parameter name is not found, or any value cannot be
            converted to float. Atomic: either all succeed or none do.

        Examples
        --------
        >>> model.set_params({"kf": 0.5, "kr": 0.1})
        """
        # Phase 1: Validate all names
        known = set(self._core.param_names)
        unknown = set(params.keys()) - known
        if unknown:
            raise ParameterError(
                f"Unknown parameter(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}"
            )
        # Phase 2: Convert all values (catches "x", None, etc.)
        converted: dict[str, float] = {}
        for name, value in params.items():
            try:
                converted[name] = float(value)
            except (TypeError, ValueError) as e:
                raise ParameterError(f"Invalid value for parameter '{name}': {value!r}") from e
        # Phase 3: Apply atomically (all validation passed)
        for name, value in converted.items():
            self._core.set_param(name, value)

    # ─── State management ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all species to their initial concentrations.

        Parameter values are **not** reset — only species concentrations.

        Equivalent to :meth:`restore_concentrations` with no label: it returns
        to the seed initial conditions, or — after an unlabeled
        :meth:`save_concentrations` — to that saved snapshot. Named snapshots
        (``save_concentrations(label=...)``) are unaffected.
        """
        self._core.reset()

    def save_concentrations(self, label: str | None = None) -> None:
        """Snapshot the current species concentrations for later restore.

        Implements BNG ``saveConcentrations()`` / ``saveConcentrations("name")``.

        Parameters
        ----------
        label : str, optional
            Name for the snapshot. When omitted (or ``None``), this preserves
            the historical single-slot behavior: the current concentrations
            become the new baseline initial state, so a subsequent :meth:`reset`
            (or :meth:`restore_concentrations` with no label) returns here rather
            than to the original ``.net`` seed. When a ``label`` is given, the
            snapshot is stored under that name in a separate multi-slot store and
            does **not** disturb the default slot; restore it later with
            ``restore_concentrations(label)``. Multiple named states coexist, so
            a multi-phase protocol (e.g. ``saveConcentrations("t=0")`` …
            ``saveConcentrations("start_competition")``) round-trips faithfully.

        Notes
        -----
        A named snapshot captures only the species concentrations (the bulk
        state vector, ordered like :attr:`species_names`); parameters and the
        current time are not part of it, matching BNG ``resetConcentrations``.
        """
        if label is None:
            self._core.save_concentrations()
            return
        # A named snapshot is a copy of the live state vector; storing get_state()
        # (which already returns a fresh array) is safe, but copy defensively so a
        # later set_state alias can never mutate a stored snapshot.
        self._named_conc_states[str(label)] = np.array(self._core.get_state(), dtype=np.float64)

    def restore_concentrations(self, label: str | None = None) -> None:
        """Restore species concentrations from a saved snapshot.

        Implements BNG ``resetConcentrations()`` / ``resetConcentrations("name")``.

        Parameters
        ----------
        label : str, optional
            Name of the snapshot to restore. When omitted (or ``None``), restores
            the default slot — identical to :meth:`reset` (the seed initial
            conditions, or the last unlabeled :meth:`save_concentrations`). When a
            ``label`` is given, restores the named snapshot saved by
            ``save_concentrations(label)``.

        Raises
        ------
        ModelError
            If ``label`` is given but no snapshot was saved under that name.
        """
        if label is None:
            self._core.reset()
            return
        key = str(label)
        snapshot = self._named_conc_states.get(key)
        if snapshot is None:
            known = ", ".join(sorted(self._named_conc_states)) or "(none)"
            raise ModelError(
                f"No saved concentration state named {key!r}. "
                f"Saved states: {known}. Call save_concentrations({key!r}) first."
            )
        self._core.set_state(snapshot)

    def has_saved_concentrations(self, label: str | None = None) -> bool:
        """Whether a named concentration snapshot is available to restore.

        Parameters
        ----------
        label : str, optional
            When given, reports whether a snapshot saved under that exact name
            exists. When omitted (or ``None``), reports whether *any* named
            snapshot exists. The default (unlabeled) slot is always restorable
            via :meth:`reset` and is not reflected here.
        """
        if label is None:
            return bool(self._named_conc_states)
        return str(label) in self._named_conc_states

    @property
    def saved_concentration_labels(self) -> list[str]:
        """Sorted names of the currently saved named concentration snapshots.

        Does not include the default (unlabeled) slot, which is restored via
        :meth:`reset` / :meth:`restore_concentrations` with no label.
        """
        return sorted(self._named_conc_states)

    def set_concentration(self, name: str, value: float) -> None:
        """Set a single species concentration by name.

        Parameters
        ----------
        name : str
            Species name (e.g. ``"A(b)"``).
        value : float
            New concentration value.

        Raises
        ------
        ModelError
            If the species name is not found.

        Notes
        -----
        Implements BNG ``setConcentration("name", value)`` action.
        """
        try:
            self._core.set_concentration(name, float(value))
        except (KeyError, RuntimeError) as e:
            raise ModelError(f"Species '{name}' not found in model") from e

    def get_concentration(self, name: str) -> float:
        """Get a single species concentration by name.

        Parameters
        ----------
        name : str
            Species name.

        Returns
        -------
        float
            Current concentration.

        Raises
        ------
        ModelError
            If the species name is not found.
        """
        try:
            return self._core.get_concentration(name)
        except (KeyError, RuntimeError) as e:
            raise ModelError(f"Species '{name}' not found in model") from e

    def get_state(self) -> np.ndarray:
        """Bulk-copy the full live species-concentration vector (GH #102).

        Returns a fresh ``float64`` array of length :attr:`n_species`, ordered
        like :attr:`species_names`. This is the low-overhead per-step
        state-exchange primitive for driving bngsim as a reaction kernel from an
        external orchestrator (e.g. a hybrid SSA/ODE splitting loop): one Python
        call marshals the entire state, so per-step exchange cost stays
        negligible next to the ODE solve even at ~100K species.

        See Also
        --------
        set_state : the inverse bulk assignment.
        species_names : the ordering of the returned vector.
        """
        return self._core.get_state()

    def set_state(self, state: np.ndarray) -> None:
        """Bulk-assign the full live species-concentration vector (GH #102).

        Parameters
        ----------
        state : array_like
            1-D array of length :attr:`n_species`, ordered like
            :attr:`species_names`. Copied into the model's live concentrations;
            observables and other derived state are recomputed on the next RHS
            or observable evaluation.

        Raises
        ------
        ValueError
            If ``state`` is not 1-D or its length differs from
            :attr:`n_species`.
        """
        self._core.set_state(np.asarray(state, dtype=np.float64))

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def n_species(self) -> int:
        """Number of species in the model."""
        return self._core.n_species

    @property
    def n_reactions(self) -> int:
        """Number of reactions in the model."""
        return self._core.n_reactions

    @property
    def n_observables(self) -> int:
        """Number of observable groups in the model."""
        return self._core.n_observables

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the model."""
        return self._core.n_parameters

    @property
    def n_functions(self) -> int:
        """Number of functions in the model."""
        return self._core.n_functions

    @property
    def param_names(self) -> list[str]:
        """List of all parameter names."""
        return self._core.param_names

    @property
    def param_is_expression(self) -> list[bool]:
        """Per-parameter ``is_expression`` flag, parallel to :attr:`param_names`.

        ``True`` for derived ``ConstantExpression`` parameters such as the
        ``_rateLaw{N}`` symbols BNG2.pl emits when a BNGL rate law is a
        compound expression (e.g. ``chi*kon``). These are not independent
        knobs — their values are computed from primary parameters and are
        re-evaluated automatically by :meth:`set_param`.
        """
        return list(self._core.param_is_expression)

    @property
    def primary_param_names(self) -> list[str]:
        """List of parameter names that are *not* derived constant expressions.

        These are the genuine knobs of the model — primary rate constants,
        initial-condition parameters, etc. Use this when you want to expose
        the model to an external optimizer or sampler that should treat
        each parameter as an independent variable; varying a primary via
        :meth:`set_param` automatically propagates to derived parameters.
        """
        names = self.param_names
        flags = self.param_is_expression
        return [n for n, f in zip(names, flags, strict=False) if not f]

    @property
    def species_names(self) -> list[str]:
        """List of all species names."""
        return self._core.species_names

    @property
    def observable_names(self) -> list[str]:
        """List of all observable group names."""
        return self._core.observable_names

    # ─── Table functions ──────────────────────────────────────────────────

    def add_table_function(
        self,
        name: str,
        *,
        file: str | Path | None = None,
        times: list[float] | None = None,
        values: list[float] | None = None,
        index: str = "time",
        method: str = "linear",
    ) -> None:
        """Add a table function (piecewise-linear interpolation of data).

        The function is registered with the expression evaluator and can be
        referenced by name in rate law expressions.

        Parameters
        ----------
        name : str
            Function name (e.g., ``"cumNcases"``).
        file : str or Path, optional
            Path to a ``.tfun`` file. Mutually exclusive with ``times``/``values``.
        times : list[float], optional
            X (index) values. Must be used with ``values``.
        values : list[float], optional
            Y (function) values. Must be used with ``times``.
        index : str
            Index variable name. Default ``"time"``. Can also be a parameter
            or observable name for non-time-indexed table functions.
        method : str
            Interpolation method: ``"linear"`` (default) or ``"step"``.

        Raises
        ------
        ModelError
            If the file cannot be read or data is invalid.
        ValueError
            If arguments are inconsistent (e.g., both ``file`` and ``times``).

        Examples
        --------
        >>> model.add_table_function("cumNcases", file="case_data.tfun")
        >>> model.add_table_function("response", file="dose.tfun", index="drug_conc")
        >>> model.add_table_function("drive", times=[0, 1, 2], values=[0, 5, 10])
        """
        if file is not None and times is not None:
            raise ValueError(
                "Cannot specify both 'file' and 'times'/'values'. Use one or the other."
            )
        normalized_method = method.strip().lower()
        if normalized_method not in {"linear", "step"}:
            raise ValueError(
                "Invalid table function interpolation method. Expected 'linear' or 'step'."
            )
        if file is not None:
            filepath = str(Path(file))
            try:
                self._core.add_table_function_file(name, filepath, index, normalized_method)
            except (ValueError, RuntimeError) as e:
                raise ModelError(f"Failed to add table function '{name}': {e}") from e
        elif times is not None and values is not None:
            try:
                self._core.add_table_function_arrays(
                    name, list(times), list(values), index, normalized_method
                )
            except (ValueError, RuntimeError) as e:
                raise ModelError(f"Failed to add table function '{name}': {e}") from e
        else:
            raise ValueError("Must specify either 'file' or both 'times' and 'values'.")

    @property
    def n_table_functions(self) -> int:
        """Number of registered table functions."""
        return self._core.n_table_functions

    @property
    def table_function_names(self) -> list[str]:
        """Names of all registered table functions."""
        return self._core.table_function_names

    # ─── Dunder methods ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Model(species={self.n_species}, reactions={self.n_reactions}, "
            f"observables={self.n_observables}, parameters={self.n_parameters})"
        )
