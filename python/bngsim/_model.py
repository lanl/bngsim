"""bngsim.Model — High-level Python wrapper for NetworkModel.

This class delegates to the C++ ``NetworkModel`` and provides Python-friendly
helpers for loading models, updating parameters, and inspecting model state.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, overload

import numpy as np

from bngsim._exceptions import (
    ModelError,
    ParameterError,
    SimulationError,
    UnderSpecifiedModelError,
)
from bngsim._extension_handle import raise_cannot_pickle, raise_cannot_shallow_copy


def _guard_function_expressions(core) -> list[tuple[str, str, str]]:
    """GH #333 rate-law guard, imported lazily.

    ``bngsim._jacobian`` pulls in ``bngsim._codegen``, so importing it at module
    scope would put both in front of every ``import bngsim``. Deferred to here,
    where it costs a ``sys.modules`` lookup per Model — and only for a model that
    has a function at all, since the guard's own substring gate exits first.
    """
    if not getattr(core, "n_functions", 0):
        return []
    from bngsim._jacobian import guard_function_expressions

    return guard_function_expressions(core)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from bngsim._bngsim_core import NetworkModel
    from bngsim.convert import ProtocolSpec

logger = logging.getLogger("bngsim")

# Suffix → factory-method name, for Model.load() dispatch. ``.sbml`` is accepted
# alongside ``.xml`` because SBML is served under both (BioModels uses .xml).
_LOAD_DISPATCH: dict[str, str] = {
    ".ant": "from_antimony",
    ".xml": "from_sbml",
    ".sbml": "from_sbml",
    ".net": "from_net",
    ".bngl": "from_bngl",
}


def _ic_carrier_name(species_id: str) -> str:
    """The synthetic parameter the SBML loader declares to carry a compound
    ``<initialAssignment>`` on ``species_id`` (issue #147). It is seeded like any
    derived parameter (issue #715), and never reported as a model symbol."""
    from bngsim._sbml_loader import _safe_name

    return _safe_name(f"_ic_{species_id}")


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
        "_codegen_attachment",
        "_codegen_for_sens",
        "_codegen_sec",
        "_codegen_cache_hit",
        "_codegen_sens_decline",
        "_libsbml_parse_sec",
        "_interpret_sec",
        "_jac_derive_sec",
        "_jac_attempted",
        "_jac_decline",
        "_net_path",
        "_ssa_issues",
        "_ar_report_map",
        "_varvol_conc_map",
        "_varvol_amount_map",
        "_varvol_ar_conc_map",
        "_varvol_ar_amount_map",
        "_varvol_target_amount_map",
        "_varvol_event_resize_map",
        "_periodic_disc_max_step",
        "_time_disc_conditions",
        "_event_disc_conditions",
        "_derived_time_disc_conditions",
        "_derived_time_disc_clocks",
        "_clock_probe_can_matter",
        "_clock_volume_params",
        "_time_disc_memo",
        "_state_switch_root_conditions",
        "_state_switch_root_covered",
        "_state_switch_root_memo",
        "_ssa_reads_clock",
        "_table_breaks",
        "_ssa_clock_functions",
        "_want_output_sens",
        "_output_sens_analysis",
        "_output_sens_family",
        "_table_function_bindings",
        "_named_conc_states",
        "_named_sens_seeds",
        "_declared_ic_sens",
        "_ic_write_log",
        "_guarded_functions",
        "_stoich_coo",
        "_volume_factors_memo",
        "_frozen_params",
        "_loaded_params",
        "_loaded_state",
    )

    def __init__(
        self, _core: NetworkModel, _guarded: list[tuple[str, str, str]] | None = None
    ) -> None:
        self._core = _core
        # Every parameter's value as the model was built, which a network-free
        # run compares the model's values with to find the ones set since, and
        # hands its backend (issue #1360, _network_free_params). A clone keeps
        # its parent's.
        self._loaded_params: dict[str, float] = {
            name: float(_core.get_param(name)) for name in _core.param_names
        }
        # And every species' amount, which it hands the backend the same way
        # where the model has changed it (issue #1373).
        self._loaded_state: tuple[float, ...] = tuple(float(x) for x in _core.get_state())
        self._codegen_so_path: str = ""
        # GH #198: whether codegen should emit the expression output-sensitivity
        # evaluator. Set by the Simulator before codegen prep (only a sensitivity
        # run needs it, since its build-time differentiation is expensive).
        self._want_output_sens: bool = False
        # GH #97: ``(key, analysis)`` memo for the #198 per-function output-sens
        # analysis (``_codegen._analyze_output_sens``), which both the C emitter
        # and the Result's support map run. Shared, so nothing may mutate it, and
        # keyed (``_codegen._output_sens_analysis_key``) so a budget override does
        # not read back an analysis made under a different one.
        self._output_sens_analysis: tuple | None = None
        # Issue #979: ``{key: analysis}``, the same analyses as a model and all
        # of its clones have made them, for the next of them to take. One dict,
        # which ``clone`` hands on.
        self._output_sens_family: dict[tuple, dict] = {}
        # Issue #979: ``(names, bindings)``, what each table function is read
        # over, as ``codegen_data()`` last gave it for that list of names
        # (``_codegen._table_function_bindings``).
        self._table_function_bindings: tuple | None = None
        # In-process MIR micro-JIT codegen source (GH #78); set when the JIT
        # backend (BNGSIM_CODEGEN_JIT=mir) prepares codegen for this model.
        self._codegen_c_source: str = ""
        # Issue #708: what the artifact above was built for, beyond the model's
        # structure: which parameters were attached to an expression
        # (``tuple(param_is_expression)``) and what the switch gate made of the
        # rate-law conditions at the parameter values of the time. The
        # generated sensitivity code carries the chain rule through each
        # attached parameter, and has an analytic right-hand side or not by the
        # gate's verdict, so an artifact built for one is the wrong code after
        # a ``set_param`` changes either. ``None`` with no artifact.
        self._codegen_attachment: tuple | None = None
        # Whether that artifact was built for a sensitivity run. Kept beside
        # it: ``_want_output_sens`` above is what the next build is to emit,
        # and every Simulator built on the model writes it.
        self._codegen_for_sens: bool = False
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
        # Why the model's last codegen prepare declined the analytic sensitivity
        # RHS, or None when it did not (issue #438). Set by the _codegen.prepare_*
        # entry points; read by Simulator.sens_rhs_decline_reason for a build that
        # has no .so to carry the decline note beside (the MIR JIT paths).
        self._codegen_sens_decline: str | None = None
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
        # Issues #506/#538: why the attempt declined, {"kind", "reason", ...}, for
        # analytical_jacobian_status and analytical_jacobian_decline.
        self._jac_decline: dict | None = None
        # Set by Model.from_net: the .net file the model was loaded from.
        # Provenance only since #803: every backend reads the built model. It
        # used to select the .net codegen path (kept for "issue #15's
        # derived-parameter chain rules", which the model path expands since
        # #99), and jacobian="jax" re-read the file it named, until step 4.
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
        # initial value (the species is emitted ``fixed``). Empty for non-AR
        # models. Set by the SBML loader (see _sbml_loader.py section 11) and
        # rebuilt by from_net from the network's own structure (#515).
        self._ar_report_map: dict[str, tuple] = {}
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
        # Populated by the SBML loader (issue #728): a concentration species that
        # a rate rule or an assignment rule sets, in a compartment a rate rule, an
        # assignment rule or an event sizes, mapped to ``(kind, column)``: where
        # its live size is read, ``"species"``, ``"expression"`` or
        # ``"observable"``. Its column holds the rule's concentration under every
        # method, so its amount is that times V_live(t).
        # Simulator._apply_varvol_target_amount_map records V_live(t) for
        # Result.as_roadrunner's bare-id selector.
        self._varvol_target_amount_map: dict[str, tuple[str, str]] = {}
        # Populated by the SBML loader (GH #131): maps a mangled species name in
        # an EVENT-RESIZED compartment to ``(comp_obs_name, V_static, hOSU)``.
        # Simulator._apply_varvol_event_resize_map applies the report-time
        # concentration correction ``× V_static/V_live`` — for every hOSU=true
        # species (both ODE and SSA) and for hOSU=false species under SSA only —
        # reading V_live from the compartment's same-named observable column. A
        # concentration species a rule sets is left to _varvol_target_amount_map.
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
        # Issue #305: the GH #72 discontinuity-trigger conditions this model
        # registered, verbatim, as source text. Empty for every model with no
        # time-dependent piecewise. Simulator.run resolves the fixed-time ones
        # to crossing times and stops the step on each, because a registered
        # root is only reachable on a step CVODE accepts. See _sbml_loader.py.
        self._time_disc_conditions: tuple[str, ...] = ()
        # The ones an event trigger reads (issue #714): a touch of the threshold
        # is a stop for these alone.
        self._event_disc_conditions: frozenset[str] = frozenset()
        # Issue #440: the same thing for a model whose loader registered
        # nothing, derived from the built core on first ask and cached here.
        # None means "not looked yet"; a tuple (often empty) means the scan has
        # run. See time_discontinuity_conditions().
        self._derived_time_disc_conditions: tuple[str, ...] | None = None
        # Issue #726: the parameter-dependent inputs it was derived at (see
        # _clock_key()). A counter is a clock by its rate's value, so the scan
        # is done again when that moves.
        self._derived_time_disc_clocks: object = None
        # Whether the scan can read a clock at all, from the texts: structural.
        self._clock_probe_can_matter: bool | None = None
        # Amount-valued species -> the live compartment-size parameter their
        # amount is read through; structural, read once a probe finds a clock.
        self._clock_volume_params: dict[int, str] | None = None
        # Every scan made, by key, shared with clones: a batch whose rows sit at
        # another clock verdict than the parent scans that verdict once.
        self._time_disc_memo: dict = {}
        # Issue #897: the rate-law conditions over model state that a run without
        # sensitivities roots on, derived on first ask and cached like the line
        # above. See state_switch_root_conditions().
        self._state_switch_root_conditions: tuple[str, ...] | None = None
        # ...the time conditions it left out as covered (issue #726), and every
        # answer by that covered set, shared with clones like the scan's.
        self._state_switch_root_covered: object = None
        self._state_switch_root_memo: dict = {}
        # Issue #719: whether an SSA run needs breakpoints; structural, so once.
        self._ssa_reads_clock: bool | None = None
        # Issue #709: where the model's tables break, with how many tables it
        # held when that was read. See table_function_breaks().
        self._table_breaks: tuple[int, list] | None = None
        self._ssa_clock_functions: list[tuple[str, str]] | None = None
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
        # Issue #81: each named snapshot's forward-sensitivity seed dx/dθ, when it
        # had one at save time — i.e. when the snapshot is a pre-equilibrated
        # state whose ∂x/∂θ is nonzero. Maps the same label to
        # (seed (n_species, n_params), param_names), so restore_concentrations()
        # puts back the state AND its θ-derivative; a state restored without it
        # would be re-seeded as a fresh start (∂x(0)/∂θ = 0), which is wrong
        # rather than approximate. Labels with no seed are simply absent.
        self._named_sens_seeds: dict[str, tuple[np.ndarray, list[str]]] = {}
        # Issue #111: ∂x_k(0)/∂θ declared by a per-point hook for the initial
        # conditions it assigns, ``{species: {param: value}}``. Written by
        # declare_ic_sensitivity(); read and cleared per point by
        # Simulator.parameter_scan, which uses a declared row verbatim instead of
        # measuring it through the hook.
        self._declared_ic_sens: dict[str, dict[str, float]] = {}
        # Issue #111: when not None, the set of species whose concentration has
        # been assigned since logging began (``"*"`` for a bulk write that
        # rewrites every species). Armed by Simulator.parameter_scan around an
        # on_point hook so it knows exactly which initial conditions the hook
        # assigned — that is the set whose ∂x_k(0)/∂θ is no longer the carried
        # derivative. Never observable to a caller that does not arm it.
        self._ic_write_log: set[str] | None = None
        # GH #333: rewrite any rate law of the form ``base^exp · ln(base)`` to its
        # limit at ``base == 0``, so the *value* path agrees with the derivative
        # path that #310/#317 already guard. Applied here rather than in either
        # loader because this constructor is the single funnel every input format
        # reaches (``.net``, SBML, ``_net_reader``, ``coupling``), and because a
        # ``.net`` model is built entirely in C++, leaving no earlier seam. Gated
        # on a substring test for a logarithm, so a model without one — 97.9% of
        # the corpus — pays nothing and never touches sympy.
        # A clone is handed its parent's list (issue #979): the core it is made
        # on has the parent's functions as the guard left them, and deciding
        # again could decide otherwise (the guard gives up on any exception,
        # running out of stack among them), which would leave two models of
        # one family with different functions and one memo of their analysis.
        self._guarded_functions: list[tuple[str, str, str]] = (
            _guard_function_expressions(_core) if _guarded is None else list(_guarded)
        )
        # Issue #523: the 0-based COO form of the stoichiometry, converted once.
        self._stoich_coo: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # Issues #697, #743: (compartment size names, their values, the reported
        # volume factors read at them), for Simulator._get_volume_factors.
        self._volume_factors_memo: (
            tuple[tuple[str, ...], tuple[float, ...], list[float]] | None
        ) = None
        # Issues #313, #695, #696: parameters the loader folded to a number,
        # each with what it was folded into. See :attr:`frozen_params`.
        self._frozen_params: dict[str, str] = {}

    # ─── Factory methods ──────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        defer_jacobian: bool | None = None,
        compartment_sizes: dict[str, float] | None = None,
    ) -> Model:
        """Load a model from a file, dispatching on its suffix.

        A single entry point over the format-specific factories, so callers who
        already know the path do not have to know the format:

        =============  =====================
        Suffix         Factory
        =============  =====================
        ``.ant``       :meth:`from_antimony`
        ``.xml``       :meth:`from_sbml`
        ``.sbml``      :meth:`from_sbml`
        ``.net``       :meth:`from_net`
        ``.bngl``      :meth:`from_bngl`
        =============  =====================

        Matching is case-insensitive. ``.bngl`` needs BNG2.pl on the machine
        (``pip install 'bngsim[bngl]'``) because bngsim has no BNGL parser — see
        :meth:`from_bngl`, and note that extra is not the parity_checks/
        ``parity`` group, which pins an exact PyBioNetGen commit for
        engine-routing provenance rather than for BNG2.pl.

        Parameters
        ----------
        path : str or Path
            Path to the model file.
        defer_jacobian : bool, optional
            GH #145 escape hatch, forwarded to the selected factory (see
            :meth:`from_sbml`). Default lazy; ``defer_jacobian=False`` derives
            the analytical Functional Jacobian eagerly at load.
        compartment_sizes : dict[str, float], optional
            Compartment id → volume, forwarded to the SBML-family factories
            (issue #164 — see :meth:`from_sbml`). Rejected for ``.net`` and
            ``.bngl``, whose volumes BNG2.pl folds into the network's rate
            constants during network generation.

        Returns
        -------
        Model
            The loaded model.

        Raises
        ------
        ImportError
            If the format's optional dependency is not installed (``antimony``
            for ``.ant`` — ``pip install 'bngsim[antimony]'``).
        FileNotFoundError
            If the file does not exist.
        ModelError
            If the suffix is not a loadable format, the file cannot be parsed,
            ``compartment_sizes`` is given for a ``.net``/``.bngl`` model, or
            BNG2.pl is unavailable for a ``.bngl`` one.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        factory = _LOAD_DISPATCH.get(suffix)
        if factory is None:
            known = ", ".join(sorted(_LOAD_DISPATCH))
            raise ModelError(
                f"Cannot infer a model format from {path.name!r} "
                f"(suffix {suffix or 'missing'!r}); expected one of: {known}. "
                f"Use the format-specific factory to load it explicitly."
            )
        if compartment_sizes and factory in ("from_net", "from_bngl"):
            # Post-BNG2.pl, the compartment volumes are already folded into the
            # network's rate constants and there is no compartment left to
            # override. Say so rather than accept a dict that would do nothing
            # (issue #164 exists because a silently-dropped volume write is the
            # expensive failure). A .bngl still HAS a `begin compartments` block,
            # but it is BNG2.pl that bakes it and the override would have to
            # happen in the source — so the honest remedy is the same one.
            kind = ".net" if factory == "from_net" else ".bngl"
            raise ModelError(
                f"compartment_sizes= is not supported for {kind} models ({path.name}): "
                f"BNG2.pl folds compartment volumes into the generated network's rate "
                f"constants, so the loaded model has no compartment size to set. "
                f"Regenerate the network from BNGL at the volume you want."
            )
        if factory == "from_bngl":
            return cls.from_bngl(path, defer_jacobian=defer_jacobian)
        if factory == "from_antimony":
            # from_antimony takes no defer_jacobian (it routes through the SBML
            # string loader, which is lazy); apply the eager hatch exactly as
            # from_sbml does so load() behaves uniformly across formats.
            model = cls.from_antimony(path, compartment_sizes=compartment_sizes)
            if defer_jacobian is False:
                model.prepare_analytical_jacobian()
            return model
        if factory == "from_net":
            return cls.from_net(path, defer_jacobian=defer_jacobian)
        return getattr(cls, factory)(
            path, defer_jacobian=defer_jacobian, compartment_sizes=compartment_sizes
        )

    @classmethod
    def from_antimony(
        cls, path: str | Path, *, compartment_sizes: dict[str, float] | None = None
    ) -> Model:
        """Load a model from an Antimony ``.ant`` file.

        Antimony is a human-readable model description language.
        Internally converts to SBML via libantimony, then loads
        via libsbml for correct SBML semantics.

        Requires: ``pip install antimony python-libsbml``

        Parameters
        ----------
        path : str or Path
            Path to the ``.ant`` file.
        compartment_sizes : dict[str, float], optional
            Compartment id → volume, applied to the converted SBML before
            interpretation (issue #164). See :meth:`from_sbml`.

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
            return load_antimony_via_sbml(path, compartment_sizes)
        except (ImportError, FileNotFoundError):
            raise
        except UnderSpecifiedModelError:
            raise  # typed refusal — see from_sbml (Antimony loads via SBML)
        except Exception as e:
            raise ModelError(f"Failed to load Antimony file {path}: {e}") from e

    @classmethod
    def from_antimony_string(
        cls, text: str, *, compartment_sizes: dict[str, float] | None = None
    ) -> Model:
        """Load a model from an Antimony string.

        Parameters
        ----------
        text : str
            Antimony model text.
        compartment_sizes : dict[str, float], optional
            Compartment id → volume, applied to the converted SBML before
            interpretation (issue #164). See :meth:`from_sbml`.

        Returns
        -------
        Model
            The loaded model.
        """
        from bngsim._sbml_loader import load_antimony_string_via_sbml

        try:
            return load_antimony_string_via_sbml(text, compartment_sizes)
        except ImportError:
            raise
        except UnderSpecifiedModelError:
            raise  # typed refusal — see from_sbml (Antimony loads via SBML)
        except Exception as e:
            raise ModelError(f"Failed to load Antimony string: {e}") from e

    @classmethod
    def from_sbml(
        cls,
        path: str | Path,
        *,
        defer_jacobian: bool | None = None,
        compartment_sizes: dict[str, float] | None = None,
    ) -> Model:
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
        compartment_sizes : dict[str, float], optional
            Compartment id → volume, applied to the parsed document before
            bngsim interprets it (issue #164). This is how a compartment volume
            is changed: :meth:`set_param` **refuses** a compartment-size write,
            because the size is folded at load into constants the write cannot
            reach — per-species volume factors, amount-declared initial
            conditions, mass-action rate constants, SSA propensity volumes, and
            the emitted RHS/sensitivity sources. Overriding here moves the size
            *before* any of those are derived, so the result is the model you
            would get by editing the ``size=`` attribute in the file. Scan or
            fit a volume by looping over loads:

            >>> for v in [1.0, 2.0, 4.0]:  # doctest: +SKIP
            ...     m = Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v})

            Ids are SBML ids as the document carries them (before bngsim's
            identifier mangling, after any ``comp`` flattening). An
            ``initialAssignment`` on an overridden compartment is dropped, since
            it would otherwise take precedence. A compartment whose size an
            *assignment rule* computes is refused — the rule, not the attribute,
            is its volume.

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
            If the file cannot be parsed, or ``compartment_sizes`` names an
            unknown compartment, a non-positive size, or an assignment-rule
            compartment.
        """
        from bngsim._sbml_loader import load_sbml

        try:
            model = load_sbml(path, compartment_sizes)
        except (ImportError, FileNotFoundError):
            raise
        except UnderSpecifiedModelError:
            # Already the precise, typed refusal (issue #323): the model reads a
            # symbol it never defines. Re-wrapping it as a generic ModelError
            # would erase the distinction a caller — and the parity taxonomy —
            # uses to tell a documented refusal from an actionable bug. Same
            # reasoning as run()'s SensitivityUnsupportedError pass-through.
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
    def from_sbml_string(
        cls,
        text: str,
        *,
        defer_jacobian: bool | None = None,
        compartment_sizes: dict[str, float] | None = None,
    ) -> Model:
        """Load a model from an SBML XML string.

        Parameters
        ----------
        text : str
            SBML XML text.
        defer_jacobian : bool, optional
            GH #145 escape hatch (see :meth:`from_sbml`). Default lazy; pass
            ``defer_jacobian=False`` (or set ``BNGSIM_EAGER_JACOBIAN=1``) to
            derive the analytical Functional Jacobian eagerly at load.
        compartment_sizes : dict[str, float], optional
            Compartment id → volume, applied before interpretation — the
            supported way to change a volume, since :meth:`set_param` refuses a
            compartment-size write (issue #164). See :meth:`from_sbml`.

        Returns
        -------
        Model
            The loaded model.
        """
        from bngsim._sbml_loader import load_sbml_string

        try:
            model = load_sbml_string(text, compartment_sizes)
        except ImportError:
            raise
        except UnderSpecifiedModelError:
            raise  # typed refusal — see from_sbml
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
        m._ar_report_map = _ar_report_map_from_net(core)
        m._frozen_params = _unit_conversion_folds(_unit_conversions_in(path), core)
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

    # The `protocol` flag changes the return type, so it is overloaded rather
    # than left as a union every caller has to narrow — `Model.load` dispatching
    # here would otherwise have to cast.
    @overload
    @classmethod
    def from_bngl(
        cls,
        path: str | Path,
        *,
        protocol: Literal[False] = ...,
        bng2_pl: str | Path | None = ...,
        net_out: str | Path | None = ...,
        timeout: int = ...,
        cache: bool = ...,
        defer_jacobian: bool | None = ...,
    ) -> Model: ...

    @overload
    @classmethod
    def from_bngl(
        cls,
        path: str | Path,
        *,
        protocol: Literal[True],
        bng2_pl: str | Path | None = ...,
        net_out: str | Path | None = ...,
        timeout: int = ...,
        cache: bool = ...,
        defer_jacobian: bool | None = ...,
    ) -> tuple[Model, ProtocolSpec]: ...

    @classmethod
    def from_bngl(
        cls,
        path: str | Path,
        *,
        protocol: bool = False,
        bng2_pl: str | Path | None = None,
        net_out: str | Path | None = None,
        timeout: int = 600,
        cache: bool = True,
        defer_jacobian: bool | None = None,
    ) -> Model | tuple[Model, ProtocolSpec]:
        """Load a rule-based BNGL (``.bngl``) model, via BNG2.pl (GH #162).

        bngsim simulates reaction *networks*, and BNGL describes *rules*, so this
        is not a parser: the model block is handed to ``BNG2.pl
        generate_network``, and the ``.net`` it emits is loaded through
        :meth:`from_net`. That makes BNG2.pl a requirement — ``pip install
        'bngsim[bngl]'`` supplies it (PyBioNetGen bundles it), or point
        ``$BNGPATH``/``$BNG2_PL`` at a BioNetGen you already have. It is a Perl
        script, so ``perl`` must be on ``PATH`` too (macOS and most Linux
        distributions ship one; stock Windows does not).

        The file's **actions block is not executed.** A ``.bngl`` typically ends
        in ``simulate({...})`` or ``parameter_scan({...})``, and running the file
        as written would run the author's entire experiment just to obtain a
        network. Only ``generate_network`` is run — carrying the source's own
        ``max_iter``/``max_agg``/``max_stoich``, since those are what make an
        unbounded rule set finite. Pass ``protocol=True`` to get the actions back
        as a :class:`~bngsim.convert.ProtocolSpec` rather than lose them.

        Parameters
        ----------
        path : str or Path
            Path to the ``.bngl`` file.
        protocol : bool
            Return ``(model, ProtocolSpec)`` instead of just the model, the spec
            being the parsed actions block (:func:`bngsim.convert.parse_bngl_protocol`
            in non-strict mode, so a fidelity-affecting action is recorded in
            ``spec.lossy`` and warned about rather than raising).
        bng2_pl : str or Path, optional
            An explicit BioNetGen folder or ``BNG2.pl`` path, outranking
            ``$BNG2_PL``, ``$BNGPATH``, ``PATH`` and an installed PyBioNetGen.
        net_out : str or Path, optional
            Write the generated network here and load it from there, instead of
            from the cache. The way to keep the ``.net`` as an artifact.
        timeout : int
            Seconds to allow ``generate_network`` (default 600). A rule set whose
            network is unbounded hits this.
        cache : bool
            Reuse a previously generated network for unchanged BNGL (default).
            The cache is content-addressed under ``~/.cache/bngsim/networks``
            (override with ``$BNGSIM_BNGL_CACHE_DIR``), keyed on the flattened
            model text *and* the BNG2.pl that produced it, so it cannot go stale.
            ``False`` regenerates into a per-process directory.
        defer_jacobian : bool, optional
            GH #145 escape hatch, forwarded to :meth:`from_net`.

        Returns
        -------
        Model or tuple[Model, ProtocolSpec]
            The loaded model, and the actions block when ``protocol=True``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ModelError
            If BNG2.pl (or ``perl``) cannot be found — with every mechanism that
            was consulted named — or if ``generate_network`` fails or times out,
            with the tail of BNG2.pl's own output attached.

        Notes
        -----
        Compartmental BNGL loads: BNG2.pl bakes each compartment's volume into
        the generated rate constants, exactly as for a hand-generated ``.net``.
        For the same reason there is no ``compartment_sizes=`` here — the volume
        has to change in the BNGL source, before generation.

        Examples
        --------
        >>> model = bngsim.Model.from_bngl("egfr.bngl")           # doctest: +SKIP
        >>> model, spec = bngsim.Model.from_bngl(                  # doctest: +SKIP
        ...     "egfr.bngl", protocol=True)
        >>> [e.t_end for e in spec.experiments]                    # doctest: +SKIP
        [100.0]
        """
        from bngsim._bngl_loader import load_bngl

        model = load_bngl(
            path,
            bng2_pl=bng2_pl,
            net_out=net_out,
            timeout=timeout,
            cache=cache,
            defer_jacobian=defer_jacobian,
        )
        if not protocol:
            return model

        from bngsim.convert import parse_bngl_protocol

        return model, parse_bngl_protocol(Path(path), strict=False)

    # ─── Time-discontinuity conditions (GH #72, issue #440) ───────────────

    def time_discontinuity_conditions(self) -> tuple[str, ...]:
        """The branch conditions this model switches on simulation time with.

        What :meth:`Simulator.run` resolves to crossing *times* and stops the
        integration step on, so an ``if(time() >= 100, ...)`` window cannot be
        stepped over (issue #305 for the mechanism, issue #440 for why a BNGL
        model needs it).

        Two sources, in that order. The SBML loader walks the document at load
        time and registers each condition it finds as a CVODE root, and hands
        the registered set over verbatim — reading it back here rather than
        re-deriving one is what keeps the stops a subset of the roots. A
        ``.net``/BNGL model is built entirely in C++, so its loader has no seam
        to register a root at and there is nothing to read back; the conditions
        are recovered from the built model's own function bodies instead, and
        only when something asks.

        The recovery is cached, keyed by what it reads from parameter values
        (issue #726): which species pass the unit-rate clock probes at the
        model's current values, and the size of the live compartment each such
        amount-valued species is read through. A BNGL counter fed at ``k_t`` is
        a clock at ``k_t = 1`` and not otherwise, so its thresholds are time
        conditions at one value and not at another; an SBML substance-only
        counter fed at its compartment's size is one only while that size is 1.
        Cached once, a model first run at ``k_t = 0`` and then at 1 got no stop
        at a clock-gated window, and one first run at 1 and then at 0.5 got
        neither a stop nor a root. Every recovery is kept by its key and shared
        with clones, so a batch whose rows sit at another verdict than the
        parent's recovers that verdict once.

        A recovered entry is not always a condition. Since issue #869 it can
        also be a bare step call of a clock, such as ``floor((time()-t0)/P)``,
        with no comparison in it, and since issue #904 a call that bends
        (``max(0, 1 - abs(time()-t0)/w)``). Such a call jumps or turns the rate
        law itself, and each jump and turn is a stop.

        The recovery runs for a model that registered conditions too, and what
        it finds beside them is added. The loader registers comparisons, so a
        ``floor`` pulse or a ``max`` pulse standing beside a ``piecewise`` in
        an SBML model had no entry at all, and was stepped over where the same
        pulse alone was not.
        """
        key = self._clock_key()
        if self._derived_time_disc_conditions is None or self._derived_time_disc_clocks != key:
            derived = self._time_disc_memo.get(key)
            if derived is None:
                try:
                    from bngsim._switch_sensitivity import (
                        _condition_key,
                        _monotone_pieces,
                        _strip_redundant_parens,
                        time_discontinuity_conditions,
                    )

                    def spelled(c: str) -> str:
                        # A condition reached through a function definition is
                        # spelled `(time())` where the same one written directly
                        # reads `time()`, and the loader registers one spelling
                        # for both (issue #1082); compared on the same key, so
                        # the one registered covers the other, which the
                        # recovery otherwise added again as a stop.
                        return _strip_redundant_parens(_condition_key(c)).replace(" ", "")

                    registered = {spelled(c) for c in self._time_disc_conditions}
                    # The loader registers a turning comparison as its two
                    # edges (issue #904), `abs(time()-5) < 0.01` as `time()-5`
                    # against 0.01 and -0.01, so one found whole is registered
                    # where both of its edges are.
                    derived = tuple(
                        c
                        for c in time_discontinuity_conditions(
                            self._core, clock_candidates=None if key is None else key[0]
                        )
                        if not all(spelled(p) in registered for p in _monotone_pieces(c) or [c])
                    )
                except Exception as e:  # pragma: no cover - defensive
                    # Recovery is best-effort: without it a model keeps the
                    # stepping it had before this existed, which is correct
                    # wherever it does not step over a switch.
                    logger.debug("Time-discontinuity scan unavailable (%s)", e)
                    derived = ()
                _remember(self._time_disc_memo, key, derived)
            # The answer under a key that matches nothing, then the key: cut
            # short anywhere here, the next ask recovers again, rather than
            # reading one verdict's answer as another's.
            self._derived_time_disc_clocks = _UNKEYED
            self._derived_time_disc_conditions = derived
            self._derived_time_disc_clocks = key
        if self._time_disc_conditions:
            return (*self._time_disc_conditions, *self._derived_time_disc_conditions)
        return self._derived_time_disc_conditions

    def _clock_key(self) -> tuple | None:
        """What :meth:`time_discontinuity_conditions`' recovery reads from
        parameter values, as its cache key (issue #726).

        ``(candidates, sizes, n_tables)``: the species at ``dc/dt = 1`` on both
        RHS probes, whether the live compartment size each amount-valued one
        among them is read through is 1 (the recovery asks nothing else of it,
        so a fit over the size does not rescan per value), and how many table
        functions the model has. What the recovery does with the candidates
        past that reads structure only.
        Where no text spells a switch the recovery never probes, and neither
        does this. ``None`` where the probe itself failed, which the recovery's
        own fallback also reads as "no clock".
        """
        from bngsim._switch_sensitivity import (
            clock_volume_params,
            texts_spell_a_switch,
            unit_rate_clock_candidates,
        )

        core = self._core
        try:
            if self._clock_probe_can_matter is None:
                self._clock_probe_can_matter = texts_spell_a_switch(core)
            if not self._clock_probe_can_matter:
                return (frozenset(), (), core.n_table_functions)
            candidates = unit_rate_clock_candidates(core)
            sizes: tuple = ()
            if candidates:
                if self._clock_volume_params is None:
                    self._clock_volume_params = clock_volume_params(core)
                live = self._clock_volume_params
                sizes = tuple(
                    (i, core.get_param(live[i]) == 1.0) for i in sorted(candidates) if i in live
                )
            return (candidates, sizes, core.n_table_functions)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Clock probe for the time-condition cache unavailable (%s)", e)
            return None

    def state_switch_root_conditions(self) -> tuple[str, ...]:
        """The rate-law conditions over model state a plain run roots on (issue #897).

        A rate law switched by a comparison that reads the state, such as
        ``if((time() >= 4*S) && (time() < 4*S + w), k, 0)``, has crossings at
        times the state decides, so no stop can be placed at them before the
        run. :meth:`Simulator.run` registers each one as a CVODE root instead,
        which stops the step at the crossing rather than letting it span a
        narrow window and miss it. A step or bend call over state, such as
        ``floor(T/100)`` or ``max(0, 1 - abs(T-50)/w)``, is rooted where it
        jumps or bends, since it spells the same pulse with no comparison
        (issue #1068).

        The crossings :meth:`time_discontinuity_conditions` already covers are
        left out, and so is every condition in a function that no rate law
        reads. Past those, the answer depends on the model's structure alone,
        so it is cached for the covered set it was derived beside, and a clone
        inherits it. The covered set moves with a clock's rate (issue #726): a
        threshold on a counter is covered while the counter is a clock and has
        to be rooted once it is not.
        """
        return self._state_switch_roots_beside(self.time_discontinuity_conditions())

    def _state_switch_roots_beside(self, covered: tuple[str, ...]) -> tuple[str, ...]:
        """:meth:`state_switch_root_conditions` beside a ``covered`` the caller
        has just read from :meth:`time_discontinuity_conditions`. A run has it
        in hand from placing its stops, and asking again would probe again."""
        if (
            self._state_switch_root_conditions is None
            or self._state_switch_root_covered != covered
        ):
            roots = self._state_switch_root_memo.get(covered)
            if roots is None:
                try:
                    from bngsim._switch_sensitivity import state_switch_root_conditions

                    roots, guards = state_switch_root_conditions(
                        self._core, covered=covered, with_guards=True
                    )
                    _remember(self._state_switch_root_memo, ("guards", covered), guards)
                except Exception as e:  # pragma: no cover - defensive
                    # Detection is best-effort: without it a model keeps the
                    # stepping it had before issue #897, which is correct
                    # wherever no step spans a whole state-gated window.
                    logger.warning(
                        "State-switch root detection failed (%s); a rate law switched by "
                        "a condition over model state will be integrated without a root "
                        "at its crossings (issue #897).",
                        e,
                    )
                    roots = ()
                _remember(self._state_switch_root_memo, covered, roots)
            # As in time_discontinuity_conditions(): unkeyed while it is written.
            self._state_switch_root_covered = _UNKEYED
            self._state_switch_root_conditions = roots
            self._state_switch_root_covered = covered
        return self._state_switch_root_conditions

    def _state_switch_guards_beside(self, covered: tuple[str, ...]) -> tuple[str, ...]:
        """Parallel to :meth:`_state_switch_roots_beside`: for each root, the
        ``if()`` branches it switches a rate law in, as an expression that reads
        1 where one of them is on, or ``""`` where it switches one outside any
        branch (issue #1147). A plain run steps on through a crossing whose
        every branch is off."""
        roots = self._state_switch_roots_beside(covered)
        guards = self._state_switch_root_memo.get(("guards", covered))
        if guards is None and roots:
            # The memo is shared with clones and cleared whole at its cap
            # (_remember), which can take the guards while this model still
            # holds its roots. Every later run would then restart at the
            # crossings the guards step through, so derive them again.
            try:
                from bngsim._switch_sensitivity import state_switch_root_conditions

                again, guards = state_switch_root_conditions(
                    self._core, covered=covered, with_guards=True
                )
            except Exception:  # pragma: no cover - defensive, as for the roots
                guards = None
            else:
                if tuple(again) == tuple(roots):
                    _remember(self._state_switch_root_memo, ("guards", covered), guards)
                else:
                    guards = None
        if guards is None or len(guards) != len(roots):
            return ("",) * len(roots)
        return guards

    def _state_switch_stops_beside(self, covered: tuple[str, ...]) -> tuple[str, ...]:
        """The roots :meth:`_state_switch_roots_beside` has for step and bend
        calls, which a run with sensitivities adds as stops (issue #1068).
        Kept in the same memo, beside the same ``covered``."""
        key = ("calls", covered)
        stops = self._state_switch_root_memo.get(key)
        if stops is None:
            try:
                from bngsim._switch_sensitivity import state_switch_root_conditions

                stops = state_switch_root_conditions(self._core, covered=covered, calls_only=True)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "State-switch root detection failed (%s); a rate law with a step or bend "
                    "call over model state will be integrated with sensitivities and "
                    "without a root where the call jumps or bends (issue #1068).",
                    e,
                )
                stops = ()
            _remember(self._state_switch_root_memo, key, stops)
        return stops

    def table_function_breaks(self) -> list:
        """Where the model's table functions break (issue #709): for each table
        with any, its index kind, the observable it is indexed by, the
        breakpoints at which a run restarts, and those at which it only ends a
        step.

        :meth:`Simulator.run` stops the integration on each one a table over
        time, or over a unit-rate counter, reaches in the run. Which
        breakpoints count is :func:`bngsim._switch_sensitivity.table_function_breaks`'s
        rule: a table of many samples of a smooth signal has none.

        Read from the tables alone, so it is kept until a table is added or
        given new data (:meth:`add_table_function` drops it), and a clone
        inherits it. A model with no table pays one attribute read.
        """
        n = self._core.n_table_functions
        if n == 0:
            return []
        if self._table_breaks is None or self._table_breaks[0] != n:
            from bngsim._switch_sensitivity import table_function_breaks

            self._table_breaks = (n, table_function_breaks(self._core))
        return self._table_breaks[1]

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
            from bngsim._jacobian import attach_functional_jacobian, last_decline

            t0 = time.perf_counter()
            attach_functional_jacobian(self._core)
            self._jac_derive_sec = time.perf_counter() - t0
            self._jac_decline = last_decline()
        except Exception as e:
            # attach_functional_jacobian is contractually no-raise (it falls back
            # to FD and logs over-budget / unsupported cases itself, GH #95); this
            # guard only surfaces a genuinely unexpected error without re-deriving.
            logger.debug("Analytical Functional Jacobian skipped: %s", e)
            self._jac_decline = {
                "kind": "unexpected_error",
                "reason": f"an unexpected error during derivation: {e}",
            }
        return bool(self._core.analytical_jacobian_complete)

    @property
    def analytical_jacobian_status(self) -> str:
        """Whether the model carries a complete analytical Jacobian, in words.

        ``"complete"`` when every reaction's derivative is closed-form — a
        mass-action model from the start, a model with functional rate laws
        once :meth:`prepare_analytical_jacobian` attached them;
        ``"pending"`` while a model with functional rate laws has not derived
        them yet (an ODE solve, or that call, derives them); and
        ``"declined: <reason>"`` when it runs on the finite-difference Jacobian,
        with the reason the derivation gave — the rate law and what stopped it,
        the derivation budget, the C++ attach gate's verdict (the entry it
        refused, with the probe state and the analytical and finite-difference
        values; issue #534), or the environment switch.
        The same reason goes to the ``bngsim`` logger at INFO when it happens;
        this is the readable form a harness can record instead of only the
        boolean :attr:`bngsim._bngsim_core.NetworkModel.analytical_jacobian_complete`
        (issue #506).
        """
        if bool(self._core.analytical_jacobian_complete):
            return "complete"
        if not self._jac_attempted:
            return "pending"
        return f"declined: {(self._jac_decline or _NO_DECLINE_RECORD)['reason']}"

    @property
    def analytical_jacobian_decline(self) -> dict | None:
        """Why the model runs on the finite-difference Jacobian, as a stable
        category a census can group by (issue #538).

        ``None`` while :attr:`analytical_jacobian_status` is ``"complete"`` or
        ``"pending"``. Otherwise a dict with ``kind``, one of
        ``bngsim._jacobian.DECLINE_KINDS``, and ``reason``, the status's text
        after ``"declined: "``. For the C++ attach gate's kinds (``fd_mismatch``,
        ``nonfinite_entry``, ...) it also carries the fields that kind's verdict
        sets — of ``rxn_idx``, ``per_observable``, ``target_idx``, ``row``,
        ``col``, ``probe``, ``analytical``, ``finite_difference``,
        ``n_nonfinite`` and ``detail`` — and the species ``row`` and ``col``
        name as ``row_species`` / ``col_species``. A field the kind does not
        set is absent, not a placeholder.

        The kind is what to group by: the reason embeds the rate law, names and
        values, and its wording changes. A clone carries it, as it carries the
        status.
        """
        if bool(self._core.analytical_jacobian_complete) or not self._jac_attempted:
            return None
        return dict(self._jac_decline or _NO_DECLINE_RECORD)

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
        species concentrations, and expression evaluator state. (What a model
        and its clones do hold together is a memo of analyses of their
        functions, which are the same for all of them: read-only once made,
        and kept under a lock.)

        This is also the answer to "give me my own copy" in general: all of a
        ``Model``'s state lives behind one handle into the compiled extension,
        so neither ``copy.copy`` nor ``copy.deepcopy`` can produce an
        independent model, and both refuse rather than hand back an alias
        (issue #643).

        Returns
        -------
        Model
            An independent deep copy.
        """
        m = Model(_core=self._core.clone(), _guarded=self._guarded_functions)
        m._loaded_params = dict(self._loaded_params)
        m._loaded_state = self._loaded_state
        m._net_path = self._net_path
        m._want_output_sens = self._want_output_sens
        m._codegen_so_path = self._codegen_so_path
        m._codegen_c_source = self._codegen_c_source
        m._codegen_attachment = self._codegen_attachment
        m._codegen_for_sens = self._codegen_for_sens
        m._codegen_sec = self._codegen_sec
        m._codegen_cache_hit = self._codegen_cache_hit
        m._codegen_sens_decline = self._codegen_sens_decline
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
        m._jac_decline = dict(self._jac_decline) if self._jac_decline is not None else None
        # GH #97: same warm-clone reasoning for the #198 output-sens analysis — a
        # clone has the parent's structure, so re-running its sympy would be N×
        # waste in parallel fitting. Shared by reference (the analysis is
        # read-only) and re-keyed on the clone's own counters, so a clone that
        # somehow did not match simply re-derives.
        m._output_sens_analysis = self._output_sens_analysis
        # Issue #979: and the family's memo itself, not its contents at the
        # time. An analysis made on a clone is there for the parent and for the
        # clones made after, so that cloning a model that was never run does
        # not start each clone from nothing.
        m._output_sens_family = self._output_sens_family
        # What the table functions are read over is the clone's too. A model
        # that has tables and was never asked is asked here, once, and not by
        # each of its clones in turn (a model with none is not asked at all).
        if self._core.n_table_functions:
            from bngsim._codegen import _table_function_bindings

            _table_function_bindings(self._core, self)
        m._table_function_bindings = self._table_function_bindings
        m._ssa_issues = list(self._ssa_issues)
        m._ar_report_map = dict(self._ar_report_map)
        m._varvol_conc_map = dict(self._varvol_conc_map)
        m._varvol_amount_map = dict(self._varvol_amount_map)
        m._varvol_ar_conc_map = dict(self._varvol_ar_conc_map)
        m._varvol_ar_amount_map = dict(self._varvol_ar_amount_map)
        m._varvol_target_amount_map = dict(self._varvol_target_amount_map)
        m._varvol_event_resize_map = dict(self._varvol_event_resize_map)
        m._periodic_disc_max_step = self._periodic_disc_max_step
        m._time_disc_conditions = self._time_disc_conditions
        m._event_disc_conditions = self._event_disc_conditions
        m._derived_time_disc_conditions = self._derived_time_disc_conditions
        m._derived_time_disc_clocks = self._derived_time_disc_clocks
        m._clock_probe_can_matter = self._clock_probe_can_matter
        m._clock_volume_params = self._clock_volume_params
        # Shared, not copied: an entry is a function of the structure, which a
        # clone has too, and of its key, so a row may answer for its siblings.
        m._time_disc_memo = self._time_disc_memo
        m._state_switch_root_conditions = self._state_switch_root_conditions
        m._state_switch_root_covered = self._state_switch_root_covered
        m._state_switch_root_memo = self._state_switch_root_memo
        m._ssa_reads_clock = self._ssa_reads_clock
        m._table_breaks = self._table_breaks
        m._ssa_clock_functions = self._ssa_clock_functions
        # Keyed on the sizes it was read at, so a clone that writes one reads it
        # again (issue #743).
        m._volume_factors_memo = self._volume_factors_memo
        m._frozen_params = dict(self._frozen_params)
        # Issue #11: carry named concentration snapshots to the clone, each a
        # fresh copy so the clone's restore can never alias the parent's stored
        # vector. (The default slot lives in the C++ core, deep-copied above.)
        m._named_conc_states = {k: v.copy() for k, v in self._named_conc_states.items()}
        # Issue #81: and each snapshot's dx/dθ, so the clone's restore is as
        # faithful as the parent's (the live/baseline seeds ride the C++ clone).
        m._named_sens_seeds = {
            k: (s.copy(), list(names)) for k, (s, names) in self._named_sens_seeds.items()
        }
        # Issue #111: pending IC-sensitivity declarations travel with the snapshot
        # they describe. The write log does not: it belongs to whoever armed it on
        # the original, and a clone is not inside that scope.
        m._declared_ic_sens = {k: dict(v) for k, v in self._declared_ic_sens.items()}
        m._ic_write_log = None
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

    def set_param(self, name: str, value: float, *, force_override: bool = False) -> None:
        """Set a parameter value by name.

        Parameters
        ----------
        name : str
            Parameter name (e.g. "kf", "Km").
        value : float
            New value.
        force_override : bool, optional
            Pin a **derived** parameter to ``value`` regardless of whether its
            expression currently produces that value, and permanently for this
            model object — neither :meth:`reset`, :meth:`clone`, nor a later
            write lifts it. The default (``False``) overrides only while the
            value differs from the expression's, which is what makes an ordinary
            write round-trip (issue #188, see Notes). Use this only when the
            caller's contract is "this parameter is an independent input"; the
            one such caller in bngsim is the function
            :func:`bngsim.jax.differentiable_solve` with ``flat=True``. Has no
            effect on a parameter that has no defining expression.

        Raises
        ------
        ParameterError
            If the parameter name is not found, or if ``name`` is one of
            :attr:`frozen_params` and ``value`` differs from the value it holds.
        ValueError
            If ``name`` is an SBML compartment size this model cannot resolve to
            a live volume, or a synthesized slot, and ``value`` differs from the
            value it holds. See Notes.

        Notes
        -----
        A compartment's value is folded at load into constants — per-species
        volume factors, amount-declared initial conditions, mass-action rate
        constants, SSA propensity volumes, the emitted RHS, and every
        ``<initialAssignment>`` that reads it. Issue #170 put the volume back
        into each fold it can reach, so most sizes are now ordinary writable
        parameters; the residue is refused by name rather than half-honoured,
        because honoring only some folds leaves the model internally
        inconsistent rather than moving the volume. They are listed in
        :attr:`unwritable_compartment_size_params`; load at the size you want
        instead (:meth:`from_sbml` accepts ``compartment_sizes={...}``). Writing
        the value it already holds is allowed, so round-tripping a full
        parameter vector still works.

        A *synthesized slot* — :attr:`param_is_internal` — is refused on the
        same unchanged-write rule. One of those is a **function's** name: the
        engine keeps the function's evaluated value in a parameter slot and
        rewrites it from the function's own expression before every derivative
        evaluation, so a write here is discarded at the next one. It used to be
        accepted — :meth:`get_param` echoed the new value back and the
        trajectory did not move (issue #227). Write the parameters the
        function's expression reads instead.

        Writing a parameter also re-derives every expression-valued parameter
        that reads it (BNG ``setParameter`` semantics), and — issue #79 — every
        **species initial condition that names one of them**. ``A() Stot`` in a
        ``.net`` species block, or an SBML ``initialAssignment`` that is a bare
        ``<ci>``, declares that species' initial condition to *be* the
        parameter, so a dose scan over a total amount moves it:

        >>> model.set_param("Stot", 1e6)
        >>> model.get_state()[0]        # doctest: +SKIP
        1000000.0

        The **declared** initial condition always follows, so :meth:`reset`
        rebuilds from current parameter values rather than from a snapshot taken
        at load. The **live** concentration follows only while the species is
        still sitting on that initial condition: a species a :meth:`Simulator.run` has
        advanced, or that :meth:`set_concentration` has assigned, keeps its
        value and picks the new initial condition up at the next :meth:`reset`.

        Two cases where the write deliberately does *not* reach an initial
        condition: after :meth:`save_concentrations` with no label (the baseline
        is now a captured state, which the declared initial condition no longer
        describes — dose such a protocol with :meth:`set_concentration`), and
        for an SBML ``initialAssignment`` too complex to be a single parameter
        reference (``2*init_X + offset``), which is evaluated once at load.

        Writing a **derived** parameter — one of the symbols in
        :attr:`param_is_expression`, ``d`` in ``d = d__FREE`` or a
        loader-synthesized ``_rateLaw{N}`` — *overrides* its expression rather
        than moving it: ``d`` stops tracking ``d__FREE``, which is BNG's
        ``setParameter`` semantics. The override is keyed on the value and so is
        reversible (issue #188): writing back the value the expression produces
        re-attaches it, and a write of the value the parameter already holds is
        not an override at all, so ``set_params(dict(zip(param_names, vec)))``
        round-trips a full parameter vector unchanged. Read the current state
        off :attr:`param_is_expression`, which goes ``False`` for the duration
        of an override while :attr:`param_expressions` keeps the defining
        expression.

        An override is a **structural** change and is meant to be: a derived
        parameter pinned to a literal no longer carries the chain rule from the
        primaries underneath it, so those primaries lose that reaction's term
        from their sensitivity columns, and the generated RHS changes to match.
        That is the correct derivative of the model you asked for — but it is
        why an accidental override used to be so hard to see, and why the
        round-trip above is the one to rely on.
        """
        self._raise_if_folded_at_load(name, float(value))
        try:
            self._core.set_param(name, float(value), force_override=force_override)
        except (KeyError, RuntimeError) as e:
            raise ParameterError(f"Parameter '{name}' not found in model") from e

    @property
    def frozen_params(self) -> list[str]:
        """Parameters that were folded to a number when the model was built
        (issues #313, #695, #696, #711).

        An SBML document may write a compartment's size, a stoichiometry or a
        conversion factor over its parameters, or set an initial value, of a
        parameter or of a species, by an initialAssignment that bngsim cannot
        keep symbolic. Each is evaluated once, at load, and the model holds
        the number. Every parameter that number was read from is listed: the
        ones the expression names, and the ones that gave each symbol it names
        its value at load (through another assignment, a rule, a reaction's
        kinetic law, a species' initial value or its compartment's size). So
        is what sized a compartment when a species' declared value was
        converted with it: a species declared in the unit it is not held in,
        in a compartment that a rate rule or an event resizes and an
        initialAssignment starts, or that an assignment rule sizes (issue
        #742; every name such a rule reads).

        A write to one does not move what it was folded into, so
        :meth:`set_param` refuses to change one, and a forward-sensitivity
        column for one is refused, where both used to be silent: the write took
        the value and left the number, and the column lacked everything that
        goes through it (an exact 0 where nothing else reads the parameter).
        Edit the document and load it again to move one.

        A compartmental BNGL model has them too (issue #711). BNG2.pl writes
        each reaction's volume factor into its rate constant as a number,
        ``0.1*kb`` for ``1/Ve`` at ``Ve = 10``, and leaves the expression in a
        comment, ``unit_conversion=1/Ve``. Every parameter such a comment names
        is listed, with what a derived parameter or a function among them
        reads: the network holds the number, and it is the BNGL source that
        has to change.

        Empty for most models.

        Returns
        -------
        list[str]
            Parameter names, in model parameter order.
        """
        frozen = self._frozen_params
        return [n for n in self.param_names if n in frozen]

    def _raise_if_folded_at_load(self, name: str, value: float) -> None:
        """Refuse a write that changes a parameter folded at load. A write of
        the value it holds is no change, which keeps a full-vector round trip
        working."""
        what = self._frozen_params.get(name)
        if what is None:
            return
        try:
            unchanged = value == self._core.get_param(name)
        except (KeyError, RuntimeError):
            return  # not a parameter: the write below says so
        if unchanged:
            return
        raise ParameterError(
            f"Parameter {name!r} cannot be changed after the model is loaded: it was read "
            f"once, when the model was built, for {what}, and the model holds the number "
            "that gave. A write would take the new value and move nothing it was folded "
            "into, which used to happen without a word (issues #313, #695, #696, #711). "
            f"Change it in {_where_a_folded_parameter_is_changed(what)}. "
            "Model.frozen_params lists every such parameter."
        )

    @property
    def compartment_size_params(self) -> list[str]:
        """Names of the parameters that are SBML compartment sizes.

        :meth:`set_param` **writes** these (issue #170): the storage convention
        a volume decides — the amount↔concentration conversion, an
        amount-declared initial condition, the mass-action scalar's ``Π V^n``,
        the SSA propensity volume — is re-derived from the parameter rather than
        left at the size the model happened to load at. What they are still not
        is *differentiable*: forward sensitivity refuses a ``d/dV`` column
        (issue #170 stage 3), which is why this list is worth having.

        A handful cannot be written even so; see
        :attr:`unwritable_compartment_size_params`.

        Empty for ``.net`` models, and for any compartment the SBML loader
        promoted to a species (rate-rule or event-resized): that one is live
        state, written with :meth:`set_concentration`, not a parameter.

        Returns
        -------
        list[str]
            Parameter names, in model parameter order.

        Examples
        --------
        >>> model = bngsim.Model.from_sbml("pbpk.xml")   # doctest: +SKIP
        >>> gradient_free = [p for p in model.param_names   # doctest: +SKIP
        ...                  if p not in set(model.compartment_size_params)]
        """
        try:
            flags = self._core.param_is_compartment_size
        except AttributeError:  # pragma: no cover - defensive
            return []
        # strict=: both lists come from the same `parameters()` vector, so a
        # length mismatch is a core/wrapper desync worth failing on, not
        # something to truncate past.
        return [n for n, f in zip(self.param_names, flags, strict=True) if f]

    @property
    def unwritable_compartment_size_params(self) -> list[str]:
        """Compartment sizes :meth:`set_param` still refuses to change (#170).

        The residue of issue #164's blanket refusal: a compartment whose size an
        **assignment rule** recomputes every step (a write would not survive the
        next evaluation); one whose storage divide a single mass-action scalar
        shares across **two compartments** that merely happen to have the same
        load-time size (that scalar stops being exact the moment they differ);
        and one an **initialAssignment** folds into a quantity no parameter
        holds — a species amount, a reaction rate — since SBML evaluates every
        such expression once, at load, against the sizes. All three are decided
        at load and named in the error.

        A subset of :attr:`compartment_size_params`, and usually empty — reload
        with ``Model.from_sbml(..., compartment_sizes={...})`` to move one.

        Returns
        -------
        list[str]
            Parameter names, in model parameter order.
        """
        try:
            flags = self._core.param_volume_write_refused
        except AttributeError:  # pragma: no cover - defensive
            return []
        return [n for n, f in zip(self.param_names, flags, strict=True) if f]

    def _is_compartment_size(self, name: str) -> bool:
        """Whether ``name`` is an SBML compartment size."""
        return name in set(self.compartment_size_params)

    def _is_volume_write_refused(self, name: str) -> bool:
        """Whether a value-changing write to ``name`` is refused (issue #170)."""
        return name in set(self.unwritable_compartment_size_params)

    def _internal_param_names(self) -> set[str]:
        """Synthesized parameters that are not knobs of the model.

        :attr:`param_is_internal` as a set: ``_V0_<comp>``, an SBML
        compartment's size as it was at load (issue #170), and the slot holding
        a function's evaluated value (issue #227). ``set_param`` refuses a
        value-changing write to either; this is how ``set_params`` learns that
        in its *validation* phase, so the refusal cannot fire halfway through
        the apply loop.
        """
        try:
            flags = self._core.param_is_internal
        except AttributeError:  # pragma: no cover - defensive
            return set()
        return {n for n, f in zip(self.param_names, flags, strict=True) if f}

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
        ValueError
            If any entry is an SBML compartment size being *changed* (issue
            #164 — see :meth:`set_param`). Checked in the validation phase with
            the names and the values, so the atomicity above holds: a dict with
            one compartment write in it applies none of its entries.

        Examples
        --------
        >>> model.set_params({"kf": 0.5, "kr": 0.1})

        Notes
        -----
        Each write goes through the same path as :meth:`set_param`, so a
        parameter that names a species initial condition re-resolves it here too
        (issue #79).
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
        # Phase 2b: refuse every write set_param would refuse HERE rather than let
        # it throw from the apply loop, which would leave the earlier entries
        # written and break the atomicity this method documents (issue #164).
        # Two kinds: an unwritable compartment size, and issue #170's internal
        # `_V0_<comp>` record. Same rule as set_param in both cases — an unchanged
        # value is not a change, which is what keeps a full-vector round trip
        # working. Every other compartment size is an ordinary writable parameter
        # now (issue #170) and falls straight through to Phase 3.
        refused = set(self.unwritable_compartment_size_params) | self._internal_param_names()
        for name, value in converted.items():
            self._raise_if_folded_at_load(name, value)
            if name in refused and value != self._core.get_param(name):
                self.set_param(name, value)  # raises with the full explanation
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

        The seed initial conditions are the ones the *current* parameter values
        imply, not the ones the network was generated with: a species whose
        initial condition names a parameter (``A() Stot``) returns to that
        parameter's value as of now (issue #79).
        """
        self._core.reset()
        # Wholesale IC change: any declared ∂x(0)/∂θ described the assignment this
        # just discarded (issue #111).
        self._declared_ic_sens.clear()
        if self._ic_write_log is not None:
            self._ic_write_log.add("*")

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

        A snapshot taken of a *pre-equilibrated* state also captures that state's
        forward-sensitivity matrix ``dx/dθ`` when one is pending, so
        :meth:`restore_concentrations` puts the θ-derivative back with the
        concentrations (issue #81). Without it, a restored equilibrated state
        would be re-seeded as a fresh start (``∂x(0)/∂θ = 0``) — wrong, not
        approximate, since the restored initial condition *is* a function of θ.
        The unlabeled form hands the derivative to the new IC baseline instead,
        so :meth:`reset` restores it too.
        """
        if label is None:
            self._core.save_concentrations()
            return
        # A named snapshot is a copy of the live state vector; storing get_state()
        # (which already returns a fresh array) is safe, but copy defensively so a
        # later set_state alias can never mutate a stored snapshot.
        key = str(label)
        self._named_conc_states[key] = np.array(self._core.get_state(), dtype=np.float64)
        # ...and its θ-derivative, when this state carries one (issue #81).
        core = self._core
        if core.has_pending_sensitivity_seed:
            self._named_sens_seeds[key] = (
                np.array(core.pending_sensitivity_seed(), dtype=np.float64),
                list(core.pending_sensitivity_seed_param_names),
            )
        else:
            self._named_sens_seeds.pop(key, None)

    def restore_concentrations(self, label: str | None = None) -> None:
        """Restore species concentrations from a saved snapshot.

        Implements BNG ``resetConcentrations()`` / ``resetConcentrations("name")``.

        A snapshot saved with a forward-sensitivity seed (a pre-equilibrated
        state — see :meth:`save_concentrations`) restores that ``dx/dθ`` along
        with the concentrations, so a following
        ``run(carry_sensitivities=True)`` measures from the right seed
        (issue #81).

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
            self.reset()
            return
        key = str(label)
        snapshot = self._named_conc_states.get(key)
        if snapshot is None:
            known = ", ".join(sorted(self._named_conc_states)) or "(none)"
            raise ModelError(
                f"No saved concentration state named {key!r}. "
                f"Saved states: {known}. Call save_concentrations({key!r}) first."
            )
        # set_state drops any pending seed (it cannot know an externally supplied
        # state's derivative) — so re-install this snapshot's own, if it has one.
        self._core.set_state(snapshot)
        if self._ic_write_log is not None:
            self._ic_write_log.add("*")
        seeded = self._named_sens_seeds.get(key)
        if seeded is not None:
            seed, names = seeded
            self._core.set_pending_sensitivity_seed(seed, names)
            # The restored state is a θ-dependent initial condition: fresh-start
            # seeding would be wrong, so keep the "needs carry_sensitivities" arm.
            self._core.ic_state_dirty = True

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
            Species name (e.g. ``"A(b)"``). A fixed species' ``$`` may be given
            or left out: the model names ``$A(b)`` in a ``.net`` ``A(b)``, and
            BNG2.pl's ``setConcentration`` ignores the ``$`` too.
        value : float
            New concentration value.

        Raises
        ------
        ModelError
            If the species name is not found.

        Notes
        -----
        Implements BNG ``setConcentration("name", value)`` action.

        The value is the start of the next :meth:`Simulator.run`, of every
        replicate of :meth:`Simulator.run_replicates`, and of
        :meth:`Simulator.parameter_scan`, whose default reset target is the
        model's state at the call. :meth:`reset` does not keep it, and
        neither does :meth:`Simulator.run_batch`, which starts every point from
        the initial conditions, and warns that it does (issue #1362). Call
        :meth:`save_concentrations` after it to make it the state :meth:`reset`
        returns to.

        For forward sensitivity, this is a **literal** initial condition:
        ``∂x_k(0)/∂θ = 0``. It therefore supersedes whatever the species' declared
        initial condition depended on — if the ``.net`` set it from a parameter
        (``R() R0``), that parameter no longer reaches this species' IC and its seed
        row is dropped (issue #113). When the value you assign *does* depend on a
        differentiated parameter, say so with :meth:`declare_ic_sensitivity`.
        """
        value = float(value)
        name = self._species_call(name, lambda n: self._core.set_concentration(n, value))[0]
        # A fresh assignment supersedes whatever this species' initial condition
        # was declared to depend on; declare again after the write (issue #111).
        self._declared_ic_sens.pop(name, None)
        if self._ic_write_log is not None:
            self._ic_write_log.add(name)

    def get_concentration(self, name: str) -> float:
        """Get a single species concentration by name.

        Parameters
        ----------
        name : str
            Species name, a fixed species' ``$`` given or left out, as for
            :meth:`set_concentration`.

        Returns
        -------
        float
            Current concentration.

        Raises
        ------
        ModelError
            If the species name is not found.
        """
        return float(self._species_call(name, self._core.get_concentration)[1])

    def _species_call(self, name: str, call: Callable[[str], Any]) -> tuple[str, Any]:
        """``(the model's name, call(it))`` for species *name*: *name* itself,
        or else *name* without a fixed species' ``$``, which the model's names
        never hold. Raises :class:`ModelError` when neither is a species."""
        try:
            return name, call(name)
        except (KeyError, RuntimeError) as e:
            bare = _unclamped(name)
            if bare != name:
                try:
                    return bare, call(bare)
                except (KeyError, RuntimeError):
                    pass
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
        self._declared_ic_sens.clear()
        if self._ic_write_log is not None:
            self._ic_write_log.add("*")

    # ─── Per-point IC-sensitivity declarations (issue #111) ───────────────

    def declare_ic_sensitivity(self, sens: Mapping[str, Mapping[str, float]]) -> None:
        """Declare ``∂x_k(0)/∂θ`` for initial conditions a per-point hook assigns.

        Call this from a :meth:`Simulator.parameter_scan` ``on_point`` hook when
        the value it assigns is *computed from* a parameter being differentiated —
        a ligand dose converted with a fitted volume, say. The scan then uses the
        declared row verbatim instead of measuring it through the hook, which is
        both exact and cheaper (a declared species is not probed).

        Parameters
        ----------
        sens : mapping
            ``{species_name: {param_name: d_ic_d_param}}``. A declared species'
            row is taken as **fully specified**: parameters not named get
            ``∂x_k(0)/∂θ = 0``. Declaring ``{}`` for a species therefore pins its
            whole row to zero (an explicitly θ-independent literal). Repeated
            calls merge per species; a species declared twice keeps the last row.

        Raises
        ------
        ModelError
            If a species or parameter name is unknown.

        Notes
        -----
        Also honoured by a plain :meth:`Simulator.run` — an initial condition
        assigned by hand is no longer described by the ``.net`` expression the
        parameter-graph seeding (issue #43) differentiates, so for a hand-assigned
        θ-dependent IC this declaration is the only way the engine can know
        ``∂x_k(0)/∂θ``.

        A declaration lasts until the initial condition it describes changes:
        assigning the same species again (:meth:`set_concentration`), or a
        wholesale :meth:`reset` / :meth:`set_state`, drops it. The scan primitive
        additionally clears declarations before each ``on_point`` call, so a hook
        declares afresh per point and points stay independent.

        Nothing needs declaring for the ordinary dose: a literal
        ``set_concentration`` inside an ``on_point`` hook has
        ``∂x_k(0)/∂θ = 0``, which the scan measures on its own.

        Examples
        --------
        >>> def on_point(model, dose_nM):    # dose in molecules, fitted volume
        ...     v = dose_nM * 1e-9 * NA * model.get_param("Vecf")
        ...     model.set_concentration("L(r)", v)
        ...     model.declare_ic_sensitivity({"L(r)": {"Vecf": v / model.get_param("Vecf")}})
        """
        species = set(self.species_names)
        params = set(self.param_names)
        staged: dict[str, dict[str, float]] = {}
        for given, row in sens.items():
            # A fixed species' `$` may be given, as for set_concentration.
            sp = given if given in species else _unclamped(given)
            if sp not in species:
                raise ModelError(
                    f"declare_ic_sensitivity: species {given!r} not found in model. "
                    "Names must match species_names, a fixed species' `$` aside."
                )
            staged[sp] = {}
            for p, d in row.items():
                if p not in params:
                    raise ModelError(
                        f"declare_ic_sensitivity: parameter {p!r} (for species "
                        f"{sp!r}) not found in model."
                    )
                staged[sp][p] = float(d)
        self._declared_ic_sens.update(staged)

    def effective_ic_sensitivity(
        self, params: Iterable[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """``∂x(0)/∂θ`` — the initial-condition seeding a run from this state would use.

        The reader paired with :meth:`declare_ic_sensitivity`, and the single
        defining site for the seeding the forward-sensitivity solver is
        initialized with. Answers from **model structure alone** — the parameter
        graph, the live initial conditions and any declarations — with no
        integration and no simulation, so a frontend can build its gradient
        routing once at setup (issue #155).

        Why a consumer needs it: :meth:`Result.output_sensitivities` with
        ``axis="parameter"`` is a **total** derivative,

        .. code-block:: text

            d_param[θ] = (right-hand-side path) + Σ_k (∂x_k(0)/∂θ)·d_ic[x_k]

        so the seeding is *already inside* the ``parameter`` axis. A caller that
        routes a fitted parameter to several columns and sums them must add an
        ``ic``-axis term only for the part of ``∂x(0)/∂θ`` reported here as
        **absent** — anything present is carried already, and adding it again
        counts it twice.

        This is the *effective* matrix, not the model file's: a species retired
        because an assignment moved it off the initial condition its expression
        describes (issue #113) is gone, and a species the caller declared
        (issue #111) carries the declared row in place of the parameter-graph
        one. It is therefore **state-dependent by design** — call it on the
        configured model, before the run whose gradients it describes.

        Parameters
        ----------
        params : iterable of str, optional
            Restrict to these parameter names. Default: every parameter. Pass
            the same list as ``sensitivity_params=`` to see exactly the rows
            that run's ``parameter`` columns will carry.

        Returns
        -------
        dict
            ``{species_name: {param_name: ∂x_k(0)/∂θ}}``, keyed by the names
            :attr:`species_names` / :attr:`param_names` report — the same ids
            ``sensitivity_params=`` accepts. A compound ``<initialAssignment>``
            lowered by issue #147 reports the **original** symbols it was
            written over, never the synthetic ``_ic_<species>`` carrier.

            A **present** entry whose value is ``0.0`` means *seeded, and the
            coefficient is zero at this state* — a chain-rule factor that
            vanishes here but need not at another point. An **absent** entry
            means *no seeding path at all*, and is the one that says "this
            parameter's initial-condition term is yours to supply". The two are
            different answers; do not collapse them.

        Examples
        --------
        >>> model.effective_ic_sensitivity(["R0", "kf"])       # doctest: +SKIP
        {'R(r)': {'R0': 1.0}}
        """
        wanted = set(self.param_names if params is None else params)
        triples, injected = self._ic_sensitivity_triples()
        species, pnames = self.species_names, self.param_names
        out: dict[str, dict[str, float]] = {}
        if injected:
            for sp_i, p_i, coeff in triples:
                if sp_i < 0:
                    continue  # the sentinel row, which seeds nothing
                pname = pnames[p_i]
                if pname not in wanted or pname == _ic_carrier_name(species[sp_i]):
                    continue  # issue #715: the carrier is seeded, never reported
                row = out.setdefault(species[sp_i], {})
                # The C++ seeding accumulates (`yS[iS][i] += coeff`): one initial
                # condition can reach the same primary by more than one path.
                row[pname] = row.get(pname, 0.0) + float(coeff)
            return out
        # Nothing injected ⇒ the C++ fallback identity loop seeds this run, so
        # report *its* rows. Reporting {} here would be a silent under-count.
        live = np.asarray(self.get_state(), dtype=np.float64)
        baseline = np.asarray(self._core.get_initial_state(), dtype=np.float64)
        for sp_i, p_i in self._core.species_ic_param_refs:
            pname = pnames[p_i]
            if pname not in wanted or live[sp_i] != baseline[sp_i]:
                continue
            out.setdefault(species[sp_i], {})[pname] = 1.0
        return out

    def _ic_sensitivity_triples(self) -> tuple[list[tuple[int, int, float]], bool]:
        """``([(species_idx0, param_idx0, ∂IC/∂param), ...], injected?)``.

        The seeding pipeline itself, shared by :meth:`effective_ic_sensitivity`
        and the solver-option path so the reported matrix cannot drift from the
        seeded one. Explicit zeros are **kept** here — the caller that builds the
        C++ list drops them (a zero seeds nothing), the caller that reports keeps
        them (absent and zero are different answers, issue #155).

        ``injected`` is False only when nothing is handed to the core at all, in
        which case its legacy ``species_ic_param_refs`` identity loop applies.
        """
        from bngsim._codegen import compute_ic_param_sens_seed

        seeds = compute_ic_param_sens_seed(self._core)
        # (#170 stage 3) The ∂x(0)/∂V column is seeded and reported now. It used
        # to be filtered out here: an ``<initialAssignment>`` may read a
        # compartment size — that is what makes ``set_param`` on the size
        # reproduce a rebuild — and the chain rule differentiated it happily, but
        # what it produced was ∂(amount)/∂V where the state is ``amount/V``, so
        # the column was the numerator's derivative with neither the ``1/V`` on
        # it nor the ``−amount/V²`` beside it. MODEL1710030000 is the case:
        # ``S21 = 393.927*0.055*cell`` is exactly V-invariant once stored, and
        # the un-corrected column said 21.67 where a rebuild-to-rebuild
        # difference says 0. ``compute_ic_param_sens_seed`` now supplies both
        # halves, and they cancel to that 0.
        declared = self._declared_ic_sens
        if seeds and self._core.ic_baseline_saved:
            # save_concentrations() made the state its own baseline: the
            # declared initial conditions no longer describe it, and
            # ``set_param`` no longer moves it (issue #79). No parameter reaches
            # it through them. The rows were kept, and a run from a saved
            # baseline was seeded as from the model file: dY/dA0 = [1/3, 2/3]
            # at the steady state of A <-> B for an A(0) that A0 does not set.
            retired = {entry[0] for entry in seeds}
        else:
            retired = self._superseded_ic_rows(seeds, declared) if seeds else set()
        if retired:
            seeds = [entry for entry in seeds if entry[0] not in retired]
        if declared:
            seeds = self._overlay_declared_ic_sens(seeds, declared)
        if (retired or declared) and not seeds:
            # An empty list means "no Python injection" to the C++ seeding, which
            # then falls back to its legacy species_ic_param_refs identity loop —
            # the very rows just retired. A sentinel row keeps the list non-empty
            # and seeds nothing: the consumer skips any entry with
            # species_idx0 < 0. (That loop applies the #113 rule too, but a
            # declaration is invisible to it, so do not rely on it here.)
            return [(-1, 0, 0.0)], True
        return seeds, bool(seeds)

    def _superseded_ic_rows(
        self,
        seeds: list[tuple[int, int, float]],
        declared: dict[str, dict[str, float]],
    ) -> set[int]:
        """Species whose ``.net`` IC expression no longer describes their state.

        ``reset()`` returns the live concentrations to ``initial_conc``, so the two
        differ exactly when an assignment (``set_concentration`` / ``set_state`` /
        an external injection) has superseded the declared initial condition. Those
        species' parameter-graph seeds are dropped (issue #113); a species the
        caller declared is left to :meth:`_overlay_declared_ic_sens`, which is the
        more specific statement.

        A caller that re-asserts a species' *own* IC value keeps its row: the
        assignment and the expression then agree numerically, and which of the two
        was meant is genuinely ambiguous — ``declare_ic_sensitivity`` says so
        either way.
        """
        rows = {entry[0] for entry in seeds}
        if not rows:
            return set()
        live = np.asarray(self.get_state(), dtype=np.float64)
        baseline = np.asarray(self._core.get_initial_state(), dtype=np.float64)
        moved = {int(i) for i in np.nonzero(live != baseline)[0]}
        if not moved:
            return set()
        spoken_for = {i for i, name in enumerate(self.species_names) if name in declared}
        return (rows & moved) - spoken_for

    def _overlay_declared_ic_sens(
        self,
        seeds: list[tuple[int, int, float]],
        declared: dict[str, dict[str, float]],
    ) -> list[tuple[int, int, float]]:
        """Replace the ∂x_k(0)/∂p rows of species the caller declared (issue #111).

        A declared zero is kept as an explicit ``0.0`` row rather than dropped:
        it seeds nothing either way, but it is the difference between "pinned to
        zero here" and "no seeding path", which issue #155 requires a reader to
        be able to tell apart.
        """
        param_idx = {name: i for i, name in enumerate(self.param_names)}
        species_idx = {name: i for i, name in enumerate(self.species_names)}
        replaced = {species_idx[sp] for sp in declared if sp in species_idx}
        out = [entry for entry in seeds if entry[0] not in replaced]
        for sp, row in declared.items():
            i = species_idx.get(sp)
            if i is None:
                continue
            for p, d in row.items():
                if p in param_idx:
                    out.append((i, param_idx[p], float(d)))
        return out

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
    def function_names(self) -> list[str]:
        """Names of the model's functions, in declaration order.

        A function is not a parameter, even though each one also names a
        parameter *slot* — where the engine stores the value it last evaluated
        to, rewritten before every derivative evaluation.
        :attr:`param_is_internal` flags those slots and
        :attr:`primary_param_names` omits them (issue #227).
        """
        return list(self._core.function_names)

    @property
    def n_events(self) -> int:
        """Number of events (SBML/Antimony ``at (...)`` triggers) in the model."""
        return self._core.n_events

    @property
    def conservation_laws(self) -> dict:
        """
        Conservation laws detected from the stoichiometry matrix.

        Found for every input format (``.net``, Antimony, SBML, programmatic
        ``ModelBuilder``). The returned dict has keys:

        - ``n_laws``: number of independent conservation laws
        - ``n_species``: number of species the laws are expressed over
        - ``dependent`` / ``independent``: 0-based species index lists
        - ``constants``: conservation constants evaluated from the initial conditions
        - ``coefficients``: ``n_laws`` x ``n_species`` coefficient matrix

        Each law is over the stored state, ``coefficients[k] @ y`` constant
        along a run, so ``coefficients @ rhs(y)`` is zero at any ``y``. What a
        reaction between compartments of different size conserves is an
        amount, so a species' coefficient there carries its compartment's
        size: ``A`` in a compartment of size 1 exchanged with ``B`` in one of
        size 2 gives ``A + 2*B`` (issue #758). Such a model's laws follow a
        size written with :meth:`set_param`. A species whose share of such a
        reaction is divided by a size that moves, a concentration in a
        compartment that a rate rule or an event resizes, is in no law.

        Consumed internally by the reduced-space Newton steady-state solver,
        which needs the independent subspace to sidestep the rank-deficient
        Jacobian these laws imply.
        """
        return self._core.conservation_laws

    # ─── Public evaluators (issue #523) ───────────────────────────────────
    # The RHS, Jacobian, stoichiometry and propensities the integrators run,
    # as NumPy callables over an arbitrary state — the provider a continuation,
    # a root classifier, or a moment/CME generator built outside bngsim needs.
    # Each one calls the C++ evaluator the corresponding engine calls, so they
    # cannot drift from what a solve sees.

    def stoichiometry_matrix(self, *, sparse: bool = False):
        """The net stoichiometry matrix ``S``, shape ``(n_species, n_reactions)``.

        ``S[i, r]`` is the net change in species ``i`` when reaction ``r`` fires:
        products positive, reactants negative, so ``A + A -> B`` puts ``-2`` in
        A's row. 0-based, in :attr:`species_names` and reaction-index order.
        For a mass-action ``.net`` model ``rhs(y) == S @ v(y)`` for the vector
        of ODE reaction rates, and every conservation law satisfies
        ``L @ S == 0`` — this is the matrix :attr:`conservation_laws` was
        row-reduced from. (An SBML model with a reaction between compartments
        of different size is the exception: its laws are found for the
        amounts, and satisfy ``L @ rhs(y) == 0`` and not ``L @ S == 0``.)

        A ``$``-prefixed boundary species has an all-zero row: the RHS zeroes
        its derivative and an SSA firing never updates it, so the matrix that
        describes what the engines do has no entry there, whatever the
        reaction line says.

        Parameters
        ----------
        sparse : bool
            ``False`` (default) returns a dense float64 ``ndarray``. ``True``
            returns a :class:`scipy.sparse.csc_array` of the same shape, which
            is the form to ask for on a large network — the dense matrix of a
            10,000-species, 100,000-reaction model is 8 GB. Needs scipy.

        Notes
        -----
        The 1-based C++ entry list is converted once per model and cached;
        each call assembles a fresh matrix from it, so the caller may modify
        what it gets back.
        """
        from bngsim._evaluators import scipy_sparse, stoichiometry_coo

        if self._stoich_coo is None:
            self._stoich_coo = stoichiometry_coo(self._core)
        rows, cols, vals = self._stoich_coo
        shape = (self.n_species, self.n_reactions)
        if sparse:
            return scipy_sparse().csc_array((vals, (rows, cols)), shape=shape)
        S = np.zeros(shape, dtype=np.float64)
        S[rows, cols] = vals
        return S

    def rhs(self, y, t: float = 0.0) -> np.ndarray:
        """``dy/dt`` at state ``y`` and time ``t``, under the live parameters.

        The same interpreted RHS the CVODE callback evaluates, with the same
        observable-and-function refresh it runs — observable totals and
        function-bound rate parameters are recomputed *at* ``y`` for a model
        that has functions and skipped for a pure mass-action one (the
        ``rhs_evaluates_observables`` gate) — against the model's current
        :meth:`get_param` values. A ``$``-fixed species reports a zero
        derivative, as it does in a solve.

        The model's stored concentrations (:meth:`get_state`) are not read and
        not written; only the caller's ``y`` enters. What does change is the
        model's cached observable and function values, exactly as one
        integrator step would change them.

        Parameters
        ----------
        y : array_like, shape ``(n_species,)``
            The state to evaluate at, in :attr:`species_names` order.
        t : float
            The time ``rateOf``, ``time``-dependent functions and table
            functions read. ``0.0`` by default.

        Returns
        -------
        ndarray, shape ``(n_species,)``
            ``f(t, y)``, float64.

        Raises
        ------
        ValueError
            If ``y`` is not 1-D or its length is not :attr:`n_species`.
        ModelError
            If rates of change are written in terms of each other's ``rateOf``
            in a ring (an algebraic loop, which has no right-hand side).

        See Also
        --------
        jacobian : ``∂f/∂y`` at the same point.
        propensities : the SSA-convention reaction rates at ``y``.
        """
        try:
            return self._core.compute_derivs(float(t), np.asarray(y, dtype=np.float64))
        except RuntimeError:
            # (Asked on this path alone: the question costs a tenth of what
            # the right-hand side of a small model does.)
            ring = self._core.rateof_ring_refusal()
            if ring:
                raise ModelError(ring) from None
            raise

    def jacobian(self, y, t: float = 0.0, *, sparse: bool = False):
        """``∂f/∂y`` at state ``y`` and time ``t`` — the ODE Jacobian.

        ``J[i, j] = ∂f_i/∂x_j``, both indices in :attr:`species_names` order.
        Built from the model's closed-form terms whenever they cover every
        reaction, and from a one-sided difference quotient of :meth:`rhs`
        otherwise; the result says which. That is the rule the ODE integrator
        and the steady-state solver apply under ``jacobian="auto"``, and the
        difference quotient is stepped by the steady-state solver's own rule,
        so a caller gets the matrix a solve would have factored at that point
        rather than a third approximation. A model whose analytical Jacobian is
        *partial* (a rate law the differentiator declined) is never handed the
        partial matrix: it is differenced, and reported as such.

        The first call on a model with functional rate laws derives the
        closed-form terms (:meth:`prepare_analytical_jacobian`), which an ODE
        solve would do at its own setup; later calls reuse them.

        Parameters
        ----------
        y : array_like, shape ``(n_species,)``
            The state to linearize at.
        t : float
            The time, for time-dependent rate laws. ``0.0`` by default.
        sparse : bool
            ``False`` (default) returns a :class:`~bngsim.JacobianMatrix`, a
            dense ``(n_species, n_species)`` float64 ``ndarray`` with a
            ``source`` attribute. ``True`` returns a
            :class:`scipy.sparse.csc_array` over the model's structural
            sparsity pattern — the same pattern from either source, so a
            consumer can allocate for it once — with the same ``source``
            attribute set on it. Needs scipy.

        Returns
        -------
        JacobianMatrix or scipy.sparse.csc_array
            ``.source`` is ``"analytical"`` or ``"finite-difference"``, the
            spellings :attr:`bngsim.SteadyStateResult.solver_jacobian_source`
            uses for the same two cases.

        Raises
        ------
        ValueError
            If ``y`` is not 1-D or its length is not :attr:`n_species`.
        ModelError
            If rates of change are written in terms of each other's ``rateOf``
            in a ring (an algebraic loop, which has no right-hand side).
        SimulationError
            If the matrix is a difference quotient whose step decides an entry
            (issue #1002): the entry moves by more than 1e-4 of itself (1e-10 of
            its column's largest, at least) when its column is taken again at
            the species' own step or at half the first.
        """
        from bngsim._evaluators import (
            ANALYTICAL,
            FINITE_DIFFERENCE,
            JacobianMatrix,
            csc_from_pattern,
            gather_pattern,
        )

        # (A difference quotient of a right-hand side that is not there came
        # back as a matrix of NaN.)
        ring = self._core.rateof_ring_refusal() if self._core.uses_rateof else ""
        if ring:
            raise ModelError(ring)
        analytical = self.prepare_analytical_jacobian()
        source = ANALYTICAL if analytical else FINITE_DIFFERENCE
        y_arr = np.asarray(y, dtype=np.float64)
        t = float(t)
        if sparse:
            if analytical:
                vals = self._core.fill_sparse_analytical_jacobian(t, y_arr)
            else:
                dense = self._core.fill_dense_fd_jacobian(t, y_arr)
                self._raise_if_the_step_decides(t, y_arr, dense)
                vals = gather_pattern(self._core, dense)
            return csc_from_pattern(self._core, vals, source)
        if analytical:
            dense = self._core.fill_dense_analytical_jacobian(t, y_arr)
        else:
            dense = self._core.fill_dense_fd_jacobian(t, y_arr)
            self._raise_if_the_step_decides(t, y_arr, dense)
        return JacobianMatrix(dense, source)

    def _raise_if_the_step_decides(self, t: float, y: np.ndarray, dense: np.ndarray) -> None:
        """Refuse the difference quotient ``dense`` where it and the same
        matrix with the columns of species far below the state's scale at their
        own step differ in an entry by more than 1e-4 of the entry (1e-10 of
        its column's largest, at least; issue #1002). The first steps every
        species by 1.5e-8 of the state's largest concentration, which reads a
        term of second order in a small species as a secant across it; the
        second misses a term that reads it through a sum with something much
        larger. The same is asked of the matrix with those columns, and those
        of species at 0, taken whole at half the first step, which a secant
        moves with and a slope does not, also in a row the own step cannot
        read. ``steady_state`` refuses the columns it would solve with such a
        matrix, by the same rule. This returned the matrix beside a warning;
        a warning beside an entry that may be wrong is no refusal."""
        floored = np.asfortranarray(dense)
        own, changed = self._core.fill_dense_fd_jacobian_own_step(t, y, floored)
        half, taken = self._core.fill_dense_fd_jacobian_half_step(t, y, floored)
        worst = None
        for other, at, columns in ((own, "1.5e-8 of itself", changed), (half, "half that", taken)):
            for j in columns:
                # Over the entries that are numbers, as steady_state takes it: an
                # entry that is none has no answer to confirm, and made the
                # column's largest NaN, which no entry exceeded.
                finite = np.isfinite(dense[:, j])
                largest = float(np.max(np.abs(dense[finite, j]))) if finite.any() else 0.0
                held_to = np.maximum(np.abs(dense[:, j]), 1e-6 * largest)
                # An entry that is no number at the second step is as far
                # apart as can be.
                apart = np.abs(other[:, j] - dense[:, j])
                apart[np.isnan(apart)] = np.inf
                exceeds = finite & (apart > 1e-4 * held_to)
                if not exceeds.any():
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = np.where(exceeds, apart / held_to, 0.0)
                share[exceeds & (held_to == 0.0)] = np.inf
                i = int(np.argmax(share))
                if worst is None or share[i] > worst[0]:
                    worst = (share[i], i, j, other[i, j], at)
        if worst is None:
            return
        _, i, j, second, at = worst
        names = self.species_names
        raise SimulationError(
            "Model.jacobian() does not return this difference quotient (issue #1002): "
            f"the entry d(rate of {names[i]})/d{names[j]} is {dense[i, j]:.6g} with "
            f"{names[j]} stepped by 1.5e-8 of the state's largest concentration, and "
            f"{second:.6g} stepped by {at}, more than 1e-4 of itself (1e-10 of its "
            f"column's largest, at least) apart. {_STEP_DECIDES_WHY} A model whose "
            "rate laws all have a closed-form Jacobian has neither failure."
        )

    def propensities(self, y, t: float = 0.0) -> np.ndarray:
        """Every reaction's SSA propensity at state ``y`` and time ``t``.

        One entry per reaction, in reaction-index order: the amount/time
        propensity a stochastic step samples from, **in the SSA volume
        convention** rather than the ODE one. Two things differ from the ODE
        rate of the same reaction. A repeated reactant takes the falling
        factorial over its molecule count ``n`` when bngsim reads the reaction as
        mass action — ``A + A -> B`` fires at ``k·n(n−1)/2``, where the ODE rate
        is ``k·n²/2``, with ``n = x·V`` for a species stored as a concentration
        in a compartment of size ``V`` (issue #692). An SBML law it does not read
        as mass action is evaluated as written, still with the falling factorial
        when the law is a product of its species; one that writes its own
        combinatorics (``k*X*(X-1)/2``) is left alone — and a reaction in a compartment
        of volume ``V`` is
        multiplied by ``V`` (``Reaction::ssa_volume_factor``, or the live value
        of the compartment-size parameter it names), converting the ODE's
        storage-units rate to a per-event rate. For a ``.net`` model with
        volume 1 and no repeated reactant the two conventions agree.

        Observable totals and function-bound parameters are refreshed at ``y``
        first, as the SSA loop refreshes them before its propensity pass, under
        the same gate :meth:`rhs` takes. The stored concentrations are not
        touched. Values are read from ``y`` as given — pass molecule counts
        for a count-valued answer.

        Parameters
        ----------
        y : array_like, shape ``(n_species,)``
            The state to evaluate at, in :attr:`species_names` order.
        t : float
            The time, for time-dependent rate laws. ``0.0`` by default.

        Returns
        -------
        ndarray, shape ``(n_reactions,)``
            The propensity vector ``a(y)``, float64.

        Raises
        ------
        ValueError
            If ``y`` is not 1-D or its length is not :attr:`n_species`.

        See Also
        --------
        stoichiometry_matrix : the state change each of these reactions makes.
        """
        return self._core.compute_propensities(float(t), np.asarray(y, dtype=np.float64))

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

        *Derived* means the value expression **references another of the model's
        symbols**, which is the condition that makes the chain rule necessary and
        the line BNG2.pl itself draws between ``# ConstantExpression`` and
        ``# Constant``. A constant written as arithmetic — ``gamma 1/7``,
        ``pi 2*asin(1)``, ``c6 ln(2)/120`` — names nothing, so it differentiates
        as a leaf exactly like a literal and reports ``False`` here (issue #227).
        It used to report ``True``, on the narrower reading that the value text
        does not parse as a float, and the recovery rate of ``SIR.net`` was
        consequently missing from :attr:`primary_param_names`.

        This is *live* state, not a property of the declaration: writing a
        derived parameter a value its expression does not currently produce
        overrides the expression and flips its entry to ``False`` until the
        override is lifted (issue #188 — see :meth:`set_param`).
        :attr:`param_expressions` keeps the defining expression throughout, so
        an entry that is ``False`` here while its expression there names
        another of the model's symbols is exactly an overridden derived
        parameter, as distinct from a genuine primary, whose text is a number
        (``2e7``, ``1/7``) or empty. That pairing is also the whole of what the
        generated C source depends on: an override moves the chain rule, and
        nothing else about a parameter write does.
        """
        return list(self._core.param_is_expression)

    @property
    def param_expressions(self) -> list[str]:
        """Per-parameter defining text, parallel to :attr:`param_names`.

        For a derived parameter (:attr:`param_is_expression`), the expression it
        is re-evaluated from, such as ``1/S0``. It stays on record while a
        :meth:`set_param` override pins the parameter to a literal, which is how
        an overridden derived parameter is told from a genuine primary.

        Anything else has a number or nothing here: a primary read from a
        ``.net`` file keeps its value as the file wrote it (``2e7``, ``1/7``),
        and most SBML primaries and every synthesized slot
        (:attr:`param_is_internal`) have ``""``. This is the declaration, which
        a write does not change, so read a parameter's value with
        :meth:`get_param`.
        """
        return list(self._core.param_expressions)

    @property
    def param_is_internal(self) -> list[bool]:
        """Per-parameter *synthesized-slot* flag, parallel to :attr:`param_names`.

        ``True`` for a slot bngsim created for its own bookkeeping rather than
        one the model declared. Two kinds, and neither is a knob — each is left
        out of :attr:`primary_param_names` and refused a value-changing
        :meth:`set_param`:

        * ``_V0_<comp>`` (issue #170) — an SBML compartment's size as it was at
          load, which the rate constants in that compartment are normalised
          against. Moving it rescales those rates while the volume stays put.
        * a **function's backing slot** (issue #227) — where the engine stores
          what the function last evaluated to. Every function that does not
          already name a declared parameter has one, so every model with a
          ``functions`` block carries these.

        Distinct from :attr:`param_is_expression`: a derived parameter is
        recomputed from primaries, these are not primaries in the first place.
        The two flags are disjoint in practice, and
        :attr:`primary_param_names` is :attr:`param_names` minus both.
        """
        return list(self._core.param_is_internal)

    @property
    def primary_param_names(self) -> list[str]:
        """List of parameter names that are the model's independent knobs.

        These are the genuine knobs of the model — primary rate constants,
        initial-condition parameters, etc. Use this when you want to expose
        the model to an external optimizer or sampler that should treat
        each parameter as an independent variable; varying a primary via
        :meth:`set_param` automatically propagates to derived parameters.
        Leave out :attr:`frozen_params` where a model has any: those are
        listed here, and a write that changes one is refused.

        Two kinds are left out, one per flag. A derived ``ConstantExpression``
        (:attr:`param_is_expression`, e.g. ``_rateLaw{N}``) is recomputed from
        its primaries, so it is not independent of them. And a synthesized slot
        (:attr:`param_is_internal`) is not a parameter at all: ``_V0_<comp>``,
        an SBML compartment's size as it was at load, which the rate constants
        in that compartment are normalised against — moving it would rescale
        those rates without moving the volume, so set the compartment size
        itself, an ordinary writable parameter since issue #170 — or the slot
        holding a **function's** evaluated value, which the engine overwrites
        before every derivative evaluation (issue #227).

        Both exclusions are about the *same* failure: a coordinate an optimizer
        cannot move. Issue #227 is where this list stopped being either. It
        listed every function name — 327 of 327 function-carrying ``.net``
        models leaked them — so ``jax.grad`` spent a coordinate on one and got
        exactly ``0.0``, forever. And it dropped a constant written as
        arithmetic (``gamma 1/7``), which is a working knob by every other
        measure, so a fit over this list held the recovery rate of ``SIR.net``
        fixed with no warning.
        """
        names = self.param_names
        flags = self.param_is_expression
        try:
            internal = list(self._core.param_is_internal)
        except AttributeError:  # pragma: no cover - defensive
            internal = [False] * len(names)
        return [n for n, f, i in zip(names, flags, internal, strict=False) if not f and not i]

    @property
    def species_names(self) -> list[str]:
        """List of all species names."""
        return self._core.species_names

    @property
    def observable_names(self) -> list[str]:
        """List of all observable group names."""
        return self._core.observable_names

    # ─── Write-only accumulator species (issue #74) ───────────────────────

    def pure_sink_species(self) -> list[str]:
        """Names of the *write-only accumulator* species in this network.

        A pure sink is a species this network only ever writes to — the
        ``degraded`` / ``produced`` / ``secreted`` pool a BNGL model carries to
        count cumulative flux. Its derivative is a non-zero constant for as long
        as its producing reactions fire, so ``||f(y)||_2 / n_species`` has a
        floor above ``tol`` and :meth:`Simulator.steady_state` reports failure
        however long it integrates — even when every other species has settled.
        These are the species to hand (negated) to ``steady_state(mask=...)``.

        Detection is purely structural — nothing is measured, no annotation is
        needed. A species qualifies when all of the following hold:

        1. it is a product of at least one reaction,
        2. it is a reactant of none,
        3. no species' derivative depends on it, and
        4. it is not a ``$``-prefixed (fixed / boundary-condition) species.

        Clause 3 is not implied by clauses 1-2 and is what makes excluding the
        species provably harmless to the rest of the system: an Elementary rate
        law reads only its reactants, but a Functional one reads observables, so
        a product-only species named in an observable that a rate law consumes
        still feeds back into the dynamics. Clause 4 drops the boundary
        conditions, whose derivative is zeroed anyway, so they never hold the
        residual up.

        What this does **not** claim is that the returned species are the only
        obstacle to convergence, or that a species *not* returned is settled.
        It answers exactly one question: which species can be dropped from the
        convergence test without changing the problem the remaining ones solve.

        Returns
        -------
        list[str]
            Species names, in species order. Empty for the common case.

        Examples
        --------
        >>> model.pure_sink_species()
        ['Ad()']
        >>> ss = sim.steady_state(mask=~model.is_pure_sink())
        >>> ss.converged
        True
        """
        names = self._core.species_names
        return [names[i] for i in self._core.pure_sink_species()]

    def is_pure_sink(self):
        """Boolean mask of the pure-sink species, shape ``(n_species,)``.

        The array form of :meth:`pure_sink_species`, so the convergence-test
        mask is a negation::

            ss = sim.steady_state(mask=~model.is_pure_sink())

        Returns
        -------
        ndarray of bool
            ``True`` where the species is a write-only accumulator.
        """
        import numpy as np

        flags = np.zeros(self._core.n_species, dtype=bool)
        idx = self._core.pure_sink_species()
        if idx:
            flags[list(idx)] = True
        return flags

    # ─── Table functions ──────────────────────────────────────────────────

    def add_table_function(
        self,
        name: str,
        *,
        file: str | Path | None = None,
        times: list[float] | None = None,
        values: list[float] | None = None,
        index: str = "time",
        method: str | None = None,
    ) -> None:
        """Give a table function of the model new data, or add a table function.

        Under the name of a table function the model has (``table_function_names``:
        a ``tfun`` the ``.net`` declares, or a ``tfun`` written inside a function,
        registered as ``<function>__tfun<k>``), the table takes the new data in
        place, and every engine reads it from then on: the interpreter, the
        compiled code, the SSA and the steady-state solvers, and a
        :class:`Simulator` made before the call. The new data has to be read over
        the same index as the table's, since that is compiled into the model's
        code. It keeps the table's interpolation unless ``method`` names
        another. A clone keeps the data it was cloned with.

        Under a new name the table is added, but nothing in the model reads it:
        a loaded model's expressions are fixed, and none of them can name a
        table the model's source did not declare. A :class:`UserWarning` says
        so, and the table changes no run.

        Parameters
        ----------
        name : str
            Table function name (e.g., ``"cumNcases"``).
        file : str or Path, optional
            Path to a ``.tfun`` file. Mutually exclusive with ``times``/``values``.
        times : list[float], optional
            X (index) values. Must be used with ``values``.
        values : list[float], optional
            Y (function) values. Must be used with ``times``.
        index : str
            Index variable name. Default ``"time"``. Can also be the name of a
            parameter, an observable or a function, for a table not read on
            the clock. New data for a table has to name the table's own index.
        method : str or None
            Interpolation method, ``"linear"`` or ``"step"``. ``None`` (the
            default) keeps the table's own for new data, as a ``.tfun`` file
            carries none, and is ``"linear"`` for a new table.

        Raises
        ------
        ModelError
            If the file cannot be read or data is invalid; if ``name`` is the
            name of a function of the model that is not a table, whose body a
            table cannot replace; or if the model's table of that name is read
            over another index (issue #772). The model is left as it was.
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
        # "" asks the core for the table's own method (linear for a new one).
        normalized_method = "" if method is None else method.strip().lower()
        if method is not None and normalized_method not in {"linear", "step"}:
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
        # The breakpoints the run stops on are read from the tables' data
        # (issue #709), and new data under a table's name moves them.
        self._table_breaks = None
        read = [t["name"] for t in self._core.table_function_tables if t["read"]]
        if name not in read:
            warnings.warn(
                f"nothing in the model reads table function '{name}': a loaded model's "
                "expressions can call only the tables its source declares, so this one "
                "changes no run. To give the model new data, use the name of a table it "
                f"reads: {read if read else 'it has none'} (issue #772).",
                UserWarning,
                stacklevel=2,
            )

    @property
    def n_table_functions(self) -> int:
        """Number of registered table functions."""
        return self._core.n_table_functions

    @property
    def table_function_names(self) -> list[str]:
        """Names of all registered table functions."""
        return self._core.table_function_names

    # ─── Dunder methods ───────────────────────────────────────────────────

    def __reduce__(self) -> NoReturn:
        raise_cannot_pickle(
            "Model",
            "Build it again on the other side from the same source — Model.from_sbml(path), "
            "Model.from_bngl(path), Model.from_net(path) — and use clone() for an "
            "independent copy within this process.",
        )

    def __copy__(self) -> NoReturn:
        # Issue #643. copy.copy() used to hand back a Model sharing this one's
        # engine handle — and since every scrap of a Model's state lives behind
        # that handle, there was nothing left for the "copy" to own: it was an
        # alias with a different id(), and a set_param or set_concentration
        # through it changed THIS model, which then simulated differently with
        # nothing raised. Refusing is loud where that was silent, and the two
        # things a caller reaches for next do not rescue them on their own:
        # copy.deepcopy already raises (it cannot pickle the handle), and
        # clone() is not discoverable from the copy path. So name it here.
        raise_cannot_shallow_copy(
            "Model",
            "everything this model has (all of a Model's state lives behind one handle "
            "into bngsim's compiled extension)",
            "Use clone() for an independent Model.",
        )

    def __repr__(self) -> str:
        return (
            f"Model(species={self.n_species}, reactions={self.n_reactions}, "
            f"observables={self.n_observables}, parameters={self.n_parameters})"
        )


# Why a difference quotient's step can decide its answer (issue #1002), said
# once for the Model.jacobian() refusal and the steady-state one alike.
_STEP_DECIDES_WHY = (
    "A species far below the state's scale is stepped by many times itself at the "
    "first, which reads a term of second order in it as a secant across it, and a "
    "secant moves with its step where a slope does not; stepped by 1.5e-8 of itself, "
    "it misses a term that reads it through a sum with something much larger (a "
    "total minus what is bound): which is right is not known."
)

# A declined attempt that recorded no reason (issue #538), for
# Model.analytical_jacobian_decline and the status read from it.
_NO_DECLINE_RECORD = {"kind": "unrecorded", "reason": "no reason was recorded"}

# How many recoveries a model and its clones keep in each of the issue #726
# memos. Each is one parameter-dependent verdict and a model has few, so a full
# memo is emptied rather than grown.
_DERIVATION_MEMO_MAX = 16

# The key of a cache slot while its answer is written: equal to no key, so an
# interrupted write is recovered again (issue #726). Not None, which is a key.
_UNKEYED = object()


def _remember(memo: dict, key, value) -> None:
    """Keep ``value`` under ``key`` in a memo a model shares with its clones.
    Each step is one dict operation, which is atomic under the GIL, so rows of
    a batch on worker threads can share it; two that race only scan twice."""
    if len(memo) >= _DERIVATION_MEMO_MAX:
        memo.clear()
    memo[key] = value


# To the end of the line or a second ``#``: BNG2.pl writes
# ``unit_conversion=1/Ve`` last and with no blank in it, and an edited file may
# have ``unit_conversion= 1 / Ve``.
_UNIT_CONVERSION = re.compile(r"\bunit_conversion\s*=\s*([^\s#][^#]*?)\s*(?:#|$)")
# A number first, so that the ``e`` of ``6.0221e+23`` is not read as a name.
_NUMBER_OR_NAME = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|([A-Za-z_]\w*)")


def _unclamped(name: str) -> str:
    """*name* without BNG's fixed-species ``$``, at the start or after an
    ``@c::`` prefix, as the .net loader drops it from the species names a model
    holds (net_file_loader.cpp)."""
    if name.startswith("$"):
        return name[1:]
    if name.startswith("@"):
        sep = name.find("::")
        if sep != -1 and name[sep + 2 : sep + 3] == "$":
            return name[: sep + 2] + name[sep + 3 :]
    return name


def _names_in(expression: str) -> list[str]:
    """The identifiers of an expression, in order, numbers' exponents left out."""
    return [m.group(1) for m in _NUMBER_OR_NAME.finditer(expression) if m.group(1)]


def _where_a_folded_parameter_is_changed(what: str) -> str:
    """What to edit to move a parameter folded at load, for a refusal's last
    line: the BNGL source of a network BNG2.pl wrote, or the SBML document. By
    what it was folded into, which a clone and a subset model carry."""
    if "unit_conversion=" in what:
        return "the BNGL source and generate the network again"
    return "the SBML document and load the model again"


def _unit_conversions_in(path: Path) -> list[tuple[str, str]]:
    """``(reaction, expression)`` for each ``unit_conversion=`` comment on a
    reaction line of a ``.net`` file (issue #711).

    The block is found line by line, by the first two words of a line with
    its comment taken off, so a ``reactions_text`` block, a commented-out line
    and a comment that says ``end reactions`` are none of them read for one.
    Lines end where the loader's end, at a newline and nowhere else. (The
    loader is stricter about the marker itself: ``begin reactions`` with one
    blank. A file it reads no reaction from may still have names listed here.)
    """
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:  # pragma: no cover - the core has just read it
        return []
    if "unit_conversion" not in text:
        return []
    found: list[tuple[str, str]] = []
    inside = False
    for line in text.split("\n"):
        code, _, comment = line.partition("#")
        words = code.split()
        if not words:
            continue
        if words[0] in ("begin", "end"):
            inside = words[:2] == ["begin", "reactions"]
            continue
        if not inside:
            continue
        said = _UNIT_CONVERSION.search(comment)
        if said:
            found.append((words[0], said.group(1)))
    return found


def _unit_conversion_folds(conversions, core: Any) -> dict[str, str]:
    """The parameters BNG2.pl folded into rate constants as numbers (issue #711).

    For a reaction of a compartmental model BNG2.pl evaluates the volume factor
    and writes it into the ``.net`` rate as a number, ``0.1*kb``, with the
    expression it came from in a trailing comment, ``#_R1 unit_conversion=1/Ve``.
    The loader reads the number. ``Ve`` stays a parameter that no rate reads:
    ``set_param("Ve", 20)`` took the value and moved nothing (Lf(5) = 14.90
    where a network generated at 20 gives 16.76), and dLf/dVe was an exact 0.

    *conversions* is what :func:`_unit_conversions_in` reads off the file.
    Returns ``{name: what it was folded into}`` for every parameter such a
    comment names, and for every parameter that a derived parameter or a
    function among them reads: with ``vol = 4*r^3`` in the comment, or a
    compartment sized by ``Vcell() = 4*rcell^3``, ``r`` and ``rcell`` set the
    number too. A function is followed and not listed: it is no parameter to
    write. The same registry as the SBML loader's
    (:attr:`Model.frozen_params`), so the write and the sensitivity column are
    refused the same way.
    """
    conversions = list(conversions or ())
    if not conversions:
        return {}
    functions = dict(zip(core.function_names, core.function_expressions, strict=True))
    # A parameter that is not derived has a number here, which names nothing.
    written = {
        str(name): str(expression or "")
        for name, expression in zip(core.param_names, core.param_expressions, strict=True)
    }
    written.update(functions)
    names = set(core.param_names) | set(functions)
    reason: dict[str, str] = {}
    for entry in conversions:
        try:
            index, expression = entry
        except (TypeError, ValueError):
            raise ValueError(
                f"unit_conversions: {entry!r} is not a (reaction, expression) pair"
            ) from None
        for name in _names_in(str(expression)):
            if name in names and name not in reason:
                reason[name] = (
                    "the volume factor BNG2.pl wrote into a rate constant as a number "
                    f"(reaction {index}: unit_conversion={expression})"
                )
    # What a derived parameter or a function among them reads set the number
    # as well. Breadth first, so that one read on two paths is given the
    # reason of the shorter, and of the earlier reaction between two as short.
    pending = list(reason)
    for name in pending:
        for read in _names_in(written.get(name, "")):
            if read in names and read not in reason:
                reason[read] = f"{name!r}, which is read for {reason[name]}"
                pending.append(read)
    return {name: what for name, what in reason.items() if name not in functions}


def _ar_report_map_from_net(
    core: Any, data: dict[str, Any] | None = None
) -> dict[str, tuple[str, str, float]]:
    """Rebuild the assignment-rule report map the flat ``.net`` has no line for.

    ``Model.from_sbml`` reports an AssignmentRule-target species at its rule's
    live value through ``_ar_report_map``, a transform ``Simulator.run`` applies
    at output time. ``sbml_to_net`` cannot write that map down, but the structure
    it writes is unambiguous: the species is emitted ``fixed`` and its rule lives
    under the species' own name — a weighted observable for a linear-on-species
    rule, a function for any other (functions win the name lookup, so every rate
    law already reads the rule; only the reported column was frozen). A fixed
    species that shares its name with a non-identity observable or with a
    function is therefore reported through it, exactly as the source model does
    (#515). Neither BNG2.pl nor a hand-written network produces that collision —
    a BNGL species name is a pattern, and the writer's per-species identity
    group is excluded — so a plain ``.net`` keeps the empty map it always had.
    ``codegen_data`` is consulted only when a name collides, and not at all
    when the caller passes the ``data`` it already holds (the ``.net`` writer).
    """
    species = list(core.species_names)
    obs_names = set(core.observable_names)
    func_names = set(core.function_names)
    hits = [i for i, name in enumerate(species) if name in obs_names or name in func_names]
    if not hits:
        return {}
    if data is None:
        data = core.codegen_data()
    entries = {
        o["name"]: [(int(i), float(f)) for i, f in o["entries"]] for o in data["observables"]
    }
    out: dict[str, tuple[str, str, float]] = {}
    for i in hits:
        s = data["species"][i]
        name = s["name"]
        if not s.get("fixed", False):
            continue
        if name in func_names:
            out[name] = ("expression", name, 1.0)
        elif entries.get(name) not in (None, [(i, 1.0)]):
            out[name] = ("observable", name, 1.0)
    return out
