"""bngsim.Model — High-level Python wrapper for NetworkModel.

This class delegates to the C++ ``NetworkModel`` and provides Python-friendly
helpers for loading models, updating parameters, and inspecting model state.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bngsim._exceptions import ModelError, ParameterError, UnderSpecifiedModelError


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

logger = logging.getLogger("bngsim")

# Suffix → factory-method name, for Model.load() dispatch. ``.sbml`` is accepted
# alongside ``.xml`` because SBML is served under both (BioModels uses .xml).
_LOAD_DISPATCH: dict[str, str] = {
    ".ant": "from_antimony",
    ".xml": "from_sbml",
    ".sbml": "from_sbml",
    ".net": "from_net",
}


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
        "_time_disc_conditions",
        "_want_output_sens",
        "_output_sens_analysis",
        "_named_conc_states",
        "_named_sens_seeds",
        "_declared_ic_sens",
        "_ic_write_log",
        "_guarded_functions",
    )

    def __init__(self, _core: NetworkModel) -> None:
        self._core = _core
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
        # Issue #305: the GH #72 discontinuity-trigger conditions this model
        # registered, verbatim, as source text. Empty for every model with no
        # time-dependent piecewise. Simulator.run resolves the fixed-time ones
        # to crossing times and stops the step on each, because a registered
        # root is only reachable on a step CVODE accepts. See _sbml_loader.py.
        self._time_disc_conditions: tuple[str, ...] = ()
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
        self._guarded_functions: list[tuple[str, str, str]] = _guard_function_expressions(_core)

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
        =============  =====================

        Matching is case-insensitive. ``.bngl`` is *not* loadable: bngsim has no
        BNGL parser, so a BNGL model must be expanded to a ``.net`` network by
        BNG2.pl first (``pip install bionetgen`` ships it). Note this is not the
        parity_checks/ ``parity`` group, which pins an exact PyBioNetGen commit
        for engine-routing provenance rather than for BNG2.pl.

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
            (issue #164 — see :meth:`from_sbml`). Rejected for ``.net``, whose
            volumes BNG2.pl already folded into the network's rate constants.

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
            or ``compartment_sizes`` is given for a ``.net`` model.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        factory = _LOAD_DISPATCH.get(suffix)
        if factory is None:
            known = ", ".join(sorted(_LOAD_DISPATCH))
            hint = ""
            if suffix == ".bngl":
                hint = (
                    " BNGL is not loadable directly — bngsim has no BNGL parser;"
                    " expand the model to a .net network with BNG2.pl first."
                )
            raise ModelError(
                f"Cannot infer a model format from {path.name!r} "
                f"(suffix {suffix or 'missing'!r}); expected one of: {known}. "
                f"Use the format-specific factory to load it explicitly.{hint}"
            )
        if compartment_sizes and factory == "from_net":
            # A .net network is post-BNG2.pl: the compartment volumes are already
            # folded into its rate constants and there is no compartment left to
            # override. Say so rather than accept a dict that would do nothing
            # (issue #164 exists because a silently-dropped volume write is the
            # expensive failure).
            raise ModelError(
                f"compartment_sizes= is not supported for .net models ({path.name}): "
                f"BNG2.pl folds compartment volumes into the generated network's rate "
                f"constants, so the loaded model has no compartment size to set. "
                f"Regenerate the network from BNGL at the volume you want."
            )
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
        # GH #97: same warm-clone reasoning for the #198 output-sens analysis — a
        # clone has the parent's structure, so re-running its sympy would be N×
        # waste in parallel fitting. Shared by reference (the analysis is
        # read-only) and re-keyed on the clone's own counters, so a clone that
        # somehow did not match simply re-derives.
        m._output_sens_analysis = self._output_sens_analysis
        m._ssa_issues = list(self._ssa_issues)
        m._ar_report_map = dict(self._ar_report_map)
        m._varvol_conc_map = dict(self._varvol_conc_map)
        m._varvol_amount_map = dict(self._varvol_amount_map)
        m._varvol_ar_conc_map = dict(self._varvol_ar_conc_map)
        m._varvol_ar_amount_map = dict(self._varvol_ar_amount_map)
        m._varvol_event_resize_map = dict(self._varvol_event_resize_map)
        m._periodic_disc_max_step = self._periodic_disc_max_step
        m._time_disc_conditions = self._time_disc_conditions
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
            one such caller in bngsim is
            :func:`bngsim.jax.differentiable_solve` with ``flat=True``. Has no
            effect on a parameter that has no defining expression.

        Raises
        ------
        ParameterError
            If the parameter name is not found.
        ValueError
            If ``name`` is one of the few SBML compartment sizes this model
            cannot resolve to a live volume, and ``value`` differs from the size
            it was loaded at. A compartment's value is folded at load into
            constants — per-species volume factors, amount-declared initial
            conditions, mass-action rate constants, SSA propensity volumes, the
            emitted RHS, and every ``<initialAssignment>`` that reads it. Issue
            #170 put the volume back into each fold it can reach, so most sizes
            are now ordinary writable parameters; the residue is refused by name
            rather than half-honoured, because honoring only some folds leaves
            the model internally inconsistent rather than moving the volume.
            :attr:`unwritable_compartment_size_params` lists them; load at the
            size you want instead (:meth:`from_sbml` accepts
            ``compartment_sizes={...}``). Writing the value it already holds is
            allowed, so round-tripping a full parameter vector still works.

            Also raised, on the same unchanged-write rule, for a *synthesized
            slot* — :attr:`param_is_internal`. One of those is a **function's**
            name: the engine keeps the function's evaluated value in a parameter
            slot and rewrites it from the function's own expression before every
            derivative evaluation, so a write here is discarded at the next one.
            It used to be accepted — :meth:`get_param` echoed the new value back
            and the trajectory did not move (issue #227). Write the parameters
            the function's expression reads instead.

        Notes
        -----
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
        still sitting on that initial condition: a species a :meth:`run` has
        advanced, or that :meth:`set_concentration` has assigned, keeps its
        value and picks the new initial condition up at the next :meth:`reset`.

        Two cases where the write deliberately does *not* reach an initial
        condition: after :meth:`save_concentrations` with no label (the baseline
        is now a captured state, which the declared initial condition no longer
        describes — dose such a protocol with :meth:`set_concentration`), and
        for an SBML ``initialAssignment`` too complex to be a single parameter
        reference (``2*init_X + offset``), which is evaluated once at load.

        Writing a **derived** parameter — one of the
        :attr:`param_is_expression` symbols, ``d`` in ``d = d__FREE`` or a
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
        try:
            self._core.set_param(name, float(value), force_override=force_override)
        except (KeyError, RuntimeError) as e:
            raise ParameterError(f"Parameter '{name}' not found in model") from e

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

        Parameters
        ----------
        label : str, optional
            Name of the snapshot to restore. When omitted (or ``None``), restores
            the default slot — identical to :meth:`reset` (the seed initial
            conditions, or the last unlabeled :meth:`save_concentrations`). When a
            ``label`` is given, restores the named snapshot saved by
            ``save_concentrations(label)``.

        A snapshot saved with a forward-sensitivity seed (a pre-equilibrated
        state — see :meth:`save_concentrations`) restores that ``dx/dθ`` along
        with the concentrations, so a following
        ``run(carry_sensitivities=True)`` measures from the right seed
        (issue #81).

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

        For forward sensitivity, this is a **literal** initial condition:
        ``∂x_k(0)/∂θ = 0``. It therefore supersedes whatever the species' declared
        initial condition depended on — if the ``.net`` set it from a parameter
        (``R() R0``), that parameter no longer reaches this species' IC and its seed
        row is dropped (issue #113). When the value you assign *does* depend on a
        differentiated parameter, say so with :meth:`declare_ic_sensitivity`.
        """
        try:
            self._core.set_concentration(name, float(value))
        except (KeyError, RuntimeError) as e:
            raise ModelError(f"Species '{name}' not found in model") from e
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
        for sp, row in sens.items():
            if sp not in species:
                raise ModelError(
                    f"declare_ic_sensitivity: species {sp!r} not found in model. "
                    "Names must match species_names exactly."
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
                if pname not in wanted:
                    continue
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

        Detected at model load time for every input format (``.net``, Antimony,
        SBML, programmatic ``ModelBuilder``). The returned dict has keys:

        - ``n_laws``: number of independent conservation laws
        - ``n_species``: number of species the laws are expressed over
        - ``dependent`` / ``independent``: 0-based species index lists
        - ``constants``: conservation constants evaluated from the initial conditions
        - ``coefficients``: ``n_laws`` x ``n_species`` coefficient matrix

        Consumed internally by the reduced-space Newton steady-state solver,
        which needs the independent subspace to sidestep the rank-deficient
        Jacobian these laws imply.
        """
        return self._core.conservation_laws

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
        a non-empty expression paired with ``False`` here is exactly an
        overridden derived parameter, as distinct from a genuine primary. That
        pairing is also the whole of what the generated C source depends on:
        an override moves the chain rule, and nothing else about a parameter
        write does.
        """
        return list(self._core.param_is_expression)

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
