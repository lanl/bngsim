"""bngsim — Embeddable simulation engine for BioNetGen reaction networks.

Usage::

    import bngsim

    model = bngsim.Model.from_net("model.net")
    sim = bngsim.Simulator(model, method="ode")

    model.set_param("kf", 0.5)
    result = sim.run(t_span=(0, 1000), n_points=1001)

    result.time         # (1001,) ndarray
    result.observables  # (1001, n_obs) ndarray
    result.species      # (1001, n_species) ndarray

See the package README for installation, usage, and API overview.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import logging
from typing import Any

from bngsim._atol import AUTO, TRACKING, TrackingAtol, derive_atol, normalize_atol_vector
from bngsim._codegen import prepare_codegen
from bngsim._eval_spec import EvaluationSpec
from bngsim._exceptions import (
    BngsimError,
    ConversionError,
    ConversionWarning,
    DenseSolverFallbackWarning,
    ModelError,
    ParameterError,
    SensitivityUnsupportedError,
    SimulationError,
    SimulationTimeout,
    SsaBoundaryWarning,
    SsaValidationError,
    StopConditionMet,
    UnderSpecifiedModelError,
)
from bngsim._model import Model
from bngsim._named_array import NamedArray
from bngsim._net_reader import build_model_from_parsed, parse_net_file
from bngsim._nfsim_session import NfsimSession
from bngsim._result import IdentifiabilityReport, Result
from bngsim._rounding import round_half_up
from bngsim._rulemonkey_session import RuleMonkeySession
from bngsim._simulator import (
    Simulator,
    SteadyStateResult,
    normalize_method,
)
from bngsim._ssa_validation import SsaIssue, validate_for_ssa
from bngsim._version import __version__
from bngsim.cache import (
    clean_codegen_cache,
    clear_codegen_cache,
    codegen_cache_info,
    prune_codegen_cache,
)
from bngsim.convert import sbml_to_net
from bngsim.coupling import (
    ConservationError,
    ConservationLedger,
    CouplingMap,
    DiscreteExchange,
    Divider,
    UnitConverter,
    get_compartment_volume,
    make_subset_model,
    moiety_total,
    round_to_counts,
    set_compartment_volume,
)
from bngsim.kernel import ReactionKernel
from bngsim.psa import psa_cost_decision

# NFsim availability flag — True when the C++ extension was built with
# -DBNGSIM_BUILD_NFSIM=ON.  Consumers should use this instead of
# reaching into _bngsim_core.
try:
    from bngsim._bngsim_core import HAS_NFSIM as _HAS_NFSIM

    HAS_NFSIM: bool = _HAS_NFSIM
except (ImportError, AttributeError):
    HAS_NFSIM = False

try:
    from bngsim._bngsim_core import HAS_RULEMONKEY as _HAS_RULEMONKEY

    HAS_RULEMONKEY: bool = _HAS_RULEMONKEY
except (ImportError, AttributeError):
    HAS_RULEMONKEY = False

# SuiteSparse/KLU availability flag — True when the C++ extension was built with
# the KLU sparse direct solver (-DBNGSIM_ENABLE_KLU=ON + SuiteSparse found).
# When False, the ODE backend has only the dense linear solver, so large/sparse
# models factorize the full N×N Jacobian at O(N³). Consumers should use this
# (or capabilities()["features"]["klu"]) to detect a dense-only install. GH #209.
try:
    from bngsim._bngsim_core import HAS_KLU as _HAS_KLU

    HAS_KLU: bool = _HAS_KLU
except (ImportError, AttributeError):
    HAS_KLU = False

# Optimized BLAS dense-factor availability flag (GH #84, promoted to the public
# namespace by GH #269) — True when the C++ extension linked a BLAS/LAPACK
# backend (macOS Accelerate, or a system LAPACK found by find_package(LAPACK)),
# so BNGSIM_LAPACK_DENSE=1 can route dense factorizations through `dgetrf`.
# When False that env var is a no-op and the dense path is always the built-in
# LU: correctness is identical either way, only speed on large dense Jacobians
# differs. Of the published wheels only the macOS ones carry it — the manylinux
# and Windows legs build with no BLAS on the CMake prefix — so on Linux this is
# a source-install capability, and there was previously no supported way to ask.
try:
    from bngsim._bngsim_core import HAS_LAPACK_DENSE as _HAS_LAPACK_DENSE

    HAS_LAPACK_DENSE: bool = _HAS_LAPACK_DENSE
except (ImportError, AttributeError):
    HAS_LAPACK_DENSE = False

# True when the vendored MIR micro-JIT codegen backend is compiled in
# (BNGSIM_ENABLE_MIR=ON, GH #78). Gate the compiler-free JIT path
# (BNGSIM_CODEGEN_JIT=mir) on this; default wheels ship it OFF.
try:
    from bngsim._bngsim_core import HAS_MIR as _HAS_MIR

    HAS_MIR: bool = _HAS_MIR
except (ImportError, AttributeError):
    HAS_MIR = False

# Stale-binary guard (issue #125). In an editable/source checkout the compiled
# _bngsim_core is built separately and does NOT auto-rebuild on import (#23), so
# it can silently lag the live C++ and drive false correctness verdicts. Warn —
# never fail — on any import when the loaded binary is older than its source.
# No-op for installed wheels (no source tree) and opt-out via BNGSIM_NO_BUILD_CHECK.
from bngsim._build_provenance import warn_if_stale as _warn_if_stale

_warn_if_stale()


def _has_module(name: str) -> bool:
    try:
        return _importlib_util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Optional Python dependency flags. libsbml is needed for any SBML- or
# Antimony-loaded model; antimony is additionally needed for .ant input.
# vivarium-core powers the optional bngsim.vivarium process shell (GH #102).
HAS_LIBSBML: bool = _has_module("libsbml")
HAS_ANTIMONY: bool = _has_module("antimony")
HAS_VIVARIUM: bool = _has_module("vivarium")


def _bngl_available() -> bool:
    """Whether ``Model.from_bngl`` can actually run here (GH #162).

    The odd one out among the flags above, and deliberately a *probe* rather than
    an import check: BNGL loading needs an external Perl toolchain, not a module.
    ``pip install 'bngsim[bngl]'`` is one way to get BNG2.pl, ``$BNGPATH`` at a
    BioNetGen you already have is another, and neither is visible to
    ``find_spec``; conversely a machine with ``bionetgen`` importable but no
    ``perl`` — stock Windows — cannot load BNGL at all. Only running the resolver
    answers it.

    Exposed as the lazy module attribute ``bngsim.HAS_BNGL`` (see
    ``__getattr__``) so ``import bngsim`` never pays for it: the resolver may
    ``import bionetgen``, a 12.8 MB package that pulls in libroadrunner, seaborn
    and networkx, and nothing about importing bngsim should.
    """
    from bngsim._bngpath import resolve_bng

    return resolve_bng().ok


# ── Behaviour-level capability probes (GH #431) ──────────────────────────────
#
# The flags above answer "what was compiled in?" and "what optional package is
# installed?". The four probes below answer a different question, and it is the
# one a fitting frontend has to settle before it commits to hours of gradient
# work: does this build COMPUTE the thing correctly?
#
# A version string cannot answer it. bngsim bumps ``__version__`` at the *start*
# of a release cycle, so every from-source build made between that bump and a
# given fix declares the same number as the release that finally carries it. Nor
# can a name probe: these fixes change what a build computes, not what it
# exposes, so nothing in the namespace appears or disappears at any of them.
#
# What makes the question expensive is that the two wrong answers are not
# symmetric. A build without one of these fixes does not refuse the run — it
# returns a finite number with a term missing. A consumer that guesses "absent"
# loses a gradient fit and falls back to something slower; a consumer that
# guesses "present" gets a fit that converges and reports a number with nothing
# wrong on its face. So bngsim publishes the answer instead, in both directions:
# a ``False`` here is an answer a consumer can act on, while a key that is not
# published at all only means "too old to have been asked".
#
# Each probe reads the half of the install that can actually be wrong. Three of
# these fixes are partly C++, and in a source checkout the extension is built
# separately and does not rebuild on import (issue #23), so the compiled half
# can lag this file; those three ask the loaded core for a binding the fix
# added. The fourth can be switched off at runtime, so it reads the switch.
# ``capabilities()["build"]`` covers the case none of them can see: a core that
# has drifted to somewhere *inside* one of these windows.


def _event_sensitivities_available() -> bool:
    """Whether forward sensitivities survive a discrete event here (GH #431).

    ``True`` means the jump this install applies at each event fire,

    .. code-block:: text

        s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p

    carries every term, including the two that used to be dropped in silence:

    * the carried ``∂h/∂x·s⁻`` of an assignment that reads the state it writes
      (``A := A + dose``, the repeat-dosing idiom) — an SBML species token binds
      to its same-named observable, so the difference that computed ``∂h/∂x``
      matched nothing and the assigned row restarted from zero (issue #144);
    * the sensitivity history at a CVODE root that fires nothing, which rewound
      the state and left the sensitivities where they were, injecting a spurious
      step into every column (issue #146).

    Both landed inside one release cycle, which is why no version floor
    separates a build that has them from one that does not. Neither failed
    loudly: measured on 0.12.1, a state-reading assignment reported ``-10.96``
    where the model's own central difference says ``-311.20``.

    What a qualifying build then *supports* is a separate and narrower question.
    A trigger it cannot differentiate — a real delay, a conjunction, a crossing
    too tangential to resolve — is refused per simulation, by name, through
    :class:`SensitivityUnsupportedError`. This key is about the ones it accepts.

    ``NetworkModel.events_with_runtime_event_time_sens`` is the binding issue
    #144 added to the compiled core, so a core that predates the fixes reports
    ``False`` here even when this file is current.
    """
    try:
        from bngsim._bngsim_core import NetworkModel
    except Exception:
        return False
    return hasattr(NetworkModel, "events_with_runtime_event_time_sens")


def _cross_compartment_sensitivities_available() -> bool:
    """Whether a cross-compartment reaction keeps the analytic ∂f/∂p (GH #431).

    A reaction whose species live in compartments of different size — the
    ``per_species_volume_scaling`` flag the SBML loader sets — used to make the
    *whole model* decline the analytic sensitivity RHS. CVODES installs one
    sensitivity-RHS callback for every column, so one such reaction put every
    column on the solver's internal difference quotient (issue #160).

    That fallback is correct, so this is not a wrong number in the way the event
    key is. It is a cost, and on a real model it is the whole run: on
    ``Smith_BMCSystBiol2013`` one transport reaction put all 25 columns on
    difference quotients and every gradient start timed out. A consumer wants to
    know that before it spends the run rather than after.

    ``False`` here means only one thing, because the emitter is unconditional:
    ``BNGSIM_NO_FUNCTIONAL_SENS_RHS=1`` is set in the environment, which is the
    A/B hatch that puts every Functional rate law back on difference quotients.

    A reaction bngsim still cannot cover — an Elementary or Michaelis–Menten law
    carrying the same flag, which no loader emits — is declined per model with a
    reason on the ``bngsim`` logger, and does not change this answer.
    """
    from bngsim._codegen import functional_sens_rhs_enabled

    return functional_sens_rhs_enabled()


def _per_species_atol_available() -> bool:
    """Whether ``Simulator.run(atol=...)`` takes a per-species vector (GH #431).

    A model spanning ten decades has to pick one absolute tolerance for both
    ends unless the tolerance can be a vector. This install routes one to
    ``CVodeSVtolerances`` (issue #196); :data:`AUTO` and
    :func:`normalize_atol_vector` are how a caller derives it.

    ``SolverOptions.atol_vec`` is the compiled setter that carries it, so a core
    without it reports ``False`` — which is the answer that matters, since a
    vector handed to a build that takes only a scalar is not a refusal a caller
    would notice.
    """
    try:
        from bngsim._bngsim_core import SolverOptions
    except Exception:
        return False
    return hasattr(SolverOptions, "atol_vec")


def _tracking_atol_available() -> bool:
    """Whether ``Simulator.run(atol=TrackingAtol(...))`` is honoured (GH #431).

    The over-time twin of the key above (issue #213). A vector read off the
    initial state cannot see a species that starts at order one and decays to
    nothing; :class:`TrackingAtol` installs a ``CVodeWFtolerances`` error-weight
    function that is re-evaluated against the state actually being integrated.

    ``SolverOptions.atol_track_decades`` is the compiled setter it drives. A
    tolerance mode that silently did not apply is the failure that looks like a
    modelling result, so a consumer that offers this mode should refuse it on a
    ``False`` rather than integrate at something else.
    """
    try:
        from bngsim._bngsim_core import SolverOptions
    except Exception:
        return False
    return hasattr(SolverOptions, "atol_track_decades")


def _build_summary() -> dict[str, Any]:
    """Which build this is, for ``capabilities()["build"]`` (GH #431).

    ``{"commit": <str | None>, "stale": <bool>}``. The work is
    ``bngsim._build_provenance.summary``; this is the supported way to read it,
    so a consumer that logs the provenance of a run — the only way to tell two
    installs declaring one version apart — does not have to import a private
    module to do it.

    The scan behind ``stale`` is a few hundred ``stat`` calls in a source
    checkout and nothing at all in an installed wheel, which ships no ``src/``
    to compare against.
    """
    from bngsim._build_provenance import summary

    return dict(summary())


def __getattr__(name: str) -> object:
    """Lazily probe ``HAS_BNGL`` (PEP 562); everything else is a normal attribute.

    Re-probed on each access rather than cached, so a caller that exports
    ``$BNGPATH`` at runtime sees the change. That stays cheap: the only expensive
    step is the first ``import bionetgen``, which ``sys.modules`` memoizes.
    """
    if name == "HAS_BNGL":
        return _bngl_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Core classes
    "Model",
    "Simulator",
    "ReactionKernel",
    # Coupling / state-exchange layer (GH #102 Stage 1)
    "UnitConverter",
    "CouplingMap",
    "DiscreteExchange",
    "round_to_counts",
    "round_half_up",
    "ConservationLedger",
    "ConservationError",
    "moiety_total",
    "Divider",
    "make_subset_model",
    "get_compartment_volume",
    "set_compartment_volume",
    "Result",
    "IdentifiabilityReport",
    "EvaluationSpec",
    "SteadyStateResult",
    "NfsimSession",
    "RuleMonkeySession",
    "NamedArray",
    # Exceptions
    "BngsimError",
    "ConversionError",
    "ConversionWarning",
    "ModelError",
    "ParameterError",
    "SimulationError",
    "SimulationTimeout",
    "SsaBoundaryWarning",
    "DenseSolverFallbackWarning",
    "SensitivityUnsupportedError",
    "SsaValidationError",
    "StopConditionMet",
    "UnderSpecifiedModelError",
    # SSA validation
    "SsaIssue",
    "validate_for_ssa",
    # Functions
    "reserved_names",
    "configure_logging",
    "normalize_method",
    # Per-species absolute tolerance (GH #196, exported by GH #212).
    # `hasattr(bngsim, "AUTO")` is the capability probe for the whole feature;
    # the version string is not one. `derive_atol` is the half a caller needs
    # when the tolerance must be a constant of the model rather than of the
    # state a run happens to start from — `Simulator.auto_atol` reads the live
    # state, this one reads whichever state you hand it.
    "AUTO",
    "derive_atol",
    "normalize_atol_vector",
    # ...and its over-time twin (GH #213): the same vector, re-evaluated against
    # the state being integrated instead of against t=0, through CVODE's
    # CVodeWFtolerances. `hasattr(bngsim, "TrackingAtol")` is the probe for it.
    # A vector says WHICH species; this says WHEN, and a species that starts at
    # order one and decays to something tiny needs the second one said.
    "TRACKING",
    "TrackingAtol",
    # PSA diagnostics (GH #15)
    "psa_cost_decision",
    # Feature flags
    "HAS_NFSIM",
    "HAS_RULEMONKEY",
    "HAS_KLU",
    "HAS_LAPACK_DENSE",
    "HAS_MIR",
    "HAS_LIBSBML",
    "HAS_ANTIMONY",
    "HAS_VIVARIUM",
    # Lazy (see __getattr__ above): a runtime probe for BNG2.pl + perl, not a
    # module check, and not paid for at import.
    "HAS_BNGL",
    "capabilities",
    # Codegen
    "prepare_codegen",
    # Codegen artifact cache (issue #205). The cache is content-addressed and
    # nothing prunes it automatically, so these are the supported way for a
    # notebook or a fitting harness to see how big it has grown and bound it —
    # the same four verbs `bngsim-cache` exposes. `bngsim.cache` carries the rest
    # (the entry taxonomy, the size/duration parsers).
    "codegen_cache_info",
    "clean_codegen_cache",
    "prune_codegen_cache",
    "clear_codegen_cache",
    # .net reader (universal parser)
    "parse_net_file",
    "build_model_from_parsed",
    # Format conversion (GH #211 / #215)
    "sbml_to_net",
]


def reserved_names() -> dict[str, list[str]]:
    """Return dict of reserved constant and function names.

    Returns
    -------
    dict
        ``{"constants": [...], "functions": [...]}``

    Example
    -------
    >>> import bngsim
    >>> names = bngsim.reserved_names()
    >>> "_pi" in names["constants"]
    True
    >>> "time" in names["functions"]
    True
    """
    from bngsim._bngsim_core import (
        reserved_names as _reserved_names,
    )

    return _reserved_names()


def configure_logging(
    level: int = logging.INFO,
    *,
    handler: logging.Handler | None = None,
    fmt: str = "%(asctime)s [bngsim] %(levelname)s %(message)s",
) -> logging.Logger:
    """Configure the ``bngsim`` logger.

    By default, bngsim is silent (no handler attached). Call this
    function to enable log output.

    Parameters
    ----------
    level : int
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    handler : logging.Handler, optional
        Custom handler. Default: ``StreamHandler`` to stderr.
    fmt : str
        Log message format string.

    Returns
    -------
    logging.Logger
        The configured ``bngsim`` logger.

    Examples
    --------
    >>> import bngsim, logging
    >>> bngsim.configure_logging(logging.DEBUG)
    >>> # Now all bngsim operations produce log output
    """
    log = logging.getLogger("bngsim")
    log.setLevel(level)

    if handler is None:
        handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))

    # Avoid duplicate handlers
    if not log.handlers:
        log.addHandler(handler)

    return log


def capabilities() -> dict[str, Any]:
    """Return a structured capability report for this bngsim install.

    Returns
    -------
    dict
        A dict with four keys:

        - ``"version"`` — the bngsim package version string.
        - ``"features"`` — ``dict[str, bool]`` mapping each feature/backend
          name to its availability flag in this install.
        - ``"missing"`` — ``dict[str, str]`` mapping each unavailable
          feature to a human-readable explanation that distinguishes a
          missing compiled backend (rebuild flag) from a missing optional
          Python dependency (``pip install ...``).
        - ``"build"`` — ``{"commit": str | None, "stale": bool}``, which build
          this is (GH #431). See below.

        ``"features"`` always contains the same keys regardless of build:
        ``nfsim``, ``rulemonkey``, ``klu``, ``lapack_dense``, ``mir``,
        ``libsbml``, ``antimony``, ``vivarium``, ``bngl``, ``sbml_import``,
        ``sbml_ssa``, ``sbml_psa``, ``antimony_import``, ``codegen``,
        ``output_sensitivities``, ``effective_ic_sensitivity``,
        ``event_sensitivities``, ``cross_compartment_sensitivities``,
        ``per_species_atol``, ``tracking_atol``.
        ``"missing"`` is empty when every feature is available.

        ``bngl`` reports whether :meth:`bngsim.Model.from_bngl` (and
        ``Model.load("x.bngl")``) can run here — GH #162. Unlike its neighbours
        this one is a **runtime probe, not an import check**: BNGL loading shells
        out to BNG2.pl, so it needs both a locatable BioNetGen (the ``bngl``
        extra, or ``$BNGPATH``/``$BNG2_PL``, or ``BNG2.pl`` on ``PATH``) *and* a
        ``perl`` to run it. Computing it is why ``capabilities()`` can be slow on
        its first call in a process where ``bionetgen`` is installed.

        ``output_sensitivities`` reports whether this install can emit the
        ``(n_times, n_outputs, n_param)`` output-sensitivity tensor via
        ``Result.output_sensitivities()`` (species/observable/expression
        derivatives w.r.t. parameters and ICs). Like ``codegen`` it is always
        ``True`` — it is the capability handshake fitting frontends (e.g.
        PyBNF) gate their gradient path on (GH #207).

        ``effective_ic_sensitivity`` reports whether
        ``Model.effective_ic_sensitivity()`` exists — the ``∂x(0)/∂θ`` matrix a
        frontend reads at setup to compose its own initial-condition chain rule
        without double-counting the seeding the ``parameter`` axis already
        carries (GH #155). A build without it cannot say what the seed matrix
        holds, and every answer a consumer could guess is silently wrong, so
        gate the gradient path on this rather than on a version string.

        ``klu`` reports whether the SuiteSparse/KLU sparse linear solver was
        compiled in. When ``False`` the ODE backend has only the dense solver,
        so large/sparse models factorize the full N×N Jacobian at O(N³) — use
        this to detect a dense-only install before a slow genome-scale run
        (GH #209).

        ``lapack_dense`` reports whether a BLAS/LAPACK backend was linked for
        the optional optimized dense factor (GH #84). Unlike ``klu`` this one
        changes speed and nothing else — the built-in dense LU is used either
        way unless ``BNGSIM_LAPACK_DENSE=1`` is set, and both produce the same
        trajectory — so it answers "will that env var do anything here?".
        ``False`` in the manylinux and Windows wheels, ``True`` in the macOS
        ones (Accelerate) and in a source build on a host with LAPACK.

        The last four are **behaviour keys** (GH #431), and they answer a
        different kind of question from their neighbours: not "was this backend
        compiled in?" but "does this build compute the thing correctly?". They
        exist because a version string cannot answer that — bngsim bumps
        ``__version__`` at the *start* of a release cycle, so a from-source
        build made before a fix declares the same number as the release that
        carries it — and neither can a ``hasattr`` probe, because these fixes
        change what a build computes rather than what it exposes. A build
        without one of them returns a finite number with a term missing rather
        than refusing, so guessing is expensive in a way it is not elsewhere
        here. See the module-level probes for the full story of each:

        - ``event_sensitivities`` — forward sensitivities survive a discrete
          event, carrying the state-reading assignment's ``∂h/∂x·s⁻`` and the
          sensitivity history across a root that fires nothing (issues #144,
          #146).
        - ``cross_compartment_sensitivities`` — a reaction whose species live in
          compartments of different size keeps the analytic ``∂f/∂p`` instead of
          putting every column of the model on CVODES' difference quotients
          (issue #160).
        - ``per_species_atol`` — ``Simulator.run(atol=...)`` accepts a vector
          (issue #196).
        - ``tracking_atol`` — ``Simulator.run(atol=TrackingAtol(...))`` is
          honoured (issue #213).

        ``"build"`` identifies the build itself. Two installs can report the
        same ``version`` and be different builds, and ``build["commit"]`` — the
        commit CMake baked into the compiled extension, or ``None`` when it was
        built outside a git checkout — is the only thing here that tells them
        apart. ``build["stale"]`` is ``True`` when that extension is older than
        the C++ source next to it, which happens only in a source checkout
        (auto-rebuild is off by design, issue #23) and which every version-,
        metadata- and feature-based check passes straight through, because
        nothing in the Python layer moved. bngsim also warns about it at import,
        but a consumer package's import happens before that package configures
        its own logging, so reading it here at a moment of your choosing is the
        supported way to surface it.

        Feature names are stable across releases; new features may be
        added but existing names will not be renamed or removed.

    Examples
    --------
    >>> import bngsim
    >>> caps = bngsim.capabilities()
    >>> set(caps) == {"version", "features", "missing", "build"}
    True
    >>> set(caps["build"]) == {"commit", "stale"}
    True
    >>> caps["features"]["nfsim"] == bngsim.HAS_NFSIM
    True
    >>> caps["features"]["sbml_ssa"] == bngsim.HAS_LIBSBML
    True
    >>> caps["features"]["klu"] == bngsim.HAS_KLU
    True
    >>> caps["features"]["lapack_dense"] == bngsim.HAS_LAPACK_DENSE
    True
    """
    features: dict[str, bool] = {
        "nfsim": HAS_NFSIM,
        "rulemonkey": HAS_RULEMONKEY,
        "klu": HAS_KLU,
        "lapack_dense": HAS_LAPACK_DENSE,
        "mir": HAS_MIR,
        "libsbml": HAS_LIBSBML,
        "antimony": HAS_ANTIMONY,
        "vivarium": HAS_VIVARIUM,
        "bngl": _bngl_available(),
        "sbml_import": HAS_LIBSBML,
        "sbml_ssa": HAS_LIBSBML,
        "sbml_psa": HAS_LIBSBML,
        "antimony_import": HAS_ANTIMONY and HAS_LIBSBML,
        "codegen": True,
        "output_sensitivities": True,
        "effective_ic_sensitivity": True,
        # Behaviour keys (GH #431): what this build computes, not what it
        # exposes. Each probe is documented at its definition above.
        "event_sensitivities": _event_sensitivities_available(),
        "cross_compartment_sensitivities": _cross_compartment_sensitivities_available(),
        "per_species_atol": _per_species_atol_available(),
        "tracking_atol": _tracking_atol_available(),
    }

    missing: dict[str, str] = {}
    if not HAS_NFSIM:
        missing["nfsim"] = (
            "NFsim backend not present in this install "
            "(vendored at third_party/nfsim/ and built by default; this "
            "install was either configured -DBNGSIM_BUILD_NFSIM=OFF or "
            "installed from a wheel that excludes NFsim)"
        )
    if not HAS_RULEMONKEY:
        missing["rulemonkey"] = (
            "RuleMonkey backend not present in this install "
            "(vendored at third_party/rulemonkey/ and built by default; "
            "this install was either configured "
            "-DBNGSIM_BUILD_RULEMONKEY=OFF or installed from a wheel that "
            "excludes RuleMonkey)"
        )
    if not HAS_KLU:
        missing["klu"] = (
            "SuiteSparse/KLU sparse linear solver not compiled into this "
            "install — the ODE backend has only the dense solver, so large/"
            "sparse models run at O(N³). Install SuiteSparse (brew install "
            "suite-sparse / apt-get install libsuitesparse-dev / conda install "
            "-c conda-forge suitesparse) and rebuild from source; if it lives "
            "on a non-standard prefix pass -DCMAKE_PREFIX_PATH or -DKLU_ROOT "
            "(GH #209). A macOS wheel is intentionally dense-only."
        )
    if not HAS_LAPACK_DENSE:
        missing["lapack_dense"] = (
            "no BLAS/LAPACK backend linked, so the optimized dense factor "
            "(BNGSIM_LAPACK_DENSE=1) is a no-op and dense factorizations use "
            "the built-in LU. Results are unaffected — this costs speed on "
            "large dense Jacobians, not correctness. macOS builds get it from "
            "Accelerate with no extra dependency; elsewhere install a LAPACK "
            "(apt-get install liblapack-dev / dnf install lapack-devel / conda "
            "install -c conda-forge openblas) and rebuild from source, passing "
            "-DCMAKE_PREFIX_PATH if it lives on a non-standard prefix (GH #84). "
            "The manylinux and Windows wheels are intentionally built without "
            "one."
        )
    if not HAS_MIR:
        # The one compiled backend that had no entry here at all, so a caller
        # asking why `features["mir"]` was False got a KeyError instead of an
        # answer. It is OFF by default rather than missing by accident, and the
        # message has to say so or it reads as a broken install.
        missing["mir"] = (
            "the MIR micro-JIT codegen backend is not compiled into this "
            "install (vendored at third_party/mir/, but OFF by default as a "
            "prototype, so no published wheel carries it — configure "
            "-DBNGSIM_ENABLE_MIR=ON and build from source to get it). Nothing "
            "needs it: codegen compiles its generated C with a system compiler, "
            "and BNGSIM_CODEGEN_JIT=mir is the opt-in that would use this "
            "in-process JIT instead, for a host with no compiler (GH #78)."
        )
    if not HAS_LIBSBML:
        libsbml_msg = "optional dependency 'python-libsbml' not installed"
        missing["libsbml"] = libsbml_msg
        missing["sbml_import"] = libsbml_msg
        missing["sbml_ssa"] = libsbml_msg
        missing["sbml_psa"] = libsbml_msg
    if not HAS_ANTIMONY:
        missing["antimony"] = "optional dependency 'antimony' not installed"
    if not HAS_VIVARIUM:
        missing["vivarium"] = "optional dependency 'vivarium-core' not installed"
    if not features["bngl"]:
        # The resolver's own trail, not a generic "install the extra": on a box
        # with three BioNetGens the useful sentence is which places were looked
        # in, and a found-BNG2.pl-but-no-perl install needs a different fix
        # entirely. `pip install 'bngsim[bngl]'` is named inside it.
        from bngsim._bngpath import resolve_bng

        missing["bngl"] = f"BNGL loading unavailable — {resolve_bng().why_not()}"
    if not features["antimony_import"]:
        if not HAS_ANTIMONY and not HAS_LIBSBML:
            missing["antimony_import"] = (
                "requires optional dependencies 'antimony' and 'python-libsbml'"
            )
        elif not HAS_ANTIMONY:
            missing["antimony_import"] = "requires optional dependency 'antimony'"
        else:
            missing["antimony_import"] = "requires optional dependency 'python-libsbml'"

    # Behaviour keys (GH #431). The remedy for three of these is a rebuild
    # rather than an install, because what is absent is a fix in the compiled
    # extension, and the fourth is an environment switch the caller set.
    _stale_note = (
        " The Python layer here is current, so this install's compiled "
        "extension is behind it — in a source checkout the extension is built "
        "separately and does not rebuild on import (GH #23). Rebuild it with "
        "`python scripts/rebuild_editable.py`, or install a wheel, which "
        "always carries both halves from one build."
    )
    if not features["event_sensitivities"]:
        missing["event_sensitivities"] = (
            "forward sensitivities do not survive a discrete event in this "
            "install: an event assignment that reads the state loses the "
            "carried ∂h/∂x·s⁻ term, and a root that fires nothing rewinds the "
            "state without rewinding the sensitivity history (GH #144, #146). "
            "Neither refuses — both return a finite tensor with the event's "
            "contribution missing — so a gradient computed over a model with "
            "events here is wrong without saying so." + _stale_note
        )
    if not features["cross_compartment_sensitivities"]:
        missing["cross_compartment_sensitivities"] = (
            "BNGSIM_NO_FUNCTIONAL_SENS_RHS=1 is set in this environment, which "
            "is the A/B hatch that puts every Functional rate law — including "
            "every cross-compartment reaction — back on CVODES' internal "
            "difference quotients (GH #160). Answers stay correct; a "
            "sensitivity run over such a model gets much slower, and on a "
            "stiff one it may not finish. Unset the variable to get the "
            "analytic ∂f/∂p back."
        )
    if not features["per_species_atol"]:
        missing["per_species_atol"] = (
            "this install's compiled extension has no per-species absolute "
            "tolerance (GH #196): SolverOptions.atol_vec is absent, so "
            "Simulator.run(atol=[...]) cannot reach CVodeSVtolerances and a "
            "model spanning many decades has to pick one atol for both "
            "ends." + _stale_note
        )
    if not features["tracking_atol"]:
        missing["tracking_atol"] = (
            "this install's compiled extension cannot follow the trajectory "
            "with the absolute tolerance (GH #213): "
            "SolverOptions.atol_track_decades is absent, so "
            "Simulator.run(atol=TrackingAtol(...)) has nothing to install and "
            "a species that starts at order one and decays to nothing cannot "
            "be resolved at both ends of its own range." + _stale_note
        )

    return {
        "version": __version__,
        "features": features,
        "missing": missing,
        "build": _build_summary(),
    }
