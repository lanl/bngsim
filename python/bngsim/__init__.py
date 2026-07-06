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

from bngsim._codegen import prepare_codegen
from bngsim._eval_spec import EvaluationSpec
from bngsim._exceptions import (
    BngsimError,
    ConversionError,
    ConversionWarning,
    DenseSolverFallbackWarning,
    ModelError,
    ParameterError,
    SimulationError,
    SimulationTimeout,
    SsaBoundaryWarning,
    SsaValidationError,
    StopConditionMet,
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
    "SsaValidationError",
    "StopConditionMet",
    # SSA validation
    "SsaIssue",
    "validate_for_ssa",
    # Functions
    "reserved_names",
    "configure_logging",
    "normalize_method",
    # Feature flags
    "HAS_NFSIM",
    "HAS_RULEMONKEY",
    "HAS_KLU",
    "HAS_MIR",
    "HAS_LIBSBML",
    "HAS_ANTIMONY",
    "HAS_VIVARIUM",
    "capabilities",
    # Codegen
    "prepare_codegen",
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
        A dict with three keys:

        - ``"version"`` — the bngsim package version string.
        - ``"features"`` — ``dict[str, bool]`` mapping each feature/backend
          name to its availability flag in this install.
        - ``"missing"`` — ``dict[str, str]`` mapping each unavailable
          feature to a human-readable explanation that distinguishes a
          missing compiled backend (rebuild flag) from a missing optional
          Python dependency (``pip install ...``).

        ``"features"`` always contains the same keys regardless of build:
        ``nfsim``, ``rulemonkey``, ``klu``, ``libsbml``, ``antimony``,
        ``vivarium``, ``sbml_import``, ``sbml_ssa``, ``sbml_psa``,
        ``antimony_import``, ``codegen``, ``output_sensitivities``.
        ``"missing"`` is empty when every feature is available.

        ``output_sensitivities`` reports whether this install can emit the
        ``(n_times, n_outputs, n_param)`` output-sensitivity tensor via
        ``Result.output_sensitivities()`` (species/observable/expression
        derivatives w.r.t. parameters and ICs). Like ``codegen`` it is always
        ``True`` — it is the capability handshake fitting frontends (e.g.
        PyBNF) gate their gradient path on (GH #207).

        ``klu`` reports whether the SuiteSparse/KLU sparse linear solver was
        compiled in. When ``False`` the ODE backend has only the dense solver,
        so large/sparse models factorize the full N×N Jacobian at O(N³) — use
        this to detect a dense-only install before a slow genome-scale run
        (GH #209).

        Feature names are stable across releases; new features may be
        added but existing names will not be renamed or removed.

    Examples
    --------
    >>> import bngsim
    >>> caps = bngsim.capabilities()
    >>> set(caps) == {"version", "features", "missing"}
    True
    >>> caps["features"]["nfsim"] == bngsim.HAS_NFSIM
    True
    >>> caps["features"]["sbml_ssa"] == bngsim.HAS_LIBSBML
    True
    >>> caps["features"]["klu"] == bngsim.HAS_KLU
    True
    """
    features: dict[str, bool] = {
        "nfsim": HAS_NFSIM,
        "rulemonkey": HAS_RULEMONKEY,
        "klu": HAS_KLU,
        "mir": HAS_MIR,
        "libsbml": HAS_LIBSBML,
        "antimony": HAS_ANTIMONY,
        "vivarium": HAS_VIVARIUM,
        "sbml_import": HAS_LIBSBML,
        "sbml_ssa": HAS_LIBSBML,
        "sbml_psa": HAS_LIBSBML,
        "antimony_import": HAS_ANTIMONY and HAS_LIBSBML,
        "codegen": True,
        "output_sensitivities": True,
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
    if not features["antimony_import"]:
        if not HAS_ANTIMONY and not HAS_LIBSBML:
            missing["antimony_import"] = (
                "requires optional dependencies 'antimony' and 'python-libsbml'"
            )
        elif not HAS_ANTIMONY:
            missing["antimony_import"] = "requires optional dependency 'antimony'"
        else:
            missing["antimony_import"] = "requires optional dependency 'python-libsbml'"

    return {
        "version": __version__,
        "features": features,
        "missing": missing,
    }
