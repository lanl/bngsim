"""bngsim exception hierarchy.

All bngsim exceptions inherit from :class:`BngsimError`, which inherits from
``RuntimeError``. This allows ``except BngsimError`` to catch all bngsim-specific
errors while still being caught by ``except RuntimeError``.

Hierarchy::

    BngsimError (RuntimeError)
    ├── ModelError            — .net parse failures, invalid model state
    │   └── UnderSpecifiedModelError — reads a symbol it never defines
    ├── SimulationError       — solver failures (convergence, NaN, etc.)
    │   └── SparseJacobianUnavailableError — force_sparse_linear_solver on a
    │                           model with no sparse Jacobian for KLU
    ├── SimulationTimeout     — wall-clock budget exceeded during run()
    ├── ParameterError        — unknown parameter name, type mismatch
    ├── SsaValidationError    — SBML construct incompatible with SSA
    ├── SensitivityUnsupportedError — construct forward sensitivity can't
    │   │                       differentiate (also a ValueError)
    │   └── SensitivityRefusedDuringRunError — the solver's refusal partway
    │       │                   through a run (also a SimulationError)
    │       └── SensitivityRefusedForParametersError — one that turns on
    │                           the parameters requested
    ├── CodegenDeclinedError  — codegen=True on a model codegen declines
    ├── CodegenToolchainError — the C toolchain cannot build any codegen library
    ├── StopConditionMet      — stop condition triggered (carries partial result)
    ├── UsageError            — an argument a call cannot take (also a ValueError)
    ├── DependencyError       — an optional package missing or broken (also an
    │                           ImportError)
    └── MissingFileError      — the file named does not exist (also a
                                FileNotFoundError)

Every one carries a ``code``, ``details``, ``remedies``, ``docs_url`` and
``to_dict()`` (issue #1702); :mod:`bngsim._error_codes` lists the codes.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bngsim._ssa_validation import SsaIssue


class BngsimError(RuntimeError):
    """Base exception for all bngsim errors.

    Besides its message, every one carries what a program needs to act on it
    without reading the message, which can change between releases:

    Attributes
    ----------
    code : str
        A stable code for the kind of failure, such as ``"param.not_found"``.
        An error with no code of its own has its class's generic one
        (``"model.error"``, ``"run.failed"``). An error of a generic class
        raised from a BNGsim error with a specific code, as a loader wraps its
        parser's, carries that one's code, details and remedies.
    details : dict
        What the code is about: the name asked for, the path, the constructs.
    remedies : list of dict
        Ways to fix it a program can act on, each a dict with a ``kind``:
        ``install_package``, ``install_compiler``, ``pass_argument``,
        ``use_name``, ``use_call``, ``use_method`` or ``loosen_tolerance``.
        Empty where BNGsim can name no fix, as for a model it cannot simulate as
        written.
    docs_url : str
        The code's entry on the error-code reference page.
    notices : list of dict
        The notices the failed call had collected when it raised, as
        :attr:`Result.notices <bngsim.Result.notices>` would have held them:
        that the run started from the last run's end state, say, or retried on
        a difference-quotient Jacobian. With them are those of the error this
        one was raised from. Empty for an error raised outside a run.
    """

    #: The code an error of this class carries when it is given none.
    default_code = "error"

    #: The notices of the call that raised it (issue #1857), each a dict as on
    #: a Result; set by bngsim._notices.collected.
    _notices: tuple[dict[str, str], ...] = ()

    def __init__(
        self,
        *args: object,
        code: str | None = None,
        details: dict | None = None,
        remedies: list[dict] | None = None,
    ) -> None:
        super().__init__(*args)
        self._code = code
        self._details = details
        self._remedies = remedies

    def _cause(self) -> BngsimError | None:
        """The BNGsim error this generic one wraps, whose code, details and
        remedies it then carries: a wrapper of a generic class (a ModelError
        "Failed to load ...") given none of its own, raised from an error with a
        specific code. A class with a specific code keeps its own."""
        from bngsim._error_codes import GENERIC

        cause = self.__cause__
        if (
            self._code is None
            and self._details is None
            and self._remedies is None
            and type(self).default_code in GENERIC
            and isinstance(cause, BngsimError)
            and cause.code not in GENERIC
        ):
            return cause
        return None

    @property
    def code(self) -> str:
        if self._code is not None:
            return self._code
        cause = self._cause()
        return cause.code if cause is not None else self.default_code

    # details and remedies are copies, so a caller that edits what it reads
    # does not change the error.
    @property
    def details(self) -> dict:
        if self._details is not None:
            return copy.deepcopy(self._details)
        cause = self._cause()
        return cause.details if cause is not None else {}

    @property
    def remedies(self) -> list[dict]:
        if self._remedies is not None:
            return copy.deepcopy(self._remedies)
        cause = self._cause()
        return cause.remedies if cause is not None else []

    @property
    def notices(self) -> list[dict[str, str]]:
        # The error this one was raised from, or raised while handling: a
        # rewrap without ``from`` chains only through the context.
        cause = self.__cause__
        if cause is None and not self.__suppress_context__:
            cause = self.__context__
        inherited = cause.notices if isinstance(cause, BngsimError) else []
        own = [dict(n) for n in self._notices]
        return inherited + [n for n in own if n not in inherited]

    def _add_notices(self, notices: list[dict[str, str]]) -> None:
        """Add a call's *notices* to those this error already has, each once."""
        have = list(self._notices)
        self._notices = (*have, *(n for n in notices if n not in have))

    @property
    def docs_url(self) -> str:
        from bngsim._error_codes import docs_url

        return docs_url(self.code)

    def to_dict(self) -> dict:
        """The error as plain data: its type, code, message, details, remedies,
        docs URL and notices, for a log, a tool's error result or an agent."""
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "details": self.details,
            "remedies": self.remedies,
            "docs_url": self.docs_url,
            "notices": self.notices,
        }


class ModelError(BngsimError):
    """Error loading or manipulating a model.

    Raised when:
    - A .net file cannot be parsed
    - A model is in an invalid state for simulation
    - A reserved name conflict is detected
    """

    default_code = "model.error"


class UnderSpecifiedModelError(ModelError):
    """The model reads a symbol it never gives a value.

    Raised at load when a ``<parameter>`` has no ``value`` attribute and no
    ``<initialAssignment>``, yet is referenced by a kineticLaw / rule / event /
    initialAssignment expression, or when a kineticLaw's local parameter has no
    ``value`` and the law reads it (issue #580). SBML leaves such a value
    undefined, so there is nothing to integrate; bngsim refuses rather than
    default it to ``0.0``, which would silently zero out every expression that
    consumes it and return a wrong-but-plausible trajectory. RoadRunner refuses
    the global case too (it runs a Level 2 local one at 0).

    That failure mode is not hypothetical. `MODEL1006230032` under
    ``BNGSIM_ALLOW_UNSET_PARAMS=1`` integrates to **all five species identically
    zero for the whole run**, and every cross-engine check — state, sensitivity,
    both engines — agrees, because AMICI defaults the same symbol the same way.
    A model computing nothing, with every verdict green.

    Typed (issue #323) so a caller can tell "this model is under-specified" from
    the other things :class:`ModelError` covers — a ``.net`` parse failure, an
    invalid model state. Parity suites bucket it ``UNSUPPORTED``: a clean,
    documented refusal, not an actionable bngsim bug. Set the parameter's value
    (a global one can take an ``<initialAssignment>`` instead), or opt into the
    legacy lenient behavior with ``BNGSIM_ALLOW_UNSET_PARAMS=1`` (which
    downgrades this to a warning).
    """

    default_code = "model.unset_symbol"


class SimulationError(BngsimError):
    """Error during simulation.

    Raised when:
    - CVODE fails to converge
    - NaN/Inf detected in species concentrations
    - Maximum number of steps exceeded
    """

    default_code = "run.failed"

    def __init__(
        self,
        *args: object,
        code: str | None = None,
        details: dict | None = None,
        remedies: list[dict] | None = None,
    ) -> None:
        # A CVODE failure names SUNDIALS' flag in its message, wherever a wrapper
        # quotes it: "... with flag=-2 (CV_TOO_MUCH_ACC)." (issue #1702).
        if code is None and type(self).default_code == "run.failed" and args:
            import re

            from bngsim._error_codes import SOLVER_FLAGS

            text = str(args[0])
            flag = re.search(r"flag=-?\d+ \((CV_[A-Z_]+)\)", text)
            if flag is not None:
                name = flag.group(1)
                code = SOLVER_FLAGS.get(name, "solver.failed")
                if details is None:
                    details = {"flag": name}
            elif "CVODE made no progress" in text:
                code = "solver.stalled"
            if remedies is None and code in ("solver.too_much_accuracy", "solver.stalled"):
                remedies = [{"kind": "loosen_tolerance"}]
        super().__init__(*args, code=code, details=details, remedies=remedies)


class ParameterError(BngsimError):
    """Error with parameter access.

    Raised when:
    - A parameter name is not found in the model
    - A parameter value is invalid (NaN, Inf, wrong type)
    """

    default_code = "param.error"


class SsaValidationError(BngsimError):
    """An SBML model contains constructs that cannot be simulated under SSA.

    Raised by :class:`bngsim.Simulator` when ``method="ssa"`` is requested
    on a model whose loader-captured SSA issues include any with
    ``severity="error"``. The full issue list (errors and warnings) is
    available as :attr:`issues`.

    The error message advertises ``strict_ssa=False`` as a workaround when
    at least one of the raised codes is overridable. ``override_attempted``
    flips the message to call out that the user did pass
    ``strict_ssa=False`` but the listed codes are non-overridable.
    """

    default_code = "ssa.unsupported_construct"

    # Codes that can NEVER be downgraded to a warning, even with
    # ``strict_ssa=False``. Kept here (not in _simulator.py) so that the
    # error envelope can format its hint without a circular import.
    NON_OVERRIDABLE_CODES = frozenset({"non_integer_stoichiometry", "fast_reaction"})

    def __init__(
        self,
        issues: list[SsaIssue],
        *,
        override_attempted: bool = False,
    ) -> None:
        self.issues = list(issues)
        self.override_attempted = override_attempted
        super().__init__(
            self._format(self.issues, override_attempted),
            details={
                "issues": [
                    {"code": i.code, "severity": i.severity, "location": i.location}
                    for i in self.issues
                    if i.severity == "error"
                ]
            },
            remedies=[{"kind": "use_method", "method": "ode"}],
        )

    def __reduce__(self) -> tuple:
        # BaseException rebuilds an error from its args, here the message, which
        # __init__ would read as the issues (issue #1702: a worker's error
        # crosses to its parent by pickle).
        # The state carries what was set after __init__, as the notices.
        return (
            _rebuild_ssa_validation_error,
            (self.issues, self.override_attempted),
            self.__dict__,
        )

    @classmethod
    def _format(cls, issues: list[SsaIssue], override_attempted: bool = False) -> str:
        errors = [i for i in issues if i.severity == "error"]
        if not errors:
            return "Model has SSA-validation issues but none are errors."
        n = len(errors)
        plural = "s" if n != 1 else ""
        if override_attempted:
            lead = (
                f"Model has SSA-validation error{plural} that strict_ssa=False "
                f"cannot override ({n} error{plural}). The codes below sit "
                "in the non-overridable set because they violate SSA's "
                "discrete-fire kernel rather than just the kineticLaw shape:"
            )
        else:
            lead = f"Model is incompatible with SSA simulation ({n} error{plural}):"
        lines = [lead]
        for iss in errors:
            loc = f" [{iss.location}]" if iss.location else ""
            lines.append(f"  - {iss.code}{loc}: {iss.message}")
        # Advertise the strict_ssa=False lever ONLY when (a) we're not
        # already inside a non-overridable raise, and (b) at least one of
        # the listed codes is in fact overridable. Otherwise the hint
        # would mislead the user into thinking the flag will help.
        if not override_attempted:
            overridable = [i for i in errors if i.code not in cls.NON_OVERRIDABLE_CODES]
            if overridable:
                lines.append("")
                lines.append(
                    "To bypass this gate and accept approximate SSA "
                    "dynamics on the overridable codes, pass "
                    "strict_ssa=False to bngsim.Simulator(...). Note: "
                    + ", ".join(sorted(cls.NON_OVERRIDABLE_CODES))
                    + " cannot be overridden — those need a model fix."
                )
        return "\n".join(lines)


def _rebuild_ssa_validation_error(
    issues: list[SsaIssue], override_attempted: bool
) -> SsaValidationError:
    return SsaValidationError(issues, override_attempted=override_attempted)


class SensitivityUnsupportedError(BngsimError, ValueError):
    """Forward sensitivity is unavailable for this model, by declaration.

    A *clean refusal*, not a failure. bngsim inspected the model, found a
    construct whose derivative it cannot produce correctly, and declined rather
    than return silently-wrong numbers. Two constructs raise it today:

    - an **event** whose crossing time moves with a requested parameter and
      whose ``dt*/dp`` bngsim cannot compute, so every post-jump sensitivity
      would be stale (GH #205, remaining subclasses tracked in issue #144); and
    - a **rate law** codegen could not differentiate to closed form — a
      non-smooth ``min``/``max``/``abs``/``floor``, ``rateOf()`` inside a rate
      law, or an unparseable expression (GH #214). bngsim refuses rather than
      fall back to finite differences.

    The solver's refusals partway through a run are its subclass
    :class:`SensitivityRefusedDuringRunError`, which is a
    :class:`SimulationError` too.

    This is the forward-sensitivity peer of :class:`SsaValidationError`: both
    say "this model exercises a feature this regime does not support", and both
    are distinguishable **by type**, so a caller can separate a declared non-run
    from an actionable bug without pattern-matching on message text. A fitting
    driver can catch it to fall back on a derivative-free optimizer; the parity
    suites bucket it as ``UNSUPPORTED`` (non-scoring) rather than ``EXCEPTION``.

    Also inherits :class:`ValueError`, which both sites raised before this class
    existed, so existing ``except ValueError`` handlers keep working.

    Environment failures are deliberately NOT this exception: "no C compiler and
    no JIT available" is a fixable local problem, not a property of the model,
    and stays a plain ``RuntimeError``.
    """

    default_code = "sens.unsupported"


class SensitivityRefusedDuringRunError(SensitivityUnsupportedError, SimulationError):
    """A forward-sensitivity run the solver refused partway, by declaration.

    The solver reached a construct whose derivative it cannot produce. Raised
    as this type itself, by a condition no choice of requested parameters
    changes: an event that fires as its trigger leaves its threshold with no
    first-order motion (issue #934), a compartment that jumps twice at one
    instant (issue #1181), a delay a ``run_batch`` row made positive (issue
    #144), and the like. The message names it and what to change. A
    :class:`SensitivityUnsupportedError`, as the refusals raised before a run
    are, so a fitting driver or a parity suite tells it from a solver failure
    by its type (issue #1279). And a :class:`SimulationError`, which all but one
    of these were raised as before they had a type, so an existing
    ``except SimulationError`` still catches it.

    A refusal whose condition turns on a requested column (a crossing or a
    fire the column moves, two times it parts) is its subclass
    :class:`SensitivityRefusedForParametersError`, because
    :meth:`Simulator.compute_all_sensitivities` retries a failed chunk of those
    one column at a time and a column alone can run, where it raises this type
    as it is. The one exception is the refusal of a held equality the state
    leaves (issue #1279), which had this type first: where a restart lands on
    the stop, a column that moves it is refused and one that does not runs.
    """

    default_code = "sens.refused_during_run"


class SensitivityRefusedForParametersError(SensitivityRefusedDuringRunError):
    """A forward-sensitivity run the solver refused partway, for the parameters
    requested.

    The condition can turn on which sensitivities were asked for: a crossing
    or a fire a requested parameter moves, two times it moves apart, or an edge
    it moves the run onto, where the result has a kink or a jump and no
    derivative (issues #49, #150, #760, #767, #945, #1025 and others). The
    message names it and, where it can, the parameters to drop. A smaller set
    may run, so
    :meth:`Simulator.compute_all_sensitivities` retries a chunk refused this way
    one column at a time (GH #243) and raises this type, naming the parameters
    refused alone and the ones that ran, when some are refused alone too.

    A :class:`SensitivityRefusedDuringRunError`, so a
    :class:`SensitivityUnsupportedError` and a :class:`SimulationError`: a
    fitting driver or a parity suite tells it from a solver failure by its type,
    and an ``except SimulationError`` that caught these before they had one
    still does (issue #1435).
    """

    default_code = "sens.refused_for_parameters"


class SparseJacobianUnavailableError(SimulationError):
    """``force_sparse_linear_solver=True`` on a model that cannot supply a sparse
    Jacobian for KLU to factorize: its analytical Jacobian is unavailable (or
    suppressed by ``jacobian="fd"``) and its sparsity pattern has no structural
    nonzero to colour for finite differences (issue #1313).

    A refusal by name, not a solver failure: drop the flag to use the dense
    solver. A :class:`SimulationError`, as it was raised before it had a type,
    so an existing ``except SimulationError`` still catches it.
    """

    default_code = "solver.sparse_jacobian_unavailable"


class CodegenDeclinedError(BngsimError):
    """``codegen=True`` on a model codegen declines, such as one whose functions
    read each other (issue #621): a statement about the model, which runs on the
    interpreted RHS with ``codegen=None`` or ``codegen=False`` (issue #1313). The
    deprecated :func:`bngsim.prepare_codegen` raises it for a decline too.

    Distinct from a codegen *build* failure (a compiler error, a timeout), which
    says nothing about the model and stays a plain ``RuntimeError``. A
    ``RuntimeError`` (through :class:`BngsimError`), as it was raised before it
    had a type, so an existing ``except RuntimeError`` still catches it.

    Unrelated to ``bngsim._codegen.CodegenDeclined``, the emitter's private
    signal to decline, which the prepare calls turn into a ``None`` return; this
    is what a caller who asked for compiled code is told.
    """

    default_code = "codegen.declined"


class CodegenToolchainError(BngsimError):
    """The C toolchain cannot build a codegen library for any model (issue #1544).

    Raised when a codegen compile has failed and the same command fails on a
    trivial source too, or when the compiler is not there at all: the failure is
    the environment's (a linker that cannot read the SDK's ``libSystem.tbd``, a
    missing compiler), not the model's. The message names the compiler, the
    linker and, on macOS, the SDK, and suggests an installed SDK that links when
    one does.

    The verdict holds while the compiler and its environment stay as they are
    (it is probed again every 5 minutes), so a later build is refused without
    compiling, and before its source is generated (a model codegen would have
    declined is refused with this too); a library already in the cache is still
    used. ``codegen=True``, forward sensitivities and, under
    ``BNGSIM_CODEGEN_STRICT``, automatic codegen raise it, each in its own words.
    A ``RuntimeError`` (through :class:`BngsimError`), as a build failure was
    raised before it had a type.
    """

    default_code = "env.broken_toolchain"

    #: The words a missing compiler's verdict says, wherever it is quoted.
    _NO_COMPILER = ("C compiler not found", "No C compiler found")

    def __init__(
        self,
        *args: object,
        code: str | None = None,
        details: dict | None = None,
        remedies: list[dict] | None = None,
    ) -> None:
        text = str(args[0]) if args else ""
        if code is None and any(words in text for words in self._NO_COMPILER):
            code = "env.no_compiler"
            if remedies is None:
                remedies = [{"kind": "install_compiler"}]
        super().__init__(*args, code=code, details=details, remedies=remedies)


class SimulationTimeout(BngsimError):
    """Raised when a simulation exceeds its wall-clock budget.

    Distinct from :class:`SimulationError` so callers (e.g. PyBNF's
    ``wall_time_sim``) can classify wall-clock terminations separately from
    solver/convergence failures. Inherits from :class:`BngsimError` (and
    therefore ``RuntimeError``).

    Attributes
    ----------
    timeout : float
        The configured wall-clock budget, in seconds.
    elapsed : float
        Actual elapsed wall-clock time at the point the timeout fired, in
        seconds. ``elapsed >= timeout`` always holds.
    partial_result : Result | None
        Reserved for future use. Currently always ``None`` — bngsim does not
        yet salvage a partial Result from a timed-out run.
    """

    default_code = "run.timeout"

    def __init__(
        self,
        message: str,
        *,
        timeout: float = 0.0,
        elapsed: float = 0.0,
        partial_result: object = None,
    ) -> None:
        from bngsim._error_codes import pass_argument

        super().__init__(
            message,
            details={"timeout": float(timeout), "elapsed": float(elapsed)},
            remedies=[pass_argument("timeout")],
        )
        self.timeout = float(timeout)
        self.elapsed = float(elapsed)
        self.partial_result = partial_result


class StopConditionMet(BngsimError):
    """A stop condition was triggered during simulation.

    The partial result up to the trigger point is attached as ``self.result``.

    Conditions are checked against a *completed* result, so this does not
    interrupt the solve: the backend integrates the whole requested interval and
    leaves the model at its end, and ``self.result`` is that trajectory
    truncated at the first row where the condition held. ``Simulator``'s
    :attr:`~bngsim.Simulator.current_time` therefore reads the end of the
    interval, not the trigger time — the two differ on purpose, and the clock is
    the one that names the state the model holds (issue #553). Take a
    :meth:`~bngsim.Simulator.snapshot` beforehand to continue from the trigger
    point instead.

    Attributes
    ----------
    result : Result
        Simulation result truncated at the stop point. The *reported* species
        block only: ``Result.state`` is not available on it, so it cannot be
        used to restore the model's integrator state.
    condition : str
        Description of the condition that triggered.
    """

    default_code = "run.stop_condition"

    def __init__(self, message: str, *, result: object = None, condition: str = "") -> None:
        super().__init__(message, details={"condition": condition})
        self.result = result
        self.condition = condition


class ConversionError(BngsimError):
    """A format conversion (e.g. SBML→.net) cannot be completed faithfully.

    Raised by :func:`bngsim.convert.sbml_to_net` when the source model uses a
    construct that the target format cannot represent without changing the
    model's meaning — for example an ``hasOnlySubstanceUnits`` species in a
    compartment whose volume is not 1, a cross-compartment reaction that needs
    per-species volume scaling, a rate-rule ODE, a Michaelis–Menten rate-law
    type, or a table (interpolation) function. The message names the offending
    construct and the model so the cause is actionable.

    Pass ``strict=False`` (API) / ``--allow-lossy`` (CLI) to downgrade these to
    a :class:`ConversionWarning` and emit a best-effort ``.net`` anyway.
    """

    default_code = "conversion.error"


class UsageError(BngsimError, ValueError):
    """An argument a call cannot take: one missing, out of range, or of a kind
    the call does not accept (issue #1702).

    Also a :class:`ValueError`, as these were raised before they had a type, so
    an existing ``except ValueError`` still catches it.
    """

    default_code = "usage.invalid_argument"


class DependencyError(BngsimError, ImportError):
    """A call needs an optional package that is not installed, or that is
    installed and cannot be imported (issue #1702). For a missing one, its
    remedy gives the ``pip install`` command.

    Also an :class:`ImportError`, as these were raised before they had a type,
    so an existing ``except ImportError`` still catches it.
    """

    default_code = "env.missing_dependency"

    def __init__(
        self,
        *args: object,
        code: str | None = None,
        details: dict | None = None,
        remedies: list[dict] | None = None,
    ) -> None:
        super().__init__(*args, code=code, details=details, remedies=remedies)
        # ImportError's own __init__ comes after RuntimeError's in the MRO and
        # does not run; set what it would.
        # As ImportError sets it: the one argument, or None; typeshed says str.
        self.msg = args[0] if len(args) == 1 else None  # type: ignore[assignment]


class MissingFileError(BngsimError, FileNotFoundError):
    """The file a call names does not exist (issue #1702). ``details["path"]``
    is the path it looked for.

    Also a :class:`FileNotFoundError`, as these were raised before they had a
    type, so an existing ``except FileNotFoundError`` still catches it.
    """

    default_code = "file.not_found"

    def __init__(self, message: str, path: object = None) -> None:
        super().__init__(message, details=None if path is None else {"path": str(path)})


class ConversionWarning(UserWarning):
    """A format conversion dropped or approximated part of the source model.

    Emitted when the network channel is preserved but something outside it was
    left behind — most commonly SBML ``<event>`` elements, which belong to the
    simulation-protocol channel (a SED-ML sidecar), not the ``.net`` network.
    Also used when ``strict=False`` downgrades an otherwise-fatal
    :class:`ConversionError`. Filter or escalate it like any ``UserWarning``::

        import warnings, bngsim
        warnings.simplefilter("error", bngsim.ConversionWarning)   # promote
        warnings.simplefilter("ignore", bngsim.ConversionWarning)  # silence
    """


class SsaBoundaryWarning(UserWarning):
    """An SSA run hit a literal-rate-law boundary condition (GH #110).

    Emitted once per run when the exact SSA either drove a species count
    negative (no non-negativity floor is applied — non-negativity is the rate
    law's responsibility, matching the CVODE path) or fired a reaction in
    reverse because its rate law evaluated negative. Both keep the SSA mean
    consistent with the ODE; the warning surfaces what would otherwise be a
    silent boundary behavior. Filter or escalate it like any ``UserWarning``::

        import warnings, bngsim
        warnings.simplefilter("error", bngsim.SsaBoundaryWarning)  # promote
        warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)  # silence

    The structured counts are always available on ``result.ssa_diagnostics``
    regardless of warning filters.
    """


class SsaRoundingWarning(UserWarning):
    """An SSA run rounded a fractional molecule count to a whole one (issue #718).

    The exact SSA fires whole molecules, so a species whose count is not a
    whole number when a run starts is rounded to the nearest one, as
    ``run_network`` does. This is emitted for every run and every ``run_until``
    leg that rounds, not only when the Simulator is built: a count set between
    legs, or left fractional by an event assignment, is rounded at the next leg.
    A parameter or compartment that an event assigns and a rate-rule target
    hold continuous values and are never rounded. ``result.ssa_diagnostics``
    carries the count (``n_rounded_populations``) regardless of filters::

        import warnings, bngsim
        warnings.simplefilter("error", bngsim.SsaRoundingWarning)  # promote
    """


class RuleMonkeyWarning(UserWarning):
    """RuleMonkey runs the model but reports how it reads part of it (issue #1022).

    Emitted when ``method="rm"`` (``"nf_exact"``) or a
    :class:`bngsim.RuleMonkeySession` loads a BNG XML for which RuleMonkey
    reports a warning-level diagnostic: a rule whose products can never be what
    it writes (BioNetGen generates no reaction for it, and RuleMonkey never
    fires it), a Michaelis-Menten law over a pattern that matches several
    species (BNG2 expands it per species pair, RuleMonkey sums the counts), an
    observable read at local and global scope in one function, a rule priority,
    ``TotalRate``, and the like. Each message names the rule or construct and
    says what RuleMonkey does with it. Errors are refused instead. Filter or
    escalate it like any ``UserWarning``::

        import warnings, bngsim
        warnings.simplefilter("error", bngsim.RuleMonkeyWarning)  # promote
        warnings.simplefilter("ignore", bngsim.RuleMonkeyWarning)  # silence
    """


class AtolScaleWarning(UserWarning):
    """An ODE run's default absolute tolerance is above species that matter (issue #1361).

    BNG2.pl's scalar ``atol`` of ``1e-8`` was chosen for species held as
    molecule counts. A model in molar units (an SBML model of species at 1e-12
    to 1e-6 M, say) holds species below it, and CVODE then controls none of
    their error: the binding example of the issue returned a negative
    concentration, and BIOMD0000000002, whose ligand is above it and whose
    complexes are below it, errors of 1e-2 against its reference. The default
    now takes each species' scale from the state the run starts from (issue
    #1420, see :meth:`Simulator.run`), which fixes both. What it cannot scale
    is a species that starts at 0, or below every scale the start gives, in a
    model where no species starts small: one whose species all start at 0
    runs at the scalar.

    Emitted by :meth:`Simulator.run` (and so ``run_until``), ``run_batch``,
    ``parameter_scan`` and ``compute_all_sensitivities`` when the caller gave
    no ``atol`` and a species the solver integrates stays below its tolerance
    all run while reaching at least 1e-6 of the run's largest such species. A
    ``$``-fixed species or an assignment-rule target is not integrated and is
    never named. ``atol="tracking"`` (a tolerance that follows each species'
    magnitude, issue #213) is the remedy it names. Filter or escalate it like
    any ``UserWarning``::

        import warnings, bngsim
        warnings.simplefilter("ignore", bngsim.AtolScaleWarning)  # silence
        warnings.simplefilter("error", bngsim.AtolScaleWarning)  # promote
    """


class LiveStartWarning(UserWarning):
    """A run starts from an earlier run's end state with its clock set back (issue #1701).

    The state lives in the :class:`~bngsim.Model`, and a run writes its end
    state back there, so a second ``run()`` continues from where the first one
    ended. A second ``run(t_span=(0, 5))`` of ``A -> B`` from ``A = 10``
    returned ``A(0) = 0.0674`` on the same time axis as the first, with nothing
    to say so. BNG2.pl's successive ``simulate`` actions and libRoadRunner's
    ``simulate`` calls do the same, so it is no bug, but a caller who runs again
    to compare, or on a finer grid to confirm, gets a trajectory from a start it
    did not mean.

    Raised before anything integrates, where the model holds exactly the state
    the last ODE, SSA or PSA run wrote back, that state is not the model's
    initial conditions, and the call starts before the time that run ended at:

    - :meth:`Simulator.run`, whose ``start="initial"`` runs from the initial
      conditions and ``start="live"`` continues on purpose, neither warned;
    - :meth:`Simulator.run_until`, from the Simulator's own clock, as a second
      Simulator's first step after another one's run is; ``set_time`` puts
      the clock at the time the state is at;
    - :meth:`Simulator.run_replicates`, :meth:`Simulator.parameter_scan`
      (unless ``reset_to`` names the start, or the scan carries an
      equilibration's ``dx/dθ`` into its points) and
      :meth:`Simulator.bifurcate`, once per call;
    - :func:`bngsim.run_protocol`, for the protocol's first run.

    A call at or after the time the run ended at continues it (``t_span=(5,
    10)`` after a run that ended at 5), and is not warned of; nor is a state
    the caller changed since, to values other than the run left. Where a call
    has no ``start=``, :meth:`Model.reset` starts it from the initial
    conditions and :meth:`Model.save_concentrations` makes the state the
    initial one. :attr:`Result.started_from` records which start a run had,
    warned or not.
    """


class ScanDoseWarning(UserWarning):
    """A parameter scan cannot move the species its parameter seeds (refs #1643).

    Each point of :meth:`Simulator.parameter_scan` starts from the state at the
    call (or the ``reset_to`` snapshot), and a parameter write moves a species
    only while it sits on its initial condition (:meth:`Model.set_param`).
    After a run it does not: a scan over ``A0 = 5`` and ``40`` started both
    points at the run's ``A = 6.07``. In a sensitivity scan no write moves a
    species at all, and after :meth:`Model.save_concentrations` with no label
    the saved state is the baseline, which no write moves. The scan keeps its
    start, which a pre-equilibrated protocol relies on, and this warning names
    the species whose dose is dropped.

    Raised once per call, before the first point runs, by a resetting scan with
    no ``on_point`` hook (a hook may be what doses). :meth:`Model.reset` first
    gives a dose scan from the initial conditions; an ``on_point`` hook that
    sets each named species doses from the current state. Filter or escalate it
    like any ``UserWarning``::

        import warnings, bngsim
        warnings.simplefilter("error", bngsim.ScanDoseWarning)  # promote
        warnings.simplefilter("ignore", bngsim.ScanDoseWarning)  # silence
    """


class OneSidedSensitivityWarning(UserWarning):
    """A sensitivity column is a one-sided derivative: the run is on a kink (GH #1822).

    Two rate-law switches share an instant, a rate law composes them (a product
    of their gates, or a window between them), and one of them cannot be moved
    alone by any requested parameter. With ``sigma = t0 + t_delta`` and
    ``t_delta = 0`` that is ``t0``: the gate at ``t0`` and the gate at ``sigma``
    open together, and a rate ``U()*Lambda()`` is on only where both are. As a
    function of ``t_delta`` the solution has a kink at 0. Raising ``t_delta``
    delays the rate; lowering it does nothing, ``t0`` then being the later of
    the two.

    The column returned for such a parameter is the derivative in the direction
    that delays its switch past the other one, which the warning names. With
    ``sigma = t0 + t_delta`` and ``t_delta >= 0`` that is the feasible side of
    the bound, the derivative a bounded optimizer wants. bngsim cannot see the
    bound: with ``sigma = t0 - t_delta`` the direction named is ``t_delta``
    decreasing, the infeasible one, and the feasible derivative is another
    number. In the other direction bngsim does not compute it. The columns of every other
    parameter are ordinary two-sided derivatives.
    :attr:`bngsim.Result.one_sided_sensitivities` lists the same thing as data.
    Filter or escalate it like any ``UserWarning``::

        import warnings, bngsim
        warnings.simplefilter("ignore", bngsim.OneSidedSensitivityWarning)  # silence
        warnings.simplefilter("error", bngsim.OneSidedSensitivityWarning)  # promote
    """


class DenseSolverFallbackWarning(UserWarning):
    """A large ODE model is running on the dense solver for lack of KLU (GH #209).

    Emitted once per process when ``Simulator.run()`` integrates a large model
    (``n_species`` past the warn threshold) on the **dense** linear solver only
    because this bngsim install was built **without** SuiteSparse/KLU — not
    because the user asked for ``force_dense_linear_solver`` or
    ``jacobian="jax"`` (both legitimately dense). Dense LU factorizes the full
    N×N Jacobian at O(N³); for a sparse genome-scale network that is the
    difference between minutes and hours. The fix is to rebuild bngsim with KLU
    (see :func:`bngsim.capabilities` and GH #209). Filter or escalate it like any
    ``UserWarning``::

        import warnings, bngsim
        warnings.simplefilter("ignore", bngsim.DenseSolverFallbackWarning)  # silence
        warnings.simplefilter("error", bngsim.DenseSolverFallbackWarning)  # promote
    """
