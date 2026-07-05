"""bngsim exception hierarchy.

All bngsim exceptions inherit from :class:`BngsimError`, which inherits from
``RuntimeError``. This allows ``except BngsimError`` to catch all bngsim-specific
errors while still being caught by ``except RuntimeError``.

Hierarchy::

    BngsimError (RuntimeError)
    ├── ModelError            — .net parse failures, invalid model state
    ├── SimulationError       — solver failures (convergence, NaN, etc.)
    ├── SimulationTimeout     — wall-clock budget exceeded during run()
    ├── ParameterError        — unknown parameter name, type mismatch
    ├── SsaValidationError    — SBML construct incompatible with SSA
    └── StopConditionMet      — stop condition triggered (carries partial result)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bngsim._ssa_validation import SsaIssue


class BngsimError(RuntimeError):
    """Base exception for all bngsim errors."""


class ModelError(BngsimError):
    """Error loading or manipulating a model.

    Raised when:
    - A .net file cannot be parsed
    - A model is in an invalid state for simulation
    - A reserved name conflict is detected
    """


class SimulationError(BngsimError):
    """Error during simulation.

    Raised when:
    - CVODE fails to converge
    - NaN/Inf detected in species concentrations
    - Maximum number of steps exceeded
    """


class ParameterError(BngsimError):
    """Error with parameter access.

    Raised when:
    - A parameter name is not found in the model
    - A parameter value is invalid (NaN, Inf, wrong type)
    """


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
        super().__init__(self._format(self.issues, override_attempted))

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

    def __init__(
        self,
        message: str,
        *,
        timeout: float = 0.0,
        elapsed: float = 0.0,
        partial_result: object = None,
    ) -> None:
        super().__init__(message)
        self.timeout = float(timeout)
        self.elapsed = float(elapsed)
        self.partial_result = partial_result


class StopConditionMet(BngsimError):
    """A stop condition was triggered during simulation.

    The partial result up to the trigger point is attached as ``self.result``.

    Attributes
    ----------
    result : Result
        Partial simulation result truncated at the stop point.
    condition : str
        Description of the condition that triggered.
    """

    def __init__(self, message: str, *, result: object = None, condition: str = "") -> None:
        super().__init__(message)
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
