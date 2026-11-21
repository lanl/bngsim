"""The notices of a run, kept on the Result it returns (issue #1703).

A run says what went wrong, or what it did instead of what was asked, in two
ways: a Python warning, and a line logged through the ``bngsim`` logger. Both
pass through filters the caller may have set, or never set: a warning the
default filter has shown once at a line is not shown again, ``-W ignore`` and
a logging level drop them, and none of them is on the Result. So a Result
read later, or by a program that never saw the console, did not say that its
Jacobian was retried on a difference quotient or that its tolerance was above
a species it reports.

Every site that warns or logs during a run goes through :func:`warn` or
:func:`log` here. Each records the notice with the call being made, if one is
collecting, and then warns or logs exactly as the site did before, so filters,
``-W error`` and ``pytest.warns`` see what they always saw.
``python/tests/test_result_notices.py`` holds every other ``warnings.warn`` and
``logger.warning`` in the package to a list of sites no run reaches.

A call that returns Results collects with :func:`collected`. The innermost
collector takes a notice: a run's own notices go on its Result, and a notice
of the call around it (a batch's warning about every row) goes on each Result
the call returns. A collector is per thread, so a batch row run on a worker
thread collects its own.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import sys
import types
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple, TypeVar

#: What each notice code says, one line each. A code is stable across releases;
#: the message is not. Add a code here before a site uses it:
#: test_result_notices.py fails on a code that is used and not listed, or listed
#: and not used.
CODES: dict[str, str] = {
    "atol.below_scale": (
        "Species stayed below their absolute tolerance throughout while reaching a "
        "share of the largest; their error was not controlled."
    ),
    "codegen.cache_key_unavailable": (
        "The structural codegen cache key could not be made; the generated source was "
        "hashed instead."
    ),
    "codegen.failed": (
        "Building the compiled right-hand side failed; the run used the interpreted one."
    ),
    "codegen.skipped": (
        "The compiled right-hand side was not built, because an earlier build of this "
        "model failed; the run used the interpreted one."
    ),
    "codegen.ssa_propensities_interpreted": (
        "The compiled SSA propensities could not be built; the run evaluated them interpreted."
    ),
    "codegen.toolchain": "The C toolchain could not build a codegen library, or builds one again.",
    "config.ignored_setting": (
        "An environment setting had a value bngsim cannot read; a default was used."
    ),
    "equality.detection_failed": (
        "Finding the rate-law equalities a state can leave failed; the run watches fewer of them."
    ),
    "jacobian.difference_quotient": (
        "The analytical Jacobian is not available for this model; the run used a "
        "difference quotient."
    ),
    "jacobian.difference_quotient_kept": (
        "An earlier run on this Simulator failed with the analytical Jacobian and was "
        "retried; this run used the difference quotient from the start."
    ),
    "jacobian.difference_quotient_retry": (
        "The run failed with the analytical Jacobian and was run again from its start "
        "with a difference quotient, whose result it returned."
    ),
    "jax.zero_gradients": "Model functions may give JAX zero gradients at discontinuities.",
    "linear_solver.dense_without_klu": (
        "A large model ran on the dense linear solver because this install has no KLU."
    ),
    "protocol.steady_state_not_reached": (
        "A protocol's run asked to stop at a steady state reached none by its end."
    ),
    "rulemonkey.unsupported": "RuleMonkey reported a construct it does not support as written.",
    "sensitivity.chain_rule_dropped": (
        "A derived parameter's expression could not be differentiated; sensitivities "
        "through it read zero along that path."
    ),
    "sensitivity.chunk_split": (
        "A group of sensitivity columns failed together and each was solved on its own."
    ),
    "sensitivity.detection_failed": (
        "Finding the switch, event or state crossings a sensitivity depends on failed; "
        "a gradient may miss the jump at one."
    ),
    "sensitivity.nan_rows": (
        "Rows of the sensitivity tensor are NaN or infinite where the chain rule they "
        "need is not available."
    ),
    "sensitivity.one_sided": (
        "Sensitivity columns are one-sided derivatives at a switch that shares its "
        "instant with another."
    ),
    "sensitivity.output_budget": (
        "Deriving expression output sensitivities ran past its time budget; the "
        "functions not reached refuse when selected."
    ),
    "sensitivity.parameters_skipped": (
        "Parameters were left out of a sensitivity request because their columns could "
        "not mean what they say."
    ),
    "sensitivity.rhs_difference_quotient": (
        "The analytic sensitivity right-hand side was declined; CVODES' difference "
        "quotient carried the sensitivity columns."
    ),
    "smooth_pulse.event_reset": (
        "A smooth bump on a variable an event assigns is placed from the variable's "
        "start and rate."
    ),
    "ssa.negative_count": "A species count went negative in a stochastic run.",
    "ssa.reverse_fire": "A reaction fired in reverse because its rate law evaluated negative.",
    "ssa.rounded_counts": "Molecule counts that were not whole numbers were rounded at the start.",
    "start.continues_last_run": (
        "The run started from the state the last run left, with the clock set back."
    ),
    "start.rows_from_initial": (
        "The rows started from the initial conditions, not the model's current state, "
        "which differs from them."
    ),
    "stop_condition.non_finite": "A stop condition read NaN or infinity at a sample.",
    "stops.detection_failed": (
        "Finding where a rate law switches on the clock, a crossing or the state "
        "failed; the integrator approaches it without a stop or a root."
    ),
    "switch.noisy_trigger": (
        "An event trigger crosses its threshold within the rounding of its own evaluation."
    ),
    "switch.over_budget": "A condition crosses more often than the stops one run places.",
    "switch.scan_failed": "The scan of a time condition failed; no stop is placed for it.",
    "switch.unread": "A condition on the clock could not be read where the scan asked.",
}


class Notice(NamedTuple):
    """One warning, fallback or complaint of a run."""

    #: The warning's class name (``"AtolScaleWarning"``), or ``"log"`` for a
    #: notice the ``bngsim`` logger gave.
    category: str
    #: A key of :data:`CODES`.
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"category": self.category, "code": self.code, "message": self.message}


_collector: contextvars.ContextVar[list[Notice] | None] = contextvars.ContextVar(
    "bngsim_notices", default=None
)

#: The code of :func:`collected`'s wrapper, which every wrapped call shares. A
#: ``stacklevel`` given to :func:`warn` counts no frame of it, so a warning
#: names the line it named before the call it is raised in was wrapped.
WRAPPER_CODES: set[types.CodeType] = set()

#: A ``stacklevel`` for :func:`warn`: the first frame outside bngsim (and
#: JAX), as ``_simulator._stacklevel_outside_bngsim`` finds it.
OUTSIDE = -1


@contextmanager
def collecting() -> Iterator[list[Notice]]:
    """Collect the notices given in this thread until the block ends, and none
    of an enclosing collector's meanwhile."""
    notes: list[Notice] = []
    token = _collector.set(notes)
    try:
        yield notes
    finally:
        _collector.reset(token)


def note(category: str, code: str, message: str) -> None:
    """Record a notice with the call being made, and give it nowhere else: for
    a site that warns or logs only once however often its condition holds, or
    whose warning is given later on another thread."""
    if code not in CODES:  # pragma: no cover - test_result_notices holds the codes
        raise ValueError(f"unregistered notice code {code!r}")
    notes = _collector.get()
    if notes is not None:
        notes.append(Notice(category, code, message))


def warn(
    message: str,
    category: type[Warning] = UserWarning,
    *,
    code: str,
    stacklevel: int = 1,
    emit: bool = True,
    record: bool = True,
) -> None:
    """``warnings.warn``, after recording the notice. ``stacklevel`` is the
    caller's own, as it would pass it to ``warnings.warn``; ``emit=False``
    records without warning, and ``record=False`` warns of a notice recorded
    already, on another thread."""
    if record:
        note(category.__name__, code, str(message))
    if not emit:
        return
    if stacklevel == OUTSIDE:
        from bngsim._simulator import _stacklevel_outside_bngsim

        # Counted from this frame, the one that calls warnings.warn.
        warnings.warn(message, category, stacklevel=_stacklevel_outside_bngsim())
        return
    # The frame *stacklevel* up from the caller, counting no wrapper frame.
    frame: types.FrameType | None = sys._getframe(1)
    level = 2  # warnings.warn's own count: 1 is this frame, 2 the caller
    for _ in range(stacklevel - 1):
        frame = frame.f_back if frame is not None else None
        level += 1
        while frame is not None and frame.f_code in WRAPPER_CODES:
            frame = frame.f_back
            level += 1
    warnings.warn(message, category, stacklevel=level)


def log(logger: logging.Logger, code: str, msg: str, *args: Any, emit: bool = True) -> None:
    """``logger.warning(msg, *args)``, after recording the notice; ``emit=False``
    records without logging."""
    note("log", code, (msg % args) if args else msg)
    if emit:
        # The record names the line that logged, not this one.
        logger.warning(msg, *args, stacklevel=2)


def seen(code: str) -> bool:
    """Whether the call being made has recorded a notice of *code*."""
    notes = _collector.get()
    return notes is not None and any(n.code == code for n in notes)


def unique(notes: Iterable[Notice]) -> tuple[Notice, ...]:
    """*notes* in order, each once."""
    return tuple(dict.fromkeys(notes))


def add(result: Any, notice: Notice) -> None:
    """Add *notice* to *result*'s own, where it is not among them: for a notice
    of one result of several that a call decides after they are made."""
    result._notices = unique((*result._notices, notice))


def attach(out: Any, notes: list[Notice]) -> None:
    """Add *notes*, a call's own, to every Result the call returns: *out*
    itself, or each Result in a list or tuple of them."""
    if not notes:
        return
    from bngsim._result import Result

    items = out if isinstance(out, (list, tuple)) else (out,)
    for item in items:
        if isinstance(item, Result):
            item._notices = unique((*item._notices, *notes))


_F = TypeVar("_F", bound=Callable[..., Any])


def collected(fn: _F) -> _F:
    """Collect the notices of each call of *fn*, and add them to the Result,
    or the list of Results, it returns, or to the result a
    :class:`~bngsim.StopConditionMet` it raises carries."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # collecting(), spelled out: a run_until step passes through two of
        # these, and a generator's context manager costs it a microsecond.
        notes: list[Notice] = []
        token = _collector.set(notes)
        try:
            out = fn(*args, **kwargs)
        except BaseException as exc:
            if notes:
                from bngsim._exceptions import StopConditionMet

                if isinstance(exc, StopConditionMet):
                    attach(exc.result, notes)
            raise
        finally:
            _collector.reset(token)
        if notes:
            attach(out, notes)
        return out

    WRAPPER_CODES.add(wrapper.__code__)  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
