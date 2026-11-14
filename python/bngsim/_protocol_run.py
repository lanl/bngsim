"""Run a BNGL protocol: a ``.bngl`` file's actions, in order, as BNG2.pl runs them.

:meth:`Model.from_bngl` loads a file's network and drops its experiment, and
``protocol=True`` hands the experiment back as a
:class:`~bngsim.convert.ProtocolSpec`. :func:`run_protocol` runs that spec on the
model, and :func:`run_bngl` does all three for a path (issue #1366).

The semantics are BNG2.pl's (``Perl2/BNGModel.pm`` and ``BNGAction.pm``, 2.9.3):

* Every run starts from the state the last one left. ``continue=>1`` also starts
  at the time it left; any other run starts at its ``t_start``, 0 by default.
* ``setParameter`` and ``setConcentration`` change the live model, and
  ``addConcentration`` adds to a species. A species whose amount is a parameter,
  or an expression, follows it until a run makes it a number: BNG2.pl writes the
  expression into the network each run reads. A parameter set to an expression
  of other parameters stays that expression, and follows them through later
  actions, scans and resets, until it is set to a number (issue #1682).
* ``saveConcentrations``/``resetConcentrations`` and
  ``saveParameters``/``resetParameters`` keep one saved state per label.
  ``resetConcentrations()`` with nothing saved goes back to the model as
  declared, and any other reset with nothing saved is an error, as in BNG2.pl.
* ``parameter_scan`` saves the state as ``"SCAN"``, sets the parameter to each
  value, resets to ``"SCAN"`` before each point when ``reset_conc`` is on (the
  default), and again at the end. The parameter keeps its last value.
* ``bifurcate`` scans up and then back down, never resetting, so each point
  starts where the one before ended.
* Tolerances default to ``run_network``'s, 1e-8 absolute and relative, held
  exactly (:class:`~bngsim.ExactAtol`).

What bngsim cannot run as written is refused before anything runs: a
network-free (NFsim) step, which needs a live NFsim session; PLA, which bngsim
does not have; an action that changes results but has no step to run
(``setVolume``, a result-changing ``setOption``, ``simulate_protocol``); and a run
argument the protocol holds only approximately. ``allow_lossy=True`` runs the
last two as nearly as the protocol holds them, with a warning.

Two things BNG2.pl refuses run here instead. A run that names no ``method`` runs
as ODE, the parser's default. A ``steady_state=>1`` run that does not reach a
steady state is warned about and the protocol goes on, where BNG2.pl stops it:
bngsim never calls a model that reads the time steady (issue #710), so stopping
would refuse protocols BNG2.pl finishes.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim._exceptions import ConversionError, ConversionWarning
from bngsim._seed import derived_seeds

if TYPE_CHECKING:
    from bngsim._model import Model
    from bngsim._result import Result
    from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

__all__ = ["ProtocolResult", "ProtocolRun", "run_bngl", "run_protocol"]

# BNG2.pl's label for an unlabeled save or reset (``Perl2/Cache.pm``).
_DEFAULT_LABEL = "DEFAULT"
# run_network's tolerances, as BNG2.pl's simulate passes them when a run gives none.
_BNG_ATOL = 1e-8
_BNG_RTOL = 1e-8
# BNG2.pl's poplevel when a PSA run gives none (it warns, and uses this).
_BNG_POPLEVEL = 100.0
# The methods a protocol step can run here, and what refuses the others.
_NETWORK_METHODS = frozenset({"ode", "ssa", "psa"})
_REFUSED_METHODS = {
    "nf": (
        "a network-free (NFsim) run needs a live NFsim session to carry its state, "
        "which a protocol replay does not run yet (issue #1482)"
    ),
    "rm": (
        "a network-free (RuleMonkey) run needs a live session to carry its state, "
        "which a protocol replay does not run yet (issue #1482)"
    ),
    "pla": "bngsim has no PLA (partitioned-leaping) method",
}
# Hash keys a run reads here.
_RUN_KEYS = frozenset({"continue", "t_start", "atol", "rtol", "seed", "steady_state", "poplevel"})
# Hash keys that only shape the files BNG2.pl writes, or the solver it uses.
_OUTPUT_KEYS = frozenset(
    {
        "print_functions",
        "print_cdat",
        "print_end",
        "print_net",
        "save_progress",
        "print_n_species_active",
        "verbose",
        "prefix",
        "get_final_state",
        "sparse",
        "update_interval",
        "expand",
        "gml",
        "complex",
    }
)
# Hash keys that change what a run does, which a replay does not do here.
_REFUSED_KEYS = {
    "stop_if": "stop_if ends a run when a condition holds",
    "print_on_stop": "print_on_stop goes with stop_if",
    "max_sim_steps": "max_sim_steps ends a run after a number of steps",
    "output_step_interval": "output_step_interval writes rows between the grid's times",
    "netfile": "netfile runs a different network file",
    "argfile": "argfile reads the run's arguments from a file",
    "check_product_scale": "check_product_scale changes how PSA scales products",
    "parallel": (
        "parallel runs a scan's points in forked processes, which leave the parameter "
        "and the clock as they were before the scan"
    ),
}
# The verbs a ProtocolSpec carries as an Experiment, and the build/IO verbs it
# drops cleanly: anything else in ``dropped`` was an unrecognized action.
_EXPERIMENT_VERBS = frozenset(
    {
        "simulate",
        "simulate_ode",
        "simulate_ssa",
        "simulate_nf",
        "simulate_psa",
        "simulate_pla",
        "parameter_scan",
        "bifurcate",
    }
)


@dataclass(frozen=True)
class ProtocolRun:
    """One experiment of a protocol, as it ran.

    A ``simulate`` has its time course in :attr:`result`. A ``parameter_scan``
    has one :class:`~bngsim.Result` per value in :attr:`scan_results`, in the
    order they ran, each carrying ``custom_attrs["scan_parameter"]`` and
    ``custom_attrs["scan_value"]`` as :meth:`Simulator.parameter_scan`'s do. A
    ``bifurcate`` has its upward sweep there and the sweep back down in
    :attr:`backward_results`.
    """

    experiment: Experiment
    result: Result | None = None
    scan_values: tuple[float, ...] = ()
    scan_results: tuple[Result, ...] = ()
    backward_values: tuple[float, ...] = ()
    backward_results: tuple[Result, ...] = ()

    @property
    def kind(self) -> str:
        """``"simulate"``, ``"scan"`` or ``"bifurcate"``."""
        return self.experiment.kind

    @property
    def label(self) -> str | None:
        """The run's ``suffix``, if it has one."""
        return self.experiment.label

    @property
    def method(self) -> str:
        return self.experiment.method


class ProtocolResult(Sequence[ProtocolRun]):
    """What :func:`run_protocol` ran: one :class:`ProtocolRun` per experiment.

    Index it by position, or by a run's ``suffix``. ``model`` is the model in
    the state the protocol left it; ``protocol`` is the spec that ran.
    """

    def __init__(self, runs: Sequence[ProtocolRun], model: Model, protocol: ProtocolSpec):
        self._runs = tuple(runs)
        self.model = model
        self.protocol = protocol

    @property
    def runs(self) -> tuple[ProtocolRun, ...]:
        return self._runs

    @property
    def labels(self) -> list[str | None]:
        """Each run's ``suffix``, in order (``None`` where it has none)."""
        return [r.label for r in self._runs]

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[ProtocolRun]:
        return iter(self._runs)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            hits = [r for r in self._runs if r.label == key]
            if len(hits) == 1:
                return hits[0]
            if not hits:
                raise KeyError(f"no run has the suffix {key!r}; suffixes: {self.labels}")
            raise KeyError(f"{len(hits)} runs have the suffix {key!r}; index them by position")
        return self._runs[key]

    def __repr__(self) -> str:
        parts = [r.label or r.kind for r in self._runs]
        return f"ProtocolResult({len(self._runs)} runs: {', '.join(parts)})"


def run_protocol(
    model: Model,
    protocol: ProtocolSpec,
    *,
    seed: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    allow_lossy: bool = False,
) -> ProtocolResult:
    """Run a BNGL protocol's steps on *model*, in order, as BNG2.pl runs them.

    Parameters
    ----------
    model : Model
        The protocol's network, as :meth:`Model.from_bngl` loads it from the same
        file: the state at the file's ``generate_network``, which already holds
        the actions before it. It is not changed; the protocol runs on a clone.
    protocol : ProtocolSpec
        The steps, from ``Model.from_bngl(path, protocol=True)`` or
        :func:`~bngsim.convert.parse_bngl_protocol`. A spec with no generation
        point (``generation_step`` ``None``: SED-ML, SBML events) runs from its
        first step.
    seed : int, optional
        Seeds the stochastic runs that do not name their own ``seed``: each draws
        its seed from this one, in order. ``None`` draws fresh seeds.
    rtol, atol : float, optional
        Override every ODE run's tolerances. By default a run uses its own
        ``rtol``/``atol``, or ``run_network``'s 1e-8; ``atol`` is held exactly.
    allow_lossy : bool
        Run a protocol that has an action bngsim cannot apply, or a run argument
        it holds only approximately, as nearly as it can, with a
        :class:`~bngsim.ConversionWarning`, instead of refusing it.

    Returns
    -------
    ProtocolResult
        One :class:`ProtocolRun` per experiment, and the model as the protocol
        left it.

    Raises
    ------
    ConversionError
        Before anything runs, if a step cannot be run here: a network-free or PLA
        run, a run argument such as ``stop_if``, an action that changes results
        but carries no step (unless ``allow_lossy``), or an argument the protocol
        holds only approximately (unless ``allow_lossy``). While running, if a
        step names a parameter or species the model does not have, or resets to
        a state nothing saved, which BNG2.pl refuses too.
    """
    _check_runnable(protocol, allow_lossy=allow_lossy)
    runner = _Runner(model.clone(), protocol, seed=seed, rtol=rtol, atol=atol)
    runs = runner.run()
    return ProtocolResult(runs, runner.model, protocol)


def run_bngl(
    path: str | Path,
    *,
    seed: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    allow_lossy: bool = False,
    bng2_pl: str | Path | None = None,
    net_out: str | Path | None = None,
    timeout: int = 600,
    cache: bool = True,
) -> ProtocolResult:
    """Load a ``.bngl`` file and run its actions, as BNG2.pl runs the file.

    The network comes from :meth:`Model.from_bngl` (``bng2_pl``, ``net_out``,
    ``timeout`` and ``cache`` are its arguments), the actions from
    :func:`~bngsim.convert.parse_bngl_protocol`, and they run through
    :func:`run_protocol` (``seed``, ``rtol``, ``atol``, ``allow_lossy``).

    Examples
    --------
    >>> out = bngsim.run_bngl("egfr_path.bngl")             # doctest: +SKIP
    >>> [run.experiment.t_span for run in out]               # doctest: +SKIP
    [(0.0, 100000.0), (0.0, 120.0)]
    >>> out[1].result.observables["Sos_act"][-1]             # doctest: +SKIP
    """
    from bngsim._bngl_loader import load_bngl
    from bngsim.convert._protocol import parse_bngl_protocol

    model = load_bngl(
        path, bng2_pl=bng2_pl, net_out=net_out, timeout=timeout, cache=cache, defer_jacobian=None
    )
    with warnings.catch_warnings():
        # run_protocol says what it cannot run, once; the parse would say it too.
        warnings.simplefilter("ignore", ConversionWarning)
        protocol = parse_bngl_protocol(Path(path), strict=False)
    return run_protocol(model, protocol, seed=seed, rtol=rtol, atol=atol, allow_lossy=allow_lossy)


def _check_runnable(protocol: ProtocolSpec, *, allow_lossy: bool) -> None:
    """Refuse, before anything runs, a protocol this replay cannot run as written."""
    from bngsim.convert._protocol import _BUILD_VERBS, Experiment, _as_bool

    problems: list[str] = []
    lossy_problems: list[str] = []
    # An action before the generation point is in the network the model holds
    # (generation_source keeps it for BNG2.pl), so only those after it count.
    for verb in protocol.lossy[protocol.lossy_before_generation :]:
        if verb.lower() not in _EXPERIMENT_VERBS:
            lossy_problems.append(f"{verb} changes results, and the protocol has no step for it")
    clean = _BUILD_VERBS | {"quit", "setoption"}
    for verb in protocol.dropped[protocol.dropped_before_generation :]:
        if verb.lower() not in clean:
            lossy_problems.append(f"{verb} is not an action the protocol recognizes")
    steps = protocol.steps[protocol.generation_step or 0 :]
    runs = [step for step in steps if isinstance(step, Experiment)]
    for i, step in enumerate(runs):
        where = f"run {i + 1} ({_describe(step)})"
        method = _method_of(step)
        if method in _REFUSED_METHODS:
            problems.append(f"{where}: {_REFUSED_METHODS[method]}")
        elif method not in _NETWORK_METHODS:
            problems.append(f"{where}: method {method!r} is not one bngsim runs")
        for key in step.extra:
            low = str(key).lower()
            if low in _REFUSED_KEYS:
                problems.append(f"{where}: {_REFUSED_KEYS[low]}, which a replay does not do")
        if step.is_scan and not step.scan_parameter:
            problems.append(f"{where}: a scan needs a parameter; BNG2.pl refuses one without")
        if step.sample_times is None and step.n_points < 2:
            problems.append(f"{where}: n_steps must be 1 or more; BNG2.pl cannot run 0")
        if step.output_start is not None and _as_bool(step.extra.get("continue", 0)):
            problems.append(
                f"{where}: an output start past t_start={step.t_span[0]:g} needs the run to "
                "start there, and continue=>1 starts it at the model time"
            )
        for note in step.approximations:
            lossy_problems.append(f"{where}: {note}")
    if problems:
        raise ConversionError(
            "this protocol has steps bngsim cannot run: " + "; ".join(problems) + "."
        )
    if lossy_problems:
        text = "; ".join(lossy_problems)
        if not allow_lossy:
            raise ConversionError(
                f"this protocol is not exactly what bngsim would run: {text}. "
                "Pass allow_lossy=True to run it as nearly as the protocol holds it."
            )
        warnings.warn(
            f"running the protocol as nearly as it holds it: {text}",
            ConversionWarning,
            stacklevel=3,
        )


def _method_of(exp: Experiment) -> str:
    method = exp.method.lower()
    if method in ("ssa", "ode") and "poplevel" in exp.extra:
        return "psa"  # BNG2.pl runs PSA whenever a poplevel is given
    return method


def _describe(exp: Experiment) -> str:
    verb = {"scan": "parameter_scan", "bifurcate": "bifurcate"}.get(exp.kind, "simulate")
    tag = f' suffix=>"{exp.label}"' if exp.label else ""
    return f'{verb} method=>"{exp.method}"{tag}'


def _scan_values(exp: Experiment, *, backward: bool = False) -> list[float]:
    """The values BNG2.pl's ``parameter_scan`` runs, in its order and arithmetic.

    ``par_scan_vals`` runs as given (and a bifurcation runs them in that order
    both ways, since BNG2.pl swaps only ``par_min`` and ``par_max``); a range is
    ``par_min + k*delta`` for ``delta = (par_max - par_min)/(n_scan_pts - 1)``,
    in natural log for ``log_scale``, from ``par_max`` down for the backward
    sweep of a bifurcation."""
    if exp.scan_values is not None:
        return list(exp.scan_values)
    lo, hi, n = exp.scan_min, exp.scan_max, exp.scan_points
    if lo is None or hi is None or not n:
        raise ConversionError(
            f"{_describe(exp)}: a scan needs par_min, par_max and n_scan_pts, or "
            "par_scan_vals; BNG2.pl refuses one without"
        )
    if n < 1 or (n == 1 and lo != hi):
        raise ConversionError(
            f"{_describe(exp)}: n_scan_pts={n} over par_min={lo:g}, par_max={hi:g}; BNG2.pl "
            "needs more than one point between two values, and at least one"
        )
    if backward:
        lo, hi = hi, lo
    if exp.scan_log:
        lo, hi = math.log(lo), math.log(hi)
    delta = 0.0 if n == 1 else (hi - lo) / (n - 1)
    values = [lo + k * delta for k in range(n)]
    return [math.exp(v) for v in values] if exp.scan_log else values


class _Runner:
    """The live state of one replay: the model, the saved states, the clock."""

    def __init__(
        self,
        model: Model,
        protocol: ProtocolSpec,
        *,
        seed: int | None,
        rtol: float | None,
        atol: float | None,
    ):
        self.model = model
        self.protocol = protocol
        self._rtol = rtol
        self._atol = atol
        self._seeds = np.random.SeedSequence(seed)
        self._sims: dict[tuple[str, float | None], Any] = {}
        self.now = 0.0
        # Species whose amount is still an expression BNG2.pl writes into the
        # network each run reads: index → ("param", name, divisor) or ("expr", text).
        self.symbolic: dict[int, tuple] = {
            int(i): ("param", self.model.param_names[int(p)], float(d))
            for (i, p), d in zip(
                self.model._core.species_ic_param_refs,
                self.model._core.species_ic_param_ref_divisors,
                strict=True,
            )
        }
        self.index = {name: i for i, name in enumerate(self.model.species_names)}
        # The model as declared, for a resetConcentrations() with nothing saved:
        # what the network holds, unless an action before generation changed it.
        self.declared: tuple[np.ndarray, dict] | None = (self._state(), dict(self.symbolic))
        self._weights: dict[str, list[tuple[int, float]]] | None = None
        # Parameters a setParameter set to an expression of other parameters:
        # name → the expression. BNG2.pl keeps such a parameter as that
        # expression, so it follows what it reads (issue #1682).
        self.param_exprs: dict[str, str] = {}
        self.saved_conc: dict[str, tuple[np.ndarray, dict] | None] = {}
        self.saved_params: dict[str, tuple[dict[str, float], set[str], dict[str, str]] | None] = {}
        self._before_generation(protocol.steps[: protocol.generation_step or 0])

    # ── bookkeeping ───────────────────────────────────────────────────────
    def _state(self) -> np.ndarray:
        return np.asarray(self.model.get_state(), dtype=float).copy()

    def _before_generation(self, steps: Sequence[Any]) -> None:
        """What the actions before generation left saved. The network holds the
        state they left, so a save among them holds the loaded state where that
        is the state it saved. Where an action between the save and generation
        changed it, the save is the loaded state with the actions that made the
        saved values applied again (issue #1668): a ``setParameter("k2",0)``
        saved and then undone is k2 = 0 again. A saved value that is the one
        declared, which the loaded network no longer holds, is unknown, and so
        is a declared seed the network does not hold; a reset to either is
        refused rather than guessed."""
        from bngsim.convert._protocol import _saved_before_generation

        before = _saved_before_generation(steps)
        if not before.declared:
            self.declared = None
        for label, back in before.conc.items():
            self.saved_conc[label] = self._reached(back, conc=True)
        for label, back in before.params.items():
            self.saved_params[label] = self._reached(back, conc=False)

    def _reached(self, back: tuple | None, *, conc: bool) -> Any:
        """The concentrations (or the parameters) the loaded model reaches with
        the actions *back* applied, which leaves the model as it was; ``None``
        where *back* is unknown or cannot be applied."""
        if back is None:
            return None
        state, symbolic = self._state(), dict(self.symbolic)
        params = self._param_state()
        try:
            for step in back:
                self._change(step)
            return (self._state(), dict(self.symbolic)) if conc else self._param_state()
        except ConversionError:
            return None
        finally:
            self._restore_params(*params)
            self.model.set_state(state)
            self.symbolic = symbolic

    def _value(self, change: StateChange) -> float:
        """A step's value: its number, or its expression in the live model, with
        its observables and functions as they are now (:meth:`_freeze`)."""
        from bngsim._bngl_loader import _value

        if change.value is not None:
            return float(change.value)
        if change.value_expr is None:
            raise ConversionError(f"{_action(change)}: no value to set")
        v = _value(self.model, self._freeze(change.value_expr))
        if v is None:
            raise ConversionError(
                f"{_action(change)}: {change.value_expr!r} does not evaluate in the model"
            )
        return v

    def _observable_weights(self) -> dict[str, list[tuple[int, float]]]:
        if self._weights is None:
            data = self.model._core.codegen_data()
            self._weights = {
                o["name"]: [(int(i), float(f)) for i, f in o["entries"]]
                for o in data["observables"]
            }
        return self._weights

    def _reads_now(self, state: np.ndarray) -> tuple[list[float], dict[str, float]]:
        """The observables, in order, and the functions, at *state* now."""
        weights = self._observable_weights()
        observables = [
            float(sum(f * float(state[i]) for i, f in weights.get(name, [])))
            for name in self.model.observable_names
        ]
        functions = dict(self.model._core._eval_functions(self.now, [float(x) for x in state]))
        return observables, {str(k): float(v) for k, v in functions.items()}

    def _state_reads(self, text: str) -> list[str]:
        """The observables and functions *text* names."""
        names = [*self.model.observable_names, *self.model.function_names]
        return [n for n in names if re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text)]

    def _freeze(self, text: str) -> str:
        """*text* with each observable and function it reads replaced by its value
        now. BNG2.pl substitutes them when the action runs, from the amounts of
        the moment (BNGModel.pm, setConcentration; Observable.pm, evaluate), and
        keeps only the parameters as names. An amount that is still an
        expression is a string there, which Perl reads as 0, so it counts as 0
        here too."""
        found = self._state_reads(text)
        if not found:
            return text
        state = self._state()
        for i in self.symbolic:
            state[i] = 0.0  # as BNG2.pl's numification reads it; see the docstring
        observables, functions = self._reads_now(state)
        values = dict(zip(self.model.observable_names, observables, strict=True))
        values.update(functions)
        for name in sorted(found, key=len, reverse=True):
            pattern = rf"(?<![\w.]){re.escape(name)}(?:\(\s*\))?(?!\w)"
            text = re.sub(pattern, f"({float(values[name])!r})", text)
        return text

    def _species(self, change: StateChange) -> int:
        from bngsim._bngl_loader import _species_name

        name = _species_name(self.model, change.target or "")
        if name is None:
            raise ConversionError(
                f"{_action(change)}: the network has no species {change.target!r}; "
                "BNG2.pl refuses it too"
            )
        return self.index[name]

    def _symbolic_value(self, entry: tuple) -> float:
        from bngsim._bngl_loader import _value

        if entry[0] == "param":
            return float(self.model.get_param(entry[1])) / entry[2]
        v = _value(self.model, entry[1])
        if v is None:
            raise ConversionError(f"{entry[1]!r} does not evaluate in the model")
        return v

    def _resolve_symbolic(self) -> None:
        """Write each expression-valued amount as its value now, as BNG2.pl writes
        the network a run reads."""
        if not self.symbolic:
            return
        state = self._state()
        for i, entry in self.symbolic.items():
            state[i] = self._symbolic_value(entry)
        self.model.set_state(state)

    def _set_params(self, values: dict[str, float]) -> None:
        """Set parameters and nothing else. bngsim's ``set_param`` moves a species
        whose amount names the parameter while it is still at its initial
        amount; BNG2.pl moves it only while its amount is still that expression,
        which :attr:`symbolic` tracks and each run reads."""
        missing = [n for n in values if n not in self.model.param_names]
        if missing:
            raise ConversionError(
                f"setParameter: the model has no parameter {missing[0]!r}; BNG2.pl refuses it too"
            )
        state = self._state()
        self.model.set_params(values)
        self.model.set_state(state)

    def _param_state(self) -> tuple[dict[str, float], set[str], dict[str, str]]:
        """The parameters a ``saveParameters`` keeps: constants and constant
        expressions (BNG2.pl's ``copyConstant``), not the slots bngsim keeps for
        a function's or an observable's value. An expression an action set
        is kept as the expression, as BNG2.pl's copy keeps it (issue #1682)."""
        names = list(self.model.param_names)
        derived = {n for n, e in zip(names, self.model.param_is_expression, strict=True) if e}
        computed = set(self.model.function_names) | set(self.model.observable_names)
        kept = [n for n in names if n not in computed]
        values = {n: float(self.model.get_param(n)) for n in kept}
        return values, derived, dict(self.param_exprs)

    def _param_reads(self, text: str) -> list[str]:
        """The parameters *text* names, the slots of functions and observables
        aside (a value that reads those is refused, :meth:`_change`)."""
        computed = set(self.model.function_names) | set(self.model.observable_names)
        names = [n for n in self.model.param_names if n not in computed]
        return [n for n in names if re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text)]

    def _defines_recursively(self, name: str, text: str) -> bool:
        """Whether setting *name* to *text* defines it through itself, directly
        or through the expressions it reads, which BNG2.pl refuses ("Parameter
        k2 is defined recursively"): those an action set, and those the
        network declares, ``k3 2*k1`` (BNG2.pl: "Max recursion depth")."""
        names = list(self.model.param_names)
        declared = {
            n: e
            for n, e, live in zip(
                names,
                self.model.param_expressions,
                self.model.param_is_expression,
                strict=True,
            )
            if live and e
        }
        seen: set[str] = set()
        stack = self._param_reads(text)
        while stack:
            read = stack.pop()
            if read == name:
                return True
            if read in seen:
                continue
            seen.add(read)
            if read in self.param_exprs:
                stack.extend(self._param_reads(self.param_exprs[read]))
            elif read in declared:
                stack.extend(self._param_reads(declared[read]))
        return False

    def _follow(self) -> None:
        """Bring each parameter an action set to an expression up to what it
        reads now, as BNG2.pl evaluates it where it is used (issue #1682). An
        expression may read another, so this repeats until none moves; none
        reads itself, which :meth:`_change` refuses."""
        from bngsim._bngl_loader import _value

        for _ in range(len(self.param_exprs) + 2):
            moved = False
            for name, text in self.param_exprs.items():
                value = _value(self.model, text)
                if value is None:
                    raise ConversionError(
                        f'setParameter("{name}","{text}"): the value does not evaluate '
                        "in the model"
                    )
                if float(self.model.get_param(name)) != value:
                    self._set_params({name: value})
                    moved = True
            if not moved:
                return
        # A chain settles in one pass per expression; one that does not reads
        # itself, which the check at the setParameter is meant to have refused.
        raise ConversionError(
            f"the parameters {sorted(self.param_exprs)} set to expressions do not settle: "
            "they are defined through each other, which BNG2.pl refuses"
        )

    # ── steps ─────────────────────────────────────────────────────────────
    def run(self) -> list[ProtocolRun]:
        from bngsim.convert._protocol import Experiment, StateChange

        runs: list[ProtocolRun] = []
        for step in self.protocol.steps[self.protocol.generation_step or 0 :]:
            if isinstance(step, Experiment):
                runs.append(self._experiment(step))
            elif isinstance(step, StateChange):
                self._change(step)
        return runs

    def _change(self, change: StateChange) -> None:
        kind = change.kind
        label = change.label or _DEFAULT_LABEL
        if kind == "set_parameter":
            name = change.target or ""
            if change.value_expr is not None and self._state_reads(change.value_expr):
                # BNG2.pl makes such a parameter a function of the state, read
                # anew through every later run (BNGModel.pm, setParameter
                # substitutes nothing); bngsim cannot turn a parameter into one.
                raise ConversionError(
                    f"{_action(change)}: the value reads "
                    f"{', '.join(self._state_reads(change.value_expr))}, which makes the "
                    "parameter a function of the state in BNG2.pl; bngsim cannot do that"
                )
            reads = (
                self._param_reads(change.value_expr)
                if change.value_expr is not None and not change.evaluated_once
                else []
            )
            if reads and self._defines_recursively(name, change.value_expr or ""):
                raise ConversionError(
                    f"{_action(change)}: the value reads {name} through itself; BNG2.pl "
                    f'refuses it too ("Parameter {name} is defined recursively")'
                )
            self._set_params({name: self._value(change)})
            # BNG2.pl keeps a value that reads other parameters as that
            # expression (BNGModel.pm, setParameter: the .net it writes lists
            # `k2 k1*3`), so the parameter follows them; a number ends that.
            self.param_exprs.pop(name, None)
            if reads:
                self.param_exprs[name] = change.value_expr or ""
            self._follow()
        elif kind == "set_concentration":
            i = self._species(change)
            if change.value is None and change.value_expr is not None:
                # BNG2.pl keeps an expression amount as one: it follows its
                # parameters until a run (BNGModel.pm, setConcentration's NewConc),
                # with its observables and functions read now.
                self.symbolic[i] = ("expr", self._freeze(change.value_expr))
                value = self._value(change)
            else:
                self.symbolic.pop(i, None)
                value = self._value(change)
            state = self._state()
            state[i] = value
            self.model.set_state(state)
        elif kind == "add_concentration":
            i = self._species(change)
            state = self._state()
            if self.symbolic.pop(i, None) is None:
                base = state[i]
            else:
                # Not a number yet: BNG2.pl adds to the species' own declared
                # amount (BNGModel.pm, addConcentration: evaluate
                # $spec->Concentration), not to what a setConcentration left.
                if self.declared is None:
                    raise ConversionError(
                        f"{_action(change)}: BNG2.pl adds to the species' declared amount, "
                        "which an action before generation changed, and the loaded network "
                        "holds only the changed one"
                    )
                declared = self.declared[1].get(i)
                base = (
                    self._symbolic_value(declared)
                    if declared is not None
                    else float(self.declared[0][i])
                )
            state[i] = base + self._value(change)
            self.model.set_state(state)
        elif kind == "save_concentrations":
            self.saved_conc[label] = (self._state(), dict(self.symbolic))
        elif kind == "reset_concentrations":
            self._reset_concentrations(label, change)
        elif kind == "save_parameters":
            self.saved_params[label] = self._param_state()
        elif kind == "reset_parameters":
            if label not in self.saved_params:
                raise ConversionError(
                    f"{_action(change)}: nothing saved under {label!r}; BNG2.pl stops here"
                )
            saved = self.saved_params[label]
            if saved is None:
                raise ConversionError(
                    f"{_action(change)}: the parameters saved under {label!r} before the "
                    "network was generated were changed before it too, so the loaded "
                    "model does not hold them"
                )
            # The saved values agree with the saved expressions: every change
            # brought those up to date before the save.
            self._restore_params(*saved)
        else:  # pragma: no cover — the parser makes no other kind
            raise ConversionError(f"unknown protocol step {kind!r}")

    def _reset_concentrations(self, label: str, change: StateChange | None = None) -> None:
        where = _action(change) if change is not None else "resetConcentrations"
        if label in self.saved_conc:
            snap = self.saved_conc[label]
            if snap is None:
                raise ConversionError(
                    f"{where}: the concentrations saved under {label!r} before the network "
                    "was generated were changed before it too, so the loaded model does "
                    "not hold them"
                )
        elif label == _DEFAULT_LABEL:
            snap = self.declared
            if snap is None:
                raise ConversionError(
                    f"{where}: with nothing saved, BNG2.pl goes back to the seed species "
                    "as declared, but an action before generation changed them, and the "
                    "loaded network holds only the changed amounts"
                )
        else:
            raise ConversionError(f"{where}: nothing saved under {label!r}; BNG2.pl stops here")
        state, symbolic = snap
        self.model.set_state(state.copy())
        self.symbolic = dict(symbolic)

    def _restore_params(
        self, values: dict[str, float], derived: set[str], exprs: dict[str, str]
    ) -> None:
        self.param_exprs = dict(exprs)
        names = list(self.model.param_names)
        now = {n: float(self.model.get_param(n)) for n in names}
        primary = {n: v for n, v in values.items() if n not in derived and now.get(n) != v}
        if primary:
            self._set_params(primary)
        # A derived parameter a setParameter overrode since the save: writing back
        # the value its expression gives re-attaches it (issue #188).
        expr_now = dict(zip(names, self.model.param_is_expression, strict=True))
        for n in derived:
            if n in values and not expr_now.get(n, True):
                self._set_params({n: values[n]})

    # ── runs ──────────────────────────────────────────────────────────────
    def _experiment(self, exp: Experiment) -> ProtocolRun:
        if exp.kind == "simulate":
            return ProtocolRun(experiment=exp, result=self._run(exp))
        values = _scan_values(exp)
        results = self._scan(exp, values, reset=exp.reset_between)
        if exp.kind == "bifurcate":
            back = _scan_values(exp, backward=True)
            back_results = self._scan(exp, back, reset=False)
            return ProtocolRun(
                experiment=exp,
                scan_values=tuple(values),
                scan_results=results,
                backward_values=tuple(back),
                backward_results=back_results,
            )
        return ProtocolRun(experiment=exp, scan_values=tuple(values), scan_results=results)

    def _scan(self, exp: Experiment, values: list[float], *, reset: bool) -> tuple[Result, ...]:
        """BNG2.pl's ``parameter_scan``: save as "SCAN", then per value set the
        parameter, reset to "SCAN" if resetting, and run; reset once more at the
        end if resetting. The parameter keeps its last value."""
        parameter = exp.scan_parameter or ""
        self.saved_conc["SCAN"] = (self._state(), dict(self.symbolic))
        results: list[Result] = []
        for value in values:
            # BNG2.pl sets each point with setParameter: a number, which ends an
            # expression the parameter held, and which those that read it follow.
            self._set_params({parameter: float(value)})
            self.param_exprs.pop(parameter, None)
            self._follow()
            if reset:
                self._reset_concentrations("SCAN")
            result = self._run(exp)
            result.custom_attrs["scan_parameter"] = parameter
            result.custom_attrs["scan_value"] = float(value)
            results.append(result)
        if reset:
            self._reset_concentrations("SCAN")
        return tuple(results)

    def _number(self, exp: Experiment, key: str) -> float | None:
        """A hash value as a number: as written, or evaluated in the live model."""
        from bngsim._bngl_loader import _value

        raw = exp.extra.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool):
            return float(raw)
        if isinstance(raw, (int, float)):
            return float(raw)
        v = _value(self.model, str(raw))
        if v is None:
            raise ConversionError(f"{_describe(exp)}: {key}=>{raw} does not evaluate in the model")
        return v

    def _own_seed(self, exp: Experiment) -> int | None:
        """The experiment's ``seed``, if it names one. An int is taken as it
        is: through :meth:`_number`'s float, a seed past 2**53 ran rounded
        (issue #1517)."""
        raw = exp.extra.get("seed")
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
        value = self._number(exp, "seed")
        return None if value is None else int(value)

    def _simulator(self, exp: Experiment):
        from bngsim._simulator import Simulator

        method = _method_of(exp)
        poplevel = None
        if method == "psa":
            poplevel = self._number(exp, "poplevel")
            if poplevel is None:
                warnings.warn(
                    f"{_describe(exp)} names no poplevel; BNG2.pl's default, "
                    f"{_BNG_POPLEVEL:g}, is used",
                    ConversionWarning,
                    stacklevel=5,
                )
                poplevel = _BNG_POPLEVEL
        key = (method, poplevel)
        if key not in self._sims:
            self._sims[key] = Simulator(self.model, method=method, poplevel=poplevel)
        return method, self._sims[key]

    def _run(self, exp: Experiment) -> Result:
        """One BNG2.pl ``simulate``: from the state the last run left, from the
        model time for ``continue=>1``, else from ``t_start``."""
        from bngsim.convert._protocol import _as_bool

        for key in exp.extra:
            low = str(key).lower()
            if low not in _RUN_KEYS and low not in _OUTPUT_KEYS and low not in _REFUSED_KEYS:
                warnings.warn(
                    f"{_describe(exp)}: {key} is not an argument this replay reads; it is ignored",
                    ConversionWarning,
                    stacklevel=5,
                )
        method, sim = self._simulator(exp)
        continued = _as_bool(exp.extra.get("continue", 0))
        named_start = self._number(exp, "t_start") if continued else None
        if named_start is not None and not math.isclose(
            named_start, self.now, rel_tol=1e-9, abs_tol=1e-12
        ):
            # The parse checks this where it knows the model time; after a run
            # that can stop early only the replay does (BNGAction.pm, simulate).
            raise ConversionError(
                f"{_describe(exp)}: continue=>1 with t_start={named_start:g}, but the model "
                f"time is {self.now:g}; BNG2.pl refuses a continuation that does not start there"
            )
        t0 = self.now if continued else float(exp.t_span[0])
        if exp.output_start is not None:
            # A SED-ML run whose output starts after it does (issue #1493): the
            # part before outputStartTime is integrated and not output.
            self._run_to(exp, method, sim, t0, float(exp.output_start))
            t0 = float(exp.output_start)
        if exp.sample_times is not None:
            times = [t0, *(t for t in exp.sample_times[1:] if t > t0)]
            t_end = times[-1]
        else:
            t_end = float(exp.t_span[1])
            times = None
        if t_end <= t0:
            warnings.warn(
                f"{_describe(exp)}: t_end ({t_end:g}) is not past t_start ({t0:g}); "
                "BNG2.pl warns too, and writes the one row at t_start",
                ConversionWarning,
                stacklevel=5,
            )
            return self._one_row(t0)
        kwargs: dict[str, Any] = {}
        steady = method == "ode" and _as_bool(exp.extra.get("steady_state", 0))
        if method == "ode":
            kwargs.update(self._tolerances(exp))
            if steady:
                kwargs["steady_state"] = True
        else:
            seed = self._own_seed(exp)
            kwargs["seed"] = (
                seed
                if seed is not None
                else int(self._seeds.spawn(1)[0].generate_state(1)[0] >> 1)
            )
        self._resolve_symbolic()
        if times is not None:
            result = sim.run(sample_times=times, **kwargs)
        else:
            result = sim.run(t_span=(t0, t_end), n_points=int(exp.n_points), **kwargs)
        if steady and not result.solver_stats.get("steady_state_reached"):
            warnings.warn(
                f"{_describe(exp)} did not reach a steady state by t_end={t_end:g}; "
                "BNG2.pl stops the protocol there",
                ConversionWarning,
                stacklevel=5,
            )
        # A run reads every amount as a number, and leaves them so (the run
        # writes its end state back to the model).
        self.symbolic.clear()
        self.now = float(result.time[-1])
        return result

    def _tolerances(self, exp: Experiment) -> dict[str, Any]:
        """An ODE run's rtol and atol: the replay's, else the action's, else
        run_network's, with the atol exact as run_network holds it."""
        from bngsim._atol import ExactAtol

        rtol = self._rtol if self._rtol is not None else self._number(exp, "rtol")
        atol = self._atol if self._atol is not None else self._number(exp, "atol")
        return {
            "rtol": _BNG_RTOL if rtol is None else rtol,
            "atol": ExactAtol(_BNG_ATOL if atol is None else atol),
        }

    def _run_to(self, exp: Experiment, method: str, sim: Any, t0: float, t1: float) -> None:
        """Integrate from *t0* to *t1* and output nothing: the part of a SED-ML
        run before its outputStartTime (issue #1493). The run that outputs then
        starts from the state this leaves, at *t1*."""
        if t1 <= t0:
            return
        kwargs: dict[str, Any] = {}
        if method == "ode":
            kwargs.update(self._tolerances(exp))
        else:
            # The run that outputs takes the action's seed. This leg takes the
            # second seed derived from it, as EvaluationSpec.evaluate does, so
            # a seeded run is one run on every replay; on a seed of its own,
            # two runs reported one seed (issue #1517). Unseeded, its own.
            seed = self._own_seed(exp)
            kwargs["seed"] = (
                derived_seeds(seed, 2)[1]
                if seed is not None
                else int(self._seeds.spawn(1)[0].generate_state(1)[0] >> 1)
            )
        self._resolve_symbolic()
        sim.run(t_span=(t0, t1), n_points=2, **kwargs)
        self.symbolic.clear()
        self.now = t1

    def _one_row(self, t0: float) -> Result:
        """A run that runs no time, as BNG2.pl's does: one row, the state at *t0*,
        whose amounts it then reads back as numbers."""
        from bngsim._result import Result

        self.now = t0
        self._resolve_symbolic()
        self.symbolic.clear()
        state = self._state()
        observables, functions = self._reads_now(state)
        names = [n for n in functions if not n.startswith("_rateLaw")]
        return Result(
            _time=np.array([t0]),
            _species=state[None, :],
            _observables=np.array([observables]),
            _expressions=np.array([[functions[n] for n in names]]) if names else None,
            _species_names=list(self.model.species_names),
            _observable_names=list(self.model.observable_names),
            _expression_names=names,
        )


def _action(change: StateChange | None) -> str:
    if change is None:
        return "the protocol"
    from bngsim.convert._protocol import _state_change_action

    return _state_change_action(change)
