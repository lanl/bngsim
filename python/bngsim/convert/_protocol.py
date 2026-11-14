"""bngsim.convert._protocol — the BNGL simulation-protocol IR + parser (GH #211).

A BioNetGen ``.net`` file carries **structure + math only**: the simulation
protocol (the ``simulate`` / ``parameter_scan`` / ``set*`` *actions*) lived in
the ``.bngl`` the network was generated from and was discarded at
network-generation time. To hand a consumer a *faithful* deliverable — one that
reproduces the modeller's actual experiment, not a synthesized default — the
converter must recover that protocol from the source ``.bngl``.

This module is the shippable bridge:

* :class:`ProtocolSpec` — an ordered, serializable sequence of protocol steps
  (each an :class:`Experiment` or a :class:`StateChange`). The single IR that the
  L3 gate (horizon + operating point) and the SED-ML/OMEX writers both consume.
* :func:`parse_bngl_protocol` — parse a ``.bngl``'s actions into a
  :class:`ProtocolSpec`, hand-rolled (no BNG2.pl, no ``parity_checks`` import).

**The action-disposition contract (GH #226).** Every BNGL action falls into one
of four tiers, decided by its verb:

* **execute** — the simulation-experiment actions (``simulate*``,
  ``parameter_scan``, ``bifurcate``, ``setParameter``, ``setConcentration``,
  ``addConcentration``, and the ``save``/``reset`` of concentrations and of
  parameters) are recovered into the :class:`ProtocolSpec` ``steps``, which
  :func:`bngsim.run_protocol` runs (issue #1366).
* **drop** — pure build/IO directives (``generate_network``, ``writeSBML``,
  ``writeBNGL``, ``setOutputDir``, ``readFile``, …) are tooling, not protocol:
  they materialize the network, write a file, name the model, or choose where
  output goes. None changes the numbers a consumer would compute, so they are
  recorded in :attr:`ProtocolSpec.dropped` and otherwise ignored — no warning,
  in either mode.
* **flag (lossy)** — actions BNGsim does not execute but whose omission *would
  change the results* (``setVolume``, ``substanceUnits``, a result-changing
  ``setOption`` such as ``NumberPerQuantityUnit``, ``simulate_protocol``) or that
  swap the model entirely (``readModel``/``readNetwork``/``readSBML``). Dropping these silently
  would be a faithfulness lie, so they are recorded in
  :attr:`ProtocolSpec.lossy`; ``strict`` **refuses** (raises
  :class:`ConversionError`) and best-effort warns.
* **unknown** — any unrecognized verb: ``strict`` refuses, best-effort warns and
  records it in ``dropped``.

``quit`` is control flow, not a no-op: in BNG2.pl it halts action processing, so
the parser **truncates** at ``quit`` (recording it in ``dropped``) rather than
replaying trailing actions the reference engine would have skipped.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, overload

from bngsim._exceptions import ConversionError, ConversionWarning

__all__ = [
    "ProtocolSpec",
    "Experiment",
    "StateChange",
    "parse_bngl_protocol",
    "write_bngl_protocol",
    "combine_protocols",
]


# ─── Action vocabulary ──────────────────────────────────────────────────────

# Simulation-experiment verbs → (kind, default method). ``parameter_scan`` is a
# sweep and ``bifurcate`` two (up, then back down, each point from the last);
# the rest are single runs whose method the hash may override.
_SIMULATE_VERBS = {
    "simulate": ("simulate", None),
    "simulate_ode": ("simulate", "ode"),
    "simulate_ssa": ("simulate", "ssa"),
    "simulate_nf": ("simulate", "nf"),
    "simulate_psa": ("simulate", "psa"),
    "simulate_pla": ("simulate", "pla"),
    "parameter_scan": ("scan", None),
    "bifurcate": ("bifurcate", None),
}
_STATE_VERBS = {
    "setparameter": "set_parameter",
    "setconcentration": "set_concentration",
    "addconcentration": "add_concentration",
    "resetconcentrations": "reset_concentrations",
    "saveconcentrations": "save_concentrations",
    "resetparameters": "reset_parameters",
    "saveparameters": "save_parameters",
}
# The kinds that carry a saved-state label rather than a target and value.
_LABEL_KINDS = frozenset(
    {"reset_concentrations", "save_concentrations", "reset_parameters", "save_parameters"}
)
# Where BNG2.pl's network exists: its first generation call or, before one, its
# first run, which generates the network itself (``BNGAction.pm``, ``simulate``).
_GENERATE_VERBS = frozenset({"generate_network", "generate_hybrid_model"})
# DROP — pure build/IO directives: they materialize the network, write a file,
# name the model, or choose where output goes. None alters the trajectory a
# consumer would compute, so each is recorded in ``dropped`` and otherwise
# ignored (no warning). ``writeBNGL`` / ``setOutputDir`` are tooling siblings of
# the already-recognized ``writeNET`` / ``writeSBML`` / ``readFile`` (GH #226).
_BUILD_VERBS = frozenset(
    {
        "generate_network",
        "generate_hybrid_model",
        "writesbml",
        "writemodel",
        "writexml",
        "writemexfile",
        "writemfile",
        "writematlab",
        "writenet",
        "writemdl",
        "writenetwork",
        "writebngl",
        # BNG2.pl's other output-only writers (Perl2/BNGModel.pm, issue #1366).
        "writecppfile",
        "writecpyfile",
        "writefile",
        "writefunctions",
        "writelatex",
        "writelatexfile",
        "writemfile_all",
        "writemfile_parametersobservables",
        "writemfile_querynames",
        "writesbmlmulti",
        "writesbmlparameters",
        "writesbmlreactions",
        "writessc",
        "writessccfg",
        "savemodel",
        "setmodelname",
        "setoutputdir",
        "readfile",
        "version",
        "visualize",
    }
)

# FLAG — recognized actions BNGsim does not execute but whose omission changes
# the results (or swaps the model), so a silent drop would be a faithfulness lie.
# Verb → plain-English reason. ``setOption`` is classified separately (by option
# name) because some of its options are cosmetic; ``quit`` is control flow,
# handled inline (truncation). Recorded in ``lossy``: ``strict`` refuses, lossy
# warns (GH #226).
_LOSSY_VERBS = {
    "setvolume": (
        "setVolume rescales a compartment volume, which changes every "
        "concentration and bimolecular rate constant in that compartment"
    ),
    "substanceunits": (
        "substanceUnits selects concentration-vs-number output semantics, "
        "which changes how the trajectory values are interpreted"
    ),
    "readmodel": (
        "readModel loads a different model mid-protocol, which a single "
        "ProtocolSpec cannot represent"
    ),
    "readnetwork": (
        "readNetwork loads a different network mid-protocol, which a single "
        "ProtocolSpec cannot represent"
    ),
    "readsbml": (
        "readSBML loads a different model mid-protocol, which a single "
        "ProtocolSpec cannot represent"
    ),
    "simulate_protocol": (
        "simulate_protocol runs the model's begin protocol block, which the "
        "protocol does not carry"
    ),
}

# ``setOption`` option names that are purely cosmetic (no effect on dynamics or
# on the user-named observables) and so DROP rather than FLAG. ``SpeciesLabel``
# only changes internal canonical species labels (cf. bng_parity
# ``_bng_common.py``); every other option — notably ``NumberPerQuantityUnit``,
# which sets the concentration→count conversion and so scales bimolecular rate
# constants — is treated as result-changing and FLAGged.
_COSMETIC_SETOPTIONS = frozenset({"specieslabel"})


# ─── IR ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StateChange:
    """A model-state action between experiments (the BNG ``set*`` family)."""

    # one of: set_parameter | set_concentration | add_concentration
    #       | save_concentrations | reset_concentrations
    #       | save_parameters | reset_parameters
    kind: str
    target: str | None = None  # parameter/species name (None for reset/save)
    value: float | None = None  # numeric value, when the action carries one
    value_expr: str | None = None  # raw string value when it is not a plain number
    # BNG named saved-state label (issue #11): the positional argument of
    # saveConcentrations("name") / resetConcentrations("name") and of their
    # saveParameters / resetParameters twins. None for the default (unlabeled)
    # slot, which BNG2.pl calls "DEFAULT". Only meaningful for the reset/save kinds.
    label: str | None = None
    # A setParameter's value_expr is kept as an expression, as BNG2.pl keeps
    # it, so the parameter follows the parameters it reads (issue #1682).
    # True where the value is evaluated once, where the step runs, and then
    # holds: a SED-ML computeChange or setValue, read from another tool's
    # document.
    evaluated_once: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = {
            "step": "state_change",
            "kind": self.kind,
            "target": self.target,
            "value": self.value,
            "value_expr": self.value_expr,
            "label": self.label,
        }
        # Only where it is set, so a protocol without one serializes as before.
        if self.evaluated_once:
            out["evaluated_once"] = True
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StateChange:
        return cls(
            kind=d["kind"],
            target=d.get("target"),
            value=d.get("value"),
            value_expr=d.get("value_expr"),
            label=d.get("label"),
            evaluated_once=bool(d.get("evaluated_once", False)),
        )


@dataclass(frozen=True)
class Experiment:
    """A single simulation run (``simulate``), a sweep (``parameter_scan``), or a
    bifurcation sweep (``bifurcate``: the sweep up, then back down, each point
    continuing from the one before)."""

    kind: str  # "simulate" | "scan" | "bifurcate"
    method: str  # "ode" | "ssa" | "psa" | "pla" | "nf"
    t_span: tuple[float, float]  # (t_start, t_end)
    n_points: int  # output points incl. t_start  (= n_steps + 1)
    # scan-only fields:
    scan_parameter: str | None = None
    scan_min: float | None = None
    scan_max: float | None = None
    scan_points: int | None = None
    scan_log: bool = False
    reset_between: bool = True  # BNG ``reset_conc`` for a scan
    label: str | None = None  # BNG ``suffix``, a per-experiment output tag
    extra: Mapping[str, Any] = field(default_factory=dict)  # other hash keys, for provenance
    # The scan's values, in the order BNG2.pl runs them, when ``par_scan_vals``
    # gave them; ``scan_min``/``scan_max``/``scan_points``/``scan_log`` then hold
    # the nearest range. None for a scan given as a range, which BNG2.pl expands
    # itself (issue #1366).
    scan_values: tuple[float, ...] | None = None
    # The output times BNG2.pl writes for ``sample_times`` (``t_start`` first),
    # when they are not the even grid ``(t_span, n_points)`` holds (issue #1366).
    sample_times: tuple[float, ...] | None = None
    # Why this run is not exactly the action as written, if it is not: each an
    # argument the protocol holds only approximately (issue #1364). A run that
    # starts at a model time an earlier run leaves unknown is not one of them:
    # ``extra["continue"]`` says where it starts, and a replay knows that time.
    approximations: tuple[str, ...] = ()
    # Where the output starts, when it is after the run's start: a SED-ML
    # uniformTimeCourse whose outputStartTime is past its initialTime. The run
    # integrates from ``t_span[0]`` and outputs ``n_points`` even times over
    # ``(output_start, t_span[1])``. None outputs from ``t_span[0]``, as every
    # BNG2.pl action does (issue #1493).
    output_start: float | None = None

    def __post_init__(self) -> None:
        if self.output_start is None:
            return
        start = float(self.output_start)
        t0, t_end = (float(t) for t in self.t_span)
        if not (math.isfinite(start) and t0 <= start <= t_end):
            raise ValueError(
                f"output_start={self.output_start!r} must lie in t_span ({t0:g}, {t_end:g}): "
                "the run integrates from t_span[0] and outputs from output_start to t_span[1]"
            )
        if self.sample_times is not None:
            raise ValueError(
                "an experiment takes output_start or sample_times, not both: "
                "sample_times already names every output time"
            )
        # An output that starts where the run does is the plain experiment, and
        # compares equal to it.
        object.__setattr__(self, "output_start", None if start == t0 else start)

    @property
    def is_scan(self) -> bool:
        return self.kind in ("scan", "bifurcate")

    @property
    def is_bifurcation(self) -> bool:
        return self.kind == "bifurcate"

    @property
    def is_deterministic(self) -> bool:
        return self.method == "ode"

    def to_dict(self) -> dict[str, Any]:
        out = {
            "step": "experiment",
            "kind": self.kind,
            "method": self.method,
            "t_span": list(self.t_span),
            "n_points": self.n_points,
            "scan_parameter": self.scan_parameter,
            "scan_min": self.scan_min,
            "scan_max": self.scan_max,
            "scan_points": self.scan_points,
            "scan_log": self.scan_log,
            "reset_between": self.reset_between,
            "label": self.label,
            "extra": dict(self.extra),
            "scan_values": None if self.scan_values is None else list(self.scan_values),
            "sample_times": None if self.sample_times is None else list(self.sample_times),
            "approximations": list(self.approximations),
        }
        # Only where it is set, so a protocol without one serializes as before.
        if self.output_start is not None:
            out["output_start"] = self.output_start
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Experiment:
        scan_values = d.get("scan_values")
        sample_times = d.get("sample_times")
        return cls(
            kind=d["kind"],
            method=d["method"],
            t_span=tuple(d["t_span"]),  # type: ignore[arg-type]
            n_points=int(d["n_points"]),
            scan_parameter=d.get("scan_parameter"),
            scan_min=d.get("scan_min"),
            scan_max=d.get("scan_max"),
            scan_points=d.get("scan_points"),
            scan_log=bool(d.get("scan_log", False)),
            reset_between=bool(d.get("reset_between", True)),
            label=d.get("label"),
            extra=dict(d.get("extra", {})),
            scan_values=None if scan_values is None else tuple(float(v) for v in scan_values),
            sample_times=None if sample_times is None else tuple(float(t) for t in sample_times),
            approximations=tuple(d.get("approximations", ())),
            output_start=None if d.get("output_start") is None else float(d["output_start"]),
        )


@dataclass(frozen=True)
class ProtocolSpec:
    """An ordered, serializable BNGL simulation protocol.

    ``steps`` is the verbatim action sequence (each an :class:`Experiment` or a
    :class:`StateChange`), preserving order so a downstream consumer can replay
    the modeller's exact experiment — including sequential continuation (a
    ``simulate`` after a ``simulate`` without an intervening reset continues from
    the prior end-state) and multi-stage scans. ``dropped`` records the build/IO
    directives that were intentionally not carried (pure tooling); ``lossy``
    records recognized actions that *would* change the results but BNGsim does not
    execute (``setVolume``, ``substanceUnits``, a result-changing ``setOption``,
    ``readModel``/…), so a consumer can tell a clean drop from a fidelity risk
    (GH #226).

    ``network_gen`` keeps the source model's ``generate_network(...)`` call
    **verbatim** (the last one, if several) when the ``.bngl`` carried one. That
    directive is still *dropped* from ``steps`` (it is a build directive, not a
    simulation action), but its text is retained here because it may carry a
    **finiteness cap** — ``max_stoich`` / ``max_agg`` / ``max_iter`` — without
    which a rule-based network generates unbounded. A materializer (e.g.
    :func:`~bngsim.convert._bngl_writer.write_bngl`) that re-emits an actions block
    honors this instead of prepending a bare ``generate_network({overwrite=>1})``,
    so a model that is finite only under its cap stays finite (lanl/PyBNF #485).
    ``None`` when the source had no ``generate_network`` (e.g. a protocol composed
    from SBML/SED-ML events), in which case the bare default is used.

    ``generation_step`` is where in ``steps`` the source's network was generated:
    at its first ``generate_network`` or, before one, at its first run, where
    BNG2.pl generates it. ``steps[:generation_step]`` came before, so a network
    :meth:`~bngsim.Model.from_bngl` loads already holds what they set, and a
    replay starts at ``steps[generation_step]`` (issue #1366). An actions block
    with no model to generate from (a block for a network loaded on its own,
    as :func:`~bngsim.convert.omex_to_net` writes beside its ``.net``) has its
    network from the start, or from a ``readFile`` of a ``.net``, so its runs
    generate nothing (issue #1687). ``None`` for a protocol with no source file
    to have generated anything from (SED-ML, SBML events), whose steps all run
    on the model they are given.
    ``lossy_before_generation`` and ``dropped_before_generation`` count the
    leading ``lossy`` and ``dropped`` entries that came before that point: the
    network holds what those did, as it holds what the steps before it set.
    """

    steps: tuple[object, ...] = ()  # tuple[Experiment | StateChange, ...]
    source: str | None = None
    dropped: tuple[str, ...] = ()
    lossy: tuple[str, ...] = ()
    network_gen: str | None = None
    generation_step: int | None = None
    lossy_before_generation: int = 0
    dropped_before_generation: int = 0

    @property
    def experiments(self) -> list[Experiment]:
        return [s for s in self.steps if isinstance(s, Experiment)]

    @property
    def is_empty(self) -> bool:
        return not self.experiments

    def primary_experiment(self) -> Experiment | None:
        """The experiment whose horizon best represents the protocol for the L3
        gate: the first **deterministic** (ODE) run/scan with a numeric ``t_end``,
        else the first experiment of any method, else ``None``."""
        exps = self.experiments
        for e in exps:
            if e.is_deterministic and e.t_span[1] > e.t_span[0]:
                return e
        return exps[0] if exps else None

    def state_changes_before(self, experiment: Experiment) -> list[StateChange]:
        """The :class:`StateChange`\\ s that precede ``experiment`` in order: the
        ``setConcentration``\\ s since the last ``resetConcentrations`` and the
        ``setParameter``\\ s since the last ``resetParameters``, as absolute
        overrides of the operating point that experiment runs at. A reset of one
        kind leaves the other's changes standing, as BNG2.pl does; an
        ``addConcentration`` is relative and is not one of them. Of two sets of
        one target, the later is the one that holds."""
        conc: dict[str | None, tuple[int, StateChange]] = {}
        params: dict[str | None, tuple[int, StateChange]] = {}
        for i, s in enumerate(self.steps):
            if s is experiment:
                break
            if isinstance(s, StateChange):
                if s.kind == "reset_concentrations":
                    conc = {}
                elif s.kind == "reset_parameters":
                    params = {}
                elif s.kind == "set_concentration":
                    conc.pop(s.target, None)
                    conc[s.target] = (i, s)
                elif s.kind == "set_parameter":
                    params.pop(s.target, None)
                    params[s.target] = (i, s)
        held = sorted([*conc.values(), *params.values()], key=lambda pair: pair[0])
        return [s for _, s in held]

    # ── serialization ──────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        out = []
        for s in self.steps:
            out.append(s.to_dict())  # type: ignore[attr-defined]
        return {
            "source": self.source,
            "dropped": list(self.dropped),
            "lossy": list(self.lossy),
            "network_gen": self.network_gen,
            "generation_step": self.generation_step,
            "lossy_before_generation": self.lossy_before_generation,
            "dropped_before_generation": self.dropped_before_generation,
            "steps": out,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ProtocolSpec:
        steps: list[object] = []
        for sd in d.get("steps", []):
            if sd.get("step") == "experiment":
                steps.append(Experiment.from_dict(sd))
            else:
                steps.append(StateChange.from_dict(sd))
        return cls(
            steps=tuple(steps),
            source=d.get("source"),
            dropped=tuple(d.get("dropped", [])),
            lossy=tuple(d.get("lossy", [])),
            network_gen=d.get("network_gen"),
            generation_step=d.get("generation_step"),
            lossy_before_generation=int(d.get("lossy_before_generation", 0)),
            dropped_before_generation=int(d.get("dropped_before_generation", 0)),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ProtocolSpec:
        return cls.from_dict(json.loads(text))


class _SavedBeforeGeneration(NamedTuple):
    """What a protocol's actions before its generation point left saved, as
    the actions that take the network they generated back to each save.

    ``conc`` and ``params`` map each label to those actions, in order (empty
    where the network holds what was saved), or to ``None`` where what was
    saved is not known from the protocol: a value as declared in the model,
    which a later action before generation changed. ``declared`` says whether
    the network holds the seed species as declared, where a
    ``resetConcentrations()`` with nothing saved goes (issue #1668)."""

    conc: dict[str, tuple[StateChange, ...] | None]
    params: dict[str, tuple[StateChange, ...] | None]
    declared: bool


def _saved_before_generation(steps: Sequence[object]) -> _SavedBeforeGeneration:
    """:class:`_SavedBeforeGeneration` for *steps*, the steps before a
    protocol's generation point (issue #1668).

    Each target's value is followed as the actions that make it: a species'
    last ``setConcentration`` and the ``addConcentration`` calls after it (or
    those alone, on its declared amount), a parameter's last
    ``setParameter``. A save keeps that record, a reset goes back to it, and
    the network holds the record at generation. A saved value the network
    does not hold is put back by its own actions where it has them; one that
    is the value as declared is not known, since the network holds only the
    changed one."""
    default = "DEFAULT"  # BNG2.pl's label for an unlabeled save or reset
    conc: dict[str, tuple[StateChange, ...]] | None = {}
    params: dict[str, StateChange] | None = {}
    conc_saves: dict[str, dict[str, tuple[StateChange, ...]] | None] = {}
    param_saves: dict[str, dict[str, StateChange] | None] = {}
    for step in steps:
        if not isinstance(step, StateChange):
            continue
        label = step.label or default
        target = step.target or ""
        if step.kind == "set_concentration":
            if conc is not None:
                conc[target] = (step,)
        elif step.kind == "add_concentration":
            if conc is not None:
                conc[target] = conc.get(target, ()) + (step,)
        elif step.kind == "save_concentrations":
            conc_saves[label] = None if conc is None else dict(conc)
        elif step.kind == "reset_concentrations":
            if label in conc_saves:
                saved = conc_saves[label]
                conc = None if saved is None else dict(saved)
            else:
                # Nothing saved: BNG2.pl goes back to the seed as declared, or
                # stops on a label, which leaves nothing to follow.
                conc = {} if label == default else None
        elif step.kind == "set_parameter":
            if params is not None:
                params[target] = step
        elif step.kind == "save_parameters":
            param_saves[label] = None if params is None else dict(params)
        elif step.kind == "reset_parameters":
            saved_p = param_saves.get(label)
            params = None if saved_p is None else dict(saved_p)

    def back_conc(saved: dict | None) -> tuple[StateChange, ...] | None:
        if saved is None or conc is None:
            return None
        out: list[StateChange] = []
        for key in dict.fromkeys([*saved, *conc]):
            was, now = saved.get(key, ()), conc.get(key, ())
            if was == now:
                continue
            if not was or was[0].kind != "set_concentration":
                return None  # the amount as declared, which the network does not hold
            out.extend(was)
        return tuple(out)

    def back_params(saved: dict | None) -> tuple[StateChange, ...] | None:
        if saved is None or params is None:
            return None
        out: list[StateChange] = []
        for key in dict.fromkeys([*saved, *params]):
            was, now = saved.get(key), params.get(key)
            if was == now:
                continue
            if was is None:
                return None  # the value as declared, which the network does not hold
            out.append(was)
        return tuple(out)

    return _SavedBeforeGeneration(
        conc={label: back_conc(saved) for label, saved in conc_saves.items()},
        params={label: back_params(saved) for label, saved in param_saves.items()},
        declared=conc == {},
    )


#: The label :func:`_on_its_network` saves the network's state under, to come
#: back to it after making each state the protocol saved before generation.
_NETWORK_LABEL = "bngsim_network"


def _on_its_network(protocol: ProtocolSpec) -> tuple[ProtocolSpec, list[str]]:
    """*protocol* as an actions block for the network it generated, loaded on
    its own: what :func:`~bngsim.convert.omex_to_net` writes beside its
    ``.net`` (issue #1687).

    That network holds what the actions before the generation point did, so
    they are left out, and BNG2.pl, which reads the network whole, applies
    every action of the block: written again, an ``addConcentration`` before
    generation was added twice. A save among them is made again on the
    network, with the actions that made what it saved (issue #1668). One whose
    values were as declared, which the network does not hold, cannot be: each
    reset the block makes to such a state is returned as a note."""
    from bngsim._protocol_run import _DEFAULT_LABEL

    generation = protocol.generation_step or 0
    if not generation:
        return protocol, []
    before = _saved_before_generation(protocol.steps[:generation])
    after = protocol.steps[generation:]
    prelude: list[object] = []
    unknown: list[tuple[str, str]] = []
    for kind, saves in (("concentrations", before.conc), ("parameters", before.params)):
        save, reset = (
            ("save_concentrations", "reset_concentrations")
            if kind == "concentrations"
            else ("save_parameters", "reset_parameters")
        )
        for label, back in saves.items():
            if back is None:
                unknown.append((reset, label))
                continue
            name = None if label == _DEFAULT_LABEL else label
            if not back:
                prelude.append(StateChange(kind=save, label=name))
                continue
            prelude.append(StateChange(kind=save, label=_NETWORK_LABEL))
            prelude.extend(back)
            prelude.append(StateChange(kind=save, label=name))
            prelude.append(StateChange(kind=reset, label=_NETWORK_LABEL))
    if not before.declared and _DEFAULT_LABEL not in before.conc:
        # An unsaved resetConcentrations() goes to the seed as declared.
        unknown.append(("reset_concentrations", _DEFAULT_LABEL))
    notes: list[str] = []
    for reset, label in unknown:
        if _goes_back_to(after, reset, label):
            what = "concentrations" if reset == "reset_concentrations" else "parameters"
            saved = label in (before.conc if what == "concentrations" else before.params)
            state = f"saved under {label!r}" if saved else "as declared"
            notes.append(
                f"the actions block goes back to the {what} {state} before the network was "
                "generated, whose values are the model's as declared; the network holds what "
                "an action before generation changed them to, so the block cannot make that "
                "state, and BNG2.pl, run on the network, goes back to another"
            )
    spec = ProtocolSpec(
        steps=tuple(prelude) + tuple(after),
        source=protocol.source,
        dropped=protocol.dropped[protocol.dropped_before_generation :],
        lossy=protocol.lossy[protocol.lossy_before_generation :],
        network_gen=protocol.network_gen,
        generation_step=0,
    )
    return spec, notes


def _goes_back_to(steps: Sequence[object], reset: str, label: str) -> bool:
    """Whether *steps* reset to *label* before saving it again."""
    from bngsim._protocol_run import _DEFAULT_LABEL

    save = "save_concentrations" if reset == "reset_concentrations" else "save_parameters"
    for step in steps:
        if not isinstance(step, StateChange) or (step.label or _DEFAULT_LABEL) != label:
            continue
        if step.kind == save:
            return False
        if step.kind == reset:
            return True
    return False


#: The label :func:`combine_protocols` saves the first protocol's state under at
#: its generation point, and resets to at each boundary between the protocols it joins.
_COMBINE_LABEL = "combined_start"


def combine_protocols(
    specs: list[ProtocolSpec] | tuple[ProtocolSpec, ...],
    *,
    reset_between: bool = True,
) -> ProtocolSpec:
    """Concatenate several :class:`ProtocolSpec`\\ s into one, preserving order.

    Used when more than one independent protocol contributes to a single
    deliverable — e.g. an OMEX archive carrying multiple SED-ML *files*, each its
    own experiment set (GH #222). Each spec's steps are appended in order; empty
    specs are skipped. Because two independent SED-ML files are *not* continuations
    of one another, a :class:`StateChange` ``resetConcentrations`` boundary is
    inserted between consecutive non-empty specs when ``reset_between`` is True
    (and the running sequence does not already end in a reset), so each
    contributed protocol starts from the model's fresh initial state. A single
    spec is returned unchanged, so the exact round-trip of the common one-file case
    is preserved (no spurious reset is introduced).

    ``source`` labels are joined with ``" + "`` and the ``dropped`` / ``lossy``
    directives are concatenated, so provenance survives the merge.

    Parameters
    ----------
    specs : sequence of ProtocolSpec
        The protocols to merge, in the order they should run.
    reset_between : bool
        Insert a ``resetConcentrations`` between independent specs (default True).

    Returns
    -------
    ProtocolSpec
    """
    non_empty = [s for s in specs if s is not None and not s.is_empty]
    if not non_empty:
        # Nothing carried an experiment; return the first spec verbatim (empty) so
        # the caller still sees its dropped/source provenance, else a blank spec.
        for s in specs:
            if s is not None:
                return s
        return ProtocolSpec()
    if len(non_empty) == 1:
        return non_empty[0]

    steps: list[object] = []
    sources: list[str] = []
    dropped: list[str] = []
    lossy: list[str] = []
    network_gen: str | None = None
    # Only the first spec's steps can come before a network is generated: the
    # rest run on the network it leaves, after the reset that separates them.
    generation_step = non_empty[0].generation_step
    # The parameters as the first spec found them at its generation point,
    # after its pre-generation actions (which the network holds, and a replay
    # skips), for every later spec to start from: one spec's setParameter
    # stood through the next (issue #1662), where its concentrations alone
    # went back. Saved before those actions, the split moved by one and an
    # addConcentration the network held was replayed on it.
    for i, spec in enumerate(non_empty):
        if i == 0:
            # The concentrations too: a reset to nothing saved goes to the seed
            # species as declared, which a network generated after a
            # pre-generation action does not hold.
            split = generation_step or 0
            steps.extend(spec.steps[:split])
            if reset_between:
                steps.append(StateChange(kind="save_concentrations", label=_COMBINE_LABEL))
                steps.append(StateChange(kind="save_parameters", label=_COMBINE_LABEL))
            steps.extend(spec.steps[split:])
        else:
            if reset_between:
                # A protocol that already ends by going back to the declared
                # state needs no second reset of the concentrations.
                last = steps[-1]
                if not (
                    isinstance(last, StateChange)
                    and last.kind == "reset_concentrations"
                    and last.label is None
                ):
                    steps.append(StateChange(kind="reset_concentrations", label=_COMBINE_LABEL))
                steps.append(StateChange(kind="reset_parameters", label=_COMBINE_LABEL))
            steps.extend(spec.steps)
        if spec.source:
            sources.append(spec.source)
        dropped.extend(spec.dropped)
        lossy.extend(spec.lossy)
        # Carry a cap through the merge: the first contributing spec that names a
        # generate_network wins (independent SED-ML files rarely carry one at all).
        if network_gen is None and spec.network_gen is not None:
            network_gen = spec.network_gen

    return ProtocolSpec(
        steps=tuple(steps),
        source=" + ".join(sources) if sources else None,
        dropped=tuple(dropped),
        lossy=tuple(lossy),
        network_gen=network_gen,
        generation_step=generation_step,
        # The first spec's entries lead the merged lists.
        lossy_before_generation=non_empty[0].lossy_before_generation,
        dropped_before_generation=non_empty[0].dropped_before_generation,
    )


# ─── Parser ─────────────────────────────────────────────────────────────────


def parse_bngl_protocol(source: str | Path, *, strict: bool = True) -> ProtocolSpec:
    """Parse a ``.bngl``'s action block into a :class:`ProtocolSpec`.

    Recovers the ordered simulation-experiment actions (``simulate*``,
    ``parameter_scan``, ``bifurcate``, ``setParameter``, ``setConcentration``,
    ``addConcentration``, and the ``save``/``reset`` of concentrations and of
    parameters). Other actions are dispositioned per the four-tier
    contract documented at the top of this module: pure build/IO directives
    (``generate_network``, ``writeSBML``, ``writeBNGL``, ``setOutputDir``, …) are
    dropped and recorded in ``ProtocolSpec.dropped``; fidelity-affecting actions
    BNGsim does not execute (``setVolume``, ``substanceUnits``, a result-changing
    ``setOption``, ``readModel``/``readNetwork``/``readSBML``) are recorded in
    ``ProtocolSpec.lossy``; ``quit`` truncates the action stream.

    Parameters
    ----------
    source : str | Path
        A path to a ``.bngl`` file, or the BNGL text itself.
    strict : bool
        Raise :class:`bngsim.ConversionError` on a fidelity-affecting action
        (the ``lossy`` tier) or an unrecognized action verb. ``False`` downgrades
        either to a :class:`bngsim.ConversionWarning` (recording the action in
        ``lossy`` / ``dropped`` respectively). Pure build/IO directives drop
        cleanly in both modes; either way, malformed action arguments raise.

    Returns
    -------
    ProtocolSpec
    """
    text = _read_text(source)
    src_label = str(source) if _looks_like_path(source) else None

    steps: list[object] = []
    dropped: list[str] = []
    lossy: list[str] = []
    network_gen: str | None = None
    generation_step: int | None = None
    lossy_before: int | None = None  # entries before the generation point
    dropped_before: int | None = None
    # BNG2.pl's model time, which a ``continue=>1`` or nf run starts from: 0 at
    # load, and the end of each simulate after it (``BNGAction.pm``:
    # ``$model->Time``), or NaN after an nf run, which leaves it undefined. None
    # once a run's end cannot be known without running it (issue #1364).
    model_time: float | None = 0.0
    # Whether a run generates the network: only where there is a model to
    # generate it from. An actions block with none runs on a network loaded on
    # its own, which a run reads as it stands (issue #1687): the block that
    # omex_to_net writes beside its .net set A and then ran from the .net's A.
    generates = _declares_a_model(text)
    for verb, argstr in _iter_action_calls(text):
        low = verb.lower()
        if low == "quit":
            # quit() halts BNG2.pl action processing — anything after it never
            # runs. Truncate so the recovered protocol never replays trailing
            # actions the reference engine would have skipped. This is faithful
            # handling, not a loss: record it and stop.
            dropped.append(verb)
            break
        if low == "readfile":
            read = str(_parse_hash(argstr).get("file", "")).strip().lower()
            if read.endswith(".net"):
                # A network read whole: it is there from this action on.
                if generation_step is None:
                    generation_step = len(steps)
                    lossy_before, dropped_before = len(lossy), len(dropped)
            else:
                generates = True  # a model, whose first run generates its network
        if generation_step is None and (
            low in _GENERATE_VERBS
            or (generates and (low in _SIMULATE_VERBS or low == "simulate_protocol"))
        ):
            generation_step = len(steps)
            lossy_before, dropped_before = len(lossy), len(dropped)
        if low in _SIMULATE_VERBS:
            exp, reasons, model_time = _build_experiment(verb, argstr, model_time)
            steps.append(exp)
            if reasons:
                _flag_lossy(
                    verb,
                    "; ".join(reasons),
                    strict=strict,
                    lossy=lossy,
                    head="has an argument the protocol cannot hold exactly",
                )
        elif low in _STATE_VERBS:
            steps.append(_build_state_change(_STATE_VERBS[low], argstr))
        elif low == "setoption":
            reason = _setoption_lossy_reason(argstr)
            if reason is None:  # cosmetic option (e.g. SpeciesLabel) → clean drop
                dropped.append(verb)
            else:
                _flag_lossy(verb, reason, strict=strict, lossy=lossy)
        elif low in _LOSSY_VERBS:
            _flag_lossy(verb, _LOSSY_VERBS[low], strict=strict, lossy=lossy)
        elif low in _BUILD_VERBS:
            dropped.append(verb)
            if low == "generate_network":
                # Drop it from the replayable steps (it is a build directive, not a
                # simulation action) but keep the call verbatim: it may carry a finiteness
                # cap (max_stoich / max_agg / max_iter) a re-emitted actions block must honor
                # rather than override with a bare default (lanl/PyBNF #485). Last one wins,
                # mirroring BNG2.pl (a later generate_network supersedes an earlier one).
                network_gen = f"{verb}({argstr})"
        else:
            msg = (
                f"unrecognized BNGL action {verb!r}: the protocol channel carries "
                "simulate/parameter_scan/set* actions; build/IO directives are "
                "dropped. Pass strict=False / --allow-lossy to drop unknown actions."
            )
            if strict:
                raise ConversionError(msg)
            warnings.warn(msg, ConversionWarning, stacklevel=2)
            dropped.append(verb)

    if generation_step is None and not generates:
        # No model and no network read: every step runs on the network given.
        generation_step, lossy_before, dropped_before = 0, 0, 0
    return ProtocolSpec(
        steps=tuple(steps),
        source=src_label,
        dropped=tuple(dropped),
        lossy=tuple(lossy),
        network_gen=network_gen,
        generation_step=len(steps) if generation_step is None else generation_step,
        lossy_before_generation=len(lossy) if lossy_before is None else lossy_before,
        dropped_before_generation=len(dropped) if dropped_before is None else dropped_before,
    )


# The blocks of a BNGL model: a file with one of them has a model, which its
# first run generates the network from. BNG2.pl also accepts each spelled with
# underscores. A ``.net``'s blocks (``reactions``, ``groups``) are a network
# already, and BNG2.pl generates nothing from one.
_MODEL_BLOCKS = frozenset(
    {
        "model",
        "parameters",
        "compartments",
        "molecule types",
        "seed species",
        "species",
        "observables",
        "functions",
        "reaction rules",
        "energy patterns",
        "population types",
        "population maps",
    }
)


def _declares_a_model(text: str) -> bool:
    """Whether a BNGL text holds a model, which its first run generates the
    network from, as opposed to an actions block alone, for a network loaded
    on its own, or a network itself (issue #1687)."""
    blocks = set()
    for line in _logical_lines(text):
        m = re.match(r"begin\s+(.+?)\s*$", line.strip(), re.I)
        if m is not None:
            blocks.add(" ".join(m.group(1).lower().replace("_", " ").split()))
    return "reactions" not in blocks and bool(blocks & _MODEL_BLOCKS)


def _setoption_lossy_reason(argstr: str) -> str | None:
    """Classify a ``setOption`` call by its option name: ``None`` if it is
    cosmetic (a clean drop), else a plain-English reason it is result-changing
    (and so belongs on the lossy channel). The first positional argument is the
    option name."""
    pos = _parse_positional(argstr)
    name = str(pos[0]).strip() if pos else ""
    if name.lower() in _COSMETIC_SETOPTIONS:
        return None
    if name.lower() == "numberperquantityunit":
        return (
            'setOption("NumberPerQuantityUnit", …) sets the concentration→count '
            "conversion, scaling every bimolecular rate constant; BNGsim does not "
            "apply it"
        )
    return (
        f"setOption({name!r}, …) may change network generation or unit "
        "interpretation; BNGsim does not apply it"
    )


def _flag_lossy(
    verb: str, reason: str, *, strict: bool, lossy: list[str], head: str | None = None
) -> None:
    """Disposition a fidelity-affecting action BNGsim does not execute: raise
    under ``strict``, else warn and record it in ``lossy``. ``head`` replaces the
    message's opening, for an action it does execute but whose arguments the
    protocol holds only approximately (issue #1364)."""
    opening = head or "affects results but BNGsim does not execute it"
    msg = (
        f"BNGL action {verb!r} {opening}: "
        f"{reason}. Pass strict=False / --allow-lossy to record it as a lossy "
        "(best-effort) conversion instead of refusing."
    )
    if strict:
        raise ConversionError(msg)
    warnings.warn(msg, ConversionWarning, stacklevel=3)
    lossy.append(verb)


# ─── Writer (ProtocolSpec → BNGL actions) ────────────────────────────────────


def _action_num(v: float | int) -> str:
    """Render a numeric action argument: an integral value as a bare int, else
    a ``float()``-round-tripping repr (BNG re-parses every value)."""
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def _action_value(v: Any) -> str:
    """Render a hash/positional value: numbers bare, strings quoted."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return _action_num(v)
    return f'"{v}"'


def _output_grid(e: Experiment) -> list[float]:
    """The even output times of an experiment with an ``output_start``."""
    import numpy as np

    assert e.output_start is not None
    return [float(t) for t in np.linspace(e.output_start, e.t_span[1], int(e.n_points))]


def _experiment_action(e: Experiment) -> str:
    """Serialize an :class:`Experiment` to a ``simulate``/``parameter_scan`` call.

    The hash keys mirror :func:`_build_experiment`'s reader so the round-trip
    ``parse_bngl_protocol(write_bngl_protocol(p))`` recovers an equal spec.
    """
    pairs: list[tuple[str, str]] = [("method", f'"{e.method}"')]
    pairs.append(("t_start", _action_num(e.t_span[0])))
    if e.output_start is not None:
        # BNG2.pl has no output start (issue #1493): the run integrates from
        # t_start and writes t_start's row, then these times. t_start is among
        # them so the list has the 3 entries BNG2.pl needs at n_points = 2.
        times = ",".join(_action_num(t) for t in (e.t_span[0], *_output_grid(e)))
        pairs.append(("sample_times", f"[{times}]"))
    elif e.sample_times is not None:
        # The written times: BNG2.pl outputs t_start first and ends at the last.
        times = ",".join(_action_num(t) for t in e.sample_times[1:])
        pairs.append(("sample_times", f"[{times}]"))
    else:
        pairs.append(("t_end", _action_num(e.t_span[1])))
        pairs.append(("n_steps", _action_num(max(e.n_points - 1, 0))))
    if e.is_scan:
        if e.scan_parameter is not None:
            pairs.append(("parameter", f'"{e.scan_parameter}"'))
        if e.scan_values is not None:
            vals = ",".join(_action_num(v) for v in e.scan_values)
            pairs.append(("par_scan_vals", f"[{vals}]"))
        else:
            if e.scan_min is not None:
                pairs.append(("par_min", _action_num(e.scan_min)))
            if e.scan_max is not None:
                pairs.append(("par_max", _action_num(e.scan_max)))
            if e.scan_points is not None:
                pairs.append(("n_scan_pts", _action_num(e.scan_points)))
            pairs.append(("log_scale", "1" if e.scan_log else "0"))
        if not e.is_bifurcation:
            pairs.append(("reset_conc", "1" if e.reset_between else "0"))
    if e.label is not None:
        pairs.append(("suffix", f'"{e.label}"'))
    written = {k for k, _ in pairs}
    for k, v in e.extra.items():
        if str(k) not in written:
            pairs.append((str(k), _action_value(v)))

    body = ",".join(f"{k}=>{v}" for k, v in pairs)
    return f"{_experiment_verb(e)}({{{body}}})"


def _experiment_verb(e: Experiment) -> str:
    return "bifurcate" if e.is_bifurcation else "parameter_scan" if e.is_scan else "simulate"


# StateChange kind → its BNG action verb, the inverse of ``_STATE_VERBS``.
_STATE_ACTIONS = {
    "set_parameter": "setParameter",
    "set_concentration": "setConcentration",
    "add_concentration": "addConcentration",
    "reset_concentrations": "resetConcentrations",
    "save_concentrations": "saveConcentrations",
    "reset_parameters": "resetParameters",
    "save_parameters": "saveParameters",
}


def _state_change_action(s: StateChange) -> str:
    """Serialize a :class:`StateChange` to its BNG ``set*`` call."""
    verb = _STATE_ACTIONS[s.kind]
    if s.kind in _LABEL_KINDS:
        return f'{verb}("{s.label}")' if s.label is not None else f"{verb}()"
    val = (
        _action_num(s.value)
        if s.value is not None
        else (f'"{s.value_expr}"' if s.value_expr is not None else "0")
    )
    return f'{verb}("{s.target}",{val})'


def write_bngl_protocol(protocol: ProtocolSpec, out_path: str | Path | None = None) -> str:
    """Serialize a :class:`ProtocolSpec` to a BNGL ``begin actions`` block.

    The mirror of :func:`parse_bngl_protocol`: each :class:`Experiment` becomes a
    ``simulate``/``parameter_scan`` call and each :class:`StateChange` its
    ``set*``/``resetConcentrations``/``saveConcentrations`` call, in order. The
    round-trip ``parse_bngl_protocol(write_bngl_protocol(p))`` recovers an equal
    spec, its ``generation_step`` included, modulo the build/IO directives a
    protocol records but never runs (``dropped``, ``network_gen``). A protocol
    whose network was generated after its first step gets a
    ``generate_network`` there (the source's own call, ``network_gen``), so the
    block reads back with the steps before it in the network; any other gets
    none — that is a build directive the model-block writer prepends when it
    needs the network materialized (see :func:`bngsim.convert._bngl_writer.write_bngl`).

    Parameters
    ----------
    protocol : ProtocolSpec
        The protocol to serialize. ``StateChange.target`` is emitted verbatim, so
        a caller targeting BNGL species must pass a compartment-qualified pattern
        (e.g. ``@comp1:S1()``); the model-block writer does that translation.
    out_path : str | Path | None
        If given, the actions text is also written here.

    Returns
    -------
    str
        The ``begin actions`` … ``end actions`` block (trailing newline included).
    """
    from bngsim.convert._bngl_writer import _generate_network_line

    lines = ["begin actions", ""]
    for k, step in enumerate(protocol.steps):
        if k == protocol.generation_step and k:
            # Where the source generated its network, so the block reads back
            # with the actions before it in the network (issue #1687).
            lines.append(_generate_network_line(protocol.network_gen))
        if isinstance(step, StateChange) and step.evaluated_once and step.value_expr:
            moved = _moved_later(protocol.steps[k + 1 :], step.value_expr)
            if moved:
                warnings.warn(
                    f"{_state_change_action(step)} is evaluated once where it runs (a SED-ML "
                    f"change); BNG2.pl keeps the value as that expression, so the parameter "
                    f"follows {', '.join(moved)} through the later actions that change "
                    f"{'it' if len(moved) == 1 else 'them'}, and runs another value there "
                    "(issue #1682)",
                    ConversionWarning,
                    stacklevel=2,
                )
        if isinstance(step, Experiment):
            if step.output_start is not None:
                warnings.warn(
                    f"{_experiment_verb(step)} integrates from "
                    f"t={step.t_span[0]:g} and outputs from t={step.output_start:g} "
                    "(a SED-ML outputStartTime past its initialTime); BNG2.pl has no "
                    "output start, so the action is written as sample_times from "
                    f"t_start={step.t_span[0]:g}, and its output has that row before them",
                    ConversionWarning,
                    stacklevel=2,
                )
            lines.append(_experiment_action(step))
        elif isinstance(step, StateChange):
            lines.append(_state_change_action(step))
        else:  # pragma: no cover — ProtocolSpec only holds the two step types
            raise ConversionError(f"unserializable protocol step: {step!r}")
    if protocol.generation_step and protocol.generation_step >= len(protocol.steps):
        lines.append(_generate_network_line(protocol.network_gen))
    lines.append("")
    lines.append("end actions")
    text = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


def _moved_later(steps: Sequence[object], text: str) -> list[str]:
    """The names *text* reads that a step of *steps* sets: a ``setParameter``
    of it, a scan over it, or a ``resetParameters``."""
    names = set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)(?!\w)", text))
    moved: list[str] = []
    for step in steps:
        if isinstance(step, StateChange):
            if step.kind == "set_parameter" and step.target in names:
                moved.append(step.target or "")
            elif step.kind == "reset_parameters":
                moved.extend(sorted(names))
        elif isinstance(step, Experiment) and step.is_scan and step.scan_parameter in names:
            moved.append(step.scan_parameter or "")
    return list(dict.fromkeys(moved))


def _num_list(v: Any) -> list[float] | None:
    """A BNGL ``[a,b,…]`` of numbers, else ``None`` (not a list, or an entry
    that cannot be read without the model)."""
    if not isinstance(v, str):
        return None
    t = v.strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    out: list[float] = []
    for tok in _split_top_level(t[1:-1]):
        x = _num_or(_scalar(tok), None)
        if x is None:
            return None
        out.append(x)
    return out


def _evenly_spaced(values: list[float], *, log: bool = False) -> bool:
    """Whether *values* (at least two) step evenly, up or down, or evenly in log10."""
    if len(values) < 2:
        return False
    if log:
        if any(x <= 0.0 for x in values):
            return False
        values = [math.log10(x) for x in values]
    step = (values[-1] - values[0]) / (len(values) - 1)
    if step == 0.0:
        return False
    scale = max(abs(values[0]), abs(values[-1]), abs(step))
    return all(abs(x - (values[0] + i * step)) <= 1e-12 * scale for i, x in enumerate(values))


def _sample_grid(samples: list[float], t_start: float, t_end: float | None) -> list[float]:
    """The output times BNG2.pl writes for ``sample_times`` from *t_start*.

    ``BNGAction.pm`` sorts the times and appends a given ``t_end``, which ends the
    run even when samples lie past it; ``run_network`` outputs ``t_start`` first,
    then only the samples strictly between it and that end (issue #1364).
    """
    times = sorted(samples)
    if t_end is not None:
        times.append(t_end)
    end = max(times[-1], t_start)
    return [t_start, *(t for t in times[:-1] if t_start < t < end), end]


def _build_experiment(
    verb: str, argstr: str, model_time: float | None
) -> tuple[Experiment, list[str], float | None]:
    """The :class:`Experiment` a ``simulate``/``parameter_scan`` call runs, the
    reasons it is not that experiment exactly (together one ``lossy`` entry), and
    the model time after it.

    The time course follows BNG2.pl's ``simulate`` (``BNGAction.pm``, issue
    #1364): ``continue=>1`` starts at the model time, where an earlier run
    ended, and a ``t_start`` that is not it is refused; ``simulate_nf`` starts
    there, leaves it undefined, which a later start reads as 0, and refuses
    ``continue``, a missing ``n_steps`` and a ``t_end`` not past ``t_start``.
    ``n_output_steps`` is ``n_steps`` (but not for nf), both is an error, and
    neither is one step. ``sample_times`` sets the output times where neither
    is given: ``t_start``, the samples after it, and the end, a given ``t_end``
    or else the last sample; fewer than three is an error. A run that can stop
    early (``steady_state``, ``stop_if``, ``max_sim_steps``) leaves the model
    time unknown. A scan reads ``par_scan_vals``, in their order, unless
    ``par_min``, ``par_max`` and ``n_scan_pts`` are all given; ``bifurcate`` is
    a scan that never resets. ``poplevel`` makes any run but an nf one PSA.
    Uneven output times and scan values are held exactly (``sample_times``,
    ``scan_values``) beside the nearest uniform grid and range (issue #1366); an
    argument the parse cannot read without the model is recorded as lossy, and
    on the run's ``approximations``, with what it was taken as.

    The reasons are the lossy entry's: a run's ``approximations``, and first
    any start at a model time an earlier run leaves unknown, which only a
    replay can know.
    """
    kind, forced_method = _SIMULATE_VERBS[verb.lower()]
    args = _parse_hash(argstr)
    method = forced_method or str(args.get("method", "ode")).lower()
    if "poplevel" in args and method != "nf":
        # BNG2.pl's simulate runs PSA whenever a poplevel is given (BNGAction.pm).
        method = "psa"
    reasons: list[str] = []
    time_reasons: list[str] = []  # starts only a replay can place; see the docstring
    if method == "protocol":
        reasons.append(
            'method=>"protocol" runs the model\'s begin protocol block, which the '
            "protocol does not carry"
        )

    nf = method == "nf"
    continued = _as_bool(args.get("continue", 0))
    if nf and continued:
        raise ConversionError(f"{verb}: NFsim does not take continue=>1; BNG2.pl refuses it")
    # NaN is BNG2.pl's undefined model time, after an nf run: a start reads it as
    # 0, and a continuation that names its t_start is refused.
    undefined = model_time is not None and math.isnan(model_time)
    t_start = _num_or(args.get("t_start"), None)
    if "t_start" in args and t_start is None and not continued:
        reasons.append(f"t_start={args['t_start']} cannot be read without the model; carried as 0")
    if (
        continued
        and "t_start" in args
        and model_time is not None
        and (
            undefined
            or (
                t_start is not None
                and not math.isclose(t_start, model_time, rel_tol=1e-9, abs_tol=1e-12)
            )
        )
    ):
        now = "undefined after an nf run" if undefined else f"{model_time:g}"
        raise ConversionError(
            f"{verb}: continue=>1 with t_start={args['t_start']}, but the model time is "
            f"{now}; BNG2.pl refuses a continuation that does not start there"
        )
    from_model_time = t_start is None and (continued or (nf and "t_start" not in args))
    if from_model_time:
        # A continuation's unreadable t_start can only be the model time: BNG2.pl
        # refuses any other.
        if model_time is None:
            starter = "continue=>1" if continued else "an nf run"
            time_reasons.append(
                f"{starter} starts at the model time, which an earlier action leaves "
                "unknown; taken as 0"
            )
        t_start = 0.0 if model_time is None or undefined else model_time
    elif t_start is None:
        t_start = 0.0

    if nf:
        # simulate_nf reads n_steps alone (n_output_steps is not one of its
        # arguments), and refuses a run with no output to write.
        steps_arg = args.get("n_steps")
        n_read = _num_or(steps_arg, None)
        if (steps_arg is None and "sample_times" not in args) or (
            n_read is not None and n_read < 1
        ):
            raise ConversionError(
                f"{verb}: an nf run needs n_steps of 1 or more; BNG2.pl refuses one without"
            )
    else:
        if "n_steps" in args and "n_output_steps" in args:
            raise ConversionError(
                f"{verb}: 'n_steps' and 'n_output_steps' are the same argument; "
                "BNG2.pl refuses both"
            )
        steps_arg = args.get("n_steps", args.get("n_output_steps"))
    # ``t_end`` may be absent or a parameter-referencing expression we can't
    # resolve without the model; leave it == t_start so the experiment carries
    # no numeric horizon and the gate falls back to its blanket grid.
    t_end = _num_or(args.get("t_end"), None)
    t_end_unread = "t_end" in args and t_end is None
    if nf and steps_arg is not None and "t_end" not in args:
        raise ConversionError(f"{verb}: an nf run needs t_end; BNG2.pl refuses one without")
    if nf and t_end is not None and t_end <= t_start:
        raise ConversionError(
            f"{verb}: an nf run needs t_end > t_start, not ({t_start:g}, {t_end:g}); "
            "BNG2.pl refuses it"
        )
    # simulate_nf has no output_step_interval; run_network writes extra rows for it.
    if "output_step_interval" in args and not nf:
        reasons.append("output_step_interval writes rows between the grid's times")
    sample_times: tuple[float, ...] | None = None
    if steps_arg is None and "sample_times" in args:
        samples = _num_list(args.get("sample_times"))
        if samples is not None and len(samples) < 3:
            raise ConversionError(
                f"{verb}: sample_times {samples} has fewer than 3 times; BNG2.pl refuses it"
            )
        if samples is None:
            reasons.append("sample_times is not a list of numbers")
            n_points = 2
        else:
            grid = _sample_grid(samples, t_start, t_end)
            t_end, n_points = grid[-1], len(grid)
            carried = f"carried as {n_points} even points over ({t_start:g}, {t_end:g})"
            if nf:
                reasons.append(
                    "BNG2.pl 2.9.3 refuses sample_times for NFsim, and later versions "
                    f"output only at those times, from t_start; {carried}"
                )
            elif t_end_unread:
                reasons.append(
                    f"sample_times end at a t_end that cannot be read without the model; {carried}"
                )
            else:
                # Held exactly, beside the even grid over the same span: a replay
                # starting where an earlier run stopped early takes the samples
                # after that time (issue #1366).
                sample_times = tuple(grid)
    else:
        # BNG2.pl's default is one step. One the parse cannot read without the
        # model keeps the blanket 100 it has always had here.
        read_steps = None if steps_arg is None else _num_or(steps_arg, None)
        if steps_arg is not None and read_steps is None:
            reasons.append(f"n_steps={steps_arg} cannot be read without the model; carried as 100")
        n_steps = 1 if steps_arg is None else int(_num_or(steps_arg, 100.0))
        n_points = max(n_steps + 1, 1)
        if t_end_unread:
            reasons.append(
                f"t_end={args['t_end']} cannot be read without the model; carried as t_start"
            )
    if t_end is None:
        t_end = t_start
    # BNG2.pl's model time is the last time the run wrote: unknown for a t_end
    # the parse cannot read, or for a run that can stop early (at a steady state,
    # when its stop_if holds, or at max_sim_steps), and t_start for one that
    # does not run because its t_end is not past it. simulate_nf leaves it
    # undefined (its n_steps branch declares a second ``my $t_end``).
    stops_early = (
        _as_bool(args.get("steady_state", 0))
        or bool(args.get("stop_if"))
        or "max_sim_steps" in args
    )
    if nf:
        after: float | None = math.nan
    elif t_end_unread or stops_early:
        after = None
    else:
        after = max(t_start, t_end)

    consumed = {
        "method",
        "t_start",
        "t_end",
        "n_steps",
        "n_output_steps",
        "sample_times",
        "parameter",
        "par_min",
        "par_max",
        "n_scan_pts",
        "par_scan_vals",
        "log_scale",
        "reset_conc",
        "suffix",
    }
    # ``continue`` stays in ``extra``: re-emitted beside the t_start it implies,
    # which BNG2.pl accepts (it checks the two agree), it keeps a phased run
    # appending to one output file (convert/_events.py writes such phases). From
    # an undefined model time it starts at 0 like any run, and BNG2.pl would
    # refuse it beside that t_start, so it is not kept.
    if continued and undefined:
        consumed.add("continue")
    if continued and model_time is None and t_start is not None:
        # A continuation's own t_start, which the parse cannot check against a
        # model time an earlier run leaves unknown: kept for a replay to check.
        consumed.discard("t_start")
    extra = {k: v for k, v in args.items() if k not in consumed}

    if kind in ("scan", "bifurcate"):
        # bifurcate sets reset_conc=>0 over whatever the call says (BNGAction.pm).
        reset_between = kind == "scan" and _as_bool(args.get("reset_conc", 1))
        scan_log = _as_bool(args.get("log_scale", 0))
        scan_min = _num_or(args.get("par_min"), None)
        scan_max = _num_or(args.get("par_max"), None)
        scan_points = (
            int(_num_or(args.get("n_scan_pts"), 0)) or None if "n_scan_pts" in args else None
        )
        for key, read in (("par_min", scan_min), ("par_max", scan_max)):
            if key in args and read is None:
                reasons.append(f"{key}={args[key]} cannot be read without the model")
        if "n_scan_pts" in args and _num_or(args.get("n_scan_pts"), None) is None:
            reasons.append(f"n_scan_pts={args['n_scan_pts']} cannot be read without the model")
        scan_values: tuple[float, ...] | None = None
        # BNG2.pl takes par_min/par_max over par_scan_vals only with n_scan_pts too.
        ranged = all(k in args for k in ("par_min", "par_max", "n_scan_pts"))
        if not ranged and "par_scan_vals" in args:
            values = _num_list(args.get("par_scan_vals"))
            if not values:
                reasons.append("par_scan_vals is not a list of numbers")
            else:
                # The values exactly, in the order BNG2.pl runs them; the range
                # below is their nearest, for a consumer that reads only that.
                scan_values = tuple(values)
                # BNG2.pl runs the values in the order given. With reset_conc=>1 each
                # point starts afresh, so the order changes only the rows of the .scan
                # file; without it each continues from the last, so the order is kept,
                # a descending scan as par_min > par_max (which BNG2.pl runs as given).
                run = sorted(values) if reset_between else values
                if len(run) == 1 or _evenly_spaced(run):
                    scan_log = False
                elif _evenly_spaced(run, log=True):
                    scan_log = True
                else:
                    run = sorted(values)
                scan_min, scan_max, scan_points = run[0], run[-1], len(run)
        # Each point is a run from the model time, which the first point moves.
        later = 0.0 if nf else after
        if from_model_time and (scan_points or 0) > 1 and later != t_start:
            moved = "an unknown time" if later is None else f"{later:g}"
            time_reasons.append(
                f"its first point starts at the model time, {t_start:g}, and the others "
                f"at {moved}, where that point leaves it; carried as starting at {t_start:g}"
            )
        return (
            Experiment(
                kind=kind,
                method=method,
                t_span=(t_start, t_end),
                n_points=n_points,
                scan_parameter=str(args["parameter"]) if "parameter" in args else None,
                scan_min=scan_min,
                scan_max=scan_max,
                scan_points=scan_points,
                scan_log=scan_log,
                reset_between=reset_between,
                label=str(args["suffix"]) if "suffix" in args else None,
                extra=extra,
                scan_values=scan_values,
                sample_times=sample_times,
                approximations=tuple(reasons),
            ),
            time_reasons + reasons,
            after,
        )
    return (
        Experiment(
            kind="simulate",
            method=method,
            t_span=(t_start, t_end),
            n_points=n_points,
            label=str(args["suffix"]) if "suffix" in args else None,
            extra=extra,
            sample_times=sample_times,
            approximations=tuple(reasons),
        ),
        time_reasons + reasons,
        after,
    )


def _build_state_change(kind: str, argstr: str) -> StateChange:
    if kind in _LABEL_KINDS:
        # BNG saveConcentrations("name") / resetConcentrations("name"), and their
        # saveParameters / resetParameters twins, carry an optional positional label
        # naming the saved state (issue #11). Preserve it so a multi-state protocol
        # round-trips; the unlabeled form → None.
        pos = _parse_positional(argstr)
        label = str(pos[0]) if pos else None
        return StateChange(kind=kind, label=label)
    pos = _parse_positional(argstr)
    if not pos:
        raise ConversionError(f'{kind}: expected ("name", value), got {argstr!r}')
    target = str(pos[0])
    if len(pos) < 2:
        return StateChange(kind=kind, target=target)
    raw = pos[1]
    if isinstance(raw, (int, float)):
        return StateChange(kind=kind, target=target, value=float(raw))
    # A string value (a parameter reference or expression) — keep verbatim; the
    # numeric consumer (the L3 gate) skips it, the carrier (SED-ML) records it.
    try:
        return StateChange(kind=kind, target=target, value=float(raw))
    except (TypeError, ValueError):
        return StateChange(kind=kind, target=target, value_expr=str(raw))


# ─── Lexing helpers ─────────────────────────────────────────────────────────


def _iter_action_calls(text: str):
    """Yield ``(verb, argstr)`` for each action-scope ``verb(...)`` statement.

    Walks the file tracking ``begin <block> … end <block>`` nesting so only
    statements at action scope (outside ``begin model … end model``, optionally
    inside ``begin actions … end actions``) are returned. Handles ``\\`` line
    continuations, ``#`` comments, and an optional trailing ``;``.
    """
    block_stack: list[str] = []
    for raw_line in _logical_lines(text):
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("begin "):
            block_stack.append(low.split(None, 1)[1].strip())
            continue
        if low.startswith("end "):
            if block_stack:
                block_stack.pop()
            continue
        # Inside the model definition (parameters/species/reactions/…) → not an
        # action. Action scope is the empty stack or an explicit `actions` block.
        if "model" in block_stack or (block_stack and block_stack[-1] != "actions"):
            continue
        m = _ACTION_RE.match(line)
        if m:
            yield m.group(1), m.group(2)


def _logical_lines(text: str):
    """Yield comment-stripped logical lines, joining ``\\``-continuations."""
    buf = ""
    for physical in text.splitlines():
        # Strip a `#` comment (BNGL has no `#` string literals in actions).
        hash_idx = physical.find("#")
        if hash_idx != -1:
            physical = physical[:hash_idx]
        stripped = physical.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1]
            continue
        buf += stripped
        # An optional trailing `;` is a statement terminator, not part of args.
        out = buf.strip()
        if out.endswith(";"):
            out = out[:-1].rstrip()
        yield out
        buf = ""
    if buf.strip():
        yield buf.strip().rstrip(";").rstrip()


_ACTION_RE = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)


def _split_top_level(s: str) -> list[str]:
    """Split on top-level commas, respecting quotes and ``{}``/``()``/``[]`` depth."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    cur = ""
    for ch in s:
        if quote is not None:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            cur += ch
        elif ch in "{[(":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _parse_hash(argstr: str) -> dict[str, Any]:
    """Parse a BNGL ``{key=>value,…}`` (or empty) argument into a dict."""
    s = argstr.strip()
    if not s:
        return {}
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if not s:
        return {}
    out: dict[str, Any] = {}
    for item in _split_top_level(s):
        if "=>" not in item:
            continue
        k, _, v = item.partition("=>")
        out[k.strip().strip("'\"")] = _scalar(v.strip())
    return out


def _parse_positional(argstr: str) -> list[Any]:
    s = argstr.strip()
    if not s:
        return []
    return [_scalar(tok.strip()) for tok in _split_top_level(s)]


def _scalar(tok: str) -> Any:
    """Coerce a BNGL scalar token: quoted → str, numeric (incl. literal
    arithmetic like ``3600*5``) → float/int, else the raw string (a parameter
    reference or expression we cannot resolve without the model)."""
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] in ("'", '"') and tok[-1] == tok[0]:
        return tok[1:-1]
    try:
        f = float(tok)
        return int(f) if f.is_integer() and "." not in tok and "e" not in tok.lower() else f
    except ValueError:
        pass
    val = _eval_arith(tok)
    return val if val is not None else tok


def _eval_arith(expr: str) -> float | None:
    """Evaluate a *literal* arithmetic expression (``+ - * / ** ()``, numbers
    only — no names) safely via ``ast``. Returns ``None`` when it references a
    name or is otherwise not a pure numeric expression."""
    import ast

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )
    for n in ast.walk(node):
        if not isinstance(n, allowed):
            return None
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            return None
    try:
        return float(eval(compile(node, "<arith>", "eval"), {"__builtins__": {}}, {}))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


@overload
def _num_or(v: Any, default: float) -> float: ...
@overload
def _num_or(v: Any, default: None) -> float | None: ...
def _num_or(v: Any, default: float | None) -> float | None:
    """A numeric value (already-coerced number, or a string we can evaluate),
    else ``default`` — never raises, so an unresolved parameter reference in an
    action arg degrades to the default instead of crashing the parse."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    out = _eval_arith(str(v))
    return out if out is not None else default


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return bool(v)


def _looks_like_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    s = str(source)
    return "\n" not in s and ("begin " not in s)


def _read_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text()
    s = str(source)
    if "\n" in s or s.lstrip().startswith(("begin", "#")):
        return s
    return Path(s).read_text()
