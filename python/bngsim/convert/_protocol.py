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
  ``parameter_scan``, ``setParameter``, ``setConcentration``,
  ``resetConcentrations``, ``saveConcentrations``) are recovered into the
  :class:`ProtocolSpec` ``steps``.
* **drop** — pure build/IO directives (``generate_network``, ``writeSBML``,
  ``writeBNGL``, ``setOutputDir``, ``readFile``, …) are tooling, not protocol:
  they materialize the network, write a file, name the model, or choose where
  output goes. None changes the numbers a consumer would compute, so they are
  recorded in :attr:`ProtocolSpec.dropped` and otherwise ignored — no warning,
  in either mode.
* **flag (lossy)** — actions BNGsim does not execute but whose omission *would
  change the results* (``setVolume``, ``substanceUnits``, a result-changing
  ``setOption`` such as ``NumberPerQuantityUnit``) or that swap the model
  entirely (``readModel``/``readNetwork``/``readSBML``). Dropping these silently
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
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

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
# sweep; the rest are single runs whose method the hash may override.
_SIMULATE_VERBS = {
    "simulate": ("simulate", None),
    "simulate_ode": ("simulate", "ode"),
    "simulate_ssa": ("simulate", "ssa"),
    "simulate_nf": ("simulate", "nf"),
    "parameter_scan": ("scan", None),
}
_STATE_VERBS = {
    "setparameter": "set_parameter",
    "setconcentration": "set_concentration",
    "resetconcentrations": "reset_concentrations",
    "saveconcentrations": "save_concentrations",
}
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
        "savemodel",
        "setmodelname",
        "setoutputdir",
        "readfile",
        "version",
        "saveparameters",
        "resetparameters",
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

    # one of: set_parameter | set_concentration | reset_concentrations | save_concentrations
    kind: str
    target: str | None = None  # parameter/species name (None for reset/save)
    value: float | None = None  # numeric value, when the action carries one
    value_expr: str | None = None  # raw string value when it is not a plain number
    # BNG named saved-state label (issue #11): the positional argument of
    # saveConcentrations("name") / resetConcentrations("name"). None for the
    # default (unlabeled) slot. Only meaningful for the reset/save kinds.
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": "state_change",
            "kind": self.kind,
            "target": self.target,
            "value": self.value,
            "value_expr": self.value_expr,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StateChange:
        return cls(
            kind=d["kind"],
            target=d.get("target"),
            value=d.get("value"),
            value_expr=d.get("value_expr"),
            label=d.get("label"),
        )


@dataclass(frozen=True)
class Experiment:
    """A single simulation run (``simulate``) or a sweep (``parameter_scan``)."""

    kind: str  # "simulate" | "scan"
    method: str  # "ode" | "ssa" | "nf"
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

    @property
    def is_scan(self) -> bool:
        return self.kind == "scan"

    @property
    def is_deterministic(self) -> bool:
        return self.method == "ode"

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Experiment:
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
    """

    steps: tuple[object, ...] = ()  # tuple[Experiment | StateChange, ...]
    source: str | None = None
    dropped: tuple[str, ...] = ()
    lossy: tuple[str, ...] = ()
    network_gen: str | None = None

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
        """The :class:`StateChange`\\ s that precede ``experiment`` in order, since
        the last ``resetConcentrations`` — the accumulated operating point at
        which that experiment runs (what the L3 gate must apply to both sides)."""
        acc: list[StateChange] = []
        for s in self.steps:
            if s is experiment:
                break
            if isinstance(s, StateChange):
                if s.kind == "reset_concentrations":
                    acc = []
                elif s.kind in ("set_parameter", "set_concentration"):
                    acc.append(s)
        return acc

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
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ProtocolSpec:
        return cls.from_dict(json.loads(text))


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
    for spec in non_empty:
        if (
            reset_between
            and steps
            and not (
                isinstance(steps[-1], StateChange) and steps[-1].kind == "reset_concentrations"
            )
        ):
            steps.append(StateChange(kind="reset_concentrations"))
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
    )


# ─── Parser ─────────────────────────────────────────────────────────────────


def parse_bngl_protocol(source: str | Path, *, strict: bool = True) -> ProtocolSpec:
    """Parse a ``.bngl``'s action block into a :class:`ProtocolSpec`.

    Recovers the ordered simulation-experiment actions (``simulate*``,
    ``parameter_scan``, ``set*``, ``resetConcentrations``,
    ``saveConcentrations``). Other actions are dispositioned per the four-tier
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
    for verb, argstr in _iter_action_calls(text):
        low = verb.lower()
        if low == "quit":
            # quit() halts BNG2.pl action processing — anything after it never
            # runs. Truncate so the recovered protocol never replays trailing
            # actions the reference engine would have skipped. This is faithful
            # handling, not a loss: record it and stop.
            dropped.append(verb)
            break
        if low in _SIMULATE_VERBS:
            steps.append(_build_experiment(verb, argstr))
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

    return ProtocolSpec(
        steps=tuple(steps),
        source=src_label,
        dropped=tuple(dropped),
        lossy=tuple(lossy),
        network_gen=network_gen,
    )


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


def _flag_lossy(verb: str, reason: str, *, strict: bool, lossy: list[str]) -> None:
    """Disposition a fidelity-affecting action BNGsim does not execute: raise
    under ``strict``, else warn and record it in ``lossy``."""
    msg = (
        f"BNGL action {verb!r} affects results but BNGsim does not execute it: "
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


def _experiment_action(e: Experiment) -> str:
    """Serialize an :class:`Experiment` to a ``simulate``/``parameter_scan`` call.

    The hash keys mirror :func:`_build_experiment`'s reader so the round-trip
    ``parse_bngl_protocol(write_bngl_protocol(p))`` recovers an equal spec.
    """
    pairs: list[tuple[str, str]] = [("method", f'"{e.method}"')]
    pairs.append(("t_start", _action_num(e.t_span[0])))
    pairs.append(("t_end", _action_num(e.t_span[1])))
    pairs.append(("n_steps", _action_num(max(e.n_points - 1, 0))))
    if e.is_scan:
        if e.scan_parameter is not None:
            pairs.append(("parameter", f'"{e.scan_parameter}"'))
        if e.scan_min is not None:
            pairs.append(("par_min", _action_num(e.scan_min)))
        if e.scan_max is not None:
            pairs.append(("par_max", _action_num(e.scan_max)))
        if e.scan_points is not None:
            pairs.append(("n_scan_pts", _action_num(e.scan_points)))
        pairs.append(("log_scale", "1" if e.scan_log else "0"))
        pairs.append(("reset_conc", "1" if e.reset_between else "0"))
    if e.label is not None:
        pairs.append(("suffix", f'"{e.label}"'))
    for k, v in e.extra.items():
        pairs.append((str(k), _action_value(v)))

    verb = "parameter_scan" if e.is_scan else "simulate"
    body = ",".join(f"{k}=>{v}" for k, v in pairs)
    return f"{verb}({{{body}}})"


def _state_change_action(s: StateChange) -> str:
    """Serialize a :class:`StateChange` to its BNG ``set*`` call."""
    if s.kind == "reset_concentrations":
        if s.label is not None:
            return f'resetConcentrations("{s.label}")'
        return "resetConcentrations()"
    if s.kind == "save_concentrations":
        if s.label is not None:
            return f'saveConcentrations("{s.label}")'
        return "saveConcentrations()"
    verb = "setParameter" if s.kind == "set_parameter" else "setConcentration"
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
    spec (modulo the dropped build/IO directives, which a protocol never carries).
    No ``generate_network`` is emitted — that is a build directive the model-block
    writer prepends when it needs the network materialized (see
    :func:`bngsim.convert._bngl_writer.write_bngl`).

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
    lines = ["begin actions", ""]
    for step in protocol.steps:
        if isinstance(step, Experiment):
            lines.append(_experiment_action(step))
        elif isinstance(step, StateChange):
            lines.append(_state_change_action(step))
        else:  # pragma: no cover — ProtocolSpec only holds the two step types
            raise ConversionError(f"unserializable protocol step: {step!r}")
    lines.append("")
    lines.append("end actions")
    text = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


def _build_experiment(verb: str, argstr: str) -> Experiment:
    kind, forced_method = _SIMULATE_VERBS[verb.lower()]
    args = _parse_hash(argstr)
    method = forced_method or str(args.get("method", "ode")).lower()
    t_start = _num_or(args.get("t_start"), 0.0)
    # ``t_end`` may be absent or a parameter-referencing expression we can't
    # resolve without the model; leave it == t_start so the experiment carries
    # no numeric horizon and the gate falls back to its blanket grid.
    t_end = _num_or(args.get("t_end"), t_start)
    n_steps = int(_num_or(args.get("n_steps"), 100.0))
    n_points = max(n_steps + 1, 1)

    consumed = {
        "method",
        "t_start",
        "t_end",
        "n_steps",
        "parameter",
        "par_min",
        "par_max",
        "n_scan_pts",
        "log_scale",
        "reset_conc",
        "suffix",
    }
    extra = {k: v for k, v in args.items() if k not in consumed}

    if kind == "scan":
        return Experiment(
            kind="scan",
            method=method,
            t_span=(t_start, t_end),
            n_points=n_points,
            scan_parameter=str(args["parameter"]) if "parameter" in args else None,
            scan_min=_num_or(args.get("par_min"), None),
            scan_max=_num_or(args.get("par_max"), None),
            scan_points=(
                int(_num_or(args.get("n_scan_pts"), 0)) or None if "n_scan_pts" in args else None
            ),
            scan_log=_as_bool(args.get("log_scale", 0)),
            reset_between=_as_bool(args.get("reset_conc", 1)),
            label=str(args["suffix"]) if "suffix" in args else None,
            extra=extra,
        )
    return Experiment(
        kind="simulate",
        method=method,
        t_span=(t_start, t_end),
        n_points=n_points,
        label=str(args["suffix"]) if "suffix" in args else None,
        extra=extra,
    )


def _build_state_change(kind: str, argstr: str) -> StateChange:
    if kind in ("reset_concentrations", "save_concentrations"):
        # BNG saveConcentrations("name") / resetConcentrations("name") carry an
        # optional positional label naming the saved state (issue #11). Preserve
        # it so a multi-state protocol round-trips; the unlabeled form → None.
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
