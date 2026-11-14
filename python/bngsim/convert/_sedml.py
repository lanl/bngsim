"""bngsim.convert._sedml — the SED-ML sidecar protocol channel (GH #218).

SBML core and BioNetGen ``.net`` both carry **structure + math only** — neither
encodes a *simulation protocol* (start/end time, number of points, which
outputs to record, solver and tolerances). SED-ML is the COMBINE-standard
sidecar that supplies it. The converter therefore has two independent channels:
the **network channel** (``sbml_to_net`` / ``net_to_sbml``, GH #215/#216) and
this **protocol channel**.

The bridge type on the bngsim side is :class:`bngsim.EvaluationSpec` — a
serializable record of ``(model source, time grid, outputs, solver options)``
whose :meth:`~bngsim.EvaluationSpec.evaluate` *is* a runnable job. This module
maps a SED-ML **uniform time course** to/from an ``EvaluationSpec``:

* :func:`read_sedml` — parse a SED-ML document → ``EvaluationSpec``.
* :func:`write_sedml` — emit a SED-ML document from an ``EvaluationSpec``.
* :func:`default_protocol` — a sensible default spec for a model (all
  observables as outputs), for the *no-sidecar* case and ``net2sbml --sidecar``.

Scope is deliberately the **uniform-time-course subset** the acceptance bar
calls for; the reader/writer are hand-rolled over the stdlib XML tools (no
``python-libsedml`` runtime dependency). The bngsim output **selectors**
(``observable:``/``expression:``/``species:``) are carried verbatim in each
data-generator's ``name`` so a bngsim→SED-ML→bngsim round-trip is exact, while a
standards-compliant ``variable`` (time ``symbol`` / species ``target``) is also
emitted for interop with other SED-ML tools.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from bngsim._atol import ExactAtol
from bngsim._eval_spec import EvaluationSpec
from bngsim._exceptions import ConversionError, ConversionWarning

if TYPE_CHECKING:
    from bngsim._model import Model

__all__ = [
    "read_sedml",
    "write_sedml",
    "default_protocol",
    "write_sedml_protocol",
    "read_sedml_protocol",
]


# ─── Namespaces & KiSAO vocabulary ──────────────────────────────────────────

_SEDML_NS = "http://sed-ml.org/sed-ml/level1/version3"
_MATHML_NS = "http://www.w3.org/1998/Math/MathML"
_SBML_LANG = "urn:sedml:language:sbml"
_TIME_SYMBOL = "urn:sedml:symbol:time"
# bngsim provenance annotation namespace (verbatim ProtocolSpec; synthesized-
# default marker). Read back by read_sedml/read_sedml_protocol.
_BNGSIM_ANNOT_NS = "urn:bngsim:protocol"

# Method ↔ KiSAO algorithm term. The reverse map also accepts a few common
# synonyms other tools emit for the same integrator family.
_METHOD_TO_KISAO = {
    "ode": "KISAO:0000019",  # CVODE
    "ssa": "KISAO:0000029",  # Gillespie direct method
}
_KISAO_NFSIM = "KISAO:0000263"  # NFSim agent-based simulation method
_KISAO_PLA = "KISAO:0000524"  # partitioned leaping method
# The protocol vocabulary (BNG2.pl's methods). read_sedml, whose spec runs a
# network, reads NFsim as the exact SSA and refuses PLA (issue #1500).
_KISAO_TO_METHOD = {
    "KISAO:0000019": "ode",  # CVODE
    "KISAO:0000496": "ode",  # CVODES
    "KISAO:0000088": "ode",  # LSODA
    "KISAO:0000071": "ode",  # generic deterministic / LSODA family
    "KISAO:0000029": "ssa",  # Gillespie direct
    "KISAO:0000027": "ssa",  # next-reaction (Gibson–Bruck)
    "KISAO:0000241": "ssa",  # Gillespie-like
    "KISAO:0000694": "ode",  # ODE solver (the generic term BioModels' SED-ML names)
    _KISAO_NFSIM: "nf",
    _KISAO_PLA: "pla",
}

# Solver options ↔ KiSAO algorithm-parameter terms.
_KISAO_RTOL = "KISAO:0000209"  # relative tolerance
_KISAO_ATOL = "KISAO:0000211"  # absolute tolerance
_KISAO_MAXSTEPS = "KISAO:0000415"  # maximum number of steps
_KISAO_SEED = "KISAO:0000488"  # random seed (issue #1517)


def _q(uri: str, local: str) -> str:
    """Clark-notation ``{uri}local`` element tag.

    ElementTree serializes this exactly as it does an ``ET.QName`` object — it
    converts a QName tag to this same text before writing — but as a plain
    ``str`` it type-checks against Element/SubElement's ``tag`` parameter.
    """
    return f"{{{uri}}}{local}"


# ─── Writing: EvaluationSpec → SED-ML ───────────────────────────────────────


def _selector_target(selector: str, targets: _SbmlTargets) -> dict[str, str]:
    """Standards-compliant ``variable`` attributes for a bngsim output selector.

    The verbatim selector rides in ``name`` (the round-trip source of truth);
    ``target`` is the SBML element it reads, by the id the SBML gives it: a
    species, or for an observable or a function the parameter bngsim's SBML
    writer makes of it (an assignment rule). A species' BNGL name is not its
    id (``A()`` is ``A``), so without the SBML none is written.
    """
    kind, _, name = selector.partition(":")
    if not name:
        return {}
    if kind == "species":
        sid = targets.species_id(name) if targets.read else None
        return {"target": targets.species_path(sid)} if sid else {}
    if kind in ("observable", "expression", "function"):
        pid = targets.param_id(name)
        return {"target": targets.param_path(pid)} if pid else {}
    return {}


_SYNTHESIZED_DEFAULT_NOTE = (
    "bngsim-generated DEFAULT protocol: the source model carried no simulation "
    "protocol (no .bngl, or a .bngl with no simulate action), so this uniform "
    "time course (t=0..100, ODE, all observables) was synthesized as a runnable "
    "placeholder. It is NOT the modeller's protocol."
)


def write_sedml(
    spec: EvaluationSpec,
    out_path: str | Path | None = None,
    *,
    model_source: str | None = None,
    synthesized_default: bool = False,
    sbml: str | Path | None = None,
    model: Model | None = None,
) -> str:
    """Serialize an :class:`~bngsim.EvaluationSpec` to a SED-ML (L1V3) document.

    Parameters
    ----------
    spec : EvaluationSpec
        The protocol to emit. Its ``method`` must map to a KiSAO algorithm term
        (``ode`` → CVODE, ``ssa`` → Gillespie); anything else raises
        :class:`bngsim.ConversionError`.
    out_path : str | Path | None
        If given, the document is also written here.
    model_source : str | None
        Overrides the ``<model source="…">`` reference (e.g. point the sidecar at
        the converted ``.net``/SBML rather than the spec's original source).
        Defaults to ``spec.model_source``.
    synthesized_default : bool
        Mark this document as a bngsim-generated *default* (used when the source
        carried no real protocol): adds a ``<bngsim:synthesizedDefault>``
        annotation and a distinguishing report name so a consumer can never
        mistake the placeholder for the modeller's protocol. The orchestrators
        also emit a :class:`bngsim.ConversionWarning` when they synthesize one.
    sbml : str | Path | None
        The SBML the document points at, as text or a path, which each output's
        ``target`` is written against: the namespace its ``sbml`` prefix binds,
        and the id the SBML writer gave each species, observable and function.
        Defaults to the model source, when that is a readable file. An output
        the SBML does not name is written without a target, with a warning.
    model : Model | None
        The model the SBML was written from, to resolve a species name as
        BNG2.pl does.

    Returns
    -------
    str
        The SED-ML document text.
    """
    if spec.method not in _METHOD_TO_KISAO:
        raise ConversionError(
            f"cannot emit SED-ML for method {spec.method!r}: no KiSAO algorithm "
            f"term mapped (supported: {sorted(_METHOD_TO_KISAO)}). The SED-ML "
            "sidecar channel covers the uniform-time-course ODE/SSA protocol."
        )
    # Issue #196 — a per-species absolute tolerance has no SED-ML spelling.
    # KISAO:0000211 is one number for the whole solve, so emitting anything here
    # would silently downgrade the protocol to a scalar and hand the reader a
    # document that does not reproduce the run it claims to describe. Same for
    # "auto", which resolves to a vector.
    if spec.atol is not None and not isinstance(spec.atol, (int, float, ExactAtol)):
        raise ConversionError(
            f"cannot emit SED-ML for atol={spec.atol!r}: this is a per-species "
            f"absolute tolerance (issue #196), and {_KISAO_ATOL} (absolute "
            "tolerance) is a single value for the whole solve. SED-ML has no "
            "per-species form, so writing one entry — or any summary of the "
            "vector — would describe a different run than the spec does. Export "
            "with a scalar atol, or keep this protocol in the EvaluationSpec "
            "JSON, which round-trips the vector exactly."
        )

    ET.register_namespace("", _SEDML_NS)
    ET.register_namespace("mml", _MATHML_NS)
    source = model_source if model_source is not None else spec.model_source
    targets = _SbmlTargets(sbml if sbml is not None else source, model)
    sed = ET.Element(_q(_SEDML_NS, "sedML"), {"level": "1", "version": "3"})
    # The prefix every target is written under.
    sed.set("xmlns:sbml", targets.ns)

    # A synthesized-default marker (first child, before the listOf* elements):
    # machine-readable provenance that this protocol was fabricated, not parsed.
    if synthesized_default:
        annot = ET.SubElement(sed, _q(_SEDML_NS, "annotation"))
        marker = ET.SubElement(annot, _q(_BNGSIM_ANNOT_NS, "synthesizedDefault"))
        marker.text = _SYNTHESIZED_DEFAULT_NOTE

    # ── listOfSimulations: a single uniform time course ────────────────────
    sims = ET.SubElement(sed, _q(_SEDML_NS, "listOfSimulations"))
    t0, t_end = spec.t_span
    # SED-ML numberOfPoints counts the steps after outputStartTime, so the output
    # has numberOfPoints+1 rows; bngsim n_points includes t_start → steps = n-1.
    n_steps = max(int(spec.n_points) - 1, 0)
    utc = ET.SubElement(
        sims,
        _q(_SEDML_NS, "uniformTimeCourse"),
        _utc_attrs("sim0", t0, spec.output_start, t_end, n_steps),
    )
    algo = ET.SubElement(
        utc, _q(_SEDML_NS, "algorithm"), {"kisaoID": _METHOD_TO_KISAO[spec.method]}
    )
    params = []
    if spec.rtol is not None:
        params.append((_KISAO_RTOL, _num(spec.rtol)))
    if spec.atol is not None:
        # An ExactAtol goes out as its number: KISAO:0000211 has no way to say
        # "exactly", and read_sedml(exact_atol=True) is how a reader takes it so.
        atol = spec.atol.value if isinstance(spec.atol, ExactAtol) else float(spec.atol)  # type: ignore[arg-type]
        params.append((_KISAO_ATOL, _num(atol)))
    if spec.max_steps is not None:
        params.append((_KISAO_MAXSTEPS, str(int(spec.max_steps))))
    if spec.seed is not None:
        params.append((_KISAO_SEED, str(spec.seed)))
    _algorithm_param_elems(algo, params)

    # ── listOfModels / listOfTasks ─────────────────────────────────────────
    models = ET.SubElement(sed, _q(_SEDML_NS, "listOfModels"))
    ET.SubElement(
        models,
        _q(_SEDML_NS, "model"),
        {
            "id": "model0",
            "language": _SBML_LANG,
            "source": source,
        },
    )
    tasks = ET.SubElement(sed, _q(_SEDML_NS, "listOfTasks"))
    ET.SubElement(
        tasks,
        _q(_SEDML_NS, "task"),
        {"id": "task0", "modelReference": "model0", "simulationReference": "sim0"},
    )

    # ── listOfDataGenerators: a leading time generator + one per output ─────
    dgs = ET.SubElement(sed, _q(_SEDML_NS, "listOfDataGenerators"))
    _data_generator(dgs, "dg_time", "time", var_attrs={"symbol": _TIME_SYMBOL})
    dataset_refs: list[tuple[str, str]] = [("dg_time", "time")]
    untargeted: list[str] = []
    for i, selector in enumerate(spec.outputs):
        dg_id = f"dg_{i}"
        var_attrs = _selector_target(selector, targets)
        if not var_attrs:
            untargeted.append(selector)
        _data_generator(dgs, dg_id, selector, var_attrs=var_attrs)
        dataset_refs.append((dg_id, selector))
    if untargeted:
        warnings.warn(
            f"SED-ML's variables for {untargeted} have no target: "
            + (
                "the SBML has no element they name"
                if targets.read
                else "naming their SBML elements needs the SBML (pass sbml=)"
            )
            + "; a standards reader cannot read them, and bngsim reads their names",
            ConversionWarning,
            stacklevel=2,
        )

    # ── listOfOutputs: a report binding every data generator ───────────────
    outs = ET.SubElement(sed, _q(_SEDML_NS, "listOfOutputs"))
    report_name = (
        "bngsim default protocol (synthesized — source carried no protocol)"
        if synthesized_default
        else "bngsim report"
    )
    report = ET.SubElement(outs, _q(_SEDML_NS, "report"), {"id": "report0", "name": report_name})
    lds = ET.SubElement(report, _q(_SEDML_NS, "listOfDataSets"))
    for i, (dg_id, label) in enumerate(dataset_refs):
        ET.SubElement(
            lds,
            _q(_SEDML_NS, "dataSet"),
            {"id": f"ds_{i}", "label": label, "dataReference": dg_id},
        )

    ET.indent(sed, space="  ")
    text = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(sed, encoding="unicode")
    if not text.endswith("\n"):
        text += "\n"
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


def _algorithm_param_elems(algo: ET.Element, params: list[tuple[str, str]]) -> None:
    """Append an algorithm's ``listOfAlgorithmParameters``, if it has any."""
    if not params:
        return
    lop = ET.SubElement(algo, _q(_SEDML_NS, "listOfAlgorithmParameters"))
    for kid, val in params:
        ET.SubElement(lop, _q(_SEDML_NS, "algorithmParameter"), {"kisaoID": kid, "value": val})


def _utc_attrs(
    sim_id: str, t0: float, output_start: float | None, t_end: float, n_steps: int
) -> dict[str, str]:
    """A ``uniformTimeCourse``'s attributes: integrate from *t0*, output from
    *output_start* (``None``: from *t0*) to *t_end* in *n_steps* steps."""
    return {
        "id": sim_id,
        "initialTime": _num(t0),
        "outputStartTime": _num(t0 if output_start is None else output_start),
        "outputEndTime": _num(t_end),
        "numberOfPoints": str(n_steps),
    }


def _data_generator(
    parent: ET.Element, dg_id: str, name: str, *, var_attrs: dict, task: str = "task0"
) -> None:
    """Append a ``<dataGenerator>`` of a single variable of *task*'s results.

    ``name`` carries the verbatim bngsim selector (the round-trip source of
    truth); ``var_attrs`` adds standards-compliant ``symbol``/``target`` for
    interop. The required ``<math>`` is a bare ``<ci>`` over the variable id.
    """
    dg = ET.SubElement(parent, _q(_SEDML_NS, "dataGenerator"), {"id": dg_id, "name": name})
    lov = ET.SubElement(dg, _q(_SEDML_NS, "listOfVariables"))
    var_id = dg_id.replace("dg_", "var_", 1)
    attrs = {"id": var_id, "name": name, "taskReference": task}
    attrs.update(var_attrs)
    ET.SubElement(lov, _q(_SEDML_NS, "variable"), attrs)
    math = ET.SubElement(dg, _q(_MATHML_NS, "math"))
    ci = ET.SubElement(math, _q(_MATHML_NS, "ci"))
    ci.text = var_id


def _num(x: float) -> str:
    """Render a float compactly but losslessly for an XML attribute."""
    f = float(x)
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


# ─── Reading: SED-ML → EvaluationSpec ───────────────────────────────────────


def _local(tag: str) -> str:
    """Strip any ``{namespace}`` prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _warn_if_synthesized_default(root: ET.Element) -> None:
    """Surface the bngsim ``synthesizedDefault`` marker as a warning on read, so
    a consumer is told the protocol is a fabricated placeholder, not the source's."""
    for el in root.iter():
        if _local(el.tag) == "synthesizedDefault":
            warnings.warn(
                "this SED-ML carries a bngsim synthesized-DEFAULT protocol "
                "(the source model had no simulation protocol); it is a runnable "
                "placeholder, not the modeller's protocol",
                ConversionWarning,
                stacklevel=3,
            )
            return


def _find(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem.iter():
        if _local(child.tag) == name:
            return child
    return None


def _findall_local(elem: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in elem.iter() if _local(e.tag) == name]


def read_sedml(
    source: str | Path,
    *,
    model_source: str | None = None,
    model_format: str | None = None,
    exact_atol: bool = False,
) -> EvaluationSpec:
    """Parse a SED-ML document into an :class:`~bngsim.EvaluationSpec`.

    Reads the first **uniform time course** and its task's model reference,
    recovering the time grid, output selectors (from each data generator), and
    the solver + tolerances (from the algorithm's KiSAO terms). Namespace-
    agnostic, so SED-ML L1 V1–V4 all parse, V4's ``numberOfSteps`` included.
    The spec integrates from ``initialTime``; an ``outputStartTime`` past it is
    the spec's ``output_start``, where its output begins (issue #1493). A
    partitioned-leaping term (``KISAO:0000524``) is refused, since bngsim has
    no PLA, and NFsim's (``KISAO:0000263``) runs as the exact SSA it samples.

    Parameters
    ----------
    source : str | Path
        A path to a ``.sedml``/``.xml`` document, or the document text itself.
    model_source, model_format : str | None
        Override the model the returned spec points at — e.g. after
        ``sbml2net`` you run the SED-ML protocol against the converted ``.net``:
        ``read_sedml(sidecar, model_source="model.net", model_format="net")``.
        ``model_format`` defaults to ``"sbml"`` (the SED-ML model language) when
        not overridden.
    exact_atol : bool
        The document's absolute tolerance (``KISAO:0000211``) is one number.
        bngsim reads a scalar ``atol`` as a ceiling: each species is held to the
        smaller of it and ``rtol`` of its own scale, so a species far below the
        number is still resolved (issue #1420; see :meth:`bngsim.Simulator.run`).
        ``True`` holds every species to the document's number exactly instead,
        as a :class:`bngsim.ExactAtol`.

    Returns
    -------
    EvaluationSpec
    """
    text = _read_text(source)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ConversionError(f"not a parseable SED-ML/XML document: {exc}") from exc
    if _local(root.tag) not in ("sedML", "sedml"):
        raise ConversionError(
            f"root element is <{_local(root.tag)}>, not <sedML>: not a SED-ML document"
        )
    _warn_if_synthesized_default(root)

    # The first task, read for its simulation, its model and its outputs (issue
    # #1675): read as the first time course, the first model and every data
    # generator, a task on the second model listed ran the first, and a
    # two-task document came back with the other task's outputs.
    task = _find(root, "task")
    task_id = task.get("id") if task is not None else None
    others = [
        e.get("id") or "?"
        for e in root.iter()
        if _local(e.tag) in ("task", "repeatedTask") and e is not task
    ]
    if task is not None and others:
        warnings.warn(
            f"read_sedml reads the first task, {task_id!r}; the document's "
            f"{', '.join(repr(o) for o in others)} are not read. read_sedml_protocol reads "
            "every task (issue #1675)",
            ConversionWarning,
            stacklevel=2,
        )
    utc = None
    if task is not None:
        ref = task.get("simulationReference") or ""
        utc = next(
            (u for u in _findall_local(root, "uniformTimeCourse") if u.get("id") == ref), None
        )
        if utc is None:
            kind = next((_local(e.tag) for e in root.iter() if e.get("id") == ref), None)
            raise ConversionError(
                f"task {task_id!r} runs simulation {ref!r}, "
                + (f"a {kind}, " if kind else "which the document does not define, ")
                + "not a uniformTimeCourse, which is all read_sedml reads; read_sedml_protocol "
                "reads every task"
            )
    if utc is None:
        utc = _find(root, "uniformTimeCourse")
    if utc is None:
        raise ConversionError(
            "no <uniformTimeCourse> found — the SED-ML sidecar channel supports "
            "the uniform-time-course protocol only"
        )

    t0, output_start, t_end, n_points = _utc_grid(utc)

    method = "ode"
    rtol = atol = None
    max_steps = None
    seed = None
    algo = _find(utc, "algorithm")
    if algo is not None:
        kid = _kisao_id(algo)
        if kid:
            mapped = _KISAO_TO_METHOD.get(kid)
            if mapped is None:
                warnings.warn(
                    f"unmapped algorithm KiSAO term {kid!r}; defaulting method to "
                    "'ode'. Set method explicitly on the returned spec if needed.",
                    ConversionWarning,
                    stacklevel=2,
                )
            elif mapped == "pla":
                raise ConversionError(
                    f"the time course's algorithm is {kid} (partitioned leaping "
                    "method), and bngsim has no PLA method to run it with (issue #1500). "
                    "read_sedml_protocol keeps it as method 'pla' for a BNG2.pl actions block."
                )
            elif mapped == "nf":
                # The spec runs a network (.net or SBML), on which the exact SSA
                # samples the process NFsim samples from the rules.
                method = "ssa"
            else:
                method = mapped
        rtol, atol, max_steps = _algorithm_params(algo)
        seed = _algorithm_seed(algo)

    # Outputs. A document bngsim wrote carries the verbatim bngsim selector in
    # @name, which is the round-trip source of truth and still wins. Every other
    # writer follows the standard the other way round: @target holds the
    # machine-readable XPath and @name a display label ("Free receptor (nM)"),
    # so preferring @name there discarded the one resolvable field and handed
    # the caller prose that no lookup can match (GH #563). Take @name only when
    # it is a typed bngsim selector, else the selector the target yields, else
    # the name as a last resort — which is what an observable or expression
    # generator has, since neither has a standard SBML element to point at.
    outputs: list[str] = []
    computed: list[str] = []
    for dg in _findall_local(root, "dataGenerator"):
        var = _find(dg, "variable")
        # A generator of another task's run is not this task's output (#1675).
        if task_id is not None and var is not None:
            of_task = var.get("taskReference")
            if of_task is not None and of_task != task_id:
                continue
        # A generator is its math over its variables. Where that is not one
        # variable as it stands, no selector names it: read as its first
        # variable, A + B came back as A (issue #1663).
        var = _generator_variable(dg)
        if var is None:
            computed.append(_generator_text(dg))
            continue
        if (var.get("symbol") or "").strip() == _TIME_SYMBOL:
            continue
        name = dg.get("name") or (var.get("name") if var is not None else None)
        target_sel = _selector_from_var(var) if var is not None else None
        label = name if _is_typed_selector(name) else (target_sel or name)
        if label and label != "time":
            outputs.append(label)
    if computed:
        warnings.warn(
            f"the document's data generators {computed} compute a value from their "
            "variables, which no output selector names; they are not among the spec's "
            "outputs. Add the selectors of their variables to the outputs to compute "
            "them from (issue #1663)",
            ConversionWarning,
            stacklevel=2,
        )

    # The task's model, else the first listed: the file at the end of its
    # source chain, and the parameter values its changes set (issue #1675).
    models = {m.get("id"): m for m in _findall_local(root, "model")}
    chain = _model_chain(models, task.get("modelReference") if task is not None else None)
    if not chain and models:
        chain = [next(iter(models.values()))]
    resolved_source = model_source
    if resolved_source is None:
        resolved_source = chain[-1].get("source") if chain else ""
    params: dict[str, float] = {}
    for elem in chain[::-1]:
        for change in _model_changes(elem):
            named = _target_name(change.get("target") or "")
            kind = _local(change.tag)
            value = _number_or_none(change.get("newValue")) if kind == "changeAttribute" else None
            if named is not None and named[0] == "param" and value is not None:
                params[named[1]] = value
                continue
            warnings.warn(
                f"model {elem.get('id')!r}: its {kind} of {change.get('target')!r} is not a "
                "parameter value, which is all an EvaluationSpec carries; read_sedml_protocol "
                "reads the document's model changes in full (issue #1675)",
                ConversionWarning,
                stacklevel=2,
            )

    return EvaluationSpec(
        model_source=resolved_source or "",
        model_format=model_format or "sbml",
        method=method,
        t_span=(t0, t_end),
        output_start=output_start,
        n_points=n_points,
        params=params,
        outputs=tuple(outputs),
        rtol=rtol,
        atol=ExactAtol(atol) if exact_atol and atol is not None else atol,
        max_steps=max_steps,
        seed=seed,
    )


def _generator_variable(dg: ET.Element) -> ET.Element | None:
    """The variable a data generator's math is, as it stands, or ``None``
    where its math computes something else from its variables: a sum, a
    scaling, a parameter of its own. With no math, a generator of one
    variable is that variable."""
    variables = {(v.get("id") or "").strip(): v for v in _findall_local(dg, "variable")}
    math = _find(dg, "math")
    terms = list(math) if math is not None else []
    if not terms:
        return next(iter(variables.values())) if len(variables) == 1 else None
    if len(terms) == 1 and _local(terms[0].tag) == "ci":
        return variables.get((terms[0].text or "").strip())
    return None


def _generator_text(dg: ET.Element) -> str:
    """A data generator, named for a warning: its id, its math as a formula
    over its variables where libsbml can write it, and the selector each
    variable names where it names one."""
    gid = dg.get("id") or "?"
    math = _find(dg, "math")
    formula = None
    if math is not None:
        try:
            import libsbml

            node = libsbml.readMathMLFromString(ET.tostring(math, encoding="unicode"))
            formula = libsbml.formulaToL3String(node) if node is not None else None
        except ImportError:  # pragma: no cover - the SBML extra is installed with the readers
            formula = None
    named = [
        f"{v.get('id')} = {sel}"
        for v in _findall_local(dg, "variable")
        if (sel := _selector_from_var(v)) is not None
    ]
    text = f"{gid} = {formula}" if formula else gid
    return text + (f" (of {', '.join(named)})" if named else "")


def _kisao_id(elem: ET.Element) -> str:
    """An element's KiSAO id, ``""`` if it has none. ``KISAO_0000694``, as
    some BioModels documents write it, is read as ``KISAO:0000694``."""
    raw = (elem.get("kisaoID") or elem.get("kisaoid") or "").strip()
    if raw[:6].upper() == "KISAO_":
        raw = "KISAO:" + raw[6:]
    return raw


def _algorithm_params(algo: ET.Element) -> tuple[float | None, float | None, int | None]:
    """An algorithm's ``(rtol, atol, max_steps)`` from its KiSAO parameters."""
    rtol = atol = None
    max_steps = None
    for ap in _findall_local(algo, "algorithmParameter"):
        apk = _kisao_id(ap)
        val = ap.get("value")
        if val is None:
            continue
        if apk == _KISAO_RTOL:
            rtol = float(val)
        elif apk == _KISAO_ATOL:
            atol = float(val)
        elif apk == _KISAO_MAXSTEPS:
            max_steps = int(float(val))
    return rtol, atol, max_steps


def _algorithm_seed(algo: ET.Element) -> int | None:
    """An algorithm's random seed (``KISAO:0000488``), ``None`` if it has none.

    It was dropped with no word, so a seeded document ran on a fresh seed each
    time (issue #1517). It is read exactly, as an int: a seed past 2**53 does
    not survive a float. A seed that is not an integer from 0 to 2**64 - 1 is
    refused. A seed another tool's document names reproduces bngsim's own run
    of it, not that tool's: the generators differ.
    """
    for ap in _findall_local(algo, "algorithmParameter"):
        if _kisao_id(ap) != _KISAO_SEED:
            continue
        raw = (ap.get("value") or "").strip()
        try:
            seed = int(raw)
        except ValueError:
            try:
                as_float = float(raw)
            except ValueError:
                as_float = math.nan
            if not as_float.is_integer():
                raise ConversionError(
                    f"the algorithm's random seed ({_KISAO_SEED}) is {raw!r}, not an integer"
                ) from None
            seed = int(as_float)
        if not 0 <= seed <= 2**64 - 1:
            raise ConversionError(
                f"the algorithm's random seed ({_KISAO_SEED}) is {seed}; a seed is an "
                "integer from 0 to 2**64 - 1"
            )
        return seed
    return None


def _steps_attr(elem: ET.Element) -> str | None:
    """A uniform grid's step count as written: SED-ML L1V4 names it
    ``numberOfSteps`` and L1V1–V3 ``numberOfPoints``, both counting the
    intervals, so the grid has one point more (issue #1493)."""
    raw = elem.get("numberOfSteps")
    return raw if raw is not None else elem.get("numberOfPoints")


def _utc_grid(utc: ET.Element) -> tuple[float, float | None, float, int]:
    """A ``uniformTimeCourse``'s ``(initial time, output start, output end,
    output points)``. SED-ML integrates from ``initialTime`` and reports only
    ``outputStartTime``..``outputEndTime``; taking the output start as the run's
    start restarted the model there from its initial state (issue #1493). The
    output start is ``None`` where it is the initial time."""
    sim = utc.get("id") or "the uniformTimeCourse"
    initial_raw, start_raw = utc.get("initialTime"), utc.get("outputStartTime")
    initial = float(initial_raw if initial_raw is not None else (start_raw or 0.0))
    start = float(start_raw) if start_raw is not None else initial
    t_end = _float_attr(utc, "outputEndTime", 100.0)
    steps = _steps_attr(utc)
    n_points = int(float(steps) if steps is not None else 100.0) + 1
    if not initial <= start <= t_end:
        raise ConversionError(
            f"{sim}: SED-ML needs initialTime <= outputStartTime <= outputEndTime, and "
            f"this one has {initial:g}, {start:g}, {t_end:g}"
        )
    return initial, (None if start == initial else start), t_end, n_points


# The kinds Result.resolve_outputs accepts, aliases included. A ``@name`` in
# this shape was written by bngsim (or by a tool following it) and is already
# the selector the caller wants; anything else is a display label.
_SELECTOR_KINDS = ("species", "observable", "expression", "state", "function")
_TYPED_SELECTOR_RE = re.compile(rf"\A(?:{'|'.join(_SELECTOR_KINDS)}):\S")

# ``species[@id='R']`` inside an SBML XPath. The namespace prefix is whatever
# the document bound (``sbml:``, ``s:``, or none at all), so it is not matched;
# either quoting style is, since both are well-formed XML.
_SPECIES_TARGET_RE = re.compile(r"""species\[\s*@id\s*=\s*['"]([^'"]+)['"]\s*\]""")


def _is_typed_selector(label: str | None) -> bool:
    """True when *label* already reads as a bngsim ``<kind>:<name>`` selector."""
    if not label:
        return False
    return _TYPED_SELECTOR_RE.match(label.strip()) is not None


def _selector_from_var(var: ET.Element) -> str | None:
    """Best-effort bngsim selector from a SED-ML variable's target/symbol."""
    target = var.get("target")
    if target:
        m = _SPECIES_TARGET_RE.search(target)
        if m:
            return f"species:{m.group(1)}"
    return None


def _read_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text()
    s = str(source)
    if s.lstrip().startswith("<"):  # inline XML (well-formed or not), not a path
        return s
    return Path(s).read_text()


def _float_attr(elem: ET.Element, name: str, default: float) -> float:
    v = elem.get(name)
    return float(v) if v is not None else default


# ─── Default protocol (no-sidecar case) ─────────────────────────────────────


def default_protocol(
    model: Model | str | Path,
    *,
    model_source: str | None = None,
    model_format: str = "sbml",
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
    method: str = "ode",
    rtol: float | None = None,
    atol: float | None = None,
) -> EvaluationSpec:
    """Build a sensible default :class:`~bngsim.EvaluationSpec` for a model.

    Outputs default to every observable (``observable:<name>``); if the model
    declares none, every species (``species:<name>``) is reported instead. This
    is the *no-sidecar* fallback and the protocol ``net2sbml --sidecar`` emits.

    ``model`` may be a loaded :class:`bngsim.Model` or a path to load. The
    returned spec's ``model_source``/``model_format`` point at ``model_source``
    (or the given path) so the sidecar references the right artifact.
    """
    from bngsim._model import Model as _Model

    if isinstance(model, (str, Path)):
        path = Path(model)
        loader = _Model.from_net if path.suffix.lower() == ".net" else _Model.from_sbml
        loaded = loader(path)
        if model_source is None:
            model_source = str(path)
            model_format = "net" if path.suffix.lower() == ".net" else "sbml"
    else:
        loaded = model

    obs = list(loaded.observable_names) if hasattr(loaded, "observable_names") else []
    if obs:
        outputs = tuple(f"observable:{n}" for n in obs)
    else:
        outputs = tuple(f"species:{n}" for n in loaded.species_names)

    return EvaluationSpec(
        model_source=model_source if model_source is not None else "",
        model_format=model_format,
        method=method,
        t_span=t_span,
        n_points=n_points,
        outputs=outputs,
        rtol=rtol,
        atol=atol,
    )


# ─── Multi-experiment protocol: ProtocolSpec ⇄ SED-ML (GH #211, Option 3) ────
# A whole BNGL actions block is a *sequence* of experiments (simulate /
# parameter_scan) with interleaved model-state changes — more than a single
# uniform time course. This pair maps a ProtocolSpec to/from SED-ML, emitting
# standards-compliant elements (one uniformTimeCourse + task per experiment; a
# repeatedTask + range per scan; changeAttribute-derived models for the
# accumulated set* overrides) for interop, PLUS a bngsim <annotation> carrying
# the verbatim ProtocolSpec JSON so the round-trip is EXACT (the same
# fidelity-via-annotation contract #218 uses for the single spec).

# BNG2.pl's methods → KiSAO (issue #1500). NFsim and PLA have terms of their own;
# filing them under Gillespie's direct method had a standards reader, bngsim's
# own load_protocol among them, run an exact SSA for a PLA run. RuleMonkey has
# none and samples the exact SSA's process, as NFsim does. PSA has none either
# and approximates the exact SSA; the writer says so when it writes one.
_KISAO_GILLESPIE = _METHOD_TO_KISAO["ssa"]
_PROTO_METHOD_TO_KISAO = {
    **_METHOD_TO_KISAO,
    "nf": _KISAO_NFSIM,
    "pla": _KISAO_PLA,
    "rm": _KISAO_GILLESPIE,
    "psa": _KISAO_GILLESPIE,
}


def _protocol_algorithm(exp, i: int) -> tuple[str, list[tuple[str, str]]]:
    """An experiment's KiSAO algorithm term and parameters, warning of what of
    the run they cannot hold (issue #1500). An ODE run's rtol and atol are
    BNG2.pl's, its defaults where the action gives none: a reader at its own
    defaults would run another experiment."""
    from bngsim._protocol_run import _BNG_ATOL, _BNG_RTOL, _method_of
    from bngsim.convert._protocol import _num_or

    method = _method_of(exp)
    kisao = _PROTO_METHOD_TO_KISAO.get(method)
    if kisao is None:
        warnings.warn(
            f"experiment {i}'s method {exp.method!r} has no KiSAO term; SED-ML names "
            f"Gillespie's direct method ({_KISAO_GILLESPIE}) for it, and the method is "
            "in the bngsim annotation",
            ConversionWarning,
            stacklevel=3,
        )
        kisao = _KISAO_GILLESPIE
    elif method == "psa":
        warnings.warn(
            f'experiment {i} runs PSA (BNG2.pl runs it for method=>"psa" or a '
            "poplevel), which has no KiSAO term; SED-ML names Gillespie's direct "
            f"method ({_KISAO_GILLESPIE}), the exact SSA it approximates, and the "
            "method and poplevel are in the bngsim annotation",
            ConversionWarning,
            stacklevel=3,
        )
    params: list[tuple[str, str]] = []
    lost: list[str] = []
    if method == "ode":
        for key, kid, default in (
            ("rtol", _KISAO_RTOL, _BNG_RTOL),
            ("atol", _KISAO_ATOL, _BNG_ATOL),
        ):
            raw = exp.extra.get(key)
            value = default if raw is None else _num_or(raw, None)
            if value is None:
                lost.append(f"{key}=>{raw}, which cannot be read without the model")
            else:
                params.append((kid, _num(value)))
    else:
        # A stochastic run's seed, as KISAO:0000488 (issue #1517). An int goes
        # out exactly; a number written as a float only when it is integral.
        raw = exp.extra.get("seed")
        if raw is not None:
            if isinstance(raw, int) and not isinstance(raw, bool):
                seed: int | None = raw
            else:
                value = _num_or(raw, None)
                seed = int(value) if value is not None and float(value).is_integer() else None
            if seed is None or not 0 <= seed <= 2**64 - 1:
                lost.append(f"seed=>{raw}, which is not an integer seed")
            else:
                params.append((_KISAO_SEED, str(seed)))
    for key in ("pla_config", "steady_state"):
        if key in exp.extra:
            lost.append(f"{key}=>{exp.extra[key]}")
    if lost:
        warnings.warn(
            f"SED-ML's algorithm for experiment {i} does not hold its "
            f"{'; '.join(lost)}; the bngsim annotation does",
            ConversionWarning,
            stacklevel=3,
        )
    return kisao, params


# ─── SBML targets ────────────────────────────────────────────────────────────
# A SED-ML target is an XPath into the SBML the document points at, under a
# prefix the document declares (SED-ML L1V3 §2.2.6). bngsim's SBML writer
# sanitizes and de-duplicates ids — `A()` is `A`, or `A_2` beside an observable
# `A` — and keeps the BNGL string as the species' name, so a BNGL pattern names
# an SBML element only through that SBML. Writing the pattern as the id
# (`species[@id='A()']`) addressed nothing, and the prefix was never declared.

# What bngsim's SBML writer emits (convert/_sbml_writer.py, SBML L3V2).
_SBML_DEFAULT_NS = "http://www.sbml.org/sbml/level3/version2/core"
_SID = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


def _sbml_text(sbml) -> str | None:
    """The SBML document *sbml* gives: its text, a path to it, or ``None``."""
    if sbml is None:
        return None
    if isinstance(sbml, Path):
        return sbml.read_text() if sbml.is_file() else None
    text = str(sbml)
    if text.lstrip().startswith("<"):
        return text
    path = Path(text)
    return path.read_text() if path.is_file() else None


class _SbmlTargets:
    """The SBML a document points at, read for what its targets need: the
    namespace the ``sbml`` prefix binds, each species' id, units and
    compartment size, and each parameter's id and value."""

    def __init__(self, sbml: str | Path | None = None, model: Model | None = None) -> None:
        self.model = model
        self.ns = _SBML_DEFAULT_NS
        # name (the BNGL string, or the id where the SBML gives no name) → id
        self.species: dict[str, str] = {}
        # species id → its size where it is amount-valued (None: not a number)
        self.amount_scale: dict[str, float | None] = {}
        self.compartment_size: dict[str, float | None] = {}  # species id → its size, if a number
        self.params: dict[str, str] = {}
        self.param_values: dict[str, float] = {}  # parameter id → its value
        # The ids an initialAssignment computes, which a change to the value
        # the element holds does not move; and for each id one reads, those.
        self.initially_assigned: set[str] = set()
        self.assignment_reads: dict[str, set[str]] = {}
        # Each parameter's or species' initialAssignment that SED-ML math can
        # say again (issue #1733): its MathML, as text, whose names are all
        # parameters of the SBML.
        self.assignments: dict[str, str] = {}
        # Each species' own initial value, where the SBML holds it as a number
        # that nothing computes.
        self.species_values: dict[str, float] = {}
        self.read = False
        text = _sbml_text(sbml)
        if text is None:
            return
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return
        if not root.tag.startswith("{") or _local(root.tag) != "sbml":
            return
        self.ns = root.tag[1:].split("}", 1)[0]
        self.read = True
        maths: dict[str, ET.Element | None] = {}
        for e in root.iter():
            if _local(e.tag) == "initialAssignment" and e.get("symbol"):
                symbol = e.get("symbol") or ""
                self.initially_assigned.add(symbol)
                for ci in _findall_local(e, "ci"):
                    self.assignment_reads.setdefault((ci.text or "").strip(), set()).add(symbol)
                math = _find(e, "math")
                maths[symbol] = math[0] if math is not None and len(math) == 1 else None
        # A size the SBML computes (an initial assignment or a rule on the
        # compartment), or leaves out, is no number here (issue #1662).
        computed = {
            e.get("symbol") or e.get("variable")
            for e in root.iter()
            if _local(e.tag) in ("initialAssignment", "assignmentRule", "rateRule")
        }
        known: dict[str, float | None] = {}
        for c in _findall_local(root, "compartment"):
            cid, text = c.get("id"), c.get("size")
            if cid:
                known[cid] = None if text is None or cid in computed else _number_or_none(text)
        for sp in _findall_local(root, "species"):
            sid = sp.get("id")
            if not sid:
                continue
            self.species[sp.get("name") or sid] = sid
            self.species.setdefault(sid, sid)
            comp = sp.get("compartment")
            size = known.get(comp, 1.0) if comp else 1.0
            self.compartment_size[sid] = size
            amount = (sp.get("hasOnlySubstanceUnits") or "false").lower() == "true"
            if amount:
                self.amount_scale[sid] = size
            held = _number_or_none(sp.get("initialAmount" if amount else "initialConcentration"))
            if held is not None and sid not in computed:
                self.species_values[sid] = held
        for p in _findall_local(root, "parameter"):
            pid = p.get("id")
            if not pid:
                continue
            self.params[p.get("name") or pid] = pid
            self.params.setdefault(pid, pid)
            value = p.get("value")
            if value is not None:
                self.param_values[pid] = float(value)
        param_ids = set(self.params.values())
        species_ids = set(self.species.values())
        for symbol, body in maths.items():
            if body is None or symbol not in param_ids | species_ids:
                continue
            # A name SED-ML math reads through a variable of the parameter's
            # value; a csymbol (time, a delay) or a call of the SBML's own
            # function has none. Nor does SED-ML's MathML have every operator
            # SBML L3V2's does.
            if any(_local(e.tag) in ("csymbol", *_NOT_IN_SEDML_MATH) for e in body.iter()):
                continue
            if all((ci.text or "").strip() in param_ids for ci in _findall_local(body, "ci")):
                self.assignments[symbol] = ET.tostring(body, encoding="unicode")

    def species_id(self, pattern: str) -> str | None:
        """The SBML id of the species a BNGL pattern names, as BNG2.pl and the
        replay resolve it (:func:`bngsim._bngl_loader._species_name`)."""
        from bngsim._bngl_loader import _molecule_key, _network_spelling, _species_name

        names: list[str] = []
        if self.model is not None:
            name = _species_name(self.model, pattern)
            if name is not None:
                names.append(name)
        names += [pattern, _network_spelling(pattern)]
        for name in names:
            if name in self.species:
                return self.species[name]
        key = _molecule_key(pattern)
        if key is not None:
            for name, sid in self.species.items():
                if _molecule_key(name) == key:
                    return sid
        return None

    def param_id(self, name: str) -> str | None:
        if name in self.params:
            return self.params[name]
        # Without the SBML, a name that is an SId is the id the writer keeps.
        return name if not self.read and _SID.match(name) else None

    @staticmethod
    def species_path(sid: str) -> str:
        return f"/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{sid}']"

    @staticmethod
    def param_path(pid: str) -> str:
        return f"/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='{pid}']"

    def initial_attribute(self, kind: str, sid: str) -> str:
        if kind == "param":
            return self.param_path(sid) + "/@value"
        attr = "initialAmount" if sid in self.amount_scale else "initialConcentration"
        return self.species_path(sid) + f"/@{attr}"


class _Change(NamedTuple):
    """One change a protocol step makes to the model's state, on SBML ids.

    ``value`` is the number written, or ``None`` where ``expr`` is the BNGL
    expression the action writes; ``add`` is an ``addConcentration``. ``scale``
    turns a BNGL amount into the SBML species' own value: its compartment's
    size for an amount-valued species (the SBML writer's ``initialAmount``),
    else 1. ``follows`` marks a parameter set to an expression that BNG2.pl
    keeps as that expression, so it is set again when what it reads moves
    (issue #1682), and a species set to one, which BNG2.pl keeps as that
    expression until a run. ``initial`` is the MathML of the SBML's initialAssignment
    of the element, where the change sets it to what that computes from the
    model as it stands (issue #1733)."""

    kind: str  # "species" | "param"
    sid: str
    value: float | None
    expr: str | None
    add: bool
    scale: float
    action: str
    follows: bool = False
    # A resetParameters back to the value the SBML gives (issue #1668).
    restores: bool = False
    initial: str | None = None


# ─── Multi-experiment writer ─────────────────────────────────────────────────


class _ProtocolWriter:
    """:func:`write_sedml_protocol`'s document, built step by step.

    Each experiment BNG2.pl starts from the model's initial state, with the
    ``set*``/``add*`` actions since then, is a task on a derived model that
    holds those changes (``changeAttribute``, or ``computeChange`` where the
    value is an expression or an addition). One BNG2.pl starts where earlier
    runs left the model is a ``repeatedTask`` (``resetModel="true"``, one
    iteration) whose ordered subtasks replay those runs on the base model, with
    the actions between them as ``setValue`` changes: in SED-ML, unlike tasks,
    "model states are not reset unless explicitly told to reset with the
    resetModel=true attribute" (L1V4; SED-ML/sed-ml#54). A simulator that resets
    between subtasks departs from that, and is not written around here.

    Each experiment's report reads the task that runs it, so the document says
    which of its tasks are the experiments and which are parts of one (issue
    #1686). A parameter set to an expression of other parameters is set again
    after each change to what it reads, as BNG2.pl keeps it as that expression
    (issue #1682).
    """

    def __init__(self, protocol, targets: _SbmlTargets, model_source: str):
        self.protocol = protocol
        self.targets = targets
        ET.register_namespace("", _SEDML_NS)
        ET.register_namespace("mml", _MATHML_NS)
        self.sed = ET.Element(_q(_SEDML_NS, "sedML"), {"level": "1", "version": "3"})
        # The prefix every target is written under (issue: it was never declared).
        self.sed.set("xmlns:sbml", targets.ns)
        annot = ET.SubElement(self.sed, _q(_SEDML_NS, "annotation"))
        payload = ET.SubElement(annot, _q(_BNGSIM_ANNOT_NS, "protocol"))
        # The source's file name, not the path it was read from: the document
        # is written to be shared, and a path holds the writer's directory
        # names (issue #1678).
        payload.text = _shareable(protocol).to_json()
        # The L1V3 schema's order for <sedML>'s lists: simulations, models,
        # tasks, data generators, outputs (issue #1677).
        self.sims = ET.SubElement(self.sed, _q(_SEDML_NS, "listOfSimulations"))
        self.models = ET.SubElement(self.sed, _q(_SEDML_NS, "listOfModels"))
        ET.SubElement(
            self.models,
            _q(_SEDML_NS, "model"),
            {"id": "model0", "language": _SBML_LANG, "source": model_source},
        )
        self.tasks = ET.SubElement(self.sed, _q(_SEDML_NS, "listOfTasks"))
        self._ids: dict[str, int] = {}
        self._runs: dict[int, str] = {}
        self._sweeps: dict[int, str] = {}
        # Per experiment: the expressions its sweep sets at each point, as they
        # stood when it ran (a sweep written later, as part of a chain, reads these).
        self._following: dict[int, list[_Change]] = {}
        # Per experiment, in order: the task that runs it, which its report reads.
        self._experiments: list[tuple[int, Any, str]] = []
        self._reads_cache: dict[_Change, frozenset[str]] = {}
        self._noted: set[str] = set()
        # Per experiment: the parameters the protocol had replaced when it ran.
        self._replaced_at: dict[int, frozenset[str]] = {}
        # Per sweep from the initial state: the species the protocol set to an
        # expression, which BNG2.pl keeps as one until a run.
        self._live: dict[int, list[_Change]] = {}
        # The species an action changed before the network was generated,
        # whose seed as the model declares it the SBML does not hold (None:
        # one the SBML does not name).
        self._undeclared: set[str | None] = set()
        # What each initialAssignment of the SBML sets its element to, as a
        # change: what BNG2.pl's derived parameter follows, and a species
        # whose seed names a parameter until a run (issue #1733).
        param_ids = set(targets.params.values())
        self._assigned: dict[str, _Change] = {
            sid: _Change(
                kind="param" if sid in param_ids else "species",
                sid=sid,
                value=None,
                expr=None,
                add=False,
                scale=1.0,
                action=f"the SBML's initialAssignment of {sid}",
                follows=True,
                initial=math,
            )
            for sid, math in targets.assignments.items()
        }

    # ── ids and small elements ─────────────────────────────────────────────
    def _id(self, stem: str) -> str:
        n = self._ids.get(stem, 0)
        self._ids[stem] = n + 1
        return f"{stem}_{n}"

    @staticmethod
    def _say(text: str) -> None:
        """What the standard elements do not hold of the protocol, said. The
        bngsim annotation holds it; :func:`~bngsim.convert.net_to_omex`
        records each in the archive (issue #1684)."""
        warnings.warn(text, ConversionWarning, stacklevel=_outside())

    def _set_as_it_runs(self, pid: str, *, species: bool, replaced: frozenset[str]) -> None:
        """Say where a setValue sets *pid* as the model runs and the SBML
        computes another value from it by an initialAssignment that SED-ML
        math cannot say again (one that reads time, or calls a function): a
        simulator evaluates that when it loads (or resets) the model, not when
        *pid* is set, where BNG2.pl's derived parameter follows it (and a
        species whose amount is an expression of it, until a run). The bngsim
        annotation holds the protocol (issue #1684). One the protocol has
        *replaced* by then (set to a number, or to an expression, which is set
        again) is what the standard elements say, and so is one they set again
        after *pid* (issue #1733)."""
        reads = self.targets.assignment_reads.get(pid, set())
        params = set(self.targets.params.values())
        follow = sorted(
            r
            for r in reads
            if (r in params or species) and r not in replaced and r not in self._assigned
        )
        if not follow or pid in self._noted:
            return
        self._noted.add(pid)
        self._say(
            f"SED-ML's standard elements set {pid} as the model runs, and the SBML computes "
            f"{follow} from it by an initialAssignment, which a simulator evaluates when it "
            f"loads the model; BNG2.pl's follow {pid} as it is set. The bngsim annotation "
            "holds the protocol"
        )

    def _math(
        self,
        parent: ET.Element,
        change: _Change,
        model_ref: str,
        ranged: dict[str, str] | None = None,
    ) -> None:
        """A change's ``listOfVariables`` and ``math`` under *parent* (a
        ``setValue`` or a ``computeChange``): its value in the SBML species'
        own units, plus the species' value for an addition. A parameter in
        *ranged* is read as the range it is set from (parameter id → range
        id), not from the model."""
        variables: list[tuple[str, str]] = []
        terms: list[ET.Element] = []
        if change.add:
            vid = self._id("v")
            variables.append((vid, self.targets.species_path(change.sid)))
            terms.append(_ci(vid))
        if change.value is not None:
            terms.append(_cn(change.value * change.scale))
        elif change.initial is not None:
            # The SBML's own math, whose names are parameters (issue #1733).
            body = ET.fromstring(change.initial)
            vids: dict[str, str] = {}
            for ci in body.iter(_q(_MATHML_NS, "ci")):
                name = (ci.text or "").strip()
                if ranged and name in ranged:
                    ci.text = ranged[name]
                    continue
                if name not in vids:
                    vids[name] = self._id("v")
                    variables.append((vids[name], self.targets.param_path(name)))
                ci.text = vids[name]
            terms.append(body)
        else:
            expr, refs = self._expression(change.expr or "")
            for name, vid in refs.items():
                pid = self.targets.param_id(name) or name
                if ranged and pid in ranged:
                    for ci in expr.iter(_q(_MATHML_NS, "ci")):
                        if (ci.text or "").strip() == vid:
                            ci.text = ranged[pid]
                    continue
                variables.append((vid, self.targets.param_path(pid)))
            terms.append(_scaled(expr, change.scale))
        if variables:
            lov = ET.SubElement(parent, _q(_SEDML_NS, "listOfVariables"))
            for vid, target in variables:
                ET.SubElement(
                    lov,
                    _q(_SEDML_NS, "variable"),
                    {"id": vid, "target": target, "modelReference": model_ref},
                )
        math = ET.SubElement(parent, _q(_MATHML_NS, "math"))
        math.append(_without_min_max(_apply("plus", terms) if len(terms) > 1 else terms[0]))

    def _ast(self, text: str):
        """A BNGL value expression as a libSBML AST, and the names it reads."""
        import libsbml

        from bngsim.convert._sbml_writer import _exprtk_to_ast

        ast = _exprtk_to_ast(
            text, where=f"the value {text!r}", scalar_names=frozenset(self.targets.params)
        )
        names: list[str] = []

        def walk(node) -> None:
            if node.getType() == libsbml.AST_NAME and node.getName() not in names:
                names.append(node.getName())
            for k in range(node.getNumChildren()):
                walk(node.getChild(k))

        walk(ast)
        return ast, names

    def _expression(self, text: str) -> tuple[ET.Element, dict[str, str]]:
        """A BNGL value expression as MathML whose names are SED-ML variables,
        and the model name each variable reads."""
        import libsbml

        ast, names = self._ast(text)
        refs: dict[str, str] = {}
        for name in names:
            vid = self._id("v")
            refs[name] = vid
            ast.renameSIdRefs(name, vid)
        root = ET.fromstring(libsbml.writeMathMLToString(ast))
        # writeMathMLToString wraps the expression in its own <math>.
        return root[0], refs

    def _model_change(self, changes: ET.Element, change: _Change, model_ref: str) -> None:
        target = self.targets.initial_attribute(change.kind, change.sid)
        if change.value is not None and not change.add:
            ET.SubElement(
                changes,
                _q(_SEDML_NS, "changeAttribute"),
                {"target": target, "newValue": _num(change.value * change.scale)},
            )
            return
        cc = ET.SubElement(changes, _q(_SEDML_NS, "computeChange"), {"target": target})
        self._math(cc, change, model_ref)

    def _set_value(
        self,
        changes: ET.Element,
        change: _Change,
        model_ref: str,
        replaced: frozenset[str],
        *,
        fresh: bool = False,
    ) -> None:
        if change.kind == "param":
            self._set_as_it_runs(change.sid, species=fresh, replaced=replaced)
        path = (
            self.targets.param_path(change.sid)
            if change.kind == "param"
            else self.targets.species_path(change.sid)
        )
        sv = ET.SubElement(
            changes, _q(_SEDML_NS, "setValue"), {"target": path, "modelReference": model_ref}
        )
        self._math(sv, change, model_ref)

    def _once(self, rt: ET.Element) -> None:
        rid = self._id("once")
        rt.set("range", rid)
        lor = ET.SubElement(rt, _q(_SEDML_NS, "listOfRanges"))
        vr = ET.SubElement(lor, _q(_SEDML_NS, "vectorRange"), {"id": rid})
        ET.SubElement(vr, _q(_SEDML_NS, "value")).text = "0"

    def _subtasks(self, rt: ET.Element, task_ids: list[str]) -> None:
        lost = ET.SubElement(rt, _q(_SEDML_NS, "listOfSubTasks"))
        for order, tid in enumerate(task_ids, start=1):
            ET.SubElement(lost, _q(_SEDML_NS, "subTask"), {"order": str(order), "task": tid})

    # ── the change a step makes ────────────────────────────────────────────
    def change(self, step) -> _Change | None:
        """The change *step* (a set or add) makes, or ``None``, said, where the
        SBML has nothing it names or its value cannot be written."""
        from bngsim.convert._protocol import _state_change_action

        action = _state_change_action(step)
        kind = "param" if step.kind == "set_parameter" else "species"
        scale: float | None = 1.0
        if kind == "param":
            sid = self.targets.param_id(step.target or "")
        else:
            sid = self.targets.species_id(step.target or "") if self.targets.read else None
            if sid:
                scale = self.targets.amount_scale.get(sid, 1.0)
        why = None
        if sid is None:
            why = (
                "the SBML has no element it names"
                if self.targets.read
                else "naming its SBML species needs the SBML (pass sbml=)"
            )
        elif scale is None:
            # An amount-valued species' value is its concentration times its
            # compartment's size, which the SBML computes or leaves out.
            why = (
                "its compartment's size is not a number in the SBML, "
                "so its amount cannot be written"
            )
        elif step.value is None:
            try:
                _, names = self._ast(step.value_expr or "")
            except ConversionError as exc:
                why = f"its value cannot be written as MathML ({exc})"
            else:
                unknown = [n for n in names if self.targets.param_id(n) is None]
                if unknown:
                    why = f"its value reads {unknown}, which the SBML has no parameter for"
        if why is not None or sid is None or scale is None:
            self._say(
                f"SED-ML's standard elements leave out {action}: {why}; the bngsim "
                "annotation holds it"
            )
            return None
        expr = step.value_expr if step.value is None else None
        return _Change(
            kind=kind,
            sid=sid,
            value=None if step.value is None else float(step.value),
            expr=expr,
            add=step.kind == "add_concentration",
            scale=scale,
            action=action,
            # BNG2.pl keeps a parameter set to an expression as that expression,
            # and a species set to one until a run.
            follows=(
                expr is not None and step.kind != "add_concentration" and not step.evaluated_once
            ),
        )

    def original(self, pid: str) -> _Change | None:
        """A parameter back at the value the SBML holds: for one the SBML
        computes by an initialAssignment, back to what that computes from the
        model as it stands, which BNG2.pl's follows (issue #1733)."""
        assigned = self._assigned.get(pid)
        if assigned is not None and assigned.kind == "param":
            return assigned._replace(action=f"resetParameters ({pid})", restores=True)
        value = self.targets.param_values.get(pid)
        if value is None:
            self._say(
                f"SED-ML's standard elements cannot reset parameter {pid}: the value it "
                "goes back to needs the SBML (pass sbml=); the bngsim annotation holds it"
            )
            return None
        return _Change(
            "param", pid, value, None, False, 1.0, f"resetParameters ({pid})", restores=True
        )

    # ── parameters that follow others (issue #1682) ────────────────────────
    def _reads(self, change: _Change) -> frozenset[str]:
        """The parameter ids an expression change reads."""
        if change not in self._reads_cache:
            if change.initial is not None:
                body = ET.fromstring(change.initial)
                reads = frozenset(
                    (ci.text or "").strip() for ci in body.iter(_q(_MATHML_NS, "ci"))
                )
            else:
                _, names = self._ast(change.expr or "")
                reads = frozenset(self.targets.param_id(n) or n for n in names)
            self._reads_cache[change] = reads
        return self._reads_cache[change]

    def _definitions(self, params: dict[str, _Change]) -> dict[str, _Change]:
        """The parameters that follow others, by id, each with the change that
        sets it again: an expression the protocol set (issue #1682), else the
        SBML's initialAssignment of one the protocol has not replaced, which
        BNG2.pl's derived parameter follows (issue #1733)."""
        out = {pid: change for pid, change in params.items() if change.follows}
        for pid, change in self._assigned.items():
            if change.kind == "param" and (pid not in params or params[pid].restores):
                out.setdefault(pid, change)
        return out

    def _seeds(self, moved: set[str], kept: frozenset[str]) -> list[_Change]:
        """The species whose seed the SBML computes from a parameter in *moved*
        by an initialAssignment, but those in *kept*, which the protocol has
        set: what BNG2.pl's amounts are once *moved* move, before a run has
        made them numbers (issue #1733)."""
        return [
            change
            for sid, change in self._assigned.items()
            if change.kind == "species" and sid not in kept and self._reads(change) & moved
        ]

    def _first_point(self, i: int, exp, changes: list[_Change], *, fresh: bool) -> list[_Change]:
        """The swept parameter at its first value, with what follows it, for a
        sweep whose points go on from each other, after *changes*: a species
        that is still an expression of it there (its seed, from the initial
        state, which is *fresh*; or one the changes set it to) takes its value
        from the first point in BNG2.pl, and SED-ML sets the species once,
        before the sweep (issue #1733). None where no species follows it."""
        pid = self.targets.param_id(exp.scan_parameter or "")
        first = _first_value(exp)
        if pid is None or first is None:
            return []
        following = self._following.get(i, [])
        moved = {pid} | {f.sid for f in following}
        _, live, kept = self._sequence(changes, fresh=fresh)
        if not any(self._reads(c) & moved for c in live) and not (
            fresh and self._seeds(moved, kept)
        ):
            return []
        start = _Change("param", pid, first, None, False, 1.0, "the sweep's first point")
        return [start, *following]

    def _followers(self, moved: set[str], params: dict[str, _Change]) -> list[_Change]:
        """The parameters that follow others (:meth:`_definitions`) that read a
        parameter in *moved*, or one another of them reads, each after those
        it reads: what BNG2.pl's parameters hold once *moved* move. One in
        *moved* that reads another of them is among them, set again after it."""
        affected: dict[str, _Change] = {}
        definitions = self._definitions(params)
        grew = True
        while grew:
            grew = False
            for pid, change in definitions.items():
                # One of *moved* that reads another of them is set again after
                # it: a reset applies every parameter at once.
                if pid in affected:
                    continue
                if self._reads(change) & (moved | affected.keys()):
                    affected[pid] = change
                    grew = True
        out: list[_Change] = []
        done: set[str] = set()
        while len(done) < len(affected):
            ready = [
                pid
                for pid, change in affected.items()
                if pid not in done and not (self._reads(change) & (affected.keys() - done - {pid}))
            ]
            if not ready:  # an expression that reads itself, which BNG2.pl refuses
                ready = [pid for pid in affected if pid not in done]
            for pid in ready:
                out.append(affected[pid])
                done.add(pid)
        return out

    def _sequence(
        self, changes: list[_Change], *, fresh: bool, say: bool = False
    ) -> tuple[list[_Change], list[_Change], frozenset[str]]:
        """*changes* in the order SED-ML applies them for one step, which
        evaluates each where it stands; the species changes among them that
        are still expressions at the run that follows; and the species that no
        longer follow their seed.

        BNG2.pl keeps a species set to an expression as that expression until
        a run, and so the seed of one that names a parameter, from the initial
        state (*fresh*). So a species' last ``setConcentration`` to an
        expression goes after every parameter change. An ``addConcentration``
        to a species that is still an expression adds to the species' declared
        seed, at the parameters as they stand (BNGModel.pm, addConcentration:
        it evaluates ``$spec->Concentration``), and makes it a number. The
        species is set to that seed first, where the model does not hold it:
        a parameter the seed reads has moved, a change before this one set the
        species, or a run did (issue #1733). Where the seed is not known (an
        action before the network was generated changed the amount the SBML
        holds), the add is to what the species was set to, *said* here."""
        out: list[_Change] = []
        live: dict[str, _Change] = {}
        numbers: set[str] = set()
        moved: set[str] = set()
        touched: set[str] = set()
        for change in changes:
            if change.kind == "param":
                moved.add(change.sid)
                out.append(change)
                continue
            sid = change.sid
            if change.follows:
                live[sid] = change
                numbers.discard(sid)
                continue
            if change.add and sid not in numbers and (fresh or sid in live):
                was = live.get(sid)
                seed = self._assigned.get(sid)
                known = sid not in self._undeclared and None not in self._undeclared
                value = self.targets.species_values.get(sid) if known else None
                if seed is not None and known:
                    if not fresh or sid in touched or self._reads(seed) & moved:
                        out.append(seed)
                elif was is not None and value is not None:
                    if not fresh or sid in touched:
                        declared = f"the declared amount of {sid}"
                        out.append(_Change("species", sid, value, None, False, 1.0, declared))
                elif was is not None or seed is not None:
                    onto = f"{was.action} set" if was is not None else "the SBML holds"
                    note = (
                        f"SED-ML's standard elements add {change.action} to the amount "
                        f"{onto}, where BNG2.pl adds to the species' declared amount, "
                        "which an action before the network was generated changed; the "
                        "bngsim annotation holds the protocol"
                    )
                    if say and note not in self._noted:
                        self._noted.add(note)
                        self._say(note)
                    if was is not None:
                        out.append(was)
            live.pop(sid, None)
            numbers.add(sid)
            touched.add(sid)
            out.append(change)
        out.extend(live.values())
        return out, list(live.values()), frozenset(numbers | live.keys())

    def _moved(self, changes: list[_Change], params: dict[str, _Change]) -> list[_Change]:
        """*changes*, then the expressions in *params* they move."""
        moved = {c.sid for c in changes if c.kind == "param"}
        return list(changes) + (self._followers(moved, params) if moved else [])

    # ── experiments ────────────────────────────────────────────────────────
    def simulation(self, i: int, exp) -> str:
        sim_id = f"sim_{i}"
        utc = ET.SubElement(
            self.sims,
            _q(_SEDML_NS, "uniformTimeCourse"),
            _utc_attrs(sim_id, exp.t_span[0], exp.output_start, exp.t_span[1], _steps(exp)),
        )
        kisao, params = _protocol_algorithm(exp, i)
        algo = ET.SubElement(utc, _q(_SEDML_NS, "algorithm"), {"kisaoID": kisao})
        _algorithm_param_elems(algo, params)
        return sim_id

    def derived_model(self, i: int, changes: list[_Change]) -> str:
        # A parameter the SBML computes, whose last change puts it back as the
        # SBML gives it (a resetParameters to the network's expression), is
        # what its initialAssignment computes from the model as changed: it
        # takes no change, and keeps the assignment.
        # So is one whose last change sets it to what its initialAssignment
        # computes (issue #1733); and the model is loaded after its changes,
        # so the assignment computes it from them.
        changes, _, _ = self._sequence(changes, fresh=True, say=True)
        last = {c.sid: c for c in changes}
        back = {
            sid
            for sid, c in last.items()
            if (c.restores or c.initial is not None) and sid in self.targets.initially_assigned
        }
        # But an expression among the changes reads the value the element
        # holds, which the assignment has not computed yet: a parameter it
        # reads, or one that one reads, is set to what its assignment computes
        # from the changes before it, and keeps the assignment.
        # A change to such a parameter before its last one stays where an
        # expression between the two reads the value it set: after
        # setParameter("k2",7), addConcentration("A()","k2") adds 7, though a
        # resetParameters after it puts k2 back.
        last_at = {c.sid: k for k, c in enumerate(changes)}

        def read_before_the_last(k: int) -> bool:
            between = changes[k + 1 : last_at[changes[k].sid]]
            return any(c.value is None and changes[k].sid in self._reads(c) for c in between)

        kept = [
            c
            for k, c in enumerate(changes)
            if (c.sid not in back or read_before_the_last(k))
            and (c.initial is None or c.kind == "species")
        ]
        needed: set[str] = set()
        for c in kept:
            if c.value is None:
                needed |= self._reads(c)
        grew = True
        while grew:
            grew = False
            for c in changes:
                if c.kind == "param" and c.initial is not None and c.sid in needed:
                    more = self._reads(c) - needed
                    if more:
                        needed |= more
                        grew = True
        # A parameter with a change kept under its assignment takes its last
        # change too, which says what the assignment computes: a reader of the
        # changes alone does not know the assignment is there, and would leave
        # the parameter at the value the kept change set.
        said_again = {c.sid for c in kept if c.sid in back}
        changes = [
            c
            for k, c in enumerate(changes)
            if c in kept
            or (c.kind == "param" and c.initial is not None and c.sid in needed)
            or (c.sid in said_again and k == last_at[c.sid])
        ]
        if not changes:
            return "model0"
        model_ref = f"model_{i}"
        dm = ET.SubElement(
            self.models,
            _q(_SEDML_NS, "model"),
            {"id": model_ref, "language": _SBML_LANG, "source": "model0"},
        )
        loc = ET.SubElement(dm, _q(_SEDML_NS, "listOfChanges"))
        # The SBML computes a derived parameter, or a species whose seed names
        # a parameter, by an initialAssignment, which overrides a change to the
        # value the element holds: setParameter("k2",7) on `k2 k1*3` ran k2 = 3.
        # BNG2.pl's action replaces that expression, and so does removing it.
        for sid in dict.fromkeys(c.sid for c in kept):
            if sid in self.targets.initially_assigned and sid not in back:
                ET.SubElement(
                    loc,
                    _q(_SEDML_NS, "removeXML"),
                    {"target": _initial_assignment_path(sid)},
                )
        for change in changes:
            self._model_change(loc, change, model_ref)
        return model_ref

    def sweep(
        self, rt_id: str, i: int, exp, model_ref: str, task_id: str, *, reset: bool, fresh: bool
    ) -> None:
        """A scan or bifurcate as a repeatedTask over its values: every point
        from the start state when *reset*, else each from where the last left
        it. A bifurcate is its sweep up and then back down, in one range. A
        parameter set to an expression of the swept one is set at each point
        too, from the range (issue #1682), and so is one the SBML computes from
        it by an initialAssignment, and, at each point of a *fresh* sweep that
        resets, a species whose seed the SBML computes from it, or that the
        protocol set to an expression of it (issue #1733).
        A *fresh* sweep that goes on from point to point starts from a model
        that :meth:`_first_point` has set."""
        values = _sweep_values(exp)
        pid = self.targets.param_id(exp.scan_parameter or "") or exp.scan_parameter
        rng_id = f"range_{rt_id}"
        rt = ET.SubElement(
            self.tasks,
            _q(_SEDML_NS, "repeatedTask"),
            {"id": rt_id, "resetModel": _xsd_bool(reset), "range": rng_id},
        )
        lor = ET.SubElement(rt, _q(_SEDML_NS, "listOfRanges"))
        if values is not None:
            vr = ET.SubElement(lor, _q(_SEDML_NS, "vectorRange"), {"id": rng_id})
            for v in values:
                ET.SubElement(vr, _q(_SEDML_NS, "value")).text = _num(v)
        else:
            ET.SubElement(
                lor,
                _q(_SEDML_NS, "uniformRange"),
                {
                    "id": rng_id,
                    "start": _num(exp.scan_min or 0.0),
                    "end": _num(exp.scan_max if exp.scan_max is not None else 1.0),
                    # GH #575 — numberOfPoints counts the INTERVALS, as it does
                    # on the uniformTimeCourse: a consumer expands a
                    # uniformRange to numberOfPoints+1 values. bngsim's
                    # scan_points is the inclusive point count (the linspace /
                    # BNG2.pl n_scan_pts convention, delta = (max-min)/(n-1)),
                    # so writing it verbatim handed every standards-compliant
                    # reader one point more than the scan bngsim ran.
                    "numberOfPoints": str(max(int(exp.scan_points or 1) - 1, 0)),
                    "type": "log" if exp.scan_log else "linear",
                },
            )
        loc = ET.SubElement(rt, _q(_SEDML_NS, "listOfChanges"))
        sv = ET.SubElement(
            loc,
            _q(_SEDML_NS, "setValue"),
            {
                "target": self.targets.param_path(pid),
                "range": rng_id,
                "modelReference": model_ref,
            },
        )
        ET.SubElement(sv, _q(_MATHML_NS, "math")).append(_ci(rng_id))
        # A sweep from the initial state (*fresh*: its first point, or every
        # point when it resets) also moves a species whose seed names the
        # parameter, in BNG2.pl: no run has made it a number yet.
        replaced = self._replaced_at.get(i, frozenset())
        self._set_as_it_runs(pid, species=fresh, replaced=replaced)
        for follower in self._following.get(i, []):
            self._set_as_it_runs(follower.sid, species=fresh, replaced=replaced)
            fv = ET.SubElement(
                loc,
                _q(_SEDML_NS, "setValue"),
                {
                    "target": self.targets.param_path(follower.sid),
                    "range": rng_id,
                    "modelReference": model_ref,
                },
            )
            self._math(fv, follower, model_ref, ranged={pid: rng_id})
        if fresh and reset:
            moved = {pid} | {f.sid for f in self._following.get(i, [])}
            # A species the protocol set to an expression is that expression
            # at each point too, and one whose seed the SBML computes.
            exprs = [c for c in self._live.get(i, []) if self._reads(c) & moved]
            for seed in exprs + self._seeds(moved, replaced):
                sv = ET.SubElement(
                    loc,
                    _q(_SEDML_NS, "setValue"),
                    {
                        "target": self.targets.species_path(seed.sid),
                        "range": rng_id,
                        "modelReference": model_ref,
                    },
                )
                self._math(sv, seed, model_ref, ranged={pid: rng_id})
        self._subtasks(rt, [task_id])

    def run(self, i: int, exp) -> str:
        """Experiment *i* on the base model, as a step of a chain: a task, or
        for a sweep that continues, its repeatedTask."""
        if i not in self._runs:
            self._runs[i] = f"run_{i}"
            ET.SubElement(
                self.tasks,
                _q(_SEDML_NS, "task"),
                {"id": f"run_{i}", "modelReference": "model0", "simulationReference": f"sim_{i}"},
            )
        if not _sweeps(exp):
            return self._runs[i]
        if i not in self._sweeps:
            self._sweeps[i] = f"sweep_{i}"
            self.sweep(f"sweep_{i}", i, exp, "model0", self._runs[i], reset=False, fresh=False)
        return self._sweeps[i]

    def step(
        self, changes: list[_Change], body: str, replaced: frozenset[str], *, fresh: bool = False
    ) -> str:
        """*body* after *changes*: a one-iteration repeatedTask that sets them
        and runs it, without resetting the model (or *body*, with none).
        *replaced* are the parameters the protocol had replaced then. A
        *fresh* step runs from the initial state, before any run, where a
        species whose seed the SBML computes from a parameter follows it, in
        BNG2.pl (issue #1733)."""
        changes, _, kept = self._sequence(changes, fresh=fresh, say=True)
        if fresh:
            moved = {c.sid for c in changes if c.kind == "param"}
            changes = changes + self._seeds(moved, kept)
        if not changes:
            return body
        rt = ET.SubElement(
            self.tasks,
            _q(_SEDML_NS, "repeatedTask"),
            {"id": self._id("step"), "resetModel": "false"},
        )
        self._once(rt)
        loc = ET.SubElement(rt, _q(_SEDML_NS, "listOfChanges"))
        for change in changes:
            self._set_value(loc, change, "model0", replaced, fresh=fresh)
        self._subtasks(rt, [body])
        return rt.get("id") or ""

    def chain(self, i: int, exp, lead: str, step: str) -> str:
        """Experiment *i* run from where *lead* left the model: a one-iteration
        repeatedTask that resets the base model once, then runs *lead* and the
        experiment's own step in order, on one model state."""
        rt = ET.SubElement(
            self.tasks, _q(_SEDML_NS, "repeatedTask"), {"id": f"chain_{i}", "resetModel": "true"}
        )
        self._once(rt)
        self._subtasks(rt, [lead, step])
        self._experiments.append((i, exp, f"chain_{i}"))
        return f"chain_{i}"

    def standalone(self, i: int, exp, changes: list[_Change]) -> None:
        """Experiment *i* from the model's initial state with *changes*."""
        if _scans(exp):
            # A species the changes set no longer follows its seed's parameter.
            _, self._live[i], set_here = self._sequence(changes, fresh=True)
            self._replaced_at[i] = self._replaced_at.get(i, frozenset()) | set_here
            if _sweeps(exp):
                changes = list(changes) + self._first_point(i, exp, changes, fresh=True)
        model_ref = self.derived_model(i, changes)
        task_id = f"task_{i}"
        ET.SubElement(
            self.tasks,
            _q(_SEDML_NS, "task"),
            {"id": task_id, "modelReference": model_ref, "simulationReference": f"sim_{i}"},
        )
        if _scans(exp):
            resets = exp.kind == "scan" and exp.reset_between
            self.sweep(f"rtask_{i}", i, exp, model_ref, task_id, reset=resets, fresh=True)
            task_id = f"rtask_{i}"
        self._experiments.append((i, exp, task_id))

    # ── what each experiment reports (issue #1686) ─────────────────────────
    def _output_selectors(self) -> list[str]:
        """What each experiment reports besides the time: the model's
        observables, as BNG2.pl's ``.gdat`` does, else its species."""
        model = self.targets.model
        if model is not None:
            observables = list(getattr(model, "observable_names", []))
            if observables:
                return [f"observable:{n}" for n in observables]
            return [f"species:{n}" for n in model.species_names]
        if self.targets.read:
            return [f"species:{sid}" for sid in dict.fromkeys(self.targets.species.values())]
        return []

    def outputs(self) -> None:
        """A data generator per quantity each experiment reports, over the task
        that runs it, and a report per experiment: a tool that runs only what
        an output reads runs the experiments, and the parts of one are not
        mistaken for one (issue #1686). An experiment that continues earlier
        runs is a chain that replays them, and SED-ML reads a repeatedTask's
        results as all of its subtasks', so its report holds the replayed runs'
        rows first and its own last; the report's name says so."""
        from bngsim._protocol_run import _describe

        if not self._experiments:
            return
        # With neither the model nor its SBML, a report names the time alone,
        # and still says which task runs each experiment.
        selectors = self._output_selectors()
        targeted = [(sel, _selector_target(sel, self.targets)) for sel in selectors]
        untargeted = [sel for sel, attrs in targeted if not attrs]
        if untargeted:
            self._say(
                f"SED-ML's variables for {untargeted} have no target: "
                + (
                    "the SBML has no element they name"
                    if self.targets.read
                    else "naming their SBML elements needs the SBML (pass sbml=)"
                )
                + "; a standards reader cannot read them, and bngsim reads their names"
            )
        dgs = ET.SubElement(self.sed, _q(_SEDML_NS, "listOfDataGenerators"))
        outs = ET.SubElement(self.sed, _q(_SEDML_NS, "listOfOutputs"))
        rows = [("time", {"symbol": _TIME_SYMBOL}), *targeted]
        for i, exp, task in self._experiments:
            name = f"experiment {i}: {_describe(exp)}"
            if task.startswith("chain_"):
                name += "; the earlier runs it continues come first, its own rows last"
            report = ET.SubElement(
                outs, _q(_SEDML_NS, "report"), {"id": f"report_{i}", "name": name}
            )
            lds = ET.SubElement(report, _q(_SEDML_NS, "listOfDataSets"))
            for k, (label, attrs) in enumerate(rows):
                dg_id = f"dg_{i}_{k}"
                _data_generator(dgs, dg_id, label, var_attrs=attrs, task=task)
                ET.SubElement(
                    lds,
                    _q(_SEDML_NS, "dataSet"),
                    {"id": f"ds_{i}_{k}", "label": label, "dataReference": dg_id},
                )

    # ── the walk ───────────────────────────────────────────────────────────
    def write(self) -> str:
        from bngsim._protocol_run import _DEFAULT_LABEL
        from bngsim.convert._protocol import (
            Experiment,
            StateChange,
            _evenly_spaced,
            _saved_before_generation,
        )

        protocol = self.protocol
        generation = protocol.generation_step or 0
        # The state BNG2.pl holds, in what SED-ML can say: `lead` is a task that
        # runs the base model from its initial state to where the last run left
        # it (None: no run since that state), as ("first", i, changes) until a
        # chain needs it written; `pending` are the changes since; `params` the
        # parameters set since the network was generated, by SBML id.
        lead: object = None
        pending: list[_Change] = []
        params: dict[str, _Change] = {}
        saved_conc: dict[str, tuple[object, list[_Change], dict[str, _Change]]] = {}
        saved_params: dict[str, dict[str, _Change]] = {}
        # Steps before the network was generated are in it, so in the SBML. A
        # save among them holds the network, with the actions after the save
        # that changed what it saved undone by the actions that made it, where
        # the protocol has those (issue #1668); `unknown` are the saves it does
        # not, whose values are the model's as declared.
        before = _saved_before_generation(protocol.steps[:generation])
        self._undeclared = {
            self.targets.species_id(step.target or "") if self.targets.read else None
            for step in protocol.steps[:generation]
            if isinstance(step, StateChange)
            and step.kind in ("set_concentration", "add_concentration")
        }
        unknown: set[tuple[str, str]] = set()
        for label, back in before.conc.items():
            changes = self._changes(back)
            if changes is None:
                unknown.add(("concentrations", label))
            else:
                saved_conc[label] = (None, changes, {})
        for label, back in before.params.items():
            changes = self._changes(back)
            if changes is None:
                unknown.add(("parameters", label))
            else:
                saved_params[label] = {c.sid: c for c in changes}
        i = -1
        for k, step in enumerate(protocol.steps):
            if isinstance(step, Experiment):
                # Numbered by position: one Experiment object held twice
                # (combine_protocols([p, p])) is two experiments (issue #1677).
                i += 1
            if k < generation:
                continue
            if isinstance(step, StateChange):
                label = step.label or _DEFAULT_LABEL
                if step.kind in ("set_concentration", "add_concentration", "set_parameter"):
                    change = self.change(step)
                    if change is not None:
                        if change.kind == "param":
                            params.pop(change.sid, None)
                            params[change.sid] = change
                        pending.extend(self._moved([change], params))
                elif step.kind == "save_concentrations":
                    saved_conc[label] = (lead, list(pending), dict(params))
                elif step.kind == "save_parameters":
                    saved_params[label] = dict(params)
                elif step.kind == "reset_concentrations":
                    if label in saved_conc:
                        lead, then, params_then = saved_conc[label]
                        # The concentrations go back; the parameters stay as set.
                        pending = list(then) + self._moved(
                            self._params_since(params_then, params), params
                        )
                    else:
                        if ("concentrations", label) in unknown or (
                            label == _DEFAULT_LABEL and not before.declared
                        ):
                            self._say(
                                "SED-ML's standard elements cannot go back to the "
                                f"concentrations of {_state_change_text(step)}: what it goes "
                                "back to is the model's declared amounts, which an action "
                                "before the network was generated changed, and the network "
                                "holds only the changed ones; the standard elements go back "
                                "to the network as generated, and run_protocol refuses it"
                            )
                        lead, pending = None, self._moved(list(params.values()), params)
                elif step.kind == "reset_parameters":
                    if ("parameters", label) in unknown:
                        self._say(
                            "SED-ML's standard elements cannot go back to the parameters "
                            f"of {_state_change_text(step)}: what it goes back to is a "
                            "value as declared, which an action before the network was "
                            "generated changed, and the network holds only the changed "
                            "one; the standard elements go back to the network as "
                            "generated, and run_protocol refuses it"
                        )
                    restored = saved_params.get(label, {})
                    moved = self._params_since(params, restored)
                    params = dict(restored)
                    pending += self._moved(moved, params)
                continue
            if not isinstance(step, Experiment):  # pragma: no cover — two step types only
                continue
            exp = step
            if exp.sample_times is not None and not _evenly_spaced(list(exp.sample_times)):
                self._say(
                    f"SED-ML holds experiment {i}'s uneven output times "
                    f"{list(exp.sample_times)} as {exp.n_points} even points; the exact "
                    "times are in the bngsim annotation"
                )
            self.simulation(i, exp)
            resets = exp.kind == "scan" and exp.reset_between and _scans(exp)
            pid = self.targets.param_id(exp.scan_parameter) if exp.scan_parameter else None
            if _scans(exp):
                # Set at each point with the swept parameter, from the range.
                self._following[i] = self._followers({pid or ""}, params)
            replaced = frozenset(pid_ for pid_, c in params.items() if not c.restores)
            self._replaced_at[i] = replaced
            if lead is None:
                self.standalone(i, exp, pending)
                if not resets:
                    lead, pending = ("first", i, list(pending), replaced), []
            elif resets:
                # Every point of BNG2.pl's scan starts where the last run left
                # the model; a repeatedTask's reset goes to the initial state.
                self._say(
                    f"experiment {i} is a parameter_scan with reset_conc=>1 that BNG2.pl "
                    "starts where the run before it left the model, and SED-ML resets a "
                    "repeatedTask's model to its initial state, not to that one; the "
                    "standard elements start each point from the initial state, and the "
                    "bngsim annotation holds the protocol"
                )
                self.standalone(i, exp, self._moved(list(params.values()), params))
            else:
                if _sweeps(exp):
                    # A species set to an expression since the last run reads
                    # the sweep's first point.
                    pending = pending + self._first_point(i, exp, pending, fresh=False)
                self.chain(
                    i, exp, self._lead(lead), self.step(pending, self.run(i, exp), replaced)
                )
                lead, pending = f"chain_{i}", []
            if exp.is_scan and exp.scan_parameter:
                last = _last_value(exp)
                if last is not None and pid is not None:
                    # The parameter keeps the value the sweep ended on (BNGAction.pm).
                    left = _Change("param", pid, last, None, False, 1.0, "the scan")
                    params.pop(pid, None)
                    params[pid] = left
                    if resets:
                        pending.extend(self._moved([left], params))
        self.outputs()
        ET.indent(self.sed, space="  ")
        text = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
            self.sed, encoding="unicode"
        )
        return text if text.endswith("\n") else text + "\n"

    def _changes(self, steps) -> list[_Change] | None:
        """The changes *steps* make, or ``None`` where any cannot be written."""
        if steps is None:
            return None
        out: list[_Change] = []
        for step in steps:
            change = self.change(step)
            if change is None:
                return None
            out.append(change)
        return out

    def _lead(self, lead) -> str:
        """The task that runs the model to where *lead* left it. The first run
        since the initial state replays on the base model with its changes as
        setValues, from that state (issue #1733)."""
        if isinstance(lead, tuple):
            _, i, changes, replaced = lead
            exp = self.protocol.experiments[i]
            if _sweeps(exp):
                changes = list(changes) + self._first_point(i, exp, changes, fresh=True)
            return self.step(changes, self.run(i, exp), replaced, fresh=True)
        return str(lead)

    def _params_since(self, before: dict[str, _Change], now: dict[str, _Change]) -> list[_Change]:
        """The parameter changes that take *before* to *now*."""
        out = []
        for pid in dict.fromkeys([*before, *now]):
            if pid in now:
                if before.get(pid) != now[pid]:
                    out.append(now[pid])
            else:
                back = self.original(pid)
                if back is not None:
                    out.append(back)
        return out


def _initial_assignment_path(sid: str) -> str:
    return (
        "/sbml:sbml/sbml:model/sbml:listOfInitialAssignments/"
        f"sbml:initialAssignment[@symbol='{sid}']"
    )


def _state_change_text(step) -> str:
    from bngsim.convert._protocol import _state_change_action

    return _state_change_action(step)


def _shareable(protocol):
    """*protocol* with its source named by file name alone (issue #1678)."""
    source = protocol.source
    if not source:
        return protocol
    names = " + ".join(re.split(r"[\\/]", part.strip())[-1] for part in source.split(" + "))
    return protocol if names == source else dataclasses.replace(protocol, source=names)


def _ci(name: str) -> ET.Element:
    el = ET.Element(_q(_MATHML_NS, "ci"))
    el.text = name
    return el


def _cn(value: float) -> ET.Element:
    el = ET.Element(_q(_MATHML_NS, "cn"))
    el.text = _num(value)
    return el


def _apply(op: str, args: list[ET.Element]) -> ET.Element:
    el = ET.Element(_q(_MATHML_NS, "apply"))
    ET.SubElement(el, _q(_MATHML_NS, op))
    el.extend(args)
    return el


# SBML L3V2's MathML operators that SED-ML L1V3's, which is SBML Level 2's,
# does not have, besides min and max (:func:`_without_min_max`).
_NOT_IN_SEDML_MATH = ("rem", "quotient", "implies")


def _without_min_max(node: ET.Element) -> ET.Element:
    """*node* with each ``min`` and ``max`` as the ``piecewise`` it is: SED-ML
    L1V3's MathML has neither, and its schema refuses them (issue #1734)."""
    for k, child in enumerate(list(node)):
        node[k] = _without_min_max(child)
    op = _local(node[0].tag) if _local(node.tag) == "apply" and len(node) > 1 else None
    if op not in ("min", "max"):
        return node
    out = node[1]
    for arg in list(node)[2:]:
        first = _apply("lt" if op == "min" else "gt", [copy.deepcopy(out), copy.deepcopy(arg)])
        piecewise = ET.Element(_q(_MATHML_NS, "piecewise"))
        ET.SubElement(piecewise, _q(_MATHML_NS, "piece")).extend([out, first])
        ET.SubElement(piecewise, _q(_MATHML_NS, "otherwise")).append(arg)
        out = piecewise
    return out


def _if_for_piecewise(node: ET.Element) -> ET.Element:
    """*node* with each ``piecewise`` that has an ``otherwise`` as a call of
    ``if(condition, value, otherwise)``, which is how BNGL, and so a replay,
    writes it."""
    for k, child in enumerate(list(node)):
        node[k] = _if_for_piecewise(child)
    pieces = [c for c in node if _local(c.tag) == "piece" and len(c) == 2]
    otherwise = [c for c in node if _local(c.tag) == "otherwise" and len(c) == 1]
    if _local(node.tag) != "piecewise" or len(otherwise) != 1 or len(pieces) + 1 != len(node):
        return node
    out = otherwise[0][0]
    for piece in reversed(pieces):
        call = ET.Element(_q(_MATHML_NS, "apply"))
        call.extend([_ci("if"), piece[1], piece[0], out])
        out = call
    return out


def _scaled(expr: ET.Element, scale: float) -> ET.Element:
    return expr if scale == 1.0 else _apply("times", [expr, _cn(scale)])


def _steps(exp) -> int:
    return max(int(exp.n_points) - 1, 0)


def _scans(exp) -> bool:
    return bool(exp.is_scan and exp.scan_parameter and exp.scan_points)


def _sweeps(exp) -> bool:
    """A scan whose points continue from each other: one without reset_conc,
    or a bifurcate."""
    return _scans(exp) and not (exp.kind == "scan" and exp.reset_between)


def _sweep_values(exp) -> list[float] | None:
    """The values a sweep runs, in order: ``par_scan_vals`` as given, a
    bifurcate's sweep up and then back down, or ``None`` for a plain range,
    which a ``uniformRange`` states."""
    from bngsim._protocol_run import _scan_values

    try:
        if exp.is_bifurcation:
            return _scan_values(exp) + _scan_values(exp, backward=True)
        if exp.scan_values is not None:
            return list(exp.scan_values)
    except ConversionError:
        return None
    return None


def _first_value(exp) -> float | None:
    from bngsim._protocol_run import _scan_values

    try:
        return float(_scan_values(exp)[0])
    except (ConversionError, IndexError):
        return None


def _last_value(exp) -> float | None:
    from bngsim._protocol_run import _scan_values

    try:
        return float(_scan_values(exp, backward=exp.is_bifurcation)[-1])
    except (ConversionError, IndexError):
        return None


def write_sedml_protocol(
    protocol,
    out_path: str | Path | None = None,
    *,
    model_source: str = "model.xml",
    sbml: str | Path | None = None,
    model: Model | None = None,
) -> str:
    """Serialize a :class:`~bngsim.convert.ProtocolSpec` to a SED-ML document.

    Every experiment gets a ``<uniformTimeCourse>``. One that BNG2.pl starts
    from the model's initial state, with the ``setConcentration``,
    ``addConcentration`` and ``setParameter`` actions before it, is a ``<task>``
    on a derived model that makes those changes (a ``<repeatedTask>`` over its
    range for a ``parameter_scan``; a bifurcate is its sweep up and back down,
    each point continuing from the last). One that BNG2.pl starts where an
    earlier run left the model is a ``<repeatedTask>`` that resets the model
    once and replays, as ordered subtasks on one model state, the runs since
    that state, with the actions between them as ``<setValue>`` changes: the
    carried state as SED-ML states it. A bngsim ``<annotation>`` carries the
    verbatim ``ProtocolSpec`` JSON, so :func:`read_sedml_protocol` recovers it
    exactly. Build/IO directives were already dropped at parse time.

    What SED-ML cannot say is named in a :class:`~bngsim.ConversionWarning`: a
    ``parameter_scan`` with ``reset_conc=>1`` that BNG2.pl starts from a
    carried state (a ``repeatedTask`` resets to the initial state), a method
    with no KiSAO term, and an action whose target or value the SBML cannot
    name.

    Parameters
    ----------
    protocol : ProtocolSpec
        The parsed protocol to emit. Must carry at least one experiment.
    out_path : str | Path | None
        If given, the document is also written here.
    model_source : str
        The ``<model source="…">`` reference (e.g. the SBML inside the archive).
    sbml : str | Path | None
        The SBML that *model_source* names, as text or a path, which every
        target is written against: its namespace, and the ids bngsim's SBML
        writer gave each species and parameter. Defaults to *model_source*,
        when that is a readable file. Without it, a species action cannot be
        written, and is left to the annotation with a warning.
    model : Model | None
        The model the SBML was written from, to resolve a BNGL species
        pattern (``@c:A()``, components in another order) as BNG2.pl does.

    Returns
    -------
    str
        The SED-ML document text.
    """
    if not protocol.experiments:
        raise ConversionError(
            "cannot emit SED-ML for an empty protocol (no simulate/parameter_scan "
            "experiments were parsed from the .bngl)"
        )
    targets = _SbmlTargets(sbml if sbml is not None else model_source, model)
    text = _ProtocolWriter(protocol, targets, model_source).write()
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


def _xsd_bool(b: bool) -> str:
    return "true" if b else "false"


def read_sedml_protocol(source: str | Path, *, sbml: str | Path | None = None):
    """Parse a SED-ML document into a :class:`~bngsim.convert.ProtocolSpec`.

    When the document carries the bngsim ``<annotation>`` that
    :func:`write_sedml_protocol` writes, the original ``ProtocolSpec`` is
    recovered **exactly** from its verbatim JSON, which names species and
    parameters as the BNGL file does; given ``sbml``, they are named by the ids
    that SBML gives them, so the protocol runs on the model loaded from it
    (issue #1685). Otherwise the protocol is read
    from the standard elements, by task (issue #1662): each ``task`` is an
    experiment on its model, from that model's initial state, with the model's
    ``changeAttribute`` and ``computeChange`` changes as the steps before it; a
    ``repeatedTask`` that sweeps a parameter is a scan, and one that does not
    runs its subtasks in order on one model state, resetting it first where
    ``resetModel`` says so, with its ``setValue`` changes as the steps before
    them. Tasks are independent, so the state is reset between them; a chain
    that repeats the start of a longer chain, as :func:`write_sedml_protocol`
    writes a run that continues an earlier one, is read once, as the longer
    chain. A time course no task runs is not run. A sweep runs the values its
    ``setValue``'s math gives at each iteration, over the range it names
    (issue #1664). A change's math is evaluated once, where it applies, as
    SED-ML evaluates it; a ``setValue`` beside a sweep that reads the range
    the swept parameter is set to is that parameter's expression, which
    follows it through the scan. Beside a sweep that resets the model, one
    that sets a species to what the SBML's initialAssignment of it computes is
    the model's own seed, which a replay's scan follows, and one that sets a
    species from the range is that species set to an expression before the
    scan, as :func:`write_sedml_protocol` writes a seed, or an amount set to
    an expression, that follows the swept parameter (issue #1733).

    A change the replay cannot apply (a compartment's size, an ``addXML``, a
    species change with no SBML to give its compartment's size, a ``setValue``
    beside a sweep that reads its range otherwise, a sweep's math over the
    model's values, a ``repeatedTask`` with more than one iteration and no
    sweep) is said, with a
    :class:`~bngsim.ConversionWarning`, and recorded in the spec's ``lossy``
    list, which :func:`bngsim.run_protocol` refuses unless ``allow_lossy``. A
    kinetic law's local parameter is named as the SBML loader names it,
    ``_lp_<reaction>_<id>``.

    Parameters
    ----------
    source : str | Path
        A path to a ``.sedml``/``.xml`` document, or the document text itself.
    sbml : str | Path | None
        The SBML the document's targets name. A species change is read through
        it: an ``initialAmount`` divided by the species' compartment size, and
        the species' own value by it where the species has only substance
        units, as :func:`write_sedml_protocol` wrote them. Without it such a
        change is lossy. An annotation's protocol is named by its ids.

    Returns
    -------
    ProtocolSpec
    """
    return _read_sedml_protocol(source, sbml=sbml, by_sbml_ids=sbml is not None)


def _read_sedml_protocol(source: str | Path, *, sbml: str | Path | None, by_sbml_ids: bool):
    """:func:`read_sedml_protocol`, with an annotation's protocol named by
    *sbml*'s ids only where *by_sbml_ids*: a protocol to run on the ``.net``
    an archive bundles keeps the BNGL names its species have there."""
    from bngsim.convert._protocol import ProtocolSpec

    text = _read_text(source)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ConversionError(f"not a parseable SED-ML/XML document: {exc}") from exc
    if _local(root.tag) not in ("sedML", "sedml"):
        raise ConversionError(
            f"root element is <{_local(root.tag)}>, not <sedML>: not a SED-ML document"
        )
    _warn_if_synthesized_default(root)

    # Exact path: the bngsim annotation carries the verbatim ProtocolSpec.
    for payload in root.iter():
        if _local(payload.tag) == "protocol" and payload.text and payload.text.strip():
            try:
                spec = ProtocolSpec.from_json(payload.text)
            except (ValueError, KeyError):
                break  # malformed annotation → fall through to reconstruction
            return _on_sbml_ids(spec, _SbmlTargets(sbml)) if by_sbml_ids and sbml else spec

    utcs = _findall_local(root, "uniformTimeCourse")
    if not utcs:
        raise ConversionError("no <uniformTimeCourse> found — not a runnable SED-ML protocol")
    listed = [e for e in root.iter() if _local(e.tag) in ("task", "repeatedTask")]
    if not listed:
        # No task to read: one experiment per time course, in document order,
        # each from the last one's end, as a document bngsim wrote before it
        # had tasks read back.
        scan_by_sim = _scan_targets(root)
        steps = [_experiment_of(utc, scan_by_sim.get(utc.get("id") or "")) for utc in utcs]
        return ProtocolSpec(steps=tuple(steps), source=None, dropped=())
    reader = _TaskReader(root, sbml)
    return reader.protocol(listed)


def _on_sbml_ids(spec, targets: _SbmlTargets):
    """*spec* with each species and parameter it names by the id the SBML
    gives it (issue #1685). The annotation names them as the BNGL file does,
    ``L(r)`` or ``@c:A()``, and a model loaded from the SBML has only its ids,
    ``L_r`` or ``c__A``: replayed on it, ``setConcentration("L(r)",50)``
    named no species. A name the SBML does not give stays as it is."""
    from bngsim.convert._protocol import Experiment, StateChange

    if not targets.read:
        return spec
    renamed = {n: pid for n, pid in targets.params.items() if n != pid and _SID.match(pid)}

    def param(name: str | None) -> str | None:
        return targets.param_id(name) or name if name else name

    def expression(text: str | None) -> str | None:
        if not text or not renamed:
            return text
        return re.sub(
            r"(?<![\w.])([A-Za-z_]\w*)(?!\w)", lambda m: renamed.get(m.group(1), m.group(1)), text
        )

    steps: list[object] = []
    for step in spec.steps:
        if isinstance(step, StateChange) and step.target:
            if step.kind in ("set_concentration", "add_concentration"):
                target = targets.species_id(step.target) or step.target
            elif step.kind == "set_parameter":
                target = param(step.target) or step.target
            else:
                target = step.target
            step = dataclasses.replace(step, target=target, value_expr=expression(step.value_expr))
        elif isinstance(step, Experiment) and step.scan_parameter:
            step = dataclasses.replace(step, scan_parameter=param(step.scan_parameter))
        steps.append(step)
    return dataclasses.replace(spec, steps=tuple(steps))


def _experiment_of(utc: ET.Element, sweep: dict | None):
    """The experiment a ``uniformTimeCourse`` runs: a scan where *sweep* says a
    ``repeatedTask`` sweeps a parameter over it, else a simulate."""
    from bngsim.convert._protocol import Experiment

    t0, output_start, t_end, n_points = _utc_grid(utc)
    method = "ode"
    extra: dict[str, float] = {}
    algo = _find(utc, "algorithm")
    if algo is not None:
        kid = _kisao_id(algo)
        mapped = _KISAO_TO_METHOD.get(kid)
        if mapped is None:
            # Issue #1500: as read_sedml says. A tau-leaping term read as an
            # ODE run became method=>"ode" in the actions block, unannounced.
            warnings.warn(
                f"{utc.get('id') or 'a uniformTimeCourse'}: unmapped algorithm KiSAO "
                f"term {kid!r}; the experiment is read as method 'ode'",
                ConversionWarning,
                stacklevel=_outside(),
            )
        else:
            method = mapped
        if method == "ode":
            # The run's tolerances, as BNG2.pl's simulate takes them.
            rtol, atol, _ = _algorithm_params(algo)
            if rtol is not None:
                extra["rtol"] = rtol
            if atol is not None:
                extra["atol"] = atol
        else:
            # A stochastic run's seed, which run_protocol reads from the
            # experiment as BNG2.pl's simulate takes it (issue #1517).
            seed = _algorithm_seed(algo)
            if seed is not None:
                extra["seed"] = seed
    if sweep is not None:
        return Experiment(
            kind="scan",
            method=method,
            t_span=(t0, t_end),
            output_start=output_start,
            extra=extra,
            n_points=n_points,
            scan_parameter=sweep.get("parameter"),
            scan_min=sweep.get("min"),
            scan_max=sweep.get("max"),
            scan_points=sweep.get("points"),
            scan_log=sweep.get("log", False),
            reset_between=sweep.get("reset", True),
            scan_values=sweep.get("values"),
        )
    return Experiment(
        kind="simulate",
        method=method,
        t_span=(t0, t_end),
        output_start=output_start,
        extra=extra,
        n_points=n_points,
    )


def _outside() -> int:
    """The stack level of the caller outside bngsim, for a warning raised from a
    helper of :func:`read_sedml` or :func:`read_sedml_protocol`."""
    from bngsim._simulator import _stacklevel_outside_bngsim

    return _stacklevel_outside_bngsim()


#: The label the reader saves the model's parameters under at the start of a
#: protocol of several tasks, and resets them to before each one.
_INITIAL_LABEL = "sedml_initial"

#: A ``repeatedTask``'s ``resetModel="true"``, where it falls in an expansion.
_RESET = object()


class _Restated(NamedTuple):
    """A step that sets an element to what the SBML's initialAssignment of it
    computes (:meth:`_TaskReader._start`)."""

    step: Any


class _PerPoint(NamedTuple):
    """A step a sweep makes at each of its points, beside the swept parameter."""

    step: Any


_TARGET_SPECIES = re.compile(
    r"""species\[\s*@id\s*=\s*['"]([^'"]+)['"]\s*\](?:/@(initialConcentration|initialAmount))?\s*$"""
)
_TARGET_LOCAL = re.compile(
    r"""reaction\[\s*@id\s*=\s*['"]([^'"]+)['"]\s*\].*(?:parameter|localParameter)\[\s*@id\s*=\s*['"]([^'"]+)['"]\s*\]"""
    r"""(?:/@value)?\s*$"""
)
_TARGET_PARAM = re.compile(r"""parameter\[\s*@id\s*=\s*['"]([^'"]+)['"]\s*\](?:/@value)?\s*$""")


def _target_name(target: str) -> tuple[str, str, str | None] | None:
    """What a SED-ML target names in the model bngsim loads from the SBML:
    ``("species", id, attribute)`` or ``("param", name, "value")``, or ``None``
    for anything else (a compartment's size, a reaction's attribute). A kinetic
    law's local parameter is the SBML loader's ``_lp_<reaction>_<id>``."""
    m = _TARGET_SPECIES.search(target)
    if m is not None:
        return ("species", m.group(1), m.group(2))
    if "reaction[" in target:
        m = _TARGET_LOCAL.search(target)
        return ("param", f"_lp_{m.group(1)}_{m.group(2)}", "value") if m is not None else None
    m = _TARGET_PARAM.search(target)
    return ("param", m.group(1), "value") if m is not None else None


def _number_or_none(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _model_chain(models: dict, model_id: str | None) -> list[ET.Element]:
    """A model and the models it derives from, nearest first: each ``source``
    that names another model's id (``#id``, or the id itself) is followed."""
    chain: list[ET.Element] = []
    seen: set[str] = set()
    mid = model_id
    while mid and mid not in seen and mid in models:
        seen.add(mid)
        elem = models[mid]
        chain.append(elem)
        source = (elem.get("source") or "").strip()
        mid = source[1:] if source.startswith("#") else source
    return chain


def _model_changes(model: ET.Element) -> list[ET.Element]:
    loc = _find(model, "listOfChanges")
    return list(loc) if loc is not None else []


class _TaskReader:
    """The standard elements of a document, read by task (issue #1662)."""

    def __init__(self, root: ET.Element, sbml: str | Path | None) -> None:
        self.root = root
        self._sbml = sbml
        self._targets: _SbmlTargets | None = None
        self.sims = {u.get("id"): u for u in _findall_local(root, "uniformTimeCourse")}
        self.models = {m.get("id"): m for m in _findall_local(root, "model")}
        self.tasks = {t.get("id"): t for t in _findall_local(root, "task")}
        self.rts = {r.get("id"): r for r in _findall_local(root, "repeatedTask")}
        self.sim_of_task = {
            tid: t.get("simulationReference") or "" for tid, t in self.tasks.items()
        }
        self.lossy: list[str] = []
        self.file: str | None = None  # the model file the first task runs, which every task runs

    @property
    def targets(self) -> _SbmlTargets | None:
        """The SBML's species and compartments, read once, at the first change
        that needs them."""
        if self._targets is None and self._sbml is not None:
            self._targets = _SbmlTargets(self._sbml)
        return self._targets

    # ── the document ───────────────────────────────────────────────────────
    def protocol(self, listed: list[ET.Element]):
        from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

        inner = {
            st.get("task") for rt in self.rts.values() for st in _findall_local(rt, "subTask")
        }
        # A task a data generator reads is a run of its own, inside a
        # repeatedTask or not; one no output reads runs inside the task that
        # names it (write_sedml_protocol's run_i, beside its chain_i).
        read = {var.get("taskReference") for var in _findall_local(self.root, "variable")}
        expansions: list[list[object]] = []
        for elem in listed:
            if elem.get("id") in inner and elem.get("id") not in read:
                continue  # run inside the repeatedTask that names it
            steps = self._task(elem) if _local(elem.tag) == "task" else self._repeated(elem, 0)
            while steps and steps[0] is _RESET:
                steps = steps[1:]  # an expansion starts from the initial state
            if any(isinstance(s, Experiment) for s in steps):
                expansions.append(steps)
                self._after_a_reset_scan(elem.get("id") or "a task", steps)
        # A chain that repeats the start of a longer one is the same runs from
        # the same state: write_sedml_protocol writes a run that continues an
        # earlier one as a repeatedTask that replays the runs before it, and
        # the task of the first run alone beside it.
        # Compared without the steps that say again, before the first run,
        # what an initialAssignment computes: write_sedml_protocol writes those
        # in the chain, on the model as it runs, and not in the first run's own
        # model, whose assignments compute them (issue #1733).
        starts = [self._start(steps) for steps in expansions]
        kept = [
            steps
            for steps, start in zip(expansions, starts, strict=True)
            if not any(
                len(other) > len(start) and other[: len(start)] == start for other in starts
            )
        ]
        reset = (
            StateChange(kind="reset_concentrations"),
            StateChange(kind="reset_parameters", label=_INITIAL_LABEL),
        )
        out: list[object] = []
        for i, steps in enumerate(kept):
            if i:
                out.extend(reset)
            for step in steps:
                if isinstance(step, (_Restated, _PerPoint)):
                    step = step.step
                out.extend(reset if step is _RESET else (step,))
        if any(isinstance(s, StateChange) and s.kind == "reset_parameters" for s in out):
            out.insert(0, StateChange(kind="save_parameters", label=_INITIAL_LABEL))
        return ProtocolSpec(steps=tuple(out), source=None, dropped=(), lossy=tuple(self.lossy))

    def _start(self, steps: list[object]) -> list[object]:
        """*steps* without those before its first run that set an element to
        what the SBML's initialAssignment of it computes, where a replay,
        whose model computes it, holds the same without them: nothing else
        sets the element before that run, and whatever sets something the
        assignment reads is followed by such a step again, before any other
        step and before the run. Else *steps* as they are."""
        from bngsim.convert._protocol import Experiment

        first = next((k for k, s in enumerate(steps) if isinstance(s, Experiment)), len(steps))
        head: list[Any] = steps[:first]
        if any(s is _RESET for s in head):
            return steps
        # A step that says an assignment again, where the next step that is
        # not one is to the same element: nothing between the two read it.
        superseded: set[int] = set()
        for k, s in enumerate(head):
            if isinstance(s, _Restated):
                after = next(
                    (o for o in head[k:] if not isinstance(o, (_Restated, _PerPoint))), None
                )
                if after is not None and after.target == s.step.target:
                    superseded.add(k)
        set_otherwise = {s.target for s in head if not isinstance(s, (_Restated, _PerPoint))}
        replaced: set[str] = set()  # set to something else so far
        restated: set[str] = set()
        stale: set[str] = set()  # restated, and behind what they read since
        for k, s in enumerate(head):
            if isinstance(s, _PerPoint) or k in superseded:
                continue  # a _PerPoint is set with the swept parameter, as the run goes
            name = s.step.target if isinstance(s, _Restated) else s.target
            if isinstance(s, _Restated) and name not in set_otherwise:
                restated.add(name)
                stale.discard(name)
                if self._assignment_reads(name, replaced) & stale:
                    stale.add(name)  # computed from a value that is behind
                continue
            if stale:
                return steps
            replaced.add(name)
            stale |= {r for r in restated if name in self._assignment_reads(r, replaced)}
        if stale:
            return steps
        return [
            s
            for k, s in enumerate(head)
            if k not in superseded and not (isinstance(s, _Restated) and s.step.target in restated)
        ] + steps[first:]

    def _assignment_reads(self, symbol: str, replaced: set[str]) -> set[str]:
        """Everything the SBML's initialAssignment of *symbol* reads, through
        the assignments of what it reads that are not *replaced*."""
        targets = self.targets
        assert targets is not None  # a step is marked as restated from the SBML
        reads: set[str] = set()
        todo = [symbol]
        while todo:
            symbol = todo.pop()
            for name, symbols in targets.assignment_reads.items():
                if symbol in symbols and name not in reads:
                    reads.add(name)
                    if name not in replaced:
                        todo.append(name)
        return reads

    def _after_a_reset_scan(self, tid: str, steps: list[object]) -> None:
        """Say where a run in *steps* starts from the state a sweep that resets
        at each point left, at any depth of chains: a replay's scan puts its
        start state back when it ends, as BNG2.pl's does, and SED-ML leaves
        the last point's state for what runs after it."""
        from bngsim.convert._protocol import Experiment

        after = False
        for step in steps:
            if step is _RESET:
                after = False
            elif isinstance(step, Experiment):
                if after:
                    self._unread(
                        f"{tid} runs a sweep that resets the model at each point and then "
                        "more from where its last point left the model, which a replay's "
                        "scan does not leave"
                    )
                    return
                after = step.kind == "scan" and step.reset_between

    # ── tasks ──────────────────────────────────────────────────────────────
    def _task(self, task: ET.Element, sweep: dict | None = None) -> list[object]:
        tid = task.get("id") or "a task"
        ref = task.get("simulationReference") or ""
        utc = self.sims.get(ref)
        if utc is None:
            kind = next((_local(e.tag) for e in self.root.iter() if e.get("id") == ref), None)
            what = f"a {kind}" if kind else "which the document does not define"
            self._unread(f"task {tid} runs simulation {ref!r}, {what}, not a uniformTimeCourse")
            return []
        chain = _model_chain(self.models, task.get("modelReference"))
        if not chain:
            self._unread(
                f"task {tid} runs model {task.get('modelReference')!r}, which the document "
                "does not define"
            )
            return []
        source = chain[-1].get("source") or ""
        if self.file is None:
            self.file = source
        elif source != self.file:
            self._unread(
                f"task {tid} runs model {task.get('modelReference')!r} from {source!r}, not "
                f"the document's {self.file!r}, and a replay runs one model"
            )
            return []
        steps: list[object] = []
        for model in reversed(chain):
            changes = _model_changes(model)
            for k, change in enumerate(changes):
                if _overridden_assignment(change, changes[k + 1 :]):
                    continue  # the change after it sets the value, as write_sedml_protocol writes
                where = f"model {model.get('id')!r}"
                step = self._change(change, where)
                if step is not None:
                    steps.append(self._restated(change, where, step))
        steps.append(_experiment_of(utc, sweep))
        return steps

    def _repeated(self, rt: ET.Element, depth: int) -> list[object]:
        rid = rt.get("id") or "a repeatedTask"
        if depth > 64:
            self._unread(f"repeatedTask {rid} nests deeper than a replay follows")
            return []
        sweep = _sweep_of(rt, self.sim_of_task)
        if sweep is not None and "unread" in sweep:
            self._unread(sweep["unread"])
            return []
        steps: list[object] = []
        if (rt.get("resetModel") or "true").lower() == "true":
            steps.append(_RESET)
        # A sweep's setValues apply at each iteration, to the model its task
        # runs, which its model's own changes have made: they go after those.
        # One that follows the swept parameter holds its last point's value
        # after the scan, as SED-ML leaves it: it is evaluated once there.
        set_values: list[object] = []
        frozen: list[object] = []
        if sweep is not None:
            # A setValue beside the swept one: a constant is a step before the
            # scan. One that reads the range the parameter is set to as it
            # stands is a parameter set to an expression of that parameter,
            # which follows it through the scan, as write_sedml_protocol writes
            # one BNG2.pl keeps (issue #1682). Any other changes per point,
            # which a scan does not do.
            ranges, swept = _range_bound(rt)
            on = sweep.get("swept")
            # A species set at each point of a sweep that resets the model is
            # one a replay's scan follows from its start state: the model's
            # own seed, or an amount set to an expression (issue #1733). One
            # set at each point of a sweep that goes on is set per point.
            resets = (rt.get("resetModel") or "true").lower() == "true"
            for sv in _findall_local(rt, "setValue"):
                if sv is swept[0]:
                    continue
                if (
                    on is not None
                    and resets
                    and self._seed_again(sv, f"repeatedTask {rid}", {on: str(sweep["parameter"])})
                ):
                    # The model's own seed, which a replay's scan follows.
                    continue
                read = _ranges_read(sv, ranges)
                if read:
                    if on is not None and read == {on} and _is_param_target(sv):
                        step = self._change(
                            sv, f"repeatedTask {rid}", names={on: str(sweep["parameter"])}
                        )
                        if step is not None:
                            set_values.append(_PerPoint(step))
                            frozen.append(dataclasses.replace(step, evaluated_once=True))
                        continue
                    if on is not None and read == {on} and resets:
                        said = len(self.lossy)
                        step = self._change(
                            sv, f"repeatedTask {rid}", names={on: str(sweep["parameter"])}
                        )
                        if step is not None and step.kind == "set_concentration":
                            set_values.append(_PerPoint(step))
                            continue
                        if len(self.lossy) > said:
                            continue
                    self._unread(
                        f"repeatedTask {rid} sets {sv.get('target')!r} from its range beside "
                        "the swept parameter, which a scan does not do"
                    )
                    continue
                step = self._change(sv, f"repeatedTask {rid}")
                if step is not None:
                    set_values.append(step)
        if sweep is None:
            iterations = _iterations(rt)
            if iterations > 1:
                self._unread(
                    f"repeatedTask {rid} runs its subtasks {iterations} times, which a replay "
                    "runs once"
                )
            # A range of one value, read by a setValue, is that value.
            ranges = {r.get("id") for r in rt.iter() if _local(r.tag).endswith("Range")}
            values = {
                r.get("id"): (r[0].text or "").strip()
                for r in rt.iter()
                if _local(r.tag) == "vectorRange" and len(list(r)) == 1
            }
            for sv in _findall_local(rt, "setValue"):
                if _reads_a_range(sv, ranges) and not all(
                    (ci.text or "").strip() in values
                    for ci in _findall_local(sv, "ci")
                    if (ci.text or "").strip() in ranges
                ):
                    self._unread(
                        f"repeatedTask {rid} sets {sv.get('target')!r} from a range of several "
                        "values without sweeping it, which a replay does not do"
                    )
                    continue
                step = self._change(sv, f"repeatedTask {rid}", values)
                if step is not None:
                    steps.append(self._restated(sv, f"repeatedTask {rid}", step))
        subs = sorted(
            _findall_local(rt, "subTask"), key=lambda s: _number_or_none(s.get("order")) or 0.0
        )
        for st in subs:
            ref = st.get("task") or ""
            if ref in self.tasks:
                run = self._task(self.tasks[ref], sweep)
                # Before its scan, after its model's changes; the first one only.
                steps.extend(run[:-1] + set_values + run[-1:] + frozen if run else set_values)
                set_values, frozen = [], []
            elif ref in self.rts:
                if sweep is not None:
                    self._unread(
                        f"repeatedTask {rid} sweeps repeatedTask {ref}, which a replay does not"
                    )
                    continue
                steps.extend(self._repeated(self.rts[ref], depth + 1))
            else:
                self._unread(
                    f"repeatedTask {rid} runs {ref!r}, which the document does not define"
                )
        return steps + set_values

    # ── changes ────────────────────────────────────────────────────────────
    def _change(
        self,
        elem: ET.Element,
        where: str,
        values: dict | None = None,
        names: dict[str, str] | None = None,
    ):
        """The :class:`StateChange` a ``changeAttribute``, ``computeChange`` or
        ``setValue`` makes, or ``None``, said. ``values`` are the one-value
        ranges a ``setValue``'s math may read, by id. ``names`` are ranges it
        reads as the parameter a sweep sets to each, by id: its value is then
        an expression of that parameter, kept as one; any other is evaluated
        once, where the step runs, as SED-ML evaluates it."""
        kind = _local(elem.tag)
        target = elem.get("target") or ""
        named = _target_name(target)
        if named is None:
            self._unread(f"{where}: {kind} of {target!r}, which is no species or parameter value")
            return None
        what, name, attr = named
        if kind == "changeAttribute":
            value = _number_or_none(elem.get("newValue"))
            if value is None:
                self._unread(
                    f"{where}: changeAttribute of {name} to {elem.get('newValue')!r}, not a number"
                )
                return None
            return self._state_change(what, name, attr, value, None, False)
        if kind not in ("computeChange", "setValue"):
            self._unread(f"{where}: {kind}, which a replay does not apply")
            return None
        formula = self._formula(elem, where, values, names)
        if formula is None:
            return None
        add = False
        if what == "species":
            own = re.escape(name)
            m = re.match(rf"^\s*{own}\s*\+\s*(.+?)\s*$", formula) or re.match(
                rf"^\s*(.+?)\s*\+\s*{own}\s*$", formula
            )
            if m is not None:
                formula, add = m.group(1), True
        value = _number_or_none(formula)
        return self._state_change(
            what, name, attr, value, None if value is not None else formula, add, once=not names
        )

    def _state_change(self, what, name, attr, value, expr, add, *, once: bool = True):
        from bngsim.convert._protocol import StateChange

        if what == "param":
            # SED-ML evaluates a change's math once, where it applies; BNG2.pl
            # keeps a setParameter's expression, which follows what it reads
            # (issue #1682), so the step says which it is.
            return StateChange(
                kind="set_parameter",
                target=name,
                value=value,
                value_expr=expr,
                evaluated_once=once and expr is not None,
            )
        if attr == "initialAmount" or attr is None:
            # The model holds every species as its concentration, amount over
            # its compartment's size, whatever the species' units say. An
            # initialAmount is an amount; a species' own value (a setValue or
            # computeChange on the species) is its amount where it has only
            # substance units, as write_sedml_protocol writes it, else its
            # concentration. Both need the SBML.
            if self.targets is None or not self.targets.read:
                self._unread(
                    f"species {name}: a change to its {attr or 'value'} needs its compartment's "
                    "size, which the SBML holds (pass sbml=)"
                )
                return None
            if attr == "initialAmount":
                size = self.targets.compartment_size.get(name, 1.0)
            else:
                size = self.targets.amount_scale.get(name, 1.0)
            if size is None:
                self._unread(
                    f"species {name}: its compartment's size is not a number in the SBML, "
                    "and its amount cannot be read as a concentration"
                )
                return None
            if size != 1.0:
                if value is not None:
                    value = value / size
                else:
                    expr = f"({expr})/{size!r}"
        kind = "add_concentration" if add else "set_concentration"
        return StateChange(kind=kind, target=name, value=value, value_expr=expr)

    def _formula(
        self,
        elem: ET.Element,
        where: str,
        values: dict | None = None,
        names: dict[str, str] | None = None,
    ) -> str | None:
        """A ``computeChange``'s or ``setValue``'s math as a formula over the
        names its variables point at (and *names* give, by id), or ``None``,
        said."""
        math = _find(elem, "math")
        if math is None:
            self._unread(f"{where}: {_local(elem.tag)} with no math")
            return None
        try:
            formula = _l3_formula(math)
        except ImportError:  # pragma: no cover - the SBML extra is installed with the readers
            self._unread(f"{where}: reading {_local(elem.tag)} math needs libsbml")
            return None
        if formula is None:
            self._unread(f"{where}: {_local(elem.tag)} math that libsbml cannot read")
            return None
        # Each variable id to the name its target names, and each of the
        # element's own parameters to its value, in one pass: substituted one
        # after another, `k2 := k1, k1 := k2` turned `3*k2` into `3*k2`.
        lookup: dict[str, str] = {}
        for var in _findall_local(elem, "variable"):
            vid = var.get("id") or ""
            named = _target_name(var.get("target") or "")
            if not vid:
                continue
            if named is None:
                self._unread(
                    f"{where}: variable {vid} of {var.get('target')!r}, which is no species "
                    "or parameter"
                )
                return None
            lookup[vid] = named[1]
        lookup.update(names or {})
        numbers = {k: _number_or_none(v) for k, v in (values or {}).items()}
        for par in _findall_local(elem, "parameter"):
            numbers[par.get("id") or ""] = _number_or_none(par.get("value"))
        for pid, value in numbers.items():
            if pid and value is not None:
                # A negative in parentheses: `d^2` with d = -5 read as -5^2.
                lookup[pid] = repr(value) if value >= 0 else f"({value!r})"
        return re.sub(
            r"(?<![\w.])([A-Za-z_]\w*)(?!\w)",
            lambda m: lookup.get(m.group(1), m.group(1)),
            formula,
        )

    def _restated(self, elem: ET.Element, where: str, step):
        """*step*, marked where *elem* sets its element to what the SBML's
        initialAssignment of it computes."""
        if _local(elem.tag) != "changeAttribute" and self._assigned_again(elem, where, {}):
            return _Restated(step)
        return step

    def _seed_again(self, sv: ET.Element, where: str, names: dict[str, str]) -> bool:
        """Whether *sv*, beside a sweep, sets a species to what the SBML's
        initialAssignment of it computes, as write_sedml_protocol writes a
        species whose seed follows the swept parameter at each point (issue
        #1733)."""
        named = _target_name(sv.get("target") or "")
        return (
            named is not None and named[0] == "species" and self._assigned_again(sv, where, names)
        )

    def _assigned_again(self, sv: ET.Element, where: str, names: dict[str, str]) -> bool:
        """Whether *sv* sets a species or a parameter to what the SBML's
        initialAssignment of it computes."""
        named = _target_name(sv.get("target") or "")
        targets = self.targets
        if named is None or targets is None:
            return False
        assigned = targets.assignments.get(named[1])
        if assigned is None:
            return False
        # The assignment's math as the writer says it again, and as it is read.
        math = ET.Element(_q(_MATHML_NS, "math"))
        math.append(_without_min_max(ET.fromstring(assigned)))
        seed = _l3_formula(math)
        if seed is None:  # pragma: no cover - the SBML's own math, read once already
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConversionWarning)
            lossy = len(self.lossy)
            formula = self._formula(sv, where, None, names)
            del self.lossy[lossy:]
        return formula is not None and formula.replace(" ", "") == seed.replace(" ", "")

    def _unread(self, text: str) -> None:
        self.lossy.append(text)
        warnings.warn(
            f"{text}; run_protocol refuses the protocol unless allow_lossy=True",
            ConversionWarning,
            stacklevel=_outside(),
        )


_TARGET_ASSIGNMENT = re.compile(
    r"""initialAssignment\[\s*@symbol\s*=\s*['"]([^'"]+)['"]\s*\]\s*$"""
)


def _l3_formula(math: ET.Element) -> str | None:
    """A ``math`` element as a formula a replay evaluates, or ``None`` where
    libsbml cannot read it."""
    import libsbml

    math = _if_for_piecewise(copy.deepcopy(math))
    node = libsbml.readMathMLFromString(ET.tostring(math, encoding="unicode"))
    return None if node is None else libsbml.formulaToL3String(node)


def _overridden_assignment(change: ET.Element, later: list[ET.Element]) -> bool:
    """Whether *change* removes an initialAssignment of an element that one of
    the *later* changes sets: the replay sets the value over the expression,
    as BNG2.pl's ``setParameter`` does, so the removal says nothing more."""
    if _local(change.tag) != "removeXML":
        return False
    m = _TARGET_ASSIGNMENT.search(change.get("target") or "")
    if m is None:
        return False
    for other in later:
        named = _target_name(other.get("target") or "")
        if named is not None and named[1] == m.group(1):
            return True
    return False


def _iterations(rt: ET.Element) -> int:
    """How many times a ``repeatedTask`` runs its subtasks: its range's length."""
    rid = rt.get("range")
    for rng in rt.iter():
        if _local(rng.tag).endswith("Range") and (rid is None or rng.get("id") == rid):
            if _local(rng.tag) == "vectorRange":
                return sum(1 for v in rng if _local(v.tag) == "value")
            if _local(rng.tag) == "uniformRange":
                return _uniform_range_points(rng) or 1
            return 1
    return 1


def _uniform_range_points(rng: ET.Element) -> int | None:
    """Inclusive point count of a ``uniformRange``, or ``None`` if unstated.

    SED-ML's ``numberOfPoints`` counts intervals — a range expands to
    ``numberOfPoints + 1`` values — which is the same convention the
    ``uniformTimeCourse`` reader above applies. bngsim's ``scan_points`` is the
    inclusive count on both sides of that conversion (GH #575). L1V4 calls the
    attribute ``numberOfSteps`` (issue #1493).
    """
    raw = _steps_attr(rng)
    if raw is None:
        return None
    try:
        intervals = int(float(raw))
    except ValueError:
        return None
    if intervals < 0:
        return None
    return intervals + 1


def _scan_targets(root: ET.Element) -> dict[str, dict]:
    """Map each scanned simulation id → its sweep (parameter/min/max/points/…).

    Walks ``repeatedTask``\\ s: a subTask's task references a simulation, and the
    repeatedTask's ``uniformRange`` + ``setValue`` give the swept parameter.
    """
    sim_of_task = {t.get("id"): t.get("simulationReference") for t in _findall_local(root, "task")}
    out: dict[str, dict] = {}
    for rt in _findall_local(root, "repeatedTask"):
        info = _sweep_of(rt, sim_of_task)
        if info is None:
            continue
        for st in _findall_local(rt, "subTask"):
            sim = sim_of_task.get(st.get("task"))
            if sim is not None:
                out[sim] = info
    return out


def _reads_a_range(sv: ET.Element, ranges: set) -> bool:
    return bool(_ranges_read(sv, ranges))


def _ranges_read(sv: ET.Element, ranges: set) -> set[str]:
    """The range ids a ``setValue`` reads: its ``range``, and each its math
    names."""
    read = {(ci.text or "").strip() for ci in _findall_local(sv, "ci")} & ranges
    if sv.get("range") in ranges:
        read.add(sv.get("range") or "")
    return read


def _is_param_target(sv: ET.Element) -> bool:
    named = _target_name(sv.get("target") or "")
    return named is not None and named[0] == "param"


def _range_bound(rt: ET.Element) -> tuple[set, list[ET.Element]]:
    """A ``repeatedTask``'s range ids, and the ``setValue``\\ s that set a
    parameter from one of them: its sweep."""
    ranges = {r.get("id") for r in rt.iter() if _local(r.tag).endswith("Range")}
    swept = [
        sv
        for sv in _findall_local(rt, "setValue")
        if (named := _target_name(sv.get("target") or "")) is not None
        and named[0] == "param"
        and _reads_a_range(sv, ranges)
    ]
    return ranges, swept


def _sweep_of(rt: ET.Element, sim_of_task: dict) -> dict | None:
    """The sweep a ``repeatedTask`` runs (parameter/min/max/points/…), or
    ``None`` where it sweeps nothing.

    A sweep sets a parameter from its own range. A repeatedTask that does not
    (one iteration that chains runs, or sets a value between them) runs its
    subtasks; it does not scan them.

    The values are the ``setValue``'s math at each iteration of the
    repeatedTask's range, over the range it names: a scan of k = 10^x over
    x = 0, 1, 2 runs k = 1, 10, 100, as does one of k over a second range
    y = 1, 10, 100, or over a ``functionalRange`` 10^x (issue #1664). Taken
    as the first range's values, each ran k = 0, 1, 2. A setValue that is its
    master range as it stands keeps that range's form: a ``uniformRange``'s
    bounds and spacing, or a ``vectorRange``'s values. ``info["swept"]`` is
    the range id the parameter is set to as it stands, where it is one.
    """
    ranges, swept = _range_bound(rt)
    if not swept:
        return None
    sv = swept[0]
    by_id = {r.get("id") or "": r for r in rt.iter() if _local(r.tag).endswith("Range")}
    if not by_id:
        return None
    master: str = rt.get("range") or ""
    if master not in by_id:
        master = next(iter(by_id))
    math = _find(sv, "math")
    terms = list(math) if math is not None else []
    bare: str | None = None
    if not terms and sv.get("range") in ranges:
        bare = sv.get("range")
    elif (
        len(terms) == 1
        and _local(terms[0].tag) == "ci"
        and (terms[0].text or "").strip() in ranges
    ):
        bare = (terms[0].text or "").strip()
    tgt = sv.get("target") or ""
    named = _target_name(tgt)
    info: dict[str, object] = {
        "parameter": (
            named[1] if named is not None else tgt.split("parameter[@id='", 1)[1].split("'", 1)[0]
        ),
        "reset": (rt.get("resetModel") or "true").lower() == "true",
    }
    rng = by_id[master]
    if bare == master and _local(rng.tag) == "uniformRange":
        info.update(
            {
                "min": _float_attr(rng, "start", 0.0),
                "max": _float_attr(rng, "end", 1.0),
                # GH #575 — the mirror of the write side: numberOfPoints intervals
                # are numberOfPoints+1 inclusive points. An absent attribute is
                # unknown rather than zero, and stays None.
                "points": _uniform_range_points(rng),
                "log": (rng.get("type") or "linear").lower() == "log",
                "swept": master,
            }
        )
        return info
    found = _range_values(by_id, master)
    if isinstance(found, str):
        info["unread"] = f"repeatedTask {rt.get('id')}: {found}"
        return info
    values: list[float] = []
    if bare is not None:
        values = found[bare][: len(found[master])]
        if len(values) < len(found[master]):
            info["unread"] = (
                f"repeatedTask {rt.get('id')} sets {tgt!r} from range {bare}, which has fewer "
                f"values than the {len(found[master])} iterations of its range {master}"
            )
            return info
        info["swept"] = bare
    else:
        # k1 = 10^x over x = (0, 1, 2): the value at each iteration.
        if _findall_local(sv, "variable"):
            info["unread"] = (
                f"repeatedTask {rt.get('id')} sets {tgt!r} from its range and the model's "
                "values, which a scan does not read"
            )
            return info
        own = _own_parameters(sv)
        for k in range(len(found[master])):
            env = {rid: vals[k] for rid, vals in found.items() if k < len(vals)}
            value = _math_value(math, {**own, **env}) if math is not None else None
            if value is None:
                info["unread"] = (
                    f"repeatedTask {rt.get('id')} sets {tgt!r} to a function of its range "
                    "that this reader does not evaluate"
                )
                return info
            values.append(value)
    if not values:
        return None
    # The values exactly (issue #1366), and their range for a reader of that.
    info.update(
        {
            "values": tuple(values),
            "min": min(values),
            "max": max(values),
            "points": len(values),
            "log": False,
        }
    )
    return info


def _own_parameters(elem: ET.Element) -> dict[str, float]:
    """An element's own ``listOfParameters``, by id."""
    out: dict[str, float] = {}
    for par in _findall_local(elem, "parameter"):
        value = _number_or_none(par.get("value"))
        if par.get("id") and value is not None:
            out[par.get("id") or ""] = value
    return out


def _range_values(by_id: dict, master: str) -> dict[str, list[float]] | str:
    """Each range's values in order, by id, or why they cannot be read: a
    ``vectorRange``'s values, a ``uniformRange``'s points (evenly spaced, or
    in log scale for ``type="log"``), and a ``functionalRange``'s math at
    each value of the range it is over."""
    out: dict[str, list[float]] = {}
    functional: list[ET.Element] = []
    for rid, rng in by_id.items():
        kind = _local(rng.tag)
        if kind == "vectorRange":
            try:
                out[rid] = [float(v.text or "") for v in rng if _local(v.tag) == "value"]
            except ValueError:
                return f"vectorRange {rid} holds a value that is not a number"
        elif kind == "uniformRange":
            points = _uniform_range_points(rng)
            if points is None:
                return f"uniformRange {rid} has no number of points"
            lo, hi = _float_attr(rng, "start", 0.0), _float_attr(rng, "end", 1.0)
            log = (rng.get("type") or "linear").lower() == "log"
            if log and (lo <= 0 or hi <= 0):
                return f"uniformRange {rid} is in log scale over {lo:g}..{hi:g}"
            a, b = (math.log(lo), math.log(hi)) if log else (lo, hi)
            step = 0.0 if points == 1 else (b - a) / (points - 1)
            vals = [a + k * step for k in range(points)]
            out[rid] = [math.exp(v) for v in vals] if log else vals
        elif kind == "functionalRange":
            functional.append(rng)
        else:
            return f"{kind} {rid}, which this reader does not read"
    # A functionalRange is over another range, which may be one too.
    for _ in range(len(functional)):
        for rng in functional:
            rid, over = rng.get("id") or "", rng.get("range") or ""
            if rid in out or over not in out:
                continue
            if _findall_local(rng, "variable"):
                return f"functionalRange {rid} reads the model's values"
            body = _find(rng, "math")
            own = _own_parameters(rng)
            vals = []
            for x in out[over]:
                value = _math_value(body, {**own, over: x}) if body is not None else None
                if value is None:
                    return f"functionalRange {rid}'s math is not one this reader evaluates"
                vals.append(value)
            out[rid] = vals
    missing = [r.get("id") for r in functional if r.get("id") not in out]
    if missing:
        return f"functionalRange {missing[0]} is over no range the repeatedTask has"
    if not out.get(master):
        return f"range {master} has no values"
    return out


def _cn_value(node: ET.Element) -> float | None:
    """A MathML ``<cn>``'s number: a real or integer, an ``e-notation``
    mantissa and exponent, or a ``rational`` numerator and denominator."""
    kind = (node.get("type") or "real").strip()
    head = (node.text or "").strip()
    sep = next((c for c in node if _local(c.tag) == "sep"), None)
    tail = (sep.tail or "").strip() if sep is not None else None
    try:
        if kind == "e-notation" and tail is not None:
            return float(head) * 10.0 ** float(tail)
        if kind == "rational" and tail is not None:
            return float(head) / float(tail)
        return float(head)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _math_value(node: ET.Element, env: dict[str, float]) -> float | None:
    """A MathML expression's value with the names in *env*, or ``None`` where
    it names something else or uses an element this does not evaluate (the
    arithmetic, ``power``, ``exp``, ``ln``, ``log``, ``root``, ``abs``,
    ``floor``, ``ceiling``, ``pi`` and ``exponentiale``)."""
    tag = _local(node.tag)
    if tag == "math":
        kids = list(node)
        return _math_value(kids[0], env) if len(kids) == 1 else None
    if tag == "cn":
        return _cn_value(node)
    if tag == "ci":
        return env.get((node.text or "").strip())
    if tag == "pi":
        return math.pi
    if tag == "exponentiale":
        return math.e
    if tag != "apply":
        return None
    kids = list(node)
    if not kids:
        return None
    op = _local(kids[0].tag)
    qualifier = {_local(k.tag): k for k in kids[1:] if _local(k.tag) in ("logbase", "degree")}
    args = [_math_value(k, env) for k in kids[1:] if _local(k.tag) not in ("logbase", "degree")]
    if any(a is None for a in args):
        return None
    xs = [float(a) for a in args if a is not None]

    def qualified(name: str, default: float) -> float | None:
        q = qualifier.get(name)
        if q is None:
            return default
        inner = list(q)
        return _math_value(inner[0], env) if len(inner) == 1 else None

    try:
        if op == "plus":
            return math.fsum(xs)
        if op == "times":
            return math.prod(xs)
        if op == "minus" and len(xs) in (1, 2):
            return -xs[0] if len(xs) == 1 else xs[0] - xs[1]
        if op == "divide" and len(xs) == 2:
            return xs[0] / xs[1]
        if op == "power" and len(xs) == 2:
            return float(xs[0] ** xs[1])
        if len(xs) != 1:
            return None
        x = xs[0]
        if op == "exp":
            return math.exp(x)
        if op == "ln":
            return math.log(x)
        if op == "log":
            base = qualified("logbase", 10.0)
            return None if base is None else math.log(x, base)
        if op == "root":
            degree = qualified("degree", 2.0)
            return None if degree is None else float(x ** (1.0 / degree))
        if op == "abs":
            return abs(x)
        if op == "floor":
            return float(math.floor(x))
        if op == "ceiling":
            return float(math.ceil(x))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    return None
