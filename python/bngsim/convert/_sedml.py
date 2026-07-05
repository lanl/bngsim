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

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

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
_KISAO_TO_METHOD = {
    "KISAO:0000019": "ode",  # CVODE
    "KISAO:0000496": "ode",  # CVODES
    "KISAO:0000088": "ode",  # LSODA
    "KISAO:0000071": "ode",  # generic deterministic / LSODA family
    "KISAO:0000029": "ssa",  # Gillespie direct
    "KISAO:0000027": "ssa",  # next-reaction (Gibson–Bruck)
    "KISAO:0000241": "ssa",  # Gillespie-like
}

# Solver options ↔ KiSAO algorithm-parameter terms.
_KISAO_RTOL = "KISAO:0000209"  # relative tolerance
_KISAO_ATOL = "KISAO:0000211"  # absolute tolerance
_KISAO_MAXSTEPS = "KISAO:0000415"  # maximum number of steps


def _q(uri: str, local: str) -> str:
    """Clark-notation ``{uri}local`` element tag.

    ElementTree serializes this exactly as it does an ``ET.QName`` object — it
    converts a QName tag to this same text before writing — but as a plain
    ``str`` it type-checks against Element/SubElement's ``tag`` parameter.
    """
    return f"{{{uri}}}{local}"


# ─── Writing: EvaluationSpec → SED-ML ───────────────────────────────────────


def _selector_target(selector: str) -> dict[str, str]:
    """Standards-compliant ``variable`` attributes for a bngsim output selector.

    A ``species:`` selector gets a real SBML XPath ``target`` (interoperable with
    other SED-ML tools); ``observable:``/``expression:`` are bngsim concepts with
    no standard SBML element, so they carry only the verbatim selector in
    ``name`` (the round-trip source of truth) and no ``target``.
    """
    kind, _, name = selector.partition(":")
    if kind == "species" and name:
        return {"target": f"/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{name}']"}
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

    ET.register_namespace("", _SEDML_NS)
    ET.register_namespace("mml", _MATHML_NS)
    sed = ET.Element(_q(_SEDML_NS, "sedML"), {"level": "1", "version": "3"})

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
        {
            "id": "sim0",
            "initialTime": _num(t0),
            "outputStartTime": _num(t0),
            "outputEndTime": _num(t_end),
            "numberOfPoints": str(n_steps),
        },
    )
    algo = ET.SubElement(
        utc, _q(_SEDML_NS, "algorithm"), {"kisaoID": _METHOD_TO_KISAO[spec.method]}
    )
    params = []
    if spec.rtol is not None:
        params.append((_KISAO_RTOL, _num(spec.rtol)))
    if spec.atol is not None:
        params.append((_KISAO_ATOL, _num(spec.atol)))
    if spec.max_steps is not None:
        params.append((_KISAO_MAXSTEPS, str(int(spec.max_steps))))
    if params:
        lop = ET.SubElement(algo, _q(_SEDML_NS, "listOfAlgorithmParameters"))
        for kid, val in params:
            ET.SubElement(
                lop,
                _q(_SEDML_NS, "algorithmParameter"),
                {"kisaoID": kid, "value": val},
            )

    # ── listOfModels / listOfTasks ─────────────────────────────────────────
    models = ET.SubElement(sed, _q(_SEDML_NS, "listOfModels"))
    ET.SubElement(
        models,
        _q(_SEDML_NS, "model"),
        {
            "id": "model0",
            "language": _SBML_LANG,
            "source": model_source if model_source is not None else spec.model_source,
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
    for i, selector in enumerate(spec.outputs):
        dg_id = f"dg_{i}"
        _data_generator(dgs, dg_id, selector, var_attrs=_selector_target(selector))
        dataset_refs.append((dg_id, selector))

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


def _data_generator(parent: ET.Element, dg_id: str, name: str, *, var_attrs: dict) -> None:
    """Append a ``<dataGenerator>`` of a single variable referencing task0.

    ``name`` carries the verbatim bngsim selector (the round-trip source of
    truth); ``var_attrs`` adds standards-compliant ``symbol``/``target`` for
    interop. The required ``<math>`` is a bare ``<ci>`` over the variable id.
    """
    dg = ET.SubElement(parent, _q(_SEDML_NS, "dataGenerator"), {"id": dg_id, "name": name})
    lov = ET.SubElement(dg, _q(_SEDML_NS, "listOfVariables"))
    var_id = dg_id.replace("dg_", "var_", 1)
    attrs = {"id": var_id, "name": name, "taskReference": "task0"}
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
) -> EvaluationSpec:
    """Parse a SED-ML document into an :class:`~bngsim.EvaluationSpec`.

    Reads the first **uniform time course** and its task's model reference,
    recovering the time grid, output selectors (from each data generator), and
    the solver + tolerances (from the algorithm's KiSAO terms). Namespace-
    agnostic, so SED-ML L1 V1–V4 all parse.

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

    utc = _find(root, "uniformTimeCourse")
    if utc is None:
        raise ConversionError(
            "no <uniformTimeCourse> found — the SED-ML sidecar channel supports "
            "the uniform-time-course protocol only"
        )

    t0 = _float_attr(utc, "outputStartTime", _float_attr(utc, "initialTime", 0.0))
    t_end = _float_attr(utc, "outputEndTime", 100.0)
    n_steps = int(_float_attr(utc, "numberOfPoints", 100.0))
    n_points = n_steps + 1  # inverse of the write-side steps = n_points - 1

    method = "ode"
    rtol = atol = None
    max_steps = None
    algo = _find(utc, "algorithm")
    if algo is not None:
        kid = algo.get("kisaoID") or algo.get("kisaoid")
        if kid is not None:
            mapped = _KISAO_TO_METHOD.get(kid.strip())
            if mapped is None:
                warnings.warn(
                    f"unmapped algorithm KiSAO term {kid!r}; defaulting method to "
                    "'ode'. Set method explicitly on the returned spec if needed.",
                    ConversionWarning,
                    stacklevel=2,
                )
            else:
                method = mapped
        for ap in _findall_local(algo, "algorithmParameter"):
            apk = (ap.get("kisaoID") or ap.get("kisaoid") or "").strip()
            val = ap.get("value")
            if val is None:
                continue
            if apk == _KISAO_RTOL:
                rtol = float(val)
            elif apk == _KISAO_ATOL:
                atol = float(val)
            elif apk == _KISAO_MAXSTEPS:
                max_steps = int(float(val))

    # Outputs: each non-time data generator's name is the verbatim bngsim
    # selector. Fall back to the variable's target/symbol when name is absent.
    outputs: list[str] = []
    for dg in _findall_local(root, "dataGenerator"):
        var = _find(dg, "variable")
        if var is not None and (var.get("symbol") or "").strip() == _TIME_SYMBOL:
            continue
        label = dg.get("name") or (var.get("name") if var is not None else None)
        if label is None and var is not None:
            label = _selector_from_var(var)
        if label and label != "time":
            outputs.append(label)

    # Model reference: resolve the task's model, else the first listed model.
    resolved_source = model_source
    if resolved_source is None:
        model_elem = _find(root, "model")
        resolved_source = model_elem.get("source") if model_elem is not None else ""

    return EvaluationSpec(
        model_source=resolved_source or "",
        model_format=model_format or "sbml",
        method=method,
        t_span=(t0, t_end),
        n_points=n_points,
        outputs=tuple(outputs),
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )


def _selector_from_var(var: ET.Element) -> str | None:
    """Best-effort bngsim selector from a SED-ML variable's target/symbol."""
    target = var.get("target")
    if target and "species[@id='" in target:
        sid = target.split("species[@id='", 1)[1].split("'", 1)[0]
        return f"species:{sid}"
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

# nf (network-free) has no clean KiSAO term; emit it as the stochastic family for
# the standards element — the annotation preserves the exact method either way.
_PROTO_METHOD_TO_KISAO = {**_METHOD_TO_KISAO, "nf": "KISAO:0000029"}


def _sbml_param_target(name: str) -> str:
    return f"/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='{name}']/@value"


def _sbml_species_conc_target(name: str) -> str:
    return (
        f"/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{name}']"
        "/@initialConcentration"
    )


def write_sedml_protocol(
    protocol,
    out_path: str | Path | None = None,
    *,
    model_source: str = "model.xml",
) -> str:
    """Serialize a :class:`~bngsim.convert.ProtocolSpec` to a SED-ML document.

    Emits one ``<uniformTimeCourse>`` + ``<task>`` per experiment (a
    ``<repeatedTask>`` + range for a ``parameter_scan``), with each experiment's
    accumulated ``setParameter``/``setConcentration`` overrides baked into a
    ``changeAttribute``-derived model — plus a bngsim ``<annotation>`` carrying
    the verbatim ``ProtocolSpec`` JSON so :func:`read_sedml_protocol` recovers it
    exactly. Build/IO directives were already dropped at parse time.

    Parameters
    ----------
    protocol : ProtocolSpec
        The parsed protocol to emit. Must carry at least one experiment.
    out_path : str | Path | None
        If given, the document is also written here.
    model_source : str
        The ``<model source="…">`` reference (e.g. the SBML inside the archive).

    Returns
    -------
    str
        The SED-ML document text.
    """
    experiments = protocol.experiments
    if not experiments:
        raise ConversionError(
            "cannot emit SED-ML for an empty protocol (no simulate/parameter_scan "
            "experiments were parsed from the .bngl)"
        )

    ET.register_namespace("", _SEDML_NS)
    ET.register_namespace("mml", _MATHML_NS)
    sed = ET.Element(_q(_SEDML_NS, "sedML"), {"level": "1", "version": "3"})

    # Verbatim ProtocolSpec for exact round-trip (interop reads the elements).
    annot = ET.SubElement(sed, _q(_SEDML_NS, "annotation"))
    payload = ET.SubElement(annot, _q(_BNGSIM_ANNOT_NS, "protocol"))
    payload.text = protocol.to_json()

    models = ET.SubElement(sed, _q(_SEDML_NS, "listOfModels"))
    ET.SubElement(
        models,
        _q(_SEDML_NS, "model"),
        {"id": "model0", "language": _SBML_LANG, "source": model_source},
    )
    sims = ET.SubElement(sed, _q(_SEDML_NS, "listOfSimulations"))
    tasks = ET.SubElement(sed, _q(_SEDML_NS, "listOfTasks"))

    for i, exp in enumerate(experiments):
        # Per-experiment derived model carrying the accumulated operating point.
        overrides = protocol.state_changes_before(exp)
        model_ref = "model0"
        numeric_overrides = [o for o in overrides if o.value is not None]
        if numeric_overrides:
            model_ref = f"model_{i}"
            dm = ET.SubElement(
                models,
                _q(_SEDML_NS, "model"),
                {"id": model_ref, "language": _SBML_LANG, "source": "model0"},
            )
            loc = ET.SubElement(dm, _q(_SEDML_NS, "listOfChanges"))
            for o in numeric_overrides:
                target = (
                    _sbml_param_target(o.target)
                    if o.kind == "set_parameter"
                    else _sbml_species_conc_target(o.target)
                )
                ET.SubElement(
                    loc,
                    _q(_SEDML_NS, "changeAttribute"),
                    {"target": target, "newValue": _num(o.value)},
                )

        t0, t_end = exp.t_span
        n_steps = max(int(exp.n_points) - 1, 0)
        sim_id = f"sim_{i}"
        utc = ET.SubElement(
            sims,
            _q(_SEDML_NS, "uniformTimeCourse"),
            {
                "id": sim_id,
                "initialTime": _num(t0),
                "outputStartTime": _num(t0),
                "outputEndTime": _num(t_end),
                "numberOfPoints": str(n_steps),
            },
        )
        ET.SubElement(
            utc,
            _q(_SEDML_NS, "algorithm"),
            {"kisaoID": _PROTO_METHOD_TO_KISAO.get(exp.method, _PROTO_METHOD_TO_KISAO["ssa"])},
        )

        base_task_id = f"task_{i}"
        ET.SubElement(
            tasks,
            _q(_SEDML_NS, "task"),
            {"id": base_task_id, "modelReference": model_ref, "simulationReference": sim_id},
        )

        if exp.is_scan and exp.scan_parameter and exp.scan_points:
            # A scan wraps the base task in a repeatedTask over the swept range.
            rng_id = f"range_{i}"
            rt = ET.SubElement(
                tasks,
                _q(_SEDML_NS, "repeatedTask"),
                {"id": f"rtask_{i}", "resetModel": _xsd_bool(exp.reset_between), "range": rng_id},
            )
            lor = ET.SubElement(rt, _q(_SEDML_NS, "listOfRanges"))
            ET.SubElement(
                lor,
                _q(_SEDML_NS, "uniformRange"),
                {
                    "id": rng_id,
                    "start": _num(exp.scan_min or 0.0),
                    "end": _num(exp.scan_max if exp.scan_max is not None else 1.0),
                    "numberOfPoints": str(int(exp.scan_points)),
                    "type": "log" if exp.scan_log else "linear",
                },
            )
            loc = ET.SubElement(rt, _q(_SEDML_NS, "listOfChanges"))
            sv = ET.SubElement(
                loc,
                _q(_SEDML_NS, "setValue"),
                {
                    "target": _sbml_param_target(exp.scan_parameter),
                    "range": rng_id,
                    "modelReference": model_ref,
                },
            )
            math = ET.SubElement(sv, _q(_MATHML_NS, "math"))
            ci = ET.SubElement(math, _q(_MATHML_NS, "ci"))
            ci.text = rng_id
            lost = ET.SubElement(rt, _q(_SEDML_NS, "listOfSubTasks"))
            ET.SubElement(
                lost,
                _q(_SEDML_NS, "subTask"),
                {"order": "1", "task": base_task_id},
            )

    ET.indent(sed, space="  ")
    text = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(sed, encoding="unicode")
    if not text.endswith("\n"):
        text += "\n"
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


def _xsd_bool(b: bool) -> str:
    return "true" if b else "false"


def read_sedml_protocol(source: str | Path):
    """Parse a SED-ML document into a :class:`~bngsim.convert.ProtocolSpec`.

    When the document carries the bngsim ``<annotation>`` that
    :func:`write_sedml_protocol` writes, the original ``ProtocolSpec`` is
    recovered **exactly** from its verbatim JSON. Otherwise a best-effort
    reconstruction is built from the standard SED-ML elements (one experiment per
    ``uniformTimeCourse``, with scans recovered from any ``repeatedTask`` range).

    Returns
    -------
    ProtocolSpec
    """
    from bngsim.convert._protocol import Experiment, ProtocolSpec

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
                return ProtocolSpec.from_json(payload.text)
            except (ValueError, KeyError):
                break  # malformed annotation → fall through to reconstruction

    # Best-effort: one Experiment per uniformTimeCourse, in document order.
    utcs = _findall_local(root, "uniformTimeCourse")
    if not utcs:
        raise ConversionError("no <uniformTimeCourse> found — not a runnable SED-ML protocol")
    # Map a uniformTimeCourse id → whether a repeatedTask sweeps a parameter on it.
    scan_by_sim = _scan_targets(root)
    steps: list[object] = []
    for utc in utcs:
        t0 = _float_attr(utc, "outputStartTime", _float_attr(utc, "initialTime", 0.0))
        t_end = _float_attr(utc, "outputEndTime", 100.0)
        n_points = int(_float_attr(utc, "numberOfPoints", 100.0)) + 1
        method = "ode"
        algo = _find(utc, "algorithm")
        if algo is not None:
            kid = (algo.get("kisaoID") or algo.get("kisaoid") or "").strip()
            method = _KISAO_TO_METHOD.get(kid, "ode")
        scan = scan_by_sim.get(utc.get("id") or "")
        if scan is not None:
            steps.append(
                Experiment(
                    kind="scan",
                    method=method,
                    t_span=(t0, t_end),
                    n_points=n_points,
                    scan_parameter=scan.get("parameter"),
                    scan_min=scan.get("min"),
                    scan_max=scan.get("max"),
                    scan_points=scan.get("points"),
                    scan_log=scan.get("log", False),
                    reset_between=scan.get("reset", True),
                )
            )
        else:
            steps.append(
                Experiment(
                    kind="simulate",
                    method=method,
                    t_span=(t0, t_end),
                    n_points=n_points,
                )
            )
    return ProtocolSpec(steps=tuple(steps), source=None, dropped=())


def _scan_targets(root: ET.Element) -> dict[str, dict]:
    """Map each scanned simulation id → its sweep (parameter/min/max/points/…).

    Walks ``repeatedTask``\\ s: a subTask's task references a simulation, and the
    repeatedTask's ``uniformRange`` + ``setValue`` give the swept parameter.
    """
    sim_of_task = {t.get("id"): t.get("simulationReference") for t in _findall_local(root, "task")}
    out: dict[str, dict] = {}
    for rt in _findall_local(root, "repeatedTask"):
        rng = _find(rt, "uniformRange")
        if rng is None:
            continue
        info: dict[str, object] = {
            "min": _float_attr(rng, "start", 0.0),
            "max": _float_attr(rng, "end", 1.0),
            "points": int(_float_attr(rng, "numberOfPoints", 0.0)) or None,
            "log": (rng.get("type") or "linear").lower() == "log",
            "reset": (rt.get("resetModel") or "true").lower() == "true",
        }
        sv = _find(rt, "setValue")
        if sv is not None:
            tgt = sv.get("target") or ""
            if "parameter[@id='" in tgt:
                info["parameter"] = tgt.split("parameter[@id='", 1)[1].split("'", 1)[0]
        for st in _findall_local(rt, "subTask"):
            sim = sim_of_task.get(st.get("task"))
            if sim is not None:
                out[sim] = info
    return out
