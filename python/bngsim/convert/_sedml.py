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

import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

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
    _algorithm_param_elems(algo, params)

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
    for dg in _findall_local(root, "dataGenerator"):
        var = _find(dg, "variable")
        if var is not None and (var.get("symbol") or "").strip() == _TIME_SYMBOL:
            continue
        name = dg.get("name") or (var.get("name") if var is not None else None)
        target_sel = _selector_from_var(var) if var is not None else None
        label = name if _is_typed_selector(name) else (target_sel or name)
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
        output_start=output_start,
        n_points=n_points,
        outputs=tuple(outputs),
        rtol=rtol,
        atol=ExactAtol(atol) if exact_atol and atol is not None else atol,
        max_steps=max_steps,
    )


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


class _HeldState:
    """One kind of model state (concentrations or parameters) as BNG2.pl holds
    it at a point in a protocol: the numeric values set since the model's
    initial state, and what else moved it there, if anything did."""

    def __init__(self, values: dict | None = None, moved: list | None = None):
        self.values: dict[str | None, tuple[float, str]] = dict(values or {})
        # (the target it moved, or None for the whole state; what moved it)
        self.moved: list[tuple[str | None, str]] = list(moved or [])

    def copy(self) -> _HeldState:
        return _HeldState(self.values, self.moved)

    def differs(
        self, described: dict[str | None, float], noun: str, ignore: str | None
    ) -> str | None:
        """How this state differs from the initial state with *described* set,
        leaving out *ignore*, a parameter the experiment sets itself."""
        moved = [text for target, text in self.moved if target is None or target != ignore]
        if moved:
            return f"the {noun} left by " + ", then ".join(moved)
        diffs = [
            f"{target}={value:g}{note}"
            for target, (value, note) in self.values.items()
            if target != ignore and described.get(target) != value
        ]
        diffs += [
            f"{t} at its initial value" for t in described if t != ignore and t not in self.values
        ]
        return f"{noun} {', '.join(diffs)}" if diffs else None


def _carried_starts(protocol) -> dict[int, str]:
    """Experiment index → how the state BNG2.pl starts it from differs from the
    one SED-ML's standard elements give it, for each experiment where they do.

    The standard elements start every experiment from the model's initial state,
    with the numeric ``setConcentration``/``setParameter`` values that
    :meth:`~bngsim.convert.ProtocolSpec.state_changes_before` collects baked
    into a derived model. BNG2.pl starts each from the state the steps before it
    left: the end of the last run, unless a ``resetConcentrations`` restored
    another; an ``addConcentration``; a value only the model can read; a scanned
    parameter where the scan left it; a saved state a reset restored. A standards
    tool restarted every one of those from the initial state, with nothing said.
    Encoding the carry-over as a ``repeatedTask``'s ordered subtasks does not
    help: tellurium resets the model between subtasks.
    """
    from bngsim._protocol_run import _DEFAULT_LABEL, _scan_values
    from bngsim.convert._protocol import Experiment, StateChange, _state_change_action

    held = {"conc": _HeldState(), "param": _HeldState()}
    # What state_changes_before gives: the latest set of each target since the
    # last reset of its kind, None where that set is not a number.
    described: dict[str, dict[str | None, float | None]] = {"conc": {}, "param": {}}
    saved: dict[str, dict[str, _HeldState]] = {"conc": {}, "param": {}}
    built_until = protocol.generation_step or 0
    out: dict[int, str] = {}
    n = 0
    for k, step in enumerate(protocol.steps):
        # A step before the network was generated is in the network the SBML
        # was written from, so it does not move the state from that one.
        built = k < built_until
        if isinstance(step, StateChange):
            kind = (
                "param"
                if step.kind.endswith("_parameter") or "parameters" in step.kind
                else "conc"
            )
            state = held[kind]
            label = step.label or _DEFAULT_LABEL
            if step.kind in ("set_concentration", "set_parameter"):
                described[kind][step.target] = step.value
                if step.value is not None:
                    state.values[step.target] = (step.value, "")
                else:
                    state.values.pop(step.target, None)
                    if not built:
                        state.moved.append(
                            (
                                step.target if kind == "param" else None,
                                f"{_state_change_action(step)}, which needs the model to read",
                            )
                        )
            elif step.kind == "add_concentration":
                if not built:
                    state.moved.append((None, _state_change_action(step)))
            elif step.kind.startswith("save_"):
                saved[kind][label] = state.copy()
            elif step.kind.startswith("reset_"):
                described[kind] = {}
                held[kind] = saved[kind].get(label, _HeldState()).copy()
            continue
        if not isinstance(step, Experiment):  # pragma: no cover — two step types only
            continue
        why = [
            reason
            for kind, noun in (("conc", "concentrations"), ("param", "parameters"))
            if (
                reason := held[kind].differs(
                    {t: v for t, v in described[kind].items() if v is not None},
                    noun,
                    # A scan sets its own parameter at every point.
                    step.scan_parameter if step.is_scan and kind == "param" else None,
                )
            )
            is not None
        ]
        if why:
            out[n] = "; ".join(why)
        if step.is_scan and step.scan_parameter:
            # The parameter keeps the value the sweep ended on (BNGAction.pm).
            try:
                values = _scan_values(step, backward=step.is_bifurcation)
            except ConversionError:
                values = []
            note = f" (where experiment {n}'s scan left it)"
            if values:
                held["param"].values[step.scan_parameter] = (float(values[-1]), note)
            else:
                held["param"].moved.append(
                    (step.scan_parameter, f"experiment {n}'s scan of {step.scan_parameter}")
                )
        if not (step.kind == "scan" and step.reset_between):
            # A scan that resets each point ends reset to where it started.
            held["conc"].moved = [(None, f"experiment {n}'s run")]
        n += 1
    return out


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

    The standard elements start each experiment from the model's initial state
    with those overrides. An experiment BNG2.pl starts from somewhere else, such
    as where the run before it ended, is named in a
    :class:`~bngsim.ConversionWarning`; a standards tool runs it from the
    initial state.

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
    from bngsim.convert._protocol import _evenly_spaced

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

    carried = _carried_starts(protocol)
    if carried:
        shown = [f"experiment {i} from {why}" for i, why in list(carried.items())[:5]]
        more = len(carried) - len(shown)
        warnings.warn(
            "SED-ML's standard elements start each experiment from the model's initial "
            "state, with the protocol's numeric setConcentration and setParameter values "
            "applied, and a standards tool runs them so; BNG2.pl starts "
            + "; ".join(shown)
            + (f"; and {more} more experiment(s) from a carried state" if more else "")
            + ". The bngsim annotation holds the protocol as BNG2.pl runs it.",
            ConversionWarning,
            stacklevel=2,
        )

    for i, exp in enumerate(experiments):
        # What the standard elements below cannot hold; the annotation holds it.
        if exp.is_bifurcation:
            warnings.warn(
                f"SED-ML holds experiment {i}, a bifurcate, as its upward sweep only; "
                "the sweep back down is in the bngsim annotation",
                ConversionWarning,
                stacklevel=2,
            )
        if exp.sample_times is not None and not _evenly_spaced(list(exp.sample_times)):
            warnings.warn(
                f"SED-ML holds experiment {i}'s uneven output times {list(exp.sample_times)} "
                f"as {exp.n_points} even points; the exact times are in the bngsim annotation",
                ConversionWarning,
                stacklevel=2,
            )
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
            _utc_attrs(sim_id, t0, exp.output_start, t_end, n_steps),
        )
        kisao, params = _protocol_algorithm(exp, i)
        algo = ET.SubElement(utc, _q(_SEDML_NS, "algorithm"), {"kisaoID": kisao})
        _algorithm_param_elems(algo, params)

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
            if exp.scan_values is not None:
                # par_scan_vals, exactly and in their order (issue #1366).
                vr = ET.SubElement(lor, _q(_SEDML_NS, "vectorRange"), {"id": rng_id})
                for v in exp.scan_values:
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
                        # on the uniformTimeCourse above: a consumer expands a
                        # uniformRange to numberOfPoints+1 values. bngsim's
                        # scan_points is the inclusive point count (the linspace /
                        # BNG2.pl n_scan_pts convention, delta = (max-min)/(n-1)),
                        # so writing it verbatim handed every standards-compliant
                        # reader one point more than the scan bngsim ran — 6 values
                        # over [1,2] where bngsim used 5, each at a different
                        # parameter value. Nothing downstream could notice: the
                        # document is well-formed and the sweep looks reasonable.
                        "numberOfPoints": str(max(int(exp.scan_points) - 1, 0)),
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
                    stacklevel=2,
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
        scan = scan_by_sim.get(utc.get("id") or "")
        if scan is not None:
            steps.append(
                Experiment(
                    kind="scan",
                    method=method,
                    t_span=(t0, t_end),
                    output_start=output_start,
                    extra=extra,
                    n_points=n_points,
                    scan_parameter=scan.get("parameter"),
                    scan_min=scan.get("min"),
                    scan_max=scan.get("max"),
                    scan_points=scan.get("points"),
                    scan_log=scan.get("log", False),
                    reset_between=scan.get("reset", True),
                    scan_values=scan.get("values"),
                )
            )
        else:
            steps.append(
                Experiment(
                    kind="simulate",
                    method=method,
                    t_span=(t0, t_end),
                    output_start=output_start,
                    extra=extra,
                    n_points=n_points,
                )
            )
    return ProtocolSpec(steps=tuple(steps), source=None, dropped=())


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
        rng = _find(rt, "uniformRange")
        vec = _find(rt, "vectorRange")
        if rng is not None:
            info: dict[str, object] = {
                "min": _float_attr(rng, "start", 0.0),
                "max": _float_attr(rng, "end", 1.0),
                # GH #575 — the mirror of the write side: numberOfPoints intervals
                # are numberOfPoints+1 inclusive points. An absent attribute is
                # unknown rather than zero, and stays None.
                "points": _uniform_range_points(rng),
                "log": (rng.get("type") or "linear").lower() == "log",
            }
        elif vec is not None:
            try:
                values = tuple(float(v.text or "") for v in vec if _local(v.tag) == "value")
            except ValueError:
                continue
            if not values:
                continue
            # The values exactly (issue #1366), and their range for a reader of that.
            info = {
                "values": values,
                "min": min(values),
                "max": max(values),
                "points": len(values),
                "log": False,
            }
        else:
            continue
        info["reset"] = (rt.get("resetModel") or "true").lower() == "true"
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
