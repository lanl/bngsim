"""bngsim.convert._omex — OMEX / COMBINE archive packaging (GH #219).

A **COMBINE Archive** (``.omex``) is the standard container that bundles a model
(SBML) together with its simulation protocol (SED-ML) and a ``manifest.xml`` that
lists every entry and its *format URI*, all inside one zip. It is how these
artifacts actually travel — BioModels distributes models this way.

This module is the **packaging layer** on top of the two content channels the
converter already has: the network channel (SBML — GH #216 ``net_to_sbml`` /
#215 ``sbml_to_net``) and the protocol channel (SED-ML — GH #218
``read_sedml`` / ``write_sedml``). No model semantics live here; it is only
container plumbing — zip in, zip out, dispatch by manifest format URI.

* :func:`read_omex` — unzip, parse ``manifest.xml``, and return an
  :class:`OmexArchive` whose entries can be dispatched to the SBML/``.net`` and
  SED-ML readers (``→`` network + protocol).
* :func:`write_omex` — bundle a set of content files plus a generated
  ``manifest.xml`` (and optional ``metadata.rdf``) into a ``.omex`` zip.

The whole thing is :mod:`zipfile` + a small manifest reader/writer over the
stdlib XML tools; there is no ``python-libcombine`` runtime dependency.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from bngsim._exceptions import ConversionError

if TYPE_CHECKING:
    from bngsim._eval_spec import EvaluationSpec
    from bngsim._model import Model

__all__ = ["read_omex", "write_omex", "OmexArchive", "OmexEntry"]


# ─── Format URIs ────────────────────────────────────────────────────────────
# The COMBINE specifications registry (identifiers.org) defines URIs for the
# standard formats. BioNetGen `.net` is not a registered COMBINE specification,
# so we mint a stable bngsim URI for it; reading is robust to either.

_MANIFEST_NS = "http://identifiers.org/combine.specifications/omex-manifest"
_FMT_OMEX = "http://identifiers.org/combine.specifications/omex"
_FMT_MANIFEST = "http://identifiers.org/combine.specifications/omex-manifest"
_FMT_SBML = "http://identifiers.org/combine.specifications/sbml"
_FMT_SEDML = "http://identifiers.org/combine.specifications/sed-ml"
_FMT_OMEX_METADATA = "http://identifiers.org/combine.specifications/omex-metadata"
# Not a registered COMBINE spec — a stable bngsim URI so .net round-trips.
_FMT_NET = "http://purl.org/bngsim/bngnet"
# The original rule-based .bngl, bundled as provenance (not a loadable model entry —
# from_net cannot read rule-based BNGL). A distinct URI so it classifies as "source".
_FMT_BNGL = "http://purl.org/bngsim/bngl"
_FMT_JSON = "application/json"

_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_DCTERMS_NS = "http://purl.org/dc/terms/"

_MANIFEST_LOCATION = "./manifest.xml"


def _q(uri: str, local: str) -> str:
    """Clark-notation ``{uri}local`` element tag.

    ElementTree serializes this exactly as it does an ``ET.QName`` object (it
    converts a QName tag to this same text before writing), but as a plain
    ``str`` it type-checks against Element/SubElement's ``tag`` parameter.
    """
    return f"{{{uri}}}{local}"


def _format_for_suffix(suffix: str) -> str:
    """Best-effort format URI from a filename suffix (used when none is given)."""
    s = suffix.lower()
    if s in (".xml", ".sbml"):
        return _FMT_SBML
    if s in (".sedml", ".sedx"):
        return _FMT_SEDML
    if s == ".net":
        return _FMT_NET
    if s == ".bngl":
        return _FMT_BNGL
    if s == ".json":
        return _FMT_JSON
    if s == ".rdf":
        return _FMT_OMEX_METADATA
    return "application/octet-stream"


def build_metadata_rdf(*, creator: str, created: str, description: str) -> str:
    """Render a COMBINE ``omex-metadata`` RDF/XML document describing the archive.

    Uses Dublin Core terms (``dcterms:creator`` / ``created`` / ``description``)
    on the archive root (``rdf:about="."``) — the channel BioModels and other
    COMBINE tools read for provenance. ``created`` is a W3CDTF timestamp string.
    """
    ET.register_namespace("rdf", _RDF_NS)
    ET.register_namespace("dcterms", _DCTERMS_NS)
    root = ET.Element(_q(_RDF_NS, "RDF"))
    desc = ET.SubElement(root, _q(_RDF_NS, "Description"), {f"{{{_RDF_NS}}}about": "."})
    ET.SubElement(desc, _q(_DCTERMS_NS, "creator")).text = creator
    created_el = ET.SubElement(desc, _q(_DCTERMS_NS, "created"))
    ET.SubElement(created_el, _q(_DCTERMS_NS, "W3CDTF")).text = created
    ET.SubElement(desc, _q(_DCTERMS_NS, "description")).text = description
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def _kind_of_format(uri: str) -> str:
    """Classify a format URI into a content kind the readers dispatch on.

    Robust to the *versioned* forms other tools emit
    (``…/sbml.level-3.version-2``, ``…/sed-ml.level-1.version-3``): we match on a
    substring, not exact equality.
    """
    u = (uri or "").lower()
    if "sed-ml" in u or "sedml" in u:
        return "sedml"
    if "sbml" in u:
        return "sbml"
    # The rule-based .bngl is provenance, not a loadable network — classify it as
    # "source" (excluded from model_entries) *before* the net check below.
    if u == _FMT_BNGL or u.endswith("/bngl"):
        return "source"
    if u == _FMT_NET or u.endswith("/bngnet") or "bionetgen" in u:
        return "net"
    if "omex-manifest" in u:
        return "manifest"
    if "omex-metadata" in u:
        return "metadata"
    if u.rstrip("/").endswith("combine.specifications/omex"):
        return "archive"
    return "other"


# ─── Entry / archive records ────────────────────────────────────────────────


@dataclass
class OmexEntry:
    """One ``<content>`` row of an OMEX manifest."""

    location: str
    format: str
    master: bool = False

    @property
    def kind(self) -> str:
        """Content kind (``sbml`` / ``sedml`` / ``net`` / …) for dispatch."""
        return _kind_of_format(self.format)


@dataclass
class OmexArchive:
    """The unpacked contents of a ``.omex`` archive.

    ``root`` is the directory the archive was extracted into; ``entries`` is the
    parsed manifest (excluding the archive-root and manifest self-entries). The
    convenience accessors dispatch the master (or first) model/SED-ML entry to
    the bngsim readers — turning a ``.omex`` into a runnable *network + protocol*.
    """

    root: Path
    entries: list[OmexEntry] = field(default_factory=list)
    _tempdir: tempfile.TemporaryDirectory | None = None

    # ── entry selection ────────────────────────────────────────────────────
    def model_entries(self) -> list[OmexEntry]:
        """All SBML and ``.net`` entries (the network channel)."""
        return [e for e in self.entries if e.kind in ("sbml", "net")]

    def sedml_entries(self) -> list[OmexEntry]:
        """All SED-ML entries (the protocol channel)."""
        return [e for e in self.entries if e.kind == "sedml"]

    def _master(self, candidates: list[OmexEntry]) -> OmexEntry | None:
        if not candidates:
            return None
        for e in candidates:
            if e.master:
                return e
        return candidates[0]

    def master_model_entry(self) -> OmexEntry | None:
        """The model entry marked ``master`` (else the first model entry)."""
        return self._master(self.model_entries())

    def master_sedml_entry(self) -> OmexEntry | None:
        """The SED-ML entry marked ``master`` (else the first SED-ML entry)."""
        return self._master(self.sedml_entries())

    def path_of(self, entry: OmexEntry) -> Path:
        """Resolve an entry's manifest location to its extracted file path."""
        rel = entry.location.lstrip("./")
        return self.root / rel

    # ── dispatch to the content readers ────────────────────────────────────
    def load_model(self) -> Model:
        """Load the master model entry (SBML or ``.net``) into a :class:`Model`."""
        from bngsim._model import Model

        entry = self.master_model_entry()
        if entry is None:
            raise ConversionError(
                "OMEX archive contains no model entry (no SBML or .net in the "
                "manifest); nothing to dispatch to the network reader"
            )
        path = self.path_of(entry)
        if not path.is_file():
            raise ConversionError(
                f"manifest lists model {entry.location!r} but it is not in the archive"
            )
        return Model.from_net(path) if entry.kind == "net" else Model.from_sbml(path)

    def load_protocol(self) -> EvaluationSpec:
        """Read the master SED-ML entry into an :class:`EvaluationSpec`.

        The returned spec points at the archive's master model file so it is
        directly runnable. If the archive carries no SED-ML, a sensible default
        protocol over the model's observables is returned instead (the SBML/.net
        carry no protocol of their own — see :func:`bngsim.convert.default_protocol`).
        """
        from bngsim.convert._sedml import default_protocol, read_sedml

        model_entry = self.master_model_entry()
        model_source = str(self.path_of(model_entry)) if model_entry is not None else None
        model_format = model_entry.kind if model_entry is not None else "sbml"

        sed_entry = self.master_sedml_entry()
        if sed_entry is None:
            if model_entry is None:
                raise ConversionError(
                    "OMEX archive contains neither a SED-ML protocol nor a model "
                    "to derive a default protocol from"
                )
            return default_protocol(
                self.path_of(model_entry),
                model_source=model_source,
                model_format=model_format,
            )
        sed_path = self.path_of(sed_entry)
        if not sed_path.is_file():
            raise ConversionError(
                f"manifest lists SED-ML {sed_entry.location!r} but it is not in the archive"
            )
        return read_sedml(sed_path, model_source=model_source, model_format=model_format)

    def load_full_protocol(self):
        """Compose **every** SED-ML entry into one :class:`ProtocolSpec`.

        Where :meth:`load_protocol` reads only the master (or first) SED-ML entry
        as a single runnable time course, this reads *all* SED-ML files in the
        archive and concatenates their experiments — every ``uniformTimeCourse`` /
        ``repeatedTask`` from every sidecar — into one ordered protocol (GH #222).
        An archive that carries several SED-ML files (each its own experiment set)
        no longer silently drops all but the master. A
        ``resetConcentrations`` boundary separates the contribution of distinct
        files, since they are independent experiments rather than continuations.

        Returns an empty :class:`ProtocolSpec` when the archive has no SED-ML.

        Returns
        -------
        ProtocolSpec
        """
        from bngsim.convert._protocol import ProtocolSpec, combine_protocols
        from bngsim.convert._sedml import read_sedml_protocol

        specs = []
        for entry in self.sedml_entries():
            path = self.path_of(entry)
            if not path.is_file():
                raise ConversionError(
                    f"manifest lists SED-ML {entry.location!r} but it is not in the archive"
                )
            specs.append(read_sedml_protocol(path))
        if not specs:
            return ProtocolSpec()
        return combine_protocols(specs)

    # ── lifecycle ──────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        """Remove the temporary extraction directory, if we created one."""
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def __enter__(self) -> OmexArchive:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    def summary(self) -> str:
        lines = [
            f"OMEX archive: {len(self.entries)} content entr"
            f"{'y' if len(self.entries) == 1 else 'ies'} (extracted to {self.root})"
        ]
        for e in self.entries:
            star = " [master]" if e.master else ""
            lines.append(f"  {e.kind:8} {e.location}{star}")
        return "\n".join(lines)


# ─── Reading ────────────────────────────────────────────────────────────────


def _parse_manifest(text: str) -> list[OmexEntry]:
    """Parse an OMEX ``manifest.xml`` into entries (root/manifest rows dropped)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ConversionError(f"manifest.xml is not parseable XML: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "omexManifest":
        raise ConversionError(
            f"manifest root is <{root.tag.rsplit('}', 1)[-1]}>, not <omexManifest>"
        )
    entries: list[OmexEntry] = []
    for child in root:
        if child.tag.rsplit("}", 1)[-1] != "content":
            continue
        location = child.get("location")
        fmt = child.get("format", "")
        if not location:
            continue
        kind = _kind_of_format(fmt)
        # The archive-root (".") and the manifest self-reference are structural,
        # not content to dispatch.
        norm = location.lstrip("./")
        if (
            kind in ("archive", "manifest")
            or location in (".", "")
            or norm
            in (
                "",
                "manifest.xml",
            )
        ):
            continue
        master = child.get("master", "false").strip().lower() == "true"
        entries.append(OmexEntry(location=location, format=fmt, master=master))
    return entries


def read_omex(source: str | Path, *, extract_dir: str | Path | None = None) -> OmexArchive:
    """Unzip a ``.omex`` archive and parse its manifest.

    Parameters
    ----------
    source : str | Path
        Path to the ``.omex`` (a zip) archive.
    extract_dir : str | Path | None
        Where to extract the contents. If None, a temporary directory is created
        and tied to the returned archive's lifetime (use it as a context manager,
        or call :meth:`OmexArchive.cleanup`, to remove it).

    Returns
    -------
    OmexArchive
        Its :meth:`~OmexArchive.load_model` / :meth:`~OmexArchive.load_protocol`
        accessors dispatch the master entries to the bngsim readers.
    """
    src = Path(source)
    if not src.is_file():
        raise ConversionError(f"no such OMEX archive: {src}")
    if not zipfile.is_zipfile(src):
        raise ConversionError(
            f"{src} is not a zip archive — a COMBINE .omex is a zip of "
            "manifest.xml + content files"
        )

    tempdir: tempfile.TemporaryDirectory | None = None
    if extract_dir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="bngsim-omex-")
        root = Path(tempdir.name)
    else:
        root = Path(extract_dir)
        root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(src) as zf:
            _safe_extractall(zf, root)
            manifest_member = _find_manifest_member(zf)
            if manifest_member is None:
                raise ConversionError(f"{src} has no manifest.xml — not a valid COMBINE archive")
            manifest_text = zf.read(manifest_member).decode("utf-8")
    except ConversionError:
        if tempdir is not None:
            tempdir.cleanup()
        raise

    entries = _parse_manifest(manifest_text)
    return OmexArchive(root=root, entries=entries, _tempdir=tempdir)


def _find_manifest_member(zf: zipfile.ZipFile) -> str | None:
    """Locate the ``manifest.xml`` member (top-level, any leading ``./``)."""
    for name in zf.namelist():
        norm = name.lstrip("./")
        if norm == "manifest.xml":
            return name
    return None


def _safe_extractall(zf: zipfile.ZipFile, root: Path) -> None:
    """Extract every member, refusing paths that escape ``root`` (zip-slip)."""
    root = root.resolve()
    for member in zf.infolist():
        dest = (root / member.filename).resolve()
        if not (dest == root or root in dest.parents):
            raise ConversionError(
                f"refusing to extract {member.filename!r}: path escapes the archive root"
            )
    zf.extractall(root)


# ─── Writing ────────────────────────────────────────────────────────────────


@dataclass
class _Content:
    """An item to bundle: its in-archive location, bytes, format URI, master flag."""

    location: str
    data: bytes
    format: str
    master: bool


def _coerce_content(
    location: str,
    payload: str | bytes | Path,
    fmt: str | None,
    master: bool,
) -> _Content:
    if isinstance(payload, Path):
        data = payload.read_bytes()
    elif isinstance(payload, bytes):
        data = payload
    else:
        data = str(payload).encode("utf-8")
    loc = location if location.startswith("./") else "./" + location.lstrip("/")
    resolved_fmt = fmt if fmt is not None else _format_for_suffix(Path(loc).suffix)
    return _Content(location=loc, data=data, format=resolved_fmt, master=master)


def _build_manifest(contents: list[_Content]) -> str:
    """Render an OMEX ``manifest.xml`` listing the archive root, self, + contents."""
    ET.register_namespace("", _MANIFEST_NS)
    root = ET.Element(_q(_MANIFEST_NS, "omexManifest"))
    # The archive root and the manifest self-entry come first, by convention.
    ET.SubElement(root, _q(_MANIFEST_NS, "content"), {"location": ".", "format": _FMT_OMEX})
    ET.SubElement(
        root,
        _q(_MANIFEST_NS, "content"),
        {"location": _MANIFEST_LOCATION, "format": _FMT_MANIFEST},
    )
    for c in contents:
        attrs = {"location": c.location, "format": c.format}
        if c.master:
            attrs["master"] = "true"
        ET.SubElement(root, _q(_MANIFEST_NS, "content"), attrs)
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def write_omex(
    out_path: str | Path,
    *,
    sbml: str | bytes | Path | None = None,
    net: str | bytes | Path | None = None,
    sedml: str | bytes | Path | None = None,
    metadata: str | bytes | Path | None = None,
    extra: list[tuple[str, str | bytes | Path, str | None]] | None = None,
    sbml_location: str = "./model.xml",
    net_location: str = "./model.net",
    sedml_location: str = "./simulation.sedml",
    metadata_location: str = "./metadata.rdf",
    master: str = "auto",
) -> Path:
    """Bundle model + protocol (+ optional metadata) into a ``.omex`` archive.

    At least one of ``sbml`` / ``net`` must be given. Each payload may be text, a
    bytes blob, or a :class:`~pathlib.Path` to read. A ``manifest.xml`` listing
    every entry's format URI is generated and bundled with them.

    Parameters
    ----------
    out_path : str | Path
        Destination ``.omex`` (zip) path.
    sbml, net, sedml, metadata : str | bytes | Path | None
        The content payloads. ``sbml``/``net`` are the model (network channel);
        ``sedml`` is the protocol; ``metadata`` is an optional RDF sidecar.
    extra : list[tuple[str, payload, format-or-None]] | None
        Additional ``(location, payload, format)`` entries to bundle verbatim
        (e.g. plots, data). ``None`` format → inferred from the suffix.
    *_location : str
        In-archive paths for each standard content (defaults follow convention).
    master : {"auto", "sbml", "net", "sedml", "none"}
        Which entry to mark ``master`` in the manifest. ``"auto"`` marks the
        model (SBML preferred over ``.net``).

    Returns
    -------
    Path
        The written archive path.
    """
    if sbml is None and net is None:
        raise ConversionError(
            "write_omex needs at least a model (sbml= or net=); a COMBINE archive "
            "with no model has nothing to dispatch"
        )

    master_kind = _resolve_master(master, has_sbml=sbml is not None, has_net=net is not None)

    contents: list[_Content] = []
    if sbml is not None:
        contents.append(_coerce_content(sbml_location, sbml, _FMT_SBML, master_kind == "sbml"))
    if net is not None:
        contents.append(_coerce_content(net_location, net, _FMT_NET, master_kind == "net"))
    if sedml is not None:
        contents.append(_coerce_content(sedml_location, sedml, _FMT_SEDML, master_kind == "sedml"))
    if metadata is not None:
        contents.append(_coerce_content(metadata_location, metadata, _FMT_OMEX_METADATA, False))
    for loc, payload, fmt in extra or []:
        contents.append(_coerce_content(loc, payload, fmt, False))

    _ensure_unique_locations(contents)
    manifest_text = _build_manifest(contents)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.xml", manifest_text)
        for c in contents:
            zf.writestr(c.location.lstrip("./"), c.data)
    return out


def _resolve_master(master: str, *, has_sbml: bool, has_net: bool) -> str:
    if master == "auto":
        return "sbml" if has_sbml else "net"
    if master not in ("sbml", "net", "sedml", "none"):
        raise ConversionError(
            f"invalid master={master!r}; expected one of 'auto', 'sbml', 'net', 'sedml', 'none'"
        )
    return master


def _ensure_unique_locations(contents: list[_Content]) -> None:
    seen: set[str] = set()
    for c in contents:
        if c.location in seen:
            raise ConversionError(
                f"duplicate archive location {c.location!r}; give each content a "
                "distinct *_location"
            )
        seen.add(c.location)
