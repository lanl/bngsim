"""bngsim.convert — model format converters (GH #211).

Public surface:

* :func:`sbml_to_net` — convert an SBML model to a BioNetGen ``.net`` network
  (GH #215). Network channel only; simulation protocol (events) is a SED-ML
  sidecar's job.
* :func:`net_to_sbml` — convert a BioNetGen ``.net`` network to SBML (GH #216),
  the reverse direction. SBML can carry amount/concentration semantics the
  plain ``.net`` text cannot, so this is the more faithful half.
* :func:`write_net` / :func:`write_sbml` — serialize an already-loaded
  :class:`bngsim.Model` to ``.net`` / SBML text (the reusable serializers).
* :func:`read_omex` / :func:`write_omex` — read/write a COMBINE archive
  (``.omex``): the standard zip container bundling SBML + SED-ML + a manifest
  (GH #219). :func:`net_to_omex` packages a ``.net`` end-to-end.
* :func:`validate_structural_l1` — structural-equivalence (L1) check between a
  source model and its conversion.
* :class:`ConversionReport` — structured result of a conversion.

Example
-------
>>> import bngsim
>>> report = bngsim.convert.sbml_to_net("model.xml", "model.net")
>>> report.ok
True
>>> back = bngsim.convert.net_to_sbml("model.net", "roundtrip.xml")
>>> back.ok
True
"""

from __future__ import annotations

import contextlib
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bngsim.convert._bngl_writer import bngl_capability_report, write_bngl
from bngsim.convert._net_writer import (
    FLUX_HELPER_PREFIX,
    SYNTHETIC_PREFIX,
    capability_report,
    write_net,
)
from bngsim.convert._omex import (
    OmexArchive,
    OmexEntry,
    read_omex,
    write_omex,
)
from bngsim.convert._protocol import (
    Experiment,
    ProtocolSpec,
    StateChange,
    combine_protocols,
    parse_bngl_protocol,
    write_bngl_protocol,
)
from bngsim.convert._sbml_writer import sbml_capability_report, write_sbml
from bngsim.convert._sedml import (
    default_protocol,
    read_sedml,
    read_sedml_protocol,
    write_sedml,
    write_sedml_protocol,
)
from bngsim.convert._validate import (
    ConversionValidationReport,
    LevelResult,
    _ar_report_delta,
    _dynamic_stoich_signatures,
    _effective_n_reactions,
    _max_rhs_delta,
    _stoich_signatures,
    grade_conversion,
    validate_conversion,
)

if TYPE_CHECKING:
    from bngsim._eval_spec import EvaluationSpec
    from bngsim._model import Model

__all__ = [
    "sbml_to_net",
    "sbml_to_bngl",
    "net_to_sbml",
    "write_net",
    "write_sbml",
    "write_bngl",
    "bngl_capability_report",
    "validate_structural_l1",
    "validate_roundtrip",
    "validate_conversion",
    "grade_conversion",
    "read_sedml",
    "write_sedml",
    "read_sedml_protocol",
    "write_sedml_protocol",
    "default_protocol",
    "parse_bngl_protocol",
    "write_bngl_protocol",
    "combine_protocols",
    "ProtocolSpec",
    "Experiment",
    "StateChange",
    "read_omex",
    "write_omex",
    "net_to_omex",
    "omex_to_net",
    "OmexArchive",
    "OmexEntry",
    "ConversionReport",
    "ConversionValidationReport",
    "LevelResult",
    "StructuralReport",
]


# ─── Structured results ───────────────────────────────────────────────────


@dataclass
class StructuralReport:
    """Result of an L1 structural-equivalence check (counts + topology)."""

    passed: bool
    n_species: tuple[int, int]
    n_reactions: tuple[int, int]
    n_parameters: tuple[int, int]
    n_observables: tuple[int, int]
    mismatches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"L1 structural {verdict}: "
            f"species {self.n_species[0]}→{self.n_species[1]}, "
            f"reactions {self.n_reactions[0]}→{self.n_reactions[1]}, "
            f"parameters {self.n_parameters[0]}→{self.n_parameters[1]}, "
            f"observables {self.n_observables[0]}→{self.n_observables[1]}"
            + ("" if self.passed else " — " + "; ".join(self.mismatches))
        )


@dataclass
class ConversionReport:
    """Structured result of a network-channel conversion (either direction)."""

    source: str
    out_path: str | None
    output_text: str
    n_species: int
    n_reactions: int
    n_parameters: int
    n_observables: int
    dropped: list[str] = field(default_factory=list)
    lossy: list[str] = field(default_factory=list)
    structural: StructuralReport | None = None
    max_rhs_delta: float | None = None
    max_ar_report_delta: float | None = None
    rhs_faithful: bool | None = None
    validation: ConversionValidationReport | None = None
    protocol: ProtocolSpec | None = None  # parsed from a source .bngl, when given
    bngl_out: str | None = None  # path of the emitted .bngl actions block, when written (GH #222)
    # What only bngsim's own reader reproduces of the written network: warned,
    # never fatal (sbml_to_net, issue #1274; sbml_to_bngl, issue #1400).
    bngsim_only: list[str] = field(default_factory=list)
    # What the SED-ML's standard elements do not hold of the protocol, each as
    # the writer warned of it (net_to_omex, issue #1684). A tool that runs the
    # standard elements runs something else there; bngsim's own reader reads
    # the protocol from its annotation. ``ok`` and ``lossy`` judge the network.
    sedml_lossy: list[str] = field(default_factory=list)

    # Direction-flavored aliases so each call site reads naturally.
    @property
    def net_text(self) -> str:
        """The produced text (SBML→.net direction reads this name)."""
        return self.output_text

    @property
    def sbml_text(self) -> str:
        """The produced text (.net→SBML direction reads this name)."""
        return self.output_text

    @property
    def ok(self) -> bool:
        """True when the conversion produced output and every check passed.

        When a full L0–L4 gate ran (``validate="full"``) its verdict is
        authoritative — every hard gate (L0–L3) must have passed; L4 never
        blocks. When the ``"L2"`` self-check ran, :attr:`rhs_faithful` is the
        authoritative gate: the round-tripped network reproduces the source's
        *reported ODE dynamics* — both the ODE right-hand side
        (:attr:`max_rhs_delta`) and any AssignmentRule-target species' reported
        value (:attr:`max_ar_report_delta`). The writer may legitimately rewrite a
        functional law as per-species signed flux, changing the reaction
        count/topology, so only species conservation is gated structurally there.
        Otherwise (``"L1"`` counts-only) the full structural equivalence decides.
        """
        if self.validation is not None:
            return self.validation.ok
        if self.rhs_faithful is not None:
            if (
                self.structural is not None
                and self.structural.n_species[0] != self.structural.n_species[1]
            ):
                return False
            return self.rhs_faithful
        if self.structural is not None:
            return self.structural.passed
        return True

    def summary(self) -> str:
        lines = [
            f"Converted {self.source}"
            + (f" → {self.out_path}" if self.out_path else " (in memory)"),
            f"  network: {self.n_species} species, {self.n_reactions} reactions, "
            f"{self.n_parameters} parameters, {self.n_observables} observables",
        ]
        for note in self.dropped:
            lines.append(f"  dropped: {note}")
        for note in self.lossy:
            lines.append(f"  lossy:   {note}")
        for note in self.bngsim_only:
            lines.append(f"  bngsim-only: {note}")
        for note in self.sedml_lossy:
            lines.append(f"  sedml lossy: {note}")
        if self.structural is not None:
            lines.append("  " + self.structural.summary())
        if self.max_rhs_delta is not None:
            lines.append(f"  numerical: max|Δ dy/dt| = {self.max_rhs_delta:.2e}")
        if self.max_ar_report_delta is not None:
            lines.append(
                f"  numerical: max|Δ assignment-rule species| = {self.max_ar_report_delta:.2e}"
            )
        if self.validation is not None:
            for lv in self.validation.levels:
                lines.append("  " + lv.summary())
        if self.bngl_out is not None:
            n_exp = len(self.protocol.experiments) if self.protocol is not None else 0
            lines.append(
                f"  actions: {self.bngl_out} ({n_exp} experiment{'' if n_exp == 1 else 's'})"
            )
        return "\n".join(lines)


# ─── L1 structural validation ─────────────────────────────────────────────
# The low-level signature/RHS primitives (_stoich_signatures,
# _dynamic_stoich_signatures, _max_rhs_delta) are defined in convert._validate
# and imported above so the framework and this surface share one definition.


def validate_structural_l1(source_model: Model, net_model: Model) -> StructuralReport:
    """Compare a source model and its ``.net`` reconstruction at level **L1**.

    L1 (#211c) is structural equivalence: same counts and topology. Concretely
    we compare ``#species``, ``#reactions``, ``#parameters``, ``#observables``
    and the multiset of per-reaction reactant/product index sets. Functions are
    excluded — the writer may synthesize helper functions to reproduce
    propensities the text format cannot otherwise express.
    """
    mismatches: list[str] = []

    def _n_params(m: Model) -> int:
        # Exclude writer-synthesized helper functions' shadow parameters (rate
        # wrappers and per-species flux helpers) — they are .net encoding
        # overhead, not source structure.
        return sum(
            1 for n in m.param_names if not n.startswith((SYNTHETIC_PREFIX, FLUX_HELPER_PREFIX))
        )

    n_sp = (source_model.n_species, net_model.n_species)
    # Fold signed-flux fragments back to their source reaction so the count
    # compares original topology, not the RHS-identical .net re-encoding.
    n_rxn = (_effective_n_reactions(source_model), _effective_n_reactions(net_model))
    n_par = (_n_params(source_model), _n_params(net_model))
    n_obs = (source_model.n_observables, net_model.n_observables)

    for label, (a, b) in (
        ("species", n_sp),
        ("reactions", n_rxn),
        ("parameters", n_par),
        ("observables", n_obs),
    ):
        if a != b:
            mismatches.append(f"{label} count {a} != {b}")

    sig_a = _stoich_signatures(source_model)
    sig_b = _stoich_signatures(net_model)
    if sig_a != sig_b:
        only_a = sig_a - sig_b
        only_b = sig_b - sig_a
        if only_a:
            mismatches.append(f"{sum(only_a.values())} reaction topolog(ies) only in source")
        if only_b:
            mismatches.append(f"{sum(only_b.values())} reaction topolog(ies) only in output")

    return StructuralReport(
        passed=not mismatches,
        n_species=n_sp,
        n_reactions=n_rxn,
        n_parameters=n_par,
        n_observables=n_obs,
        mismatches=mismatches,
    )


# ─── Orchestrator ─────────────────────────────────────────────────────────


def sbml_to_net(
    sbml_path: str | Path,
    out_path: str | Path | None = None,
    *,
    validate: str | None = "L2",
    strict: bool = True,
    sedml: str | Path | None = None,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
    rhs_tol: float = 1e-6,
) -> ConversionReport:
    """Convert an SBML model to a BioNetGen ``.net`` network.

    Parameters
    ----------
    sbml_path : str | Path
        Path to the source SBML ``.xml``.
    out_path : str | Path | None
        Where to write the ``.net``. If None, the text is returned in the report
        but no file is written.
    validate : {"L1", "L2", "full", None}
        ``"L2"`` (default) runs the lightweight structural check **plus** two
        trajectory-free faithfulness probes at the initial and a nonlinear-probe
        state (no integration): a direct ODE-RHS identity self-check
        (:attr:`~ConversionReport.max_rhs_delta`) that reloads the ``.net`` and
        confirms it reproduces the source right-hand side, and an assignment-rule
        report check (:attr:`~ConversionReport.max_ar_report_delta`) that compares
        what the two models *report* for each AssignmentRule-target species emitted
        ``fixed`` (its live value is a Simulator report transform the reloaded
        network rebuilds from the ``.net``'s structure, #515 — invisible to the RHS
        probe because its ``dy/dt`` is zero in both models; GH #18). Together
        they close the GH #223 / GH #18 silent-loss hole: a network whose constructs
        were emitted as constants the ``.net`` cannot drive (assignment/rate-rule
        forcing the writer can't carry) is flagged (``strict=True`` raises;
        ``strict=False`` warns and records
        :attr:`ConversionReport.rhs_faithful` False). ``"L1"`` runs only the
        structural-equivalence check (counts + topology). ``"full"`` runs the
        complete L0–L4 conversion-validation ladder (GH #217) and gates on the
        hard levels L0–L3 — "convert *and prove faithful*"; the L4 symbolic check
        is recorded but never blocks. ``None`` skips validation. A ``"full"``
        gate's verdict is attached as :attr:`ConversionReport.validation` and
        decides :attr:`ConversionReport.ok`.
    strict : bool
        Raise :class:`bngsim.ConversionError` on constructs the ``.net`` text
        format cannot carry faithfully. ``False`` downgrades to a
        :class:`bngsim.ConversionWarning` and emits a best-effort network.
    sedml : str | Path | None
        The SED-ML simulation protocol that accompanies the SBML (the COMBINE
        sidecar). When given, its time course is parsed and the ``"full"`` gate's
        L3 numerical comparison runs over the model's *own* horizon — the exact
        mirror of :func:`net_to_sbml`'s ``bngl=`` (avoids the blanket-grid
        stiff-hang and exercises the trajectory the modeller actually ran). The
        parsed :class:`~bngsim.convert.ProtocolSpec` is attached to the report.
    t_span, n_points : tuple[float, float], int
        Fallback time grid for the ``"full"`` gate's L3 comparison when no
        ``sedml`` protocol horizon is available (ignored for other ``validate``
        values).
    rhs_tol : float
        Scale-relative tolerance for the ``"full"`` gate's L2 ODE-RHS identity.

    Returns
    -------
    ConversionReport
    """
    from bngsim._exceptions import ConversionError, ConversionWarning, ModelError
    from bngsim._model import Model

    sbml_path = Path(sbml_path)
    model = Model.from_sbml(sbml_path)

    protocol = None
    if sedml is not None:
        protocol = read_sedml_protocol(sedml)

    caps = capability_report(model)
    header = f"sbml2net: {sbml_path.name}"
    net_text = write_net(model, out_path, strict=strict, header=header, model_name=sbml_path.name)

    structural: StructuralReport | None = None
    rhs_faithful: bool | None = None
    max_rhs_delta: float | None = None
    max_ar_report_delta: float | None = None
    validation: ConversionValidationReport | None = None
    if validate in ("L1", "L2"):
        # Reload through the .net reader to validate the writer↔reader round-trip.
        # A best-effort (strict=False) emission of an already-lossy model can write
        # a .net the reader cannot load (e.g. a function referencing a rateOf
        # csymbol — caught by capability_report above); record that as not-faithful
        # rather than propagating the loader error.
        try:
            if out_path is not None:
                net_model = Model.from_net(out_path)
            else:
                with tempfile.NamedTemporaryFile("w", suffix=".net", delete=False) as tmp:
                    tmp.write(net_text)
                    tmp_path = tmp.name
                try:
                    net_model = Model.from_net(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
        except ModelError:
            if not caps["lossy"]:
                raise  # a faithful model that still won't reload is a real defect
            return ConversionReport(
                source=str(sbml_path),
                out_path=str(out_path) if out_path is not None else None,
                output_text=net_text,
                n_species=model.n_species,
                n_reactions=model.n_reactions,
                n_parameters=model.n_parameters,
                n_observables=model.n_observables,
                dropped=caps["dropped"],
                lossy=caps["lossy"],
                bngsim_only=caps["bngsim_only"],
                rhs_faithful=False,
                protocol=protocol,
            )
        structural = validate_structural_l1(model, net_model)
        if validate == "L2":
            # Does the reloaded .net reproduce the source's *reported ODE dynamics*?
            # Two independent probes at the initial + a nonlinear-perturbation state:
            #   • max_rhs_delta — direct ODE-RHS identity (species order is preserved,
            #     so index-aligned). Catches forcing the flat .net cannot carry
            #     (assignment/rate-rule constructs frozen to constants) that the
            #     structural counts miss — the GH #223 silent-loss class.
            #   • max_ar_report_delta — whether every AssignmentRule-target species
            #     (emitted ``fixed``, its live value applied as a Simulator report
            #     transform) is *reported* the same by the reloaded network, which
            #     rebuilds that transform from the .net's structure (#515). The RHS
            #     probe is *blind* to this: the frozen species has dy/dt = 0 in both
            #     models, so a lost rule gives max_rhs_delta ≈ 0 yet the .net reports
            #     the species frozen at its initial value (GH #18).
            max_rhs_delta = _max_rhs_delta(model, net_model)
            max_ar_report_delta = _ar_report_delta(model, net_model)
            rhs_faithful = max_rhs_delta <= rhs_tol and max_ar_report_delta <= rhs_tol
            if not rhs_faithful:
                if max_rhs_delta > rhs_tol:
                    note = (
                        f"the round-tripped .net does not reproduce the source ODE "
                        f"right-hand side (max scale-relative |Δ dy/dt| = "
                        f"{max_rhs_delta:.2e} > {rhs_tol:.0e}) — a construct was emitted "
                        "as a constant the flat .net cannot drive (e.g. assignment-rule "
                        "or time-dependent forcing); the network is not faithful"
                    )
                else:
                    note = (
                        f"one or more assignment-rule-target species are reported "
                        f"differently by the round-tripped .net (max scale-relative "
                        f"|Δ| = {max_ar_report_delta:.2e} > {rhs_tol:.0e}) — the rule's "
                        "live value is a run-time report transform the reloaded network "
                        "could not rebuild from the .net, so it reports these species "
                        "frozen at their initial value; the network reports them "
                        "unfaithfully"
                    )
                if strict:
                    raise ConversionError(
                        f"{sbml_path.name}: {note}. Pass strict=False "
                        "(--allow-lossy) to emit the best-effort network anyway."
                    )
                warnings.warn(
                    f"{sbml_path.name}: lossy conversion — {note}",
                    ConversionWarning,
                    stacklevel=2,
                )
    elif validate == "full":
        validation = grade_conversion(
            "sbml2net",
            model,
            net_text,
            levels="all",
            strict=strict,
            caps=caps,
            source=str(sbml_path),
            source_stem=sbml_path.stem,
            t_span=t_span,
            n_points=n_points,
            rhs_tol=rhs_tol,
            protocol=protocol,
        )
    elif validate is not None:
        raise ValueError(f"unknown validate={validate!r}; expected None, 'L1', 'L2', or 'full'")

    return ConversionReport(
        source=str(sbml_path),
        out_path=str(out_path) if out_path is not None else None,
        output_text=net_text,
        n_species=model.n_species,
        n_reactions=model.n_reactions,
        n_parameters=model.n_parameters,
        n_observables=model.n_observables,
        dropped=caps["dropped"],
        lossy=caps["lossy"],
        bngsim_only=caps["bngsim_only"],
        structural=structural,
        max_rhs_delta=max_rhs_delta,
        max_ar_report_delta=max_ar_report_delta,
        rhs_faithful=rhs_faithful,
        validation=validation,
        protocol=protocol,
    )


# ─── SBML→cBNGL orchestrator (GH #224) ────────────────────────────────────


def sbml_to_bngl(
    sbml_path: str | Path,
    out_path: str | Path | None = None,
    *,
    strict: bool = True,
    validate: str | None = None,
    rhs_tol: float = 1e-6,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
) -> ConversionReport:
    """Convert an SBML model to a compartmental BNGL (``.bngl``) model block.

    Mirrors :func:`sbml_to_net`, but targets **cBNGL** so static compartment
    volumes are recovered (GH #224 deliverable 1): a non-unit-volume model that
    plain ``.net`` refuses round-trips faithfully through
    ``BNG2.pl generate_network`` → :func:`bngsim.Model.from_net`. The emitted text
    is the ``begin model`` … ``end model`` block; when the source SBML carries
    **fixed-time events** they are translated into a trailing ``begin actions``
    block (#224 phase 2 — ``simulate`` phases with ``setConcentration`` state
    changes at each fire time). State-triggered events (and non-constant trigger
    times / delays / assignment values) have no actions form and are refused
    fail-loud.

    Parameters
    ----------
    sbml_path : str | Path
        Path to the source SBML ``.xml``.
    out_path : str | Path | None
        Where to write the ``.bngl``. If None, the text is returned in the report
        but no file is written.
    strict : bool
        Raise :class:`bngsim.ConversionError` on constructs deliverable 1 cannot
        carry — state-triggered events, live/time-varying volumes,
        cross-compartment reactant reactions, Michaelis–Menten kinetics,
        amount-valued non-unit species (see
        :func:`bngsim.convert.bngl_capability_report`). ``False`` downgrades to a
        :class:`bngsim.ConversionWarning` and emits a best-effort model.
    validate : {None, "bng2"}
        ``None`` (default): faithfulness rests on the capability check (no fatal
        ``lossy`` notes) — fast, no external tool. ``"bng2"``: additionally run the
        **BNG2.pl round-trip oracle** — flatten the emitted ``.bngl`` via
        ``BNG2.pl generate_network``, reload through :func:`bngsim.Model.from_net`,
        and compare the ODE right-hand side to the source's (probed at several t>0
        instants), and what each species an assignment rule sets reports, which
        the right-hand side cannot show (issue #1400). Sets
        :attr:`ConversionReport.rhs_faithful` / ``max_rhs_delta`` /
        ``max_ar_report_delta``; a divergence in either raises
        :class:`bngsim.ConversionError` under ``strict`` (else warns). Needs
        ``BNG2.pl`` on ``$BNGPATH`` or ``PATH`` (raises if absent, times out, or the
        ``.bngl`` does not build — faithfulness unprovable). This is the
        authoritative cBNGL gate; there is no in-tree cBNGL reader by design.
    rhs_tol : float
        Scale-relative RHS tolerance for ``validate="bng2"`` (default 1e-6).
    t_span, n_points : tuple[float, float], int
        The horizon for the events→actions ``simulate`` phases (the trailing
        phase runs to ``t_span[1]``; ignored when the model has no events).

    Returns
    -------
    ConversionReport
        With ``validate="bng2"``, :attr:`rhs_faithful` / :attr:`max_rhs_delta` /
        :attr:`max_ar_report_delta` carry the oracle verdict; otherwise they are
        None and ``ok`` reflects the capability check (no fatal ``lossy`` notes).
    """
    from bngsim._exceptions import ConversionError, ConversionWarning
    from bngsim._model import Model
    from bngsim.convert._bngl_writer import _EVENT_GENERIC
    from bngsim.convert._events import sbml_events_to_protocol

    if validate not in (None, "bng2"):
        raise ValueError(f"unknown validate={validate!r}; expected None or 'bng2'")

    sbml_path = Path(sbml_path)
    model = Model.from_sbml(sbml_path)

    # Classify the source SBML's events (the loaded Model exposes only a count):
    # fixed-time events become a BNGL actions block; state-triggered / non-constant
    # events are the specific refusals the capability report then carries.
    protocol = None
    event_override: Any = _EVENT_GENERIC
    if int(getattr(model._core, "n_events", 0) or 0):
        protocol, event_override = sbml_events_to_protocol(
            sbml_path, model, t_span=t_span, n_points=n_points
        )
        # event_override is now [] when all events were carried into actions,
        # else the specific state-triggered / non-constant refusals.

    caps = bngl_capability_report(model, event_override=event_override)
    bngl_text = write_bngl(
        model,
        out_path,
        strict=strict,
        model_name=sbml_path.name,
        protocol=protocol,
        event_override=event_override,
    )

    rhs_faithful: bool | None = None
    max_rhs_delta: float | None = None
    max_ar_report_delta: float | None = None
    if validate == "bng2":
        # Authoritative cBNGL gate: round-trip the emitted .bngl through BNG2.pl and
        # compare the reloaded ODE RHS to the source's. Skip when capability already
        # found the model lossy (strict would have raised in write_bngl; a lenient
        # best-effort .bngl is not expected to build faithfully).
        from bngsim.convert._bng2 import roundtrip_deltas

        if not caps["lossy"]:
            # And what each species an assignment rule sets reports, which the
            # right-hand side cannot see (dy/dt = 0 on both sides; issue #1400).
            max_rhs_delta, max_ar_report_delta, _ = roundtrip_deltas(
                model, bngl_text, stem=sbml_path.stem
            )
            rhs_faithful = max_rhs_delta <= rhs_tol and max_ar_report_delta <= rhs_tol
            if not rhs_faithful:
                if max_rhs_delta > rhs_tol:
                    note = (
                        f"the BNG2.pl round-trip does not reproduce the source ODE "
                        f"right-hand side (max scale-relative |Δ dy/dt| = "
                        f"{max_rhs_delta:.2e} > {rhs_tol:.0e}); the cBNGL is not faithful"
                    )
                else:
                    note = (
                        f"the BNG2.pl round-trip does not report a species an assignment "
                        f"rule sets as the source does (max scale-relative |Δ| = "
                        f"{max_ar_report_delta:.2e} > {rhs_tol:.0e}); the cBNGL is not "
                        "faithful"
                    )
                if strict:
                    raise ConversionError(f"{sbml_path.name}: {note}.")
                warnings.warn(
                    f"{sbml_path.name}: lossy conversion — {note}",
                    ConversionWarning,
                    stacklevel=2,
                )

    return ConversionReport(
        source=str(sbml_path),
        out_path=str(out_path) if out_path is not None else None,
        output_text=bngl_text,
        n_species=model.n_species,
        n_reactions=model.n_reactions,
        n_parameters=model.n_parameters,
        n_observables=model.n_observables,
        dropped=caps["dropped"],
        lossy=caps["lossy"],
        bngsim_only=caps.get("bngsim_only", []),
        max_rhs_delta=max_rhs_delta,
        max_ar_report_delta=max_ar_report_delta,
        rhs_faithful=rhs_faithful,
        protocol=protocol,
    )


# ─── net→SBML round-trip validation ───────────────────────────────────────


def validate_roundtrip(
    source_model: Model, reloaded_model: Model, *, rhs_tol: float = 1e-6
) -> StructuralReport:
    """Validate a ``.net``→SBML conversion by reloading and comparing.

    The structural gate is **species count + reaction count + reaction
    topology** plus an **ODE right-hand-side** numerical check — the
    equivalence that actually matters and that seeds #217's L2
    round-trip-identity gate. Observable and parameter counts are *recorded but
    not gated*: ``from_net`` honors a model's explicit ``begin groups`` while
    ``from_sbml`` auto-reports every species as an observable (and stores a
    summing rule as a time-varying parameter), so a strict count comparison
    across the two loaders would flag a benign labeling difference, not a
    conversion defect.
    """
    mismatches: list[str] = []

    n_sp = (source_model.n_species, reloaded_model.n_species)
    n_rxn = (source_model.n_reactions, reloaded_model.n_reactions)
    n_par = (source_model.n_parameters, reloaded_model.n_parameters)
    n_obs = (source_model.n_observables, reloaded_model.n_observables)

    if n_sp[0] != n_sp[1]:
        mismatches.append(f"species count {n_sp[0]} != {n_sp[1]}")
    if n_rxn[0] != n_rxn[1]:
        mismatches.append(f"reactions count {n_rxn[0]} != {n_rxn[1]}")

    # Topology over *dynamic* species only: a fixed ($) species is a reactant in
    # the .net but `from_sbml` folds a constant boundary species into the rate
    # law (0-reactant functional reaction), so comparing it would flag a benign
    # cross-loader representation difference. Species order is preserved through
    # the round-trip, so index→species agrees across both models.
    sig_a = _dynamic_stoich_signatures(source_model)
    sig_b = _dynamic_stoich_signatures(reloaded_model)
    if sig_a != sig_b:
        only_a = sig_a - sig_b
        only_b = sig_b - sig_a
        if only_a:
            mismatches.append(f"{sum(only_a.values())} reaction topolog(ies) only in source")
        if only_b:
            mismatches.append(f"{sum(only_b.values())} reaction topolog(ies) only in output")

    if not mismatches:
        delta = _max_rhs_delta(source_model, reloaded_model)
        if not (delta <= rhs_tol):
            mismatches.append(f"ODE RHS differs (max scale-relative |Δ| = {delta:.2e})")

    return StructuralReport(
        passed=not mismatches,
        n_species=n_sp,
        n_reactions=n_rxn,
        n_parameters=n_par,
        n_observables=n_obs,
        mismatches=mismatches,
    )


def _split_species_line(body: str) -> tuple[str, str]:
    """A ``.bngl`` seed-species line, its leading index gone: the pattern, up to
    the first blank outside parentheses, and the amount after it."""
    depth = 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch.isspace() and depth == 0:
            return body[:i], body[i:].strip()
    return body, ""


def _declared(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]] | None:
    """A ``.bngl``'s parameters, ``(name, value)``, and seed species,
    ``(pattern, amount)``, as its blocks declare them; ``None`` where a line
    is continued, which this reading does not join."""
    params: list[tuple[str, str]] = []
    seeds: list[tuple[str, str]] = []
    block: str | None = None
    for physical in text.splitlines():
        line = physical.split("#", 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("begin "):
            name = " ".join(low.split()[1:])
            block = name if name in ("parameters", "seed species", "species") else None
            continue
        if low.startswith("end "):
            block = None
            continue
        if block is None:
            continue
        if line.endswith("\\"):
            return None
        body = re.sub(r"^\d+\s+", "", line)
        if block == "parameters":
            m = re.match(r"([A-Za-z_]\w*)\s*(?:=\s*)?(\S.*)$", body)
            if m is not None:
                params.append((m.group(1), m.group(2).strip()))
        else:
            pattern, amount = _split_species_line(body)
            if amount:
                seeds.append((pattern, amount))
    return params, seeds


def _not_as_generated(model: Any, bngl: Path, protocol: ProtocolSpec | None) -> list[str]:
    """How the ``.net`` *model* was loaded from differs from the network
    *bngl* generates (issue #1683): each parameter declared as a number, and
    each seed species, with the actions before the file's ``generate_network``
    applied, against the ``.net``'s own value. BNG2.pl writes the ``.net``
    again at a ``simulate`` with no ``suffix`` once a run or an action has
    moved the state, with the parameters and amounts of that moment; nothing
    else in the file says so. Empty where they agree, or where the file says
    something this reading cannot follow (a continued line, a reset or a value
    that does not evaluate before generation): a ``.net`` is refused only on
    a difference shown."""
    from bngsim._bngl_loader import _species_name, _value

    try:
        text = bngl.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    declared = _declared(text)
    if declared is None:
        return []
    params, seeds = declared
    # A .bngl with no rules generates no network: it carries a protocol for a
    # .net made some other way, whose species it declares nothing about.
    generates = re.search(r"^\s*begin\s+reaction\s+rules\b", text, re.M) is not None

    def close(a: float, b: float) -> bool:
        # BNG2.pl writes a value it computed to 15 significant digits.
        return abs(a - b) <= 1e-12 * max(abs(a), abs(b)) or a == b

    before = protocol.steps[: protocol.generation_step or 0] if protocol is not None else ()
    want_params: dict[str, float] = {}
    for pname, declared_text in params:
        try:
            want_params[pname] = float(declared_text)
        except ValueError:
            continue  # an expression: it follows the parameters it reads
    sets: list[StateChange] = []
    for step in before:
        if not isinstance(step, StateChange):
            continue
        if step.kind == "set_parameter" and step.value is not None and step.target:
            want_params[step.target] = float(step.value)
        elif step.kind in ("set_concentration", "add_concentration") and step.target:
            sets.append(step)
        else:
            return []
    out: list[str] = []
    names = set(model.param_names)
    for name, value in want_params.items():
        if name in names:
            held = float(model.get_param(name))
            if not close(held, value):
                out.append(f"parameter {name} is {held:.15g} there, {value:.15g} as declared")
    if out:
        return out  # the seed amounts read the parameters: judged once they agree
    want_species: dict[str, float] = {}
    every_seed_read = True
    for pattern, amount in seeds:
        sname = _species_name(model, pattern)
        declared_value = _value(model, amount)
        if sname is not None and declared_value is not None:
            want_species[sname] = declared_value
        else:
            every_seed_read = False
    for step in sets:
        sname = _species_name(model, step.target or "")
        set_value = (
            float(step.value) if step.value is not None else _value(model, step.value_expr or "")
        )
        if sname is None or set_value is None:
            return []
        base = want_species.get(sname, 0.0) if step.kind == "add_concentration" else 0.0
        want_species[sname] = base + set_value
    for sname, want in want_species.items():
        held = float(model.get_concentration(sname))
        if not close(held, want):
            out.append(f"{sname} is {held:.15g} there, {want:.15g} as declared")
    # A species the network generated holds 0 at generation; one a run filled
    # says the file was written after it, where the seeds happen to agree.
    # Judged only where the .bngl generates this network and every seed was
    # read: a seed this reading could not match would pass for a filled one,
    # and a .bngl of actions alone, beside a .net made another way, declares
    # no seed at all.
    if every_seed_read and seeds and generates:
        for sname in model.species_names:
            if sname in want_species:
                continue
            held = float(model.get_concentration(sname))
            if held != 0.0:
                out.append(f"{sname} is {held:.15g} there, 0 as generated")
    return out


def _not_generated_notes(report: ConversionReport) -> list[str]:
    """The note :func:`net_to_sbml` leaves where the ``.net`` is not the
    network its ``.bngl`` generates (issue #1683), for the callers that write
    a protocol onto the model and refuse it."""
    return [note for note in report.lossy if "is not the network" in note]


def net_to_sbml(
    net_path: str | Path,
    out_path: str | Path | None = None,
    *,
    validate: str | None = "L1",
    strict: bool = True,
    bngl: str | Path | None = None,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
    rhs_tol: float = 1e-6,
) -> ConversionReport:
    """Convert a BioNetGen ``.net`` network to SBML (Level 3 Version 2).

    The reverse of :func:`sbml_to_net`. SBML can express amount/concentration
    semantics and non-unit compartment volumes that the plain ``.net`` text
    cannot, so this is the more faithful half of the round-trip: each kinetic
    law is scaled by its reaction's compartment volume, so static non-unit
    volumes (including cross-compartment reactions) round-trip RHS-exact (see
    :func:`bngsim.convert.write_sbml`).

    Parameters
    ----------
    net_path : str | Path
        Path to the source ``.net`` network.
    out_path : str | Path | None
        Where to write the ``.xml``. If None, the text is returned in the report
        but no file is written.
    validate : {"L1", "full", None}
        ``"L1"`` (default) reloads the emitted SBML and runs
        :func:`validate_roundtrip` (structural + ODE-RHS equivalence). ``"full"``
        runs the complete L0–L4 conversion-validation ladder (GH #217) and gates
        on the hard levels L0–L3 — "convert *and prove faithful*"; the L4
        symbolic check is recorded but never blocks. ``None`` skips validation. A
        ``"full"`` gate's verdict is attached as
        :attr:`ConversionReport.validation` and decides
        :attr:`ConversionReport.ok`.
    strict : bool
        Raise :class:`bngsim.ConversionError` on constructs SBML cannot carry
        faithfully (live/time-varying volumes, ``tfun`` table functions, …).
        ``False`` downgrades to a :class:`bngsim.ConversionWarning` and emits a
        best-effort document.
    bngl : str | Path | None
        The source ``.bngl`` the ``.net`` was generated from. When given, its
        ``simulate`` protocol is parsed and the ``"full"`` gate's L3 numerical
        comparison runs over the model's *own* horizon (avoiding the blanket-grid
        stiff-hang and exercising the trajectory the modeller actually ran). The
        parsed :class:`~bngsim.convert.ProtocolSpec` is attached to the report.
        The ``.net`` must be the network as ``generate_network`` wrote it: its
        parameters and seed species are checked against what the ``.bngl``
        declares (issue #1683). ``BNG2.pl model.bngl`` rewrites ``model.net``
        at a ``simulate`` with no ``suffix`` once a run or an action has moved
        the state, and the protocol, replayed on top of that state, starts
        from the wrong one; such a ``.net`` is warned of and noted in
        ``lossy`` here, and refused by :func:`net_to_omex` and
        ``bngsim-net2sbml --sidecar`` unless ``strict=False`` /
        ``--allow-lossy``.
    t_span, n_points : tuple[float, float], int
        Fallback time grid for the ``"full"`` gate's L3 comparison when no
        ``bngl`` protocol horizon is available (ignored for other ``validate``
        values).
    rhs_tol : float
        Scale-relative tolerance for the ``"full"`` gate's L2 ODE-RHS identity.

    Returns
    -------
    ConversionReport
    """
    from bngsim._exceptions import ConversionWarning
    from bngsim._model import Model

    net_path = Path(net_path)
    model = Model.from_net(net_path)

    protocol = None
    not_generated: list[str] = []
    bngl_name = Path(bngl).name if bngl is not None else ""
    if bngl is not None:
        from bngsim.convert._protocol import parse_bngl_protocol

        protocol = parse_bngl_protocol(bngl, strict=strict)
        not_generated = _not_as_generated(model, Path(bngl), protocol)
        if not_generated:
            message = (
                f"{net_path.name} is not the network {bngl_name} generates: "
                + "; ".join(not_generated[:3])
                + (f"; and {len(not_generated) - 3} more" if len(not_generated) > 3 else "")
                + ". BNG2.pl rewrites the .net at a simulate with no suffix once a run "
                "or an action has moved the state, and the protocol replayed on that "
                "state starts from the wrong one (issue #1683). Write the network as "
                "generate_network does with bngsim.Model.from_bngl(bngl, net_out=...), "
                "or give every simulate a suffix"
            )
            # The SBML is the .net as given, which is a model in its own right;
            # the state matters where a protocol is written onto it, which
            # net_to_omex and the sidecar refuse under strict (the note below).
            from bngsim._simulator import _stacklevel_outside_bngsim

            warnings.warn(message, ConversionWarning, stacklevel=_stacklevel_outside_bngsim())

    caps = sbml_capability_report(model)
    if not_generated:
        caps["lossy"] = list(caps["lossy"]) + [message]
    sbml_text = write_sbml(model, out_path, strict=strict, model_id=net_path.stem)

    structural: StructuralReport | None = None
    max_delta: float | None = None
    validation: ConversionValidationReport | None = None
    if validate == "L1":
        if out_path is not None:
            sbml_model = Model.from_sbml(out_path)
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as tmp:
                tmp.write(sbml_text)
                tmp_path = tmp.name
            try:
                sbml_model = Model.from_sbml(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        structural = validate_roundtrip(model, sbml_model)
        max_delta = _max_rhs_delta(model, sbml_model)
    elif validate == "full":
        validation = grade_conversion(
            "net2sbml",
            model,
            sbml_text,
            levels="all",
            strict=strict,
            caps=caps,
            source=str(net_path),
            source_stem=net_path.stem,
            t_span=t_span,
            n_points=n_points,
            rhs_tol=rhs_tol,
            protocol=protocol,
        )
    elif validate is not None:
        raise ValueError(f"unknown validate={validate!r}; expected None, 'L1', or 'full'")

    return ConversionReport(
        source=str(net_path),
        out_path=str(out_path) if out_path is not None else None,
        output_text=sbml_text,
        n_species=model.n_species,
        n_reactions=model.n_reactions,
        n_parameters=model.n_parameters,
        n_observables=model.n_observables,
        dropped=caps["dropped"],
        lossy=caps["lossy"],
        structural=structural,
        max_rhs_delta=max_delta,
        validation=validation,
        protocol=protocol,
    )


# ─── OMEX packaging orchestrator ──────────────────────────────────────────


_GATE_TO_VALIDATE = {"none": None, "L1": "L1", "full": "full"}


def net_to_omex(
    net_path: str | Path,
    out_path: str | Path,
    *,
    bngl: str | Path | None = None,
    protocol: EvaluationSpec | None = None,
    gate: str = "full",
    include_source: bool = True,
    provenance: bool = True,
    created: str | None = None,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
    strict: bool = True,
) -> ConversionReport:
    """Package a BioNetGen ``.net`` network as a COMBINE archive (``.omex``).

    Converts the network to SBML (the BioModels-standard model carrier, GH #216),
    bundles a SED-ML simulation protocol (GH #218), and writes both plus a
    generated ``manifest.xml`` into one ``.omex`` zip — the standard, *verified
    faithful* consumer container (GH #211, Option 3).

    The protocol carried depends on what is supplied, in precedence order:

    1. an explicit ``protocol`` :class:`~bngsim.EvaluationSpec` → a single uniform
       time course (GH #218);
    2. a source ``bngl`` → its **whole** actions block (every ``simulate`` /
       ``parameter_scan`` with its accumulated overrides) via
       :func:`~bngsim.convert.write_sedml_protocol` — the modeller's actual
       experiment, not a synthesized default;
    3. neither → a default uniform time course over every observable.

    Parameters
    ----------
    net_path : str | Path
        Source ``.net`` network.
    out_path : str | Path
        Destination ``.omex`` archive.
    bngl : str | Path | None
        The source ``.bngl`` the ``.net`` was generated from. Supplies the real
        simulation protocol carried into the SED-ML *and* the L3 horizon for the
        gate (see :func:`net_to_sbml`).
    protocol : EvaluationSpec | None
        An explicit single-run protocol; takes precedence over ``bngl``.
    gate : {"full", "L1", "none"}
        How hard to validate the ``.net``→SBML conversion before packaging
        (default ``"full"`` — the OMEX is the faithful-deliverable container, so
        it ships the L0–L4 verdict). The verdict is attached to the returned
        report; :attr:`ConversionReport.ok` is False (CLI exit 1) on a hard-gate
        failure.
    include_source : bool
        When True (default), also bundle the **original source files** into the
        archive for provenance/completeness: the ``.net`` (a secondary, non-master
        model entry) and — when given — the rule-based ``.bngl`` (a ``source``
        entry; ``from_net`` cannot read rule-based BNGL, so it is provenance, not a
        dispatchable model). The SBML stays the ``master`` curated entry, so this
        is non-breaking for SBML-only consumers and lets a published archive carry
        the modeller's actual rule-based formulation, not just the flattened SBML.
        COMBINE/BioModels explicitly accept such supporting files.
    provenance : bool
        When True (default), record how the archive was produced: a COMBINE-standard
        ``metadata.rdf`` (``dcterms`` creator = ``bngsim <version>``, created date,
        description) that BioModels/COMBINE tools read, plus a human/machine-readable
        ``bngsim-conversion.json`` carrying the full faithfulness verdict (gate level
        and per-level L0–L4 result, ``ok`` / ``rhs_faithful`` / ``max_rhs_delta``,
        dropped/lossy notes, source→target, counts). Makes the *verified-faithful*
        claim auditable by anyone who opens the archive.
    created : str | None
        Timestamp (ISO-8601 / W3CDTF) stamped into the provenance **and onto every
        zip entry**, which is what makes the archive byte-reproducible: two builds
        from identical inputs and the same ``created`` are byte-identical, however
        far apart they run. ``None`` (default) uses the current time for both, so
        the archive is *not* reproducible — the entry headers carry the wall clock
        to the second. Applies even with ``provenance=False``, where it reaches
        only the entry headers. Any UTC offset is dropped from the entry stamp (a
        zip header has no timezone field); ``metadata.rdf`` keeps it in full.
    t_span, n_points : tuple[float, float], int
        Time grid for the default protocol / the gate's L3 fallback when no
        ``bngl`` horizon is available.
    strict : bool
        Passed through to the ``.net``→SBML conversion and the ``.bngl`` parse
        (refuse vs. best-effort on unrepresentable constructs).

    Returns
    -------
    ConversionReport
        The underlying ``.net``→SBML report (with the L0–L4 ``validation`` and the
        parsed ``protocol``), ``out_path`` pointing at the ``.omex`` archive.
    """
    from bngsim.convert._omex import _zip_date_time, write_omex
    from bngsim.convert._sedml import default_protocol as _default_protocol
    from bngsim.convert._sedml import write_sedml, write_sedml_protocol

    if gate not in _GATE_TO_VALIDATE:
        raise ValueError(f"unknown gate={gate!r}; expected one of {sorted(_GATE_TO_VALIDATE)}")
    # Parsed here only to reject a malformed `created` now rather than after the
    # whole conversion and gate have run; write_omex parses it again for real.
    _zip_date_time(created)

    net_path = Path(net_path)
    out_path = Path(out_path)

    # Reuse the faithful network channel (gated); keep the SBML in memory. The
    # bngl is parsed once here and reused for both the gate horizon and the SED-ML.
    report = net_to_sbml(
        net_path,
        out_path=None,
        validate=_GATE_TO_VALIDATE[gate],
        strict=strict,
        bngl=bngl,
        t_span=t_span,
        n_points=n_points,
    )
    # The protocol is written onto this network's state: a .net BNG2.pl
    # rewrote after a run is refused here, where net_to_sbml only said so
    # (issue #1683); strict=False packs it with the warning and the note.
    if strict:
        from bngsim._exceptions import ConversionError

        for note in _not_generated_notes(report):
            raise ConversionError(note)
    sbml_text = report.output_text
    parsed_protocol = report.protocol

    # The SED-ML protocol references the SBML model inside the archive.
    sbml_location = "./model.xml"
    loc = sbml_location.lstrip("./")
    # Every SED-ML target is an id in this SBML (and, for a species pattern, the
    # network species the model resolves it to).
    from bngsim._model import Model as _Model

    network = _Model.from_net(str(net_path))
    # What the SED-ML's standard elements cannot hold, the writers warn of;
    # each is recorded in the report and in the archive, for whoever runs the
    # standard elements with another tool (issue #1684).
    sedml_lossy: list[str] = []
    if protocol is not None:
        with _recording(sedml_lossy):
            sedml_text = write_sedml(protocol, model_source=loc, sbml=sbml_text, model=network)
    elif parsed_protocol is not None and not parsed_protocol.is_empty:
        with _recording(sedml_lossy):
            sedml_text = write_sedml_protocol(
                parsed_protocol, model_source=loc, sbml=sbml_text, model=network
            )
    else:
        # No real protocol available — fabricate a runnable default, but warn
        # loudly and mark it as synthesized so it is never mistaken for the
        # modeller's protocol (GH #211 fidelity).
        why = (
            "the .bngl carried no simulate action"
            if bngl is not None
            else "no .bngl protocol source was supplied"
        )
        from bngsim._exceptions import ConversionWarning

        warnings.warn(
            f"no simulation protocol available ({why}); bundling a "
            "bngsim-generated DEFAULT uniform time course "
            f"(t={t_span[0]:g}..{t_span[1]:g}, {n_points} points, ODE) — this is a "
            "runnable placeholder, NOT the modeller's protocol",
            ConversionWarning,
            stacklevel=2,
        )
        default = _default_protocol(
            net_path,
            model_source=loc,
            model_format="sbml",
            t_span=t_span,
            n_points=n_points,
        )
        with _recording(sedml_lossy):
            sedml_text = write_sedml(
                default, model_source=loc, synthesized_default=True, sbml=sbml_text, model=network
            )

    # Bundle the original source files for provenance (the SBML stays master): the
    # flattened .net as a secondary model entry, and the rule-based .bngl — the
    # modeller's actual formulation — as a `source` entry. So a BioModels deposit
    # carries the real BNGL, not just the SBML projection.
    from bngsim.convert._protocol import _looks_like_path, _read_text

    source_net: bytes | None = None
    net_location = "./model.net"
    extra: list[tuple[str, str | bytes | Path, str | None]] = []
    if include_source:
        from bngsim.convert._omex import _FMT_BNGL

        source_net = net_path.read_bytes()
        net_location = "./" + net_path.name
        if bngl is not None:
            bngl_text = _read_text(bngl)
            bngl_name = Path(bngl).name if _looks_like_path(bngl) else "model.bngl"
            extra.append(("./" + bngl_name, bngl_text, _FMT_BNGL))

    # Provenance: how this archive was produced + the faithfulness verdict, both as
    # a COMBINE-standard metadata.rdf (tools read it) and a bngsim-conversion.json
    # (humans/auditors read it). Makes the "verified faithful" claim portable.
    metadata_rdf: str | None = None
    if provenance:
        import json as _json
        from datetime import datetime, timezone

        from bngsim import __version__ as _version
        from bngsim.convert._omex import _FMT_JSON, build_metadata_rdf

        ts = created or datetime.now(timezone.utc).isoformat(timespec="seconds")
        if bngl is None:
            protocol_source = None
        elif _looks_like_path(bngl):
            protocol_source = Path(bngl).name
        else:
            protocol_source = "<inline bngl>"
        record = {
            "tool": "bngsim",
            "version": _version,
            "created": ts,
            "conversion": {
                "direction": "net_to_omex",
                "source": net_path.name,
                "source_format": "bngnet",
                "target_format": "sbml",
                "protocol_source": protocol_source,
                "gate": gate,
                "ok": report.ok,
                "rhs_faithful": report.rhs_faithful,
                "max_rhs_delta": report.max_rhs_delta,
                "validation": (
                    [
                        {
                            "level": lv.level,
                            "name": lv.name,
                            "status": lv.status,
                            "detail": lv.detail,
                        }
                        for lv in report.validation.levels
                    ]
                    if report.validation is not None
                    else None
                ),
                "dropped": report.dropped,
                "lossy": report.lossy,
                # Of the SED-ML's standard elements, not the network: what they
                # do not hold of the protocol (issue #1684).
                "sedml_lossy": sedml_lossy,
            },
            "counts": {
                "species": report.n_species,
                "reactions": report.n_reactions,
                "parameters": report.n_parameters,
                "observables": report.n_observables,
            },
        }
        extra.append(("./bngsim-conversion.json", _json.dumps(record, indent=2), _FMT_JSON))
        metadata_rdf = build_metadata_rdf(
            creator=f"bngsim {_version}",
            created=ts,
            description=(
                f"Generated by bngsim {_version}: BioNetGen .net → SBML "
                f"(gate={gate}, {'PASS' if report.ok else 'FAIL'}). "
                + (
                    f"The SED-ML's standard elements do not hold {len(sedml_lossy)} "
                    f"part{'' if len(sedml_lossy) == 1 else 's'} of the protocol exactly "
                    "(sedml_lossy); its bngsim annotation does. "
                    if sedml_lossy
                    else ""
                )
                + "See bngsim-conversion.json for the faithfulness verdict."
            ),
        )

    write_omex(
        out_path,
        sbml=sbml_text,
        net=source_net,
        sedml=sedml_text,
        metadata=metadata_rdf,
        sbml_location=sbml_location,
        net_location=net_location,
        extra=extra or None,
        master="sbml",
        created=created,
    )

    return ConversionReport(
        source=str(net_path),
        out_path=str(out_path),
        output_text=sbml_text,
        n_species=report.n_species,
        n_reactions=report.n_reactions,
        n_parameters=report.n_parameters,
        n_observables=report.n_observables,
        dropped=report.dropped,
        lossy=report.lossy,
        validation=report.validation,
        protocol=parsed_protocol,
        sedml_lossy=sedml_lossy,
    )


@contextlib.contextmanager
def _recording(notes: list[str]):
    """Collect the :class:`~bngsim.ConversionWarning` messages raised inside
    into *notes*, and raise every warning again as it was raised: the caller
    sees what it saw before, and the archive keeps a record (issue #1684)."""
    from bngsim._exceptions import ConversionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        if issubclass(w.category, ConversionWarning):
            notes.append(str(w.message))
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)


def omex_to_net(
    omex_path: str | Path,
    out_path: str | Path | None = None,
    *,
    gate: str = "full",
    strict: bool = True,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
    actions_out: str | Path | None = None,
    write_actions: bool = True,
) -> ConversionReport:
    """Unpack a COMBINE archive (``.omex``) to a BioNetGen ``.net`` network.

    The reverse of :func:`net_to_omex`: read the archive's master SBML model and
    its SED-ML protocol sidecar, convert the SBML to a ``.net`` (GH #215), and use
    the carried protocol to drive the gate's L3 horizon — the model's *own*
    integrable time range, exactly as :func:`net_to_omex` uses the source ``.bngl``
    going the other way. This closes the round-trip container symmetry: a ``.omex``
    written by :func:`net_to_omex` reads back to an equivalent network here.

    Multi-experiment / multi-file protocols (GH #222). The archive's **whole**
    protocol is recovered, not just the master SED-ML entry's first experiment:
    every ``uniformTimeCourse`` / ``repeatedTask`` from **every** SED-ML file is
    composed (via :func:`~bngsim.convert.combine_protocols`) into one ordered
    :class:`~bngsim.convert.ProtocolSpec` and attached to the report. When more
    than one SED-ML file is present a note is emitted (the master alone no longer
    silently wins). The composed protocol is also emitted as a **``.bngl`` actions
    block** alongside the ``.net`` (``write_actions``) — the natural BNGL form of a
    multi-phase experiment (``simulate`` / ``setConcentration`` / ``setParameter``
    / ``resetConcentrations``), restoring the symmetry with :func:`net_to_omex`,
    which already carries the whole protocol forward. The gate's L3 horizon still
    uses the **representative** experiment (the first deterministic run — see
    :meth:`ProtocolSpec.primary_experiment`); gating every experiment's horizon is
    left as a deliberate non-goal (it risks a stiff-grid hang for no added
    faithfulness signal over the representative run).

    Parameters
    ----------
    omex_path : str | Path
        Source ``.omex`` archive (SBML model + SED-ML protocol + manifest).
    out_path : str | Path | None
        Where to write the ``.net``. If None, the text is returned in the report
        but no file is written.
    gate : {"full", "L1", "none"}
        How hard to validate the SBML→``.net`` conversion before accepting it
        (default ``"full"`` — the OMEX is the *verified faithful* container, so the
        unpack ships the L0–L4 verdict, mirroring :func:`net_to_omex`'s default).
        The verdict is attached to the returned report; :attr:`ConversionReport.ok`
        is False (CLI exit 1) on a hard-gate failure.
    strict : bool
        Passed through to the SBML→``.net`` conversion (refuse vs. best-effort on
        constructs plain ``.net`` cannot carry — events, live/AR volumes, MM).
    t_span, n_points : tuple[float, float], int
        Fallback L3 grid when the archive carries no SED-ML horizon.
    actions_out : str | Path | None
        Where to write the ``.bngl`` actions block. ``None`` (default) derives it
        from ``out_path`` (``<stem>.bngl``); ignored when ``write_actions`` is
        False or the archive carries no experiment.
    write_actions : bool
        Emit the composed protocol as a ``.bngl`` actions block (default True). No
        file is written when the archive carries no protocol, or when neither
        ``actions_out`` nor ``out_path`` gives a destination.

    Returns
    -------
    ConversionReport
        The SBML→``.net`` report (with the L0–L4 ``validation``, the **composed**
        multi-experiment SED-ML ``protocol``, and ``bngl_out`` pointing at the
        emitted actions block when written), ``out_path`` pointing at the written
        ``.net`` (or None when only the in-memory text was requested).
    """
    from bngsim._exceptions import ConversionError, ConversionWarning

    if gate not in _GATE_TO_VALIDATE:
        raise ValueError(f"unknown gate={gate!r}; expected one of {sorted(_GATE_TO_VALIDATE)}")

    omex_path = Path(omex_path)
    with read_omex(omex_path) as archive:
        model_entry = archive.master_model_entry()
        if model_entry is None:
            raise ConversionError(
                f"{omex_path.name}: archive has no model entry to convert to .net"
            )
        if model_entry.kind != "sbml":
            raise ConversionError(
                f"{omex_path.name}: master model is a {model_entry.kind!r}, not SBML; "
                "omex_to_net unpacks the SBML→.net direction (the archive already "
                "carries a network — extract it with bngsim-omex unpack)"
            )
        sbml_path = archive.path_of(model_entry)
        sed_entry = archive.master_sedml_entry()
        sedml_path = archive.path_of(sed_entry) if sed_entry is not None else None

        report = sbml_to_net(
            sbml_path,
            out_path,
            validate=_GATE_TO_VALIDATE[gate],
            strict=strict,
            sedml=sedml_path,
            t_span=t_span,
            n_points=n_points,
        )

        # Compose the WHOLE protocol — every experiment from every SED-ML file —
        # not just the master entry's first run that drove the gate horizon.
        sed_entries = archive.sedml_entries()
        if len(sed_entries) > 1:
            warnings.warn(
                f"{omex_path.name}: archive carries {len(sed_entries)} SED-ML files; "
                "composing every experiment from all of them into one protocol "
                "(previously only the master/first file was used)",
                ConversionWarning,
                stacklevel=2,
            )
        full_protocol = archive.load_full_protocol() if sed_entries else report.protocol

    if full_protocol is not None and not full_protocol.is_empty:
        report.protocol = full_protocol
        if write_actions:
            dest = actions_out
            if dest is None and out_path is not None:
                dest = Path(out_path).with_suffix(".bngl")
            if dest is not None:
                from bngsim.convert._protocol import write_bngl_protocol

                write_bngl_protocol(full_protocol, dest)
                report.bngl_out = str(dest)

    # Re-label the source so the report reads as an OMEX unpack, not a bare SBML.
    report.source = str(omex_path)
    return report
