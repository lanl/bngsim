"""Tests for bngsim.convert.sbml_to_net — SBML→.net converter (GH #215).

Covers the productized network-channel converter: structural (L1) round-trip
parity over a tracked SBML set, numerical equivalence of the converted network's
ODE right-hand side, the capability boundary (events dropped; volume-bearing
constructs refused under strict), and the CLI entry point.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert import (
    ConversionReport,
    sbml_to_net,
    validate_structural_l1,
    write_net,
)
from bngsim.convert._net_writer import (
    _psvs_needs_per_species_divide,
    _rateof_refs,
    capability_report,
)

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

_BNGSIM = Path(__file__).resolve().parents[2]
_EVENTS = _BNGSIM / "benchmarks" / "sbml_events"
_SMITH = _BNGSIM / "benchmarks" / "models" / "sbml" / "Smith2013_BIOMD0000000474_petab.xml"
_CORPUS = _BNGSIM / "parity_checks" / "rr_parity" / "models"


def _corpus_xml(model_id: str) -> Path:
    """Path to an rr_parity corpus model's SBML, skipping when absent."""
    hits = sorted(_CORPUS.glob(f"{model_id}/*.xml"))
    if not hits:
        pytest.skip(f"rr_parity corpus model {model_id} not present")
    return hits[0]


@pytest.fixture
def data_dir() -> Path:
    d = Path(__file__).resolve().parent.parent.parent / "tests" / "data"
    assert d.is_dir(), f"Test data directory not found: {d}"
    return d


def _event_model(name: str) -> Path:
    p = _EVENTS / name
    if not p.is_file():
        pytest.skip(f"tracked SBML not present: {p}")
    return p


# Curated tracked, non-lossy models. BIOMD3 is event-free (in tests/data); the
# BIOMD event models exercise the event-drop path while staying network-faithful.
_EVENT_MODELS = [
    "BIOMD0000000001.xml",
    "BIOMD0000000007.xml",
    "BIOMD0000000077.xml",
    "BIOMD0000000101.xml",
]


def _all_clean_sources(data_dir: Path) -> list[Path]:
    out = [data_dir / "BIOMD0000000003.xml"]
    out += [_EVENTS / n for n in _EVENT_MODELS]
    return [p for p in out if p.is_file()]


# ─── L1 structural round-trip ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "src",
    ["data/BIOMD0000000003.xml", *(f"events/{n}" for n in _EVENT_MODELS)],
)
def test_l1_structural_roundtrip(src: str, data_dir: Path, tmp_path: Path) -> None:
    sbml = (data_dir / Path(src).name) if src.startswith("data/") else _event_model(Path(src).name)
    out = tmp_path / (sbml.stem + ".net")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = sbml_to_net(sbml, out, validate="L1", strict=True)

    assert isinstance(report, ConversionReport)
    assert out.is_file()
    assert report.structural is not None
    assert report.structural.passed, report.structural.summary()
    assert report.ok
    # No lossy notes for these curated models (events are dropped, not lossy).
    assert report.lossy == []


@pytest.mark.parametrize(
    "src",
    ["data/BIOMD0000000003.xml", *(f"events/{n}" for n in _EVENT_MODELS)],
)
def test_output_loads_via_from_net(src: str, data_dir: Path, tmp_path: Path) -> None:
    sbml = (data_dir / Path(src).name) if src.startswith("data/") else _event_model(Path(src).name)
    out = tmp_path / (sbml.stem + ".net")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        sbml_to_net(sbml, out, validate=None, strict=True)
        src_model = bngsim.Model.from_sbml(sbml)
        net_model = bngsim.Model.from_net(out)

    assert net_model.n_species == src_model.n_species
    # The writer re-encodes functional laws as per-species signed flux, so the raw
    # reaction count may grow; the source topology is recovered by folding those
    # fragments back (the L1 structural check gates on the folded count).
    from bngsim.convert._validate import _effective_n_reactions

    assert _effective_n_reactions(net_model) == _effective_n_reactions(src_model)
    assert net_model.n_observables == src_model.n_observables


# ─── Numerical equivalence of the converted network (ODE RHS) ──────────────


@pytest.mark.parametrize(
    "src",
    ["data/BIOMD0000000003.xml", *(f"events/{n}" for n in _EVENT_MODELS)],
)
def test_network_rhs_parity(src: str, data_dir: Path, tmp_path: Path) -> None:
    """The converted network must reproduce the source's dy/dt. Compared via the
    RHS (not a trajectory) so it stays valid even when events were dropped —
    events perturb state discretely but never enter the RHS."""
    sbml = (data_dir / Path(src).name) if src.startswith("data/") else _event_model(Path(src).name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        src_model = bngsim.Model.from_sbml(sbml)
        text = write_net(src_model, tmp_path / "m.net", strict=True)
        net_model = bngsim.Model.from_net(tmp_path / "m.net")
    assert text

    y0 = np.asarray(src_model.get_state(), dtype=float)
    perturbed = y0 * 1.37 + 0.05  # exercise nonlinear terms; avoid all-zeros
    for t, y in ((0.0, y0), (1.0, perturbed)):
        a = np.asarray(src_model._core._eval_rhs(t, y), dtype=float)
        b = np.asarray(net_model._core._eval_rhs(t, y), dtype=float)
        scale = max(float(np.abs(a).max()), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-6 * scale, (
            f"{sbml.name} RHS mismatch at t={t}: max|Δ|={np.max(np.abs(a - b)):.3e}"
        )


# ─── Capability boundary ──────────────────────────────────────────────────


def test_events_dropped_with_warning(tmp_path: Path) -> None:
    sbml = _event_model("BIOMD0000000001.xml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = sbml_to_net(sbml, tmp_path / "m.net", validate=None, strict=True)
    assert any(isinstance(w.message, bngsim.ConversionWarning) for w in caught)
    assert any("event" in str(w.message).lower() for w in caught)
    assert any("event" in note.lower() for note in report.dropped)


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 SBML not present")
def test_lossy_model_strict_raises(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        with pytest.raises(bngsim.ConversionError) as exc:
            sbml_to_net(_SMITH, tmp_path / "smith.net", strict=True)
    # Message names the construct and offers the escape hatch.
    msg = str(exc.value)
    assert "strict=False" in msg or "--allow-lossy" in msg
    assert "amount-valued" in msg or "volume" in msg


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 SBML not present")
def test_lossy_model_allow_lossy(tmp_path: Path) -> None:
    out = tmp_path / "smith.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = sbml_to_net(_SMITH, out, validate="L1", strict=False)
    assert out.is_file()
    assert report.lossy, "expected lossy notes for an amount-valued multi-compartment model"
    # Structure is still preserved even though the conversion is numerically lossy.
    assert report.structural is not None and report.structural.passed


# ─── Cross-compartment reactions at ONE shared volume (#192 follow-up) ─────
#
# The capability gate keyed "needs a per-species divisor" off the
# `per_species_volume_scaling` FLAG. That was the same question until #192, which
# routes a cross-compartment reaction whose compartments merely share a load-time
# size to that flag as well — correct for a live model, where one of those sizes
# can be written and pull the volumes apart, but a `.net` freezes its volumes at
# export, so one shared divisor stays shared. Refusing them cost 77 rr_parity
# models their faithful conversion, and the gate's own wording ("with differing
# volumes") already described the narrower thing.
#
# These fixtures are SBML strings on purpose. The two tests that caught the
# regression are gated on the gitignored rr_parity corpus, so they SKIP in any
# worktree and in CI — the whole defect reached main that way.

_XCOMP_SHARED_V = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp">
    <listOfCompartments>
      <compartment id="C1" size="{v}" constant="true"/>
      <compartment id="C2" size="{v}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C2" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="Km" value="20" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="transport" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/>
            <apply><times/><ci>k</ci><ci>A</ci></apply>
            <apply><plus/><ci>Km</ci><ci>A</ci></apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _xcomp_shared(v: float) -> bngsim.Model:
    return bngsim.Model.from_sbml_string(_XCOMP_SHARED_V.format(v=repr(float(v))))


@pytest.mark.parametrize("v", [1.0, 5.0])
def test_one_shared_volume_converts_and_round_trips(v: float, tmp_path: Path) -> None:
    """The #192 shape converts under STRICT and reproduces the source RHS.

    ``v=5.0`` is the load-bearing row: the shared divisor is not 1, so the
    conversion is only faithful because the flux expansion folds ``1/V`` into each
    per-species rate — the same fold ``_vd_<rid>_unified`` used to carry at load.
    At ``v=1.0`` the divide is a no-op, which is why every one of the 77 corpus
    models (all of them ``V=1``) would pass even a writer that dropped it."""
    src = _xcomp_shared(v)
    rxns = src._core.codegen_data()["reactions"]
    assert any(r["per_species_volume_scaling"] for r in rxns), (
        "fixture must exercise the flag; #192 routes this shape to the per-species branch"
    )
    assert not capability_report(src)["lossy"]

    out = tmp_path / "xcomp.net"
    write_net(src, out, strict=True)  # must not raise
    net = bngsim.Model.from_net(out)
    y = np.asarray(src._core.get_state(), dtype=float)
    a = np.asarray(src._core._eval_rhs(0.0, y), dtype=float)
    b = np.asarray(net._core._eval_rhs(0.0, y), dtype=float)
    assert np.allclose(a, b, rtol=1e-12, atol=0.0), f"V={v}: {a} vs {b}"


def test_differing_volumes_are_still_refused(tmp_path: Path) -> None:
    """The case the note was always about stays refused — the relaxation is keyed
    on the divisors actually coinciding, not on waving the flag through."""
    src = bngsim.Model.from_sbml_string(
        _XCOMP_SHARED_V.format(v="1.0").replace('id="C2" size="1.0"', 'id="C2" size="4.0"')
    )
    notes = capability_report(src)["lossy"]
    assert any("per-species volume scaling" in n for n in notes), notes
    with pytest.raises(bngsim.ConversionError, match="per-species volume scaling"):
        write_net(src, tmp_path / "x.net", strict=True)


# The two reaction shapes that decide the gate, as `codegen_data()` reports them.
# Table-driven against the predicate rather than through an SBML fixture: the
# rate-rule shape below is what BIOMD0000000006 / 207 / 241 / 527 / MODEL2002070001
# actually emit, and a fixture that merely *tried* to reproduce it would silently
# skip — which is indistinguishable from coverage.
_RXN_TRANSPORT = {  # a #192 cross-compartment transport reaction
    "type": "functional",
    "function_name": "transport",
    "stat_factor": 1.0,
    "apply_species_factor": False,
    "is_rate_rule_ode": False,
    "per_species_volume_scaling": True,
    "reactants": [0],
    "products": [1],
}
_RXN_RATE_RULE = {  # a rate-rule ODE emission, same flag, unrelated meaning
    "type": "functional",
    "function_name": "_rhs_u",
    "stat_factor": 1.0,
    "apply_species_factor": True,
    "is_rate_rule_ode": True,
    "per_species_volume_scaling": True,
    "reactants": [],
    "products": [1],
}


def _sp(*vols, live=()):
    return [
        {"volume_factor": v, "ode_live_volume_idx0": (3 if i in live else -1)}
        for i, v in enumerate(vols)
    ]


@pytest.mark.parametrize(
    ("rxn", "species", "lossy", "why"),
    [
        (_RXN_TRANSPORT, _sp(1.0, 1.0), False, "one shared unit volume — no divide needed"),
        (_RXN_TRANSPORT, _sp(5.0, 5.0), False, "one shared volume — the expansion folds 1/V"),
        (_RXN_TRANSPORT, _sp(1.0, 4.0), True, "differing volumes — the original case"),
        (_RXN_TRANSPORT, _sp(2.0, 2.0, live=(1,)), True, "live volume — equal now, not later"),
        (_RXN_RATE_RULE, _sp(1.0, 1.0), True, "rate-rule ODE — same flag, other meaning"),
        (_RXN_RATE_RULE, _sp(3.0, 3.0), True, "rate-rule ODE stays refused at any volume"),
    ],
)
def test_which_psvs_reactions_really_need_a_per_species_divide(rxn, species, lossy, why) -> None:
    """The rule, stated where a corpus sweep cannot reach it.

    Relaxing for the rate-rule rows admitted five corpus models, of which
    ``BIOMD0000000006`` reloads to a trajectory **31%** off while
    ``rhs_faithful`` still reports True — the pointwise probe is blind to it, the
    same way GH #18's reactant-guard divergence was. Measured, not assumed."""
    assert _psvs_needs_per_species_divide(rxn, species) is lossy, why


# ─── L2 RHS-identity self-check (GH #223 silent-loss guard) ────────────────


def test_flux_expansion_fixes_reactant_independent_law(tmp_path: Path) -> None:
    """A functional reaction whose propensity is reactant-*independent* (here a
    constant influx onto an initially-empty species) round-trips RHS-faithfully:
    the writer emits every ``asf=False`` functional law as per-species signed flux
    instead of the reactant-division-guarded form that wrongly zeroed it. Before
    this fix BIOMD0000000060 diverged by ~1.0 and was a silent .net loss.

    The re-encoding grows the raw reaction count (each law becomes one zero-reactant
    ``0 -> species`` flux per affected species); the source topology is recovered by
    folding those fragments back, so the L1 structural check still passes."""
    sbml = _corpus_xml("BIOMD0000000060")
    out = tmp_path / "m.net"
    report = sbml_to_net(sbml, out)  # default L2 + strict: must NOT raise now
    assert report.ok and report.rhs_faithful is True
    assert report.max_rhs_delta is not None and report.max_rhs_delta <= 1e-6
    # the broken reaction became a per-species signed flux (zero-reactant synthesis)
    net_model = bngsim.Model.from_net(out)
    assert net_model.n_reactions > report.n_reactions  # at least one reaction split


def test_flux_guard_zero_crossing_reactant_roundtrips_faithfully(tmp_path: Path) -> None:
    """GH #18: a functional reaction whose reactant *crosses zero mid-trajectory*
    (a rate-driven pseudo-species going negative) round-trips faithfully.

    Koo2013 (BIOMD0000000468) has reactant-independent laws (e.g. ``s63 -> s64`` at a
    rate independent of ``s63``) whose reactant starts positive but is driven below
    zero. The old reactant-division guard ``if(r>1e-300, P/r, 0)`` silently zeroed
    the flux once the reactant reached zero — the initial-state probe could not see
    the later crossing, so ``sbml_to_net`` falsely reported ``rhs_faithful=True``
    while the ``.net`` integrated to a ~100%-wrong trajectory. Emitting every
    functional law as a signed flux removes the guard entirely, so the conversion is
    now faithful under *integration*, not just at the two RHS probe points."""
    import numpy as np

    sbml = _corpus_xml("BIOMD0000000468")
    out = tmp_path / "koo.net"
    report = sbml_to_net(sbml, out)  # default L2 + strict: must NOT raise now
    assert report.ok and report.rhs_faithful is True

    # Prove faithfulness under integration (the pointwise probe cannot see the
    # mid-trajectory zero-crossing that used to be lost). The divergence appears
    # within ~0.5% of the model's horizon, so a short window is decisive.
    src = bngsim.Model.from_sbml(sbml)
    net = bngsim.Model.from_net(out)
    ts, npts = (0.0, 1000.0), 201
    a = np.asarray(bngsim.Simulator(src, method="ode").run(t_span=ts, n_points=npts).species)
    b = np.asarray(bngsim.Simulator(net, method="ode").run(t_span=ts, n_points=npts).species)
    scale = max(float(np.abs(a).max()), 1.0)
    assert np.max(np.abs(a[-1] - b[-1])) <= 1e-4 * scale, (
        f"from_net diverged from from_sbml: {np.max(np.abs(a[-1] - b[-1])) / scale:.2e}"
    )


def test_ar_report_frozen_species_refused(tmp_path: Path) -> None:
    """GH #18: a model with a *varying* AssignmentRule-target species is refused.

    vonDassow2000 (BIOMD0000001065) emits 12 AssignmentRule-target ``_T`` totals as
    ``fixed`` species; their live rule value is a Simulator report transform the flat
    ``.net`` cannot carry, so the reloaded network reports them frozen at their
    initial value. The ODE-RHS probe is blind (a fixed species has ``dy/dt = 0`` in
    both models, giving ``max_rhs_delta ≈ 0``); the assignment-rule report probe
    catches the varying rule and refuses under strict."""
    sbml = _corpus_xml("BIOMD0000001065")
    with pytest.raises(bngsim.ConversionError) as exc:
        sbml_to_net(sbml, tmp_path / "vd.net")  # default L2 + strict
    msg = str(exc.value)
    assert "assignment-rule" in msg and "frozen" in msg
    assert "strict=False" in msg or "--allow-lossy" in msg

    # strict=False emits a best-effort network and records the loss, not raises.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = sbml_to_net(sbml, tmp_path / "vd.net", strict=False)
    assert report.rhs_faithful is False and report.ok is False
    # the RHS probe is blind; the AR-report probe is what fires
    assert report.max_rhs_delta is not None and report.max_rhs_delta <= 1e-6
    assert report.max_ar_report_delta is not None and report.max_ar_report_delta > 1e-6
    assert any(issubclass(w.category, bngsim.ConversionWarning) for w in caught)


def test_ar_report_constant_rule_passes(tmp_path: Path) -> None:
    """A *constant* AssignmentRule target must NOT be refused (no false positive).

    BIOMD0000000264 assigns a constant input species by rule; freezing it in the
    ``.net`` reports the same value the source reports, so the conversion is
    faithful and the assignment-rule report probe stays ≈ 0."""
    sbml = _corpus_xml("BIOMD0000000264")
    report = sbml_to_net(sbml, tmp_path / "m.net")  # default L2 + strict: must NOT raise
    assert report.ok and report.rhs_faithful is True
    assert report.max_ar_report_delta is not None and report.max_ar_report_delta <= 1e-6


def test_rateof_refs_detects_undefined_csymbol() -> None:
    """_rateof_refs flags rateOf csymbols a function references but nothing defines
    (a reaction-derivative wired at run time, unrepresentable in the flat channel),
    and excludes a real function that happens to be named ``rate_of_*``."""
    data = {
        "functions": [
            {"name": "_rhs_sum", "expression": "abs(rate_of__x8) + rate_of__x6"},
            {"name": "plain", "expression": "k1 * S0"},
        ]
    }
    assert _rateof_refs(data) == ["rate_of__x6", "rate_of__x8"]
    # A defined function named rate_of_* is not a dangling csymbol reference.
    defined = {
        "functions": [
            {"name": "rate_of_foo", "expression": "k * S"},
            {"name": "g", "expression": "rate_of_foo + 1"},
        ]
    }
    assert _rateof_refs(defined) == []


def test_rateof_csymbol_refused(tmp_path: Path) -> None:
    """A model whose function references a rateOf csymbol cannot survive the flat
    .net (or cBNGL) text channel — the reloaded function fails to compile. Both
    targets must refuse fail-loud under strict, not crash, and best-effort
    (strict=False) must not crash either (rhs_faithful=False)."""
    from bngsim.convert import sbml_to_bngl

    sbml = _corpus_xml("BIOMD0000000696")  # abs(rate_of__x8) in an assignment rule
    for fn, kw in ((sbml_to_net, {}), (sbml_to_bngl, {})):
        with pytest.raises(bngsim.ConversionError, match="rateOf"):
            fn(sbml, tmp_path / "m.out", **kw)  # default strict=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = sbml_to_net(sbml, tmp_path / "m.net", strict=False)
    assert any("rateOf" in n for n in rep.lossy)
    assert rep.rhs_faithful is False  # reload failed → recorded, not raised


def test_l2_gate_raises_on_rhs_divergence(monkeypatch, data_dir: Path, tmp_path: Path) -> None:
    """The L2 gate is the authoritative silent-loss guard: when the round-tripped
    .net does not reproduce the source ODE RHS, ``strict=True`` raises and
    ``strict=False`` records the loss (``rhs_faithful=False``, ``ok=False``,
    a warning). Driven deterministically by forcing a nonzero RHS delta, since the
    writer is now faithful on every cap-clean corpus model."""
    import bngsim.convert as _convert

    monkeypatch.setattr(_convert, "_max_rhs_delta", lambda a, b: 0.5)
    sbml = data_dir / "BIOMD0000000003.xml"

    with pytest.raises(bngsim.ConversionError) as exc:
        sbml_to_net(sbml, tmp_path / "m.net")  # default L2 + strict
    assert "right-hand side" in str(exc.value)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = sbml_to_net(sbml, tmp_path / "m.net", strict=False)
    assert report.rhs_faithful is False
    assert report.max_rhs_delta == 0.5
    assert report.ok is False
    assert any(issubclass(w.category, bngsim.ConversionWarning) for w in caught)
    # The counts-only escape hatch (validate="L1") does not run the RHS check.
    rep_l1 = sbml_to_net(sbml, tmp_path / "m.net", validate="L1")
    assert rep_l1.rhs_faithful is None


def test_l2_default_passes_faithful_model(tmp_path: Path) -> None:
    """A model whose assignment rules fold into live .net functions round-trips
    RHS-faithfully and passes the default L2 gate (no false positive)."""
    sbml = _corpus_xml("BIOMD0000000183")  # 46 assignment rules → live functions
    report = sbml_to_net(sbml, tmp_path / "m.net")
    assert report.ok and report.rhs_faithful is True
    assert report.max_rhs_delta is not None and report.max_rhs_delta <= 1e-6


# ─── API surface ──────────────────────────────────────────────────────────


def test_write_net_in_memory_no_file(data_dir: Path) -> None:
    model = bngsim.Model.from_sbml(data_dir / "BIOMD0000000003.xml")
    text = write_net(model, strict=True)
    assert "begin reactions" in text and "begin parameters" in text
    assert "begin species" in text


def test_validate_structural_l1_direct(data_dir: Path, tmp_path: Path) -> None:
    sbml = data_dir / "BIOMD0000000003.xml"
    src_model = bngsim.Model.from_sbml(sbml)
    write_net(src_model, tmp_path / "m.net", strict=True)
    net_model = bngsim.Model.from_net(tmp_path / "m.net")
    rep = validate_structural_l1(src_model, net_model)
    assert rep.passed, rep.summary()


def test_capability_report_clean_model(data_dir: Path) -> None:
    model = bngsim.Model.from_sbml(data_dir / "BIOMD0000000003.xml")
    rep = capability_report(model)
    assert rep["dropped"] == [] and rep["lossy"] == []


def test_full_gate_uses_sedml_horizon(data_dir: Path, tmp_path: Path) -> None:
    """sedml= drives the full gate's L3 over the model's own horizon — the
    symmetric mirror of net2sbml's bngl=. The parsed protocol is attached."""
    sbml = data_dir / "BIOMD0000000003.xml"
    sedml = tmp_path / "m.sedml"
    sedml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sedML xmlns="http://sed-ml.org/sed-ml/level1/version4" level="1" version="4">\n'
        "  <listOfSimulations>\n"
        '    <uniformTimeCourse id="s1" initialTime="0" outputStartTime="0" '
        'outputEndTime="42" numberOfSteps="10">\n'
        '      <algorithm kisaoID="KISAO:0000019"/>\n'
        "    </uniformTimeCourse>\n"
        "  </listOfSimulations>\n"
        "</sedML>\n"
    )
    out = tmp_path / "m.net"
    report = sbml_to_net(sbml, out, validate="full", strict=True, sedml=sedml)
    assert report.ok, report.summary()
    assert report.protocol is not None
    l3 = report.validation.level("L3")
    assert l3.metrics["t_span"] == [0.0, 42.0]
    assert l3.metrics["horizon_source"] == "sedml protocol"


def test_sedml_sibling_autodetect_cli(data_dir: Path, tmp_path: Path) -> None:
    """The CLI auto-detects a sibling <stem>.sedml for the L3 horizon (mirror of
    net2sbml's sibling .bngl detection)."""
    from bngsim.convert._cli import main

    sbml = tmp_path / "model.xml"
    sbml.write_text((data_dir / "BIOMD0000000003.xml").read_text())
    (tmp_path / "model.sedml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sedML xmlns="http://sed-ml.org/sed-ml/level1/version4" level="1" version="4">\n'
        "  <listOfSimulations>\n"
        '    <uniformTimeCourse id="s1" initialTime="0" outputStartTime="0" '
        'outputEndTime="7" numberOfSteps="5"><algorithm kisaoID="KISAO:0000019"/>'
        "</uniformTimeCourse>\n"
        "  </listOfSimulations>\n"
        "</sedML>\n"
    )
    rc = main([str(sbml), "-o", str(tmp_path / "out.net"), "--gate", "full", "--quiet"])
    assert rc == 0


# ─── CLI ──────────────────────────────────────────────────────────────────


def test_cli_smoke(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import main

    out = tmp_path / "out.net"
    rc = main([str(data_dir / "BIOMD0000000003.xml"), "-o", str(out), "--quiet"])
    assert rc == 0
    assert out.is_file()
    model = bngsim.Model.from_net(out)
    assert model.n_reactions == 7


def test_cli_lossy_strict_exits_nonzero(tmp_path: Path) -> None:
    if not _SMITH.is_file():
        pytest.skip("Smith2013 SBML not present")
    from bngsim.convert._cli import main

    rc = main([str(_SMITH), "-o", str(tmp_path / "smith.net")])
    assert rc == 1  # strict refusal → non-zero exit


_DECAY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="decay">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="v1" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S</ci>{extra}</apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _rate_law_params(model: bngsim.Model) -> list[tuple[str, bool, str]]:
    return [
        (p["name"], p["is_const"], p["expression"])
        for p in model._core.codegen_data()["parameters"]
        if p["name"].startswith("_rateLaw")
    ]


def _max_sensitivity(model: bngsim.Model) -> float:
    sim = bngsim.Simulator(
        model, method="ode", sensitivity_params=["k"], sensitivity_method="staggered"
    )
    result = sim.run(t_span=(0.0, 5.0), n_points=20, rtol=1e-10, atol=1e-12)
    return float(np.max(np.abs(np.asarray(result.sensitivities))))


def test_synthesized_rate_law_keeps_its_parameter_reference(tmp_path: Path) -> None:
    """A synthesized _rateLaw_* must not be folded to a literal in the .net.

    Folding drops its reference to the model parameter it was built from. The
    parameter stays declared, so nothing objects, but the right-hand side no
    longer mentions it and a forward sensitivity comes back identically zero,
    finite and unflagged, disagreeing with the same model via from_sbml.
    """
    src = tmp_path / "decay.xml"
    src.write_text(_DECAY_SBML.format(extra=""), encoding="utf-8")
    out = tmp_path / "decay.net"

    sbml_to_net(src, out, validate=None)

    from_sbml = bngsim.Model.from_sbml(src)
    from_net = bngsim.Model.from_net(out)

    # The reference survives the round trip rather than becoming a constant.
    (_, net_is_const, net_expr) = _rate_law_params(from_net)[0]
    assert not net_is_const
    assert "k" in net_expr

    # A model may only be sensitivity-solved once without a reset, so take
    # each measurement a single time.
    net_max = _max_sensitivity(from_net)
    sbml_max = _max_sensitivity(from_sbml)

    # dS/dk for S(t) = S0 exp(-kt) peaks at t = 1/k, giving 10 * (1/0.3) / e.
    expected = 10.0 * (1.0 / 0.3) * np.exp(-1.0)
    assert net_max == pytest.approx(expected, rel=1e-3)
    assert net_max == pytest.approx(sbml_max)


def test_rate_law_with_explicit_compartment_factor_is_unchanged(tmp_path: Path) -> None:
    """A law carrying its own compartment factor synthesizes no _rateLaw_*.

    That path already round-tripped correctly and must stay on literals.
    """
    src = tmp_path / "decay_cf.xml"
    src.write_text(_DECAY_SBML.format(extra="<ci>c</ci>"), encoding="utf-8")
    out = tmp_path / "decay_cf.net"

    sbml_to_net(src, out, validate=None)

    from_net = bngsim.Model.from_net(out)
    assert _rate_law_params(from_net) == []
    assert _max_sensitivity(from_net) == pytest.approx(
        _max_sensitivity(bngsim.Model.from_sbml(src))
    )
