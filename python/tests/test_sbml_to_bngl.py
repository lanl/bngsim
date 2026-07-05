"""Tests for bngsim.convert.sbml_to_bngl — SBML→cBNGL writer (GH #224, deliverable 1).

Deliverable 1 recovers **static compartment volumes**: a non-unit-volume model
that the flat ``.net`` channel refuses is emitted as compartmental BNGL and
round-trips faithfully through ``BNG2.pl generate_network`` →
``Model.from_net``. Covers the ``begin compartments`` reconstruction, the
capability boundary (cross-compartment/transport, events, live volumes, and
assignment-rule species refused fail-loud — those are deliverable 1b / #205/#223),
the ExprTk→BNGL expression normalization, and the BNG2.pl round-trip RHS
equivalence over a handful of clean within-compartment models.

The round-trip suite shells out to ``BNG2.pl`` (``$BNGPATH/BNG2.pl``) and skips
cleanly when it — or the rr_parity corpus — is absent, like the other
oracle-dependent tests.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim._exceptions import ConversionError, ConversionWarning
from bngsim.convert import (
    Experiment,
    ProtocolSpec,
    StateChange,
    parse_bngl_protocol,
    sbml_to_bngl,
    write_bngl,
    write_bngl_protocol,
)
from bngsim.convert._bng2 import find_bng2 as _bng2_find
from bngsim.convert._bng2 import roundtrip_rhs_delta
from bngsim.convert._bngl_writer import (
    _bng_stat_factor,
    _normalize_bngl_expr,
    bngl_capability_report,
)
from bngsim.convert._events import sbml_events_to_protocol

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

_BNGSIM = Path(__file__).resolve().parents[2]
_CORPUS = _BNGSIM / "parity_checks" / "rr_parity" / "models"


def _find_bng2() -> Path | None:
    """Locate ``BNG2.pl`` via the shipped :func:`bngsim.convert._bng2.find_bng2`
    ($BNGPATH / PATH) or an explicit ``$BNG2_PL`` override (test-only convenience
    fallback); ``None`` when unavailable (→ the round-trip suite skips)."""
    found = _bng2_find()
    if found is not None:
        return found
    env = os.environ.get("BNG2_PL")
    return Path(env) if env and Path(env).is_file() else None


_BNG2 = _find_bng2()
_HAS_PERL = shutil.which("perl") is not None


def _corpus_xml(model_id: str) -> Path:
    """Path to a corpus model's SBML, skipping the test when the corpus is absent."""
    hits = sorted(_CORPUS.glob(f"{model_id}/*.xml"))
    if not hits:
        pytest.skip(f"rr_parity corpus model {model_id} not present")
    return hits[0]


def _load(model_id: str) -> bngsim.Model:
    return bngsim.Model.from_sbml(_corpus_xml(model_id))


# ── clean within-compartment models that round-trip faithfully ─────────────
# Static non-unit compartment volumes, every reaction single-compartment, no
# events / live volumes / assignment-rule species. (The named #224 models —
# BIOMD600/705/737 — are transport-heavy and belong to deliverable 1b.)
_CLEAN = [
    "BIOMD0000000150",  # 4 sp, single compartment V=1e-12
    "BIOMD0000000329",  # 3 sp, single compartment
    "BIOMD0000000002",  # 13 sp, single compartment V=1e-16
    "MODEL1511170000",  # 5 sp, single compartment V=5.5e-17
    "BIOMD0000000162",  # multi-compartment (4 distinct volumes), no transport
    "BIOMD0000000233",  # catalytic identical-reactant rxn X+X->X+Y (symmetry-factor fix)
]

# Cross-compartment transport models (deliverable 1b): dynamic species span >1
# volume — the per-species 1/V asymmetry recovered by the signed-flux split. Covers
# both product-cross-compartment (reactant in one volume, products elsewhere) and
# reactant-cross-compartment (the reactants themselves in different volumes — the
# split divides each species' flux by its *own* volume, so it needs no single
# reactant compartment). These are the named #224 payoff class (BIOMD600 etc.).
_TRANSPORT = [
    "BIOMD0000000321",  # 3 sp, 3 volumes, 2 transports (one-way flux)
    "BIOMD0000000041",  # 10 sp, 2 volumes, 5 reversible diffusions (Vin·k·Ai − Vout·k·A)
    "BIOMD0000000600",  # the named #224 transport-heavy model
    "BIOMD0000000075",  # reactant-cross-compartment (reactants span 2 volumes)
    "BIOMD0000000161",  # 5 reactant-cross-compartment reactions
]


# ─── Unit tests (no BNG2.pl) ───────────────────────────────────────────────


def test_in_public_surface():
    assert "sbml_to_bngl" in bngsim.convert.__all__
    assert "write_bngl" in bngsim.convert.__all__
    assert callable(bngsim.convert.sbml_to_bngl)
    assert callable(bngsim.convert.write_bngl)


def test_compartments_block_reconstructed():
    """A non-unit single-volume model emits one ``begin compartments`` entry with
    the right static volume; every species is compartment-qualified."""
    model = _load("BIOMD0000000150")  # all species in one non-unit volume
    vol = float(model._core.codegen_data()["species"][0]["volume_factor"])
    assert vol != 1.0
    text = write_bngl(model)

    comp = _block(text, "compartments")
    # one compartment, dimension 3, the model's static volume
    assert len(comp) == 1, comp
    name, dim, size = comp[0].split()[:3]
    assert dim == "3"
    assert float(size) == pytest.approx(vol)

    # every species line is @<comp>:<Mol>() — qualified by the reconstructed comp
    species = _block(text, "species")
    assert species
    for line in species:
        assert line.startswith(f"@{name}:"), line

    # one molecule type per species
    mols = _block(text, "molecule types")
    assert len(mols) == model.n_species


def test_multi_volume_distinct_compartments():
    """Distinct static volumes become distinct compartments (one per volume)."""
    model = _load("BIOMD0000000162")  # 4 distinct static volumes
    data = model._core.codegen_data()
    n_vol = len({float(s.get("volume_factor", 1.0)) for s in data["species"]})
    text = write_bngl(model)
    comp = _block(text, "compartments")
    assert len(comp) == n_vol
    sizes = sorted(float(c.split()[2]) for c in comp)
    assert sizes == sorted({float(s.get("volume_factor", 1.0)) for s in data["species"]})


def test_report_fields_populated():
    rep = sbml_to_bngl(_corpus_xml("BIOMD0000000150"))
    assert rep.output_text.lstrip().startswith("# Generated by bngsim.convert.sbml_to_bngl")
    assert "begin model" in rep.output_text and "end model" in rep.output_text
    assert rep.n_species == _load("BIOMD0000000150").n_species
    assert rep.ok  # no lossy notes on a clean model


@pytest.mark.parametrize(
    ("model_id", "keyword"),
    [
        ("BIOMD0000000101", "event"),  # events refused on the generic path
        ("BIOMD0000000009", "assignment-rule"),  # _ar_report_map (#205/#223)
        ("BIOMD0000000429", "time-varying volume"),  # live/varvol volume
        ("MODEL1006230049", "not-equal operator"),  # `!=` BNG2.pl can't parse
        ("MODEL1703150000", "non-finite"),  # ±inf FBA flux-bound parameters
    ],
)
def test_refuses_fail_loud(model_id, keyword):
    """Out-of-scope constructs are refused fail-loud under strict, with a readable
    plain-English reason; strict=False downgrades to a warning instead.

    This exercises the *generic* ``write_bngl`` path (no SBML-event
    classification), so any event model refuses here; the SBML-aware
    :func:`sbml_to_bngl` reclassifies fixed-time events into an actions block
    (see the event tests below)."""
    model = _load(model_id)
    caps = bngl_capability_report(model)
    assert any(keyword in note for note in caps["lossy"]), caps["lossy"]

    with pytest.raises(ConversionError, match="cannot convert faithfully to cBNGL"):
        write_bngl(model, strict=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConversionWarning):
            write_bngl(model, strict=False)


# ─── events → BNGL actions (GH #224 phase 2) ───────────────────────────────


def test_write_bngl_protocol_roundtrip():
    """``parse_bngl_protocol(write_bngl_protocol(p))`` recovers an equal spec for a
    multi-phase simulate sequence with interleaved state changes."""
    proto = ProtocolSpec(
        steps=(
            Experiment(kind="simulate", method="ode", t_span=(0.0, 10.0), n_points=101),
            StateChange(kind="set_concentration", target="@comp1:X()", value=5.0),
            Experiment(
                kind="simulate",
                method="ode",
                t_span=(10.0, 100.0),
                n_points=91,
                extra={"continue": 1},
            ),
            StateChange(kind="set_parameter", target="k1", value=2.5),
            StateChange(kind="reset_concentrations"),
        )
    )
    text = write_bngl_protocol(proto)
    assert text.startswith("begin actions")
    back = parse_bngl_protocol(text)
    assert back.steps == proto.steps


def test_write_bngl_protocol_scan_roundtrip():
    """A ``parameter_scan`` experiment round-trips through the writer."""
    proto = ProtocolSpec(
        steps=(
            Experiment(
                kind="scan",
                method="ode",
                t_span=(0.0, 50.0),
                n_points=51,
                scan_parameter="k",
                scan_min=1.0,
                scan_max=10.0,
                scan_points=5,
                scan_log=True,
                reset_between=False,
                label="sweep",
            ),
        )
    )
    back = parse_bngl_protocol(write_bngl_protocol(proto))
    assert back.steps == proto.steps


def test_fixed_time_event_converts_to_actions():
    """A fixed-time event model emits a ``begin actions`` block: ``simulate``
    phases with ``setConcentration`` at each fire time, targets translated to the
    compartment-qualified pattern."""
    rep = sbml_to_bngl(_corpus_xml("BIOMD0000000316"))
    assert rep.ok, rep.lossy
    text = rep.output_text
    assert "begin actions" in text and "end actions" in text
    assert "generate_network" in text
    acts = _block(text, "actions")
    assert any(a.startswith("simulate(") for a in acts)
    setc = [a for a in acts if a.startswith("setConcentration(")]
    assert setc
    # target is the compartment-qualified species pattern, not the raw name
    assert all('"@' in a and "()" in a for a in setc), setc
    # the protocol is carried on the report
    assert rep.protocol is not None and not rep.protocol.is_empty


@pytest.mark.parametrize(
    ("model_id", "fragment"),
    [
        ("BIOMD0000000144", "state-triggered"),  # trigger depends on a species
        ("BIOMD0000000007", "state-triggered"),  # trigger depends on a species
        ("BIOMD0000000121", "rising time threshold"),  # pulse window, unschedulable
    ],
)
def test_state_or_complex_event_refused(model_id, fragment):
    """State-triggered and non-schedulable (pulse/expression) events are refused
    fail-loud by sbml_to_bngl, naming the specific event and reason."""
    with pytest.raises(ConversionError, match=fragment):
        sbml_to_bngl(_corpus_xml(model_id))


def test_events_to_protocol_classification():
    """The classifier returns a protocol for a fixed-time model and None + a
    specific note for a state-triggered one."""
    m_ok = _load("BIOMD0000000316")
    proto, lossy = sbml_events_to_protocol(_corpus_xml("BIOMD0000000316"), m_ok)
    assert proto is not None and not lossy

    m_state = _load("BIOMD0000000144")
    proto2, lossy2 = sbml_events_to_protocol(_corpus_xml("BIOMD0000000144"), m_state)
    assert proto2 is None
    assert any("state-triggered" in n for n in lossy2)


def test_strict_false_emits_text():
    """A best-effort (strict=False) conversion still produces a model block."""
    model = _load("BIOMD0000000101")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = write_bngl(model, strict=False)
    assert "begin model" in text and "begin reaction rules" in text


def test_normalize_bngl_expr():
    """ExprTk dialect → BNGL: and/or → &&/||, natural log → ln, log10 kept,
    log2/log1p → ln identities; if()/time()/_pi/^ pass through."""
    assert _normalize_bngl_expr("a and b") == "a && b"
    assert _normalize_bngl_expr("a or b") == "a || b"
    assert _normalize_bngl_expr("log(x)") == "ln(x)"
    assert _normalize_bngl_expr("log10(x)") == "log10(x)"
    assert "ln(2)" in _normalize_bngl_expr("log2(y)")
    assert _normalize_bngl_expr("ln(1 + (z))") == _normalize_bngl_expr("log1p(z)")
    # a bareword merely *containing* and/or/log is untouched
    assert _normalize_bngl_expr("ligand*factor") == "ligand*factor"
    # if/time/power pass through; _pi folds to its numeric literal (BNG2.pl rejects
    # the symbol both bare and as a parameter)
    out = _normalize_bngl_expr("if(time()>0, _pi*x^2, 0)")
    assert "_pi" not in out and "3.14159" in out
    assert out.startswith("if(time()>0, ") and "*x^2, 0)" in out


def test_bng_stat_factor_accounts_for_preserved_reactants():
    """BNG's symmetry divisor is ∏ⱼ pⱼ!·(mⱼ−pⱼ)!, not ∏mⱼ!: identical reactants
    the reaction preserves vs consumes are distinguishable. Cross-checked against
    BNG2.pl 2.9.3 (see _bng_stat_factor)."""
    # distinct reactants → 1 (covers the common path, unchanged by the fix)
    assert _bng_stat_factor([0, 1], [2]) == 1
    # X+X -> Y : both consumed → 2! = 2
    assert _bng_stat_factor([0, 0], [1]) == 2
    # X+X -> X+Y : one preserved, one consumed → 1!·1! = 1 (the BIOMD233 bug)
    assert _bng_stat_factor([0, 0], [0, 1]) == 1
    # X+X -> X+X : both preserved → 2!·0! = 2
    assert _bng_stat_factor([0, 0], [0, 0]) == 2
    # 3-body: X+X+X -> X+Y : 1 preserved, 2 consumed → 1!·2! = 2
    assert _bng_stat_factor([0, 0, 0], [0, 1]) == 2
    # 3-body: X+X+X -> Y : all consumed → 3! = 6
    assert _bng_stat_factor([0, 0, 0], [1]) == 6
    # mixed: X+X+Y -> Z : 2!·1! over X, 1 over Y → 2
    assert _bng_stat_factor([0, 0, 1], [2]) == 2


# ─── CLI (bngsim-sbml2bngl) ────────────────────────────────────────────────


def test_cli_writes_bngl(tmp_path):
    """``bngsim-sbml2bngl`` writes a model block and exits 0 on a clean model."""
    from bngsim.convert._cli import sbml2bngl_main

    src = _corpus_xml("BIOMD0000000150")
    out = tmp_path / "out.bngl"
    rc = sbml2bngl_main([str(src), "-o", str(out), "-q"])
    assert rc == 0
    text = out.read_text()
    assert "begin model" in text and "end model" in text
    assert "begin compartments" in text


@pytest.mark.skipif(_BNG2 is None or not _HAS_PERL, reason="BNG2.pl / perl not available")
def test_cli_gate_roundtrip(tmp_path, monkeypatch):
    """``bngsim-sbml2bngl --gate`` runs the BNG2.pl round-trip and exits 0 when the
    cBNGL is RHS-faithful."""
    from bngsim.convert._cli import sbml2bngl_main

    monkeypatch.setenv("BNGPATH", str(_BNG2.parent))
    src = _corpus_xml("BIOMD0000000150")
    rc = sbml2bngl_main([str(src), "-o", str(tmp_path / "out.bngl"), "--gate", "-q"])
    assert rc == 0


def test_cli_default_out_path(tmp_path):
    """No ``-o`` → the .bngl lands beside the input with a .bngl suffix."""
    from bngsim.convert._cli import sbml2bngl_main

    src = shutil.copy(_corpus_xml("BIOMD0000000150"), tmp_path / "m.xml")
    rc = sbml2bngl_main([str(src), "-q"])
    assert rc == 0
    assert (tmp_path / "m.bngl").is_file()


def test_cli_missing_file_exits_2(tmp_path):
    from bngsim.convert._cli import sbml2bngl_main

    assert sbml2bngl_main([str(tmp_path / "nope.xml")]) == 2


def test_cli_refuses_lossy_then_allows(tmp_path, capsys):
    """A still-unsupported model exits 1 (refused); ``--allow-lossy`` emits anyway."""
    from bngsim.convert._cli import sbml2bngl_main

    src = _corpus_xml("BIOMD0000000429")  # live (time-varying) compartment volume
    out = tmp_path / "out.bngl"
    assert sbml2bngl_main([str(src), "-o", str(out), "-q"]) == 1
    assert not out.exists()

    assert sbml2bngl_main([str(src), "-o", str(out), "-q", "--allow-lossy"]) == 0
    assert "begin model" in out.read_text()


# ─── BNG2.pl round-trip (oracle) ───────────────────────────────────────────


@pytest.mark.skipif(_BNG2 is None or not _HAS_PERL, reason="BNG2.pl / perl not available")
def test_bng2_executes_event_actions():
    """A fixed-time event model's cBNGL + actions block is valid, executable BNGL:
    BNG2.pl builds the network, runs the ``simulate`` phases applying each
    ``setConcentration``, and finishes — producing a .gdat. (This validates the
    actions *syntax/semantics*; the model-block RHS faithfulness is covered above.)"""
    model_id = "BIOMD0000000316"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = sbml_to_bngl(_corpus_xml(model_id))
    d = Path(tempfile.mkdtemp())
    bp = d / f"{model_id}.bngl"
    bp.write_text(rep.output_text)
    proc = subprocess.run(
        [str(_BNG2), "--outdir", str(d), str(bp)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    gdat = d / f"{model_id}.gdat"
    assert gdat.is_file(), (
        f"BNG2.pl did not run the actions for {model_id}:\n"
        f"{proc.stdout[-1500:]}\n{proc.stderr[-500:]}"
    )
    # The square-wave event fires repeatedly; the output must span the full
    # horizon (the continue-phases ran past every setConcentration).
    last_line = [ln for ln in gdat.read_text().splitlines() if ln.strip()][-1]
    t_final = float(last_line.split()[0])
    assert t_final >= 99.0, f"{model_id}: actions stopped early at t={t_final}"


@pytest.mark.skipif(_BNG2 is None or not _HAS_PERL, reason="BNG2.pl / perl not available")
@pytest.mark.parametrize("model_id", _CLEAN + _TRANSPORT)
def test_bng2_roundtrip_rhs_faithful(model_id):
    """cBNGL → BNG2.pl generate_network → .net → from_net reproduces the source
    ODE right-hand side (the deterministic faithfulness measure, probed at t>0).
    Covers the within-compartment baseline (_CLEAN), product-cross-compartment and
    reactant-cross-compartment transport (_TRANSPORT, the per-species flux split).
    Exercises the shared :func:`roundtrip_rhs_delta` the production
    ``sbml_to_bngl(validate="bng2")`` gate uses."""
    src = _load(model_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bngl_text = write_bngl(src, model_name=model_id)
    delta, n_rl = roundtrip_rhs_delta(src, bngl_text, stem=model_id, bng2=_BNG2)
    assert n_rl == src.n_species, f"{model_id}: species count {src.n_species}→{n_rl}"
    assert delta <= 1e-6, f"{model_id}: max scale-relative |Δ rhs| = {delta:.2e}"


@pytest.mark.skipif(_BNG2 is None or not _HAS_PERL, reason="BNG2.pl / perl not available")
def test_validate_bng2_gate(monkeypatch):
    """sbml_to_bngl(validate="bng2") runs the round-trip oracle and records the
    verdict on the report; a forced RHS divergence raises under strict and warns
    under strict=False. The boundary-artifact pulse model (MODEL1112110002) passes
    because the gate probes t>0, not the measure-zero t=0 BNG >=→> edge."""
    monkeypatch.setenv("BNGPATH", str(_BNG2.parent))
    rep = sbml_to_bngl(_corpus_xml("BIOMD0000000150"), validate="bng2")
    assert rep.rhs_faithful is True and rep.max_rhs_delta is not None
    # t>0 probing: the pulse-train model is faithful at every t>0 (delta 0).
    rep_pulse = sbml_to_bngl(_corpus_xml("MODEL1112110002"), validate="bng2", strict=False)
    assert rep_pulse.rhs_faithful is True

    import bngsim.convert._bng2 as _b

    monkeypatch.setattr(_b, "roundtrip_rhs_delta", lambda *a, **k: (0.5, 99))
    with pytest.raises(ConversionError, match="right-hand side"):
        sbml_to_bngl(_corpus_xml("BIOMD0000000150"), validate="bng2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep_bad = sbml_to_bngl(_corpus_xml("BIOMD0000000150"), validate="bng2", strict=False)
    assert rep_bad.rhs_faithful is False and rep_bad.ok is False


def test_validate_bng2_requires_bng2(monkeypatch):
    """validate="bng2" raises a clear error when BNG2.pl is unavailable, rather than
    silently skipping validation."""
    monkeypatch.delenv("BNGPATH", raising=False)
    import bngsim.convert._bng2 as _b

    monkeypatch.setattr(_b, "find_bng2", lambda: None)
    with pytest.raises(ConversionError, match="BNG2.pl"):
        sbml_to_bngl(_corpus_xml("BIOMD0000000150"), validate="bng2")


def test_transport_split_emitted():
    """A transport model emits per-species flux reactions (0 -> Mol) with the 1/V
    helper, not a single symmetric cross-compartment reaction (no unit-volume
    .net can carry that asymmetry)."""
    model = _load("BIOMD0000000321")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = write_bngl(model)
    rules = _block(text, "reaction rules")
    # at least one pure-flux synthesis (0 -> @comp:Mol()) with a __flux helper
    flux = [r for r in rules if r.startswith("0 ->") and "__" in r]
    assert flux, rules
    funcs = _block(text, "functions")
    assert any(ln.startswith("__flux") for ln in funcs), funcs


def test_transport_converts_not_refused():
    """The named #224 transport model converts (was refused pre-1b)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = sbml_to_bngl(_corpus_xml("BIOMD0000000600"))
    assert rep.ok, rep.lossy


def test_reactant_cross_compartment_converts_not_refused():
    """A reactant-cross-compartment model (reactants in different volumes) now
    converts: the per-species signed-flux split divides each species' flux by its
    *own* volume, so it carries an arbitrary spread of reactant compartments — no
    single-reactant-compartment assumption. (Was refused 'reactants in more than
    one compartment' before this fix; BNG2.pl round-trip RHS faithfulness is
    asserted in test_bng2_roundtrip_rhs_faithful via _TRANSPORT.)"""
    model = _load("BIOMD0000000075")
    caps = bngl_capability_report(model)
    assert not any("compartment" in n for n in caps["lossy"]), caps["lossy"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = sbml_to_bngl(_corpus_xml("BIOMD0000000075"))
    assert rep.ok, rep.lossy


# ─── helpers ───────────────────────────────────────────────────────────────


def _block(text: str, name: str) -> list[str]:
    """Return the stripped content lines of a ``begin <name>`` … ``end <name>`` block."""
    m = re.search(rf"begin {re.escape(name)}\b(.*?)end {re.escape(name)}\b", text, re.DOTALL)
    assert m, f"block {name!r} not found"
    return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
