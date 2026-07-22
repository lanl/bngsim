"""GH #179 regression: full multi-phase protocol replay for dirty_carryover models.

GH #177's option-1 (state_setup_prefix) bakes a model's pre-simulate
``setParameter`` / ``setConcentration`` state into the ``.net``, but it CANNOT
reproduce a representative simulate whose initial state is a PRIOR simulate's end
state — BNG2.pl carries concentrations across simulates by default (``continue=>1``
governs only the time/output axis, not state). Those models carry
``dirty_carryover=True``.

Option-2 (:func:`_bng_common.multi_segment_replay`) drives bngsim IN-PROCESS through
the ordered action protocol — simulate -> setParameter -> simulate -> … — carrying
each segment's end state forward (``Model.set_state``) and applying the intervening
live-model mutations, then captures the representative segment. This is the genuine
(provably-bngsim, GH #175) path the golden generator uses for a multi-phase network
model.

These tests pin:

  * Unit (cheap, always run): the protocol parser (:func:`parse_protocol`) —
    typed/ordered step extraction and the CONTINUATION-AWARE representative selection
    + ``t_start`` inference (the GH #179 secondary bug: a ``continue=>1`` segment
    written ``t_end=>60`` continuing from t=40 is a [40,60] run, NOT a misread [0,60]).
  * Engine (needs BNG2.pl + perl + a built bngsim): the DECISIVE proof — the replayed
    representative segment reproduces native BNG2.pl's multi-phase trajectory to
    integration tolerance, for both ``setConcentration``- and ``setParameter``-perturbed
    protocols; and the surgical gate keeps single-phase models on the direct path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import _core
import numpy as np
import pytest

_BNG_PARITY = Path(__file__).resolve().parent.parent / "bng_parity"
if str(_BNG_PARITY) not in sys.path:
    sys.path.insert(0, str(_BNG_PARITY))


def _load_bng_common():
    spec = importlib.util.spec_from_file_location("_bng_common", _BNG_PARITY / "_bng_common.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_bng_common()
MODELS = _BNG_PARITY / "models"
_ODE = MODELS / "original/bngl_models/my_models/ode"

_MODEL_HDR = """begin model
begin parameters
 k 1
end parameters
begin species
 A() 100
end species
begin reaction rules
 A() -> 0  k
end reaction rules
end model
"""


def _acts(action_lines: str) -> str:
    return (
        _MODEL_HDR
        + "begin actions\ngenerate_network({overwrite=>1})\n"
        + action_lines
        + "end actions\n"
    )


# --------------------------------------------------------------------------- #
# Unit: protocol parsing + continuation-aware representative selection.
# --------------------------------------------------------------------------- #
def test_parse_protocol_extracts_typed_ordered_steps():
    """setParameter/setConcentration/save/reset between simulates become typed steps."""
    text = _acts(
        'setConcentration("A()",50)\n'
        'simulate({method=>"ode",t_end=>10,n_steps=>10})\n'
        'setParameter("k",2)\n'
        'simulate({method=>"ode",t_end=>40,n_steps=>30,continue=>1})\n'
    )
    steps, rep = C.parse_protocol(
        text, methods=C._ODE_METHODS, default_method="ode", net_params={}
    )
    kinds = [s["kind"] for s in steps]
    assert kinds == ["setconc", "sim", "setparam", "sim"]
    assert steps[0]["name"] == "A()" and steps[0]["value"] == 50.0
    assert steps[2]["name"] == "k" and steps[2]["value"] == 2.0
    # Representative is the larger continuation-aware span: [10,40] span 30 > [0,10].
    assert rep == 3 and steps[rep]["t_start"] == 10.0 and steps[rep]["t_end"] == 40.0


def test_continuation_aware_selection_not_misread_as_zero_start():
    """A continue=>1 seg t_end=>60 from t=40 is span 20, so the [0,40] seg stays the rep.

    The GH #179 secondary bug: with t_start defaulting to 0 the continue segment is
    misread as a [0,60] span (60) and wrongly picked as representative.
    """
    text = _acts(
        'simulate({method=>"ode",t_end=>40,n_steps=>40})\n'
        'setParameter("k",2)\n'
        'simulate({method=>"ode",t_end=>60,n_steps=>20,continue=>1})\n'
    )
    steps, rep = C.parse_protocol(
        text, methods=C._ODE_METHODS, default_method="ode", net_params={}
    )
    assert steps[rep]["t_start"] == 0.0 and steps[rep]["t_end"] == 40.0


def test_continuation_aware_tstart_inferred_when_continue_is_representative():
    """When the continue seg IS the rep (real span 50 > 10), its t_start is inferred = 10."""
    text = _acts(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})\n'
        'setParameter("k",2)\n'
        'simulate({method=>"ode",t_end=>60,n_steps=>50,continue=>1})\n'
    )
    steps, rep = C.parse_protocol(
        text, methods=C._ODE_METHODS, default_method="ode", net_params={}
    )
    assert steps[rep]["t_start"] == 10.0 and steps[rep]["t_end"] == 60.0
    # The prior phase-1 simulate is retained so its end state carries into the rep.
    assert any(s["kind"] == "sim" and s["t_end"] == 10.0 for s in steps[:rep])


def test_explicit_tstart_wins_over_continuation_inference():
    """An explicit t_start=> always wins (matches continue.bngl / ExampleModel3)."""
    text = _acts(
        'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10})\n'
        'simulate({method=>"ode",t_start=>10,t_end=>40,n_steps=>30,continue=>1})\n'
    )
    steps, rep = C.parse_protocol(
        text, methods=C._ODE_METHODS, default_method="ode", net_params={}
    )
    assert steps[rep]["t_start"] == 10.0 and steps[rep]["t_end"] == 40.0


def test_parameter_scan_is_never_a_segment_or_representative():
    """GH #69/#179: a ``parameter_scan`` (or ``bifurcate``) is a parameter SWEEP, not a
    single time-series simulate — it writes per-point scan output, not one m.cdat. It
    must never be replayed as a protocol segment nor chosen as the representative, even
    when its t-span is the largest. The representative stays the last real simulate; the
    earlier bug picked the scan and the matrix then compared mismatched segments.
    """
    text = _acts(
        'simulate({method=>"ode",suffix=>"eq",t_end=>30,n_steps=>30})\n'
        'parameter_scan({method=>"ode",t_end=>1000,n_steps=>10,parameter=>"k",'
        "par_min=>1,par_max=>5,n_scan_pts=>5})\n"
    )
    steps, rep = C.parse_protocol(
        text, methods=C._ODE_METHODS, default_method="ode", net_params={}
    )
    # The scan (span 1000 > 30) is dropped entirely; only the real simulate survives.
    assert [s["kind"] for s in steps] == ["sim"]
    assert rep == 0 and steps[rep]["t_end"] == 30.0


# --------------------------------------------------------------------------- #
# Unit (GH #181): parsing parameter-dependent seed-species ICs from a .net.
# --------------------------------------------------------------------------- #
def test_read_net_species_ics_parses_symbolic_seed_ics(tmp_path):
    """Only non-numeric (parameter-dependent) seed-species ICs are returned, 0-based.

    BNG2.pl writes a symbolic seed-species IC as a ``_InitialConc<N>`` / parameter
    reference in the ``.net`` species block (a network species name carries no spaces,
    so the third whitespace field is the IC token). Numeric ICs and generated complexes
    (IC 0) are omitted — those are not re-evaluated on setParameter (GH #181).
    """
    net = tmp_path / "m.net"
    net.write_text(
        "begin parameters\n 1 IgEtot 500\n end parameters\n"
        "begin species\n"
        "    1 Ag(b~0,e0) _InitialConc1\n"  # symbolic -> idx0 0
        "    2 IgE(Fab,Fab) IgEtot\n"  # symbolic (direct param) -> idx0 1
        "    3 Ag(b~1!1,e0).IgE(Fab!1,Fab) 0\n"  # numeric complex -> omitted
        "end species\n"
    )
    ics = C.read_net_species_ics(net)
    assert ics == {0: "_InitialConc1", 1: "IgEtot"}


# --------------------------------------------------------------------------- #
# Unit (GH #179): matrix helpers — continuation-aware spec + session replayability.
# --------------------------------------------------------------------------- #
def test_representative_spec_is_continuation_aware():
    """representative_spec picks the continuation-aware rep + reports dirty_carryover.

    The two-phase continue protocol [0,40] then [40,60] (span 20) keeps phase 1 as the
    representative (span 40 > 20) — the fix for the parse_sim_spec t_start=0 bug that
    would misread phase 2 as [0,60]. dirty_carryover is True (phase 2 inherits phase 1).
    """
    text = _acts(
        'simulate({method=>"ode",t_end=>40,n_steps=>40})\n'
        'setParameter("k",2)\n'
        'simulate({method=>"ode",t_end=>60,n_steps=>20,continue=>1})\n'
    )
    sp = C.representative_spec(text, track="ode", net_params={})
    assert sp is not None
    assert sp["t_start"] == 0.0 and sp["t_end"] == 40.0  # phase 1, not a misread [0,60]
    assert sp["dirty_carryover"] is True
    assert sp["rep_stmt_idx"] is not None and sp["steps"] is not None


def test_protocol_session_replayable_rejects_cross_engine():
    """A network track with an nf segment (cross-engine) is not session-replayable."""
    steps_net = [{"kind": "sim", "method": "ode"}, {"kind": "sim", "method": "ssa"}]
    steps_xeng = [{"kind": "sim", "method": "ode"}, {"kind": "sim", "method": "nf"}]
    steps_nf = [
        {"kind": "sim", "method": "nf"},
        {"kind": "setconc"},
        {"kind": "sim", "method": "nf"},
    ]
    assert C.protocol_session_replayable(steps_net, "ssa") is True
    assert C.protocol_session_replayable(steps_xeng, "nf") is False
    assert C.protocol_session_replayable(steps_xeng, "ode") is False
    assert C.protocol_session_replayable(steps_nf, "nf") is True


# --------------------------------------------------------------------------- #
# Engine: replay reproduces native BNG2.pl's multi-phase trajectory.
# --------------------------------------------------------------------------- #
_BNG = _core.resolve_bng()


def _bngsim_ready() -> bool:
    """Whether the bridge's ``run_bngsim_job`` has a usable bngsim behind it."""
    try:
        import bionetgen  # noqa: PLC0415 — engine-only import

        return bool(getattr(bionetgen, "BNGSIM_AVAILABLE", False))
    except Exception:
        return False


def _bng2_pl():
    return str(_BNG.bng2_pl) if _BNG.ok and _bngsim_ready() else None


# Skips name what the resolver searched (and the fix) rather than just reporting
# absence — see _core.bngpath.
engine = pytest.mark.skipif(
    _bng2_pl() is None,
    reason=_BNG.why_not() if not _BNG.ok else "needs a built bngsim (bionetgen.BNGSIM_AVAILABLE)",
)


def _native_oracle_cdat(bng2: str, bngl_path: Path):
    """Run the model's REAL protocol through BNG2.pl natively; return (time, vals, names)."""
    wd = Path(tempfile.mkdtemp(prefix="p179_oracle_"))
    (wd / "m.bngl").write_text(bngl_path.read_text())
    subprocess.run(
        ["perl", bng2, str(wd / "m.bngl")],
        capture_output=True,
        text=True,
        cwd=str(wd),
        timeout=240,
    )
    cdats = sorted(wd.glob("*.cdat"))
    assert cdats, "BNG2.pl produced no .cdat for the native protocol"
    return C.read_dat(cdats[0])


def _max_rel_err_against_oracle(bng2: str, model_rel: str):
    """run_bngsim_job replay vs native BNG2.pl, max rel err over the rep segment body."""
    mdl = _ODE / model_rel
    out = Path(tempfile.mkdtemp(prefix="p179_replay_"))
    info = C.run_bngsim_job(str(mdl), out, bng2, timeout=240)
    rt, rv, _ = C.read_dat(out / "model.cdat")
    ot, ov, _ = _native_oracle_cdat(bng2, mdl)
    # Map oracle rows onto the replayed segment's time grid; exclude the shared
    # continue=>1 boundary row (the segment join, where the oracle row is the prior
    # segment's end and the replay's is this segment's start — a known output offset).
    idx = [int(np.argmin(np.abs(ot - t))) for t in rt]
    oo = ov[idx]
    a, b = oo[1:], rv[1:]
    nsp = min(a.shape[1], b.shape[1])
    rel = np.abs(a[:, :nsp] - b[:, :nsp]) / np.maximum(np.abs(a[:, :nsp]), 1e-30)
    return info, float(rel.max())


@engine
@pytest.mark.parametrize(
    "model_rel",
    ["continue.bngl", "ExampleModel3.bngl", "test_5.bngl"],
)
def test_replay_reproduces_native_bng2_multiphase(model_rel):
    """The replayed representative segment matches BNG2.pl's native multi-phase run.

    continue.bngl / ExampleModel3 perturb via setConcentration mid-protocol; test_5
    flips a setParameter ("switch" 0->1) mid-protocol — both inherit the prior
    simulate's end state, which option-1 could not reproduce.
    """
    bng2 = _bng2_pl()
    info, max_rel = _max_rel_err_against_oracle(bng2, model_rel)
    pp = info.get("protocol_prefix")
    assert pp and pp.get("replayed") is True and pp.get("segments", 0) >= 2
    assert max_rel < 1e-3, (
        f"{model_rel}: replay diverges from native BNG2.pl (max_rel={max_rel:.2e})"
    )


@engine
def test_setparameter_reinitializes_dependent_seed_species(tmp_path):
    """GH #181: setParameter on a param used in a seed-species IC re-inits the species.

    A seed species ``S0() min(S0init,cap)`` whose IC is a parameter expression; the
    protocol sets ``S0init`` to 999 BEFORE the first simulate, then continues into the
    representative segment. BNG2.pl re-evaluates and re-initializes S0 to 999 (min cap is
    large); the in-process replay must do the same (``Model.set_param`` alone does not,
    the #181 bug). The replayed representative reproduces native BNG2.pl, and the replay
    reports ``reinit_ics >= 1`` (the fix fired).
    """
    bng2 = _bng2_pl()
    mdl = tmp_path / "p181.bngl"
    mdl.write_text(
        "begin model\n"
        "begin parameters\n S0init 10\n cap 1e6\n k 0.5\nend parameters\n"
        "begin species\n S0() min(S0init,cap)\n P() 0\nend species\n"
        "begin reaction rules\n S0() -> P() k\nend reaction rules\n"
        "end model\n"
        "begin actions\n"
        "generate_network({overwrite=>1})\n"
        "saveConcentrations()\n"
        'setParameter("S0init",999)\n'
        'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>10})\n'
        'setParameter("k",0.1)\n'
        'simulate({method=>"ode",t_start=>1,t_end=>5,n_steps=>40,continue=>1})\n'
        "end actions\n"
    )
    out = Path(tempfile.mkdtemp(prefix="p181_replay_"))
    info = C.run_bngsim_job(str(mdl), out, bng2, timeout=120)
    pp = info.get("protocol_prefix")
    assert pp and pp.get("replayed") is True and pp.get("reinit_ics", 0) >= 1
    rt, rv, _ = C.read_dat(out / "model.cdat")
    ot, ov, _ = _native_oracle_cdat(bng2, mdl)
    idx = [int(np.argmin(np.abs(ot - t))) for t in rt]
    oo = ov[idx]
    from _core import differ  # noqa: PLC0415 — engine-only import

    nsp = min(oo.shape[1], rv.shape[1])
    v = differ.deterministic_verdict(oo[1:, :nsp], rv[1:, :nsp])
    assert v["passed"], f"#181 replay diverges from native BNG2.pl (max_rel={v['max_rel']:.2e})"
    # S0 (species 0) at the representative start must reflect the re-init, not the
    # netgen IC of 10: at S0init=999 it decays from 999 over [0,1] before continuing.
    assert oo[0, 0] > 100.0 and abs(oo[0, 0] - rv[0, 0]) <= 1e-6 * max(abs(oo[0, 0]), 1.0)


@engine
def test_labeled_snapshots_restore_their_own_label(tmp_path):
    """GH #186: saveConcentrations/resetConcentrations is a LABELED cache, not one slot.

    The model saves TWO distinct labels — ``"t0"`` (the declared, still-symbolic seed
    IC) and ``"perturbed"`` (a later setConcentration state) — then the representative
    segment is reached via ``resetConcentrations("t0")`` AFTER ``"perturbed"`` was the
    most recent save. The single-slot bug restored the most-recent ``"perturbed"`` save
    (S=0, P=777) onto ``"t0"``, scrambling the seed species (the HarmonicOscillator/IGF1R
    false DIFF: the dimer IC landed on ``~cold`` IGF1). With the labeled cache the rep
    starts from ``"t0"`` — S re-evaluated from its parameter-expression IC, P=0 — matching
    native BNG2.pl. Also covers the symbolic-IC set per label: ``reinit_ics >= 1`` proves
    ``"t0"`` carried S as still-re-evaluable, so the rep re-resolved it from ``Stot``.
    """
    bng2 = _bng2_pl()
    mdl = tmp_path / "p186.bngl"
    mdl.write_text(
        "begin model\n"
        "begin parameters\n NA 6.022e23\n V 1e-21\n conc 0.2\n"
        " Stot = conc*(NA*V)\n k 0.3\nend parameters\n"  # Stot symbolic seed IC (~120.4)
        "begin species\n S() Stot\n P() 0\nend species\n"
        "begin reaction rules\n S() -> P() k\nend reaction rules\n"
        "end model\n"
        "begin actions\n"
        "generate_network({overwrite=>1})\n"
        'saveConcentrations("t0")\n'  # label A: declared, S symbolic
        'simulate({suffix=>"eqA",method=>"ode",t_start=>0,t_end=>5,n_steps=>5})\n'
        'setConcentration("S()",0)\n setConcentration("P()",777)\n'
        'saveConcentrations("perturbed")\n'  # label B: the most-recent save (the trap)
        'simulate({suffix=>"eqB",method=>"ode",t_start=>5,t_end=>6,n_steps=>2,continue=>1})\n'
        'resetConcentrations("t0")\n'  # MUST restore A, not the more recent B
        'simulate({suffix=>"rep",method=>"ode",t_start=>0,t_end=>50,n_steps=>10})\n'  # representative
        "end actions\n"
    )
    out = Path(tempfile.mkdtemp(prefix="p186_replay_"))
    info = C.run_bngsim_job(str(mdl), out, bng2, timeout=120)
    pp = info.get("protocol_prefix")
    assert pp and pp.get("replayed") is True and pp.get("reinit_ics", 0) >= 1
    rt, rv, _ = C.read_dat(out / "model.cdat")
    # The representative is the ``rep``-suffixed segment (largest span). Its native .cdat is
    # NOT cdats[0] (that is ``m_eqA.cdat``), so compare against the rep cdat explicitly.
    owd = Path(tempfile.mkdtemp(prefix="p186_oracle_"))
    (owd / "m.bngl").write_text(mdl.read_text())
    subprocess.run(
        ["perl", bng2, str(owd / "m.bngl")],
        capture_output=True,
        text=True,
        cwd=str(owd),
        timeout=120,
    )
    rep_cdats = sorted(owd.glob("*rep*.cdat"))
    assert rep_cdats, "native run produced no rep .cdat"
    ot, ov, _ = C.read_dat(rep_cdats[0])
    idx = [int(np.argmin(np.abs(ot - t))) for t in rt]
    oo = ov[idx]
    from _core import differ  # noqa: PLC0415 — engine-only import

    nsp = min(oo.shape[1], rv.shape[1])
    v = differ.deterministic_verdict(oo[1:, :nsp], rv[1:, :nsp])
    assert v["passed"], (
        f"#186 labeled-snapshot replay diverges from native (max_rel={v['max_rel']:.2e})"
    )
    # The rep starts from "t0": S (species 0) re-evaluated to Stot (~120), P (species 1) == 0
    # — NOT the "perturbed" save (S=0, P=777) the single-slot bug would have restored.
    assert oo[0, 0] > 1.0 and abs(oo[0, 0] - rv[0, 0]) <= 1e-6 * max(abs(oo[0, 0]), 1.0)
    assert abs(rv[0, 1]) <= 1e-6, f"P should restore to 0 at t0, got {rv[0, 1]} (perturbed leak)"


@engine
def test_netfree_nf_carryover_replays_through_session(tmp_path):
    """GH #179: an all-NF equilibrate->perturb->continue protocol carries agent state.

    seg1 (no ligand) lets R dimerize; saveConcentrations/resetConcentrations checkpoint
    that state; seg2 adds ligand and continues. The representative seg2 must START from
    the carried-over dimerized state (Dimer >> 0 at t0) — proof the stateful NfsimSession
    carried the population across segments (a single-segment run from the declared IC
    would start fully monomeric, Dimer == 0).
    """
    bng2 = _bng2_pl()
    mdl = tmp_path / "nf_carry.bngl"
    mdl.write_text(
        "begin model\n"
        "begin parameters\n kdim 0.004\n kbind 0.01\n R0 300\n L0 0\n Lstim 300\nend parameters\n"
        "begin molecule types\n R(d,l)\n L(r)\nend molecule types\n"
        "begin seed species\n R(d,l) R0\n L(r) L0\nend seed species\n"
        "begin observables\n Molecules Dimer R(d!1).R(d!1)\n Molecules RL R(l!1).L(r!1)\n"
        "end observables\n"
        "begin reaction rules\n R(d)+R(d)->R(d!1).R(d!1) kdim\n"
        " R(l)+L(r)->R(l!1).L(r!1) kbind\nend reaction rules\n"
        "end model\n"
        "begin actions\n"
        'saveConcentrations()\n setConcentration("L(r)",0)\n'
        'simulate({suffix=>"1",method=>"nf",t_start=>0,t_end=>30,n_steps=>30,get_final_state=>1})\n'
        'saveConcentrations()\n resetConcentrations()\n setConcentration("L(r)","Lstim")\n'
        'simulate({suffix=>"2",method=>"nf",t_start=>0,t_end=>30,n_steps=>30})\n'
        "end actions\n"
    )
    out = Path(tempfile.mkdtemp(prefix="p179_nf_"))
    info = C.run_bngsim_job(str(mdl), out, bng2, timeout=120)
    pp = info.get("protocol_prefix")
    assert info["track"] == "nf"
    assert pp and pp.get("replayed") is True and pp.get("segments", 0) >= 2
    t, v, names = C.read_dat(out / "model.gdat")
    assert np.isfinite(v).all()
    dimer = v[:, names.index("Dimer")]
    rl = v[:, names.index("RL")]
    assert dimer[0] > 50.0, "seg2 did not inherit seg1's dimerized state (no carry-over)"
    assert rl[0] == 0.0 and rl[-1] > 0.0, "ligand binding (added in seg2) did not proceed"


@engine
def test_netfree_cross_engine_carryover_falls_back(tmp_path):
    """A cross-engine (ode -> nf) carry-over is not session-replayable -> graceful fallback.

    No NfsimSession can run the prior ODE segment, so the replay raises and
    run_bngsim_job falls back to the option-1 single-segment best-effort path, recording
    replayed=False + replay_error (a transparent degrade, never a silent substitution),
    and still writes the representative artifact.
    """
    bng2 = _bng2_pl()
    mdl = tmp_path / "nf_xeng.bngl"
    mdl.write_text(
        "begin model\n"
        "begin parameters\n kf 0.01\n A0 100\nend parameters\n"
        "begin molecule types\n A(b)\n B(b)\nend molecule types\n"
        "begin seed species\n A(b) A0\n B(b) A0\nend seed species\n"
        "begin observables\n Molecules Afree A(b)\nend observables\n"
        "begin reaction rules\n A(b)+B(b)->A(b!1).B(b!1) kf\nend reaction rules\n"
        "end model\n"
        "begin actions\n"
        "generate_network({overwrite=>1})\n"
        'simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5})\n'
        'simulate({method=>"nf",t_start=>0,t_end=>5,n_steps=>5})\n'
        "end actions\n"
    )
    out = Path(tempfile.mkdtemp(prefix="p179_xeng_"))
    info = C.run_bngsim_job(str(mdl), out, bng2, timeout=120)
    pp = info.get("protocol_prefix")
    assert info["track"] == "nf"
    assert pp and pp.get("replayed") is False and "cross-engine" in pp.get("replay_error", "")
    assert (out / "model.gdat").exists()  # fallback still produced the representative


@engine
def test_matrix_native_oracle_matches_bngsim_replay(tmp_path):
    """GH #179 matrix: native_protocol_oracle reproduces the bngsim multi_segment_replay.

    A dirty_carryover ODE protocol (phase 1 then a setParameter + continue phase) is the
    representative; the matrix drives bngsim through multi_segment_replay and the legacy
    reference through native_protocol_oracle (full BNG2.pl protocol, representative
    segment). The two must agree to integration tolerance, aligned onto the bngsim
    representative grid by nearest time (the GH #180 methodology).
    """
    bng2 = _bng2_pl()
    import importlib.util

    rspec = importlib.util.spec_from_file_location("bng_ode_run", _BNG_PARITY / "bng_ode_run.py")
    R = importlib.util.module_from_spec(rspec)
    rspec.loader.exec_module(R)

    mdl = tmp_path / "matrix_dirty.bngl"
    mdl.write_text(
        "begin model\n"
        "begin parameters\n k 0.3\n A0 100\nend parameters\n"
        "begin species\n A() A0\n B() 0\nend species\n"
        "begin reaction rules\n A() <-> B() k,0.1\nend reaction rules\n"
        "end model\n"
        "begin actions\n"
        "generate_network({overwrite=>1})\n"
        'simulate({method=>"ode",suffix=>"eq",t_start=>0,t_end=>20,n_steps=>40})\n'
        'setParameter("k",1.5)\n'
        'simulate({method=>"ode",t_start=>20,t_end=>60,n_steps=>80,continue=>1})\n'
        "end actions\n"
    )
    text = mdl.read_text()
    wd = Path(tempfile.mkdtemp(prefix="p179_matrix_"))
    net, _, err = C.generate_network(
        text,
        bng2,
        wd / "clean",
        timeout=120,
        gen_network=C._model_gen_network(text),
        state_prefix="",
    )
    assert net, err
    sp = C.representative_spec(text, track="ode", net_params=C.read_net_parameters(net))
    assert sp["dirty_carryover"] is True and sp["t_start"] == 20.0 and sp["t_end"] == 60.0
    result, _info = C.multi_segment_replay(
        net,
        sp["steps"],
        sp["rep_index"],
        track="ode",
        atol=sp["atol"],
        rtol=sp["rtol"],
        seed=None,
        poplevel=0.0,
    )
    bn = (np.asarray(result.time), np.asarray(result.species), list(result.species_names))
    rn = C.native_protocol_oracle(
        text, bng2, wd / "native", track="ode", rep_stmt_idx=sp["rep_stmt_idx"], timeout=120
    )
    status, max_rel, _comment, _metric, _tol, _max_abs = R._compare_ode_multiseg(bn, rn)
    assert status == "pass", f"matrix native oracle diverges from replay (max_rel={max_rel:.2e})"


@engine
def test_single_phase_model_keeps_the_direct_path():
    """A single-phase (non-dirty) model must NOT take the replay path — byte-identical."""
    import glob

    bng2 = _bng2_pl()
    pick = None
    for p in glob.glob(str(_ODE / "*.bngl")):
        try:
            _, pinfo = C.state_setup_prefix(Path(p).read_text(errors="replace"), track="ode")
        except Exception:
            continue
        if pinfo["dirty_carryover"] is False and pinfo["dropped_sims"] == 0:
            pick = p
            break
    assert pick, "no single-phase ODE model found in the corpus"
    out = Path(tempfile.mkdtemp(prefix="p179_direct_"))
    info = C.run_bngsim_job(pick, out, bng2, timeout=240)
    pp = info.get("protocol_prefix")
    assert pp is None or not pp.get("replayed"), "single-phase model wrongly took the replay path"
