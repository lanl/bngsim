"""Tests for bngsim.convert._protocol — the BNGL simulation-protocol IR + parser.

The shippable bngl-actions parser that recovers the simulation protocol a
``.net`` discarded at network-generation time, so the converter can hand a
consumer a *faithful* deliverable (GH #211, Option 3). Covers the action-verb
vocabulary, hash/positional argument parsing (incl. literal arithmetic), the
bare-vs-``begin actions`` block forms, line continuations/comments/terminators,
the build/IO-directive drop set, the strict-vs-lossy unknown-action boundary,
sequential continuation + multi-stage scans, the ``ProtocolSpec`` query helpers,
and JSON round-trip.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim.convert._protocol import (
    Experiment,
    ProtocolSpec,
    StateChange,
    combine_protocols,
    parse_bngl_protocol,
)

_BNGSIM = Path(__file__).resolve().parents[2]
_BNGL = _BNGSIM / "benchmarks" / "models" / "bngl"


# ─── Single simulate ────────────────────────────────────────────────────────


_SINGLE = """\
begin model
end model
generate_network({overwrite=>1})
simulate({method=>"ode",t_end=>3000,n_steps=>300,print_functions=>1})
"""


def test_single_simulate() -> None:
    p = parse_bngl_protocol(_SINGLE)
    assert p.dropped == ("generate_network",)
    assert len(p.steps) == 1
    e = p.experiments[0]
    assert e.kind == "simulate" and e.method == "ode"
    assert e.t_span == (0.0, 3000.0) and e.n_points == 301
    assert e.extra.get("print_functions") == 1  # unconsumed hash keys kept for provenance


# ─── Sequential continuation (two simulates, a setParameter between) ─────────


_SEQUENTIAL = """\
begin model
end model
generate_network();
simulate({method=>"ode",t_start=>0,t_end=>600,n_steps=>10})
setParameter("Ligand_isPresent",1)
simulate({method=>"ode",t_start=>0,t_end=>60,n_steps=>60})
"""


def test_sequential_continuation() -> None:
    p = parse_bngl_protocol(_SEQUENTIAL)
    exps = p.experiments
    assert len(exps) == 2
    assert exps[0].t_span == (0.0, 600.0) and exps[0].n_points == 11
    assert exps[1].t_span == (0.0, 60.0) and exps[1].n_points == 61
    # The setParameter precedes the *second* experiment, not the first.
    assert p.state_changes_before(exps[0]) == []
    before2 = p.state_changes_before(exps[1])
    assert len(before2) == 1
    assert before2[0] == StateChange("set_parameter", "Ligand_isPresent", 1.0)


# ─── Multi-stage scans with resetConcentrations ─────────────────────────────


_MULTISTAGE = """\
begin model
end model
generate_network({overwrite=>1})
setParameter("coopRT",0)
setParameter("coopB",0)
parameter_scan({method=>"ode",t_end=>10,n_steps=>10,par_min=>0,par_max=>1,n_scan_pts=>20,reset_conc=>1,parameter=>"log_P_ox",suffix=>"00"})
resetConcentrations()
setParameter("coopRT",1)
parameter_scan({method=>"ode",t_end=>10,n_steps=>10,par_min=>0,par_max=>1,n_scan_pts=>20,parameter=>"log_P_ox",suffix=>"10"})
resetConcentrations()
"""


def test_multistage_scan() -> None:
    p = parse_bngl_protocol(_MULTISTAGE)
    scans = [e for e in p.experiments if e.is_scan]
    assert len(scans) == 2
    s0 = scans[0]
    assert s0.scan_parameter == "log_P_ox"
    assert (s0.scan_min, s0.scan_max, s0.scan_points) == (0.0, 1.0, 20)
    assert s0.reset_between is True and s0.label == "00"
    # The reset clears accumulated overrides: before scan[1] only coopRT=1 holds
    # (the pre-reset coopRT=0/coopB=0 are gone).
    before = p.state_changes_before(scans[1])
    assert before == [StateChange("set_parameter", "coopRT", 1.0)]


# ─── Argument parsing edge cases ────────────────────────────────────────────


def test_arithmetic_and_string_args() -> None:
    p = parse_bngl_protocol(
        'begin model\nend model\nsimulate({method=>"ode",t_end=>3600*5,n_steps=>100})\n'
    )
    e = p.experiments[0]
    assert e.t_span[1] == 18000.0  # literal arithmetic evaluated


def test_setconcentration_numeric_and_expr() -> None:
    p = parse_bngl_protocol(
        'begin model\nend model\n'
        'setConcentration("A(b)",100)\n'
        'setConcentration("B",A0*2)\n'
        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
    )
    sc = [s for s in p.steps if isinstance(s, StateChange)]
    assert sc[0] == StateChange("set_concentration", "A(b)", 100.0)
    # A parameter-referencing value can't be resolved here — kept verbatim.
    assert sc[1].kind == "set_concentration" and sc[1].target == "B"
    assert sc[1].value is None and sc[1].value_expr == "A0*2"


def test_begin_actions_block_and_comments() -> None:
    text = """\
begin model
  begin parameters
    k1 1.0   # a rate constant, not an action
  end parameters
end model
begin actions
  # leading comment
  simulate({method=>"ssa",\\
            t_end=>50,n_steps=>50});
end actions
"""
    p = parse_bngl_protocol(text)
    assert len(p.experiments) == 1
    e = p.experiments[0]
    assert e.method == "ssa" and e.t_span == (0.0, 50.0)  # continuation joined


def test_model_block_params_not_actions() -> None:
    # `simulate`-looking lines never appear in a model block, but ensure a
    # parameter assignment inside begin/end model is not mistaken for an action.
    text = "begin model\nbegin parameters\n simulate_rate 2.0\nend parameters\nend model\n"
    p = parse_bngl_protocol(text)
    assert p.experiments == []


# ─── Capability boundary: build verbs drop, unknown verbs gate ──────────────


def test_build_directives_dropped() -> None:
    text = (
        "begin model\nend model\n"
        "generate_network({overwrite=>1})\n"
        'writeSBML({})\n'
        "saveConcentrations()\n"
        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
    )
    p = parse_bngl_protocol(text)
    assert "generate_network" in p.dropped and "writeSBML" in p.dropped
    # saveConcentrations IS a protocol state action, not a build directive.
    assert any(isinstance(s, StateChange) and s.kind == "save_concentrations" for s in p.steps)


def test_unknown_action_strict_raises_lossy_warns() -> None:
    text = (
        "begin model\nend model\nfrobnicate({x=>1})\n"
        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
    )
    with pytest.raises(bngsim.ConversionError, match="unrecognized BNGL action"):
        parse_bngl_protocol(text, strict=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = parse_bngl_protocol(text, strict=False)
    assert any("frobnicate" in str(w.message) for w in caught)
    assert "frobnicate" in p.dropped
    assert len(p.experiments) == 1  # the valid simulate still parsed


# ─── Action disposition: drop / flag-lossy / quit-truncate (GH #226) ─────────


def _hdr(body: str) -> str:
    return "begin model\nend model\n" + body


_SIM1 = 'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'


def test_pure_io_siblings_drop_cleanly_under_strict() -> None:
    """writeBNGL / setOutputDir are tooling siblings of writeNET/writeSBML/
    readFile and must drop cleanly — no hard-error, no warning — even under the
    default strict mode (the consistency bug from GH #226)."""
    text = _hdr('writeBNGL()\nsetOutputDir("out")\nreadFile("m.bngl")\n' + _SIM1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        p = parse_bngl_protocol(text, strict=True)
    assert set(p.dropped) == {"writeBNGL", "setOutputDir", "readFile"}
    assert p.lossy == ()
    assert len(p.experiments) == 1


@pytest.mark.parametrize(
    "action,verb",
    [
        ('setVolume("comp",2)', "setVolume"),
        ('substanceUnits("Number")', "substanceUnits"),
        ('readModel("other.bngl")', "readModel"),
        ('readNetwork("other.net")', "readNetwork"),
        ('readSBML("other.xml")', "readSBML"),
        ('setOption("NumberPerQuantityUnit",6.0221e23)', "setOption"),
    ],
)
def test_fidelity_affecting_actions_flag_lossy(action: str, verb: str) -> None:
    """Result-changing actions BNGsim does not execute must REFUSE under strict
    (not silently drop) and warn + record on the lossy channel otherwise."""
    text = _hdr(action + "\n" + _SIM1)
    with pytest.raises(bngsim.ConversionError, match="affects results"):
        parse_bngl_protocol(text, strict=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = parse_bngl_protocol(text, strict=False)
    assert verb in p.lossy and verb not in p.dropped
    assert any(verb in str(w.message) for w in caught)
    assert len(p.experiments) == 1  # the valid simulate still parsed


def test_setoption_cosmetic_drops_while_result_changing_flags() -> None:
    """setOption is classified by option name: SpeciesLabel is cosmetic (clean
    drop), NumberPerQuantityUnit changes the unit conversion (lossy)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = parse_bngl_protocol(_hdr('setOption("SpeciesLabel","HNauty")\n' + _SIM1), strict=True)
    assert p.dropped == ("setOption",) and p.lossy == ()


def test_quit_truncates_action_stream() -> None:
    """quit() halts BNG2.pl action processing; the parser truncates there so it
    never replays trailing actions — including ones that would otherwise raise."""
    text = _hdr(
        _SIM1
        + "quit()\n"
        + 'simulate({method=>"ode",t_end=>999,n_steps=>1})\n'
        + "frobnicate({x=>1})\n"  # unknown verb AFTER quit must NOT raise
    )
    p = parse_bngl_protocol(text, strict=True)
    assert len(p.experiments) == 1 and p.experiments[0].t_span == (0.0, 1.0)
    assert p.dropped == ("quit",) and p.lossy == ()


def test_lossy_survives_json_round_trip() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        p = parse_bngl_protocol(_hdr('setVolume("c",2)\n' + _SIM1), strict=False)
    again = ProtocolSpec.from_json(p.to_json())
    assert again.lossy == p.lossy == ("setVolume",)
    assert again.to_dict() == p.to_dict()


# ─── ProtocolSpec query helpers ─────────────────────────────────────────────


def test_primary_experiment_prefers_deterministic() -> None:
    text = (
        "begin model\nend model\n"
        'simulate({method=>"ssa",t_end=>5,n_steps=>5})\n'
        'simulate({method=>"ode",t_end=>200,n_steps=>20})\n'
    )
    p = parse_bngl_protocol(text)
    pe = p.primary_experiment()
    assert pe is not None and pe.method == "ode" and pe.t_span == (0.0, 200.0)


def test_empty_protocol() -> None:
    p = parse_bngl_protocol("begin model\nend model\n")
    assert p.is_empty and p.primary_experiment() is None


# ─── Serialization ──────────────────────────────────────────────────────────


def test_json_round_trip() -> None:
    p = parse_bngl_protocol(_MULTISTAGE)
    again = ProtocolSpec.from_json(p.to_json())
    assert again.to_dict() == p.to_dict()
    assert again.experiments[0].scan_parameter == "log_P_ox"


def test_protocolspec_dataclasses_frozen() -> None:
    import dataclasses

    e = Experiment("simulate", "ode", (0.0, 1.0), 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.method = "ssa"  # type: ignore[misc]


# ─── Real corpus smoke (guarded) ────────────────────────────────────────────


def test_real_benchmark_corpus_parses() -> None:
    if not _BNGL.is_dir():
        pytest.skip("benchmark bngl corpus not present")
    files = sorted(_BNGL.rglob("*.bngl"))
    if not files:
        pytest.skip("no benchmark bngl files present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        for f in files:
            p = parse_bngl_protocol(f, strict=False)
            # every parse must JSON-round-trip
            assert ProtocolSpec.from_json(p.to_json()).to_dict() == p.to_dict(), f


# ─── combine_protocols: multi-experiment / multi-file composition (GH #222) ──


def test_combine_protocols_concatenates_with_reset_boundary() -> None:
    """Two independent specs merge in order, separated by a resetConcentrations
    boundary (they are not continuations of one another)."""
    p1 = ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 10.0), 11),), source="a")
    p2 = ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 50.0), 51),), source="b")
    merged = combine_protocols([p1, p2])
    assert len(merged.experiments) == 2
    assert [type(s).__name__ for s in merged.steps] == [
        "Experiment", "StateChange", "Experiment"
    ]
    assert merged.steps[1].kind == "reset_concentrations"
    assert merged.source == "a + b"


def test_combine_protocols_no_reset_when_disabled() -> None:
    p1 = ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 10.0), 11),))
    p2 = ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 50.0), 51),))
    merged = combine_protocols([p1, p2], reset_between=False)
    assert [type(s).__name__ for s in merged.steps] == ["Experiment", "Experiment"]


def test_combine_protocols_single_spec_returned_verbatim() -> None:
    """A single spec is returned unchanged — no spurious reset is introduced, so
    the common one-file round trip stays exact."""
    only = ProtocolSpec(
        steps=(
            Experiment("simulate", "ode", (0.0, 10.0), 11),
            StateChange(kind="set_parameter", target="k", value=2.0),
            Experiment("simulate", "ode", (0.0, 5.0), 6),
        ),
        source="x",
    )
    assert combine_protocols([only]) is only


def test_combine_protocols_skips_empty_specs() -> None:
    empty = ProtocolSpec(steps=())
    real = ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 10.0), 11),))
    assert combine_protocols([empty, real]) is real
    # All-empty → an empty spec (its provenance preserved), never a crash.
    assert combine_protocols([empty, empty]).is_empty


def test_combine_protocols_concatenates_dropped_and_lossy() -> None:
    """Provenance — both the dropped (pure IO) and lossy (fidelity-affecting)
    channels survive the merge so the combined deliverable reports every loss."""
    p1 = ProtocolSpec(
        steps=(Experiment("simulate", "ode", (0.0, 10.0), 11),),
        dropped=("writeBNGL",), lossy=("setVolume",),
    )
    p2 = ProtocolSpec(
        steps=(Experiment("simulate", "ode", (0.0, 50.0), 51),),
        dropped=("writeNET",), lossy=("substanceUnits",),
    )
    merged = combine_protocols([p1, p2])
    assert merged.dropped == ("writeBNGL", "writeNET")
    assert merged.lossy == ("setVolume", "substanceUnits")


def test_combine_protocols_no_double_reset() -> None:
    """A spec already ending in resetConcentrations is not given a second one."""
    p1 = ProtocolSpec(steps=(
        Experiment("simulate", "ode", (0.0, 10.0), 11),
        StateChange(kind="reset_concentrations"),
    ))
    p2 = ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 5.0), 6),))
    merged = combine_protocols([p1, p2])
    resets = [s for s in merged.steps if isinstance(s, StateChange)
              and s.kind == "reset_concentrations"]
    assert len(resets) == 1
