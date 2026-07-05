"""GH #177 regression: netgen must replay pre-simulate state-setup actions.

The bug: ``_netgen_bngl`` / ``_writexml_bngl`` stripped the ENTIRE actions block
before handing the model to BNG2.pl, silently dropping every pre-simulate
``setParameter`` / ``setConcentration``. The representative simulate then ran from
the wrong initial state — silently wrong for ~106 corpus models, and catastrophic
for the two whose stripped ``setParameter`` was what kept the run tractable
(``scaling_example``'s ``setParameter("S0",1)`` replacing a 331e6 default
population; ``BSA_v10``'s ``setParameter("kon",0)``).

The fix: :func:`state_setup_prefix` extracts the state-setup actions that PRECEDE
the chosen representative simulate; the netgen/writeXML builders replay them so
BNG2.pl bakes that state into the emitted ``.net`` / ``.xml`` WITHOUT running a sim.

These tests pin:

  * Unit (cheap, always run): the prefix extraction (which actions are kept, which
    simulates are dropped), the single-phase vs multi-phase classification
    (``dirty_carryover``), and that an empty prefix leaves the emitted artifact
    byte-identical (so unaffected models are unchanged).
  * Engine (needs BNG2.pl + perl): the DECISIVE proof — ``generate_xml`` with the
    prefix bakes ``S0=1`` into the XML parameter table, vs the 331e6 default
    without it.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

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
_SCALING = MODELS / "original/bngl_models/my_models/ode/scaling_example.bngl"
_BSA = MODELS / "original/bngl_models/my_models/nf/BSA_v10.bngl"


# --------------------------------------------------------------------------- #
# Unit: prefix extraction (no engine needed — pure text -> kept-action prefix).
# --------------------------------------------------------------------------- #
def test_bsa_v10_keeps_pre_simulate_setparameters():
    """The two ``setParameter`` lines before the representative NF sim are kept."""
    prefix, info = C.state_setup_prefix(_BSA.read_text(), track="nf")
    assert 'setParameter("kon",0)' in prefix
    assert 'setParameter("Mconc_uM",0)' in prefix
    assert info["kept"] == 2
    assert info["dropped_sims"] == 0
    assert info["dirty_carryover"] is False  # single-phase -> fully reproducible


def test_scaling_example_nf_keeps_setparameter_and_drops_intervening_sim():
    """The NF representative's prefix keeps ``setParameter("S0",1)`` and drops the
    intervening ODE sim, which a later resetConcentrations erases (clean)."""
    prefix, info = C.state_setup_prefix(_SCALING.read_text(), track="nf")
    assert 'setParameter("S0",1)' in prefix
    assert info["dropped_sims"] == 1  # the "deterministic" ODE sim before the NF run
    assert info["dirty_carryover"] is False  # erased by the resetConcentrations between


def test_scaling_example_ode_track_does_not_inherit_later_setparameter():
    """The ODE representative ("deterministic") runs the FULL 331e6 population — the
    ``setParameter("S0",1)`` lives AFTER it (it guards the NF run) so must NOT leak
    into the ODE prefix."""
    prefix, info = C.state_setup_prefix(_SCALING.read_text(), track="ode")
    assert "S0" not in prefix
    assert info["dirty_carryover"] is False


def test_multiphase_continue_protocol_flagged_dirty():
    """A ``continue=>1`` chain (representative inherits a prior sim's end state) is
    flagged for full-replay review, not silently run from declared state."""
    bngl = (
        "begin model\nbegin parameters\nk 1\nend parameters\n"
        "begin species\nA() 100\nend species\n"
        "begin reaction rules\nA()->0 k\nend reaction rules\nend model\n"
        "begin actions\n"
        "generate_network({overwrite=>1})\n"
        'simulate({method=>"ode",t_end=>20,n_steps=>20})\n'
        'setParameter("k",2)\n'
        'simulate({method=>"ode",t_end=>100,n_steps=>100,continue=>1})\n'
        "end actions\n"
    )
    prefix, info = C.state_setup_prefix(bngl, track="ode")
    assert 'setParameter("k",2)' in prefix  # still applied best-effort
    assert info["dropped_sims"] == 1
    assert info["dirty_carryover"] is True  # representative inherits the first sim's end state


def test_labeled_save_reset_slot_forces_dirty_flag():
    """A labeled ``resetConcentrations("ckpt")`` is not single-slot modelable, so it
    is conservatively flagged dirty (never mis-read as a clean prefix)."""
    bngl = (
        "begin model\nbegin parameters\nk 1\nend parameters\n"
        "begin species\nA() 100\nend species\n"
        "begin reaction rules\nA()->0 k\nend reaction rules\nend model\n"
        "begin actions\n"
        "generate_network({overwrite=>1})\n"
        'saveConcentrations("ckpt")\n'
        'simulate({method=>"ode",t_end=>20,n_steps=>20})\n'
        'resetConcentrations("ckpt")\n'
        'setParameter("k",2)\n'
        'simulate({method=>"ode",t_end=>100,n_steps=>100})\n'
        "end actions\n"
    )
    _prefix, info = C.state_setup_prefix(bngl, track="ode")
    assert info["dirty_carryover"] is True


def test_no_pre_simulate_state_yields_empty_prefix():
    """A plain single-simulate model keeps nothing — empty prefix, no flag."""
    bngl = (
        "begin model\nbegin parameters\nk 1\nend parameters\n"
        "begin species\nA() 100\nend species\n"
        "begin reaction rules\nA()->0 k\nend reaction rules\nend model\n"
        "begin actions\ngenerate_network({overwrite=>1})\n"
        'simulate({method=>"ode",t_end=>10,n_steps=>10})\nend actions\n'
    )
    prefix, info = C.state_setup_prefix(bngl, track="ode")
    assert prefix == ""
    assert info == {
        "representative_found": True,
        "kept": 0,
        "dropped_sims": 0,
        "dirty_carryover": False,
    }


def test_empty_prefix_leaves_netgen_and_writexml_byte_identical():
    """An unaffected model's emitted .net/.xml must be unchanged by this feature."""
    text = _BSA.read_text()
    assert C._netgen_bngl(text, None, "") == C._netgen_bngl(text, None)
    assert C._writexml_bngl(text, "") == C._writexml_bngl(text)


def test_state_prefix_inserted_before_appended_terminal():
    """The kept prefix must sit BEFORE the appended writeXML()/generate_network — for
    the .net path, state set after generate_network is NOT baked in (order matters)."""
    out = C._writexml_bngl("begin model\nend model\n", 'setParameter("S0",1)')
    assert out.index('setParameter("S0",1)') < out.index("writeXML()")
    out2 = C._netgen_bngl("begin model\nend model\n", None, 'setParameter("S0",1)')
    assert out2.index('setParameter("S0",1)') < out2.index("generate_network")


# --------------------------------------------------------------------------- #
# GH #176 follow-up: a netgen-affecting setOption must survive into netgen.
# --------------------------------------------------------------------------- #
def test_netgen_preserves_NumberPerQuantityUnit_setoption():
    """``setOption("NumberPerQuantityUnit",N)`` (energy-BNG, before ``begin model``)
    must survive into the netgen/writeXML body — it sets the bimolecular unit
    conversion ``1/(N·V)``; stripping it makes BNG2.pl default to ``1/V`` and emit
    rate constants ~N too large (catalysis became unintegrable). Other actions —
    including ``setOption("SpeciesLabel",...)``, which changes only internal labels —
    stay stripped."""
    text = (
        'setOption("NumberPerQuantityUnit",6.0221e23)\n'
        'setOption("SpeciesLabel","HNauty")\n'
        "begin model\nbegin parameters\nk 1\nend parameters\n"
        "begin species\nA() 100\nend species\n"
        "begin reaction rules\nA()->0 k\nend reaction rules\nend model\n"
        "begin actions\n"
        'setOption("NumberPerQuantityUnit",1)\n'  # inside actions block -> dropped
        "generate_network({overwrite=>1})\n"
        'simulate({method=>"ode",t_end=>10,n_steps=>10})\n'
        "end actions\n"
    )
    for emitted in (C._netgen_bngl(text), C._writexml_bngl(text)):
        # The model-header unit option is kept (exactly once — the in-actions one is dropped).
        assert emitted.count('setOption("NumberPerQuantityUnit",6.0221e23)') == 1
        assert 'setOption("NumberPerQuantityUnit",1)' not in emitted  # in-actions dup dropped
        # SpeciesLabel is not unit-affecting -> still stripped.
        assert "SpeciesLabel" not in emitted
        # The runtime actions are still stripped.
        assert "simulate(" not in emitted


# --------------------------------------------------------------------------- #
# Engine: the decisive proof — the prefix actually bakes the state into the XML.
# --------------------------------------------------------------------------- #
def _bng2_pl():
    """The bundled BNG2.pl path (+ perl), or None when unavailable."""
    import shutil

    bngpath = None
    try:
        from bionetgen.main import get_conf

        bngpath = get_conf().get("bngpath")
    except Exception:
        bngpath = os.environ.get("BNGPATH") or os.environ.get("BNG2_PL")
    if not bngpath:
        return None
    p = Path(bngpath)
    bng2 = p / "BNG2.pl" if p.is_dir() else p
    if not bng2.exists() or shutil.which("perl") is None:
        return None
    return str(bng2)


engine = pytest.mark.skipif(_bng2_pl() is None, reason="needs a bundled BNG2.pl + perl")


def _xml_param(xml: str, pid: str) -> str | None:
    m = re.search(rf'<Parameter id="{pid}"[^>]*?value="([^"]*)"', xml)
    return m.group(1) if m else None


@engine
def test_generate_xml_bakes_setparameter_state_into_the_xml():
    """``setParameter("S0",1)`` in the prefix drops S0 from 331e6 to 1 in the emitted
    XML — the difference between an instant NFsim run and 331M never-finishing
    particles (GH #177's catastrophic case)."""
    bng2 = _bng2_pl()
    text = _SCALING.read_text()
    prefix, _ = C.state_setup_prefix(text, track="nf")

    with_dir = Path(tempfile.mkdtemp(prefix="p177_with_"))
    without_dir = Path(tempfile.mkdtemp(prefix="p177_without_"))
    xml_with, _, err_w = C.generate_xml(text, bng2, with_dir, timeout=180, state_prefix=prefix)
    xml_without, _, err_wo = C.generate_xml(text, bng2, without_dir, timeout=180)
    assert xml_with is not None, err_w
    assert xml_without is not None, err_wo

    assert _xml_param(xml_with.read_text(), "S0") == "1"
    # default population baked in when the prefix is dropped (the bug)
    assert float(_xml_param(xml_without.read_text(), "S0")) == pytest.approx(331e6)


@engine
def test_catalysis_netgen_emits_correct_bimolecular_unit_conversion():
    """GH #176 follow-up — the DECISIVE proof: with ``NumberPerQuantityUnit``
    preserved, BNG2.pl emits the correct ``1/(NA·volC)`` bimolecular conversion (not
    the ``1/volC`` default that made catalysis unintegrable), and bngsim integrates
    the resulting network."""
    import bngsim

    bng2 = _bng2_pl()
    catalysis = MODELS / "original/bngl_models/my_models/ode/catalysis.bngl"
    wd = Path(tempfile.mkdtemp(prefix="cata176_"))
    net, _, err = C.generate_network(catalysis.read_text(), bng2, wd, timeout=180)
    assert net is not None, err
    net_text = net.read_text()
    # The NA-scaled conversion is present; the broken 1/volC default is not.
    assert "1/(6.0221e+23*volC)" in net_text
    assert "unit_conversion=1/volC" not in net_text
    # And the corrected network integrates (it did not before the fix).
    m = bngsim.Model.from_net(str(net))
    r = bngsim.Simulator(m, method="ode").run(t_span=(0, 3600), n_points=121, rtol=1e-7, atol=1e-3)
    assert r.n_times == 121


_BARE_GEN_MODEL = (
    "begin model\n"
    "begin parameters\n k1 1.0\nend parameters\n"
    "begin species\n A() 100\nend species\n"
    "begin observables\n Molecules Atot A()\nend observables\n"
    "begin reaction rules\n A() -> A() + A() k1\nend reaction rules\n"
    "end model\n"
    # A bare generate_network() (no overwrite=>1), like LV.bngl/birth-death.bngl —
    # _model_gen_network preserves this exact call, so the netgen-only copy injects it.
    "begin actions\ngenerate_network()\nsimulate({method=>\"ode\",t_end=>1,n_steps=>10})\nend actions\n"
)


@engine
def test_generate_network_reruns_into_a_dirty_workdir():
    """GH #192 regression: a re-sweep must regenerate, not ABORT on a stale .net.

    When the model's OWN ``generate_network(...)`` call lacks ``overwrite=>1`` (and
    netgen preserves it to keep any caps), a second ``generate_network`` into a
    workdir that still holds the first run's ``model.net`` made BNG2.pl ABORT
    ("Previously generated ./model.net exists") — surfacing as a truncated
    ``BNG2.pl netgen failed: ... at line N``. This is exactly what crashed 12
    golden models on the #192 full re-sweep into the existing tree. The fix clears
    the stale artifact first, so the second call succeeds byte-for-byte like the
    first."""
    bng2 = _bng2_pl()
    wd = Path(tempfile.mkdtemp(prefix="gh192_dirty_netgen_"))
    gen = C._model_gen_network(_BARE_GEN_MODEL)
    assert gen == "generate_network()", gen  # preserved bare, no overwrite=>1

    net1, _, err1 = C.generate_network(_BARE_GEN_MODEL, bng2, wd, timeout=120, gen_network=gen)
    assert net1 is not None, err1
    first = net1.read_text()

    # Second run into the SAME (now dirty) workdir — pre-fix this ABORTed.
    net2, _, err2 = C.generate_network(_BARE_GEN_MODEL, bng2, wd, timeout=120, gen_network=gen)
    assert net2 is not None, f"re-run into dirty workdir failed: {err2!r}"
    assert net2.read_text() == first
