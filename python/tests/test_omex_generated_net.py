"""The packer refuses a .net that is not the network its .bngl generates (issue #1683).

The export guide gave ``BNG2.pl model.bngl`` and then ``bngsim-omex pack
model.net --bngl model.bngl``. At a ``simulate`` with no ``suffix``, BNG2.pl
writes ``model.net`` again with the state of that moment, once a run or an
action has moved it, and the packer took that file for the state at
``generate_network`` and replayed the protocol on top of it: for A -> 0 at
k = 0.5 from A = 100, an ``addConcentration`` of 50 then one run, the archive
started at 200 for BNG2.pl's 150, with every gate level PASS. The packer now
checks the .net's parameters and seed species against what the .bngl declares,
with the actions before its ``generate_network`` applied, and refuses a .net
that differs; the guide writes the network with ``Model.from_bngl(net_out=)``.

The .net files here are written by hand, as BNG2.pl writes them, so every test
but the last runs without BNG2.pl.
"""

from __future__ import annotations

import math
import zipfile

import bngsim
import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim._bngpath import resolve_bng
from bngsim.convert import net_to_omex, net_to_sbml, read_sedml_protocol

_BNG = resolve_bng()

MODEL = """begin model
begin parameters
 k 0.5
 A0 100
end parameters
begin seed species
 A() A0
end seed species
begin reaction rules
 A() -> 0 k
end reaction rules
end model
"""
ADD_THEN_RUN = (
    "generate_network({overwrite=>1})\n"
    'addConcentration("A()",50)\n'
    'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
)


def net(k="0.5", a0="100", amount="A0"):
    """A .net as BNG2.pl writes it: at generation the seed amount is the
    declared expression; written again at a run, it is the number."""
    return (
        "# Created by BioNetGen 2.9.3\n"
        f"begin parameters\n    1 k   {k}  # Constant\n    2 A0  {a0}  # Constant\n"
        "end parameters\n"
        f"begin species\n    1 A() {amount}\nend species\n"
        "begin reactions\n    1 1 0 k #_R1\nend reactions\n"
    )


def write(tmp_path, bngl, net_text):
    (tmp_path / "model.bngl").write_text(bngl)
    (tmp_path / "model.net").write_text(net_text)
    return tmp_path / "model.net", tmp_path / "model.bngl"


def pack(net_path, bngl_path, **kw):
    out = net_path.with_suffix(".omex")
    return out, net_to_omex(
        net_path, out, bngl=bngl_path, gate="L1", created="2026-01-01T00:00:00Z", **kw
    )


def first_run(omex):
    """The archive's first run, replayed on its own SBML: (start, end) of A."""
    z = zipfile.ZipFile(omex)
    spec = read_sedml_protocol(z.read("simulation.sedml").decode())
    model = bngsim.Model.from_sbml_string(z.read("model.xml").decode())
    run = bngsim.run_protocol(model, spec, rtol=1e-10, atol=1e-12)[0].result
    return float(run.species[0, 0]), float(run.species[-1, 0])


# ── The issue's shape: an add, then one run ──────────────────────────────────


def test_the_net_bng2_pl_wrote_at_the_run_is_refused(tmp_path):
    """BNG2.pl's model.net holds A = 150, the state at the run. Replayed with
    the add on top, the archive started at 200."""
    paths = write(tmp_path, MODEL + ADD_THEN_RUN, net(amount="150"))
    with pytest.raises(ConversionError, match=r"A\(\) is 150 there, 100 as declared"):
        pack(*paths)


def test_the_generated_net_packs_and_replays_from_its_start(tmp_path):
    """The network as generate_network writes it: the archive's run starts at
    150 and ends at 150 e^-0.5, the closed form of A -> 0 (rel 1e-8: the
    replay runs at rtol 1e-10)."""
    out, report = pack(*write(tmp_path, MODEL + ADD_THEN_RUN, net()))
    assert report.ok
    start, end = first_run(out)
    assert start == pytest.approx(150.0, rel=1e-12)
    assert end == pytest.approx(150.0 * math.exp(-0.5), rel=1e-8)


def test_a_parameter_set_after_generation_shows_too(tmp_path):
    """BNG2.pl writes the parameters of the moment as well: k = 2 after a
    setParameter that the protocol replays."""
    bngl = MODEL + (
        "generate_network({overwrite=>1})\n"
        'setParameter("k",2)\n'
        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
    )
    with pytest.raises(ConversionError, match="parameter k is 2 there, 0.5 as declared"):
        pack(*write(tmp_path, bngl, net(k="2")))


def test_net_to_sbml_says_it_too_and_converts(tmp_path):
    """The check is the network channel's, which the packer calls. The SBML is
    the .net as given, a model in its own right, so net_to_sbml warns and notes
    it in lossy; the packer, which writes the protocol onto it, refuses."""
    with pytest.warns(ConversionWarning, match="is not the network model.bngl generates"):
        report = net_to_sbml(
            *write(tmp_path, MODEL + ADD_THEN_RUN, net(amount="150"))[:1],
            bngl=tmp_path / "model.bngl",
        )
    assert report.output_text and any("is not the network" in n for n in report.lossy)


def test_allow_lossy_packs_it_with_a_warning_and_a_note(tmp_path):
    paths = write(tmp_path, MODEL + ADD_THEN_RUN, net(amount="150"))
    with pytest.warns(ConversionWarning, match="is not the network model.bngl generates"):
        _, report = pack(*paths, strict=False)
    assert any("issue #1683" in note for note in report.lossy)


# ── What generation itself changes is the network's ──────────────────────────


@pytest.mark.parametrize(
    ("before", "held"),
    [
        ('setParameter("A0",40)\n', {"a0": "40"}),
        ('setConcentration("A()",7)\n', {"amount": "7"}),
        ('setParameter("k",3)\n', {"k": "3"}),
    ],
    ids=["a-seed-parameter", "a-seed-amount", "a-rate-constant"],
)
def test_an_action_before_generate_network_is_in_the_network(tmp_path, before, held):
    """Control: an action before generate_network is in the network it writes,
    so its value is the one declared."""
    bngl = (
        MODEL
        + before
        + "generate_network({overwrite=>1})\n"
        + ('simulate({method=>"ode",t_end=>1,n_steps=>1})\n')
    )
    _, report = pack(*write(tmp_path, bngl, net(**held)))
    assert report.ok


def test_the_declared_network_packs_with_no_run(tmp_path):
    """Control: the declared network itself packs, with the default protocol a
    file with no run gets."""
    paths = write(tmp_path, MODEL + "generate_network({overwrite=>1})\n", net())
    with pytest.warns(ConversionWarning, match="no simulation protocol available"):
        _, report = pack(*paths)
    assert report.ok


# ── The documented pipeline, through BNG2.pl ─────────────────────────────────


@pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")
def test_the_documented_pipeline(tmp_path):
    """BNG2.pl's own model.net is refused; Model.from_bngl's, which the guide
    now gives, packs, and its archive starts where BNG2.pl's run does."""
    import subprocess

    bngl = tmp_path / "model.bngl"
    bngl.write_text(MODEL + ADD_THEN_RUN)
    subprocess.run(
        ["perl", str(_BNG.bng2_pl), "model.bngl"], cwd=tmp_path, check=True, capture_output=True
    )
    with pytest.raises(ConversionError, match="is not the network"):
        pack(tmp_path / "model.net", bngl)
    ours = tmp_path / "ours"
    ours.mkdir()
    (ours / "model.bngl").write_text(bngl.read_text())
    with pytest.warns(Warning):  # from_bngl says it leaves the later actions out
        bngsim.Model.from_bngl(ours / "model.bngl", net_out=ours / "model.net", cache=False)
    out, report = pack(ours / "model.net", ours / "model.bngl")
    assert report.ok
    assert first_run(out)[0] == pytest.approx(150.0, rel=1e-12)


# ── A species the network generated, filled by a run (Fable's addition) ───────


MODEL_AB = """begin model
begin parameters
 k 0.5
 A0 100
end parameters
begin seed species
 A() A0
end seed species
begin reaction rules
 A() -> B() k
end reaction rules
end model
"""


def net_ab(a="A0", b="0"):
    return (
        "# Created by BioNetGen 2.9.3\n"
        "begin parameters\n    1 k   0.5  # Constant\n    2 A0  100  # Constant\n"
        "end parameters\n"
        f"begin species\n    1 A() {a}\n    2 B() {b}\nend species\n"
        "begin reactions\n    1 1 2 k #_R1\nend reactions\n"
    )


def test_a_generated_species_a_run_filled_is_refused(tmp_path):
    """The seed back at its declared value (a run, then setConcentration of it
    to A0 and a run), but B(), which the network generated at 0, holds what the
    first run made: the file was written after a run."""
    actions = (
        'generate_network({overwrite=>1})\nsimulate({method=>"ode",t_end=>1,n_steps=>1})\n'
        'setConcentration("A()","A0")\nsimulate({method=>"ode",t_end=>1,n_steps=>1})\n'
    )
    paths = write(tmp_path, MODEL_AB + actions, net_ab(a="A0", b="39.3469340287367"))
    with pytest.raises(ConversionError, match=r"B\(\) is 39\.3469340287367 there, 0 as generated"):
        pack(*paths)


def test_the_generated_species_check_waits_for_every_seed(tmp_path):
    """A seed this reading cannot evaluate (an amount that is no expression of
    the network's) leaves the check off: a filled B() is not refused on it."""
    model = MODEL_AB.replace(" A() A0\n", " A() A_declared_elsewhere\n")
    paths = write(
        tmp_path, model + "generate_network({overwrite=>1})\n", net_ab(a="100", b="39.3")
    )
    with pytest.warns(ConversionWarning, match="no simulation protocol available"):
        _, report = pack(*paths)
    assert report.ok


# ── BNG2.pl's own files, on nine protocol shapes (Fable's addition) ──────────

SHAPES = {
    "run-suffix-add-run": (
        'simulate({method=>"ode",suffix=>"e",t_end=>10,n_steps=>1})\naddConcentration("A()",50)\nsimulate({method=>"ode",t_end=>10,n_steps=>1})',
        True,
    ),
    "add-run": ('addConcentration("A()",50)\nsimulate({method=>"ode",t_end=>1,n_steps=>1})', True),
    "run-run": (
        'simulate({method=>"ode",t_end=>10,n_steps=>1})\nsimulate({method=>"ode",t_end=>10,n_steps=>1})',
        True,
    ),
    "set-conc-run": (
        'setConcentration("A()",30)\nsimulate({method=>"ode",t_end=>1,n_steps=>1})',
        True,
    ),
    "set-param-run": ('setParameter("k",2)\nsimulate({method=>"ode",t_end=>1,n_steps=>1})', True),
    "run-reset-run": (
        'simulate({method=>"ode",t_end=>10,n_steps=>1})\nresetConcentrations()\nsimulate({method=>"ode",t_end=>10,n_steps=>1})',
        False,
    ),
    "suffixes": (
        'simulate({method=>"ode",suffix=>"a",t_end=>10,n_steps=>1})\nsimulate({method=>"ode",suffix=>"b",t_end=>10,n_steps=>1})',
        False,
    ),
    "prefix": (
        'addConcentration("A()",50)\nsimulate({method=>"ode",prefix=>"x",t_end=>1,n_steps=>1})',
        False,
    ),
    "one-run": ('simulate({method=>"ode",t_end=>10,n_steps=>1})', False),
}


@pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")
@pytest.mark.parametrize(("actions", "refused"), SHAPES.values(), ids=SHAPES.keys())
def test_bng2_pls_own_file_on_nine_shapes(tmp_path, actions, refused):
    """BNG2.pl writes model.net again at a simulate with no suffix, prefix or
    netfile once the model changed: those files are refused, and the ones it
    left as generated (a suffix or prefix on the run, a reset before it, a
    single run from generation) pack."""
    import subprocess

    bngl = tmp_path / "model.bngl"
    bngl.write_text(MODEL + "generate_network({overwrite=>1})\n" + actions + "\n")
    subprocess.run(
        ["perl", str(_BNG.bng2_pl), "model.bngl"], cwd=tmp_path, check=True, capture_output=True
    )
    if refused:
        with pytest.raises(ConversionError, match="is not the network"):
            pack(tmp_path / "model.net", bngl)
    else:
        _, report = pack(tmp_path / "model.net", bngl)
        assert report.ok


def test_a_bngl_of_actions_alone_declares_nothing_about_the_net(tmp_path):
    """A .bngl with no model block beside a hand-written .net (the tests' own
    shape): it generates no network, so a species the .net holds is no filled
    one, and the file packs."""
    bngl = (
        "begin model\nend model\ngenerate_network({overwrite=>1})\n"
        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
    )
    paths = write(tmp_path, bngl, net_ab(a="100", b="39.3"))
    _, report = pack(*paths)
    assert report.ok


# ── Round two: net_to_sbml says it, net_to_omex and the sidecar refuse it ──────


def test_net_to_sbml_says_it_and_notes_it_but_converts(tmp_path):
    """The SBML is the .net as given; the wrong start matters where a protocol
    is written onto it, so net_to_sbml warns and notes, and converts."""
    paths = write(tmp_path, MODEL + ADD_THEN_RUN, net(amount="150"))
    with pytest.warns(ConversionWarning, match="is not the network"):
        report = net_to_sbml(paths[0], out_path=None, validate=None, bngl=paths[1])
    assert report.output_text and any("is not the network" in n for n in report.lossy)


def test_the_cli_converts_without_a_sidecar_and_refuses_with_one(tmp_path, capsys):
    from bngsim.convert._cli import net2sbml_main

    paths = write(tmp_path, MODEL + ADD_THEN_RUN, net(amount="150"))
    assert net2sbml_main([str(paths[0]), "-o", str(tmp_path / "m.xml"), "--no-validate"]) == 0
    capsys.readouterr()
    rc = net2sbml_main(
        [str(paths[0]), "-o", str(tmp_path / "m2.xml"), "--no-validate", "--sidecar"]
    )
    err = capsys.readouterr().err
    assert rc == 1 and "is not the network" in err
