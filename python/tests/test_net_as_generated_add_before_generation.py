"""An addConcentration before generate_network makes its species a number (issue #1740).

``net_to_sbml(bngl=...)`` and ``net_to_omex`` check that the ``.net`` is the
network the ``.bngl`` generates (#1683): each seed, with the actions before
``generate_network`` applied, against the ``.net``'s amount. The check read
every seed at the parameters' final values. BNG2.pl's ``addConcentration``
reads the seed when it runs and makes the species the number it adds up to
(BNGModel.pm: ``addConcentration`` evaluates ``$spec->Concentration``), which a
later ``setParameter`` does not move. So seed ``B() B0`` (B0 = 10), then
``addConcentration("B()",60)``, ``setParameter("B0",20)`` gives B = 70 in
BNG2.pl's ``.net``, and the check wanted 80 and refused BNG2.pl's own network.

The actions are now replayed in order, each with the parameters of its moment.
A seed, and a ``setConcentration`` to an expression, still follow their
parameters to generation, as BNG2.pl keeps them. A parameter a setParameter
writes is put back at what the file declares for the steps before it, a number
or an expression (``B0 10*k1``), since BNG2.pl's ``.net`` holds only the number
last set. The replay writes a copy of the model, and only where an add and a
setParameter both come before generation.

Oracle: the ``.net`` BNG2.pl 2.9.3 writes for each protocol. Without BNG2.pl
the ``.net`` text holds the amount it writes, worked out from BNGModel.pm's
rules; with it, BNG2.pl writes the ``.net`` and the test reads its amount.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim.convert import net_to_sbml

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

BNGL = """begin model
begin parameters
  k1 1
  k2 1
{declared}
end parameters
begin seed species
  A() 100
  B() B0
end seed species
begin observables
  Molecules Btot B()
end observables
begin reaction rules
  A() <-> B() k1, k2
end reaction rules
end model
{actions}
generate_network({{overwrite=>1}})
simulate({{method=>"ode",t_end=>1,n_steps=>1}})
"""

NET = """begin parameters
    1 k1  {k1}
    2 k2  1
    3 B0  {b0}
end parameters
begin species
    1 A() 100
    2 B() {b}
end species
begin reactions
    1 1 2 k1 #_R1
    2 2 1 k2 #_reverse__R1
end reactions
begin groups
    1 Btot  2
end groups
"""

# (actions before generate_network, B0 at generation, B in BNG2.pl's .net)
CASES = {
    "the issue: add, then a setParameter of what the seed reads": (
        'addConcentration("B()",60)\nsetParameter("B0",20)',
        20,
        70,
    ),
    "a setParameter, then the add": ('setParameter("B0",20)\naddConcentration("B()",60)', 20, 80),
    "a setConcentration to an expression still follows": (
        'setConcentration("B()","B0*2")\nsetParameter("B0",20)',
        20,
        40,
    ),
    "two adds around a setParameter": (
        'addConcentration("B()",5)\nsetParameter("B0",20)\naddConcentration("B()",5)',
        20,
        20,
    ),
    "an add whose own value reads the parameter, then a setParameter": (
        'addConcentration("B()","B0")\nsetParameter("B0",20)',
        20,
        20,
    ),
    # A setConcentration to a number is a number, and an add after it adds to it.
    "a setConcentration to a number, a setParameter, then an add": (
        'setConcentration("B()",7)\nsetParameter("B0",20)\naddConcentration("B()",3)',
        20,
        10,
    ),
    # An add after a setConcentration to an expression adds to the seed, 10, not
    # to the expression's 20: BNGModel.pm evaluates $spec->Concentration for a
    # species that is not a number, and that is the seed.
    "a setConcentration to an expression, an add, then a setParameter": (
        'setConcentration("B()","B0*2")\naddConcentration("B()",1)\nsetParameter("B0",20)',
        20,
        11,
    ),
    "a setConcentration to an expression, then an add": (
        'setConcentration("B()","B0*2")\naddConcentration("B()",1)',
        10,
        11,
    ),
    "the parameter set twice, an add between": (
        'setParameter("B0",15)\naddConcentration("B()",1)\nsetParameter("B0",20)',
        20,
        16,
    ),
    # A parameter the file declares as an expression: BNG2.pl's .net holds the
    # number setParameter gave it, so the 10 the add read is in the .bngl alone.
    "the seed's parameter declared as an expression of another": (
        'addConcentration("B()",60)\nsetParameter("B0",20)',
        20,
        70,
    ),
    "the seed's parameter declared as arithmetic on numbers": (
        'addConcentration("B()",60)\nsetParameter("B0",20)',
        20,
        70,
    ),
    # B0 reads k1, and both are set after the add.
    "the seed's parameter declared over another, both set later": (
        'addConcentration("B()",60)\nsetParameter("k1",3)\nsetParameter("B0",20)',
        20,
        70,
    ),
    # B0 stays an expression in the .net, and follows the k1 a setParameter moves.
    "the seed's parameter an expression of the one that is set": (
        'addConcentration("B()",60)\nsetParameter("k1",3)',
        "10*k1",
        70,
    ),
    # B0 10*k1 follows k1 until its own setParameter: at the add it is 10*3, so
    # B is 31. Read as the 20 the .net holds, the check wanted 21 and refused
    # BNG2.pl's own network (the review of #1774).
    "the seed's parameter follows what it reads until it is set": (
        'setParameter("k1",3)\naddConcentration("B()",1)\nsetParameter("B0",20)',
        20,
        31,
    ),
    # Its own setParameter makes it a number: the add after reads 20, not 10*k1.
    "the seed's parameter set, then an add": (
        'setParameter("B0",20)\naddConcentration("B()",1)',
        20,
        21,
    ),
    # Moved twice before its own setParameter, an add after each: 31, then 32.
    "the seed's parameter follows two setParameters of what it reads": (
        'setParameter("k1",3)\naddConcentration("B()",1)\nsetParameter("k1",4)\n'
        'addConcentration("B()",1)\nsetParameter("B0",20)',
        20,
        32,
    ),
}

# What the .bngl declares B0 as (and beside it), where that is not "B0 10", and
# k1 in the .net, where a setParameter moved it.
DECLARED = {
    "the seed's parameter declared as an expression of another": "  B0 10*k1",
    "the seed's parameter declared as arithmetic on numbers": "  B0 = 5 * 2",
    "the seed's parameter declared over another, both set later": "  B0 10*k1",
    "the seed's parameter an expression of the one that is set": "  B0 10*k1",
    "the seed's parameter follows what it reads until it is set": "  B0 10*k1",
    "the seed's parameter set, then an add": "  B0 10*k1",
    "the seed's parameter follows two setParameters of what it reads": "  B0 10*k1",
}
K1_IN_NET = {
    "the seed's parameter declared over another, both set later": 3,
    "the seed's parameter an expression of the one that is set": 3,
    "the seed's parameter follows what it reads until it is set": 3,
    "the seed's parameter follows two setParameters of what it reads": 4,
}


def _bngl(case: str, actions: str) -> str:
    return BNGL.format(actions=actions, declared=DECLARED.get(case, "  B0 10"))


def _net(case: str, b0: float | str, b: float) -> str:
    return NET.format(k1=K1_IN_NET.get(case, 1), b0=b0, b=b)


def _not_generated(
    tmp_path: Path, actions: str, b0: float | str, b: float, case: str = ""
) -> list[str]:
    bngl = tmp_path / "m.bngl"
    bngl.write_text(_bngl(case, actions), encoding="utf-8")
    net = tmp_path / "m.net"
    net.write_text(_net(case, b0, b), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    return [note for note in report.lossy if "is not the network" in note]


@pytest.mark.parametrize("case", list(CASES))
def test_bng2pls_own_network_is_taken(tmp_path, case):
    actions, b0, b = CASES[case]
    assert _not_generated(tmp_path, actions, b0, b, case) == []


@pytest.mark.parametrize("case", list(CASES))
def test_a_network_with_another_amount_is_still_refused(tmp_path, case):
    actions, b0, b = CASES[case]
    (note,) = _not_generated(tmp_path, actions, b0, b + 1, case)
    assert f"B() is {b + 1:g} there, {b:g} as declared" in note


def test_a_net_that_a_run_rewrote_is_still_refused(tmp_path):
    # #1683's case, which the check exists for: no actions, and a .net whose B a
    # run filled.
    (note,) = _not_generated(tmp_path, "", 10, 37.5)
    assert "B() is 37.5 there, 10 as declared" in note


def test_net_to_omex_takes_the_issues_network(tmp_path):
    from bngsim.convert import net_to_omex

    actions, b0, b = CASES["the issue: add, then a setParameter of what the seed reads"]
    (tmp_path / "m.bngl").write_text(_bngl("", actions), encoding="utf-8")
    (tmp_path / "m.net").write_text(_net("", b0, b), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_omex(
            tmp_path / "m.net", tmp_path / "m.omex", bngl=tmp_path / "m.bngl", gate="none"
        )
    # Main raised: "m.net is not the network m.bngl generates: B() is 70 there, 80 as declared".
    assert (tmp_path / "m.omex").is_file()
    assert not [n for n in report.lossy if "is not the network" in n]


# --- against BNG2.pl, where it is installed --------------------------------------

# Absolute: BNG2.pl runs with cwd=tmp_path, where a relative BNGPATH names nothing,
# to perl or to BNG2.pl, which reads the variable itself.
BNGPATH = str(Path(os.environ["BNGPATH"]).resolve()) if os.environ.get("BNGPATH") else ""
HAS_BNG2 = bool(BNGPATH) and (Path(BNGPATH) / "BNG2.pl").is_file() and shutil.which("perl")


def _bng2pl(tmp_path: Path) -> None:
    subprocess.run(
        ["perl", str(Path(BNGPATH) / "BNG2.pl"), "m.bngl"],
        cwd=tmp_path,
        env={**os.environ, "BNGPATH": BNGPATH},  # BNG2.pl reads it too
        check=True,
        capture_output=True,
        timeout=300,
    )


@pytest.mark.skipif(not HAS_BNG2, reason="needs BNG2.pl on BNGPATH")
@pytest.mark.parametrize("case", list(CASES))
def test_bng2pl_writes_the_amount_the_case_names_and_it_is_taken(tmp_path, case):
    actions, _, b = CASES[case]
    bngl = tmp_path / "m.bngl"
    bngl.write_text(_bngl(case, actions), encoding="utf-8")
    _bng2pl(tmp_path)
    net = tmp_path / "m.net"
    # B's amount as BNG2.pl's .net gives it: a number, or for a setConcentration
    # to an expression a parameter it writes (NewConc1 = B0*2), kept symbolic.
    assert bngsim.Model.from_net(str(net)).get_concentration("B()") == pytest.approx(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    assert not [n for n in report.lossy if "is not the network" in n]


# --- a compartment volume BNG2.pl folded into the rate constants -------------------
#
# The .net holds 0.05*kf for kf/Ve at Ve = 20, so Model.frozen_params lists Ve and
# set_param refuses to change it. The check must still give its verdict: it reads
# the parameters, it does not run the model.

CBNGL = """begin model
begin parameters
  Ve {ve}
  kf 1
  kb 2
  L0 100
end parameters
begin compartments
  EC 3 Ve
  PM 2 1 EC
end compartments
begin molecule types
  L(r)
  R(l)
end molecule types
begin seed species
  @EC:L(r) L0*Ve
  @PM:R(l) 50
end seed species
begin observables
  Molecules Bound L(r!1).R(l!1)
end observables
begin reaction rules
  L(r) + R(l) <-> L(r!1).R(l!1) kf, kb
end reaction rules
end model
{actions}
generate_network({{overwrite=>1}})
"""

CNET = """begin parameters
    1 Ve  {ve}
    2 kf  1
    3 kb  2
    4 L0  100
    5 _InitialConc1  L0*Ve
end parameters
begin species
    1 @EC::L(r) {lig}
    2 @PM::R(l) 50
    3 @PM::L(r!1)@EC.R(l!1) 0
end species
begin reactions
    1 1,2 3 {factor}*kf #_R1 unit_conversion=1/Ve
    2 3 1,2 kb #_reverse__R1
end reactions
begin groups
    1 Bound  3
end groups
"""

# (Ve declared, actions, Ve in the .net, the volume factor there, L in BNG2.pl's .net)
FROZEN = {
    "a setParameter of the volume alone": (
        "10",
        'setParameter("Ve",20)',
        "20",
        "0.05",
        "_InitialConc1",
    ),
    "an add, then a setParameter of the volume the seed reads": (
        "10",
        'addConcentration("@EC:L(r)",5)\nsetParameter("Ve",20)',
        "20",
        "0.05",
        "1005",
    ),
    "a setParameter of the volume, then an add": (
        "10",
        'setParameter("Ve",20)\naddConcentration("@EC:L(r)",5)',
        "20",
        "0.05",
        "2005",
    ),
}


def _frozen_notes(tmp_path: Path, bngl_text: str, net_text: str | None) -> list[str]:
    (tmp_path / "m.bngl").write_text(bngl_text, encoding="utf-8")
    if net_text is not None:
        (tmp_path / "m.net").write_text(net_text, encoding="utf-8")
    assert "Ve" in bngsim.Model.from_net(str(tmp_path / "m.net")).frozen_params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(tmp_path / "m.net"), bngl=str(tmp_path / "m.bngl"), validate=None)
    return [note for note in report.lossy if "is not the network" in note]


@pytest.mark.parametrize("case", list(FROZEN))
def test_a_frozen_volume_set_before_generation_gets_a_verdict(tmp_path, case):
    # A ParameterError came out of net_to_sbml here: the replay wrote Ve = 10
    # into a copy that refuses any change to Ve.
    ve, actions, ve_net, factor, lig = FROZEN[case]
    bngl_text = CBNGL.format(ve=ve, actions=actions)
    assert _frozen_notes(tmp_path, bngl_text, CNET.format(ve=ve_net, factor=factor, lig=lig)) == []


def test_a_frozen_volume_case_still_refuses_another_amount(tmp_path):
    ve, actions, ve_net, factor, _ = FROZEN[
        "an add, then a setParameter of the volume the seed reads"
    ]
    (note,) = _frozen_notes(
        tmp_path,
        CBNGL.format(ve=ve, actions=actions),
        CNET.format(ve=ve_net, factor=factor, lig="2005"),
    )
    assert "@EC::L(r) is 2005 there, 1005 as declared" in note


@pytest.mark.skipif(not HAS_BNG2, reason="needs BNG2.pl on BNGPATH")
@pytest.mark.parametrize("case", list(FROZEN))
def test_bng2pls_network_with_a_frozen_volume_is_taken(tmp_path, case):
    ve, actions, _, _, lig = FROZEN[case]
    (tmp_path / "m.bngl").write_text(CBNGL.format(ve=ve, actions=actions), encoding="utf-8")
    _bng2pl(tmp_path)
    if lig != "_InitialConc1":
        held = bngsim.Model.from_net(str(tmp_path / "m.net")).get_concentration("@EC::L(r)")
        assert held == pytest.approx(float(lig))
    assert _frozen_notes(tmp_path, CBNGL.format(ve=ve, actions=actions), None) == []


# --- what the replay costs, and what it must not touch ------------------------------


def _held(model) -> tuple[list[float], list[float]]:
    return [model.get_param(n) for n in model.param_names], list(model.get_state())


def test_the_model_is_copied_only_for_an_add_beside_a_setparameter(tmp_path, monkeypatch):
    from bngsim.convert import _not_as_generated
    from bngsim.convert._protocol import parse_bngl_protocol

    copies = []
    clone = bngsim.Model.clone
    monkeypatch.setattr(bngsim.Model, "clone", lambda self: copies.append(1) or clone(self))
    bngl, net = tmp_path / "m.bngl", tmp_path / "m.net"
    for actions, b0, expected in (
        ("", 10, 0),
        ('setParameter("B0",20)', 20, 0),
        ('addConcentration("B()",60)', 10, 0),
        ('setConcentration("B()","B0*2")\nsetParameter("B0",20)', 20, 0),
        ('addConcentration("B()",60)\nsetParameter("B0",20)', 20, 1),
    ):
        bngl.write_text(_bngl("", actions), encoding="utf-8")
        net.write_text(_net("", b0, 70), encoding="utf-8")
        model = bngsim.Model.from_net(str(net))
        held = _held(model)
        copies.clear()
        _not_as_generated(model, bngl, parse_bngl_protocol(bngl))
        assert len(copies) == expected, actions
        # The model converted afterwards is the .net's, not one the replay rewound.
        assert _held(model) == held, actions


def test_a_declared_expression_may_read_a_parameter_declared_below_it(tmp_path):
    # BNG2.pl 2.9.3 refuses this order where a seed reads the parameter, so there
    # is no network of its own to check against: the rewinding alone.
    from bngsim.convert import _at_declared_parameters, _write_parameter

    (tmp_path / "m.net").write_text(_net("", 20, 70), encoding="utf-8")
    model = bngsim.Model.from_net(str(tmp_path / "m.net"))
    work = _at_declared_parameters(model, [("B0", "2*k2"), ("k2", "4")], {"B0", "k2"})
    assert (work.get_param("k2"), work.get_param("B0")) == (4.0, 8.0)
    assert (model.get_param("k2"), model.get_param("B0")) == (1.0, 20.0)
    # A parameter the .bngl does not declare cannot be put back.
    assert _at_declared_parameters(model, [("k2", "4")], {"B0", "k2"}) is None
    # A write the model refuses is a file the check cannot follow, not an error.
    assert _write_parameter(work, "not_a_parameter", 1.0) is False


# --- where the check cannot follow BNG2.pl, no verdict ---------------------------
#
# A .net is refused only on a difference shown. These are readings the check
# would get wrong, and it now gives none, as for a reset or a continued line.


@pytest.mark.parametrize("b", [61, 999])
def test_an_add_that_reads_an_observable_gets_no_verdict(tmp_path, b):
    # BNG2.pl reads Btot from the amounts before generation; the .net's Btot,
    # read here, is after it and holds the add's own result.
    assert _not_generated(tmp_path, 'addConcentration("B()","Btot")', 10, b) == []


def test_an_add_to_a_species_whose_seed_is_not_read_gets_no_verdict(tmp_path):
    # A seed amount that does not evaluate here: the add's base would be taken
    # as 0, where BNG2.pl adds to the seed.
    bngl = tmp_path / "m.bngl"
    bngl.write_text(
        _bngl("", 'addConcentration("B()",5)').replace("  B() B0\n", "  B() B0*nosuch\n"),
        encoding="utf-8",
    )
    assert "B0*nosuch" in bngl.read_text()
    net = tmp_path / "m.net"
    # 15: the seed, taken as 10, plus the 5 BNG2.pl adds to it. The check read
    # the base as 0, wanted 5, and refused it.
    net.write_text(_net("", 10, 15), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    assert not [n for n in report.lossy if "is not the network" in n]


# --- a chain of parameters declared as expressions, each set later ---------------
#
# kk 2*k1 and B0 10*kk both follow k1 until their own setParameter: at the add,
# kk = 6 and B0 = 60, so B is 61. BNG2.pl 2.9.3 wrote this network.

CHAIN_BNGL = BNGL.format(
    declared="  kk 2*k1\n  B0 10*kk",
    actions='setParameter("k1",3)\naddConcentration("B()",1)\n'
    'setParameter("kk",50)\nsetParameter("B0",20)',
)
CHAIN_NET = """begin parameters
    1 k1    3  # Constant
    2 k2    1  # Constant
    3 kk    50  # Constant
    4 B0    20  # Constant
end parameters
begin species
    1 A() 100
    2 B() 61
end species
begin reactions
    1 1 2 k1 #_R1
    2 2 1 k2 #_reverse__R1
end reactions
begin groups
    1 Btot                 2
end groups
"""


@pytest.mark.parametrize("b", [61, 62])
@pytest.mark.parametrize("order", ["as declared", "reversed"])
def test_a_chain_of_expressions_follows_in_the_order_they_read(tmp_path, order, b):
    # Reversed, B0 is declared above the kk it reads (BNG2.pl refuses that file,
    # so it has no network of its own); the reading must not depend on the order.
    bngl_text = CHAIN_BNGL
    if order == "reversed":
        bngl_text = bngl_text.replace("  kk 2*k1\n  B0 10*kk", "  B0 10*kk\n  kk 2*k1")
    assert ("B0 10*kk\n  kk" in bngl_text) == (order == "reversed")
    bngl = tmp_path / "m.bngl"
    bngl.write_text(bngl_text, encoding="utf-8")
    net = tmp_path / "m.net"
    net.write_text(CHAIN_NET.replace("2 B() 61", f"2 B() {b}"), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    notes = [n for n in report.lossy if "is not the network" in n]
    if b == 61:
        assert notes == []
    else:
        (note,) = notes
        assert "B() is 62 there, 61 as declared" in note


def test_an_add_after_a_set_to_an_expression_on_an_unread_seed_gets_no_verdict(tmp_path):
    # The add adds to the seed, BNG2.pl's $spec->Concentration, not to the
    # expression; with the seed unread, the check took the expression.
    bngl = tmp_path / "m.bngl"
    bngl.write_text(
        _bngl("", 'setConcentration("B()","B0*2")\naddConcentration("B()",1)').replace(
            "  B() B0\n", "  B() B0*nosuch\n"
        ),
        encoding="utf-8",
    )
    net = tmp_path / "m.net"
    net.write_text(_net("", 10, 11), encoding="utf-8")  # the seed, taken as 10, plus 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    assert not [n for n in report.lossy if "is not the network" in n]


# The chain can run through a parameter no setParameter writes: B0 R*10, R 2*Q,
# Q 2*k1, with Q and B0 set later. R follows Q in the copy by itself, so B0 reads
# Q only through it, and the order of the pending ones (Q, B0) has to be taken
# through R. Declared B0 first, B0 was written before Q, from a stale R, and the
# check wanted 41 for 121 (a second review of #1774). BNG2.pl 2.9.3 wrote this
# network from the file in dependency order, and refuses the reversed one.

CHAIN2_BNGL = BNGL.format(
    declared="  Q 2*k1\n  R 2*Q\n  B0 R*10",
    actions='setParameter("k1",3)\naddConcentration("B()",1)\n'
    'setParameter("Q",1)\nsetParameter("B0",20)',
)
CHAIN2_NET = """begin parameters
    1 k1    3  # Constant
    2 k2    1  # Constant
    3 Q     1  # Constant
    4 R     2*Q  # ConstantExpression
    5 B0    20  # Constant
end parameters
begin species
    1 A() 100
    2 B() 121
end species
begin reactions
    1 1 2 k1 #_R1
    2 2 1 k2 #_reverse__R1
end reactions
begin groups
    1 Btot                 2
end groups
"""


@pytest.mark.parametrize("b", [121, 122])
@pytest.mark.parametrize("order", ["as declared", "reversed"])
def test_a_chain_through_a_parameter_no_action_writes_is_followed(tmp_path, order, b):
    bngl_text = CHAIN2_BNGL
    if order == "reversed":
        bngl_text = bngl_text.replace(
            "  Q 2*k1\n  R 2*Q\n  B0 R*10", "  B0 R*10\n  R 2*Q\n  Q 2*k1"
        )
    assert ("B0 R*10\n  R 2*Q\n  Q 2*k1" in bngl_text) == (order == "reversed")
    bngl = tmp_path / "m.bngl"
    bngl.write_text(bngl_text, encoding="utf-8")
    net = tmp_path / "m.net"
    net.write_text(CHAIN2_NET.replace("2 B() 121", f"2 B() {b}"), encoding="utf-8")
    assert f"2 B() {b}" in net.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    notes = [n for n in report.lossy if "is not the network" in n]
    if b == 121:
        assert notes == []
    else:
        (note,) = notes
        assert "B() is 122 there, 121 as declared" in note
