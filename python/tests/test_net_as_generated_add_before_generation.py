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
parameters to generation, as BNG2.pl keeps them.

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
  B0 10
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
    1 k1  1
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
}


def _not_generated(tmp_path: Path, actions: str, b0: float, b: float) -> list[str]:
    bngl = tmp_path / "m.bngl"
    bngl.write_text(BNGL.format(actions=actions))
    net = tmp_path / "m.net"
    net.write_text(NET.format(b0=b0, b=b))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    return [note for note in report.lossy if "is not the network" in note]


@pytest.mark.parametrize("case", list(CASES))
def test_bng2pls_own_network_is_taken(tmp_path, case):
    actions, b0, b = CASES[case]
    assert _not_generated(tmp_path, actions, b0, b) == []


@pytest.mark.parametrize("case", list(CASES))
def test_a_network_with_another_amount_is_still_refused(tmp_path, case):
    actions, b0, b = CASES[case]
    (note,) = _not_generated(tmp_path, actions, b0, b + 1)
    assert f"B() is {b + 1:g} there, {b:g} as declared" in note


def test_a_net_that_a_run_rewrote_is_still_refused(tmp_path):
    # #1683's case, which the check exists for: no actions, and a .net whose B a
    # run filled.
    (note,) = _not_generated(tmp_path, "", 10, 37.5)
    assert "B() is 37.5 there, 10 as declared" in note


def test_net_to_omex_takes_the_issues_network(tmp_path):
    from bngsim.convert import net_to_omex

    actions, b0, b = CASES["the issue: add, then a setParameter of what the seed reads"]
    (tmp_path / "m.bngl").write_text(BNGL.format(actions=actions))
    (tmp_path / "m.net").write_text(NET.format(b0=b0, b=b))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_omex(
            tmp_path / "m.net", tmp_path / "m.omex", bngl=tmp_path / "m.bngl", gate="none"
        )
    # Main raised: "m.net is not the network m.bngl generates: B() is 70 there, 80 as declared".
    assert (tmp_path / "m.omex").is_file()
    assert not [n for n in report.lossy if "is not the network" in n]


# --- against BNG2.pl, where it is installed --------------------------------------

BNGPATH = os.environ.get("BNGPATH", "")
HAS_BNG2 = bool(BNGPATH) and (Path(BNGPATH) / "BNG2.pl").is_file() and shutil.which("perl")


@pytest.mark.skipif(not HAS_BNG2, reason="needs BNG2.pl on BNGPATH")
@pytest.mark.parametrize("case", list(CASES))
def test_bng2pl_writes_the_amount_the_case_names_and_it_is_taken(tmp_path, case):
    actions, _, b = CASES[case]
    bngl = tmp_path / "m.bngl"
    bngl.write_text(BNGL.format(actions=actions))
    subprocess.run(
        ["perl", str(Path(BNGPATH) / "BNG2.pl"), "m.bngl"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=300,
    )
    net = tmp_path / "m.net"
    # B's amount as BNG2.pl's .net gives it: a number, or for a setConcentration
    # to an expression a parameter it writes (NewConc1 = B0*2), kept symbolic.
    assert bngsim.Model.from_net(str(net)).get_concentration("B()") == pytest.approx(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_sbml(str(net), bngl=str(bngl), validate=None)
    assert not [n for n in report.lossy if "is not the network" in n]
