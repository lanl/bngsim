"""A written .net lists each function after the functions it reads (issue #1791).

BioNetGen's run_network reads a .net functions block top to bottom and refuses a
function that reads one defined below it: "Could not find variable". sbml_to_net
wrote the block in the SBML's assignment-rule order, and a rule may read one listed
after it (`y := 3*x` before `x := k/2`; BIOMD0000000268 has five). bngsim sorts
functions when it loads a .net, so its own reload, and the converter's validate
gate, read the file fine.

The oracle is run_network itself, and, where it is not installed, the block's order.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert import sbml_to_net

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

_BNGPATH = os.environ.get("BNGPATH")
_RUN_NETWORK = Path(_BNGPATH).resolve() / "bin" / "run_network" if _BNGPATH else None
HAS_RUN_NETWORK = _RUN_NETWORK is not None and _RUN_NETWORK.is_file()


def _sbml(rules: list[tuple[str, str]]) -> str:
    """A model whose A -> 0 at y*A, with the assignment rules *rules* in this order."""
    mathml = {
        "y": "<apply><times/><ci>x</ci><cn>3</cn></apply>",
        "x": "<apply><times/><ci>k</ci><cn>0.5</cn></apply>",
        "z": "<apply><plus/><ci>y</ci><ci>x</ci></apply>",
    }
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"{mathml[v]}</math></assignmentRule>"
        for v, _ in rules
    )
    params = "".join(f'<parameter id="{v}" constant="false"/>' for v, _ in rules)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
  </listOfCompartments>
  <listOfSpecies><species id="A" compartment="c" initialAmount="10" hasOnlySubstanceUnits="true"
    boundaryCondition="false" constant="false"/></listOfSpecies>
  <listOfParameters><parameter id="k" value="2" constant="true"/>{params}</listOfParameters>
  <listOfRules>{rule_xml}</listOfRules>
  <listOfReactions>
   <reaction id="R1" reversible="false">
    <listOfReactants>
     <speciesReference species="A" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><times/><ci>y</ci><ci>A</ci></apply>
    </math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""


def _write(tmp_path, rules) -> Path:
    src = tmp_path / "m.xml"
    src.write_text(_sbml(rules), encoding="utf-8")
    net = tmp_path / "m.net"
    sbml_to_net(str(src), str(net))
    return net


def _function_order(net: Path) -> list[str]:
    block = net.read_text(encoding="utf-8").split("begin functions")[1].split("end functions")[0]
    return [ln.split()[1].removesuffix("()") for ln in block.splitlines() if ln.strip()]


def _forward_references(net: Path) -> list[tuple[str, str]]:
    block = net.read_text(encoding="utf-8").split("begin functions")[1].split("end functions")[0]
    lines = [ln for ln in block.splitlines() if ln.strip()]
    order = [ln.split()[1].removesuffix("()") for ln in lines]
    pos = {n: i for i, n in enumerate(order)}
    return [
        (order[i], r)
        for i, ln in enumerate(lines)
        for r in re.findall(r"[A-Za-z_]\w*", " ".join(ln.split()[2:]))
        if pos.get(r, -1) > i
    ]


@pytest.mark.parametrize(
    "rules",
    [
        [("y", ""), ("x", "")],  # the issue's: y := 3*x before x := k/2
        [("z", ""), ("y", ""), ("x", "")],  # a chain read backwards, two deep
    ],
)
def test_each_function_follows_the_ones_it_reads(tmp_path, rules):
    net = _write(tmp_path, rules)
    assert _forward_references(net) == []


def test_a_block_already_in_order_is_written_as_it_was(tmp_path):
    # Seeded in declaration order: rules already in order keep it, line for line.
    net = _write(tmp_path, [("x", ""), ("y", ""), ("z", "")])
    assert _function_order(net)[:3] == ["x", "y", "z"]


def test_bngsim_reads_the_written_file_to_the_closed_form(tmp_path):
    # y = 3*(k/2) = 3, so A = 10*exp(-3t).
    net = _write(tmp_path, [("y", ""), ("x", "")])
    r = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
        t_span=(0, 1), n_points=3
    )
    np.testing.assert_allclose(r.species[:, 0], 10 * np.exp(-3 * np.array([0, 0.5, 1])), rtol=1e-5)


@pytest.mark.skipif(
    not HAS_RUN_NETWORK, reason="no BioNetGen install (BNG2.pl and its run_network) found"
)
@pytest.mark.parametrize("rules", [[("y", ""), ("x", "")], [("z", ""), ("y", ""), ("x", "")]])
def test_run_network_reads_the_written_file(tmp_path, rules):
    net = _write(tmp_path, rules)
    args = ["-o", str(tmp_path / "out"), "-p", "cvode", "-a", "1e-10", "-r", "1e-10"]
    args += ["--cdat", "1", "--fdat", "0", "-g", str(net), str(net), "0.5", "2"]
    done = subprocess.run(
        [str(_RUN_NETWORK), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    cdat = np.loadtxt(tmp_path / "out.cdat")
    np.testing.assert_allclose(cdat[:, 1], 10 * np.exp(-3 * cdat[:, 0]), rtol=1e-6)


def test_functions_that_read_each_other_are_written_as_declared():
    # A cycle has no order; the writer keeps the declaration order it had, as
    # before, rather than refusing a file it wrote until now.
    from bngsim.convert._net_writer import _functions_in_dependency_order

    functions = [{"name": "a", "expression": "b+1"}, {"name": "b", "expression": "a+1"}]
    assert _functions_in_dependency_order(functions) == functions


def test_a_function_that_reads_nothing_does_not_move_ahead_of_one_in_order():
    # A plain topological sort put every function that reads nothing first, so
    # reaction5 moved ahead of reaction4 although reaction4's reads were already
    # above it: 521 of 1291 corpus files rewritten for nothing. A stable sort
    # leaves a block already in order as it is.
    from bngsim.convert._net_writer import _functions_in_dependency_order

    functions = [
        {"name": "V1", "expression": "k*2"},
        {"name": "reaction4", "expression": "V1*M"},
        {"name": "reaction5", "expression": "k*M"},
    ]
    assert _functions_in_dependency_order(functions) == functions


def test_a_forward_read_is_put_after_what_it_reads():
    from bngsim.convert._net_writer import _functions_in_dependency_order

    functions = [
        {"name": "a", "expression": "1"},
        {"name": "y", "expression": "x*3"},  # reads x, declared below it
        {"name": "b", "expression": "2"},
        {"name": "x", "expression": "k/2"},
    ]
    names = [f["name"] for f in _functions_in_dependency_order(functions)]
    assert names == ["a", "b", "x", "y"]


def test_a_function_that_reads_its_own_name_does_not_stop_the_sort():
    # A read of itself is no order constraint on the others: counted, it made a
    # cycle of one, and the whole block, forward read and all, stayed as declared.
    from bngsim.convert._net_writer import _functions_in_dependency_order

    functions = [{"name": "y", "expression": "x*3"}, {"name": "x", "expression": "x0 + 0*x"}]
    assert [f["name"] for f in _functions_in_dependency_order(functions)] == ["x", "y"]
