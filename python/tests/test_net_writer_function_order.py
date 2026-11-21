"""A written .net lists each function after the functions it reads (issue #1791).

BioNetGen's run_network reads a .net functions block top to bottom and refuses a
function that reads one defined below it: "Could not find variable". sbml_to_net
wrote the block in the SBML's assignment-rule order, and a rule may read one listed
after it (`y := 3*x` before `x := k/2`; BIOMD0000000268 has five). bngsim sorts
functions when it loads a .net, so its own reload, and the converter's validate
gate, read the file fine.

The oracle is run_network itself where it is installed. Elsewhere the order is
checked against what each function reads, tokenized here independently of the
writer.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngpath import resolve_bng
from bngsim.convert import sbml_to_net

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

_EXE = "run_network.exe" if sys.platform == "win32" else "run_network"
_BNG = resolve_bng()  # as the sibling writer tests find it, BNGPATH or not
_RUN_NETWORK = _BNG.root / "bin" / _EXE if _BNG.ok and _BNG.root is not None else None
HAS_RUN_NETWORK = _RUN_NETWORK is not None and _RUN_NETWORK.is_file()

_MATHML = {
    "y": "<apply><times/><ci>x</ci><cn>3</cn></apply>",
    "x": "<apply><times/><ci>k</ci><cn>0.5</cn></apply>",
    "z": "<apply><plus/><ci>y</ci><ci>x</ci></apply>",
}


def _sbml(rules: list[str]) -> str:
    """A model whose A -> 0 at y*A, with the assignment rules *rules* in this order."""
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"{_MATHML[v]}</math></assignmentRule>"
        for v in rules
    )
    params = "".join(f'<parameter id="{v}" constant="false"/>' for v in rules)
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


def _write(tmp_path, rules: list[str]) -> Path:
    src = tmp_path / "m.xml"
    src.write_text(_sbml(rules), encoding="utf-8")
    net = tmp_path / "m.net"
    sbml_to_net(str(src), str(net))
    return net


def _block(net: Path) -> list[tuple[str, str]]:
    text = net.read_text(encoding="utf-8").split("begin functions")[1].split("end functions")[0]
    rows = [ln.split(None, 2)[1:] for ln in text.splitlines() if ln.strip()]
    return [(name.removesuffix("()"), expr) for name, expr in rows]


def _forward_reads(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    # Independent of the writer's tokenizer: a read is a function name standing
    # whole, not inside a longer name and not a number's exponent.
    import re

    pos = {n: i for i, (n, _) in enumerate(rows)}
    return [
        (n, other)
        for i, (n, expr) in enumerate(rows)
        for other, j in pos.items()
        if j > i
        and re.search(
            rf"(?<![\w.]){re.escape(other)}(?![\w(])|(?<![\w.]){re.escape(other)}\(\)", expr
        )
    ]


@pytest.mark.parametrize("rules", [["y", "x"], ["z", "y", "x"]])
def test_each_function_follows_the_ones_it_reads(tmp_path, rules):
    block = _block(_write(tmp_path, rules))
    names = [n for n, _ in block]
    assert set(rules) <= set(names)  # every rule is written
    assert names.index("x") < names.index("y")  # and x, which y reads, above it
    assert _forward_reads(block) == []


def test_a_block_already_in_order_is_written_as_it_was(tmp_path):
    # Stable: rules already in order keep it, line for line.
    names = [n for n, _ in _block(_write(tmp_path, ["x", "y", "z"]))]
    assert names[:3] == ["x", "y", "z"]


def test_bngsim_reads_the_written_file_to_the_closed_form(tmp_path):
    # y = 3*(k/2) = 3, so A = 10*exp(-3t).
    net = _write(tmp_path, ["y", "x"])
    r = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
        t_span=(0, 1), n_points=3
    )
    np.testing.assert_allclose(r.species[:, 0], 10 * np.exp(-3 * np.array([0, 0.5, 1])), rtol=1e-5)


@pytest.mark.skipif(
    not HAS_RUN_NETWORK, reason="no BioNetGen install (BNG2.pl and its run_network) found"
)
@pytest.mark.parametrize("rules", [["y", "x"], ["z", "y", "x"]])
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


# --- the sort itself ---------------------------------------------------------------


def _order(pairs):
    from bngsim.convert._net_writer import _functions_in_dependency_order

    return [n for n, _ in _functions_in_dependency_order(pairs)]


def test_a_function_that_reads_nothing_does_not_move_ahead_of_one_in_order():
    # A plain topological sort put every function that reads nothing first, so
    # reaction5 moved ahead of reaction4 although reaction4's reads were already
    # above it: 521 of 1291 corpus files rewritten for nothing.
    pairs = [("V1", "k*2"), ("reaction4", "V1*M"), ("reaction5", "k*M")]
    assert _order(pairs) == ["V1", "reaction4", "reaction5"]


def test_a_forward_read_is_put_after_what_it_reads():
    pairs = [("a", "1"), ("y", "x*3"), ("b", "2"), ("x", "k/2")]
    assert _order(pairs) == ["a", "b", "x", "y"]


def test_a_function_that_reads_its_own_name_is_warned_and_does_not_stop_the_sort():
    # run_network refuses a self-read as it refuses a cycle; the rest is sorted.
    with pytest.warns(bngsim.ConversionWarning, match="read each other, or themselves"):
        assert _order([("y", "x*3"), ("x", "x0 + 0*x")]) == ["x", "y"]


def test_a_number_exponent_is_no_read():
    # 2e5 holds no name: a function named e5 is not read by it, and no false
    # cycle (e5 reads y, y "reads" e5) turns the sort off.
    pairs = [("y", "2e5*x"), ("e5", "y + 1"), ("x", "k")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a false read would make a false cycle, and warn
        assert _order(pairs) == ["x", "y", "e5"]


def test_a_name_is_read_whole_not_by_its_parts():
    # x is no read of x_total, nor k of k1.
    pairs = [("y", "x_total*k1"), ("x", "1"), ("x_total", "2"), ("k1", "3")]
    assert _order(pairs) == ["x", "x_total", "k1", "y"]


def test_a_cycle_leaves_the_rest_sorted_and_warns():
    pairs = [("y", "x*3"), ("x", "k/2"), ("a", "b+1"), ("b", "a+1")]
    with pytest.warns(bngsim.ConversionWarning, match="read each other, or themselves"):
        assert _order(pairs) == ["x", "y", "a", "b"]  # y after x; the cycle kept together


def test_a_function_that_reads_a_cycle_goes_after_it():
    # The cycle a <-> b is placed as one function: z, which reads a, goes after it,
    # and w, which reads z, after z. Appending everything unplaced as declared put
    # w above z, a forward read no cycle forced.
    pairs = [("w", "z+a"), ("z", "a*2"), ("a", "b"), ("b", "a")]
    with pytest.warns(bngsim.ConversionWarning):
        assert _order(pairs) == ["a", "b", "z", "w"]


def test_a_cycle_in_a_large_block_is_found_without_recursion():
    # 20000 functions in one chain read backwards, closed into a cycle: Tarjan's
    # algorithm, done recursively, would pass Python's recursion limit.
    n = 20000
    pairs = [(f"f{i}", f"f{i + 1}") for i in range(n - 1)] + [(f"f{n - 1}", "f0")]
    with pytest.warns(bngsim.ConversionWarning):
        assert len(_order(pairs)) == n


def test_no_warning_without_a_cycle():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _order([("y", "x*3"), ("x", "k/2")])


def test_a_cycle_takes_its_place_by_its_first_declared_member():
    # Stable for a cycle too: a <-> c, declared first and third, goes before b,
    # declared second and read by neither, as a alone would.
    pairs = [("a", "c+1"), ("b", "k"), ("c", "a+1")]
    with pytest.warns(bngsim.ConversionWarning):
        assert _order(pairs) == ["a", "c", "b"]


def test_a_block_in_order_with_a_self_read_is_kept_and_warned():
    # Nothing to move, so the block is returned as it is, and the self-read is
    # still said: run_network refuses it.
    pairs = [("x", "x0 + 0*x"), ("y", "x*3")]
    with pytest.warns(bngsim.ConversionWarning, match="read each other, or themselves"):
        assert _order(pairs) == ["x", "y"]


def test_names_are_read_as_the_engines_sort_reads_them():
    # for_each_identifier (src/model_builder.cpp): a dot ends a name, so R1.x
    # reads x, and a number is stepped over whole, exponent and all.
    assert _order([("y", "R1.x*2"), ("x", "1")]) == ["x", "y"]
    assert _order([("y", "1.5e3*2.e5"), ("e3", "1"), ("e5", "2")]) == ["y", "e3", "e5"]


def test_a_name_written_twice_is_read_as_its_first_copy():
    # The `_vd_` copies: a reader of the first is in order, and stays where it is.
    pairs = [("_vd", "2"), ("r1", "_vd*k"), ("_vd", "2")]
    assert _order(pairs) == ["_vd", "r1", "_vd"]
