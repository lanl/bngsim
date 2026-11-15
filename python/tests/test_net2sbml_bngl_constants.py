"""BNGL's constants, and symbols spelled like MathML's, through net_to_sbml (issue #1741).

BNG2.pl 2.9.3 writes a derived parameter that uses ``_pi()`` or ``_e()`` into
the .net as it was written (``k2 _pi()*k1  # ConstantExpression``). net_to_sbml
rewrote ``_pi`` to its pi sentinel but left the call's ``()``, so the sentinel
read as an unknown function and the network was refused; ``_e`` it did not
know at all. Both are now MathML's ``<pi/>`` and ``<exponentiale/>``, bare or
called, in the SBML and in the SED-ML an action's value is written to.

The same parse read a model symbol spelled like a constant the parser knows,
``exponentiale``, ``avogadro``, ``true``, ``false``, ``inf``, ``infinity``,
``nan`` or ``notanumber`` (any case), as that constant, silently: a derived
parameter reading a parameter named ``avogadro`` was written with 6.02e23 in
its place. Only ``pi`` was kept a name (GH #8). Every such symbol is now kept.

Oracles: the .net's own values as bngsim loads it, which BNG2.pl's agree with
(``_pi()*k1`` is pi at k1 = 1), and the closed forms pi and e.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import warnings
import zipfile
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert import net_to_sbml

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

NET = """begin parameters
    1 {name} 0.25
    2 k1 1
    3 k2 {k2}
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1 2 k2
end reactions
begin groups
    1 Atot 1
end groups
"""


def _sbml_k2(tmp_path: Path, k2: str, name: str = "p0") -> tuple[float, float, str]:
    """k2 as the .net gives it, as the SBML net_to_sbml writes gives it, and that SBML."""
    net = tmp_path / "m.net"
    net.write_text(NET.format(name=name, k2=k2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = bngsim.Model.from_net(str(net)).get_param("k2")
        text = net_to_sbml(str(net), validate=None).output_text
        xml = tmp_path / "m.xml"
        xml.write_text(text)
        got = bngsim.Model.from_sbml(str(xml)).get_param("k2")
    return want, got, text


# --- the issue: _pi() and _e() as BNG2.pl writes them -----------------------


@pytest.mark.parametrize(
    ("k2", "value", "tag"),
    [
        ("_pi()*k1", math.pi, "<pi/>"),
        ("_e()*k1", math.e, "<exponentiale/>"),
        ("_pi*k1", math.pi, "<pi/>"),
        ("_e*k1", math.e, "<exponentiale/>"),
        ("_pi()*_e()*k1", math.pi * math.e, "<exponentiale/>"),
    ],
    ids=["_pi()", "_e()", "_pi", "_e", "both"],
)
def test_a_derived_parameter_on_a_bngl_constant_converts(tmp_path, k2, value, tag):
    # Each reads k1, so it is written as an initialAssignment (MathML); one that
    # reads no parameter is written as its value.
    want, got, text = _sbml_k2(tmp_path, k2)
    assert want == pytest.approx(value, rel=1e-15)
    assert got == pytest.approx(value, rel=1e-15)
    assert tag in text


def test_the_run_agrees_with_the_net(tmp_path):
    net = tmp_path / "m.net"
    net.write_text(NET.format(name="p0", k2="_e()*k1/_pi()"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = tmp_path / "m.xml"
        xml.write_text(net_to_sbml(str(net), validate=None).output_text)
        a = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
        b = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
    # A decays at k2 = e/pi: A(2) = 100 exp(-2e/pi).
    assert a.species[-1, 0] == pytest.approx(100 * math.exp(-2 * math.e / math.pi), rel=1e-6)
    np.testing.assert_allclose(b.species, a.species, rtol=1e-10, atol=1e-10)


def test_translator_reads_each_spelling():
    import libsbml
    from bngsim.convert._sbml_writer import _exprtk_to_ast

    for text, mathml in [
        ("_pi()", "<pi/>"),
        ("_pi ( )", "<pi/>"),
        ("_e()", "<exponentiale/>"),
        ("_e", "<exponentiale/>"),
    ]:
        assert mathml in libsbml.writeMathMLToString(_exprtk_to_ast(text, where="t"))
    # Names that only contain the letters are names.
    ast = _exprtk_to_ast("k_e + _ex + x_pi", where="t")
    assert libsbml.formulaToL3String(ast) == "k_e + _ex + x_pi"


def test_an_e_spelled_name_is_a_name_even_unlisted():
    # As for pi (GH #8): with no model symbols given, a bareword the parser
    # reads as e is still a name, since genuine e arrives only as _e.
    import libsbml
    from bngsim.convert._sbml_writer import _exprtk_to_ast

    ast = _exprtk_to_ast("exponentiale + ExponentialE", where="t")
    assert libsbml.formulaToL3String(ast) == "exponentiale + ExponentialE"
    assert "<exponentiale/>" not in libsbml.writeMathMLToString(ast)


def test_a_bngl_constant_given_an_argument_is_refused():
    from bngsim.convert._sbml_writer import _exprtk_to_ast

    for text in ("_pi(2)", "_e(k1)"):
        with pytest.raises(bngsim.ConversionError, match="cannot translate"):
            _exprtk_to_ast(text, where="t")


def test_a_volume_on_e_is_constant():
    # An assignment-rule volume is constant where every symbol it reads is; _e
    # read as an unknown symbol made it "varying".
    from bngsim.convert._sbml_writer import _symbol_time_invariant

    for body in ("_e*V0", "_e()*V0", "_pi()*V0"):
        assert _symbol_time_invariant("V", {"V0"}, {"V": body}, set(), set(), {})


# --- symbols spelled like the parser's constants ----------------------------

CONSTANT_SPELLED = [
    "exponentiale",
    "ExponentialE",
    "avogadro",
    "AVOGADRO",
    "true",
    "false",
    "inf",
    "infinity",
    "nan",
    "NaN",
    "notanumber",
    "pi",
    "Pi",
]


@pytest.mark.parametrize("name", CONSTANT_SPELLED)
def test_a_parameter_spelled_like_a_constant_is_read_as_itself(tmp_path, name):
    want, got, _ = _sbml_k2(tmp_path, f"{name}*k1+_pi", name=name)
    assert want == pytest.approx(0.25 + math.pi, rel=1e-15)
    assert got == pytest.approx(want, rel=1e-15)


@pytest.mark.parametrize("name", ["exponentiale", "avogadro", "true", "inf"])
def test_a_function_reading_one_is_read_as_itself(tmp_path, name):
    # The function path: k_f(t) = name * Atot, a rate law A -> B.
    net = tmp_path / "m.net"
    net.write_text(
        f"""begin parameters
    1 {name} 0.25
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin groups
    1 Atot 1
end groups
begin functions
    1 kf() {name}/(1+Atot/100)
end functions
begin reactions
    1 1 2 kf
end reactions
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = tmp_path / "m.xml"
        xml.write_text(net_to_sbml(str(net), validate=None).output_text)
        a = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
        b = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
    assert 0 < a.species[-1, 0] < 100
    np.testing.assert_allclose(b.species, a.species, rtol=1e-9, atol=1e-9)


def test_without_a_model_symbol_true_is_the_literal():
    # ExprTk reads true/false as truth values; only a model symbol is renamed.
    import libsbml
    from bngsim.convert._sbml_writer import _exprtk_to_ast

    ast = _exprtk_to_ast("if(true, 1, 2)", where="t")
    assert "<true/>" in libsbml.writeMathMLToString(ast)
    ast = _exprtk_to_ast("if(true, 1, 2)", where="t", scalar_names=frozenset({"true"}))
    assert "<true/>" not in libsbml.writeMathMLToString(ast)


# --- the issue's own model, through BNG2.pl and net_to_omex -----------------

BNGPATH = os.environ.get("BNGPATH", "")
HAS_BNG2 = bool(BNGPATH) and (Path(BNGPATH) / "BNG2.pl").is_file() and shutil.which("perl")

MODEL = """begin model
begin parameters
  k1 1
  k2 {fn}*k1
  kr 1
end parameters
begin seed species
  A() 100
  B() 0
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() <-> B() k1, kr
end reaction rules
end model
generate_network({{overwrite=>1}})
setParameter("kr","{fn}*k1")
simulate({{method=>"ode",t_end=>1,n_steps=>1}})
"""


@pytest.mark.skipif(not HAS_BNG2, reason="needs BNG2.pl on BNGPATH")
@pytest.mark.parametrize(("fn", "tag"), [("_pi()", "<pi/>"), ("_e()", "<exponentiale/>")])
def test_the_issues_network_and_action_through_net_to_omex(tmp_path, fn, tag):
    from bngsim.convert import net_to_omex

    bngl = tmp_path / "m.bngl"
    bngl.write_text(MODEL.format(fn=fn))
    subprocess.run(
        ["perl", str(Path(BNGPATH) / "BNG2.pl"), "m.bngl"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=300,
    )
    net = tmp_path / "gen.net"
    bngsim.Model.from_bngl(bngl, net_out=net, cache=False)
    assert f"{fn}*k1" in net.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = net_to_omex(net, tmp_path / "m.omex", bngl=bngl, gate="none")
    # On main the network was refused, and the action left out as lossy.
    assert report.sedml_lossy == []
    with zipfile.ZipFile(tmp_path / "m.omex") as z:
        sbml = z.read("model.xml").decode()
        sedml = z.read("simulation.sedml").decode()
    assert tag in sbml
    # The setParameter action's value, as SED-ML's MathML (namespace-prefixed).
    change = sedml[sedml.index("<computeChange") : sedml.index("</computeChange>")]
    assert "parameter[@id='kr']" in change
    assert re.search(rf"<mml:{tag[1:-2]} ?/>", change)


def test_issue_18s_true_and_false_parameters_keep_their_rate(tmp_path):
    # tests/data/exprtk_reserved_words.net (issue #18, from BNG2.pl): parameters
    # named true and false, both 0, in _rateLaw1 = ((k*const)+true)-false = 1.
    # Main wrote them as <true/> and <false/>: rate 2, A(1) = 100/e^2.
    net = Path(__file__).resolve().parents[2] / "tests" / "data" / "exprtk_reserved_words.net"
    xml = tmp_path / "m.xml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml.write_text(net_to_sbml(str(net), validate=None).output_text)
        model = bngsim.Model.from_sbml(str(xml))
        run = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    assert model.get_param("_rateLaw1") == pytest.approx(1.0, rel=1e-15)
    assert run.species[-1, 0] == pytest.approx(100 / math.e, rel=1e-6)


@pytest.mark.parametrize(
    ("groups", "functions"),
    [
        ("    1 Avogadro 1\n", "    1 kf() 0.25/(1+Avogadro/100)\n"),
        ("    1 Atot 1\n", "    1 true() 0.25/(1+Atot/100)\n    2 kf() 2*true\n"),
    ],
    ids=["an observable named Avogadro", "a function named true"],
)
def test_an_observable_or_function_spelled_like_a_constant_is_itself(tmp_path, groups, functions):
    net = tmp_path / "m.net"
    net.write_text(
        "begin parameters\n    1 k0 1\nend parameters\n"
        "begin species\n    1 A() 100\n    2 B() 0\nend species\n"
        f"begin groups\n{groups}end groups\n"
        f"begin functions\n{functions}end functions\n"
        "begin reactions\n    1 1 2 kf\nend reactions\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = tmp_path / "m.xml"
        xml.write_text(net_to_sbml(str(net), validate=None).output_text)
        a = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
        b = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
    assert 0 < a.species[-1, 0] < 100
    np.testing.assert_allclose(b.species, a.species, rtol=1e-9, atol=1e-9)
