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
import re
import warnings
import zipfile
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngpath import resolve_bng
from bngsim.convert import net_to_sbml, write_bngl

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
    net.write_text(NET.format(name=name, k2=k2), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = bngsim.Model.from_net(str(net)).get_param("k2")
        text = net_to_sbml(str(net), validate=None).output_text
        xml = tmp_path / "m.xml"
        xml.write_text(text, encoding="utf-8")
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
    net.write_text(NET.format(name="p0", k2="_e()*k1/_pi()"), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = tmp_path / "m.xml"
        xml.write_text(net_to_sbml(str(net), validate=None).output_text, encoding="utf-8")
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

    # The message names the constant as the model spells it, not the
    # placeholder the translator puts in its place.
    for text, name in (("_pi(2)", "_pi"), ("_e(k1)", "_e")):
        with pytest.raises(bngsim.ConversionError, match="cannot translate") as err:
            _exprtk_to_ast(text, where="t")
        assert f"function call(s) '{name}' " in str(err.value)
        assert "__bngsim" not in str(err.value)


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
    "time",
    "Time",
    "TIME",
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
""",
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = tmp_path / "m.xml"
        xml.write_text(net_to_sbml(str(net), validate=None).output_text, encoding="utf-8")
        a = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
        b = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
    assert 0 < a.species[-1, 0] < 100
    np.testing.assert_allclose(b.species, a.species, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("name", "body", "rate"),
    [
        # The engine reads a bare ``time`` the model declares as that symbol
        # and the call as the clock (issue #776); ``Time`` is no word of its.
        ("time", "time", "p"),
        ("time", "time()", "t"),
        ("time", "time+time()", "p+t"),
        ("time", "time ( ) * time", "p*t"),
        ("Time", "Time+time()", "p+t"),
        ("TIME", "TIME*2", "2p"),
        ("p0", "p0+time()", "p+t"),
    ],
)
def test_a_symbol_named_time_is_itself_and_the_call_is_the_clock(tmp_path, name, body, rate):
    # A -> B at kf(t), with the parameter p = 0.25: A(2) = 100 exp(-int_0^2 kf).
    integral = {"p": 0.5, "t": 2.0, "p+t": 2.5, "p*t": 0.5, "2p": 1.0}[rate]
    net = tmp_path / "m.net"
    net.write_text(
        f"begin parameters\n    1 {name} 0.25\nend parameters\n"
        "begin species\n    1 A() 100\n    2 B() 0\nend species\n"
        "begin groups\n    1 Atot 1\nend groups\n"
        f"begin functions\n    1 kf() {body}\nend functions\n"
        "begin reactions\n    1 1 2 kf\nend reactions\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = net_to_sbml(str(net), validate=None).output_text
        xml = tmp_path / "m.xml"
        xml.write_text(text, encoding="utf-8")
        a = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
        b = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
    assert a.species[-1, 0] == pytest.approx(100 * math.exp(-integral), rel=1e-5)
    np.testing.assert_allclose(b.species, a.species, rtol=1e-9, atol=1e-9)
    # The clock is written exactly where the .net calls it.
    assert ("symbols/time" in text) == ("(" in body)


def test_an_action_value_reading_a_parameter_named_time_reads_the_parameter():
    # The SED-ML writer's path: its names are the SBML's parameters.
    import libsbml
    from bngsim.convert._sbml_writer import _exprtk_to_ast

    for name in ("time", "Time", "TIME"):
        ast = _exprtk_to_ast(f"{name}*2", where="t", scalar_names=frozenset({name}))
        assert ast.getChild(0).getType() == libsbml.AST_NAME
        assert libsbml.formulaToL3String(ast) == f"{name} * 2"
        assert "csymbol" not in libsbml.writeMathMLToString(ast)
    # With no such symbol in the model, and in the call, it is the clock.
    for text, names in (("time()*2", frozenset({"time"})), ("time()*2", frozenset())):
        ast = _exprtk_to_ast(text, where="t", scalar_names=names)
        assert ast.getChild(0).getType() == libsbml.AST_NAME_TIME


def test_without_a_model_symbol_true_is_the_literal():
    # ExprTk reads true/false as truth values; only a model symbol is renamed.
    import libsbml
    from bngsim.convert._sbml_writer import _exprtk_to_ast

    ast = _exprtk_to_ast("if(true, 1, 2)", where="t")
    assert "<true/>" in libsbml.writeMathMLToString(ast)
    ast = _exprtk_to_ast("if(true, 1, 2)", where="t", scalar_names=frozenset({"true"}))
    assert "<true/>" not in libsbml.writeMathMLToString(ast)


# --- the issue's own model, through BNG2.pl and net_to_omex -----------------

_BNG = resolve_bng()
needs_bng2 = pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")

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


@needs_bng2
@pytest.mark.parametrize(("fn", "tag"), [("_pi()", "<pi/>"), ("_e()", "<exponentiale/>")])
def test_the_issues_network_and_action_through_net_to_omex(tmp_path, fn, tag):
    from bngsim.convert import net_to_omex

    bngl = tmp_path / "m.bngl"
    bngl.write_text(MODEL.format(fn=fn), encoding="utf-8")
    # BNG2.pl writes the network (the loader runs it, wherever BNGPATH points).
    net = tmp_path / "gen.net"
    bngsim.Model.from_bngl(bngl, net_out=net, cache=False)
    assert f"{fn}*k1" in net.read_text(encoding="utf-8")
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
        xml.write_text(net_to_sbml(str(net), validate=None).output_text, encoding="utf-8")
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
        "begin reactions\n    1 1 2 kf\nend reactions\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = tmp_path / "m.xml"
        xml.write_text(net_to_sbml(str(net), validate=None).output_text, encoding="utf-8")
        a = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
        b = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode").run(
            t_span=(0, 2), n_points=5
        )
    assert 0 < a.species[-1, 0] < 100
    np.testing.assert_allclose(b.species, a.species, rtol=1e-9, atol=1e-9)


# --- the sibling writer: the same expressions as BNGL -----------------------


def test_the_bngl_writer_folds_each_spelling_of_a_bngl_constant():
    # BNG2.pl reads only the call form and aborts on the bare name; the engine
    # takes both. Main left the call's () behind pi's number and kept _e.
    from bngsim.convert._bngl_writer import _normalize_bngl_expr

    pi, e = repr(math.pi), repr(math.e)
    for text, want in [
        ("_pi()*k1", f"{pi}*k1"),
        ("_pi ( )*k1", f"{pi}*k1"),
        ("_pi*k1", f"{pi}*k1"),
        ("_e()*k1", f"{e}*k1"),
        ("_e*k1", f"{e}*k1"),
        ("2^_e/_pi()", f"2^{e}/{pi}"),
        # Names that only contain the letters are names.
        ("k_e + _ex + x_pi", "k_e + _ex + x_pi"),
    ]:
        assert _normalize_bngl_expr(text) == want


@pytest.mark.parametrize("k2", ["_pi()*k1", "_pi*k1", "_e()*k1", "_e*k1"])
def test_write_bngl_keeps_a_derived_parameter_on_a_bngl_constant(tmp_path, k2):
    net = tmp_path / "m.net"
    net.write_text(NET.format(name="p0", k2=k2), encoding="utf-8")
    value = math.pi if "pi" in k2 else math.e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = write_bngl(bngsim.Model.from_net(str(net)))
    line = next(ln.split() for ln in text.splitlines() if ln.split()[1:2] == ["k2"])
    assert line[2] == f"{value!r}*k1"


@needs_bng2
@pytest.mark.parametrize("k2", ["_pi()*k1/10", "_e*k1/10"])
def test_bng2pl_reads_the_bngl_written_for_a_bngl_constant(tmp_path, k2):
    # On main BNG2.pl refused the file: "invalid trailing characters at ()*k1/10"
    # and "Parameter '_e' is referenced but not defined".
    net = tmp_path / "m.net"
    net.write_text(NET.format(name="p0", k2=k2), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_net(str(net))
        bngl = tmp_path / "out.bngl"
        bngl.write_text(write_bngl(model), encoding="utf-8")
        back = bngsim.Model.from_bngl(bngl, cache=False)
    assert back.get_param("k2") == pytest.approx(model.get_param("k2"), rel=1e-15)
