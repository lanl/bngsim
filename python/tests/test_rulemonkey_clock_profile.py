"""GH #1022: how bngsim reads a BNG XML's use of the clock before RuleMonkey runs it.

RuleMonkey integrates a rate that reads the clock in steps of up to a
thousandth of a call, and a step can lie across a short window in the rate and
see none of it. So bngsim runs a clock-reading model only where it can say when
the rates change with the clock, and cuts every engine call at those instants.
``clock_profile`` is what says so. A function may read the clock

* smoothly (sums, products, whole powers, ``exp`` and the like): nothing to cut at;
* as a switch (the clock compared with a number or a parameter): cut at the threshold;
* through a table indexed by the clock: cut at the table's points;

and anything else is refused. These tests are the reading; the values a run
gives are in test_rulemonkey_model_clock.py.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import clock_profile, rulemonkey_clock_profile

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
# A(s~0) -> A(s~1) at f() = if(time()>=6, k, 0), one parameter k, one observable A0.
SWITCH = DATA / "time_switch.xml"
PLAIN = DATA / "first_order_switch.xml"
TFUN = DATA / "tfun_new_format"
FUNCTION = """      <Function id="f">
        <ListOfReferences>
          <Reference name="k" type="Constant"/>
        </ListOfReferences>
        <Expression> if(time()>=6,k,0) </Expression>
      </Function>
"""


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


needs_rulemonkey = pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in")


def _with(tmp_path, expression, *, parameters=(), more_functions="", name="m"):
    """time_switch.xml with f's expression replaced, extra parameters declared,
    and extra ``<Function>`` elements after f."""
    text = SWITCH.read_text(encoding="utf-8")
    assert text.count(FUNCTION) == 1
    escaped = expression.replace("&", "&amp;").replace("<", "&lt;")
    text = text.replace(
        FUNCTION, FUNCTION.replace(" if(time()>=6,k,0) ", f" {escaped} ") + more_functions
    )
    declared = "".join(
        f'      <Parameter id="{p}" type="Constant" value="{v}" expr="{v}"/>\n'
        for p, v in parameters
    )
    marker = "    </ListOfParameters>\n"
    assert text.count(marker) == 1
    xml = tmp_path / f"{name}.xml"
    xml.write_text(text.replace(marker, declared + marker), encoding="utf-8")
    return xml


def _function(name, expression):
    return (
        f'      <Function id="{name}">\n'
        f"        <Expression> {expression} </Expression>\n      </Function>\n"
    )


# ── The fixtures ──────────────────────────────────────────────────────────


def test_a_model_without_a_clock_reads_none():
    profile = clock_profile(PLAIN)
    assert profile.readers == () and profile.thresholds == () and profile.refusal is None
    assert not profile.reads_clock
    # Tables indexed by a parameter, an observable and a function of neither.
    for name in ("parameter", "observable", "function"):
        assert not clock_profile(TFUN / f"valid_inline_{name}.xml").reads_clock, name


def test_a_switch_is_cut_at_its_threshold():
    profile = clock_profile(SWITCH)
    assert profile.readers == ("f",)
    assert profile.thresholds == (6.0,)
    assert profile.refusal is None


@pytest.mark.parametrize(
    "name", ["inline_step_time", "inline_linear_time", "file_step_time", "file_linear_time"]
)
def test_a_table_indexed_by_time_is_cut_at_its_points(name):
    profile = clock_profile(TFUN / f"valid_{name}.xml")
    assert profile.readers == ("tfun_rate",)
    assert profile.thresholds == (0.0, 1.0, 2.0)
    assert profile.refusal is None


# ── One function at a time ────────────────────────────────────────────────

FREE, SMOOTH, REFUSED = "free", "smooth", "refused"

SHAPES = [
    # Does not read the clock: names that merely contain `t` or `time`.
    ("k*tau", FREE),
    ("k*t_half", FREE),
    ("k*rate_t", FREE),
    ("k*lifetime()", FREE),
    ("k*exp(-kt)", FREE),
    ("if(A0>100,k,0)", FREE),
    # Smooth in the clock, under every spelling the engine reads as the clock.
    ("k*time()", SMOOTH),
    ("k*time", SMOOTH),
    ("k*t", SMOOTH),
    ("k*t()", SMOOTH),
    ("2*(t+1)", SMOOTH),
    ("k*(time +1)", SMOOTH),
    ("time ()", SMOOTH),
    # A number before a name is that number times the name, as ExprTk reads it.
    ("k*2t", SMOOTH),
    ("k*2time()", SMOOTH),
    ("k*.5t", SMOOTH),
    ("k*1e0t", SMOOTH),
    ("k0*exp(((-k)*time()))", SMOOTH),
    ("(k*time())*A0", SMOOTH),
    ("((a*time())+b)+(c*((time()-1)^2))", SMOOTH),
    ("k*time()/2", SMOOTH),
    ("k*time()/a*sin(b*time())", SMOOTH),
    ("k*tanh(time()-3)+cos(time())", SMOOTH),
    # A switch: the clock against a number or a parameter, nothing else.
    ("if(time()>=6,k,0)", (6.0,)),
    ("if((time()>1),k,0)", (1.0,)),
    ("if(time()>=6, if(time()<6.02, 50, 0), 0)", (6.0, 6.02)),
    ("if((time()>1),if((time()<1.01),k,0),0)", (1.0, 1.01)),
    ("k*(time()>=6)", (6.0,)),
    ("if(6<=time(),k,0)", (6.0,)),
    ("if(time()>-2.5,k,0)", (-2.5,)),
    ("if(-2.5<time(),k,0)", (-2.5,)),
    ("if(t<3 || t>=8,k,0)", (3.0, 8.0)),
    ("if(time()>=6 && A0>100,k,0)", (6.0,)),
    ("if(time()>=a,k,0)", ("a",)),
    ("if((time()>=a) && (time()<b),k,0)", ("a", "b")),
    ("floor(2.5*(time()>3))*k", (3.0,)),
    ("max(time()>3, A0>5)*k", (3.0,)),
    # Read some other way: where the rate changes cannot be said.
    ("k*floor(time()/5)", REFUSED),
    ("k*abs(time()-6)", REFUSED),
    ("if(abs(time()-6)<a,k,0)", REFUSED),
    ("if(time()>=6+a,k,0)", REFUSED),
    ("if(time()>=6, if(time()<6+0.02, 50, 0), 0)", REFUSED),
    ("if(2*time()>=6,k,0)", REFUSED),
    ("if(time()>=A0,k,0)", REFUSED),
    ("if(time()>=-a,k,0)", REFUSED),
    ("if(time()==6,k,0)", REFUSED),
    ("if(time()!=6,k,0)", REFUSED),
    ("if(0<time()<5,k,0)", REFUSED),
    ("if(-time()>-6,k,0)", REFUSED),
    ("k*max(0,time()-6)", REFUSED),
    ("k*mod(time(),5)", REFUSED),
    ("k*(time()%5)", REFUSED),
    # Smooth and a switch in one expression: which is which takes a parser.
    ("0.1*time()+if(time()>1,k,0)", REFUSED),
    # A pole, an edge of a domain, or a power that could be either.
    ("k/(time()-6)", REFUSED),
    ("k/time()", REFUSED),
    ("k*time()/a^2", REFUSED),
    ("k*sqrt(time())", REFUSED),
    ("k*ln(time()+1)", REFUSED),
    ("k*tan(time())", REFUSED),
    ("k*2^time()", REFUSED),
    ("k*time()^0.5", REFUSED),
    ("k*time()^-1", REFUSED),
    ("k*time()^a", REFUSED),
    ("k*time()^2^3", REFUSED),
    ("k*pi*time()", REFUSED),
]


@pytest.mark.parametrize(("expression", "expected"), SHAPES)
def test_how_one_function_reads_the_clock(tmp_path, expression, expected):
    xml = _with(tmp_path, expression, parameters=[("a", 2), ("b", 3), ("c", 4), ("k0", 5)])
    profile = clock_profile(xml)
    if expected == REFUSED:
        assert profile.refusal is not None and "'f'" in profile.refusal
    elif expected == FREE:
        assert profile.refusal is None and profile.readers == ()
    elif expected == SMOOTH:
        assert profile.refusal is None and profile.readers == ("f",)
        assert profile.thresholds == ()
    else:
        assert profile.refusal is None and profile.readers == ("f",)
        assert profile.thresholds == expected


def test_a_parameter_named_t_is_not_the_clock(tmp_path):
    """BNG2.pl accepts a parameter named t, and RuleMonkey reads a bare `t` as
    that parameter. The call `time()` is the clock whatever is named `time`."""
    xml = _with(tmp_path, "k*t", parameters=[("t", 2.0)])
    assert not clock_profile(xml).reads_clock
    xml = _with(tmp_path, "k*time", parameters=[("time", 2.0)], name="named_time")
    assert not clock_profile(xml).reads_clock
    xml = _with(tmp_path, "k*time()", parameters=[("time", 2.0)], name="called_time")
    assert clock_profile(xml).readers == ("f",)


def test_markup_inside_an_expression_is_read_through(tmp_path):
    """An element inside <Expression> splits its text; the engine reads the
    whole of it, and so does this."""
    xml = _with(tmp_path, "k*XXtime()")
    xml.write_text(xml.read_text(encoding="utf-8").replace("k*XXtime()", "k*<x/>time()"), "utf-8")
    assert clock_profile(xml).readers == ("f",)


def test_a_comment_no_xml_parser_accepts_is_read_past(tmp_path):
    """`--` inside a comment is not well-formed XML, but RuleMonkey reads the
    file, so this has to as well rather than refuse a model that runs."""
    text = SWITCH.read_text(encoding="utf-8").replace(
        "<sbml ", "<!-- a note -- with two dashes -->\n<sbml ", 1
    )
    xml = tmp_path / "dashes.xml"
    xml.write_text(text, encoding="utf-8")
    assert clock_profile(xml).thresholds == (6.0,)


def test_a_file_that_cannot_be_read_has_no_profile(tmp_path):
    """None, not "no clock": that would be a claim with nothing behind it."""
    assert clock_profile(DATA / "malformed.xml") is None
    assert clock_profile(tmp_path / "missing.xml") is None


# ── Functions built from functions ────────────────────────────────────────


def test_a_function_of_a_switch_is_a_switch_whatever_it_does_to_it(tmp_path):
    xml = _with(
        tmp_path, "floor(2.5*g())+abs(g)", more_functions=_function("g", "if(time()>=4,1,0)")
    )
    profile = clock_profile(xml)
    assert profile.readers == ("f", "g") and profile.thresholds == (4.0,)


def test_a_function_that_only_scales_a_switch_is_still_a_switch(tmp_path):
    """g2 = k*h() is written smoothly but moves only where h switches, so a
    step may be applied to it."""
    more = _function("g2", "k*h()") + _function("h", "if(time()>=4,1,0)")
    xml = _with(tmp_path, "floor(2.5*g2())", more_functions=more)
    profile = clock_profile(xml)
    assert profile.refusal is None
    assert profile.readers == ("f", "g2", "h") and profile.thresholds == (4.0,)


def test_a_smooth_function_of_a_smooth_function_is_smooth(tmp_path):
    xml = _with(tmp_path, "k*g()*exp(-time())", more_functions=_function("g", "1+time()"))
    profile = clock_profile(xml)
    assert profile.readers == ("f", "g") and profile.thresholds == () and profile.refusal is None


def test_a_smooth_function_times_a_switch_carries_the_threshold(tmp_path):
    xml = _with(tmp_path, "k*g()*time()", more_functions=_function("g", "if(time()>=4,1,0)"))
    assert clock_profile(xml).thresholds == (4.0,)


def test_a_step_applied_to_a_smooth_function_is_refused(tmp_path):
    """floor(g()) jumps wherever g passes a whole number, and g = 1 + time()
    does so at times this does not work out."""
    xml = _with(tmp_path, "k*floor(g())", more_functions=_function("g", "1+time()"))
    refusal = clock_profile(xml).refusal
    assert refusal is not None and "'f'" in refusal and "'g'" in refusal and "'floor'" in refusal


def test_the_refusal_names_the_function_that_reads_the_clock_badly(tmp_path):
    """f only calls g; it is g that cannot be placed, whichever is read first."""
    xml = _with(tmp_path, "k*g()", more_functions=_function("g", "floor(time())"))
    refusal = clock_profile(xml).refusal
    assert refusal is not None and "function 'g'" in refusal


def test_functions_that_call_each_other_in_a_ring_are_refused(tmp_path):
    xml = _with(tmp_path, "k*g()", more_functions=_function("g", "f()+time()"))
    assert "refers back to itself" in clock_profile(xml).refusal


# ── Tables ────────────────────────────────────────────────────────────────


def _table(tmp_path, old, new, source="valid_inline_step_time.xml"):
    text = (TFUN / source).read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    xml = tmp_path / "table.xml"
    xml.write_text(text.replace(old, new), encoding="utf-8")
    return xml


def test_a_step_table_may_be_wrapped_in_anything(tmp_path):
    xml = _table(tmp_path, "k_base*__TFUN_VAL__", "if(__TFUN_VAL__>5,k_base,0)")
    profile = clock_profile(xml)
    assert profile.refusal is None and profile.thresholds == (0.0, 1.0, 2.0)


def test_a_linear_table_may_only_be_wrapped_smoothly(tmp_path):
    """Between its points a linear table moves, so `if(table > 5, ...)` switches
    at a time that is not one of them."""
    source = "valid_inline_linear_time.xml"
    xml = _table(tmp_path, "k_base*__TFUN_VAL__", "if(__TFUN_VAL__>5,k_base,0)", source)
    assert clock_profile(xml).refusal is not None
    xml = _table(tmp_path, "k_base*__TFUN_VAL__", "k_base*exp(-__TFUN_VAL__)", source)
    assert clock_profile(xml).refusal is None


def test_a_table_counts_even_where_its_wrapper_leaves_it_out(tmp_path):
    """The engine counts a function with a time-indexed table as moving with
    the clock whether or not its expression reads the table."""
    xml = _table(tmp_path, "k_base*__TFUN_VAL__", "k_base")
    profile = clock_profile(xml)
    assert profile.readers == ("tfun_rate",) and profile.thresholds == (0.0, 1.0, 2.0)


def test_a_table_indexed_by_a_function_of_the_clock_is_refused(tmp_path):
    """The table's value then changes where that function passes a table point,
    not at the table points."""
    text = (TFUN / "valid_inline_function.xml").read_text(encoding="utf-8")
    assert clock_profile(TFUN / "valid_inline_function.xml").refusal is None
    counter = text.split('ctrName="')[1].split('"')[0]
    body = text.split(f'<Function id="{counter}"')[1].split("</Function>")[0]
    old = body.split("<Expression>")[1].split("</Expression>")[0]
    xml = tmp_path / "table_of_clock.xml"
    xml.write_text(
        text.replace(f"<Expression>{old}</Expression>", "<Expression> 2*time() </Expression>"),
        "utf-8",
    )
    refusal = clock_profile(xml).refusal
    assert refusal is not None and "table indexed by" in refusal


def test_a_table_file_that_is_missing_is_refused(tmp_path):
    text = (TFUN / "valid_file_step_time.xml").read_text(encoding="utf-8")
    xml = tmp_path / "no_file.xml"
    xml.write_text(text, encoding="utf-8")  # the .tfun file is not beside it
    assert "table could not be read" in clock_profile(xml).refusal


def test_a_changed_table_file_is_read_again(tmp_path):
    """The reading is kept per file, and a table's points live in another one."""
    xml = tmp_path / "t.xml"
    xml.write_text((TFUN / "valid_file_step_time.xml").read_text(encoding="utf-8"), "utf-8")
    table = tmp_path / "time_step.tfun"
    table.write_text("# time tfun_rate\n0 0\n1 10\n2 20\n", encoding="utf-8")
    assert clock_profile(xml).thresholds == (0.0, 1.0, 2.0)
    assert clock_profile(xml) is clock_profile(xml)
    table.write_text("# time tfun_rate\n0 0\n1 10\n2 20\n3.5 0\n", encoding="utf-8")
    assert clock_profile(xml).thresholds == (0.0, 1.0, 2.0, 3.5)


# ── Thresholds that are parameters ────────────────────────────────────────


def test_breakpoints_read_each_parameter_when_asked(tmp_path):
    xml = _with(
        tmp_path,
        "if((time()>=a) && (time()<b), k, 0) + 0*(time()<1.5)",
        parameters=[("a", 2), ("b", 3)],
    )
    profile = clock_profile(xml)
    assert profile.thresholds == (1.5, "a", "b")
    assert profile.breakpoints({"a": 2.0, "b": 3.0}.__getitem__) == [1.5, 2.0, 3.0]
    # An override made before the run is the one that counts, and two
    # thresholds on one instant are one.
    assert profile.breakpoints({"a": 9.0, "b": 1.5}.__getitem__) == [1.5, 9.0]


# ── The refusal reaches the user ──────────────────────────────────────────


@needs_rulemonkey
def test_a_model_rulemonkey_cannot_be_trusted_with_is_refused_when_built(tmp_path):
    xml = _with(tmp_path, "k*floor(time()/5)")
    model = bngsim.Model.from_net(str(DATA.parent / "simple_decay.net"))
    match = r"RuleMonkey .* cannot run this model: function 'f' .*'floor'.* \(issue #1022\)"
    with pytest.raises(ModelError, match=match):
        bngsim.Simulator(model, method="rm", xml_path=str(xml))
    with pytest.raises(ModelError, match=match):
        bngsim.RuleMonkeySession(str(xml))
    with pytest.raises(ModelError, match="method='ssa' or method='ode'"):
        rulemonkey_clock_profile(xml)


@needs_rulemonkey
def test_a_model_the_reading_cannot_parse_is_refused_when_built(monkeypatch):
    """If the engine loads a file this cannot read, the run is refused rather
    than made on the guess that the model has no clock."""
    monkeypatch.setattr("bngsim._network_free_checks.clock_profile", lambda _path: None)
    model = bngsim.Model.from_net(str(DATA.parent / "simple_decay.net"))
    with pytest.raises(ModelError, match="could not be read to see how its rate laws read"):
        bngsim.Simulator(model, method="rm", xml_path=str(PLAIN))
    with pytest.raises(ModelError, match="could not be read to see how its rate laws read"):
        bngsim.RuleMonkeySession(str(PLAIN))


@needs_rulemonkey
def test_the_engines_own_error_comes_first_for_a_file_it_cannot_load():
    """malformed.xml has no profile, but RuleMonkey's own message says more."""
    with pytest.raises(RuntimeError) as err:
        bngsim.RuleMonkeySession(str(DATA / "malformed.xml"))
    assert "could not be read to see" not in str(err.value)
