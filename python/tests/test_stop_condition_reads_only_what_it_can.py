"""Issue #1509: a stop condition that cannot read a name is refused, not read as never met.

``add_stop_condition`` accepted an expression that read a name it could not bind (a
misspelled observable, a parameter, a model function, a builtin such as ``abs``) or
used BNGL's ``^``. Every sample that reached the name raised, each failure was
logged at DEBUG and skipped, and the run reported no condition met: a typo turned
the condition off, and behind ``or`` it moved the cut. Those are now refused where
the condition is added; an expression that raises at a sample raises from the run;
and a callable's exception propagates.

A() decays at k = 0.1 from 100, observed as A_tot: A_tot < 10 first holds at
t = 10 ln 10 = 23.03, the sample t = 24 of a 0..100 grid at unit spacing.
"""

from __future__ import annotations

import re
from pathlib import Path

import bngsim
import pytest

NET = (
    "begin parameters\n 1 k 0.1\nend parameters\n"
    "begin species\n 1 A() 100\nend species\n"
    "begin reactions\n 1 1 0 k\nend reactions\n"
    "begin groups\n 1 A_tot 1\nend groups\n"
    "begin functions\n 1 frac() A_tot/100\nend functions\n"
)
KW = dict(t_span=(0, 100), n_points=101)
MET_AT = 24.0  # the first sample where 100*exp(-0.1 t) < 10

# S := 2*A by an SBML assignment rule, which loads as a model function.
SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
      <parameter id="S" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>2</cn><ci>A</ci></apply></math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="r" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

NF_DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"


def _available(method: str) -> bool:
    if method == "nf":
        return bool(getattr(bngsim, "HAS_NFSIM", False))
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# Literal reasons, as the other network-free tests give them (test_skip_audit.py).
NF_METHODS = [
    pytest.param(
        "nf", marks=pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
    ),
    pytest.param(
        "rm",
        marks=pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in"),
    ),
]


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "decay.net"
    path.write_text(NET)
    return str(path)


def _sim(net: str) -> bngsim.Simulator:
    return bngsim.Simulator(bngsim.Model.from_net(net), method="ode")


def _stop_time(sim: bngsim.Simulator) -> float | None:
    try:
        sim.run(**KW)
    except bngsim.StopConditionMet as stop:
        return float(stop.result.time[-1])
    return None


@pytest.mark.parametrize(
    "condition, why",
    [
        ("A_totl < 10", "A_totl is not a name the model defines"),
        ("k*A_tot < 1", "k is a parameter"),
        ("frac < 0.1", "frac is a function of the model"),
        ("abs(A_tot - 5) < 1", "abs is a Python builtin"),
        # Behind and/or the typo is reached at some samples only: under 'and' the
        # condition never held, and under 'or' the cut moved from t = 7 to t = 30.
        ("A_tot < 10 and A_totl > -1", "A_totl is not a name the model defines"),
        ("A_tot < 5 or A_totl < 50", "A_totl is not a name the model defines"),
    ],
)
def test_a_name_the_condition_cannot_read_is_refused_where_it_is_added(net, condition, why):
    """Each was accepted, and the run met no condition (or met a moved one)."""
    with pytest.raises(ValueError, match=re.escape(why)) as exc:
        _sim(net).add_stop_condition(condition)
    # The refusal says what a condition can read.
    message = str(exc.value)
    assert "the observables (A_tot)" in message and "A() as A__" in message


@pytest.mark.parametrize("options", [{"method": "ssa"}, {"method": "psa", "poplevel": 100}])
def test_the_refusal_does_not_depend_on_the_method(net, options):
    sim = bngsim.Simulator(bngsim.Model.from_net(net), **options)
    with pytest.raises(ValueError, match="A_totl is not a name the model defines"):
        sim.add_stop_condition("A_totl < 10")


def test_an_sbml_assignment_rule_is_refused_as_a_function(tmp_path):
    """S := 2*A loads as a model function, which no sample binds: 'S < 5' never fired."""
    path = tmp_path / "m.xml"
    path.write_text(SBML)
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode")
    with pytest.raises(ValueError, match="S is a function of the model"):
        sim.add_stop_condition("S < 5")
    sim.add_stop_condition("A < 5")  # Control: the species reads as before.


def test_caret_is_refused_with_the_power_spelled(net):
    """BNGL's power is XOR in Python and raised TypeError at every sample."""
    with pytest.raises(ValueError, match=re.escape("write ** (A_tot**2 < 100)")):
        _sim(net).add_stop_condition("A_tot^2 < 100")


def test_a_tuple_is_refused(net):
    """'A_tot < 10, A_tot > 0' is a tuple, truthy at every sample: it was met at t = 0."""
    with pytest.raises(ValueError, match="is a tuple"):
        _sim(net).add_stop_condition("A_tot < 10, A_tot > 0")


def test_a_number_is_met_where_it_is_positive(net):
    """A number meets the condition where it is > 0, as the docstring has always said.
    Any nonzero value met it, so 10 - A_tot (-90 at t = 0) was met at t = 0."""
    sim = _sim(net)
    sim.add_stop_condition("10 - A_tot")
    assert _stop_time(sim) == MET_AT


def test_a_number_that_is_nan_does_not_meet_the_condition(net):
    """(A_tot - A_tot)/(A_tot - A_tot) is 0/0 = NaN at every sample, and NaN is not
    > 0, so the condition is never met. NaN is truthy, so it was met at t = 0."""
    sim = _sim(net)
    sim.add_stop_condition("(A_tot - A_tot) / (A_tot - A_tot)")
    with pytest.warns(RuntimeWarning):
        assert _stop_time(sim) is None


def test_a_value_neither_a_truth_value_nor_a_number_raises_from_the_run(net):
    """A list comprehension is a list, truthy at every sample: it was met at t = 0."""
    sim = _sim(net)
    sim.add_stop_condition("[x < 10 for x in (A_tot,)]")
    with pytest.raises(ValueError, match="gives list at t=0, not a truth value or a number"):
        sim.run(**KW)


def test_an_expression_that_raises_at_a_sample_raises_from_the_run(net):
    """'A_tot < 10 & t > 0' reads only names it can, but & binds before <, so 10 & t
    raises TypeError at every sample. The run reported no condition met."""
    sim = _sim(net)
    sim.add_stop_condition("A_tot < 10 & t > 0")
    with pytest.raises(ValueError, match="could not be evaluated at t=0: TypeError"):
        sim.run(**KW)


def test_a_batch_row_that_cannot_be_evaluated_raises(net):
    """run_batch returned the row with stop_condition=None, as one that met none."""
    sim = _sim(net)
    sim.add_stop_condition("A_tot < 10 & t > 0")
    with pytest.raises(ValueError, match="could not be evaluated"):
        sim.run_batch(params=[{"k": 0.1}], **KW)


def test_a_callable_that_raises_is_not_read_as_not_met(net):
    """The KeyError was swallowed and the run returned to t = 100."""
    sim = _sim(net)
    sim.add_stop_condition(lambda r: r.observables["A_totl"][-1] < 10, label="typo")
    with pytest.raises(KeyError, match="A_totl"):
        sim.run(**KW)


def test_a_callable_that_returns_a_column_raises(net):
    """Without [-1] the callable returns an array, whose truth value numpy refuses;
    the error was swallowed and the run returned to t = 100."""
    sim = _sim(net)
    sim.add_stop_condition(lambda r: r.observables["A_tot"] < 10, label="column")
    with pytest.raises(ValueError, match="truth value"):
        sim.run(**KW)


@pytest.mark.parametrize("method", NF_METHODS)
def test_a_species_is_refused_on_a_network_free_run(method):
    """A network-free result carries no species, so a condition on one never fired."""
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(NF_DATA / "model_parameters.net")),
        method=method,
        xml_path=str(NF_DATA / "model_parameters.xml"),
    )
    with pytest.raises(ValueError, match="L_r_ is a species, and a network-free run reports none"):
        sim.add_stop_condition("L_r_ < 100")
    sim.add_stop_condition("Lfree < 100")  # Control: an observable reads as before.


@pytest.mark.parametrize(
    "condition",
    [
        "A_tot < 10",
        "A__ < 10",
        "t >= 24",
        "time >= 24",
        "A_tot**2 < 100",
        "(lambda x: x < 10)(A_tot)",
        "[x < 10 for x in (A_tot,)][0]",
        "(a := A_tot) < 10 and a >= 0",
        "A_tot.real < 10",
    ],
)
def test_the_names_a_condition_can_read_read_as_before(net, condition):
    """Control: the observables, a species by its identifier spelling, the time, a
    power spelled **, names the expression binds itself and an attribute are read,
    and each condition is met at t = 24."""
    sim = _sim(net)
    sim.add_stop_condition(condition)
    assert _stop_time(sim) == MET_AT
