"""A size that moves with the run and whose rate bngsim cannot follow is
refused by name, not left undiluted (issue #1193).

A concentration species in a compartment an assignment rule sizes is diluted
by the size's rate, V̇, which the loader writes through the chain rule (#1052).
Where V̇ is unknown, the compartment was left out of the dilution pass with no
word, and its concentrations came back as if the size held:

* ``cv := 2 + rateOf(S3)`` with S3 decaying: the chain rule read the ``rateOf``
  token as a constant, so V̇ was 0. [X] was 4·e^(-t/4), the trajectory of a size
  that holds, where the amount over cv(t) gives 4, 3.0826, 2.3782.
* The same through a rule, ``r := rateOf(S3)``, and a cycle of sizes that moves
  (``cv := 1 + 0.1*cw + 0.01*S3``, ``cw := 1 + 0.1*cv``): V̇ was None, and the
  pass skipped it.

Each is now refused, naming the compartment, its rule and why its rate is
unknown. A size that holds is not: the rateOf of a species that never moves is
0 for good, and a cycle of constants is solved once (#621).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import ModelError

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_ROF = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/rateOf">'
    "rateOf</csymbol>"
)


def a(op, *x):
    return f"<apply><{op}/>{''.join(x)}</apply>"


def ci(x):
    return f"<ci>{x}</ci>"


def cn(x):
    return f"<cn>{x}</cn>"


def _species(sid, comp, conc):
    return (
        f'<species id="{sid}" compartment="{comp}" initialConcentration="{conc}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _decay(rid, sid, law):
    return (
        f'<reaction id="{rid}" reversible="false"><listOfReactants>'
        f'<speciesReference species="{sid}" stoichiometry="1" constant="true"/>'
        f"</listOfReactants><kineticLaw><math {_M}>{law}</math></kineticLaw></reaction>"
    )


def _compartment(cid, size):
    return f'<compartment id="{cid}" size="{size}" constant="false"/>'


def doc(rules, params="", comps=None, s3_decays=True):
    """S3 in c (decaying at 0.1·S3 unless told not to), X in cv at 0.25·X·cv,
    and the rules given."""
    comps = comps or _compartment("cv", 2)
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {_M}>{m}</math></assignmentRule>' for v, m in rules
    )
    reactions = _decay("dX", "X", a("times", cn(0.25), ci("X"), ci("cv")))
    if s3_decays:
        reactions = _decay("d3", "S3", a("times", cn(0.1), ci("S3"), ci("c"))) + reactions
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        '<model id="m"><listOfCompartments><compartment id="c" size="1" constant="true"/>'
        f"{comps}</listOfCompartments>"
        f"<listOfSpecies>{_species('S3', 'c', 2)}{_species('X', 'cv', 4)}</listOfSpecies>"
        f"<listOfParameters>{params}</listOfParameters><listOfRules>{rule_xml}</listOfRules>"
        f"<listOfReactions>{reactions}</listOfReactions></model></sbml>"
    )


def _x(xml):
    model = bngsim.Model.from_sbml_string(xml)
    result = bngsim.Simulator(model, method="ode").run(sample_times=[0, 1, 2])
    return np.asarray(result.species)[:, list(result.species_names).index("X")]


HELD = 4 * np.exp(-np.array([0, 1, 2]) / 4)  # [X] where cv holds: X' = -0.25·X·cv / cv
ROF_S3 = f"<apply>{_ROF}<ci>S3</ci></apply>"
CYCLE = _compartment("cv", 1) + _compartment("cw", 1)


def _refused(xml, *names):
    with pytest.raises(ModelError) as info:
        bngsim.Model.from_sbml_string(xml)
    for name in ("compartment cv", "moves with the run", "would not be diluted", *names):
        assert name in str(info.value), (name, str(info.value))


def test_a_size_that_reads_a_rate_is_refused_by_name():
    """The issue's model: cv := 2 + rateOf(S3), S3 decaying."""
    _refused(doc([("cv", a("plus", cn(2), ROF_S3))]), "cv := 2 + rateOf(S3)", "rateOf(S3)")


def test_a_size_that_reads_a_rate_through_a_rule_names_the_rule():
    xml = doc(
        [("r", ROF_S3), ("cv", a("plus", cn(2), ci("r")))],
        params='<parameter id="r" value="0" constant="false"/>',
    )
    _refused(xml, "through r", "rateOf(S3)")


def test_a_cycle_of_sizes_that_moves_is_refused_by_name():
    rules = [
        ("cv", a("plus", cn(1), a("times", cn(0.1), ci("cw")), a("times", cn(0.01), ci("S3")))),
        ("cw", a("plus", cn(1), a("times", cn(0.1), ci("cv")))),
    ]
    _refused(doc(rules, comps=CYCLE), "a cycle of rules")


def test_a_size_that_reads_the_rate_of_a_species_that_never_moves_loads():
    """The issue's control: S3 in no reaction, so rateOf(S3) is 0 and cv holds."""
    xml = doc([("cv", a("plus", cn(2), ROF_S3))], s3_decays=False)
    np.testing.assert_allclose(_x(xml), HELD, rtol=1e-6)


def test_a_cycle_of_constant_sizes_loads():
    rules = [
        ("cv", a("plus", cn(1), a("times", cn(0.1), ci("cw")))),
        ("cw", a("plus", cn(1), a("times", cn(0.1), ci("cv")))),
    ]
    np.testing.assert_allclose(_x(doc(rules, comps=CYCLE)), HELD, rtol=1e-6)


def test_a_size_that_reads_a_species_is_still_diluted():
    """#1052's case, cv := 2 + 0.1*S3: its rate is known, and it is diluted."""
    x = _x(doc([("cv", a("plus", cn(2), a("times", cn(0.1), ci("S3"))))]))
    assert not np.allclose(x, HELD, rtol=1e-3)


def test_a_cycle_of_sizes_through_conditions_is_refused_by_name():
    """#1185 item 1: each size's jump reads the other size. After t = 2 both
    (1, 5) and (5, 1) are consistent; it ran with NaN amounts and no jump."""
    t = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'

    def pw(other):
        cond = a("or", a("lt", t, cn(2)), a("gt", ci(other), cn(2)))
        return f"<piecewise><piece>{cn(1)}{cond}</piece><otherwise>{cn(5)}</otherwise></piecewise>"

    _refused(doc([("cv", pw("cw")), ("cw", pw("cv"))], comps=CYCLE), "a cycle of rules")
