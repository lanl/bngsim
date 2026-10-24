"""Two size conditions that flip at one root are one root, however a power in
them is spelled (issue #1187).

A piecewise size jumps at its conditions' roots, one pair of rescale events
per root (#1054), so conditions that flip together are grouped first: two
relations share a root where ``cancel`` reads their differences as a number
times each other. SBML writes a power's exponent as a real, ``<cn>2</cn>``,
which reaches sympy as ``Float(2.0)``, and ``cancel`` reads ``(S + 1)**2.0`` as
no polynomial. So ``(S + 1)^2 > 25`` beside ``S^2 + 2*S + 1 < 25`` were two
roots, the size's two jumps fired on one instant, and the one drawn last
stayed: [Sn] 3.333 for 1.667 on 3 seeds of 4, with no error.

A whole Float exponent is now read as its Integer for the grouping (the
triggers keep the text as written). Oracles: closed forms. S = 10·e^(-t/2),
and Sn keeps its amount, so [Sn] = 5·V(0)/V(t).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
SEEDS = range(4)


def _ap(op, *args):
    return f"<apply><{op}/>{''.join(args)}</apply>"


def _ci(x):
    return f"<ci>{x}</ci>"


def _cn(x):
    return f"<cn>{x}</cn>"


def _sbml(first, second, rules=(), params=()):
    """cv := piecewise(1, first, 3, second, 2), Sn in cv, S decaying in c."""
    cv = (
        f"<piecewise><piece>{_cn(1)}{first}</piece><piece>{_cn(3)}{second}</piece>"
        f"<otherwise>{_cn(2)}</otherwise></piecewise>"
    )
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {_M}>{m}</math></assignmentRule>' for v, m in rules
    )
    param_xml = "".join(
        f'<parameter id="{p}" value="{v}" constant="{str(c).lower()}"/>' for p, v, c in params
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
<listOfCompartments><compartment id="c" size="1" constant="true"/>
<compartment id="cv" size="1" constant="false"/></listOfCompartments>
<listOfSpecies>
<species id="S" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="Sn" compartment="cv" initialConcentration="5" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters><parameter id="k" value="0.5" constant="true"/>{param_xml}</listOfParameters>
<listOfRules>{rule_xml}<assignmentRule variable="cv"><math {_M}>{cv}</math></assignmentRule>
</listOfRules>
<listOfReactions><reaction id="degS" reversible="false"><listOfReactants>
<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
<kineticLaw><math {_M}><apply><times/><ci>k</ci><ci>S</ci><ci>c</ci></apply></math></kineticLaw>
</reaction></listOfReactions></model></sbml>"""


def _power_and_expansion(x, n, c, above=None):
    """``(x + 1)^n > c`` and the same polynomial expanded, ``< c``."""
    first = _ap("gt", _ap("power", _ap("plus", x, _cn(1)), _cn(n)), above or _cn(c))
    if n == 2:
        terms = (_ap("power", x, _cn(2)), _ap("times", _cn(2), x), _cn(1))
    else:
        terms = (
            _ap("power", x, _cn(3)),
            _ap("times", _cn(3), _ap("power", x, _cn(2))),
            _ap("times", _cn(3), x),
            _cn(1),
        )
    return first, _ap("lt", _ap("plus", *terms), _cn(c))


def _sn(xml, times, seed):
    model = bngsim.Model.from_sbml_string(xml)
    result = bngsim.Simulator(model, method="ode").run(sample_times=list(times), seed=seed)
    return np.asarray(result.species)[:, list(result.species_names).index("Sn")]


# S = 10·e^(-t/2) reaches 4 (where (S + 1)^2 = 25 and (S + 1)^3 = 125) at
# t = 2·ln 2.5 ≈ 1.833, where V steps from 1 to 3: [Sn] 5, then 5/3.
TIMES = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]
STEP = [5.0, 5.0, 5.0, 5.0 / 3, 5.0 / 3, 5.0 / 3]


@pytest.mark.parametrize(("n", "c"), [(2, 25), (3, 125)], ids=["square", "cube"])
def test_a_power_and_its_expansion_are_one_root(n, c):
    xml = _sbml(*_power_and_expansion(_ci("S"), n, c))
    assert len(bngsim.Model.from_sbml_string(xml)._core.event_trigger_sources()) == 2
    for seed in SEEDS:
        np.testing.assert_allclose(_sn(xml, TIMES, seed), STEP, rtol=1e-6)


def test_the_triggers_keep_the_power_as_written():
    """The grouping reads the Integer; the trigger it writes is the condition
    as the model spells it."""
    xml = _sbml(*_power_and_expansion(_ci("S"), 2, 25))
    triggers = bngsim.Model.from_sbml_string(xml)._core.event_trigger_sources()
    assert all("^(2.0)" in t for t in triggers), triggers


def test_a_power_beside_its_expansion_at_one_run_constant():
    """``(S + 1)^2 > p1`` beside its expansion ``< p2``, p1 = p2: the roots a
    run constant puts on one instant (#1124) are found through the same
    difference, so the power reads as a polynomial there too."""
    first, second = _power_and_expansion(_ci("S"), 2, 25, above=_ci("p1"))
    second = second.replace(_cn(25), _ci("p2"))
    xml = _sbml(first, second, params=[("p1", 25, True), ("p2", 25, True)])
    for seed in SEEDS:
        np.testing.assert_allclose(_sn(xml, TIMES, seed), STEP, rtol=1e-6)


def _diamond(n):
    rules = [("r0", _ci("S")), ("r1", _ap("divide", _ci("S"), _cn(2)))]
    rules += [
        (f"r{i}", _ap("plus", _ap("sin", _ci(f"r{i - 1}")), _ap("cos", _ci(f"r{i - 2}"))))
        for i in range(2, n)
    ]
    return rules, [(f"r{i}", 0, False) for i in range(n)]


def _diamond_sn(n, t):
    """5·V(0)/V(t), V read from the rules in closed form."""

    def size(t):
        s = 10 * math.exp(-0.5 * t)
        r = [s, s / 2]
        for i in range(2, n):
            r.append(math.sin(r[i - 1]) + math.cos(r[i - 2]))
        x = r[n - 1]
        return 1 if (x + 1) ** 2 > 5.76 else 3 if x * x + 2 * x + 1 < 5.76 else 2

    return 5 * size(0) / size(t)


@pytest.mark.parametrize(
    ("n", "times", "seeds"),
    [(6, [0.0, 0.5, 2.0, 5.0], SEEDS), (20, [0.0, 0.3, 1.5, 4.5], [0])],
    ids=["6 rules", "20 rules"],
)
def test_a_power_and_its_expansion_over_a_diamond_of_rules(n, times, seeds):
    """The blind review's shape, ``(r + 1)^2 > 5.76`` beside its expansion over
    ``r_i := sin(r_{i-1}) + cos(r_{i-2})``: the size goes 3, 1, 3 as r crosses
    1.4 twice, so [Sn] is 5, 15, 5. On main it never made the first jump and
    ended at 5/3 or 10/3. At 20 rules the differences pass 5,000 tree nodes and
    are read term by term and at two points before ``cancel``."""
    rules, params = _diamond(n)
    xml = _sbml(*_power_and_expansion(_ci(f"r{n - 1}"), 2, 5.76), rules=rules, params=params)
    assert len(bngsim.Model.from_sbml_string(xml)._core.event_trigger_sources()) == 2
    want = [_diamond_sn(n, t) for t in times]
    assert want[1:] == [5.0, 15.0, 5.0]
    for seed in seeds:
        np.testing.assert_allclose(_sn(xml, times, seed), want, rtol=1e-6)


def test_a_power_that_is_not_whole_is_left_as_written():
    """``S^0.5 > 2`` beside ``S < 4`` share a root, but their differences are
    no number times each other, Integer exponent or not: they stay two roots,
    as on main. That coincidence is #1181's, for the event loop."""
    first = _ap("gt", _ap("power", _ci("S"), _cn(0.5)), _cn(2))
    xml = _sbml(first, _ap("lt", _ci("S"), _cn(4)))
    assert len(bngsim.Model.from_sbml_string(xml)._core.event_trigger_sources()) == 4
