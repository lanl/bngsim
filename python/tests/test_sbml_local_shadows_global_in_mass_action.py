"""A kinetic law's local parameter shadows a global of the same id in a
mass-action law (issue #1503).

SBML scoping: inside a kinetic law, a local parameter overrides any global
with the same id, compartment and species included. The mass-action
classifier tested a name as a species and then as a compartment before it
tested the law's locals, so in ``C*k*S`` with a local ``C`` the local was
cancelled against the storage volume as the compartment, and in ``k*S*Q``
with a local ``Q`` the local was multiplied in as the species ``Q``. The ODE,
the Jacobian, the SSA propensity and both codegen paths shared the wrong rate,
with no warning. A law with no constant parameter besides the shadowed id took
the Functional path, which already read locals first, which is why the SBML
test suite's shadowing cases passed.

The model is S -> (nothing) in compartment C (size 1 unless said), with
k = 0.1, [S](0) = 3 and a species Q = 1.5 in no reaction. Under SBML scoping
every law here is 0.7*[S] in amount per time, so d[S]/dt = -0.7*[S]/V and
[S](t) = 3*exp(-0.7 t / V). The right-hand side at t = 0 is algebra, compared
at rel 1e-12; trajectories run at rtol 1e-10 / atol 1e-12 and are compared
with their closed forms at rel 1e-7, far inside the factor of 3 and more the
defect moved them.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest

TOL = {"rtol": 1e-10, "atol": 1e-12}
REL = 1e-7


def sbml(
    law, locals_=(), *, size=1, level=(3, 2), reversible=False, products="", extra="", params=""
):
    """S -> (products) with kinetic law *law* (MathML) and its *locals_*."""
    lp_list = "listOfLocalParameters"
    lp = "".join(f'<localParameter id="{i}" value="{v}"/>' for i, v in locals_)
    ns = f"http://www.sbml.org/sbml/level{level[0]}/version{level[1]}/core"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="{ns}" level="{level[0]}" version="{level[1]}"><model id="m">
<listOfCompartments>
<compartment id="C" spatialDimensions="3" size="{size}" constant="true"/>
</listOfCompartments>
<listOfSpecies>
<species id="S" compartment="C" initialConcentration="3" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="T" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="Q" compartment="C" initialConcentration="1.5" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters>
<parameter id="k" value="0.1" constant="true"/>
<parameter id="kf" value="0.1" constant="true"/>
<parameter id="kr" value="0.05" constant="true"/>
{params}
</listOfParameters>
{extra}
<listOfReactions><reaction id="J" reversible="{str(reversible).lower()}">
<listOfReactants>
<speciesReference species="S" stoichiometry="1" constant="true"/>
</listOfReactants>
{products}
<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">{law}</math>
<{lp_list}>{lp}</{lp_list}>
</kineticLaw></reaction></listOfReactions></model></sbml>"""


def times(*names):
    return "<apply><times/>" + "".join(f"<ci>{n}</ci>" for n in names) + "</apply>"


def load(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Model.from_sbml_string(text)


def dsdt0(model):
    names = list(model.species_names)
    y0 = [model.get_concentration(s) for s in names]
    return float(model.rhs(y0)[names.index("S")])


def s_at(model, t, **kw):
    names = list(model.species_names)
    r = bngsim.Simulator(model, method="ode", **kw).run(t_span=(0, t), n_points=3, **TOL)
    return float(np.asarray(r.species)[-1, names.index("S")])


# ── The issue's rows ─────────────────────────────────────────────────────────

ROWS = {
    "a-local-named-like-the-compartment": (times("C", "k", "S"), (("C", 7),)),
    "a-local-named-like-a-species-outside-the-reaction": (times("k", "S", "Q"), (("Q", 7),)),
    "control-a-local-named-like-a-global-parameter": (times("C", "k", "S"), (("k", 0.7),)),
}


@pytest.mark.parametrize(("law", "locals_"), ROWS.values(), ids=ROWS.keys())
def test_the_law_reads_its_local(law, locals_):
    """d[S]/dt(0) = -0.7*3 = -2.1 and [S](2) = 3 e^-1.4 in every row. On main
    the local C gave -0.3 (2.456 at t = 2) and the local Q -0.45 (2.222)."""
    model = load(sbml(law, locals_))
    assert dsdt0(model) == pytest.approx(-2.1, rel=1e-12)
    assert s_at(model, 2.0) == pytest.approx(3 * math.exp(-1.4), rel=REL)


def test_the_propensity_and_codegen_read_it_too():
    """The SSA propensity is the amount rate, 0.7*3 = 2.1 (V = 1); the codegen
    right-hand side agrees with the interpreted one."""
    text = sbml(times("C", "k", "S"), (("C", 7),))
    model = load(text)
    assert float(np.asarray(model.propensities([3.0, 1.0, 1.5]))[0]) == pytest.approx(
        2.1, rel=1e-12
    )
    assert s_at(load(text), 2.0, codegen=True) == pytest.approx(3 * math.exp(-1.4), rel=REL)


def test_a_compartment_of_another_size_is_still_shadowed():
    """V = 2: the amount rate is 0.7*[S], so d[S]/dt = -0.7*3/2 = -1.05 and
    [S](2) = 3 e^-0.7. The control without the local reads the compartment,
    C*k*[S]/V = 0.1*[S]: -0.3."""
    model = load(sbml(times("C", "k", "S"), (("C", 7),), size=2))
    assert dsdt0(model) == pytest.approx(-1.05, rel=1e-12)
    assert s_at(model, 2.0) == pytest.approx(3 * math.exp(-0.7), rel=REL)
    assert dsdt0(load(sbml(times("C", "k", "S"), size=2))) == pytest.approx(-0.3, rel=1e-12)


def test_the_reversible_splitter_reads_the_local():
    """C*(kf*S - kr*T) with a local C = 7, S <-> T, V = 1: d[S]/dt =
    -7*(0.1*S - 0.05*T), so S + T = 4 and S(t) = s* + (3 - s*) e^(-1.05 t)
    with s* = 4*0.05/0.15. d[S]/dt(0) = -7*(0.3 - 0.05) = -1.75."""
    law = (
        "<apply><times/><ci>C</ci><apply><minus/>"
        + times("kf", "S")
        + times("kr", "T")
        + "</apply></apply>"
    )
    products = (
        '<listOfProducts><speciesReference species="T" stoichiometry="1" constant="true"/>'
        "</listOfProducts>"
    )
    model = load(sbml(law, (("C", 7),), reversible=True, products=products))
    assert dsdt0(model) == pytest.approx(-1.75, rel=1e-12)
    eq = 4 * 0.05 / 0.15
    assert s_at(model, 2.0) == pytest.approx(eq + (3 - eq) * math.exp(-1.05 * 2), rel=REL)


def test_a_local_named_like_the_reactant_itself_in_l3v1():
    """L3V1 allows a local to shadow a species the reaction references: k*S
    with a local S = 7 is the constant 0.7, so [S](2) = 3 - 0.7*2 = 1.6."""
    model = load(sbml(times("k", "S"), (("S", 7),), level=(3, 1)))
    assert dsdt0(model) == pytest.approx(-0.7, rel=1e-12)
    assert s_at(model, 2.0) == pytest.approx(1.6, rel=REL)


def test_the_model_agrees_with_itself():
    """An assignment rule q := J reads the law in its own scope, 7*0.1*3 = 2.1;
    the right-hand side now integrates the same rate."""
    rule = (
        '<listOfRules><assignmentRule variable="q"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<ci>J</ci></math></assignmentRule></listOfRules>"
    )
    q = '<parameter id="q" value="0" constant="false"/>'
    model = load(sbml(times("C", "k", "S"), (("C", 7),), extra=rule, params=q))
    assert dsdt0(model) == pytest.approx(-2.1, rel=1e-12)


def test_the_sensitivity_to_the_local_is_not_a_structural_zero():
    """d[S](2)/dC_local for [S] = 3 e^(-C k t): -3 k t e^(-C k t) = -0.6 e^-1.4.
    On main the column was zero: the local carried no rate. rel 1e-5: forward
    sensitivities at rtol 1e-10 carry their own error terms."""
    model = load(sbml(times("C", "k", "S"), (("C", 7),)))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["_lp_J_C"])
    r = sim.run(t_span=(0, 2), n_points=3, **TOL)
    i = list(model.species_names).index("S")
    got = float(np.asarray(r.sensitivities)[-1, i, 0])
    assert got == pytest.approx(-0.6 * math.exp(-1.4), rel=1e-5)


# ── The volume taint reads the law in its own scope too ─────────────────────


def _with_ia_reading_j(law, locals_):
    ia = (
        '<listOfInitialAssignments><initialAssignment symbol="p">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>J</ci></math>'
        "</initialAssignment></listOfInitialAssignments>"
    )
    p = '<parameter id="p" value="0" constant="true"/>'
    return sbml(law, locals_, extra=ia, params=p)


def test_a_shadowing_local_does_not_refuse_a_write_to_the_compartment():
    """A parameter set from J's initial value: with a local C the law reads no
    compartment, so a write to C's size is not refused as volume-dependent, and
    the model then runs as one loaded with C = 2 (d[S]/dt(0) = -1.05). On main
    the law's raw names tainted J with C and the write was refused. What p was
    folded from stays frozen: J's local, as `_lp_J_C`, and k."""
    model = load(_with_ia_reading_j(times("C", "k", "S"), (("C", 7),)))
    model.set_param("C", 2.0)
    assert dsdt0(model) == pytest.approx(-1.05, rel=1e-12)
    for frozen in ("_lp_J_C", "k"):
        with pytest.raises(
            bngsim.ParameterError, match="cannot be changed after the model is loaded"
        ):
            model.set_param(frozen, 1.0)


def test_a_law_that_reads_the_compartment_still_refuses_the_write():
    """Control: with the global C in the law, J reads the size and p's initial
    value folds it, so the write stays refused."""
    model = load(_with_ia_reading_j(times("C", "k", "S"), ()))
    with pytest.raises(ValueError, match="Cannot set 'C'"):
        model.set_param("C", 2.0)
