"""GH #706: the non-finite-RHS retry clamps only concentrations.

The GH #135 retry re-evaluates a non-finite RHS on a copy of the state with
negative components set to 0, which rescues a sqrt/fractional-power law at a
concentration the predictor pushed a hair below zero. It set EVERY negative
component to 0, so a quantity that is negative by design followed the wrong
ODE for the rest of the run, silently: an SBML rate-rule parameter V = -61
relaxing toward -70 ran away past -287, and a rate-rule X fed by a boundary
species B = -5 froze at -4.32. Now only slots that are concentrations by
construction are clamped.

In every model S is consumed at C*k*sqrt(S), reaches 0 at t = 2 and parks a
hair below it, which is what fires the retry: S(t) = (1 - t/2)^2 up to t = 2,
then 0. V and X do not read S: V(t) = -70 + 9 exp(-0.1 t) and
X(t) = N (1 - exp(-t)) for the value N it relaxes to.

The runs use the finite-difference Jacobian. With S pinned at 0 the analytical
d/dS sqrt(S) is infinite even after the clamp, so the closed form is refused
there (issue #867) and ``jacobian="auto"`` would only reach these same difference
quotients after a failed first attempt. The difference quotients go through the
RHS and its retry, which is what this file is about.

What decides "negative by design" for a species is the DECLARED initial value,
not the IC baseline: save_concentrations() can make the baseline a depleted
concentration a hair below 0, and that one must keep the rescue.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest

M = "http://www.w3.org/1998/Math/MathML"
_SQRT_SINK = (
    '<reaction id="R1" reversible="false"><listOfReactants>'
    '<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>'
    f'<kineticLaw><math xmlns="{M}"><apply><times/><ci>C</ci><ci>k</ci>'
    "<apply><root/><ci>S</ci></apply></apply></math></kineticLaw></reaction>"
)
_S = (
    '<species id="S" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
)


def _doc(species: str, params: str, rule: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>{_S}{species}</listOfSpecies>
<listOfParameters><parameter id="k" value="1" constant="true"/>{params}</listOfParameters>
<listOfRules>{rule}</listOfRules>
<listOfReactions>{_SQRT_SINK}</listOfReactions>
</model></sbml>"""


v1 = _doc(
    "",
    '<parameter id="V" value="-61" constant="false"/>'
    '<parameter id="g" value="0.1" constant="true"/>'
    '<parameter id="E" value="-70" constant="true"/>',
    f'<rateRule variable="V"><math xmlns="{M}"><apply><times/><ci>g</ci>'
    "<apply><minus/><ci>E</ci><ci>V</ci></apply></apply></math></rateRule>",
)


def _relax_to(species: str, name: str) -> str:
    # A rate-rule X relaxing to the value of `name`, which `species` declares.
    return _doc(
        species,
        '<parameter id="X" value="0" constant="false"/>',
        f'<rateRule variable="X"><math xmlns="{M}">'
        f"<apply><minus/><ci>{name}</ci><ci>X</ci></apply></math></rateRule>",
    )


v2 = _relax_to(
    '<species id="B" compartment="C" initialConcentration="-5" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="true" constant="true"/>',
    "B",
)
# A floating species declared negative, which no reaction touches.
v3 = _relax_to(
    '<species id="N" compartment="C" initialConcentration="-3" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>',
    "N",
)


def _col(r, name):
    if name in r.species_names:
        return np.asarray(r.species[:, list(r.species_names).index(name)])
    return np.asarray(r.observables[name])


def _run(doc, codegen, t_span=(0, 40)):
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(doc), method="ode", codegen=codegen, jacobian="fd"
    )
    return sim, sim.run(t_span=t_span, n_points=int(t_span[1] - t_span[0]) + 1)


def _check_s(r):
    t = np.asarray(r.time)
    s = _col(r, "S")
    assert s.min() < 0.0  # the retry did fire
    np.testing.assert_allclose(s, np.where(t < 2.0, (1.0 - t / 2.0) ** 2, 0.0), atol=1e-6)


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
def test_a_negative_rate_rule_parameter_follows_its_own_ode(codegen):
    _, r = _run(v1, codegen)
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "V"), -70.0 + 9.0 * np.exp(-0.1 * t), rtol=1e-4)
    _check_s(r)


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
@pytest.mark.parametrize("doc, n", [(v2, -5.0), (v3, -3.0)], ids=["boundary", "floating"])
def test_a_species_declared_negative_keeps_its_value(doc, n, codegen):
    _, r = _run(doc, codegen)
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "X"), n * (1.0 - np.exp(-t)), rtol=1e-4, atol=1e-6)
    _check_s(r)


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
def test_a_saved_depleted_concentration_keeps_the_rescue(codegen):
    # Equilibrate, save and restore: S's baseline is now a hair below 0. It is
    # still a concentration, so a restart from it must still be rescued, not
    # fail at the first RHS call; N keeps the value it was declared with.
    sim, r = _run(v3, codegen, t_span=(0, 10))
    assert _col(r, "S")[-1] < 0.0
    sim.save_concentrations()
    sim.restore_concentrations()
    r = sim.run(t_span=(0, 10), n_points=11)
    np.testing.assert_allclose(_col(r, "S"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_col(r, "N"), -3.0)


# Issue #1661: a species negative where the model has it so, but not declared
# negative. V reads no other species, dV/dt = g (E - V), so
# V(t) = E + (V(0) - E) exp(-g t) whatever S does; S is the sink above. The
# retry read V at 0 too, though no non-finite law reads V: from t = 2, where S
# parks below 0, V followed dV/dt = g E = -7. Now the retry first clamps only
# the slots a non-finite row of f (or of the Jacobian) reads.
_V_BESIDE_SINK = """model m
  species S, V; S = 1; V = %s;
  k = 1; g = 0.1; E = -70;
  R1: S -> ; k*sqrt(S);
  R2: -> V; g*(E - V);
end"""

_ROUTES = [
    pytest.param(False, False, id="interp-dense"),
    pytest.param(True, False, id="codegen-dense"),
    pytest.param(
        False,
        True,
        id="interp-klu",
        marks=pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled"),
    ),
    pytest.param(
        True,
        True,
        id="codegen-klu",
        marks=pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled"),
    ),
]


def _run_v(model, codegen, sparse):
    sim = bngsim.Simulator(
        model,
        method="ode",
        codegen=codegen,
        jacobian="fd",
        force_sparse_linear_solver=sparse,
    )
    r = sim.run(t_span=(0, 40), n_points=41)
    # The route the run took: a codegen leg that fell back to the interpreter
    # would test the interpreted retry twice.
    assert (sim.codegen_backend != "exprtk") == codegen
    assert (r.solver_stats["linear_solver"] == 1) == sparse
    return r


def _check_v(r, v0):
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "V"), -70.0 + (v0 + 70.0) * np.exp(-0.1 * t), rtol=1e-6)


@pytest.mark.parametrize(("codegen", "sparse"), _ROUTES)
def test_a_species_written_negative_follows_its_own_ode(codegen, sparse):
    # Main: V(40) = -328.63 for -69.8352.
    model = bngsim.Model.from_antimony_string(_V_BESIDE_SINK % "0")
    model.set_concentration("V", -61.0)
    r = _run_v(model, codegen, sparse)
    _check_v(r, -61.0)
    _check_s(r)


@pytest.mark.parametrize(("codegen", "sparse"), _ROUTES)
def test_a_species_its_own_law_takes_below_zero_follows_its_own_ode(codegen, sparse):
    # Declared at 10, V crosses 0 at t = 2.5 on its own. Main: V(40) = -270.50
    # for -68.5347.
    r = _run_v(bngsim.Model.from_antimony_string(_V_BESIDE_SINK % "10"), codegen, sparse)
    _check_v(r, 10.0)
    _check_s(r)


def test_a_batch_row_whose_species_crosses_zero_follows_its_own_ode():
    # A row's dose of V0 = 10 sets V's declared value at 10, so the row is the
    # crossing case above. Main: -270.50 for -68.5347. (V0 = -61 was right on
    # main: a dose through a parameter re-declares V negative, #706.)
    model = bngsim.Model.from_antimony_string(
        (_V_BESIDE_SINK % "V0").replace("k = 1;", "k = 1; V0 = 0;")
    )
    rows = bngsim.Simulator(model, method="ode", jacobian="fd").run_batch(
        (0, 40), 41, params=[{"V0": 10.0}, {"V0": -61.0}]
    )
    for row, v0 in zip(rows, (10.0, -61.0), strict=True):
        _check_v(row, v0)
        _check_s(row)


@pytest.mark.parametrize(("codegen", "sparse"), _ROUTES)
@pytest.mark.parametrize(
    "declared, set_to", [("0", -61.0), ("10", None)], ids=["written", "crosses"]
)
def test_a_law_that_reads_the_negative_species_too_leaves_it_in_place(
    declared, set_to, codegen, sparse
):
    # The non-finite law reads V as well, so its row reads S and V both, and
    # V was read at 0 with S even once the clamp is narrowed to the rows. Each
    # slot the narrowed reading moves from outside its band is now put back
    # where the reading stays finite without it: V is, S (below 0 by far more
    # than its atol, and NaN under sqrt there) is not. Main: V(40) = -328.63
    # and -270.50, as for the law that does not read V.
    model = bngsim.Model.from_antimony_string(
        (_V_BESIDE_SINK % declared).replace("k*sqrt(S);", "k*sqrt(S)*(1 + 0*V);")
    )
    if set_to is not None:
        model.set_concentration("V", set_to)
    r = _run_v(model, codegen, sparse)
    _check_v(r, set_to if set_to is not None else float(declared))
    _check_s(r)


@pytest.mark.parametrize("order", ["S, V", "V, S"])
def test_a_slot_the_nan_law_needs_moved_stays_moved(order):
    # S written to -1e-3, far past any rounding of 0: k*sqrt(S) is NaN there,
    # so S, tried back at its value, keeps the reading NaN and is read at 0
    # again (dS/dt = 0, S stays), while V is put back. In the order "V, S" the
    # last slot tried is S, so the reading is taken once more at the state kept.
    model = bngsim.Model.from_antimony_string(
        f"model m\n  species {order}; S = 1; V = 0;\n  k = 1; g = 0.1; E = -70;\n"
        "  R1: S -> ; k*sqrt(S)*(1 + 0*V);\n  R2: -> V; g*(E - V);\nend"
    )
    model.set_concentration("S", -1e-3)
    model.set_concentration("V", -61.0)
    r = _run_v(model, False, False)
    _check_v(r, -61.0)
    np.testing.assert_array_equal(_col(r, "S"), -1e-3)


@pytest.mark.parametrize("codegen", [False, True])
@pytest.mark.parametrize(
    "law", ["k*A*P^1.5", "k*A*f", "k*A*h"], ids=["direct", "one function", "two functions"]
)
def test_the_rows_a_function_reads_through_count_as_read(law, codegen):
    """The NaN is in f() = P^1.5, which J reads directly, through one function,
    or through two (h = 2*f2, f2 = 0.5*f). The slots J reads come from the
    Jacobian's pattern, which follows the functions: P is clamped, V is not, and
    V relaxes to E from -61, V(40) = -70 + 9 e^-4. Main read V at 0 too: -341."""
    pytest.importorskip("antimony")
    src = f"""model m
  species A = 1, P = 0, V = 0;
  k = 0.1; c = 1e-30; g = 0.1; E = -70;
  f := P^1.5; h := 2*f2; f2 := 0.5*f;
  P' = -c;
  J: A -> ; {law};
  R2: -> V; g*(E - V);
end"""
    model = bngsim.Model.from_antimony_string(src)
    model.set_concentration("V", -61.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
            t_span=(0, 40), n_points=5, atol=bngsim.ExactAtol(1e-9), rtol=1e-8
        )
    v = dict(zip(result.species_names, np.asarray(result.species)[-1], strict=True))["V"]
    assert v == pytest.approx(-70 + 9 * math.exp(-4), abs=1e-4)
