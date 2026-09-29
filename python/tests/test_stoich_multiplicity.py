"""Issue #801: a stoichiometric coefficient is one right-hand-side update, not one per unit.

A reaction's reactants and products are index lists with one entry per unit of
stoichiometry, so an SBML ``stoichiometry="1000000"`` is a million entries. The C
emitter wrote one ``ydot[i] += rate;`` per entry (66.7 MB of C for
BIOMD0000000608, which then could not compile), and the C++ right-hand side and
the JAX translator looped the same way. All three now apply ``m * rate`` once per
species; a multiplicity of 1 keeps the statement it had.
"""

from __future__ import annotations

import importlib.util

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _multiplicity, _times_multiplicity, prepare_model_codegen_source


def _sbml(coefficient: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>
<species id="A" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters><parameter id="k" value="0.1" constant="true"/></listOfParameters>
<listOfReactions><reaction id="R" reversible="false">
<listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/>
</listOfReactants>
<listOfProducts><speciesReference species="B" stoichiometry="{coefficient}" constant="true"/>
</listOfProducts>
<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
<apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
</reaction></listOfReactions>
</model></sbml>"""


_NET = """begin parameters
    1 k 0.1
end parameters
begin species
    1 A() 1
    2 B() 0
end species
begin reactions
    1 {reaction} k
end reactions
begin groups
    1 Btot 2
end groups
"""


def _net_model(tmp_path, reaction: str) -> bngsim.Model:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(reaction=reaction), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def test_the_helpers_fold_in_order_of_first_appearance():
    assert _multiplicity([2, 0, 2, -1, 2]) == [(2, 3), (0, 1)]
    assert _times_multiplicity(1, "rate") == "rate"
    assert _times_multiplicity(3, "rate * inv_vf[1]") == "3.0 * (rate * inv_vf[1])"


def test_a_million_coefficient_is_one_update():
    src = prepare_model_codegen_source(bngsim.Model.from_sbml_string(_sbml("1000000")))
    assert src.count("ydot[1] ") == 1
    assert "1000000.0 * (rate)" in src
    assert len(src) < 100_000  # was 21 MB


def test_the_interpreter_applies_the_coefficient_once():
    # 1e6 repeated additions of 0.1 drifted to 100000.00000133288.
    m = bngsim.Model.from_sbml_string(_sbml("1000000"))
    assert m.rhs(np.array([1.0, 0.0])).tolist() == [-0.1, 100000.0]


@pytest.mark.parametrize("reaction", ["1 2", "1 2,2,2", "1 2,2,2,2,2,2,2"])
def test_repeated_products_agree_exactly_between_interpreter_and_compiled(tmp_path, reaction):
    # The rate is the same arithmetic on both paths here (one reactant), so the
    # m * rate update is what is compared, and it is exact.
    runs = []
    for codegen in (False, True):
        sim = bngsim.Simulator(
            _net_model(tmp_path, reaction), method="ode", codegen=codegen, jacobian="fd"
        )
        if codegen:
            assert sim.codegen_backend in ("cc", "mir")
        runs.append(np.asarray(sim.run(t_span=(0, 5), n_points=6).species))
    assert np.array_equal(runs[0], runs[1])


def test_repeated_reactants_and_products_integrate_to_the_closed_form(tmp_path):
    # 2A -> 3B at k: A' = -2 k A^2, so A = 1/(1 + 2 k t) and B = 1.5 (1 - A).
    for codegen in (False, True):
        sim = bngsim.Simulator(_net_model(tmp_path, "1,1 2,2,2"), method="ode", codegen=codegen)
        r = sim.run(t_span=(0, 5), n_points=6, rtol=1e-10, atol=1e-12)
        a = 1.0 / (1.0 + 2 * 0.1 * np.asarray(r.time))
        np.testing.assert_allclose(np.asarray(r.species)[:, 0], a, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(r.species)[:, 1], 1.5 * (1 - a), rtol=1e-8)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
def test_the_jax_rhs_applies_the_coefficient_once(tmp_path):
    import jax.numpy as jnp
    from bngsim._jax_rhs import generate_jax_rhs

    m = _net_model(tmp_path, "1,1 2,2,2")
    rhs = generate_jax_rhs(m)
    y = jnp.array([1.0, 0.0], dtype=jnp.float64)
    got = np.asarray(rhs(y, 0.0, jnp.array([0.1], dtype=jnp.float64)))
    np.testing.assert_allclose(got, m.rhs(np.array([1.0, 0.0])), rtol=1e-15)


def test_a_side_wider_than_the_linear_scan_folds_the_same():
    # 20 distinct products, each three times and interleaved, so the fold
    # passes its linear-scan limit and keeps meeting repeats afterwards.
    from bngsim._bngsim_core import ModelBuilder
    from bngsim._model import Model

    def model() -> Model:
        b = ModelBuilder()
        b.add_parameter("k", 0.1, "0.1", False)
        b.add_species("A()", 1.0, False)
        for i in range(20):
            b.add_species(f"P{i}()", 0.0, False)
        b.add_observable("Atot", [(0, 1.0)])
        b.add_reaction([0], [1 + i for _ in range(3) for i in range(20)], "elementary", "k", 1.0)
        return Model(_core=b.build())

    assert model().rhs(np.array([1.0] + [0.0] * 20)).tolist() == [-0.1] + [3.0 * 0.1] * 20
    runs = []
    for codegen in (False, True):
        sim = bngsim.Simulator(model(), method="ode", codegen=codegen, jacobian="fd")
        runs.append(np.asarray(sim.run(t_span=(0, 5), n_points=6).species))
    assert np.array_equal(runs[0], runs[1])
