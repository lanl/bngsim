"""A species on both sides of a reaction has nothing of the rate in its derivative.

The right-hand side took a reaction's rate off each reactant and then put it on
each product. A species written on both sides (a catalyst, ``E + S -> E + P``)
got ``(x - rate) + rate``, which is the rounding of the rate and not nothing:
``0 -> E`` at 0.01 beside a mass-action rate of 1e15 gave dE/dt = 0, and E did
not move. The Jacobian, the stoichiometry matrix and the conservation laws were
built from the net change of each species all along. The interpreted
right-hand side, the generated one and the JAX translator now add each species'
net change times the rate, once.

A net change of zero is ``0 * rate``, not no statement: a rate law that is NaN
or infinite reached the derivative of every free species its reaction names,
and failed the run there; it still does.

Oracles: the derivative in closed form, exactly (a species with no net change
has none of a finite rate, whatever the rate); the same reaction with the
catalyst written on neither side; ``E(t) = E0 + ksyn*t``; main, for which runs
are refused.
"""

from __future__ import annotations

import importlib.util

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder
from bngsim._codegen import prepare_model_codegen_source
from bngsim._model import Model

KSYN = 0.01


def _catalysed(k: float = 1e-3, e0: float = 1e6, s0: float = 1e12, catalyst: bool = True) -> Model:
    """``0 -> E`` at ksyn and ``E + S -> E + P`` at k (rate k*E*S: 1e15 at the
    defaults), or with *catalyst* false ``S -> P`` at k*e0."""
    b = ModelBuilder()
    e = b.add_species("E()", e0)
    s = b.add_species("S()", s0)
    p = b.add_species("P()", 0.0)
    b.add_parameter("ksyn", KSYN, repr(KSYN), False)
    b.add_parameter("k", k, repr(k), False)
    b.add_parameter("kE", k * e0, repr(k * e0), False)
    b.add_observable("Etot", [(e, 1.0)])
    b.add_reaction([], [e], "elementary", "ksyn", 1.0)
    if catalyst:
        b.add_reaction([e, s], [e, p], "elementary", "k", 1.0)
    else:
        b.add_reaction([s], [p], "elementary", "kE", 1.0)
    return Model(_core=b.build())


def _state(m: Model) -> np.ndarray:
    return np.asarray(m._core.get_state(), dtype=float)


def test_the_helper_folds_two_sides_to_the_net_change():
    from bngsim._codegen import _net_multiplicity

    # what a firing takes (a net of zero with it), then what it gives
    assert _net_multiplicity([0, 1], [0, 2]) == [(0, 0), (1, -1), (2, 1)]
    assert _net_multiplicity([0, 0, 1], [0, 2, 2]) == [(0, -1), (1, -1), (2, 2)]
    assert _net_multiplicity([0], [0, 0]) == [(0, 1)]
    assert _net_multiplicity([0, 0], [0]) == [(0, -1)]
    assert _net_multiplicity([1, 0], [0, 0, 2]) == [(1, -1), (0, 1), (2, 1)]
    assert _net_multiplicity([3, -1], [4]) == [(3, -1), (4, 1)]
    assert _net_multiplicity([0], [0]) == [(0, 0)]
    assert _net_multiplicity([], []) == []


def test_a_catalysts_derivative_is_the_synthesis_alone_beside_a_rate_of_1e15():
    m = _catalysed()
    f = m.rhs(_state(m))
    assert f[0] == KSYN  # was 0.0: (0.01 - 1e15) + 1e15
    assert f[1] == -1e15 and f[2] == 1e15


@pytest.mark.parametrize("codegen", [False, True])
def test_a_catalyst_moves_by_its_synthesis_alone(codegen):
    # E(t) = 1e6 + 0.01 t. On main E did not move at all.
    sim = bngsim.Simulator(_catalysed(), method="ode", codegen=codegen)
    if codegen:
        assert sim.codegen_backend in ("cc", "mir")
    r = sim.run(sample_times=[0.0, 1e-4, 1e-3])
    e = np.asarray(r.species)[:, 0]
    assert e[-1] - e[0] == pytest.approx(KSYN * 1e-3, rel=1e-3)
    # ...and as with the catalyst written on neither side, to the rounding of
    # E's own steps. (Not to the bit on every machine: S is a last place apart
    # in the two models, since k*E*S follows E and kE*S does not, so the two
    # runs need not take the same steps, and E moves by 1e-7 a step in doubles
    # 1.2e-10 apart.)
    control = bngsim.Simulator(_catalysed(catalyst=False), method="ode", codegen=codegen)
    e_control = np.asarray(control.run(sample_times=[0.0, 1e-4, 1e-3]).species)[:, 0]
    np.testing.assert_allclose(e, e_control, rtol=0.0, atol=1e-8)


def test_the_generated_right_hand_side_takes_zero_times_the_rate_off_a_catalyst():
    src = prepare_model_codegen_source(_catalysed())
    # E: the synthesis, and zero times the catalysed rate. S and P as before.
    assert src.count("ydot[0] +=") == 1
    assert src.count("ydot[0] -=") == 1 and "ydot[0] -= 0.0 * (rate);" in src
    assert src.count("ydot[1] -= rate;") == 1 and src.count("ydot[2] += rate;") == 1


def test_a_rate_that_is_not_finite_still_reaches_a_catalysts_derivative():
    # k*E*S overflows: the rate is infinite, and dE/dt is NaN as it was.
    m = _catalysed(k=1e300)
    f = m.rhs(_state(m))
    assert np.isnan(f[0])
    assert f[1] == -np.inf and f[2] == np.inf


def _changes_no_free_species(which: str) -> Model:
    """``0 -> E`` at ksyn beside a reaction that changes no free species, at a
    rate that is not a finite number."""
    b = ModelBuilder()
    e = b.add_species("E()", 2.0)
    s = b.add_species("S()", 5.0, which == "fixed")
    p = b.add_species("P()", 0.0, which == "fixed")
    b.add_parameter("ksyn", KSYN, repr(KSYN), False)
    b.add_parameter("k", float("nan"), "nan", False)
    b.add_parameter("kinf", float("inf"), "inf", False)
    b.add_parameter("zero", 0.0, "0", False)
    b.add_observable("Etot", [(e, 1.0)])
    b.add_observable("Stot", [(s, 1.0)])
    b.add_function("out_of_domain", "sqrt(-1 - Etot)")
    b.add_function("over_zero", "Stot/(zero*Etot)")
    b.add_reaction([], [e], "elementary", "ksyn", 1.0)
    if which == "fixed":  # E + $S -> E + $P
        b.add_reaction([e, s], [e, p], "elementary", "k", 1.0)
    elif which == "itself":  # E -> E
        b.add_reaction([e], [e], "elementary", "kinf", 1.0)
    elif which == "law":  # E + S -> E + S at sqrt of a negative number
        b.add_reaction([e, s], [e, s], "functional", "out_of_domain", 1.0)
    else:  # E -> E at a division by zero
        b.add_reaction([e], [e], "functional", "over_zero", 1.0)
    return Model(_core=b.build())


@pytest.mark.parametrize("codegen", [False, True])
@pytest.mark.parametrize("which", ["fixed", "itself", "law", "divide"])
def test_a_rate_that_is_not_finite_fails_the_run_where_its_reaction_changes_nothing(
    which, codegen
):
    """The one check an ODE run has of a rate law outside its domain is that the
    derivatives are finite. Such a rate reached a catalyst's derivative as
    ``(x - rate) + rate``; with no statement at all for the catalyst it reached
    none, and the run returned the model without the reaction (E(10) = 2.1)
    while a stochastic run of the same model was refused."""
    m = _changes_no_free_species(which)
    assert np.isnan(m.rhs(_state(m))[0])
    sim = bngsim.Simulator(_changes_no_free_species(which), method="ode", codegen=codegen)
    with pytest.raises(bngsim.SimulationError, match="non-finite"):
        sim.run(t_span=(0, 10), n_points=3)


def _one_reaction(reactants, products, y0, k=0.5, volumes=None, psvs=False) -> Model:
    b = ModelBuilder()
    for i, y in enumerate(y0):
        b.add_species(f"X{i}()", y, False, 1.0 if volumes is None else volumes[i])
    b.add_parameter("k", k, repr(k), False)
    b.add_observable("X0tot", [(0, 1.0)])
    b.add_reaction(reactants, products, "elementary", "k", 1.0, True, 1.0, psvs)
    return Model(_core=b.build())


# (reactants, products, y0, the derivative in closed form with rate = k * prod(reactants))
_UNEQUAL = [
    # A + B -> A + A: A gains one, B loses one
    ([0, 1], [0, 0], [3.0, 5.0], lambda r: [r, -r]),
    # A -> A + A
    ([0], [0, 0], [3.0], lambda r: [r]),
    # A + A -> A
    ([0, 0], [0], [3.0], lambda r: [-r]),
    # 2 E + S -> 2 E + P
    ([0, 0, 1], [0, 0, 2], [3.0, 5.0, 0.0], lambda r: [0.0, -r, r]),
    # 2 A + B -> 3 A + 2 C: A gains one, B loses one, C gains two
    ([0, 0, 1], [0, 0, 0, 2, 2], [3.0, 5.0, 0.0], lambda r: [r, -r, 2.0 * r]),
    # 3 A -> A: A loses two
    ([0, 0, 0], [0], [3.0], lambda r: [-2.0 * r]),
]


@pytest.mark.parametrize("reactants, products, y0, closed", _UNEQUAL, ids=range(len(_UNEQUAL)))
def test_a_species_on_both_sides_gets_its_net_change(reactants, products, y0, closed):
    m = _one_reaction(reactants, products, y0)
    rate = 0.5 * float(np.prod([y0[i] for i in reactants]))  # exact: halves and small integers
    f = m.rhs(np.asarray(y0, dtype=float))
    assert f.tolist() == closed(rate)
    # ...and the generated right-hand side integrates the same.
    runs = []
    for codegen in (False, True):
        sim = bngsim.Simulator(
            _one_reaction(reactants, products, y0), method="ode", codegen=codegen, jacobian="fd"
        )
        runs.append(np.asarray(sim.run(t_span=(0, 0.05), n_points=4).species))
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-12, atol=1e-300)


def test_a_species_on_one_side_keeps_the_statement_it_had():
    # S -> P + P + P: the text the emitter wrote before, to the character.
    src = prepare_model_codegen_source(_one_reaction([0], [1, 1, 1], [1.0, 0.0]))
    assert "ydot[0] -= rate;" in src and "ydot[1] += 3.0 * (rate);" in src


def test_a_catalyst_across_compartments_has_nothing_of_the_rate_either():
    # The cross-compartment accumulator divides the rate by each species'
    # volume: (x - rate/V) + rate/V for a species on both sides.
    m = _one_reaction([0, 1], [0, 2], [1e6, 1e12, 0.0], k=1e-3, volumes=[3.0, 3.0, 7.0], psvs=True)
    f = m.rhs(_state(m))
    assert f[0] == 0.0
    assert f[1] == -1e15 / 3.0 and f[2] == 1e15 / 7.0
    src = prepare_model_codegen_source(m)
    assert src.count("ydot[0] -= 0.0 * (") == 1 and "ydot[0] +=" not in src
    assert src.count("ydot[1] -=") == 1 and src.count("ydot[2] +=") == 1


def test_a_conservation_law_is_kept_beside_a_catalyst_at_any_rate():
    # S + P is conserved; what the right-hand side does to it is read from the
    # same per-reaction terms the derivative is summed from.
    m = _catalysed()
    law, drift, size = m._core.conservation_law_drift()
    assert law == -1, (law, drift, size)


def test_a_stochastic_run_fires_by_the_two_sides_as_before():
    # SSA takes the reactants off and puts the products on, whole molecules:
    # nothing of this changes there. E + S -> E + P with no synthesis keeps E.
    b = ModelBuilder()
    e = b.add_species("E()", 5.0)
    s = b.add_species("S()", 200.0)
    p = b.add_species("P()", 0.0)
    b.add_parameter("k", 0.01, "0.01", False)
    b.add_observable("Etot", [(e, 1.0)])
    b.add_reaction([e, s], [e, p], "elementary", "k", 1.0)
    r = bngsim.Simulator(Model(_core=b.build()), method="ssa").run(
        t_span=(0, 50), n_points=11, seed=7
    )
    y = np.asarray(r.species)
    assert np.all(y[:, 0] == 5.0)
    assert np.all(y[:, 1] + y[:, 2] == 200.0)
    assert y[-1, 2] > 100.0


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
def test_the_jax_right_hand_side_takes_the_net_change_too():
    import jax.numpy as jnp
    from bngsim._jax_rhs import generate_jax_rhs

    m = _catalysed()
    rhs = generate_jax_rhs(m)
    y = jnp.array(_state(m), dtype=jnp.float64)
    p = jnp.array([KSYN, 1e-3, 1e3], dtype=jnp.float64)
    got = np.asarray(rhs(y, 0.0, p))
    assert got[0] == KSYN
    np.testing.assert_allclose(got, m.rhs(_state(m)), rtol=1e-15)
    # A rate that overflows is still NaN in the catalyst's derivative, traced
    # and compiled: zero times the rate is not folded away.
    import jax

    huge = jnp.array([KSYN, 1e300, 1e306], dtype=jnp.float64)
    assert np.isnan(np.asarray(rhs(y, 0.0, huge))[0])
    assert np.isnan(np.asarray(jax.jit(rhs)(y, 0.0, huge))[0])
