"""bngsim.ModelBuilder builds a model in Python that runs like a loaded one (#1706)."""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


def _run(model, t_end=1.0, **kw):
    return bngsim.Simulator(model, **kw).run(t_span=(0, t_end), n_points=2, rtol=1e-10, atol=1e-10)


def _decay(**reaction):
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)
    b.add_species("A", 100)
    b.add_reaction(["A"], [], **reaction)
    return b.build()


@pytest.mark.parametrize(
    ("reaction", "rate"),
    [
        ({"rate": 0.5}, 0.5),  # a number
        ({"rate": "k"}, 0.5),  # a parameter: mass action
        ({"rate": "2*k"}, 1.0),  # an expression: mass action
        ({"law": "k*A"}, 0.5),  # the whole rate
    ],
)
def test_rate_and_law(reaction, rate):
    a = _run(_decay(**reaction)).observables["A"][-1]
    assert a == pytest.approx(100 * np.exp(-rate), rel=1e-7)


def test_a_rate_can_name_a_function():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.25)
    b.add_species("A", 100)
    b.add_function("twice_k", "2*k")
    b.add_reaction(["A"], [], rate="twice_k")
    assert _run(b.build()).observables["A"][-1] == pytest.approx(100 * np.exp(-0.5), rel=1e-7)


def test_a_rate_can_name_an_observable_or_a_species():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)
    b.add_species("A", 10)
    b.add_species("E", 0.1, fixed=True)
    b.add_observable("enzyme", ["E"])
    b.add_reaction(["A"], [], rate="enzyme")  # rate constant 0.1, read from E
    b.add_reaction(["A"], [], rate="E")
    assert _run(b.build()).observables["A"][-1] == pytest.approx(10 * np.exp(-0.2), rel=1e-7)


def test_a_users_name_never_collides_with_a_hidden_one():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)
    b.add_species("A", 100)
    b.add_function("_rate1", "1000*k")  # the builder's own first name
    b.add_reaction(["A"], [], rate="2*k")
    assert _run(b.build()).observables["A"][-1] == pytest.approx(100 * np.exp(-1), rel=1e-7)


def test_a_rate_naming_nothing_says_so():
    b = bngsim.ModelBuilder()
    b.add_species("A", 1)
    b.add_reaction(["A"], [], rate="undefined_k")
    with pytest.raises(ValueError, match="'undefined_k', which is not defined"):
        b.build()


def test_a_reactant_listed_twice_counts_twice():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.01)
    b.add_species("A", 100)
    b.add_reaction(["A", "A"], [], rate="k")  # rate k*A*A, two A per firing
    a = _run(b.build()).observables["A"][-1]
    assert a == pytest.approx(100 / (1 + 2 * 0.01 * 100), rel=1e-7)


def test_a_fixed_species_is_read_and_not_changed():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)
    b.add_species("S", 10, fixed=True)
    b.add_species("P", 0)
    b.add_reaction(["S"], ["S", "P"], rate="k")
    r = _run(b.build(), t_end=2.0)
    assert r.observables["S"][-1] == 10
    assert r.observables["P"][-1] == pytest.approx(10.0)


def test_observables_and_events():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)
    b.add_species("A", 100)
    b.add_species("B", 0)
    b.add_observable("total", ["A", "B"])
    b.add_observable("weighted", {"A": 1, "B": 2})
    b.add_reaction(["A"], ["B"], rate="k")
    b.add_event("time() >= 1", {"A": "A + 100"}, name="dose")
    r = _run(b.build(), t_end=2.0)
    assert r.observables["total"][-1] == pytest.approx(200, rel=1e-7)
    a = (100 * np.exp(-0.5) + 100) * np.exp(-0.5)
    assert r.observables["A"][-1] == pytest.approx(a, rel=1e-6)
    assert r.observables["weighted"][-1] == pytest.approx(a + 2 * (200 - a), rel=1e-6)


def test_a_built_model_runs_every_method():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)
    b.add_species("A", 100)
    b.add_reaction(["A"], [], rate="k")
    model = b.build()
    assert isinstance(model, bngsim.Model)
    ssa = bngsim.Simulator(model, method="ssa").run(t_span=(0, 1), n_points=2, seed=1)
    assert ssa.observables["A"][-1] < 100
    model.reset()
    sens = bngsim.Simulator(model, sensitivity_params=["k"]).run(t_span=(0, 1), n_points=2)
    assert sens.sensitivities[-1, 0, 0] == pytest.approx(-100 * np.exp(-0.5), rel=1e-5)


def test_mistakes_are_named():
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 1)
    b.add_species("A", 1)
    with pytest.raises(ValueError, match="no species named 'Z'"):
        b.add_reaction(["Z"], [], rate="k")
    with pytest.raises(ValueError, match="already defined"):
        b.add_species("k", 1)
    with pytest.raises(ValueError, match="is not a name"):
        b.add_species("A()", 1)
    with pytest.raises(TypeError, match="list of species names"):
        b.add_reaction("A", [], rate="k")
    with pytest.raises(ValueError, match="rate= or law="):
        b.add_reaction(["A"], [], rate="k", law="k")
    b.build()
    with pytest.raises(RuntimeError, match="built its model"):
        b.add_parameter("k2", 1)
