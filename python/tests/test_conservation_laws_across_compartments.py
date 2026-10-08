"""Conservation laws of a model whose reactions span compartments of
different size (issue #758).

A reaction between species in compartments of different size has a rate in
amount per time, and each species' stored concentration moves by that rate
over its own compartment's volume. The detector row-reduced the plain
stoichiometry, so for ``A`` (V = 1) ``<->`` ``B`` (V = 2) it reported ``A + B``
where what the dynamics keep is ``A + 2·B``. Every consumer held the wrong
total, with nothing raised or logged:

- ``Model.conservation_laws`` gave ``[1, 1]``, and ``L·y`` went from 1 to 0.75
  over a run;
- ``steady_state(method="newton")`` reported the eigenvalue -1.5 for -2, and
  with three species a root about 1e-7 off;
- ``steady_state(sensitivity_params=...)`` was 33% off (refused since #704).

The laws are found for the amounts now and carry each species' volume, and a
model keeps its own where a compartment size can be written.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

TWO = (
    "compartment c1, c2; c1 = 1; c2 = {v2}; species A in c1, B in c2; A = 1; B = 0;\n"
    "k1 = 1; k2 = 1;\nR1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
)
# A in the cytoplasm (V = 1); B and C in a nucleus of size V2. The law is
# A + V2*(B + C).
THREE = (
    "compartment cyt, nuc; cyt = 1; nuc = {v2}; species A in cyt, B in nuc, C in nuc;\n"
    "A = 1; B = 0; C = 0; k1 = 1; k2 = 0.5; k3 = 2; k4 = 1;\n"
    "R1: A -> B; k1*A*cyt\nR2: B -> A; k2*B*nuc\nR3: B -> C; nuc*(k3*B - k4*C)\n"
)


def _two(v2: float = 2.0) -> bngsim.Model:
    return bngsim.Model.from_antimony_string(TWO.format(v2=v2))


def _law(model: bngsim.Model) -> np.ndarray:
    laws = model.conservation_laws
    assert laws["n_laws"] == 1
    row = np.asarray(laws["coefficients"][0], dtype=float)
    return row / row[list(model.species_names).index("A")]


def _kept(model: bngsim.Model) -> float:
    """The largest rate at which a reported law's total changes, over three
    states, beside the size of the terms that should cancel."""
    coefficients = np.asarray(model.conservation_laws["coefficients"], dtype=float)
    worst = 0.0
    for seed in (1, 2, 3):
        state = np.random.default_rng(seed).uniform(0.2, 3.0, model.n_species)
        rate = np.asarray(model.rhs(state))
        size = np.abs(coefficients) @ np.abs(rate)
        worst = max(worst, float(np.max(np.abs(coefficients @ rate) / np.maximum(size, 1e-300))))
    return worst


@pytest.mark.parametrize("v2", [2.0, 0.25, 1e-3])
def test_the_law_carries_the_volumes(v2):
    """``A + V2·B``, and the right-hand side keeps it at any state. It was
    ``A + B``, which the right-hand side changes at a third of its terms."""
    model = _two(v2)
    np.testing.assert_allclose(_law(model), [1.0, v2], rtol=1e-12)
    assert _kept(model) < 1e-12


def test_the_total_is_kept_over_a_run():
    """``L·y`` along a trajectory: 1 at every sample. It fell to 0.75."""
    model = _two()
    row = _law(model)
    out = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 20.0), n_points=6, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(out.species) @ row, 1.0, rtol=1e-8)


def test_a_newton_root_and_its_eigenvalue():
    """A* = k2/(k1 + k2) and B* = k1/(V2·(k1 + k2)); the one eigenvalue of the
    reduced system is -(k1 + k2) = -2. It was -1.5."""
    out = bngsim.Simulator(_two(), method="ode").steady_state(method="newton")
    np.testing.assert_allclose(np.asarray(out.concentrations), [0.5, 0.25], rtol=1e-9)
    np.testing.assert_allclose(np.real(np.asarray(out.eigenvalues)), [-2.0], rtol=1e-8)


def test_the_steady_state_sensitivities():
    """dA*/dk1 = -0.25, dB*/dk1 = 0.125, dA*/dk2 = 0.25, dB*/dk2 = -0.125. They
    came back -0.1667 and 0.1667, and were refused from #704 on."""
    out = bngsim.Simulator(_two(), method="ode").steady_state(
        sensitivity_params=["k1", "k2"], tol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out.sensitivity).reshape(2, 2), [[-0.25, 0.25], [0.125, -0.125]], rtol=1e-7
    )


def test_a_size_that_is_written_moves_the_law():
    """A compartment size is a parameter (#170). After ``set_param("c2", 4)``
    the law is ``A + 4·B``, for the model and for a clone of it, and the
    steady state is B* = 0.125 with dB*/dk1 = 0.0625."""
    model = _two()
    np.testing.assert_allclose(_law(model), [1.0, 2.0])
    model.set_param("c2", 4.0)
    model.reset()
    np.testing.assert_allclose(_law(model), [1.0, 4.0])
    np.testing.assert_allclose(_law(model.clone()), [1.0, 4.0])
    assert _kept(model) < 1e-12
    sim = bngsim.Simulator(model, method="ode")
    out = sim.steady_state(sensitivity_params=["k1"], tol=1e-12)
    np.testing.assert_allclose(np.asarray(out.concentrations), [0.5, 0.125], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [-0.25, 0.0625], rtol=1e-7)
    np.testing.assert_allclose(_law(_two()), [1.0, 2.0])  # another model keeps its own


def test_a_batch_row_that_writes_a_size_holds_its_own_law():
    """Each row of ``steady_state_batch`` is solved on a clone with the row's
    size: B* = 1/(2·V2)."""
    sim = bngsim.Simulator(_two(), method="ode")
    rows = sim.steady_state_batch(params=[{"c2": 4.0}, {"c2": 2.0}, {"c2": 8.0}], method="newton")
    got = [np.asarray(row.concentrations)[1] for row in rows]
    np.testing.assert_allclose(got, [0.125, 0.25, 0.0625], rtol=1e-8)


def _settled(text: str, name: str) -> np.ndarray:
    model = bngsim.Model.from_antimony_string(text)
    out = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 400.0), n_points=2, rtol=1e-12, atol=1e-14
    )
    return np.asarray(out.species)[-1]


@pytest.mark.parametrize("v2", [4.0, 50.0])
@pytest.mark.parametrize("param,value", [("k1", 1), ("k3", 2)])
def test_three_species_against_runs_to_the_steady_state(v2, param, value):
    """The law is ``A + V2·(B + C)``. The columns are checked against central
    differences of plain runs to t = 400, which read no conservation law. At
    V2 = 50, dA*/dk1 came back -0.0151 for -0.1453."""
    text = THREE.format(v2=v2)
    model = bngsim.Model.from_antimony_string(text)
    np.testing.assert_allclose(_law(model), [1.0, v2, v2], rtol=1e-12)
    out = bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=[param], tol=1e-13)
    h = 1e-4 * value
    moved = [
        _settled(text.replace(f"{param} = {value};", f"{param} = {value + s * h!r};"), param)
        for s in (1, -1)
    ]
    want = (moved[0] - moved[1]) / (2 * h)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], want, rtol=2e-5, atol=1e-9)
    assert np.max(np.abs(want)) > 1e-3


def test_amounts_across_compartments():
    """Species held as amounts: stored as the amount over the size at load,
    so the law has the sizes in it there too."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; substanceOnly species A in c1, B in c2;\n"
        "A = 1; B = 0; k1 = 1; k2 = 1;\nR1: A -> B; k1*A\nR2: B -> A; k2*B\n"
    )
    assert model.conservation_laws["n_laws"] == 1
    assert _kept(model) < 1e-12


def test_compartments_of_one_size_are_as_they_were():
    """Control. With both at size 1 the law is ``A + B``, as the plain
    stoichiometry gives it."""
    model = _two(1.0)
    np.testing.assert_allclose(_law(model), [1.0, 1.0], rtol=0, atol=0)
    assert _kept(model) < 1e-12


def test_a_net_model_is_as_it_was(tmp_path):
    """Control. No volumes: the laws of the plain stoichiometry, to the bit."""
    path = tmp_path / "iso.net"
    path.write_text(
        "begin parameters\n    1 kf 1\n    2 kr 0.5\nend parameters\n"
        "begin species\n    1 A() 3\n    2 B() 0\n    3 E() 1\nend species\n"
        "begin reactions\n    1 1,3 2,3 kf\n    2 2 1 kr\nend reactions\n"
    )
    laws = bngsim.Model.from_net(str(path)).conservation_laws
    assert laws["n_laws"] == 2
    assert sorted(map(tuple, laws["coefficients"])) == [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]


def test_a_law_the_right_hand_side_does_not_keep_is_refused():
    """The solvers hold each reported total. Before a solve the right-hand
    side is asked whether it keeps them, at two states off the model's own,
    and a law it does not keep is an error. A size below zero is one way to
    such a law: the detector takes a size that is not positive for 1, and the
    right-hand side divides by it. The solve came back A* = 0.25, B* = -0.125."""
    model = _two()
    assert model._core.conservation_law_drift() == (-1, 0.0, 0.0)
    model.set_param("c2", -2.0)
    law, drift, size = model._core.conservation_law_drift()
    assert law == 0 and drift > 0.1 * size > 0.0
    sim = bngsim.Simulator(model, method="ode")
    for method in ("newton", "integration"):
        with pytest.raises(
            bngsim.SimulationError, match=r"over A, B.*not kept\s+by its own.*#758"
        ):
            sim.steady_state(method=method)
    with pytest.raises(bngsim.SimulationError, match=r"steady_state_batch\(\) entry 0 .*#758"):
        sim.steady_state_batch(params=[{"k1": 2.0}])


def test_a_batch_entry_is_asked_of_its_own_model():
    """An entry of ``steady_state_batch`` is solved on a clone that has the
    entry's parameters, sizes among them, so the question is put to the
    clone: a size below zero in the second entry is refused there, and a
    model at such a size is solved where every entry writes one above."""
    sim = bngsim.Simulator(_two(), method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"entry 1 is not supported.*not kept\s+by its own"
    ):
        sim.steady_state_batch(params=[{"c2": 4.0}, {"c2": -2.0}])
    with pytest.raises(bngsim.SimulationError, match=r"Batch 1 failed.*entry 1 is not supported"):
        sim.steady_state_batch(params=[{"c2": 4.0}, {"c2": -2.0}], n_workers=2)
    model = _two()
    model.set_param("c2", -2.0)
    rows = bngsim.Simulator(model, method="ode").steady_state_batch(params=[{"c2": 4.0}])
    np.testing.assert_allclose(np.asarray(rows[0].concentrations), [0.5, 0.125], rtol=1e-8)


def test_the_question_is_asked_off_a_steady_state():
    """At a steady state every rate is rounding, and the rounding of two rates
    need not cancel. The law is asked at states off the model's own, so a
    model left at its steady state is solved again: here after a run to
    t = 200, where the rates are 1e-17."""
    text = THREE.format(v2=4.0)
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 200.0), n_points=2, rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(model.rhs(model.get_state()))) < 1e-12
    assert model._core.conservation_law_drift()[0] == -1
    out = sim.steady_state(method="newton")
    np.testing.assert_allclose(np.asarray(out.concentrations), _settled(text, "k1"), rtol=1e-8)
