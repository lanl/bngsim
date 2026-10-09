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
        np.asarray(out.sensitivity).reshape(2, 2),
        [[-0.25, 0.25], [0.125, -0.125]],
        rtol=1e-7,
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


SHAPES = {
    # a chain through three sizes: A + 2*B + 5*C
    "chain": (
        "compartment c1, c2, c3; c1 = 1; c2 = 2; c3 = 5; species A in c1, B in c2, C in c3;\n"
        "A = 1; B = 0.5; C = 0.2; k1 = 1; k2 = 0.5; k3 = 0.3; k4 = 0.7;\n"
        "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\nR3: B -> C; k3*B*c2\nR4: C -> B; k4*C*c3\n",
        [[1.0, 2.0, 5.0]],
    ),
    # a law across compartments beside one inside a compartment
    "two laws": (
        "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2, E in c2, EB in c2;\n"
        "A = 1; B = 0.5; E = 1; EB = 0.3; k1 = 1; k2 = 0.5; k3 = 2; k4 = 1;\n"
        "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
        "R3: B + E -> EB; c2*k3*B*E\nR4: EB -> B + E; c2*k4*EB\n",
        [[1.0, 2.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0]],
    ),
    # A + B -> C, each in a compartment of its own: A + 5*C and B + 2.5*C
    "binding": (
        "compartment c1, c2, c3; c1 = 1; c2 = 2; c3 = 5; species A in c1, B in c2, C in c3;\n"
        "A = 1; B = 0.5; C = 0.2; k1 = 1; k2 = 0.5;\n"
        "R1: A + B -> C; k1*A*B\nR2: C -> A + B; k2*C\n",
        [[1.0, 0.0, 5.0], [0.0, 1.0, 2.5]],
    ),
    # 2 A -> B: half an A for a B in amounts, A + 6*B in what is stored
    "dimer": (
        "compartment c1, c2; c1 = 1; c2 = 3; species A in c1, B in c2; A = 1; B = 0.5;\n"
        "k1 = 1; k2 = 0.5;\nR1: 2 A -> B; k1*A*A*c1\nR2: B -> 2 A; k2*B*c2\n",
        [[1.0, 6.0]],
    ),
    # a boundary species is in no law
    "boundary": (
        "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2, $E in c2;\n"
        "A = 1; B = 0.5; E = 1; k1 = 1; k2 = 0.5;\n"
        "R1: A + E -> B + E; k1*A*E*c1\nR2: B -> A; k2*B*c2\n",
        [[1.0, 2.0, 0.0]],
    ),
    # held as amounts, and one of each kind
    "amounts": (
        "compartment c1, c2; c1 = 1; c2 = 2; substanceOnly species A in c1, B in c2;\n"
        "A = 1; B = 0.5; k1 = 1; k2 = 0.5;\nR1: A -> B; k1*A\nR2: B -> A; k2*B\n",
        [[1.0, 2.0]],
    ),
    "an amount and a concentration": (
        "compartment c1, c2; c1 = 1; c2 = 2; substanceOnly species A in c1; species B in c2;\n"
        "A = 1; B = 0.5; k1 = 1; k2 = 0.5;\nR1: A -> B; k1*A\nR2: B -> A; k2*B*c2\n",
        [[1.0, 2.0]],
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_laws_of_other_shapes(shape):
    """Each law over what is stored, with its dependent species at 1 and no
    other law's dependent in it, its constant the total at the initial state,
    and the right-hand side keeping it. Every one was the plain
    stoichiometry's, which the right-hand side changes at 20% to 67% of its
    terms."""
    text, want = SHAPES[shape]
    model = bngsim.Model.from_antimony_string(text)
    laws = model.conservation_laws
    rows = np.asarray(laws["coefficients"], dtype=float)
    assert laws["n_laws"] == len(want)
    for k, dependent in enumerate(laws["dependent"]):
        np.testing.assert_array_equal(rows[:, dependent], np.eye(len(want))[k])
    # the laws span what is expected, whichever species each is solved for
    np.testing.assert_allclose(
        rows @ np.linalg.pinv(np.asarray(want)) @ np.asarray(want), rows, atol=1e-12
    )
    np.testing.assert_allclose(laws["constants"], rows @ model.get_state(), rtol=1e-14)
    assert _kept(model) < 1e-12
    assert model._core.conservation_law_drift()[0] == -1


def test_a_species_in_a_resized_compartment_is_in_no_law():
    """An event resizes ``c2``, and B's share of a reaction across the
    compartments is divided by the size as it is then. No constant weight
    makes a law of A and B, so there is none: ``A + B`` was reported, which
    the right-hand side changes at 43% of its terms."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2; A = 1; B = 0.5;\n"
        "k1 = 1; k2 = 0.5;\nR1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
        "E1: at (time > 5): c2 = 4;\n"
    )
    names = list(model.species_names)
    rows = np.asarray(model.conservation_laws["coefficients"], dtype=float)
    assert np.all(rows[:, [names.index("A"), names.index("B")]] == 0.0)
    assert _kept(model) < 1e-12


def test_sizes_written_equal_before_the_laws_are_first_asked_for():
    """``c2`` is written to ``c1``'s size before anything asks for the laws,
    which are ``A + B`` there, and then to 4: whether a model's laws follow
    its sizes is decided by whether a write can move them, not by whether the
    sizes differ when the laws are first asked for."""
    model = _two()
    model.set_param("c2", 1.0)
    np.testing.assert_allclose(_law(model), [1.0, 1.0])
    model.set_param("c2", 4.0)
    np.testing.assert_allclose(_law(model), [1.0, 4.0])
    assert _kept(model) < 1e-12
    model.set_param("c1", 8.0)
    np.testing.assert_allclose(_law(model), [1.0, 0.5])
    assert _kept(model) < 1e-12


def test_sizes_equal_at_load_cannot_be_written():
    """Control. With both sizes equal at load the reaction is divided by one
    size for both species, and a write to either is refused, so ``A + B``
    there cannot go stale."""
    model = _two(1.0)
    with pytest.raises(ValueError, match="cannot resolve to a live volume"):
        model.set_param("c2", 4.0)
    np.testing.assert_allclose(_law(model), [1.0, 1.0])


def test_a_clone_made_before_the_write_keeps_its_law():
    """The laws are a model's own where a size can move them: a clone taken
    at ``c2 = 2`` reports ``A + 2*B`` after the original is written to 4."""
    model = _two()
    clone = model.clone()
    model.set_param("c2", 4.0)
    np.testing.assert_allclose(_law(model), [1.0, 4.0])
    np.testing.assert_allclose(_law(clone), [1.0, 2.0])


def _unscaled(volume_of_b: float):
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("kf", 0.3)
    b.add_parameter("kr", 0.2)
    a = b.add_species("A", 3.0, False, 1.0)
    bb = b.add_species("B", 1.0, False, volume_of_b)
    b.add_reaction([a], [bb], "elementary", "kf")
    b.add_reaction([bb], [a], "elementary", "kr")
    return b.build()


@pytest.mark.parametrize("volume_of_b", [1e-13, 1e-9, 5.0, 1e13])
def test_an_unscaled_reaction_between_volumes_keeps_the_plain_law(volume_of_b):
    """Control. A reaction built without the per-species divide moves each
    species by the same rate whatever its volume factor, so its law is the
    plain one, ``A + B``, with B's volume factor anywhere from 1e-13 to
    1e13. A model with no reaction that has the divide reads no volume."""
    core = _unscaled(volume_of_b)
    laws = core.conservation_laws
    assert laws["n_laws"] == 1
    assert laws["coefficients"] == [[1.0, 1.0]]
    rate = np.asarray(core.compute_derivs(0.0, np.array([2.0, 0.7])))
    assert rate[0] != 0.0 and rate[0] + rate[1] == 0.0


@pytest.mark.parametrize("volume_of_b", [1e-13, 1e-9, 5.0, 1e13])
def test_both_species_of_such_a_law_are_held(volume_of_b):
    """Which species a law holds is asked of the coefficients over the
    weights the laws were found with, which are 1 where no reaction has the
    divide: B is held at any volume factor, and the law is kept."""
    core = _unscaled(volume_of_b)
    assert core.conservation_law_members() == [[0, 1]]
    assert core.conservation_law_drift()[0] == -1


def test_a_built_model_with_both_kinds_of_reaction_between_volumes():
    """A (volume 1) <-> B (5) with the divide, beside B <-> C (2) without it:
    C moves at B's rate, so the law is ``A + 5*B + 5*C``, not ``5*B + 2*C``
    for the second pair."""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("kf", 0.3)
    b.add_parameter("kr", 0.2)
    a = b.add_species("A", 3.0, False, 1.0)
    bb = b.add_species("B", 1.0, False, 5.0)
    c = b.add_species("C", 0.5, False, 2.0)
    b.add_observable("A", [(a, 1.0)])
    b.add_observable("B", [(bb, 1.0)])
    b.add_function("fwd", "kf*A")
    b.add_function("bwd", "kr*B*5")
    kw = {"apply_species_factor": False, "per_species_volume_scaling": True}
    b.add_reaction([a], [bb], "functional", "fwd", **kw)
    b.add_reaction([bb], [a], "functional", "bwd", **kw)
    b.add_reaction([bb], [c], "elementary", "kf")
    b.add_reaction([c], [bb], "elementary", "kr")
    core = b.build()
    row = np.asarray(core.conservation_laws["coefficients"][0])
    np.testing.assert_allclose(row / row[0], [1.0, 5.0, 5.0], rtol=1e-14)
    assert core.conservation_law_members() == [[0, 1, 2]]
    assert core.conservation_law_drift()[0] == -1


def _built(laws: bool = True, live: bool = False):
    """A (size parameter V1 = 1) <-> B (V2 = 5) with the per-species divide,
    from the builder. ``live`` puts a one-compartment reaction beside it, B
    <-> C, with B's and C's shares of the first divided by a live volume."""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("kf", 0.3)
    b.add_parameter("kr", 0.2)
    v1 = b.add_parameter("V1", 1.0, "", False, True)
    v2 = b.add_parameter("V2", 5.0, "", False, True)
    a = b.add_species("A", 3.0, False, 1.0)
    bb = b.add_species("B", 1.0, False, 5.0)
    b.set_species_volume_param(a, v1)
    b.set_species_volume_param(bb, v2)
    b.add_observable("A", [(a, 1.0)])
    b.add_observable("B", [(bb, 1.0)])
    b.add_function("fwd", "kf*A")
    b.add_function("bwd", "kr*B*V2")
    kw = {"apply_species_factor": False, "per_species_volume_scaling": True}
    b.add_reaction([a], [bb], "functional", "fwd", **kw)
    b.add_reaction([bb], [a], "functional", "bwd", **kw)
    if live:
        c = b.add_species("C", 0.5, False, 5.0)
        size = b.add_species("V", 5.0, True, 1.0)
        b.add_reaction([bb], [c], "elementary", "kf")
        b.add_reaction([c], [bb], "elementary", "kr")
        b.set_species_ode_live_volume(bb, size)
    b.set_compute_conservation_laws(laws)
    return b.build()


def test_a_built_model():
    """The detector is the builder's, whatever loads the model: ``A + 5*B``,
    following ``V2`` when it is written."""
    core = _built()
    np.testing.assert_allclose(core.conservation_laws["coefficients"], [[1.0, 5.0]], rtol=1e-15)
    assert core.conservation_law_drift()[0] == -1
    core.set_param("V2", 8.0)
    np.testing.assert_allclose(core.conservation_laws["coefficients"], [[1.0, 8.0]], rtol=1e-15)
    assert core.conservation_law_drift()[0] == -1


def test_laws_switched_off_stay_off():
    """Control. ``set_compute_conservation_laws(False)`` gives a model no
    laws, and a model whose laws would follow its sizes has none either,
    before a size is written and after."""
    core = _built(laws=False)
    assert core.conservation_laws["n_laws"] == 0
    core.set_param("V2", 8.0)
    assert core.conservation_laws["n_laws"] == 0


def test_no_law_is_asked_where_laws_are_switched_off():
    core = _built(laws=False)
    assert core.conservation_law_members() == []
    assert core.conservation_law_drift() == (-1, 0.0, 0.0)


def test_a_live_volume_takes_only_the_species_it_divides_out_of_the_laws():
    """B's share of the reaction across the compartments is divided by a live
    volume, so B is in no law, and with it neither A nor C, which only B
    ties to anything. The one-compartment reaction beside it does not make a
    law of B and C on its own: B is changed by the other reaction too."""
    core = _built(live=True)
    laws = core.conservation_laws
    rows = np.asarray(laws["coefficients"], dtype=float).reshape(laws["n_laws"], 4)
    assert np.all(rows[:, :3] == 0.0)
    assert core.conservation_law_drift()[0] == -1


def test_a_live_volume_no_reaction_divides_by_takes_nothing_out():
    """Control. Only a reaction with the per-species divide reads a species'
    live volume. B has one and is in no such reaction, so ``A + B`` is the
    law, as it was."""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("kf", 0.3)
    b.add_parameter("kr", 0.2)
    a = b.add_species("A", 3.0, False, 1.0)
    bb = b.add_species("B", 1.0, False, 5.0)
    size = b.add_species("V", 5.0, True, 1.0)
    b.add_reaction([a], [bb], "elementary", "kf")
    b.add_reaction([bb], [a], "elementary", "kr")
    b.set_species_ode_live_volume(bb, size)
    core = b.build()
    laws = core.conservation_laws
    assert [1.0, 1.0, 0.0] in np.asarray(laws["coefficients"], dtype=float).tolist()
    rate = np.asarray(core.compute_derivs(0.0, np.array([2.0, 0.7, 5.0])))
    assert rate[0] != 0.0 and rate[0] + rate[1] == 0.0


def test_a_one_compartment_reaction_in_a_resized_compartment_keeps_its_law():
    """Control. A and B in the one compartment an event resizes move at the
    one rate, and ``A + B`` is their law between events, as it was."""
    model = bngsim.Model.from_antimony_string(
        "compartment c; c = 2; species A in c, B in c; A = 1; B = 0.5; k1 = 1; k2 = 0.5;\n"
        "R1: A -> B; k1*A*c\nR2: B -> A; k2*B*c\nE1: at (time > 5): c = 4;\n"
    )
    names = list(model.species_names)
    rows = np.asarray(model.conservation_laws["coefficients"], dtype=float)
    held = rows[:, [names.index("A"), names.index("B")]]
    assert [1.0, 1.0] in held.tolist()
    assert _kept(model) < 1e-12


def test_which_laws_span_sizes():
    """A law spans sizes where its species have different volume factors and
    a reaction divides by them: ``A + 2*B`` does, and the plain law of a
    built model with no such reaction does not, whatever its species' volume
    factors are."""
    spans = bngsim.Simulator._laws_across_sizes
    assert spans(_two()) == [[0, 1]]
    assert spans(_two(1.0)) == []
    assert spans(bngsim.Model(_core=_unscaled(5.0))) == []
    assert spans(bngsim.Model(_core=_built())) == [[0, 1]]


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
    assert sorted(map(tuple, laws["coefficients"])) == [
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ]


def test_a_law_the_right_hand_side_does_not_keep_is_refused():
    """The solvers hold each reported total. Before a solve the right-hand
    side is asked whether it keeps them, at two states off the model's own,
    and a law it does not keep is an error. A size below zero is one way to
    such a law: the detector takes a size that is not positive for 1, and the
    right-hand side divides by it. The solve came back A* = 0.5, B* = -0.25."""
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


def test_every_law_is_asked():
    """A law inside a compartment, kept, ahead of one across compartments
    that a size below zero breaks: the second is the one named."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; species X in c1, Y in c1, A in c1, B in c2;\n"
        "X = 1; Y = 0.2; A = 1; B = 0.5; k1 = 1; k2 = 0.5; k3 = 2; k4 = 1;\n"
        "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\nR3: X -> Y; c1*k3*X\nR4: Y -> X; c1*k4*Y\n"
    )
    rows = np.asarray(model.conservation_laws["coefficients"], dtype=float)
    np.testing.assert_allclose(rows, [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]])
    assert model._core.conservation_law_drift()[0] == -1
    model.set_param("c2", -2.0)
    assert model._core.conservation_law_drift()[0] == 1
    with pytest.raises(bngsim.SimulationError, match=r"\(over A, B\) is not kept"):
        bngsim.Simulator(model, method="ode").steady_state()


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_the_question_is_put_relative_to_the_terms(scale):
    """A law's total moves by rounding where the law is kept and by a share
    of its terms where it is not, whatever the size of the rates: a kept law
    passes with rate constants of 1e12, where the rounding is 1e-4, and a
    broken one is found with rate constants of 1e-12."""
    text = SHAPES["chain"][0]
    for name, value in (("k1", 1), ("k2", 0.5), ("k3", 0.3), ("k4", 0.7)):
        text = text.replace(f"{name} = {value};", f"{name} = {value * scale!r};")
    model = bngsim.Model.from_antimony_string(text)
    assert abs(model.get_param("k4") - 0.7 * scale) < 1e-9 * scale
    assert model._core.conservation_law_drift() == (-1, 0.0, 0.0)
    model.set_param("c3", -5.0)
    law, drift, size = model._core.conservation_law_drift()
    assert law == 0 and 0.01 * size < drift <= size


CRUMBS = (
    "begin parameters\n    1 k0 {k}\n    2 k1 {k}\n    3 k2 {k}\n    4 k3 {k}\n    5 kx 1.943\n"
    "end parameters\n"
    "begin species\n    1 S0() 0.959\n    2 S1() 0.884\n    3 S2() 0.601\n    4 S3() 1.134\n"
    "    5 S4() 1.658\nend species\n"
    "begin reactions\n    1 1,1,1 2,4,2,3 k0\n    2 4,4,5 2,4,1,2,2 k1\n    3 3,1 3,2 k2\n"
    "    4 5,2 5 k3\n    5 1 2 kx\nend reactions\n"
)


def test_rounding_left_on_a_species_in_no_law_is_not_the_law(tmp_path):
    """Row reduction leaves rounding on species a law does not hold: here
    ``S2 - S3 + S4`` with -1.1e-16 on S0 (where the arithmetic leaves any;
    1e-14 to 1e-40 over the corpus). With four rate constants at zero only
    ``S0 -> S1`` runs, so nothing flows through the three species of the law,
    and the rounding times S0's rate was the whole of its total and of what
    it was measured against. The law is asked over the species it holds,
    which do not move. (MODEL1009150002, 1,604 species, has laws in that
    state where the question is asked.)"""
    path = tmp_path / "crumbs.net"
    path.write_text(CRUMBS.format(k=0.0))
    model = bngsim.Model.from_net(str(path))
    assert model._core.conservation_law_members() == [[2, 3, 4]]
    rate = np.asarray(model.rhs(np.array([1.3, 0.7, 0.9, 1.1, 0.4])))
    assert rate[0] < -1.0 and np.all(rate[2:] == 0.0)
    assert model._core.conservation_law_drift() == (-1, 0.0, 0.0)
    out = bngsim.Simulator(model, method="ode").steady_state()
    np.testing.assert_allclose(
        np.asarray(out.concentrations),
        [0.0, 0.959 + 0.884, 0.601, 1.134, 1.658],
        atol=1e-6,
    )
    path.write_text(CRUMBS.format(k=0.7))
    assert bngsim.Model.from_net(str(path))._core.conservation_law_drift()[0] == -1


@pytest.mark.parametrize("v2", [1e-12, 1e12])
def test_sizes_twelve_orders_apart(v2):
    """A cell of 1e-12 beside a medium of 1: the law is ``A + V2*B``, and
    both species are held by it, though one coefficient is 1e-12 of the
    other. Which species a law holds is asked of the coefficients over the
    sizes."""
    model = _two(v2)
    np.testing.assert_allclose(_law(model), [1.0, v2], rtol=1e-12)
    assert model._core.conservation_law_members() == [[0, 1]]
    assert model._core.conservation_law_drift()[0] == -1
    assert _kept(model) < 1e-12


def test_the_columns_with_sizes_twelve_orders_apart():
    """A* = 1/2 and B* = 1/(2*V2) at V2 = 1e12, with dA*/dk1 = -1/4 and
    dB*/dk1 = 1/(4*V2)."""
    sim = bngsim.Simulator(_two(1e12), method="ode")
    out = sim.steady_state(sensitivity_params=["k1"], tol=1e-12)
    assert out.converged
    np.testing.assert_allclose(np.asarray(out.concentrations), [0.5, 0.5e-12], rtol=1e-9)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [-0.25, 0.25e-12], rtol=1e-9)


@pytest.mark.parametrize("v2", [1e-12, 1e12])
def test_a_moved_state_refuses_the_initial_amount_of_either_species(v2):
    """On a state a run has advanced, a parameter that sets the initial amount
    of a species in a law is refused (issue #704), for A and for B alike: both
    are in the law, whatever their sizes."""
    text = TWO.format(v2=v2).replace("A = 1; B = 0;", "A0 = 1; B0 = 0; A = A0; B = B0;")
    for name in ("A0", "B0"):
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
        sim.run(t_span=(0.0, 0.1), n_points=2)
        with pytest.raises(bngsim.SensitivityUnsupportedError, match="carried-over"):
            sim.steady_state(sensitivity_params=[name])


def test_the_question_is_asked_at_a_second_state():
    """A rate law that has no value at the first state asked, ``sqrt(A - 2)``
    with A at 1.88 there, has one at the second, A at 2.33: the law is asked
    there."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2; A = 1; B = 0.25;\n"
        "k1 = 1; k2 = 1;\nR1: A -> B; k1*A*c1*sqrt(A - 2)\nR2: B -> A; k2*B*c2\n"
    )
    assert model._core.conservation_law_drift()[0] == -1
    model.set_param("c2", -2.0)
    assert model._core.conservation_law_drift()[0] == 0


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


# B goes to P and to Q, neither of which comes back, and each is exchanged with
# a partner in the other compartment. Where the run ends depends on the split
# between them, so the steady state is one of a continuum.
CONTINUUM = (
    "compartment c1, c2; c1 = 0.7; c2 = {v2};\n"
    "species A in c1, B in c2, P in c2, P2 in c1, Q in c2, Q2 in c1;\n"
    "A = 1.3; B = 0.2; P = 0; P2 = 0; Q = 0; Q2 = 0;\n"
    "k1 = 0.37; k2 = 1.91; kp = 0.83; kq = 0.29; a = 0.61; b = 1.17; d = 0.43; e = 0.77;\n"
    "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
    "R3: B -> P; c2*kp*B/(0.7 + B)\nR4: B -> Q; c2*kq*B\n"
    "R5: P -> P2; c2*a*P\nR6: P2 -> P; c1*b*P2\nR7: Q -> Q2; c2*d*Q\nR8: Q2 -> Q; c1*e*Q2\n"
)


def test_a_continuum_across_sizes_is_refused():
    """A steady state that is one of a continuum has a singular reduced
    Jacobian. With one size the pivot is an exact zero and the solve is
    refused for it; with sizes 0.7 and 2.3 it is rounding, 1e-17 of the
    largest, and the columns came back with a warning beside them:
    dP*/dkp = -329,603 where runs to the steady state give 0.0774. Such a
    model was refused for its law (issue #704), then for a law across sizes
    at a ratio min|U|/max|U| below 1e-8 (issue #758), and is refused now for
    what it is, a root that is not isolated (issue #995): the condition
    number of the reduced Jacobian is 1e16 in any units. (Which measure says
    so first is the platform's arithmetic: the pivot that is rounding on one
    is an exact zero on another, and the determinant says it there.)"""
    model = bngsim.Model.from_antimony_string(CONTINUUM.format(v2=2.3))
    np.testing.assert_allclose(_law(model), [1, 2.3 / 0.7, 2.3 / 0.7, 1, 2.3 / 0.7, 1], rtol=1e-12)
    sim = bngsim.Simulator(model, method="ode")
    for name in ("kp", "k1"):
        with pytest.raises(
            bngsim.SimulationError,
            match=r"#995.*not an isolated root",
        ):
            sim.steady_state(sensitivity_params=[name], tol=1e-10)
        model.reset()
    out = sim.steady_state(tol=1e-10)  # the state itself is where a run ends
    assert out.converged and abs(np.asarray(out.concentrations)[1]) < 1e-8


def test_a_zero_pivot_beside_a_law_across_sizes_is_refused_as_having_no_gradient():
    """X goes to P and to nothing, so P is in no law and no rate reads it: its
    column of the reduced Jacobian is zero, whatever the sizes of the law
    beside it. The factorization stops at that pivot and the columns are not
    finite. The refusal is the one that says no gradient exists."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2, X in c1, P in c1;\n"
        "A = 1; B = 0.2; X = 1; P = 0; k1 = 1; k2 = 0.5; kp = 0.3; kd = 0.2;\n"
        "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\nR3: X -> P; c1*kp*X\nR4: X -> ; c1*kd*X\n"
    )
    assert model._core.conservation_law_members() == [[0, 1]]
    with pytest.raises(bngsim.SimulationError, match="dY_ss/dp does not exist") as caught:
        bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=["kp"], tol=1e-10)
    assert not isinstance(caught.value, bngsim.SensitivityUnsupportedError)


def test_the_continuum_in_amounts_under_a_rate_rule_is_refused():
    """The same network with its species held as amounts and ``c2`` under a
    rate rule that settles at 2.3. ``c2`` is then a species and not a size
    parameter, and the law still spans two sizes: dP*/dkp came back 396,941
    on a pivot of 6e-17 of the largest, for 0.089. It was refused for a law
    across sizes at such a ratio (issue #758), and is refused for a root that
    is not isolated (issue #995)."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 0.7; c2 = 1.1; c2' = 0.9*(2.3 - c2);\n"
        "substanceOnly species A in c1, B in c2, P in c2, P2 in c1, Q in c2, Q2 in c1;\n"
        "A = 1.3; B = 0.2; P = 0; P2 = 0; Q = 0; Q2 = 0;\n"
        "k1 = 0.37; k2 = 1.91; kp = 0.83; kq = 0.29; a = 0.61; b = 1.17; d = 0.43; e = 0.77;\n"
        "R1: A -> B; k1*A\nR2: B -> A; k2*B\nR3: B -> P; kp*B/(0.7 + B)\nR4: B -> Q; kq*B\n"
        "R5: P -> P2; a*P\nR6: P2 -> P; b*P2\nR7: Q -> Q2; d*Q\nR8: Q2 -> Q; e*Q2\n"
    )
    assert model.compartment_size_params == ["c1"]
    with pytest.raises(bngsim.SimulationError, match=r"#995.*not an isolated root"):
        bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=["kp"], tol=1e-10)


def test_the_same_continuum_in_one_size_is_refused_as_it_was():
    """Control. With both sizes 0.7 the pivot is an exact zero, the solve
    returns non-finite columns, and that is the refusal there has been."""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(CONTINUUM.format(v2=0.7)), method="ode"
    )
    with pytest.raises(
        bngsim.SimulationError,
        match=r"dY_ss/dp does not exist.*sens_jacobian_rcond is 0.00e\+00",
    ):
        sim.steady_state(sensitivity_params=["kp"], tol=1e-10)


CATALYST = (
    "compartment c; c = 1; species E in c, I in c, EI in c, S in c, P in c;\n"
    "E = 1; I = 2; EI = 0; S = 5; P = 0; kon = 1e-3; koff = 2e-3; kcat = {kcat};\n"
    "R1: E + I -> EI; kon*E*I\nR2: EI -> E + I; koff*EI\n"
    "R3: E + S -> E + P; kcat*E*S\nR4: P -> S; kcat*P\n"
)


@pytest.mark.parametrize("kcat", [1.0, 1e6, 1e9, 1e12])
def test_a_catalyst_of_a_fast_reaction_does_not_break_its_law(kcat):
    """E is on both sides of ``E + S -> E + P``, so its rate takes the
    reaction's rate off and puts it back, and is left with that rate's
    rounding: 1e-10 at kcat = 1e6, beside binding terms of 1e-3. ``E + EI``
    is kept all the same. The total is measured against the fluxes through
    E and EI, which is what its rounding is of, and not against their net
    rates, against which it was 1.3e-8 and refused."""
    model = bngsim.Model.from_antimony_string(CATALYST.format(kcat=kcat))
    assert [0, 2] in model._core.conservation_law_members()
    assert model._core.conservation_law_drift() == (-1, 0.0, 0.0)


def test_the_steady_state_beside_a_fast_catalytic_step():
    """Control. E* = 1 - EI* with EI* = (5 - sqrt(17))/2, whatever kcat is:
    the solve that the question about the law first refused."""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(CATALYST.format(kcat=1e6)), method="ode"
    )
    out = sim.steady_state(method="newton")
    assert out.converged
    np.testing.assert_allclose(np.asarray(out.concentrations)[0], 1 - (5 - 17**0.5) / 2, rtol=1e-7)


@pytest.mark.parametrize("fast", [1.0, 1e6, 1e9])
def test_a_broken_law_beside_a_fast_exchange_is_found(fast):
    """A law that a size below zero breaks, with a fast ``B <-> C`` among its
    species: the total moves by 3 against fluxes of ``fast``, and is found
    down to 1e-10 of them."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2, C in c2;\n"
        f"A = 1; B = 0.5; C = 0.2; k1 = 1; k2 = 0.5; kf = {fast}; kb = {fast};\n"
        "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\nR3: B -> C; c2*kf*B\nR4: C -> B; c2*kb*C\n"
    )
    assert model._core.conservation_law_drift()[0] == -1
    model.set_param("c2", -2.0)
    assert model._core.conservation_law_drift()[0] == 0


MASKED = (
    "compartment c1, c2; c1 = 1; c2 = {v2}; species A in c1, B in c2, D in c1;\n"
    "A0 = 1; A = A0; B = 0; D = 0; k1 = 1; k2 = 0.5; k3 = 0.7; k4 = 0.2;\n"
    "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\nR3: A -> D; c1*k3*A\nR4: D -> A; c1*k4*D\n"
)


@pytest.mark.parametrize("v2", [2.0, 1e6, 1e10, 1e12])
def test_a_masked_out_species_is_one_its_law_holds_at_any_size(v2):
    """Whether the law holds D was asked of the coefficients without the
    sizes, where D's is 1e-10 of B's at V2 = 1e10, and D was taken for a
    species no law holds. Asked with them, the law holds all three at any
    size. (This was read off dA*/dA0 with D masked out, 0 where the law keeps
    its total and 1/3 where it was not found to. A's rate reads D, so that
    request is refused now, issue #995: with D held, 0 is not the derivative
    either.)"""
    model = bngsim.Model.from_antimony_string(MASKED.format(v2=v2))
    assert [sorted(members) for members in model._core.conservation_law_members()] == [[0, 1, 2]]
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(
        bngsim.SimulationError, match=r"#995.*mask= leaves out D, which the rate of A reads"
    ):
        sim.steady_state(sensitivity_params=["A0"], mask=["A", "B"], tol=1e-10)


WITH_A_TOTAL = (
    "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2; A = 1; B = 0;\n"
    "k1 = 1; k2 = 0.5; tot := A*c1 + B*c2;\nR1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
)


def test_a_batch_leaves_its_model_as_it_found_it():
    """Control. The entries are solved on clones. The model itself is not
    evaluated anywhere: its state and the value of a function of it are what
    they were after the run before."""
    model = bngsim.Model.from_antimony_string(WITH_A_TOTAL)
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 0.3), n_points=2)
    state, total = list(model.get_state()), model.get_param("tot")
    assert abs(total - 1.0) < 1e-6
    sim.steady_state_batch(params=[{"k1": 2.0}, {"k1": 3.0}])
    assert list(model.get_state()) == state and model.get_param("tot") == total


def test_asking_a_law_puts_the_function_values_back():
    """The question is put at two states off the model's own, which writes
    the observable totals and the function values. They are put back as
    they were: ``tot`` after a run, and where the law is broken too."""
    model = bngsim.Model.from_antimony_string(WITH_A_TOTAL)
    bngsim.Simulator(model, method="ode").run(t_span=(0.0, 0.3), n_points=2)
    total = model.get_param("tot")
    assert model._core.conservation_law_drift()[0] == -1
    assert model.get_param("tot") == total
    model.set_param("c2", -2.0)
    total = model.get_param("tot")
    assert model._core.conservation_law_drift()[0] == 0
    assert model.get_param("tot") == total


def test_a_rate_of_function_keeps_its_value_through_the_question():
    """``ra := rateOf(A)`` holds -2 on a model that has only been loaded.
    Asking the law leaves it there, and a refused ``steady_state`` leaves what
    the simulator's own set-up wrote: an evaluation at the model's state
    would have written 0, the buffer ``rateOf`` reads being empty until a
    run fills it."""
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; species A in c1, B in c2; A = 1; B = 0;\n"
        "k1 = 1; k2 = 0.5; ra := rateOf(A); tot := A*c1 + B*c2;\n"
        "R1: A -> B; 2*k1*A*c1\nR2: B -> A; k2*B*c2\n"
    )
    assert (model.get_param("ra"), model.get_param("tot")) == (-2.0, 1.0)
    assert model._core.conservation_law_drift()[0] == -1
    assert (model.get_param("ra"), model.get_param("tot")) == (-2.0, 1.0)
    sim = bngsim.Simulator(model, method="ode")
    before = (model.get_param("ra"), model.get_param("tot"))
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"compartment\s+size"):
        sim.steady_state(sensitivity_params=["c2"])
    assert (model.get_param("ra"), model.get_param("tot")) == before


def test_a_size_that_is_not_a_number():
    """A size written NaN is taken for 1 by the detector, and the rates are
    not finite, so the law is not asked. The laws are the same at every
    call."""
    model = _two()
    model.set_param("c2", float("nan"))
    first = model.conservation_laws
    assert first["coefficients"] == [[1.0, 1.0]]
    assert model.conservation_laws == first
    assert model._core.conservation_law_members() == [[0, 1]]
    assert model._core.conservation_law_drift() == (-1, 0.0, 0.0)
    model.set_param("c2", 4.0)
    np.testing.assert_allclose(_law(model), [1.0, 4.0])


@pytest.mark.parametrize("n, tail", [(4, ""), (5, ", ...")])
def test_the_refusal_names_up_to_four_species(n, tail):
    """ "(over A, S1, S2, S3)" for a law of four, and ", ..." after four of
    five."""
    names = [f"S{i}" for i in range(1, n)]
    chain = "".join(
        f"F{i}: {a} -> {b}; c2*{a}\nG{i}: {b} -> {a}; c2*{b}\n"
        for i, (a, b) in enumerate(zip(names, names[1:], strict=False))
    )
    model = bngsim.Model.from_antimony_string(
        "compartment c1, c2; c1 = 1; c2 = 2; "
        f"species A in c1, {', '.join(s + ' in c2' for s in names)};\n"
        f"A = 1; {'; '.join(s + ' = 0.1' for s in names)};\n"
        f"R1: A -> S1; A*c1\nR2: S1 -> A; S1*c2\n{chain}"
    )
    assert model._core.conservation_law_members() == [list(range(n))]
    model.set_param("c2", -2.0)
    with pytest.raises(bngsim.SimulationError) as caught:
        bngsim.Simulator(model, method="ode").steady_state()
    assert f"(over A, S1, S2, S3{tail}) is not kept" in str(caught.value)


RULED = (
    "compartment c1, c2; c1 = 1; {size}; species A in c1, B in c2; A = 1; B = 0.2;\n"
    "k1 = 1; k2 = 0.5;\nR1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
)


@pytest.mark.parametrize("size", ["c2 := 1 + A", "p = 2; c2 := 2*p"])
def test_a_law_across_a_rule_sized_compartment_is_still_refused(size):
    """An assignment rule sizes ``c2``, and the right-hand side divides B's
    share by the size the model loaded at (issue #745): with ``c2 := 1 + A``
    the run ends at A = 0.354 where the amounts give 0.467, and the columns
    of that system came back -0.329 for -0.311 with nothing said. Such a
    model was refused for its law and stays refused, also where the rule is
    a constant and the columns would be right."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(RULED.format(size=size)), "ode")
    with pytest.raises(
        bngsim.SensitivityUnsupportedError,
        match=r"assignment rule\s+sets the size of c2.*#745",
    ):
        sim.steady_state(sensitivity_params=["k1"])
    assert sim.steady_state().converged  # the state is solved for, as it was
