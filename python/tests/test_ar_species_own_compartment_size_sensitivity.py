"""The sensitivity of an assignment-rule species to the size of its own
compartment (issue #724).

A ``hasOnlySubstanceUnits`` species that an assignment rule sets is reported
as ``rule/V``, with ``V`` the compartment's size, read live since the size
became a writable parameter (#170). Its sensitivity row is the rule's own row
over ``V``, and in the column of that size it has a second term, the reported
value over the size with its sign turned. Nothing added it: ``T := 3·A`` with A
decaying at ``k`` from 100 is reported as ``300·e^(−kt)/C``, and dT/dC came back
0 at every time for ``−300·e^(−kt)/C²``, in ``Result.sensitivities``, in
``output_sensitivities("species:T")``, in ``compute_all_sensitivities()`` (whose
default columns hold the size) and in ``steady_state()``.

Expected values are closed forms.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

DECAY = (
    "compartment C; C = {c}; species A in C; substanceOnly species T in C; A = 100; k = 0.3\n"
    "J: A -> ; C*k*A\nT := {rule}\n"
)
TIMES = np.linspace(0.0, 4.0, 5)


def _decay(c=2.0, rule="3*A"):
    return bngsim.Model.from_antimony_string(DECAY.format(c=c, rule=rule))


def _run(model, params=("C", "k")):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    return sim.run(t_span=(0.0, 4.0), n_points=5, rtol=1e-10, atol=1e-12)


def _row(result, species="T"):
    return np.asarray(result.sensitivities)[:, list(result.species_names).index(species), :]


@pytest.mark.parametrize("c", [2.0, 1.0, 0.25])
def test_the_column_of_the_size_has_the_value_over_the_size(c):
    """At a size of 1 the divisor is 1 and the term is still there."""
    result = _run(_decay(c))
    decay = np.exp(-0.3 * TIMES)
    names = list(result.species_names)
    np.testing.assert_allclose(
        np.asarray(result.species)[:, names.index("T")], 300.0 * decay / c, rtol=1e-8
    )
    got = _row(result)
    np.testing.assert_allclose(got[:, 0], -300.0 * decay / c**2, rtol=1e-7)
    np.testing.assert_allclose(got[:, 1], -300.0 * TIMES * decay / c, rtol=1e-7, atol=1e-9)


def test_the_selector_has_it_too():
    result = _run(_decay(2.0))
    got = np.asarray(result.output_sensitivities(["species:T"]))[:, 0, :]
    np.testing.assert_allclose(got, _row(result), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[:, 0], -75.0 * np.exp(-0.3 * TIMES), rtol=1e-7)


def test_every_column_at_once_has_it():
    """``compute_all_sensitivities()`` takes the size as one of its columns
    without being asked."""
    sim = bngsim.Simulator(_decay(2.0), method="ode")
    result = sim.compute_all_sensitivities(t_span=(0.0, 4.0), n_points=5, rtol=1e-10, atol=1e-12)
    column = list(result.sensitivity_params).index("C")
    got = _row(result)[:, column]
    np.testing.assert_allclose(got, -75.0 * np.exp(-0.3 * TIMES), rtol=1e-7)


def test_a_rule_that_reads_the_size_itself():
    """``T := 3·A·C`` is reported as ``3·A``, which does not move with C: the
    rule's own derivative over the size and the new term cancel. The row came
    back ``3·A/C``."""
    result = _run(_decay(2.0, rule="3*A*C"))
    np.testing.assert_allclose(_row(result)[:, 0], 0.0, atol=1e-7)


def test_the_size_is_read_as_it_is_set():
    """Loaded at a size of 1 and set to 4: the term is over the size the run
    was made at."""
    model = _decay(1.0)
    model.set_param("C", 4.0)
    result = _run(model)
    np.testing.assert_allclose(_row(result)[:, 0], -300.0 * np.exp(-0.3 * TIMES) / 16.0, rtol=1e-7)


def test_each_row_of_a_batch_has_it_at_its_own_size():
    sim = bngsim.Simulator(_decay(2.0), method="ode", sensitivity_params=["C", "k"])
    rows = sim.run_batch(
        params=[{"C": 2.0}, {"C": 5.0}], t_span=(0.0, 4.0), n_points=5, rtol=1e-10, atol=1e-12
    )
    for row, c in zip(rows, (2.0, 5.0), strict=True):
        np.testing.assert_allclose(
            _row(row)[:, 0], -300.0 * np.exp(-0.3 * TIMES) / c**2, rtol=1e-7
        )


STEADY = (
    "compartment C; C = 2; species A in C; substanceOnly species T in C; A = 0; k = 0.3; kp = 6\n"
    "J0: -> A; C*kp\nJ: A -> ; C*k*A\nT := 3*A\n"
)


def test_the_steady_state_has_it():
    """A settles at kp/k = 20 and T is reported as 3·20/C: dT/dC = −15, and
    came back 0."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(STEADY), method="ode")
    out = sim.steady_state(sensitivity_params=["C", "k", "kp"], tol=1e-12)
    names = list(out.species_names)
    assert float(np.asarray(out.concentrations)[names.index("T")]) == pytest.approx(30.0)
    got = np.asarray(out.sensitivity)[names.index("T")]
    np.testing.assert_allclose(got, [-15.0, -100.0, 5.0], rtol=1e-7)


def test_the_other_columns_and_species_are_as_they_were():
    """Control. The rate constant's column of T, and every column of A, do
    not go through the divisor's derivative."""
    result = _run(_decay(2.0))
    decay = np.exp(-0.3 * TIMES)
    np.testing.assert_allclose(_row(result)[:, 1], -150.0 * TIMES * decay, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(_row(result, "A")[:, 0], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        _row(result, "A")[:, 1], -100.0 * TIMES * decay, rtol=1e-7, atol=1e-9
    )


def test_a_rule_species_that_is_a_concentration_has_no_such_term():
    """Control. Without ``hasOnlySubstanceUnits`` the species is reported as
    the rule's value, with no divisor."""
    text = (
        "compartment C; C = 2; species A in C, T in C; A = 100; k = 0.3\n"
        "J: A -> ; C*k*A\nT := 3*A\n"
    )
    result = _run(bngsim.Model.from_antimony_string(text))
    np.testing.assert_allclose(_row(result)[:, 0], 0.0, atol=1e-7)


def test_the_initial_condition_axis_is_as_it_was():
    """Control. The size does not move with an initial condition, in a run
    that has the size's own column as well."""
    sim = bngsim.Simulator(
        _decay(2.0), method="ode", sensitivity_params=["C", "k"], sensitivity_ic=["A"]
    )
    result = sim.run(t_span=(0.0, 4.0), n_points=5, rtol=1e-10, atol=1e-12)
    names = list(result.species_names)
    want = 1.5 * np.exp(-0.3 * TIMES)
    got = np.asarray(result.sensitivities_ic)[:, names.index("T"), 0]
    np.testing.assert_allclose(got, want, rtol=1e-7)
    selected = np.asarray(result.output_sensitivities(["species:T"], axis="ic"))[:, 0, 0]
    np.testing.assert_allclose(selected, want, rtol=1e-7)


def test_the_term_goes_to_the_size_s_column_wherever_that_is():
    """With the size last, and between two others: in the tensor, in the
    selector, and at the steady state."""
    decay = np.exp(-0.3 * TIMES)
    for params in (["k", "C"], ["k", "C", "k"]):
        result = _run(_decay(2.0), params)
        column = params.index("C")
        got = _row(result)
        np.testing.assert_allclose(got[:, column], -75.0 * decay, rtol=1e-7)
        np.testing.assert_allclose(got[:, 0], -150.0 * TIMES * decay, rtol=1e-7, atol=1e-9)
        selected = np.asarray(result.output_sensitivities(["species:T"]))[:, 0, :]
        np.testing.assert_array_equal(selected, got)
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(STEADY), method="ode")
    out = sim.steady_state(sensitivity_params=["k", "kp", "C"], tol=1e-12)
    got = np.asarray(out.sensitivity)[list(out.species_names).index("T")]
    np.testing.assert_allclose(got, [-100.0, 5.0, -15.0], rtol=1e-7)


def test_the_selector_on_rows_stacked_at_one_size():
    """A batch squeezed into one result keeps the redirect where every row
    has the same size, and the selector applies the term to the stack."""
    sim = bngsim.Simulator(_decay(2.0), method="ode", sensitivity_params=["k", "C"])
    stacked = sim.run_batch(
        params=[{"k": 0.3}, {"k": 0.6}],
        t_span=(0.0, 4.0),
        n_points=5,
        rtol=1e-10,
        atol=1e-12,
        squeeze=True,
    )
    column = list(stacked.species_names).index("T")
    tensor = np.asarray(stacked.sensitivities)[..., column, :]
    selected = np.asarray(stacked.output_sensitivities(["species:T"]))
    np.testing.assert_array_equal(np.squeeze(selected), np.squeeze(tensor))
    for row, k in zip(np.squeeze(tensor), (0.3, 0.6), strict=True):
        np.testing.assert_allclose(row[:, 1], -75.0 * np.exp(-k * TIMES), rtol=1e-7)


def test_columns_computed_one_at_a_time_and_stitched_have_it_once():
    sim = bngsim.Simulator(_decay(2.0), method="ode")
    result = sim.compute_all_sensitivities(
        t_span=(0.0, 4.0), n_points=5, rtol=1e-10, atol=1e-12, params=["k", "C"], chunk_size=1
    )
    got = _row(result)
    np.testing.assert_allclose(got[:, 1], -75.0 * np.exp(-0.3 * TIMES), rtol=1e-7)
    # Each column is its own run, and the selector reads one run's values:
    # the same to the runs' tolerance, and not to the last bit.
    selected = np.asarray(result.output_sensitivities(["species:T"]))[:, 0, :]
    np.testing.assert_allclose(selected, got, rtol=1e-7, atol=1e-9)
