"""Issue #1513 — Result.squeeze refuses rows that do not share row 0's axes.

A stack has one time axis and one name list per kind of column, both taken from
row 0, and every row's columns are laid under them by position. Rows run on
other times, or with their species in another order, were labelled with row 0's
and read wrong everywhere, with nothing said: A -> B at k = 1 run on
t = 0, 10, 20 beside a run on t = 0, 1, 2 read A(10) = 0.0045 at "t = 1", where
the closed form A(1) = 100 e^-1 = 36.79.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 k 1
end parameters
begin species
    1 {first}
    2 {second}
end species
begin reactions
    1 {reaction} k
end reactions
"""


@pytest.fixture
def nets(tmp_path):
    ab, ba = tmp_path / "ab.net", tmp_path / "ba.net"
    ab.write_text(NET.format(first="A() 100", second="B() 0", reaction="1 2"))
    ba.write_text(NET.format(first="B() 0", second="A() 100", reaction="2 1"))
    return ab, ba


def _run(path, t_end, **sim_kwargs):
    model = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(model, method="ode", **sim_kwargs)
    return sim.run(t_span=(0, t_end), n_points=3, rtol=1e-10, atol=bngsim.ExactAtol(1e-12))


def test_rows_on_other_times_are_refused(nets):
    ab, _ = nets
    row0, row1 = _run(ab, 2), _run(ab, 20)
    # What the stack used to say row 1 holds at its labelled t = 1 is A(10);
    # the closed form there is A(1).
    a = row1.species_names.index("A()")
    assert row1.species[1, a] == pytest.approx(100 * math.exp(-10), rel=1e-6)
    with pytest.raises(ValueError, match=r"row 1 has time point 1 at t = 10\.0 .* t = 1\.0"):
        bngsim.Result.squeeze([row0, row1])


def test_rows_with_species_in_another_order_are_refused(nets):
    ab, ba = nets
    with pytest.raises(ValueError, match=r"row 1 has species \['B\(\)', 'A\(\)'\]"):
        bngsim.Result.squeeze([_run(ab, 2), _run(ba, 2)])


def test_rows_with_sensitivity_parameters_in_another_order_are_refused(tmp_path):
    net = tmp_path / "two.net"
    net.write_text(
        "begin parameters\n 1 k 1\n 2 k2 0.5\nend parameters\n"
        "begin species\n 1 A() 100\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 k\n 2 2 1 k2\nend reactions\n"
    )
    first = _run(net, 2, sensitivity_params=["k", "k2"])
    second = _run(net, 2, sensitivity_params=["k2", "k"])
    with pytest.raises(ValueError, match=r"sensitivity parameters \['k2', 'k'\]"):
        bngsim.Result.squeeze([first, second])


@pytest.mark.parametrize("sens_first", [False, True])
def test_a_block_only_some_rows_have_is_refused(nets, sens_first):
    # Row 0 alone decided whether to stack a block: [plain, with_sens] dropped
    # the sensitivities and the reverse raised numpy's shape error. The name
    # check meets it first, so the message is the same either way round.
    ab, _ = nets
    plain, with_sens = _run(ab, 2), _run(ab, 2, sensitivity_params=["k"])
    rows = [with_sens, plain] if sens_first else [plain, with_sens]
    with pytest.raises(ValueError, match=r"row 1 has sensitivity parameters"):
        bngsim.Result.squeeze(rows)


def _row(**overrides):
    """A raw two-species, three-time Result, with *overrides* replacing a field."""
    fields = {
        "_time": np.array([0.0, 1.0, 2.0]),
        "_species": np.ones((3, 2)),
        "_observables": np.ones((3, 1)),
        "_species_names": ["A", "B"],
        "_observable_names": ["Atot"],
        "_expression_names": [],
    }
    fields.update(overrides)
    return bngsim.Result(**fields)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"_observable_names": ["Btot"]}, r"observables \['Btot'\] where row 0 has \['Atot'\]"),
        (
            {"_expressions": np.ones((3, 1)), "_expression_names": ["f"]},
            r"expressions \['f'\] where row 0 has \[\]",
        ),
        (
            {"_reaction_labels": ["r1"], "_reaction_firing_counts": np.ones((3, 1))},
            r"reaction labels \['r1'\]",
        ),
        ({"_reaction_firing_counts": np.ones((3, 1))}, r"has reaction firing counts where row 0"),
        ({"_time": np.array([0.0, 1.0])}, r"has 2 time points where row 0 has 3"),
    ],
)
def test_each_disagreement_is_named(override, message):
    if "_time" in override:
        override = {**override, "_species": np.ones((2, 2)), "_observables": np.ones((2, 1))}
    with pytest.raises(ValueError, match=message):
        bngsim.Result.squeeze([_row(), _row(**override)])


def test_rows_that_agree_still_stack_with_their_own_values(nets):
    ab, _ = nets
    rows = [_run(ab, 2), _run(ab, 2), _run(ab, 2)]
    stack = bngsim.Result.squeeze(rows)
    assert stack.species.shape == (3, 3, 2)
    np.testing.assert_array_equal(stack.time, rows[0].time)
    a = stack.species_names.index("A()")
    # The closed form A(t) = 100 e^-t at every row's labelled time.
    closed = np.broadcast_to(100 * np.exp(-stack.time), (3, 3))
    np.testing.assert_allclose(stack.species[:, :, a], closed, rtol=1e-8)
