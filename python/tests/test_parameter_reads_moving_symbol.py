"""A parameter cannot be defined in terms of something a run moves.

A derived parameter is re-evaluated when a parameter is written, and at no
other time. One that read time(), a function, an observable or a rate accessor
therefore held whatever that symbol read at build for the whole run: with
``kf = c + time()`` a function and ``k2 = 0.5*kf`` a parameter, ``0 -> A`` at
k2 ran at 0 under ODE and SSA alike, where A(4) is 6. A quantity that moves
with time is a function, and the builder now refuses the parameter and says so.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder


def _builder(rows, *, functions=(), observable=False):
    b = ModelBuilder()
    b.add_parameter("c", 1.0)
    for name, expr in functions:
        b.add_function(name, expr)
    for name, expr in rows:
        b.add_parameter(name, 0.0, expr, True)
    b.add_species("A", 0.0)
    if observable:
        b.add_observable("Atot", [(0, 1.0)])
    return b


@pytest.mark.parametrize(
    ("rows", "functions", "observable", "named"),
    [
        pytest.param([("k2", "0.5*(c + time())")], (), False, "time()", id="time"),
        pytest.param([("k2", "0.5*kf")], [("kf", "c + time()")], False, "'kf'", id="function"),
        pytest.param([("k2", "2*Atot")], (), True, "'Atot'", id="observable"),
        pytest.param(
            [("k3", "2*k2"), ("k2", "c*time()")], (), False, "time()", id="through-a-derived-one"
        ),
    ],
)
def test_a_parameter_that_reads_a_moving_symbol_is_refused(rows, functions, observable, named):
    b = _builder(rows, functions=functions, observable=observable)
    b.add_reaction([], [0], "elementary", rows[-1][0])
    with pytest.raises(RuntimeError) as exc:
        b.build()
    msg = str(exc.value)
    assert named in msg and "changes during a run" in msg and "as a function instead" in msg


def test_the_same_quantity_as_a_function_runs():
    """``k2`` as a function: dA/dt = 0.5·(1 + t), A(4) = 6."""
    b = _builder((), functions=[("kf", "c + time()"), ("k2", "0.5*kf")], observable=True)
    b.add_reaction([], [0], "functional", "k2")
    m = bngsim.Model(_core=b.build())
    r = bngsim.Simulator(m, method="ode").run(t_span=(0, 4), n_points=2)
    assert np.asarray(r.species)[-1, 0] == pytest.approx(6.0, rel=1e-6)


def test_a_parameter_of_parameters_still_builds():
    b = _builder([("k2", "0.5*c")])
    b.add_reaction([], [0], "elementary", "k2")
    m = bngsim.Model(_core=b.build())
    assert m.get_param("k2") == 0.5


def test_a_function_slot_seeded_by_an_expression_still_builds():
    """An SBML assignment rule arrives as a parameter row and a function of the
    same name. The row's expression only seeds the slot, which the function
    overwrites every step, so reading time there is not a definition."""
    b = ModelBuilder()
    b.add_parameter("c", 1.0)
    b.add_parameter("kf", 0.0, "c + time()", True)
    b.add_function("kf", "c + time()")
    b.add_species("A", 0.0)
    b.add_reaction([], [0], "functional", "kf")
    m = bngsim.Model(_core=b.build())
    r = bngsim.Simulator(m, method="ode").run(t_span=(0, 2), n_points=2)
    assert np.asarray(r.species)[-1, 0] == pytest.approx(4.0, rel=1e-6)
