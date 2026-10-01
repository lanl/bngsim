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
    assert named in msg and "as a function instead" in msg
    # the parameter that reads it, not one further down the chain
    assert "parameter 'k2'" in msg


def test_a_rate_accessor_is_refused():
    b = _builder([("k2", "2*rate_of__A")])
    b.enable_rateof()
    b.add_reaction([], [0], "elementary", "k2")
    with pytest.raises(RuntimeError, match="the rate accessor 'rate_of__A'"):
        b.build()


def test_a_slot_a_same_named_function_owns_is_refused():
    """The `.net` spelling of an SBML assignment rule: a parameter row ``kf``
    and a function ``kf`` that rewrites it every step. ``k2 = 0.5*kf`` read
    the row's seed at build and, after a same-value write, whatever the last
    run left in the slot."""
    b = ModelBuilder()
    b.add_parameter("c", 1.0)
    b.add_parameter("kf", 1.0)
    b.add_function("kf", "c + time()")
    b.add_parameter("k2", 0.0, "0.5*kf", True)
    b.add_species("A", 0.0)
    b.add_reaction([], [0], "elementary", "k2")
    with pytest.raises(RuntimeError, match="the function 'kf'"):
        b.build()


def test_a_self_reference_is_named_as_one():
    b = _builder([("k2", "k2 + time()")])
    b.add_reaction([], [0], "elementary", "k2")
    with pytest.raises(RuntimeError, match="defined in terms of itself"):
        b.build()


def test_names_that_are_not_the_clock_still_build():
    """Only the call ``time()`` is the clock. A declared scalar named ``time``
    (issue #776) and names that merely contain it are parameters."""
    b = ModelBuilder()
    b.add_parameter("time", 5.0)
    b.add_parameter("time_scale", 3.0)
    b.add_parameter("k2", 0.0, "2*time + 0*time_scale", True)
    b.add_species("A", 0.0)
    b.add_observable("runtime", [(0, 1.0)])
    b.add_reaction([], [0], "elementary", "k2")
    m = bngsim.Model(_core=b.build())
    assert m.get_param("k2") == 10.0
    r = bngsim.Simulator(m, method="ode").run(t_span=(0, 4), n_points=2)
    assert np.asarray(r.species)[-1, 0] == pytest.approx(40.0, rel=1e-9)


def test_through_a_net_file(tmp_path):
    p = tmp_path / "k.net"
    p.write_text(
        "begin parameters\n    1 c 1\n    2 k2 c*time()\nend parameters\n"
        "begin species\n    1 A() 0\nend species\n"
        "begin reactions\n    1 0 1 k2\nend reactions\n"
    )
    with pytest.raises(Exception, match="parameter 'k2' = c\\*time\\(\\) reads time\\(\\)"):
        bngsim.Model.from_net(str(p))


def test_an_initial_value_that_reads_the_clock_names_the_species(tmp_path):
    """``A() 2+time()``: the `.net` loader carries the expression on a helper
    parameter. It loaded as A = 2 and moved to 6 after a same-value write and a
    reset. The refusal speaks of A's initial value, not of the helper."""
    p = tmp_path / "ic.net"
    p.write_text(
        "begin parameters\n    1 k 0\nend parameters\n"
        "begin species\n    1 A() 2+time()\nend species\n"
        "begin reactions\n    1 0 1 k\nend reactions\n"
    )
    with pytest.raises(Exception, match="initial value of species 'A\\(\\)'") as exc:
        bngsim.Model.from_net(str(p))
    assert "_InitialConc" not in str(exc.value)


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
