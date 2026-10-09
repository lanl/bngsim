"""A column that has no frame for an edge its parameter moves (issue #1003).

A window ``k1·s·(1-s)^(a-1)`` with ``s = (t - onset)/D`` closes at
``onset + D`` as a power, and for 1 < a < 2 the forcing of a column that moves
that edge is unbounded there. Such a column is integrated in a frame that
moves with the edge (issues #545, #760), at the rate ``c = ∂t*/∂p`` the edge's
time changes with the parameter. The generator reads ``c`` off the power's
base, and makes a case of it where it is a number over the parameters.

For a parameter the edge's time is not plain in, it is not one. With the
onset written ``on/(wc + wb - wa)``, ``∂N/∂wb`` of the base's numerator still
holds the time, and there was no case for ``wb``, ``wc`` or ``wa``: their
columns were integrated plain, and came back 12%, 0.48% and 0.55% off at
a = 1.1, with nothing said. The same for a derived parameter in the
denominator, for a scale written as a quotient (``dd/dw``, asked for ``dw``),
and for the width under a base that is not linear in the time
(``(1 - s²)^(a-1)``: 0.3% off). Under an opening power the run ended in
"CVODE made no progress".

Such a column is refused now, and only where the power is singular at the
run's own values: at a crossing its parameter moves where the power's base is
0, or at any crossing its parameter moves where the base cannot be asked at
the crossing's time (it reads a counter, as the ``.net`` windows here do, or a
state). A parameter that is in the base and does not move its zero (``kk`` in
``(kk·(1-s))^(a-1)``) keeps its column, which was right.

Every expected value is a central difference of plain runs at ``rtol=1e-12``,
at two steps and extrapolated, on models built with the parameter moved.
Nothing is sampled on a window's edge.
"""

from __future__ import annotations

import re

import bngsim
import numpy as np
import pytest

T = [0.0, 1.0, 2.0, 4.0, 5.0, 5.5, 6.5, 6.75, 6.99, 7.01, 8.0, 10.0]

NET = """begin parameters
    1 k0    0.1
    2 k1    2.0
    3 a     {a}
    4 on    {on}
    5 D     {D}
    6 kdeg  0.3
    7 _rateLaw1 1
{extra}end parameters
begin functions
    1 s() (t-({onset}))/({scale})
    2 prod() k0+if(t>=({onset}),if(t<=(({onset})+({scale})),k1*{shape},0),0)
end functions
begin species
    1 X() 0
    2 Tc() 0
end species
begin reactions
    1 0 1 prod
    2 1 0 kdeg
    3 0 2 _rateLaw1
end reactions
begin groups
    1 t 2
end groups
"""

SHAPES = {
    "closing": "s()*((1-s())^(a-1))",
    "opening": "(s()^(a-1))*(1-s())",
    "both": "(s()^(a-1))*((1-s())^(a-1))",
    "square": "s()*((1-s()^2)^(a-1))",
    "scaled_open": "((kk*s())^(a-1))*(1-s())",
    "scaled_close": "s()*((kk*(1-s()))^(a-1))",
}

# name: (shape, the onset, the scale, the parameters added). Each is the window
# of ``plain`` at these values: an onset of 3 and a width of 4.
QUOTIENT = (("wc", 0.5), ("wb", 1.5), ("wa", 1.0))
WINDOWS = {
    "plain": ("closing", "on", "D", ()),
    "quotient": ("closing", "on/(wc+wb-wa)", "D", QUOTIENT),
    "opening": ("opening", "on/(wc+wb-wa)", "D", QUOTIENT),
    "both": ("both", "on/(wc+wb-wa)", "D", QUOTIENT),
    "derived": ("closing", "on/wd", "D", (("wb", 2.0), ("wa", 1.0), ("wd", "wb-wa"))),
    "product": ("closing", "on/(kk*(wb-wa))", "D", (("kk", 1.0), ("wb", 2.0), ("wa", 1.0))),
    "scale": ("closing", "on", "dd/dw", (("dd", 8.0), ("dw", 2.0))),
    "square": ("square", "on", "D", ()),
    "scaled_open": ("scaled_open", "on", "D", (("kk", 1.0),)),
    "scaled_close": ("scaled_close", "on", "D", (("kk", 1.0),)),
}
START = {"on": 3.0, "D": 4.0}


def _text(window, a, shape=None, **moved) -> str:
    """The ``.net`` of a window, named or given as its four parts, with any of
    its primary parameters at another value."""
    named_shape, onset, scale, added = WINDOWS[window] if isinstance(window, str) else window
    values = dict(START) | dict(added) | moved
    extra = "".join(
        f"   {8 + i} {name} {value if isinstance(value, str) else repr(value)}\n"
        for i, (name, value) in enumerate((name, values[name]) for name, _start in added)
    )
    return NET.format(
        a=a,
        on=repr(values["on"]),
        D=repr(values["D"]),
        onset=onset,
        scale=scale,
        extra=extra,
        shape=SHAPES[named_shape] if shape is None else shape,
    )


def _load(tmp_path, text):
    path = tmp_path / "window.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def _model(tmp_path, window, a, shape=None, **moved):
    return _load(tmp_path, _text(window, a, shape, **moved))


def _column(model, param, times=T, **kw):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param], **kw)
    run = sim.run(sample_times=list(times), rtol=1e-8, atol=1e-10, timeout=120)
    return np.asarray(run.sensitivities)[:, 0, 0]


def _differences(tmp_path, window, a, param, times=T, h=1e-4):
    """dX/dparam at every sample time, from plain runs of models built with
    the parameter moved."""
    added = WINDOWS[window][3] if isinstance(window, str) else window[3]
    start = (dict(START) | dict(added))[param]

    def x(value):
        model = _model(tmp_path, window, a, **{param: value})
        run = bngsim.Simulator(model, method="ode").run(
            sample_times=list(times), rtol=1e-12, atol=1e-14, timeout=120
        )
        return np.asarray(run.species)[:, 0]

    coarse = (x(start + h) - x(start - h)) / (2 * h)
    fine = (x(start + h / 2) - x(start - h / 2)) / h
    return (4 * fine - coarse) / 3


def _worst(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _refused(model, param, params=None, times=T, **kw):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params or [param], **kw)
    with pytest.raises(
        bngsim.SimulationError, match=rf"parameter '{param}' moves.*singular.*\(issue #1003\)"
    ):
        sim.run(sample_times=list(times), rtol=1e-8, atol=1e-10, timeout=120)


def _source(model) -> str:
    from bngsim import _codegen

    return _codegen.generate_sens_from_model(model._core, functional=True, emit_term_scale=True)


EXPORT = "int bngsim_codegen_edge_without_case(int iP, double t,"


def _without_a_case(model, source=None) -> dict[str, str]:
    """``{parameter: its C test}`` as the sensitivity source lists them."""
    source = _source(model) if source is None else source
    if EXPORT not in source:
        return {}
    body = source.split(EXPORT)[1].split("\n}")[0]
    names = [p["name"] for p in model._core.codegen_data()["parameters"]]
    return {
        names[int(index)]: test
        for index, test in re.findall(r"case (\d+): return \((.*)\) \? 1 : 0;", body)
    }


def _with_a_case(model) -> set[str]:
    names = [p["name"] for p in model._core.codegen_data()["parameters"]]
    head = "int bngsim_codegen_comoving_case(int iP, int k, const double* p,"
    body = _source(model).split(head)[1].split("\n}")[0]
    return {names[int(index)] for index in re.findall(r"case (\d+):\n        if \(k == 0\)", body)}


SINGULAR = "((p[2] - 1.0) < 1.0 && (p[2] - 1.0) != 0.0)"

# ─── The refusal ─────────────────────────────────────────────────────────────

NO_CASE = [
    ("quotient", "wb"),  # 12% off: -2.40245 for -2.10906 at t = 7.01
    ("quotient", "wc"),  # 0.48% off
    ("quotient", "wa"),  # 0.55% off
    ("derived", "wd"),
    ("derived", "wb"),  # CVODE made no progress
    ("derived", "wa"),  # 0.55% off
    ("product", "kk"),  # 0.55% off
    ("product", "wb"),  # CVODE made no progress
    ("scale", "dw"),  # 0.37% off
    ("square", "D"),  # 0.30% off
    ("opening", "wb"),  # CVODE made no progress
    ("both", "wb"),  # CVODE made no progress
]


@pytest.mark.parametrize("window, param", NO_CASE)
def test_a_column_with_no_case_for_the_edge_it_moves_is_refused(tmp_path, window, param):
    """Each of these moves the edge of the power, and its shift is not a
    number over the parameters, so it has no case. Its column came back up to
    12% off, or the run ended in "CVODE made no progress"."""
    _refused(_model(tmp_path, window, 1.1), param)


@pytest.mark.parametrize("a", [1.02, 1.5, 1.9])
def test_refused_at_every_exponent_under_one(tmp_path, a):
    """The forcing is unbounded at the edge for every exponent between 0 and
    1. At a = 1.5 the plain column was 2.7e-6 off at rtol 1e-8, which is two
    hundred times the tolerance and not the 12% it is at 1.1; no line between
    the two holds for every model, and #958 draws none either."""
    _refused(_model(tmp_path, "quotient", a), "wb")


def test_the_parameter_is_named_among_others(tmp_path):
    """The columns of ``k1`` and ``on`` are right; ``wb`` is the one refused."""
    _refused(_model(tmp_path, "quotient", 1.1), "wb", params=["k1", "on", "wb"])


def test_a_second_simulator_is_refused_from_the_cached_code(tmp_path):
    """The list is in the compiled code, so a model that takes it from the
    cache is told as the one that built it."""
    _refused(_model(tmp_path, "quotient", 1.1), "wb")
    _refused(_model(tmp_path, "quotient", 1.1), "wb")


def test_the_exponent_is_asked_when_the_run_asks(tmp_path):
    """Built where nothing is singular and set to where the power is: the
    refusal reads the run's values, not those the code was written at."""
    model = _model(tmp_path, "quotient", 3.0)
    model.set_param("a", 1.1)
    _refused(model, "wb")


def test_an_exponent_set_back_to_one_or_more_runs(tmp_path):
    """Control. The other way round: built where the power is singular, and
    run where it is not."""
    model = _model(tmp_path, "quotient", 1.1)
    model.set_param("a", 3.0)
    want = _differences(tmp_path, "quotient", 3.0, "wb")
    assert _worst(_column(model, "wb"), want) < 5e-6


@pytest.mark.parametrize(
    "window, param", [("quotient", "on"), ("quotient", "D"), ("scale", "dd"), ("square", "on")]
)
def test_a_column_that_has_its_case_is_carried(tmp_path, window, param):
    """Control. The edge's time is plain in these, and each has its case."""
    want = _differences(tmp_path, window, 1.1, param)
    assert _worst(_column(_model(tmp_path, window, 1.1), param), want) < 5e-6


@pytest.mark.parametrize("a", [2.0, 3.0])
@pytest.mark.parametrize("window, param", [("quotient", "wb"), ("scale", "dw"), ("square", "D")])
def test_a_power_that_is_not_singular_keeps_its_column(tmp_path, window, param, a):
    """Control. With an exponent of 1 or more nothing is unbounded at the
    edge, and the plain column is right."""
    want = _differences(tmp_path, window, a, param)
    assert _worst(_column(_model(tmp_path, window, a), param), want) < 5e-6


@pytest.mark.parametrize("window", ["quotient", "opening"])
def test_an_exponent_of_exactly_zero_keeps_its_column(tmp_path, window):
    """Control. At a = 1 the power is the constant 1: a window with a step at
    its edge, which the jump at a moved crossing carries."""
    want = _differences(tmp_path, window, 1.0, "wb")
    assert _worst(_column(_model(tmp_path, window, 1.0), "wb"), want) < 5e-6


@pytest.mark.parametrize("window", ["scaled_open", "scaled_close"])
def test_a_parameter_in_the_base_that_does_not_move_the_edge(tmp_path, window):
    """Control. ``kk`` is in the base ``kk·(1-s)`` and its zero is where it
    was: ``∂N/∂kk`` is 0 where the numerator is. The column's forcing is the
    power times a number, and it is not listed."""
    model = _model(tmp_path, window, 1.1)
    assert "kk" not in _without_a_case(model)
    want = _differences(tmp_path, window, 1.1, "kk")
    assert _worst(_column(model, "kk"), want) < 5e-6


def test_a_run_that_reaches_no_crossing_the_parameter_moves(tmp_path):
    """Control. The window opens at 3. A run that ends at 2 has no crossing
    for ``wb`` to move, and its column, which is 0, is returned."""
    times = [0.0, 0.5, 1.0, 2.0]
    got = _column(_model(tmp_path, "quotient", 1.1), "wb", times=times)
    assert np.all(got == 0.0)


def test_a_run_that_ends_inside_a_window_on_a_counter_is_refused(tmp_path):
    """The run ends at 5, before the closing edge at 7. ``wb`` moves the
    opening at 3, and the base of a power of a counter cannot be asked where
    the counter is at a crossing, so the column is refused at any crossing its
    parameter moves."""
    _refused(_model(tmp_path, "quotient", 1.1), "wb", times=[0.0, 1.0, 4.0, 5.0])


# ─── Through SBML, and beside an event ───────────────────────────────────────


def _antimony(a, event=False, wb=2.0, shape="closing"):
    """The window on the literal time, its onset over ``wb - wa``."""
    onset = "(on/(wb - wa))"
    s = f"((time - {onset})/D)"
    pulse = {"closing": f"{s}*(1 - {s})^(a - 1)", "opening": f"{s}^(a - 1)*(1 - {s})"}[shape]
    return bngsim.Model.from_antimony_string(
        f"species X; X = 0; k0 = 0.1; k1 = 2; a = {a}; on = 3; D = 4; kdeg = 0.3; "
        f"wb = {wb!r}; wa = 1; q = 0\n"
        f"J1: -> X; k0 + piecewise(piecewise(k1*{pulse}, "
        f"time <= {onset} + D, 0), time >= {onset}, 0)\n"
        "J2: X -> ; kdeg*X\n" + ("E1: at (time > 5): q = 1\n" if event else "")
    )


def _antimony_differences(a, times, shape="closing", h=1e-4):
    def x(wb):
        run = bngsim.Simulator(_antimony(a, wb=wb, shape=shape), method="ode").run(
            sample_times=list(times), rtol=1e-12, atol=1e-14, timeout=120
        )
        return np.asarray(run.species)[:, 0]

    coarse = (x(2 + h) - x(2 - h)) / (2 * h)
    fine = (x(2 + h / 2) - x(2 - h / 2)) / h
    return (4 * fine - coarse) / 3


def _refused_in_sbml(model, param, issue=1003):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param])
    with pytest.raises(bngsim.SimulationError, match=rf"singular.*\(issue #{issue}\)"):
        sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)


@pytest.mark.parametrize("shape", ["closing", "opening"])
@pytest.mark.parametrize("param", ["wb", "wa"])
def test_refused_through_sbml(shape, param):
    """On the literal time the base is asked at the crossing: 0 at the close
    of the closing power and at the opening of the opening one."""
    _refused_in_sbml(_antimony(1.1, shape=shape), param)


@pytest.mark.parametrize("rtol", [1e-4, 1e-10, 1e-12])
def test_refused_before_the_run_is_taken(rtol):
    """Refused where the run starts, for the switch times it has. Asked only
    at the crossing, the plain column is first carried up to the closing
    edge, where at a tight tolerance the run ends in a solver error."""
    sim = bngsim.Simulator(_antimony(1.1), method="ode", sensitivity_params=["wb"])
    with pytest.raises(bngsim.SimulationError, match=r"singular.*\(issue #1003\)"):
        sim.run(sample_times=T, rtol=rtol, atol=rtol * 1e-2, timeout=120)


def test_the_base_is_asked_to_rounding():
    """The edge's time ``on/(wb - wa) + D`` is no round number here, and the
    base's numerator there is what rounding leaves of 0, not 0."""
    model = bngsim.Model.from_antimony_string(
        "species X; X = 0; k0 = 0.1; k1 = 2; a = 1.1; on = 3.1; D = 4.3; kdeg = 0.3; "
        "wb = 2.3; wa = 0.7\n"
        "J1: -> X; k0 + piecewise(piecewise(k1*((time - on/(wb - wa))/D)"
        "*(1 - (time - on/(wb - wa))/D)^(a - 1), time <= on/(wb - wa) + D, 0), "
        "time >= on/(wb - wa), 0)\n"
        "J2: X -> ; kdeg*X\n"
    )
    _refused_in_sbml(model, "wb")


def _only_a_root(a, wb=2.0):
    """The window opens where a state crosses, and never closes: its edge is
    no switch time the run has before it starts."""
    return bngsim.Model.from_antimony_string(
        f"species X, Z; X = 1; Z = 0; k0 = 0.1; k1 = 1; a = {a}; on = 10; D = 20; "
        f"wb = {wb!r}; wa = 1\n"
        "J0: -> X; (k0 + k1*piecewise(((time - on/(wb - wa))/D)^(a - 1), "
        "time - Z >= on/(wb - wa), 0))*X\n"
    )


@pytest.mark.parametrize("a", [1.9, 1.999])
def test_an_edge_that_is_only_found_as_a_root_is_refused_at_the_root(a):
    """Nothing is known of this edge before the run, so the column is refused
    where the root is found. (At a = 1.5 the plain column does not get that
    far: the run ends short of the root in CVODE's no-progress error, as it
    did.)"""
    sim = bngsim.Simulator(_only_a_root(a), method="ode", sensitivity_params=["wb"])
    with pytest.raises(bngsim.SimulationError, match=r"singular.*\(issue #1003\)"):
        sim.run(sample_times=[0.0, 5.0, 12.0, 18.0], rtol=1e-8, atol=1e-10, timeout=120)


def test_an_edge_that_is_only_found_as_a_root_where_nothing_is_singular():
    """Control."""
    times = [0.0, 5.0, 12.0, 18.0]

    def x(wb):
        run = bngsim.Simulator(_only_a_root(3.0, wb), method="ode").run(
            sample_times=times, rtol=1e-12, atol=1e-14, timeout=120
        )
        return np.asarray(run.species)[:, list(run.species_names).index("X")]

    h = 1e-4
    coarse = (x(2 + h) - x(2 - h)) / (2 * h)
    fine = (x(2 + h / 2) - x(2 - h / 2)) / h
    sim = bngsim.Simulator(_only_a_root(3.0), method="ode", sensitivity_params=["wb"])
    run = sim.run(sample_times=times, rtol=1e-8, atol=1e-10, timeout=120)
    got = np.asarray(run.sensitivities)[:, list(run.species_names).index("X"), 0]
    assert _worst(got, (4 * fine - coarse) / 3) < 1e-5


def test_on_the_literal_time_a_crossing_that_is_not_the_edge_is_let_by():
    """Control. The run ends at 5, inside the window. ``wb`` moves the opening
    at 3, where the base ``1 - s`` of the closing power is 1: that crossing is
    not the power's edge, and up to 5 nothing is unbounded."""
    times = [0.0, 1.0, 2.0, 3.5, 4.0, 5.0]
    got = _column(_antimony(1.1), "wb", times=times)
    assert _worst(got, _antimony_differences(1.1, times)) < 5e-6


def _never_zero(t0=3.3):
    """``(Ca/S)^m`` with ``Ca`` switched on at ``t0``, as in BIOMD0000000276.
    A singular power by its shape: its exponent reads the time, and nothing
    says its base is never 0. At ``t0`` the base is ``Ca0/S``."""
    return bngsim.Model.from_antimony_string(
        f"species X; X = 0; Ca0 = 1.255; Ca1 = 0.18; t0 = {t0!r}; alpha = 0.4; S = 1.1; "
        "m1 = 0.3; m2 = 0.4; beta = 2; R = 1.2; A = 2; B = 0.5; kdeg = 0.3\n"
        "Ca := piecewise(Ca0, time < t0, Ca0 - Ca1*(1 - exp(-alpha*(time - t0))))\n"
        "m := m1/(1 + exp(-beta*(R - Ca))) + m2\n"
        "J1: -> X; (A - B)/(1 + (Ca/S)^m) + B\n"
        "J2: X -> ; kdeg*X\n"
    )


def _switched_on(ca0, t0=3.3):
    """``(Ca/S)^m`` with ``Ca`` at ``Ca0`` up to ``t0`` and rising from it.
    With ``Ca0 = 0`` the base is 0 at ``t0``, the power's opening edge. The
    shift of ``t0`` holds the condition and is dropped: ``t0`` has no case."""
    return bngsim.Model.from_antimony_string(
        f"species X; X = 0; Ca0 = {ca0!r}; Ca1 = 0.18; t0 = {t0!r}; alpha = 0.4; S = 1.1; "
        "m = 0.5; A = 2; kdeg = 0.3\n"
        "Ca := piecewise(Ca0, time < t0, Ca0 + Ca1*(1 - exp(-alpha*(time - t0))))\n"
        "J1: -> X; A*(Ca/S)^m\n"
        "J2: X -> ; kdeg*X\n"
    )


def test_a_base_that_is_zero_where_its_parameter_switches_it_on():
    model = _switched_on(0.0)
    assert "t0" in _without_a_case(model)
    _refused_in_sbml(model, "t0")


def test_the_base_is_asked_at_the_runs_values():
    """Built where the base is 0.5 at ``t0`` and set to where it is 0."""
    model = _switched_on(0.5)
    model.set_param("Ca0", 0.0)
    _refused_in_sbml(model, "t0")


def test_a_base_that_is_not_zero_where_its_parameter_switches_it_on():
    """Control. The other way round: built at 0 and set to 0.5, where the
    crossing at ``t0`` is no edge of the power. No sample is on ``t0``."""

    def x(t0):
        run = bngsim.Simulator(_switched_on(0.5, t0), method="ode").run(
            sample_times=T, rtol=1e-12, atol=1e-14, timeout=120
        )
        return np.asarray(run.species)[:, 0]

    h = 1e-4
    coarse = (x(3.3 + h) - x(3.3 - h)) / (2 * h)
    fine = (x(3.3 + h / 2) - x(3.3 - h / 2)) / h
    model = _switched_on(0.0)
    model.set_param("Ca0", 0.5)
    assert _worst(_column(model, "t0"), (4 * fine - coarse) / 3) < 5e-6


def test_a_power_whose_base_is_never_zero_is_listed_with_its_base():
    """``t0`` moves the crossing at ``t0`` and has no case, so it is listed,
    and the test it is listed with asks the base at the crossing's time."""
    listed = _without_a_case(_never_zero())
    assert "t0" in listed
    assert listed["t0"].startswith("(!(fabs(") and "> 1e-9 * (fabs(" in listed["t0"]


def test_a_power_whose_base_is_never_zero_keeps_its_columns():
    """Control. The column of ``t0``, which is right, is returned: the
    crossing at ``t0`` is not an edge of the power. No sample is on ``t0``."""

    def x(t0):
        run = bngsim.Simulator(_never_zero(t0), method="ode").run(
            sample_times=T, rtol=1e-12, atol=1e-14, timeout=120
        )
        return np.asarray(run.species)[:, 0]

    h = 1e-4
    coarse = (x(3.3 + h) - x(3.3 - h)) / (2 * h)
    fine = (x(3.3 + h / 2) - x(3.3 - h / 2)) / h
    assert _worst(_column(_never_zero(), "t0"), (4 * fine - coarse) / 3) < 5e-6


@pytest.mark.parametrize("param", ["wb", "wa"])
def test_refused_beside_an_event(param):
    """A model with an event has no frames at all, and a column that had a
    case is refused there for that (issue #958). One that never had a case
    was plain with or without the event, and is refused for this."""
    _refused_in_sbml(_antimony(1.1, event=True), param)


def test_beside_an_event_a_column_with_a_case_is_refused_for_the_event():
    """Control. ``on`` has its case, and the event takes its frame."""
    _refused_in_sbml(_antimony(1.1, event=True), "on", issue=958)


def test_through_sbml_a_power_that_is_not_singular_keeps_its_column():
    """Control."""
    assert _worst(_column(_antimony(3.0), "wb"), _antimony_differences(3.0, T)) < 5e-6


# ─── What the generator writes ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "window, listed, cased",
    [
        ("quotient", {"wc", "wb", "wa"}, {"on", "D"}),
        ("opening", {"wc", "wb", "wa"}, {"on"}),
        ("derived", {"wd", "wb", "wa"}, {"on", "D"}),
        ("product", {"kk", "wb", "wa"}, {"on", "D"}),
        ("scale", {"dw"}, {"on", "dd"}),
        ("square", {"D"}, {"on"}),
    ],
)
def test_the_parameters_the_source_lists(tmp_path, window, listed, cased):
    """Each parameter that moves the edge is listed or has a case, and none
    is both."""
    model = _model(tmp_path, window, 1.1)
    assert _without_a_case(model) == dict.fromkeys(listed, SINGULAR)
    assert _with_a_case(model) == cased


@pytest.mark.parametrize("window", ["plain", "scaled_open", "scaled_close"])
def test_a_model_in_which_every_shift_is_a_number_lists_nothing(tmp_path, window):
    """Control. Nothing is listed, and the export is not written at all: the
    source of such a model is what it was."""
    assert EXPORT not in _source(_model(tmp_path, window, 1.1))


def test_an_exponent_that_is_a_number_is_listed_at_every_value(tmp_path):
    """``(1-s)^0.1`` is singular whatever the parameters are."""
    model = _model(tmp_path, "quotient", 3.0, shape="s()*((1-s())^0.1)")
    assert _without_a_case(model) == dict.fromkeys({"wc", "wb", "wa"}, "1")
    _refused(model, "wb")


CHOSEN = ("closing", "on/(wc+wb-wa)", "D", (*QUOTIENT, ("a2", 3.0), ("tsw", 5.0)))


def _chosen(tmp_path, a, a2):
    """The exponent is ``a`` before ``tsw`` and ``a2`` after it, as a year's
    exponent is chosen in a model of several seasons."""
    shape = "s()*((1-s())^(if(t<tsw,a,a2)-1))"
    return _model(tmp_path, CHOSEN, a, shape=shape, a2=a2)


def test_an_exponent_chosen_by_a_condition_is_asked_branch_by_branch(tmp_path):
    """Each value the exponent can take is asked at the run's parameters,
    whichever the run is on at the edge."""
    one = "((p[2] - 1.0) < 1.0 && (p[2] - 1.0) != 0.0)"
    other = "((p[10] - 1.0) < 1.0 && (p[10] - 1.0) != 0.0)"
    listed = _without_a_case(_chosen(tmp_path, 3.0, 3.0))
    assert set(listed) == {"wc", "wb", "wa"}
    assert set(listed["wb"].split(" || ")) == {one, other}
    _refused(_chosen(tmp_path, 3.0, 1.1), "wb")
    _refused(_chosen(tmp_path, 1.1, 3.0), "wb")


def test_an_exponent_chosen_by_a_condition_and_singular_on_no_branch(tmp_path):
    """Control. Neither value is under 1, and the column is returned. A test
    that could not ask a chosen exponent would refuse every such model at
    every value: a model of several seasons with no singular power in any."""
    model = _chosen(tmp_path, 3.0, 2.5)
    assert np.all(np.isfinite(_column(model, "wb")))


def test_an_exponent_chosen_among_numbers_of_one_or_more_is_not_listed(tmp_path):
    """Control. ``if(t<tsw, 2, 3) - 1`` is no number, so the power is singular
    by its shape, and neither value is under 1: nothing to list."""
    model = _model(tmp_path, CHOSEN, 3.0, shape="s()*((1-s())^(if(t<tsw,2,3)-1))")
    assert _without_a_case(model) == {}
    assert np.all(np.isfinite(_column(model, "wb")))


SEASONS = ("closing", "if(t<tsw,on,on2)", "D", (("on2", 30.0), ("tsw", 20.0)))


@pytest.mark.parametrize("a", [1.1, 1.5])
def test_an_onset_chosen_by_a_condition_on_a_parameter(tmp_path, a):
    """The onset is ``on`` up to ``tsw = 20`` and ``on2`` after it, as a
    season's onset is chosen in a model of several years whose year ends are
    parameters. A shift of 1 is read off for ``on``, and it does not remove
    the power, whose base holds the condition: ``on`` has no case. Its run
    ended in CVODE's no-progress error at a = 1.1 and returned at 1.5."""
    model = _model(tmp_path, SEASONS, a)
    assert _without_a_case(model) == {"on": SINGULAR, "on2": SINGULAR}
    _refused(model, "on")


def test_an_onset_chosen_by_a_condition_where_nothing_is_singular(tmp_path):
    """Control."""
    for param in ("on", "D"):
        want = _differences(tmp_path, SEASONS, 3.0, param)
        assert _worst(_column(_model(tmp_path, SEASONS, 3.0), param), want) < 5e-6


def test_a_listed_parameter_that_moves_no_crossing_keeps_its_column(tmp_path):
    """Control. ``kq`` is under the power, in ``(1 - s^kq)^(a-1)``, and the
    zero of that base is at ``s = 1`` whatever ``kq`` is. The base is not
    linear in the time, so that is not worked out and ``kq`` is listed; it
    moves no crossing, and nothing asks."""
    window = ("closing", "on", "D", (("kq", 2.0),))
    shape = "s()*((1-s()^kq)^(a-1))"
    model = _model(tmp_path, window, 1.1, shape=shape)
    assert "kq" in (_without_a_case(model) or {"kq": ""})

    def x(kq):
        run = bngsim.Simulator(
            _model(tmp_path, window, 1.1, shape=shape, kq=kq), method="ode"
        ).run(sample_times=T, rtol=1e-12, atol=1e-14, timeout=120)
        return np.asarray(run.species)[:, 0]

    h = 1e-4
    coarse = (x(2 + h) - x(2 - h)) / (2 * h)
    fine = (x(2 + h / 2) - x(2 - h / 2)) / h
    assert _worst(_column(model, "kq"), (4 * fine - coarse) / 3) < 5e-5


def test_two_windows_and_a_case_for_one_of_them(tmp_path):
    """``D`` is the plain width of the first window and is under a square in
    the second. It has its case for the first and none for the second, and a
    parameter with any edge it has no case for is listed."""
    text = (
        _text(("closing", "on", "D", (("on2", 20.0),)), 1.1)
        .replace(
            "end functions",
            "    3 s2() (t-on2)/D\n"
            "    4 prod2() if(t>=on2,if(t<=(on2+D),k1*s2()*((1-s2()^2)^(a-1)),0),0)\n"
            "end functions",
        )
        .replace("end reactions", "    4 0 1 prod2\nend reactions")
    )
    model = _load(tmp_path, text)
    assert _without_a_case(model) == {"D": SINGULAR}
    assert _with_a_case(model) == {"on", "D", "on2"}
    _refused(model, "D")


# ─── A plan that ends early ──────────────────────────────────────────────────


def _ending_early(monkeypatch, error):
    from bngsim import _codegen

    def ends(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(_codegen, "_functional_comoving_plan", ends)


@pytest.mark.parametrize("window", ["plain", "quotient"])
@pytest.mark.parametrize("budget", [True, False])
def test_a_plan_that_ends_early_lists_by_name(tmp_path, monkeypatch, window, budget):
    """The plan of the cases runs under the derivation budget, and where it
    ran out, or raised, the model had no case: every column that needed one
    was plain (dX/d(on) 0.45% off) in a library that was then kept. Every
    parameter a singular power's base reads is listed then, ``on`` and ``D``
    with the rest."""
    from bngsim._jacobian import _DerivationBudgetExceeded

    model = _model(tmp_path, window, 1.1)
    _ending_early(monkeypatch, _DerivationBudgetExceeded() if budget else RuntimeError("plan"))
    source = _source(model)
    assert "bngsim_codegen_comoving_case" not in source
    in_base = {"on", "D"} | {name for name, _value in WINDOWS[window][3]}
    assert _without_a_case(model, source) == dict.fromkeys(in_base, SINGULAR)


def test_by_name_a_derived_parameter_and_what_it_is_defined_from(tmp_path, monkeypatch):
    """``wd = wb - wa`` is in the base: it is listed, and so are ``wb`` and
    ``wa``, which move the edge through it."""
    model = _model(tmp_path, "derived", 1.1)
    _ending_early(monkeypatch, RuntimeError("plan"))
    assert set(_without_a_case(model)) == {"on", "D", "wd", "wb", "wa"}


def test_by_name_a_law_that_is_not_read_lists_every_parameter(tmp_path, monkeypatch):
    """Where a rate law cannot be read, nothing says it has no such power:
    every parameter is listed, at every value."""
    import sys

    from bngsim import _jacobian

    model = _model(tmp_path, "plain", 1.1)
    _ending_early(monkeypatch, RuntimeError("plan"))
    real = _jacobian._exprtk_to_sympy

    def unread(text, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_edge_parameters_by_name":
            return None
        return real(text, *args, **kwargs)

    monkeypatch.setattr(_jacobian, "_exprtk_to_sympy", unread)
    names = [p["name"] for p in model._core.codegen_data()["parameters"]]
    assert _without_a_case(model) == dict.fromkeys(names, "1")


def test_by_name_a_model_with_no_singular_power_lists_nothing(tmp_path, monkeypatch):
    """Control. A window that closes as a step has no such power."""
    model = _model(tmp_path, "plain", 1.1, shape="s()")
    _ending_early(monkeypatch, RuntimeError("plan"))
    assert EXPORT not in _source(model)


def test_derived_axes_that_run_out_of_budget_leave_the_derived_parameter_listed(
    tmp_path, monkeypatch
):
    """The derived parameters are asked after the primaries, and where the
    budget runs out there the primaries keep their cases. A derived onset
    that would have had its own is listed, where it was plain."""
    from bngsim import _codegen
    from bngsim._jacobian import _DerivationBudgetExceeded

    model = _model(tmp_path, ("closing", "ond", "D", (("lam", 1.5), ("ond", "2*lam"))), 1.1)
    assert _with_a_case(model) == {"D", "lam", "ond"}
    assert _without_a_case(model) == {}

    def out_of_budget(*_args, **_kwargs):
        raise _DerivationBudgetExceeded

    monkeypatch.setattr(_codegen, "_comoving_derived_axes", out_of_budget)
    assert _with_a_case(model) == {"D", "lam"}
    assert _without_a_case(model) == {"ond": SINGULAR}


# ─── A model that is given no case at all (issue #749) ───────────────────────

PULSE = (
    "k0 + piecewise(piecewise(k1*(({clock}-on)/D)^(a-1)*(1-({clock}-on)/D), "
    "{clock} <= on + D, 0), {clock} >= on, 0)"
)
# Tot := T + X with T an amount in a compartment whose size is a parameter: T's
# weight in Tot is that size, which a shift has no number for, and a rate law
# reads Tot.
LIVE_SUM = """
model counter
  compartment C = 1;
  substanceOnly species T in C;
  species X in C, Tot in C;
  X = 0; T = 0;
  Tot := T + X;
  k0 = 0.1; k1 = 2; a = {a}; on = 3; D = 4; kdeg = 0.3; kd2 = 0.05;
  Jt: -> T; C;
  J1: -> X; {pulse};
  J2: X -> ; kdeg*X;
  J3: X -> ; kd2*X*Tot;
end
"""
# The pulse is gated on a compartment that grows at 1, and a transfer puts X's
# row over that compartment's live size.
CLOCK_COMPARTMENT = """
model growing
  compartment C1 = 1, C2 = 2;
  C1' = 1;
  species X in C1, Y in C2;
  X = 0; Y = 1;
  k0 = 0.1; k1 = 2; a = {a}; on = 3; D = 4; kdeg = 0.3; kt = 0.4;
  Jx: Y => X; kt*Y;
  J1: -> X; {pulse};
  J2: X -> ; kdeg*X;
end
"""
NO_CASES = {
    "a sum weighted by a live size": (LIVE_SUM, "T", [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
    "a compartment that is the clock": (
        CLOCK_COMPARTMENT,
        "C1",
        [0.0, 1.0, 2.5, 4.0, 5.0, 6.5, 8.0, 10.0],
    ),
}


def _no_cases(kind, a):
    text, clock, times = NO_CASES[kind]
    return text.format(a=a, pulse=PULSE.format(clock=clock)), times


@pytest.mark.parametrize("kind", list(NO_CASES))
def test_a_model_with_no_case_refuses_the_column_that_needed_one(kind):
    """Two kinds of model are given no comoving case, because no case writes
    how their rows move with the clock (issue #749). Their plain columns are
    right where nothing is singular, which is what #749 measured, at a = 3.
    At a = 1.5 the onset's column needs its frame, and the run ended in a
    solver error (CV_ERR_FAILURE) that named a non-finite right-hand side."""
    text, times = _no_cases(kind, 1.5)
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["on"]
    )
    with pytest.raises(
        bngsim.SimulationError, match=r"parameter 'on' moves.*singular.*\(issue #1003\)"
    ):
        sim.run(sample_times=times, rtol=1e-8, atol=1e-10, timeout=120)


@pytest.mark.parametrize("kind", list(NO_CASES))
@pytest.mark.parametrize("a, param", [(3.0, "on"), (1.5, "D"), (1.5, "k1")])
def test_a_model_with_no_case_keeps_the_columns_that_need_none(kind, a, param):
    """Control. The onset's column where nothing is singular, and the columns
    of ``D`` and ``k1``, which do not move the edge the pulse opens at."""
    text, times = _no_cases(kind, a)
    start = {"on": 3.0, "D": 4.0, "k1": 2.0}[param]

    def x(value):
        model = bngsim.Model.from_antimony_string(text)
        model.set_param(param, value)
        run = bngsim.Simulator(model, method="ode").run(
            sample_times=times, rtol=1e-12, atol=1e-14, timeout=120
        )
        return np.asarray(run.species)[:, list(run.species_names).index("X")]

    def central(h):
        return (x(start + h) - x(start - h)) / (2 * h)

    want = (4.0 * central(1.5e-3) - central(3e-3)) / 3.0
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=[param]
    )
    run = sim.run(sample_times=times, rtol=1e-8, atol=1e-10, timeout=120)
    got = np.asarray(run.sensitivities)[:, list(run.species_names).index("X"), 0]
    assert _worst(got, want) < 1e-5


# ─── Whether a parameter moves the zero of a base ────────────────────────────


def test_whether_a_parameter_moves_the_zero_of_a_numerator():
    import sympy as sp
    from bngsim._codegen import _edge_is_not_moved

    t, on, D, kk, w = sp.symbols("t on D kk w")

    def moved(numerator, param):
        return not _edge_is_not_moved(
            numerator, sp.diff(numerator, param), t, sp.diff(numerator, t), sp
        )

    assert not moved(kk * (t - on), kk)  # a factor of the base
    assert not moved(kk * (D + on - t), kk)
    assert moved(kk * (t - on), on)
    assert moved((D - t) * w + on, w)  # the denominator of the onset
    assert moved(D**2 - (t - on) ** 2, D)  # not linear in the time: not worked out
    assert moved(kk * (D**2 - (t - on) ** 2), kk)  # likewise, though it does not move it
