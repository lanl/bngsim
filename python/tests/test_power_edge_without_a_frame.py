"""A window edge that is a singular power, in a column that has no frame for it
(issues #958, #948).

A window ``k1·s·(1-s)^(a-1)`` with ``s = (t-on)/D`` closes at ``on + D`` as a
power, and ``k1·s^(a-1)·(1-s)`` opens at ``on`` as one. For 1 < a < 2 the
forcing of a column that moves the edge is unbounded there, ``(1-s)^(a-2)``
before a close and ``s^(a-2)`` after an opening, and no step resolves it. Such a
column is integrated in a frame that moves with the edge (issues #545, #760).

Two kinds of column have no frame:

- any column, in a model with an event. An event restarts the integration under
  the column, so a run with one keeps every column plain. dX/dD came back 0.4%
  off beside an event that assigns a parameter nothing reads, flat in the
  tolerance, and an opening edge ended in a solver error that named neither
  the edge nor the event (issue #958);
- a column that moves a counter clock itself, by the counter's initial value or
  its rate. The cases are of the parameters the power is written in. dX/dT0
  came back 0.2% to 0.4% off under a closing power, and the run stalled under an
  opening one (issue #948).

Both are refused now, at the crossing where the column would have entered its
frame, and only where a power is singular at the run's own values: at a = 3
nothing is unbounded and the plain column is right.

X is linear in the pulse, so every expected value is a quadrature of the pulse
against ``e^(-kdeg·(T-t))``, differenced in the parameter at two steps and
extrapolated. Nothing is sampled on a window's edge.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from scipy.integrate import quad

K0, K1, KDEG, ON, WIDTH = 0.1, 2.0, 0.3, 3.0, 4.0
T = [0.0, 2.0, 5.0, 6.5, 6.9, 7.1, 8.0, 10.0]

SHAPES = {
    "closing": ("{s}*(1-{s})^(a-1)", lambda u, a: u * (1 - u) ** (a - 1)),
    "opening": ("{s}^(a-1)*(1-{s})", lambda u, a: u ** (a - 1) * (1 - u)),
}


def _x(shape, a, t_end, lo, width, rate=1.0):
    """X at ``t_end`` for a window the clock opens at time ``lo`` and that is
    ``width`` long on the clock, which runs at ``rate``."""
    f = SHAPES[shape][1]
    hi = lo + width / rate
    base = K0 * (1 - np.exp(-KDEG * t_end)) / KDEG
    if t_end <= lo:
        return base
    val, _ = quad(
        lambda t: K1 * f(rate * (t - lo) / width, a) * np.exp(-KDEG * (t_end - t)),
        lo,
        min(hi, t_end),
        epsabs=1e-13,
        epsrel=1e-13,
        limit=400,
    )
    return base + val


def _slope(fn):
    """d fn/dh at 0, two central differences extrapolated."""

    def d(h):
        return (fn(h) - fn(-h)) / (2 * h)

    return (4 * d(1e-4) - d(2e-4)) / 3


def _expected(shape, a, param, times=T):
    """dX/d``param`` at each of ``times``, for the window on literal time, open
    from 3 to 7, or on a counter that starts at T0 = 1 and runs at r = 1 with
    its onset at 4, which is open over the same times."""
    moved = {
        "on": lambda h, t: _x(shape, a, t, ON + h, WIDTH),
        "D": lambda h, t: _x(shape, a, t, ON, WIDTH + h),
        # A counter that starts h later reaches every threshold h sooner.
        "T0": lambda h, t: _x(shape, a, t, ON - h, WIDTH),
        # One that runs at 1 + h is on - T0 = 3 further along at 3/(1 + h).
        "r": lambda h, t: _x(shape, a, t, ON / (1 + h), WIDTH, rate=1 + h),
        "k1": lambda h, t: _x(shape, a, t, ON, WIDTH)
        + h * (_x(shape, a, t, ON, WIDTH) - K0 * (1 - np.exp(-KDEG * t)) / KDEG) / K1,
    }[param]
    return np.array([_slope(lambda h, t=t: moved(h, t)) for t in times])


# ─── A model with an event (issue #958) ─────────────────────────────────────


def _with_event(shape, a, event=True, extra=""):
    """The window on literal time. The event assigns a parameter nothing reads."""
    law = SHAPES[shape][0].format(s="((time-on)/D)")
    return bngsim.Model.from_antimony_string(
        f"species X; X = 0; k0 = {K0}; k1 = {K1}; a = {a}; on = {ON}; D = {WIDTH}; "
        f"kdeg = {KDEG}; q = 0; kj = 1.5\n"
        f"J1: -> X; k0 + piecewise(piecewise(k1*{law}, time < on + D, 0), time >= on, 0)"
        f"{extra}\n"
        "J2: X -> ; kdeg*X\n" + ("E1: at (time > 5): q = 1\n" if event else "")
    )


def _run(model, params, rtol=1e-8, **kw):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params, **kw)
    out = sim.run(sample_times=T, rtol=rtol, atol=rtol * 1e-2, timeout=120)
    names = list(out.species_names)
    return np.asarray(out.sensitivities)[:, names.index("X") if "X" in names else 0, :]


def _refused(model, params, issue, rtol=1e-8, **kw):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params, **kw)
    with pytest.raises(bngsim.SimulationError, match=rf"singular.*\(issue #{issue}\)"):
        sim.run(sample_times=T, rtol=rtol, atol=rtol * 1e-2, timeout=120)


@pytest.mark.parametrize(
    "shape, params",
    [
        ("closing", ["D"]),
        ("closing", ["on"]),
        ("closing", ["k1", "D"]),
        ("opening", ["on"]),
    ],
)
def test_a_singular_edge_beside_an_event_is_refused(shape, params):
    """An event that assigns a parameter nothing reads turns the frames off.
    Closing, dX/dD came back 1.0789 for 1.0830 and dX/d(on) 0.6989 for 0.7030,
    at any tolerance. Opening, the run ended in CV_FIRST_SRHSFUNC_ERR: the
    plain column's right-hand side is not finite on the edge."""
    _refused(_with_event(shape, 1.1), params, 958)


@pytest.mark.parametrize("shape, param", [("closing", "D"), ("closing", "on"), ("opening", "on")])
def test_without_the_event_the_column_has_its_frame(shape, param):
    """Control. The same model with no event is integrated in its frame."""
    got = _run(_with_event(shape, 1.1, event=False), [param])[:, 0]
    np.testing.assert_allclose(got, _expected(shape, 1.1, param), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("shape, param", [("closing", "D"), ("closing", "on"), ("opening", "on")])
def test_a_power_that_is_not_singular_runs_beside_an_event(shape, param):
    """Control. At a = 3 the window closes and opens as a square, nothing is
    unbounded, and the plain column is right: whether a power is singular is
    asked at the run's own values."""
    got = _run(_with_event(shape, 3.0), [param])[:, 0]
    np.testing.assert_allclose(got, _expected(shape, 3.0, param), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("shape", ["closing", "opening"])
def test_a_column_that_does_not_move_the_window_runs_beside_an_event(shape):
    """Control. k1 scales the pulse and moves neither edge."""
    got = _run(_with_event(shape, 1.1), ["k1"])[:, 0]
    np.testing.assert_allclose(got, _expected(shape, 1.1, "k1"), rtol=2e-5, atol=2e-7)


def test_a_run_that_ends_before_the_window_opens_is_not_refused():
    """Control. No edge is reached, so no column would have entered a frame."""
    sim = bngsim.Simulator(_with_event("closing", 1.1), method="ode", sensitivity_params=["D"])
    out = sim.run(sample_times=[0.0, 1.0, 2.5], rtol=1e-8, atol=1e-10)
    assert np.all(np.asarray(out.sensitivities) == 0.0)


def test_a_crossing_the_parameter_moves_at_another_rate_is_not_the_frames():
    """Control. A step at D/2 moves with D at a half, and the window's edge at
    1: the D column has a case for the edge and none for the step, where it
    would have stayed plain with its frames on. The run ends before the window
    opens."""
    model = _with_event("closing", 1.1, extra=" + piecewise(kj, time >= D/2, 0)")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["D"])
    times = [0.0, 1.0, 2.25, 2.5]
    out = sim.run(sample_times=times, rtol=1e-9, atol=1e-11)
    # X gains kj from D/2 on, so dX/dD = -(kj/2)·e^(-kdeg·(t - D/2)) past it.
    want = [0.0 if t < WIDTH / 2 else -0.75 * np.exp(-KDEG * (t - WIDTH / 2)) for t in times]
    np.testing.assert_allclose(np.asarray(out.sensitivities)[:, 0, 0], want, rtol=1e-6, atol=1e-9)


def test_a_step_on_the_closing_edge_beside_an_event_is_refused():
    """Issue #949's crossing, a step on the instant the window closes, in a
    model with an event. The frames are off, so the #949 refusal was never
    reached, and dX/dD at t = 8 came back 1.0784 for 0.8047."""
    model = _with_event("closing", 1.1, extra=" + piecewise(kj, time >= 7, 0)")
    _refused(model, ["D"], 958)


def test_a_step_on_the_closing_edge_is_refused_with_no_event():
    """Control. The same crossing with the frames on is issue #949's refusal."""
    model = _with_event("closing", 1.1, event=False, extra=" + piecewise(kj, time >= 7, 0)")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["D"])
    with pytest.raises(bngsim.SimulationError, match=r"\(issue #949\)"):
        sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)


@pytest.mark.parametrize("shape, param", [("closing", "D"), ("opening", "on")])
def test_the_exponent_is_asked_again_after_set_param(shape, param):
    """Singular at a = 1.1 and not at a = 3, on one Simulator, each way round."""
    for first, then in ((1.1, 3.0), (3.0, 1.1)):
        model = _with_event(shape, first)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param])
        for a in (first, then):
            model.reset()
            model.set_param("a", a)
            if a == 3.0:
                out = sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)
                np.testing.assert_allclose(
                    np.asarray(out.sensitivities)[:, 0, 0],
                    _expected(shape, 3.0, param),
                    rtol=2e-5,
                    atol=2e-7,
                )
            else:
                with pytest.raises(bngsim.SimulationError, match=r"\(issue #958\)"):
                    sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)


# A window that opens where the state crosses: `time - Z >= on` with Z a species
# nothing makes. The run roots on it, and a column enters its frame at the root.
STATE_T = [0.0, 5.0, 12.0, 20.0, 29.0, 33.0, 38.0]


def _on_a_state_crossing(on, jump, event, a=1.9):
    return bngsim.Model.from_antimony_string(
        f"species X, Z; X = 1; Z = 0; k0 = 0.1; k1 = 1; a = {a}; on = {on!r}; D = 20; "
        "kj = 0.05; q = 0\n"
        "s := piecewise((time - on)/D, (time - Z >= on) && (time <= on + D), 0)\n"
        "J0: -> X; (k0 + k1*s^(a - 1)*(1 - s)"
        + (" + piecewise(kj, time - Z >= on, 0)" if jump else "")
        + ")*X\n"
        + ("E1: at (time > 35): q = 1\n" if event else "")
    )


STATE_CASES = pytest.mark.parametrize(
    "jump, a",
    [(False, 1.9), (False, 1.999), (True, 1.9)],
    ids=["opens-from-0", "opens-nearly-as-a-ramp", "opens-with-a-step"],
)


@STATE_CASES
def test_a_singular_edge_on_a_state_crossing_beside_an_event_is_refused(jump, a):
    """The crossing is a root of the state and not a switch time. The run reads
    the law as jumping there at a = 1.9, with a step or without one, and as
    continuous at a = 1.999, and asks in each place. All three ended in
    CV_FIRST_SRHSFUNC_ERR."""
    sim = bngsim.Simulator(
        _on_a_state_crossing(10.0, jump, True, a), method="ode", sensitivity_params=["on"]
    )
    with pytest.raises(bngsim.SimulationError, match=r"singular.*\(issue #958\)"):
        sim.run(sample_times=STATE_T, rtol=1e-8, atol=1e-10, timeout=120)


@STATE_CASES
def test_without_the_event_a_state_crossing_enters_the_frame(jump, a):
    """Control. Against central differences of plain runs in `on`, extrapolated."""

    def plain(on):
        out = bngsim.Simulator(_on_a_state_crossing(on, jump, False, a), method="ode").run(
            sample_times=STATE_T, rtol=1e-12, atol=1e-14
        )
        return np.asarray(out.species)[:, list(out.species_names).index("X")]

    def d(h):
        return (plain(10.0 + h) - plain(10.0 - h)) / (2 * h)

    want = (4 * d(5e-4) - d(1e-3)) / 3
    sim = bngsim.Simulator(
        _on_a_state_crossing(10.0, jump, False, a), method="ode", sensitivity_params=["on"]
    )
    out = sim.run(sample_times=STATE_T, rtol=1e-8, atol=1e-10, timeout=120)
    got = np.asarray(out.sensitivities)[:, list(out.species_names).index("X"), 0]
    # X is of order 1e3 past the window, where the column is 0.
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)


def _only_a_root(on, event, a=1.9):
    """The window opens where the state crosses and never closes."""
    return bngsim.Model.from_antimony_string(
        f"species X, Z; X = 1; Z = 0; k0 = 0.1; k1 = 1; a = {a}; on = {on!r}; D = 20; q = 0\n"
        "J0: -> X; (k0 + k1*piecewise(((time - on)/D)^(a - 1), time - Z >= on, 0))*X\n"
        + ("E1: at (time > 15): q = 1\n" if event else "")
    )


@pytest.mark.parametrize("a", [1.9, 1.999], ids=["read-as-a-jump", "read-as-continuous"])
def test_a_singular_edge_that_is_only_a_root_beside_an_event_is_refused(a):
    """No switch time moves at the rate the edge does, so nothing is known
    before the run: the column is refused at the root, where it would have
    entered its frame, whichever way the run reads the law there."""
    sim = bngsim.Simulator(_only_a_root(10.0, True, a), method="ode", sensitivity_params=["on"])
    with pytest.raises(bngsim.SimulationError, match=r"singular.*\(issue #958\)"):
        sim.run(sample_times=[0.0, 5.0, 12.0, 18.0], rtol=1e-8, atol=1e-10, timeout=120)


def test_without_the_event_an_edge_that_is_only_a_root_enters_the_frame():
    """Control. Against central differences of plain runs in `on`."""
    times = [0.0, 5.0, 12.0, 18.0]

    def plain(on):
        out = bngsim.Simulator(_only_a_root(on, False), method="ode").run(
            sample_times=times, rtol=1e-12, atol=1e-14
        )
        return np.asarray(out.species)[:, list(out.species_names).index("X")]

    def d(h):
        return (plain(10.0 + h) - plain(10.0 - h)) / (2 * h)

    sim = bngsim.Simulator(_only_a_root(10.0, False), method="ode", sensitivity_params=["on"])
    out = sim.run(sample_times=times, rtol=1e-8, atol=1e-10, timeout=120)
    got = np.asarray(out.sensitivities)[:, list(out.species_names).index("X"), 0]
    np.testing.assert_allclose(got, (4 * d(5e-4) - d(1e-3)) / 3, rtol=1e-5, atol=1e-6)


# ─── A column that moves the counter itself (issue #948) ────────────────────

COUNTER = """begin parameters
    1 k0 0.1
    2 k1 2.0
    3 a {a}
    4 on 4.0
    5 D 4.0
    6 kdeg 0.3
    7 r 1
    8 T0 1.0
end parameters
begin functions
    1 s() (t-on)/D
    2 prod() k0+if(t>=on,if(t<=(on+D),k1*{shape},0),0)
end functions
begin species
    1 X() 0
    2 Tc() T0
end species
begin reactions
    1 0 1 prod
    2 1 0 kdeg
    3 0 2 r
end reactions
begin groups
    1 t 2
end groups
"""

PLAIN_GATE = COUNTER.replace("k1*{shape}", "k1")


def _on_a_counter(tmp_path, shape, a, text=COUNTER):
    path = tmp_path / f"{shape}_{a}.net"
    path.write_text(text.format(a=a, shape=SHAPES[shape][0].format(s="s()")))
    return bngsim.Model.from_net(path)


@pytest.mark.parametrize("shape", ["closing", "opening"])
@pytest.mark.parametrize("params", [["T0"], ["r"], ["on", "T0"], ["T0", "D"]])
def test_a_column_that_moves_the_counter_is_refused_under_a_singular_power(
    tmp_path, shape, params
):
    """The counter's own initial value and its rate. Closing, dX/dT0 came back
    -0.6817 for -0.6843; opening, the run ended in CVODE's no-progress error."""
    _refused(_on_a_counter(tmp_path, shape, 1.1), params, 948)


@pytest.mark.parametrize("shape", ["closing", "opening"])
def test_the_counters_own_initial_condition_axis_is_refused(tmp_path, shape):
    """The same column, asked for as an initial-condition axis."""
    model = _on_a_counter(tmp_path, shape, 1.1)
    sim = bngsim.Simulator(model, method="ode", sensitivity_ic=["Tc()"])
    with pytest.raises(bngsim.SimulationError, match=r"singular.*\(issue #948\)"):
        sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)


@pytest.mark.parametrize("shape", ["closing", "opening"])
@pytest.mark.parametrize("params", [["on"], ["D"], ["on", "D"]])
def test_a_column_the_power_is_written_in_has_its_frame(tmp_path, shape, params):
    """Control. The window's own parameters are integrated in their frames."""
    got = _run(_on_a_counter(tmp_path, shape, 1.1), params)
    for j, param in enumerate(params):
        np.testing.assert_allclose(got[:, j], _expected(shape, 1.1, param), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("shape", ["closing", "opening"])
@pytest.mark.parametrize("param", ["T0", "r"])
def test_a_power_that_is_not_singular_runs_in_the_counters_column(tmp_path, shape, param):
    """Control. At a = 3 the counter's own column is plain, and right."""
    got = _run(_on_a_counter(tmp_path, shape, 3.0), [param])[:, 0]
    np.testing.assert_allclose(got, _expected(shape, 3.0, param), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("param", ["T0", "r"])
def test_a_plain_gate_runs_in_the_counters_column(tmp_path, param):
    """Control. A window that opens and closes as a step has no power at all
    (issue #725's column)."""
    model = _on_a_counter(tmp_path, "closing", 1.1, text=PLAIN_GATE)
    got = _run(model, [param])[:, 0]

    def x(t_end, lo, rate=1.0):
        hi = lo + WIDTH / rate
        base = K0 * (1 - np.exp(-KDEG * t_end)) / KDEG
        if t_end <= lo:
            return base
        top = min(hi, t_end)
        return base + K1 * (np.exp(-KDEG * (t_end - top)) - np.exp(-KDEG * (t_end - lo))) / KDEG

    moved = {
        "T0": lambda h, t: x(t, ON - h),
        "r": lambda h, t: x(t, ON / (1 + h), 1 + h),
    }[param]
    want = np.array([_slope(lambda h, t=t: moved(h, t)) for t in T])
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-7)


ON_TIME = COUNTER.replace("    1 s() (t-on)/D\n", "    1 s() (time()-(on-1))/D\n").replace(
    "k0+if(t>=on,if(t<=(on+D),k1*{shape},0),0)",
    "k0+if(time()>=(on-1),if(time()<=(on-1+D),k1*{shape},0),0)+if(t>=5.5,1.5,0)",
)


@pytest.mark.parametrize("shape", ["closing", "opening"])
def test_a_singular_power_on_time_itself_is_not_the_counters(tmp_path, shape):
    """Control. The window is on literal time, which no column moves, and the
    counter gates a step. The counter's own column has nothing unbounded in
    it."""
    got = _run(_on_a_counter(tmp_path, shape, 1.1, text=ON_TIME), ["T0"])[:, 0]
    # The counter reads 5.5 at t = 4.5 - (T0 - 1), and X gains 1.5 from there on.
    want = [0.0 if t < 4.5 else 1.5 * np.exp(-KDEG * (t - 4.5)) for t in T]
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("shape", ["closing", "opening"])
def test_the_counters_column_is_asked_again_after_set_param(tmp_path, shape):
    """Singular at a = 1.1 and not at a = 3, on one Simulator, each way round."""
    for first, then in ((1.1, 3.0), (3.0, 1.1)):
        model = _on_a_counter(tmp_path, shape, first)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["T0"])
        for a in (first, then):
            model.reset()
            model.set_param("a", a)
            if a == 3.0:
                out = sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)
                np.testing.assert_allclose(
                    np.asarray(out.sensitivities)[:, 0, 0],
                    _expected(shape, 3.0, "T0"),
                    rtol=2e-5,
                    atol=2e-7,
                )
            else:
                with pytest.raises(bngsim.SimulationError, match=r"\(issue #948\)"):
                    sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)


# A window written in numbers: nothing requested moves its edges, so it has no
# comoving case at all.
NUMBERS = COUNTER.replace("    1 s() (t-on)/D\n", "    1 s() (t-4)/4\n").replace(
    "k0+if(t>=on,if(t<=(on+D),k1*{shape},0),0)", "k0+if(t>=4,if(t<=8,k1*{shape},0),0)"
)


@pytest.mark.parametrize("params", [["T0"], ["r"], ["k1", "T0"]])
def test_a_window_written_in_numbers_is_as_singular(tmp_path, params):
    """``s = (t - 4)/4`` and ``if(t >= 4, if(t <= 8, ...))``: no parameter is in
    the power's base, so the model has no comoving case, and the refusal went
    by the cases. dX/dT0 came back -0.6816449 for -0.6842936 and dX/dr
    -6.2537523 for -6.2693167."""
    _refused(_on_a_counter(tmp_path, "closing", 1.1, text=NUMBERS), params, 948)


def test_a_window_written_in_numbers_and_the_counters_own_axis(tmp_path):
    model = _on_a_counter(tmp_path, "closing", 1.1, text=NUMBERS)
    sim = bngsim.Simulator(model, method="ode", sensitivity_ic=["Tc()"])
    with pytest.raises(bngsim.SimulationError, match=r"singular.*\(issue #948\)"):
        sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)


def test_a_window_written_in_numbers_runs_in_a_column_that_moves_nothing(tmp_path):
    """Control. k1 scales the pulse."""
    got = _run(_on_a_counter(tmp_path, "closing", 1.1, text=NUMBERS), ["k1"])[:, 0]
    np.testing.assert_allclose(got, _expected("closing", 1.1, "k1"), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize(
    "shape, a, params",
    [
        ("closing", 1.5, ["T0"]),
        ("closing", 1.5, ["r"]),
        ("opening", 1.1, ["T0"]),
        ("closing", 1.1, ["r"]),
        ("opening", 1.1, ["r"]),
        ("opening", 1.5, ["r"]),
    ],
    ids=[
        "its-seed",
        "its-rate",
        "an-opening-edge",
        "its-rate-under-a-sharper-close",
        "its-rate-at-an-opening-edge",
        "its-rate-at-a-milder-opening-edge",
    ],
)
def test_a_counter_edge_found_as_a_root_is_refused(tmp_path, shape, a, params):
    """The window opens where ``t - z >= on`` and closes where
    ``t - z <= on + D``, with z a species nothing makes: roots of the state,
    with no switch time for either of the counter's crossings. dX/dT0 was
    1.1e-4 off at a tolerance of 1e-6.

    Each column is refused where the run starts. The seed's has a row of the
    counter there. The rate constant's has moved nothing yet, and is known by
    the rate at which its row leaves 0: asked later, at a crossing or at the
    end, it was asked after the run had ended in ``CVODE made no progress``
    just short of an opening edge, or in ``CV_REPTD_SRHSFUNC_ERR``."""
    text = (
        COUNTER.replace("if(t>=on,", "if((t-Zobs)>=on,")
        .replace("if(t<=(on+D),", "if((t-Zobs)<=(on+D),")
        .replace("    2 Tc() T0\n", "    2 Tc() T0\n    3 Z() 0\n")
        .replace("    1 t 2\n", "    1 t 2\n    2 Zobs 3\n")
    )
    model = _on_a_counter(tmp_path, shape, a, text=text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
    with pytest.raises(bngsim.SimulationError, match=r"singular.*\(issue #948\)") as refusal:
        sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)
    assert "(at t=0)" in str(refusal.value)


MIXED = (
    COUNTER.replace(
        "k0+if(t>=on,if(t<=(on+D),k1*{shape},0),0)",
        "k0+if(time()>=on,if(time()<=(on+D),k1*{shape}*((t/8)^(b-1)),0),0)",
    )
    .replace("    1 s() (t-on)/D\n", "    1 s() (time()-on)/D\n")
    .replace("    8 T0 1.0\n", "    8 T0 1.0\n    9 b 3\n")
)


def test_a_power_of_the_counter_that_is_not_singular_beside_one_of_the_time(tmp_path):
    """Control. The window is on literal time and closes as a singular power;
    the law is also a square of the counter. The counter's power is not
    singular, and its own column runs."""
    model = _on_a_counter(tmp_path, "closing", 1.1, text=MIXED)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["T0"])
    out = sim.run(sample_times=T, rtol=1e-9, atol=1e-11, timeout=120)
    got = np.asarray(out.sensitivities)[:, 0, 0]

    def plain(t0):
        m = _on_a_counter(tmp_path, "closing", 1.1, text=MIXED)
        m.set_param("T0", t0)
        run = bngsim.Simulator(m, method="ode").run(sample_times=T, rtol=1e-12, atol=1e-14)
        return np.asarray(run.species)[:, 0]

    def d(h):
        return (plain(1.0 + h) - plain(1.0 - h)) / (2 * h)

    np.testing.assert_allclose(got, (4 * d(5e-4) - d(1e-3)) / 3, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("rtol", [1e-4, 1e-10, 1e-12])
def test_a_closing_edge_beside_an_event_is_refused_before_the_run(rtol):
    """Refused where the run starts, for the switch times it has. Asked only
    at the crossing, the plain column was first carried up to the edge, and at
    a tolerance of 1e-10 the run ended there in CVODE's no-progress error."""
    _refused(_with_event("closing", 1.1), ["D"], 958, rtol=rtol)


def test_an_opening_edge_at_an_exponent_of_exactly_0_beside_an_event_is_refused():
    """At a = 1 the power is the constant 1 and the window steps on, and the
    plain column's forcing is 0·∞ on the edge: CV_FIRST_SRHSFUNC_ERR."""
    _refused(_with_event("opening", 1.0), ["on"], 958)


def test_a_closing_edge_at_an_exponent_of_exactly_0_runs_beside_an_event():
    """Control. The closing power is the constant 1 there, with nothing
    unbounded before its edge, and the plain column is right."""
    got = _run(_with_event("closing", 1.0), ["D"])[:, 0]
    np.testing.assert_allclose(got, _expected("closing", 1.0, "D"), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("shape", ["closing", "opening"])
def test_a_counter_under_an_exponent_of_exactly_0_runs_in_its_own_column(tmp_path, shape):
    """Control. The same for a column that moves the counter."""
    got = _run(_on_a_counter(tmp_path, shape, 1.0), ["T0"])[:, 0]
    np.testing.assert_allclose(got, _expected(shape, 1.0, "T0"), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("shape", ["closing", "opening"])
def test_an_exponent_of_exactly_0_runs_with_no_event(shape):
    """Control. In its frame the same column is right."""
    got = _run(_with_event(shape, 1.0, event=False), ["on"])[:, 0]
    np.testing.assert_allclose(got, _expected(shape, 1.0, "on"), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("shape", ["closing", "opening"])
def test_an_exponent_that_is_a_species_builds(shape):
    """The exponent of a power read from a species: asked for at the run's
    parameter values it has none, and written out it named ``obs[]`` in a
    function that has no such thing, so no sensitivity could be run on the
    model at all, for a closing power. It is taken to be under 1, and a column
    that moves nothing runs.

    The opening case: Control. It built as it does now."""
    law = SHAPES[shape][0].format(s="((time-on)/D)").replace("(a-1)", "E")
    text = (
        "species X, E; X = 0; E = 0.5; k0 = 0.1; k1 = 2; on = 3; D = 4; kdeg = 0.3\n"
        f"J1: -> X; k0 + piecewise(piecewise(k1*{law}, time < on + D, 0), time >= on, 0)\n"
        "J2: X -> ; kdeg*X\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["k1"]
    )
    assert sim.has_analytic_sens_rhs
    out = sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)
    names = list(out.species_names)
    got = np.asarray(out.sensitivities)[:, names.index("X"), 0]
    want = np.array(
        [(_x(shape, 1.5, t, ON, WIDTH) - K0 * (1 - np.exp(-KDEG * t)) / KDEG) / K1 for t in T]
    )
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-7)


WEIGHTED = COUNTER.replace("    1 s() (t-on)/D\n", "    1 s() (t2/2-on)/D\n").replace(
    "    1 t 2\n", "    1 t 2\n    2 t2 2*2\n"
)


@pytest.mark.parametrize("params", [["T0"], ["r"]])
def test_a_power_that_reads_the_counter_through_a_weighted_observable(tmp_path, params):
    """The gate reads the counter and the power reads an observable that sums
    it with a weight of 2: no clock symbol is in the power's base. dX/dT0 came
    back -0.681383 for -0.684294."""
    _refused(_on_a_counter(tmp_path, "closing", 1.1, text=WEIGHTED), params, 948)


NO_ANALYTIC = COUNTER.replace(
    "k1*{shape},0),0)\n", "k1*{shape},0),0)+0.001*max(xo,0.5)\n"
).replace("    1 t 2\n", "    1 t 2\n    2 xo 1\n")


@pytest.mark.parametrize("params", [["T0"], ["r"]])
def test_a_counter_under_a_power_with_no_analytic_right_hand_side(tmp_path, params):
    """Control. A ``max()`` in another term leaves the model with no analytic
    sensitivity right-hand side, and the list of counters under a power is in
    that code, so nothing here asks. On the difference quotient the counter's
    own columns came back 11% off. Such a run is refused since issue #938: a
    condition on a counter that a requested column moves."""
    model = _on_a_counter(tmp_path, "closing", 1.1, text=NO_ANALYTIC)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
        assert not sim.has_analytic_sens_rhs
        sim.run(sample_times=T, rtol=1e-8, atol=1e-10, timeout=120)


def test_the_counters_are_listed_whatever_becomes_of_the_plan(tmp_path, monkeypatch):
    """The plan of the comoving cases runs under the derivation budget, and
    where that ran out the source was written without the counters' export:
    the counter's column then came back as it did before, and the library
    written that way was kept."""
    from bngsim import _codegen
    from bngsim._jacobian import _DerivationBudgetExceeded

    def out_of_budget(*args, **kwargs):
        raise _DerivationBudgetExceeded

    monkeypatch.setattr(_codegen, "_functional_comoving_plan", out_of_budget)
    core = _on_a_counter(tmp_path, "closing", 1.1)._core
    src = _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)
    assert "int bngsim_codegen_counter_power(int k, const double* p)" in src
    assert "bngsim_codegen_comoving_approach" not in src


def test_a_rate_law_that_is_not_read_is_taken_to_hold_a_power(tmp_path, monkeypatch):
    """Where a rate law cannot be read, nothing says it has no power of a
    counter under 1. The counters are listed with a test that holds at every
    value, where a law that is read is asked its exponent."""
    import sys

    from bngsim import _codegen, _jacobian

    def body(src):
        head = "int bngsim_codegen_counter_power(int k, const double* p)"
        return src.split(head)[1].split("\n}")[0]

    def source():
        return _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)

    core = _on_a_counter(tmp_path, "closing", 3.0)._core
    assert "if (((p[2] - 1.0) < 1.0 && (p[2] - 1.0) != 0.0)) {" in body(source())
    real = _jacobian._exprtk_to_sympy

    def unread(text, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_counter_powers":
            raise ValueError("not read")
        return real(text, *args, **kwargs)

    monkeypatch.setattr(_jacobian, "_exprtk_to_sympy", unread)
    listed = body(source())
    assert "if (1) {" in listed
    assert "if (k == 0) return 1;" in listed


# ─── Which counter, and which exponent ──────────────────────────────────────

TWO_COUNTERS = (
    COUNTER.replace("    8 T0 1.0\n", "    8 T0 1.0\n    9 r2 1\n   10 T02 0.5\n")
    .replace("k1*{shape},0),0)\n", "k1*{shape},0),0)+if(t2>=3,0.5,0)\n")
    .replace("    2 Tc() T0\n", "    2 Tc() T0\n    3 Tc2() T02\n")
    .replace("    3 0 2 r\n", "    3 0 2 r\n    4 0 3 r2\n")
    .replace("    1 t 2\n", "    1 t 2\n    2 t2 3\n")
)


@pytest.mark.parametrize("params", [["T02"], ["r2"]])
def test_a_counter_no_power_reads_keeps_its_own_columns(tmp_path, params):
    """Control. A second counter gates a step and is under no power. Its seed
    and its rate move the step alone: the production gains 0.5 from where
    ``T02 + r2*t`` reaches 3, at t = 2.5, so X(t) gains
    0.5*(1 - exp(-kdeg*(t - 2.5)))/kdeg after it, and that time moves by -1
    with the seed and by -2.5 with the rate."""
    model = _on_a_counter(tmp_path, "closing", 1.1, text=TWO_COUNTERS)
    got = _run(model, params)[:, 0]
    moved = -1.0 if params == ["T02"] else -2.5
    want = [0.0 if t <= 2.5 else -moved * 0.5 * np.exp(-KDEG * (t - 2.5)) for t in T]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-8)


def test_the_counter_under_the_power_is_still_refused_beside_another(tmp_path):
    _refused(_on_a_counter(tmp_path, "closing", 1.1, text=TWO_COUNTERS), ["T0"], 948)


CHOSEN = COUNTER.replace("    3 a {a}\n", "    3 a1 {a}\n    9 a2 3.0\n   10 tc 100\n").replace(
    "    1 s() (t-on)/D\n", "    1 s() (t-on)/D\n    3 a() if(t<tc,a1,a2)\n"
)


def test_an_exponent_a_condition_chooses_is_asked_branch_by_branch(tmp_path):
    """``a() = if(t < tc, a1, a2)`` has no one value to ask. Each branch has:
    with 3 and 3 nothing is singular and the counter's own column runs, where
    it was refused as an exponent that cannot be asked; with 1.1 in either
    branch it is refused, whichever the run is on."""
    model = _on_a_counter(tmp_path, "closing", 3.0, text=CHOSEN)
    np.testing.assert_allclose(
        _run(model, ["T0"])[:, 0], _expected("closing", 3.0, "T0"), rtol=2e-5, atol=2e-7
    )
    model.set_param("a2", 1.1)
    model.reset()
    _refused(model, ["T0"], 948)
    _refused(_on_a_counter(tmp_path, "closing", 1.1, text=CHOSEN), ["T0"], 948)


DECAY = COUNTER.replace("k1*{shape}", "k1*(1+t/D)^(-a)")


@pytest.mark.parametrize("a", [2.0, 0.5])
def test_a_power_of_a_counter_under_0_is_held_to_what_a_number_is(tmp_path, a):
    """Control. ``(1 + t/D)^(-a)`` beside the gate: a power-law decay, whose
    base is never 0 in the run. Written with a number, ``(1 + t/D)^(-2)``, it
    was never taken for singular; written with a parameter it was, at every
    value under 1. Both are asked the same now, between 0 and 1, and the rate
    constant's column runs. It is checked against the same law with the
    exponent a number."""
    by_parameter = _on_a_counter(tmp_path, "closing", a, text=DECAY)
    (tmp_path / "n").mkdir()
    by_number = _on_a_counter(
        tmp_path / "n", "closing", a, text=DECAY.replace("^(-a)", f"^(-{a})")
    )
    np.testing.assert_allclose(
        _run(by_parameter, ["r"])[:, 0], _run(by_number, ["r"])[:, 0], rtol=1e-6, atol=1e-9
    )


# ─── An exponent under 0: a pole, where the law gates on the base's zero ─────


@pytest.mark.parametrize("shape", ["closing", "opening"])
@pytest.mark.parametrize("params", [["T0"], ["r"]])
def test_a_pole_on_the_window_s_own_edge_is_refused_in_the_counters_column(
    tmp_path, shape, params
):
    """At ``a = 0.99`` the power is ``(1 - s)^(-0.01)``: a pole at the edge the
    law's own condition closes on, integrable, and as wrong in the plain column
    of what moves the counter as a root is. dX/dT0 came back -0.8334 for
    -0.8251, and dX/dr -7.682 for -7.634; at ``a = 0.5`` the run stalled."""
    _refused(_on_a_counter(tmp_path, shape, 0.99), params, 948)


def test_a_pole_written_with_a_number_is_held_to_the_same(tmp_path):
    """``(1 - s)^(-0.01)`` with the exponent a number was never asked about,
    and came back 4e-4 off."""
    text = COUNTER.replace("k1*{shape}", "k1*s()*(1-s())^(-0.01)")
    _refused(_on_a_counter(tmp_path, "closing", 3.0, text=text), ["T0"], 948)


def test_a_pole_does_not_cost_the_window_its_own_columns(tmp_path):
    """Control. ``on`` and ``D`` are integrated in their frames at an exponent
    under 0 as at one over it."""
    model = _on_a_counter(tmp_path, "closing", 0.99)
    np.testing.assert_allclose(
        _run(model, ["on"])[:, 0], _expected("closing", 0.99, "on"), rtol=2e-4, atol=2e-6
    )


GATED_BY_ANOTHER = COUNTER.replace("    8 T0 1.0\n", "    8 T0 1.0\n    9 off 8.0\n").replace(
    "if(t<=(on+D),", "if(t<=off,"
)


def _by_plain_runs(model, param, value, h=1e-4):
    """dX/d``param`` by central differences of runs with no sensitivities."""

    def x(at):
        model.set_param(param, at)
        model.reset()
        out = bngsim.Simulator(model, method="ode").run(
            sample_times=T, rtol=1e-11, atol=1e-13, timeout=120
        )
        return np.asarray(out.species)[:, list(out.species_names).index("X()")]

    slope = (x(value + h) - x(value - h)) / (2 * h)
    model.set_param(param, value)
    model.reset()
    return slope


def test_a_pole_that_meets_the_gate_at_the_run_s_values_is_refused_there(tmp_path):
    """``(1 - s)^(-0.01)`` under ``t <= off``, with ``off`` a parameter of its
    own. At ``off = on + D = 8`` the pole is on the gate and dX/dT0 came back
    -0.8334 for -0.8251, as with the gate written ``on + D``. Asked at the
    run's values, in both directions: the model built at 8 and moved to 7.7
    runs, and moved back is refused."""
    model = _on_a_counter(tmp_path, "closing", 0.99, text=GATED_BY_ANOTHER)
    _refused(model, ["T0"], 948)
    model.set_param("off", 7.7)
    model.reset()
    np.testing.assert_allclose(
        _run(model, ["T0"])[:, 0], _by_plain_runs(model, "T0", 1.0), rtol=2e-5, atol=2e-7
    )
    model.set_param("off", 8.0)
    model.reset()
    _refused(model, ["T0"], 948)
    # 0 to rounding: D - off + on is 2.8e-17 at these values, and not 0.
    model.set_params({"on": 0.1, "D": 0.2, "off": 0.3, "T0": 0.0})
    model.reset()
    _refused(model, ["T0"], 948)


@pytest.mark.parametrize("a", [0.99, 0.5])
def test_a_pole_the_gate_closes_short_of_is_not_on_an_edge(tmp_path, a):
    """Control. The same law closed at 7.7, short of the pole at 8: the rate is
    finite wherever the run is, and the counter's own column is right."""
    model = _on_a_counter(tmp_path, "closing", a, text=GATED_BY_ANOTHER)
    model.set_param("off", 7.7)
    model.reset()
    np.testing.assert_allclose(
        _run(model, ["T0"])[:, 0], _by_plain_runs(model, "T0", 1.0), rtol=2e-5, atol=2e-7
    )


def test_a_pole_gated_under_the_counter_s_other_name_is_refused(tmp_path):
    """The law closes on ``u``, a second observable of the counter species, and
    its power reads ``t``: the same edge, with no one threshold to put into the
    base. Taken to be on it. dX/dT0 came back -0.8334 for -0.8251."""
    text = COUNTER.replace("    1 t 2\n", "    1 t 2\n    2 u 2\n").replace(
        "if(t<=(on+D),", "if(u<=(on+D),"
    )
    _refused(_on_a_counter(tmp_path, "closing", 0.99, text=text), ["T0"], 948)
    np.testing.assert_allclose(
        _run(_on_a_counter(tmp_path, "closing", 3.0, text=text), ["T0"])[:, 0],
        _expected("closing", 3.0, "T0"),
        rtol=2e-5,
        atol=2e-7,
    )


def test_a_pole_on_a_gate_that_is_not_solved_for_is_refused(tmp_path):
    """``(8.5 - t - cos(t))^(-0.01)`` under ``t + cos(t) <= 8.5``: the edge has
    no closed form to put into the base, and is taken to be on its zero. The
    run ended in CVODE's no-progress error at the edge, after 20,000 steps."""
    text = COUNTER.replace(
        "k0+if(t>=on,if(t<=(on+D),k1*{shape},0),0)",
        "k0+if(t+cos(t)<=8.5,k1*(8.5-t-cos(t))^(a-1),0)",
    )
    _refused(_on_a_counter(tmp_path, "closing", 0.99, text=text), ["T0"], 948)


def test_a_pole_on_a_gate_that_reads_a_state_is_refused(tmp_path):
    """The law closes at ``t <= z``, with ``z`` a species that stays at 8,
    which is ``on + D``: the pole is on the gate by the value of a state, and
    the base there is nothing that can be asked of the parameters. Taken to be
    on its zero, and refused where the run starts. The run ended in CVODE's
    no-progress error at the edge."""
    text = (
        COUNTER.replace("    2 Tc() T0\n", "    2 Tc() T0\n    3 Z() 8\n")
        .replace("    1 t 2\n", "    1 t 2\n    2 z 3\n")
        .replace("if(t<=(on+D),", "if(t<=z,")
    )
    _refused(_on_a_counter(tmp_path, "closing", 0.99, text=text), ["T0"], 948)


BY_YEAR = COUNTER.replace(
    "    1 s() (t-on)/D\n", "    1 s() (t-onx())/D\n    3 onx() if(t<100,on,on+1)\n"
).replace("if(t>=on,if(t<=(on+D),", "if(t>=onx(),if(t<=(onx()+D),")


def test_a_gate_written_through_a_selection_is_not_solved_for(tmp_path, monkeypatch):
    """The onset is chosen by year, ``onx() = if(t < 100, on, on + 1)``, as a
    seasonal model writes it. Solving such a gate for its threshold cost
    SIR_v4 its whole derivation budget, 50 s, and with it the frame of every
    column: the run then stalled at the first edge. It is taken to be on the
    power's zero without being worked out: nothing is simplified for it, an
    exponent under 0 is refused, and one over 1 runs."""
    import sys

    import sympy
    from bngsim import _codegen

    worked_out = [0]
    real = sympy.simplify

    def counted(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "gated_at_its_zero":
            worked_out[0] += 1
        return real(*args, **kwargs)

    def generate(model):
        return _codegen.generate_sens_from_model(
            model._core, functional=True, emit_term_scale=True
        )

    monkeypatch.setattr(sympy, "simplify", counted)
    (tmp_path / "base").mkdir()
    by_year = _on_a_counter(tmp_path, "closing", 0.99, text=BY_YEAR)
    assert "if (((p[2] - 1.0) < 1.0 && (p[2] - 1.0) != 0.0)) {" in generate(by_year)
    assert worked_out[0] == 0
    # The same where the selection is in the power's base alone, under a
    # gate that is linear in the counter.
    in_the_base = BY_YEAR.replace("if(t>=onx(),if(t<=(onx()+D),", "if(t>=on,if(t<=(on+D),")
    in_the_base = _on_a_counter(tmp_path / "base", "closing", 0.99, text=in_the_base)
    assert "if (((p[2] - 1.0) < 1.0 && (p[2] - 1.0) != 0.0)) {" in generate(in_the_base)
    assert worked_out[0] == 0
    # A gate that is linear in the counter is worked out.
    generate(_on_a_counter(tmp_path, "closing", 0.99))
    assert worked_out[0] > 0
    monkeypatch.undo()
    _refused(by_year, ["T0"], 948)
    np.testing.assert_allclose(
        _run(_on_a_counter(tmp_path, "closing", 3.0, text=BY_YEAR), ["T0"])[:, 0],
        _expected("closing", 3.0, "T0"),
        rtol=2e-5,
        atol=2e-7,
    )


LONG_CHAIN = COUNTER.replace(
    "    3 a {a}\n",
    "    3 a1 {a}\n"
    + "".join(f"   {8 + i} tc{i} {100 * i}\n" for i in range(1, 6))
    + "".join(f"   {12 + i} a{i} {{a}}\n" for i in range(2, 7)),
).replace(
    "    1 s() (t-on)/D\n",
    "    1 s() (t-on)/D\n"
    "    3 a() if(t<tc1,a1,if(t<tc2,a2,if(t<tc3,a3,if(t<tc4,a4,if(t<tc5,a5,a6)))))\n",
)
BY_A_PARAMETER = COUNTER.replace(
    "    3 a {a}\n", "    3 a1 {a}\n    9 a2 1.1\n   10 q 1\n"
).replace("    1 s() (t-on)/D\n", "    1 s() (t-on)/D\n    3 a() if(q>0,a1,a2)\n")


def test_an_exponent_chosen_by_a_parameter_is_evaluated(tmp_path):
    """Control. ``if(q > 0, a1, a2)`` is evaluated where the run is: the value
    it does not take, 1.1 here, is no reason to refuse."""
    model = _on_a_counter(tmp_path, "closing", 3.0, text=BY_A_PARAMETER)
    np.testing.assert_allclose(
        _run(model, ["T0"])[:, 0], _expected("closing", 3.0, "T0"), rtol=2e-5, atol=2e-7
    )


def test_an_exponent_chosen_among_six_values_runs(tmp_path):
    """A selection of six values, each 3, is asked value by value however deep
    it is written, and the counter's own column runs. Such a model did not
    build: its exponent, which reads the counter, was written into a function
    that has only the parameters."""
    model = _on_a_counter(tmp_path, "closing", 3.0, text=LONG_CHAIN)
    np.testing.assert_allclose(
        _run(model, ["T0"])[:, 0], _expected("closing", 3.0, "T0"), rtol=2e-5, atol=2e-7
    )


def test_the_last_of_six_values_counts_as_each_does(tmp_path):
    """The sixth value of the selection, which the run never takes, at 1.1:
    refused, as with 1.1 in any other. dX/dT0 came back -0.6591 for -0.6620
    under a chain of five."""
    model = _on_a_counter(tmp_path, "closing", 3.0, text=LONG_CHAIN)
    model.set_param("a6", 1.1)
    model.reset()
    _refused(model, ["T0"], 948)


# ─── What the generator says of a case ──────────────────────────────────────


def test_the_generator_says_where_the_plain_column_fails_and_which_counters_are_under_a_power(
    tmp_path,
):
    """Every case has an entry now, with a bit for an exponent under 1 in any
    of its powers (4). And whether a rate law has such a power of a counter is
    said on its own, with the counter species, whether or not there is a case:
    a window written in numbers has none."""
    from bngsim import _codegen

    def source(core):
        return _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)

    def entries(src):
        head = "int bngsim_codegen_comoving_approach(int case_idx, const double *p)"
        body = src.split(head)[1].split("\n}")[0]
        return [
            line.split("return ", 1)[1] for line in body.splitlines() if "if (case_idx ==" in line
        ]

    def counters(src):
        head = "int bngsim_codegen_counter_power(int k, const double* p)"
        if head not in src:
            return None
        body = src.split(head)[1].split("\n}")[0]
        return [
            int(line.split("return ")[1].rstrip(";"))
            for line in body.splitlines()
            if "if (k ==" in line
        ]

    closing = source(_on_a_counter(tmp_path, "closing", 1.1)._core)
    assert len(entries(closing)) == 2
    assert all("? 4 : 0);" in e and "((0) ? 4 : 0)" not in e for e in entries(closing))
    assert counters(closing) == [1]
    opening = source(_on_a_counter(tmp_path, "opening", 1.1)._core)
    assert len(entries(opening)) == 1
    assert entries(opening)[0].startswith("((0) ? 1 : (0) ? 2 : 0) | ((")
    assert counters(opening) == [1]
    # A window written in numbers: no parameter moves its edges, so no case,
    # and the power of the counter is said all the same.
    numbers = source(_on_a_counter(tmp_path, "closing", 1.1, text=NUMBERS)._core)
    assert "bngsim_codegen_comoving_approach" not in numbers
    assert counters(numbers) == [1]
    # On literal time there is no counter.
    for shape in ("closing", "opening"):
        assert counters(source(_with_event(shape, 1.1)._core)) is None
