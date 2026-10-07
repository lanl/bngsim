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
    with pytest.raises(bngsim.SimulationError, match=rf"singular power.*\(issue #{issue}\)"):
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
    with pytest.raises(bngsim.SimulationError, match=r"singular power.*\(issue #958\)"):
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
    with pytest.raises(bngsim.SimulationError, match=r"singular power.*\(issue #948\)"):
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


# ─── What the generator says of a case ──────────────────────────────────────


def test_the_generator_says_which_cases_open_singular_and_read_a_counter(tmp_path):
    """Every case has an entry now: whether a power of it opens singular at the
    run's values (4), and whether one reads a counter and not time itself (8)."""
    from bngsim import _codegen

    def entries(core):
        src = _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)
        head = "int bngsim_codegen_comoving_approach(int case_idx, const double *p)"
        body = src.split(head)[1].split("\n}")[0]
        return [
            line.split("return ", 1)[1] for line in body.splitlines() if "if (case_idx ==" in line
        ]

    # On a counter: both cases of a closing window close; `on` alone moves an
    # opening power's edge.
    closing = entries(_on_a_counter(tmp_path, "closing", 1.1)._core)
    assert len(closing) == 2 and all(e.endswith("| ((0) ? 4 : 0) | 8;") for e in closing)
    opening = entries(_on_a_counter(tmp_path, "opening", 1.1)._core)
    assert len(opening) == 1
    assert opening[0].startswith("((0) ? 1 : (0) ? 2 : 0) | ((") and opening[0].endswith("| 8;")
    assert "? 4 : 0)" in opening[0] and "((0) ? 4 : 0)" not in opening[0]
    # On literal time nothing reads a counter.
    timed = entries(_with_event("closing", 1.1)._core) + entries(_with_event("opening", 1.1)._core)
    assert len(timed) == 3 and all(e.endswith("| 0;") for e in timed)
