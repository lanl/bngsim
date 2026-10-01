"""The closing edge of a window, in the column that moves it (issue #760).

A window ``k1·s·(1-s)^(a-1)`` with ``s = (t-on)/D`` closes at ``on + D`` as a
power. The forcing of the D column goes as ``(1-s)^(a-2)`` before the crossing,
which no step resolves for a < 2. A comoving column ``V = S + c·f`` has no such
term (issue #545), but a column entered its frame at the crossing it moves,
after the approach had been integrated: dX/dD came back 0.37% low at a = 1.1,
flat in the tolerance, and the run raised at a = 1.05.

The column for ``on`` was right all along. It enters at the opening edge, which
it moves at the same rate as the closing one, and is still in its frame when
the window closes.

A column whose next switch time is approached through such a power now enters
before it, at a stop the run takes a sixteenth of the way along the last stretch
between switch times that leads to it. Not at the crossing that begins the
stretch: a window that opens as a power too, ``s^(a-1)·(1-s)^(a-1)``, has an f
whose slope is unbounded just past the opening. And not on an earlier stretch,
where the frame would meet the edges of windows the column does not move.

X is linear in the pulse, so every expected value is a quadrature of the pulse
against ``e^(-kdeg·(T-t))``, differenced in the parameter at two steps and
extrapolated.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from scipy.integrate import quad

ON, WIDTH, K1, KDEG = 3.0, 4.0, 2.0, 0.3
# Nothing is sampled on the onset (3) or the close (7): the window's own edge
# is a kink in D there. The D column enters at 3.25.
T = [0.0, 1.0, 2.0, 4.0, 5.0, 5.5, 6.5, 6.99, 7.01, 8.0, 10.0]

NET = """begin parameters
    1 k0    0.1
    2 k1    2.0
    3 a     {a}
    4 on    {on}
    5 D     4.0
    6 kdeg  0.3
    7 _rateLaw1 1
    8 tmid  {tmid}
end parameters
begin functions
    1 s() (t-on)/D
    2 prod() k0+if(t>=on,if(t{close}(on+D),k1*{shape},0),0){extra}
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
    "closing": ("s()*((1-s())^(a-1))", lambda u, a: u * (1 - u) ** (a - 1)),
    "both": ("(s()^(a-1))*((1-s())^(a-1))", lambda u, a: u ** (a - 1) * (1 - u) ** (a - 1)),
    "opening": ("(s()^(a-1))*(1-s())", lambda u, a: u ** (a - 1) * (1 - u)),
}


def _model(tmp_path, shape, a, close="<=", extra="", tmid=5.0, state_switch=None, on=ON):
    text = NET.format(a=a, close=close, shape=SHAPES[shape][0], extra=extra, tmid=tmid, on=on)
    if state_switch is not None:
        # A third species whose rate switches where X crosses a level. It feeds
        # nothing back: the run restarts there and X's columns are what they were.
        text = (
            text.replace(
                "end functions", f"    3 zr() if(Xg>{state_switch!r},0.2,0.1)\nend functions"
            )
            .replace("end species", "    3 Z() 0\nend species")
            .replace("end reactions", "    4 0 3 zr\nend reactions")
            .replace("end groups", "    2 Xg 1\nend groups")
        )
    path = tmp_path / "m.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def _pulse(shape, a, t_end, on, width, k1=K1, kdeg=KDEG):
    """What one window has put into X by t_end."""
    pulse = SHAPES[shape][1]
    hi = min(on + width, t_end)
    if hi <= on:
        return 0.0
    value, _err = quad(
        lambda u: k1 * pulse(u, a) * np.exp(-kdeg * (t_end - on - width * u)) * width,
        0.0,
        (hi - on) / width,
        epsabs=0.0,
        epsrel=1e-13,
        limit=400,
    )
    return value


def _slope(x_of, times, h=1e-4):
    """d x_of(t, by)/d by at by = 0, at every sample time."""
    out = []
    for t_end in times:
        coarse = (x_of(t_end, h) - x_of(t_end, -h)) / (2 * h)
        fine = (x_of(t_end, h / 2) - x_of(t_end, -h / 2)) / h
        out.append((4 * fine - coarse) / 3)
    return np.array(out)


def _exact(shape, a, param, on=ON, times=T, width=WIDTH, k1=K1, kdeg=KDEG):
    """dX(t)/dparam at every sample time, for one window."""

    def moved(t_end, by):
        if param == "on":
            return _pulse(shape, a, t_end, on + by, width, k1, kdeg)
        return _pulse(shape, a, t_end, on, width + by, k1, kdeg)

    return _slope(moved, times, h=1e-4 * max(1.0, width if param == "D" else 1.0))


def _column(model, param, rtol=1e-8, atol=1e-10, times=T):
    run = bngsim.Simulator(model, method="ode", sensitivity_params=[param]).run(
        sample_times=list(times), rtol=rtol, atol=atol
    )
    return np.asarray(run.sensitivities)[:, 0, 0]


def _worst(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.mark.parametrize("close", ["<=", "<"])
@pytest.mark.parametrize("a", [1.05, 1.1, 1.2, 1.3])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_the_column_that_moves_the_closing_edge(tmp_path, shape, a, close):
    """dX/dD at every sample: one on the stop the column enters at (5), and one
    0.01 either side of the close. It was 0.5% off at a = 1.05 and 1.1e-5 at
    a = 1.3, at rtol 1e-8, and the closing shape raised at a = 1.05."""
    got = _column(_model(tmp_path, shape, a, close), "D")
    assert _worst(got, _exact(shape, a, "D")) < 5e-6


@pytest.mark.parametrize("a", [1.05, 1.1, 1.2, 1.3])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_the_same_column_at_a_looser_tolerance(tmp_path, shape, a):
    """At rtol 1e-6 the column is as good as the onset column, 3e-5. It was
    1.2e-4 to 7e-3."""
    got = _column(_model(tmp_path, shape, a), "D", rtol=1e-6, atol=1e-8)
    assert _worst(got, _exact(shape, a, "D")) < 6e-5


@pytest.mark.parametrize("shape", ["closing", "both"])
def test_both_edges_requested_together(tmp_path, shape):
    """With `on` requested too, the opening edge is a switch record and not a
    fixed crossing, and it is the record's jump that asks for the stop the D
    column enters at. A window that opens as a power is the one where entering
    at the opening itself fails: f has an unbounded slope just past it."""
    run = bngsim.Simulator(
        _model(tmp_path, shape, 1.1), method="ode", sensitivity_params=["on", "D"]
    ).run(sample_times=T, rtol=1e-8, atol=1e-10)
    got = np.asarray(run.sensitivities)[:, 0, :]
    assert _worst(got[:, 0], _exact(shape, 1.1, "on")) < 5e-6
    assert _worst(got[:, 1], _exact(shape, 1.1, "D")) < 5e-6


@pytest.mark.parametrize("shape", ["closing", "both"])
@pytest.mark.parametrize("a", [1.05, 1.1, 1.2])
def test_a_window_already_open_at_the_start(tmp_path, shape, a):
    """With on = 0 the window is open when the run starts and the close, at
    D = 4, is the first switch time ahead. The D column asks at the start for
    the stop it enters at. Not the start itself: the window that opens as a
    power is opening there, and a first cut that entered at the start raised."""
    times = [0.0, 1.0, 2.0, 3.0, 3.99, 4.01, 5.0, 8.0]
    got = _column(_model(tmp_path, shape, a, on=0.0), "D", times=times)
    assert got[0] == 0.0
    assert _worst(got, _exact(shape, a, "D", on=0.0, times=times)) < 5e-6


@pytest.mark.parametrize("tmid", [4.0, 5.0, 6.9])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_crossing_inside_the_window_asks_again(tmp_path, shape, tmid):
    """A fixed crossing at tmid, inside the window, restarts the run and leaves
    every frame. The column asks for a new stop halfway from there to the close.
    At 5 the crossing is on the stop the run had planned; at 6.9 the new stop is
    0.05 before the close."""
    extra = "+if(t>=tmid,0.0,0.0)*0"
    got = _column(_model(tmp_path, shape, 1.1, extra=extra, tmid=tmid), "D")
    assert _worst(got, _exact(shape, 1.1, "D")) < 5e-6


@pytest.mark.parametrize(
    ("shape", "level"),
    [("closing", 0.6), ("closing", 1.85), ("both", 2.0), ("both", 3.6)],
    ids=["closing-before-the-stop", "closing-after", "both-before-the-stop", "both-after"],
)
def test_a_state_switch_inside_the_window(tmp_path, shape, level):
    """X crosses the level near t = 4.3 or t = 6, either side of the stop at 5
    the D column enters at. A state switch restarts the run and leaves every
    frame; f is smooth there, so the column enters again on the spot."""
    got = _column(_model(tmp_path, shape, 1.1, state_switch=level), "D")
    assert _worst(got, _exact(shape, 1.1, "D")) < 5e-6


# ─── The same window through SBML, on literal time ──────────────────────────

SBML_SHAPES = {
    "closing": "s*(1-s)^(a-1)",
    "both": "s^(a-1)*(1-s)^(a-1)",
    "opening": "s^(a-1)*(1-s)",
}


def _sbml(shape, a, extra="", close="<=", on=3.0):
    law = SBML_SHAPES[shape].replace("s", "((time-on)/D)")
    return bngsim.Model.from_antimony_string(
        f"species X; X = 0; k0 = 0.1; k1 = 2; a = {a}; on = {on!r}; D = 4; kdeg = 0.3\n"
        "tmid = 5.5\n"
        f"J1: -> X; k0 + piecewise(piecewise(k1*{law}, time {close} on + D, 0), time >= on, 0)"
        f"{extra}\n"
        "J2: X -> ; kdeg*X\n"
    )


@pytest.mark.parametrize("close", ["<=", "<"])
@pytest.mark.parametrize("param", ["D", "on"])
@pytest.mark.parametrize("a", [1.1, 1.3])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_the_closing_edge_through_sbml(shape, a, param, close):
    """An SBML ``time <= on + D`` is a switch record and a registered root as
    well. With ``<=`` the root is reported a few ulp after the record's stop,
    and with ``<`` at the stop, before the record's jump. Either way the frame
    was left at the root against f read 1e-9 back, which is before the switch,
    where a window closing as (1-s)^0.1 is still 0.13 of its height: the onset
    column was 32% off past the close, with no warning, and the D column
    raised."""
    got = _column(_sbml(shape, a, close=close), param)
    assert _worst(got, _exact(shape, a, param)) < 5e-6


@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_root_inside_the_window_through_sbml(shape):
    """A fixed ``time >= 5.5`` inside the window is a root with no record. The
    D column is in its frame by then, leaves at the root and asks for a stop
    halfway to the close."""
    got = _column(_sbml(shape, 1.1, " + 0*piecewise(1, time >= tmid, 0)"), "D")
    assert _worst(got, _exact(shape, 1.1, "D")) < 5e-6


@pytest.mark.parametrize("param", ["D", "on"])
def test_a_window_that_only_opens_as_a_power_through_sbml(param):
    """Control."""
    got = _column(_sbml("opening", 1.5), param)
    assert _worst(got, _exact("opening", 1.5, param)) < 5e-6


@pytest.mark.parametrize("a", [1.1, 1.5])
@pytest.mark.parametrize("shape", ["closing", "both", "opening"])
def test_the_onset_column_is_as_it_was(tmp_path, shape, a):
    """Control. The onset moves both edges at 1, so its column was in its frame
    across the whole window already. It enters ahead now at the start of the
    run, where the first sample has to read 0 and not c·f."""
    got = _column(_model(tmp_path, shape, a), "on")
    assert got[0] == 0.0
    assert _worst(got, _exact(shape, a, "on")) < 5e-6


@pytest.mark.parametrize("a", [1.1, 1.5])
def test_a_window_that_only_opens_as_a_power_is_as_it_was(tmp_path, a):
    """Control. ``s^(a-1)·(1-s)`` closes smoothly: nothing enters ahead for D,
    whose column at the opening edge has no shift at all."""
    got = _column(_model(tmp_path, "opening", a), "D")
    assert _worst(got, _exact("opening", a, "D")) < 5e-6


@pytest.mark.parametrize("param", ["on", "D"])
def test_nothing_enters_ahead_of_a_window_that_only_opens_as_a_power(tmp_path, param):
    """Control. Before this window opens neither column has moved, and both
    are exactly 0."""
    times = [0.0, 0.5, 1.0, 2.0, 2.9]
    got = _column(_model(tmp_path, "opening", 1.1), param, times=times)
    assert np.all(got == 0.0)


def test_only_a_scale_the_numerator_reads_is_split():
    """The closing base ``(on + D − t)/D`` is written as two powers. The opening
    base ``(t − on)/D`` is left as it is written: its numerator does not read D,
    and a model with no closing power emits the code it always did."""
    import sympy as sp
    from bngsim import _codegen

    t, on, width, a = sp.symbols("t on D a")
    values = {on: sp.Float(3.0), width: sp.Float(4.0), a: sp.Float(1.1)}
    opening = ((t - on) / width) ** (a - 1)
    closing = (1 - (t - on) / width) ** (a - 1)
    assert _codegen._split_shared_scale(opening, {"t"}, values, sp) == opening
    split = _codegen._split_shared_scale(closing, {"t"}, values, sp)
    assert split != closing
    assert {factor.exp for factor in split.args} == {a - 1, 1 - a}
    point = {t: 5.5, on: 3.0, width: 4.0, a: 1.1}
    assert float(split.subs(point)) == pytest.approx(float(closing.subs(point)), rel=1e-14)


def test_the_generator_marks_the_case_a_closing_edge_approaches(tmp_path):
    """The D column has a case only where the edge it moves is singular, and
    that case is marked as one to enter ahead of."""
    from bngsim import _codegen

    def source(shape):
        core = _model(tmp_path, shape, 1.1)._core
        return _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)

    def approach(src):
        body = src.split("int bngsim_codegen_comoving_approach(int case_idx)")[1].split("\n}")[0]
        return body.count("return 1;")

    assert "bngsim_codegen_comoving_approach" in source("closing")
    assert approach(source("closing")) == 2  # on and D
    assert approach(source("both")) == 2
    assert approach(source("opening")) == 0


# ─── Other windows, and other roots ─────────────────────────────────────────

TWO = """begin parameters
    1 k0 0.1
    2 a {a}
    3 on1 2
    4 D1 3
    5 k1 2
    6 on2 {on2}
    7 D2 {D2}
    8 k2 3
    9 kdeg 0.3
    10 _rateLaw1 1
end parameters
begin functions
    1 s1() (t-on1)/D1
    2 s2() (t-{start2})/{width2}
    3 w1() if(t>=on1,if(t<=(on1+D1),k1*{p1},0),0)
    4 w2() if(t>={start2},if(t<=({start2}+{width2}),k2*{p2},0),0)
    5 prod() k0+w1()+w2()
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
NET_SHAPES = {
    "closing": "{s}*((1-{s})^(a-1))",
    "both": "({s}^(a-1))*((1-{s})^(a-1))",
}


def _two(tmp_path, shape, a=1.1, start2="on2", width2="D2", on2=7, d2=4):
    text = TWO.format(
        a=a,
        on2=on2,
        D2=d2,
        start2=start2,
        width2=width2,
        p1=NET_SHAPES[shape].format(s="s1()"),
        p2=NET_SHAPES[shape].format(s="s2()"),
    )
    path = tmp_path / "two.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_window_the_column_does_not_move_is_left_to_the_plain_column(tmp_path, shape):
    """Two windows, [2, 5] and [7, 11], and only the second one's parameters
    requested. In its frame a column carries c·∂f/∂t for everything f does, so a
    column that entered at the start met the first window's closing power there,
    which it does not move: 0.003 where the column is exactly 0, or the solver's
    error. It enters on the last stretch before its own switch time: the width
    column after the second window has opened, where it is exactly 0 until then,
    and the onset column after the first has closed, where it reads back as
    rounding from there to the opening.

    The onset column is a control: main has it right."""
    times = [0.0, 1.0, 3.0, 4.9, 5.6, 6.0, 6.9, 8.0, 9.5, 10.9, 11.1, 13.0]
    run = bngsim.Simulator(
        _two(tmp_path, shape), method="ode", sensitivity_params=["on2", "D2"]
    ).run(sample_times=times, rtol=1e-8, atol=1e-10)
    got = np.asarray(run.sensitivities)[:, 0, :]
    assert np.all(got[:4] == 0.0)  # both columns are plain while the first window is open
    assert np.all(got[:7, 1] == 0.0)
    assert np.abs(got[:7, 0]).max() < 1e-12
    for j, param in enumerate(["on", "D"]):
        want = _exact(shape, 1.1, param, on=7.0, times=times, width=4.0, k1=3.0)
        assert _worst(got[:, j], want) < 5e-6


@pytest.mark.parametrize("shape", ["closing", "both"])
def test_two_windows_that_share_a_width(tmp_path, shape):
    """Windows [2, 5] and [5, 8], both of width D1. D1 moves the first close at
    1 and the second at 2, so after the first the column is in a frame that is
    not the one the second needs, and has to be asked about all the same. Left
    in the first frame it was 0.2% off."""
    times = [0.0, 1.0, 3.0, 4.9, 5.1, 6.0, 7.0, 7.9, 8.1, 10.0, 12.0]
    model = _two(tmp_path, shape, start2="(on1+D1)", width2="D1")
    got = _column(model, "D1", times=times)

    def total(t_end, by):
        return _pulse(shape, 1.1, t_end, 2.0, 3.0 + by, 2.0) + _pulse(
            shape, 1.1, t_end, 5.0 + by, 3.0 + by, 3.0
        )

    assert _worst(got, _slope(total, times)) < 5e-6


def test_an_exponent_far_above_the_singular_range(tmp_path):
    """Control. a = 61 with a width of 1e6: nothing is singular, but the
    exponent is a parameter, so the power is split all the same, and N^60 and
    D^(-60) each overflow where their product does not. A first cut raised. The
    two are written over the width's own value."""
    text = (
        NET.replace("    4 on    {on}", "    4 on    3e5")
        .replace("    5 D     4.0", "    5 D     1e6")
        .replace("    6 kdeg  0.3", "    6 kdeg  3e-6")
        .format(a=61, close="<=", shape=SHAPES["closing"][0], extra="", tmid=5.0)
    )
    path = tmp_path / "m.net"
    path.write_text(text)
    times = [0.0, 1e5, 4e5, 8e5, 1.2e6, 1.29e6, 1.31e6, 1.5e6, 2e6]
    got = _column(bngsim.Model.from_net(path), "D", times=times)
    want = _exact("closing", 61, "D", on=3e5, times=times, width=1e6, kdeg=3e-6)
    assert _worst(got, want) < 5e-6


@pytest.mark.parametrize("shape", ["closing", "both", "opening"])
def test_a_root_just_after_the_onset_late_in_time(shape):
    """A window at t = 1e6 and a fixed root 5e-4 after its onset. A read 1e-9·t
    back from that root is 1e-3 back, before the onset: the frame was left
    against f from the other side of it, and the onset column was 30% to 64%
    off or raised. The read back is four times the root finder's tolerance."""
    t0 = 1.0e6
    times = [t0 - 1.0, t0 + 1.0, t0 + 2.5, t0 + 3.5, t0 + 3.99, t0 + 4.01, t0 + 5.0, t0 + 7.0]
    model = _sbml(shape, 1.1, extra=" + 0*piecewise(1, time >= 1000000.0005, 0)", on=t0)
    got = _column(model, "on", times=times)
    assert _worst(got, _exact(shape, 1.1, "on", on=t0, times=times)) < 5e-6
