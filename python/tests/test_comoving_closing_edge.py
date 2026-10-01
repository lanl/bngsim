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
before it, at a stop the run takes a sixteenth short of it along the last
stretch between crossings that leads to it. Not at the crossing that begins the
stretch: a window that opens as a power too, ``s^(a-1)·(1-s)^(a-1)``, has an f
whose slope is unbounded just past the opening. And a frame does not reach a
crossing that is not the column's own: one entered at a crossing is left at
that same stop, a sixteenth short of the next, where it would meet the edge of
a window the column does not move.

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
# is a kink in D there. 6.75 is the stop the D column enters at.
T = [0.0, 1.0, 2.0, 4.0, 5.0, 5.5, 6.5, 6.75, 6.99, 7.01, 8.0, 10.0]

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
    """dX/dD at every sample: one on the stop the column enters at (6.75), and
    one 0.01 either side of the close. It was 0.5% off at a = 1.05 and 1.1e-5
    at a = 1.3, at rtol 1e-8, and the closing shape raised at a = 1.05."""
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


@pytest.mark.parametrize("tmid", [4.0, 6.75, 6.9])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_crossing_inside_the_window_asks_again(tmp_path, shape, tmid):
    """A fixed crossing at tmid, inside the window, restarts the run and leaves
    every frame. Until the run reaches it the close is not the next crossing,
    so the column stays plain, and from tmid it asks for a stop a sixteenth of
    the stretch short of the close: 6.8125, 6.984 or 6.994."""
    extra = "+if(t>=tmid,0.0,0.0)*0"
    got = _column(_model(tmp_path, shape, 1.1, extra=extra, tmid=tmid), "D")
    assert _worst(got, _exact(shape, 1.1, "D")) < 5e-6


@pytest.mark.parametrize(
    ("shape", "level"),
    [("closing", 1.85), ("closing", 2.58), ("both", 3.6), ("both", 4.09)],
    ids=["closing-before-the-stop", "closing-after", "both-before-the-stop", "both-after"],
)
def test_a_state_switch_inside_the_window(tmp_path, shape, level):
    """X crosses the level near t = 6 or t = 6.9, either side of the stop at
    6.75 the D column enters at. A state switch restarts the run and leaves
    every frame, and the column asks for a stop on the stretch from there."""
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
    """A fixed ``time >= 5.5`` inside the window is a root with no record, and
    begins a stretch of its own: the D column enters a sixteenth short of the
    close on it."""
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
    """Control. The onset moves both edges at 1, so its column is in its frame
    across the whole window, from the opening to the close, as it always was.
    Before the window it reads 0."""
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
    so a window that only opens emits the powers it always did."""
    import sympy as sp
    from bngsim import _codegen

    t, on, width, a = sp.symbols("t on D a")
    opening = ((t - on) / width) ** (a - 1)
    closing = (1 - (t - on) / width) ** (a - 1)
    assert _codegen._split_shared_scale(opening, {"t"}, sp) == opening
    split = _codegen._split_shared_scale(closing, {"t"}, sp)
    assert split != closing
    assert {factor.exp for factor in split.args} == {a - 1, 1 - a}
    point = {t: 5.5, on: 3.0, width: 4.0, a: 1.1}
    assert float(split.subs(point)) == pytest.approx(float(closing.subs(point)), rel=1e-14)


def test_the_generator_marks_the_case_a_closing_edge_approaches(tmp_path):
    """A case with a power that closes at its crossing says whether to enter
    ahead of it at the run's values, and, where it has nothing but closing
    powers, that it is of no use when none of them is singular."""
    from bngsim import _codegen

    def table(shape):
        core = _model(tmp_path, shape, 1.1)._core
        src = _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)
        head = "int bngsim_codegen_comoving_approach(int case_idx, const double *p)"
        body = src.split(head)[1].split("\n}")[0]
        return sorted(
            "(0) ? 2 : 0;" not in line for line in body.splitlines() if "if (case_idx ==" in line
        )

    # on and D: the one singular power closes at the crossing each of them
    # moves, and each case can be of no use.
    assert table("closing") == [True, True]
    # D as before; on moves the opening power's edge too, where its frame is of use.
    assert table("both") == [False, True]
    # A window that only opens has no closing power.
    assert table("opening") == []


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


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_an_exponent_far_above_the_singular_range(tmp_path):
    """Control. a = 61 with a width of 1e6: nothing is singular, but the
    exponent is a parameter, so the power is split all the same. Neither column
    is entered ahead, and a first cut, which entered the width's, raised: N^60
    and D^(-60) each overflow where their product does not."""
    text = (
        NET.replace("    4 on    {on}", "    4 on    3e5")
        .replace("    5 D     4.0", "    5 D     1e6")
        .replace("    6 kdeg  0.3", "    6 kdeg  3e-6")
        .format(a=61, close="<=", shape=SHAPES["closing"][0], extra="", tmid=5.0)
    )
    path = tmp_path / "m.net"
    path.write_text(text)
    times = [0.0, 1e5, 4e5, 8e5, 1.2e6, 1.29e6, 1.31e6, 1.5e6, 2e6]
    # The onset column's bar is the oracle's: main is 1.9e-5 from it too.
    for param, bar in (("D", 5e-6), ("on", 5e-5)):
        got = _column(bngsim.Model.from_net(path), param, times=times)
        want = _exact("closing", 61, param, on=3e5, times=times, width=1e6, kdeg=3e-6)
        assert _worst(got, want) < bar


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_a_split_power_is_evaluated_far_above_the_singular_range(tmp_path):
    """Control. Two windows of one width, [3e5, 1.3e6] and [8e5, 1.8e6], at
    a = 61. The width moves both closes at the same rate, and its case has
    nothing but closing powers, each with an exponent of 60, so its column
    stays plain. Entered at the first close and kept to the second, with the
    second window open, it would evaluate the split power there."""
    text = (
        TWO.format(
            a=61,
            on2="8e5",
            D2=4,
            start2="on2",
            width2="D1",
            p1=NET_SHAPES["closing"].format(s="s1()"),
            p2=NET_SHAPES["closing"].format(s="s2()"),
        )
        .replace("    3 on1 2", "    3 on1 3e5")
        .replace("    4 D1 3", "    4 D1 1e6")
        .replace("    9 kdeg 0.3", "    9 kdeg 3e-6")
    )
    path = tmp_path / "two.net"
    path.write_text(text)
    times = [0.0, 1e5, 6e5, 1.0e6, 1.29e6, 1.31e6, 1.5e6, 1.79e6, 1.81e6, 2.2e6]

    def total(t_end, by):
        return _pulse("closing", 61, t_end, 3e5, 1e6 + by, 2.0, 3e-6) + _pulse(
            "closing", 61, t_end, 8e5, 1e6 + by, 3.0, 3e-6
        )

    got = _column(bngsim.Model.from_net(path), "D1", times=times)
    assert _worst(got, _slope(total, times, h=1e2)) < 5e-6


# ─── A frame does not reach a crossing that is not the column's own ─────────


@pytest.mark.parametrize("param", ["on2", "D2", "on1", "D1"])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_frame_is_left_before_another_windows_closing_edge(tmp_path, shape, param):
    """Windows [2, 5] and [4, 10]: each closes while the other is open or has
    just been. A column entered its frame at a crossing it moves and kept it to
    the next restart, which here is the other window's close. On the way there
    the frame carries c·∂f/∂t for that window's closing power, the singular
    forcing again in a column that never had it: 0.2% to 0.6% off, or the
    solver's error. A frame is left a sixteenth short of the next crossing."""
    times = [0.0, 1.0, 3.0, 3.9, 4.1, 4.9, 5.1, 6.0, 8.0, 9.9, 10.1, 12.0]
    got = _column(_two(tmp_path, shape, on2=4, d2=6), param, times=times)
    if param.endswith("2"):
        want = _exact(shape, 1.1, {"on2": "on", "D2": "D"}[param], 4.0, times, 6.0, 3.0)
    else:
        want = _exact(shape, 1.1, {"on1": "on", "D1": "D"}[param], 2.0, times, 3.0, 2.0)
    assert _worst(got, want) < 5e-6


# ─── An exponent that is a parameter ────────────────────────────────────────


def _beside_a_background(tmp_path, a, k0):
    text = NET.replace("    1 k0    0.1", f"    1 k0    {k0!r}").format(
        a=a, close="<=", shape=SHAPES["closing"][0], extra="", tmid=5.0, on=ON
    )
    path = tmp_path / "m.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def test_an_exponent_above_two_is_not_entered_ahead(tmp_path):
    """Control. At a = 3 the window closes as (1-s)², and nothing is unbounded.
    The exponent is a parameter, so the case is emitted all the same, but
    whether its approach is singular is asked at the run's own values. A column
    in its frame reads back as V − c·f, to the tolerance of c·f: beside a
    background flux of 100 a cut that entered ahead here was 1.2e-3 off at
    rtol 1e-6, where the plain column is right to 4e-6."""
    got = _column(_beside_a_background(tmp_path, 3.0, 100.0), "D", rtol=1e-6, atol=1e-8)
    assert _worst(got, _exact("closing", 3.0, "D")) < 5e-5


@pytest.mark.parametrize(("first", "second"), [(3.0, 1.1), (1.1, 3.0)])
def test_the_exponent_is_read_when_the_run_asks(tmp_path, first, second):
    """The code is generated at one value of a and run at another, across 2.
    Decided when the code was written, the model made at a = 3 kept its column
    plain at a = 1.1 and was 0.37% off again. From 1.1 to 3 is a control."""
    model = _model(tmp_path, "closing", first)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["D"])
    sim.run(sample_times=T, rtol=1e-8, atol=1e-10)
    model.reset()
    model.set_param("a", second)
    run = sim.run(sample_times=T, rtol=1e-8, atol=1e-10)
    got = np.asarray(run.sensitivities)[:, 0, 0]
    assert _worst(got, _exact("closing", second, "D")) < 5e-6


# ─── Other ways to write the window ─────────────────────────────────────────


def _written(tmp_path, shape, s, close, window):
    text = NET.replace("    4 on    {on}\n    5 D     4.0\n", window)
    text = text.replace("    1 s() (t-on)/D", f"    1 s() {s}")
    text = text.replace("if(t{close}(on+D)", "if(t{close}" + close)
    text = text.format(a=1.1, close="<=", shape=SHAPES[shape][0], extra="", tmid=5.0)
    path = tmp_path / "m.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


@pytest.mark.parametrize("param", ["on", "off"])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_window_written_from_on_to_off(tmp_path, shape, param):
    """``s = (t − on)/(off − on)``: the closing edge moves with `off` alone, and
    the opening with `on` alone, which also stretches the window. Each column is
    in its frame at one edge and has to be out of it, or in another, at the
    other. dX/d(off) was 0.3% off, and dX/d(on) 0.3% off for the window that
    opens as a power too. (The closing shape's onset column is a control.)"""
    window = "    4 on    3.0\n    5 off   7.0\n"
    model = _written(tmp_path, shape, "(t-on)/(off-on)", "off", window)
    got = _column(model, param)

    def moved(t_end, by):
        on, off = (ON + by, ON + WIDTH) if param == "on" else (ON, ON + WIDTH + by)
        return _pulse(shape, 1.1, t_end, on, off - on)

    assert _worst(got, _slope(moved, T)) < 5e-6


@pytest.mark.parametrize("param", ["D2", "on2"])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_another_windows_close_just_before_the_columns_own(tmp_path, shape, param):
    """Windows [2, 10.9] and [7, 11]. The second window's columns would enter
    a sixteenth of the stretch from 7 short of 11, at 10.75, and the first
    window closes at 10.9, between the two. A column entered there carries the
    first window's closing power to 10.9: 0.2% to 0.3% off, or the solver's
    error. It enters on the stretch from 10.9 instead."""
    times = [0.0, 1.0, 5.0, 8.0, 10.5, 10.8, 10.95, 11.05, 12.0, 14.0]
    text = TWO.replace("    4 D1 3", "    4 D1 8.9").format(
        a=1.1,
        on2=7,
        D2=4,
        start2="on2",
        width2="D2",
        p1=NET_SHAPES[shape].format(s="s1()"),
        p2=NET_SHAPES[shape].format(s="s2()"),
    )
    path = tmp_path / "two.net"
    path.write_text(text)
    got = _column(bngsim.Model.from_net(path), param, times=times)
    which = {"D2": "D", "on2": "on"}[param]
    assert _worst(got, _exact(shape, 1.1, which, 7.0, times, 4.0, 3.0)) < 5e-6


# ─── Too close to an edge to stand off from it ──────────────────────────────


def _late(shape, after):
    t0 = 1.0e6
    times = [t0 - 1.0, t0 + 1.0, t0 + 2.5, t0 + 3.5, t0 + 3.99, t0 + 4.01, t0 + 5.0, t0 + 7.0]
    extra = f" + 0*piecewise(1, time >= {t0 + after!r}, 0)"
    return _sbml(shape, 1.1, extra=extra, on=t0), times, t0


@pytest.mark.parametrize("after", [1e-3, 5e-2])
@pytest.mark.parametrize("shape", ["closing", "both", "opening"])
def test_a_crossing_after_the_onset_late_in_time(shape, after):
    """A window at t = 1e6 and a fixed crossing 1e-3 or 0.05 after its onset.
    A read 1e-9·t back from there is 1e-3 back, on or before the onset: the
    frame was left against f from the wrong side of it, and the onset column
    was 30% to 68% off or raised. The frame is left at a stop of the run's own
    just before the crossing, against f where it is."""
    model, times, t0 = _late(shape, after)
    got = _column(model, "on", times=times)
    assert _worst(got, _exact(shape, 1.1, "on", on=t0, times=times)) < 5e-6


@pytest.mark.parametrize("after", [5e-4, 5e-5])
@pytest.mark.parametrize("shape", ["closing", "both", "opening"])
def test_a_crossing_too_soon_after_the_onset_is_refused(shape, after):
    """The same with the crossing 5e-4 or 5e-5 after the onset. A column left
    plain that close to an edge that opens as a power has an unbounded forcing
    behind it: 1e-4 to 8e-2 off, measured out to 4.4e-4 at this time. Main
    raised or was 50% to 64% off. Inside 7.3e-4 here the run is refused, and
    for a window that only closes as a power too, where it need not be."""
    model, times, _t0 = _late(shape, after)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["on"])
    with pytest.raises(Exception, match="after a switch time.*issue #760"):
        sim.run(sample_times=times, rtol=1e-8, atol=1e-10)


def test_a_run_that_starts_too_close_before_the_close_is_refused():
    """A run that starts 1e-10 before the window closes. The D column would
    have to enter its frame inside that, with the forcing unbounded on both
    sides of wherever it did: 1.2e-3 off, where main fails the run."""
    sim = bngsim.Simulator(_sbml("closing", 1.1), method="ode", sensitivity_params=["D"])
    with pytest.raises(Exception, match="before a switch time.*issue #760"):
        sim.run(sample_times=[7.0 - 1e-10, 7.5, 8.0, 10.0], rtol=1e-8, atol=1e-10)


# ─── A close that is not a round number ──────────────────────────────────────


@pytest.mark.parametrize("param", ["on", "D"])
@pytest.mark.parametrize("a", [1.1, 2.01, 2.5])
def test_a_close_that_does_not_round_onto_its_own_edge(tmp_path, a, param):
    """on = 1.2 and D = 2.604, a window whose ``on + D`` is not where
    ``(t − on)/D`` reaches 1 in floating point. Every other window here closes
    on a time that is exact.

    At a = 2.01 and 2.5 this is a control: main is right, and an earlier cut
    failed the run. Nothing is singular there, but the exponent is a parameter,
    so the width has a case all the same, and its column entered that frame at
    the close, where the closing power's base rounds to just under 0 and a
    power of it is not a number. A case with nothing but closing powers, none
    of them singular at the run's values, is now left alone."""
    on, width = 1.2, 2.604
    text = NET.replace("    5 D     4.0", f"    5 D     {width!r}").format(
        a=a, close="<=", shape=SHAPES["closing"][0], extra="", tmid=5.0, on=repr(on)
    )
    path = tmp_path / "m.net"
    path.write_text(text)
    times = sorted(
        {0.0, 0.6, on + 0.03 * width, on + 0.5 * width, on + 0.9 * width, on + 0.9975 * width}
        | {on + 1.0025 * width, on + width + 1.0, on + width + 3.0}
    )
    got = _column(bngsim.Model.from_net(path), param, times=times)
    want = _exact("closing", a, param, on=on, times=times, width=width)
    assert _worst(got, want) < 5e-6


# ─── A crossing on the edge itself ──────────────────────────────────────────


@pytest.mark.parametrize("shift", [-1e-12, 0.0, 1e-12])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_step_on_the_closing_edge_is_refused(tmp_path, shape, shift):
    """A zero-order step ``if(t >= tj, kj, 0)`` on the close, or 1e-12 either
    side. Two crossings on one instant are told apart by raising one threshold
    a hair with the clock past the instant (issue #375), which reads this
    window's closing power a hair from its zero: a jump of 4% of the window's
    height where there is none, and dX/dD 27% to 41% off, on main too (issue
    #949). A crossing that shares its instant and is the edge of such a window
    is refused. The step 1e-12 before the close is met first, as a restart too
    close before the edge."""
    text = NET.replace("    8 tmid  {tmid}", "    8 tmid  {tmid}\n    9 tj " + repr(7.0 + shift))
    text = text.format(
        a=1.1, close="<=", shape=SHAPES[shape][0], extra="+if(t>=tj,1.5,0)", tmid=5.0, on=ON
    )
    path = tmp_path / "m.net"
    path.write_text(text)
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=["D"])
    reason = "before a switch time.*issue #760" if shift < 0 else "shares its instant.*issue #949"
    with pytest.raises(Exception, match=reason):
        sim.run(sample_times=T, rtol=1e-8, atol=1e-10)


def test_a_step_on_an_edge_that_is_not_singular_runs(tmp_path):
    """Control. The step on the close again, at a = 3. The closing power is
    ``(1 − s)²`` there, and read a hair from its zero it is nothing, so the two
    crossings are told apart as they are anywhere else. An earlier cut refused
    every crossing that shares its instant with the edge of a window."""
    text = NET.replace("    8 tmid  {tmid}", "    8 tmid  {tmid}\n    9 tj 7.0")
    text = text.format(
        a=3, close="<=", shape=SHAPES["closing"][0], extra="+if(t>=tj,1.5,0)", tmid=5.0, on=ON
    )
    path = tmp_path / "m.net"
    path.write_text(text)
    got = _column(bngsim.Model.from_net(path), "D")
    assert _worst(got, _exact("closing", 3, "D")) < 5e-6


@pytest.mark.parametrize("param", ["on1", "D1", "on2"])
def test_abutting_windows_that_are_not_singular_run(tmp_path, param):
    """Control. Windows [2, 5] and [5, 9] at a = 3: the first closes where the
    second opens, and neither edge is singular. An earlier cut refused it."""
    times = [0.0, 1.0, 3.0, 4.9, 5.1, 6.0, 8.0, 8.9, 9.1, 11.0, 13.0]
    model = _two(tmp_path, "closing", a=3, on2=5, d2=4)

    def total(t_end, by):
        first = (
            (2.0 + by, 3.0) if param == "on1" else (2.0, 3.0 + by) if param == "D1" else (2.0, 3.0)
        )
        second = 5.0 + by if param == "on2" else 5.0
        return _pulse("closing", 3, t_end, first[0], first[1], 2.0) + _pulse(
            "closing", 3, t_end, second, 4.0, 3.0
        )

    assert _worst(_column(model, param, times=times), _slope(total, times)) < 5e-6


@pytest.mark.parametrize("shape", ["closing", "both", "opening"])
def test_a_crossing_within_an_instant_of_the_onset_is_refused(shape):
    """A fixed crossing 2e-7 after an onset at t = 1e6 has the onset's twelve
    digits, so the detector takes the two for one instant. Where the window
    closes as a singular power that is a crossing on an edge the onset column
    was entered ahead of. Where it only opens, it is a restart too soon after
    the onset."""
    model, times, _t0 = _late(shape, 2e-7)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["on"])
    reason = "after a switch time.*issue #760" if shape == "opening" else "issue #949"
    with pytest.raises(Exception, match=reason):
        sim.run(sample_times=times, rtol=1e-8, atol=1e-10)


# ─── An exponent below 1, and how many cases ────────────────────────────────


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("param", ["on", "D"])
@pytest.mark.parametrize(("shape", "a"), [("closing", 0.9), ("closing", 0.5), ("both", 0.9)])
def test_a_window_that_diverges_at_its_edge(tmp_path, shape, a, param):
    """With a below 1 the window itself is unbounded at the edge, integrably:
    (1-s)^(a-1) with a negative exponent. Its derivative is unbounded there as
    for 1 < a < 2, and the approach is as much one to enter ahead of. A cut
    that asked for an exponent between 0 and 1 left the onset column plain
    from the stop where frames are left, and the run failed where main is
    right: the onset column is a control. The D column fails the run on
    main."""
    got = _column(_model(tmp_path, shape, a), param)
    assert _worst(got, _exact(shape, a, param)) < 5e-6


@pytest.mark.parametrize(("shape", "cases"), [("closing", 2), ("both", 2), ("opening", 1)])
def test_one_shift_is_one_case(tmp_path, shape, cases):
    """`on` and `D` each have one case for a window that closes as a power, and
    `on` alone for one that only opens. The split base carries its scale's
    value as a float, so the shift read off it was 1.0 beside the opening
    base's 1: two keys and two cases for one frame, three in all for the window
    singular at both edges, and twice the source to derive for twenty such
    windows. (The window that only opens is a control.)"""
    from bngsim import _codegen

    core = _model(tmp_path, shape, 1.1)._core
    src = _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)
    body = src.split("int bngsim_codegen_comoving_case(")[1].split("\n}\n")[0]
    assert body.count("*c_out =") == cases


def test_the_source_does_not_read_a_parameters_value(tmp_path):
    """The generated source is cached under a key that leaves parameter values
    out, so two models that differ only in a value share one artifact. A cut
    that wrote the split power over the width's value, and split only where
    that value was positive, emitted a source that the next model with the same
    structure did not ask for."""
    from bngsim import _codegen

    def source(on, width, a):
        text = NET.replace("    5 D     4.0", f"    5 D     {width!r}").format(
            a=a, close="<=", shape=SHAPES["closing"][0], extra="", tmid=5.0, on=repr(on)
        )
        path = tmp_path / f"m_{on}_{width}_{a}.net"
        path.write_text(text)
        core = bngsim.Model.from_net(path)._core
        return _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)

    assert source(3.0, 4.0, 1.1) == source(1.2, 2.604, 2.5) == source(3.0, -4.0, 1.7)
