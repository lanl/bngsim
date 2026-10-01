"""A switch time's jump is taken at its own instant (issue #737).

A fitted switch time's whole gradient is the jump ``s⁺ = s⁻ + (f⁻ − f⁺)·∂t*/∂p``
at its crossing (issue #48). The core stops the integrator on ``t*`` and reads
``f⁻`` and ``f⁺`` by nudging the clock 64 ulp either side of it.

Whether the run had *reached* a crossing was decided with a window of
``1e-9·max(1, horizon)``, a billion times wider than that nudge. Any return
that fell inside the window short of ``t*`` was taken for the crossing: an
output time, an event root, another switch's stop. The jump was then read from
a bracket that sat wholly on the before-branch, so it was 0 or a neighbouring
switch's, and the crossing was marked done. The real one passed with no jump. A
switch that close after the start of the run was dropped as already behind. All
with no warning, and the trajectory right.

Now a crossing is reached only when the run stands within the nudge's own reach
of it, and the detector groups crossings by that same reach.

Each model here is ``X' = k`` from its switch time on, so ``X(t) = k·(t − τ)``
and ``dX/dτ = −k`` for ``t > τ``, exactly.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _switch_sensitivity as sw
from bngsim._switch_sensitivity import _q

ONE = "species X; X = 0; k = 2; tau = {tau!r}\nJ0: -> X; piecewise(k, time >= tau, 0)\n"
TWO = (
    "species X, Y; X = 0; Y = 0; k1 = 2; k2 = 3; s1 = {s1!r}; s2 = {s2!r}\n"
    "J0: -> X; piecewise(k1, time >= s1, 0)\n"
    "J1: -> Y; piecewise(k2, time >= s2, 0)\n"
)
DAILY = [float(t) for t in range(101)]


def _sens(text, params, times, exact=None):
    """`exact` sets parameters to the double itself: a value written into the
    model text keeps 15 significant digits, which is not enough to place two
    switch times an ulp apart."""
    model = bngsim.Model.from_antimony_string(text)
    for name, value in (exact or {}).items():
        model.set_param(name, value)
        assert model.get_param(name) == value
    run = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
        sample_times=list(times), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    s = np.asarray(run.sensitivities)
    return {n: s[:, i, :] for i, n in enumerate(names)}, np.asarray(run.species)


@pytest.mark.parametrize("past", [5e-8, 1e-9, 1e-11, 1e-13, 0.0, -5e-8])
def test_a_switch_just_after_an_output_time(past):
    """τ sits `past` after the output at t = 30 on a horizon of 100, where the
    old window was 1e-7. From 5e-8 down to 1e-11 the output's return was taken
    for the crossing and dX/dτ was 0 at every sample. At 1e-13 the two are one
    instant by the nudge's own reach."""
    tau = 30.0 + past
    s, x = _sens(ONE.format(tau=tau), ["tau"], DAILY)
    t = np.asarray(DAILY)
    after = t > 30.5
    np.testing.assert_allclose(s["X"][after, 0], -2.0, rtol=1e-9)
    np.testing.assert_allclose(s["X"][t < 29.5, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(x[after, 0], 2.0 * (t[after] - tau), rtol=1e-9)


@pytest.mark.parametrize("short", [2e-9, 1e-12])
def test_an_output_time_just_before_a_switch(short):
    """The issue's first reproduction: a sample `short` before τ = 5 on a
    horizon of 10."""
    s, _ = _sens(ONE.format(tau=5.0), ["tau"], [0.0, 2.0, 5.0 - short, 8.0, 10.0])
    np.testing.assert_allclose(s["X"][:, 0], [0.0, 0.0, 0.0, -2.0, -2.0], atol=1e-9)


@pytest.mark.parametrize("gap", [3e-8, 1e-10])
def test_an_event_root_just_before_a_switch(gap):
    """An unrelated event fires `gap` before τ. Its root return was taken for
    the crossing."""
    text = (
        f"species X, Z; X = 0; Z = 0; k = 2; tau = {30.0 + gap!r}\n"
        "J0: -> X; piecewise(k, time >= tau, 0)\n"
        "E: at (time >= 30): Z = 1\n"
    )
    s, x = _sens(text, ["tau"], DAILY)
    np.testing.assert_allclose(s["X"][31:, 0], -2.0, rtol=1e-9)
    assert x[-1, 1] == 1.0


@pytest.mark.parametrize("gap", [5e-8, 1e-10, 1e-12])
def test_two_switches_a_hair_apart_keep_their_own_columns(gap):
    """s1 = 35 turns X on and s2 = 35 + gap turns Y on. Stopped at s1, the old
    window applied s2's record too, from a bracket about s1 that s2 is not in:
    d[X, Y]/ds2 came out [-2, 0], s1's jump in s2's column and s2's own lost.
    Each is now stopped at on its own."""
    s, _ = _sens(TWO.format(s1=35.0, s2=35.0 + gap), ["s1", "s2"], DAILY)
    np.testing.assert_allclose(s["X"][-1], [-2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(s["Y"][-1], [0.0, -3.0], atol=1e-9)


@pytest.mark.parametrize("tau", [5e-8, 1e-12])
def test_a_switch_just_after_the_start_of_the_run(tau):
    """τ a hair after t = 0 was dropped as a crossing already behind."""
    s, _ = _sens(ONE.format(tau=tau), ["tau"], DAILY)
    np.testing.assert_allclose(s["X"][1:, 0], -2.0, rtol=1e-9)


@pytest.mark.parametrize("sens", [False, True], ids=["plain", "sensitivities"])
@pytest.mark.parametrize("k", [1e6, 1e12])
def test_a_fixed_crossing_just_after_an_output_time(sens, k):
    """The same window decided when a crossing no parameter moves had been
    reached (issue #305's stop). With the jump in the rate law at 30.00000005
    and an output at 30, the output's return was taken for the crossing, the
    run restarted there, and the real jump 5e-8 later had no stop: the step
    across it cannot pass the error test, and the run ended in CVODE's
    no-progress error, with or without sensitivities."""
    tau = 30.00000005
    text = (
        f"species X, Y; X = 0; Y = 1; k = {k!r}; d = 0.3\n"
        f"J0: -> X; piecewise(k, time >= {tau!r}, 0)\n"
        "J1: Y -> ; d*Y*X\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    kw = {"sensitivity_params": ["k", "d"]} if sens else {}
    run = bngsim.Simulator(model, method="ode", **kw).run(
        sample_times=DAILY, rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)[-1, names.index("X")]
    assert x == pytest.approx(k * (100.0 - tau), rel=1e-9)
    if sens:
        s = np.asarray(run.sensitivities)[-1, names.index("X")]
        np.testing.assert_allclose(s, [100.0 - tau, 0.0], rtol=1e-9, atol=1e-12)


# ─── what the nudge flips together is one group ─────────────────────────────


def _bucket_edge_pair():
    """Two doubles 2 ulp apart that the 12-digit key puts in different buckets."""
    edge = 35.00000000005
    a, b = np.nextafter(edge, 0.0), np.nextafter(edge, 100.0)
    assert _q(float(a)) != _q(float(b)) and b - a < 3 * np.spacing(edge)
    return float(a), float(b)


def test_two_switches_either_side_of_a_bucket_edge_are_isolated():
    """s1 and s2 are 2 ulp apart, so one nudge of the clock flips both. The
    detector grouped crossings by a 12-digit key, which these two straddle, so
    neither was isolated, and every column came out 0."""
    s1, s2 = _bucket_edge_pair()
    s, _ = _sens(TWO.format(s1=35.0, s2=35.0), ["s1", "s2"], DAILY, exact={"s1": s1, "s2": s2})
    np.testing.assert_allclose(s["X"][-1], [-2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(s["Y"][-1], [0.0, -3.0], atol=1e-9)


def test_two_switches_inside_the_nudge_below_t_equal_one():
    """Below t = 1 the nudge is 64 ulp of 1, about 1.4e-14, which is wider than
    the 12-digit key's buckets there. Two switches 1e-14 apart at t = 1e-3 are
    flipped together and were not grouped."""
    s1 = 1e-3
    s2 = s1 + 1e-14
    assert _q(s1) != _q(s2)
    s, _ = _sens(TWO.format(s1=s1, s2=s2), ["s1", "s2"], DAILY)
    np.testing.assert_allclose(s["X"][-1], [-2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(s["Y"][-1], [0.0, -3.0], atol=1e-9)


COUNTER = """\
begin parameters
    1 c0 1e6
    2 thr1 1000001
    3 thr2 {thr2!r}
    4 k1 2
    5 k2 3
    6 one 1
end parameters
begin functions
    1 f1() if(Cobs>=thr1,k1,0)
    2 f2() if(Cobs>=thr2,k2,0)
end functions
begin species
    1 C() c0
    2 X() 0
    3 Y() 0
end species
begin reactions
    1 0 1 one
    2 0 2 f1
    3 0 3 f2
end reactions
begin groups
    1 Cobs 1
end groups
"""


@pytest.mark.parametrize("gap", [5e-9, 2e-10])
def test_two_thresholds_on_a_counter_clock_inside_its_nudge(tmp_path, gap):
    """A counter that starts at 1e6 is nudged by 64 ulp of 1e6, about 1.4e-8,
    to read a crossing. Two thresholds on it `gap` apart are both inside that,
    while their crossing times, `gap` apart at t = 1, are two instants and two
    12-digit keys. So neither was isolated and each column took both jumps:
    d[X, Y]/dthr1 came out [-2, -3]."""
    path = tmp_path / "counter.net"
    path.write_text(COUNTER.format(thr2=1000001.0 + gap))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["thr1", "thr2"]
    ).run(t_span=(0.0, 4.0), n_points=5, rtol=1e-10, atol=1e-12)
    names = list(run.species_names)
    s = np.asarray(run.sensitivities)[-1]
    np.testing.assert_allclose(s[names.index("X()")], [-2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(s[names.index("Y()")], [0.0, -3.0], atol=1e-9)


class _C:
    """The three fields `_instant_groups` reads."""

    def __init__(self, t_star, clock_idx0=-1, threshold=None):
        self.t_star = t_star
        self.clock_idx0 = clock_idx0
        self.threshold = t_star if threshold is None else threshold


def _sizes(crossings):
    return sorted(len(g) for g in sw._instant_groups(crossings))


def test_the_grouping_is_by_the_reach_of_the_nudge():
    a, b = _bucket_edge_pair()
    assert sw._same_instant(a, b)
    assert _sizes([_C(a), _C(b)]) == [2]
    # 5e-8 apart is two instants: each gets its own stop.
    assert _sizes([_C(35.0), _C(35.00000005)]) == [1, 1]
    # One 12-digit key still groups, as before the issue.
    assert _q(35.0) == _q(35.0 + 1e-12) and not sw._same_instant(35.0, 35.0 + 1e-12)
    assert _sizes([_C(35.0), _C(35.0 + 1e-12)]) == [2]
    # Grouping chains: a-b and b-c within reach puts all three together.
    step = 100 * np.spacing(35.0)
    assert _sizes([_C(35.0), _C(35.0 + step), _C(35.0 + 2 * step)]) == [3]


def test_a_counter_clock_is_grouped_by_its_thresholds():
    """A counter that starts at 1e6 is nudged by 64 ulp of 1e6, about 1.4e-8,
    whatever the time. Two thresholds on it 5e-9 apart are flipped together
    though their times, 5e-9 apart at t = 1, are two instants."""

    def pair(gap, clock_b=3):
        return [
            _C(1.0, clock_idx0=3, threshold=1e6),
            _C(1.0 + gap, clock_idx0=clock_b, threshold=1e6 + gap),
        ]

    assert _sizes(pair(5e-9)) == [2]
    assert _sizes(pair(5e-9, clock_b=4)) == [1, 1]
    assert _sizes(pair(5e-7)) == [1, 1]
