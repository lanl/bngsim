"""An ODE output point too close to the time a run stands at for CVODE.

CVODE does not integrate over less than about twice the doubles' spacing at
the time it stands at ("tout too close to t0"), and bngsim takes an output
point inside that band as reached, its row the state where the run stands
(issue #1501): replicate samples at t = 0, ``[0, 0, 0, 5]``. Far from 0 the band
is wide: at t = 1e17 the doubles are 16 apart and the band 44. A -> B at k = 1
over (1e17, 1e17 + 64) came back B = 0 at t0 + 16 and t0 + 32, where B is 1,
and over (1e17, 1e17 + 10) its 11 output times rounded onto t0 six times and
t0 + 16 five, every row B = 0: no error and no warning.

Such an output is refused where any species would move past its error weight
over the distance, and a grid finer than the doubles across it is refused
before the run starts. An output inside the band where the model barely moves
keeps the row it had, as repeated samples do.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

DECAY = "species A = 1, B = 0; k = 1; J1: A -> B; k*(1 + 0*time)*A;"  # reads the clock
EVENT = DECAY + " E1: at time > 2e17: A = 1;"


def _sim(text: str = DECAY, **kw) -> bngsim.Simulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode", **kw)


@pytest.mark.parametrize(("span", "n_points"), [(10.0, 11), (16.0, 5), (32.0, 5)])
def test_a_grid_finer_than_the_doubles_is_refused_before_it_runs(span, n_points):
    sim = _sim()
    t0 = 1e17
    with pytest.raises(bngsim.UsageError, match="round onto each other") as info:
        sim.run(t_span=(t0, t0 + span), n_points=int(n_points), start="initial")
    assert info.value.code == "usage.span_out_of_reach"
    assert sim.current_time == 0.0


@pytest.mark.parametrize(
    ("text", "times"),
    [
        (DECAY, {"t_span": (1e17, 1e17 + 64), "n_points": 5}),
        (DECAY, {"t_span": (1e17, 1e17 + 160), "n_points": 11}),
        (DECAY, {"sample_times": [1e17, 1e17 + 16, 1e17 + 64]}),
        (DECAY, {"t_span": (1e17, 1e17 + 10), "n_points": 2}),
        # An event takes the run down the other loop.
        (EVENT, {"sample_times": [1e17, 1e17 + 16, 1e17 + 64]}),
    ],
    ids=["grid", "a longer grid", "sample times", "two points", "the event loop"],
)
def test_an_output_inside_the_band_where_the_model_moves_is_refused(text, times):
    sim = _sim(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(bngsim.UsageError, match="CVODE cannot integrate over") as info:
            sim.run(**times, start="initial")
    assert info.value.code == "usage.span_out_of_reach"
    assert "larger unit" in str(info.value)
    assert not [w for w in caught if issubclass(w.category, bngsim.RoundoffStepWarning)]
    # And it moved nothing: the same Simulator runs a span inside reach.
    r = sim.run(t_span=(0.0, 5.0), n_points=6, start="initial")
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], np.exp(-np.arange(6.0)), rtol=1e-5)


def test_repeated_samples_at_the_start_keep_their_rows():
    # Issue #1501: the start state at each time named again.
    r = _sim().run(sample_times=[0.0, 0.0, 0.0, 5.0], start="initial")
    s = np.asarray(r.species)
    np.testing.assert_array_equal(s[:3, 1], 0.0)
    np.testing.assert_allclose(s[3, 1], 1.0 - np.exp(-5.0), rtol=1e-6)


def test_an_output_inside_the_band_where_the_model_barely_moves_is_kept():
    # k = 1e-30: over 16 time units B moves 1.6e-29, under its error weight,
    # so the start state is the row there, as it was.
    r = _sim(DECAY.replace("k = 1;", "k = 1e-30;")).run(
        sample_times=[1e17, 1e17 + 16, 1e17 + 1e20], start="initial"
    )
    s = np.asarray(r.species)
    assert s[1, 1] == 0.0
    np.testing.assert_allclose(s[2, 1], 1e-10, rtol=1e-6)
