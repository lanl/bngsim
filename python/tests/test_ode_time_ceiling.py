"""Issue #1838: an ODE run over a span CVODE cannot step through.

``A -> B`` at k = 1 over ``(0, 7e307)`` never returned, and ``timeout=5`` did
not stop it. ``(1e308, 1.7e308)`` is the same run once #1753 moves it onto its
own clock; for a model that reads the clock it is not moved, and came back as
``[1, 0, nan, nan, nan]`` with no error, beside a ``RoundoffStepWarning`` that
blamed steps too small to move the clock.

CVODE is given no largest step, and grows its step up to ten times at a time.
Within a factor of ten of the largest double the step overflows to inf before
the run reaches its end, and the clock with it. With h = inf, the test that asks
the Newton iteration for a fresh Jacobian reads inf / inf, a NaN, and never
asks, so the iteration retries for ever inside one step: 16.7 million calls of
the right-hand side in 8 s, the step count fixed at 171. ``max_steps`` counts
steps and could not see it, and the wall-clock budget was read only between
CVode calls. Where the clock passed the end at inf instead, CVODE interpolated
the rows there, and they were NaN.

An ODE run is refused, before it starts, where a time in it is 2^512 or more
from 0, or the run is 2^512 or longer.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest

CEILING = 2.0**512  # the first time an ODE run may not reach, ~1.34e154
BELOW = math.nextafter(CEILING, 0.0)


def _net(law, a0=1.0):
    # A -> B at the rate f()*A, from A = a0.
    return (
        "begin parameters\n 1 k 1\nend parameters\n"
        f"begin functions\n 1 f() {law}\nend functions\n"
        f"begin species\n 1 A() {a0}\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 f\nend reactions\n"
        "begin groups\n 1 Atot 1\n 2 Btot 2\nend groups\n"
    )


# $S -> $S + A at k = 1: A' = 1, so A = t - t_start, which BDF integrates
# exactly whatever its step.
CLOCK_NET = (
    "begin parameters\n 1 k 1\nend parameters\n"
    "begin species\n 1 A() 0\n 2 $S() 1\nend species\n"
    "begin reactions\n 1 2 1,2 k\nend reactions\n"
    "begin groups\n 1 Atot 1\nend groups\n"
)

READS_NO_CLOCK = "k"
READS_THE_CLOCK = "k*(1+0*time())"  # not moved onto the run's own clock


def _sim(tmp_path, text, **kw):
    net = tmp_path / "m.net"
    net.write_text(text)
    return bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", **kw)


def _roundoff_warnings(caught):
    return [w for w in caught if issubclass(w.category, bngsim.RoundoffStepWarning)]


SPANS_PAST_THE_CEILING = [
    pytest.param((0.0, 7e307), id="the issue's run that never returned"),
    pytest.param((1e308, 1.7e308), id="the issue's run with nan rows"),
    pytest.param((-1.7e308, -1e308), id="a far run below 0"),
    pytest.param((-7e307, 0.0), id="a far start that ends at 0"),
    pytest.param((0.0, CEILING), id="an end at the ceiling"),
    pytest.param((-CEILING / 2, CEILING / 2), id="a run as long as the ceiling"),
]


@pytest.mark.parametrize("law", [READS_NO_CLOCK, READS_THE_CLOCK], ids=["no clock", "clock"])
@pytest.mark.parametrize("span", SPANS_PAST_THE_CEILING)
def test_a_span_cvode_cannot_step_through_is_refused_before_it_runs(tmp_path, span, law):
    sim = _sim(tmp_path, _net(law))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # timeout=5 as in the issue, so that a refusal that goes missing fails
        # here rather than hanging the suite.
        with pytest.raises(bngsim.UsageError, match=r"2\^512") as info:
            sim.run(span, 5, timeout=5)
    assert isinstance(info.value, ValueError)
    assert info.value.code == "usage.invalid_argument"
    # It took no step, so none too small to move the clock: (1e308, 1.7e308)
    # of the model that reads the clock warned of 155 alongside its nan rows.
    assert _roundoff_warnings(caught) == []
    assert all(n["code"] != "solver.roundoff_steps" for n in info.value.notices)
    # And it moved nothing: the same Simulator runs a span inside the ceiling,
    # A(t) = e^-t.
    assert sim.current_time == 0.0
    r = sim.run((0.0, 5.0), 6)
    np.testing.assert_allclose(
        np.asarray(r.species)[:, 0], np.exp(-np.arange(6.0)), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize(
    "span",
    [
        pytest.param((1e308, 1.7e308), id="far from 0"),
        pytest.param((-CEILING / 2, CEILING / 2), id="too long"),
    ],
)
def test_the_refusal_says_what_to_do(tmp_path, span):
    with pytest.raises(bngsim.UsageError) as info:
        _sim(tmp_path, _net(READS_NO_CLOCK)).run(span, 5, timeout=5)
    message = str(info.value)
    assert "larger unit" in message
    assert "too long" in message if span[0] < 0.0 < span[1] else "from t = 0" in message


def test_run_batch_refuses_it_too(tmp_path):
    sim = _sim(tmp_path, _net(READS_NO_CLOCK))
    with pytest.raises(bngsim.UsageError, match=r"2\^512"):
        sim.run_batch(t_span=(0.0, 7e307), n_points=5, params=[{"k": 1.0}, {"k": 2.0}], timeout=5)


@pytest.mark.parametrize(
    "span",
    [
        pytest.param((0.0, BELOW), id="from 0 to just below the ceiling"),
        pytest.param((CEILING / 4, BELOW), id="a far run to just below it"),
        pytest.param((-BELOW / 2, BELOW / 2), id="just shorter than it"),
    ],
)
def test_a_span_just_inside_the_ceiling_still_runs(tmp_path, span):
    # A' = 1, so A = t - t_start at every row. The run below 0 takes steps too
    # small to move the clock out there, and says so; its rows are right.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.RoundoffStepWarning)
        r = _sim(tmp_path, CLOCK_NET).run(span, 5, timeout=5)
    t = np.asarray(r.time)
    assert t[0] == span[0] and t[-1] == span[1]
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], t - span[0], rtol=1e-6, atol=0.0)
