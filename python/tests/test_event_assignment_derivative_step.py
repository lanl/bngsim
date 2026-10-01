"""The state derivative of an event assignment far larger than the state it reads.

The event sensitivity jump needs ∂c/∂x for an assignment ``X = c(x)``. It was a
central difference over a millionth of the state. ``X = X + D`` with D = 100
moves by 2e-9 over that step at X = 1e-3, which keeps five digits of ∂c/∂X = 1,
and by 2e-15 at X = 1e-9, which is under one ulp of 100: the derivative came
back 0, and the sensitivity the species carried into the event was dropped.

A species a zero-order process has run down holds rounding, 1e-12, with a
sensitivity that is not small. A dose into it lost that sensitivity whole.

The derivative by a parameter was taken the same way, over a millionth of the
parameter: ``X = D + q*Y`` with q = 1e-9 gave ∂c/∂q = 0 for Y.

Each difference is now taken a second time over a millionth of the value and
kept where the two agree to the first one's rounding. Where they do not, the
assignment is not linear over the wide step and the narrow one stands.

Every expected value is a closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

TIMES = [0.0, 1.5, 3.0, 4.5, 6.0]


def _end_sens(text, params, **values):
    model = bngsim.Model.from_antimony_string(text)
    for name, value in values.items():
        model.set_param(name, value)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
        sample_times=TIMES, rtol=1e-10, atol=1e-14
    )
    return np.asarray(run.species)[-1, 0], np.asarray(run.sensitivities)[-1, 0, :]


DOSE = (
    "species X; X = 0; a = 2; left = 0; D = 100\n"
    "X = 6 + left\n"
    "J0: X -> ; a\n"
    "E1: at (time >= 3): X = X + D\n"
)


@pytest.mark.parametrize("left", [1e-3, 1e-6, 1e-9, 0.0])
def test_a_dose_into_a_species_that_has_run_down(left):
    """X falls at the rate a from 6 + left and holds `left` at t = 3, where 100
    is added: X(6) = left + 100 − 3·a + ... = 6 + left + 100 − 6·a, so
    dX/da = −6 and dX/dD = 1. With 1e-9 or nothing left, dX/da came back −3:
    the −3 the species carried into the event was multiplied by a ∂c/∂X of 0.
    With 1e-6 left it was −6.006, and with 1e-3 left −6.00001."""
    x, s = _end_sens(DOSE, ["a", "D"], left=left)
    assert x == pytest.approx(6.0 + left + 100.0 - 12.0, rel=1e-9)
    assert s[0] == pytest.approx(-6.0, rel=1e-7)
    assert s[1] == pytest.approx(1.0, rel=1e-7)


def test_a_dose_into_a_species_the_size_of_the_dose_is_as_it_was():
    """Control. With 1 left the narrow difference has its digits already."""
    _x, s = _end_sens(DOSE, ["a", "D"], left=1.0)
    assert s[0] == pytest.approx(-6.0, rel=1e-7)
    assert s[1] == pytest.approx(1.0, rel=1e-7)


def test_an_assignment_that_is_not_linear_over_the_wide_step_keeps_the_narrow_one():
    """Control. c = 100·X/(K + X) at X = 1e-3 with K = 1e-4 is 91, so a millionth
    of the value is a tenth of X and the wide difference is off by a percent.
    The two disagree by far more than the narrow one's rounding, and the narrow
    one is kept: dX/da = −3·c'(1e-3) − 3 with c' = 100·K/(K + X)²."""
    text = (
        "species X; X = 0; a = 2; left = 0; K = 1e-4\n"
        "X = 6 + left\n"
        "J0: X -> ; a\n"
        "E1: at (time >= 3): X = 100*X/(K + X)\n"
    )
    left, big_k = 1e-3, 1e-4
    slope = 100.0 * big_k / (big_k + left) ** 2
    _x, s = _end_sens(text, ["a"], left=left)
    assert s[0] == pytest.approx(-3.0 * slope - 3.0, rel=1e-6)


SMALL_TERM = (
    "species X, Y; X = 1; Y = 3; q = 1; D = 100\n"
    "J0: X -> ; 0.1*X\n"
    "E1: at (time >= 3): X = D + q*Y\n"
)


@pytest.mark.parametrize("q", [1e-3, 1e-6, 1e-9, 1e-12])
def test_a_parameter_whose_term_is_a_small_part_of_the_value(q):
    """X is set to D + q·Y at t = 3 and decays at 0.1 from there, so
    dX/dq = Y·e^(−0.3) and dX/dD = e^(−0.3), whatever q is. dX/dq came back 0
    at q = 1e-9 and below, 2.2266 for 2.2225 at 1e-6, and a millionth off at
    1e-3."""
    _x, s = _end_sens(SMALL_TERM, ["q", "D"], q=q)
    assert s[0] == pytest.approx(3.0 * np.exp(-0.3), rel=1e-7)
    assert s[1] == pytest.approx(np.exp(-0.3), rel=1e-7)


def test_a_parameter_the_size_of_the_value_is_as_it_was():
    """Control."""
    _x, s = _end_sens(SMALL_TERM, ["q", "D"], q=1.0)
    assert s[0] == pytest.approx(3.0 * np.exp(-0.3), rel=1e-7)
    assert s[1] == pytest.approx(np.exp(-0.3), rel=1e-7)


def test_a_value_not_linear_in_the_parameter_over_the_wide_step_keeps_the_narrow_one():
    """Control. c = 100·q/(K + q) at q = 1e-3 with K = 1e-4: the wide step is a
    tenth of q, the two differences disagree, and the narrow one is kept.
    dX/dq = c'(q)·e^(−0.3) with c' = 100·K/(K + q)²."""
    text = (
        "species X; X = 1; q = 1e-3; K = 1e-4\n"
        "J0: X -> ; 0.1*X\n"
        "E1: at (time >= 3): X = 100*q/(K + q)\n"
    )
    _x, s = _end_sens(text, ["q"])
    assert s[0] == pytest.approx(100.0 * 1e-4 / (1.1e-3) ** 2 * np.exp(-0.3), rel=1e-6)
