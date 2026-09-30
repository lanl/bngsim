"""Issue #735 — an event assignment that reads ``time`` carries ∂h/∂t·∂t*/∂p.

For x⁺ = h(x⁻(t*), p, t*(p)) the total derivative of the assigned value is

    ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p + ∂h/∂t·∂t*/∂p

and the jump computed every term but the last, because each difference was
taken at the fixed fire time. An assignment that reads the clock (``Tlast =
time``, ``END_M = time + 1000``, as on BIOMD0000000675) got a sensitivity short
by exactly ∂h/∂t·∂t*/∂p, with no warning, and so did everything downstream.

Closed forms: A(t) = e^(−k·t) with k = 0.5. The state trigger ``A < 0.5`` fires at
t* = ln2/k, so dt*/dk = −ln2/k²; the clock trigger ``time >= T0`` fires at
t* = T0, so dt*/dT0 = 1.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

K, T0 = 0.5, 1.3


def _dB(event: str, param: str) -> float:
    text = (
        f"compartment C = 1; species A in C, B in C; A = A0; B = 0; k = {K}; T0 = {T0}; A0 = 1\n"
        f"J1: A -> ; k*A\nf := time^2\nE1: at ({event}\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=[param]).run(
        t_span=(0.0, 3.0), n_points=4, rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    return float(np.asarray(run.sensitivities)[-1, names.index("B"), 0])


@pytest.mark.parametrize(
    "event,param,exact",
    [
        # Each missed exactly ∂h/∂t·∂t*/∂p before the fix.
        ("A < 0.5): B = time + 1", "k", -math.log(2) / K**2),
        ("A < 0.5): B = k*time", "k", 0.0),  # t* + k·dt*/dk = 0
        ("time >= T0): B = time", "T0", 1.0),
        ("time >= T0): B = A + time", "T0", 1.0 - K * math.exp(-K * T0)),
        # Controls with no ∂h/∂t term, right before and after.
        ("time >= T0): B = A", "T0", -K * math.exp(-K * T0)),
        ("A < 0.5): B = 3*k", "k", 3.0),
        # Time read through a rule: the functions must be re-evaluated at each
        # time offset, not only the clock moved. dB/dk = 2·t*·dt*/dk.
        ("A < 0.5): B = f", "k", 2 * (math.log(2) / K) * (-math.log(2) / K**2)),
    ],
    ids=[
        "state-time",
        "state-k-time",
        "clock-time",
        "clock-A-plus-time",
        "clock-A",
        "state-3k",
        "state-rule-time-squared",
    ],
)
def test_the_assigned_value_carries_its_time_dependence(event, param, exact):
    assert _dB(event, param) == pytest.approx(exact, rel=1e-6, abs=1e-9)


def test_an_initial_condition_column_carries_the_time_term_too():
    """Under a state trigger an IC column moves the fire time as well:
    t* = ln(2·A0)/k, so dt*/dA0 = 1/(k·A0) = 2 and B = time gives dB/dA0 = 2."""
    text = (
        f"compartment C = 1; species A in C, B in C; A = A0; B = 0; k = {K}; A0 = 1\n"
        "J1: A -> ; k*A\nE1: at (A < 0.5): B = time\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_ic=["A"]).run(
        t_span=(0.0, 3.0), n_points=4, rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    got = float(np.asarray(run.sensitivities_ic)[-1, names.index("B"), 0])
    assert got == pytest.approx(1.0 / (K * 1.0), rel=1e-6)
