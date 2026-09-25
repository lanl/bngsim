"""GH #712: the periodic dosing step bound covers schedules slower than t = 128.

`_periodic_time_disc_max_step` scanned at most t in [0, 128] for dose-window
edges. A schedule slower than about 51 time units (weekly dosing in hours,
daily in minutes) or a first pulse after t = 127 left fewer than two edges, or
only the off-interval, in view; the bound came back None, and CVODE stepped
over every pulse after the first. The window now scales with the fastest
period, so it holds several complete cycles, and each component of the scan's
signature is bisected on its own, so a pulse narrower than the grid pitch is
still seen as two edges rather than one.

dose = floor((t - t0)/P) - floor((t - t0 - w)/P) is 1 on [t0 + nP, t0 + nP + w)
and 0 elsewhere, so X' = dose gives X(T) = w * (number of pulses begun by T).
"""

from __future__ import annotations

import bngsim
import pytest

_T = "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>"


def _floor_shift(shift: float, period: float) -> str:
    return (
        f"<apply><floor/><apply><divide/><apply><minus/>{_T}<cn>{shift}</cn></apply>"
        f"<cn>{period}</cn></apply></apply>"
    )


def _model(period: float, width: float, t0: float) -> bngsim.Model:
    dose = f"<apply><minus/>{_floor_shift(t0, period)}{_floor_shift(t0 + width, period)}</apply>"
    doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="pulses"><listOfParameters>
<parameter id="dose" value="0" constant="false"/><parameter id="X" value="0" constant="false"/>
</listOfParameters><listOfRules>
<assignmentRule variable="dose"><math xmlns="http://www.w3.org/1998/Math/MathML">{dose}</math></assignmentRule>
<rateRule variable="X"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>dose</ci></math></rateRule>
</listOfRules></model></sbml>"""
    return bngsim.Model.from_sbml_string(doc)


@pytest.mark.parametrize(
    ("period", "width", "t0", "horizon"),
    [
        (24.0, 1.0, 0.0, 264.0),  # daily in hours: worked before, must still
        (168.0, 1.0, 0.0, 1764.0),  # weekly in hours
        (168.0, 8.0, 0.0, 1764.0),
        (1440.0, 60.0, 0.0, 15840.0),  # daily in minutes
        (168.0, 1.0, 140.0, 1764.0),  # first pulse after t = 127
        # Pulses narrower than the widened window's grid pitch (period/500): both
        # edges share a coarse cell, and only per-component bisection finds the
        # second. The first row is one the 128 window already resolved.
        (168.0, 0.2, 10.0, 1764.0),
        (168.0, 0.1, 0.0, 1764.0),
        (1440.0, 0.5, 0.0, 15840.0),  # a 30 s pulse daily, in minutes
        (19.272, 0.0066, 12.721, 202.356),  # and a fast one
    ],
)
def test_every_pulse_is_integrated(period, width, t0, horizon):
    m = _model(period, width, t0)
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, horizon), n_points=11)
    pulses = sum(1 for n in range(10_000) if t0 + n * period < horizon)
    x_end = (
        r.observables["X"][-1]
        if "X" in r.observable_names
        else r.species[-1, list(r.species_names).index("X")]
    )
    assert x_end == pytest.approx(pulses * width, rel=1e-6)
