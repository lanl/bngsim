"""Issue #717 — an SBML event that fires at t_start keeps forward sensitivities.

With ``initialValue=false`` and a trigger already true at t_start, SBML L3 §3.4.5
fires the event at the first instant. The sensitivity jump needs s⁻ there, and it
was read with CVodeGetSens, which interpolates at the last return time. Before
the first step CVODES has none: at t_start = 0 it computed (t − tn)/h = 0/0, so
every s⁻ was NaN and the run failed with a convergence error blaming the model;
at any other t_start it returned CV_BAD_T. s⁻ at run start is the seed itself.

Closed forms: A is reset to 8 at t_start and then decays at k1 = 0.5 into B, so
dA/dk1 = −8·τ·e^(−k1·τ) and dB/dk1 = +8·τ·e^(−k1·τ), with τ = t − t_start.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")


def _model(t_fire: float):
    return bngsim.Model.from_antimony_string(
        "species A, B; A = 10; B = 0; k1 = 0.5\nR1: A -> B; k1*A\n"
        f"E1: at (time >= {t_fire}), t0=false: A = 8\n"
    )


@pytest.mark.parametrize("t_start", [0.0, 1.0], ids=["t0", "t1"])
def test_a_run_start_event_keeps_finite_closed_form_sensitivities(t_start):
    t = np.array([t_start, t_start + 1.0, t_start + 2.0])
    run = bngsim.Simulator(_model(t_start), method="ode", sensitivity_params=["k1"]).run(
        sample_times=list(t), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    s = np.asarray(run.sensitivities)[:, :, 0]
    tau = t - t_start
    want = 8.0 * tau * np.exp(-0.5 * tau)
    np.testing.assert_allclose(s[:, names.index("A")], -want, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(s[:, names.index("B")], want, rtol=1e-6, atol=1e-10)


def test_an_initial_condition_column_is_reset_with_the_state():
    """The event overwrites A with a constant, so A no longer depends on its
    initial value: the IC column is exactly 0 after the fire, and finite."""
    run = bngsim.Simulator(_model(0.0), method="ode", sensitivity_ic=["A"]).run(
        sample_times=[0.0, 1.0, 2.0], rtol=1e-10, atol=1e-12
    )
    s = np.asarray(run.sensitivities_ic)
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s, 0.0, atol=1e-12)
