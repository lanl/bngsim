"""Issue #1298: a sensitivity run integrates the trajectory its plain run does,
where the value RHS's clamp is what that trajectory is.

Where the value RHS is non-finite at a state with a concentration below zero
(``pow(x, 1.36)`` of a species the predictor took negative), ``cvode_rhs``
retries at the clamped state, and the run integrates ``f(clamp(y))`` (GH #135).
The sensitivity RHS had only GH #392's narrower retry, which snaps a component
inside ``(-atol_i, 0)``. That retry lets a run step a hair past zero; once
there, the value RHS's clamp holds the species where it is rather than
restoring it, it drifts past ``-atol_i``, and the sensitivity RHS refused every
state from then on. CVODES cut ``h`` until it gave up, and the GH #176 retry
returned another run's trajectory, so every column was wrong and the state
moved off the plain run's.

The fixture is MODEL2001200002 (Mpekris 2017, BioModels, CC0) with its unused
parameters dropped. ``L`` falls to about 1e-16 by t = 25 and sits a hair below
zero from there, where ``D = d_im*(L/T)^1.36/(s + (L/T)^1.36)*T`` is NaN. Before
this the sensitivity run's state was 2.6e-3 of its peak off the plain run's,
and its columns 0.28 to 1.0 off a central difference of plain runs; with
``BNGSIM_SENS_CLAMP_NUMERIC_ZERO=0`` (no #392 retry) it was right.

The sensitivity RHS now retries at the value RHS's clamped state too, but only
where ``f(y)`` itself is non-finite, so that the columns are those of the flow
the run integrates. Where ``f(y)`` is finite at the negative base, the run
integrates ``f(y)``, and a column read at ``clamp(y)`` would be another
state's; that NaN still goes to the GH #384/#386 refusal (the control below).
"""

from __future__ import annotations

import os
import sys

import bngsim
import numpy as np
import pytest
from bngsim import SimulationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _fd_sens import central_difference, mismatches  # noqa: E402

MPEKRIS = """
model mpekris2017
  D := d_im*((L/T)^lamda_im/(s + (L/T)^lamda_im))*T;
  J1:  => C; G*S_ftr*C + p_CT*Csc + p_IT*I;
  J2: T => ; c*N*T + D + (p_TC + p_TI)*T;
  J3:  => Csc; a_CSC*GS_CSC*Csc + p_TC*T + p_IC*I;
  J4: Csc => ; c_CSC*N*Csc + D + (p_CT + p_CI)*Csc;
  J5:  => I; a_I*GS_I*I + p_TI*T + p_CI*Csc;
  J6: I => ; c*N*I + D + (p_IT + p_IC)*I;
  J7:  => N; sigma_nk + (g_NK*T*T/(h + T*T))*N;
  J8: N => ; f_NK*N + p_im*N*T + lamda_reg*Treg*N;
  J9:  => L; (j_T8*D*D/(k_im + D*D))*L + r*N*T;
  J10: L => ; m_T8*L + q*L*T + lamda_reg*Treg*L;
  J11:  => Treg; g_reg*Treg;
  J12: Treg => ; m_reg*Treg;
  C = 1; T = 1; Csc = 1; I = 1; N = 1; L = 1; Treg = 1;
  G = 0.1; S_ftr = 0.1; GS_CSC = 0.1; c_CSC = 0.1; GS_I = 0.1; a_CSC = 1; a_I = 1;
  c = 3.23e-07; d_im = 1.43; lamda_im = 1.36; s = 2.73; sigma_nk = 13000;
  f_NK = 0.0412; m_T8 = 0.02; m_reg = 0.02; g_NK = 0.025; g_reg = 0.0375;
  h = 20200000; p_im = 1e-07; k_im = 20200000; q = 3.42e-10; r = 1.1e-07;
  lamda_reg = 100; p_TC = 0.55; p_CT = 1; p_CI = 0.58; p_IC = 0.96; p_TI = 0.21;
  p_IT = 1; j_T8 = 0.0375;
end
"""

PARAMS = ("sigma_nk", "g_reg")
RTOL, ATOL = 1e-9, 1e-12
SPAN, N_POINTS = (0.0, 100.0), 101
# The columns judged: the species that stay above atol. T, Csc and I fall to
# 1e-34 and L sits below zero, where a plain run's own error is larger than
# the value and the difference reads that noise as kinks.
JUDGED = ("C", "N", "Treg")


def _plain(rtol, atol, species=None, **values):
    model = bngsim.Model.from_antimony_string(MPEKRIS)
    for name, value in values.items():
        model.set_param(name, value)
    run = bngsim.Simulator(model, method="ode").run(
        t_span=SPAN, n_points=N_POINTS, rtol=rtol, atol=atol
    )
    x = np.asarray(run.species)
    if species is None:
        return x
    return x[:, [list(run.species_names).index(s) for s in species]]


def _sens(method):
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(MPEKRIS),
        method="ode",
        sensitivity_params=list(PARAMS),
        sensitivity_method=method,
    )
    return sim.run(t_span=SPAN, n_points=N_POINTS, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("method", ["staggered", "simultaneous"])
def test_the_sensitivity_run_holds_the_plain_runs_state(method):
    """Staggered: 3.4e-3 of the peak apart before, a few times rtol now, the two
    runs taking their own steps. Simultaneous was right here already, and is
    here so that it stays so."""
    sens = np.asarray(_sens(method).species)
    plain = _plain(RTOL, ATOL)
    assert np.abs(sens - plain).max() <= 1e-6 * np.abs(plain).max()


@pytest.mark.parametrize("param", PARAMS)
def test_the_columns_are_that_trajectorys_derivatives(param):
    """Judged by a central difference of plain runs, which never enter the
    sensitivity RHS: 0.28 (sigma_nk) and 0.96 (g_reg) of the column's peak off
    before, a few 1e-7 now."""
    run = _sens("staggered")
    rows = [list(run.species_names).index(s) for s in JUDGED]
    S = np.asarray(run.sensitivities)[:, rows, PARAMS.index(param)]
    p0 = bngsim.Model.from_antimony_string(MPEKRIS).get_param(param)
    fd = central_difference(
        lambda value, rtol: _plain(rtol, rtol * 1e-3, JUDGED, **{param: value}), p0, rtol=1e-11
    )
    assert mismatches(S, fd, p0, rtol=RTOL, atol=ATOL, resolution=1e-4) == []


# A drained to -1 at a constant rate. Its rate law in B, vmax*A^4/(Kd + A^4),
# is finite there, so the run integrates f(y) at the negative A and the value
# RHS's clamp never engages; the exponent column holds A^n*ln(A), which is NaN.
DRAIN = """
model drain
  J1: A -> ; v
  J2: -> B; vmax*A^n/(Kd + A^n)
  A = 1; B = 0; v = 0.5; vmax = 1; n = 4; Kd = 1
end
"""


def _drain(param):
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(DRAIN), method="ode", sensitivity_params=[param]
    )
    return sim.run(t_span=(0.0, 4.0), n_points=41, rtol=1e-8, atol=1e-12)


def test_a_column_the_flow_has_no_derivative_for_is_still_refused():
    """The control: answered at the clamped state, the exponent column would be
    a finite number for a state the run never integrated."""
    with pytest.raises(SimulationError, match="sensitivity RHS returned a non-finite"):
        _drain("n")


def test_a_column_that_is_finite_there_still_runs():
    """And the rate's own column, finite at the negative A, is untouched:
    dA/dv = -t, so -4 at t = 4."""
    S = np.asarray(_drain("v").sensitivities)
    assert S[-1, 0, 0] == pytest.approx(-4.0, rel=1e-6)
