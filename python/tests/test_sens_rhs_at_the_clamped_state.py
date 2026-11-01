"""Issue #1298: a sensitivity run reads its columns at the state its value RHS
read f at.

Where the value RHS is non-finite at a state with a concentration below zero
(``pow(x, 1.36)`` of a species the predictor took negative), it retries at the
clamped state, and, where that is still non-finite, at the reading near 0 after
it (GH #135, #1340); the run integrates f there. The sensitivity RHS had only GH
#392's narrower retry, which snaps a component inside ``(-atol_i, 0)``, so it
refused states the run was integrating. The value RHS now records the state it
read f at, and the sensitivity RHS reads its columns there.

The fixture is MODEL2001200002 (Mpekris 2017, BioModels, CC0) with its unused
parameters dropped. ``L`` falls to about 1e-16 by t = 25 and sits a hair below
zero from there, where ``D = d_im*(L/T)^1.36/(s + (L/T)^1.36)*T`` is NaN. When
the issue was filed, the staggered run's state was 2.6e-3 of its peak off the
plain run's and its columns 0.28 to 1.0 off; #1343 holds those now. What was
left: the simultaneous run, on the closed-form Jacobian, failed at t = 22, and
the GH #176 retry took it again on the difference quotient, at 4.9 times the
plain run's steps.

Where ``f(y)`` is finite at the negative base, the run integrates ``f(y)``, and
a column read at ``clamp(y)`` would be another state's; that NaN still goes to
the GH #384/#386 refusal (the control below).
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


def _sens(method, jacobian="auto"):
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(MPEKRIS),
        method="ode",
        sensitivity_params=list(PARAMS),
        sensitivity_method=method,
        jacobian=jacobian,
    )
    return sim.run(t_span=SPAN, n_points=N_POINTS, rtol=RTOL, atol=ATOL)


def _plain_steps():
    model = bngsim.Model.from_antimony_string(MPEKRIS)
    run = bngsim.Simulator(model, method="ode").run(
        t_span=SPAN, n_points=N_POINTS, rtol=RTOL, atol=ATOL
    )
    return run.solver_stats["n_steps"]


@pytest.mark.parametrize("method", ["staggered", "simultaneous"])
def test_the_sensitivity_run_holds_the_plain_runs_state(method):
    """A few times rtol apart, the two runs taking their own steps. Staggered
    was 3.4e-3 of the peak apart when the issue was filed; #1343 holds it now,
    and this keeps it so."""
    sens = np.asarray(_sens(method).species)
    plain = _plain(RTOL, ATOL)
    assert np.abs(sens - plain).max() <= 1e-6 * np.abs(plain).max()


@pytest.mark.parametrize("param", PARAMS)
def test_the_columns_are_that_trajectorys_derivatives(param):
    """Judged by a central difference of plain runs, which never enter the
    sensitivity RHS: a few 1e-7 off. They were 0.28 (sigma_nk) and 0.96 (g_reg)
    of the column's peak off when the issue was filed; #1343 holds them now."""
    run = _sens("staggered")
    rows = [list(run.species_names).index(s) for s in JUDGED]
    S = np.asarray(run.sensitivities)[:, rows, PARAMS.index(param)]
    p0 = bngsim.Model.from_antimony_string(MPEKRIS).get_param(param)
    fd = central_difference(
        lambda value, rtol: _plain(rtol, rtol * 1e-3, JUDGED, **{param: value}), p0, rtol=1e-11
    )
    assert mismatches(S, fd, p0, rtol=RTOL, atol=ATOL, resolution=1e-4) == []


def test_the_simultaneous_run_needs_no_second_attempt():
    """On the closed-form Jacobian, where no difference quotient stands in. The
    sensitivity RHS refused the states the run integrated, and the run failed at
    t = 22 (CV_REPTD_SRHSFUNC_ERR); under ``auto`` the GH #176 retry took it
    again on the quotient."""
    sens = np.asarray(_sens("simultaneous", jacobian="analytical").species)
    plain = _plain(RTOL, ATOL)
    assert np.abs(sens - plain).max() <= 1e-6 * np.abs(plain).max()


def test_the_simultaneous_run_takes_about_the_plain_runs_steps():
    """It took 4.9 times the plain run's steps, on the GH #176 retry's
    difference quotient; 1.06 times now. The sensitivity columns' own error
    test adds steps of its own, so the bound is 1.5, well short of 4.9."""
    steps = _sens("simultaneous").solver_stats["n_steps"]
    assert steps <= 1.5 * _plain_steps()


def test_with_the_reading_off_the_run_fails_as_it_did(monkeypatch):
    """BNGSIM_SENS_AT_VALUE_STATE=0 turns the reading at the value RHS's state
    off (the #1301 guard sets it), and the simultaneous run fails at t = 22
    again."""
    monkeypatch.setenv("BNGSIM_SENS_AT_VALUE_STATE", "0")
    with pytest.raises(SimulationError, match="CV_REPTD_SRHSFUNC_ERR"):
        _sens("simultaneous", jacobian="analytical")


# S is masked, drained far below 0 by a zero-order sink and read by S^1.5; P is a
# rate-rule target sinking from 0 at 1e-30, inside its atol, read by P^1.5 in B's
# law. Where S is below 0, the value RHS's clamp reads S at 0, its reading is
# still NaN for P^1.5, and the reading near 0 after it (#1340) reads P at 0 too:
# the run integrates B' = -kb*B, so B = exp(-kb*t) and dB/dkb = -t*B exactly.
# #392's retry reads P at 0 but S where it is, and a reading at the clamped state
# alone reads S at 0 but P where it is: both were NaN, and the run was refused.
BAND = """
model band
  species S = 1, A = 1, B = 1;
  P = 0; P' = -c;
  cs = 1; k = 0.1; kb = 0.2; c = 1e-30;
  J1: S -> ; cs;
  J2: A -> ; k*A*S^1.5;
  J3: B -> ; kb*B*P^1.5 + kb*B;
end
"""


@pytest.mark.parametrize("method", ["staggered", "simultaneous"])
def test_the_columns_are_read_where_the_reading_near_zero_reads_too(method):
    """The column is -t*B to within 1e-5 of itself (3e-7 measured, at rtol
    1e-8, where CVODES does not hold the columns to the run's own tolerance)."""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(BAND),
        method="ode",
        sensitivity_params=["kb"],
        sensitivity_method=method,
    )
    run = sim.run(t_span=(0.0, 10.0), n_points=11, rtol=1e-8, atol=1e-9)
    names = list(run.species_names)
    b = np.asarray(run.species)[:, names.index("B")]
    t = np.asarray(run.time)
    np.testing.assert_allclose(b, np.exp(-0.2 * t), rtol=1e-6)
    s = np.asarray(run.sensitivities)[:, names.index("B"), 0]
    np.testing.assert_allclose(s, -t * b, rtol=1e-5, atol=1e-12)


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
