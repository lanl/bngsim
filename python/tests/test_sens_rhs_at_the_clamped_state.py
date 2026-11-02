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
the GH #384/#386 refusal (the control below). So does a state where a slot the
value RHS read at 0 is below 0 in earnest, not by the run's own error: put there
by an assignment and not back at or above 0 since, or driven further below 0 by
the flow there. J at the clamped state keeps its slope at 0 in that slot, which
the flow does not have, and the columns read there were silently wrong.
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


def test_a_restart_that_leaves_l_where_it_is_keeps_the_reading():
    """An event at t = 60, with L a hair below 0 since t = 25, sets a species no
    law reads. The restart leaves L where the run's own error put it, so L is
    no slot an assignment put below 0, and the simultaneous run on the
    closed-form Jacobian still runs to the plain run's state. Marking every
    slot below 0 at a restart refused it from t = 60 on."""
    src = MPEKRIS.replace("\nend\n", "\n  species Z = 0;\n  E1: at (time >= 60): Z = 1;\nend\n")
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(src),
        method="ode",
        sensitivity_params=list(PARAMS),
        sensitivity_method="simultaneous",
        jacobian="analytical",
    )
    run = sim.run(t_span=SPAN, n_points=N_POINTS, rtol=RTOL, atol=ATOL)
    plain = bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode").run(
        t_span=SPAN, n_points=N_POINTS, rtol=RTOL, atol=ATOL
    )
    sens, ref = np.asarray(run.species), np.asarray(plain.species)
    assert np.abs(sens - ref).max() <= 1e-6 * np.abs(ref).max()


def test_with_the_reading_off_the_run_fails_as_it_did(monkeypatch):
    """BNGSIM_SENS_AT_VALUE_STATE=0 turns the reading at the value RHS's state
    off (the #1301 tests set it), and the simultaneous run fails at t = 22
    again."""
    monkeypatch.setenv("BNGSIM_SENS_AT_VALUE_STATE", "0")
    with pytest.raises(SimulationError, match="CV_REPTD_SRHSFUNC_ERR"):
        _sens("simultaneous", jacobian="analytical")


# Where a slot the clamp reads at 0 is below 0 in earnest, J at the clamped
# state keeps its slope at 0 and carries the slot's column into the rest, which
# the flow the run integrates does not; main's refusal stands there. On an
# accepted path such a slot got below 0 one of two ways, and the guard
# (sens_may_read_at) refuses both: an assignment put it there (the run's start,
# an event) and it has not been back at or above 0 at an accepted step since,
# or the flow there drives it further below 0. Each case below was answered
# wrong, with no error, by the reading without the guard, where main refuses;
# the first two are the flow's, the rest assignments'.
#
# S drained at a constant rate cs beside S^1.5 and a mass-action S -> P at k4*S:
# past t_c = ln(1 + k4/cs)/k4 = 49.7 the run reads S at 0, so P stops and S
# falls at cs. Read at the clamped state, the columns came back with
# dS/dcs = -3.33 and dP/dcs = -51.7 at t = 55, for -8.62 and -46.4.
SLOW = """
model slow
  species S = 1, A = 1, P = 0;
  cs = 1e-7; k = 0.1; k4 = 0.3;
  J1: S -> ; cs;
  J2: A -> ; k*A*S^1.5;
  J4: S -> P; k4*S;
end
"""

# The same with k4 = 1e6: S crosses 0 at once and falls at cs, and the stiff k4
# holds S's own entry of the column near 0. dS/dcs = -1e-6 and dP/dcs = -3 at
# t = 3 came back, for -3.0 and 0.0 (S falls at cs, P stops).
FAST = SLOW.replace("k4 = 0.3;", "k4 = 1e6;")

# An event leaves S at -e0 with nothing moving it: from t = 1, dS/de0 is -1 and
# dP/de0 is 0. At e0 = 0.5 the columns came back -0.07 and -0.93, and so they
# did at e0 = 0.005 at default tolerances and at e0 = 1e-5 at the corpus job's
# tolerance, through guards that measured S's depth, or its own entry of the
# column, against the run's tolerances.
EVENT = """
model left_below
  species S = 1, A = 1, P = 0;
  k = 0.1; k4 = 0.3; e0 = E0;
  J2: A -> ; k*A*S^1.5;
  J4: S -> P; k4*S;
  E1: at (time >= 1): S = -e0;
end
"""

# An event leaves S at -0.5 and a source q lifts it back beside a stiff S -> P:
# dS/dq = 1e-6 and dP/dq = 10 at t = 10 came back, for 9 and 1, and so they did
# at rtol 1e-2, where 100 of the run's error units span S's whole range.
RETURN = """
model return
  species S = 1, A = 1, P = 0;
  k = 0.1; k4 = 1e6; q = 0.01;
  J0: -> S; q;
  J2: A -> ; k*A*S^1.5;
  J4: S -> P; k4*S;
  E1: at (time >= 1): S = -0.5;
end
"""

# An event leaves S at -1e-6 and a weak source lifts it back beside a
# mass-action S -> P: dS/dq1 = 3.12 and dP/dq1 = 6.01 at t = 10 came back, for
# 9.0 and 0.136, through a guard that also held S's own entry of the column
# within 1e3 of its tolerance (the weak source makes the tolerance large).
WEAK = """
model weak
  species S = 1, A = 1, P = 0;
  k = 0.1; k4 = 0.3; q1 = 1e-7;
  J0: -> S; q1;
  J2: A -> ; k*A*S^1.5;
  J4: S -> P; k4*S;
  E1: at (time >= 1): S = -1e-6;
end
"""

# An event leaves S 1e-7 below 0 and P, a rate-rule target, is declared -1e-30:
# from t = 1 the clamp reads S at 0, its reading is still NaN for P^1.5, and
# the reading near 0 after it (#1340) reads P at 0 too. Neither column depends
# on kb, so B's would come back right, -t*B; but nothing at the clamped state
# says so, and with a slope in S or P at 0 it would not.
NEAR = """
model near
  species S = 1, A = 1, B = 1;
  P = -1e-30; P' = 0;
  k = 0.1; kb = 0.2;
  J2: A -> ; k*A*S^1.5;
  J3: B -> ; kb*B*P^1.5 + kb*B;
  E1: at (time >= 1): S = -1e-7;
end
"""


@pytest.mark.parametrize("method", ["staggered", "simultaneous"])
@pytest.mark.parametrize(
    "src, param, run",
    [
        (SLOW, "cs", dict(t_span=(0.0, 55.0), rtol=1e-8, atol=1e-12)),
        (FAST, "cs", dict(t_span=(0.0, 3.0), rtol=1e-8, atol=1e-12)),
        (EVENT.replace("E0", "0.5"), "e0", dict(t_span=(0.0, 10.0), rtol=1e-8, atol=1e-9)),
        (EVENT.replace("E0", "0.005"), "e0", dict(t_span=(0.0, 10.0))),
        (EVENT.replace("E0", "1e-5"), "e0", dict(t_span=(0.0, 10.0), rtol=1e-6, atol=1e-9)),
        (RETURN, "q", dict(t_span=(0.0, 10.0), rtol=1e-8, atol=1e-9)),
        (RETURN, "q", dict(t_span=(0.0, 10.0), rtol=1e-2, atol=1e-6)),
        (WEAK, "q1", dict(t_span=(0.0, 10.0), rtol=1e-8, atol=1e-9)),
        (NEAR, "kb", dict(t_span=(0.0, 10.0), rtol=1e-8, atol=1e-9)),
    ],
    ids=[
        "driven",
        "driven-stiff",
        "far",
        "far-default-tolerances",
        "shallow",
        "returning",
        "returning-loose-rtol",
        "weak-source",
        "near-zero-reading",
    ],
)
def test_a_slot_below_zero_in_earnest_is_refused(method, src, param, run):
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(src),
        method="ode",
        sensitivity_params=[param],
        sensitivity_method=method,
    )
    with pytest.raises(SimulationError, match="sensitivity RHS returned a non-finite"):
        sim.run(n_points=12, **run)


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
