"""A sensitivity run whose RHS refuses the corrected state of a step (issue #1301).

In its staggered sensitivity corrector, CVODES evaluates f at the state the
step converged to. Upstream SUNDIALS (7.2.1 through the current main) takes a
recoverable failure there back round the step loop with neither cvRestore nor a
smaller step, so the next attempt predicts again from the advanced time: the
step covers 2h, and its first h is crossed by the uncorrected predictor, which
neither the error test nor the root search sees. On MODEL2001200002 (Mpekris
2017) that happened 2,333 times in one run. The state came back 9e-3 off and
the sensitivities 1.4e-2 off, with no error. Where an output time fell in an
uncorrected stretch, CVode returned it without reading the state there, and
the row took the previous row's state and zero sensitivities, as the issue
reported.

BNGsim carries the fix in its fetched SUNDIALS
(third_party/sundials/carry_cvodes_staggered_restore.cmake). The run checks
each CV_NORMAL return as well, and refuses an output time CVODE's last step
does not span. BNGSIM_SUNDIALS_STAGGERED_RESTORE=0 turns the carry off, for an
A/B only.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# MODEL2001200002 (Mpekris 2017), 7 species, as #1300's
# test_sens_rhs_at_the_clamped_state.py writes it.
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

_PARAMS = {"sigma_nk": ("13000", 13000.0), "g_reg": ("0.0375", 0.0375)}
_GRID = np.linspace(0, 100, 101)
_RUN = {"t_span": (0, 100), "n_points": len(_GRID), "rtol": 1e-9, "atol": 1e-12}
# The independent reference: plain runs, which never enter the staggered
# corrector, at tolerances well below the run's.
_TIGHT = {"t_span": (0, 100), "n_points": len(_GRID), "rtol": 1e-12, "atol": 1e-14}


def _model(name=None, value=None):
    src = MPEKRIS
    if name is not None:
        text, _ = _PARAMS[name]
        assert f"{name} = {text};" in src
        src = src.replace(f"{name} = {text};", f"{name} = {value!r};")
    return bngsim.Model.from_antimony_string(src)


def _sens_run(**overrides):
    sim = bngsim.Simulator(_model(), method="ode", sensitivity_params=list(_PARAMS))
    return sim.run(**{**_RUN, **overrides})


def _plain_species(name=None, value=None):
    run = bngsim.Simulator(_model(name, value), method="ode").run(**_TIGHT)
    return np.asarray(run.species)


@pytest.fixture(scope="module")
def sens_run():
    return _sens_run()


def test_the_model_still_reaches_the_upstream_defect(monkeypatch):
    """The guard: with the carry turned off, this run reaches the defect and is
    refused where an output time falls in a stretch CVODE never corrected.
    Were it not refused, the tests below could pass on a model that no longer
    reaches the defect, so this one fails instead and asks for another model.
    Its message is also the refusal a system SUNDIALS, which lacks the carry,
    gives."""
    monkeypatch.setenv("BNGSIM_SUNDIALS_STAGGERED_RESTORE", "0")
    with pytest.raises(bngsim.SimulationError, match="does not span it"):
        _sens_run()


def test_rows_are_at_the_times_asked_for(sens_run):
    np.testing.assert_array_equal(np.asarray(sens_run.time), _GRID)


def test_the_state_matches_a_plain_run(sens_run):
    """With the carry off, 9e-3 from t = 41 on."""
    ref = _plain_species()
    x = np.asarray(sens_run.species)
    scale = np.abs(ref).max(axis=0)
    assert np.max(np.abs(x - ref) / scale) < 1e-6


def test_the_sensitivities_match_finite_differences(sens_run):
    """Central differences of plain runs: with the carry off, 5e-3 and 1.4e-2
    from t = 40 on, and rows of zeros where an output time fell in an
    uncorrected stretch (the issue's report)."""
    s = np.asarray(sens_run.sensitivities)
    for k, (name, (_, base)) in enumerate(_PARAMS.items()):
        step = 1e-4 * base
        fd = (_plain_species(name, base + step) - _plain_species(name, base - step)) / (2 * step)
        scale = np.abs(fd).max(axis=0)
        # A column the parameter does not reach (Treg for sigma_nk) is the
        # difference's own noise, within a factor of 10 of 1e-8 of the largest
        # and moving with the platform's libm, so the cut sits well above it.
        cols = scale > 1e-6 * scale.max()
        err = np.abs(s[:, cols, k] - fd[:, cols]) / scale[cols]
        assert err.max() < 1e-4, f"d/d{name}: {err.max():.3e}"


def test_a_fine_grid_keeps_its_times():
    """Rows read where the run stood instead of where they were asked for put
    several rows on one time wherever the grid is finer than the steps."""
    grid = np.linspace(0, 100, 2001)
    run = _sens_run(n_points=len(grid))
    np.testing.assert_array_equal(np.asarray(run.time), grid)


def test_a_start_time_named_twice_reads_the_seed():
    """A row recorded before the first step is the seed. CVODES has no step
    size there yet, so its interpolation divides 0 by 0: the per-row read
    refused the nan, and main, reading at another time, recorded dA/dA0 = 0
    at the second t = 2 with no error."""
    model = bngsim.Model.from_antimony_string(
        "model a\n J1: A => B; k*A\n J2: B => ; k2*B\n A = 1; B = 0; k = 0.5; k2 = 0.3\nend"
    )
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"], sensitivity_ic=["A"])
    run = sim.run(sample_times=[2.0, 2.0, 3.0], t_span=(2.0, 3.0))
    a = list(run.species_names).index("A")
    s_ic = np.asarray(run.sensitivities_ic)
    s_p = np.asarray(run.sensitivities)
    np.testing.assert_array_equal(s_ic[1], s_ic[0])
    np.testing.assert_array_equal(s_p[1], s_p[0])
    assert s_ic[1, a, 0] == 1.0
    np.testing.assert_allclose(s_ic[2, a, 0], np.exp(-0.5), rtol=1e-5)
