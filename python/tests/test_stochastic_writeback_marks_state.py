"""A stochastic run marks the state it leaves as carried over, as an ODE run does
(issue #1749).

An SSA or PSA run writes its final state back to the model, but only the CVODE
write-backs set GH #210's carried-over mark. So an ODE forward-sensitivity run
from a state an SSA run produced seeded dx(0)/dθ = 0 with no error, where the
same run from an ODE-advanced state is refused; a ``sensitivity_ic`` run seeded
e_k as if the state were still the initial condition (issue #1528's axis); and
a seed an earlier sensitivity run left was still there for
``carry_sensitivities=True`` to pick up, though the SSA run had moved the state
on. The SSA/PSA write-back now sets both marks and drops the seed.

A state taken as fixed data, with dx(0)/dθ = 0, is still a call away, and the
refusal says so: ``set_state()`` the sample after a ``reset()`` to a baseline no
run advanced, or on a fresh ``Model`` once ``save_concentrations()`` has made an
advanced state the baseline, since ``reset()`` then returns to that carried-over
state (issue #1735). Oracle for that:
0 -> A (kp), A -> 0 (kd) from a fixed A(0) = a has
A(t) = a e^(-kd t) + (kp/kd)(1 - e^(-kd t)).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 kp 200.0
    2 kd 0.5
end parameters
begin species
    1 A() 0
end species
begin reactions
    1 0 1 kp
    2 1 0 kd
end reactions
"""
KP, KD = 200.0, 0.5
# Tight solver tolerances, so the closed forms below can be compared at rel 1e-7
# (rel 1e-8 for the state): three orders above rtol leaves room for the error the
# integrator accumulates over t = 1, and a wrong seed or rate is off by far more.
SENS = {"rtol": 1e-10, "atol": 1e-12}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "m.net"
    path.write_text(NET)
    return bngsim.Model.from_net(str(path))


def _stochastic(model, method: str) -> None:
    kw = {"poplevel": 100} if method == "psa" else {}
    bngsim.Simulator(model, method=method, **kw).run(t_span=(0, 50), n_points=2, seed=1)


def _sens(model, **kw):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kp", "kd"])
    return sim.run(sample_times=[0.0, 1.0], **SENS, **kw)


# --- the issue -------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_sensitivity_run_after_a_stochastic_run_is_refused(model, method):
    _stochastic(model, method)
    assert model._core.ic_state_dirty
    assert model.get_concentration("A()") > 100  # the run moved A toward 400
    # Main seeded dA(0)/d(kp, kd) = [0, 0] here, with no error.
    with pytest.raises(bngsim.SimulationError, match="carried-over species state"):
        _sens(model)


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_an_ic_sensitivity_run_after_a_stochastic_run_is_refused(model, method):
    _stochastic(model, method)
    assert model._core.advanced_since_baseline
    sim = bngsim.Simulator(model, method="ode", sensitivity_ic=["A()"])
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        sim.run(sample_times=[0.0, 1.0], **SENS)


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_seed_left_before_a_stochastic_run_is_not_carried_past_it(model, method):
    # An ODE sensitivity run leaves its dx/dθ for carry_sensitivities. A stochastic
    # run then moves the state: that seed no longer describes it.
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kp", "kd"])
    sim.run(t_span=(0, 5), n_points=2, **SENS)
    assert np.asarray(model._core.pending_sensitivity_seed()).size == 2
    _stochastic(model, method)
    assert np.asarray(model._core.pending_sensitivity_seed()).size == 0
    with pytest.raises(bngsim.SimulationError, match="no matching forward-sensitivity seed"):
        sim.run(sample_times=[0.0, 1.0], carry_sensitivities=True, **SENS)


def test_the_refusal_names_the_fixed_data_route(model):
    _stochastic(model, "ssa")
    with pytest.raises(bngsim.SimulationError) as raised:
        _sens(model)
    message = str(raised.value)
    assert "set_state() it after a reset() to a baseline no run advanced" in message
    assert "or on a fresh Model" in message
    assert "an SSA or PSA run leaves none" in message


# --- the remedies the refusal names, run -----------------------------------------


def test_reset_then_set_state_takes_the_sample_as_fixed_data(model):
    _stochastic(model, "ssa")
    state = np.asarray(model.get_state(), dtype=float).copy()
    a = float(state[0])
    model.reset()
    model.set_state(state)
    result = _sens(model)
    t = 1.0
    e = math.exp(-KD * t)
    want_a = a * e + KP / KD * (1 - e)
    want_dkp = (1 - e) / KD
    want_dkd = -a * t * e - KP / KD**2 * (1 - e) + KP / KD * t * e
    sens = np.asarray(result.sensitivities)
    assert float(result.species[-1, 0]) == pytest.approx(want_a, rel=1e-8)
    assert sens[0, 0, :].tolist() == [0.0, 0.0]
    assert sens[-1, 0, 0] == pytest.approx(want_dkp, rel=1e-7)
    assert sens[-1, 0, 1] == pytest.approx(want_dkd, rel=1e-7)


def test_after_an_advanced_save_the_fixed_data_route_is_a_fresh_model(model, tmp_path):
    # A saved baseline a run advanced (an SSA run here) keeps its mark through
    # reset() (issue #1528), and set_state() never clears it, so on this model
    # reset() then set_state() is refused again; a fresh Model takes the numbers.
    _stochastic(model, "ssa")
    model.save_concentrations()
    _stochastic(model, "ssa")
    state = np.asarray(model.get_state(), dtype=float).copy()
    model.reset()
    model.set_state(state)
    assert model._core.ic_state_dirty
    with pytest.raises(bngsim.SimulationError, match="or on a fresh Model"):
        _sens(model)
    path = tmp_path / "m.net"
    path.write_text(NET)
    fresh = bngsim.Model.from_net(str(path))
    fresh.set_state(state)
    sens = np.asarray(_sens(fresh).sensitivities)
    e = math.exp(-KD)
    assert sens[0, 0, :].tolist() == [0.0, 0.0]
    assert sens[-1, 0, 0] == pytest.approx((1 - e) / KD, rel=1e-7)


def test_reset_alone_is_a_fresh_start(model):
    _stochastic(model, "ssa")
    model.reset()
    assert not model._core.ic_state_dirty
    sens = np.asarray(_sens(model).sensitivities)
    assert sens[0, 0, :].tolist() == [0.0, 0.0]  # from A(0) = 0, as declared


def test_a_save_makes_the_sample_the_initial_condition_for_ic_sensitivities(model):
    _stochastic(model, "ssa")
    model.save_concentrations()
    sim = bngsim.Simulator(model, method="ode", sensitivity_ic=["A()"])
    result = sim.run(sample_times=[0.0, 1.0], **SENS)
    # dA(t)/dA(0) = e^(-kd t) from any start.
    assert np.asarray(result.sensitivities_ic)[-1, 0, 0] == pytest.approx(math.exp(-KD), rel=1e-7)


# --- what does not change ----------------------------------------------------------


def test_a_fresh_model_still_runs_its_sensitivities(model):
    sens = np.asarray(_sens(model).sensitivities)
    assert sens[-1, 0, 0] == pytest.approx((1 - math.exp(-KD)) / KD, rel=1e-7)


def test_after_an_ode_run_it_is_refused_as_before(model):
    bngsim.Simulator(model, method="ode").run(t_span=(0, 50), n_points=2)
    with pytest.raises(bngsim.SimulationError, match="carried-over species state"):
        _sens(model)
