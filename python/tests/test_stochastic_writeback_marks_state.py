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
    assert model._core.has_pending_sensitivity_seed
    assert list(model._core.pending_sensitivity_seed_param_names) == ["kp", "kd"]
    _stochastic(model, method)
    assert not model._core.has_pending_sensitivity_seed
    assert np.asarray(model._core.pending_sensitivity_seed()).size == 0
    with pytest.raises(
        bngsim.SimulationError, match="no matching forward-sensitivity seed"
    ) as raised:
        sim.run(sample_times=[0.0, 1.0], carry_sensitivities=True, **SENS)
    # The run between the phases is named, not only the phase the caller did run.
    assert "an SSA or PSA run included, drops the seed" in str(raised.value)


def test_the_refusal_names_the_fixed_data_route(model):
    _stochastic(model, "ssa")
    with pytest.raises(bngsim.SimulationError) as raised:
        _sens(model)
    message = str(raised.value)
    assert "set_state() it after a reset() to a baseline no run advanced" in message
    assert "or on a fresh Model" in message
    assert "an SSA or PSA run leaves none" in message
    assert "the run's last result row" in message


# --- the remedies the refusal names, run -----------------------------------------


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_reset_then_set_state_takes_the_sample_as_fixed_data(model, method):
    _stochastic(model, method)
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


def test_the_fixed_data_route_in_a_compartment_whose_size_moves():
    """There an SSA run's state is counts over the load-time size (issue #1463),
    which set_state() after a reset() reads as concentrations: the result's last
    row is the sample to set. c grows from 2 to 3 with X's 100 molecules kept
    (k = 0: nothing fires), so [X] = 100/3 and the amount 100, not 150."""
    pytest.importorskip("antimony")
    text = "k = 0; compartment c = 2;\nspecies X in c; X = 50;\nc' = 0.5;\nJ1: X -> ; k*X;\n"
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ssa").run(t_span=(0, 2), n_points=2, seed=1)
    row = np.asarray(run.species, dtype=float)[-1].copy()
    assert row.tolist() == pytest.approx([100.0 / 3.0, 3.0], rel=1e-12)
    held = np.asarray(model.get_state(), dtype=float).copy()
    assert held.tolist() == pytest.approx([50.0, 3.0], rel=1e-12)  # counts over size 2
    for state, amount in ((row, 100.0), (held, 150.0)):
        model.reset()
        model.set_state(state)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
        out = np.asarray(sim.run(sample_times=[0.0, 1.0], **SENS).as_roadrunner(["X"]))
        assert out[0, 0] == pytest.approx(amount, rel=1e-9)


# --- what the mark also decides, as after an ODE run --------------------------------

DOSED = """begin parameters
    1 L0 100.0
    2 kf 1.0
    3 kr 2.0
end parameters
begin species
    1 A() L0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
"""


@pytest.fixture
def dosed(tmp_path):
    path = tmp_path / "dosed.net"
    path.write_text(DOSED)
    return bngsim.Model.from_net(str(path))


def _advance(model, method: str) -> None:
    kw = {"seed": 1} if method == "ssa" else {}
    bngsim.Simulator(model, method=method).run(t_span=(0, 1), n_points=2, **kw)


@pytest.mark.parametrize("method", ["ssa", "ode"])
def test_an_ic_parameter_does_not_redose_a_state_a_run_left(dosed, method):
    # The #79 rule: set_param of an IC parameter moves a species only while it
    # sits on its declared IC and no run has advanced it. A set_state() back to
    # the IC values does not undo the run's mark; reset() does, and
    # set_concentration() doses in any state. On main an SSA run left no mark,
    # so this set_param moved A to 300 after an SSA run but not after an ODE run.
    x0 = np.asarray(dosed.get_state(), dtype=float).copy()
    _advance(dosed, method)
    dosed.set_state(x0)
    dosed.set_param("L0", 300.0)
    assert dosed.get_concentration("A()") == 100.0
    dosed.set_concentration("A()", 300.0)
    assert dosed.get_concentration("A()") == 300.0
    dosed.reset()
    dosed.set_param("L0", 900.0)
    assert dosed.get_concentration("A()") == 900.0


@pytest.mark.parametrize("method", ["ssa", "ode"])
def test_a_steady_state_in_an_ic_parameter_after_a_run_is_refused(dosed, method):
    # L0 sets the total of the law A + B, which a state a run advanced no longer
    # says (issue #704): refused after either run, and solved after a reset().
    _advance(dosed, method)
    sim = bngsim.Simulator(dosed, method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="carried-over species state"):
        sim.steady_state(sensitivity_params=["L0"])
    dosed.reset()
    bngsim.Simulator(dosed, method="ode").steady_state(sensitivity_params=["L0"])


def test_the_wrapped_stochastic_paths_leave_the_callers_model_unmarked(dosed):
    # run_replicates and run_batch run on copies; parameter_scan puts back the
    # invocation's state and marks at each point, so every point is dosed.
    sim = bngsim.Simulator(dosed, method="ssa")
    sim.run_replicates(3, t_span=(0, 1), n_points=2, seed=0)
    sim.run_batch(params=[{"kf": 1.0}, {"kf": 2.0}], t_span=(0, 1), n_points=2, seed=0)
    assert not dosed._core.ic_state_dirty
    assert not dosed._core.advanced_since_baseline
    scan = sim.parameter_scan("L0", [100.0, 300.0, 900.0], t_span=(0, 1), n_points=2, seed=1)
    assert [float(np.asarray(r.species)[0, 0]) for r in scan] == [100.0, 300.0, 900.0]


# --- what does not change ----------------------------------------------------------


def test_a_fresh_model_still_runs_its_sensitivities(model):
    sens = np.asarray(_sens(model).sensitivities)
    assert sens[-1, 0, 0] == pytest.approx((1 - math.exp(-KD)) / KD, rel=1e-7)


def test_after_an_ode_run_it_is_refused_as_before(model):
    bngsim.Simulator(model, method="ode").run(t_span=(0, 50), n_points=2)
    with pytest.raises(bngsim.SimulationError, match="carried-over species state"):
        _sens(model)
