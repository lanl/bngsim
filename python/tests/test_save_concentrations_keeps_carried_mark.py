"""save_concentrations() keeps the carried-over mark of a state a run advanced
(issue #1528).

A sensitivity run on a state a run advanced, with no carried dx/dθ, raises:
fresh seeding would assume ∂x(0)/∂θ = 0, and the state is x(θ). Saving that
state with no seed cleared the mark, taking "no seed" for "literal initial
conditions", so the next sensitivity run started from dx(0)/dθ = 0 with no
error. reset() to such a baseline, a named snapshot of it restored, a clone of
it, and steady_state(sensitivity_params=...) on it did the same.

The model is 0 -> A (kp), A -> 0 (kd), A + B -> B (kc), B a catalyst that is
0 unless dosed. Equilibrated with B = 0 it sits at A_ss = kp/kd = 4, where
dA/d(kp, kd) = (1/kd, -kp/kd^2) = (2, -8) exactly. From a literal A(0) = a with
B = 0, A(t) = kp/kd + (a - kp/kd) e^(-kd t) in closed form. Those are the
oracles; trajectories run at rtol 1e-10 / atol 1e-12 and forward sensitivities
are compared at rel 1e-6, inside which they carry their own error terms and far
inside the defect (0 for 2 at t = 0).
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest

KP, KD, KC = 2.0, 0.5, 1.0
NAMES = ["kp", "kd"]
TOL = {"rtol": 1e-10, "atol": 1e-12}
TS = [0.0, 2.0]
EQ = dict(t_span=(0, 2000), n_points=2, steady_state=True, steady_state_tol=1e-13, **TOL)
CARRIED = "carried-over species state"

NET = f"""begin parameters
    1 kp {KP}
    2 kd {KD}
    3 kc {KC}
end parameters
begin species
    1 A() 0
    2 B() 0
end species
begin reactions
    1 0 1 kp
    2 1 0 kd
    3 1,2 2 kc
end reactions
"""


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "catalysed_decay.net"
    path.write_text(NET)
    return bngsim.Model.from_net(str(path))


def plain_equilibration(model):
    bngsim.Simulator(model, method="ode").run(**EQ)


def sens(model):
    return bngsim.Simulator(model, method="ode", sensitivity_params=NAMES)


def dA(result):
    """dA/d(kp, kd) at each sample time, shape (n_times, 2)."""
    return np.asarray(result.sensitivities)[:, 0, :]


def from_literal(a):
    """dA/d(kp, kd) at TS from a literal A(0) = a, with B = 0."""
    t = np.array(TS)
    e = np.exp(-KD * t)
    dkp = (1 - e) / KD
    dkd = -KP / KD**2 * (1 - e) - t * (a - KP / KD) * e
    return np.stack([dkp, dkd], axis=1)


def refuses(model_or_sim, **kw):
    sim = model_or_sim if isinstance(model_or_sim, bngsim.Simulator) else sens(model_or_sim)
    with pytest.raises(bngsim.SimulationError, match=CARRIED):
        sim.run(sample_times=TS, **TOL, **kw)


# ── The issue's paths: each raised without the save, and returned 0 with it ──


def test_a_saved_equilibrated_state_still_refuses(model):
    """Plain equilibration -> save_concentrations() -> sensitivity run. On main
    dA(0)/d(kp, kd) came back [0, 0] where it is [2, -8]."""
    plain_equilibration(model)
    model.save_concentrations()
    assert model._core.ic_state_dirty is True
    refuses(model)


def test_a_dose_saved_as_set_concentration_advises_still_refuses(model):
    """Sensitivity equilibration -> set_concentration dose (which drops the
    carried seed, as documented) -> save_concentrations(), the save that
    set_concentration's docstring advises -> sensitivity run. On main [0, 0] at
    t = 0 where it is [2, -8], and wrong at every t after."""
    sim = sens(model)
    sim.run(**EQ)
    model.set_concentration("B()", 1.0)
    model.save_concentrations()
    refuses(sim)


def test_reset_to_a_baseline_saved_from_an_advanced_state_refuses(model):
    """reset() returns to the saved baseline, and to its mark with it."""
    plain_equilibration(model)
    model.save_concentrations()
    model.reset()
    assert model.get_concentration("A()") == pytest.approx(KP / KD, rel=1e-9)
    assert model._core.ic_state_dirty is True
    refuses(model)


def test_a_named_snapshot_of_an_advanced_state_restores_its_mark(model):
    """save_concentrations("eq") -> reset() to the declared ICs (a fresh start)
    -> restore_concentrations("eq") -> sensitivity run."""
    plain_equilibration(model)
    model.save_concentrations("eq")
    model.reset()
    assert model._core.ic_state_dirty is False
    model.restore_concentrations("eq")
    refuses(model)


def test_a_clone_keeps_the_mark_and_its_baseline(model):
    """clone() of the saved state, and that clone reset() to its baseline; a
    run_batch row is a clone reset() this way."""
    plain_equilibration(model)
    model.save_concentrations()
    clone = model.clone()
    refuses(clone)
    clone.reset()
    refuses(clone)


def test_a_clone_restores_a_named_advanced_snapshot_with_its_mark(model):
    """A clone carries the named snapshots, and which of them a run advanced."""
    plain_equilibration(model)
    model.save_concentrations("eq")
    model.reset()
    clone = model.clone()
    assert clone._core.ic_state_dirty is False
    clone.restore_concentrations("eq")
    refuses(clone)


def test_restoring_a_literal_snapshot_keeps_a_mark_already_set(model):
    """Restoring never clears the mark, as set_state() does not: a literal
    snapshot restored into a model a run advanced still refuses. Clearing it
    would seed fresh a snapshot of the declared ICs restored after a set_param
    of its IC parameter, numbers that no longer follow that parameter."""
    model.set_concentration("A()", 1.0)
    model.save_concentrations("lit")
    plain_equilibration(model)
    model.restore_concentrations("lit")
    assert model.get_concentration("A()") == 1.0
    assert model._core.ic_state_dirty is True
    refuses(model)


def test_a_simulator_snapshot_of_an_advanced_state_restores_its_mark(model):
    """Simulator.snapshot() after run_until, then reset() to the declared ICs,
    then Simulator.restore(): A is back at A_ss = 4 and so is the mark. On main
    the restore wrote the species with set_concentration, which leaves the
    flag as reset() left it, and the next sensitivity run gave [0, 0] for
    [2, -8]."""
    sim = bngsim.Simulator(model, method="ode")
    sim.run_until(2000.0, **TOL)
    snap = sim.snapshot()
    model.reset()
    assert model._core.ic_state_dirty is False
    sim.restore(snap)
    assert model.get_concentration("A()") == pytest.approx(KP / KD, rel=1e-9)
    refuses(model)


def test_a_simulator_restore_keeps_a_mark_already_set(model):
    """Simulator.restore() of a snapshot taken before any run, into the model
    the run then advanced: the mark stays, as for restore_concentrations()."""
    sim = bngsim.Simulator(model, method="ode")
    snap = sim.snapshot()
    sim.run_until(5.0, **TOL)
    sim.restore(snap)
    assert model.get_concentration("A()") == 0.0
    assert model._core.ic_state_dirty is True
    refuses(model)


def test_a_simulator_snapshot_of_a_fresh_state_restores_clean(model):
    """Control: a snapshot taken before any run carries no mark, and restoring
    it after reset() is a fresh start, the closed form from A(0) = 0."""
    sim = bngsim.Simulator(model, method="ode")
    snap = sim.snapshot()
    sim.run_until(5.0, **TOL)
    model.reset()
    sim.restore(snap)
    assert model._core.ic_state_dirty is False
    np.testing.assert_allclose(
        dA(sens(model).run(sample_times=TS, **TOL)), from_literal(0.0), rtol=1e-6
    )


def test_a_batch_from_a_saved_equilibrated_state_refuses(model):
    """run_batch's rows are clones reset() to the baseline. On main each row of
    a sensitivity batch from the saved equilibrated state came back with
    dA(0)/d(kp, kd) = [0, 0], with no warning."""
    plain_equilibration(model)
    model.save_concentrations()
    with pytest.raises(bngsim.SimulationError, match=CARRIED):
        sens(model).run_batch((0.0, 2.0), 2, params=[{"kc": 1.0}, {"kc": 2.0}], **TOL)


def test_run_until_then_save_still_refuses(model):
    """run_until -> Simulator.save_concentrations() -> sensitivity run."""
    sim = bngsim.Simulator(model, method="ode")
    sim.run_until(2000.0, **TOL)
    sim.save_concentrations()
    refuses(model)


def test_steady_state_sensitivity_to_a_conserved_total_still_refuses(tmp_path):
    """A <-> B with A(0) = A0: the steady state moves with A0 through the total.
    After a plain run the state no longer says by how much, so the column is
    refused; after a save it came back as 0, where the protocol's own
    derivative is kr/(kf + kr) = 1/3 for A."""
    path = tmp_path / "exchange.net"
    path.write_text(
        """begin parameters
    1 kf 1.0
    2 kr 0.5
    3 A0 3.0
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
"""
    )
    m = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(m, method="ode")
    sim.run(t_span=(0, 5), n_points=2, **TOL)
    m.save_concentrations()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="carried-over"):
        sim.steady_state(sensitivity_params=["A0"])


# ── What was right stays right ────────────────────────────────────────────────


def test_a_literal_dose_on_a_fresh_model_saves_as_literal_ics(model):
    """No run advanced the state: the saved baseline is θ-independent literal
    ICs, and a sensitivity run from it, after reset() too, is the closed form
    from a literal A(0) = 1. Its dA(2)/dkp = (1 - e^-1)/kd = 1.264241 does not
    depend on A(0): it is the figure the bug printed in the issue's case 2,
    where the saved equilibrated state was seeded as if it were literal."""
    model.set_concentration("A()", 1.0)
    model.save_concentrations()
    assert model._core.ic_state_dirty is False
    got = dA(sens(model).run(sample_times=TS, **TOL))
    np.testing.assert_allclose(got, from_literal(1.0), rtol=1e-6)
    assert got[1, 0] == pytest.approx(1.264241, abs=1e-6)
    model.reset()
    assert model._core.ic_state_dirty is False
    np.testing.assert_allclose(
        dA(sens(model).run(sample_times=TS, **TOL)), from_literal(1.0), rtol=1e-6
    )


def test_a_named_literal_snapshot_restores_as_literal_ics(model):
    """A named snapshot of a state no run advanced restores clean."""
    model.set_concentration("A()", 1.0)
    model.save_concentrations("lit")
    model.restore_concentrations("lit")
    assert model._core.ic_state_dirty is False
    np.testing.assert_allclose(
        dA(sens(model).run(sample_times=TS, **TOL)), from_literal(1.0), rtol=1e-6
    )


def test_a_carried_equilibration_measures_from_its_seed(model):
    """Sensitivity equilibration -> save -> run(carry_sensitivities=True): the
    seed is the steady state's own, [2, -8], and with B = 0 it stays there. The
    same after reset() to the saved baseline, which restores the seed (#81)."""
    sim = sens(model)
    sim.run(**EQ)
    model.save_concentrations()
    want = np.array([[1 / KD, -KP / KD**2]] * len(TS))
    np.testing.assert_allclose(
        dA(sim.run(sample_times=TS, carry_sensitivities=True, **TOL)), want, rtol=1e-6
    )
    model.reset()
    np.testing.assert_allclose(
        dA(sim.run(sample_times=TS, carry_sensitivities=True, **TOL)), want, rtol=1e-6
    )


def test_a_reset_to_the_declared_ics_is_a_fresh_start(model):
    """Without a save, reset() after a run goes back to the declared ICs, A = 0,
    and a sensitivity run from there is the closed form from A(0) = 0."""
    plain_equilibration(model)
    model.reset()
    assert model._core.ic_state_dirty is False
    np.testing.assert_allclose(
        dA(sens(model).run(sample_times=TS, **TOL)), from_literal(0.0), rtol=1e-6
    )


def test_the_steady_state_column_from_the_initial_conditions(tmp_path):
    """Control for the refusal above: from the initial conditions the column is
    returned, d A_ss / d A0 = kr/(kf + kr) = 1/3."""
    path = tmp_path / "exchange.net"
    path.write_text(
        """begin parameters
    1 kf 1.0
    2 kr 0.5
    3 A0 3.0
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
"""
    )
    m = bngsim.Model.from_net(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=["A0"])
    col = np.asarray(ss.sensitivity)[:, 0]
    assert col[0] == pytest.approx(0.5 / 1.5, rel=1e-6)
    assert math.isclose(col[0] + col[1], 1.0, rel_tol=1e-6)
