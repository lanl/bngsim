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


def test_a_subset_model_keeps_the_mark_of_its_baseline(model):
    """make_subset_model rebuilds the model and saves its baseline on the new
    core; it kept the state and dropped the mark, so the subset's sensitivity
    run gave [0, 0] at t = 0, the literal start's numbers after."""
    from bngsim.coupling import make_subset_model

    plain_equilibration(model)
    model.save_concentrations()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = make_subset_model(model, keep_reactions=[0, 1])
    assert sub.get_concentration("A()") == pytest.approx(KP / KD, rel=1e-9)
    refuses(sub)
    sub.reset()
    refuses(sub)


def test_a_label_saved_again_from_a_fresh_state_restores_clean(model):
    """A label's mark is that of its latest save: saved from an advanced state,
    then saved again after reset() to the declared ICs, it restores as a
    fresh start, the closed form from A(0) = 0."""
    plain_equilibration(model)
    model.save_concentrations("x")
    model.reset()
    model.save_concentrations("x")
    model.restore_concentrations("x")
    assert model._core.ic_state_dirty is False
    np.testing.assert_allclose(
        dA(sens(model).run(sample_times=TS, **TOL)), from_literal(0.0), rtol=1e-6
    )


# ── Initial-condition sensitivities, ∂A(t)/∂A(0) = e^(-kd t) with B = 0 ─────


def ic_sens(model):
    r = bngsim.Simulator(model, method="ode", sensitivity_ic=["A()"]).run(sample_times=TS, **TOL)
    return np.asarray(r.sensitivities_ic)[:, 0, 0]


def test_ic_sensitivities_from_a_saved_equilibrated_state(model):
    """The saved state is the model's initial condition, carried over or not,
    and ∂A(t)/∂A(0) from it is e^(-kd t), as main returned; the mark the save
    keeps must not refuse it. The same after reset() to that baseline."""
    want = np.exp(-KD * np.array(TS))
    plain_equilibration(model)
    model.save_concentrations()
    np.testing.assert_allclose(ic_sens(model), want, rtol=1e-6)
    model.reset()
    np.testing.assert_allclose(ic_sens(model), want, rtol=1e-6)


def test_ic_sensitivities_off_the_saved_baseline_still_refuse(model):
    """A state that is not the initial condition is refused, as before: after
    the save, a dose of B and a run move A off the saved baseline."""
    plain_equilibration(model)
    model.save_concentrations()
    model.set_concentration("B()", 1.0)
    bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2, **TOL)
    assert model.get_concentration("A()") < KP / KD
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        ic_sens(model)


def test_ic_sensitivities_from_a_restored_advanced_snapshot_refuse(model):
    """A named snapshot of an advanced state, restored after reset() to the
    declared ICs, is not the initial condition, which is still A = 0: refused.
    Main let it through only because the snapshot lost its mark."""
    plain_equilibration(model)
    model.save_concentrations("eq")
    model.reset()
    model.restore_concentrations("eq")
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        ic_sens(model)


def test_ic_sensitivities_after_a_literal_dose_on_the_saved_baseline(model, tmp_path):
    """A set_concentration() on the saved baseline is a literal change of the
    initial condition, as on a fresh model, and ∂A(t)/∂A(0) from it seeds e_A.
    With B = 1 held by A + B -> B, dA/dt = kp - (kd + kc) A is linear, so
    ∂A(t)/∂A(0) = e^(-(kd + kc) t). The PR's first form compared the state with
    the baseline bit for bit and refused this; a fresh model with the same dose
    gives the same column. A dose undone is the baseline again, e^(-kd t). (A
    sensitivity run advances the state like any run, hence the reset() between
    the two measurements.)"""
    want = np.exp(-(KD + KC) * np.array(TS))
    plain_equilibration(model)
    model.save_concentrations()
    model.set_concentration("B()", 1.0)
    model.set_concentration("B()", 0.0)
    np.testing.assert_allclose(ic_sens(model), np.exp(-KD * np.array(TS)), rtol=1e-6)
    model.reset()
    model.set_concentration("B()", 1.0)
    np.testing.assert_allclose(ic_sens(model), want, rtol=1e-6)
    fresh = bngsim.Model.from_net(str(tmp_path / "catalysed_decay.net"))
    fresh.set_concentration("B()", 1.0)
    np.testing.assert_allclose(ic_sens(fresh), want, rtol=1e-6)


def test_ic_sensitivities_after_a_run_that_ends_where_it_started_refuse(tmp_path):
    """A run that advanced the state is refused even where it put the same
    numbers back: declared at its fixed point A = kp/kd = 4, B = 0, the
    right-hand side is exactly 0 and a run leaves the state bit-identical to
    the declared ICs. The bit-for-bit comparison let it through, as if no run
    had happened; main refused it, and so does the flag. The same numbers set
    by hand on a fresh model are a literal start."""
    path = tmp_path / "at_fixed_point.net"
    path.write_text(NET.replace("1 A() 0", f"1 A() {KP / KD}"))
    model = bngsim.Model.from_net(str(path))
    bngsim.Simulator(model, method="ode").run(t_span=(0, 3), n_points=2, **TOL)
    assert list(model.get_state()) == [KP / KD, 0.0]
    assert model._core.ic_state_dirty is True
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        ic_sens(model)
    model.reset()
    np.testing.assert_allclose(ic_sens(model), np.exp(-KD * np.array(TS)), rtol=1e-6)


def test_a_simulator_snapshot_of_an_advanced_state_refuses_ic_sensitivities(model):
    """Simulator.snapshot() of a state a run advanced, restored after reset()
    to the declared ICs, is not the initial condition, like a named snapshot:
    refused. A snapshot taken at the saved baseline restores as the initial
    condition, and gives e^(-kd t)."""
    sim = bngsim.Simulator(model, method="ode")
    sim.run_until(2000.0, **TOL)
    snap = sim.snapshot()
    model.reset()
    sim.restore(snap)
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        ic_sens(model)
    model.save_concentrations()
    at_baseline = sim.snapshot()
    model.reset()
    sim.restore(at_baseline)
    np.testing.assert_allclose(ic_sens(model), np.exp(-KD * np.array(TS)), rtol=1e-6)


def test_a_named_snapshot_saved_at_the_baseline_restores_as_the_initial_condition(model):
    """The named snapshot is saved after save_concentrations() made the
    equilibrated state the baseline: no run has moved the state since, so
    restored after reset() it is the initial condition, and gives e^(-kd t),
    where the same label saved before the unlabeled save is refused (above)."""
    plain_equilibration(model)
    model.save_concentrations()
    model.save_concentrations("eq")
    model.reset()
    model.restore_concentrations("eq")
    assert model._core.ic_state_dirty is True
    np.testing.assert_allclose(ic_sens(model), np.exp(-KD * np.array(TS)), rtol=1e-6)


def test_a_scan_leaves_the_initial_condition_axis_as_it_found_it(model):
    """parameter_scan puts the carry-over state back as it found it. From the
    saved baseline, the axis is open before the scan and after it; from an
    unsaved advanced state, refused before and after."""
    plain_equilibration(model)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        ic_sens(model)
    sim.parameter_scan("kc", [0.5, 1.0], t_span=(0, 2), n_points=2, **TOL)
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        ic_sens(model)
    model.save_concentrations()
    want = np.exp(-KD * np.array(TS))
    np.testing.assert_allclose(ic_sens(model), want, rtol=1e-6)
    model.reset()  # that measurement was a run too
    sim.parameter_scan("kc", [0.5, 1.0], t_span=(0, 2), n_points=2, **TOL)
    np.testing.assert_allclose(ic_sens(model), want, rtol=1e-6)


def test_a_clone_keeps_the_initial_condition_axis(model):
    """A clone of a state a run advanced refuses sensitivity_ic as the model
    does; reset() to the declared ICs opens it. A clone of the saved baseline
    is at its baseline."""
    plain_equilibration(model)
    clone = model.clone()
    with pytest.raises(bngsim.SimulationError, match="sensitivity_ic"):
        ic_sens(clone)
    clone.reset()
    want = np.exp(-KD * np.array(TS))
    np.testing.assert_allclose(ic_sens(clone), want, rtol=1e-6)
    model.save_concentrations()
    np.testing.assert_allclose(ic_sens(model.clone()), want, rtol=1e-6)


def test_an_ic_parameter_write_after_an_advanced_restore_leaves_the_species(tmp_path):
    """A write to an initial-condition parameter does not re-dose a state a run
    advanced (#79, #141): set_param("B0") leaves B where the run left it. A
    snapshot of such a state restored after reset() now carries the run's mark,
    so it behaves the same, through restore_concentrations() and through
    Simulator.restore(): B stays 0, and a run from there stays at the saved
    steady state, A(2) = kp/kd = 4. On main the first such restore let the
    write move B, and the run gave kp/(kd+kc) + (4 - kp/(kd+kc)) e^(-(kd+kc) 2)
    = 1.466099; a later one did not, once the reset had moved B's baseline to
    the dose. On a model no run advanced, the write moves B (control)."""
    path = tmp_path / "dosed_by_parameter.net"
    path.write_text(
        NET.replace("end parameters", "    4 B0 0\nend parameters").replace("2 B() 0", "2 B() B0")
    )
    fresh = bngsim.Model.from_net(str(path))
    fresh.set_param("B0", 1.0)
    assert fresh.get_concentration("B()") == 1.0

    m = bngsim.Model.from_net(str(path))
    plain_equilibration(m)
    m.save_concentrations("eq")
    m.set_param("B0", 1.0)
    assert m.get_concentration("B()") == 0.0
    m.set_param("B0", 0.0)
    m.reset()
    m.restore_concentrations("eq")
    m.set_param("B0", 1.0)
    assert m.get_concentration("B()") == 0.0
    a = bngsim.Simulator(m, method="ode").run(sample_times=TS, **TOL).species[:, 0]
    np.testing.assert_allclose(a, KP / KD, rtol=1e-9)

    m = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(m, method="ode")
    sim.run_until(2000.0, **TOL)
    snap = sim.snapshot()
    m.reset()
    sim.restore(snap)
    m.set_param("B0", 1.0)
    assert m.get_concentration("B()") == 0.0
    a = bngsim.Simulator(m, method="ode").run(sample_times=TS, **TOL).species[:, 0]
    np.testing.assert_allclose(a, KP / KD, rtol=1e-9)


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


def test_reset_then_set_state_is_a_literal_start_while_the_baseline_is_clean(model):
    """Control, as on main: with no save of the advanced state, reset() to the
    declared ICs and set_state() of the numbers the run left make them a literal
    start, from which dA/d(kp, kd) is the closed form from A(0) = kp/kd."""
    plain_equilibration(model)
    x = model.get_state().copy()
    model.reset()
    model.set_state(x)
    assert model._core.ic_state_dirty is False
    np.testing.assert_allclose(
        dA(sens(model).run(sample_times=TS, **TOL)), from_literal(KP / KD), rtol=1e-6
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
