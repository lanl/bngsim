"""steady_state_batch says where its entries start (issue #1521).

After ``Model.set_concentration`` (or a ``run()``), ``steady_state_batch`` solved
every entry from the initial conditions: its clone's ``reset()`` dropped the
write. In ``A <-> B`` it returned the steady state of the seed total, 10, where
the caller had set 20; in a bistable model it returned the other branch. Each
entry said ``converged=True``, and nothing warned, though ``run_batch`` warns on
the same state (issue #1362).

The reset stays: it is what moves an initial-condition parameter's species to
the value an entry writes (issue #129), so a dose scan after a ``run()`` still
moves its totals. The batch now warns, as ``run_batch`` does, and its docs say
how to start from the current state. ``bngsim.jax.differentiable_solve``, which
resets its clone the same way, warns too.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

WARNING = r"steady_state_batch starts every entry from the model's initial conditions"
NET = (
    "begin parameters\n 1 kf 1.0\n 2 kr 0.5\n 3 A0 10\nend parameters\n"
    "begin species\n 1 A() A0\n 2 B() 0\nend species\n"
    "begin reactions\n 1 1 2 kf\n 2 2 1 kr\nend reactions\n"
)
# A <-> B conserves T = A + B; its steady state is T * [kr, kf] / (kf + kr).
SEED = np.array([10 * 0.5, 10 * 1.0]) / 1.5
WRITTEN = np.array([20 * 0.5, 20 * 1.0]) / 1.5


@pytest.fixture
def model(tmp_path: Path) -> bngsim.Model:
    path = tmp_path / "ab.net"
    path.write_text(NET)
    return bngsim.Model.from_net(str(path))


def _no_start_warning():
    """A context in which the start warning would fail the test."""
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.filterwarnings("error", message=WARNING)
    return ctx


def test_a_written_state_is_said_to_be_dropped(model):
    """The issue's A <-> B: the entry answers for the seed total, and says so."""
    model.set_concentration("A()", 20.0)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.warns(UserWarning, match=WARNING) as caught:
        (entry,) = sim.steady_state_batch(params=[{}])
    np.testing.assert_allclose(entry.concentrations, SEED, rtol=1e-6)
    message = str(caught[0].message)
    assert "states=[model.get_state()] * len(params)" in message
    assert "Model.save_concentrations()" in message


def test_the_end_of_a_run_is_said_to_be_dropped(model):
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 1.0), n_points=2)
    with pytest.warns(UserWarning, match=WARNING):
        sim.steady_state_batch(params=[{}])


def test_the_bistable_case_warns():
    """No conservation law: the seed X = 0.1 falls to the low branch, the
    written X = 3 to the high one (the issue's 0.1367 for 1.4633)."""
    pytest.importorskip("antimony")
    m = bngsim.Model.from_antimony_string(
        "J0: -> X; k0 + k1*X^2/(1 + X^2)\nJ1: X -> ; kd*X\nX = 0.1; k0 = 0.05; k1 = 1; kd = 0.5\n"
    )
    m.set_concentration("X", 3.0)
    sim = bngsim.Simulator(m, method="ode")
    high = float(np.asarray(sim.steady_state().concentrations)[0])
    with pytest.warns(UserWarning, match=WARNING):
        (entry,) = sim.steady_state_batch(params=[{}])
    assert float(np.asarray(entry.concentrations)[0]) < 0.5 < high


# ── The ways the warning names ───────────────────────────────────────────────


def test_states_starts_from_the_current_state_without_a_word(model):
    """Control: passes on main. The remedy the warning names."""
    model.set_concentration("A()", 20.0)
    sim = bngsim.Simulator(model, method="ode")
    ctx = _no_start_warning()
    try:
        (entry,) = sim.steady_state_batch(params=[{}], states=[model.get_state()])
    finally:
        ctx.__exit__(None, None, None)
    np.testing.assert_allclose(entry.concentrations, WRITTEN, rtol=1e-6)


def test_save_concentrations_makes_the_written_state_the_start(model):
    """Control: passes on main. The other remedy the warning names."""
    model.set_concentration("A()", 20.0)
    model.save_concentrations()
    sim = bngsim.Simulator(model, method="ode")
    ctx = _no_start_warning()
    try:
        (entry,) = sim.steady_state_batch(params=[{}])
    finally:
        ctx.__exit__(None, None, None)
    np.testing.assert_allclose(entry.concentrations, WRITTEN, rtol=1e-6)


@pytest.mark.parametrize("prepare", ["fresh", "reset-after-a-write"])
def test_a_model_at_its_initial_conditions_does_not_warn(model, prepare):
    """Control: passes on main."""
    if prepare == "reset-after-a-write":
        model.set_concentration("A()", 20.0)
        model.reset()
    sim = bngsim.Simulator(model, method="ode")
    ctx = _no_start_warning()
    try:
        (entry,) = sim.steady_state_batch(params=[{}])
    finally:
        ctx.__exit__(None, None, None)
    np.testing.assert_allclose(entry.concentrations, SEED, rtol=1e-6)


def test_a_dose_scan_after_a_run_still_moves_its_totals(model):
    """Control: passes on main. The reset is what puts each entry's A0 into
    A(): after a run, the totals are the doses, 5 and 40, not the run's 10."""
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 1.0), n_points=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        low, high = sim.steady_state_batch(params=[{"A0": 5.0}, {"A0": 40.0}])
    assert float(np.sum(low.concentrations)) == pytest.approx(5.0, rel=1e-6)
    assert float(np.sum(high.concentrations)) == pytest.approx(40.0, rel=1e-6)


def test_run_batch_still_says_the_same(model):
    """Control: passes on main. The message, moved into a shared helper, is
    the one ``test_replicates_and_batch_start_state.py`` pins."""
    model.set_concentration("A()", 20.0)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.warns(UserWarning) as caught:
        sim.run_batch(t_span=(0.0, 1.0), n_points=2, params=[{}])
    texts = [str(w.message) for w in caught]
    assert (
        "run_batch starts every point from the model's initial conditions (Model.reset), "
        "not from its current state, which differs from them (a set_concentration, or the "
        "end of a run). To start each point from the current state, use parameter_scan, or "
        "call Model.save_concentrations() first, which makes it the state reset returns "
        "to. Call Model.reset() first where the initial conditions are the start you want."
    ) in texts


# ── The JAX sibling ──────────────────────────────────────────────────────────


def test_differentiable_solve_says_so_too(model):
    """It resets its clone the same way: row 0 after a write of A() = 20 is
    the initial A() = 10."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from bngsim.jax import differentiable_solve

    jax.config.update("jax_enable_x64", True)
    model.set_concentration("A()", 20.0)
    p = jnp.asarray([model.get_param(n) for n in model.primary_param_names])
    with pytest.warns(UserWarning, match="differentiable_solve starts every solve"):
        y = differentiable_solve(model, p, (0.0, 1.0), 2)
    assert float(np.asarray(y)[0, 0]) == pytest.approx(10.0)
