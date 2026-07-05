"""ReactionKernel — the framework-agnostic reaction kernel facade (GH #102).

Covers Stage 0 deliverable #1/#4: a small object wrapping ``Model`` +
``Simulator`` that an external orchestrator drives per step via bulk state
exchange + ``advance(dt)``. The headline acceptance invariant is round-trip
equality — a standalone ``Simulator.run`` over ``[0, T]`` must equal the same
model advanced step-wise through the kernel, to integrator tolerance.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder
from bngsim.kernel import ReactionKernel


def _linear_chain(n: int = 5, *, observables: bool = False, conservation: bool = True):
    """A → B → C → ... first-order chain built via ModelBuilder.

    species[0] starts at 100, the rest at 0; rates spread so the trajectory is
    non-trivial. With ``observables=True`` adds two observables: the total mass
    (all species, factor 1) and the tail half (last ceil(n/2) species).
    """
    b = ModelBuilder()
    sp = []
    for i in range(n):
        b.add_parameter(f"k{i}", 0.1 * (i + 1))
        sp.append(b.add_species(f"S{i}", 100.0 if i == 0 else 0.0))
    for i in range(n - 1):
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    if observables:
        b.add_observable("Total", [(i, 1.0) for i in range(n)])
        b.add_observable("Tail", [(i, 1.0) for i in range(n // 2, n)])
    b.set_compute_conservation_laws(conservation)
    return b.build()


def _kernel(n: int = 5, **kw) -> ReactionKernel:
    return ReactionKernel(bngsim.Model(_core=_linear_chain(n, **kw)))


class TestConstruction:
    def test_wraps_model_and_exposes_simulator(self):
        model = bngsim.Model(_core=_linear_chain(4))
        k = ReactionKernel(model)
        assert isinstance(k.simulator, bngsim.Simulator)
        assert k.model is model
        assert k.method == "ode"
        assert k.time == 0.0

    def test_exported_from_package(self):
        assert bngsim.ReactionKernel is ReactionKernel
        assert "ReactionKernel" in bngsim.__all__

    def test_method_forwarded_to_simulator(self):
        k = ReactionKernel(bngsim.Model(_core=_linear_chain(4)), method="ssa")
        assert k.method == "ssa"

    def test_simulator_kwargs_forwarded(self):
        # force_dense_linear_solver is a Simulator kwarg; confirm it lands.
        k = ReactionKernel(bngsim.Model(_core=_linear_chain(4)), force_dense_linear_solver=True)
        assert k.simulator._force_dense_linear_solver is True

    def test_non_model_rejected(self):
        with pytest.raises(TypeError):
            ReactionKernel(object())  # type: ignore[arg-type]

    def test_from_simulator_adopts_existing(self):
        sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(4)))
        sim.run_until(t=5.0, n_points=2)
        k = ReactionKernel.from_simulator(sim)
        assert k.simulator is sim
        assert k.time == 5.0  # adopts the simulator's interactive clock

    def test_from_simulator_rejects_non_simulator(self):
        with pytest.raises(TypeError):
            ReactionKernel.from_simulator(object())  # type: ignore[arg-type]


class TestIntrospection:
    def test_names_and_counts(self):
        k = _kernel(5, observables=True)
        assert k.state_names == [f"S{i}" for i in range(5)]
        assert k.species_names == k.state_names
        assert k.observable_names == ["Total", "Tail"]
        assert k.n_species == 5
        assert k.n_observables == 2

    def test_state_names_match_get_state_order(self):
        k = _kernel(4)
        state = k.get_state()
        assert state.shape == (len(k.state_names),)
        for i, name in enumerate(k.state_names):
            assert state[i] == k.model.get_concentration(name)

    def test_repr(self):
        k = _kernel(3)
        assert "ReactionKernel" in repr(k)
        assert "n_species=3" in repr(k)


class TestStateExchange:
    def test_get_state_delegates_to_model(self):
        k = _kernel(4)
        np.testing.assert_array_equal(k.get_state(), k.model.get_state())

    def test_set_state_round_trips(self):
        k = _kernel(4)
        new = np.array([1.0, 2.0, 3.0, 4.0])
        k.set_state(new)
        np.testing.assert_array_equal(k.get_state(), new)

    def test_set_state_wrong_length_raises(self):
        k = _kernel(3)
        with pytest.raises(ValueError):
            k.set_state(np.array([1.0, 2.0]))


class TestAdvance:
    def test_advance_moves_time_and_returns_state(self):
        k = _kernel(5)
        s = k.advance(10.0)
        assert k.time == 10.0
        assert isinstance(s, np.ndarray)
        np.testing.assert_array_equal(s, k.get_state())

    def test_advance_returned_array_is_a_copy(self):
        k = _kernel(4)
        s = k.advance(5.0)
        s[0] = -123.0
        assert k.get_state()[0] != -123.0  # mutating the return doesn't leak

    def test_advance_accumulates_time(self):
        k = _kernel(4)
        k.advance(3.0)
        k.advance(4.0)
        assert k.time == pytest.approx(7.0)

    def test_advance_nonpositive_dt_raises(self):
        k = _kernel(4)
        with pytest.raises(ValueError):
            k.advance(0.0)
        with pytest.raises(ValueError):
            k.advance(-1.0)

    def test_last_result_exposed(self):
        k = _kernel(4)
        assert k.last_result is None
        k.advance(5.0)
        assert isinstance(k.last_result, bngsim.Result)
        # 2 output points by default: endpoints of the step.
        assert k.last_result.time.shape == (2,)

    def test_advance_n_points_controls_substeps(self):
        k = _kernel(4)
        k.advance(10.0, n_points=11)
        assert k.last_result.time.shape == (11,)
        np.testing.assert_allclose(k.last_result.time, np.linspace(0.0, 10.0, 11))


class TestAdvanceRoundTrip:
    """The Stage 0 acceptance invariant: standalone run == step-wise kernel."""

    @pytest.mark.parametrize("n_steps", [1, 4, 8, 20])
    def test_stepwise_equals_standalone(self, n_steps):
        T = 40.0
        dt = T / n_steps

        standalone = bngsim.Simulator(bngsim.Model(_core=_linear_chain(6)))
        ref_final = standalone.run(t_span=(0.0, T), n_points=2).species[-1]

        k = _kernel(6)
        final = k.get_state()
        for _ in range(n_steps):
            k.set_state(final)
            final = k.advance(dt)

        np.testing.assert_allclose(final, ref_final, rtol=1e-6, atol=1e-9)
        assert k.time == pytest.approx(T)

    def test_stepwise_observables_match_standalone(self):
        T, n_steps = 30.0, 6
        dt = T / n_steps

        standalone = bngsim.Simulator(bngsim.Model(_core=_linear_chain(6, observables=True)))
        ref = standalone.run(t_span=(0.0, T), n_points=2)
        ref_obs = ref.observables[-1]

        k = _kernel(6, observables=True)
        for _ in range(n_steps):
            k.advance(dt)

        np.testing.assert_allclose(k.observables(), ref_obs, rtol=1e-6, atol=1e-9)


class TestObservables:
    def test_observables_before_advance_match_initial_state(self):
        # Pre-advance observables come from the side-effect-free clone probe.
        k = _kernel(5, observables=True)
        obs = k.observables()
        # Total = sum of all species = 100 (only S0 seeded); Tail excludes S0..
        assert obs[0] == pytest.approx(100.0)
        assert obs[1] == pytest.approx(0.0)

    def test_observables_probe_does_not_mutate_model(self):
        k = _kernel(5, observables=True)
        before = k.get_state().copy()
        t_before = k.time
        _ = k.observables()
        np.testing.assert_array_equal(k.get_state(), before)
        assert k.time == t_before  # the clone probe never touched our clock

    def test_observables_reflect_set_state_before_advance(self):
        k = _kernel(4, observables=True)
        k.set_state(np.array([10.0, 20.0, 30.0, 40.0]))
        obs = k.observables()
        assert obs[0] == pytest.approx(100.0)  # Total = 10+20+30+40

    def test_observables_after_advance_come_from_step_result(self):
        k = _kernel(5, observables=True)
        k.advance(10.0)
        np.testing.assert_allclose(k.observables(), k.last_result.observables[-1])

    def test_no_observables_returns_empty(self):
        k = _kernel(4, observables=False)
        assert k.observables().shape == (0,)


class TestReset:
    def test_reset_restores_initial_state_and_time(self):
        k = _kernel(5)
        ic = k.get_state().copy()
        k.advance(20.0)
        assert k.time == 20.0
        assert not np.allclose(k.get_state(), ic)

        k.reset()
        assert k.time == 0.0
        assert k.last_result is None
        np.testing.assert_array_equal(k.get_state(), ic)

    def test_reset_then_advance_reproduces_first_run(self):
        k = _kernel(5)
        first = k.advance(15.0)
        k.reset()
        second = k.advance(15.0)
        np.testing.assert_array_equal(first, second)


class TestStochasticBackends:
    """SSA/PSA are stateful, so advance() works (round-trip equality does not
    hold for stochastic dynamics — only smoke + invariants)."""

    def test_ssa_advance_steps_and_conserves_mass(self):
        k = ReactionKernel(bngsim.Model(_core=_linear_chain(5)), method="ssa")
        total0 = k.get_state().sum()
        for step in range(5):
            k.advance(2.0, seed=step)
        assert k.time == pytest.approx(10.0)
        state = k.get_state()
        assert np.all(np.isfinite(state))
        # Pure first-order transfer chain conserves total molecule count.
        assert state.sum() == pytest.approx(total0)

    def test_psa_advance_steps(self):
        k = ReactionKernel(bngsim.Model(_core=_linear_chain(5)), method="psa", poplevel=50.0)
        k.advance(5.0, seed=1)
        assert k.time == pytest.approx(5.0)
        assert np.all(np.isfinite(k.get_state()))
