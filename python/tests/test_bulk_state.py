"""Bulk state-vector API + kernel-loop round-trip (GH #102).

Covers the hard prerequisite for driving bngsim as a pluggable reaction kernel
from an external orchestrator: a vectorized bulk get/set of the live species
state as a numpy array (``Model.get_state`` / ``Model.set_state`` and the
``Simulator`` delegators), plus the ``ModelBuilder`` conservation-law opt-out
that keeps setup O(reactions) for large ODE-only networks.

The state vector is the raw per-species ``concentration`` storage, ordered like
``species_names()`` — the same values the ODE/SSA backends read as the initial
condition and write back as the final state.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder


def _linear_chain(n: int = 5, *, conservation: bool = True):
    """A → B → C → ... linear first-order chain, built via ModelBuilder.

    Returns the built C++ core. Rates are spread so the trajectory is
    non-trivial; species[0] starts at 100, the rest at 0.
    """
    b = ModelBuilder()
    sp = []
    for i in range(n):
        b.add_parameter(f"k{i}", 0.1 * (i + 1))
        sp.append(b.add_species(f"S{i}", 100.0 if i == 0 else 0.0))
    for i in range(n - 1):
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    b.set_compute_conservation_laws(conservation)
    return b.build()


class TestBulkStateRoundTrip:
    def test_get_state_matches_per_name(self):
        core = _linear_chain(5)
        model = bngsim.Model(_core=core)
        state = model.get_state()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float64
        names = model.species_names
        assert state.shape == (len(names),)
        # Ordering matches species_names() and the per-name accessor.
        for i, name in enumerate(names):
            assert state[i] == model.get_concentration(name)

    def test_set_state_round_trips(self):
        core = _linear_chain(4)
        model = bngsim.Model(_core=core)
        new = np.array([1.0, 2.0, 3.0, 4.0])
        model.set_state(new)
        np.testing.assert_array_equal(model.get_state(), new)
        for i, name in enumerate(model.species_names):
            assert model.get_concentration(name) == new[i]

    def test_set_state_copies_not_aliases(self):
        core = _linear_chain(3)
        model = bngsim.Model(_core=core)
        buf = np.array([5.0, 6.0, 7.0])
        model.set_state(buf)
        buf[0] = 999.0  # mutate caller's buffer after the call
        assert model.get_state()[0] == 5.0  # model state is unaffected

    def test_get_state_returns_fresh_array(self):
        core = _linear_chain(3)
        model = bngsim.Model(_core=core)
        a = model.get_state()
        a[0] = -1.0
        assert model.get_state()[0] != -1.0  # mutating the copy doesn't leak back

    def test_set_state_wrong_length_raises(self):
        core = _linear_chain(3)
        model = bngsim.Model(_core=core)
        with pytest.raises(ValueError):
            model.set_state(np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            model.set_state(np.zeros((3, 1)))  # not 1-D

    def test_set_state_forcecast_from_int(self):
        core = _linear_chain(3)
        model = bngsim.Model(_core=core)
        model.set_state(np.array([1, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(model.get_state(), [1.0, 2.0, 3.0])


class TestConservationToggle:
    def test_off_yields_no_laws_but_same_trajectory(self):
        # Conservation laws are consumed only by the steady-state solver; the
        # ODE path must be byte-identical whether or not they were detected.
        on = bngsim.Simulator(bngsim.Model(_core=_linear_chain(5, conservation=True)))
        off = bngsim.Simulator(bngsim.Model(_core=_linear_chain(5, conservation=False)))
        r_on = on.run(t_span=(0.0, 50.0), n_points=51)
        r_off = off.run(t_span=(0.0, 50.0), n_points=51)
        np.testing.assert_allclose(r_off.species, r_on.species, rtol=0, atol=0)

    def test_off_leaves_conservation_laws_empty(self):
        core = _linear_chain(5, conservation=False)
        cl = core.conservation_laws
        # n_species recorded, but no laws detected when the detector is skipped.
        assert cl["n_laws"] == 0


class TestKernelLoopRoundTrip:
    """Standalone run vs the same model advanced step-wise through the bulk
    state exchange — the MVP kernel-loop invariant (GH #102)."""

    def test_stepwise_via_get_set_matches_standalone(self):
        T, n_steps = 40.0, 8
        dt = T / n_steps

        standalone = bngsim.Simulator(bngsim.Model(_core=_linear_chain(6)))
        ref = standalone.run(t_span=(0.0, T), n_points=2)
        ref_final = ref.species[-1]

        sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(6)))
        state = sim.get_state()
        for s in range(n_steps):
            sim.set_state(state)  # inject the exchanged state
            sim.run_until(t=(s + 1) * dt, n_points=2)
            state = sim.get_state()  # extract for the next step

        np.testing.assert_allclose(state, ref_final, rtol=1e-6, atol=1e-9)

    def test_simulator_delegators_match_model(self):
        sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(4)))
        np.testing.assert_array_equal(sim.get_state(), sim._model.get_state())
        sim.set_state(np.array([9.0, 8.0, 7.0, 6.0]))
        np.testing.assert_array_equal(sim.get_state(), [9.0, 8.0, 7.0, 6.0])


class TestForceDenseLinearSolver:
    """GH #102 benchmarking knob — force the dense linear solver over auto KLU.

    The flag overrides only the linear-solver *kind*; results must be identical
    to the auto path (which selects sparse KLU above SPARSE_THRESHOLD when the
    build has it). Without KLU the auto path is already dense, so the flag is a
    harmless no-op and these tests still hold.
    """

    def test_default_is_false(self):
        sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(5)))
        assert sim._force_dense_linear_solver is False

    def test_force_dense_matches_auto(self):
        # N=60 > SPARSE_THRESHOLD with density well under SPARSE_DENSITY_MAX, so
        # the auto path picks KLU when available; forced dense must still match.
        auto = bngsim.Simulator(bngsim.Model(_core=_linear_chain(60)))
        dense = bngsim.Simulator(
            bngsim.Model(_core=_linear_chain(60)), force_dense_linear_solver=True
        )
        r_auto = auto.run(t_span=(0.0, 30.0), n_points=31)
        r_dense = dense.run(t_span=(0.0, 30.0), n_points=31)
        np.testing.assert_allclose(r_dense.species, r_auto.species, rtol=1e-7, atol=1e-9)
