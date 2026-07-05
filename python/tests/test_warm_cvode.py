"""Warm CVODE hot path — CVodeReInit reuse parity (GH #102 Stage 0).

The warm path (``CvodeSimulator::run`` reusing persistent CVODE memory across
calls via ``CVodeReInit``) is a pure performance optimization: it must produce
trajectories byte-identical to the cold rebuild path for every model it covers
(ODE, no events, no sensitivities, non-JAX Jacobian). The ``BNGSIM_NO_WARM_CVODE``
environment escape hatch forces the cold path, so each test runs the *same*
model both ways and asserts exact equality.

These tests pin the optimization's correctness contract: equal results across
the warm/cold toggle, across repeated interactive steps, and across the
fingerprint-invalidation triggers (tolerance change, parameter edit) that force
a warm rebuild mid-session.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

_WARM_OFF = "BNGSIM_NO_WARM_CVODE"


def _linear_chain(n: int = 6, *, observables: bool = False):
    """A → B → ... first-order chain; species[0]=100, rest 0; spread rates."""
    b = ModelBuilder()
    sp = []
    for i in range(n):
        b.add_parameter(f"k{i}", 0.1 * (i + 1))
        sp.append(b.add_species(f"S{i}", 100.0 if i == 0 else 0.0))
    for i in range(n - 1):
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    if observables:
        b.add_observable("Total", [(i, 1.0) for i in range(n)])
    b.set_compute_conservation_laws(False)
    return b.build()


def _warm(monkeypatch):
    monkeypatch.delenv(_WARM_OFF, raising=False)


def _cold(monkeypatch):
    monkeypatch.setenv(_WARM_OFF, "1")


class TestWarmColdParity:
    @pytest.mark.parametrize("n", [6, 60, 300])
    def test_single_run_identical(self, monkeypatch, n):
        # n=6 dense, n=60/300 cross SPARSE_THRESHOLD → exercise both linsols.
        _warm(monkeypatch)
        warm = bngsim.Simulator(bngsim.Model(_core=_linear_chain(n))).run(
            t_span=(0.0, 50.0), n_points=51
        )
        _cold(monkeypatch)
        cold = bngsim.Simulator(bngsim.Model(_core=_linear_chain(n))).run(
            t_span=(0.0, 50.0), n_points=51
        )
        np.testing.assert_array_equal(warm.species, cold.species)
        np.testing.assert_array_equal(warm.observables, cold.observables)

    def test_stepwise_loop_identical(self, monkeypatch):
        # The kernel hot path: many small run_until steps reusing one simulator.
        def loop():
            sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(60)))
            for s in range(20):
                sim.run_until(t=(s + 1) * 2.0, n_points=2)
            return sim.get_state()

        _warm(monkeypatch)
        warm = loop()
        _cold(monkeypatch)
        cold = loop()
        np.testing.assert_array_equal(warm, cold)

    def test_steady_state_early_stop_identical(self, monkeypatch):
        def run():
            sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(60)))
            return sim.run(t_span=(0.0, 500.0), n_points=501, steady_state=True)

        _warm(monkeypatch)
        w = run()
        _cold(monkeypatch)
        c = run()
        # Same truncation point and same values.
        assert w.species.shape == c.species.shape
        np.testing.assert_array_equal(w.species, c.species)
        assert w.solver_stats["steady_state_reached"] == c.solver_stats["steady_state_reached"]


class TestWarmReuseInvalidation:
    """A warm simulator must rebuild its persistent state when a setup-affecting
    input changes, and still match a fresh cold run."""

    def test_tolerance_change_midsession(self, monkeypatch):
        # Second run on the same warm simulator at a tighter tolerance must match
        # a cold run at that tolerance (fingerprint mismatch → rebuild).
        _warm(monkeypatch)
        sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(60)))
        sim.run_until(t=10.0, n_points=2, rtol=1e-6, atol=1e-6)
        warm_after = sim.run_until(t=20.0, n_points=2, rtol=1e-12, atol=1e-12).species[-1]

        _cold(monkeypatch)
        ref = bngsim.Simulator(bngsim.Model(_core=_linear_chain(60)))
        ref.run_until(t=10.0, n_points=2, rtol=1e-6, atol=1e-6)
        cold_after = ref.run_until(t=20.0, n_points=2, rtol=1e-12, atol=1e-12).species[-1]

        np.testing.assert_array_equal(warm_after, cold_after)

    def test_param_edit_picked_up_after_warm_step(self, monkeypatch):
        # A direct model.set_param between warm run_until calls (no intervene)
        # must be reflected, identically to the cold path.
        def run():
            model = bngsim.Model(_core=_linear_chain(60))
            sim = bngsim.Simulator(model)
            sim.run_until(t=5.0, n_points=2)
            model.set_param("k0", 5.0)  # speed up S0→S1 drain
            return sim.run_until(t=15.0, n_points=2).species[-1]

        _warm(monkeypatch)
        warm = run()
        _cold(monkeypatch)
        cold = run()
        np.testing.assert_array_equal(warm, cold)


class TestWarmObservablesAndExpressions:
    def test_observables_identical_with_functions(self, monkeypatch):
        # A model carrying observables exercises the warm recording path.
        def run():
            sim = bngsim.Simulator(bngsim.Model(_core=_linear_chain(60, observables=True)))
            return sim.run(t_span=(0.0, 40.0), n_points=41)

        _warm(monkeypatch)
        w = run()
        _cold(monkeypatch)
        c = run()
        np.testing.assert_array_equal(w.observables, c.observables)
