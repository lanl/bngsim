"""Smoke test for the GH #102 reaction-kernel MVP demo.

Imports the demo module from ``benchmarks/kernel`` and exercises its synthetic
network generator + a tiny kernel loop, so the demo cannot silently bit-rot when
the bulk-state / ModelBuilder API changes. The heavy sweeps in the demo are not
run here — only the small, fast paths.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import bngsim
import numpy as np
import pytest

_DEMO = Path(__file__).resolve().parents[2] / "benchmarks" / "kernel" / "mvp_kernel_demo.py"


@pytest.fixture(scope="module")
def demo():
    if not _DEMO.exists():
        pytest.skip(f"demo not found at {_DEMO}")
    spec = importlib.util.spec_from_file_location("mvp_kernel_demo", _DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_shapes_and_determinism(demo):
    core = demo.build_random_linear_network(40, seed=3)
    assert core.n_species == 40
    assert core.n_reactions == 40
    # Each first-order reaction contributes a diagonal + one off-diagonal entry.
    assert core.codegen_jacobian_plan()["nnz"] == 80
    # Reproducible from the seed.
    again = demo.build_random_linear_network(40, seed=3)
    np.testing.assert_array_equal(core.get_state(), again.get_state())


def test_has_klu_returns_bool(demo):
    assert isinstance(demo.has_klu(), bool)


def test_tiny_kernel_loop_round_trips(demo):
    core = demo.build_random_linear_network(30, seed=1)
    sim = bngsim.Simulator(bngsim.Model(_core=core))
    state = sim.get_state()
    for step in range(3):
        sim.set_state(state)
        sim.run_until(t=(step + 1) * 2.0, n_points=2)
        state = sim.get_state()
    assert state.shape == (30,)
    assert np.all(np.isfinite(state))


def test_roundtrip_validation_passes_small(demo):
    assert demo.roundtrip_validation(n=80, seed=2, n_steps=4, t_end=8.0) is True
