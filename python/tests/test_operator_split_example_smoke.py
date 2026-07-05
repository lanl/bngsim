"""Smoke test for the GH #102 Stage 1 operator-split worked example.

Loads ``benchmarks/kernel/operator_split_example.py`` and runs its parts (in
``--quick`` form) so the documented two-subset acceptance example cannot silently
bit-rot when the coupling API changes. The numerical acceptance itself is pinned
by ``test_coupling.py::TestOperatorSplit``; this only guards the example runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "kernel" / "operator_split_example.py"
)


@pytest.fixture(scope="module")
def example():
    if not _EXAMPLE.exists():
        pytest.skip(f"example not found at {_EXAMPLE}")
    spec = importlib.util.spec_from_file_location("operator_split_example", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builders_produce_conserving_networks(example):
    chain = example.linear_chain([0.3, 0.5, 0.7])
    assert chain.n_species == 4 and chain.n_reactions == 3
    net = example.random_first_order_network(50, init_total=1000.0, seed=1)
    assert net.n_species == 50 and net.n_reactions == 50
    assert net.get_state().sum() == pytest.approx(1000.0)  # all mass seeded into S0


def test_strang_step_conserves_for_ode_ode(example):
    from bngsim import ReactionKernel, make_subset_model

    full = example.linear_chain([0.4, 0.6, 0.5])
    ka = ReactionKernel(make_subset_model(full, keep_reactions=[0, 2]))
    kb = ReactionKernel(make_subset_model(full, keep_reactions=[1]))
    state = ka.get_state()
    total0 = state.sum()
    for _ in range(5):
        state = example.strang_step(ka, kb, state, 1.0)
    assert state.sum() == pytest.approx(total0, abs=1e-7)
    assert np.all(state >= -1e-9)


def test_all_parts_run(example):
    # Each part prints and asserts its own invariants; just confirm no error.
    example.part1_ode_ode(quick=True)
    example.part2_ode_ssa(quick=True)
    example.part3_at_scale(scale=120)
    example.part4_division()
