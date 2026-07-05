"""Smoke test for the GH #102 reaction-kernel worked example.

Loads ``benchmarks/kernel/kernel_example.py`` and runs its parts so the
documented direct-call example cannot silently bit-rot when the ReactionKernel
API changes. Also re-checks the acceptance invariant the example demonstrates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "benchmarks" / "kernel" / "kernel_example.py"


@pytest.fixture(scope="module")
def example():
    if not _EXAMPLE.exists():
        pytest.skip(f"example not found at {_EXAMPLE}")
    spec = importlib.util.spec_from_file_location("kernel_example", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_demo_model(example):
    model = example.build_demo_model()
    assert model.n_species == 5
    assert model.observable_names == ["Total", "Downstream"]


def test_acceptance_invariant_holds(example):
    # The example's part 1 returns the relative error vs a standalone run.
    assert example.part1_acceptance_invariant() < 1e-6


def test_drive_loop_and_observables_run(example):
    # Exercise the remaining parts end-to-end (they print; assert no error).
    example.part2_per_step_drive_loop()
    example.part3_observables_readout()
