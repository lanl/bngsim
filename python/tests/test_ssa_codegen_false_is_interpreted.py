"""``Simulator(method="ssa", codegen=False)`` runs the interpreted propensities
on a MIR build too (issue #1213).

``codegen=False`` kept Python from building the cc propensity library and went
no further. The C++ selector read only ``BNGSIM_SSA_NO_CODEGEN``, so with no
library to load, its default fell through to ``MirJit::available()`` and
JIT'd the propensities in-process: the branch for an install with no compiler
(GH #139/#140) could not tell that apart from a caller who asked for no
codegen, and reported ``propensity_backend == "mir"``. Every place an SSA
engine is built now passes the caller's choice down: a plain run, the engine
rebuilt after ``intervene``, ``run_replicates``' per-thread engines and
``run_batch``'s per-row ones.

On a build without MIR the interpreted path was already taken, so these tests
pass there on main too; on a MIR build they failed.
"""

from __future__ import annotations

import bngsim
import pytest
from bngsim._bngsim_core import HAS_MIR, SsaSimulator, TimeSpec

NET = """begin parameters
    1 k 0.01
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1,1 2 k
end reactions
begin groups
    1 Aobs 1
end groups
"""
RUN = {"t_span": (0.0, 1.0), "n_points": 3, "seed": 1}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "dimer.net"
    path.write_text(NET)
    return bngsim.Model.from_net(str(path))


def _backend(result):
    return result.ssa_diagnostics["propensity_backend"]


def test_a_run_is_interpreted(model):
    sim = bngsim.Simulator(model, method="ssa", codegen=False)
    assert _backend(sim.run(**RUN)) == "interpreted"


def test_the_engine_rebuilt_after_intervene_is_interpreted(model):
    sim = bngsim.Simulator(model, method="ssa", codegen=False)
    sim.run(**RUN)
    sim.intervene({"k": 0.02})
    assert _backend(sim.run(**RUN)) == "interpreted"


def test_replicates_on_threads_are_interpreted(model):
    sim = bngsim.Simulator(model, method="ssa", codegen=False)
    results = sim.run_replicates(4, num_processors=2, **RUN)
    assert {_backend(r) for r in results} == {"interpreted"}


def test_batch_rows_are_interpreted(model):
    sim = bngsim.Simulator(model, method="ssa", codegen=False)
    results = sim.run_batch((0.0, 1.0), 3, params=[{"k": 0.01}, {"k": 0.03}], seed=1)
    assert {_backend(r) for r in results} == {"interpreted"}


@pytest.mark.parametrize(("enabled", "want"), [(False, "interpreted"), (True, "mir")])
def test_the_core_engine_keeps_to_set_codegen(model, enabled, want):
    """``SsaSimulator.set_codegen`` with no library to load, what the Python
    layer now passes for ``codegen=False``. Left on, a MIR build JITs."""
    if want == "mir" and not HAS_MIR:
        want = "interpreted"
    sim = SsaSimulator(model._core)
    sim.set_codegen(enabled)
    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 1.0, 3
    assert sim.run(times, 1).ssa_diagnostics.propensity_backend == want


@pytest.mark.skipif(
    not HAS_MIR, reason="built without the MIR backend, whose in-process fallback this is"
)
def test_left_on_with_no_library_a_mir_build_still_jits(model):
    """The compiler-less fallback stays where codegen is left on: an engine
    with no cc library (the one rebuilt after intervene) JITs on MIR."""
    sim = bngsim.Simulator(model, method="ssa")
    sim.run(**RUN)
    sim.intervene({"k": 0.02})
    assert _backend(sim.run(**RUN)) == "mir"
