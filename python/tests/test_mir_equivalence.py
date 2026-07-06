"""MIR micro-JIT codegen backend — equivalence with the default cc backend (GH #2).

The MIR backend (``BNGSIM_ENABLE_MIR=ON``, runtime ``BNGSIM_CODEGEN_JIT=mir``)
JIT-compiles *the same* code-generated C RHS source that the default backend
hands to ``cc`` + ``dlopen``. Only the header preamble differs (system
``#include``s stripped, libc/libm forward-declared and resolved via the import
resolver); the RHS body that determines the numerics is byte-identical. So a
correct MIR build must produce results bit-identical to the cc path.

This module is the CI assertion of that equivalence (issue #2). It is skipped
entirely on builds without the MIR backend, so it is a no-op on the default
wheels and only exercises anything on a ``-DBNGSIM_ENABLE_MIR=ON`` build run
under the MIR CI job.
"""

import os
import shutil

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import CvodeSimulator, NetworkModel, SolverOptions, TimeSpec
from bngsim._codegen import prepare_codegen, prepare_codegen_source

# Skip the whole module unless this extension actually embeds MIR.
pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_MIR", False),
    reason="bngsim built without the MIR backend (configure with -DBNGSIM_ENABLE_MIR=ON)",
)

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH for the cc baseline")

# Honor BNGSIM_TEST_DATA (run_tests.sh copies tests to a temp dir, breaking the
# __file__-relative path) exactly as test_codegen.py does.
DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)

# (net file, t_end, n_points) — the same well-characterized models the cc codegen
# suite pins, spanning linear decay, reversible mass action, an MM/QSSA rate law,
# and derived-parameter rate constants.
MODELS = [
    ("simple_decay.net", 50.0, 51),
    ("two_species_reversible.net", 1000.0, 101),
    ("mm_tqssa.net", 20.0, 41),
    ("derived_rate_const.net", 10.0, 50),
]


def _timespec(t_end, n_points):
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_points
    return ts


def _run_cc(net_path, ts):
    """Default backend: compile the codegen C source with cc, dlopen, integrate."""
    so = str(prepare_codegen(net_path))
    m = NetworkModel.from_net(net_path)
    s = CvodeSimulator(m)
    opts = SolverOptions()
    opts.codegen_so_path = so
    return s.run(ts, opts)


def _run_mir(net_path, ts):
    """MIR backend: JIT the same codegen C source in-process (codegen_c_source)."""
    src = prepare_codegen_source(net_path)
    assert src, "prepare_codegen_source returned empty C source"
    m = NetworkModel.from_net(net_path)
    s = CvodeSimulator(m)
    opts = SolverOptions()
    opts.codegen_c_source = src
    return s.run(ts, opts)


def _run_exprtk(net_path, ts):
    """Compiler-free baseline: the ExprTk-interpreted RHS (no codegen at all)."""
    m = NetworkModel.from_net(net_path)
    s = CvodeSimulator(m)
    return s.run(ts, SolverOptions())


@needs_cc
@pytest.mark.parametrize("net,t_end,n_points", MODELS, ids=[m[0] for m in MODELS])
def test_mir_matches_cc(net, t_end, n_points):
    """MIR JITs the identical C source cc compiles → trajectories must agree.

    Because both backends compile the *same* generated RHS, they are typically
    bit-identical (they are on x86_64). The residual is sub-ULP and purely a
    compiler codegen choice: clang at -O3 contracts multiply-adds into FMA
    instructions (aggressively on aarch64, where FMA is baseline), while MIR_gen
    does not fuse the same way — so on arm64 the two agree to ~1e-13 rather than
    bitwise. That is a rounding difference, not a numerical disagreement, so we
    assert equivalence to a tolerance far tighter than the solver's own.
    """
    path = os.path.join(DATA, net)
    ts = _timespec(t_end, n_points)
    r_cc = _run_cc(path, ts)
    r_mir = _run_mir(path, ts)

    sp_cc = np.array(r_cc.species_data)
    sp_mir = np.array(r_mir.species_data)
    assert sp_cc.shape == sp_mir.shape
    assert np.allclose(sp_cc, sp_mir, rtol=1e-9, atol=1e-12), (
        f"{net}: cc vs MIR max abs diff {np.max(np.abs(sp_cc - sp_mir)):.3e}"
    )
    # The near-identical RHS drives the adaptive integrator down the same step
    # sequence (the FMA residual is far below the step-size controller's floor).
    assert r_cc.solver_stats.n_steps == r_mir.solver_stats.n_steps


@pytest.mark.parametrize("net,t_end,n_points", MODELS, ids=[m[0] for m in MODELS])
def test_mir_matches_exprtk_to_tolerance(net, t_end, n_points):
    """MIR RHS matches the interpreted ExprTk RHS to solver tolerance (no cc needed).

    This is the assertion that survives on a compiler-free machine — the whole
    reason the MIR backend exists — so it does not carry the ``needs_cc`` mark.
    """
    path = os.path.join(DATA, net)
    ts = _timespec(t_end, n_points)
    r_ref = _run_exprtk(path, ts)
    r_mir = _run_mir(path, ts)

    sp_ref = np.array(r_ref.species_data)
    sp_mir = np.array(r_mir.species_data)
    assert sp_ref.shape == sp_mir.shape
    assert np.allclose(sp_ref, sp_mir, rtol=1e-6, atol=1e-9), (
        f"{net}: MIR vs ExprTk max abs diff {np.max(np.abs(sp_ref - sp_mir)):.3e}"
    )


def test_capabilities_reports_mir():
    """The public capability surface agrees this build has MIR."""
    assert bngsim.HAS_MIR is True
    assert bngsim.capabilities()["features"]["mir"] is True
