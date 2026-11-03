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

import ast
import errno
import os
import shutil
import subprocess
import sys

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import (
    CvodeSimulator,
    NetworkModel,
    SolverOptions,
    TimeSpec,
    _mir_jit_compile,
)
from bngsim._codegen import prepare_model_codegen, prepare_model_codegen_source

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
    so = str(prepare_model_codegen(bngsim.Model.from_net(net_path)))
    m = NetworkModel.from_net(net_path)
    s = CvodeSimulator(m)
    opts = SolverOptions()
    opts.codegen_so_path = so
    return s.run(ts, opts)


def _run_mir(net_path, ts):
    """MIR backend: JIT the same codegen C source in-process (codegen_c_source)."""
    src = prepare_model_codegen_source(bngsim.Model.from_net(net_path))
    assert src, "prepare_model_codegen_source returned empty C source"
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


# ── Issue #1215: the address space each MIR context reserves for its code ────
# third_party/mir/README.md.bngsim says why CMake sets 32 MB.
_RESERVE_PROBE = """
import os, subprocess, sys
import bngsim

def vsize_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1])
    except OSError:
        pass
    out = subprocess.run(["ps", "-o", "vsz=", "-p", str(os.getpid())],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])

def live():
    sim = bngsim.Simulator(bngsim.Model.from_net(sys.argv[1]), method="ode", codegen=True)
    sim.run(t_span=(0.0, 1.0), n_points=2)
    assert sim.codegen_backend == "mir", sim.codegen_backend
    return sim

keep = [live() for _ in range(3)]
before = vsize_kb()
keep += [live() for _ in range(int(sys.argv[2]))]
print((vsize_kb() - before) / 1024 / int(sys.argv[2]))
"""


def test_a_live_mir_model_holds_far_less_than_upstreams_reservation():
    """Address space, not memory: 100 live MIR-JIT'd models of a two-species
    network. With upstream's 128 MB reservation each held 128 MB; with the
    default 32 MB, a little over 32. The bound catches a fall back to
    upstream's size, not the exact setting: a build configured at 0 passes too."""
    if sys.platform == "win32":
        pytest.skip("POSIX-specific: MIR reserves no address space for its code on Windows")
    net = os.path.join(DATA, "simple_decay.net")
    env = {**os.environ, "BNGSIM_CODEGEN_JIT": "mir"}
    done = subprocess.run(
        [sys.executable, "-c", _RESERVE_PROBE, str(net), "100"],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    per_model_mb = float(done.stdout.strip().splitlines()[-1])
    # Measured on macOS arm64: 32.1 MB a model at the default 32, 128.1 at
    # upstream's 128 and 0.1 at 0, so a model's other mappings come to about
    # 0.1 MB. The growth is the whole address space's, though, and that holds
    # the heap too, about 1.1 MB a live model, which the macos-14 and -15
    # runners map in regions of up to 128 MB, 128-512 MB for 300 MB of
    # mallocs (#1402). Two taken among 20 models read as 45.65 MB a model
    # against a bound of 40. Over 100, 512 MB is 5.1 MB a model, and 64,
    # halfway to upstream's 128 by ratio, leaves room for it, while upstream's
    # 128 still fails it by a factor of two.
    assert per_model_mb <= 64.0, per_model_mb


# ── Issue #1215: running out is a RuntimeError, not the end of the process ───
# MIR answered a failed allocation or code mapping by ending the process: exit(1)
# from its error function and from its growable arrays, or a write through the
# NULL code address a failed mapping returned. MirJit now hands each context an
# allocator, a code allocator and an error function that return to it, and
# raises. _mir_jit_compile fails one allocation or code mapping of a compile, so
# these fail each in turn, in a child process: a crash is a failed test there,
# not a lost worker.
_SWEEP_PROBE = """
import sys
from bngsim._bngsim_core import _mir_jit_compile

src = open(sys.argv[1]).read()
kind = sys.argv[2]
expected = "(out of memory)" if kind == "allocation" else "(no address space to map code)"
wrong = []
for k in map(int, sys.argv[3:]):
    try:
        if kind == "allocation":
            _mir_jit_compile(src, k)
        else:
            _mir_jit_compile(src, -1, k)
        wrong.append((k, "no exception"))
    except RuntimeError as e:
        if expected not in str(e):
            wrong.append((k, str(e)))
# The thread compiles once the failures stop. (Not checked against the clean
# allocation count: under ASan one compile once took one allocation fewer.)
_mir_jit_compile(src)
print(wrong[:5])
"""

# A compile MIR failed in keeps its heap (see MirJit::throw_mir_failure), up to
# about 2 MB on simple_decay, so no child fails more than this many.
_FAILURES_PER_CHILD = 200


def _sweep(tmp_path, net, kind, step):
    src = tmp_path / "model.c"
    src.write_text(prepare_model_codegen_source(bngsim.Model.from_net(os.path.join(DATA, net))))
    allocations, _ = _mir_jit_compile(src.read_text())
    _, mappings = _mir_jit_compile(src.read_text(), -1, 1 << 40)  # the reservation refused
    n = allocations if kind == "allocation" else mappings
    points = sorted(set(range(min(n, 300))) | set(range(0, n, step)) | {n - 1})
    for i in range(0, len(points), _FAILURES_PER_CHILD):
        chunk = [str(k) for k in points[i : i + _FAILURES_PER_CHILD]]
        done = subprocess.run(
            [sys.executable, "-c", _SWEEP_PROBE, str(src), kind, *chunk],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert done.returncode == 0, (chunk[0], done.returncode, done.stderr[-2000:])
        assert done.stdout.strip().splitlines()[-1] == "[]", done.stdout
    return n


def test_every_allocation_mir_makes_can_fail_into_a_runtime_error(tmp_path):
    """35,028 allocations compile simple_decay; each of the first 300 (MIR_init2,
    c2mir_init and the start of c2mir) and every 23rd after is failed in turn.
    On main the first ended the process with "Not enough memory for ctx" and a
    later one in a write through freed or uninitialized MIR state."""
    n = _sweep(tmp_path, "simple_decay.net", "allocation", 23)
    assert n > 1000, n


def test_a_mapping_refused_for_another_reason_is_not_called_running_out():
    """A refusal other than ENOMEM is a policy against executable memory (SELinux
    deny_execmem, a hardened runtime without the JIT entitlement), which no cap
    on the address space explains: it is reported as what it is."""
    src = prepare_model_codegen_source(
        bngsim.Model.from_net(os.path.join(DATA, "simple_decay.net"))
    )
    with pytest.raises(RuntimeError) as refused:
        _mir_jit_compile(src, -1, 0, errno.EACCES)
    assert "cannot map executable memory for code: " in str(refused.value)
    assert "ran out of memory" not in str(refused.value)
    with pytest.raises(RuntimeError, match=r"ran out of memory or address space"):
        _mir_jit_compile(src, -1, 0, errno.ENOMEM)


def test_every_code_mapping_mir_makes_can_fail_into_a_runtime_error(tmp_path):
    """With the reservation refused MIR maps its code a page at a time, first in
    MIR_init2 (the wrapper-end thunk) and then per function it generates. Each
    mapping is failed in turn; on main a failed one came back as a NULL code
    address that MIR wrote through."""
    for net in ("simple_decay.net", "egfr_simple_seed.net"):
        assert _sweep(tmp_path, net, "mapping", 1) >= 1, net


_ABANDON_PROBE = """
import os, subprocess, sys
from bngsim._bngsim_core import _mir_jit_compile

def vsize_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1])
    except OSError:
        pass
    out = subprocess.run(["ps", "-o", "vsz=", "-p", str(os.getpid())],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])

src = open(sys.argv[1]).read()
n = int(sys.argv[2])
# Nine tenths of the way: the reservation is MIR_init2's first mapping, so it
# is long mapped, and every compile gets there. Not the last allocation: on
# arm64 a compile now and then makes fewer than the first did, and finishes.
# Late in a compile a failure keeps less heap than midway, 0.8 MB against 2.
late = _mir_jit_compile(src)[0] * 9 // 10

def fail():
    try:
        _mir_jit_compile(src, late)
    except RuntimeError:
        return
    raise AssertionError("the injected failure did not raise")

for _ in range(3):
    fail()
before = vsize_kb()
for _ in range(n):
    fail()
print((vsize_kb() - before) / 1024 / n)
"""


def test_a_compile_that_runs_out_gives_back_its_address_space(tmp_path):
    """MIR cannot finish a context it failed in, so MirJit abandons it, but it
    unmaps the context's code first: the reservation is what a cap on the
    address space counts. Failing an allocation late in a compile leaves the
    reservation mapped; 100 such failures keep about nothing of it each."""
    if sys.platform == "win32":
        pytest.skip("POSIX-specific: MIR reserves no address space for its code on Windows")
    src = tmp_path / "model.c"
    src.write_text(
        prepare_model_codegen_source(bngsim.Model.from_net(os.path.join(DATA, "simple_decay.net")))
    )
    done = subprocess.run(
        [sys.executable, "-c", _ABANDON_PROBE, str(src), "100"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    per_failure_mb = float(done.stdout.strip().splitlines()[-1])
    # A failure that kept its reservation keeps all 32 MB of it. What else
    # grows is the heap an abandoned context keeps, about 1 MB a failure, and
    # macOS maps a heap in regions of up to 128 MB on arm64 that its own
    # malloc statistics leave out: one taken among 20 failures read as 6.8-7.25
    # MB a failure on the macos-14 runners (issue #1402). Over 100 failures,
    # 0.8 MB a failure on macOS x86_64; three such regions on top of that
    # would read under 5. A kept reservation reads 32, four times the bound.
    assert per_failure_mb <= 8.0, per_failure_mb


_CAP_PROBE = """
import resource, sys
from bngsim._bngsim_core import _mir_jit_compile

def vsize_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024

src = open(sys.argv[1]).read()
_mir_jit_compile(src)
_, hard = resource.getrlimit(resource.RLIMIT_AS)
outcomes = []
for headroom_kb in (0, 64, 256, 1024, 4096, 16384, 65536):
    resource.setrlimit(resource.RLIMIT_AS, (vsize_bytes() + headroom_kb * 1024, hard))
    try:
        _mir_jit_compile(src)
        outcome = "compiled"
    except RuntimeError as e:
        outcome = "RuntimeError" if "MIR ran out of memory" in str(e) else "other: " + str(e)
    except MemoryError:
        outcome = "MemoryError"
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
    outcomes.append(outcome)
print(outcomes)
"""


def test_a_compile_at_a_cap_on_the_address_space_raises(tmp_path):
    """The issue's own setting, without the rest of a run: MIR compiles a model
    under a cap on the address space a little above the process's size, from
    nothing to 64 MB, the cap lifted between tries. Each compile that does not
    fit used to end the process; now it raises, and one with room compiles."""
    if not sys.platform.startswith("linux"):
        pytest.skip("needs RLIMIT_AS, which only Linux enforces")
    src = tmp_path / "model.c"
    src.write_text(
        prepare_model_codegen_source(bngsim.Model.from_net(os.path.join(DATA, "simple_decay.net")))
    )
    done = subprocess.run(
        [sys.executable, "-c", _CAP_PROBE, str(src)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, (done.returncode, done.stdout[-1000:], done.stderr[-2000:])
    outcomes = ast.literal_eval(done.stdout.strip().splitlines()[-1])
    assert set(outcomes) <= {"compiled", "RuntimeError", "MemoryError"}, outcomes
    assert "RuntimeError" in outcomes, outcomes
    assert outcomes[-1] == "compiled", outcomes
