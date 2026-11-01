"""A state switch's residual reads as its condition does, on every platform (issue #963).

A switch on ``lhs > rhs`` is rooted on the residual ``(lhs)-(rhs)``, and the
solver reads the residual's sign as the side of the surface the state is on.
That holds only if the residual rounds lhs and rhs as the condition does. With
A held on thr, ``Aobs^2 > thr^2`` compares two equal rounded squares and reads
false. Where the compiler fused ``(Aobs^2)-(thr^2)`` into one multiply-add
(arm64, whose cores all fuse), the residual read +5.75e-17, the rounding error
of thr*thr, so the state looked across a surface it had not crossed, and a run
sat on it until the wall clock ended it.

bngsim now compiles ExprTk without contraction, in each target that compiles
it (bngsim's evaluator, and NFsim, whose copy of ExprTk's templates the linker
could otherwise keep), and codegen compiles its C the same way. Unfused, the
difference of two doubles is 0 exactly when they are equal.

The evaluator test can fail only on a build that fuses, which is arm64 (CI's
macos-15). The codegen tests compile the C themselves, under whatever flags let
this machine's compiler fuse (found by asking it), so they fail on x86-64
runners too.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg

THR = 1.0 + 5e-8
KB = 3.0

SWITCH = """begin parameters
    1 thr {thr!r}
    2 kb {kb!r}
    3 k 3
end parameters
begin functions
    1 fY() if({cond},kb,0)
end functions
begin species
    1 $A() {a!r}
    2 Y() 0
end species
begin reactions
    1 0 2 fY
end reactions
begin groups
    1 Aobs 1
end groups
"""


def _ulps_around(x, n):
    """The 2n + 1 doubles nearest x, in order, x in the middle."""
    below, above = [], []
    lo = hi = x
    for _ in range(n):
        lo = float(np.nextafter(lo, -np.inf))
        hi = float(np.nextafter(hi, np.inf))
        below.append(lo)
        above.append(hi)
    return below[::-1] + [x] + above


@pytest.mark.parametrize(
    "cond",
    ["Aobs^2>thr^2", "Aobs*Aobs>thr*thr", "Aobs*k>thr*k"],
    ids=["squared", "multiplied", "scaled"],
)
def test_the_residual_reads_as_its_condition_does_beside_the_surface(tmp_path, cond):
    """For A on each of the 33 doubles nearest thr, the residual is positive
    exactly where the condition is true, and 0 with A on thr. Fused, all three
    read a residual of the rounding error of thr*thr (or 3*thr) with A on thr
    and the condition false."""
    path = tmp_path / "switch.net"
    path.write_text(SWITCH.format(thr=THR, kb=KB, cond=cond, a=THR))
    core = bngsim.Model.from_net(path)._core
    residual, why = core.state_switch_residual(cond)
    assert residual, why
    for a in _ulps_around(THR, 16):
        taken = core.compute_derivs(0.0, np.array([a, 0.0]))[1] == KB
        g = core.evaluate_over_times(residual, np.array([0.0]))[0]
        assert (g > 0.0) == taken, (a, g, taken)
        if a == THR:
            assert g == 0.0


# ── Codegen ──────────────────────────────────────────────────────────────────

DIFFS = """begin parameters
    1 thr {thr!r}
    2 k 3
end parameters
begin functions
    1 fY() Aobs*Aobs-thr*thr
    2 fZ() Aobs*k-thr*k
end functions
begin species
    1 $A() {thr!r}
    2 Y() 0
    3 Z() 0
end species
begin reactions
    1 0 2 fY
    2 0 3 fZ
end reactions
begin groups
    1 Aobs 1
end groups
"""


def _compiler() -> list[str] | None:
    try:
        return cg._find_c_compiler()
    except Exception:
        return None


COMPILER = _compiler()
MSVC = COMPILER is not None and COMPILER[0].lower().endswith("cl")

# Fuses `a*a - c` where the compiler contracts by default (GCC in a GNU
# dialect, Clang) and the machine has a fused multiply-add: 5.75e-17 for 0.
_FUSION_PROBE = """volatile double a = 1.00000005;
int main(void) {
    volatile double c = a * a;
    return a * a - c != 0.0;
}
"""


@pytest.fixture(scope="module")
def fusing_flags(tmp_path_factory):
    """The flags under which this machine's C compiler fuses, found by asking
    it: none on arm64, -mfma on an x86-64 core that has one. A core without
    one runs the probe into an illegal instruction, and is passed over."""
    work = tmp_path_factory.mktemp("fusion_probe")
    src = work / "probe.c"
    src.write_text(_FUSION_PROBE)
    exe = work / ("probe.exe" if sys.platform == "win32" else "probe")
    for flags in ([], ["-mfma"]):
        built = subprocess.run(
            [*COMPILER, *flags, "-O2", str(src), "-o", str(exe)], capture_output=True, timeout=120
        )
        if built.returncode == 0 and subprocess.run([str(exe)], timeout=60).returncode == 1:
            return flags
    pytest.skip("this machine's C compiler has no fused multiply-add to use")


class _UserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("tfun_ctx", ctypes.c_void_p),
        ("tfun_eval", ctypes.c_void_p),
    ]


def _rhs(so_path, y, params):
    fn = ctypes.CDLL(str(so_path)).bngsim_codegen_rhs
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    values = (ctypes.c_double * len(params))(*params)
    ud = _UserData(param_values=values, tfun_ctx=None, tfun_eval=None)
    ya = (ctypes.c_double * len(y))(*y)
    yd = (ctypes.c_double * len(y))()
    assert fn(0.0, ya, yd, ctypes.byref(ud)) == 0
    return list(yd)


def _model(tmp_path):
    path = tmp_path / "diffs.net"
    path.write_text(DIFFS.format(thr=THR))
    model = bngsim.Model.from_net(path)
    params = [p["value"] for p in model._core.codegen_data()["parameters"]]
    return model, params


@pytest.mark.skipif(COMPILER is None, reason="no C compiler available")
@pytest.mark.skipif(MSVC, reason="gcc/clang only: the flags here are not MSVC's")
@pytest.mark.parametrize("build", ["serial", "sharded"])
def test_codegen_rounds_as_the_evaluator_does(tmp_path, monkeypatch, fusing_flags, build):
    """``Aobs*Aobs-thr*thr`` and ``Aobs*k-thr*k`` with A on thr are 0, and the
    generated C, compiled where it could fuse, reads them as the interpreter
    does, to the bit. Fused it read 5.75e-17 for the first, and a rate law's
    condition read in that C need not have agreed with its residual read by
    ExprTk."""
    if build == "sharded":
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "1")
    model, params = _model(tmp_path)
    y = [THR, 0.0, 0.0]
    interpreted = list(model._core.compute_derivs(0.0, np.array(y)))
    assert interpreted[1:] == [0.0, 0.0]

    src = cg.generate_rhs_from_model(model)
    so = tmp_path / f"diffs{cg._shared_lib_suffix()}"
    compiler = COMPILER + fusing_flags
    if build == "serial":
        c_path = tmp_path / "diffs.c"
        c_path.write_text(src)
        cmd = cg._build_compile_cmd(c_path, so, "-O3")
        cmd = compiler + cmd[len(COMPILER) :]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
    else:
        split = cg._split_sharded_source(src)
        assert split is not None
        driver, units = split
        cg._compile_sharded(driver, units, so, "-O3", 2, None, compiler)
    assert _rhs(so, y, params) == interpreted
