"""GH #160 — sharded parallel compilation of chunked codegen sources.

A chunked codegen source is one big translation unit compiled by a single serial
``cc``; for genome-scale models that compile dominates Simulator construction.
``compile_rhs`` splits the NOINLINE blocks into independent translation units and
compiles them with an allocation-aware, memory-bounded pool of ``cc -c``, then
links the ``.o`` files into the ``.so``.

These tests pin the invariants from the issue's "design constraints":
  * The job count is allocation-aware (never ``os.cpu_count()``) and
    memory-bounded; ``1`` (or a 1-core allocation) is the unchanged serial path.
  * The split is a pure function of the source — independent of the job count —
    and each unit compiles standalone.
  * The sharded ``.so`` is **bit-identical regardless of job count** and
    numerically bit-identical to the serial build (RHS and sensitivity RHS).
"""

from __future__ import annotations

import ctypes
import random
import subprocess

import pytest
from bngsim import _codegen as cg


def _synthetic_net(path, n_species: int, n_rxn: int, seed: int = 1234) -> int:
    """Write an all-elementary mass-action .net (⇒ both rxn and jacv blocks).
    Returns n_rxn (== n_params)."""
    rng = random.Random(seed)
    lines = ["begin parameters"]
    for i in range(n_rxn):
        lines.append(f"{i + 1} k{i} {round(abs(rng.gauss(0.5, 0.3)) + 0.02, 5)}")
    lines += ["end parameters", "begin species"]
    for i in range(n_species):
        lines.append(f"{i + 1} S{i}() {round(abs(rng.gauss(2.0, 1.0)) + 0.1, 5)}")
    lines += ["end species", "begin reactions"]
    for i in range(n_rxn):
        a, b = rng.randrange(n_species) + 1, rng.randrange(n_species) + 1
        c = rng.randrange(n_species) + 1
        lines.append(f"{i + 1} {a},{b} {c} k{i} #_R{i + 1}")
    lines += ["end reactions", "begin groups", "end groups"]
    path.write_text("\n".join(lines) + "\n")
    return n_rxn


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")
# Sharding is gated to gcc/clang; MSVC keeps the serial path.
requires_unix_cc = pytest.mark.skipif(
    not _has_cc() or cg._find_c_compiler()[0].lower().endswith("cl"),
    reason="sharded compile path is gcc/clang only",
)


def _chunked_combined(net_path, monkeypatch, block_size="4") -> tuple[str, int, int]:
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", block_size)
    src, has_sens = cg.generate_combined_c(str(net_path))
    assert cg._CHUNK_MARKER in src[:512]
    assert has_sens
    return src


def _call_rhs(so_path, n_sp, n_par, y, p, t=0.41):
    class UD(ctypes.Structure):
        _fields_ = [("param_values", ctypes.POINTER(ctypes.c_double)),
                    ("tfun_ctx", ctypes.c_void_p), ("tfun_eval", ctypes.c_void_p)]

    lib = ctypes.CDLL(str(so_path))
    fn = lib.bngsim_codegen_rhs
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double), ctypes.c_void_p]
    pa = (ctypes.c_double * n_par)(*p)
    ud = UD(param_values=pa, tfun_ctx=None, tfun_eval=None)
    ya, yd = (ctypes.c_double * n_sp)(*y), (ctypes.c_double * n_sp)()
    fn(t, ya, yd, ctypes.byref(ud))
    return list(yd)


def _call_sens(so_path, n_sp, n_par, y, yS, p, iS, t=0.41):
    class SUD(ctypes.Structure):
        _fields_ = [("param_values", ctypes.POINTER(ctypes.c_double)),
                    ("plist", ctypes.POINTER(ctypes.c_int)), ("n_sens", ctypes.c_int)]

    lib = ctypes.CDLL(str(so_path))
    fn = lib.bngsim_codegen_sens_rhs
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double)]
    pa = (ctypes.c_double * n_par)(*p)
    plist = (ctypes.c_int * n_par)(*range(n_par))
    sud = SUD(param_values=pa, plist=plist, n_sens=n_par)
    ya, ydot = (ctypes.c_double * n_sp)(*y), (ctypes.c_double * n_sp)()
    ySa, ySdot = (ctypes.c_double * n_sp)(*yS), (ctypes.c_double * n_sp)()
    t1, t2 = (ctypes.c_double * n_sp)(), (ctypes.c_double * n_sp)()
    fn(n_par, t, ya, ydot, iS, ySa, ySdot, ctypes.byref(sud), t1, t2)
    return list(ySdot)


class TestShardSplit:
    """The split is a pure, job-count-independent function of the source."""

    def test_split_structure(self, monkeypatch, tmp_path):
        net = tmp_path / "m.net"
        _synthetic_net(net, 30, 120)
        src = _chunked_combined(net, monkeypatch)

        # External linkage + sentinels + prototypes in the source.
        assert "static void rxn_blk_" not in src
        assert "BNGSIM_NOINLINE void rxn_blk_" in src
        assert "BNGSIM_NOINLINE void jacv_blk_" in src
        assert src.count(cg._SHARD_BLOCK_OPEN) == src.count(cg._SHARD_BLOCK_CLOSE) > 0

        split = cg._split_sharded_source(src)
        assert split is not None
        driver, units = split
        assert len(units) >= 2
        # Sentinels and block bodies live only in units, not the driver.
        assert cg._SHARD_BLOCK_OPEN not in driver
        assert "BNGSIM_NOINLINE void rxn_blk_000(" not in driver
        # Driver keeps the dispatcher and the prototypes it calls.
        assert "int bngsim_codegen_rhs(" in driver
        assert "void rxn_blk_000(" in driver  # prototype survives
        # Each unit is self-contained (carries the source prefix / includes).
        assert all("#include <math.h>" in u for u in units)
        assert all("BNGSIM_NOINLINE void" in u for u in units)

    def test_split_returns_none_when_unchunked(self, monkeypatch, tmp_path):
        net = tmp_path / "m.net"
        _synthetic_net(net, 6, 8)
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
        src, _ = cg.generate_combined_c(str(net))
        assert cg._CHUNK_MARKER not in src
        assert cg._split_sharded_source(src) is None

    def test_partition_independent_of_unit_count(self, monkeypatch, tmp_path):
        # The number of units is fixed by the source (blocks / _SHARD_UNIT_BLOCKS),
        # not by anything the caller passes — so two splits of the same source agree.
        net = tmp_path / "m.net"
        _synthetic_net(net, 30, 120)
        src = _chunked_combined(net, monkeypatch)
        d1, u1 = cg._split_sharded_source(src)
        d2, u2 = cg._split_sharded_source(src)
        assert d1 == d2 and u1 == u2


class TestShardJobs:
    """Allocation-aware (never os.cpu_count()) and memory-bounded job sizing."""

    def test_single_unit_is_serial(self):
        assert cg._resolve_codegen_jobs(1) == 1
        assert cg._resolve_codegen_jobs(0) == 1

    def test_explicit_one_is_serial(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "1")
        assert cg._resolve_codegen_jobs(99) == 1

    def test_explicit_count_capped_by_units(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "8")
        assert cg._resolve_codegen_jobs(3) == 3
        assert cg._resolve_codegen_jobs(99) <= 8

    def test_invalid_env_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "garbage")
        jobs = cg._resolve_codegen_jobs(99)
        assert 1 <= jobs <= cg._allocation_cpu_count()

    def test_auto_never_exceeds_allocation(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_CODEGEN_JOBS", raising=False)
        assert cg._resolve_codegen_jobs(999) <= cg._allocation_cpu_count()

    def test_slurm_cpus_per_task_caps_allocation(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
        assert cg._allocation_cpu_count() <= 2

    def test_allocation_is_at_least_one(self):
        assert cg._allocation_cpu_count() >= 1

    def test_memory_cap_bounds_jobs(self, monkeypatch):
        # With a known available-RAM budget, the job count is capped by
        # available / per-job — even when more cores and units are on offer.
        # Pin the probe so the test is platform-independent (macOS has no
        # readable cgroup/meminfo).
        monkeypatch.setattr(cg, "_available_memory_bytes", lambda: 4 * 1024**3)  # 4 GB
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "1000")  # ask for far more
        monkeypatch.setenv("BNGSIM_CODEGEN_MEM_PER_JOB", "1024")  # 1 GB/job ⇒ ≤4
        assert cg._resolve_codegen_jobs(99) == 4

    def test_memory_cap_can_force_serial(self, monkeypatch):
        monkeypatch.setattr(cg, "_available_memory_bytes", lambda: 512 * 1024**2)  # 512 MB
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "1000")
        monkeypatch.setenv("BNGSIM_CODEGEN_MEM_PER_JOB", "4096")  # 4 GB/job ⇒ 1
        assert cg._resolve_codegen_jobs(99) == 1

    def test_no_memory_cap_when_unreadable(self, monkeypatch):
        # Where no memory limit is readable, fall back to the CPU/unit cap only.
        monkeypatch.setattr(cg, "_available_memory_bytes", lambda: None)
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "3")
        assert cg._resolve_codegen_jobs(99) == 3

    # ── macOS memory cap (GH #168 follow-up) ─────────────────────────────────
    # macOS has no /proc/meminfo or cgroup files, so the parallel-compile cap was
    # silently disabled there; a cold genome-scale codegen under memory pressure
    # could overcommit and get jetsam-killed. vm_stat supplies the budget.
    _VM_STAT = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                    100000.\n"
        "Pages active:                                  200000.\n"
        "Pages inactive:                                 50000.\n"
        "Pages speculative:                                 500.\n"
        "Pages wired down:                              100000.\n"
        "Pages purgeable:                                 1500.\n"
    )

    def test_macos_vm_stat_parsed_as_available(self, monkeypatch):
        import types

        monkeypatch.setattr(
            cg.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(stdout=self._VM_STAT),
        )
        # free + inactive + speculative + purgeable = 152000 pages × 16384 B.
        assert cg._macos_available_memory_bytes() == 152000 * 16384

    def test_macos_vm_stat_failure_is_none(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("vm_stat missing")

        monkeypatch.setattr(cg.subprocess, "run", boom)
        assert cg._macos_available_memory_bytes() is None

    def test_macos_available_used_when_no_proc(self, monkeypatch):
        # On Darwin with no cgroup/meminfo, _available_memory_bytes uses vm_stat.
        # Neutralize the Linux sources so this holds on a Linux CI box too.
        monkeypatch.setattr(cg.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cg, "_read_int_file", lambda p: None)
        monkeypatch.setattr(cg, "_macos_available_memory_bytes", lambda: 7 * 1024**3)
        real_open = open

        def fake_open(path, *a, **k):
            if str(path) == "/proc/meminfo":
                raise FileNotFoundError(path)
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", fake_open)
        assert cg._available_memory_bytes() == 7 * 1024**3

    def test_macos_cap_throttles_under_pressure(self, monkeypatch):
        # End to end: a low vm_stat budget on Darwin caps the parallel jobs.
        monkeypatch.setattr(cg.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cg, "_read_int_file", lambda p: None)
        monkeypatch.setattr(cg, "_macos_available_memory_bytes", lambda: 3 * 1024**3)  # 3 GB
        real_open = open

        def fake_open(path, *a, **k):
            if str(path) == "/proc/meminfo":
                raise FileNotFoundError(path)
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", fake_open)
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "1000")
        monkeypatch.setenv("BNGSIM_CODEGEN_MEM_PER_JOB", "1024")  # 1 GB/job ⇒ ≤3
        assert cg._resolve_codegen_jobs(99) == 3


@requires_unix_cc
class TestShardCompile:
    """Sharded .so: numerically identical to serial, deterministic across jobs."""

    def _build_serial(self, src, out):
        cg._find_c_compiler()
        (out.parent / (out.name + ".c")).write_text(src)
        cmd = cg._build_compile_cmd(out.parent / (out.name + ".c"), out, "-O2")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr

    def _build_sharded(self, src, out, jobs):
        driver, units = cg._split_sharded_source(src)
        cg._compile_sharded(driver, units, out, "-O2", jobs, None, cg._find_c_compiler())
        return len(units)

    def test_rhs_and_sens_bit_identical_to_serial(self, monkeypatch, tmp_path):
        net = tmp_path / "m.net"
        n_par = _synthetic_net(net, 40, 200)
        n_sp = 40
        src = _chunked_combined(net, monkeypatch)

        suffix = cg._shared_lib_suffix()
        so_serial = tmp_path / f"serial{suffix}"
        so_shard = tmp_path / f"shard{suffix}"
        self._build_serial(src, so_serial)
        n_units = self._build_sharded(src, so_shard, jobs=4)
        assert n_units >= 2

        rng = random.Random(7)
        for _ in range(6):
            y = [abs(rng.gauss(1.0, 0.6)) + 0.05 for _ in range(n_sp)]
            p = [abs(rng.gauss(1.0, 0.6)) + 0.05 for _ in range(n_par)]
            assert _call_rhs(so_serial, n_sp, n_par, y, p) == \
                   _call_rhs(so_shard, n_sp, n_par, y, p)
            yS = [rng.gauss(0.0, 1.0) for _ in range(n_sp)]
            iS = rng.randrange(n_par)
            assert _call_sens(so_serial, n_sp, n_par, y, yS, p, iS) == \
                   _call_sens(so_shard, n_sp, n_par, y, yS, p, iS)

    def test_so_byte_identical_across_job_counts(self, monkeypatch, tmp_path):
        net = tmp_path / "m.net"
        _synthetic_net(net, 40, 200)
        src = _chunked_combined(net, monkeypatch)

        suffix = cg._shared_lib_suffix()
        so_j2 = tmp_path / f"j2{suffix}"
        so_j4 = tmp_path / f"j4{suffix}"
        self._build_sharded(src, so_j2, jobs=2)
        self._build_sharded(src, so_j4, jobs=4)
        assert so_j2.read_bytes() == so_j4.read_bytes()
        # And the determinism survives a repeat build at the same job count.
        so_j2b = tmp_path / f"j2b{suffix}"
        self._build_sharded(src, so_j2b, jobs=2)
        assert so_j2.read_bytes() == so_j2b.read_bytes()

    def test_compile_rhs_serial_and_sharded_agree(self, monkeypatch, tmp_path):
        net = tmp_path / "m.net"
        n_par = _synthetic_net(net, 40, 200)
        n_sp = 40
        src = _chunked_combined(net, monkeypatch)

        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "1")
        so_serial = cg.compile_rhs(src, "serialhash")
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "4")
        so_shard = cg.compile_rhs(src, "shardhash")
        assert so_serial.name != so_shard.name

        rng = random.Random(11)
        for _ in range(4):
            y = [abs(rng.gauss(1.0, 0.5)) + 0.1 for _ in range(n_sp)]
            p = [abs(rng.gauss(1.0, 0.5)) + 0.1 for _ in range(n_par)]
            assert _call_rhs(so_serial, n_sp, n_par, y, p) == \
                   _call_rhs(so_shard, n_sp, n_par, y, p)

        # Second call hits the cache and returns the same path (no recompile).
        assert cg.compile_rhs(src, "shardhash") == so_shard
