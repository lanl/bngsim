"""Tier-1 large-model codegen chunking (dev/reaction_rhs_chunking_plan.md).

A flat RHS / sensitivity jac_vec over N reactions is one giant basic block whose
optimizer passes are superlinear in size, so large models take hours to compile.
At/above ``_chunk_threshold()`` reactions the body is split into NOINLINE helper
functions so compile time stays ~linear and the source compiles at -O2.

These tests pin the two invariants the change must hold:
  * Below the gate the emitted C is unchanged (no marker, no blocks) — existing
    models keep their cached .so and exact numerics.
  * Above the gate the chunked .so is *bit-identical* to the flat one (the split
    preserves reaction order, so every ydot/Jv_out accumulation order is kept).
"""

import ctypes
import os
import subprocess

import pytest
from bngsim import _codegen as cg

DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)


# ─── struct mirrors for ctypes calls into the compiled .so ──────────────────
class _CodegenUserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("tfun_ctx", ctypes.c_void_p),
        ("tfun_eval", ctypes.c_void_p),
    ]


class _SensUserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("plist", ctypes.POINTER(ctypes.c_int)),
        ("n_sens", ctypes.c_int),
    ]


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


def _build_so(c_source: str, tmp_path, tag: str) -> str:
    c_path = tmp_path / f"{tag}.c"
    so_path = tmp_path / f"{tag}{cg._shared_lib_suffix()}"
    c_path.write_text(c_source)
    cmd = cg._build_compile_cmd(c_path, so_path, "-O1")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, f"compile failed for {tag}:\n{res.stderr}"
    return str(so_path)


def _call_rhs(so_path, n_sp, n_par, y, p, t=0.41):
    lib = ctypes.CDLL(so_path)
    fn = lib.bngsim_codegen_rhs
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    parr = (ctypes.c_double * n_par)(*p)
    ud = _CodegenUserData(param_values=parr, tfun_ctx=None, tfun_eval=None)
    ya = (ctypes.c_double * n_sp)(*y)
    yd = (ctypes.c_double * n_sp)()
    fn(t, ya, yd, ctypes.byref(ud))
    return list(yd)


def _call_jac(so_path, n_sp, n_par, y, p, t=0.41):
    """Call the compiled dense ``bngsim_codegen_jac``; return the flat
    column-major ``jac[j*N_SPECIES + i]`` matrix."""
    lib = ctypes.CDLL(so_path)
    fn = lib.bngsim_codegen_jac
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    parr = (ctypes.c_double * n_par)(*p)
    ud = _CodegenUserData(param_values=parr, tfun_ctx=None, tfun_eval=None)
    ya = (ctypes.c_double * n_sp)(*y)
    jac = (ctypes.c_double * (n_sp * n_sp))()
    assert fn(t, ya, jac, ctypes.byref(ud)) == 0
    return list(jac)


def _call_outputs(so_path, n_sp, n_par, n_obs, n_func, y, p, t=0.41):
    """Call the compiled ``bngsim_codegen_outputs``; return (obs_out, func_out)."""
    lib = ctypes.CDLL(so_path)
    fn = lib.bngsim_codegen_outputs
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    parr = (ctypes.c_double * n_par)(*p)
    ud = _CodegenUserData(param_values=parr, tfun_ctx=None, tfun_eval=None)
    ya = (ctypes.c_double * n_sp)(*y)
    obs_out = (ctypes.c_double * n_obs)()
    func_out = (ctypes.c_double * n_func)()
    assert fn(t, ya, obs_out, func_out, ctypes.byref(ud)) == 0
    return list(obs_out), list(func_out)


def _call_sens(so_path, n_sp, n_par, y, yS, p, iS, t=0.41):
    lib = ctypes.CDLL(so_path)
    fn = lib.bngsim_codegen_sens_rhs
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    plist = list(range(n_par))
    parr = (ctypes.c_double * n_par)(*p)
    pl = (ctypes.c_int * n_par)(*plist)
    ud = _SensUserData(param_values=parr, plist=pl, n_sens=n_par)
    ya = (ctypes.c_double * n_sp)(*y)
    yda = (ctypes.c_double * n_sp)()
    ySa = (ctypes.c_double * n_sp)(*yS)
    ySd = (ctypes.c_double * n_sp)()
    t1 = (ctypes.c_double * n_sp)()
    t2 = (ctypes.c_double * n_sp)()
    fn(n_par, t, ya, yda, iS, ySa, ySd, ctypes.byref(ud), t1, t2)
    return list(ySd)


class TestChunkConfig:
    def test_threshold_env(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_CODEGEN_CHUNK", raising=False)
        assert cg._chunk_threshold() == cg._DEFAULT_CHUNK_THRESHOLD
        for val, exp in [("off", None), ("0", None), ("none", None), ("false", None)]:
            monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", val)
            assert cg._chunk_threshold() is exp
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
        assert cg._chunk_threshold() == 1
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "5000")
        assert cg._chunk_threshold() == 5000
        # garbage falls back to default, does not raise
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "lots")
        assert cg._chunk_threshold() == cg._DEFAULT_CHUNK_THRESHOLD

    def test_should_chunk_boundary(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_CODEGEN_CHUNK", raising=False)
        thr = cg._DEFAULT_CHUNK_THRESHOLD
        assert not cg._should_chunk(thr - 1)
        assert cg._should_chunk(thr)
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
        assert not cg._should_chunk(10**9)

    def test_block_size_env(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_CODEGEN_CHUNK_SIZE", raising=False)
        assert cg._chunk_block_size() == cg._DEFAULT_CHUNK_SIZE
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "64")
        assert cg._chunk_block_size() == 64
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "junk")
        assert cg._chunk_block_size() == cg._DEFAULT_CHUNK_SIZE

    def test_opt_flag_chunked_is_medium(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_CODEGEN_OPT", raising=False)
        # A chunked source compiles at -O2 at any size (the size-based -O0/-O1
        # downshift exists only to tame the flat giant function).
        assert cg._resolve_opt_flag("cc", 10**9, chunked=True) == "-O2"
        assert cg._resolve_opt_flag("cl", 10**9, chunked=True) == "/O2"
        # ... but an explicit BNGSIM_CODEGEN_OPT still wins.
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "none")
        assert cg._resolve_opt_flag("cc", 10**9, chunked=True) == "-O0"

    def test_emit_chunked_blocks_preserves_lines(self):
        groups = [[f"    a[{i}] += 1.0;"] for i in range(5)]
        defs, calls, protos = cg._emit_chunked_blocks(
            groups,
            fn_prefix="blk",
            signature_params="double* a",
            call_args="a",
            block_size=2,
            preamble=("double tmp;",),
        )
        assert len(calls) == 3  # ceil(5/2)
        assert len(protos) == 3  # one forward declaration per block
        joined = "\n".join(defs)
        # block indices are zero-padded to a minimum width of 3; blocks have
        # external linkage (not static) so they can compile as separate units.
        assert "BNGSIM_NOINLINE void blk_000(double* a)" in joined
        assert "static void blk_000" not in joined
        assert "void blk_000(double* a);" in protos  # plain forward declaration
        # every block is wrapped in shard sentinels (one open/close apiece)
        assert defs.count(cg._SHARD_BLOCK_OPEN) == 3
        assert defs.count(cg._SHARD_BLOCK_CLOSE) == 3
        assert joined.count("double tmp;") == 3  # preamble once per block
        for i in range(5):
            assert f"a[{i}] += 1.0;" in joined  # every line survives


class TestChunkGatingStructure:
    """Below the gate: no marker, no blocks. Above: marker + blocks present."""

    def test_flat_below_gate(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_CODEGEN_CHUNK", raising=False)
        src = cg.generate_rhs_c(os.path.join(DATA, "two_species_reversible.net"))
        assert cg._CHUNK_MARKER not in src
        assert "rxn_blk_" not in src
        # The NOINLINE macro is *defined* in every source now (it shares the
        # prelude with BNGSIM_EXPORT, which every entry point needs on Windows),
        # but below the gate it must go unused — no NOINLINE-decorated block.
        assert "BNGSIM_NOINLINE void" not in src

    def test_chunked_above_gate(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "1")
        src = cg.generate_rhs_c(os.path.join(DATA, "two_species_reversible.net"))
        assert cg._CHUNK_MARKER in src[:512]
        assert src.count("BNGSIM_NOINLINE void rxn_blk_") >= 2
        # each block has external linkage (not static) so it can compile as a
        # separate translation unit (GH #160)
        assert "static void rxn_blk_" not in src
        # each block appears as a prototype, a definition, and a call
        n_blocks = src.count("BNGSIM_NOINLINE void rxn_blk_")
        assert src.count("rxn_blk_") == 3 * n_blocks  # proto + def + call apiece


@requires_cc
class TestNetChunkEquivalence:
    """Chunked .net RHS is bit-identical to the flat one."""

    @pytest.mark.parametrize(
        "net",
        ["two_species_reversible", "saturation", "mm_tqssa", "func_composition", "fixed_species"],
    )
    def test_rhs_bit_identical(self, net, monkeypatch, tmp_path):
        import random

        path = os.path.join(DATA, f"{net}.net")
        parsed = cg._parse_net_file(path)
        n_sp = len(parsed["species"])
        n_par = len(parsed["parameters"])

        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
        flat = cg.generate_rhs_c(path)
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "1")  # 1 rxn/block: max split
        chunked = cg.generate_rhs_c(path)

        assert cg._CHUNK_MARKER not in flat
        assert cg._CHUNK_MARKER in chunked[:512]

        so_flat = _build_so(flat, tmp_path, f"{net}_flat")
        so_chunk = _build_so(chunked, tmp_path, f"{net}_chunk")

        rng = random.Random(hash(net) & 0xFFFF)
        for _ in range(4):
            y = [abs(rng.gauss(1.0, 0.6)) + 0.05 for _ in range(n_sp)]
            p = [abs(rng.gauss(1.0, 0.6)) + 0.05 for _ in range(n_par)]
            yf = _call_rhs(so_flat, n_sp, n_par, y, p)
            yc = _call_rhs(so_chunk, n_sp, n_par, y, p)
            assert yf == yc  # bit-identical, not just close


@requires_cc
class TestModelChunkEquivalence:
    """Chunked model-based RHS + sensitivity are bit-identical to flat."""

    def _build_model(self, n_sp=30, n_rxn=70, seed=7):
        import random

        from bngsim import Model
        from bngsim._bngsim_core import ModelBuilder

        rng = random.Random(seed)
        b = ModelBuilder()
        for i in range(n_rxn):
            b.add_parameter(f"k{i}", round(abs(rng.gauss(0.5, 0.3)) + 0.02, 4))
        for i in range(n_sp):
            b.add_species(f"S{i}", round(abs(rng.gauss(2.0, 1.0)) + 0.1, 4))
        for i in range(n_rxn):
            a, c = rng.randrange(n_sp), rng.randrange(n_sp)
            if rng.random() < 0.5:
                b.add_reaction([a, rng.randrange(n_sp)], [c], "elementary", f"k{i}")
            else:
                b.add_reaction([a], [c], "elementary", f"k{i}")
        return Model(b.build()), n_sp, n_rxn

    def test_combined_rhs_and_sens_bit_identical(self, monkeypatch, tmp_path):
        import random

        model, n_sp, n_par = self._build_model()

        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
        flat, has_sens = cg.generate_combined_from_model(model)
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "3")
        chunked, _ = cg.generate_combined_from_model(model)

        assert has_sens  # all-elementary ⇒ analytical sens RHS emitted
        assert cg._CHUNK_MARKER not in flat
        assert cg._CHUNK_MARKER in chunked[:512]
        assert "rxn_blk_" in chunked and "jacv_blk_" in chunked

        so_flat = _build_so(flat, tmp_path, "model_flat")
        so_chunk = _build_so(chunked, tmp_path, "model_chunk")

        rng = random.Random(99)
        for _ in range(4):
            y = [abs(rng.gauss(1.0, 0.6)) + 0.05 for _ in range(n_sp)]
            p = [abs(rng.gauss(0.5, 0.3)) + 0.02 for _ in range(n_par)]
            assert _call_rhs(so_flat, n_sp, n_par, y, p) == _call_rhs(so_chunk, n_sp, n_par, y, p)
            for iS in (0, n_par // 2, n_par - 1):
                yS = [rng.gauss(0.0, 1.0) for _ in range(n_sp)]
                sf = _call_sens(so_flat, n_sp, n_par, y, yS, p, iS)
                sc = _call_sens(so_chunk, n_sp, n_par, y, yS, p, iS)
                assert sf == sc

    def test_combined_dense_jacobian_bit_identical(self, monkeypatch, tmp_path):
        """The chunked analytical Jacobian (GH #165) is bit-identical to the flat
        one — the per-contribution scatter is lifted into NOINLINE shard blocks but
        called in the same fill order, so every ``jac[]`` accumulation is unchanged."""
        import random

        model, n_sp, n_par = self._build_model()
        # All-Elementary ⇒ the closed-form dense Jacobian is complete from the C++
        # build; warm it anyway (a no-op here) to mirror the ODE-solve setup.
        model.prepare_analytical_jacobian()

        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
        flat, _ = cg.generate_combined_from_model(model)
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "3")
        chunked, _ = cg.generate_combined_from_model(model)

        assert "bngsim_codegen_jac(" in flat  # dense (ns < sparse threshold)
        assert "jac_blk_" not in flat  # flat Jacobian below the gate
        assert "jac_blk_" in chunked  # Jacobian contributions sharded above it

        so_flat = _build_so(flat, tmp_path, "jac_flat")
        so_chunk = _build_so(chunked, tmp_path, "jac_chunk")

        rng = random.Random(7)
        for _ in range(5):
            y = [abs(rng.gauss(1.0, 0.6)) + 0.05 for _ in range(n_sp)]
            p = [abs(rng.gauss(0.5, 0.3)) + 0.02 for _ in range(n_par)]
            assert _call_jac(so_flat, n_sp, n_par, y, p) == _call_jac(so_chunk, n_sp, n_par, y, p)


@requires_cc
class TestObsFuncChunkEquivalence:
    """The obs[]/func[] computation — recomputed by the RHS, the analytical
    Jacobian, and the output evaluator — is itself a large basic block at genome
    scale. GH #165 shards it into NOINLINE blocks so it is not the serial driver
    wall. These pin that the chunked obs/func emit is bit-identical to the flat one
    across all three consumers, exercised on a model *with functions and
    observables* (the all-Elementary equivalence tests above never reach it)."""

    # Functional reaction (r1, rate g2) + an elementary reaction, two functions in
    # dependency order (g2 reads g1), and two observables one function reads — so
    # obs[]→func[] array references and the func→func chain are all exercised.
    _NET = "\n".join(
        [
            "begin parameters",
            "    1 k 0.3",
            "    2 a 1.5",
            "end parameters",
            "begin species",
            "    1 S1() 4.0",
            "    2 S2() 2.0",
            "    3 S3() 1.0",
            "end species",
            "begin functions",
            "    1 g1() a*O1 + 2.0",
            "    2 g2() k*g1 + O2",
            "end functions",
            "begin reactions",
            "    1 1 2 g2   #r1",
            "    2 2 3 k     #r2",
            "end reactions",
            "begin groups",
            "    1 O1 1",
            "    2 O2 1,2",
            "end groups",
            "",
        ]
    )

    def _model(self, tmp_path):
        from bngsim import Model

        net = tmp_path / "funcful.net"
        net.write_text(self._NET)
        m = Model.from_net(str(net))
        m.prepare_analytical_jacobian()
        return m

    def test_combined_obs_func_bit_identical(self, monkeypatch, tmp_path):
        import random

        m = self._model(tmp_path)
        core = m._core
        n_sp = core.n_species
        n_par = len(core.codegen_data()["parameters"])
        n_obs = core.n_observables
        n_func = core.n_functions

        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
        flat, _ = cg.generate_combined_from_model(m)
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "1")  # 1 item/block: max split
        chunked, _ = cg.generate_combined_from_model(m)

        # obs[]/func[] sharded in the RHS, the (dense, ns<50) Jacobian, and outputs.
        for pfx in (
            "rhs_obs_blk_",
            "rhs_func_blk_",
            "jac_obs_blk_",
            "jac_func_blk_",
            "out_obs_blk_",
            "out_func_blk_",
        ):
            assert pfx in chunked, f"expected {pfx} in chunked source"
            assert pfx not in flat, f"unexpected {pfx} in flat source"
        assert "bngsim_codegen_jac(" in flat  # dense Jacobian present to compare

        so_flat = _build_so(flat, tmp_path, "ff_flat")
        so_chunk = _build_so(chunked, tmp_path, "ff_chunk")

        rng = random.Random(5)
        for _ in range(6):
            y = [abs(rng.gauss(2.0, 1.0)) + 0.05 for _ in range(n_sp)]
            p = [abs(rng.gauss(0.6, 0.3)) + 0.02 for _ in range(n_par)]
            assert _call_rhs(so_flat, n_sp, n_par, y, p) == _call_rhs(so_chunk, n_sp, n_par, y, p)
            assert _call_jac(so_flat, n_sp, n_par, y, p) == _call_jac(so_chunk, n_sp, n_par, y, p)
            assert _call_outputs(so_flat, n_sp, n_par, n_obs, n_func, y, p) == _call_outputs(
                so_chunk, n_sp, n_par, n_obs, n_func, y, p
            )
