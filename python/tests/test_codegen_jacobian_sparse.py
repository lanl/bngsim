"""Compiled-C *sparse* analytical Jacobian (GH #162) — ``bngsim_codegen_jac_sparse``.

For models the CVODE solver routes to the sparse KLU path (``ns >= 50`` with a
low-density Jacobian), the codegen ``.so`` now carries a C mirror of
``NetworkModel::fill_sparse_analytical_jacobian`` that fills the nnz-length CSC
value array (``jac_data[data_idx]``) instead of a dense ``n×n`` buffer — a dense
emit is infeasible at scale (a 75k-species dense Jacobian ≈ 45 GB).

These tests build a synthetic large *sparse* model (a linear elementary chain plus
a Michaelis–Menten reaction, a saturable Functional reaction over an aggregating
observable, and a fixed sink — so every contribution block and the fixed-row
zeroing are exercised) and assert:

  * the emitter routes to the sparse symbol (not the dense one) past the gate;
  * the compiled CSC values, scattered back to dense, match the interpreted
    analytical assembly (``_dense_analytical_jacobian`` — the FD-self-checked
    oracle the dense CVODE callback and the FD cross-check also use) to ~1e-12,
    and match a finite-difference Jacobian to FD tolerance;
  * an end-to-end KLU ODE run with the compiled sparse Jacobian matches one with
    the interpreted sparse Jacobian to solver tolerance;
  * the emit is byte-deterministic across ``PYTHONHASHSEED`` values;
  * a small model is unaffected — it still emits the dense symbol.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import textwrap

import bngsim
import numpy as np
import pytest
from bngsim._codegen import (
    generate_jacobian_from_model,
    prepare_codegen,
    prepare_model_codegen,
)

pytest.importorskip("sympy")
_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")


def _klu_available() -> bool:
    tiny = bngsim.Model.from_sbml_string(_TINY_SBML)
    tiny.prepare_analytical_jacobian()
    return bool(tiny._core.codegen_jacobian_plan()["has_klu"])


class _CodegenUserData(ctypes.Structure):
    # Mirrors CodegenUserDataForSO / the codegen typedef. The synthetic models
    # carry no table functions, so tfun_* stay NULL.
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("tfun_ctx", ctypes.c_void_p),
        ("tfun_eval", ctypes.c_void_p),
    ]


def _make_sparse_net(n_chain: int) -> str:
    """A synthetic *sparse* reaction network whose analytical Jacobian is complete.

    Layout (1-based species): ``A1..A{n_chain}`` form a linear elementary chain,
    ``E`` is an enzyme, and ``$Sink`` is a fixed boundary species. The Jacobian is
    bidiagonal-ish (a few nonzeros per column) so density ≪ 0.10 and, with
    ``n_chain >= ~50``, the model routes to KLU. Contributions covered:

      * Elementary mass-action — the chain ``A_i -> A_{i+1}`` and a drain.
      * Michaelis–Menten (tQSSA) — ``E + A1 -> E + A2``.
      * Functional (saturable, per-observable) — ``A3 -> Sink`` with rate
        ``fr() = ksat*Stot/(Ksat+Stot)`` over the aggregating observable ``Stot``.
      * A fixed-species row (``$Sink``) to exercise the CSC fixed-row zeroing.
    """
    e_idx = n_chain + 1
    sink_idx = n_chain + 2
    lines = ["begin parameters"]
    lines += [
        "    1 kf 0.3",
        "    2 kcat 1.2",
        "    3 Km 4.0",
        "    4 ksat 0.7",
        "    5 Ksat 2.5",
        "end parameters",
        "begin species",
    ]
    for i in range(1, n_chain + 1):
        lines.append(f"    {i} A{i}() {10.0 + i * 0.1}")
    lines.append(f"    {e_idx} E() 5.0")
    lines.append(f"    {sink_idx} $Sink() 0")
    lines.append("end species")
    lines += [
        "begin functions",
        "    1 fr() ksat*Stot/(Ksat+Stot)",
        "end functions",
        "begin reactions",
    ]
    r = 1
    for i in range(1, n_chain):
        lines.append(f"    {r} {i} {i + 1} kf   #chain_{i}")
        r += 1
    lines.append(f"    {r} {e_idx},1 {e_idx},2 MM kcat Km   #mm")
    r += 1
    lines.append(f"    {r} 3 {sink_idx} fr   #sat_deg")
    r += 1
    lines.append(f"    {r} {n_chain} {sink_idx} kf   #drain")
    r += 1
    lines += ["end reactions", "begin groups", "    1 Stot 1,2,3,4", "end groups", ""]
    return "\n".join(lines)


_TINY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sat_deg">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="2" constant="true"/>
      <parameter id="Km" value="5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/>
            <apply><times/><ci>k</ci><ci>S</ci></apply>
            <apply><plus/><ci>Km</ci><ci>S</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


needs_klu = pytest.mark.skipif(
    not (_CC and _klu_available()), reason="KLU sparse solver not built into this core"
)


def _write_net(tmp_path, n_chain: int):
    p = tmp_path / f"sparse_chain_{n_chain}.net"
    p.write_text(_make_sparse_net(n_chain))
    return p


def _prepared_sparse_model(tmp_path, n_chain: int):
    m = bngsim.Model.from_net(str(_write_net(tmp_path, n_chain)))
    # Functional models derive the analytical Jacobian lazily (GH #145); warm it
    # before codegen, exactly as the ODE-solve setup does.
    m.prepare_analytical_jacobian()
    return m


def _scatter_csc_to_dense(vals, col_ptrs, row_indices, ns):
    """Dense matrix J[row][col] from a CSC value array."""
    j_dense = np.zeros((ns, ns))
    for col in range(ns):
        for k in range(int(col_ptrs[col]), int(col_ptrs[col + 1])):
            j_dense[int(row_indices[k]), col] = vals[k]
    return j_dense


def _compiled_sparse_vals(core, so_path, plan, t, conc):
    """Call the compiled ``bngsim_codegen_jac_sparse`` → length-nnz CSC values."""
    lib = ctypes.CDLL(str(so_path))
    fn = lib.bngsim_codegen_jac_sparse
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    params = [p["value"] for p in core.codegen_data()["parameters"]]
    pbuf = (ctypes.c_double * len(params))(*params)
    ud = _CodegenUserData(
        param_values=ctypes.cast(pbuf, ctypes.POINTER(ctypes.c_double)),
        tfun_ctx=None,
        tfun_eval=None,
    )
    ns = core.n_species
    nnz = int(plan["nnz"])
    y = (ctypes.c_double * ns)(*conc)
    vals = (ctypes.c_double * nnz)()
    assert fn(float(t), y, vals, ctypes.byref(ud)) == 0
    return np.array(vals, dtype=float)


def _interpreted_dense(core, t, conc):
    """Interpreted analytical Jacobian as a dense J[row][col] (the oracle)."""
    ns = core.n_species
    # _dense_analytical_jacobian returns flat column-major jac[j*ns + i] = ∂f_i/∂x_j.
    flat = np.array(core._dense_analytical_jacobian(t, list(conc)), dtype=float)
    return flat.reshape(ns, ns).T  # -> J[i][j]


def _fd_dense(core, t, conc):
    """Forward finite-difference Jacobian J[row][col]."""
    ns = core.n_species
    conc = np.asarray(conc, dtype=float)
    f0 = np.array(core._eval_rhs(t, list(conc)), dtype=float)
    j_fd = np.zeros((ns, ns))
    for col in range(ns):
        step = 1e-6 * max(1.0, abs(conc[col]))
        cp = conc.copy()
        cp[col] += step
        f1 = np.array(core._eval_rhs(t, list(cp)), dtype=float)
        j_fd[:, col] = (f1 - f0) / step
    return j_fd


# ── Routing / emit ───────────────────────────────────────────────────────────


@needs_cc
@needs_klu
def test_routes_to_sparse_and_emits_sparse_symbol(tmp_path):
    m = _prepared_sparse_model(tmp_path, 70)
    core = m._core
    plan = core.codegen_jacobian_plan()
    assert plan["available"] and core.analytical_jacobian_complete
    # Structurally sparse: KLU gate (ns >= 50, density < 0.10, nonempty).
    assert plan["n_species"] >= 50
    assert int(plan["nnz"]) > 0 and float(plan["density"]) < 0.10
    # The CSC plan additions are present.
    assert "col_ptrs" in plan and "row_indices" in plan

    src = generate_jacobian_from_model(m)
    assert src is not None
    assert "bngsim_codegen_jac_sparse" in src
    # The dense entry point must NOT be emitted for a sparse-routed model.
    assert "int bngsim_codegen_jac(" not in src


@needs_cc
@needs_klu
def test_small_model_still_emits_dense(tmp_path):
    # A small functional model stays below the KLU gate → dense symbol, never the
    # sparse one. Confirms the existing dense path is untouched.
    m = bngsim.Model.from_sbml_string(_TINY_SBML)
    m.prepare_analytical_jacobian()
    src = generate_jacobian_from_model(m)
    assert src is not None
    assert "int bngsim_codegen_jac(" in src
    assert "bngsim_codegen_jac_sparse" not in src


# ── Values vs interpreted oracle + finite differences ────────────────────────


@needs_cc
@needs_klu
@pytest.mark.parametrize("n_chain", [55, 120, 300])
def test_compiled_sparse_matches_interpreted_and_fd(tmp_path, n_chain):
    m = _prepared_sparse_model(tmp_path, n_chain)
    core = m._core
    plan = core.codegen_jacobian_plan()
    assert "bngsim_codegen_jac_sparse" in (generate_jacobian_from_model(m) or "")
    so = prepare_model_codegen(m)
    assert so is not None and so.exists()

    col_ptrs = np.asarray(plan["col_ptrs"])
    row_indices = np.asarray(plan["row_indices"])
    ns = core.n_species

    rng = np.random.default_rng(n_chain)
    worst_interp = 0.0
    worst_fd = 0.0
    for _ in range(3):
        conc = rng.uniform(0.2, 8.0, ns)
        vals = _compiled_sparse_vals(core, so, plan, 0.0, conc)
        j_comp = _scatter_csc_to_dense(vals, col_ptrs, row_indices, ns)

        j_interp = _interpreted_dense(core, 0.0, conc)
        denom = np.maximum(np.maximum(np.abs(j_comp), np.abs(j_interp)), 1e-9)
        worst_interp = max(worst_interp, float((np.abs(j_comp - j_interp) / denom).max()))

        j_fd = _fd_dense(core, 0.0, conc)
        denom_fd = np.maximum(np.maximum(np.abs(j_comp), np.abs(j_fd)), 1e-3)
        worst_fd = max(worst_fd, float((np.abs(j_comp - j_fd) / denom_fd).max()))

    assert worst_interp < 1e-12, f"compiled-sparse vs interpreted worst rel err {worst_interp:g}"
    assert worst_fd < 1e-4, f"compiled-sparse vs finite-diff worst rel err {worst_fd:g}"


@needs_cc
@needs_klu
def test_fixed_species_row_is_zeroed(tmp_path):
    # The compiled sparse Jacobian must zero the fixed ($Sink) species' row across
    # every column, mirroring fill_sparse_analytical_jacobian.
    m = _prepared_sparse_model(tmp_path, 70)
    core = m._core
    plan = core.codegen_jacobian_plan()
    assert plan["fixed_rows"], "fixture must have a fixed species"
    so = prepare_model_codegen(m)
    ns = core.n_species
    col_ptrs = np.asarray(plan["col_ptrs"])
    row_indices = np.asarray(plan["row_indices"])
    conc = np.random.default_rng(7).uniform(0.5, 6.0, ns)
    vals = _compiled_sparse_vals(core, so, plan, 0.0, conc)
    j_comp = _scatter_csc_to_dense(vals, col_ptrs, row_indices, ns)
    for row in plan["fixed_rows"]:
        assert np.allclose(j_comp[int(row), :], 0.0), f"fixed row {row} not zeroed"


# ── End-to-end trajectory parity (KLU solve) ─────────────────────────────────


@needs_cc
@needs_klu
def test_end_to_end_sparse_jac_matches_interpreted_trajectory(tmp_path):
    # Full KLU ODE integration with the compiled sparse Jacobian vs the interpreted
    # one (the sparse KLU dispatch prefers the compiled symbol when codegen is
    # active). Both are analytical, so they agree to solver tolerance. The KLU
    # linear solve itself is identical — only the Jacobian *setup* path differs.
    net = str(_write_net(tmp_path, 90))

    def run(disable):
        if disable:
            os.environ["BNGSIM_NO_CODEGEN_JAC"] = "1"
        else:
            os.environ.pop("BNGSIM_NO_CODEGEN_JAC", None)
        m = bngsim.Model.from_net(net)
        m.prepare_analytical_jacobian()
        so = prepare_model_codegen(m)
        assert so is not None
        m._codegen_so_path = str(so)
        sim = bngsim.Simulator(m, method="ode", jacobian="analytical")
        return np.asarray(sim.run((0.0, 40.0), 81).species, dtype=float)

    try:
        compiled = run(False)
        interp = run(True)
    finally:
        os.environ.pop("BNGSIM_NO_CODEGEN_JAC", None)
    peak = np.maximum(np.abs(interp).max(axis=0), 1.0)
    assert float((np.abs(compiled - interp) / peak).max()) < 1e-9


# ── Determinism ──────────────────────────────────────────────────────────────


_CHILD = textwrap.dedent(
    """
    import os, sys, hashlib
    os.environ["BNGSIM_NO_CODEGEN"] = "1"
    import bngsim
    from bngsim import _codegen
    m = bngsim.Model.from_net(sys.argv[1])
    m.prepare_analytical_jacobian()
    src = _codegen.generate_jacobian_from_model(m)
    assert src and "bngsim_codegen_jac_sparse" in src, "expected sparse jac"
    sys.stdout.write(hashlib.sha256(src.encode()).hexdigest())
    """
)


@needs_cc
@needs_klu
def test_sparse_emit_is_pythonhashseed_independent(tmp_path):
    net = str(_write_net(tmp_path, 80))

    def _hash_with_seed(seed: int) -> str:
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, net],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, f"child failed (seed={seed}):\n{proc.stderr}"
        out = proc.stdout.strip()
        assert len(out) == 64, f"unexpected child output (seed={seed}): {proc.stdout!r}"
        return out

    hashes = {seed: _hash_with_seed(seed) for seed in (0, 1, 2, 3)}
    assert len(set(hashes.values())) == 1, f"sparse emit varies with PYTHONHASHSEED: {hashes}"


# ── .net codegen path carries the compiled Jacobian (GH #162) ────────────────


def _so_has_symbol(so_path, name: str) -> bool:
    lib = ctypes.CDLL(str(so_path))
    try:
        getattr(lib, name)
        return True
    except AttributeError:
        return False


@needs_cc
@needs_klu
def test_net_codegen_path_appends_sparse_jac(tmp_path):
    # The .net codegen entry point (prepare_codegen), given the built model,
    # appends the compiled sparse Jacobian onto the .net RHS in one .so — so a
    # .net-loaded large sparse model gets a compiled per-step Jacobian, not the
    # interpreted fallback. The RHS and the Jacobian must coexist correctly.
    m = _prepared_sparse_model(tmp_path, 70)
    core = m._core
    net = str(tmp_path / "sparse_chain_70.net")

    so = prepare_codegen(net, m)
    assert _so_has_symbol(so, "bngsim_codegen_rhs")
    assert _so_has_symbol(so, "bngsim_codegen_jac_sparse")
    assert not _so_has_symbol(so, "bngsim_codegen_jac")  # sparse-routed → no dense

    # Same .net WITHOUT a model stays RHS-only (historical behavior) and gets a
    # DISTINCT cache key, so it never collides with the Jacobian-carrying .so.
    so_rhs_only = prepare_codegen(net)
    assert not _so_has_symbol(so_rhs_only, "bngsim_codegen_jac_sparse")
    assert so_rhs_only != so

    # The appended sparse Jacobian still matches the interpreted oracle.
    plan = core.codegen_jacobian_plan()
    col_ptrs = np.asarray(plan["col_ptrs"])
    row_indices = np.asarray(plan["row_indices"])
    ns = core.n_species
    conc = np.random.default_rng(11).uniform(0.3, 7.0, ns)
    vals = _compiled_sparse_vals(core, so, plan, 0.0, conc)
    j_comp = _scatter_csc_to_dense(vals, col_ptrs, row_indices, ns)
    j_interp = _interpreted_dense(core, 0.0, conc)
    denom = np.maximum(np.maximum(np.abs(j_comp), np.abs(j_interp)), 1e-9)
    assert float((np.abs(j_comp - j_interp) / denom).max()) < 1e-12


@needs_cc
@needs_klu
def test_net_codegen_true_end_to_end_uses_compiled_sparse_jac(tmp_path):
    # The genome-scale workflow: load from .net, Simulator(codegen=True) — no
    # manual prepare / _codegen_so_path. The compiled sparse Jacobian must drive a
    # KLU solve that matches the interpreted-Jacobian trajectory to tolerance.
    net = str(_write_net(tmp_path, 90))

    def run(disable):
        if disable:
            os.environ["BNGSIM_NO_CODEGEN_JAC"] = "1"
        else:
            os.environ.pop("BNGSIM_NO_CODEGEN_JAC", None)
        m = bngsim.Model.from_net(net)
        sim = bngsim.Simulator(m, method="ode", jacobian="analytical", codegen=True)
        return np.asarray(sim.run((0.0, 40.0), 81).species, dtype=float)

    try:
        compiled = run(False)
        interp = run(True)
    finally:
        os.environ.pop("BNGSIM_NO_CODEGEN_JAC", None)
    peak = np.maximum(np.abs(interp).max(axis=0), 1.0)
    assert float((np.abs(compiled - interp) / peak).max()) < 1e-9


@needs_cc
@needs_klu
def test_net_codegen_fd_jacobian_appends_nothing(tmp_path):
    # jacobian="fd" must not append the analytical Jacobian to the .net .so (the
    # solver uses colored FD; the analytical terms are never derived). Confirms the
    # Simulator gate passes model=None for non-analytical strategies.
    # Writes sparse_chain_70.net into tmp_path; the model object itself is unused here.
    _prepared_sparse_model(tmp_path, 70)
    net = str(tmp_path / "sparse_chain_70.net")
    # Mirror the Simulator gate: fd → no model passed.
    so = prepare_codegen(net, None)
    assert _so_has_symbol(so, "bngsim_codegen_rhs")
    assert not _so_has_symbol(so, "bngsim_codegen_jac_sparse")
    assert not _so_has_symbol(so, "bngsim_codegen_jac")


@needs_cc
@needs_klu
def test_sparse_emit_is_stable_within_process(tmp_path):
    # The lazy CSC lookup + numpy fixed-row scatter must produce byte-identical
    # source on repeated emits in one process (cache-key stability).
    m = _prepared_sparse_model(tmp_path, 70)
    a = generate_jacobian_from_model(m)
    b = generate_jacobian_from_model(m)
    assert a == b
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()


def _build_so_direct(c_source: str, tmp_path, tag: str) -> str:
    """Compile a generated combined source straight to a .so (bypassing the cache)."""
    from bngsim import _codegen as cg

    c_path = tmp_path / f"{tag}.c"
    so_path = tmp_path / f"{tag}{cg._shared_lib_suffix()}"
    c_path.write_text(c_source)
    res = subprocess.run(
        cg._build_compile_cmd(c_path, so_path, "-O1"), capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, f"compile failed for {tag}:\n{res.stderr}"
    return str(so_path)


@needs_cc
@needs_klu
def test_chunked_sparse_jacobian_bit_identical(tmp_path, monkeypatch):
    """The chunked sparse Jacobian (GH #165) is bit-identical to the flat one.

    Past the chunk gate the per-contribution CSC scatter is lifted into NOINLINE
    ``jac_sparse_blk_*`` shard blocks (so the parallel shard compile splits it off
    the serial driver), but the blocks are called in the same fill order — every
    ``jac_data[csc]`` accumulation is unchanged, so the filled CSC value array is
    identical to the flat emit's down to the bit."""
    from bngsim import _codegen as cg

    m = _prepared_sparse_model(tmp_path, 60)
    core = m._core
    plan = core.codegen_jacobian_plan()

    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
    flat, _ = cg.generate_combined_from_model(m)
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "3")
    chunked, _ = cg.generate_combined_from_model(m)

    assert "bngsim_codegen_jac_sparse" in flat
    assert "jac_sparse_blk_" not in flat  # flat Jacobian below the gate
    assert "jac_sparse_blk_" in chunked  # CSC scatter sharded above it

    so_flat = _build_so_direct(flat, tmp_path, "sparse_jac_flat")
    so_chunk = _build_so_direct(chunked, tmp_path, "sparse_jac_chunk")

    rng = np.random.default_rng(2025)
    for _ in range(5):
        conc = rng.uniform(0.2, 8.0, core.n_species)
        vf = _compiled_sparse_vals(core, so_flat, plan, 0.41, conc)
        vc = _compiled_sparse_vals(core, so_chunk, plan, 0.41, conc)
        assert np.array_equal(vf, vc), "chunked sparse Jacobian not bit-identical to flat"
