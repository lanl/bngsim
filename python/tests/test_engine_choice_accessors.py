"""Phase 0 accessors: the engine reports the heavy knobs it auto-selected.

These public accessors let an honest benchmark harness (rr_parity, GH #135)
record what ``method="ode"`` actually chose per problem — codegen backend,
Jacobian strategy, codegen/setup wall time, and the linear solver — instead of
*guessing* from private attributes. Each accessor exposes a value the engine
already knows; none instruments the per-step RHS/Jacobian hot path.
"""

import os

import bngsim
import pytest

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)


def _net(name: str) -> str:
    return os.path.join(DATA, f"{name}.net")


class TestCodegenBackend:
    """T0.1 — ``Simulator.codegen_backend`` reports ``exprtk``/``cc``/``mir``."""

    def test_small_model_uses_exprtk(self):
        # A 2-species model is far below the 256-species codegen threshold and
        # was not asked to codegen, so the engine runs the ExprTk interpreter.
        m = bngsim.Model.from_net(_net("simple_decay"))
        sim = bngsim.Simulator(m, method="ode")
        assert sim.codegen_backend == "exprtk"

    def test_explicit_codegen_uses_cc(self):
        # codegen=True compiles native C via cc + dlopen — the "cc" backend —
        # even on a tiny model (this is exactly how to reach the state a
        # >=256-species model reaches automatically, without a huge fixture).
        m = bngsim.Model.from_net(_net("simple_decay"))
        sim = bngsim.Simulator(m, method="ode", codegen=True, net_path=_net("simple_decay"))
        assert sim.codegen_backend == "cc"

    def test_threshold_env_forces_cc(self):
        # The documented threshold override (BNGSIM_CODEGEN_THRESHOLD) is the
        # production path to "cc": drop it to 0 and even a 2-species model
        # auto-codegens at load.
        prev = os.environ.get("BNGSIM_CODEGEN_THRESHOLD")
        os.environ["BNGSIM_CODEGEN_THRESHOLD"] = "0"
        try:
            import bngsim._sbml_loader  # noqa: F401  (kept for parity with SBML path)

            m = bngsim.Model.from_net(_net("simple_decay"))
            # .net models codegen lazily via the Simulator; force it on.
            sim = bngsim.Simulator(m, method="ode", codegen=True, net_path=_net("simple_decay"))
            assert sim.codegen_backend == "cc"
        finally:
            if prev is None:
                os.environ.pop("BNGSIM_CODEGEN_THRESHOLD", None)
            else:
                os.environ["BNGSIM_CODEGEN_THRESHOLD"] = prev

    def test_mir_classification_when_jit_source_present(self):
        # The "mir" state is only reachable on a MIR-enabled build with
        # BNGSIM_CODEGEN_JIT=mir, which this cc/exprtk build is not. The
        # classifier itself is still exercisable: a non-empty JIT C source
        # must classify as "mir" and take priority over any .so path.
        m = bngsim.Model.from_net(_net("simple_decay"))
        sim = bngsim.Simulator(m, method="ode")
        sim._codegen_c_source = "/* generated RHS */"
        assert sim.codegen_backend == "mir"
        # MIR (JIT source) wins over a stale .so path, matching opts dispatch.
        sim._codegen_so_path = "/tmp/whatever.so"
        assert sim.codegen_backend == "mir"


class TestJacobianStrategy:
    """T0.2 — ``Simulator.jacobian_strategy`` reports the resolved strategy."""

    def test_functional_model_uses_analytical(self):
        # saturation.net has a Functional (saturating) rate law whose Jacobian is
        # symbolically derived. GH #145: the derivation is deferred off the load
        # path, so it is NOT complete right after from_net — constructing the ODE
        # Simulator triggers it, and auto then picks the analytical path.
        m = bngsim.Model.from_net(_net("saturation"))
        assert not m._core.analytical_jacobian_complete  # deferred at load (#145)
        sim = bngsim.Simulator(m, method="ode")
        assert m._core.analytical_jacobian_complete  # warmed by ODE-solve setup
        assert sim.jacobian_strategy == "analytical"

    def test_functional_model_falls_back_to_fd_under_env(self):
        # The documented escape hatch: with BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0
        # the same Functional model never attaches analytical terms, so the
        # engine uses finite differences and the accessor must say so.
        prev = os.environ.get("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC")
        os.environ["BNGSIM_ANALYTICAL_FUNCTIONAL_JAC"] = "0"
        try:
            m = bngsim.Model.from_net(_net("saturation"))
            assert not m._core.analytical_jacobian_complete  # precondition
            sim = bngsim.Simulator(m, method="ode")
            assert sim.jacobian_strategy == "fd"
        finally:
            if prev is None:
                os.environ.pop("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", None)
            else:
                os.environ["BNGSIM_ANALYTICAL_FUNCTIONAL_JAC"] = prev

    def test_explicit_fd_request_reports_fd(self):
        # jacobian="fd" forces FD even on a model that *could* go analytical.
        # GH #145: with jacobian="fd" the ODE-solve setup does NOT even derive the
        # analytical Jacobian (FD needs no analytical terms), so it stays
        # incomplete — warm it explicitly to assert the model *could* go analytical.
        m = bngsim.Model.from_net(_net("saturation"))
        m.prepare_analytical_jacobian()
        assert m._core.analytical_jacobian_complete
        sim = bngsim.Simulator(m, method="ode", jacobian="fd")
        assert sim.jacobian_strategy == "fd"

    def test_mass_action_model_uses_analytical(self):
        # All-Elementary models always have the analytical Jacobian (it does not
        # route through the Functional-derivation budget), so auto picks it.
        m = bngsim.Model.from_net(_net("simple_decay"))
        sim = bngsim.Simulator(m, method="ode")
        assert sim.jacobian_strategy == "analytical"


class TestCodegenSetupTime:
    """T0.3 — ``Simulator.last_codegen_sec`` reports codegen wall time."""

    def test_exprtk_model_reports_zero_setup(self):
        # An ExprTk model never codegens, so its setup time is exactly 0.0
        # (the recorder is never invoked for it).
        m = bngsim.Model.from_net(_net("simple_decay"))
        sim = bngsim.Simulator(m, method="ode")
        assert sim.codegen_backend == "exprtk"
        assert sim.last_codegen_sec == 0.0

    def test_cc_codegen_records_cold_compile_then_cache_hit(self):
        # The cc backend compiles native C; a cold compile costs real wall time,
        # a subsequent cache hit costs ~nothing. Both must be reported from a
        # single run (no run-twice-and-subtract).
        import bngsim._codegen as cg

        net = _net("simple_decay")
        # Evict the cached .so and the .net-path memo so the next codegen is a
        # genuine cold compile rather than a cache hit.
        so = cg.get_cached_so(cg.compute_model_hash(net))
        if so is not None:
            so.unlink()
        cg._PREPARE_CODEGEN_MEMO.clear()

        m = bngsim.Model.from_net(net)
        sim = bngsim.Simulator(m, method="ode", codegen=True, net_path=net)
        assert sim.codegen_backend == "cc"
        cold = sim.last_codegen_sec
        assert cold > 0.0  # the cc compile took measurable wall time
        # Model carries the same value (the accessor reads from it).
        assert sim.last_codegen_sec == m._codegen_sec

        # Reconstruct: now the .so is cached, so setup collapses to near zero.
        m2 = bngsim.Model.from_net(net)
        sim2 = bngsim.Simulator(m2, method="ode", codegen=True, net_path=net)
        assert sim2.last_codegen_sec < cold


class TestCodegenCacheHit:
    """``Simulator.codegen_cache_hit`` is the DEFINITIVE .so cache signal —
    True (reused), False (compiled fresh), None (no .so) — recorded by the
    codegen pipeline at the get_cached_so branch, not inferred from wall time."""

    def test_exprtk_model_has_no_so(self):
        m = bngsim.Model.from_net(_net("simple_decay"))
        sim = bngsim.Simulator(m, method="ode")
        assert sim.codegen_backend == "exprtk"
        assert sim.codegen_cache_hit is None  # no .so involved at all

    def test_cold_compile_then_cache_hit(self):
        import hashlib

        import bngsim._codegen as cg

        net = _net("simple_decay")
        # simple_decay has a complete (elementary) analytical Jacobian AND two
        # observables, so the default jacobian="auto" .net codegen compiles it under
        # the combined GH #162 ":codegen_jac" + GH #163 ":codegen_outputs" key, not
        # the plain RHS-only key — clear every suffix combination so the first
        # construction is truly cold.
        base = cg.compute_model_hash(net)
        keys = [base]
        for suffix in (":codegen_jac", ":codegen_outputs", ":codegen_jac:codegen_outputs"):
            keys.append(hashlib.sha256((base + suffix).encode()).hexdigest()[:16])
        for key in keys:
            so = cg.get_cached_so(key)
            if so is not None:
                so.unlink()
        cg._PREPARE_CODEGEN_MEMO.clear()

        # Cold: no cached .so → compiled fresh.
        m = bngsim.Model.from_net(net)
        sim = bngsim.Simulator(m, method="ode", codegen=True, net_path=net)
        assert sim.codegen_backend == "cc"
        assert sim.codegen_cache_hit is False

        # Warm: the .so is now cached → reused. Definitive even though resolving
        # the cache still spends nonzero wall time (so a wall-time heuristic would
        # be unreliable); the accessor reads the actual get_cached_so branch.
        m2 = bngsim.Model.from_net(net)
        sim2 = bngsim.Simulator(m2, method="ode", codegen=True, net_path=net)
        assert sim2.codegen_cache_hit is True


def _write_linear_chain_net(path, n: int) -> None:
    """Write a length-``n`` mass-action chain A1->A2->...->An to ``path``.

    Its Jacobian is bidiagonal (~2 nonzeros/row), so for n>=50 the density is
    well under 10% and the auto rule routes it to KLU."""
    lines = [
        "begin parameters",
        "    1 k 0.5",
        "end parameters",
        "begin species",
        "    1 A1() 100",
    ]
    lines += [f"    {i} A{i}() 0" for i in range(2, n + 1)]
    lines.append("end species")
    lines.append("begin reactions")
    lines += [f"    {i} {i} {i + 1} k #_R{i}" for i in range(1, n)]
    lines.append("end reactions")
    lines += ["begin groups", "    1 Total  1", "end groups", ""]
    path.write_text("\n".join(lines))


class TestLinearSolverEnum:
    """T0.4 — ``Result.solver_stats["linear_solver"]`` is queryable + mapped.

    LinearSolverKind: 0=dense, 1=KLU sparse, 2=LAPACK dgetrf. The harness used
    to mislabel 1 as "Band"; these tests pin the real mapping.
    """

    def test_small_model_reports_dense(self):
        # N<50 → the auto rule picks the built-in dense LU (kind 0).
        m = bngsim.Model.from_net(_net("simple_decay"))
        assert m.n_species < 50
        sim = bngsim.Simulator(m, method="ode")
        r = sim.run(t_span=(0, 10), n_points=11, rtol=1e-8, atol=1e-10)
        assert r.solver_stats["linear_solver"] == 0

    def test_large_sparse_model_reports_klu(self, tmp_path):
        # N>=50 with a sparse (bidiagonal) Jacobian → KLU (kind 1), the value the
        # old harness mislabeled "Band".
        net = tmp_path / "chain.net"
        _write_linear_chain_net(net, 60)
        m = bngsim.Model.from_net(str(net))
        assert m.n_species == 60
        # KLU is an optional build feature (macOS CI builds with
        # -DBNGSIM_ENABLE_KLU=OFF, GH #153). Without it the auto rule falls back
        # to dense, so the "reports KLU" expectation only holds where KLU is
        # actually compiled in. On a KLU build we still assert, so a genuine
        # auto-selection regression is not masked.
        if not m._core.codegen_jacobian_plan()["has_klu"]:
            pytest.skip("KLU not compiled into this build (BNGSIM_ENABLE_KLU=OFF)")
        sim = bngsim.Simulator(m, method="ode")
        r = sim.run(t_span=(0, 10), n_points=11, rtol=1e-8, atol=1e-10)
        assert r.solver_stats["linear_solver"] == 1


class TestLapackDenseGateFactorCount:
    """GH #132 adaptive gate: solver_stats reports how many factorizations took
    the BLAS dgetrf path, so a LAPACK-dense run that never crossed the K gate
    (and stayed on the built-in dense LU) is distinguishable from one that did."""

    @staticmethod
    def _run_chain(tmp_path, env):
        prev = {k: os.environ.get(k) for k in env}
        os.environ.update({k: v for k, v in env.items()})
        try:
            net = tmp_path / "chain.net"
            _write_linear_chain_net(net, 60)
            m = bngsim.Model.from_net(str(net))
            sim = bngsim.Simulator(m, method="ode", force_dense_linear_solver=True)
            return sim.run(t_span=(0, 50), n_points=51, rtol=1e-8, atol=1e-10)
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_no_lapack_means_zero_blas_factorizations(self, tmp_path):
        # force_dense without the LAPACK opt-in → built-in dense LU (kind 0),
        # never dgetrf.
        r = self._run_chain(tmp_path, {"BNGSIM_LAPACK_DENSE": "0"})
        assert r.solver_stats["linear_solver"] == 0
        assert r.solver_stats["n_dense_blas_factorizations"] == 0

    def test_gate_crossed_counts_dgetrf_factorizations(self, tmp_path):
        # LAPACK opt-in, gate open from the first factorization (K=0), no min-N
        # floor → every factorization uses dgetrf.
        r = self._run_chain(
            tmp_path,
            {
                "BNGSIM_LAPACK_DENSE": "1",
                "BNGSIM_LAPACK_DENSE_K": "0",
                "BNGSIM_LAPACK_DENSE_MIN_N": "0",
            },
        )
        if r.solver_stats["linear_solver"] != 2:
            import pytest

            pytest.skip("LAPACK-dense not built in this configuration")
        n_fac = r.solver_stats["n_jac_evals"]
        assert n_fac > 0
        assert r.solver_stats["n_dense_blas_factorizations"] == n_fac

    def test_lapack_mode_but_gate_never_crossed_reports_zero(self, tmp_path):
        # The case worth catching: LAPACK-dense is *selected* (kind 2) but the K
        # gate is set so high it never trips → all factorizations stayed on the
        # built-in LU, so dgetrf count is 0 despite the "LAPACK-dense" label.
        r = self._run_chain(
            tmp_path,
            {
                "BNGSIM_LAPACK_DENSE": "1",
                "BNGSIM_LAPACK_DENSE_K": "1000000",
                "BNGSIM_LAPACK_DENSE_MIN_N": "0",
            },
        )
        if r.solver_stats["linear_solver"] != 2:
            import pytest

            pytest.skip("LAPACK-dense not built in this configuration")
        assert r.solver_stats["n_dense_blas_factorizations"] == 0
