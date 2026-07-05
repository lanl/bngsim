"""Tests for codegen sensitivity RHS (Session 27).

Verifies that the code-generated analytical sensitivity RHS produces
identical results to CVODES internal FD sensitivity.
"""

import os
from pathlib import Path

import numpy as np
from bngsim._codegen import (
    generate_combined_c,
    generate_sens_rhs_c,
)

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _get_simple_decay_net() -> str:
    p = DATA_DIR / "simple_decay.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_reversible_net() -> str:
    p = DATA_DIR / "two_species_reversible.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_derived_rate_const_net() -> str:
    p = DATA_DIR / "derived_rate_const.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_derived_quotient_net() -> str:
    p = DATA_DIR / "derived_quotient.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


class TestSensRhsCodeGeneration:
    """Test the C code generation for sensitivity RHS."""

    def test_elementary_model_generates_code(self):
        """Simple decay is all-Elementary → should produce sens RHS code."""
        code = generate_sens_rhs_c(_get_simple_decay_net())
        assert code is not None
        assert "bngsim_codegen_sens_rhs" in code
        assert "bngsim_dfdp" in code
        assert "bngsim_jac_vec" in code

    def test_reversible_model_generates_code(self):
        """Reversible model is all-Elementary → should produce code."""
        code = generate_sens_rhs_c(_get_reversible_net())
        assert code is not None
        assert "bngsim_codegen_sens_rhs" in code

    def test_combined_generation(self):
        """generate_combined_c should produce both RHS + sens RHS."""
        combined, has_sens = generate_combined_c(_get_simple_decay_net())
        assert has_sens is True
        assert "bngsim_codegen_rhs" in combined
        assert "bngsim_codegen_sens_rhs" in combined

    def test_dfdp_switch_cases(self):
        """Generated code should have switch cases for rate param indices."""
        code = generate_sens_rhs_c(_get_simple_decay_net())
        assert "switch (iP)" in code
        assert "case " in code

    def test_jac_vec_reactions(self):
        """Generated Jacobian-vector product should reference reactions."""
        code = generate_sens_rhs_c(_get_reversible_net())
        assert "Reaction" in code
        assert "dv_dxj" in code


class TestSensRhsCompilation:
    """Test that the generated sensitivity C code compiles."""

    def test_combined_compiles(self):
        """Combined RHS + sens RHS should compile to .so."""
        from bngsim._codegen import compile_rhs

        net_path = _get_simple_decay_net()
        combined, has_sens = generate_combined_c(net_path)
        assert has_sens

        # Use a unique hash to avoid cache collisions
        import hashlib

        test_hash = "test_sens_" + hashlib.sha256(combined.encode()).hexdigest()[:8]

        so_path = compile_rhs(combined, test_hash)
        assert so_path.exists()

        # Verify both symbols are present via dlopen
        import ctypes

        lib = ctypes.CDLL(str(so_path))
        assert hasattr(lib, "bngsim_codegen_rhs")
        assert hasattr(lib, "bngsim_codegen_sens_rhs")


class TestSensRhsCorrectness:
    """Test that codegen sens RHS matches CVODES internal FD sensitivity."""

    def test_simple_decay_matches_fd(self):
        """Codegen sensitivity should match CVODES FD for simple_decay."""
        import bngsim
        from bngsim._codegen import prepare_codegen

        net_path = _get_simple_decay_net()

        # Clear cache for this model to force re-generation
        import platform

        from bngsim._codegen import CACHE_DIR, compute_model_hash

        model_hash = compute_model_hash(net_path)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        cached = CACHE_DIR / f"rhs_{model_hash}{suffix}"
        if cached.exists():
            cached.unlink()

        # Run with CVODES internal FD (no codegen)
        m1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(m1, method="ode", sensitivity_params=["k1"])
        r1 = sim1.run(t_span=(0, 10), n_points=101)

        # Run with codegen (includes sens RHS for Elementary models)
        str(prepare_codegen(net_path))
        m2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(
            m2, method="ode", sensitivity_params=["k1"], codegen=True, net_path=net_path
        )
        r2 = sim2.run(t_span=(0, 10), n_points=101)

        # Species should match
        np.testing.assert_allclose(r1.species, r2.species, atol=1e-8)

        # Sensitivities should match (both are accurate)
        np.testing.assert_allclose(
            r1.sensitivities,
            r2.sensitivities,
            atol=1e-4,
            rtol=1e-3,
            err_msg="Codegen sens != CVODES FD sens",
        )

    def test_reversible_two_params(self):
        """Two-param sensitivity with codegen should match FD."""
        import platform

        import bngsim
        from bngsim._codegen import CACHE_DIR, compute_model_hash

        net_path = _get_reversible_net()

        # Clear cache
        model_hash = compute_model_hash(net_path)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        cached = CACHE_DIR / f"rhs_{model_hash}{suffix}"
        if cached.exists():
            cached.unlink()

        # FD baseline
        m1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(m1, method="ode", sensitivity_params=["kf", "kr"])
        r1 = sim1.run(t_span=(0, 10), n_points=51)

        # Codegen
        m2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(
            m2, method="ode", sensitivity_params=["kf", "kr"], codegen=True, net_path=net_path
        )
        r2 = sim2.run(t_span=(0, 10), n_points=51)

        # Sensitivities should be close
        np.testing.assert_allclose(
            r1.sensitivities,
            r2.sensitivities,
            atol=1e-4,
            rtol=1e-3,
            err_msg="Codegen sens != CVODES FD sens (reversible)",
        )


class TestDerivedRateConstantSens:
    """Issue #2 regression: ConstantExpression rate-constant parameters
    (BNG2.pl ``_rateLaw{N} = chi*kon`` style) must propagate the chain rule
    so sensitivity for the underlying primary parameter (``kon``) is correct.
    """

    @staticmethod
    def _fd_sens_kon(net_path, eps=1e-5):
        """Reference: 2-pt centered finite difference of trajectories."""
        import bngsim

        nominal = 1.0  # value for kon in derived_rate_const.net
        sample_times = list(np.linspace(0.0, 5.0, 51))

        def _traj(kon_val):
            m = bngsim.Model.from_net(net_path)
            m.set_param("kon", kon_val)
            sim = bngsim.Simulator(m, method="ode")
            r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
            return r.species

        dp = nominal * eps
        return (_traj(nominal + eps) - _traj(nominal - eps)) / (2.0 * dp)

    @staticmethod
    def _bngsim_sens_kon(net_path, codegen):
        import bngsim

        sample_times = list(np.linspace(0.0, 5.0, 51))
        m = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(
            m,
            method="ode",
            sensitivity_params=["kon"],
            codegen=codegen,
            net_path=(net_path if codegen else ""),
        )
        r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
        return r.sensitivities[:, :, 0]

    def test_codegen_chain_rule_matches_fd(self):
        """Codegen sens for ``kon`` must include chain rule via ``_rateLaw1 = chi*kon``."""
        import platform

        from bngsim._codegen import CACHE_DIR, compute_model_hash

        net = _get_derived_rate_const_net()
        # Clear cache so the (possibly updated) codegen runs.
        h = compute_model_hash(net)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        cached = CACHE_DIR / f"rhs_{h}{suffix}"
        if cached.exists():
            cached.unlink()

        fd = self._fd_sens_kon(net)
        sx = self._bngsim_sens_kon(net, codegen=True)

        # Drop t=0 (sx is 0 trivially) and skip near-zero entries to avoid
        # tiny denominators dominating relative error.
        denom = np.maximum(np.abs(fd[1:]), np.abs(sx[1:]))
        mask = denom > 1e-9
        rel = np.abs(fd[1:] - sx[1:])[mask] / denom[mask]
        assert mask.any(), "FD reference is identically zero — bad test setup"
        assert rel.max() < 1e-3, (
            f"codegen sens for kon does not match FD (max relerr={rel.max():.3e}); "
            f"chain rule through _rateLaw1 likely dropped"
        )
        # Sign agreement is the diagnostic that originally surfaced issue #2.
        assert np.all(np.sign(fd[1:][mask]) == np.sign(sx[1:][mask])), (
            "codegen sens for kon has wrong sign (issue #2 regression)"
        )

    # (Removed test_cvode_fd_chain_rule_matches_fd: it exercised the interpreted
    # CVODES-internal-FD sensitivity path with codegen=False, retired in GH #214.
    # Chain-rule correctness is covered by test_codegen_chain_rule_matches_fd.)


class TestDerivedQuotientChainRule:
    """Regression: non-product derived rate constants (e.g., ``m1 = 5/MEK``
    as in tcr_signaling). Generalized chain rule via sympy must emit the
    ``-5/pow(MEK, 2)`` contribution to ``dfdp[*][MEK]``.
    """

    def test_dfdp_emits_quotient_partial(self):
        """Generated dfdp must reference ``-5/pow(p[idx_MEK], 2)``-style
        partial in the case branch for MEK."""
        from bngsim._codegen import _parse_net_file, generate_sens_rhs_c

        net = _get_derived_quotient_net()
        code = generate_sens_rhs_c(net)
        assert code is not None

        m = _parse_net_file(net)
        idxs = {name: i for i, (_, name, _, _) in enumerate(m["parameters"])}
        mek_idx = idxs["MEK"]

        import re

        case_match = re.search(rf"    case {mek_idx}:.*?break;", code, re.DOTALL)
        assert case_match, "MEK case missing from generated dfdp switch"
        snippet = case_match.group(0)
        assert f"pow(p[{mek_idx}], 2)" in snippet, (
            f"chain rule -5/pow(p[{mek_idx}], 2) not emitted; got:\n{snippet}"
        )

    def test_quotient_chain_rule_matches_fd(self):
        """Codegen sens for ``MEK`` must include the ``∂(5/MEK)/∂MEK`` chain rule."""
        import platform

        import bngsim
        from bngsim._codegen import CACHE_DIR, compute_model_hash

        net = _get_derived_quotient_net()
        h = compute_model_hash(net)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        cached = CACHE_DIR / f"rhs_{h}{suffix}"
        if cached.exists():
            cached.unlink()

        sample_times = list(np.linspace(0.0, 5.0, 51))
        nominal = 2.0  # MEK in the fixture

        def _traj(mek_val):
            mod = bngsim.Model.from_net(net)
            mod.set_param("MEK", mek_val)
            sim = bngsim.Simulator(mod, method="ode")
            r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
            return r.species

        eps = 1e-5
        fd = (_traj(nominal + eps) - _traj(nominal - eps)) / (2.0 * eps)

        mod = bngsim.Model.from_net(net)
        sim = bngsim.Simulator(
            mod,
            method="ode",
            sensitivity_params=["MEK"],
            codegen=True,
            net_path=net,
        )
        r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
        sx = r.sensitivities[:, :, 0]

        denom = np.maximum(np.abs(fd[1:]), np.abs(sx[1:]))
        mask = denom > 1e-9
        assert mask.any()
        rel = np.abs(fd[1:] - sx[1:])[mask] / denom[mask]
        assert rel.max() < 1e-3, (
            f"codegen sens for MEK does not match FD (max relerr={rel.max():.3e}); "
            f"chain rule through m1 = 5/MEK likely dropped"
        )
