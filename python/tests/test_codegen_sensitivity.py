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


def _get_nested_derived_net() -> str:
    p = DATA_DIR / "nested_derived_rate_const.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_ic_direct_net() -> str:
    p = DATA_DIR / "ic_direct.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_ic_derived_net() -> str:
    p = DATA_DIR / "ic_derived.net"
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


class TestNestedDerivedChainRule:
    """Issue #41 regression: a NESTED derived rate constant — a
    ConstantExpression that references ANOTHER ConstantExpression — must
    propagate the chain rule down to the underlying primary. Mirrors the igf1r
    detailed-balance structure ``a1prime = kcr``, ``a2prime = 3*a1prime`` where
    the fitted primary ``kcr`` reaches the rate laws only through the derived
    parameters. Pre-#41 the single-level ``kcr -> a1prime`` term survived but the
    nested ``kcr -> a1prime -> a2prime`` term was silently dropped, so the
    analytic sensitivity came back too small.
    """

    @staticmethod
    def _clear_cache(net):
        import platform

        from bngsim._codegen import CACHE_DIR, compute_model_hash

        h = compute_model_hash(net)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        for prefix in ("rhs_", "sens_"):
            cached = CACHE_DIR / f"{prefix}{h}{suffix}"
            if cached.exists():
                cached.unlink()

    def test_dfdp_case_includes_nested_reaction(self):
        """The generated ``bngsim_dfdp`` case for ``kcr`` must carry the
        contribution from the a2prime reaction (species C), not only the
        a1prime reaction — that species-C term is exactly what was dropped."""
        import re

        from bngsim._codegen import _parse_net_file, generate_sens_rhs_c

        net = _get_nested_derived_net()
        code = generate_sens_rhs_c(net)
        assert code is not None

        m = _parse_net_file(net)
        idxs = {name: i for i, (_, name, _, _) in enumerate(m["parameters"])}
        kcr_idx = idxs["kcr"]

        case_match = re.search(rf"    case {kcr_idx}:.*?break;", code, re.DOTALL)
        assert case_match, "kcr case missing from generated dfdp switch"
        snippet = case_match.group(0)
        # Species C is index 2 (0-based); it is produced only by the a2prime
        # reaction, so its dfdp entry appears iff the nested chain was followed.
        assert "dfdp_out[2]" in snippet, (
            f"nested chain kcr -> a1prime -> a2prime dropped from dfdp; got:\n{snippet}"
        )

    def test_nested_chain_rule_matches_fd(self):
        """Codegen analytic sens for ``kcr`` must match finite difference,
        including species C which depends on kcr only through the nested
        a2prime = 3*a1prime = 3*kcr path."""
        import bngsim

        net = _get_nested_derived_net()
        self._clear_cache(net)

        sample_times = list(np.linspace(0.0, 5.0, 51))
        nominal = 0.33  # kcr in the fixture

        def _traj(kcr_val):
            mod = bngsim.Model.from_net(net)
            mod.set_param("kcr", kcr_val)
            sim = bngsim.Simulator(mod, method="ode")
            r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
            return r.species

        eps = 1e-6
        fd = (_traj(nominal + eps) - _traj(nominal - eps)) / (2.0 * eps)

        mod = bngsim.Model.from_net(net)
        sim = bngsim.Simulator(
            mod, method="ode", sensitivity_params=["kcr"], codegen=True, net_path=net
        )
        r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
        sx = r.sensitivities[:, :, 0]

        denom = np.maximum(np.abs(fd[1:]), np.abs(sx[1:]))
        mask = denom > 1e-9
        assert mask.any(), "FD reference is identically zero — bad test setup"
        rel = np.abs(fd[1:] - sx[1:])[mask] / denom[mask]
        assert rel.max() < 1e-3, (
            f"codegen sens for kcr does not match FD (max relerr={rel.max():.3e}); "
            f"nested chain kcr -> a1prime -> a2prime likely dropped (issue #41)"
        )

        # Species C (index 2) is driven ONLY by the nested a2prime chain, so its
        # sensitivity is the direct witness that the fix took effect: it must be
        # substantially non-zero and agree with FD.
        c_analytic = np.abs(sx[:, 2]).max()
        assert c_analytic > 1e-3, (
            "species-C sensitivity is ~0: the nested a2prime chain was dropped"
        )
        np.testing.assert_allclose(sx[:, 2], fd[:, 2], rtol=1e-3, atol=1e-6)


class TestDerivedICParamSens:
    """Issue #43 regression: a species initial condition set by a DERIVED
    (ConstantExpression) parameter must seed the forward-sensitivity initial
    condition ∂x_i(0)/∂primary for the underlying fitted primary.

    ``ic_derived.net`` sets ``R() Rtot`` with ``Rtot = R0`` and a fitted primary
    ``R0`` that appears in NO rate law — its only influence on the trajectory is
    the initial condition of R. Pre-#43 the derived-parameter IC dropped the seed
    (the C++ seeding matched only the EXACT named parameter and hard-coded the
    coefficient to 1), so the analytic ∂R/∂R0 came back identically 0 instead of
    the rebuild-FD value 1. The direct-IC baseline (``ic_direct.net``, ``R() R0``)
    already seeded correctly and anchors the comparison.

    The finite-difference reference REBUILDS the model from a perturbed .net (a
    fresh ``Model.from_net``), NOT ``set_param``: ``set_param`` re-evaluates
    derived parameters but does not re-resolve baked species initial
    concentrations (issue #43 design question 1), so a set_param-based FD would
    silently miss the very IC dependence under test.
    """

    @staticmethod
    def _clear_cache(net):
        import platform

        from bngsim._codegen import CACHE_DIR, compute_model_hash

        h = compute_model_hash(net)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        for prefix in ("rhs_", "sens_"):
            cached = CACHE_DIR / f"{prefix}{h}{suffix}"
            if cached.exists():
                cached.unlink()

    _SAMPLE_TIMES = list(np.linspace(0.0, 3.0, 31))

    @classmethod
    def _rebuild_fd(cls, net, tmp_path, r0_lo=99.99, r0_hi=100.01):
        """Central FD of the trajectory w.r.t. R0, taken by REBUILDING the model
        from a perturbed .net (see class docstring)."""
        import re

        import bngsim

        src = Path(net).read_text()

        def _traj(r0):
            # Replace the numeric value on the ``R0`` parameter line only; the
            # first (and only) name-then-number match is that declaration.
            txt = re.sub(r"(\bR0\s+)[0-9.]+", rf"\g<1>{r0}", src, count=1)
            p = tmp_path / f"ic_{r0}.net"
            p.write_text(txt)
            m = bngsim.Model.from_net(str(p))
            r = bngsim.Simulator(m, method="ode").run(
                sample_times=cls._SAMPLE_TIMES, rtol=1e-11, atol=1e-13, max_steps=10**6
            )
            return np.asarray(r.species)

        return (_traj(r0_hi) - _traj(r0_lo)) / (r0_hi - r0_lo)

    @classmethod
    def _analytic(cls, net):
        import bngsim

        m = bngsim.Model.from_net(net)
        r = bngsim.Simulator(
            m, method="ode", sensitivity_params=["R0"], codegen=True, net_path=net
        ).run(sample_times=cls._SAMPLE_TIMES, rtol=1e-11, atol=1e-13, max_steps=10**6)
        return np.asarray(r.sensitivities)[:, :, 0]

    def test_seed_helper_coefficients(self):
        """compute_ic_param_sens_seed maps each parameter-referenced species IC
        to (species_idx0, PRIMARY_idx0, coeff): coefficient 1 for the direct IC
        and — the fix — for the derived ``Rtot = R0``, keyed on R0 (the primary),
        never on the derived Rtot index."""
        import bngsim
        from bngsim._codegen import compute_ic_param_sens_seed

        m_direct = bngsim.Model.from_net(_get_ic_direct_net())
        names_d = list(m_direct._core.param_names)
        seeds_d = compute_ic_param_sens_seed(m_direct._core)
        assert (0, names_d.index("R0"), 1.0) in seeds_d  # species R (idx 0) ← R0

        m_der = bngsim.Model.from_net(_get_ic_derived_net())
        names = list(m_der._core.param_names)
        seeds = compute_ic_param_sens_seed(m_der._core)
        assert (0, names.index("R0"), 1.0) in seeds, (
            "derived IC Rtot = R0 must seed R0 with coefficient 1 (issue #43)"
        )
        rtot_idx = names.index("Rtot")
        assert all(prim != rtot_idx for _, prim, _ in seeds), (
            "seed must key on the primary R0, never on the derived Rtot index"
        )

    def test_direct_ic_matches_rebuild_fd(self, tmp_path):
        net = _get_ic_direct_net()
        self._clear_cache(net)
        fd = self._rebuild_fd(net, tmp_path)
        sx = self._analytic(net)
        assert abs(sx[0, 0] - 1.0) < 1e-6, "∂R(0)/∂R0 must seed to 1 for a direct IC"
        np.testing.assert_allclose(sx[:, 0], fd[:, 0], rtol=1e-4, atol=1e-6)

    def test_derived_ic_matches_rebuild_fd(self, tmp_path):
        """The core #43 regression: derived-parameter IC must seed ∂R/∂R0."""
        net = _get_ic_derived_net()
        self._clear_cache(net)
        fd = self._rebuild_fd(net, tmp_path)
        sx = self._analytic(net)
        assert np.abs(sx[:, 0]).max() > 1e-3, (
            "derived-parameter IC seed dropped: ∂R/∂R0 is ~0 (issue #43)"
        )
        assert abs(sx[0, 0] - 1.0) < 1e-6, "∂R(0)/∂R0 must seed to 1 through Rtot = R0"
        np.testing.assert_allclose(sx[:, 0], fd[:, 0], rtol=1e-4, atol=1e-6)

    def test_direct_and_derived_ic_agree(self):
        """The two fixtures are identical up to the derived indirection, so their
        R0-sensitivity trajectories must coincide."""
        direct = self._analytic(_get_ic_direct_net())
        derived = self._analytic(_get_ic_derived_net())
        np.testing.assert_allclose(direct[:, 0], derived[:, 0], rtol=1e-6, atol=1e-9)
