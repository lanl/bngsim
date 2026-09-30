"""Tests for codegen sensitivity RHS (Session 27).

Verifies that the code-generated analytical sensitivity RHS produces
identical results to CVODES internal FD sensitivity.
"""

import os
import tempfile
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._codegen import (
    generate_combined_from_model,
    generate_sens_from_model,
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


def _get_ic_derived_compound_net() -> str:
    p = DATA_DIR / "ic_derived_compound.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _sens_c(net: str):
    """The sensitivity RHS a .net model compiles to (from the built model, #803)."""
    return generate_sens_from_model(bngsim.Model.from_net(net))


def _combined_c(net: str):
    return generate_combined_from_model(bngsim.Model.from_net(net))


def _param_index(net: str, name: str) -> int:
    """The ``p[]`` slot, and ``bngsim_dfdp`` case label, of parameter *name*."""
    params = bngsim.Model.from_net(net)._core.codegen_data()["parameters"]
    return [q["name"] for q in params].index(name)


class TestSensRhsCodeGeneration:
    """Test the C code generation for sensitivity RHS."""

    def test_elementary_model_generates_code(self):
        """Simple decay is all-Elementary → should produce sens RHS code."""
        code = _sens_c(_get_simple_decay_net())
        assert code is not None
        assert "bngsim_codegen_sens_rhs" in code
        assert "bngsim_dfdp" in code
        assert "bngsim_jac_vec" in code

    def test_reversible_model_generates_code(self):
        """Reversible model is all-Elementary → should produce code."""
        code = _sens_c(_get_reversible_net())
        assert code is not None
        assert "bngsim_codegen_sens_rhs" in code

    def test_combined_generation(self):
        """The combined source should carry both RHS + sens RHS."""
        combined, has_sens = _combined_c(_get_simple_decay_net())
        assert has_sens is True
        assert "bngsim_codegen_rhs" in combined
        assert "bngsim_codegen_sens_rhs" in combined

    def test_dfdp_switch_cases(self):
        """Generated code should have switch cases for rate param indices."""
        code = _sens_c(_get_simple_decay_net())
        assert "switch (iP)" in code
        assert "case " in code

    def test_jac_vec_reactions(self):
        """Generated Jacobian-vector product should reference reactions."""
        code = _sens_c(_get_reversible_net())
        assert "Reaction" in code
        assert "dv_dxj" in code


class TestSensRhsCompilation:
    """Test that the generated sensitivity C code compiles."""

    def test_combined_compiles(self, tmp_path, monkeypatch):
        """Combined RHS + sens RHS should compile to .so."""
        import bngsim._codegen as cg

        net_path = _get_simple_decay_net()
        combined, has_sens = _combined_c(net_path)
        assert has_sens

        # A hash this test invents is a key no install will ever look up, so the
        # .so it produces is born orphaned — it must not be compiled into a cache
        # that outlives the test (#372). Four such `test_sens_*` artifacts were
        # still sitting in a developer's ~/.cache months after the run that wrote
        # them. Captured first so the assertion below can prove that stopped.
        session_cache = cg.CACHE_DIR
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)

        # Use a unique hash to avoid cache collisions
        import hashlib

        test_hash = "test_sens_" + hashlib.sha256(combined.encode()).hexdigest()[:8]

        so_path = cg.compile_rhs(combined, test_hash)
        assert so_path.exists()
        assert so_path.parent == tmp_path, f"{so_path} not under the test's own cache"
        assert not list(session_cache.glob("*test_sens_*")), "invented key leaked into the cache"

        # Verify both symbols are present via dlopen
        import ctypes

        lib = ctypes.CDLL(str(so_path))
        assert hasattr(lib, "bngsim_codegen_rhs")
        assert hasattr(lib, "bngsim_codegen_sens_rhs")


class TestSensRhsCorrectness:
    """The codegen sensitivity RHS against oracles that share none of its code: a
    closed form, and central finite differences of interpreted trajectories.

    Until #803 these compared against "CVODES internal FD", a ``Simulator`` built
    without ``codegen=True``. GH #214 had already retired that path — a
    sensitivity run always compiles — so both arms were the same compiled RHS.
    """

    def test_simple_decay_matches_the_closed_form(self):
        """A → B at k1 from A0 = 100: ``A = A0·e^{-k1·t}``, so
        ``∂A/∂k1 = -A0·t·e^{-k1·t} = -∂B/∂k1``."""
        m = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"], codegen=True)
        assert sim._codegen_so_path or sim._codegen_c_source
        r = sim.run(t_span=(0, 10), n_points=101, rtol=1e-10, atol=1e-12)

        t = np.asarray(r.time)
        a = 100.0 * np.exp(-0.1 * t)
        np.testing.assert_allclose(np.asarray(r.species)[:, 0], a, rtol=1e-8, atol=1e-10)
        s = np.asarray(r.sensitivities)[:, :, 0]
        np.testing.assert_allclose(s[:, 0], -t * a, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(s[:, 1], t * a, rtol=1e-6, atol=1e-8)

    def test_reversible_two_params(self):
        """A + B <-> C, both rate constants, against a central FD of the
        interpreted forward trajectory (no sensitivity machinery in the oracle)."""
        net_path = _get_reversible_net()
        span, n = (0.0, 10.0), 51

        m = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kf", "kr"], codegen=True)
        sens = np.asarray(sim.run(t_span=span, n_points=n, rtol=1e-10, atol=1e-12).sensitivities)

        def traj(name, value):
            mod = bngsim.Model.from_net(net_path)
            mod.set_param(name, value)
            r = bngsim.Simulator(mod, method="ode", codegen=False).run(
                t_span=span, n_points=n, rtol=1e-12, atol=1e-14
            )
            return np.asarray(r.species)

        for j, (name, value) in enumerate((("kf", 0.001), ("kr", 0.1))):
            h = value * 1e-4
            fd = (traj(name, value + h) - traj(name, value - h)) / (2.0 * h)
            scale = np.abs(fd).max()
            assert scale > 1.0, f"d(species)/d{name} is ~0: bad test setup"
            np.testing.assert_allclose(
                sens[:, :, j],
                fd,
                rtol=1e-5,
                atol=1e-6 * scale,
                err_msg=f"codegen sensitivity to {name} != FD of the interpreted trajectory",
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
        )
        r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
        return r.sensitivities[:, :, 0]

    def test_codegen_chain_rule_matches_fd(self):
        """Codegen sens for ``kon`` must include chain rule via ``_rateLaw1 = chi*kon``."""
        net = _get_derived_rate_const_net()
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
        net = _get_derived_quotient_net()
        code = _sens_c(net)
        assert code is not None
        mek_idx = _param_index(net, "MEK")

        import re

        case_match = re.search(rf"    case {mek_idx}:.*?break;", code, re.DOTALL)
        assert case_match, "MEK case missing from generated dfdp switch"
        snippet = case_match.group(0)
        assert f"pow(p[{mek_idx}], 2)" in snippet, (
            f"chain rule -5/pow(p[{mek_idx}], 2) not emitted; got:\n{snippet}"
        )

    def test_quotient_chain_rule_matches_fd(self):
        """Codegen sens for ``MEK`` must include the ``∂(5/MEK)/∂MEK`` chain rule."""
        net = _get_derived_quotient_net()
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

    def test_dfdp_case_includes_nested_reaction(self):
        """The generated ``bngsim_dfdp`` case for ``kcr`` must carry the
        contribution from the a2prime reaction (species C), not only the
        a1prime reaction — that species-C term is exactly what was dropped."""
        import re

        net = _get_nested_derived_net()
        code = _sens_c(net)
        assert code is not None
        kcr_idx = _param_index(net, "kcr")

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
        sim = bngsim.Simulator(mod, method="ode", sensitivity_params=["kcr"], codegen=True)
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
        r = bngsim.Simulator(m, method="ode", sensitivity_params=["R0"], codegen=True).run(
            sample_times=cls._SAMPLE_TIMES, rtol=1e-11, atol=1e-13, max_steps=10**6
        )
        return np.asarray(r.sensitivities)[:, :, 0]

    def test_seed_helper_coefficients(self):
        """compute_ic_param_sens_seed maps each parameter-referenced species IC
        to (species_idx0, param_idx0, coeff): coefficient 1 for the direct IC
        and — the #43 fix — for the derived ``Rtot = R0`` on R0 (the primary).
        Issue #715 adds the identity row on Rtot itself, the column a
        force_override pin of Rtot makes real; it only takes effect when Rtot is
        requested, so R0's column is unaffected."""
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
        assert (0, rtot_idx, 1.0) in seeds, (
            "derived IC Rtot must also seed its own column with coefficient 1 (issue #715)"
        )

    def test_direct_ic_matches_rebuild_fd(self, tmp_path):
        net = _get_ic_direct_net()
        fd = self._rebuild_fd(net, tmp_path)
        sx = self._analytic(net)
        assert abs(sx[0, 0] - 1.0) < 1e-6, "∂R(0)/∂R0 must seed to 1 for a direct IC"
        np.testing.assert_allclose(sx[:, 0], fd[:, 0], rtol=1e-4, atol=1e-6)

    def test_derived_ic_matches_rebuild_fd(self, tmp_path):
        """The core #43 regression: derived-parameter IC must seed ∂R/∂R0."""
        net = _get_ic_derived_net()
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

    def test_compound_condition_ic_matches_rebuild_fd(self, tmp_path):
        """Issue #56 at the IC-seed call site: the same derived IC, but written
        with a compound condition (``if((sel>=1)&&(sel<10), R0, 0.5*R0)``). The
        live branch is the same one ``ic_derived.net`` takes unconditionally, so
        the answer is unchanged — but the condition used to defeat the partial
        and leave the seed at 0, with the trajectory itself unaffected."""
        net = _get_ic_derived_compound_net()
        fd = self._rebuild_fd(net, tmp_path)
        sx = self._analytic(net)
        assert np.abs(sx[:, 0]).max() > 1e-3, (
            "compound-condition IC seed dropped: ∂R/∂R0 is ~0 (issue #56)"
        )
        assert abs(sx[0, 0] - 1.0) < 1e-6, "∂R(0)/∂R0 must seed to 1 through the true branch"
        np.testing.assert_allclose(sx[:, 0], fd[:, 0], rtol=1e-4, atol=1e-6)
        # And it must agree with the plain derived IC, which it is equivalent to.
        np.testing.assert_allclose(
            sx[:, 0], self._analytic(_get_ic_derived_net())[:, 0], rtol=1e-6, atol=1e-9
        )


class TestSupersededICParamSens:
    """An assignment retires the parameter-graph IC seed it superseded (issue #113).

    ``∂(IC)/∂p`` differentiates ``species[].initial_conc`` — the initial condition
    the model *declares*. Once ``set_concentration`` (or any bulk assignment) has
    moved a species off that baseline, the parameter cannot reach its initial
    condition at all, so keeping the row reports a gradient through an initial
    condition the model no longer has.

    ``ic_direct.net`` makes that exact: ``R0`` appears in **no rate law**, so with
    ``R`` pinned to a literal the true ``∂R(t)/∂R0`` is identically zero — no
    tolerance argument, and a rebuild finite difference confirms it.
    """

    _T = list(np.linspace(0.0, 3.0, 7))
    _TOL = dict(rtol=1e-11, atol=1e-13)

    def _sens(self, net, mutate=None, params=("R0",)):
        import bngsim

        m = bngsim.Model.from_net(net)
        if mutate is not None:
            mutate(m)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(params))
        r = sim.run(sample_times=self._T, **self._TOL)
        return m, np.asarray(r.sensitivities)[:, m.species_names.index("R()"), 0]

    @staticmethod
    def _pin(value=7.0):
        return lambda m: m.set_concentration("R()", value)

    def test_pinned_species_reports_no_gradient_through_its_ic(self):
        """The issue's reproducer: reported `e^{-kf t}`, truth 0."""
        _, sx = self._sens(_get_ic_direct_net(), self._pin())
        np.testing.assert_allclose(sx, np.zeros_like(sx), atol=1e-12)

    def test_pinned_species_matches_a_rebuild_finite_difference(self):
        """The oracle, independent of the seeding code: rebuild at R0 ± h with the
        same pin applied. R0 reaches nothing else, so the difference is exactly 0."""
        import re

        import bngsim

        src = Path(_get_ic_direct_net()).read_text()

        def traj(r0, tmp):
            txt = re.sub(r"(\bR0\s+)[0-9.]+", rf"\g<1>{r0}", src, count=1)
            p = tmp / f"ic_pinned_{r0}.net"
            p.write_text(txt)
            m = bngsim.Model.from_net(str(p))
            m.set_concentration("R()", 7.0)
            r = bngsim.Simulator(m, method="ode").run(sample_times=self._T, **self._TOL)
            return np.asarray(r.species)[:, m.species_names.index("R()")]

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fd = (traj(100.01, tmp) - traj(99.99, tmp)) / 0.02
        np.testing.assert_array_equal(fd, np.zeros_like(fd))
        _, sx = self._sens(_get_ic_direct_net(), self._pin())
        np.testing.assert_allclose(sx, fd, atol=1e-12)

    def test_untouched_model_keeps_its_seed(self):
        """No behaviour change where nothing was assigned — the #43 seed stands."""
        _, sx = self._sens(_get_ic_direct_net())
        assert abs(sx[0] - 1.0) < 1e-9
        assert np.abs(sx).max() > 1e-3

    def test_derived_ic_is_retired_too(self):
        """The rule is about the baseline, not about how the IC was written, so the
        derived-parameter IC (``Rtot = R0``) retires the same way."""
        _, pinned = self._sens(_get_ic_derived_net(), self._pin())
        np.testing.assert_allclose(pinned, np.zeros_like(pinned), atol=1e-12)
        _, plain = self._sens(_get_ic_derived_net())
        assert abs(plain[0] - 1.0) < 1e-9

    def test_reset_puts_the_species_back_and_the_row_returns(self):
        """reset() restores the live state to the baseline, so the IC expression
        describes it again."""

        def pin_then_reset(m):
            m.set_concentration("R()", 7.0)
            m.reset()

        _, sx = self._sens(_get_ic_direct_net(), pin_then_reset)
        assert abs(sx[0] - 1.0) < 1e-9

    def test_only_the_assigned_species_loses_its_row(self):
        """Per species, not per model: assigning P() leaves R()'s seed alone."""
        _, sx = self._sens(_get_ic_direct_net(), lambda m: m.set_concentration("P()", 5.0))
        assert abs(sx[0] - 1.0) < 1e-9

    def test_a_declaration_still_wins(self):
        """issue #111's declaration is the more specific statement: it survives the
        retirement, with the value the caller gave (0.5, not 1 and not 0)."""

        def pin_and_declare(m):
            m.set_concentration("R()", 7.0)
            m.declare_ic_sensitivity({"R()": {"R0": 0.5}})

        _, sx = self._sens(_get_ic_direct_net(), pin_and_declare)
        assert sx[0] == pytest.approx(0.5, rel=1e-9)

    def test_legacy_cpp_identity_seeding_applies_the_same_rule(self):
        """The C++ fallback loop (no Python injection — sympy unavailable) has the
        same defect and the same fix; reproduce it by neutering the injection.
        ``Simulator`` uses ``__slots__``, so patch the class, not the instance."""
        import bngsim

        orig = bngsim.Simulator._apply_ic_param_sens_seed
        bngsim.Simulator._apply_ic_param_sens_seed = lambda self, opts, model: None
        try:
            _, plain = self._sens(_get_ic_direct_net())
            _, pinned = self._sens(_get_ic_direct_net(), self._pin())
        finally:
            bngsim.Simulator._apply_ic_param_sens_seed = orig
        assert abs(plain[0] - 1.0) < 1e-9  # the legacy identity seed still fires...
        np.testing.assert_allclose(pinned, np.zeros_like(pinned), atol=1e-12)  # ...but not here

    def test_baseline_getter_tracks_the_lifecycle(self):
        """get_initial_state() is the baseline reset() returns to, and
        save_concentrations() redefines it to the live state."""
        import bngsim

        m = bngsim.Model.from_net(_get_ic_direct_net())
        np.testing.assert_array_equal(m.get_state(), m._core.get_initial_state())
        m.set_concentration("R()", 7.0)
        assert not np.array_equal(m.get_state(), m._core.get_initial_state())
        m.save_concentrations()
        np.testing.assert_array_equal(m.get_state(), m._core.get_initial_state())
