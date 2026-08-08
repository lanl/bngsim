"""Steady state honors the codegen artifact, and dY_ss/dp is closed-form (issue #63).

Two defects, both silent from Python:

1. ``Simulator.steady_state()`` / ``steady_state_batch()`` built a
   ``SteadyStateOptions`` and never set ``codegen_so_path`` — and
   ``steady_state.cpp`` never read it either, nor did the struct have a field for
   the MIR JIT source. So a Simulator whose ``codegen_backend`` reported ``"cc"``
   solved for steady state on the interpreted ExprTk RHS, indistinguishably from
   one built with ``codegen=False``.

2. ``dY_ss/dp = -J⁻¹·(∂f/∂p)`` built BOTH factors from finite differences at a
   fixed ~sqrt(eps) step — J by one interpreted RHS evaluation per species even
   on a model whose complete analytical Jacobian the rest of the library uses
   under ``jacobian="auto"``, and ∂f/∂p by perturbing each parameter in place.
   Nothing warned; the result looked like every other sensitivity result.

The regression assertions are the provenance fields the fix adds
(``rhs_backend``, ``sens_jacobian_source``, ``sens_dfdp_source``) plus numerical
agreement between the closed-form assembly and the finite-difference one it
replaced.
"""

from __future__ import annotations

import logging
import os

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import SteadyStateOptions, find_steady_state
from bngsim._codegen import prepare_codegen

# run_tests.sh copies the tests to a temp dir, so resolve data via the env var
# first (same convention as test_codegen.py / test_mir_equivalence.py).
DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)


def net(name: str) -> str:
    return os.path.join(DATA, name)


@pytest.fixture
def reversible():
    """A + B <-> C: two conservation laws, so the reduced-space sensitivity solve
    runs — and law 1's coefficient row references law 0's dependent species, so
    the reconstruction chain rule is genuinely sequential rather than diagonal."""
    return bngsim.Model.from_net(net("two_species_reversible.net"))


# ── 1. The codegen artifact reaches the solver ────────────────────────────────


class TestCodegenReachesTheSolver:
    def test_codegen_true_runs_the_compiled_rhs(self, reversible):
        """The defect: this reported "cc" while the solve ran interpreted."""
        sim = bngsim.Simulator(reversible, method="ode", codegen=True)
        assert sim.codegen_backend in ("cc", "mir")
        ss = sim.steady_state(tol=1e-10)
        assert ss.converged
        # Whatever backend the Simulator advertises is the one that ran.
        expected = "codegen-jit" if sim.codegen_backend == "mir" else "codegen-so"
        assert ss.rhs_backend == expected

    def test_codegen_false_runs_the_interpreter(self, reversible):
        sim = bngsim.Simulator(reversible, method="ode", codegen=False)
        assert sim.codegen_backend == "exprtk"
        assert sim.steady_state(tol=1e-10).rhs_backend == "exprtk"

    def test_batch_entries_get_the_codegen_rhs_too(self, reversible):
        """steady_state_batch()'s _run_one had the same omission."""
        sim = bngsim.Simulator(reversible, method="ode", codegen=True)
        results = sim.steady_state_batch([{"kf": 0.1}, {"kf": 1.0}], tol=1e-9)
        expected = "codegen-jit" if sim.codegen_backend == "mir" else "codegen-so"
        assert [r.rhs_backend for r in results] == [expected, expected]

    @pytest.mark.parametrize("method", ["integration", "newton"])
    def test_compiled_and_interpreted_steady_states_agree(self, method):
        """Both solver methods: swapping the RHS backend must not move the root.

        Covers the CVODE march, the residual check, and (for "newton") the KINSOL
        polish — every RHS call site the fix rerouted.
        """
        interp = bngsim.Simulator(
            bngsim.Model.from_net(net("two_species_reversible.net")),
            method="ode",
            codegen=False,
        ).steady_state(method=method, tol=1e-10)
        compiled = bngsim.Simulator(
            bngsim.Model.from_net(net("two_species_reversible.net")),
            method="ode",
            codegen=True,
        ).steady_state(method=method, tol=1e-10)

        assert interp.converged and compiled.converged
        assert compiled.rhs_backend != "exprtk"
        np.testing.assert_allclose(
            compiled.concentrations, interp.concentrations, rtol=1e-9, atol=1e-12
        )

    def test_batch_compiled_matches_interpreted(self):
        doses = [{"kf": d} for d in (0.05, 0.5, 5.0)]
        interp = bngsim.Simulator(
            bngsim.Model.from_net(net("two_species_reversible.net")),
            method="ode",
            codegen=False,
        ).steady_state_batch(doses, tol=1e-10)
        compiled = bngsim.Simulator(
            bngsim.Model.from_net(net("two_species_reversible.net")),
            method="ode",
            codegen=True,
        ).steady_state_batch(doses, tol=1e-10)
        for a, b in zip(interp, compiled, strict=True):
            np.testing.assert_allclose(b.concentrations, a.concentrations, rtol=1e-9, atol=1e-12)

    def test_mir_jit_source_is_representable(self, reversible):
        """SteadyStateOptions had no codegen_c_source field at all, so the MIR
        backend could not even be *expressed* for a steady-state solve."""
        opts = SteadyStateOptions()
        assert hasattr(opts, "codegen_c_source")
        opts.codegen_c_source = "/* probe */"
        assert opts.codegen_c_source == "/* probe */"


# ── 2. dY_ss/dp uses the analytical RHS, or refuses ───────────────────────────


class TestSensitivityUsesAnalyticalRhs:
    def test_mass_action_uses_closed_form_for_both_factors(self, reversible):
        sim = bngsim.Simulator(reversible, method="ode")
        ss = sim.steady_state(sensitivity_params=["kf", "kr"], tol=1e-11)
        assert ss.converged
        assert ss.sens_jacobian_source in ("codegen", "analytical")
        assert ss.sens_dfdp_source == "codegen"

    def test_refuses_when_codegen_is_switched_off(self, reversible):
        """Same GH #214 policy as Simulator(sensitivity_params=...) and
        compute_all_sensitivities: refuse rather than silently degrade."""
        sim = bngsim.Simulator(reversible, method="ode", codegen=False)
        with pytest.raises(ValueError, match="requires code generation"):
            sim.steady_state(sensitivity_params=["kf"])

    def test_refuses_under_bngsim_no_codegen(self, reversible, monkeypatch):
        monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
        sim = bngsim.Simulator(reversible, method="ode")
        with pytest.raises(ValueError, match="requires code generation"):
            sim.steady_state(sensitivity_params=["kf"])

    def test_plain_steady_state_still_works_without_codegen(self, reversible):
        """The refusal is scoped to sensitivity requests — a plain solve on an
        explicitly interpreted Simulator must keep working."""
        sim = bngsim.Simulator(reversible, method="ode", codegen=False)
        ss = sim.steady_state(tol=1e-10)
        assert ss.converged and ss.rhs_backend == "exprtk"

    def test_michaelis_menten_no_longer_differences_dfdp(self, caplog):
        """Michaelis–Menten used to have no analytical ∂f/∂p, so the steady-state
        solve differenced that factor. #55's MM stage supplies the closed form,
        and ``steady_state.cpp`` picks the symbol up for free — the same
        ``finite-difference`` → ``codegen`` move #67 made for Functional models,
        retiring #76's √eps step floor for MM too.

        This test used to have to wrap the call in ``pytest.raises`` — the solve
        overshoots to ``S = -7e-8``, and the *clamp* made the tQSSA rate
        identically zero in a neighbourhood there, so ``f ≡ 0``, every direction
        was an equilibrium direction, and the entry point refused the gradient as
        a continuum. GH #93 removed the clamp, and with it that degeneracy: the
        rate varies through negative S, ∂f_S/∂S at the root is the honest
        ``-kcat·E/(Km + E)``, and the root is isolated. So the assertions below
        are the ones the test always wanted to make and could not."""
        sim = bngsim.Simulator(bngsim.Model.from_net(net("mm_tqssa.net")), method="ode")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            ss = sim.steady_state(sensitivity_params=["kcat", "Km"], tol=1e-10)
        assert not any("no analytical ∂f/∂p" in r.message for r in caplog.records), [
            r.message for r in caplog.records
        ]
        assert ss.converged
        assert ss.sens_dfdp_source == "codegen"
        # GH #93: an isolated root, not the continuum this used to report. The
        # pre-fix rcond was 0.0 and the solve returned NaN.
        assert ss.sens_jacobian_rcond > 0.1
        # The steady state is E=10, S=0, P=100 — all substrate converted — which
        # does not depend on kcat or Km, so the gradient is 0. Cross-checked
        # against a re-solve at kcat, Km ± h, which agrees to solver noise.
        assert np.max(np.abs(np.asarray(ss.sensitivity))) < 1e-8

    def test_a_model_that_still_declines_differences_dfdp_and_says_so(self, tmp_path, caplog):
        """The FD fallback has not gone away — a Functional law with a kink
        (``abs()``) has no analytic ∂f/∂p at any stage of #55 — and when it is
        taken it must still say so. Michaelis–Menten was this test's vehicle
        until MM gained its closed form."""
        text = """\
begin parameters
    1 kf     0.4  # Constant
    2 kr     0.2  # Constant
end parameters
begin functions
    1 kink() kf*abs(Atot - 1)
end functions
begin species
    1 A() 5.0
    2 B() 0.0
end species
begin reactions
    1 1 2 kink #_R1
    2 2 1 kr #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""
        path = tmp_path / "kink.net"
        path.write_text(text)
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            ss = sim.steady_state(sensitivity_params=["kf", "kr"], tol=1e-10)
        assert ss.converged
        assert ss.sens_dfdp_source == "finite-difference"
        assert any("no analytical ∂f/∂p" in r.message for r in caplog.records)


# ── 3. The closed-form assembly is numerically right ──────────────────────────


def _core_sensitivity(net_path, params, *, so_path="", jacobian="auto"):
    """Drive find_steady_state directly so the finite-difference assembly the fix
    replaced stays reachable for comparison (the Python entry point now refuses a
    codegen-less sensitivity request)."""
    model = bngsim.Model.from_net(net_path)
    model.prepare_analytical_jacobian()
    opts = SteadyStateOptions()
    opts.tol = 1e-12
    opts.jacobian = jacobian
    opts.sensitivity_params = list(params)
    if so_path:
        opts.codegen_so_path = so_path
    return find_steady_state(model._core, opts)


class TestSensitivityNumerics:
    def test_closed_form_matches_the_finite_difference_assembly(self):
        """The two factors' closed forms must agree with the all-FD path they
        replace, to FD's own accuracy — including through the conservation-law
        reduced-space solve, which this model exercises (2 laws, and law 1's
        coefficient row references law 0's dependent species, so the sequential
        chain rule matters)."""
        path = net("two_species_reversible.net")
        warm = bngsim.Model.from_net(path)
        warm.prepare_analytical_jacobian()
        so = str(prepare_codegen(path, warm, emit_jac=True))

        fd = _core_sensitivity(path, ["kf", "kr"], jacobian="fd")
        closed = _core_sensitivity(path, ["kf", "kr"], so_path=so, jacobian="auto")

        assert fd.sens_jacobian_source == "finite-difference"
        assert fd.sens_dfdp_source == "finite-difference"
        assert closed.sens_jacobian_source == "codegen"
        assert closed.sens_dfdp_source == "codegen"

        a = np.array(fd.sensitivity_data)
        b = np.array(closed.sensitivity_data)
        scale = max(np.abs(a).max(), 1e-30)
        # 1e-5 relative is far inside the one-sided FD path's own ~1e-7 noise and
        # far outside anything a wrong chain rule would produce.
        assert np.abs(a - b).max() / scale < 1e-5

    def test_reversible_binding_sensitivity_matches_closed_form(self, reversible):
        """A + B <-> C, conserving A−B and A+C.

        Equilibrium solves ``G(c) = kf·(A0−c)·(B0−c) − kr·c = 0`` for ``c = C_ss``,
        so implicit differentiation gives, with ``G' = −kf·(A0+B0−2c) − kr``,

            dC/dkf = (A0−c)(B0−c) / (kf·(A0+B0−2c) + kr)
            dC/dkr = −c / (kf·(A0+B0−2c) + kr)

        and dA/dp = dB/dp = −dC/dp from the two conservation laws. This pins the
        reduced-space solve AND the dependent-species reconstruction against a
        derivation that owes the solver nothing.
        """
        sim = bngsim.Simulator(reversible, method="ode")
        kf = reversible.get_param("kf")
        kr = reversible.get_param("kr")
        ss = sim.steady_state(sensitivity_params=["kf", "kr"], tol=1e-12)
        assert ss.converged

        names = list(ss.species_names)
        ia, ib, ic = names.index("A()"), names.index("B()"), names.index("C()")
        a, b, c = ss.concentrations[ia], ss.concentrations[ib], ss.concentrations[ic]
        a0, b0 = a + c, b + c  # conserved totals, recovered from the root

        gprime = kf * (a0 + b0 - 2 * c) + kr
        dC_dkf = (a0 - c) * (b0 - c) / gprime
        dC_dkr = -c / gprime

        np.testing.assert_allclose(ss.sensitivity[ic], [dC_dkf, dC_dkr], rtol=1e-6)
        np.testing.assert_allclose(ss.sensitivity[ia], [-dC_dkf, -dC_dkr], rtol=1e-6)
        np.testing.assert_allclose(ss.sensitivity[ib], [-dC_dkf, -dC_dkr], rtol=1e-6)

    def test_derived_rate_constant_chain_rule(self):
        """``_rateLaw1 = chi*kon``: A_ss = koff/(koff + chi·kon).

        The old finite-difference ∂f/∂p wrote the perturbed value straight into
        the Parameter vector, which nothing re-derives — so ``_rateLaw1`` kept its
        nominal value and the whole column came back zero. Both the analytical
        path and the repaired FD fallback now carry it.
        """
        path = net("derived_rate_const.net")
        warm = bngsim.Model.from_net(path)
        warm.prepare_analytical_jacobian()
        so = str(prepare_codegen(path, warm, emit_jac=True))
        params = ["kon", "chi", "koff", "_rateLaw1"]

        kon, chi, koff = 1.0, 10.0, 0.5
        rl = chi * kon
        d = (koff + rl) ** 2
        expected_A = np.array([-koff * chi / d, -koff * kon / d, rl / d, -koff / d])

        for so_path in ("", so):  # FD fallback and analytical must both be right
            r = _core_sensitivity(path, params, so_path=so_path)
            sens = np.array(r.sensitivity_data)
            ia = list(r.species_names).index("A()")
            np.testing.assert_allclose(sens[ia], expected_A, rtol=1e-5, atol=1e-9)

    def test_well_posed_system_reports_a_healthy_conditioning(self, reversible):
        sim = bngsim.Simulator(reversible, method="ode")
        ss = sim.steady_state(sensitivity_params=["kf", "kr"], tol=1e-12)
        assert ss.sens_jacobian_rcond > 1e-4

    def test_ill_conditioned_but_nonsingular_sensitivity_warns(self, tmp_path, caplog):
        """The warning branch: a root that is isolated and full rank, but whose
        reduced Jacobian is badly conditioned. The gradient is *correct* — the
        warning is advisory, exactly as its own text says.

        Two decoupled reversible pairs, ten orders of magnitude apart in rate:
        ``A ⇌ B`` at 1 and ``C ⇌ D`` at 1e10. Each pair conserves its own total,
        so the reduction leaves a 2x2 whose LU pivots are -(kf+kr) and
        -(kff+kfr): ``rcond`` is their ratio, ``1e-10``, analytically and to the
        last digit. That is two orders below ``_SS_SENS_RCOND_FLOOR`` (so the
        warning fires) and six orders *above* machine epsilon (so no rounding can
        collapse it to a zero pivot and tip the solve into the sibling's refusal
        branch).

        That last property is the point, and it is why this test does not use
        ``nested_derived_rate_const.net`` (lanl/bngsim#176). That model's
        equilibrium set is genuinely a line — J is rank 2 of 4 with one
        conservation law, so the reduced 3x3 is exactly singular in exact
        arithmetic — and the "ill-conditioned" pivot ratio this test used to
        assert was 1.26e-17, *below* eps. It was rounding noise, not conditioning,
        so which of the two branches fired was decided by the LU implementation:
        on one machine the same macOS/Accelerate build warns under LAPACK
        ``dgetrf`` and refuses under SUNDIALS' built-in GETRF. A fixture that
        cannot be held on one side of the line cannot discriminate the two
        branches, so the refusal branch keeps its own structural fixture
        (``test_singular_solve_is_refused_rather_than_returning_nan``) and this
        one gets a root that is honestly, stably ill-conditioned.
        """
        text = """\
begin parameters
    1 kf     1.0     # Constant
    2 kr     1.0     # Constant
    3 kff    1e10    # Constant
    4 kfr    1e10    # Constant
end parameters
begin species
    1 A() 1.0
    2 B() 0.0
    3 C() 1.0
    4 D() 0.0
end species
begin reactions
    1 1 2 kf   #_R_A_to_B
    2 2 1 kr   #_R_B_to_A
    3 3 4 kff  #_R_C_to_D
    4 4 3 kfr  #_R_D_to_C
end reactions
begin groups
    1 Atot 1
    2 Btot 2
    3 Ctot 3
    4 Dtot 4
end groups
"""
        path = tmp_path / "two_timescale_reversible.net"
        path.write_text(text)
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            ss = sim.steady_state(sensitivity_params=["kf", "kff"], tol=1e-12)

        assert ss.converged
        assert ss.sens_jacobian_rcond == pytest.approx(1e-10, rel=1e-9)
        assert any("badly conditioned" in r.message for r in caplog.records)

        # Isolated root, reached exactly: each pair equilibrates at half its total.
        assert np.asarray(ss.concentrations) == pytest.approx([0.5, 0.5, 0.5, 0.5], abs=1e-9)

        # ...and the flagged gradient is right. y* = (kr/(kf+kr), kf/(kf+kr)) per
        # pair, so dA/dkf = -kr/(kf+kr)^2 and the pairs do not cross-couple. A
        # warning the caller can act on has to be one they can also overrule.
        kf = kr = 1.0
        kff = kfr = 1e10
        exact = np.array(
            [
                [-kr / (kf + kr) ** 2, 0.0],
                [kr / (kf + kr) ** 2, 0.0],
                [0.0, -kfr / (kff + kfr) ** 2],
                [0.0, kfr / (kff + kfr) ** 2],
            ]
        )
        assert np.asarray(ss.sensitivity) == pytest.approx(exact, rel=1e-12, abs=1e-30)

    def test_singular_solve_is_refused_rather_than_returning_nan(self, tmp_path):
        """When the reduced LU hits an exact zero pivot there is no answer at all.

        SUNDIALS' dense solver has no least-squares fallback, so the result comes
        back NaN. That is the one case a refusal needs no threshold for — and the
        only refusal the corpus supports, since no cut on the conditioning
        separates correct gradients from wrong ones (see
        ``Simulator._SS_SENS_RCOND_FLOOR``).

        The vehicle is ``A -> B``, ``A -> C``: B and C are only ever produced,
        both fed from the same irreversible step, so the equilibrium set is the
        line ``B + C = const`` and the reduced Jacobian is exactly singular. That
        is the cause the refusal message itself names, and it is a *structural*
        degeneracy — nothing about it depends on a rate law's numerics.

        ``mm_tqssa.net`` was this test's vehicle until GH #93. It only looked
        degenerate because the MM clamp flattened the rate below S = 0; without
        the clamp that model has an isolated root and rcond 1.0, so it can no
        longer reach this path (see
        ``test_michaelis_menten_no_longer_differences_dfdp``).
        """
        text = """\
begin parameters
    1 kf     0.4  # Constant
end parameters
begin species
    1 A() 5.0
    2 B() 0.0
    3 C() 0.0
end species
begin reactions
    1 1 2 kf #_R1
    2 1 3 kf #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
    3 Ctot                 3
end groups
"""
        path = tmp_path / "two_sinks.net"
        path.write_text(text)
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
        with pytest.raises(bngsim.SimulationError) as exc:
            sim.steady_state(sensitivity_params=["kf"], tol=1e-10)
        msg = str(exc.value)
        assert "does not exist" in msg
        assert "continuum" in msg
        assert "B()" in msg  # names the species whose gradient came back non-finite

    def test_well_conditioned_sensitivity_is_not_refused(self, reversible):
        """The refusal must stay scoped to a solve that actually failed."""
        sim = bngsim.Simulator(reversible, method="ode")
        ss = sim.steady_state(sensitivity_params=["kf", "kr"], tol=1e-12)
        assert ss.converged
        assert np.all(np.isfinite(ss.sensitivity))

    def test_jacobian_fd_still_pins_the_difference_quotient(self, reversible):
        """jacobian="fd" is the escape hatch everywhere else; keep it meaningful
        here so the closed-form path stays A/B-checkable."""
        sim = bngsim.Simulator(reversible, method="ode", jacobian="fd")
        ss = sim.steady_state(sensitivity_params=["kf"], tol=1e-11)
        assert ss.sens_jacobian_source == "finite-difference"
