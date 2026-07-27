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

    def test_non_elementary_model_differences_dfdp_and_says_so(self, caplog):
        """A Functional/MM model has no analytical ∂f/∂p to emit (issue #55), so
        that factor is still differenced — but no longer silently.

        ``mm_tqssa.net`` also happens to land on a degenerate root (the solve
        overshoots to ``S = -7e-8``, where the tQSSA rate is identically zero in a
        neighborhood, so ``f ≡ 0`` and every direction is an equilibrium
        direction). The Python entry point therefore refuses it — but the ∂f/∂p
        diagnostic must still reach the log on the way out, which is what this
        pins. ``sens_dfdp_source`` is checked on the core result, which has no
        such gate.
        """
        sim = bngsim.Simulator(bngsim.Model.from_net(net("mm_tqssa.net")), method="ode")
        with (
            caplog.at_level(logging.WARNING, logger="bngsim"),
            pytest.raises(bngsim.SimulationError, match="does not exist"),
        ):
            sim.steady_state(sensitivity_params=["kcat", "Km"], tol=1e-10)
        assert any("finite differences" in r.message for r in caplog.records)

        core = _core_sensitivity(net("mm_tqssa.net"), ["kcat", "Km"])
        assert core.sens_dfdp_source == "finite-difference"


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

    def test_degenerate_steady_state_is_flagged(self, caplog):
        """``nested_derived_rate_const.net`` runs A→B→D and A→C with no reverse
        reactions, so equilibrium is A=B=0 with any C+D=1 — a continuum, not a
        point, and dY_ss/dp does not exist.

        With the old finite-difference Jacobian its ~sqrt(eps) noise perturbed the
        singular direction into invertibility and the solve returned a modest,
        meaningless answer. An exact Jacobian does not launder that.

        This is a TRUE degeneracy, unlike the three large models issue #63
        originally reported: those were an ill-posed conservation-law reduction
        (see test_conservation_laws.py) and are full rank once it is repaired.
        Here the equilibrium set really is a line, so the LU returns finite
        numbers only because the pivots stay nonzero — the conditioning warning is
        what surfaces it.
        """
        model = bngsim.Model.from_net(net("nested_derived_rate_const.net"))
        sim = bngsim.Simulator(model, method="ode")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            ss = sim.steady_state(sensitivity_params=["kcr", "kf"], tol=1e-12)
        assert ss.converged
        assert ss.sens_jacobian_rcond < 1e-8
        assert any("badly conditioned" in r.message for r in caplog.records)

    def test_singular_solve_is_refused_rather_than_returning_nan(self):
        """When the reduced LU hits an exact zero pivot there is no answer at all.

        SUNDIALS' dense solver has no least-squares fallback, so the result comes
        back NaN. That is the one case a refusal needs no threshold for — and the
        only refusal the corpus supports, since no cut on the conditioning
        separates correct gradients from wrong ones (see
        ``Simulator._SS_SENS_RCOND_FLOOR``).
        """
        sim = bngsim.Simulator(bngsim.Model.from_net(net("mm_tqssa.net")), method="ode")
        with pytest.raises(bngsim.SimulationError) as exc:
            sim.steady_state(sensitivity_params=["kcat"], tol=1e-10)
        msg = str(exc.value)
        assert "does not exist" in msg
        assert "continuum" in msg
        assert "S()" in msg  # names the species whose gradient came back non-finite

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
