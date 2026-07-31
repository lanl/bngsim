"""Regression tests for GH #78 — ``method="newton"`` returned the saddle.

On a bistable model the two-tier solver's seed-stability guard cannot tell an
attractor from a saddle. It asks whether *refining the seed* moves the root;
near a separatrix the trajectory slows to a crawl, so two successively tighter
bursts hand KINSOL near-identical seeds a few percent from the saddle, both
polish to it, they agree — and the guard is satisfied exactly where it is
needed. The Gardner 2000 toggle at ``alpha_2 = 53.526315789473685`` returned
``[28.245, 1.830]`` with ``converged=True`` and residual 2.8e-10: an
equilibrium one part in 1e6 either side of which runs away to a *different*
attractor, while ``method="integration"`` was right at that dose and at the
other 19 of the 20-point scan.

The fix certifies the polished root's linear stability — the eigenvalues of the
Jacobian restricted to the species the polish solved for — and keeps
integrating when one has a positive real part. These tests pin the observable
contract: the saddle is a genuine root of ``f(y)=0`` and genuinely unstable, and
``method="newton"`` no longer returns it.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# Vendored (git-tracked) copies of the published RuleHub .net files — see the
# note in test_steady_state_gh27.py about why the generated corpus is not used.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NETS_DIR = _REPO_ROOT / "benchmarks" / "models" / "net" / "ode"

_GARDNER = "genetic_switch.net"  # Gardner 2000 genetic toggle
_BLBR = "blbr.net"  # bivalent ligand / bivalent receptor, 2 conservation laws

# The dose the issue reports: point 3 of the model's own
# parameter_scan(alpha_2, 1 -> 500, 20 points).
_SADDLE_DOSE = 53.526315789473685
# What the solver used to return there, and the branch integration reaches.
# Both to full precision (Newton on f(y)=0, residual ~1e-15), so the
# initial-condition case below really starts AT the root rather than 4e-8 away.
_SADDLE = np.array([28.245215769071677, 1.8302588776274484])
_ATTRACTOR = np.array([0.0075962079094201856, 53.122784076898327])


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    return str(path)


def _toggle(dose: float = _SADDLE_DOSE, **sim_kwargs):
    """Simulator on the Gardner toggle at `dose`, plus its Model."""
    model = bngsim.Model.from_net(_net(_GARDNER))
    model.set_param("alpha_2", dose)
    return bngsim.Simulator(model, method="ode", **sim_kwargs), model


def _max_real_eigenvalue(model, y) -> float:
    """max Re(lambda) of the Jacobian at `y`, scaled by the spectral radius.

    Computed here with numpy, independently of the solver's own certificate, so
    these tests do not grade the fix against itself.
    """
    n = model._core.n_species
    assert model._core.analytical_jacobian_complete
    jac = np.array(model._core._dense_analytical_jacobian(0.0, list(y))).reshape(n, n, order="F")
    ev = np.linalg.eigvals(jac)
    return float(ev.real.max() / max(np.abs(ev).max(), 1e-300))


class TestGH78Premise:
    """What the solver used to return, independent of the solver."""

    def test_the_saddle_is_a_root_of_f(self):
        """It converged for a reason: f(saddle) really is ~0."""
        _sim, model = _toggle()
        f = np.asarray(model._core._eval_rhs(0.0, list(_SADDLE)), dtype=float)
        residual = float(np.linalg.norm(f) / len(f))
        assert residual < 1e-9, f"the reported saddle is not a root: ||f||/n = {residual:g}"

    def test_the_saddle_is_unstable_and_the_attractor_is_not(self):
        """And it is a state no trajectory can rest on: +0.406 against -0.863."""
        sim, model = _toggle()
        sim.steady_state(method="integration")  # realizes the functional Jacobian
        assert _max_real_eigenvalue(model, _SADDLE) > 1e-2
        assert _max_real_eigenvalue(model, _ATTRACTOR) < -1e-2


class TestGH78SaddleRejected:
    def test_newton_does_not_return_the_saddle(self):
        sim, _model = _toggle()
        r = sim.steady_state(method="newton")
        conc = np.asarray(r.concentrations, dtype=float)
        assert r.converged
        assert np.all(np.isfinite(conc))
        assert not np.allclose(conc, _SADDLE, rtol=1e-3), "newton returned the saddle"
        np.testing.assert_allclose(conc, _ATTRACTOR, rtol=1e-6)

    def test_newton_agrees_with_integration_at_the_saddle_dose(self):
        sim, _model = _toggle()
        newton = np.asarray(sim.steady_state(method="newton").concentrations, dtype=float)
        sim2, _m2 = _toggle()
        integ = np.asarray(sim2.steady_state(method="integration").concentrations, dtype=float)
        np.testing.assert_allclose(newton, integ, rtol=1e-6)

    def test_the_rejection_is_reported(self):
        """A discarded root is on the result, not just absent from it."""
        sim, _model = _toggle()
        r = sim.steady_state(method="newton")
        assert r.n_unstable_roots_rejected == 1
        assert r.root_stability == "stable"

    def test_the_polish_still_happens_after_a_rejection(self):
        """Rejecting the saddle must not cost the tighter root the polish buys:
        a later rung lands on the attractor and that one is accepted."""
        sim, _model = _toggle()
        r = sim.steady_state(method="newton")
        assert r.method_used == "newton"
        integ_residual = _toggle()[0].steady_state(method="integration").residual
        assert r.residual < integ_residual

    def test_finite_difference_jacobian_rejects_it_too(self):
        """The certificate assembles J the way dY_ss/dp does, so jacobian="fd"
        exercises the differenced path — the verdict must not depend on it."""
        sim, _model = _toggle(jacobian="fd")
        r = sim.steady_state(method="newton")
        conc = np.asarray(r.concentrations, dtype=float)
        assert not np.allclose(conc, _SADDLE, rtol=1e-3)
        np.testing.assert_allclose(conc, _ATTRACTOR, rtol=1e-6)
        assert r.n_unstable_roots_rejected == 1

    def test_whole_dose_scan_agrees_with_integration(self):
        """The issue found one bad dose in 20; the other 19 must stay right."""
        for dose in np.linspace(1.0, 500.0, 20):
            sim, _model = _toggle(float(dose))
            newton = sim.steady_state(method="newton")
            sim2, _m2 = _toggle(float(dose))
            integ = sim2.steady_state(method="integration")
            assert newton.converged and integ.converged
            np.testing.assert_allclose(
                np.asarray(newton.concentrations, dtype=float),
                np.asarray(integ.concentrations, dtype=float),
                rtol=1e-5,
                err_msg=f"newton and integration disagree at alpha_2 = {dose}",
            )
            assert newton.root_stability in ("stable", "undetermined")


class TestGH78Reporting:
    def test_an_initial_condition_at_the_saddle_is_reported_not_hidden(self):
        """Handed the saddle as the IC, there is nowhere else to fall back to —
        integration would return the same state. Report the verdict instead of
        acting on it."""
        sim, model = _toggle()
        model.set_concentration("R1()", float(_SADDLE[0]))
        model.set_concentration("R2()", float(_SADDLE[1]))
        r = sim.steady_state(method="newton")
        assert r.converged
        assert r.method_used == "newton"
        np.testing.assert_allclose(np.asarray(r.concentrations, dtype=float), _SADDLE, rtol=1e-6)
        assert r.root_stability == "unstable"
        assert r.n_unstable_roots_rejected == 0

    def test_integration_carries_no_verdict(self):
        """A trajectory cannot come to rest on an unstable equilibrium, so the
        integration path needs no certificate and claims none."""
        sim, _model = _toggle()
        r = sim.steady_state(method="integration")
        assert r.converged
        assert r.root_stability == ""
        assert r.n_unstable_roots_rejected == 0


class TestGH78NoCollateral:
    def test_conservation_law_model_still_takes_the_polish(self):
        """The certificate reads the same restricted system the polish solved
        (conservation-law independents), so the zero eigenvalue every conserved
        quantity contributes must not be read as an instability."""
        model = bngsim.Model.from_net(_net(_BLBR))
        assert model._core.conservation_laws["n_laws"] > 0
        r = bngsim.Simulator(model, method="ode").steady_state(method="newton")
        assert r.converged
        assert r.method_used == "newton"
        assert r.root_stability == "stable"
        assert r.n_unstable_roots_rejected == 0

    def test_simple_decay_is_unaffected(self):
        """A unique-root model still confirms on the first pair of bursts."""
        decay = _REPO_ROOT / "tests" / "data" / "simple_decay.net"
        if not decay.is_file():
            pytest.skip(f"test data not available: {decay}")
        model = bngsim.Model.from_net(str(decay))
        r = bngsim.Simulator(model, method="ode").steady_state(method="newton")
        assert r.converged
        assert r.method_used == "newton"
        assert r.root_stability in ("stable", "undetermined")
        assert r.n_unstable_roots_rejected == 0
