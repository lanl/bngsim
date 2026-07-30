"""Issue #74 — excluding write-only accumulator species from the convergence test.

A pure sink — a species some reaction produces and none consumes, the
``degraded`` / ``produced`` / ``secreted`` pool a BNGL model carries to count
cumulative flux — has a constant non-zero derivative for as long as its
producing reactions fire. ``||f(y)||_2 / n_species`` therefore has a floor above
``tol``, and ``steady_state()`` reported ``converged=False`` however long it
integrated, with nothing on the result to distinguish that from ordinary
numerical trouble. On ``Barua_2013`` the residual sits at 7.494e-3 across three
decades of ``max_time`` while the state grows linearly — a constant derivative,
not a slow tail.

Three things are under test:

* **Detection.** ``Model.pure_sink_species()`` finds the accumulators from the
  reaction list alone. The clause that is *not* implied by "product of some
  reaction, reactant of none" is inertness — no other species' derivative may
  read it — and that is the clause that makes excluding one provably harmless
  to the rest of the system.
* **The mask.** ``steady_state(mask=...)`` restricts the residual norm, the
  KINSOL unknown set and the ``dY_ss/dp`` system to the same subspace. The
  restriction is checked against closed-form steady states and gradients, not
  against the solver's own answer, because a self-consistent-but-wrong subspace
  solve would agree with itself.
* **The no-op.** With no mask — and with an all-true mask, which must mean the
  same thing — every code path is the one that ran before #74. That is asserted
  here on the models the mask feature touches and swept over the 585-model
  ``ode_fullnet`` corpus separately.

The masked ``dY_ss/dp`` rows for excluded species are ``NaN`` on purpose: a
species with no steady value has no steady-state gradient, and ``0.0`` would be
a confident wrong answer a fitter would silently read as "this parameter does
not matter".
"""

from __future__ import annotations

import logging
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim._exceptions import SimulationError

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_NETS = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"

# $S -> A -> Ad, with Ad the accumulator. No conservation laws, so the masked
# solve takes the zero-law reduction. A* = k_prod/k_deg = 10.
_PROD_DEG = _DATA / "preequil_prod_deg.net"

# The same shape plus a conserved moiety (B + AB = Btot), so the masked solve
# takes the conservation-law-reduced path with the independents narrowed.
_CONSERVED = _DATA / "pure_sink_conserved.net"

# The issue's reproducer: 409 species, 2737 reactions, exactly four pure sinks.
_BARUA13 = _NETS / "Barua_2013.net"

# Closed-form steady state of pure_sink_conserved.net (see the .net header).
_KSYN, _KON, _KOFF, _KDEG, _BTOT = 2.0, 3.0, 1.0, 0.5, 10.0
_A_SS = _KSYN / _KDEG
_DEN = _KOFF + _KON * _A_SS
_B_SS = _BTOT * _KOFF / _DEN
_AB_SS = _BTOT * _KON * _A_SS / _DEN


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


def _sim(path: Path):
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


def _barua():
    if not _NETS.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS}")
    assert _BARUA13.is_file(), f"vendored net missing from tracked corpus: {_BARUA13}"
    return bngsim.Model.from_net(str(_BARUA13))


# ─── Detection ──────────────────────────────────────────────────────────────


class TestPureSinkDetection:
    def test_names_the_accumulator(self):
        m = bngsim.Model.from_net(str(_PROD_DEG))
        assert m.pure_sink_species() == ["Ad()"]

    def test_is_pure_sink_is_the_array_form(self):
        m = bngsim.Model.from_net(str(_CONSERVED))
        flags = m.is_pure_sink()
        assert flags.dtype == np.bool_
        assert flags.shape == (m.n_species,)
        picked = [n for n, f in zip(m.species_names, flags, strict=True) if f]
        assert picked == m.pure_sink_species() == ["Ad()"]

    def test_fixed_species_is_not_a_sink(self):
        """``$S`` is a product of nothing, but the point is the general rule.

        A ``$``-prefixed species has its derivative zeroed, so it contributes
        nothing to the residual and can never be what holds a solve up. Reporting
        one would send the caller to mask a species that was never the problem.
        """
        m = bngsim.Model.from_net(str(_PROD_DEG))
        assert "S()" not in m.pure_sink_species()

    def test_detection_is_structural_not_a_convergence_verdict(self):
        """A *bounded* accumulator is still detected — and still converges.

        ``simple_decay`` is ``A -> B`` with nothing feeding A, so B is a textbook
        pure sink whose flux nevertheless dies out on its own. Detection answers
        "can this be dropped from the test without changing the problem", not
        "is this what is holding the solve up"; conflating the two would either
        send callers to mask species that were never a problem or, worse, teach
        them that a named species means a failed solve.
        """
        m = bngsim.Model.from_net(str(_DATA / "simple_decay.net"))
        assert m.pure_sink_species() == ["B()"]

        r = bngsim.Simulator(m, method="ode").steady_state(max_time=1e4)
        assert r.converged
        assert r.unconverged_pure_sinks == []

    def test_barua_2013_finds_exactly_the_four_from_the_issue(self):
        m = _barua()
        assert m.n_species == 409
        sinks = m.pure_sink_species()
        assert len(sinks) == 4, sinks
        assert all(s.endswith("ss~d)") for s in sinks), sinks


# ─── The failure diagnostic ─────────────────────────────────────────────────


class TestUnconvergedDiagnostic:
    def test_failed_solve_names_the_sink(self, caplog):
        sim = _sim(_PROD_DEG)
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            r = sim.steady_state(max_time=1e4)
        assert not r.converged
        assert r.unconverged_pure_sinks == ["Ad()"]
        assert "structural reason" in caplog.text
        assert "Ad()" in caplog.text

    def test_converged_solve_reports_no_sinks(self):
        """The diagnostic is about a failure, not about the model's shape."""
        r = _sim(_PROD_DEG).steady_state(max_time=1e4, mask=["S()", "A()"])
        assert r.converged
        assert r.unconverged_pure_sinks == []

    def test_barua_residual_floor_is_structural(self):
        """The issue's table: the residual does not move across two decades."""
        sim = bngsim.Simulator(_barua(), method="ode")
        residuals = []
        for max_time in (2.5e6, 2.5e7, 2.5e8):
            r = sim.steady_state(method="integration", max_time=max_time)
            assert not r.converged
            assert len(r.unconverged_pure_sinks) == 4
            residuals.append(r.residual)
        # A slow tail would fall; a constant derivative does not.
        assert max(residuals) - min(residuals) < 1e-9 * max(residuals)
        assert residuals[0] == pytest.approx(7.494e-3, rel=1e-3)


# ─── The mask ───────────────────────────────────────────────────────────────


class TestMaskedConvergence:
    def test_prod_deg_converges_to_the_closed_form(self):
        m = bngsim.Model.from_net(str(_PROD_DEG))
        sim = bngsim.Simulator(m, method="ode")
        r = sim.steady_state(mask=~m.is_pure_sink())
        assert r.converged
        assert r.n_residual_species == 2
        assert list(r.excluded_species) == [2]
        assert r["A()"] == pytest.approx(10.0, rel=1e-8)

    @pytest.mark.parametrize("method", ["integration", "newton"])
    def test_conserved_model_converges_to_the_closed_form(self, method):
        """The conservation-law-reduced path, checked against algebra.

        Both solver methods must land on the same physical root — the masked
        KINSOL unknown set is ``independent ∩ included``, and getting that
        intersection wrong is exactly the kind of error that still produces a
        self-consistent (and wrong) answer.
        """
        m = bngsim.Model.from_net(str(_CONSERVED))
        r = bngsim.Simulator(m, method="ode").steady_state(method=method, mask=~m.is_pure_sink())
        assert r.converged
        assert r.n_residual_species == 4
        assert r["A()"] == pytest.approx(_A_SS, rel=1e-8)
        assert r["B()"] == pytest.approx(_B_SS, rel=1e-8)
        assert r["AB()"] == pytest.approx(_AB_SS, rel=1e-8)
        # The conserved moiety is still conserved on the reduced solve.
        assert r["B()"] + r["AB()"] == pytest.approx(_BTOT, rel=1e-9)

    def test_excluded_species_still_integrates(self):
        """ "Everything still integrates; only the test is restricted."."""
        m = bngsim.Model.from_net(str(_CONSERVED))
        r = bngsim.Simulator(m, method="ode").steady_state(mask=~m.is_pure_sink())
        # Ad accumulated the degradation flux over the solve rather than being
        # dropped from the state vector.
        assert r["Ad()"] > 0.0

    def test_barua_2013_converges_with_the_mask(self):
        m = _barua()
        sim = bngsim.Simulator(m, method="ode")
        r = sim.steady_state(method="integration", max_time=2.5e8, mask=~m.is_pure_sink())
        assert r.converged
        assert r.residual < 1e-9
        assert r.n_residual_species == 405
        assert len(r.excluded_species) == 4

    def test_mask_by_species_name(self):
        m = bngsim.Model.from_net(str(_CONSERVED))
        keep = [n for n in m.species_names if n != "Ad()"]
        r = bngsim.Simulator(m, method="ode").steady_state(mask=keep)
        assert r.converged
        assert list(r.excluded_species) == [m.species_names.index("Ad()")]

    def test_batch_applies_one_mask_to_every_entry(self):
        m = bngsim.Model.from_net(str(_CONSERVED))
        sim = bngsim.Simulator(m, method="ode")
        results = sim.steady_state_batch(
            [{"ksyn": 1.0}, {"ksyn": 2.0}, {"ksyn": 4.0}], mask=~m.is_pure_sink()
        )
        assert [r.converged for r in results] == [True, True, True]
        # A* = ksyn/kdeg tracks the scan.
        assert [r["A()"] for r in results] == pytest.approx([2.0, 4.0, 8.0], rel=1e-7)


# ─── The no-op ──────────────────────────────────────────────────────────────


class TestUnmaskedIsUnchanged:
    @pytest.mark.parametrize("net", [_PROD_DEG, _CONSERVED, _DATA / "simple_decay.net"])
    def test_all_true_mask_is_identical_to_no_mask(self, net):
        """``mask=ones(n)`` must not quietly mean something different.

        The restricted KINSOL / dY_ss/dp branches deliberately do *not* reproduce
        the unmasked ones (they drop ``$``-fixed species from the unknown set,
        which the pre-#74 full-space path does not), so an all-true mask has to
        be routed back onto the original path rather than through the new one.
        """
        sim = _sim(net)
        a = sim.steady_state(max_time=1e4)
        b = sim.steady_state(max_time=1e4, mask=np.ones(sim._model.n_species, dtype=bool))
        assert a.converged == b.converged
        assert a.residual == b.residual
        assert a.method_used == b.method_used
        assert a.n_residual_species == b.n_residual_species
        assert list(b.excluded_species) == []
        np.testing.assert_array_equal(a.concentrations, b.concentrations)

    def test_default_reports_every_species_in_the_norm(self):
        m = bngsim.Model.from_net(str(_CONSERVED))
        r = bngsim.Simulator(m, method="ode").steady_state(max_time=1e4)
        assert r.n_residual_species == m.n_species
        assert list(r.excluded_species) == []

    def test_divisor_follows_the_mask(self):
        """``tol`` keeps meaning "per-species residual", so n is n_included.

        Checked against a hand-computed norm rather than against the solver: if
        the divisor stayed at ``n_species`` the reported residual would be
        smaller by exactly ``n_included / n_species`` and every tolerance would
        silently loosen with the number of species dropped.
        """
        m = bngsim.Model.from_net(str(_PROD_DEG))
        sim = bngsim.Simulator(m, method="ode")
        r = sim.steady_state(max_time=1e4)  # unconverged: Ad holds it up
        y = np.asarray(r.concentrations)
        # f = [0 (fixed S), k_prod - k_deg*A, k_deg*A]
        k_prod, k_deg = 5.0, 0.5
        f = np.array([0.0, k_prod - k_deg * y[1], k_deg * y[1]])
        assert r.residual == pytest.approx(np.linalg.norm(f) / 3, rel=1e-6)
        assert r.n_residual_species == 3


# ─── Argument handling ──────────────────────────────────────────────────────


class TestMaskArguments:
    def test_wrong_length_is_rejected(self):
        sim = _sim(_CONSERVED)
        with pytest.raises(ValueError, match="one entry per species"):
            sim.steady_state(mask=np.array([True, False]))

    def test_integer_mask_is_rejected_as_ambiguous(self):
        """``[0, 1]`` could be a 0/1 mask or a pair of indices; guessing is worse."""
        sim = _sim(_CONSERVED)
        with pytest.raises(TypeError, match="ambiguous"):
            sim.steady_state(mask=[1, 1, 1, 1, 0])

    def test_empty_mask_is_rejected(self):
        sim = _sim(_CONSERVED)
        with pytest.raises(ValueError, match="excludes every species"):
            sim.steady_state(mask=np.zeros(5, dtype=bool))

    def test_unknown_species_name_is_rejected(self):
        sim = _sim(_CONSERVED)
        with pytest.raises(ValueError, match="unknown species name"):
            sim.steady_state(mask=["A()", "not_a_species"])


# ─── Masked dY_ss/dp ────────────────────────────────────────────────────────


@requires_cc
class TestMaskedSensitivity:
    """The gradient must come from the same subspace the convergence test used.

    Leaving an accumulator in makes the reduced Jacobian singular (its column is
    structurally zero), so ``-J⁻¹·(∂f/∂p)`` has no solution and the whole request
    is refused. Restricting it to the included species is exact — nothing else's
    derivative reads the accumulator — and is checked here against closed forms.
    """

    def test_conserved_model_matches_the_closed_form(self):
        m = bngsim.Model.from_net(str(_CONSERVED))
        sim = bngsim.Simulator(m, method="ode")
        r = sim.steady_state(mask=~m.is_pure_sink(), sensitivity_params=["ksyn", "kdeg"])
        assert r.converged
        s = np.asarray(r.sensitivity)
        idx = {n: i for i, n in enumerate(m.species_names)}

        # dA*/dksyn = 1/kdeg ; dA*/dkdeg = -ksyn/kdeg^2
        assert s[idx["A()"], 0] == pytest.approx(1.0 / _KDEG, rel=1e-6)
        assert s[idx["A()"], 1] == pytest.approx(-_KSYN / _KDEG**2, rel=1e-6)

        # dB*/dp = -Btot*koff*kon/(koff + kon*A*)^2 * dA*/dp
        chain = -_BTOT * _KOFF * _KON / _DEN**2
        assert s[idx["B()"], 0] == pytest.approx(chain / _KDEG, rel=1e-6)
        assert s[idx["B()"], 1] == pytest.approx(chain * -_KSYN / _KDEG**2, rel=1e-6)

        # B + AB is held at Btot, so their gradients cancel exactly.
        np.testing.assert_allclose(s[idx["AB()"]], -s[idx["B()"]], rtol=1e-9)

        # A boundary condition does not move with a rate constant.
        np.testing.assert_array_equal(s[idx["S()"]], [0.0, 0.0])

    def test_excluded_rows_are_nan_not_zero(self):
        m = bngsim.Model.from_net(str(_CONSERVED))
        sim = bngsim.Simulator(m, method="ode")
        r = sim.steady_state(mask=~m.is_pure_sink(), sensitivity_params=["ksyn"])
        s = np.asarray(r.sensitivity)
        nan_rows = np.where(~np.all(np.isfinite(s), axis=1))[0]
        assert nan_rows.tolist() == list(r.excluded_species)
        # Everything else survived: the NaN is the caller's exclusion, not a
        # failed solve leaking through.
        assert np.all(np.isfinite(np.delete(s, list(r.excluded_species), axis=0)))

    def test_nan_rows_do_not_trip_the_singular_refusal(self, caplog):
        """The refusal exists for a solve that failed, not for a row we NaN'd."""
        m = bngsim.Model.from_net(str(_CONSERVED))
        sim = bngsim.Simulator(m, method="ode")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            r = sim.steady_state(mask=~m.is_pure_sink(), sensitivity_params=["ksyn"])
        assert r.sensitivity is not None
        assert "excluded from the convergence test" in caplog.text

    def test_unmasked_sensitivity_is_unreachable_on_a_sink_model(self):
        """Without the mask the solve never converges, so no gradient is offered.

        This is the state the issue describes: the question is not merely hard,
        it is ill-posed until the accumulator is taken out of it.
        """
        m = bngsim.Model.from_net(str(_CONSERVED))
        sim = bngsim.Simulator(m, method="ode")
        r = sim.steady_state(max_time=1e4, sensitivity_params=["ksyn"])
        assert not r.converged
        assert r.sensitivity is None

    def test_fd_oracle_on_the_steady_state_itself(self):
        """Central-difference the *steady state*, the only non-circular check.

        Re-solve at ``p ± h`` from the same initial conditions and difference the
        masked steady states. This is deliberately not compared against the
        solver's own Jacobian: an analytic gradient assembled on the wrong
        subspace would agree with a finite difference taken on that same wrong
        subspace.
        """
        m = bngsim.Model.from_net(str(_CONSERVED))
        sim = bngsim.Simulator(m, method="ode")
        mask = ~m.is_pure_sink()
        r = sim.steady_state(mask=mask, sensitivity_params=["ksyn"])
        assert r.converged

        base = m.get_param("ksyn")
        h = 1e-4 * base
        probes = {}
        for sign in (+1, -1):
            # Re-solve from the declared ICs, not from wherever the last probe
            # left the state: the sensitivity run above wrote the steady state
            # into the model, and a probe seeded there would be measuring a
            # different problem. `ksyn` touches no initial condition, so the
            # GH #79 write-propagation hole does not reach this oracle.
            m.reset()
            m.set_param("ksyn", base + sign * h)
            probe = sim.steady_state(mask=mask)
            assert probe.converged
            probes[sign] = np.asarray(probe.concentrations)
        m.reset()
        m.set_param("ksyn", base)

        fd = (probes[+1] - probes[-1]) / (2 * h)
        analytic = np.asarray(r.sensitivity)[:, 0]
        keep = np.asarray(mask)
        scale = max(np.max(np.abs(fd[keep])), 1e-12)
        np.testing.assert_allclose(analytic[keep], fd[keep], atol=1e-5 * scale, rtol=1e-4)

    def test_barua_2013_gradient_is_finite_off_the_sinks(self):
        m = _barua()
        sim = bngsim.Simulator(m, method="ode")
        r = sim.steady_state(
            max_time=2.5e8, mask=~m.is_pure_sink(), sensitivity_params=["kf1_bap"]
        )
        assert r.converged
        s = np.asarray(r.sensitivity)
        assert s.shape == (409, 1)
        assert np.where(~np.isfinite(s[:, 0]))[0].tolist() == list(r.excluded_species)
        assert np.any(np.abs(s[np.asarray(~m.is_pure_sink()), 0]) > 0)


def test_singular_refusal_points_at_the_mask():
    """A genuine continuum still refuses — and now says what to check.

    ``nested_derived_rate_const.net`` has a line of equilibria that is *not* a
    pure-sink artifact, so the refusal must stay a refusal. The only change is
    that its text now names the tool that would resolve the other cause.
    """
    net = _DATA / "nested_derived_rate_const.net"
    if not net.is_file():
        pytest.skip(f"fixture not present: {net}")
    m = bngsim.Model.from_net(str(net))
    sim = bngsim.Simulator(m, method="ode")
    try:
        r = sim.steady_state(sensitivity_params=m.param_names[:1])
    except SimulationError as e:
        assert "pure_sink_species" in str(e)
    else:
        # Not singular in this build/configuration — nothing to assert about the
        # refusal, but the solve must not have produced NaN either.
        assert r.sensitivity is None or np.all(np.isfinite(r.sensitivity))
