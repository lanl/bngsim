"""Observable output sensitivities via the runtime linear chain rule (GH #197).

BNGL observables are linear in species, ``obs_j = Σ_i c_ji·x_i``, so

    d obs_j/dθ = Σ_i c_ji·dx_i/dθ

is computed in C++ at simulation time from the CVODES species sensitivities —
no codegen required. These tests check that the recorded observable
sensitivities (parameter axis AND initial-condition axis) equal the exact
chain-rule reconstruction from the species sensitivities, agree with central
finite differences, and are exposed through the ``observable:`` selectors.

Finite-difference references use high solver tolerances, central differences,
and a scale-relative tolerance, per the issue's acceptance criteria.
"""

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# A --k1--> B --k2--> C, all elementary mass action (so the codegen sensitivity
# RHS engages and the IC axis is available). Observables:
#   A_obs = A             (single species, unit coefficient)
#   BC2   = B + 2*C       (multi-species, non-unit coefficient, time-varying)
#   Total = A + B + C     (conserved = 100)
CHAIN_NET = str(DATA_DIR / "obs_sens_chain.net")
# Atot = A + 3*C with a Functional rate law (interpreted param-sens path).
NONUNIT_NET = str(DATA_DIR / "per_observable_jac.net")


def _empty_observable_net(tmp_path: Path) -> str:
    """A mass-action model with NO groups block (zero observables)."""
    src = (
        "begin parameters\n 1 k1 0.1\nend parameters\n"
        "begin species\n 1 A() 100\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 k1 #A_to_B\nend reactions\n"
    )
    p = tmp_path / "no_obs.net"
    p.write_text(src)
    return str(p)


# Tight tolerances + a coarse-enough grid keep the stiff-ish chain inside the
# default step budget while giving the FD reference clean derivatives.
_RUN_KW = dict(rtol=1e-11, atol=1e-13, max_steps=10**6)
_FD_KW = dict(rtol=1e-12, atol=1e-14, max_steps=10**6)
T_SPAN = (0.0, 15.0)
N_POINTS = 31


def _chain_result(params=("k1", "k2"), ic=None):
    """Run the chain model with the requested sensitivity axes (codegen)."""
    model = bngsim.Model.from_net(CHAIN_NET)
    sim = bngsim.Simulator(
        model,
        method="ode",
        sensitivity_params=list(params),
        sensitivity_ic=list(ic) if ic else None,
        codegen=True,
        net_path=CHAIN_NET,
    )
    return sim.run(t_span=T_SPAN, n_points=N_POINTS, **_RUN_KW)


def _fd_observable_param(net, name, param, nominal, *, rel=1e-4, t_span=T_SPAN, n_points=N_POINTS):
    """Central FD of an observable column wrt a parameter."""
    h = nominal * rel

    def _obs(value):
        m = bngsim.Model.from_net(net)
        m.set_param(param, value)
        r = bngsim.Simulator(m, method="ode").run(t_span=t_span, n_points=n_points, **_FD_KW)
        return r.observables[name]

    return (_obs(nominal + h) - _obs(nominal - h)) / (2.0 * h)


def _fd_observable_ic(net, name, species, nominal, *, h=1e-3, t_span=T_SPAN, n_points=N_POINTS):
    """Central FD of an observable column wrt a species' initial concentration."""

    def _obs(value):
        m = bngsim.Model.from_net(net)
        m.set_concentration(species, value)
        r = bngsim.Simulator(m, method="ode").run(t_span=t_span, n_points=n_points, **_FD_KW)
        return r.observables[name]

    return (_obs(nominal + h) - _obs(nominal - h)) / (2.0 * h)


def _max_relerr(a, b, floor=1e-7):
    """Scale-relative max error, ignoring entries where both are ~0."""
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.maximum(np.abs(a), np.abs(b))
    mask = denom > floor
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(a - b)[mask] / denom[mask]))


# ── Storage shape / availability ─────────────────────────────────────────────


class TestAvailability:
    def test_block_shape_param_axis(self):
        r = _chain_result(params=("k1", "k2"))
        assert r.has_sensitivities_observables
        # (n_times, n_observables, n_params)
        assert r.sensitivities_observables.shape == (N_POINTS, 3, 2)
        assert r.sensitivity_params == ["k1", "k2"]

    def test_no_sensitivity_run_leaves_blocks_empty(self):
        model = bngsim.Model.from_net(CHAIN_NET)
        r = bngsim.Simulator(model, method="ode").run(t_span=T_SPAN, n_points=N_POINTS, **_RUN_KW)
        assert not r.has_sensitivities_observables
        assert r.sensitivities_observables.shape == (0, 0, 0)

    def test_param_sens_zero_at_t0(self):
        r = _chain_result(params=("k1", "k2"))
        # Parameter sensitivities are identically zero at t=0.
        assert np.allclose(r.sensitivities_observables[0], 0.0)


# ── Parameter-axis chain rule + FD ───────────────────────────────────────────


class TestParameterAxis:
    def test_chain_rule_exact_multispecies(self):
        """BC2 = B + 2*C ⇒ block == dB/dp + 2·dC/dp, to floating-point."""
        r = _chain_result(params=("k1", "k2"))
        sp = r.sensitivities
        b, c = r.species_names.index("B()"), r.species_names.index("C()")
        bc2 = r.observable_names.index("BC2")
        expected = sp[:, b, :] + 2.0 * sp[:, c, :]
        np.testing.assert_allclose(r.sensitivities_observables[:, bc2, :], expected, atol=1e-10)

    def test_conserved_observable_has_zero_param_sensitivity(self):
        """Total = A+B+C is conserved, so d Total/dp ≈ 0 on every axis."""
        r = _chain_result(params=("k1", "k2"))
        tot = r.observable_names.index("Total")
        assert np.max(np.abs(r.sensitivities_observables[:, tot, :])) < 1e-9

    def test_fd_central_difference(self):
        r = _chain_result(params=("k1", "k2"))
        bc2 = r.observable_names.index("BC2")
        for pi, (param, nominal) in enumerate([("k1", 0.30), ("k2", 0.15)]):
            fd = _fd_observable_param(CHAIN_NET, "BC2", param, nominal)
            block = r.sensitivities_observables[:, bc2, pi]
            assert _max_relerr(fd, block) < 1e-4

    def test_nonunit_coefficient_matches_species_sum_and_fd(self):
        """Atot = A + 3*C (Functional model, interpreted sens path)."""
        model = bngsim.Model.from_net(NONUNIT_NET)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1", "k2"])
        r = sim.run(t_span=(0, 20), n_points=21, rtol=1e-9, atol=1e-11, max_steps=10**6)
        sp = r.sensitivities
        a, c = r.species_names.index("A()"), r.species_names.index("C()")
        atot = r.observable_names.index("Atot")
        expected = sp[:, a, :] + 3.0 * sp[:, c, :]
        np.testing.assert_allclose(r.sensitivities_observables[:, atot, :], expected, atol=1e-9)

        fd = _fd_observable_param(NONUNIT_NET, "Atot", "k1", 0.001, t_span=(0, 20), n_points=21)
        assert _max_relerr(fd, r.sensitivities_observables[:, atot, 0]) < 1e-3


# ── Initial-condition-axis chain rule + FD ───────────────────────────────────


class TestICAxis:
    def test_block_shape_ic_axis(self):
        r = _chain_result(params=(), ic=("A()",))
        assert r.has_sensitivities_observables_ic
        assert r.sensitivities_observables_ic.shape == (N_POINTS, 3, 1)
        assert r.sensitivity_ic_species == ["A()"]

    def test_chain_rule_exact_multispecies(self):
        r = _chain_result(params=(), ic=("A()",))
        spi = r.sensitivities_ic
        b, c = r.species_names.index("B()"), r.species_names.index("C()")
        bc2 = r.observable_names.index("BC2")
        expected = spi[:, b, :] + 2.0 * spi[:, c, :]
        np.testing.assert_allclose(r.sensitivities_observables_ic[:, bc2, :], expected, atol=1e-10)

    def test_conservation_ic_identity(self):
        """Total = A+B+C ⇒ d Total/dA(0) == 1 at all times (mass is conserved)."""
        r = _chain_result(params=(), ic=("A()",))
        tot = r.observable_names.index("Total")
        np.testing.assert_allclose(r.sensitivities_observables_ic[:, tot, 0], 1.0, atol=1e-8)

    def test_ic_coefficient_at_t0(self):
        """At t=0, d obs/dY_k(0) is just the observable's coefficient on k."""
        r = _chain_result(params=(), ic=("A()",))
        a_obs = r.observable_names.index("A_obs")  # A_obs = A, coeff 1
        bc2 = r.observable_names.index("BC2")  # BC2 = B + 2C, no A term ⇒ 0
        assert r.sensitivities_observables_ic[0, a_obs, 0] == pytest.approx(1.0)
        assert r.sensitivities_observables_ic[0, bc2, 0] == pytest.approx(0.0)

    def test_fd_central_difference(self):
        r = _chain_result(params=(), ic=("A()",))
        bc2 = r.observable_names.index("BC2")
        fd = _fd_observable_ic(CHAIN_NET, "BC2", "A()", 100.0)
        assert _max_relerr(fd, r.sensitivities_observables_ic[:, bc2, 0]) < 1e-4


# ── Empty-observable model ───────────────────────────────────────────────────


class TestEmptyObservableModel:
    def test_empty_arrays(self, tmp_path):
        net = _empty_observable_net(tmp_path)
        model = bngsim.Model.from_net(net)
        r = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"]).run(
            t_span=(0, 10), n_points=11, rtol=1e-9, atol=1e-11
        )
        assert r.observable_names == []
        assert not r.has_sensitivities_observables
        assert r.sensitivities_observables.shape == (0, 0, 0)
        # Species sensitivities are unaffected.
        assert r.sensitivities.shape == (11, 2, 1)


# ── Selector API (output_sensitivities) ──────────────────────────────────────


class TestSelectorAPI:
    def test_selector_matches_block_param_axis(self):
        r = _chain_result(params=("k1", "k2"))
        out = r.output_sensitivities(["observable:BC2", "observable:A_obs"])
        assert out.shape == (N_POINTS, 2, 2)
        bc2 = r.observable_names.index("BC2")
        a_obs = r.observable_names.index("A_obs")
        np.testing.assert_array_equal(out[:, 0, :], r.sensitivities_observables[:, bc2, :])
        np.testing.assert_array_equal(out[:, 1, :], r.sensitivities_observables[:, a_obs, :])

    def test_selector_matches_block_ic_axis(self):
        r = _chain_result(params=(), ic=("A()",))
        out = r.output_sensitivities("observable:BC2", axis="ic")
        assert out.shape == (N_POINTS, 1, 1)
        bc2 = r.observable_names.index("BC2")
        np.testing.assert_array_equal(out[:, 0, :], r.sensitivities_observables_ic[:, bc2, :])

    def test_species_selector_param_axis(self):
        r = _chain_result(params=("k1", "k2"))
        out = r.output_sensitivities("species:B()")
        b = r.species_names.index("B()")
        np.testing.assert_array_equal(out[:, 0, :], r.sensitivities[:, b, :])

    def test_empty_selector_list(self):
        r = _chain_result(params=("k1", "k2"))
        out = r.output_sensitivities([])
        assert out.shape == (N_POINTS, 0, 2)

    def test_unresolved_selector_errors(self):
        r = _chain_result(params=("k1",))
        with pytest.raises(ValueError, match="Unresolved selector"):
            r.output_sensitivities("observable:NoSuch")

    def test_expression_selector_computed_via_codegen(self):
        """Expression output sensitivities are now computed by the codegen
        evaluator (GH #198). ``fr() = k1*Atot`` has the chain-rule derivative
        ``d fr/d k1 = Atot + k1·dAtot/dk1`` (direct parameter term plus the
        observable chain), checked against the observable sensitivity block."""
        model = bngsim.Model.from_net(NONUNIT_NET)  # fr() = k1*Atot
        r = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"]).run(
            t_span=(0, 10), n_points=11, rtol=1e-9, atol=1e-11, max_steps=10**6
        )
        assert "fr" in r.expression_names
        out = r.output_sensitivities("expression:fr")
        assert out.shape == (11, 1, 1)
        assert np.all(np.isfinite(out))
        atot = np.asarray(r.observables["Atot"])
        d_atot_dk1 = r.output_sensitivities("observable:Atot")[:, 0, 0]
        expected = atot + 0.001 * d_atot_dk1  # k1 = 0.001 in per_observable_jac.net
        assert np.allclose(out[:, 0, 0], expected, rtol=1e-6, atol=1e-9)

    def test_bad_axis_errors(self):
        r = _chain_result(params=("k1",))
        with pytest.raises(ValueError, match="axis must be"):
            r.output_sensitivities("observable:BC2", axis="bogus")

    def test_axis_not_computed_errors(self):
        r = _chain_result(params=("k1",))  # no IC axis
        with pytest.raises(ValueError, match="no initial-condition sensitivities"):
            r.output_sensitivities("observable:BC2", axis="ic")

    def test_no_sensitivities_at_all_errors(self):
        model = bngsim.Model.from_net(CHAIN_NET)
        r = bngsim.Simulator(model, method="ode").run(t_span=T_SPAN, n_points=N_POINTS, **_RUN_KW)
        with pytest.raises(ValueError, match="no parameter sensitivities"):
            r.output_sensitivities("observable:BC2")
