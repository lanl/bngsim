"""Issue #76 — the finite-difference probes behind ``dY_ss/dp`` are relative.

``compute_ss_sensitivity`` differences two factors when no closed form is
available, and both probes used ``eps * max(|x|, 1)``. That absolute floor
overrides the relative step for everything smaller than 1, so the probe stops
being small compared with what it perturbs:

* a rate constant of 1e-9 was moved by **1500% of itself**, and the quotient
  came back as a secant across the rate law's curvature rather than a
  derivative — 15.9x low for ``dY_ss/dKD`` on ``ode/before_bunching``;
* a species in a nanomolar model was moved by fifteen times its own value.

The floor also cuts the other way: in a model carrying molecule counts (~1e6)
the same 1.49e-8 is 1e-14 of the state, which is cancellation noise. Both probes
are now relative — to the parameter itself, and to the species floored at the
state's own magnitude (unlike parameters, every species is a concentration in
one unit, so the state HAS a typical scale to floor a zero species against).

Issue #123 is the other end of the same tradeoff, and ``TestCancelledTerm``
covers it: a parameter whose own term is a small fraction of the derivative it
sits in moves that derivative by less than its roundoff at a step relative to the
parameter, so the quotient is noise where the wide step's was not. Each probe now
takes two steps and each component keeps the one that carried a response above
its own roundoff floor.

Each model here has a closed-form steady state AND a closed-form gradient
derived in its ``.net`` header, so the assertions owe the solver nothing. Every
one of them fails on the step rule it is about — by 79-94% for #76, and by 100%
(a fabricated exact zero) for #123.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

# A <-> B with the forward rate constant derived as kf = kon/KD, KD = 1e-9 — the
# shape ode/before_bunching carries, and the reason the RHS depends on KD
# non-linearly (a mass-action constant alone is linear, so any step is exact).
_RECIPROCAL = _DATA / "reciprocal_rate_const.net"

# A + A <-> B at nanomolar concentrations. Homodimerization is quadratic in A,
# so the species probe does not cancel out of the Jacobian column the way a
# bilinear term's does.
_DIMER = _DATA / "nanomolar_dimer.net"

# The same model plus a write-only accumulator sitting at 1e3 — the issue #74
# interaction: a species with no steady value must not set the probe scale for
# the species that have one.
_DIMER_SINK = _DATA / "nanomolar_dimer_sink.net"

# 0 -> A at flux 100 alongside 0 -> A at flux 1e-9: the trace parameter's own
# term is 1e-11 of the derivative it sits in, so a probe relative to it moves
# dA/dt by less than dA/dt's roundoff (issue #123).
_CANCELLED = _DATA / "cancelled_parameter_term.net"

# Closed forms, from the .net headers.
_KON, _KD, _KOFF = 1.0e-9, 1.0e-9, 1.0
_KF = _KON / _KD  # 1.0
_A_SS = _KOFF / (_KF + _KOFF)  # 0.5
_DA = np.array(
    [
        _KOFF * _KON / (_KD**2 * (_KF + _KOFF) ** 2),  # dA*/dKD   =  2.5e8
        -_KOFF / (_KD * (_KF + _KOFF) ** 2),  # dA*/dkon  = -2.5e8
        _KF / (_KF + _KOFF) ** 2,  # dA*/dkoff =  0.25
    ]
)
# perKD() = A_tot/KD, so d/dp carries an explicit ∂/∂KD term as well as the
# state chain — the two finite differences compute_ss_output_sensitivity takes.
_DPERKD = np.array([-_A_SS / _KD**2 + _DA[0] / _KD, _DA[1] / _KD, _DA[2] / _KD])

_KDIM, _KDIS = 2.0e18, 4.0e9
_R = _KDIM / (2.0 * _KDIS)  # 2.5e8
_A_DIMER = 1.0e-9
_B_DIMER = _R * _A_DIMER**2  # 2.5e-10
_DA_DR = -1.0 / (16.0 * _R**2)  # -1e-18
_DA_DIMER = np.array(
    [
        _DA_DR / (2.0 * _KDIS),  # dA*/dkdim = -1.25e-28
        _DA_DR * (-_KDIM / (2.0 * _KDIS**2)),  # dA*/dkdis =  6.25e-20
    ]
)


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


def _core_sensitivity(net_path, params, *, so_path="", jacobian="auto", **opt):
    """Drive ``find_steady_state`` directly, which is how the finite-difference
    assembly stays reachable — the Python entry point refuses a codegen-less
    sensitivity request, and with a codegen artifact attached ``∂f/∂p`` comes
    from the compiled sensitivity RHS instead (the same reason
    ``test_steady_state_codegen.py`` drives the core here)."""
    model = bngsim.Model.from_net(str(net_path))
    model.prepare_analytical_jacobian()
    opts = SteadyStateOptions()
    opts.tol = 1e-12
    opts.jacobian = jacobian
    opts.sensitivity_params = list(params)
    for k, v in opt.items():
        setattr(opts, k, v)
    if so_path:
        opts.codegen_so_path = so_path
    result = find_steady_state(model._core, opts)
    sens = np.array(result.sensitivity_data).reshape(len(result.species_names), -1)
    return result, sens


class TestParameterProbe:
    """``∂f/∂p`` differenced in a parameter far below 1."""

    def test_small_rate_constant_gradient_matches_the_closed_form(self):
        """``dA*/dKD`` at KD = 1e-9, against the derivation in the .net header.

        On the absolute floor the probe lands at KD = 1.59e-8, which drags the
        derived ``kf = kon/KD`` from 1.0 to 0.063 — the quotient is a secant
        across that whole interval and comes back 15.9x low (rel. err. 0.937).
        """
        result, sens = _core_sensitivity(_RECIPROCAL, ["KD", "kon", "koff"])
        assert result.sens_dfdp_source == "finite-difference"
        assert result.converged
        np.testing.assert_allclose(result.concentrations, [_A_SS, 1.0 - _A_SS], rtol=1e-9)
        # 1e-6 is inside a one-sided difference quotient's own ~1e-8 accuracy
        # and far outside the 94% the absolute floor produced.
        np.testing.assert_allclose(sens[0], _DA, rtol=1e-6)
        np.testing.assert_allclose(sens[1], -_DA, rtol=1e-6)

    def test_well_scaled_parameters_in_the_same_model_are_unaffected(self):
        """``kon`` and ``koff`` were never mis-stepped — ``koff`` is 1.0, and the
        RHS is linear in ``kon``, so any step is exact for it. They are asserted
        together with KD above; this pins that the repair did not move them,
        which is what makes the KD column's 15.9x attributable to the step."""
        _, sens = _core_sensitivity(_RECIPROCAL, ["KD", "kon", "koff"])
        np.testing.assert_allclose(sens[0][1:], _DA[1:], rtol=1e-8)

    @requires_cc
    def test_finite_difference_dfdp_agrees_with_the_compiled_one(self):
        """The same column from the compiled ``bngsim_codegen_sens_rhs``, which
        is exact and owes the step size nothing."""
        warm = bngsim.Model.from_net(str(_RECIPROCAL))
        warm.prepare_analytical_jacobian()
        # Issue #217: the sens RHS is emitted only for a build that asks for it.
        # ``Simulator`` sets this from ``sensitivity_params``; this test drives
        # ``prepare_codegen`` directly, so it says so itself.
        warm._want_output_sens = True
        so = str(cg.prepare_codegen(str(_RECIPROCAL), warm, emit_jac=True))

        fd_result, fd = _core_sensitivity(_RECIPROCAL, ["KD", "kon", "koff"])
        an_result, an = _core_sensitivity(_RECIPROCAL, ["KD", "kon", "koff"], so_path=so)

        assert fd_result.sens_dfdp_source == "finite-difference"
        assert an_result.sens_dfdp_source == "codegen"
        scale = max(np.abs(an).max(), 1e-30)
        assert np.abs(fd - an).max() / scale < 1e-6

    def test_expression_sensitivity_carries_the_same_probe(self):
        """``compute_ss_output_sensitivity`` differences ``∂func/∂p`` with the
        same step. ``perKD() = A_tot/KD`` is non-linear in KD, so the explicit
        term is 16x low on the absolute floor even though the state chain that
        rides along with it is fine."""
        result, _ = _core_sensitivity(_RECIPROCAL, ["KD", "kon", "koff"])
        assert result.sens_output_source == "finite-difference"
        assert list(result.expression_names) == ["perKD"]
        got = np.asarray(result.expression_sensitivity_data)[0]
        np.testing.assert_allclose(got, _DPERKD, rtol=1e-6)


class TestStateProbe:
    """The Jacobian factor differenced in a species far below 1."""

    def test_nanomolar_jacobian_gradient_matches_the_closed_form(self):
        """``jacobian="fd"`` pins the finite-difference assembly. At A* = 1e-9
        the absolute floor perturbs the species by 15x its own value, and the
        spurious ``-kdim*h`` term it leaves in ∂f_A/∂A is 7.5x the real one —
        the gradient comes back 79% low."""
        result, sens = _core_sensitivity(
            _DIMER, ["kdim", "kdis"], jacobian="fd", atol=1e-24, rtol=1e-12
        )
        assert result.sens_jacobian_source == "finite-difference"
        assert result.converged
        np.testing.assert_allclose(result.concentrations, [_A_DIMER, _B_DIMER], rtol=1e-6)
        np.testing.assert_allclose(sens[0], _DA_DIMER, rtol=1e-6)
        np.testing.assert_allclose(sens[1], -_DA_DIMER / 2.0, rtol=1e-6)

    def test_finite_difference_jacobian_agrees_with_the_analytical_one(self):
        """Same model, same root, the two Jacobian factors against each other."""
        _, fd = _core_sensitivity(_DIMER, ["kdim", "kdis"], jacobian="fd", atol=1e-24, rtol=1e-12)
        an_result, an = _core_sensitivity(
            _DIMER, ["kdim", "kdis"], jacobian="analytical", atol=1e-24, rtol=1e-12
        )
        assert an_result.sens_jacobian_source == "analytical"
        scale = max(np.abs(an).max(), 1e-30)
        assert np.abs(fd - an).max() / scale < 1e-6

    def test_a_masked_out_accumulator_does_not_set_the_probe_scale(self):
        """The state scale is taken over the species that have a steady value.

        A write-only accumulator (issue #74) holds whatever integration left it
        at — unbounded, and 1e3 here against a 1e-9 state. Letting it into the
        scale would probe every species at 1.49e-5, which is 1.5e4 times A*:
        exactly the defect this issue is about, arriving through a species that
        was masked out of the problem.
        """
        model = bngsim.Model.from_net(str(_DIMER_SINK))
        model.prepare_analytical_jacobian()
        names = list(model._core.species_names)
        mask = [0 if n.startswith("Ad") else 1 for n in names]

        opts = SteadyStateOptions()
        opts.tol, opts.jacobian = 1e-12, "fd"
        opts.atol, opts.rtol = 1e-24, 1e-12
        opts.sensitivity_params = ["kdim", "kdis"]
        opts.steady_state_mask = mask
        result = find_steady_state(model._core, opts)

        assert result.converged
        assert result.sens_jacobian_source == "finite-difference"
        assert list(result.excluded_species) == [names.index("Ad()")]
        sens = np.array(result.sensitivity_data).reshape(len(result.species_names), -1)
        ia, ib = names.index("A()"), names.index("B()")
        np.testing.assert_allclose(
            [result.concentrations[ia], result.concentrations[ib]],
            [_A_DIMER, _B_DIMER],
            rtol=1e-6,
        )
        np.testing.assert_allclose(sens[ia], _DA_DIMER, rtol=1e-6)
        np.testing.assert_allclose(sens[ib], -_DA_DIMER / 2.0, rtol=1e-6)


class TestCancelledTerm:
    """Issue #123 — a parameter whose response the relative step cannot resolve.

    ``dA/dt = ksyn + ktrace - kdeg*A`` with ``ksyn`` = 100 and ``ktrace`` = 1e-9.
    The step relative to ``ktrace`` is 1.5e-17, which moves ``dA/dt`` by 1.5e-17
    against a roundoff floor of ~2.2e-14 — the response is three orders below the
    noise, so the quotient carries no information at all. The pre-#76 absolute
    step moves it by 1.5e-8, 6.7e5 times the floor, and ``dA/dt`` is linear in
    ``ktrace`` so that quotient is good to ~1.5e-6.
    """

    #: ``A* = (ksyn + ktrace)/kdeg``; J and every ``∂f/∂p`` are constants here, so
    #: the gradient is exact and owes the root nothing.
    _EXPECTED = np.array([1.0, 1.0, -100.000000001])  # d/d[ktrace, ksyn, kdeg]

    def test_a_trace_parameter_keeps_its_gradient(self):
        """#76's relative step alone returns **exactly 0.0** for ``dA*/dktrace``
        — the shape a fitter reads as "this parameter does not matter" — while
        the two columns whose responses are resolvable are unchanged."""
        result, sens = _core_sensitivity(_CANCELLED, ["ktrace", "ksyn", "kdeg"])
        assert result.sens_dfdp_source == "finite-difference"
        assert result.converged
        ia = list(result.species_names).index("A()")
        np.testing.assert_allclose(result.concentrations[ia], 100.000000001, rtol=1e-9)
        np.testing.assert_allclose(sens[ia], self._EXPECTED, rtol=1e-5)

    @requires_cc
    def test_the_trace_column_matches_the_compiled_dfdp(self):
        """Same column from the compiled ``bngsim_codegen_sens_rhs``, which owes
        the step size nothing."""
        warm = bngsim.Model.from_net(str(_CANCELLED))
        warm.prepare_analytical_jacobian()
        # Issue #217: the sens RHS is emitted only for a build that asks for it.
        # ``Simulator`` sets this from ``sensitivity_params``; this test drives
        # ``prepare_codegen`` directly, so it says so itself.
        warm._want_output_sens = True
        so = str(cg.prepare_codegen(str(_CANCELLED), warm, emit_jac=True))
        params = ["ktrace", "ksyn", "kdeg"]

        fd_result, fd = _core_sensitivity(_CANCELLED, params)
        an_result, an = _core_sensitivity(_CANCELLED, params, so_path=so)

        assert fd_result.sens_dfdp_source == "finite-difference"
        assert an_result.sens_dfdp_source == "codegen"
        assert np.abs(fd - an).max() / max(np.abs(an).max(), 1e-30) < 1e-5
