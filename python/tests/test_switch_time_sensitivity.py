"""Issue #48: forward sensitivity w.r.t. *switch-time* parameters.

A switch time is a fitted parameter that sets **when** a step in the dynamics
occurs rather than how fast something happens inside a branch — the
``if(t>=sigma, ...)`` onset times of the Lin2021 COVID model, gated on a
unit-rate counter clock.

Before #48 such a parameter was silently wrong *and* hung the integrator:

* ``∂f/∂sigma`` is a clean ``0`` inside each branch (sympy drops the boundary
  delta when the parameter appears only in the condition), so the variational
  source term carried no switch information at all; and
* with that column requested, CVODES' internal finite-difference probe perturbed
  ``sigma`` itself, moving the switch into the approach to the crossing —
  error control then collapsed ``h`` to ~1e-16 and the run never returned.

The fix is the jump ``s(t*⁺) = s(t*⁻) + (f⁻ − f⁺)·∂t*/∂p`` applied at a
``CVodeSetStopTime`` stop on the crossing, with the switch parameter pinned
against the FD probe. These tests pin the numbers the prototype in
``dev/switch_time_sensitivity/`` validated against finite differences on both a
minimal model and the 26-species Lin2021 exemplar.

Coverage:

* **minimal model** — ``dX/dsigma = −k`` against closed form and central FD, at
  several tolerances (the answer must not drift with rtol), plus the pre-crossing
  value of 0 and the fact that the run terminates at all.
* **sequential + derived thresholds** — the Lin2021 structure in miniature: a
  clock with a non-zero initial offset, ``sigma = t0 + t_delta`` and
  ``tau1 = sigma + t_delta2`` so the chain rule must place a ``t0`` jump at two
  crossings and a ``t_delta`` jump at only one, a jump propagated through a
  nonlinear coupling, and an *uncrossed* third threshold whose column must be
  exactly zero.
* **detection** — records/pins are produced only for parameters that actually
  move a crossing inside the reported window.
* **refusal** — a parameter that both sets a switch time and acts inside a branch
  is rejected rather than answered with the jump alone.
* **no false positives** — an ``if()`` model with fixed thresholds, and a model
  with no clock at all, are left completely untouched.
"""

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder
from bngsim._exceptions import SensitivityUnsupportedError
from bngsim._switch_sensitivity import compute_switch_time_sens

# ─── Minimal model: counter T (dT/dt=1, T(0)=0 so `t` IS simulation time) and
# X with dX/dt = if(t>=sigma, k, 0). Analytic: X = k·max(0, t−sigma), so
# ∂X/∂sigma = −k and ∂X/∂k = max(0, t−sigma) past the switch. ─────────────────
SIGMA, K = 3.0, 2.0


def _minimal(sigma=SIGMA, k=K):
    b = ModelBuilder()
    b.add_parameter("sigma", sigma)
    b.add_parameter("k", k)
    b.add_parameter("rate_counter", 1.0)
    t_idx = b.add_species("T()", 0.0)
    x_idx = b.add_species("X()", 0.0)
    b.add_observable("t", [(t_idx, 1.0)])
    b.add_function("rate_X", "if((t>=sigma),k,0)")
    b.add_reaction([], [t_idx], "elementary", "rate_counter")
    b.add_reaction([], [x_idx], "functional", "rate_X")
    return bngsim.Model(_core=b.build()), x_idx


# ─── Lin2021 structure in miniature ──────────────────────────────────────────
# counter seeded at 1 (so the observable `t` reads 1 + t_sim, exactly the
# Lin2021 offset), two sequential crossings driven by a derived threshold, and a
# third threshold that the run never reaches.
#
#   t0 = 2, t_delta = 1, t_delta2 = 5;  sigma = t0+t_delta, tau1 = sigma+t_delta2
#   dA/dt = if(t>=t0, a, 0)
#   dB/dt = if((t>=sigma) && (t<tau1), b·A, 0)
#
# crossings land at t_sim = t0−1 = 1, sigma−1 = 2, tau1−1 = 7.
# For t in [sigma−1, tau1−1]:
#   A(t) = a·(t − t0 + 1)
#   B(t) = (a·b/2)·[(t − t0 + 1)² − t_delta²]
# and for t ≥ tau1−1, B freezes at (a·b/2)·[(t_delta + t_delta2)² − t_delta²]
# — which does not involve t0 at all, so ∂B/∂t0 must cancel to 0 across the
# three jumps. That cancellation is the sharpest check of the chain rule here.
T0, T_DELTA, T_DELTA2, A_RATE, B_RATE = 2.0, 1.0, 5.0, 1.0, 1.0
SEQ_PARAMS = ["t0", "t_delta", "t_delta2"]


def _sequential(t0=T0, t_delta=T_DELTA, t_delta2=T_DELTA2):
    b = ModelBuilder()
    b.add_parameter("t0", t0)
    b.add_parameter("t_delta", t_delta)
    b.add_parameter("t_delta2", t_delta2)
    b.add_parameter("sigma", t0 + t_delta, "t0+t_delta", True)
    b.add_parameter("tau1", t0 + t_delta + t_delta2, "(t0+t_delta)+t_delta2", True)
    b.add_parameter("a", A_RATE)
    b.add_parameter("b", B_RATE)
    b.add_parameter("rate_counter", 1.0)
    c_idx = b.add_species("counter()", 1.0)
    a_idx = b.add_species("A()", 0.0)
    b_idx = b.add_species("B()", 0.0)
    b.add_observable("t", [(c_idx, 1.0)])
    b.add_observable("Aobs", [(a_idx, 1.0)])
    b.add_function("rate_A", "if((t>=t0),a,0)")
    b.add_function("rate_B", "if(((t>=sigma)&&(t<tau1)),b*Aobs,0)")
    b.add_reaction([], [c_idx], "elementary", "rate_counter")
    b.add_reaction([], [a_idx], "functional", "rate_A")
    b.add_reaction([], [b_idx], "functional", "rate_B")
    return bngsim.Model(_core=b.build()), a_idx, b_idx


def _seq_closed_form(t, t0=T0, t_delta=T_DELTA, t_delta2=T_DELTA2):
    """(A, B) at simulation time *t* for the sequential model."""
    a, bb = A_RATE, B_RATE
    A = a * max(0.0, t - (t0 - 1.0))
    t_eff = min(t, t0 + t_delta + t_delta2 - 1.0)
    if t_eff <= t0 + t_delta - 1.0:
        B = 0.0
    else:
        B = (a * bb / 2.0) * ((t_eff - t0 + 1.0) ** 2 - t_delta**2)
    return A, B


def _sens(model, params, t_end, n_points=26, rtol=1e-10, atol=1e-12):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params, codegen=True)
    r = sim.run(
        sample_times=list(np.linspace(0.0, t_end, n_points)),
        rtol=rtol,
        atol=atol,
        max_steps=10**7,
    )
    return np.asarray(r.species), np.asarray(r.sensitivities)


class TestMinimalSwitch:
    """dX/dt = if(t>=sigma, k, 0) — the smallest model with a switch time."""

    def test_switch_time_gradient_matches_closed_form(self):
        _, S = _sens(_minimal()[0], ["sigma", "k"], 5.0)
        # The whole gradient is the jump: X = k(t−sigma) past the switch.
        assert S[-1, 1, 0] == pytest.approx(-K, abs=1e-9)
        assert S[-1, 1, 1] == pytest.approx(5.0 - SIGMA, rel=1e-6)
        # The clock itself does not depend on the switch time.
        assert S[-1, 0, 0] == pytest.approx(0.0, abs=1e-12)

    def test_matches_central_finite_difference(self):
        """FD reference on plain runs, which never had the stall."""

        def x_end(sigma, k):
            m, x_idx = _minimal(sigma=sigma, k=k)
            r = bngsim.Simulator(m, method="ode").run(
                sample_times=list(np.linspace(0.0, 5.0, 26)),
                rtol=1e-11,
                atol=1e-13,
                max_steps=10**7,
            )
            return np.asarray(r.species)[-1, x_idx]

        h = 1e-6
        fd_sigma = (x_end(SIGMA + h, K) - x_end(SIGMA - h, K)) / (2 * h)
        fd_k = (x_end(SIGMA, K + h) - x_end(SIGMA, K - h)) / (2 * h)
        _, S = _sens(_minimal()[0], ["sigma", "k"], 5.0)
        assert S[-1, 1, 0] == pytest.approx(fd_sigma, abs=1e-5)
        assert S[-1, 1, 1] == pytest.approx(fd_k, abs=1e-5)

    @pytest.mark.parametrize("rtol", [1e-8, 1e-10, 1e-12])
    def test_answer_does_not_drift_with_tolerance(self, rtol):
        """The pre-#48 failure scaled as √rtol — the FD probe's step size.

        A jump read at CVODES' *perturbed* parameter point scales the whole
        column by (1 ∓ √rtol), so a tolerance sweep is what distinguishes a
        correct jump from a plausible-looking one.
        """
        _, S = _sens(_minimal()[0], ["sigma", "k"], 5.0, rtol=rtol, atol=rtol * 1e-2)
        assert S[-1, 1, 0] == pytest.approx(-K, abs=1e-9)

    def test_zero_before_the_crossing(self):
        """Stopping short of the switch is the one case that always worked."""
        _, S = _sens(_minimal()[0], ["sigma", "k"], 2.5)
        assert S[-1, :, :] == pytest.approx(np.zeros_like(S[-1]), abs=1e-12)

    def test_sensitivity_recorded_at_every_sample(self):
        """The column is right-continuous: a sample landing on t* reads s⁺.

        This grid puts a sample exactly on the crossing (t=3.0). The jump is
        applied when the integrator stops there, before the point is recorded,
        so that sample carries the post-jump value — while the *state* stays
        continuous through the crossing.
        """
        X, S = _sens(_minimal()[0], ["sigma", "k"], 5.0, n_points=11)
        assert np.all(np.isfinite(S))
        for i, t in enumerate(np.linspace(0.0, 5.0, 11)):
            expected = -K if t >= SIGMA else 0.0
            assert S[i, 1, 0] == pytest.approx(expected, abs=1e-8)
            assert X[i, 1] == pytest.approx(K * max(0.0, t - SIGMA), abs=1e-8)


class TestSequentialAndDerivedThresholds:
    """Two crossings, a derived threshold, and one threshold never reached."""

    def test_forward_solution_matches_closed_form(self):
        X, _ = _sens(_sequential()[0], SEQ_PARAMS, 5.0)
        A_exp, B_exp = _seq_closed_form(5.0)
        assert X[-1, 1] == pytest.approx(A_exp, rel=1e-8)
        assert X[-1, 2] == pytest.approx(B_exp, rel=1e-8)

    def test_chain_rule_attributes_jumps_to_the_fitted_primaries(self):
        """t0 moves both crossings; t_delta moves only sigma."""
        _, S = _sens(_sequential()[0], SEQ_PARAMS, 5.0)
        t = 5.0
        # A only sees the t0 switch: A = a·(t − t0 + 1).
        assert S[-1, 1, 0] == pytest.approx(-A_RATE, abs=1e-7)
        assert S[-1, 1, 1] == pytest.approx(0.0, abs=1e-7)
        # B = (a·b/2)·[(t − t0 + 1)² − t_delta²].
        assert S[-1, 2, 0] == pytest.approx(-A_RATE * B_RATE * (t - T0 + 1.0), rel=1e-6)
        assert S[-1, 2, 1] == pytest.approx(-A_RATE * B_RATE * T_DELTA, rel=1e-6)

    def test_uncrossed_threshold_column_is_exactly_zero(self):
        """tau1 is never reached by t=5, so t_delta2 cannot move anything."""
        _, S = _sens(_sequential()[0], SEQ_PARAMS, 5.0)
        assert np.max(np.abs(S[-1, :, 2])) == 0.0

    def test_third_crossing_and_the_t0_cancellation(self):
        """Past tau1, B no longer depends on t0 — the two jumps must cancel."""
        t_end = 10.0
        X, S = _sens(_sequential()[0], SEQ_PARAMS, t_end, n_points=41)
        A_exp, B_exp = _seq_closed_form(t_end)
        assert X[-1, 2] == pytest.approx(B_exp, rel=1e-7)
        # B(t ≥ tau1−1) = (a·b/2)·[(t_delta+t_delta2)² − t_delta²]: no t0.
        assert S[-1, 2, 0] == pytest.approx(0.0, abs=1e-5)
        assert S[-1, 2, 1] == pytest.approx(A_RATE * B_RATE * T_DELTA2, rel=1e-5)
        assert S[-1, 2, 2] == pytest.approx(A_RATE * B_RATE * (T_DELTA + T_DELTA2), rel=1e-5)
        # A has no upper gate, so its t0 column is unchanged.
        assert S[-1, 1, 0] == pytest.approx(-A_RATE, abs=1e-6)

    def test_matches_central_finite_difference(self):
        def end_state(**kw):
            m, _a, _b = _sequential(**kw)
            r = bngsim.Simulator(m, method="ode").run(
                sample_times=list(np.linspace(0.0, 5.0, 26)),
                rtol=1e-11,
                atol=1e-13,
                max_steps=10**7,
            )
            return np.asarray(r.species)[-1]

        _, S = _sens(_sequential()[0], SEQ_PARAMS, 5.0)
        for j, pname in enumerate(SEQ_PARAMS):
            nominal = {"t0": T0, "t_delta": T_DELTA, "t_delta2": T_DELTA2}[pname]
            h = 1e-6 * max(1.0, abs(nominal))
            hi = end_state(**{pname: nominal + h})
            lo = end_state(**{pname: nominal - h})
            fd = (hi - lo) / (2 * h)
            assert S[-1, :, j] == pytest.approx(fd, abs=2e-5)


class TestDetection:
    """compute_switch_time_sens: what gets a record, and what gets pinned."""

    def test_minimal_model_records_one_crossing(self):
        core = _minimal()[0]._core
        records, pinned = compute_switch_time_sens(core, ["sigma", "k"], 0.0, 5.0)
        assert len(records) == 1
        t_star, clock_idx0, threshold, dtstar = records[0][:4]
        assert t_star == pytest.approx(SIGMA)
        assert clock_idx0 == 0  # the counter species
        assert threshold == pytest.approx(SIGMA)
        assert dtstar == [1.0, 0.0]
        assert pinned == [0]  # `sigma` only; `k` is an in-branch rate constant

    def test_clock_offset_is_applied(self):
        """counter(0)=1 ⇒ a threshold of `t0` fires at t_sim = t0 − 1."""
        core = _sequential()[0]._core
        records, _ = compute_switch_time_sens(core, SEQ_PARAMS, 0.0, 5.0)
        assert [r[0] for r in records] == pytest.approx([T0 - 1.0, T0 + T_DELTA - 1.0])
        assert [r[3] for r in records] == [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]

    def test_crossings_outside_the_window_are_dropped(self):
        core = _sequential()[0]._core
        assert len(compute_switch_time_sens(core, SEQ_PARAMS, 0.0, 5.0)[0]) == 2
        assert len(compute_switch_time_sens(core, SEQ_PARAMS, 0.0, 10.0)[0]) == 3
        assert compute_switch_time_sens(core, SEQ_PARAMS, 0.0, 0.5)[0] == []

    def test_non_switch_parameters_produce_nothing(self):
        """`a` and `b` scale branches; they do not move a crossing."""
        core = _sequential()[0]._core
        records, pinned = compute_switch_time_sens(core, ["a", "b"], 0.0, 10.0)
        assert records == []
        assert pinned == []

    def test_model_without_a_clock_is_untouched(self):
        b = ModelBuilder()
        b.add_parameter("k", 0.5)
        s = b.add_species("S", 100.0)
        b.add_reaction([s], [], "elementary", "k")
        b.add_observable("Sobs", [(s, 1.0)])
        core = b.build()
        assert compute_switch_time_sens(core, ["k"], 0.0, 10.0) == ([], [])

    def test_fixed_threshold_needs_no_jump(self):
        """`if(t>=3, ...)` with a literal threshold moves with no parameter."""
        b = ModelBuilder()
        b.add_parameter("k", 2.0)
        b.add_parameter("rate_counter", 1.0)
        t_idx = b.add_species("T()", 0.0)
        x_idx = b.add_species("X()", 0.0)
        b.add_observable("t", [(t_idx, 1.0)])
        b.add_function("rate_X", "if((t>=3),k,0)")
        b.add_reaction([], [t_idx], "elementary", "rate_counter")
        b.add_reaction([], [x_idx], "functional", "rate_X")
        core = b.build()
        assert compute_switch_time_sens(core, ["k"], 0.0, 5.0) == ([], [])


class TestDualRoleParameter:
    """A switch time that also acts in-branch (`if(t>=sigma, sigma*2, 0)`).

    Its gradient is the interior variational term PLUS the crossing jump, not
    the jump alone. Whether bngsim delivers that sum is decided by the
    sensitivity RHS the run installs (issue #358):

    * with an analytic RHS, ``bngsim_dfdp`` already carries the in-branch
      ``∂f/∂sigma`` (the clean ``Piecewise`` derivative, no boundary delta), so
      the two terms sum to the correct total — accepted;
    * on CVODES' difference quotient there is no ``bngsim_dfdp``, and pinning
      holds the in-branch term at a wrong 0 — refused, as before.

    Closed form for ``X = sigma*2*(t-sigma)`` past the switch:
    ``∂X/∂sigma = 2t - 4*sigma`` = ``-2`` at ``t=5, sigma=3`` (interior ``+4``,
    jump ``-6``).
    """

    CLOSED_FORM_DXDSIGMA = 2 * 5.0 - 4 * SIGMA  # = -2.0 at t_end = 5

    @staticmethod
    def _dual_role():
        b = ModelBuilder()
        b.add_parameter("sigma", SIGMA)
        b.add_parameter("rate_counter", 1.0)
        t_idx = b.add_species("T()", 0.0)
        x_idx = b.add_species("X()", 0.0)
        b.add_observable("t", [(t_idx, 1.0)])
        # `sigma` sets the crossing AND scales the post-switch rate, so the jump
        # is not the whole gradient — there is also a non-zero in-branch ∂f/∂p.
        b.add_function("rate_X", "if((t>=sigma),sigma*2,0)")
        b.add_reaction([], [t_idx], "elementary", "rate_counter")
        b.add_reaction([], [x_idx], "functional", "rate_X")
        return bngsim.Model(_core=b.build())

    # ── The detection helper: refuse by default, accept on the analytic path ──

    def test_detection_refuses_without_an_analytic_rhs(self):
        """The conservative default: a caller that cannot tell which sensitivity
        RHS the run installs keeps the pre-#358 refusal."""
        with pytest.raises(SensitivityUnsupportedError, match="if\\(\\) switch time AND appears"):
            compute_switch_time_sens(self._dual_role()._core, ["sigma"], 0.0, 5.0)

    def test_detection_accepts_with_an_analytic_rhs(self):
        """Told the run carries an analytic sensitivity RHS, the same parameter
        is accepted: its in-branch ∂f/∂p is `bngsim_dfdp`'s to supply, and the
        detector returns the crossing record for the jump to add on top."""
        records, pinned = compute_switch_time_sens(
            self._dual_role()._core, ["sigma"], 0.0, 5.0, has_analytic_sens_rhs=True
        )
        assert len(records) == 1  # the single crossing at t* = sigma = 3
        t_star, _clock, threshold, dtstar = records[0][:4]
        assert t_star == pytest.approx(SIGMA)
        assert threshold == pytest.approx(SIGMA)
        assert dtstar == [pytest.approx(1.0)]  # ∂t*/∂sigma = 1
        assert pinned  # sigma is still pinned (inert on the analytic path)

    # ── Through the public API ──────────────────────────────────────────────

    def test_run_on_the_analytic_path_sums_the_two_terms(self):
        """The whole point of #358: codegen=True installs an analytic sensitivity
        RHS, so the run succeeds and lands on the closed form rather than
        refusing. Checked across tolerances — a jump read at a perturbed switch
        would scale the column by (1 ∓ √rtol)."""
        for rtol in (1e-8, 1e-10, 1e-12):
            sim = bngsim.Simulator(
                self._dual_role(), method="ode", sensitivity_params=["sigma"], codegen=True
            )
            r = sim.run(
                sample_times=list(np.linspace(0.0, 5.0, 26)),
                rtol=rtol,
                atol=rtol * 1e-2,
                max_steps=10**7,
            )
            x_idx = 1  # X() is the second species
            got = np.asarray(r.sensitivities)[-1, x_idx, 0]
            assert got == pytest.approx(self.CLOSED_FORM_DXDSIGMA, abs=1e-8)

    def test_run_matches_central_finite_difference(self):
        """FD reference on plain runs (which never had the stall), the oracle #358
        asked for on a model where a parameter is both a switch time and an
        in-branch coefficient."""

        def x_end(sigma):
            b = ModelBuilder()
            b.add_parameter("sigma", sigma)
            b.add_parameter("rate_counter", 1.0)
            t_idx = b.add_species("T()", 0.0)
            x_idx = b.add_species("X()", 0.0)
            b.add_observable("t", [(t_idx, 1.0)])
            b.add_function("rate_X", "if((t>=sigma),sigma*2,0)")
            b.add_reaction([], [t_idx], "elementary", "rate_counter")
            b.add_reaction([], [x_idx], "functional", "rate_X")
            m = bngsim.Model(_core=b.build())
            r = bngsim.Simulator(m, method="ode").run(
                sample_times=list(np.linspace(0.0, 5.0, 26)),
                rtol=1e-12,
                atol=1e-14,
                max_steps=10**7,
            )
            return np.asarray(r.species)[-1, x_idx]

        h = 1e-6
        fd = (x_end(SIGMA + h) - x_end(SIGMA - h)) / (2 * h)
        sim = bngsim.Simulator(
            self._dual_role(), method="ode", sensitivity_params=["sigma"], codegen=True
        )
        r = sim.run(
            sample_times=list(np.linspace(0.0, 5.0, 26)), rtol=1e-11, atol=1e-13, max_steps=10**7
        )
        got = np.asarray(r.sensitivities)[-1, 1, 0]
        assert got == pytest.approx(fd, abs=1e-5)

    def test_run_on_the_difference_quotient_still_refuses(self, monkeypatch):
        """The refusal has to STAY where it is load-bearing. Forcing the
        Functional model back onto CVODES' difference quotient (the pre-#67
        hatch) restores the exact conditions #358 does not narrow: no
        `bngsim_dfdp`, and pinning holds the in-branch term at a wrong 0."""
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        sim = bngsim.Simulator(
            self._dual_role(), method="ode", sensitivity_params=["sigma"], codegen=True
        )
        with pytest.raises(SensitivityUnsupportedError, match="issue #358"):
            sim.run(t_span=(0.0, 5.0), n_points=11, rtol=1e-10, atol=1e-12)

    # ── The refusal is TYPED (issue #320) ───────────────────────────────────
    #
    # A declared capability gap, not a bug. Before it was typed, the amici_parity
    # sweep scored it EXCEPTION ("AMICI ran and bngsim broke"). The refusal still
    # inherits `ValueError`, the back-compat contract.

    def test_the_detection_refusal_is_typed(self):
        with pytest.raises(SensitivityUnsupportedError):
            compute_switch_time_sens(self._dual_role()._core, ["sigma"], 0.0, 5.0)
        with pytest.raises(ValueError):  # the base it has always raised
            compute_switch_time_sens(self._dual_role()._core, ["sigma"], 0.0, 5.0)

    def test_it_is_not_typed_as_the_codegen_refusal(self):
        """Distinct declared gaps share a type but must keep distinct messages —
        'split the parameter' and 'the rate law is not differentiable' call for
        different user actions."""
        with pytest.raises(SensitivityUnsupportedError) as ei:
            compute_switch_time_sens(self._dual_role()._core, ["sigma"], 0.0, 5.0)
        assert "closed form" not in str(ei.value)
        assert "Split the parameter" in str(ei.value)


class TestClockThresholdBehindAFunction:
    """A clock threshold hidden behind a function reference is still detected.

    ``compute_switch_time_sens`` inlines function references before scanning,
    exactly as the issue #150 state detector already does — "the gate and the
    detector must scan the SAME text". Here the crossing is written
    ``if((clk>=0), k, 0)`` with ``clk = t - sigma``: the raw body's atom is
    ``clk>=0``, which names no clock and was silently skipped, so the crossing
    (and its jump) went missing and the switch-time column came back a wrong
    zero. This is the pattern BIOMD0000001007's ``heav_x = if((x<0), 0, ...)``
    over ``x = time() - ModelValue_27`` carries, whose ``Tdam`` column was short
    by exactly this crossing.
    """

    @staticmethod
    def _model():
        b = ModelBuilder()
        b.add_parameter("sigma", SIGMA)
        b.add_parameter("k", K)
        b.add_parameter("rate_counter", 1.0)
        t_idx = b.add_species("T()", 0.0)
        x_idx = b.add_species("X()", 0.0)
        b.add_observable("t", [(t_idx, 1.0)])
        b.add_function("clk", "t-sigma")  # the clock, one function-reference away
        b.add_function("rate_X", "if((clk>=0),k,0)")
        b.add_reaction([], [t_idx], "elementary", "rate_counter")
        b.add_reaction([], [x_idx], "functional", "rate_X")
        return bngsim.Model(_core=b.build())

    def test_the_crossing_is_detected(self):
        records, pinned = compute_switch_time_sens(self._model()._core, ["sigma", "k"], 0.0, 5.0)
        assert len(records) == 1
        t_star, _clock, threshold, dtstar = records[0][:4]
        assert t_star == pytest.approx(SIGMA)
        assert threshold == pytest.approx(SIGMA)
        assert dtstar[0] == pytest.approx(1.0)  # ∂t*/∂sigma = 1, through `clk`
        assert dtstar[1] == pytest.approx(0.0)  # k does not move the crossing
        assert pinned == [0]  # sigma pinned; k is not a switch-time parameter

    def test_the_gradient_is_the_jump_not_a_wrong_zero(self):
        """Without the inline, the crossing is invisible and ∂X/∂sigma comes back
        0; with it, the jump lands the closed form ``X = k·max(0, t-sigma)``."""
        sim = bngsim.Simulator(
            self._model(), method="ode", sensitivity_params=["sigma", "k"], codegen=True
        )
        r = sim.run(
            sample_times=list(np.linspace(0.0, 5.0, 26)), rtol=1e-10, atol=1e-12, max_steps=10**7
        )
        S = np.asarray(r.sensitivities)
        assert S[-1, 1, 0] == pytest.approx(-K, abs=1e-8)  # ∂X/∂sigma = -k
        assert S[-1, 1, 1] == pytest.approx(5.0 - SIGMA, rel=1e-6)  # ∂X/∂k = t - sigma


class TestUncompensatedCrossingGuard:
    """A switch time that also gates a crossing nothing brackets is refused.

    Pinning a switch-time parameter (issue #48) holds it nominal against CVODES'
    difference-quotient probe, which is safe only when *every* crossing it moves
    is compensated. If the parameter ALSO reads a condition nobody brackets,
    pinning holds that crossing's dependence on it at a wrong 0 —
    MODEL1708310001's ``cycle_int`` measured ∂/∂cycle_int = 0 against a finite
    difference of ~19 — so the run is refused rather than answered with a silent
    zero. The refusal stands whether or not an analytic RHS is claimed: an
    uncompensated crossing is exactly what puts the model on the difference
    quotient (issue #68), so there is no analytic path here to sum anything on.

    The witness used to be a plain ``floor()``-periodic pulse, and issue #436
    retired that one by compensating it: ``period`` sets the clock switch
    ``time() < period*4`` AND drives ``rem(time(), period) >= 0.5``, and both of
    those are now crossings the detector places. So the witness moved one level
    deeper, to a remainder OF a remainder — the shape MODEL1708310001 itself
    writes for "one dose a day inside a twenty-one day cycle" — which nothing
    enumerates. ``TestAFloorPulseIsCompensatedNotRefused`` below holds the
    retired case, because a guard that has stopped firing for the right reason
    still has to be shown to have stopped.
    """

    #: The inner remainder, written once so the nesting below stays readable.
    _REM = "(time()-(floor((time()/period))*period))"

    @classmethod
    def _model(cls, condition: str):
        b = ModelBuilder()
        b.add_parameter("period", 3.0)
        b.add_parameter("amp", 1.0)
        b.add_parameter("rate_counter", 1.0)
        t_idx = b.add_species("T()", 0.0)
        x_idx = b.add_species("X()", 0.0)
        b.add_observable("t", [(t_idx, 1.0)])
        b.add_function("rate_X", f"if((time()<(period*4)),if(({condition}),amp,0),0)")
        b.add_reaction([], [t_idx], "elementary", "rate_counter")
        b.add_reaction([], [x_idx], "functional", "rate_X")
        return bngsim.Model(_core=b.build())

    @classmethod
    def _nested(cls):
        """A daily dose inside the cycle: ``rem(rem(time(), period), 1) >= 0.5``."""
        return cls._model(f"({cls._REM}-(floor(({cls._REM}/1.0))*1.0))>=0.5")

    @pytest.mark.parametrize("has_analytic", [False, True])
    def test_a_switch_time_that_also_gates_an_unbracketed_crossing_is_refused(self, has_analytic):
        with pytest.raises(SensitivityUnsupportedError, match="nothing compensates"):
            compute_switch_time_sens(
                self._nested()._core,
                ["period", "amp"],
                0.0,
                20.0,
                has_analytic_sens_rhs=has_analytic,
            )

    def test_a_parameter_that_only_gates_the_pulse_is_not_a_switch_time(self):
        """``amp`` moves no crossing — it scales the pulse — so it is neither a
        switch time nor caught by the guard, and requesting it alone is a no-op
        (the switch-time detector leaves it to the ordinary sensitivity path)."""
        records, pinned = compute_switch_time_sens(self._nested()._core, ["amp"], 0.0, 20.0)
        assert records == [] and pinned == []


class TestAFloorPulseIsCompensatedNotRefused:
    """Issue #436, on the model the guard above was written around.

    ``period`` does two things at once here: it sets the clock switch that ends
    the treatment at ``time() = period*4``, and it is the period of the pulse
    ``rem(time(), period) >= 0.5`` that runs until then. That combination is what
    used to make the model unanswerable, and it is now the reason it is a good
    test: the pulse's fourth off-edge falls at ``4*period`` and so lands on the
    clock switch exactly, both move with ``period`` by the same 4, and the two
    have to be read as one crossing rather than two coinciding ones (issue #375).
    """

    @staticmethod
    def _model(period: float = 3.0, amp: float = 1.0):
        b = ModelBuilder()
        b.add_parameter("period", period)
        b.add_parameter("amp", amp)
        b.add_parameter("rate_counter", 1.0)
        t_idx = b.add_species("T()", 0.0)
        x_idx = b.add_species("X()", 0.0)
        b.add_observable("t", [(t_idx, 1.0)])
        b.add_function(
            "rate_X",
            "if((time()<(period*4)),if(((time()-(floor((time()/period))*period))>=0.5),amp,0),0)",
        )
        b.add_reaction([], [t_idx], "elementary", "rate_counter")
        b.add_reaction([], [x_idx], "functional", "rate_X")
        return bngsim.Model(_core=b.build())

    def test_the_run_is_no_longer_refused_and_the_edges_are_placed(self):
        records, pinned = compute_switch_time_sens(
            self._model()._core, ["period", "amp"], 0.0, 20.0, has_analytic_sens_rhs=True
        )
        # On at 0.5, 3.5, 6.5, … and off at 3, 6, 9, …; the on-edge of the first
        # period sits at a fixed 0.5 and moves for nobody, so it emits nothing.
        assert [r.t_star for r in records] == pytest.approx(
            [3.0, 3.5, 6.0, 6.5, 9.0, 9.5, 12.0, 12.5, 15.0, 15.5, 18.0, 18.5]
        )
        # ∂t*/∂period is k for the off-edge at k*period and for the on-edge just
        # after it; ``amp`` moves none of them.
        assert [r.dtstar[0] for r in records] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        assert [r.dtstar[1] for r in records] == [0.0] * 12
        assert pinned == [0]

    def test_the_pulse_edge_and_the_clock_switch_at_12_are_one_record(self):
        """``period*4`` is 12 and the pulse's fourth boundary is 12 too, and both
        move with ``period`` by 4. Two records there would charge ``period``
        twice for one branch flip; the detector folds them into one because the
        crossings agree on ∂t*/∂primary as well as on the instant (issue #375)."""
        records, _pinned = compute_switch_time_sens(
            self._model()._core, ["period"], 0.0, 20.0, has_analytic_sens_rhs=True
        )
        at_twelve = [r for r in records if r.t_star == pytest.approx(12.0)]
        assert len(at_twelve) == 1
        assert at_twelve[0].dtstar == [4.0]
        assert at_twelve[0].isolate_param_idx0 == []

    def test_both_columns_match_a_central_difference(self):
        """``X`` accumulates ``amp`` for the ``period − 0.5`` of every period the
        treatment lasts, so ``X(20) = 4*amp*(period − 0.5)`` and the two columns
        are 4*amp = 4 and 4*(period − 0.5) = 10 exactly.

        ``max_step`` is not a tolerance knob here, it is what makes the reference
        trustworthy: a plain run of this model steps clean over the half-unit gaps
        between pulses and reports ``X(20) = 11.5``, the answer for a pulse that
        never turns off. The sensitivity run does not have that problem — the
        crossing records are stop times — so a finite difference of two plain runs
        would be comparing against a trajectory the solver got wrong."""
        sample_times = list(np.arange(0.0, 20.001, 0.5) + 0.13)

        def run(sens=None, **overrides):
            sim = bngsim.Simulator(
                self._model(**overrides), method="ode", sensitivity_params=sens
            )
            return sim.run(
                sample_times=sample_times, rtol=1e-11, atol=1e-13, max_step=0.01
            )

        analytic = np.asarray(run(sens=["period", "amp"]).sensitivities)
        assert analytic[-1, 1, 0] == pytest.approx(4.0, rel=1e-6)
        assert analytic[-1, 1, 1] == pytest.approx(10.0, rel=1e-6)
        for col, (name, nominal, step) in enumerate([("period", 3.0, 3e-4), ("amp", 1.0, 1e-4)]):
            up = np.asarray(run(**{name: nominal + step}).species)
            down = np.asarray(run(**{name: nominal - step}).species)
            fd = (up - down) / (2.0 * step)
            scale = float(np.max(np.abs(fd)))
            np.testing.assert_allclose(
                analytic[:, :, col], fd, rtol=1e-4, atol=1e-5 * scale
            )


# ─── SBML shape ──────────────────────────────────────────────────────────────
# The same idiom SBML models use: `piecewise(kin, time >= T0, 0)` as a kinetic
# law, with the onset T0 a fitted parameter. Two things differ from the BNGL
# models above and both are load-bearing:
#
#   * the clock is the literal `time` csymbol, not a counter species; and
#   * the loader registers a GH #72 *discontinuity root* at exactly this
#     threshold, so the crossing comes back as CV_ROOT_RETURN rather than
#     CV_TSTOP_RETURN — and CVODE only auto-clears its stop time on the latter.
#
# Closed form for t ≥ T0: X = (kin/d)·(1 − exp(−d·(t − T0))).
SBML_TIME_SWITCH = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="onset">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kin" value="3" constant="true"/>
      <parameter id="T0" value="2" constant="true"/>
      <parameter id="d" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="prod" reversible="false">
        <listOfProducts>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <piecewise>
              <piece>
                <ci>kin</ci>
                <apply><geq/>
                  <csymbol encoding="text"
                    definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <ci>T0</ci></apply>
              </piece>
              <otherwise><cn>0</cn></otherwise>
            </piecewise>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>d</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

SBML_KIN, SBML_T0, SBML_D, SBML_T_END = 3.0, 2.0, 0.5, 6.0
SBML_PARAMS = ["T0", "kin", "d"]


class TestSbmlPiecewiseInTime:
    """SBML `piecewise(..., time >= T0, ...)` — literal-time clock + a root."""

    @staticmethod
    def _model(**overrides):
        m = bngsim.Model.from_sbml_string(SBML_TIME_SWITCH)
        for name, value in overrides.items():
            m.set_param(name, value)
        return m

    def test_literal_time_clock_is_detected(self):
        """No counter species — the clock is `time()`, flagged by index −1."""
        core = self._model()._core
        records, pinned = compute_switch_time_sens(core, SBML_PARAMS, 0.0, SBML_T_END)
        assert len(records) == 1
        t_star, clock_idx0, threshold, dtstar = records[0][:4]
        assert t_star == pytest.approx(SBML_T0)
        assert clock_idx0 == -1  # literal simulation time, not a species
        assert threshold == pytest.approx(SBML_T0)
        assert dtstar == [1.0, 0.0, 0.0]
        # `pinned` carries MODEL parameter indices (what the solver perturbs),
        # not positions in the requested column order.
        assert pinned == [list(core.param_names).index("T0")]

    def test_crosses_a_discontinuity_root_without_a_stale_stop_time(self):
        """Regression: the loader puts a root on the very time we stop at.

        CVODE clears its stop time only on CV_TSTOP_RETURN. With the root
        present the crossing returns CV_ROOT_RETURN instead, so an unmanaged
        stop time stays armed at a moment now in the past and the next CVode()
        call fails outright with "tstop is behind current t".
        """
        m = self._model()
        assert m._core.n_discontinuity_triggers == 1
        assert m._core.n_events == 0
        _, S = _sens(m, SBML_PARAMS, SBML_T_END, n_points=25)
        assert np.all(np.isfinite(S))

    def test_onset_time_gradient_matches_closed_form(self):
        _, S = _sens(self._model(), SBML_PARAMS, SBML_T_END, n_points=25)
        dt = SBML_T_END - SBML_T0
        expected = {
            # X = (kin/d)(1 − e^{−d(t−T0)})
            "T0": -SBML_KIN * np.exp(-SBML_D * dt),
            "kin": (1 - np.exp(-SBML_D * dt)) / SBML_D,
            "d": SBML_KIN
            * ((np.exp(-SBML_D * dt) - 1) / SBML_D**2 + dt * np.exp(-SBML_D * dt) / SBML_D),
        }
        for j, p in enumerate(SBML_PARAMS):
            assert S[-1, 0, j] == pytest.approx(expected[p], rel=1e-6)

    def test_matches_central_finite_difference(self):
        def x_end(**overrides):
            r = bngsim.Simulator(self._model(**overrides), method="ode").run(
                sample_times=list(np.linspace(0.0, SBML_T_END, 25)),
                rtol=1e-11,
                atol=1e-13,
                max_steps=10**7,
            )
            return np.asarray(r.species)[-1, 0]

        _, S = _sens(self._model(), SBML_PARAMS, SBML_T_END, n_points=25)
        nominal = {"T0": SBML_T0, "kin": SBML_KIN, "d": SBML_D}
        for j, p in enumerate(SBML_PARAMS):
            h = 1e-6 * max(1.0, abs(nominal[p]))
            fd = (x_end(**{p: nominal[p] + h}) - x_end(**{p: nominal[p] - h})) / (2 * h)
            assert S[-1, 0, j] == pytest.approx(fd, rel=1e-4)


# ─── Issue #82: the clock must land ON its threshold at the crossing ─────────
# The stop time puts t exactly on t*, but the `if()` condition reads the COUNTER
# SPECIES, and that counter is integrated: pre-#82 it came back 1–2e-14 BELOW the
# threshold it defines. The restart at the crossing then re-entered on the
# *before* branch, so the discontinuity landed inside the first step after the
# restart — the one thing the stop time exists to prevent — and CVODES failed the
# error test at every step size down to ~1e-10 before returning CV_ERR_FAILURE.
#
# Two guards below. The first is the invariant and is deterministic at every
# threshold. The second is an end-to-end knife-edge point: a frozen pre-onset
# phase, a wide magnitude spread (S ~ 1e7 against I = 1), and a second crossing
# that moves with the same parameter — the Lin2021 `nyc_multiphase` shape that
# lost 25% of otherwise-integrable fit candidates to this.
SEIR_S0 = 1.0e7


def _seir_lite(t0, t_delta, beta=2.0, lam=0.1, p0=0.9):
    b = ModelBuilder()
    b.add_parameter("t0", t0)
    b.add_parameter("t_delta", t_delta)
    b.add_parameter("sigma", t0 + t_delta, "t0+t_delta", True)
    b.add_parameter("beta", beta)
    b.add_parameter("lam", lam)
    b.add_parameter("p0", p0)
    b.add_parameter("S0", SEIR_S0)
    b.add_parameter("kL", 0.9)
    b.add_parameter("cI", 0.12)
    b.add_parameter("rate_counter", 1.0)
    c_idx = b.add_species("counter()", 1.0)
    sm_idx = b.add_species("S(state~M)", SEIR_S0)
    sp_idx = b.add_species("S(state~P)", 0.0)
    e_idx = b.add_species("E()", 0.0)
    i_idx = b.add_species("I()", 1.0)
    r_idx = b.add_species("R()", 0.0)
    b.add_observable("t", [(c_idx, 1.0)])
    b.add_observable("Iobs", [(i_idx, 1.0)])
    # Nothing at all happens before t0 — no transmission, no distancing — so the
    # restart at the crossing sizes h from an identically-zero RHS.
    b.add_function("phi", "if((t>=t0),Iobs,0)")
    b.add_function("k_inf", "(beta/S0)*phi")
    b.add_function("k_dist", "if((t>=sigma),lam*p0,0)")
    b.add_reaction([], [c_idx], "elementary", "rate_counter")
    b.add_reaction([sm_idx], [e_idx], "functional", "k_inf")
    b.add_reaction([sm_idx], [sp_idx], "functional", "k_dist")
    b.add_reaction([e_idx], [i_idx], "elementary", "kL")
    b.add_reaction([i_idx], [r_idx], "elementary", "cI")
    return bngsim.Model(_core=b.build()), c_idx


class TestClockLandsOnItsThreshold:
    """Issue #82: the crossing must resume on the after-branch, not one ulp short."""

    @pytest.mark.parametrize("sigma", [2.9, 3.7, 11.3, 17.9, 23.7, 29.3, 32.904353, 47.7])
    def test_counter_reaches_the_threshold_at_the_crossing(self, sigma):
        """The recorded clock at t* is at or above the threshold it defines.

        Pre-#82 this was *below* it at every one of these thresholds — by up to
        2.5e-14 — which is exactly what put the resumed integration on the wrong
        branch. Sampling lands on t* itself, so this reads the state CVODES
        restarts from.
        """
        model, x_idx = _minimal(sigma=sigma, k=2.0e6)
        clock_idx = 0  # T() is added first in _minimal
        t_star = sigma  # _minimal's counter starts at 0, so t* == sigma
        times = sorted({0.0, t_star, *np.linspace(0.0, sigma + 20.0, 41)})
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["sigma"], codegen=True)
        r = sim.run(sample_times=times, rtol=1e-8, atol=1e-8, max_steps=10**6)
        clock_at_crossing = np.asarray(r.species)[times.index(t_star), clock_idx]
        assert clock_at_crossing >= sigma
        # …and only by a rounding step: this is a correction, not a nudge.
        assert clock_at_crossing - sigma <= 4.0 * np.spacing(sigma)
        # The answer is untouched: X = k·max(0, t−sigma) ⇒ ∂X/∂sigma = −k.
        assert np.asarray(r.sensitivities)[-1, x_idx, 0] == pytest.approx(-2.0e6, rel=1e-9)


class TestWideSpreadSwitchOnIntegrates:
    """Issue #82 end-to-end: the point that used to die at the second crossing."""

    # t0=29.3 with the published Lin2021 t_delta is one of the isolated spikes:
    # pre-#82 it returned CV_ERR_FAILURE at t≈29 while the plain solve was fine,
    # and t_delta=3.0 at t0=23.7 is a second, unrelated one.
    @pytest.mark.parametrize(
        ("t0", "t_delta"), [(29.3, 0.072681), (23.7, 3.0), (32.831672, 0.072681)]
    )
    def test_switch_time_sensitivity_integrates(self, t0, t_delta):
        model, _ = _seir_lite(t0, t_delta)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["t0"], codegen=True)
        r = sim.run(
            sample_times=list(np.linspace(0.0, 120.0, 121)),
            rtol=1e-8,
            atol=1e-8,
            max_steps=10**6,
        )
        S = np.asarray(r.sensitivities)
        assert np.all(np.isfinite(S))
        assert np.abs(S).max() > 1.0e5  # the switch-on really is being felt

    def test_matches_central_finite_difference(self):
        """The gradient is right, not merely finite.

        rtol/atol stop at 1e-10 deliberately: the FD oracle's own *plain* solves
        step into the `if(t>=t0)` kink with no stop time to break the step, and
        below 1e-10 that collapses h (the issue #54 case) — a limit of the oracle,
        not of the answer under test. The 1e-9 relative floor below is that
        oracle's noise: rtol·max|y|/h with max|y| ~ 1e7 and h ~ 3e-5.
        """
        t0, t_delta, t_end, tol = 29.3, 0.072681, 120.0, 1e-10
        times = list(np.linspace(0.0, t_end, 121))

        def end_state(t0_value):
            r = bngsim.Simulator(_seir_lite(t0_value, t_delta)[0], method="ode").run(
                sample_times=times, rtol=tol, atol=tol, max_steps=10**7
            )
            return np.asarray(r.species)[-1]

        sim = bngsim.Simulator(
            _seir_lite(t0, t_delta)[0], method="ode", sensitivity_params=["t0"], codegen=True
        )
        S = np.asarray(
            sim.run(sample_times=times, rtol=tol, atol=tol, max_steps=10**7).sensitivities
        )
        h = 1e-6 * t0
        fd = (end_state(t0 + h) - end_state(t0 - h)) / (2 * h)
        floor = tol * SEIR_S0 / h
        # The counter row is not differentiable w.r.t. t0 and is exactly 0 in
        # both; compare the epidemic rows, which span 1 to 1e7.
        assert S[-1, 1:, 0] == pytest.approx(fd[1:], rel=1e-5, abs=10.0 * floor)


class TestNoRegression:
    """Models with no fitted switch time must be completely unaffected."""

    def test_plain_run_is_unchanged_by_the_feature(self):
        """No sensitivities requested ⇒ no detection, no stop times."""
        m, x_idx = _minimal()
        r = bngsim.Simulator(m, method="ode").run(
            sample_times=[0.0, 2.0, 4.0], rtol=1e-10, atol=1e-12, max_steps=10**7
        )
        assert np.asarray(r.species)[-1, x_idx] == pytest.approx(K * (4.0 - SIGMA), abs=1e-6)

    def test_in_branch_parameter_alone_needs_no_stop(self):
        """Requesting only `k` leaves the integration exactly as it was."""
        core = _minimal()[0]._core
        assert compute_switch_time_sens(core, ["k"], 0.0, 5.0) == ([], [])
        _, S = _sens(_minimal()[0], ["k"], 5.0)
        assert S[-1, 1, 0] == pytest.approx(5.0 - SIGMA, rel=1e-6)
