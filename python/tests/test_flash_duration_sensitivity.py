"""Issue #361: forward sensitivity w.r.t. a *flash-duration* parameter.

``BIOMD0000000326`` and ``BIOMD0000000578`` (photoreceptor flash models) drive a
stimulus with a **duration-normalised amplitude** over an ``if()`` window::

    mag       = flashMag / flashDur
    testflash = if((time >= flashDel) && (time <= flashDel + flashDur), mag, 0)

``flashDur`` is therefore an *impure* switch-time parameter (issue #358): it sets
the window's closing edge ``flashDel + flashDur`` **and** appears inside the
branch through ``mag``. Its analytic column is the sum of an in-branch term and
the crossing jump.

#361 reported the column as a "silent zero" — analytic ``0`` while a central
finite difference of the trajectory was not — measured with the shipped
``flashMag = 0``. Verification showed the analytic zero is **correct**, and the
issue's finite difference was a moving-root artefact:

* With ``flashMag = 0`` the branch value ``mag`` is identically zero, so
  ``testflash`` is zero on both sides of the window and the RHS does not depend on
  ``flashDur`` at all. The true trajectory sensitivity is exactly ``0``.
* The finite difference the issue reported was solver noise: on the real models
  ``max|x(flashDur+δ) − x(flashDur−δ)|`` collapses as the tolerance tightens
  (~0.1 at rtol 1e-6 down to ~5e-7 at rtol 1e-12) and the central quotient
  *diverges* as the step shrinks — the signature of dividing a fixed noise floor
  by ``2h``. The ``if()`` window boundary is a root whose position moves with
  ``flashDur``, so perturbing it reshuffles the adaptive step sequence.
* The emitter does **not** fold the term to zero at the current ``flashMag = 0``
  (the fear #361 raised): it emits ``-flashMag/flashDur**2`` gated by the window,
  symbolic in ``flashMag``, so the column becomes correct the moment ``flashMag``
  is non-zero — which #358 already made true.

These tests pin all of that on a minimal model whose closed form is exact, so a
regression toward either a dropped term (silent zero when ``flashMag != 0``) or a
folded constant (wrong after ``set_param``) fails loudly.

Coverage:

* **shipped zero amplitude** — ``flashMag = 0`` gives an analytic ``flashDur``
  column that is exactly zero, at every duration and tolerance. This is the
  correct answer, not the reported defect.
* **real effect** — ``flashMag != 0`` gives a column matching the closed form
  (in-branch term inside the window, the saltation jump cancelling it after) and
  a bounded-step central finite difference, with no drift across tolerances.
* **continuity in amplitude** — the column is exactly linear in ``flashMag`` and
  passes through the origin, so the shipped zero is the ``flashMag -> 0`` limit
  rather than a term the emitter dropped.
* **symbolic emission** — the ``flashDur`` sensitivity term references
  ``flashMag``'s parameter slot in the emitted C, so it is not folded to a
  constant at the current parameter value.
"""

import bngsim
import numpy as np
import pytest
from bngsim import _codegen
from bngsim._bngsim_core import ModelBuilder

# ─── Minimal duration-normalised flash ───────────────────────────────────────
# dX/dt = testflash, X(0) = 0, with the BIOMD0000000326/578 shape:
#   mag       = flashMag / flashDur         (amplitude normalised by duration)
#   testflash = if(del <= time <= del+dur, mag, 0)
# so X(t) is 0 before the window, ramps at rate `mag` inside it, and settles at
# mag*dur = flashMag afterwards (the "duration-normalised" total dose does not
# depend on the duration — which is exactly why ∂X/∂flashDur must return to 0
# past the window). `time()` (the SBML time csymbol) is the clock, matching the
# real models; a counter observable would desync from wall time whenever a run
# starts at t != 0.
FLASH_DEL, FLASH_DUR = 1.0, 0.4  # window [1.0, 1.4], comfortably resolvable
FLASH_MAG = 1.0  # a real (non-shipped) amplitude for the correctness checks


def _flash(flash_mag=FLASH_MAG, flash_dur=FLASH_DUR, flash_del=FLASH_DEL):
    b = ModelBuilder()
    b.add_parameter("flashMag", flash_mag)
    b.add_parameter("flashDur", flash_dur)
    b.add_parameter("flashDel", flash_del)
    x_idx = b.add_species("X()", 0.0)
    b.add_function("mag", "flashMag/flashDur")
    b.add_function(
        "testflash",
        "if(((time()>=flashDel)&&(time()<=(flashDel+flashDur))),mag,0)",
    )
    b.add_reaction([], [x_idx], "functional", "testflash")
    return bngsim.Model(_core=b.build()), x_idx


def _closed_form_dX_dflashDur(t, flash_mag, flash_dur=FLASH_DUR, flash_del=FLASH_DEL):
    """Exact ∂X/∂flashDur at simulation time *t*.

    X = 0 (t <= del); (flashMag/flashDur)(t − del) in the window; flashMag after.
    Differentiating each piece w.r.t. flashDur gives the in-branch term inside
    the window and 0 past it — past the window the saltation jump at the moving
    edge exactly cancels the accumulated in-branch term (X = flashMag is constant
    in flashDur there).
    """
    if t <= flash_del:
        return 0.0
    if t <= flash_del + flash_dur:
        return -flash_mag * (t - flash_del) / flash_dur**2
    return 0.0


# Sample times bracketing the window: before, four points inside, and two after.
_TS = [0.5, 1.1, 1.2, 1.3, 1.39, 1.6, 2.0]


def _analytic_flashDur_column(flash_mag, rtol=1e-11, atol=1e-13):
    """The analytic ∂X/∂flashDur column at the sample times, on a fresh sim."""
    model, x_idx = _flash(flash_mag=flash_mag)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["flashDur"])
    r = sim.run(sample_times=_TS, rtol=rtol, atol=atol, max_steps=10**7)
    return np.asarray(r.sensitivities)[:, x_idx, 0]


def _plain_X(flash_mag, flash_dur, max_step):
    """X at the sample times from a plain (no-sensitivity) run.

    ``max_step`` is required: with dX/dt = 0 outside the window the adaptive
    integrator would otherwise step clear over a narrow pulse (the sensitivity
    run resolves it because #48 forces a stop at the crossing; a plain run has no
    such stop). Bounding the step below the window width is what makes a plain
    run a faithful finite-difference oracle here.
    """
    model, x_idx = _flash(flash_mag=flash_mag, flash_dur=flash_dur)
    r = bngsim.Simulator(model, method="ode").run(
        sample_times=_TS, rtol=1e-12, atol=1e-14, max_step=max_step, max_steps=10**7
    )
    return np.asarray(r.species)[:, x_idx]


class TestShippedZeroAmplitude:
    """flashMag = 0 (the shipped value): the analytic column is a correct zero."""

    def test_flash_duration_column_is_exactly_zero(self):
        # mag = flashMag/flashDur = 0, so testflash is 0 on both sides of the
        # window: the RHS does not depend on flashDur and the true sensitivity is
        # exactly 0. This is #361's "silent zero" — and it is the right answer.
        S = _analytic_flashDur_column(flash_mag=0.0)
        np.testing.assert_allclose(S, np.zeros_like(S), atol=1e-12)

    @pytest.mark.parametrize("rtol", [1e-8, 1e-10, 1e-12])
    def test_zero_holds_across_tolerances(self, rtol):
        # A folded/dropped term would still read 0 here, but a column carried on
        # CVODES' difference quotient (the fallback if the term were refused)
        # would show the moving-root noise that fooled the issue's FD — noise
        # that scales with the tolerance. A tolerance-invariant exact zero is the
        # signature of the analytic path answering correctly.
        S = _analytic_flashDur_column(flash_mag=0.0, rtol=rtol, atol=rtol * 1e-2)
        np.testing.assert_allclose(S, np.zeros_like(S), atol=1e-12)


class TestRealAmplitude:
    """flashMag != 0: the impure switch-time column is right (issue #358)."""

    def test_matches_closed_form(self):
        S = _analytic_flashDur_column(flash_mag=FLASH_MAG)
        expected = np.array([_closed_form_dX_dflashDur(t, FLASH_MAG) for t in _TS])
        # Inside the window the in-branch term −flashMag·(t−del)/flashDur² is
        # non-zero; past the window it returns to 0 as the saltation jump cancels
        # it. Both halves must be present and summed.
        np.testing.assert_allclose(S, expected, atol=1e-6)
        assert np.max(np.abs(S)) > 1.0  # the column is genuinely non-trivial

    @pytest.mark.parametrize("rtol", [1e-8, 1e-10, 1e-12])
    def test_does_not_drift_with_tolerance(self, rtol):
        S = _analytic_flashDur_column(flash_mag=FLASH_MAG, rtol=rtol, atol=rtol * 1e-2)
        expected = np.array([_closed_form_dX_dflashDur(t, FLASH_MAG) for t in _TS])
        np.testing.assert_allclose(S, expected, atol=1e-6)

    def test_matches_central_finite_difference(self):
        # An independent cross-check of the closed form, on plain runs with the
        # step bounded so the pulse is resolved.
        h = 1e-6
        max_step = FLASH_DUR / 8.0
        fd = (
            _plain_X(FLASH_MAG, FLASH_DUR + h, max_step)
            - _plain_X(FLASH_MAG, FLASH_DUR - h, max_step)
        ) / (2 * h)
        S = _analytic_flashDur_column(flash_mag=FLASH_MAG)
        np.testing.assert_allclose(S, fd, atol=1e-4)


class TestAmplitudeContinuity:
    """The shipped zero is the flashMag -> 0 limit, not a dropped term."""

    def test_column_is_linear_in_amplitude(self):
        # Both contributions (in-branch term and crossing jump) are proportional
        # to flashMag, so the whole column scales linearly and passes through the
        # origin. Read at an interior sample where the closed form is non-zero.
        interior = _TS.index(1.2)
        ref = _analytic_flashDur_column(flash_mag=1.0)[interior]
        assert ref == pytest.approx(_closed_form_dX_dflashDur(1.2, 1.0), abs=1e-6)
        for mag in (0.1, 0.01, 0.001):
            val = _analytic_flashDur_column(flash_mag=mag)[interior]
            assert val == pytest.approx(mag * ref, rel=1e-6)
        # ...and the limit is exactly zero, not merely small.
        assert _analytic_flashDur_column(flash_mag=0.0)[interior] == 0.0


def test_flash_duration_term_emitted_symbolically():
    """The ∂/∂flashDur term keeps flashMag symbolic even when flashMag = 0.

    #361 feared the emitter might simplify ``-flashMag/flashDur**2`` to ``0``
    using the current ``flashMag = 0``, which would both drop this column and make
    it wrong the moment ``set_param`` changed ``flashMag``. It does not: the
    emitted term references flashMag's parameter slot, so it is a live expression.
    """
    model, _ = _flash(flash_mag=0.0)  # shipped zero amplitude
    data = model._core.codegen_data()
    idx = {p["name"]: i for i, p in enumerate(data["parameters"])}
    terms, decline = _codegen._functional_dfdp_terms(model._core, data, None)
    assert decline is None

    flashdur_terms = [c for lst in terms.values() for (pi, c) in lst if pi == idx["flashDur"]]
    assert flashdur_terms, "flashDur has no emitted ∂f/∂p term"
    flashmag_ref = f"p[{idx['flashMag']}]"
    assert any(flashmag_ref in c for c in flashdur_terms), (
        f"flashDur term folded flashMag to a constant: {flashdur_terms}"
    )
