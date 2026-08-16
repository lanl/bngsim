"""Issue #368 — when a trajectory finite difference is *not* a reference for a
``piecewise``-in-time sensitivity column.

``MODEL1411210000`` (35 species, 167 parameters) drives its blood-flow forcing
through an SBML ``piecewise`` over ``time``, which the loader renders as
bngsim's own ``if()``::

    Fin_t = if((time() >= t1) && (time() <= tend), (1 + alphaF)*F0, time())

#368 reported ``∂Nag/∂Vg`` as wrong in sign and ~50x in magnitude against a
central difference of two plain ``Simulator.run`` trajectories, and asked which
path the model took and whether the ``piecewise`` conditions were classified
correctly.

Verification says the analytic column is **right** and the reported reference is
not a reference. Two independent traps produced it, and both are pinned below on
a model whose closed form is exact:

* **an unresolvable step.** ``Vg`` moves ``Nag`` by a relative ``1.5e-8``, so at
  the issue's ``h = 1e-6·Vg`` the whole finite-difference signal is ~1e-12 on a
  trajectory of ~35 — a factor of 30 *below* the solver's own ``rtol`` floor.
  The quotient is then noise divided by ``2h``: it diverges as the step shrinks
  and collapses as the tolerance tightens. Refining the step instead (``h`` up to
  ``0.2·Vg``, ``rtol=1e-13``) converges to ``2.188287e-06`` against the analytic
  ``2.188187e-06`` — five digits, in sign and magnitude.

* **the switch node itself.** ``s(t) = ∂x/∂t*`` is ``0`` before the crossing and
  jumps to ``f⁻ − f⁺`` after it, so at exactly ``t = t*`` it is one-sidedly
  differentiable. bngsim reports the right-limit, the same right-continuous
  convention the trajectory uses; a *central* difference there straddles the
  crossing and converges to the average of the two one-sided derivatives, which
  is **exactly half** the analytic value at every step size. A sweep of this
  model's 106 finite-differenceable parameter columns found nothing else: every
  other cell agrees with a step-refined quotient, ``Hb_OP``'s 6% being ordinary
  ``h²`` truncation that refines away.

The classification question has the same answer from the other side. The model
is not declined (``_functional_dfdp_terms`` returns no decline and emits five
live ``∂f/∂Vg`` terms), and it should not be: every ``piecewise`` condition here
compares ``time`` against constants, which is issue #48's crossing-known-a-priori
shape, not #150's moving state threshold. The crossing parameter ``t1`` carries
**no** ``∂f/∂p`` term at all — its whole column is the #48 jump — and it too
matches a step-refined difference to five digits away from the node.

So there is nothing here to warn about, and the tests below are what keeps the
next reader from re-deriving that: a column that regressed to a silent zero, to a
dropped crossing jump, or to the left-limit at the node fails loudly.

The sweep that established this did turn up a real defect, on other models:
crossings that share a threshold *value* merged into one record, so two switch
times that happened to be equal charged each other's jump (issue #375). That one
was real and is fixed; its reproducer and the isolation that repairs it live in
``test_coincident_switch_time_isolation.py``. Worth keeping the pair in view — a
switch-time column can be wrong for a reason a finite difference sees clearly, and
wrong-looking for a reason it cannot see at all.
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest
from bngsim import _codegen
from bngsim._bngsim_core import ModelBuilder

# ─── A minimal model carrying both of #368's shapes ──────────────────────────
# dX/dt = if(time() >= tSw, kPost, kPre) + cSmall*p,   X(0) = 0
#
# The `if()` is what the SBML loader produces for MODEL1411210000's `piecewise`
# over `time` (checked above), so this is the same construct reaching the same
# emitter, not a re-spelling of it.
#
#   X(t)      = (kPre + cSmall*p)*t                                    t <= tSw
#             = (kPre + cSmall*p)*tSw + (kPost + cSmall*p)*(t - tSw)   t >= tSw
#   dX/dtSw   = 0 for t < tSw, kPre - kPost for t > tSw   (a pure crossing jump)
#   dX/dp     = cSmall*t                                  (exact, and linear)
#
# `p` is the stand-in for `Vg`: it is deliberately scaled so its relative effect
# on X is ~1e-8, the regime where a trajectory difference stops being a
# reference. `tSw` is the stand-in for `t1`.
T_SW, K_PRE, K_POST = 1.0, 3.0, 0.5
P0, C_SMALL = 0.25, 5e-8

# t=0 first: `sample_times` overrides `t_span`, so the integration starts at its
# first entry and a grid beginning past 0 would silently drop the pre-history.
_TS = [0.0, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0, 10.0]
_NODE = _TS.index(T_SW)

_RUN = dict(rtol=1e-12, atol=1e-14, max_steps=10**7)


def _switch_model(t_sw=T_SW, p=P0):
    b = ModelBuilder()
    for name, value in (
        ("tSw", t_sw),
        ("kPre", K_PRE),
        ("kPost", K_POST),
        ("p", p),
        ("cSmall", C_SMALL),
    ):
        b.add_parameter(name, value)
    x_idx = b.add_species("X()", 0.0)
    b.add_function("rate", "if(time()>=tSw,kPost,kPre)+cSmall*p")
    b.add_reaction([], [x_idx], "functional", "rate")
    return bngsim.Model(_core=b.build()), x_idx


def _columns(rtol=1e-12, atol=1e-14):
    """``(∂X/∂tSw, ∂X/∂p)`` at ``_TS``, from one analytic sensitivity run."""
    model, x_idx = _switch_model()
    r = bngsim.Simulator(model, method="ode", sensitivity_params=["tSw", "p"]).run(
        sample_times=_TS, rtol=rtol, atol=atol, max_steps=10**7
    )
    S = np.asarray(r.sensitivities)
    return S[:, x_idx, 0], S[:, x_idx, 1]


def _plain_X(t_sw=T_SW, p=P0, max_step=0.05, rtol=_RUN["rtol"], atol=_RUN["atol"]):
    """X at ``_TS`` from a plain run — the finite-difference oracle.

    ``max_step`` bounds the step below the distance between the crossing and its
    neighbouring samples: a plain run has no stop at ``tSw`` (the sensitivity run
    gets one from issue #48), so an unbounded step could integrate across the
    discontinuity and blunt the very edge being differentiated.
    """
    model, x_idx = _switch_model(t_sw=t_sw, p=p)
    r = bngsim.Simulator(model, method="ode").run(
        sample_times=_TS, max_step=max_step, rtol=rtol, atol=atol, max_steps=10**7
    )
    return np.asarray(r.species)[:, x_idx]


def _closed_form_dX_dtSw():
    return np.array([0.0 if t < T_SW else K_PRE - K_POST for t in _TS])


def _closed_form_dX_dp():
    return C_SMALL * np.asarray(_TS)


class TestCrossingColumn:
    """``tSw`` appears only in the condition, so its column *is* the #48 jump."""

    def test_matches_closed_form(self):
        d_tsw, _ = _columns()
        np.testing.assert_allclose(d_tsw, _closed_form_dX_dtSw(), atol=1e-9)
        # Zero before the crossing is the answer, not a dropped term: the jump
        # after it is what a regression to a silent zero would lose.
        assert d_tsw[_NODE - 1] == 0.0
        assert d_tsw[-1] == pytest.approx(K_PRE - K_POST, rel=1e-9)

    @pytest.mark.parametrize("rtol", [1e-8, 1e-10, 1e-12])
    def test_does_not_drift_with_tolerance(self, rtol):
        # A column carried on CVODES' difference quotient instead would integrate
        # the variational equation straight through the crossing and show the
        # tolerance-dependent drift that fooled #368's reference.
        d_tsw, _ = _columns(rtol=rtol, atol=rtol * 1e-2)
        np.testing.assert_allclose(d_tsw, _closed_form_dX_dtSw(), atol=1e-6)

    @pytest.mark.parametrize("h", [1e-1, 1e-2, 1e-3])
    def test_matches_central_difference_away_from_the_node(self, h):
        fd = (_plain_X(t_sw=T_SW + h) - _plain_X(t_sw=T_SW - h)) / (2 * h)
        d_tsw, _ = _columns()
        off_node = [i for i in range(len(_TS)) if i != _NODE]
        np.testing.assert_allclose(d_tsw[off_node], fd[off_node], atol=1e-6)


class TestAtTheSwitchNode:
    """The one cell where a central difference is *expected* to disagree."""

    @pytest.mark.parametrize("h", [1e-1, 1e-2, 1e-3])
    def test_central_difference_is_exactly_half(self, h):
        # X(t*; tSw+h) is unchanged (the crossing has not happened yet at t*),
        # while X(t*; tSw-h) has already accumulated h*(kPre-kPost). The central
        # quotient is therefore (kPre-kPost)/2 — the average of a right-hand
        # derivative of 0 and a left-hand one of kPre-kPost — for *every* h, so
        # this is a property of the quantity, not a step artefact that refines
        # away. Reading it as "the analytic value is 2x too big" is #368's trap.
        fd = (_plain_X(t_sw=T_SW + h) - _plain_X(t_sw=T_SW - h)) / (2 * h)
        assert fd[_NODE] == pytest.approx((K_PRE - K_POST) / 2.0, rel=1e-6)

    def test_the_column_reports_the_right_limit(self):
        # bngsim reports the post-jump value at t = t*, the same right-continuous
        # convention the trajectory itself uses. Pinning it keeps a future change
        # from silently switching to the left-limit, which would read as a
        # one-sample-late jump.
        d_tsw, _ = _columns()
        assert d_tsw[_NODE] == pytest.approx(d_tsw[_NODE + 1], rel=1e-9)
        assert d_tsw[_NODE] == pytest.approx(K_PRE - K_POST, rel=1e-9)
        assert d_tsw[_NODE] != d_tsw[_NODE - 1]


class TestUnresolvableStep:
    """#368's actual cell: a real column under an unusable reference."""

    def test_column_matches_closed_form(self):
        # `p` enters both branches, so this column has no crossing content at
        # all — it is cSmall*t exactly, at every sample.
        _, d_p = _columns()
        np.testing.assert_allclose(d_p, _closed_form_dX_dp(), rtol=1e-6, atol=1e-14)

    # The admissible bound scales as 1/h, and that is the whole lesson rather
    # than a fudge: X is exactly linear in p, so neither step carries truncation
    # error, and what separates the quotient from the column is only the two
    # runs' own error divided by 2h. A decade smaller step buys a decade looser
    # bound — the same arithmetic that leaves #368's h = 1e-6·p with no bound at
    # all. Both are still three to four orders better than that step's 25%.
    @pytest.mark.parametrize("rel_h, bound", [(1e-1, 1e-3), (1e-2, 1e-2)])
    def test_a_resolvable_step_confirms_it(self, rel_h, bound):
        # The oracle runs a decade tighter than the column it checks, so the
        # floor being divided is the oracle's and not the column's.
        h = rel_h * P0
        tol = dict(rtol=1e-13, atol=1e-16)
        fd = (_plain_X(p=P0 + h, **tol) - _plain_X(p=P0 - h, **tol)) / (2 * h)
        _, d_p = _columns()
        np.testing.assert_allclose(fd, d_p, rtol=bound, atol=1e-14)

    def test_the_reported_step_is_below_the_solver_floor(self):
        # The issue's h = 1e-6*p. The difference it takes is smaller than the
        # tolerance the two trajectories were each computed to, so the quotient
        # is the solver's own error divided by 2h and carries no information
        # about the derivative — which is why it came back with the wrong sign.
        h = 1e-6 * P0
        signal = np.abs(_plain_X(p=P0 + h) - _plain_X(p=P0 - h))
        floor = _RUN["rtol"] * np.abs(_plain_X())
        assert np.all(signal[1:] < floor[1:]), (
            f"signal {signal} is not below the tolerance floor {floor}; "
            "the model no longer reproduces #368's unresolvable-step regime"
        )


class TestSampleTimesCoverTheInterval:
    """The grid a finite-difference reference is read on is also the run.

    ``sample_times`` overrides ``t_span`` entirely, so its first entry is the
    integration *start*, not merely the first thing reported. A grid that opens
    after ``t = 0`` therefore drops the pre-history and answers a different
    question — an easy way to build two trajectories that disagree for reasons
    that have nothing to do with the parameter being perturbed. The docstring
    said "at least 3 values" where the code takes 2; both halves are pinned here
    because this is the sharp edge that a reference-building script meets first.
    """

    def _ramp(self):
        b = ModelBuilder()
        b.add_parameter("k", 1.0)
        x_idx = b.add_species("X()", 0.0)
        b.add_reaction([], [x_idx], "elementary", "k")  # dX/dt = 1, X(t) = t
        return bngsim.Model(_core=b.build()), x_idx

    @pytest.mark.parametrize(
        "sample_times, expected",
        [([0.0, 10.0], 10.0), ([5.0, 10.0], 5.0), ([9.0, 10.0], 1.0)],
    )
    def test_the_first_sample_time_starts_the_run(self, sample_times, expected):
        model, x_idx = self._ramp()
        r = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, 10.0), sample_times=sample_times, **_RUN
        )
        assert r.time[0] == pytest.approx(sample_times[0])
        assert np.asarray(r.species)[-1, x_idx] == pytest.approx(expected, rel=1e-9)

    def test_two_points_are_accepted(self):
        model, x_idx = self._ramp()
        r = bngsim.Simulator(model, method="ode").run(sample_times=[0.0, 10.0], **_RUN)
        assert len(r.time) == 2

    def test_one_point_is_refused(self):
        model, _ = self._ramp()
        with pytest.raises(ValueError, match="at least 2 points"):
            bngsim.Simulator(model, method="ode").run(sample_times=[10.0], **_RUN)


class TestTheModelIsNotDeclined:
    """A ``time`` crossing against constants is #48's shape — nothing to warn."""

    def test_no_decline_and_the_in_branch_term_is_live(self):
        model, _ = _switch_model()
        data = model._core.codegen_data()
        idx = {p["name"]: i for i, p in enumerate(data["parameters"])}
        terms, decline = _codegen._functional_dfdp_terms(model._core, data, None)
        assert decline is None

        p_terms = [c for lst in terms.values() for (pi, c) in lst if pi == idx["p"]]
        assert p_terms, "the in-branch parameter has no emitted ∂f/∂p term"
        assert any(f"p[{idx['cSmall']}]" in c for c in p_terms), (
            f"∂f/∂p folded cSmall to a constant: {p_terms}"
        )

    def test_the_crossing_parameter_has_no_in_branch_term(self):
        # `tSw` occurs only in the condition, so ∂f/∂tSw is a clean 0 in both
        # branch interiors and that *is* the answer there. Its column comes
        # entirely from the crossing jump — which is why an emitted term here
        # would mean the jump is being counted twice.
        model, _ = _switch_model()
        data = model._core.codegen_data()
        idx = {p["name"]: i for i, p in enumerate(data["parameters"])}
        terms, _decline = _codegen._functional_dfdp_terms(model._core, data, None)
        tsw_terms = [c for lst in terms.values() for (pi, c) in lst if pi == idx["tSw"]]
        assert tsw_terms == []

    def test_nothing_is_logged_as_refused(self, caplog):
        model, _ = _switch_model()
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            bngsim.Simulator(model, method="ode", sensitivity_params=["tSw", "p"]).run(
                sample_times=_TS, **_RUN
            )
        refusals = [
            r.getMessage()
            for r in caplog.records
            if "sensitivity RHS is declined" in r.getMessage()
        ]
        assert refusals == []
