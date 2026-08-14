"""Issue #326 — a parameter that carries no scale must not reject the setup.

``setup_forward_sensitivities`` non-dimensionalizes each sensitivity column by
``pbar[iS]``, the parameter's own magnitude, and states that column's absolute
tolerance as ``atolS[iS][i] = atol·scale[i]/pbar[iS]`` (GH #214). Both are
statements about the *size* of a parameter, so both need a size to be there.

``pbar[i] = |p| unless p == 0`` was the whole guard, and ``±inf`` walks straight
through it. Genome-scale FBC models spell a missing flux bound exactly that way:
``MODEL1703150000`` carries ``_lp_r_0553_UPPER_BOUND = inf`` and
``_lp_r_0185_LOWER_BOUND = -inf`` among its 8566 parameters, and the corpus sweep
selects them as sensitivity targets like any other. The consequence is not a
wrong derivative but a refusal to start:

    pbar   = inf
    atolS  = atol·scale/inf = 0            for every row of that column
    yS(0)  = 0                             a fresh parameter column seeds to zero
    ewtS   = 1/(rtol·0 + 0)                → CVODES: "Initial ewtS has
                                             component(s) equal to zero (illegal)"

which surfaces as a bare ``CV_ILL_INPUT`` (flag −22) naming neither the parameter
nor the tolerance — the whole model, 2129 species of it, refused over four
columns whose true derivative is identically zero. AMICI integrates the same
model because it passes ``pbar = NULL`` and lets CVODES use 1.0.

So a value that is not a scale falls back to 1.0 — CVODES' own default — and the
tolerance itself is floored strictly positive, because guarding ``pbar`` alone
leaves the same zero reachable from two perfectly finite numbers: an enormous
parameter puts ``atol/pbar`` below the smallest subnormal and it underflows.

The model here is ``S' = -k·S``, ``S(0) = 100``, so every column has a closed
form and nothing in this module is a finite difference:

    S(t)     = 100·e^{−k·t}
    dS/dk(t) = −100·t·e^{−k·t}

with an extra parameter that no rate law mentions — the shape of an FBC bound —
whose column is therefore exactly zero.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

K = 0.5
S0 = 100.0
T_SPAN = (0.0, 4.0)
N_POINTS = 9


def _model(bound_value: float) -> bngsim.Model:
    """``S' = -k·S`` plus ``bound``, a parameter no rate law reads."""
    b = ModelBuilder()
    b.add_parameter("k", K)
    b.add_parameter("bound", bound_value)
    s_idx = b.add_species("S", S0)
    b.add_reaction([s_idx], [], "elementary", "k")
    return bngsim.Model(b.build())


def _run(bound_value: float, atol: float = 1e-12, rtol: float = 1e-9):
    sim = bngsim.Simulator(_model(bound_value), method="ode", sensitivity_params=["k", "bound"])
    return sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=rtol, atol=atol)


def _columns(res):
    sx = np.asarray(res.sensitivities)
    return np.asarray(res.time), sx[:, 0, 0], sx[:, 0, 1]


# ── The refusal that was ────────────────────────────────────────────────────


@pytest.mark.parametrize("bound_value", [math.inf, -math.inf], ids=["plus_inf", "minus_inf"])
def test_an_infinite_parameter_still_integrates(bound_value):
    """The CV_ILL_INPUT itself: before the guard, neither sign started at all."""
    _t, dk, dbound = _columns(_run(bound_value))
    assert np.isfinite(dk).all()
    assert np.isfinite(dbound).all()


def test_a_nan_parameter_gets_a_tolerance_and_not_a_nan():
    """The quieter half, which is why ``nan`` takes the same fallback.

    ``nan`` does not trip the CV_ILL_INPUT — CVODES' "is this weight zero" test
    is a comparison, and every comparison against ``nan`` is false, so a ``nan``
    ``pbar`` propagates a ``nan`` tolerance into the error test instead of being
    rejected at setup. On this model that is survivable (the column is
    identically zero, so nothing weights it), which is exactly the problem: the
    same tolerance on a column that is *not* zero removes that column from error
    control with no flag and no message. Pinned here so the fallback keeps
    covering the failure mode that does not announce itself.
    """
    _t, dk, dbound = _columns(_run(math.nan))
    assert np.isfinite(dk).all()
    assert np.isfinite(dbound).all()


@pytest.mark.parametrize("bound_value", [math.inf, -math.inf], ids=["plus_inf", "minus_inf"])
def test_an_unreferenced_bound_has_an_identically_zero_column(bound_value):
    """What the answer has to be, not merely that one came back.

    ``bound`` appears in no rate law, so ``∂f/∂bound = 0``, the variational
    equation collapses to ``ṡ = J·s`` on a column seeded at zero, and the column
    stays zero for the whole horizon. This is the answer AMICI returns for
    ``MODEL1703150000``'s four ``±inf`` FBC bounds, and matching it is the point
    of the fallback — a run that merely *completes* while reporting noise there
    would be a different bug wearing this one's clothes.
    """
    _t, _dk, dbound = _columns(_run(bound_value))
    assert np.array_equal(dbound, np.zeros_like(dbound))


def test_the_live_column_is_unharmed_by_the_neighbour():
    """``dS/dk`` is still the closed form with ``bound = inf`` beside it.

    The fallback changes one column's tolerance. It must not change the
    derivative of the column next to it, which is checked against the analytic
    solution rather than against a recorded tensor so this stays an oracle.
    """
    t, dk, _dbound = _columns(_run(math.inf))
    np.testing.assert_allclose(dk, -S0 * t * np.exp(-K * t), rtol=1e-6, atol=1e-9)


def test_a_finite_parameter_is_untouched():
    """The control. A parameter with a magnitude keeps ``pbar = |p|``.

    Without this, "fall back to 1.0" could quietly become "always 1.0" and every
    test above would still pass while GH #214's scaling was gone.
    """
    t, dk, dbound = _columns(_run(2.5))
    np.testing.assert_allclose(dk, -S0 * t * np.exp(-K * t), rtol=1e-6, atol=1e-9)
    assert np.array_equal(dbound, np.zeros_like(dbound))


# ── The floor under the quotient ────────────────────────────────────────────


def test_a_tolerance_that_underflows_to_zero_is_floored():
    """``atol/pbar`` reaching zero from two finite numbers is the same refusal.

    ``pbar = 1e308`` with a tight ``atol`` puts the quotient below the smallest
    subnormal, so the column's tolerance is exactly 0 and CVODES rejects the
    setup for the same reason the ``inf`` did — with nothing non-finite anywhere
    in the model. Guarding ``pbar`` alone would leave this reachable.
    """
    _t, dk, dbound = _columns(_run(1e308, atol=1e-20))
    assert np.isfinite(dk).all()
    assert np.isfinite(dbound).all()


def test_rtol_only_error_control_still_works():
    """The clamp's control, on a path that was already working.

    ``atol = 0`` — a caller controlling the state on ``rtol`` alone — reaches the
    same all-zero ``atolS`` the ``1e308`` case reaches by underflow, and yet it
    integrated before this change as well. So it is not a second instance of the
    bug, and it is here for the opposite reason: the clamp must leave a working
    path exactly where it found it.
    """
    t, dk, dbound = _columns(_run(2.5, atol=0.0))
    np.testing.assert_allclose(dk, -S0 * t * np.exp(-K * t), rtol=1e-6, atol=1e-9)
    assert np.isfinite(dbound).all()
