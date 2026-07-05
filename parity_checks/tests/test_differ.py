"""Lock the shared cross-engine comparison protocol (`_core/differ.py`).

`differ.deterministic_verdict` / `ensemble_verdict` are the verdict both parity
suites gate on (issue #69 worries about this protocol drifting), so each
contract below is anchored to an oracle, not to a frozen number:

deterministic_verdict
  * identical arrays pass with zero residual (the trivial floor)
  * a concentrated relative blow-up past the hard ceiling is a DIFF even when
    it is a vanishing fraction of cells (ceilings beat the budget)
  * a few soft cells within the fail-fraction budget are forgiven; one more
    than the budget allows is a DIFF (the budget boundary, parametrized)
  * an underflow-scale disagreement (both sides below the file-scale floor) is
    forgiven by the near-zero backstop
  * a cell where exactly one side is non-finite (NaN/inf vs a finite value) is
    an unconditional hard fail — the regression guard for the bug where the
    near-zero backstop silently forgave it
  * both-NaN is a zero-diff pass
  * the verdict is invariant under a positive global rescaling of both inputs
    (metamorphic: the protocol is scale-aware), and `forgive_mask` only ever
    loosens (superset property)
  * a shape mismatch raises

ensemble_verdict
  * equal means pass; a sub-particle cell (one side floored to 0, the other a
    small nonzero with tiny SEM — the BIOMD2 class in miniature) is a DIFF
  * a difference within K SEM passes; the relative escape hatch passes a
    deterministic cell whose SEM has collapsed
  * a near-zero cell is skipped (counts as a pass)
  * one-side-NaN mean fails, both-NaN passes
  * the frac_pass>=0.99 gate is exact at the boundary (parametrized)
  * a shape mismatch raises
"""

from __future__ import annotations

import numpy as np
import pytest
from _core import differ


def _passed(a, b, **kw):
    v = differ.deterministic_verdict(np.asarray(a, float), np.asarray(b, float), **kw)
    return v["passed"], v


# --------------------------------------------------------------------------- #
# deterministic_verdict
# --------------------------------------------------------------------------- #
def test_identical_arrays_pass_with_zero_residual():
    """a == b ⇒ PASS, no failing cells, zero reported residual."""
    rng = np.random.default_rng(0)
    a = rng.uniform(1e-3, 10.0, size=(50, 6))
    ok, v = _passed(a, a.copy())
    assert ok
    assert v["n_fail"] == 0
    assert v["max_rel"] == 0.0 and v["max_abs"] == 0.0


def test_concentrated_relative_blowup_is_diff_despite_tiny_fraction():
    """One 20% divergence on a unit-magnitude cell is a DIFF.

    It is 1/500 of cells — far inside the fail-fraction budget — but the hard
    relative ceiling (0.05) is unconditional, so the budget cannot forgive it.
    Oracle: a concentrated real divergence must never be averaged away.
    """
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    b[0, 0] = 1.2  # 16.7% relative on a magnitude-carrying cell
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] == 1
    assert v["max_rel"] == pytest.approx(0.2 / 1.2, rel=1e-6)


@pytest.mark.parametrize("n_soft", [3, 50, 300])
def test_subceiling_column_forgiven_regardless_of_count(n_soft):
    """A column whose worst divergence is a 1% relative shift (above REL_TOL=1e-4,
    below HARD_REL_CEILING=0.05) at its own peak scale never diverges past the
    ceiling, so the peak-relative significance gate forgives it OUTRIGHT — no
    matter how many of its cells trip the tight per-cell soft bar. This is the
    decayed-tail / integrator-noise class the gate was added for (rr_check forgives
    it on reldiv<0.05): an arbitrarily large fraction of sub-ceiling soft cells is
    not a divergence. (Before the gate this DIFFed once the soft fraction passed
    FAIL_FRAC_BUDGET — the behavior change this test pins.)"""
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    flat = b.reshape(-1)
    flat[:n_soft] = 1.01  # 1% relative on a unit-magnitude column: < HARD_REL_CEILING
    ok, v = _passed(a, b)
    assert ok
    assert v["n_hard_fail"] == 0
    assert v["max_rel"] == 0.0  # nothing survives to the verdict


def test_budget_forgives_soft_cells_within_a_real_column():
    """The fail-fraction budget still forgives a few soft cells that coexist with
    a genuine (real) divergence in the same column. The column has one 10% hard
    cell (so it is a real divergence the gate does NOT forgive) plus n soft 1%
    cells. The model DIFFs either way (the hard cell), but the budget controls
    whether the soft cells are reported as forgiven vs counted — the budget
    bookkeeping the gate keeps."""
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    b[0, 0] = 1.10  # hard: 9.1% of col peak > HARD_REL_CEILING → column is real
    b[1, 0] = b[2, 0] = 1.01  # 2 soft cells (2/500 = 0.004 <= FAIL_FRAC_BUDGET)
    ok, v = _passed(a, b)
    assert not ok  # the hard cell DIFFs regardless
    assert v["n_hard_fail"] == 1
    assert v["budget_forgiven"] == 2  # the 2 soft cells stay within budget


def test_near_zero_backstop_forgives_underflow_disagreement():
    """Two columns: a unit-magnitude one (sets the file scale ~1) and a column
    where both sides are ~1e-15 but differ by 2x. The tiny column is below the
    file-scale near-zero floor, so its (relatively huge) disagreement is
    underflow, not signal — forgiven. Oracle: sub-1e-12-of-scale noise is not a
    divergence."""
    a = np.array([[1.0, 1e-15], [1.0, 2e-15], [1.0, 3e-15]])
    b = np.array([[1.0, 2e-15], [1.0, 4e-15], [1.0, 6e-15]])
    ok, _ = _passed(a, b)
    assert ok


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_one_side_nonfinite_is_unconditional_hard_fail(bad):
    """A cell where exactly one engine produced NaN/inf while the other gave a
    finite value is an unambiguous divergence and must DIFF — regardless of
    magnitude. Regression guard: np.maximum collapses the cell's colmag to 0,
    so without an explicit guard the near-zero backstop forgives it and a real
    blow-up passes silently (the bug this test was written to catch)."""
    a = np.ones((5, 2))
    b = np.ones((5, 2))
    a[0, 0] = bad
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] >= 1


def test_both_nan_cell_is_a_pass():
    """Both engines NaN at the same cell ⇒ zero diff there, no fail."""
    a = np.ones((4, 2))
    b = np.ones((4, 2))
    a[1, 1] = b[1, 1] = np.nan
    ok, _ = _passed(a, b)
    assert ok


@pytest.mark.parametrize("c", [1.0, 1e3, 1e6])
def test_verdict_is_invariant_under_positive_rescaling(c):
    """Scaling both inputs by c>0 scales absd, the file scale, col_peak and
    colmag identically, so every per-cell tolerance term scales with the data:
    the pass/fail verdict is invariant. Metamorphic oracle for scale-awareness.
    Mixed case: mostly-agreeing with two 1%-soft cells and one 10%-hard cell."""
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    b[0, 0] = 1.01
    b[1, 0] = 1.01
    b[2, 1] = 1.10  # hard
    base_ok, base = _passed(a, b)
    scaled_ok, scaled = _passed(c * a, c * b)
    assert scaled_ok is base_ok
    assert scaled["n_hard_fail"] == base["n_hard_fail"]


def test_decayed_tail_relative_divergence_is_forgiven():
    """A species that peaks at the file scale then decays to a tiny tail, where
    the two engines disagree RELATIVELY on that tail (per-cell reld ~0.9, hard
    under the old per-cell rule), is forgiven by the peak-relative gate:
    |a-b|/col_peak ≈ 1e-3 ≪ HARD_REL_CEILING — the tail is measured against the
    species' OWN peak, not its vanishing value. Oracle: a divergence below the
    species' dynamic range is integrator underflow, the artifact class the gate
    was added for (rr_check forgives it on reldiv<0.05)."""
    n = 50
    a = np.ones((n, 2))  # col1 = 1.0 sets the file scale; col0 peaks at 1.0
    a[10:, 0] = 1e-4  # col0 decays to a 1e-4 tail (peak stays 1.0)
    b = a.copy()
    b[10:, 0] = 1.1e-3  # tail disagrees: per-cell reld ≈ 0.91, |a-b|=1e-3=1e-3*peak
    ok, v = _passed(a, b)
    assert ok
    assert v["n_hard_fail"] == 0
    assert v["max_rel"] == 0.0


def test_significant_species_past_ceiling_still_fails():
    """A meaningful-magnitude species (signif > SIGNIF_FLOOR) that diverges past
    HARD_REL_CEILING at its own peak scale is a real divergence the gate must NOT
    forgive. col0 carries ~12% of the file scale (well above the 1e-3 floor) and
    disagrees ~17% at its peak — a hard fail, DIFF."""
    n = 50
    a = np.ones((n, 2))
    a[:, 1] = 10.0  # scale-setter: file peak = 10
    a[:, 0] = 1.0  # col0 peak ≈ 0.1*scale → signif ≈ 0.1 ≫ SIGNIF_FLOOR
    b = a.copy()
    b[5, 0] = 1.2  # 0.2/1.2 = 16.7% of col0 peak > HARD_REL_CEILING
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] == 1
    assert v["max_rel"] == pytest.approx(0.2 / 1.2, rel=1e-6)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_one_side_nonfinite_on_subsignificance_column_is_still_hard(bad):
    """A one-side-non-finite cell is an unconditional hard fail even when its
    column is far below the dynamic range (signif ≪ SIGNIF_FLOOR): a blow-up on
    one engine is never forgiven by the significance gate. Regression guard that
    the dynamic-range forgiveness explicitly excludes one_side_nonfinite."""
    n = 20
    a = np.ones((n, 2))
    a[:, 1] = 1e6  # scale-setter: file peak = 1e6
    a[:, 0] = 1e-3  # col0 ≈ 1e-9 of scale → sub-significant
    b = a.copy()
    b[0, 0] = bad  # one engine blows up on the sub-significance column
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] >= 1


def test_forgive_mask_only_loosens():
    """forgive_mask forgives cells *before* the gate/budget, so masking a genuine
    divergence can only flip DIFF→PASS, never the reverse. Here one 10% cell on a
    significant column is a real divergence (DIFF); masking it → PASS. Confirms
    omitting the mask (rr_parity passes None) is strictly tighter."""
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    b[0, 0] = 1.10  # real: column diverges past HARD_REL_CEILING at its own peak
    assert not _passed(a, b)[0]  # DIFF
    mask = np.zeros((100, 5), dtype=bool)
    mask[0, 0] = True
    assert _passed(a, b, forgive_mask=mask)[0]  # forgiven → PASS


def test_deterministic_shape_mismatch_raises():
    with pytest.raises(ValueError):
        differ.deterministic_verdict(np.ones((3, 2)), np.ones((3, 3)))


# --------------------------------------------------------------------------- #
# ensemble_verdict
# --------------------------------------------------------------------------- #
def _ens(ma, sa, mb, sb):
    return differ.ensemble_verdict(
        np.asarray(ma, float),
        np.asarray(sa, float),
        np.asarray(mb, float),
        np.asarray(sb, float),
    )


def test_ensemble_equal_means_pass():
    """Identical ensemble means ⇒ every cell passes the K-sigma test."""
    rng = np.random.default_rng(1)
    m = rng.uniform(1.0, 100.0, size=(20, 4))
    s = np.full_like(m, 0.5)
    v = _ens(m, s, m.copy(), s.copy())
    assert v["passed"] and v["frac_pass"] == 1.0


def test_ensemble_sub_particle_cell_is_diff():
    """The BIOMD2 sub-particle class in miniature: bngsim floors to 0, RR keeps
    a small nonzero value with a tiny SEM. The mean difference is many SEM and
    far above the relative escape hatch, so the cell fails and frac_pass < 0.99.
    Oracle: a 0-vs-nonzero ensemble split is a genuine ensemble divergence."""
    mean_a = np.zeros((10, 3))
    mean_b = np.full((10, 3), 1e-5)
    sem = np.full((10, 3), 1e-9)
    v = _ens(mean_a, sem, mean_b, sem)
    assert not v["passed"]
    assert v["frac_pass"] == 0.0
    assert np.isinf(v["max_z"]) or v["max_z"] > 100


def test_ensemble_within_k_sigma_passes():
    """Means differing by 2 combined SEM pass at K=3. se = sqrt(sa²+sb²)."""
    mean_a = np.full((5, 2), 10.0)
    sa = sb = np.full((5, 2), 1.0)
    se = np.sqrt(sa**2 + sb**2)
    mean_b = mean_a + 2.0 * se  # 2 sigma < K=3
    v = _ens(mean_a, sa, mean_b, sb)
    assert v["passed"]


def test_ensemble_k_for_scales_with_sqrt_n():
    """Effect-size-preserving gate (GH #190): K = ENSEMBLE_K·sqrt(N/base), unchanged
    at the calibration N, 2× at 4× the replicates, sqrt(10)× at 10×."""
    assert differ.ensemble_k_for(differ.ENSEMBLE_K_BASE_N) == pytest.approx(differ.ENSEMBLE_K)
    assert differ.ensemble_k_for(4 * differ.ENSEMBLE_K_BASE_N) == pytest.approx(2.0 * differ.ENSEMBLE_K)
    assert differ.ensemble_k_for(100) == pytest.approx(
        differ.ENSEMBLE_K * (100 / differ.ENSEMBLE_K_BASE_N) ** 0.5
    )


def test_ensemble_verdict_honors_scaled_k():
    """A 5σ difference FAILS at the base K=3 but PASSES at the N=100 gate (K≈9.49):
    more replicates sharpen precision without flagging the same meaningful diff."""
    mean_a = np.full((5, 2), 10.0)
    sa = sb = np.full((5, 2), 1.0)
    se = np.sqrt(sa**2 + sb**2)
    mean_b = mean_a + 5.0 * se
    assert not differ.ensemble_verdict(mean_a, sa, mean_b, sb)["passed"]
    scaled = differ.ensemble_verdict(mean_a, sa, mean_b, sb, k=differ.ensemble_k_for(100))
    assert scaled["passed"]
    assert scaled["k"] == pytest.approx(differ.ensemble_k_for(100))


def test_ensemble_relative_escape_hatch_passes_collapsed_sem():
    """A near-deterministic cell (SEM → 0) whose means differ by < REL_TOL is
    passed by the relative escape hatch, even though the sigma test alone would
    reject any nonzero difference at zero SEM. Oracle: effectively-deterministic
    agreement should not be punished for having no sampling spread."""
    mean_a = np.full((5, 2), 1.0)
    mean_b = np.full((5, 2), 1.0 + 5e-5)  # 5e-5 < REL_TOL (1e-4)
    zero = np.zeros((5, 2))
    v = _ens(mean_a, zero, mean_b, zero)
    assert v["passed"]


def test_ensemble_near_zero_cell_is_skipped():
    """A cell where both means are below NEAR_ZERO_REL*scale is skipped (treated
    as a pass): with one O(100) cell setting the scale, an O(1e-9*scale) cell
    that disagrees does not drag frac_pass down."""
    mean_a = np.array([[100.0, 1e-8]])
    mean_b = np.array([[100.0, 5e-8]])  # tiny cell disagrees 5x, but near-zero
    sem = np.array([[1.0, 1e-12]])
    v = _ens(mean_a, sem, mean_b, sem)
    assert v["passed"] and v["frac_pass"] == 1.0


def test_ensemble_one_side_nan_fails_both_nan_passes():
    """One-side-NaN mean is a fail; both-NaN is a pass (the cells are aligned
    missing values, not a divergence)."""
    mean_a = np.array([[1.0, np.nan], [1.0, 1.0]])
    mean_b = np.array([[1.0, 2.0], [1.0, 1.0]])
    sem = np.full((2, 2), 0.1)
    assert not _ens(mean_a, sem, mean_b, sem)["passed"]
    mean_a2 = np.array([[1.0, np.nan]])
    mean_b2 = np.array([[1.0, np.nan]])
    assert _ens(mean_a2, np.full((1, 2), 0.1), mean_b2, np.full((1, 2), 0.1))["passed"]


@pytest.mark.parametrize(
    "n_fail, expect_pass",
    [
        (1, True),  # 99/100 pass = 0.99 == ENSEMBLE_PASS_FRAC → PASS
        (2, False),  # 98/100 = 0.98 < 0.99 → DIFF
    ],
)
def test_ensemble_frac_pass_gate_boundary(n_fail, expect_pass):
    """The model passes iff >= 99% of cells pass. 100 cells ⇒ the gate sits
    between 1 and 2 failing cells. Failing cells are a 0-vs-nonzero split."""
    n = 100
    mean_a = np.full((n, 1), 5.0)
    mean_b = np.full((n, 1), 5.0)
    sem = np.full((n, 1), 0.01)
    mean_b[:n_fail, 0] = 50.0  # gross, many-sigma, not near-zero, not rel-ok
    v = _ens(mean_a, sem, mean_b, sem)
    assert v["passed"] is expect_pass
    assert v["frac_pass"] == pytest.approx((n - n_fail) / n)


def test_ensemble_shape_mismatch_raises():
    with pytest.raises(ValueError):
        differ.ensemble_verdict(np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 3)))
