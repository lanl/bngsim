"""Issue #316 — the noise floor's audit numbers must answer the audit question.

#312 added a solver-resolution noise floor to the forward-sensitivity oracle and,
with it, a per-row ``n_noise_forgiven``. The floor is fine. The count was not the
quantity either the PR or the docstring described:

    the count of cells the floor silenced, recorded so a run can never quietly
    forgive its way to a PASS without that being visible in the report.

What it actually counted was ``mask.sum()`` — every cell *inside* the floor. That
is a strict superset of the cells the floor rescued: it includes every cell the
two engines already agreed on, and in particular every cell where both return
exactly ``0.0``, which is the common case the floor exists for. So a run with
**zero disagreement anywhere** reported every one of its cells as forgiven, and
``noise 4820`` in a matrix comment said nothing about whether that row would have
passed without the floor.

The count could not be fixed where it lived. ``sens_verdict`` computed the mask
*before* calling ``differ`` and never saw ``fail_mask``, so it had nothing to
intersect against. The forgiving happens inside ``differ``:

    effective_fail = (fail_mask & ~(forgive | near_zero_mask | below_dyn_range))
                     | one_side_nonfinite

so "rescued by this mask" means: failing, cleared by the mask, and cleared by
nothing else — a one-side-non-finite cell is re-added unconditionally, and a cell
the near-zero backstop or the dynamic-range gate already forgave owes the mask
nothing.

Three numbers now, because one cannot do three jobs:

* ``n_below_noise_floor`` — how much of the tensor is under solver resolution.
  The old number, under a name that says what it is.
* ``n_noise_rescued`` — how much of the verdict rested on the floor.
* ``noise_decisive`` — whether the floor is what made this row a PASS. Today this
  coincides with ``passed and n_noise_rescued > 0`` (see
  ``test_decisiveness_currently_coincides_with_a_nonzero_rescue`` for why the two
  cannot come apart under the current constants), but it is read off the
  **recomputed** verdict rather than argued from a count, so it stays right if
  those constants or the gate ordering move.

``n_noise_forgiven`` is deliberately not kept as an alias — its meaning would
change under a name already written into every row of the shipped reports.
"""

from __future__ import annotations

import numpy as np
import pytest
from _core import differ

import _amici_sens as asens  # isort: skip  (suite dir is on sys.path via conftest)

_ATOL = 1e-12
_P = [1.0, 1.0]


# ── The issue's own two demonstrations ──────────────────────────────────────


def test_exact_agreement_forgives_nothing_and_says_so():
    """Issue #316, demonstration 1. Nothing was failing, so nothing was forgiven.

    This reported ``n_noise_forgiven == n_cells == 12`` — a run in *perfect*
    agreement claiming every cell was rescued by the floor.
    """
    bn = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2) * 1e3
    v = asens.sens_verdict(bn, bn.copy(), param_values=_P, atol=_ATOL)

    assert v["passed"] and v["n_fail"] == 0
    assert v["n_noise_rescued"] == 0
    assert v["noise_decisive"] is False
    # The old quantity is still available, under a name that describes it.
    assert v["n_below_noise_floor"] == v["n_cells"] == 12


def test_one_rescued_cell_among_trivially_agreeing_ones_counts_as_one():
    """Issue #316, demonstration 2. This reported 100.

    The tensor is 100 cells, 99 of them exact-zero-vs-exact-zero. Exactly one
    cell was failing and exactly one was rescued.
    """
    bn = np.zeros((10, 5, 2))
    am = np.zeros((10, 5, 2))
    am[0, 0, 1] = 1e-11
    v = asens.sens_verdict(bn, am, param_values=_P, atol=_ATOL)

    assert v["passed"] and v["n_fail"] == 1
    assert v["n_noise_rescued"] == 1
    assert v["n_below_noise_floor"] == 100
    # And this row really would have been a DIFF without the floor.
    assert v["noise_decisive"] is True


def test_the_two_quantities_are_not_the_same_number():
    """The distinction, stated as one assertion rather than inferred from two."""
    bn = np.zeros((10, 5, 2))
    am = np.zeros((10, 5, 2))
    am[0, 0, 1] = 1e-11
    v = asens.sens_verdict(bn, am, param_values=_P, atol=_ATOL)
    assert v["n_below_noise_floor"] > v["n_noise_rescued"]


# ── `noise_decisive` is a separate question from the count ──────────────────


def test_a_diff_is_never_decisive_however_much_was_rescued():
    """The floor cannot have "forgiven its way to a PASS" on a row that DIFFed."""
    rng = np.random.default_rng(3)
    bn = rng.normal(size=(12, 4, 2)) * 1e6
    am = bn.copy()
    am[:, :, 1] *= 3.0  # a real, large-magnitude divergence the floor cannot touch
    v = asens.sens_verdict(bn, am, param_values=_P, atol=_ATOL)

    assert not v["passed"]
    assert v["noise_decisive"] is False


def test_decisiveness_currently_coincides_with_a_nonzero_rescue():
    """The two agree today, and a failure here means they have decoupled.

    ``noise_decisive`` is read off the recomputed verdict, not inferred from
    ``n_noise_rescued > 0``. Under the current constants the two cannot differ: a
    rescued cell that is only *soft* can exist solely in a column the
    significance gate calls real, and a column is real only when some cell
    exceeds ``HARD_REL_CEILING`` (0.05) — 500x ``REL_TOL`` — so that cell is
    itself a hard fail that no fail-fraction budget absorbs.

    That is a property of two constants and a gate ordering, not of the field. If
    this test starts failing, the fields have genuinely come apart and the
    computed one is the right one to trust — which is the reason it is computed.
    """
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n_t, n_c = int(rng.integers(2, 25)), int(rng.integers(1, 5))
        scale = 10.0 ** rng.integers(-6, 6)
        a = rng.normal(size=(n_t, n_c)) * scale
        b = a.copy()
        k = int(rng.integers(1, max(2, a.size // 2)))
        idx = rng.choice(a.size, size=k, replace=False)
        pert = 10.0 ** rng.uniform(-14, 1, size=k)
        b.flat[idx] = a.flat[idx] * (1 + pert) + pert * scale * rng.uniform(0, 1, size=k)
        flat = np.zeros(a.size, bool)
        flat[rng.choice(a.size, size=int(rng.integers(1, a.size + 1)), replace=False)] = True

        v = differ.deterministic_verdict(a, b, forgive_mask=flat.reshape(a.shape))
        decisive = v["passed"] and not v["passed_without_forgive"]
        assert decisive == (v["passed"] and v["n_forgive_rescued"] > 0)


# ── What "rescued" means, in `differ`'s own terms ───────────────────────────


def test_a_cell_the_near_zero_backstop_already_forgave_is_not_credited():
    """Attribution: the mask only gets credit where it was the deciding gate."""
    a = np.ones((50, 2))
    b = a.copy()
    a[0, 1] = 0.0
    b[0, 1] = 1e-30  # both sides far below the file-scale floor -> underflow
    mask = np.zeros_like(a, bool)
    mask[0, 1] = True

    v = differ.deterministic_verdict(a, b, forgive_mask=mask)
    assert v["n_fail"] == 0  # the backstop got there first
    assert v["n_forgive_rescued"] == 0
    assert v["passed_without_forgive"] is True


def test_a_one_side_nonfinite_cell_is_never_credited_as_rescued():
    """It is re-added to ``effective_fail`` unconditionally, so nothing rescued it."""
    a = np.ones((20, 2))
    b = a.copy()
    b[0, 1] = np.nan
    mask = np.ones_like(a, bool)  # try to forgive everything

    v = differ.deterministic_verdict(a, b, forgive_mask=mask)
    assert not v["passed"], "a one-side NaN must survive any forgive mask"
    assert v["n_forgive_rescued"] == 0
    assert v["passed_without_forgive"] is False


def test_a_mask_over_cells_that_were_not_failing_rescues_nothing():
    """The superset bug in its simplest form: masking agreement is free."""
    a = np.ones((20, 3))
    v = differ.deterministic_verdict(a, a.copy(), forgive_mask=np.ones_like(a, bool))
    assert v["n_forgive_rescued"] == 0
    assert v["passed_without_forgive"] is True


# ── The keys' presence is itself information ────────────────────────────────


def test_the_audit_keys_are_absent_when_no_mask_was_passed():
    """ "No mask" must not be readable as "the mask rescued nothing"."""
    a = np.zeros((50, 2))
    b = a.copy()
    b[0, 1] = 1e-11
    v = differ.deterministic_verdict(a, b)
    assert "n_forgive_rescued" not in v
    assert "passed_without_forgive" not in v


def test_sens_verdict_without_a_floor_reports_honest_zeros():
    """`sens_verdict` always fills all three, so a row's shape does not vary."""
    bn = np.zeros((6, 2, 2))
    am = np.zeros((6, 2, 2))
    am[:, :, 0] = 1e-11
    v = asens.sens_verdict(bn, am)  # no param_values/atol -> no floor
    assert not v["passed"]
    assert v["n_below_noise_floor"] == 0
    assert v["n_noise_rescued"] == 0
    assert v["noise_decisive"] is False


def test_the_misleading_name_is_gone_rather_than_redefined():
    """A census must never read a new definition off an old key.

    Every row of the shipped reports carries ``n_noise_forgiven`` under the
    superset meaning. Reusing the name for the rescued count would make the two
    generations of report silently incomparable.
    """
    bn = np.zeros((4, 2, 2))
    am = np.zeros((4, 2, 2))
    am[:, :, 0] = 1e-11
    v = asens.sens_verdict(bn, am, param_values=_P, atol=_ATOL)
    assert "n_noise_forgiven" not in v


# ── The refactor that made the counterfactual exact ─────────────────────────


@pytest.mark.parametrize(
    "case",
    ["agreement", "one_soft", "hard_divergence", "one_side_nan"],
    ids=lambda c: c,
)
def test_passing_an_empty_mask_matches_passing_no_mask(case):
    """``passed_without_forgive`` is the real verdict, not a re-derivation of it.

    The counterfactual runs the same code path with the mask emptied, so this
    pins that an empty mask and no mask agree on every reported field — the
    property that makes the second verdict trustworthy.
    """
    rng = np.random.default_rng(11)
    a = rng.normal(size=(30, 3)) * 10.0
    b = a.copy()
    if case == "one_soft":
        b[0, 1] += 1e-6
    elif case == "hard_divergence":
        b[:, 2] *= 5.0
    elif case == "one_side_nan":
        b[3, 0] = np.nan

    bare = differ.deterministic_verdict(a, b)
    empty = differ.deterministic_verdict(a, b, forgive_mask=np.zeros_like(a, bool))

    assert empty["passed_without_forgive"] == bare["passed"]
    assert {k: v for k, v in empty.items() if k in bare} == bare
