"""Lock the golden-reference reductions (`_core/fingerprint.py`).

`checksum` and `fingerprint_max_rel` are the two gates a consumer regenerating
a golden job runs (`bng_parity/golden/README.md`): the checksum must match
byte-for-byte on the same platform, and when it cannot — different BLAS,
different rounding — `fingerprint_max_rel` is the cross-platform fallback,
compared against a tolerance. Both are therefore only ever consulted about a
run that already disagrees, so their failure mode that matters is a *false
pass*.

checksum
  * identical input hashes identically; a sig-fig-level perturbation does not
    change it, but a perturbation above the sig-fig floor does
  * the three blow-up modes (+inf, -inf, NaN) hash to three DIFFERENT digests
    — the issue #572 regression guard for collapsing them all to NaN, which let
    a run that blew up one way byte-match a golden that blew up another
  * every committed golden trajectory still reproduces the checksum recorded
    for it in `golden.json` (the non-finite handling must not touch finite data)

fingerprint_max_rel
  * identical fingerprints score 0.0; a finite difference scores the relative
    difference the name promises (oracle, not a frozen number)
  * the `abs_floor` keeps a vanishing denominator from exploding the score
  * a stat that is non-finite on exactly one side scores inf — the issue #572
    regression guard for the NaN-through-the-ratio path that scored the worst
    possible divergence 0.0, in either argument order, whole-column or only in
    the final point
  * two DIFFERENT blow-ups (+inf vs -inf, inf vs NaN) also score inf
  * the SAME blow-up on both sides (both NaN, both +inf, both -inf) agrees: 0.0
  * a var-set or n_time mismatch is structural: inf
  * end to end, a regenerated run whose variable went NaN fails BOTH gates
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from _core import fingerprint as fp

_GOLDEN = Path(__file__).resolve().parent.parent / "bng_parity" / "golden"

TIME = np.linspace(0.0, 1.0, 5)
NAMES = ["A", "B"]


def _values(b_col) -> np.ndarray:
    """A 5x2 block: column A finite and flat, column B whatever is under test."""
    v = np.tile([1.0, 2.0], (TIME.size, 1))
    v[:, 1] = b_col
    return v


def _fp(b_col) -> dict:
    return fp.fingerprint(TIME, _values(b_col), NAMES)


# --------------------------------------------------------------------------- #
# checksum
# --------------------------------------------------------------------------- #
def test_checksum_is_stable_and_sensitive_at_the_sigfig_floor():
    """Identical input hashes identically; below CHECKSUM_SIGFIGS is noise, above it is a change."""
    good = _values(2.0)
    assert fp.checksum(TIME, good, NAMES) == fp.checksum(TIME, good.copy(), NAMES)

    rel = 10.0 ** -(fp.CHECKSUM_SIGFIGS + 2)
    below = good.copy()
    below[2, 1] *= 1.0 + rel
    assert fp.checksum(TIME, below, NAMES) == fp.checksum(TIME, good, NAMES)

    above = good.copy()
    above[2, 1] *= 1.0 + 10.0 ** -(fp.CHECKSUM_SIGFIGS - 2)
    assert fp.checksum(TIME, above, NAMES) != fp.checksum(TIME, good, NAMES)


def test_the_three_blowup_modes_hash_distinctly():
    """+inf, -inf and NaN are three different outcomes and must be three digests.

    Regression guard (issue #572): `_round_sig` used to canonicalize every
    non-finite cell to NaN, so `checksum(+inf) == checksum(-inf) == checksum(NaN)`.
    The checksum is the *strongest* gate — a match ends the comparison — so a
    regenerated run that blew up to NaN byte-matched a golden that blew up to
    +inf and was certified an exact reproduction.
    """
    digests = {
        kind: fp.checksum(TIME, _values(val), NAMES)
        for kind, val in (("+inf", np.inf), ("-inf", -np.inf), ("nan", np.nan))
    }
    assert len(set(digests.values())) == 3, digests
    # and each is still distinct from the finite run it diverged from
    assert fp.checksum(TIME, _values(2.0), NAMES) not in set(digests.values())


def test_nan_payload_is_canonicalized():
    """Two NaNs with different bit patterns must hash the same — a payload is not portable."""
    neg_nan = np.array([np.nan], dtype=float)
    neg_nan.view(np.uint64)[0] |= np.uint64(1) << np.uint64(63)  # sign bit: "-nan"
    assert neg_nan.view(np.uint64)[0] != np.array([np.nan]).view(np.uint64)[0]
    assert fp.checksum(TIME, _values(np.nan), NAMES) == fp.checksum(
        TIME, _values(float(neg_nan[0])), NAMES
    )


def _committed_jobs_with_trajectories():
    payload = json.loads((_GOLDEN / "golden.json").read_text())
    return [e for e in payload["golden"] if e.get("trajectory")]


def test_committed_goldens_still_reproduce_their_checksum():
    """The non-finite branch must not perturb a single byte of finite data.

    `golden.json` records a spanning checksum per job, built the way
    `parity_golden._build_from_files` builds it (sha256 over the sorted per-file
    checksums), and the allow-listed subset commits the rounded trajectory it
    was computed from. Re-hashing those trajectories is a direct test that a
    change to `_round_sig` has not invalidated the committed corpus.
    """
    jobs = _committed_jobs_with_trajectories()
    assert jobs, "no committed trajectories to check"
    for entry in jobs:
        traj = json.loads((_GOLDEN / entry["trajectory"]).read_text())
        per_file = {}
        for fname, tr in traj["artifacts"].items():
            t = np.asarray(tr["time"], float)
            v = np.asarray(tr["values"], float)
            assert np.isfinite(t).all() and np.isfinite(v).all()
            per_file[fname] = fp.checksum(t, v, list(tr["names"]))
        spanning = hashlib.sha256(
            "\n".join(f"{k}={per_file[k]}" for k in sorted(per_file)).encode()
        ).hexdigest()
        assert spanning == entry["checksum"], entry["model_id"]


# --------------------------------------------------------------------------- #
# fingerprint_max_rel — the finite contract
# --------------------------------------------------------------------------- #
def test_identical_fingerprints_score_zero():
    f = _fp(2.0)
    assert fp.fingerprint_max_rel(f, f) == 0.0


def test_finite_difference_scores_the_relative_difference():
    """The oracle: b scaled by (1+eps) is eps away from b, on every stat."""
    eps = 1e-3
    a, b = _fp(2.0 * (1.0 + eps)), _fp(2.0)
    assert fp.fingerprint_max_rel(a, b) == pytest.approx(eps, rel=1e-9)


def test_abs_floor_keeps_a_vanishing_denominator_from_exploding_the_score():
    """Two stats far below `abs_floor` differ by an absolute hair, not a huge ratio."""
    a, b = _fp(1e-18), _fp(1e-30)
    assert fp.fingerprint_max_rel(a, b, abs_floor=1e-9) == pytest.approx(1e-18 / 1e-9)


def test_structural_mismatch_is_inf():
    """A different var set or a different number of time points is not comparable."""
    a = _fp(2.0)
    renamed = fp.fingerprint(TIME, _values(2.0), ["A", "C"])
    assert fp.fingerprint_max_rel(a, renamed) == float("inf")
    shorter = fp.fingerprint(TIME[:4], _values(2.0)[:4], NAMES)
    assert fp.fingerprint_max_rel(a, shorter) == float("inf")


# --------------------------------------------------------------------------- #
# fingerprint_max_rel — the non-finite contract (issue #572)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("blown", [np.nan, np.inf, -np.inf], ids=["nan", "+inf", "-inf"])
@pytest.mark.parametrize("partial", [False, True], ids=["whole-column", "final-point"])
def test_one_side_non_finite_is_inf(blown, partial):
    """A blow-up on exactly one side is an unambiguous divergence, in either order.

    Regression guard (issue #572): `denom = max(abs(y), abs_floor)` and
    `worst = max(worst, ...)` are Python's builtin `max`, which keeps its first
    argument when the comparison is False. A non-finite stat made `denom`
    non-finite, the ratio NaN, and `max(0.0, nan)` 0.0 — the fallback reported
    PERFECT agreement for a run where a variable blew up. It is consulted only
    after a checksum mismatch, which a blow-up always produces, so this was the
    whole cross-platform gate.

    The `final-point` case is the stronger one: only the last time point is
    non-finite, so `min`/`max`/`mean` (which summarize the finite cells only)
    still agree exactly and just `last` diverges.
    """
    col = np.full(TIME.size, 2.0)
    col[-1 if partial else slice(None)] = blown
    good, bad = _fp(2.0), _fp(col)
    assert fp.fingerprint_max_rel(bad, good) == float("inf")
    assert fp.fingerprint_max_rel(good, bad) == float("inf")


@pytest.mark.parametrize("blown", [np.nan, np.inf, -np.inf], ids=["nan", "+inf", "-inf"])
def test_the_same_blowup_on_both_sides_agrees(blown):
    """Reproducing a blow-up exactly is agreement, not divergence: 0.0."""
    f = _fp(blown)
    assert fp.fingerprint_max_rel(f, f) == 0.0
    assert fp.fingerprint_max_rel(f, _fp(blown)) == 0.0


@pytest.mark.parametrize(
    "x, y",
    [(np.inf, -np.inf), (np.inf, np.nan), (-np.inf, np.nan)],
    ids=["+inf-vs--inf", "+inf-vs-nan", "-inf-vs-nan"],
)
def test_two_different_blowups_are_inf(x, y):
    """Both sides non-finite is not agreement unless it is the SAME non-finite."""
    assert fp.fingerprint_max_rel(_fp(x), _fp(y)) == float("inf")
    assert fp.fingerprint_max_rel(_fp(y), _fp(x)) == float("inf")


def test_a_blown_up_regeneration_fails_both_golden_gates():
    """End to end, the way `golden/README.md` says a consumer runs the comparison.

    Step 3 is the byte checksum; step 4 falls back to `fingerprint_max_rel`
    against a tolerance when the checksum cannot match across platforms. A run
    whose variable went NaN must fail both — before issue #572 it failed the
    first and then scored 0.0 on the second, which is the direction that turns a
    hard failure into a reported perfect match.
    """
    good, bad = _values(2.0), _values(np.nan)
    gold_cs, gold_fp = fp.golden_pair(TIME, good, NAMES)
    regen_cs, regen_fp = fp.golden_pair(TIME, bad, NAMES)
    assert regen_cs != gold_cs
    score = fp.fingerprint_max_rel(regen_fp, gold_fp)
    assert score == float("inf")
    assert not score <= 1e-6  # above ANY tolerance a caller could pick
