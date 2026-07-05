"""Hermetic tests for the SSA biomodels parity screen's low-count z-gate floor.

The broad SSA parity screen (``parity_checks/rr_parity/ssa_screen.py``) gates
each (t>0, species) cell on a cross-engine mean z-score. On near-deterministic
sub-particle cells the z-test blows up: a species at <=1 molecule has ~zero
replicate variance, so a 1-molecule MC difference reads as a many-sigma "fail"
that is noise, not a real divergence (GH #107.3). ``_compare`` therefore skips a
cell when *both* engines' means convert to <= ``min_mean_count`` molecules
(amount = concentration x compartment volume).

These tests pin the contract that matters: the floor must clear the low-count
noise *without* masking a real divergence -- a one-sided cell (one engine holds
a real population, the other is near zero) and a genuine two-sided divergence at
meaningful counts both still fail. They exercise the pure ``_compare`` numerics
directly (no SBML, no engines), so they run anywhere the package tests do.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load the screen module by path -- it is dev tooling outside the package, but
# self-bootstraps its own imports (it inserts parity_checks/ and its own dir on
# sys.path before ``from _core import ...`` / ``import _rr_common``), so loading
# the file is enough.
_HARNESS = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "ssa_screen.py"
if not _HARNESS.exists():  # pragma: no cover - layout guard
    pytest.skip(f"screen not found at {_HARNESS}", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("_ssa_biomodels_parity_screen", _HARNESS)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_compare = _mod._compare
_classify_diff = _mod._classify_diff
_cell_tracks_own_ode = _mod._cell_tracks_own_ode
_odes_agree_on_cell = _mod._odes_agree_on_cell
_sign_indefinite_annotation = _mod._sign_indefinite_annotation
DEFAULT_FLOOR = _mod.DEFAULT_MIN_MEAN_COUNT
SELF_Z_TOL = _mod.DEFAULT_SELF_Z_TOL
ODE_REL_TOL = _mod.ODE_REL_TOL


def _cell(bn_mean, rr_mean, *, species="A", n=30, jitter=0.5):
    """One-species, one-comparison-time (t0 + t1) replicate arrays whose t1
    column means are exactly ``bn_mean`` / ``rr_mean`` with nonzero variance.

    Returns the args ``_compare`` consumes: (times, bn_arr, names, rr_arr).
    A symmetric +/- ``jitter`` on two replicates gives a realistic standard
    error without an RNG, so the z-score is deterministic.
    """
    times = np.array([0.0, 1.0])
    bn = np.zeros((n, 2, 1))
    rr = np.zeros((n, 2, 1))
    bn[:, 1, 0] = bn_mean
    rr[:, 1, 0] = rr_mean
    for arr in (bn, rr):
        arr[0, 1, 0] += jitter
        arr[1, 1, 0] -= jitter
    return times, bn, rr, [species]


def _run(bn_mean, rr_mean, *, volume=1.0, floor=DEFAULT_FLOOR, tol=5.0, jitter=0.5):
    times, bn, rr, names = _cell(bn_mean, rr_mean, jitter=jitter)
    return _compare(
        times,
        bn,
        names,
        times,
        rr,
        names,
        tol,
        50,
        species_volumes={names[0]: volume},
        min_mean_count=floor,
    )


def test_both_engines_low_count_cell_is_skipped_not_failed():
    # bn extinct (0), rr ~1 molecule: the 1053-Gi artifact. Without the floor
    # this is a huge-z "fail"; with it the cell is skipped and tallied.
    res = _run(0.0, 1.0, jitter=0.2)
    assert res["status"] == "pass"
    assert res["n_skipped_lowcount"] == 1
    assert res["n_compared"] == 0
    assert res["n_z_failures"] == 0


def test_genuine_two_sided_divergence_at_meaningful_counts_fails():
    # 863-dTeff shape: ~100 vs ~28 molecules. Above the floor -> tested -> fail.
    res = _run(100.0, 28.0)
    assert res["status"] == "fail"
    assert res["n_skipped_lowcount"] == 0
    assert res["n_compared"] == 1
    assert res["n_z_failures"] == 1
    f = res["z_failures"][0]
    assert f["bn_count"] == pytest.approx(100.0)
    assert f["rr_count"] == pytest.approx(28.0)


def test_one_sided_divergence_is_not_masked():
    # The key anti-masking guarantee: bn extinct, rr holds 50 molecules
    # (an RR-side population artifact). max(count) > floor, so it still fails.
    res = _run(0.0, 50.0)
    assert res["status"] == "fail"
    assert res["n_skipped_lowcount"] == 0


def test_floor_uses_molecule_count_not_concentration_tiny_volume():
    # Large concentration but a tiny compartment -> amount <= floor -> skipped.
    # 1000 conc x 1e-3 vol = 1 molecule. Proves the conversion uses volume.
    res = _run(0.0, 1000.0, volume=1e-3, jitter=0.2)
    assert res["status"] == "pass"
    assert res["n_skipped_lowcount"] == 1


def test_floor_uses_molecule_count_not_concentration_large_volume():
    # Small concentration but a large compartment -> amount > floor -> tested.
    # 0.5 conc x 10 vol = 5 molecules. Without the volume factor (0.5 < 2) this
    # would wrongly skip; with it the divergence is correctly surfaced.
    res = _run(0.0, 0.5, volume=10.0)
    assert res["status"] == "fail"
    assert res["n_compared"] == 1


def test_above_floor_agreeing_cell_passes_and_is_counted():
    # Both engines at 50, identical: compared (not skipped), z ~ 0, pass.
    res = _run(50.0, 50.0, jitter=0.0)
    assert res["status"] == "pass"
    assert res["n_skipped_lowcount"] == 0
    assert res["n_compared"] == 1
    assert res["max_mean_z"] == pytest.approx(0.0)


def test_default_volume_is_unit_when_map_omits_species():
    # species_volumes=None -> volume defaults to 1.0 (concentration == count).
    times, bn, rr, names = _cell(0.0, 1.0, jitter=0.2)
    res = _compare(times, bn, names, times, rr, names, 5.0, 50)
    assert res["status"] == "pass"
    assert res["n_skipped_lowcount"] == 1


# --- DIFF attribution classifier (_classify_diff) -------------------------
#
# The oracle aggregates three booleans over the failing cells: does bngsim SSA
# track bngsim ODE, does RR SSA track RR ODE, and do the two ODEs agree. The
# classifier maps those to a status.


def test_classify_rr_side_diff_is_not_bngsim():
    # bngsim tracks its ODE, RR does not, ODEs agree -> explained, RR-side.
    assert _classify_diff(True, False, True)[0] == "diff_not_bngsim"


def test_classify_bngsim_suspect_stays_fail():
    # bngsim's own SSA does not track bngsim's ODE -> bngsim-suspect, stays red,
    # regardless of RR.
    assert _classify_diff(False, False, True)[0] == "fail"
    assert _classify_diff(False, True, True)[0] == "fail"


def test_classify_ode_level_takes_precedence():
    # ODEs disagree -> ODE/loader-level; do not blame either SSA engine, even if
    # one SSA also looks off.
    assert _classify_diff(True, False, False)[0] == "diff_ode_level"
    assert _classify_diff(False, False, False)[0] == "diff_ode_level"


def test_classify_both_track_odes_yet_differ_is_unexplained():
    # Both SSAs track their (agreeing) ODEs but still differ across engines ->
    # cannot attribute, stays fail.
    assert _classify_diff(True, True, True)[0] == "fail"


# --- molecule-aware cell predicates ---------------------------------------
#
# These are what keep the attribution from being fooled by the two artifacts
# the manual triage of MODEL1001150000 exposed (see dev notes).


def test_cell_tracks_via_zscore_when_variance_present():
    # SSA mean 20.4 vs ODE 20.8, SE 0.2 -> ~2 SE -> tracks (within self_z_tol).
    assert _cell_tracks_own_ode(20.4, 20.8, 0.2, 1.0, SELF_Z_TOL, DEFAULT_FLOOR)
    # SSA mean 0 vs ODE 20, SE tiny -> way beyond z AND beyond the floor -> not.
    assert not _cell_tracks_own_ode(0.0, 20.0, 1e-9, 1.0, SELF_Z_TOL, DEFAULT_FLOOR)


def test_cell_tracks_subparticle_ode_via_molecule_floor():
    # The MODEL1001150000 `Ca` trap: bngsim SSA=0 (exact extinction), bngsim
    # ODE=0.107 molecule, SE=0. The z-score is infinite but the molecule gap is
    # 0.107 <= floor, so bngsim correctly *tracks* its ODE here (not a bug).
    assert _cell_tracks_own_ode(0.0, 0.107, 0.0, 1.0, SELF_Z_TOL, DEFAULT_FLOOR)


def test_odes_agree_ignores_stiff_floor_factor_gap():
    # The KCaMcomplex trap: 1e-11 vs 1.5e-12 -> "relatively" 0.85 apart but
    # absolutely ~0; molecule gap << floor, so the ODEs AGREE.
    assert _odes_agree_on_cell(1.02e-11, 1.5e-12, 1.0, ODE_REL_TOL, DEFAULT_FLOOR)
    # A real ODE-level gap (20 vs 5 molecules, 75% relative) -> disagree.
    assert not _odes_agree_on_cell(20.0, 5.0, 1.0, ODE_REL_TOL, DEFAULT_FLOOR)


# --- runtime sign-indefinite annotation (GH #109) --------------------------
#
# The hook reads bngsim's own GH #110 SSA boundary diagnostics: if a rate law
# went negative on the trajectory, the channel was fired in reverse, which the
# static t0 corpus gate cannot detect. The annotation is context only -- it must
# never imply a verdict.


def test_sign_indefinite_none_when_no_reverse_fires():
    # A clean model (no negative rates) gets no annotation.
    clean = {"n_reverse_fires": 0, "first_reverse_reaction": ""}
    assert _sign_indefinite_annotation(clean) is None
    # Robust to a missing key (non-SSA / default-constructed diagnostics).
    assert _sign_indefinite_annotation({}) is None
    assert _sign_indefinite_annotation(None) is None


def test_sign_indefinite_reports_reaction_and_count():
    # The BIOMD863 shape: R8 fired in reverse 1213 times.
    ann = _sign_indefinite_annotation(
        {"n_reverse_fires": 1213, "first_reverse_reaction": "R8 (0 -> nTeff)"}
    )
    assert ann is not None
    assert ann["reverse_fires"] == 1213
    assert ann["reaction"] == "R8 (0 -> nTeff)"
    assert "GH #110" in ann["note"] and "GH #109" in ann["note"]


def test_sign_indefinite_falls_back_when_reaction_name_missing():
    # reverse fires recorded but the name field is empty -> still annotate.
    ann = _sign_indefinite_annotation({"n_reverse_fires": 5, "first_reverse_reaction": ""})
    assert ann is not None and ann["reaction"] == "(unknown)"
