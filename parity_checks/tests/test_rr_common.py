"""Lock the rr_parity engine-agnostic plumbing (`_rr_common`).

These functions carry no model math — their contracts are structural:

align_common
  * compares over the INTERSECTION of species names, in a shared sorted order,
    with indices that actually point at the matching column in each engine
    (RoadRunner emits only floating species and neither column order is
    assumed, so name-keyed alignment is the whole job)
  * a fully disjoint species set returns None (a loud structural divergence,
    never a silent empty comparison)

schedule
  * runs every spec and returns results in SPEC order regardless of completion
    order (a slow early job must not reorder the report)
  * a job that overruns its cap is killed and reported "timeout", and one bad
    job does not stop the others from completing
"""

from __future__ import annotations

import os
from pathlib import Path

import _rr_common as rc
import pytest
import smoke_workers as sw


def _biomd3_xml() -> str:
    """Path to the committed BIOMD0000000003.xml (the only SBML fixture that
    ships in tests/data/; honors BNGSIM_TEST_DATA like the C++/python suites)."""
    base = os.environ.get("BNGSIM_TEST_DATA")
    data = Path(base) if base else Path(__file__).resolve().parents[2] / "tests" / "data"
    return str(data / "BIOMD0000000003.xml")


# --------------------------------------------------------------------------- #
# align_common
# --------------------------------------------------------------------------- #
def test_align_common_intersection_indices_point_at_matching_columns():
    """The returned indices must satisfy, for every k:
    common[k] == bn_names[bn_idx[k]] == rr_names[rr_idx[k]], over the sorted
    intersection — the only correctness property that matters for the diff."""
    bn = ["x", "y", "z", "w"]  # bngsim: superset, arbitrary order
    rr = ["z", "x", "q"]  # roadrunner: overlaps on {x, z}, has its own q
    bn_idx, rr_idx, common = rc.align_common(bn, rr)
    assert common == ["x", "z"]  # sorted intersection
    for k, name in enumerate(common):
        assert bn[bn_idx[k]] == name
        assert rr[rr_idx[k]] == name


def test_align_common_disjoint_returns_none():
    """No shared species ⇒ None (the caller surfaces this as a loud DIFF)."""
    assert rc.align_common(["a", "b"], ["c", "d"]) is None


def test_align_common_full_overlap_is_a_permutation():
    """Identical name sets in different orders ⇒ a complete alignment whose
    index maps are inverse permutations recovering the same species."""
    bn = ["c", "a", "b"]
    rr = ["a", "b", "c"]
    bn_idx, rr_idx, common = rc.align_common(bn, rr)
    assert common == ["a", "b", "c"]
    assert [bn[i] for i in bn_idx] == common
    assert [rr[i] for i in rr_idx] == common


# --------------------------------------------------------------------------- #
# schedule (spawn-based; workers live in smoke_workers so they pickle)
# --------------------------------------------------------------------------- #
def test_schedule_returns_results_in_spec_order():
    """Results come back keyed to the input spec order even though the workers
    finish concurrently. Oracle: the doubled value matches each spec's own
    value at the same index (a reordering would misalign them)."""
    specs = [{"key": f"k{i}", "value": i} for i in range(5)]
    out = rc.schedule(specs, sw.echo_worker, workers=3, timeout_of=lambda s: 30.0)
    assert [r["key"] for r in out] == [s["key"] for s in specs]
    assert [r["doubled"] for r in out] == [s["value"] * 2 for s in specs]


def test_schedule_kills_overrunning_job_and_completes_the_rest():
    """A job that sleeps past its cap is terminated and reported 'timeout'; the
    sibling jobs still complete normally. Confirms one pathological model can't
    wedge the screen."""
    specs = [
        {"key": "fast1", "value": 1, "cap": 30.0},
        {"key": "slow", "value": 2, "cap": 0.5},
        {"key": "fast2", "value": 3, "cap": 30.0},
    ]
    # The worker must be module-level (spawn pickles it by qualified name), so a
    # single dispatcher branches on the spec key rather than a per-spec closure.
    out = rc.schedule(specs, sw.branch_worker, workers=2, timeout_of=lambda s: s["cap"])
    by_key = {r["key"]: r for r in out}
    assert by_key["slow"]["status"] == "timeout"
    assert by_key["fast1"]["status"] == "ok" and by_key["fast2"]["status"] == "ok"


# --------------------------------------------------------------------------- #
# bn_ode timing config (T1.2): built from the engine's accessors, not guessed
# --------------------------------------------------------------------------- #
def test_bn_ode_config_is_built_from_engine_accessors():
    """The timing config must report what the engine actually chose (T0.1/T0.2/
    T0.4 accessors), not the old hardcoded guesses. Oracle: the values are the
    accessor token forms, never the retired "Band"/"CMake"/"Analytical" labels,
    and they match an independently-constructed Simulator on the same model."""
    bngsim = pytest.importorskip("bngsim")
    xml = _biomd3_xml()

    _t, _v, _n, timing = rc.bn_ode(xml, 0.0, 20.0, 21, rc.DEFAULT_RTOL, rc.DEFAULT_ATOL)
    cfg = timing["config"]

    # New accessor token forms — never the retired guesses.
    assert cfg["codegen"] in ("exprtk", "cc", "mir")
    assert cfg["jacobian"] in ("analytical", "fd", "jax")
    assert cfg["linear_solver"] in set(rc.LINEAR_SOLVER_NAMES.values())
    assert cfg["linear_solver"] != "Band"  # the bug this fixes

    # Built FROM the accessors: the config matches a fresh Simulator's reports.
    m = bngsim.Model.from_sbml(xml)
    sim = bngsim.Simulator(m, method="ode")
    assert cfg["codegen"] == sim.codegen_backend
    assert cfg["jacobian"] == sim.jacobian_strategy
    assert cfg["cached"] == sim.codegen_cache_hit
    # BIOMD0000000003 is tiny → ExprTk interpreter, dense LU, analytical Jacobian.
    assert cfg["codegen"] == "exprtk"
    assert cfg["linear_solver"] == "Dense"
    assert cfg["jacobian"] == "analytical"
    # ExprTk → no .so, so the definitive cache signal is None (not a wall-time
    # heuristic guess).
    assert cfg["cached"] is None


def test_bn_ode_timing_phases_are_non_overlapping_and_single_run():
    """Per-phase timing is sane: an ExprTk model reports codegen_sec == 0 and
    parse_interpret_sec is the load time (no codegen subtracted)."""
    pytest.importorskip("bngsim")
    _t, _v, _n, timing = rc.bn_ode(_biomd3_xml(), 0.0, 20.0, 21, rc.DEFAULT_RTOL, rc.DEFAULT_ATOL)
    assert timing["codegen_sec"] == 0.0  # ExprTk → no codegen
    assert timing["parse_interpret_sec"] > 0.0
    assert timing["integrate_sec"] > 0.0


# --------------------------------------------------------------------------- #
# --config combo definitions (T3.1) — KLU vs LAPACK-dense reachability
# --------------------------------------------------------------------------- #
def test_lapack_combo_forces_dense_to_reach_dgetrf():
    """SUNLinSol_LapackDense (LinearSolverKind 2) is only chosen on the DENSE
    path, so the lapack combo must force dense — otherwise a KLU-eligible model
    stays on KLU and dgetrf is never exercised (a no-op vs auto)."""
    import rr_run

    combos = rr_run._CONFIG_COMBOS
    lapack = combos["lapack"]
    assert lapack["env"].get("BNGSIM_LAPACK_DENSE") == "1"
    assert lapack["force_dense"] is True  # the fix: must force dense to reach dgetrf

    # force-dense is the built-in dense LU counterpart (no LAPACK env), so the
    # pair isolates dgetrf vs the SUNDIALS dense factor on the same models.
    assert combos["force-dense"]["force_dense"] is True
    assert "BNGSIM_LAPACK_DENSE" not in combos["force-dense"]["env"]

    # auto forces nothing.
    assert combos["auto"] == {"env": {}, "force_dense": False}


# --------------------------------------------------------------------------- #
# _ensemble_stats — SSA per-replicate ensemble timing (issue #135)
# --------------------------------------------------------------------------- #
class TestEnsembleStats:
    def test_distribution_and_cold_warm_split(self):
        # rep-1 is the COLD trajectory (page-fault / CPU ramp); reps 2..N are warm.
        # The headline median is over ALL reps; the warm median excludes rep-1.
        s = rc._ensemble_stats([0.010, 0.002, 0.004, 0.002])
        assert s["n_rep"] == 4
        assert s["rep_cold_sec"] == 0.010  # first replicate, in seed order
        assert s["rep_min_sec"] == 0.002
        assert s["rep_max_sec"] == 0.010
        assert s["rep_median_sec"] == 0.003  # median([2,4,2,10] ms) = (2+4)/2
        assert s["rep_warm_median_sec"] == 0.002  # median([2,4,2] ms) of reps 2..N
        assert s["ensemble_sec"] == 0.018  # total integration wall over all reps
        assert s["rep_mean_sec"] == 0.0045

    def test_empty_is_no_reps_not_a_crash(self):
        # An engine that raised before the loop has no timed replicate.
        assert rc._ensemble_stats([]) == {"n_rep": 0}

    def test_single_replicate_has_no_warm(self):
        # With one replicate there is no warm sample → warm median is None, and
        # cold == the only rep. (Guards the reps[1:] slice on a length-1 list.)
        s = rc._ensemble_stats([0.005])
        assert s["n_rep"] == 1
        assert s["rep_cold_sec"] == 0.005
        assert s["rep_warm_median_sec"] is None
        assert s["ensemble_sec"] == 0.005
