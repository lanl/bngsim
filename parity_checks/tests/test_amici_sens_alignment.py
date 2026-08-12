"""Lock the amici_parity forward-sensitivity plumbing (`_amici_sens`).

Everything here is the *structural* machinery around the sensitivity comparison
— no model math, no engine required. Three contracts, each of which has a
failure mode that would silently corrupt a whole sweep rather than announce
itself:

bn_param_to_sbml_id / shared_sensitivity_params
  The two engines name SBML **local** (per-``kineticLaw``) parameters
  differently — bngsim ``_lp_J0_V1``, AMICI ``J0_V1``. Get the mapping wrong and
  the intersection comes out empty on every local-parameter model, which the
  runner reports as BAD_TEST: a whole corpus quietly not being compared, looking
  like "no oracle" rather than "the adapter is broken".

select_params
  The cap must be deterministic across re-runs and spread over the model, not
  clustered on whichever reaction sorts first.

flatten_tensor
  The (n_t, n_species, n_param) -> (n_t, n_species*n_param) reshape feeds
  ``_core.differ``, which is column-oriented. A transposed or mis-strided
  flatten still has the right *shape*, so nothing would raise — the verdict
  would just be computed against the wrong pairing.

ensure_build_path
  A regression guard: the venv's script dir must be discoverable via
  ``sysconfig``, not via ``Path(sys.executable).resolve().parent`` (which in a
  uv venv resolves the symlink out to the base interpreter's bin, where AMICI's
  pinned swig is not). When that entry is wrong, cmake cannot find SWIG and
  EVERY model lands in REFERENCE_FAILED/compile — a sweep of false negatives
  that reads as an AMICI feature gap.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig

import _amici_sens as asens
import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Parameter-name mapping
# --------------------------------------------------------------------------- #
class TestParamNameMapping:
    def test_local_parameter_prefix_is_stripped(self):
        assert asens.bn_param_to_sbml_id("_lp_J0_V1") == "J0_V1"

    def test_global_parameter_passes_through(self):
        assert asens.bn_param_to_sbml_id("tau_mRNA") == "tau_mRNA"

    def test_only_the_leading_prefix_is_stripped(self):
        """A parameter whose own name embeds the marker keeps it — the prefix is
        positional, not a substring to scrub."""
        assert asens.bn_param_to_sbml_id("_lp_J0__lp_x") == "J0__lp_x"


class TestSharedSensitivityParams:
    def test_matches_locals_across_the_naming_gap(self):
        """The case the whole mapping exists for: bngsim's flattened local names
        line up with AMICI's ids, and the returned map points back at the name
        bngsim itself wants."""
        shared, bn_by_id, n_cand = asens.shared_sensitivity_params(
            ["_lp_J0_V1", "_lp_J0_Ki", "uVol"],
            ["uVol"],
            ["J0_V1", "J0_Ki"],
        )
        assert shared == ["J0_Ki", "J0_V1"]
        assert bn_by_id == {"J0_V1": "_lp_J0_V1", "J0_Ki": "_lp_J0_Ki"}
        assert n_cand == 2

    def test_compartment_size_params_are_excluded(self):
        """bngsim refuses compartment-size parameters as sensitivity targets, so
        one must never reach the shared list even when AMICI offers it."""
        shared, _bn, _n = asens.shared_sensitivity_params(["uVol", "k1"], ["uVol"], ["uVol", "k1"])
        assert shared == ["k1"]

    def test_amici_fixed_parameters_are_excluded(self):
        """Only AMICI's FREE ids are passed in; a parameter bngsim knows but AMICI
        holds fixed has no sx column, so it cannot be compared."""
        shared, _bn, _n = asens.shared_sensitivity_params(["k1", "k2"], [], ["k1"])
        assert shared == ["k1"]

    def test_no_overlap_yields_an_empty_list(self):
        """Empty is a legitimate outcome (the runner turns it into BAD_TEST), not
        an exception — a model with nothing differentiable in common simply has no
        oracle."""
        shared, bn_by_id, n_cand = asens.shared_sensitivity_params(["a"], [], ["b"])
        assert shared == [] and bn_by_id == {} and n_cand == 0

    def test_a_colliding_stripped_name_is_dropped_not_guessed(self):
        """If a global parameter and a stripped local name collide, which quantity
        an AMICI id refers to is ambiguous. Comparing the wrong pair would produce
        a confident-looking DIFF, so the id is dropped instead."""
        shared, _bn, _n = asens.shared_sensitivity_params(
            ["J0_V1", "_lp_J0_V1", "k1"], [], ["J0_V1", "k1"]
        )
        assert shared == ["k1"]

    def test_cap_is_reported_against_the_uncapped_candidate_count(self):
        """``n_candidates`` must reflect what was shared BEFORE the cap, so the
        report can disclose what the cap dropped instead of silently truncating."""
        ids = [f"k{i:02d}" for i in range(30)]
        shared, _bn, n_cand = asens.shared_sensitivity_params(ids, [], ids, cap=5)
        assert len(shared) == 5
        assert n_cand == 30


class TestSelectParams:
    def test_under_the_cap_everything_is_kept_sorted(self):
        assert asens.select_params(["b", "a", "c"], 10) == ["a", "b", "c"]

    def test_cap_zero_means_uncapped(self):
        ids = [f"k{i:02d}" for i in range(40)]
        assert asens.select_params(ids, 0) == sorted(ids)

    def test_selection_is_deterministic_and_order_independent(self):
        """Sorting first makes the pick independent of either engine's internal
        ordering — the same set on a re-run, and on a machine where AMICI happens
        to enumerate its free parameters differently."""
        ids = [f"k{i:02d}" for i in range(30)]
        a = asens.select_params(ids, 7)
        b = asens.select_params(list(reversed(ids)), 7)
        assert a == b == asens.select_params(ids, 7)

    def test_selection_spreads_over_the_id_range(self):
        """Not the alphabetically-first N: a sorted SBML id list clusters by
        reaction prefix, so taking a prefix would sample one reaction's locals and
        call it a parameter sweep."""
        ids = [f"J{i // 10}_p{i % 10}" for i in range(60)]
        chosen = asens.select_params(ids, 6)
        prefixes = {c.split("_")[0] for c in chosen}
        assert len(chosen) == 6
        assert len(prefixes) >= 5, f"selection clustered on {prefixes}"

    def test_endpoints_are_included(self):
        ids = [f"k{i:02d}" for i in range(20)]
        chosen = asens.select_params(ids, 4)
        assert chosen[0] == "k00" and chosen[-1] == "k19"

    def test_no_duplicates(self):
        ids = [f"k{i:02d}" for i in range(11)]
        chosen = asens.select_params(ids, 7)
        assert len(chosen) == len(set(chosen))


class TestFlattenTensor:
    def test_time_axis_is_preserved_and_columns_are_species_param_pairs(self):
        sx = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        flat = asens.flatten_tensor(sx)
        assert flat.shape == (2, 12)
        # Column (i*n_param + j) must be exactly the (species i, param j) series —
        # this is the pairing differ's per-column terms are computed against.
        for i in range(3):
            for j in range(4):
                assert np.array_equal(flat[:, i * 4 + j], sx[:, i, j])

    def test_a_single_parameter_is_not_squeezed_away(self):
        sx = np.zeros((5, 3, 1))
        assert asens.flatten_tensor(sx).shape == (5, 3)


class TestSensVerdict:
    def test_identical_tensors_pass(self):
        rng = np.random.default_rng(0)
        sx = rng.normal(size=(6, 3, 2))
        v = asens.sens_verdict(sx, sx.copy())
        assert v["passed"]

    def test_one_corrupted_parameter_column_fails(self):
        """The BIOMD0000000012 shape: every parameter agrees except one whose
        column is wrong throughout. That must not be forgiven by the fail-fraction
        budget just because it is a minority of cells."""
        rng = np.random.default_rng(1)
        a = rng.normal(size=(20, 4, 5)) * 100
        b = a.copy()
        b[:, :, 2] *= 3.0
        assert not asens.sens_verdict(a, b)["passed"]

    def test_a_nan_column_against_finite_values_fails(self):
        """The literal GH #310 signature — bngsim NaN, AMICI finite. A one-sided
        non-finite cell is an unconditional hard fail in differ; assert it here so
        the sensitivity path can never regress into silently passing it."""
        a = np.ones((10, 3, 2))
        b = np.ones((10, 3, 2))
        a[:, :, 1] = np.nan
        assert not asens.sens_verdict(a, b)["passed"]


class TestEnsureBuildPath:
    def test_returns_this_environments_script_dir(self):
        assert asens.ensure_build_path() == sysconfig.get_path("scripts")

    def test_the_script_dir_is_first_on_path_and_not_duplicated(self):
        saved = os.environ.get("PATH", "")
        try:
            bindir = asens.ensure_build_path()
            asens.ensure_build_path()
            parts = os.environ["PATH"].split(os.pathsep)
            assert parts[0] == bindir
            assert parts.count(bindir) == 1
        finally:
            os.environ["PATH"] = saved

    def test_it_does_not_resolve_the_interpreter_symlink(self):
        """Regression guard. ``Path(sys.executable).resolve().parent`` was the
        first implementation; in a uv-created venv that follows
        ``.venv/bin/python`` out to the base interpreter, whose bin has no swig,
        so cmake failed to find SWIG and every model became
        REFERENCE_FAILED/compile. Only meaningful where the two actually differ.
        """
        from pathlib import Path

        resolved = Path(sys.executable).resolve().parent
        scripts = Path(sysconfig.get_path("scripts"))
        if resolved == scripts:
            pytest.skip("interpreter is not a symlink in this environment")
        assert Path(asens.ensure_build_path()) == scripts

    def test_swig_is_reachable_once_the_path_is_set(self):
        """The end the fix exists for: after the call, cmake's PATH search for
        SWIG succeeds. Skipped where the amici group is not installed."""
        saved = os.environ.get("PATH", "")
        try:
            bindir = asens.ensure_build_path()
            if not (shutil.which("swig", path=bindir) or shutil.which("swig.exe", path=bindir)):
                pytest.skip("could not import amici's pinned swig (dependency group absent)")
            assert shutil.which("swig") is not None
        finally:
            os.environ["PATH"] = saved


# --------------------------------------------------------------------------- #
# Job expansion / ordering
# --------------------------------------------------------------------------- #
class _FakeJob:
    """The two Job attributes make_specs reads."""

    def __init__(self, model_id, overrides=()):
        self.model_id = model_id
        self.model = f"models/{model_id}/{model_id}.xml"
        self.params = {"t_start": 0.0, "t_end": 10.0, "n_points": 11}
        self.overrides = list(overrides)


class TestMakeSpecs:
    @staticmethod
    def _specs(models, methods):
        import amici_sens_run as run

        return run.make_specs(
            [_FakeJob(m) for m in models],
            list(methods),
            rtol=1e-9,
            atol=1e-12,
            timeout=None,
            param_cap=20,
            config_env={},
        )

    def test_expands_to_one_spec_per_model_and_method(self):
        specs, _ = self._specs(["A", "B", "C"], ["staggered", "simultaneous"])
        assert len(specs) == 6
        assert len({s["key"] for s in specs}) == 6

    def test_ordering_is_method_major(self):
        """The race-avoidance property. Model-major order would put a model's two
        methods in flight together and make several workers duplicate the same
        expensive C++ compile; method-major guarantees the second pass is a cache
        hit. Asserted on the order because that is the only thing enforcing it."""
        specs, _ = self._specs(["A", "B", "C"], ["staggered", "simultaneous"])
        methods_in_order = [s["sens_method"] for s in specs]
        assert methods_in_order == ["staggered"] * 3 + ["simultaneous"] * 3

    def test_one_method_is_unaffected(self):
        specs, _ = self._specs(["A", "B"], ["staggered"])
        assert [s["model_id"] for s in specs] == ["A", "B"]

    def test_tol_override_is_counted_once_per_model_not_per_method(self):
        """Two methods of one overridden model are ONE overridden model; counting
        per spec would report double the truth in the report's overrides block."""
        import amici_sens_run as run

        class _Ov:
            field = "tol"
            value = {"rtol": 1e-6, "atol": 1e-9}

        specs, n_ov = run.make_specs(
            [_FakeJob("A", overrides=[_Ov()]), _FakeJob("B")],
            ["staggered", "simultaneous"],
            rtol=1e-9,
            atol=1e-12,
            timeout=None,
            param_cap=20,
            config_env={},
        )
        assert n_ov == 1
        a_specs = [s for s in specs if s["model_id"] == "A"]
        assert all(s["params"]["rtol"] == 1e-6 for s in a_specs)
        assert all(s["params"]["rtol"] == 1e-9 for s in specs if s["model_id"] == "B")


# --------------------------------------------------------------------------- #
# Solver-resolution noise floor
# --------------------------------------------------------------------------- #
class TestSensitivityNoiseFloor:
    """The floor exists for one specific, common situation that no scale-relative
    tolerance can handle: a sensitivity that is *identically zero*.

    A parameter that does not influence a species has dx/dp == 0. One engine
    returns exact 0.0, the other returns its own integration noise, and
    |a-b|/max(|a|,|b|) is then exactly 1.0 however tiny both numbers are — the
    ratio is scale-free, so differ's per-column and file-peak terms cannot forgive
    it. Observed on BIOMD0000000569, where finite differences on bngsim's own
    trajectories confirm the true value is 0, bngsim returns 0, and AMICI returns
    ~1e-11.

    The floor must NOT be a blanket softening: the two guards below (a real
    large-magnitude divergence, and the GH #310 NaN column) are the properties
    that keep it honest.
    """

    ATOL = 1e-12

    def test_zero_derivative_versus_reference_noise_is_forgiven(self):
        p = [1.0, 1.0]
        bn = np.zeros((8, 3, 2))
        am = np.zeros((8, 3, 2))
        am[:, :, 1] = 1e-11  # reference noise where the true derivative is 0
        assert not asens.sens_verdict(bn, am)["passed"], "precondition: fails without the floor"
        v = asens.sens_verdict(bn, am, param_values=p, atol=self.ATOL)
        assert v["passed"]
        assert v["n_noise_forgiven"] > 0

    def test_a_real_large_magnitude_divergence_still_fails(self):
        """BIOMD0000000457's shape: one parameter column completely wrong at a
        magnitude far above any solver noise. The floor must not touch it."""
        rng = np.random.default_rng(3)
        bn = rng.normal(size=(12, 4, 3)) * 1e6
        am = bn.copy()
        am[:, :, 1] *= 2.0
        v = asens.sens_verdict(bn, am, param_values=[1.0] * 3, atol=self.ATOL)
        assert not v["passed"]

    def test_nan_column_still_fails_with_the_floor_active(self):
        """GH #310's signature. differ never forgives a one-sided non-finite cell,
        and the floor must not create a path around that — a blow-up is not noise."""
        bn = np.ones((10, 3, 2))
        am = np.ones((10, 3, 2))
        bn[:, :, 1] = np.nan
        assert not asens.sens_verdict(bn, am, param_values=[1.0, 1.0], atol=self.ATOL)["passed"]

    def test_floor_scales_inversely_with_the_parameter_value(self):
        """CVODES resolves s_ij only to atol/|p_j|, so a large-valued parameter has
        a TIGHTER floor in sx units. Getting this backwards would forgive real
        differences on exactly the parameters measured most precisely."""
        big = asens.sensitivity_noise_mask(
            np.zeros((2, 1, 1)), np.full((2, 1, 1), 1e-9), [1e6], self.ATOL
        )
        small = asens.sensitivity_noise_mask(
            np.zeros((2, 1, 1)), np.full((2, 1, 1), 1e-9), [1e-6], self.ATOL
        )
        assert not big.any(), "a 1e-9 gap must NOT be noise for |p|=1e6"
        assert small.all(), "a 1e-9 gap must be noise for |p|=1e-6"

    def test_zero_valued_parameter_falls_back_to_raw_atol(self):
        """|p| = 0 has no atol/|p| scaling. Falling back to atol is the
        conservative choice — it forgives less than any 0 < |p| < 1 would."""
        mask = asens.sensitivity_noise_mask(
            np.zeros((2, 1, 1)), np.full((2, 1, 1), 1e-9), [0.0], self.ATOL
        )
        assert not mask.any()

    def test_omitting_the_inputs_reproduces_the_unfloored_verdict(self):
        """Back-compat: a caller that does not know the parameter values gets
        exactly the original scale-only behavior, and an honest zero count."""
        bn = np.zeros((6, 2, 2))
        am = np.zeros((6, 2, 2))
        am[:, :, 0] = 1e-11
        v = asens.sens_verdict(bn, am)
        assert not v["passed"]
        assert v["n_noise_forgiven"] == 0
