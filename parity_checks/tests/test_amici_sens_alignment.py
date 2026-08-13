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

_classify_failure / _is_declared_refusal
  A model bngsim *declares* it cannot differentiate must land in UNSUPPORTED,
  not EXCEPTION. EXCEPTION means "AMICI ran and bngsim broke" — the bucket a
  reader triages — so a documented capability gap sitting in it is pure noise.
  The recognition is by exception TYPE; the tests below pin that, because a
  message match would keep passing right up until someone rewords a refusal.
"""

from __future__ import annotations

import json
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


# --------------------------------------------------------------------------- #
# AMICI's `amici_` id-collision prefix (issue #321)
# --------------------------------------------------------------------------- #
class TestAmiciIdPrefix:
    """AMICI renames SBML ids that collide with its own generated C++ symbols
    (`x` IS the state vector there), so `<species id="x">` comes back as
    `amici_x`. An exact-id intersection then yields NOTHING and the job is
    reported as a structural loader divergence at value=inf — the loudest
    verdict the suite emits — on models where the engines actually agree.

    Species-side twin of the `_lp_` parameter prefix above, and it inherits the
    same two rules: the prefix is positional, and an ambiguous strip is dropped
    rather than guessed at.
    """

    @staticmethod
    def _ac():
        import _amici_common as ac

        return ac

    def test_the_biomd114_shape_now_aligns(self):
        """`BIOMD0000000114` declares `<species id="x">` / `<species id="y">`; the
        whole corpus regression is this one line."""
        got = self._ac().align_common(["x", "y"], ["amici_x", "amici_y"])
        assert got is not None
        bn_idx, am_idx, common = got
        assert common == ["x", "y"]
        assert bn_idx == [0, 1] and am_idx == [0, 1]

    def test_unprefixed_ids_are_untouched(self):
        got = self._ac().align_common(["A", "B"], ["A", "B"])
        assert got == ([0, 1], [0, 1], ["A", "B"])

    def test_the_prefix_is_positional_not_a_substring_to_scrub(self):
        """A species whose own id merely contains the marker keeps it."""
        assert self._ac().normalize_amici_names(["x_amici_y"]) == ["x_amici_y"]
        assert self._ac().normalize_amici_names(["amici_z"]) == ["z"]

    def test_an_ambiguous_strip_is_dropped_not_guessed(self):
        """A model declaring BOTH `x` and `amici_x` makes the strip ambiguous.
        Pairing the wrong series would produce a confident-looking DIFF, which is
        strictly worse than the loud non-comparison it would replace — so both
        are left raw."""
        assert self._ac().normalize_amici_names(["x", "amici_x"]) == ["x", "amici_x"]

    def test_the_returned_index_addresses_the_original_amici_order(self):
        """The caller slices `rdata` columns with `am_idx`, so it must index the
        ids AMICI actually reported — not the de-prefixed list. Getting this
        wrong silently transposes two species' trajectories."""
        bn_idx, am_idx, common = self._ac().align_common(["x", "y"], ["amici_y", "amici_x"])
        assert common == ["x", "y"]
        assert am_idx == [1, 0]

    def test_a_genuinely_disjoint_set_still_returns_none(self):
        """The de-prefixing must not manufacture a match where none exists —
        None is how a real loader divergence stays loud."""
        assert self._ac().align_common(["a"], ["amici_b"]) is None

    def test_the_shared_rr_helper_is_left_generic(self):
        """The rule is AMICI's, so it lives in the AMICI adapter. RoadRunner never
        emits the prefix, and teaching the shared helper about it would make
        rr_parity silently strip a legitimate `amici_`-named species."""
        from rr_parity import _rr_common as rc

        assert rc.align_common(["x"], ["amici_x"]) is None


# --------------------------------------------------------------------------- #
# Declared refusals -> UNSUPPORTED
# --------------------------------------------------------------------------- #
_REFUSAL_PROSE = (
    "Output sensitivities are not supported for this model's events: the "
    "event-time sensitivity dt*/dp is non-zero"
)


class TestIsDeclaredRefusal:
    """The recognition contract: by exception TYPE, never by message text.

    The alternative considered was matching the refusal's message prefix in the
    runner — no library change, no public API. These three tests are the reason
    it was rejected: the prefix and the type disagree in both directions, and
    only one of the two disagreements is loud.
    """

    def test_the_real_exception_is_recognized(self):
        import amici_sens_run as run
        import bngsim

        exc = bngsim.SensitivityUnsupportedError(_REFUSAL_PROSE)
        assert run._is_declared_refusal(exc)

    def test_a_reworded_refusal_is_still_recognized(self):
        """The property the typed exception buys. Both messages are long prose
        citing GH issue numbers and get edited; under a prefix match this row
        would silently revert to EXCEPTION, with no test able to catch it and no
        symptom except a quietly worse number in the next report."""
        import amici_sens_run as run
        import bngsim

        exc = bngsim.SensitivityUnsupportedError("completely different wording")
        assert run._is_declared_refusal(exc)

    def test_the_refusal_prose_on_a_plain_value_error_is_not_recognized(self):
        """The mirror image, and the reason type beats text even ignoring
        rewording: an unrelated site could quote or wrap this prose, and a
        prefix match would launder a real bngsim failure into a non-scoring
        row."""
        import amici_sens_run as run

        assert not run._is_declared_refusal(ValueError(_REFUSAL_PROSE))

    def test_an_ordinary_bngsim_failure_is_not_recognized(self):
        import amici_sens_run as run
        import bngsim

        assert not run._is_declared_refusal(bngsim.SimulationError("CVODE failed"))
        assert not run._is_declared_refusal(RuntimeError("no C compiler"))


class TestUnderSpecifiedIsAlsoDeclared:
    """Issue #323. `UnderSpecifiedModelError` is raised at LOAD — the model reads
    a symbol it never defines — so it reaches BOTH amici jobs, not just the
    sensitivity one. AMICI accepts the same models by defaulting the symbol to 0,
    which is why bngsim looks like the one that "broke".

    It is not a bngsim bug and not a capability gap in the sensitivity machinery;
    it is bngsim declining to guess. 12 corpus models sat in EXCEPTION for this.
    """

    def test_it_is_recognized_as_a_declared_refusal(self):
        import _amici_common as ac
        import bngsim

        assert ac.is_declared_refusal(bngsim.UnderSpecifiedModelError("no value for 'k'"))

    def test_a_plain_model_error_is_not(self):
        """The guard that keeps the bucket meaningful: `ModelError` also covers
        .net parse failures and invalid model state, which ARE actionable."""
        import _amici_common as ac
        import bngsim

        assert not ac.is_declared_refusal(bngsim.ModelError("failed to parse .net"))

    def test_the_ode_runner_classifies_it_too(self):
        """`align_common`'s lesson, applied to the taxonomy: this refusal is
        load-time, so the state-parity job hits it identically. Fixing only the
        sensitivity runner would leave the ODE matrix mislabelled."""
        import amici_run as run

        assert run._classify_failure("bngsim: UnderSpecifiedModelError: x", "", True)[0] == (
            "unsupported"
        )
        assert run._classify_failure("bngsim: ModelError: x", "")[0] == "exception"

    def test_the_ode_runner_flag_defaults_off(self):
        import amici_run as run

        assert run._classify_failure("bngsim: x", "")[0] == "exception"
        assert run._classify_failure("", "amici: x")[0] == "reference_failed"
        assert run._classify_failure("bngsim: x", "amici: y")[0] == "bad_test"


class TestDegeneracyWitnesses:
    """Issue #328. A job can be PASS when the whole sensitivity tensor lies below
    what either solver resolves — nothing was compared, but the row looks like a
    real pass.

    The floor is applied PER PARAMETER COLUMN. Reducing it with `max` over the
    parameter axis (the first thing I wrote) lets one tiny-valued parameter
    inflate the threshold for the entire tensor and marks a live model
    degenerate — the same global-reduction mistake as issue #322's transversality
    floor, one module over.
    """

    ATOL = 1e-12

    def test_floors_are_per_column_not_reduced(self):
        f = asens.sens_resolution_floors([1.0, 1e-6], self.ATOL)
        assert f is not None and len(f) == 2
        assert f[1] > f[0], "a smaller |p| must give a LOOSER floor"

    def test_absent_inputs_report_not_assessed(self):
        """`None`, not a fabricated 0 — the field must not claim a verdict the
        run had no basis for."""
        assert asens.sens_resolution_floors(None, self.ATOL) is None
        assert asens.sens_resolution_floors([1.0], None) is None
        assert asens.resolvable_param_columns(np.zeros((2, 1, 1)), None, None) is None

    def test_an_all_tiny_tensor_has_no_resolvable_column(self):
        """`MODEL1907180003`: max|sx| = 4.7e-18 against a floor of 2.8e-10."""
        sx = np.full((5, 3, 2), 1e-18)
        assert asens.resolvable_param_columns(sx, [1.0, 1.0], self.ATOL) == 0

    def test_a_healthy_tensor_has_every_column_resolvable(self):
        sx = np.full((5, 3, 2), 1.0)
        assert asens.resolvable_param_columns(sx, [1.0, 1.0], self.ATOL) == 2

    def test_one_tiny_valued_parameter_does_not_condemn_the_whole_tensor(self):
        """The `BIOMD0000000002` regression. It has real dynamics and a genuine
        `max|sx|`, but one small-valued parameter raises that column's floor above
        the tensor peak. Under a global `max` floor the model read as entirely
        unresolvable; per column, the other parameters still count."""
        sx = np.zeros((4, 2, 2))
        sx[:, :, 0] = 1e-3  # ordinary column, |p| = 1  -> floor 1e-10
        sx[:, :, 1] = 1e-9  # column whose |p| is tiny  -> very loose floor
        n = asens.resolvable_param_columns(sx, [1.0, 1e-9], self.ATOL)
        assert n == 1, "the ordinary column must still be counted as resolvable"

    def test_a_column_exactly_at_its_floor_is_not_resolvable(self):
        """Boundary: `>` not `>=`, so a value indistinguishable from the floor is
        not claimed as signal."""
        floors = asens.sens_resolution_floors([1.0], self.ATOL)
        sx = np.full((3, 1, 1), float(floors[0]))
        assert asens.resolvable_param_columns(sx, [1.0], self.ATOL) == 0


class TestClassifyFailure:
    @staticmethod
    def _cls(bn, am, unsup=False):
        import amici_sens_run as run

        return run._classify_failure(bn, am, unsup)

    def test_a_declared_refusal_is_unsupported(self):
        st, msg = self._cls("bngsim: SensitivityUnsupportedError: ...", "", unsup=True)
        assert st == "unsupported"
        assert "SensitivityUnsupportedError" in msg

    def test_a_declared_refusal_wins_over_both_raised(self):
        """bngsim's declaration is a fact about bngsim and this model; it stays
        true whatever AMICI did. Both buckets are non-scoring, so preferring the
        named one costs no signal — and BAD_TEST ('nothing to compare') cannot be
        counted as bngsim's sensitivity gap, which is the number this exists to
        produce. The AMICI text is kept either way."""
        st, msg = self._cls("bngsim: SensitivityUnsupportedError: x", "amici: y", unsup=True)
        assert st == "unsupported"
        assert "amici: y" in msg and "SensitivityUnsupportedError" in msg

    def test_an_undeclared_bngsim_failure_is_still_an_exception(self):
        """The guard that keeps the bucket meaningful: UNSUPPORTED must not
        become a place real bugs go quiet."""
        assert self._cls("bngsim: SimulationError: boom", "")[0] == "exception"

    def test_the_pre_existing_attributions_are_unchanged(self):
        assert self._cls("", "amici: compile failed")[0] == "reference_failed"
        assert self._cls("bngsim: x", "amici: y")[0] == "bad_test"
        assert self._cls("bngsim: x", "")[0] == "exception"

    def test_the_flag_defaults_off(self):
        """``_classify_failure`` is called positionally in the worker; the default
        keeps any other caller on the old behavior."""
        import amici_sens_run as run

        assert run._classify_failure("bngsim: x", "")[0] == "exception"


class TestUnsupportedIsNonScoring:
    def test_the_status_maps_to_the_unsupported_outcome(self):
        import amici_sens_run as run
        from _core import Outcome

        assert run._OUTCOME["unsupported"] is Outcome.UNSUPPORTED

    def test_it_is_excluded_from_the_failing_tally(self):
        """The point of the whole change. ``main()``'s exit code is
        ``sum(counts[o] for o in FAILING)``, so a declared refusal must not move
        it — otherwise a model bngsim honestly declined still reads as a failure
        and the sweep cannot go green."""
        from _core import FAILING, Outcome, tally

        counts = tally([Outcome.PASS, Outcome.UNSUPPORTED, Outcome.UNSUPPORTED])
        assert counts["UNSUPPORTED"] == 2
        assert sum(counts.get(o.value, 0) for o in FAILING) == 0

    def test_it_has_a_progress_tag_and_shows_the_reason(self, capsys):
        """A status with no entry in the tag map prints its raw key, and one
        missing from the detail branch prints no reason at all. The run log is the
        only live view of a multi-hour sweep, so both matter."""
        import amici_sens_run as run

        run._make_progress(None)(
            1,
            2,
            {
                "status": "unsupported",
                "model_id": "BIOMD0000000342",
                "sens_method": "staggered",
                "exception": "bngsim: SensitivityUnsupportedError: events",
            },
        )
        line = capsys.readouterr().out
        assert "UNSUP" in line
        assert "BIOMD0000000342" in line
        assert "SensitivityUnsupportedError" in line


class TestMatrixRendersUnsupported:
    """The rendered page must show the bucket, or the run log is the only place
    it exists and the matrix over-reports EXCEPTION-free-ness without saying why.
    """

    @staticmethod
    def _render(tmp_path):
        import generate_amici_sens_matrix as gen

        report = tmp_path / "report_sens.json"
        report.write_text(
            json.dumps(
                {
                    "_meta": {
                        "tally": {"PASS": 3, "UNSUPPORTED": 2, "DIFF": 0},
                        "n_models": 3,
                        "sens_methods": ["staggered"],
                    },
                    "results": [
                        {
                            "model_id": "BIOMD0000000342",
                            "method": "sens/staggered",
                            "outcome": "UNSUPPORTED",
                            "exception": "bngsim: SensitivityUnsupportedError: events",
                            "extra": {"sens_method": "staggered"},
                        }
                    ],
                }
            )
        )
        out = tmp_path / "matrix.html"
        gen.generate_html(report, out)
        return out.read_text()

    def test_the_count_is_carded(self, tmp_path):
        html = self._render(tmp_path)
        assert "unsupported" in html

    def test_the_legend_explains_the_bucket(self, tmp_path):
        html = self._render(tmp_path)
        assert "UNSUPPORTED" in html
        assert "SensitivityUnsupportedError" in html

    def test_the_row_is_refused_grey_not_triage_yellow(self, tmp_path):
        """A declared refusal needs no triage; colouring it like an EXCEPTION
        would put it back in the reader's queue, which is the thing being fixed."""
        from generate_amici_matrix import classify_row

        assert classify_row("UNSUPPORTED") == ("status-refused", "REFUSED")
        assert "status-refused" in self._render(tmp_path)


# --------------------------------------------------------------------------- #
# Initial-time alignment
# --------------------------------------------------------------------------- #
class TestInitialTimeAlignment:
    """AMICI's t0 defaults to 0 no matter where the output grid starts.

    bngsim and RoadRunner both apply the initial state AT ``t_start`` (rr_parity's
    ``bn_ode`` documents this: the SED-ML ``initialTime`` is passed as ``t_start``
    so pre-``outputStartTime`` dynamics fire). Without an explicit ``set_t0``,
    AMICI instead integrates ``[0, t_start]`` first and reports an already-evolved
    state at the grid's first point — the engines silently solve different initial
    value problems.

    This was not hypothetical: it made BIOMD0000000569 DIFF with a state
    disagreement frozen at 1.6e-4 across four orders of magnitude of integration
    tolerance (the tell that it was structural, not numerical), and it reaches 21
    of the 1323 corpus models. Asserted here with a recording double, because the
    real failure needs a nonzero-``initialTime`` model from the gitignored corpus
    and a ~15 s AMICI compile.
    """

    T_START = 1e-5

    def _fake_built(self, recorder):
        ss = pytest.importorskip("amici.sim.sundials")

        class _RData:
            status = 0
            cpu_time = 1.0
            x = np.zeros((3, 1))
            sx = np.zeros((3, 1, 1))
            state_ids = ["A"]

        class _Solver:
            def set_relative_tolerance(self, v): ...
            def set_absolute_tolerance(self, v): ...
            def set_sensitivity_order(self, v): ...
            def set_sensitivity_method(self, v): ...
            def set_internal_sensitivity_method(self, v): ...
            def get_linear_solver(self):
                return int(getattr(ss, "LinearSolver_KLU", 2))

        class _Model:
            def get_free_parameter_ids(self):
                return ["k1"]

            def set_parameter_scale(self, v): ...
            def set_parameter_list(self, v): ...
            def create_solver(self):
                return _Solver()

            def set_t0(self, t):
                recorder.append(t)

            def set_timepoints(self, ts):
                recorder.append(("ts", float(ts[0])))

            def simulate(self, solver=None):
                return _RData()

        zero = dict.fromkeys(
            ("parse_sec", "interpret_sec", "jac_derive_sec", "codegen_sec", "compile_sec"), 0.0
        )
        return (_Model(), {**zero, "load_sec": 0.0}, True)

    def test_t0_is_pinned_to_t_start(self):
        rec = []
        asens.amici_sens(
            self._fake_built(rec), self.T_START, 1.0, 3, 1e-9, 1e-12, ["k1"], "staggered"
        )
        assert self.T_START in rec, f"set_t0({self.T_START}) was never called; got {rec}"

    def test_t0_is_set_before_the_timepoints(self):
        """Ordering matters to AMICI: t0 must be in place when the output grid is
        installed, or the grid is interpreted against the stale t0."""
        rec = []
        asens.amici_sens(
            self._fake_built(rec), self.T_START, 1.0, 3, 1e-9, 1e-12, ["k1"], "staggered"
        )
        i_t0 = rec.index(self.T_START)
        i_ts = next(i for i, e in enumerate(rec) if isinstance(e, tuple) and e[0] == "ts")
        assert i_t0 < i_ts

    def test_the_grid_still_starts_at_t_start(self):
        """The fix must not shift the reported grid — only where the IC is applied."""
        rec = []
        t, _x, _sx, _names, _timing = asens.amici_sens(
            self._fake_built(rec), self.T_START, 1.0, 3, 1e-9, 1e-12, ["k1"], "staggered"
        )
        assert t[0] == self.T_START
