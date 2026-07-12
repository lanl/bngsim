"""Native ``parameter_scan`` / ``bifurcate`` primitives (issue #11).

The scan primitive's ``reset_conc`` semantics must match BNG2.pl: each point
resets to the state *at scan invocation* (or to a named snapshot), applying only
the scanned parameter and any per-point ``on_point`` overrides — NOT re-deriving
every species from the ``.net`` seed initializers. The distinction matters for a
pre-equilibrate → intervene → scan protocol, where re-syncing to seed silently
discards the carried post-intervention state.

Oracle: ``simple_decay.net`` is A --k1--> B with A(t) = A0·exp(-k1·t), so the
per-point trajectory has a closed form for any starting A0 and any scanned k1.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


def _A(result) -> np.ndarray:
    """The A() species column of a Result."""
    return result.species[:, result.species_names.index("A()")]


class TestResetConcSemantics:
    def test_reset_to_snapshot_not_seed(self, simple_decay_net: Path):
        """The core issue: each point starts from the pre-equilibrated snapshot,
        not the .net seed (A=100)."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        # Pre-equilibrate 10 s at k1=0.1: A: 100 -> 100·e^-1 ≈ 36.79.
        sim.run_until(10.0)
        a_snap = m.get_concentration("A()")
        assert a_snap == pytest.approx(100.0 * np.exp(-1.0), rel=1e-4)
        sim.save_concentrations("post_incub")

        vals = [0.1, 0.2, 0.4]
        results = sim.parameter_scan(
            "k1", vals, t_span=(0, 10), n_points=11, reset_conc=True, reset_to="post_incub"
        )
        assert len(results) == 3
        for r, k in zip(results, vals, strict=True):
            a = _A(r)
            assert a[0] == pytest.approx(a_snap, rel=1e-9)  # from snapshot, not 100
            assert a[-1] == pytest.approx(a_snap * np.exp(-k * 10.0), rel=1e-4)

    def test_contrast_with_seed_resync(self, simple_decay_net: Path):
        """A hand-rolled seed re-sync (what run_batch does: clone+reset) starts
        every point from the seed; parameter_scan reset-to-snapshot does not.
        This is exactly the defect the issue reports."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        sim.run_until(10.0)
        a_snap = m.get_concentration("A()")
        sim.save_concentrations("s")

        scan = sim.parameter_scan("k1", [0.1], t_span=(0, 10), n_points=11, reset_to="s")
        batch = sim.run_batch(
            t_span=(0, 10), n_points=11, params=[{"k1": 0.1}]
        )  # clones + reset() -> seed
        assert _A(scan[0])[0] == pytest.approx(a_snap, rel=1e-9)  # snapshot ≈ 36.79
        assert _A(batch[0])[0] == pytest.approx(100.0)  # seed
        assert a_snap < 50.0  # the two are genuinely different

    def test_reset_to_none_captures_live_state(self, simple_decay_net: Path):
        """With no named target, the reset target is the live state at the call."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        sim.run_until(5.0)
        a_live = m.get_concentration("A()")
        results = sim.parameter_scan("k1", [0.1, 0.5], t_span=(0, 10), n_points=6)
        for r in results:
            assert _A(r)[0] == pytest.approx(a_live, rel=1e-9)

    def test_on_point_override_tracks_scanned_value(self, simple_decay_net: Path):
        """on_point runs after reset + set_param, so a coupled concentration can
        track the scanned parameter (the IGF1 cold-ligand pattern)."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")

        def couple(model, value):
            # Set A to a value derived from the scanned k1 (stand-in for a
            # setConcentration expression like value*NA*Vecf).
            model.set_concentration("A()", 200.0 * value)

        vals = [0.1, 0.3]
        results = sim.parameter_scan("k1", vals, t_span=(0, 10), n_points=11, on_point=couple)
        for r, k in zip(results, vals, strict=True):
            a = _A(r)
            assert a[0] == pytest.approx(200.0 * k, rel=1e-9)
            assert a[-1] == pytest.approx(200.0 * k * np.exp(-k * 10.0), rel=1e-4)


class TestBifurcate:
    def test_continuation_compounds_state(self, simple_decay_net: Path):
        """reset_conc=0: each point continues from the previous end-state."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        # Three identical 10 s legs at k1=0.1 compound: A·e^-1, A·e^-2, A·e^-3.
        res = sim.bifurcate("k1", [0.1, 0.1, 0.1], t_span=(0, 10), n_points=3)
        ends = [_A(r)[-1] for r in res]
        assert ends[0] == pytest.approx(100 * np.exp(-1), rel=1e-4)
        assert ends[1] == pytest.approx(100 * np.exp(-2), rel=1e-4)
        assert ends[2] == pytest.approx(100 * np.exp(-3), rel=1e-4)

    def test_bifurcate_equals_parameter_scan_reset_false(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        a = sim.bifurcate("k1", [0.2, 0.2], t_span=(0, 5), n_points=3)
        b = sim.parameter_scan("k1", [0.2, 0.2], t_span=(0, 5), n_points=3, reset_conc=False)
        for ra, rb in zip(a, b, strict=True):
            np.testing.assert_allclose(_A(ra), _A(rb), rtol=1e-9)


class TestScanBookkeeping:
    def test_model_left_pristine(self, simple_decay_net: Path):
        """After a scan the persistent model is restored: parameter back to its
        pre-scan value, concentrations back to the invocation state."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        sim.run_until(5.0)
        a_before = m.get_concentration("A()")
        k_before = m.get_param("k1")

        sim.parameter_scan("k1", [1.0, 2.0, 3.0], t_span=(0, 4), n_points=5)
        assert m.get_param("k1") == pytest.approx(k_before)
        assert m.get_concentration("A()") == pytest.approx(a_before, rel=1e-9)

        # Repeatable: a second scan yields identical results to the first.
        r1 = sim.parameter_scan("k1", [0.5], t_span=(0, 4), n_points=5)
        r2 = sim.parameter_scan("k1", [0.5], t_span=(0, 4), n_points=5)
        np.testing.assert_allclose(_A(r1[0]), _A(r2[0]), rtol=1e-12)

    def test_results_carry_scan_metadata(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        vals = [0.1, 0.2, 0.5]
        results = sim.parameter_scan("k1", vals, t_span=(0, 2), n_points=3)
        for r, v in zip(results, vals, strict=True):
            assert r.custom_attrs["scan_parameter"] == "k1"
            assert r.custom_attrs["scan_value"] == pytest.approx(v)

    def test_squeeze_returns_stacked_result(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        squeezed = sim.parameter_scan(
            "k1", [0.1, 0.2, 0.4], t_span=(0, 5), n_points=6, squeeze=True
        )
        # (n_sims, n_times, n_species)
        assert squeezed.species.shape[0] == 3
        assert squeezed.species.shape[1] == 6


class TestScanValueGeneration:
    def test_linear_range(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        results = sim.parameter_scan(
            "k1", par_min=0.1, par_max=0.5, n_scan_pts=5, t_span=(0, 1), n_points=2
        )
        got = [r.custom_attrs["scan_value"] for r in results]
        np.testing.assert_allclose(got, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_log_range(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        results = sim.parameter_scan(
            "k1",
            par_min=0.01,
            par_max=10.0,
            n_scan_pts=4,
            log_scale=True,
            t_span=(0, 1),
            n_points=2,
        )
        got = [r.custom_attrs["scan_value"] for r in results]
        np.testing.assert_allclose(got, [0.01, 0.1, 1.0, 10.0], rtol=1e-9)

    def test_single_point_range(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        results = sim.parameter_scan(
            "k1", par_min=0.3, par_max=0.9, n_scan_pts=1, t_span=(0, 1), n_points=2
        )
        assert [r.custom_attrs["scan_value"] for r in results] == [0.3]


class TestScanErrors:
    def test_unknown_parameter(self, simple_decay_net: Path):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ode")
        with pytest.raises(bngsim.ParameterError):
            sim.parameter_scan("not_a_param", [0.1], t_span=(0, 1), n_points=2)

    def test_unknown_reset_to_label(self, simple_decay_net: Path):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ode")
        with pytest.raises(ValueError, match="no saved concentration state"):
            sim.parameter_scan("k1", [0.1], t_span=(0, 1), n_points=2, reset_to="nope")

    def test_empty_par_scan_vals(self, simple_decay_net: Path):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ode")
        with pytest.raises(ValueError, match="non-empty"):
            sim.parameter_scan("k1", [], t_span=(0, 1), n_points=2)

    def test_missing_range_args(self, simple_decay_net: Path):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ode")
        with pytest.raises(ValueError, match="par_scan_vals"):
            sim.parameter_scan("k1", par_min=0.1, t_span=(0, 1), n_points=2)

    def test_log_scale_nonpositive(self, simple_decay_net: Path):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ode")
        with pytest.raises(ValueError, match="positive"):
            sim.parameter_scan(
                "k1",
                par_min=0.0,
                par_max=1.0,
                n_scan_pts=3,
                log_scale=True,
                t_span=(0, 1),
                n_points=2,
            )

    def test_bad_n_scan_pts(self, simple_decay_net: Path):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ode")
        with pytest.raises(ValueError, match="n_scan_pts"):
            sim.parameter_scan(
                "k1", par_min=0.1, par_max=1.0, n_scan_pts=0, t_span=(0, 1), n_points=2
            )

    def test_sensitivity_simulator_refused(self, simple_decay_net: Path):
        """A scan resets each point off the seed, so per-point forward
        sensitivities would be mis-seeded — refuse rather than mislead."""
        sim = bngsim.Simulator(
            bngsim.Model.from_net(str(simple_decay_net)),
            method="ode",
            sensitivity_params=["k1"],
        )
        with pytest.raises(ValueError, match="do not support output sensitivities"):
            sim.parameter_scan("k1", [0.1, 0.2], t_span=(0, 1), n_points=2)


class TestScanStochastic:
    def test_ssa_scan_seeded_per_point(self, reversible_net: Path):
        """SSA scan gives one Result per value, each with its own seed."""
        m = bngsim.Model.from_net(str(reversible_net))
        sim = bngsim.Simulator(m, method="ssa")
        results = sim.parameter_scan("kf", [0.001, 0.002], t_span=(0, 10), n_points=11, seed=123)
        assert len(results) == 2
        assert results[0].seed != results[1].seed  # base_seed + i

    def test_ssa_scan_reproducible(self, reversible_net: Path):
        m = bngsim.Model.from_net(str(reversible_net))
        sim = bngsim.Simulator(m, method="ssa")
        r1 = sim.parameter_scan("kf", [0.001, 0.002], t_span=(0, 10), n_points=11, seed=7)
        r2 = sim.parameter_scan("kf", [0.001, 0.002], t_span=(0, 10), n_points=11, seed=7)
        for a, b in zip(r1, r2, strict=True):
            np.testing.assert_array_equal(a.species, b.species)
