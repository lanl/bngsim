"""Tests for bngsim.NfsimSession — the public session-based NFsim API."""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


pytestmark = pytest.mark.skipif(
    not _has_nfsim(),
    reason="bngsim compiled without NFsim support",
)


@pytest.fixture
def dummy_model(simple_decay_net: Path) -> bngsim.Model:
    return bngsim.Model.from_net(str(simple_decay_net))


# ─── Construction & lifecycle ────────────────────────────────────────────


class TestNfsimSessionLifecycle:
    def test_create_and_destroy(self, nfsim_xml):
        session = bngsim.NfsimSession(str(nfsim_xml))
        assert not session.initialized
        assert not session.destroyed
        session.destroy()
        assert session.destroyed

    def test_context_manager(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            assert not nf.destroyed
        assert nf.destroyed

    def test_double_destroy_is_safe(self, nfsim_xml):
        session = bngsim.NfsimSession(str(nfsim_xml))
        session.destroy()
        session.destroy()  # should not raise

    def test_repr(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            assert "created" in repr(nf)
            nf.initialize(seed=1)
            assert "initialized" in repr(nf)
        assert "destroyed" in repr(nf)

    def test_xml_path_property(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            assert nf.xml_path == str(nfsim_xml)


# ─── Simulation ──────────────────────────────────────────────────────────


class TestNfsimSessionSimulation:
    def test_basic_simulate(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            result = nf.simulate(0, 1, n_points=11)

        assert isinstance(result, bngsim.Result)
        assert result.n_times == 11
        assert result.n_observables == 6
        np.testing.assert_almost_equal(result.time[0], 0.0)
        np.testing.assert_almost_equal(result.time[-1], 1.0)

    def test_multi_segment(self, nfsim_xml):
        """Simulate two consecutive segments."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            r1 = nf.simulate(0, 0.5, n_points=6)
            r2 = nf.simulate(0.5, 1.0, n_points=6)

        assert r1.n_times == 6
        assert r2.n_times == 6
        np.testing.assert_almost_equal(r1.time[-1], 0.5)
        np.testing.assert_almost_equal(r2.time[0], 0.5)

    def test_deterministic_seeding(self, nfsim_xml):
        """Same seed produces identical trajectories."""
        results = []
        for _ in range(2):
            with bngsim.NfsimSession(str(nfsim_xml)) as nf:
                nf.initialize(seed=42)
                results.append(nf.simulate(0, 1, n_points=11))

        np.testing.assert_array_equal(
            np.asarray(results[0].observables),
            np.asarray(results[1].observables),
        )

    def test_different_seeds_differ(self, nfsim_xml):
        results = []
        for seed in [42, 99]:
            with bngsim.NfsimSession(str(nfsim_xml)) as nf:
                nf.initialize(seed=seed)
                results.append(nf.simulate(0, 1, n_points=11))

        assert not np.array_equal(
            np.asarray(results[0].observables),
            np.asarray(results[1].observables),
        )


# ─── Parameters ──────────────────────────────────────────────────────────


class TestNfsimSessionParams:
    def test_set_param_before_init(self, nfsim_xml):
        """Setting kon=0 before init → no XY formation."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.set_param("kon", 0.0)
            nf.initialize(seed=42)
            result = nf.simulate(0, 1, n_points=11)

        xy_idx = result.observable_names.index("XY")
        obs = np.asarray(result.observables)
        assert np.all(obs[:, xy_idx] == 0)

    def test_clear_param_overrides(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.set_param("kon", 0.0)
            nf.clear_param_overrides()
            nf.initialize(seed=42)
            result = nf.simulate(0, 1, n_points=11)

        # After clearing, kon is back to default → XY should form
        xy_idx = result.observable_names.index("XY")
        obs = np.asarray(result.observables)
        assert obs[-1, xy_idx] > 0


# ─── Issue #20: ExprTk-driven set_param propagation ─────────────────────


class TestSetParamPropagation:
    """set_param() must re-evaluate every <Parameter expr=...> dependent.

    Before the fix: NFsim stored only precomputed `value=` for every
    parameter. setParameter() updated one cell of paramMap and left
    every derived parameter pinned at its XML-time value, so a
    parameter scan over `kon_scale` left `kon = kon_base*kon_scale`
    unchanged. See internal#20.
    """

    def test_pre_init_propagates_chain(self, nfsim_param_prop_xml):
        """Pre-init set_param fires ExprTk on dependents."""
        with bngsim.NfsimSession(str(nfsim_param_prop_xml)) as nf:
            nf.set_param("kon_scale", 10.0)
            nf.initialize(seed=42)
            assert nf.get_parameter("kon_scale") == pytest.approx(10.0)
            # kon = kon_base * kon_scale = 10 * 10 = 100
            assert nf.get_parameter("kon") == pytest.approx(100.0)
            # use_fast = if(kon_scale>=threshold(2),1,0) → 1
            assert nf.get_parameter("use_fast") == pytest.approx(1.0)
            # kon_eff = kon*(1-use_fast)+kon*100*use_fast = 100*100 = 10000
            assert nf.get_parameter("kon_eff") == pytest.approx(10000.0)

    def test_post_init_set_param_updates_named_param(self, nfsim_param_prop_xml):
        """Post-init set_param actually lands; previously was silently dropped."""
        with bngsim.NfsimSession(str(nfsim_param_prop_xml)) as nf:
            nf.initialize(seed=42)
            assert nf.get_parameter("kon_scale") == pytest.approx(1.0)
            nf.set_param("kon_scale", 5.0)
            assert nf.get_parameter("kon_scale") == pytest.approx(5.0)

    def test_post_init_propagates_chain(self, nfsim_param_prop_xml):
        """Post-init set_param fires ExprTk on dependents on the live System."""
        with bngsim.NfsimSession(str(nfsim_param_prop_xml)) as nf:
            nf.initialize(seed=42)
            nf.set_param("kon_scale", 10.0)
            assert nf.get_parameter("kon") == pytest.approx(100.0)
            assert nf.get_parameter("use_fast") == pytest.approx(1.0)
            assert nf.get_parameter("kon_eff") == pytest.approx(10000.0)

    def test_clear_overrides_restores_xml_namespace(self, nfsim_param_prop_xml):
        """clear_param_overrides on a live session reverts the whole table."""
        with bngsim.NfsimSession(str(nfsim_param_prop_xml)) as nf:
            nf.initialize(seed=42)
            nf.set_param("kon_scale", 10.0)
            assert nf.get_parameter("kon") == pytest.approx(100.0)
            nf.clear_param_overrides()
            # All derived parameters should go back to their XML-time values.
            assert nf.get_parameter("kon_scale") == pytest.approx(1.0)
            assert nf.get_parameter("kon") == pytest.approx(10.0)
            assert nf.get_parameter("use_fast") == pytest.approx(0.0)

    def test_evaluate_with_overrides(self, nfsim_param_prop_xml):
        """evaluate(expr, overrides) layers extra bindings on simulator state."""
        with bngsim.NfsimSession(str(nfsim_param_prop_xml)) as nf:
            nf.initialize(seed=42)
            assert nf.evaluate("kon_base*kon_scale") == pytest.approx(10.0)
            # Layered override doesn't mutate session state.
            assert nf.evaluate(
                "kon_base*kon_scale", overrides={"kon_scale": 7.0}
            ) == pytest.approx(70.0)
            assert nf.get_parameter("kon_scale") == pytest.approx(1.0)

    def test_evaluate_handles_if_and_builtins(self, nfsim_param_prop_xml):
        """evaluate() understands BNG built-ins and if()."""
        with bngsim.NfsimSession(str(nfsim_param_prop_xml)) as nf:
            nf.initialize(seed=42)
            assert nf.evaluate("if(kon_scale>=threshold,1,0)") == pytest.approx(0.0)
            # _NA is a built-in (Avogadro). 1/_NA gives the per-molecule mole fraction.
            np.testing.assert_allclose(nf.evaluate("1/_NA"), 1.0 / 6.02214076e23, rtol=1e-12)

    def test_set_param_changes_simulation_propensity(self, nfsim_param_prop_xml):
        """End-to-end: kon=0 (via kon_scale=0) zeroes XY formation."""
        with bngsim.NfsimSession(str(nfsim_param_prop_xml)) as nf:
            nf.set_param("kon_scale", 0.0)
            nf.initialize(seed=42)
            result = nf.simulate(0, 1, n_points=11)
        xy_idx = result.observable_names.index("XY")
        obs = np.asarray(result.observables)
        # kon = kon_base*kon_scale = 0; rate-law for binding rule references
        # `kon` directly, so no XY may form.
        assert np.all(obs[:, xy_idx] == 0)


# ─── Issue #29: pre-init set_param re-evaluates seed-species concentrations ──


class TestSetParamSeedConcentration:
    """Pre-init set_param() must propagate to `<Species concentration="X">`.

    Before the fix: NFsim parsed `<Species concentration="X_init">` against
    the unmodified XML parameter map and baked agent counts in at parse
    time. set_param() ran AFTER prepareForSimulation, too late to alter
    the agent population — so models that used setParameter between
    `generate_network` and a `method=>"nf"` simulate started with the
    pre-override count. See internal#29.
    """

    def test_pre_init_set_param_changes_agent_count(self, nfsim_seed_concentration_xml):
        """Direct override of a seed-species parameter changes agent count."""
        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml)) as nf:
            nf.set_param("X_init", 1.0)
            nf.initialize(seed=42)
            assert nf.get_molecule_count("X") == 1

    def test_pre_init_set_param_propagates_through_derived_concentration(
        self, nfsim_seed_concentration_xml
    ):
        """Override of base param cascades to species using derived expr."""
        # Y_init=500 is unchanged; X_init drops to 17. Y(x) is independent.
        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml)) as nf:
            nf.set_param("X_init", 17.0)
            nf.initialize(seed=42)
            assert nf.get_molecule_count("X") == 17
            assert nf.get_molecule_count("Y") == 500

    def test_pre_init_no_overrides_uses_xml_concentration(self, nfsim_seed_concentration_xml):
        """Baseline: without overrides, agent counts come from XML values."""
        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml)) as nf:
            nf.initialize(seed=42)
            assert nf.get_molecule_count("X") == 5000
            assert nf.get_molecule_count("Y") == 500

    def test_pre_init_respects_molecule_limit_after_override(self, nfsim_seed_concentration_xml):
        """Issue #29's scaling_example pattern: override avoids tripping gml."""
        # With molecule_limit=10 and X_init=5000, init must fail.
        # With X_init=1, init must succeed under that same limit.
        with (
            bngsim.NfsimSession(str(nfsim_seed_concentration_xml), molecule_limit=10) as nf,
            pytest.raises(bngsim.SimulationError),
        ):
            nf.initialize(seed=42)

        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml), molecule_limit=10) as nf:
            nf.set_param("X_init", 1.0)
            nf.set_param("Y_init", 1.0)
            nf.initialize(seed=42)
            assert nf.get_molecule_count("X") == 1
            assert nf.get_molecule_count("Y") == 1


# ─── Molecule mutations ──────────────────────────────────────────────────


class TestNfsimSessionMolecules:
    def test_get_molecule_count(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            # simple_system.xml has 5000 X molecules initially
            count = nf.get_molecule_count("X")
            assert count == 5000

    def test_add_molecules(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            before = nf.get_molecule_count("Y")
            nf.add_molecules("Y", 100)
            after = nf.get_molecule_count("Y")
            assert after == before + 100

    def test_add_molecules_negative_raises(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            with pytest.raises(ValueError, match="positive"):
                nf.add_molecules("Y", -10)

    def test_add_molecules_zero_raises(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            with pytest.raises(ValueError, match="positive"):
                nf.add_molecules("Y", 0)


# ─── Exact species mutations ────────────────────────────────────────────────


class TestNfsimSessionSpecies:
    def test_get_species_count_separates_states(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            assert nf.get_species_count("X(p~0,y)") == 5000
            assert nf.get_species_count("X(p~1,y)") == 0

    def test_add_species_targets_exact_state(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.add_species("X(p~1,y)", 25)

            assert nf.get_species_count("X(p~0,y)") == 5000
            assert nf.get_species_count("X(p~1,y)") == 25
            assert nf.get_molecule_count("X") == 5025

    def test_remove_species_targets_exact_state(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.remove_species("X(p~0,y)", 10)

            assert nf.get_species_count("X(p~0,y)") == 4990
            assert nf.get_species_count("X(p~1,y)") == 0
            assert nf.get_molecule_count("X") == 4990

    def test_set_species_count_adds_and_removes(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.set_species_count("X(p~1,y)", 20)
            assert nf.get_species_count("X(p~1,y)") == 20

            nf.set_species_count("X(p~1,y)", 7)
            assert nf.get_species_count("X(p~1,y)") == 7

    def test_remove_species_too_many_raises(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            with pytest.raises(bngsim.SimulationError, match="insufficient matches"):
                nf.remove_species("X(p~1,y)", 1)

    def test_species_mutation_requires_exact_pattern(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            with pytest.raises(bngsim.SimulationError, match="every component listed"):
                nf.set_species_count("X(p~0)", 10)

    def test_species_mutation_rejects_complex_patterns(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            with pytest.raises(bngsim.SimulationError, match="single-molecule"):
                nf.set_species_count("X(p~0,y!1).Y(x!1)", 1)


# ─── Symmetric-site (BLBR-style) species API — issue #21 ───────────────────
#
# NFsim's XML loader renames duplicate `<ComponentType id="r">` entries to
# internal names `r1`/`r2`, while the BNGL surface keeps the bare name `r`.
# These tests cover the resolver's handling of symmetric stateful and
# stateless equivalency classes via the public species API.


class TestNfsimSessionSymmetricSites:
    def test_get_species_count_homogeneous_states(self, nfsim_sym_state_xml):
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            assert nf.get_species_count("L(r~u,r~u)") == 1000
            assert nf.get_species_count("L(r~c,r~c)") == 0

    def test_set_species_count_homogeneous_states(self, nfsim_sym_state_xml):
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            nf.set_species_count("L(r~u,r~u)", 200)
            assert nf.get_species_count("L(r~u,r~u)") == 200
            assert nf.get_molecule_count("L") == 200

    def test_add_species_homogeneous_states(self, nfsim_sym_state_xml):
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            nf.add_species("L(r~c,r~c)", 75)
            assert nf.get_species_count("L(r~c,r~c)") == 75
            assert nf.get_molecule_count("L") == 1075

    def test_remove_species_homogeneous_states(self, nfsim_sym_state_xml):
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            nf.remove_species("L(r~u,r~u)", 50)
            assert nf.get_species_count("L(r~u,r~u)") == 950
            assert nf.get_molecule_count("L") == 950

    def test_heterogeneous_states_match_either_ordering(self, nfsim_sym_state_xml):
        # NFsim treats both physical orderings of a symmetric-site species as
        # the same species. The resolver must compare the sorted multiset of
        # states across class members so that a single get_species_count
        # query returns the total without double-counting.
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            nf.add_species("L(r~u,r~c)", 30)
            assert nf.get_species_count("L(r~u,r~c)") == 30
            assert nf.get_species_count("L(r~c,r~u)") == 30

    def test_explicit_index_disambiguation_still_works(self, nfsim_sym_state_xml):
        # NFsim exposes the renamed `r1`/`r2` slot names; resolver should
        # accept those directly as well, in case the user picks the explicit
        # form for clarity.
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            assert nf.get_species_count("L(r1~u,r2~u)") == 1000

    def test_stateful_class_requires_explicit_state(self, nfsim_sym_state_xml):
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            with pytest.raises(bngsim.SimulationError, match="explicit states"):
                nf.set_species_count("L(r,r)", 100)

    def test_overspecified_class_rejected(self, nfsim_sym_state_xml):
        # Three `r` entries on a 2-site class is over-specification.
        with bngsim.NfsimSession(str(nfsim_sym_state_xml)) as nf:
            nf.initialize(seed=1)
            with pytest.raises(bngsim.SimulationError, match="duplicate component"):
                nf.set_species_count("L(r~u,r~u,r~u)", 100)

    def test_stateless_symmetric_sites(self, nfsim_sym_stateless_xml):
        with bngsim.NfsimSession(str(nfsim_sym_stateless_xml)) as nf:
            nf.initialize(seed=1)
            assert nf.get_species_count("L(r,r)") == 100
            nf.set_species_count("L(r,r)", 25)
            assert nf.get_species_count("L(r,r)") == 25
            assert nf.get_molecule_count("L") == 25


# ─── Observables ─────────────────────────────────────────────────────────


class TestNfsimSessionObservables:
    def test_get_observable_names(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            names = nf.get_observable_names()
            assert "X_free" in names
            assert "XY" in names
            assert len(names) == 6

    def test_get_observable_values(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            vals = nf.get_observable_values()
            assert len(vals) == 6
            # At t=0, Xtotal should be 5000
            names = nf.get_observable_names()
            xtotal_idx = names.index("Xtotal")
            assert vals[xtotal_idx] == 5000


# ─── -bscb / -utl ────────────────────────────────────────────────────────


class TestNfsimSessionBscbUtl:
    def test_default_bscb_on_runs(self, nfsim_xml):
        """bngsim defaults ``block_same_complex_binding=True``; session runs."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            r = nf.simulate(0, 1, n_points=11)
        assert r.n_times == 11

    def test_bscb_explicit_false_runs(self, nfsim_xml):
        """Opt-out via ``block_same_complex_binding=False`` is accepted."""
        with bngsim.NfsimSession(str(nfsim_xml), block_same_complex_binding=False) as nf:
            nf.initialize(seed=42)
            r = nf.simulate(0, 1, n_points=11)
        assert r.n_times == 11

    def test_traversal_limit_runs(self, nfsim_xml):
        """Explicit ``-utl N`` value is accepted by the session."""
        with bngsim.NfsimSession(str(nfsim_xml), traversal_limit=5) as nf:
            nf.initialize(seed=42)
            r = nf.simulate(0, 1, n_points=11)
        assert r.n_times == 11

    def test_bscb_and_utl_combined(self, nfsim_xml):
        """``-bscb`` together with ``-utl N`` matches BNGL ``param=>"-bscb -utl 5"``."""
        with bngsim.NfsimSession(
            str(nfsim_xml),
            block_same_complex_binding=True,
            traversal_limit=5,
        ) as nf:
            nf.initialize(seed=42)
            r = nf.simulate(0, 1, n_points=11)
        assert r.n_times == 11


# ─── molecule_limit ──────────────────────────────────────────────────────


class TestNfsimSessionMoleculeLimit:
    def test_molecule_limit_too_low_fails(self, nfsim_xml):
        # gml=10 is too low for X=5000; NFsim fails at initialize
        with (
            bngsim.NfsimSession(str(nfsim_xml), molecule_limit=10) as nf,
            pytest.raises(bngsim.SimulationError),
        ):
            nf.initialize(seed=42)

    def test_molecule_limit_high_succeeds(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml), molecule_limit=10000) as nf:
            nf.initialize(seed=42)
            result = nf.simulate(0, 1, n_points=2)
            assert result.n_times == 2


# ─── State guards ────────────────────────────────────────────────────────


class TestNfsimSessionStateGuards:
    def test_simulate_before_init_raises(self, nfsim_xml):
        with (
            bngsim.NfsimSession(str(nfsim_xml)) as nf,
            pytest.raises(bngsim.SimulationError, match="not initialized"),
        ):
            nf.simulate(0, 1, n_points=11)

    def test_simulate_after_destroy_raises(self, nfsim_xml):
        nf = bngsim.NfsimSession(str(nfsim_xml))
        nf.initialize(seed=42)
        nf.destroy()
        with pytest.raises(bngsim.SimulationError, match="destroyed"):
            nf.simulate(0, 1, n_points=11)

    def test_set_param_after_destroy_raises(self, nfsim_xml):
        nf = bngsim.NfsimSession(str(nfsim_xml))
        nf.destroy()
        with pytest.raises(bngsim.SimulationError, match="destroyed"):
            nf.set_param("kon", 0.0)

    def test_get_molecule_count_before_init_raises(self, nfsim_xml):
        with (
            bngsim.NfsimSession(str(nfsim_xml)) as nf,
            pytest.raises(bngsim.SimulationError, match="not initialized"),
        ):
            nf.get_molecule_count("X")


# ─── Empty / degenerate models (issue #40 B-2) ───────────────────────────


class TestNfsimSessionEmptyModel:
    """A BNGL file with a ``simulate`` action but no observables / molecule
    types / reactions must run to completion without exception and report
    an empty Result. This matches BNG2.pl subprocess behavior (writes a
    ``.gdat`` with only the ``# time`` header) and lets batch harnesses
    classify "ran with no data" via ``Result.has_simulation_data`` rather
    than catching an exception.
    """

    def test_empty_model_runs_to_completion(self, nfsim_empty_model_xml):
        with bngsim.NfsimSession(str(nfsim_empty_model_xml)) as nf:
            nf.initialize(seed=42)
            r = nf.simulate(0.0, 120.0, n_points=121)

        assert r.n_times == 121
        assert r.n_observables == 0
        assert r.n_expressions == 0
        assert r.n_species == 0
        assert list(r.observable_names) == []

    def test_empty_model_has_simulation_data_is_false(self, nfsim_empty_model_xml):
        with bngsim.NfsimSession(str(nfsim_empty_model_xml)) as nf:
            nf.initialize(seed=42)
            r = nf.simulate(0.0, 120.0, n_points=121)
        assert r.has_simulation_data is False

    def test_non_empty_model_has_simulation_data_is_true(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            r = nf.simulate(0, 1, n_points=11)
        assert r.has_simulation_data is True


# ─── State artifact: save_species (issue #40 B-3b) ───────────────────────


class TestNfsimSessionSaveSpecies:
    """``save_species(path)`` writes a BNG-format ``.species`` file matching
    BNG2.pl's ``simulate({method=>"nf", get_final_state=>1})`` artifact.
    PyBioNetGen reads this to thread NF state across segments while the
    in-process snapshot path remains blocked on upstream NFsim
    (internal#52).
    """

    def test_save_species_writes_bng_format_file(self, nfsim_xml, tmp_path):
        out = tmp_path / "final.species"
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1.0, n_points=11)
            nf.save_species(out)

        assert out.exists()
        lines = out.read_text().splitlines()
        # Two header comment lines plus at least one species record
        assert lines[0].startswith("# nfsim generated species list")
        assert len(lines) >= 3
        species_lines = [line for line in lines if line and not line.startswith("#")]
        assert species_lines, "expected at least one species record"
        # Each non-comment line is `<pattern>  <count>` with an integer count
        for line in species_lines:
            parts = line.rsplit(None, 1)
            assert len(parts) == 2, f"unexpected line shape: {line!r}"
            assert int(parts[1]) > 0

    def test_save_species_preserves_bonded_complexes(self, nfsim_xml, tmp_path):
        """simple_system creates bound X.Y complexes; the saved file must
        encode the bond using BNGL ``!N`` notation."""
        out = tmp_path / "complexes.species"
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1.0, n_points=11)
            nf.save_species(out)

        text = out.read_text()
        # Bonded multi-molecule complex appears as `<molA(...!1)>.<molB(...!1)>  <count>`
        assert ".Y(" in text or ").X(" in text, (
            f"expected at least one bonded complex line, got:\n{text}"
        )
        assert "!" in text, "bond notation missing"

    def test_save_species_requires_initialized_session(self, nfsim_xml, tmp_path):
        nf = bngsim.NfsimSession(str(nfsim_xml))
        with pytest.raises(bngsim.SimulationError, match="not initialized"):
            nf.save_species(tmp_path / "x.species")
        nf.destroy()


# ─── BNG2.pl simulate_nf time-axis convention (issue #40 t_start opt-in) ──


class TestNfsimSessionRelativeTime:
    """``simulate(..., relative_time=True)`` rebases the returned time axis
    to start at 0, matching BNG2.pl's ``simulate({method=>"nf", ...})``
    convention ("NFsim timepoints are reported as time elapsed since
    t_start=$t_start"). The internal NFsim clock is unaffected — a host
    can opt in segment-by-segment without losing multi-segment threading.
    """

    def test_default_remains_absolute(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            result = nf.simulate(5.0, 7.0, n_points=11)
        np.testing.assert_almost_equal(result.time[0], 5.0)
        np.testing.assert_almost_equal(result.time[-1], 7.0)

    def test_relative_time_rebases_to_zero(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            result = nf.simulate(5.0, 7.0, n_points=11, relative_time=True)
        np.testing.assert_almost_equal(result.time[0], 0.0)
        np.testing.assert_almost_equal(result.time[-1], 2.0)
        # Uniform spacing preserved
        np.testing.assert_array_almost_equal(np.diff(result.time), np.full(10, 0.2))

    def test_relative_only_changes_time_axis(self, nfsim_xml):
        """Same seed + same segment: only the time labels differ. Observable
        trajectories must be byte-identical so the flag is purely cosmetic
        relative to the underlying simulation."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            r_abs = nf.simulate(5.0, 7.0, n_points=11)
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            r_rel = nf.simulate(5.0, 7.0, n_points=11, relative_time=True)

        np.testing.assert_array_equal(
            np.asarray(r_abs.observables),
            np.asarray(r_rel.observables),
        )
        # And the time axes differ by the constant t_start offset
        np.testing.assert_array_almost_equal(
            np.asarray(r_abs.time) - np.asarray(r_rel.time),
            np.full(11, 5.0),
        )

    def test_relative_time_preserves_internal_clock(self, nfsim_xml):
        """A second segment after a relative_time=True simulate must still
        advance from the absolute end of the first segment — the flag only
        relabels the Result, not the live NFsim System."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            r1 = nf.simulate(0.0, 0.5, n_points=6, relative_time=True)
            r2 = nf.simulate(0.5, 1.0, n_points=6)

        # r1 was rebased; r2 was not.
        np.testing.assert_almost_equal(r1.time[0], 0.0)
        np.testing.assert_almost_equal(r1.time[-1], 0.5)
        np.testing.assert_almost_equal(r2.time[0], 0.5)
        np.testing.assert_almost_equal(r2.time[-1], 1.0)


# ─── save/restore concentrations (issue #52) ─────────────────────────────


class TestNfsimSessionSaveRestore:
    """In-process save_concentrations()/restore_concentrations().

    Regression coverage for internal#52: resetConcentrations() used to
    SIGSEGV inside SystemSnapshot::restore() because destroyAllMolecules()
    freed objects still owned by the MoleculeList pool (use-after-free on the
    next genDefaultMolecule(), double free at ~MoleculeList) and deleted the
    Complex objects the recycled pool molecules still referenced.
    """

    def _obs(self, nf) -> dict[str, float]:
        return dict(zip(nf.get_observable_names(), nf.get_observable_values(), strict=False))

    @pytest.mark.parametrize("bscb", [False, True])
    def test_save_mutate_restore_roundtrip(self, nfsim_xml, bscb):
        """save → simulate (mutate) → restore returns to the saved state, and
        the session keeps simulating afterwards. Runs with complex bookkeeping
        both off and on, since the two failure modes (pool UAF vs. stale
        complex IDs) are gated on bscb."""
        with bngsim.NfsimSession(str(nfsim_xml), block_same_complex_binding=bscb) as nf:
            nf.initialize(seed=42)
            assert nf.has_saved_concentrations() is False

            # Equilibrate so bonds (XY) and component states (Xp) are populated:
            # this is the bonded-complex state the snapshot must round-trip.
            nf.simulate(0, 2, n_points=2)
            saved = self._obs(nf)
            assert saved["XY"] > 0  # bonds actually formed
            assert saved["X_p_total"] > 0  # component states actually changed

            nf.save_concentrations()
            assert nf.has_saved_concentrations() is True

            # Mutate further, then rewind.
            nf.simulate(2, 8, n_points=2)
            nf.restore_concentrations()
            assert self._obs(nf) == saved

            # Session is still usable after the restore.
            nf.simulate(8, 10, n_points=2)

    def test_restore_without_save_raises(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=7)
            with pytest.raises(bngsim.SimulationError):
                nf.restore_concentrations()

    def test_save_requires_initialization(self, nfsim_xml):
        nf = bngsim.NfsimSession(str(nfsim_xml))
        with pytest.raises(bngsim.SimulationError):
            nf.save_concentrations()
        nf.destroy()

    def test_has_saved_is_false_before_init(self, nfsim_xml):
        nf = bngsim.NfsimSession(str(nfsim_xml))
        assert nf.has_saved_concentrations() is False
        nf.destroy()

    def test_save_overwrites_previous_snapshot(self, nfsim_xml):
        """A second save_concentrations() rebases what restore() returns to."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1, n_points=2)
            nf.save_concentrations()
            nf.simulate(1, 5, n_points=2)
            second = self._obs(nf)
            nf.save_concentrations()  # overwrite with the later state
            nf.simulate(5, 9, n_points=2)
            nf.restore_concentrations()
            assert self._obs(nf) == second

    def test_initialize_clears_snapshot(self, nfsim_xml):
        """Re-initializing the session drops any prior snapshot."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1, n_points=2)
            nf.save_concentrations()
            assert nf.has_saved_concentrations() is True
            nf.initialize(seed=42)  # fresh session state
            assert nf.has_saved_concentrations() is False
            with pytest.raises(bngsim.SimulationError):
                nf.restore_concentrations()


class TestNfsimSessionNamedSaveRestore:
    """Labeled save/restore, mirroring Model.save_concentrations(label=...).

    The NFsim backend holds true multi-slot named concentration states in C++
    (a per-label SystemSnapshot map), for full parity with the network-based
    Model (issue #11). Distinct labels coexist; named and default slots are
    independent.
    """

    def _obs(self, nf) -> dict[str, float]:
        return dict(zip(nf.get_observable_names(), nf.get_observable_values(), strict=False))

    def test_labeled_save_and_restore_roundtrip(self, nfsim_xml):
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 2, n_points=2)
            saved = self._obs(nf)
            nf.save_concentrations("start_competition")
            assert nf.has_saved_concentrations("start_competition") is True
            assert nf.has_saved_concentrations("other") is False
            assert nf.has_saved_concentrations() is True  # any snapshot held
            assert nf.saved_concentration_labels == ["start_competition"]

            nf.simulate(2, 8, n_points=2)
            nf.restore_concentrations("start_competition")
            assert self._obs(nf) == saved

    def test_restore_unsaved_label_raises(self, nfsim_xml):
        """Restoring a label that was never saved fails loudly; the labels that
        *were* saved remain restorable."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1, n_points=2)
            nf.save_concentrations("phase_a")
            with pytest.raises(bngsim.SimulationError, match="phase_b"):
                nf.restore_concentrations("phase_b")
            # The saved label still works after the failed restore.
            nf.restore_concentrations("phase_a")
            # The default (unlabeled) slot was never saved, so restoring it raises.
            with pytest.raises(bngsim.SimulationError):
                nf.restore_concentrations()

    def test_named_and_default_slots_independent(self, nfsim_xml):
        """A named snapshot and the default (unlabeled) slot are separate: an
        unlabeled restore rewinds the default slot, not a named one."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 2, n_points=2)
            default_state = self._obs(nf)
            nf.save_concentrations()  # default/unlabeled slot

            nf.simulate(2, 6, n_points=2)
            named_state = self._obs(nf)
            nf.save_concentrations("checkpoint")  # separate named slot

            nf.simulate(6, 10, n_points=2)
            nf.restore_concentrations()  # default slot, unaffected by the named save
            assert self._obs(nf) == default_state

            nf.restore_concentrations("checkpoint")
            assert self._obs(nf) == named_state

    def test_named_slots_coexist(self, nfsim_xml):
        """Multiple named slots coexist: a later labeled save does NOT overwrite
        an earlier one, and both are restorable."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1, n_points=2)
            first = self._obs(nf)
            nf.save_concentrations("first")

            nf.simulate(1, 5, n_points=2)
            second = self._obs(nf)
            nf.save_concentrations("second")

            assert nf.has_saved_concentrations("first") is True
            assert nf.has_saved_concentrations("second") is True
            assert nf.saved_concentration_labels == ["first", "second"]

            # Restore the earlier slot, then the later one — both intact.
            nf.simulate(5, 9, n_points=2)
            nf.restore_concentrations("first")
            assert self._obs(nf) == first
            nf.restore_concentrations("second")
            assert self._obs(nf) == second

    def test_resave_same_label_overwrites_only_that_slot(self, nfsim_xml):
        """Saving to an existing label replaces just that slot, leaving other
        named slots untouched."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1, n_points=2)
            keep = self._obs(nf)
            nf.save_concentrations("keep")
            nf.save_concentrations("rewrite")

            nf.simulate(1, 5, n_points=2)
            rewritten = self._obs(nf)
            nf.save_concentrations("rewrite")  # overwrite only "rewrite"

            nf.simulate(5, 9, n_points=2)
            nf.restore_concentrations("rewrite")
            assert self._obs(nf) == rewritten
            nf.restore_concentrations("keep")
            assert self._obs(nf) == keep

    def test_saved_labels_cleared_on_reinitialize(self, nfsim_xml):
        """Re-initializing the session drops every named slot."""
        with bngsim.NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=42)
            nf.simulate(0, 1, n_points=2)
            nf.save_concentrations("a")
            nf.save_concentrations("b")
            assert nf.saved_concentration_labels == ["a", "b"]
            nf.initialize(seed=42)
            assert nf.saved_concentration_labels == []
            assert nf.has_saved_concentrations("a") is False
            with pytest.raises(bngsim.SimulationError):
                nf.restore_concentrations("a")
