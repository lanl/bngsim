"""Tests for RuleMonkey integration via bngsim."""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_rulemonkey(),
    reason="RuleMonkey not compiled in",
)


@pytest.fixture
def dummy_model(simple_decay_net: Path) -> bngsim.Model:
    return bngsim.Model.from_net(str(simple_decay_net))


class TestRuleMonkeySimulator:
    def test_basic_run(self, dummy_model, nfsim_xml):
        sim = bngsim.Simulator(dummy_model, method="rm", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 1), n_points=11, seed=42)

        assert result.n_times == 11
        assert result.n_observables == 6
        assert result.observable_names == [
            "X_free",
            "X_p_total",
            "Xp_free",
            "XY",
            "Ytotal",
            "Xtotal",
        ]
        np.testing.assert_allclose(result.time[[0, -1]], [0.0, 1.0])

    @pytest.mark.parametrize("method", ["nf_exact", "rulemonkey", "rm"])
    def test_aliases_dispatch(self, dummy_model, nfsim_xml, method):
        sim = bngsim.Simulator(dummy_model, method=method, xml_path=str(nfsim_xml))
        assert sim.method == "rulemonkey"
        assert sim.requested_method == method


class TestRuleMonkeySession:
    def test_basic_session(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            result = rm.simulate(0, 1, n_points=11)

        assert isinstance(result, bngsim.Result)
        assert result.n_times == 11
        assert result.n_observables == 6

    def test_multi_segment_preserves_user_labels(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            first = rm.simulate(0, 1, n_points=2)
            second = rm.simulate(0, 1, n_points=2)

        np.testing.assert_allclose(first.time, [0.0, 1.0])
        np.testing.assert_allclose(second.time, [0.0, 1.0])

    def test_set_param_before_init(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.set_param("kon", 0.0)
            rm.initialize(seed=42)
            result = rm.simulate(0, 1, n_points=11)

        xy_idx = result.observable_names.index("XY")
        assert np.all(np.asarray(result.observables)[:, xy_idx] == 0)


class TestRuleMonkeySessionDerivedSeedAmounts:
    """Issue #44 Bug 2: pre-init set_param must re-derive derived seed amounts.

    A seed species whose amount is a *derived* expression (``Ntot = 100*scale``)
    must respond to a pre-init ``set_param('scale', ...)`` the same way the NFsim
    session does — otherwise the two engines silently run different initial
    conditions.

    bngsim used to close this gap itself, by re-evaluating the XML's
    ``<Parameter expr=>`` graph and rebuilding the engine from a re-baked copy
    of the XML on every pre-init ``set_param``. Since RuleMonkey v3.7.0 the
    override cascade re-derives from ``expr=`` upstream, so the adapter just
    forwards and lets the engine propagate (GH #115); these assertions are
    unchanged across that swap, which is the point of keeping them.
    """

    def test_pre_init_rederives_derived_seed_amount(self, nfsim_switchable_rate_xml):
        with bngsim.RuleMonkeySession(str(nfsim_switchable_rate_xml)) as rm:
            rm.set_param("scale", 0.5)  # Ntot = 100*scale -> 50
            rm.initialize(seed=1)
            assert rm.get_parameter("Ntot") == pytest.approx(50.0)
            assert rm.get_species_count("A(b)") == 50
            assert rm.get_species_count("B(b)") == 50

    def test_matches_nfsim_initial_condition(self, nfsim_switchable_rate_xml):
        """Both engines must start from identical seed-species counts."""
        counts = {}
        for name, Sess in (
            ("nf", bngsim.NfsimSession),
            ("rm", bngsim.RuleMonkeySession),
        ):
            with Sess(str(nfsim_switchable_rate_xml)) as s:
                s.set_param("scale", 0.5)
                s.initialize(seed=1)
                counts[name] = s.get_species_count("A(b)")
        assert counts["rm"] == counts["nf"] == 50

    def test_fractional_derived_seed_rounds_like_nfsim(self, nfsim_switchable_rate_xml):
        """Fractional derived seed amounts round half-up identically (GH #51)."""
        counts = {}
        for name, Sess in (
            ("nf", bngsim.NfsimSession),
            ("rm", bngsim.RuleMonkeySession),
        ):
            with Sess(str(nfsim_switchable_rate_xml)) as s:
                s.set_param("scale", 0.337)  # Ntot = 33.7 -> round-half-up 34
                s.initialize(seed=1)
                counts[name] = s.get_species_count("A(b)")
        assert counts["rm"] == counts["nf"] == 34

    def test_clear_overrides_reverts_seed_amount(self, nfsim_switchable_rate_xml):
        with bngsim.RuleMonkeySession(str(nfsim_switchable_rate_xml)) as rm:
            rm.set_param("scale", 0.5)
            rm.clear_param_overrides()
            rm.initialize(seed=1)
            assert rm.get_species_count("A(b)") == 100  # back to Ntot = 100


def _core_simulator(xml_path):
    """A bare ``RuleMonkeySimulator`` for multi-point scan tests.

    ``RuleMonkeySession.destroy()`` is terminal and ``set_param`` refuses while
    a session is live, so the session wrapper cannot walk a dose curve on one
    loaded model. The scan pattern GH #115 is about — one parse, one
    ``set_param`` per point — only exists at the core level, so these tests
    drive it directly.
    """
    from bngsim._bngsim_core import RuleMonkeySimulator

    return RuleMonkeySimulator(str(xml_path))


class TestRuleMonkeyScanOnOneSimulator:
    """GH #115: a pre-init set_param is a parameter update, not a model rebuild.

    The adapter used to answer every pending override by re-baking the XML and
    reconstructing the engine from scratch. Upstream RuleMonkey v3.7.0
    propagates an override through its own ``<Parameter expr=>`` derivations, so
    the re-bake is gone. These pin the three things that swap could have moved:
    the override still reaches a derived seed amount, it no longer perturbs
    anything outside its dependency cone, and clearing it takes effect.
    """

    def test_dose_reaches_the_derived_fractional_seed(self, dose_seed_precision_xml):
        sim = _core_simulator(dose_seed_precision_xml)
        sim.initialize(1)
        assert sim.get_species_count("L(b)") == 1807  # LT = 1806.6422, half-up

        # LT = ((dose_nM*1e-9)*NA)*V_sim, so the seed tracks the dose linearly
        # and each point is rounded half-up on its own.
        for dose, expected in ((0.5, 903), (1.0, 1807), (2.5, 4517), (4.0, 7227)):
            sim.destroy_session()
            sim.set_param("dose_nM", dose)
            sim.initialize(1)
            assert sim.get_species_count("L(b)") == expected, f"dose={dose}"

    def test_override_does_not_perturb_outside_its_cone(self, dose_seed_precision_xml):
        """A dose must not move a rate constant it does not feed.

        ``kf = ((K*1e9)*kr)/(NA*V_sim)`` divides by Avogadro's number, which
        BNG2 writes with fewer digits in ``value=`` than in ``expr=``. Anything
        that re-evaluates the whole graph under an override — as the dropped
        re-bake did — silently re-rounds ``kf`` by ~1e-8 relative at every scan
        point, in a model where nothing about ``kf`` changed.
        """
        sim = _core_simulator(dose_seed_precision_xml)
        kf_cold = sim.get_parameter("kf")

        sim.set_param("dose_nM", 2.5)
        sim.initialize(1)
        assert sim.get_parameter("LT") != 1806.6422  # the cone did move
        assert sim.get_parameter("kf") == kf_cold  # ... and nothing else did

    def test_no_op_override_changes_nothing_at_all(self, nfsim_empty_model_xml):
        """Re-assigning a parameter its own value must be a no-op, bit-for-bit.

        Swept over a real BNG2 parameter block (24 parameters, most of them
        derived through ``pi`` / ``NA`` / ``Vecf``) rather than a hand-authored
        one — the same gate upstream asserts across its own corpus. The
        ``initialize()`` matters: it is the point the dropped re-bake ran, so
        without it this sweep would pass on either side of GH #115.
        """
        sim = _core_simulator(nfsim_empty_model_xml)
        names = [
            "f", "NA", "pi", "Vecf", "Dcell", "Hepm", "Vepm", "eta_ecf", "eta_mem",
            "mwt_BSA", "mwt_DNP", "mwt_DCT", "n", "mwt_DNPnBSA", "D_BSA", "Lconc_nM",
            "Lconc", "Lcpc", "Rcpc", "KA_DCT", "kon_DCT", "koff_DCT", "kon", "koff",
        ]  # fmt: skip
        before = {p: sim.get_parameter(p) for p in names}

        sim.set_param("Lconc_nM", before["Lconc_nM"])
        sim.initialize(1)

        after = {p: sim.get_parameter(p) for p in names}
        assert after == before

    def test_a_real_override_moves_exactly_its_cone(self, nfsim_empty_model_xml):
        """The complement of the no-op sweep: everything downstream, nothing else."""
        sim = _core_simulator(nfsim_empty_model_xml)
        names = ["f", "NA", "pi", "Vecf", "Vepm", "Lconc_nM", "Lconc", "Lcpc", "Rcpc", "kon"]
        before = {p: sim.get_parameter(p) for p in names}

        sim.set_param("Lconc_nM", 2.0)
        sim.initialize(1)

        after = {p: sim.get_parameter(p) for p in names}
        moved = {p for p in names if after[p] != before[p]}
        # Lconc = 1e-9*Lconc_nM and Lcpc = Lconc*(NA*Vecf) are the whole cone.
        assert moved == {"Lconc_nM", "Lconc", "Lcpc"}
        assert after["Lcpc"] == pytest.approx(2 * before["Lcpc"])

    def test_seed_rounding_follows_a_scan_across_integral_points(self, dose_seed_precision_xml):
        """The half-up seed policy is re-derived per point, not pinned once.

        The rounding is applied as an engine-level pin on top of the resolved
        ``concentration`` expression, and a pin deliberately outlives a
        ``set_param``. ``rscale = 0.5`` lands ``RT`` on an exact 150 right after
        a point that pinned 152, so a pin left over from the previous point
        shows up here as an off-by-two.
        """
        sim = _core_simulator(dose_seed_precision_xml)
        for rscale, expected in ((1.0, 300), (0.505, 152), (0.5, 150), (0.337, 101), (2.0, 600)):
            sim.destroy_session()
            sim.set_param("rscale", rscale)
            sim.initialize(1)
            assert sim.get_species_count("R(b)") == expected, f"rscale={rscale}"

    def test_clear_after_a_run_reverts_the_seed_amount(self, dose_seed_precision_xml):
        """Clearing an override must reach the seed the run already consumed.

        The re-bake could not do this: it rebuilt the engine only when an
        override was *pending*, so after a run had baked a dose into the model,
        clearing left the baked seed in place and the next run silently kept
        using the cleared dose.
        """
        sim = _core_simulator(dose_seed_precision_xml)
        sim.set_param("dose_nM", 2.5)
        sim.initialize(1)
        assert sim.get_species_count("L(b)") == 4517

        sim.destroy_session()
        sim.clear_param_overrides()
        sim.initialize(1)
        assert sim.get_species_count("L(b)") == 1807
        assert sim.get_parameter("LT") == 1806.6422

    def test_repeated_run_under_a_live_override_is_stable(self, dose_seed_precision_xml):
        """Re-running with the same override must not drift.

        Each ``run()`` used to re-bake and reconstruct; each ``set_param`` now
        re-derives the seed pins in place. Either way the model must be
        idempotent under repetition — this is the assertion that catches a pin
        or a bake accumulating across points.
        """
        sim = _core_simulator(dose_seed_precision_xml)
        sim.set_param("dose_nM", 2.5)
        seen = []
        for _ in range(4):
            sim.destroy_session()
            sim.initialize(1)
            seen.append((sim.get_species_count("L(b)"), sim.get_parameter("LT")))
        assert seen == [seen[0]] * 4
        assert seen[0][0] == 4517


# RuleMonkey canonicalizes X's components as `X(y,p~0)` — note this is the
# component order the runtime pattern parser accepts for exact-species lookup
# (see TestRuleMonkeySessionPatternOrder for the order-sensitivity caveat).
_X_UNPHOS = "X(y,p~0)"
_X_PHOS = "X(y,p~1)"
_Y = "Y(x)"


class TestRuleMonkeySessionSpeciesAndExpr:
    """Issue #38 item 2: exact-species + expression session methods."""

    def test_get_species_count_separates_states(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            # Seed species: 5000 unphosphorylated X, 0 phosphorylated, 500 Y.
            assert rm.get_species_count(_X_UNPHOS) == 5000
            assert rm.get_species_count(_X_PHOS) == 0
            assert rm.get_species_count(_Y) == 500

    def test_add_species_targets_exact_state(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            rm.add_species(_X_PHOS, 25)
            assert rm.get_species_count(_X_UNPHOS) == 5000
            assert rm.get_species_count(_X_PHOS) == 25
            assert rm.get_molecule_count("X") == 5025

    def test_remove_species_targets_exact_state(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            rm.remove_species(_X_UNPHOS, 10)
            assert rm.get_species_count(_X_UNPHOS) == 4990
            assert rm.get_molecule_count("X") == 4990

    def test_set_species_count_adds_and_removes(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            rm.set_species_count(_X_PHOS, 20)
            assert rm.get_species_count(_X_PHOS) == 20
            rm.set_species_count(_X_PHOS, 7)
            assert rm.get_species_count(_X_PHOS) == 7

    def test_remove_species_too_many_raises(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            with pytest.raises(bngsim.SimulationError):
                rm.remove_species(_X_PHOS, 1)  # none live

    def test_add_species_nonpositive_raises(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            with pytest.raises(ValueError):
                rm.add_species(_X_PHOS, 0)
            with pytest.raises(ValueError):
                rm.remove_species(_X_PHOS, -3)
            with pytest.raises(ValueError):
                rm.set_species_count(_X_PHOS, -1)

    def test_evaluate_resolves_params_and_observables(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            # Parameter lookup.
            assert rm.evaluate("kon") == pytest.approx(rm.get_parameter("kon"))
            # Observable resolved against the live pool (Xtotal == 5000 at t=0).
            xtot_idx = rm.get_observable_names().index("Xtotal")
            assert rm.evaluate("Xtotal") == pytest.approx(rm.get_observable_values()[xtot_idx])
            # Overrides shadow model symbols for one evaluation.
            assert rm.evaluate("2*z", {"z": 21.0}) == pytest.approx(42.0)

    def test_evaluate_requires_initialized(self, nfsim_xml):
        # Parity divergence vs NfsimSession.evaluate (which only needs alive):
        # the RM engine resolves expressions against the live pool.
        rm = bngsim.RuleMonkeySession(str(nfsim_xml))
        with pytest.raises(bngsim.SimulationError):
            rm.evaluate("kon")
        rm.destroy()

    @pytest.mark.parametrize(
        "method,args",
        [
            ("get_species_count", (_X_UNPHOS,)),
            ("add_species", (_X_PHOS, 1)),
            ("remove_species", (_X_UNPHOS, 1)),
            ("set_species_count", (_X_PHOS, 1)),
            ("save_species", ("/tmp/_rm_unused.species",)),
            ("save_state", ("/tmp/_rm_unused.state",)),
        ],
    )
    def test_methods_require_initialized(self, nfsim_xml, method, args):
        rm = bngsim.RuleMonkeySession(str(nfsim_xml))
        with pytest.raises(bngsim.SimulationError):
            getattr(rm, method)(*args)
        rm.destroy()


class TestRuleMonkeySessionSaveSpecies:
    """Issue #38 item 2: save_species (.species writer) — the PyBioNetGen hook."""

    def test_save_species_bng_format(self, nfsim_xml, tmp_path):
        out = tmp_path / "final.species"
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            rm.simulate(0, 1, n_points=2)
            x_total = rm.get_molecule_count("X")
            y_total = rm.get_molecule_count("Y")
            rm.save_species(out)

        text = out.read_text()
        lines = text.splitlines()
        # BNG `.species` format: `#` comment header + `<pattern>  <count>` data.
        header = [ln for ln in lines if ln.startswith("#")]
        data = [ln for ln in lines if ln and not ln.startswith("#")]
        assert header, "expected a # comment header"
        assert data, "expected at least one species data line"
        # The summed counts must equal the live molecule populations: every
        # X-complex contributes one X, every Y-complex one Y (no X-Y binding
        # has occurred over this short segment, but the invariant holds on the
        # per-molecule totals regardless).
        # Count molecule-token occurrences per complex so bound molecules
        # (e.g. Y inside an `X(...).Y(...)` complex) are still attributed:
        # total X molecules = Σ_line count × (#"X(" in pattern).
        x_sum = y_sum = 0
        for ln in data:
            pat, cnt = ln.rsplit(None, 1)
            n = int(cnt)
            x_sum += n * pat.count("X(")
            y_sum += n * pat.count("Y(")
        assert x_sum == x_total
        assert y_sum == y_total


class TestRuleMonkeySessionState:
    """Issue #38 item 1: save_state / load_state binary snapshot round-trip."""

    def test_save_load_state_roundtrip(self, nfsim_xml, tmp_path):
        snap = tmp_path / "session.state"

        # Run a segment, snapshot, then capture the post-snapshot trajectory.
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            rm.simulate(0, 5, n_points=6)
            t_snap = rm.current_time
            x_snap = rm.get_species_count(_X_UNPHOS)
            rm.save_state(snap)
            cont = rm.simulate(5, 10, n_points=6)

        # A fresh simulator loading the snapshot must reproduce the same
        # continuation exactly (same RNG stream, same toolchain).
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm2:
            rm2.load_state(snap)
            assert rm2.initialized is True
            assert rm2.seed is None  # not recoverable from a snapshot
            assert rm2.current_time == pytest.approx(t_snap)
            assert rm2.get_species_count(_X_UNPHOS) == x_snap
            cont2 = rm2.simulate(5, 10, n_points=6)

        np.testing.assert_allclose(np.asarray(cont2.observables), np.asarray(cont.observables))

    def test_load_state_schema_mismatch_raises(self, nfsim_xml, tmp_path):
        bad = tmp_path / "not_a_snapshot.state"
        bad.write_text("garbage not a valid snapshot")
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm, pytest.raises(bngsim.SimulationError):
            rm.load_state(bad)


class TestRuleMonkeySessionPatternOrder:
    """Exact-species methods are component-order-insensitive (matches NFsim).

    RuleMonkey 3.2.1 (richardposner/RuleMonkey#13, vendored via #14+#15)
    canonicalizes component order on the match path, so a non-canonical but
    semantically identical pattern (``X(p~0,y)`` vs the canonical
    ``X(y,p~0)``) resolves to the same species — the behavior NFsim already
    had. Before the fix, ``get`` silently returned 0 for the swapped order
    while ``add``/``set`` canonicalized, so ``set_species_count`` with a
    non-canonical pattern diffed against a wrong baseline and overshot.
    These are the regression tests for that fix.
    """

    def test_get_species_count_order_insensitive(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            # Both component orders must resolve to the same species.
            assert rm.get_species_count("X(y,p~0)") == 5000
            assert rm.get_species_count("X(p~0,y)") == 5000

    def test_set_species_count_order_insensitive(self, nfsim_xml):
        with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=42)
            rm.add_species("X(y,p~1)", 25)
            # Non-canonical order must diff against the live 25 and land on
            # exactly 100 (pre-fix this overshot to 125).
            rm.set_species_count("X(p~1,y)", 100)
            assert rm.get_species_count("X(y,p~1)") == 100
            assert rm.get_species_count("X(p~1,y)") == 100
