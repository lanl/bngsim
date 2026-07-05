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
    reason="bngsim compiled without RuleMonkey support",
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
