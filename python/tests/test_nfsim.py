"""Tests for NFsim integration via bngsim.Simulator(method="nfsim").

Phase C.5 Step 3: Python bindings for NfsimSimulator.

Uses simple_system.xml which has:
  - 4 reaction rules (bind, unbind, catalyze, dephos)
  - 2 molecule types: X(y,p~0~1) and Y(x)
  - Initial: 5000 X(p~0,y), 500 Y(x)
  - 6 observables: X_free, X_p_total, Xp_free, XY, Ytotal, Xtotal
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _has_nfsim() -> bool:
    """Check if NFsim support is compiled in."""
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return HAS_NFSIM
    except ImportError:
        return False


# Skip entire module if NFsim not available
pytestmark = pytest.mark.skipif(
    not _has_nfsim(),
    reason="bngsim compiled without NFsim support",
)


# We need a dummy Model for the Simulator constructor (NFsim doesn't use it,
# but the Python API requires one). Use the simplest .net we have.
@pytest.fixture
def dummy_model(simple_decay_net: Path) -> bngsim.Model:
    """A dummy Model (NFsim ignores it; only xml_path matters)."""
    return bngsim.Model.from_net(str(simple_decay_net))


# ─── Test: basic NFsim run via Simulator ──────────────────────────────────────


class TestNfsimMethod:
    """Tests for Simulator(method='nfsim')."""

    def test_basic_run(self, dummy_model, nfsim_xml):
        """NFsim produces a Result with correct shape."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 1), n_points=11, seed=42)

        assert result.n_times == 11
        assert result.n_observables == 6
        assert len(result.time) == 11
        np.testing.assert_almost_equal(result.time[0], 0.0)
        np.testing.assert_almost_equal(result.time[-1], 1.0)

    def test_observable_names(self, dummy_model, nfsim_xml):
        """Observable names match the XML definition."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 0.1), n_points=2, seed=42)

        expected = ["X_free", "X_p_total", "Xp_free", "XY", "Ytotal", "Xtotal"]
        assert result.observable_names == expected

    def test_tfun_xml_runs(self, dummy_model, nfsim_tfun_xml):
        """TFUN XML fixture initializes and runs without crashing."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_tfun_xml))
        result = sim.run(t_span=(0, 1), n_points=6, seed=42)

        assert result.n_times == 6
        assert result.n_observables == 3
        assert result.observable_names == ["X_unphos", "X_phos", "X_total"]

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "valid_file_linear_time.xml",
            "valid_file_step_time.xml",
            "valid_inline_linear_time.xml",
            "valid_inline_step_time.xml",
            "valid_inline_parameter.xml",
            "valid_inline_observable.xml",
            "valid_inline_function.xml",
        ],
    )
    def test_tfun_new_format_valid_fixtures_run(
        self,
        dummy_model,
        nfsim_tfun_new_format_dir: Path,
        fixture_name: str,
    ):
        """All valid new-format TFUN fixtures parse and simulate."""
        xml_path = nfsim_tfun_new_format_dir / fixture_name
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(xml_path))
        result = sim.run(t_span=(0, 1), n_points=6, seed=42)

        assert result.n_times == 6
        assert result.n_observables == 3
        assert result.observable_names == ["X_unphos", "X_phos", "X_total"]

    def test_tfun_new_format_step_vs_linear_file_mode_differs(
        self,
        dummy_model,
        nfsim_tfun_new_format_dir: Path,
    ):
        """Linear and step interpolation produce different trajectories."""
        linear = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_tfun_new_format_dir / "valid_file_linear_time.xml"),
        )
        step = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_tfun_new_format_dir / "valid_file_step_time.xml"),
        )

        r_linear = linear.run(t_span=(0, 2), n_points=11, seed=123)
        r_step = step.run(t_span=(0, 2), n_points=11, seed=123)

        assert not np.array_equal(
            np.asarray(r_linear.observables),
            np.asarray(r_step.observables),
        )

    def test_observables_nonnegative(self, dummy_model, nfsim_xml):
        """All observable values should be non-negative."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 10), n_points=101, seed=12345)

        obs = np.asarray(result.observables)
        assert np.all(obs >= 0), "Found negative observable values"

    def test_conservation(self, dummy_model, nfsim_xml):
        """Xtotal=5000 and Ytotal=500 are conserved."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 10), n_points=101, seed=42)

        xtotal = result.observables["Xtotal"]
        ytotal = result.observables["Ytotal"]
        np.testing.assert_allclose(xtotal, 5000.0, atol=0.5)
        np.testing.assert_allclose(ytotal, 500.0, atol=0.5)

    def test_deterministic_seeding(self, dummy_model, nfsim_xml):
        """Same seed produces identical results."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        r1 = sim.run(t_span=(0, 1), n_points=11, seed=99999)
        r2 = sim.run(t_span=(0, 1), n_points=11, seed=99999)

        np.testing.assert_array_equal(
            np.asarray(r1.observables),
            np.asarray(r2.observables),
        )

    def test_seeded_reference_snapshot(self, dummy_model, nfsim_xml):
        """Pinned seeded trajectory for the default single-draw selector path."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 1), n_points=6, seed=42)

        # Trajectory re-pinned for the System::stepTo over-binding fix
        # (PyBNF issue #391): stepTo() no longer discards the boundary-crossing
        # reaction-time sample, so the seed=42 trajectory shifts from output
        # step 2 onward. Conserved columns (Ytotal=500, Xtotal=5000) and the
        # monotonic trends are intact; the run stays deterministic.
        expected = np.array(
            [
                [5000.0, 0.0, 0.0, 0.0, 500.0, 5000.0],
                [4432.0, 68.0, 68.0, 500.0, 500.0, 5000.0],
                [4364.0, 136.0, 136.0, 500.0, 500.0, 5000.0],
                [4322.0, 178.0, 178.0, 500.0, 500.0, 5000.0],
                [4271.0, 229.0, 229.0, 500.0, 500.0, 5000.0],
                [4221.0, 279.0, 279.0, 500.0, 500.0, 5000.0],
            ]
        )

        np.testing.assert_array_equal(np.asarray(result.observables), expected)
        assert result.solver_stats["n_steps"] == 6155

    def test_seeded_reference_snapshot_v1143_compat(self, dummy_model, nfsim_xml):
        """Compatibility mode preserves the standalone NFsim v1.14.3 trajectory."""
        sim = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_xml),
            nfsim_v1143_compat=True,
        )
        result = sim.run(t_span=(0, 1), n_points=6, seed=42)

        # Trajectory re-pinned for the System::stepTo over-binding fix
        # (PyBNF issue #391); see the note in test_seeded_reference_snapshot.
        expected = np.array(
            [
                [5000.0, 0.0, 0.0, 0.0, 500.0, 5000.0],
                [4430.0, 70.0, 70.0, 500.0, 500.0, 5000.0],
                [4374.0, 126.0, 126.0, 500.0, 500.0, 5000.0],
                [4307.0, 193.0, 193.0, 500.0, 500.0, 5000.0],
                [4256.0, 244.0, 244.0, 500.0, 500.0, 5000.0],
                [4221.0, 279.0, 279.0, 500.0, 500.0, 5000.0],
            ]
        )

        np.testing.assert_array_equal(np.asarray(result.observables), expected)
        assert result.solver_stats["n_steps"] == 6507

    def test_connectivity_kwarg_is_forwarded_to_core(self, dummy_model, nfsim_xml, monkeypatch):
        """High-level Simulator forwards explicit connectivity to NFsim core."""
        import bngsim._bngsim_core as core

        calls = []

        class FakeNfsimSimulator:
            def __init__(self, xml_path):
                self.xml_path = xml_path

            def set_molecule_limit(self, limit):
                calls.append(("gml", limit))

            def set_connectivity(self, enabled):
                calls.append(("connectivity", enabled))

            def set_nfsim_v1143_compat(self, enabled):
                calls.append(("compat", enabled))

            def set_block_same_complex_binding(self, enabled):
                calls.append(("bscb", enabled))

            def set_traversal_limit(self, limit):
                calls.append(("utl", limit))

        monkeypatch.setattr(core, "NfsimSimulator", FakeNfsimSimulator)

        sim = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_xml),
            gml=123,
            connectivity=False,
            nfsim_v1143_compat=True,
            block_same_complex_binding=False,
            traversal_limit=7,
        )

        assert isinstance(sim._sim, FakeNfsimSimulator)
        assert ("gml", 123) in calls
        assert ("connectivity", False) in calls
        assert ("compat", True) in calls
        assert ("bscb", False) in calls
        assert ("utl", 7) in calls

    def test_interactive_helpers_are_not_supported(self, dummy_model, nfsim_xml):
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))

        with pytest.raises(NotImplementedError, match="Interactive simulation helpers"):
            sim.run_until(t=1.0)
        with pytest.raises(NotImplementedError, match="Interactive simulation helpers"):
            sim.intervene({"kon": 0.0})
        with pytest.raises(NotImplementedError, match="Interactive simulation helpers"):
            sim.snapshot()
        with pytest.raises(NotImplementedError, match="Interactive simulation helpers"):
            sim.restore()

    def test_different_seeds(self, dummy_model, nfsim_xml):
        """Different seeds produce different results."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        r1 = sim.run(t_span=(0, 10), n_points=101, seed=11111)
        r2 = sim.run(t_span=(0, 10), n_points=101, seed=22222)

        # Not all values can be identical (stochastic system)
        assert not np.array_equal(
            np.asarray(r1.observables),
            np.asarray(r2.observables),
        ), "Different seeds produced identical results"

    def test_dynamics(self, dummy_model, nfsim_xml):
        """X_p_total starts at 0 and increases over time."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 10), n_points=101, seed=42)

        xp = result.observables["X_p_total"]
        assert xp[0] == 0, f"X_p_total at t=0 should be 0, got {xp[0]}"
        assert xp[-1] > 0, f"X_p_total at t=10 should be > 0, got {xp[-1]}"

    def test_solver_stats(self, dummy_model, nfsim_xml):
        """Solver stats are populated."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 10), n_points=101, seed=42)

        assert result.solver_stats["n_steps"] > 0

    def test_no_species_data(self, dummy_model, nfsim_xml):
        """NFsim Result has 0 species columns (rule-based, no species list)."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 0.1), n_points=2, seed=42)

        assert result.n_species == 0

    def test_compartment_xml_smoke(self, dummy_model, nfsim_compartment_xml):
        """cBNGL smoke test: compartmental XML parses and simulates."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_compartment_xml))
        result = sim.run(t_span=(0, 0.1), n_points=2, seed=42)
        assert result.n_observables == 6
        assert result.observable_names == [
            "X_free",
            "X_p_total",
            "Xp_free",
            "XY",
            "Ytotal",
            "Xtotal",
        ]


class TestNfsimValidation:
    """Validation and error handling tests."""

    def test_missing_xml_path_raises(self, dummy_model):
        """method='nfsim' without xml_path raises ValueError."""
        with pytest.raises(ValueError, match="xml_path"):
            bngsim.Simulator(dummy_model, method="nfsim")

    def test_invalid_xml_path_raises(self, dummy_model):
        """Non-existent XML path raises RuntimeError on run()."""
        sim = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path="/nonexistent/model.xml",
        )
        with pytest.raises(bngsim.SimulationError):
            sim.run(t_span=(0, 1), n_points=2, seed=42)

    @pytest.mark.parametrize(
        "fixture_name, expected_fragment",
        [
            ("invalid_bad_method.xml", "unsupported method"),
            (
                "invalid_missing_ctrname.xml",
                "can't find counter name for tfun function",
            ),
            (
                "invalid_inline_missing_ydata.xml",
                "requires both xdata and ydata",
            ),
            (
                "invalid_inline_mismatched_lengths.xml",
                "mismatched xdata/ydata lengths",
            ),
            (
                "invalid_inline_nonmonotonic_x.xml",
                "xdata must be strictly increasing",
            ),
            (
                "invalid_expression_missing_placeholder.xml",
                "expression must contain __tfun_val__",
            ),
        ],
    )
    def test_tfun_new_format_invalid_fixtures_raise_with_reason(
        self,
        dummy_model,
        nfsim_tfun_new_format_dir: Path,
        fixture_name: str,
        expected_fragment: str,
    ):
        """Malformed TFUN metadata fails with a specific validation reason."""
        xml_path = nfsim_tfun_new_format_dir / fixture_name
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(xml_path))
        with pytest.raises(bngsim.SimulationError) as exc:
            sim.run(t_span=(0, 1), n_points=2, seed=42)

        msg = str(exc.value).lower()
        assert expected_fragment in msg


class TestNfsimLowLevel:
    """Tests for the low-level _bngsim_core.NfsimSimulator binding."""

    def test_core_nfsim_simulator(self, nfsim_xml):
        """Direct NfsimSimulator C++ binding works."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_xml))
        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 11

        result = sim.run(times, 42)

        assert result.n_times == 11
        assert result.n_observables == 6
        assert len(result.observable_names) == 6
        assert result.observable_names[0] == "X_free"

    def test_core_v1143_compat_flag_changes_seeded_path(self, nfsim_xml):
        """Low-level binding exposes the NFsim v1.14.3 compatibility flag."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 6

        sim_default = NfsimSimulator(str(nfsim_xml))
        default_result = sim_default.run(times, 42)

        sim_compat = NfsimSimulator(str(nfsim_xml))
        sim_compat.set_nfsim_v1143_compat(True)
        compat_result = sim_compat.run(times, 42)

        assert not np.array_equal(
            np.asarray(default_result.observable_data),
            np.asarray(compat_result.observable_data),
        )

    def test_core_set_param(self, nfsim_xml):
        """set_param on NfsimSimulator zeroes out binding."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_xml))
        sim.set_param("kon", 0.0)

        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 11

        result = sim.run(times, 42)

        # Find XY observable index
        xy_idx = result.observable_names.index("XY")
        obs = np.array(result.observable_data)

        # With kon=0, XY should remain 0
        assert np.all(obs[:, xy_idx] == 0), "XY should be 0 with kon=0"

    def test_has_nfsim_flag(self):
        """HAS_NFSIM module attribute is True."""
        from bngsim._bngsim_core import HAS_NFSIM

        assert HAS_NFSIM is True

    def test_set_molecule_limit_low_fails(self, nfsim_xml):
        """set_molecule_limit(10) should fail for a model with 5000 X molecules."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_xml))
        sim.set_molecule_limit(10)  # Way too low for X=5000

        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 2

        with pytest.raises(RuntimeError):
            sim.run(times, 42)

    def test_set_molecule_limit_high_succeeds(self, nfsim_xml):
        """set_molecule_limit(10000) should succeed for X=5000, Y=500."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_xml))
        sim.set_molecule_limit(10000)

        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 2

        result = sim.run(times, 42)
        assert result.n_times == 2

    def test_gml_kwarg_on_simulator(self, dummy_model, nfsim_xml):
        """Simulator(gml=10) should fail; gml=10000 should work."""
        # gml too low → should fail
        sim_low = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_xml),
            gml=10,
        )
        with pytest.raises(Exception):  # noqa: B017  NFsim wraps multiple low-level error types
            sim_low.run(t_span=(0, 1), n_points=2, seed=42)

        # gml high enough → should succeed
        sim_ok = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_xml),
            gml=10000,
        )
        result = sim_ok.run(t_span=(0, 1), n_points=2, seed=42)
        assert result.n_times == 2


# ─── Test: BNGL global/composite functions surface as expression columns ────


class TestNfsimFunctionColumns:
    """BNGL `begin functions` entries surface as Result expression columns.

    funccols.xml declares a GlobalFunction (`phos_ratio = Xp/Xtot`, observables
    only) and a CompositeFunction (`phos_percent = 100*phos_ratio()`, a
    function-of-function). NFsim evaluates both internally for rate laws;
    these tests confirm bngsim also reports them as output columns.
    """

    def test_core_run_reports_function_columns(self, nfsim_funccols_xml):
        """Low-level NfsimSimulator.run() fills expression columns."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_funccols_xml))
        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 10.0
        times.n_points = 5
        result = sim.run(times, 42)

        assert list(result.expression_names) == ["phos_ratio", "phos_percent"]
        assert result.n_expressions == 2

        obs_names = list(result.observable_names)
        for i in range(result.n_times):
            obs = np.asarray(result.observable_data[i]).ravel()
            exprs = np.asarray(result.expression_data[i]).ravel()
            xp = obs[obs_names.index("Xp")]
            xtot = obs[obs_names.index("Xtot")]
            # GlobalFunction: phos_ratio == Xp/Xtot
            assert exprs[0] == pytest.approx(xp / xtot)
            # CompositeFunction: phos_percent == 100*phos_ratio()
            assert exprs[1] == pytest.approx(100.0 * exprs[0])

    def test_simulator_run_reports_function_columns(self, dummy_model, nfsim_funccols_xml):
        """High-level Simulator(method="nfsim") surfaces functions via Result.expressions."""
        sim = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_funccols_xml),
        )
        result = sim.run(t_span=(0, 10), n_points=5, seed=42)

        assert list(result.expression_names) == ["phos_ratio", "phos_percent"]
        exprs = np.asarray(result.expressions)
        assert exprs.shape == (5, 2)
        np.testing.assert_allclose(exprs[:, 1], 100.0 * exprs[:, 0])

    def test_session_simulate_reports_function_columns(self, nfsim_funccols_xml):
        """Session NfsimSession.simulate() surfaces the same function columns."""
        with bngsim.NfsimSession(str(nfsim_funccols_xml)) as nf:
            nf.initialize(seed=42)
            result = nf.simulate(0, 10, n_points=5)

        assert list(result.expression_names) == ["phos_ratio", "phos_percent"]
        exprs = np.asarray(result.expressions)
        assert exprs.shape == (5, 2)
        np.testing.assert_allclose(exprs[:, 1], 100.0 * exprs[:, 0])

    def test_model_without_functions_has_no_expression_columns(self, nfsim_xml):
        """A model with an empty `begin functions` block reports zero expressions."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_xml))
        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 3
        result = sim.run(times, 42)

        assert result.n_expressions == 0
        assert list(result.expression_names) == []

    def test_nf_gdat_function_headers_are_bare(self, nfsim_funccols_xml, tmp_path):
        """An nf result's .gdat function columns use bare headers (no "()").

        Issue #58 reverses the per-method "()" convention #53 introduced for
        the NFsim path: bngsim now emits one bare convention for every method.
        """
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_funccols_xml))
        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 10.0
        times.n_points = 5
        result = sim.run(times, 42)

        gdat = tmp_path / "nf_funcs.gdat"
        result.to_gdat(str(gdat), True)  # print_functions=True
        header = gdat.read_text().splitlines()[0]
        cols = header.lstrip("#").split()
        assert "phos_ratio" in cols
        assert "phos_percent" in cols
        assert "()" not in header
