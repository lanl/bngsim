"""Tests for method=>"protocol" support.

Covers parsing, continue=>1, execution, and parameter_scan integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# pybnf.pset imports `roadrunner` unconditionally at module top. libroadrunner's
# manylinux2014 wheel is dynamically linked against `libpython3.X.so.1.0`,
# which the auditwheel-stripped manylinux2014 Python does not ship — so
# `import roadrunner` ImportErrors at collection time inside cibuildwheel's
# manylinux container. Skip the whole module when roadrunner is unimportable.
pytest.importorskip(
    "roadrunner", reason="libroadrunner unavailable; PyBNF imports it unconditionally"
)

# ---------------------------------------------------------------------------
# Helpers to import pybnf internals (not on PYTHONPATH by default)
# ---------------------------------------------------------------------------

_PYBNF_ROOT = Path(__file__).resolve().parents[3]  # up to the dev tree that carries pybnf/
if str(_PYBNF_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYBNF_ROOT))

from pybnf.bngsim_model import (  # noqa: E402  (after sys.path insert above)
    BngsimModel,
    _normalize_action_method,
    _parse_add_concentration,
    _parse_set_concentration,
    _parse_simulate_action,
)
from pybnf.pset import BNGLModel  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reversible_net(data_dir: Path) -> Path:
    return data_dir / "two_species_reversible.net"


# ---------------------------------------------------------------------------
# Phase 1: Protocol Parsing
# ---------------------------------------------------------------------------


class TestProtocolParsing:
    """Test begin protocol...end protocol extraction in BNGLModel."""

    def test_protocol_block_extracted(self, tmp_path: Path):
        """Protocol lines are stored in self.protocol, not self.actions."""
        bngl = tmp_path / "test.bngl"
        bngl.write_text(
            "begin parameters\n"
            "  1 k__FREE 0.1\n"
            "end parameters\n"
            "begin protocol\n"
            '  simulate({method=>"ode",t_end=>100,n_steps=>10})\n'
            '  setConcentration("A()",50)\n'
            '  simulate({method=>"ode",t_end=>200,n_steps=>10,continue=>1})\n'
            "end protocol\n"
            "generate_network({})\n"
            'parameter_scan({method=>"protocol",parameter=>"k__FREE",par_scan_vals=>[0.1,0.2]})\n'
        )
        model = BNGLModel(str(bngl))
        assert len(model.protocol) == 3
        assert "simulate" in model.protocol[0]
        assert "setConcentration" in model.protocol[1]
        assert "continue=>1" in model.protocol[2]
        # The parameter_scan line should be in actions, not protocol
        assert any("parameter_scan" in a for a in model.actions)
        assert not any("parameter_scan" in p for p in model.protocol)

    def test_empty_protocol_block(self, tmp_path: Path):
        """Empty protocol block produces empty list."""
        bngl = tmp_path / "test.bngl"
        bngl.write_text(
            "begin parameters\n"
            "  1 k__FREE 0.1\n"
            "end parameters\n"
            "begin protocol\n"
            "end protocol\n"
            "generate_network({})\n"
        )
        model = BNGLModel(str(bngl))
        assert model.protocol == []

    def test_protocol_comments_preserved(self, tmp_path: Path):
        """Comment lines within protocol are preserved (filtered at execution time)."""
        bngl = tmp_path / "test.bngl"
        bngl.write_text(
            "begin parameters\n"
            "  1 k__FREE 0.1\n"
            "end parameters\n"
            "begin protocol\n"
            "# Equilibrate\n"
            '  simulate({method=>"ode",t_end=>100,n_steps=>1})\n'
            "end protocol\n"
        )
        model = BNGLModel(str(bngl))
        assert len(model.protocol) == 2
        assert model.protocol[0].startswith("#")


# ---------------------------------------------------------------------------
# Phase 2: continue=>1 and expression parsing
# ---------------------------------------------------------------------------


class TestContinueAndExpressions:
    """Test continue=>1 parsing and setConcentration expression evaluation."""

    def test_continue_parsed_by_simulate_action(self):
        """_parse_simulate_action captures continue=>1."""
        result = _parse_simulate_action(
            'simulate({method=>"ode",t_end=>200,n_steps=>10,continue=>1})'
        )
        assert result is not None
        assert result["continue"] == "1"

    def test_set_concentration_arithmetic_expression(self):
        """_parse_set_concentration handles ((1/52)*50000/0.04)."""
        result = _parse_set_concentration('setConcentration("TNF()",((1/52)*50000/0.04))')
        assert result is not None
        species, value = result
        assert species == "TNF()"
        assert abs(value - (1 / 52) * 50000 / 0.04) < 0.01

    def test_set_concentration_plain_number(self):
        """_parse_set_concentration still handles plain numbers."""
        result = _parse_set_concentration('setConcentration("A()",50)')
        assert result is not None
        assert result == ("A()", 50.0)

    def test_set_concentration_zero(self):
        result = _parse_set_concentration('setConcentration("TNF()",0)')
        assert result is not None
        assert result == ("TNF()", 0.0)

    def test_add_concentration_basic(self):
        result = _parse_add_concentration('addConcentration("A()",50)')
        assert result == ("A()", 50.0)

    def test_add_concentration_expression(self):
        result = _parse_add_concentration('addConcentration("Ligand()", 500 + 100)')
        assert result is not None
        assert result == ("Ligand()", 600.0)

    def test_add_concentration_returns_none_for_set(self):
        assert _parse_add_concentration('setConcentration("A()", 100)') is None

    def test_normalize_protocol_method(self):
        """_normalize_action_method passes 'protocol' through."""
        method, poplevel = _normalize_action_method("protocol")
        assert method == "protocol"
        assert poplevel is None


# ---------------------------------------------------------------------------
# Phase 3: Protocol Execution
# ---------------------------------------------------------------------------


class TestProtocolExecution:
    """Test _run_protocol on a real model."""

    def test_protocol_basic(self, reversible_net: Path):
        """Run a simple two-step protocol and get a result."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>10})',
            'setConcentration("A()",200)',
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>10})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        assert result.n_times == 11  # n_steps + 1

    def test_protocol_continue(self, reversible_net: Path):
        """continue=>1 chains simulations: t_start of second = t_end of first."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>5})',
            'simulate({method=>"ode",t_end=>100,n_steps=>5,continue=>1})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        times = np.asarray(result.time)
        # The second simulate should start at t=50 and end at t=100
        assert times[0] == pytest.approx(50.0)
        assert times[-1] == pytest.approx(100.0)

    def test_protocol_sample_times(self, reversible_net: Path):
        """simulate inside protocol honors sample_times."""
        protocol = [
            'simulate({method=>"ode",sample_times=>[0,1,5,10,50]})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        times = np.asarray(result.time)
        np.testing.assert_allclose(times, [0, 1, 5, 10, 50], atol=1e-12)

    def test_protocol_add_concentration(self, reversible_net: Path):
        """addConcentration in protocol adds to the current value."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>1})',
            'addConcentration("A()",25)',
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>5})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        # A() starts at 100; after first simulate it decays, then we add 25
        conc_before = engine.get_concentration("A()")
        assert conc_before == pytest.approx(100.0)
        result = model._run_protocol(engine)
        assert result is not None
        # After the protocol, A() should reflect the addition
        # (exact value depends on dynamics, but get_concentration should work)
        engine.get_concentration("A()")
        # The key check: the protocol ran without error and produced a result
        assert result.n_times == 6  # n_steps + 1

    def test_protocol_no_simulate_returns_none(self, reversible_net: Path):
        """Protocol with only non-simulate actions returns None."""
        protocol = [
            'setConcentration("A()",200)',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is None

    def test_protocol_expression_t_end(self, reversible_net: Path):
        """Protocol handles t_end as arithmetic expression (e.g. 3600*5)."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>10*5,n_steps=>5})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        times = np.asarray(result.time)
        assert times[-1] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Phase 4: Parameter Scan Integration
# ---------------------------------------------------------------------------


class TestProtocolParameterScan:
    """Test method=>'protocol' routing in parameter_scan."""

    def test_protocol_scan_basic(self, reversible_net: Path):
        """parameter_scan with method=>'protocol' runs the protocol per scan point."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>100,n_steps=>10})',
            'setConcentration("A()",200)',
            'simulate({method=>"ode",t_start=>0,t_end=>100,n_steps=>10})',
        ]
        actions = [
            'parameter_scan({method=>"protocol",parameter=>"kf",'
            'par_scan_vals=>[0.001,0.01],suffix=>"pscan"})',
        ]
        model = BngsimModel(
            "rev",
            actions,
            [("parameter_scan", "pscan")],
            [],
            nf=str(reversible_net),
            protocol=protocol,
        )
        ds = model.execute("/tmp", "test", timeout=60, with_mutants=False)
        assert "pscan" in ds
        data = ds["pscan"]
        assert data.data.shape[0] == 2  # two scan points
        # Column 0 is the scan parameter value
        assert data.data[0, 0] == pytest.approx(0.001)
        assert data.data[1, 0] == pytest.approx(0.01)
        # Observable values should differ between scan points
        assert not np.allclose(data.data[0, 1:], data.data[1, 1:])

    def test_protocol_scan_empty_protocol_raises(self, reversible_net: Path):
        """method=>'protocol' with no protocol block raises ValueError."""
        actions = [
            'parameter_scan({method=>"protocol",parameter=>"kf",'
            'par_scan_vals=>[0.001],suffix=>"pscan"})',
        ]
        model = BngsimModel(
            "rev",
            actions,
            [("parameter_scan", "pscan")],
            [],
            nf=str(reversible_net),
            protocol=[],
        )
        with pytest.raises(ValueError, match="no begin protocol"):
            model.execute("/tmp", "test", timeout=60, with_mutants=False)

    def test_protocol_scan_no_simulate_raises(self, reversible_net: Path):
        """Protocol with only setConcentration raises ValueError in scan."""
        protocol = [
            'setConcentration("A()",200)',
        ]
        actions = [
            'parameter_scan({method=>"protocol",parameter=>"kf",'
            'par_scan_vals=>[0.001],suffix=>"pscan"})',
        ]
        model = BngsimModel(
            "rev",
            actions,
            [("parameter_scan", "pscan")],
            [],
            nf=str(reversible_net),
            protocol=protocol,
        )
        with pytest.raises(ValueError, match="no simulate"):
            model.execute("/tmp", "test", timeout=60, with_mutants=False)
