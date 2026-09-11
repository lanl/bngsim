"""Issue #506: every path that puts a model on the finite-difference Jacobian says
why — one INFO line on the ``bngsim`` logger at the decline site, and the same
reason readable as ``Model.analytical_jacobian_status``.

Before, four of the decline paths — an un-inlinable function, a parse failure, a
symbol that survived inlining, a derivative no emitter prints — and the C++
attach gate went silent at every level; a harness could record nothing but the
boolean, and recovering a reason took a monkeypatch of ``differentiate_rate_law``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bngsim
import pytest
import sympy as sp
from bngsim import _jacobian as J


def _decline_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "bngsim"
        and r.levelno == logging.INFO
        and "analytical Jacobian" in r.getMessage()
    ]


@pytest.fixture
def info(caplog):
    with caplog.at_level(logging.INFO, logger="bngsim"):
        yield caplog


# ─── The decline sites in differentiate_rate_law ─────────────────────────────


def test_function_cycle_is_logged(info):
    assert J.build_per_observable_terms("g*A", {"g": "g+1"}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "'g*A'" in line and "cycle" in line and "finite-difference" in line


def test_parse_failure_is_logged(info):
    assert J.build_per_observable_terms("k*A*", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "'k*A*'" in line and "could not be parsed" in line


def test_surviving_symbol_is_named(info):
    assert J.build_per_observable_terms("k*A*Z", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "['Z']" in line and "hidden state" in line


def test_underivable_function_is_named(info):
    assert J.build_per_observable_terms("k*sign(A)", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "applies sign to a differentiation variable" in line


def test_non_emittable_derivative_names_the_node_and_the_variable(info, monkeypatch):
    reason = "carries the function Heaviside, which no emitter spells"
    monkeypatch.setattr(J, "_non_emittable_reason", lambda e: reason)
    # differentiate_rate_law directly: the term builders try the native saturable
    # path first, and a rate law inside that family never reaches sympy.
    assert J.differentiate_rate_law("k*exp(A)", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "with respect to A carries the function Heaviside" in line


def test_a_rate_law_that_derives_logs_nothing(info):
    assert J.build_per_observable_terms("k*A*B", {}, {"A", "B"}, {"k"}) is not None
    assert _decline_lines(info) == []


def test_non_emittable_reasons_name_the_offending_node():
    x, y = sp.symbols("x y")
    assert "Heaviside" in J._non_emittable_reason(sp.Heaviside(x))
    assert "zoo" in J._non_emittable_reason(sp.zoo * x)
    assert "Derivative" in J._non_emittable_reason(sp.Derivative(sp.sign(x), x))
    assert "Xor" in J._non_emittable_reason(sp.Xor(x > 0, y > 0))
    assert J._non_emittable_reason(sp.exp(x) * sp.Piecewise((x, x > 0), (0, True))) is None
    assert J._is_emittable(sp.exp(x)) is True


# ─── Model.analytical_jacobian_status ────────────────────────────────────────

_NET_SIGN = """\
begin parameters
    1 k 0.7
end parameters
begin species
    1 A() 1.0
    2 B() 0.0
end species
begin functions
    1 fSign() k*sign(A_tot)
end functions
begin reactions
    1 1 2 fSign #A_to_B
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
end groups
"""

_NET_SQRT_AT_ZERO = (
    _NET_SIGN.replace("1 A() 1.0", "1 A() 0.0")
    .replace("fSign() k*sign(A_tot)", "fSign() k*sqrt(A_tot)")
    .replace("1 1 2 fSign", "1 0 2 fSign")
)


def _model(tmp_path: Path, text: str) -> bngsim.Model:
    p = tmp_path / "m.net"
    p.write_text(text)
    return bngsim.Model.from_net(str(p))


def test_status_is_complete_for_a_mass_action_model(data_dir: Path):
    m = bngsim.Model.from_net(str(data_dir / "simple_decay.net"))
    assert m.analytical_jacobian_status == "complete"  # closed form from the start


def test_status_is_pending_until_derived_then_declined_with_the_reason(tmp_path, info):
    m = _model(tmp_path, _NET_SIGN)
    assert m.analytical_jacobian_status == "pending"
    assert m.prepare_analytical_jacobian() is False
    status = m.analytical_jacobian_status
    assert status.startswith("declined: rate law 'k*sign(A_tot)' applies sign"), status
    (line,) = _decline_lines(info)
    assert status[len("declined: ") :] in line  # the same reason the log line carries


def test_status_after_the_cpp_gate_declines(tmp_path, info):
    m = _model(tmp_path, _NET_SQRT_AT_ZERO)
    assert m.prepare_analytical_jacobian() is False
    assert m.analytical_jacobian_status.startswith("declined: the C++ attach declined")
    assert any("C++ attach declined" in line for line in _decline_lines(info))


def test_status_is_complete_once_attached(data_dir: Path):
    m = bngsim.Model.from_net(str(data_dir / "jac_selfcheck_switching_surface.net"))
    assert m.analytical_jacobian_status == "pending"
    assert m.prepare_analytical_jacobian() is True
    assert m.analytical_jacobian_status == "complete"


def test_clone_carries_the_status(tmp_path):
    m = _model(tmp_path, _NET_SIGN)
    m.prepare_analytical_jacobian()
    assert m.clone().analytical_jacobian_status == m.analytical_jacobian_status


def test_env_switch_is_a_named_decline(tmp_path, monkeypatch, info):
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", "0")
    m = _model(tmp_path, _NET_SIGN)
    assert m.prepare_analytical_jacobian() is False
    expected = "declined: disabled by BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0"
    assert m.analytical_jacobian_status == expected
