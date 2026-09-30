"""Issue #715 — a derived parameter that sets a species' initial condition gets
its own sensitivity column.

bngsim answers a requested derived parameter on its own terms: the column a
``force_override`` pin of it makes real. For ``B() Rt`` with ``Rt = 3*R0`` that
column starts at ∂B(0)/∂Rt = 1. Since issue #43 the IC seed was computed in
Python and routed only to the primaries the expression reaches, and any Python
seed replaces the C++ identity loop that used to cover the derived index, so the
Rt column started at 0: all zero when Rt only sets the IC, and only the
right-hand-side part when Rt also drives a rate. No warning.

Closed forms (linear ODEs, no bngsim code):

* ``B -> 0`` at kd = 0.5 from B(0) = Rt:  dB/dRt = e^(−kd·t), dB/dR0 = 3·e^(−kd·t).
* Nested, Rt = 2*Q and Q = 3*R0:  dB/dQ = 2·e^(−kd·t), dB/dRt = e^(−kd·t).
* ``0 -> B`` at rate Rt as well, with Rt = 1:  B = 2 − e^(−t/2) in each
  derivative, so dB/dRt = 2 − e^(−t/2).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

T = np.array([0.0, 1.0, 2.0])
DECAY = np.exp(-0.5 * T)

DERIVED = """\
begin parameters
    1 R0  10
    2 kd  0.5
    3 Rt  3*R0
end parameters
begin species
    1 B() Rt
end species
begin reactions
    1 1 0 kd
end reactions
"""

NESTED = """\
begin parameters
    1 R0  10
    2 kd  0.5
    3 Q   3*R0
    4 Rt  2*Q
end parameters
begin species
    1 B() Rt
end species
begin reactions
    1 1 0 kd
end reactions
"""

# Rt = 1 (through R0 = 1/3) sets B(0) and is also the production rate.
PRODUCED = """\
begin parameters
    1 R0  0.3333333333333333
    2 kd  0.5
    3 Rt  3*R0
end parameters
begin species
    1 B() Rt
end species
begin reactions
    1 0 1 Rt
    2 1 0 kd
end reactions
"""


def _column(tmp_path, text, params):
    net = tmp_path / "m.net"
    net.write_text(text)
    model = bngsim.Model.from_net(net)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=list(params)).run(
        sample_times=list(T), rtol=1e-11, atol=1e-13
    )
    return np.asarray(run.sensitivities)[:, 0, :]


def test_the_derived_column_starts_at_one(tmp_path):
    s = _column(tmp_path, DERIVED, ["Rt", "R0"])
    np.testing.assert_allclose(s[:, 0], DECAY, rtol=1e-8)
    np.testing.assert_allclose(s[:, 1], 3.0 * DECAY, rtol=1e-8)


def test_the_derived_column_alone(tmp_path):
    """Requested by itself, the same column: no primary row to lean on."""
    s = _column(tmp_path, DERIVED, ["Rt"])
    np.testing.assert_allclose(s[:, 0], DECAY, rtol=1e-8)


def test_every_derived_parameter_on_the_chain_gets_its_column(tmp_path):
    s = _column(tmp_path, NESTED, ["Q", "Rt", "R0"])
    np.testing.assert_allclose(s[:, 0], 2.0 * DECAY, rtol=1e-8)
    np.testing.assert_allclose(s[:, 1], DECAY, rtol=1e-8)
    np.testing.assert_allclose(s[:, 2], 6.0 * DECAY, rtol=1e-8)


def test_an_ic_parameter_that_also_drives_a_rate_carries_both_parts(tmp_path):
    """Before the fix this returned only the rate part, 2·(1 − e^(−t/2)):
    0 at t = 0 by construction and 44% low at t = 1."""
    s = _column(tmp_path, PRODUCED, ["Rt"])
    np.testing.assert_allclose(s[:, 0], 2.0 - DECAY, rtol=1e-7)


def test_the_effective_seed_reports_the_derived_row(tmp_path):
    net = tmp_path / "m.net"
    net.write_text(DERIVED)
    model = bngsim.Model.from_net(net)
    assert model.effective_ic_sensitivity(["Rt", "R0"]) == {"B()": {"R0": 3.0, "Rt": 1.0}}


def test_an_sbml_initial_assignment_gets_its_column_too(tmp_path):
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "species S; p1 = 10; p2 = 3*p1; S = p2; kd = 0.5; J0: S -> ; kd*S"
    )
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["p2"]).run(
        sample_times=list(T), rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(np.asarray(run.sensitivities)[:, 0, 0], DECAY, rtol=1e-8)
