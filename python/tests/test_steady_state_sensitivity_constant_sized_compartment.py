"""A compartment an assignment rule sizes to a constant does not refuse a
steady state's columns (issue #961).

The right-hand side divides by the size the model loaded at, not the one an
assignment rule gives (issue #745), so a model whose conservation law spans
compartments of different size, and that sizes one by a rule, was refused
model-wide. A rule that reads nothing that moves, ``c2 := 2*p`` with ``p`` not
asked for, gives the loaded size all along, and the columns are right.

c1 = 1, c2 = 2p = 3, A ⇌ B at kf, kr (amounts per time), so c1·A + c2·B = 1
and kf·A = kr·B: A* = kr/(kr + c2·kf) = 1/7.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

KF, KR, C2 = 1.0, 0.5, 3.0


def _model(rule: str, extra: str = "") -> bngsim.Model:
    return bngsim.Model.from_antimony_string(
        f"compartment c1 = 1, c2; p = 1.5; {extra} c2 := {rule}\n"
        f"species A in c1 = 1, B in c2 = 0; kf = {KF}; kr = {KR}\n"
        "J1: A -> B; kf*A; J2: B -> A; kr*B\n"
    )


def test_a_size_a_rule_holds_constant_keeps_the_columns():
    """Refused on main for every parameter."""
    out = bngsim.Simulator(_model("2*p"), method="ode").steady_state(
        sensitivity_params=["kf", "kr"]
    )
    d = (KR + C2 * KF) ** 2
    want = [[-C2 * KR / d, C2 * KF / d], [KR / d, -KF / d]]  # dA*, dB* by [kf, kr]
    np.testing.assert_allclose(out.concentrations, [1 / 7, 2 / 7], rtol=1e-7)
    np.testing.assert_allclose(out.sensitivity, want, rtol=1e-6)


@pytest.mark.parametrize(
    ("rule", "extra"),
    [("2*p", ""), ("q", "q := 2*p;"), ("q", "q = 2*p;")],
    ids=["read directly", "through a rule", "through a derived parameter"],
)
def test_a_parameter_the_size_is_read_from_is_still_refused(rule, extra):
    """Its column moves the size, which the right-hand side does not."""
    sim = bngsim.Simulator(_model(rule, extra), method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="'p'"):
        sim.steady_state(sensitivity_params=["kf", "p"])


def test_a_size_a_rule_moves_with_the_state_is_still_refused():
    with pytest.raises(bngsim.SimulationError):
        bngsim.Simulator(_model("2*p + 0*A"), method="ode").steady_state(sensitivity_params=["kf"])
