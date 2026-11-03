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


@pytest.mark.parametrize(
    "rule",
    ["2*p + 1e-20", "2*p*avogadro/avogadro", "2*p*pi/pi", "2*p*exp(0)"],
    ids=["an exponent", "a large one", "a built-in constant", "a built-in call"],
)
def test_a_constant_size_written_with_a_number_or_a_builtin_keeps_the_columns(rule):
    """`1e-20`'s `e` read as a name, and `_pi` as no parameter: both were
    taken to move, and the columns were refused as for a rule over the state."""
    out = bngsim.Simulator(_model(rule), method="ode").steady_state(sensitivity_params=["kf"])
    assert np.all(np.isfinite(out.sensitivity))


@pytest.mark.parametrize(
    ("rule", "extra", "asked", "moves"),
    [
        ("2*p", "", ["kf"], False),
        ("2*p + 1e-20", "", ["kf"], False),
        ("2*p*pi/pi", "", ["kf"], False),
        ("2*p", "", ["p"], True),
        ("q", "q := 2*p;", ["p"], True),
        ("q", "q = 2*p;", ["p"], True),
        ("2*p + 0*A", "", ["kf"], True),
        ("2*p + 0*time", "", ["kf"], True),
        ("q", "q := 2*p + 0*time;", ["kf"], True),
    ],
    ids=[
        "constant",
        "an exponent",
        "a built-in constant",
        "asked",
        "asked through a rule",
        "asked through a derived parameter",
        "a species",
        "the time",
        "the time through a rule",
    ],
)
def test_which_sizes_move(rule, extra, asked, moves):
    """What the refusal asks of each size. The refusals before it answer every
    moving case here first (a parameter read at load, the time, the isolated
    root), so it is asked of the helper itself. `time()`, as the engine spells
    the clock, read as a call and not as the time."""
    from bngsim._simulator import Simulator

    got = Simulator._rule_sizes_that_move(_model(rule, extra), ["c2"], asked)
    assert got == (["c2"] if moves else [])
