"""A derived expression sympy cannot differentiate declines instead of crashing.

Issue #441. Sympy answers ``d/dP floor(P)`` with an unevaluated ``Derivative``
object rather than with a number or an exception, and calling ``evalf`` on a bare
one recurses until Python raises ``RecursionError``. That is not an exception any
caller on this path expects, so it escaped the codegen build and came out of a
plain simulation as ``SimulationError: maximum recursion depth exceeded``.

Whether it crashed depended on the shape of the expression, which is why it was
easy to miss: ``floor(P)`` crashed, while ``floor(P)*7`` came back as a clean
refusal, because a product containing an unevaluated ``Derivative`` raises
``TypeError`` before the recursion starts.

The right answer for all of these is to decline. A value that steps as a
parameter moves has no useful derivative with respect to it, so the caller gets
the same empty result it already gets for any expression that cannot be
differentiated, and the existing warning says the chain rule was dropped.
"""

from __future__ import annotations

import logging

import bngsim
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

# The model from the issue: an initial condition built from a step function.
STEP_IC = """\
begin parameters
    1 P       24.0  # Constant
    2 k       0.1  # Constant
    3 A0      floor(P)
end parameters
begin species
    1 A() A0
end species
begin reactions
    1 1 0 k #_R1
end reactions
begin groups
    1 A                    1
end groups
"""

_PRIMARIES = {"P"}
_IDX = {"P": 0}
_VALUES = [24.0]


class TestAStepFunctionDeclinesRatherThanCrashes:
    """The numeric partials, which is where every affected caller goes."""

    @pytest.mark.parametrize(
        "expr",
        [
            "floor(P)",  # the bare shape that recursed
            "ceil(P)",
            "sign(P)",
            "floor(P)*7",  # the shape that already declined, unchanged
            "floor(P)+2*P",
            "floor(P/2)*P",
            "rint(P)",  # a name sympy does not know at all, same answer
        ],
    )
    def test_the_partials_come_back_empty(self, expr):
        assert cg._derived_expr_partials_numeric(expr, _PRIMARIES, _IDX, _VALUES, {}) == {}

    def test_the_decline_is_reported(self, caplog):
        """An empty result is indistinguishable from a primary that genuinely
        does not appear, so the warning is the only thing that separates the two
        (issue #56). It has to name the expression and the parameter whose chain
        rule was dropped."""
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert (
                cg._derived_expr_partials_numeric("floor(P)", _PRIMARIES, _IDX, _VALUES, {}) == {}
            )
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "could not differentiate" in message
        assert "'floor(P)'" in message and "P" in message
        assert "steps rather than moves" in message

    def test_a_partial_that_does_differentiate_still_survives(self):
        """The refusal is per parameter, not per expression: ``Q`` is an ordinary
        multiplier of a stepped value and keeps its partial, which is what the
        rest of this path does with a lost one."""
        out = cg._derived_expr_partials_numeric(
            "Q*floor(P)", {"P", "Q"}, {"P": 0, "Q": 1}, [24.0, 3.0], {}
        )
        assert out == pytest.approx({"Q": 24.0})

    def test_the_c_emitting_twin_gives_the_same_reason(self):
        """``sp.ccode`` refuses the object too, so the model declined either way.
        Asking first is what makes the reason readable, since it is published
        with the run's df/dp verdict (issue #438)."""
        partials, reason = cg._direct_derived_partials("floor(P)", _PRIMARIES, (), _IDX, None)
        assert partials is None
        assert reason == cg._STEP_DERIVATIVE_REASON
        assert "steps rather than moves" in reason

    def test_an_ordinary_expression_is_untouched(self):
        """The guard is structural, so the thing to check is that it does not
        also catch expressions that differentiate perfectly well."""
        out = cg._derived_expr_partials_numeric("2*P + P*P", _PRIMARIES, _IDX, _VALUES, {})
        assert out == pytest.approx({"P": 2.0 + 2.0 * 24.0})


@requires_cc
class TestTheIssuesModelRuns:
    """The report's own repro, end to end. The initial condition ``A0 = floor(P)``
    is the derived-parameter seeding path, which is one of the callers the switch
    threshold guard added for issue #436 never covered."""

    def test_a_step_initial_condition_runs_and_reports_a_zero_column(self, tmp_path, caplog):
        net = tmp_path / "step_ic.net"
        net.write_text(STEP_IC)
        model = bngsim.Model.from_net(net)
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            result = bngsim.Simulator(model, sensitivity_params=["P"]).run(
                t_span=(0.0, 1.0), n_points=3
            )
        # The value is seeded from the parameter as usual — only its derivative
        # is refused.
        assert result.species[0, 0] == pytest.approx(24.0)
        assert result.sensitivities is not None
        assert (result.sensitivities == 0.0).all()
        assert any("could not differentiate" in r.getMessage() for r in caplog.records)
