"""Regression tests for issue #28: Observable referenced as a zero-arg
call (`obs()`) in BNGL must compile.

BNGL's grammar (bionetgen/bng2/Perl2/Expression.pm:870-927) accepts an
Observable as a zero-arg call (`obs()`) anywhere a bareword `obs` is
valid. BNG2.pl preserves whichever form the user wrote when emitting
the .net file. ExprTk's grammar would parse `obs()` as `obs * ()` and
reject the empty parens with ERR248, so bngsim's
``ExprTkEvaluator::compile()`` strips `name()` → `name` for any name
registered as a scalar variable (parameters, observables, species,
synthetic function-result parameters, built-in constants).

The 1-arg form `obs(s)` (LocalFunction in BNGL) is resolved by BNG2.pl
during ``generate_network`` into per-instance constant parameters and
never reaches the .net file, so bngsim has no responsibility for it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from bngsim import Model, Simulator


class TestObsZeroArgCall:
    """Models that reference observables as `obs()` must load and run."""

    def test_loads_without_crash(self, obs_zero_arg_call_net: Path):
        """The .net loads — primary regression for issue #28."""
        model = Model.from_net(obs_zero_arg_call_net)
        assert model.n_species == 2
        assert "Atot" in model.observable_names
        assert "Btot" in model.observable_names

    def test_zero_arg_call_resolves_to_observable_value(self, obs_zero_arg_call_net: Path):
        """`Atot()` in the rate law evaluates to the observable's total —
        identical semantics to the bareword `Atot`. The model is
        first-order decay A → B with rate k1*Atot (== k1*A since A is
        the only molecule contributing to Atot at t=0)."""
        model = Model.from_net(obs_zero_arg_call_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0.0, 5.0), n_points=11)

        # k1=0.5; analytical: A(t) = A0*exp(-k1*t) when reverse rate is
        # negligible, but here _rateLaw2 = k1*Btot drives B back to A.
        # At equilibrium A == B == 5 (mass conservation, equal forward
        # and reverse rates). Test the conservation invariant exactly.
        a_end = result.species[-1, 0]
        b_end = result.species[-1, 1]
        assert a_end + b_end == pytest.approx(10.0, rel=1e-6)
        # By t=5 with k1=0.5, the system is well past the time constant
        # 1/(2*k1)=1, so A and B should be near 5.
        assert a_end == pytest.approx(5.0, abs=0.5)
        assert b_end == pytest.approx(5.0, abs=0.5)

    def test_mixed_bareword_and_zero_arg_call(self, obs_zero_arg_call_net: Path):
        """The fixture uses `Atot()` in _rateLaw1 and bareword `Btot`
        in _rateLaw2. Both must resolve to the same observable totals
        for the equilibrium to hold (asserted in the prior test)."""
        # Already covered by the equilibrium assertion above; this test
        # exists to document intent — both forms must coexist.
        model = Model.from_net(obs_zero_arg_call_net)
        sim = Simulator(model, method="ode")
        # Long-time behavior should be stable equilibrium, not divergent
        result = sim.run(t_span=(0.0, 50.0), n_points=2)
        assert math.isfinite(result.species[-1, 0])
        assert math.isfinite(result.species[-1, 1])
