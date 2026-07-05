"""Regression tests for issue #24: BNGL identifiers literally named `t`.

bngsim used to register `t` as an alias for the simulation-time function
alongside `time`, which collided with the (very common) BNG2.pl pattern of
using `t` as a parameter or observable name. The fix is to register only
`time()` (matching BNG2.pl's muParser convention) and leave `t` free as a
user identifier.

The 2026-05-09 PyBioNetGen parity sweep counted 43 corpus models that
failed to load solely on this collision; this test pins the fix.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from bngsim import Model, Simulator


class TestTAsModelIdentifier:
    """Models with `t` as a parameter / observable / species must load."""

    def test_loads_without_crash(self, t_as_observable_net: Path):
        """The .net loads — this is the primary regression for issue #24."""
        model = Model.from_net(t_as_observable_net)
        # 3 species: A, B, t (the counter molecule)
        assert model.n_species == 3
        # `t` is registered as an observable group
        assert "t" in model.observable_names

    def test_function_resolves_t_to_observable(self, t_as_observable_net: Path):
        """A function body that references `t` binds to the observable, not
        a reserved time symbol. The .net's `_rateLaw1 = k1*(1+0*t)` keeps the
        rate equal to k1 regardless of t's value, so the trajectory matches
        simple first-order decay."""
        model = Model.from_net(t_as_observable_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
        # A(10) = 100 * exp(-0.1 * 10) ≈ 36.79
        A_end = result.species[-1, 0]
        expected = 100.0 * math.exp(-0.1 * 10.0)
        assert A_end == pytest.approx(expected, rel=1e-4)


class TestTimeStillWorks:
    """time() must continue to expose the simulation-time variable."""

    def test_time_function_in_rate_law(self, time_func_net: Path):
        """The pre-existing time_dependent_func.net uses `time()` in a
        functional rate law; this test confirms that the time() binding
        survives the issue #24 fix.
        """
        model = Model.from_net(time_func_net)
        sim = Simulator(model, method="ode")
        # rate = time() + 1, so dB/dt = t + 1, B(t) = t^2/2 + t.
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
        # B is species index 1 (A is 0). B(10) = 50 + 10 = 60.
        B_end = result.species[-1, 1]
        assert B_end == pytest.approx(60.0, rel=1e-3)


class TestReservedNamesAdvertisesNoT:
    """Issue #24 fix: `t` is no longer reserved; BNG authors can use it."""

    def test_t_absent_from_reserved_functions(self):
        import bngsim

        names = bngsim.reserved_names()
        assert "time" in names["functions"]
        assert "t" not in names["functions"]
