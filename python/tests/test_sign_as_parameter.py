"""Regression test for the `sign`-collision corner case.

bngsim registers a custom ExprTk function `sign` (alongside the built-in
`sgn`) at expression.cpp:319. BNG2.pl's parser does not consider `sign`
a built-in (see %functions in bionetgen/bng2/Perl2/Expression.pm:56-96),
so it accepts `sign` as a user parameter name and emits the .net
verbatim. Pre-fix, bngsim's define_variable("sign", ...) failed at
registration with a confusing "name already registered" error because
`sign` is not on ExprTk's reserved_symbols[] list (so no mangling)
yet IS in bngsim's symbol table (so add_variable rejects).

The fix adds `sign` (and the other bngsim-only aliases `ln`, `rint`,
`mratio`, `time`) to is_exprtk_reserved()'s mangling set, so the user
parameter is registered under the mangled key `r_sign` and rate-law
references resolve through it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from bngsim import Model, Simulator


def test_sign_loads(sign_as_parameter_net: Path):
    """The .net loads — primary regression."""
    model = Model.from_net(sign_as_parameter_net)
    assert model.n_species == 1
    assert "sign" in model.param_names


def test_sign_evaluates_correctly(sign_as_parameter_net: Path):
    """Rate law `sign` (the parameter, value=1.0) drives A → 0 with
    first-order kinetics. A(t) = 10*exp(-1*t)."""
    model = Model.from_net(sign_as_parameter_net)
    sim = Simulator(model, method="ode")
    result = sim.run(t_span=(0.0, 5.0), n_points=6)
    a_end = result.species[-1, 0]
    assert a_end == pytest.approx(10.0 * math.exp(-5.0), rel=1e-4)
