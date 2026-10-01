"""A compartment resize and an event assignment to a species in it (issues #936,
#741).

A resize conserves each contained species' amount, by an injected concentration
rescale. That rescale read the state from before the event batch: when an
earlier event of the same instant had assigned the species, the resize undid
the assignment (#936). Under SSA, where the stored value of such a species is
amount/V_static, an event assigning it a concentration c stored c·V_static
instead of c·V_live (#741). Both now read the state as it stands when the event
executes.

Oracles are hand-computed: amounts are conserved across a resize, and an
assignment sets the concentration in the compartment as it is then.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

BATCH = (
    "compartment Cc = 1; species X in Cc, Y in Cc; X = 0; Y = 1; p = 1; b = 2;"
    " J0: => X; p;"
    " Egrow: at (time >= 2), priority = {g}: Cc = 2;"
    " Eset: at (time >= 2), priority = {s}: Y = b*p"
)


def _conc(r, name, row):
    return float(r.as_roadrunner(selections=[f"[{name}]"])[f"[{name}]"][row])


def _amount(r, name, row):
    return float(r.as_roadrunner(selections=[name])[name][row])


@pytest.mark.parametrize("method", ["ode", "ssa"])
@pytest.mark.parametrize(("g", "s", "Y"), [(1, 2, 1.0), (2, 1, 2.0)])
def test_a_same_instant_assignment_and_resize(method, g, s, Y):
    """Set then grow: [Y] = 2 in a volume of 1, an amount of 2, then [Y] = 1 in a
    volume of 2. Grow then set: [Y] = 2."""
    m = bngsim.Model.from_antimony_string(BATCH.format(g=g, s=s))
    r = bngsim.Simulator(m, method=method).run(t_span=(0, 4), n_points=5, seed=1)
    assert _conc(r, "Y", 3) == pytest.approx(Y, rel=1e-9)


def test_the_sensitivity_through_that_batch():
    """Set then grow: [Y] = b·p/2, so d/db = p/2 and d/dp = b/2."""
    m = bngsim.Model.from_antimony_string(BATCH.format(g=1, s=2))
    r = bngsim.Simulator(m, sensitivity_params=["b", "p"]).run(t_span=(0, 4), n_points=5)
    np.testing.assert_allclose(
        np.ravel(np.asarray(r.output_sensitivities("species:Y"))[3]), [0.5, 1.0], rtol=1e-5
    )


@pytest.mark.parametrize(
    ("text", "conc", "amount"),
    [
        # Resized to 2 at 1, then S = 50: an amount of 100.
        (
            "compartment C = 1; species S in C = 10; E1: at time > 1: C = 2;"
            " E2: at time > 2: S = 50;",
            50.0,
            100.0,
        ),
        # C' = 1, so C(2) = 3: S = 50 is an amount of 150, and [S](3) = 150/4.
        (
            "compartment C = 1; C' = 1; species S in C = 10; E2: at time > 2: S = 50;",
            37.5,
            150.0,
        ),
    ],
)
@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_an_assignment_in_a_resized_compartment(text, conc, amount, method):
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method=method).run(
        t_span=(0, 4), n_points=5, seed=1
    )
    assert _conc(r, "S", 3) == pytest.approx(conc, rel=1e-6)
    assert _amount(r, "S", 3) == pytest.approx(amount, rel=1e-6)


def test_a_delayed_assignment_beside_its_own_resize_keeps_its_frozen_value():
    """``Y = X`` is frozen at the trigger (X(2) = 2) and applied at 3 with the
    resize to 2: [Y](3) = 2·1/2 = 1, not X(3)/2."""
    text = (
        "compartment C = 1; species X in C = 0; species Y in C = 0; J: => X; 1;"
        " E: at 1 after (time >= 2): C = 2, Y = X;"
    )
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(text)).run(
        t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12
    )
    assert _conc(r, "Y", 3) == pytest.approx(1.0, rel=1e-6)
