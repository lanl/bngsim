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


# Each against libRoadRunner 2.9.2 on the same model: [Y] after the events.
CASES = {
    # The new size reads what an earlier fire of the instant changed; the
    # compartment takes the size frozen at the trigger.
    "new_size_reads_a_parameter_set_earlier": (
        "compartment Cc = 1; species Y in Cc; Y = 1; q = 2;"
        " Eset: at (time >= 2), priority = 2: q = 4; Egrow: at (time >= 2), priority = 1: Cc = q",
        0.5,
    ),
    "two_resizes_the_second_reading_the_size": (
        "compartment Cc = 1; species Y in Cc; Y = 1;"
        " E1: at (time >= 2), priority = 2: Cc = 2;"
        " E2: at (time >= 2), priority = 1: Cc = 2*Cc",
        0.5,
    ),
    "a_rule_sized_compartment": (
        "compartment D; D := k; k = 1; q = 2; species Y in D = 1;"
        " E1: at time >= 2, priority = 2: q = 3; E2: at time >= 2, priority = 1: k = q",
        0.5,
    ),
    "new_size_reads_the_species_assigned_earlier": (
        "compartment Cc = 1; species Y in Cc; Y = 1;"
        " Eset: at (time >= 2), priority = 2: Y = 4; Egrow: at (time >= 2), priority = 1: Cc = Y",
        4.0,
    ),
    "an_own_assignment_after_an_earlier_resize": (
        "compartment Cc = 1; species Y in Cc; Y = 1;"
        " E1: at (time >= 2), priority = 2: Cc = 2;"
        " E2: at (time >= 2), priority = 1: Y = 5, Cc = 4",
        2.5,
    ),
    # Delayed: the amount at the apply, the sizes frozen at the trigger.
    "delayed_own_assignment_with_the_size_changed_meanwhile": (
        "compartment Cc = 1; species Y in Cc; Y = 1;"
        " E1: at 1 after (time >= 1): Y = 5, Cc = 4; E2: at (time >= 1.5): Cc = 2",
        2.5,
    ),
    "delayed_new_size_reads_a_parameter_changed_meanwhile": (
        "compartment Cc = 1; species Y in Cc; Y = 1; q = 2;"
        " E1: at 1 after (time >= 1): Cc = q; E2: at (time >= 1.5): q = 4",
        0.5,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_against_libroadrunner(name):
    text, want = CASES[name]
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(text)).run(
        sample_times=[0, 1, 2.25, 3], rtol=1e-10, atol=1e-12
    )
    assert _conc(r, "Y", 2) == pytest.approx(want, rel=1e-6)


def test_a_dividing_cell_keeps_its_amounts():
    """``Cc' = 0.5``, and half a time unit after reaching 2 the cell divides:
    Y, which the division does not assign, keeps its amount of 4."""
    text = (
        "compartment Cc = 1; Cc' = 0.5; species Y in Cc = 4; species S in Cc = 8;"
        " E: at 0.5 after (Cc >= 2): Cc = Cc/2, S = S/2;"
    )
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(text)).run(
        t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12
    )
    assert _amount(r, "Y", 4) == pytest.approx(4.0, rel=1e-6)


REFUSED = {
    # The second resize reads a size an earlier fire of the instant changed.
    "two_resizes": (
        "compartment Cc = 1; species Y in Cc; Y = 1; k = 1; J: Y => ; k*Y;"
        " E1: at (time >= 2), priority = 2: Cc = 2;"
        " E2: at (time >= 2), priority = 1: Cc = 2*Cc",
        "k",
    ),
    # An earlier fire writes what the new size reads.
    "new_size_reads_a_parameter_written_earlier": (
        "compartment C = 1; species Y in C = 1; species X in C = 0; p = 1; q = 2; k = 0.3;"
        " J: Y => X; k*Y; E1: at time >= 2, priority = 2: q = 2*p;"
        " E2: at time >= 2, priority = 1: C = q",
        "p",
    ),
    # A reset to the value it holds moves no value but does move the
    # derivative, which a comparison of values cannot see.
    "a_reset_to_the_value_held": (
        "compartment C = 2; species Y in C = 1; q0 = 1; q = 1; k = 0.3; J: Y => ; k*Y;"
        " E1: at time >= 2, priority = 2: q = q0; E2: at time >= 2, priority = 1: C = q",
        "q0",
    ),
    # An own assignment after an earlier fire moved the size (main got this
    # one silently wrong too).
    "an_own_assignment_after_a_resize": (
        "compartment C = 1; species Y in C = 1; p = 0.5; k = 0.3; J: Y => ; k*Y;"
        " E1: at time >= 2, priority = 2: C = 2*p;"
        " E2: at time >= 2, priority = 1: Y = 5, C = 4",
        "p",
    ),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_the_sensitivity_through_a_composition_it_cannot_differentiate_is_refused(name):
    """The values are right; their derivative through such a batch is refused,
    not approximated."""
    text, p = REFUSED[name]
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), sensitivity_params=[p])
    with pytest.raises(Exception, match="issue #936"):
        sim.run(t_span=(0, 4), n_points=5)


def test_a_resize_to_zero_with_an_amount_in_it_is_refused():
    text = (
        "compartment C = 1; species Y in C = 1;"
        " E1: at (time >= 2), priority = 2: C = 0; E2: at (time >= 2), priority = 1: C = 2"
    )
    with pytest.raises(Exception, match="resizes a compartment to 0"):
        bngsim.Simulator(bngsim.Model.from_antimony_string(text)).run(t_span=(0, 4), n_points=5)


@pytest.mark.parametrize(
    ("text", "conc"),
    [
        # Issue #740: frozen at the trigger (t = 1, X = 1), applied at 2; the
        # amount 10 is kept, so [S] = 10/2.
        (
            "compartment C = 1; species S in C = 10; X = 0; X' = 1;"
            " E: at 1 after (time >= 1): C = 1 + X;",
            5.0,
        ),
        (
            "compartment C; C := k; k = 1; species S in C = 10; X = 0; X' = 1;"
            " E: at 1 after (time >= 1): k = 1 + X;",
            5.0,
        ),
        # S = X(1) = 1 in the old volume, then C = 4.
        (
            "compartment C = 1; species S in C = 10; X = 0; X' = 1;"
            " E: at 1 after (time >= 1): C = 4, S = X;",
            0.25,
        ),
    ],
)
def test_a_delayed_resize_uses_its_trigger_time_size(text, conc):
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(text)).run(
        t_span=(0, 3), n_points=4, rtol=1e-10, atol=1e-12
    )
    assert _conc(r, "S", 3) == pytest.approx(conc, rel=1e-6)
