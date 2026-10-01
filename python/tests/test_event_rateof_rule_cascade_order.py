"""An event triggered through a rule over rateOf by another event's assignment.

SBML L3 §3.4: when one event's assignment makes another event's trigger true,
the new event joins the fires pending at that instant and they execute in
priority order. After each fire bngsim re-reads every trigger to find such
rises. That re-read evaluated the model's functions and then refreshed the
``rateOf`` buffer, so a trigger that reads ``rateOf`` through an assignment rule
(``r := rateOf(A)``) was read against the rates from before the assignment. The
rise was missed there and only found once the whole batch was done, so the
event fired last whatever its priority. Found while fixing issue #910, which is
the same read one step behind in the pass that confirms a located root.

Here E1 (priority 3) raises A's decay constant when A < 2, which takes
``rateOf(A)`` from −1 to −20. E2 (priority 2) triggers on ``r < −10`` and sets
Y = 5. E3 (priority 1) triggers with E1 and copies Y into Z, reading Y when it
executes. In SBML order (E1, E2, E3) Z = 5; with E2 last, Z = 0.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

MODEL = (
    "species A, Y, Z; A = 10; Y = 0; Z = 0; k = 0.5\n"
    "J0: A -> ; k*A\n{rule}"
    "E1: at (A < 2), priority = 3: k = 10\n"
    "E2: at ({r} < -10), priority = 2: Y = 5\n"
    "E3: at (A < 2), priority = 1, fromTrigger = false: Z = Y\n"
)


@pytest.mark.parametrize(
    ("rule", "r"),
    [("r := rateOf(A)\n", "r"), ("", "rateOf(A)")],
    ids=["through-a-rule", "written-out"],
)
def test_an_event_raised_by_an_assignment_fires_in_priority_order(rule, r):
    """The written-out trigger reads the buffer itself and was always right."""
    model = bngsim.Model.from_antimony_string(MODEL.format(rule=rule, r=r))
    run = bngsim.Simulator(model, method="ode").run(
        sample_times=[0.0, 2.0, 4.0, 6.0], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)[-1]
    assert x[names.index("Y")] == 5.0
    assert x[names.index("Z")] == 5.0
