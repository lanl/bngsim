"""Issue #54's stall fixture, written so that it still wedges.

``tests/data/switch_discontinuity_stall.net`` switches a rate law on when a
counter species reaches ``sigma``. As written it no longer stalls: bngsim works
out that the counter reaches ``sigma`` at t = sigma and stops the step exactly
there (issue #443). With that stop stood down, the counter is pinned an ulp
short of ``sigma``, and a run pinned on a state-switch surface has the species
the threshold reads moved the few ulp that put it across (issue #928).

So the condition is written on the time itself, ``time() >= sigma``, and the
crossing stop is stood down by emptying the conditions the model derived. There
is no species to move, and the step size collapses at the rate jump. Emptying
the derived conditions is how a model whose crossing time cannot be resolved
reaches the integrator, which is the population the bounded retry covers.
"""

from __future__ import annotations

from pathlib import Path

import bngsim

STALL_NET = (
    Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "data"
    / "switch_discontinuity_stall.net"
)


def stalling_model(tmp_path):
    assert STALL_NET.exists(), f"test data not found: {STALL_NET}"
    text = STALL_NET.read_text()
    assert text.count("if(t>=sigma,") == 2
    path = tmp_path / "stall_on_time.net"
    path.write_text(text.replace("if(t>=sigma,", "if(time()>=sigma,"))
    model = bngsim.Model.from_net(str(path))
    assert model.time_discontinuity_conditions(), (
        "the fixture no longer derives a crossing at all, so standing it down "
        "proves nothing; check what changed in the scan"
    )
    model._derived_time_disc_conditions = ()
    return model
