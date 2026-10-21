"""Issue #1059 — an SSA run from a large t_start keeps its waiting times.

Each firing used to be placed at ``t + tau`` in doubles. From t_start = 1e12 the
doubles near t are 1.2e-4 apart, the first waiting times of a decay from 1e5
copies are about 2e-5, and they rounded away: the run fired 1.6 times too many
events and came back at z = -287 with no error. The clock is now an origin plus
the elapsed time, so a waiting time is added at the scale of the run's span.
From t_start = 0 the two are the same double and the run is what it was.
"""

from __future__ import annotations

import contextlib
import io
import math

import bngsim
import numpy as np
import pytest

# A(s~0) -> A(s~1) at k = 0.5 from 1e5 copies; the mean at t_start + 2 is 1e5/e.
NET = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A0() 100000
    2 A1() 0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 A0obs 1
end groups
"""

SEEDS = range(1, 11)


def _counts(tmp_path, t0: float, **sim) -> tuple[list[float], str]:
    p = tmp_path / "decay.net"
    p.write_text(NET)
    vals, backend = [], ""
    for seed in SEEDS:
        with contextlib.redirect_stderr(io.StringIO()):
            m = bngsim.Model.from_net(str(p))
            r = bngsim.Simulator(m, method="ssa", **sim).run(
                sample_times=[t0, t0 + 2.0], seed=seed
            )
        vals.append(float(r.observables["A0obs"][1]))
        backend = r.ssa_diagnostics["propensity_backend"]
    return vals, backend


# The default run takes the compiled recompute-all loop where a compiler (or MIR)
# is present; codegen=False always takes the interpreted main loop. Both place
# firings on the clock.
LOOPS = [({}, None), ({"codegen": False}, "interpreted")]


@pytest.mark.parametrize(("sim", "backend"), LOOPS, ids=["default-loop", "main-loop"])
@pytest.mark.parametrize("t0", [0.0, 1e9, 1e12, 1e13])
def test_a_decay_from_a_far_start_matches_its_closed_form(tmp_path, sim, backend, t0):
    vals, ran = _counts(tmp_path, t0, **sim)
    assert backend is None or ran == backend
    p = math.exp(-1)
    z = (np.mean(vals) - 1e5 * p) / math.sqrt(1e5 * p * (1 - p) / len(vals))
    assert abs(z) < 5, (t0, z)


def test_the_start_time_does_not_change_a_run(tmp_path):
    """The same seeds give the same counts whether the run starts at 0 or 1e9."""
    assert _counts(tmp_path, 1e9)[0] == _counts(tmp_path, 0.0)[0]
