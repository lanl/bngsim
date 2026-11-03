"""A rounding law's answer does not depend on the output grid (issue #1342).

Two models #1306's generator drew (PR #1317), each shrunk to one or two
reactions: a law over ``floor`` and ``round`` of the state, and one over
``trunc`` of a species sliding up to 1 beside an ``if`` on the clock. At rtol
1e-8 their answer moved with how many output points were asked for, by up to
1e3 times the tolerance, and the second failed with CV_CONV_FAILURE at rtol
1e-12 on one grid and ran on another. Both hold to their tolerance on both
grids now; these pin that.

Oracle: a converged run (rtol 1e-12; 1e-11 for the second, as the issue had
it), read at the same 21 times; the error is the largest over them, as a part
of the trajectory's peak.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

# The issue's laws, verbatim, split only to fit a line.
_LAW_FLOOR = (
    "k1*((floor((O1 + O1)) - (round(O2) - O1)))^2/(1 + ((floor((O1 + O1)) - (round(O2) - O1)))^2)"
)
_LAW_TRUNC = "k1*((exp(-(time())^2) - trunc(O2)))^2/(1 + ((exp(-(time())^2) - trunc(O2)))^2)"
_LAW_CLOCK = (
    "k2*(if(time() > 1.0, O1, log(1 + (1.0)^2)))^2"
    "/(1 + (if(time() > 1.0, O1, log(1 + (1.0)^2)))^2)"
)

_FLOOR_ROUND = f"""\
begin parameters
    1 k1 0.811
end parameters
begin species
    1 S1() 2.048
    2 S2() 2.948
end species
begin functions
    1 f1() {_LAW_FLOOR}
end functions
begin reactions
    1 1 0 f1
end reactions
begin groups
    1 O1 1
    2 O2 2
end groups
"""

_TRUNC_CLOCK = f"""\
begin parameters
    1 k1 0.23
    2 k2 1.735
end parameters
begin species
    1 S1() 4.925
    2 S2() 2.735
end species
begin functions
    1 f1() {_LAW_TRUNC}
    2 f2() {_LAW_CLOCK}
end functions
begin reactions
    1 2 1 f1
    2 0 1 f2
end reactions
begin groups
    1 O1 1
    2 O2 2
end groups
"""

_CASES = {
    "floor and round": (_FLOOR_ROUND, 1e-12),
    "trunc beside the clock": (_TRUNC_CLOCK, 1e-11),
}


def _run(tmp_path, text: str, n: int, rtol: float) -> np.ndarray:
    path = tmp_path / "m.net"
    path.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
            t_span=(0, 10), n_points=n, rtol=rtol, atol=rtol * 1e-2
        )
    return np.asarray(r.species)


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("n", [21, 161])
def test_the_answer_holds_to_its_tolerance_on_either_grid(tmp_path, case, n):
    """At rtol 1e-8: 4.9e-5 and 8.4e-3 of the peak on 161 points when filed."""
    text, ref_rtol = _CASES[case]
    ref = _run(tmp_path, text, 21, ref_rtol)
    y = _run(tmp_path, text, n, 1e-8)[:: (n - 1) // 20]
    assert np.max(np.abs(y - ref)) / np.max(np.abs(ref)) < 1e-6


def test_the_trunc_model_runs_at_a_tight_tolerance_on_either_grid(tmp_path):
    """It failed with CV_CONV_FAILURE at t = 8.5 at rtol 1e-12 on 21 points."""
    for n in (21, 161):
        assert np.all(np.isfinite(_run(tmp_path, _TRUNC_CLOCK, n, 1e-12)))
