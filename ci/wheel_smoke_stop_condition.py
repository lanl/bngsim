"""Wheel smoke test: a stop condition that meets 0/0 runs, on every CPython built.

Issue #1556. A stop condition is evaluated with no builtins, and on Python 3.13
and later a warning numpy issues there raises ``KeyError: '__import__'``. The
pull-request suites run Python 3.12 only, where the bug does not show, and the
wheels are the one place every CPython bngsim supports is exercised. So this
runs beside ``wheel_smoke_klu.py`` in cibuildwheel's ``test-command``.

A() decays from 100 at k = 0.1, so ``A_tot < 10`` first holds at t = 24 on a unit
grid. ``(A_tot < 10) * (t / t)`` is 0/0 = NaN at t = 0, which is not met, and the
same condition after: it must stop at t = 24, with one RuntimeWarning that names
it and comes from outside the condition (``<stop_condition>``).

Self-contained like the KLU smoke test: bngsim and numpy only, no compiler, no
data files. Run: ``python ci/wheel_smoke_stop_condition.py``.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import bngsim

NET = (
    "begin parameters\n 1 k 0.1\nend parameters\n"
    "begin species\n 1 A() 100\nend species\n"
    "begin reactions\n 1 1 0 k\nend reactions\n"
    "begin groups\n 1 A_tot 1\nend groups\n"
)
CONDITION = "(A_tot < 10) * (t / t)"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decay.net"
        path.write_text(NET)
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    sim.add_stop_condition(CONDITION)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sim.run(t_span=(0, 100), n_points=101)
            stop = None
        except bngsim.StopConditionMet as met:
            stop = float(met.result.time[-1])
    told = [w for w in caught if repr(CONDITION) in str(w.message)]
    inside = [w for w in caught if w.filename == "<stop_condition>"]
    print(f"Python {sys.version.split()[0]}, bngsim {bngsim.__version__}: stop at {stop}")
    for w in caught:
        print(f"  {w.category.__name__} from {w.filename}: {w.message}")
    if stop != 24.0 or len(told) != 1 or inside:
        print("FAIL: expected a stop at t = 24 and one warning from outside the condition")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
