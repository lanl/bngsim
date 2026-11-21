"""Wheel smoke test: the example models are in the wheel and match their exact solutions.

Issue #1706. ``bngsim.examples`` reads its model files from the installed
package, so they are only there if the wheel carries them. This runs against
each built wheel, in cibuildwheel's ``test-command``, with no extras: each
model in each format that needs none (net, sbml, python) must match its exact
solution. Run: ``python ci/wheel_smoke_examples.py``.
"""

from __future__ import annotations

import bngsim
import numpy as np

E = bngsim.examples
checked = 0
for name in E.names():
    for fmt in E.formats(name):
        assert E.path(name, fmt).is_file(), (name, fmt)
    if name == "schlogl":
        ss = bngsim.Simulator(E.load(name)).steady_state()
        assert abs(ss["X"] - 100) < 1e-4, ss["X"]
        continue
    for fmt in ("net", "sbml", "python"):
        if fmt not in E.formats(name):
            continue
        r = bngsim.Simulator(E.load(name, fmt)).run(t_span=(0, 10), n_points=11)
        for obs, values in E.exact(name, r.time).items():
            assert np.allclose(r.observables[obs], values, rtol=1e-4), (name, fmt, obs)
        checked += 1
print(
    f"examples: {len(E.names())} models in the wheel, {checked} runs match their exact solutions"
)
