"""T4 — colored finite-difference sparse-Jacobian scratch buffers.

cvode_colored_jac (the sparse FD-Jacobian fallback, used when a sparse model has
no complete analytical Jacobian) heap-allocated three length-n_species vectors
on every Jacobian evaluation. They are now persisted in CvodeUserData and sized
once when the colored-FD callback is selected, so the per-eval hot path
allocates nothing. The computation is unchanged — same perturbations, same
finite differences — so results are byte-identical.

The colored-FD path is reached by forcing ``jacobian="fd"`` on a sparse model
(n_species ≥ the sparse threshold): that skips the analytical Jacobian and
selects the colored-FD callback. These tests confirm it runs correctly (matches
the analytical Jacobian to FD tolerance) and that the persisted buffers behave
correctly across repeated runs on one simulator (warm reuse).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Simulator

# egfr_net: 356 species → sparse linear solver → colored-FD Jacobian under
# jacobian="fd". Lives in the benchmarks corpus; skip cleanly if absent.
_NET = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode" / "egfr_net.net"
)
needs_net = pytest.mark.skipif(not _NET.exists(), reason="egfr_net.net not present")


@needs_net
def test_colored_fd_matches_analytical() -> None:
    """Forced colored-FD Jacobian matches the analytical Jacobian to FD tol."""
    m = Model.from_net(str(_NET))
    assert m._core.n_species >= 50  # large enough to take the sparse solver path

    r_fd = Simulator(m, method="ode", jacobian="fd").run(t_span=(0, 100), n_points=51)
    r_auto = Simulator(Model.from_net(str(_NET)), method="ode", jacobian="auto").run(
        t_span=(0, 100), n_points=51
    )

    # Colored finite differences vs the analytical Jacobian: same trajectory to
    # within FD accuracy (the integrator is robust to small Jacobian error).
    assert np.allclose(r_fd.species, r_auto.species, rtol=1e-6, atol=1e-6)


@needs_net
def test_colored_fd_warm_reuse_identical() -> None:
    """Repeated runs on one simulator reuse the persisted buffers and agree."""
    m = Model.from_net(str(_NET))
    sim = Simulator(m, method="ode", jacobian="fd")
    r1 = sim.run(t_span=(0, 100), n_points=51)
    m.reset()  # restore initial concentrations; replay from the same state
    r2 = sim.run(t_span=(0, 100), n_points=51)
    # Persisted colored-FD scratch reused across runs → byte-identical replay.
    assert np.array_equal(r1.species, r2.species)
    assert np.array_equal(r1.observables, r2.observables)
