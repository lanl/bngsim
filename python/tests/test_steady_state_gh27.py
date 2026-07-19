"""Regression tests for GH #27 — steady-state (KINSOL) wrong/NaN roots.

The default ``method="newton"`` used to seed KINSOL at the raw initial
condition and only fall back to integration on non-*convergence*. That
returned:

* Bug 1 — a ``NaN`` residual reported ``converged=True`` (``NaN >= tol`` is
  false), so ``steady_state(method="newton")`` returned ``conc=[nan, …]``
  (Gardner 2000 genetic toggle: Newton walks a species negative, Hill laws
  give NaN).
* Bug 2 — Newton converged to a spurious root of ``f(y)=0`` the dynamics
  never reach, so the fallback never fired (Hlavacek 2001 kinetic
  proofreading: 52% off the physical state; Kocieniewski 2012).
* Bug 3 — the dense KINSOL linear-solver setup failed unrecoverably at ~400
  species (Barua 2013, 409 sp), spamming stderr before falling back.

The two-tier solver (integrate FIRST into the physical basin, then accept a
KINSOL polish only once it is seed-stable) fixes all three. These tests assert
the observable contract: ``method="newton"`` agrees with the physical steady
state from ``method="integration"`` and is always finite.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# Published RuleHub .net files the bug report was found with. Skip cleanly when
# running against an installed wheel without the benchmark tree.
_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "suites" / "ode_fullnet" / "nets"

_GARDNER = "slow__rulehub__Published__Gardner2000__genetic_switch_gardner2000.bngl.net"
_KINETIC = "fast__rulehub__Published__Hlavacek2001__kinetic_proofreading_hlavacek2001.bngl.net"
_KOCIEN = "slow__rulehub__Published__Kocieniewski2012__Kocieniewski_2012.bngl.net"
_BARUA13 = "slow__rulehub__Published__Barua2013__Barua_2013__PATCHED.bngl.net"


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not path.is_file():
        pytest.skip(f"published net not available: {path}")
    return str(path)


def _physical_ss(net: str) -> np.ndarray:
    """Physical steady state: CVODE integrated to the parity tolerance."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    r = sim.steady_state(method="integration", tol=1e-9, max_time=1e8, max_steps=500_000)
    return np.asarray(r.concentrations, dtype=float)


def _max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / (1e-6 + 1e-2 * np.abs(b))) * 1e-2)


class TestGH27WrongRoots:
    def test_bug1_gardner_no_nan_converged(self):
        """Gardner toggle: newton must NOT return a NaN result labeled converged."""
        net = _net(_GARDNER)
        r = bngsim.Simulator(bngsim.Model.from_net(net), method="ode").steady_state(
            method="newton"
        )
        conc = np.asarray(r.concentrations, dtype=float)
        # The core Bug 1 defect: converged + all-NaN concentrations.
        assert np.all(np.isfinite(conc)), "newton returned non-finite concentrations"
        assert r.converged
        assert np.isfinite(r.residual)
        # And it is the physical steady state.
        assert _max_rel_err(conc, _physical_ss(net)) < 1e-2

    def test_bug2_kinetic_proofreading_physical_root(self):
        """Kinetic proofreading: newton must reach the physical root, not a
        spurious f(y)=0 root (was ~52% off at default parameters)."""
        net = _net(_KINETIC)
        r = bngsim.Simulator(bngsim.Model.from_net(net), method="ode").steady_state(
            method="newton"
        )
        conc = np.asarray(r.concentrations, dtype=float)
        assert np.all(np.isfinite(conc))
        assert _max_rel_err(conc, _physical_ss(net)) < 1e-2

    def test_bug2_kocieniewski_physical_root(self):
        """Kocieniewski 2012 (85 sp): another multi-root model that used to
        return spurious roots on most scanned doses."""
        net = _net(_KOCIEN)
        r = bngsim.Simulator(bngsim.Model.from_net(net), method="ode").steady_state(
            method="newton"
        )
        conc = np.asarray(r.concentrations, dtype=float)
        assert np.all(np.isfinite(conc))
        assert _max_rel_err(conc, _physical_ss(net)) < 1e-2

    def test_bug3_barua2013_graceful(self):
        """Barua 2013 (409 sp): the dense KINSOL setup fails (structurally
        singular reduced Jacobian). The solver must fall back to integration
        and return a finite result — no crash, no NaN, no wrong root."""
        net = _net(_BARUA13)
        r = bngsim.Simulator(bngsim.Model.from_net(net), method="ode").steady_state(
            method="newton"
        )
        conc = np.asarray(r.concentrations, dtype=float)
        assert np.all(np.isfinite(conc))
        # KINSOL is not viable here, so the reported method is integration.
        assert r.method_used == "integration"
        # Matches a plain integration solve (same physical / plateau state).
        ss_int = bngsim.Simulator(bngsim.Model.from_net(net), method="ode").steady_state(
            method="integration"
        )
        np.testing.assert_allclose(conc, np.asarray(ss_int.concentrations, dtype=float), rtol=1e-3)
