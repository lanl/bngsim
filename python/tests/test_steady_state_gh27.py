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

# Published RuleHub .net files the bug report was found with. These are the
# copies VENDORED (and git-tracked) under benchmarks/models/net/ode/ — byte
# identical to the generated benchmarks/suites/ode_fullnet/nets/ corpus these
# tests used to read, but that corpus is a build artifact: it is untracked and
# exists only in a checkout that has run the ode_fullnet suite. Pointed there,
# all four tests skipped silently in any fresh clone or git worktree — the
# regression guards for GH #27 were effectively not running in CI-like trees.
_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"

_GARDNER = "genetic_switch.net"  # Gardner 2000 genetic toggle
_KINETIC = "kinetic_proofreading.net"  # Hlavacek 2001
_KOCIEN = "Kocieniewski_2012.net"
_BARUA13 = "Barua_2013.net"


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        # No benchmark tree at all (e.g. testing an installed wheel) — the only
        # legitimate reason to skip.
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    # The tree is here, so a missing file means the corpus moved or lost a
    # model. Fail loudly rather than skipping a regression guard into silence.
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
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
