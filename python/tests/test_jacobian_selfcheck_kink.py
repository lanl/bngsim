"""Issue #511: the analytical-Jacobian FD self-check must not reject a correct Jacobian
because the initial state sits on a switching surface of a piecewise rate law.

``NetworkModel::set_functional_jacobian`` (``src/model.cpp``) validates the freshly
attached analytical Jacobian against reliability-gated central finite differences at
the initial state and three spread-out probe states, and drops the whole Jacobian on
the first trustworthy mismatch. A rate law such as ``if(V > 0, V*k, 0)`` switches on
a quantity that is exactly 0 at the seed state whenever the species it compares are
still at their seed values, so the initial probe sits ON the switching surface: the
forward probe evaluates one branch, the backward probe the other, the central
difference is the mean of the two one-sided slopes — Richardson-converged, since both
step sizes straddle the same surface — and the analytical value, the slope of the
branch the point itself is on, is flagged as a mismatch. Five corpus models (a wave
equation, a Schrödinger discretization, BIOMD0000000075/161/613) ran on the
finite-difference Jacobian for their whole integration that way.

The fix judges an entry only where a derivative exists: the two one-sided differences
(free, given the unperturbed RHS the gate already evaluates) must agree, or the entry
is a switching point with no finite-difference reference at that probe. The other
probe states do not sit on the seed's surfaces and judge the entry there — so a
Jacobian that is wrong on either branch is still caught, as the negative controls
below pin.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _jacobian as J

FIXTURE = "jac_selfcheck_switching_surface.net"


def _clear_selfcheck_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run against the DEFAULT gate: no env overrides may relax or bypass it."""
    for var in (
        "BNGSIM_JAC_SELFCHECK_DENSE_MAX",
        "BNGSIM_JAC_SELFCHECK_SAMPLE",
        "BNGSIM_JAC_NO_SELFCHECK",
    ):
        monkeypatch.delenv(var, raising=False)


def _derived_terms(core):
    """The derived terms exactly as attach_functional_jacobian hands them over: a .net
    functional rate with reactants goes through the per-observable path (the C++ side
    applies the mass-action product rule), a reactant-free one through the per-species
    path."""
    ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    obs_groups = {n: [(int(s), float(f)) for s, f in g] for n, g in ctx["observables"]}
    obs_idx = {n: i for i, (n, _g) in enumerate(ctx["observables"])}
    smeta = {i: (bool(a), float(v)) for i, (a, v) in enumerate(ctx["species_meta"])}
    consts = set(ctx["constant_names"])
    terms = []
    for rxn in ctx["functional_reactions"]:
        if rxn["apply_species_factor"] and len(rxn["reactant_idx0"]) > 0:
            t = J.build_per_observable_terms(rxn["rate_expr"], func_map, set(obs_groups), consts)
            assert t is not None, rxn["rate_expr"]
            terms.append((rxn["rxn_idx"], True, [(obs_idx[n], e) for n, e in t]))
        else:
            t = J.build_per_species_terms(rxn["rate_expr"], func_map, obs_groups, smeta, consts)
            assert t is not None, rxn["rate_expr"]
            terms.append((rxn["rxn_idx"], False, [(int(j), e) for j, e in t]))
    return terms


def _corrupt(terms, rxn_idx):
    """Double every derivative of reaction ``rxn_idx`` — a 100% divergence on each branch."""
    return [
        (ri, po, [(j, f"2.0*({e})") for j, e in dl]) if ri == rxn_idx else (ri, po, dl)
        for ri, po, dl in terms
    ]


def _central(m: bngsim.Model, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    n = y.size
    D = np.zeros((n, n))
    for j in range(n):
        s = h * max(abs(y[j]), 1.0)
        yp, ym = y.copy(), y.copy()
        yp[j] += s
        ym[j] -= s
        D[:, j] = (np.asarray(m.rhs(yp)) - np.asarray(m.rhs(ym))) / (2.0 * s)
    return D


# ─── The regression ──────────────────────────────────────────────────────────


def test_switching_surface_fixture_attaches_under_default_gate(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Before the fix this was False: the central difference at the seed state read
    k/2 for an entry whose analytical value, 0, is the else-branch slope."""
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True
    assert m._core.analytical_jacobian_complete is True


def test_seed_state_is_on_the_surface_and_the_analytical_value_is_the_branch_slope(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The mechanism, pinned: at y0 the one-sided slopes of d(f_P2)/dP2 are k·P1 (up,
    into the ``if`` branch) and 0 (down, staying on ``else``), so the central
    difference is k·P1/2 while the analytical entry is 0 — and 0 is right."""
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True
    names = list(m.species_names)
    iP2 = next(i for i, n in enumerate(names) if n.startswith("P2"))
    y0 = np.asarray(m.get_state(), float)
    k, P1 = 0.5, 1.0
    h = 1e-6
    up, dn = y0.copy(), y0.copy()
    up[iP2] += h
    dn[iP2] -= h
    f0, fu, fd = (np.asarray(m.rhs(v), float)[iP2] for v in (y0, up, dn))
    assert (fu - f0) / h == pytest.approx(k * P1, rel=1e-6)  # forward: the if branch
    assert (f0 - fd) / h == pytest.approx(0.0, abs=1e-12)  # backward: still else
    assert (fu - fd) / (2 * h) == pytest.approx(k * P1 / 2, rel=1e-6)  # the old oracle
    Jy0 = m.jacobian(y0)
    assert Jy0.source == "analytical"
    assert Jy0[iP2, iP2] == pytest.approx(0.0, abs=1e-12)


def test_attached_jacobian_matches_finite_differences_off_the_surface(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Away from V = 0 the derivative exists and the accepted Jacobian is exact to FD
    precision on both branches."""
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True
    names = list(m.species_names)
    iP1 = next(i for i, n in enumerate(names) if n.startswith("P1"))
    iP2 = next(i for i, n in enumerate(names) if n.startswith("P2"))
    rng = np.random.default_rng(511)
    seen_branches = set()
    for _ in range(8):
        y = rng.uniform(0.1, 2.0, len(names))
        seen_branches.add(y[iP2] > y[iP1])
        Jy = m.jacobian(y)
        assert Jy.source == "analytical"
        np.testing.assert_allclose(np.asarray(Jy), _central(m, y), rtol=1e-6, atol=1e-9)
    assert seen_branches == {True, False}, "both branches of the if() should be sampled"


# ─── Negative controls: the gate is not blinded ─────────────────────────────


def test_wrong_smooth_derivative_is_still_rejected(data_dir: Path, monkeypatch):
    """A 2× error on the smooth reaction's derivatives is judged at every probe."""
    _clear_selfcheck_env(monkeypatch)
    core = bngsim.Model.from_net(str(data_dir / FIXTURE))._core
    terms = _derived_terms(core)
    smooth = next(ri for ri, po, _dl in terms if not po)  # the reactant-free source of Q
    assert core.set_functional_jacobian(_corrupt(terms, smooth)) is False
    assert core.analytical_jacobian_complete is False


def test_wrong_branch_derivative_is_still_rejected_off_the_surface(data_dir: Path, monkeypatch):
    """A 2× error on the piecewise reaction's derivatives cannot be judged at the seed
    state (V = 0, no derivative there) but is caught at the spread-out probes, where
    V ≠ 0 and the ``if`` branch's slope is a real finite-difference reference."""
    _clear_selfcheck_env(monkeypatch)
    core = bngsim.Model.from_net(str(data_dir / FIXTURE))._core
    terms = _derived_terms(core)
    switch = next(ri for ri, po, _dl in terms if po)  # P1 -> P2 at the piecewise rate
    assert any("if" in e for _ri, po, dl in terms if po for _k, e in dl), terms
    assert core.set_functional_jacobian(_corrupt(terms, switch)) is False
    assert core.analytical_jacobian_complete is False


def test_correct_terms_attach_through_the_same_path(data_dir: Path, monkeypatch):
    _clear_selfcheck_env(monkeypatch)
    core = bngsim.Model.from_net(str(data_dir / FIXTURE))._core
    terms = _derived_terms(core)
    assert {po for _ri, po, _dl in terms} == {True, False}, terms  # both paths exercised
    assert core.set_functional_jacobian(terms) is True
    assert core.analytical_jacobian_complete is True


# ─── A rejection reaches the logger (issue #506's self-check path) ───────────

# k·sqrt(A) with A = 0 at the seed: the RHS is a finite 0 there but ∂/∂A = k/(2·sqrt(A))
# is not, which the gate rightly refuses (the Newton solve would see the inf).
_NET_SQRT_AT_ZERO = """\
begin parameters
    1 k 0.7
end parameters
begin species
    1 A() 0.0
    2 B() 1.0
end species
begin functions
    1 fRoot() k*sqrt(A_tot)
end functions
begin reactions
    1 0 2 fRoot #source_B
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
end groups
"""


def test_attach_decline_is_logged(tmp_path: Path, monkeypatch, caplog):
    _clear_selfcheck_env(monkeypatch)
    p = tmp_path / "sqrt_at_zero.net"
    p.write_text(_NET_SQRT_AT_ZERO)
    with caplog.at_level(logging.INFO, logger="bngsim"):
        m = bngsim.Model.from_net(str(p))
        assert m.prepare_analytical_jacobian() is False
    assert any("C++ attach declined" in r.getMessage() for r in caplog.records), caplog.text
