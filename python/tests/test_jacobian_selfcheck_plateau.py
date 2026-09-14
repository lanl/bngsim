"""Issue #533: the analytical-Jacobian FD self-check must not mistake a
double-precision rounding plateau for a converged slope.

The gate (``NetworkModel::set_functional_jacobian``, ``src/model.cpp``) judged an
entry once its central differences at ``1e-5·|y|`` and ``5e-7·|y|`` agreed
(Richardson). BIOMD0000000393's rate laws carry the quadratic root
``(N/2)·(sqrt(1 + 4x/N) − 1)`` with ``4x/N ≈ 1e-6``, which doubles resolve to
only 8e-10 relative. Its rounding structure is the same at both steps, so the
two differences agreed to 1e-6 while both sat 3% off the true slope on an
entry — ``∂f_HePc/∂TPc`` — whose value is a small residual of ten O(1)
contributions, and the correct Jacobian was rejected.

Rounding error in a central difference grows as the step shrinks and
truncation error shrinks, so a third, coarser step at ``2e-4·|y|`` moves off the
plateau. The gate now requires all three differences to agree before it judges
an entry, and skips it as an FD artifact otherwise.

``tests/data/jac_selfcheck_rounding_plateau.net`` is the model's five rate laws
that move ``HePc`` and read ``TPc``, with every other quantity frozen at the
probe state where the rejection happened, so the plateau is at the seed.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _jacobian as J

FIXTURE = "jac_selfcheck_rounding_plateau.net"
# The entry BIOMD0000000393 was rejected on, as the full model computes it.
ENTRY_IN_BIOMD393 = 3.125867e-4


def _clear_selfcheck_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "BNGSIM_JAC_SELFCHECK_DENSE_MAX",
        "BNGSIM_JAC_SELFCHECK_SAMPLE",
        "BNGSIM_JAC_NO_SELFCHECK",
    ):
        monkeypatch.delenv(var, raising=False)


def _entry(m: bngsim.Model):
    names = list(m.species_names)
    return names.index("HePc()"), names.index("TPc()")


def _central(m: bngsim.Model, y: np.ndarray, i: int, j: int, h: float) -> float:
    s = h * abs(y[j])
    yp, ym = y.copy(), y.copy()
    yp[j] += s
    ym[j] -= s
    return float((np.asarray(m.rhs(yp))[i] - np.asarray(m.rhs(ym))[i]) / (2.0 * s))


# ─── The mechanism, pinned on the engine's own RHS ───────────────────────────


def test_the_two_gate_steps_sit_on_a_plateau_the_coarser_step_leaves(data_dir: Path):
    """What the old rule saw: the differences at 1e-5 and 5e-7 agree to 1e-6 and
    both sit 3% below the true slope; at 2e-4 the difference is 0.15% from it."""
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    i, j = _entry(m)
    y0 = np.asarray(m.get_state(), float)
    fd0, fd1, fd2 = (_central(m, y0, i, j, h) for h in (2e-4, 1e-5, 5e-7))
    assert abs(fd1 - fd2) < 1e-5 * abs(fd1), "the old rule's two steps agree"
    assert abs(fd0 - fd1) > 2e-2 * abs(fd0), "and the coarser step disagrees by ~3%"
    assert abs(fd0 - ENTRY_IN_BIOMD393) < 5e-3 * ENTRY_IN_BIOMD393, "the coarse step is right"


def test_plateau_fixture_attaches_under_default_gate(data_dir: Path, monkeypatch):
    """Before the fix: False, with ``FD mismatch at (row=HePc, col=TPc)`` under
    ``BNGSIM_JAC_DEBUG=1``."""
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    assert m.analytical_jacobian_status == "complete"
    i, j = _entry(m)
    Jy = m.jacobian(np.asarray(m.get_state(), float))
    assert Jy.source == "analytical"
    assert Jy[i, j] == pytest.approx(ENTRY_IN_BIOMD393, rel=1e-6)


def test_plateau_fixture_attaches_under_sparse_gate(data_dir: Path, monkeypatch):
    """The sparse, column-sampled path carries the same three-step rule."""
    _clear_selfcheck_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_JAC_SELFCHECK_DENSE_MAX", "0")
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status


# ─── Negative control: the third step does not blind the gate ────────────────


def _derived_terms(core):
    ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    obs_groups = {n: [(int(s), float(f)) for s, f in g] for n, g in ctx["observables"]}
    smeta = {i: (bool(a), float(v)) for i, (a, v) in enumerate(ctx["species_meta"])}
    consts = set(ctx["constant_names"])
    terms = []
    for rxn in ctx["functional_reactions"]:
        t = J.build_per_species_terms(rxn["rate_expr"], func_map, obs_groups, smeta, consts)
        assert t is not None, rxn["rate_expr"]
        terms.append((rxn["rxn_idx"], False, [(int(j), e) for j, e in t]))
    return terms


def test_wrong_derivatives_are_still_rejected(data_dir: Path, monkeypatch):
    """∂f_HePc/∂HePc is O(1e-2) and converges across all three steps, so a 2×
    error on every derivative is judged there and caught."""
    _clear_selfcheck_env(monkeypatch)
    core = bngsim.Model.from_net(str(data_dir / FIXTURE))._core
    terms = _derived_terms(core)
    corrupted = [(ri, po, [(j, f"2.0*({e})") for j, e in dl]) for ri, po, dl in terms]
    assert core.set_functional_jacobian(corrupted) is False
    assert core.analytical_jacobian_complete is False
    core = bngsim.Model.from_net(str(data_dir / FIXTURE))._core
    assert core.set_functional_jacobian(_derived_terms(core)) is True


# ─── The corpus model itself ─────────────────────────────────────────────────

_B393 = Path("parity_checks/rr_parity/models/BIOMD0000000393/BIOMD0000000393_url.xml")


_B393_ABSENT = "rr_parity corpus model BIOMD0000000393 not present"


@pytest.mark.skipif(not _B393.exists(), reason=_B393_ABSENT)
def test_biomd393_keeps_its_analytical_jacobian(monkeypatch):
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_sbml(str(_B393))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    names = list(m.species_names)
    i, j = names.index("HePc"), names.index("TPc")
    n = len(names)
    y0 = np.asarray(m.get_state(), float)
    # the third probe state, where the old gate rejected it
    key = 0.7548776662
    yk = np.array([abs(y0[k]) * (0.4 + 1.8 * (((k + 1) * key) % 1.0)) + 1e-3 for k in range(n)])
    an = float(m.jacobian(yk)[i, j])
    assert an == pytest.approx(ENTRY_IN_BIOMD393, rel=1e-5)
    assert abs(_central(m, yk, i, j, 2e-4) - an) < 5e-3 * an
