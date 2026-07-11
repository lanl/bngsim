"""GH #9: the RoadRunner SSA oracle must refuse homo-oligomerization (repeated
reactant) ``.net``\\s, whose SBML mass-action law is ODE-faithful but not
propensity-faithful under RoadRunner's Gillespie.

For ``2A -> C`` the exact-SSA propensity is the falling factorial ``k*A*(A-1)`` —
what bngsim's ``NetworkModel::compute_propensity`` and the ``net_gillespie`` oracle
both compute — but ``net_to_sbml`` emits the deterministic mass-action kinetic law
``k*A*A`` (the correct ODE RHS, a different function of the integer count). The
RHS-faithfulness gate (``max_rhs_delta`` ≈ 0) is therefore blind to the discrepancy,
so ``net_roadrunner`` screens for the repeated reactant structurally and stays
unscored rather than emit a DIFF that would be misattributed to bngsim's SSA.

The structural gate needs only libsbml + ``bngsim.convert`` (no libroadrunner), so it
runs in CI. The optional end-to-end confirmation — that an ungated RoadRunner run
really would diverge from the exact answer — is skipped when libroadrunner is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bngsim  # noqa: E402
import net_roadrunner as nrr  # noqa: E402

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_LIBSBML", True), reason="requires libsbml"
)

_DATA = HERE.parent.parent / "tests" / "data"
_HOMODIMER = _DATA / "homodimer_ssa.net"  # 2A -> C, rate 0.5*k1
_CLEAN = _DATA / "two_species_reversible.net"  # A <-> B, no repeated reactant


def _sbml(net: Path) -> str:
    from bngsim.convert import net_to_sbml

    return net_to_sbml(net, out_path=None, validate=None, strict=True).output_text


def test_has_repeated_reactant_true_for_homodimer() -> None:
    assert _HOMODIMER.is_file(), _HOMODIMER
    assert nrr._has_repeated_reactant(_sbml(_HOMODIMER)) is True


def test_has_repeated_reactant_false_for_clean_net() -> None:
    assert _CLEAN.is_file(), _CLEAN
    assert nrr._has_repeated_reactant(_sbml(_CLEAN)) is False


def test_faithful_sbml_refuses_repeated_reactant() -> None:
    # RHS-faithful (A*A IS the right ODE RHS) yet stochastically unfaithful → refuse.
    assert nrr._faithful_sbml_text(_HOMODIMER) is None


def test_faithful_sbml_accepts_clean_net() -> None:
    # Guard against over-refusal: a repeated-reactant-free net is still accepted.
    assert nrr._faithful_sbml_text(_CLEAN) is not None


@pytest.mark.skipif(not nrr.roadrunner_available(), reason="libroadrunner not installed")
def test_ungated_roadrunner_would_diverge_on_homodimer() -> None:
    """Why the gate exists: with A(0)=1 the exact-SSA answer for C is identically 0
    (can't pick 2 of 1), and bngsim's SSA gives 0 — but RoadRunner's Gillespie, fed the
    ``0.5*k1*A*A`` law, fires the reaction and grows C. This is the DIFF the gate
    suppresses; asserting it keeps the granularity mismatch pinned."""
    import warnings

    import numpy as np

    n_rep = 64

    model = bngsim.Model.from_net(str(_HOMODIMER))
    sim = bngsim.Simulator(model, method="ssa")
    ci = list(sim.run(t_span=(0.0, 10.0), n_points=11, seed=0).observable_names).index("C_tot")
    bn_final = []
    for rep in range(n_rep):
        model.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
            r = sim.run(t_span=(0.0, 10.0), n_points=11, seed=rep)
        bn_final.append(np.asarray(r.observables)[-1, ci])
    assert float(np.mean(bn_final)) == 0.0  # exact SSA: propensity A*(A-1) = 0 at A=1

    # Bypass the gate deliberately to exercise the raw RoadRunner path on this net.
    from bngsim.convert import net_to_sbml

    report = net_to_sbml(_HOMODIMER, out_path=None, validate="L1", strict=True)
    assert report.max_rhs_delta is not None and report.max_rhs_delta <= nrr.RHS_FAITHFUL_TOL
    text = report.output_text
    assert nrr._has_repeated_reactant(text) is True  # gate would have caught it

    import roadrunner

    rr = roadrunner.RoadRunner(text)
    rr.integrator = "gillespie"
    rr.integrator.variable_step_size = False
    id_map = nrr._observable_id_map(text)
    rr.timeCourseSelections = ["time", id_map["C_tot"]]
    rr_final = []
    for rep in range(n_rep):
        rr.reset()
        rr.integrator.seed = rep
        res = np.asarray(rr.simulate(0.0, 10.0, 11))
        rr_final.append(res[-1, 1])
    assert float(np.mean(rr_final)) > 1.0  # ungated RoadRunner diverges from the exact 0
