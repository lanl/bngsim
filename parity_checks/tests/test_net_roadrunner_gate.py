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

import signal
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
    suppresses; asserting it keeps the granularity mismatch pinned.

    Each run is capped at 50 output rows (49 events). Since libRoadRunner 2.9.3 a
    Gillespie run no longer stops when a species goes negative (RoadRunner #1322),
    so A falls 1, -1, -3, ... and the law's propensity grows as A**2: a run has
    infinitely many events before a finite time (about 4.5 on average at k1 = 1),
    and 64 uncapped replicates to t = 10 never return."""
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
    rr.integrator.variable_step_size = True  # where max_output_rows applies: a row per event
    rr.integrator.max_output_rows = 50
    id_map = nrr._observable_id_map(text)
    rr.timeCourseSelections = ["time", id_map["C_tot"]]
    rr_final = []
    for rep in range(n_rep):
        rr.reset()
        rr.integrator.seed = rep
        with warnings.catch_warnings():
            # The cap stopping a run short of t = 10 is the point.
            warnings.filterwarnings("ignore", "Simulation requested end time point", UserWarning)
            res = np.asarray(rr.simulate(0.0, 10.0))
        rr_final.append(res[-1, 1])
    # Ungated RoadRunner diverges from the exact 0: it fires 2A -> C at A = 1.
    assert sum(c > 0 for c in rr_final) > n_rep // 2


# ── A wall budget that holds inside a replicate (issue #1288) ────────────────


@pytest.mark.skipif(not nrr.roadrunner_available(), reason="libroadrunner not installed")
def test_the_ensemble_in_its_child_is_the_ensemble_in_process() -> None:
    """The budget runs the replicates in a child process, fixed-step as before, so
    the draws, and every reference recorded from them, are unchanged."""
    import numpy as np

    grid = np.linspace(0.0, 5.0, 11)
    got = nrr.net_roadrunner_ensemble(_CLEAN, grid, n_rep=8, seed_base=3)
    assert got is not None
    text = nrr._faithful_sbml_text(_CLEAN)
    ids = [nrr._observable_id_map(text)[n] for n in got[2]]
    direct = nrr._gillespie_ensemble(text, ids, 0.0, 5.0, 11, 8, 3)
    np.testing.assert_array_equal(got[1], direct)


def test_a_child_past_the_budget_is_stopped() -> None:
    import time

    t0 = time.perf_counter()
    assert nrr._within_wall_budget(time.sleep, (60,), 1.0) is None
    assert time.perf_counter() - t0 < 30


def test_a_child_that_dies_is_a_refusal_at_once() -> None:
    """A child that dies without a result (a segfault in RoadRunner) is read as a
    refusal when it dies, not when the budget runs out."""
    import os
    import time

    t0 = time.perf_counter()
    assert nrr._within_wall_budget(os._exit, (3,), 60.0) is None
    assert time.perf_counter() - t0 < 30


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="no SIGALRM on this platform")
def test_a_child_whose_parent_is_gone_ends_itself() -> None:
    """A scheduler that terminates the parent at its per-job cap skips the daemon
    cleanup, so nothing would kill a replicate that never returns. The child ends
    itself at its own alarm; here no parent reads or kills it."""
    import multiprocessing as mp
    import time

    ctx = mp.get_context("spawn")
    _, send_end = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=nrr._child, args=(time.sleep, (60,), send_end, 1.0))
    proc.start()
    proc.join(timeout=30)
    assert not proc.is_alive()
    assert proc.exitcode == -signal.SIGALRM


def _roadrunner_stops_at_a_negative_count() -> bool:
    if not nrr.roadrunner_available():
        return True
    import roadrunner
    from packaging.version import Version

    return Version(roadrunner.__version__) < Version("2.9.3")


@pytest.mark.skipif(
    _roadrunner_stops_at_a_negative_count(),
    reason="libroadrunner before 2.9.3 stops a Gillespie run at a negative count",
)
def test_a_replicate_that_never_returns_is_stopped_by_the_budget() -> None:
    """The homodimer, ungated: on libRoadRunner >= 2.9.3, ``0.5*k1*A*A`` fires
    infinitely many events before t = 10, and 64 replicates never return. The
    budget now stops them inside a replicate, where it used to be read only
    between replicates."""
    import time

    from bngsim.convert import net_to_sbml

    text = net_to_sbml(_HOMODIMER, out_path=None, validate="L1", strict=True).output_text
    ids = [nrr._observable_id_map(text)["C_tot"]]
    t0 = time.perf_counter()
    vals = nrr._within_wall_budget(nrr._gillespie_ensemble, (text, ids, 0.0, 10.0, 11, 64, 0), 5.0)
    assert vals is None
    assert time.perf_counter() - t0 < 60
