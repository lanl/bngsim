"""Issue #183 — sensitivity rows the column cannot resolve to rtol.

#177 floored ``atolS`` at the roundoff of the sum that *forms* ``ṡ``. This is the
roundoff of the column that *carries* ``s``, and neither implies the other.

``col_floor = ε‖s‖∞`` is the absolute noise every entry of a sensitivity column
inherits. A row asked for ``rtol·|s_i|`` finer than that is being asked for
digits the column does not have, and no step size supplies them — CVODES shrinks
``h`` chasing them, which is the same failure #177 fixed one level up. So when
``rtol·|s_i| < ε‖s‖∞`` the row is floored at ``|s_i|`` itself: not error-
controlled beyond its own size, which is the honest reading of a value that is
noise.

The number is measured, not chosen. On ``Smith2013`` at ``extracellular_ROS =
15000`` a bisection over the sensitivity error test names rows ``Akt_P2`` and
``PKC_P`` — and only those two — for every column that fails, and the ``atolS``
each needs is ≈ 1.6·|s_i| (k7) and ≈ 0.73·|s_i| (k8), from columns whose norms
differ by seven orders. The test that fires it separates the two regimes by ten
orders: at Smith's t≈19.3 the binding row sits 36 ulps of ‖s‖∞ above zero, while
``sens_scale_cancellation``'s one live row sits 4.5e15 of them.

The relaxation is a per-(row, column) high-water mark. A floor that TIGHTENS
mid-run is its own hazard, and this rule is not naturally monotone: ``Akt_P2``
leaves the unresolvable regime by growing 13 orders between t=19 and t=24. Left
unsticky it worked at one ladder spacing and failed at both a coarser and a finer
one — the signature of a lucky refresh time rather than a rule.

Delivery matters as much as the number. The floor is re-derived at every output
point, and on a coarse grid — Smith's own reproduction samples at t = 0, 120,
240 — the first refresh after t=0 lands long after the transition that needs it.
Hence an early refresh ladder, driven by single-stepping (``CV_ONE_STEP``) rather
than by extra output targets: CVODES sizes its first step from the distance to
the first ``tout``, so an extra early target rewrites ``h0`` and with it the whole
step sequence — measured on the corpus as moving nearly every model, which is why
that ladder shipped disarmed for everything that did not already need it at t=0,
and therefore for 15 of Smith's 16 columns.

``test_ladder_does_not_move_a_model_it_does_not_help`` and
``test_event_at_a_sample_time_is_not_applied_early`` pin the two ways
single-stepping can go wrong. ``BNGSIM_SENS_FLOOR_UNRESOLVABLE=0`` and
``BNGSIM_SENS_FLOOR_LADDER=off`` restore the prior behaviour from the same
binary.
"""

from __future__ import annotations

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

_BNGSIM = Path(__file__).resolve().parents[2]
_SMITH = _BNGSIM / "benchmarks" / "models" / "sbml" / "Smith2013_BIOMD0000000474_petab.xml"

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else _BNGSIM / "tests" / "data"

# The four columns issue #183 was filed for. k7/kminus7a died with CV_ERR_FAILURE
# at t=23.2961; k8/kminus8 did not finish at all. All four fail on rows Akt_P2
# and PKC_P — the same two rows, which is what says the columns are not the
# variable.
SMITH_COLUMNS = ["k7", "kminus7a", "k8", "kminus8"]

# Wall clock, not steps: Smith has events, and CVodeGetNumSteps counts from the
# last re-initialization, so a step assertion on an event model measures only the
# final segment (#182). The budget is deliberately two orders above the ~0.14 s
# these columns take once fixed, because the failure it guards is not "slow" but
# "does not terminate" — on the frozen scale these run until something kills
# them.
SMITH_BUDGET_S = 20.0


def _smith_model(ros_molecules: float = 15000.0):
    """The issue's reproduction: Smith2013 at a ROS dose that turns the stall on."""
    import libsbml

    sm = libsbml.SBMLReader().readSBMLFromFile(str(_SMITH)).getModel()

    def conc_factor(sid: str) -> float:
        sp = sm.getSpecies(sid)
        if not sp.getHasOnlySubstanceUnits():
            return 1.0
        return 1.0 / sm.getCompartment(sp.getCompartment()).getSize()

    m = bngsim.Model.from_sbml(str(_SMITH))
    m.set_param("t_ins", 240.0)
    m.set_param("indicator_jnk", 1.0)
    m.set_param("indicator_foxo", 1.0)
    m.set_param("k4", 0.000333333)
    m.set_param("kminus4", 0.003)
    m.set_param("k_irs1_basal_syn", 130.0)
    m.set_concentration("Ins", 5.0 * conc_factor("Ins"))
    m.set_concentration("E2F1", 150.0 * conc_factor("E2F1"))
    m.set_concentration("extracellular_ROS", ros_molecules * conc_factor("extracellular_ROS"))
    m.save_concentrations()
    m.reset()
    return m


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 benchmark not present")
@pytest.mark.parametrize("column", SMITH_COLUMNS)
def test_smith_sensitivity_column_finishes(column):
    """The reported defect, one column at a time.

    Per column and not all sixteen at once, because the whole-model run says only
    "the sensitivity error test costs everything" and never which column pays —
    that distinction is what unstuck both #177 and this.
    """
    sim = bngsim.Simulator(_smith_model(), method="ode", sensitivity_params=[column])
    r = sim.run(
        t_span=(0.0, 240.0),
        n_points=3,
        sample_times=[0.0, 120.0, 240.0],
        timeout=SMITH_BUDGET_S,
    )
    s = np.asarray(r.sensitivities)
    assert s.shape[0] == 3
    assert np.isfinite(s).all(), f"column {column} produced non-finite sensitivities"


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 benchmark not present")
def test_the_binding_rows_go_unresolvable_and_come_back():
    """The discriminator, and how far it separates the two regimes.

    ``Akt_P2`` and ``PKC_P`` are what a row bisection names for every failing
    column. What is true of them at the failure and false of a row that carries
    its column is ``rtol·|s_i| < ε‖s‖∞`` — the row's relative band is finer than
    the absolute noise the column carries — and here that holds by many orders
    rather than marginally, which is what makes the rule a rule and not a tuned
    constant.

    They come back, too: the same rows leave the regime by growing thirteen
    orders over t ∈ (19, 24). A relaxation that were not a high-water mark would
    be withdrawn at exactly that point, and the tolerances would tighten under a
    step size chosen while they were loose.
    """
    ts = [0.0, 12.7, 15.7, 19.3, 23.8, 30.0, 120.0, 240.0]
    r = bngsim.Simulator(_smith_model(), method="ode", sensitivity_params=["k7"]).run(
        t_span=(0.0, ts[-1]), n_points=len(ts), sample_times=ts, timeout=SMITH_BUDGET_S
    )
    s = np.asarray(r.sensitivities)[:, :, 0]
    names = list(r.species_names)
    eps = float(np.finfo(float).eps)
    rtol = 1e-8  # the run's default, and what the rule is stated against

    i = names.index("Akt_P2")
    col_noise = eps * np.abs(s).max(axis=1)
    band = rtol * np.abs(s[:, i])
    dip = [k for k in range(len(ts)) if 12.0 <= ts[k] <= 20.0]
    assert dip, "the sample grid no longer covers the window the rows go quiet in"
    assert (band[dip] < col_noise[dip]).all(), (
        "Akt_P2's sensitivity is no longer under its column's own noise floor in "
        "t ∈ (12, 20); this test needs a new witness"
    )
    # And by a wide margin, not a whisker.
    assert (col_noise[dip] / np.maximum(band[dip], 1e-300)).min() > 1e3

    late = ts.index(30.0)
    assert band[late] > col_noise[late], (
        "Akt_P2 no longer climbs back out of the unresolvable regime, which is "
        "what the high-water mark exists to survive"
    )


def test_a_row_that_carries_its_column_is_never_relaxed():
    """The other side of the discriminator, on the model whose accuracy it must keep.

    ``sens_scale_cancellation.net`` is #177's reproduction, and its one live
    sensitivity row IS its column — ``rtol·|s|`` sits about 4.5e15 ulps of ‖s‖∞
    above the noise, the whole of float64. A rule that touched it would give away
    the resolvable part of that model's answer, which
    ``test_sensitivities_are_accurate_where_resolvable`` in the #177 module pins;
    reading the same margin here says why it does not, rather than only that it
    did not.
    """
    net = DATA_DIR / "sens_scale_cancellation.net"
    if not net.is_file():
        pytest.skip(f"test data not found: {net}")
    ts = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    r = bngsim.Simulator(bngsim.Model.load(str(net)), method="ode", sensitivity_params=["p"]).run(
        t_span=(0.0, ts[-1]), n_points=len(ts), sample_times=ts, timeout=120.0
    )
    s = np.asarray(r.sensitivities)[:, :, 0]
    eps = float(np.finfo(float).eps)
    live = np.abs(s).max(axis=1) > 0.0
    margin = (1e-8 * np.abs(s).max(axis=1)) / np.maximum(eps * np.abs(s).max(axis=1), 1e-300)
    assert live.any()
    assert margin[live].min() > 1e6, (
        f"the live row is only {margin[live].min():.2e} above the unresolvable test; "
        "it should be the whole of float64, and if it is not, the rule is close to "
        "firing on a row that carries its column"
    )


# A model the floor does nothing for. NOT sens_scale_cancellation.net, which is
# #177's own reproduction and whose floor binds by construction — the bit-identity
# claim below is about models the ladder cannot help, so the witness has to be one
# where every refresh reports moved=0.
CLEAN_NET = "expr_sens_chain.net"
CLEAN_PARAM = "k1"


def _clean_model():
    return bngsim.Model.load(str(DATA_DIR / CLEAN_NET))


def test_ladder_does_not_move_a_model_it_does_not_help(monkeypatch):
    """Single-stepping is a delivery mechanism, not a tolerance change.

    ``CV_ONE_STEP`` takes exactly the steps ``CV_NORMAL`` would take toward the
    same ``tout`` and simply hands each one back, so a run whose floor never
    binds must be bit-identical with the ladder and without it. The extra-output-
    target ladder this replaced could not make that claim: it moved the first
    ``tout``, so it moved ``h0``, so it moved almost every model in the corpus
    whether or not the floor ever did anything for them.
    """

    def run(ladder: str):
        monkeypatch.setenv("BNGSIM_SENS_FLOOR_LADDER", ladder)
        sim = bngsim.Simulator(_clean_model(), method="ode", sensitivity_params=[CLEAN_PARAM])
        r = sim.run(t_span=(0.0, 10.0), n_points=11, timeout=120.0)
        return np.asarray(r.species), np.asarray(r.sensitivities), r.solver_stats["n_steps"]

    x_off, s_off, n_off = run("off")
    x_on, s_on, n_on = run("always")
    assert n_on == n_off, f"ladder changed the step count {n_off} -> {n_on}"
    assert np.array_equal(x_on, x_off), "ladder moved the state trajectory"
    assert np.array_equal(s_on, s_off), "ladder moved the sensitivities"


def test_event_at_a_sample_time_is_not_applied_early(monkeypatch):
    """The ordering ``CV_ONE_STEP`` must not disturb.

    ``CV_ONE_STEP`` does not stop short for ``tout``, so a step allowed to run
    past an output point can find a root sitting exactly there and apply the
    event *before* the sample is recorded — handing back a post-event state for a
    trigger that is strictly false at that instant. Caught on
    ``BIOMD0000000104`` (``time > 1``, sampled at t=1) as a 60% state difference
    at an identical step count, which is what a discrete jump landing on the
    wrong side of a sample looks like and no tolerance change ever produces.

    The scalar path never single-steps, so it is the reference for what the
    sample should hold.
    """
    xml = DATA_DIR / "sens_ladder_event_ordering.xml"
    if not xml.is_file():
        pytest.skip(f"test data not found: {xml}")

    def run(sens: bool, ladder: str):
        monkeypatch.setenv("BNGSIM_SENS_FLOOR_LADDER", ladder)
        kw = {"sensitivity_params": ["p"]} if sens else {}
        sim = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode", **kw)
        r = sim.run(t_span=(0.0, 10.0), n_points=11, timeout=120.0)
        return np.asarray(r.species)

    reference = run(False, "off")
    for ladder in ("off", "always"):
        x = run(True, ladder)
        assert np.abs(x - reference).max() / max(np.abs(reference).max(), 1e-300) < 1e-6, (
            f"with the ladder {ladder}, a sensitivity run's trajectory left the scalar path — "
            "an event was applied on the wrong side of a sample"
        )
