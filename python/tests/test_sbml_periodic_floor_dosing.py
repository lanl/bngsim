"""ODE correctness for periodic floor()/modulo dosing discontinuities (GH #88).

The #72 machinery (test_sbml_time_piecewise_discontinuity.py) registers a CVODE
root for every inequality that compares the SBML ``time`` csymbol DIRECTLY against
a fixed threshold (``time < 21``) — a monotonic edge CVODE brackets regardless of
step size. A *periodic* chemo schedule instead encodes its dose edges through
floor()/modulo time arithmetic routed via intermediate assignment-rule parameters
(MODEL1708310001 / Claret2009: ``exposure`` switches on ``rem_time = time mod
cycle`` and ``frac = rem_time - floor(rem_time)``). Those edges are invisible to
the direct-``time`` scan, and a single boolean root for a periodic pulse is
non-monotonic — CVODE can step straight over a narrow "on" window. On an
exponentially growing state the missed dose-decay compounds into a persistent
offset (bngsim read y(100)=1603 analytical / 1578 fd, RoadRunner 1570, vs the
exact segmented answer 953.07; every engine only converges to 953 as its
tolerance is tightened enough to resolve the 0.0625-day pulses).

The loader detects the periodic structure and bounds the integrator step below
the narrowest dose window, so no step can span a pulse. This test locks:

  1. the loader derives a step bound (< the window width) ONLY for a model with a
     time-dependent floor/modulo feeding the ODE RHS;
  2. a narrow periodic pulse is delivered every cycle, matching a closed-form
     oracle on both coarse and fine grids, and tol-stably;
  3. what disabling the bound (``max_step<=0``) does now — see below;
  4. a plain ``time<const`` schedule gets NO bound (it stays on the #72 root
     path, unchanged), and the bound survives ``Model.clone``.

Item 3 used to read "disabling the bound reproduces the stepped-over wrong answer
— i.e. the bound is what fixes it", and it was true: MODEL1708310001 jumped its
0.0625-day pulses in 221 steps and overshot to y(100)≈1602.95. GH #259 gave that
model five more discontinuity roots and they bracket the same pulse edges, so it
reaches 953.07 without the bound and stopped witnessing necessity.

GH #262 asked whether ANY model still does. It does not. A two-arm sweep
(``max_step`` default vs ``max_step=-1``, at rtol and rtol/100) over all 25
rr_parity models the loader gives a bound found no model whose answer the bound
changes: 23 agree to within their own tolerance stability, and the other two
(BIOMD0000000858/859) are not tol-stable in either arm and are byte-identical
between them — the bound never binds there, so they cannot adjudicate anything.
The bound is inert on 15 of the 25 (identical step counts both arms) and, where
it does bind, costs up to 2.5x the steps (MODEL0406553884: 439,367 vs 176,919)
for no change in the answer.

So item 3 is now the opposite assertion, on the class GH #262 flagged as the one
roots provably cannot reach — a model carrying a bound and NO discontinuity roots
at all — plus one rooted model where the bound demonstrably binds. Both say the
same thing: the bound changes step selection and nothing else. Whether it should
then be narrowed or retired is a separate call; these pin the measurement so it
cannot rot silently the way the necessity claim did.

Oracle is closed-form. For ``dy/dt = (g - kd·dose(t))·y`` with ``dose = D`` while
``frac(time) = time - floor(time) < w`` (else 0) and constant ``kd``, each unit
interval multiplies y by ``exp((g-kd·D)·w + g·(1-w)) = exp(g - kd·D·w)``, so
``y(N) = y0·exp(N·(g - kd·D·w))`` at integer N.
"""

import math
import os

import bngsim
import numpy as np
import pytest

# ── Minimal periodic-floor dosing model ─────────────────────────────────────
# y grows at rate g, knocked down by a daily dose of strength D active only on
# the first W of each unit interval (frac(time) < W). The window is far narrower
# than a natural integrator step, so without a step bound the dose is stepped
# over and y grows unchecked (exp(+g·t)); resolved, y follows exp((g-kd·D·W)·t).
GVAL, KDVAL, DVAL, WVAL, Y0 = 0.1, 1.0, 2.0, 0.1, 100.0


def _floor_dose_sbml(width: float) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="floor_dose">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="y" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="0.1" constant="true"/>
      <parameter id="kd" value="1" constant="true"/>
      <parameter id="D" value="2" constant="true"/>
      <parameter id="w" value="{width}" constant="true"/>
      <parameter id="dose" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="dose">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece>
              <ci>D</ci>
              <apply><lt/>
                <apply><minus/>
                  <csymbol encoding="text"
                    definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <apply><floor/>
                    <csymbol encoding="text"
                      definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  </apply>
                </apply>
                <ci>w</ci>
              </apply>
            </piece>
            <otherwise><cn>0</cn></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
      <rateRule variable="y">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
            <apply><minus/><ci>g</ci><apply><times/><ci>kd</ci><ci>dose</ci></apply></apply>
            <ci>y</ci>
          </apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


# A `time < const` schedule (no floor/modulo): must stay on the #72 root path
# with NO periodic step bound.
SBML_TIME_THRESHOLD = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="time_threshold">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="y" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="0.1" constant="true"/>
      <parameter id="dose" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="dose">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece><cn>1</cn>
              <apply><lt/>
                <csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                <cn>5</cn></apply>
            </piece>
            <otherwise><cn>0</cn></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
      <rateRule variable="y">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
            <apply><minus/><ci>g</ci><ci>dose</ci></apply><ci>y</ci></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


def _floor_oracle(n: int) -> float:
    """Closed-form y(n) at integer n for the periodic floor-dose model."""
    return Y0 * math.exp(n * (GVAL - KDVAL * DVAL * WVAL))


def _y(result):
    names = list(result.species_names)
    return np.asarray(result.species)[:, names.index("y")]


def test_loader_derives_step_bound_below_window():
    model = bngsim.Model.from_sbml_string(_floor_dose_sbml(WVAL))
    ms = model._periodic_disc_max_step
    assert ms is not None
    # Bound must keep a step from spanning the W-wide window; the loader targets
    # window/3, so it sits comfortably below W.
    assert 0.0 < ms < WVAL
    assert ms == pytest.approx(WVAL / 3.0, rel=1e-3)


def test_time_threshold_schedule_gets_no_periodic_bound():
    # `time < 5` has no floor/modulo → #72 root path, no periodic step bound.
    model = bngsim.Model.from_sbml_string(SBML_TIME_THRESHOLD)
    assert model._periodic_disc_max_step is None
    assert model._core.n_discontinuity_triggers == 1  # the `time < 5` root


def test_clone_preserves_step_bound():
    model = bngsim.Model.from_sbml_string(_floor_dose_sbml(WVAL))
    clone = model.clone()
    assert clone._periodic_disc_max_step == model._periodic_disc_max_step


@pytest.mark.parametrize("jac", ["analytical", "fd"])
@pytest.mark.parametrize("rtol", [1e-9, 1e-11])
def test_periodic_pulse_resolved_matches_closed_form(jac, rtol):
    """Every daily pulse is delivered (y decays), matching the closed form
    tol-stably and independent of the Jacobian — the signature of resolved
    discontinuities, not a tol/grid coincidence."""
    model = bngsim.Model.from_sbml_string(_floor_dose_sbml(WVAL))
    sim = bngsim.Simulator(model, method="ode", jacobian=jac)
    # Coarse output grid whose samples never land inside a [n, n+W) window.
    r = sim.run(
        t_span=(0.0, 10.0), n_points=11, rtol=rtol, atol=1e-12, max_steps=10_000_000, timeout=60
    )
    y, t = _y(r), np.asarray(r.time)
    for n in (3, 5, 8, 10):
        got = y[int(np.argmin(np.abs(t - n)))]
        assert got == pytest.approx(_floor_oracle(n), rel=2e-3), f"n={n}: {got}"


def test_fine_grid_also_resolved():
    """A fine grid (samples inside the windows) must resolve too — not a
    coarse-grid coincidence."""
    model = bngsim.Model.from_sbml_string(_floor_dose_sbml(WVAL))
    sim = bngsim.Simulator(model, method="ode")
    r = sim.run(
        t_span=(0.0, 10.0), n_points=2001, rtol=1e-10, atol=1e-12, max_steps=10_000_000, timeout=60
    )
    y, t = _y(r), np.asarray(r.time)
    got = y[int(np.argmin(np.abs(t - 10.0)))]
    assert got == pytest.approx(_floor_oracle(10), rel=5e-3)


# ── The real model that surfaced the bug ────────────────────────────────────
_MODEL1708 = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "parity_checks",
    "rr_parity",
    "models",
    "MODEL1708310001",
    "MODEL1708310001.xml",
)


@pytest.mark.skipif(not os.path.exists(_MODEL1708), reason="MODEL1708310001 SBML not present")
@pytest.mark.parametrize("jac", ["analytical", "fd"])
def test_model1708310001_converges_to_segmented_oracle(jac):
    """Claret2009 colorectal-cancer OS: the periodic chemo schedule (floor/modulo
    cycle arithmetic) drives y; resolved, bngsim reaches the exact segmented
    answer y(100)=953.07 tol-stably for either Jacobian (pre-fix: 1603 analytical
    / 1578 fd at the sweep tol, only converging to 953 as tol→1e-11)."""
    assert bngsim.Model.from_sbml(_MODEL1708)._periodic_disc_max_step is not None
    for rtol in (1e-9, 1e-11):
        # fresh state each run (run() does not reset a reused model)
        m = bngsim.Model.from_sbml(_MODEL1708)
        s = bngsim.Simulator(m, method="ode", jacobian=jac)
        r = s.run(
            t_span=(0.0, 100.0),
            n_points=101,
            rtol=rtol,
            atol=1e-12,
            max_steps=50_000_000,
            timeout=120,
        )
        y = np.asarray(r.species)[-1, list(r.species_names).index("y")]
        assert y == pytest.approx(953.07, rel=5e-3), f"jac={jac} rtol={rtol}: {y}"


@pytest.mark.skipif(not os.path.exists(_MODEL1708), reason="MODEL1708310001 SBML not present")
def test_model1708310001_roots_resolve_the_schedule_without_the_bound():
    """This model no longer witnesses the *necessity* of the step bound.

    It used to: with the bound disabled (``max_step <= 0``) the integrator
    jumped the 0.0625-day chemo pulses in 221 steps and overshot to
    y(100)≈1602.95, and the bound was the only thing collapsing that to the
    exact segmented 953.07. GH #259 gave the model five more discontinuity roots
    (1 → 6) by reading its ``rem_time - floor(rem_time)`` cycle arithmetic as
    time-dependent on *both* sides of each relational, and those roots now
    bracket the same pulse edges: bound disabled, it reaches 953.069 in 7,632
    steps, tol-stably.

    So this asserts what is true now — the roots alone resolve the schedule —
    and it still fails loudly if they regress, because the step-over answer is
    68% high. The bound itself is still derived (the test above pins that) and
    still on by default; whether any model still *needs* it is GH #262.
    """
    for rtol in (1e-9, 1e-11):
        m = bngsim.Model.from_sbml(_MODEL1708)
        sim = bngsim.Simulator(m, method="ode", jacobian="analytical")
        r = sim.run(
            t_span=(0.0, 100.0),
            n_points=101,
            rtol=rtol,
            atol=1e-12,
            max_steps=50_000_000,
            max_step=-1,
            timeout=120,
        )
        y = np.asarray(r.species)[-1, list(r.species_names).index("y")]
        assert y == pytest.approx(953.07, rel=5e-3), f"rtol={rtol}: {y}"
        # Emphatically not the pre-#259 step-over, which was ≈1602.95.
        assert y < 1.1 * 953.07
        assert r.solver_stats["n_steps"] > 1000, (
            "221 steps was the step-over signature; resolving the pulses costs "
            f"thousands (rtol={rtol}: {r.solver_stats['n_steps']})"
        )


# ── GH #262: is the bound still doing work anywhere? ────────────────────────
_MODELS = os.path.join(
    os.path.dirname(__file__), "..", "..", "parity_checks", "rr_parity", "models"
)


def _rr_model(model_id: str) -> str:
    # The corpus carries BioModels entries as `<id>_url.xml` and the older
    # MODEL* ones as plain `<id>.xml`; return whichever is there.
    for fn in (f"{model_id}_url.xml", f"{model_id}.xml"):
        path = os.path.join(_MODELS, model_id, fn)
        if os.path.exists(path):
            return path
    return os.path.join(_MODELS, model_id, f"{model_id}_url.xml")


def _final_state_rel_diff(a, b) -> float:
    """Max relative difference between two final-state vectors.

    Floored at the run's absolute tolerance and at 1e-8 of the largest state, so
    a species parked at 1e-20 — where the two arms differ only in roundoff —
    cannot dominate a *relative* comparison and manufacture a disagreement.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    floor = max(1e-10, 1e-8 * float(np.max(np.abs(np.concatenate([a, b]))) or 1.0))
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    return float(np.max(np.abs(a - b) / denom))


def _two_arm(path: str, t_end: float, n_points: int):
    """(bound-on, bound-off) x (rtol, rtol/100) final states and step counts."""
    out = {}
    for arm, max_step in (("on", None), ("off", -1.0)):
        for tag, rtol in (("loose", 1e-8), ("tight", 1e-10)):
            kw = dict(
                t_span=(0.0, t_end),
                n_points=n_points,
                rtol=rtol,
                atol=1e-10,
                max_steps=20_000_000,
                timeout=120,
            )
            if max_step is not None:
                kw["max_step"] = max_step
            sim = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode")
            r = sim.run(**kw)
            out[f"{arm}_{tag}"] = (
                np.asarray(r.species)[-1, :],
                int(r.solver_stats["n_steps"]),
            )
    return out


@pytest.mark.skipif(
    not os.path.exists(_rr_model("BIOMD0000000312")), reason="BIOMD0000000312 SBML not present"
)
def test_bound_unnecessary_even_where_no_root_can_reach():
    """The one class GH #262 kept the bound for turns out not to need it either.

    #262's caveat was that "the bound also covers schedules whose thresholds the
    loader cannot emit as an ExprTk condition at all", so a model with a bound
    and *zero* discontinuity roots is where the bound is the only mechanism in
    play — nothing else can be resolving its pulses. BIOMD0000000312 is exactly
    that, and its bound is not inert (the two arms take different numbers of
    steps, which is what keeps this test from being vacuous). Disabling it
    changes the answer by ~1e-10, four orders below either arm's own tolerance
    stability.
    """
    path = _rr_model("BIOMD0000000312")
    model = bngsim.Model.from_sbml(path)
    assert model._periodic_disc_max_step is not None
    assert model._core.n_discontinuity_triggers == 0, (
        "this fixture's whole point is that no root can be resolving the "
        "schedule — pick another zero-root model if this one gains roots"
    )

    arms = _two_arm(path, t_end=10.0, n_points=1001)
    # The bound must actually constrain the integration, or agreement is vacuous.
    assert arms["on_tight"][1] != arms["off_tight"][1], (
        f"bound is inert here ({arms['on_tight'][1]} steps in both arms) — it "
        "cannot witness anything either way"
    )
    for arm in ("on", "off"):
        stability = _final_state_rel_diff(arms[f"{arm}_loose"][0], arms[f"{arm}_tight"][0])
        assert stability < 1e-5, f"{arm} arm is not tol-stable ({stability:.2e})"
    diff = _final_state_rel_diff(arms["on_tight"][0], arms["off_tight"][0])
    assert diff < 1e-6, f"the bound changes this model's answer by {diff:.2e}"


@pytest.mark.skipif(
    not os.path.exists(_rr_model("MODEL0847869198")), reason="MODEL0847869198 SBML not present"
)
def test_bound_costs_steps_and_buys_nothing_on_a_rooted_model():
    """Same verdict on the rooted side, and here the price is visible.

    MODEL0847869198 carries both a bound and four discontinuity roots. The bound
    binds hard — it forces >20% more internal steps — and buys an answer that
    matches the unbounded one to ~1e-9. This is the shape of the cost #262
    weighed: ``max_step`` is a blunt instrument that shortens every step over the
    whole horizon, not just the ones near a pulse edge.
    """
    path = _rr_model("MODEL0847869198")
    model = bngsim.Model.from_sbml(path)
    assert model._periodic_disc_max_step is not None
    assert model._core.n_discontinuity_triggers > 0

    arms = _two_arm(path, t_end=100.0, n_points=101)
    steps_on, steps_off = arms["on_tight"][1], arms["off_tight"][1]
    assert steps_on > 1.2 * steps_off, (
        f"expected the bound to cost steps ({steps_on} with vs {steps_off} without); "
        "if it no longer binds, this model has stopped being a cost witness"
    )
    for arm in ("on", "off"):
        stability = _final_state_rel_diff(arms[f"{arm}_loose"][0], arms[f"{arm}_tight"][0])
        assert stability < 1e-5, f"{arm} arm is not tol-stable ({stability:.2e})"
    diff = _final_state_rel_diff(arms["on_tight"][0], arms["off_tight"][0])
    assert diff < 1e-6, f"the bound changes this model's answer by {diff:.2e}"
