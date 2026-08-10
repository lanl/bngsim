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

GH #274 acted on that: the bound is now derived ONLY where the roots provably
cannot reach. The test is value-position reachability from the ODE RHS through
the assignment rules, pruning every piecewise condition that IS a registered
root (:func:`_sbml_loader._periodic_disc_escapes_roots`). A floor/modulo whose
only influence is a rooted condition needs no bound — the root already forces
the stop; one whose *value* reaches the RHS (``dose = D * frac``) does, because
its jump is an RHS discontinuity no relational brackets. On the corpus that
keeps the bound on 10 of the 25 and drops it on 15, cutting 1,015,702 internal
steps to 519,472 with no answer moving.

Which means most of this module now exercises the NO-bound path, and the
closed-form agreement below is what says dropping it there was safe. Two
fixtures carry the other side:

  * ``_EQ_GATED_PULSE`` — the necessity witness, and the reason "retire" would
    have been wrong. Its pulse is gated by an EQUALITY on a floor
    (``sub - 20*cyc == 0``, after BIOMD0000000589's ``i == 0`` cycle-index
    selector), which the inequality-only emitters cannot root, while an
    unrelated ``time < 5`` threshold IS rooted. So the model has roots and still
    needs the bound: with it, y(10) = 36.788 against an exact 36.788; without,
    182.212 — 395% high, at BOTH tolerances, which is the step-over signature
    (it never sees the pulses, so tightening rtol cannot help). A predicate that
    merely counted roots would drop the bound here and return the wrong number.
  * ``BIOMD0000000312`` — a real model in the kept set, where the bound binds
    and disabling it changes nothing. The keep set is deliberately conservative;
    this pins that.

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


# ── GH #274: a pulse the roots cannot reach ─────────────────────────────────
# Same dy/dt and the same closed form, but the dose window is selected by an
# EQUALITY on a floor rather than an inequality on a fraction. The root scan
# emits conditions for relationals (`<`, `<=`, `>`, `>=`); an `==` on a step
# function is not one, so this edge gets no root however many the model has —
# and it has one, from the deliberately unrelated `time < 5` threshold driving
# `z`. That combination (rooted model, unrooted pulse) is what separates a
# root-COUNT predicate from a per-path one.
#
#   cyc  = floor(time)             integer cycle
#   sub  = floor(time * 20)        1/20-of-a-cycle index
#   dose = D  while  sub - 20*cyc == 0   (the first 0.05 of every cycle)
EQ_K, EQ_W, EQ_D = 20, 0.05, 4.0

_EQ_GATED_PULSE = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="equality_gated_pulse">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="y" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="z" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="{GVAL}" constant="true"/>
      <parameter id="kd" value="{KDVAL}" constant="true"/>
      <parameter id="D" value="{EQ_D}" constant="true"/>
      <parameter id="cyc" value="0" constant="false"/>
      <parameter id="sub" value="0" constant="false"/>
      <parameter id="dose" value="0" constant="false"/>
      <parameter id="other" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="cyc">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><floor/>
            <csymbol encoding="text"
              definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
          </apply>
        </math>
      </assignmentRule>
      <assignmentRule variable="sub">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><floor/>
            <apply><times/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
              <cn>{EQ_K}</cn>
            </apply>
          </apply>
        </math>
      </assignmentRule>
      <assignmentRule variable="dose">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece>
              <ci>D</ci>
              <apply><eq/>
                <apply><minus/>
                  <ci>sub</ci>
                  <apply><times/><cn>{EQ_K}</cn><ci>cyc</ci></apply>
                </apply>
                <cn>0</cn>
              </apply>
            </piece>
            <otherwise><cn>0</cn></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
      <assignmentRule variable="other">
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
            <apply><minus/><ci>g</ci><apply><times/><ci>kd</ci><ci>dose</ci></apply></apply>
            <ci>y</ci>
          </apply>
        </math>
      </rateRule>
      <rateRule variable="z">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>other</ci></math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


def _floor_oracle(n: int) -> float:
    """Closed-form y(n) at integer n for the periodic floor-dose model."""
    return Y0 * math.exp(n * (GVAL - KDVAL * DVAL * WVAL))


def _eq_gated_oracle(n: int) -> float:
    """The same closed form for the equality-gated fixture's own D and w."""
    return Y0 * math.exp(n * (GVAL - KDVAL * EQ_D * EQ_W))


def _y(result):
    names = list(result.species_names)
    return np.asarray(result.species)[:, names.index("y")]


def test_loader_derives_step_bound_below_window():
    """The derivation arithmetic, on a schedule that still gets a bound (#274).

    Measured on the equality-gated fixture rather than ``_floor_dose_sbml``,
    whose pulse condition is a rootable inequality and so no longer earns one.
    """
    model = bngsim.Model.from_sbml_string(_EQ_GATED_PULSE)
    ms = model._periodic_disc_max_step
    assert ms is not None
    # Bound must keep a step from spanning the window; the loader targets
    # window/3, so it sits comfortably below it.
    assert 0.0 < ms < EQ_W
    assert ms == pytest.approx(EQ_W / 3.0, rel=1e-3)


def test_rooted_pulse_condition_gets_no_bound():
    """``_floor_dose_sbml``'s edge is an inequality, so #72's scan roots it and
    #274 declines the bound: the root already forces the stop it would buy.

    The closed-form tests below run this same fixture unbounded and still hit
    the oracle, which is what makes the decline safe rather than merely cheap.
    """
    model = bngsim.Model.from_sbml_string(_floor_dose_sbml(WVAL))
    assert model._core.n_discontinuity_triggers >= 1
    assert model._periodic_disc_max_step is None


def test_time_threshold_schedule_gets_no_periodic_bound():
    # `time < 5` has no floor/modulo → #72 root path, no periodic step bound.
    model = bngsim.Model.from_sbml_string(SBML_TIME_THRESHOLD)
    assert model._periodic_disc_max_step is None
    assert model._core.n_discontinuity_triggers == 1  # the `time < 5` root


def test_clone_preserves_step_bound():
    # On a fixture that actually carries a bound — cloning `None` to `None`
    # would pass without testing anything.
    model = bngsim.Model.from_sbml_string(_EQ_GATED_PULSE)
    assert model._periodic_disc_max_step is not None
    clone = model.clone()
    assert clone._periodic_disc_max_step == model._periodic_disc_max_step


def test_equality_gated_pulse_still_needs_the_bound():
    """GH #274's necessity witness — and why the bound is narrowed, not retired.

    The pulse edge is ``sub - 20*cyc == 0``. An equality on a step function is
    not something the root scan emits, so nothing brackets this edge, and the
    step bound is the only thing standing between the integrator and a 0.05-wide
    window. The model nevertheless HAS a root (the unrelated ``time < 5``
    threshold driving ``z``), which is exactly the shape a root-count predicate
    gets wrong.

    Disabling the bound does not degrade the answer, it destroys it: 182.2
    against an exact 36.8, identically at rtol 1e-9 and 1e-11. Tol-stably wrong
    is the signature of a pulse that is never sampled at all.
    """
    model = bngsim.Model.from_sbml_string(_EQ_GATED_PULSE)
    assert model._core.n_discontinuity_triggers >= 1, "the unrelated time<5 root"
    assert model._periodic_disc_max_step is not None, (
        "a rooted model can still have an unrooted pulse — this is the case a "
        "root-count predicate would drop the bound on"
    )

    exact = _eq_gated_oracle(10)
    for rtol in (1e-9, 1e-11):
        r = bngsim.Simulator(bngsim.Model.from_sbml_string(_EQ_GATED_PULSE), method="ode").run(
            t_span=(0.0, 10.0),
            n_points=11,
            rtol=rtol,
            atol=1e-12,
            max_steps=10_000_000,
            timeout=120,
        )
        got = _y(r)[-1]
        assert got == pytest.approx(exact, rel=1e-3), f"bound on, rtol={rtol}: {got}"

        off = bngsim.Simulator(bngsim.Model.from_sbml_string(_EQ_GATED_PULSE), method="ode").run(
            t_span=(0.0, 10.0),
            n_points=11,
            rtol=rtol,
            atol=1e-12,
            max_steps=10_000_000,
            max_step=-1,
            timeout=120,
        )
        stepped_over = _y(off)[-1]
        assert stepped_over > 3.0 * exact, (
            f"rtol={rtol}: expected the unbounded arm to jump the pulses "
            f"(~{Y0 * math.exp(10 * GVAL):.1f}), got {stepped_over}"
        )


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
    # GH #274 — this model no longer gets a bound at all: every disc node in its
    # cycle arithmetic reaches the RHS only through conditions #259 roots. The
    # oracle agreement below is now the evidence that the drop was safe.
    assert bngsim.Model.from_sbml(_MODEL1708)._periodic_disc_max_step is None
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
    68% high. Since GH #274 the ``max_step=-1`` here is redundant rather than
    contrarian: this model gets no bound to disable. Keeping it explicit means
    the test still pins root coverage if the derivation ever comes back.
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


def _two_arm(path: str, t_end: float, n_points: int, on_max_step: float | None = None):
    """(bound-on, bound-off) x (rtol, rtol/100) final states and step counts.

    ``on_max_step`` forces a specific ceiling for the "on" arm. Needed for a
    model GH #274 declines to bound, where the default arm is already unbounded
    and the comparison would otherwise be against itself.
    """
    out = {}
    for arm, max_step in (("on", on_max_step), ("off", -1.0)):
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
def test_kept_bound_is_conservative_on_a_zero_root_model():
    """A model GH #274 KEEPS the bound for, where measurement says it need not.

    BIOMD0000000312 has a bound and *zero* discontinuity roots, so nothing else
    can be resolving its schedule and #274's predicate keeps the bound — as it
    must, since no root gates anything here. Empirically the bound is not needed:
    it binds (the arms take different step counts, which is what keeps this test
    from being vacuous) and disabling it moves the answer by ~1e-10, four orders
    below either arm's own tolerance stability.

    Both halves are the point. The predicate is deliberately conservative — it
    asks whether a root *provably* covers the edge, not whether the model happens
    to integrate fine without one — and this pins the gap between the two so a
    future tightening has a witness to argue against.
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
def test_dropped_bound_matches_a_finely_bounded_run():
    """A model GH #274 drops the bound on, checked against a forced ceiling.

    MODEL0847869198 has four discontinuity roots, and every disc node in its
    schedule reaches the RHS only through conditions those roots cover — so the
    loader no longer bounds it. The claim that has to hold is not "the old bound
    was harmless" but the stronger "running unbounded equals running with a step
    ceiling fine enough to resolve anything": 0.05 here, well under the 0.333 the
    pre-#274 loader derived, so it cannot be the ceiling that is too coarse to
    disagree.

    The price the drop recovers is visible in the step counts — the forced
    ceiling costs >20% more internal steps for an answer that matches to ~1e-9.
    ``max_step`` shortens every step over the whole horizon, not just the ones
    near a pulse edge.
    """
    path = _rr_model("MODEL0847869198")
    model = bngsim.Model.from_sbml(path)
    assert model._core.n_discontinuity_triggers > 0
    assert model._periodic_disc_max_step is None, (
        "expected #274 to drop this model's bound — its schedule is fully rooted"
    )

    arms = _two_arm(path, t_end=100.0, n_points=101, on_max_step=0.05)
    steps_on, steps_off = arms["on_tight"][1], arms["off_tight"][1]
    assert steps_on > 1.2 * steps_off, (
        f"expected the forced ceiling to cost steps ({steps_on} with vs "
        f"{steps_off} without); if it no longer binds, lower it"
    )
    for arm in ("on", "off"):
        stability = _final_state_rel_diff(arms[f"{arm}_loose"][0], arms[f"{arm}_tight"][0])
        assert stability < 1e-5, f"{arm} arm is not tol-stable ({stability:.2e})"
    diff = _final_state_rel_diff(arms["on_tight"][0], arms["off_tight"][0])
    assert diff < 1e-6, f"unbounded disagrees with a fine ceiling by {diff:.2e}"
