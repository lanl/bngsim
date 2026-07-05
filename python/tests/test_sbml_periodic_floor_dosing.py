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
  3. disabling the bound (``max_step<=0``) reproduces the stepped-over wrong
     answer — i.e. the bound is what fixes it;
  4. a plain ``time<const`` schedule gets NO bound (it stays on the #72 root
     path, unchanged), and the bound survives ``Model.clone``.

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
def test_model1708310001_disabling_bound_steps_over_pulses():
    """Disabling the bound (max_step<=0) reproduces the original step-over: the
    integrator jumps over the 0.0625-day chemo pulses, misses the dose-decay, and
    overshoots to y(100)≈1603 at the sweep tol — the bound is what collapses it to
    the exact 953.07."""
    m = bngsim.Model.from_sbml(_MODEL1708)
    sim = bngsim.Simulator(m, method="ode", jacobian="analytical")
    r = sim.run(
        t_span=(0.0, 100.0),
        n_points=101,
        rtol=1e-9,
        atol=1e-12,
        max_steps=50_000_000,
        max_step=-1,
        timeout=120,
    )
    y = np.asarray(r.species)[-1, list(r.species_names).index("y")]
    assert y > 1.3 * 953.07  # emphatically stepped over (≈1603, not 953)
