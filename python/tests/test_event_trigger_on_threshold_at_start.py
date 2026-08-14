"""Issue #340 — a trigger whose residual starts exactly ON its threshold.

``BIOMD0000000285`` declares ``PIdeath > 0`` on a species whose initial amount is
0. The trigger is false at ``t_start`` (``0 > 0``), but its residual sits exactly
on zero, and the whole aggregation cascade that feeds ``PIdeath`` starts at zero
too — so ``d(PIdeath)/dt`` at ``t_start`` is zero as well. The species
nevertheless creeps positive once the cascade turns over, the boolean trigger
flips, and bngsim used to read that as a rising edge and fire at ``t = 2.7e-27``:
``kalive := 0`` before the model had moved, freezing the whole trajectory at its
initial state. Two such events fired at the same instant, which is how forward
sensitivity saw it (an ambiguous ``dt*/dp`` at a crossing whose time is set by
the step controller rather than by the model).

What separates that from a trigger that genuinely crosses at ``t_start`` is
``dg/dt`` along the flow, read there and nowhere else — after one step the
residual is off zero and the coincidence has left no trace:

* ``dg/dt > 0`` — the trajectory leaves the threshold into the trigger's true
  side. ``time > 0`` is this case (nine corpus models), and so is a species with
  a nonzero production rate at ``t_start``. Both still fire.
* ``dg/dt <= 0`` — it does not leave, so a root reported at ``t_start`` is the
  initial condition being re-read rather than a transition, and does not fire.
  A later, genuine crossing of the same trigger is untouched.

This is the rule SUNDIALS applies to its own root functions (a ``g_i``
identically zero at ``t0`` is deactivated until it moves away), which bngsim
cannot inherit because the root it registers is the *boolean* trigger minus 0.5
and that is never zero. AMICI, which roots on the residual, gets it from
SUNDIALS and does not fire here; bngsim now agrees with it on this model.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "parity_checks"
    / "rr_parity"
    / "models"
    / "BIOMD0000000285"
)


def _sbml(species: str, rules: str, trigger: str, parameters: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="on_threshold">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      {species}
    </listOfSpecies>
    <listOfParameters>
      <parameter id="fired" value="0" constant="false"/>
      {parameters}
    </listOfParameters>
    <listOfRules>
      {rules}
    </listOfRules>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            {trigger}
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="fired">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


_S = (
    '<species id="S" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"'
    ' boundaryCondition="true" constant="false"/>'
)
_A = (
    '<species id="A" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"'
    ' boundaryCondition="true" constant="false"/>'
)
_RATE_S_IS_A = (
    '<rateRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<ci>A</ci></math></rateRule>"
)
_RATE_A_IS_ONE = (
    '<rateRule variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<cn>1</cn></math></rateRule>"
)
_RATE_A_IS_TWO = (
    '<rateRule variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<cn>2</cn></math></rateRule>"
)
_RATE_S_IS_ONE = (
    '<rateRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<cn>1</cn></math></rateRule>"
)
_TRIGGER_S_GT_0 = "<apply><gt/><ci>S</ci><cn>0</cn></apply>"


def _run(xml: str, t_end: float = 2.0, n_points: int = 21):
    model = bngsim.Model.from_sbml_string(xml)
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=1e-9, atol=1e-12
    )
    # An event-assigned parameter is promoted to a state, so it comes back as an
    # observable column rather than an expression.
    obs = list(res.observable_names)
    fired = np.asarray(res.observables)[:, obs.index("fired")]
    return np.asarray(res.time), fired, res


def test_zero_flow_on_the_threshold_does_not_fire():
    """The BIOMD0000000285 shape, minimal: ``S(0) = 0`` with ``dS/dt(0) = 0``.

    ``S = t^2/2`` is positive for every ``t > 0``, so the boolean trigger does
    flip — but nothing crosses anything at ``t_start``, and the instant a root
    lands on is whatever the first step happened to be. It must not fire.
    """
    t, fired, res = _run(_sbml(_S + _A, _RATE_S_IS_A + _RATE_A_IS_ONE, _TRIGGER_S_GT_0))

    assert not fired.any(), f"event fired at t={t[np.argmax(fired > 0)]}"
    # The trigger's residual really does go (numerically) positive — the run is
    # not passing because S stayed at zero.
    s = np.asarray(res.species)[:, list(res.species_names).index("S")]
    assert s[-1] == pytest.approx(t[-1] ** 2 / 2, rel=1e-6)


def test_positive_flow_on_the_threshold_still_fires():
    """The discriminator: same trigger, same ``S(0) = 0``, but ``dS/dt(0) = 1``.

    The trajectory leaves the threshold into the trigger's true side, so the
    crossing is real and sits at ``t_start``. Suppressing this one would be the
    over-correction the ``dg/dt`` test exists to avoid.
    """
    t, fired, _ = _run(_sbml(_S, _RATE_S_IS_ONE, _TRIGGER_S_GT_0))

    assert fired[0] == 0.0, "the t_start row is recorded before the root fires"
    assert fired[-1] == 1.0
    assert fired[1] == 1.0, f"expected a fire by t={t[1]}"


def test_time_greater_than_zero_still_fires():
    """``time > 0`` — nine corpus models spell "at the start" this way.

    Its residual is exactly zero at ``t_start`` too, but ``dg/dt = 1``: the
    crossing is genuine, located at ``t_start``, and unaffected.
    """
    _, fired, _ = _run(
        _sbml(
            _S,
            _RATE_S_IS_ONE,
            '<apply><gt/><csymbol encoding="text" '
            'definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
            "<cn>0</cn></apply>",
        )
    )

    assert fired[0] == 0.0
    assert fired[-1] == 1.0


def test_a_later_genuine_crossing_still_fires():
    """Starting on the threshold does not disarm the trigger for the whole run.

    ``A(0) = -1``, ``dA/dt = 2``, ``dS/dt = A`` gives ``S = t^2 - t``: the
    residual starts exactly on zero with ``dg/dt = -1``, dips to the false side,
    and crosses back up transversally at ``t = 1``. That crossing is real and
    must fire — the suppression is one-shot and scoped to ``t_start``.
    """
    t, fired, _ = _run(
        _sbml(
            _S,
            _RATE_S_IS_A + _RATE_A_IS_TWO,
            _TRIGGER_S_GT_0,
            parameters='<parameter id="A" value="-1" constant="false"/>',
        ),
        t_end=2.0,
        n_points=201,
    )

    assert not fired[t < 0.99].any(), "fired before the crossing at t=1"
    assert fired[t > 1.01].all(), "the genuine crossing at t=1 did not fire"


def _corpus_xml() -> Path | None:
    xmls = sorted(_MODEL_DIR.glob("*.xml"))
    return xmls[0] if xmls else None


def test_biomd285_runs_its_model_instead_of_killing_the_cell():
    """The reported model: ``kalive`` must survive t=0 and the cascade must run.

    Before the fix both death events fired at ``t = 3e-27``, ``kalive`` went to
    0, and every species froze at its initial value. The values asserted here
    are AMICI's for the same horizon and tolerances.
    """
    xml = _corpus_xml()
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODEL_DIR}")

    model = bngsim.Model.from_sbml(str(xml))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 100.0), n_points=11, rtol=1e-9, atol=1e-12
    )
    obs = list(res.observable_names)
    values = np.asarray(res.observables)

    kalive = values[:, obs.index("kalive")]
    assert (kalive == 1.0).all(), "a death event fired; the cell died at t=0"

    # AMICI (SUNDIALS deactivates a root identically zero at t0, so it never
    # fires either) at t=100, rtol=1e-9 atol=1e-12.
    assert values[-1, obs.index("MisP")] == pytest.approx(38.91257262, rel=1e-6)
    assert values[-1, obs.index("AggP_Proteasome")] == pytest.approx(3.06989638e-04, rel=1e-6)
    assert values[-1, obs.index("PIdeath")] == pytest.approx(2.56821027e-10, rel=1e-6)


def test_biomd285_forward_sensitivity_no_longer_hits_the_simultaneous_fire():
    """The path the issue was reported through.

    Both death events fired at the same instant with crossing times that moved
    differently, so ``apply_event_sensitivity_jump`` refused an ambiguous
    ``dt*/dp``. With neither event firing there is nothing ambiguous left, and
    the run produces a finite sensitivity tensor.
    """
    xml = _corpus_xml()
    if xml is None:
        pytest.skip(f"rr_parity corpus model not present: {_MODEL_DIR}")

    model = bngsim.Model.from_sbml(str(xml))
    res = bngsim.Simulator(model, method="ode", sensitivity_params=["kPIdeath", "kp38death"]).run(
        t_span=(0.0, 100.0), n_points=11, rtol=1e-9, atol=1e-12
    )

    sens = np.asarray(res.sensitivities)
    assert np.isfinite(sens).all()
