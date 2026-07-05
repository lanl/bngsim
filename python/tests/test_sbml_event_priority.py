"""GH #233 — simultaneous events order by their REAL-valued ``<priority>``.

When two events trigger at the same instant, the one with the higher priority
fires first (SBML L3v2 §4.11.3); since both here assign the same variable, the
one that fires *last* determines the final value. SBML priorities are arbitrary
real numbers, but the loader constant-folded a priority into an integer field —
truncating a fractional priority (2.8 → 2) so two distinct priorities collapsed
into a tie, and the events fired in declaration order instead of priority order.

The fix routes a non-integer constant priority through the double-valued
``priority_expr`` path (which the C++ event dispatcher already compares as a
double), evaluated exactly at trigger time. Integer priorities keep the fast int
field, so every existing model — including the deterministically-ordered,
equal-priority RandomEventExecution cases — is byte-identical.

Suite cases this covers: 01714 (priority `k1` vs 2.5) and 01533 (dozens of
fractional priorities from `exp`/`arccos`/`ln`/… each compared against 0.55).
"""

import bngsim
import numpy as np
import pytest


def _sbml(events: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="prio">
    <listOfParameters>
      <parameter id="w" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>{events}</listOfEvents>
  </model>
</sbml>"""


def _event(eid: str, priority_ml: str, value: int) -> str:
    return f"""
      <event id="{eid}" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>1</cn></apply>
          </math>
        </trigger>
        <priority>
          <math xmlns="http://www.w3.org/1998/Math/MathML">{priority_ml}</math>
        </priority>
        <listOfEventAssignments>
          <eventAssignment variable="w">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn type="integer">{value}</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>"""


def _final_w(sbml: str) -> float:
    model = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 5), n_points=6, rtol=1e-10, atol=1e-12
    )
    return float(np.asarray(r.observables["w"])[-1])


def test_fractional_priority_not_truncated_to_tie():
    """Ea (priority 2.2, w=10) declared before Eb (priority 2.8, w=20). Eb has the
    higher priority so it fires FIRST; Ea fires last, so final w=10.

    Under the old int-truncation both priorities became 2 → a tie → declaration
    order → Ea first, Eb last → w=20 (wrong). The fix keeps them distinct doubles.
    """
    sbml = _sbml(
        _event("Ea", "<cn>2.2</cn>", 10) + _event("Eb", "<cn>2.8</cn>", 20)
    )
    assert _final_w(sbml) == pytest.approx(10.0)


def test_fractional_vs_integer_priority_orders_correctly():
    """01714 shape: Elo (priority 2, w=20) vs Ehi (priority 2.5, w=10). Ehi is
    higher, fires first; Elo fires last → w=20.

    Old truncation: int(2.5)=2 ties with 2 → declaration order → Elo first (w=20),
    Ehi last → w=10 (wrong). The fix orders Ehi(2.5) > Elo(2) correctly.
    """
    sbml = _sbml(
        _event("Elo", "<cn type=\"integer\">2</cn>", 20)
        + _event("Ehi", "<cn>2.5</cn>", 10)
    )
    assert _final_w(sbml) == pytest.approx(20.0)
