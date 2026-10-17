"""GH #1067: NFsim refuses a compartment model.

NFsim's compartment support runs compartment models wrong with no error, in
ways a check of the XML cannot bound. Against BNG's generated network:

* a species-level compartment on a pattern is ignored: ``Molecules @EC:A()``
  counted every A (300 against 200), ``@EC:A() -> 0`` degraded the A in other
  compartments too (Atot 2 against 101), ``0 -> @EC:A2()`` lost its volume
  factor (RuleWorld/nfsim#106);
* no volume factor on a rule with two or more reactants: AB 133.6 against 100
  at size 2 (RuleWorld/nfsim#108);
* a molecule a rule creates with no compartment written gets none: ``C@EC()``
  0 against 86.5 (RuleWorld/nfsim#109);
* a seed complex puts every molecule in the species' compartment, binding
  ignores compartment adjacency, and MoveConnected moves the whole complex
  (RuleWorld/nfsim#110);
* BNG2.pl's writeXML drops NumberPerQuantityUnit, so no check can see it
  (RuleWorld/bionetgen#348).

Two review rounds of a narrower envelope kept missing cases, so NfsimSession
and Simulator(method='nf') refuse every model that declares compartments, naming
them, until the fixes are vendored. #854's zero-order volume refresh is still
tested through the core NfsimSimulator (test_nfsim_session.py,
test_nfsim_rate_refresh.py).
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import compartments

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False),
    reason="NFsim not compiled in",
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"

COMPARTMENT_MODELS = [
    ("pattern_compartment_observable.xml", "'EC', 'PM', 'CP'"),
    ("pattern_compartment_rule.xml", "'EC', 'PM', 'CP'"),
    ("pattern_compartment_synthesis.xml", "'EC'"),
    ("compartment_bimolecular_size2.xml", "'EC'"),
    ("compartment_created_molecule.xml", "'EC', 'PM', 'CP'"),
    ("molecule_compartment_degradation.xml", "'EC', 'PM', 'CP'"),
    ("zero_order_switchable_rate_compartment.xml", "'c'"),
    ("compartment_test/simple_compartment_system.xml", None),
]


@pytest.mark.parametrize("fixture, names", COMPARTMENT_MODELS)
def test_the_session_refuses_a_compartment_model(fixture, names):
    named = rf"compartments {names}\)" if names else "compartments "
    match = rf"cannot run a compartment model \({named}.*RuleWorld/nfsim#106.*#1067"
    with pytest.raises(ModelError, match=match):
        bngsim.NfsimSession(str(DATA / fixture))


@pytest.mark.parametrize("fixture, names", COMPARTMENT_MODELS)
def test_the_simulator_refuses_a_compartment_model(fixture, names, simple_decay_net):
    m = bngsim.Model.from_net(str(simple_decay_net))
    with pytest.raises(ModelError, match="cannot run a compartment model"):
        bngsim.Simulator(m, method="nf", xml_path=str(DATA / fixture))


def test_a_model_without_compartments_runs():
    xml = DATA / "zero_order_switchable_rate.xml"
    assert compartments(xml) == []
    bngsim.NfsimSession(str(xml)).destroy()
