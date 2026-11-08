"""A protocol's SED-ML says which experiments it cannot start where BNG2.pl does.

``write_sedml_protocol`` starts every experiment, in SED-ML's standard elements,
from the model's initial state with the protocol's numeric ``setConcentration``
and ``setParameter`` values baked into a derived model. BNG2.pl starts each
from the state the steps before it left: an equilibrate-then-stimulate protocol
runs its stimulation from the equilibrated state, where a standards tool ran it
from the initial one, with nothing said. A ``repeatedTask`` whose ordered
subtasks chain the runs does not carry it either: tellurium 2.2.13 resets the
model between subtasks. So the writer names each experiment whose start differs,
and why, and the bngsim annotation keeps the protocol exact.
"""

from __future__ import annotations

import re
import warnings

import pytest
from bngsim import ConversionWarning
from bngsim.convert import parse_bngl_protocol, read_sedml_protocol, write_sedml_protocol
from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

CARRIED = "standard elements start each experiment"


def _sim() -> Experiment:
    return Experiment(kind="simulate", method="ode", t_span=(0.0, 10.0), n_points=11)


def _scan(*, reset: bool = True, parameter: str = "k") -> Experiment:
    return Experiment(
        kind="scan",
        method="ode",
        t_span=(0.0, 10.0),
        n_points=11,
        scan_parameter=parameter,
        scan_min=1.0,
        scan_max=3.0,
        scan_points=3,
        reset_between=reset,
    )


def _carried(protocol: ProtocolSpec) -> str | None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_sedml_protocol(protocol)
    said = [str(w.message) for w in caught if CARRIED in str(w.message)]
    assert len(said) <= 1
    return said[0] if said else None


@pytest.mark.parametrize(
    ("steps", "start"),
    [
        pytest.param(
            [_sim(), StateChange("set_concentration", "L()", 5.0), _sim()],
            "experiment 1 from the concentrations left by experiment 0's run",
            id="equilibrate-then-stimulate",
        ),
        pytest.param(
            [StateChange("add_concentration", "A()", 5.0), _sim()],
            'experiment 0 from the concentrations left by addConcentration("A()",5)',
            id="add-is-relative",
        ),
        pytest.param(
            [StateChange("set_parameter", "k", None, "2*k0"), _sim()],
            'experiment 0 from the parameters left by setParameter("k","2*k0"), which needs',
            id="value-needs-the-model",
        ),
        pytest.param(
            [_scan(), _sim()],
            "experiment 1 from parameters k=3 (where experiment 0's scan left it)",
            id="scan-leaves-its-parameter",
        ),
        pytest.param(
            [
                StateChange("set_concentration", "A()", 5.0),
                StateChange("save_concentrations"),
                StateChange("set_concentration", "A()", 7.0),
                StateChange("reset_concentrations"),
                _sim(),
            ],
            "experiment 0 from concentrations A()=5",
            id="reset-restores-a-saved-set",
        ),
        pytest.param(
            [_sim(), StateChange("save_concentrations", label="eq"), _sim()]
            + [StateChange("reset_concentrations", label="eq"), _sim()],
            "experiment 1 from the concentrations left by experiment 0's run; "
            "experiment 2 from the concentrations left by experiment 0's run",
            id="reset-to-a-post-run-save",
        ),
    ],
)
def test_a_start_bng2_carries_is_named(steps, start):
    said = _carried(ProtocolSpec(steps=tuple(steps)))
    assert said is not None
    assert start in said


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([_sim()], id="one-run"),
        pytest.param([_sim(), StateChange("reset_concentrations"), _sim()], id="reset-between"),
        pytest.param(
            [StateChange("set_concentration", "A()", 5.0), _sim()], id="numeric-set-is-baked-in"
        ),
        pytest.param([_scan(), _scan()], id="a-scan-sets-its-own-parameter"),
        pytest.param(
            [_scan(), StateChange("set_parameter", "k", 1.0), _sim()], id="parameter-set-back"
        ),
        pytest.param(
            [_scan(parameter="p"), StateChange("set_parameter", "p", 1.0), _scan()],
            id="reset-scans-end-where-they-start",
        ),
    ],
)
def test_a_start_the_standard_elements_describe_is_not_flagged(steps):
    assert _carried(ProtocolSpec(steps=tuple(steps))) is None


def test_more_than_five_carried_starts_are_counted():
    steps = [_sim()]
    for _ in range(7):
        steps += [StateChange("set_concentration", "L()", 5.0), _sim()]
    said = _carried(ProtocolSpec(steps=tuple(steps)))
    assert said is not None
    assert len(re.findall(r"experiment \d from", said)) == 5
    assert "and 2 more experiment(s) from a carried state" in said


MODEL = "begin model\nend model\n"


def test_steps_before_the_network_is_generated_are_in_it():
    # The network the SBML is written from already holds what came before
    # generate_network, so only what follows it moves the state.
    before = MODEL + (
        'setConcentration("A()",5)\naddConcentration("A()",1)\n'
        'setParameter("k","2*k0")\ngenerate_network({overwrite=>1})\n'
        'simulate({method=>"ode",t_end=>10,n_steps=>10})\n'
    )
    assert _carried(parse_bngl_protocol(before)) is None
    after = MODEL + (
        'generate_network({overwrite=>1})\naddConcentration("A()",1)\n'
        'simulate({method=>"ode",t_end=>10,n_steps=>10})\n'
    )
    said = _carried(parse_bngl_protocol(after))
    assert said is not None
    assert 'addConcentration("A()",1)' in said


def test_the_annotation_still_holds_the_protocol_exactly():
    protocol = ProtocolSpec(steps=(_sim(), StateChange("set_concentration", "L()", 5.0), _sim()))
    with pytest.warns(ConversionWarning, match=CARRIED):
        text = write_sedml_protocol(protocol)
    assert read_sedml_protocol(text) == protocol
