"""The runs a SED-ML document's tasks make, read back in order and once each (issue #1739).

``write_sedml_protocol`` writes a run that continues earlier ones as a chain
(a ``repeatedTask`` that resets and replays the runs before it), beside each
earlier run's own task. ``read_sedml_protocol``, reading the standard elements,
dropped every task whose steps were a prefix of another's. That dropped a run
that started from the initial state again (it repeats a chain's first run),
ran a replay two chains share twice, and put a chain after an unrelated run
that came before it. Each read task is now run in order: from where the steps
are, from a state they reached before (saved there), or from the initial state.

Oracle: the same document read with its bngsim annotation, which carries the
protocol exactly. Read from the standard elements alone, it must run the same
experiments, in the same order, to the bit.
"""

from __future__ import annotations

import random
import re
import warnings

import bngsim
import numpy as np
import pytest
from bngsim import ConversionWarning
from bngsim.convert import parse_bngl_protocol, read_sedml_protocol, write_sedml_protocol

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SED-ML replay requires libsbml")

NET = """begin parameters
    1 kf 1
    2 kr {kr}
    3 A0 100
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
begin groups
    1 Atot 1
end groups
"""


def run(t_end: float = 1.0) -> str:
    return f'simulate({{method=>"ode",t_end=>{t_end},n_steps=>2}})'


SCAN = (
    'parameter_scan({method=>"ode",parameter=>"kf",par_scan_vals=>[1,2],reset_conc=>0,'
    "t_end=>1,n_steps=>2})"
)

# The issue's four protocols, on this network (the issue's k1, k2 are kf, kr).
CASES = {
    "a run from the start again repeats a chain's first run": (
        "0.5",
        [run(), run(), "resetConcentrations()", run()],
    ),
    "two chains share a first run (the scan)": (
        "0.5",
        [
            SCAN,
            "saveConcentrations()",
            run(),
            "resetConcentrations()",
            'setConcentration("A()",50)',
            run(),
        ],
    ),
    "a reset to a derived parameter's expression before a chain": (
        "kf*3",
        ["saveParameters()", 'setParameter("kr",7)', "resetParameters()", run(), run()],
    ),
    "two saved states, reset to in turn": (
        "0.5",
        [
            run(1),
            'saveConcentrations("a")',
            run(2),
            'saveConcentrations("b")',
            "resetConcentrations()",
            run(3),
            'resetConcentrations("a")',
            run(4),
            'resetConcentrations("b")',
            run(5),
        ],
    ),
    "a chain after an unrelated run": (
        "0.5",
        [
            run(1),
            'saveConcentrations("c1")',
            "resetConcentrations()",
            run(2),
            'resetConcentrations("c1")',
            run(3),
        ],
    ),
}


def _write(tmp_path, kr: str, actions: list[str]):
    net = tmp_path / "m.net"
    net.write_text(NET.format(kr=kr))
    text = "begin model\nend model\ngenerate_network()\n" + "\n".join(actions) + "\n"
    protocol = parse_bngl_protocol(text)
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConversionWarning)
        xml = write_sedml_protocol(
            protocol, model_source="model.xml", sbml=sbml, model=bngsim.Model.from_net(str(net))
        )
    lossy = [w for w in seen if "lossy" in str(w.message).lower()]
    return net, protocol, sbml, xml, lossy


def _runs(protocol, net) -> list[np.ndarray]:
    out = []
    for r in bngsim.run_protocol(bngsim.Model.from_net(str(net)), protocol):
        if r.result is not None:
            out.append(np.asarray(r.result.species))
        else:
            out.extend(np.asarray(x.species) for x in (*r.scan_results, *r.backward_results))
    return out


def _signature(spec) -> list[tuple]:
    return [(e.kind, e.t_span[1]) for e in spec.experiments]


def _check(net, sbml, xml) -> tuple[int, list[tuple]]:
    exact = read_sedml_protocol(xml)
    bare = re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S)
    assert "<protocol" not in bare
    back = read_sedml_protocol(bare, sbml=sbml)
    assert back.lossy == ()
    assert _signature(back) == _signature(exact)
    a, b = _runs(exact, net), _runs(back, net)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    return len(a), _signature(back)


@pytest.mark.parametrize("case", list(CASES))
def test_the_issues_protocols_read_back_as_written(tmp_path, case):
    kr, actions = CASES[case]
    net, protocol, sbml, xml, lossy = _write(tmp_path, kr, actions)
    assert not lossy
    _, sig = _check(net, sbml, xml)
    assert sig == [(e.kind, e.t_span[1]) for e in protocol.experiments]


def test_the_sequence_saves_where_a_later_chain_continues(tmp_path):
    kr, actions = CASES["a chain after an unrelated run"]
    net, _, sbml, xml, _ = _write(tmp_path, kr, actions)
    bare = re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S)
    steps = read_sedml_protocol(bare, sbml=sbml).steps
    kinds = [getattr(s, "kind", None) for s in steps]
    first_run = kinds.index("simulate")
    # A save of the state right after the first run, and a reset to it before
    # the third, with the same label.
    save = next(i for i, k in enumerate(kinds) if k == "save_concentrations")
    assert save == first_run + 1
    label = steps[save].label
    resets = [s for s in steps if getattr(s, "kind", None) == "reset_concentrations"]
    assert any(s.label == label for s in resets)
    assert kinds.count("simulate") == 3


# --- randomized protocols ------------------------------------------------------


def _random_actions(rng: random.Random) -> list[str]:
    actions: list[str] = []
    saved = {"": False, '"s1"': False}
    for _ in range(rng.randint(2, 7)):
        roll = rng.random()
        if roll < 0.35 or not actions:
            actions.append(run(rng.choice([1, 2])))
        elif roll < 0.5:
            actions.append(f'setConcentration("A()",{rng.choice([20, 50, 80])})')
        elif roll < 0.6:
            actions.append(f'setParameter("kf",{rng.choice([0.5, 2, 3])})')
        elif roll < 0.75:
            label = rng.choice(list(saved))
            saved[label] = True
            actions.append(f"saveConcentrations({label})")
        else:
            label = rng.choice([k for k, v in saved.items() if v] or [""])
            actions.append(f"resetConcentrations({label})")
    if not any(a.startswith("simulate") for a in actions[-2:]):
        actions.append(run())
    return actions


@pytest.mark.parametrize("seed", range(40))
def test_random_protocols_read_back_as_written(tmp_path, seed):
    rng = random.Random(1739 + seed)
    actions = _random_actions(rng)
    net, _, sbml, xml, lossy = _write(tmp_path, "0.5", actions)
    # Every seed here is one the writer writes losslessly; a protocol that is
    # not would be no test of the reader.
    assert not lossy
    _check(net, sbml, xml)


# --- a document another tool wrote: a resetting repeatedTask inside a chain ----

SED = "http://sed-ml.org/sed-ml/level1/version3"


def _utc(sid: str, t_end: float) -> str:
    return (
        f'<uniformTimeCourse id="{sid}" initialTime="0" outputStartTime="0" '
        f'outputEndTime="{t_end}" numberOfPoints="2"><algorithm kisaoID="KISAO:0000019"/>'
        "</uniformTimeCourse>"
    )


def _rt(rid: str, subtasks: list[str]) -> str:
    subs = "".join(f'<subTask order="{i}" task="{t}"/>' for i, t in enumerate(subtasks, 1))
    return (
        f'<repeatedTask id="{rid}" resetModel="true" range="one"><listOfRanges>'
        '<vectorRange id="one"><value>0</value></vectorRange></listOfRanges>'
        f"<listOfSubTasks>{subs}</listOfSubTasks></repeatedTask>"
    )


def _generator(task: str) -> str:
    return (
        f'<dataGenerator id="dg_{task}"><listOfVariables><variable id="v_{task}" '
        f'taskReference="{task}" target="/sbml:sbml/sbml:model/sbml:listOfSpecies/'
        f"sbml:species[@id='A']\"/></listOfVariables><math xmlns=\"http://www.w3.org/1998/"
        f'Math/MathML"><ci>v_{task}</ci></math></dataGenerator>'
    )


def test_a_reset_inside_a_chain_starts_its_state_again(tmp_path):
    """r_b resets and runs B; chain_y runs A then r_b, so its own run is B from
    the start; chain_w runs A, r_b and C, so its replay ends at B from the
    start, which only the steps since its reset name, and C continues B.
    Read: A, B from the start, C from where B left, which is BNGL's run,
    reset, run, run."""
    net = tmp_path / "m.net"
    net.write_text(NET.format(kr="0.5"))
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    tasks = "".join(
        f'<task id="{t}" modelReference="m" simulationReference="{t}_s"/>'
        for t in ("t_a", "t_b", "t_c")
    )
    doc = (
        f'<sedML xmlns="{SED}" xmlns:sbml="http://www.sbml.org/sbml/level3/version2/core" '
        'level="1" version="3"><listOfSimulations>'
        + _utc("t_a_s", 1)
        + _utc("t_b_s", 2)
        + _utc("t_c_s", 3)
        + '</listOfSimulations><listOfModels><model id="m" '
        'language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>'
        f"<listOfTasks>{tasks}"
        + _rt("r_b", ["t_b"])
        + _rt("chain_y", ["t_a", "r_b"])
        + _rt("chain_w", ["t_a", "r_b", "t_c"])
        + "</listOfTasks><listOfDataGenerators>"
        + "".join(_generator(t) for t in ("t_a", "chain_y", "chain_w"))
        + "</listOfDataGenerators></sedML>"
    )
    spec = read_sedml_protocol(doc, sbml=sbml)
    assert spec.lossy == ()
    assert _signature(spec) == [("simulate", 1.0), ("simulate", 2.0), ("simulate", 3.0)]
    want = parse_bngl_protocol(
        "begin model\nend model\ngenerate_network()\n"
        + "\n".join([run(1), "resetConcentrations()", run(2), run(3)])
        + "\n"
    )
    for x, y in zip(_runs(want, net), _runs(spec, net), strict=True):
        assert np.allclose(x, y, rtol=1e-12, atol=1e-12)
