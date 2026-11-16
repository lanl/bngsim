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
    "a run continues a run that continued a saved state": (
        "0.5",
        [run(1), 'saveConcentrations("a")', run(2), 'resetConcentrations("a")', run(3), run(4)],
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
    # Every warning the writer gives says what the document does not hold.
    lossy = [w for w in seen if issubclass(w.category, ConversionWarning)]
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


def test_a_protocol_the_writer_cannot_hold_is_seen_as_lossy(tmp_path):
    """The guard the other tests lean on: a scan that resets at each point,
    after a run, is one the writer warns it cannot hold, and ``_write`` says
    so. It filtered on the word "lossy", which no writer warning has."""
    actions = [run(), SCAN.replace("reset_conc=>0", "reset_conc=>1"), run()]
    *_, lossy = _write(tmp_path, "0.5", actions)
    assert lossy


def test_the_sequence_saves_where_a_later_chain_continues(tmp_path):
    kr, actions = CASES["a chain after an unrelated run"]
    net, _, sbml, xml, lossy = _write(tmp_path, kr, actions)
    assert not lossy
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


# --- chains another tool wrote, read as main read them (review of #1770) --------


def _document(tasks: str, read: list[str]) -> str:
    return (
        f'<sedML xmlns="{SED}" xmlns:sbml="http://www.sbml.org/sbml/level3/version2/core" '
        'level="1" version="3"><listOfSimulations>'
        + _utc("t_a_s", 1)
        + _utc("t_b_s", 2)
        + _utc("t_c_s", 3)
        + '</listOfSimulations><listOfModels><model id="m" '
        'language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>'
        f"<listOfTasks>{tasks}</listOfTasks><listOfDataGenerators>"
        + "".join(_generator(t) for t in read)
        + "</listOfDataGenerators></sedML>"
    )


def _task(tid: str) -> str:
    return f'<task id="{tid}" modelReference="m" simulationReference="{tid}_s"/>'


_ABC = _task("t_a") + _task("t_b") + _task("t_c")

# Each: the tasks, the tasks an output reads, and the t_end of the runs it
# makes, which continue one another with no reset between.
FOREIGN = {
    "a chain adds two runs to a task already run": (
        _ABC + _rt("R", ["t_a", "t_b", "t_c"]),
        ["t_a", "R"],
        [1.0, 2.0, 3.0],
    ),
    "a chain of a chain adds two runs to a task already run": (
        _ABC + _rt("R1", ["t_a", "t_b"]) + _rt("R2", ["R1", "t_c"]),
        ["t_a", "R2"],
        [1.0, 2.0, 3.0],
    ),
    "a chain adds one run": (
        _task("t_a") + _task("t_b") + _rt("R", ["t_a", "t_b"]),
        ["t_a", "R"],
        [1.0, 2.0],
    ),
    "a chain listed before the task it starts with": (
        _task("t_b") + _rt("R", ["t_a", "t_b"]) + _task("t_a"),
        ["t_a", "R"],
        [1.0, 2.0],
    ),
    "a chain listed before a shorter chain and the task they start with": (
        _rt("R", ["t_a", "t_b", "t_c"]) + _rt("R1", ["t_a", "t_b"]) + _ABC,
        ["t_a", "R1", "R"],
        [1.0, 2.0, 3.0],
    ),
    "a shorter chain listed before the longer": (
        _ABC + _rt("R1", ["t_a", "t_b"]) + _rt("R", ["t_a", "t_b", "t_c"]),
        ["R1", "R"],
        [1.0, 2.0, 3.0],
    ),
}


@pytest.mark.parametrize("case", list(FOREIGN))
def test_a_foreign_chain_runs_each_of_its_tasks_once(tmp_path, case):
    """A task is run once however many chains start with it, and wherever the
    document lists it: SED-ML gives the order of listOfTasks no meaning. The
    chain's runs continue the task's, with no reset and no save between."""
    tasks, read, t_ends = FOREIGN[case]
    net = tmp_path / "m.net"
    net.write_text(NET.format(kr="0.5"), encoding="utf-8")
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    spec = read_sedml_protocol(_document(tasks, read), sbml=sbml)
    assert spec.lossy == ()
    assert [getattr(s, "kind", None) for s in spec.steps] == ["simulate"] * len(t_ends)
    assert [e.t_span[1] for e in spec.experiments] == t_ends
    want = parse_bngl_protocol(
        "begin model\nend model\ngenerate_network()\n" + "\n".join(run(t) for t in t_ends) + "\n"
    )
    for x, y in zip(_runs(want, net), _runs(spec, net), strict=True):
        assert np.array_equal(x, y)


def test_a_chain_of_tasks_never_run_continues_equal_steps(tmp_path):
    """t_a2 has t_a's settings and is not t_a. The chain R = [t_a, t_b] names
    t_a, which no output reads and nothing ran; its replay names the state
    t_a2's equal run reached, and R's second run continues it, as main read
    the two: an ode run from one state with one setting is one run."""
    tasks = (
        _task("t_a")
        + _task("t_b")
        + '<task id="t_a2" modelReference="m" simulationReference="t_a_s"/>'
        + _rt("R", ["t_a", "t_b"])
    )
    net = tmp_path / "m.net"
    net.write_text(NET.format(kr="0.5"), encoding="utf-8")
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    spec = read_sedml_protocol(_document(tasks, ["t_a2", "R"]), sbml=sbml)
    assert spec.lossy == ()
    assert [getattr(s, "kind", None) for s in spec.steps] == ["simulate", "simulate"]
    assert [e.t_span[1] for e in spec.experiments] == [1.0, 2.0]


def test_a_chain_of_one_task_beside_that_task_is_a_run_of_its_own(tmp_path):
    """R = [t_a] and t_a, both read: R is the start of no longer chain, so it
    runs, from the initial state again, as main read the two."""
    net = tmp_path / "m.net"
    net.write_text(NET.format(kr="0.5"), encoding="utf-8")
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    spec = read_sedml_protocol(
        _document(_task("t_a") + _rt("R", ["t_a"]), ["t_a", "R"]), sbml=sbml
    )
    assert spec.lossy == ()
    assert [getattr(s, "kind", None) for s in spec.steps] == [
        "save_parameters",
        "simulate",
        "reset_concentrations",
        "reset_parameters",
        "simulate",
    ]
    out = _runs(spec, net)
    assert np.array_equal(out[0], out[1])


def test_a_chain_continues_where_the_steps_are_though_that_state_is_saved(tmp_path):
    """C and D both continue t_a, so t_a's state is saved for D. R runs t_c,
    resets and runs t_a again, and E, which continues t_a, is listed after R:
    the steps are in that state, and E's run continues them, with no reset to
    the save."""
    tasks = (
        _ABC
        + _rt("C", ["t_a", "t_b"])
        + _rt("D", ["t_a", "t_c"])
        + _rt("r_a", ["t_a"])
        + _rt("R", ["t_c", "r_a"])
        + _rt("E", ["t_a", "t_b"])
    )
    net = tmp_path / "m.net"
    net.write_text(NET.format(kr="0.5"), encoding="utf-8")
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    spec = read_sedml_protocol(_document(tasks, ["t_a", "C", "D", "R", "E"]), sbml=sbml)
    assert spec.lossy == ()
    assert [e.t_span[1] for e in spec.experiments] == [1.0, 2.0, 3.0, 3.0, 1.0, 2.0]
    assert [getattr(s, "kind", None) for s in spec.steps[-4:]] == [
        "reset_concentrations",
        "reset_parameters",
        "simulate",
        "simulate",
    ]
    assert spec.steps[-4].label is None  # to the initial state, inside R


# --- no save where the steps already are, one save for one state ----------------


def _chain_labels(steps) -> list[tuple[str, str]]:
    return [
        (s.kind, s.label)
        for s in steps
        if str(getattr(s, "label", None) or "").startswith("__sedml_chain_")
    ]


def test_a_run_after_a_scan_with_a_follower_continues_it(tmp_path):
    """kr follows kf, so the scan's task leaves kr set to its last point's
    value: a step after the lead's run. The run after it continues where the
    steps are; it was read as a state reached earlier, saved and reset to,
    with kr set twice."""
    net, protocol, sbml, xml, lossy = _write(tmp_path, "kf*3", [SCAN, run()])
    assert not lossy
    _check(net, sbml, xml)
    bare = re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S)
    steps = read_sedml_protocol(bare, sbml=sbml).steps
    assert _chain_labels(steps) == []
    assert [getattr(s, "kind", None) for s in steps] == [
        "set_parameter",
        "scan",
        "set_parameter",
        "simulate",
    ]


def test_every_return_to_one_state_resets_to_one_save(tmp_path):
    """Three runs continue the first: its state is saved once, and each of
    them resets to that label. Each return saved the state again, under a new
    label, right after resetting to the last."""
    actions = [run(1), 'saveConcentrations("a")']
    for t in (2, 3, 4):
        actions += [run(t), 'resetConcentrations("a")']
    actions.append(run(5))
    net, _, sbml, xml, lossy = _write(tmp_path, "0.5", actions)
    assert not lossy
    _check(net, sbml, xml)
    bare = re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S)
    steps = read_sedml_protocol(bare, sbml=sbml).steps
    label = "__sedml_chain_0"
    assert _chain_labels(steps) == [
        ("save_concentrations", label),
        ("save_parameters", label),
        *[("reset_concentrations", label), ("reset_parameters", label)] * 3,
    ]


def test_a_chain_continues_the_run_it_names_not_an_equal_one(tmp_path):
    """Two unseeded ssa runs with one setting end in two states. The third run
    continues the first, which its chain names; it was read as continuing the
    second, the state the steps were in, whose steps are equal."""
    ssa = 'simulate({method=>"ssa",t_end=>1,n_steps=>2})'
    actions = [ssa, 'saveConcentrations("a")', "resetConcentrations()", ssa]
    actions += ['resetConcentrations("a")', run(2)]
    net, _, sbml, xml, lossy = _write(tmp_path, "0.5", actions)
    assert not lossy
    bare = re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S)
    spec = read_sedml_protocol(bare, sbml=sbml)
    assert spec.lossy == ()
    kinds = [getattr(s, "kind", None) for s in spec.steps]
    first = kinds.index("simulate")
    assert kinds[first + 1 : first + 3] == ["save_concentrations", "save_parameters"]
    label = spec.steps[first + 1].label
    third = len(kinds) - 1
    assert kinds[third] == "simulate"
    assert [(s.kind, s.label) for s in spec.steps[third - 2 : third]] == [
        ("reset_concentrations", label),
        ("reset_parameters", label),
    ]
    out = _runs(spec, net)
    assert len(out) == 3
    assert np.array_equal(out[2][0], out[0][-1])
