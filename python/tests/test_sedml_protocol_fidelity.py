"""A BNGL protocol through SED-ML and OMEX, and back (issues #1663, #1664, #1668,
#1677, #1678, #1682, #1684, #1685, #1686, #1687).

Each test holds an export or a replay to a closed form. The models are
A <-> B at k1, k2 from A = 100, B = 0, where
A(t) = Aeq + (A0 - Aeq) e^(-(k1 + k2) t) with Aeq = A0 k2 / (k1 + k2), and
A -> 0 at k, where A(t) = A0 e^(-k t). The SED-ML a test exports is run two
ways: by bngsim's own reader of the standard elements, and by
:func:`_run_report`, a reading of SED-ML L1V3 written here on libRoadRunner,
which shares no code with bngsim's. The numbers BNG2.pl 2.9.3 gives for the
same files are in the issues.
"""

from __future__ import annotations

import copy
import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import bngsim
import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import (
    ProtocolSpec,
    StateChange,
    combine_protocols,
    net_to_omex,
    omex_to_net,
    parse_bngl_protocol,
    read_omex,
    read_sedml,
    read_sedml_protocol,
    write_bngl_protocol,
    write_sedml_protocol,
)

SED = "http://sed-ml.org/sed-ml/level1/version3"
NS = {"s": SED, "sbml": "http://www.sbml.org/sbml/level3/version2/core"}
MATHML = "http://www.w3.org/1998/Math/MathML"
RTOL = 1e-6

# A <-> B, k1 = k2 = 1, A(0) = 100, B(0) = 0, as BNG2.pl's generate_network
# writes it, and the model block of the .bngl it is generated from.
NET = """begin parameters
    1 k1 1
    2 k2 1
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1 2 k1
    2 2 1 k2
end reactions
begin groups
    1 Atot 1
end groups
"""
MODEL = """begin model
begin parameters
  k1 1
  k2 1
end parameters
begin seed species
  A() 100
  B() 0
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() <-> B() k1, k2
end reaction rules
end model
"""
RUN = 'simulate({method=>"ode",t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n'


def _eq(k1: float, k2: float, a0: float = 100.0, t: float = 1.0) -> float:
    """A(t) of A <-> B from A = a0, B = 0."""
    aeq = a0 * k2 / (k1 + k2)
    return aeq + (a0 - aeq) * math.exp(-(k1 + k2) * t)


def _files(tmp_path: Path, actions: str, net: str = NET, model: str = MODEL):
    (tmp_path / "m.net").write_text(net)
    (tmp_path / "m.bngl").write_text(model + actions)
    return tmp_path / "m.net", tmp_path / "m.bngl"


def _pack(tmp_path: Path, actions: str, **kw):
    """The archive net_to_omex writes for *actions*, and the warnings it raised."""
    net, bngl = _files(tmp_path, actions, **{k: v for k, v in kw.items() if k in ("net", "model")})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = net_to_omex(net, tmp_path / "m.omex", bngl=bngl, gate="none")
    return report, [str(w.message) for w in caught if issubclass(w.category, ConversionWarning)]


def _replay(net_text: str, protocol, tmp_path: Path, **kw):
    path = tmp_path / "replay.net"
    path.write_text(net_text)
    return bngsim.run_protocol(bngsim.Model.from_net(str(path)), protocol, **kw)


def _end(result, name: str = "A") -> float:
    names = list(result.species_names)
    hit = next(i for i, n in enumerate(names) if n in (name, f"{name}()"))
    return float(result.species[-1, hit])


# ─── an independent reading of the SED-ML, on libRoadRunner ─────────────────


def _formula(math_elem: ET.Element) -> str:
    import libsbml

    node = libsbml.readMathMLFromString(ET.tostring(math_elem, encoding="unicode"))
    return libsbml.formulaToString(node)


def _piecewise(*args: float) -> float:
    """piecewise(value, condition, ..., otherwise)."""
    return next((v for v, c in zip(args[::2], args[1::2], strict=False) if c), args[-1])


def _evaluate(math_elem: ET.Element, env: dict[str, float]) -> float:
    names = {
        "pow": pow,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "lt": lambda a, b: a < b,
        "gt": lambda a, b: a > b,
        "piecewise": _piecewise,
    }
    return float(eval(_formula(math_elem), {"__builtins__": {}}, {**names, **env}))


def _target_value(sbml: ET.Element, target: str) -> float:
    path, attr = (target.rsplit("/@", 1) + [None])[:2]
    elem = sbml.find(path.replace("/sbml:sbml/", "./"), NS)
    if attr is None:
        attr = "value" if elem.tag.endswith("parameter") else "initialConcentration"
    return float(elem.get(attr))


def _rr_key(target: str) -> str:
    m = re.search(r"(species|parameter)\[@id='([^']+)'\]", target)
    return f"[{m.group(2)}]" if m.group(1) == "species" else m.group(2)


def _run_report(omex: Path) -> dict[str, dict[str, float]]:
    """Each report of the archive's SED-ML: its data sets' values at the end of
    the task each reads, by SED-ML L1V3's semantics (a task starts from its
    model as changed, changes applied in order on the model as changed so far;
    a repeatedTask resets where resetModel says so, applies its setValues and
    runs its subtasks in order on one model state)."""
    return {
        rid: {label: points[-1] for label, points in values.items()}
        for rid, values in _run_points(omex).items()
    }


def _run_points(omex: Path) -> dict[str, dict[str, list[float]]]:
    """:func:`_run_report`, with each data set's value at the end of each
    iteration of the task it reads (one value for a task)."""
    roadrunner = pytest.importorskip("roadrunner")
    z = zipfile.ZipFile(omex)
    sed = ET.fromstring(z.read("simulation.sedml"))
    base = ET.fromstring(z.read("model.xml"))
    ET.register_namespace("", NS["sbml"])
    sims = {u.get("id"): u for u in sed.iter(f"{{{SED}}}uniformTimeCourse")}
    models = {m.get("id"): m for m in sed.iter(f"{{{SED}}}model")}
    tasks = {t.get("id"): t for t in sed.find(f"{{{SED}}}listOfTasks")}

    def sbml_of(model_id: str) -> str:
        chain = []
        while model_id in models:
            chain.append(models[model_id])
            model_id = models[model_id].get("source")
        doc = ET.fromstring(ET.tostring(base))
        for model in reversed(chain):
            for change in model.iterfind("s:listOfChanges/*", NS):
                target = change.get("target")
                if change.tag.endswith("removeXML"):
                    parent = doc.find(target.rsplit("/", 1)[0].replace("/sbml:sbml/", "./"), NS)
                    parent.remove(doc.find(target.replace("/sbml:sbml/", "./"), NS))
                    continue
                if change.tag.endswith("changeAttribute"):
                    value = float(change.get("newValue"))
                else:
                    env = {
                        v.get("id"): _target_value(doc, v.get("target"))
                        for v in change.iterfind("s:listOfVariables/s:variable", NS)
                    }
                    value = _evaluate(change.find(f"{{{MATHML}}}math"), env)
                path, attr = target.rsplit("/@", 1)
                doc.find(path.replace("/sbml:sbml/", "./"), NS).set(attr, repr(value))
        return ET.tostring(doc, encoding="unicode")

    def run(task: ET.Element, rr, each=None):
        if task.tag.endswith("}task"):
            if rr is None:
                rr = roadrunner.RoadRunner(sbml_of(task.get("modelReference")))
            rr.integrator.relative_tolerance = 1e-10
            rr.integrator.absolute_tolerance = 1e-12
            utc = sims[task.get("simulationReference")]
            rr.simulate(float(utc.get("initialTime")), float(utc.get("outputEndTime")), 2)
            return rr
        ranges = {}
        for rng in task.iterfind("s:listOfRanges/*", NS):
            if rng.tag.endswith("vectorRange"):
                ranges[rng.get("id")] = [float(v.text) for v in rng]
            else:
                n, lo, hi = (
                    int(rng.get("numberOfPoints")),
                    float(rng.get("start")),
                    float(rng.get("end")),
                )
                if rng.get("type") == "log":
                    points = [lo * (hi / lo) ** (k / n) for k in range(n + 1)]
                else:
                    points = [lo + k * (hi - lo) / n for k in range(n + 1)]
                ranges[rng.get("id")] = points
        subs = sorted(
            task.iterfind("s:listOfSubTasks/s:subTask", NS), key=lambda s: int(s.get("order"))
        )
        first = tasks[subs[0].get("task")]
        while not first.tag.endswith("}task"):
            first = tasks[first.find("s:listOfSubTasks/s:subTask", NS).get("task")]
        if rr is None:
            rr = roadrunner.RoadRunner(sbml_of(first.get("modelReference")))
        for k in range(len(ranges[task.get("range")])):
            if task.get("resetModel") == "true":
                rr.resetToOrigin()
            for sv in task.iterfind("s:listOfChanges/s:setValue", NS):
                env = {rid: vals[k] for rid, vals in ranges.items()}
                for v in sv.iterfind("s:listOfVariables/s:variable", NS):
                    env[v.get("id")] = rr[_rr_key(v.get("target"))]
                rr[_rr_key(sv.get("target"))] = _evaluate(sv.find(f"{{{MATHML}}}math"), env)
            for sub in subs:
                run(tasks[sub.get("task")], rr)
            if each is not None:
                each(rr)
        return rr

    generators = {g.get("id"): g for g in sed.iter(f"{{{SED}}}dataGenerator")}
    out: dict[str, dict[str, list[float]]] = {}
    for report in sed.iter(f"{{{SED}}}report"):
        values: dict[str, list[float]] = {}
        for ds in report.iter(f"{{{SED}}}dataSet"):
            var = generators[ds.get("dataReference")].find(".//s:variable", NS)
            points: list[float] = []

            def read(rr, var=var, points=points):
                points.append(
                    float(rr.getCurrentTime())
                    if var.get("symbol")
                    else rr[_rr_key(var.get("target"))]
                )

            task = tasks[var.get("taskReference")]
            rr = run(task, None, read)
            if task.tag.endswith("}task"):
                read(rr)
            values[ds.get("label")] = points
        out[report.get("id")] = values
    return out


# ─── #1687: an actions block with no model runs on the network it is given ──


@pytest.mark.parametrize(
    ("actions", "want"),
    [
        ('setConcentration("A()",50)\n' + RUN, _eq(1, 1, a0=50)),
        ('begin actions\nsetConcentration("A()",50)\n' + RUN + "end actions\n", _eq(1, 1, a0=50)),
        ('setParameter("k2",3)\n' + RUN, _eq(1, 3)),
    ],
    ids=["set-concentration", "actions-block", "set-parameter"],
)
def test_an_actions_block_alone_runs_every_action_on_the_network(tmp_path, actions, want):
    """The block omex_to_net writes beside its .net: its first run generates
    nothing, so the set before it applies. It was skipped as already in the
    network (A(1) = 36.79 for 18.39 at A = 50)."""
    protocol = parse_bngl_protocol(actions)
    assert protocol.generation_step == 0
    (run,) = _replay(NET, protocol, tmp_path)
    assert _end(run.result) == pytest.approx(want, rel=RTOL)


def test_a_model_file_still_generates_its_network_at_its_first_run():
    """With a model block the first run generates the network, which holds
    what the actions before it set (issue #1366)."""
    assert parse_bngl_protocol(MODEL + 'setParameter("k2",3)\n' + RUN).generation_step == 1
    assert parse_bngl_protocol("begin model\nend model\n" + RUN).generation_step == 0


def test_a_network_read_whole_is_there_from_the_read_on():
    protocol = parse_bngl_protocol(
        'setParameter("k2",3)\nreadFile({file=>"m.net"})\nsetParameter("k1",2)\n' + RUN
    )
    assert protocol.generation_step == 1
    model_read = parse_bngl_protocol('readFile({file=>"m.bngl"})\nsetParameter("k1",2)\n' + RUN)
    assert model_read.generation_step == 1  # a model, whose first run generates


def test_omex_to_net_writes_the_actions_for_the_network_it_writes(tmp_path):
    """The archive's protocol generated its network at its first run, after an
    addConcentration the network holds: written again beside the .net, it
    added 50 twice. A save before generation is made again on that network."""
    actions = (
        'setParameter("k2",2)\nsaveParameters()\nsetParameter("k2",3)\n'
        'addConcentration("A()",50)\n' + RUN + "resetParameters()\n" + RUN
    )
    net = NET.replace("1 A() 100", "1 A() 150").replace("2 k2 1", "2 k2 3")
    _pack(tmp_path, actions, net=net)
    report = omex_to_net(tmp_path / "m.omex", tmp_path / "back.net", gate="none")
    assert "addConcentration" not in Path(report.bngl_out).read_text()
    back = parse_bngl_protocol(report.bngl_out)
    assert back.generation_step == 0
    first, second = bngsim.run_protocol(
        bngsim.Model.from_net(str(tmp_path / "back.net")), back, rtol=1e-10, atol=1e-12
    )
    a1 = _eq(1, 3, a0=150)
    assert _end(first.result) == pytest.approx(a1, rel=RTOL)
    # From A = a1 of 150 at the saved k2 = 2: A -> 150*2/3 = 100.
    assert _end(second.result) == pytest.approx(100 + (a1 - 100) * math.exp(-3), rel=RTOL)


# ─── #1682: a parameter set to an expression follows what it reads ─────────


def test_a_parameter_set_to_an_expression_follows_what_it_reads(tmp_path):
    """setParameter("k2","k1*3") then setParameter("k1",5): BNG2.pl runs k2 = 15
    (A(1) = 75.000); the replay ran k2 = 3 (37.521)."""
    protocol = parse_bngl_protocol('setParameter("k2","k1*3")\nsetParameter("k1",5)\n' + RUN)
    out = _replay(NET, protocol, tmp_path)
    assert out.model.get_param("k2") == 15.0
    assert _end(out[0].result) == pytest.approx(_eq(5, 15), rel=RTOL)


def test_a_scan_moves_a_parameter_set_to_an_expression_of_it(tmp_path):
    """BNG2.pl: 75.46, 75.00, 75.00 over k1 = 1, 3, 5 (k2 = 3 k1); and k2 = 15
    after. The replay gave 75.46, 50.12, 37.52."""
    protocol = parse_bngl_protocol(
        'setParameter("k2","k1*3")\nparameter_scan({method=>"ode",parameter=>"k1",par_min=>1,'
        "par_max=>5,n_scan_pts=>3,t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n"
    )
    out = _replay(NET, protocol, tmp_path)
    got = [_end(r) for r in out[0].scan_results]
    assert got == pytest.approx([_eq(k, 3 * k) for k in (1, 3, 5)], rel=RTOL)
    assert out.model.get_param("k2") == 15.0


def test_a_saved_expression_comes_back_as_the_expression(tmp_path):
    """BNG2.pl's saveParameters keeps k2 = k1*3; after k2 = 1, k1 = 5 and the
    reset, k1 = 1 and k2 = 3 (A(1) = 75.458)."""
    protocol = parse_bngl_protocol(
        'setParameter("k2","k1*3")\nsaveParameters()\nsetParameter("k2",1)\n'
        'setParameter("k1",5)\nresetParameters()\nsetParameter("k1",2)\n' + RUN
    )
    out = _replay(NET, protocol, tmp_path)
    assert (out.model.get_param("k1"), out.model.get_param("k2")) == (2.0, 6.0)
    assert _end(out[0].result) == pytest.approx(_eq(2, 6), rel=RTOL)


def test_a_number_ends_the_expression(tmp_path):
    protocol = parse_bngl_protocol(
        'setParameter("k2","k1*3")\nsetParameter("k2",2)\nsetParameter("k1",5)\n' + RUN
    )
    out = _replay(NET, protocol, tmp_path)
    assert out.model.get_param("k2") == 2.0


def test_a_scan_of_a_parameter_set_to_an_expression_runs_its_values(tmp_path):
    """BNG2.pl sets each point with a number, which ends k1's expression: the
    points run k1 = 1, 3, not k2*2 = 2."""
    protocol = parse_bngl_protocol(
        'setParameter("k1","k2*2")\nparameter_scan({method=>"ode",parameter=>"k1",'
        "par_scan_vals=>[1,3],t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n"
    )
    out = _replay(NET, protocol, tmp_path)
    assert [_end(r) for r in out[0].scan_results] == pytest.approx(
        [_eq(1, 1), _eq(3, 1)], rel=RTOL
    )


# k0 beside A <-> B, read by nothing but the actions.
NET0 = NET.replace("    2 k2 1\n", "    2 k2 1\n    3 k0 1\n")
MODEL0 = MODEL.replace("  k2 1\n", "  k2 1\n  k0 1\n")
# k1 = 2 k0 and k2 = 3 k1, then k0 = 5: k1 = 10, k2 = 30. And k2 = k0 + k1,
# set before k1 = 2 k0, then k0 = 5: k2 is set after k1, k1 = 10, k2 = 15.
CHAINS = [
    ('setParameter("k2","k1*3")\nsetParameter("k1","k0*2")\nsetParameter("k0",5)\n', 10, 30),
    ('setParameter("k2","k0+k1")\nsetParameter("k1","k0*2")\nsetParameter("k0",5)\n', 10, 15),
]


@pytest.mark.parametrize(("sets", "k1", "k2"), CHAINS, ids=["chained", "crossed"])
def test_expressions_follow_one_another(tmp_path, sets, k1, k2):
    out = _replay(NET0, parse_bngl_protocol(sets + RUN), tmp_path)
    assert (out.model.get_param("k1"), out.model.get_param("k2")) == (k1, k2)
    assert _end(out[0].result) == pytest.approx(_eq(k1, k2), rel=RTOL)


@pytest.mark.parametrize(("sets", "k1", "k2"), CHAINS, ids=["chained", "crossed"])
def test_the_sedml_sets_expressions_in_the_order_they_read_one_another(tmp_path, sets, k1, k2):
    _pack(tmp_path, "generate_network({overwrite=>1})\n" + sets + RUN, net=NET0, model=MODEL0)
    (values,) = _run_report(tmp_path / "m.omex").values()
    assert values["observable:Atot"] == pytest.approx(_eq(k1, k2), rel=1e-5)


@pytest.mark.parametrize(
    "sets",
    ['setParameter("k2","k2*2")', 'setParameter("k1","k2*2")\nsetParameter("k2","k1+1")'],
    ids=["itself", "through-another"],
)
def test_a_parameter_defined_through_itself_is_refused(tmp_path, sets):
    """BNG2.pl stops: "Parameter k2 is defined recursively"."""
    protocol = parse_bngl_protocol(sets + "\n" + RUN)
    with pytest.raises(ConversionError, match="defined recursively"):
        _replay(NET, protocol, tmp_path)


def test_a_parameter_defined_through_the_networks_own_expression_is_refused(tmp_path):
    """k3 = 2 k1 in the network, then setParameter("k1","k3"): BNG2.pl stops
    ("Max recursion depth"); the replay ran k1 = 8, k3 = 16 (A(1) = 11.12)."""
    net = NET.replace("    2 k2 1\n", "    2 k2 1\n    3 k3 2*k1\n")
    with pytest.raises(ConversionError, match="defined recursively"):
        _replay(net, parse_bngl_protocol('setParameter("k1","k3")\n' + RUN), tmp_path)


def test_the_sedml_sets_the_expression_again_after_what_it_reads(tmp_path):
    """The archive's standard elements, run on libRoadRunner, gave 37.521 for
    BNG2.pl's 75.000; a scan's points gave 75.46, 50.12, 37.52."""
    actions = 'generate_network({overwrite=>1})\nsetParameter("k2","k1*3")\nsetParameter("k1",5)\n'
    _pack(tmp_path, actions + RUN)
    (values,) = _run_report(tmp_path / "m.omex").values()
    assert values["observable:Atot"] == pytest.approx(_eq(5, 15), rel=1e-5)


def test_a_sedml_scan_sets_the_expression_at_each_point(tmp_path):
    actions = (
        "generate_network({overwrite=>1})\n"
        'setParameter("k2","k1*3")\nparameter_scan({method=>"ode",parameter=>"k1",'
        "par_scan_vals=>[1,3,5],t_end=>1,n_steps=>1})\n"
    )
    _pack(tmp_path, actions)
    sed = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml").decode()
    bare = re.sub(r"<annotation>.*?</annotation>", "", sed, flags=re.S)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        spec = read_sedml_protocol(bare)
    assert spec.lossy == ()
    out = _replay(NET, spec, tmp_path, rtol=1e-10, atol=1e-12)
    got = [_end(r) for r in out[-1].scan_results]
    assert got == pytest.approx([_eq(k, 3 * k) for k in (1, 3, 5)], rel=RTOL)


def test_a_sedml_change_is_evaluated_once(tmp_path):
    """SED-ML evaluates a computeChange where it applies: k2 = 3*k1 then
    k1 = 5 holds k2 = 3, where a BNGL action would follow k1."""
    sbml = bngsim.convert.net_to_sbml(_files(tmp_path, RUN)[0], validate=None).sbml_text
    param = "/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='{}']"
    doc = f"""<sedML xmlns="{SED}" xmlns:sbml="{NS["sbml"]}" level="1" version="3">
 <listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
   outputEndTime="1" numberOfPoints="1"><algorithm kisaoID="KISAO:0000019"/></uniformTimeCourse>
 </listOfSimulations>
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="model.xml">
  <listOfChanges><computeChange target="{param.format("k2")}/@value">
   <listOfVariables><variable id="v" target="{param.format("k1")}"/></listOfVariables>
   <math xmlns="{MATHML}"><apply><times/><cn>3</cn><ci>v</ci></apply></math></computeChange>
  <changeAttribute target="{param.format("k1")}/@value" newValue="5"/>
 </listOfChanges></model></listOfModels>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/></listOfTasks>
</sedML>"""
    spec = read_sedml_protocol(doc)
    assert spec.steps[0].evaluated_once
    (run,) = bngsim.run_protocol(bngsim.Model.from_sbml_string(sbml), spec, rtol=1e-10, atol=1e-12)
    assert _end(run.result) == pytest.approx(_eq(5, 3), rel=RTOL)
    # BNG2.pl would keep the expression: the actions block says so.
    with pytest.warns(ConversionWarning, match="evaluated once where it runs"):
        write_bngl_protocol(spec)


# ─── #1668: a save before generation ─────────────────────────────────────────

SAVED_BEFORE = (
    'setParameter("k2",0)\nsetConcentration("A()",50)\nsaveParameters()\nsaveConcentrations()\n'
    'setParameter("k2",1)\nsetConcentration("A()",100)\ngenerate_network({overwrite=>1})\n'
    "resetParameters()\nresetConcentrations()\n" + RUN
)


def test_a_save_before_generation_is_exported(tmp_path):
    """BNG2.pl resets to the saved k2 = 0, A = 50: A(1) = 50 e^-1 = 18.394.
    The exported task ran the network as generated (56.767), and bngsim's own
    replay of the archive refused it."""
    report, said = _pack(tmp_path, SAVED_BEFORE)
    assert report.sedml_lossy == [] and said == []
    want = 50 * math.exp(-1)
    (values,) = _run_report(tmp_path / "m.omex").values()
    assert values["observable:Atot"] == pytest.approx(want, rel=1e-5)
    with read_omex(tmp_path / "m.omex") as archive:
        (run,) = bngsim.run_protocol(
            archive.load_model(), archive.load_full_protocol(), rtol=1e-10, atol=1e-12
        )
    assert _end(run.result) == pytest.approx(want, rel=RTOL)


def test_a_save_before_generation_replays(tmp_path):
    protocol = parse_bngl_protocol(MODEL + SAVED_BEFORE)
    (run,) = _replay(NET, protocol, tmp_path)
    assert _end(run.result) == pytest.approx(50 * math.exp(-1), rel=RTOL)


def test_a_save_of_a_value_as_declared_is_refused_and_said(tmp_path):
    """Saved before k2 changed from its declared value: the network holds only
    the changed one, so neither the replay nor SED-ML can go back to it."""
    actions = (
        'saveParameters()\nsetParameter("k2",3)\ngenerate_network({overwrite=>1})\n'
        "resetParameters()\n" + RUN
    )
    protocol = parse_bngl_protocol(MODEL + actions)
    net = NET.replace("2 k2 1", "2 k2 3")
    with pytest.raises(ConversionError, match="saved under 'DEFAULT' before the network"):
        _replay(net, protocol, tmp_path)
    report, said = _pack(tmp_path, actions, net=net)
    assert any("cannot go back to the parameters" in note for note in report.sedml_lossy)
    assert report.sedml_lossy == [n for n in said if n in report.sedml_lossy]


# ─── #1684: what the SED-ML does not hold is recorded in the archive ──────────


def test_a_sedml_loss_is_recorded_in_the_archive(tmp_path):
    """A reset_conc scan after a run: SED-ML starts each point from the initial
    state. Only a warning at export said so; the archive read PASS."""
    actions = (
        "generate_network({overwrite=>1})\n"
        + RUN
        + 'parameter_scan({method=>"ode",parameter=>"k2",par_min=>0,par_max=>2,'
        "n_scan_pts=>3,t_end=>1,n_steps=>1})\n"
    )
    report, said = _pack(tmp_path, actions)
    assert len(report.sedml_lossy) == 1 and "reset_conc=>1" in report.sedml_lossy[0]
    assert report.sedml_lossy[0] in said
    z = zipfile.ZipFile(tmp_path / "m.omex")
    record = json.loads(z.read("bngsim-conversion.json"))["conversion"]
    assert record["sedml_lossy"] == report.sedml_lossy
    assert "do not hold 1 part of the protocol exactly" in z.read("metadata.rdf").decode()
    assert "sedml lossy:" in report.summary()


def test_an_exact_export_records_no_loss(tmp_path):
    report, _ = _pack(tmp_path, "generate_network({overwrite=>1})\n" + RUN)
    z = zipfile.ZipFile(tmp_path / "m.omex")
    assert report.sedml_lossy == []
    assert json.loads(z.read("bngsim-conversion.json"))["conversion"]["sedml_lossy"] == []
    assert "do not hold" not in z.read("metadata.rdf").decode()


# ─── #1686: the SED-ML says what each experiment reports ─────────────────────

DECAY_NET = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 100
end species
begin reactions
    1 1 0 k
end reactions
begin groups
    1 Atot 1
end groups
"""
DECAY = """begin model
begin parameters
  k 0.5
end parameters
begin seed species
  A() 100
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() -> 0 k
end reaction rules
end model
generate_network({overwrite=>1})
"""


def test_each_experiment_has_a_report_of_the_task_that_runs_it(tmp_path):
    """run / add 50 / run: BNG2.pl ends its runs at 100 e^-5 = 0.6738 and
    (0.6738 + 50) e^-5 = 0.3414. The SED-ML had five tasks and no output."""
    run = 'simulate({method=>"ode",t_end=>10,n_steps=>1})\n'
    _pack(tmp_path, run + 'addConcentration("A()",50)\n' + run, net=DECAY_NET, model=DECAY)
    sed = ET.fromstring(zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml"))
    reports = list(sed.iter(f"{{{SED}}}report"))
    assert [r.get("id") for r in reports] == ["report_0", "report_1"]
    read = {
        v.get("taskReference") for v in sed.iter(f"{{{SED}}}variable") if v.get("taskReference")
    }
    assert read == {"task_0", "chain_1"}
    first = 100 * math.exp(-5)
    values = _run_report(tmp_path / "m.omex")
    assert values["report_0"]["observable:Atot"] == pytest.approx(first, rel=1e-5)
    assert values["report_1"]["observable:Atot"] == pytest.approx(
        (first + 50) * math.exp(-5), rel=1e-5
    )
    assert values["report_1"]["time"] == pytest.approx(10.0)


def test_read_sedml_reads_a_protocol_documents_first_experiment(tmp_path):
    """The first task's outputs are now in the document, and read_sedml has them."""
    _pack(tmp_path, RUN)
    sed = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml").decode()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        spec = read_sedml(sed, model_source="model.xml")
    assert spec.outputs == ("observable:Atot",)


# ─── #1677: the document follows the L1V3 schema ─────────────────────────────

# The sequence the L1V3 schema gives <sedML>'s children (SedBase's first).
SCHEMA_ORDER = [
    "notes",
    "annotation",
    "listOfDataDescriptions",
    "listOfSimulations",
    "listOfModels",
    "listOfTasks",
    "listOfDataGenerators",
    "listOfOutputs",
]


def test_the_lists_are_in_the_schemas_order(tmp_path):
    report, _ = _pack(tmp_path, RUN + 'setConcentration("A()",3)\n' + RUN)
    sed = ET.fromstring(zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml"))
    children = [c.tag.rsplit("}", 1)[-1] for c in sed]
    assert children == sorted(children, key=SCHEMA_ORDER.index)
    assert children[1:3] == ["listOfSimulations", "listOfModels"]


def test_one_experiment_held_twice_is_two_experiments():
    """Numbered by identity, combine_protocols([p, p]) wrote sim_1 and task_1 twice."""
    p = parse_bngl_protocol(RUN)
    twice = ProtocolSpec(steps=p.steps * 2, generation_step=0)
    for spec in (twice, combine_protocols([p, p])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConversionWarning)
            text = write_sedml_protocol(spec)
        ids = re.findall(r' id="([^"]+)"', text)
        assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)
        assert "sim_1" in ids


# ─── #1678: the archive holds no path of the exporter's ──────────────────────


def test_the_archive_names_the_bngl_by_its_file_name(tmp_path):
    private = tmp_path / "a_private_folder_name"
    private.mkdir()
    _pack(private, RUN)
    with zipfile.ZipFile(private / "m.omex") as z:
        for name in z.namelist():
            assert "a_private_folder_name" not in z.read(name).decode("utf-8", "replace"), name
        sed = z.read("simulation.sedml").decode()
    assert read_sedml_protocol(sed).source == "m.bngl"


def test_a_combined_source_is_named_by_its_file_names():
    p = ProtocolSpec(
        steps=parse_bngl_protocol(RUN).steps, source="/home/me/x/a.bngl + C:\\me\\b.bngl"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        assert read_sedml_protocol(write_sedml_protocol(p)).source == "a.bngl + b.bngl"


# ─── #1685: a replay of the archive names species as its model does ─────────


def _named_net(species: str) -> str:
    compartments = "begin compartments\n    1 c 3 2\nend compartments\n" if "@" in species else ""
    return (
        "begin parameters\n    1 k 1\nend parameters\n"
        + compartments
        + f"begin species\n    1 {species} 100\nend species\n"
        + "begin reactions\n    1 1 0 k\nend reactions\n"
    )


def _named_model(species: str, pattern: str) -> str:
    compartments = "begin compartments\n  c 3 2\nend compartments\n" if "@" in species else ""
    return (
        "begin model\nbegin parameters\n  k 1\nend parameters\n"
        + compartments
        + f"begin seed species\n  {pattern} 100\nend seed species\n"
        + f"begin reaction rules\n  {pattern.split(':')[-1]} -> 0 k\nend reaction rules\n"
        + "end model\n"
        + "generate_network({overwrite=>1})\n"
    )


@pytest.mark.parametrize(
    ("species", "pattern"),
    [("A()", "A()"), ("L(r)", "L(r)"), ("X(s~u)", "X(s~u)"), ("@c::A()", "@c:A()")],
    ids=["plain", "component", "state", "compartment"],
)
def test_an_archive_replays_on_each_of_its_models(tmp_path, species, pattern):
    """setConcentration of the species to 50, then a run: 50 e^-1 = 18.394 by
    every route. On the archive's SBML the replay named no species 'L(r)',
    and BNG2.pl stopped on omex_to_net's actions beside its .net."""
    actions = f'setConcentration("{pattern}",50)\n' + RUN
    _pack(tmp_path, actions, net=_named_net(species), model=_named_model(species, pattern))
    want = 50 * math.exp(-1)
    with read_omex(tmp_path / "m.omex") as archive:
        (on_sbml,) = bngsim.run_protocol(archive.load_model(), archive.load_full_protocol())
        net_entry = next(e for e in archive.model_entries() if e.kind == "net")
        (on_net,) = bngsim.run_protocol(
            bngsim.Model.from_net(str(archive.path_of(net_entry))),
            archive.load_full_protocol(net_entry),
        )
    assert float(on_sbml.result.species[-1, 0]) == pytest.approx(want, rel=1e-5)
    assert float(on_net.result.species[-1, 0]) == pytest.approx(want, rel=1e-5)
    report = omex_to_net(tmp_path / "m.omex", tmp_path / "back.net", gate="none")
    back = bngsim.Model.from_net(str(tmp_path / "back.net"))
    (target,) = re.findall(r'setConcentration\("([^"]+)"', Path(report.bngl_out).read_text())
    assert target in back.species_names
    (again,) = bngsim.run_protocol(back, parse_bngl_protocol(report.bngl_out))
    assert float(again.result.species[-1, 0]) == pytest.approx(want, rel=1e-5)


def test_a_protocol_without_sbml_keeps_its_bngl_names(tmp_path):
    _pack(tmp_path, 'generate_network({overwrite=>1})\nsetConcentration("A()",50)\n' + RUN)
    sed = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml").decode()
    assert read_sedml_protocol(sed).steps[0].target == "A()"
    with read_omex(tmp_path / "m.omex") as archive:
        assert archive.load_full_protocol().steps[0].target == "A"


# ─── #1663: a data generator that computes is not read as one variable ──────

DECAY_SBML_NS = NS["sbml"]


def _generator_doc(variables: str, math_xml: str) -> str:
    var = "".join(
        f'<variable id="{v.lower()}" taskReference="t" target="/sbml:sbml/sbml:model/'
        f"sbml:listOfSpecies/sbml:species[@id='{v}']\"/>"
        for v in variables
    )
    return f"""<sedML xmlns="{SED}" xmlns:sbml="{DECAY_SBML_NS}" level="1" version="3">
 <listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
   outputEndTime="2" numberOfPoints="20"><algorithm kisaoID="KISAO:0000019"/></uniformTimeCourse>
 </listOfSimulations>
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/></listOfTasks>
 <listOfDataGenerators><dataGenerator id="gen"><listOfVariables>{var}</listOfVariables>
  <math xmlns="{MATHML}">{math_xml}</math></dataGenerator></listOfDataGenerators>
</sedML>"""


@pytest.mark.parametrize(
    ("variables", "math_xml"),
    [
        ("AB", "<apply><plus/><ci>a</ci><ci>b</ci></apply>"),
        ("BA", "<apply><plus/><ci>b</ci><ci>a</ci></apply>"),
        ("A", "<apply><times/><cn>100</cn><ci>a</ci></apply>"),
    ],
    ids=["a-plus-b", "b-plus-a", "scaled"],
)
def test_a_generator_that_computes_is_left_out_and_said(variables, math_xml):
    """A + B came back as species:A (0.135 for 1), and 100*A as species:A."""
    with pytest.warns(ConversionWarning, match=r"gen = .* compute a value from their variables"):
        spec = read_sedml(_generator_doc(variables, math_xml))
    assert spec.outputs == ()


def test_a_generator_that_is_its_variable_is_read():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        spec = read_sedml(_generator_doc("A", "<ci>a</ci>"))
    assert spec.outputs == ("species:A",)


# ─── #1664: a scan runs its setValue's math over the range it names ─────────

DECAY_SBML = (
    f'<sbml xmlns="{DECAY_SBML_NS}" level="3" version="2"><model id="m">'
    '<listOfCompartments><compartment id="C" spatialDimensions="3" size="1" constant="true"/>'
    '</listOfCompartments><listOfSpecies><species id="A" compartment="C" initialConcentration="1" '
    'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>'
    '<listOfParameters><parameter id="k" value="1" constant="true"/></listOfParameters>'
    '<listOfReactions><reaction id="J" reversible="false"><listOfReactants><speciesReference '
    'species="A" stoichiometry="1" constant="true"/></listOfReactants><kineticLaw>'
    f'<math xmlns="{MATHML}"><apply><times/><ci>C</ci><ci>k</ci><ci>A</ci></apply></math>'
    "</kineticLaw></reaction></listOfReactions></model></sbml>"
)


def _scan_doc(ranges: str, named: str, math_xml: str) -> str:
    return f"""<sedML xmlns="{SED}" xmlns:sbml="{DECAY_SBML_NS}" level="1" version="3">
 <listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
   outputEndTime="1" numberOfPoints="10"><algorithm kisaoID="KISAO:0000019"/></uniformTimeCourse>
 </listOfSimulations>
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/>
  <repeatedTask id="scan" resetModel="true" range="x"><listOfRanges>{ranges}</listOfRanges>
   <listOfChanges><setValue modelReference="m" range="{named}"
     target="/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='k']">
    <math xmlns="{MATHML}">{math_xml}</math></setValue></listOfChanges>
   <listOfSubTasks><subTask order="1" task="t"/></listOfSubTasks></repeatedTask>
 </listOfTasks>
</sedML>"""


def _vector(rid: str, values) -> str:
    return (
        f'<vectorRange id="{rid}">'
        + "".join(f"<value>{v}</value>" for v in values)
        + "</vectorRange>"
    )


POWER = "<apply><power/><cn>10</cn><ci>x</ci></apply>"


@pytest.mark.parametrize(
    ("ranges", "named", "math_xml"),
    [
        (_vector("x", [0, 1, 2]), "x", POWER),
        (
            _vector("x", [0, 1, 2])
            + f'<functionalRange id="kx" range="x"><math xmlns="{MATHML}">{POWER}</math>'
            "</functionalRange>",
            "kx",
            "<ci>kx</ci>",
        ),
        (_vector("x", [0, 1, 2]) + _vector("y", [1, 10, 100]), "y", "<ci>y</ci>"),
        (
            '<uniformRange id="x" start="1" end="100" numberOfPoints="2" type="log"/>',
            "x",
            "<ci>x</ci>",
        ),
    ],
    ids=["math", "functional-range", "second-range", "log-range"],
)
def test_a_scan_runs_the_values_its_set_value_gives(ranges, named, math_xml):
    """k = 10^x over x = 0, 1, 2 runs k = 1, 10, 100: A(1) = e^-k. Each ran
    k = 0, 1, 2 (A(1) = 1, 0.368, 0.135) with no word."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        spec = read_sedml_protocol(_scan_doc(ranges, named, math_xml))
    (run,) = bngsim.run_protocol(
        bngsim.Model.from_sbml_string(DECAY_SBML), spec, rtol=1e-10, atol=1e-14
    )
    assert list(run.scan_values) == pytest.approx([1.0, 10.0, 100.0])
    got = [float(r.species[-1, 0]) for r in run.scan_results]
    assert got == pytest.approx([math.exp(-k) for k in (1, 10, 100)], rel=1e-5, abs=1e-12)


def test_a_scan_over_a_range_it_cannot_evaluate_is_lossy():
    math_xml = "<apply><csymbol>unknown</csymbol><ci>x</ci></apply>"
    with pytest.warns(ConversionWarning, match="does not evaluate"):
        spec = read_sedml_protocol(_scan_doc(_vector("x", [0, 1]), "x", math_xml))
    assert len(spec.lossy) == 1 and spec.experiments == []


def test_a_state_change_records_evaluated_once_only_when_set():
    """The JSON of a protocol without one is as before."""
    assert "evaluated_once" not in StateChange("set_parameter", "k", value_expr="2*j").to_dict()
    once = StateChange("set_parameter", "k", value_expr="2*j", evaluated_once=True)
    assert StateChange.from_dict(once.to_dict()) == once


# ─── found in review: the export after a reset, a value the SBML computes ───


@pytest.mark.parametrize("before", ["", "saveConcentrations()\n"], ids=["declared", "saved"])
def test_a_reset_sets_a_followed_expression_after_what_it_reads(tmp_path, before):
    """k2 = k1*3, k1 = 5, a run, resetConcentrations(), a run: both at k2 = 15
    (75.00). The second run's model set k2 = k1*3 before k1 = 5 (37.52)."""
    actions = (
        "generate_network({overwrite=>1})\n"
        + before
        + 'setParameter("k2","k1*3")\nsetParameter("k1",5)\n'
        + RUN
        + "resetConcentrations()\n"
        + RUN
    )
    report, _ = _pack(tmp_path, actions)
    values = _run_report(tmp_path / "m.omex")
    for rid in ("report_0", "report_1"):
        assert values[rid]["observable:Atot"] == pytest.approx(_eq(5, 15), rel=1e-5), rid


@pytest.mark.parametrize(
    ("net", "model", "actions", "want"),
    [
        (
            NET.replace("2 k2 1", "2 k2 k1*3"),
            MODEL.replace("  k2 1", "  k2 k1*3"),
            'setParameter("k2",7)\n',
            _eq(1, 7),
        ),
        (
            NET.replace("    2 k2 1\n", "    2 k2 1\n    3 A0 100\n").replace(
                "1 A() 100", "1 A() A0"
            ),
            MODEL.replace("  k2 1\n", "  k2 1\n  A0 100\n").replace("A() 100", "A() A0"),
            'setConcentration("A()",50)\n',
            _eq(1, 1, a0=50),
        ),
    ],
    ids=["derived-parameter", "seed-names-a-parameter"],
)
def test_a_change_to_a_value_the_sbml_computes_takes(tmp_path, net, model, actions, want):
    """The SBML computes k2 = k1*3, and A from A0, by an initialAssignment,
    which overrode the exported changeAttribute: k2 ran 3 (75.46 for 87.50).
    BNG2.pl's action replaces the expression; so does the removeXML now
    written before the change."""
    _pack(tmp_path, "generate_network({overwrite=>1})\n" + actions + RUN, net=net, model=model)
    (values,) = _run_report(tmp_path / "m.omex").values()
    assert values["observable:Atot"] == pytest.approx(want, rel=1e-5)
    sed = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml").decode()
    assert "removeXML" in sed
    bare = re.sub(r"<annotation>.*?</annotation>", "", sed, flags=re.S)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        spec = read_sedml_protocol(bare)
    with read_omex(tmp_path / "m.omex") as archive:
        (run,) = bngsim.run_protocol(archive.load_model(), spec, rtol=1e-10, atol=1e-12)
    assert _end(run.result) == pytest.approx(want, rel=RTOL)


SWEEP_SBML = DECAY_SBML.replace(
    '<parameter id="k" value="1" constant="true"/>',
    '<parameter id="k" value="1" constant="true"/><parameter id="k2" value="1" constant="true"/>',
).replace("<ci>k</ci><ci>A</ci>", "<ci>k2</ci><ci>A</ci>")


def test_a_sweep_follower_holds_its_last_value_after_the_scan():
    """A chain: a sweep of k = x, k2 = 3x over x = 0.1, 0.2, each point on from
    the last, then k = 1 and a run. SED-ML leaves k2 = 0.6, so A ends at
    e^-0.9 e^-0.6 = 0.2231; k2 kept following k, to 3 (0.0202)."""
    k, k2 = (
        f"/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='{p}']"
        for p in ["k", "k2"]
    )
    sv = '<setValue modelReference="m"{r} target="{t}"><math xmlns="{m}">{x}</math></setValue>'
    sweep = sv.format(r=' range="x"', t=k, m=MATHML, x="<ci>x</ci>") + sv.format(
        r=' range="x"', t=k2, m=MATHML, x="<apply><times/><cn>3</cn><ci>x</ci></apply>"
    )
    step = sv.format(r="", t=k, m=MATHML, x="<cn>1</cn>")
    a = "/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']"
    doc = f"""<sedML xmlns="{SED}" xmlns:sbml="{DECAY_SBML_NS}" level="1" version="3">
 <listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
   outputEndTime="1" numberOfPoints="1"><algorithm kisaoID="KISAO:0000019"/></uniformTimeCourse>
 </listOfSimulations>
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/>
  <repeatedTask id="sweep" resetModel="false" range="x">
   <listOfRanges>{_vector("x", [0.1, 0.2])}</listOfRanges><listOfChanges>{sweep}</listOfChanges>
   <listOfSubTasks><subTask order="1" task="t"/></listOfSubTasks></repeatedTask>
  <repeatedTask id="step" resetModel="false" range="o">
   <listOfRanges>{_vector("o", [0])}</listOfRanges><listOfChanges>{step}</listOfChanges>
   <listOfSubTasks><subTask order="1" task="t"/></listOfSubTasks></repeatedTask>
  <repeatedTask id="chain" resetModel="true" range="c"><listOfRanges>{_vector("c", [0])}
   </listOfRanges><listOfSubTasks><subTask order="1" task="sweep"/>
   <subTask order="2" task="step"/></listOfSubTasks></repeatedTask>
 </listOfTasks>
 <listOfDataGenerators><dataGenerator id="g"><listOfVariables>
   <variable id="a" taskReference="chain" target="{a}"/></listOfVariables>
   <math xmlns="{MATHML}"><ci>a</ci></math></dataGenerator></listOfDataGenerators>
</sedML>"""
    spec = read_sedml_protocol(doc)
    assert spec.lossy == ()
    runs = bngsim.run_protocol(
        bngsim.Model.from_sbml_string(SWEEP_SBML), spec, rtol=1e-10, atol=1e-14
    )
    assert float(runs[-1].result.species[-1, 0]) == pytest.approx(math.exp(-1.5), rel=1e-6)


def test_the_bngl_writer_marks_where_the_network_was_generated():
    """A model file's addConcentration before its first run is in the network:
    written as a bare block, it read back as after generation, and ran twice."""
    p = parse_bngl_protocol(MODEL + 'addConcentration("A()",50)\n' + RUN)
    assert p.generation_step == 1
    again = parse_bngl_protocol(write_bngl_protocol(p))
    assert (again.generation_step, again.steps) == (1, p.steps)


UNKNOWN_SAVE = (
    'saveConcentrations()\nsetConcentration("A()",50)\ngenerate_network({overwrite=>1})\n'
    "resetConcentrations()\n" + RUN
)


def test_omex_to_net_refuses_actions_it_cannot_write(tmp_path):
    """Saved at A = 100 as declared, then A = 50 before generation: the .net
    holds 50, so its actions cannot go back to 100. BNG2.pl on the .bngl gives
    56.77; on the .net and the block, 28.38."""
    _pack(tmp_path, UNKNOWN_SAVE, net=NET.replace("1 A() 100", "1 A() 50"))
    with pytest.raises(ConversionError, match="cannot make that state"):
        omex_to_net(tmp_path / "m.omex", tmp_path / "back.net", gate="none")
    with pytest.warns(ConversionWarning, match="cannot make that state"):
        report = omex_to_net(tmp_path / "m.omex", tmp_path / "back.net", gate="none", strict=False)
    assert any(note.startswith("actions: ") for note in report.lossy)
    assert "# bngsim: the actions block goes back" in Path(report.bngl_out).read_text()


def test_a_computed_generator_names_its_variables():
    with pytest.warns(ConversionWarning, match=r"of a = species:A, b = species:B"):
        read_sedml(_generator_doc("AB", "<apply><plus/><ci>a</ci><ci>b</ci></apply>"))


def test_expressions_that_do_not_settle_are_refused(tmp_path, monkeypatch):
    """The backstop behind the recursion check: k1 = k3 with the network's
    k3 = 2 k1 doubles each pass, and stops with an error, not a number."""
    from bngsim._protocol_run import _Runner

    monkeypatch.setattr(_Runner, "_defines_recursively", lambda self, name, text: False)
    net = NET.replace("    2 k2 1\n", "    2 k2 1\n    3 k3 2*k1\n")
    with pytest.raises(ConversionError, match="do not settle"):
        _replay(net, parse_bngl_protocol('setParameter("k1","k3")\n' + RUN), tmp_path)


# ─── found in the review's second round ─────────────────────────────────────

DECLARED_NET = NET.replace("2 k2 1", "2 k2 k1*3")
DECLARED_MODEL = MODEL.replace("  k2 1", "  k2 k1*3")


def test_a_reset_to_the_networks_expression_keeps_its_assignment(tmp_path):
    """Declared k2 = k1*3; save, k2 = 7, reset, k1 = 5: BNG2.pl runs k2 = 15
    (75.00). The export removed k2's assignment and wrote k2 = 3 (37.52)."""
    actions = (
        'generate_network({overwrite=>1})\nsaveParameters()\nsetParameter("k2",7)\n'
        'resetParameters()\nsetParameter("k1",5)\n' + RUN
    )
    report, _ = _pack(tmp_path, actions, net=DECLARED_NET, model=DECLARED_MODEL)
    (values,) = _run_report(tmp_path / "m.omex").values()
    assert values["observable:Atot"] == pytest.approx(_eq(5, 15), rel=1e-5)
    assert report.sedml_lossy == []
    with read_omex(tmp_path / "m.omex") as archive:
        (run,) = bngsim.run_protocol(archive.load_model(), archive.load_full_protocol())
    assert _end(run.result) == pytest.approx(_eq(5, 15), rel=1e-6)


def test_a_value_that_is_not_a_number_is_refused_as_one(tmp_path):
    """BNG2.pl stops on k1*0/0; the replay said it was defined recursively."""
    protocol = parse_bngl_protocol('setParameter("k2","k1*0/0")\n' + RUN)
    with pytest.raises(ConversionError, match="does not evaluate"):
        _replay(NET, protocol, tmp_path)


def _sbml_assigning_k2(tmp_path: Path, formula: str) -> str:
    """The SBML of DECLARED_NET with k2's initialAssignment as *formula*,
    which may call triple, a function of the SBML's own."""
    import libsbml

    (tmp_path / "declared.net").write_text(DECLARED_NET)
    text = bngsim.convert.net_to_sbml(tmp_path / "declared.net", validate=None).sbml_text
    doc = libsbml.readSBMLFromString(text)
    m = doc.getModel()
    fd = m.createFunctionDefinition()
    fd.setId("triple")
    fd.setMath(libsbml.parseL3Formula("lambda(x, 3*x)"))
    m.getInitialAssignment("k2").setMath(libsbml.parseL3Formula(formula))
    return libsbml.writeSBMLToString(doc)


@pytest.mark.parametrize(
    ("actions", "noted"),
    [
        (
            'parameter_scan({method=>"ode",parameter=>"k1",par_scan_vals=>[1,3,5],t_end=>1,n_steps=>1})\n',
            True,
        ),
        (RUN + 'setParameter("k1",5)\n' + RUN, True),
        ('setParameter("k1",5)\n' + RUN, False),
        ('setParameter("k2",7)\n' + RUN + 'setParameter("k1",5)\n' + RUN, False),
    ],
    ids=["scan", "set-between-runs", "set-before-the-run", "replaced-before"],
)
@pytest.mark.parametrize(
    "formula",
    [
        "triple(k1)",
        "k1*3 + 0*time",
        "rem(k1*3, 7)",
        "quotient(k1*30, 7)",
        "piecewise(k1*3, implies(k1 > 5, k1 > 6), 1)",
    ],
    ids=["function", "time", "rem", "quotient", "implies"],
)
def test_an_assignment_sedml_math_cannot_say_is_recorded_where_it_is_lost(
    tmp_path, actions, noted, formula
):
    """k2 = triple(k1), one that reads the SBML's time, or one with an operator
    of SBML L3V2's MathML that SED-ML L1V3's has not: SED-ML math names no
    SBML function, its time is the simulation's, and its schema refuses the
    operator, so the standard elements cannot set k2 again as k1 moves, and
    say so (issue #1684). A change made before the model is loaded is exact,
    and says nothing."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sbml = _sbml_assigning_k2(tmp_path, formula)
        write_sedml_protocol(parse_bngl_protocol(actions), sbml=sbml)
    said = [str(w.message) for w in caught if issubclass(w.category, ConversionWarning)]
    hits = [n for n in said if "set k1 as the model runs" in n and "['k2']" in n]
    assert bool(hits) == noted


def test_a_reset_sweep_before_more_of_a_chain_is_lossy():
    """A chain runs a sweep that resets at each point, then a task: SED-ML
    continues from the last point (0.5488 e^-0.6 = 0.3012), and a replay's
    scan puts its start state back (0.5488 from A = 1)."""
    k2 = "/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='k2']"
    sweep = (
        f'<setValue modelReference="m" range="x" target="{k2}"><math xmlns="{MATHML}">'
        "<ci>x</ci></math></setValue>"
    )
    doc = f"""<sedML xmlns="{SED}" xmlns:sbml="{DECAY_SBML_NS}" level="1" version="3">
 <listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
   outputEndTime="1" numberOfPoints="1"><algorithm kisaoID="KISAO:0000019"/></uniformTimeCourse>
 </listOfSimulations>
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/>
  <repeatedTask id="sweep" resetModel="true" range="x">
   <listOfRanges>{_vector("x", [0.3, 0.6])}</listOfRanges><listOfChanges>{sweep}</listOfChanges>
   <listOfSubTasks><subTask order="1" task="t"/></listOfSubTasks></repeatedTask>
  <repeatedTask id="chain" resetModel="true" range="c"><listOfRanges>{_vector("c", [0])}
   </listOfRanges><listOfSubTasks><subTask order="1" task="sweep"/>
   <subTask order="2" task="t"/></listOfSubTasks></repeatedTask>
 </listOfTasks>
 <listOfDataGenerators><dataGenerator id="g"><listOfVariables>
   <variable id="a" taskReference="chain"
     target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']"/></listOfVariables>
   <math xmlns="{MATHML}"><ci>a</ci></math></dataGenerator></listOfDataGenerators>
</sedML>"""
    with pytest.warns(ConversionWarning, match="a sweep that resets the model at each point"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1


def test_a_strict_refusal_leaves_no_net(tmp_path):
    _pack(tmp_path, UNKNOWN_SAVE, net=NET.replace("1 A() 100", "1 A() 50"))
    with pytest.raises(ConversionError, match="cannot make that state"):
        omex_to_net(tmp_path / "m.omex", tmp_path / "back.net", gate="none")
    assert not (tmp_path / "back.net").exists()


SEEDED_NET = NET.replace("    2 k2 1\n", "    2 k2 1\n    3 A0 100\n").replace(
    "1 A() 100", "1 A() A0"
)
SEEDED_MODEL = MODEL.replace("  k2 1\n", "  k2 1\n  A0 100\n").replace("A() 100", "A() A0")


def test_a_reset_sweep_inside_a_chain_inside_a_chain_is_lossy():
    """The sweep ends an inner chain, and the outer one runs a task after it:
    SED-ML continues from the last point (0.3012), a replay from A = 1 (0.5488)."""
    k2 = "/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='k2']"
    sweep = (
        f'<setValue modelReference="m" range="x" target="{k2}"><math xmlns="{MATHML}">'
        "<ci>x</ci></math></setValue>"
    )

    def chain(rid, reset, subtasks):
        subs = "".join(f'<subTask order="{n}" task="{t}"/>' for n, t in enumerate(subtasks, 1))
        return (
            f'<repeatedTask id="{rid}" resetModel="{reset}" range="o{rid}"><listOfRanges>'
            f"{_vector('o' + rid, [0])}</listOfRanges><listOfSubTasks>{subs}</listOfSubTasks>"
            "</repeatedTask>"
        )

    doc = f"""<sedML xmlns="{SED}" xmlns:sbml="{DECAY_SBML_NS}" level="1" version="3">
 <listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
   outputEndTime="1" numberOfPoints="1"><algorithm kisaoID="KISAO:0000019"/></uniformTimeCourse>
 </listOfSimulations>
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/>
  <repeatedTask id="sweep" resetModel="true" range="x">
   <listOfRanges>{_vector("x", [0.3, 0.6])}</listOfRanges><listOfChanges>{sweep}</listOfChanges>
   <listOfSubTasks><subTask order="1" task="t"/></listOfSubTasks></repeatedTask>
  {chain("inner", "false", ["sweep"])}{chain("outer", "true", ["inner", "t"])}
 </listOfTasks>
 <listOfDataGenerators><dataGenerator id="g"><listOfVariables>
   <variable id="a" taskReference="outer"
     target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']"/></listOfVariables>
   <math xmlns="{MATHML}"><ci>a</ci></math></dataGenerator></listOfDataGenerators>
</sedML>"""
    with pytest.warns(ConversionWarning, match="a sweep that resets the model at each point"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1


# ─── #1733: what the SBML computes by an initialAssignment, as the model runs ─


def _from(a: float, k1: float, k2: float, total: float = 100.0, t: float = 1.0) -> float:
    """A(t) of A <-> B from A = a, B = total - a."""
    aeq = total * k2 / (k1 + k2)
    return aeq + (a - aeq) * math.exp(-(k1 + k2) * t)


def _continued(a: float, rates, total: float = 100.0) -> list[float]:
    """A at the end of each run of a sweep that goes on from A = a, at each
    (k1, k2) in turn."""
    out = []
    for k1, k2 in rates:
        a = _from(a, k1, k2, total)
        out.append(a)
    return out


SCAN = (
    'parameter_scan({{method=>"ode",parameter=>"{p}",par_scan_vals=>[{v}],{extra}'
    "t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10}})\n"
)
CHAINED_NET = NET.replace("    2 k2 1\n", "    2 kx k1*3\n    3 k2 kx/3*2\n")
CHAINED_MODEL = MODEL.replace("  k2 1\n", "  kx k1*3\n  k2 kx/3*2\n")
SEED_OF_DERIVED_NET = NET.replace(
    "    2 k2 1\n", "    2 k2 1\n    3 Ab 50\n    4 A0 Ab*2\n"
).replace("1 A() 100", "1 A() A0")
SEED_OF_DERIVED_MODEL = MODEL.replace("  k2 1\n", "  k2 1\n  Ab 50\n  A0 Ab*2\n").replace(
    "A() 100", "A() A0"
)
# A <-> B at k1, kr, beside k2 = k1*3, which no reaction reads.
BESIDE_NET = NET.replace("    2 k2 1\n", "    2 k2 k1*3\n    3 kr 1\n").replace(
    "2 2 1 k2", "2 2 1 kr"
)
BESIDE_MODEL = MODEL.replace("  k2 1\n", "  k2 k1*3\n  kr 1\n").replace("k1, k2", "k1, kr")
# k2 = k1*3 up to 6, as a min and as an if: SED-ML L1V3's MathML has no min,
# and a replay writes a piecewise as BNGL's if.
CAPPED_NET = NET.replace("2 k2 1", "2 k2 min((k1*3),6)")
CAPPED_MODEL = MODEL.replace("  k2 1", "  k2 min(k1*3,6)")
STEPPED_NET = NET.replace("2 k2 1", "2 k2 if((k1>2),6,(k1*3))")
STEPPED_MODEL = MODEL.replace("  k2 1", "  k2 if(k1>2,6,k1*3)")
_A1 = _eq(1, 3)  # the first run with k2 = k1*3 declared
# Each case: the network, its model, the actions after generate_network, and
# A at the end of each run of the last experiment, as BNG2.pl 2.9.3 gives it
# (to 4 decimals: the comment after each).
INITIAL_ASSIGNMENT_CASES = {
    # 75.4579, 75.0002, 75.0000; the standard elements gave 75.46, 50.12, 37.52
    "scan-of-what-a-derived-parameter-reads": (
        "declared",
        SCAN.format(p="k1", v="1,3,5", extra=""),
        [_eq(k, 3 * k) for k in (1, 3, 5)],
    ),
    # 75.4579, 75.0000, 75.0000
    "scan-without-reset": (
        "declared",
        SCAN.format(p="k1", v="1,3,5", extra="reset_conc=>0,"),
        _continued(100, [(1, 3), (3, 9), (5, 15)]),
    ),
    # 75.0000; the standard elements gave 37.51
    "set-between-runs": ("declared", RUN + 'setParameter("k1",5)\n' + RUN, [_from(_A1, 5, 15)]),
    # 75.0000: the run that the second continues sets k1 from the base model
    "set-before-a-run-that-a-run-continues": (
        "declared",
        'setParameter("k1",5)\n' + RUN + RUN,
        [_from(_eq(5, 15), 5, 15)],
    ),
    # 87.5040, 58.3333: k2 set to a number no longer follows k1
    "a-number-ends-the-assignment": (
        "declared",
        'setParameter("k2",7)\n' + RUN + 'setParameter("k1",5)\n' + RUN,
        [_from(_eq(1, 7), 5, 7)],
    ),
    # 75.0000: k2 is back to k1*3, and follows k1
    "reset-to-the-networks-expression-between-runs": (
        "declared",
        "saveParameters()\n"
        + RUN
        + 'setParameter("k2",7)\n'
        + RUN
        + 'resetParameters()\nsetParameter("k1",5)\n'
        + RUN,
        [_from(_from(_A1, 1, 7), 5, 15)],
    ),
    # 75.0000: the reset puts back k2's expression, which reads k1 = 5;
    # the number the SBML holds for k2 is 3 (37.50)
    "reset-to-the-networks-expression-after-what-it-reads-moved": (
        "declared",
        'setParameter("k1",5)\nsaveParameters()\n'
        + RUN
        + 'setParameter("k2",7)\n'
        + RUN
        + "resetParameters()\n"
        + RUN,
        [_from(_from(_eq(5, 15), 5, 7), 5, 15)],
    ),
    # 75.0084, 75.0000, 75.0000
    "scan-that-continues-a-run": (
        "declared",
        RUN + SCAN.format(p="k1", v="1,3,5", extra="reset_conc=>0,"),
        _continued(_A1, [(1, 3), (3, 9), (5, 15)]),
    ),
    # 75.4579, 75.0002, 75.0000 up, then back down from there
    "bifurcate": (
        "declared",
        'bifurcate({method=>"ode",parameter=>"k1",par_min=>1,par_max=>3,n_scan_pts=>3,'
        "t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n",
        _continued(100, [(1, 3), (2, 6), (3, 9), (3, 9), (2, 6), (1, 3)]),
    ),
    # 68.3262, 66.6708, 66.6667: k2 reads kx, which reads k1
    "scan-through-two-derived-parameters": (
        "chained",
        SCAN.format(p="k1", v="1,3,5", extra=""),
        [_eq(k, 2 * k) for k in (1, 3, 5)],
    ),
    # 75.4579, 66.6708, 54.5462: k2 = min(k1*3, 6), said as a piecewise
    "scan-of-what-a-min-reads": (
        "capped",
        SCAN.format(p="k1", v="1,3,5", extra=""),
        [_eq(1, 3), _eq(3, 6), _eq(5, 6)],
    ),
    # 75.4579, 66.6708, 54.5462: k2 = if(k1>2, 6, k1*3)
    "scan-of-what-an-if-reads": (
        "stepped",
        SCAN.format(p="k1", v="1,3,5", extra=""),
        [_eq(1, 3), _eq(3, 6), _eq(5, 6)],
    ),
    # 28.3834, 56.7668, 85.1501; the standard elements were flat at 56.77
    "dose-scan": (
        "seeded",
        SCAN.format(p="A0", v="50,100,150", extra=""),
        [_eq(1, 1, a0=a) for a in (50, 100, 150)],
    ),
    # 28.3834, 25.4579, 25.0620: A follows A0 at the first point only
    "dose-scan-without-reset": (
        "seeded",
        SCAN.format(p="A0", v="50,100,150", extra="reset_conc=>0,"),
        _continued(50, [(1, 1)] * 3, total=50),
    ),
    # 28.3834, 25.4579, 25.0620 up, then on down
    "dose-bifurcate": (
        "seeded",
        'bifurcate({method=>"ode",parameter=>"A0",par_min=>50,par_max=>150,n_scan_pts=>3,'
        "t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n",
        _continued(50, [(1, 1)] * 6, total=50),
    ),
    # 25.0084: the chain replays the sweep from A = A0 = 50
    "run-after-a-dose-scan-without-reset": (
        "seeded",
        SCAN.format(p="A0", v="50,100,150", extra="reset_conc=>0,") + RUN,
        [_from(_continued(50, [(1, 1)] * 3, total=50)[-1], 1, 1, total=50)],
    ),
    # 75.0000: k2 follows k1 through the replayed sweep and after it
    "run-after-a-scan-without-reset": (
        "declared",
        SCAN.format(p="k1", v="1,3,5", extra="reset_conc=>0,") + 'setParameter("k1",2)\n' + RUN,
        [_from(_continued(100, [(1, 3), (3, 9), (5, 15)])[-1], 2, 6)],
    ),
    # 25.4579: the run the second continues starts from A = A0 = 50
    "seed-set-before-a-run-that-a-run-continues": (
        "seeded",
        'setParameter("A0",50)\n' + RUN + RUN,
        [_from(_eq(1, 1, a0=50), 1, 1, total=50)],
    ),
    # 45.4134 at each point: setConcentration replaced A's seed
    "dose-scan-after-the-seed-is-replaced": (
        "seeded",
        'setConcentration("A()",80)\n' + SCAN.format(p="A0", v="50,100,150", extra=""),
        [_eq(1, 1, a0=80)] * 3,
    ),
    # 40.7326: so did this one, before the seed's parameter was set
    "seed-set-after-the-seed-is-replaced": (
        "seeded",
        'setConcentration("A()",80)\nsetParameter("A0",50)\n' + RUN + RUN,
        [_from(_eq(1, 1, a0=80), 1, 1, total=80)],
    ),
    # 50.9158: a run has made A a number, which A0 no longer moves
    "seed-set-between-runs": (
        "seeded",
        RUN + 'setParameter("A0",50)\n' + RUN,
        [_from(_eq(1, 1), 1, 1)],
    ),
    # 28.3834, 56.7668, 85.1501: A0 = Ab*2, and A from A0
    "dose-scan-through-a-derived-parameter": (
        "seed-of-derived",
        SCAN.format(p="Ab", v="25,50,75", extra=""),
        [_eq(1, 1, a0=2 * ab) for ab in (25, 50, 75)],
    ),
    # 85.7143: kr = k2*2 reads k2 = k1*3 at k1 = 5; the number the SBML holds
    # for k2 is 3 (54.5462)
    "an-expression-of-a-derived-parameter-after-what-it-reads-moved": (
        "beside",
        'setParameter("k1",5)\nsetParameter("kr","k2*2")\n' + RUN,
        [_eq(5, 30)],
    ),
    # 45.0000: A = k2*4 reads k2 = 15 (41.2500 from A = 12)
    "an-amount-of-a-derived-parameter-after-what-it-reads-moved": (
        "declared",
        'setParameter("k1",5)\nsetConcentration("A()","k2*4")\n' + RUN,
        [_eq(5, 15, a0=60)],
    ),
    # 34.0601: addConcentration adds to the seed A0 = 50; the amount the SBML
    # holds for A is 100 (62.4434)
    "add-to-a-seed-after-its-parameter-moved": (
        "seeded",
        'setParameter("A0",50)\naddConcentration("A()",10)\n' + RUN,
        [_eq(1, 1, a0=60)],
    ),
    # 30.5495: and the run a second continues, replayed on the base model
    "add-to-a-seed-after-its-parameter-moved-continued": (
        "seeded",
        'setParameter("A0",50)\naddConcentration("A()",10)\n' + RUN + RUN,
        [_from(_eq(1, 1, a0=60), 1, 1, total=60)],
    ),
    # 34.0601: the add made A a number, which A0 = 70 after it does not move
    "add-to-a-seed-between-two-values-of-its-parameter": (
        "seeded",
        'setParameter("A0",50)\naddConcentration("A()",10)\nsetParameter("A0",70)\n' + RUN,
        [_eq(1, 1, a0=60)],
    ),
    # 62.4434: an add to an amount that is an expression adds to the seed, 100
    "add-to-an-expression": (
        "seeded",
        'setConcentration("A()","A0*2")\naddConcentration("A()",10)\nsetParameter("A0",30)\n'
        + RUN,
        [_eq(1, 1, a0=110)],
    ),
    # 81.1346: after a run too, where the model holds the run's amount
    "add-to-an-expression-after-a-run": (
        "seeded",
        RUN + 'setConcentration("A()","A0*2")\naddConcentration("A()",10)\n' + RUN,
        [_from(110, 1, 1, total=110 + 100 - _eq(1, 1))],
    ),
    # 62.4434: and after a change that set the species, to the seed, 100
    "add-to-an-expression-of-a-species-set-before": (
        "plain",
        'setConcentration("A()",30)\nsetConcentration("A()","k1*20")\n'
        'addConcentration("A()",10)\n' + RUN,
        [_eq(1, 1, a0=110)],
    ),
    # 36.8984: a second add is to the number the first made, not to the seed
    "two-adds-to-a-seed-after-its-parameter-moved": (
        "seeded",
        'setParameter("A0",50)\naddConcentration("A()",10)\naddConcentration("A()",5)\n' + RUN,
        [_eq(1, 1, a0=65)],
    ),
    # 62.4434: the seed, 100, though a change before the expression set A
    "add-to-an-expression-of-a-seed-set-before": (
        "seeded",
        'setConcentration("A()",30)\nsetConcentration("A()","A0*2")\n'
        'addConcentration("A()",10)\n' + RUN,
        [_eq(1, 1, a0=110)],
    ),
    # 101.2271: the add reads k2 = 7 (B = 35), though the reset after it puts
    # k2 back to k1*3; the change that set k2 was left out (86.5018, B = 15)
    "an-add-that-reads-a-number-a-reset-puts-back-after-it": (
        "declared",
        'saveParameters()\nsetParameter("k2",7)\naddConcentration("B()","k2*5")\n'
        "resetParameters()\n" + RUN,
        [_from(100, 1, 3, total=135)],
    ),
    # 93.3333: B = k2*4 reads k2 = kx/3*2 = 10, which reads kx = k1*3 = 15
    "an-amount-of-a-parameter-derived-twice-after-what-it-reads-moved": (
        "chained",
        'setParameter("k1",5)\nsetConcentration("B()","k2*4")\n' + RUN,
        [_from(100, 5, 10, total=140)],
    ),
    # 47.0745, 46.6786, 46.6250: A = A0 reads A0 = Ab*2 at the first point
    "scan-that-continues-a-run-of-what-an-amount-reads-through-a-derived-parameter": (
        "seed-of-derived",
        RUN
        + 'setConcentration("A()","A0")\n'
        + SCAN.format(p="Ab", v="25,50,75", extra="reset_conc=>0,"),
        _continued(50, [(1, 1)] * 3, total=50 + 100 - _eq(1, 1)),
    ),
    # 25.0084: the chain replays the sweep from A = A0 = Ab*2 = 50
    "run-after-a-dose-scan-without-reset-through-a-derived-parameter": (
        "seed-of-derived",
        SCAN.format(p="Ab", v="25,50,75", extra="reset_conc=>0,") + RUN,
        [_from(_continued(50, [(1, 1)] * 3, total=50)[-1], 1, 1, total=50)],
    ),
    # 11.3534, 22.7067, 34.0601; the standard elements gave 113.53 at each
    "scan-of-what-an-amount-set-to-an-expression-reads": (
        "seeded",
        'setConcentration("A()","A0*2")\n' + SCAN.format(p="A0", v="10,20,30", extra=""),
        [_eq(1, 1, a0=2 * a) for a in (10, 20, 30)],
    ),
    # 11.3534, 10.1832, 10.0248: A reads A0 at the first point only
    "scan-without-reset-of-what-an-amount-set-to-an-expression-reads": (
        "seeded",
        'setConcentration("A()","A0*2")\n'
        + SCAN.format(p="A0", v="10,20,30", extra="reset_conc=>0,"),
        _continued(20, [(1, 1)] * 3, total=20),
    ),
    # 30.0445, 31.4039, 31.5878: so does one set after a run
    "scan-that-continues-a-run-of-what-an-amount-set-to-an-expression-reads": (
        "seeded",
        RUN
        + 'setConcentration("A()","A0*2")\n'
        + SCAN.format(p="A0", v="10,20,30", extra="reset_conc=>0,"),
        _continued(20, [(1, 1)] * 3, total=20 + 100 - _eq(1, 1)),
    ),
}
_NETWORKS = {
    "plain": (NET, MODEL),
    "beside": (BESIDE_NET, BESIDE_MODEL),
    "declared": (DECLARED_NET, DECLARED_MODEL),
    "capped": (CAPPED_NET, CAPPED_MODEL),
    "stepped": (STEPPED_NET, STEPPED_MODEL),
    "chained": (CHAINED_NET, CHAINED_MODEL),
    "seeded": (SEEDED_NET, SEEDED_MODEL),
    "seed-of-derived": (SEED_OF_DERIVED_NET, SEED_OF_DERIVED_MODEL),
}


def _replay_points(run) -> list[float]:
    if run.result is not None:
        return [_end(run.result)]
    return [_end(r) for r in (*run.scan_results, *run.backward_results)]


@pytest.fixture(params=sorted(INITIAL_ASSIGNMENT_CASES))
def assigned(request, tmp_path):
    """A case of INITIAL_ASSIGNMENT_CASES packed, with what packing it said."""
    network, actions, want = INITIAL_ASSIGNMENT_CASES[request.param]
    net, model = _NETWORKS[network]
    report, said = _pack(
        tmp_path, "generate_network({overwrite=>1})\n" + actions, net=net, model=model
    )
    return tmp_path / "m.omex", report, said, want


def test_the_sedml_sets_what_an_initial_assignment_computes_as_the_model_runs(assigned):
    """The standard elements, run on libRoadRunner, give BNG2.pl's numbers: at
    each point of a sweep, or at the end of a chain."""
    omex, report, said, want = assigned
    points = _run_points(omex)
    got = points[sorted(points, key=lambda r: int(r.split("_")[1]))[-1]]["observable:Atot"]
    assert got == pytest.approx(want if len(got) > 1 else want[-1:], rel=1e-6)


def test_following_an_initial_assignment_is_no_loss(assigned):
    omex, report, said, want = assigned
    assert report.sedml_lossy == []
    assert not [s for s in said if "as the model runs" in s]


def test_the_standard_elements_read_back_run_what_bng2pl_runs(assigned):
    """bngsim's reader of the standard elements, with the archive's SBML,
    reads the setValues that follow an initialAssignment, and the replay
    runs BNG2.pl's numbers."""
    omex, report, said, want = assigned
    sed = zipfile.ZipFile(omex).read("simulation.sedml").decode()
    bare = re.sub(r"<annotation>.*?</annotation>", "", sed, flags=re.S)
    with read_omex(omex) as archive:
        sbml = archive.path_of(archive.master_model_entry())
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConversionWarning)
            spec = read_sedml_protocol(bare, sbml=sbml)
        assert spec.lossy == ()
        runs = bngsim.run_protocol(archive.load_model(), spec, rtol=1e-10, atol=1e-12)
    assert _replay_points(runs[-1]) == pytest.approx(want, rel=1e-6)


def test_the_archive_replays_what_bng2pl_runs(assigned):
    omex, report, said, want = assigned
    with read_omex(omex) as archive:
        runs = bngsim.run_protocol(
            archive.load_model(), archive.load_full_protocol(), rtol=1e-10, atol=1e-12
        )
    assert _replay_points(runs[-1]) == pytest.approx(want, rel=1e-6)


def test_a_sweep_sets_what_an_assignment_computes_from_its_range(tmp_path):
    """k2 = k1*3 at each point of a scan of k1 reads the range k1 is set
    from, not k1's value, which a simulator may read before the point's
    changes."""
    _pack(
        tmp_path,
        "generate_network({overwrite=>1})\n" + SCAN.format(p="k1", v="1,3,5", extra=""),
        net=DECLARED_NET,
        model=DECLARED_MODEL,
    )
    sed = ET.fromstring(zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml"))
    (rt,) = sed.iter(f"{{{SED}}}repeatedTask")
    swept, k2 = rt.iterfind("s:listOfChanges/s:setValue", NS)
    assert swept.get("target").endswith("[@id='k1']")
    assert k2.get("target").endswith("[@id='k2']")
    assert _formula(k2.find(f"{{{MATHML}}}math")).replace(" ", "") == f"{rt.get('range')}*3"
    assert k2.find("s:listOfVariables", NS) is None


@pytest.mark.parametrize(
    ("actions", "removed"),
    [
        ('setParameter("k1",5)\n' + RUN, False),
        ('setParameter("k1",5)\nsetParameter("k2",7)\n' + RUN, True),
    ],
    ids=["kept", "replaced"],
)
def test_a_derived_model_keeps_the_assignments_the_protocol_does_not_replace(
    tmp_path, actions, removed
):
    """Changes made before the model is loaded leave k2's initialAssignment,
    which computes it from them, unless the protocol sets k2 itself."""
    _pack(
        tmp_path,
        "generate_network({overwrite=>1})\n" + actions,
        net=DECLARED_NET,
        model=DECLARED_MODEL,
    )
    sed = ET.fromstring(zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml"))
    changes = [c for m in sed.iter(f"{{{SED}}}model") for c in m.iterfind("s:listOfChanges/*", NS)]
    assert any(c.tag.endswith("removeXML") for c in changes) == removed
    assert not [c for c in changes if c.tag.endswith("computeChange")]


def test_a_parameter_a_reset_puts_back_keeps_its_assignment_under_an_earlier_change(tmp_path):
    """k2 = 7, an addition that reads it, then a reset back to k2 = k1*3: the
    derived model sets k2 to 7 for the addition, and leaves k2's
    initialAssignment, which computes k2 when the model is loaded."""
    _pack(
        tmp_path,
        "generate_network({overwrite=>1})\nsaveParameters()\n"
        'setParameter("k2",7)\naddConcentration("B()","k2*5")\nresetParameters()\n' + RUN,
        net=DECLARED_NET,
        model=DECLARED_MODEL,
    )
    sed = ET.fromstring(zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml"))
    changes = [c for m in sed.iter(f"{{{SED}}}model") for c in m.iterfind("s:listOfChanges/*", NS)]
    assert not [c for c in changes if c.tag.endswith("removeXML")]
    first = changes[0]
    assert first.tag.endswith("changeAttribute") and first.get("newValue") == "7"
    assert first.get("target").endswith("[@id='k2']/@value")


def test_a_change_kept_under_an_assignment_is_followed_by_what_the_assignment_computes(tmp_path):
    """kx = k1*3 and k2 = kx/3*2 declared; kx = 9, then a reset back. k2,
    which reads kx, is set again between the two, so kx = 9 stays in the
    derived model, under kx's initialAssignment. Alone, it left a reader of
    the changes with kx = 9 for the run (A(1) = 85.7273), where BNG2.pl's kx
    is back at k1*3 (A(1) = 68.3262). kx's last change, to what the
    assignment computes, is written after it."""
    _pack(
        tmp_path,
        "generate_network({overwrite=>1})\nsaveParameters()\n"
        'setParameter("kx",9)\nresetParameters()\n' + RUN,
        net=CHAINED_NET,
        model=CHAINED_MODEL,
    )
    want = _eq(1, 2)
    assert _run_report(tmp_path / "m.omex")["report_0"]["observable:Atot"] == pytest.approx(
        want, rel=1e-6
    )
    sed = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml").decode()
    bare = re.sub(r"<annotation>.*?</annotation>", "", sed, flags=re.S)
    changes = list(ET.fromstring(bare).iterfind("s:listOfModels/s:model/s:listOfChanges/*", NS))
    assert [c.tag.rsplit("}", 1)[1] for c in changes] == ["changeAttribute", "computeChange"]
    assert all(c.get("target").endswith("[@id='kx']/@value") for c in changes)
    with read_omex(tmp_path / "m.omex") as archive:
        sbml = archive.path_of(archive.master_model_entry())
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConversionWarning)
            specs = [read_sedml_protocol(bare, sbml=sbml), read_sedml_protocol(bare)]
        for spec in specs:
            (run,) = bngsim.run_protocol(archive.load_model(), spec, rtol=1e-10, atol=1e-12)
            assert _end(run.result) == pytest.approx(want, rel=1e-6)


def test_an_add_to_an_expression_whose_seed_the_network_lost_is_recorded(tmp_path):
    """BNG2.pl's addConcentration to an amount set to an expression adds to
    the species' declared seed (A = 110, with A(1) = 101.4103 beside
    k2 = k1*3); an action before generate_network changed the amount the
    network holds, and the standard elements add to the expression (310)."""
    report, said = _pack(
        tmp_path,
        'setConcentration("A()",70)\ngenerate_network({overwrite=>1})\n'
        'setConcentration("A()","A0*3")\naddConcentration("A()",10)\n' + RUN,
        net=SEEDED_NET.replace("1 A() A0", "1 A() 70"),
        model=SEEDED_MODEL,
    )
    (note,) = [n for n in report.sedml_lossy if "declared amount" in n]
    assert note in said
    # What the note says they do: 10 added to A0*3 = 300.
    got = _run_report(tmp_path / "m.omex")["report_0"]["observable:Atot"]
    assert got == pytest.approx(_eq(1, 1, a0=310), rel=1e-6)


def _bare_and_sbml(omex: Path) -> tuple[str, str]:
    """The archive's SED-ML without its bngsim annotation, and its SBML."""
    sed = zipfile.ZipFile(omex).read("simulation.sedml").decode()
    with read_omex(omex) as archive:
        sbml = Path(archive.path_of(archive.master_model_entry())).read_text()
    return re.sub(r"<annotation>.*?</annotation>", "", sed, flags=re.S), sbml


@pytest.mark.parametrize(
    ("network", "actions", "want"),
    [
        # k2 = k1*3 declared: the chain sets k2 again after k1, as the model runs.
        (
            "declared",
            'setParameter("k1",5)\n' + RUN + RUN,
            [_eq(5, 15), _from(_eq(5, 15), 5, 15)],
        ),
        # The seed A() A0: the chain sets A to it after A0.
        (
            "seeded",
            'setParameter("A0",50)\n' + RUN + RUN,
            [_eq(1, 1, a0=50), _from(_eq(1, 1, a0=50), 1, 1, total=50)],
        ),
        # kx = k1*3 and k2 = kx/3*2, and two changes to k1: each sets both again.
        (
            "chained",
            'setParameter("k1",5)\nsetParameter("k1",2)\n' + RUN + RUN,
            [_eq(2, 4), _from(_eq(2, 4), 2, 4)],
        ),
        # A dose scan that goes on, as the first run of a chain, with the seed
        # A() A0 and A0 = Ab*2: A0 and A are set at the first point, and A0 at
        # each point.
        (
            "seed-of-derived",
            SCAN.format(p="Ab", v="25,50", extra="reset_conc=>0,") + RUN,
            _continued(50, [(1, 1)] * 3, total=50)[1:],
        ),
        # k2 = k1*3 is said again after k1, then the action sets k2 itself.
        (
            "declared",
            'setParameter("k1",5)\nsetParameter("k2","k1*2")\n' + RUN + RUN,
            [_eq(5, 10), _from(_eq(5, 10), 5, 10)],
        ),
        # kx = k1*3 and k2 = kx/3*2: with kx set, k2 no longer reads k1.
        (
            "chained",
            'setParameter("kx",6)\nsetParameter("k1",5)\n' + RUN + RUN,
            [_eq(5, 4), _from(_eq(5, 4), 5, 4)],
        ),
        # A dose scan from exp(ln(8)) = 7.999999999999998, as BNG2.pl's log
        # scale runs it: the chain's setValue of it is read to every digit, as
        # the changeAttribute of the first run's own model is.
        (
            "seeded",
            'parameter_scan({method=>"ode",parameter=>"A0",par_min=>8,par_max=>120,'
            "n_scan_pts=>3,log_scale=>1,reset_conc=>0,t_end=>1,n_steps=>1,atol=>1e-12,"
            "rtol=>1e-10})\n" + RUN,
            _continued(math.exp(math.log(8)), [(1, 1)] * 4, total=math.exp(math.log(8)))[2:],
        ),
        # k2 = k1*3 is said again after k1, and the action sets k2 itself after
        # a change that does not read it.
        (
            "declared",
            'setParameter("k1",5)\nsetConcentration("B()",10)\nsetParameter("k2",7)\n' + RUN + RUN,
            [_from(100, 5, 7, total=110), _from(_from(100, 5, 7, total=110), 5, 7, total=110)],
        ),
        # kr = k1*2, which an action set, is said again after each change to
        # k1, before k2 = k1*3 is: it does not read k2, which is behind then.
        (
            "beside",
            'setParameter("kr","k1*2")\nsetParameter("k1",5)\nsetParameter("k1",2)\n' + RUN + RUN,
            [_eq(2, 4), _from(_eq(2, 4), 2, 4)],
        ),
    ],
    ids=[
        "derived",
        "seed",
        "derived-of-derived",
        "sweep",
        "replaced",
        "cut-off",
        "digits",
        "replaced-after-another-change",
        "another-change-while-behind",
    ],
)
def test_a_chain_read_back_runs_its_first_run_once(tmp_path, network, actions, want):
    """A run that continues the first replays it in a chain, on the model as
    it runs, where what an initialAssignment computes is set again after what
    it reads. The first run's own task has no such changes: its model's
    assignments compute them. Read back, the two are one run from one state,
    not the first run twice."""
    net, model = _NETWORKS[network]
    _pack(tmp_path, "generate_network({overwrite=>1})\n" + actions, net=net, model=model)
    bare, sbml = _bare_and_sbml(tmp_path / "m.omex")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        spec = read_sedml_protocol(bare, sbml=sbml)
    assert spec.lossy == ()
    assert len(spec.experiments) == 2
    with read_omex(tmp_path / "m.omex") as archive:
        runs = bngsim.run_protocol(archive.load_model(), spec, rtol=1e-10, atol=1e-12)
    assert [_replay_points(run)[-1] for run in runs] == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize(
    ("network", "experiments", "said"),
    [("declared", 3, True), ("plain", 2, False)],
    ids=["derived", "plain"],
)
def test_a_chain_read_without_the_sbml_says_where_its_first_run_may_be_read_twice(
    tmp_path, network, experiments, said
):
    """Without the SBML the reader cannot tell that the chain's setValue of
    k2 says again what an initialAssignment computes: the first run's own
    task is read as a run of its own, with the same numbers, and the reader
    says so. With no such setValue the two are one run, and it says nothing."""
    net, model = _NETWORKS[network]
    _pack(
        tmp_path,
        'generate_network({overwrite=>1})\nsetParameter("k1",5)\n' + RUN + RUN,
        net=net,
        model=model,
    )
    bare, _ = _bare_and_sbml(tmp_path / "m.omex")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = read_sedml_protocol(bare)
    notes = [str(w.message) for w in caught if issubclass(w.category, ConversionWarning)]
    assert spec.lossy == ()
    assert len(spec.experiments) == experiments
    assert bool(notes) == said
    if said:
        (note,) = notes
        assert "task_0" in note and "chain_1" in note and "['k2']" in note and "sbml=" in note
    k2 = 15 if network == "declared" else 1
    with read_omex(tmp_path / "m.omex") as archive:
        runs = bngsim.run_protocol(archive.load_model(), spec, rtol=1e-10, atol=1e-12)
    first = _eq(5, k2)
    assert [_end(run.result) for run in runs] == pytest.approx(
        [first] * (experiments - 1) + [_from(first, 5, k2)], rel=1e-6
    )


def _before(changes: list[ET.Element]) -> list[ET.Element]:
    """k2 = k1*3 before k1 = 5, not after: the run is at k2 = 3."""
    k1, k2 = changes
    return [k2, k1]


def _read_between(changes: list[ET.Element]) -> list[ET.Element]:
    """k2 = k1*3 before k1 = 5 and again after the addition of k2 to B, which
    reads k2 = 3."""
    k1, k2, add = changes
    return [k2, k1, add, copy.deepcopy(k2)]


def _behind(changes: list[ET.Element]) -> list[ET.Element]:
    """kx = k1*3 and k2 = kx/3*2, then k1 = 5, then k2 before kx: k2 is
    computed from the kx of k1 = 1."""
    k1, kx, k2 = changes
    return [copy.deepcopy(kx), copy.deepcopy(k2), k1, k2, kx]


@pytest.mark.parametrize(
    ("network", "actions", "edit"),
    [
        ("declared", 'setParameter("k1",5)\n', _before),
        ("declared", 'setParameter("k1",5)\naddConcentration("B()","k2")\n', _read_between),
        ("chained", 'setParameter("k1",5)\n', _behind),
    ],
    ids=["before", "read-between", "behind"],
)
def test_a_chain_whose_first_run_is_not_the_first_task_is_read_as_its_own_run(
    tmp_path, network, actions, edit
):
    """A chain as written is read as two runs. With its setValues in an order
    that leaves a value an initialAssignment computes behind what it reads,
    where the run or another change reads it, its first run is not the first
    task's run, and is read as a run of its own."""
    net, model = _NETWORKS[network]
    _pack(
        tmp_path, "generate_network({overwrite=>1})\n" + actions + RUN + RUN, net=net, model=model
    )
    bare, sbml = _bare_and_sbml(tmp_path / "m.omex")
    assert len(read_sedml_protocol(bare, sbml=sbml).experiments) == 2
    root = ET.fromstring(bare)
    changes = next(
        rt.find("s:listOfChanges", NS)
        for rt in root.iter(f"{{{SED}}}repeatedTask")
        if rt.find("s:listOfChanges", NS) is not None
    )
    changes[:] = edit(list(changes))
    spec = read_sedml_protocol(ET.tostring(root, encoding="unicode"), sbml=sbml)
    assert len(spec.experiments) == 3


def test_a_chain_that_resets_before_its_first_run_is_read(tmp_path):
    """A chain that sets k2, then resets the model in the repeatedTask it
    runs first: its changes before the first run are read as they are."""
    _pack(
        tmp_path,
        "generate_network({overwrite=>1})\n" + 'setParameter("k1",5)\n' + RUN + RUN,
        net=DECLARED_NET,
        model=DECLARED_MODEL,
    )
    bare, sbml = _bare_and_sbml(tmp_path / "m.omex")
    root = ET.fromstring(bare)
    step, chain = root.iter(f"{{{SED}}}repeatedTask")
    assert step.get("resetModel") == "false" and chain.get("resetModel") == "true"
    step.set("resetModel", "true")
    k2 = copy.deepcopy(step.find("s:listOfChanges", NS))
    k2.remove(k2[0])
    chain.insert(1, k2)
    spec = read_sedml_protocol(ET.tostring(root, encoding="unicode"), sbml=sbml)
    assert len(spec.experiments) == 3
    kinds = [getattr(s, "kind", None) for s in spec.steps]
    assert kinds.count("reset_concentrations") == 2


def test_a_species_set_at_each_point_of_a_sweep_that_goes_on_is_not_a_seed(tmp_path):
    """A sweep that resets sets a species whose seed follows the swept
    parameter at each point, which a replay's scan does itself. One that goes
    on from point to point and sets it at each is no scan a replay runs."""
    _pack(
        tmp_path,
        "generate_network({overwrite=>1})\n" + SCAN.format(p="A0", v="50,100,150", extra=""),
        net=SEEDED_NET,
        model=SEEDED_MODEL,
    )
    with read_omex(tmp_path / "m.omex") as archive:
        sbml = Path(archive.path_of(archive.master_model_entry())).read_text()
    sed = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml").decode()
    bare = re.sub(r"<annotation>.*?</annotation>", "", sed, flags=re.S)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        spec = read_sedml_protocol(bare, sbml=sbml)
    assert spec.lossy == ()
    # The scan alone: the seed is the model's, not a setConcentration read.
    assert [type(step).__name__ for step in spec.steps] == ["Experiment"]
    going_on = bare.replace('id="rtask_0" resetModel="true"', 'id="rtask_0" resetModel="false"')
    assert going_on != bare
    with pytest.warns(ConversionWarning, match="from its range beside the swept parameter"):
        spec = read_sedml_protocol(going_on, sbml=sbml)
    assert len(spec.lossy) == 1


def test_a_piecewise_of_several_pieces_is_read_as_ifs_in_their_order():
    """piecewise(10, x < 1, 20, x < 2, 30) takes its first piece whose
    condition holds: if(x < 1, 10, if(x < 2, 20, 30))."""
    from bngsim.convert._sedml import _l3_formula

    math = ET.fromstring(
        f'<math xmlns="{MATHML}"><piecewise>'
        "<piece><cn>10</cn><apply><lt/><ci>x</ci><cn>1</cn></apply></piece>"
        "<piece><cn>20</cn><apply><lt/><ci>x</ci><cn>2</cn></apply></piece>"
        "<otherwise><cn>30</cn></otherwise></piecewise></math>"
    )
    assert _l3_formula(math).replace(" ", "") == "if(x<1,10,if(x<2,20,30))"


def test_an_expression_with_a_min_and_an_if_is_read_back_and_replays(tmp_path):
    """setParameter("k2","if(k1>2,k1*3,min(k1/2,7))") before a run, then k1 = 3
    and a run on: BNG2.pl's A(1) is 74.9998 after 48.2087. The standard
    elements say the min as a piecewise, and the reader says each piecewise
    as BNGL's if, which a replay evaluates (it refused piecewise(...))."""
    _pack(
        tmp_path,
        "generate_network({overwrite=>1})\n"
        'setParameter("k2","if(k1>2,k1*3,min(k1/2,7))")\n' + RUN + 'setParameter("k1",3)\n' + RUN,
    )
    want = _from(_eq(1, 0.5), 3, 9)
    points = _run_points(tmp_path / "m.omex")
    assert points["report_1"]["observable:Atot"] == pytest.approx([want], rel=1e-6)
    sed = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml").decode()
    bare = re.sub(r"<annotation>.*?</annotation>", "", sed, flags=re.S)
    assert "min" not in bare
    with read_omex(tmp_path / "m.omex") as archive:
        sbml = archive.path_of(archive.master_model_entry())
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConversionWarning)
            spec = read_sedml_protocol(bare, sbml=sbml)
        runs = bngsim.run_protocol(archive.load_model(), spec, rtol=1e-10, atol=1e-12)
    assert _replay_points(runs[-1]) == pytest.approx([want], rel=1e-6)
