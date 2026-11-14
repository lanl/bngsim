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


def _evaluate(math_elem: ET.Element, env: dict[str, float]) -> float:
    names = {"pow": pow, "exp": math.exp, "log": math.log, "sqrt": math.sqrt}
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

    def run(task: ET.Element, rr):
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
                ranges[rng.get("id")] = [lo + k * (hi - lo) / n for k in range(n + 1)]
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
        return rr

    generators = {g.get("id"): g for g in sed.iter(f"{{{SED}}}dataGenerator")}
    out: dict[str, dict[str, float]] = {}
    for report in sed.iter(f"{{{SED}}}report"):
        values: dict[str, float] = {}
        for ds in report.iter(f"{{{SED}}}dataSet"):
            var = generators[ds.get("dataReference")].find(".//s:variable", NS)
            rr = run(tasks[var.get("taskReference")], None)
            values[ds.get("label")] = (
                float(rr.getCurrentTime()) if var.get("symbol") else rr[_rr_key(var.get("target"))]
            )
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
