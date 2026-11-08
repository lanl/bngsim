"""A protocol's SED-ML runs each experiment from the state BNG2.pl runs it from.

BNG2.pl starts each run from the state the steps before it left: an
equilibrate-then-stimulate protocol stimulates the equilibrated model. In
SED-ML a task starts from its model's initial state, but within a repeatedTask
"model states are not reset unless explicitly told to reset with the
'resetModel=true' attribute" (L1V4; SED-ML/sed-ml#54), so ordered subtasks run
on one model state. ``write_sedml_protocol`` writes such a run as a
repeatedTask that resets once and replays the runs since the last clean state,
with the actions between them as ``setValue`` changes.

The oracle is the spec itself, run: ``_SpecRunner`` below executes a document
by SED-ML's task semantics on bngsim, and each experiment's element must
reproduce bngsim's BNG2.pl replay (``run_protocol``) of the same actions. The
document must also be well formed on its own terms: every target an XPath under
a declared ``sbml`` prefix that names exactly one element of the SBML.
"""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
import zipfile

import bngsim
import libsbml
import numpy as np
import pytest
from bngsim import ConversionWarning
from bngsim.convert import (
    net_to_omex,
    parse_bngl_protocol,
    read_sedml_protocol,
    write_sedml,
    write_sedml_protocol,
)

SEDML = "http://sed-ml.org/sed-ml/level1/version3"
MATHML = "http://www.w3.org/1998/Math/MathML"

NET = """begin parameters
    1 kf 1
    2 kr 0.5
    3 k0 0.25
    4 A0 100
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
    2 Btot 2
end groups
"""
MODEL = """begin model
begin parameters
kf 1
kr 0.5
k0 0.25
A0 100
end parameters
begin molecule types
A()
B()
end molecule types
begin seed species
A() A0
B() 0
end seed species
begin observables
Molecules Atot A()
Molecules Btot B()
end observables
begin reaction rules
A() <-> B() kf, kr
end reaction rules
end model
generate_network({overwrite=>1})
"""
TOLS = "atol=>1e-12,rtol=>1e-10"
RUN = f'simulate({{method=>"ode",t_end=>2,n_steps=>4,{TOLS}}})'
SCAN = (
    'parameter_scan({{method=>"ode",t_end=>1,n_steps=>2,parameter=>"kf",'
    "par_min=>0.5,par_max=>2,n_scan_pts=>4,reset_conc=>{reset}," + TOLS + "}})"
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class _SpecRunner:
    """SED-ML L1V3's task semantics, run on bngsim: what a simulator that
    follows the spec computes from a document. A task resets its model and runs
    its simulation from ``initialTime``, reporting from ``outputStartTime``. A
    repeatedTask, at each value of its range, resets its model only where
    ``resetModel`` is true, applies its ``setValue`` changes, and runs its
    subtasks in order, on one model state."""

    def __init__(self, sedml: str, sbml: str, net_path):
        self.doc = ET.fromstring(sedml)
        root = ET.fromstring(sbml)
        self.species = {
            e.get("id"): e.get("name") or e.get("id")
            for e in root.iter()
            if _local(e.tag) == "species"
        }
        self.params = {
            e.get("id"): e.get("name") or e.get("id")
            for e in root.iter()
            if _local(e.tag) == "parameter"
        }
        self.net = str(net_path)
        self.by_id = {e.get("id"): e for e in self.doc.iter() if e.get("id")}
        self.model: bngsim.Model | None = None
        self.sims: dict[str, bngsim.Simulator] = {}

    # ── the model ──────────────────────────────────────────────────────────
    def _name(self, target: str) -> tuple[str, str]:
        sid = re.search(r"@id='([^']+)'", target).group(1)
        if "listOfSpecies" in target:
            return "species", self.species[sid]
        return "param", self.params[sid]

    def _set(self, model, target: str, value: float) -> None:
        kind, name = self._name(target)
        if kind == "species":
            model.set_concentration(name, value)
        else:
            model.set_param(name, value)

    def _read(self, model, target: str) -> float:
        kind, name = self._name(target)
        return float(model.get_concentration(name) if kind == "species" else model.get_param(name))

    def _value(self, change: ET.Element, model, extra: dict[str, float]) -> float:
        names = dict(extra)
        for var in change.iter(f"{{{SEDML}}}variable"):
            names[var.get("id")] = self._read(model, var.get("target"))
        mathml = ET.tostring(next(change.iter(f"{{{MATHML}}}math")), encoding="unicode")
        formula = libsbml.formulaToL3String(libsbml.readMathMLFromString(mathml))
        return float(eval(formula.replace("^", "**"), {"__builtins__": {}}, names))  # noqa: S307

    def reset(self, model_id: str) -> None:
        """A fresh instance of *model_id*: its source, then its changes."""
        el = self.by_id[model_id]
        source = el.get("source")
        if source in self.by_id:
            self.reset(source)
        else:
            self.model = bngsim.Model.from_net(self.net)
        for change in el.iter():
            tag = _local(change.tag)
            if tag == "changeAttribute":
                self._set(self.model, change.get("target"), float(change.get("newValue")))
            elif tag == "computeChange":
                self._set(self.model, change.get("target"), self._value(change, self.model, {}))
        self.sims = {}

    def _model_of(self, task_id: str) -> str:
        el = self.by_id[task_id]
        if _local(el.tag) == "task":
            return el.get("modelReference")
        first = min(el.iter(f"{{{SEDML}}}subTask"), key=lambda s: int(s.get("order")))
        return self._model_of(first.get("task"))

    # ── simulations ────────────────────────────────────────────────────────
    def simulate(self, sim_id: str) -> bngsim.Result:
        utc = self.by_id[sim_id]
        algo = next(utc.iter(f"{{{SEDML}}}algorithm"))
        method = {"KISAO:0000019": "ode", "KISAO:0000029": "ssa"}[algo.get("kisaoID")]
        params = {
            p.get("kisaoID"): float(p.get("value"))
            for p in algo.iter(f"{{{SEDML}}}algorithmParameter")
        }
        kw = {}
        if method == "ode":
            kw = {
                "rtol": params["KISAO:0000209"],
                "atol": bngsim.ExactAtol(params["KISAO:0000211"]),
            }
        if method not in self.sims:
            self.sims[method] = bngsim.Simulator(self.model, method=method)
        sim = self.sims[method]
        t0, start = float(utc.get("initialTime")), float(utc.get("outputStartTime"))
        end, steps = float(utc.get("outputEndTime")), int(utc.get("numberOfPoints"))
        if start > t0:
            sim.run(t_span=(t0, start), n_points=2, **kw)
        return sim.run(t_span=(start, end), n_points=steps + 1, **kw)

    def run(self, task_id: str, *, top: bool = True) -> list[bngsim.Result]:
        el = self.by_id[task_id]
        if _local(el.tag) == "task":
            if top:
                self.reset(el.get("modelReference"))
            return [self.simulate(el.get("simulationReference"))]
        rng = self.by_id[el.get("range")]
        if _local(rng.tag) == "vectorRange":
            values = [float(v.text) for v in rng.iter(f"{{{SEDML}}}value")]
        else:
            n = int(rng.get("numberOfPoints"))
            lo, hi = float(rng.get("start")), float(rng.get("end"))
            space = np.geomspace if rng.get("type") == "log" else np.linspace
            values = list(space(lo, hi, n + 1))
        if top:
            self.reset(self._model_of(task_id))
        out: list[bngsim.Result] = []
        for value in values:
            if el.get("resetModel") == "true":
                self.reset(self._model_of(task_id))
            for sv in el.iter(f"{{{SEDML}}}setValue"):
                value_now = self._value(sv, self.model, {el.get("range"): value})
                self._set(self.model, sv.get("target"), value_now)
            for sub in sorted(el.iter(f"{{{SEDML}}}subTask"), key=lambda s: int(s.get("order"))):
                out += self.run(sub.get("task"), top=False)
        return out

    def experiment(self, i: int) -> list[bngsim.Result]:
        """Experiment *i*'s element run: its chain, its scan, or its task."""
        for tid in (f"chain_{i}", f"rtask_{i}", f"task_{i}"):
            if tid in self.by_id:
                return self.run(tid)
        raise AssertionError(f"no element for experiment {i}")


def _export(tmp_path, actions: str):
    net, bngl = tmp_path / "m.net", tmp_path / "m.bngl"
    net.write_text(NET)
    bngl.write_text(MODEL + actions + "\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net_to_omex(net, tmp_path / "m.omex", bngl=bngl, gate="none")
    with zipfile.ZipFile(tmp_path / "m.omex") as zf:
        sedml = next(zf.read(n).decode() for n in zf.namelist() if n.endswith(".sedml"))
        sbml = zf.read("model.xml").decode()
    return sedml, sbml, net, bngl, [str(w.message) for w in caught]


def _bad_targets(sedml: str, sbml: str) -> list[str]:
    """The targets that do not name exactly one SBML element: an XPath of
    ``/sbml:…/…[@id='x']`` steps (and an ``/@attribute``), under the ``sbml``
    prefix the document declares."""
    head = sedml.split("<sedML", 1)[1].split(">", 1)[0]
    declared = dict(re.findall(r'xmlns:(\w+)="([^"]+)"', head))
    root = ET.fromstring(sbml)
    bad = []
    for el in ET.fromstring(sedml).iter():
        target = el.get("target")
        if target is None:
            continue
        path, _, attr = target.partition("/@")
        steps = path.strip("/").split("/")
        if "sbml" not in declared or not all(s.startswith("sbml:") for s in steps):
            bad.append(target)
            continue
        hits = root.findall("./" + "/".join(steps[1:]), {"sbml": declared["sbml"]})
        if len(hits) != 1 or (attr and attr not in hits[0].attrib):
            bad.append(target)
    return bad


CONTINUED = f'simulate({{continue=>1,method=>"ode",t_end=>5,n_steps=>3,{TOLS}}})'
BIFURCATE = (
    'bifurcate({method=>"ode",t_end=>1,n_steps=>2,parameter=>"kf",par_min=>0.5,'
    f"par_max=>2,n_scan_pts=>3,{TOLS}}})"
)
CASES = {
    "equilibrate-then-stimulate": f'{RUN}\nsetConcentration("A()",50)\n{RUN}',
    "add-is-relative": f'{RUN}\naddConcentration("A()",25)\n{RUN}',
    "values-read-the-model": (
        f'{RUN}\nsetParameter("kf","2*k0")\nsetConcentration("A()","A0")\n{RUN}'
    ),
    "a-reset-to-a-saved-run": (
        f'{RUN}\nsaveConcentrations()\nsetConcentration("B()",80)\n{RUN}\n'
        f'resetConcentrations()\nsetParameter("kr",2)\n{RUN}'
    ),
    "a-parameter-reset": (
        f'saveParameters()\nsetParameter("kf",3)\n{RUN}\nresetParameters()\n{RUN}'
    ),
    "a-reset-between-runs": f'{RUN}\nresetConcentrations()\nsetParameter("kf",4)\n{RUN}',
    "a-scan-that-continues": f"{RUN}\n{SCAN.format(reset=0)}\n{RUN}",
    "a-resetting-scan-then-a-run": f"{SCAN.format(reset=1)}\n{RUN}",
    "a-bifurcate-then-a-run": f"{BIFURCATE}\n{RUN}",
    "a-continued-run": f"{RUN}\n{CONTINUED}",
    # From the initial state, as a derived model: computeChange for each.
    "an-add-and-an-expression-before-the-first-run": (
        f'addConcentration("A()",25)\nsetConcentration("B()","A0/4")\n{RUN}\n{RUN}'
    ),
}


@pytest.mark.parametrize("actions", CASES.values(), ids=CASES.keys())
def test_each_experiment_runs_as_bng2_runs_it(tmp_path, actions):
    sedml, sbml, net, bngl, said = _export(tmp_path, actions)
    assert not [m for m in said if "standard elements" in m], said
    assert _bad_targets(sedml, sbml) == []
    replay = bngsim.run_protocol(bngsim.Model.from_net(str(net)), parse_bngl_protocol(bngl))
    runner = _SpecRunner(sedml, sbml, net)
    for i, run in enumerate(replay):
        expected = (
            [run.result] if run.result is not None else [*run.scan_results, *run.backward_results]
        )
        got = runner.experiment(i)[-len(expected) :]
        assert len(got) == len(expected)
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_allclose(g.time, e.time, rtol=1e-12)
            np.testing.assert_allclose(g.species, e.species, rtol=1e-8, atol=1e-10)


def test_a_carried_run_is_a_chain_of_the_runs_before_it(tmp_path):
    sedml, *_ = _export(tmp_path, CASES["equilibrate-then-stimulate"])
    doc = ET.fromstring(sedml)
    tasks = {e.get("id"): e for e in doc.iter() if _local(e.tag) in ("task", "repeatedTask")}
    chain = tasks["chain_1"]
    assert chain.get("resetModel") == "true"
    orders = [s.get("order") for s in chain.iter(f"{{{SEDML}}}subTask")]
    assert orders == ["1", "2"]
    # A task for experiment 1 alone would describe a run from the initial state.
    assert "task_1" not in tasks


def test_a_carried_start_is_not_the_initial_one(tmp_path):
    # What the chain is for: stimulated from the initial state, the model does
    # not come to what BNG2.pl computes.
    _, _, net, bngl, _ = _export(tmp_path, CASES["equilibrate-then-stimulate"])
    replay = bngsim.run_protocol(bngsim.Model.from_net(str(net)), parse_bngl_protocol(bngl))
    model = bngsim.Model.from_net(str(net))
    model.set_concentration("A()", 50)
    fresh = bngsim.Simulator(model).run(t_span=(0, 2), n_points=5, rtol=1e-10)
    assert not np.allclose(fresh.species, replay[1].result.species, rtol=1e-3)


def test_a_resetting_scan_from_a_carried_state_is_said(tmp_path):
    *_, said = _export(tmp_path, f"{RUN}\n{SCAN.format(reset=1)}")
    assert any("reset_conc=>1" in m and "initial state" in m for m in said)


def test_the_standard_elements_alone_read_back_as_the_experiments(tmp_path):
    # The chains, and the steps that set values between runs, are not scans to
    # a reader of the standard elements, which reads an experiment per run.
    sedml, *_ = _export(tmp_path, CASES["a-scan-that-continues"])
    bare = re.sub(r"<annotation>.*?</annotation>", "", sedml, flags=re.S)
    exps = read_sedml_protocol(bare).experiments
    assert [e.kind for e in exps] == ["simulate", "scan", "simulate"]
    assert exps[1].scan_parameter == "kf"


def test_without_the_sbml_a_species_action_is_left_out_and_said():
    protocol = parse_bngl_protocol(MODEL + CASES["equilibrate-then-stimulate"])
    with pytest.warns(ConversionWarning, match="needs the SBML"):
        text = write_sedml_protocol(protocol)
    assert "listOfSpecies" not in text
    assert 'xmlns:sbml="http://www.sbml.org/sbml/level3/version2/core"' in text


def test_the_single_run_document_names_its_outputs_by_sbml_id(tmp_path):
    _, sbml, *_ = _export(tmp_path, RUN)
    spec = bngsim.EvaluationSpec(
        model_source="model.xml",
        model_format="sbml",
        outputs=("species:A()", "observable:Atot"),
    )
    text = write_sedml(spec, sbml=sbml)
    assert _bad_targets(text, sbml) == []
    assert "species[@id='A']" in text
    assert "parameter[@id='Atot']" in text
    with pytest.warns(ConversionWarning, match="have no target"):
        write_sedml(spec)
