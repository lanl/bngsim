"""Issues #1662 and #1675: a SED-ML document is read by its tasks.

``read_sedml_protocol`` built one experiment per ``uniformTimeCourse`` and ran
them one after another from where the last left the model, reading no
``task``, no ``model`` and no model change. On BioModels' own SED-ML for
BIOMD0000000010 the second task, on a model with 14 ``changeAttribute``\\ s,
gave MAPK_PP(12000) = 79.8 for libRoadRunner's 92.7, with no warning. It now
reads each task as a run of its model, from that model's initial state, with
the model's changes as the steps before it; tasks are independent, a chain
(``repeatedTask``) runs its subtasks on one state, and a time course no task
runs is not run. ``read_sedml`` likewise reads its first task's model,
simulation and outputs (#1675).

The oracle is the closed form of A <-> B with k1 = k2 = 1 from A = 100, B = 0:
A(t) = (A0 + B0)/2 + (A0 - (A0 + B0)/2) e^(-(k1 + k2) t) for equal rates, and
A(t) = A0 e^(-k1 t) with k2 = 0; for the real document, a plain bngsim run of
the changed model, which agrees with libRoadRunner to 1.6e-6 (the issue).
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import (
    parse_bngl_protocol,
    read_sedml,
    read_sedml_protocol,
    write_bngl_protocol,
    write_sedml_protocol,
)

SBML_NS = "http://www.sbml.org/sbml/level3/version2/core"
MATHML = "http://www.w3.org/1998/Math/MathML"


def _species(sid, value):
    return (
        f'<species id="{sid}" compartment="C" initialConcentration="{value}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _reaction(rid, src, dst, rate, local=None):
    lp = (
        f'<listOfLocalParameters><localParameter id="{local[0]}" value="{local[1]}"/>'
        "</listOfLocalParameters>"
        if local
        else ""
    )
    return (
        f'<reaction id="{rid}" reversible="false">'
        f'<listOfReactants><speciesReference species="{src}" stoichiometry="1" constant="true"/>'
        f'</listOfReactants><listOfProducts><speciesReference species="{dst}" stoichiometry="1" '
        f'constant="true"/></listOfProducts><kineticLaw><math xmlns="{MATHML}"><apply><times/>'
        f"<ci>C</ci><ci>{rate}</ci><ci>{src}</ci></apply></math>{lp}</kineticLaw></reaction>"
    )


# A <-> B, k1 = k2 = 1, A(0) = 100, B(0) = 0, in a compartment of size 1.
SBML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    f'<sbml xmlns="{SBML_NS}" level="3" version="2"><model id="m">'
    '<listOfCompartments><compartment id="C" spatialDimensions="3" size="1" constant="true"/>'
    "</listOfCompartments><listOfSpecies>" + _species("A", 100) + _species("B", 0) + ""
    '</listOfSpecies><listOfParameters><parameter id="k1" value="1" constant="true"/>'
    '<parameter id="k2" value="1" constant="true"/></listOfParameters><listOfReactions>'
    + _reaction("J1", "A", "B", "k1")
    + _reaction("J2", "B", "A", "k2")
    + "</listOfReactions></model></sbml>"
)
# The same with the forward rate a local parameter k of J1 (k = 1), and k2 = 0.
SBML_LOCAL = SBML.replace(
    _reaction("J1", "A", "B", "k1"), _reaction("J1", "A", "B", "k", ("k", 1))
).replace('id="k2" value="1"', 'id="k2" value="0"')

XP = "/sbml:sbml/sbml:model/sbml:"
PARAM = XP + "listOfParameters/sbml:parameter[@id='{}']/@value"
SPECIES = XP + "listOfSpecies/sbml:species[@id='{}']/@initialConcentration"
LOCAL = (
    XP + "listOfReactions/sbml:reaction[@id='{}']/sbml:kineticLaw/sbml:listOfLocalParameters/"
    "sbml:localParameter[@id='{}']/@value"
)
COMPARTMENT = XP + "listOfCompartments/sbml:compartment[@id='C']/@size"


def _change(target, value):
    return f'<changeAttribute target="{target}" newValue="{value}"/>'


def _compute(target, math_xml, variables=()):
    lov = "".join(
        f'<variable id="{vid}" target="{tgt}" modelReference="m"/>' for vid, tgt in variables
    )
    lov = f"<listOfVariables>{lov}</listOfVariables>" if lov else ""
    return (
        f'<computeChange target="{target}">{lov}<math xmlns="{MATHML}">{math_xml}</math>'
        "</computeChange>"
    )


def _doc(sims, models, tasks, generators=None):
    """A SED-ML L1V3 document. ``sims``: (id, t_end); ``models``: (id, source,
    changes xml); ``tasks``: (id, model, sim); ``generators``: (task, species),
    default one of A per task."""
    if generators is None:
        generators = [(tid, "A") for tid, _, _ in tasks]
    sim_xml = "".join(
        f'<uniformTimeCourse id="{sid}" initialTime="0" outputStartTime="0" outputEndTime="{end}" '
        f'numberOfPoints="10"><algorithm kisaoID="KISAO:0000019"/></uniformTimeCourse>'
        for sid, end in sims
    )
    model_xml = "".join(
        f'<model id="{mid}" language="urn:sedml:language:sbml" source="{src}">'
        + (f"<listOfChanges>{changes}</listOfChanges>" if changes else "")
        + "</model>"
        for mid, src, changes in models
    )
    task_xml = "".join(
        f'<task id="{tid}" modelReference="{mid}" simulationReference="{sid}"/>'
        for tid, mid, sid in tasks
    )
    gen_xml = "".join(
        f'<dataGenerator id="g{i}"><listOfVariables><variable id="v{i}" taskReference="{tid}" '
        f"target=\"{XP}listOfSpecies/sbml:species[@id='{sp}']\"/></listOfVariables>"
        f'<math xmlns="{MATHML}"><ci>v{i}</ci></math></dataGenerator>'
        for i, (tid, sp) in enumerate(generators)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" xmlns:sbml="{SBML_NS}" '
        'level="1" version="3">'
        f"<listOfSimulations>{sim_xml}</listOfSimulations><listOfModels>{model_xml}</listOfModels>"
        f"<listOfTasks>{task_xml}</listOfTasks><listOfDataGenerators>{gen_xml}</listOfDataGenerators>"
        "</sedML>"
    )


def _a_at_1(doc, sbml=SBML, **kw):
    """A(1) of each run of *doc* on *sbml*, at rtol 1e-10 / atol 1e-12."""
    spec = read_sedml_protocol(doc)
    runs = bngsim.run_protocol(
        bngsim.Model.from_sbml_string(sbml), spec, rtol=1e-10, atol=1e-12, **kw
    )
    return [float(r.result.species[-1, r.result.species_names.index("A")]) for r in runs]


# Closed forms. The runs integrate at rtol 1e-10; 1e-6 leaves four decades for
# the solver's own error, and the wrong answers of the issue are 20 to 55% off.
ON = 50 + 50 * math.exp(-2)  # k1 = k2 = 1 from A = 100
OFF = 100 * math.exp(-1)  # k2 = 0
REL = 1e-6
BASE = [("s", 1.0)], [("m", "model.xml", "")]


def test_a_models_change_attribute_is_applied():
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", _change(PARAM.format("k2"), 0))], [("t", "m", "s")]
    )
    assert _a_at_1(doc) == pytest.approx([OFF], rel=REL)


def test_independent_tasks_each_start_from_the_initial_state():
    """Chained, the second task returned A(1) = 50 + 50 e^-4 = 50.92."""
    doc = _doc(
        [("s0", 1.0), ("s1", 1.0)],
        [("m0", "model.xml", ""), ("m1", "model.xml", "")],
        [("t0", "m0", "s0"), ("t1", "m1", "s1")],
    )
    assert _a_at_1(doc) == pytest.approx([ON, ON], rel=REL)


def test_a_changed_model_does_not_leak_into_the_next_task():
    doc = _doc(
        [("s0", 1.0), ("s1", 1.0)],
        [("m0", "model.xml", _change(PARAM.format("k2"), 0)), ("m1", "model.xml", "")],
        [("t0", "m0", "s0"), ("t1", "m1", "s1")],
    )
    assert _a_at_1(doc) == pytest.approx([OFF, ON], rel=REL)


def test_two_tasks_on_one_time_course_are_two_runs():
    """Read by time course, the task on the k2 = 0 model was never run."""
    doc = _doc(
        [("s", 1.0)],
        [("m0", "model.xml", ""), ("m1", "model.xml", _change(PARAM.format("k2"), 0))],
        [("t0", "m0", "s"), ("t1", "m1", "s")],
    )
    assert _a_at_1(doc) == pytest.approx([ON, OFF], rel=REL)


def test_a_time_course_no_task_runs_is_not_run():
    doc = _doc([("s", 1.0), ("unused", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")])
    assert _a_at_1(doc) == pytest.approx([ON], rel=REL)


def test_a_derived_model_inherits_its_sources_changes():
    doc = _doc(
        [("s", 1.0)],
        [
            ("base", "model.xml", _change(PARAM.format("k2"), 0)),
            ("derived", "#base", _change(SPECIES.format("A"), 10)),
        ],
        [("t", "derived", "s")],
    )
    spec = read_sedml_protocol(doc)
    assert [(s.kind, s.target, s.value) for s in spec.steps[:2]] == [
        ("set_parameter", "k2", 0.0),
        ("set_concentration", "A", 10.0),
    ]
    assert _a_at_1(doc) == pytest.approx([10 * math.exp(-1)], rel=REL)


def test_a_species_change_attribute_sets_its_start():
    doc = _doc(*BASE, [("t", "m", "s")]).replace(
        '<model id="m" language="urn:sedml:language:sbml" source="model.xml"></model>',
        f'<model id="m" language="urn:sedml:language:sbml" source="model.xml"><listOfChanges>'
        f"{_change(SPECIES.format('A'), 10)}</listOfChanges></model>",
    )
    assert _a_at_1(doc) == pytest.approx([5 + 5 * math.exp(-2)], rel=REL)


def test_a_compute_change_of_the_species_plus_a_term_is_an_addition():
    """What write_sedml_protocol writes for addConcentration: A <- A + 50."""
    cc = _compute(
        SPECIES.format("A"),
        "<apply><plus/><ci>v</ci><cn>50</cn></apply>",
        [("v", XP + "listOfSpecies/sbml:species[@id='A']")],
    )
    doc = _doc([("s", 1.0)], [("m", "model.xml", cc)], [("t", "m", "s")])
    (step, _) = read_sedml_protocol(doc).steps
    assert (step.kind, step.target, step.value) == ("add_concentration", "A", 50.0)
    assert _a_at_1(doc) == pytest.approx([75 + 75 * math.exp(-2)], rel=REL)


def test_a_compute_change_with_an_expression_is_evaluated_in_the_model():
    """k2 <- 2 k1, so A(inf) = 200/3 and A(1) = 200/3 + (100 - 200/3) e^-3."""
    cc = _compute(
        PARAM.format("k2"),
        "<apply><times/><cn>2</cn><ci>v</ci></apply>",
        [("v", XP + "listOfParameters/sbml:parameter[@id='k1']")],
    )
    doc = _doc([("s", 1.0)], [("m", "model.xml", cc)], [("t", "m", "s")])
    (step, _) = read_sedml_protocol(doc).steps
    assert (step.kind, step.target, step.value, step.value_expr) == (
        "set_parameter",
        "k2",
        None,
        "2 * k1",
    )
    assert _a_at_1(doc) == pytest.approx([200 / 3 + (100 - 200 / 3) * math.exp(-3)], rel=REL)


def test_a_kinetic_laws_local_parameter_is_named_as_the_loader_names_it():
    """BioModels' documents change kinetic-law locals; bngsim loads J1's k as
    _lp_J1_k. Set to 3 with k2 = 0, A(1) = 100 e^-3."""
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", _change(LOCAL.format("J1", "k"), 3))], [("t", "m", "s")]
    )
    (step, _) = read_sedml_protocol(doc).steps
    assert (step.kind, step.target, step.value) == ("set_parameter", "_lp_J1_k", 3.0)
    assert _a_at_1(doc, sbml=SBML_LOCAL) == pytest.approx([100 * math.exp(-3)], rel=REL)


def test_a_change_the_replay_cannot_apply_is_said_and_refused():
    doc = _doc([("s", 1.0)], [("m", "model.xml", _change(COMPARTMENT, 2))], [("t", "m", "s")])
    with pytest.warns(ConversionWarning, match="compartment.*no species or parameter value"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1 and "compartment" in spec.lossy[0]
    model = bngsim.Model.from_sbml_string(SBML)
    with pytest.raises(ConversionError, match="allow_lossy"):
        bngsim.run_protocol(model, spec)
    with pytest.warns(ConversionWarning, match="as nearly as it holds it"):
        runs = bngsim.run_protocol(model, spec, allow_lossy=True, rtol=1e-10, atol=1e-12)
    assert float(runs[0].result.species[-1, 0]) == pytest.approx(ON, rel=REL)


def test_the_actions_block_of_a_foreign_document_resets_between_tasks():
    doc = _doc(
        [("s0", 1.0), ("s1", 1.0)],
        [("m0", "model.xml", _change(PARAM.format("k2"), 0)), ("m1", "model.xml", "")],
        [("t0", "m0", "s0"), ("t1", "m1", "s1")],
    )
    text = write_bngl_protocol(read_sedml_protocol(doc))
    calls = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#") and "actions" not in line
    ]
    assert calls == [
        'saveParameters("sedml_initial")',
        'setParameter("k2",0)',
        'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>10})',
        "resetConcentrations()",
        'resetParameters("sedml_initial")',
        'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>10})',
    ]


# ── The real document ───────────────────────────────────────────────────────

BIOMD10 = (
    Path(__file__).resolve().parents[2] / "parity_checks/rr_parity/smoke/models/BIOMD0000000010"
)


@pytest.mark.skipif(
    not BIOMD10.is_dir(), reason="rr_parity corpus model BIOMD0000000010 not present"
)
def test_biomodels_own_document_for_biomd0000000010():
    """Kholodenko's cascade: a base model and one with 14 changeAttributes on
    kinetic-law locals, a task on each. The second task gave 79.8 for 92.7."""
    spec = read_sedml_protocol(BIOMD10 / "BIOMD0000000010_url.sedml")
    assert len(spec.experiments) == 2 and spec.lossy == ()
    sets = [s for s in spec.steps if getattr(s, "kind", None) == "set_parameter"]
    assert len(sets) == 14 and all(s.target.startswith("_lp_J") for s in sets)
    model = bngsim.Model.from_sbml(str(BIOMD10 / "BIOMD0000000010_url.xml"))
    runs = bngsim.run_protocol(model, spec, rtol=1e-10, atol=1e-12)
    i = runs[1].result.species_names.index("MAPK_PP")
    # libRoadRunner 2.11.0 on the changed SBML gives 92.73 at the issue's two
    # decimals; a plain bngsim run of it agrees with libRoadRunner to 1.6e-6.
    assert float(runs[1].result.species[-1, i]) == pytest.approx(92.73, abs=0.01)
    changed = bngsim.Model.from_sbml(str(BIOMD10 / "BIOMD0000000010_url.xml"))
    changed.set_params({s.target: s.value for s in sets})
    plain = bngsim.Simulator(changed, method="ode").run(
        t_span=(0, 12000), n_points=1001, rtol=1e-10, atol=1e-12
    )
    # The same engine on the same changed model, step for step.
    assert np.array_equal(np.asarray(plain.species), np.asarray(runs[1].result.species))


# ── bngsim's own documents, without their annotation ────────────────────────

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
RUN = 'simulate({method=>"ode",t_end=>2,n_steps=>4})'
SCAN = (
    'parameter_scan({method=>"ode",t_end=>2,n_steps=>4,parameter=>"kf",par_min=>1,par_max=>2,'
    "n_scan_pts=>3,reset_conc=>0})"
)
CHAINS = {
    "equilibrate-then-stimulate": f'{RUN}\nsetConcentration("A()",50)\n{RUN}',
    "add-is-relative": f'{RUN}\naddConcentration("A()",25)\n{RUN}',
    "a-reset-between-runs": f'{RUN}\nresetConcentrations()\nsetParameter("kf",4)\n{RUN}',
    "a-scan-that-continues": f"{RUN}\n{SCAN}\n{RUN}",
    "a-parameter-reset": (
        f'saveParameters()\nsetParameter("kf",3)\n{RUN}\nresetParameters()\n{RUN}'
    ),
}


def _runs(protocol, net):
    out = []
    for run in bngsim.run_protocol(bngsim.Model.from_net(str(net)), protocol):
        if run.result is not None:
            out.append(np.asarray(run.result.species))
        else:
            out.extend(np.asarray(r.species) for r in (*run.scan_results, *run.backward_results))
    return out


@pytest.mark.parametrize("actions", CHAINS.values(), ids=CHAINS.keys())
def test_a_chain_bngsim_wrote_reads_back_as_the_runs_bngl_made(tmp_path, actions):
    """write_sedml_protocol writes a run that continues an earlier one as a
    repeatedTask that resets once and replays the runs before it, beside the
    first run's own task. Read by task with that prefix folded in, the bare
    document runs as the annotated one does, to the bit."""
    net = tmp_path / "m.net"
    net.write_text(NET)
    protocol = parse_bngl_protocol(f"begin model\nend model\ngenerate_network()\n{actions}\n")
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        xml = write_sedml_protocol(
            protocol, model_source="model.xml", sbml=sbml, model=bngsim.Model.from_net(str(net))
        )
    bare = re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S)
    assert "<protocol" not in bare
    spec = read_sedml_protocol(bare, sbml=sbml)
    assert spec.lossy == ()
    exact, back = _runs(read_sedml_protocol(xml), net), _runs(spec, net)
    assert (
        len(exact)
        == len(back)
        == len(protocol.experiments) + (2 if "parameter_scan" in actions else 0)
    )
    for a, b in zip(exact, back, strict=True):
        assert np.array_equal(a, b)


def test_a_chain_that_repeats_a_lead_runs_it_as_the_document_says(tmp_path):
    """A run that continues a saved state is written as a chain of the runs
    before it; a reader of the standard elements runs that chain as written,
    so the lead is run once more, from the same state, with the same numbers."""
    net = tmp_path / "m.net"
    net.write_text(NET)
    actions = (
        f'{RUN}\nsaveConcentrations()\nsetConcentration("B()",80)\n{RUN}\n'
        f'resetConcentrations()\nsetParameter("kr",2)\n{RUN}'
    )
    protocol = parse_bngl_protocol(f"begin model\nend model\ngenerate_network()\n{actions}\n")
    sbml = bngsim.convert.net_to_sbml(net, out_path=None, validate=None).output_text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        xml = write_sedml_protocol(
            protocol, model_source="model.xml", sbml=sbml, model=bngsim.Model.from_net(str(net))
        )
    exact = _runs(read_sedml_protocol(xml), net)
    back = _runs(
        read_sedml_protocol(
            re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S), sbml=sbml
        ),
        net,
    )
    assert len(exact) == 3 and len(back) == 4
    for a, b in zip(exact, (back[0], back[1], back[3]), strict=True):
        assert np.array_equal(a, b)
    assert np.array_equal(back[2], back[0])


# ── read_sedml (issue #1675) ────────────────────────────────────────────────


def test_read_sedml_takes_the_first_tasks_model_simulation_and_outputs():
    doc = _doc(
        [("s0", 1.0), ("s1", 2.0)],
        [("slow", "slow.xml", ""), ("fast", "fast.xml", "")],
        [("t0", "fast", "s1"), ("t1", "slow", "s0")],
        generators=[("t0", "A"), ("t1", "A"), ("t1", "B")],
    )
    spec = read_sedml(doc)
    assert spec.model_source == "fast.xml"
    assert spec.t_span == (0.0, 2.0)
    assert spec.outputs == ("species:A",)


def test_read_sedml_carries_a_parameter_change_and_says_what_it_cannot():
    doc = _doc(
        [("s", 1.0)],
        [
            ("base", "model.xml", _change(PARAM.format("k2"), 0)),
            ("m", "#base", _change(SPECIES.format("A"), 10) + _change(LOCAL.format("J1", "k"), 3)),
        ],
        [("t", "m", "s")],
    )
    with pytest.warns(ConversionWarning, match="not a parameter value"):
        spec = read_sedml(doc)
    assert spec.model_source == "model.xml"
    assert spec.params == {"k2": 0.0, "_lp_J1_k": 3.0}


# ── Opus's probes, folded in (P1 P2 P3 P5 P6, #1675's unread tasks) ───────────


def _rt(rid, subtask, *, reset="true", ranges="", changes=""):
    lor = f"<listOfRanges>{ranges}</listOfRanges>" if ranges else ""
    loc = f"<listOfChanges>{changes}</listOfChanges>" if changes else ""
    rng = ' range="r"' if ranges else ""
    return (
        f'<repeatedTask id="{rid}" resetModel="{reset}"{rng}>{lor}{loc}'
        f'<listOfSubTasks><subTask order="1" task="{subtask}"/></listOfSubTasks></repeatedTask>'
    )


def _set_value(target, math_xml, rng=None, variables=()):
    lov = "".join(
        f'<variable id="{vid}" target="{tgt}" modelReference="m"/>' for vid, tgt in variables
    )
    lov = f"<listOfVariables>{lov}</listOfVariables>" if lov else ""
    attr = f' range="{rng}"' if rng else ""
    return (
        f'<setValue target="{target}" modelReference="m"{attr}>{lov}'
        f'<math xmlns="{MATHML}">{math_xml}</math></setValue>'
    )


def test_a_compute_changes_own_parameter_is_substituted():
    """The dose named in the change's listOfParameters, not a model parameter:
    with no such model parameter the run raised, and with one of 7 it added 7."""
    cc = (
        f'<computeChange target="{SPECIES.format("A")}"><listOfVariables>'
        f'<variable id="v" target="{XP}listOfSpecies/sbml:species[@id=\'A\']" modelReference="m"/>'
        '</listOfVariables><listOfParameters><parameter id="dose" value="50"/></listOfParameters>'
        f'<math xmlns="{MATHML}"><apply><plus/><ci>v</ci><ci>dose</ci></apply></math>'
        "</computeChange>"
    )
    doc = _doc([("s", 1.0)], [("m", "model.xml", cc)], [("t", "m", "s")])
    (step, _) = read_sedml_protocol(doc).steps
    assert (step.kind, step.target, step.value) == ("add_concentration", "A", 50.0)
    assert _a_at_1(doc) == pytest.approx([75 + 75 * math.exp(-2)], rel=REL)


def test_variables_are_substituted_in_one_pass():
    """k2 := k1 and k1 := k2 at once: 3*k2 is 3*k1, not 3*k2 again."""
    cc = _compute(
        PARAM.format("k2"),
        "<apply><times/><cn>3</cn><ci>k2</ci></apply>",
        [
            ("k2", XP + "listOfParameters/sbml:parameter[@id='k1']"),
            ("k1", XP + "listOfParameters/sbml:parameter[@id='k2']"),
        ],
    )
    doc = _doc([("s", 1.0)], [("m", "model.xml", cc)], [("t", "m", "s")])
    (step, _) = read_sedml_protocol(doc).steps
    assert step.value_expr == "3 * k1"


def test_an_initial_amount_is_read_in_its_compartment_with_the_sbml():
    """In C = 2, initialAmount 60 is a concentration of 30: A(1) = 15 + 15 e^-2.
    Without the SBML the size is unknown, and the change is lossy."""
    sbml = SBML.replace('size="1"', 'size="2"')
    change = _change(XP + "listOfSpecies/sbml:species[@id='A']/@initialAmount", 60)
    doc = _doc([("s", 1.0)], [("m", "model.xml", change)], [("t", "m", "s")])
    spec = read_sedml_protocol(doc, sbml=sbml)
    assert (spec.steps[0].kind, spec.steps[0].value) == ("set_concentration", 30.0)
    runs = bngsim.run_protocol(bngsim.Model.from_sbml_string(sbml), spec, rtol=1e-10, atol=1e-12)
    assert float(runs[0].result.species[-1, 0]) == pytest.approx(15 + 15 * math.exp(-2), rel=REL)
    with pytest.warns(ConversionWarning, match="initialAmount needs its compartment"):
        bare = read_sedml_protocol(doc)
    assert len(bare.lossy) == 1 and [s.kind for s in bare.steps] == ["simulate"]


def test_a_constant_set_value_beside_a_sweep_is_a_step_before_it():
    """k1 swept over [1, 2] with k2 set to 0 beside it: each point decays at
    its k1 alone, A(1) = 100 e^-k1. The old reader dropped the k2."""
    ranges = '<vectorRange id="r"><value>1</value><value>2</value></vectorRange>'
    changes = _set_value(PARAM.format("k1"), "<ci>r</ci>", rng="r") + _set_value(
        PARAM.format("k2"), "<cn>0</cn>"
    )
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")], generators=[("sweep", "A")]
    ).replace(
        "</listOfTasks>", _rt("sweep", "t", ranges=ranges, changes=changes) + "</listOfTasks>"
    )
    spec = read_sedml_protocol(doc)
    assert [getattr(s, "kind", None) for s in spec.steps] == ["set_parameter", "scan"]
    assert spec.lossy == ()
    (run,) = bngsim.run_protocol(bngsim.Model.from_sbml_string(SBML), spec, rtol=1e-10, atol=1e-12)
    ends = [float(r.species[-1, 0]) for r in run.scan_results]
    assert ends == pytest.approx([100 * math.exp(-1), 100 * math.exp(-2)], rel=REL)


def test_a_set_value_that_reads_the_range_beside_the_sweep_is_lossy():
    ranges = '<vectorRange id="r"><value>1</value><value>2</value></vectorRange>'
    changes = _set_value(PARAM.format("k1"), "<ci>r</ci>", rng="r") + _set_value(
        PARAM.format("k2"), "<apply><times/><cn>2</cn><ci>r</ci></apply>"
    )
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")], generators=[("sweep", "A")]
    ).replace(
        "</listOfTasks>", _rt("sweep", "t", ranges=ranges, changes=changes) + "</listOfTasks>"
    )
    with pytest.warns(ConversionWarning, match="from its range beside the swept"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1 and [getattr(s, "kind", None) for s in spec.steps] == ["scan"]


def test_a_task_on_a_second_model_file_is_lossy():
    doc = _doc(
        [("s0", 1.0), ("s1", 1.0)],
        [("m0", "model.xml", ""), ("m1", "other.xml", "")],
        [("t0", "m0", "s0"), ("t1", "m1", "s1")],
    )
    with pytest.warns(ConversionWarning, match="from 'other.xml', not the document's"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1 and len(spec.experiments) == 1


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (
            f'<addXML target="{XP}listOfSpecies"><newXML>{_species("Q", 1)}</newXML></addXML>',
            "addXML",
        ),
        (_change(COMPARTMENT, 2), "compartment"),
    ],
    ids=["add-xml", "compartment"],
)
def test_a_change_the_protocol_cannot_make_is_lossy(change, said):
    doc = _doc([("s", 1.0)], [("m", "model.xml", change)], [("t", "m", "s")])
    with pytest.warns(ConversionWarning, match=said):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1 and [getattr(s, "kind", None) for s in spec.steps] == ["simulate"]


def test_a_task_on_another_kind_of_simulation_is_lossy():
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s"), ("t1", "m", "one")]
    ).replace(
        "</listOfSimulations>",
        '<oneStep id="one" step="1"><algorithm kisaoID="KISAO:0000019"/></oneStep>'
        "</listOfSimulations>",
    )
    with pytest.warns(ConversionWarning, match="a oneStep, not a uniformTimeCourse"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1 and len(spec.experiments) == 1


def test_a_single_task_document_reads_as_it_did():
    """One task, no change: the one simulate, with no save or reset around it."""
    doc = _doc([("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")])
    spec = read_sedml_protocol(doc)
    assert [getattr(s, "kind", None) for s in spec.steps] == ["simulate"]
    assert _a_at_1(doc) == pytest.approx([ON], rel=REL)


def test_read_sedml_says_which_tasks_it_leaves_unread():
    doc = _doc(
        [("s0", 1.0), ("s1", 1.0)],
        [("m", "model.xml", "")],
        [("t0", "m", "s0"), ("t1", "m", "s1")],
    )
    with pytest.warns(ConversionWarning, match="'t1' are not read.*read_sedml_protocol"):
        spec = read_sedml(doc)
    assert spec.t_span == (0.0, 1.0)


def test_a_negative_parameter_of_a_change_keeps_its_sign_under_a_power():
    """d = -5 in A + d^2 is A + 25: substituted bare, -5^2 read as -25, and A
    started at 75 for 125. A0 = 125 gives A(1) = 62.5 + 62.5 e^-2."""
    cc = (
        f'<computeChange target="{SPECIES.format("A")}"><listOfVariables>'
        f'<variable id="v" target="{XP}listOfSpecies/sbml:species[@id=\'A\']" modelReference="m"/>'
        '</listOfVariables><listOfParameters><parameter id="d" value="-5"/></listOfParameters>'
        f'<math xmlns="{MATHML}"><apply><plus/><ci>v</ci><apply><power/><ci>d</ci><cn>2</cn>'
        "</apply></apply></math></computeChange>"
    )
    doc = _doc([("s", 1.0)], [("m", "model.xml", cc)], [("t", "m", "s")])
    (step, _) = read_sedml_protocol(doc).steps
    assert step.kind == "add_concentration" and step.value_expr == "(-5.0)^2"
    assert _a_at_1(doc) == pytest.approx([62.5 + 62.5 * math.exp(-2)], rel=REL)


def test_a_task_on_a_model_the_document_does_not_define_is_lossy():
    """It ran on the model given, with no change, as if it were the base model."""
    doc = _doc(
        [("s", 1.0), ("s1", 1.0)],
        [("m", "model.xml", "")],
        [("t", "m", "s"), ("t1", "nope", "s1")],
    )
    with pytest.warns(ConversionWarning, match="model 'nope', which the document does not define"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1 and len(spec.experiments) == 1


# ── Round two: the combined patch's own review (A, B, D, E, F, G, H6, H7) ─────


def test_a_sweep_that_sets_a_function_of_its_range_is_lossy():
    """k1 = 10^r over r = (1, 2) scanned k1 = 1, 2: 56.77 / 36.65 for 9.09 / 0.99."""
    ranges = '<vectorRange id="r"><value>1</value><value>2</value></vectorRange>'
    changes = _set_value(
        PARAM.format("k1"), "<apply><power/><cn>10</cn><ci>r</ci></apply>", rng="r"
    )
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")], generators=[("sweep", "A")]
    ).replace(
        "</listOfTasks>", _rt("sweep", "t", ranges=ranges, changes=changes) + "</listOfTasks>"
    )
    with pytest.warns(ConversionWarning, match="a function of its range"):
        spec = read_sedml_protocol(doc)
    assert len(spec.lossy) == 1 and spec.experiments == []


def test_a_swept_task_a_generator_reads_runs_alone_too():
    """t has its own output and is swept: its own run, then the scan. Without
    the output it runs inside the sweep only."""
    ranges = '<vectorRange id="r"><value>1</value><value>2</value></vectorRange>'
    changes = _set_value(PARAM.format("k1"), "<ci>r</ci>", rng="r")
    base = _doc([("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")])
    swept = base.replace(
        "</listOfTasks>", _rt("sweep", "t", ranges=ranges, changes=changes) + "</listOfTasks>"
    )
    kinds = [getattr(s, "kind", None) for s in read_sedml_protocol(swept).steps]
    assert kinds == [
        "save_parameters",
        "simulate",
        "reset_concentrations",
        "reset_parameters",
        "scan",
    ]
    unread = swept.replace(' taskReference="t"', ' taskReference="sweep"')
    assert [getattr(s, "kind", None) for s in read_sedml_protocol(unread).steps] == ["scan"]


def test_a_species_own_value_is_its_amount_where_it_has_only_substance_units():
    """A step's setValue on a species sets the species' value: its amount for a
    species with only substance units, its concentration otherwise (so the
    writer writes an amount-valued species' setConcentration times its size).
    Read with the SBML, an amount 100 in a compartment of size 2 is the
    concentration 50 the model holds; without the SBML the change is lossy.
    bngsim's own .net exports fold a compartment's volume
    into the rates and write size-1 SBML, so this needs a hand-written one."""
    sbml = SBML.replace('size="1"', 'size="2"').replace(
        _species("A", 100),
        '<species id="A" compartment="C" initialAmount="200" hasOnlySubstanceUnits="true" '
        'boundaryCondition="false" constant="false"/>',
    )
    assert bngsim.Model.from_sbml_string(sbml).get_state()[0] == pytest.approx(100.0)
    changes = _set_value(XP + "listOfSpecies/sbml:species[@id='A']", "<cn>100</cn>")
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")], generators=[("step", "A")]
    ).replace(
        "</listOfTasks>",
        _rt(
            "step",
            "t",
            ranges='<vectorRange id="r"><value>0</value></vectorRange>',
            changes=changes,
        )
        + "</listOfTasks>",
    )
    spec = read_sedml_protocol(doc, sbml=sbml)
    assert spec.lossy == ()
    assert (spec.steps[0].kind, spec.steps[0].value) == ("set_concentration", 50.0)
    runs = bngsim.run_protocol(bngsim.Model.from_sbml_string(sbml), spec, rtol=1e-10, atol=1e-12)
    # The run starts at the concentration the set left, 100 over the size 2.
    # (A enters its rate law as an amount here, so no concentration closed
    # form applies to the run itself.)
    assert float(runs[0].result.species[0, 0]) == pytest.approx(50.0, rel=1e-12)
    with pytest.warns(ConversionWarning, match="needs its compartment's size"):
        assert len(read_sedml_protocol(doc).lossy) == 1


def test_combined_protocols_start_each_from_the_parameters_the_first_found():
    """File 1 sets k2 = 0 and runs; file 2 runs plain: its A(1) is the k2 = 1
    value, 56.77, not file 1's 36.79."""
    from bngsim.convert import combine_protocols

    first = read_sedml_protocol(
        _doc([("s", 1.0)], [("m", "model.xml", _change(PARAM.format("k2"), 0))], [("t", "m", "s")])
    )
    second = read_sedml_protocol(_doc(*BASE, [("t", "m", "s")]))
    runs = bngsim.run_protocol(
        bngsim.Model.from_sbml_string(SBML),
        combine_protocols([first, second]),
        rtol=1e-10,
        atol=1e-12,
    )
    ends = [float(r.result.species[-1, 0]) for r in runs]
    assert ends == pytest.approx([OFF, ON], rel=REL)


def test_read_sedml_with_no_task_reads_the_first_time_course_quietly():
    doc = _doc([("s", 1.0)], [("m", "model.xml", "")], []).replace(
        "<listOfTasks></listOfTasks>", ""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        spec = read_sedml(doc)
    assert spec.t_span == (0.0, 1.0)


def test_read_sedml_refuses_a_first_task_on_another_kind_of_simulation():
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t1", "m", "one"), ("t", "m", "s")]
    ).replace(
        "</listOfSimulations>",
        '<oneStep id="one" step="1"><algorithm kisaoID="KISAO:0000019"/></oneStep>'
        "</listOfSimulations>",
    )
    with pytest.raises(ConversionError, match="a oneStep, not a uniformTimeCourse"):
        read_sedml(doc)


def test_targets_with_blanks_and_double_quotes_are_read():
    ranges = '<vectorRange id="r"><value>1</value><value>2</value></vectorRange>'
    target = XP + "listOfParameters/sbml:parameter[ @id = &quot;k1&quot; ]"
    doc = _doc(
        [("s", 1.0)],
        [
            (
                "m",
                "model.xml",
                _change(XP + "listOfParameters/sbml:parameter[ @id = 'k2' ]/@value", 0),
            )
        ],
        [("t", "m", "s")],
        generators=[("sweep", "A")],
    ).replace(
        "</listOfTasks>",
        _rt("sweep", "t", ranges=ranges, changes=_set_value(target, "<ci>r</ci>", rng="r"))
        + "</listOfTasks>",
    )
    spec = read_sedml_protocol(doc)
    assert spec.lossy == ()
    assert [getattr(s, "kind", None) for s in spec.steps] == ["set_parameter", "scan"]
    assert spec.steps[1].scan_parameter == "k1"


def test_a_one_value_range_a_step_reads_is_its_value():
    """A one-iteration repeatedTask whose setValue on a species reads its
    range: A starts at the range's 10 (A(1) = 5 + 5 e^-2); a range of two values
    without a sweep is lossy."""
    one = '<vectorRange id="r"><value>10</value></vectorRange>'
    changes = _set_value(XP + "listOfSpecies/sbml:species[@id='A']", "<ci>r</ci>")
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")], generators=[("step", "A")]
    ).replace("</listOfTasks>", _rt("step", "t", ranges=one, changes=changes) + "</listOfTasks>")
    spec = read_sedml_protocol(doc, sbml=SBML)
    assert [(getattr(s, "kind", None), getattr(s, "value", None)) for s in spec.steps] == [
        ("set_concentration", 10.0),
        ("simulate", None),
    ]
    runs = bngsim.run_protocol(bngsim.Model.from_sbml_string(SBML), spec, rtol=1e-10, atol=1e-12)
    assert float(runs[0].result.species[-1, 0]) == pytest.approx(5 + 5 * math.exp(-2), rel=REL)
    two = '<vectorRange id="r"><value>10</value><value>20</value></vectorRange>'
    with pytest.warns(ConversionWarning, match="several values"):
        spec2 = read_sedml_protocol(doc.replace(one, two), sbml=SBML)
    assert any("several values" in note for note in spec2.lossy)


def test_a_compartment_sized_by_a_rule_makes_an_amount_change_lossy():
    sbml = SBML.replace(
        "</listOfCompartments>",
        "</listOfCompartments><listOfInitialAssignments><initialAssignment symbol='C'>"
        f'<math xmlns="{MATHML}"><cn>2</cn></math></initialAssignment></listOfInitialAssignments>',
    )
    change = _change(XP + "listOfSpecies/sbml:species[@id='A']/@initialAmount", 60)
    doc = _doc([("s", 1.0)], [("m", "model.xml", change)], [("t", "m", "s")])
    with pytest.warns(ConversionWarning, match="not a number in the SBML"):
        spec = read_sedml_protocol(doc, sbml=sbml)
    assert len(spec.lossy) == 1


def test_a_rule_sized_compartment_makes_an_amount_valued_species_own_value_lossy():
    """A species with only substance units holds its amount, so its setValue
    is an amount over its compartment's size; a size the SBML computes (here
    an initialAssignment on C) is no number for the reader or the writer, and
    the change is said lossy rather than scaled by 1 (issue #1662)."""
    sbml = SBML.replace(
        _species("A", 100),
        '<species id="A" compartment="C" initialAmount="100" hasOnlySubstanceUnits="true" '
        'boundaryCondition="false" constant="false"/>',
    ).replace(
        "</listOfCompartments>",
        "</listOfCompartments><listOfInitialAssignments><initialAssignment symbol='C'>"
        f'<math xmlns="{MATHML}"><cn>2</cn></math></initialAssignment>'
        "</listOfInitialAssignments>",
    )
    changes = _set_value(XP + "listOfSpecies/sbml:species[@id='A']", "<cn>100</cn>")
    doc = _doc(
        [("s", 1.0)], [("m", "model.xml", "")], [("t", "m", "s")], generators=[("step", "A")]
    ).replace(
        "</listOfTasks>",
        _rt(
            "step",
            "t",
            ranges='<vectorRange id="r"><value>0</value></vectorRange>',
            changes=changes,
        )
        + "</listOfTasks>",
    )
    with pytest.warns(ConversionWarning, match="not a number in the SBML"):
        spec = read_sedml_protocol(doc, sbml=sbml)
    assert len(spec.lossy) == 1
    assert [getattr(s, "kind", None) for s in spec.steps] == ["simulate"]
    protocol = parse_bngl_protocol(
        'begin model\nend model\ngenerate_network()\nsetConcentration("A()",50)\n'
        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
    )
    with pytest.warns(ConversionWarning, match="not a number in the SBML"):
        xml = write_sedml_protocol(protocol, model_source="model.xml", sbml=sbml)
    assert "newValue" not in xml and "<changeXML" not in xml
