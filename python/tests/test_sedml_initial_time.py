"""Issue #1493 — a SED-ML time course integrates from initialTime.

SED-ML integrates a ``uniformTimeCourse`` from ``initialTime`` and reports only
``outputStartTime``..``outputEndTime``. bngsim's readers took
``outputStartTime`` as the start of the run, so a document with
``outputStartTime > initialTime`` restarted the model from its initial state at
the output start: A -> 0 at rate k*A, A(0) = 1, initialTime 0 and output 5..10
reported A(5) = 1 where the closed form is exp(-5). The same readers ignored
SED-ML L1V4's ``numberOfSteps``, so every V4 document got 101 points.

The oracle throughout is A(t) = exp(-k t) and dA/dk = -t exp(-k t), k = 1.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET

import bngsim
import numpy as np
import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import (
    read_sedml,
    read_sedml_protocol,
    write_sedml,
    write_sedml_protocol,
)
from bngsim.convert._protocol import (
    Experiment,
    ProtocolSpec,
    parse_bngl_protocol,
    write_bngl_protocol,
)

DECAY = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="decay">
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies><species id="A" compartment="c" initialAmount="1"
    hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/></listOfSpecies>
  <listOfParameters><parameter id="k" value="1" constant="true"/></listOfParameters>
  <listOfReactions><reaction id="r1" reversible="false">
   <listOfReactants>
    <speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
  </reaction></listOfReactions>
 </model>
</sbml>"""

TIGHT = (
    '<listOfAlgorithmParameters><algorithmParameter kisaoID="KISAO:0000209" value="1e-10"/>'
    '<algorithmParameter kisaoID="KISAO:0000211" value="1e-14"/></listOfAlgorithmParameters>'
)


def _doc(grid: str, *, version: int = 3, algorithm: str = TIGHT, extra: str = "") -> str:
    ns = f"http://sed-ml.org/sed-ml/level1/version{version}"
    return f"""<sedML xmlns="{ns}" level="1" version="{version}">
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="decay.xml"/></listOfModels>
 <listOfSimulations><uniformTimeCourse id="s" {grid}>
  <algorithm kisaoID="KISAO:0000019">{algorithm}</algorithm></uniformTimeCourse>
 </listOfSimulations>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/>{extra}</listOfTasks>
</sedML>"""


LATE = 'initialTime="0" outputStartTime="5" outputEndTime="10" numberOfPoints="5"'
OUTPUT_TIMES = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


@pytest.fixture
def sbml(tmp_path):
    path = tmp_path / "decay.xml"
    path.write_text(DECAY)
    return path


def _closed(times):
    return np.exp(-np.asarray(times))


def test_read_sedml_integrates_from_initial_time(sbml):
    spec = read_sedml(_doc(LATE), model_source=str(sbml), model_format="sbml", exact_atol=True)
    assert spec.t_span == (0.0, 10.0)
    assert spec.output_start == 5.0
    assert spec.n_points == 6
    result = spec.evaluate()
    np.testing.assert_allclose(result.time, OUTPUT_TIMES)
    # Restarting at t = 5 gave exp(-(t - 5)): A(5) = 1, 148 times exp(-5).
    np.testing.assert_allclose(result.species[:, 0], _closed(OUTPUT_TIMES), rtol=1e-7)


def test_run_protocol_integrates_from_initial_time(sbml):
    protocol = read_sedml_protocol(_doc(LATE))
    (exp,) = protocol.experiments
    assert (exp.t_span, exp.output_start, exp.n_points) == ((0.0, 10.0), 5.0, 6)
    assert exp.extra == {"rtol": 1e-10, "atol": 1e-14}
    # The L3 validation horizon of sbml_to_net(sedml=...) is the whole run.
    assert protocol.primary_experiment().t_span == (0.0, 10.0)
    (run,) = bngsim.run_protocol(bngsim.Model.from_sbml(str(sbml)), protocol)
    np.testing.assert_allclose(run.result.time, OUTPUT_TIMES)
    np.testing.assert_allclose(run.result.species[:, 0], _closed(OUTPUT_TIMES), rtol=1e-7)


def test_sensitivities_cross_the_unreported_part(sbml):
    spec = read_sedml(_doc(LATE), model_source=str(sbml), model_format="sbml", exact_atol=True)
    spec = bngsim.EvaluationSpec.from_dict({**spec.to_dict(), "sensitivity_params": ["k"]})
    result = spec.evaluate()
    t = np.asarray(OUTPUT_TIMES)
    np.testing.assert_allclose(result.sensitivities[:, 0, 0], -t * np.exp(-t), rtol=1e-6)


def test_ic_sensitivities_with_an_output_start_are_refused(sbml):
    spec = read_sedml(_doc(LATE), model_source=str(sbml), model_format="sbml")
    spec = bngsim.EvaluationSpec.from_dict({**spec.to_dict(), "sensitivity_ic": ["A"]})
    with pytest.raises(ValueError, match="IC sensitivities are not carried"):
        spec.evaluate()


@pytest.mark.parametrize(
    ("version", "grid"),
    [
        (3, 'initialTime="0" outputStartTime="0" outputEndTime="10" numberOfPoints="5"'),
        (4, 'initialTime="0" outputStartTime="0" outputEndTime="10" numberOfSteps="5"'),
    ],
)
def test_both_names_of_the_step_count_are_read(version, grid, sbml):
    doc = _doc(grid, version=version)
    spec = read_sedml(doc, model_source=str(sbml), model_format="sbml")
    assert spec.n_points == 6
    assert len(spec.evaluate().time) == 6
    assert read_sedml_protocol(doc).experiments[0].n_points == 6


def test_a_v4_uniform_range_reads_number_of_steps():
    scan = """<repeatedTask id="rt" range="r" resetModel="true"><listOfRanges>
      <uniformRange id="r" start="1" end="2" numberOfSteps="4" type="linear"/></listOfRanges>
      <listOfChanges><setValue range="r" modelReference="m"
        target="/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='k']/@value"/>
      </listOfChanges>
      <listOfSubTasks><subTask order="1" task="t"/></listOfSubTasks></repeatedTask>"""
    grid = 'initialTime="0" outputStartTime="0" outputEndTime="10" numberOfSteps="5"'
    (exp,) = read_sedml_protocol(_doc(grid, version=4, extra=scan)).experiments
    assert exp.is_scan
    assert exp.scan_points == 5


@pytest.mark.parametrize(
    "grid",
    [
        'initialTime="5" outputStartTime="0" outputEndTime="10" numberOfPoints="5"',
        'initialTime="0" outputStartTime="12" outputEndTime="10" numberOfPoints="5"',
    ],
)
def test_an_output_window_outside_the_run_is_refused(grid):
    with pytest.raises(ConversionError, match="initialTime <= outputStartTime <= outputEndTime"):
        read_sedml(_doc(grid))
    with pytest.raises(ConversionError, match="initialTime <= outputStartTime <= outputEndTime"):
        read_sedml_protocol(_doc(grid))


def _utc(xml: str) -> dict[str, str]:
    for el in ET.fromstring(xml).iter():
        if el.tag.rsplit("}", 1)[-1] == "uniformTimeCourse":
            return dict(el.attrib)
    raise AssertionError("no uniformTimeCourse")


def test_write_sedml_round_trips_the_output_start(sbml):
    spec = bngsim.EvaluationSpec(
        model_source=str(sbml), model_format="sbml", t_span=(0, 10), output_start=5, n_points=6
    )
    text = write_sedml(spec)
    attrs = _utc(text)
    assert (attrs["initialTime"], attrs["outputStartTime"], attrs["outputEndTime"]) == (
        "0",
        "5",
        "10",
    )
    back = read_sedml(text, model_source=str(sbml), model_format="sbml")
    assert (back.t_span, back.output_start, back.n_points) == ((0.0, 10.0), 5.0, 6)
    assert bngsim.EvaluationSpec.from_json(spec.to_json()) == spec


def test_a_spec_without_an_output_start_serializes_as_before():
    spec = bngsim.EvaluationSpec(model_source="m.net", t_span=(0, 10))
    assert "output_start" not in spec.to_dict()
    # An output start at the run's start is the plain spec.
    same = bngsim.EvaluationSpec(model_source="m.net", t_span=(0, 10), output_start=0.0)
    assert same == spec and same.output_start is None


@pytest.mark.parametrize("bad", [11.0, -1.0, float("nan")])
def test_an_output_start_outside_t_span_is_refused(bad):
    with pytest.raises(ValueError, match="must lie in t_span"):
        bngsim.EvaluationSpec(model_source="m.net", t_span=(0, 10), output_start=bad)
    with pytest.raises(ValueError, match="must lie in t_span"):
        Experiment(kind="simulate", method="ode", t_span=(0, 10), n_points=6, output_start=bad)


def test_an_experiment_takes_output_start_or_sample_times():
    with pytest.raises(ValueError, match="not both"):
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(0, 10),
            n_points=3,
            output_start=5.0,
            sample_times=(0.0, 5.0, 10.0),
        )


def _late_protocol() -> ProtocolSpec:
    exp = Experiment(
        kind="simulate",
        method="ode",
        t_span=(0.0, 10.0),
        n_points=6,
        output_start=5.0,
        extra={"rtol": 1e-10, "atol": 1e-14},
    )
    return ProtocolSpec(steps=(exp,))


def test_the_protocol_writer_emits_both_times_and_the_annotation_keeps_them():
    protocol = _late_protocol()
    text = write_sedml_protocol(protocol)
    attrs = _utc(text)
    assert (attrs["initialTime"], attrs["outputStartTime"]) == ("0", "5")
    assert read_sedml_protocol(text) == protocol
    assert ProtocolSpec.from_json(protocol.to_json()) == protocol


def test_a_bngl_actions_block_writes_it_as_sample_times_from_the_start(sbml):
    with pytest.warns(ConversionWarning, match="BNG2.pl has no output start"):
        actions = write_bngl_protocol(_late_protocol())
    assert "sample_times=>[0,5,6,7,8,9,10]" in actions
    bngl = sbml.with_name("late.bngl")
    bngl.write_text(actions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        reparsed = parse_bngl_protocol(bngl, strict=False)
    (exp,) = reparsed.experiments
    # BNG2.pl's run: one integration from 0, writing t = 0 and then the window.
    assert exp.t_span == (0.0, 10.0)
    (run,) = bngsim.run_protocol(bngsim.Model.from_sbml(str(sbml)), reparsed)
    np.testing.assert_allclose(run.result.time, [0.0, *OUTPUT_TIMES])
    np.testing.assert_allclose(run.result.species[1:, 0], _closed(OUTPUT_TIMES), rtol=1e-7)


def test_a_continuation_with_an_output_start_is_refused(sbml):
    exp = Experiment(
        kind="simulate",
        method="ode",
        t_span=(0.0, 10.0),
        n_points=6,
        output_start=5.0,
        extra={"continue": 1},
    )
    with pytest.raises(ConversionError, match="continue=>1 starts it at the model time"):
        bngsim.run_protocol(bngsim.Model.from_sbml(str(sbml)), ProtocolSpec(steps=(exp,)))
