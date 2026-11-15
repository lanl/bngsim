"""A SED-ML change to a species is evaluated once, where it applies (issue #1737).

SED-ML evaluates a ``computeChange``'s or ``setValue``'s math once, when it
applies: a species set by ``A = A0*2`` keeps that amount. ``read_sedml_protocol``
marked a parameter's such change ``evaluated_once`` (#1682) but not a species',
and ``run_protocol`` kept the species' expression live until the next run, as
BNG2.pl keeps a ``setConcentration`` expression; a scan of A0 that resets went
back to the live expression at each point, and the species followed the sweep.

Oracles: A <-> B at k1 = k2 = 1 has A(t) = a/2 + (a/2) e^(-2t) from A(0) = a,
B(0) = 0, so A(1) = a/2 (1 + e^-2). SED-ML L1V3's semantics give a.

The exception, kept: the per-point ``setValue`` of a repeatedTask whose math
reads the sweep's own range, which the reader reads as the species set to an
expression of the swept parameter at each point (#1733), as
``write_sedml_protocol`` writes BNG2.pl's following species.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim import ConversionWarning
from bngsim.convert import (
    net_to_sbml,
    read_sedml_protocol,
    write_bngl_protocol,
    write_sedml_protocol,
)
from bngsim.convert._protocol import ProtocolSpec, StateChange

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SED-ML replay requires libsbml")

NET = """begin parameters
    1 k1 1
    2 k2 1
    3 A0 100
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
SED = "http://sed-ml.org/sed-ml/level1/version3"
MML = "http://www.w3.org/1998/Math/MathML"
SBML_NS = "http://www.sbml.org/sbml/level3/version2/core"
P = "/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='{}']"
S = "/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='A']"
SIM = (
    '<listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0" '
    'outputEndTime="1" numberOfPoints="1"><algorithm kisaoID="KISAO:0000019"/>'
    "</uniformTimeCourse></listOfSimulations>"
)
A0_TIMES_2 = (
    f'<computeChange target="{S}/@initialConcentration"><listOfVariables>'
    f'<variable id="a0" target="{P.format("A0")}" modelReference="m"/></listOfVariables>'
    f'<math xmlns="{MML}"><apply><times/><ci>a0</ci><cn>2</cn></apply></math></computeChange>'
)


def a_at_1(a: float) -> float:
    return a / 2 * (1 + math.exp(-2))


@pytest.fixture
def files(tmp_path):
    net = tmp_path / "m.net"
    net.write_text(NET)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sbml = net_to_sbml(str(net), validate=None).sbml_text
    (tmp_path / "model.xml").write_text(sbml)
    assert ' id="A"' in sbml
    return net, tmp_path / "model.xml"


def _doc(changes: str, tasks: str) -> str:
    return (
        f'<sedML xmlns="{SED}" xmlns:sbml="{SBML_NS}" level="1" version="3">{SIM}'
        f'<listOfModels><model id="m" language="urn:sedml:language:sbml" source="model.xml">'
        f"<listOfChanges>{changes}</listOfChanges></model></listOfModels>"
        f'<listOfTasks><task id="t" modelReference="m" simulationReference="s"/>{tasks}'
        "</listOfTasks></sedML>"
    )


def _scan(extra_change: str = "") -> str:
    return (
        '<repeatedTask id="r" resetModel="true" range="x"><listOfRanges>'
        '<vectorRange id="x"><value>10</value><value>20</value><value>30</value></vectorRange>'
        "</listOfRanges><listOfChanges>"
        f'<setValue modelReference="m" range="x" target="{P.format("A0")}">'
        f'<math xmlns="{MML}"><ci>x</ci></math></setValue>{extra_change}'
        '</listOfChanges><listOfSubTasks><subTask order="1" task="t"/></listOfSubTasks>'
        "</repeatedTask>"
    )


def _read(doc: str, sbml: Path) -> ProtocolSpec:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        return read_sedml_protocol(doc, sbml=str(sbml))


def _replay(net: Path, spec: ProtocolSpec):
    return bngsim.run_protocol(bngsim.Model.from_net(str(net)), spec, rtol=1e-10, atol=1e-12)


def _scan_ends(runs) -> list[float]:
    return [float(r.species[-1, 0]) for r in runs[-1].scan_results]


# --- the issue's case -------------------------------------------------------


def test_the_issues_reset_scan_holds_the_species_where_the_change_put_it(files):
    net, sbml = files
    spec = _read(_doc(A0_TIMES_2, _scan()), sbml)
    assert spec.lossy == ()
    got = _scan_ends(_replay(net, spec))
    # Main replayed 11.35, 22.71, 34.06: A followed A0*2 through the sweep.
    assert got == pytest.approx([a_at_1(200)] * 3, rel=1e-6)


def test_the_change_is_read_as_evaluated_once(files):
    _, sbml = files
    spec = _read(_doc(A0_TIMES_2, ""), sbml)
    (change,) = [s for s in spec.steps if isinstance(s, StateChange)]
    assert change.kind == "set_concentration" and change.target == "A"
    assert change.value_expr is not None and change.evaluated_once
    # It survives the spec's own serialization.
    assert StateChange.from_dict(change.to_dict()) == change


def test_a_parameter_changed_after_it_does_not_move_it(files):
    # The model's changes in order: A = A0*2 (= 200), then A0 = 5.
    net, sbml = files
    later = f'<changeAttribute target="{P.format("A0")}/@value" newValue="5"/>'
    spec = _read(_doc(A0_TIMES_2 + later, ""), sbml)
    (run,) = _replay(net, spec)
    # Main gave a_at_1(10): A followed A0 = 5 into the run.
    assert float(run.result.species[-1, 0]) == pytest.approx(a_at_1(200), rel=1e-6)


def test_an_added_amount_is_evaluated_once_too(files):
    # A = A + A0: 100 + 100 at load, so 200 at each point of the reset scan.
    net, sbml = files
    add = (
        f'<computeChange target="{S}/@initialConcentration"><listOfVariables>'
        f'<variable id="a" target="{S}" modelReference="m"/>'
        f'<variable id="a0" target="{P.format("A0")}" modelReference="m"/></listOfVariables>'
        f'<math xmlns="{MML}"><apply><plus/><ci>a</ci><ci>a0</ci></apply></math></computeChange>'
    )
    spec = _read(_doc(add, _scan()), sbml)
    assert _scan_ends(_replay(net, spec)) == pytest.approx([a_at_1(200)] * 3, rel=1e-6)


# --- the exception: a per-point setValue from the sweep's range (#1733) ------


def test_a_per_point_set_value_of_the_range_still_follows_the_sweep(files):
    net, sbml = files
    per_point = (
        f'<setValue modelReference="m" range="x" target="{S}">'
        f'<math xmlns="{MML}"><apply><times/><ci>x</ci><cn>2</cn></apply></math></setValue>'
    )
    spec = _read(_doc("", _scan(per_point)), sbml)
    assert _scan_ends(_replay(net, spec)) == pytest.approx(
        [a_at_1(2 * v) for v in (10, 20, 30)], rel=1e-6
    )


# --- the writers ------------------------------------------------------------


def test_a_bngl_block_says_where_bng2pl_would_follow_the_scan(files):
    _, sbml = files
    spec = _read(_doc(A0_TIMES_2, _scan()), sbml)
    with pytest.warns(ConversionWarning, match=r"species follows A0.*issue #1737"):
        write_bngl_protocol(spec)


def _spec(*steps) -> ProtocolSpec:
    from bngsim.convert._protocol import Experiment

    run = Experiment(kind="simulate", method="ode", t_span=(0.0, 1.0), n_points=2)
    return ProtocolSpec(steps=tuple(run if s == "run" else s for s in steps))


ONCE = StateChange(kind="set_concentration", target="A", value_expr="A0*2", evaluated_once=True)


def test_a_bngl_block_warns_where_a_change_before_the_run_moves_it():
    moved = StateChange(kind="set_parameter", target="A0", value=5.0)
    with pytest.warns(ConversionWarning, match="species follows A0"):
        write_bngl_protocol(_spec(ONCE, moved, "run"))


def test_a_bngl_block_does_not_warn_where_the_change_comes_after_the_run():
    # BNG2.pl's setConcentration expression holds only until the run.
    moved = StateChange(kind="set_parameter", target="A0", value=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        write_bngl_protocol(_spec(ONCE, "run", moved, "run"))


def test_the_sedml_written_from_it_reads_back_the_same(files, tmp_path):
    net, sbml = files
    spec = _read(_doc(A0_TIMES_2, _scan()), sbml)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        written = write_sedml_protocol(spec, model_source="model.xml", sbml=str(sbml))
    again = _read(written, sbml)
    assert _scan_ends(_replay(net, again)) == pytest.approx([a_at_1(200)] * 3, rel=1e-6)
