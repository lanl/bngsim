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
import subprocess
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim import ConversionWarning
from bngsim._bngpath import resolve_bng
from bngsim.convert import (
    net_to_sbml,
    read_sedml_protocol,
    write_bngl,
    write_bngl_protocol,
    write_sedml_protocol,
)
from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SED-ML replay requires libsbml")
_BNG = resolve_bng()
needs_bng2 = pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")

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


def test_the_model_block_writer_warns_too(files):
    # write_bngl rewrites each species step for the model's patterns; it dropped
    # evaluated_once there, so the block it wrote had no warning.
    _, sbml = files
    spec = _read(_doc(A0_TIMES_2, _scan()), sbml)
    model = bngsim.Model.from_sbml(str(sbml))
    with pytest.warns(ConversionWarning, match=r"species follows A0.*issue #1737") as caught:
        text = write_bngl(model, protocol=spec, strict=False)
    assert 'setConcentration("@comp1:A()","A0 * 2")' in text
    # Reported at the caller, not at a frame inside bngsim.
    (warning,) = [w for w in caught if "#1737" in str(w.message)]
    assert Path(warning.filename).name == Path(__file__).name


def test_an_addition_after_it_adds_to_the_amount_it_left(files):
    # The model's changes in order: A = A0*2 (= 200), then A = A + A0 (= 300).
    # Main added to the declared 100, as BNG2.pl does to an expression amount.
    net, sbml = files
    add = (
        f'<computeChange target="{S}/@initialConcentration"><listOfVariables>'
        f'<variable id="a" target="{S}" modelReference="m"/>'
        f'<variable id="a0" target="{P.format("A0")}" modelReference="m"/></listOfVariables>'
        f'<math xmlns="{MML}"><apply><plus/><ci>a</ci><ci>a0</ci></apply></math></computeChange>'
    )
    spec = _read(_doc(A0_TIMES_2 + add, ""), sbml)
    (run,) = _replay(net, spec)
    assert float(run.result.species[0, 0]) == pytest.approx(300.0)
    with pytest.warns(ConversionWarning, match=r"adds to the species' declared amount") as caught:
        write_bngl_protocol(spec)
    assert not [w for w in caught if "species follows" in str(w.message)]


# --- a setValue in the sweep that reads the swept parameter ------------------
#
# SED-ML applies a repeatedTask's setValues at each iteration, and "variable
# references retrieve always the current value of the model variable ... at the
# current iteration" (L1V3, SetValue). So a species set from the swept
# parameter through a variable, listed after the setValue that sweeps it, is
# set from the point's value, and follows the sweep: A(0) = 2 A0 at each point.
# Listed before it, it reads the parameter as the reset left it, 100.


def _set_from_a0() -> str:
    return (
        f'<setValue modelReference="m" target="{S}"><listOfVariables>'
        f'<variable id="a0" target="{P.format("A0")}" modelReference="m"/></listOfVariables>'
        f'<math xmlns="{MML}"><apply><times/><ci>a0</ci><cn>2</cn></apply></math></setValue>'
    )


def test_a_setvalue_after_the_swept_one_reads_the_points_value(files):
    net, sbml = files
    spec = _read(_doc("", _scan(_set_from_a0())), sbml)
    assert spec.lossy == ()
    (change,) = [s for s in spec.steps if isinstance(s, StateChange)]
    assert change.kind == "set_concentration" and not change.evaluated_once
    assert _scan_ends(_replay(net, spec)) == pytest.approx(
        [a_at_1(2 * a0) for a0 in (10, 20, 30)], rel=1e-8
    )


def test_a_setvalue_before_the_swept_one_reads_the_reset_value(files):
    net, sbml = files
    before = _scan().replace("<listOfChanges>", "<listOfChanges>" + _set_from_a0())
    spec = _read(_doc("", before), sbml)
    assert spec.lossy == ()
    (change,) = [s for s in spec.steps if isinstance(s, StateChange)]
    assert change.kind == "set_concentration" and change.evaluated_once
    assert _scan_ends(_replay(net, spec)) == pytest.approx([a_at_1(200.0)] * 3, rel=1e-8)


def test_a_setvalue_after_the_swept_one_that_reads_another_parameter_is_evaluated_once(files):
    # k1 is not swept, so the value is the same at every point and one step
    # before the scan gives it: A(0) = 200 k1 = 200, evaluated once (#1737).
    net, sbml = files
    from_k1 = (
        f'<setValue modelReference="m" target="{S}"><listOfVariables>'
        f'<variable id="k" target="{P.format("k1")}" modelReference="m"/></listOfVariables>'
        f'<math xmlns="{MML}"><apply><times/><ci>k</ci><cn>200</cn></apply></math></setValue>'
    )
    spec = _read(_doc("", _scan(from_k1)), sbml)
    assert spec.lossy == ()
    (change,) = [s for s in spec.steps if isinstance(s, StateChange)]
    assert change.kind == "set_concentration" and change.evaluated_once
    assert _scan_ends(_replay(net, spec)) == pytest.approx([a_at_1(200.0)] * 3, rel=1e-8)


# --- where BNG2.pl keeps the expression, case by case -----------------------
#
# BNG2.pl keeps a setConcentration's expression as the amount until a run
# writes a number over it, or a setConcentration, an addConcentration or a
# resetConcentrations replaces it; saveConcentrations copies the expression,
# so a reset to that copy (a resetting scan's, at each point and at its end)
# brings it back. Each row: the actions, whether a run starts from the
# expression after what it reads changed, whether an addConcentration finds
# it, and what BNG2.pl 2.9.3 started its runs from (A, with A0 = 100 declared;
# a scan's entry is Atot at the end of each point). run_protocol holds 200.

BNGL_MODEL = """begin model
begin parameters
  k1 1
  k2 1
  A0 100
end parameters
begin molecule types
  A()
  B()
end molecule types
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

CASES = [
    # actions                                    follows  added   BNG2.pl
    ("once setp run", True, False),  # 10
    ("once run setp run", False, False),  # 200, then on from its end
    ("once scan:k1:1 setp run", True, False),  # scan from 200; the run from 10
    ("once scan:k1:0 setp run", False, False),  # the run goes on from the scan's end
    ("once scan:k1:1 run", False, False),  # 200
    ("once scan:A0:1 run", True, False),  # scan from 2, 4; the run from 4
    ("once scan:A0:0 run", True, False),  # scan from 2, then on
    ("once save run setp reset run", True, False),  # 200, 10
    ("once save:x run reset:x run", False, False),  # 200, 200
    ("once run setp reset run", False, False),  # 200, 100: nothing saved the expression
    ("once run save:x setp reset:x run", False, False),  # the save holds a number
    # The scan's own save, "SCAN", replaces one that held the expression.
    ("once save:SCAN run scan:k1:1 setp reset:SCAN run", False, False),
    ("once num setp run", False, False),  # 3: a number replaced it
    ("once reset setp run", False, False),  # 100: the declared amount did
    ("save:x once reset:x setp run", False, False),  # 100
    ("once save:x num setp reset:x run", True, False),  # 10
    ("once savep setp resetp run", False, False),  # 200: A0 is back at 100
    ("once setp savep setp resetp run", True, False),  # 10
    ("savep:p setp once resetp:p run", True, False),  # 200, where the change gave 10
    ("once:exp(1)*2 savep resetp run", False, False),  # 5.44: it reads no parameter
    ("savep once:exp(1)*2 resetp run", False, False),  # 5.44: exp is a function
    ("onceB num setp run", True, False),  # B from 10: the number was A's
    ("once add run", False, True),  # 101: added to the declared 100
    ("once setp add run", False, True),  # 101
    ("once num add run", False, False),  # 4
    ("once save:x run reset:x add run", False, True),  # 200, 101
]


def _steps(actions: str) -> ProtocolSpec:
    steps: list[object] = []
    for n, word in enumerate(actions.split()):
        verb, _, rest = word.partition(":")
        if verb in ("once", "onceB"):
            target = "B()" if verb == "onceB" else "A()"
            steps.append(
                StateChange(
                    kind="set_concentration",
                    target=target,
                    value_expr=rest or "A0*2",
                    evaluated_once=True,
                )
            )
        elif verb == "run":
            steps.append(
                Experiment(
                    kind="simulate", method="ode", t_span=(0.0, 1.0), n_points=2, label=f"r{n}"
                )
            )
        elif verb == "scan":
            name, _, reset = rest.partition(":")
            steps.append(
                Experiment(
                    kind="scan",
                    method="ode",
                    t_span=(0.0, 1.0),
                    n_points=2,
                    scan_parameter=name,
                    scan_min=1.0,
                    scan_max=2.0,
                    scan_points=2,
                    reset_between=reset == "1",
                    label=f"s{n}",
                )
            )
        elif verb == "setp":
            steps.append(StateChange(kind="set_parameter", target="A0", value=5.0))
        elif verb == "num":
            steps.append(StateChange(kind="set_concentration", target="A()", value=3.0))
        elif verb == "add":
            steps.append(StateChange(kind="add_concentration", target="A()", value=1.0))
        else:
            kind = {
                "save": "save_concentrations",
                "reset": "reset_concentrations",
                "savep": "save_parameters",
                "resetp": "reset_parameters",
            }[verb]
            steps.append(StateChange(kind=kind, label=rest or None))
    return ProtocolSpec(steps=tuple(steps))


def _warned(spec: ProtocolSpec) -> tuple[bool, bool, str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = write_bngl_protocol(spec)
    said = [str(w.message) for w in caught if "#1737" in str(w.message)]
    return (
        any("species follows" in m for m in said),
        any("declared amount" in m for m in said),
        text,
    )


@pytest.mark.parametrize(("actions", "follows", "added"), CASES, ids=[c[0] for c in CASES])
def test_a_bngl_block_warns_where_bng2pl_runs_from_the_expression(actions, follows, added):
    assert _warned(_steps(actions))[:2] == (follows, added)


def test_the_cases_cover_both_answers_of_both_warnings():
    assert {c[1] for c in CASES} == {True, False} and {c[2] for c in CASES} == {True, False}


@needs_bng2
@pytest.mark.parametrize("actions", [c[0] for c in CASES])
def test_the_block_warns_exactly_where_bng2pl_runs_it_differently(actions, tmp_path):
    """The table above against BNG2.pl itself: the block's runs differ from
    ``run_protocol``'s where a warning was given, and nowhere else."""
    spec = _steps(actions)
    follows, added, block = _warned(spec)
    block = block.replace("begin actions\n", "begin actions\ngenerate_network({overwrite=>1})\n")
    (tmp_path / "m.bngl").write_text(BNGL_MODEL + block, encoding="utf-8")
    (tmp_path / "m.net").write_text(NET, encoding="utf-8")
    done = subprocess.run(
        ["perl", str(Path(_BNG.bng2_pl).resolve()), "m.bngl"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]

    def rows(path: Path) -> list[list[float]]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [[float(x) for x in line.split()] for line in lines if not line.startswith("#")]

    ours: list[float] = []
    theirs: list[float] = []
    for run in bngsim.run_protocol(bngsim.Model.from_net(str(tmp_path / "m.net")), spec):
        if run.kind == "scan":
            # Atot at the end of each point.
            ours += [float(r.species[-1, 0]) for r in run.scan_results]
            theirs += [row[1] for row in rows(tmp_path / f"m_{run.label}.scan")]
        else:
            # Where the run starts: A, then B.
            ours += [float(x) for x in run.result.species[0, :2]]
            theirs += rows(tmp_path / f"m_{run.label}.cdat")[0][1:3]
    assert len(ours) == len(theirs) and ours
    differs = ours != pytest.approx(theirs, rel=1e-5, abs=1e-6)
    assert differs == (follows or added), (ours, theirs)
