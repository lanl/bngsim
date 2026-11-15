"""A setParameter of a number ends the network's expression, whatever the number
(issue #1738).

BNG2.pl's ``setParameter("k2",0.5)`` makes k2 the constant 0.5, even where the
network declares ``k2 k1*0.5`` and that is what it gives. ``run_protocol``
wrote the number with ``set_param``, which overrides a derived parameter only
while the value differs from its expression's (#188): the same number left the
expression live, and a later ``setParameter("k1",3)`` moved k2 to 1.5. The
same happened after a scan or a bifurcate of k2 that ended on that value.

Oracles: A <-> B at k1, k2 from A = 100, B = 0 has
A(t) = 100 (k2 + k1 e^(-(k1+k2) t)) / (k1 + k2), and BNG2.pl 2.9.3 on the
issue's two models where it is installed.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

NET = """begin parameters
    1 k1 1
    2 k2 {k2}
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
TOL = {"rtol": 1e-10, "atol": 1e-12}


def a_at(t: float, k1: float, k2: float) -> float:
    return 100 * (k2 + k1 * math.exp(-(k1 + k2) * t)) / (k1 + k2)


def _model(tmp_path: Path, k2: str) -> bngsim.Model:
    net = tmp_path / "m.net"
    net.write_text(NET.format(k2=k2))
    return bngsim.Model.from_net(str(net))


def _set(name: str, value: float | str) -> StateChange:
    if isinstance(value, str):
        return StateChange(kind="set_parameter", target=name, value_expr=value)
    return StateChange(kind="set_parameter", target=name, value=float(value))


def _run(t_end: float = 1.0) -> Experiment:
    return Experiment(kind="simulate", method="ode", t_span=(0.0, t_end), n_points=2)


def _scan(parameter: str, values, *, kind="parameter_scan", reset=True) -> Experiment:
    return Experiment(
        kind=kind,
        method="ode",
        t_span=(0.0, 1.0),
        n_points=2,
        scan_parameter=parameter,
        scan_values=tuple(float(v) for v in values),
        reset_between=reset,
    )


def _replay(model, *steps):
    return bngsim.run_protocol(model, ProtocolSpec(steps=tuple(steps)), **TOL)


def _end(result) -> float:
    return float(result.species[-1, 0])


# --- the issue's two cases --------------------------------------------------


def test_a_number_equal_to_the_expression_still_ends_it(tmp_path):
    # k2 k1*0.5 = 0.5 at k1 = 1; setParameter("k2",0.5), then k1 = 3.
    model = _model(tmp_path, "k1*0.5")
    (run,) = _replay(model, _set("k2", 0.5), _set("k1", 3), _run(0.25))
    # BNG2.pl 2.9.3: 50.01674456212. Main: 54.9768, k2 = 1.5.
    assert _end(run.result) == pytest.approx(a_at(0.25, 3, 0.5), rel=1e-8)
    assert _end(run.result) == pytest.approx(50.01674456212, rel=1e-8)


def test_a_scan_after_a_number_equal_to_the_expression(tmp_path):
    # Bill's form: k2 k1*3; setParameter("k2","k1*2"), a run, then
    # setParameter("k2",3) (= k1*3 at k1 = 1), and a scan of k1.
    model = _model(tmp_path, "k1*3")
    runs = _replay(model, _set("k2", "k1*2"), _run(), _set("k2", 3), _scan("k1", [1, 3, 5]))
    got = [_end(r) for r in runs[-1].scan_results]
    # Each point resets to where the first run left A (and B), with k2 = 3.
    a0, b0 = (float(x) for x in runs[0].result.species[-1, :2])
    want = []
    for k1 in (1, 3, 5):
        eq = (a0 + b0) * 3 / (k1 + 3)
        want.append(eq + (a0 - eq) * math.exp(-(k1 + 3)))
    assert got == pytest.approx(want, rel=1e-8)
    # BNG2.pl 2.9.3 (the issue's comment): 74.8778, 50.0454, 37.5103.
    # Main: 74.8778, 75.0, 75.0.
    assert got == pytest.approx([74.8778, 50.0454, 37.5103], abs=1e-4)


def test_a_scan_of_the_parameter_ending_on_its_expressions_value(tmp_path):
    # A scan of k2 whose last point is k1*0.5 at k1 = 1: k2 keeps that number.
    model = _model(tmp_path, "k1*0.5")
    runs = _replay(model, _scan("k2", [0.9, 0.5]), _set("k1", 3), _run(0.25))
    assert _end(runs[-1].result) == pytest.approx(a_at(0.25, 3, 0.5), rel=1e-8)


def test_a_bifurcate_that_ends_where_it_started(tmp_path):
    # The issue's other form: a bifurcate of k2 runs back to its first value.
    model = _model(tmp_path, "k1*0.5")
    # A range, which BNG2.pl's bifurcate runs up and then back down from its
    # max; explicit values it runs in the same order both ways.
    bif = Experiment(
        kind="bifurcate",
        method="ode",
        t_span=(0.0, 1.0),
        n_points=2,
        scan_parameter="k2",
        scan_min=0.5,
        scan_max=1.0,
        scan_points=3,
        reset_between=False,
    )
    runs = _replay(model, bif, _set("k1", 3), _run(0.25))
    assert runs[0].backward_values[-1] == 0.5
    # From where the bifurcate left the species, k2 = 0.5 against 1.5.
    start = runs[0].backward_results[-1].species[-1]
    k1, k2, t = 3.0, 0.5, 0.25
    total = float(start[0] + start[1])
    eq = total * k2 / (k1 + k2)
    want = eq + (float(start[0]) - eq) * math.exp(-(k1 + k2) * t)
    assert _end(runs[-1].result) == pytest.approx(want, rel=1e-8)


# --- saves and resets --------------------------------------------------------


def test_a_save_after_the_number_keeps_it_a_number(tmp_path):
    # BNG2.pl's saveParameters copies k2 as the constant it now is.
    model = _model(tmp_path, "k1*0.5")
    reset = StateChange(kind="reset_parameters")
    save = StateChange(kind="save_parameters")
    runs = _replay(model, _set("k2", 0.5), save, _set("k1", 2), reset, _set("k1", 3), _run(0.25))
    assert _end(runs[-1].result) == pytest.approx(a_at(0.25, 3, 0.5), rel=1e-8)


def test_a_reset_to_before_the_number_brings_the_expression_back(tmp_path):
    model = _model(tmp_path, "k1*0.5")
    save = StateChange(kind="save_parameters")
    reset = StateChange(kind="reset_parameters")
    runs = _replay(model, save, _set("k2", 0.5), reset, _set("k1", 3), _run(0.25))
    # k2 follows k1*0.5 again: 1.5.
    assert _end(runs[-1].result) == pytest.approx(a_at(0.25, 3, 1.5), rel=1e-8)


# --- what does not change -----------------------------------------------------


def test_a_derived_parameter_no_action_sets_still_follows(tmp_path):
    model = _model(tmp_path, "k1*0.5")
    (run,) = _replay(model, _set("k1", 3), _run(0.25))
    assert _end(run.result) == pytest.approx(a_at(0.25, 3, 1.5), rel=1e-8)


def test_a_number_other_than_the_expressions_ends_it_as_before(tmp_path):
    model = _model(tmp_path, "k1*0.5")
    (run,) = _replay(model, _set("k2", 0.7), _set("k1", 3), _run(0.25))
    assert _end(run.result) == pytest.approx(a_at(0.25, 3, 0.7), rel=1e-8)


def test_a_primary_parameter_set_to_its_own_value(tmp_path):
    # Nothing to end: k1 has no expression, and k2 still follows it.
    model = _model(tmp_path, "k1*0.5")
    (run,) = _replay(model, _set("k1", 1), _set("k1", 3), _run(0.25))
    assert _end(run.result) == pytest.approx(a_at(0.25, 3, 1.5), rel=1e-8)


def test_a_setparameter_to_an_expression_after_the_number(tmp_path):
    # The number ends the network's expression; an expression after it is the
    # action's own (issue #1682), and follows k1.
    model = _model(tmp_path, "k1*0.5")
    (run,) = _replay(model, _set("k2", 0.5), _set("k2", "k1*2"), _set("k1", 3), _run(0.25))
    assert _end(run.result) == pytest.approx(a_at(0.25, 3, 6.0), rel=1e-8)


# --- against BNG2.pl, where it is installed ----------------------------------

BNGPATH = os.environ.get("BNGPATH", "")
HAS_BNG2 = bool(BNGPATH) and (Path(BNGPATH) / "BNG2.pl").is_file() and shutil.which("perl")

BNGL = """begin model
begin parameters
  k1 1
  k2 {k2}
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
generate_network({{overwrite=>1}})
{actions}
"""
CASES = {
    "the issue": (
        "k1*0.5",
        'setParameter("k2",0.5)\nsetParameter("k1",3)\n'
        'simulate({method=>"ode",t_end=>0.25,n_steps=>1,atol=>1e-12,rtol=>1e-10})',
        "m.gdat",
    ),
    "the comment's scan": (
        "k1*3",
        'setParameter("k2","k1*2")\n'
        'simulate({method=>"ode",suffix=>"w",t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n'
        'setParameter("k2",3)\n'
        'parameter_scan({method=>"ode",suffix=>"x",parameter=>"k1",par_scan_vals=>[1,3,5],'
        "t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-10})",
        "m_x.scan",
    ),
}


@pytest.mark.skipif(not HAS_BNG2, reason="needs BNG2.pl on BNGPATH")
@pytest.mark.parametrize("case", list(CASES))
def test_run_bngl_agrees_with_bng2pl(tmp_path, case):
    k2, actions, out = CASES[case]
    bngl = tmp_path / "m.bngl"
    bngl.write_text(BNGL.format(k2=k2, actions=actions))
    subprocess.run(
        ["perl", str(Path(BNGPATH) / "BNG2.pl"), "m.bngl"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=300,
    )
    table = np.atleast_2d(np.loadtxt(tmp_path / out))
    want = table[-1:, 1] if out.endswith(".gdat") else table[:, 1]
    model = bngsim.Model.from_bngl(bngl)
    runs = bngsim.run_protocol(model, bngsim.convert.parse_bngl_protocol(bngl), **TOL)
    last = runs[-1]
    results = last.scan_results if last.scan_results else (last.result,)
    names = list(results[0].species_names)
    i = next(k for k, n in enumerate(names) if n.startswith("A"))
    got = [float(r.species[-1, i]) for r in results]
    assert got == pytest.approx(want.tolist(), rel=1e-7)
