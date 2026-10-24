"""The functions only a size jump's or a resize's assignments read are not
evaluated at every step (issue #1186).

#1145 writes a size jump read through a diamond of rules with each large
shared node of its texts as a function of its own, ``_szj_*``, and #1135 an
event's resize through one as ``_rsz_*``. Every function was evaluated at
every right-hand side and root evaluation, though the sides of a jump and a
resize's new size are read only by the assignments, when the event executes:
+66% per right-hand side at 22 rules, and 1.65-1.8x the run.

Those functions are now deferred: the right-hand side, interpreted and
compiled, and the roots leave them where they were, and they are evaluated
where an event executes. A helper a trigger reads is evaluated at
every root as before.

Oracles: closed forms, the rules evaluated in numpy. S = 10·e^(-kt), and Sn
keeps its amount, so [Sn] = 5·V(0)/V(t) across a jump, and 5·e^(-kn·t)/V(t)
across a resize.
"""

from __future__ import annotations

import re

import bngsim
import numpy as np
import pytest
from bngsim._codegen import generate_rhs_from_model

from .test_size_jump_diamond_of_rules import _ap, _away, _ci, _cn, _diamond, _pw, _sbml

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_T = "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>"
_JUMP = _pw(_ci("S"), _ap("gt", _ci("S"), _cn(3)), _ap("times", _cn(2), _ci("S")))


def _rules(n, bottom, cv):
    rules = [("r0", bottom), ("r1", _ap("plus", _ci("r0"), _cn(1)))]
    rules += [
        (f"r{i}", _ap("plus", _ap("sin", _ci(f"r{i - 1}")), _ap("cos", _ci(f"r{i - 2}"))))
        for i in range(2, n)
    ]
    return [*rules, ("cv", cv(_ci(f"r{n - 1}")))]


def _jump_doc(n, shape):
    """``bottom``: the size jumps where S falls through 3, every helper on its
    sides. ``both``: and a condition over the diamond, whose helpers the
    triggers read."""
    if shape == "bottom":
        cv = lambda r: _ap("plus", _cn(1), _ap("times", _cn(1e-6), r))  # noqa: E731
    else:
        cv = lambda r: _pw(_cn(1), _ap("gt", r, _cn(1.4)), _cn(2))  # noqa: E731
    return _sbml(_rules(n, _JUMP, cv))


def _jump_size(t, n, shape, k=0.5):
    s = 10 * np.exp(-k * t)
    r0 = np.where(s > 3, s, 2 * s)
    r = _diamond(r0, r0 + 1, n)
    return 1 + 1e-6 * r if shape == "bottom" else np.where(r > 1.4, 1.0, 2.0)


def _resize_doc(n):
    """``cv := 1 + 0.1*r_{n-1}`` over ``r0 := P``, and an event at t = 2 sets
    P from 1 to 3; Sn decays at kn."""
    xml = _sbml(
        _rules(n, _ci("P"), lambda r: _ap("plus", _cn(1), _ap("times", _cn(0.1), r))),
        [("kn", 0.3)],
    )
    xml = xml.replace(
        '<parameter id="r0" constant="false"/>',
        '<parameter id="r0" constant="false"/><parameter id="P" value="1" constant="false"/>',
    )
    degrade = (
        '<reaction id="degSn" reversible="false"><listOfReactants>'
        '<speciesReference species="Sn" stoichiometry="1" constant="true"/></listOfReactants>'
        f"<kineticLaw><math {_M}>{_ap('times', _ci('kn'), _ci('Sn'), _ci('cv'))}</math>"
        "</kineticLaw></reaction></listOfReactions>"
    )
    event = (
        '<listOfEvents><event id="e" useValuesFromTriggerTime="true"><trigger '
        f'initialValue="false" persistent="true"><math {_M}>{_ap("geq", _T, _cn(2))}</math>'
        f'</trigger><listOfEventAssignments><eventAssignment variable="P"><math {_M}>'
        f"{_cn(3)}</math></eventAssignment></listOfEventAssignments></event></listOfEvents>"
    )
    return xml.replace("</listOfReactions>", degrade).replace("</model>", event + "</model>")


def _helpers(model):
    return [
        f
        for f in model._core.codegen_data()["functions"]
        if f["name"].startswith(("_szj_", "_rsz_"))
    ]


def _run(model, times, **kw):
    result = bngsim.Simulator(model, method="ode", jacobian="fd", **kw).run(
        sample_times=list(times), rtol=1e-10, atol=1e-12
    )
    return result, list(result.species_names).index("Sn")


def test_a_jump_s_sides_are_deferred_and_its_trigger_s_helpers_are_not():
    model = bngsim.Model.from_sbml_string(_jump_doc(16, "both"))
    helpers = _helpers(model)
    deferred = {f["name"] for f in helpers if f["deferred"]}
    eager = {f["name"] for f in helpers if not f["deferred"]}
    assert deferred and eager
    # What the triggers read is evaluated at every root, and nothing that is
    # evaluated there reads a deferred function.
    read = set(re.findall(r"\w+", " ".join(model._core.event_trigger_sources())))
    assert read & eager and not read & deferred
    for f in model._core.codegen_data()["functions"]:
        if not f["deferred"]:
            assert not set(re.findall(r"\w+", f["expression"])) & deferred, f["name"]

    # A jump whose trigger reads none of them defers them all, and so does a
    # resize.
    assert all(
        f["deferred"] for f in _helpers(bngsim.Model.from_sbml_string(_jump_doc(22, "bottom")))
    )
    resize = _helpers(bngsim.Model.from_sbml_string(_resize_doc(20)))
    assert resize and all(f["deferred"] for f in resize)


def test_the_right_hand_side_leaves_the_deferred_functions_alone():
    model = bngsim.Model.from_sbml_string(_jump_doc(16, "both"))
    helpers = _helpers(model)
    deferred = [f["name"] for f in helpers if f["deferred"]]
    eager = [f["name"] for f in helpers if not f["deferred"]]
    y = np.asarray(model._core.get_initial_state(), float)
    model._core.compute_derivs(0.0, y)
    before = [model.get_param(f) for f in deferred + eager]
    y[list(model.species_names).index("S")] = 7.0
    model._core.compute_derivs(0.0, y)
    after = [model.get_param(f) for f in deferred + eager]
    assert after[: len(deferred)] == before[: len(deferred)]
    assert after[len(deferred) :] != before[len(deferred) :]

    source = generate_rhs_from_model(model._core)
    assert not [f for f in deferred if f"/* {f} */" in source]
    assert all(f"/* {f} */" in source for f in eager)


@pytest.mark.parametrize("codegen", [False, True])
@pytest.mark.parametrize(("shape", "n"), [("bottom", 22), ("both", 16)])
def test_a_jump_read_through_deferred_functions(shape, n, codegen):
    times = np.linspace(0.0, 4.0, 81)
    v = _jump_size(times, n, shape)
    assert len(set(np.round(v, 12))) > 1
    away = _away(np.round(v, 3)) if shape == "both" else np.ones_like(v, dtype=bool)
    result, i = _run(bngsim.Model.from_sbml_string(_jump_doc(n, shape)), times, codegen=codegen)
    sn = np.asarray(result.species)[:, i]
    np.testing.assert_allclose(sn[away], (5 * v[0] / v)[away], rtol=1e-8)


@pytest.mark.parametrize("codegen", [False, True])
def test_a_resize_read_through_deferred_functions(codegen):
    n = 20
    times = np.linspace(0.0, 4.0, 41)
    p = np.where(times >= 2, 3.0, 1.0)
    v = 1 + 0.1 * _diamond(p, p + 1, n)
    result, i = _run(bngsim.Model.from_sbml_string(_resize_doc(n)), times, codegen=codegen)
    sn = np.asarray(result.species)[:, i]
    np.testing.assert_allclose(sn, 5 * v[0] * np.exp(-0.3 * times) / v, rtol=1e-8)
