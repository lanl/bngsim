"""Issue #1082 — a condition reached through a function definition is spelled
as the same condition written directly, so it is registered once.

A formal bound to a call such as ``time()`` used to be spliced into the body
as ``(time())``, while the same comparison at top level read ``time()``. The
condition scans de-duplicate and compare by text, so each edge of the window
was registered twice: harmless extra roots, and duplicate entries in
``Model.time_discontinuity_conditions()``.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

libsbml = pytest.importorskip("libsbml")


def _model(tmp_path, law: str, trigger: str, functions=()):
    doc = libsbml.SBMLDocument(3, 1)
    m = doc.createModel()
    c = m.createCompartment()
    c.setId("C")
    c.setSize(1)
    c.setConstant(True)
    for sid in ("Y", "Z"):
        s = m.createSpecies()
        s.setId(sid)
        s.setCompartment("C")
        s.setInitialConcentration(0)
        s.setHasOnlySubstanceUnits(False)
        s.setBoundaryCondition(False)
        s.setConstant(False)
    for fid, body in functions:
        f = m.createFunctionDefinition()
        f.setId(fid)
        f.setMath(libsbml.parseL3Formula(body))
    r = m.createReaction()
    r.setId("R")
    r.setReversible(False)
    p = r.createProduct()
    p.setSpecies("Z")
    p.setStoichiometry(1)
    p.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    e = m.createEvent()
    e.setId("E")
    e.setUseValuesFromTriggerTime(True)
    t = e.createTrigger()
    t.setMath(libsbml.parseL3Formula(trigger))
    t.setInitialValue(False)
    t.setPersistent(True)
    a = e.createEventAssignment()
    a.setVariable("Y")
    a.setMath(libsbml.parseL3Formula("Y + 1"))
    path = tmp_path / f"m{len(list(tmp_path.glob('*.xml')))}.xml"
    path.write_text(libsbml.writeSBMLToString(doc))
    return bngsim.Model.from_sbml(str(path))


NEAR = [("near", "lambda(x, c, w, abs(x - c) < w)")]
PEAK = [("peak", "lambda(s, sin(s) >= 1)")]
WINDOW = "abs(time - 2.5) < 0.005"


@pytest.mark.parametrize(
    ("direct", "through", "functions", "trigger", "t_end"),
    [
        (
            f"piecewise(10, {WINDOW}, 0)",
            "piecewise(10, near(time, 2.5, 0.005), 0)",
            NEAR,
            WINDOW,
            5,
        ),
        (
            "piecewise(10, sin(time) >= 1, 0)",
            "piecewise(10, peak(time), 0)",
            PEAK,
            "sin(time) >= 1",
            20,
        ),
    ],
    ids=["abs-window", "sin-peak"],
)
def test_a_function_body_spells_a_call_argument_as_the_call_site_does(
    tmp_path, direct, through, functions, trigger, t_end
):
    plain = _model(tmp_path, direct, trigger)
    called = _model(tmp_path, through, trigger, functions)
    conditions = called.time_discontinuity_conditions()
    assert len(conditions) == len(set(conditions))
    assert sorted(conditions) == sorted(plain.time_discontinuity_conditions())
    # ...and the run is what it was: the extra roots never changed a result.
    runs = [
        bngsim.Simulator(m, method="ode").run(t_span=(0, t_end), n_points=2)
        for m in (_model(tmp_path, direct, trigger), _model(tmp_path, through, trigger, functions))
    ]
    np.testing.assert_array_equal(np.asarray(runs[0].species), np.asarray(runs[1].species))
