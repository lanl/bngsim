"""A rate law too long written out declines the analytical Jacobian up front
(issue #1255).

A compartment whose size is read through a diamond of rules, ``r_i :=
sin(r_{i-1}) + cos(r_{i-2})``, gives the dilution term a rate law whose
functions, written out, are exponential in the depth, and do not collect
(issue #1124 collects ``r_{i-1} + r_{i-2}``, not this). Building the Simulator
derived, checked and printed a derivative as long: 3.1 s at 12 rules, 57 s at
18 and 110 s at 22, where ``jacobian="fd"`` built in under 10 ms. Neither the
native saturable path nor the GH #95 budget stopped it: the native path wrote
the text out, and the budget is asked only before each ``sp.diff``.

Past the inlining limit, a law that does not collect now declines (``kind
"written_out"``) from its predicted length, before the text is written, and the
model runs on the finite-difference Jacobian it would otherwise have paid for.
"""

from __future__ import annotations

import time

import bngsim
import numpy as np
import pytest
from bngsim import _jacobian as J
from bngsim import _saturable_jacobian as S

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _ap(op, *a):
    return f"<apply><{op}/>{''.join(a)}</apply>"


def _ci(x):
    return f"<ci>{x}</ci>"


def _cn(x):
    return f"<cn>{x}</cn>"


def _dilution(n: int) -> str:
    """The issue's model: S decays in c; cv := 1 + 1e-6·r_{n-1}, a size read
    through the diamond, over a jump in S; Sn is diluted in cv."""
    bottom = (
        f"<piecewise><piece>{_ci('S')}{_ap('gt', _ci('S'), _cn(3))}</piece>"
        f"<otherwise>{_ap('times', _cn(2), _ci('S'))}</otherwise></piecewise>"
    )
    rules = [("r0", bottom), ("r1", _ap("plus", _ci("r0"), _cn(1)))]
    rules += [
        (f"r{i}", _ap("plus", _ap("sin", _ci(f"r{i - 1}")), _ap("cos", _ci(f"r{i - 2}"))))
        for i in range(2, n)
    ]
    rules.append(("cv", _ap("plus", _cn(1), _ap("times", _cn(1e-6), _ci(f"r{n - 1}")))))
    rx = "".join(
        f'<assignmentRule variable="{v}"><math {_M}>{b}</math></assignmentRule>' for v, b in rules
    )
    ps = "".join(f'<parameter id="r{i}" constant="false"/>' for i in range(n))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
<listOfCompartments><compartment id="c" size="1" constant="true"/>
<compartment id="cv" size="1" constant="false"/></listOfCompartments>
<listOfSpecies>
<species id="S" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="Sn" compartment="cv" initialConcentration="5" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
</listOfSpecies>
<listOfParameters><parameter id="k" value="0.5" constant="true"/>{ps}</listOfParameters>
<listOfRules>{rx}</listOfRules>
<listOfReactions><reaction id="degS" reversible="false"><listOfReactants>
<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
<kineticLaw><math {_M}><apply><times/><ci>k</ci><ci>S</ci><ci>c</ci></apply></math></kineticLaw>
</reaction></listOfReactions></model></sbml>"""


def _built(n: int, **kw):
    model = bngsim.Model.from_sbml_string(_dilution(n))
    t0 = time.perf_counter()
    sim = bngsim.Simulator(model, method="ode", **kw)
    return model, sim, time.perf_counter() - t0


def _y(sim):
    r = sim.run(t_span=(0.0, 4.0), n_points=9, rtol=1e-9, atol=1e-12)
    return np.asarray(r.species)


@pytest.mark.parametrize("n", [12, 18, 30])
def test_the_issues_model_builds_in_milliseconds_and_runs_as_fd(n):
    """Main took 3.1 s at 12 rules and 57 s at 18 (30 was not measured; 22 took
    110 s)."""
    model, sim, built = _built(n)
    assert built < 2.0, built
    assert model.analytical_jacobian_decline["kind"] == "written_out"
    assert "written out" in model.analytical_jacobian_status
    _fd_model, fd, _ = _built(n, jacobian="fd")
    np.testing.assert_array_equal(_y(sim), _y(fd))


def test_the_codegen_run_builds_fast_too():
    model, sim, built = _built(18, codegen=True)
    assert built < 30.0, built  # the C compile, not the derivation
    assert model.analytical_jacobian_decline["kind"] == "written_out"
    _fd_model, fd, _ = _built(18, jacobian="fd", codegen=True)
    np.testing.assert_allclose(_y(sim), _y(fd), rtol=1e-12, atol=0)


def test_a_law_under_the_limit_keeps_its_analytical_jacobian():
    """Control: at 10 rules the law writes out to 18 thousand characters, under
    the limit, and is derived as before."""
    model, _sim, _ = _built(10)
    assert model.analytical_jacobian_status == "complete"


def _context(n: int):
    model = bngsim.Model.from_sbml_string(_dilution(n))
    funcs = dict(zip(model._core.function_names, model._core.function_expressions, strict=True))
    return model, funcs


def test_the_length_is_asked_before_the_text_is_written(monkeypatch):
    """Neither derivation path writes the law out: at 30 rules it would be tens
    of millions of characters."""
    _model, funcs = _context(30)
    law = funcs["cv"]
    assert J._inlined_length(law, funcs) > 100 * J._INLINE_TEXT_LIMIT

    def refuse(*_a, **_k):
        raise AssertionError("the law was written out")

    monkeypatch.setattr(J, "_inline_functions", refuse)
    text, written_out = J._inlined_for_derivation(law, funcs)
    assert text is None and written_out == J._inlined_length(law, funcs)
    assert S.differentiate_rate_law_native(law, funcs, {"S"}, {"k"}) is None
    assert J.differentiate_rate_law(law, funcs, {"S"}, {"k"}) is None


def test_a_law_that_collects_past_the_limit_is_not_declined():
    """Control: #1124's sum diamond passes the limit written out and collects;
    it gets its collected form, not a decline (its own test pins the answer)."""
    rules = {"r0": "S", "r1": "(S + 1)"}
    rules.update({f"r{i}": f"r{i - 1} + r{i - 2}" for i in range(2, 26)})
    law = "k*r25"
    assert J._inlined_length(law, rules) > J._INLINE_TEXT_LIMIT
    text, written_out = J._inlined_for_derivation(law, rules)
    assert written_out == 0 and text is not None and len(text) < J._INLINE_TEXT_LIMIT


def test_the_decline_has_a_kind_a_census_can_group_by():
    assert "written_out" in J.DECLINE_KINDS
