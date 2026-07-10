"""GH #231 — uncommon MathML, the ``avogadro`` csymbol, and L3v2 operators.

Two MathML translators must agree: ``_eval_ast_numeric`` (the t=0 numeric folder
used for initialAssignment / assignmentRule / stoichiometryMath) and
``_ast_to_exprtk_recursive`` (the runtime expression emitter). Several constructs
were handled by one but not the other:

* the ``avogadro`` csymbol was handled by the ExprTk emitter but not the numeric
  folder, so an initialAssignment reading it silently stayed at its default;
  both now fold to the SI-exact 6.02214076e23 (the suite grades against the older
  SBML-spec 6.02214179e23, so the avogadro precision cases stay failing — a
  deliberate physical-correctness choice);
* L3v2 ``quotient`` / ``rem`` / ``max`` / ``min`` / ``implies`` and ``xor`` were
  absent from the numeric folder, so an initialAssignment using them silently
  stayed at its default;
* ``factorial`` emitted ``exp(lgamma(...))`` — ExprTk has no ``lgamma`` either,
  so every factorial model *load-failed*. It now emits ``tgamma`` (a registered
  C++ function);
* reciprocal / inverse-hyperbolic trig (``sec``, ``csc``, ``cot``, ``arcsinh``,
  ``arccoth`` …) were absent from the numeric folder (00958);
* a 0-arg ``<plus/>`` / ``<times/>`` / ``<and/>`` / ``<or/>`` emitted an empty
  ``()`` and failed to compile.

The avogadro fold is deliberately suppressed in event delay/priority expressions
(``avogadro_value=None``) — folding it to a static priority flipped
simultaneous-event ordering (suite case 01662).
"""

import math

import bngsim

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
_AVOGADRO = 6.02214076e23  # SI-exact; used for both _NA and the SBML csymbol


def _params_from_initial_assignments(assignments):
    """Build a model whose parameters are each fixed by an initialAssignment to
    a MathML ``<apply>`` snippet, and return ``{id: folded value}``."""
    params = "".join(f'<parameter id="{pid}" constant="true"/>' for pid, _ in assignments)
    ias = "".join(
        f'<initialAssignment symbol="{pid}">'
        f'<math xmlns="http://www.w3.org/1998/Math/MathML">{mathml}</math>'
        f"</initialAssignment>"
        for pid, mathml in assignments
    )
    sbml = (
        f'{_HDR}\n<model id="m">'
        f"<listOfParameters>{params}</listOfParameters>"
        f"<listOfInitialAssignments>{ias}</listOfInitialAssignments>"
        f"</model></sbml>"
    )
    model = bngsim.Model.from_sbml_string(sbml)
    return {pid: model.get_param(pid) for pid, _ in assignments}


_CN = lambda v: f"<cn>{v}</cn>"  # noqa: E731
_CSYM_AVO = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/avogadro">avogadro</csymbol>'
)


def test_avogadro_csymbol_folds_in_initial_assignment():
    # The csymbol now folds in the numeric path (previously stayed at default),
    # to the SI-exact value bngsim uses everywhere.
    vals = _params_from_initial_assignments([("p", _CSYM_AVO)])
    assert vals["p"] == _AVOGADRO


def test_l3v2_operators_fold_in_initial_assignments():
    cases = {
        "q": ("<apply><quotient/><cn>9</cn><cn>2</cn></apply>", 4.0),
        "r": ("<apply><rem/><cn>9</cn><cn>2</cn></apply>", 1.0),
        "mx": ("<apply><max/><cn>3</cn><cn>30</cn><cn>7</cn></apply>", 30.0),
        "mn": ("<apply><min/><cn>3</cn><cn>30</cn><cn>7</cn></apply>", 3.0),
        "im_tf": ("<apply><implies/><true/><false/></apply>", 0.0),
        "im_ft": ("<apply><implies/><false/><true/></apply>", 1.0),
        "fac": ("<apply><factorial/><cn type='integer'>4</cn></apply>", 24.0),
        # 1-child xor ⇒ the operand's truth value; xor(true,false) ⇒ true.
        "xor1": ("<apply><xor/><true/></apply>", 1.0),
        "xor2": ("<apply><xor/><true/><false/></apply>", 1.0),
        "xor3": ("<apply><xor/><true/><true/></apply>", 0.0),
    }
    vals = _params_from_initial_assignments([(k, v[0]) for k, v in cases.items()])
    for k, (_, expected) in cases.items():
        assert vals[k] == expected, f"{k}: got {vals[k]} expected {expected}"


def test_reciprocal_and_inverse_trig_fold():
    cases = {
        "sec": ("<apply><sec/><cn>0.5</cn></apply>", 1.0 / math.cos(0.5)),
        "csc": ("<apply><csc/><cn>4.5</cn></apply>", 1.0 / math.sin(4.5)),
        "cot": ("<apply><cot/><cn>0.2</cn></apply>", math.cos(0.2) / math.sin(0.2)),
        "arcsinh": ("<apply><arcsinh/><cn>99</cn></apply>", math.asinh(99)),
        "arccoth": ("<apply><arccoth/><cn>8.2</cn></apply>", math.atanh(1.0 / 8.2)),
        "arcsec": ("<apply><arcsec/><cn>2.3</cn></apply>", math.acos(1.0 / 2.3)),
    }
    vals = _params_from_initial_assignments([(k, v[0]) for k, v in cases.items()])
    for k, (_, expected) in cases.items():
        assert math.isclose(vals[k], expected, rel_tol=1e-12), f"{k}: {vals[k]} != {expected}"


def test_factorial_runtime_expression_compiles_and_runs():
    # An assignmentRule factorial reaches the ExprTk emitter (tgamma), which used
    # to load-fail on the undefined ``lgamma``.
    sbml = (
        f'{_HDR}\n<model id="m">'
        "<listOfParameters>"
        '<parameter id="n" value="4" constant="true"/>'
        '<parameter id="f" constant="false"/>'
        "</listOfParameters>"
        '<listOfRules><assignmentRule variable="f">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<apply><factorial/><ci>n</ci></apply></math></assignmentRule></listOfRules>"
        "</model></sbml>"
    )
    model = bngsim.Model.from_sbml_string(sbml)
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    assert result.n_times == 2


def test_empty_nary_operators_compile_in_kinetic_law():
    # A kinetic law of an empty <plus/> (=0) and empty <times/> (=1) must compile
    # (ExprTk previously emitted "()" ⇒ load failure).
    for op, kind in (("plus", "additive"), ("times", "multiplicative")):
        sbml = (
            f'{_HDR}\n<model id="m_{op}">'
            '<listOfCompartments><compartment id="C" size="1" constant="true"/>'
            "</listOfCompartments>"
            '<listOfSpecies><species id="S" compartment="C" initialAmount="1"'
            ' hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>'
            "</listOfSpecies>"
            '<listOfReactions><reaction id="J" reversible="false">'
            '<listOfProducts><speciesReference species="S" stoichiometry="1"'
            ' constant="true"/></listOfProducts>'
            '<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"<apply><{op}/></apply></math></kineticLaw></reaction></listOfReactions>"
            "</model></sbml>"
        )
        model = bngsim.Model.from_sbml_string(sbml)
        result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
        assert result.n_times == 2, kind


def test_avogadro_in_event_priority_orders_correctly():
    # Regression guard for suite case 01662: two events fire together at t>=1.
    # The higher-priority event (priority = avogadro/6.022e23 ≈ 100, assigns
    # p:=5) executes FIRST; the lower-priority one (priority 1, assigns p:=3)
    # executes LAST and so wins ⇒ p settles at 3. Folding avogadro to a static
    # priority must not disturb this ordering (it is kept on the dynamic path
    # via avogadro_value=None), so p must still be 3, never 5.
    sbml = (
        f'{_HDR}\n<model id="m_pri">'
        '<listOfParameters><parameter id="p" value="0" constant="false"/>'
        "</listOfParameters>"
        "<listOfEvents>"
        '<event id="E0" useValuesFromTriggerTime="true">'
        '<trigger initialValue="false" persistent="true"><math '
        'xmlns="http://www.w3.org/1998/Math/MathML"><apply><geq/>'
        '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">'
        "time</csymbol><cn>1</cn></apply></math></trigger>"
        '<priority><math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<cn type='integer'>1</cn></math></priority>"
        '<listOfEventAssignments><eventAssignment variable="p"><math '
        'xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">3</cn>'
        "</math></eventAssignment></listOfEventAssignments></event>"
        '<event id="E1" useValuesFromTriggerTime="true">'
        '<trigger initialValue="false" persistent="true"><math '
        'xmlns="http://www.w3.org/1998/Math/MathML"><apply><geq/>'
        '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">'
        "time</csymbol><cn>1</cn></apply></math></trigger>"
        '<priority><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><divide/>'
        f"{_CSYM_AVO}<cn type='e-notation'>6.022<sep/>23</cn></apply></math></priority>"
        '<listOfEventAssignments><eventAssignment variable="p"><math '
        'xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">5</cn>'
        "</math></eventAssignment></listOfEventAssignments></event>"
        "</listOfEvents></model></sbml>"
    )
    model = bngsim.Model.from_sbml_string(sbml)
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 3), n_points=4)
    # The event target is promoted to a tracked state ("p" species column).
    names = list(result.species_names)
    final_p = result.species[-1, names.index("p")]
    assert final_p == 3.0, f"expected p=3 (low-priority event last-wins), got {final_p}"
