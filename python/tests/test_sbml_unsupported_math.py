"""GH #97 — fail-closed SBML math translation + dead-translator deletion.

Two systemic loader fixes (see ``dev/notes/gh97_followups_kickoff.md``):

  * **Fail closed.** The live MathML→ExprTk translator used to return ``"0"``
    for any AST node it did not recognise — a *silent wrong RHS* that loaded
    fine and mis-simulated. It now raises :class:`bngsim.ModelError`. The
    blast-radius survey (BioModels + benchmarks: 0 fallback hits; SBML Test
    Suite: only ``distrib`` random-draw csymbols) confirms the only constructs
    reaching the fallback are genuinely unsupported, so this is a loud reject,
    not a regression.
  * **Dead translator removed.** ``_ast_to_exprtk`` / ``_piecewise_to_exprtk``
    were unreachable and had drifted from the live ``_ast_to_exprtk_recursive``
    (min/max/quotient/rem/implies existed only in the live one). They are gone;
    ``test_dead_translator_removed`` keeps them from creeping back.

The L3v2-operator test doubles as the positive case: the operators the dead
copy lacked translate *and simulate* correctly through the surviving live path.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _sbml_loader
from bngsim._exceptions import ModelError

# distrib-package "normal(mu, sigma)" random draw — libsbml parses this csymbol
# to AST_DISTRIB_FUNCTION_NORMAL (type 500) only when the distrib package
# namespace is declared on the document (below). It is a stochastic draw with no
# deterministic ExprTk translation: the canonical genuinely-unsupported node.
_DISTRIB_NORMAL = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/distrib/normal"> normal </csymbol>'
)


def _sbml_l3v2(model_body: str) -> str:
    """Wrap a ``<model>`` body in a minimal SBML L3V2 document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
        'level="3" version="2">\n'
        f"{model_body}\n"
        "</sbml>\n"
    )


def _sbml_l3v1_distrib(model_body: str) -> str:
    """Wrap a ``<model>`` body in an SBML L3V1 document with the distrib package
    enabled, so distrib csymbols parse to their ``AST_DISTRIB_FUNCTION_*`` types
    rather than degrading to bare names."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" '
        'xmlns:distrib="http://www.sbml.org/sbml/level3/version1/distrib/version1" '
        'level="3" version="1" distrib:required="true">\n'
        f"{model_body}\n"
        "</sbml>\n"
    )


# ─── Fail closed: a genuinely-unsupported node raises, never silent 0 ─────────


def test_distrib_draw_raises_modelerror_not_silent_zero():
    """A ``distrib`` random draw in a kinetic law has no deterministic
    translation. Pre-#97 it silently became ``0`` (the reaction rate quietly
    dropped the factor and the model mis-simulated); now the load fails closed
    with an actionable :class:`ModelError`.
    """
    body = f"""
  <model id="distrib_kl">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci>
            <apply>{_DISTRIB_NORMAL}<cn>0</cn><cn>1.5</cn></apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml_string(_sbml_l3v1_distrib(body))

    msg = str(excinfo.value)
    # The error must be actionable: name the AST type number, its symbolic name,
    # and the offending construct (not a generic failure).
    assert "500" in msg
    assert "AST_DISTRIB_FUNCTION_NORMAL" in msg
    assert "normal(0, 1.5)" in msg


def test_unsupported_ast_error_helper_is_actionable():
    """Unit-level check on the fail-closed error builder: type number, symbolic
    name, infix form, and the distrib-specific hint."""
    import libsbml

    # Build a type-500 node by parsing a distrib-namespaced fragment.
    doc = libsbml.readSBMLFromString(
        _sbml_l3v1_distrib(
            """
  <model id="m">
    <listOfParameters><parameter id="p" value="0" constant="false"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="p"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>"""
            + _DISTRIB_NORMAL
            + """<cn>0</cn><cn>1</cn></apply>
      </math></assignmentRule>
    </listOfRules>
  </model>"""
        )
    )
    node = doc.getModel().getRule(0).getMath()
    err = _sbml_loader._unsupported_ast_error(node, node.getType())
    assert isinstance(err, ModelError)
    text = str(err)
    assert "500" in text and "AST_DISTRIB_FUNCTION_NORMAL" in text
    assert "distrib" in text  # the targeted hint fired


# ─── Positive case: the L3v2 operators the dead copy lacked work via live path ─


# Each rate rule integrates a constant built from one L3v2 operator, so the
# species is an exact ramp ``X(t) = rate · t`` — a derivative-free oracle. The
# operators (and their expected constant value):
_OPERATORS = {
    "mx": ("<apply><max/><cn>3</cn><cn>5</cn></apply>", 5.0),  # max(3,5)
    "mn": ("<apply><min/><cn>2</cn><cn>9</cn></apply>", 2.0),  # min(2,9)
    "qt": ("<apply><quotient/><cn>17</cn><cn>5</cn></apply>", 3.0),  # floor(17/5)
    "rm": ("<apply><rem/><cn>17</cn><cn>5</cn></apply>", 2.0),  # 17 - 5·3
    # implies(0,0) = (not 0) or 0 = 1; distinct from both `and`(=0) and `or`(=0),
    # so a correct rate of 1 pins the `(not p) or q` translation specifically.
    "im": ("<apply><implies/><cn>0</cn><cn>0</cn></apply>", 1.0),
}


@pytest.mark.parametrize("codegen", [False, True])
def test_l3v2_operators_translate_and_simulate(codegen):
    """min / max / quotient / rem / implies in rate rules translate via the live
    path and simulate to their closed-form ramps — interpreted and codegen."""

    def species(sid):
        return (
            f'<species id="{sid}" compartment="c" initialConcentration="0" '
            'hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>'
        )

    def rate_rule(sid, mathml):
        return (
            f'<rateRule variable="{sid}">'
            '<math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"{mathml}</math></rateRule>"
        )

    species_xml = "".join(species(sid) for sid in _OPERATORS)
    rules_xml = "".join(rate_rule(sid, ml) for sid, (ml, _) in _OPERATORS.items())
    body = f"""
  <model id="l3v2_ops">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>{species_xml}</listOfSpecies>
    <listOfRules>{rules_xml}</listOfRules>
  </model>"""

    model = bngsim.Model.from_sbml_string(_sbml_l3v2(body))
    res = bngsim.Simulator(model, method="ode", codegen=codegen).run(t_span=(0.0, 4.0), n_points=5)
    t = np.asarray(res.time)
    names = list(res.species_names)
    for sid, (_, rate) in _OPERATORS.items():
        y = np.asarray(res.species[:, names.index(sid)])
        assert np.allclose(y, rate * t, atol=1e-9, rtol=1e-7), (
            f"{sid}: expected ramp at rate {rate}, got {y}"
        )


# ─── FIX 2 guard: the dead translator stays dead ─────────────────────────────


def test_dead_translator_removed():
    """``_ast_to_exprtk`` / ``_piecewise_to_exprtk`` were unreachable and had
    drifted from the live translator; #97 deleted them. If either reappears,
    coverage can silently diverge again — fail loudly here instead.
    """
    assert not hasattr(_sbml_loader, "_ast_to_exprtk")
    assert not hasattr(_sbml_loader, "_piecewise_to_exprtk")
    # The single surviving implementation is still present.
    assert hasattr(_sbml_loader, "_ast_to_exprtk_recursive")
    assert hasattr(_sbml_loader, "_ast_to_exprtk_with_funcdefs")
