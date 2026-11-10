"""SBML Level 1 writes a stoichiometry as an integer over a positive integer.

A Level 1 ``<speciesReference>`` carries ``stoichiometry`` and ``denominator``:
``stoichiometry="1" denominator="2"`` is 1/2. Levels 2 and 3 have no denominator,
and every reader in bngsim's SBML loader takes ``getStoichiometry()``, the
numerator alone. So a Level 1 model with a denominator ran with the wrong
stoichiometry and no warning:

| model | bngsim before | closed form |
|---|---|---|
| 1/2 A -> 3/2 B at k*A, A(0) = 3: A(1), B(1) | 1.1036, 5.6891 | 1.8196, 3.5412 |
| 4/2 A -> B at k*A*A, A(0) = 10: A(0.1) | 2.0000 | 3.3333 |

The loader now folds each denominator into its stoichiometry once, at load.

libSBML's own Level 1 -> Level 2 conversion writes the denominator as a Level 2
``stoichiometryMath`` holding a bare rational, ``<cn type="rational"> 1 <sep/> 2
</cn>``, and libSBML (5.21) reads exactly that element back into the reference's
stoichiometry and denominator (1 and 2) and keeps no stoichiometryMath, with
nothing logged. bngsim read 1 there too, and so does RoadRunner. The same fold
covers it.

Oracles: the closed forms, among them the exact solution of a linear system
shaped like SBML Test Suite case 00022; that case's own expected results at Level
1 where the suite is checked out (the suite runner grades the highest level a
case has, so no Level 1 file was graded, and the suite is LGPL, so it is not
copied here); libSBML's own Level 1 -> Level 2 conversion, which bngsim reads
through another path; and, for the stochastic engine, the stoichiometry matrix it
fires and the parity of the count.
"""

from __future__ import annotations

import csv
import gzip
import os
import warnings
import zipfile
from pathlib import Path

import bngsim
import libsbml
import numpy as np
import pytest
from bngsim._sbml_loader import _fold_stoichiometry_denominators


def _level1(reactant=("1", "2"), product=("3", "2"), law="k*A", a0="3") -> str:
    """A -> B at ``law``, k = 1, in a compartment of volume 1. ``reactant`` and
    ``product`` are (stoichiometry, denominator); a denominator of None leaves the
    attribute out, which Level 1 reads as 1."""

    def ref(species, stoich, den):
        den_attr = "" if den is None else f' denominator="{den}"'
        return f'<speciesReference species="{species}" stoichiometry="{stoich}"{den_attr}/>'

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level1" level="1" version="2"><model name="m">
<listOfCompartments><compartment name="c" volume="1"/></listOfCompartments>
<listOfSpecies><species name="A" compartment="c" initialAmount="{a0}" boundaryCondition="false"/>
<species name="B" compartment="c" initialAmount="0" boundaryCondition="false"/></listOfSpecies>
<listOfParameters><parameter name="k" value="1"/></listOfParameters>
<listOfReactions><reaction name="r" reversible="false">
<listOfReactants>{ref("A", *reactant)}</listOfReactants>
<listOfProducts>{ref("B", *product)}</listOfProducts>
<kineticLaw formula="{law}"/></reaction></listOfReactions></model></sbml>"""


def _ode(model, t_end, n=11, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Simulator(model, method="ode", **kw).run(
            (0.0, t_end), n, rtol=1e-10, atol=1e-12
        )


def _column(result, name):
    return np.asarray(result.species)[:, list(result.species_names).index(name)]


@pytest.mark.parametrize("load", ["string", "file"])
def test_a_level_1_denominator_is_part_of_the_stoichiometry(load, tmp_path):
    """1/2 A -> 3/2 B at k*A: [A] = 3 exp(-t/2) and [B] = 3 (3 - [A])."""
    xml = _level1()
    if load == "file":
        path = tmp_path / "level1.xml"
        path.write_text(xml)
        model = bngsim.Model.from_sbml(str(path))
    else:
        model = bngsim.Model.from_sbml_string(xml)
    r = _ode(model, 4.0)
    a = 3 * np.exp(-np.asarray(r.time) / 2)
    np.testing.assert_allclose(_column(r, "A"), a, rtol=1e-8)
    np.testing.assert_allclose(_column(r, "B"), 3 * (3 - a), rtol=1e-8)


def test_a_denominator_that_leaves_a_whole_number():
    """4/2 A -> B at k*A*A is 2A -> B: [A] = 10 / (1 + 2*10*t)."""
    r = _ode(bngsim.Model.from_sbml_string(_level1(("4", "2"), ("1", None), "k*A*A", "10")), 0.5)
    np.testing.assert_allclose(_column(r, "A"), 10 / (1 + 20 * np.asarray(r.time)), rtol=1e-8)


def test_libsbml_level_2_conversion_reads_the_same():
    """libSBML's Level 1 -> Level 2 conversion writes the denominator as a
    ``stoichiometryMath`` of 1/2 and 3/2, which bngsim reads through its own,
    separate path; the two documents give one trajectory."""
    doc = libsbml.readSBMLFromString(_level1())
    assert doc.setLevelAndVersion(2, 4, False)
    ref = doc.getModel().getReaction(0).getReactant(0)
    assert ref.isSetStoichiometryMath()
    level2 = libsbml.writeSBMLToString(doc)
    a = _ode(bngsim.Model.from_sbml_string(_level1()), 4.0)
    b = _ode(bngsim.Model.from_sbml_string(level2), 4.0)
    for name in ("A", "B"):
        np.testing.assert_allclose(_column(a, name), _column(b, name), rtol=1e-12)


def test_the_sensitivity_carries_the_denominator():
    """d[A]/dk = -t/2 * 3 exp(-k t/2) at k = 1, the 1/2 in it the denominator's."""
    model = bngsim.Model.from_sbml_string(_level1())
    r = _ode(model, 2.0, sensitivity_params=["k"])
    t = np.asarray(r.time)
    ia = list(r.species_names).index("A")
    np.testing.assert_allclose(
        np.asarray(r.sensitivities)[:, ia, 0], -t / 2 * 3 * np.exp(-t / 2), rtol=1e-6, atol=1e-10
    )


def test_ssa_fires_the_folded_stoichiometry():
    """4/2 A -> B is a mass-action 2A -> B: the stoichiometry matrix the engines
    use reads -2 (the numerator alone read -4), and the stochastic engine consumes
    two A per firing, so A stays even and never goes below 0 from 10."""
    model = bngsim.Model.from_sbml_string(_level1(("4", "2"), ("1", None), "k*A*A", "10"))
    assert model.stoichiometry_matrix()[:, 0].tolist() == [-2.0, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(model, method="ssa").run((0.0, 1.0), 21, seed=7)
    a = _column(r, "A")
    assert a[0] == 10 and a[-1] < 10
    assert (np.mod(a, 2) == 0).all() and (a >= 0).all()


@pytest.mark.parametrize("method,kw", [("ssa", {}), ("psa", {"poplevel": 100})])
def test_the_stochastic_engines_refuse_a_fractional_level_1_stoichiometry(method, kw):
    """1/2 and 3/2 are not whole firings; SSA and PSA say so by name rather than
    fire the numerator (PSA ran 1/2 A -> 3/2 B as A -> 3 B)."""
    model = bngsim.Model.from_sbml_string(_level1())
    with pytest.raises(bngsim.SsaValidationError, match="non_integer_stoichiometry"):
        bngsim.Simulator(model, method=method, **kw)


def test_a_missing_denominator_reads_as_one():
    r = _ode(bngsim.Model.from_sbml_string(_level1(("1", None), ("1", None))), 2.0)
    a = 3 * np.exp(-np.asarray(r.time))
    np.testing.assert_allclose(_column(r, "A"), a, rtol=1e-8)


def test_a_denominator_that_is_not_positive_is_refused():
    with pytest.raises(bngsim.ModelError, match="denominator 0"):
        bngsim.Model.from_sbml_string(_level1(("1", "0")))


def test_a_stoichiometry_with_no_denominator_is_left_alone():
    """A Level 3 stoichiometry of 2 has no denominator and stays 2."""
    doc = libsbml.readSBMLFromString(_level1(("4", "2")))
    assert doc.setLevelAndVersion(3, 2, False)
    model = doc.getModel()
    reaction = model.getReaction(0)
    ref = reaction.getReactant(0)
    ref.setStoichiometry(2.0)
    _fold_stoichiometry_denominators(doc)
    assert ref.getStoichiometry() == 2.0


# --------------------------------------------------------------------------- #
# A Level 2 stoichiometryMath that is a bare rational, which libSBML drops
# --------------------------------------------------------------------------- #
_RATIONAL = '<cn type="rational"> {} <sep/> {} </cn>'


def _level2(reactant=None, product=None) -> str:
    """A -> B at k*A, k = 1, A(0) = 3, Level 2 Version 4. ``reactant`` and
    ``product`` are the MathML of each reference's stoichiometryMath, or None."""

    def ref(species, math):
        if math is None:
            return f'<speciesReference species="{species}"/>'
        return (
            f'<speciesReference species="{species}"><stoichiometryMath>'
            f'<math xmlns="http://www.w3.org/1998/Math/MathML">{math}</math>'
            "</stoichiometryMath></speciesReference>"
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="m">
<listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
<listOfSpecies><species id="A" compartment="c" initialAmount="3"/>
<species id="B" compartment="c" initialAmount="0"/></listOfSpecies>
<listOfParameters><parameter id="k" value="1"/></listOfParameters>
<listOfReactions><reaction id="r" reversible="false">
<listOfReactants>{ref("A", reactant)}</listOfReactants>
<listOfProducts>{ref("B", product)}</listOfProducts>
<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
</reaction></listOfReactions></model></sbml>"""


def test_libsbml_reads_a_bare_rational_stoichiometry_math_as_a_denominator():
    """The premise of the fold at Level 2: libSBML keeps no stoichiometryMath and
    puts 1/2 on the reference as stoichiometry 1 over denominator 2. If this
    starts failing, libSBML has changed how it reads the element."""
    doc = libsbml.readSBMLFromString(_level2(_RATIONAL.format(1, 2)))
    model = doc.getModel()
    reaction = model.getReaction(0)
    ref = reaction.getReactant(0)
    assert doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) == 0
    assert not ref.isSetStoichiometryMath()
    assert (ref.getStoichiometry(), ref.getDenominator()) == (1.0, 2)


@pytest.mark.parametrize("load", ["string", "file", "gzip", "zip"])
def test_a_bare_rational_stoichiometry_math_is_read_as_its_value(load, tmp_path):
    """1/2 A -> 3/2 B written as two Level 2 rationals: the Level 1 closed form."""
    xml = _level2(_RATIONAL.format(1, 2), _RATIONAL.format(3, 2))
    if load == "string":
        model = bngsim.Model.from_sbml_string(xml)
    elif load == "file":
        (tmp_path / "m.xml").write_text(xml)
        model = bngsim.Model.from_sbml(str(tmp_path / "m.xml"))
    else:
        if not libsbml.SBMLReader.hasZlib():
            pytest.skip("libSBML was built without zlib")
        path = tmp_path / f"m.xml.{'gz' if load == 'gzip' else 'zip'}"
        if load == "gzip":
            path.write_bytes(gzip.compress(xml.encode()))
        else:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("m.xml", xml)
        model = bngsim.Model.from_sbml(str(path))
    r = _ode(model, 4.0)
    a = 3 * np.exp(-np.asarray(r.time) / 2)
    np.testing.assert_allclose(_column(r, "A"), a, rtol=1e-8)
    np.testing.assert_allclose(_column(r, "B"), 3 * (3 - a), rtol=1e-8)


def test_a_stoichiometry_math_libsbml_kept_is_left_as_it_is():
    """<cn> 0.5 </cn> libSBML keeps as stoichiometryMath; beside it a 3/2 it
    read as a denominator is folded."""
    r = _ode(bngsim.Model.from_sbml_string(_level2("<cn> 0.5 </cn>", _RATIONAL.format(3, 2))), 4.0)
    a = 3 * np.exp(-np.asarray(r.time) / 2)
    np.testing.assert_allclose(_column(r, "B"), 3 * (3 - a), rtol=1e-8)


def test_a_level_2_rational_over_zero_is_refused():
    with pytest.raises(bngsim.ModelError, match="denominator 0"):
        bngsim.Model.from_sbml_string(_level2(_RATIONAL.format(1, 0)))


def _two_reactions(level: int) -> str:
    """1/2 A + 2 B -> C at k1*A and D -> 3/2 E at k2*D, k1 = 1, k2 = 0.5, with
    A(0) = 3, B(0) = 20, D(0) = 2: each reference keeps its own denominator."""
    if level == 1:

        def ref(sp, num, den=None):
            d = "" if den is None else f' denominator="{den}"'
            return f'<speciesReference species="{sp}" stoichiometry="{num}"{d}/>'

        def species(sp, x0):
            return (
                f'<species name="{sp}" compartment="c" initialAmount="{x0}" '
                'boundaryCondition="false"/>'
            )

        head = (
            '<sbml xmlns="http://www.sbml.org/sbml/level1" level="1" version="2"><model name="m">'
            '<listOfCompartments><compartment name="c" volume="1"/></listOfCompartments>'
        )
        params = '<parameter name="k1" value="1"/><parameter name="k2" value="0.5"/>'

        def reaction(rid, reactants, products, law):
            return (
                f'<reaction name="{rid}" reversible="false"><listOfReactants>{reactants}'
                f"</listOfReactants><listOfProducts>{products}</listOfProducts>"
                f'<kineticLaw formula="{law}"/></reaction>'
            )

        r1 = reaction("r1", ref("A", 1, 2) + ref("B", 2), ref("C", 1), "k1*A")
        r2 = reaction("r2", ref("D", 1), ref("E", 3, 2), "k2*D")
    else:

        def ref(sp, num, den=None):
            if den is None:
                return f'<speciesReference species="{sp}" stoichiometry="{num}"/>'
            return (
                f'<speciesReference species="{sp}"><stoichiometryMath>'
                f'<math xmlns="http://www.w3.org/1998/Math/MathML">{_RATIONAL.format(num, den)}'
                "</math></stoichiometryMath></speciesReference>"
            )

        def species(sp, x0):
            return f'<species id="{sp}" compartment="c" initialAmount="{x0}"/>'

        head = (
            '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">'
            '<model id="m"><listOfCompartments><compartment id="c" size="1"/></listOfCompartments>'
        )
        params = '<parameter id="k1" value="1"/><parameter id="k2" value="0.5"/>'

        def reaction(rid, reactants, products, law):
            a, b = law.split("*")
            return (
                f'<reaction id="{rid}" reversible="false"><listOfReactants>{reactants}'
                f"</listOfReactants><listOfProducts>{products}</listOfProducts><kineticLaw>"
                '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/>'
                f"<ci>{a}</ci><ci>{b}</ci></apply></math></kineticLaw></reaction>"
            )

        r1 = reaction("r1", ref("A", 1, 2) + ref("B", 2), ref("C", 1), "k1*A")
        r2 = reaction("r2", ref("D", 1), ref("E", 3, 2), "k2*D")
    sp = "".join(species(x, x0) for x, x0 in (("A", 3), ("B", 20), ("C", 0), ("D", 2), ("E", 0)))
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>{head}<listOfSpecies>{sp}</listOfSpecies>'
        f"<listOfParameters>{params}</listOfParameters><listOfReactions>{r1}{r2}"
        "</listOfReactions></model></sbml>"
    )


@pytest.mark.parametrize("level", [1, 2])
def test_each_reference_keeps_its_own_denominator(level):
    """[A] = 3 exp(-t/2), [B] = 20 - 4 (3 - [A]), [C] = 2 (3 - [A]),
    [D] = 2 exp(-t/2), [E] = 3/2 (2 - [D])."""
    r = _ode(bngsim.Model.from_sbml_string(_two_reactions(level)), 4.0)
    t = np.asarray(r.time)
    a, d = 3 * np.exp(-t / 2), 2 * np.exp(-0.5 * t)
    for name, want in (
        ("A", a),
        ("B", 20 - 4 * (3 - a)),
        ("C", 2 * (3 - a)),
        ("D", d),
        ("E", 1.5 * (2 - d)),
    ):
        np.testing.assert_allclose(_column(r, name), want, rtol=1e-8, err_msg=name)


def test_a_linear_system_shaped_like_suite_case_00022():
    """S1 -> 3/10 S2 at V*k1*S1 and 7/10 S2 -> S1 at V*k2*S2 in a compartment of
    V = 0.3, Level 1, the shape of SBML Test Suite case 00022 in a model of our
    own: the exact solution is expm(M t) x0 for d[S]/dt = M [S]."""
    from scipy.linalg import expm

    def species(sp, x0):
        return (
            f'<species name="{sp}" compartment="cell" initialAmount="{x0}" '
            'boundaryCondition="false"/>'
        )

    def ref(sp, frac=""):
        return f'<speciesReference species="{sp}"{frac}/>'

    def reaction(rid, reactant, product, law):
        return (
            f'<reaction name="{rid}" reversible="false">'
            f"<listOfReactants>{reactant}</listOfReactants>"
            f"<listOfProducts>{product}</listOfProducts>"
            f'<kineticLaw formula="{law}"/></reaction>'
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sbml xmlns="http://www.sbml.org/sbml/level1" level="1" version="2"><model name="m">'
        '<listOfCompartments><compartment name="cell" volume="0.3"/></listOfCompartments>'
        f"<listOfSpecies>{species('S1', 0.0015)}{species('S2', 0)}</listOfSpecies>"
        '<listOfParameters><parameter name="k1" value="0.35"/>'
        '<parameter name="k2" value="180"/></listOfParameters><listOfReactions>'
        + reaction(
            "fwd", ref("S1"), ref("S2", ' stoichiometry="3" denominator="10"'), "cell*k1*S1"
        )
        + reaction(
            "back", ref("S2", ' stoichiometry="7" denominator="10"'), ref("S1"), "cell*k2*S2"
        )
        + "</listOfReactions></model></sbml>"
    )
    r = _ode(bngsim.Model.from_sbml_string(xml), 25.0, n=51)
    k1, k2 = 0.35, 180.0
    m = np.array([[-k1, k2], [0.3 * k1, -0.7 * k2]])  # concentrations: rates over V cancel V
    x0 = np.array([0.0015 / 0.3, 0.0])
    want = np.array([expm(m * ti) @ x0 for ti in np.asarray(r.time)])
    got = np.column_stack([_column(r, "S1"), _column(r, "S2")])
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-14)


# --------------------------------------------------------------------------- #
# SBML Test Suite case 00022, the one Level 1 case with a denominator
# --------------------------------------------------------------------------- #
_SUITE = Path(
    os.environ.get(
        "SBML_TEST_SUITE_DIR", os.path.expanduser("~/Code/sbml-test-suite/cases/semantic")
    )
)


def test_sbml_test_suite_case_00022_at_level_1():
    """S1 -> 0.3 S2 and 0.7 S2 -> S1, written 3/10 and 7/10 at Level 1, in a
    compartment of 0.3. Graded by the suite's rule, |sim - expected| <= 1e-6 +
    1e-4 |expected|, against its results; and the Level 1 file gives the Level 3
    file's trajectory."""
    case = _SUITE / "00022"
    if not (case / "00022-sbml-l1v2.xml").is_file():
        pytest.skip("SBML Test Suite case 00022 not present")
    with open(case / "00022-results.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    names, expected = rows[0][1:], np.array(rows[1:], dtype=float)
    runs = {}
    for level in ("l1v2", "l3v2"):
        model = bngsim.Model.from_sbml(str(case / f"00022-sbml-{level}.xml"))
        r = _ode(model, 25.0, n=51)
        runs[level] = np.column_stack([_column(r, n) * 0.3 for n in names])  # amounts
    tol = 1e-6 + 1e-4 * np.abs(expected[:, 1:])
    assert (np.abs(runs["l1v2"] - expected[:, 1:]) <= tol).all()
    np.testing.assert_allclose(runs["l1v2"], runs["l3v2"], rtol=1e-12, atol=1e-18)
