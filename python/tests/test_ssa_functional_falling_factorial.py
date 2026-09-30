"""SSA falling factorial for SBML laws the loader evaluates as written.

A reaction the mass-action classifier declines is emitted Functional and its
kinetic law is evaluated literally. Under SSA a species the law reads m times
then contributed n^m, where the number of distinct m-tuples of its molecules is
n(n−1)…(n−m+1): ``2B -> P`` at ``k·B·B`` kept firing with a single B left,
while the same reaction from a `.net` (``$B + $B``), or a law the classifier
lifts to mass action, fires at 0. Such a law now gets the falling factorial
when it is a product of its species; one that writes its own combinatorics
(``X·(X−1)``) or reads a species anywhere else is left as written.

Oracles are closed-form propensities and a Poisson count.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _sbml(*, comps, species, reactants, products, law, params, rules=""):
    cs = "".join(
        f'<compartment id="{c}" spatialDimensions="3" size="{v}" constant="{str(k).lower()}"/>'
        for c, v, k in comps
    )
    sp = "".join(
        f'<species id="{s}" compartment="{c}" initialAmount="{n}" '
        f'hasOnlySubstanceUnits="{str(h).lower()}" boundaryCondition="{str(b).lower()}" '
        'constant="false"/>'
        for s, c, n, h, b in species
    )
    pr = "".join(f'<parameter id="{k}" value="{v}" constant="true"/>' for k, v in params)

    def refs(items):
        return "".join(
            f'<speciesReference species="{s}" stoichiometry="{st}" constant="true"/>'
            for s, st in items
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="m">
<listOfCompartments>{cs}</listOfCompartments>
<listOfSpecies>{sp}</listOfSpecies>
<listOfParameters>{pr}</listOfParameters>
{rules}
<listOfReactions><reaction id="J" reversible="false">
<listOfReactants>{refs(reactants)}</listOfReactants>
<listOfProducts>{refs(products)}</listOfProducts>
<kineticLaw><math {MATH}>{law}</math></kineticLaw>
</reaction></listOfReactions>
</model>
</sbml>"""


def _times(*names):
    return "<apply><times/>" + "".join(f"<ci>{n}</ci>" for n in names) + "</apply>"


def _boundary_dimer(v=10.0, nb=3, k=0.01):
    """``2B -> P`` with B a boundary species: the classifier declines it."""
    return bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", v, True)],
            species=[("B", "C", nb, False, True), ("P", "C", 0, False, False)],
            reactants=[("B", 2)],
            products=[("P", 1)],
            law=_times("k", "B", "B", "C"),
            params=[("k", k)],
        )
    )


def test_a_boundary_reactant_takes_the_falling_factorial():
    v, k = 10.0, 0.01
    m = _boundary_dimer(v=v, k=k)
    for n in (0, 1, 2, 3, 7):
        got = m.propensities([n / v, 0.0])[0]
        assert got == pytest.approx(k * n * (n - 1) / v, rel=1e-14, abs=0.0), (n, got)
    assert m.propensities([1 / v, 0.0])[0] == 0.0


def test_a_single_boundary_molecule_never_pairs():
    """With one B the reaction has no pair to fire on. It used to fire at k/V
    for ever: P grew without bound."""
    m = _boundary_dimer(nb=1)
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 1e4), n_points=3, seed=1)
    assert np.asarray(r.species)[-1, list(r.species_names).index("P")] == 0.0


def test_boundary_dimer_count_is_poisson_at_the_pair_rate():
    """B = 3 is constant, so P(T) is Poisson(k·3·2/V·T) = Poisson(6)."""
    v, k, t_end, reps = 10.0, 0.01, 1000.0, 400
    m = _boundary_dimer(v=v, k=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaRoundingWarning)
        r = bngsim.Simulator(m, method="ssa").run_replicates(
            reps, t_span=(0, t_end), n_points=2, seed=4, squeeze=True
        )
    p = np.asarray(r.species)[:, -1, list(r.species_names).index("P")] * v
    want = k * 3 * 2 / v * t_end
    assert abs(p.mean() - want) <= 4.5 * np.sqrt(want / reps), (p.mean(), want)


def test_reactants_in_compartments_of_different_sizes():
    """``2A + B -> P`` with A in C1 (V = 2) and B in C2 (V = 5): the law
    ``k·[A]²·[B]·C1`` gives the propensity k·nA(nA−1)·nB/(V1·V2). A lone A
    dimerised and went to −9."""
    v1, v2, k = 2.0, 5.0, 1.0
    m = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C1", v1, True), ("C2", v2, True)],
            species=[
                ("A", "C1", 1, False, False),
                ("B", "C2", 5, False, False),
                ("P", "C1", 0, False, False),
            ],
            reactants=[("A", 2), ("B", 1)],
            products=[("P", 1)],
            law=_times("k", "A", "A", "B", "C1"),
            params=[("k", k)],
        )
    )
    for na, nb in ((1, 5), (2, 5), (6, 3)):
        got = m.propensities([na / v1, nb / v2, 0.0])[0]
        want = k * na * (na - 1) * nb / (v1 * v2)
        assert got == pytest.approx(want, rel=1e-14, abs=0.0), (na, nb, got, want)
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 100), n_points=2, seed=1)
    names = list(r.species_names)
    assert np.asarray(r.species)[-1, names.index("A")] * v1 == 1.0


def test_an_amount_law_divided_by_its_volume():
    """hOSU=true ``k·A·A/C``: a division, so the classifier declines it."""
    v, k = 10.0, 0.01
    law = f"<apply><divide/>{_times('k', 'A', 'A')}<ci>C</ci></apply>"
    m = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", v, True)],
            species=[("A", "C", 20, True, False), ("B", "C", 0, True, False)],
            reactants=[("A", 2)],
            products=[("B", 1)],
            law=law,
            params=[("k", k)],
        )
    )
    for n in (20, 2, 1):
        got = m.propensities([n / v, 0.0])[0]
        assert got == pytest.approx(k * n * (n - 1) / v, rel=1e-14, abs=0.0), (n, got)


def test_an_assignment_rule_compartment():
    """A compartment whose size an assignment rule sets keeps the law
    Functional (its live size stays a symbol)."""
    rules = (
        f'<listOfRules><assignmentRule variable="C"><math {MATH}><ci>Vp</ci></math>'
        "</assignmentRule></listOfRules>"
    )
    m = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", 10.0, False)],
            species=[("A", "C", 20, False, False), ("B", "C", 0, False, False)],
            reactants=[("A", 2)],
            products=[("B", 1)],
            law=_times("k", "A", "A", "C"),
            params=[("k", 0.01), ("Vp", 10.0)],
            rules=rules,
        )
    )
    for n in (20, 2, 1):
        got = m.propensities([n / 10.0, 0.0])[0]
        assert got == pytest.approx(0.01 * n * (n - 1) / 10.0, rel=1e-12, abs=0.0), (n, got)


@pytest.mark.parametrize(
    ("label", "law", "want"),
    [
        # writes its own combinatorics: k·X·(X−1)/2·C with X a boundary species
        (
            "own-combinatorics",
            "<apply><times/><ci>k</ci><ci>B</ci><apply><minus/><ci>B</ci><cn>1</cn></apply>"
            "<ci>C</ci></apply>",
            lambda k, c, v: k * c * (c - 1) * v,
        ),
        # a species in the denominator
        (
            "denominator",
            f"<apply><divide/>{_times('k', 'B', 'B', 'C')}<ci>B</ci></apply>",
            lambda k, c, v: k * c * v,
        ),
    ],
)
def test_a_law_that_is_not_a_product_of_its_species_is_left_as_written(label, law, want):
    v, k = 10.0, 0.01
    m = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", v, True)],
            species=[("B", "C", 3, False, True), ("P", "C", 0, False, False)],
            reactants=[("B", 2)],
            products=[("P", 1)],
            law=law,
            params=[("k", k)],
        )
    )
    c = 0.5  # 5 molecules at V = 10
    assert m.propensities([c, 0.0])[0] == pytest.approx(want(k, c, v), rel=1e-14), label


def test_the_ode_is_unchanged():
    """The correction is SSA-only: the ODE still integrates the law as
    written, dP/dt = k·[B]²·C / C with B constant."""
    v, k, nb = 10.0, 0.01, 3
    m = _boundary_dimer(v=v, nb=nb, k=k)
    r = bngsim.Simulator(m, method="ode").run(t_span=(0, 100), n_points=2)
    p = np.asarray(r.species)[-1, list(r.species_names).index("P")]
    assert p == pytest.approx(k * (nb / v) ** 2 * 100, rel=1e-8)
