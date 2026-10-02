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


def _sbml(*, comps, species, reactants, products, law, params, rules="", modifiers=(), local=()):
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
    pr = "".join(
        f'<parameter id="{k}" value="{v}" constant="{str(not rest or rest[0]).lower()}"/>'
        for k, v, *rest in params
    )
    mods = "".join(f'<modifierSpeciesReference species="{m}"/>' for m in modifiers)
    mods = f"<listOfModifiers>{mods}</listOfModifiers>" if mods else ""
    lps = "".join(f'<localParameter id="{k}" value="{v}"/>' for k, v in local)
    lps = f"<listOfLocalParameters>{lps}</listOfLocalParameters>" if lps else ""

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
<listOfProducts>{refs(products)}</listOfProducts>{mods}
<kineticLaw><math {MATH}>{law}</math>{lps}</kineticLaw>
</reaction></listOfReactions>
</model>
</sbml>"""


def _times(*names):
    return (
        "<apply><times/>"
        + "".join(n if n.startswith("<") else f"<ci>{n}</ci>" for n in names)
        + "</apply>"
    )


def _pow(base, exp):
    base = base if base.startswith("<") else f"<ci>{base}</ci>"
    return f"<apply><power/>{base}{exp}</apply>"


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


# ── shapes the walker reads ─────────────────────────────────────────────────


def _boundary_law(law, *, nb=3, v=10.0, k=0.01, stoich=2, local=()):
    return bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", v, True)],
            species=[("B", "C", nb, False, True), ("P", "C", 0, False, False)],
            reactants=[("B", stoich)],
            products=[("P", 1)],
            law=law,
            params=[("k", k)],
            local=local,
        )
    )


@pytest.mark.parametrize(
    "law",
    [
        pytest.param(_times("k", _pow("B", "<cn>3</cn>"), "C"), id="power"),
        pytest.param(_times("k", "B", _pow("B", "<cn>2</cn>"), "C"), id="factor-and-power"),
        pytest.param(_times("k", _pow("B", "<cn>3.0</cn>"), "C"), id="real-exponent"),
        pytest.param(
            _times("k", _pow("B", '<cn type="rational">6<sep/>2</cn>'), "C"), id="rational"
        ),
        pytest.param(
            f"<apply><divide/>{_times('k', _pow(_times('B', 'C'), '<cn>2</cn>'), 'B')}"
            "<ci>C</ci></apply>",
            id="power-of-a-product",
        ),
        pytest.param(_times("k", "B", "B", "B", "C", _pow("B", "<cn>0</cn>")), id="x-to-the-0"),
    ],
)
def test_a_trimer_takes_three_factors(law):
    """``3B -> P``: n(n−1)(n−2), however the law writes B³. Two B molecules
    cannot make a triple."""
    v, k = 10.0, 0.01
    m = _boundary_law(law, stoich=3)
    for n in (0, 1, 2, 3, 7):
        got = m.propensities([n / v, 0.0])[0]
        want = k * n * (n - 1) * (n - 2) / v**2
        assert got == pytest.approx(want, rel=1e-13, abs=0.0), (n, got, want)


PIECE_ON = "<apply><lt/><ci>k</ci><cn>1</cn></apply>"
PIECE_READS_B = "<apply><gt/><ci>B</ci><cn>0</cn></apply>"


@pytest.mark.parametrize("cond", [PIECE_ON, PIECE_READS_B], ids=["param-cond", "species-cond"])
def test_a_piecewise_whose_live_branches_agree(cond):
    """``piecewise(k·B·B·C, cond, 0)``: whichever branch is in force, the
    value is 0 or the same product of B, so the pair count applies. A
    condition only selects; it may read B."""
    v, k = 10.0, 0.01
    law = (
        f"<piecewise><piece>{_times('k', 'B', 'B', 'C')}{cond}</piece>"
        "<otherwise><cn>0</cn></otherwise></piecewise>"
    )
    m = _boundary_law(law)
    for n in (1, 2, 3):
        got = m.propensities([n / v, 0.0])[0]
        assert got == pytest.approx(k * n * (n - 1) / v, rel=1e-14, abs=0.0), (n, got)


def test_a_piecewise_whose_branches_disagree_is_left_as_written():
    v, k = 10.0, 0.01
    law = (
        f"<piecewise><piece>{_times('k', 'B', 'B', 'C')}{PIECE_ON}</piece>"
        f"<otherwise>{_times('k', 'B', 'C')}</otherwise></piecewise>"
    )
    m = _boundary_law(law)
    for n in (1, 3):
        got = m.propensities([n / v, 0.0])[0]
        assert got == pytest.approx(k * n * n / v, rel=1e-14), (n, got)


def test_a_local_parameter_that_shadows_a_species():
    """A local ``B`` inside the law is a number, not the species."""
    v, k = 10.0, 0.01
    m = _boundary_law(_times("k", "B", "B", "C"), local=[("B", 0.5)])
    for n in (1, 3):
        assert m.propensities([n / v, 0.0])[0] == pytest.approx(k * 0.25 * v, rel=1e-14)


@pytest.mark.parametrize(
    "exp", ["<infinity/>", "<notanumber/>", "<cn>INF</cn>"], ids=["inf", "nan", "cn-inf"]
)
@pytest.mark.parametrize("boundary", [True, False], ids=["boundary", "floating"])
def test_a_non_finite_exponent_loads(exp, boundary):
    """``B^inf`` is not an integer power. Both the mass-action classifier and
    the walker used to take ``int()`` of it and raise OverflowError."""
    bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", 10.0, True)],
            species=[("B", "C", 3, False, boundary), ("P", "C", 0, False, False)],
            reactants=[("B", 2)],
            products=[("P", 1)],
            law=_times("k", _pow("B", exp), "C"),
            params=[("k", 0.01)],
        )
    )


RATE_RULE_X = (
    f'<listOfRules><rateRule variable="X"><math {MATH}><cn>0</cn></math></rateRule></listOfRules>'
)
ASSIGN_X = (
    f'<listOfRules><assignmentRule variable="X"><math {MATH}><ci>q</ci></math>'
    "</assignmentRule></listOfRules>"
)


@pytest.mark.parametrize("rules", [RATE_RULE_X, ASSIGN_X], ids=["rate-rule", "assignment-rule"])
def test_a_rule_target_is_not_a_count(rules):
    """X is continuous (a rule sets it), so ``X·X`` is its square, not a pair
    count: 1/V of X is 0.1, not "one molecule". A is a single reactant."""
    v, k = 10.0, 0.01
    m = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", v, True)],
            species=[
                ("A", "C", 3, False, True),
                ("X", "C", 1, False, False),
                ("P", "C", 0, False, False),
            ],
            reactants=[("A", 1)],
            products=[("P", 1)],
            modifiers=["X"],
            law=_times("k", "A", "X", "X", "C"),
            params=[("k", k), ("q", 0.1)],
            rules=rules,
        )
    )
    names = list(m.species_names)
    x = np.zeros(len(names))
    x[names.index("A")] = 0.1
    x[names.index("X")] = 0.1
    # J is the last reaction: a rate rule is emitted first, as its own ``[] -> X``
    assert m.propensities(x)[-1] == pytest.approx(k * 0.1 * 0.01 * v, rel=1e-12)


def test_a_conversion_factor_scales_the_change_not_the_pairs():
    """A conversionFactor on the product sends the reaction through the
    per-species-change emission; its propensity is still the pair count."""
    v, k = 10.0, 0.01
    s = _sbml(
        comps=[("C", v, True)],
        species=[("A", "C", 3, False, False), ("P", "C", 0, False, False)],
        reactants=[("A", 2)],
        products=[("P", 1)],
        law=_times("k", "A", "A", "C"),
        params=[("k", k), ("cfp", 3.0)],
    )
    sp = '<species id="P" compartment="C"'
    s = s.replace(sp, sp + ' conversionFactor="cfp"')
    m = bngsim.Model.from_sbml_string(s)
    for n in (1, 2, 3):
        got = m.propensities([n / v, 0.0])[0]
        assert got == pytest.approx(k * n * (n - 1) / v, rel=1e-14, abs=0.0), (n, got)
    # A cf other than 1 is refused under SSA (its noise is wrong); the pair
    # count is what this checks, so the run opts out of the refusal.
    r = bngsim.Simulator(m, method="ssa", strict_ssa=False).run(
        t_span=(0, 1e4), n_points=2, seed=2
    )
    a = np.asarray(r.species)[-1, list(r.species_names).index("A")] * v
    assert a == 1.0  # one A left, which cannot pair


def test_the_writesbml_shape_with_an_assigned_rate_constant():
    """BNG's writeSBML emits ``2A -> P`` as ``0.5·rl·A·A`` with ``rl`` an
    assignment rule. The rule keeps the law Functional; A still pairs."""
    k = 0.01
    rules = (
        f'<listOfRules><assignmentRule variable="rl"><math {MATH}><ci>k</ci></math>'
        "</assignmentRule></listOfRules>"
    )
    m = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", 1.0, True)],
            species=[("A", "C", 3, False, False), ("P", "C", 0, False, False)],
            reactants=[("A", 2)],
            products=[("P", 1)],
            law=_times("<cn>0.5</cn>", "rl", "A", "A"),
            params=[("k", k), ("rl", 0.0, False)],
            rules=rules,
        )
    )
    names = list(m.species_names)
    for n in (1, 2, 3):
        x = np.zeros(len(names))
        x[names.index("A")] = n
        got = m.propensities(x)[0]
        assert got == pytest.approx(0.5 * k * n * (n - 1), rel=1e-14, abs=0.0), (n, got)


def test_psa_takes_the_same_propensity():
    m = _boundary_dimer(nb=1)
    r = bngsim.Simulator(m, method="psa", poplevel=100).run(t_span=(0, 1e4), n_points=3, seed=1)
    assert np.asarray(r.species)[-1, list(r.species_names).index("P")] == 0.0


def test_the_codegen_key_of_other_models_is_unchanged(tmp_path):
    """The fields are exported only when set: the structural codegen key
    hashes the whole dict, so every other model keeps its cached kernel."""
    rx = _boundary_dimer()._core.codegen_data()["reactions"]
    assert rx[0]["ssa_falling_factorial"] == [(0, 2)]
    net = tmp_path / "m.net"
    net.write_text(
        "begin parameters\n    1 k 1\nend parameters\n"
        "begin species\n    1 A() 10\nend species\n"
        "begin reactions\n    1 1,1 0 k\nend reactions\n"
    )
    cd = bngsim.Model.from_net(str(net))._core.codegen_data()
    assert "ssa_falling_factorial" not in cd["reactions"][0]
    assert "ssa_volume_param_idx0" not in cd["reactions"][0]
    assert "initial_amount" not in cd["species"][0]


# ── the builder setter ──────────────────────────────────────────────────────


def _builder(rtype):
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    b.add_species("A", 5.0)
    if rtype == "functional":
        b.add_function("f", "k*A*A")
        b.add_reaction([0, 0], [], "functional", "f", apply_species_factor=False)
    else:
        b.add_reaction([0, 0], [], "elementary", "k")
    return b


def test_the_setter_refuses_an_elementary_reaction():
    """An elementary reaction takes its falling factorial from its reactants,
    and the compiled SSA kernel never reads this field: setting it would make
    the two SSA backends disagree."""
    with pytest.raises(ValueError, match="not Functional"):
        _builder("elementary").set_reaction_ssa_falling_factorial(0, [(0, 2)])


def test_the_setter_refuses_a_species_listed_twice():
    with pytest.raises(ValueError, match="listed twice"):
        _builder("functional").set_reaction_ssa_falling_factorial(0, [(0, 2), (0, 2)])


# ── make_subset_model carries what the engine reads ─────────────────────────


def test_a_subset_keeps_the_falling_factorial():
    from bngsim.coupling import make_subset_model

    v = 10.0
    full = _boundary_dimer(nb=1)
    sub = make_subset_model(full, keep_reactions=[0])
    for n in (1, 2, 5):
        x = [n / v, 0.0]
        assert sub.propensities(x)[0] == full.propensities(x)[0], n
    r = bngsim.Simulator(sub, method="ssa").run(t_span=(0, 1e4), n_points=2, seed=1)
    assert np.asarray(r.species)[-1, list(r.species_names).index("P")] == 0.0


def test_a_subset_follows_a_compartment_write():
    """An initialAmount is stored as amount/V. A write to V re-divides it in
    the model; the subset kept the load-time amount/V (A = 2, not 1)."""
    from bngsim.coupling import make_subset_model

    full = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", 5.0, True)],
            species=[("A", "C", 10, False, False), ("P", "C", 0, False, False)],
            reactants=[("A", 1)],
            products=[("P", 1)],
            law=_times("k", "A", "C"),
            params=[("k", 0.1)],
        )
    )
    sub = make_subset_model(full)
    assert sub.compartment_size_params == full.compartment_size_params == ["C"]
    for mm in (full, sub):
        mm.set_param("C", 10.0)
        mm.reset()
    np.testing.assert_array_equal(sub.get_state(), full.get_state())
    np.testing.assert_array_equal(sub.get_state(), [1.0, 0.0])


def test_a_subset_keeps_a_refused_compartment_write():
    """``A(C1) + B(C2)`` in two compartments that happen to share a size is
    one mass-action scalar, exact only while the sizes agree, so the model
    refuses a write to either. The subset accepted it and ran on."""
    from bngsim.coupling import make_subset_model

    full = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C1", 2.0, True), ("C2", 2.0, True)],
            species=[
                ("A", "C1", 10, False, False),
                ("B", "C2", 10, False, False),
                ("P", "C1", 0, False, False),
            ],
            reactants=[("A", 1), ("B", 1)],
            products=[("P", 1)],
            law=_times("k", "A", "B", "C1"),
            params=[("k", 0.1)],
        )
    )
    sub = make_subset_model(full)
    assert sub.unwritable_compartment_size_params == full.unwritable_compartment_size_params
    for mm in (full, sub):
        with pytest.raises(ValueError, match="compartment size"):
            mm.set_param("C1", 4.0)


def test_a_subset_follows_an_initial_value_parameter(tmp_path):
    """``A() A0``: a write to A0 moves A's initial value (#79). The subset
    dropped the reference and kept A = 50."""
    from bngsim.coupling import make_subset_model

    net = tmp_path / "ic.net"
    net.write_text(
        "begin parameters\n    1 k 0.1\n    2 A0 50\nend parameters\n"
        "begin species\n    1 A() A0\n    2 B() 0\nend species\n"
        "begin reactions\n    1 1 2 k\nend reactions\n"
    )
    full = bngsim.Model.from_net(str(net))
    sub = make_subset_model(full)
    for mm in (full, sub):
        mm.set_param("A0", 80.0)
        mm.reset()
    np.testing.assert_array_equal(sub.get_state(), [80.0, 0.0])


def test_a_subset_keeps_a_saved_baseline(tmp_path):
    """After ``save_concentrations()`` the model resets to the saved state and a
    write no longer re-derives initial values. The subset re-resolved ``A0``."""
    from bngsim.coupling import make_subset_model

    net = tmp_path / "ic.net"
    net.write_text(
        "begin parameters\n    1 k 0.1\n    2 A0 50\nend parameters\n"
        "begin species\n    1 A() A0\n    2 B() 0\nend species\n"
        "begin reactions\n    1 1 2 k\nend reactions\n"
    )
    full = bngsim.Model.from_net(str(net))
    bngsim.Simulator(full, method="ode").run(t_span=(0, 5), n_points=2)
    full.save_concentrations()
    sub = make_subset_model(full)
    for mm in (full, sub):
        mm.set_param("A0", 80.0)
        mm.reset()
    np.testing.assert_array_equal(sub.get_state(), full.get_state())
    assert sub.get_state().sum() == pytest.approx(50.0, rel=1e-12)


def test_a_subset_keeps_the_loaders_refusals():
    """A ``fast="true"`` reaction is refused under ODE; the subset ran it as an
    ordinary reaction."""
    from bngsim.coupling import make_subset_model

    s = _sbml(
        comps=[("C", 1.0, True)],
        species=[("A", "C", 10, False, False), ("B", "C", 0, False, False)],
        reactants=[("A", 1)],
        products=[("B", 1)],
        law=_times("k", "A"),
        params=[("k", 0.5)],
    )
    s = s.replace(
        'level3/version2/core" level="3" version="2"',
        'level3/version1/core" level="3" version="1"',
    )
    s = s.replace(
        '<reaction id="J" reversible="false"', '<reaction id="J" reversible="false" fast="true"'
    )
    sub = make_subset_model(bngsim.Model.from_sbml_string(s))
    with pytest.raises(bngsim.ModelError, match="fast"):
        bngsim.Simulator(sub, method="ode").run(t_span=(0, 1), n_points=2)


def test_the_codegen_key_ignores_a_declared_amount():
    """``initial_amount`` is exported for make_subset_model. As a value it
    would give every initial condition its own compiled kernel."""
    from bngsim._codegen import compute_model_codegen_hash

    def mk(n0):
        return bngsim.Model.from_sbml_string(
            _sbml(
                comps=[("C", 5.0, True)],
                species=[("A", "C", n0, False, False), ("P", "C", 0, False, False)],
                reactants=[("A", 1)],
                products=[("P", 1)],
                law=_times("k", "A", "C"),
                params=[("k", 0.1)],
            )
        )

    a, b = mk(10), mk(20)
    assert a._core.codegen_data()["species"][0]["initial_amount"] == 10.0
    assert compute_model_codegen_hash(a) == compute_model_codegen_hash(b)


@pytest.mark.parametrize(
    "exp", ['<cn type="integer">1000000</cn>', "<cn>3e9</cn>"], ids=["1e6", "3e9"]
)
@pytest.mark.parametrize("boundary", [True, False], ids=["boundary", "floating"])
def test_a_huge_power_loads_and_evaluates(exp, boundary):
    """``B^3e9`` failed the load (an int the engine cannot hold) or hung it
    (the mass-action classifier expanded the power into that many factors);
    ``B^1e6`` loaded but one propensity took a million multiplications."""
    m = bngsim.Model.from_sbml_string(
        _sbml(
            comps=[("C", 10.0, True)],
            species=[("B", "C", 3, False, boundary), ("P", "C", 0, False, False)],
            reactants=[("B", 2)],
            products=[("P", 1)],
            law=_times("k", _pow("B", exp), "C"),
            params=[("k", 0.01)],
        )
    )
    assert m.propensities([0.3, 0.0])[0] == 0.0
