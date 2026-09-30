"""The SSA counts molecules, whatever units a species is stored in (issue #692).

bngsim stores an SBML species as ``amount/V``. The SSA's falling factorial for a
repeated reactant was taken on that stored value, so ``2A -> B`` in a
compartment of size ``V != 1`` fired at ``k·n(n − V)/V`` instead of
``k·n(n − 1)/V``: it stalled at ``V`` molecules when ``V > 1``, fired in reverse
below them, and dimerised a lone molecule forever when ``V < 1``. Every firing
also moved the stored value by an inexact ``±1/V``, so counts walked off whole
numbers (an extinct species ended at -1.9e-7 molecules after 1e5 firings at
``V = 10``). The compiled propensity kernel baked an amount-valued species'
volume in as a literal, so a compartment write after the Simulator was built
left it firing at ``(V_old/V_new)^n`` times the right rate (issue #723).

Oracles, none of which share code with the SSA:

* the exact chemical master equation for ``2A -> B`` (a generator matrix
  propagated with ``scipy.linalg.expm``);
* the closed form ``E[n(t)] = n0·e^{-kt}`` of a pure-death process;
* hand-computed propensities ``k·n(n − 1)/V`` and ``k·n``.

Both propensity backends are run: the compiled ``cc`` kernel and the
interpreted path (``codegen=False``). The kernel is also evaluated directly
through ``ctypes`` and held against the interpreted ``Model.propensities``.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import warnings

import bngsim
import numpy as np
import pytest
import scipy.linalg as sla
from bngsim._bngsim_core import emit_ssa_propensity_source_structure as emit
from bngsim._codegen import prepare_ssa_propensity_lib

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc") or shutil.which("cl")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

BACKENDS = [
    pytest.param({}, id="cc", marks=needs_cc),
    pytest.param({"codegen": False}, id="interpreted"),
]


def _sbml(*, v, species, law, reactants, products, params, events="", rules="", constant=True):
    """A one-compartment L3v2 model with a single irreversible reaction."""
    sp = "".join(
        f'<species id="{sid}" compartment="C" initialAmount="{n0}" '
        f'hasOnlySubstanceUnits="{"true" if hosu else "false"}" '
        f'boundaryCondition="false" constant="false"/>'
        for sid, n0, hosu in species
    )
    pr = "".join(f'<parameter id="{k}" value="{val}" constant="true"/>' for k, val in params)

    def refs(items):
        return "".join(
            f'<speciesReference species="{s}" stoichiometry="{st}" constant="true"/>'
            for s, st in items
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="m">
<listOfCompartments>
<compartment id="C" spatialDimensions="3" size="{v}" constant="{str(constant).lower()}"/>
</listOfCompartments>
<listOfSpecies>{sp}</listOfSpecies>
<listOfParameters>{pr}</listOfParameters>
{rules}
<listOfReactions><reaction id="J1" reversible="false">
<listOfReactants>{refs(reactants)}</listOfReactants>
<listOfProducts>{refs(products)}</listOfProducts>
<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">{law}</math></kineticLaw>
</reaction></listOfReactions>
{events}
</model>
</sbml>"""


_TIMES = "<times/>"


def _dimer(v, n0, k, *, hosu=False, stoich=2):
    """``2A -> B``. hOSU=false: the concentration law ``k·[A]²·C``; hOSU=true:
    the amount law ``k·A²``. Either way the SSA propensity is ``k·n(n−1)/V``
    and ``k·n(n−1)`` respectively, n the count of A."""
    a = "<ci>A</ci>" * stoich
    law = (
        f"<apply>{_TIMES}<ci>k</ci>{a}</apply>"
        if hosu
        else f"<apply>{_TIMES}<ci>k</ci>{a}<ci>C</ci></apply>"
    )
    return bngsim.Model.from_sbml_string(
        _sbml(
            v=v,
            species=[("A", n0, hosu), ("B", 0, hosu)],
            law=law,
            reactants=[("A", stoich)],
            products=[("B", 1)],
            params=[("k", k)],
        )
    )


def _cme_mean(prop, n0, t):
    """E[n_A(t)] for ``2A -> B`` from the exact master equation."""
    q = np.zeros((n0 + 1, n0 + 1))
    for n in range(2, n0 + 1):
        q[n, n] -= prop(n)
        q[n, n - 2] += prop(n)
    p0 = np.zeros(n0 + 1)
    p0[n0] = 1.0
    return float((p0 @ sla.expm(q * t)) @ np.arange(n0 + 1))


def _counts(result, v):
    return np.asarray(result.species) * v


# ── The propensity itself ────────────────────────────────────────────────────


@pytest.mark.parametrize("v", [10.0, 0.5, 2.3])
def test_repeated_reactant_propensity_is_the_count_falling_factorial(v):
    """``2A -> B`` fires at ``k·n(n−1)/V``, so it is exactly 0 at one molecule
    and never negative. The old value, ``k·n(n−V)/V``, was 0.2 instead of 0.38
    at n = 20, V = 10, and negative at 5 molecules."""
    k = 0.01
    m = _dimer(v, 20, k)
    for n in (20, 5, 2, 1, 0):
        got = m.propensities([n / v, 0.0])[0]
        want = k * n * (n - 1) / v
        assert got == pytest.approx(want, rel=1e-14, abs=0.0), (v, n, got, want)
    # One molecule: exactly zero, not an ulp either side of it.
    assert m.propensities([1 / v, 0.0])[0] == 0.0


def test_amount_valued_repeated_reactant_propensity():
    """The hOSU=true species was already counted by amount; the same terms in
    the new form must still give ``k·n(n−1)`` and 0 at one molecule."""
    k, v = 0.01, 10.0
    m = _dimer(v, 20, k, hosu=True)
    for n in (20, 3, 1):
        got = m.propensities([n / v, 0.0])[0]
        assert got == pytest.approx(k * n * (n - 1), rel=1e-14, abs=0.0), (n, got)
    assert m.propensities([1 / v, 0.0])[0] == 0.0


def test_net_source_is_unchanged(tmp_path):
    """A ``.net`` model has volume 1 and no volume parameter, so its emitted
    kernel keeps the pre-#692 text byte for byte — and with it every cached
    ``.so`` keyed on that text."""
    net = tmp_path / "dimer.net"
    net.write_text(
        "begin parameters\n 1 k 0.01\nend parameters\n"
        "begin species\n 1 A() 20\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1,1 2 k\nend reactions\n"
        "begin groups\n 1 Atot 1\nend groups\n"
    )
    m = bngsim.Model.from_net(str(net))
    src, n_unsupported = emit(m._core)
    assert n_unsupported == 0
    assert "x[0] * (x[0] - 1);" in src, src
    assert "/ p[" not in src


def test_the_kernel_reads_the_species_volume_at_runtime():
    """The falling-factorial offset and an amount-valued species' volume are
    ``p[]`` reads of the compartment size, never literals that go stale on a
    write (#723)."""
    for hosu in (False, True):
        m = _dimer(10.0, 20, 0.01, hosu=hosu)
        src, n_unsupported = emit(m._core)
        assert n_unsupported == 0
        c = list(m.param_names).index("C")
        assert f"(x[0] - 1.0 / p[{c}])" in src, src
        if hosu:
            assert f"(x[0] * p[{c}])" in src, src
        # No species read is scaled by a literal (the old `(10 * x[0])`).
        assert re.search(r"[0-9.] \* x\[", src) is None, src


# ── The compiled kernel against the interpreted path ─────────────────────────


def _kernel(model):
    so = prepare_ssa_propensity_lib(model)
    assert so, "expected a compiled SSA propensity library"
    fn = ctypes.CDLL(so).bngsim_ssa_propensities
    dp = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [dp, dp, dp]
    fn.restype = None

    def evaluate(x):
        x = np.ascontiguousarray(x, dtype=float)
        p = np.array([q["value"] for q in model._core.codegen_data()["parameters"]], dtype=float)
        a = np.zeros(model.n_reactions)
        fn(x.ctypes.data_as(dp), p.ctypes.data_as(dp), a.ctypes.data_as(dp))
        return a

    return evaluate


@needs_cc
@pytest.mark.parametrize("hosu", [False, True], ids=["hosu_false", "hosu_true"])
def test_kernel_matches_interpreted_across_a_compartment_write(hosu):
    """The kernel is compiled at V = 10 and then evaluated after a write to
    V = 4: it must agree with the interpreted propensity at the new size, and
    both must be exactly 0 at one molecule. Before #723 the hOSU=true kernel
    kept ``10 * x`` and fired 2.5² = 6.25 times too fast."""
    m = _dimer(10.0, 20, 0.01, hosu=hosu)
    kernel = _kernel(m)
    m.set_param("C", 4.0)
    for n in (20, 7, 2, 1, 0):
        x = [n / 4.0, 0.0]
        want = 0.01 * n * (n - 1) * (1.0 if hosu else 1 / 4.0)
        got_i = m.propensities(x)[0]
        got_k = kernel(x)[0]
        assert got_i == pytest.approx(want, rel=1e-14, abs=0.0), (n, got_i, want)
        assert got_k == pytest.approx(got_i, rel=1e-14, abs=0.0), (n, got_k, got_i)
    assert kernel([1 / 4.0, 0.0])[0] == 0.0


# ── Ensembles against the master equation ───────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("v", "k", "hosu"),
    [(10.0, 0.01, False), (0.5, 0.0005, False), (1.0, 0.001, False), (10.0, 0.001, True)],
    ids=["V10", "V0.5", "V1-control", "V10-hosu"],
)
def test_dimerisation_mean_matches_the_master_equation(backend, v, k, hosu):
    """E[n_A(T)] for ``2A -> B`` from 20 molecules, against the exact CME. At
    V = 10 bngsim gave 12.17 against 6.87 (140 standard errors); V = 1 is the
    control that was always right."""
    n0, t_end, reps = 20, 50.0, 2000
    m = _dimer(v, n0, k, hosu=hosu)
    per_pair = k if hosu else k / v
    want = _cme_mean(lambda n: per_pair * n * (n - 1), n0, t_end)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no boundary warning may fire
        r = bngsim.Simulator(m, method="ssa", **backend).run_replicates(
            reps, t_span=(0, t_end), n_points=2, seed=7, squeeze=True
        )
    if backend:
        assert r.ssa_diagnostics["propensity_backend"] == "interpreted"
    n_a = np.asarray(r.species)[:, -1, 0] * v
    assert np.array_equal(n_a, np.round(n_a)), "counts must stay whole numbers"
    assert n_a.min() >= 0
    se = n_a.std(ddof=1) / np.sqrt(reps)
    assert abs(n_a.mean() - want) <= 4.5 * se, (n_a.mean(), want, se)


def test_below_v_molecules_no_reaction_runs_backwards():
    """At V = 10 and 5 molecules the old propensity was -0.025, so the
    dimerisation ran in reverse and B went to -3. Now at most two pairs form
    and nothing goes negative."""
    m = _dimer(10.0, 5, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 2000), n_points=3, seed=1)
    n = _counts(r, 10.0)
    assert (n >= 0).all(), n
    assert n[-1].tolist() == [1.0, 2.0], n


def test_a_lone_molecule_does_not_dimerise():
    """At V = 0.5 one molecule had a positive propensity, dimerised, and the
    run never returned. Its propensity is 0: nothing fires."""
    m = _dimer(0.5, 1, 1.0)
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 10), n_points=3, seed=1, timeout=5.0)
    assert _counts(r, 0.5).tolist() == [[1.0, 0.0]] * 3


# ── Counts stay whole numbers ────────────────────────────────────────────────


def _decay(v, n0, k=1.0, *, hosu=False):
    law = (
        f"<apply>{_TIMES}<ci>k</ci><ci>A</ci></apply>"
        if hosu
        else f"<apply>{_TIMES}<ci>k</ci><ci>A</ci><ci>C</ci></apply>"
    )
    return bngsim.Model.from_sbml_string(
        _sbml(
            v=v,
            species=[("A", n0, hosu), ("B", 0, hosu)],
            law=law,
            reactants=[("A", 1)],
            products=[("B", 1)],
            params=[("k", k)],
        )
    )


@pytest.mark.parametrize("v", [10.0, 3.0, 0.7])
@pytest.mark.parametrize(
    ("method", "kw"),
    [
        pytest.param("ssa", {}, id="ssa-cc", marks=needs_cc),
        pytest.param("ssa", {"codegen": False}, id="ssa-interpreted"),
        pytest.param("psa", {"poplevel": 100}, id="psa"),
    ],
)
def test_an_extinct_species_holds_exactly_zero(v, method, kw):
    """``A -> B`` from 1e5 molecules to extinction. Each firing used to add an
    inexact ±1/V to the stored value, and the rounding walked: A ended at
    -1.9e-7 molecules at V = 10, which the negative-count diagnostic reported,
    and a residue of the other sign would have left A a propensity to fire on.
    A count is now kept on whole numbers, so A ends at exactly 0 and B at
    exactly 1e5, with no warning."""
    n0 = 100_000
    m = _decay(v, n0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = bngsim.Simulator(m, method=method, **kw).run(t_span=(0, 40), n_points=5, seed=3)
    n = _counts(r, v)
    assert np.all(np.abs(n - np.round(n)) <= 1e-9 * np.maximum(1.0, np.abs(n))), n
    assert r.species[-1][0] == 0.0 and not np.signbit(r.species[-1][0])
    assert np.round(n[-1, 1]) == n0
    assert np.all(np.round(n).sum(axis=1) == n0)


# ── A compartment write on a Simulator that already exists (#723) ─────────────


@needs_cc
def test_existing_simulator_follows_a_compartment_write():
    """An hOSU=true ``A -> ∅`` at ``k·A`` (amounts) decays at k whatever the
    volume: E[n(1)] = 1000·e^{-1} = 367.88. The Simulator is built at V = 2
    and run after a write to V = 4. Before #723 its compiled kernel kept the
    literal 2 and the same Simulator gave 605.5 = 1000·e^{-1/2}."""
    m = bngsim.Model.from_sbml_string(
        _sbml(
            v=2.0,
            species=[("A", 1000, True)],
            law=f"<apply>{_TIMES}<ci>k</ci><ci>A</ci></apply>",
            reactants=[("A", 1)],
            products=[],
            params=[("k", 1.0)],
        )
    )
    sim = bngsim.Simulator(m, method="ssa")
    m.set_param("C", 4.0)
    m.reset()
    reps = 400
    r = sim.run_replicates(reps, t_span=(0, 1), n_points=2, seed=11, squeeze=True)
    assert r.ssa_diagnostics["propensity_backend"] == "cc"
    n = np.asarray(r.species)[:, -1, 0] * 4.0
    want = 1000 * np.exp(-1.0)
    se = n.std(ddof=1) / np.sqrt(reps)
    assert abs(n.mean() - want) <= 4.5 * se, (n.mean(), want, se)


# ── Review additions ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("hosu", [False, True], ids=["hosu_false", "hosu_true"])
def test_three_copies_of_a_reactant(hosu):
    """``3A -> B`` fires at ``k·n(n−1)(n−2)/V²`` (hOSU=false) or
    ``k·n(n−1)(n−2)`` (amounts), exactly 0 at one and two molecules."""
    k, v = 1e-4, 10.0
    m = _dimer(v, 30, k, hosu=hosu, stoich=3)
    scale = 1.0 if hosu else 1 / v**2
    for n in (30, 5, 3, 2, 1, 0):
        got = m.propensities([n / v, 0.0])[0]
        want = k * n * (n - 1) * (n - 2) * scale
        assert got == pytest.approx(want, rel=1e-14, abs=0.0), (n, got, want)
    assert m.propensities([2 / v, 0.0])[0] == 0.0
    assert m.propensities([1 / v, 0.0])[0] == 0.0


@needs_cc
def test_a_literal_volume_in_the_kernel_is_a_float_division():
    """A compartment set by an assignment rule has no volume parameter, so the
    kernel keeps its size as a literal. ``1 / 10`` would be integer 0 in C; the
    offset must be emitted as ``1.0 / 10`` and agree with the interpreted path."""
    mathml = 'xmlns="http://www.w3.org/1998/Math/MathML"'
    rules = (
        f'<listOfRules><assignmentRule variable="C"><math {mathml}><ci>Vp</ci></math>'
        "</assignmentRule></listOfRules>"
    )
    m = bngsim.Model.from_sbml_string(
        _sbml(
            v=1.0,
            species=[("A", 20, True), ("B", 0, True)],
            law=f"<apply>{_TIMES}<ci>k</ci><ci>A</ci><ci>A</ci></apply>",
            reactants=[("A", 2)],
            products=[("B", 1)],
            params=[("k", 0.01), ("Vp", 10.0)],
            rules=rules,
            constant=False,
        )
    )
    src, n_unsupported = emit(m._core)
    assert n_unsupported == 0, src
    assert "(x[0] - 1.0 / 10)" in src, src
    kernel = _kernel(m)
    for n in (20, 2, 1):
        x = [n / 10.0, 0.0]
        assert kernel(x)[0] == pytest.approx(m.propensities(x)[0], rel=1e-14, abs=0.0)
        assert m.propensities(x)[0] == pytest.approx(0.01 * n * (n - 1), rel=1e-14, abs=0.0)
    assert kernel([1 / 10.0, 0.0])[0] == 0.0


@pytest.mark.parametrize(
    ("v", "n0", "poplevel"),
    [(10.0, 99, 1.01), (1.0, 98, 2.0), (10.0, 98, 2.0)],
)
def test_psa_leaps_keep_counts_whole(v, n0, poplevel):
    """A PSA leap moved a count by 1/(1/m), which is not m for many m
    (1/(1/98) = 98.00000000000001), so a species leapt to extinction ended at
    -1e-14 and tripped the negative-count warning. The leap is m itself."""
    m = _decay(v, n0, hosu=True)
    for seed in range(5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = bngsim.Simulator(m, method="psa", poplevel=poplevel).run(
                t_span=(0, 40), n_points=9, seed=seed
            )
        n = _counts(r, v)
        assert np.array_equal(n, np.round(n)), (seed, n)
        assert n[-1, 0] == 0.0, (seed, n[-1])
        m.reset()


def test_psa_leap_to_extinction_in_a_net_model(tmp_path):
    """The `.net` form of the same defect: ``A -> 0`` from 98 at poplevel 2
    leapt by 49.00000000000001 and ended at -7.1e-15."""
    net = tmp_path / "decay.net"
    net.write_text(
        "begin parameters\n 1 k 1\nend parameters\nbegin species\n 1 A() 98\nend species\n"
        "begin reactions\n 1 1 0 k\nend reactions\nbegin groups\n 1 Atot 1\nend groups\n"
    )
    m = bngsim.Model.from_net(str(net))
    for seed in range(5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = bngsim.Simulator(m, method="psa", poplevel=2).run(
                t_span=(0, 60), n_points=7, seed=seed
            )
        a = np.asarray(r.species)[:, 0]
        assert np.array_equal(a, np.round(a)) and a[-1] == 0.0, (seed, a)
        m.reset()


def test_a_fractional_count_an_event_assigns_is_carried():
    """A count kept on whole numbers must not round one an event set to a
    fraction: 2.5 molecules decay through 1.5 and 0.5."""
    mathml = 'xmlns="http://www.w3.org/1998/Math/MathML"'
    events = (
        '<listOfEvents><event id="E" useValuesFromTriggerTime="true"><trigger '
        f'initialValue="false" persistent="true"><math {mathml}><apply><geq/><csymbol '
        'encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
        "<cn>1</cn></apply></math></trigger><listOfEventAssignments><eventAssignment "
        f'variable="A"><math {mathml}><cn>0.25</cn></math></eventAssignment>'
        "</listOfEventAssignments></event></listOfEvents>"
    )
    m = bngsim.Model.from_sbml_string(
        _sbml(
            v=10.0,
            species=[("A", 0, False), ("B", 0, False)],
            law=f"<apply>{_TIMES}<ci>k</ci><ci>A</ci><ci>C</ci></apply>",
            reactants=[("A", 1)],
            products=[("B", 1)],
            params=[("k", 1.0)],
            events=events,
        )
    )
    seen = set()
    for seed in range(20):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
            r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 3), n_points=31, seed=seed)
        a = _counts(r, 10.0)[:, 0]
        after = a[np.asarray(r.time) >= 1.0]
        seen.update(after.tolist())
        m.reset()
    assert 2.5 in seen and 1.5 in seen, seen
    assert all(x - np.floor(x) == 0.5 for x in seen), seen


@needs_cc
def test_existing_simulator_dimerisation_follows_a_compartment_write():
    """The second-order case of the write test: built at V = 10, run at V = 4,
    against the master equation at V = 4."""
    k, n0, t_end, reps = 0.01, 20, 20.0, 2000
    m = _dimer(10.0, n0, k)
    sim = bngsim.Simulator(m, method="ssa")
    m.set_param("C", 4.0)
    m.reset()
    r = sim.run_replicates(reps, t_span=(0, t_end), n_points=2, seed=13, squeeze=True)
    assert r.ssa_diagnostics["propensity_backend"] == "cc"
    n_a = np.asarray(r.species)[:, -1, 0] * 4.0
    want = _cme_mean(lambda n: k / 4.0 * n * (n - 1), n0, t_end)
    se = n_a.std(ddof=1) / np.sqrt(reps)
    assert abs(n_a.mean() - want) <= 4.5 * se, (n_a.mean(), want, se)
