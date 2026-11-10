"""The evidence behind rr_parity's NO_ORACLE_ADJUDICATED ``confirm`` entries (#1559).

A ``confirm`` entry says bngsim's run of a model RoadRunner refuses is right. Its
earlier evidence cited "GH #117", an unrelated issue here, and two files that were
never in this repository, and a third entry from the same set was wrong (#1542,
#1547). Every ``confirm`` entry now rests on a run of ``scipy_oracle``, which builds
the model's right-hand side from the SBML text and integrates it with scipy. That
shares no code with bngsim or SUNDIALS. For each entry, this file checks:

* the oracle's right-hand side is RoadRunner's, at the start state and at a random
  one (RoadRunner refuses to integrate these models, not to read them);
* two scipy integrators agree on the trajectory;
* bngsim, run as ``rr_run`` runs it, passes against the oracle under the suite's
  own verdict, on every species, at the precision the entry's reason quotes;
* no species below the suite's atol has a positive ``df_i/dx_i``, which is how
  #1542's engines held a species on its repelling equilibrium and passed each
  other on a trajectory that was not a solution.

The model files are the gitignored rr_parity corpus, so those tests skip where it
is absent. The oracle's own tests need only scipy and libsbml.
"""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path

import numpy as np
import pytest

libsbml = pytest.importorskip("libsbml")

import scipy_oracle as so  # noqa: E402
from _core import differ, read_manifest  # noqa: E402

_RR = Path(__file__).resolve().parent.parent / "rr_parity"


def _load_rr_overrides():
    """rr_parity's overrides.py, by path: bng_parity has an ``overrides`` module too,
    and a bare import resolves to whichever directory reached ``sys.path`` first."""
    spec = importlib.util.spec_from_file_location("rr_parity_overrides_1559", _RR / "overrides.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ov = _load_rr_overrides()

# How the oracle integrates each confirmed model. Two scipy integrators per model,
# so the trajectory is not one method's artifact. MODEL2001200002 runs in log
# coordinates: its species decay to 1e-34..1e-42, where a linear-coordinate trial
# step can go negative and make (L/T)^1.36 NaN, and x = exp(u) cannot.
ORACLE_RUNS = {
    "BIOMD0000000250:ode": {
        "coords": "linear",
        "runs": [("Radau", 1e-11, 1e-25), ("BDF", 1e-11, 1e-25)],
    },
    "MODEL2001200002:ode": {
        "coords": "log",
        "runs": [("DOP853", 1e-12, 1e-12), ("BDF", 1e-10, 1e-10)],
    },
}

# What "bngsim confirmed" is held to, beyond the suite's verdict, which forgives
# more than an entry's reason claims: every species compared, an err/movement the
# reasons' 2.9e-9 and 1.7e-8 sit far inside, and every species within this
# fraction of its own peak (the worst is BIOMD0000000250's pCREB_n at 2.7e-6). A
# one-percent error fails it wherever a species is above a thousandth of its peak,
# in any species whose peak is above the suite's atol (23 of BIOMD0000000250's 47
# never rise above 4.3e-30, numerically 0, and a relative error there is not
# looked for).
MAX_ERR_OVER_MOVEMENT = 1e-6
MAX_ERR_OF_PEAK = 1e-5


def _confirm_keys() -> set[str]:
    return {k for k, v in ov.NO_ORACLE_ADJUDICATED.items() if v["verdict"] == "confirm"}


@cache
def _jobs() -> dict:
    return {f"{j.model_id}:{j.method}": j for j in read_manifest(_RR / "ode_jobs.json")[1]}


def _job(key: str):
    return _jobs()[key]


def _model_xml(key: str) -> str:
    path = _RR / _job(key).model
    if not path.is_file():
        pytest.skip(f"rr_parity corpus model for {key} not present")
    return path.read_text()


def _oracle_runs(key: str, xml: str):
    p = _job(key).params
    t0 = float(p.get("initial_time", p["t_start"]))
    ode = so.SbmlOde(xml)
    spec = ORACLE_RUNS[key]
    runs = [
        ode.solve(t0, p["t_end"], p["n_points"], method=m, rtol=r, atol=a, coords=spec["coords"])
        for m, r, a in spec["runs"]
    ]
    return ode, runs


# --------------------------------------------------------------------------- #
# The entries
# --------------------------------------------------------------------------- #
def test_every_confirm_entry_is_settled_by_an_oracle_run_here():
    """A new ``confirm`` entry needs an ORACLE_RUNS row, so its evidence is a test."""
    assert _confirm_keys() == set(ORACLE_RUNS)


def test_no_entry_cites_evidence_that_is_not_in_this_repository():
    """#117 here is an unrelated issue, and dev/ never existed in this repository."""
    for key, entry in ov.NO_ORACLE_ADJUDICATED.items():
        assert entry["issue"] != "GH #117", key
        assert "dev/" not in entry["reason"], key
    source = (_RR / "overrides.py").read_text()
    assert "issue117_adjudication" not in source
    assert "oracle_117" not in source


def test_the_baked_manifest_carries_the_entries_as_written():
    """ode_jobs.json is a generated cache of overrides.py; the two must agree."""
    for key, entry in ov.NO_ORACLE_ADJUDICATED.items():
        (baked,) = [o for o in _job(key).overrides if o.field == "no_oracle_adjudicated"]
        assert baked.value == {"issue": entry["issue"], "verdict": entry["verdict"]}
        assert baked.reason == entry["reason"]


# --------------------------------------------------------------------------- #
# The oracle on its own
# --------------------------------------------------------------------------- #
_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">"""
_MATH = '<math xmlns="http://www.w3.org/1998/Math/MathML">{}</math>'
_LAW = "<apply><times/><ci>c</ci><apply><ci>f</ci><ci>k</ci><ci>A</ci></apply></apply>"


def _decay_sbml(
    extra: str = "", law: str = _LAW, local_k: str = "0.7", stoich_a: str = "1"
) -> str:
    """A -> B in a compartment of size 2, at k*A through a function definition
    with a local parameter shadowing a global one, and an assignment rule."""
    return (
        _HEAD
        + "<listOfFunctionDefinitions><functionDefinition id='f'>"
        + _MATH.format(
            "<lambda><bvar><ci>p</ci></bvar><bvar><ci>x</ci></bvar>"
            "<apply><times/><ci>p</ci><ci>x</ci></apply></lambda>"
        )
        + "</functionDefinition></listOfFunctionDefinitions>"
        + "<listOfCompartments><compartment id='c' size='2' constant='true'/></listOfCompartments>"
        + "<listOfSpecies>"
        + "<species id='A' compartment='c' initialConcentration='3' hasOnlySubstanceUnits='false' boundaryCondition='false' constant='false'/>"
        + "<species id='B' compartment='c' initialAmount='2' hasOnlySubstanceUnits='false' boundaryCondition='false' constant='false'/>"
        + "</listOfSpecies>"
        + "<listOfParameters><parameter id='k' value='99' constant='true'/>"
        + "<parameter id='tot' constant='false'/></listOfParameters>"
        + "<listOfRules><assignmentRule variable='tot'>"
        + _MATH.format("<apply><plus/><ci>A</ci><ci>B</ci></apply>")
        + "</assignmentRule></listOfRules>"
        + "<listOfReactions><reaction id='r' reversible='false'>"
        + f"<listOfReactants><speciesReference species='A' stoichiometry='{stoich_a}' constant='true'/></listOfReactants>"
        + "<listOfProducts><speciesReference species='B' stoichiometry='1' constant='true'/></listOfProducts>"
        + f"<kineticLaw>{_MATH.format(law)}"
        + f"<listOfLocalParameters><localParameter id='k' value='{local_k}'/></listOfLocalParameters>"
        + "</kineticLaw></reaction></listOfReactions>"
        + extra
        + "</model></sbml>"
    )


def _solve(xml: str, coords: str = "linear", method: str = "Radau"):
    return so.SbmlOde(xml).solve(
        0.0, 4.0, 41, method=method, rtol=1e-11, atol=1e-13, coords=coords
    )


@pytest.mark.parametrize(
    "coords,method", [("linear", "Radau"), ("linear", "DOP853"), ("log", "BDF")]
)
def test_the_oracle_reproduces_a_closed_form(coords, method):
    """[A] = 3 exp(-0.7 t) and [B] = 1 + 3 (1 - exp(-0.7 t)): the local k = 0.7, not
    the global 99; the rate in amount/time divided by the size 2; B's amount 2 read
    as concentration 1; the function definition expanded."""
    t, v, names = _solve(_decay_sbml(), coords, method)
    a = 3 * np.exp(-0.7 * t)
    assert names == ["A", "B"]
    np.testing.assert_allclose(v[:, 0], a, rtol=1e-8)
    np.testing.assert_allclose(v[:, 1], 1 + 3 - a, rtol=1e-8)


def test_functions_and_stoichiometry_reproduce_a_closed_form():
    """The law c * |-k| * root(3, A^3) * log10(10) * log_2(8)/3 * ln(e) * exp(ln 2)/2
    * pi/3.141592653589793 * (2 + -1) * E is c*k*A, with E a boundary species held
    at 1 that the reaction also consumes, and A's stoichiometry is 2, so
    [A] = 3 exp(-1.4 t) and [B] = 1 + (3 - [A]) / 2. A misread of any one of them (a
    log's base and argument swapped, root's degree and radicand swapped, exp read
    as a log, a wrong pi, a unary minus or abs dropped, the stoichiometry dropped,
    E integrated rather than held) moves [A]."""
    law = (
        "<apply><times/><ci>c</ci>"
        "<apply><abs/><apply><minus/><ci>k</ci></apply></apply>"
        "<apply><root/><degree><cn>3</cn></degree><apply><power/><ci>A</ci><cn>3</cn></apply></apply>"
        "<apply><log/><cn>10</cn></apply>"
        "<apply><divide/><apply><log/><logbase><cn>2</cn></logbase><cn>8</cn></apply><cn>3</cn></apply>"
        "<apply><ln/><exponentiale/></apply>"
        "<apply><divide/><apply><exp/><apply><ln/><cn>2</cn></apply></apply><cn>2</cn></apply>"
        "<apply><divide/><pi/><cn>3.141592653589793</cn></apply>"
        "<apply><plus/><cn>2</cn><apply><minus/><cn>1</cn></apply></apply>"
        "<ci>E</ci></apply>"
    )
    xml = (
        _decay_sbml(law=law, stoich_a="2")
        .replace(
            "</listOfSpecies>",
            "<species id='E' compartment='c' initialConcentration='1' hasOnlySubstanceUnits='false' "
            "boundaryCondition='true' constant='false'/></listOfSpecies>",
        )
        .replace(
            "</listOfReactants>",
            "<speciesReference species='E' stoichiometry='1' constant='true'/></listOfReactants>",
        )
    )
    t, v, names = _solve(xml)
    assert names == ["A", "B"]
    a = 3 * np.exp(-1.4 * t)
    np.testing.assert_allclose(v[:, 0], a, rtol=1e-8)
    np.testing.assert_allclose(v[:, 1], 1 + (3 - a) / 2, rtol=1e-8)


def test_a_negative_base_is_raised_as_written():
    """c * k^2 * A with k = -2 is 4*c*A, written with a negative local parameter or
    a negative number, not -(2^2): the bug was ``-2.0 ** 2.0``, which Python reads as
    -4. Under a fractional power the base gives NaN, as in RoadRunner, not a Python
    complex number."""
    square = (
        "<apply><times/><ci>c</ci><apply><power/><ci>k</ci><cn>2</cn></apply><ci>A</ci></apply>"
    )
    literal = (
        "<apply><times/><ci>c</ci><apply><power/><cn>-2</cn><cn>2</cn></apply><ci>A</ci></apply>"
    )
    for xml in (_decay_sbml(law=square, local_k="-2"), _decay_sbml(law=literal)):
        t, v, _ = _solve(xml)
        np.testing.assert_allclose(v[:, 0], 3 * np.exp(-4 * t), rtol=1e-8, atol=1e-12)
    root = square.replace("<cn>2</cn>", "<cn>1.5</cn>")
    with np.errstate(invalid="ignore"):
        rate = so.SbmlOde(_decay_sbml(law=root, local_k="-2")).rhs(0.0, [3.0, 1.0])
    assert rate.dtype == float and np.isnan(rate).all()


def test_a_species_set_by_a_rule_is_read_from_its_rule():
    """C := 2*A, with no start value of its own, and the law c * k * C / 2 = c*k*A:
    C is not a state, and [A] = 3 exp(-0.7 t)."""
    law = "<apply><divide/><apply><times/><ci>c</ci><ci>k</ci><ci>C</ci></apply><cn>2</cn></apply>"
    rule = (
        "<listOfSpecies><species id='C' compartment='c' hasOnlySubstanceUnits='false' "
        "boundaryCondition='false' constant='false'/></listOfSpecies>"
    )
    xml = (
        _decay_sbml(law=law)
        .replace("</listOfSpecies>", rule.replace("<listOfSpecies>", ""), 1)
        .replace(
            "</listOfProducts>",
            "</listOfProducts><listOfModifiers><modifierSpeciesReference species='C'/>"
            "</listOfModifiers>",
        )
        .replace(
            "</listOfRules>",
            "<assignmentRule variable='C'>"
            + _MATH.format("<apply><times/><cn>2</cn><ci>A</ci></apply>")
            + "</assignmentRule></listOfRules>",
        )
    )
    t, v, names = _solve(xml)
    assert names == ["A", "B"]
    np.testing.assert_allclose(v[:, 0], 3 * np.exp(-0.7 * t), rtol=1e-8)


_LEVEL_1 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level1" level="1" version="2"><model name="m">
<listOfCompartments><compartment name="c" volume="1"/></listOfCompartments>
<listOfSpecies><species name="A" compartment="c" initialAmount="3" boundaryCondition="false"/>
<species name="B" compartment="c" initialAmount="0" boundaryCondition="false"/></listOfSpecies>
<listOfParameters><parameter name="k" value="1"/></listOfParameters>
<listOfReactions><reaction name="r" reversible="false">
<listOfReactants><speciesReference species="A" stoichiometry="1" denominator="2"/></listOfReactants>
<listOfProducts><speciesReference species="B" stoichiometry="3" denominator="2"/></listOfProducts>
<kineticLaw formula="k*A"/></reaction></listOfReactions></model></sbml>"""


@pytest.mark.parametrize(
    "xml",
    [
        _decay_sbml(
            "<listOfEvents><event id='e' useValuesFromTriggerTime='true'><trigger initialValue='true' persistent='true'>"
            + _MATH.format("<apply><gt/><ci>A</ci><cn>1</cn></apply>")
            + "</trigger><listOfEventAssignments><eventAssignment variable='B'>"
            + _MATH.format("<cn>0</cn>")
            + "</eventAssignment></listOfEventAssignments></event></listOfEvents>"
        ),
        _decay_sbml(
            law="<piecewise><piece><ci>A</ci><apply><gt/><ci>A</ci><cn>1</cn></apply></piece>"
            "<otherwise><cn>0</cn></otherwise></piecewise>"
        ),
        _decay_sbml(law="<apply><sin/><ci>A</ci></apply>"),
        _decay_sbml().replace("value='0.7'", "value='INF'"),
        _decay_sbml().replace("<localParameter id='k' value='0.7'/>", "<localParameter id='k'/>"),
        _decay_sbml().replace(
            "species='A' stoichiometry='1' constant='true'", "species='A' constant='true'"
        ),
        _LEVEL_1,
    ],
    ids=[
        "event",
        "piecewise",
        "sin",
        "infinite-value",
        "local-parameter-without-value",
        "unset-stoichiometry",
        "level-1-denominator",
    ],
)
def test_the_oracle_refuses_what_it_does_not_read(xml):
    with pytest.raises(so.Unsupported):
        so.SbmlOde(xml)


def test_log_coordinates_refuse_a_species_that_starts_at_zero():
    xml = _decay_sbml().replace("initialConcentration='3'", "initialConcentration='0'")
    with pytest.raises(so.Unsupported):
        so.SbmlOde(xml).solve(0.0, 1.0, 3, method="BDF", rtol=1e-8, atol=1e-8, coords="log")


def test_self_rates_are_exact_at_any_scale():
    """The complex step against closed forms, where a finite difference small enough
    to stay near A = 1e-30 is lost to rounding beside B's O(1) rate: df_A/dA is -0.7
    at A = 1e-30 and at A = 3, and df_B/dB is 0. Under A^2.5 it is -2.5 A^1.5,
    -2.5e-45 at A = 1e-30, which a step not scaled to A misreads (-7.1e-31 for a
    step of 1e-20), and under A^1.1 at A = 1e-250 it is -1.1e-25, which even a
    step of 1e-200 misreads. Under |A| at A < 0 it is +1, which a complex step reads only
    through an analytic |A|. Under A^0.5 at A < 0 the right-hand side is NaN, and so
    is the rate."""
    ode = so.SbmlOde(_decay_sbml())
    for a in (1e-30, 3.0):
        assert ode.self_rates(0.0, [a, 1.0]) == pytest.approx([-0.7, 0.0], abs=1e-15)
    assert ode.self_rates(0.0, [1e-30, 1.0], which=[0]) == pytest.approx([-0.7], abs=1e-15)

    def law(inner):
        return so.SbmlOde(_decay_sbml(law=f"<apply><times/><ci>c</ci>{inner}</apply>"))

    power = law("<apply><power/><ci>A</ci><cn>2.5</cn></apply>")
    assert power.self_rates(0.0, [1e-30, 1.0], which=[0])[0] == pytest.approx(
        -2.5e-45, rel=1e-9, abs=0
    )
    # and at A = 1e-250, where no fixed step stays small beside A: -1.1 A^0.1.
    power = law("<apply><power/><ci>A</ci><cn>1.1</cn></apply>")
    assert power.self_rates(0.0, [1e-250, 1.0], which=[0])[0] == pytest.approx(
        -1.1e-25, rel=1e-9, abs=0
    )
    absolute = law("<apply><abs/><ci>A</ci></apply>")
    assert absolute.self_rates(0.0, [-1e-30, 1.0], which=[0])[0] == pytest.approx(1.0)
    root = law("<apply><power/><ci>A</ci><cn>0.5</cn></apply>")
    assert np.isnan(root.self_rates(0.0, [-1e-15, 1.0], which=[0])[0])


# --------------------------------------------------------------------------- #
# The oracle on each confirmed model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(ORACLE_RUNS))
def test_the_oracle_reads_the_model_as_roadrunner_does(key):
    """RoadRunner refuses to integrate these models, not to read them, so its
    right-hand side checks the oracle's reading of the SBML."""
    if importlib.util.find_spec("roadrunner") is None:
        pytest.skip("roadrunner is not installed")
    import roadrunner

    xml = _model_xml(key)
    ode = so.SbmlOde(xml)
    rr = roadrunner.RoadRunner(xml)
    ids = list(rr.model.getFloatingSpeciesIds())
    assert set(ode.names) <= set(ids)  # RoadRunner also lists species a rule sets
    rng = np.random.default_rng(1559)
    for y in (ode.y0, ode.y0 + rng.uniform(0.1, 50.0, ode.y0.size)):
        for name, value in zip(ode.names, y, strict=True):
            rr[f"[{name}]"] = value
        rates = dict(zip(ids, rr.model.getFloatingSpeciesConcentrationRates(), strict=True))
        theirs = np.array([rates[n] for n in ode.names])
        np.testing.assert_allclose(
            ode.rhs(0.0, y), theirs, rtol=1e-12, atol=1e-12 * np.abs(theirs).max()
        )


@pytest.mark.parametrize("key", sorted(ORACLE_RUNS))
def test_the_oracle_confirms_bngsim(key):
    """The entry's claim, re-derived: bngsim as rr_run runs it passes against the
    oracle on every species, at the precision its reason quotes, and the oracle's
    two integrators agree with each other."""
    pytest.importorskip("bngsim")
    import _rr_common as rc
    import rr_run

    xml = _model_xml(key)
    ode, (first, second) = _oracle_runs(key, xml)
    status, _, comment = rr_run._compare_ode(first, second)[:3]
    assert status == "pass", f"the oracle's two integrators disagree: {comment}"

    p = _job(key).params
    t0 = float(p.get("initial_time", p["t_start"]))
    bn = rc.bn_ode(xml, t0, p["t_end"], p["n_points"], rc.DEFAULT_RTOL, rc.DEFAULT_ATOL)[:3]
    for ref in (first, second):
        status, _, comment = rr_run._compare_ode(bn, ref)[:3]
        assert status == "pass", f"bngsim vs the scipy oracle: {comment}"
        bn_idx, ref_idx, common = rc.align_common(bn[2], ref[2])
        assert common == sorted(ode.names)  # every species, not just some
        a, b = bn[1][:, bn_idx], ref[1][:, ref_idx]
        verdict = differ.deterministic_verdict(a, b, atol=rc.DEFAULT_ATOL)
        assert verdict["n_cells"] == a.size and verdict["n_fail"] == 0
        assert verdict["max_err_over_movement"] <= MAX_ERR_OVER_MOVEMENT
        peak = np.maximum(np.abs(b).max(axis=0), rc.DEFAULT_ATOL)
        worst = np.abs(a - b).max(axis=0) / peak
        assert worst.max() <= MAX_ERR_OF_PEAK, common[int(worst.argmax())]


def _worst_self_rate(ode, run, atol: float) -> float:
    """The largest ``df_i/dx_i`` over the cells of ``run`` whose species is below
    ``atol`` and not 0. Below 1e-280 a value is 0 to this check: the complex step's
    h, 1e-20 of the value, would leave the normal doubles. A species at 0 cannot be
    held: the hold sits at -source/rate, which is 0 only when nothing feeds it, and
    then 0 is the solution. A rate the complex step cannot read (NaN, where the
    right-hand side is not finite) fails, in whichever row it is."""
    t, v, _ = run
    small = (np.abs(v) < atol) & (np.abs(v) > 1e-280)
    assert small.any(), "no cell below atol to look at"
    rates = np.concatenate(
        [ode.self_rates(tk, v[k], which=np.flatnonzero(small[k])) for k, tk in enumerate(t)]
    )
    assert np.isfinite(rates).all(), "a self-rate the complex step cannot read"
    return float(rates.max())


_TIME = (
    "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>"
)


def _held_sbml() -> str:
    """X' = lam(t) * (X - 1e-13), lam(t) = 4t(1 - t), started at 1e-13: X sits still
    on an equilibrium whose own rate lam is 0 at t = 0 and 1, and 1 at t = 0.5.
    Y' = -Y beside it, from 5e-13: below atol too, and attracted (rate -1)."""
    lam = f"<apply><times/><cn>4</cn>{_TIME}<apply><minus/><cn>1</cn>{_TIME}</apply></apply>"

    def species(sid, x0):
        return (
            f"<species id='{sid}' compartment='c' initialConcentration='{x0}' "
            "hasOnlySubstanceUnits='false' boundaryCondition='false' constant='false'/>"
        )

    def reaction(rid, side, sid, law):
        refs = f"<speciesReference species='{sid}' stoichiometry='1' constant='true'/>"
        return (
            f"<reaction id='{rid}' reversible='false'><{side}>{refs}</{side}>"
            f"<kineticLaw>{_MATH.format(law)}</kineticLaw></reaction>"
        )

    return (
        _HEAD
        + "<listOfCompartments><compartment id='c' size='1' constant='true'/></listOfCompartments>"
        + f"<listOfSpecies>{species('X', 1e-13)}{species('Y', 5e-13)}</listOfSpecies>"
        + "<listOfReactions>"
        + reaction(
            "grow", "listOfProducts", "X", f"<apply><times/><ci>c</ci>{lam}<ci>X</ci></apply>"
        )
        + reaction(
            "leak", "listOfReactants", "X", f"<apply><times/><ci>c</ci>{lam}<cn>1e-13</cn></apply>"
        )
        + reaction("decay", "listOfReactants", "Y", "<apply><times/><ci>c</ci><ci>Y</ci></apply>")
        + "</listOfReactions></model></sbml>"
    )


def test_the_repelling_check_finds_a_planted_hold():
    """#1542's hold in one species, beside one that is not held: the check reads
    X's largest rate, 1 at t = 0.5, not Y's -1, and not the 0 at either end."""
    ode = so.SbmlOde(_held_sbml())
    run = ode.solve(0.0, 1.0, 11, method="Radau", rtol=1e-12, atol=1e-30)
    t, v, names = run
    assert names == ["X", "Y"]
    assert (v[:, 0] == 1e-13).all()  # held, exactly
    np.testing.assert_allclose(v[:, 1], 5e-13 * np.exp(-t), rtol=1e-8)
    assert _worst_self_rate(ode, run, 1e-12) == pytest.approx(1.0)


def test_the_repelling_check_fails_a_rate_it_cannot_read_in_any_row():
    """Under A^0.5 a small A below 0 has no real rate. In the first row or a later
    one, the check fails rather than read the other cells (#1559 round 3)."""
    ode = so.SbmlOde(
        _decay_sbml(
            law="<apply><times/><ci>c</ci><apply><power/><ci>A</ci><cn>0.5</cn></apply></apply>"
        )
    )
    good, bad = [1e-13, 1.0], [-1e-13, 1.0]
    for rows in ([bad, good, good], [good, good, bad]):
        run = (np.array([0.0, 0.5, 1.0]), np.array(rows), ["A", "B"])
        with pytest.raises(AssertionError, match="cannot read"):
            _worst_self_rate(ode, run, 1e-12)


@pytest.mark.parametrize("key", sorted(ORACLE_RUNS))
def test_no_small_species_sits_on_a_repelling_equilibrium(key):
    """#1542: a species below atol whose own rate grows with it (df_i/dx_i > 0),
    held where a source term balances it, is not followed by an error test that
    cannot see it. Here every species between 0 and the suite's atol is attracted
    to where it is (df_i/dx_i <= 0), so the oracle's trajectory is not such a hold."""
    import _rr_common as rc

    xml = _model_xml(key)
    ode, (first, _) = _oracle_runs(key, xml)
    assert _worst_self_rate(ode, first, rc.DEFAULT_ATOL) <= 0.0
