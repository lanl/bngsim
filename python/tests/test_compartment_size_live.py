"""Issue #170 — an SBML compartment size is a live parameter.

A volume plays two roles in a loaded model: a **symbol** in kinetic laws, which
is an ordinary ``p[]`` a write has always moved, and the **storage convention**
(bngsim stores ``amount/V_c``), which was folded at load into constants nothing
re-derived. Issue #164 refused the write outright because which of the two
halves landed was not even uniform inside one model. This module is the fix:
each fold is put back on the parameter, so a write reproduces a *rebuild at the
new size* — the thing SBML actually means, and what RoadRunner agrees with.

The whole file is one oracle, stated once: for every shape of kinetic law and
initial-condition declaration issue #170 tabulated,

    load(V=a); set_param(C, b)   ==   load(V=b)

**bit for bit**, and both agree with RoadRunner. Equality is exact, not
``allclose``: an approximate assertion here would pass on the pre-#170 behaviour
for the rows where the volume nearly cancels, which is exactly how #164's
scoping went wrong. Where the two must be identical, say identical.

Two rows of that table are still refused rather than honored, and
:mod:`test_compartment_size_write` owns those: the emitted C keeps a literal
volume for an amount-valued (``hasOnlySubstanceUnits``) species and for the
per-species divide of a cross-compartment reaction, so honoring the write would
mean honoring it with codegen off and half-applying it with codegen on. That is
the stage-2 boundary, and it is a refusal with a reason rather than a wrong
number.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────
#
# One template, parameterised over the two axes issue #170's table varies: the
# kinetic law's compartment power, and how the initial condition is declared.

ONE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="one">
    <listOfCompartments>
      <compartment id="C" size="{v}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" {ic} hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">{law}</math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# The BNG convention: the compartment cancels against the storage divide, so the
# volume never reaches `sf` and this row was already correct before #170.
L_CkA = "<apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>"
# The common BioModels form: net power -1, folded as sf = k/V.
L_kA = "<apply><times/><ci>k</ci><ci>A</ci></apply>"
# Net power +1, folded as sf = k*V — the opposite direction, so a fix that got
# the sign or the reciprocal wrong passes the row above and fails this one.
L_CCkA = "<apply><times/><ci>C</ci><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>"
# Not mass-action: takes the Functional path, where the volume is the storage
# divide emitted against the compartment symbol rather than a scalar fold.
L_SAT = (
    "<apply><divide/><apply><times/><ci>k</ci><ci>A</ci></apply>"
    "<apply><plus/><cn>1</cn><ci>A</ci></apply></apply>"
)

T_SPAN = (0.0, 20.0)
N_POINTS = 5


def _src(v, law=L_CkA, ic='initialConcentration="100"'):
    return ONE.format(v=repr(float(v)), law=law, ic=ic)


def _traj(model):
    return np.asarray(
        bngsim.Simulator(model, method="ode")
        .run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14)
        .species
    )


def _rebuild_vs_write(law, ic, v_load, v_new):
    """``(rebuilt_at_v_new, written_from_v_load)`` trajectories."""
    rebuilt = _traj(bngsim.Model.from_sbml_string(_src(v_new, law, ic)))
    m = bngsim.Model.from_sbml_string(_src(v_load, law, ic))
    m.set_param("C", v_new)
    return rebuilt, _traj(m)


# ── The table ───────────────────────────────────────────────────────────────

ROWS = [
    # (id, law, ic-declaration, load-time V, written V)
    ("compartment_times_k_A", L_CkA, 'initialConcentration="100"', 1.0, 3.0),
    ("bare_k_A", L_kA, 'initialConcentration="100"', 1.0, 3.0),
    ("compartment_squared", L_CCkA, 'initialConcentration="100"', 1.0, 3.0),
    ("initial_amount", L_CkA, 'initialAmount="100"', 1.0, 3.0),
    # The row pair that made #170's case: the same law and the same model, one
    # loaded at V=1 and one at V=4. At V=1 the storage divide used to be
    # normalised away as a ÷1 no-op, so the write was dropped; at V≠1 it was
    # emitted against the live symbol and the write was honored. Opposite
    # outcomes decided by nothing but the load-time size.
    ("saturable_loaded_at_1", L_SAT, 'initialConcentration="100"', 1.0, 8.0),
    ("saturable_loaded_at_4", L_SAT, 'initialConcentration="100"', 4.0, 8.0),
    # Awkward binary sizes, still exact: the scalar goes through the ratio as
    # k/(V_new/V_load)/V_load rather than k/V_new, and those agree here.
    ("awkward_sizes", L_kA, 'initialConcentration="100"', 0.1, 0.3),
]


@pytest.mark.parametrize(
    ("law", "ic", "v_load", "v_new"),
    [pytest.param(*r[1:], id=r[0]) for r in ROWS],
)
def test_a_write_reproduces_a_rebuild_exactly(law, ic, v_load, v_new):
    rebuilt, written = _rebuild_vs_write(law, ic, v_load, v_new)
    assert np.array_equal(rebuilt, written)


@pytest.mark.parametrize(
    ("law", "ic", "v_load", "v_new"),
    [pytest.param(*r[1:], id=r[0]) for r in ROWS],
)
def test_the_rebuild_is_what_roadrunner_means(law, ic, v_load, v_new):
    """The rebuild is not merely self-consistent — it is SBML's own answer, so
    the equality above is a correctness claim and not a tautology. Kept as a
    separate test from the exactness one so a RoadRunner-less environment still
    gets the bit-for-bit assertion."""
    roadrunner = pytest.importorskip("roadrunner")
    rr = roadrunner.RoadRunner(_src(v_new, law, ic))
    rr.integrator.relative_tolerance = 1e-12
    rr.integrator.absolute_tolerance = 1e-14
    truth = np.asarray(rr.simulate(T_SPAN[0], T_SPAN[1], N_POINTS, ["time", "[A]", "[B]"]))[:, 1:]
    _, written = _rebuild_vs_write(law, ic, v_load, v_new)
    assert np.allclose(written, truth, rtol=1e-5, atol=1e-9)


def test_the_double_rounding_through_the_ratio_is_at_most_one_ulp():
    """The one place the equality above is a claim about rounding rather than
    arithmetic. A written scalar is ``k/(V_new/V_load)/V_load`` — the load-time
    fold left intact, times the live ratio — where a rebuild computes ``k/V_new``
    directly. Those agree exactly for every ordinary pair of sizes (the table
    above spans 0.1→0.3 and 3→7), but two roundings are two roundings, and a
    subnormal-scale pair finds the gap. Bound it rather than pretend it is not
    there: one ulp on the scalar, not a re-derivation of the model.
    """
    v_load, v_new = 1.65e-11, 7.3e-11
    written = bngsim.Model.from_sbml_string(_src(v_load, L_kA))
    written.set_param("C", v_new)
    rebuilt = bngsim.Model.from_sbml_string(_src(v_new, L_kA))

    def scalar(m):
        k = m.get_param("_rateLaw_r") if "_rateLaw_r" in m.param_names else m.get_param("k")
        return k * m._core.codegen_data()["reactions"][0]["stat_factor"]

    a, b = scalar(written), scalar(rebuilt)
    assert a != b, "if this ever becomes exact, fold this test back into the table"
    assert abs(a - b) <= np.spacing(abs(b))


def test_a_cancelling_compartment_is_not_perturbed_by_the_machinery():
    """``compartment·k·A`` nets to zero volume power, so #170 must add *nothing*
    to it: no derived rate parameter, no storage divide, the same scalar. This
    is the byte-identity half of the corpus A/B, asserted structurally rather
    than by hashing a trajectory."""
    m = bngsim.Model.from_sbml_string(_src(4.0, L_CkA))
    assert not [n for n in m.param_names if n.startswith("_rateLaw_")]
    assert m._core.codegen_data()["reactions"][0]["stat_factor"] == 1.0


def test_the_volume_ratio_is_exactly_one_at_the_nominal_point():
    """The whole corpus-wide safety of #170 rests on this: the volume rides the
    rate parameter as ``k · (C/V_load)``, and ``C/V_load`` must be *exactly* 1.0
    at load or every mass-action rate constant in every SBML model shifts by an
    ulp. It is not enough to print ``V_load`` as a decimal literal — ExprTk's own
    literal parser is not correctly rounded, and 1.65e-11 (a repr round-trip that
    is exact in Python) came back 1 ulp low, turning a 5e-05 rate constant into
    4.999999999999999e-05 and moving the RHS of a real corpus model."""
    for v in (1.0, 3.0, 1.65e-11, 0.1, 1e300):
        m = bngsim.Model.from_sbml_string(_src(v, L_kA))
        assert m.get_param("_rateLaw_r") == m.get_param("k"), v


def test_the_declared_amount_is_the_invariant_not_the_stored_value():
    """An ``initialAmount`` species stores ``amount/V``, so a volume write moves
    the state's *meaning*: the amount is what stays fixed. The stored initial
    condition has to follow — and follow the issue #79 rule, so a species the
    dynamics have not moved comes along with it."""
    m = bngsim.Model.from_sbml_string(_src(1.0, L_CkA, 'initialAmount="100"'))
    assert m.get_state()[0] == 100.0
    m.set_param("C", 4.0)
    assert m.get_state()[0] == 25.0
    m.reset()
    assert m.get_state()[0] == 25.0


def test_an_integrated_state_is_not_rewritten_under_the_protocol():
    """The other half of the #79 rule: once a run has advanced the species, its
    value is no longer the declared initial condition and a mid-protocol volume
    change must not discard it. The *next* reset picks the new one up."""
    m = bngsim.Model.from_sbml_string(_src(1.0, L_CkA, 'initialAmount="100"'))
    bngsim.Simulator(m, method="ode").run(t_span=(0.0, 5.0), n_points=2)
    advanced = m.get_state()[0]
    assert advanced != 100.0
    m.set_param("C", 4.0)
    assert m.get_state()[0] == advanced
    m.reset()
    assert m.get_state()[0] == 25.0


def test_a_scan_over_a_volume_is_a_scan(monkeypatch):
    """The user-facing point of the whole issue: ``parameter_scan`` over a
    compartment used to return one trajectory N times (issue #164 refused it
    outright rather than ship that). Each point must now equal its own rebuild.
    """
    sizes = [1.0, 2.0, 5.0]
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(_src(1.0, L_kA)), method="ode")
    scan = sim.parameter_scan("C", sizes, t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14)
    got = [np.asarray(r.species) for r in scan]
    assert not np.array_equal(got[0], got[-1]), "a scan that returns the same curve is the bug"
    for v, y in zip(sizes, got, strict=True):
        assert np.allclose(y, _traj(bngsim.Model.from_sbml_string(_src(v, L_kA))), rtol=1e-9)


def test_the_write_survives_the_compiled_rhs():
    """Codegen reads the volume through the same ``p[]`` the interpreter does —
    the mass-action fold rides the rate parameter and the Functional storage
    divide is emitted against the compartment symbol — so the compiled and
    interpreted backends must agree on a written volume. They are the two halves
    issue #164 found disagreeing."""
    interp = bngsim.Model.from_sbml_string(_src(1.0, L_kA))
    interp.set_param("C", 3.0)
    compiled = bngsim.Model.from_sbml_string(_src(1.0, L_kA))
    compiled.set_param("C", 3.0)
    y_i = _traj(interp)
    y_c = np.asarray(
        bngsim.Simulator(compiled, method="ode", codegen=True)
        .run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14)
        .species
    )
    assert np.allclose(y_i, y_c, rtol=1e-9, atol=1e-12)


def test_net_models_are_untouched(simple_decay_net):
    """A ``.net`` network is post-BNG2.pl: the volumes are already folded into
    its rate constants, so nothing here may fire on one."""
    m = bngsim.Model.from_net(simple_decay_net)
    assert m.compartment_size_params == []
    assert m.unwritable_compartment_size_params == []
    assert not [n for n in m.param_names if n.startswith("_V0_")]
