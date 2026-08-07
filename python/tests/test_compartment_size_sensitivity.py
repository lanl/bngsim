"""Issue #170 stage 3 — d(trajectory)/d(compartment size), analytically.

Stage 1 made an SBML compartment size a writable parameter and stage 2 made the
emitted C read it, so ``f`` is a function of ``p[V]``. What was still missing was
the derivative: the sensitivity RHS carried the kinetic-law half of ``d/dV`` — the
volume where the model wrote it as a symbol — and none of the **storage** half,
the places bngsim's own ``amount/V`` convention puts a volume that the document
never mentions. There are four, and each has a test below that fails without it:

===============================  =========================================
the volume is put in by          what it costs to leave it out
===============================  =========================================
the GH #75 amount factor         an hOSU reactant's ``∏V_c^m`` — the whole
                                 column, reported as 0
the GH #160 row divide           ``-ν·rate/V²`` — #164's headline, a
                                 nonzero column on a V-invariant species
an amount-declared ``x(0)``      ``-x(0)/V`` — the column is 0 at every t
an observable's units            ``Σ factor·x_i`` — the whole column at
                                 t=0, where every ``dx/dθ`` is still zero
===============================  =========================================

**The oracle is a rebuild at V ± h** (``compartment_sizes=``), never a finite
difference through ``set_param`` and never CVODES' own difference quotient: both
of those move the parameter through the same machinery under test and inherit
whatever it gets wrong. #192 settled that an override is bit-identical to editing
the document, so the rebuild arm is ground truth.

The refusal that used to cover every compartment size lives on, narrowed to the
sizes ``set_param`` itself refuses — see ``test_compartment_size_write.py``, which
owns that half. The contract joining them: with the emitted text held fixed (stage
2's invariant), moving ``p[V]`` *is* reloading at the new volume, so the column is
exactly as trustworthy as the write.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

T_SPAN = (0.0, 20.0)
N_POINTS = 5
# Three well-separated steps; a trajectory FD's noise is systematic in h, so one
# step can agree to 4 digits and still be the noise. Accept the best.
STEPS = (1e-6, 1e-4, 1e-3)

# ── Fixtures ────────────────────────────────────────────────────────────────

# Issue #164's own model. `A` is *exactly* C1-invariant: C1 appears once in
# transport's law and once in A's storage divide, and they cancel. `B` genuinely
# depends on both volumes. The reaction is cross-compartment, so it takes the
# GH #160 per-species divide.
XCOMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp">
    <listOfCompartments>
      <compartment id="C1" size="%(v1)s" constant="true"/>
      <compartment id="C2" size="%(v2)s" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C2" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="kb" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="transport" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C1</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="degB" reversible="false" fast="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C2</ci><apply><times/><ci>kb</ci><ci>B</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# One compartment, one A → B reaction, with the declaration and the rate law
# swappable — #170's own acceptance table, minus the rows that differ only in the
# load-time size (a write is not under test here, a derivative is).
ONE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="one">
    <listOfCompartments><compartment id="C" size="%(v)s" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" %(ic)s hasOnlySubstanceUnits="%(hosu)s" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="%(hosu)s" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">%(law)s</math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

MASS_ACTION = "<apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>"
# k*A/(50+A) — a Functional law, so its ∂/∂V goes through the observable that
# carries A's amount rather than through an Elementary geometry factor.
SATURABLE = (
    "<apply><divide/><apply><times/><ci>k</ci><ci>A</ci></apply>"
    "<apply><plus/><cn>50</cn><ci>A</ci></apply></apply>"
)


def _one(v, *, law=MASS_ACTION, ic='initialConcentration="100"', hosu="false"):
    return ONE % {"v": v, "law": law, "ic": ic, "hosu": hosu}


def _xcomp(v1=1.0, v2=5.0):
    return XCOMP % {"v1": v1, "v2": v2}


def _species(src, **kw):
    m = bngsim.Model.from_sbml_string(src)
    r = bngsim.Simulator(m, method="ode", **kw).run(
        t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14
    )
    return r


def _column(src, comp, **kw):
    """The analytic ``d(species)/d(comp)`` column, shape ``(n_points, n_species)``."""
    r = _species(src, sensitivity_params=[comp], **kw)
    s = np.asarray(r.sensitivities)
    return s[:, :, 0] if s.ndim == 3 else s


def _rebuild_fd(build, v=3.0, value=lambda r: np.asarray(r.species)):
    """Central difference of a REBUILD at ``v ± h``, best over three steps.

    ``build(v)`` returns the SBML source at that size — a load, not a write, so
    nothing about the analytic path is reused to referee it.
    """
    out = []
    for rel in STEPS:
        h = abs(v) * rel
        out.append((value(_species(build(v + h))) - value(_species(build(v - h)))) / (2.0 * h))
    return out


def _agrees(analytic, fds, tol=1e-5):
    """Max relative error over components, taking the best step per component.

    Normalised by the column's ‖·‖∞, never by a component's own magnitude: a
    component the volume does not move sits at integration noise while the
    analytic answer is ~1e-17, and dividing by either turns exact agreement into 1.
    """
    analytic = np.asarray(analytic, dtype=float)
    best = np.full(analytic.shape, np.inf)
    for fd in fds:
        fd = np.asarray(fd, dtype=float)
        scale = max(float(np.max(np.abs(analytic))), float(np.max(np.abs(fd))), 1e-300)
        best = np.minimum(best, np.abs(fd - analytic) / scale)
    return float(np.max(best)) < tol, float(np.max(best))


# ── #164's headline: the cross-compartment row divide (GH #160) ─────────────


def test_the_c1_invariant_species_gets_a_structurally_zero_column():
    """Issue #164's first symptom, from the derivative side.

    ``A``'s row is ``-transport/C1`` and ``transport`` is ``C1·k·A``, so ``dA/dC1``
    is zero — not small, *zero*, at every state. The kinetic-law half alone gives
    ``+k·A/C1`` and #164 measured the resulting column at 36.6 where the truth is
    0. What cancels it is the storage half, ``-ν·rate/C1²``, and because the two
    are emitted as the same product with opposite signs the cancellation is exact
    in floating point rather than merely close — which is the assertion that
    would survive nothing but the right term.
    """
    col = _column(_xcomp(), "C1")
    assert np.array_equal(col[:, 0], np.zeros(N_POINTS))
    # ...and it is not zero because the whole column is: B genuinely moves.
    assert abs(col[-1, 1]) > 1.0


def test_every_compartment_column_of_the_164_model_matches_a_rebuild():
    """The other three (species, compartment) cells, against the rebuild oracle.

    #164 also measured ``dB/dC2`` as 0 where the truth is nonzero — the mirror
    failure, a column dropped rather than invented, on the row whose divide is a
    compartment the kinetic law's ``C2`` had already cancelled out of.
    """
    for comp, build in (
        ("C1", lambda v: _xcomp(v1=v)),
        ("C2", lambda v: _xcomp(v2=v)),
    ):
        v0 = 1.0 if comp == "C1" else 5.0
        ok, err = _agrees(_column(_xcomp(), comp), _rebuild_fd(build, v0))
        assert ok, f"{comp}: rel {err:.2e}"
    # The one #164 quotes as reported-zero is really this big.
    assert abs(_column(_xcomp(), "C2")[-1, 1]) > 0.5


# ── The GH #75 amount factor ────────────────────────────────────────────────


def test_an_hosu_reactant_carries_its_amount_factor_into_the_column():
    """``hasOnlySubstanceUnits=true`` makes the law read ``A`` as an *amount*, so
    the rate carries a ``∏V_c^m`` the document never wrote. Nothing differentiated
    it, and since it is the only place ``V`` survives on this model the reported
    column was exactly 0 while the trajectory moves with ``V``.
    """
    col = _column(_one(3.0, hosu="true"), "C")
    ok, err = _agrees(col, _rebuild_fd(lambda v: _one(v, hosu="true")))
    assert ok, f"rel {err:.2e}"
    assert np.abs(col).max() > 0.0, "the pre-#170-stage-3 answer was a hard zero"


def test_a_functional_law_reads_the_amount_through_an_observable():
    """The same conversion, one road over.

    A Functional rate law names the species, and an amount-valued species reaches
    it as ``obs[j] = factor·V·y[j]`` — so ``V`` is the *units* the law's inputs are
    quoted in, not a symbol of the law. ``sp.diff`` w.r.t. the size sees only the
    explicit occurrences (here: none), and the column came out 14% low at V = 3.
    A percentage error, unlike the hard zero above, is the kind that survives a
    casual sanity check.
    """
    ok, err = _agrees(
        _column(_one(3.0, law=SATURABLE, hosu="true"), "C"),
        _rebuild_fd(lambda v: _one(v, law=SATURABLE, hosu="true")),
    )
    assert ok, f"rel {err:.2e}"


# ── The initial condition ───────────────────────────────────────────────────


def test_an_amount_declared_initial_condition_seeds_the_column():
    """``initialAmount`` means the stored ``x(0) = amount/V`` moves with the
    volume, so ``∂x(0)/∂V = -x(0)/V`` — a seed, not an RHS term.

    This model's RHS is exactly V-free (``C`` cancels between the law and the
    storage divide), so the entire column is the seed. Without it the answer is 0
    at every output point while a rebuild moves the trajectory by 11 units.
    """
    src = _one(3.0, ic='initialAmount="100"')
    col = _column(src, "C")
    ok, err = _agrees(col, _rebuild_fd(lambda v: _one(v, ic='initialAmount="100"')))
    assert ok, f"rel {err:.2e}"
    assert np.abs(col).max() > 1.0

    # The seed itself, reported rather than inferred (issue #155's contract: the
    # parameter axis is the TOTAL derivative, so ∂x(0)/∂V belongs in it).
    m = bngsim.Model.from_sbml_string(src)
    seed = m.effective_ic_sensitivity(["C"])
    assert seed["A"]["C"] == pytest.approx(-(100.0 / 3.0) / 3.0, rel=1e-12)


def test_a_concentration_declared_initial_condition_has_no_seed():
    """The companion that keeps the rule from being "every species in a writable
    compartment". A concentration IC stores the declared number at every volume,
    so its seed is a true zero — seeding ``-x(0)/V`` there would invent a
    dependence the document does not have."""
    m = bngsim.Model.from_sbml_string(_one(3.0, hosu="true"))
    assert m.effective_ic_sensitivity(["C"]) == {}


def test_a_volume_scaled_initial_assignment_cancels_to_zero():
    """The case that needs *both* IC halves, and the one the pre-#170-stage-3 code
    filtered out rather than answer.

    ``A(0) = 7·C`` as an amount over a stored ``amount/V`` is exactly V-invariant:
    the parameter-graph chain rule gives ``∂(7C)/∂C = 7``, the storage divide
    turns that into ``7/V``, and the explicit ``-x(0)/V`` is ``-7/V``. They cancel.
    Report only the first and the column reads 7 where the truth is 0 — which is
    what ``MODEL1710030000`` did (21.67 against a rebuild-to-rebuild 0), and why
    the seed used to be dropped on the floor instead.
    """
    src = ONE % {"v": 3.0, "law": MASS_ACTION, "hosu": "true", "ic": ""}
    src = src.replace(
        "</listOfSpecies>",
        "</listOfSpecies>\n    <listOfInitialAssignments>"
        '<initialAssignment symbol="A"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<apply><times/><cn>7</cn><ci>C</ci></apply></math></initialAssignment>"
        "</listOfInitialAssignments>",
    )
    m = bngsim.Model.from_sbml_string(src)
    assert m._core.get_initial_state()[0] == pytest.approx(7.0)
    row = m.effective_ic_sensitivity(["C"]).get("A", {})
    # Present (there IS a seeding path) *and* zero (the two halves cancel).
    # Issue #155 makes those different answers, and asserting only the value
    # would pass against the code that dropped the row on the floor — which is
    # what main does here, reporting {} rather than a cancelled 0.
    assert "C" in row, "the ∂x(0)/∂V seeding path must be reported, not dropped"
    assert row["C"] == pytest.approx(0.0, abs=1e-12)


# ── The observable's own units ──────────────────────────────────────────────


def test_an_observable_of_an_hosu_species_carries_its_units_at_t_zero():
    """``obs = V·x`` for an amount-valued species, so ``d obs/dV`` has a direct
    ``x`` on top of the chain rule ``V·dx/dV``.

    At ``t = 0`` every ``dx/dθ`` is exactly zero, so the direct term is the
    *entire* answer — and it is ``x(0)``, a number this test can state outright
    rather than compare to a difference. Omit it and the column starts at 0.
    """
    r = _species(_one(3.0, law=SATURABLE, hosu="true"), sensitivity_params=["C"])
    col = np.squeeze(np.asarray(r.output_sensitivities("observable:A")))
    assert col[0] == pytest.approx(100.0, rel=1e-12)
    ok, err = _agrees(
        col,
        _rebuild_fd(
            lambda v: _one(v, law=SATURABLE, hosu="true"),
            3.0,
            value=lambda rr: np.squeeze(np.asarray(rr.outputs("observable:A"))),
        ),
    )
    assert ok, f"rel {err:.2e}"


# ── The lifted entry points ─────────────────────────────────────────────────


def test_compute_all_sensitivities_now_includes_a_writable_size():
    """``params=None`` used to drop every compartment column with a warning. A
    writable size is computable now, so it stays in the tensor and the warning
    does not fire — see test_compartment_size_write.py for the unwritable half,
    where both still do."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(_xcomp()), method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_all_sensitivities(t_span=T_SPAN, n_points=N_POINTS)
    assert "C1" in res.sensitivity_params and "C2" in res.sensitivity_params
    skips = [w for w in caught if "compartment size" in str(w.message)]
    assert not skips, [str(w.message)[:120] for w in skips]


def test_steady_state_sensitivity_accepts_a_writable_size():
    """``dY_ss/dp`` reads ∂f/∂p out of the same emitted sensitivity RHS, so it is
    right for a writable size exactly when that column is. It carries no IC seed —
    a steady state has forgotten x(0) — so only the ∂f/∂V half is in play."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(_xcomp()), method="ode")
    ss = sim.steady_state(sensitivity_params=["C1", "C2"])
    assert ss is not None


# ── Both backends ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("codegen", [None, True], ids=["auto", "codegen"])
def test_the_column_is_backend_independent(codegen):
    """The storage half is emitted once, in the shared codegen the interpreted and
    compiled paths both read, so the two backends must agree to the last digit.
    A difference here is the signature of a fold that survived on one side only —
    which is exactly how #170 stage 2 found its own per-species leftover."""
    col = _column(_one(3.0, hosu="true"), "C", codegen=codegen)
    ref = _column(_one(3.0, hosu="true"), "C")
    assert np.allclose(col, ref, rtol=1e-12, atol=1e-14)
