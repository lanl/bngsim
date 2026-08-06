"""Issue #192 — a compartment-size write must not RECLASSIFY a reaction.

Issue #170 made an SBML compartment size a live parameter, on one contract:

    load(V=a); set_param(C, b)   ==   load(..., compartment_sizes={C: b})

Stages 1 and 2 chased that contract through every *fold* of the volume — the
Elementary scalar, the storage divide, the emitted C's ``inv_vf`` table, an
``<initialAssignment>``. Every one of them is a stale **value**, and putting the
value back on the parameter is enough. This module owns the failure that is not
a value at all.

The Functional emitter gates its single-rate shortcut on
``unified_ok = (not non_integer) and len(involved_vs) <= 1``, and ``involved_vs``
holds volume **VALUES**, not compartment ids. A reaction whose species span
several compartments that merely *share a load-time size* therefore passed, and
took a divide by one representative compartment (``_vd_<rid>_unified =
law/rep_comp``). That is exact only while the sizes agree. Write one of them and
the representative stops being the right divisor for the other compartments'
species — while a fresh load at the same size sees unequal volumes and takes the
per-species branch. The two arms are then structurally **different models**, so
no amount of re-deriving parameters closes the gap: it is the shape of the
emitted math that differs, not a number in it.

Measured on the corpus that way, the write was wrong on 50 (model, compartment)
pairs across 14 models, up to 1.07 relative, with nothing on stderr — and
RoadRunner sides with the rebuild. The fix routes such a reaction to the
per-species branch up front, which is the GH #144 case-4 fix two lines above the
gate for the variable-volume flavour, applied to the writable-static-volume
flavour it never covered. It is free at the nominal point (the per-species
divisors all equal the representative's there): 150 reactions across 22 corpus
models change branch and the RHS stays bit-identical on 214/214.

So this module asserts the *classification*, not only the answer. A test that
only compared trajectories would pass again the moment someone re-widened the
shortcut for a model whose volumes happen to be equal, which is exactly how the
defect survived every #164/#170 sweep: they all scaled every writable size by
the SAME factor, which keeps equal volumes equal and the shortcut valid. One
compartment at a time is the perturbation that sees it.
"""

from __future__ import annotations

import hashlib

import bngsim
import numpy as np
import pytest
from bngsim import _codegen
from bngsim._exceptions import ModelError

# ── Fixtures ────────────────────────────────────────────────────────────────
#
# Two compartments, one reaction across them, and a law that is deliberately NOT
# mass-action (a saturable ``k·A/(Km+A)``) so the reaction reaches the Functional
# emitter. The mass-action flavour of the same shape is already refused on the
# Elementary path via ``volume_unresolvable`` — see test_compartment_size_write —
# which is why that path is not in this list.

XCOMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp">
    <listOfCompartments>
      <compartment id="C1" size="{v1}" constant="true"/>
      <compartment id="C2" size="{v2}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"{cfa}/>
      <species id="B" compartment="C2" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"{cfb}/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="Km" value="20" constant="true"/>
      <parameter id="cfA" value="1.0" constant="true"/>
      <parameter id="cfB" value="2.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="transport" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/>
            <apply><times/><ci>k</ci><ci>A</ci></apply>
            <apply><plus/><ci>Km</ci><ci>A</ci></apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

T_SPAN = (0.0, 20.0)
N_POINTS = 5
V_EQUAL = 5.0  # the load-time size BOTH compartments share — the whole point
V_NEW = 3.0

CODEGEN = [pytest.param(None, id="interpreted"), pytest.param(True, id="codegen")]
# Either compartment: the representative divide picks ONE of them, so writing the
# other is the half that used to be a silent no-op rather than a wrong number.
COMPS = ["C1", "C2"]


def _src(v1, v2, mixed_cf=False):
    return XCOMP.format(
        v1=repr(float(v1)),
        v2=repr(float(v2)),
        cfa=' conversionFactor="cfA"' if mixed_cf else "",
        cfb=' conversionFactor="cfB"' if mixed_cf else "",
    )


def _load(v1, v2, **kw):
    return bngsim.Model.from_sbml_string(_src(v1, v2), **kw)


def _traj(model, codegen=None):
    return np.asarray(
        bngsim.Simulator(model, method="ode", codegen=codegen)
        .run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14)
        .species
    )


def _written(comp, v, codegen=None):
    """``load at the shared size, then write one compartment`` — the arm under test."""
    m = _load(V_EQUAL, V_EQUAL)
    m.set_param(comp, v)
    return m


def _rebuilt(comp, v):
    """The oracle: a fresh load carrying the new size. Bit-identical to editing
    the document — ``compartment_sizes=`` moves the size before any fold happens,
    which the four-arm RoadRunner adjudication measured at exactly 0.0."""
    sizes = {"C1": V_EQUAL, "C2": V_EQUAL} | {comp: v}
    return _load(sizes["C1"], sizes["C2"])


# ── The classification, which is the actual defect ──────────────────────────


def test_the_reaction_takes_the_per_species_divide_at_equal_load_sizes():
    """The invariant the fix installs, stated where a value test cannot reach it.

    At equal load-time sizes this reaction used to emit
    ``_vd_transport_unified = transport/C1`` and carry
    ``per_species_volume_scaling=False`` — one divisor for species in two
    compartments. It must take the per-species branch instead, the SAME branch
    the unequal-size load below already takes, so that the two builds of one
    model cannot diverge in shape."""
    for v1, v2 in ((V_EQUAL, V_EQUAL), (V_NEW, V_EQUAL)):
        rxns = _load(v1, v2)._core.codegen_data()["reactions"]
        assert len(rxns) == 1
        assert rxns[0]["per_species_volume_scaling"] is True, (v1, v2)
        assert rxns[0]["function_name"] == "transport", (v1, v2)


def test_a_write_and_a_rebuild_build_the_same_model():
    """Structural equality of the two arms — what ``classdiff.py`` reports on the
    corpus. Trajectory equality follows from this; the reverse does not, which is
    why it is asserted separately."""

    def shape(m):
        cd = m._core.codegen_data()
        return (
            [
                (r["type"], r["function_name"], r["per_species_volume_scaling"])
                for r in cd["reactions"]
            ],
            sorted(f["name"] for f in cd["functions"]),
        )

    for comp in COMPS:
        assert shape(_written(comp, V_NEW)) == shape(_rebuilt(comp, V_NEW)), comp


# ── The contract ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("codegen", CODEGEN)
@pytest.mark.parametrize("comp", COMPS)
def test_a_write_reproduces_a_rebuild_exactly(comp, codegen):
    """#170's contract, on the shape that violated it. Exact, not ``allclose``:
    the two arms are the same model now, so anything short of equality would be
    hiding a residue. Pre-fix this was 0.40 relative on ``C1`` and a silent no-op
    on ``C2``."""
    assert np.array_equal(
        _traj(_written(comp, V_NEW, codegen), codegen),
        _traj(_rebuilt(comp, V_NEW), codegen),
    )


@pytest.mark.parametrize("comp", COMPS)
def test_this_shape_really_does_move_with_the_size(comp):
    """Non-vacuity. Both compartments genuinely enter the answer — ``C1`` through
    ``A``'s storage divide and ``C2`` through ``B``'s — so the equality above is
    a claim about a moving quantity, not about two identical numbers."""
    base = _traj(_load(V_EQUAL, V_EQUAL))
    assert not np.array_equal(base, _traj(_rebuilt(comp, V_NEW)))


def test_a_write_of_the_size_it_already_shares_changes_nothing():
    """The degenerate direction. Re-routing means the per-species branch is taken
    even when the volumes agree, so writing a compartment back to the shared size
    has to land on the untouched load exactly — the arithmetic the corpus
    bit-identity result rests on (``law/V`` per species with every V equal is the
    unified divide, to the last bit)."""
    m = _load(V_EQUAL, V_EQUAL)
    m.set_param("C1", V_NEW)
    m.set_param("C1", V_EQUAL)
    assert np.array_equal(_traj(m), _traj(_load(V_EQUAL, V_EQUAL)))


def test_the_stochastic_propensity_moved_too():
    """The re-route swaps ``ssa_volume_factor=common_vs`` on one unified emission
    for ``1.0`` plus per-species scaling, so SSA is a second consumer of the
    classification and not a corollary. Seeded, so this is an exact claim: the
    same model integrates the same path. Pre-fix the write gave ``B=2.2`` where
    the rebuild gives ``0.8``."""

    def ssa(model):
        return np.asarray(
            bngsim.Simulator(model, method="ssa").run(t_span=T_SPAN, n_points=3, seed=7).species
        )

    assert np.array_equal(ssa(_written("C1", V_NEW)), ssa(_rebuilt("C1", V_NEW)))


# ── Why it is safe on the compiled backend ──────────────────────────────────


EMITTERS = [
    ("rhs", _codegen.generate_rhs_from_model),
    ("jac", _codegen.generate_jacobian_from_model),
    ("outputs", _codegen.generate_outputs_from_model),
    ("sens", _codegen.generate_sens_from_model),
    ("output_sens", _codegen.generate_output_sens_from_model),
]


@pytest.mark.parametrize(("emitter", "emit"), [pytest.param(n, f, id=n) for n, f in EMITTERS])
@pytest.mark.parametrize("comp", COMPS)
def test_a_write_does_not_change_the_emitted_source(comp, emitter, emit):
    """Stage 2's invariant, which the re-route has to keep: the ``.so`` a model
    was loaded with stays valid after a write. A per-species divisor that came
    out as a load-time literal rather than a ``p[]`` read would move the text
    here — and since the cache key is a hash of it, the write would silently
    recompile instead of being honored by the loaded binary."""
    m = _load(V_EQUAL, V_EQUAL)
    before = emit(m)
    m.set_param(comp, V_NEW)
    after = emit(m)
    if before is None or after is None:
        assert before is None and after is None, f"{emitter} declines on one side only"
        return
    assert (
        hashlib.sha256(before.encode()).hexdigest() == hashlib.sha256(after.encode()).hexdigest()
    )


@pytest.mark.parametrize("codegen", CODEGEN)
@pytest.mark.parametrize("comp", COMPS)
def test_a_write_after_the_source_is_generated_still_lands(comp, codegen):
    """The order the defect actually reached a caller: ``parameter_scan`` and any
    post-construction ``set_param`` write into an already-generated model, where
    a classification chosen at load is frozen."""
    rebuilt = _traj(_rebuilt(comp, V_NEW), codegen)
    m = _load(V_EQUAL, V_EQUAL)
    sim = bngsim.Simulator(m, method="ode", codegen=codegen)
    m.set_param(comp, V_NEW)
    assert np.array_equal(
        rebuilt,
        np.asarray(sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14).species),
    )


# ── The one shape that has to be refused instead ────────────────────────────


def test_mixed_conversion_factors_refuse_the_write_rather_than_re_route():
    """There is nowhere to re-route a cross-compartment reaction whose changed
    species carry DIFFERENT ``conversionFactor``s: the per-species branch refuses
    that combination outright (GH #232), which is why the rebuild below cannot
    even be built. A write that cannot reproduce a rebuild must not be silently
    honored, so the sizes are refused by name — the same answer
    ``volume_unresolvable`` already gives on the Elementary path. No corpus model
    is shaped this way; this is the branch's only coverage."""
    equal = bngsim.Model.from_sbml_string(_src(V_EQUAL, V_EQUAL, mixed_cf=True))
    assert sorted(equal.unwritable_compartment_size_params) == ["C1", "C2"]
    with pytest.raises(ValueError, match="equal size"):
        equal.set_param("C1", V_NEW)
    # And the reason it is refused rather than re-routed: the rebuild is not a
    # model bngsim can build at all.
    with pytest.raises(ModelError, match="conversionFactors"):
        bngsim.Model.from_sbml_string(_src(V_NEW, V_EQUAL, mixed_cf=True))


def test_the_refusal_does_not_disturb_the_model_it_refuses():
    """Refusing a size marks a parameter unwritable; it must not change what the
    model computes. The mixed-cf model still emits the unified divide and
    integrates exactly as it did before the size was flagged."""
    m = bngsim.Model.from_sbml_string(_src(V_EQUAL, V_EQUAL, mixed_cf=True))
    rxns = m._core.codegen_data()["reactions"]
    assert all(r["function_name"] == "_vd_transport_unified" for r in rxns)
    assert np.isfinite(_traj(m)).all()


# ── Independent oracle ──────────────────────────────────────────────────────


@pytest.mark.parametrize("comp", COMPS)
def test_the_rebuild_is_what_roadrunner_says(comp):
    """The arm this fix moves the write ONTO. The whole defect was adjudicated
    this way rather than assumed: an actually-edited document, the
    ``compartment_sizes=`` override, the write, and RoadRunner on the edited
    document. The override matched the edit at exactly 0.0, RoadRunner matched
    both, and the write was the outlier."""
    roadrunner = pytest.importorskip("roadrunner")
    sizes = {"C1": V_EQUAL, "C2": V_EQUAL} | {comp: V_NEW}
    rr = roadrunner.RoadRunner(_src(sizes["C1"], sizes["C2"]))
    rr.integrator.relative_tolerance = 1e-12
    rr.integrator.absolute_tolerance = 1e-14
    truth = np.asarray(rr.simulate(T_SPAN[0], T_SPAN[1], N_POINTS, ["time", "[A]", "[B]"]))[:, 1:]
    assert np.allclose(_traj(_written(comp, V_NEW)), truth, rtol=1e-6, atol=1e-9)
