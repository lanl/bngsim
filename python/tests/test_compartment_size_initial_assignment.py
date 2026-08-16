"""Issue #170 — an ``<initialAssignment>`` is a load-time fold of the volume.

Issue #164 refused a compartment-size write because the size is folded at load
into constants nothing re-derives, and stages 1 and 2 of #170 enumerated those
folds and put each one back on the parameter. The enumeration was incomplete.
SBML evaluates every ``<initialAssignment>`` once, at load, against a numeric
context seeded from the compartment sizes, and hands the *number* to the model.
Anything defined that way — a parameter, an initial condition — keeps its
load-time value when a write moves the size.

That is not a stale copy in a corner. The PBPK family
BIOMD0000001027/1028/1029/1039 copies each compartment size into its own
parameter (``Compartment_18 = StomachLumen``) and weights assignment rules by
the copy, so after stage 2 ``set_param("StomachLumen", …)`` returned a
trajectory 77% away from the rebuild it is contracted to reproduce — with
nothing on stderr, because every other fold had been fixed and agreed. All five
residual models were REFUSED before stage 2, so stage 2 traded a loud refusal
for a silent wrong answer on exactly the class this codebase refuses over.

The fix is the move stage 1 made for the rate constant: the fold is lifted back
onto the size as a *derived* expression, and issue #43's chain rule re-evaluates
it on the write. What cannot be lifted is refused by name.

The oracle throughout is ``compartment_sizes=``, which is separately adjudicated
as bit-for-bit a rewrite of the document's ``size=`` attribute and which
RoadRunner agrees with. Equality is exact: an ``allclose`` assertion here would
pass on the broken behaviour for every shape where the copy only weights an
output. The fixture's constants are all binary-exact for that reason — the one
place two roundings genuinely differ has its own bounded test below, following
:mod:`test_compartment_size_live`.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

T_SPAN = (0.0, 20.0)
N_POINTS = 9

# ── Fixture ─────────────────────────────────────────────────────────────────
#
# One compartment `C`, one parameter `P` defined by an initialAssignment, and a
# decay whose rate is scaled by `P`. `P` is what makes the model sensitive to
# the fold: it is not the compartment symbol, so every stage-1/2 fold sees a
# plain constant and reports itself consistent.

MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_volume">
    <listOfCompartments>
      <compartment id="C" size="{v}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" {a_ic} hasOnlySubstanceUnits="{hosu}" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="{k}" constant="true"/>
      <parameter id="off" value="1.5" constant="true"/>
      <parameter id="P" value="-1" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="P">
        <math xmlns="http://www.w3.org/1998/Math/MathML">{ia}</math>
      </initialAssignment>
      {extra_ia}
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>P</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# `P = C` — a bare <ci> on the size. The BioModels PBPK shape verbatim.
IA_BARE = "<ci>C</ci>"
# `P = 2*C + off` — compound, over the size and another constant parameter.
IA_COMPOUND = "<apply><plus/><apply><times/><cn>2</cn><ci>C</ci></apply><ci>off</ci></apply>"
# `P = A` — reads a species. Volume-dependent only when `A` is amount-valued and
# declared by concentration, because that is the pair section 0 converts through
# V; nothing in the parameter vector holds the result either way.
IA_SPECIES = "<ci>A</ci>"
# `P = off` — reads no volume at all. The control: this model stays writable.
IA_CONST = "<ci>off</ci>"

# `A(0) = k*C` — the initial-condition half of the same fold (MODEL1710030000).
IC_IA = """<initialAssignment symbol="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>C</ci></apply>
        </math>
      </initialAssignment>"""


def _src(v, ia=IA_BARE, extra_ia="", a_ic='initialConcentration="128"', hosu="false", k=0.25):
    return MODEL.format(
        v=repr(float(v)), ia=ia, extra_ia=extra_ia, a_ic=a_ic, hosu=hosu, k=repr(float(k))
    )


def _traj(model):
    return np.asarray(
        bngsim.Simulator(model, method="ode")
        .run(t_span=T_SPAN, n_points=N_POINTS, rtol=1e-12, atol=1e-14)
        .species
    )


def _rebuild_vs_write(v_load, v_new, ia=IA_BARE, extra_ia=""):
    """``(rebuilt_at_v_new, written_from_v_load)`` trajectories."""
    src = _src(v_load, ia, extra_ia)
    rebuilt = _traj(bngsim.Model.from_sbml_string(src, compartment_sizes={"C": v_new}))
    m = bngsim.Model.from_sbml_string(src)
    m.set_param("C", v_new)
    return rebuilt, _traj(m)


# ── The contract ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ia", "extra_ia"),
    [
        pytest.param(IA_BARE, "", id="parameter_is_the_size"),
        pytest.param(IA_COMPOUND, "", id="parameter_is_an_expression_over_the_size"),
        pytest.param(IA_BARE, IC_IA, id="and_an_initial_condition_reads_it_too"),
    ],
)
@pytest.mark.parametrize(("v_load", "v_new"), [(1.0, 4.0), (0.5, 2.0), (4.0, 0.125)])
def test_a_write_reproduces_a_rebuild_exactly(ia, extra_ia, v_load, v_new):
    rebuilt, written = _rebuild_vs_write(v_load, v_new, ia, extra_ia)
    assert np.array_equal(rebuilt, written)


def test_the_initial_condition_half_moves_on_its_own():
    """``A(0) = k*C`` with no volume-dependent parameter anywhere.

    A different site from the table above: the IC is lowered to the synthetic
    derived parameter issue #147 already builds (``_ic_<species>``), not to the
    target parameter itself. Without the fix the IC keeps ``k*V_load``, so the
    state at t=0 is already wrong and every later sample inherits it.
    """
    src = _src(1.0, ia=IA_CONST, extra_ia=IC_IA)
    rebuilt = bngsim.Model.from_sbml_string(src, compartment_sizes={"C": 4.0})
    written = bngsim.Model.from_sbml_string(src)
    written.set_param("C", 4.0)
    assert written.get_state()[0] == rebuilt.get_state()[0] == 1.0
    assert np.array_equal(_traj(rebuilt), _traj(written))


def test_the_lift_does_not_move_the_nominal_point():
    """Nothing may move until something is written.

    The lifted parameter is still *seeded* with the number section 0 folded, and
    a derived parameter is only re-evaluated by a write — which is what makes
    this safe to apply to every model rather than only the writable ones. The
    corpus arm of the same claim is the RHS fingerprint: bit-identical on
    212/212.
    """
    for ia, expected in ((IA_BARE, 2.5), (IA_COMPOUND, 2 * 2.5 + 1.5)):
        assert bngsim.Model.from_sbml_string(_src(2.5, ia)).get_param("P") == expected


def test_the_lifted_parameter_follows_the_size():
    m = bngsim.Model.from_sbml_string(_src(0.5, IA_COMPOUND))
    assert m.get_param("P") == 2 * 0.5 + 1.5
    m.set_param("C", 4.0)
    assert m.get_param("P") == 2 * 4.0 + 1.5


def test_a_lifted_parameter_is_no_longer_a_primary():
    """The cost of the lift, pinned.

    ``P`` was a plain writable parameter and is now derived, so it drops out of
    ``primary_param_names`` exactly as ``_rateLaw_<rid>`` does. That is what the
    SBML says it is — a quantity defined by the compartment size — and the
    alternative is a parameter a caller can write *and* the size can silently
    overwrite.
    """
    m = bngsim.Model.from_sbml_string(_src(1.0))
    assert "P" not in m.primary_param_names
    assert "k" in m.primary_param_names
    # A parameter no initialAssignment defines is untouched.
    assert "off" in m.primary_param_names
    # (#313) And the lift is not about volumes: an initialAssignment over an
    # ordinary parameter costs the target its primary slot on the same terms,
    # because the same thing is true of it — `off` is what defines its value, so
    # a write to `P` is a write the next re-derivation overwrites.
    const = bngsim.Model.from_sbml_string(_src(1.0, IA_CONST))
    assert "P" not in const.primary_param_names
    assert "off" in const.primary_param_names


def test_the_double_rounding_through_the_lifted_parameter_is_at_most_one_ulp():
    """Where exactness above is a claim about arithmetic rather than rounding.

    A rebuild folds ``P·k`` at the new size and divides by it once
    (``sf = 1/V_new``); a write folds ``P·k`` at the load-time size and reaches
    the same number through the live ratio. Both are the same product of the
    same reals, and they agree exactly for every binary-exact pair — which is
    why the table above uses ``k = 0.25`` and power-of-two sizes. Two roundings
    are still two roundings, so ``k = 0.3`` at 1→3 separates them: bound the
    scalar rather than pretend otherwise. The defect this module exists for is
    0.77 relative, eleven orders away.
    """
    v_load, v_new, k = 1.0, 3.0, 0.3
    src = _src(v_load, ia=IA_BARE, k=k)
    written = bngsim.Model.from_sbml_string(src)
    written.set_param("C", v_new)
    rebuilt = bngsim.Model.from_sbml_string(src, compartment_sizes={"C": v_new})

    def scalar(m):
        return m.get_param("_rateLaw_r") * m._core.codegen_data()["reactions"][0]["stat_factor"]

    a, b = scalar(written), scalar(rebuilt)
    assert a != b, "if this ever becomes exact, fold this row back into the table"
    assert abs(a - b) <= np.spacing(abs(b))


# ── The refusal ─────────────────────────────────────────────────────────────


def test_an_amount_valued_species_fold_now_names_the_size():
    """``P = A`` with ``A`` amount-valued and declared by concentration.

    Section 0 binds an ``hasOnlySubstanceUnits`` species symbol to its *amount*,
    so the seed is ``conc·V`` and ``P`` folded the volume without ever naming
    it. That used to be unliftable — no parameter holds a species amount — so
    #164's honest answer was to refuse the size and point at
    ``compartment_sizes=``.

    There *is* something to lift onto: the conversion itself. ``A``'s symbol is
    worth ``conc·V`` and ``conc`` is a declaration constant, so the substitution
    emits ``128.0*C`` and the size stays symbolic (issue #379). The refusal is
    earned away rather than relaxed — the assertion below is the property it
    protected, that a write lands exactly where a rebuild lands.
    """
    src = _src(1.0, IA_SPECIES, a_ic='initialConcentration="128"', hosu="true")
    m = bngsim.Model.from_sbml_string(src)
    assert m.unwritable_compartment_size_params == []
    assert m.get_param("P") == pytest.approx(128.0)

    written = bngsim.Model.from_sbml_string(src)
    written.set_param("C", 4.0)
    rebuilt = bngsim.Model.from_sbml_string(src, compartment_sizes={"C": 4.0})
    assert written.get_param("P") == pytest.approx(512.0)  # conc 128 at V=4
    assert written.get_param("P") == rebuilt.get_param("P")
    assert np.array_equal(np.asarray(written.get_state()), np.asarray(rebuilt.get_state()))


@pytest.mark.parametrize(
    ("ia", "a_ic", "hosu"),
    [
        # No volume anywhere in the fold.
        pytest.param(IA_CONST, 'initialConcentration="128"', "false", id="reads_a_constant"),
        # `P = A` where A's declaration and its symbol's meaning MATCH, so
        # section 0 seeds it without touching V and the fold is volume-free.
        # The taint has to be this precise or the refusal swallows the corpus.
        pytest.param(IA_SPECIES, 'initialConcentration="128"', "false", id="reads_a_conc_species"),
        pytest.param(IA_SPECIES, 'initialAmount="128"', "true", id="reads_an_amount_species"),
    ],
)
def test_a_size_no_fold_reads_stays_writable(ia, a_ic, hosu):
    """The refusal is per-compartment and per-cause, not a blanket."""
    m = bngsim.Model.from_sbml_string(_src(1.0, ia, a_ic=a_ic, hosu=hosu))
    assert m.unwritable_compartment_size_params == []
    m.set_param("C", 4.0)
    assert m.get_param("C") == 4.0


TWO_COMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="nested">
    <listOfCompartments>
      <compartment id="C1" size="1" constant="true"/>
      <compartment id="C2" size="2" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="128" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C1" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.25" constant="true"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="C1">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><ci>C2</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_a_size_defined_from_another_size_refuses_the_one_it_reads():
    """``<initialAssignment symbol="C1">2*C2`` — legal SBML, and how a nested
    or scaled compartment gets written.

    Section 1 folds ``C1`` to a number, so ``set_param("C2", …)`` moves ``C2``
    and leaves ``C1`` where a rebuild moves both. ``C1`` is not an SBML
    ``<parameter>``, so there is no §2 slot to lift its expression onto — the
    size it reads is refused instead. ``C1`` itself stays writable: writing it
    directly is well defined, it is only the *other* direction that breaks.
    """
    m = bngsim.Model.from_sbml_string(TWO_COMP)
    assert m.get_param("C1") == 4.0  # the IA won over size="1"
    assert m.unwritable_compartment_size_params == ["C2"]
    with pytest.raises(ValueError, match="compartment size"):
        m.set_param("C2", 8.0)
    m.set_param("C1", 8.0)
    assert m.get_param("C1") == 8.0


# ── The corpus witnesses ────────────────────────────────────────────────────
#
# The two real models that made the case. Kept as tests because the synthetic
# fixture cannot show what made this hard to find: on BIOMD0000001027 every
# other fold agrees, so the error is a plain wrong trajectory with nothing
# anywhere to say so.


def _corpus(name):
    p = Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events" / f"{name}.xml"
    if not p.exists():  # pragma: no cover - the corpus ships with the repo
        pytest.skip(f"{name} not available")
    return str(p)


def test_biomd1027_write_reproduces_its_rebuild():
    """The PBPK cluster: four sizes written at once, four copies that must follow."""
    path = _corpus("BIOMD0000001027")
    base = bngsim.Model.from_sbml(path)
    unwritable = set(base.unwritable_compartment_size_params)
    live = [c for c in base.compartment_size_params if c not in unwritable]
    assert sorted(live) == ["Feces", "IntestineLumen", "StomachLumen", "Urine"]
    sizes = {c: base.get_param(c) * 2.5 for c in live}

    written = bngsim.Model.from_sbml(path)
    for c, v in sizes.items():
        written.set_param(c, v)
    rebuilt = bngsim.Model.from_sbml(path, compartment_sizes=sizes)
    # Each size is copied into a `Compartment_<n>` parameter by an
    # initialAssignment; the copy is what a write used to leave behind.
    for pid, cid in (
        ("Compartment_0", "IntestineLumen"),
        ("Compartment_5", "Urine"),
        ("Compartment_6", "Feces"),
        ("Compartment_18", "StomachLumen"),
    ):
        assert written.get_param(pid) == sizes[cid] == rebuilt.get_param(pid)

    def run(m):
        return np.asarray(
            bngsim.Simulator(m, method="ode")
            .run(t_span=(0.0, 10.0), n_points=6, rtol=1e-10, atol=1e-12)
            .species
        )

    assert np.array_equal(run(written), run(rebuilt))


def test_biomd327_refuses_the_size_its_neighbour_is_defined_from():
    """The corpus case for a size folded into another size — and the one a
    uniform corpus scan cannot see.

    ``lumen = cell/vr``. On stage 2 ``set_param("cell", 2.5)`` moved ``cell``
    and left ``lumen`` at 0.1 where a rebuild gives 0.25: a 1.6e-03 trajectory
    error. Every corpus sweep run for #170 missed it because they scale *every*
    writable size by the same factor, which keeps ``lumen == cell/vr`` true by
    accident — the perturbation has to move one compartment at a time to break
    the relation the fold encodes.

    ``cell`` is refused; ``lumen`` and ``plasma`` stay writable, because the
    refusal is per-compartment and it is only the direction ``cell → lumen``
    that a write cannot carry.
    """
    m = bngsim.Model.from_sbml(_corpus("BIOMD0000000327"))
    assert m.unwritable_compartment_size_params == ["cell"]
    with pytest.raises(ValueError, match="compartment size"):
        m.set_param("cell", 2.5)
    for cid in ("plasma", "lumen"):
        w = bngsim.Model.from_sbml(_corpus("BIOMD0000000327"))
        w.set_param(cid, w.get_param(cid) * 2.5)
        r = bngsim.Model.from_sbml(
            _corpus("BIOMD0000000327"),
            compartment_sizes={
                cid: bngsim.Model.from_sbml(_corpus("BIOMD0000000327")).get_param(cid) * 2.5
            },
        )
        assert np.array_equal(np.asarray(w.get_state()), np.asarray(r.get_state())), cid


def test_model1710030000_now_honours_the_write_exactly():
    """The other real shape — and the refusal has been earned away.

    ``cell`` feeds four species initial conditions. One (``S21 =
    393.927*0.055*cell``) always lowered; the other three also read
    ``numConcFactor``, and they used to be rejected, leaving three folded ICs
    out of four — a write that reproduces most of a rebuild, so the size was
    refused. Issue #379's substitution lowers all four, which makes the size
    genuinely writable, so the refusal is gone.

    The assertion is the property the refusal existed to protect, not the
    refusal itself: a write must land exactly where a rebuild lands.
    """
    m = bngsim.Model.from_sbml(_corpus("MODEL1710030000"))
    assert m.compartment_size_params == ["cell"]
    assert m.unwritable_compartment_size_params == []

    new = m.get_param("cell") * 2.5
    written = bngsim.Model.from_sbml(_corpus("MODEL1710030000"))
    written.set_param("cell", new)
    rebuilt = bngsim.Model.from_sbml(_corpus("MODEL1710030000"), compartment_sizes={"cell": new})
    assert np.array_equal(np.asarray(written.get_state()), np.asarray(rebuilt.get_state()))
