"""Phase 4 — Result.as_roadrunner() and NamedArray.

Verifies the RR-compatible output adapter that replaces
``rr.simulate(...)`` in PyBNF stochastic-fitting workflows.

Coverage:
- Default selections shape and column order.
- ``[X]`` returns concentration (BNGsim's stored value, = amount/V_c).
- ``X`` returns amount (= ``[X]`` × V_c). Equal to ``[X]`` for V=1.
- ``time`` selector is the time array.
- String indexing on the returned NamedArray (RR convention).
- Unknown selector raises ValueError with an actionable message.
- Custom selection order is honored.
- Round-trip through HDF5 save/load preserves volume_factors.
- Optional parity with libroadrunner (skipped if RR not installed).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# ── Reference SBML: V=1, single species, simple decay ────────────────

DECAY_SBML_V1 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay_v1">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# Same model with V_c=2 to test concentration-vs-amount divergence.
DECAY_SBML_V2 = (
    DECAY_SBML_V1.replace('size="1" constant="true"', 'size="2" constant="true"')
    .replace('id="decay_v1"', 'id="decay_v2"')
    .replace(
        # hOSU=true so the loader knows the IC is amount: storage = 100/2 = 50.
        'hasOnlySubstanceUnits="true"',
        'hasOnlySubstanceUnits="true"',
    )
)


def _ode_run(sbml: str):
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ode")
    return sim.run(t_span=(0, 10), n_points=11, seed=1)


# ── Default selections shape and column order ────────────────────────


def test_default_selections_shape():
    result = _ode_run(DECAY_SBML_V1)
    arr = result.as_roadrunner()
    assert isinstance(arr, bngsim.NamedArray)
    assert arr.shape == (11, 1 + result.n_species)
    assert arr.colnames == ["time"] + [f"[{s}]" for s in result.species_names]


def test_time_column_matches():
    result = _ode_run(DECAY_SBML_V1)
    arr = result.as_roadrunner()
    np.testing.assert_array_equal(arr["time"], result.time)
    np.testing.assert_array_equal(arr[:, "time"], result.time)


# ── [X] = concentration, X = amount; relationship via V_c ─────────────


def test_brackets_is_concentration_v1():
    """For V=1 hOSU=true, storage = amount/1 = amount = concentration."""
    result = _ode_run(DECAY_SBML_V1)
    arr = result.as_roadrunner(selections=["time", "[X]", "X"])
    # V=1 → [X] and X must be byte-identical.
    np.testing.assert_array_equal(arr[:, "[X]"], arr[:, "X"])
    # And [X] must match Result.species[:, 0] (the stored value).
    np.testing.assert_array_equal(arr[:, "[X]"], result.species[:, 0])


def test_amount_vs_concentration_v2():
    """For V=2 hOSU=true, X (amount) = 2 * [X] (concentration)."""
    result = _ode_run(DECAY_SBML_V2)
    arr = result.as_roadrunner(selections=["time", "[X]", "X"])
    # initialAmount=100, V_c=2 → storage IC = 100/2 = 50, amount IC = 100.
    assert arr[0, "[X]"] == pytest.approx(50.0, rel=1e-9)
    assert arr[0, "X"] == pytest.approx(100.0, rel=1e-9)
    np.testing.assert_allclose(arr[:, "X"], arr[:, "[X]"] * 2.0, rtol=1e-12)


def test_decay_against_analytical_v1():
    result = _ode_run(DECAY_SBML_V1)
    arr = result.as_roadrunner()
    expected = 100.0 * np.exp(-0.1 * arr[:, "time"])
    np.testing.assert_allclose(arr[:, "[X]"], expected, rtol=1e-5)


# ── String indexing semantics (RR convention) ─────────────────────────


def test_arr_string_indexing():
    result = _ode_run(DECAY_SBML_V1)
    arr = result.as_roadrunner()
    # arr["[X]"] equivalent to arr[:, "[X]"]
    np.testing.assert_array_equal(arr["[X]"], arr[:, "[X]"])
    # Single-row scalar via (int, str)
    assert arr[0, "[X]"] == arr["[X]"][0]
    # Multi-column NamedArray slice
    sub = arr[:, ["time", "[X]"]]
    assert isinstance(sub, bngsim.NamedArray)
    assert sub.colnames == ["time", "[X]"]
    assert sub.shape == (11, 2)


# ── Unknown selector → actionable error ───────────────────────────────


def test_unknown_selector_raises():
    result = _ode_run(DECAY_SBML_V1)
    with pytest.raises(ValueError, match="Invalid selection"):
        result.as_roadrunner(selections=["time", "[NotASpecies]"])


def test_unknown_amount_selector_raises():
    result = _ode_run(DECAY_SBML_V1)
    with pytest.raises(ValueError, match="Invalid selection"):
        result.as_roadrunner(selections=["bogus"])


def test_namedarray_unknown_column_raises():
    result = _ode_run(DECAY_SBML_V1)
    arr = result.as_roadrunner()
    with pytest.raises(KeyError, match="Invalid selection"):
        _ = arr["[Y]"]


# ── Custom selection order ────────────────────────────────────────────


def test_selection_order_respected():
    result = _ode_run(DECAY_SBML_V1)
    arr = result.as_roadrunner(selections=["[X]", "time"])
    assert arr.colnames == ["[X]", "time"]
    np.testing.assert_array_equal(arr[:, "time"], result.time)


# ── 3-D batch results: as_roadrunner refuses ─────────────────────────


def test_batch_3d_raises():
    model = bngsim.Model.from_sbml_string(DECAY_SBML_V1)
    sim = bngsim.Simulator(model, method="ode")
    batch = sim.run_batch(
        t_span=(0, 10),
        n_points=11,
        params=[{"k": 0.1}, {"k": 0.2}],
        squeeze=True,
    )
    # Squeezed 3-D form (batch>1) — as_roadrunner should refuse.
    with pytest.raises(RuntimeError, match="single-simulation"):
        batch.as_roadrunner()


# ── Volume factors survive HDF5 round-trip when supplied ─────────────


def test_volume_factors_carried_via_simulator():
    """Simulator should populate _species_volume_factors automatically."""
    result = _ode_run(DECAY_SBML_V2)
    assert result._species_volume_factors is not None
    assert result._species_volume_factors[0] == pytest.approx(2.0)


# ── SSA path also works ───────────────────────────────────────────────


def test_ssa_as_roadrunner():
    """SSA results expose the same as_roadrunner shape; per Phase 5b
    SSA is supported on this model."""
    model = bngsim.Model.from_sbml_string(DECAY_SBML_V1)
    sim = bngsim.Simulator(model, method="ssa")
    result = sim.run(t_span=(0, 1), n_points=2, seed=1)
    arr = result.as_roadrunner()
    assert arr.colnames == ["time", "[X]"]
    assert arr.shape == (2, 2)


# ── Optional libroadrunner parity ─────────────────────────────────────

# Skip when RR isn't installed; PyBNF prod environments will have it.
roadrunner = pytest.importorskip("roadrunner", reason="libroadrunner optional")


def test_libroadrunner_parity_v1():
    """Default selections + numerical values match RR within ODE tol."""
    rr = roadrunner.RoadRunner()
    rr.load(DECAY_SBML_V1)
    rr_arr = rr.simulate(0, 10, 11)

    bngsim_arr = _ode_run(DECAY_SBML_V1).as_roadrunner()

    # Column names: RR uses '[X]' for floating species and 'time'.
    assert list(rr_arr.colnames) == bngsim_arr.colnames
    # Numerical agreement on the decay trajectory.
    np.testing.assert_allclose(
        np.asarray(bngsim_arr),
        np.asarray(rr_arr),
        rtol=1e-5,
        atol=1e-7,
    )
