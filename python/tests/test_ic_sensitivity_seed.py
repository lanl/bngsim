"""``Result.ic_sensitivity_seed``: the ∂x(0)/∂θ the parameter axis already carries.

``output_sensitivities(axis='parameter')`` is a **total** derivative — the
right-hand-side path plus the initial-condition seeding. ``axis='ic'`` is the
same trajectory differentiated w.r.t. an initial value held independent. For a
parameter that seeds an initial condition the two therefore overlap:

    d_param[θ] = (RHS path) + Σ_k (∂x_k(0)/∂θ)·d_ic[x_k]

A consumer that routes one fitted parameter to several native columns and sums
them (PyBNF, GH #155) double-counts the seeding unless it knows which rows this
engine already carries. That mapping is what ``ic_sensitivity_seed`` reports.

The tests below pin the *decomposition* against closed forms, not just the
plumbing: a mapping that agreed with the tensor's shape but not its arithmetic
would be exactly as useless as no mapping at all.
"""

import bngsim
import numpy as np
import pytest

# dS/dt = -(k + b)·S with S(0) = b·v0.  `b` reaches the trajectory through BOTH
# paths, which is the case no comparison of d_param against a single d_ic column
# can resolve.  S(t) = b·v0·e^{-(k+b)t}, so with a = k + b:
#     d_ic[u]     = e^{-a t}
#     ∂u(0)/∂b    = v0
#     d_param[b]  = v0·e^{-a t}  -  t·b·v0·e^{-a t}      (seeding + RHS)
SBML_BOTH_PATHS = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_both_paths">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="u" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="b" value="0.3" constant="true"/>
      <parameter id="v0" value="-60" constant="true"/>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="u">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>b</ci><ci>v0</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="u">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/>
            <apply><plus/><ci>k</ci><ci>b</ci></apply><ci>u</ci></apply></apply>
      </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""

# X(0) = a·X0, dX/dt = -k·X — a compound seed whose coefficients are neither 1
# nor equal to each other, and neither parameter touches the RHS.
SBML_COMPOUND = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_compound">
    <listOfParameters>
      <parameter id="X0" value="0.1" constant="true"/>
      <parameter id="a" value="3" constant="true"/>
      <parameter id="k" value="0.5" constant="true"/>
      <parameter id="X" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>a</ci><ci>X0</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>X</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""

B, V0, K = 0.3, -60.0, 0.5


def _run(sbml, params, ic, t_end=4.0, n=5, model=None):
    m = model if model is not None else bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=params, sensitivity_ic=ic)
    r = sim.run(t_span=(0, t_end), n_points=n, rtol=1e-12, atol=1e-14)
    return m, r


class TestReportsWhatWasSeeded:
    def test_compound_initial_assignment_reports_both_coefficients(self):
        """``X(0) = a*X0`` ⇒ ∂X(0)/∂a = X0 = 0.1 and ∂X(0)/∂X0 = a = 3."""
        _m, r = _run(SBML_COMPOUND, ["X0", "a"], ["X"])
        assert r.ic_sensitivity_seed == {"X": {"X0": pytest.approx(3.0), "a": pytest.approx(0.1)}}

    def test_reported_coefficients_are_the_ones_the_tensor_used(self):
        """The mapping is only worth having if it reproduces the overlap exactly:
        with neither parameter in the RHS, d_param must be the coefficient times
        the ic column, to round-off."""
        _m, r = _run(SBML_COMPOUND, ["X0", "a"], ["X"])
        d_ic = r.output_sensitivities("species:X", axis="ic")[:, 0, 0]
        d_param = r.output_sensitivities("species:X", axis="parameter")
        seed = r.ic_sensitivity_seed["X"]
        for i, p in enumerate(r.sensitivity_params):
            np.testing.assert_allclose(d_param[:, 0, i], seed[p] * d_ic, rtol=1e-9, atol=1e-14)

    def test_the_decomposition_holds_when_theta_drives_the_rhs_too(self):
        """The case the bit-comparison probe of GH #155 cannot see. Subtracting
        the reported seeding term must leave exactly the RHS derivative."""
        _m, r = _run(SBML_BOTH_PATHS, ["b"], ["u"])
        t = np.asarray(r.time)
        d_param = r.output_sensitivities("species:u", axis="parameter")[:, 0, 0]
        d_ic = r.output_sensitivities("species:u", axis="ic")[:, 0, 0]

        assert r.ic_sensitivity_seed == {"u": {"b": pytest.approx(V0)}}
        rhs_only = d_param - r.ic_sensitivity_seed["u"]["b"] * d_ic
        np.testing.assert_allclose(
            rhs_only, -t * B * V0 * np.exp(-(K + B) * t), rtol=1e-6, atol=1e-9
        )
        # And the probe the issue was reduced to genuinely fails here, so this
        # is not a case the old workaround already covered.
        assert np.max(np.abs(d_param - d_ic)) / np.max(np.abs(d_ic)) > 1.0

    def test_unit_seed_is_the_bit_identical_case(self):
        """A parameter that reaches the trajectory ONLY through the IC, with
        coefficient 1, gives two bit-identical columns — the Raia observation of
        GH #155. The mapping says so up front instead of by measurement."""
        sbml = SBML_COMPOUND.replace("<apply><times/><ci>a</ci><ci>X0</ci></apply>", "<ci>X0</ci>")
        _m, r = _run(sbml, ["X0"], ["X"])
        assert r.ic_sensitivity_seed == {"X": {"X0": pytest.approx(1.0)}}
        d_param = r.output_sensitivities("species:X", axis="parameter")[:, 0, 0]
        d_ic = r.output_sensitivities("species:X", axis="ic")[:, 0, 0]
        assert np.array_equal(d_param, d_ic)

    def test_a_parameter_that_does_not_seed_is_absent(self):
        """`k` is RHS-only, so it must not appear — absence is the positive
        statement 'this column needs your own IC term, if any'."""
        _m, r = _run(SBML_COMPOUND, ["X0", "k"], ["X"])
        assert r.ic_sensitivity_seed == {"X": {"X0": pytest.approx(3.0)}}

    def test_a_model_with_no_seeded_ic_reports_empty_not_none(self):
        """{} and None are different answers: {} is 'nothing seeds', None is
        'not recorded'. Conflating them is what forces a numeric probe."""
        sbml = SBML_COMPOUND.replace(
            """    <listOfInitialAssignments>
      <initialAssignment symbol="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>a</ci><ci>X0</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
""",
            "",
        ).replace(
            '<parameter id="X" constant="false"/>',
            '<parameter id="X" value="0.3" constant="false"/>',
        )
        _m, r = _run(sbml, ["k"], ["X"])
        assert r.ic_sensitivity_seed == {}

    def test_not_recorded_is_none(self):
        """An IC-only request has no `parameter` axis to describe."""
        _m, r = _run(SBML_COMPOUND, None, ["X"])
        assert r.ic_sensitivity_seed is None


class TestFollowsTheEngineRatherThanTheFile:
    def test_declared_row_replaces_the_parameter_graph_row(self):
        """issue #111: a hand-assigned IC is no longer described by the model's
        own expression, and the report must follow the engine, not the file."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        m.set_concentration("X", 0.7)
        m.declare_ic_sensitivity({"X": {"X0": 2.5}})
        _m, r = _run(None, ["X0", "a"], ["X"], model=m)
        assert r.ic_sensitivity_seed == {"X": {"X0": pytest.approx(2.5)}}

    def test_a_superseded_ic_retires_its_row(self):
        """issue #113: once an assignment moves the species off the initial
        condition the expression describes, the parameter no longer reaches it —
        and d_param really is RHS-only, so the report must be empty."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        m.set_concentration("X", 0.7)
        _m, r = _run(None, ["X0", "a"], ["X"], model=m)
        assert r.ic_sensitivity_seed == {}
        # Not a vacuous assertion: with no seeding, both columns are zero,
        # because X0 and a reach the trajectory only through the IC.
        d_param = r.output_sensitivities("species:X", axis="parameter")
        assert np.max(np.abs(d_param)) == 0.0

    def test_round_trips_through_declare_ic_sensitivity(self):
        """The report is shaped as declare_ic_sensitivity's input on purpose:
        re-declaring what was reported must reproduce the same gradients."""
        _m, base = _run(SBML_COMPOUND, ["X0", "a"], ["X"])
        m2 = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        m2.declare_ic_sensitivity(base.ic_sensitivity_seed)
        _m2, redeclared = _run(None, ["X0", "a"], ["X"], model=m2)
        assert redeclared.ic_sensitivity_seed == base.ic_sensitivity_seed
        np.testing.assert_array_equal(
            redeclared.output_sensitivities("species:X"),
            base.output_sensitivities("species:X"),
        )


class TestOtherResultPaths:
    def test_batch_rows_carry_their_own_point_dependent_matrix(self):
        """A nonlinear derived IC makes the coefficient move with the point, so
        a per-row matrix is the only correct answer — ∂X(0)/∂X0 = a."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["X0"])
        rows = sim.run_batch((0, 2), 3, params=[{"a": 3.0}, {"a": 5.0}], squeeze=False)
        assert [r.ic_sensitivity_seed for r in rows] == [
            {"X": {"X0": pytest.approx(3.0)}},
            {"X": {"X0": pytest.approx(5.0)}},
        ]

    def test_squeeze_refuses_to_pick_one_when_rows_disagree(self):
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["X0"])
        squeezed = sim.run_batch((0, 2), 3, params=[{"a": 3.0}, {"a": 5.0}], squeeze=True)
        assert squeezed.ic_sensitivity_seed is None

    def test_squeeze_keeps_an_agreed_matrix(self):
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["X0"])
        squeezed = sim.run_batch((0, 2), 3, params=[{"k": 0.5}, {"k": 0.9}], squeeze=True)
        assert squeezed.ic_sensitivity_seed == {"X": {"X0": pytest.approx(3.0)}}

    def test_chunked_path_reports_the_union_over_chunks(self):
        """compute_all_sensitivities partitions the parameter list; each chunk
        sees only its own subset, so the stitched matrix must be their union."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["X0", "a", "k"])
        r = sim.compute_all_sensitivities(t_span=(0, 2), n_points=3, chunk_size=1)
        assert r.ic_sensitivity_seed == {"X": {"X0": pytest.approx(3.0), "a": pytest.approx(0.1)}}

    def test_a_carried_seed_run_reports_none_not_the_unused_rows(self):
        """GH #210/#81: a measurement phase started from a carried dx/dθ does NOT
        seed from the model's initial conditions — the engine discards those rows.
        Reporting them would describe a seed that was never used, which is the
        exact failure mode this property exists to prevent."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["X0"])
        fresh = sim.run(t_span=(0, 1), n_points=3)
        assert fresh.ic_sensitivity_seed == {"X": {"X0": pytest.approx(3.0)}}
        carried = sim.run(t_span=(1, 2), n_points=3, carry_sensitivities=True)
        assert carried.ic_sensitivity_seed is None


class TestModelAccessor:
    """The accessor a fitting frontend actually needs: answerable at *setup*,
    from model structure alone, with no simulation (GH #155 requirement 1)."""

    def test_answers_without_running_anything(self):
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        assert m.effective_ic_sensitivity(["X0", "a"]) == {
            "X": {"X0": pytest.approx(3.0), "a": pytest.approx(0.1)}
        }

    def test_defaults_to_every_parameter(self):
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        assert m.effective_ic_sensitivity() == {
            "X": {"X0": pytest.approx(3.0), "a": pytest.approx(0.1)}
        }

    def test_restricts_to_the_requested_parameters(self):
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        assert m.effective_ic_sensitivity(["a"]) == {"X": {"a": pytest.approx(0.1)}}

    def test_reports_original_symbols_never_the_synthetic_carrier(self):
        """GH #155 requirement 3: issue #147 lowers a compound initialAssignment
        onto a synthetic `_ic_<species>` derived parameter. A frontend binds its
        fit parameters to model ids by name, so the report must name `a`/`X0`."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        assert any(n.startswith("_ic_") for n in m.param_names)  # the carrier exists
        reported = {p for row in m.effective_ic_sensitivity().values() for p in row}
        assert reported == {"X0", "a"}
        assert not any(p.startswith("_ic_") for p in reported)

    def test_a_declared_zero_is_present_not_absent(self):
        """GH #155 requirement 2: 'seeded, coefficient zero at this state' and
        'no seeding path at all' are different answers. A chain-rule factor that
        merely vanishes here must not read as a column that does not exist."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        m.declare_ic_sensitivity({"X": {"X0": 0.0}})
        eff = m.effective_ic_sensitivity(["X0", "a"])
        assert eff == {"X": {"X0": 0.0}}
        assert "X0" in eff["X"]  # present...
        assert "a" not in eff["X"]  # ...where `a` is genuinely absent

    def test_a_vanishing_parameter_graph_coefficient_is_present_too(self):
        """The same distinction without a declaration: ∂X(0)/∂X0 = a, so setting
        a = 0 makes the coefficient vanish while the seeding path still exists."""
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        m.set_param("a", 0.0)
        eff = m.effective_ic_sensitivity(["X0", "a"])
        assert eff["X"]["X0"] == 0.0
        assert eff["X"]["a"] == pytest.approx(0.1)

    def test_tracks_state_the_way_the_engine_does(self):
        m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
        m.set_concentration("X", 0.7)  # issue #113 supersedes the IC expression
        assert m.effective_ic_sensitivity() == {}
        m.reset()
        assert m.effective_ic_sensitivity(["X0"]) == {"X": {"X0": pytest.approx(3.0)}}


def test_model_and_result_report_the_same_matrix():
    """The invariant that keeps the reader from drifting from the seeding: both
    read one derivation (`Model._ic_sensitivity_triples`), so a change to either
    that did not change the other would break here rather than silently ship a
    matrix describing a seed the solver never used."""
    m = bngsim.Model.from_sbml_string(SBML_COMPOUND)
    at_setup = m.effective_ic_sensitivity(["X0", "a"])
    _m, r = _run(None, ["X0", "a"], ["X"], model=m)
    assert r.ic_sensitivity_seed == at_setup


def test_capability_flag_is_advertised():
    caps = bngsim.capabilities()
    assert caps["features"]["effective_ic_sensitivity"] is True
