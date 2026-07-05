"""GH #205 (b): output sensitivities for SBML AssignmentRule-target species.

An AssignmentRule-target species is emitted ``fixed`` — the loader zeroes its
ODE derivative and the value path overwrites its column from the rule's live
value (an *observable* for a linear-on-species rule, GH #197; a *function /
expression* otherwise, GH #198). The raw integrated forward-sensitivity ``yS``
for that frozen slot is therefore meaningless (~0). So ``species:<name>`` output
sensitivity must follow the **assignment expression**: the sensitivity of the
rule's observable/expression, not the raw state sensitivity. The raw tensor
stays available as the low-level ``Result.sensitivities_species``.

``from_sbml`` + sensitivities is a never-before-tested intersection, so these are
authored from scratch. Oracles, in order of authority:

  * **Analytic** — A decays as ``A(t) = A0·e^(-kd·t)`` (kd-only), so the rule
    derivatives are closed-form: linear ``S = A`` ⇒ ``dS/dkd = -A0·t·e^(-kd·t)``,
    ``dS/dk = 0``; nonlinear ``S2 = A²`` ⇒ ``dS2/dkd = 2·A·dA/dkd``.
  * **Finite difference** — central FD of the *emitted* (overwritten) value
    column wrt kd; this is the primary numeric oracle (matches the DoD: "output
    sensitivity matches emitted value").
  * **roadrunner** — optional value cross-check (``importorskip``); it is the
    natural AR oracle but flaky on this machine, so it only sanity-checks the
    value the FD oracle differentiates, never gates the suite.
"""

import bngsim
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _force_codegen(monkeypatch):
    """A nonlinear AssignmentRule is emitted as a function, whose output
    sensitivity (GH #198) needs the compiled ``.so``. Force codegen on for every
    test here; monkeypatch restores the environment afterwards."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)


def _ar_sbml(rule_math: str, sid: str) -> str:
    """A → ∅ (law kd·A); B synthesised at k·<sid>; <sid> set by an AssignmentRule.

    ``rule_math`` is the MathML for the rule body; ``sid`` is the rule-target
    species id (``S`` for the linear case, ``S2`` for the nonlinear case).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="{sid}" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="kd" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="{sid}">
        <math xmlns="http://www.w3.org/1998/Math/MathML">{rule_math}</math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kd</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="synthB" reversible="false">
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <listOfModifiers>
          <modifierSpeciesReference species="{sid}"/>
        </listOfModifiers>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>{sid}</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


LINEAR = _ar_sbml("<ci>A</ci>", "S")
NONLINEAR = _ar_sbml("<apply><times/><ci>A</ci><ci>A</ci></apply>", "S2")

A0, K, KD = 10.0, 0.3, 0.5
T_SPAN, N = (0.0, 10.0), 21
_RUN = dict(rtol=1e-10, atol=1e-12, max_steps=10**6)
_FD = dict(rtol=1e-11, atol=1e-13, max_steps=10**6)


def _sim(sbml, params=("k", "kd")):
    m = bngsim.Model.from_sbml_string(sbml)
    return bngsim.Simulator(m, method="ode", sensitivity_params=list(params))


def _value_kd(sbml, sid, kd):
    """Emitted (rule-overwritten) value column of *sid* at parameter kd."""
    m = bngsim.Model.from_sbml_string(sbml)
    m.set_param("kd", kd)
    r = bngsim.Simulator(m, method="ode").run(t_span=T_SPAN, n_points=N, **_FD)
    return np.asarray(r.outputs(f"species:{sid}"))[:, 0]


def _fd_kd(sbml, sid, h=5e-7):
    return (_value_kd(sbml, sid, KD + h) - _value_kd(sbml, sid, KD - h)) / (2 * h)


def _assert_close(analytic, reference, *, rtol=2e-4, atol=1e-6):
    """Scale-relative comparison so a genuinely-zero derivative (whose FD is pure
    solver noise) is not flagged by a divide-by-near-zero relative error."""
    analytic = np.asarray(analytic)
    reference = np.asarray(reference)
    scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(analytic))))
    tol = atol + rtol * scale
    err = float(np.max(np.abs(analytic - reference)))
    assert err <= tol, f"max abs err {err:.3e} > tol {tol:.3e}"


# ── Linear-on-species rule → observable redirect (GH #197) ──────────────────


class TestLinearARSpecies:
    def test_ar_sens_map_records_observable(self):
        r = _sim(LINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        # Linear rule S = A is emitted as an observable named S.
        assert r._ar_sens_map == {"S": ("observable", "S", 1.0)}

    def test_species_selector_follows_rule_analytic(self):
        r = _sim(LINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        t = np.asarray(r.time)
        os_ = r.output_sensitivities("species:S")  # (N, 1, 2)
        ik = r.sensitivity_params.index("k")
        ikd = r.sensitivity_params.index("kd")
        # dS/dkd = dA/dkd = -A0·t·e^(-kd·t); dS/dk = 0 (A decays via kd only).
        _assert_close(os_[:, 0, ikd], -A0 * t * np.exp(-KD * t))
        assert np.max(np.abs(os_[:, 0, ik])) == pytest.approx(0.0, abs=1e-9)

    def test_species_selector_matches_fd(self):
        r = _sim(LINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        ikd = r.sensitivity_params.index("kd")
        os_ = r.output_sensitivities("species:S")
        _assert_close(os_[:, 0, ikd], _fd_kd(LINEAR, "S"))

    def test_redirect_equals_observable_block(self):
        # The species selector returns exactly the observable's sensitivity.
        r = _sim(LINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        np.testing.assert_array_equal(
            r.output_sensitivities("species:S"),
            r.output_sensitivities("observable:S"),
        )

    def test_raw_state_sensitivity_is_low_level(self):
        # Contract: the raw integrated yS for the frozen AR slot stays the
        # low-level tensor (~0 here) and is NOT what species: returns.
        r = _sim(LINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        iS = list(r.species_names).index("S")
        assert np.max(np.abs(r.sensitivities_species[:, iS, :])) == pytest.approx(0.0, abs=1e-9)
        # ... and it genuinely differs from the rule-following output.
        assert np.max(np.abs(r.output_sensitivities("species:S")[:, 0, :])) > 1.0


# ── Nonlinear rule → function/expression redirect (GH #198, codegen) ────────


class TestNonlinearARSpecies:
    def test_ar_sens_map_records_expression(self):
        r = _sim(NONLINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        # Nonlinear rule S2 = A² is emitted as a function/expression named S2.
        assert r._ar_sens_map == {"S2": ("expression", "S2", 1.0)}
        assert r.has_sensitivities_expressions

    def test_species_selector_follows_rule_analytic(self):
        r = _sim(NONLINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        t = np.asarray(r.time)
        A = A0 * np.exp(-KD * t)
        os_ = r.output_sensitivities("species:S2")
        ik = r.sensitivity_params.index("k")
        ikd = r.sensitivity_params.index("kd")
        # dS2/dkd = 2·A·dA/dkd = 2·A·(-A0·t·e^(-kd·t)); dS2/dk = 0.
        _assert_close(os_[:, 0, ikd], 2.0 * A * (-A0 * t * np.exp(-KD * t)))
        assert np.max(np.abs(os_[:, 0, ik])) == pytest.approx(0.0, abs=1e-9)

    def test_species_selector_matches_fd(self):
        r = _sim(NONLINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        ikd = r.sensitivity_params.index("kd")
        os_ = r.output_sensitivities("species:S2")
        _assert_close(os_[:, 0, ikd], _fd_kd(NONLINEAR, "S2"))

    def test_redirect_equals_expression_block(self):
        r = _sim(NONLINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        np.testing.assert_array_equal(
            r.output_sensitivities("species:S2"),
            r.output_sensitivities("expression:S2"),
        )

    def test_raw_state_sensitivity_is_low_level(self):
        r = _sim(NONLINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        iS2 = list(r.species_names).index("S2")
        assert np.max(np.abs(r.sensitivities_species[:, iS2, :])) == pytest.approx(0.0, abs=1e-9)
        assert np.max(np.abs(r.output_sensitivities("species:S2")[:, 0, :])) > 1.0


# ── compute_all_sensitivities (stitched) carries the redirect ───────────────


class TestComputeAllSensitivitiesRedirect:
    """The AR redirect map and its source sensitivity block survive the parallel
    chunk → stitch path of compute_all_sensitivities (GH #205). The sim is built
    WITHOUT sensitivity_params (that entry point's convention), so the map must
    ride through ``_stitch_sensitivity_results`` from a stamped chunk."""

    def test_linear_observable_redirect_stitched(self):
        # The observable block (GH #197) is a runtime chain rule needing no
        # codegen, so it stitches on the interpreted path — this isolates the
        # _ar_sens_map propagation through the stitch.
        m = bngsim.Model.from_sbml_string(LINEAR)
        sim = bngsim.Simulator(m, method="ode")
        r = sim.compute_all_sensitivities(
            t_span=T_SPAN, n_points=N, params=["k", "kd"], chunk_size=1, **_RUN
        )
        assert r._ar_sens_map == {"S": ("observable", "S", 1.0)}
        ikd = r.sensitivity_params.index("kd")
        t = np.asarray(r.time)
        _assert_close(r.output_sensitivities("species:S")[:, 0, ikd], -A0 * t * np.exp(-KD * t))

    def test_nonlinear_expression_redirect_stitched(self, monkeypatch):
        # The expression block (GH #198) needs codegen WITH the output-sensitivity
        # evaluator. The autouse fixture's threshold=1 would force a plain
        # construction-time codegen .so that shadows it (the no-op guard), so drop
        # the threshold: construction stays interpreted and
        # compute_all_sensitivities owns the codegen — exercising the GH #205
        # ``_want_output_sens`` fix that makes the SBML path emit output sens.
        monkeypatch.delenv("BNGSIM_CODEGEN_THRESHOLD", raising=False)
        m = bngsim.Model.from_sbml_string(NONLINEAR)
        sim = bngsim.Simulator(m, method="ode")
        assert not sim._codegen_so_path  # construction stayed interpreted
        r = sim.compute_all_sensitivities(
            t_span=T_SPAN, n_points=N, params=["k", "kd"], chunk_size=1, **_RUN
        )
        assert r.has_sensitivities_expressions
        assert r._ar_sens_map == {"S2": ("expression", "S2", 1.0)}
        ikd = r.sensitivity_params.index("kd")
        t = np.asarray(r.time)
        A = A0 * np.exp(-KD * t)
        _assert_close(
            r.output_sensitivities("species:S2")[:, 0, ikd],
            2.0 * A * (-A0 * t * np.exp(-KD * t)),
        )

    def test_nonlinear_redirect_when_plain_codegen_preattached(self):
        # GH #205 follow-up: when a plain-RHS codegen artifact is ALREADY attached
        # at construction (here pinned via codegen=True; also happens for a
        # large model above the species threshold, or an inherited .so), it was
        # built without the output-sensitivity evaluator (_want_output_sens was
        # False with no sensitivity_params). compute_all_sensitivities must
        # regenerate it WITH output sens rather than no-op on the plain artifact —
        # otherwise the expression block (and the nonlinear-AR species: redirect)
        # comes back empty.
        m = bngsim.Model.from_sbml_string(NONLINEAR)
        sim = bngsim.Simulator(m, method="ode", codegen=True)
        assert sim._codegen_so_path  # plain-RHS codegen pinned at construction
        r = sim.compute_all_sensitivities(
            t_span=T_SPAN, n_points=N, params=["k", "kd"], chunk_size=1, **_RUN
        )
        assert r.has_sensitivities_expressions
        assert r._ar_sens_map == {"S2": ("expression", "S2", 1.0)}
        ikd = r.sensitivity_params.index("kd")
        t = np.asarray(r.time)
        A = A0 * np.exp(-KD * t)
        _assert_close(
            r.output_sensitivities("species:S2")[:, 0, ikd],
            2.0 * A * (-A0 * t * np.exp(-KD * t)),
        )


# ── Non-AR species are untouched by the redirect (regression guard) ─────────


class TestNonARSpeciesUnaffected:
    @pytest.mark.parametrize("name", ["A", "B"])
    def test_plain_species_selector_reads_raw_block(self, name):
        r = _sim(NONLINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        i = list(r.species_names).index(name)
        # A and B are ordinary integrated species: species:<name> is exactly the
        # raw species sensitivity slice, no redirect.
        np.testing.assert_array_equal(
            r.output_sensitivities(f"species:{name}")[:, 0, :],
            r.sensitivities_species[:, i, :],
        )


# ── Optional roadrunner cross-check of the value the FD oracle differentiates ─


class TestRoadrunnerValueCrossCheck:
    @pytest.mark.parametrize(
        "sbml,sid,closed_form",
        [
            (LINEAR, "S", lambda t: A0 * np.exp(-KD * t)),
            (NONLINEAR, "S2", lambda t: (A0 * np.exp(-KD * t)) ** 2),
        ],
    )
    def test_emitted_value_matches_roadrunner(self, sbml, sid, closed_form):
        rr = pytest.importorskip("roadrunner")
        r = _sim(sbml).run(t_span=T_SPAN, n_points=N, **_RUN)
        ours = np.asarray(r.outputs(f"species:{sid}"))[:, 0]
        # Confirm our emitted value tracks the closed form (so FD of it is a
        # trustworthy sensitivity oracle); roadrunner is the independent witness.
        t = np.asarray(r.time)
        np.testing.assert_allclose(ours, closed_form(t), rtol=1e-6, atol=1e-8)
        try:
            sim = rr.RoadRunner(sbml)
            sim.timeCourseSelections = ["time", sid]
            data = sim.simulate(T_SPAN[0], T_SPAN[1], N)
        except Exception as e:  # pragma: no cover - RR flaky on this machine
            pytest.skip(f"roadrunner failed on this model: {e}")
        # Loose tolerance: roadrunner runs its own integrator at its own default
        # tolerances, so this is a sanity cross-check of the value, not a
        # precision comparison (the analytic/FD asserts above are the oracle).
        np.testing.assert_allclose(np.asarray(data[:, 1]), ours, rtol=1e-4, atol=1e-6)
