"""GH #205 (b) / GH #221: sensitivities for SBML AssignmentRule-target species.

An AssignmentRule-target species is emitted ``fixed`` — the loader zeroes its
ODE derivative and the value path overwrites its column from the rule's live
value (an *observable* for a linear-on-species rule, GH #197; a *function /
expression* otherwise, GH #198). The raw integrated forward-sensitivity ``yS``
for that frozen slot is therefore meaningless: identically zero, because the
frozen slot's variational RHS is zero too. So the derivative must follow the
**assignment expression**: the sensitivity of the rule's observable/expression.

GH #205 did that for the ``species:<name>`` *selector*
(``Result.output_sensitivities``). GH #221 does it for the
``Result.sensitivities`` **tensor**, which is what ``Result.gradient`` /
``Result.sse_gradient`` contract ``dL/dY`` against row-for-row — so before it,
a fit scoring an assignment-rule species (``IRS_total``, ``InR_active``: the
*reported* quantities of an SBML model, which is what assignment rules are for)
read a gradient that was zero in every direction, and could not tell that from a
flat objective. Where the chain rule is unavailable the row is ``NaN`` and the
run warns, never a structural zero.

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

import warnings

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

    def test_tensor_row_follows_the_rule(self):
        # GH #221 — the tensor row IS the redirect now, not the frozen slot's yS
        # (which is identically 0 and used to sit here silently).
        r = _sim(LINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        iS = list(r.species_names).index("S")
        np.testing.assert_array_equal(
            r.sensitivities_species[:, iS, :],
            r.output_sensitivities("species:S")[:, 0, :],
        )
        assert np.max(np.abs(r.sensitivities_species[:, iS, :])) > 1.0


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

    def test_tensor_row_follows_the_rule(self):
        r = _sim(NONLINEAR).run(t_span=T_SPAN, n_points=N, **_RUN)
        iS2 = list(r.species_names).index("S2")
        np.testing.assert_array_equal(
            r.sensitivities_species[:, iS2, :],
            r.output_sensitivities("species:S2")[:, 0, :],
        )
        assert np.max(np.abs(r.sensitivities_species[:, iS2, :])) > 1.0


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


# ── GH #221: the sensitivity TENSOR row, and what reads it ──────────────────
#
# The selector API was already right after GH #205. What #221 is about is the
# tensor: Result.gradient / sse_gradient / fisher_information never go through a
# selector, they contract the raw (n_times, n_species, n_params) block, so an
# untouched AR row made the objective look flat in every direction.


_RULES = [
    pytest.param(LINEAR, "S", id="linear"),
    pytest.param(NONLINEAR, "S2", id="nonlinear"),
]


class TestARRowInTheTensor:
    @pytest.mark.parametrize(
        "sbml,sid,closed_form",
        [
            # dS/dkd  = dA/dkd     = -A0·t·e^(-kd·t)
            # dS2/dkd = 2·A·dA/dkd = 2·(A0·e^(-kd·t))·(-A0·t·e^(-kd·t))
            pytest.param(LINEAR, "S", lambda t: -A0 * t * np.exp(-KD * t), id="linear"),
            pytest.param(
                NONLINEAR,
                "S2",
                lambda t: 2.0 * A0 * np.exp(-KD * t) * (-A0 * t * np.exp(-KD * t)),
                id="nonlinear",
            ),
        ],
    )
    def test_tensor_row_matches_the_closed_form(self, sbml, sid, closed_form):
        r = _sim(sbml).run(t_span=T_SPAN, n_points=N, **_RUN)
        i = list(r.species_names).index(sid)
        ikd = r.sensitivity_params.index("kd")
        _assert_close(r.sensitivities[:, i, ikd], closed_form(np.asarray(r.time)))

    @pytest.mark.parametrize("sbml,sid", _RULES)
    def test_tensor_row_matches_fd_of_the_reported_value(self, sbml, sid):
        # The independent oracle: central difference of the column the value path
        # actually emits. It never touches the sensitivity machinery.
        r = _sim(sbml).run(t_span=T_SPAN, n_points=N, **_RUN)
        i = list(r.species_names).index(sid)
        ikd = r.sensitivity_params.index("kd")
        _assert_close(r.sensitivities[:, i, ikd], _fd_kd(sbml, sid))

    @pytest.mark.parametrize("sbml,sid", _RULES)
    def test_gradient_over_the_ar_species_is_not_flat(self, sbml, sid):
        """The reported harm: an SSE gradient scored on the AR species alone.

        Oracle is a central difference of the loss itself — no sensitivity
        tensor on either side of the comparison, so this pins the number a
        fitter would read, not just the tensor's self-consistency.
        """
        r = _sim(sbml).run(t_span=T_SPAN, n_points=N, **_RUN)
        i = list(r.species_names).index(sid)
        target = np.zeros(N)  # L(kd) = Σ_t value(t; kd)²

        def loss_at(kd):
            return float(np.sum((_value_kd(sbml, sid, kd) - target) ** 2))

        def dL_dY(species, time):
            g = np.zeros_like(species)
            g[:, i] = 2.0 * (species[:, i] - target)
            return g

        grad = r.gradient(dL_dY)
        ikd = r.sensitivity_params.index("kd")
        h = 5e-6
        fd = (loss_at(KD + h) - loss_at(KD - h)) / (2 * h)
        assert abs(grad[ikd]) > 0.0
        assert grad[ikd] == pytest.approx(fd, rel=1e-4)

    def test_ic_axis_row_follows_the_rule_too(self):
        # ∂S/∂A(0) = ∂A/∂A(0) = e^(-kd·t) for the linear rule S = A.
        m = bngsim.Model.from_sbml_string(LINEAR)
        r = bngsim.Simulator(m, method="ode", sensitivity_ic=["A"]).run(
            t_span=T_SPAN, n_points=N, **_RUN
        )
        i = list(r.species_names).index("S")
        _assert_close(r.sensitivities_ic[:, i, 0], np.exp(-KD * np.asarray(r.time)))
        np.testing.assert_array_equal(
            r.sensitivities_ic[:, i, :],
            r.output_sensitivities("species:S", axis="ic")[:, 0, :],
        )

    @pytest.mark.parametrize("sbml,sid", _RULES)
    def test_only_the_ar_row_moved(self, sbml, sid):
        """Every non-AR row is bit-identical to the block the integrator wrote.

        The pass copies into a fresh array, so "it only touched one row" is a
        claim worth pinning rather than assuming: rebuild the untouched block by
        stacking the ordinary species' own selector slices.
        """
        r = _sim(sbml).run(t_span=T_SPAN, n_points=N, **_RUN)
        names = list(r.species_names)
        for name in names:
            if name == sid:
                continue
            i = names.index(name)
            np.testing.assert_array_equal(
                r.sensitivities[:, i, :], r.output_sensitivities(f"species:{name}")[:, 0, :]
            )

    def test_stitched_result_carries_the_row(self):
        # compute_all_sensitivities fills each chunk's AR row from that chunk's
        # own observable block, then concatenates along the parameter axis.
        m = bngsim.Model.from_sbml_string(LINEAR)
        r = bngsim.Simulator(m, method="ode").compute_all_sensitivities(
            t_span=T_SPAN, n_points=N, params=["k", "kd"], chunk_size=1, **_RUN
        )
        i = list(r.species_names).index("S")
        ikd = r.sensitivity_params.index("kd")
        t = np.asarray(r.time)
        _assert_close(r.sensitivities[:, i, ikd], -A0 * t * np.exp(-KD * t))
        np.testing.assert_array_equal(
            r.sensitivities[:, i, :], r.output_sensitivities("species:S")[:, 0, :]
        )


# ── GH #221: refusal — NaN and a named warning, never a structural zero ─────

# ``T`` is an hOSU=true AssignmentRule target in a compartment driven by a rate
# rule, so its reported value is (rule / vdiv) · V_static/V_live(t): the redirect
# models only the constant vdiv, and the missing d V_live/dθ makes the row wrong
# rather than merely imprecise. GH #205 refuses the selector by name; #221 must
# not leave the tensor row at 0.0, which no consumer could tell from a
# measurement.
BLOCKED_VARVOL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_varvol">
    <listOfCompartments><compartment id="C" size="1" constant="false"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="T" compartment="C" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="g" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><ci>g</ci><ci>C</ci></apply>
      </math></rateRule>
      <assignmentRule variable="T"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>3</cn><ci>A</ci></apply>
      </math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


class TestRefusedARRow:
    def _run(self):
        """Run the blocked model, letting the refusal warning through quietly —
        ``test_warns_naming_the_species`` is what pins the warning itself, so the
        other assertions here stay independent of it."""
        m = bngsim.Model.from_sbml_string(BLOCKED_VARVOL)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k", "g"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sim.run(t_span=(0.0, 5.0), n_points=6, **_RUN)

    def test_warns_naming_the_species(self):
        m = bngsim.Model.from_sbml_string(BLOCKED_VARVOL)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k", "g"])
        with pytest.warns(UserWarning, match=r"Assignment-rule species \['T'\].*NaN"):
            sim.run(t_span=(0.0, 5.0), n_points=6, **_RUN)

    def test_the_fixture_is_actually_blocked(self):
        # Guard the fixture itself: if the loader ever stops classifying T as a
        # variable-volume AR species, every assertion below passes vacuously.
        m = bngsim.Model.from_sbml_string(BLOCKED_VARVOL)
        assert "T" in (getattr(m, "_ar_report_map", None) or {})
        assert "T" in (getattr(m, "_varvol_conc_map", None) or {})

    def test_row_is_nan_not_zero(self):
        r = self._run()
        i = list(r.species_names).index("T")
        assert np.isnan(r.sensitivities[:, i, :]).all()
        assert r.ar_sensitivity_refused == frozenset({"T"})

    def test_the_declared_nan_does_not_trip_the_384_refusal(self):
        """Issue #384 refuses a tensor that comes back non-finite. This row is
        non-finite ON PURPOSE, so the two must not collide.

        They do not, structurally rather than by exclusion list: #384 checks
        inside the solver, and a declared row is written by the Python layer
        afterwards. Asserted here because that ordering is the whole reason the
        check needs no knowledge of this feature, and a future move of either
        one would break it silently — this run would start raising."""
        assert self._run() is not None

    def test_ordinary_rows_survive_the_refusal(self):
        r = self._run()
        i = list(r.species_names).index("A")
        assert np.isfinite(r.sensitivities[:, i, :]).all()
        assert np.max(np.abs(r.sensitivities[:, i, :])) > 0.0

    def test_selector_still_raises_by_name(self):
        r = self._run()
        with pytest.raises(ValueError, match="time-varying volume rescale"):
            r.output_sensitivities("species:T")

    def test_a_loss_that_ignores_the_refused_row_still_gets_a_gradient(self):
        """IEEE makes ``0 · NaN`` NaN, so one refused row would otherwise poison
        every parameter of a fit that never scored that species."""
        r = self._run()
        names = list(r.species_names)
        iA, iT = names.index("A"), names.index("T")

        def dL_dY(species, time):
            g = np.zeros_like(species)
            g[:, iA] = 2.0 * species[:, iA]  # weight A only
            return g

        grad = r.gradient(dL_dY)
        assert np.isfinite(grad).all()
        # And it is the same number the refused row's absence would give.
        expected = np.einsum(
            "tsi,ts->i", r.sensitivities[:, [iA], :], dL_dY(np.asarray(r.species), r.time)[:, [iA]]
        )
        np.testing.assert_array_equal(grad, expected)
        assert np.max(np.abs(grad)) > 0.0
        # sse_gradient's own way of saying "not this species" agrees.
        data = np.zeros((len(r.time), 1))
        _, sub = r.sse_gradient(data, species_indices=[iA])
        assert np.isfinite(sub).all()
        assert iT != iA  # the refused row really is in the full array

    def test_fisher_information_has_no_unweighted_row_to_drop(self):
        """The FIM weights every output it is built over, so the exemption the
        gradients get does not apply — naming the outputs is the way out, and
        that is what the docstring promises."""
        r = self._run()
        assert np.isnan(r.fisher_information()).all()
        assert np.isfinite(r.fisher_information(outputs=["species:A"])).all()

    def test_a_loss_that_weights_the_refused_row_gets_nan(self):
        # The other half: an unknown derivative the objective actually depends on
        # must not come back looking like a number.
        r = self._run()
        iT = list(r.species_names).index("T")

        def dL_dY(species, time):
            g = np.zeros_like(species)
            g[:, iT] = 1.0
            return g

        assert np.isnan(r.gradient(dL_dY)).all()


# ``F``'s rule is a piecewise, which codegen lowers to an `if()` and then
# declines to differentiate (GH #198 refuses rather than guess). The rule's own
# expression row is already NaN; what #221 adds is that the SPECIES row stops
# disagreeing with it. This is the common refusal on the corpus — 50 of the 639
# AR rows across 215 models, versus 15 blocked by a variable volume — and it is
# the one the issue's `IRS_total`-style report does not cover.
DECLINED_PIECEWISE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_declined">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="F" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kd" value="0.5" constant="true"/>
      <parameter id="ton" value="5" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="F"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <piecewise>
          <piece><apply><times/><cn>2</cn><ci>A</ci></apply>
            <apply><lt/><csymbol encoding="text"
              definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><ci>ton</ci></apply>
          </piece>
          <otherwise><ci>A</ci></otherwise>
        </piecewise>
      </math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>kd</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


class TestDeclinedExpressionARRow:
    def _run(self):
        m = bngsim.Model.from_sbml_string(DECLINED_PIECEWISE)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kd"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sim.run(t_span=(0.0, 10.0), n_points=6, **_RUN)

    def test_the_fixture_really_is_declined(self):
        r = self._run()
        assert r._ar_sens_map["F"][0] == "expression"
        assert "if()" in (r._expression_sens_support.get("F") or "")

    def test_species_row_agrees_with_the_expression_row_it_mirrors(self):
        r = self._run()
        i = list(r.species_names).index("F")
        j = list(r.expression_names).index("F")
        assert np.isnan(r.sensitivities_expressions[:, j, :]).all()
        assert np.isnan(r.sensitivities[:, i, :]).all()
        assert r.ar_sensitivity_refused == frozenset({"F"})

    def test_selector_reports_the_construct_that_declined(self):
        r = self._run()
        with pytest.raises(ValueError, match=r"if\(\) conditional"):
            r.output_sensitivities("species:F")

    def test_the_integrated_species_is_unaffected(self):
        r = self._run()
        i = list(r.species_names).index("A")
        _assert_close(
            r.sensitivities[:, i, 0], -A0 * np.asarray(r.time) * np.exp(-KD * np.asarray(r.time))
        )


# ── GH #221: one vdiv, resolved live (#170 writable compartment size) ───────


class TestWritableVolumeDivisor:
    """``vdiv`` is ``V_c(target)`` for an hOSU=true AR species, and #170 makes a
    compartment size writable — so the value pass reads it live. The redirect
    used to keep the *load-time* number, which put the reported value and its
    derivative out by exactly the write's factor. One resolution site now."""

    _SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_writable_v">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="T" compartment="C" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="T"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>3</cn><ci>A</ci></apply>
      </math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
    _V_NEW = 3.0
    _T = (0.0, 5.0)
    _N = 6

    def _model(self, k=0.3):
        m = bngsim.Model.from_sbml_string(self._SBML)
        m.set_param("C", self._V_NEW)
        m.set_param("k", k)
        return m

    def test_value_and_derivative_use_the_same_divisor(self):
        r = bngsim.Simulator(self._model(), method="ode", sensitivity_params=["k"]).run(
            t_span=self._T, n_points=self._N, **_RUN
        )
        i = list(r.species_names).index("T")
        assert r._ar_sens_map["T"][2] == pytest.approx(self._V_NEW)
        np.testing.assert_array_equal(
            r.sensitivities[:, i, :], r.output_sensitivities("species:T")[:, 0, :]
        )

    def test_matches_fd_of_the_written_model(self):
        r = bngsim.Simulator(self._model(), method="ode", sensitivity_params=["k"]).run(
            t_span=self._T, n_points=self._N, **_RUN
        )
        i = list(r.species_names).index("T")

        def value(k):
            sim = bngsim.Simulator(self._model(k), method="ode")
            return np.asarray(sim.run(t_span=self._T, n_points=self._N, **_FD).species)[:, i]

        h = 0.3 * 1e-5
        _assert_close(r.sensitivities[:, i, 0], (value(0.3 + h) - value(0.3 - h)) / (2 * h))


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


# ── GH #221: the refusal survives an HDF5 round trip ───────────────────────


def test_refused_set_round_trips_through_hdf5(tmp_path):
    """A reloaded result must still know its NaN rows are refusals rather than a
    failed solve — otherwise ``gradient`` silently loses the zero-weight
    exemption and a fit that never scored the species gets NaN everywhere."""
    pytest.importorskip("h5py")
    m = bngsim.Model.from_sbml_string(BLOCKED_VARVOL)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k", "g"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sim.run(t_span=(0.0, 5.0), n_points=6, **_RUN)
    path = tmp_path / "r.h5"
    r.save(path)
    back = bngsim.Result.load(path)
    assert back.ar_sensitivity_refused == frozenset({"T"})
    iA = list(back.species_names).index("A")

    def dL_dY(species, time):
        g = np.zeros_like(species)
        g[:, iA] = 1.0
        return g

    assert np.isfinite(back.gradient(dL_dY)).all()
