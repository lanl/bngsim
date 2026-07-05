"""Tests for the model-based analytical sensitivity-RHS codegen path
(``generate_sens_from_model`` and ``prepare_model_codegen``).

These cover the new code path added so that any model whose reactions are
all Elementary — including those built directly via ``ModelBuilder`` —
gets a combined RHS + sensitivity-RHS .so, byte-equivalent to what the
.net path emits via ``generate_sens_rhs_c``.

Issue refs:
- #15: ``codegen_data()`` schema gap (no ``is_const``/expression on params),
       which is why the model path skips the derived-param chain rule.
- #16: SBML/Antimony loader emits Elementary for trivially-mass-action
       kinetic laws (closed 2026-05-05). Antimony models with mass-action
       laws now get analytical sens RHS via the model-based path; only
       laws the classifier rejects (MM/Hill/funcdef calls/multi-compartment)
       still fall through to RHS-only + CVODES FD.
"""

from __future__ import annotations

import ctypes

import bngsim
import numpy as np
import pytest
from bngsim import Model
from bngsim._bngsim_core import ModelBuilder
from bngsim._codegen import (
    generate_combined_from_model,
    generate_rhs_from_model,
    generate_sens_from_model,
    prepare_model_codegen,
)


def _build_decay_model() -> Model:
    """Tiny Elementary model: A -> B with rate k1*A."""
    b = ModelBuilder()
    b.add_parameter("k1", 0.3)
    b.add_species("A", 10.0)
    b.add_species("B", 0.0)
    b.add_reaction([0], [1], "elementary", "k1")
    return Model(b.build())


def _build_reversible_model() -> Model:
    """Two-param Elementary model: A <-> B with rates kf*A, kr*B."""
    b = ModelBuilder()
    b.add_parameter("kf", 0.5)
    b.add_parameter("kr", 0.1)
    b.add_species("A", 10.0)
    b.add_species("B", 0.0)
    b.add_reaction([0], [1], "elementary", "kf")
    b.add_reaction([1], [0], "elementary", "kr")
    return Model(b.build())


class TestModelCodegenSensGeneration:
    """Unit tests on the C source emitted by generate_sens_from_model."""

    def test_elementary_decay_emits_sens_rhs(self):
        m = _build_decay_model()
        sens = generate_sens_from_model(m)
        assert sens is not None
        assert "bngsim_codegen_sens_rhs" in sens
        assert "bngsim_dfdp" in sens
        assert "bngsim_jac_vec" in sens
        # N_PARAMS / N_SPECIES macros must reflect the model
        assert "#define N_SPECIES 2" in sens
        assert "#define N_PARAMS  1" in sens

    def test_combined_includes_both_rhs_symbols(self):
        m = _build_decay_model()
        combined, has_sens = generate_combined_from_model(m)
        assert has_sens is True
        assert "bngsim_codegen_rhs" in combined
        assert "bngsim_codegen_sens_rhs" in combined

    def test_reversible_two_params_emit_dfdp_cases(self):
        m = _build_reversible_model()
        sens = generate_sens_from_model(m)
        assert sens is not None
        # dfdp switch must have a case for each rate-constant param.
        assert "case 0:" in sens
        assert "case 1:" in sens

    def test_from_antimony_mass_action_emits_sens_rhs(self):
        """Issue #16 is closed (2026-05-05): the SBML/Antimony loader now
        emits Elementary for trivially-mass-action kinetic laws, so
        ``generate_sens_from_model`` produces analytical sens RHS.
        """
        pytest.importorskip("antimony")
        m = Model.from_antimony_string("model decay; A=10; B=0; k1=0.3; J0: A -> B; k1*A; end")
        sens = generate_sens_from_model(m)
        assert sens is not None
        assert "bngsim_codegen_sens_rhs" in sens
        combined, has_sens = generate_combined_from_model(m)
        assert has_sens is True
        assert "bngsim_codegen_rhs" in combined
        assert "bngsim_codegen_sens_rhs" in combined

    def test_from_antimony_michaelis_menten_returns_none(self):
        """Non-mass-action laws (MM here) still fall through to RHS-only —
        the classifier in ``_sbml_loader._classify_mass_action`` rejects
        anything with division/sums in the kinetic law.
        """
        pytest.importorskip("antimony")
        m = Model.from_antimony_string(
            "model mm; S=10; P=0; Vmax=1; Km=2; J0: S -> P; Vmax*S/(Km + S); end"
        )
        sens = generate_sens_from_model(m)
        assert sens is None
        combined, has_sens = generate_combined_from_model(m)
        assert has_sens is False
        assert "bngsim_codegen_rhs" in combined
        assert "bngsim_codegen_sens_rhs" not in combined


class TestModelCodegenSensCompilation:
    """Compile the emitted source and verify both symbols are exported."""

    def test_combined_compiles_and_exports_both_symbols(self):
        m = _build_decay_model()
        so = prepare_model_codegen(m)
        assert so is not None and so.exists()

        lib = ctypes.CDLL(str(so))
        assert hasattr(lib, "bngsim_codegen_rhs")
        assert hasattr(lib, "bngsim_codegen_sens_rhs")


class TestModelCodegenSensCorrectness:
    """End-to-end: codegen analytical sens must match an externally-
    computed FD reference. Tolerance is ~1e-4 abs (FD is the limit, not
    codegen) — analytical sens against FD typically agrees to ~1e-8.
    """

    @staticmethod
    def _ext_fd_sens(
        build_fn, param_name: str, nominal: float, t_span, n_points, eps: float = 1e-5
    ) -> np.ndarray:
        """2-pt centered FD of trajectories at param_name = nominal ± eps."""

        def _traj(val: float) -> np.ndarray:
            m = build_fn()
            m.set_param(param_name, val)
            sim = bngsim.Simulator(m, method="ode")
            return sim.run(t_span=t_span, n_points=n_points, rtol=1e-12, atol=1e-14).species

        return (_traj(nominal + eps) - _traj(nominal - eps)) / (2.0 * eps)

    def test_decay_codegen_sens_matches_external_fd(self):
        # Codegen-on path
        m = _build_decay_model()
        so = prepare_model_codegen(m)
        assert so is not None
        m._codegen_so_path = str(so)

        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        r = sim.run(t_span=(0, 5), n_points=51, rtol=1e-10, atol=1e-12)

        fd = self._ext_fd_sens(_build_decay_model, "k1", 0.3, (0, 5), 51)
        err = np.abs(r.sensitivities[..., 0] - fd).max()
        assert err < 1e-6, f"codegen sens vs external FD: max abs err = {err:.3e}"

    def test_reversible_two_param_codegen_sens_matches_external_fd(self):
        m = _build_reversible_model()
        so = prepare_model_codegen(m)
        assert so is not None
        m._codegen_so_path = str(so)

        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kf", "kr"])
        r = sim.run(t_span=(0, 5), n_points=51, rtol=1e-10, atol=1e-12)

        fd_kf = self._ext_fd_sens(_build_reversible_model, "kf", 0.5, (0, 5), 51)
        fd_kr = self._ext_fd_sens(_build_reversible_model, "kr", 0.1, (0, 5), 51)
        err_kf = np.abs(r.sensitivities[..., 0] - fd_kf).max()
        err_kr = np.abs(r.sensitivities[..., 1] - fd_kr).max()
        # The centered-FD reference (eps=1e-5, ODE rtol=1e-12) has a noise
        # floor near 1e-7 (ODE tolerance / eps). Coupled two-species systems
        # amplify that into the low 1e-6 range on macOS arm64 (observed:
        # err_kf = 2.014e-06 in CI run 25736978049). The decay test in this
        # same file stays at 1e-6 because a single-species model does not
        # amplify FD noise the same way. 5e-6 keeps a ~25x safety margin
        # over the FD reference noise floor while still catching a real
        # analytical-sens regression.
        assert err_kf < 5e-6, f"kf codegen sens err = {err_kf:.3e}"
        assert err_kr < 5e-6, f"kr codegen sens err = {err_kr:.3e}"


class TestCodegenFDSensCorrectness:
    """Regression: when the codegen .so contains RHS but no analytical sens
    RHS (e.g., a Functional/MM model), CVODES falls back to internal FD on
    sens_p. The codegen RHS callback must mirror sens_p into its parameter
    buffer before forwarding to the .so — otherwise the perturbations are
    invisible and every sensitivity column comes back identically zero.

    Pre-fix bug: ``cvode_codegen_rhs`` (in cvode_simulator.cpp) read from
    a separate buffer that was never re-synced from sens_p; the prior
    session's auto-trigger experiment surfaced this as 11 sens-test
    failures. See memory/project_codegen_default_for_sens.md.
    """

    @staticmethod
    def _antimony_mm() -> Model:
        # Michaelis-Menten kinetics — the classifier in _sbml_loader rejects
        # this (division in the AST), so prepare_model_codegen produces
        # RHS-only and CVODES uses internal FD for sensitivity, exercising
        # the cvode_codegen_rhs sens_p sync path.
        return Model.from_antimony_string(
            "model mm; S=10; P=0; k1=0.3; Km=2; J0: S -> P; k1*S/(Km + S); end"
        )

    def test_rhs_only_codegen_uses_fd_correctly(self):
        pytest.importorskip("antimony")
        m = self._antimony_mm()
        so = prepare_model_codegen(m)
        assert so is not None
        m._codegen_so_path = str(so)

        # Confirm the .so really lacks the sens RHS — otherwise this test
        # would silently exercise the analytical path instead of FD.
        lib = ctypes.CDLL(str(so))
        assert hasattr(lib, "bngsim_codegen_rhs")
        assert not hasattr(lib, "bngsim_codegen_sens_rhs")

        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        r = sim.run(t_span=(0, 5), n_points=51, rtol=1e-10, atol=1e-12)

        # External FD reference (no codegen)
        def _traj(k1: float) -> np.ndarray:
            mm = self._antimony_mm()
            mm.set_param("k1", k1)
            return (
                bngsim.Simulator(mm, method="ode")
                .run(t_span=(0, 5), n_points=51, rtol=1e-12, atol=1e-14)
                .species
            )

        eps = 1e-5
        fd = (_traj(0.3 + eps) - _traj(0.3 - eps)) / (2.0 * eps)

        # First the diagnostic that originally failed: sensitivity must not
        # be identically zero. (Before the fix it was — silently.)
        max_sens = np.abs(r.sensitivities).max()
        assert max_sens > 1e-3, (
            f"codegen-RHS + CVODES-FD sens is identically zero "
            f"(max={max_sens:.3e}); cvode_codegen_rhs is not syncing sens_p"
        )

        # Then the correctness check.
        err = np.abs(r.sensitivities[..., 0] - fd).max()
        assert err < 1e-4, f"codegen-RHS + CVODES-FD sens vs external FD: max abs err = {err:.3e}"


class TestSensAutoTrigger:
    """Codegen auto-enables for ANY sensitivity workflow (GH #214: forward
    sensitivity requires an analytical codegen RHS — the size gate and the
    interpreted finite-difference fallback were retired). codegen=False or
    BNGSIM_NO_CODEGEN with sensitivities now raises rather than degrading.
    """

    def test_small_model_codegens_for_sensitivity(self):
        # Hard requirement (GH #214): forward sensitivity always builds an
        # analytical codegen RHS — there is no size gate and no interpreted
        # finite-difference fallback. decay model: 2 species, 1 param.
        m = _build_decay_model()
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        assert sim._codegen_so_path or sim._codegen_c_source, (
            "tiny sensitivity model must still codegen (no size gate)"
        )

    def test_small_model_auto_codegens_when_threshold_lowered(self, monkeypatch):
        # The trigger mechanics still fire once the coupled system clears the
        # gate (forced here via the shared BNGSIM_CODEGEN_THRESHOLD knob).
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        m = _build_decay_model()
        assert m._codegen_so_path == ""

        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        # Either backend proves the sens auto-trigger fired: the cc path sets a
        # .so, the MIR JIT backend (BNGSIM_CODEGEN_JIT=mir) stashes C source.
        assert sim._codegen_so_path or sim._codegen_c_source, (
            "Simulator did not auto-enable codegen for sensitivity workflow"
        )
        # Simulator should have cached the codegen artifact back onto the model
        # so subsequent Simulators reuse it without re-preparing.
        if sim._codegen_c_source:
            assert m._codegen_c_source == sim._codegen_c_source
        else:
            assert m._codegen_so_path == sim._codegen_so_path

    def test_no_sens_does_not_trigger_codegen(self):
        # Tiny model without sensitivity_params — codegen should NOT fire
        # (ExprTk is faster than codegen below ~150 species).
        m = _build_decay_model()
        sim = bngsim.Simulator(m, method="ode")
        assert sim._codegen_so_path == "", (
            "Auto-codegen fired without sensitivity_params; the Simulator-level "
            "trigger should only fire on sens workflows"
        )

    def test_bngsim_no_codegen_env_raises_for_sensitivity(self, monkeypatch):
        # Hard requirement (GH #214): BNGSIM_NO_CODEGEN + sensitivities is a
        # contradiction (the interpreted FD sens path was retired) → raise.
        monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
        m = _build_decay_model()
        with pytest.raises(ValueError, match="requires code generation"):
            bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])

    def test_existing_codegen_so_is_not_clobbered(self):
        # If the model already carries a _codegen_so_path (e.g., the SBML
        # loader threshold fired), the Simulator should reuse it, not
        # re-prepare a fresh one.
        m = _build_decay_model()
        m._codegen_so_path = "/nonexistent/path.so"
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        assert sim._codegen_so_path == "/nonexistent/path.so"

    def test_explicit_codegen_false_raises_for_sensitivity(self):
        # Hard requirement (GH #214): codegen=False + sensitivities → raise,
        # rather than silently using the retired interpreted FD sens path.
        m = _build_decay_model()
        with pytest.raises(ValueError, match="requires code generation"):
            bngsim.Simulator(m, method="ode", sensitivity_params=["k1"], codegen=False)

    def test_mir_jit_backend_routes_sens_to_c_source(self, monkeypatch):
        # GH #78: with BNGSIM_CODEGEN_JIT=mir the sensitivity auto-trigger must
        # stash the codegen C source for the in-process MIR JIT instead of
        # compiling a .so. Source generation is pure Python, so this routing
        # check needs neither the MIR-enabled binary nor a .run().
        monkeypatch.setenv("BNGSIM_CODEGEN_JIT", "mir")
        # Lower the size gate so the tiny decay model clears it — this test pins
        # the MIR *routing*, not the size policy (covered separately).
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        m = _build_decay_model()
        assert m._codegen_c_source == ""
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        assert sim._codegen_so_path == "", "MIR backend must not compile a .so"
        assert sim._codegen_c_source != "", "MIR backend did not generate JIT source"
        # The combined source carries the analytical sensitivity RHS, not just
        # the plain RHS (the whole point of the sens auto-trigger).
        assert "bngsim_codegen_sens_rhs" in sim._codegen_c_source
        # Cached back onto the model for reuse by subsequent Simulators.
        assert m._codegen_c_source == sim._codegen_c_source


class TestSensGateBoundaryAndBypass:
    """Forward sensitivity always codegens (GH #214: the size gate was retired),
    including IC sensitivities on a tiny model."""

    def _codegened(self, sim) -> bool:
        return bool(sim._codegen_so_path or sim._codegen_c_source)

    def test_reversible_model_codegens_for_sensitivity(self):
        # reversible model: 2 species, 2 params. With no size gate, sensitivity
        # always builds the analytical RHS regardless of coupled-system size.
        m = _build_reversible_model()
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kf", "kr"])
        assert self._codegened(sim), "sensitivity must codegen (no size gate)"

    def test_ic_sensitivity_codegens(self):
        # IC sensitivities REQUIRE codegen (CVODES has no parameter to perturb);
        # a tiny model must still codegen.
        m = _build_decay_model()
        sim = bngsim.Simulator(m, method="ode", sensitivity_ic=["A"])
        assert self._codegened(sim), "IC sensitivity must codegen"


# ── GH #75: amount_valued (hOSU) species under codegen ─────────────────
#
# An hasOnlySubstanceUnits=true species symbol denotes its *amount*
# (stored × V_c), not the stored concentration. The C++ engine restores
# this in compute_species_factor_ode (Elementary rate carries the constant
# ∏ V_c^mult over amount_valued reactants) and update_observables
# (observable sums fold in V_c). The codegen RHS/sens emitters mirror both:
# the Elementary rate gets the per-reaction amount_factor, the observable
# coefficients fold in V_c (so a Functional law referencing the species via
# its same-named observable reads the amount), and the analytical sens RHS
# carries the same amount_factor. These tests pin that mirror against an
# analytical amount-law oracle and an external FD reference.

# hOSU=true V=1e-3 bimolecular A+B→C, rate kf*A*B. Amounts: A,B start at
# N0=100, the amount-law is amount_A(t)=N0/(1+N0·kf·t); the engine reports
# storage [A]=amount_A/V. amount_factor = V·V = 1e-6 (≠1, ≠ canceled by the
# 1/V storage divide), so this genuinely exercises the constant.
_HOSU_BIMOL_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_v_bimol">
    <listOfCompartments>
      <compartment id="V" size="1e-3" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="V" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="V" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="C" compartment="V" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="0.02" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="bind" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kf</ci><ci>A</ci><ci>B</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# hOSU=true V=4 saturating (Functional) law kf*S/(Km+S). The rate references
# S through its same-named observable, which folds in V_c=4 → reads the
# amount. Non-mass-action ⇒ sens RHS is RHS-only (validates the observable-
# folding flows into a Functional rate under codegen).
_HOSU_FUNCTIONAL_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_functional">
    <listOfCompartments>
      <compartment id="cell" size="4" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cell" initialAmount="400"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="6.0" constant="true"/>
      <parameter id="Km" value="200.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><divide/>
              <apply><times/><ci>kf</ci><ci>S</ci></apply>
              <apply><plus/><ci>Km</ci><ci>S</ci></apply>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

_HOSU_V, _HOSU_N0, _HOSU_KF = 1e-3, 100.0, 0.02


class TestModelCodegenHosuAmountFactor:
    """GH #75: codegen RHS/sens must read amount_valued species as amounts."""

    def test_elementary_amount_factor_in_emitted_c(self):
        # The Elementary rate carries amount_factor = V·V, and the per-species
        # shadow observables fold in V_c. Pin the emitted constants so a
        # silent drop of the amount_factor term is caught at the source level.
        m = Model.from_sbml_string(_HOSU_BIMOL_SBML)
        c = generate_rhs_from_model(m)
        rate_lines = [ln for ln in c.splitlines() if ln.strip().startswith("rate =")]
        assert len(rate_lines) == 1
        # amount_factor = (1e-3)^2 = 1e-06 appears as a standalone factor.
        assert repr(_HOSU_V * _HOSU_V) in rate_lines[0], rate_lines[0]
        # Observable coefficients fold in V_c = 1e-3 (= 0.001).
        obs_lines = [ln for ln in c.splitlines() if ln.strip().startswith("obs[")]
        assert any(repr(_HOSU_V) in ln for ln in obs_lines), obs_lines

    def test_elementary_codegen_rhs_matches_amount_law(self):
        # Codegen ODE trajectory must match the analytical amount-law oracle
        # AND the ExprTk engine (which reads amounts via amount_valued).
        m = Model.from_sbml_string(_HOSU_BIMOL_SBML)
        so = prepare_model_codegen(m)
        assert so is not None and so.exists()
        m._codegen_so_path = str(so)
        r = bngsim.Simulator(m, method="ode").run(
            t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-16
        )
        t = np.asarray(r.time)
        names = list(r.species_names)
        A = np.asarray(r.species)[:, names.index("A")]
        C = np.asarray(r.species)[:, names.index("C")]

        amount_A = _HOSU_N0 / (1.0 + _HOSU_N0 * _HOSU_KF * t)
        expected_A_storage = amount_A / _HOSU_V
        np.testing.assert_allclose(A, expected_A_storage, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(A + C, _HOSU_N0 / _HOSU_V, rtol=1e-6, atol=1e-3)

        # ExprTk reference (fresh model, no codegen — sub-threshold + no sens).
        m_et = Model.from_sbml_string(_HOSU_BIMOL_SBML)
        assert m_et._codegen_so_path == ""
        r_et = bngsim.Simulator(m_et, method="ode").run(
            t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-16
        )
        A_et = np.asarray(r_et.species)[:, list(r_et.species_names).index("A")]
        np.testing.assert_allclose(A, A_et, rtol=1e-8, atol=1e-8)

    def test_functional_codegen_rhs_reads_amounts(self):
        # A Functional (non-mass-action) law referencing an hOSU species must
        # read the amount under codegen too — via the same-named observable.
        m = Model.from_sbml_string(_HOSU_FUNCTIONAL_SBML)
        so = prepare_model_codegen(m)
        assert so is not None and so.exists()
        # Non-mass-action ⇒ RHS-only (no analytical sens RHS).
        lib = ctypes.CDLL(str(so))
        assert hasattr(lib, "bngsim_codegen_rhs")
        assert not hasattr(lib, "bngsim_codegen_sens_rhs")
        m._codegen_so_path = str(so)
        r_cg = bngsim.Simulator(m, method="ode").run(
            t_span=(0, 60), n_points=31, rtol=1e-10, atol=1e-14
        )
        S = np.asarray(r_cg.species)[:, list(r_cg.species_names).index("S")]
        P = np.asarray(r_cg.species)[:, list(r_cg.species_names).index("P")]
        assert S[0] == pytest.approx(100.0, rel=1e-9)  # amount 400 / V_c 4
        # amount conservation a_S + a_P = 400 ⇒ (S + P)·V_c = 400.
        np.testing.assert_allclose((S + P) * 4.0, 400.0, rtol=1e-7, atol=1e-5)

        # Must match the ExprTk engine reading amounts via the observable.
        m_et = Model.from_sbml_string(_HOSU_FUNCTIONAL_SBML)
        r_et = bngsim.Simulator(m_et, method="ode").run(
            t_span=(0, 60), n_points=31, rtol=1e-10, atol=1e-14
        )
        P_et = np.asarray(r_et.species)[:, list(r_et.species_names).index("P")]
        np.testing.assert_allclose(P, P_et, rtol=1e-7, atol=1e-7)

    def test_elementary_codegen_sens_matches_fd(self):
        # The analytical sens RHS carries amount_factor; validate ∂[A]/∂kf
        # against an external centered-FD reference. Concentrations here are
        # O(1e5) (amount/V, V=1e-3), so compare with a relative tolerance.
        m = Model.from_sbml_string(_HOSU_BIMOL_SBML)
        so = prepare_model_codegen(m)
        assert so is not None
        lib = ctypes.CDLL(str(so))
        assert hasattr(lib, "bngsim_codegen_sens_rhs"), "elementary hOSU model should get sens RHS"
        m._codegen_so_path = str(so)

        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kf"])
        r = sim.run(t_span=(0, 10), n_points=21, rtol=1e-12, atol=1e-16)
        sens_kf = r.sensitivities[..., 0]

        def _traj(kf: float) -> np.ndarray:
            mm = Model.from_sbml_string(_HOSU_BIMOL_SBML)
            mm.set_param("kf", kf)
            return (
                bngsim.Simulator(mm, method="ode")
                .run(t_span=(0, 10), n_points=21, rtol=1e-12, atol=1e-16)
                .species
            )

        eps = _HOSU_KF * 1e-5
        fd = (_traj(_HOSU_KF + eps) - _traj(_HOSU_KF - eps)) / (2.0 * eps)
        # FD noise floor scales with the O(1e5) concentration; a relative
        # check (with a small abs floor for the ~0 columns) is the robust
        # oracle. A dropped amount_factor would mis-scale the whole column
        # by V (1e3), far outside this band.
        np.testing.assert_allclose(sens_kf, fd, rtol=2e-4, atol=1e-3)


class TestSensAutoTriggerNetPathRouting:
    """The auto-trigger should route from_net models to the .net codegen
    path (which handles derived-parameter chain rules) instead of the
    model-based path (which does not — issue #15).
    """

    @staticmethod
    def _derived_rate_const_net() -> str:
        import os
        from pathlib import Path

        # Honor BNGSIM_TEST_DATA so this module works under run_tests.sh,
        # which copies tests to a temp dir (breaking __file__-relative
        # resolution).
        env = os.environ.get("BNGSIM_TEST_DATA")
        base = (
            Path(env) if env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"
        )
        p = base / "derived_rate_const.net"
        assert p.exists(), f"Test data not found: {p}"
        return str(p)

    def test_from_net_carries_net_path(self):
        m = bngsim.Model.from_net(self._derived_rate_const_net())
        assert m._net_path == self._derived_rate_const_net()

    def test_clone_preserves_net_path(self):
        m = bngsim.Model.from_net(self._derived_rate_const_net())
        m2 = m.clone()
        assert m2._net_path == m._net_path

    def test_auto_trigger_routes_from_net_to_dot_net_codegen_path(self, monkeypatch):
        """End-to-end correctness: derived-rate-constant chain rule must
        land for `from_net` models without the caller having to pass
        ``codegen=True, net_path=...`` manually. Pre-routing, the auto-
        trigger would call `prepare_model_codegen` which silently dropped
        the chain rule and produced zero sens for the primary parameter.
        """
        # This toy .net is below the codegen size gate; lower the threshold so
        # the auto-trigger fires (this test pins .net routing + chain rule, not
        # the size policy).
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        net_path = self._derived_rate_const_net()
        sample_times = list(np.linspace(0.0, 5.0, 51))

        # Auto-trigger path (codegen kwarg unset — defaults to None / auto)
        m = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kon"])
        assert sim._codegen_so_path or sim._codegen_c_source, "auto-trigger did not fire"
        r_auto = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)
        sx_auto = r_auto.sensitivities[:, :, 0]

        # External FD reference (no codegen, ExprTk path)
        nominal = 1.0  # value for kon in derived_rate_const.net

        def _traj(kon_val: float) -> np.ndarray:
            mm = bngsim.Model.from_net(net_path)
            mm.set_param("kon", kon_val)
            sim2 = bngsim.Simulator(mm, method="ode", codegen=False)
            return sim2.run(
                sample_times=sample_times, rtol=1e-12, atol=1e-14, max_steps=10**6
            ).species

        eps = 1e-5
        fd = (_traj(nominal + eps) - _traj(nominal - eps)) / (2.0 * eps)

        # Diagnostic: must not be identically zero (chain rule not dropped)
        assert np.abs(sx_auto).max() > 1e-3, (
            "auto-triggered codegen produced identically-zero sens — chain "
            "rule was dropped (model-based path was used instead of .net)"
        )
        # Correctness: sign agreement and bounded relative error.
        denom = np.maximum(np.abs(fd[1:]), np.abs(sx_auto[1:]))
        mask = denom > 1e-9
        rel = np.abs(fd[1:] - sx_auto[1:])[mask] / denom[mask]
        assert mask.any()
        assert rel.max() < 1e-3, (
            f"auto-triggered codegen sens for kon does not match FD "
            f"(max relerr={rel.max():.3e}); chain rule dropped"
        )
