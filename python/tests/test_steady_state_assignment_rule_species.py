"""GH #247: an SBML ``<assignmentRule>`` species at steady state.

An AR-target species is emitted ``fixed`` — the loader zeroes its ODE derivative
— so it is not an unknown of ``f(y) = 0`` at all: its value is dictated by the
rule, and its Jacobian row is identically zero. Two consequences, both fixed
here, and the first is the worse one:

* ``ss.concentrations`` reported the value the frozen slot was **seeded** with at
  t=0. On the fixture below that is ``2.0`` where the steady value is ``20.0``,
  while ``run()`` on the same model reports ``20.0`` — two entry points, one
  quantity, a factor of ten, with the steady-state one presenting an initial
  condition as an equilibrium.
* the zero row made ``J`` singular, so ``-J⁻¹·(∂f/∂p)`` refused the **whole
  model** — including the perfectly well-posed gradient of every integrated
  species — under a message about a conservation-law continuum, which is a real
  but different cause.

The fix mirrors what already existed on the two neighbouring paths: the value
comes from the rule's observable/function evaluated at the returned state (the
steady-state analogue of ``_apply_ar_report_map``), and the species is folded out
of the solved subspace the way issue #74 folds out a write-only accumulator — an
accumulator contributes a structurally zero *column*, a rule target a zero *row*.
Its ``dY_ss/dp`` row is then the chain rule through the assignment, which is the
same thing GH #221 fills the time-course tensor with.

Oracles, in order of authority:

* **Closed form** — ``A`` is a source/sink balance ``ks/kd``, so ``A* = 10``,
  ``S = 2A ⇒ S* = 20``, ``dA*/dks = 1/kd = 2`` and ``dS*/dks = 2/kd = 4``. The
  nonlinear rule ``S2 = A²`` gives ``S2* = 100`` and ``dS2*/dks = 2A*/kd = 40``.
* **The time course** — ``run()`` integrated far past the transient, which shares
  no code with the steady-state solver.
* **Finite difference of the steady state itself** — re-solve at ``ks ± h`` and
  difference, which touches no sensitivity machinery on either side.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

KS, KD = 5.0, 0.5
A_STAR = KS / KD  # 10.0


def _sbml(rule_math: str, sid: str) -> str:
    """``∅ → A`` at ``ks``, ``A → ∅`` at ``kd·A``; ``sid`` set by an AssignmentRule.

    ``A`` has an isolated steady state (so the Jacobian is non-singular once the
    rule target is out of it) and no conservation law, which keeps the refusal
    this module is about attributable to the assignment rule alone.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ss_ar">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="{sid}" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="ks" value="{KS}" constant="true"/>
      <parameter id="kd" value="{KD}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="{sid}">
        <math xmlns="http://www.w3.org/1998/Math/MathML">{rule_math}</math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="syn" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>ks</ci></math></kineticLaw>
      </reaction>
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


LINEAR = _sbml("<apply><times/><cn>2</cn><ci>A</ci></apply>", "S")
NONLINEAR = _sbml("<apply><times/><ci>A</ci><ci>A</ci></apply>", "S2")

# (sbml, rule-target id, its steady value, d(target)/dks)
RULES = [
    pytest.param(LINEAR, "S", 2 * A_STAR, 2.0 / KD, id="linear"),
    pytest.param(NONLINEAR, "S2", A_STAR**2, 2 * A_STAR / KD, id="nonlinear"),
]
_SS = dict(tol=1e-12, max_time=1e5)


@pytest.fixture(autouse=True)
def _force_codegen(monkeypatch):
    """A nonlinear rule is emitted as a function, and its steady-state output
    sensitivity (GH #12/#75) is the compiled chain rule."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)


def _solve(sbml, **kw):
    m = bngsim.Model.from_sbml_string(sbml)
    return bngsim.Simulator(m, method="ode").steady_state(**_SS, **kw)


# ── The value ───────────────────────────────────────────────────────────────


class TestReportedValue:
    @pytest.mark.parametrize("sbml,sid,value,_d", RULES)
    def test_value_is_the_rule_not_the_frozen_slot(self, sbml, sid, value, _d):
        ss = _solve(sbml)
        assert ss.converged
        n = list(ss.species_names)
        assert ss.concentrations[n.index("A")] == pytest.approx(A_STAR, rel=1e-9)
        assert ss.concentrations[n.index(sid)] == pytest.approx(value, rel=1e-9)

    @pytest.mark.parametrize("sbml,sid,value,_d", RULES)
    def test_agrees_with_the_time_course(self, sbml, sid, value, _d):
        """The independent oracle: the two entry points used to disagree by 10x."""
        ss = _solve(sbml)
        m = bngsim.Model.from_sbml_string(sbml)
        r = bngsim.Simulator(m, method="ode").run(
            t_span=(0.0, 400.0), n_points=3, rtol=1e-11, atol=1e-13
        )
        late = float(np.asarray(r.outputs(f"species:{sid}"))[-1, 0])
        assert late == pytest.approx(value, rel=1e-7)
        assert ss.concentrations[list(ss.species_names).index(sid)] == pytest.approx(
            late, rel=1e-7
        )

    @pytest.mark.parametrize("method", ["integration", "newton"])
    def test_both_solver_methods_report_the_rule(self, method):
        ss = _solve(LINEAR, method=method)
        assert ss.converged
        i = list(ss.species_names).index("S")
        assert ss.concentrations[i] == pytest.approx(2 * A_STAR, rel=1e-8)

    def test_a_second_solve_on_one_simulator_starts_where_the_first_did(self):
        """Evaluating the rule must not carry the converged state onto the model.

        The value pass sets the species state to evaluate observables at the root.
        Leaving it there makes a second ``steady_state()`` on the same Simulator
        start from the first one's answer — measured on a `.net` accumulator as
        49990 becoming 99990, i.e. integrated twice. Regression guard.
        """
        m = bngsim.Model.from_sbml_string(LINEAR)
        sim = bngsim.Simulator(m, method="ode")
        first = np.asarray(sim.steady_state(**_SS).concentrations)
        second = np.asarray(sim.steady_state(**_SS).concentrations)
        np.testing.assert_allclose(first, second, rtol=1e-9)


# ── The gradient the zero row used to take down with it ─────────────────────


class TestSensitivity:
    @pytest.mark.parametrize("sbml,sid,_v,dks", RULES)
    def test_whole_model_gradient_comes_back(self, sbml, sid, _v, dks):
        ss = _solve(sbml, sensitivity_params=["ks"])
        n = list(ss.species_names)
        S = np.asarray(ss.sensitivity)
        # The integrated species — refused outright before, with nothing wrong
        # with it: dA*/dks = 1/kd.
        assert S[n.index("A")][0] == pytest.approx(1.0 / KD, rel=1e-6)
        # ...and the rule target follows its assignment.
        assert S[n.index(sid)][0] == pytest.approx(dks, rel=1e-6)
        assert np.isfinite(S).all()

    def test_the_jacobian_is_no_longer_singular(self):
        ss = _solve(LINEAR, sensitivity_params=["ks"])
        # 0.0 before: the rule target's zero row made the reduced solve rank
        # deficient, which is what produced the continuum message.
        assert ss.sens_jacobian_rcond > 1e-3

    @pytest.mark.parametrize("sbml,sid,_v,dks", RULES)
    def test_matches_a_finite_difference_of_the_steady_state(self, sbml, sid, _v, dks):
        """No sensitivity machinery on either side — re-solve at ks ± h."""

        def value_at(ks):
            m = bngsim.Model.from_sbml_string(sbml)
            m.set_param("ks", ks)
            ss = bngsim.Simulator(m, method="ode").steady_state(**_SS)
            return float(ss.concentrations[list(ss.species_names).index(sid)])

        h = KS * 1e-6
        fd = (value_at(KS + h) - value_at(KS - h)) / (2 * h)
        assert fd == pytest.approx(dks, rel=1e-5)
        ss = _solve(sbml, sensitivity_params=["ks"])
        analytic = float(np.asarray(ss.sensitivity)[list(ss.species_names).index(sid)][0])
        assert analytic == pytest.approx(fd, rel=1e-5)

    def test_output_sensitivities_agree_with_the_species_row(self):
        ss = _solve(LINEAR, sensitivity_params=["ks"])
        i = list(ss.species_names).index("S")
        row = np.asarray(ss.sensitivity)[i]
        np.testing.assert_allclose(row, ss.output_sensitivities(["observable:S"])[0], rtol=1e-12)


# ── The subspace the species is folded out of ───────────────────────────────


class TestSubspace:
    def test_the_rule_target_is_excluded_automatically(self):
        ss = _solve(LINEAR)
        n = list(ss.species_names)
        assert list(ss.excluded_species) == [n.index("S")]
        assert ss.n_residual_species == len(n) - 1

    def test_a_caller_mask_is_intersected_not_overridden(self):
        # Keeping only A is already what the auto-exclusion does here; the point
        # is that asking for it explicitly does not put S back in.
        ss = _solve(LINEAR, mask=["A", "S"])
        n = list(ss.species_names)
        assert list(ss.excluded_species) == [n.index("S")]

    def test_a_model_with_no_rules_is_untouched(self):
        """Control: no AR species ⇒ no mask is invented, so the BNG2.pl parity
        criterion still runs over every species."""
        # Built by dropping the whole <listOfRules>, not by patching its body:
        # a substitution that silently fails to match leaves the rule in place
        # and the control passes for the wrong reason.
        full = _sbml("<apply><times/><cn>2</cn><ci>A</ci></apply>", "S")
        head, _, tail = full.partition("<listOfRules>")
        plain = head + tail.partition("</listOfRules>")[2]
        assert "assignmentRule" not in plain
        m = bngsim.Model.from_sbml_string(plain)
        assert not (getattr(m, "_ar_report_map", None) or {})
        ss = bngsim.Simulator(m, method="ode").steady_state(**_SS)
        assert list(ss.excluded_species) == []
        assert ss.n_residual_species == len(list(ss.species_names))


# ── The second entry point ──────────────────────────────────────────────────


class TestBatch:
    """``steady_state_batch`` builds its own results from a per-entry clone, so
    it is a second site that has to run the same two passes — the failure mode a
    dose scan would show is every entry reporting the frozen value."""

    def test_every_entry_reports_the_rule(self):
        sim = bngsim.Simulator(bngsim.Model.from_sbml_string(LINEAR), method="ode")
        results = sim.steady_state_batch([{"ks": 5.0}, {"ks": 10.0}, {"ks": 20.0}], **_SS)
        assert [r.converged for r in results] == [True, True, True]
        # S* = 2·ks/kd tracks the scan instead of sitting at its seeded 2.0.
        got = [float(r.concentrations[list(r.species_names).index("S")]) for r in results]
        assert got == pytest.approx([2 * k / KD for k in (5.0, 10.0, 20.0)], rel=1e-7)

    def test_the_rule_target_is_excluded_in_every_entry(self):
        sim = bngsim.Simulator(bngsim.Model.from_sbml_string(LINEAR), method="ode")
        results = sim.steady_state_batch([{"ks": 5.0}, {"ks": 10.0}], **_SS)
        for r in results:
            assert list(r.excluded_species) == [list(r.species_names).index("S")]
