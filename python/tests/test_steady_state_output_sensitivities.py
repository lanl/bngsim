"""Steady-state output sensitivities at the observable / expression level (GH #12).

``Simulator.steady_state(sensitivity_params=[...])`` already returns the exact
species steady-state sensitivity ``dY_ss/dp`` (KINSOL / implicit function
theorem). These tests cover the observable- and expression-level accessor added
in GH #12 — :meth:`SteadyStateResult.output_sensitivities` — which projects that
species sensitivity onto the model's observables and global functions so a
gradient consumer reads ``∂(observable)/∂θ`` directly, mirroring
:meth:`bngsim.Result.output_sensitivities` on a CVODE run.

The oracle is that CVODE path: a long forward-sensitivity ``run()`` converges to
the steady state, so its last-time-point ``output_sensitivities`` must match the
Newton steady-state ones. Observables use an exact linear projection; functions
use the total derivative (state chain + explicit ∂func/∂p) whose values are
additionally pinned against closed-form derivatives.

Since issue #75 that function block prefers the compiled
``bngsim_codegen_output_sens`` — literally the evaluator the CVODES oracle above
runs — and falls back to finite differences per function where it declines.
``TestCompiledOutputSensPath`` covers which path ran and that both give the same
answer.

Fixtures:
- ``ss_output_sens.net``      — closed A↔B↔C, conserved ⇒ reduced-space solve.
- ``ss_birthdeath.net``       — open birth-death, no conservation ⇒ full-space solve.
- ``ss_expr_sens_derived.net``— A⇌B with ``_rateLaw1 = chi*kon``; full closed form.
- ``ss_expr_sens_mixed.net``  — same skeleton carrying one function of each #198
  status (ok / unsupported / skipped) ⇒ the mixed codegen+FD block.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import bngsim
import numpy as np
import pytest

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

CLOSED_NET = str(DATA_DIR / "ss_output_sens.net")  # A<->B<->C, conserved
OPEN_NET = str(DATA_DIR / "ss_birthdeath.net")  # dS/dt = k_prod - k_deg*S
DERIVED_NET = str(DATA_DIR / "ss_expr_sens_derived.net")  # A<->B, _rateLaw1 = chi*kon
MIXED_NET = str(DATA_DIR / "ss_expr_sens_mixed.net")  # ok / unsupported / skipped
DECLINED_NET = str(DATA_DIR / "obs_zero_arg_call.net")  # no user-selectable functions

# High-accuracy CVODES so the last time point is a faithful steady-state oracle.
_RUN = dict(rtol=1e-11, atol=1e-13, max_steps=10**6)


@pytest.fixture(autouse=True)
def _force_codegen(monkeypatch):
    """Expression (global-function) output sensitivities require the compiled
    ``.so`` on the CVODE ``run()`` oracle, and prefer it on the steady-state path
    since issue #75; force codegen on. monkeypatch restores the environment
    afterwards."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)


def _steady_state(net, params):
    m = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(m, method="ode")
    return sim.steady_state(sensitivity_params=list(params))


def _run_to_steady_state(net, params, t_end):
    """Last-time-point CVODES output sensitivities — the steady-state oracle."""
    m = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(params))
    m.reset()
    return sim.run(t_span=(0.0, t_end), n_points=2, **_RUN)


# ── Parity with the CVODE forward-sensitivity run ────────────────────────────


class TestParityWithCvode:
    """SS output sensitivities == last-point CVODES output sensitivities."""

    @pytest.mark.parametrize(
        "net, params, selectors, t_end",
        [
            (
                CLOSED_NET,
                ["k1", "k3", "amp", "Km"],
                [
                    "observable:A_tot",
                    "observable:BC2",
                    "observable:Total",
                    "expression:satA",
                    "expression:lin",
                    "species:A()",
                    "species:B()",
                    "species:C()",
                ],
                400.0,
            ),
            (
                OPEN_NET,
                ["k_prod", "k_deg", "scale"],
                ["observable:Stot", "expression:sq", "species:S()"],
                200.0,
            ),
        ],
    )
    def test_matches_long_run(self, net, params, selectors, t_end):
        ss = _steady_state(net, params)
        assert ss.converged
        ss_os = ss.output_sensitivities(selectors)

        run = _run_to_steady_state(net, params, t_end)
        # The run must actually have reached the same steady state, else its
        # last-point sensitivities are not the steady-state ones.
        np.testing.assert_allclose(run.species[-1], ss.concentrations, rtol=1e-5, atol=1e-5)
        run_os = run.output_sensitivities(selectors, axis="parameter")[-1]

        assert ss_os.shape == (len(selectors), len(params))
        np.testing.assert_allclose(ss_os, run_os, rtol=1e-4, atol=1e-6)


# ── Observable projection is the EXACT linear group map ──────────────────────


class TestObservableProjectionExact:
    """d(obs)/dp is exactly Σ factor · dY_ss/dp — no finite differences."""

    def test_closed_group_factors(self):
        ss = _steady_state(CLOSED_NET, ["k1", "k3", "amp", "Km"])
        sp = {n: i for i, n in enumerate(ss.species_names)}
        sens = ss.sensitivity  # (n_species, n_params)

        # A_tot = A ; BC2 = B + 2*C ; Total = A + B + C (from the .net groups).
        expected_A_tot = sens[sp["A()"]]
        expected_BC2 = sens[sp["B()"]] + 2.0 * sens[sp["C()"]]
        expected_Total = sens[sp["A()"]] + sens[sp["B()"]] + sens[sp["C()"]]

        got = ss.output_sensitivities(["observable:A_tot", "observable:BC2", "observable:Total"])
        np.testing.assert_allclose(got[0], expected_A_tot, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got[1], expected_BC2, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got[2], expected_Total, rtol=0, atol=1e-12)

    def test_sensitivities_observables_array(self):
        """The bulk array property matches selector-addressed rows in order."""
        ss = _steady_state(CLOSED_NET, ["k1", "k3", "amp", "Km"])
        block = ss.sensitivities_observables
        assert block.shape == (len(ss.observable_names), len(ss.sensitivity_params))
        stacked = ss.output_sensitivities([f"observable:{n}" for n in ss.observable_names])
        np.testing.assert_allclose(block, stacked, rtol=0, atol=0)


# ── Function explicit-parameter term (isolated) ──────────────────────────────


class TestExplicitParameterTerm:
    """A function-only parameter has zero species sensitivity but nonzero
    function output sensitivity — the explicit ∂func/∂p contribution."""

    def test_amp_and_Km_isolated(self):
        params = ["amp", "Km"]
        ss = _steady_state(CLOSED_NET, params)
        # amp/Km do not appear in any reaction, so the steady state and every
        # species/observable sensitivity w.r.t. them is exactly zero.
        np.testing.assert_allclose(ss.sensitivity, 0.0, atol=1e-9)

        # satA = amp*A_tot/(Km+A_tot). At the steady state A_tot = A* = 10/3:
        #   ∂satA/∂amp = A_tot/(Km+A_tot);  ∂satA/∂Km = -amp*A_tot/(Km+A_tot)^2
        a = ss["A()"]
        amp, km = 3.0, 4.0
        expected = np.array([a / (km + a), -amp * a / (km + a) ** 2])
        got = ss.output_sensitivities(["expression:satA"])[0]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)

    def test_birthdeath_explicit_scale(self):
        ss = _steady_state(OPEN_NET, ["k_prod", "k_deg", "scale"])
        # sq = scale*Stot ⇒ ∂sq/∂scale = Stot = S* (explicit); others via chain.
        s_star = ss["S()"]  # 2.5
        got = ss.output_sensitivities(["expression:sq"])[0]
        # columns: [k_prod, k_deg, scale]
        assert got[2] == pytest.approx(s_star, rel=1e-6)


# ── Function explicit-parameter term through a DERIVED parameter ─────────────


class TestDerivedParameterChain:
    """``flux() = _rateLaw1*A_tot`` with ``_rateLaw1 = chi*kon``.

    The explicit ∂flux/∂p term is finite-differenced by perturbing the Parameter
    vector directly, and neither ``update_observables`` nor ``evaluate_functions``
    re-derives ConstantExpression parameters — only ``set_param`` does. So a probe
    of ``kon`` used to leave ``_rateLaw1`` at its nominal value, dropping
    ∂flux/∂_rateLaw1 · ∂_rateLaw1/∂kon and returning the bare state-chain term:
    sign-flipped and 20x too large. Same defect class as issues #2 / #41 and the
    ∂f/∂p hole fixed under #63; the probe now routes through
    ``SteadyStateRhs::sync_params(pi)``.

    ``ss_expr_sens_derived.net`` is A ⇌ B with a = chi·kon and b = koff, so
    conservation gives A+B = 1 and every quantity below has a closed form.
    """

    KON, CHI, KOFF = 1.0, 10.0, 0.5

    def _closed_form(self):
        a, b = self.CHI * self.KON, self.KOFF
        d = (a + b) ** 2
        return {
            # A_ss = b/(a+b); flux = a·A_ss = a·b/(a+b)
            "A_ss": b / (a + b),
            "flux": a * b / (a + b),
            # dflux/da = b²/(a+b)², and a = chi·kon
            "dflux": {
                "kon": self.CHI * b**2 / d,
                "chi": self.KON * b**2 / d,
                "koff": a**2 / d,  # dflux/db = a²/(a+b)²
                "_rateLaw1": b**2 / d,
            },
            # dA_ss/da = -b/(a+b)², dA_ss/db = a/(a+b)²
            "dA": {
                "kon": -self.CHI * b / d,
                "chi": -self.KON * b / d,
                "koff": a / d,
                "_rateLaw1": -b / d,
            },
        }

    def test_expression_sensitivity_matches_closed_form(self):
        params = ["kon", "chi", "koff", "_rateLaw1"]
        ss = _steady_state(DERIVED_NET, params)
        assert ss.converged
        cf = self._closed_form()

        # The state the derivatives are taken at, and the species sensitivity the
        # projection rides on — both pinned so a failure localizes.
        assert ss["A()"] == pytest.approx(cf["A_ss"], rel=1e-7)
        ia = list(ss.species_names).index("A()")
        np.testing.assert_allclose(
            ss.sensitivity[ia], [cf["dA"][p] for p in params], rtol=1e-5, atol=1e-9
        )

        got = ss.output_sensitivities(["expression:flux"])[0]
        np.testing.assert_allclose(got, [cf["dflux"][p] for p in params], rtol=1e-5, atol=1e-8)

    def test_state_chain_alone_is_not_accepted(self):
        """Pin the failure mode, not just the answer.

        Dropping the derived chain leaves ``a·dA_ss/dkon`` — a NEGATIVE number
        where the total derivative is positive. An rtol-only assertion on a
        near-cancelling sum can pass for the wrong reason; this asserts the two
        candidates are far apart and that the right one was returned.
        """
        params = ["kon", "chi"]
        ss = _steady_state(DERIVED_NET, params)
        cf = self._closed_form()
        a = self.CHI * self.KON

        state_chain_only = np.array([a * cf["dA"][p] for p in params])
        total = np.array([cf["dflux"][p] for p in params])
        assert np.all(state_chain_only < 0) and np.all(total > 0)

        got = ss.output_sensitivities(["expression:flux"])[0]
        np.testing.assert_allclose(got, total, rtol=1e-5, atol=1e-8)

    def test_matches_long_cvode_run(self):
        """Independent oracle: the compiled ``bngsim_codegen_output_sens`` chain
        rule at the last point of a converged forward-sensitivity run."""
        params = ["kon", "chi", "koff"]
        ss = _steady_state(DERIVED_NET, params)
        run = _run_to_steady_state(DERIVED_NET, params, 50.0)
        np.testing.assert_allclose(run.species[-1], ss.concentrations, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            ss.output_sensitivities(["expression:flux"]),
            run.output_sensitivities(["expression:flux"], axis="parameter")[-1],
            rtol=1e-4,
            atol=1e-8,
        )

    def test_model_is_left_with_nominal_parameter_values(self):
        """The probe writes the Parameter vector and re-derives; the restore must
        undo BOTH, or the derived parameters keep their perturbed values and
        corrupt every later column and the caller's model."""
        m = bngsim.Model.from_net(DERIVED_NET)
        sim = bngsim.Simulator(m, method="ode")
        ss = sim.steady_state(sensitivity_params=["kon", "chi", "koff"])
        assert ss.converged
        assert m.get_param("kon") == pytest.approx(self.KON, rel=0, abs=0)
        assert m.get_param("_rateLaw1") == pytest.approx(self.CHI * self.KON, rel=1e-15)


# ── Names / shapes / species passthrough ─────────────────────────────────────


class TestNamesShapesAndSpecies:
    def test_names_populated_on_sensitivity_run(self):
        ss = _steady_state(CLOSED_NET, ["k1"])
        assert ss.observable_names == ["A_tot", "BC2", "Total"]
        # _rateLawN internals filtered out, matching Result.expression_names.
        assert ss.expression_names == ["satA", "lin"]

    def test_shapes(self):
        params = ["k1", "k3"]
        ss = _steady_state(CLOSED_NET, params)
        assert ss.sensitivities_observables.shape == (3, 2)
        assert ss.sensitivities_expressions.shape == (2, 2)
        assert ss.output_sensitivities(["observable:A_tot", "expression:lin"]).shape == (2, 2)

    def test_species_selector_matches_sensitivity_row(self):
        ss = _steady_state(OPEN_NET, ["k_prod", "k_deg"])
        got = ss.output_sensitivities(["species:S()"])
        np.testing.assert_allclose(got[0], ss.sensitivity[0], rtol=0, atol=0)

    def test_single_selector_string(self):
        ss = _steady_state(OPEN_NET, ["k_prod", "k_deg"])
        # A bare string behaves like a one-element list.
        one = ss.output_sensitivities("observable:Stot")
        assert one.shape == (1, 2)

    def test_empty_selector_list(self):
        ss = _steady_state(OPEN_NET, ["k_prod", "k_deg"])
        got = ss.output_sensitivities([])
        assert got.shape == (0, 2)

    def test_bare_and_aliased_selectors(self):
        ss = _steady_state(CLOSED_NET, ["k1"])
        # Bare unique name, plus function:/state: aliases.
        bare = ss.output_sensitivities(["A_tot"])
        typed = ss.output_sensitivities(["observable:A_tot"])
        np.testing.assert_allclose(bare, typed, atol=0)
        # function: alias for expression:
        assert ss.output_sensitivities(["function:satA"]).shape == (1, 1)


# ── Error handling ───────────────────────────────────────────────────────────


class TestErrors:
    def test_ic_axis_unavailable(self):
        ss = _steady_state(OPEN_NET, ["k_prod"])
        with pytest.raises(ValueError, match="ic.*axis|initial condition|∂x"):
            ss.output_sensitivities(["observable:Stot"], axis="ic")

    def test_bad_axis(self):
        ss = _steady_state(OPEN_NET, ["k_prod"])
        with pytest.raises(ValueError, match="axis must be"):
            ss.output_sensitivities(["observable:Stot"], axis="bogus")

    def test_no_sensitivity_requested(self):
        m = bngsim.Model.from_net(OPEN_NET)
        sim = bngsim.Simulator(m, method="ode")
        ss = sim.steady_state()  # no sensitivity_params
        assert ss.sensitivity is None
        with pytest.raises(ValueError, match="no steady-state sensitivities"):
            ss.output_sensitivities(["observable:Stot"])

    def test_unknown_selector(self):
        ss = _steady_state(OPEN_NET, ["k_prod"])
        with pytest.raises(ValueError, match="Unresolved|no observable named"):
            ss.output_sensitivities(["observable:does_not_exist"])

    def test_unknown_kind(self):
        ss = _steady_state(OPEN_NET, ["k_prod"])
        with pytest.raises(ValueError, match="Unknown selector kind"):
            ss.output_sensitivities(["bogus:Stot"])


# ── The compiled d(func)/dp evaluator, and its fallback (issue #75) ──────────


@contextmanager
def _as_before_75():
    """Put ``steady_state()`` back on its pre-issue-#75 codegen prep: attach an
    artifact without the ``_want_output_sens`` re-prep, so
    ``bngsim_codegen_output_sens`` is never emitted and the block
    finite-differences.

    Reproduces the pre-fix path without stashing the source. The patch goes on the
    class (``Simulator`` uses ``__slots__``), and is restored by hand rather than
    through ``monkeypatch.undo()`` — the ``monkeypatch`` fixture is shared with the
    autouse ``_force_codegen`` above, so an ``undo()`` mid-test would silently drop
    its environment too.
    """
    from bngsim._simulator import _codegen_jit_backend

    def _pre_75(sim):
        sim._auto_codegen_for_sensitivity(jit_backend=_codegen_jit_backend())

    saved = bngsim.Simulator._prepare_output_sens_codegen
    bngsim.Simulator._prepare_output_sens_codegen = _pre_75
    try:
        yield
    finally:
        bngsim.Simulator._prepare_output_sens_codegen = saved


def _core_steady_state(net, params):
    """``Simulator.steady_state`` down at the core, for the RAW function block the
    Python wrapper filters to the user-selectable rows."""
    from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

    m = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(m, method="ode")
    sim._prepare_output_sens_codegen()
    opts = SteadyStateOptions()
    opts.method = "integration"
    opts.jacobian = "auto"
    if sim._codegen_so_path:
        opts.codegen_so_path = sim._codegen_so_path
    if sim._codegen_c_source:
        opts.codegen_c_source = sim._codegen_c_source
    opts.sensitivity_params = list(params)
    return find_steady_state(m._core, opts)


class TestCompiledOutputSensPath:
    """``d(func)/dp`` runs through ``bngsim_codegen_output_sens`` (issue #75).

    The measured blocker #75 records: the symbol is emitted only when the model
    carries ``_want_output_sens``, which ``Simulator.__init__`` sets from its
    *constructor* ``sensitivity_params`` — while ``steady_state()`` takes its own
    as a *method* argument. So the documented usage
    ``Simulator(m, method="ode").steady_state(sensitivity_params=[...])`` reached
    the solver with no symbol to call, and every test in this file measured the
    finite-difference fallback.
    """

    @pytest.mark.parametrize(
        "net, params",
        [
            (CLOSED_NET, ["k1", "k3", "amp", "Km"]),
            (OPEN_NET, ["k_prod", "k_deg", "scale"]),
            (DERIVED_NET, ["kon", "chi", "koff", "_rateLaw1"]),
        ],
    )
    def test_ordinary_construction_gets_the_compiled_evaluator(self, net, params):
        m = bngsim.Model.from_net(net)
        sim = bngsim.Simulator(m, method="ode")  # no constructor sensitivity_params
        assert m._want_output_sens is False
        ss = sim.steady_state(sensitivity_params=params)
        assert ss.converged
        # The re-prep ran, and every function row came from the compiled chain rule.
        assert m._want_output_sens is True
        assert ss.sens_output_source == "codegen"

    def test_plain_artifact_does_not_shadow_the_sensitivity_one(self):
        """``codegen=True`` attaches a plain-RHS ``.so`` at construction, built
        without output sens. ``_auto_codegen_for_sensitivity`` no-ops on an
        already-attached artifact, so without the drop-and-regenerate half of the
        dance that plain ``.so`` shadows the one carrying the symbol."""
        m = bngsim.Model.from_net(DERIVED_NET)
        sim = bngsim.Simulator(m, method="ode", codegen=True)
        assert sim._codegen_so_path or sim._codegen_c_source  # attached at ctor
        ss = sim.steady_state(sensitivity_params=["kon", "chi", "koff"])
        assert ss.sens_output_source == "codegen"

    def test_beats_finite_differences_against_the_closed_form(self):
        """Accuracy, the reason #75 leads with it: the FD explicit term is a
        one-sided √eps quotient. Same model, same closed form, both paths."""
        params = ["kon", "chi", "koff", "_rateLaw1"]
        cf = TestDerivedParameterChain()._closed_form()
        exact = np.array([cf["dflux"][p] for p in params])

        with _as_before_75():
            fd = _steady_state(DERIVED_NET, params)
        cg = _steady_state(DERIVED_NET, params)

        assert fd.sens_output_source == "finite-difference"
        assert cg.sens_output_source == "codegen"

        fd_got = fd.output_sensitivities(["expression:flux"])[0]
        cg_got = cg.output_sensitivities(["expression:flux"])[0]
        # Both are right; the compiled one is right by a wide margin (measured
        # 8.6e-8 vs 1.1e-9 — and that floor is the steady-state solve, not the
        # chain rule). An order of magnitude is the assertion; the difference is
        # nearly two.
        np.testing.assert_allclose(fd_got, exact, rtol=1e-5)
        fd_err = np.max(np.abs(fd_got - exact) / exact)
        cg_err = np.max(np.abs(cg_got - exact) / exact)
        assert cg_err < fd_err / 10.0

    def test_fallback_still_answers_when_the_symbol_is_absent(self):
        """The FD block is a live fallback, not dead code: with no symbol the
        answers are unchanged (that is what every other test here measured before
        #75), and the result says so."""
        params = ["k1", "k3", "amp", "Km"]
        with _as_before_75():
            fd = _steady_state(CLOSED_NET, params)
        cg = _steady_state(CLOSED_NET, params)

        assert fd.sens_output_source == "finite-difference"
        assert cg.sens_output_source == "codegen"
        sel = ["expression:satA", "expression:lin"]
        np.testing.assert_allclose(
            cg.output_sensitivities(sel), fd.output_sensitivities(sel), rtol=1e-6, atol=1e-9
        )

    def test_regeneration_failure_keeps_the_working_artifact(self, monkeypatch):
        """The re-prep DROPS an attached artifact to rebuild it with output sens.
        If that rebuild fails, the drop must not turn a call that used to succeed
        into a refusal — the old artifact goes back and the block differences."""
        m = bngsim.Model.from_net(DERIVED_NET)
        sim = bngsim.Simulator(m, method="ode", codegen=True)

        def _boom(self, **kw):
            if not (self._codegen_so_path or self._codegen_c_source):
                raise RuntimeError("simulated codegen failure")

        monkeypatch.setattr(bngsim.Simulator, "_auto_codegen_for_sensitivity", _boom)
        ss = sim.steady_state(sensitivity_params=["kon", "chi", "koff"])
        assert ss.converged
        assert ss.rhs_backend.startswith("codegen")  # the old artifact came back
        assert ss.sens_output_source == "finite-difference"

    def test_regeneration_failure_with_nothing_to_restore_propagates(self, monkeypatch):
        """With no artifact to put back, the refusal is the real GH #214 one."""
        m = bngsim.Model.from_net(DERIVED_NET)
        sim = bngsim.Simulator(m, method="ode")

        def _boom(self, **kw):
            raise RuntimeError("simulated codegen failure")

        monkeypatch.setattr(bngsim.Simulator, "_auto_codegen_for_sensitivity", _boom)
        with pytest.raises(RuntimeError, match="simulated codegen failure"):
            sim.steady_state(sensitivity_params=["kon"])

    def test_declined_model_is_all_finite_difference(self):
        """A whole-model decline emits no symbol at all — here because every
        function is an auto-generated ``_rateLawN`` and #198 differentiates only
        the user-selectable closure. The block must still be filled."""
        m = bngsim.Model.from_net(DECLINED_NET)
        sim = bngsim.Simulator(m, method="ode")
        ss = sim.steady_state(sensitivity_params=["k1"])
        assert ss.converged
        assert ss.expression_names == []  # nothing user-facing to select
        assert ss.sens_output_source == "finite-difference"

    # ── The mixed block ──────────────────────────────────────────────────────

    KON, CHI, KOFF, GAIN = 1.0, 10.0, 0.5, 2.0

    def _mixed_closed_form(self):
        a, b = self.CHI * self.KON, self.KOFF
        d = (a + b) ** 2
        return {
            # flux = a·A_ss = a·b/(a+b) — the "ok" row, from the codegen.
            "flux": {
                "kon": self.CHI * b**2 / d,
                "chi": self.KON * b**2 / d,
                "koff": a**2 / d,
                "gain": 0.0,
            },
            # capA = gain·A_ss (max() selects A_tot) — the "unsupported" row, FD.
            "capA": {
                "kon": -self.GAIN * self.CHI * b / d,
                "chi": -self.GAIN * self.KON * b / d,
                "koff": self.GAIN * a / d,
                "gain": b / (a + b),
            },
            # _rateLaw2 = koff — the "skipped" row, also FD.
            "_rateLaw2": {"kon": 0.0, "chi": 0.0, "koff": 1.0, "gain": 0.0},
        }

    def test_mixed_block_is_reported_and_correct(self):
        """One solve, one function of each #198 status. The compiled evaluator
        answers ``flux``; ``capA`` (``max()``) comes back as a NaN sentinel and
        ``_rateLaw2`` (outside the differentiated closure) is left untouched, so
        both fall through to FD — and neither may leak a NaN or a silent zero.
        """
        params = ["kon", "chi", "koff", "gain"]
        m = bngsim.Model.from_net(MIXED_NET)
        sim = bngsim.Simulator(m, method="ode")
        ss = sim.steady_state(sensitivity_params=params)
        assert ss.converged
        assert ss.sens_output_source == "mixed"
        assert ss.expression_names == ["flux", "capA"]  # _rateLaw2 filtered out

        cf = self._mixed_closed_form()
        got = ss.output_sensitivities(["expression:flux", "expression:capA"])
        assert np.all(np.isfinite(got)), "a declined row leaked its NaN sentinel"
        np.testing.assert_allclose(
            got,
            [[cf["flux"][p] for p in params], [cf["capA"][p] for p in params]],
            rtol=1e-5,
            atol=1e-8,
        )

    def test_skipped_row_keeps_its_finite_difference_value(self):
        """The emitter leaves a function outside the user closure at whatever the
        caller pre-filled. Pre-filling with NaN is what routes it to FD instead —
        pre-filling with zero would have made the RAW block read a confident 0.0
        for ``d(_rateLaw2)/dkoff``, which is 1.0.

        The skipped row is filtered out of the user-facing block, so this reads
        the core result's ``raw_expression_*`` directly.
        """
        params = ["kon", "chi", "koff", "gain"]
        core = _core_steady_state(MIXED_NET, params)
        assert core.sens_output_source == "mixed"
        raw_names = list(core.raw_expression_names)
        assert raw_names == ["flux", "capA", "_rateLaw2"]
        cf = self._mixed_closed_form()["_rateLaw2"]
        np.testing.assert_allclose(
            core.raw_expression_sensitivity_data[raw_names.index("_rateLaw2")],
            [cf[p] for p in params],
            rtol=1e-6,
            atol=1e-9,
        )


# ── Amount-valued (hOSU) observables carry their volume factor (issue #119) ──


# Birth-death on one SBML species with hasOnlySubstanceUnits="true", in a
# compartment of size V. `{V}` is substituted per parametrization.
#
#   d(amount)/dt = k_prod - k_deg·amount,  k_prod = 4, k_deg = 0.5
#     amount*            = k_prod/k_deg   = 8
#     d(amount*)/dk_prod = 1/k_deg        = 2
#     d(amount*)/dk_deg  = -k_prod/k_deg² = -16
#
# The loader registers a same-named observable shadowing the species, so the
# observable denotes the AMOUNT and its sensitivity is the closed form above —
# independent of V. The stored (concentration) state and its sensitivity are that
# divided by V, which is what makes the two blocks distinguishable.
_HOSU_BIRTHDEATH_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_birthdeath">
    <listOfCompartments>
      <compartment id="cell" size="{V}" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_prod" value="4" constant="true"/>
      <parameter id="k_deg" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="birth" reversible="false">
        <listOfProducts>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <ci>k_prod</ci>
        </math></kineticLaw>
      </reaction>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k_deg</ci><ci>X</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

_HOSU_PARAMS = ["k_prod", "k_deg"]
_HOSU_AMOUNT_SENS = np.array([2.0, -16.0])  # d(amount*)/d[k_prod, k_deg]


def _hosu_steady_state(volume):
    m = bngsim.Model.from_sbml_string(_HOSU_BIRTHDEATH_SBML.format(V=volume))
    sim = bngsim.Simulator(m, method="ode")
    return m, sim.steady_state(sensitivity_params=_HOSU_PARAMS)


class TestAmountValuedObservableVolumeFactor:
    """``d(observable)/dp`` must use the same weight the observable's VALUE does.

    ``update_observables`` defines ``obs_j = Σ_i factor_ji·v_i·x_i``, where ``v_i``
    is the species' ``volume_factor`` when it is ``amount_valued`` (SBML
    ``hasOnlySubstanceUnits="true"``) — a hOSU symbol denotes an amount, not the
    stored concentration. The steady-state projection used the bare
    ``entry.factor``, so it returned a derivative of something the result does not
    report, off by the compartment volume. The CVODES path (``obs_sens_terms``) and
    the compiled emitter (``_emit_obs_sens_lines``) always carried ``v_i``.
    """

    @pytest.mark.parametrize("volume", [1.0, 2.0, 3.0])
    def test_matches_closed_form_at_any_compartment_volume(self, volume):
        """The sharpest statement of the bug: the true answer does not depend on
        the compartment volume, and the pre-fix one was inversely proportional to
        it (V=1 correct, V=2 half, V=3 a third)."""
        m, ss = _hosu_steady_state(volume)
        assert ss.converged
        # The state is stored as a concentration, so it DOES scale with V — this
        # pins that the fixture really has a non-unit factor to get wrong.
        assert ss["X"] == pytest.approx(8.0 / volume, rel=1e-6)
        np.testing.assert_allclose(
            ss.sensitivities_observables[0], _HOSU_AMOUNT_SENS, rtol=1e-6, atol=1e-9
        )

    def test_species_row_is_not_given_the_factor(self):
        """Guard against over-correcting: ``ss.sensitivity`` is the STORED-state
        derivative and must stay unscaled, i.e. the amount row divided by V."""
        m, ss = _hosu_steady_state(2.0)
        np.testing.assert_allclose(
            ss.sensitivity[0], _HOSU_AMOUNT_SENS / 2.0, rtol=1e-6, atol=1e-9
        )
        # ...and the two blocks must therefore genuinely differ, or this test and
        # the one above could both pass on a single wrong weight.
        assert not np.allclose(ss.sensitivity[0], ss.sensitivities_observables[0])

    def test_pre_fix_value_is_far_from_the_answer(self):
        """Pin the failure mode, not just the answer: the dropped-factor value is
        a clean 2x away, so an rtol assertion cannot pass for the wrong reason."""
        m, ss = _hosu_steady_state(2.0)
        dropped_factor = _HOSU_AMOUNT_SENS / 2.0  # what entry.factor alone gave
        got = ss.sensitivities_observables[0]
        # State the separation as the factor of two it is, with a tolerance.
        # `max|got - dropped| > 0.5*max|got|` says the same thing exactly, and
        # is a knife edge: both sides are half the answer, so it reduces to
        # `|got| > |exact|` and turns on the last digit of the steady state.
        # Issue #127 moved that digit — installing the solver's own Jacobian
        # stops the march one roundoff below 4.0 instead of one above.
        assert np.max(np.abs(got - dropped_factor)) == pytest.approx(
            0.5 * np.max(np.abs(_HOSU_AMOUNT_SENS)), rel=1e-6
        )
        np.testing.assert_allclose(got, _HOSU_AMOUNT_SENS, rtol=1e-6, atol=1e-9)

    def test_matches_the_cvode_run(self):
        """Same-model consistency, the invariant the bug broke: the steady-state
        block and a converged forward-sensitivity run's last point must agree."""
        m, ss = _hosu_steady_state(2.0)
        m2 = bngsim.Model.from_sbml_string(_HOSU_BIRTHDEATH_SBML.format(V=2.0))
        sim2 = bngsim.Simulator(m2, method="ode", sensitivity_params=_HOSU_PARAMS)
        m2.reset()
        run = sim2.run(t_span=(0.0, 400.0), n_points=2, **_RUN)
        np.testing.assert_allclose(run.species[-1], ss.concentrations, rtol=1e-6, atol=1e-8)
        run_obs = run.output_sensitivities(["observable:X"], axis="parameter")[-1]
        np.testing.assert_allclose(
            ss.output_sensitivities(["observable:X"]), run_obs, rtol=1e-6, atol=1e-9
        )
