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

    **Suppressing the re-prep is no longer enough to isolate one symbol** (issue
    #217). ``_want_output_sens`` gates three emissions now — GH #198's
    ``bngsim_codegen_output_sens``, #177's ``bngsim_codegen_sens_term_scale``, and
    since #209/#217 ``bngsim_codegen_sens_rhs`` itself — so a bare
    ``_auto_codegen_for_sensitivity`` builds an artifact missing all three. That is
    not the pre-#75 state; pre-#75 had ∂f/∂p compiled and only ``d(func)/dp``
    finite-differenced, which is what the accuracy comparison below is about. So the
    re-prep runs normally and exactly one emitter is silenced.

    The private cache dir is not optional with that patch in place: the key says
    "output sensitivities wanted" while the source deliberately lacks the symbol, so
    sharing the real cache would hand this artifact to a later sensitivity run — the
    issue #51 inertness trap, inside the suite.
    """
    import tempfile

    from bngsim import _codegen as cg

    saved_prep = bngsim.Simulator._prepare_output_sens_codegen
    saved_emit = cg.generate_output_sens_from_model
    saved_cache = cg.CACHE_DIR
    tmp = tempfile.TemporaryDirectory()
    try:
        cg.generate_output_sens_from_model = lambda model: None
        cg.CACHE_DIR = Path(tmp.name)
        yield
    finally:
        bngsim.Simulator._prepare_output_sens_codegen = saved_prep
        cg.generate_output_sens_from_model = saved_emit
        cg.CACHE_DIR = saved_cache
        tmp.cleanup()


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

    def test_matches_the_closed_form(self):
        """The compiled row against the closed form. Before issue #1002 a row
        with no compiled code was differenced, a one-sided √eps quotient
        right to 8.6e-8 here where the compiled one is right to 1.1e-9."""
        params = ["kon", "chi", "koff", "_rateLaw1"]
        cf = TestDerivedParameterChain()._closed_form()
        exact = np.array([cf["dflux"][p] for p in params])
        cg = _steady_state(DERIVED_NET, params)
        assert cg.sens_output_source == "codegen"
        cg_got = cg.output_sensitivities(["expression:flux"])[0]
        np.testing.assert_allclose(cg_got, exact, rtol=1e-8)

    def test_no_symbol_is_answered_by_dual_numbers(self):
        """With no symbol every function row was differenced before issue
        #1002, and NaN after it. Each is exact now, by dual numbers through the
        interpreter (issue #1330): the rows the compiled chain rule gives."""
        params = ["k1", "k3", "amp", "Km"]
        with _as_before_75():
            ad = _steady_state(CLOSED_NET, params)
        cg = _steady_state(CLOSED_NET, params)
        assert ad.sens_output_source == "automatic-differentiation"
        assert cg.sens_output_source == "codegen"
        selection = ["expression:satA", "expression:lin"]
        np.testing.assert_allclose(
            ad.output_sensitivities(selection),
            cg.output_sensitivities(selection),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_no_symbol_is_refused_by_name_not_differenced(self, monkeypatch):
        """With no dual evaluation either, the rows are NaN, and
        ``output_sensitivities`` refuses each by name: a quotient of a function
        over the state can be wrong with nothing to show it."""
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        params = ["k1", "k3", "amp", "Km"]
        with _as_before_75():
            fd = _steady_state(CLOSED_NET, params)
        cg = _steady_state(CLOSED_NET, params)

        assert fd.sens_output_source == "none"
        assert cg.sens_output_source == "codegen"
        np.testing.assert_array_equal(fd.sensitivity, cg.sensitivity)
        np.testing.assert_allclose(
            fd.output_sensitivities(["observable:A_tot"]),
            cg.output_sensitivities(["observable:A_tot"]),
            rtol=1e-12,
        )
        with pytest.raises(ValueError) as refusal:
            fd.output_sensitivities(["expression:satA", "expression:lin"])
        message = str(refusal.value)
        assert "'expression:satA'" in message and "'expression:lin'" in message
        assert "the compiled code it was solved with carries no chain rule" in message
        assert "issue #1002" in message

    def test_regeneration_failure_keeps_the_working_artifact(self, monkeypatch):
        """The re-prep DROPS an attached artifact to rebuild it with output sens.
        If that rebuild fails, the drop must not turn a call that used to succeed
        into a refusal — the old artifact goes back. Its block has no chain
        rule, and is not differenced (issue #1002): the solve returns, and the
        function rows are refused by name. (With no dual evaluation: with one,
        they are exact, issue #1330.)"""
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        m = bngsim.Model.from_net(DERIVED_NET)
        sim = bngsim.Simulator(m, method="ode", codegen=True)

        def _boom(self, **kw):
            if not (self._codegen_so_path or self._codegen_c_source):
                raise RuntimeError("simulated codegen failure")

        monkeypatch.setattr(bngsim.Simulator, "_auto_codegen_for_sensitivity", _boom)
        ss = sim.steady_state(sensitivity_params=["kon", "chi", "koff"])
        assert ss.converged
        assert ss.rhs_backend.startswith("codegen")  # the old artifact came back
        assert ss.sens_output_source == "none"
        assert np.isfinite(ss.sensitivity).all()
        with pytest.raises(ValueError, match=r"'expression:flux'.*carries no chain rule"):
            ss.output_sensitivities(["expression:flux"])

    def test_regeneration_failure_with_nothing_to_restore_propagates(self, monkeypatch):
        """With no artifact to put back, the refusal is the real GH #214 one.

        ``codegen=False`` is what leaves nothing to put back: since #803 a .net
        model auto-codegens at construction like any other, and ``_force_codegen``
        drops the threshold to one species."""
        m = bngsim.Model.from_net(DERIVED_NET)
        sim = bngsim.Simulator(m, method="ode", codegen=False)
        assert not (sim._codegen_so_path or sim._codegen_c_source)

        def _boom(self, **kw):
            raise RuntimeError("simulated codegen failure")

        monkeypatch.setattr(bngsim.Simulator, "_auto_codegen_for_sensitivity", _boom)
        with pytest.raises(RuntimeError, match="simulated codegen failure"):
            sim.steady_state(sensitivity_params=["kon"])

    def test_declined_model_has_no_block_to_report(self):
        """A whole-model decline emits no symbol at all — here because every
        function is an auto-generated ``_rateLawN`` and #198 differentiates only
        the user-selectable closure. No function is returned, so there is
        nothing to report (it said "finite-difference" before issue #1002)."""
        m = bngsim.Model.from_net(DECLINED_NET)
        sim = bngsim.Simulator(m, method="ode")
        ss = sim.steady_state(sensitivity_params=["k1"])
        assert ss.converged
        assert ss.expression_names == []  # nothing user-facing to select
        assert ss.sens_output_source == ""

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
            # capA = gain·A_ss (max() selects A_tot) — the "unsupported" row.
            "capA": {
                "kon": -self.GAIN * self.CHI * b / d,
                "chi": -self.GAIN * self.KON * b / d,
                "koff": self.GAIN * a / d,
                "gain": b / (a + b),
            },
            # _rateLaw2 = koff — the "skipped" row.
            "_rateLaw2": {"kon": 0.0, "chi": 0.0, "koff": 1.0, "gain": 0.0},
        }

    def test_mixed_block_is_exact(self):
        """One solve, one function of each #198 status. The compiled evaluator
        answers ``flux``; ``capA`` (``max()``) it leaves, and dual numbers give
        it exactly (issue #1330); both are their closed forms."""
        params = ["kon", "chi", "koff", "gain"]
        m = bngsim.Model.from_net(MIXED_NET)
        ss = bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=params)
        assert ss.sens_output_source == "codegen+automatic-differentiation"
        cf = self._mixed_closed_form()
        got = ss.output_sensitivities(["expression:flux", "expression:capA"])
        want = [[cf[f][p] for p in params] for f in ("flux", "capA")]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-8)

    def test_mixed_block_is_reported_and_correct(self, monkeypatch):
        """With no dual evaluation: ``capA`` comes back as a NaN sentinel and
        ``_rateLaw2`` (outside the differentiated closure) is left untouched.
        ``capA`` was differenced before issue #1002, and is refused by name now,
        with the construct; ``flux`` is returned, and neither row is a silent 0.
        """
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        params = ["kon", "chi", "koff", "gain"]
        m = bngsim.Model.from_net(MIXED_NET)
        sim = bngsim.Simulator(m, method="ode")
        ss = sim.steady_state(sensitivity_params=params)
        assert ss.converged
        assert ss.sens_output_source == "partial"
        assert ss.expression_names == ["flux", "capA"]  # _rateLaw2 filtered out

        cf = self._mixed_closed_form()
        got = ss.output_sensitivities(["expression:flux"])
        np.testing.assert_allclose(got, [[cf["flux"][p] for p in params]], rtol=1e-5, atol=1e-8)
        with pytest.raises(ValueError, match=r"'expression:capA'.*unsupported construct: max\(\)"):
            ss.output_sensitivities(["expression:flux", "expression:capA"])
        assert np.isnan(ss.sensitivities_expressions[1]).all()

    def test_skipped_row_is_exact(self):
        """The emitter leaves a function outside the user closure at whatever the
        caller pre-filled; dual numbers give it, ``d(_rateLaw2)/dθ`` = (0, 0,
        1, 0) (issue #1330), not a confident 0.0 for dkoff. The skipped row is
        filtered out of the user-facing block, so this reads the core result's
        ``raw_expression_*`` directly."""
        params = ["kon", "chi", "koff", "gain"]
        core = _core_steady_state(MIXED_NET, params)
        raw_names = list(core.raw_expression_names)
        np.testing.assert_array_equal(
            core.raw_expression_sensitivity_data[raw_names.index("_rateLaw2")],
            [self._mixed_closed_form()["_rateLaw2"][p] for p in params],
        )

    def test_skipped_row_is_not_a_confident_zero(self, monkeypatch):
        """With no dual evaluation, pre-filled with zero, the RAW block would
        read a confident 0.0 for ``d(_rateLaw2)/dkoff``, which is 1.0. It is
        NaN: the pre-fill, which nothing differences now (issue #1002).
        """
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        params = ["kon", "chi", "koff", "gain"]
        core = _core_steady_state(MIXED_NET, params)
        assert core.sens_output_source == "partial"
        raw_names = list(core.raw_expression_names)
        assert raw_names == ["flux", "capA", "_rateLaw2"]
        assert np.isnan(core.raw_expression_sensitivity_data[raw_names.index("_rateLaw2")]).all()


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


# ── The function block beside a species with no row ──────────────────────────
#
# ``P`` is made at ``kp`` and lost at ``kd*P``: P* = kp/kd = 5, and
# dP*/d(kp, kd) = (1/kd, -kp/kd²) = (5, -25). An if() conditional is a function
# the compiled evaluator declines, so ``piecewise(...)`` put a row through the
# finite-difference fallback, which multiplied every species' row into every
# function's: a function that does not read a species with no row took 0·NaN.
# A rule target (``T := ...``) is left out of the solve, and so is a species
# ``mask=`` leaves out, and the row of each is NaN. Since issue #1002 nothing
# is differenced: ``piecewise(...)`` is refused by name, and the compiled
# functions here are what carries the rows.

_MADE_AND_LOST = (
    "compartment D = 1; species P in D; P = 0; kp = 1; kd = 0.2;\nR3: => P; kp; R4: P => ; kd*P;\n"
)
_DP = np.array([5.0, -25.0])
_KP_KD = ["kp", "kd"]
_TIGHT = {"tol": 1e-12}


def _ant_steady_state(text, params=_KP_KD, values=None, **kw):
    m = bngsim.Model.from_antimony_string(text)
    for name, value in (values or {}).items():
        m.set_param(name, value)
    if kw.pop("mask_sinks", False):
        kw["mask"] = ~m.is_pure_sink()
    return bngsim.Simulator(m, method="ode").steady_state(
        sensitivity_params=list(params) if params else None, **_TIGHT, **kw
    )


def _expression_value(ss, name):
    return float(np.asarray(ss._expression_values)[ss.expression_names.index(name)])


def _central_row(text, read, rel=1e-4):
    """d(read(ss))/d(kp, kd) by central differences of steady_state() itself,
    each solve on a fresh model."""
    base = {"kp": 1.0, "kd": 0.2}
    row = []
    for name in _KP_KD:
        h = rel * base[name]
        up = _ant_steady_state(text, None, {**base, name: base[name] + h})
        down = _ant_steady_state(text, None, {**base, name: base[name] - h})
        row.append((read(up) - read(down)) / (2 * h))
    return np.array(row)


class TestFunctionBlockBesideARowlessSpecies:
    def test_a_rule_target_does_not_poison_an_expression(self):
        """Control. ``W`` is ``P`` here, so its row is (5, -25). Main returned
        NaN once ``T := 2*P`` was beside it, though ``W`` does not read ``T``,
        where ``W`` was differenced."""
        text = _MADE_AND_LOST + "W := P;\nspecies T in D; T := 2*P;\n"
        ss = _ant_steady_state(text)
        assert ss.sens_output_source == "codegen"
        assert [ss.species_names[i] for i in ss.excluded_species] == ["T"]
        got = ss.output_sensitivities(["expression:W"])[0]
        np.testing.assert_allclose(got, _DP, rtol=1e-7)
        np.testing.assert_allclose(
            got, _central_row(text, lambda s: _expression_value(s, "W")), rtol=1e-6
        )
        np.testing.assert_allclose(ss.output_sensitivities(["species:T"])[0], 2 * _DP, rtol=1e-7)

    def test_a_declined_expression_beside_a_rule_target_is_exact(self):
        """``W := piecewise(P, ...)`` has no compiled derivative; dual numbers
        give it, d(W)/dθ = dP*/dθ on the branch P* = 5 is on (issue #1330), and
        T's row is its rule's."""
        text = _MADE_AND_LOST + "W := piecewise(P, P < 100, 2*P);\nspecies T in D; T := 2*P;\n"
        ss = _ant_steady_state(text)
        np.testing.assert_allclose(ss.output_sensitivities(["expression:W"])[0], _DP, rtol=1e-7)
        np.testing.assert_allclose(ss.output_sensitivities(["species:T"])[0], 2 * _DP, rtol=1e-7)

    def test_a_declined_expression_beside_a_rule_target_is_refused_by_name(self, monkeypatch):
        """With no dual evaluation it is refused by name, and T's row is its
        rule's (issue #1002)."""
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        text = _MADE_AND_LOST + "W := piecewise(P, P < 100, 2*P);\nspecies T in D; T := 2*P;\n"
        ss = _ant_steady_state(text)
        assert ss.sens_output_source == "none"
        with pytest.raises(ValueError, match=r"'expression:W'.*if\(\) conditional.*#1002"):
            ss.output_sensitivities(["expression:W"])
        np.testing.assert_allclose(ss.output_sensitivities(["species:T"])[0], 2 * _DP, rtol=1e-7)

    def test_a_rule_target_whose_rule_is_declined_is_exact(self):
        """``T := piecewise(2*P, ...)``: T's row is its rule's, which dual
        numbers give, 2·dP*/dθ on the branch P* = 5 is on (issue #1330)."""
        text = _MADE_AND_LOST + "species T in D; T := piecewise(2*P, P < 100, 3*P);\n"
        ss = _ant_steady_state(text)
        np.testing.assert_allclose(ss.output_sensitivities(["species:T"])[0], 2 * _DP, rtol=1e-7)
        np.testing.assert_allclose(ss.output_sensitivities(["species:P"])[0], _DP, rtol=1e-7)

    def test_a_rule_target_whose_rule_is_declined_is_refused_by_name(self, monkeypatch):
        """With no dual evaluation T's row is NaN and ``species:T`` is refused
        naming the rule (issue #1002). It was differenced before."""
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        text = _MADE_AND_LOST + "species T in D; T := piecewise(2*P, P < 100, 3*P);\n"
        ss = _ant_steady_state(text)
        assert np.isnan(np.asarray(ss.sensitivity)[ss.species_names.index("T")]).all()
        with pytest.raises(
            ValueError, match=r"'species:T'.*none for its rule 'T' \(uses unsupported"
        ):
            ss.output_sensitivities(["species:T"])
        np.testing.assert_allclose(ss.output_sensitivities(["species:P"])[0], _DP, rtol=1e-7)

    @pytest.mark.parametrize(
        "rule,want",
        [
            ("2*P", 2 * _DP),  # an observable 2·P
            ("P^2", 2 * 5.0 * _DP),  # a function: d(P²) = 2P·dP
        ],
    )
    def test_an_expression_that_reads_a_rule_target_has_the_rule_s_derivative(self, rule, want):
        """``Y`` is ``T``. It reads the rule under ``T``'s name, not ``T``'s own
        slot, so its derivative comes through ``P``'s row: (10, -50) for
        ``2*P`` and (50, -250) for ``P^2``. Main returned NaN for both where
        ``Y`` was differenced; compiled, it has the rule's derivative."""
        text = _MADE_AND_LOST + f"Y := T;\nspecies T in D; T := {rule};\n"
        ss = _ant_steady_state(text)
        np.testing.assert_allclose(ss.output_sensitivities(["expression:Y"])[0], want, rtol=1e-7)
        np.testing.assert_allclose(ss.output_sensitivities(["species:T"])[0], want, rtol=1e-7)

    def test_a_masked_species_is_refused_only_in_what_reads_it(self):
        """``Z`` collects what ``P`` loses at ``kz*P`` and has no steady value,
        so ``mask=`` leaves it out. P* = kp/(kd + kz), and ``W`` = ``P`` has the
        row (1/0.3, -1/0.09); main returned NaN where it was differenced. ``V``
        reads ``Z`` and has no derivative: its row was NaN, and is now refused
        by name (issue #1124, item 10). The species tensor keeps Z's NaN row."""
        text = _MADE_AND_LOST + (
            "species Z in D; Z = 0; kz = 0.1; R5: P => Z; kz*P;\nW := P; V := 2*Z;\n"
        )
        ss = _ant_steady_state(text, mask_sinks=True)
        assert [ss.species_names[i] for i in ss.excluded_species] == ["Z"]
        np.testing.assert_allclose(
            ss.output_sensitivities(["expression:W"])[0], [1 / 0.3, -1 / 0.09], rtol=1e-7
        )
        with pytest.raises(ValueError, match=r"'expression:V'.*reads Z, which mask= left out"):
            ss.output_sensitivities(["expression:V"])
        with pytest.raises(ValueError, match=r"'species:Z'.*is Z, which mask= left out"):
            ss.output_sensitivities(["W", "species:Z"])
        assert np.isnan(np.asarray(ss.sensitivity)[ss.species_names.index("Z")]).all()

    # ``C := 1 + P`` is 1 + P* = 6, and ``T`` is an amount 2·P in it, reported
    # over the live size: [T] = 2P/C, d[T] = 2/C²·dP.
    _SIZED = "C := 1 + P;\nsubstanceOnly species T in C; T := 2*P;\n"

    def test_a_size_by_a_rule_gives_an_amount_its_row(self):
        """Control. Main raised that ``mask=`` had left the size out of the
        solve, with no ``mask=`` given, where the size's rule was differenced:
        the size's row was NaN from ``T``'s."""
        text = _MADE_AND_LOST + self._SIZED
        ss = _ant_steady_state(text)
        np.testing.assert_allclose(ss.output_sensitivities(["expression:C"])[0], _DP, rtol=1e-7)
        got = ss.output_sensitivities(["species:T"])[0]
        np.testing.assert_allclose(got, 2 / 36 * _DP, rtol=1e-7)
        np.testing.assert_allclose(got, _central_row(text, lambda s: s["T"]), rtol=1e-6)

    def test_a_size_by_a_declined_rule_is_exact(self):
        """``C := piecewise(1 + P, ...)`` has no compiled derivative; dual
        numbers give it (issue #1330), and T = 2P in C = 1 + P is the
        concentration 2P/(1 + P), whose row is 2/(1 + P)²·dP*/dθ."""
        text = _MADE_AND_LOST + (
            "C := piecewise(1 + P, P < 100, 2 + P);\nsubstanceOnly species T in C; T := 2*P;\n"
        )
        ss = _ant_steady_state(text)
        np.testing.assert_allclose(
            ss.output_sensitivities(["species:T"])[0], 2 / 36 * _DP, rtol=1e-7
        )

    def test_a_size_by_a_declined_rule_is_refused_by_name(self, monkeypatch):
        """With no dual evaluation it was differenced, and T's row with it; the
        solve is refused now, naming the size and the construct (issue
        #1002)."""
        monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")
        text = _MADE_AND_LOST + (
            "C := piecewise(1 + P, P < 100, 2 + P);\nsubstanceOnly species T in C; T := 2*P;\n"
        )
        with pytest.raises(
            bngsim.SimulationError,
            match=r"\['T'\].*for T, its size C has no steady-state derivative \(the compiled "
            r"code has none for its rule: uses unsupported construct: if\(\) conditional; "
            r"issue #1002\).*written without that construct",
        ):
            _ant_steady_state(text)

    def test_a_size_that_reads_a_masked_species_is_refused_by_name(self):
        """The size's rule reads ``Z``, which ``mask=`` leaves out, so the size
        has no row and neither has ``T``. The refusal names ``Z`` as the
        cause; main said only that ``mask=`` left the size, or a species its
        rule reads, out of the solve."""
        text = _MADE_AND_LOST + (
            "species Z in D; Z = 0; kz = 0.1; R5: P => Z; kz*P;\n"
            "C := 1 + Z;\nsubstanceOnly species T in C; T := 2*P;\n"
        )
        with pytest.raises(
            bngsim.SimulationError,
            match=r"\['T'\].*for T, its size C reads Z, which mask= left out of the solve\. "
            r"Keep \['Z'\] in the mask",
        ):
            _ant_steady_state(text, mask_sinks=True)


def test_an_observable_of_a_masked_species_is_refused_by_name():
    """Issue #1124, item 10. ``A`` is made at k1 and lost to ``B`` at k2·A, so
    A* = k1/k2 = 2 and dA*/d(k1, k2) = (1/k2, -k1/k2²) = (2, -4); ``B`` only
    collects, and ``mask=`` leaves it out. An observable of B was a NaN row."""
    m = bngsim.Model.from_antimony_string(
        "species A = 1; species B = 0; k1 = 1; k2 = 0.5;\nJ0: -> A; k1; J1: A -> B; k2*A;"
    )
    ss = bngsim.Simulator(m, method="ode").steady_state(
        sensitivity_params=["k1", "k2"], mask=~np.asarray(m.is_pure_sink()), **_TIGHT
    )
    np.testing.assert_allclose(ss.output_sensitivities("observable:A")[0], [2, -4], rtol=1e-7)
    with pytest.raises(ValueError, match=r"'observable:B'.*reads B, which mask= left out"):
        ss.output_sensitivities(["observable:A", "observable:B"])


def test_an_expression_names_the_one_masked_species_it_reads():
    """Issue #1124, item 10. Two pure sinks, ``B`` and ``C``, are masked; ``h``
    reads ``B`` alone. The expression block records the first species with no
    row that each expression reads, so the refusal names ``B``, not "one or
    more of B, C"."""
    m = bngsim.Model.from_antimony_string(
        "species A = 1; species B = 0; species C = 0; k1 = 1; k2 = 0.5;\n"
        "J0: -> A; k1; J1: A -> B; k2*A; J2: A -> C; 0.1*A; h := B + 1;"
    )
    ss = bngsim.Simulator(m, method="ode").steady_state(
        sensitivity_params=["k1", "k2"], mask=~np.asarray(m.is_pure_sink()), **_TIGHT
    )
    with pytest.raises(ValueError, match=r"'expression:h'.*reads B, which mask= left out"):
        ss.output_sensitivities("expression:h")


def test_a_rule_target_beside_a_mask_is_not_named_as_masked():
    """Issue #1124, item 10. ``T := 2*A`` is a species set by a rule, which the
    solve leaves out by itself (GH #247) and gives its rule's row, (4, -8). Only
    ``B`` is left out by ``mask=``, so B's refusal names B alone; it named "one
    or more of B, T"."""
    m = bngsim.Model.from_antimony_string(
        "species A = 1; species B = 0; species T = 0; k1 = 1; k2 = 0.5;\n"
        "J0: -> A; k1; J1: A -> B; k2*A;\nT := 2*A;"
    )
    ss = bngsim.Simulator(m, method="ode").steady_state(
        sensitivity_params=["k1", "k2"], mask=~np.asarray(m.is_pure_sink()), **_TIGHT
    )
    np.testing.assert_allclose(ss.output_sensitivities("observable:T")[0], [4, -8], rtol=1e-7)
    with pytest.raises(ValueError, match=r"'observable:B'.*: it reads B, which mask= left out"):
        ss.output_sensitivities("observable:B")


@pytest.mark.parametrize("sink", ["", "B masked"], ids=["no mask", "B masked"])
def test_an_output_with_no_finite_value_is_refused_without_blaming_the_mask(sink):
    """``logL := log10(L)`` with L held at 0 (it only decays, from 0) is -inf at
    the steady state, so it has no derivative there, whatever the mask. Every
    partial of -inf is NaN, so with ``B`` masked the solve's record named B, and
    the refusal said logL reads B. It is refused for its value."""
    text = "species A = 1; species L = 0; k1 = 1; k2 = 0.5; kL = 1;\n"
    if sink:
        text += "species B = 0; J0: -> A; k1; J1: A -> B; k2*A;"
    else:
        text += "J0: -> A; k1; J1: A -> ; k2*A;"
    m = bngsim.Model.from_antimony_string(text + " JL: L -> ; kL*L;\nlogL := log10(L);")
    kw = {"mask": ~np.asarray(m.is_pure_sink())} if sink else {}
    ss = bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=["k1", "k2"], **kw)
    np.testing.assert_allclose(ss.output_sensitivities("observable:A")[0], [2, -4], rtol=1e-6)
    with pytest.raises(ValueError, match=r"'expression:logL'.*value at the steady state is -inf"):
        ss.output_sensitivities("expression:logL")


def test_every_selector_without_a_derivative_is_named_at_once():
    """Issue #1124, item 10. A selection with two outputs that read a masked
    species was refused for the first alone, so the caller met the second only
    after fixing the first. Both are named in one refusal, and an output with a
    row is not."""
    m = bngsim.Model.from_antimony_string(
        "species A = 1; species B = 0; k1 = 1; k2 = 0.5;\nJ0: -> A; k1; J1: A -> B; k2*A;"
    )
    ss = bngsim.Simulator(m, method="ode").steady_state(
        sensitivity_params=["k1", "k2"], mask=~np.asarray(m.is_pure_sink()), **_TIGHT
    )
    with pytest.raises(ValueError) as refusal:
        ss.output_sensitivities(["observable:A", "observable:B", "species:B"])
    message = str(refusal.value)
    assert "'observable:B'" in message and "'species:B'" in message
    assert "'observable:A'" not in message


@pytest.mark.parametrize("body", ["log10(L)", "sqrt(L - 1)"], ids=["-inf", "nan"])
def test_the_solve_records_no_missing_species_for_a_value_that_is_not_finite(body):
    """The FD sweep took any nonzero partial for a read, and every partial of
    -inf is NaN, so ``logL := log10(L)`` at L = 0 recorded the masked ``B`` as
    the species with no row it read. It reads none, and the record says so.
    A value that is NaN (``sqrt(L - 1)``) moves with every step, since NaN
    equals nothing, and is not taken for a read either: a control, since
    main's sweep had the same guard, which the step that only asks whether a
    value moves keeps (issue #1002)."""
    m = bngsim.Model.from_antimony_string(
        "species A = 1; species B = 0; species L = 0; k1 = 1; k2 = 0.5; kL = 1;\n"
        f"J0: -> A; k1; J1: A -> B; k2*A; JL: L -> ; kL*L;\nlogL := {body};"
    )
    ss = bngsim.Simulator(m, method="ode").steady_state(
        sensitivity_params=["k1", "k2"], mask=~np.asarray(m.is_pure_sink())
    )
    assert ss._expression_sens_missing[ss.expression_names.index("logL")] is None
    with pytest.raises(ValueError, match=r"'expression:logL'.*value at the steady state is"):
        ss.output_sensitivities("expression:logL")
