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
use a finite-difference total derivative (state chain + explicit ∂func/∂p) whose
values are additionally pinned against closed-form derivatives.

Two fixtures span both steady-state sensitivity solves:
- ``ss_output_sens.net`` — closed A↔B↔C, conserved ⇒ reduced-space solve.
- ``ss_birthdeath.net``  — open birth-death, no conservation ⇒ full-space solve.
"""

from __future__ import annotations

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

CLOSED_NET = str(DATA_DIR / "ss_output_sens.net")  # A<->B<->C, conserved
OPEN_NET = str(DATA_DIR / "ss_birthdeath.net")  # dS/dt = k_prod - k_deg*S

# High-accuracy CVODES so the last time point is a faithful steady-state oracle.
_RUN = dict(rtol=1e-11, atol=1e-13, max_steps=10**6)


@pytest.fixture(autouse=True)
def _force_codegen(monkeypatch):
    """Expression (global-function) output sensitivities on the CVODE ``run()``
    oracle require the compiled ``.so``; force codegen on. The steady-state path
    itself is finite-difference and needs no codegen. monkeypatch restores the
    environment afterwards."""
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
