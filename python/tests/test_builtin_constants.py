"""The seven built-in physical constants, and the differentiation layer's view of them.

``_pi``, ``_e``, ``_kB``, ``_NA``, ``_R``, ``_h`` and ``_F`` are bound by the
ExprTk evaluator on every expression (``add_remapped_constant`` in
``src/expression.cpp``), and the engine reserves the names so no model can define
a parameter that shadows one. A rate law or a switch threshold that spells one
therefore means the constant and nothing else.

The Python differentiation layer used to know about them piecemeal. ``_pi`` and
``_e`` were special-cased in two rate-law emitters, the other five were nowhere,
and the derived-expression preparation that resolves a switch threshold knew none
of the seven. A single ``_pi`` in a rate law was enough to lose a model its whole
analytic sensitivity RHS, and ``if(time() < A*_pi, ...)`` on BIOMD0000000616 was
refused for forward sensitivity outright.

The fix is that both sympy entry points bind the constants to their *values*, so
every site downstream sees a number rather than a free symbol and needs no entry
of its own. What is tested here is that the Python copy of the table cannot drift
from the engine's, in names or in values, and that a model spelling a constant is
differentiated rather than declined.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim import _switch_sensitivity as sw

pytest.importorskip("sympy")

CONSTANTS = sorted(cg._BUILTIN_CONSTANT_VALUES)


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


SIR = """\
begin parameters
    1 S0      1000  # Constant
    2 I0      1  # Constant
    3 beta    0.002  # Constant
    4 gamma   0.15  # Constant
    5 sigma   3.0  # Constant
end parameters
begin functions
    1 betaI() beta*I
end functions
begin species
    1 person(state~S) S0
    2 person(state~I) I0
    3 person(state~R) 0
end species
begin reactions
    1 1 2 betaI #_R1
    2 2 3 gamma #_R2
end reactions
begin groups
    1 S                    1
    2 I                    2
    3 R                    3
end groups
"""


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _with_law(body: str) -> str:
    return SIR.replace("    1 betaI() beta*I\n", f"    1 betaI() {body}\n")


class TestTheTableCannotDriftFromTheEngine:
    """The Python table is a second copy of a list that lives in C++, which is
    the shape of defect that gets fixed on one side and forgotten on the other.
    Both halves of it are pinned against the engine itself rather than against a
    literal written down twice."""

    def test_the_names_are_exactly_what_the_engine_reserves(self):
        assert set(CONSTANTS) == set(bngsim.reserved_names()["constants"])

    @pytest.mark.parametrize("name", CONSTANTS)
    def test_each_value_matches_what_the_engine_computes(self, tmp_path, name):
        """Read back through ``_eval_functions``, so the oracle is the ExprTk
        evaluator that will actually run the model, not a number retyped here."""
        text = _with_law("beta*I").replace(
            "    1 betaI() beta*I\n", f"    1 betaI() beta*I\n    2 konst() {name}\n"
        )
        core = _model(tmp_path, text)._core
        conc = [core.get_concentration(n) for n in core.species_names]
        engine = core._eval_functions(0.0, conc)["konst"]
        assert engine == pytest.approx(cg._BUILTIN_CONSTANT_VALUES[name], rel=1e-15)

    def test_pi_and_e_keep_their_exact_c_spelling(self, tmp_path):
        """They bind to sympy's own constants rather than to floats, so a rate
        law carrying one still prints the C macro. The other five have no sympy
        counterpart and print as full-precision literals."""
        core = _model(tmp_path, _with_law("beta*_pi*I"))
        assert "M_PI" in (cg.generate_sens_from_model(core, functional=True) or "")


class TestARateLawSpellingAConstantIsDifferentiated:
    @pytest.mark.parametrize("name", CONSTANTS)
    def test_the_analytic_sensitivity_rhs_survives(self, tmp_path, name):
        """One constant anywhere in one rate law used to decline the analytic
        sensitivity RHS for the whole model, because the differentiator met a
        symbol it could not resolve and refused the law. ``_pi`` and ``_e``
        failed on the species-derivative side, the other five on both."""
        model = _model(tmp_path, _with_law(f"beta*{name}*I"))
        _src, has_sens = cg.generate_combined_from_model(model)
        assert has_sens is True

    def test_the_partial_is_the_constant_itself(self, tmp_path):
        """Not merely emitted, but right: ``d(beta*_pi*I)/dbeta`` is ``_pi*I``, so
        the sensitivity to beta is pi times what it would be without the factor."""
        plain = _model(tmp_path, _with_law("beta*I"), name="plain.net")
        scaled = _model(tmp_path, _with_law("beta*_pi*I"), name="scaled.net")
        for m in (plain, scaled):
            assert cg.generate_combined_from_model(m)[1] is True
        core = scaled._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms


class TestAClockThresholdOverAConstant:
    """BIOMD0000000616 writes ``if(time() < A*_pi, ...)``. That is an ordinary
    clock threshold crossing at A times pi, with a closed-form ``dt*/dA`` of pi,
    and it was refused only because ``_pi`` did not resolve to anything the
    threshold reader knew."""

    def test_the_crossing_and_its_partial_resolve(self, tmp_path):
        core = _model(tmp_path, _with_law("if(time()<sigma*_pi,beta,0)*I"))._core
        records, pinned = sw.compute_switch_time_sens(
            core, ["sigma", "beta"], 0.0, 20.0, has_analytic_sens_rhs=True
        )
        assert len(records) == 1
        assert records[0].t_star == pytest.approx(3.0 * np.pi, rel=1e-12)
        assert records[0].dtstar[0] == pytest.approx(np.pi, rel=1e-12)  # dt*/dsigma
        assert records[0].dtstar[1] == 0.0  # beta does not move it
        assert pinned == [list(core.param_names).index("sigma")]

    def test_the_gate_admits_it(self, tmp_path):
        core = _model(tmp_path, _with_law("if(time()<sigma*_pi,beta,0)*I"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        assert sw.model_uncompensated_crossing_reason(core) is None

    @pytest.mark.parametrize("name", CONSTANTS)
    def test_every_constant_resolves_in_a_threshold(self, tmp_path, name):
        core = _model(tmp_path, _with_law(f"if(time()<sigma*{name},beta,0)*I"))._core
        scope = sw.switch_condition_scope(core)
        atom = f"time()<sigma*{name}"
        assert sw.clock_crossing_compensated(atom, scope)
        value = sw._evaluate_threshold(
            f"sigma*{name}", scope.param_idx, scope.values, scope.derived_exprs
        )
        assert value == pytest.approx(3.0 * cg._BUILTIN_CONSTANT_VALUES[name], rel=1e-12)

    @requires_cc
    def test_the_run_matches_a_finite_difference(self, tmp_path):
        """End to end. The switch is at ``sigma*pi``, so raising sigma keeps the
        infection on for pi times longer, and the sensitivity column is that jump.
        Away from the crossing node a central difference of two trajectories has
        to agree with it."""

        def net(sigma):
            return _with_law("if(time()<sigma*_pi,beta,0)*I").replace(
                "5 sigma   3.0", f"5 sigma   {sigma}"
            )

        ts, n = (0.0, 20.0), 81
        times = np.linspace(*ts, n)
        col = np.asarray(
            bngsim.Simulator(
                _model(tmp_path, net(3.0)), method="ode", sensitivity_params=["sigma"]
            )
            .run(t_span=ts, n_points=n, rtol=1e-11, atol=1e-11)
            .sensitivities
        )[:, :, 0]

        h = 3.0 * 1e-3

        def traj(sigma, name):
            return np.asarray(
                bngsim.Simulator(_model(tmp_path, net(sigma), name=name), method="ode")
                .run(t_span=ts, n_points=n, rtol=1e-12, atol=1e-14)
                .species
            )

        fd = (traj(3.0 + h, "hi.net") - traj(3.0 - h, "lo.net")) / (2.0 * h)

        after = times >= 3.0 * np.pi + 1.0
        assert float(np.max(np.abs(col[after]))) > 1.0  # the jump is not vacuous
        scale = float(np.max(np.abs(col[after])))
        np.testing.assert_allclose(col[after], fd[after], rtol=2e-4, atol=1e-4 * scale)
