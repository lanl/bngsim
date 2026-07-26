"""Regression tests for issue #28: Observable referenced as a zero-arg
call (`obs()`) in BNGL must compile.

BNGL's grammar (bionetgen/bng2/Perl2/Expression.pm:870-927) accepts an
Observable as a zero-arg call (`obs()`) anywhere a bareword `obs` is
valid. BNG2.pl preserves whichever form the user wrote when emitting
the .net file. ExprTk's grammar would parse `obs()` as `obs * ()` and
reject the empty parens with ERR248, so bngsim's
``ExprTkEvaluator::compile()`` strips `name()` → `name` for any name
registered as a scalar variable (parameters, observables, species,
synthetic function-result parameters, built-in constants).

The 1-arg form `obs(s)` (LocalFunction in BNGL) is resolved by BNG2.pl
during ``generate_network`` into per-instance constant parameters and
never reaches the .net file, so bngsim has no responsibility for it.

The same `()` has to be dropped on the two paths that consume expression
*strings* rather than the ExprTk evaluator, or a model the interpreter runs
fine is broken by turning codegen on:

* the codegen emitters (``bngsim._codegen``) rewrite an observable to a C
  scalar — ``obs[0]`` / ``obs_Atot`` — so a surviving ``()`` emits
  ``obs[0]()``, which no C compiler accepts;
* the sympy-facing differentiator (``bngsim._jacobian``) reads ``Atot()`` as
  an *applied undefined function* sharing no symbol with the bareword, so
  ``d/d Atot`` of ``100*Atot()`` silently collapses to zero.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Simulator
from bngsim._codegen import (
    _build_ident_lookup,
    _build_ident_lookup_model,
    _translate_expr,
    _translate_expr_to_c,
    generate_combined_c,
)
from bngsim._jacobian import (
    differentiate_expression_output_partials,
    differentiate_rate_law,
)

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

# `obs[0]()`, `obs_Atot()`, `p[1]()`, `y[0]()`, `func[2]()`, `t()` — a scalar C
# reference with an empty argument list, i.e. the pre-fix emitter output. Every
# call the emitters legitimately produce carries arguments, so this never
# matches valid generated C.
_CALL_ON_SCALAR_RE = re.compile(r"(?:(?:obs|func|p|y)(?:\[[^\]]*\]|_\w+)|\bt)\s*\(\s*\)")


class TestObsZeroArgCall:
    """Models that reference observables as `obs()` must load and run."""

    def test_loads_without_crash(self, obs_zero_arg_call_net: Path):
        """The .net loads — primary regression for issue #28."""
        model = Model.from_net(obs_zero_arg_call_net)
        assert model.n_species == 2
        assert "Atot" in model.observable_names
        assert "Btot" in model.observable_names

    def test_zero_arg_call_resolves_to_observable_value(self, obs_zero_arg_call_net: Path):
        """`Atot()` in the rate law evaluates to the observable's total —
        identical semantics to the bareword `Atot`. The model is
        first-order decay A → B with rate k1*Atot (== k1*A since A is
        the only molecule contributing to Atot at t=0)."""
        model = Model.from_net(obs_zero_arg_call_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0.0, 5.0), n_points=11)

        # k1=0.5; analytical: A(t) = A0*exp(-k1*t) when reverse rate is
        # negligible, but here _rateLaw2 = k1*Btot drives B back to A.
        # At equilibrium A == B == 5 (mass conservation, equal forward
        # and reverse rates). Test the conservation invariant exactly.
        a_end = result.species[-1, 0]
        b_end = result.species[-1, 1]
        assert a_end + b_end == pytest.approx(10.0, rel=1e-6)
        # By t=5 with k1=0.5, the system is well past the time constant
        # 1/(2*k1)=1, so A and B should be near 5.
        assert a_end == pytest.approx(5.0, abs=0.5)
        assert b_end == pytest.approx(5.0, abs=0.5)

    def test_mixed_bareword_and_zero_arg_call(self, obs_zero_arg_call_net: Path):
        """The fixture uses `Atot()` in _rateLaw1 and bareword `Btot`
        in _rateLaw2. Both must resolve to the same observable totals
        for the equilibrium to hold (asserted in the prior test)."""
        # Already covered by the equilibrium assertion above; this test
        # exists to document intent — both forms must coexist.
        model = Model.from_net(obs_zero_arg_call_net)
        sim = Simulator(model, method="ode")
        # Long-time behavior should be stable equilibrium, not divergent
        result = sim.run(t_span=(0.0, 50.0), n_points=2)
        assert math.isfinite(result.species[-1, 0])
        assert math.isfinite(result.species[-1, 1])


class TestIdentLookupEatsEmptyParens:
    """Unit: every *model* name in either codegen identifier table eats a
    trailing `()`, because every one of them denotes a C scalar. Only the
    built-ins that map to real C functions/operators keep their parens."""

    def test_net_path_named_locals(self):
        lookup = _build_ident_lookup(
            {"k1": 0},
            {"Atot": 0, "divide": 1},
            [(1, "_rateLaw1", "100*divide()")],
            use_arrays=False,
        )
        assert _translate_expr("100*divide()", lookup) == "100.0*obs_divide"
        assert _translate_expr("k1()*Atot()", lookup) == "p[0]*obs_Atot"
        assert _translate_expr("_rateLaw1()", lookup) == "func__rateLaw1"

    def test_net_path_arrays(self):
        """The sharded (GH #165) form reads the arrays, and must strip too."""
        lookup = _build_ident_lookup(
            {"k1": 0},
            {"Atot": 0, "divide": 1},
            [(1, "_rateLaw1", "100*divide()")],
            use_arrays=True,
        )
        assert _translate_expr("(1-divide())*1", lookup) == "(1.0-obs[1])*1.0"
        assert _translate_expr("_rateLaw1()", lookup) == "func[0]"

    def test_model_path(self):
        lookup = _build_ident_lookup_model(
            {"k1": "p[0]"},
            {"A()": "y[0]"},
            {"Atot": "obs[0]"},
            {"plain": "func[0]"},
            {"rate_of__A": "current_derivs[0]"},
        )
        assert _translate_expr_to_c("k1()*Atot()", lookup) == "p[0]*obs[0]"
        assert _translate_expr_to_c("plain()", lookup) == "func[0]"
        assert _translate_expr_to_c("rate_of__A()", lookup) == "current_derivs[0]"
        # ``_pi``/``_e`` are remapped *constants* on the ExprTk evaluator, so
        # strip_empty_parens() strips them there as well.
        assert _translate_expr_to_c("_pi()", lookup) == "M_PI"

    def test_c_function_builtins_keep_their_parens(self):
        """The strip must not touch a name that really is a call: `abs`, `ln`,
        `rint`, `max`/`min` take arguments, and an unknown identifier (math.h)
        passes through whole."""
        lookup = _build_ident_lookup_model({"k1": "p[0]"}, {}, {"Atot": "obs[0]"}, {})
        assert _translate_expr_to_c("abs(Atot)", lookup) == "fabs(obs[0])"
        assert _translate_expr_to_c("ln(Atot)", lookup) == "log(obs[0])"
        assert _translate_expr_to_c("max(Atot,k1)", lookup) == "fmax(obs[0],p[0])"
        assert _translate_expr_to_c("tanh(Atot)", lookup) == "tanh(obs[0])"


class TestZeroArgCallSymbolicDifferentiation:
    """The sympy-facing differentiator must read `obs()` as the observable, not
    as an applied undefined function (which differentiates to nothing)."""

    def test_rate_law_partial_matches_bareword(self):
        """`100*divide()` and `100*divide` are the same rate law, so they must
        produce the same ∂rate/∂obs — pre-fix the call form gave `{}`, i.e. a
        silently wrong (missing) analytical Jacobian entry."""
        func_map = {"_rateLaw3": "100*divide()"}
        obs, const = {"divide"}, {"Vmax"}
        assert differentiate_rate_law("_rateLaw3", func_map, obs, const) == differentiate_rate_law(
            "100*divide", {}, obs, const
        )

    def test_output_partials_match_bareword(self):
        """Same for the GH #198 `d func/dθ` partials, in both the
        silently-zero shape (constant coefficient) and the shape that used to
        decline outright (parameter coefficient)."""
        crefs = dict(
            species_cref={},
            observable_cref={"Atot": "obs[0]"},
            param_cref={"scale": "p[1]"},
            function_cref={},
        )
        for call_form, bareword in (
            ("100*Atot()", "100*Atot"),
            ("scale*Atot()", "scale*Atot"),
        ):
            got, reason = differentiate_expression_output_partials(call_form, **crefs)
            assert reason is None, f"{call_form}: {reason}"
            assert got == differentiate_expression_output_partials(bareword, **crefs)[0]
            assert got["observable"]  # the ∂/∂Atot term is present, not dropped

    def test_time_call_is_still_the_time_placeholder(self):
        """`time()` is the one zero-arg call that is *not* a scalar reference;
        stripping must run after its rewrite, leaving d/dk1 of `k1*time()` = t."""
        got, reason = differentiate_expression_output_partials(
            "k1*time()",
            species_cref={},
            observable_cref={},
            param_cref={"k1": "p[0]"},
            function_cref={},
        )
        assert reason is None
        assert got["param"] == {"k1": "t"}


@needs_cc
class TestObsZeroArgCallCodegen:
    """Turning codegen on must not break a model the interpreter runs fine.

    The emitters substitute a C *scalar* for the observable, so the `()` has to
    go with it; leaving it in emitted `obs[0]()` and the compile failed with
    "called object type 'double' is not a function or function pointer" — which
    made a forward-sensitivity Simulator refuse the model outright, since
    sensitivity requires the compiled RHS.
    """

    def test_no_zero_arg_call_survives_into_the_generated_c(
        self, obs_zero_arg_call_sens_net: Path
    ):
        model = Model.from_net(obs_zero_arg_call_sens_net)
        src, has_sens = generate_combined_c(
            str(obs_zero_arg_call_sens_net),
            model=model,
            emit_jac=True,
            emit_outputs=True,
            emit_output_sens=True,
        )
        assert has_sens, "elementary model: the analytical sens RHS should be emitted"
        hit = _CALL_ON_SCALAR_RE.search(src)
        assert hit is None, f"emitted a call on a C scalar: {hit.group(0) if hit else ''}"
        # Both reference forms are exercised: the .net RHS emits named locals,
        # the model-based emitters (outputs / Jacobian / output-sens) arrays.
        assert "100.0*obs_Atot" in src
        assert "100.0*obs[0]" in src

    def test_codegen_trajectory_matches_the_interpreter(self, obs_zero_arg_call_net: Path):
        """The compiled RHS must reproduce the ExprTk RHS to solver tolerance —
        `k1*Atot()` and `k1*Atot` are the same rate law."""
        interp = Simulator(Model.from_net(obs_zero_arg_call_net), method="ode", codegen=False)
        compiled = Simulator(Model.from_net(obs_zero_arg_call_net), method="ode", codegen=True)
        run = dict(t_span=(0.0, 5.0), n_points=11, rtol=1e-10, atol=1e-12)
        assert compiled.codegen_backend != "none"
        np.testing.assert_allclose(
            compiled.run(**run).species, interp.run(**run).species, rtol=1e-9, atol=1e-11
        )

    def test_sensitivity_simulator_constructs(self, obs_zero_arg_call_net: Path):
        """The reported symptom: constructing a forward-sensitivity Simulator
        raised RuntimeError("… Codegen compilation failed") because the RHS
        could not be compiled."""
        model = Model.from_net(obs_zero_arg_call_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0.0, 2.0), n_points=5)
        assert np.all(np.isfinite(result.sensitivities))
        # A ⇄ B conserves mass, so dA/dk1 + dB/dk1 == 0 at every output row.
        np.testing.assert_allclose(
            result.sensitivities[:, 0, 0] + result.sensitivities[:, 1, 0], 0.0, atol=1e-6
        )

    def test_expression_output_sensitivities_are_not_silently_zero(
        self, obs_zero_arg_call_sens_net: Path, monkeypatch
    ):
        """`d func/dθ` over bodies written with the call form, against the
        fixture's closed forms. Pre-fix `100*Atot()` differentiated to nothing,
        so `d plain/dk1` came back as exactly 0.0 with no error raised."""
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
        model = Model.from_net(obs_zero_arg_call_sens_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1", "scale"])
        r = sim.run(t_span=(0.0, 4.0), n_points=5, rtol=1e-11, atol=1e-13)

        t = np.asarray(r.time)
        es = r.sensitivities_expressions
        k1, scale = (r.sensitivity_params.index(n) for n in ("k1", "scale"))
        plain, scaled, mixed = (r.expression_names.index(n) for n in ("plain", "scaled", "mixed"))

        # A(t) = 10·e^{-k1 t} ⇒ dA/dk1 = -10·t·e^{-k1 t}
        da_dk1 = -10.0 * t * np.exp(-0.5 * t)
        atot = np.asarray(r.observables["Atot"])
        np.testing.assert_allclose(es[:, plain, k1], 100.0 * da_dk1, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(es[:, scaled, k1], 3.0 * da_dk1, rtol=1e-6, atol=1e-8)
        # ∂/∂scale reaches `scaled` only through the parameter term.
        np.testing.assert_allclose(es[:, scaled, scale], atot, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(es[:, plain, scale], 0.0, atol=1e-10)
        # mixed = Atot + Btot is conserved, so every derivative vanishes.
        np.testing.assert_allclose(es[:, mixed, :], 0.0, atol=1e-8)
