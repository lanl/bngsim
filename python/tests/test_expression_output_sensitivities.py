"""Expression / global-function output sensitivities via the codegen evaluator
(GH #198).

Global functions are nonlinear in their inputs, so ``d func/dθ`` needs the full
chain rule over the same expression graph the *values* use:

    df/dθ = Σ_i ∂f/∂x_i·dx_i/dθ + Σ_j ∂f/∂obs_j·dobs_j/dθ
          + Σ_k ∂f/∂p_k·dp_k/dθ + Σ_m ∂f/∂f_m·df_m/dθ

emitted as compiled C (``bngsim_codegen_output_sens``) in the same ``.so`` as the
RHS/sens-RHS. These tests validate every chain-rule dependency kind against
central finite differences (high solver tolerances, scale-relative tolerance),
the initial-condition axis, the derived-parameter chain, and that every
unsupported construct fails loudly with a targeted error.
"""

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._jacobian import differentiate_expression_output_partials

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# A --k1--> B --k2--> C (elementary mass action ⇒ codegen sens RHS + IC axis).
# Observables: A_obs = A, BC2 = B + 2*C. Global functions:
#   scaled() = scale*A_obs              parameter + observable
#   ratio()  = A_obs/(BC2+eps)          two observables + parameter
#   combo()  = scaled() + ratio()       earlier functions
#   tdep()   = k1*A_obs*time()          time (drops) + parameter + observable
CHAIN_NET = str(DATA_DIR / "expr_sens_chain.net")
# kd = 2*kbase (derived); dfn() = kd*A_obs exercises the derived-param chain.
DERIVED_NET = str(DATA_DIR / "expr_sens_derived.net")
# good() supported; if/cmp/abs/min/max/floor/ceil/rint unsupported; trans() transitive.
UNSUPPORTED_NET = str(DATA_DIR / "expr_sens_unsupported.net")
# response() = tfun('…', drug_conc): a table-function-backed function (unsupported).
TFUN_NET = str(DATA_DIR / "tfun_paren_param.net")

_RUN = dict(rtol=1e-11, atol=1e-13, max_steps=10**6)
_FD = dict(rtol=1e-12, atol=1e-14, max_steps=10**6)
T_SPAN = (0.0, 8.0)
N_POINTS = 9

CHAIN_PARAMS = ("k1", "k2", "scale", "eps")
CHAIN_BASE = {"k1": 0.30, "k2": 0.15, "scale": 2.0, "eps": 0.50}


@pytest.fixture(autouse=True)
def _force_codegen(monkeypatch):
    """Expression output sensitivities require the compiled ``.so``; force codegen
    on for every test here. monkeypatch restores the environment afterwards so the
    threshold never leaks into threshold-sensitive tests elsewhere."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)


def _run_chain(overrides=None, *, params=CHAIN_PARAMS, ic=None, run_kw=_RUN):
    m = bngsim.Model.from_net(CHAIN_NET)
    for k, v in (overrides or {}).items():
        m.set_param(k, v)
    sim = bngsim.Simulator(
        m,
        method="ode",
        sensitivity_params=list(params),
        sensitivity_ic=list(ic) if ic else None,
    )
    m.reset()
    return sim.run(t_span=T_SPAN, n_points=N_POINTS, **run_kw)


def _assert_fd_close(analytic, fd, value_scale, *, rtol=2e-4, atol=1e-7):
    """Scale-relative comparison: the error must be small relative to the
    *function* magnitude, so a genuinely-zero derivative (whose FD is pure solver
    noise) is not flagged by a divide-by-near-zero relative error."""
    tol = atol + rtol * max(value_scale, np.max(np.abs(fd)))
    err = np.max(np.abs(np.asarray(analytic) - np.asarray(fd)))
    assert err <= tol, f"max abs err {err:.3e} > tol {tol:.3e}"


class TestParameterAxisFD:
    """Every (expression, parameter) sensitivity matches central FD."""

    @pytest.mark.parametrize("ename", ["scaled", "ratio", "combo", "tdep"])
    @pytest.mark.parametrize("pname", CHAIN_PARAMS)
    def test_fd(self, ename, pname):
        r = _run_chain()
        es = r.sensitivities_expressions
        ei = r.expression_names.index(ename)
        pi = r.sensitivity_params.index(pname)
        analytic = es[:, ei, pi]

        h = CHAIN_BASE[pname] * 1e-6
        rp = _run_chain({pname: CHAIN_BASE[pname] + h}, run_kw=_FD)
        rm = _run_chain({pname: CHAIN_BASE[pname] - h}, run_kw=_FD)
        fd = (np.asarray(rp.expressions[ename]) - np.asarray(rm.expressions[ename])) / (2 * h)

        scale = float(np.max(np.abs(r.expressions[ename])))
        _assert_fd_close(analytic, fd, scale)

    def test_time_term_drops(self):
        """tdep() = k1*A_obs*time(): the explicit ``time()`` factor has zero
        derivative w.r.t. every parameter, so d tdep/d k1 = (A_obs + k1·dA_obs/dk1)·t."""
        r = _run_chain()
        es = r.sensitivities_expressions
        ti = r.expression_names.index("tdep")
        k1 = r.sensitivity_params.index("k1")
        t = np.asarray(r.time)
        a = np.asarray(r.observables["A_obs"])
        da_dk1 = r.output_sensitivities("observable:A_obs")[:, 0, 0]
        expected = (a + 0.30 * da_dk1) * t
        assert np.allclose(es[:, ti, k1], expected, rtol=1e-6, atol=1e-9)


class TestObservableChainConsistency:
    """The expression chain reuses the SAME observable derivatives as GH #197."""

    def test_scaled_equals_scale_times_obs_sens(self):
        # scaled = scale*A_obs ⇒ d scaled/d k1 = scale·dA_obs/dk1 (scale ⟂ k1),
        # tying the expression block to the #197 observable block.
        r = _run_chain()
        si = r.expression_names.index("scaled")
        k1 = r.sensitivity_params.index("k1")
        d_scaled = r.sensitivities_expressions[:, si, k1]
        d_aobs = r.output_sensitivities("observable:A_obs")[:, 0, 0]
        assert np.allclose(d_scaled, 2.0 * d_aobs, rtol=1e-10, atol=1e-12)

    def test_combo_is_sum_of_components(self):
        # combo = scaled + ratio ⇒ d combo/dθ = d scaled/dθ + d ratio/dθ for all θ.
        r = _run_chain()
        es = r.sensitivities_expressions
        ci, si, ri = (r.expression_names.index(n) for n in ("combo", "scaled", "ratio"))
        assert np.allclose(es[:, ci, :], es[:, si, :] + es[:, ri, :], rtol=1e-10, atol=1e-12)


class TestICAxis:
    """Initial-condition axis: d func/dY(0)."""

    def test_t0_seed_and_closed_form(self):
        # scaled = scale*A ; A(0)=100, A(t)=A0·e^{-k1 t} in isolation from B,C.
        # d scaled/dA(0) = scale·dA/dA(0) = scale·e^{-k1 t}; at t=0 it is exactly scale.
        r = _run_chain(ic=["A()", "B()", "C()"])
        es = r.sensitivities_expressions_ic
        si = r.expression_names.index("scaled")
        ai = r.sensitivity_ic_species.index("A()")
        assert es[0, si, ai] == pytest.approx(2.0, rel=1e-9)
        closed = 2.0 * np.exp(-0.30 * np.asarray(r.time))
        assert np.allclose(es[:, si, ai], closed, rtol=1e-5, atol=1e-7)

    def test_ic_fd(self):
        # FD reference: perturb A(0) AFTER reset (reset reverts set_concentration).
        def run_a(a0):
            m = bngsim.Model.from_net(CHAIN_NET)
            sim = bngsim.Simulator(
                m, method="ode", sensitivity_params=["k1"], sensitivity_ic=["A()", "B()", "C()"]
            )
            m.reset()
            m.set_concentration("A()", a0)
            return sim.run(t_span=T_SPAN, n_points=N_POINTS, **_FD)

        r = run_a(100.0)
        es = r.sensitivities_expressions_ic
        h = 1e-2
        rp, rm = run_a(100.0 + h), run_a(100.0 - h)
        for ename in ("scaled", "ratio", "combo"):
            ei = r.expression_names.index(ename)
            ai = r.sensitivity_ic_species.index("A()")
            fd = (np.asarray(rp.expressions[ename]) - np.asarray(rm.expressions[ename])) / (2 * h)
            scale = float(np.max(np.abs(r.expressions[ename])))
            _assert_fd_close(es[:, ei, ai], fd, scale)


class TestDerivedParameter:
    """A function referencing a derived (ConstantExpression) parameter."""

    def test_derived_chain_fd(self):
        # dfn = kd*A_obs, kd = 2*kbase. d dfn/d kbase includes ∂dfn/∂kd·∂kd/∂kbase
        # = A_obs·2, plus the dynamics term through A_obs (kbase is the A->B rate).
        def run(kbase):
            m = bngsim.Model.from_net(DERIVED_NET)
            m.set_param("kbase", kbase)
            sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kbase", "k2"])
            m.reset()
            return sim.run(t_span=T_SPAN, n_points=N_POINTS, **_FD)

        r = run(0.30)
        di = r.expression_names.index("dfn")
        kb = r.sensitivity_params.index("kbase")
        analytic = r.sensitivities_expressions[:, di, kb]
        h = 0.30 * 1e-6
        fd = (
            np.asarray(run(0.30 + h).expressions["dfn"])
            - np.asarray(run(0.30 - h).expressions["dfn"])
        ) / (2 * h)
        _assert_fd_close(analytic, fd, float(np.max(np.abs(r.expressions["dfn"]))))


class TestNoObservables:
    """A model with global functions but NO observables (so the codegen emits no
    obs[] buffer) must still compile and compute parameter-only sensitivities."""

    def test_param_only_function(self, tmp_path):
        net = tmp_path / "noobs_fn.net"
        net.write_text(
            "begin parameters\n 1 k1 0.1\n 2 scale 2.0\nend parameters\n"
            "begin species\n 1 A() 100\n 2 B() 0\nend species\n"
            "begin reactions\n 1 1 2 k1 #A_to_B\nend reactions\n"
            "begin functions\n 1 pf() scale*k1\nend functions\n"
        )
        m = bngsim.Model.from_net(str(net))
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1", "scale"])
        m.reset()
        r = sim.run(t_span=(0.0, 2.0), n_points=3, **_RUN)
        es = r.sensitivities_expressions  # (3, 1, 2)
        pi = r.expression_names.index("pf")
        # pf = scale*k1 ⇒ d pf/d k1 = scale = 2.0; d pf/d scale = k1 = 0.1 (constant).
        assert np.allclose(es[:, pi, r.sensitivity_params.index("k1")], 2.0)
        assert np.allclose(es[:, pi, r.sensitivity_params.index("scale")], 0.1)


class TestSelectorAPI:
    def test_shapes_and_stacking(self):
        r = _run_chain()
        out = r.output_sensitivities(["expression:scaled", "expression:combo"])
        assert out.shape == (N_POINTS, 2, len(CHAIN_PARAMS))

    def test_function_call_convention(self):
        # A trailing () is the .gdat header convention; resolve it to the column.
        r = _run_chain()
        assert r.output_sensitivities("expression:ratio()").shape == (
            N_POINTS,
            1,
            len(CHAIN_PARAMS),
        )

    def test_ic_axis_selector(self):
        r = _run_chain(params=("k1",), ic=["A()", "B()", "C()"])
        out = r.output_sensitivities("expression:scaled", axis="ic")
        assert out.shape == (N_POINTS, 1, 3)


class TestUnsupportedConstructs:
    """Every unsupported construct fails loudly with a targeted reason (#198)."""

    @pytest.mark.parametrize(
        "ename,needle",
        [
            ("if_fn", "if()"),
            ("cmp_fn", "comparison"),
            ("abs_fn", "abs()"),
            ("min_fn", "min()"),
            ("max_fn", "max()"),
            ("floor_fn", "floor()"),
            ("ceil_fn", "ceil()"),
            ("rint_fn", "rounding"),
            ("trans", "depends on unsupported function"),
        ],
    )
    def test_unsupported_raises(self, ename, needle):
        m = bngsim.Model.from_net(UNSUPPORTED_NET)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        m.reset()
        r = sim.run(t_span=T_SPAN, n_points=4)
        with pytest.raises(ValueError, match=needle):
            r.output_sensitivities(f"expression:{ename}")

    def test_supported_sibling_still_works(self):
        m = bngsim.Model.from_net(UNSUPPORTED_NET)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        m.reset()
        r = sim.run(t_span=T_SPAN, n_points=4)
        assert np.all(np.isfinite(r.output_sensitivities("expression:good")))

    def test_table_function_unsupported(self):
        # Table functions are not differentiated, so their output sensitivity is
        # rejected loudly (not deferred — this is permanent).
        m = bngsim.Model.from_net(TFUN_NET)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
        m.reset()
        r = sim.run(t_span=(0.0, 5.0), n_points=4)
        with pytest.raises(ValueError, match="table-function"):
            r.output_sensitivities("expression:response")


class TestCodegenInactive:
    def test_no_codegen_with_sensitivity_raises(self, monkeypatch):
        # Hard requirement (GH #214): BNGSIM_NO_CODEGEN + sensitivity_params is a
        # contradiction (the interpreted finite-difference sens path was retired).
        # The refusal now happens at construction, before any run. Override the
        # autouse fixture.
        monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
        monkeypatch.delenv("BNGSIM_CODEGEN_THRESHOLD", raising=False)
        m = bngsim.Model.from_net(CHAIN_NET)
        with pytest.raises(ValueError, match="requires code generation"):
            bngsim.Simulator(m, method="ode", sensitivity_params=["k1", "scale"])


class TestDifferentiatorUnit:
    """Direct unit coverage of the no-inline arbitrary-symbol differentiator,
    including the species-partial path (functions rarely reference a species
    directly in practice — auto-observables shadow them — so exercise it here)."""

    _CREF = dict(
        species_cref={"A": "y[0]", "B": "y[1]"},
        observable_cref={"Atot": "obs[0]"},
        param_cref={"k": "p[0]", "eps": "p[1]"},
        function_cref={"g": "func[0]"},
    )

    def test_species_partial(self):
        # ∂(k·A²)/∂A = 2·k·A (species partial, in y[0]/p[0]); ∂/∂k = A².
        partials, reason = differentiate_expression_output_partials("k*A^2", **self._CREF)
        assert reason is None
        assert set(partials["species"]) == {"A"} and set(partials["param"]) == {"k"}
        assert "y[0]" in partials["species"]["A"] and "p[0]" in partials["species"]["A"]

    def test_mixed_kinds(self):
        partials, reason = differentiate_expression_output_partials("k*Atot + g", **self._CREF)
        assert reason is None
        assert set(partials["param"]) == {"k"}
        assert set(partials["observable"]) == {"Atot"}
        assert set(partials["function"]) == {"g"}

    @pytest.mark.parametrize(
        "body,needle",
        [
            ("if(A>1,A,0)", "if()"),
            ("A*(A>1)", "comparison"),
            ("abs(A)", "abs()"),
            ("floor(A)", "floor()"),
            ("rint(A)", "rounding"),
            ("A*zzz", "unrecognized symbol"),
        ],
    )
    def test_unsupported_reasons(self, body, needle):
        partials, reason = differentiate_expression_output_partials(body, **self._CREF)
        assert partials is None
        assert needle in reason

    def test_constant_is_supported_empty(self):
        partials, reason = differentiate_expression_output_partials("5.0", **self._CREF)
        assert reason is None
        assert partials == {"species": {}, "observable": {}, "param": {}, "function": {}}
