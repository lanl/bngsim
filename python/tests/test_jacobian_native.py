"""Unit + integration tests for the native, SymPy-free saturable-family Jacobian
(``bngsim._saturable_jacobian``, GH #151).

The native differentiator recognizes the fixed saturable rate-law family — Hill
terms, rational/saturation (``Sat``/``Hill`` ``.net`` rewrites, #48), basal +
regulated production, and products / shared-denominator sums over several
regulators — and emits closed-form derivatives directly, with no SymPy.

Oracle (same as ``test_jacobian_symbolic.py``): an emitted derivative is correct
iff, evaluated at a point, it matches a central finite difference of the original
rate law. The emitted ExprTk string is re-parsed and evaluated, and the emitted C
string is evaluated as Python (C ``pow``/``sqrt``/``log``/``exp`` and the
repeated-multiply / ``1.0/(…)`` power idioms are all valid Python too), so both
emitters are checked end-to-end. Fallback contracts assert ``None`` (→ SymPy →
FD) rather than a wrong answer for inputs outside the family.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# The native engine itself imports no SymPy; the FD oracle re-parses emitted
# ExprTk via the SymPy helper, so the *tests* need SymPy even though the code
# under test does not.
sympy = pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim import _jacobian as J  # noqa: E402
from bngsim import _saturable_jacobian as N  # noqa: E402

_TIME = J._TIME_SYM


def _lambdify(exprtk_str, varnames):
    e = J._exprtk_to_sympy(exprtk_str)
    assert e is not None, f"re-parse failed: {exprtk_str!r}"
    return sp.lambdify([sp.Symbol(v) for v in varnames], e, "numpy")


def _fd(f, base, var, h_rel=1e-6):
    h = h_rel * max(abs(base[var]), 1.0)
    bp = dict(base)
    bp[var] += h
    bm = dict(base)
    bm[var] -= h
    return (float(f(**bp)) - float(f(**bm))) / (2.0 * h)


def _eval_c(node, env):
    """Emit ``node`` as C and evaluate it as Python. C ``pow``/``sqrt``/``log``/
    ``exp`` map to ``math``; ``((x)*(x))`` and ``(1.0/(x))`` are valid Python."""
    keys = {name: f"_v{i}" for i, name in enumerate(env)}

    def resolve(name):
        if name == _TIME:
            return "_t"
        return keys.get(name)

    c = N.emit_c(node, resolve)
    assert c is not None, "emit_c returned None"
    ns = {keys[name]: env[name] for name in env}
    ns.update(
        {
            "pow": math.pow,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "_t": env.get(_TIME, 0.0),
        }
    )
    return float(eval(c, {"__builtins__": {}}, ns))  # noqa: S307 — trusted emitter output


# Every form named in the GH #151 scope. (name, expr, observables, constants).
# Includes the exact synthetic bodies the .net loader emits for Sat/Hill (#48).
FAMILY_CASES = [
    ("michaelis_menten", "(Vmax*A)/(Km+A)", {"A"}, {"Vmax", "Km"}),
    ("sat_k_over_K_plus_S", "k3 / (K4 + G)", {"G"}, {"k3", "K4"}),
    ("sat_two_constants", "k / ((K1 + S1) * (K2 + S2))", {"S1", "S2"}, {"k", "K1", "K2"}),
    # Loader-emitted Sat body (parenthesize_product of one term).
    ("sat_loader_body", "k3 / ((K4 + Stot))", {"Stot"}, {"k3", "K4"}),
    ("hill_theta_powers", "(S^h)/((K^h)+(S^h))", {"S"}, {"K", "h"}),
    ("hill_theta_ratio", "((S/K)^h)/(1+((S/K)^h))", {"S"}, {"K", "h"}),
    # Loader-emitted Hill body: Vmax * S^(h-1) / (Kh^h + S^h).
    (
        "hill_loader_body",
        "k * (Stot ^ (n - 1)) / ((K ^ n) + (Stot ^ n))",
        {"Stot"},
        {"k", "K", "n"},
    ),
    (
        "basal_plus_regulated",
        "(k0 + k1*((S^n)/((Km^n)+(S^n))))*P",
        {"S", "P"},
        {"k0", "k1", "Km", "n"},
    ),
    (
        "multi_regulator_product",
        "((A^2)/((Ka^2)+(A^2)))*((B^3)/((Kb^3)+(B^3)))",
        {"A", "B"},
        {"Ka", "Kb"},
    ),
    ("shared_denominator_sum", "((A/Ka)+(B/Kb))/(1+((A/Ka)+(B/Kb)))", {"A", "B"}, {"Ka", "Kb"}),
    ("const_scalar_factor", "vol * k * A / (K + A)", {"A"}, {"vol", "k", "K"}),
    ("with_time_factor", "k*A*time()", {"A"}, {"k"}),
    ("sqrt_and_exp", "k*sqrt(A)*exp(-B/Km)", {"A", "B"}, {"k", "Km"}),
]

_VALS = {
    "A": 1.7,
    "B": 0.6,
    "C": 2.3,
    "G": 1.7,
    "P": 2.4,
    "S": 1.3,
    "Stot": 1.3,
    "S1": 0.9,
    "S2": 2.1,
    "k": 0.5,
    "k0": 0.2,
    "k1": 1.3,
    "k3": 0.7,
    "K": 0.8,
    "Ka": 0.55,
    "Kb": 1.2,
    "K1": 0.4,
    "K2": 1.1,
    "K4": 0.9,
    "Km": 0.8,
    "Kh": 0.9,
    "Vmax": 3.0,
    "vol": 1.5,
    "h": 2.0,
    "n": 3.0,
    _TIME: 4.2,
}


@pytest.mark.parametrize("name,expr,obs,const", FAMILY_CASES, ids=[c[0] for c in FAMILY_CASES])
def test_native_exprtk_matches_finite_difference(name, expr, obs, const):
    """Every native ExprTk ∂rate/∂obs matches a central FD of the original."""
    dd = N.differentiate_rate_law_native(expr, {}, obs, const)
    assert dd is not None, f"{name}: native differentiator unexpectedly fell back"

    varnames = sorted(obs | const) + [_TIME]
    f = _lambdify(expr, varnames)
    base = {v: _VALS[v] for v in varnames}
    for obs_name in obs:
        node = dd.get(obs_name)
        if node is None:
            # Absent ⇒ the derivative must be genuinely zero.
            assert _fd(f, base, obs_name) == pytest.approx(0.0, abs=1e-7)
            continue
        s = N.emit_exprtk(node)
        assert s is not None, f"{name}: ExprTk emission failed for d/d{obs_name}"
        analytic = float(_lambdify(s, varnames)(**base))
        fd = _fd(f, base, obs_name)
        assert analytic == pytest.approx(fd, rel=1e-8, abs=1e-9), (
            f"{name} d/d{obs_name}: analytic={analytic} fd={fd} expr={s}"
        )


@pytest.mark.parametrize("name,expr,obs,const", FAMILY_CASES, ids=[c[0] for c in FAMILY_CASES])
def test_native_c_matches_finite_difference(name, expr, obs, const):
    """Every native C ∂rate/∂obs matches a central FD of the original."""
    dd = N.differentiate_rate_law_native(expr, {}, obs, const)
    assert dd is not None
    varnames = sorted(obs | const) + [_TIME]
    f = _lambdify(expr, varnames)
    base = {v: _VALS[v] for v in varnames}
    env = {v: _VALS[v] for v in (obs | const)}
    env[_TIME] = _VALS[_TIME]  # bind time() consistently with the FD oracle
    for obs_name in obs:
        node = dd.get(obs_name)
        if node is None:
            continue
        analytic = _eval_c(node, env)
        fd = _fd(f, base, obs_name)
        assert analytic == pytest.approx(fd, rel=1e-8, abs=1e-9), (
            f"{name} d/d{obs_name}: C analytic={analytic} fd={fd}"
        )


# ─── Fallback contracts: outside the family ⇒ None (never a wrong derivative) ──

FALLBACK_CASES = [
    ("piecewise_if", "if(A>Km,k*A,k*Km)", {"A"}, {"k", "Km"}),
    ("comparison", "k*(A>Km)", {"A"}, {"k", "Km"}),
    ("logical_and", "k*A*(A and Km)", {"A"}, {"k", "Km"}),
    ("unknown_function", "k*foo(A)", {"A"}, {"k"}),
    ("unknown_symbol", "k*A*Z", {"A"}, {"k"}),  # Z neither observable nor constant
    ("keyword_named_param", "lambda*A/(Km+A)", {"A"}, {"lambda", "Km"}),
    ("modulo_operator", "k*A%Km", {"A"}, {"k", "Km"}),
    # General power with the state in BOTH base and exponent (x^x kind): the
    # closed form divides by the base / takes ln(base) — a removable 0/0 at
    # base=0 — so the native path declines it (SymPy simplifies A/A→1 better).
    ("general_power_x_pow_x", "A^A", {"A"}, set()),
    ("general_power_in_base_and_exp", "k*A^(A+1)", {"A"}, {"k"}),
]


@pytest.mark.parametrize("name,expr,obs,const", FALLBACK_CASES, ids=[c[0] for c in FALLBACK_CASES])
def test_native_falls_back_outside_family(name, expr, obs, const):
    """An expression outside the recognized family yields None, deferring to the
    SymPy path — never a silently-wrong derivative."""
    assert N.differentiate_rate_law_native(expr, {}, obs, const) is None


def test_constant_rate_is_a_success_not_a_fallback():
    """A rate law with no observable dependence returns ``{}`` (a success — zero
    Jacobian column), distinct from ``None`` (could-not-differentiate)."""
    dd = N.differentiate_rate_law_native("k*Km + 3.0", {}, {"A"}, {"k", "Km"})
    assert dd == {}


def test_function_inlining_then_native():
    """A rate law referencing a user function is inlined, then differentiated
    natively (the func body is itself in the family)."""
    dd = N.differentiate_rate_law_native(
        "theta", {"theta": "(S^h)/((K^h)+(S^h))"}, {"S"}, {"K", "h"}
    )
    assert dd is not None and "S" in dd


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: end-to-end .net Sat/Hill with zero SymPy invocations
# ═══════════════════════════════════════════════════════════════════════════════


def _spy_no_sympy(monkeypatch):
    """Make every SymPy-path entry point in ``bngsim._jacobian`` raise, so a test
    fails loudly if the native path does not fully cover the model."""

    def _boom(*_a, **_k):
        raise AssertionError("SymPy Jacobian path was invoked")

    monkeypatch.setattr(J, "differentiate_rate_law", _boom)
    monkeypatch.setattr(J, "build_per_species_sympy", _boom)


def test_net_sat_analytical_jacobian_complete_no_sympy(sat_rewrite_net: Path, monkeypatch):
    """A legacy ``Sat`` ``.net`` model attaches a *complete* analytical Jacobian
    using only the native path — proven by making the SymPy path raise."""
    from bngsim import Model

    _spy_no_sympy(monkeypatch)
    with pytest.warns(UserWarning, match="Sat"):
        model = Model.from_net(sat_rewrite_net)
    assert model.prepare_analytical_jacobian() is True
    assert model._core.analytical_jacobian_complete is True


def test_net_hill_analytical_jacobian_complete_no_sympy(hill_rewrite_net: Path, monkeypatch):
    """A legacy ``Hill`` ``.net`` model attaches a *complete* analytical Jacobian
    using only the native path."""
    from bngsim import Model

    _spy_no_sympy(monkeypatch)
    with pytest.warns(UserWarning, match="Hill"):
        model = Model.from_net(hill_rewrite_net)
    assert model.prepare_analytical_jacobian() is True
    assert model._core.analytical_jacobian_complete is True


def test_net_sat_codegen_jacobian_emits_no_sympy(sat_rewrite_net: Path, monkeypatch):
    """The codegen analytical-Jacobian emitter produces C for the Sat model with
    no SymPy invocation."""
    from bngsim import Model
    from bngsim._codegen import generate_jacobian_from_model

    with pytest.warns(UserWarning, match="Sat"):
        model = Model.from_net(sat_rewrite_net)
    assert model.prepare_analytical_jacobian() is True
    _spy_no_sympy(monkeypatch)
    c_src = generate_jacobian_from_model(model)
    assert c_src is not None and "bngsim_codegen_jac" in c_src


@pytest.mark.parametrize("net_fixture", ["sat_rewrite_net", "hill_rewrite_net"])
def test_net_analytical_matches_finite_difference_trajectory(net_fixture, request):
    """The analytical-Jacobian ODE trajectory matches the finite-difference one:
    the C++ FD self-check passes (native derivatives are correct) and the solve
    agrees regardless of which Jacobian drives it."""
    import numpy as np
    from bngsim import Model, Simulator

    net_path: Path = request.getfixturevalue(net_fixture)

    with pytest.warns(UserWarning):
        model = Model.from_net(net_path)
    assert model.prepare_analytical_jacobian() is True
    analytical = Simulator(model, method="ode").run(t_span=(0, 20), n_points=41)

    with pytest.warns(UserWarning):
        model_fd = Model.from_net(net_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", "0")
    try:
        model_fd.prepare_analytical_jacobian()
        fd_run = Simulator(model_fd, method="ode").run(t_span=(0, 20), n_points=41)
    finally:
        monkeypatch.undo()

    np.testing.assert_allclose(analytical.species, fd_run.species, rtol=1e-7, atol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse + sampled self-validation gate (GH #151): the large-model path that
# validates the analytical Jacobian without a dense n×n buffer. Forced on a small
# model via BNGSIM_JAC_SELFCHECK_DENSE_MAX=0 so it runs in CI; it must accept the
# (correct) analytical Jacobian and integrate identically to the dense-validated
# default path.
# ═══════════════════════════════════════════════════════════════════════════════


def test_sparse_selfcheck_matches_dense_net(sat_rewrite_net: Path):
    """Forcing the sparse+sampled self-check path yields the same complete
    analytical Jacobian and the same ODE trajectory as the dense path."""
    import numpy as np
    from bngsim import Model, Simulator

    with pytest.warns(UserWarning):
        m_dense = Model.from_net(sat_rewrite_net)
    assert m_dense.prepare_analytical_jacobian() is True
    r_dense = Simulator(m_dense, method="ode").run(t_span=(0, 20), n_points=41)

    mp = pytest.MonkeyPatch()
    mp.setenv("BNGSIM_JAC_SELFCHECK_DENSE_MAX", "0")  # ns>0 ⇒ sparse self-check
    try:
        with pytest.warns(UserWarning):
            m_sparse = Model.from_net(sat_rewrite_net)
        assert m_sparse.prepare_analytical_jacobian() is True
        assert m_sparse._core.analytical_jacobian_complete is True
        r_sparse = Simulator(m_sparse, method="ode").run(t_span=(0, 20), n_points=41)
    finally:
        mp.undo()

    np.testing.assert_allclose(r_dense.species, r_sparse.species, rtol=1e-12, atol=1e-14)


def test_sparse_selfcheck_matches_dense_sbml(data_dir: Path):
    """Same, on an SBML model with per-species Functional reactions (the SBML
    chain-rule path, distinct from the .net per-observable path)."""
    import numpy as np
    from bngsim import Model, Simulator

    sbml = data_dir / "BIOMD0000000003.xml"
    if not sbml.exists():
        pytest.skip("BIOMD0000000003.xml fixture not present")

    m_dense = Model.from_sbml(sbml)
    assert m_dense.prepare_analytical_jacobian() is True
    r_dense = Simulator(m_dense, method="ode").run(t_span=(0, 50), n_points=51)

    mp = pytest.MonkeyPatch()
    mp.setenv("BNGSIM_JAC_SELFCHECK_DENSE_MAX", "0")
    try:
        m_sparse = Model.from_sbml(sbml)
        assert m_sparse.prepare_analytical_jacobian() is True
        assert m_sparse._core.analytical_jacobian_complete is True
        r_sparse = Simulator(m_sparse, method="ode").run(t_span=(0, 50), n_points=51)
    finally:
        mp.undo()

    np.testing.assert_allclose(r_dense.species, r_sparse.species, rtol=1e-10, atol=1e-12)


def test_stable_quotient_no_overflow_at_extreme_state():
    """The quotient rule emits the numerically stable da/b − (a/b)·(db/b) form, so
    a saturable term whose denominator blows up at an extreme state stays finite
    (the naive (da·b − a·db)/b² overflows to inf − inf = nan). Regression for a
    genome-scale-model self-check failure."""
    # Hill term in conc/volume with a very small compartment volume, the shape that
    # makes the intermediate denominator enormous.
    expr = "k1 * (S / 1.75e-12) * (((R / 1.75e-12) / K)^h / (1 + ((R / 1.75e-12) / K)^h))"
    dd = N.differentiate_rate_law_native(expr, {}, {"S", "R"}, {"k1", "K", "h"})
    assert dd is not None and "R" in dd
    s = N.emit_exprtk(dd["R"])
    assert s is not None
    # Evaluate ∂/∂R at a state large enough that the naive numerator would overflow.
    f = _lambdify(s, ["S", "R", "k1", "K", "h"])
    val = float(f(S=1e-3, R=1e-3, k1=1.0, K=1.0, h=8.0))
    assert math.isfinite(val), f"derivative overflowed to non-finite: {val}"
