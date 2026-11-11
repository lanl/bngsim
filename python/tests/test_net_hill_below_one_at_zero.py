"""A legacy ``.net`` Hill law with 0 < h < 1 at zero substrate (issue #1497).

The loader rewrites ``Hill Vmax Kh h`` into the functional rate constant
``Vmax*S^(h-1)/(Kh^h + S^h)``, which the reaction multiplies by S. For
0 < h < 1 that is ``inf*0`` = NaN at S = 0, where the law Vmax*S^h/(Kh^h + S^h)
and run_network's HILL are 0. Every run reaching S = 0 failed: ODE (interpreted
and compiled), SSA and PSA, and ``Model.rhs`` returned NaN.

The loader now writes ``if((S > 0) or (h <= 0) or (h >= 1), <body>, 0)``. Every
S > 0 reads the body as before, and so does every S at h >= 1 and h <= 0, so
those models compute what they did (``test_what_must_not_change``).

Oracles: the closed form where nothing makes S (S = P = 0, X = e^-t), the
absorbing state where S is only consumed (S = 0, P = S(0)), and run_network 3.0
(BioNetGen 2.9.3) on the same bytes where S is made, pinned here as literals
because CI has no run_network. Each was measured with ``-a 1e-12 -r 1e-10``.
"""

from __future__ import annotations

import contextlib
import io
import math
import shutil
import warnings

import bngsim
import numpy as np
import pytest
from bngsim import _jacobian as jac
from bngsim import _saturable_jacobian as sat
from bngsim._net_reader import build_model_from_parsed, parse_net_file

NET = """begin parameters
    1 Vmax 1
    2 Kh 2
    3 h {h}
    4 k 1
    5 kin {kin}
end parameters
begin species
    1 S() {s0}
    2 P() 0
    3 X() 1
    4 E() 3
end species
begin reactions
    1 {lhs} {rhs} {law} Vmax Kh h
    2 3 0 k
    3 0 1 kin
end reactions
begin groups
    1 S_obs 1
    2 P_obs 2
    3 X_obs 3
end groups
"""

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH for the codegen leg")

E_INV = math.exp(-1.0)


def _net(tmp_path, *, h=0.5, s0=0, kin=0, lhs="1", rhs="2", law="Hill") -> str:
    path = tmp_path / "hill.net"
    path.write_text(NET.format(h=h, s0=s0, kin=kin, lhs=lhs, rhs=rhs, law=law))
    return str(path)


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Model.from_net(path)


def _ode(path, t_end, **kw):
    with contextlib.redirect_stderr(io.StringIO()):
        sim = bngsim.Simulator(_load(path), method="ode", **kw)
        r = sim.run(t_span=(0.0, t_end), n_points=2, rtol=1e-10, atol=1e-12, timeout=60)
    return np.asarray(r.species)[-1][:3]


ODE_ROUTES = [
    pytest.param({}, id="auto"),
    pytest.param({"jacobian": "analytical"}, id="analytical"),
    pytest.param({"jacobian": "fd"}, id="fd"),
    pytest.param({"codegen": True}, id="codegen", marks=needs_cc),
]


@pytest.mark.parametrize("kw", ODE_ROUTES)
@pytest.mark.parametrize("h", [0.001, 0.5, 0.999])
def test_nothing_makes_s_and_the_run_is_the_closed_form(tmp_path, kw, h):
    """S(0) = 0 and nothing makes S: S = P = 0 and X = e^-t. Raised
    CV_FIRST_RHSFUNC_ERR on main, on every route."""
    np.testing.assert_allclose(
        _ode(_net(tmp_path, h=h), 1.0, **kw), [0.0, 0.0, E_INV], rtol=1e-7, atol=1e-12
    )


@pytest.mark.parametrize("kw", ODE_ROUTES)
@pytest.mark.parametrize(
    ("h", "run_network"),
    [
        (0.001, [2.50097, 2.49903]),
        (0.5, [2.7960281, 2.2039719]),
        (0.999, [2.8988689, 2.1011311]),
    ],
)
def test_s_made_from_zero_matches_run_network(tmp_path, kw, h, run_network):
    """S(0) = 0 and an influx of 1: the Hill flux leaves S = 0 with an infinite
    slope. S and P at t = 5 against run_network, to the 7 digits pinned (the
    run's own tolerance is 1e-10)."""
    got = _ode(_net(tmp_path, h=h, kin=1), 5.0, **kw)
    np.testing.assert_allclose(got[:2], run_network, rtol=0, atol=6e-7)


@pytest.mark.parametrize(
    ("lhs", "rhs", "law", "run_network"),
    [
        ("1,4", "2,4", "Hill", [0.4887865, 4.5112135]),
        ("1", "2", "2*Hill", [1.2578603, 3.7421397]),
    ],
    ids=["S + E -> P + E", "2*Hill"],
)
def test_the_other_shapes_of_the_rewrite(tmp_path, lhs, rhs, law, run_network):
    """A second reactant multiplies the rate, and `2*Hill` scales it (#1230):
    both reach the same body. At S(0) = 0 with nothing making S, the closed
    form; with an influx, run_network at t = 5."""
    np.testing.assert_allclose(
        _ode(_net(tmp_path, lhs=lhs, rhs=rhs, law=law), 1.0),
        [0.0, 0.0, E_INV],
        rtol=1e-7,
        atol=1e-12,
    )
    got = _ode(_net(tmp_path, kin=1, lhs=lhs, rhs=rhs, law=law), 5.0)
    np.testing.assert_allclose(got[:2], run_network, rtol=0, atol=6e-7)


@pytest.mark.parametrize("lhs", ["1", "1,4"])
def test_a_run_that_drains_s_reaches_the_absorbing_state(tmp_path, lhs):
    """S(0) = 20, only consumed: S reaches 0 in finite time for h < 1 and stays
    there, so S = 0 and P = 20. Main and run_network both fail this run at the
    step that crosses 0 (a negative S to a non-integer power is NaN)."""
    rhs = "2" if lhs == "1" else "2,4"
    got = _ode(_net(tmp_path, s0=20, lhs=lhs, rhs=rhs), 200.0)
    np.testing.assert_allclose(got[:2], [0.0, 20.0], rtol=0, atol=1e-8)


def test_s_held_at_zero_by_a_saturated_flux(tmp_path):
    """An influx of 1 against S + E -> P + E whose flux is about 1.5 at the
    smallest positive S when h = 0.001: S slides along 0 and P gains the whole
    influx, so P(5) = S(0) + 5. run_network's CVODE fails here (error -4).
    Guarded with `S == 0` instead of `S > 0`, the run ground at S = 0 until the
    timeout, because the core roots an inequality over state and not an
    equality; this pins the spelling."""
    got = _ode(_net(tmp_path, h=0.001, s0=1e-6, kin=1, lhs="1,4", rhs="2,4"), 5.0)
    np.testing.assert_allclose(got[:2], [0.0, 5.000001], rtol=0, atol=1e-8)


def test_a_run_that_starts_before_zero(tmp_path):
    """S(0) = 0 at t = -2 with the influx: the run is the one from t = 0 moved
    by 2, so at t = 3 it is run_network's t = 5 (run_network ``-i -2`` gives
    the same digits)."""
    path = _net(tmp_path, kin=1)
    with contextlib.redirect_stderr(io.StringIO()):
        r = bngsim.Simulator(_load(path), method="ode").run(
            t_span=(-2.0, 3.0), n_points=2, rtol=1e-10, atol=1e-12, timeout=60
        )
    np.testing.assert_allclose(
        np.asarray(r.species)[-1][:2], [2.7960281, 2.2039719], rtol=0, atol=6e-7
    )


def test_the_steady_state_from_zero(tmp_path):
    """0 -> S at 0.2, S -> P by Hill(1, 2, 0.5), P -> 0 at 2, from S = P = 0.
    The flux sqrt(S)/(sqrt(2) + sqrt(S)) = 0.2 gives S* = 1/8 and P* = 0.1.
    Main's march started on the NaN and returned NaN."""
    net = tmp_path / "ss.net"
    net.write_text(
        "begin parameters\n 1 Vmax 1\n 2 Kh 2\n 3 h 0.5\n 4 kin 0.2\n 5 kp 2\nend parameters\n"
        "begin species\n 1 S() 0\n 2 P() 0\nend species\n"
        "begin reactions\n 1 1 2 Hill Vmax Kh h\n 2 0 1 kin\n 3 2 0 kp\nend reactions\n"
        "begin groups\n 1 S_obs 1\n 2 P_obs 2\nend groups\n"
    )
    with contextlib.redirect_stderr(io.StringIO()):
        ss = bngsim.Simulator(_load(str(net)), method="ode").steady_state()
    assert ss.converged
    np.testing.assert_allclose(np.ravel(ss.concentrations), [0.125, 0.1], rtol=1e-9)


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_stochastic_run_reaches_the_absorbing_state(tmp_path, method):
    """S(0) = 20 molecules: the run used to stop at the last one consumed, on
    a NaN propensity. The absorbing state is S = 0, P = 20."""
    kw = {"poplevel": 100} if method == "psa" else {}
    sim = bngsim.Simulator(_load(_net(tmp_path, s0=20)), method=method, **kw)
    r = sim.run(t_span=(0.0, 200.0), n_points=2, seed=1)
    assert np.asarray(r.species)[-1][:2].tolist() == [0.0, 20.0]


@needs_cc
def test_forward_sensitivities_from_zero(tmp_path):
    """Forward sensitivities of S and P at t = 5 to h, Vmax and kin, from S = 0
    with the influx. They differentiate the guarded function by parameter, where
    d/dh of the body carries log(S), -inf at S = 0. On main the run refused the
    NaN at t = 0.

    Two oracles. The Hill reaction only turns S into P, and only the influx
    makes either, so S + P = kin*t: its sensitivity is exactly 0 to h and Vmax
    and exactly t = 5 to kin. And each column against a central difference of
    two runs with the parameter moved by 1e-5 of itself. At rtol 1e-12 a run's
    error over that step is about 1e-7 of a column, and the truncation error is
    about 1e-10, so the columns are held to 1e-6. Measured: 7e-8 at most, and
    the sums within 1e-9."""
    sens_params = ["h", "Vmax", "kin"]
    tol = {"rtol": 1e-12, "atol": 1e-14, "timeout": 60}

    def run(model, **kw):
        with contextlib.redirect_stderr(io.StringIO()):
            sim = bngsim.Simulator(model, method="ode", **kw)
            return sim.run(t_span=(0.0, 5.0), n_points=2, **tol)

    path = _net(tmp_path, kin=1)
    r = run(_load(path), sensitivity_params=sens_params)
    sens = np.asarray(r.sensitivities)[-1][:2]  # rows S, P; columns as sens_params
    assert np.all(np.isfinite(sens))
    np.testing.assert_allclose(sens.sum(axis=0), [0.0, 0.0, 5.0], rtol=0, atol=1e-8)

    base = {"h": 0.5, "Vmax": 1.0, "kin": 1.0}
    for j, name in enumerate(sens_params):
        step = 1e-5 * base[name]
        ends = []
        for value in (base[name] + step, base[name] - step):
            m = _load(path)
            m.set_param(name, value)
            ends.append(np.asarray(run(m).species)[-1][:2])
        central = (ends[0] - ends[1]) / (2 * step)
        np.testing.assert_allclose(sens[:, j], central, rtol=0, atol=1e-6, err_msg=name)


def test_rhs_and_the_parsed_dict_read_zero(tmp_path):
    """Model.rhs at S = 0, and the model `parse_net_file` hands to
    `build_model_from_parsed`, which reads the same C++ rewrite."""
    path = _net(tmp_path)
    y = [0.0, 0.0, 1.0, 3.0]
    assert _load(path).rhs(y).tolist() == [0.0, 0.0, -1.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert build_model_from_parsed(parse_net_file(path)).rhs(y).tolist() == [
            0.0,
            0.0,
            -1.0,
            0.0,
        ]


@pytest.mark.parametrize("h", [1.0, 2.0, 0.0, 0.5])
def test_what_must_not_change(tmp_path, h):
    """The guard reads the body wherever S > 0, and at every S for h >= 1 or
    h <= 0. So at S > 0 the RHS is the body's, bit for bit, at every h; and at
    S = 0 it is the body's for h = 1 (rate 0, Jacobian slope Vmax/Kh), h = 2
    (0, 0) and h = 0 (NaN, as on main: run_network's law is Vmax/2 there, not
    0). Only 0 < h < 1 changes, and only at S <= 0."""
    m = _load(_net(tmp_path, h=h))
    for s in (1e-300, 1e-8, 0.3, 7.0):
        body = 1.0 * s ** (h - 1) / (2.0**h + s**h) * s
        assert m.rhs([s, 0.0, 1.0, 3.0])[1] == body
    at_zero = m.rhs([0.0, 0.0, 1.0, 3.0])[1]
    slope = m.jacobian([0.0, 0.0, 1.0, 3.0])[1, 0]
    if h == 1.0:
        assert (at_zero, slope) == (0.0, 0.5)
    elif h == 2.0:
        assert (at_zero, slope) == (0.0, 0.0)
    elif h == 0.0:
        assert math.isnan(at_zero)
    else:
        assert at_zero == 0.0
        assert m.rhs([-1e-9, 0.0, 1.0, 3.0])[1] == 0.0


def test_the_native_jacobian_still_takes_the_rewrite(tmp_path):
    """The rewrite is in the saturable family #151 differentiates without
    SymPy, so a guarded Hill model keeps that path: one native derivation, no
    SymPy call, and its derivative is main's under the same guard."""
    calls = {"native": 0, "sympy": 0}
    native, symbolic = jac._native_per_observable_terms, jac.differentiate_rate_law
    out = []

    def count_native(*a, **k):
        r = native(*a, **k)
        calls["native"] += r is not None
        out.append(r)
        return r

    def count_sympy(*a, **k):
        calls["sympy"] += 1
        return symbolic(*a, **k)

    jac._native_per_observable_terms = count_native
    jac.differentiate_rate_law = count_sympy
    try:
        _ode(_net(tmp_path, h=0.5, s0=100), 1.0)
    finally:
        jac._native_per_observable_terms, jac.differentiate_rate_law = native, symbolic
    assert calls == {"native": 1, "sympy": 0}
    ((obs, text),) = out[0]
    assert text.startswith(f"if(({obs} > 0) or (h <= 0) or (h >= 1), ")
    assert text.endswith(", 0)")


def _hill_function(model) -> str:
    core = model._core
    pairs = zip(core.function_names, core.function_expressions, strict=True)
    (text,) = [e for n, e in pairs if n.startswith("__bngsim_net_rewrite_Hill")]
    return text


@pytest.mark.parametrize("h", ["2", "1", "3.5", "0", "-1"])
def test_an_h_known_outside_the_interval_keeps_main_s_body(tmp_path, h):
    """An h whose value at load is >= 1 or <= 0 can never meet the guard, and
    the guard is a root the core watches every step: 300 such reactions ran 10x
    slower with it. So the loader writes main's bare body, and no root."""
    m = _load(_net(tmp_path, h=h))
    assert _hill_function(m) == (
        "Vmax * (__bngsim_net_rewrite_obs_r1_1 ^ (h - 1)) / "
        "((Kh ^ h) + (__bngsim_net_rewrite_obs_r1_1 ^ h))"
    )
    assert m.state_switch_root_conditions() == ()


@pytest.mark.parametrize(
    "h", ["0.5", "0.999", "1/2", "h_base*2"], ids=["0.5", "0.999", "1/2", "h_base*2"]
)
def test_an_h_in_the_interval_or_not_known_takes_the_guard(tmp_path, h):
    """An h in (0, 1), or one whose value is an expression, so not known until
    the run, takes the guard; with it, the root that holds a run sliding at 0."""
    text = NET.format(h=h, s0=0, kin=0, lhs="1", rhs="2", law="Hill").replace(
        "    5 kin 0\n", "    5 kin 0\n    6 h_base 0.25\n"
    )
    path = tmp_path / "hill.net"
    path.write_text(text)
    m = _load(str(path))
    assert _hill_function(m).startswith("if((__bngsim_net_rewrite_obs_r1_1 > 0) or ")
    assert m.state_switch_root_conditions() != ()
    assert m.rhs([0.0, 0.0, 1.0, 3.0]).tolist() == [0.0, 0.0, -1.0, 0.0]


def test_the_guard_reads_h_as_the_run_has_it(tmp_path):
    """The guard tests h at run time, so a set_param that moves a guarded h
    out of (0, 1) gives back the body at S = 0: at h = 1 the rate 0 and the
    slope Vmax/Kh, and at h = 0 the NaN main gives (run_network's law there is
    Vmax/2, which a 0 would not be either)."""
    m = _load(_net(tmp_path, h=0.5))
    m.set_param("h", 1.0)
    assert m.rhs([0.0, 0.0, 1.0, 3.0])[1] == 0.0
    assert m.jacobian([0.0, 0.0, 1.0, 3.0])[1, 0] == 0.5
    m.set_param("h", 0.0)
    assert math.isnan(m.rhs([0.0, 0.0, 1.0, 3.0])[1])


@pytest.mark.parametrize(
    "law",
    [
        # Not the loader's guard: the condition, the bounds or the else differ.
        "if((S > 0) or (h <= 0), k*S^(h-1), 0)",
        "if((S > 0) or (h <= 0) or (h >= 2), k*S^(h-1), 0)",
        "if((S > 0) or (h <= 0) or (n >= 1), k*S^(h-1), 0)",
        "if((S > 0) or (h <= 0) or (h >= 1), k*S^(h-1), 1)",
        "if((S >= 0) or (h <= 0) or (h >= 1), k*S^(h-1), 0)",
        "if((S > 1) or (h <= 0) or (h >= 1), k*S^(h-1), 0)",
        "if(S > 0, k*S, 0)",
        "k*(S > 0)",
    ],
)
def test_no_other_if_enters_the_native_family(law):
    """The native parser reads the loader's guard and nothing else: any other
    `if`, comparison or logical operator still defers to SymPy (None)."""
    assert sat.differentiate_rate_law_native(law, {}, {"S"}, {"k", "h", "n"}) is None


@pytest.mark.parametrize(
    ("law", "observables"),
    [
        ("if((S > 0) or (h <= 0) or (h >= 1), k*log(S), 0)", {"S"}),
        ("if((A > 0) or (h <= 0) or (h >= 1), k*A^B, 0)", {"A", "B"}),
    ],
    ids=["log of the state (#336)", "state in base and exponent (#574)"],
)
def test_a_guarded_body_keeps_the_deferrals_inside_it(law, observables):
    """The guard is no way around #336's and #574's deferrals: a body that
    takes, or would synthesize, a logarithm of the state still goes to SymPy."""
    assert sat.differentiate_rate_law_native(law, {}, observables, {"k", "h"}) is None


def test_a_power_inside_a_guard_counts_as_in_the_sum():
    """#388's rewrite divides ``a + S^n`` through by ``S^n`` only when ``a``
    holds no ``S^n`` of its own, or it trades one overflow for another. A guard
    is no hiding place: ``S^n`` inside it is in ``a``, so nothing is rewritten."""
    node = sat._parse("k * S^n / (if((S > 0) or (n <= 0) or (n >= 1), S^n, 0) + S^n)")
    assert sat.rewrite_saturating_ratio(node) == node


def test_the_guard_differentiates_to_the_guarded_derivative():
    """d/dS of guard(body) is guard(d/dS body): the same text, inside."""
    body = "k * (S ^ (h - 1)) / ((K ^ h) + (S ^ h))"
    plain = sat.differentiate_rate_law_native(body, {}, {"S"}, {"k", "K", "h"})
    guarded = sat.differentiate_rate_law_native(
        f"if((S > 0) or (h <= 0) or (h >= 1), {body}, 0)", {}, {"S"}, {"k", "K", "h"}
    )
    inner = sat.emit_exprtk(plain["S"])
    assert sat.emit_exprtk(guarded["S"]) == f"if((S > 0) or (h <= 0) or (h >= 1), {inner}, 0)"
    c = sat.emit_c(guarded["S"], {"S": "y[0]", "k": "p[0]", "K": "p[1]", "h": "p[2]"}.get)
    assert c.startswith("(((y[0] > 0.0) || (p[2] <= 0.0) || (p[2] >= 1.0)) ? (") and c.endswith(
        ") : 0.0)"
    )


def test_the_writers_carry_it(tmp_path):
    """write_bngl refuses `!=` (BNG2.pl's parser has none), so the guard is
    spelled with `>`, `<=`, `>=` and `or`, which it writes as `||`. write_net
    reads back to the same run."""
    from bngsim.convert import write_bngl, write_net

    path = _net(tmp_path, kin=1)
    m = _load(path)
    assert "(h >= 1)" in write_bngl(m)
    out = tmp_path / "rt.net"
    write_net(m, str(out))
    np.testing.assert_allclose(_ode(str(out), 5.0)[:2], [2.7960281, 2.2039719], rtol=0, atol=6e-7)


def test_the_load_warning_writes_the_law_as_it_is_defined(tmp_path):
    """The warning used to advise `Vmax*S^(h-1)/(Kh^h+S^h)`, which has the same
    singularity in a BNGL functional rate law."""
    with pytest.warns(UserWarning) as caught:
        bngsim.Model.from_net(_net(tmp_path))
    text = " ".join(str(w.message) for w in caught)
    assert "if(S > 0, Vmax*S^(h-1)/(Kh^h+S^h), 0)" in text
