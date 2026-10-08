"""Review tests for lanl/bngsim#803 step 4: the JAX RHS and ``run_diffrax`` on the
built model, checked against oracles that do not read the model through bngsim.

``Model.rhs`` (the engine's interpreted RHS) and the JAX RHS both read the model
bngsim's loader built, so agreement between them cannot catch a loader misreading
they share. Every expectation here instead comes from a hand-written ODE system
integrated with scipy, a closed form, a closed-form Jacobian (itself checked by
central differences of the hand-written RHS), or BioNetGen's own ``run_network``.

Some tests here fail on the commit they review; each says why in its docstring.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import os
import subprocess
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

pytest.importorskip("jax")
import bngsim._jax_rhs as jr  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jr.jax_available()  # float64

REPO = Path(__file__).resolve().parents[2]

# ── A model with one of everything a .net model has ─────────────────────────
#
# synthesis (null reactant), an elementary rate that is a DERIVED parameter,
# a Functional rate over an observable, a $-fixed species with a parameter-valued
# initial condition, a homodimerization with a stat factor, a Michaelis-Menten
# reaction, and an initial condition written as an expression.
SPREAD = """\
begin parameters
    1 kbase  0.4
    2 kf     2*kbase
    3 ksyn   1.5
    4 kdeg   0.8
    5 K      3.0
    6 X0     2.0
    7 kx     0.25
    8 kcat   1.2
    9 Km     2.0
end parameters
begin species
    1 A()   0
    2 B()   2*X0
    3 $X()  X0
    4 C()   0
    5 S()   10
    6 E()   1.5
    7 P()   0
end species
begin reactions
    1 0 1 ksyn #_R1
    2 1 2 kf #_R2
    3 2 0 fdeg #_R3
    4 3 4 kx #_R4
    5 1,1 4 0.5*kf #_R5
    6 5,6 7,6 MM kcat Km #_R6
end reactions
begin groups
    1 Btot 2
end groups
begin functions
    1 fdeg() kdeg/(K+Btot)
end functions
"""

DEFAULTS = dict(kbase=0.4, ksyn=1.5, kdeg=0.8, K=3.0, X0=2.0, kx=0.25, kcat=1.2, Km=2.0)


def _complex(S, E, Km):
    """tQSSA enzyme-substrate complex: the smaller root of C^2 - (S+E+Km) C + S E,
    in its cancellation-free form. BioNetGen's MM rate is kcat*C."""
    T = S + E + Km
    return 2.0 * S * E / (T + math.sqrt(T * T - 4.0 * S * E))


def _spread_rhs(p):
    kf = 2.0 * p["kbase"]

    def f(t, y):
        A, B, X, C, S, E, P = y
        mm = p["kcat"] * _complex(S, E, p["Km"])
        dimer = 0.5 * kf * A * A
        return [
            p["ksyn"] - kf * A - 2.0 * dimer,
            kf * A - p["kdeg"] / (p["K"] + B) * B,
            0.0,
            p["kx"] * X + dimer,
            -mm,
            0.0,
            mm,
        ]

    return f


def _spread_y0(p):
    return [0.0, 2.0 * p["X0"], p["X0"], 0.0, 10.0, 1.5, 0.0]


def _spread_jac(p, y):
    """Closed-form Jacobian of the hand-written system."""
    kf = 2.0 * p["kbase"]
    A, B, X, C, S, E, P = y
    T = S + E + p["Km"]
    D = math.sqrt(T * T - 4.0 * S * E)
    dC_dS = 0.5 * (1.0 - (T - 2.0 * E) / D)
    dC_dE = 0.5 * (1.0 - (T - 2.0 * S) / D)
    J = np.zeros((7, 7))
    J[0, 0] = -kf - 2.0 * kf * A
    J[1, 0] = kf
    J[1, 1] = -p["kdeg"] * p["K"] / (p["K"] + B) ** 2
    J[3, 0] = kf * A
    J[3, 2] = p["kx"]
    J[4, 4] = -p["kcat"] * dC_dS
    J[4, 5] = -p["kcat"] * dC_dE
    J[6, 4] = p["kcat"] * dC_dS
    J[6, 5] = p["kcat"] * dC_dE
    return J


def _scipy(f, y0, t_end, n, **kw):
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp

    ts = np.linspace(0.0, t_end, n)
    opts = dict(method="Radau", rtol=1e-11, atol=1e-13)
    opts.update(kw)
    sol = solve_ivp(f, (0.0, t_end), y0, t_eval=ts, **opts)
    assert sol.success, sol.message
    return ts, sol.y.T


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def _load(path):
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(str(path))


def _params(m):
    core = m._core
    return np.array([core.get_param(n) for n in core.param_names], dtype=np.float64)


def _diffrax():
    pytest.importorskip("diffrax")
    from bngsim._diffrax_solver import run_diffrax

    return run_diffrax


def _engine_run(path, t_end, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(_load(path), method="ode").run(
            t_span=(0.0, t_end), n_points=n, rtol=1e-11, atol=1e-13
        )
    return np.asarray(r.species)


RUN = dict(rtol=1e-10, atol=1e-12)


# ── run_diffrax against hand-written ODEs ───────────────────────────────────


class TestRunDiffraxAgainstHandWrittenOdes:
    def test_defaults(self, tmp_path):
        run_diffrax = _diffrax()
        net = _write(tmp_path, "spread.net", SPREAD)
        r = run_diffrax(_load(net), t_end=5.0, n_points=11, **RUN)
        ts, ref = _scipy(_spread_rhs(DEFAULTS), _spread_y0(DEFAULTS), 5.0, 11)
        np.testing.assert_allclose(r["time"], ts)
        np.testing.assert_allclose(r["species"], ref, rtol=1e-6, atol=1e-9)
        assert r["species_names"] == ["A()", "B()", "X()", "C()", "S()", "E()", "P()"]

    def test_overrides_reach_derived_parameters_and_initial_conditions(self, tmp_path):
        """kbase moves kf; X0 moves B(0) = 2*X0 and the fixed X(0) = X0; Km moves
        the MM law. The caller's model is left as it was."""
        run_diffrax = _diffrax()
        net = _write(tmp_path, "spread.net", SPREAD)
        m = _load(net)
        over = {"kbase": 0.1, "X0": 3.0, "Km": 0.5}
        r = run_diffrax(m, over, t_end=5.0, n_points=11, **RUN)
        p = {**DEFAULTS, **over}
        _, ref = _scipy(_spread_rhs(p), _spread_y0(p), 5.0, 11)
        np.testing.assert_allclose(r["species"], ref, rtol=1e-6, atol=1e-9)
        assert m.get_param("kbase") == 0.4 and m.get_param("kf") == 0.8

    def test_the_old_contract_a_full_consistent_dict_and_a_path(self, tmp_path):
        """``run_diffrax(net_path, every_parameter)`` as callers wrote it before
        step 4, with the derived value supplied consistently, still gives the
        answer."""
        run_diffrax = _diffrax()
        net = _write(tmp_path, "spread.net", SPREAD)
        p = {**DEFAULTS, "kbase": 0.1, "X0": 3.0}
        full = {**p, "kf": 2.0 * p["kbase"]}
        r = run_diffrax(str(net), full, t_end=5.0, n_points=11, **RUN)
        _, ref = _scipy(_spread_rhs(p), _spread_y0(p), 5.0, 11)
        np.testing.assert_allclose(r["species"], ref, rtol=1e-6, atol=1e-9)

    def test_param_dict_key_order_does_not_change_the_answer(self, tmp_path):
        """FAILS on afc0551. The same mapping, two key orders, two trajectories.

        An old-contract caller passes every parameter, derived ones included, and
        a stale derived value is the common case (the caller changed a primary).
        ``set_param`` pins a derived parameter only when the written value differs
        from its expression's value *at that moment* (#188), so
        ``{"kbase": 0.1, "kf": 0.8}`` pins kf at 0.8 while
        ``{"kf": 0.8, "kbase": 0.1}`` writes kf unchanged (no pin) and then lets
        kbase move it to 0.2. A dict built in sorted or any non-declaration order
        silently gets the other answer.
        """
        run_diffrax = _diffrax()
        net = _write(tmp_path, "spread.net", SPREAD)
        a = run_diffrax(_load(net), {"kbase": 0.1, "kf": 0.8}, t_end=3.0, n_points=4, **RUN)
        b = run_diffrax(_load(net), {"kf": 0.8, "kbase": 0.1}, t_end=3.0, n_points=4, **RUN)
        np.testing.assert_allclose(a["species"], b["species"], rtol=1e-9, atol=1e-12)

    def test_a_time_dependent_rate_and_no_observables(self, tmp_path):
        """dA/dt = -k0 exp(-t/tau) A with an empty groups block:
        A = A0 exp(-k0 tau (1 - exp(-t/tau)))."""
        run_diffrax = _diffrax()
        net = _write(
            tmp_path,
            "td.net",
            """\
begin parameters
    1 k0  0.9
    2 tau 1.7
end parameters
begin species
    1 A() 4
    2 B() 0
end species
begin reactions
    1 1 2 f #_R1
end reactions
begin groups
end groups
begin functions
    1 f() k0*exp(-time()/tau)
end functions
""",
        )
        r = run_diffrax(_load(net), t_end=6.0, n_points=13, **RUN)
        t = r["time"]
        A = 4.0 * np.exp(-0.9 * 1.7 * (1.0 - np.exp(-t / 1.7)))
        np.testing.assert_allclose(r["species"][:, 0], A, rtol=1e-7)
        np.testing.assert_allclose(r["species"][:, 1], 4.0 - A, rtol=1e-7, atol=1e-10)


# ── The Jacobian against a closed form ──────────────────────────────────────


def _state():
    return np.array([0.7, 3.1, 2.0, 0.4, 6.5, 1.5, 2.2])


def test_the_closed_form_jacobian_is_right():
    """The oracle's own control: central differences of the hand-written RHS."""
    f = _spread_rhs(DEFAULTS)
    y = _state()
    J = _spread_jac(DEFAULTS, y)
    fd = np.zeros_like(J)
    for j in range(7):
        h = 1e-6 * max(1.0, abs(y[j]))
        e = np.zeros(7)
        e[j] = h
        fd[:, j] = (np.asarray(f(0.0, y + e)) - np.asarray(f(0.0, y - e))) / (2 * h)
    np.testing.assert_allclose(fd, J, rtol=1e-7, atol=1e-9)


def test_prepare_jax_jacobian_matches_the_closed_form(tmp_path):
    """What CVODE receives: the flat, column-major array ``prepare_jax_jacobian``
    returns, fed the parameter vector the Simulator builds."""
    m = _load(_write(tmp_path, "spread.net", SPREAD))
    evaluate, n = jr.prepare_jax_jacobian(m)
    assert n == 7
    y = _state()
    flat = np.asarray(evaluate(y, 1.3, _params(m)))
    J = flat.reshape(7, 7, order="F")
    np.testing.assert_allclose(J, _spread_jac(DEFAULTS, y), rtol=1e-12, atol=1e-14)


def test_net_path_naming_another_model_is_ignored(tmp_path):
    """Before step 4 ``jacobian="jax"`` differentiated whatever file ``net_path``
    named; now it is ignored (with a DeprecationWarning), so a stale or wrong path
    can no longer hand CVODE another model's Jacobian."""
    m = _load(_write(tmp_path, "spread.net", SPREAD))
    other = _write(
        tmp_path,
        "other.net",
        "begin parameters\n    1 k 9\nend parameters\n"
        "begin species\n    1 Q() 1\nend species\n"
        "begin reactions\n    1 1 0 k\nend reactions\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "2-80x slower"
        with pytest.warns(DeprecationWarning, match="net_path"):
            sim = bngsim.Simulator(m, method="ode", jacobian="jax", net_path=str(other))
    y = _state()
    J = np.asarray(sim._jax_jac_evaluator(y, 0.0, _params(m))).reshape(7, 7, order="F")
    np.testing.assert_allclose(J, _spread_jac(DEFAULTS, y), rtol=1e-12, atol=1e-14)


# ── BioNetGen's own integrator ──────────────────────────────────────────────


def _run_network():
    from bngsim._bngpath import resolve_bng

    res = resolve_bng()
    rn = (res.root / "bin" / "run_network") if res.root is not None else None
    if rn is None or not rn.is_file():
        pytest.skip(
            "BNG2.pl/run_network not available: " + (res.why_not() or "no bin/run_network")
        )
    return rn


def test_run_diffrax_matches_bionetgen_run_network(tmp_path):
    """run_network reads the .net text itself; nothing of bngsim's is involved."""
    run_diffrax = _diffrax()
    rn = _run_network()
    net = _write(tmp_path, "spread.net", SPREAD)
    t_end, n = 5.0, 10
    subprocess.run(
        [str(rn), "-o", str(tmp_path / "rn"), "-p", "cvode", "-a", "1e-13", "-r", "1e-11",
         "--cdat", "1", "--fdat", "0", "-g", str(net), str(net), str(t_end / n), str(n)],
        check=True, capture_output=True, cwd=tmp_path, timeout=120,
    )  # fmt: skip
    cdat = np.loadtxt(tmp_path / "rn.cdat", ndmin=2)
    r = run_diffrax(_load(net), t_end=t_end, n_points=n + 1, **RUN)
    np.testing.assert_allclose(r["time"], cdat[:, 0], atol=1e-12)
    np.testing.assert_allclose(r["species"], cdat[:, 1:], rtol=1e-6, atol=1e-9)


# ── check_rhs_against_engine ────────────────────────────────────────────────


def test_the_check_sees_a_term_whose_species_starts_at_zero(tmp_path, monkeypatch):
    """Positive control for the check's choice of state: an error on a term
    proportional to A, which is 0 at t=0, is still refused."""
    m = _load(_write(tmp_path, "spread.net", SPREAD))
    real = jr.generate_jax_rhs

    def off_by_a_term(model):
        rhs = real(model)

        def wrong(y, t, p):
            return rhs(y, t, p).at[1].add(0.3 * y[0])

        wrong.n_species, wrong.n_params = rhs.n_species, rhs.n_params
        return wrong

    monkeypatch.setattr(jr, "generate_jax_rhs", off_by_a_term)
    with pytest.raises(ValueError, match="disagrees with the engine"):
        jr.prepare_jax_jacobian(m)


def _refuses_or_matches(run, expected, **tol):
    """``run()`` must raise ValueError (refused) or return ``expected``."""
    try:
        got = run()
    except ValueError:
        return
    np.testing.assert_allclose(got, expected, **tol)


MODULO_IN_TIME = """\
begin parameters
    1 k 1.0
end parameters
begin species
    1 A() 5
    2 B() 0
end species
begin reactions
    1 1 2 f #_R1
end reactions
begin functions
    1 f() k*(1 + (5 - time()) % 3)
end functions
"""


def test_a_mismatch_that_starts_after_t0_is_refused_or_absent(tmp_path):
    """FAILS on afc0551. The check compares the two RHS at t = 0 only.

    ``%`` is ExprTk's fmod in the engine and Python's floor-mod in the JAX
    translation (a pre-existing translator defect). They agree while
    ``5 - time()`` is non-negative, so the check passes at t = 0 -- and
    ``run_diffrax`` then integrates a different rate law from t = 5 on. The oracle
    is the closed form under fmod: A = 5 exp(-int_0^t (1 + fmod(5 - s, 3)) ds).
    """
    quad = pytest.importorskip("scipy.integrate").quad

    run_diffrax = _diffrax()
    net = _write(tmp_path, "mod.net", MODULO_IN_TIME)
    ts = np.linspace(0.0, 9.0, 10)

    def A(t):
        pts = [x for x in (2.0, 5.0, 8.0) if x < t]
        return 5.0 * math.exp(
            -quad(lambda s: 1.0 + math.fmod(5.0 - s, 3.0), 0.0, t, points=pts or None)[0]
        )

    expected = np.array([A(t) for t in ts])
    _refuses_or_matches(
        lambda: run_diffrax(_load(net), t_end=9.0, n_points=10, **RUN)["species"][:, 0],
        expected,
        rtol=1e-5,
        atol=1e-9,
    )


FAST_AND_SLOW = """\
begin parameters
    1 ks 0.01
    2 k0 1
    3 kf 1e9
end parameters
begin species
    1 S() 1
    2 F() 1
    3 G() 0
end species
begin reactions
    1 1 0 fs #_R1
    2 2 3 kf #_R2
end reactions
begin functions
    1 fs() ks*(2 + (k0 - 2) % 3)
end functions
"""


def test_a_fast_reaction_does_not_hide_a_slow_species_error(tmp_path):
    """FAILS on afc0551. The check's tolerance is 1e-8 * max|f| over ALL species.

    Here fs is ks*1 under the engine's fmod and ks*4 under the JAX floor-mod --
    at t = 0, so the check does see it -- but F -> G at 1e9 makes max|f| ~ 1e9 and
    the tolerance ~ 10, so an error of 3*ks*S (~0.04 at the check's state) on dS/dt
    passes. The same defect in the model without the fast reaction IS refused.
    ``run_diffrax`` then returns S(100) = e^-4 = 0.018 where the model says
    e^-1 = 0.37.
    """
    run_diffrax = _diffrax()
    net = _write(tmp_path, "fs.net", FAST_AND_SLOW)
    t = np.array([0.0, 50.0, 100.0])

    def solve():
        r = run_diffrax(_load(net), t_end=100.0, n_points=3, rtol=1e-8, atol=1e-10)
        return r["species"][:, 0]

    _refuses_or_matches(solve, np.exp(-0.01 * t), rtol=1e-5)


MM_ON_A_FUNCTION_SLOT = """\
begin parameters
    1 kc 0
    2 Km 3
end parameters
begin species
    1 S() 20
    2 E() 2
    3 P() 0
end species
begin reactions
    1 1,2 3,2 MM kc Km #_R1
end reactions
begin functions
    1 kc() 0.5*time()
end functions
"""


def test_mm_rate_constant_that_a_function_overwrites(tmp_path):
    """``kc`` is both a parameter row and a function. The engine's Michaelis-
    Menten law read the slot the function rewrites every step, and the JAX
    right-hand side read ``params[kc]``, a snapshot of it: two readings of one
    model (BioNetGen's run_network takes the parameter row too). The JAX path
    refused the shape; the model is refused where it is built now, for every
    engine that runs a built model (issue #931).
    """
    net = _write(tmp_path, "mmfs.net", MM_ON_A_FUNCTION_SLOT)
    with pytest.raises(bngsim.ModelError, match=r"MichaelisMenten.*issue #931"):
        _load(net)


def test_matching_infinities_at_the_test_state_are_not_a_mismatch(tmp_path):
    """FAILS on afc0551. The check's state adds 5-15% of the largest initial
    concentration to every species, so an inert Z (0 on the whole trajectory)
    sits at >= 50 there and exp(100*Z) overflows in both RHS alike. inf - inf is
    NaN, the check calls it a disagreement ("JAX -inf, engine -inf"), and a model
    both RHS evaluate identically is refused. (Corpus instance:
    m_740761d15525e4520a844c81975be6dc.net.) Closed form: A = 1000 exp(-0.3 t).
    """
    run_diffrax = _diffrax()
    net = _write(
        tmp_path,
        "inf.net",
        """\
begin parameters
    1 k 0.3
end parameters
begin species
    1 A() 1000
    2 B() 0
    3 Z() 0
end species
begin reactions
    1 1 2 f #_R1
end reactions
begin groups
    1 Ztot 3
end groups
begin functions
    1 f() k*exp(100*Ztot)
end functions
""",
    )
    r = run_diffrax(_load(net), t_end=4.0, n_points=5, **RUN)
    np.testing.assert_allclose(r["species"][:, 0], 1000.0 * np.exp(-0.3 * r["time"]), rtol=1e-7)


# ── _refuse_unsupported / the translator ────────────────────────────────────


def test_a_parameter_is_not_shadowed_by_a_same_named_species(tmp_path):
    """FAILS on afc0551. Step 4 added a species-name pass to the JAX translator
    (for SBML rules, which are refused anyway) and runs it BEFORE the parameter
    pass. In a .net model neither the engine nor run_network binds a species name
    in an expression -- both read the parameter B = 7 here (A = 5 exp(-1.4 t)) --
    but the JAX RHS reads the species B, so the check refuses a model it could
    have handled."""
    run_diffrax = _diffrax()
    net = _write(
        tmp_path,
        "spb.net",
        """\
begin parameters
    1 k 0.2
    2 B 7
end parameters
begin species
    1 A 5
    2 B 1
end species
begin reactions
    1 1 2 f #_R1
end reactions
begin groups
    1 Atot 1
end groups
begin functions
    1 f() k*B
end functions
""",
    )
    r = run_diffrax(_load(net), t_end=2.0, n_points=3, **RUN)
    np.testing.assert_allclose(r["species"][:, 0], 5.0 * np.exp(-1.4 * r["time"]), rtol=1e-7)


def test_an_integer_literal_power_beyond_int64(tmp_path):
    """lanl/bngsim#837: an integer literal power (1500^6) was evaluated as a Python
    int, which overflows int64 when it meets a JAX array, so OverflowError escaped
    Simulator(jacobian='jax') and run_diffrax (corpus model
    m_ae32b88d25287652e5f9b663542644eb.net). The translator now spells integer
    literals as floats, as the engine evaluates them."""
    run_diffrax = _diffrax()
    net = _write(
        tmp_path,
        "big.net",
        """\
begin parameters
    1 k 0.4
end parameters
begin species
    1 A() 3
    2 B() 0
end species
begin reactions
    1 1 2 f #_R1
end reactions
begin groups
    1 Atot 1
end groups
begin functions
    1 f() k*((1500^6)+Atot)/(1500^6)
end functions
""",
    )
    r = run_diffrax(_load(net), t_end=2.0, n_points=3, **RUN)
    np.testing.assert_allclose(r["species"][:, 0], 3.0 * np.exp(-0.4 * r["time"]), rtol=1e-7)


def test_a_saturated_steep_hill_term_has_a_finite_jacobian(tmp_path):
    net = _write(
        tmp_path,
        "steep.net",
        """\
begin parameters
    1 k  2.0
    2 K  1.0
    3 n  999.898
    4 ks 1.0
end parameters
begin species
    1 A() 5
    2 B() 0
end species
begin reactions
    1 0 2 f #_R1
    2 1 0 ks #_R2
end reactions
begin groups
    1 Atot 1
end groups
begin functions
    1 f() k/(1+(Atot/K)^n)
end functions
""",
    )
    m = _load(net)
    y = np.array([5.0, 1.0])
    J = np.asarray(jr.generate_jax_jacobian(m)(jnp.asarray(y), 0.0, jnp.asarray(_params(m))))
    # d/dA k/(1+x^n) = -k n x^(n-1) / (K (1+x^n)^2), x = A/K = 5: exp(~ -1000 ln 5) = 0.
    log_mag = math.log(2.0 * 999.898) + 998.898 * math.log(5.0) - 2.0 * 999.898 * math.log(5.0)
    expected = np.array([[-1.0, 0.0], [-math.exp(log_mag), 0.0]])
    np.testing.assert_allclose(J, expected, atol=1e-300)


# ── The removed parser has no importers left ────────────────────────────────


def test_nothing_in_the_tree_imports_the_removed_net_parser():
    """FAILS on afc0551: harness/comparison/bench_ode_scipy_diffrax.py still does
    ``from bngsim._codegen import _parse_net_file`` in three engines, each of which
    now dies with ImportError."""
    gone = {"_parse_net_file", "_classify_rate_law", "_parse_species_line", "generate_rhs_c"}
    offenders = []
    for top in ("python", "harness", "benchmarks", "parity_checks", "scripts", "docs"):
        root = REPO / top
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if ".venv" in path.parts or "_deps" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bngsim"):
                    hit = gone & {a.name for a in node.names}
                    if hit:
                        offenders.append(
                            f"{os.path.relpath(path, REPO)}:{node.lineno} {sorted(hit)}"
                        )
    assert not offenders, offenders
