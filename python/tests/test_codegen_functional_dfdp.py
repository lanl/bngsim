"""GH #66 — analytic ``∂f/∂p`` for Functional rate laws, emitted but gated off.

``generate_sens_from_model`` returned ``None`` on the first non-Elementary
reaction, so a single Functional rate law put a whole model on CVODES' internal
difference quotient. This stage derives the missing half,

    ∂f_i/∂p = Σ_r  stat_r · netstoich_ir · (∂func_r/∂p) · ∏R_r

behind a keyword, and proves it against an oracle. (GH #67 has since supplied the
other half — the Functional ``J·yS`` — and turned the keyword on for production
callers; ``test_codegen_functional_sens_rhs.py`` covers that. This file stays the
∂f/∂p half's test.) Two things are being tested, and they pull in opposite
directions:

* **Nothing moved for Elementary models.** They emit byte-identical C whether or
  not the keyword is set — a Functional model is the only input the new code can
  even reach.
* **The derivative is right.** The oracle is a central finite difference of the
  *emitted* ``bngsim_codegen_rhs`` with respect to each parameter, compared
  against the emitted ``∂f/∂p``, both called through ctypes. No integrator, no
  tolerance tuning: the two sides read the same ``p[]`` array and the same
  ``obs[]``/``func[]`` intermediates, so agreement is expected to FD precision.
  A parameter perturbation goes through ``set_param`` (which re-derives every
  ConstantExpression parameter) precisely so the #15/#41 chain rule is under
  test rather than assumed.

The third claim, from #56 by way of #55: an undifferentiable Functional law must
decline the **whole model** — ``CVodeSensInit1`` takes one callback for every
sensitivity column, so there is no per-reaction fallback — and must say what
blocked it rather than emitting ``∂func/∂p = 0``, which reads downstream as a
converged gradient of exactly zero.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging

import bngsim
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")

EPS = 2.220446049250313e-16


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# Written as ``.net`` text because that is the shape the Functional population
# actually arrives in: BNG2.pl emits every ``if()``-gated or saturating rate law
# as a function, and stores a compound rate constant as a ``# ConstantExpression``
# parameter.

# The canonical case, and every moving part at once: a Functional law reading an
# observable (``I``), a *derived* rate constant (``beta = 1/S0``, so ∂f/∂S0 only
# exists through the chain rule), and an Elementary reaction sharing the switch.
SIR = """\
begin parameters
    1 S0     2e7  # Constant
    2 I0     1  # Constant
    3 beta   1/S0  # ConstantExpression
    4 gamma  1/7  # Constant
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

# Saturation + a Hill exponent + ``time()``: ∂/∂Km and ∂/∂n are the derivatives
# most likely to come back un-emittable, and the explicit time dependence pins
# that the ``_TIME_SYM`` placeholder round-trips to ``t``.
HILL = """\
begin parameters
    1 kmax   3.5  # Constant
    2 Km     4.0  # Constant
    3 n      2.0  # Constant
    4 kdeg   0.2  # Constant
    5 ramp   0.05  # Constant
end parameters
begin functions
    1 vsat() kmax*(1 + ramp*time())*Atot^n/(Km^n + Atot^n)
end functions
begin species
    1 A() 6.0
    2 B() 1.0
end species
begin reactions
    1 1 2 vsat #_R1
    2 2 1 kdeg #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""

# A nested user function over two observables, so ``_inline_functions`` has to
# flatten before differentiating and the derivative mixes two ``obs[]`` slots.
NESTED = """\
begin parameters
    1 ka     0.7  # Constant
    2 kb     1.3  # Constant
    3 c0     2.0  # Constant
end parameters
begin functions
    1 mix() ka*Atot + kb*Btot
    2 drive() c0*mix()/(1 + Atot)
end functions
begin species
    1 A() 3.0
    2 B() 5.0
    3 C() 0.0
end species
begin reactions
    1 1 3 drive #_R1
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""

ELEMENTARY = """\
begin parameters
    1 k1     0.3  # Constant
    2 scale  2.0  # Constant
    3 k2     k1*scale  # ConstantExpression
end parameters
begin functions
    1 report() Atot*2
end functions
begin species
    1 A() 10.0
    2 B() 0.0
end species
begin reactions
    1 1 2 k1 #_R1
    2 2 1 k2 #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""


# The BNGL whitespace Michaelis–Menten form. Out of scope for #66, supported as
# of #55's MM stage — kept here because two of this file's decline tests used to
# ride on it.
MM_NET = """\
begin parameters
    1 kcat  1  # Constant
    2 Km    1  # Constant
end parameters
begin species
    1 S() 100
    2 E() 100
    3 P() 0
end species
begin reactions
    1 1,2 3,2 MM kcat Km #_R1
end reactions
begin groups
    1 St                   1
end groups
"""


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _with_rate_law(body: str) -> str:
    """SIR with its rate law swapped — the vehicle for the decline cases."""
    return SIR.replace("    1 betaI() beta*I\n", f"    1 betaI() {body}\n")


# The per-model context the differentiation reads, spelled out so a rate law that
# no loadable ``.net`` can carry — a table-function call, an SBML ``rateOf``
# accessor — is still testable. Mirrors what _functional_dfdp_terms assembles:
# ``Itot``/``Atot`` are observables, ``beta``/``kf`` parameters, ``kd`` derived.
def _scope(**over):
    fields = {
        "func_map": {},
        "c_ref": {"Itot": "obs[0]", "Atot": "obs[1]", "beta": "p[0]", "kf": "p[1]", "kd": "p[2]"},
        "param_of_alias": {"beta": "beta", "kf": "kf", "kd": "kd"},
        "param_idx_by_name": {"beta": 0, "kf": 1, "kd": 2},
        "primary_param_names": {"beta", "kf"},
        "derived_exprs": {"kd": "beta*kf"},
    }
    fields.update(over)
    return cg._FunctionalDfdpScope(**fields)


# ─── the keyword ───────────────────────────────────────────────────────────


class TestTheKeyword:
    """``functional=`` is what reaches the ∂f/∂p derived below. GH #67 has since
    turned it on for production callers (see
    ``test_codegen_functional_sens_rhs.py``), so what is pinned here is the
    keyword's own contract, not the old no-behavior-change claim."""

    def test_a_functional_model_declines_without_it(self, tmp_path):
        assert cg.generate_sens_from_model(_model(tmp_path, SIR)) is None

    def test_the_keyword_is_what_opens_it(self, tmp_path):
        src = cg.generate_sens_from_model(_model(tmp_path, SIR), functional=True)
        assert src is not None and "bngsim_dfdp" in src

    def test_elementary_emission_is_byte_identical(self, tmp_path):
        """An Elementary model reaches none of the new code, so opening the gate
        must not move a single byte — including its #15 derived-parameter chain."""
        model = _model(tmp_path, ELEMENTARY)
        shut = cg.generate_sens_from_model(model)
        open_ = cg.generate_sens_from_model(model, functional=True)
        assert shut is not None
        assert shut == open_
        assert "GH #66" not in shut


# ─── the derivative itself ─────────────────────────────────────────────────


class TestEmittedTerms:
    def test_sir_switch_carries_both_the_direct_and_the_chain_column(self, tmp_path):
        """``beta`` gets its own column (∂f/∂beta = -I·S) and ``S0`` gets one
        only through ``beta = 1/S0`` — the #15/#41 chain rule reaching a
        Functional law for the first time."""
        model = _model(tmp_path, SIR)
        core = model._core
        data = core.codegen_data()
        names = [p["name"] for p in data["parameters"]]
        terms, decline = cg._functional_dfdp_terms(core, data)
        assert decline is None
        by_param = {names[k]: c for k, c in terms[0]}
        assert set(by_param) == {"beta", "S0"}
        assert by_param["beta"] == "obs[1]"
        assert "p[0]" in by_param["S0"]  # ∂beta/∂S0 folded in

    def test_a_parameter_free_rate_law_is_a_success_not_a_decline(self, tmp_path):
        """An empty term list means a genuinely zero ∂f/∂p column. Conflating it
        with a failure is exactly the ambiguity #56 was filed about."""
        model = _model(tmp_path, _with_rate_law("I*I"))
        core = model._core
        terms, decline = cg._functional_dfdp_terms(core, core.codegen_data())
        assert decline is None
        assert terms == {0: []}

    def test_a_derived_parameter_gets_both_its_own_column_and_its_primaries(self):
        """``kd = beta*kf`` read directly by a rate law: ∂/∂kd is the direct
        partial (the Elementary path's convention for a ``_rateLaw{N}``) and
        ∂/∂beta, ∂/∂kf come only from the chain rule."""
        terms, decline = cg._functional_rate_law_partials("kd*Itot", _scope())
        assert decline is None
        assert {k for k, _c in terms} == {0, 1, 2}

    def test_a_function_bound_parameter_is_not_a_differentiation_variable(self, tmp_path):
        """``betaI`` is both the rate-law function and a synthetic parameter
        holding no independent value; differentiating w.r.t. it would emit a
        column the RHS cannot move."""
        model = _model(tmp_path, SIR)
        core = model._core
        data = core.codegen_data()
        names = [p["name"] for p in data["parameters"]]
        terms, _ = cg._functional_dfdp_terms(core, data)
        assert "betaI" in names
        assert names.index("betaI") not in {k for k, _c in terms[0]}


# ─── decline loudly ────────────────────────────────────────────────────────


class TestDeclinesLoudly:
    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            # sympy differentiates every one of these happily, dropping the
            # boundary jump — a token pre-scan is the only reliable rejection,
            # which is why unsupported_expr_construct is reused rather than
            # re-spelled. GH #68 lifted the conditional class *conditionally*,
            # issue #150 lifted more of it (`if(I > 3, ...)` is admitted now that
            # its crossing is rooted and jumped) and #381 lifted the equality
            # spelling of that same surface. What still declines is a crossing NO
            # machinery can bracket — a clock threshold quadratic in the clock,
            # so no stop time can be solved from it, over no live state to root
            # on; or a comparison with no `if()` head to locate a threshold in
            # (see test_codegen_switch_condition_sens.py).
            ("if(time()*time() > gamma, beta, 0)*I", "neither a recognized clock threshold"),
            ("beta*(I > 1)", "is not inside an if() condition"),
            ("beta*abs(I)", "abs()"),
            ("beta*max(I, 1)", "max()"),
            ("beta*min(I, 1)", "min()"),
            ("beta*floor(I)", "floor()"),
            ("beta*ceil(I)", "ceil()"),
        ],
    )
    def test_reason_names_what_blocked_it(self, tmp_path, body, fragment):
        model = _model(tmp_path, _with_rate_law(body))
        core = model._core
        terms, decline = cg._functional_dfdp_terms(core, core.codegen_data())
        assert terms == {}
        assert decline is not None and fragment in decline
        assert cg.generate_sens_from_model(model, functional=True) is None

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            # A table-function call parses as an applied *undefined* function, so
            # it has no free symbol to reject — the call-head scan is what stops
            # it before sympy renders Subs(Derivative(...)).
            ("beta*tfun_drive(Itot)", "calls unsupported function(s): tfun_drive"),
            ("beta*helper(Itot)", "calls unsupported function(s): helper"),
            # A rateOf accessor is an evaluator variable, not a model parameter,
            # so it is caught as a free symbol nothing can resolve. Differentiating
            # it as if it were a constant is precisely the silent-zero failure.
            ("beta*rate_of__A", "unrecognized symbol(s): rate_of__A"),
            ("beta*mystery", "unrecognized symbol(s): mystery"),
        ],
    )
    def test_constructs_no_loadable_net_can_carry_are_refused_too(self, body, fragment):
        terms, decline = cg._functional_rate_law_partials(body, _scope())
        assert terms is None
        assert fragment in decline

    def test_a_cycle_in_the_function_graph_is_refused(self):
        scope = _scope(func_map={"a": "b*Itot", "b": "a + 1"})
        terms, decline = cg._functional_rate_law_partials("beta*a", scope)
        assert terms is None and "cycle" in decline

    def test_the_decline_is_warned_not_silent(self, tmp_path, caplog):
        model = _model(tmp_path, _with_rate_law("if(time()*time() > gamma, beta, 0)*I"))
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert cg.generate_sens_from_model(model, functional=True) is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "neither a recognized clock threshold" in m and "betaI" in m for m in warnings
        ), warnings

    def test_every_decline_path_warns_not_just_the_differentiation_ones(
        self, tmp_path, caplog, monkeypatch
    ):
        """A decline that returns a reason and drops it on the floor is still a
        silent decline. The paths that bypass the differentiation entirely need
        their own coverage.

        Michaelis–Menten used to be this test's vehicle (it declined on
        rate-law type); it is supported as of #55's MM stage, so the vehicle is
        now the decline *that* stage introduced — an MM model whose analytical
        Jacobian plan is unavailable, so there is nothing to build ``J·yS``
        from. Both are non-differentiation declines reached before any rate law
        is touched."""
        mm = _model(tmp_path, MM_NET)
        core = mm._core
        monkeypatch.setattr(
            type(core),
            "codegen_jacobian_plan",
            lambda self: {"available": False},
            raising=False,
        )
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert cg.generate_sens_from_model(mm, functional=True) is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("no analytical Jacobian plan" in m for m in warnings), warnings

    def test_an_unreachable_table_function_value_declines_out_loud_too(self, tmp_path, caplog):
        """The one decline the per-reaction loop cannot see coming: every rate law
        differentiates, but ``bngsim_dfdp`` is written in ``func[]`` and one of the
        model's functions is a table lookup, which dispatches through
        ``data->tfun_eval`` — a field ``CodegenSensUserData`` does not have (#65).
        The right answer is still to decline; the wrong one is to do it quietly."""
        (tmp_path / "drive.tfun").write_text("# time drive\n0 0\n1 1\n2 2\n")
        text = SIR.replace(
            "    1 betaI() beta*I\n",
            "    1 betaI() beta*I\n    2 drive()  tfun('drive.tfun')\n",
        )
        model = _model(tmp_path, text)
        core = model._core
        # The rate law itself is fine — the decline is the emitter's, not the
        # differentiation's, which is exactly why it needed its own warning.
        terms, decline = cg._functional_dfdp_terms(core, core.codegen_data())
        assert decline is None and terms
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert cg.generate_sens_from_model(model, functional=True) is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("table function or rateOf" in m for m in warnings), warnings

    def test_a_construct_hidden_inside_a_nested_function_is_still_caught(self):
        """The scan runs on the *inlined* text; a rate law that looks smooth but
        calls a function containing ``abs()`` must not slip through."""
        scope = _scope(func_map={"inner": "abs(Itot - 1)"})
        terms, decline = cg._functional_rate_law_partials("beta*inner", scope)
        assert terms is None and "abs()" in decline

    def test_one_bad_law_declines_the_whole_model(self, tmp_path):
        """``CVodeSensInit1`` is all-or-nothing, so a second, perfectly
        differentiable Functional reaction must not survive on its own."""
        text = SIR.replace(
            "    1 betaI() beta*I\n",
            "    1 betaI() beta*I\n    2 bad() gamma*abs(R)\n",
        ).replace("    2 2 3 gamma #_R2\n", "    2 2 3 gamma #_R2\n    3 3 1 bad #_R3\n")
        model = _model(tmp_path, text)
        core = model._core
        terms, decline = cg._functional_dfdp_terms(core, core.codegen_data())
        assert terms == {} and "abs()" in decline
        assert cg.generate_sens_from_model(model, functional=True) is None

    def test_michaelis_menten_is_no_longer_out_of_scope(self, tmp_path):
        """#66 declined MM on rate-law type; #55's MM stage supplies its closed
        form. Kept as a marker of the reversal — the coverage lives in
        ``test_codegen_mm_sens.py``."""
        model = _model(tmp_path, MM_NET)
        core = model._core
        terms, decline = cg._functional_dfdp_terms(core, core.codegen_data())
        # The Functional pass has nothing to say about an MM reaction either way:
        # no Functional rate laws, so no terms and no decline.
        assert terms == {} and decline is None
        assert cg.generate_sens_from_model(model, functional=True) is not None


# ─── the finite-difference oracle ──────────────────────────────────────────


class _CodegenUserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("tfun_ctx", ctypes.c_void_p),
        ("tfun_eval", ctypes.c_void_p),
    ]


class _SensUserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("plist", ctypes.POINTER(ctypes.c_int)),
        ("n_sens", ctypes.c_int),
    ]


class _Compiled:
    """RHS + sensitivity RHS in one .so, both reachable through ctypes."""

    def __init__(self, model, tmp_path, monkeypatch):
        core = model._core
        self.data = core.codegen_data()
        self.n_sp = len(self.data["species"])
        self.n_par = len(self.data["parameters"])
        sens = cg.generate_sens_from_model(model, functional=True)
        assert sens is not None
        src = cg.generate_rhs_from_model(model) + "\n" + sens
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
        so = cg.compile_rhs(src, hashlib.sha256(src.encode()).hexdigest()[:16])
        self.lib = ctypes.CDLL(str(so))
        dp = ctypes.POINTER(ctypes.c_double)
        self.lib.bngsim_codegen_rhs.restype = ctypes.c_int
        self.lib.bngsim_codegen_rhs.argtypes = [ctypes.c_double, dp, dp, ctypes.c_void_p]
        self.lib.bngsim_codegen_sens_rhs.restype = ctypes.c_int
        self.lib.bngsim_codegen_sens_rhs.argtypes = [
            ctypes.c_int,
            ctypes.c_double,
            dp,
            dp,
            ctypes.c_int,
            dp,
            dp,
            ctypes.c_void_p,
            dp,
            dp,
        ]

    def f(self, t, y, p):
        pbuf = (ctypes.c_double * self.n_par)(*p)
        ud = _CodegenUserData(
            param_values=ctypes.cast(pbuf, ctypes.POINTER(ctypes.c_double)),
            tfun_ctx=None,
            tfun_eval=None,
        )
        ydot = (ctypes.c_double * self.n_sp)()
        assert (
            self.lib.bngsim_codegen_rhs(
                float(t), (ctypes.c_double * self.n_sp)(*y), ydot, ctypes.byref(ud)
            )
            == 0
        )
        return list(ydot)

    def dfdp(self, t, y, p, iP):
        """An all-zero ``yS`` zeroes J·yS exactly, so ySdot comes back as the
        bare ∂f/∂p column — the same trick steady_state.cpp's eval_dfdp uses."""
        pbuf = (ctypes.c_double * self.n_par)(*p)
        ud = _SensUserData(param_values=pbuf, plist=(ctypes.c_int * 1)(int(iP)), n_sens=1)
        ySdot = (ctypes.c_double * self.n_sp)()
        scratch = [(ctypes.c_double * self.n_sp)() for _ in range(4)]
        assert (
            self.lib.bngsim_codegen_sens_rhs(
                1,
                float(t),
                (ctypes.c_double * self.n_sp)(*y),
                scratch[0],
                0,
                scratch[1],
                ySdot,
                ctypes.byref(ud),
                scratch[2],
                scratch[3],
            )
            == 0
        )
        return list(ySdot)


def _perturbed(core, data, k, rel):
    """``(p_plus, p_minus)`` for parameter ``k`` at relative step ``rel``.

    A *primary* parameter moves through ``set_param``, which re-derives every
    ConstantExpression parameter — the runtime semantics the emitted chain rule
    mirrors, and the only way the ∂f/∂S0 column is under test at all. A derived
    parameter moves in the raw vector alone, matching the direct partial the
    emitter writes for its own column.

    A **function's** slot (issue #227) takes that same raw-vector branch: it is a
    ``p[]`` entry the emitted ``f`` overwrites from the function's own expression
    before reading it, so its column stays under test — it must be zero, and the
    difference is what says so — while ``set_param`` refuses the write.
    """
    params = data["parameters"]
    names = [p["name"] for p in params]
    base = [float(p["value"]) for p in params]
    v = base[k]
    h = rel * abs(v) if v != 0.0 else rel
    if list(core.param_is_internal)[k] or not bool(params[k].get("is_const", True)):
        plus, minus = list(base), list(base)
        plus[k], minus[k] = v + h, v - h
        return plus, minus
    out = []
    for sign in (+1, -1):
        core.set_param(names[k], v + sign * h)
        out.append([float(core.get_param(n)) for n in names])
    core.set_param(names[k], v)
    return out[0], out[1]


# Central-difference cancellation noise falls as 1/h while truncation grows as
# h², so a component that agrees at ANY of three well-separated steps is a
# component the analytic derivative got right; only a genuine error survives all
# three. Without this the oracle reports its own noise as a codegen bug.
_STEPS = (1e-7, 1e-5, 1e-3)


def _assert_dfdp_matches_fd(model, tmp_path, monkeypatch, states, rtol=1e-6):
    comp = _Compiled(model, tmp_path, monkeypatch)
    core = model._core
    data = comp.data
    params = data["parameters"]
    base = [float(p["value"]) for p in params]
    worst, where = 0.0, None
    for k in range(len(params)):
        for t, y in states:
            an = comp.dfdp(t, y, base, k)
            best = None
            for rel in _STEPS:
                plus, minus = _perturbed(core, data, k, rel)
                step = plus[k] - minus[k]
                if step == 0.0:
                    continue
                fp, fm = comp.f(t, y, plus), comp.f(t, y, minus)
                fd = [(a - b) / step for a, b in zip(fp, fm, strict=True)]
                # A component of f is assembled from rate terms as large as the
                # biggest one in the vector, so the resolution of the difference
                # is set by ‖f‖∞, not by |f_i|.
                fscale = max(max(abs(v) for v in fp), max(abs(v) for v in fm), max(map(abs, an)))
                noise = EPS * fscale / abs(step / 2.0)
                ratios = [
                    abs(d - a) / (rtol * max(abs(a), abs(d)) + 8.0 * noise + 1e-300)
                    for d, a in zip(fd, an, strict=True)
                ]
                if best is None:
                    best = list(zip(ratios, fd, strict=True))
                else:
                    best = [
                        min(o, n) for o, n in zip(best, zip(ratios, fd, strict=True), strict=True)
                    ]
            if best is None:
                continue
            ratio, fd_i = max(best)
            if ratio > worst:
                worst = ratio
                where = (params[k]["name"], t, fd_i)
    assert worst <= 1.0, f"∂f/∂p disagrees with the finite difference at {where} (ratio {worst:g})"
    return worst


@requires_cc
class TestFiniteDifferenceOracle:
    def test_sir_observable_law_and_derived_chain(self, tmp_path, monkeypatch):
        states = [(0.0, [2e7, 1.0, 0.0]), (3.5, [1.1e7, 4.2e5, 8.7e6])]
        _assert_dfdp_matches_fd(_model(tmp_path, SIR), tmp_path, monkeypatch, states)

    def test_hill_saturation_exponent_and_time(self, tmp_path, monkeypatch):
        states = [(0.0, [6.0, 1.0]), (12.0, [0.4, 9.1]), (40.0, [3.3, 3.3])]
        _assert_dfdp_matches_fd(_model(tmp_path, HILL), tmp_path, monkeypatch, states)

    def test_nested_functions_over_two_observables(self, tmp_path, monkeypatch):
        states = [(0.0, [3.0, 5.0, 0.0]), (2.0, [0.7, 11.0, 4.0])]
        _assert_dfdp_matches_fd(_model(tmp_path, NESTED), tmp_path, monkeypatch, states)

    def test_elementary_model_is_covered_by_the_same_oracle(self, tmp_path, monkeypatch):
        """The oracle has to agree where the answer was already known, or it is
        measuring itself rather than the new derivative."""
        states = [(0.0, [10.0, 0.0]), (5.0, [2.5, 7.5])]
        _assert_dfdp_matches_fd(_model(tmp_path, ELEMENTARY), tmp_path, monkeypatch, states)
