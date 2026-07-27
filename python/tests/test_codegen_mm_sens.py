"""#55's Michaelis–Menten stage — the analytic sensitivity RHS for ``MM kcat Km``.

#66/#67/#68 covered Functional rate laws and left Michaelis–Menten declining on
rate-law type, on the argument that it was worth exactly one extra corpus model.
That is the wrong axis: ``MM(kcat, Km)`` is a first-class BNGL rate law, and a
modeller who writes one should not silently lose the analytic gradient.

Unlike the Functional path there is no symbolic differentiation here. The tQSSA
rate is closed form,

    δ = S − Km − E,   D = √(δ² + 4·Km·S),   sFree = ½(δ + D)
    rate = kcat·stat·E·sFree/(Km + sFree)

so both partials are written out:

    ∂rate/∂kcat = stat·E·sFree/(Km + sFree)                    ( = rate/kcat )
    ∂rate/∂Km   = kcat·stat·E·(∂sFree/∂Km·Km − sFree)/(Km + sFree)²
                  with ∂sFree/∂Km = ½(−1 + (2S − δ)/D)

Both were checked against ``sympy.diff`` — symbolically and over random parameter
points — before being written down, and ``test_the_partials_match_sympy`` keeps
that check in the suite rather than in a notebook someone has to trust.

Two things are worth knowing about this file's tolerances.

**The grouping is load-bearing.** An algebraically identical form of ∂/∂Km with
the ½ factors cancelled is 14 digits worse on log-spread parameters. What is
emitted mirrors the shipped Jacobian's own ∂rate/∂E grouping, and
``test_the_emitted_grouping_is_the_stable_one`` pins that.

**There is an accuracy floor, and it is not this stage's.** ``sFree = ½(δ + D)``
cancels catastrophically once ``|δ| ≫ √(4·Km·S)``, roughly two digits per decade.
That is shipped behaviour in the RHS and the analytical Jacobian; a model in that
regime has a wrong trajectory before it has a wrong gradient. So the oracles here
run in the well-conditioned regime, and the degenerate corner is asserted only to
the extent that ∂f/∂p is no worse than the rate it differentiates.
"""

from __future__ import annotations

import ctypes
import hashlib
import math

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# Parameters are deliberately *not* symmetric: kcat != Km, E != S, and a second
# Elementary reaction so the MM contribution has to coexist with the mass-action
# switch rather than being the only thing in the emitted C.

MM = """\
begin parameters
    1 kcat  2.0  # Constant
    2 Km    35.0  # Constant
    3 kdeg  0.05  # Constant
end parameters
begin species
    1 S() 120
    2 E() 25
    3 P() 0
end species
begin reactions
    1 1,2 3,2 MM kcat Km #_R1
    2 3 0 kdeg #_R2
end reactions
begin groups
    1 St                   1
    2 Et                   2
    3 Pt                   3
end groups
"""

# kcat derived from two primaries, so the #15/#41 chain rule has to reach them
# through the closed form (∂rate/∂kcat · ∂kcat/∂primary).
MM_DERIVED = """\
begin parameters
    1 kbase  0.5  # Constant
    2 boost  4.0  # Constant
    3 kcat   kbase*boost  # ConstantExpression
    4 Km     35.0  # Constant
end parameters
begin species
    1 S() 120
    2 E() 25
    3 P() 0
end species
begin reactions
    1 1,2 3,2 MM kcat Km #_R1
end reactions
begin groups
    1 St                   1
    2 Pt                   3
end groups
"""


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


# ─── the derivative, against sympy ─────────────────────────────────────────


def _tqssa_partials(kcat, Km, E, S, stat=1.0):
    """Exactly the arithmetic the emitted C performs."""
    delta = S - Km - E
    D = math.sqrt(delta * delta + 4.0 * Km * S)
    sF = 0.5 * (delta + D)
    if not (sF > 0.0 and D > 0.0):
        return 0.0, 0.0
    dkcat = stat * E * sF / (Km + sF)
    dsF_dKm = 0.5 * (-1.0 + (2.0 * S - delta) / D)
    KpsF = Km + sF
    dKm = kcat * stat * E * (dsF_dKm * Km - sF) / (KpsF * KpsF)
    return dkcat, dKm


class TestThePartials:
    def test_the_partials_match_sympy(self):
        """The claim the whole stage rests on, checked symbolically rather than
        by spot values: sympy differentiates the tQSSA rate and the difference
        against what is emitted simplifies to exactly zero."""
        import sympy as sp

        kcat, Km, E, S, stat = sp.symbols("kcat Km E S stat", positive=True)
        delta = S - Km - E
        D = sp.sqrt(delta**2 + 4 * Km * S)
        sF = (delta + D) / 2
        rate = kcat * stat * E * sF / (Km + sF)

        mine_kcat = stat * E * sF / (Km + sF)
        dsF_dKm = (-1 + (2 * S - delta) / D) / 2
        mine_Km = kcat * stat * E * (dsF_dKm * Km - sF) / (Km + sF) ** 2

        assert sp.simplify(sp.diff(rate, kcat) - mine_kcat) == 0
        assert sp.simplify(sp.diff(rate, Km) - mine_Km) == 0

    def test_the_emitted_grouping_is_the_stable_one(self):
        """Algebraic identity is not enough. The 'simplified' ∂/∂Km — the ½
        factors cancelled, ``2·kcat·E·(E−S−D+Km·B/D)/(A+D)²`` — is identical on
        paper and worthless in float64 on log-spread parameters. This pins that
        the *emitted* grouping is the one that survives, so a future tidy-up
        that 'simplifies' it fails here instead of silently in a gradient."""
        kcat, Km, E, S = 29.9, 2.88e-4, 510.0, 2.4e-4
        emitted = _tqssa_partials(kcat, Km, E, S)[1]

        d = S - Km - E
        A, B = S + Km - E, S + Km + E
        D = math.sqrt(d * d + 4.0 * Km * S)
        simplified = 2.0 * kcat * E * (E - S - D + Km * B / D) / ((A + D) * (A + D))

        # They are the same expression; they are not the same number.
        assert abs(simplified - emitted) > 0.1 * abs(emitted)

    def test_both_partials_vanish_where_the_rate_does(self):
        """At S = 0 the rate is 0 for *every* kcat and Km (S = 0 forces
        sFree = 0 whatever Km is), so the guard returning 0 is the correct
        derivative, not a fallback."""
        for Km in (0.1, 1.0, 100.0):
            assert _tqssa_partials(2.0, Km, 25.0, 0.0) == (0.0, 0.0)


# ─── the gate ──────────────────────────────────────────────────────────────


class TestTheGate:
    def test_an_mm_model_now_gets_an_analytic_sens_rhs(self, tmp_path):
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, MM))
        assert has_sens is True

    def test_the_ab_hatch_still_restores_the_difference_quotient(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, MM))
        assert has_sens is False

    def test_no_jacobian_plan_declines_out_loud(self, tmp_path, caplog, monkeypatch):
        """``J·yS`` for an MM reaction is built from the analytical Jacobian
        plan, and the plan is also what says which species is the enzyme. With
        no plan there is nothing to build it from — decline, and say so."""
        import logging

        model = _model(tmp_path, MM)
        monkeypatch.setattr(
            type(model._core),
            "codegen_jacobian_plan",
            lambda self: {"available": False},
            raising=False,
        )
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert cg.generate_sens_from_model(model, functional=True) is None
        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("no analytical Jacobian plan" in m for m in msgs), msgs

    def test_a_plan_that_disagrees_declines_rather_than_guessing(self, tmp_path):
        """∂f/∂p is read off ``codegen_data`` and ``J·v`` off the plan. If the
        two disagree about which species is the enzyme, emitting both would pair
        a derivative with a matvec from a different reaction."""
        model = _model(tmp_path, MM)
        core = model._core
        data = core.codegen_data()
        plan_mm = [dict(m) for m in core.codegen_jacobian_plan()["mm"]]
        plan_mm[0]["e_idx"] = int(plan_mm[0]["e_idx"]) + 1  # a lie
        param_idx = {p["name"]: i for i, p in enumerate(data["parameters"])}
        terms, why = cg._mm_dfdp_terms(data, plan_mm, param_idx, set(param_idx), {})
        assert terms == {}
        assert why is not None and "disagrees with the analytical Jacobian plan" in why

    def test_an_elementary_model_is_untouched(self, tmp_path):
        """Nothing in this stage may reach a model with no MM reaction."""
        text = MM.replace("    1 1,2 3,2 MM kcat Km #_R1\n", "")
        model = _model(tmp_path, text)
        shut = cg.generate_sens_from_model(model)
        open_ = cg.generate_sens_from_model(model, functional=True)
        assert shut is not None and shut == open_


# ─── one reconstruction, two consumers ─────────────────────────────────────


class TestTheSharedBuilder:
    def test_the_jacobian_and_the_matvec_come_from_one_builder(self, tmp_path):
        """The #67 property, extended to MM: ``generate_jacobian_from_model``
        and ``bngsim_jac_vec`` differ only in the scatter they pass. Asserted by
        driving the builder with both and checking the emitted bodies agree
        line-for-line apart from the left-hand sides."""
        model = _model(tmp_path, MM)
        core = model._core
        plan_mm = core.codegen_jacobian_plan()["mm"]

        def dense_add(col, row, value_c, prefix):
            return f"{prefix}jac[{col}*N_SPECIES + {row}] += {value_c};"

        jac = cg._mm_jacobian_groups(plan_mm, dense_add)
        jacv = cg._mm_jacobian_groups(plan_mm, cg._jacv_add)
        assert jac is not None and jacv is not None
        assert len(jac) == len(jacv) == 1

        def maths_only(lines):
            return [ln for ln in lines if "jac[" not in ln and "Jv_out[" not in ln]

        assert maths_only(jac[0]) == maths_only(jacv[0])
        # ... and the scatter really is the fused matvec, not an n×n write.
        assert any("Jv_out[" in ln and ") * v[" in ln for ln in jacv[0])
        assert not any("jac[" in ln for ln in jacv[0])

    def test_a_scatter_that_cannot_place_an_entry_declines(self, tmp_path):
        model = _model(tmp_path, MM)
        plan_mm = model._core.codegen_jacobian_plan()["mm"]
        assert cg._mm_jacobian_groups(plan_mm, lambda *a: None) is None


# ─── the emitted C ─────────────────────────────────────────────────────────


class TestTheEmission:
    def test_both_rate_constants_get_a_column(self, tmp_path):
        model = _model(tmp_path, MM)
        src = cg.generate_sens_from_model(model, functional=True)
        assert src is not None
        names = [p["name"] for p in model._core.codegen_data()["parameters"]]
        for name in ("kcat", "Km"):
            assert f"    case {names.index(name)}:" in src
        assert "double sFree = 0.5*(delta + Dmm);" in src

    def test_the_enzyme_row_is_not_scattered(self, tmp_path):
        """An MM enzyme is on both sides, so its net stoichiometry is 0. Emitting
        ``dfdp_out[e] += (0) * v`` would be one dead line per reaction per
        column."""
        src = cg.generate_sens_from_model(_model(tmp_path, MM), functional=True)
        assert "(0) * v" not in src

    def test_a_derived_rate_constant_reaches_its_primaries(self, tmp_path):
        """``kcat = kbase*boost``: ∂f/∂kbase exists only through the chain rule,
        and a lost chain rule here reads downstream as a converged zero (#56)."""
        model = _model(tmp_path, MM_DERIVED)
        core = model._core
        data = core.codegen_data()
        names = [p["name"] for p in data["parameters"]]
        param_idx = {n: i for i, n in enumerate(names)}
        primaries = {p["name"] for p in data["parameters"] if p.get("is_const", True)}
        derived = {
            p["name"]: p["expression"]
            for p in data["parameters"]
            if not p.get("is_const", True) and p.get("expression")
        }
        terms, why = cg._mm_dfdp_terms(
            data, core.codegen_jacobian_plan()["mm"], param_idx, primaries, derived
        )
        assert why is None
        cols = {k for k, _lines in terms[0]}
        assert param_idx["kcat"] in cols  # its own direct column
        assert param_idx["kbase"] in cols and param_idx["boost"] in cols


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
            ctypes.c_int, ctypes.c_double, dp, dp, ctypes.c_int,
            dp, dp, ctypes.c_void_p, dp, dp,
        ]  # fmt: skip

    def f(self, t, y, p):
        pbuf = (ctypes.c_double * self.n_par)(*p)
        ud = _CodegenUserData(
            param_values=ctypes.cast(pbuf, ctypes.POINTER(ctypes.c_double)),
            tfun_ctx=None,
            tfun_eval=None,
        )
        ydot = (ctypes.c_double * self.n_sp)()
        rc = self.lib.bngsim_codegen_rhs(
            float(t), (ctypes.c_double * self.n_sp)(*y), ydot, ctypes.byref(ud)
        )
        assert rc == 0
        return list(ydot)

    def dfdp(self, t, y, p, iP):
        """An all-zero ``yS`` zeroes J·yS exactly, so ySdot is the bare ∂f/∂p."""
        pbuf = (ctypes.c_double * self.n_par)(*p)
        ud = _SensUserData(param_values=pbuf, plist=(ctypes.c_int * 1)(int(iP)), n_sens=1)
        ySdot = (ctypes.c_double * self.n_sp)()
        scratch = [(ctypes.c_double * self.n_sp)() for _ in range(4)]
        rc = self.lib.bngsim_codegen_sens_rhs(
            1, float(t), (ctypes.c_double * self.n_sp)(*y), scratch[0], 0,
            scratch[1], ySdot, ctypes.byref(ud), scratch[2], scratch[3],
        )  # fmt: skip
        assert rc == 0
        return list(ySdot)


# Cancellation noise falls as 1/h while truncation grows as h², so a component
# that agrees at ANY of three well-separated steps is one the analytic
# derivative got right; only a genuine error survives all three.
_STEPS = (1e-7, 1e-5, 1e-3)

_STATES = [
    (0.0, [120.0, 25.0, 0.0]),
    (1.0, [80.0, 25.0, 40.0]),
    (5.0, [12.0, 25.0, 100.0]),
    (9.0, [0.3, 25.0, 118.0]),
]


@requires_cc
class TestAgainstFiniteDifference:
    """Central FD of the *emitted* RHS against the *emitted* ∂f/∂p, both called
    through ctypes on the same ``p[]``. No integrator, no tolerance tuning."""

    @pytest.mark.parametrize("text", [MM, MM_DERIVED])
    def test_dfdp_matches_a_finite_difference_of_the_rhs(self, tmp_path, monkeypatch, text):
        model = _model(tmp_path, text)
        comp = _Compiled(model, tmp_path, monkeypatch)
        core = model._core
        params = comp.data["parameters"]
        names = [p["name"] for p in params]
        base = [float(p["value"]) for p in params]

        worst, where = 0.0, None
        for k in range(len(params)):
            for t, y in _STATES:
                analytic = comp.dfdp(t, y, base, k)
                floor = max(max(abs(v) for v in comp.f(t, y, base)), 1e-30) * 1e-9
                best = None
                for rel in _STEPS:
                    v = base[k]
                    h = rel * abs(v) if v != 0.0 else rel
                    if bool(params[k].get("is_const", True)):
                        # A primary moves through set_param, which re-derives
                        # every ConstantExpression — the runtime semantics the
                        # emitted chain rule mirrors.
                        core.set_param(names[k], v + h)
                        pp = [float(core.get_param(n)) for n in names]
                        core.set_param(names[k], v - h)
                        pm = [float(core.get_param(n)) for n in names]
                        core.set_param(names[k], v)
                    else:
                        pp, pm = list(base), list(base)
                        pp[k], pm[k] = v + h, v - h
                    fd = [
                        (a - b) / (2 * h)
                        for a, b in zip(comp.f(t, y, pp), comp.f(t, y, pm), strict=False)
                    ]
                    err = max(
                        abs(a - d) / max(abs(d), floor) for a, d in zip(analytic, fd, strict=False)
                    )
                    best = err if best is None else min(best, err)
                if best > worst:
                    worst, where = best, (names[k], t)
        assert worst < 1e-6, f"worst {worst:.3e} at {where}"


@requires_cc
class TestEndToEnd:
    def test_mm_sensitivities_match_a_resolved_trajectory(self, tmp_path):
        """The claim a caller cares about. Ground truth is an independent
        re-integration at p ± h, which knows nothing about the code path.

        Structurally-zero columns are asserted to *be* zero rather than compared:
        ``kdeg`` is P's decay rate and P feeds nothing back, so ∂S/∂kdeg is
        identically 0 and the trajectory FD reports only its own noise there
        (~1e-3 against a column scale of ~880). Comparing against that would be
        testing the oracle, not the gradient."""
        params = ["kcat", "Km", "kdeg"]
        model = _model(tmp_path, MM)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
        run = sim.run(t_span=(0.0, 30.0), n_points=31, rtol=1e-12, atol=1e-13)
        got = np.asarray(run.sensitivities)  # (time, species, param)

        checked = 0
        for k, name in enumerate(params):
            base = _model(tmp_path, MM, name=f"fd_{name}.net")
            p0 = float(base.get_param(name))
            h = 1e-6 * abs(p0)

            def traj(value, nm=name):
                m = _model(tmp_path, MM, name=f"fd_{nm}.net")
                m.set_param(nm, value)
                return np.asarray(
                    bngsim.Simulator(m, method="ode")
                    .run(t_span=(0.0, 30.0), n_points=31, rtol=1e-12, atol=1e-13)
                    .species
                )

            fd = (traj(p0 + h) - traj(p0 - h)) / (2 * h)
            scale = max(float(np.max(np.abs(fd))), 1e-300)
            for j in range(got.shape[1]):
                a_col, fd_col = got[:, j, k], fd[:, j]
                if not np.any(a_col) and float(np.max(np.abs(fd_col))) < 1e-4 * scale:
                    continue  # structurally zero; the FD is reporting its noise
                np.testing.assert_allclose(
                    a_col, fd_col, rtol=1e-4, atol=1e-6 * scale,
                    err_msg=f"parameter {name}, species {j}",
                )  # fmt: skip
                checked += 1
        # Guard against the skip rule quietly swallowing everything.
        assert checked >= 5, f"only {checked} (species, parameter) columns were compared"

    def test_the_analytic_path_and_the_difference_quotient_agree(self, tmp_path, monkeypatch):
        """The migration check: turning MM onto the analytic RHS must not move
        the answer where the DQ converges."""
        params = ["kcat", "Km", "kdeg"]

        def run(name):
            m = _model(tmp_path, MM, name=name)
            sim = bngsim.Simulator(m, method="ode", sensitivity_params=params)
            return np.asarray(
                sim.run(t_span=(0.0, 30.0), n_points=31, rtol=1e-11, atol=1e-12).sensitivities
            )

        analytic = run("a.net")
        with monkeypatch.context() as mp:
            mp.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
            dq = run("dq.net")
        scale = max(float(np.max(np.abs(analytic))), 1e-300)
        np.testing.assert_allclose(analytic, dq, rtol=1e-5, atol=1e-6 * scale)
        assert float(np.max(np.abs(analytic[:, :, 1]))) > 0.0  # Km column is live
