"""GH #67 — the analytic sensitivity RHS switched on for Functional rate laws.

#66 derived the ∂f/∂p half and left it gated off, because ``bngsim_jac_vec`` was
still Elementary-only: a Functional model's ``bngsim_codegen_sens_rhs`` computed
``ySdot = J·yS + ∂f/∂p`` with the Functional reactions missing from ``J``. This
stage supplies that half and opens the gate, so a Functional model gets

    ySdot = J(t,y)·yS + ∂f/∂p

with **both** terms analytic. ``J·yS`` is not a second derivation: it is the same
per-species chain rule and per-observable product rule the compiled analytical
Jacobian already emits, reconstructed by the one shared builder
(``_functional_jacobian_groups``) with the matvec fused into the scatter, so
``Jv_out[i] += coeff·dj·v[j]`` replaces ``jac[j*n+i] += coeff·dj``. No n×n scratch
buffer lives inside the CVODES callback, and the two consumers cannot drift.

Four things are under test, and the last two are what make the first two
believable:

* **The gate opens for the right models.** A Functional model with smooth algebra
  gets the analytic RHS; a conditional or non-smooth one still declines to
  CVODES' difference quotient (GH #68 owns lifting the first class).
* **Elementary models did not move.** They reach none of this code, so their
  emitted C — signatures included — must be byte-identical.
* **J·v is right.** Central finite difference of the emitted
  ``bngsim_codegen_rhs`` with respect to each *species*, against the J·yS the
  sensitivity RHS computes. Solver-free, both sides reading the same ``obs[]`` /
  ``func[]`` intermediates.
* **J·v is the same J the analytical Jacobian emits.** The dense
  ``bngsim_codegen_jac`` from the very same .so, multiplied by v in Python, must
  reproduce ``bngsim_jac_vec`` to the last bit reordering allows. That is the
  claim "one reconstruction, two consumers" stated as an assertion.
"""

from __future__ import annotations

import ctypes
import hashlib

import bngsim
import numpy as np
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

# The TestEndToEnd cases below carried an xfail(strict) quarantine for GH #85 —
# under the MIR JIT backend a Functional model *constructed with*
# ``sensitivity_params`` did not compile, because the JIT prelude never supplied
# the ``size_t`` the GH #198 ``bngsim_codegen_output_sens`` block names. Fixed in
# mir_jit.hpp; these run on both backends now, and
# test_codegen_jit_prelude.py owns the regression.


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# The .net models exercise the **per-observable** reconstruction (BNG2.pl writes
# a rate law as a function of observables and multiplies by ∏R); the SBML one
# exercises the **per-species** reconstruction (the SBML loader bakes the
# reactant factor into the kinetic law, so apply_species_factor is off). Both
# branches feed bngsim_jac_vec and neither is reachable from the other's input
# format, so both fixture kinds are needed.

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

# Saturation, a Hill exponent and ``time()``: ∂/∂A of a ratio is where a dropped
# quotient-rule term would show up, and the explicit t pins that bngsim_jac_vec
# forwards its time argument to the shard blocks.
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

# Two observables through a nested function: the per-observable product rule has
# to sum two ∂func/∂obs columns into the same species column where the groups
# overlap, which a single-observable law never tests.
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

# A second Functional reaction sharing species with the first, plus a fixed
# ($) species — the row bngsim_jac_vec must zero after accumulating into it.
TWO_LAWS = """\
begin parameters
    1 kf     0.4  # Constant
    2 Kd     2.5  # Constant
    3 kr     0.15  # Constant
end parameters
begin functions
    1 bind()    kf*Etot/(Kd + Etot)
    2 unbind()  kr*Ctot
end functions
begin species
    1 $E() 4.0
    2 S() 9.0
    3 C() 0.5
end species
begin reactions
    1 2 3 bind #_R1
    2 3 2 unbind #_R2
end reactions
begin groups
    1 Etot                 1
    2 Stot                 2
    3 Ctot                 3
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

# A condition puts the model in GH #68's class: sympy differentiates the
# Piecewise to a clean 0 w.r.t. a condition-only parameter, dropping the jump.
# #67 declined all four of these; #68 separated the first two — a threshold on
# simulation time is a crossing issue #48 compensates, a threshold on an
# observable was not — #150 admitted the second by rooting on its residual and
# jumping the saltation term there, and #381 the third, whose equality bounds its
# own true-set with the surface `I − 3 = 0` that `I > 3` names. The fourth is
# what none of them can bracket: quadratic in the clock, so there is no stop time
# to solve for, and over no live state, so there is no residual to root on.
SWITCHED = SIR.replace("    1 betaI() beta*I\n", "    1 betaI() if(time() > 3, beta, 0)*I\n")
STATE_SWITCHED = SIR.replace("    1 betaI() beta*I\n", "    1 betaI() if(I > 3, beta, 0)*I\n")
EQ_SWITCHED = SIR.replace("    1 betaI() beta*I\n", "    1 betaI() if(I == 3, beta, 0)*I\n")
UNBRACKETED = SIR.replace(
    "    1 betaI() beta*I\n", "    1 betaI() if(time()*time() > 3, beta, 0)*I\n"
)

# Michaelis–Menten written as an SBML kinetic law: loaded as a *Functional*
# reaction with apply_species_factor off, so the whole rate (including the
# reactant factor) is differentiated per species. This is the only fixture that
# reaches build_per_species_c.
MM_ANTIMONY = "model mm; S=10; P=0; Vmax=1.4; Km=2.5; J0: S -> P; Vmax*S/(Km + S); end"


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _antimony(text):
    pytest.importorskip("antimony")
    return bngsim.Model.from_antimony_string(text)


# ─── the gate ──────────────────────────────────────────────────────────────


class TestTheGate:
    def test_a_functional_model_now_gets_an_analytic_sens_rhs(self, tmp_path):
        """The whole of #67 in one assertion: the only production path to the
        sensitivity emitter used to return None for this model."""
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, SIR))
        assert has_sens is True

    def test_a_clock_conditional_rate_law_is_admitted_by_68(self, tmp_path):
        """GH #68's class. #67 declined every condition; #68 admits the ones
        whose crossing issue #48 stops at and jumps across — here a threshold on
        simulation time, which needs no counter species."""
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, SWITCHED))
        assert has_sens is True

    def test_a_state_conditional_rate_law_is_admitted_by_150(self, tmp_path):
        """The other half of #68, reopened by issue #150. `if(I>3, ...)` reads an
        observable, so its crossing moves with the trajectory — but that crossing
        is now located as a CVODE root and its saltation jump applied there, so
        the in-branch derivative the emitter produces is again the whole
        in-branch story."""
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, STATE_SWITCHED))
        assert has_sens is True

    def test_an_equality_conditional_rate_law_is_admitted_by_381(self, tmp_path):
        """Issue #381. A continuous trajectory holds ``I == 3`` for an instant
        rather than an interval, but the surface bounding that instant is
        ``I − 3 = 0``, which is where ``I > 3`` changes branch as well — so it
        resolves to the same residual and reaches the same root."""
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, EQ_SWITCHED))
        assert has_sens is True

    def test_a_crossing_neither_machinery_brackets_still_declines(self, tmp_path):
        """What is left after #150 and #381. ``time()*time() > 3`` is quadratic
        in the clock, so issue #48's affine solver cannot produce the stop time
        its jump is applied at, and it reads no live state, so issue #150 has no
        residual to root on."""
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, UNBRACKETED))
        assert has_sens is False

    def test_the_ab_hatch_restores_the_difference_quotient(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, SIR))
        assert has_sens is False

    def test_elementary_emission_is_byte_identical(self, tmp_path):
        """An Elementary model reaches none of the new code — not the jac_vec
        groups, not the signature widening — so not a byte may move."""
        model = _model(tmp_path, ELEMENTARY)
        shut = cg.generate_sens_from_model(model)
        open_ = cg.generate_sens_from_model(model, functional=True)
        assert shut is not None and shut == open_
        assert "GH #67" not in shut
        # The exact pre-#67 signature, packed two parameters per line.
        assert "static void bngsim_jac_vec(double t, const double* y,\n" in shut
        assert "                           const double* p, const double* v,\n" in shut
        assert "                           double* Jv_out) {" in shut

    def test_jac_vec_takes_only_the_context_it_reads(self, tmp_path):
        """SIR's J·v is written in ``func[]`` and ``p[]`` but never ``obs[]``,
        while its ∂f/∂p is written in ``obs[]`` and never ``func[]``. Each
        function's signature is decided by what it actually references, so
        neither carries a dead array."""
        src = cg.generate_sens_from_model(_model(tmp_path, SIR), functional=True)
        jacv = src.split("static void bngsim_jac_vec")[1].split("BNGSIM_EXPORT")[0]
        assert "const double* func, double* Jv_out) {" in jacv
        assert "const double* obs" not in jacv.split(") {")[0]
        assert "bngsim_dfdp(iP, t, y, p, obs, dfdp);" in src
        assert "bngsim_jac_vec(t, y, p, yS, func, Jv);" in src

    def test_the_source_no_longer_calls_itself_incomplete(self, tmp_path):
        src = cg.generate_sens_from_model(_model(tmp_path, SIR), functional=True)
        assert "INCOMPLETE" not in src
        assert "GH #66/#67" in src

    def test_chunked_blocks_carry_t_and_the_context(self, tmp_path, monkeypatch):
        """Above the GH #165 threshold the J·v body is lifted into NOINLINE shard
        blocks, and those take a fixed parameter list. A Functional derivative may
        read ``t`` (HILL's does) and ``func[]``, neither of which the Elementary
        block signature carries — so the signature has to widen with the groups or
        the shards do not compile. Exactly one corpus model is big enough to reach
        this naturally, so it is forced here instead."""
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "1")  # chunk everything
        src = cg.generate_sens_from_model(_model(tmp_path, HILL), functional=True)
        proto = [ln for ln in src.splitlines() if ln.startswith("void jacv_blk_")]
        assert proto, "expected chunked jacv blocks"
        assert proto[0] == (
            "void jacv_blk_000(double t, const double* y, const double* p, "
            "const double* v, const double* obs, const double* func, double* Jv_out);"
        )
        assert "    jacv_blk_000(t, y, p, v, obs, func, Jv_out);" in src

    def test_chunked_elementary_blocks_are_unchanged(self, tmp_path, monkeypatch):
        """The other side of it: with no Functional groups the block signature is
        the pre-#67 one, so a chunked Elementary model is byte-identical."""
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "1")
        src = cg.generate_sens_from_model(_model(tmp_path, ELEMENTARY), functional=True)
        assert (
            "void jacv_blk_000(const double* y, const double* p, "
            "const double* v, double* Jv_out);" in src
        )


# ─── the ctypes harness ────────────────────────────────────────────────────


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
    """The model's whole combined .so — RHS, sensitivity RHS and (when the model
    qualifies) the dense analytical Jacobian — reachable through ctypes.

    A ``Simulator`` is constructed first: ``codegen_jacobian_plan()["available"]``
    is False until the functional Jacobian is populated, and without it
    ``generate_jacobian_from_model`` declines and the cross-check below would
    silently skip.
    """

    def __init__(self, model, tmp_path, monkeypatch):
        bngsim.Simulator(model, method="ode")
        core = model._core
        self.data = core.codegen_data()
        self.n_sp = len(self.data["species"])
        self.n_par = len(self.data["parameters"])
        src, has_sens = cg.generate_combined_from_model(model)
        assert has_sens, "model did not get an analytic sensitivity RHS"
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
        so = cg.compile_rhs(src, hashlib.sha256(src.encode()).hexdigest()[:16])
        self.lib = ctypes.CDLL(str(so))
        self.has_jac = hasattr(self.lib, "bngsim_codegen_jac")
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
        if self.has_jac:
            self.lib.bngsim_codegen_jac.restype = ctypes.c_int
            self.lib.bngsim_codegen_jac.argtypes = [ctypes.c_double, dp, dp, ctypes.c_void_p]

    def _rhs_ud(self, p):
        pbuf = (ctypes.c_double * self.n_par)(*p)
        return pbuf, _CodegenUserData(
            param_values=ctypes.cast(pbuf, ctypes.POINTER(ctypes.c_double)),
            tfun_ctx=None,
            tfun_eval=None,
        )

    def f(self, t, y, p):
        _keep, ud = self._rhs_ud(p)
        ydot = (ctypes.c_double * self.n_sp)()
        assert (
            self.lib.bngsim_codegen_rhs(
                float(t), (ctypes.c_double * self.n_sp)(*y), ydot, ctypes.byref(ud)
            )
            == 0
        )
        return list(ydot)

    def jac(self, t, y, p):
        """Dense column-major ∂f_i/∂x_j as an (n, n) array, or None."""
        if not self.has_jac:
            return None
        _keep, ud = self._rhs_ud(p)
        buf = (ctypes.c_double * (self.n_sp * self.n_sp))()
        assert (
            self.lib.bngsim_codegen_jac(
                float(t), (ctypes.c_double * self.n_sp)(*y), buf, ctypes.byref(ud)
            )
            == 0
        )
        return np.array(buf).reshape(self.n_sp, self.n_sp).T  # col-major -> [i][j]

    def sens_rhs(self, t, y, p, iP, yS):
        pbuf = (ctypes.c_double * self.n_par)(*p)
        ud = _SensUserData(param_values=pbuf, plist=(ctypes.c_int * 1)(int(iP)), n_sens=1)
        ySdot = (ctypes.c_double * self.n_sp)()
        scratch = [(ctypes.c_double * self.n_sp)() for _ in range(3)]
        assert (
            self.lib.bngsim_codegen_sens_rhs(
                1,
                float(t),
                (ctypes.c_double * self.n_sp)(*y),
                scratch[0],
                0,
                (ctypes.c_double * self.n_sp)(*yS),
                ySdot,
                ctypes.byref(ud),
                scratch[1],
                scratch[2],
            )
            == 0
        )
        return np.array(list(ySdot))

    def jac_vec(self, t, y, p, v):
        """``J·v`` alone. ``ySdot = J·yS + ∂f/∂p`` and ∂f/∂p does not depend on
        yS, so differencing against yS = 0 cancels it exactly — the mirror of the
        trick #66 used to isolate ∂f/∂p."""
        zero = [0.0] * self.n_sp
        return self.sens_rhs(t, y, p, 0, v) - self.sens_rhs(t, y, p, 0, zero)


# Central-difference cancellation noise falls as 1/h while truncation grows as
# h², so a component that agrees at ANY of three well-separated steps is one the
# analytic derivative got right; only a genuine error survives all three. Without
# this the oracle reports its own noise as a codegen bug.
_STEPS = (1e-7, 1e-5, 1e-3)


def _assert_jac_vec_matches_fd(model, tmp_path, monkeypatch, states, rtol=1e-6):
    """Every column of J, from ``bngsim_jac_vec`` via a unit ``yS``, against a
    central finite difference of the compiled RHS in that species."""
    comp = _Compiled(model, tmp_path, monkeypatch)
    p = [float(q["value"]) for q in comp.data["parameters"]]
    worst, where = 0.0, None
    for t, y in states:
        for j in range(comp.n_sp):
            e_j = [0.0] * comp.n_sp
            e_j[j] = 1.0
            an = comp.jac_vec(t, y, p, e_j)
            best = None
            for rel in _STEPS:
                h = rel * abs(y[j]) if y[j] != 0.0 else rel
                yp, ym = list(y), list(y)
                yp[j], ym[j] = y[j] + h, y[j] - h
                step = yp[j] - ym[j]
                if step == 0.0:
                    continue
                fp, fm = comp.f(t, yp, p), comp.f(t, ym, p)
                fd = np.array([(a - b) / step for a, b in zip(fp, fm, strict=True)])
                # A component of f is assembled from rate terms as large as the
                # biggest one in the vector, so the resolution of the difference
                # is set by ‖f‖∞, not by |f_i|.
                fscale = max(
                    max(abs(v) for v in fp), max(abs(v) for v in fm), float(np.max(np.abs(an)))
                )
                noise = EPS * fscale / abs(step / 2.0)
                ratios = np.abs(fd - an) / (
                    rtol * np.maximum(np.abs(an), np.abs(fd)) + 8.0 * noise + 1e-300
                )
                best = ratios if best is None else np.minimum(best, ratios)
            if best is None:
                continue
            i = int(np.argmax(best))
            if best[i] > worst:
                worst = float(best[i])
                where = (f"row {i}, col {j}", t, float(an[i]))
    assert worst <= 1.0, f"J*v disagrees with the finite difference at {where} (ratio {worst:g})"
    return worst


def _assert_jac_vec_matches_the_analytical_jacobian(model, tmp_path, monkeypatch, states):
    """``bngsim_jac_vec`` against the dense ``bngsim_codegen_jac`` from the same
    .so. Both come from ``_functional_jacobian_groups``; a mismatch means the
    matvec fusion, not the derivative, is wrong — which no FD oracle separates
    from a bad derivative."""
    comp = _Compiled(model, tmp_path, monkeypatch)
    assert comp.has_jac, "no compiled analytical Jacobian to cross-check against"
    p = [float(q["value"]) for q in comp.data["parameters"]]
    rng = np.random.default_rng(20260727)
    for t, y in states:
        J = comp.jac(t, y, p)
        for v in (*np.eye(comp.n_sp), rng.normal(size=comp.n_sp)):
            np.testing.assert_allclose(
                comp.jac_vec(t, y, p, list(v)), J @ v, rtol=1e-12, atol=1e-12
            )


@requires_cc
class TestJacVecAgainstFiniteDifference:
    def test_sir_per_observable_law(self, tmp_path, monkeypatch):
        states = [(0.0, [2e7, 1.0, 0.0]), (3.5, [1.1e7, 4.2e5, 8.7e6])]
        _assert_jac_vec_matches_fd(_model(tmp_path, SIR), tmp_path, monkeypatch, states)

    def test_hill_saturation_exponent_and_time(self, tmp_path, monkeypatch):
        states = [(0.0, [6.0, 1.0]), (12.0, [0.4, 9.1]), (40.0, [3.3, 3.3])]
        _assert_jac_vec_matches_fd(_model(tmp_path, HILL), tmp_path, monkeypatch, states)

    def test_nested_functions_over_two_observables(self, tmp_path, monkeypatch):
        states = [(0.0, [3.0, 5.0, 0.0]), (2.0, [0.7, 11.0, 4.0])]
        _assert_jac_vec_matches_fd(_model(tmp_path, NESTED), tmp_path, monkeypatch, states)

    def test_two_functional_reactions_and_a_fixed_species(self, tmp_path, monkeypatch):
        states = [(0.0, [4.0, 9.0, 0.5]), (7.0, [4.0, 2.2, 7.3])]
        _assert_jac_vec_matches_fd(_model(tmp_path, TWO_LAWS), tmp_path, monkeypatch, states)

    def test_sbml_per_species_reconstruction(self, tmp_path, monkeypatch):
        """The other reconstruction branch: an SBML kinetic law carries its own
        reactant factor (apply_species_factor off), so the derivative is taken
        per species rather than through observable groups."""
        states = [(0.0, [10.0, 0.0]), (4.0, [3.1, 6.9])]
        _assert_jac_vec_matches_fd(_antimony(MM_ANTIMONY), tmp_path, monkeypatch, states)

    def test_elementary_model_is_covered_by_the_same_oracle(self, tmp_path, monkeypatch):
        """The oracle has to agree where the answer was already known, or it is
        measuring itself rather than the new J·v."""
        states = [(0.0, [10.0, 0.0]), (5.0, [2.5, 7.5])]
        _assert_jac_vec_matches_fd(_model(tmp_path, ELEMENTARY), tmp_path, monkeypatch, states)


@requires_cc
class TestJacVecIsTheAnalyticalJacobian:
    def test_sir(self, tmp_path, monkeypatch):
        states = [(0.0, [2e7, 1.0, 0.0]), (3.5, [1.1e7, 4.2e5, 8.7e6])]
        _assert_jac_vec_matches_the_analytical_jacobian(
            _model(tmp_path, SIR), tmp_path, monkeypatch, states
        )

    def test_hill(self, tmp_path, monkeypatch):
        states = [(12.0, [0.4, 9.1]), (40.0, [3.3, 3.3])]
        _assert_jac_vec_matches_the_analytical_jacobian(
            _model(tmp_path, HILL), tmp_path, monkeypatch, states
        )

    def test_two_laws_with_a_fixed_row(self, tmp_path, monkeypatch):
        states = [(7.0, [4.0, 2.2, 7.3])]
        _assert_jac_vec_matches_the_analytical_jacobian(
            _model(tmp_path, TWO_LAWS), tmp_path, monkeypatch, states
        )

    def test_sbml_per_species(self, tmp_path, monkeypatch):
        states = [(4.0, [3.1, 6.9])]
        _assert_jac_vec_matches_the_analytical_jacobian(
            _antimony(MM_ANTIMONY), tmp_path, monkeypatch, states
        )


# ─── end to end ────────────────────────────────────────────────────────────


def _run_sens(model, params, t_end=20.0, n=41):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    return sim.run(t_span=(0.0, t_end), n_points=n, rtol=1e-10, atol=1e-12)


def _trajectory_fd(tmp_path, text, param, t_end, n=41, rel=1e-6):
    """``∂y(t)/∂p`` by re-solving the ODE at ``p ± h`` and differencing the whole
    trajectory.

    The only ground truth here that is not circular. CVODES' own difference
    quotient is *not* usable as the oracle: it perturbs inside the integration, so
    when it is the thing under suspicion (and on SIR below it is — it burns its
    entire step budget and returns zeros) it cannot referee itself. Re-solving is
    an independent integration at tighter tolerances, and it moves the parameter
    through ``set_param``, so a derived rate constant is re-derived exactly as a
    real caller's would be.

    Only valid for a parameter that does **not** set an initial condition: GH #79
    (open) means ``set_param`` does not rebuild a species IC that references the
    parameter, so the re-solve would hold ``y(0)`` fixed and report ∂y/∂p missing
    its IC term. SIR's ``S0`` is exactly that parameter, so it is refereed by the
    ∂f/∂p and J·v oracles above instead of here.
    """
    base = bngsim.Model.from_net(_write(tmp_path, text, "fd.net"))
    p0 = float(base.get_param(param))
    h = rel * abs(p0) if p0 != 0.0 else rel

    def traj(value):
        m = bngsim.Model.from_net(_write(tmp_path, text, "fd.net"))
        m.set_param(param, value)
        return np.asarray(
            bngsim.Simulator(m, method="ode")
            .run(t_span=(0.0, t_end), n_points=n, rtol=1e-12, atol=1e-14)
            .species
        )

    return (traj(p0 + h) - traj(p0 - h)) / (2.0 * h)


def _write(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text)
    return path


@requires_cc
class TestEndToEnd:
    """The claim that matters to a caller: the trajectory sensitivities CVODES
    integrates from the analytic RHS are the right ones."""

    @pytest.mark.parametrize(
        ("text", "params", "t_end"),
        [
            (SIR, ["gamma"], 40.0),
            (HILL, ["kmax", "Km"], 20.0),
            (TWO_LAWS, ["kf", "Kd", "kr"], 20.0),
        ],
    )
    def test_analytic_matches_a_resolved_trajectory(self, tmp_path, text, params, t_end):
        run = _run_sens(_model(tmp_path, text), params, t_end=t_end)
        got = np.asarray(run.sensitivities)  # (time, species, param)
        for k, name in enumerate(params):
            fd = _trajectory_fd(tmp_path, text, name, t_end)
            scale = max(float(np.max(np.abs(fd))), 1e-300)
            np.testing.assert_allclose(
                got[:, :, k], fd, rtol=1e-5, atol=1e-6 * scale, err_msg=f"parameter {name}"
            )

    @pytest.mark.parametrize(
        ("text", "params", "t_end"),
        [(HILL, ["kmax", "Km"], 20.0), (TWO_LAWS, ["kf", "Kd", "kr"], 20.0)],
    )
    def test_analytic_matches_the_difference_quotient(
        self, tmp_path, monkeypatch, text, params, t_end
    ):
        """Where the difference quotient does converge, switching to the analytic
        RHS must not move the answer — the migration check, distinct from the
        correctness check above. (SIR is excluded on purpose: its DQ does not
        converge. See the next test.)"""
        analytic = _run_sens(_model(tmp_path, text), params, t_end=t_end)
        with monkeypatch.context() as mp:
            mp.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
            dq = _run_sens(_model(tmp_path, text, name="dq.net"), params, t_end=t_end)
        a, d = np.asarray(analytic.sensitivities), np.asarray(dq.sensitivities)
        assert a.shape == d.shape
        scale = max(np.max(np.abs(a)), np.max(np.abs(d)), 1e-300)
        # The DQ is the *less* accurate of the two, so this is a loose agreement
        # check, not an equality: far tighter than the DQ's own truncation error,
        # far looser than the two integrations' step-history difference.
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5 * scale)

    def test_the_step_count_collapse_on_sir(self, tmp_path, monkeypatch):
        """SIR is why #55 was filed. Its difference-quotient sensitivity run
        exhausts CVODES' step budget and returns a gradient of exactly zero —
        a converged-looking answer that is simply wrong. The analytic RHS solves
        it in a normal number of steps and lands on the re-solved trajectory.

        Asserted on ``n_steps`` rather than wall-clock: deterministic, and it is
        the quantity that actually degrades (DQ noise wrecks step-size control,
        so the gap compounds with the horizon instead of tracking
        reactions × parameters)."""
        params = ["gamma", "S0"]
        analytic = _run_sens(_model(tmp_path, SIR), params, t_end=40.0)
        with monkeypatch.context() as mp:
            mp.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
            dq = _run_sens(_model(tmp_path, SIR, name="dq.net"), params, t_end=40.0)
        a_steps = int(analytic.solver_stats["n_steps"])
        d_steps = int(dq.solver_stats["n_steps"])
        assert a_steps * 10 < d_steps, f"analytic {a_steps} steps vs DQ {d_steps}"
        # ... and the cheap answer is the correct one, which is the whole point.
        got = np.asarray(analytic.sensitivities)
        fd = _trajectory_fd(tmp_path, SIR, "gamma", 40.0)
        np.testing.assert_allclose(
            got[:, :, 0], fd, rtol=1e-5, atol=1e-6 * float(np.max(np.abs(fd)))
        )

    def test_steady_state_dfdp_stops_differencing(self, tmp_path, monkeypatch):
        """A second consumer picks this up for free, and it is worth pinning
        because nothing in the #67 diff mentions it: ``steady_state.cpp`` reads
        the same ``bngsim_codegen_sens_rhs`` at ``yS = 0`` to get the bare ∂f/∂p
        for ``dY_ss/dp = -J⁻¹·(∂f/∂p)``. A Functional model used to have no such
        symbol, so that factor was a √eps difference quotient — the step-floor
        problem GH #76 is about. Now it is the analytic column.

        Checked against a re-solve of the steady state at p ± h, not against the
        difference quotient it replaces."""
        src = (
            "model m; A=0.1; kmax=3; Km=2; kdeg=0.5; J0: -> A; kmax/(Km+A); J1: A -> ; kdeg*A; end"
        )
        names = ["kmax", "Km", "kdeg"]
        ss = bngsim.Simulator(_antimony(src), method="ode").steady_state(sensitivity_params=names)
        assert ss.sens_dfdp_source == "codegen"
        with monkeypatch.context() as mp:
            mp.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
            before = bngsim.Simulator(_antimony(src), method="ode").steady_state(
                sensitivity_params=names
            )
        assert before.sens_dfdp_source == "finite-difference"  # what #67 replaced

        def y_ss(p, v):
            m = _antimony(src)
            m.set_param(p, v)
            return np.asarray(bngsim.Simulator(m, method="ode").steady_state().concentrations)

        base = _antimony(src)
        got = np.asarray(ss.sensitivity)
        for k, p in enumerate(names):
            v = float(base.get_param(p))
            h = 1e-6 * v
            fd = (y_ss(p, v + h) - y_ss(p, v - h)) / (2.0 * h)
            np.testing.assert_allclose(got[:, k], fd, rtol=1e-6, err_msg=p)

    def test_the_analytic_rhs_is_actually_the_one_installed(self, tmp_path):
        """Agreement proves nothing if the run did not take the new path. The
        artifact the Simulator installed has to carry the symbol — and for a
        .net-loaded model that means the emitter reached it through
        ``generate_combined_c``, which is a different hook from the model-based
        entry points.

        Written against whichever artifact this backend produces, because the MIR
        matrix runs this file with ``BNGSIM_CODEGEN_JIT=mir``: there is a C source
        string and no ``.so`` then, and asserting on the ``.so`` alone would turn
        the JIT legs into a false failure (or, if skipped, a false green). Both
        branches are live — GH #85 was what kept the JIT one from ever running."""
        model = _model(tmp_path, SIR)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["gamma"])
        so = getattr(model, "_codegen_so_path", "")
        src = getattr(model, "_codegen_c_source", "")
        assert so or src, "no codegen artifact was installed for the sensitivity run"
        if so:
            assert hasattr(ctypes.CDLL(str(so)), "bngsim_codegen_sens_rhs")
        else:
            assert "bngsim_codegen_sens_rhs" in src
        assert sim.run(t_span=(0.0, 10.0), n_points=11).sensitivities is not None
