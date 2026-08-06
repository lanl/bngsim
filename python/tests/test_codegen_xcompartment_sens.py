"""GH #160 — the analytic sensitivity RHS covers cross-compartment reactions.

A ``per_species_volume_scaling`` reaction — one whose affected species live in
compartments of different size — used to decline the analytic ``∂f/∂p`` for the
*whole model*: ``CVodeSensInit1`` installs one callback for every column, so a
single such reaction put every column on CVODES' internal difference quotient.
The decline was right at the time. A cross-compartment kinetic law evaluates to
amount/time while each affected species stores amount/V_c with a V_c of its own,
so every accumulation row divides by its own compartment volume, and the ∂f/∂p
scatter had no form for that divide.

It has one now, and it is not a new derivation: ``_psvs_row_divisor`` is the
lookup the RHS scatter (which *defines* the divide) and the ``J·yS`` half already
share, and the ∂f/∂p scatter applies the same divisor to its coefficient — folded
at emit time for a compartment that is neither writable nor an ODE state, read
from ``p[k]`` when the size is a writable parameter (issue #170 stage 2), and a
runtime divide by the live volume for a variable one (GH #171). So
``ySdot = J·yS + ∂f/∂p`` is analytic on both halves for these models.

The oracle is the one #66 introduced: a central finite difference of the
**emitted** ``bngsim_codegen_rhs`` with respect to each parameter, against the
emitted ``∂f/∂p``, both called through ctypes. Solver-free, and both sides read
the same ``obs[]``/``func[]`` intermediates — so it is the emitted RHS's own
divide the derivative is checked against, not a restatement of the emitter.

**Every fixture here is parametrized over the compartment volume, and that is
load-bearing.** At V = 1 the missing divide is the identity: an emitter that
drops it entirely passes the oracle at V₂ = 1 with a ratio of 3e-08 and fails at
V₂ = 5 with 8e+05 (measured against a mutant that returns ``(-1, 1.0)`` from
``_psvs_row_divisor``). A fixture at the default volume would test nothing — the
same trap GH #119's dropped ``volume_factor`` hid in for as long as it did.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging

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


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# The SBML string is the issue's own reproduction: two compartments, one
# transport reaction across them. ``size(C2)`` is the only thing that varies —
# at 1 the loader emits two Elementary reactions and nothing is
# per_species_volume_scaling; at anything else ``transport`` becomes a
# cross-compartment Functional reaction and every row of it divides.

_XCOMP_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp">
    <listOfCompartments>
      <compartment id="C1" size="1" constant="true"/>
      <compartment id="C2" size="{v2}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="100" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C2" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="kb" value="0.1" constant="true"/>{influx_param}
    </listOfParameters>
    <listOfReactions>{influx_reaction}
      <reaction id="transport" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C1</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="degB" reversible="false" fast="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C2</ci><apply><times/><ci>kb</ci><ci>B</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# Static compartment volumes, spanning both sides of 1.0. 1.0 itself is the
# control: it is the one value at which an undivided ∂f/∂p is correct.
_STATIC_VOLUMES = ("1", "5", "0.37")

# GH #171's variable-volume cross-compartment models (shared verbatim with
# test_codegen_jacobian.py, where the same three pin the compiled Jacobian).
# Here the divisor is not a constant: a row whose compartment is itself an ODE
# state divides by the live y[live_idx]. ``explicit_rr`` is the interesting one —
# the law names ``cell`` too, so the ∂/∂cell that comes out of differentiating
# the law and the one that comes out of the divide have to coexist.
_VARVOL = {
    "cross_rr": (
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; cell'=g; J1: A+B=>P; k*A*B; end"
    ),
    "both_rr": (
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; h=0.07; cell'=g; dish'=h; J1: A+B=>P; k*A*B; end"
    ),
    "explicit_rr": (
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; cell'=g; J1: A+B=>P; cell*k*A*B; end"
    ),
}


# A zeroth-order influx into A and a competing single-compartment sink, off by
# default. The issue's model decays to the origin, where every dY_ss/dp is zero
# and the steady-state consumer has nothing to be checked against. Opened up, the
# equilibrium is closed form —
#
#     A_ss = kin/(k + kA)          B_ss = C1·k·kin / (V₂·kb·(k + kA))
#
# — and every column of dY_ss/dp is nonzero, including ``dB_ss/dk``, the one the
# cross-compartment reaction's own rate constant controls. The competing sink is
# what makes that column nonzero: without it every molecule of A ends up in B
# regardless of how fast the transport runs, and ``B_ss`` does not depend on ``k``
# at all — a steady-state check that could not see this fix either way.
_OPEN_PARAMS = (
    '\n      <parameter id="kin" value="4.0" constant="true"/>'
    '\n      <parameter id="kA" value="0.2" constant="true"/>'
)
_OPEN_REACTIONS = """
      <reaction id="prodA" reversible="false" fast="false">
        <listOfProducts><speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C1</ci><ci>kin</ci></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="degA" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C1</ci><apply><times/><ci>kA</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>"""


def _xcomp_text(v2, *, open_=False):
    return _XCOMP_SBML.format(
        v2=v2,
        influx_param=_OPEN_PARAMS if open_ else "",
        influx_reaction=_OPEN_REACTIONS if open_ else "",
    )


def _xcomp(v2, *, open_=False):
    m = bngsim.Model.from_sbml_string(_xcomp_text(v2, open_=open_))
    bngsim.Simulator(m, method="ode")  # warms codegen_jacobian_plan()
    return m


def _varvol(name):
    pytest.importorskip("antimony")
    m = bngsim.Model.from_antimony_string(_VARVOL[name])
    m.prepare_analytical_jacobian()
    bngsim.Simulator(m, method="ode")
    return m


def _psvs_reactions(model):
    return [
        (i, r["type"])
        for i, r in enumerate(model._core.codegen_data()["reactions"])
        if r.get("per_species_volume_scaling", False)
    ]


# ─── the gate ──────────────────────────────────────────────────────────────


class TestTheGate:
    @pytest.mark.parametrize("v2", _STATIC_VOLUMES)
    def test_a_cross_compartment_model_now_gets_an_analytic_sens_rhs(self, v2):
        """The whole of #160 in one assertion. Parametrized over the volume
        because V₂ = 1 is not cross-compartment at all — it is the control that
        shows the fixture, not the fix, is what changed."""
        model = _xcomp(v2)
        assert (v2 != "1") == bool(_psvs_reactions(model)), "fixture is not what it claims"
        _src, has_sens = cg.generate_combined_from_model(model)
        assert has_sens is True

    @pytest.mark.parametrize("name", list(_VARVOL))
    def test_a_variable_volume_cross_compartment_model_too(self, name):
        model = _varvol(name)
        assert _psvs_reactions(model), "fixture is not cross-compartment"
        _src, has_sens = cg.generate_combined_from_model(model)
        assert has_sens is True

    def test_the_decline_no_longer_warns(self, caplog):
        """The decline was not silent — it warned, once per model — so its
        absence is the observable a caller sees."""
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            cg.generate_sens_from_model(_xcomp("5"), functional=True)
        assert "cross-compartment" not in caplog.text

    def test_the_ab_hatch_still_restores_the_difference_quotient(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        _src, has_sens = cg.generate_combined_from_model(_xcomp("5"))
        assert has_sens is False


class TestTheEmittedScatter:
    @pytest.mark.parametrize("v2", ("2", "5", "0.37"))
    def test_the_row_in_the_other_compartment_carries_one_over_its_volume(self, v2):
        """The divide, read off the emitted C. ``B`` lives in C2, so its row is
        divided by C2's volume; ``A`` lives in C1, so its row is divided by C1's.

        Issue #170 stage 2 moved that divisor from a folded literal to ``p[k]`` —
        both compartments here are writable sizes, and a folded ``1/V₂`` would
        freeze at the volume the source was generated at. The emitted form is
        ``coeff / p[k] * v``, not ``coeff * (1.0/p[k]) * v``: the pre-#170 text
        folded ``coeff / sdiv`` in Python, and one correctly-rounded divide of the
        same two doubles followed by the same multiply is what reproduces it."""
        m = _xcomp(v2)
        pn = [q["name"] for q in m._core.codegen_data()["parameters"]]
        c1, c2 = pn.index("C1"), pn.index("C2")
        src = cg.generate_sens_from_model(m, functional=True)
        body = src.split("static void bngsim_dfdp")[1].split("static void bngsim_jac_vec")[0]
        assert f"dfdp_out[1] += 1.0 / p[{c2}] * v;" in body
        assert f"dfdp_out[0] += -1.0 / p[{c1}] * v;" in body
        # ...and the divisor really is each row's OWN compartment, not one shared
        # index — the whole point of a per-species divide.
        assert c1 != c2

    @pytest.mark.parametrize("name", list(_VARVOL))
    def test_a_live_volume_row_divides_at_runtime(self, name):
        """A compartment that is itself an ODE state cannot be folded at emit
        time, so the row keeps the same ``y[L] > 0.0 ? y[L] : V_static`` guard the
        RHS scatter uses."""
        src = cg.generate_sens_from_model(_varvol(name), functional=True)
        body = src.split("static void bngsim_dfdp")[1].split("static void bngsim_jac_vec")[0]
        assert " * v / (y[" in body and "> 0.0 ? y[" in body

    def test_a_same_volume_row_still_divides_by_its_live_size(self):
        """A V = 1 row used to emit the plain ``-= v`` — ``row_divisor`` recorded
        only rows whose folded divisor differed from 1.0. Issue #170 stage 2 has to
        record it anyway when the size is a *writable parameter*: the write that
        moves it off 1.0 comes after the source is generated, so a row that skipped
        the divide would silently keep dividing by 1.

        That is a text change with no value change — ``-1.0 / p[k] * v`` at
        ``p[k] == 1.0`` is ``-v`` exactly — and it is not asserted numerically here
        because the corpus A/B already covers it: over the 214-model SBML corpus the
        RHS fingerprint and the trajectory are bit-identical to the pre-stage-2
        build on every model, interpreted and codegen alike."""
        m = _xcomp("5")
        pn = [q["name"] for q in m._core.codegen_data()["parameters"]]
        src = cg.generate_sens_from_model(m, functional=True)
        body = src.split("static void bngsim_dfdp")[1].split("static void bngsim_jac_vec")[0]
        assert f"dfdp_out[0] += -1.0 / p[{pn.index('C1')}] * v;" in body
        assert "1.0 * v" not in body

    def test_the_divisor_lookup_is_the_one_the_rhs_uses(self):
        """``_psvs_row_divisor`` is shared with the RHS emitter and the J·v
        reconstruction rather than re-derived here; pin its contract directly so
        a change to it has to face this test and not just the FD oracle."""
        species = [
            {"volume_factor": 1.0},
            {"volume_factor": 5.0},
            {"volume_factor": 2.0, "ode_live_volume_idx0": 3},
            {},
            # (#170 stage 2) a writable compartment size: the divisor is p[7], and
            # the promoted-compartment index stays -1 — the two are exclusive.
            {"volume_factor": 4.0, "volume_param_idx0": 7},
        ]
        assert cg._psvs_row_divisor(species, 0) == (-1, 1.0, -1)
        assert cg._psvs_row_divisor(species, 1) == (-1, 5.0, -1)
        assert cg._psvs_row_divisor(species, 2) == (3, 2.0, -1)
        assert cg._psvs_row_divisor(species, 3) == (-1, 1.0, -1)
        assert cg._psvs_row_divisor(species, 4) == (-1, 4.0, 7)


class TestTheClassesStillDeclined:
    """The divide reaches ∂f/∂p through ``row_divisor`` and J·yS through
    ``_functional_jacobian_groups`` — both of which are Functional-only. An
    Elementary or Michaelis–Menten reaction carrying the flag would get a divided
    ∂f/∂p and an undivided J·v, which is worse than declining. No loader emits
    one; this is what keeps that a fact rather than an assumption."""

    def _elementary_xcomp(self):
        from bngsim._bngsim_core import ModelBuilder

        b = ModelBuilder()
        b.add_parameter("k", 0.3)
        a = b.add_species("A", 100.0, False, 1.0)
        p = b.add_species("B", 0.0, False, 5.0)
        b.add_observable("A", [(a, 1.0)])
        b.add_observable("B", [(p, 1.0)])
        b.add_reaction([a], [p], "elementary", "k", per_species_volume_scaling=True)
        return b.build()

    def test_an_elementary_cross_compartment_reaction_declines(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert cg.generate_sens_from_model(self._elementary_xcomp(), functional=True) is None
        assert "cross-compartment" in caplog.text
        assert "elementary" in caplog.text

    def test_it_declines_on_the_elementary_only_path_too(self):
        """``functional=False`` is the pre-#67 entry point, and an all-Elementary
        model reaches the emitter through it — so the guard cannot live behind the
        keyword."""
        assert cg.generate_sens_from_model(self._elementary_xcomp()) is None


# ─── the ctypes harness ────────────────────────────────────────────────────
#
# Same shape as test_codegen_functional_dfdp.py's; kept local because that one
# compiles the ∂f/∂p half alone and this file needs J·v out of the same .so.


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
        assert sens is not None, "model did not get an analytic sensitivity RHS"
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

    def dfdp(self, t, y, p, iP):
        """An all-zero ``yS`` zeroes J·yS exactly, leaving the bare ∂f/∂p."""
        return self.sens_rhs(t, y, p, iP, [0.0] * self.n_sp)

    def jac_vec(self, t, y, p, v):
        """``J·v`` alone: ∂f/∂p does not depend on yS, so differencing against
        yS = 0 cancels it exactly."""
        return self.sens_rhs(t, y, p, 0, v) - self.dfdp(t, y, p, 0)


def _perturbed(core, data, k, rel):
    """``(p_plus, p_minus)`` for parameter ``k``, moved through ``set_param`` so
    every ConstantExpression parameter is re-derived as a real caller's would be.

    Issue #164 — a compartment size no longer goes through ``set_param``: that
    write is refused, because the size is folded at load into constants the
    write cannot reach. What is wanted here is narrower than a model mutation,
    though. This differences the *emitted* ``f`` with respect to its own ``p[]``
    argument, so the vector is built directly for those columns; ``set_param``
    is only needed for the chain rule it applies on the way, and that has
    nothing to re-derive unless some parameter is expression-valued. Asserted
    below rather than assumed, so a future fixture with a compartment-dependent
    derived parameter fails here instead of silently losing the chain term."""
    params = data["parameters"]
    names = [p["name"] for p in params]
    base = [float(p["value"]) for p in params]
    v = base[k]
    h = rel * abs(v) if v != 0.0 else rel
    is_compartment = list(core.param_is_compartment_size)[k]
    if is_compartment:
        assert not any(p.get("is_expression") for p in params), (
            "a compartment-dependent derived parameter needs the set_param chain "
            "rule this branch skips (issue #164 refuses the write; see issue #170)"
        )
    if is_compartment or not bool(params[k].get("is_const", True)):
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
# h², so a component that agrees at ANY of three well-separated steps is one the
# analytic derivative got right; only a genuine error survives all three.
_STEPS = (1e-7, 1e-5, 1e-3)


def _agreement_ratios(an, fp, fm, step, rtol):
    """Per-component |fd − analytic| in units of "the tolerance this component
    deserves" — 1.0 is the pass line. The FD's resolution is set by ‖f‖∞, not by
    |f_i|: a component of f is assembled from rate terms as large as the biggest
    one in the vector."""
    fd = np.array([(a - b) / step for a, b in zip(fp, fm, strict=True)])
    fscale = max(max(abs(v) for v in fp), max(abs(v) for v in fm), float(np.max(np.abs(an))), 0.0)
    noise = EPS * fscale / abs(step / 2.0)
    return fd, np.abs(fd - an) / (rtol * np.maximum(np.abs(an), np.abs(fd)) + 8.0 * noise + 1e-300)


def _assert_dfdp_matches_fd(model, tmp_path, monkeypatch, states, rtol=1e-6):
    comp = _Compiled(model, tmp_path, monkeypatch)
    core = model._core
    params = comp.data["parameters"]
    base = [float(p["value"]) for p in params]
    # (#170 stage 2) The compartment-size columns are OUT OF SCOPE for this
    # oracle, and now visibly so. The emitted f divides a cross-compartment row by
    # p[V] rather than by a literal, so differencing f with respect to that column
    # sees the storage half of d/dV — which the emitted ∂f/∂p does not carry (it has
    # only the kinetic-law half, through the derived rate parameter's chain). That
    # is exactly why every entry point refuses a compartment sensitivity column:
    # `sensitivity_params=["C1"]` raises, `compute_all_sensitivities` skips it with
    # a warning, and `steady_state(sensitivity_params=...)` raises. A partial column
    # is a confidently wrong gradient. Issue #170 stage 3 owns it — including the
    # `-amount/V²` initial-condition seed, which no term here could supply. Asserted
    # rather than assumed: if a compartment column ever stops being refused, the
    # check below fails and this skip has to be revisited.
    _is_comp = list(core.param_is_compartment_size)
    assert any(_is_comp) == any(
        p["name"] in set(getattr(model, "compartment_size_params", []) or []) for p in params
    ), "compartment-size marking disagrees between the core flags and the model"
    worst, where = 0.0, None
    for k in range(len(params)):
        if _is_comp[k]:
            continue
        for t, y in states:
            an = comp.dfdp(t, y, base, k)
            best = None
            for rel in _STEPS:
                plus, minus = _perturbed(core, comp.data, k, rel)
                step = plus[k] - minus[k]
                if step == 0.0:
                    continue
                _fd, ratios = _agreement_ratios(
                    an, comp.f(t, y, plus), comp.f(t, y, minus), step, rtol
                )
                best = ratios if best is None else np.minimum(best, ratios)
            if best is None:
                continue
            i = int(np.argmax(best))
            if best[i] > worst:
                worst, where = float(best[i]), (params[k]["name"], f"row {i}", t)
    assert worst <= 1.0, f"∂f/∂p disagrees with the finite difference at {where} ({worst:g})"
    return worst


def _assert_jac_vec_matches_fd(model, tmp_path, monkeypatch, states, rtol=1e-6):
    """The other half of ``ySdot``. It was already divided (GH #171 gave
    ``_functional_jacobian_groups`` the same lookup), but no cross-compartment
    model ever reached a sensitivity RHS to exercise it — the ∂f/∂p decline took
    the whole model out first."""
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
                _fd, ratios = _agreement_ratios(an, comp.f(t, yp, p), comp.f(t, ym, p), step, rtol)
                best = ratios if best is None else np.minimum(best, ratios)
            if best is None:
                continue
            i = int(np.argmax(best))
            if best[i] > worst:
                worst, where = float(best[i]), (f"row {i}, col {j}", t)
    assert worst <= 1.0, f"J*v disagrees with the finite difference at {where} ({worst:g})"
    return worst


_XCOMP_STATES = [(0.0, [100.0, 0.0]), (4.0, [41.0, 63.0]), (9.0, [3.2, 7.7])]


def _varvol_states(name):
    # Volumes strictly positive and away from 1.0, so neither the live divide nor
    # its `<= 0` fallback is exercised by accident.
    tail = [1.06, 1.04] if name == "both_rr" else [1.06]
    return [
        (0.0, [100.0, 100.0, 0.0] + [1.0] * len(tail)),
        (0.6, [83.0, 71.0, 17.0] + tail),
        (3.0, [12.0, 31.0, 88.0] + [1.30, 1.21][: len(tail)]),
    ]


@requires_cc
class TestFiniteDifferenceOracle:
    @pytest.mark.parametrize("v2", _STATIC_VOLUMES)
    def test_static_cross_compartment_dfdp(self, tmp_path, monkeypatch, v2):
        _assert_dfdp_matches_fd(_xcomp(v2), tmp_path, monkeypatch, _XCOMP_STATES)

    @pytest.mark.parametrize("name", list(_VARVOL))
    def test_variable_volume_cross_compartment_dfdp(self, tmp_path, monkeypatch, name):
        _assert_dfdp_matches_fd(_varvol(name), tmp_path, monkeypatch, _varvol_states(name))

    @pytest.mark.parametrize("v2", _STATIC_VOLUMES)
    def test_static_cross_compartment_jac_vec(self, tmp_path, monkeypatch, v2):
        _assert_jac_vec_matches_fd(_xcomp(v2), tmp_path, monkeypatch, _XCOMP_STATES)

    @pytest.mark.parametrize("name", list(_VARVOL))
    def test_variable_volume_cross_compartment_jac_vec(self, tmp_path, monkeypatch, name):
        _assert_jac_vec_matches_fd(_varvol(name), tmp_path, monkeypatch, _varvol_states(name))


# ─── end to end ────────────────────────────────────────────────────────────


_FD_RTOL, _FD_ATOL = 1e-12, 1e-14


def _trajectory_fd(v2, param, t_end, n, rel=1e-5):
    """``∂y(t)/∂p`` by re-solving the ODE at ``p ± h``, with the per-species noise
    floor the difference carries.

    Independent of the sensitivity machinery entirely, and it moves the parameter
    through ``set_param``. The floor is not optional: the two re-solves agree only
    to the integration tolerance, so each species column carries a difference of
    about ``rtol·max_t|y_i| / h`` that is pure solver noise. On this model that is
    ~1e-5 in the ``A`` column while the true ``∂A/∂kb`` is **exactly zero** — so a
    tolerance scaled by the largest entry of the whole array (the ``B`` column,
    O(100)) would still call the noise a disagreement. Returns
    ``(fd, floor)`` with ``floor`` one value per species."""

    def traj(value):
        m = bngsim.Model.from_sbml_string(_xcomp_text(v2))
        m.set_param(param, value)
        return np.asarray(
            bngsim.Simulator(m, method="ode")
            .run(t_span=(0.0, t_end), n_points=n, rtol=_FD_RTOL, atol=_FD_ATOL)
            .species
        )

    p0 = float(bngsim.Model.from_sbml_string(_xcomp_text(v2)).get_param(param))
    h = rel * abs(p0) if p0 != 0.0 else rel
    hi, lo = traj(p0 + h), traj(p0 - h)
    fd = (hi - lo) / (2.0 * h)
    ymax = np.maximum(np.abs(hi), np.abs(lo)).max(axis=0)
    return fd, (_FD_RTOL * ymax + _FD_ATOL) / h


@requires_cc
class TestEndToEnd:
    """The claim a caller sees: the trajectory sensitivities CVODES integrates
    from the analytic RHS are the right ones."""

    @pytest.mark.parametrize("v2", _STATIC_VOLUMES)
    def test_analytic_matches_a_resolved_trajectory(self, v2):
        params = ["k", "kb"]
        run = bngsim.Simulator(_xcomp(v2), method="ode", sensitivity_params=params).run(
            t_span=(0.0, 20.0), n_points=41, rtol=1e-11, atol=1e-13
        )
        got = np.asarray(run.sensitivities)  # (time, species, param)
        for j, name in enumerate(params):
            fd, floor = _trajectory_fd(v2, name, 20.0, 41)
            tol = 1e-5 * np.maximum(np.abs(fd), np.abs(got[:, :, j])) + 8.0 * floor
            worst = float(np.max(np.abs(got[:, :, j] - fd) / tol))
            assert worst <= 1.0, f"parameter {name}: analytic vs re-solved FD, ratio {worst:g}"

    @pytest.mark.parametrize("v2", ("5", "0.37"))
    def test_analytic_matches_the_difference_quotient_it_replaces(self, monkeypatch, v2):
        """The migration check, distinct from the correctness check above: where
        the fallback converges, switching to the analytic RHS must not move the
        answer. Deliberately at a loose tolerance — the difference quotient is
        the less accurate of the two."""
        params = ["k", "kb"]

        def go():
            return np.asarray(
                bngsim.Simulator(_xcomp(v2), method="ode", sensitivity_params=params)
                .run(t_span=(0.0, 20.0), n_points=41)
                .sensitivities
            )

        analytic = go()
        with monkeypatch.context() as mp:
            mp.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
            dq = go()
        scale = max(float(np.max(np.abs(analytic))), float(np.max(np.abs(dq))), 1e-300)
        np.testing.assert_allclose(analytic, dq, rtol=1e-4, atol=1e-4 * scale)

    def test_the_analytic_rhs_is_actually_the_one_installed(self):
        """Agreement proves nothing if the run did not take the new path."""
        model = _xcomp("5")
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
        so = getattr(model, "_codegen_so_path", "")
        src = getattr(model, "_codegen_c_source", "")
        assert so or src, "no codegen artifact was installed for the sensitivity run"
        if so:
            assert hasattr(ctypes.CDLL(str(so)), "bngsim_codegen_sens_rhs")
        else:
            assert "bngsim_codegen_sens_rhs" in src
        assert sim.run(t_span=(0.0, 10.0), n_points=11).sensitivities is not None

    def test_steady_state_dfdp_stops_differencing(self, monkeypatch):
        """The second consumer, picked up for free: steady_state.cpp reads the
        same ``bngsim_codegen_sens_rhs`` at ``yS = 0`` for the bare ∂f/∂p of
        ``dY_ss/dp = -J⁻¹·(∂f/∂p)``. A cross-compartment model had no such symbol,
        so that factor was a √eps finite difference (the GH #76 step-floor
        problem); now it is the analytic column.

        Checked against the closed form rather than a re-solve, because the
        closed form is where the compartment volume is visible: every ``B_ss``
        column carries a 1/V₂ that the same model read as single-compartment
        would not have."""
        names = ["kin", "k", "kb"]
        model = _xcomp("5", open_=True)
        ss = bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=names)
        assert ss.sens_dfdp_source == "codegen"
        with monkeypatch.context() as mp:
            mp.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
            before = bngsim.Simulator(_xcomp("5", open_=True), method="ode").steady_state(
                sensitivity_params=names
            )
        assert before.sens_dfdp_source == "finite-difference"  # what #160 replaces

        rows = list(ss.species_names)
        got = np.asarray(ss.sensitivity)
        c1, v2, kin, k, ka, kb = 1.0, 5.0, 4.0, 0.3, 0.2, 0.1
        tot = k + ka
        expected = {
            ("A", "kin"): 1.0 / tot,
            ("A", "k"): -kin / tot**2,
            ("A", "kb"): 0.0,
            ("B", "kin"): c1 * k / (v2 * kb * tot),
            ("B", "k"): c1 * kin * ka / (v2 * kb * tot**2),
            ("B", "kb"): -c1 * k * kin / (v2 * kb**2 * tot),
        }
        for (sp, name), want in expected.items():
            assert got[rows.index(sp), names.index(name)] == pytest.approx(
                want, rel=1e-6, abs=1e-9
            ), f"dY_ss/d{name} for {sp}"
