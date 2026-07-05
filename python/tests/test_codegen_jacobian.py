"""Compiled-C analytical Jacobian (GH #76 Task 4) — ``bngsim_codegen_jac``.

The codegen ``.so`` now carries a C mirror of
``NetworkModel::fill_dense_analytical_jacobian``.  These tests load the compiled
function via ctypes and assert it reproduces the interpreted assembly
(``_dense_analytical_jacobian``, the same method the CVODE dense callback and the
FD self-check use) to ~1e-12 across the four contribution blocks:

  * Elementary mass-action closed form,
  * Michaelis–Menten (tQSSA) closed form,
  * Functional per-species (SBML chain rule),
  * Functional per-observable (.net product rule),

plus the fixed-species row zeroing.  An end-to-end test confirms an ODE run with
the compiled Jacobian matches one with the interpreted Jacobian.

The interpreted assembly is the FD-self-checked correctness oracle (it is the
same code the analytical-vs-FD trajectory tests validate), so matching it pins
the compiled path to a real reference, not a re-derivation of the same code.
"""

from __future__ import annotations

import ctypes
import os
import shutil

import bngsim
import numpy as np
import pytest
from bngsim._codegen import generate_jacobian_from_model, prepare_model_codegen

pytest.importorskip("sympy")
_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")


class _CodegenUserData(ctypes.Structure):
    # Mirrors CodegenUserDataForSO / the codegen typedef (param_values, tfun_ctx,
    # tfun_eval). The test models carry no table functions, so tfun_* stay NULL.
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("tfun_ctx", ctypes.c_void_p),
        ("tfun_eval", ctypes.c_void_p),
    ]


def _compiled_jac(core, so_path, t, conc):
    """Call the compiled ``bngsim_codegen_jac`` at ``(t, conc)``; return the flat
    column-major matrix (``jac[j*ns + i]``), matching ``_dense_analytical_jacobian``."""
    lib = ctypes.CDLL(str(so_path))
    fn = lib.bngsim_codegen_jac
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    params = [p["value"] for p in core.codegen_data()["parameters"]]
    pbuf = (ctypes.c_double * len(params))(*params)
    ud = _CodegenUserData(
        param_values=ctypes.cast(pbuf, ctypes.POINTER(ctypes.c_double)),
        tfun_ctx=None,
        tfun_eval=None,
    )
    ns = core.n_species
    y = (ctypes.c_double * ns)(*conc)
    jac = (ctypes.c_double * (ns * ns))()
    assert fn(float(t), y, jac, ctypes.byref(ud)) == 0
    return np.array(jac, dtype=float)


def _assert_matches(model, states, tol=1e-12):
    # GH #145: the analytical Jacobian is derived lazily (off the load path), so
    # warm it before exercising the codegen emitter — the ODE-solve setup would do
    # the same. A no-op for the all-Elementary / Michaelis–Menten fixtures (their
    # closed-form Jacobian is complete from the C++ build).
    model.prepare_analytical_jacobian()
    core = model._core
    assert core.analytical_jacobian_complete is True
    src = generate_jacobian_from_model(model)
    assert src is not None and "bngsim_codegen_jac" in src
    so = prepare_model_codegen(model)
    assert so is not None and so.exists()
    worst = 0.0
    for t, conc in states:
        comp = _compiled_jac(core, so, t, conc)
        interp = np.array(core._dense_analytical_jacobian(t, conc), dtype=float)
        denom = np.maximum(np.maximum(np.abs(comp), np.abs(interp)), 1e-9)
        worst = max(worst, float((np.abs(comp - interp) / denom).max()))
    assert worst < tol, f"compiled vs interpreted Jacobian worst rel err {worst:g}"
    return worst


# Per-species Functional: saturating degradation rate = k*S/(Km+S). ∂rate/∂S is
# a non-trivial smooth derivative — the canonical SBML chain-rule case.
_SBML_SAT_DEG = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sat_deg">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="2" constant="true"/>
      <parameter id="Km" value="5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/>
            <apply><times/><ci>k</ci><ci>S</ci></apply>
            <apply><plus/><ci>Km</ci><ci>S</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@needs_cc
def test_functional_per_species_matches_interpreted():
    m = bngsim.Model.from_sbml_string(_SBML_SAT_DEG)
    _assert_matches(m, [(0.0, [10.0, 0.0]), (1.5, [3.0, 7.0]), (0.0, [0.2, 1.0])])


@needs_cc
def test_michaelis_menten_matches_interpreted(data_dir):
    m = bngsim.Model.from_net(str(data_dir / "mm_tqssa.net"))
    _assert_matches(
        m, [(0.0, [10.0, 100.0, 0.0]), (0.0, [80.0, 30.0, 12.0]), (0.0, [5.0, 5.0, 50.0])]
    )


@needs_cc
def test_functional_per_observable_matches_interpreted(data_dir):
    m = bngsim.Model.from_net(str(data_dir / "per_observable_jac.net"))
    ns = m._core.n_species
    rng = np.random.default_rng(1)
    states = [(0.0, list(rng.uniform(0.1, 5.0, ns))) for _ in range(5)]
    _assert_matches(m, states)


@needs_cc
def test_elementary_matches_interpreted(data_dir):
    m = bngsim.Model.from_net(str(data_dir / "simple_decay.net"))
    ns = m._core.n_species
    rng = np.random.default_rng(2)
    _assert_matches(m, [(0.0, list(rng.uniform(0.1, 5.0, ns))) for _ in range(3)])


@needs_cc
def test_declines_when_codegen_jac_disabled():
    # The A/B escape hatch (used by bench_codegen_jac.py) suppresses emission so
    # dispatch falls back to the interpreted Jacobian.
    m = bngsim.Model.from_sbml_string(_SBML_SAT_DEG)
    # GH #145: warm the lazily-derived Jacobian so the emitter has terms to emit
    # (the env hatch below, not the empty terms, must be what makes it decline).
    m.prepare_analytical_jacobian()
    os.environ["BNGSIM_NO_CODEGEN_JAC"] = "1"
    try:
        assert generate_jacobian_from_model(m) is None
    finally:
        os.environ.pop("BNGSIM_NO_CODEGEN_JAC", None)
    # Without the hatch it emits again.
    assert generate_jacobian_from_model(m) is not None


@needs_cc
def test_end_to_end_codegen_jac_matches_interpreted_trajectory(data_dir):
    # Full ODE integration with the compiled Jacobian vs the interpreted one (the
    # dense dispatch prefers the compiled symbol when codegen is active). Both are
    # analytical, so they agree to solver tolerance.
    def run(disable):
        if disable:
            os.environ["BNGSIM_NO_CODEGEN_JAC"] = "1"
        else:
            os.environ.pop("BNGSIM_NO_CODEGEN_JAC", None)
        m = bngsim.Model.from_net(str(data_dir / "mm_tqssa.net"))
        so = prepare_model_codegen(m)
        assert so is not None
        m._codegen_so_path = str(so)
        sim = bngsim.Simulator(m, method="ode", jacobian="analytical")
        return np.asarray(sim.run((0.0, 50.0), 101).species, dtype=float)

    try:
        compiled = run(False)
        interp = run(True)
    finally:
        os.environ.pop("BNGSIM_NO_CODEGEN_JAC", None)
    peak = np.maximum(np.abs(interp).max(axis=0), 1.0)
    assert float((np.abs(compiled - interp) / peak).max()) < 1e-9


# ── (#171 Parts 2-3) Codegen for cross-compartment variable-volume reactions ────
# The RHS divides each species's accumulation by the LIVE compartment volume
# conc[ode_live_volume_idx0]; #144 declined codegen for these (the static
# 1/volume_factor table can't express a runtime divide). #171 Part 2 emits
# `rate / y[live_idx]` in the RHS scatter and Part 3 emits the runtime-divided
# existing columns + the new −func/y[live]² column in the Jacobian. These mirror
# the interpreted analytical Jacobian (the FD-self-checked oracle) exactly.

_C171_CODEGEN_MODELS = {
    "cross_rr": (  # one varvol (cell) + static dish; A,P live, B static
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; cell'=g; J1: A+B=>P; k*A*B; end"
    ),
    "both_rr": (  # both varvol: A,P → cell, B → dish (all rows live, no inv_vf use)
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; h=0.07; cell'=g; dish'=h; J1: A+B=>P; k*A*B; end"
    ),
    "explicit_rr": (  # explicit cell factor: ∂/∂cell existing + new column cancel
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; cell'=g; J1: A+B=>P; cell*k*A*B; end"
    ),
}


@needs_cc
@pytest.mark.parametrize("name", list(_C171_CODEGEN_MODELS))
def test_varvol_codegen_no_longer_declines(name):
    """#171 Part 2: the RHS codegen no longer raises NotImplementedError for a
    cross-compartment variable-volume model, and the Jacobian codegen emits."""
    from bngsim._codegen import generate_rhs_from_model

    m = bngsim.Model.from_antimony_string(_C171_CODEGEN_MODELS[name])
    m.prepare_analytical_jacobian()
    rhs = generate_rhs_from_model(m)  # must not raise
    assert "bngsim_codegen_rhs" in rhs
    # A live row divides by the live volume; a static row keeps inv_vf.
    assert "> 0.0 ?" in rhs
    jac = generate_jacobian_from_model(m)
    assert jac is not None and "bngsim_codegen_jac" in jac


@needs_cc
@pytest.mark.parametrize("name", list(_C171_CODEGEN_MODELS))
def test_varvol_codegen_jacobian_matches_interpreted(name):
    """#171 Part 3: the compiled Jacobian (live divide + the new −func/V_live²
    column) reproduces the interpreted analytical Jacobian at spread-out states —
    including the ``explicit_rr`` case whose existing and new columns cancel."""
    m = bngsim.Model.from_antimony_string(_C171_CODEGEN_MODELS[name])
    m.prepare_analytical_jacobian()
    ns = m._core.n_species
    states = [(0.0, m._core.get_state())]
    for key in (0.6180339887, 0.3819660113):
        y0 = np.asarray(m._core.get_state(), dtype=float)
        frac = (np.arange(1, ns + 1) * key) % 1.0
        states.append((0.3, (np.abs(y0) * (0.4 + 1.8 * frac) + 0.5).tolist()))
    _assert_matches(m, states, tol=1e-10)


@needs_cc
@pytest.mark.parametrize("name", list(_C171_CODEGEN_MODELS))
def test_varvol_codegen_trajectory_matches_interpreted(name):
    """End-to-end: an ODE run with the compiled varvol RHS+Jacobian matches the
    interpreted path (both correct — codegen is optimization, not a behavior
    change)."""

    def run_P(codegen):
        m = bngsim.Model.from_antimony_string(_C171_CODEGEN_MODELS[name])
        sim = bngsim.Simulator(m, method="ode", codegen=codegen)
        r = sim.run((0.0, 10.0), n_points=51)
        arr = np.asarray(r.species)
        arr = arr if arr.shape[0] == r.n_times else arr.T
        return arr[:, list(r.species_names).index("P")]

    assert float(np.max(np.abs(run_P(True) - run_P(False)))) < 1e-6
