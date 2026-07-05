"""GH #145 — the analytical Functional Jacobian (GH #76) is derived lazily.

The SymPy derivation is deferred off the model-load path and triggered on the
first ODE-solve setup that needs it (``jacobian`` ∈ {auto, analytical}). These
tests pin the deferral contract:

  * ``from_sbml`` / ``from_net`` do NOT derive — the model loads with the
    Jacobian incomplete, no SymPy imported, ``last_jacobian_sec == 0``;
  * the first ODE Simulator derives once and produces a **bit-identical**
    trajectory to the eager (pre-#145) path;
  * SSA / PSA / model-inspection never derive;
  * repeated solves / Simulators on one model derive at most once (the sentinel);
  * a clone of a warmed parent carries the Jacobian with no re-derive;
  * the eager escape hatch (``defer_jacobian=False`` / ``BNGSIM_EAGER_JACOBIAN``)
    restores derive-at-load;
  * the ≥256-species auto-codegen still runs, ordered AFTER the attach.

The Jacobian is consumed only by ODE solves (CVODE dense Jacobian, steady-state
Newton, codegen's analytical-Jacobian emitter), so deferral is answer-invariant.
"""

from __future__ import annotations

import subprocess
import sys

import bngsim
import bngsim._codegen as _codegen
import bngsim._jacobian as _jacobian
import numpy as np

# A small genuinely-Functional model (saturating Michaelis–Menten-shaped kinetic
# law): rate = k*S/(Km+S). Only Functional rate laws hit the SymPy derivation;
# mass-action and Michaelis–Menten carry the closed-form C++ Jacobian at build.
_FUNCTIONAL_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model>
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="c" initialConcentration="10"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
   <species id="P" compartment="c" initialConcentration="0"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="1" constant="true"/>
   <parameter id="Km" value="2" constant="true"/>
  </listOfParameters>
  <listOfReactions>
   <reaction id="r" reversible="false">
    <listOfReactants>
     <speciesReference species="S" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <listOfProducts>
     <speciesReference species="P" stoichiometry="1" constant="true"/>
    </listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><divide/>
        <apply><times/><ci>k</ci><ci>S</ci></apply>
        <apply><plus/><ci>Km</ci><ci>S</ci></apply>
      </apply>
    </math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""


def _functional_model():
    return bngsim.Model.from_sbml_string(_FUNCTIONAL_SBML)


# ─── Load does not derive ────────────────────────────────────────────────────


def test_from_sbml_does_not_derive():
    m = _functional_model()
    assert m._jac_attempted is False
    assert m._core.analytical_jacobian_complete is False
    assert m._jac_derive_sec == 0.0


def test_from_net_does_not_derive(data_dir):
    m = bngsim.Model.from_net(str(data_dir / "saturation.net"))
    assert m._jac_attempted is False
    assert m._core.analytical_jacobian_complete is False
    assert m._jac_derive_sec == 0.0


def test_load_path_imports_no_sympy():
    """A fresh process that only loads (never solves) must not import SymPy."""
    code = (
        "import sys, bngsim\n"
        f"m = bngsim.Model.from_sbml_string({_FUNCTIONAL_SBML!r})\n"
        "assert 'sympy' not in sys.modules, 'sympy leaked onto the load path'\n"
        "assert m._jac_attempted is False and m._jac_derive_sec == 0.0\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ─── First ODE solve derives once, bit-identically ───────────────────────────


def test_first_ode_setup_derives_once():
    m = _functional_model()
    sim = bngsim.Simulator(m, method="ode")
    assert m._jac_attempted is True
    assert m._core.analytical_jacobian_complete is True
    assert m._jac_derive_sec > 0.0
    assert sim.jacobian_strategy == "analytical"
    assert sim.last_jacobian_sec > 0.0


def test_lazy_trajectory_bit_identical_to_eager():
    """The deferred derivation must be answer-invariant: the lazy ODE trajectory
    is byte-for-byte identical to the eager (pre-#145) path."""
    span, n = (0.0, 50.0), 101
    lazy = np.asarray(bngsim.Simulator(_functional_model(), method="ode").run(span, n).species)
    eager_model = bngsim.Model.from_sbml_string(_FUNCTIONAL_SBML, defer_jacobian=False)
    assert eager_model._core.analytical_jacobian_complete is True  # derived at load
    eager = np.asarray(bngsim.Simulator(eager_model, method="ode").run(span, n).species)
    assert np.array_equal(lazy, eager)


def test_fd_mode_does_not_derive():
    m = _functional_model()
    sim = bngsim.Simulator(m, method="ode", jacobian="fd")
    assert m._jac_attempted is False
    assert sim.jacobian_strategy == "fd"


# ─── Non-ODE paths never derive ──────────────────────────────────────────────


def test_ssa_setup_does_not_derive(data_dir):
    m = bngsim.Model.from_net(str(data_dir / "saturation.net"))
    bngsim.Simulator(m, method="ssa", strict_ssa=False)
    assert m._jac_attempted is False
    assert m._core.analytical_jacobian_complete is False


def test_inspection_does_not_derive():
    """Loading a model and reading its structure (no Simulator) never derives."""
    m = _functional_model()
    _ = (m.n_species, m.n_reactions, m.species_names)
    assert m._jac_attempted is False
    assert m._core.analytical_jacobian_complete is False


# ─── Sentinel: at most one derivation per model ──────────────────────────────


def test_repeated_solves_derive_at_most_once(monkeypatch):
    calls = {"n": 0}
    real = _jacobian.attach_functional_jacobian

    def spy(core):
        calls["n"] += 1
        return real(core)

    monkeypatch.setattr(_jacobian, "attach_functional_jacobian", spy)
    m = _functional_model()
    sim = bngsim.Simulator(m, method="ode")
    sim.run((0.0, 10.0), 11)
    sim.run((0.0, 10.0), 11)
    # A second Simulator on the same (already-warmed) model must not re-derive.
    bngsim.Simulator(m, method="ode").run((0.0, 10.0), 11)
    assert calls["n"] == 1


def test_prepare_is_idempotent():
    m = _functional_model()
    assert m.prepare_analytical_jacobian() is True
    first = m._jac_derive_sec
    # Second call is a no-op: returns the same verdict, does not re-time/re-derive.
    assert m.prepare_analytical_jacobian() is True
    assert m._jac_derive_sec == first


def test_fd_fallback_does_not_retry(monkeypatch):
    """A model whose derivation falls back to FD (here: forced off) must not
    re-run the attempt on the next solve — the sentinel, not
    analytical_jacobian_complete, gates re-attempts."""
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", "0")
    calls = {"n": 0}
    real = _jacobian.attach_functional_jacobian

    def spy(core):
        calls["n"] += 1
        return real(core)

    monkeypatch.setattr(_jacobian, "attach_functional_jacobian", spy)
    m = _functional_model()
    bngsim.Simulator(m, method="ode").run((0.0, 5.0), 6)
    bngsim.Simulator(m, method="ode").run((0.0, 5.0), 6)
    assert m._jac_attempted is True
    assert m._core.analytical_jacobian_complete is False  # fell back to FD
    assert calls["n"] == 1  # attempted once, never retried


# ─── Clone carries the warmed Jacobian ───────────────────────────────────────


def test_clone_after_warm_carries_jacobian(monkeypatch):
    parent = _functional_model()
    parent.prepare_analytical_jacobian()
    assert parent._jac_attempted and parent._core.analytical_jacobian_complete

    calls = {"n": 0}
    real = _jacobian.attach_functional_jacobian

    def spy(core):
        calls["n"] += 1
        return real(core)

    monkeypatch.setattr(_jacobian, "attach_functional_jacobian", spy)
    clone = parent.clone()
    assert clone._jac_attempted is True
    assert clone._core.analytical_jacobian_complete is True  # re-compiled, no SymPy
    # Solving the clone must not re-derive (the C++ clone already carried the terms).
    bngsim.Simulator(clone, method="ode").run((0.0, 10.0), 11)
    assert calls["n"] == 0


def test_clone_of_cold_parent_derives_per_clone():
    """Documented cost: a clone of an UN-warmed parent inherits the unattempted
    sentinel and derives on its own first solve (hence warm-before-clone)."""
    parent = _functional_model()
    assert parent._jac_attempted is False
    clone = parent.clone()
    assert clone._jac_attempted is False
    bngsim.Simulator(clone, method="ode").run((0.0, 10.0), 11)
    assert clone._jac_attempted is True
    assert clone._core.analytical_jacobian_complete is True


# ─── Eager escape hatch ──────────────────────────────────────────────────────


def test_eager_env_var_derives_at_load(monkeypatch):
    monkeypatch.setenv("BNGSIM_EAGER_JACOBIAN", "1")
    m = _functional_model()
    assert m._jac_attempted is True
    assert m._core.analytical_jacobian_complete is True
    assert m._jac_derive_sec > 0.0


def test_eager_defer_false_derives_at_load():
    m = bngsim.Model.from_sbml_string(_FUNCTIONAL_SBML, defer_jacobian=False)
    assert m._jac_attempted is True
    assert m._core.analytical_jacobian_complete is True


def test_from_net_eager_defer_false(data_dir):
    m = bngsim.Model.from_net(str(data_dir / "saturation.net"), defer_jacobian=False)
    assert m._jac_attempted is True
    assert m._core.analytical_jacobian_complete is True


# ─── ≥256-species auto-codegen ordering (relocated to ODE-solve setup) ────────


def test_auto_codegen_runs_after_attach(monkeypatch):
    """The relocated large-model auto-codegen fires at ODE-solve setup, ordered
    AFTER the Jacobian attach — so the codegen emitter sees a complete analytical
    Jacobian (the load-time ordering invariant, preserved)."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)

    seen = {}
    real = _codegen.prepare_model_codegen

    def spy(model):
        seen["complete_at_codegen"] = bool(model._core.analytical_jacobian_complete)
        return real(model)

    monkeypatch.setattr(_codegen, "prepare_model_codegen", spy)
    m = _functional_model()
    assert not (m._codegen_so_path or m._codegen_c_source)  # nothing at load
    bngsim.Simulator(m, method="ode")
    assert m._codegen_so_path or m._codegen_c_source  # auto-codegen fired
    # Whichever backend ran, the attach preceded it (jac complete when codegen ran).
    if "complete_at_codegen" in seen:
        assert seen["complete_at_codegen"] is True


def test_below_threshold_does_not_auto_codegen():
    m = _functional_model()  # 2 species, default threshold 256
    bngsim.Simulator(m, method="ode")
    assert not (m._codegen_so_path or m._codegen_c_source)
