"""Codegen observable/expression output evaluator (GH #136).

For models with many observables/expressions, evaluating them at every
trajectory output row through the interpreted ExprTk path dominated wall time —
far exceeding the ODE integration. The fix:

  * the recorder no longer re-evaluates every function 3× per row (one
    ``evaluate_functions`` pass now caches the values; the assignment-rule
    copy-back and the expression recording both read the cache), and the
    per-row O(n_func × n_species) assignment-rule name scan is resolved once;
  * when a model is codegen-compiled, the warm integration path calls a
    compiled ``bngsim_codegen_outputs`` function to fill the observable and
    expression buffers, instead of the interpreted pass.

These tests pin (1) the topological function-emission order the codegen RHS and
output evaluator both depend on, (2) that the compiled output evaluator matches
the interpreted recorder, and (3) that the source emitter declines the cases it
cannot compile.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap

import bngsim
import numpy as np
import pytest
from bngsim._codegen import (
    _codegen_emit_flags,
    _topological_function_order,
    generate_outputs_from_model,
    prepare_codegen,
)

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

# ─── Topological function-emission order ─────────────────────────────────────


def test_topological_order_forward_reference():
    """``a := b`` declared before ``b`` must be emitted after it."""
    funcs = [{"name": "a", "expression": "b"}, {"name": "b", "expression": "0.5*S"}]
    assert _topological_function_order(funcs) == [1, 0]


def test_topological_order_already_sorted_is_identity():
    """A model already in dependency order keeps its order (stable sort), so the
    emitted C stays byte-identical for the entire real corpus."""
    funcs = [{"name": "b", "expression": "0.5*S"}, {"name": "a", "expression": "b"}]
    assert _topological_function_order(funcs) == [0, 1]


def test_topological_order_chain():
    """c ← b ← a: each function follows the one it references."""
    funcs = [
        {"name": "a", "expression": "b + 1"},
        {"name": "b", "expression": "c * 2"},
        {"name": "c", "expression": "S"},
    ]
    order = _topological_function_order(funcs)
    pos = {idx: k for k, idx in enumerate(order)}
    assert pos[2] < pos[1] < pos[0]  # c before b before a


def test_topological_order_no_dependencies_is_identity():
    funcs = [{"name": f"f{i}", "expression": "S"} for i in range(5)]
    assert _topological_function_order(funcs) == [0, 1, 2, 3, 4]


def test_topological_order_cycle_falls_back_to_declaration_order():
    """A malformed self-referential cycle still yields every index exactly once
    (declaration-order fallback), never dropping a function."""
    funcs = [{"name": "a", "expression": "b"}, {"name": "b", "expression": "a"}]
    order = _topological_function_order(funcs)
    assert sorted(order) == [0, 1]


# ─── SBML fixtures ───────────────────────────────────────────────────────────


def _decay_with_rules_sbml() -> str:
    """Two species, two assignment-rule expressions (one forward-referenced), a
    decay reaction whose rate is a function. dS/dt = -a = -(0.5·S) ⇒ S0·e^-0.5t.

    The assignment rules are declared in REVERSE dependency order (``a := b``
    before ``b := 0.5·S``) so the codegen path must topologically reorder them.
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay_rules">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="a" constant="false"/>
      <parameter id="b" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="a">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>b</ci></math>
      </assignmentRule>
      <assignmentRule variable="b">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>0.5</cn><ci>S</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>a</ci></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _run(sbml: str, *, codegen: bool, monkeypatch):
    if codegen:
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    else:
        monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
    m = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(m, method="ode")
    m.reset()
    r = sim.run(t_span=(0.0, 4.0), n_points=21, rtol=1e-9, atol=1e-12)
    return r


# ─── Compiled output evaluator vs interpreted recorder ───────────────────────


def test_codegen_outputs_match_interpreted(monkeypatch):
    """The compiled output evaluator must reproduce the interpreted recorder's
    observables (exactly) and expressions (within solver/FMA tolerance)."""
    sbml = _decay_with_rules_sbml()
    r_cg = _run(sbml, codegen=True, monkeypatch=monkeypatch)
    r_et = _run(sbml, codegen=False, monkeypatch=monkeypatch)

    o_cg, o_et = np.asarray(r_cg.observables), np.asarray(r_et.observables)
    e_cg, e_et = np.asarray(r_cg.expressions), np.asarray(r_et.expressions)
    assert o_cg.shape == o_et.shape
    assert e_cg.shape == e_et.shape
    # Observables are linear combinations of species — bit-identical between the
    # two paths (same compiled RHS trajectory, same coefficients).
    assert np.array_equal(o_cg, o_et)
    # Expressions go through arithmetic that may FMA-contract under -O3; tolerate
    # machine-epsilon, reject anything resembling a real divergence.
    np.testing.assert_allclose(e_cg, e_et, rtol=1e-9, atol=1e-12)


def test_codegen_forward_reference_integrates_correctly(monkeypatch):
    """The topological emission order makes the codegen RHS resolve a
    forward-referenced assignment rule; the trajectory matches S0·e^(-0.5t)
    rather than diverging on an uninitialised func[] slot."""
    r = _run(_decay_with_rules_sbml(), codegen=True, monkeypatch=monkeypatch)
    # Confirm codegen actually engaged (else this would silently test the
    # interpreted path).
    assert r is not None
    t = np.asarray(r.time)
    s = np.asarray(r.species)[:, list(r.species_names).index("S")]
    exact = 10.0 * np.exp(-0.5 * t)
    assert float(np.max(np.abs(s - exact))) < 1e-5


def test_codegen_engages_for_forward_reference_model(monkeypatch):
    """Guard: with the threshold forced to 1, the decay model is codegen-
    compiled — so the matching/forward-reference tests above exercise the
    compiled output evaluator, not the interpreted fallback.

    GH #145: the large-model auto-codegen moved off the load path to the ODE-solve
    setup, so constructing the ODE Simulator (not from_sbml_string alone) is what
    triggers it and stamps the prepared output back onto the model."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    m = bngsim.Model.from_sbml_string(_decay_with_rules_sbml())
    bngsim.Simulator(m, method="ode")
    assert getattr(m, "_codegen_so_path", "") or getattr(m, "_codegen_c_source", "")


# ─── Source-emitter gating ───────────────────────────────────────────────────


def test_generate_outputs_emits_symbol_and_copy_loops(monkeypatch):
    monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")  # don't auto-compile on load
    m = bngsim.Model.from_sbml_string(_decay_with_rules_sbml())
    src = generate_outputs_from_model(m)
    assert src is not None
    assert "int bngsim_codegen_outputs(" in src
    assert "obs_out[_i] = obs[_i];" in src
    assert "func_out[_i] = func[_i];" in src


def test_generate_outputs_declines_rateof_model(monkeypatch):
    """A rateOf function body needs the live dx/dt the RHS probe publishes, so
    the standalone output evaluator declines (the simulator keeps interpreting
    these rare models)."""
    monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
    rateof_sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="rateof">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="5"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="r" constant="false"/>
      <parameter id="k" value="0.2" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="r">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <csymbol encoding="text"
              definitionURL="http://www.sbml.org/sbml/symbols/rateOf">rateOf</csymbol>
            <ci>S</ci>
          </apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
    m = bngsim.Model.from_sbml_string(rateof_sbml)
    assert generate_outputs_from_model(m) is None
    # The .net cache-key predicate must agree with the emitter's decline so the
    # interpreted recorder is kept for these models (GH #163): want_outputs is
    # False regardless of the Jacobian strategy.
    assert _codegen_emit_flags(m, emit_jac=True)[1] is False
    assert _codegen_emit_flags(m, emit_jac=False)[1] is False


# ─── .net codegen path carries the output evaluator (GH #163) ─────────────────
#
# bngsim_codegen_outputs was emitted only by the model-based path; a model loaded
# via Model.from_net + Simulator(codegen=True) — the genome-scale workflow — got a
# compiled RHS (+ Jacobian since #162) but the *interpreted* per-row recording.
# These pin that the .net codegen now appends the compiled evaluator, that it is
# emitted independently of the Jacobian strategy, that recorded values match the
# interpreted recorder, and that the decline-cleanly cases still fall back.


def _chain_net_with_obs_and_func(n_species: int = 3) -> str:
    """An ``n_species`` elementary chain ``A1->A2->...->An`` with one observable
    group per species and a function over the first two.

    The function ``frac() = scale*A2tot/(A1tot+A2tot+1)`` references observables, so
    the compiled evaluator must fill obs[] first and read them in the func[] block
    — exercising the obs→func dependency the per-row recorder relies on. One group
    per species scales the per-row obs[] copy loop for the medium/large cases. All
    reactions are mass-action (Elementary), so the model also has a complete
    analytical Jacobian (lets the same fixture cover the jac-strategy decoupling).
    """
    assert n_species >= 3
    lines = ["begin parameters", "    1 k 0.4", "    2 scale 1.5", "end parameters"]
    lines.append("begin species")
    lines.append("    1 A1() 12.0")
    lines += [f"    {i} A{i}() 0" for i in range(2, n_species + 1)]
    lines.append("end species")
    lines += ["begin functions", "    1 frac() scale*A2tot/(A1tot + A2tot + 1.0)", "end functions"]
    lines.append("begin reactions")
    lines += [f"    {i} {i} {i + 1} k #r{i}" for i in range(1, n_species)]
    lines.append("end reactions")
    lines.append("begin groups")
    lines += [f"    {i} A{i}tot {i}" for i in range(1, n_species + 1)]
    lines += ["end groups", ""]
    return "\n".join(lines)


def _no_obs_net() -> str:
    """A bare chain with no observable groups and no functions — the no-obs-no-func
    decline case for the output evaluator."""
    return "\n".join(
        [
            "begin parameters",
            "    1 k 0.1",
            "end parameters",
            "begin species",
            "    1 A() 5",
            "    2 B() 0",
            "end species",
            "begin reactions",
            "    1 1 2 k #r1",
            "end reactions",
            "begin groups",
            "end groups",
            "",
        ]
    )


def _write(tmp_path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _so_has_symbol(so_path, name: str) -> bool:
    lib = ctypes.CDLL(str(so_path))
    try:
        getattr(lib, name)
        return True
    except AttributeError:
        return False


@needs_cc
def test_net_codegen_path_appends_outputs(tmp_path):
    # prepare_codegen, given the built model, appends bngsim_codegen_outputs onto
    # the .net RHS in one .so — so a .net-loaded model gets the compiled per-row
    # recorder, not the interpreted fallback.
    net = _write(tmp_path, "chain.net", _chain_net_with_obs_and_func())
    m = bngsim.Model.from_net(net)
    m.prepare_analytical_jacobian()

    so = prepare_codegen(net, m)
    assert _so_has_symbol(so, "bngsim_codegen_rhs")
    assert _so_has_symbol(so, "bngsim_codegen_outputs")

    # The same .net WITHOUT a model stays RHS-only (historical behavior) and gets a
    # DISTINCT cache key, so it never collides with the outputs-carrying .so.
    so_rhs_only = prepare_codegen(net)
    assert not _so_has_symbol(so_rhs_only, "bngsim_codegen_outputs")
    assert so_rhs_only != so


@needs_cc
def test_net_codegen_outputs_independent_of_jacobian_strategy(tmp_path):
    # The key GH #163 decoupling: the output evaluator is emitted for EVERY
    # jacobian strategy, while the analytical Jacobian is gated on emit_jac. With
    # emit_jac=False (jacobian="fd"/"jax"), the .so must carry the outputs symbol
    # but NOT the Jacobian symbol.
    net = _write(tmp_path, "chain.net", _chain_net_with_obs_and_func())
    m = bngsim.Model.from_net(net)
    m.prepare_analytical_jacobian()

    so_fd = prepare_codegen(net, m, emit_jac=False)
    assert _so_has_symbol(so_fd, "bngsim_codegen_outputs")
    assert not _so_has_symbol(so_fd, "bngsim_codegen_jac")
    assert not _so_has_symbol(so_fd, "bngsim_codegen_jac_sparse")

    # With the Jacobian wanted, both symbols coexist in a distinct .so.
    so_jac = prepare_codegen(net, m, emit_jac=True)
    assert _so_has_symbol(so_jac, "bngsim_codegen_outputs")
    assert _so_has_symbol(so_jac, "bngsim_codegen_jac")  # small/dense model
    assert so_jac != so_fd


@needs_cc
def test_net_codegen_declines_no_observables(tmp_path):
    # A .net with no observables and no functions declines cleanly — RHS only, no
    # output evaluator (the simulator keeps the interpreted recorder).
    net = _write(tmp_path, "bare.net", _no_obs_net())
    m = bngsim.Model.from_net(net)
    assert _codegen_emit_flags(m, emit_jac=True)[1] is False

    so = prepare_codegen(net, m)
    assert _so_has_symbol(so, "bngsim_codegen_rhs")
    assert not _so_has_symbol(so, "bngsim_codegen_outputs")


def _run_net(net: str, *, codegen: bool, jacobian: str, monkeypatch):
    if codegen:
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    else:
        monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
    m = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(
        m,
        method="ode",
        jacobian=jacobian,
        codegen=True if codegen else None,
        net_path=net,
    )
    r = sim.run(t_span=(0.0, 6.0), n_points=31, rtol=1e-9, atol=1e-12)
    return sim, m, r


@needs_cc
@pytest.mark.parametrize("jacobian", ["analytical", "fd"])
@pytest.mark.parametrize("n_species", [3, 60, 200], ids=["small", "medium", "large"])
def test_net_codegen_true_outputs_match_interpreted(tmp_path, monkeypatch, jacobian, n_species):
    # The genome-scale workflow: load from .net, Simulator(codegen=True). The
    # compiled per-row recorder must reproduce the interpreted recorder's
    # observables (exactly — linear in species) and expressions (to FP tolerance),
    # across small/medium/large synthetic models and for both an analytical-Jacobian
    # and an FD run (outputs are emitted independently of the Jacobian strategy).
    net = _write(tmp_path, "chain.net", _chain_net_with_obs_and_func(n_species))

    sim_cg, m_cg, r_cg = _run_net(net, codegen=True, jacobian=jacobian, monkeypatch=monkeypatch)
    _, _, r_et = _run_net(net, codegen=False, jacobian=jacobian, monkeypatch=monkeypatch)

    # The compiled artifact actually carries the symbol the warm recording loop
    # calls. The default cc backend emits a .so (inspect its symbol table); the
    # MIR backend (BNGSIM_CODEGEN_JIT=mir) JITs the same C source in-process, so
    # there is no .so — assert the source it JITs defines the output function.
    backend = sim_cg.codegen_backend
    assert backend in ("cc", "mir")
    if backend == "cc":
        assert _so_has_symbol(sim_cg._codegen_so_path, "bngsim_codegen_outputs")
    else:
        assert "bngsim_codegen_outputs" in sim_cg._codegen_c_source

    o_cg, o_et = np.asarray(r_cg.observables), np.asarray(r_et.observables)
    e_cg, e_et = np.asarray(r_cg.expressions), np.asarray(r_et.expressions)
    assert o_cg.shape == o_et.shape and o_cg.shape[1] == n_species  # one group per species
    assert e_cg.shape == e_et.shape and e_cg.shape[1] == 1
    assert np.array_equal(o_cg, o_et)
    np.testing.assert_allclose(e_cg, e_et, rtol=1e-9, atol=1e-12)


_DET_CHILD = textwrap.dedent(
    """
    import os, sys, hashlib
    os.environ["BNGSIM_NO_CODEGEN"] = "1"
    import bngsim
    from bngsim._codegen import generate_combined_c
    m = bngsim.Model.from_net(sys.argv[1])
    m.prepare_analytical_jacobian()
    src, _ = generate_combined_c(sys.argv[1], m, emit_jac=True, emit_outputs=True)
    assert "int bngsim_codegen_outputs(" in src, "expected output evaluator"
    sys.stdout.write(hashlib.sha256(src.encode()).hexdigest())
    """
)


@needs_cc
def test_net_codegen_outputs_pythonhashseed_independent(tmp_path):
    # Byte-determinism: the combined .net source (RHS + Jacobian + output evaluator)
    # must be identical across PYTHONHASHSEED values — the emitter sorts every
    # set/dict iteration, so the obs[]/func[] copy loops never reorder.
    net = _write(tmp_path, "chain.net", _chain_net_with_obs_and_func())

    def _hash_with_seed(seed: int) -> str:
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        proc = subprocess.run(
            [sys.executable, "-c", _DET_CHILD, net],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, f"child failed (seed={seed}):\n{proc.stderr}"
        out = proc.stdout.strip()
        assert len(out) == 64, f"unexpected child output (seed={seed}): {proc.stdout!r}"
        return out

    hashes = {seed: _hash_with_seed(seed) for seed in (0, 1, 2, 3)}
    assert len(set(hashes.values())) == 1, f"emit varies with PYTHONHASHSEED: {hashes}"
