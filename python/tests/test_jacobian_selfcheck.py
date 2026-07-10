"""Regression tests for the analytical-Jacobian FD self-validation gate
(``NetworkModel::set_functional_jacobian``, ``src/model.cpp``).

GH #168: the gate's finite-difference cross-check used a perturbation step floored
at ``1.0`` (``h = 1e-5 * max(|y[j]|, 1.0)``). For *amount-scaled* models — species
≈ 1e-12, as produced when an SBML compartmental model (concentration × a tiny
compartment volume) is converted to ``.net`` — that floor makes the step ~1e7× the
species value, which (a) leaves the linear regime and (b) drives a nonnegative
species across the converter-emitted division guards ``if(X > 1e-300, expr/X, 0)``.
The downward half-step drops ``X`` below the guard, so the central difference reads
*exactly half* the true slope; both step sizes cross the guard identically, fooling
the Richardson-convergence guard, and the **correct** analytical Jacobian is
rejected → FD fallback → stiff integration fails (CVODE flag −4).

The fix makes the FD step **scale-relative** (``h = 1e-5 * |y[j]|``) and skips —
rather than rejects — a zero/too-small species where a central difference is not a
valid oracle. For ``|y[j]| ≥ 1`` the new step is byte-identical to the old one, so
O(1)-scale models (and the genuine-mismatch safety check below) are unchanged.

The synthetic reproducer ``tests/data/jac_selfcheck_amount_scaled.net`` was
committed with the bug report; these tests assert it now attaches a *correct*
analytical Jacobian under the default gate, and that the fix did not weaken
detection of a genuinely-wrong Jacobian at O(1) scale.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _jacobian as J

FIXTURE = "jac_selfcheck_amount_scaled.net"


def _clear_selfcheck_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run against the DEFAULT gate: no env overrides may relax or bypass it."""
    for var in (
        "BNGSIM_JAC_SELFCHECK_DENSE_MAX",
        "BNGSIM_JAC_SELFCHECK_SAMPLE",
        "BNGSIM_JAC_NO_SELFCHECK",
    ):
        monkeypatch.delenv(var, raising=False)


def test_amount_scaled_fixture_attaches_under_default_gate(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """GH #168 core regression: the amount-scaled fixture's CORRECT analytical
    Jacobian is accepted by the *default* (dense, exhaustive) self-check gate.

    Before the fix this returned ``False`` (oversized FD step → spurious mismatch
    → analytical Jacobian discarded → FD fallback)."""
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True
    assert m._core.analytical_jacobian_complete is True


def test_amount_scaled_fixture_attaches_under_sparse_gate(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Neither self-check path may spuriously reject: force the sparse+sampled
    path (``DENSE_MAX=0`` ⇒ ns>0 routes to the sampled CSC check) and confirm the
    same correct analytical Jacobian still attaches."""
    _clear_selfcheck_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_JAC_SELFCHECK_DENSE_MAX", "0")
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True
    assert m._core.analytical_jacobian_complete is True


def test_attached_jacobian_matches_finite_difference(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The Jacobian the gate now accepts is genuinely correct: it matches a
    properly scaled (step ∝ |y|) central finite difference of the engine RHS, and
    the one nontrivial entry equals the closed-form value ``d(k·[A])/d[A] = k``."""
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_net(str(data_dir / FIXTURE))
    assert m.prepare_analytical_jacobian() is True
    core = m._core
    assert core.analytical_jacobian_complete is True

    y = np.asarray(core.get_state(), dtype=float)
    ns = y.size
    # Column-major flat (jac[j*ns + i] = ∂f_i/∂y_j), per fill_dense_analytical_jacobian.
    flat = np.asarray(core._dense_analytical_jacobian(0.0, list(y)), dtype=float)
    jac = flat.reshape(ns, ns, order="F")  # jac[i, j] = ∂f_i/∂y_j

    def rhs(state: np.ndarray) -> np.ndarray:
        return np.asarray(core._eval_rhs(0.0, list(state)), dtype=float)

    fd = np.zeros((ns, ns))
    for j in range(ns):
        h = 1e-6 * abs(y[j]) if y[j] != 0.0 else 1e-18
        yp, ym = y.copy(), y.copy()
        yp[j] += h
        ym[j] -= h
        fd[:, j] = (rhs(yp) - rhs(ym)) / (2.0 * h)

    np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=1e-9)

    names = list(core.species_names)
    iA = next(i for i, n in enumerate(names) if n.startswith("A"))
    iB = next(i for i, n in enumerate(names) if n.startswith("B"))
    # k = 2.0 in the fixture; A is the mass-action reactant, B the product.
    assert jac[iB, iA] == pytest.approx(2.0, abs=1e-9)
    assert jac[iA, iA] == pytest.approx(-2.0, abs=1e-9)


# ─── Safety: the scale-relative step must NOT weaken genuine-mismatch detection ──
# At O(1) scale the new step is byte-identical to the old floored one, so a
# deliberately-wrong analytical Jacobian is still rejected. (The companion check in
# test_jacobian_symbolic.py::TestSelfCheckEndToEnd uses the same SBML model; this is
# a self-contained restatement guarding specifically against a #168 regression.)
_SBML_MM_DEG = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="mm_deg">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="2" constant="true"/>
      <parameter id="Km" value="5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/><apply><times/><ci>k</ci><ci>S</ci></apply>
            <apply><plus/><ci>Km</ci><ci>S</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _per_species_terms(core):
    ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    obs_groups = {n: [(int(s), float(f)) for s, f in g] for n, g in ctx["observables"]}
    smeta = {i: (bool(a), float(v)) for i, (a, v) in enumerate(ctx["species_meta"])}
    consts = set(ctx["constant_names"])
    terms = []
    for rxn in ctx["functional_reactions"]:
        t = J.build_per_species_terms(rxn["rate_expr"], func_map, obs_groups, smeta, consts)
        assert t is not None
        terms.append((rxn["rxn_idx"], False, [(int(j), e) for j, e in t]))
    return terms


def test_unit_scale_correct_jacobian_still_attaches(monkeypatch: pytest.MonkeyPatch):
    _clear_selfcheck_env(monkeypatch)
    core = bngsim.Model.from_sbml_string(_SBML_MM_DEG)._core
    terms = _per_species_terms(core)
    assert terms and terms[0][2], "expected a non-empty per-species Jacobian term"
    assert core.set_functional_jacobian(terms) is True
    assert core.analytical_jacobian_complete is True


def test_unit_scale_wrong_jacobian_still_rejected(monkeypatch: pytest.MonkeyPatch):
    _clear_selfcheck_env(monkeypatch)
    core = bngsim.Model.from_sbml_string(_SBML_MM_DEG)._core
    terms = _per_species_terms(core)
    ri, po, dl = terms[0]
    # Double the (correct) ∂rate/∂S — a ~100% divergence the gate must still catch.
    corrupted = [(ri, po, [(dl[0][0], f"2.0*({dl[0][1]})"), *dl[1:]])]
    assert core.set_functional_jacobian(corrupted) is False
    assert core.analytical_jacobian_complete is False
