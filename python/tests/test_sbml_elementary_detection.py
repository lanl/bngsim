"""Tests for the trivially-mass-action classifier in ``_sbml_loader.py``.

Issue #16 closed 2026-05-05 — the SBML/Antimony loader now emits a single
Elementary reaction for kinetic laws of the form

    [c *] [V *] k * x_1^{m_1} * x_2^{m_2} * ...

with ``c`` numeric, ``V`` an optional compartment-volume factor (required
when species are concentration-based and V != 1), ``k`` exactly one
parameter, and the species multiset matching the SBML reactants list.
Anything else (MM, Hill, function-def calls, multi-compartment, two
distinct parameter symbols multiplying, ...) stays Functional and the
existing per-species emission path takes over.

Cases here cover both the positive matches and the negative rejects, plus
the headline outcome — a mass-action Antimony model now matches an
external-FD reference at analytical-precision (~1e-8) instead of FD
precision (~1e-4).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import Model
from bngsim._codegen import generate_sens_from_model

pytest.importorskip("antimony")


def _rxn_types(m: Model) -> list[str]:
    return [r["type"] for r in m._core.codegen_data()["reactions"]]


def _rxn(m: Model, idx: int) -> dict:
    return m._core.codegen_data()["reactions"][idx]


# ─── Positive cases: must land Elementary ─────────────────────────────────────


def test_decay_lands_elementary():
    """``A -> B; k1*A`` → single Elementary reaction."""
    m = Model.from_antimony_string("A=10; B=0; k1=0.3; J0: A -> B; k1*A;")
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    assert r["reactants"] == [0]
    assert r["products"] == [1]
    assert r["stat_factor"] == 1.0
    assert r["function_name"] == "k1"
    # The kinetic-law function is no longer registered for mass-action laws.
    fnames = [f["name"] for f in m._core.codegen_data()["functions"]]
    assert "J0" not in fnames


def test_reversible_binding_lands_elementary():
    """``A + B <-> C`` with mass-action kf, kr → two Elementary reactions
    with the right reactant multiplicities and stoichiometric structure."""
    ant = """
        A=1; B=2; C=0; kf=0.1; kr=0.05;
        J0: A + B -> C; kf*A*B;
        J1: C -> A + B; kr*C;
    """
    m = Model.from_antimony_string(ant)
    assert _rxn_types(m) == ["elementary", "elementary"]
    r0, r1 = _rxn(m, 0), _rxn(m, 1)
    assert sorted(r0["reactants"]) == [0, 1]
    assert r0["products"] == [2]
    assert r0["function_name"] == "kf"
    assert r0["stat_factor"] == 1.0
    assert r1["reactants"] == [2]
    assert sorted(r1["products"]) == [0, 1]
    assert r1["function_name"] == "kr"
    assert r1["stat_factor"] == 1.0


def test_dimerization_multiplicity_two():
    """``2 A -> B; k*A*A`` → reactant multiset is {A, A} so the BNGsim
    reactant index list has two entries for A."""
    m = Model.from_antimony_string("A=5; B=0; k=0.2; J0: 2 A -> B; k*A*A;")
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    assert r["reactants"] == [0, 0]
    assert r["products"] == [1]
    assert r["function_name"] == "k"


def test_dimerization_pow_form():
    """``2 A -> B; k*A^2`` is the same model as ``k*A*A`` once flattened —
    AST_POWER with a positive integer exponent expands."""
    m = Model.from_antimony_string("A=5; B=0; k=0.2; J0: 2 A -> B; k*A^2;")
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    assert r["reactants"] == [0, 0]


def test_numeric_constant_folds_into_stat_factor():
    """``2*k*A`` keeps a single Elementary reaction with stat_factor=2.0
    (constant numeric leaves are absorbed into the stat factor)."""
    m = Model.from_antimony_string("A=10; B=0; k=0.3; J0: A -> B; 2*k*A;")
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    assert r["stat_factor"] == 2.0
    assert r["function_name"] == "k"


def test_sir_infection_reaction_lands_elementary():
    """SIR's ``S -> I; beta*S*I`` has I in the kinetic law but not in
    ``<listOfReactants>``. The classifier accepts this by computing
    ``P_count[I] = c - b + a = 1 - 0 + 1 = 2``: I appears once in
    BNGsim reactants and twice in BNGsim products, so ydot[I] is
    ``-rate + 2*rate = +rate`` — exactly what SBML's net stoichiometry
    of +1 for I says it should be.
    """
    m = Model.from_antimony_string(
        "model sir; S=1000; I=10; Rec=0; b=1e-4; g=0.1;\n"
        "R1: S -> I; b*S*I;\n"
        "R2: I -> Rec; g*I;\n"
        "end"
    )
    assert _rxn_types(m) == ["elementary", "elementary"]
    r0 = _rxn(m, 0)
    # reactants = [S, I] in some order; products = [I, I].
    assert sorted(r0["reactants"]) == [0, 1]
    assert r0["products"] == [1, 1]
    assert r0["function_name"] == "b"


def test_constant_rate_reaction_stays_functional():
    """``S -> P; k`` has no S in the kinetic law, so consumption can't
    be expressed via Elementary's ``rate * Π y[reactant]`` form. Falls
    through to per-species Functional emission."""
    m = Model.from_antimony_string("S=10; P=0; k=0.5; J0: S -> P; k;")
    assert all(t == "functional" for t in _rxn_types(m))


def test_first_order_decay_with_higher_stoich_stays_functional():
    """``2 A -> B; k*A`` — kinetic law is first-order but stoich consumes
    2 As. Cannot be expressed as a single Elementary reaction (the
    target product count for A would be ``0 - 2 + 1 = -1``)."""
    m = Model.from_antimony_string("A=10; B=0; k=0.3; J0: 2 A -> B; k*A;")
    assert all(t == "functional" for t in _rxn_types(m))


def test_enzyme_catalysis_keeps_e_in_both_lists():
    """``E + S -> E + P; k*E*S`` — the catalyst E appears in both reactant
    and product lists with multiplicity 1, so the Elementary RHS sums to
    zero net change for E (BNGsim does not pre-cancel reactant/product
    multisets, and CVODES integrates ``ydot[E] = +rate - rate = 0``)."""
    m = Model.from_antimony_string("E=1; S=10; P=0; k=0.1; J0: E + S -> E + P; k*E*S;")
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    assert sorted(r["reactants"]) == [0, 1]  # [E, S]
    assert sorted(r["products"]) == [0, 2]  # [E, P]
    assert r["function_name"] == "k"


# ─── V handling and multi-compartment ────────────────────────────────────────


def test_multi_compartment_v_eq_1_lifts_to_elementary():
    """When all involved compartments have V=1 (BioModels' common shape:
    nominal medium / cell / cellsurface compartments), multi-compartment
    reactions whose kinetic laws include a `* compartment` factor lift to
    Elementary. The compartment factor folds to 1.0 (V=1) and the
    per-species `/V` step is a no-op, so the Elementary form reproduces
    SBML's dynamics exactly.
    """
    ant = """
        compartment cellsurface = 1;
        compartment medium = 1;
        species E in cellsurface = 1;
        species L in medium = 10;
        species C in cellsurface = 0;
        kon = 0.1;
        J0: E + L -> C; kon * E * L * cellsurface;
    """
    m = Model.from_antimony_string(ant)
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    # ``cellsurface`` factor folds to 1.0; sf stays at 1.
    assert r["stat_factor"] == 1.0
    assert r["function_name"] == "kon"


def test_compartment_factor_with_v_neq_1_cancels():
    """For a single-compartment reaction with V != 1 where the kinetic law
    includes ``compartment * k * A``, the loader's per-species ``/V``
    division and the kinetic law's ``* V`` factor cancel: stat_factor = 1.
    """
    ant = """
        compartment cell = 2.5;
        species A in cell = 10;
        species B in cell = 0;
        k = 0.3;
        J0: A -> B; cell * k * A;
    """
    m = Model.from_antimony_string(ant)
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    # sf = numeric_const * V_cell / V_common = 1 * 2.5 / 2.5 = 1.0
    assert abs(r["stat_factor"] - 1.0) < 1e-12
    assert r["function_name"] == "k"


def test_v_neq_1_without_compartment_factor_lifts_with_v_in_stat_factor():
    """When V != 1 and the kinetic law is ``k * A`` (no compartment factor —
    Antimony's default emission for non-default volumes), the loader's
    per-species ``/V`` step still applies. The classifier folds 1/V into
    stat_factor: sf = 1/V_common, so the Elementary form
    ``rate = sf * k * A`` reproduces SBML's ``rate / V = k*A/V`` exactly.
    """
    ant = """
        compartment cell = 2.5;
        species A in cell = 10;
        species B in cell = 0;
        k = 0.3;
        J0: A -> B; k * A;
    """
    m = Model.from_antimony_string(ant)
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    assert abs(r["stat_factor"] - 1.0 / 2.5) < 1e-12
    assert r["function_name"] == "k"


# ─── Derived rate-constant synthesis (loader) ────────────────────────────────


def test_two_constant_param_product_synthesizes_derived_rate():
    """``kt * Bmax * cell`` (two constant SBML parameters multiplying) gets
    a derived parameter ``_rateLaw_<rid>`` synthesized in the loader,
    with ``is_const=False`` and ``expression="kt * Bmax"`` — so codegen
    can chain-rule through it."""
    ant = """
        compartment cell = 1;
        species EpoR in cell = 0;
        kt = 0.0329;
        Bmax = 516;
        J_synth: -> EpoR; kt * Bmax * cell;
    """
    m = Model.from_antimony_string(ant)
    assert _rxn_types(m) == ["elementary"]
    r = _rxn(m, 0)
    assert r["function_name"].startswith("_rateLaw_")

    # Derived parameter must be present with the right expression and
    # is_const=False so the chain rule fires in generate_sens_from_model.
    derived = next(
        p for p in m._core.codegen_data()["parameters"] if p["name"] == r["function_name"]
    )
    assert derived["is_const"] is False
    assert "kt" in derived["expression"] and "Bmax" in derived["expression"]
    assert abs(derived["value"] - 0.0329 * 516) < 1e-9


def test_chain_rule_sens_through_synthesized_derived():
    """Sensitivity of a primary parameter that participates in a synthesized
    derived rate constant (``kt`` inside ``_rateLaw_J_synth = kt * Bmax``)
    must match an external FD reference at analytical precision —
    confirming the loader's synthesis combined with the codegen chain
    rule reproduces ``∂rate/∂kt = Bmax * sf * Π y[reactant]``."""
    ant = """
        compartment cell = 1;
        species EpoR in cell = 0;
        kt = 0.0329;
        Bmax = 516;
        J_synth: -> EpoR; kt * Bmax * cell;
        J_decay: EpoR -> ; kt * EpoR * cell;
    """

    def _build():
        return Model.from_antimony_string(ant)

    sim = bngsim.Simulator(_build(), method="ode", sensitivity_params=["kt"])
    r = sim.run(t_span=(0, 60), n_points=21, rtol=1e-12, atol=1e-14)

    eps = 0.0329 * 1e-4

    def _traj(val: float) -> np.ndarray:
        mm = _build()
        mm.set_param("kt", val)
        return (
            bngsim.Simulator(mm, method="ode")
            .run(t_span=(0, 60), n_points=21, rtol=1e-12, atol=1e-14)
            .species
        )

    fd = (_traj(0.0329 + eps) - _traj(0.0329 - eps)) / (2.0 * eps)
    err = np.abs(r.sensitivities[..., 0] - fd).max()
    rel = err / max(np.abs(fd).max(), 1e-30)
    assert rel < 1e-6, f"chain-rule sens vs FD: max abs err = {err:.3e}, rel = {rel:.3e}"


# ─── Negative cases: must stay Functional ─────────────────────────────────────


def test_michaelis_menten_stays_functional():
    """MM kinetics have ``+`` in the denominator and ``/`` at the root —
    flattening rejects on the first non-Times/Power node."""
    m = Model.from_antimony_string("S=10; P=0; Vmax=1; Km=2; J0: S -> P; Vmax*S/(Km + S);")
    # Functional emission produces one reaction per affected species.
    assert all(t == "functional" for t in _rxn_types(m))


def test_hill_stays_functional():
    """Hill: ``Vmax*S^n/(K^n+S^n)`` — the AST_DIVIDE at the root rejects."""
    m = Model.from_antimony_string(
        "S=10; P=0; Vmax=1; K=2; n=2; J0: S -> P; Vmax*S^n/(K^n + S^n);"
    )
    assert all(t == "functional" for t in _rxn_types(m))


def test_function_definition_call_stays_functional():
    """User-defined function calls in the kinetic law (AST_FUNCTION) are
    not in the accepted-leaves list — flatten rejects."""
    ant = """
        function myrate(s, kk) kk*s; end;
        S=10; P=0; k=0.1;
        J0: S -> P; myrate(S, k);
    """
    m = Model.from_antimony_string(ant)
    assert all(t == "functional" for t in _rxn_types(m))


# ─── Headline outcome: end-to-end analytical sens precision ───────────────────


def test_mass_action_antimony_model_gets_analytical_sens():
    """The auto-trigger now produces analytical sens RHS for a mass-action
    Antimony model. Compare ``sim.run(...).sensitivities`` against an
    externally-computed centered FD reference: analytical sens is exact
    up to ODE tolerance (~1e-8 here), versus the ~1e-4 ceiling of FD."""
    ant_text = "model decay; A=10; B=0; k1=0.3; J0: A -> B; k1*A; end"

    # Analytical sens via auto-trigger.
    m = Model.from_antimony_string(ant_text)
    # generate_sens_from_model returning a string is the contract: every
    # reaction is Elementary so analytical sens is supported.
    assert generate_sens_from_model(m) is not None

    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"])
    r = sim.run(t_span=(0, 5), n_points=51, rtol=1e-12, atol=1e-14)
    sens_codegen = r.sensitivities[..., 0]

    # External centered FD reference (no codegen).
    eps = 1e-5

    def _traj(k1: float) -> np.ndarray:
        mm = Model.from_antimony_string(ant_text)
        mm.set_param("k1", k1)
        return (
            bngsim.Simulator(mm, method="ode")
            .run(t_span=(0, 5), n_points=51, rtol=1e-12, atol=1e-14)
            .species
        )

    fd = (_traj(0.3 + eps) - _traj(0.3 - eps)) / (2.0 * eps)

    err = np.abs(sens_codegen - fd).max()
    # FD is the precision floor; analytical typically agrees to ~1e-8 with
    # high-order centered differences. We assert the analytical-precision
    # band — anything looser would mask FD-vs-analytical equivalence.
    assert err < 1e-6, f"analytical sens vs external FD: max abs err = {err:.3e}"
