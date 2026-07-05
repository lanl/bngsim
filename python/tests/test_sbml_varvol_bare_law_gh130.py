"""ODE correctness for a *bare* mass-action law in a variable-volume compartment (#130).

A mass-action reaction may be written two ways in a compartment ``Cc``:

  * the BNG / COPASI convention ``Cc·k·A·B`` (the compartment appears as an
    explicit factor, power ``p = 1``), or
  * the bare concentration-rate form ``k·A·B`` (no compartment factor,
    ``p = 0``) — what antimony emits by default.

bngsim's Elementary path folds a single scalar ``sf`` for a mass-action
reaction:  ``sf = numeric · Π_c V_static[c]^p_c / V_static[storage]``. The
storage compartment's net volume power in ``sf`` is ``p − 1``. For ``p = 1`` it
is zero, so ``sf`` is volume-independent and stays correct as ``V`` changes. For
the bare law ``p = 0``, ``sf`` bakes ``V_static^{-1}`` — once the compartment
volume moves (an event resize or a rate rule), the amount-rate is wrong by
``V_static / V_live`` and the ODE diverges from RoadRunner (GH #130).

The fix routes such a reaction (all-hOSU=false, single storage compartment,
non-zero net varvol power) to the **Functional** path, where the live
compartment symbol stays inside the law (``law / V_live``) and reproduces SBML's
``d[s]/dt = law / V_live(t)`` exactly for any power. The ``p = 1`` convention
keeps cancelling and stays on the Elementary path (so the GH #81 SSA varvol
support is untouched). Reactions in a *constant*-volume compartment are
unaffected (the guard only fires when a varvol compartment survives in ``sf``).

Oracles are dependency-free (a SciPy reference integration of the amount ODE),
plus an optional libRoadRunner cross-check — the engine the issue's repro used.
"""

import bngsim
import numpy as np
import pytest

K = 0.004
A0 = B0 = 50.0  # initial CONCENTRATIONS
V0 = 2.0
TE = 3.3  # event time, deliberately off the integer sample grid
T_END = 8.0
N_POINTS = 9  # grid 0,1,...,8 — TE is between samples


def _bimolecular_sbml(*, vary):
    """``A + B -> P`` with the bare law ``k·A·B`` in a variable-volume ``Cc``.

    ``vary='event'`` doubles ``Cc`` at ``t > TE``; ``vary='rate_rule'`` grows
    ``Cc`` continuously at ``dCc/dt = 0.3``.
    """
    if vary == "event":
        dynamics = f"""
    <listOfEvents>
      <event id="resize" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/><csymbol encoding="text"
              definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
              <cn>{TE}</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="Cc">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><ci>Cc</ci><cn>2</cn></apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>"""
    elif vary == "rate_rule":
        dynamics = """
    <listOfRules>
      <rateRule variable="Cc">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.3</cn></math>
      </rateRule>
    </listOfRules>"""
    else:  # pragma: no cover - test programming error
        raise ValueError(vary)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="bare_law_{vary}">
    <listOfCompartments>
      <compartment id="Cc" size="{V0}" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="Cc" initialConcentration="{A0}"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="Cc" initialConcentration="{B0}"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="Cc" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="{K}" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="J1" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><apply><times/><ci>A</ci><ci>B</ci></apply></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>{dynamics}
  </model>
</sbml>"""


# Same model, BNG ``Cc·k·A·B`` convention (p=1) — must keep cancelling.
def _p1_event_sbml():
    return _bimolecular_sbml(vary="event").replace(
        "<apply><times/><ci>k</ci><apply><times/><ci>A</ci><ci>B</ci></apply></apply>",
        "<apply><times/><ci>Cc</ci><apply><times/><ci>k</ci>"
        "<apply><times/><ci>A</ci><ci>B</ci></apply></apply></apply>",
    )


def _run_P(sbml):
    model = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, T_END), n_points=N_POINTS, rtol=1e-11, atol=1e-13
    )
    t = np.asarray(r.time)
    P = np.asarray(r.species)[:, list(r.species_names).index("P")]
    return t, P


def _event_amount_oracle(t, *, v_rate_post):
    """Reference [P](t) for the event-resize bare law.

    Integrates AMOUNTS (continuous across the resize), then reports the
    concentration ``[P] = amount_P / V_live``. ``v_rate_post`` selects the
    volume used in the rate AFTER the resize: ``2*V0`` is correct (the live
    volume); ``V0`` reproduces the pre-fix bug (sf baked the static volume).
    Pre-resize both use ``V0`` (where the bug did not yet bite).
    """
    from scipy.integrate import solve_ivp

    aA0 = A0 * V0  # initial amounts
    aB0 = B0 * V0

    def rhs(V):
        def f(_t, y):
            r = K * y[0] * y[1] / V**2  # k·[A]·[B] = k·aA·aB/V²  (amount/time)
            return [-r, -r, r]

        return f

    pre = [p for p in t.tolist() if p <= TE]
    sol_pre = solve_ivp(
        rhs(V0), (0, TE), [aA0, aB0, 0.0], t_eval=pre + [TE], rtol=1e-12, atol=1e-14
    )
    aA, aB, aP = sol_pre.y[:, -1]  # amounts continuous across the event
    post = [p for p in t.tolist() if p > TE]
    sol_post = solve_ivp(
        rhs(v_rate_post),
        (TE, T_END),
        [aA, aB, aP],
        t_eval=[TE] + post,
        rtol=1e-12,
        atol=1e-14,
    )
    out = np.empty_like(t)
    out[t <= TE] = np.interp(t[t <= TE], sol_pre.t, sol_pre.y[2]) / V0
    out[t > TE] = np.interp(t[t > TE], sol_post.t, sol_post.y[2]) / (2 * V0)
    return out


def _rate_rule_amount_oracle(t):
    """Reference [P](t) for the rate-rule bare law: V(t) = V0 + 0.3·t."""
    from scipy.integrate import solve_ivp

    def V(_t):
        return V0 + 0.3 * _t

    def rhs(_t, y):
        r = K * y[0] * y[1] / V(_t) ** 2
        return [-r, -r, r]

    sol = solve_ivp(
        rhs,
        (0, T_END),
        [A0 * V0, B0 * V0, 0.0],
        t_eval=t.tolist(),
        rtol=1e-12,
        atol=1e-14,
    )
    return sol.y[2] / V(t)


def test_event_resize_bare_law_matches_amount_oracle():
    """The bare ``k·A·B`` law follows the live-volume amount ODE, NOT the
    pre-fix baked-static-volume trajectory."""
    t, P = _run_P(_bimolecular_sbml(vary="event"))
    correct = _event_amount_oracle(t, v_rate_post=2 * V0)  # V_live after resize
    buggy = _event_amount_oracle(t, v_rate_post=V0)  # pre-fix: stale V_static

    np.testing.assert_allclose(P, correct, rtol=1e-5, atol=1e-4)

    # Anti-regression: the fix must land unambiguously on the correct oracle and
    # far from the pre-fix bug (the issue reported max|bngsim−RR| ≈ 1.04).
    after = t > TE
    assert np.max(np.abs(P[after] - correct[after])) < 1e-3
    assert np.max(np.abs(correct[after] - buggy[after])) > 0.5
    assert np.max(np.abs(P[after] - buggy[after])) > 0.5


def test_rate_rule_bare_law_matches_amount_oracle():
    """Continuous V(t): the bare law follows k·[A]·[B]/V(t), reported [P]=aP/V(t)."""
    t, P = _run_P(_bimolecular_sbml(vary="rate_rule"))
    correct = _rate_rule_amount_oracle(t)
    np.testing.assert_allclose(P, correct, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("vary", ["event", "rate_rule"])
def test_bare_law_matches_roadrunner(vary):
    """Cross-check against the engine the GH #130 repro used."""
    roadrunner = pytest.importorskip("roadrunner")
    sbml = _bimolecular_sbml(vary=vary)
    t, P = _run_P(sbml)
    rr = roadrunner.RoadRunner(sbml)
    rr.selections = ["time", "[P]"]
    ref = np.asarray(rr.simulate(0, T_END, N_POINTS))
    np.testing.assert_allclose(P, ref[:, 1], rtol=1e-4, atol=1e-3)


def test_p1_convention_unaffected_and_matches_roadrunner():
    """The BNG ``Cc·k·A·B`` (p=1) form still cancels the compartment, stays on
    the Elementary path, and matches RoadRunner — the fix does not touch it."""
    roadrunner = pytest.importorskip("roadrunner")
    sbml = _p1_event_sbml()
    t, P = _run_P(sbml)
    rr = roadrunner.RoadRunner(sbml)
    rr.selections = ["time", "[P]"]
    ref = np.asarray(rr.simulate(0, T_END, N_POINTS))
    np.testing.assert_allclose(P, ref[:, 1], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("vary", ["event", "rate_rule"])
def test_bare_law_routes_to_functional_under_ssa(vary):
    """Routing signal: a bare (p≠1) varvol law no longer classifies as
    mass-action, so it takes the Functional live-symbol-divide path. As of
    GH #144 (case 2) this irreversible single-compartment monomial is SUPPORTED
    under SSA — the loader tags the Functional reaction with the scalar
    (V_static/V_live)^(n_f-1) propensity correction instead of refusing it (the
    numeric agreement with an independent Extrande sampler is pinned in
    test_ssa_variable_volume.py). The p=1 form stays Elementary and also clean.
    """
    bare = bngsim.Model.from_sbml_string(_bimolecular_sbml(vary=vary))
    bare_errs = [i for i in bngsim.validate_for_ssa(bare) if i.severity == "error"]
    assert bare_errs == [], bare_errs

    p1 = bngsim.Model.from_sbml_string(_p1_event_sbml())
    p1_errs = [i for i in bngsim.validate_for_ssa(p1) if i.severity == "error"]
    assert p1_errs == [], p1_errs


def test_constant_volume_bare_law_stays_elementary():
    """Guard sanity: with no varvol compartment the bare law is untouched — it
    classifies as mass-action (Elementary) and validates clean for SSA."""
    sbml = _bimolecular_sbml(vary="event").replace(
        'constant="false"/>\n    </listOfCompartments>',
        'constant="true"/>\n    </listOfCompartments>',
    )
    # Drop the resize event so the compartment is genuinely constant.
    sbml = sbml.split("<listOfEvents>")[0] + "  </model>\n</sbml>"
    model = bngsim.Model.from_sbml_string(sbml)
    errs = [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    assert errs == [], errs
