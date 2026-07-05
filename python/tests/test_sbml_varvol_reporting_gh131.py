"""Method-aware reporting for variable-volume compartments (GH #131 findings 1–2).

bngsim stores every species as ``amount / V_static`` (the load-time compartment
size). The reported quantities are ``X`` (amount = ``stored·V_static``) and
``[X]`` (concentration = ``amount / V_live(t)``). When a compartment's size
changes at runtime the two report selectors must stay consistent, but *which*
column already carries the live volume depends on the species kind (hOSU) AND
the simulation method:

* **ODE** integrates in concentration space and applies the #86 dilution term /
  the #74 event ``V_old/V_new`` rescale, so a hOSU=false species' stored column
  is already the live ``amount/V_live``.
* **SSA/PSA** preserve molecule *counts* across a volume change (those ODE-only
  corrections are skipped), so the stored column stays ``amount/V_static`` — the
  conserved count — and the live concentration must be recovered at report time.

Before GH #131 the reporting layer was method-blind: it reused the ODE-oriented
``amount = conc·V_live`` recovery on an SSA result (whose column is a count, not
a concentration), so SSA ``[S]`` read the stale static concentration and the
bare amount read ``count·V_live``. Event-resized compartments were excluded from
the report maps entirely, so an hOSU=true ``[H]`` stayed stale under BOTH methods.

The oracle here is exact and deterministic: reporting is a pure function of the
trajectory, so for any single run the invariants ``[X]·V_live == X`` and (for a
conserved boundary species) ``X == const`` hold to machine precision regardless
of the stochastic path — no statistical tolerance is needed.
"""

import bngsim
import numpy as np
import pytest

# All models here are built via Model.from_antimony_string, which needs the
# optional `antimony` dependency (GH #153). It is excluded from the
# cibuildwheel test env, so skip the whole module when it is unavailable.
pytest.importorskip("antimony")

# ── Rate-rule compartment: Cc(t) = V0 + g·t ──────────────────────────────────
G = 0.4


def _rate_rule_model(v0, hosu_boundary):
    """A rate-rule compartment with a conserved boundary species (hOSU per the
    flag) plus an all-hOSU=false p=1 mass-action reaction so the model is
    SSA-runnable. The boundary species has a closed-form amount/concentration."""
    kind = "substanceOnly species" if hosu_boundary else "species"
    return bngsim.Model.from_antimony_string(f"""
    model rr
      compartment Cc = {v0};
      {kind} $S in Cc = 5;
      species A in Cc = 60; species B in Cc = 60; species P in Cc = 0;
      k = 0.01; g = {G};
      Cc' = g;
      J1: A + B -> P; Cc*k*A*B;
    end
    """)


def _run(model, method, t_end=6.0, n_points=7, seed=1):
    sim = bngsim.Simulator(model, method=method)
    if method == "ode":
        return sim.run(t_span=(0.0, t_end), n_points=n_points)
    return sim.run_batch(t_span=(0.0, t_end), n_points=n_points, params=[{}], seed=seed)[0]


@pytest.mark.parametrize("method", ["ode", "ssa"])
@pytest.mark.parametrize("v0", [1.0, 2.0])
@pytest.mark.parametrize("hosu", [False, True])
def test_rate_rule_boundary_species_reporting(method, v0, hosu):
    """A conserved boundary species in a rate-rule compartment: the amount is
    constant (``conc0·V_static``) and the concentration dilutes as ``A/V_live``.
    Finding 1 (hOSU=false) and the hOSU=true vmap path, under both methods.

    Pre-#131 the SSA reports were ``[S] = const`` (stale static concentration)
    and ``S = const·V_live`` (count·V_live) — both wrong."""
    model = _rate_rule_model(v0, hosu_boundary=hosu)
    r = _run(model, method)
    t = np.asarray(r.time)
    v_live = v0 + G * t
    # Antimony's ``= 5`` is an AMOUNT for a substanceOnly (hOSU=true) species and
    # a CONCENTRATION for an hOSU=false one, so the conserved amount differs.
    amount = 5.0 if hosu else 5.0 * v0

    # ODE integrates the #86 dilution term numerically, so the recovered amount
    # of an hOSU=false boundary species drifts by ~1e-8 over the run; SSA conserves
    # the count exactly. rtol=1e-6 covers the ODE integration noise.
    rr = r.as_roadrunner(["S", "[S]"])
    np.testing.assert_allclose(rr[:, 0], amount, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(rr[:, 1], amount / v_live, rtol=1e-6, atol=1e-8)
    # The drift is material (guards a t0-only coincidence): the concentration at
    # t_end is well below its static initial value amount/V_static.
    assert rr[-1, 1] < 0.9 * (amount / v0)


@pytest.mark.parametrize("v0", [1.0, 2.0])
def test_rate_rule_reacting_species_ssa_invariant(v0):
    """For a *reacting* hOSU=false species in a rate-rule compartment under SSA,
    the two report selectors stay consistent: ``[X]·V_live == X`` exactly, and
    the bare amount equals the conserved count times V_static (the raw column).
    This locks Finding 1 for species whose count actually changes."""
    model = _rate_rule_model(v0, hosu_boundary=False)
    r = _run(model, "ssa")
    t = np.asarray(r.time)
    v_live = v0 + G * t
    names = list(r.species_names)
    raw = np.asarray(r.species)
    for x in ("A", "B", "P"):
        conc = r.as_roadrunner([f"[{x}]"])[:, 0]
        amount = r.as_roadrunner([x])[:, 0]
        # Reported concentration is the reported amount over the live volume.
        np.testing.assert_allclose(conc * v_live, amount, rtol=1e-9, atol=1e-9)
        # The bare amount is the conserved integer count (raw·V_static); the raw
        # SSA column is the count/V_static, so amount must be integer-valued.
        np.testing.assert_allclose(amount, raw[:, names.index(x)] * v0, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(amount, np.round(amount), atol=1e-9)


# ── Event-resized compartment: Cc steps 2 → 4 at an off-grid event time ───────
def _event_resize_model(hosu_boundary):
    kind = "substanceOnly species" if hosu_boundary else "species"
    return bngsim.Model.from_antimony_string(f"""
    model ev
      compartment Cc = 2.0;
      {kind} $H in Cc = 10;
      species A in Cc = 80; species B in Cc = 80; species P in Cc = 0;
      k = 0.005;
      J1: A + B -> P; Cc*k*A*B;
      E1: at (time > 2.5): Cc = 4;
    end
    """)


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_event_resize_hosu_true_reporting(method):
    """An hOSU=true conserved boundary species in an event-resized compartment:
    the amount is invariant across the resize while the concentration steps from
    ``amount/2`` to ``amount/4``. Finding 2 — pre-#131 ``[H]`` stayed stale at 5
    under BOTH ODE and SSA because event-resized compartments were excluded from
    every report map. V_live is read from the compartment's observable column,
    making the check robust to where the event lands on the sample grid."""
    model = _event_resize_model(hosu_boundary=True)
    r = _run(model, method)
    v_live = np.asarray(r.observables)[:, list(r.observable_names).index("Cc")]
    assert np.isclose(v_live[0], 2.0) and np.isclose(v_live[-1], 4.0)  # resize happened
    rr = r.as_roadrunner(["H", "[H]"])
    np.testing.assert_allclose(rr[:, 0], 10.0, rtol=1e-9, atol=1e-9)  # amount conserved
    np.testing.assert_allclose(rr[:, 1], 10.0 / v_live, rtol=1e-9, atol=1e-9)  # [H]=amt/V_live


def test_event_resize_hosu_false_ode_amount_is_conc_times_live_volume():
    """hOSU=false species in an event-resized compartment under ODE: the #74
    ``V_old/V_new`` rescale already ran, so the stored column is the live
    concentration and the bare ``X`` amount must be ``conc·V_live`` (RoadRunner's
    amount), not the stale ``conc·V_static``. Surfaced by the GH #131 RR/COPASI
    cross-check — pre-fix the amount was off by the resize ratio after the event.
    V_live is read from the compartment's observable column."""
    model = _event_resize_model(hosu_boundary=False)
    r = _run(model, "ode")
    v_live = np.asarray(r.observables)[:, list(r.observable_names).index("Cc")]
    assert np.isclose(v_live[0], 2.0) and np.isclose(v_live[-1], 4.0)  # resize happened
    for x in ("H", "A", "P"):
        conc = r.as_roadrunner([f"[{x}]"])[:, 0]
        amount = r.as_roadrunner([x])[:, 0]
        np.testing.assert_allclose(amount, conc * v_live, rtol=1e-7, atol=1e-9)
        # Post-resize the correct amount differs materially from the stale
        # conc·V_static the volume factor alone would have produced.
        assert abs(amount[-1] - conc[-1] * 2.0) > 1e-6


def test_event_resize_hosu_false_ssa_reporting():
    """A reacting hOSU=false species in an event-resized compartment under SSA:
    the molecule count is preserved across the resize (no concentration rescale),
    so the reported amount is the conserved count and ``[S] = amount/V_live``.
    Finding 2 SSA half — pre-#131 ``[S]`` read the stale ``amount/V_static``."""
    model = _event_resize_model(hosu_boundary=False)
    r = _run(model, "ssa")
    v_live = np.asarray(r.observables)[:, list(r.observable_names).index("Cc")]
    names = list(r.species_names)
    raw = np.asarray(r.species)
    for x in ("A", "B", "P"):
        conc = r.as_roadrunner([f"[{x}]"])[:, 0]
        amount = r.as_roadrunner([x])[:, 0]
        np.testing.assert_allclose(conc * v_live, amount, rtol=1e-9, atol=1e-9)
        # V_static == 2 here, so the bare amount is raw·2 (the conserved count).
        np.testing.assert_allclose(amount, raw[:, names.index(x)] * 2.0, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(amount, np.round(amount), atol=1e-9)
