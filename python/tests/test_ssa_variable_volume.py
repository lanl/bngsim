"""SSA with a variable-volume compartment (GH #81 Tier 1 + Tier 2).

A compartment whose size changes at runtime — discretely via an event (Tier 1)
or continuously via a rate rule (Tier 2) — makes a reaction's propensity depend
on the live volume. bngsim stores every species as ``amount/V_static`` (the
load-time size), so the molecule COUNT (``conc·V_static``) is conserved when V
moves, but each hOSU=false concentration factor in a rate law is then stale by
``V_static/V_live``. For an all-hOSU=false single-compartment mass-action
monomial with ``n_h`` such factors the exact propensity correction is
``(V_static/V_live)^(n_h-1)`` (unimolecular: no-op; bimolecular: ∝1/V;
zeroth-order: ∝V). The loader tags such reactions ``ssa_live_volume_*`` and the
engine applies the correction.

The same scalar-correction machinery extends to monomials that route to the
Functional path: GH #144 lifts the gate for (case 1) an hOSU=true amount-valued
law factor — ``cell*k*H`` — and (case 2) a bare ``k*A*B`` law where the
compartment power doesn't cancel (p≠1), for single-compartment IRREVERSIBLE
reactions. The Functional emission already carries the live volume (live-symbol
or numeric-V_static divide), so the propensity correction is exponent ``n_f - 1``
(hOSU=false law-factor count) for the live divide, ``0`` for the numeric divide.
Still refused: cross-compartment reactions (#144 case 4), reversible non-mass-
action laws, and genuine non-mass-action kinetics (MM/Hill) — all
``varvol_non_mass_action`` (or ``reversible_non_mass_action``).

Two engine pieces make this correct:

* **Tier 1 count-preserving resize** — the loader's per-species concentration
  rescale ``s := s·V_old/V_new`` injected at a compartment resize is marked
  ODE-only and skipped under SSA, so molecule counts are unchanged across the
  event (only the propensity's live-volume factor updates).
* **Tier 2 dilution skip** — the #86 concentration-dilution term ``-[S]·V̇/V``
  for a rate-rule compartment is an ODE-only reaction, excluded from SSA
  entirely (firing it would spuriously consume molecules; the volume's effect on
  rates is carried by the propensity correction).

Oracle
------
An independent hand-rolled **Extrande** sampler (``_extrande_reference``) — a
thinning/rejection algorithm for time-varying propensities, a *different*
algorithm from bngsim's sub-step direct method, so agreement is strong evidence
of correctness rather than a shared bug. V(t) is supplied to the oracle as a
piecewise-constant function of time (Tier 1) or a continuous ``cont`` variable
(Tier 2). For the unimolecular case the SSA mean equals the closed form (the
propensity is V-independent), used directly.
"""

import math
import os
import sys

import bngsim
import numpy as np
import pytest

# Every test here builds its model via Model.from_antimony_string, so the whole
# module needs the optional `antimony` dependency (GH #153). antimony is
# deliberately excluded from the cibuildwheel test env, so guard at import.
pytest.importorskip("antimony")

# Sibling helper import under pytest's importlib mode.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _extrande_reference as ext  # noqa: E402


def _ssa_counts(model, t_end, n_points, reps, seed):
    """Return (names, array[reps, n_points, nsp]) of an SSA batch."""
    res = bngsim.Simulator(model, method="ssa").run_batch(
        t_span=(0.0, t_end), n_points=n_points, params=[{} for _ in range(reps)], seed=seed
    )
    names = list(res[0].species_names)
    arr = np.stack([np.asarray(r.species) for r in res], axis=0)
    return names, arr


# ── Tier 1 — bimolecular A + B → P with an event that doubles the volume ──────
# Law is the BNG ``compartment*k*A*B`` convention (the compartment appears once,
# so it cancels and bngsim's variable-volume ODE matches RoadRunner). The
# amount/time propensity is Cc·k·[A][B] = k·n_A·n_B/V(t), which halves once the
# event doubles V at TE.
_T1_BIMOL = """
model t1_bimol
  compartment Cc = 1.0;
  species A in Cc = 100;
  species B in Cc = 100;
  species P in Cc = 0;
  k = 0.002;
  J1: A + B -> P; Cc*k*A*B;
  E1: at (time >= 5): Cc = 2.0;
end
"""
_T1_K, _T1_A0, _T1_TE, _T1_VPRE, _T1_VPOST = 0.002, 100, 5.0, 1.0, 2.0


def test_tier1_resize_counts_conserved():
    """Molecule counts are conserved across the resize (no concentration rescale
    under SSA). For A+B→P the amounts obey A+P=A0 and B+P=B0 exactly, in every
    replicate at every sample time — the signature that the count-preserving
    resize (ODE-only rescale skip) works."""
    model = bngsim.Model.from_antimony_string(_T1_BIMOL)
    names, arr = _ssa_counts(model, 10.0, 11, reps=200, seed=4)
    ai, bi, pi = names.index("A"), names.index("B"), names.index("P")
    assert np.abs(arr[:, :, ai] + arr[:, :, pi] - _T1_A0).max() == 0.0
    assert np.abs(arr[:, :, bi] + arr[:, :, pi] - _T1_A0).max() == 0.0
    # Integer counts (no fractional molecules introduced by a stray rescale).
    assert np.allclose(arr[:, :, pi], np.round(arr[:, :, pi]))


def test_tier1_resize_matches_extrande():
    """SSA mean of P tracks the Extrande oracle with a step-V(t) propensity. The
    resize slows the bimolecular rate (∝1/V) after TE — the live-volume
    correction; a baked V would not change the rate."""
    reps, seed = 1500, 7
    model = bngsim.Model.from_antimony_string(_T1_BIMOL)
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    pi = names.index("P")
    t = np.linspace(0.0, 10.0, 11)

    def Vt(s):
        return _T1_VPOST if s["t"] >= _T1_TE else _T1_VPRE

    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": _T1_A0, "B": _T1_A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: _T1_K * s["A"] * s["B"] / Vt(s),
            )
        ],
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    p_bng, p_ext = arr[:, :, pi].mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(arr[:, :, pi].var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(p_bng[i] - p_ext[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"Tier1 t={t[i]:.1f}: bng={p_bng[i]:.2f} ext={p_ext[i]:.2f} z={z:.2f}"


# ── Tier 1 — unimolecular decay is volume-independent (count unchanged) ───────
# A → ∅ at rate k·A. The propensity k·n_A does not depend on V, so the resize
# changes nothing about the counts: the SSA mean equals the closed-form decay
# A0·e^(-k·t) with no jump at the event (count is preserved, not rescaled).
_T1_UNI = """
model t1_uni
  compartment Cc = 1.0;
  species A in Cc = 200;
  k = 0.3;
  J1: A => ; Cc*k*A;
  E1: at (time >= 3): Cc = 4.0;
end
"""


def test_tier1_unimolecular_volume_independent():
    reps, seed = 600, 12
    model = bngsim.Model.from_antimony_string(_T1_UNI)
    names, arr = _ssa_counts(model, 6.0, 7, reps=reps, seed=seed)
    ai = names.index("A")
    t = np.linspace(0.0, 6.0, 7)
    mean = arr[:, :, ai].mean(0)
    se = arr[:, :, ai].std(0, ddof=1) / math.sqrt(reps)
    closed = 200.0 * np.exp(-0.3 * t)
    for i in range(1, len(t)):
        z = abs(mean[i] - closed[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"unimolecular A at t={t[i]:.1f}: {mean[i]:.2f} vs {closed[i]:.2f} (z={z:.2f})"
        )
    # Counts never exceed A0 and are integer — a stray V_old/V_new rescale would
    # break both (it would multiply the stored count by 1/4 at the event).
    assert arr[:, :, ai].max() <= 200
    assert np.allclose(arr[:, :, ai], np.round(arr[:, :, ai]))


# ── Tier 2 — bimolecular A + B → P in a compartment that grows by a rate rule ─
# Cc' = g (linear growth), V(t) = 1 + g·t. The propensity ∝ 1/V(t) varies
# continuously between fires; the #86 dilution term must be skipped under SSA so
# molecule counts stay conserved.
_T2_BIMOL = """
model t2_bimol
  compartment Cc = 1.0;
  species A in Cc = 100;
  species B in Cc = 100;
  species P in Cc = 0;
  k = 0.002;
  g = 0.15;
  Cc' = g;
  J1: A + B -> P; Cc*k*A*B;
end
"""
_T2_K, _T2_A0, _T2_G = 0.002, 100, 0.15


def test_tier2_rate_rule_counts_conserved():
    """Counts conserved (the dilution term is excluded from SSA, so a growing
    volume does not consume molecules) and the compartment integrates as V=1+g·t."""
    model = bngsim.Model.from_antimony_string(_T2_BIMOL)
    names, arr = _ssa_counts(model, 8.0, 9, reps=200, seed=5)
    ai, pi = names.index("A"), names.index("P")
    assert np.abs(arr[:, :, ai] + arr[:, :, pi] - _T2_A0).max() == 0.0
    if "Cc" in names:
        t = np.linspace(0.0, 8.0, 9)
        np.testing.assert_allclose(
            arr[:, :, names.index("Cc")].mean(0), 1.0 + _T2_G * t, atol=1e-6
        )


def test_tier2_rate_rule_matches_extrande():
    """SSA mean of P tracks the Extrande oracle with V as a continuous variable."""
    reps, seed = 1500, 11
    model = bngsim.Model.from_antimony_string(_T2_BIMOL)
    names, arr = _ssa_counts(model, 8.0, 9, reps=reps, seed=seed)
    pi = names.index("P")
    t = np.linspace(0.0, 8.0, 9)
    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": _T2_A0, "B": _T2_A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: _T2_K * s["A"] * s["B"] / s["V"],
            )
        ],
        cont={"V": lambda s: _T2_G},
        c0={"V": 1.0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.05)
    p_bng, p_ext = arr[:, :, pi].mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(arr[:, :, pi].var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(p_bng[i] - p_ext[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"Tier2 t={t[i]:.1f}: bng={p_bng[i]:.2f} ext={p_ext[i]:.2f} z={z:.2f}"


# ── Gating — non-mass-action reactions in a varvol compartment are refused ────
_T2_FUNCTIONAL = """
model t2_func
  compartment Cc = 1.0;
  species A in Cc = 100;
  species P in Cc = 0;
  Vmax = 5.0; Km = 10.0; g = 0.1;
  Cc' = g;
  J1: A -> P; Vmax*A/(Km + A);
end
"""


def test_varvol_non_mass_action_rejected():
    """A non-mass-action (saturable) reaction in a rate-rule compartment is
    refused under SSA — the scalar live-volume correction is exact only for a
    mass-action monomial."""
    model = bngsim.Model.from_antimony_string(_T2_FUNCTIONAL)
    codes = {i.code for i in bngsim.validate_for_ssa(model)}
    assert "varvol_non_mass_action" in codes
    with pytest.raises(bngsim.SsaValidationError):
        bngsim.Simulator(model, method="ssa").run(t_span=(0, 2), n_points=3, seed=1)


# A *bare* concentration-rate law ``k*A*B`` (no compartment factor, p=0) is a
# different beast from the BNG ``Cc*k*A*B`` (p=1): bngsim's variable-volume ODE
# bakes the static volume and diverges from RoadRunner after the volume moves,
# so SSA must refuse it rather than run a value the ODE cannot reproduce.
_T2_BARE = """
model t2_bare
  compartment Cc = 1.0;
  species A in Cc = 100;
  species B in Cc = 100;
  species P in Cc = 0;
  k = 0.002;
  g = 0.1;
  Cc' = g;
  J1: A + B -> P; k*A*B;
end
"""


def test_bare_concentration_law_in_varvol_rejected():
    model = bngsim.Model.from_antimony_string(_T2_BARE)
    codes = {i.code for i in bngsim.validate_for_ssa(model)}
    assert "varvol_non_mass_action" in codes
    with pytest.raises(bngsim.SsaValidationError):
        bngsim.Simulator(model, method="ssa").run(t_span=(0, 2), n_points=3, seed=1)


# ── V_static != 1: the compartment loads at 2.0 ──────────────────────────────
# Exercises a non-trivial ssa_volume_factor (= V_static = 2) and the exponent
# n_h - p with V_static != 1. Species values are concentrations, so the molecule
# counts are conc·V_static; the propensity is k·n_A·n_B/V(t) and must follow the
# resize from V=2 to V=4.
_T1_VNE1 = """
model t1_vne1
  compartment Cc = 2.0;
  species A in Cc = 50;
  species B in Cc = 50;
  species P in Cc = 0;
  k = 0.004;
  J1: A + B -> P; Cc*k*A*B;
  E1: at (time >= 4): Cc = 4.0;
end
"""


def test_tier1_vstatic_ne1_matches_extrande():
    Vs, A0c, k, TE, Vpost = 2.0, 50.0, 0.004, 4.0, 4.0
    reps, seed = 1500, 3
    model = bngsim.Model.from_antimony_string(_T1_VNE1)
    names, arr = _ssa_counts(model, 8.0, 9, reps=reps, seed=seed)
    ai, pi = names.index("A"), names.index("P")
    # Stored values are concentrations (= count/V_static); the count invariant
    # (A+P) is conserved in concentration units too.
    assert np.abs(arr[:, :, ai] + arr[:, :, pi] - A0c).max() == 0.0
    # Convert the reported concentration to counts (× V_static) to compare.
    p_bng = arr[:, :, pi] * Vs
    n0 = int(round(A0c * Vs))
    t = np.linspace(0.0, 8.0, 9)

    def Vt(s):
        return Vpost if s["t"] >= TE else Vs

    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": n0, "B": n0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: k * s["A"] * s["B"] / Vt(s),
            )
        ],
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.08)
    pb, pe = p_bng.mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(p_bng.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"V≠1 P at t={t[i]:.1f}: bngsim {pb[i]:.2f} vs extrande {pe[i]:.2f} (z={z:.2f})"
        )


# ── Zeroth-order synthesis in a variable-volume compartment (GH #144) ─────────
# ``∅ → P`` with the BNG ``compartment*k`` law (p == 1). There are no reactant
# concentration factors (n_h = 0), so the live-volume exponent is n_h - p = -1:
# the amount/time propensity is Cc·k = k·V(t), i.e. ∝ V_live. As a pure birth
# process the count N_P(t) is a time-inhomogeneous Poisson variable with mean
# k·∫V dt — an exact closed form — which we check alongside the independent
# Extrande sampler. Before #144 this case was refused (varvol_non_mass_action);
# the engine already handled the negative exponent.
_T1_SYNTH = """
model t1_synth
  compartment Cc = 1.0;
  species P in Cc = 0;
  k = 5.0;
  J1: -> P; Cc*k;
  E1: at (time >= 5): Cc = 2.0;
end
"""


def test_tier1_synthesis_matches_extrande():
    """Event-resize zeroth-order synthesis: P accrues faster once V doubles at
    TE (propensity ∝ V_live, exponent -1). A baked static volume would keep the
    rate flat across the event."""
    k, TE, Vpre, Vpost = 5.0, 5.0, 1.0, 2.0
    reps, seed = 1500, 11
    model = bngsim.Model.from_antimony_string(_T1_SYNTH)
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    pi = names.index("P")
    nP = arr[:, :, pi]  # V_static = 1, so the stored column is the molecule count
    # Pure birth process: integer counts, non-decreasing in time.
    assert np.allclose(nP, np.round(nP))
    assert (np.diff(nP, axis=1) >= 0).all()
    t = np.linspace(0.0, 10.0, 11)

    # Exact mean of the time-inhomogeneous Poisson birth process: k·∫V dt.
    mean_cf = k * (Vpre * np.minimum(t, TE) + Vpost * np.maximum(0.0, t - TE))
    pb = nP.mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - mean_cf[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"synth Tier1 closed-form t={t[i]:.1f}: "
            f"bng={pb[i]:.2f} exact={mean_cf[i]:.2f} z={z:.2f}"
        )

    # Independent stochastic oracle (Extrande) with a step-V(t) propensity.
    def Vt(s):
        return Vpost if s["t"] >= TE else Vpre

    ref = ext.RefModel(
        species=["P"],
        x0={"P": 0},
        reactions=[ext.ReactionSpec(stoich={"P": 1}, propensity=lambda s: k * Vt(s))],
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pe = oref[:, :, 0].mean(0)
    se2 = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 0].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se2[i] + 1e-12)
        assert z < 5.0, f"synth Tier1 t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


_T2_SYNTH = """
model t2_synth
  compartment cell = 1.0;
  species P in cell = 0;
  k = 5.0; g = 0.2;
  cell' = g;
  J1: -> P; cell*k;
end
"""


def test_tier2_synthesis_matches_extrande():
    """Rate-rule zeroth-order synthesis in ``cell(t) = 1 + g·t``: propensity
    k·cell(t) grows linearly, so the mean P is k·(V0·t + ½·g·t²). Exercises the
    continuous live-volume read (exponent -1)."""
    k, g, V0 = 5.0, 0.2, 1.0
    reps, seed = 1500, 13
    model = bngsim.Model.from_antimony_string(_T2_SYNTH)
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    pi = names.index("P")
    nP = arr[:, :, pi]  # V_static = 1
    assert np.allclose(nP, np.round(nP))
    t = np.linspace(0.0, 10.0, 11)

    mean_cf = k * (V0 * t + 0.5 * g * t**2)
    pb = nP.mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - mean_cf[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"synth Tier2 closed-form t={t[i]:.1f}: "
            f"bng={pb[i]:.2f} exact={mean_cf[i]:.2f} z={z:.2f}"
        )

    ref = ext.RefModel(
        species=["P"],
        x0={"P": 0},
        reactions=[ext.ReactionSpec(stoich={"P": 1}, propensity=lambda s: k * s["cell"])],
        cont={"cell": lambda s: g},
        c0={"cell": V0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pe = oref[:, :, 0].mean(0)
    se2 = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 0].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se2[i] + 1e-12)
        assert z < 5.0, f"synth Tier2 t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


# A *bare* zeroth-order law ``k`` (no compartment factor, p=0) in a varvol
# compartment is refused for the same reason as the bare bimolecular law: the
# ODE bakes the static volume, so SSA cannot reproduce it after V moves. Only
# the BNG ``cell*k`` (p=1) synthesis is supported (#144).
_T2_SYNTH_BARE = """
model t2_synth_bare
  compartment cell = 1.0;
  species P in cell = 0;
  k = 5.0; g = 0.2;
  cell' = g;
  J1: -> P; k;
end
"""


def test_bare_synthesis_in_varvol_rejected():
    model = bngsim.Model.from_antimony_string(_T2_SYNTH_BARE)
    codes = {i.code for i in bngsim.validate_for_ssa(model)}
    assert "varvol_non_mass_action" in codes
    with pytest.raises(bngsim.SsaValidationError):
        bngsim.Simulator(model, method="ssa").run(t_span=(0, 2), n_points=3, seed=1)


# ── Case 1 (GH #144): hOSU=true (amount-valued) law factor in a varvol comp ───
# An hOSU=true reactant in a law that carries the compartment as an explicit
# factor (``cell*k*H``) routes to the Functional path (GH #131 finding 4 — the
# baked V_static would be wrong once V moves). The engine reads H as its amount
# (= molecule count), so the propensity is cell·k·n_H = k·V_live·n_H: a per-
# molecule decay rate that GROWS with the compartment. For a pure death process
# the SSA mean is the exact deterministic solution n_H0·exp(-k·∫V dt). The
# Functional emission divides by the numeric V_static for an all-hOSU=true law,
# so NO scalar correction is needed (exponent 0) — this test pins that. Reactions
# use ``=>`` (irreversible): antimony ``->`` is reversible, which keeps a non-
# classifying law refused (the forward-minus-reverse SSA hazard).
_C1_RR = """
model c1_rr
  compartment cell = 1.0;
  substanceOnly species H in cell = 200;
  k = 0.05; g = 0.2;
  cell' = g;
  J1: H => ; cell*k*H;
end
"""


def test_case1_hosu_decay_rate_rule_matches_oracle():
    k, g, V0, H0 = 0.05, 0.2, 1.0, 200
    reps, seed = 1500, 21
    model = bngsim.Model.from_antimony_string(_C1_RR)
    # No SSA validation error: the hOSU=true varvol law is now supported (#144).
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    hi = names.index("H")
    nH = arr[:, :, hi]  # hOSU=true, V_static = 1 ⇒ stored column is the count
    assert np.allclose(nH, np.round(nH))
    t = np.linspace(0.0, 10.0, 11)

    # Exact mean of the linear (per-capita k·V_live) death process: H0·exp(-k·∫V).
    mean_cf = H0 * np.exp(-k * (V0 * t + 0.5 * g * t**2))
    hb = nH.mean(0)
    se = np.sqrt(nH.var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(hb[i] - mean_cf[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"case1 RR closed-form t={t[i]:.1f}: bng={hb[i]:.2f} exact={mean_cf[i]:.2f} z={z:.2f}"
        )

    ref = ext.RefModel(
        species=["H"],
        x0={"H": H0},
        reactions=[
            ext.ReactionSpec(stoich={"H": -1}, propensity=lambda s: k * s["cell"] * s["H"])
        ],
        cont={"cell": lambda s: g},
        c0={"cell": V0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    he = oref[:, :, 0].mean(0)
    se2 = np.sqrt(nH.var(0, ddof=1) / reps + oref[:, :, 0].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(hb[i] - he[i]) / (se2[i] + 1e-12)
        assert z < 5.0, f"case1 RR t={t[i]:.1f}: bng={hb[i]:.2f} ext={he[i]:.2f} z={z:.2f}"


_C1_EV = """
model c1_ev
  compartment cell = 1.0;
  substanceOnly species H in cell = 200;
  k = 0.05;
  J1: H => ; cell*k*H;
  E1: at (time >= 5): cell = 2.0;
end
"""


def test_case1_hosu_decay_event_matches_oracle():
    k, TE, Vpre, Vpost, H0 = 0.05, 5.0, 1.0, 2.0, 200
    reps, seed = 1500, 23
    model = bngsim.Model.from_antimony_string(_C1_EV)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    hi = names.index("H")
    nH = arr[:, :, hi]
    assert np.allclose(nH, np.round(nH))
    t = np.linspace(0.0, 10.0, 11)

    int_V = Vpre * np.minimum(t, TE) + Vpost * np.maximum(0.0, t - TE)
    mean_cf = H0 * np.exp(-k * int_V)
    hb = nH.mean(0)
    se = np.sqrt(nH.var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(hb[i] - mean_cf[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"case1 EV closed-form t={t[i]:.1f}: bng={hb[i]:.2f} exact={mean_cf[i]:.2f} z={z:.2f}"
        )

    def Vt(s):
        return Vpost if s["t"] >= TE else Vpre

    ref = ext.RefModel(
        species=["H"],
        x0={"H": H0},
        reactions=[ext.ReactionSpec(stoich={"H": -1}, propensity=lambda s: k * Vt(s) * s["H"])],
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    he = oref[:, :, 0].mean(0)
    se2 = np.sqrt(nH.var(0, ddof=1) / reps + oref[:, :, 0].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(hb[i] - he[i]) / (se2[i] + 1e-12)
        assert z < 5.0, f"case1 EV t={t[i]:.1f}: bng={hb[i]:.2f} ext={he[i]:.2f} z={z:.2f}"


# ── GH #170: bare amount law (hOSU=true, NO compartment factor) in a varvol comp ─
# Case 1's ``cell*k*H`` carries the compartment as an explicit factor, so it
# routes to the Functional path (#131 finding 4). DROP the factor — ``k*A`` with A
# hOSU=true and no compartment in the law — and the reaction stays Elementary (no
# compartment factor for the classifier to route on). It hit the Elementary varvol
# gate ("gate A") and was refused ``varvol_non_mass_action`` even though the engine
# reads A as its amount (= molecule count), making the propensity k·n_A purely
# volume-INDEPENDENT: d(n_A)/dt = −k·n_A is plain exponential decay, no live-volume
# term. #170 admits this on gate A with live-volume exponent 0 (run uncorrected),
# the Elementary-path twin of case 1. The mean is the exact closed form
# n_A(t) = n_A0·exp(−k·t) (V-independent), which we check alongside Extrande. With
# V_static = 1 the stored column is the integer count. ``=>`` is irreversible.
_C170_RR = """
model c170_rr
  compartment cell = 1.0;
  substanceOnly species A in cell = 200;
  k = 0.05; g = 0.2;
  cell' = g;
  J1: A => ; k*A;
end
"""


def test_gh170_bare_hosu_decay_rate_rule_matches_oracle():
    k, g, V0, A0 = 0.05, 0.2, 1.0, 200
    reps, seed = 1500, 41
    model = bngsim.Model.from_antimony_string(_C170_RR)
    # No SSA validation error: the bare hOSU=true varvol amount law is supported.
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    ai = names.index("A")
    nA = arr[:, :, ai]  # hOSU=true, V_static = 1 ⇒ stored column is the count
    assert np.allclose(nA, np.round(nA))
    t = np.linspace(0.0, 10.0, 11)

    # The propensity k·n_A has NO live-volume term: the compartment growing under
    # ``cell' = g`` must NOT change the decay. Exact mean is the V-independent
    # closed form A0·exp(-k·t).
    mean_cf = A0 * np.exp(-k * t)
    ab = nA.mean(0)
    se = np.sqrt(nA.var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(ab[i] - mean_cf[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"#170 RR closed-form t={t[i]:.1f}: bng={ab[i]:.2f} exact={mean_cf[i]:.2f} z={z:.2f}"
        )

    # Independent Extrande oracle with a continuous ``cell`` variable whose value
    # the V-independent propensity ignores — pins that the growing volume is inert.
    ref = ext.RefModel(
        species=["A"],
        x0={"A": A0},
        reactions=[ext.ReactionSpec(stoich={"A": -1}, propensity=lambda s: k * s["A"])],
        cont={"cell": lambda s: g},
        c0={"cell": V0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    ae = oref[:, :, 0].mean(0)
    se2 = np.sqrt(nA.var(0, ddof=1) / reps + oref[:, :, 0].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(ab[i] - ae[i]) / (se2[i] + 1e-12)
        assert z < 5.0, f"#170 RR t={t[i]:.1f}: bng={ab[i]:.2f} ext={ae[i]:.2f} z={z:.2f}"


_C170_EV = """
model c170_ev
  compartment cell = 1.0;
  substanceOnly species A in cell = 200;
  k = 0.05;
  J1: A => ; k*A;
  E1: at (time >= 5): cell = 3.0;
end
"""


def test_gh170_bare_hosu_decay_event_matches_closed_form():
    k, A0 = 0.05, 200
    reps, seed = 1500, 43
    model = bngsim.Model.from_antimony_string(_C170_EV)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    nA = arr[:, :, names.index("A")]
    assert np.allclose(nA, np.round(nA))
    t = np.linspace(0.0, 10.0, 11)

    # A discrete resize at t=5 likewise cannot touch the V-independent propensity:
    # the closed form is exp(-k·t) straight through the event.
    mean_cf = A0 * np.exp(-k * t)
    ab = nA.mean(0)
    se = np.sqrt(nA.var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(ab[i] - mean_cf[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"#170 EV closed-form t={t[i]:.1f}: bng={ab[i]:.2f} exact={mean_cf[i]:.2f} z={z:.2f}"
        )


# A bimolecular all-hOSU=true bare law ``k*A*B`` (no compartment factor) is the
# same volume-independent lane (#170): the propensity is k·n_A·n_B with no live V,
# so the growing compartment is inert. The mean is nonlinear in the fluctuating
# counts, so Extrande (with a V-independent propensity) is the oracle.
_C170_BIMOL = """
model c170_bimol
  compartment cell = 1.0;
  substanceOnly species A in cell = 60; substanceOnly species B in cell = 90;
  substanceOnly species P in cell = 0;
  k = 0.01; g = 0.15;
  cell' = g;
  J1: A + B => P; k*A*B;
end
"""


def test_gh170_bare_hosu_bimolecular_matches_extrande():
    k, g, V0, A0, B0 = 0.01, 0.15, 1.0, 60, 90
    reps, seed = 2000, 45
    model = bngsim.Model.from_antimony_string(_C170_BIMOL)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 8.0, 9, reps=reps, seed=seed)
    nP = arr[:, :, names.index("P")]
    assert np.allclose(nP, np.round(nP))
    t = np.linspace(0.0, 8.0, 9)

    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": A0, "B": B0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1}, propensity=lambda s: k * s["A"] * s["B"]
            )
        ],
        cont={"cell": lambda s: g},
        c0={"cell": V0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.05)
    pb = nP.mean(0)
    pe = oref[:, :, 2].mean(0)
    se2 = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se2[i] + 1e-12)
        assert z < 5.0, f"#170 bimol t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


# Negative scope (#170): the fix must stay confined to the volume-INDEPENDENT
# sub-case. A bare law with a MIXED hOSU set — ``k*A*B`` with A hOSU=true but B
# hOSU=false — keeps a surviving V_static from B's concentration factor, so it is
# NOT volume-independent and the Elementary ODE bakes the static volume (the #131
# finding 4 hazard). It must stay refused ``varvol_non_mass_action``.
_C170_MIXED = """
model c170_mixed
  compartment cell = 1.0;
  substanceOnly species A in cell = 60; species B in cell = 90; species P in cell = 0;
  k = 0.01; g = 0.15;
  cell' = g;
  J1: A + B => P; k*A*B;
end
"""


def test_gh170_mixed_hosu_bare_law_still_refused():
    model = bngsim.Model.from_antimony_string(_C170_MIXED)
    errs = [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    assert [i.code for i in errs] == ["varvol_non_mass_action"]


# ── Case 2 (GH #144): bare (p≠1) concentration-rate law in a varvol comp ──────
# A bare ``k*A*B`` law (no compartment factor, p=0) is a genuinely different
# model from the BNG ``cell*k*A*B`` (p=1): the rate is k·[A][B] = k·n_A·n_B/V²,
# routed to the Functional live-symbol divide (GH #130, ODE cross-checked vs
# RoadRunner + COPASI). The Functional emission already divides by the live
# symbol, so the SSA propensity needs the scalar correction (V_static/V_live)^(n_f-1)
# = (V_static/V_live)^1 for the bimolecular case. The mean is nonlinear in the
# fluctuating counts, so the ODE diverges from SSA — the independent Extrande
# sampler is the oracle.
_C2_RR = """
model c2_rr
  compartment cell = 1.0;
  species A in cell = 100; species B in cell = 100; species P in cell = 0;
  k = 0.02; g = 0.1;
  cell' = g;
  J1: A + B => P; k*A*B;
end
"""


def test_case2_bare_bimolecular_rate_rule_matches_extrande():
    k, g, V0, A0 = 0.02, 0.1, 1.0, 100
    reps, seed = 2000, 31
    model = bngsim.Model.from_antimony_string(_C2_RR)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    ai, pi = names.index("A"), names.index("P")
    nP = arr[:, :, pi]  # hOSU=false, V_static = 1 ⇒ counts
    assert np.abs(arr[:, :, ai] + nP - A0).max() == 0.0  # A + P conserved
    t = np.linspace(0.0, 10.0, 11)

    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": A0, "B": A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: k * s["A"] * s["B"] / s["cell"] ** 2,
            )
        ],
        cont={"cell": lambda s: g},
        c0={"cell": V0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pb, pe = nP.mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"case2 RR t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


_C2_EV = """
model c2_ev
  compartment cell = 1.0;
  species A in cell = 100; species B in cell = 100; species P in cell = 0;
  k = 0.02;
  J1: A + B => P; k*A*B;
  E1: at (time >= 5): cell = 2.0;
end
"""


def test_case2_bare_bimolecular_event_matches_extrande():
    k, TE, Vpre, Vpost, A0 = 0.02, 5.0, 1.0, 2.0, 100
    reps, seed = 2000, 33
    model = bngsim.Model.from_antimony_string(_C2_EV)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    pi = names.index("P")
    nP = arr[:, :, pi]
    t = np.linspace(0.0, 10.0, 11)

    def Vt(s):
        return Vpost if s["t"] >= TE else Vpre

    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": A0, "B": A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: k * s["A"] * s["B"] / Vt(s) ** 2,
            )
        ],
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pb, pe = nP.mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"case2 EV t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


# ── (#144 case 4) Cross-compartment variable-volume reactions ───────────────────
#
# A + B => P with the reactants in DIFFERENT compartments, at least one of which
# changes size at runtime. Each variable-volume compartment c contributes its own
# (V_c,static / V_c,live)^m_c propensity factor (m_c = the hOSU=false reactant
# law-factor count in c) — a per-compartment vector, since the single scalar
# correction (cases 1–3) cannot hold a product over compartments. The bare-law
# monomial is supported (irreversible, all-hOSU=false); explicit-compartment-factor
# and hOSU=true cross-compartment shapes stay refused. Validated against the
# independent Extrande sampler (Tier 1 event + Tier 2 rate rule, one & both varvol)
# and the exact cross-compartment ODE semantics (RK4). The ODE path is additionally
# cross-checked vs RoadRunner + COPASI in dev/investigations/
# xcheck_144_xcompartment_varvol.py (standalone — COPASI pollutes pytest).

# One variable-volume compartment (cell, rate-rule) + one static (dish), P in cell.
_C4_CROSS_RR = """
model c4_cross_rr
  compartment cell = 1.0; compartment dish = 1.0;
  species A in cell = 100; species B in dish = 100; species P in cell = 0;
  k = 0.02; g = 0.1;
  cell' = g;
  J1: A + B => P; k*A*B;
end
"""

# One variable-volume compartment (cell, EVENT resize 1->2 at t=5) + static dish.
_C4_CROSS_EV = """
model c4_cross_ev
  compartment cell = 1.0; compartment dish = 1.0;
  species A in cell = 100; species B in dish = 100; species P in cell = 0;
  k = 0.02;
  J1: A + B => P; k*A*B;
  E1: at (time >= 5): cell = 2.0;
end
"""

# BOTH compartments variable-volume (cell + dish rate rules), P in cell.
_C4_BOTH_RR = """
model c4_both_rr
  compartment cell = 1.0; compartment dish = 1.0;
  species A in cell = 100; species B in dish = 100; species P in cell = 0;
  k = 0.02; g = 0.1; h = 0.07;
  cell' = g; dish' = h;
  J1: A + B => P; k*A*B;
end
"""


def test_c4_cross_compartment_now_supported():
    # The formerly-refused bare-law cross-compartment varvol monomial loads cleanly
    # under SSA (the varvol_non_mass_action gate is lifted for it).
    for src in (_C4_CROSS_RR, _C4_CROSS_EV, _C4_BOTH_RR):
        model = bngsim.Model.from_antimony_string(src)
        assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]


def test_c4_cross_one_varvol_rate_rule_matches_extrande():
    # Tier 2: A in a rate-rule (continuously growing) compartment, B in a static one.
    # ρ = k·[A]·[B] = k·A·B/(V_cell·V_dish), V_dish ≡ 1.  V0 = 1 ⇒ stored == count.
    k, g, A0 = 0.02, 0.1, 100
    reps, seed = 2000, 41
    model = bngsim.Model.from_antimony_string(_C4_CROSS_RR)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    nP = arr[:, :, names.index("P")]
    t = np.linspace(0.0, 10.0, 11)
    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": A0, "B": A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: k * s["A"] * s["B"] / s["cell"],
            )
        ],
        cont={"cell": lambda s: g},
        c0={"cell": 1.0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pb, pe = nP.mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"c4 RR t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


def test_c4_cross_one_varvol_event_matches_extrande():
    # Tier 1: A in an EVENT-resized compartment (cell 1->2 at t=5), B in static dish.
    k, TE, Vpre, Vpost, A0 = 0.02, 5.0, 1.0, 2.0, 100
    reps, seed = 2000, 43
    model = bngsim.Model.from_antimony_string(_C4_CROSS_EV)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    nP = arr[:, :, names.index("P")]
    t = np.linspace(0.0, 10.0, 11)

    def Vt(s):
        return Vpost if s["t"] >= TE else Vpre

    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": A0, "B": A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: k * s["A"] * s["B"] / Vt(s),
            )
        ],
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pb, pe = nP.mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"c4 EV t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


def test_c4_cross_both_varvol_matches_extrande():
    # BOTH reactants in variable-volume compartments ⇒ a TWO-element live-volume
    # vector (V_cell/V_cell,live)·(V_dish/V_dish,live) — the case the single scalar
    # correction cannot express.
    k, g, h, A0 = 0.02, 0.1, 0.07, 100
    reps, seed = 2000, 45
    model = bngsim.Model.from_antimony_string(_C4_BOTH_RR)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    nP = arr[:, :, names.index("P")]
    t = np.linspace(0.0, 10.0, 11)
    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": A0, "B": A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                propensity=lambda s: k * s["A"] * s["B"] / (s["cell"] * s["dish"]),
            )
        ],
        cont={"cell": lambda s: g, "dish": lambda s: h},
        c0={"cell": 1.0, "dish": 1.0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pb, pe = nP.mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"c4 both t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


def test_c4_cross_compartment_ode_matches_semantics():
    # The ODE fix (per-species LIVE-volume divide) reproduces the exact SBML
    # cross-compartment semantics — ρ = k·[A]·[B], dAmount/dt = ±ρ, [X]=amount/V_live
    # — for a model with V0 != 1 (so the V_static bug would visibly diverge). RK4
    # reference (self-contained); the buggy /V_static path drifts ~30% (see the
    # standalone RR+COPASI xcheck). cell is rate-rule varvol (V0=2), dish static V=3.
    src = (
        "model c4_ode; compartment cell=2.0; compartment dish=3.0; "
        "species A in cell=80; species B in dish=120; species P in cell=0; "
        "k=0.02; g=0.15; cell'=g; J1: A+B=>P; k*A*B; end"
    )
    model = bngsim.Model.from_antimony_string(src)
    t_end, npts = 8.0, 17
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=npts, rtol=1e-10, atol=1e-12
    )
    # Compare AMOUNTS (volume-invariant): bngsim reports them via as_roadrunner(ids).
    bam = res.as_roadrunner(["A", "B", "P"])
    t = np.linspace(0.0, t_end, npts)

    # RK4 reference of the literal SBML semantics.
    k, g, Vd = 0.02, 0.15, 3.0
    A, B, P, Vc = 80.0 * 2.0, 120.0 * 3.0, 0.0, 2.0  # initial AMOUNTS, V_cell(0)

    def deriv(state):
        A, B, P, Vc = state
        rho = k * (A / Vc) * (B / Vd)
        return np.array([-rho, -rho, rho, g])

    dt = 1e-3
    state = np.array([A, B, P, Vc])
    tt = 0.0
    grid = set(np.round(t, 9))
    rows = {0.0: state.copy()}
    while tt < t_end - 1e-12:
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        tt += dt
        r = round(tt, 9)
        if r in grid:
            rows[r] = state.copy()
    for i, ti in enumerate(t):
        ref_row = rows[round(ti, 9)]
        for j, sp in enumerate(("A", "B", "P")):
            assert abs(bam[i, j] - ref_row[j]) <= 1e-4 * max(abs(ref_row[j]), 1.0), (
                f"c4 ODE {sp} t={ti:.2f}: bng={bam[i, j]:.4f} ref={ref_row[j]:.4f}"
            )


# ── #172 — EXPLICIT variable-volume compartment factor (p != 1 cross-compartment) ──
# `cell·k·A·B` carries the live compartment volume as an explicit law factor. This is
# the cross-compartment analogue of #130's single-compartment p != 1 fix. Before
# #172 it was the deliberately-deferred "widen later" shape: refused under SSA but
# SILENTLY WRONG under ODE (per-species ÷V_static instead of ÷V_live, ~60% off vs
# RR+COPASI). It now routes to the SAME per-species ÷V_live path as the bare law:
# the explicit factor lives in base_func, and both the ODE divide and the SSA
# m_c = (species-factor count per varvol compartment) correction are INDEPENDENT of
# the explicit factor power (it cancels — see _classify_mass_action_ast / #172).

# Explicit factor `cell·k·A·B`, cell rate-rule varvol (V0=1 ⇒ stored == count),
# dish static. True propensity ρ = cell·k·[A]·[B] = k·n_A·n_B/V_dish — the live cell
# volume cancels (it multiplies the law but divides [A]); with V_dish ≡ 1, ρ = k·A·B.
_C5_EXPLICIT_RR = """
model c5_explicit_rr
  compartment cell = 1.0; compartment dish = 1.0;
  species A in cell = 100; species B in dish = 100; species P in cell = 0;
  k = 0.02; g = 0.1;
  cell' = g;
  J1: A + B => P; cell*k*A*B;
end
"""


def test_c5_explicit_compartment_factor_now_supported():
    # The formerly-refused explicit-factor cross-compartment varvol monomial now
    # loads cleanly under SSA (the varvol_non_mass_action gate is lifted for it).
    model = bngsim.Model.from_antimony_string(_C5_EXPLICIT_RR)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]


def test_c5_explicit_one_varvol_rate_rule_matches_extrande():
    # Tier 2 SSA: A in a rate-rule (growing) compartment that appears as an explicit
    # law factor, B in a static one. ρ = cell·k·[A]·[B] = k·A·B/V_dish; V_dish ≡ 1.
    k, g, A0 = 0.02, 0.1, 100
    reps, seed = 2000, 47
    model = bngsim.Model.from_antimony_string(_C5_EXPLICIT_RR)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    names, arr = _ssa_counts(model, 10.0, 11, reps=reps, seed=seed)
    nP = arr[:, :, names.index("P")]
    t = np.linspace(0.0, 10.0, 11)
    ref = ext.RefModel(
        species=["A", "B", "P"],
        x0={"A": A0, "B": A0, "P": 0},
        reactions=[
            ext.ReactionSpec(
                stoich={"A": -1, "B": -1, "P": 1},
                # cell·k·[A]·[B] = cell·k·(A/cell)·(B/1) = k·A·B (cell cancels).
                propensity=lambda s: k * s["A"] * s["B"],
            )
        ],
        cont={"cell": lambda s: g},
        c0={"cell": 1.0},
    )
    oref = ext.simulate_batch(ref, t, reps, seed=seed + 1, look_ahead=0.1)
    pb, pe = nP.mean(0), oref[:, :, 2].mean(0)
    se = np.sqrt(nP.var(0, ddof=1) / reps + oref[:, :, 2].var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(pb[i] - pe[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"c5 RR t={t[i]:.1f}: bng={pb[i]:.2f} ext={pe[i]:.2f} z={z:.2f}"


def test_c5_explicit_compartment_factor_ode_matches_semantics():
    # The ODE fix reproduces the exact SBML semantics for the explicit-factor law
    # ρ = cell·k·[A]·[B] (per-species ÷V_live, base_func carrying the live cell
    # factor). RK4 reference; the buggy ÷V_static path drifts ~60% (the #172 repro).
    # cell is rate-rule varvol (V0=2, so V_static ≠ V_live diverges), dish static V=3.
    src = (
        "model c5_ode; compartment cell=2.0; compartment dish=3.0; "
        "species A in cell=80; species B in dish=120; species P in cell=0; "
        "k=0.02; g=0.15; cell'=g; J1: A+B=>P; cell*k*A*B; end"
    )
    model = bngsim.Model.from_antimony_string(src)
    t_end, npts = 8.0, 17
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=npts, rtol=1e-10, atol=1e-12
    )
    bam = res.as_roadrunner(["A", "B", "P"])  # amounts (volume-invariant)
    t = np.linspace(0.0, t_end, npts)

    # RK4 reference of the literal SBML semantics. State is AMOUNTS + V_cell.
    k, g, Vd = 0.02, 0.15, 3.0
    A, B, P, Vc = 80.0 * 2.0, 120.0 * 3.0, 0.0, 2.0

    def deriv(state):
        A, B, P, Vc = state
        # ρ = cell·k·[A]·[B] = Vc·k·(A/Vc)·(B/Vd) = k·A·B/Vd (Vc cancels).
        rho = Vc * k * (A / Vc) * (B / Vd)
        return np.array([-rho, -rho, rho, g])

    dt = 1e-3
    state = np.array([A, B, P, Vc])
    tt = 0.0
    grid = set(np.round(t, 9))
    rows = {0.0: state.copy()}
    while tt < t_end - 1e-12:
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        tt += dt
        r = round(tt, 9)
        if r in grid:
            rows[r] = state.copy()
    for i, ti in enumerate(t):
        ref_row = rows[round(ti, 9)]
        for j, sp in enumerate(("A", "B", "P")):
            assert abs(bam[i, j] - ref_row[j]) <= 1e-4 * max(abs(ref_row[j]), 1.0), (
                f"c5 ODE {sp} t={ti:.2f}: bng={bam[i, j]:.4f} ref={ref_row[j]:.4f}"
            )


# Exotic cross-compartment shapes stay refused (the hOSU=true and reversible
# variants keep their gate; only all-hOSU=false irreversible monomials are lifted).
def test_c4_hosu_cross_compartment_still_refused():
    # An hOSU=true (amount-valued) reactant in a cross-compartment varvol reaction
    # is out of scope (its volume bookkeeping differs) — stays refused.
    src = (
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; substanceOnly species B in dish=100; species P in cell=0; "
        "k=0.02; g=0.1; cell'=g; J1: A+B=>P; k*A*B; end"
    )
    model = bngsim.Model.from_antimony_string(src)
    codes = {i.code for i in bngsim.validate_for_ssa(model) if i.severity == "error"}
    assert "varvol_non_mass_action" in codes


def test_c4_reversible_cross_compartment_still_refused():
    # A reversible cross-compartment varvol law is a forward-minus-reverse
    # difference, not a monomial — refused (reversible_non_mass_action).
    src = (
        "model m; compartment cell=1.0; compartment dish=1.0; "
        "species A in cell=100; species B in dish=100; species P in cell=0; "
        "k=0.02; kr=0.001; g=0.1; cell'=g; J1: A+B -> P; k*A*B - kr*P; end"
    )
    model = bngsim.Model.from_antimony_string(src)
    codes = {i.code for i in bngsim.validate_for_ssa(model) if i.severity == "error"}
    assert "varvol_non_mass_action" in codes or "reversible_non_mass_action" in codes


# ── (#171) Analytical Jacobian for cross-compartment variable-volume reactions ──
#
# These reactions always SIMULATE correctly (the RHS divides each species's
# accumulation by its LIVE compartment volume conc[ode_live_volume_idx0]); #144
# left them on a finite-difference Jacobian + interpreted RHS. #171 restores the
# ANALYTICAL Jacobian: the existing columns swap the baked-in 1/V_static for the
# runtime 1/V_live, plus a new column ∂(dSᵢ/dt)/∂V_live = −(varvol RHS of i)/V_live.
# This is optimization only — the ODE/SSA solution is unchanged.
#
# The load-time self-check is ALL-OR-NOTHING and SILENT: on any analytical↔FD
# mismatch it clears the analytical Jacobian and reverts the whole model to FD, so
# a matching trajectory does NOT prove the analytical path is live (the FD
# fallback also matches). Every assertion here checks
# ``analytical_jacobian_complete is True`` — the flag, not the trajectory — and
# independently FD-cross-checks the assembled Jacobian (including the new column).

_C171_MODELS = {
    "_C4_CROSS_RR": _C4_CROSS_RR,  # one varvol (cell) + static dish; A,P live, B static
    "_C4_CROSS_EV": _C4_CROSS_EV,  # event-resized cell + static dish
    "_C4_BOTH_RR": _C4_BOTH_RR,  # both varvol: A,P → cell, B → dish (two live columns)
    "_C5_EXPLICIT_RR": _C5_EXPLICIT_RR,  # explicit cell factor: ∂/∂cell cancels to 0
}


@pytest.mark.parametrize("name", list(_C171_MODELS))
def test_c4_c5_analytical_jacobian_is_complete(name, monkeypatch):
    """#171 core gate: the analytical Jacobian ATTACHES (no FD fallback) for every
    cross-compartment variable-volume case. Run under the DEFAULT self-check — no
    env may relax or bypass it — so ``analytical_jacobian_complete is True`` means
    the assembled Jacobian genuinely survived the FD self-validation at load."""
    for var in (
        "BNGSIM_JAC_SELFCHECK_DENSE_MAX",
        "BNGSIM_JAC_SELFCHECK_SAMPLE",
        "BNGSIM_JAC_NO_SELFCHECK",
    ):
        monkeypatch.delenv(var, raising=False)
    m = bngsim.Model.from_antimony_string(_C171_MODELS[name])
    assert m.prepare_analytical_jacobian() is True
    assert m._core.analytical_jacobian_complete is True


@pytest.mark.parametrize("name", list(_C171_MODELS))
def test_c4_c5_analytical_jacobian_matches_finite_difference(name):
    """Independent guard against a self-check FALSE-fail masking a math bug: the
    assembled analytical Jacobian (test hook ``_dense_analytical_jacobian``) must
    equal central differences of the RHS (``_eval_rhs``) at spread-out states —
    exercising the runtime 1/V_live divide AND the new −func/V_live² column."""
    m = bngsim.Model.from_antimony_string(_C171_MODELS[name])
    assert m.prepare_analytical_jacobian() is True
    core = m._core
    ns = core.n_species
    y0 = np.array(core.get_state(), dtype=float)
    # Probe the initial state plus spread-out states where V_live ≠ V_static so
    # the new column (and the live divide) actually bite.
    probes = [y0]
    for key in (0.6180339887, 0.3819660113, 0.7548776662):
        frac = (np.arange(1, ns + 1) * key) % 1.0
        probes.append(np.abs(y0) * (0.4 + 1.8 * frac) + 0.5)
    for y in probes:
        an = np.array(core._dense_analytical_jacobian(0.3, list(y))).reshape(ns, ns, order="F")
        fd = np.zeros((ns, ns))
        for j in range(ns):
            hj = 1e-6 * max(abs(y[j]), 1.0)
            yp, ym = list(y), list(y)
            yp[j] += hj
            ym[j] -= hj
            fd[:, j] = (np.array(core._eval_rhs(0.3, yp)) - np.array(core._eval_rhs(0.3, ym))) / (
                2 * hj
            )
        abs_err = np.abs(an - fd)
        denom = np.maximum(np.maximum(np.abs(an), np.abs(fd)), 1e-6)
        # Genuine mismatch needs BOTH a rate-scale absolute error and a large
        # relative error (an analytically-exact 0 vs FD noise ~1e-8 is a match).
        bad = (abs_err > 1e-6) & (abs_err / denom > 1e-4)
        assert not bad.any(), (
            f"{name}: analytical≠FD at {np.argwhere(bad).tolist()} an={an[bad]} fd={fd[bad]}"
        )


def test_c4_analytical_and_fd_jacobian_agree_on_trajectory():
    """The analytical Jacobian is genuinely EXERCISED in ODE integration (an
    explicit ``jacobian='analytical'`` raises if it falls back) and reproduces the
    correct FD-path trajectory — optimization, not a behavior change."""
    for src in (_C4_CROSS_RR, _C4_BOTH_RR, _C5_EXPLICIT_RR):
        traj = {}
        for strat in ("analytical", "fd"):
            m = bngsim.Model.from_antimony_string(src)
            sim = bngsim.Simulator(m, method="ode", jacobian=strat)
            r = sim.run((0.0, 10.0), n_points=51)
            if strat == "analytical":
                assert sim._ode_jacobian_fell_back is False
            names = list(r.species_names)
            arr = np.asarray(r.species)
            arr = arr if arr.shape[0] == r.n_times else arr.T
            traj[strat] = arr[:, names.index("P")]
        assert np.max(np.abs(traj["analytical"] - traj["fd"])) < 1e-6
