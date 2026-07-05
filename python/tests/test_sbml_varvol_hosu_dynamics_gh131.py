"""hOSU=true dynamics in a variable-volume compartment (GH #131 findings 3–4).

An amount-valued (``hasOnlySubstanceUnits=true``) species in a compartment whose
size changes at runtime is stored as ``amount/V_static``. The kinetic law's rate
must scale with the LIVE compartment volume, but two loader paths baked the
load-time ``V_static`` instead, silently throttling the dynamics:

* **Finding 3 (Functional).** The §9 storage divide ``law / V`` used the live
  compartment symbol for an *event-resized* compartment (``vstatic_divide_comps``
  excluded it). For an hOSU=true species that cancels the law's own volume
  dependence — a ``synth(cell, k)`` synthesis stayed flat across a resize. Fix:
  divide by the numeric ``V_static`` for every variable-volume class (the engine
  restores the amount as ``stored·V_static``, so the rate keeps the live ``cell``
  factor from the law).

* **Finding 4 (Elementary).** The mass-action classifier admitted hOSU=true laws
  like ``cell·k`` / ``cell·k·H`` as Elementary, folding the load-time ``V_static``
  into the scalar rate. Because the engine reads an amount-valued reactant as
  ``stored·V_c``, the storage divide is undone and any variable-volume compartment
  appearing as an explicit law factor bakes ``V_static`` where ``V_live(t)`` is
  required. Fix: route such reactions to the Functional path (finding 3 makes it
  correct). A bare amount law (``k·H``) is volume-independent and stays Elementary.

Oracles are exact closed forms for the integrated amount; ``cell(t)`` is either a
rate rule ``cell(0)+g·t`` or an event step ``2 → 4``.
"""

import bngsim
import numpy as np
import pytest

# All models here are built via Model.from_antimony_string, which needs the
# optional `antimony` dependency (GH #153). It is excluded from the
# cibuildwheel test env, so skip the whole module when it is unavailable.
pytest.importorskip("antimony")

K_SYNTH = 0.3
K_DECAY = 0.05
G = 0.5
V0 = 2.0
TE = 3.0  # event resize time


def _amount(antimony, species, t_end=6.0, n_points=13):
    model = bngsim.Model.from_antimony_string(antimony)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=1e-10, atol=1e-12
    )
    return np.asarray(r.time), r.as_roadrunner([species])[:, 0]


# ∫₀ᵗ cell dt for the two cell(t) profiles, used by the decay/synthesis oracles.
def _int_cell_rate_rule(t):
    return V0 * t + 0.5 * G * t**2


def _int_cell_event(t):
    return np.where(t <= TE, V0 * t, V0 * TE + 4.0 * (t - TE))


# ── Finding 4 — Elementary hOSU=true laws routed to Functional ───────────────
def test_elementary_synth_rate_rule():
    """``∅ → P`` with law ``cell·k`` (Elementary syntax) in a rate-rule compartment:
    d(amount)/dt = cell(t)·k ⇒ amount = k·∫cell. Pre-#131 it baked V_static and
    grew linearly as k·V_static·t."""
    t, got = _amount(
        f"""
    model m
      compartment cell = {V0}; substanceOnly species P in cell = 0;
      k={K_SYNTH}; g={G}; cell' = g; J1: -> P; cell*k;
    end""",
        "P",
    )
    np.testing.assert_allclose(got, K_SYNTH * _int_cell_rate_rule(t), rtol=1e-5, atol=1e-7)
    # The throttled (buggy) result k·V_static·t is materially different at t_end.
    assert abs(got[-1] - K_SYNTH * V0 * t[-1]) > 0.1 * got[-1]


def test_elementary_synth_event_resize():
    """``cell·k`` synthesis across an event resize cell: 2 → 4."""
    t, got = _amount(
        f"""
    model m
      compartment cell = {V0}; substanceOnly species P in cell = 0;
      k={K_SYNTH}; J1: -> P; cell*k; E1: at (time > {TE}): cell = 4;
    end""",
        "P",
    )
    np.testing.assert_allclose(got, K_SYNTH * _int_cell_event(t), rtol=1e-5, atol=1e-7)
    assert got[-1] > 1.2 * (K_SYNTH * V0 * t[-1])  # post-resize rate doubled


@pytest.mark.parametrize("profile", ["rate_rule", "event"])
def test_elementary_decay(profile):
    """``H → ∅`` with law ``cell·k·H`` (H amount-valued): d(amount)/dt = -cell·k·H ⇒
    H(t) = H0·exp(-k·∫cell). Pre-#131 it followed exp(-k·V_static·t)."""
    if profile == "rate_rule":
        antimony = f"""
        model m
          compartment cell = {V0}; substanceOnly species H in cell = 10;
          k={K_DECAY}; g={G}; cell' = g; J1: H -> ; cell*k*H;
        end"""
        integral = _int_cell_rate_rule
    else:
        antimony = f"""
        model m
          compartment cell = {V0}; substanceOnly species H in cell = 10;
          k={K_DECAY}; J1: H -> ; cell*k*H; E1: at (time > {TE}): cell = 4;
        end"""
        integral = _int_cell_event
    t, got = _amount(antimony, "H")
    np.testing.assert_allclose(got, 10.0 * np.exp(-K_DECAY * integral(t)), rtol=1e-5, atol=1e-7)
    # The throttled static-volume decay exp(-k·V_static·t) is materially slower.
    assert abs(got[-1] - 10.0 * np.exp(-K_DECAY * V0 * t[-1])) > 1e-3


def test_bare_amount_law_stays_volume_independent():
    """Control: a bare amount law ``k·H`` (no compartment factor) is independent of
    the compartment volume and must stay Elementary — H(t) = H0·exp(-k·t)
    regardless of the rate-rule growth of ``cell``."""
    t, got = _amount(
        f"""
    model m
      compartment cell = {V0}; substanceOnly species H in cell = 10;
      k={K_DECAY}; g={G}; cell' = g; J1: H -> ; k*H;
    end""",
        "H",
    )
    np.testing.assert_allclose(got, 10.0 * np.exp(-K_DECAY * t), rtol=1e-5, atol=1e-7)


# ── Finding 3 — Functional hOSU=true law divides by V_static ─────────────────
@pytest.mark.parametrize("profile", ["rate_rule", "event"])
def test_functional_synth(profile):
    """A functionDefinition-wrapped law ``synth(cell, k) = cell·k`` classifies
    Functional. The storage divide must use V_static so the live ``cell`` factor
    survives: amount = k·∫cell. Pre-#131 the event-resize case divided by the live
    symbol and stayed flat (P(6)=3.6 instead of 5.4)."""
    head = "function synth(v, kk) v*kk end\n"
    if profile == "rate_rule":
        antimony = (
            head
            + f"""
        model m
          compartment cell = {V0}; substanceOnly species P in cell = 0;
          k={K_SYNTH}; g={G}; cell' = g; J1: -> P; synth(cell, k);
        end"""
        )
        integral = _int_cell_rate_rule
    else:
        antimony = (
            head
            + f"""
        model m
          compartment cell = {V0}; substanceOnly species P in cell = 0;
          k={K_SYNTH}; J1: -> P; synth(cell, k); E1: at (time > {TE}): cell = 4;
        end"""
        )
        integral = _int_cell_event
    t, got = _amount(antimony, "P")
    np.testing.assert_allclose(got, K_SYNTH * integral(t), rtol=1e-5, atol=1e-7)
    assert abs(got[-1] - K_SYNTH * V0 * t[-1]) > 0.1 * got[-1]
