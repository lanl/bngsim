"""GH #242 — same-instant event cascade + seed-keyed random tie-break.

Two coupled pieces, shipped together:

1. **Immediate (delay-0) same-instant cascade** (SBML L3v2 §4.11.6). After an
   event fires, a delay-0 event its assignment just pushed false→true joins the
   SAME instant's batch. The CVODE root finder cannot see this discrete jump, so
   without the cascade the second event never fires. This is the mechanism the
   last GH #233 case (suite 00978) needs.

2. **Seed-keyed random tie-break.** Among simultaneous events sharing the maximum
   priority, one is chosen at random each round (§4.11.6) via a per-run RNG seeded
   from ``SolverOptions.event_seed``. The draw happens ONLY at a genuine tie, so
   distinct-priority ordering stays deterministic and a fixed seed makes an
   equal-priority model fully reproducible (the PyBNF-fitting requirement). The
   ODE path uses a fixed default seed, so a no-arg run is reproducible out of the
   box; passing ``seed=`` selects an independent event-ordering realization.

These models are hand-built (no test-suite dependency); the suite cases they
mirror are named in each test.
"""

import bngsim
import numpy as np
import pytest


def _run(sbml: str, *, t_end: float, n_points: int, seed=None):
    model = bngsim.Model.from_sbml_string(sbml)
    return model, bngsim.Simulator(model, method="ode").run(
        t_span=(0, t_end), n_points=n_points, rtol=1e-9, atol=1e-11, seed=seed
    )


def _final(result, name: str) -> float:
    return float(np.asarray(result.observables[name])[-1])


# ─── Part 1: immediate same-instant cascade ──────────────────────────────────

# A delay-0 event whose trigger a *just-fired* event's assignment satisfies. The
# trigger flips via a discrete jump (trig 0→1), which the root finder never sees;
# only the same-instant cascade re-check fires B. Old (delayed-only) cascade left
# x=0. This is 00978's core mechanism in miniature.
_CASCADE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="cascade">
    <listOfParameters>
      <parameter id="trig" value="0" constant="false"/>
      <parameter id="x" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="A" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>1</cn></apply></math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="trig">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="B" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>trig</ci><cn>0.5</cn></apply></math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><plus/><ci>x</ci><cn>1</cn></apply></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_immediate_cascade_fires_delay0_triggered_event():
    """A sets trig=1 at t=1; B (trigger trig>0.5, delay 0) must fire in the SAME
    instant — a discrete jump the root finder can't detect. x=1 proves the
    immediate cascade; the old delayed-only cascade left x=0."""
    _, r = _run(_CASCADE_SBML, t_end=5, n_points=6)
    assert _final(r, "x") == pytest.approx(1.0)


# 00978 in miniature: five higher-priority flip events alternately arm/disarm a
# counter's trigger (rise, fall, rise, fall, rise = 3 rises). The persistent,
# assignment-time counter increments once per rise ⇒ x=3. A single-batch drain
# without the same-instant cascade would fire the counter 0 times.
_MULTIRISE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="multirise">
    <listOfParameters>
      <parameter id="m" value="0" constant="false"/>
      <parameter id="x" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      {flips}
      <event id="C" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>m</ci><cn>0.5</cn></apply></math>
        </trigger>
        <priority>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
        </priority>
        <listOfEventAssignments>
          <eventAssignment variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><plus/><ci>x</ci><cn>1</cn></apply></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def _flip(eid: str, priority: int, value: int) -> str:
    return f"""
      <event id="{eid}" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>1</cn></apply></math>
        </trigger>
        <priority>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>{priority}</cn></math>
        </priority>
        <listOfEventAssignments>
          <eventAssignment variable="m">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>{value}</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>"""


def test_multi_rise_cascade_counts_priority_ordered_rises():
    """Higher-priority flips (priorities 10..6) set m to 1,0,1,0,1 in that order;
    the priority-1 counter C fires once per 0→1 rise ⇒ x=3 (mirrors 00978's z)."""
    flips = (
        _flip("E0", 10, 1)
        + _flip("E1", 9, 0)
        + _flip("E2", 8, 1)
        + _flip("E3", 7, 0)
        + _flip("E4", 6, 1)
    )
    _, r = _run(_MULTIRISE_SBML.replace("{flips}", flips), t_end=5, n_points=6)
    assert _final(r, "x") == pytest.approx(3.0)


# ─── Part 2: seed-keyed random tie-break ─────────────────────────────────────

# 00952 in miniature: two equal-priority, non-persistent events with the same
# trigger, each resetting the shared clock (disabling the other) and bumping its
# own counter. Exactly one fires per round, chosen at random. Q+R is therefore
# seed-independent (it counts rounds); which of Q/R gets the increment is not.
_COMPETE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="compete">
    <listOfParameters>
      <parameter id="reset" value="0" constant="false"/>
      <parameter id="Q" value="0" constant="false"/>
      <parameter id="R" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      {qinc}
      {rinc}
    </listOfEvents>
  </model>
</sbml>"""


def _compete_event(eid: str, counter: str) -> str:
    return f"""
      <event id="{eid}" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="false">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <apply><minus/>
                <csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
                <ci>reset</ci></apply>
              <cn>1</cn></apply></math>
        </trigger>
        <priority>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
        </priority>
        <listOfEventAssignments>
          <eventAssignment variable="reset">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol></math>
          </eventAssignment>
          <eventAssignment variable="{counter}">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><plus/><ci>{counter}</ci><cn>1</cn></apply></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>"""


_COMPETE = _COMPETE_SBML.replace("{qinc}", _compete_event("Qinc", "Q")).replace(
    "{rinc}", _compete_event("Rinc", "R")
)


def test_competing_events_exactly_one_fires_per_round():
    """Each round exactly one of the mutually-disabling equal-priority events
    fires: Q+R equals the round count, and neither runs away (0 < Q < Q+R)."""
    _, r = _run(_COMPETE, t_end=20, n_points=201, seed=7)
    q, rr = _final(r, "Q"), _final(r, "R")
    # 20 time units, trigger every 1.0 ⇒ ~19 completed rounds.
    assert q + rr == pytest.approx(round(q + rr))  # integer number of fires
    assert q + rr >= 15
    assert 0 < q < q + rr  # both competitors get selected at least once


def test_same_seed_is_reproducible():
    """A fixed seed reproduces the full trajectory bit-for-bit — the PyBNF
    reproducibility contract for equal-priority event models."""
    _, r1 = _run(_COMPETE, t_end=20, n_points=201, seed=12345)
    _, r2 = _run(_COMPETE, t_end=20, n_points=201, seed=12345)
    np.testing.assert_array_equal(
        np.asarray(r1.observables["Q"]), np.asarray(r2.observables["Q"])
    )


def test_different_seeds_change_the_realization():
    """The seed actually selects the event ordering: at least one of several
    seeds splits Q/R differently from seed 0 (the RNG is genuinely wired)."""
    base_q = _final(_run(_COMPETE, t_end=40, n_points=401, seed=0)[1], "Q")
    assert any(
        _final(_run(_COMPETE, t_end=40, n_points=401, seed=s)[1], "Q") != base_q
        for s in (1, 2, 3, 4, 5)
    )


def test_default_seed_is_reproducible_across_runs():
    """With no explicit seed the ODE path uses a fixed default, so consecutive
    no-arg runs are identical (not drawn fresh from entropy like SSA)."""
    _, r1 = _run(_COMPETE, t_end=20, n_points=201)
    _, r2 = _run(_COMPETE, t_end=20, n_points=201)
    np.testing.assert_array_equal(
        np.asarray(r1.observables["Q"]), np.asarray(r2.observables["Q"])
    )


# Two events with DISTINCT priorities assigning the same variable: the RNG must
# not be consulted, so the outcome is identical for every seed (byte-identical to
# the old deterministic drain — the guarantee that protects non-tie models).
_DISTINCT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="distinct">
    <listOfParameters>
      <parameter id="w" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="Elo" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>1</cn></apply></math>
        </trigger>
        <priority><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math></priority>
        <listOfEventAssignments>
          <eventAssignment variable="w">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>10</cn></math></eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="Ehi" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>1</cn></apply></math>
        </trigger>
        <priority><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math></priority>
        <listOfEventAssignments>
          <eventAssignment variable="w">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>20</cn></math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_distinct_priorities_are_seed_independent():
    """Ehi (priority 2) fires before Elo (priority 1); Elo fires last ⇒ w=10, the
    same for every seed. Distinct priorities never touch the RNG."""
    finals = [_final(_run(_DISTINCT_SBML, t_end=5, n_points=6, seed=s)[1], "w") for s in range(8)]
    assert all(w == pytest.approx(10.0) for w in finals)


# ─── Seed stamping contract ──────────────────────────────────────────────────


def test_event_model_stamps_seed():
    """An ODE run of a model WITH events exposes its event seed on the Result."""
    _, r = _run(_COMPETE, t_end=5, n_points=6, seed=4242)
    assert r.seed == 4242


def test_event_free_ode_seed_is_none():
    """An ODE run of a model with NO events stays seed-less (the deterministic-
    ODE contract from test_seed_semantics is preserved)."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="free">
    <listOfParameters><parameter id="p" value="1" constant="false"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="p">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><cn>1</cn>
            <csymbol encoding="text"
              definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol></apply></math>
      </assignmentRule>
    </listOfRules>
  </model>
</sbml>"""
    _, r = _run(sbml, t_end=5, n_points=6)
    assert r.seed is None


# ─── Delayed competing events (01590) ────────────────────────────────────────

# Both competitors carry a delay, so at the trigger instant neither has yet reset
# the shared clock — both are scheduled. The disabling happens when the FIRST
# delayed apply lands: the second's non-persistent trigger lapses and it is
# cancelled before it can apply. Net: exactly one increments per round (Q+R is
# the round count), not both. Without the interleaved cancellation both applied
# and Q==R every round (maxdiff stayed 0). Mirrors suite 01590.
_DELAYED_COMPETE = _COMPETE_SBML.replace(
    "{qinc}",
    _compete_event("Qinc", "Q").replace(
        "<listOfEventAssignments>",
        "<delay><math xmlns=\"http://www.w3.org/1998/Math/MathML\"><cn>0.25</cn></math></delay>"
        "<listOfEventAssignments>",
    ),
).replace(
    "{rinc}",
    _compete_event("Rinc", "R").replace(
        "<listOfEventAssignments>",
        "<delay><math xmlns=\"http://www.w3.org/1998/Math/MathML\"><cn>0.25</cn></math></delay>"
        "<listOfEventAssignments>",
    ),
)


def test_delayed_competing_events_mutually_exclude():
    """With a delay on both competitors, the interleaved delayed-apply cancels the
    loser once the winner resets the clock, so exactly ONE increments per round:
    Q+R counts rounds (~16 over t=20 at a ~1.25 period), NOT double. The old
    apply-all-at-once bug fired both every round (Q+R ≈ 32, Q==R). Mirrors 01590."""
    splits = set()
    for seed in range(6):
        _, r = _run(_DELAYED_COMPETE, t_end=20, n_points=201, seed=seed)
        q, rr = _final(r, "Q"), _final(r, "R")
        total = q + rr
        assert total == pytest.approx(round(total))  # whole number of fires
        assert 10 <= total <= 22  # one fire/round (~16), decisively NOT ~32
        splits.add((q, rr))
    # The random tie-break yields more than one (Q,R) split across seeds.
    assert len(splits) > 1
