"""What CVODE's root function costs per call (issue #1669).

Two costs made bngsim several times slower than libRoadRunner on re-runs of a
tail of BioModels:

* The root function evaluated every observable and every model function on
  each call, though most triggers read the clock alone. CVODE calls it after
  every step and at every output row. It now refreshes only what its
  expressions read (``NetworkModel::read_plan``). That changes no value, so
  the tests for it are controls: crossings read through each path the plan
  has to follow, found by CVODE's root search between output rows, where a
  plan that left something out would read it stale.
* A state switch's residual is its condition with every function it reads
  written in, so a rule read three times is written three times, at every
  level of a chain of rules. BIOMD0000000385's ten residuals came to 64,746
  nodes of 183 different subexpressions. Each repeated parenthesised group is
  now computed once (``ExprTkEvaluator::compile_preprocessed_shared``).
"""

from __future__ import annotations

import math
import time

import bngsim
import numpy as np
import pytest

# ─── The root function's read plan: controls ─────────────────────────────────
#
# S' = -S from 10 crosses 5 at t = ln 2, between the rows at 0 and 3, so CVODE's
# root search finds the crossing. The condition reads S itself, or a chain of
# functions d2 := d := S, or S through a function that also reads parameters;
# the plan has to follow each. Closed forms at t = 3. (A clock, q' = 1, would
# not do: a crossing of the clock is a stop the run schedules, not a root.)

_TIGHT = dict(rtol=1e-10, atol=1e-14)
_READS = [
    pytest.param("S", "", id="the species"),
    pytest.param("d2", "d := S; d2 := d;", id="a chain of functions"),
    pytest.param("2*h - 5", "h := (S + 5)/2;", id="a function of a parameter"),
]


def _at3(text: str, name: str) -> float:
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(sample_times=[0.0, 3.0], **_TIGHT)
    return float(np.asarray(r.species)[-1, list(r.species_names).index(name)])


@pytest.mark.parametrize(("x", "defs"), _READS)
def test_an_event_fires_at_its_crossing(x, defs):
    """Control. A' = on, and the event turns on on where S reaches 5:
    A(3) = 3 - ln 2."""
    text = (
        f"species S = 10; J1: S -> ; S; {defs} A = 0; A' = on; on = 0; E1: at ({x} <= 5): on = 1;"
    )
    assert _at3(text, "A") == pytest.approx(3.0 - math.log(2.0), rel=1e-8)


@pytest.mark.parametrize(("x", "defs"), _READS)
def test_a_rate_law_switches_at_its_crossing(x, defs):
    """Control. B is made at rate 1 once S is below 5: B(3) = 3 - ln 2."""
    text = (
        f"species S = 10; species B = 0; J1: S -> ; S; {defs} J2: -> B; piecewise(1, {x} < 5, 0)"
    )
    assert _at3(text, "B") == pytest.approx(3.0 - math.log(2.0), rel=1e-8)


@pytest.mark.parametrize(("x", "defs"), _READS)
def test_a_state_switch_bends_at_its_crossing(x, defs):
    """Control. S' = -S (1 + |x - 5|) bends where S reaches 5, at
    t1 = ln(3)/4 (above it S = 4/(1 - 0.6 exp(-4t))); below it
    S = 6/(1 + 0.2 exp(6 (t - t1)))."""
    text = f"species S = 10; {defs} J1: S -> ; S*(1 + abs({x} - 5))"
    t1 = math.log(3.0) / 4.0
    exact = 6.0 / (1.0 + 0.2 * math.exp(6.0 * (3.0 - t1)))
    assert _at3(text, "S") == pytest.approx(exact, rel=1e-6)


def test_an_assignment_reads_what_the_trigger_does_not():
    """Control. ``S = w`` reads w := 10*q, which the trigger does not, so the
    root function no longer evaluates it. The event fires at t = ln 2, and
    reads w there (10 ln 2), not where the plan left it."""
    text = (
        "species S = 10; species T = 3; J1: S -> ; S;"
        " q = 0; q' = 1; w := 10*q; E1: at (S <= 5): T = w;"
    )
    assert _at3(text, "T") == pytest.approx(10.0 * math.log(2.0), rel=1e-8)


def test_a_trigger_read_through_rate_of():
    """Control. A trigger that reads rateOf takes the whole model: S' = -S
    reaches -5 where S reaches 5."""
    text = (
        "species S = 10; J1: S -> ; S; A = 0; A' = on; on = 0; E1: at (rateOf(S) >= -5): on = 1;"
    )
    assert _at3(text, "A") == pytest.approx(3.0 - math.log(2.0), rel=1e-8)


# ─── A state switch's residual, its repeats computed once ────────────────────


def _chain(n: int, law: str) -> str:
    """r_i reads r_(i-1) three times, so r_n written in reads S 3**n times."""
    rules = "; ".join(
        f"r{i} := (r{i - 1} + r{i - 1}*r{i - 1} + 1/(1 + r{i - 1}))/3" for i in range(1, n + 1)
    )
    return f"species S = 10; k = 1; r0 := S/10; {rules}; J1: S -> ; k*S*(1 + {law})"


def test_the_residual_compiles_its_repeats_once():
    """The residual of |r_4 - 0.5| is 3**4 copies of the chain's bottom; what is
    compiled for it computes each level once."""
    model = bngsim.Model.from_antimony_string(_chain(4, "abs(r4 - 0.5)"))
    (cond,) = model.state_switch_root_conditions()
    source, program = model._core._state_switch_residual_program(cond)
    assert source and program
    assert program.count("var ") >= 4
    assert len(program) < len(source) / 3


@pytest.mark.parametrize(
    "law",
    [
        pytest.param("abs(r4 - 0.5)", id="abs over a chain"),
        pytest.param("max(r3, 2*r3*r3 - 0.25)", id="max over a chain"),
        pytest.param("min(r3, 0.4 + 0*r3^2)", id="min over a chain"),
    ],
)
def test_the_residual_reads_as_it_is_written(law):
    """What the run reads of the residual is the value of its text as written,
    at states along the run and off it: to the last bit at most of them (96
    to 100 in a hundred, measured), and within 1e-12 relative at every one.
    Sharing a group changes which of ExprTk's nodes compute it, and with them
    the rounding of the same arithmetic. Near a root the residual cancels, and
    a few units in the last place of its terms are more of what is left: a
    fuzz of 350 models read 1.3e-13 at worst. At the one state of a hundred
    where the chain below differed, the shared value was the nearer one to the
    exact value."""
    model = bngsim.Model.from_antimony_string(_chain(4, law))
    conds = model.state_switch_root_conditions()
    assert conds
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=51)
    rng = np.random.default_rng(1669)
    exact = 0
    total = 0
    for cond in conds:
        for t, y in zip(r.time, np.asarray(r.species, float), strict=True):
            for scale in (1.0, 1.0 + 0.3 * rng.standard_normal()):
                compiled, written = model._core._state_switch_residual_values(
                    cond, float(t), (y * scale).tolist()
                )
                total += 1
                exact += compiled == written
                assert compiled == pytest.approx(written, rel=1e-12, abs=1e-300)
    assert exact >= 0.75 * total


def test_a_chain_of_rules_under_a_switch_costs_what_the_chain_does():
    """BIOMD0000000385's shape, made larger. A re-run of the model with the
    switch costs a few times one without it (the root search reads the
    residual at every step); with the residual written out in full it cost
    ~50-60 times (9.6 ms against 0.2 ms here), and costs ~4 times now."""

    def rerun(law: str) -> float:
        model = bngsim.Model.from_antimony_string(_chain(6, law))
        sim = bngsim.Simulator(model, method="ode", jacobian="fd")
        sim.run(t_span=(0.0, 10.0), n_points=101)
        best = math.inf
        for _ in range(5):
            model.reset()
            t0 = time.perf_counter()
            sim.run(t_span=(0.0, 10.0), n_points=101)
            best = min(best, time.perf_counter() - t0)
        return best

    with_switch = rerun("abs(r6 - 0.5)")
    smooth = rerun("(r6 - 0.5)^2")
    assert with_switch < 15.0 * smooth, (with_switch, smooth)
