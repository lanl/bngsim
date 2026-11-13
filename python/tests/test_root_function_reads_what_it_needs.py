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


def _final(text: str, name: str, grid=(0.0, 10.0), **kw) -> float:
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(
        sample_times=list(grid), rtol=1e-10, atol=1e-12, **kw
    )
    return float(np.asarray(r.species)[-1, list(r.species_names).index(name)])


# ─── The root function's read plan: controls ─────────────────────────────────


@pytest.mark.parametrize(
    "trigger",
    [
        pytest.param("q >= 2.5", id="the state itself"),
        pytest.param("g >= 5", id="a chain of functions"),
        pytest.param("2*h >= 5", id="one function"),
        pytest.param("q + 0*rateOf(q) >= 2.5", id="rateOf (the whole model)"),
    ],
)
def test_a_crossing_between_rows_fires_at_its_time(trigger):
    """Control. q = t crosses 2.5 between the rows at 0 and 10, so CVODE's root
    search finds it, reading the trigger through what it reads. A turns on
    there: A(10) = 10 - 2.5. Unrelated functions (pad1, pad2) are left out of
    the plan."""
    text = (
        "species S = 3; q = 0; q' = 1; h := q; g := 2*h; pad1 := 7*S; pad2 := pad1 + 1;"
        f" A = 0; A' = on; on = 0; E1: at ({trigger}): on = 1;"
    )
    assert _final(text, "A") == pytest.approx(7.5, rel=1e-8)


def test_a_crossing_read_through_an_observable():
    """Control. Q is fed by a reaction, so in the SBML model its token is the
    same-named observable's total: the plan has to sum that observable for the
    root search to see Q reach 2.5 at t = 2.5."""
    text = "species Q = 0; J1: -> Q; 1; A = 0; A' = on; on = 0; E1: at (Q >= 2.5): on = 1;"
    assert _final(text, "A") == pytest.approx(7.5, rel=1e-8)


def test_an_assignment_reads_what_the_trigger_does_not():
    """Control. ``S = w`` reads w := 10*q, which the trigger does not, so the
    root function no longer evaluates it. The event fires between rows, at
    t = 2.5, and reads w there (25), not where the plan left it."""
    text = "species S = 3; q = 0; q' = 1; w := 10*q; E1: at (q >= 2.5): S = w;"
    assert _final(text, "S") == pytest.approx(25.0, rel=1e-10)


def test_a_state_switch_read_through_a_function():
    """Control. S' = -(1 + |d|) with d := S - 5: the residual's crossing at
    S = 5 is found by the root search, at t = ln 6. Closed form after it:
    S(t) = 6 - exp(t - ln 6)."""
    text = "species S = 10; k = 1; d := S - 5; J1: S -> ; k*(1 + abs(d))"
    assert _final(text, "S", grid=(0.0, 1.0, 3.0)) == pytest.approx(
        6.0 - math.exp(3.0) / 6.0, rel=1e-7
    )


# ─── A state switch's residual, its repeats computed once ────────────────────


def _chain(n: int, law: str) -> str:
    """r_i reads r_(i-1) three times, so r_n written in reads S 3**n times."""
    rules = "; ".join(f"r{i} := (r{i - 1} + r{i - 1}*r{i - 1} + 1/(1 + r{i - 1}))/3" for i in range(1, n + 1))
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
    at states along the run and off it: to the last bit at nine in ten of them
    or more, and within a few units in the last place at every one. Sharing a
    group changes which of ExprTk's nodes compute it, and with them the
    rounding of the same arithmetic; at the one state of a hundred where the
    BIOMD0000000385-like chain below differs, the shared value was the nearer
    one to the exact value."""
    model = bngsim.Model.from_antimony_string(_chain(4, law))
    conds = model.state_switch_root_conditions()
    assert conds
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=51)
    rng = np.random.default_rng(1669)
    exact = 0
    total = 0
    for cond in conds:
        for t, y in zip(r.time, np.asarray(r.species, float)):
            for scale in (1.0, 1.0 + 0.3 * rng.standard_normal()):
                compiled, written = model._core._state_switch_residual_values(
                    cond, float(t), (y * scale).tolist()
                )
                total += 1
                exact += compiled == written
                assert compiled == pytest.approx(written, rel=1e-14, abs=1e-300)
    assert exact >= 0.9 * total


def test_a_chain_of_rules_under_a_switch_costs_what_the_chain_does():
    """BIOMD0000000385's shape, made larger. A re-run of the model with the
    switch costs a few times one without it (the root search reads the
    residual at every step); with the residual written out in full it cost
    ~50 times (9.6 ms against 0.2 ms here)."""

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
    assert with_switch < 10.0 * smooth, (with_switch, smooth)
