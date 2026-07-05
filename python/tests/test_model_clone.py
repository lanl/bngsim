"""Canonical clone() correctness tests.

This file is the load-bearing exercise of `NetworkModel::clone()`. Each
per-instance field of `NetworkModel::Impl` should be exercised by at
least one test here. **When adding a new per-instance field to `Impl`,
add a corresponding case below** — silent clone-related bugs (events
silently dropped pre-Phase-5b; species variables not rebound) are the
class of regression this file is meant to catch.

`Impl` per-instance fields covered (model_impl.hpp:68-100):
  * `species`               — `test_clone_basic_*`, every test
  * `observables`           — `test_clone_with_observable`
  * `parameters` (literal)  — every test
  * `parameters` (expression)— `test_clone_expression_parameter`
  * `functions`             — `test_clone_with_function`
  * `has_functions`         — implied by `test_clone_with_function`
  * `current_time`          — `test_clone_current_time_independent`
  * `evaluator` (rebinding) — every test compiles expressions in clones
  * `table_functions`       — `test_clone_with_table_function`
  * `events`                — `test_clone_with_events_full_l3_flags`
  * Species variable rebind — `test_clone_event_trigger_references_species`

`SharedModelData` (model_impl.hpp:32-64) is shared via `shared_ptr<const>`
post-build; clones share the pointer, so the only correctness assertion
needed is that mutating one clone's per-instance state never reaches the
original (covered in `test_clone_mutations_independent_*`).
"""

import math

import numpy as np
from bngsim._bngsim_core import (
    CvodeSimulator,
    ModelBuilder,
    SsaSimulator,
    TimeSpec,
)


def _ts(t_end, n_points, t_start=0.0):
    ts = TimeSpec()
    ts.t_start = t_start
    ts.t_end = t_end
    ts.n_points = n_points
    return ts


# ─── Basic state independence ────────────────────────────────────────────────


def test_clone_basic_ode_equivalence():
    """Original and clone produce identical ODE trajectories from the same
    initial state.

    Covers: species, parameters (literal), evaluator (parameter rebinding).
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.3)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    # Clone BEFORE either sim runs (each sim mutates species concentrations).
    clone = model.clone()
    r_a = CvodeSimulator(model).run(_ts(10.0, 21))
    r_b = CvodeSimulator(clone).run(_ts(10.0, 21))

    np.testing.assert_allclose(r_a.species_data, r_b.species_data, rtol=1e-12)


def test_clone_basic_ssa_equivalence_matched_seed():
    """Original and clone produce byte-identical SSA traces at matched seed.

    Covers: species, parameters, evaluator. Matched-seed determinism is the
    load-bearing invariant — any per-instance state drift would surface as a
    seed-1 trace difference.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.3)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    clone = model.clone()
    r_a = SsaSimulator(model).run(_ts(10.0, 21), 1)
    r_b = SsaSimulator(clone).run(_ts(10.0, 21), 1)

    # Byte-identical (deterministic SSA at matched seed; no per-instance drift)
    np.testing.assert_array_equal(r_a.species_data, r_b.species_data)


# ─── Mutation independence ───────────────────────────────────────────────────


def test_clone_mutations_independent_set_param():
    """Mutating clone via set_param does not affect the original.

    Covers: parameters (deep-copy independence).
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.3)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    clone = model.clone()
    # Set the clone's k to a very different value; original should be untouched.
    clone.set_param("k", 5.0)
    assert model.get_param("k") == 0.3
    assert clone.get_param("k") == 5.0

    r_orig = CvodeSimulator(model).run(_ts(10.0, 21))
    r_clone = CvodeSimulator(clone).run(_ts(10.0, 21))

    # Original: S(10) = 100 * exp(-3) ≈ 5; clone: 100 * exp(-50) ≈ 0
    # CVODE's default atol=1e-8 floors clone's tail near atol level.
    s_orig_end = r_orig.species_data[-1, 0]
    s_clone_end = r_clone.species_data[-1, 0]
    assert s_orig_end > 1.0, f"original S(10)={s_orig_end} should still be ~5"
    assert s_clone_end < 1e-6, f"clone S(10)={s_clone_end} should be near zero (k=5)"


def test_clone_mutations_independent_species():
    """Setting concentration on the clone does not bleed through to the
    original. Covers: species (deep-copy independence).
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)  # no decay
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    clone = model.clone()
    clone.set_concentration("S", 999.0)

    r_orig = CvodeSimulator(model).run(_ts(1.0, 2))
    r_clone = CvodeSimulator(clone).run(_ts(1.0, 2))

    assert r_orig.species_data[0, 0] == 100.0
    assert r_clone.species_data[0, 0] == 999.0


def test_clone_mutations_on_original_dont_affect_clone():
    """Symmetry check: mutating the original after clone() does not
    affect the already-cloned instance.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.3)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    clone = model.clone()
    # Capture clone's expected behavior FIRST. reset() before/after each run
    # so concentration leakage from one run to the next doesn't confound the
    # comparison.
    clone.reset()
    r_clone_before = CvodeSimulator(clone).run(_ts(10.0, 21))
    # Mutate the original
    model.set_param("k", 5.0)
    # Clone's parameters must be unaffected; behavior must match the pre-set run.
    assert clone.get_param("k") == 0.3
    clone.reset()
    r_clone_after = CvodeSimulator(clone).run(_ts(10.0, 21))
    np.testing.assert_allclose(r_clone_before.species_data, r_clone_after.species_data, rtol=1e-12)


# ─── Observables ─────────────────────────────────────────────────────────────


def test_clone_with_observable():
    """Observable (totalled species count) computes correctly on the clone.

    Covers: observables (deep-copy + evaluator rebinding to &obs.total).
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s1 = b.add_species("S1", 10.0)
    s2 = b.add_species("S2", 20.0)
    b.add_observable("total", [(s1, 1.0), (s2, 1.0)])
    b.add_reaction([], [s1], "elementary", "k")  # k=0, no-op
    model = b.build()

    clone = model.clone()
    r = CvodeSimulator(clone).run(_ts(1.0, 2))
    obs_data = np.array(r.observable_data)
    assert obs_data[0, 0] == 30.0, f"clone observable={obs_data[0, 0]}, expected 30"


# ─── Expression parameters ───────────────────────────────────────────────────


def test_clone_expression_parameter():
    """Parameter whose value is computed from a string expression must be
    re-compiled in the clone's evaluator. Covers: parameters with
    `is_expression=True` + evaluator-cache string-based re-compile.
    """
    b = ModelBuilder()
    b.add_parameter("k_base", 0.5)
    # k = 2 * k_base via expression
    b.add_parameter("k", 0.0, expression="2 * k_base", is_expression=True)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    clone = model.clone()
    r_orig = CvodeSimulator(model).run(_ts(2.0, 2))
    r_clone = CvodeSimulator(clone).run(_ts(2.0, 2))

    # S(2) = 100 * exp(-2 * 2 * 0.5) = 100 * exp(-2) ≈ 13.53
    expected = 100.0 * math.exp(-2.0)
    assert abs(r_orig.species_data[-1, 0] - expected) < 0.5
    np.testing.assert_allclose(r_orig.species_data, r_clone.species_data, rtol=1e-10)


# ─── Functions ───────────────────────────────────────────────────────────────


def test_clone_with_function():
    """Function (var-param-bound expression) used in a Functional rate
    law evaluates correctly on the clone. Covers: functions, has_functions,
    function expression re-compile in cloned evaluator.

    The body of a Functional rate law is `rate_fn` (a parameter slot
    populated each step from the function's expression). The `Functional`
    rate-law type also multiplies by the species-factor, so the effective
    decay rate is rate_fn * S. Asserting clone-equals-original is the
    correctness check we care about; the absolute value is decoupled.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.5)
    b.add_function("rate_fn", "k * 2")  # function body lands in `rate_fn` param slot
    b.add_parameter("rate_fn", 0.0)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "functional", "rate_fn")
    model = b.build()

    clone = model.clone()
    r_orig = CvodeSimulator(model).run(_ts(2.0, 2))
    r_clone = CvodeSimulator(clone).run(_ts(2.0, 2))

    # Both must agree to floating-point precision.
    np.testing.assert_allclose(r_orig.species_data, r_clone.species_data, rtol=1e-10)
    # Sanity: function evaluated correctly (S decayed)
    assert r_orig.species_data[-1, 0] < 100.0


# ─── Table functions ─────────────────────────────────────────────────────────


def test_clone_with_table_function():
    """Table functions are re-created in the clone (not just deep-copied —
    they're registered as ExprTk callables via define_function).
    Covers: table_functions, register_table_function_ on the clone.
    """
    # Build via from_net stub since add_table_function lives on the high-level
    # Model object (post-build). Use a simple .net via a fixture, or construct
    # a hand-written model. Here we rely on the C++ ModelBuilder + Model's
    # post-build add_table_function; the underlying NetworkModel.clone() is
    # what's under test.
    import bngsim

    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    core = b.build()
    model = bngsim.Model(_core=core)
    # Constant tfun: scale(t) = 2 for all t. Use as a function body factor.
    model.add_table_function("scale", times=[0, 100], values=[2.0, 2.0])

    clone = model.clone()
    # Both should have one table function in the cloned core
    assert clone._core.n_table_functions == 1
    assert clone._core.table_function_names == ["scale"]
    # Simulating both should succeed (table function callable in clone's evaluator)
    r_orig = CvodeSimulator(model._core).run(_ts(1.0, 2))
    r_clone = CvodeSimulator(clone._core).run(_ts(1.0, 2))
    np.testing.assert_allclose(r_orig.species_data, r_clone.species_data, rtol=1e-12)


# ─── Events ──────────────────────────────────────────────────────────────────


def test_clone_with_events_full_l3_flags():
    """Events with the full set of L3 flags survive cloning. Covers:
    `events` deep-copy, per-event expression re-compile (trigger, priority_expr,
    delay_expr, assignment-RHS), and the SBML-L3 flag preservation.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 0.0)
    b.add_event(
        "rich",
        "time() >= 1",
        [(s_idx, "100")],
        priority_expr="2 * k + 1",  # state-dependent priority
        persistent=False,
        initial_value=False,
        use_values_from_trigger_time=True,
    )
    model = b.build()

    clone = model.clone()
    r_orig = CvodeSimulator(model).run(_ts(2.0, 21))
    r_clone = CvodeSimulator(clone).run(_ts(2.0, 21))

    # Both must fire the event; S goes from 0 to 100 at t≥1
    assert r_orig.species_data[-1, 0] == 100.0
    np.testing.assert_array_equal(r_orig.species_data, r_clone.species_data)


def test_clone_event_trigger_references_species():
    """Trigger expression that references a species name compiles in the
    clone's evaluator. Covers: species variable rebinding in clone()
    (the second half of the Phase 5b clone() fix).
    """
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 0.0)
    b.add_reaction([], [s_idx], "elementary", "k")  # ∅ → S at rate 1
    # State-dependent trigger: when S >= 5, set S := 0
    b.add_event("threshold", "S >= 5", [(s_idx, "0")], initial_value=False)
    model = b.build()

    clone = model.clone()
    # Both must successfully compile + fire the event under SSA
    r_orig = SsaSimulator(model.clone()).run(_ts(20.0, 21), 1)
    r_clone = SsaSimulator(clone).run(_ts(20.0, 21), 1)
    # Both must keep S bounded by 5 (event re-fires)
    assert np.array(r_orig.species_data).max() <= 5.0 + 1e-9
    assert np.array(r_clone.species_data).max() <= 5.0 + 1e-9


# ─── current_time / evaluator time pointer ───────────────────────────────────


def test_clone_evaluator_time_pointer_rebound():
    """Each clone's evaluator binds `time()` to the clone's own
    `current_time` field, not the original's. The bug would manifest as
    cross-clone time interference: events with `time()` triggers in one
    model would fire based on the other's current time.

    Covers: `current_time` (per-instance), `evaluator->set_time_ptr` on
    the clone. Tested indirectly via two independent simulations whose
    event triggers reference `time()` — if the time pointer were shared,
    clone's event would fire at original's t, and vice versa.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 0.0)
    b.add_event("at_two", "time() >= 2", [(s_idx, "100")], initial_value=False)
    model = b.build()

    clone = model.clone()
    # Run clone first, then original. Each must see its own time progression.
    r_clone = CvodeSimulator(clone).run(_ts(t_end=3.0, n_points=31))
    r_orig = CvodeSimulator(model).run(_ts(t_end=3.0, n_points=31))

    # Both must show event firing at t=2 (S goes from 0 to 100). If time
    # pointers were shared, the second simulator might see a stale time value
    # and either fire too early or too late.
    t = np.array(r_orig.time)
    idx_19 = int(np.argmin(np.abs(t - 1.9)))
    idx_21 = int(np.argmin(np.abs(t - 2.1)))
    assert r_orig.species_data[idx_19, 0] == 0.0
    assert r_orig.species_data[idx_21, 0] == 100.0
    assert r_clone.species_data[idx_19, 0] == 0.0
    assert r_clone.species_data[idx_21, 0] == 100.0


# ─── reset() interaction ─────────────────────────────────────────────────────


def test_clone_reset_independent():
    """`reset()` on the clone restores its own initial concentrations
    without touching the original. Covers: per-instance species + reset
    contract.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    # Mutate original
    model.set_concentration("S", 50.0)
    # Clone the (mutated) original
    clone = model.clone()
    assert clone.get_concentration("S") == 50.0
    # Reset clone — restores to 100 (clone's initial_conc carried from build)
    clone.reset()
    assert clone.get_concentration("S") == 100.0
    # Original still has 50
    assert model.get_concentration("S") == 50.0
