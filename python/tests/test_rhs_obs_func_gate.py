"""T1 — ODE RHS observable/function-eval gate.

``compute_derivs_core`` refreshes observable totals + function-bound parameters
(``update_observables`` + ``evaluate_functions``) only when the model has
functions. The ExprTk evaluator is the sole RHS consumer of an observable sum,
and it runs on the RHS solely inside ``evaluate_functions``; so for a pure
mass-action model both passes are dead work and are skipped — while staying
byte-identical, since the rate loop reads species concentrations and parameter
values directly.

These tests use the instrumentation counters exposed on the core model:
``rhs_eval_count`` (every RHS call) and ``rhs_observable_eval_count`` (the subset
that ran the two passes). Mass-action ⇒ the latter stays 0; functional ⇒ the two
are equal.

The ODE default path (``codegen=False``) is the interpreted ``compute_derivs_core``
this gate lives in.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from bngsim import Model, Simulator


def test_mass_action_skips_rhs_obs_func_eval(simple_decay_net: Path) -> None:
    """A pure mass-action model must never run the observable/function passes."""
    model = Model.from_net(simple_decay_net)
    # Build-time gate decision: no functions ⇒ the RHS skips both passes.
    assert model._core.rhs_evaluates_observables is False

    sim = Simulator(model, method="ode")
    model._core.reset_rhs_counters()
    result = sim.run(t_span=(0, 50), n_points=51)

    # The RHS ran many times (CVODE takes many internal steps)...
    assert model._core.rhs_eval_count > 0
    # ...but NOT ONCE did it run update_observables / evaluate_functions.
    assert model._core.rhs_observable_eval_count == 0

    # And the answer is still correct: A(t) = 100*exp(-0.1 t).
    for i in range(result.n_times):
        t = result.time[i]
        assert result.species[i, 0] == pytest.approx(100.0 * math.exp(-0.1 * t), abs=1e-4)


def test_functional_model_runs_rhs_obs_func_eval(time_func_net: Path) -> None:
    """A model with functions must run both passes on every RHS evaluation."""
    model = Model.from_net(time_func_net)
    assert model._core.rhs_evaluates_observables is True

    sim = Simulator(model, method="ode")
    model._core.reset_rhs_counters()
    sim.run(t_span=(0, 10), n_points=21)

    # Every RHS call ran the observable/function passes (gate always true).
    assert model._core.rhs_eval_count > 0
    assert model._core.rhs_observable_eval_count == model._core.rhs_eval_count


def test_reset_rhs_counters(simple_decay_net: Path) -> None:
    """reset_rhs_counters() zeroes both counters."""
    model = Model.from_net(simple_decay_net)
    sim = Simulator(model, method="ode")
    sim.run(t_span=(0, 10), n_points=11)
    assert model._core.rhs_eval_count > 0

    model._core.reset_rhs_counters()
    assert model._core.rhs_eval_count == 0
    assert model._core.rhs_observable_eval_count == 0
