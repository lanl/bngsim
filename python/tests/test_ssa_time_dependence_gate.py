"""Issue #654 — the SSA's time-dependence gate must not be decided by sampling.

``SsaSimulator::run_internal`` gates its piecewise-constant sub-stepping on
whether the model's rates move with the clock. That question used to be
answered by evaluating every function at three times — ``t_start``, the
midpoint and ``t_end`` — and concluding "time-invariant" when the three values
agreed.

Three values do not pin down a function. A rate of period 5 run over ``[0, 10]``
is probed at 0, 5 and 10 — one whole period apart each time — so it read as
constant, sub-stepping never engaged, the rate froze at its ``t_start`` value
for the whole run, and the simulator returned a trajectory computed against a
rate that is not the model's, with no warning and no error.

The load-bearing test is :func:`test_aliasing_horizon_matches_analytic`: the
model below has an exact answer, so this is not a comparison of two plausible
stochastic results. ``A -> B`` with ``kf(t) = 0.5·(1 + sin(2πt/5))`` is a pure
decay with time-varying rate, so

    A(10) = 200·exp(−∫₀¹⁰ kf) = 200·exp(−5) = 1.348

and the aliased run returned 9.135 — 6.8× too high. Its companion
:func:`test_horizon_does_not_change_the_answer` states the same defect as an
invariant with no analytics in it: A(10) cannot depend on whether the run was
asked to stop at 10 or at 11.

The gate is now syntactic (``NetworkModel::functions_use_time``): does a
function expression name ``time``, or read a time-indexed table function. That
cannot alias. It can over-report — a function that names ``time`` but is
constant over the window now sub-steps — and that direction costs only time.
"""

from __future__ import annotations

import math

import numpy as np
from bngsim._bngsim_core import ModelBuilder, SsaSimulator, TimeSpec

# Period 5, so the old probe points over [0, 10] land 0, 2 and 4 periods along.
PERIOD = 5.0
KF = f"0.5*(1.0 + sin(2*{math.pi:.15f}*time()/{PERIOD}))"
A0 = 200.0
N_REPS = 200


def _ts(t_end: float, n_points: int) -> TimeSpec:
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_points
    return ts


def _decay_model():
    """``A -> B`` at the periodic rate ``kf``; nothing else moves."""
    b = ModelBuilder()
    a = b.add_species("A", A0)
    bb = b.add_species("B", 0.0)
    b.add_observable("Atot", [(a, 1.0)])
    b.add_function("kf", KF)
    b.add_reaction([a], [bb], "functional", "kf")
    return b.build()


def _a_at(t_end: float, n_points: int, row: int) -> np.ndarray:
    """A at output row `row` for each of N_REPS seeded runs of `[0, t_end]`."""
    out = np.empty(N_REPS)
    for s in range(N_REPS):
        model = _decay_model()
        result = SsaSimulator(model).run(_ts(t_end, n_points), s + 1)
        species = np.asarray(result.species_data).reshape(-1, result.n_species)
        out[s] = species[row, 0]
    return out


# ─── The probe really does alias on this model ───────────────────────────────


def test_the_three_old_probe_points_coincide():
    """Why this horizon: `kf` takes the same value at 0, 5 and 10.

    Without this, a later reader cannot tell that `[0, 10]` is the adversarial
    horizon rather than an arbitrary one, and could "simplify" the test into
    one the old gate would also have passed. The comparison is the old gate's
    own — `|v - base| > 1e-12·(1 + |base|)` — not exact equality, because that
    is the threshold the three samples had to clear to be called moving.
    """

    def kf(t):
        return 0.5 * (1.0 + math.sin(2 * math.pi * t / PERIOD))

    base = kf(0.0)
    for t in (0.5 * 10.0, 10.0):
        assert abs(kf(t) - base) <= 1e-12 * (1.0 + abs(base)), t
    # And it is emphatically not constant in between.
    assert kf(1.25) > base + 0.4


# ─── The dynamics ────────────────────────────────────────────────────────────


def test_aliasing_horizon_matches_analytic():
    """A(10) over the aliased horizon must be 200·e⁻⁵, not 6.8× that.

    ∫₀¹⁰ kf = 5 exactly (the sine integrates to zero over two whole periods),
    so every replicate is Binomial(200, e⁻⁵) and the analytic mean is 1.348.
    Pre-fix this horizon returned 9.135.
    """
    expected = A0 * math.exp(-5.0)
    a = _a_at(10.0, 11, 10)
    se = a.std(ddof=1) / math.sqrt(N_REPS)
    z = abs(a.mean() - expected) / (se + 1e-12)
    assert z < 5.0, (
        f"mean A(10) = {a.mean():.3f} over {N_REPS} seeds vs the analytic "
        f"{expected:.3f} (z = {z:.1f}, 5·SE = {5 * se:.3f}); the periodic rate "
        f"is not being tracked."
    )


def test_horizon_does_not_change_the_answer():
    """Same model, same seeds, same reported time — two horizons, one answer.

    The horizon decides only where the old probe points landed, so this states
    the defect without an oracle: `[0, 11]` probes 0, 5.5 and 11 and never
    aliased, `[0, 10]` probes 0, 5 and 10 and did. Both report t = 10.
    """
    aliased = _a_at(10.0, 11, 10)
    honest = _a_at(11.0, 12, 10)
    se = math.sqrt(aliased.var(ddof=1) + honest.var(ddof=1)) / math.sqrt(N_REPS)
    z = abs(aliased.mean() - honest.mean()) / (se + 1e-12)
    assert z < 5.0, (
        f"A(10) = {aliased.mean():.3f} when the run stops at t=10 but "
        f"{honest.mean():.3f} when it stops at t=11 (z = {z:.1f}); the "
        f"trajectory depends on the horizon."
    )


# ─── The gate itself ─────────────────────────────────────────────────────────


def _one_reaction_model(func_expr: str, *, tfun_index: str | None = None):
    b = ModelBuilder()
    a = b.add_species("A", 5.0)
    bb = b.add_species("B", 0.0)
    b.add_parameter("p", 1.0)
    b.add_function("kf", func_expr)
    if tfun_index is not None:
        b.add_inline_table_function_spec(
            "kf", [0.0, 5.0, 10.0], [1.0, 2.0, 3.0], tfun_index, "linear"
        )
    b.add_reaction([a], [bb], "functional", "kf")
    return b.build()


def test_gate_is_true_for_a_function_naming_time():
    assert _decay_model().functions_use_time


def test_gate_is_false_for_a_constant_function():
    """The flag is exact in this direction: no `time` token, no sub-stepping."""
    assert not _one_reaction_model("0.3*p").functions_use_time


def test_gate_is_true_for_a_time_indexed_table_function():
    """A tfun indexed on the clock reads it without naming `time` itself."""
    for index in ("time", "t"):
        assert _one_reaction_model("kf", tfun_index=index).functions_use_time, index


def test_gate_is_false_for_a_parameter_indexed_table_function():
    assert not _one_reaction_model("kf", tfun_index="p").functions_use_time


# `clone()` carries the flag — covered canonically by
# test_model_clone.py::test_clone_carries_functions_use_time.
