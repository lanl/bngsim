"""Issue #764 — an event assignment that reads rateOf(species) gets its jump.

The forward-sensitivity event jump differentiates each assignment h by central
differences over the state and the requested parameters. rateOf(A) is bound to a
buffer that only a derivative probe refreshes, and the difference never ran that
probe, so every perturbation read the same frozen dA/dt: ∂h/∂x = 0. The
assignment's support also listed no parameters for a rateOf read, so every
parameter column was skipped. The assigned row came back 0 in every column, and
so did every species downstream of it, with no warning. The trajectory was
right.

Closed form, with a fixed fire time τ (so no ∂t*/∂p term): A(t) = A0·e^(−a·t),
the assignment sets Y = rateOf(A)(τ) = −a·A(τ), and B, made at rate −Y after
the event, reaches B(T) = a·A(τ)·(T − τ). So

    dY/da  = −A(τ)·(1 − a·τ)        dY/dA0 = −a·e^(−a·τ)
    dB/da  = −dY/da·(T − τ)         dB/dA0 = −dY/dA0·(T − τ)

The same model with the assignment written out as ``Y = −a*A`` is the control:
it never read rateOf and was always right.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

A0, A_RATE, TAU, T_END = 10.0, 0.5, 3.0, 6.0

MODEL = """
species A, Y, B
A = A0; Y = 0; B = 0
A0 = {A0}; a = {a}; tau = {tau}
J0: A -> ; a*A
J1: -> B; -Y
E: at (time >= tau): Y = {rhs}
"""


def _run(rhs: str):
    text = MODEL.format(A0=A0, a=A_RATE, tau=TAU, rhs=rhs)
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "A0"]).run(
        sample_times=np.linspace(0.0, T_END, 7), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    s = np.asarray(run.sensitivities)[-1]
    return s[names.index("Y")], s[names.index("B")], np.asarray(run.species)[-1], names


def _closed_form():
    a_tau = A0 * math.exp(-A_RATE * TAU)
    dy = np.array([-a_tau * (1.0 - A_RATE * TAU), -A_RATE * math.exp(-A_RATE * TAU)])
    return dy, -dy * (T_END - TAU)


@pytest.mark.parametrize("rhs", ["rateOf(A)", "-a*A"], ids=["rateOf", "written-out"])
def test_the_assigned_row_and_its_downstream_match_the_closed_form(rhs):
    dy, db = _run(rhs)[:2]
    want_dy, want_db = _closed_form()
    np.testing.assert_allclose(dy, want_dy, rtol=1e-6)
    np.testing.assert_allclose(db, want_db, rtol=1e-6)


def test_the_trajectory_was_never_the_problem():
    """Y⁺ = −a·A(τ) and B(T) = a·A(τ)·(T − τ) on the rateOf spelling too."""
    _, _, x, names = _run("rateOf(A)")
    a_tau = A0 * math.exp(-A_RATE * TAU)
    assert x[names.index("Y")] == pytest.approx(-A_RATE * a_tau, rel=1e-7)
    assert x[names.index("B")] == pytest.approx(A_RATE * a_tau * (T_END - TAU), rel=1e-7)


def test_rateof_through_an_assignment_rule():
    """``r := rateOf(A)`` then ``Y = r``: the rule is a function, evaluated
    before the rateOf buffer was refreshed, so each difference read the
    previous state's buffer and the column came out −½× the truth."""
    text = MODEL.format(A0=A0, a=A_RATE, tau=TAU, rhs="r").replace(
        "J0: A -> ; a*A\n", "J0: A -> ; a*A\nr := rateOf(A)\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "A0"]).run(
        sample_times=np.linspace(0.0, T_END, 7), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    s = np.asarray(run.sensitivities)[-1]
    want_dy, want_db = _closed_form()
    np.testing.assert_allclose(s[names.index("Y")], want_dy, rtol=1e-6)
    np.testing.assert_allclose(s[names.index("B")], want_db, rtol=1e-6)


@pytest.mark.parametrize(
    "params",
    [["a", "A0", "thr"], ["thr"], ["thr", "a"], ["a"], ["A0", "thr"], ["thr", "a", "A0"]],
    ids="-".join,
)
def test_a_trigger_reading_rateof_through_a_rule(params):
    """The trigger side, through sync_model_at: ``r := rateOf(A)`` fires when
    ``r > −thr``, at t* = ln(a·A0/thr)/a, and B then grows at 1, so
    dB/dθ = −dt*/dθ: [(ln(a·A0/thr) − 1)/a², −1/(a·A0), 1/(a·thr)]. The thr
    column came out −4 (−2×), read one sync behind like the assignment above.

    Whatever is requested, and in any order (issue #910). The root pass that
    confirms the rising edge evaluated the rule before it refreshed the rateOf
    buffer, so it read ``r`` one probe behind and could miss the rise. The event
    was then fired a moment later by the re-check that follows every root, with
    no sensitivity jump, and every column came back 0. Whether it did depended on
    where the root landed, which the requested columns move: only requests with
    both ``a`` and ``A0`` in them were right."""
    model = bngsim.Model.from_antimony_string(
        "species A, Y, B; A = A0; Y = 0; B = 0; A0 = 10; a = 0.5; thr = 1\n"
        "J0: A -> ; a*A\nJ1: -> B; Y\nr := rateOf(A)\n"
        "E: at (r > -thr): Y = 1\n"
    )
    run = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
        sample_times=np.linspace(0.0, T_END, 7), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    got = np.asarray(run.sensitivities)[-1, names.index("B"), :]
    thr = 1.0
    want = {
        "a": (math.log(A_RATE * A0 / thr) - 1.0) / A_RATE**2,
        "A0": -1.0 / (A_RATE * A0),
        "thr": 1.0 / (A_RATE * thr),
    }
    np.testing.assert_allclose(got, [want[p] for p in params], rtol=1e-5)
