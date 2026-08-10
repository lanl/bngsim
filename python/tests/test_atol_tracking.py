"""Tracking absolute tolerance — issue #213.

Issue #196 gave the state axis a per-species absolute tolerance, which removes
the **cross-species** compromise: a model whose species span decades no longer
has to pick one number that is a tolerance for the largest and the smallest at
once. It does not touch the **within-species, over-time** version of the same
problem. Whatever number species ``i`` is given, it keeps for the whole run, so
a species that starts at order one and decays to something tiny outgrows its own
tolerance partway through and stops being error-controlled from there on.

``deep_decay.net`` is the minimum reproducer, and it is built so the #196 vector
is *provably* a no-op on it: both live species start at exactly 1.0, so
``atol="auto"`` derives 1e-8 for every species — the same number the scalar it
replaces would have used. One of them then decays sixteen decades and the other
barely moves. At the default tolerance the decaying one comes back 4000x too
large.

CVODE's construct for this is ``CVodeWFtolerances``, a user-supplied error-weight
function, spelled :class:`bngsim.TrackingAtol` here. The rule::

    atol_i(y) = clamp(rtol * |y_i|, ceiling_i * 10**-decades, ceiling_i)

The analytical solution is the oracle throughout — every accuracy number below
is measured against ``exp(-k t)``, not against another bngsim run.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# The package namespace, not ``bngsim._atol`` — same reason as #196/#212: the
# tests should exercise the surface a consumer actually gets.
from bngsim import TrackingAtol

# Matches the .net header. D decays sixteen decades over the run; C is here to
# show that a species which does NOT decay is left alone.
_T_END = 30.0
_N_POINTS = 31
_D0, _KD = 1.0, 1.0
_C0, _KC = 1.0, 1.0e-4
_RTOL = 1e-8

# D(30) = exp(-30). Written out so a reader can check the assertions by hand.
_D_END_EXACT = 9.357622968840175e-14


@pytest.fixture
def decay_net(data_dir: Path) -> Path:
    """Path to deep_decay.net."""
    return data_dir / "deep_decay.net"


@pytest.fixture
def decay_sim(decay_net: Path):
    """Simulator over the deep-decay model, plus the model."""
    model = bngsim.Model.from_net(str(decay_net))
    return bngsim.Simulator(model, method="ode"), model


def _run(sim, model, **run_kwargs):
    """Run and return (worst relative error in D, worst in C, D(t_end), n_steps)."""
    model.reset()
    result = sim.run(
        t_span=(0.0, _T_END), n_points=_N_POINTS, rtol=_RTOL, max_steps=1_000_000, **run_kwargs
    )
    names = model.species_names
    t = result.time
    d = result.species[:, names.index("D()")]
    c = result.species[:, names.index("C()")]
    exact_d = _D0 * np.exp(-_KD * t)
    exact_c = _C0 * np.exp(-_KC * t)
    return (
        float(np.max(np.abs(d - exact_d) / exact_d)),
        float(np.max(np.abs(c - exact_c) / exact_c)),
        float(d[-1]),
        int(result.solver_stats.get("n_steps", 0)),
    )


# ─── The premise: the #196 vector cannot see this ─────────────────────────────


def test_the_per_species_vector_is_a_no_op_on_this_model(decay_sim):
    """Both species start at 1.0, so ``"auto"`` derives the scalar it replaces.

    This is the whole reason #213 is a separate issue rather than a wider
    version of #196: the cross-species fix has nothing to grip on here.
    """
    sim, model = decay_sim
    assert list(sim.auto_atol(rtol=_RTOL)) == [_RTOL] * model.n_species

    scalar = _run(sim, model, atol=_RTOL)
    vector = _run(sim, model, atol="auto")
    assert vector == scalar, "the per-species vector should change nothing here"


def test_the_decaying_species_is_unresolved_under_a_fixed_tolerance(decay_sim):
    """The defect: below its own atol, D stops being error-controlled."""
    sim, model = decay_sim
    err_d, err_c, d_end, _ = _run(sim, model, atol="auto")

    assert err_d > 1e3, "expected D to be wrong by orders of magnitude"
    assert d_end > 1e3 * _D_END_EXACT, "D(30) comes back thousands of times too large"
    assert err_c < 1e-6, "...while C, which never leaves order one, is fine — that is the point"


# ─── The fix ──────────────────────────────────────────────────────────────────


def test_tracking_resolves_the_decay(decay_sim):
    """Against the analytical exp(-t), across sixteen decades."""
    sim, model = decay_sim
    err_d, err_c, d_end, _ = _run(sim, model, atol=TrackingAtol())

    assert err_d < 1e-4, "D is error-controlled against its own current magnitude now"
    assert d_end == pytest.approx(_D_END_EXACT, rel=1e-4)
    assert err_c < 1e-6, "...and C is unharmed"


def test_the_string_token_is_the_default_spec(decay_sim):
    sim, model = decay_sim
    assert _run(sim, model, atol="tracking") == _run(sim, model, atol=TrackingAtol())


def test_tracking_costs_steps(decay_sim):
    """Stated rather than hidden: resolving twelve decades is real work."""
    sim, model = decay_sim
    _, _, _, n_fixed = _run(sim, model, atol="auto")
    _, _, _, n_tracking = _run(sim, model, atol=TrackingAtol())

    assert n_tracking > 2 * n_fixed


# ─── The two ends of the clamp ────────────────────────────────────────────────


def test_depth_zero_reproduces_the_ceiling_vector(decay_sim):
    """``decades=0`` is the ceiling vector exactly — a strict extension of #196.

    The ceiling here is deliberately NOT the auto vector: one entry is pinned
    twelve orders tighter, on a species that only ever grows. That makes the
    test discriminate the clamp's *upper* end. Without it, ``atol_i`` at
    ``decades=0`` would be ``max(rtol*|y_i|, ceiling_i)`` rather than
    ``ceiling_i``, that species would run at 1e-8 instead of 1e-20, and the run
    would land on the third row below instead of the first two.

    (The two agreed bit-for-bit when this was written, which is what the
    structure predicts — both compute ``rtol*|y| + atol`` with the same two
    roundings — but that is a property of the compiler, not a contract, so the
    assertion is a tolerance far tighter than the effect it has to catch.)
    """
    sim, model = decay_sim
    ceiling = [1e-8] * model.n_species
    ceiling[model.species_names.index("Dd()")] = 1e-20

    plain_vector = _run(sim, model, atol=ceiling)
    depth_zero = _run(sim, model, atol=TrackingAtol(decades=0, ceiling=ceiling))
    untightened = _run(sim, model, atol=1e-8)

    assert depth_zero[0] == pytest.approx(plain_vector[0], rel=1e-9)
    assert depth_zero[3] == plain_vector[3]
    # ...and that is a different run from the one a dropped upper clamp gives.
    assert plain_vector[3] != untightened[3]


def test_depth_bounds_how_far_the_tolerance_tightens(decay_sim):
    """The clamp's *lower* end: ``decades`` is a real limit, not a formality.

    Monotone in the depth, by orders of magnitude per rung, which is what says
    the floor is binding rather than the rule being tight everywhere anyway.
    """
    sim, model = decay_sim
    errors = [_run(sim, model, atol=TrackingAtol(decades=d))[0] for d in (0, 3, 6, 12)]

    assert errors[0] > 1e3  # the ceiling vector: unresolved
    assert 1.0 < errors[1] < errors[0]  # 3 decades: better, still wrong
    assert 1e-3 < errors[2] < 1.0  # 6 decades: within 10%
    assert errors[3] < 1e-4  # 12 decades: resolved


def test_tracking_is_never_looser_than_its_ceiling(decay_sim):
    """The clamp's upper end again, as the property a caller can rely on.

    Turning tracking on can only tighten the tolerance a species is held to, so
    it can only improve — never degrade — the accuracy the ceiling alone gives.
    """
    sim, model = decay_sim
    ceiling = list(sim.auto_atol(rtol=_RTOL))

    plain = _run(sim, model, atol=ceiling)
    tracked = _run(sim, model, atol=TrackingAtol(ceiling=ceiling))

    assert tracked[0] <= plain[0]
    assert tracked[1] <= plain[1] * 10  # C is at roundoff either way


# ─── The mode actually reaches CVODE, and only where asked ────────────────────


def test_depth_invalidates_the_warm_cvode_cache(decay_sim):
    """``CVodeReInit`` does not touch tolerances, so the depth is fingerprinted.

    Without it, the second run of an alternating sequence would silently keep
    the first run's error-weight function (or keep running without one).
    """
    sim, model = decay_sim

    scalar_first = _run(sim, model, atol=1e-8)
    tracked_first = _run(sim, model, atol=TrackingAtol())
    scalar_again = _run(sim, model, atol=1e-8)
    tracked_again = _run(sim, model, atol=TrackingAtol())
    shallower = _run(sim, model, atol=TrackingAtol(decades=6))

    assert scalar_again == scalar_first
    assert tracked_again == tracked_first
    assert tracked_first[3] != scalar_first[3]
    # A different depth is a different configuration, not a cache hit.
    assert shallower[3] != tracked_first[3]


def test_set_tolerances_pins_the_ceiling_and_is_cleared_by_a_scalar(decay_net: Path):
    """The Simulator-level form, and that the three modes replace each other."""
    model = bngsim.Model.from_net(str(decay_net))
    sim = bngsim.Simulator(model, method="ode")

    sim.set_tolerances(_RTOL, TrackingAtol())
    tracked = _run(sim, model)
    assert tracked[0] < 1e-4

    sim.set_tolerances(_RTOL, 1e-8)
    assert _run(sim, model) == _run(sim, model, atol=1e-8)


def test_a_batch_freezes_one_ceiling_for_every_row(decay_net: Path):
    """Same reason ``"auto"`` is frozen: rows held to different tolerances are
    not rows you can compare. The rule stays state-dependent *within* a row."""
    model = bngsim.Model.from_net(str(decay_net))
    sim = bngsim.Simulator(model, method="ode")
    idx = model.species_names.index("D()")

    results = sim.run_batch(
        (0.0, _T_END),
        _N_POINTS,
        params=[{"kD": _KD}, {"kD": _KD}],
        rtol=_RTOL,
        atol="tracking",
        max_steps=1_000_000,
    )
    for result in results:
        assert result.species[-1, idx] == pytest.approx(_D_END_EXACT, rel=1e-4)


def test_parameter_scan_reaches_the_tracking_path(decay_net: Path):
    model = bngsim.Model.from_net(str(decay_net))
    sim = bngsim.Simulator(model, method="ode")
    idx = model.species_names.index("D()")

    results = sim.parameter_scan(
        "kD",
        [_KD, _KD],
        t_span=(0.0, _T_END),
        n_points=_N_POINTS,
        rtol=_RTOL,
        atol="tracking",
        max_steps=1_000_000,
    )
    for result in results:
        assert result.species[-1, idx] == pytest.approx(_D_END_EXACT, rel=1e-4)


def test_steady_state_march_takes_the_mode_too(decay_net: Path):
    """The march is shared plumbing (issue #196 wired the vector into it).

    The convergence criterion is ``||f(y)||_2/n < tol``, a norm with no
    per-species reading, so tracking cannot change the *test* for having
    arrived — only the accuracy of the state that arrives. What this pins is
    that the option reaches the march at all and does not break the solve.
    """
    model = bngsim.Model.from_net(str(decay_net))
    sim = bngsim.Simulator(model, method="ode")
    mask = ~model.is_pure_sink()  # Dd/Cd are accumulators — issue #74

    model.reset()
    plain = sim.steady_state(atol="auto", mask=mask, max_time=1e6, tol=1e-12)
    model.reset()
    tracked = sim.steady_state(atol="tracking", mask=mask, max_time=1e6, tol=1e-12)

    assert plain.converged and tracked.converged
    assert tracked.n_steps != plain.n_steps


# ─── The sensitivity axis (the #196 interaction, re-decided) ──────────────────


def test_sensitivities_run_under_tracking_and_come_back_resolved(decay_net: Path):
    """``atolS`` keeps reading the CEILING — deliberately — and costs nothing here.

    #196 made ``atolS`` derive from the state axis's per-species tolerance. With
    a tracking state tolerance there is no vector to read, so the derivation
    reads the ceiling: turning tracking on leaves the sensitivity tolerances
    exactly where the same vector would have put them without it. The
    alternative, re-deriving ``atolS`` from the live state, would make that base
    *tighten* mid-run, which is the hazard the issue #183 high-water mark exists
    to avoid.

    The measurement is why that is affordable rather than merely defensible: the
    sensitivity column is resolved anyway, because the state axis is what drives
    the step size. ``d/dkD[exp(-kD t)] = -t exp(-kD t)`` is the oracle.
    """
    model = bngsim.Model.from_net(str(decay_net))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kD"])
    idx = model.species_names.index("D()")

    def sens_err(**kwargs) -> float:
        model.reset()
        result = sim.run(
            t_span=(0.0, _T_END),
            n_points=_N_POINTS,
            rtol=_RTOL,
            max_steps=1_000_000,
            **kwargs,
        )
        got = result.sensitivities[:, idx, 0]
        exact = -result.time * _D0 * np.exp(-_KD * result.time)
        live = exact != 0.0
        return float(np.max(np.abs(got[live] - exact[live]) / np.abs(exact[live])))

    fixed = sens_err(atol="auto")
    tracked = sens_err(atol="tracking")

    assert fixed > 1e2, "the sensitivity of a species under its own atol is noise too"
    assert tracked < 1e-4


# ─── The contract ─────────────────────────────────────────────────────────────


def test_a_zero_ceiling_entry_is_rejected(decay_sim):
    """A zero ceiling is a zero floor, and a zero floor is an infinite weight.

    Legal for a *fixed* vector (CVODE's own weight routines test for it before
    inverting) and not legal here, because CVODE explicitly skips that test for
    a user-supplied weight function.
    """
    sim, model = decay_sim
    ceiling = [1e-8] * model.n_species
    ceiling[1] = 0.0
    with pytest.raises(ValueError, match=r"strictly > 0 under a tracking"):
        sim.run(t_span=(0.0, 1.0), atol=TrackingAtol(ceiling=ceiling))


def test_a_wrong_length_ceiling_is_rejected(decay_sim):
    sim, model = decay_sim
    with pytest.raises(ValueError, match=r"entries but the model has"):
        sim.run(t_span=(0.0, 1.0), atol=TrackingAtol(ceiling=[1e-8, 1e-8]))


@pytest.mark.parametrize("decades", [-1.0, float("inf"), float("nan")])
def test_an_unusable_depth_is_rejected_at_construction(decades):
    with pytest.raises(ValueError, match=r"finite and >= 0"):
        TrackingAtol(decades=decades)


def test_a_floor_that_underflows_is_rejected_by_name(decay_sim):
    """Rather than passed on: CVODE reports an unmeetable tolerance as a
    corrector convergence failure at t=0, which names nothing."""
    sim, model = decay_sim
    with pytest.raises(ValueError, match=r"subnormal or zero"):
        sim.run(t_span=(0.0, 1.0), atol=TrackingAtol(decades=20, ceiling=1e-300))


def test_a_mistyped_token_names_the_two_that_exist(decay_sim):
    sim, model = decay_sim
    with pytest.raises(ValueError, match=r"unknown atol token"):
        sim.run(t_span=(0.0, 1.0), atol="trackng")


def test_a_solver_failure_under_tracking_names_tracking(decay_sim):
    """CVODE's own report never mentions the tolerance mode that caused it.

    Measured on 391 rr_parity models that integrate at the default tolerance: 6
    do not at ``decades=12``, 1 at 6, none at 3. So a tracking depth is much the
    likeliest reason a model that integrated a moment ago suddenly does not, and
    "CVODE integration failed ... with flag=-4" points nowhere.

    Provoked here with an absurd depth rather than a corpus model, deliberately:
    the models that fail at the *default* depth are all corpus models, and a
    test gated on the corpus skips in every worktree and in CI.
    """
    sim, model = decay_sim
    with pytest.raises(bngsim.SimulationError, match=r"tracking absolute tolerance 50 decades"):
        sim.run(
            t_span=(0.0, _T_END),
            n_points=6,
            rtol=_RTOL,
            atol=TrackingAtol(decades=50, ceiling=1e-250),
            max_steps=5000,
        )


def test_a_solver_failure_without_tracking_says_nothing_about_it(decay_sim):
    """The other half: the hint is not glued onto every failure."""
    sim, model = decay_sim
    with pytest.raises(bngsim.SimulationError) as excinfo:
        sim.run(t_span=(0.0, _T_END), n_points=6, rtol=_RTOL, atol=1e-250, max_steps=5000)
    assert "tracking" not in str(excinfo.value)


def test_the_capability_is_feature_detectable():
    """``hasattr`` is the probe; the version string is not one (issue #212)."""
    assert hasattr(bngsim, "TrackingAtol")
    assert bngsim.TRACKING == "tracking"
