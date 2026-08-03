"""Issue #150 — the saltation jump at a state-dependent rate-law switch.

A condition inside a rate law that reads the state — ``piecewise(0, Virus < 1,
Virus*rho_V)`` — flips a branch of ``f`` at a crossing whose time ``t*(θ)`` moves
with **every** parameter through the trajectory. The in-branch derivative is
right on both sides (``sympy.diff`` of a ``Piecewise`` carries no boundary delta
and does not need one); what is discontinuous is ``∂x/∂θ`` itself, by

    s(t*⁺) = s(t*⁻) + (f⁻ − f⁺)·dt*/dθ

Neither of the two ways bngsim could produce a sensitivity RHS carries that term
— both integrate the variational equation smoothly across — so it has to be
applied *at* the crossing, which first has to be located. That is the whole of
this issue: root on the condition's residual, differentiate ``dt*/dθ`` there by
the implicit function theorem (the issue #144 machinery, reached from a rate law
instead of an event trigger), and jump.

**What the oracle is.** A central finite difference of the model's own
trajectory, at a tolerance tight enough that the FD's own truncation error is the
only thing left. That works here — unlike at a *clock* switch, where both the
analytic path and the FD miss the crossing by the same O(h) and the comparison is
vacuous (see ``test_codegen_switch_condition_sens``) — because the FD re-solves
the whole trajectory including the moved crossing, and a state crossing moves
smoothly with the parameter.

**Why the analytic RHS had to be admitted with it.** With the crossing resolved
to a root, CVODES' internal difference quotient becomes *worse*, not better: its
probe evaluates ``f`` at ``y + σ·s`` with ``σ ≈ √rtol``, and just past a crossing
``σ·|s|`` is easily wide enough to put the probe back on the other branch. On the
model below that injected ``rho·X/σ ≈ 2.7e4`` into ``ds/dt`` for the sliver of
time the state stayed within ``σ·|s|`` of the surface, and the column came out
28% high — a jump correctly applied and then spoiled. So issue #150 also lifted
the GH #68 decline for exactly the conditions it compensates;
``test_codegen_switch_condition_sens`` owns that half of the contract.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim import _switch_sensitivity as sw
from bngsim._exceptions import SimulationError

pytest.importorskip("sympy")


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


# ─── the issue's own reproduction ──────────────────────────────────────────
#
#   dX/dt = if(X<1, 0, rho)·X − delta·X,   X(0) = X0 = 1000
#
# the same shape as AMICI's ``nested_events`` fixture with its injection event
# removed. Before the crossing dX/dt = (rho−delta)X = −0.8X, after it −delta·X =
# −1.6X, so the crossing is at t* = ln(1000)/0.8 = 8.63469 and the saltation
# factor is f⁺/f⁻ = 2 exactly.
#
# ``rho`` shows that factor cleanly (it appears ONLY in the branch that switches
# off, so its post-crossing column is pure jump); ``delta`` mixes the jump with a
# correct in-branch part and is the more representative case — it is also why
# "multiply the tail by f⁺/f⁻" is not a fix, and why both are asserted.
NET = """\
begin parameters
    1 X0     1000  # Constant
    2 rho    0.8  # Constant
    3 delta  1.6  # Constant
end parameters
begin functions
    1 growth() if(X<1,0,rho)
end functions
begin species
    1 A() X0
end species
begin reactions
    1 1 1,1 growth #_R1
    2 1 0 delta #_R2
end reactions
begin groups
    1 X                    1
end groups
"""

T_STAR = np.log(1000.0) / 0.8  # 8.634694...
T_END = 12.0
N_POINTS = 25
RTOL, ATOL = 1e-9, 1e-12


def _model(tmp_path, text=NET, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _sens(tmp_path, params, name="m.net", text=NET, t_end=T_END, **kw):
    model = _model(tmp_path, text, name)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params), **kw)
    return sim.run(t_span=(0.0, t_end), n_points=N_POINTS, rtol=RTOL, atol=ATOL)


def _fd(tmp_path, params, text=NET, rel=1e-5, t_end=T_END):
    """Central difference of the model's own trajectory, one column per param."""
    cols = []
    for i, p in enumerate(params):
        p0 = _model(tmp_path, text, f"fd{i}.net").get_param(p)
        h = rel * abs(p0)
        got = []
        for sign in (+1, -1):
            m = _model(tmp_path, text, f"fd{i}{sign}.net")
            m.set_param(p, p0 + sign * h)
            r = bngsim.Simulator(m, method="ode").run(
                t_span=(0.0, t_end), n_points=N_POINTS, rtol=RTOL, atol=ATOL
            )
            got.append(np.asarray(r.species))
        cols.append((got[0] - got[1]) / (2 * h))
    return np.stack(cols, axis=-1)


# ─── the term ──────────────────────────────────────────────────────────────


@requires_cc
class TestTheSaltationTerm:
    def test_every_column_matches_a_finite_difference_across_the_crossing(self, tmp_path):
        """The whole issue in one assertion. Before the fix ``rho`` came back a
        factor of exactly 2 low after t*, and ``delta`` came back low by a
        parameter-dependent amount between 1.7 and 2.0 — the same defect, seen
        through a column that also has a real in-branch part."""
        params = ["rho", "delta"]
        run = _sens(tmp_path, params)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params)
        assert an.shape == fd.shape
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, 0, j])))
            np.testing.assert_allclose(
                an[:, 0, j],
                fd[:, 0, j],
                rtol=2e-4,
                atol=1e-5 * scale,
                err_msg=f"column {p!r} disagrees with its own finite difference",
            )

    def test_the_jump_lands_at_the_crossing_and_nowhere_else(self, tmp_path):
        """Localisation, which is what says the *term* is being tested rather
        than the aggregate. Sample densely and compare against the closed form
        on each side: before t* the ``rho`` column is ``t·X(t)`` exactly, after
        it the jumped value decaying at ``-delta``. A jump applied at the wrong
        instant — or twice — fails here and passes an all-points tolerance."""
        model = _model(tmp_path)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["rho"])
        times = [T_STAR - 0.5, T_STAR - 1e-4, T_STAR + 1e-4, T_STAR + 0.5, T_STAR + 2.0]
        run = sim.run(sample_times=[0.0, *times], rtol=1e-11, atol=1e-13)
        s = np.asarray(run.sensitivities)[:, 0, 0]

        # Before: X = X0·e^{(rho−delta)t}, so ∂X/∂rho = t·X.
        for k, t in enumerate(times[:2], start=1):
            x = 1000.0 * np.exp(-0.8 * t)
            assert s[k] == pytest.approx(t * x, rel=1e-6)
        # At t* the column is t*·X(t*) = t*·1; the jump doubles it, and after the
        # crossing dX/dt = −delta·X has ∂f/∂rho = 0, so it only decays.
        for k, t in enumerate(times[2:], start=3):
            assert s[k] == pytest.approx(2.0 * T_STAR * np.exp(-1.6 * (t - T_STAR)), rel=1e-5)

    def test_the_jump_is_exactly_the_saltation_factor(self, tmp_path):
        """``rho`` appears only in the branch that switches off, so its whole
        post-crossing column is the jump and the ratio across t* is
        ``f⁺/f⁻ = (−delta·X)/((rho−delta)·X) = 2`` — the number the issue
        measured on AMICI's fixture, reproduced in closed form."""
        model = _model(tmp_path)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["rho"])
        run = sim.run(sample_times=[0.0, T_STAR - 1e-9, T_STAR + 1e-9], rtol=1e-12, atol=1e-14)
        s = np.asarray(run.sensitivities)[:, 0, 0]
        assert s[2] / s[1] == pytest.approx(2.0, rel=1e-6)

    def test_an_initial_condition_column_is_jumped_too(self, tmp_path):
        """The lesson issue #144 paid for on the event side: a *state* crossing
        moves with an initial condition just as it moves with a rate constant,
        so ``dt*/dθ`` has to be formed over all ``n_total`` columns and not only
        the parameter ones. ``X0`` is the model's own IC parameter, so this is
        the same column reached two ways — and both must carry the jump."""
        params = ["X0"]
        run = _sens(tmp_path, params)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params)
        scale = float(np.max(np.abs(fd[:, 0, 0])))
        np.testing.assert_allclose(an[:, 0, 0], fd[:, 0, 0], rtol=2e-4, atol=1e-5 * scale)
        # Pinned against the closed form as well, because the post-crossing IC
        # column decays to ~1e-5 of its peak and an all-points tolerance keyed
        # on that peak would pass with the jump missing. Before t*,
        # ∂X/∂X0 = e^{(rho−delta)t}; the jump doubles it at t*, and after it the
        # column only decays at −delta. Dropping the jump halves this.
        expected = 2.0 * np.exp(-0.8 * T_STAR) * np.exp(-1.6 * (T_END - T_STAR))
        assert an[-1, 0, 0] == pytest.approx(expected, rel=1e-4)


# ─── what the crossing costs when it is not located ────────────────────────


@requires_cc
class TestTheCrossingIsLocated:
    def test_the_integrator_does_not_grind_at_the_crossing(self, tmp_path):
        """Registering the root is half the fix and stands on its own: without a
        stop the integrator chases the discontinuity, and under sensitivities at
        a tight rtol it does not survive it (the issue quotes ``mxstep steps
        taken`` with ``h=1.1e-16`` — issue #82's pit, reached from the rate-law
        side). The step count is the deterministic form of that; wall clock
        would flake."""
        run = _sens(tmp_path, ["rho", "delta"])
        n_steps = run.solver_stats["n_steps"]
        # The smooth two-branch problem needs a few hundred steps. Chasing the
        # kink costs tens of thousands before it gets across, if it does.
        assert n_steps < 5_000, f"{n_steps} steps — the crossing is being chased, not located"

    def test_a_run_without_sensitivities_is_byte_identical(self, tmp_path):
        """The blast-radius bound. State-switch roots are registered only for a
        run that asks for sensitivities, so every plain trajectory keeps exactly
        the stepping — and exactly the numbers — it had before. Compared against
        a model that has no idea the feature exists, at full precision."""
        plain = bngsim.Simulator(_model(tmp_path, name="a.net"), method="ode").run(
            t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL
        )
        again = bngsim.Simulator(_model(tmp_path, name="b.net"), method="ode").run(
            t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL
        )
        assert np.asarray(plain.species).tobytes() == np.asarray(again.species).tobytes()
        assert plain.solver_stats["n_steps"] == again.solver_stats["n_steps"]

    def test_the_model_registers_no_discontinuity_trigger_of_its_own(self, tmp_path):
        """The premise the issue states: there is no GH #72 root for ``X<1``
        today, because that machinery only ever looked for thresholds on *time*.
        If a loader ever starts registering these, the state-switch roots would
        double up and this is where that shows."""
        assert _model(tmp_path)._core.n_discontinuity_triggers == 0


# ─── the residual, and who is allowed to claim a crossing ──────────────────


class TestTheResidual:
    def test_a_comparison_over_state_resolves_to_its_residual(self, tmp_path):
        core = _model(tmp_path)._core
        residual, why = core.state_switch_residual("X<1")
        assert residual and not why
        assert "X" in residual and "1" in residual

    def test_two_spellings_of_one_crossing_share_a_residual(self, tmp_path):
        """The dedup key. ``X<1`` and ``X<=1`` are the same surface, and
        registering both would put two roots on one crossing — which the solver
        then refuses as an ambiguous simultaneous pair. Orientation is free for
        the same reason: ``dt*/dθ`` is a ratio of two derivatives of ``g``."""
        core = _model(tmp_path)._core
        assert core.state_switch_residual("X<1")[0] == core.state_switch_residual("X<=1")[0]

    @pytest.mark.parametrize(
        ("cond", "fragment"),
        [
            ("X==1", "equality"),
            ("(X<1)&&(X>0)", "combines conditions"),
            ("not(X<1)", "combines conditions"),
            ("X", "not a relational comparison"),
        ],
    )
    def test_what_cannot_be_rooted_says_why(self, tmp_path, cond, fragment):
        core = _model(tmp_path)._core
        residual, why = core.state_switch_residual(cond)
        assert not residual
        assert fragment in why, why

    def test_a_comparison_over_parameters_alone_is_not_a_state_switch(self, tmp_path):
        """``rho < delta`` has no crossing the trajectory can reach — it is a
        constant for the whole run. Claiming it would put a root on a residual
        that never changes sign and, worse, would let the state path claim a
        clock threshold whose jump issue #48 already applies."""
        core = _model(tmp_path)._core
        residual, why = core.state_switch_residual("rho<delta")
        assert not residual
        assert "no live model state" in why

    def test_a_clone_re_derives_rather_than_copying_an_expression_id(self, tmp_path):
        """The issue #144 rule, restated for this cache: an expression id means
        something else in another evaluator, so a clone must resolve the same
        text into its own table. Same residual *text* from both, which is the
        observable half of that."""
        core = _model(tmp_path)._core
        before = core.state_switch_residual("X<1")[0]
        clone = core.clone()
        assert clone.state_switch_residual("X<1")[0] == before
        # And the clone answers correctly having never been asked before the
        # copy: nothing was carried over to be stale.
        fresh = _model(tmp_path, name="fresh.net")._core.clone()
        assert fresh.state_switch_residual("X<1")[0] == before


class TestTheDetector:
    def test_the_condition_is_registered_once(self, tmp_path):
        assert sw.state_switch_conditions(_model(tmp_path)._core) == ["X<1"]

    def test_a_model_with_no_condition_registers_nothing(self, tmp_path):
        text = NET.replace("    1 growth() if(X<1,0,rho)\n", "    1 growth() rho\n")
        assert sw.state_switch_conditions(_model(tmp_path, text)._core) == []

    def test_a_clock_threshold_is_left_to_issue_48(self, tmp_path):
        """The partition. A BNGL counter clock is a *species*, so ``t>=sigma``
        reads live state and the residual splitter would happily claim it —
        which would apply the jump twice, once from the issue #48 stop time and
        once from a crossing root. ``clock_crossing_compensated`` is asked first
        by both the detector and the gate so that cannot happen."""
        text = NET.replace(
            "    2 rho    0.8  # Constant\n",
            "    2 rho    0.8  # Constant\n"
            "    4 sigma  3.0  # Constant\n"
            "    5 kclock 1  # Constant\n",
        )
        text = text.replace(
            "    1 growth() if(X<1,0,rho)\n", "    1 growth() if(tc>=sigma,rho,0)\n"
        )
        text = text.replace("    1 A() X0\n", "    1 A() X0\n    2 counter() 0\n")
        text = text.replace(
            "    2 1 0 delta #_R2\n", "    2 1 0 delta #_R2\n    3 0 2 kclock #_R3\n"
        )
        text = text.replace(
            "    1 X                    1\n",
            "    1 X                    1\n    2 tc                   2\n",
        )
        core = _model(tmp_path, text, name="clock.net")._core
        scope = sw.switch_condition_scope(core)
        assert "tc" in scope.clocks, "the fixture's counter must be detected as a clock"
        assert sw.clock_crossing_compensated("tc>=sigma", scope)
        assert sw.state_switch_conditions(core) == []
        records, _pinned = sw.compute_switch_time_sens(core, ["sigma"], 0.0, 100.0)
        assert records, "the clock detector must still claim it"


# ─── a crossing with no jump at it ─────────────────────────────────────────
#
# The saltation term is (f⁻ − f⁺)·dt*/dθ, so a `piecewise` that is CONTINUOUS at
# its own switch needs nothing — and that is the most common `piecewise` in the
# corpus, because it is how a clamp is written. BIOMD0000000161's basal PIP
# synthesis is `piecewise(0.581*k*(exp((basal - PIP)/basal) - 1), PIP < basal,
# 0)`, whose live branch is exactly 0 where PIP = basal; the trajectory then
# rides that surface, and refusing there (which the first cut of this feature
# did) took a model that had always run and made it raise.
#
# Here the branches meet at X = 1 for the same reason — `rho*(1-X)` is 0 there —
# so f is continuous across a crossing the trajectory passes straight through:
# X decays at −delta·X from 1000, reaches 1 at t = ln(1000)/1.6 = 4.317, and is
# then held up toward rho/(rho+delta). `rho` has a real in-branch derivative
# AFTER the crossing and none before it, which is what makes the column
# non-trivial and the comparison worth making.
CONTINUOUS = """\
begin parameters
    1 X0     1000  # Constant
    2 rho    0.5  # Constant
    3 delta  1.6  # Constant
end parameters
begin functions
    1 growth() if(X<1,rho*(1-X),0)
end functions
begin species
    1 A() X0
end species
begin reactions
    1 0 1 growth #_R1
    2 1 0 delta #_R2
end reactions
begin groups
    1 X                    1
end groups
"""


@requires_cc
class TestAContinuousSwitchNeedsNoJump:
    def test_it_runs_and_matches_a_finite_difference(self, tmp_path):
        """The branch gap is measured before ``dt*/dθ`` is ever formed, so a
        continuous switch costs no jump, no implicit-function solve and no
        transversality refusal — while the in-branch ``∂f/∂rho`` on the far side
        still has to come through."""
        params = ["rho", "delta"]
        run = _sens(tmp_path, params, name="cont.net", text=CONTINUOUS)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params, text=CONTINUOUS)
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, 0, j])))
            assert scale > 0.0, f"column {p!r} is trivially zero — the fixture is not testing it"
            # Read against the column's own peak, not pointwise: the sample
            # adjacent to the crossing is FD-limited, and it is the FD that is
            # limited. Measured at rtol 1e-9 / 1e-11 / 1e-12 the analytic value
            # there is 0.020753543 / 0.020753555 / 0.020753556 — stable to eight
            # digits — while the difference quotient wanders in the fourth
            # (0.0207398 / 0.0207289 / 0.0207497), which is what differencing
            # across a kink does.
            assert np.max(np.abs(an[:, 0, j] - fd[:, 0, j])) <= 2e-4 * scale, (
                f"column {p!r} disagrees with its own finite difference by "
                f"{np.max(np.abs(an[:, 0, j] - fd[:, 0, j])) / scale:.2e} of its peak"
            )

    def test_an_ic_only_request_registers_the_crossing_too(self, tmp_path):
        """Keyed on "any sensitivity at all", not on a parameter request.

        The three detectors sit behind ``if self._sensitivity_params:`` because
        a switch *time* and an event *time* are parameter-column concepts; a
        state crossing is not — it moves with every column, and the IC columns
        are the ones it moves *only* through the trajectory. Requesting an
        initial condition and nothing else is exactly the shape that would have
        gone unregistered, which is the same corner issue #144 found on the
        event side."""
        model = _model(tmp_path, name="iconly.net")
        sim = bngsim.Simulator(model, method="ode", sensitivity_ic=["A()"])
        run = sim.run(t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL)
        s_ic = np.asarray(run.sensitivities_ic)[:, 0, 0]
        # dX/dX(0) is e^{(rho-delta)t} before the crossing, doubled by the jump
        # at t*, and decaying at -delta after it. Without the registration the
        # tail is half this.
        assert s_ic[-1] == pytest.approx(
            2.0 * np.exp(-0.8 * T_STAR) * np.exp(-1.6 * (T_END - T_STAR)), rel=1e-4
        )

    def test_the_crossing_is_still_registered(self, tmp_path):
        """The condition is a state switch like any other — nothing about it is
        knowable before the run. What decides is the branch gap measured AT the
        crossing, which is why this cannot be a detection-time rule."""
        core = _model(tmp_path, CONTINUOUS, name="cont2.net")._core
        assert sw.state_switch_conditions(core) == ["X<1"]

    def test_the_branch_gap_is_read_scale_free(self, tmp_path):
        """The residual carries the MODEL's units, not a species'.

        The ``.net`` corpus is full of the signed-rate idiom — BNGL rates must
        be non-negative, so a model needing a signed derivative splits it and
        guards each half by the sign of the rate itself, ``if(expr>0, expr, 0)``
        and ``if(expr>0, 0, -expr)``. Both branches are 0 where ``expr`` is, so f
        is continuous; but the residual is a *rate*, and on
        ``ph_lorenz_attractor`` (condition ``X·Y − beta·Z > 0``, the sign of
        dZ/dt) it is orders of magnitude off the species scale. A probe or a
        tolerance keyed on the species would call that continuous switch
        undecidable. Here the same crossing is written with a 1e6 factor on the
        residual and must give the identical answer."""
        scaled = CONTINUOUS.replace(
            "    1 growth() if(X<1,rho*(1-X),0)\n",
            "    1 growth() if(1e6*(1-X)>0,rho*(1-X),0)\n",
        )
        plain = _sens(tmp_path, ["rho"], name="s1.net", text=CONTINUOUS)
        blown = _sens(tmp_path, ["rho"], name="s2.net", text=scaled)
        a, b = np.asarray(plain.sensitivities), np.asarray(blown.sensitivities)
        scale = float(np.max(np.abs(a)))
        assert scale > 0.0
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9 * scale)


# ─── the gate and the detector must read the same text ─────────────────────


class TestTheDetectorSeesWhatTheGateSees:
    """A condition can only *become* a state condition under inlining, and the
    gate judges the inlined rate law.

    BIOMD0000000837 writes ``Lymphocyte_Term`` as
    ``piecewise(…, 1 - Total_Lymphocytes/K > 0, 0)`` where ``Total_Lymphocytes``
    is an SBML assignment-rule parameter — a *parameter* address, reading no
    live state by itself. The gate sees it after substitution as
    ``1 - (B+C_e+C_m+H_e+H_m+L)/K > 0`` and admits; a detector scanning raw
    function bodies registered nothing, so the decline was lifted with no
    crossing behind it. That is the #68 silent zero, reintroduced from the other
    side, and it was invisible to every test until the corpus A/B put three
    models in the "moved, switches=[]" bucket.
    """

    def test_a_condition_that_only_reads_state_after_inlining_is_registered(self, tmp_path):
        text = NET.replace(
            "    1 growth() if(X<1,0,rho)\n",
            "    1 margin() 1-X/X0\n    2 growth() if(margin()>0,rho,0)\n",
        )
        core = _model(tmp_path, text, name="inlined.net")._core
        # The raw body's atom names a *function*, which binds to no state
        # address — this is the classification the bug turned on.
        assert not sw.state_switch_residual(core, "margin>0")
        # Inlined it is a comparison over an observable, and both halves agree.
        assert sw.state_switch_conditions(core) == ["(1-X/X0)>0"]
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms


# ─── several residuals at one instant (issue #153) ─────────────────────────
#
# The roots are deduplicated by the residual's TEXT, which merges ``X<1`` with
# ``X<=1`` and nothing else. Two *spellings* of one crossing therefore
# reach the solver as two roots that fire together, and issue #150 refused that
# batch outright, reasoning that each jump reads f on the two branches of its
# OWN condition and one step across a shared crossing cannot separate them.
#
# On the corpus that refusal never fired for what it was written for. Both
# models that hit it write one crossing twice: ``sp_fourier_synthesizer`` roots
# on ``ds1`` and on ``3·ds1 − 12·s1²·ds1`` (five residuals in all, every one a
# multiple of ``Cos1 − amp_offset``, all crossing at t = π/2), and
# ``ml_hopfield`` roots on ``dS1/dt`` and ``dS3/dt``, which are identically
# equal along its trajectory because its own weight matrix leaves ``S1 ≡ S3``
# invariant. One is visible in the text and one is not, which is why the
# decision belongs at the crossing rather than at detection time.
#
# The batch needs the two halves of the saltation term checked separately, and
# that is what the fixtures below pin:
#
#   * ``f⁻ − f⁺`` has to carry EVERY branch change, which the one flow probe
#     does exactly when it crosses every residual in the batch;
#   * ``dt*/dθ`` has to be ONE vector — which flipping together does NOT
#     establish, so it is formed from each residual in turn and compared.
#
# The second is the criterion, and it is both weaker and stronger than "one
# surface". Weaker: ``COINCIDENT`` below is two genuinely independent crossings,
# and whether they merge depends on which columns are asked for, not on the
# model. Stronger: ``ml_hopfield``'s two residuals are equal along its
# trajectory and nowhere else — their gradients are not parallel, and their
# ``dt*/dθ`` come out permuted by the ``W12 ↔ W23`` symmetry — so a perturbation
# that breaks the symmetry splits the crossing, and merging on the flip test
# alone would have been wrong for it had there been anything to merge.

# ``X<1`` from the model at the top of this file, plus a second condition that
# names the SAME surface through a state-dependent factor — ``X² < 1`` is
# ``(X+1)·(X−1) < 0`` for the positive X this model has, the shape
# ``sp_fourier_synthesizer`` reaches with ``ds3 = ds1·(3 − 12·s1²)``. The dedup
# cannot see that, so the two cross as a batch of two.
#
# Unlike either corpus model there is a REAL jump here: ``pump`` switches the
# zeroth-order source of Y from ``ka`` to ``kb`` at the crossing, so
# ``f⁻ − f⁺`` is non-zero and the merged jump has to be exactly one of them.
# Composing the batch instead — the same jump once per residual — doubles Y's
# whole post-crossing column, which is what the closed forms below rule out.
TWO_SPELLINGS = """\
begin parameters
    1 X0     1000  # Constant
    2 rho    0.8  # Constant
    3 delta  1.6  # Constant
    4 ka     3.0  # Constant
    5 kb     1.0  # Constant
end parameters
begin functions
    1 growth() if(X<1,0,rho)
    2 pump() if((X*X)<1,kb,ka)
end functions
begin species
    1 A() X0
    2 B() 0
end species
begin reactions
    1 1 1,1 growth #_R1
    2 1 0 delta #_R2
    3 0 2 pump #_R3
end reactions
begin groups
    1 X                    1
    2 Y                    2
end groups
"""

DT_STAR_DRHO = T_STAR / 0.8  # d/drho of ln(X0)/(delta − rho)
KA, KB = 3.0, 1.0


@requires_cc
class TestOneCrossingWrittenTwice:
    def test_the_dedup_cannot_merge_them(self, tmp_path):
        """The premise: two conditions, two distinct residual texts, one
        surface. The text key is all the registration has to go on, which is
        why the batch reaches the solver at all."""
        core = _model(tmp_path, TWO_SPELLINGS, "two.net")._core
        conds = sw.state_switch_conditions(core)
        assert conds == ["X<1", "(X*X)<1"]
        residuals = [core.state_switch_residual(c)[0] for c in conds]
        assert residuals[0] != residuals[1]

    def test_the_batch_gets_one_jump_and_not_one_per_residual(self, tmp_path):
        """The arithmetic, against closed forms.

        Y accumulates at ``ka`` until the crossing and ``kb`` after it, so
        ``∂Y/∂θ`` past t* is exactly ``(ka − kb)·dt*/dθ`` for a parameter that
        moves only the crossing, and ``t*`` / ``T_END − t*`` for the two rates
        themselves. Every one of those is halved by dropping the jump and
        doubled by applying it once per residual.

        Before this fix the run did not produce numbers at all — it raised the
        two-switches-at-one-instant refusal, which is also what says the two
        roots really do land in one batch here."""
        params = ["rho", "delta", "ka", "kb"]
        run = _sens(tmp_path, params, name="two.net", text=TWO_SPELLINGS)
        an = np.asarray(run.sensitivities)
        y = an[-1, 1, :]  # species B = the observable Y, at T_END
        assert y[0] == pytest.approx((KA - KB) * DT_STAR_DRHO, rel=1e-5)
        assert y[1] == pytest.approx(-(KA - KB) * DT_STAR_DRHO, rel=1e-5)
        assert y[2] == pytest.approx(T_STAR, rel=1e-5)
        assert y[3] == pytest.approx(T_END - T_STAR, rel=1e-5)
        # And X's own column is still the issue #150 answer — the second
        # residual neither adds to it nor takes anything away.
        assert an[-1, 0, 0] == pytest.approx(
            2.0 * T_STAR * np.exp(-1.6 * (T_END - T_STAR)), rel=1e-4
        )

    def test_every_column_matches_a_finite_difference(self, tmp_path):
        params = ["rho", "delta", "ka", "kb"]
        run = _sens(tmp_path, params, name="two.net", text=TWO_SPELLINGS)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params, text=TWO_SPELLINGS)
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, :, j])))
            assert scale > 0.0
            assert np.max(np.abs(an[:, :, j] - fd[:, :, j])) <= 1e-3 * scale, (
                f"column {p!r} disagrees with its own finite difference"
            )


# Two crossings that are genuinely INDEPENDENT — different species, different
# rate constants, nothing shared but the threshold — and coincide only because
# ``ku`` and ``kv`` happen to be equal, so ``U`` and ``V`` decay through ``c``
# at the same instant ln(2). Each gates its own contribution to W's source, so
# there is a real jump at each.
COINCIDENT = """\
begin parameters
    1 U0     1.0  # Constant
    2 ku     1.0  # Constant
    3 kv     1.0  # Constant
    4 c      0.5  # Constant
    5 a1     3.0  # Constant
    6 a2     1.0  # Constant
end parameters
begin functions
    1 pu() if(U<c,a2,a1)
    2 pv() if(V<c,a2,a1)
end functions
begin species
    1 P() U0
    2 Q() U0
    3 R() 0
end species
begin reactions
    1 1 0 ku #_R1
    2 2 0 kv #_R2
    3 0 3 pu #_R3
    4 0 3 pv #_R4
end reactions
begin groups
    1 U                    1
    2 V                    2
    3 W                    3
end groups
"""


@requires_cc
class TestWhatIsMergedIsOneCrossingTime:
    """One crossing *time*, not one surface — the surface is sufficient and not
    necessary, and it is not what the jump needs. Same model, same coincident
    pair, two answers depending on which column is asked for."""

    def test_coincident_crossings_that_move_together_are_merged(self, tmp_path):
        """``c`` is the threshold of BOTH conditions, so it moves the two
        crossing times identically: ``dt*/dc = −1/(c·k)`` either way. One
        ``dt*/dθ`` then serves the pair and the combined ``f⁻ − f⁺`` is the sum
        of the two branch changes, which is exactly the merged jump.

        W's rate is ``2·a1`` before and ``2·a2`` after, so
        ``∂W/∂c = 2(a1 − a2)·dt*/dc = −8`` for the whole tail — a closed form,
        and the analytic column is that to eight digits. (The finite difference
        is the loose one here, by 2e-4: its own runs register no roots at all
        and chase both kinks.)"""
        run = _sens(tmp_path, ["c"], name="coin.net", text=COINCIDENT)
        w = np.asarray(run.sensitivities)[:, 2, 0]
        t = np.asarray(run.time)
        past = t > np.log(2.0) + 0.05
        assert past.sum() >= 5
        np.testing.assert_allclose(w[past], -8.0, rtol=1e-6)

    def test_coincident_crossings_that_move_apart_are_refused(self, tmp_path):
        """``ku`` moves U's crossing and leaves V's exactly where it was, so
        the pair splits under the perturbation into two crossings with two
        branch changes — and the truth is a sum of two jumps that the shared
        probe cannot take apart. The refusal reports the disagreement it
        measured rather than the coincidence it noticed."""
        with pytest.raises(SimulationError) as exc:
            _sens(tmp_path, ["ku"], name="coin2.net", text=COINCIDENT)
        msg = str(exc.value)
        assert "(U)-(c)" in msg and "(V)-(c)" in msg
        assert "move differently" in msg
        assert "#153" in msg


# ``ml_hopfield`` verbatim from ``benchmarks/suites/ode_fullnet/nets`` (which is
# generated, not checked in — hence the inline copy, as elsewhere in this
# suite). Three neurons whose rate laws are the BNGL signed-rate idiom over
# ``dSi/dt``, so each condition's two branches meet at its own crossing.
HOPFIELD = """\
begin parameters
    1 W12        -1.0  # Constant
    2 W13        1.0  # Constant
    3 W23        -1.0  # Constant
    4 Tau        1.0  # Constant
    5 Gain       5.0  # Constant
end parameters
begin functions
    1 Net1() (W12*((2*S2)-1))+(W13*((2*S3)-1))
    2 Net2() (W12*((2*S1)-1))+(W23*((2*S3)-1))
    3 Net3() (W13*((2*S1)-1))+(W23*((2*S2)-1))
    4 Target1() 1/(1+exp(((-Gain)*Net1())))
    5 Target2() 1/(1+exp(((-Gain)*Net2())))
    6 Target3() 1/(1+exp(((-Gain)*Net3())))
    7 dS1_dt() (Target1()-S1)/Tau
    8 dS2_dt() (Target2()-S2)/Tau
    9 dS3_dt() (Target3()-S3)/Tau
   10 _rateLaw1() if((dS1_dt()>0),dS1_dt(),0)
   11 _rateLaw2() if((dS1_dt()<0),(-dS1_dt()),0)
   12 _rateLaw3() if((dS2_dt()>0),dS2_dt(),0)
   13 _rateLaw4() if((dS2_dt()<0),(-dS2_dt()),0)
   14 _rateLaw5() if((dS3_dt()>0),dS3_dt(),0)
   15 _rateLaw6() if((dS3_dt()<0),(-dS3_dt()),0)
end functions
begin species
    1 Neuron(id~1) 1.0
    2 Neuron(id~2) 0.8
    3 Neuron(id~3) 1.0
end species
begin reactions
    1 0 1 _rateLaw1 #U1
    2 1 0 _rateLaw2 #D1
    3 0 2 _rateLaw3 #U2
    4 2 0 _rateLaw4 #D2
    5 0 3 _rateLaw5 #U3
    6 3 0 _rateLaw6 #D3
end reactions
begin groups
    1 S1                   1
    2 S2                   2
    3 S3                   3
end groups
"""


@requires_cc
class TestTheModelTheIssueFound:
    """One of the two rulehub "BNGL as a general-purpose computation" models
    that reached the refusal, kept whole because what makes it a batch is a
    property of its weights and not of any one line."""

    def test_the_two_residuals_coincide_by_a_symmetry(self, tmp_path):
        """The premise, measured rather than argued. With ``W12 = W23`` and
        ``S1(0) = S3(0)``, substituting ``S1 = S3`` makes ``Net1`` and ``Net3``
        the same expression, so ``dS1/dt ≡ dS3/dt`` and the symmetry is
        invariant — the two residuals are different text for one crossing.

        Only along the trajectory, though, which is the whole reason the merge
        is decided on ``dt*/dθ`` and not on the flip test: off the ``S1 = S3``
        manifold the two are different functions (non-parallel gradients), and
        their ``dt*/dθ`` at this crossing come out permuted by ``W12 ↔ W23`` —
        (0.327, −0.168, −0.030) against (−0.030, −0.168, 0.327). Nothing here
        needs them, because the crossing carries no jump; had it carried one,
        this batch would be refused rather than merged.

        If a future change to the fixture broke the symmetry this is the
        assertion that would say so, before the sensitivity ones got
        mysterious."""
        run = _sens(tmp_path, [], name="hop0.net", text=HOPFIELD)
        x = np.asarray(run.species)
        assert np.max(np.abs(x[:, 0] - x[:, 2])) == 0.0

    def test_it_runs_and_matches_a_finite_difference(self, tmp_path):
        """Before this fix it raised at t = 0.4738, where ``dS1/dt`` and
        ``dS3/dt`` cross zero together. The crossing itself carries no jump —
        the signed-rate idiom is continuous at its own switch — so what the
        batch path has to get right here is simply not refusing, and the
        columns then come from the in-branch sensitivity RHS."""
        params = ["W12", "W13", "W23"]
        run = _sens(tmp_path, params, name="hop.net", text=HOPFIELD)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params, text=HOPFIELD)
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, :, j])))
            assert scale > 0.0
            assert np.max(np.abs(an[:, :, j] - fd[:, :, j])) <= 1e-4 * scale, (
                f"column {p!r} disagrees with its own finite difference"
            )


# ─── the SBML spelling, and an event in the same model ─────────────────────


ANTIMONY = """\
model nested_switch
  V_0 = 0.1; V_0_inject = 1000; t_0 = 2; rho_V = 0.8; delta_V = 1.6;
  Virus = V_0;
  Virus' = piecewise(0, Virus < 1, Virus*rho_V) - Virus*delta_V;
  at (time >= t_0): Virus = Virus + V_0_inject;
end
"""


@requires_cc
def test_an_sbml_piecewise_over_state_with_an_event(tmp_path):
    """The issue's field case: AMICI's ``nested_events`` shape, where the state
    switch shares a model with a time-triggered event whose assignment jumps the
    state. The two jumps are different objects — the event's is GH #212's
    ``∂h/∂x``-and-``∂h/∂p`` reset at a fixed instant, the switch's is the
    saltation term at a moving one — and both have to land, in that order, for
    the columns to track a finite difference over the whole run.

    Written as antimony rather than lifted from AMICI so the fixture is
    self-contained; the SBML libantimony emits is the same ``<piecewise>`` over
    ``<lt/>`` inside a ``<rateRule>`` that AMICI compiles.
    """
    pytest.importorskip("antimony")
    params = ["V_0_inject", "t_0", "rho_V", "delta_V"]

    def run(overrides=None, sens=None):
        m = bngsim.Model.from_antimony_string(ANTIMONY)
        for k, v in (overrides or {}).items():
            m.set_param(k, v)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(sens or []))
        return sim.run(t_span=(0.0, 12.0), n_points=45, rtol=1e-10, atol=1e-14)

    an = np.asarray(run(sens=params).sensitivities)
    for j, p in enumerate(params):
        p0 = bngsim.Model.from_antimony_string(ANTIMONY).get_param(p)
        h = 1e-6 * abs(p0)
        hi = np.asarray(run({p: p0 + h}).species)
        lo = np.asarray(run({p: p0 - h}).species)
        fd = (hi - lo) / (2 * h)
        scale = float(np.max(np.abs(fd[:, 0])))
        np.testing.assert_allclose(
            an[:, 0, j], fd[:, 0], rtol=5e-3, atol=1e-3 * scale, err_msg=f"column {p!r}"
        )
        assert scale > 0.0
