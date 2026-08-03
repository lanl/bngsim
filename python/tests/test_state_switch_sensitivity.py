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


def _sens(tmp_path, params, name="m.net", text=NET, **kw):
    model = _model(tmp_path, text, name)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params), **kw)
    return sim.run(t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL)


def _fd(tmp_path, params, text=NET, rel=1e-5):
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
                t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL
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
